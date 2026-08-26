"""One-step strong Q16 / separated-LEV / free-wake FSI transaction."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any

import numpy as np
import torch
import warp as wp

from bing_joint_ptera_gpu import CudaJointLEVTEVSolver
from fluxvortex.q16_ancf_shell import q16_shape
from fluxvortex.warp_fsi import config
from fluxvortex.warp_fsi.q16_aero_load_packet import Q16CudaAerodynamicLoadPacket
from fluxvortex.warp_fsi.q16_lev_impulse_transfer import (
    Q16CudaLEVImpulseStripLoad,
)
from fluxvortex.warp_fsi.q16_mandatory_aero_mode import (
    require_q16_mandatory_aero_mode,
)
from fluxvortex.warp_fsi.q16_ptera_resolved_transfer import (
    Q16CompleteAeroLoadResult,
    Q16CudaCompleteAeroLoadTransfer,
    Q16PteraResolvedTransferResult,
)
from fluxvortex.warp_fsi.q16_prescribed_endpoint_load import (
    Q16CudaPrescribedEndpointLoad,
)
from fluxvortex.warp_fsi.q16_structural_solver import (
    Q16CudaNewmarkStepper,
    Q16StructuralStepResult,
    Q16StructuralWorkBalance,
)
from yamano2020_author_aero_projection import (
    build_yamano2020_author_added_mass_projection,
)
from q16_incremental_ptera_owner import Q16CudaIncrementalAeroSession
from q16_ptera_trial_kinematics import Q16CudaPteraIncrementalGeometry
from q16_real_aero_branch_transaction import (
    Q16CudaAeroSolverOwner,
)

_RESULT_SCHEMA = "flux-v5m-q16-real-strong-fsi-step-v6"
_AERODYNAMIC_SUBSTEP_SCHEMES = {
    "endpoint_hold",
    "block_linear",
    "author_corrector",
}


def _positive_float(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise TypeError(f"{name} must be a real scalar")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _positive_int(name: str, value: int) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive exact int")
    return value


def _warp_sha256(name: str, value: Any) -> str:
    if not isinstance(value, wp.array):
        raise TypeError(f"{name} must be a Warp array")
    if not value.device.is_cuda:
        raise ValueError(f"{name} must reside on CUDA")
    if value.dtype != config.DTYPE:
        raise TypeError(f"{name} must use Warp float64")
    if value.ndim != 2 or value.shape[0] != 1 or value.shape[1] <= 0:
        raise ValueError(f"{name} must have shape (1, positive_dof_count)")
    host = np.ascontiguousarray(value.numpy(), dtype=np.float64)
    if not bool(np.isfinite(host).all()):
        raise FloatingPointError(f"{name} contains non-finite values")
    header = json.dumps(
        {"dtype": "float64", "shape": list(host.shape)},
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(header + b"\0" + host.tobytes(order="C")).hexdigest()


def _state_sha256(state: wp.array, velocity: wp.array, acceleration: wp.array) -> str:
    payload = json.dumps(
        {
            "domain": "flux-v5m-q16-structural-owner-v1",
            "state_sha256": _warp_sha256("state", state),
            "velocity_sha256": _warp_sha256("velocity", velocity),
            "acceleration_sha256": _warp_sha256("acceleration", acceleration),
        },
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _clone_state(value: wp.array) -> wp.array:
    return wp.clone(value)


def _total_external_force(
    aerodynamic_force: wp.array,
    prescribed_load: Q16CudaPrescribedEndpointLoad | None,
) -> wp.array:
    if prescribed_load is None:
        return aerodynamic_force
    prescribed_load.validate()
    prescribed = prescribed_load.generalized_force
    if aerodynamic_force.shape != prescribed.shape:
        raise ValueError("aerodynamic and prescribed Q16 load shapes differ")
    if aerodynamic_force.device.alias != prescribed.device.alias:
        raise ValueError("aerodynamic and prescribed Q16 loads use different devices")
    if aerodynamic_force.dtype != config.DTYPE:
        raise TypeError("aerodynamic Q16 generalized force must use Warp float64")
    aerodynamic = wp.to_torch(aerodynamic_force)
    independent = wp.to_torch(prescribed)
    if any(
        value.device.type != "cuda" or value.dtype is not torch.float64
        for value in (aerodynamic, independent)
    ):
        raise RuntimeError("Q16 external-load composition left CUDA float64")
    return wp.clone(
        wp.from_torch(
            aerodynamic + independent,
            dtype=config.DTYPE,
            requires_grad=False,
        )
    )


def _combine_cuda_forces(
    first: wp.array,
    second: wp.array,
    *,
    first_factor: float = 1.0,
    second_factor: float = 1.0,
) -> wp.array:
    if not isinstance(first, wp.array) or not isinstance(second, wp.array):
        raise TypeError("force combination inputs must be Warp arrays")
    if first.shape != second.shape or first.device.alias != second.device.alias:
        raise ValueError("force combination inputs differ")
    if (
        not first.device.is_cuda
        or first.dtype != config.DTYPE
        or second.dtype != config.DTYPE
    ):
        raise ValueError("force combination requires CUDA float64")
    values = float(first_factor) * wp.to_torch(first) + float(
        second_factor
    ) * wp.to_torch(second)
    if values.device.type != "cuda" or values.dtype is not torch.float64:
        raise RuntimeError("force combination left CUDA float64")
    return wp.clone(wp.from_torch(values, dtype=config.DTYPE, requires_grad=False))


def _ptera_unsteady_generalized_force(
    packet: Q16CudaAerodynamicLoadPacket,
    resolved_transfer: Any,
    structural_state: wp.array,
) -> wp.array:
    """Map only Ptera's fifth (outer-step ``dGamma``) force block.

    The block-linear Q16 path owns acceleration through the explicit Yamano
    ``Mf1`` matrix.  Keeping Ptera's finite-difference ``dGamma/dt`` panel
    force at the same time would apply that non-circulatory response twice.
    KJ forces from the first four blocks remain untouched, including all
    separated-LEV and free-wake influence on the bound solve.
    """

    packet.validate()
    if packet.point_count % 5 != 0:
        raise RuntimeError("Ptera five-block resolved topology drift")
    panel_count = packet.point_count // 5
    forces = torch.zeros_like(packet.point_forces_w)
    forces[-panel_count:] = packet.point_forces_w[-panel_count:]
    total_force = torch.sum(forces, dim=0)
    total_moment = torch.sum(
        torch.linalg.cross(packet.point_positions_w, forces, dim=1), dim=0
    )
    unsteady_packet = Q16CudaAerodynamicLoadPacket.from_tensors(
        point_positions_w=packet.point_positions_w,
        point_forces_w=forces,
        unresolved_impulse_force_w=torch.zeros_like(packet.unresolved_impulse_force_w),
        source_total_force_w=total_force,
        source_total_moment_w=total_moment,
    )
    return resolved_transfer.map(unsteady_packet, structural_state).generalized_force


def _average_cuda_state(first: wp.array, second: wp.array) -> wp.array:
    return _combine_cuda_forces(
        first,
        second,
        first_factor=0.5,
        second_factor=0.5,
    )


def _interpolate_aerodynamic_force(
    start: wp.array,
    end: wp.array,
    fraction: float,
) -> wp.array:
    """Interpolate one block's generalized aero load on CUDA float64.

    This is the structural-clock part of Yamano's two-pass coupling: the
    expensive bound/wake solve owns only the block endpoints, while each Q16
    structural substep consumes the time-interpolated endpoint operator.  It
    deliberately does not advance LEV, TEV or the material wake.
    """

    if not isinstance(start, wp.array) or not isinstance(end, wp.array):
        raise TypeError("aerodynamic interpolation inputs must be Warp arrays")
    if start.shape != end.shape or start.device.alias != end.device.alias:
        raise ValueError("aerodynamic interpolation endpoints differ")
    if (
        not start.device.is_cuda
        or start.dtype != config.DTYPE
        or end.dtype != config.DTYPE
    ):
        raise ValueError("aerodynamic interpolation requires CUDA float64")
    if isinstance(fraction, bool) or not isinstance(
        fraction, (int, float, np.integer, np.floating)
    ):
        raise TypeError("aerodynamic interpolation fraction must be real")
    beta = float(fraction)
    if not math.isfinite(beta) or beta < 0.0 or beta > 1.0:
        raise ValueError("aerodynamic interpolation fraction must be in [0,1]")
    start_t = wp.to_torch(start)
    end_t = wp.to_torch(end)
    interpolated = start_t + beta * (end_t - start_t)
    if interpolated.device.type != "cuda" or interpolated.dtype is not torch.float64:
        raise RuntimeError("aerodynamic interpolation left CUDA float64")
    return wp.clone(
        wp.from_torch(interpolated, dtype=config.DTYPE, requires_grad=False)
    )


@dataclass(frozen=True, slots=True, eq=False)
class _Q16CudaFrozenKJVelocityLoad:
    """Frozen endpoint KJ operator applied to live structural velocity.

    Ptera's four resolved leg-force blocks contain
    ``rho*Gamma_eff*(V_solution - V_surface) x dl``.  The circulation,
    geometry, wake and conservative point stencil belong to one issued outer
    trial; only ``V_surface`` is re-evaluated at the structural clock.  This is
    the Q16 analogue of Yamano's interpolated ``Qf_p_lift2_mat`` term.
    """

    transfer: Any
    resolved: Q16PteraResolvedTransferResult
    effective_strengths: torch.Tensor
    leg_vectors_w: torch.Tensor
    fluid_density: float

    @classmethod
    def from_solver(
        cls,
        solver: CudaJointLEVTEVSolver,
        complete: Q16CompleteAeroLoadResult,
        transfer: Any,
    ) -> _Q16CudaFrozenKJVelocityLoad:
        require_q16_mandatory_aero_mode(solver)
        complete.validate()
        resolved = complete.resolved
        panel_count = resolved.point_count // 5
        if resolved.point_count != 5 * panel_count or panel_count <= 0:
            raise ValueError("Ptera velocity load requires five point blocks")
        gamma = solver._cuda_bound_strengths
        strengths = torch.cat(solver._effective_strengths(gamma), dim=0)
        if tuple(strengths.shape) != (4 * panel_count,):
            raise RuntimeError("Ptera effective-strength topology drift")
        vectors_gp = torch.as_tensor(
            np.concatenate(
                (
                    solver.stackRbrv_GP1,
                    solver.stackFbrv_GP1,
                    solver.stackLbrv_GP1,
                    solver.stackBbrv_GP1,
                ),
                axis=0,
            ),
            device=solver.cuda_device,
            dtype=torch.float64,
        )
        vectors_w = solver._v5m_gp_vectors_to_scientific_cuda(vectors_gp)
        if not bool(
            torch.isfinite(strengths).all().item()
            and torch.isfinite(vectors_w).all().item()
        ):
            raise FloatingPointError("frozen KJ velocity operator is non-finite")
        return cls(
            transfer=transfer,
            resolved=resolved,
            effective_strengths=strengths.detach().clone(),
            leg_vectors_w=vectors_w.detach().clone(),
            fluid_density=float(solver.current_operating_point.rho),
        )

    def map(self, structural_velocity: wp.array) -> wp.array:
        point_velocity = self.transfer._interpolate_frozen_point_direction_prechecked(
            self.resolved, structural_velocity
        )
        point_forces = torch.zeros(
            (self.resolved.point_count, 3),
            device=point_velocity.device,
            dtype=torch.float64,
        )
        leg_count = int(self.effective_strengths.shape[0])
        # Ptera's movement velocity is -V_surface.
        point_forces[:leg_count] = -(
            self.fluid_density
            * self.effective_strengths.unsqueeze(1)
            * torch.linalg.cross(point_velocity[:leg_count], self.leg_vectors_w, dim=1)
        )
        if not bool(torch.isfinite(point_forces).all().item()):
            raise FloatingPointError("KJ structural-velocity force is non-finite")
        return self.transfer._transpose_frozen_point_forces_prechecked(
            self.resolved, point_forces
        )


@dataclass(frozen=True, slots=True, eq=False)
class _Q16CudaFrozenLift2VelocityLoad:
    """Yamano ``Qf_p_lift2`` from the frozen bound-circulation gradient."""

    transfer: Any
    resolved: Q16PteraResolvedTransferResult
    dp_lift2_w: torch.Tensor
    panel_normals_w: torch.Tensor
    panel_areas: torch.Tensor
    pressure_to_generalized: torch.Tensor

    @classmethod
    def from_solver(
        cls,
        solver: CudaJointLEVTEVSolver,
        complete: Q16CompleteAeroLoadResult,
        transfer: Any,
        vertex_transfer: Any,
        structural_state: wp.array,
    ) -> _Q16CudaFrozenLift2VelocityLoad:
        require_q16_mandatory_aero_mode(solver)
        complete.validate()
        resolved = complete.resolved
        _, span_count, chord_count = solver._panel_grid()
        panel_count = chord_count * span_count
        vertices = wp.to_torch(vertex_transfer.interpolate(structural_state))[0]
        grid = vertices.reshape(chord_count + 1, span_count + 1, 3)
        chord_vector = 0.5 * (
            (grid[1:, :-1] - grid[:-1, :-1]) + (grid[1:, 1:] - grid[:-1, 1:])
        )
        span_vector = 0.5 * (
            (grid[:-1, 1:] - grid[:-1, :-1]) + (grid[1:, 1:] - grid[1:, :-1])
        )
        chord_length = torch.linalg.vector_norm(chord_vector, dim=2)
        span_length = torch.linalg.vector_norm(span_vector, dim=2)
        chord_tangent = chord_vector / chord_length.unsqueeze(2)
        span_tangent = span_vector / span_length.unsqueeze(2)
        # Ptera's bound rings use the orientation opposite to the Yamano
        # author convention.  Pressure gradients own author circulation.
        gamma = -solver._cuda_bound_strengths.reshape(chord_count, span_count)
        dgamma_chord = torch.empty_like(gamma)
        dgamma_chord[0] = gamma[0] / chord_length[0]
        if chord_count > 1:
            dgamma_chord[1:] = (gamma[1:] - gamma[:-1]) / chord_length[1:]
        dgamma_span = torch.zeros_like(gamma)
        if span_count > 1:
            dgamma_span[:, 0] = gamma[:, 0] / span_length[:, 0]
            dgamma_span[:, -1] = -gamma[:, -1] / span_length[:, -1]
            if span_count > 2:
                dgamma_span[:, 1:-1] = (gamma[:, 2:] - gamma[:, :-2]) / (
                    2.0 * span_length[:, 1:-1]
                )
        dp_lift2 = float(solver.current_operating_point.rho) * (
            chord_tangent * dgamma_chord.unsqueeze(2)
            + span_tangent * dgamma_span.unsqueeze(2)
        )
        normals_gp = torch.as_tensor(
            np.array(solver.stackUnitNormals_GP1, dtype=np.float64, copy=True),
            device=solver.cuda_device,
            dtype=torch.float64,
        )
        normals_w = solver._v5m_gp_vectors_to_scientific_cuda(normals_gp)
        areas = torch.as_tensor(
            np.array(solver.panel_areas, dtype=np.float64, copy=True),
            device=solver.cuda_device,
            dtype=torch.float64,
        )
        if resolved.point_count != 5 * panel_count or not bool(
            torch.isfinite(dp_lift2).all().item()
            and torch.isfinite(normals_w).all().item()
            and torch.isfinite(areas).all().item()
        ):
            raise FloatingPointError("frozen lift2 operator is invalid")
        pressure_to_generalized = _q16_distributed_pressure_map(solver, transfer)
        return cls(
            transfer=transfer,
            resolved=resolved,
            dp_lift2_w=dp_lift2.reshape(panel_count, 3).detach().clone(),
            panel_normals_w=normals_w.detach().clone(),
            panel_areas=areas.detach().clone(),
            pressure_to_generalized=pressure_to_generalized,
        )

    def map(self, structural_velocity: wp.array) -> wp.array:
        point_velocity = self.transfer._interpolate_frozen_point_direction_prechecked(
            self.resolved, structural_velocity
        )
        panel_count = int(self.panel_areas.shape[0])
        collocation_velocity = point_velocity[4 * panel_count :]
        pressure = -torch.sum(collocation_velocity * self.dp_lift2_w, dim=1)
        generalized = torch.matmul(
            pressure, self.pressure_to_generalized.transpose(0, 1)
        ).unsqueeze(0)
        if not bool(torch.isfinite(generalized).all().item()):
            raise FloatingPointError("lift2 force is non-finite")
        return wp.from_torch(generalized, dtype=config.DTYPE, requires_grad=False)


def _p_interp_uniform_weights(
    local_chord: float,
    chord_panel: int,
    chordwise_panel_count: int,
) -> tuple[float, float, float]:
    """Author ``p_interp.m`` weights on a uniform chordwise panel grid."""

    if not 0.0 <= local_chord <= 1.0:
        raise ValueError("local panel chord must lie in [0,1]")
    if local_chord <= 0.75:
        previous = (3.0 - 4.0 * local_chord) / 4.0
        current = (1.0 + 4.0 * local_chord) / 4.0
        following = 0.0
        if chord_panel == 0:
            previous = 0.0
            current = 1.0
    else:
        previous = 0.0
        if chord_panel < chordwise_panel_count - 1:
            current = (7.0 - 4.0 * local_chord) / 4.0
            following = (4.0 * local_chord - 3.0) / 4.0
        else:
            current = 4.0 - 4.0 * local_chord
            following = 0.0
    return previous, current, following


def _strict_partition(
    lower: float,
    upper: float,
    candidates: list[float],
) -> tuple[float, ...]:
    tolerance = 256.0 * np.finfo(np.float64).eps
    values = [lower, upper]
    values.extend(
        value for value in candidates if lower + tolerance < value < upper - tolerance
    )
    result = tuple(sorted(set(values)))
    if len(result) < 2 or any(
        result[index + 1] <= result[index] for index in range(len(result) - 1)
    ):
        raise RuntimeError("Q16 pressure integration partition is invalid")
    return result


def _q16_added_mass_projection_setup(
    vertex_map: Any,
    *,
    chordwise_panel_count: int,
    spanwise_panel_count: int,
) -> tuple[int, int, np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Build immutable Q16 collocation and distributed-pressure setup maps.

    Only topology, polynomial shapes and quadrature weights are formed on the
    host.  AIC solves, normal contractions, matrix products and every dynamic
    state action remain CUDA float64.  Integrating each aero-panel and 3/4-
    chord interpolation segment separately makes the piecewise polynomial
    projection exact to quadrature precision even when one Q16 macro element
    spans several aerodynamic panels.
    """

    mesh = vertex_map.mesh
    reference = np.asarray(mesh.reference_rows[:, :3], dtype=np.float64)
    connectivity = np.asarray(mesh.connectivity, dtype=np.int64)
    centroids = np.mean(reference[connectivity], axis=1)
    chord_levels = np.unique(np.round(centroids[:, 0], decimals=13))
    span_levels = np.unique(np.round(centroids[:, 1], decimals=13))
    q16_chord_count = int(chord_levels.size)
    q16_span_count = int(span_levels.size)
    if (
        q16_chord_count <= 0
        or q16_span_count <= 0
        or q16_chord_count * q16_span_count != mesh.element_count
    ):
        raise RuntimeError("Q16 pressure projection requires a rectangular grid")
    chord_length = float(np.ptp(reference[:, 0]))
    span_length = float(np.ptp(reference[:, 1]))
    if not math.isfinite(chord_length) or not math.isfinite(span_length):
        raise FloatingPointError("Q16 pressure projection extent is non-finite")
    if chord_length <= 0.0 or span_length <= 0.0:
        raise ValueError("Q16 pressure projection extent is degenerate")

    chord_count = chordwise_panel_count
    span_count = spanwise_panel_count
    panel_count = chord_count * span_count
    collocation_elements = np.empty(panel_count, dtype=np.int64)
    collocation_shapes = np.empty((panel_count, 16), dtype=np.float64)
    for chord_panel in range(chord_count):
        for span_panel in range(span_count):
            panel = chord_panel * span_count + span_panel
            chord_fraction = (chord_panel + 0.75) / chord_count
            span_fraction = (span_panel + 0.5) / span_count
            chord_element = min(
                int(math.floor(chord_fraction * q16_chord_count)),
                q16_chord_count - 1,
            )
            span_element = min(
                int(math.floor(span_fraction * q16_span_count)),
                q16_span_count - 1,
            )
            element = span_element * q16_chord_count + chord_element
            xi = 2.0 * (chord_fraction * q16_chord_count - chord_element) - 1.0
            eta = 2.0 * (span_fraction * q16_span_count - span_element) - 1.0
            collocation_elements[panel] = element
            collocation_shapes[panel] = q16_shape(xi, eta)[0]

    pressure_shapes = np.zeros((mesh.element_count, 16, panel_count), dtype=np.float64)
    gauss_points, gauss_weights = np.polynomial.legendre.leggauss(3)
    chord_candidates = [value / chord_count for value in range(1, chord_count)] + [
        (value + 0.75) / chord_count for value in range(chord_count)
    ]
    span_candidates = [value / span_count for value in range(1, span_count)]
    for span_element in range(q16_span_count):
        span_lower = span_element / q16_span_count
        span_upper = (span_element + 1) / q16_span_count
        span_partition = _strict_partition(span_lower, span_upper, span_candidates)
        for chord_element in range(q16_chord_count):
            chord_lower = chord_element / q16_chord_count
            chord_upper = (chord_element + 1) / q16_chord_count
            chord_partition = _strict_partition(
                chord_lower, chord_upper, chord_candidates
            )
            element = span_element * q16_chord_count + chord_element
            for chord_segment in range(len(chord_partition) - 1):
                chord_a = chord_partition[chord_segment]
                chord_b = chord_partition[chord_segment + 1]
                chord_midpoint = 0.5 * (chord_a + chord_b)
                chord_panel = min(
                    int(math.floor(chord_midpoint * chord_count)),
                    chord_count - 1,
                )
                for span_segment in range(len(span_partition) - 1):
                    span_a = span_partition[span_segment]
                    span_b = span_partition[span_segment + 1]
                    span_midpoint = 0.5 * (span_a + span_b)
                    span_panel = min(
                        int(math.floor(span_midpoint * span_count)),
                        span_count - 1,
                    )
                    for chord_gauss, chord_weight in zip(
                        gauss_points, gauss_weights, strict=True
                    ):
                        chord_fraction = 0.5 * (
                            chord_a + chord_b + chord_gauss * (chord_b - chord_a)
                        )
                        local_chord = chord_fraction * chord_count - chord_panel
                        pressure_weights = _p_interp_uniform_weights(
                            local_chord, chord_panel, chord_count
                        )
                        xi = (
                            2.0 * (chord_fraction * q16_chord_count - chord_element)
                            - 1.0
                        )
                        for span_gauss, span_weight in zip(
                            gauss_points, gauss_weights, strict=True
                        ):
                            span_fraction = 0.5 * (
                                span_a + span_b + span_gauss * (span_b - span_a)
                            )
                            eta = (
                                2.0 * (span_fraction * q16_span_count - span_element)
                                - 1.0
                            )
                            shape = q16_shape(xi, eta)[0]
                            area_weight = (
                                chord_weight
                                * span_weight
                                * 0.25
                                * (chord_b - chord_a)
                                * chord_length
                                * (span_b - span_a)
                                * span_length
                            )
                            for offset, coefficient in zip(
                                (-1, 0, 1), pressure_weights, strict=True
                            ):
                                neighbor = chord_panel + offset
                                if coefficient == 0.0 or not (
                                    0 <= neighbor < chord_count
                                ):
                                    continue
                                panel = neighbor * span_count + span_panel
                                pressure_shapes[element, :, panel] += (
                                    area_weight * coefficient * shape
                                )
    if not bool(
        np.isfinite(collocation_shapes).all() and np.isfinite(pressure_shapes).all()
    ):
        raise FloatingPointError("Q16 added-mass projection map is non-finite")
    return (
        q16_chord_count,
        q16_span_count,
        np.ascontiguousarray(collocation_elements),
        np.ascontiguousarray(collocation_shapes),
        np.ascontiguousarray(pressure_shapes),
        chord_length,
        span_length,
    )


def _q16_distributed_pressure_map(
    solver: CudaJointLEVTEVSolver,
    transfer: Any,
) -> torch.Tensor:
    """Map author-interpolated panel pressure to Q16 generalized force."""

    _, span_count, chord_count = solver._panel_grid()
    panel_count = chord_count * span_count
    vertex_map = transfer.vertex_map
    mesh = vertex_map.mesh
    (
        q16_chord_count,
        q16_span_count,
        _,
        _,
        pressure_shapes,
        chord_length,
        span_length,
    ) = _q16_added_mass_projection_setup(
        vertex_map,
        chordwise_panel_count=chord_count,
        spanwise_panel_count=span_count,
    )
    connectivity = torch.as_tensor(
        np.array(mesh.connectivity, dtype=np.int64, copy=True),
        device=solver.cuda_device,
        dtype=torch.int64,
    )
    normals_gp = torch.as_tensor(
        np.array(solver.stackUnitNormals_GP1, dtype=np.float64, copy=True),
        device=solver.cuda_device,
        dtype=torch.float64,
    )
    normals_w = solver._v5m_gp_vectors_to_scientific_cuda(normals_gp)
    areas = torch.as_tensor(
        np.array(solver.panel_areas, dtype=np.float64, copy=True),
        device=solver.cuda_device,
        dtype=torch.float64,
    )
    reference_panel_area = chord_length * span_length / float(chord_count * span_count)
    area_ratio = areas / reference_panel_area
    pressure_shape_t = torch.as_tensor(
        pressure_shapes,
        device=solver.cuda_device,
        dtype=torch.float64,
    )
    result = torch.zeros(
        (mesh.dof_count, panel_count),
        device=solver.cuda_device,
        dtype=torch.float64,
    )
    components = torch.arange(6, device=solver.cuda_device)
    if mesh.element_count != q16_chord_count * q16_span_count:
        raise RuntimeError("Q16 distributed-pressure element grid drift")
    for element in range(mesh.element_count):
        element_dofs = (
            connectivity[element].unsqueeze(1) * 6 + components.unsqueeze(0)
        ).reshape(-1)
        local = torch.zeros(
            (96, panel_count),
            device=solver.cuda_device,
            dtype=torch.float64,
        )
        for local_node in range(16):
            coefficient = pressure_shape_t[element, local_node] * area_ratio
            for component in range(3):
                local[local_node * 6 + component] = (
                    coefficient * normals_w[:, component]
                )
        result[element_dofs, :] += local
    if not bool(torch.isfinite(result).all().item()):
        raise FloatingPointError("Q16 distributed-pressure map is invalid")
    return result


@dataclass(frozen=True, slots=True, eq=False)
class _Q16CudaFrozenAddedMassLoad:
    """Q16-distributed UVLM added-mass force at one frozen outer endpoint.

    The panel Neumann operator is evaluated at the 3/4-chord collocation
    points, while pressure is integrated over each Q16 macro element with the
    author's chordwise ``p_interp`` rule.  Generic runs retain global Neumann
    columns after the dense AIC inverse.  ``author_element_local`` is the
    historical Q16-macro diagnostic.  The production Yamano mode uses the
    paper's aerodynamic Q4 element topology as an intermediate assembly space
    and projects it to Q16 by a kinematic/virtual-work transpose pair.
    """

    transfer: Any
    resolved: Q16PteraResolvedTransferResult
    generalized_matrix: torch.Tensor
    neumann_map: torch.Tensor
    gamma_rate_map: torch.Tensor
    pressure_to_generalized: torch.Tensor
    column_scope: str

    @classmethod
    def from_solver(
        cls,
        solver: CudaJointLEVTEVSolver,
        complete: Q16CompleteAeroLoadResult,
        transfer: Any,
    ) -> _Q16CudaFrozenAddedMassLoad:
        require_q16_mandatory_aero_mode(solver)
        complete.validate()
        resolved = complete.resolved
        _, span_count, chord_count = solver._panel_grid()
        panel_count = chord_count * span_count
        if resolved.point_count != 5 * panel_count:
            raise RuntimeError("added-mass panel topology drift")
        aic = solver._v5m_author_aic_cuda().detach().clone()
        if tuple(aic.shape) != (panel_count, panel_count):
            raise RuntimeError("added-mass AIC topology drift")
        normals_gp = torch.as_tensor(
            np.array(solver.stackUnitNormals_GP1, dtype=np.float64, copy=True),
            device=solver.cuda_device,
            dtype=torch.float64,
        )
        normals_w = solver._v5m_gp_vectors_to_scientific_cuda(normals_gp)
        areas = torch.as_tensor(
            np.array(solver.panel_areas, dtype=np.float64, copy=True),
            device=solver.cuda_device,
            dtype=torch.float64,
        )
        if tuple(normals_w.shape) != (panel_count, 3) or tuple(areas.shape) != (
            panel_count,
        ):
            raise RuntimeError("added-mass panel topology drift")
        if not bool(
            torch.isfinite(aic).all().item()
            and torch.isfinite(normals_w).all().item()
            and torch.isfinite(areas).all().item()
            and torch.all(areas > 0.0).item()
        ):
            raise FloatingPointError("frozen added-mass operator is invalid")
        vertex_map = transfer.vertex_map
        mesh = vertex_map.mesh
        dof_count = mesh.dof_count
        (
            q16_chord_count,
            q16_span_count,
            collocation_elements,
            collocation_shapes,
            pressure_shapes,
            chord_length,
            span_length,
        ) = _q16_added_mass_projection_setup(
            vertex_map,
            chordwise_panel_count=chord_count,
            spanwise_panel_count=span_count,
        )
        connectivity = torch.as_tensor(
            np.array(mesh.connectivity, dtype=np.int64, copy=True),
            device=solver.cuda_device,
            dtype=torch.int64,
        )
        collocation_element_t = torch.as_tensor(
            collocation_elements,
            device=solver.cuda_device,
            dtype=torch.int64,
        )
        collocation_shape_t = torch.as_tensor(
            collocation_shapes,
            device=solver.cuda_device,
            dtype=torch.float64,
        )
        panel_indices = torch.arange(
            panel_count, device=solver.cuda_device, dtype=torch.int64
        )
        neumann_map = torch.zeros(
            (panel_count, dof_count),
            device=solver.cuda_device,
            dtype=torch.float64,
        )
        for local_node in range(16):
            global_node = connectivity[collocation_element_t, local_node]
            for component in range(3):
                neumann_map[panel_indices, global_node * 6 + component] += (
                    collocation_shape_t[:, local_node] * normals_w[:, component]
                )
        gamma_rate_map = torch.linalg.solve(aic, neumann_map)
        reference_panel_area = (
            chord_length * span_length / float(chord_count * span_count)
        )
        area_ratio = areas / reference_panel_area
        pressure_shape_t = torch.as_tensor(
            pressure_shapes,
            device=solver.cuda_device,
            dtype=torch.float64,
        )
        column_scope = solver.jcfg.q16_added_mass_column_scope
        pressure_to_generalized = torch.zeros(
            (dof_count, panel_count),
            device=solver.cuda_device,
            dtype=torch.float64,
        )
        components = torch.arange(6, device=solver.cuda_device)
        density = float(solver.current_operating_point.rho)
        expected_element_count = q16_chord_count * q16_span_count
        if mesh.element_count != expected_element_count:
            raise RuntimeError("Q16 added-mass element grid drift")
        generalized_matrix = torch.zeros(
            (dof_count, dof_count),
            device=solver.cuda_device,
            dtype=torch.float64,
        )
        if column_scope == "author_aerodynamic_element_projection":
            projection = build_yamano2020_author_added_mass_projection(
                mesh,
                q16_chord_count=q16_chord_count,
                q16_span_count=q16_span_count,
                aerodynamic_chord_count=chord_count,
                aerodynamic_span_count=span_count,
                chord_length=chord_length,
                span_length=span_length,
                aic=aic,
                panel_normals=normals_w,
                density=density,
            )
            return cls(
                transfer=transfer,
                resolved=resolved,
                generalized_matrix=projection.generalized_matrix,
                neumann_map=projection.neumann_map,
                gamma_rate_map=projection.gamma_rate_map,
                pressure_to_generalized=projection.pressure_to_generalized,
                column_scope=column_scope,
            )
        for element in range(mesh.element_count):
            element_dofs = (
                connectivity[element].unsqueeze(1) * 6 + components.unsqueeze(0)
            ).reshape(-1)
            pressure_map = torch.zeros(
                (96, panel_count),
                device=solver.cuda_device,
                dtype=torch.float64,
            )
            for local_node in range(16):
                coefficient = pressure_shape_t[element, local_node] * area_ratio
                for component in range(3):
                    pressure_map[local_node * 6 + component] = (
                        coefficient * normals_w[:, component]
                    )
            pressure_to_generalized[element_dofs, :] += pressure_map
            if column_scope == "author_element_local":
                local_matrix = -density * (
                    pressure_map @ gamma_rate_map[:, element_dofs]
                )
                generalized_matrix[
                    element_dofs.unsqueeze(1), element_dofs.unsqueeze(0)
                ] += local_matrix
        if column_scope == "global":
            generalized_matrix = -density * (
                pressure_to_generalized @ gamma_rate_map
            )
        if tuple(generalized_matrix.shape) != (dof_count, dof_count) or not bool(
            torch.isfinite(generalized_matrix).all().item()
        ):
            raise FloatingPointError("assembled added-mass matrix is invalid")
        return cls(
            transfer=transfer,
            resolved=resolved,
            generalized_matrix=generalized_matrix.detach().clone(),
            neumann_map=neumann_map.detach().clone(),
            gamma_rate_map=gamma_rate_map.detach().clone(),
            pressure_to_generalized=pressure_to_generalized.detach().clone(),
            column_scope=column_scope,
        )

    def map(self, structural_acceleration: wp.array) -> wp.array:
        acceleration = wp.to_torch(structural_acceleration)
        force = torch.matmul(acceleration, self.generalized_matrix.transpose(0, 1))
        if force.device.type != "cuda" or force.dtype is not torch.float64:
            raise RuntimeError("added-mass matrix action left CUDA float64")
        return wp.from_torch(force, dtype=config.DTYPE, requires_grad=False)


@dataclass(frozen=True, slots=True, eq=False)
class _Q16CudaFrozenAuthorCirculatoryLoad:
    """Yamano ``Qf_p_global`` pressure load on the Q16 field.

    The source model does not transpose Ptera's four line-vortex point forces
    to the structure.  It evaluates ``dp_lift1 = V_surf1 dot grad(Gamma)`` at
    panel collocation points, applies the paper's chordwise ``p_interp``
    pressure reconstruction in its Q4 aerodynamic assembly space, and only
    then maps virtual work to Q16.  The separated-LEV impulse remains an
    independent source-owned generalized load and is added exactly once.
    """

    circulatory_pressure_pa: torch.Tensor
    wake_motion_pressure_pa: torch.Tensor
    pressure_pa: torch.Tensor
    pressure_to_generalized: torch.Tensor
    wake_motion_generalized_force: wp.array
    lev_impulse_generalized_force: wp.array
    generalized_force: wp.array

    @classmethod
    def from_solver(
        cls,
        solver: CudaJointLEVTEVSolver,
        complete: Q16CompleteAeroLoadResult,
        lift2: _Q16CudaFrozenLift2VelocityLoad,
        added_mass: _Q16CudaFrozenAddedMassLoad,
    ) -> _Q16CudaFrozenAuthorCirculatoryLoad:
        require_q16_mandatory_aero_mode(solver)
        complete.validate()
        if added_mass.column_scope != "author_aerodynamic_element_projection":
            raise ValueError(
                "author circulatory pressure requires the paper Q4 projection"
            )
        panel_count = int(lift2.dp_lift2_w.shape[0])
        collocation_gp = torch.as_tensor(
            np.array(solver.stackCpp_GP1_CgP1, dtype=np.float64, copy=True),
            device=solver.cuda_device,
            dtype=torch.float64,
        )
        if tuple(collocation_gp.shape) != (panel_count, 3):
            raise RuntimeError("author circulatory collocation topology drift")
        velocity_gp = solver._solution_velocity_cuda(collocation_gp)
        velocity_w = solver._v5m_gp_vectors_to_scientific_cuda(velocity_gp)
        # ``lift2.dp_lift2_w`` is rho*(tau_x*dGamma/dx +
        # tau_y*dGamma/dy).  The velocity-independent source term uses the
        # positive contraction; the structural-velocity lift2 term below uses
        # its negative contraction.
        circulatory_pressure = torch.sum(
            velocity_w * lift2.dp_lift2_w, dim=1
        )
        wake_motion_pressure = solver._author_wake_motion_pressure_cuda()
        if wake_motion_pressure.shape != circulatory_pressure.shape:
            raise RuntimeError("author wake-motion pressure topology drift")
        pressure = circulatory_pressure + wake_motion_pressure
        pressure_map = added_mass.pressure_to_generalized
        if tuple(pressure_map.shape) != (
            complete.resolved.generalized_force.shape[1],
            panel_count,
        ):
            raise RuntimeError("author circulatory pressure projection drift")
        wake_motion_generalized_t = torch.matmul(
            wake_motion_pressure, pressure_map.transpose(0, 1)
        ).unsqueeze(0)
        generalized_t = torch.matmul(
            pressure, pressure_map.transpose(0, 1)
        ).unsqueeze(0) + wp.to_torch(complete.lev_impulse_generalized_force)
        if (
            pressure.device.type != "cuda"
            or pressure.dtype is not torch.float64
            or generalized_t.device.type != "cuda"
            or generalized_t.dtype is not torch.float64
            or not bool(
                torch.isfinite(circulatory_pressure).all().item()
                and torch.isfinite(wake_motion_pressure).all().item()
                and torch.isfinite(pressure).all().item()
                and torch.isfinite(wake_motion_generalized_t).all().item()
                and torch.isfinite(generalized_t).all().item()
            )
        ):
            raise FloatingPointError("author circulatory pressure load is invalid")
        return cls(
            circulatory_pressure_pa=circulatory_pressure.detach().clone(),
            wake_motion_pressure_pa=wake_motion_pressure.detach().clone(),
            pressure_pa=pressure.detach().clone(),
            pressure_to_generalized=pressure_map.detach().clone(),
            wake_motion_generalized_force=wp.clone(
                wp.from_torch(
                    wake_motion_generalized_t,
                    dtype=config.DTYPE,
                    requires_grad=False,
                )
            ),
            lev_impulse_generalized_force=wp.clone(
                complete.lev_impulse_generalized_force
            ),
            generalized_force=wp.clone(
                wp.from_torch(
                    generalized_t,
                    dtype=config.DTYPE,
                    requires_grad=False,
                )
            ),
        )

    def map(self) -> wp.array:
        return wp.clone(self.generalized_force)


@dataclass(frozen=True, slots=True, eq=False)
class _Q16CudaInterpolatedAddedMassAction:
    """One structural-substep UVLM acceleration Jacobian on CUDA."""

    generalized_matrix: torch.Tensor

    @classmethod
    def between(
        cls,
        current: _Q16CudaFrozenAddedMassLoad,
        endpoint: _Q16CudaFrozenAddedMassLoad,
        fraction: float,
    ) -> _Q16CudaInterpolatedAddedMassAction:
        beta = float(fraction)
        if not math.isfinite(beta) or beta < 0.0 or beta > 1.0:
            raise ValueError("added-mass interpolation fraction must lie in [0, 1]")
        matrix = torch.lerp(
            current.generalized_matrix,
            endpoint.generalized_matrix,
            beta,
        )
        if matrix.device.type != "cuda" or matrix.dtype is not torch.float64:
            raise RuntimeError("interpolated added-mass operator left CUDA float64")
        return cls(generalized_matrix=matrix)

    def __call__(self, structural_acceleration: wp.array) -> wp.array:
        acceleration = wp.to_torch(structural_acceleration)
        force = torch.matmul(acceleration, self.generalized_matrix.transpose(0, 1))
        if force.device.type != "cuda" or force.dtype is not torch.float64:
            raise RuntimeError("interpolated added-mass action left CUDA float64")
        return wp.from_torch(force, dtype=config.DTYPE, requires_grad=False)


@dataclass(frozen=True, slots=True, eq=False)
class _Q16CudaFrozenMf21VelocityLoad:
    """Yamano ``Qf_p_mat0`` normal-rate force on the Q16 structural clock."""

    transfer: Any
    resolved: Q16PteraResolvedTransferResult
    vertex_transfer: Any
    vertex_positions_w: torch.Tensor
    aic: torch.Tensor
    panel_normals_w: torch.Tensor
    panel_areas: torch.Tensor
    external_flow_w: torch.Tensor
    chordwise_panel_count: int
    spanwise_panel_count: int
    fluid_density: float
    pressure_to_generalized: torch.Tensor

    @classmethod
    def from_solver(
        cls,
        solver: CudaJointLEVTEVSolver,
        complete: Q16CompleteAeroLoadResult,
        transfer: Any,
        vertex_transfer: Any,
        structural_state: wp.array,
    ) -> _Q16CudaFrozenMf21VelocityLoad:
        require_q16_mandatory_aero_mode(solver)
        complete.validate()
        resolved = complete.resolved
        panels, span_count, chord_count = solver._panel_grid()
        panel_count = chord_count * span_count
        if resolved.point_count != 5 * panel_count:
            raise RuntimeError("Mf2_1 point topology drift")
        vertices_w = wp.to_torch(vertex_transfer.interpolate(structural_state))[0]
        if tuple(vertices_w.shape) != (
            (chord_count + 1) * (span_count + 1),
            3,
        ):
            raise RuntimeError("Mf2_1 vertex topology drift")
        normals_gp = torch.as_tensor(
            np.array(solver.stackUnitNormals_GP1, dtype=np.float64, copy=True),
            device=solver.cuda_device,
            dtype=torch.float64,
        )
        normals_w = solver._v5m_gp_vectors_to_scientific_cuda(normals_gp)
        collocation_gp = torch.as_tensor(
            np.array(solver.stackCpp_GP1_CgP1, dtype=np.float64, copy=True),
            device=solver.cuda_device,
            dtype=torch.float64,
        )
        wake_gp = solver._wake_velocity(collocation_gp)
        # Match Yamano's Qf_p_mat0 owner: the historical material wake enters
        # this slip term.  Separated DVM particles already enter the outer
        # Neumann/KJ solve; querying their field here would both double-count
        # that owner and mutate the live particle diagnostic counter.
        wake_w = solver._v5m_gp_vectors_to_scientific_cuda(wake_gp)
        v_inf_gp = torch.as_tensor(
            np.array(
                solver.current_operating_point.vInf_GP1__E,
                dtype=np.float64,
                copy=True,
            ),
            device=solver.cuda_device,
            dtype=torch.float64,
        )
        v_inf_w = solver._v5m_gp_vectors_to_scientific_cuda(
            v_inf_gp.unsqueeze(0)
        )[0]
        external_flow_w = wake_w + v_inf_w.unsqueeze(0)
        areas = torch.as_tensor(
            np.array(solver.panel_areas, dtype=np.float64, copy=True),
            device=solver.cuda_device,
            dtype=torch.float64,
        )
        aic = solver._v5m_author_aic_cuda().detach().clone()
        finite = (
            torch.isfinite(vertices_w).all()
            & torch.isfinite(normals_w).all()
            & torch.isfinite(external_flow_w).all()
            & torch.isfinite(areas).all()
            & torch.isfinite(aic).all()
        )
        if not bool(finite.item()) or not bool(torch.all(areas > 0.0).item()):
            raise FloatingPointError("frozen Mf2_1 operator is invalid")
        pressure_to_generalized = _q16_distributed_pressure_map(solver, transfer)
        return cls(
            transfer=transfer,
            resolved=resolved,
            vertex_transfer=vertex_transfer,
            vertex_positions_w=vertices_w.detach().clone(),
            aic=aic,
            panel_normals_w=normals_w.detach().clone(),
            panel_areas=areas.detach().clone(),
            external_flow_w=external_flow_w.detach().clone(),
            chordwise_panel_count=chord_count,
            spanwise_panel_count=span_count,
            fluid_density=float(solver.current_operating_point.rho),
            pressure_to_generalized=pressure_to_generalized,
        )

    def map(self, structural_velocity: wp.array) -> wp.array:
        chord_count = self.chordwise_panel_count
        span_count = self.spanwise_panel_count
        vertex_velocity = wp.to_torch(
            self.vertex_transfer.interpolate(structural_velocity)
        )[0].reshape(chord_count + 1, span_count + 1, 3)
        position = self.vertex_positions_w.reshape(chord_count + 1, span_count + 1, 3)
        diagonal_one = position[1:, 1:] - position[:-1, :-1]
        diagonal_two = position[:-1, 1:] - position[1:, :-1]
        diagonal_one_rate = vertex_velocity[1:, 1:] - vertex_velocity[:-1, :-1]
        diagonal_two_rate = vertex_velocity[:-1, 1:] - vertex_velocity[1:, :-1]
        cross = torch.linalg.cross(diagonal_one, diagonal_two, dim=2)
        cross_rate = torch.linalg.cross(
            diagonal_one_rate, diagonal_two, dim=2
        ) + torch.linalg.cross(diagonal_one, diagonal_two_rate, dim=2)
        cross_norm = torch.linalg.vector_norm(cross, dim=2)
        normal = cross / cross_norm.unsqueeze(2)
        normal_rate_unprojected = cross_rate / cross_norm.unsqueeze(2)
        normal_rate = normal_rate_unprojected - normal * torch.sum(
            normal * normal_rate_unprojected, dim=2, keepdim=True
        )
        # The diagonal cross product depends on the structural vertex ordering,
        # whereas the AIC and pressure projection are owned by Ptera's panel
        # orientation.  Orient both n and dn/dt to that frozen aerodynamic
        # normal; using opposite conventions here reverses Mf2_1 damping while
        # leaving every dimensional/unit check apparently valid.
        aerodynamic_normal = self.panel_normals_w.reshape(chord_count, span_count, 3)
        orientation_dot = torch.sum(normal * aerodynamic_normal, dim=2)
        if bool(torch.any(torch.abs(orientation_dot) <= 1.0e-12).item()):
            raise FloatingPointError(
                "Mf2_1 structural and aerodynamic normals are orthogonal"
            )
        orientation = torch.sign(orientation_dot).unsqueeze(2)
        normal = normal * orientation
        normal_rate = normal_rate * orientation
        point_velocity = self.transfer._interpolate_frozen_point_direction_prechecked(
            self.resolved, structural_velocity
        )
        panel_count = chord_count * span_count
        collocation_velocity = point_velocity[4 * panel_count :]
        slip = collocation_velocity - self.external_flow_w
        scalar = torch.sum(slip * normal_rate.reshape(panel_count, 3), dim=1)
        # Author convention: A*Gamma=V_normal and this term is
        # A^{-1}[(v_surface-v_inf-v_wake) dot dn/dt].  The former extra minus
        # belonged only to the opposite Ptera ring orientation.
        pressure = torch.linalg.solve(self.aic, scalar)
        generalized = self.fluid_density * torch.matmul(
            pressure, self.pressure_to_generalized.transpose(0, 1)
        ).unsqueeze(0)
        if not bool(torch.isfinite(generalized).all().item()):
            raise FloatingPointError("Mf2_1 velocity force is non-finite")
        return wp.from_torch(generalized, dtype=config.DTYPE, requires_grad=False)


def _aggregate_structural_work(
    balances: tuple[Q16StructuralWorkBalance, ...],
    state_start: wp.array,
    state_end: wp.array,
) -> Q16StructuralWorkBalance:
    if not balances:
        raise ValueError("structural work aggregation requires at least one substep")
    kinetic_change = sum(value.kinetic_energy_change for value in balances)
    internal_work = sum(value.internal_trapezoidal_work for value in balances)
    damping_work = sum(value.damping_trapezoidal_work for value in balances)
    external_work = sum(value.external_trapezoidal_work for value in balances)
    balance_residual = kinetic_change + internal_work + damping_work - external_work
    scale = max(
        1.0,
        abs(kinetic_change),
        abs(internal_work),
        abs(damping_work),
        abs(external_work),
    )
    increment = wp.to_torch(state_end) - wp.to_torch(state_start)
    if increment.device.type != "cuda" or increment.dtype is not torch.float64:
        raise RuntimeError("Q16 substep work aggregation left CUDA float64")
    increment_norm = float(torch.linalg.vector_norm(increment).item())
    values = (
        balances[0].kinetic_energy_start,
        balances[-1].kinetic_energy_end,
        kinetic_change,
        internal_work,
        damping_work,
        external_work,
        balance_residual,
        abs(balance_residual) / scale,
        increment_norm,
        balances[-1].deformation_norm_end,
        balances[-1].velocity_norm_end,
        balances[-1].acceleration_norm_end,
    )
    if not all(math.isfinite(value) for value in values):
        raise FloatingPointError("Q16 substep work aggregation became non-finite")
    return Q16StructuralWorkBalance(*values)


def _relative_fixed_point_error(
    state: wp.array,
    velocity: wp.array,
    reference_state: wp.array,
    reference_velocity: wp.array,
    *,
    delta_time: float,
) -> float:
    q = wp.to_torch(state)
    v = wp.to_torch(velocity)
    q_reference = wp.to_torch(reference_state)
    v_reference = wp.to_torch(reference_velocity)
    state_error = torch.linalg.vector_norm(q - q_reference)
    velocity_error = delta_time * torch.linalg.vector_norm(v - v_reference)
    scale = torch.maximum(
        torch.ones((), dtype=torch.float64, device=q.device),
        torch.maximum(
            torch.linalg.vector_norm(q), torch.linalg.vector_norm(q_reference)
        ),
    )
    result = torch.maximum(state_error, velocity_error) / scale
    scalar = float(result.item())
    if not math.isfinite(scalar) or scalar < 0.0:
        raise FloatingPointError("Q16 FSI fixed-point residual became non-finite")
    return scalar


def _relax(current: wp.array, target: wp.array, factor: float) -> wp.array:
    current_t = wp.to_torch(current)
    target_t = wp.to_torch(target)
    relaxed = current_t + factor * (target_t - current_t)
    return wp.clone(wp.from_torch(relaxed, dtype=config.DTYPE, requires_grad=False))


class _Q16CudaAitkenRelaxer:
    """Scalar Aitken relaxation from the full CUDA q / dt*dq residual."""

    __slots__ = (
        "factor",
        "factor_history",
        "maximum_factor",
        "minimum_factor",
        "previous_state_residual",
        "previous_velocity_residual",
    )

    def __init__(
        self,
        initial_factor: float,
        *,
        minimum_factor: float = 0.05,
        maximum_factor: float = 1.0,
    ) -> None:
        self.factor = initial_factor
        self.minimum_factor = minimum_factor
        self.maximum_factor = maximum_factor
        self.previous_state_residual: torch.Tensor | None = None
        self.previous_velocity_residual: torch.Tensor | None = None
        self.factor_history: list[float] = []

    def advance(
        self,
        current_state: wp.array,
        current_velocity: wp.array,
        target_state: wp.array,
        target_velocity: wp.array,
        *,
        delta_time: float,
        dynamic: bool,
    ) -> tuple[wp.array, wp.array]:
        current_q = wp.to_torch(current_state)
        current_v = wp.to_torch(current_velocity)
        state_residual = wp.to_torch(target_state) - current_q
        velocity_residual = delta_time * (wp.to_torch(target_velocity) - current_v)
        values = (current_q, current_v, state_residual, velocity_residual)
        if any(
            value.device.type != "cuda" or value.dtype is not torch.float64
            for value in values
        ):
            raise RuntimeError("Q16 Aitken relaxation left CUDA float64")

        factor = self.factor
        if (
            dynamic
            and self.previous_state_residual is not None
            and self.previous_velocity_residual is not None
        ):
            delta_state = state_residual - self.previous_state_residual
            delta_velocity = velocity_residual - self.previous_velocity_residual
            denominator = torch.sum(delta_state * delta_state) + torch.sum(
                delta_velocity * delta_velocity
            )
            numerator = torch.sum(self.previous_state_residual * delta_state) + (
                torch.sum(self.previous_velocity_residual * delta_velocity)
            )
            if bool(torch.isfinite(denominator).item()) and float(denominator) > 0.0:
                candidate = -factor * numerator / denominator
                candidate = torch.clamp(
                    candidate,
                    min=self.minimum_factor,
                    max=self.maximum_factor,
                )
                if bool(torch.isfinite(candidate).item()):
                    factor = float(candidate)
        self.previous_state_residual = state_residual.detach().clone()
        self.previous_velocity_residual = velocity_residual.detach().clone()
        self.factor = factor
        self.factor_history.append(factor)

        relaxed_state = current_q + factor * state_residual
        relaxed_velocity = current_v + (factor / delta_time) * velocity_residual
        return (
            wp.clone(
                wp.from_torch(relaxed_state, dtype=config.DTYPE, requires_grad=False)
            ),
            wp.clone(
                wp.from_torch(relaxed_velocity, dtype=config.DTYPE, requires_grad=False)
            ),
        )


class Q16RealFSIStepStopped(RuntimeError):
    """A strong-coupling step stopped before either owner was published."""

    def __init__(
        self,
        message: str,
        *,
        phase: str,
        coupling_iteration_count: int,
        aerodynamic_evaluation_count: int,
        relative_residual: float,
        residual_history: tuple[float, ...],
        relaxation_factor_history: tuple[float, ...],
    ) -> None:
        super().__init__(message)
        self.phase = phase
        self.coupling_iteration_count = coupling_iteration_count
        self.aerodynamic_evaluation_count = aerodynamic_evaluation_count
        self.relative_residual = relative_residual
        self.residual_history = residual_history
        self.relaxation_factor_history = relaxation_factor_history


class Q16CudaRealFSIOwner:
    """Joint owner for one structural state and one committed aero solver."""

    __slots__ = (
        "_acceleration",
        "_aero_owner",
        "_generation",
        "_state",
        "_state_sha256",
        "_velocity",
    )

    def __init__(
        self,
        *,
        aero_owner: Q16CudaAeroSolverOwner,
        state: wp.array,
        velocity: wp.array,
        acceleration: wp.array,
    ) -> None:
        if type(aero_owner) is not Q16CudaAeroSolverOwner:
            raise TypeError("aero_owner must be an exact Q16CudaAeroSolverOwner")
        state_sha = _warp_sha256("state", state)
        velocity_sha = _warp_sha256("velocity", velocity)
        acceleration_sha = _warp_sha256("acceleration", acceleration)
        if not (state.shape == velocity.shape == acceleration.shape):
            raise ValueError("structural owner input shapes differ")
        if not (
            state.device.alias == velocity.device.alias == acceleration.device.alias
        ):
            raise ValueError("structural owner input devices differ")
        frozen = tuple(_clone_state(value) for value in (state, velocity, acceleration))
        if (
            _warp_sha256("state", frozen[0]) != state_sha
            or _warp_sha256("velocity", frozen[1]) != velocity_sha
            or _warp_sha256("acceleration", frozen[2]) != acceleration_sha
        ):
            raise RuntimeError("structural state changed while being detached")
        self._aero_owner = aero_owner
        self._state, self._velocity, self._acceleration = frozen
        self._generation = 0
        self._state_sha256 = _state_sha256(*frozen)

    @property
    def aero_owner(self) -> Q16CudaAeroSolverOwner:
        return self._aero_owner

    @property
    def state(self) -> wp.array:
        return self._state

    @property
    def velocity(self) -> wp.array:
        return self._velocity

    @property
    def acceleration(self) -> wp.array:
        return self._acceleration

    @property
    def generation(self) -> int:
        return self._generation

    @property
    def state_sha256(self) -> str:
        return self._state_sha256

    def _assert_live(self) -> None:
        if _state_sha256(self._state, self._velocity, self._acceleration) != (
            self._state_sha256
        ):
            raise RuntimeError("live Q16 structural owner drift")

    def _prepare_publish(
        self, result: Q16StructuralStepResult
    ) -> _Q16PreparedStructuralPublish:
        frozen = tuple(
            _clone_state(value)
            for value in (result.state, result.velocity, result.acceleration)
        )
        return _Q16PreparedStructuralPublish(
            state=frozen[0],
            velocity=frozen[1],
            acceleration=frozen[2],
            state_sha256=_state_sha256(*frozen),
        )

    def _publish_prechecked(self, prepared: _Q16PreparedStructuralPublish) -> None:
        self._state = prepared.state
        self._velocity = prepared.velocity
        self._acceleration = prepared.acceleration
        self._state_sha256 = prepared.state_sha256
        self._generation += 1


def _result_sha256(
    *,
    structural_state_sha256: str,
    aero_proposal_sha256: str,
    complete_load_sha256: str,
    prescribed_load_sha256: tuple[str | None, ...],
    total_external_force_sha256: str,
    structural_substep_count: int,
    aerodynamic_substep_scheme: str,
    coupling_iteration_count: int,
    aerodynamic_evaluation_count: int,
    relative_residual: float,
    work_balance: Q16StructuralWorkBalance,
    relaxation_factor_history: tuple[float, ...],
) -> str:
    payload = json.dumps(
        {
            "schema": _RESULT_SCHEMA,
            "structural_state_sha256": structural_state_sha256,
            "aero_proposal_sha256": aero_proposal_sha256,
            "complete_load_sha256": complete_load_sha256,
            "prescribed_load_sha256": list(prescribed_load_sha256),
            "structural_substep_count": structural_substep_count,
            "aerodynamic_substep_scheme": aerodynamic_substep_scheme,
            "total_external_force_sha256": total_external_force_sha256,
            "coupling_iteration_count": coupling_iteration_count,
            "aerodynamic_evaluation_count": aerodynamic_evaluation_count,
            "relative_residual_hex": relative_residual.hex(),
            "relaxation_factor_history_hex": [
                value.hex() for value in relaxation_factor_history
            ],
            "work_balance": {
                "acceleration_norm_end_hex": (work_balance.acceleration_norm_end.hex()),
                "balance_residual_hex": work_balance.balance_residual.hex(),
                "deformation_norm_end_hex": (work_balance.deformation_norm_end.hex()),
                "damping_trapezoidal_work_hex": (
                    work_balance.damping_trapezoidal_work.hex()
                ),
                "external_trapezoidal_work_hex": (
                    work_balance.external_trapezoidal_work.hex()
                ),
                "internal_trapezoidal_work_hex": (
                    work_balance.internal_trapezoidal_work.hex()
                ),
                "kinetic_energy_change_hex": (work_balance.kinetic_energy_change.hex()),
                "kinetic_energy_end_hex": work_balance.kinetic_energy_end.hex(),
                "kinetic_energy_start_hex": (work_balance.kinetic_energy_start.hex()),
                "relative_balance_residual_hex": (
                    work_balance.relative_balance_residual.hex()
                ),
                "state_increment_norm_hex": (work_balance.state_increment_norm.hex()),
                "velocity_norm_end_hex": work_balance.velocity_norm_end.hex(),
            },
        },
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True, eq=False)
class _Q16PreparedStructuralPublish:
    state: wp.array
    velocity: wp.array
    acceleration: wp.array
    state_sha256: str


@dataclass(frozen=True, slots=True, eq=False)
class Q16RealFSIStepResult:
    structural: Q16StructuralStepResult
    committed_solver: CudaJointLEVTEVSolver
    complete_load: Q16CompleteAeroLoadResult
    prescribed_load: Q16CudaPrescribedEndpointLoad | None
    prescribed_substep_loads: tuple[Q16CudaPrescribedEndpointLoad | None, ...]
    structural_substep_count: int
    aerodynamic_substep_scheme: str
    total_external_force: wp.array
    total_external_force_sha256: str
    coupling_iteration_count: int
    aerodynamic_evaluation_count: int
    relative_residual: float
    work_balance: Q16StructuralWorkBalance
    relaxation_factor_history: tuple[float, ...]
    owner_generation: int
    result_sha256: str


class Q16CudaRealFSIStepper:
    """Outer strong-coupling solve over one immutable pre-step owner."""

    __slots__ = (
        "binder",
        "complete_transfer",
        "coupling_tolerance",
        "max_coupling_iterations",
        "relaxation",
        "relaxation_method",
        "required_separated_source",
        "structural_stepper",
    )

    def __init__(
        self,
        *,
        structural_stepper: Q16CudaNewmarkStepper,
        binder: Q16CudaPteraIncrementalGeometry,
        complete_transfer: Q16CudaCompleteAeroLoadTransfer,
        coupling_tolerance: float,
        max_coupling_iterations: int,
        relaxation: float,
        relaxation_method: str = "aitken",
        required_separated_source: str | None = None,
    ) -> None:
        if type(structural_stepper) is not Q16CudaNewmarkStepper:
            raise TypeError("structural_stepper must have exact production type")
        if type(binder) is not Q16CudaPteraIncrementalGeometry:
            raise TypeError("binder must have exact production type")
        if type(complete_transfer) is not Q16CudaCompleteAeroLoadTransfer:
            raise TypeError("complete_transfer must have exact production type")
        tolerance = _positive_float("coupling_tolerance", coupling_tolerance)
        if tolerance >= 1.0:
            raise ValueError("coupling_tolerance must be less than one")
        relax = _positive_float("relaxation", relaxation)
        if relax > 1.0:
            raise ValueError("relaxation must be at most one")
        if type(relaxation_method) is not str or relaxation_method not in {
            "aitken",
            "fixed",
        }:
            raise ValueError("relaxation_method must be 'aitken' or 'fixed'")
        if required_separated_source is not None and (
            type(required_separated_source) is not str
            or required_separated_source not in {"hirato_ring", "dvm_node_ribbon"}
        ):
            raise ValueError("required_separated_source is unsupported")
        if structural_stepper.device != binder.surface_transfer.device:
            raise ValueError("structural and Ptera binders use different CUDA devices")
        if (
            structural_stepper.dof_count
            != complete_transfer.resolved_transfer.structural_dof_count
            or structural_stepper.dof_count
            != binder.surface_transfer.structural_dof_count
        ):
            raise ValueError("Q16 structural and aerodynamic DOF owners differ")
        self.structural_stepper = structural_stepper
        self.binder = binder
        self.complete_transfer = complete_transfer
        self.coupling_tolerance = tolerance
        self.max_coupling_iterations = _positive_int(
            "max_coupling_iterations", max_coupling_iterations
        )
        self.relaxation = relax
        self.relaxation_method = relaxation_method
        self.required_separated_source = required_separated_source

    def advance(
        self,
        owner: Q16CudaRealFSIOwner,
        *,
        delta_time: float,
        prescribed_load: Q16CudaPrescribedEndpointLoad | None = None,
        structural_substep_count: int = 1,
        prescribed_substep_loads: (
            tuple[Q16CudaPrescribedEndpointLoad | None, ...] | None
        ) = None,
        aerodynamic_substep_scheme: str = "endpoint_hold",
    ) -> Q16RealFSIStepResult:
        if type(owner) is not Q16CudaRealFSIOwner:
            raise TypeError("owner must be an exact Q16CudaRealFSIOwner")
        dt = _positive_float("delta_time", delta_time)
        owner._assert_live()
        mode = require_q16_mandatory_aero_mode(owner.aero_owner.current_solver)
        if (
            self.required_separated_source is not None
            and mode.separated_source != self.required_separated_source
        ):
            raise RuntimeError(
                "Q16 FSI aerodynamic source differs from its frozen contract"
            )
        aero_dt = float(owner.aero_owner.current_solver.delta_time)
        if not math.isfinite(aero_dt) or aero_dt <= 0.0 or dt != aero_dt:
            raise ValueError("FSI delta_time must exactly match aerodynamic delta_time")
        if owner.state.shape[1] != self.structural_stepper.dof_count:
            raise ValueError("owner and Q16 structural DOF counts differ")
        substep_count = _positive_int(
            "structural_substep_count", structural_substep_count
        )
        if (
            type(aerodynamic_substep_scheme) is not str
            or aerodynamic_substep_scheme not in _AERODYNAMIC_SUBSTEP_SCHEMES
        ):
            raise ValueError(
                "unsupported aerodynamic_substep_scheme"
            )
        if prescribed_load is not None and prescribed_substep_loads is not None:
            raise ValueError("provide one prescribed load form, not both")
        if prescribed_load is not None and substep_count != 1:
            raise ValueError("one prescribed_load requires one structural substep")
        if prescribed_substep_loads is None:
            schedule: tuple[Q16CudaPrescribedEndpointLoad | None, ...] = (
                (prescribed_load,)
                if substep_count == 1
                else tuple(None for _ in range(substep_count))
            )
        else:
            if type(prescribed_substep_loads) is not tuple:
                raise TypeError("prescribed_substep_loads must be an exact tuple")
            if len(prescribed_substep_loads) != substep_count:
                raise ValueError(
                    "prescribed substep schedule length differs from count"
                )
            schedule = prescribed_substep_loads
        previous_endpoint = -math.inf
        for load in schedule:
            if load is None:
                continue
            if type(load) is not Q16CudaPrescribedEndpointLoad:
                raise TypeError(
                    "prescribed substep load must have exact production type"
                )
            load.validate()
            if load.generalized_force.shape != owner.state.shape:
                raise ValueError("prescribed load and structural owner shapes differ")
            if load.generalized_force.device.alias != self.structural_stepper.device:
                raise ValueError("prescribed load and structure use different devices")
            if load.endpoint_time_s <= previous_endpoint:
                raise ValueError("prescribed substep endpoint times must increase")
            previous_endpoint = load.endpoint_time_s
        state_n = owner.state
        velocity_n = owner.velocity
        acceleration_n = owner.acceleration
        if (
            aerodynamic_substep_scheme == "author_corrector"
            and self.structural_stepper.nonsymmetric_solver == "reference_dense"
        ):
            self.structural_stepper.refresh_reference_tangent(state_n)
        structural_generation = owner.generation
        aero_generation = owner.aero_owner.generation
        latest_complete: Q16CompleteAeroLoadResult | None = None
        latest_kj_velocity_operator: _Q16CudaFrozenKJVelocityLoad | None = None
        latest_lift2_operator: _Q16CudaFrozenLift2VelocityLoad | None = None
        latest_added_mass_operator: _Q16CudaFrozenAddedMassLoad | None = None
        latest_mf21_operator: _Q16CudaFrozenMf21VelocityLoad | None = None
        latest_author_circulatory: (
            _Q16CudaFrozenAuthorCirculatoryLoad | None
        ) = None
        latest_ptera_unsteady_force: wp.array | None = None
        latest_endpoint_velocity: wp.array | None = None

        current_aerodynamic_force: wp.array | None = None
        current_aerodynamic_constant: wp.array | None = None
        current_kj_velocity_operator: _Q16CudaFrozenKJVelocityLoad | None = None
        current_lift2_operator: _Q16CudaFrozenLift2VelocityLoad | None = None
        current_added_mass_operator: _Q16CudaFrozenAddedMassLoad | None = None
        current_mf21_operator: _Q16CudaFrozenMf21VelocityLoad | None = None
        current_author_circulatory: (
            _Q16CudaFrozenAuthorCirculatoryLoad | None
        ) = None
        current_ptera_unsteady_force: wp.array | None = None
        if aerodynamic_substep_scheme in {"block_linear", "author_corrector"}:
            current_solver = owner.aero_owner.current_solver
            current_packet = Q16CudaAerodynamicLoadPacket.from_solver(current_solver)
            current_lev = Q16CudaLEVImpulseStripLoad.from_solver(current_solver)
            current_complete = self.complete_transfer.map(
                current_packet, current_lev, state_n
            )
            current_aerodynamic_force = current_complete.generalized_force
            current_ptera_unsteady_force = _ptera_unsteady_generalized_force(
                current_packet,
                self.complete_transfer.resolved_transfer,
                state_n,
            )
            current_kj_velocity_operator = _Q16CudaFrozenKJVelocityLoad.from_solver(
                current_solver,
                current_complete,
                self.complete_transfer.resolved_transfer,
            )
            current_lift2_operator = _Q16CudaFrozenLift2VelocityLoad.from_solver(
                current_solver,
                current_complete,
                self.complete_transfer.resolved_transfer,
                self.binder.surface_transfer,
                state_n,
            )
            current_added_mass_operator = _Q16CudaFrozenAddedMassLoad.from_solver(
                current_solver,
                current_complete,
                self.complete_transfer.resolved_transfer,
            )
            # Yamano's zero ``old_Qf_p_mat_global`` belongs to the discarded
            # first predictor pass.  Before the retained corrector replay the
            # code executes ``old=current_first`` and keeps
            # ``Qf_p_mat_global_a=current_first``.  The committed trajectory
            # therefore interpolates the first fluid anchor to the new trial
            # endpoint; it must not carry the predictor's zero history.
            current_mf21_operator = _Q16CudaFrozenMf21VelocityLoad.from_solver(
                current_solver,
                current_complete,
                self.complete_transfer.resolved_transfer,
                self.binder.surface_transfer,
                state_n,
            )
            if aerodynamic_substep_scheme == "author_corrector":
                current_author_circulatory = (
                    _Q16CudaFrozenAuthorCirculatoryLoad.from_solver(
                        current_solver,
                        current_complete,
                        current_lift2_operator,
                        current_added_mass_operator,
                    )
                )
                current_aerodynamic_constant = current_author_circulatory.map()
            else:
                # Remove the Ptera endpoint movement and finite-difference
                # dGamma terms once.  Their distributed lift2 and Mf1
                # counterparts are applied on every structural substep below.
                current_velocity_force = current_kj_velocity_operator.map(velocity_n)
                current_aerodynamic_constant = _combine_cuda_forces(
                    _combine_cuda_forces(
                        current_aerodynamic_force,
                        current_ptera_unsteady_force,
                        second_factor=-1.0,
                    ),
                    current_velocity_force,
                    second_factor=-1.0,
                )

        def integrate_structure(
            aerodynamic_force_end: wp.array,
            *,
            audit_work: bool,
        ) -> tuple[
            Q16StructuralStepResult,
            Q16StructuralWorkBalance | None,
            wp.array,
        ]:
            substep_dt = dt / substep_count
            state = state_n
            velocity = velocity_n
            acceleration = acceleration_n
            balances: list[Q16StructuralWorkBalance] = []
            newton_iterations = 0
            cg_iterations = 0
            gmres_iterations = 0
            direct_solves = 0
            live_tangent_refreshes = 0
            indefinite_fallbacks = 0
            residual_max = 0.0
            last_result: Q16StructuralStepResult | None = None
            last_total: wp.array | None = None

            if aerodynamic_substep_scheme in {"block_linear", "author_corrector"}:
                if (
                    current_aerodynamic_constant is None
                    or current_kj_velocity_operator is None
                    or current_lift2_operator is None
                    or current_added_mass_operator is None
                    or current_mf21_operator is None
                    or latest_kj_velocity_operator is None
                    or latest_lift2_operator is None
                    or latest_added_mass_operator is None
                    or latest_mf21_operator is None
                    or latest_ptera_unsteady_force is None
                    or latest_endpoint_velocity is None
                ):
                    raise AssertionError(
                        "block-linear velocity operator evidence is missing"
                    )
                if aerodynamic_substep_scheme == "author_corrector":
                    if latest_author_circulatory is None:
                        raise AssertionError(
                            "author circulatory endpoint evidence is missing"
                        )
                    endpoint_aerodynamic_constant = (
                        latest_author_circulatory.map()
                    )
                else:
                    endpoint_velocity_force = latest_kj_velocity_operator.map(
                        latest_endpoint_velocity
                    )
                    endpoint_aerodynamic_constant = _combine_cuda_forces(
                        _combine_cuda_forces(
                            aerodynamic_force_end,
                            latest_ptera_unsteady_force,
                            second_factor=-1.0,
                        ),
                        endpoint_velocity_force,
                        second_factor=-1.0,
                    )

                def scheduled_force(
                    anchor: wp.array,
                    endpoint: wp.array,
                    beta: float,
                ) -> wp.array:
                    # The final Yamano corrector replay is anchor-to-endpoint
                    # interpolation.  A roughly doubled value visible just
                    # before the endpoint fluid solve belongs to the discarded
                    # predictor and is not the committed structural history.
                    return _interpolate_aerodynamic_force(anchor, endpoint, beta)

                def aerodynamic_at(
                    structural_velocity: wp.array,
                    beta: float,
                ) -> wp.array:
                    constant = scheduled_force(
                        current_aerodynamic_constant,
                        endpoint_aerodynamic_constant,
                        beta,
                    )
                    current_velocity_force_at = current_lift2_operator.map(
                        structural_velocity
                    )
                    endpoint_velocity_force_at = latest_lift2_operator.map(
                        structural_velocity
                    )
                    velocity_force = scheduled_force(
                        current_velocity_force_at,
                        endpoint_velocity_force_at,
                        beta,
                    )
                    current_mf21 = current_mf21_operator.map(structural_velocity)
                    endpoint_mf21 = latest_mf21_operator.map(structural_velocity)
                    mf21_force = scheduled_force(
                        current_mf21,
                        endpoint_mf21,
                        beta,
                    )
                    return _combine_cuda_forces(
                        _combine_cuda_forces(constant, velocity_force),
                        mf21_force,
                    )

            for substep_index, load in enumerate(schedule):
                if aerodynamic_substep_scheme in {
                    "block_linear",
                    "author_corrector",
                }:
                    # In solve_structure.m the retained Yamano replay starts
                    # at ``time_fluid`` itself, so its first substep consumes
                    # beta=0 and its last consumes (N-1)/N.  The generic
                    # block-linear scheme remains endpoint-inclusive.
                    beta = (
                        substep_index / substep_count
                        if aerodynamic_substep_scheme == "author_corrector"
                        else (substep_index + 1) / substep_count
                    )
                    added_mass_action = (
                        _Q16CudaInterpolatedAddedMassAction.between(
                            current_added_mass_operator,
                            latest_added_mass_operator,
                            beta,
                        )
                    )
                    aerodynamic_force = aerodynamic_at(
                        velocity,
                        beta,
                    )
                    predictor_total = _total_external_force(aerodynamic_force, load)
                    predictor = self.structural_stepper.step(
                        state,
                        velocity,
                        acceleration,
                        predictor_total,
                        delta_time=substep_dt,
                        acceleration_load_action=added_mass_action,
                    )
                    average_velocity = _average_cuda_state(velocity, predictor.velocity)
                    aerodynamic_force = aerodynamic_at(
                        average_velocity,
                        beta,
                    )
                else:
                    aerodynamic_force = aerodynamic_force_end
                total = _total_external_force(aerodynamic_force, load)
                result = self.structural_stepper.step(
                    state,
                    velocity,
                    acceleration,
                    total,
                    delta_time=substep_dt,
                    acceleration_load_action=(
                        added_mass_action
                        if aerodynamic_substep_scheme
                        in {"block_linear", "author_corrector"}
                        else None
                    ),
                )
                if aerodynamic_substep_scheme in {
                    "block_linear",
                    "author_corrector",
                }:
                    newton_iterations += predictor.newton_iteration_count
                    cg_iterations += predictor.cg_iteration_count
                    gmres_iterations += predictor.gmres_iteration_count
                    direct_solves += predictor.direct_solve_count
                    live_tangent_refreshes += (
                        predictor.live_tangent_refresh_count
                    )
                    indefinite_fallbacks += predictor.indefinite_fallback_count
                    residual_max = max(residual_max, predictor.relative_residual_max)
                effective_total = total
                if aerodynamic_substep_scheme in {
                    "block_linear",
                    "author_corrector",
                }:
                    effective_total = _combine_cuda_forces(
                        total, added_mass_action(result.acceleration)
                    )
                if audit_work:
                    balances.append(
                        self.structural_stepper.audit_step_work(
                            state,
                            velocity,
                            acceleration,
                            result.state,
                            result.velocity,
                            result.acceleration,
                            effective_total,
                        )
                    )
                newton_iterations += result.newton_iteration_count
                cg_iterations += result.cg_iteration_count
                gmres_iterations += result.gmres_iteration_count
                direct_solves += result.direct_solve_count
                live_tangent_refreshes += result.live_tangent_refresh_count
                indefinite_fallbacks += result.indefinite_fallback_count
                residual_max = max(residual_max, result.relative_residual_max)
                state = result.state
                velocity = result.velocity
                acceleration = result.acceleration
                last_result = result
                last_total = effective_total
            if last_result is None or last_total is None:
                raise AssertionError("Q16 structural substep schedule is empty")
            aggregate = Q16StructuralStepResult(
                state=last_result.state,
                velocity=last_result.velocity,
                acceleration=last_result.acceleration,
                reaction=last_result.reaction,
                delta_time=dt,
                newton_iteration_count=newton_iterations,
                cg_iteration_count=cg_iterations,
                gmres_iteration_count=gmres_iterations,
                direct_solve_count=direct_solves,
                live_tangent_refresh_count=live_tangent_refreshes,
                indefinite_fallback_count=indefinite_fallbacks,
                relative_residual_max=residual_max,
            )
            work = (
                _aggregate_structural_work(tuple(balances), state_n, aggregate.state)
                if audit_work
                else None
            )
            return aggregate, work, last_total

        def evaluate_trial(
            solver: CudaJointLEVTEVSolver,
            q_trial: wp.array,
            dq_trial: wp.array,
        ) -> wp.array:
            nonlocal latest_complete, latest_endpoint_velocity
            nonlocal latest_kj_velocity_operator, latest_lift2_operator
            nonlocal latest_added_mass_operator
            nonlocal latest_mf21_operator
            nonlocal latest_author_circulatory
            nonlocal latest_ptera_unsteady_force
            session = Q16CudaIncrementalAeroSession.resume(solver)
            self.binder.bind_next_state(session, q_trial, dq_trial)
            session.advance_one_step()
            packet = Q16CudaAerodynamicLoadPacket.from_solver(session.solver)
            lev_load = Q16CudaLEVImpulseStripLoad.from_solver(session.solver)
            complete = self.complete_transfer.map(packet, lev_load, q_trial)
            latest_complete = complete
            latest_ptera_unsteady_force = _ptera_unsteady_generalized_force(
                packet,
                self.complete_transfer.resolved_transfer,
                q_trial,
            )
            latest_kj_velocity_operator = _Q16CudaFrozenKJVelocityLoad.from_solver(
                session.solver,
                complete,
                self.complete_transfer.resolved_transfer,
            )
            latest_lift2_operator = _Q16CudaFrozenLift2VelocityLoad.from_solver(
                session.solver,
                complete,
                self.complete_transfer.resolved_transfer,
                self.binder.surface_transfer,
                q_trial,
            )
            latest_added_mass_operator = _Q16CudaFrozenAddedMassLoad.from_solver(
                session.solver,
                complete,
                self.complete_transfer.resolved_transfer,
            )
            latest_mf21_operator = _Q16CudaFrozenMf21VelocityLoad.from_solver(
                session.solver,
                complete,
                self.complete_transfer.resolved_transfer,
                self.binder.surface_transfer,
                q_trial,
            )
            if aerodynamic_substep_scheme == "author_corrector":
                latest_author_circulatory = (
                    _Q16CudaFrozenAuthorCirculatoryLoad.from_solver(
                        session.solver,
                        complete,
                        latest_lift2_operator,
                        latest_added_mass_operator,
                    )
                )
            latest_endpoint_velocity = wp.clone(dq_trial)
            return complete.generalized_force

        def fork_incremental(
            solver: CudaJointLEVTEVSolver,
        ) -> CudaJointLEVTEVSolver:
            return Q16CudaIncrementalAeroSession.resume(solver).fork().solver

        transaction = owner.aero_owner.begin(
            evaluate_trial,
            branch_factory=fork_incremental,
        )
        q_guess, dq_guess = self.structural_stepper.predict_kinematics(
            state_n,
            velocity_n,
            acceleration_n,
            delta_time=dt,
        )
        latest_residual = math.inf
        residual_history: list[float] = []
        relaxer = _Q16CudaAitkenRelaxer(self.relaxation)
        try:
            for iteration in range(1, self.max_coupling_iterations + 1):
                proposal = transaction.evaluate(q_guess, dq_guess)
                structural, _, _ = integrate_structure(
                    proposal.generalized_force,
                    audit_work=False,
                )
                latest_residual = _relative_fixed_point_error(
                    structural.state,
                    structural.velocity,
                    q_guess,
                    dq_guess,
                    delta_time=dt,
                )
                residual_history.append(latest_residual)
                if latest_residual <= self.coupling_tolerance:
                    formal_proposal = transaction.evaluate(
                        structural.state, structural.velocity
                    )
                    formal_structural, work_balance, formal_total_force = (
                        integrate_structure(
                            formal_proposal.generalized_force,
                            audit_work=True,
                        )
                    )
                    formal_residual = _relative_fixed_point_error(
                        formal_structural.state,
                        formal_structural.velocity,
                        structural.state,
                        structural.velocity,
                        delta_time=dt,
                    )
                    latest_residual = formal_residual
                    residual_history.append(formal_residual)
                    if formal_residual <= self.coupling_tolerance:
                        if latest_complete is None:
                            raise RuntimeError("complete aero load evidence is missing")
                        if work_balance is None:
                            raise AssertionError(
                                "Q16 structural work evidence is missing"
                            )
                        owner._assert_live()
                        if (
                            owner.generation != structural_generation
                            or owner.aero_owner.generation != aero_generation
                        ):
                            raise RuntimeError("Q16 FSI owner generation drift")
                        # Complete every fallible validation, clone and hash
                        # before either live owner changes.  The final section
                        # contains no callback or numerical operation: it only
                        # swaps the two already-prepared owners.
                        prepared_aero = transaction._prepare_commit(formal_proposal)
                        prepared_structural = owner._prepare_publish(formal_structural)
                        result_sha = _result_sha256(
                            structural_state_sha256=(prepared_structural.state_sha256),
                            aero_proposal_sha256=formal_proposal.proposal_sha256,
                            complete_load_sha256=latest_complete.result_sha256,
                            prescribed_load_sha256=tuple(
                                None if load is None else load.load_sha256
                                for load in schedule
                            ),
                            total_external_force_sha256=_warp_sha256(
                                "total external force", formal_total_force
                            ),
                            structural_substep_count=substep_count,
                            aerodynamic_substep_scheme=(aerodynamic_substep_scheme),
                            coupling_iteration_count=iteration,
                            aerodynamic_evaluation_count=(transaction.evaluation_count),
                            relative_residual=formal_residual,
                            work_balance=work_balance,
                            relaxation_factor_history=(tuple(relaxer.factor_history)),
                        )
                        result = Q16RealFSIStepResult(
                            structural=formal_structural,
                            committed_solver=formal_proposal.proposed_solver,
                            complete_load=latest_complete,
                            prescribed_load=prescribed_load,
                            prescribed_substep_loads=schedule,
                            structural_substep_count=substep_count,
                            aerodynamic_substep_scheme=(aerodynamic_substep_scheme),
                            total_external_force=wp.clone(formal_total_force),
                            total_external_force_sha256=_warp_sha256(
                                "total external force", formal_total_force
                            ),
                            coupling_iteration_count=iteration,
                            aerodynamic_evaluation_count=(transaction.evaluation_count),
                            relative_residual=formal_residual,
                            work_balance=work_balance,
                            relaxation_factor_history=(tuple(relaxer.factor_history)),
                            owner_generation=structural_generation + 1,
                            result_sha256=result_sha,
                        )
                        transaction._commit_prepared_prechecked(prepared_aero)
                        owner._publish_prechecked(prepared_structural)
                        return result
                    q_guess, dq_guess = relaxer.advance(
                        structural.state,
                        structural.velocity,
                        formal_structural.state,
                        formal_structural.velocity,
                        delta_time=dt,
                        dynamic=self.relaxation_method == "aitken",
                    )
                else:
                    q_guess, dq_guess = relaxer.advance(
                        q_guess,
                        dq_guess,
                        structural.state,
                        structural.velocity,
                        delta_time=dt,
                        dynamic=self.relaxation_method == "aitken",
                    )
        except Exception:
            if transaction.status == "open":
                transaction.abort()
            raise
        if transaction.status == "open":
            transaction.abort()
        raise Q16RealFSIStepStopped(
            "Q16 strong FSI fixed-point iteration did not converge",
            phase="coupling_convergence",
            coupling_iteration_count=self.max_coupling_iterations,
            aerodynamic_evaluation_count=transaction.evaluation_count,
            relative_residual=latest_residual,
            residual_history=tuple(residual_history),
            relaxation_factor_history=tuple(relaxer.factor_history),
        )


__all__ = [
    "Q16CudaRealFSIOwner",
    "Q16CudaRealFSIStepper",
    "Q16RealFSIStepResult",
    "Q16RealFSIStepStopped",
]
