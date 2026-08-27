"""Author pressure operators on the direct Q16/native-FLUX-V5M data path.

This module contains no legacy structural or aerodynamic runtime object.  The immutable
quadrature topology is prepared once on the host, while every scientific
operator, action and solve is assembled and evaluated on CUDA float64.
"""
from __future__ import annotations

# Set True only around an active CUDA-graph capture region: host-side
# validation gates must defer (they sync); the capture wrapper validates
# the replayed static outputs after the capture ends.
CAPTURING = False

from dataclasses import dataclass
import math
from typing import Any

import numpy as np
import torch
import warp as wp

from fluxvortex.q16_work_conjugate_transfer import Q16SurfaceTransferMap

from . import config
from .kernels_q16_transfer import Q16CudaSurfaceTransfer


_FOUR_PI = 4.0 * math.pi


def _p_interp_uniform_weights(
    local_chord: float,
    chord_panel: int,
    chordwise_panel_count: int,
) -> tuple[float, float, float]:
    """Yamano ``p_interp.m`` weights for a uniform chordwise panel grid."""

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
    lower: float, upper: float, candidates: tuple[float, ...]
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
        raise RuntimeError("Q16 author-pressure quadrature partition is invalid")
    return result


def _axis_owner(value: float, element_count: int) -> tuple[int, float]:
    scaled = value * element_count
    owner = min(int(math.floor(scaled)), element_count - 1)
    local = 2.0 * (scaled - owner) - 1.0
    return owner, min(max(local, -1.0), 1.0)


@dataclass(frozen=True, slots=True)
class _PressureQuadratureTopology:
    transfer: Q16CudaSurfaceTransfer
    interpolation_area_weights: torch.Tensor
    target_panels: torch.Tensor


def _build_pressure_quadrature(surface: Any) -> _PressureQuadratureTopology:
    """Build the exact piecewise ``p_interp``/Q16 quadrature topology."""

    qx = int(surface.q16_chordwise_elements)
    qy = int(surface.q16_spanwise_elements)
    nc = int(surface.nc)
    ns = int(surface.ns)
    chord_length = float(np.ptp(surface.mesh.reference_rows[:, 0]))
    span_length = float(np.ptp(surface.mesh.reference_rows[:, 1]))
    if chord_length <= 0.0 or span_length <= 0.0:
        raise ValueError("formal Q16 pressure surface is degenerate")
    gauss_points, gauss_weights = np.polynomial.legendre.leggauss(3)
    chord_candidates = tuple(
        [value / nc for value in range(1, nc)]
        + [(value + 0.75) / nc for value in range(nc)]
    )
    span_candidates = tuple(value / ns for value in range(1, ns))
    elements: list[int] = []
    coordinates: list[tuple[float, float, float]] = []
    weight_rows: list[int] = []
    weight_cols: list[int] = []
    weight_values: list[float] = []
    target_panels: list[int] = []
    point = 0
    for span_element in range(qy):
        span_lower = span_element / qy
        span_upper = (span_element + 1) / qy
        span_partition = _strict_partition(
            span_lower, span_upper, span_candidates
        )
        for chord_element in range(qx):
            chord_lower = chord_element / qx
            chord_upper = (chord_element + 1) / qx
            chord_partition = _strict_partition(
                chord_lower, chord_upper, chord_candidates
            )
            element = span_element * qx + chord_element
            for chord_a, chord_b in zip(
                chord_partition[:-1], chord_partition[1:], strict=True
            ):
                chord_midpoint = 0.5 * (chord_a + chord_b)
                chord_panel = min(int(math.floor(chord_midpoint * nc)), nc - 1)
                for span_a, span_b in zip(
                    span_partition[:-1], span_partition[1:], strict=True
                ):
                    span_midpoint = 0.5 * (span_a + span_b)
                    span_panel = min(int(math.floor(span_midpoint * ns)), ns - 1)
                    for chord_gauss, chord_weight in zip(
                        gauss_points, gauss_weights, strict=True
                    ):
                        chord_fraction = 0.5 * (
                            chord_a + chord_b + chord_gauss * (chord_b - chord_a)
                        )
                        local_chord = chord_fraction * nc - chord_panel
                        pressure_weights = _p_interp_uniform_weights(
                            local_chord, chord_panel, nc
                        )
                        _, xi = _axis_owner(chord_fraction, qx)
                        for span_gauss, span_weight in zip(
                            gauss_points, gauss_weights, strict=True
                        ):
                            span_fraction = 0.5 * (
                                span_a + span_b + span_gauss * (span_b - span_a)
                            )
                            _, eta = _axis_owner(span_fraction, qy)
                            area_weight = (
                                chord_weight
                                * span_weight
                                * 0.25
                                * (chord_b - chord_a)
                                * chord_length
                                * (span_b - span_a)
                                * span_length
                            )
                            elements.append(element)
                            coordinates.append((xi, eta, 0.0))
                            target_panels.append(chord_panel * ns + span_panel)
                            for offset, coefficient in zip(
                                (-1, 0, 1), pressure_weights, strict=True
                            ):
                                neighbor = chord_panel + offset
                                if coefficient == 0.0 or not 0 <= neighbor < nc:
                                    continue
                                weight_rows.append(point)
                                weight_cols.append(neighbor * ns + span_panel)
                                weight_values.append(area_weight * coefficient)
                            point += 1
    transfer_map = Q16SurfaceTransferMap(
        mesh=surface.mesh,
        element_indices=np.ascontiguousarray(elements, dtype=np.int64),
        parametric_coordinates=np.ascontiguousarray(
            coordinates, dtype=np.float64
        ),
    )
    device = torch.device(surface.device)
    weights = torch.zeros(
        (point, nc * ns), device=device, dtype=torch.float64
    )
    weights.index_put_(
        (
            torch.tensor(weight_rows, device=device, dtype=torch.int64),
            torch.tensor(weight_cols, device=device, dtype=torch.int64),
        ),
        torch.tensor(weight_values, device=device, dtype=torch.float64),
        accumulate=True,
    )
    if not bool(torch.isfinite(weights).all().item()):
        raise FloatingPointError("Q16 author-pressure weights are non-finite")
    return _PressureQuadratureTopology(
        transfer=Q16CudaSurfaceTransfer(
            transfer_map, device=str(surface.device)
        ),
        interpolation_area_weights=weights,
        target_panels=torch.tensor(
            target_panels, device=device, dtype=torch.int64
        ),
    )


def material_ring_velocity_derivative_expanded(
    points: torch.Tensor,
    point_velocities: torch.Tensor,
    rings: torch.Tensor,
    ring_velocities: torch.Tensor,
) -> torch.Tensor:
    """Analytic material derivative of Yamano's unregularized ring influence."""

    values = (points, point_velocities, rings, ring_velocities)
    if any(type(value) is not torch.Tensor for value in values):
        raise TypeError("material ring derivative inputs must be exact tensors")
    device = points.device
    if any(
        value.device != device
        or value.device.type != "cuda"
        or value.dtype is not torch.float64
        for value in values
    ):
        raise ValueError("material ring derivative requires CUDA float64")
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("material ring targets must have shape (n,3)")
    if point_velocities.shape != points.shape:
        raise ValueError("material ring target positions/velocities differ")
    if rings.ndim != 3 or rings.shape[1:] != (4, 3):
        raise ValueError("material rings must have shape (m,4,3)")
    if ring_velocities.shape != rings.shape:
        raise ValueError("material ring positions/velocities differ")

    def line_derivative(
        origin: torch.Tensor,
        destination: torch.Tensor,
        origin_velocity: torch.Tensor,
        destination_velocity: torch.Tensor,
    ) -> torch.Tensor:
        a = points[:, None, :] - origin[None, :, :]
        b = points[:, None, :] - destination[None, :, :]
        da = point_velocities[:, None, :] - origin_velocity[None, :, :]
        db = point_velocities[:, None, :] - destination_velocity[None, :, :]
        cross = torch.linalg.cross(a, b, dim=2)
        cross_rate = torch.linalg.cross(da, b, dim=2) + torch.linalg.cross(
            a, db, dim=2
        )
        cross_norm_sq = torch.sum(cross * cross, dim=2)
        a_norm = torch.linalg.vector_norm(a, dim=2)
        b_norm = torch.linalg.vector_norm(b, dim=2)
        singular = (cross_norm_sq <= 0.0) | (a_norm <= 0.0) | (b_norm <= 0.0)
        if not CAPTURING and bool(torch.any(singular).item()):
            raise FloatingPointError("material ring derivative is singular")
        unit_difference = a / a_norm[:, :, None] - b / b_norm[:, :, None]
        unit_difference_rate = (
            da / a_norm[:, :, None]
            - a
            * (torch.sum(da * a, dim=2) / a_norm**3)[:, :, None]
            - db / b_norm[:, :, None]
            + b
            * (torch.sum(db * b, dim=2) / b_norm**3)[:, :, None]
        )
        edge = a - b
        edge_rate = da - db
        scalar = torch.sum(edge * unit_difference, dim=2)
        scalar_rate = torch.sum(edge_rate * unit_difference, dim=2) + torch.sum(
            edge * unit_difference_rate, dim=2
        )
        cross_fraction_rate = cross_rate / cross_norm_sq[:, :, None] - 2.0 * (
            cross
            * (
                torch.sum(cross_rate * cross, dim=2) / cross_norm_sq**2
            )[:, :, None]
        )
        return cross_fraction_rate * scalar[:, :, None] + (
            cross / cross_norm_sq[:, :, None]
        ) * scalar_rate[:, :, None]

    derivative = torch.zeros(
        (points.shape[0], rings.shape[0], 3),
        device=device,
        dtype=torch.float64,
    )
    for leg in range(4):
        destination = (leg + 1) % 4
        derivative += line_derivative(
            rings[:, leg],
            rings[:, destination],
            ring_velocities[:, leg],
            ring_velocities[:, destination],
        )
    # Native ring orientation is the direct [1->2->3->4->1] circulation.
    derivative /= _FOUR_PI
    if not CAPTURING and not bool(torch.isfinite(derivative).all().item()):
        raise FloatingPointError("material ring derivative is non-finite")
    return derivative


@dataclass(frozen=True, slots=True, eq=False)
class Q16NativeAddedMassAction:
    generalized_matrix: torch.Tensor

    def __call__(self, structural_acceleration: wp.array) -> wp.array:
        acceleration = wp.to_torch(structural_acceleration)
        force = acceleration @ self.generalized_matrix.T
        if force.device.type != "cuda" or force.dtype is not torch.float64:
            raise RuntimeError("native Mf1 action left CUDA float64")
        return wp.from_torch(force, dtype=config.DTYPE, requires_grad=False)

    @classmethod
    def between(
        cls,
        anchor: "Q16NativeAddedMassAction",
        endpoint: "Q16NativeAddedMassAction",
        beta: float,
    ) -> "Q16NativeAddedMassAction":
        fraction = float(beta)
        if not 0.0 <= fraction <= 1.0:
            raise ValueError("native Mf1 interpolation fraction is invalid")
        return cls(torch.lerp(anchor.generalized_matrix, endpoint.generalized_matrix, fraction))


@dataclass(frozen=True, slots=True, eq=False)
class Q16NativeAuthorEndpointLoad:
    """Frozen author pressure decomposition at one native V5M endpoint."""

    surface: Any
    structural_state: wp.array
    geometry: Any
    aic: torch.Tensor
    gamma: torch.Tensor
    external_flow: torch.Tensor
    gamma_gradient: torch.Tensor
    pressure_to_generalized: torch.Tensor
    dp_lift1: torch.Tensor
    mf2_history: torch.Tensor
    constant_pressure: torch.Tensor
    constant_generalized_force: wp.array
    added_mass: Q16NativeAddedMassAction
    # Fluid density: the velocity part (lift2 + Mf2_1) must carry rho exactly
    # like the constant part (lift1 + Mf2) — the frozen reference paths
    # (warp_fsi/coupled.py dp_lift1_flat_kernel, platform/warp_vpm/
    # q16_real_fsi_coupling.py) multiply every pressure term by rho.
    density: float = 1.0
    # Lazy LU of the fixed aic; the same aic is solved ~80x per outer step.
    _aic_lu: Any = None

    def velocity_force(self, structural_velocity: wp.array) -> wp.array:
        live = self.surface.evaluate(self.structural_state, structural_velocity)
        # Both velocity-part pressures carry rho (the reference frozen paths
        # bake rho into dp_lift2_w and multiply the Mf2_1 solve by rho): a
        # bare -V·grad(Gamma) is m^2/s^2 and cannot be summed with the Pa
        # constant part.
        lift2_pressure = -self.density * torch.sum(
            live.collocation_velocity * self.gamma_gradient, dim=1
        )
        diagonal_31 = self.geometry.rings[:, 3] - self.geometry.rings[:, 1]
        diagonal_24 = self.geometry.rings[:, 2] - self.geometry.rings[:, 0]
        diagonal_31_rate = live.ring_velocity[:, 3] - live.ring_velocity[:, 1]
        diagonal_24_rate = live.ring_velocity[:, 2] - live.ring_velocity[:, 0]
        cross = torch.linalg.cross(diagonal_31, diagonal_24, dim=1)
        cross_rate = torch.linalg.cross(
            diagonal_31_rate, diagonal_24, dim=1
        ) + torch.linalg.cross(diagonal_31, diagonal_24_rate, dim=1)
        cross_norm = torch.linalg.vector_norm(cross, dim=1)
        normal_rate_unprojected = cross_rate / cross_norm[:, None]
        normal_rate = normal_rate_unprojected - self.geometry.normals * torch.sum(
            self.geometry.normals * normal_rate_unprojected,
            dim=1,
            keepdim=True,
        )
        bound_influence_rate = material_ring_velocity_derivative_expanded(
            self.geometry.collocation,
            live.collocation_velocity,
            self.geometry.rings,
            live.ring_velocity,
        )
        bound_rate = torch.sum(
            bound_influence_rate * self.gamma[None, :, None], dim=1
        )
        slip = live.collocation_velocity - self.external_flow
        mf21_rhs = torch.sum(slip * normal_rate, dim=1) - torch.sum(
            bound_rate * self.geometry.normals, dim=1
        )
        lu = self._aic_lu
        if lu is None:
            lu = torch.linalg.lu_factor(self.aic)
            object.__setattr__(self, "_aic_lu", lu)
        mf21_pressure = self.density * torch.linalg.lu_solve(
            lu[0], lu[1], mf21_rhs.unsqueeze(1)
        ).squeeze(1)
        pressure = lift2_pressure + mf21_pressure
        generalized = pressure @ self.pressure_to_generalized.T
        if not CAPTURING and not bool(torch.isfinite(generalized).all().item()):
            raise FloatingPointError("native lift2/Mf2_1 force is non-finite")
        return wp.from_torch(
            generalized.unsqueeze(0), dtype=config.DTYPE, requires_grad=False
        )


class Q16NativeAuthorLoadAssembler:
    """Direct Q16 CUDA assembly of ``Qf_p_global``, ``Mf1`` and ``Mf2``."""

    def __init__(self, surface: Any, *, density: float) -> None:
        self.surface = surface
        self.device = torch.device(surface.device)
        self.density = float(density)
        self._quadrature = _build_pressure_quadrature(surface)

    def _neumann_map(self, geometry: Any) -> torch.Tensor:
        nc, ns = self.surface.nc, self.surface.ns
        panel_count = nc * ns
        dof_count = self.surface.mesh.dof_count
        unit_panel_forces = torch.zeros(
            (panel_count, panel_count, 3),
            device=self.device,
            dtype=torch.float64,
        )
        indices = torch.arange(panel_count, device=self.device)
        unit_panel_forces[indices, indices] = geometry.normals
        mapped = self.surface.panel_load_transfer.map_batch(unit_panel_forces)
        result = wp.to_torch(mapped).contiguous()
        if tuple(result.shape) != (panel_count, dof_count):
            raise RuntimeError("native Mf1 Neumann topology drift")
        return result

    def _pressure_map(self, geometry: Any) -> torch.Tensor:
        panel_count = self.surface.nc * self.surface.ns
        reference_area = (
            float(np.ptp(self.surface.mesh.reference_rows[:, 0]))
            * float(np.ptp(self.surface.mesh.reference_rows[:, 1]))
            / panel_count
        )
        area_ratio = geometry.areas / reference_area
        weights = self._quadrature.interpolation_area_weights
        forces = (
            weights.T[:, :, None]
            * area_ratio[:, None, None]
            * geometry.normals[:, None, :]
        )
        mapped = self._quadrature.transfer.transpose(
            wp.from_torch(forces, dtype=config.VEC3, requires_grad=False)
        )
        result = wp.to_torch(mapped).T.contiguous()
        if tuple(result.shape) != (self.surface.mesh.dof_count, panel_count):
            raise RuntimeError("native author pressure-map topology drift")
        if not bool(torch.isfinite(result).all().item()):
            raise FloatingPointError("native author pressure map is non-finite")
        return result

    def pressure_map(self, geometry: Any) -> torch.Tensor:
        """Return the direct-Q16 CUDA pressure transpose for one geometry."""

        return self._pressure_map(geometry)

    def _normal_shape_matrix(self, geometry: Any) -> torch.Tensor:
        """SN[i, q] = S_i(x_q) · n̂(x_q): normal-projected shape function at
        each pressure-quadrature point.

        Computed by applying the Q16 transfer to unit normal forces at each
        quadrature point, then transposing. This is the exact analogue of the
        author's ``Sc_mat_v^T · n̂`` contraction.
        """
        n_quad = int(self._quadrature.interpolation_area_weights.shape[0])
        panel_count = self.surface.nc * self.surface.ns

        # Interpolate panel normals at quadrature points (using p_interp
        # weights WITHOUT area, then renormalize)
        w = self._quadrature.interpolation_area_weights  # (n_quad, n_panels)
        w_raw = w / w.sum(dim=1, keepdim=True).clamp(min=1e-30)
        quad_normals = w_raw @ geometry.normals  # (n_quad, 3)
        quad_normals = quad_normals / torch.norm(
            quad_normals, dim=1, keepdim=True).clamp(min=1e-30)

        # Create batched unit normal forces: (n_quad, n_quad, 3)
        # entry [q, q, :] = normal at quadrature point q
        unit_forces = torch.zeros(
            (n_quad, n_quad, 3), device=self.device, dtype=torch.float64)
        idx = torch.arange(n_quad, device=self.device)
        unit_forces[idx, idx, :] = quad_normals

        # Apply Q16 transfer to get generalized force per quadrature point
        mapped = self._quadrature.transfer.transpose(
            wp.from_torch(unit_forces, dtype=config.VEC3,
                          requires_grad=False))
        sn = wp.to_torch(mapped).T.contiguous()  # (n_dof, n_quad)
        return sn

    def assemble(
        self,
        *,
        structural_state: wp.array,
        geometry: Any,
        aic: torch.Tensor,
        gamma: torch.Tensor,
        external_flow: torch.Tensor,
        gamma_gradient: torch.Tensor,
        mf2_history: torch.Tensor,
    ) -> Q16NativeAuthorEndpointLoad:
        pressure_map = self._pressure_map(geometry)
        neumann = self._neumann_map(geometry)
        gamma_rate = torch.linalg.solve(aic, neumann)

        # Author's Mf1 (calc_fluid_force.m:154 + strong [2-3]):
        #   Step 1: neumann[p, j] = n̂_p · S_j(panel_center)   (bare, no area)
        #   Step 2: gamma_rate = A⁻¹ @ neumann                 (DOF→panel pressure)
        #   Step 3: Mf1 = ρ · SN @ (P_w @ gamma_rate)          (integrate to DOF)
        # where SN[i, q] = S_i(x_q) · n̂(x_q) is the normal-projected shape
        # function at each quadrature point, and P_w[q, p] are the p_interp
        # area-weighted interpolation weights. This exactly mirrors the
        # author's three-step assembly: nvec_Sc_global → Mf1_mat → Qf_p_mat_i.
        sn = self._normal_shape_matrix(geometry)     # (n_dof, n_quad)
        pw = self._quadrature.interpolation_area_weights  # (n_quad, n_panels)
        # Author's Mf1: the dynamic-pressure convention q=ρU²/2 means the
        # added-mass force carries ρ/2. The non-dimensional Mf1_mat is
        # converted to dimensional generalized force via ρ/2, matching the
        # author's non-dimensional formulation where the factor is absorbed.
        added_mass_matrix = 0.5 * self.density * (
            sn @ (pw @ gamma_rate))                 # (n_dof, n_dof)
        dp_lift1 = self.density * torch.sum(external_flow * gamma_gradient, dim=1)
        constant_pressure = dp_lift1 + self.density * mf2_history
        constant_generalized = constant_pressure @ pressure_map.T
        values = (
            pressure_map,
            neumann,
            gamma_rate,
            added_mass_matrix,
            dp_lift1,
            constant_pressure,
            constant_generalized,
        )
        if not all(bool(torch.isfinite(value).all().item()) for value in values):
            raise FloatingPointError("native author load assembly is non-finite")
        return Q16NativeAuthorEndpointLoad(
            surface=self.surface,
            structural_state=wp.clone(structural_state),
            geometry=geometry,
            aic=aic.detach().clone(),
            gamma=gamma.detach().clone(),
            external_flow=external_flow.detach().clone(),
            gamma_gradient=gamma_gradient.detach().clone(),
            pressure_to_generalized=pressure_map.detach().clone(),
            dp_lift1=dp_lift1.detach().clone(),
            mf2_history=mf2_history.detach().clone(),
            constant_pressure=constant_pressure.detach().clone(),
            constant_generalized_force=wp.clone(
                wp.from_torch(
                    constant_generalized.unsqueeze(0),
                    dtype=config.DTYPE,
                    requires_grad=False,
                )
            ),
            added_mass=Q16NativeAddedMassAction(added_mass_matrix.detach().clone()),
            density=self.density,
        )


__all__ = [
    "Q16NativeAddedMassAction",
    "Q16NativeAuthorEndpointLoad",
    "Q16NativeAuthorLoadAssembler",
    "material_ring_velocity_derivative_expanded",
]
