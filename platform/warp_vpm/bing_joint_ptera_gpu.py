"""Strict CUDA numerical backend for the V5M Ptera chassis.

The attached and active-LEV post-hoc modes use one airplane, one wing, no image
surface and a prescribed wake.  Unsupported configurations fail closed instead
of falling back to NumPy/Numba.  Joint LEV/TEV is enabled only after its
augmented CUDA system passes the dedicated production gates.

Python and Ptera still own geometry construction, object lifecycle and result
serialization.  The aerodynamic numerics -- Biot--Savart evaluation, AIC/RHS,
dense solve, force/moment evaluation, reductions, and the LESP ledger -- execute
as float64 Torch CUDA operations.
"""

from __future__ import annotations

import math
import os
from typing import Any

import numpy as np

# Time-marching wake tensors grow every step.  Expandable CUDA allocator
# segments reuse that growth in-place instead of retaining one cached block per
# historical shape (which otherwise reserved almost the full 24 GiB device for
# <1 GiB of live tensors).  Respect an explicit operator override.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from pterasoftware import _vortices

from bing_joint_ptera import JointConfig, JointLEVTEVSolver, LESP_FACTOR
from bing_joint_solver import GateError
from ldvm_source_bank_gpu import CudaLDVMSourceBank
from pfield_torch_gpu import CudaParticleField

_EPS = float(np.finfo(float).eps)
_TOL = 1.0e-10
_FOUR_LAMB = 5.02572
_SQUIRE = 1.0e-4
_FOUR_PI = 4.0 * math.pi


def _dvm_ptera_pin_active_mask(
    cell_active: torch.Tensor,
    ptera_separated: torch.Tensor,
) -> torch.Tensor:
    """Keep material-release events distinct from separated boundary state."""

    if (
        type(cell_active) is not torch.Tensor
        or type(ptera_separated) is not torch.Tensor
    ):
        raise TypeError("DVM/Ptera activity masks must be exact torch tensors")
    if (
        cell_active.device.type != "cuda"
        or ptera_separated.device != cell_active.device
    ):
        raise ValueError("DVM/Ptera activity masks must share one CUDA device")
    if cell_active.dtype is not torch.bool or ptera_separated.dtype is not torch.bool:
        raise TypeError("DVM/Ptera activity masks must use torch.bool")
    if cell_active.ndim != 1 or ptera_separated.shape != cell_active.shape:
        raise ValueError("DVM/Ptera activity masks must share one vector shape")
    return cell_active | ptera_separated


def _transform_wrench_cuda(
    rotation: torch.Tensor,
    translation: torch.Tensor,
    forces: torch.Tensor,
    moments: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Transform forces and moments about the translated world origin.

    For ``x_W = R x_G + t``, the corresponding wrench is
    ``F_W = R F_G`` and ``M_W = R M_G + t x F_W``.  Adding ``t`` directly to
    a moment is dimensionally invalid and loses rigid-body moment balance.
    """

    if any(
        type(value) is not torch.Tensor
        for value in (rotation, translation, forces, moments)
    ):
        raise TypeError("wrench transform inputs must be exact torch tensors")
    device = rotation.device
    for name, value in (
        ("rotation", rotation),
        ("translation", translation),
        ("forces", forces),
        ("moments", moments),
    ):
        if value.device.type != "cuda" or value.device != device:
            raise ValueError(f"{name} must remain on one CUDA device")
        if value.dtype is not torch.float64:
            raise TypeError(f"{name} must use torch.float64")
        if not bool(torch.isfinite(value).all().item()):
            raise FloatingPointError(f"{name} contains non-finite values")
    if tuple(rotation.shape) != (3, 3) or tuple(translation.shape) != (3,):
        raise ValueError("rotation/translation shapes must be (3,3)/(3,)")
    if (
        forces.ndim != 2
        or tuple(forces.shape) != tuple(moments.shape)
        or forces.shape[1] != 3
    ):
        raise ValueError("forces and moments must share shape (count,3)")
    force_w = torch.einsum("ij,nj->ni", rotation, forces)
    moment_w = torch.einsum("ij,nj->ni", rotation, moments)
    moment_w = moment_w + torch.linalg.cross(
        translation.expand_as(force_w), force_w, dim=1
    )
    return force_w, moment_w


def _cuda_tensor(value: Any, device: torch.device) -> torch.Tensor:
    """Copy an orchestration-owned array to CUDA float64."""
    if type(value) is torch.Tensor:
        if value.device != device or value.device.type != "cuda":
            raise ValueError("internal science tensor crossed CUDA device boundary")
        if value.dtype is not torch.float64:
            raise TypeError("internal science tensor must use torch.float64")
        return value
    host = np.array(value, dtype=np.float64, order="C", copy=True)
    return torch.from_numpy(host).to(device=device)


def _line_velocity_expanded(
    points: torch.Tensor,
    starts: torch.Tensor,
    ends: torch.Tensor,
    strengths: torch.Tensor,
    core_radii: torch.Tensor,
    ages: torch.Tensor,
    nu: float,
) -> torch.Tensor:
    """Ptera's regularised finite-line Biot--Savart law on CUDA.

    The result has shape ``(n_points, n_vortices, 3)``.  Singularity branches
    exactly follow Ptera's scale-invariant zero-contribution convention.
    """
    if starts.shape[0] == 0:
        return torch.zeros(
            (points.shape[0], 0, 3), dtype=torch.float64, device=points.device
        )
    r0v = ends - starts
    r0 = torch.linalg.vector_norm(r0v, dim=1)
    r1v = starts.unsqueeze(0) - points.unsqueeze(1)
    r2v = ends.unsqueeze(0) - points.unsqueeze(1)
    r3v = torch.linalg.cross(r1v, r2v, dim=2)
    r1 = torch.linalg.vector_norm(r1v, dim=2)
    r2 = torch.linalg.vector_norm(r2v, dim=2)
    r3_sq = torch.sum(r3v * r3v, dim=2)
    r3 = torch.sqrt(r3_sq)
    r1r2 = r1 * r2
    dot12 = torch.sum(r1v * r2v, dim=2)

    rc_sq = (
        core_radii * core_radii
        + _FOUR_LAMB * (float(nu) + _SQUIRE * torch.abs(strengths)) * ages
    )
    c1 = strengths / _FOUR_PI
    c2 = r0 * r0 * rc_sq
    denom = r1r2 * (r3_sq + c2.unsqueeze(0))
    numer = c1.unsqueeze(0) * (r1 + r2) * (r1r2 - dot12)
    safe_denom = torch.where(denom != 0.0, denom, torch.ones_like(denom))
    c4 = numer / safe_denom

    valid = torch.broadcast_to(r0.unsqueeze(0) >= _EPS, r1.shape).clone()
    valid = valid & (r1 >= r0.unsqueeze(0) * _TOL)
    valid = valid & (r2 >= r0.unsqueeze(0) * _TOL)
    valid = valid & (r3 >= _TOL * r1r2)
    valid = valid & (denom != 0.0)
    return torch.where(valid.unsqueeze(2), c4.unsqueeze(2) * r3v, 0.0)


def _ring_velocity_expanded(
    points: torch.Tensor,
    br: torch.Tensor,
    fr: torch.Tensor,
    fl: torch.Tensor,
    bl: torch.Tensor,
    strengths: torch.Tensor,
    core_radii: torch.Tensor,
    ages: torch.Tensor,
    nu: float,
) -> torch.Tensor:
    # Evaluate all four ring legs as one larger CUDA batch.  The previous
    # implementation launched the complete line-vortex tensor program four
    # times in Python, which left small/medium panel grids launch-bound.  The
    # flattened layout preserves the exact leg order in the final reduction.
    starts = torch.stack((br, fr, fl, bl), dim=0).reshape(-1, 3)
    ends = torch.stack((fr, fl, bl, br), dim=0).reshape(-1, 3)
    expanded = _line_velocity_expanded(
        points,
        starts,
        ends,
        strengths.repeat(4),
        core_radii.repeat(4),
        ages.repeat(4),
        nu,
    )
    return expanded.reshape(points.shape[0], 4, strengths.shape[0], 3).sum(dim=1)


_COMPILED_BOUND_RING_VELOCITY: Any = None
_COMPILED_COLLAPSED_RING_VELOCITY: Any = None


def _fused_ring_velocity_expanded(
    points: torch.Tensor,
    br: torch.Tensor,
    fr: torch.Tensor,
    fl: torch.Tensor,
    bl: torch.Tensor,
    strengths: torch.Tensor,
    core_radii: torch.Tensor,
    ages: torch.Tensor,
    nu: float,
) -> torch.Tensor:
    """Fused production path shared by bound and changing wake topologies.

    Small grids stay eager because compiler cold-start dominates them.  Paper
    meshes lazily prime one shape-polymorphic CUDA program.  Once primed, the
    same graph accepts changing wake-row counts, which keeps the time-marching
    loop from relaunching dozens of PyTorch microkernels per ring evaluation.
    """
    global _COMPILED_BOUND_RING_VELOCITY
    interactions = points.shape[0] * strengths.shape[0]
    enabled = os.environ.get("FLUXV_V5M_FUSE", "1") != "0"
    # Keep the frozen small-grid arithmetic independent of whether an earlier
    # test/run happened to prime the global compiled graph with a paper mesh.
    # Letting cache history switch small cases from eager to compiled changed
    # last-bit loads and could even move a LESP value across its activation
    # threshold.
    if not enabled or interactions < 4096:
        return _ring_velocity_expanded(
            points, br, fr, fl, bl, strengths, core_radii, ages, nu
        )
    if _COMPILED_BOUND_RING_VELOCITY is None:
        _COMPILED_BOUND_RING_VELOCITY = torch.compile(
            _ring_velocity_expanded,
            fullgraph=True,
            dynamic=True,
        )
    return _COMPILED_BOUND_RING_VELOCITY(
        points, br, fr, fl, bl, strengths, core_radii, ages, nu
    )


def _bound_ring_velocity_expanded(
    points: torch.Tensor,
    br: torch.Tensor,
    fr: torch.Tensor,
    fl: torch.Tensor,
    bl: torch.Tensor,
    strengths: torch.Tensor,
    core_radii: torch.Tensor,
    ages: torch.Tensor,
    nu: float,
) -> torch.Tensor:
    """Compatibility name for the fused bound-ring call sites."""
    return _fused_ring_velocity_expanded(
        points, br, fr, fl, bl, strengths, core_radii, ages, nu
    )


def _ring_velocity_collapsed(
    points: torch.Tensor,
    br: torch.Tensor,
    fr: torch.Tensor,
    fl: torch.Tensor,
    bl: torch.Tensor,
    strengths: torch.Tensor,
    core_radii: torch.Tensor,
    ages: torch.Tensor,
    nu: float,
) -> torch.Tensor:
    """Return total ring-induced velocity without retaining ring columns."""
    return _ring_velocity_expanded(
        points, br, fr, fl, bl, strengths, core_radii, ages, nu
    ).sum(dim=1)


def _fused_ring_velocity_collapsed(
    points: torch.Tensor,
    br: torch.Tensor,
    fr: torch.Tensor,
    fl: torch.Tensor,
    bl: torch.Tensor,
    strengths: torch.Tensor,
    core_radii: torch.Tensor,
    ages: torch.Tensor,
    nu: float,
) -> torch.Tensor:
    """Shape-polymorphic fused reduction for bound/wake velocity queries."""
    global _COMPILED_COLLAPSED_RING_VELOCITY
    interactions = points.shape[0] * strengths.shape[0]
    enabled = os.environ.get("FLUXV_V5M_FUSE", "1") != "0"
    if not enabled or interactions < 4096:
        return _ring_velocity_collapsed(
            points, br, fr, fl, bl, strengths, core_radii, ages, nu
        )
    if _COMPILED_COLLAPSED_RING_VELOCITY is None:
        _COMPILED_COLLAPSED_RING_VELOCITY = torch.compile(
            _ring_velocity_collapsed,
            fullgraph=True,
            dynamic=True,
        )
    return _COMPILED_COLLAPSED_RING_VELOCITY(
        points, br, fr, fl, bl, strengths, core_radii, ages, nu
    )


def _chunked_ring_velocity_collapsed(
    points: torch.Tensor,
    br: torch.Tensor,
    fr: torch.Tensor,
    fl: torch.Tensor,
    bl: torch.Tensor,
    strengths: torch.Tensor,
    core_radii: torch.Tensor,
    ages: torch.Tensor,
    nu: float,
    *,
    target_chunk: int = 1024,
    source_chunk: int = 1024,
) -> torch.Tensor:
    """GPU-only bounded-memory ring reduction for growing material wakes."""

    interactions = points.shape[0] * strengths.shape[0]
    if interactions <= target_chunk * source_chunk:
        return _fused_ring_velocity_collapsed(
            points, br, fr, fl, bl, strengths, core_radii, ages, nu
        )
    velocity = torch.zeros_like(points)
    for target_start in range(0, points.shape[0], target_chunk):
        target_stop = min(target_start + target_chunk, points.shape[0])
        partial = torch.zeros_like(points[target_start:target_stop])
        for source_start in range(0, strengths.shape[0], source_chunk):
            source_stop = min(source_start + source_chunk, strengths.shape[0])
            partial = partial + _fused_ring_velocity_collapsed(
                points[target_start:target_stop],
                br[source_start:source_stop],
                fr[source_start:source_stop],
                fl[source_start:source_stop],
                bl[source_start:source_stop],
                strengths[source_start:source_stop],
                core_radii[source_start:source_stop],
                ages[source_start:source_stop],
                nu,
            )
        velocity[target_start:target_stop] = partial
    return velocity


def _unregularized_ring_velocity_expanded(
    points: torch.Tensor, rings: torch.Tensor
) -> torch.Tensor:
    """CUDA port of the frozen joint-system finite-ring column oracle."""

    epsilon = 10000.0 * 2.2204e-16
    origins = rings.transpose(0, 1).reshape(-1, 3)
    destinations = torch.roll(rings, shifts=-1, dims=1).transpose(0, 1).reshape(-1, 3)
    r1 = points[:, None] - origins[None]
    r2 = points[:, None] - destinations[None]
    cross = torch.linalg.cross(r1, r2, dim=2)
    n1 = torch.linalg.vector_norm(r1, dim=2)
    n2 = torch.linalg.vector_norm(r2, dim=2)
    n3 = torch.linalg.vector_norm(cross, dim=2)
    valid = (n1 > epsilon) & (n2 > epsilon) & (n3 > epsilon)
    n1_safe = torch.clamp(n1, min=epsilon)
    n2_safe = torch.clamp(n2, min=epsilon)
    n3_safe = torch.clamp(n3, min=epsilon)
    part1 = cross / (n3_safe * n3_safe).unsqueeze(2)
    part2 = (n1_safe + n2_safe).unsqueeze(2)
    dot = torch.sum(r1 * r2, dim=2)
    part3 = (1.0 - dot / (n1_safe * n2_safe)).unsqueeze(2)
    velocity = torch.where(valid.unsqueeze(2), part1 * part2 * part3 / _FOUR_PI, 0.0)
    return velocity.reshape(points.shape[0], 4, rings.shape[0], 3).sum(dim=1)


def _material_ring_velocity_derivative_expanded(
    points: torch.Tensor,
    point_velocities: torch.Tensor,
    r1: torch.Tensor,
    r2: torch.Tensor,
    r3: torch.Tensor,
    r4: torch.Tensor,
    v1: torch.Tensor,
    v2: torch.Tensor,
    v3: torch.Tensor,
    v4: torch.Tensor,
) -> torch.Tensor:
    """Material derivative of the authors' unregularized finite rings.

    This is a CUDA-float64 port of Yamano's ``dt_generate_q1234_mat.m``.
    Its output has shape ``(target_count, ring_count, 3)`` and is the time
    derivative of the unit-circulation velocity influence, not a finite
    difference of already collapsed wake velocities.
    """

    values = (points, point_velocities, r1, r2, r3, r4, v1, v2, v3, v4)
    if any(type(value) is not torch.Tensor for value in values):
        raise TypeError("material ring derivative inputs must be exact tensors")
    device = points.device
    if any(
        value.device != device
        or value.device.type != "cuda"
        or value.dtype is not torch.float64
        for value in values
    ):
        raise ValueError(
            "material ring derivative requires one CUDA float64 device"
        )
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("material ring derivative targets must have shape (n,3)")
    if point_velocities.shape != points.shape:
        raise ValueError("target positions and velocities differ")
    ring_shape = r1.shape
    if (
        r1.ndim != 2
        or r1.shape[1] != 3
        or any(value.shape != ring_shape for value in values[3:])
    ):
        raise ValueError("material ring corners and velocities must share (m,3)")

    def line_derivative(
        origin: torch.Tensor,
        destination: torch.Tensor,
        origin_velocity: torch.Tensor,
        destination_velocity: torch.Tensor,
    ) -> torch.Tensor:
        a = points.unsqueeze(1) - origin.unsqueeze(0)
        b = points.unsqueeze(1) - destination.unsqueeze(0)
        da = point_velocities.unsqueeze(1) - origin_velocity.unsqueeze(0)
        db = point_velocities.unsqueeze(1) - destination_velocity.unsqueeze(0)
        cross = torch.linalg.cross(a, b, dim=2)
        cross_rate = torch.linalg.cross(da, b, dim=2) + torch.linalg.cross(
            a, db, dim=2
        )
        cross_norm_sq = torch.sum(cross * cross, dim=2)
        a_norm = torch.linalg.vector_norm(a, dim=2)
        b_norm = torch.linalg.vector_norm(b, dim=2)
        singular = (cross_norm_sq <= 0.0) | (a_norm <= 0.0) | (b_norm <= 0.0)
        if bool(torch.any(singular).item()):
            raise FloatingPointError(
                "material ring derivative encountered a singular target/leg"
            )
        unit_difference = a / a_norm.unsqueeze(2) - b / b_norm.unsqueeze(2)
        unit_difference_rate = (
            da / a_norm.unsqueeze(2)
            - a
            * (torch.sum(da * a, dim=2) / a_norm**3).unsqueeze(2)
            - db / b_norm.unsqueeze(2)
            + b
            * (torch.sum(db * b, dim=2) / b_norm**3).unsqueeze(2)
        )
        edge = a - b
        edge_rate = da - db
        scalar = torch.sum(edge * unit_difference, dim=2)
        scalar_rate = torch.sum(edge_rate * unit_difference, dim=2) + torch.sum(
            edge * unit_difference_rate, dim=2
        )
        cross_fraction_rate = cross_rate / cross_norm_sq.unsqueeze(2) - 2.0 * (
            cross
            * (
                torch.sum(cross_rate * cross, dim=2) / cross_norm_sq**2
            ).unsqueeze(2)
        )
        return cross_fraction_rate * scalar.unsqueeze(2) + (
            cross / cross_norm_sq.unsqueeze(2)
        ) * scalar_rate.unsqueeze(2)

    derivative = -(
        line_derivative(r1, r4, v1, v4)
        + line_derivative(r2, r1, v2, v1)
        + line_derivative(r3, r2, v3, v2)
        + line_derivative(r4, r3, v4, v3)
    ) / _FOUR_PI
    if not bool(torch.isfinite(derivative).all().item()):
        raise FloatingPointError("material ring derivative is non-finite")
    return derivative


def _ptera_ring_velocity_expanded(
    points: torch.Tensor,
    rings: torch.Tensor,
    core_radii: torch.Tensor,
    nu: float,
) -> torch.Tensor:
    """Evaluate readable ``[FL, FR, BR, BL]`` rings with Ptera's CUDA kernel."""

    strengths = torch.ones(rings.shape[0], dtype=torch.float64, device=points.device)
    ages = torch.zeros_like(strengths)
    return _ring_velocity_expanded(
        points,
        rings[:, 2],
        rings[:, 1],
        rings[:, 0],
        rings[:, 3],
        strengths,
        core_radii,
        ages,
        float(nu),
    )


class CudaJointLEVTEVSolver(JointLEVTEVSolver):
    """V5M chassis whose authorised aerodynamic modes execute on CUDA."""

    def __init__(
        self,
        unsteady_problem: Any,
        cfg: JointConfig | None = None,
        *,
        device: str = "cuda:0",
    ) -> None:
        selected = cfg or JointConfig(enable_lev=False)
        if unsteady_problem.only_final_results:
            raise ValueError(
                "CUDA attached chassis requires only_final_results=False; "
                "host final-load aggregation is forbidden"
            )
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is mandatory; no CPU numerical fallback is allowed"
            )
        self.cuda_device = torch.device(device)
        if self.cuda_device.type != "cuda":
            raise ValueError("CudaJointLEVTEVSolver requires a CUDA device")
        torch.empty(1, dtype=torch.float64, device=self.cuda_device)
        super().__init__(unsteady_problem, selected)
        self.lev_pf = CudaParticleField(
            capacity=selected.particle_capacity,
            device=self.cuda_device,
        )
        self._last_impulse_cuda: torch.Tensor | None = None
        self._last_impulse_strip_cuda: torch.Tensor | None = None
        self._cuda_lev_history_total = torch.zeros(
            (), device=self.cuda_device, dtype=torch.float64
        )
        self._cuda_tev_history_total = torch.zeros(
            (), device=self.cuda_device, dtype=torch.float64
        )
        if len(self.steady_problems[0].airplanes) != 1:
            raise ValueError("CUDA attached chassis currently supports one airplane")
        if len(self.steady_problems[0].airplanes[0].wings) != 1:
            raise ValueError("CUDA attached chassis currently supports one wing")
        self.cuda_numerical_contract = "torch-cuda-float64-no-cpu-fallback-v1"
        # FLUX-V5M owns one canonical scientific frame.  Ptera's GP arrays are
        # used only as object-container storage and are numerically identical
        # to the V5M/author frame: +x chord/downstream, +y span and +z oriented
        # surface normal.  Ptera's GP-to-W matrix is a presentation convention
        # and must never enter Q16 interpolation, wake history or load transfer.
        self.v5m_scientific_frame_schema = (
            "flux-v5m-author-x-chord-y-span-z-normal-v1"
        )
        self._v5m_gp_to_scientific_rotation_cuda = torch.eye(
            3, device=self.cuda_device, dtype=torch.float64
        )
        self._v5m_gp_to_scientific_translation_cuda = torch.zeros(
            3, device=self.cuda_device, dtype=torch.float64
        )
        self._q16_cuda_wake_vertex_velocity_grids_gp: list[
            torch.Tensor | None
        ] = [None] * self.num_steps
        self.cuda_counters = {
            "aic": 0,
            "wake": 0,
            "solve": 0,
            "velocity": 0,
            "loads": 0,
            "ledger": 0,
            "wake_convection": 0,
            "particle_velocity": 0,
            "particle_advance": 0,
            "particle_shed": 0,
            "impulse": 0,
            "dvm_source_steps": 0,
            "dvm_ribbon_shed": 0,
            "dvm_frontier_advance": 0,
        }
        if selected.separated_source not in {"hirato_ring", "dvm_node_ribbon"}:
            raise ValueError(
                f"unsupported separated_source {selected.separated_source!r}"
            )
        if selected.separated_source == "dvm_node_ribbon":
            if selected.enable_lev is not True or selected.joint_tev is not True:
                raise ValueError(
                    "dvm_node_ribbon requires separated LEV and joint Ptera TEV"
                )
            if selected.lev_start_step != 0:
                raise ValueError("dvm_node_ribbon must be active from step zero")
        if selected.bound_core_radius_chord is not None:
            core = float(selected.bound_core_radius_chord)
            if not math.isfinite(core) or core < 0.0:
                raise ValueError(
                    "bound_core_radius_chord must be finite and non-negative"
                )
        if selected.q16_added_mass_column_scope not in {
            "global",
            "author_element_local",
            "author_aerodynamic_element_projection",
        }:
            raise ValueError("unsupported q16_added_mass_column_scope")

    def _v5m_gp_points_to_scientific_cuda(
        self, points: torch.Tensor
    ) -> torch.Tensor:
        """Convert Ptera-container points to the canonical V5M frame."""

        values = _cuda_tensor(points, self.cuda_device)
        return (
            torch.einsum(
                "ij,nj->ni", self._v5m_gp_to_scientific_rotation_cuda, values
            )
            + self._v5m_gp_to_scientific_translation_cuda
        )

    def _v5m_scientific_points_to_gp_cuda(
        self, points: torch.Tensor
    ) -> torch.Tensor:
        """Install canonical V5M points into the Ptera GP container."""

        values = _cuda_tensor(points, self.cuda_device)
        return torch.einsum(
            "ij,nj->ni",
            self._v5m_gp_to_scientific_rotation_cuda.transpose(0, 1),
            values - self._v5m_gp_to_scientific_translation_cuda,
        )

    def _v5m_gp_vectors_to_scientific_cuda(
        self, vectors: torch.Tensor
    ) -> torch.Tensor:
        values = _cuda_tensor(vectors, self.cuda_device)
        return torch.einsum(
            "ij,nj->ni", self._v5m_gp_to_scientific_rotation_cuda, values
        )

    def _v5m_scientific_vectors_to_gp_cuda(
        self, vectors: torch.Tensor
    ) -> torch.Tensor:
        values = _cuda_tensor(vectors, self.cuda_device)
        return torch.einsum(
            "ij,nj->ni",
            self._v5m_gp_to_scientific_rotation_cuda.transpose(0, 1),
            values,
        )

    def _v5m_author_aic_cuda(self) -> torch.Tensor:
        """Return the author-oriented AIC from Ptera's opposite ring order."""

        aic = -self._cuda_aic
        if not bool(torch.isfinite(aic).all().item()):
            raise FloatingPointError("V5M author AIC is non-finite")
        if bool(torch.any(torch.diagonal(aic) >= 0.0).item()):
            raise GateError("V5M author AIC diagonal orientation drift")
        return aic

    def _require_v5m_scientific_surface_contract(self) -> None:
        """Fail closed when container presentation contaminates V5M science."""

        panels, span_count, chord_count = self._panel_grid()
        d = self.cuda_device
        v_inf_gp = _cuda_tensor(self.current_operating_point.vInf_GP1__E, d)
        v_inf = self._v5m_gp_vectors_to_scientific_cuda(
            v_inf_gp.unsqueeze(0)
        )[0]
        speed = torch.linalg.vector_norm(v_inf)
        if not bool(torch.isfinite(speed).item()) or bool((speed <= 0.0).item()):
            raise GateError("V5M scientific freestream is invalid")
        front_left = _cuda_tensor(
            [panels[0, span].Flpp_GP1_CgP1 for span in range(span_count)], d
        )
        front_right = _cuda_tensor(
            [panels[0, span].Frpp_GP1_CgP1 for span in range(span_count)], d
        )
        back_left = _cuda_tensor(
            [
                panels[chord_count - 1, span].Blpp_GP1_CgP1
                for span in range(span_count)
            ],
            d,
        )
        back_right = _cuda_tensor(
            [
                panels[chord_count - 1, span].Brpp_GP1_CgP1
                for span in range(span_count)
            ],
            d,
        )
        front_left_s = self._v5m_gp_points_to_scientific_cuda(front_left)
        front_right_s = self._v5m_gp_points_to_scientific_cuda(front_right)
        back_left_s = self._v5m_gp_points_to_scientific_cuda(back_left)
        back_right_s = self._v5m_gp_points_to_scientific_cuda(back_right)
        chord = 0.5 * (
            back_left_s + back_right_s - front_left_s - front_right_s
        )
        chord_flow = torch.sum(chord * v_inf.unsqueeze(0), dim=1)
        chord_scale = torch.linalg.vector_norm(chord, dim=1) * speed
        if bool(torch.any(chord_flow <= 4096.0 * _EPS * chord_scale).item()):
            raise GateError("V5M chord is not aligned with downstream freestream")
        span = 0.5 * (
            front_right_s + back_right_s - front_left_s - back_left_s
        )
        oriented_normal = torch.linalg.cross(chord, span, dim=1)
        normal_gp = _cuda_tensor(
            [panels[0, span_index].unitNormal_GP1 for span_index in range(span_count)],
            d,
        )
        normal_s = self._v5m_gp_vectors_to_scientific_cuda(normal_gp)
        orientation = torch.sum(oriented_normal * normal_s, dim=1)
        orientation_scale = torch.linalg.vector_norm(oriented_normal, dim=1)
        if bool(
            torch.any(orientation <= 4096.0 * _EPS * orientation_scale).item()
        ):
            raise GateError("V5M panel normal orientation drift")
        self._v5m_scientific_surface_contract_step = self._current_step

    def _require_v5m_downstream_wake_contract(self, grid_gp: torch.Tensor) -> None:
        """Require material wake rows to remain ordered downstream."""

        if (
            type(grid_gp) is not torch.Tensor
            or grid_gp.device != self.cuda_device
            or grid_gp.dtype is not torch.float64
            or grid_gp.ndim != 3
            or grid_gp.shape[0] < 2
            or grid_gp.shape[2] != 3
        ):
            raise GateError("V5M wake grid has invalid CUDA topology")
        grid = self._v5m_gp_points_to_scientific_cuda(
            grid_gp.reshape(-1, 3)
        ).reshape_as(grid_gp)
        v_inf = self._v5m_gp_vectors_to_scientific_cuda(
            _cuda_tensor(
                self.current_operating_point.vInf_GP1__E, self.cuda_device
            ).unsqueeze(0)
        )[0]
        increments = grid[1:] - grid[:-1]
        downstream = torch.sum(increments * v_inf, dim=2)
        scale = torch.linalg.vector_norm(increments, dim=2) * torch.linalg.vector_norm(
            v_inf
        )
        if bool(torch.any(downstream <= 4096.0 * _EPS * scale).item()):
            raise GateError("V5M free-wake row is not downstream ordered")

    def _assert_supported_step(self) -> None:
        if self.current_operating_point.surfaceReflect_T_act_GP1_CgP1 is not None:
            raise ValueError("CUDA attached chassis does not support image surfaces")

    def _bound_rings(self) -> tuple[torch.Tensor, ...]:
        d = self.cuda_device
        br = _cuda_tensor(self.stackBrbrvp_GP1_CgP1, d)
        fr = _cuda_tensor(self.stackFrbrvp_GP1_CgP1, d)
        fl = _cuda_tensor(self.stackFlbrvp_GP1_CgP1, d)
        bl = _cuda_tensor(self.stackBlbrvp_GP1_CgP1, d)
        if self.jcfg.full_trailing_edge_bound_ring:
            _, span_count, chord_count = self._panel_grid()
            trailing = (
                (chord_count - 1) * span_count
                + torch.arange(span_count, device=d, dtype=torch.int64)
            )
            br = br.clone()
            bl = bl.clone()
            br[trailing] = fr[trailing] + 2.0 * (
                br[trailing] - fr[trailing]
            )
            bl[trailing] = fl[trailing] + 2.0 * (
                bl[trailing] - fl[trailing]
            )
        return br, fr, fl, bl

    def _bound_core_radii(self) -> torch.Tensor:
        native = _cuda_tensor(self._currentStackBoundRc0s, self.cuda_device)
        override = self.jcfg.bound_core_radius_chord
        if override is None:
            return native
        mean_chord = self.current_airplanes[0].wings[0].standard_mean_chord
        if mean_chord is None or not math.isfinite(float(mean_chord)):
            raise RuntimeError("bound-core override requires a finite mean chord")
        return torch.full_like(native, float(override) * float(mean_chord))

    def _wake_rings(self) -> tuple[torch.Tensor, ...]:
        d = self.cuda_device
        return (
            _cuda_tensor(self._currentStackBrwrvp_GP1_CgP1, d),
            _cuda_tensor(self._currentStackFrwrvp_GP1_CgP1, d),
            _cuda_tensor(self._currentStackFlwrvp_GP1_CgP1, d),
            _cuda_tensor(self._currentStackBlwrvp_GP1_CgP1, d),
        )

    def _bound_velocity(self, points: torch.Tensor) -> torch.Tensor:
        br, fr, fl, bl = self._bound_rings()
        n = br.shape[0]
        return _chunked_ring_velocity_collapsed(
            points,
            br,
            fr,
            fl,
            bl,
            (
                self._cuda_bound_strengths
                if hasattr(self, "_cuda_bound_strengths")
                else _cuda_tensor(
                    self._current_bound_vortex_strengths, self.cuda_device
                )
            ),
            self._bound_core_radii(),
            torch.zeros(n, dtype=torch.float64, device=self.cuda_device),
            float(self.current_operating_point.nu),
        )

    def _wake_velocity(self, points: torch.Tensor) -> torch.Tensor:
        if self._current_step == 0 or len(self._current_wake_vortex_strengths) == 0:
            return torch.zeros_like(points)
        br, fr, fl, bl = self._wake_rings()
        span_count = self.current_airplanes[0].wings[0].num_spanwise_panels
        if span_count is None or br.shape[0] % span_count:
            raise RuntimeError("wake topology cannot be mapped to CUDA ages")
        row_count = br.shape[0] // span_count
        ages = torch.repeat_interleave(
            torch.arange(row_count, device=self.cuda_device, dtype=torch.float64)
            * torch.as_tensor(
                self.delta_time, device=self.cuda_device, dtype=torch.float64
            ),
            span_count,
        )
        return _chunked_ring_velocity_collapsed(
            points,
            br,
            fr,
            fl,
            bl,
            _cuda_tensor(self._current_wake_vortex_strengths, self.cuda_device),
            _cuda_tensor(self._currentStackWakeRc0s, self.cuda_device),
            ages,
            float(self.current_operating_point.nu),
        )

    def _author_wake_motion_pressure_cuda(self) -> torch.Tensor:
        """Return Yamano ``rho*Mf2_vec1`` on the current panel ordering."""

        d = self.cuda_device
        if getattr(self, "_v5m_scientific_surface_contract_step", None) != (
            self._current_step
        ):
            raise GateError(
                "V5M scientific frame must pass before author wake pressure"
            )
        points = _cuda_tensor(self.stackCpp_GP1_CgP1, d)
        panel_count = points.shape[0]
        if self._current_step == 0 or len(self._current_wake_vortex_strengths) == 0:
            return torch.zeros(panel_count, device=d, dtype=torch.float64)
        if getattr(self, "_v5m_downstream_wake_contract_step", None) != (
            self._current_step
        ):
            raise GateError(
                "V5M downstream wake gate must pass before author wake pressure"
            )
        wake_vertex_velocity = self._q16_cuda_wake_vertex_velocity_grids_gp[
            self._current_step
        ]
        if type(wake_vertex_velocity) is not torch.Tensor:
            raise RuntimeError("current material wake has no CUDA vertex velocity")
        if (
            wake_vertex_velocity.device != d
            or wake_vertex_velocity.dtype is not torch.float64
            or wake_vertex_velocity.ndim != 3
            or wake_vertex_velocity.shape[2] != 3
        ):
            raise ValueError("material wake vertex velocity topology/device drift")
        _, span_count, _ = self._panel_grid()
        ring_count = len(self._current_wake_vortex_strengths)
        if ring_count % span_count:
            raise RuntimeError("material wake ring topology is not rectangular")
        row_count = ring_count // span_count
        expected_shape = (row_count + 1, span_count + 1, 3)
        if tuple(wake_vertex_velocity.shape) != expected_shape:
            raise RuntimeError("material wake vertex velocity shape drift")
        br, fr, fl, bl = self._wake_rings()
        velocity_fr = wake_vertex_velocity[:-1, 1:].reshape(-1, 3)
        velocity_br = wake_vertex_velocity[1:, 1:].reshape(-1, 3)
        velocity_bl = wake_vertex_velocity[1:, :-1].reshape(-1, 3)
        velocity_fl = wake_vertex_velocity[:-1, :-1].reshape(-1, 3)
        if self._current_step < 1:
            point_velocity = torch.zeros_like(points)
        else:
            point_velocity = (
                points - _cuda_tensor(self._stackLastCpp_GP1_CgP1, d)
            ) / float(self.delta_time)
        # MATLAB corner order is (front-right, back-right, back-left,
        # front-left).  Its four directed legs are therefore identical to
        # Ptera's br->fr->fl->bl->br ring orientation.
        influence_rate = _material_ring_velocity_derivative_expanded(
            points,
            point_velocity,
            fr,
            br,
            bl,
            fl,
            velocity_fr,
            velocity_br,
            velocity_bl,
            velocity_fl,
        )
        # Ptera stores the same physical rings with the orientation opposite
        # to Yamano's (front-right -> back-right -> back-left -> front-left)
        # convention.  Convert both the AIC and circulation explicitly instead
        # of relying on two cancelling undocumented signs.
        strengths = -_cuda_tensor(self._current_wake_vortex_strengths, d)
        wake_velocity_rate = torch.sum(
            influence_rate * strengths.unsqueeze(0).unsqueeze(2), dim=1
        )
        normals = _cuda_tensor(self.stackUnitNormals_GP1, d)
        normal_rate = torch.sum(wake_velocity_rate * normals, dim=1)
        mf2 = torch.linalg.solve(self._v5m_author_aic_cuda(), -normal_rate)
        pressure = float(self.current_operating_point.rho) * mf2
        if not bool(torch.isfinite(pressure).all().item()):
            raise FloatingPointError("author wake-motion pressure is non-finite")
        return pressure

    def _populate_next_airplanes_wake_vortices(self) -> None:
        """Copy wake objects without executing the parent's host age arithmetic.

        Wake age is a derived CUDA row coordinate in :meth:`_wake_velocity`.
        The Ptera objects remain orchestration containers only; their ``age``
        member is deliberately not used or advanced by this backend.
        """
        if self._current_step >= self.num_steps - 1:
            return
        next_airplanes = self.steady_problems[self._current_step + 1].airplanes
        for airplane_id, next_airplane in enumerate(next_airplanes):
            for wing_id, this_wing in enumerate(
                self.current_airplanes[airplane_id].wings
            ):
                next_wing = next_airplane.wings[wing_id]
                next_points = next_wing.gridWrvp_GP1_CgP1
                if next_points is None:
                    raise RuntimeError("missing next CUDA wake-point grid")
                chordwise_points, spanwise_points = next_points.shape[:2]
                current_vortices = this_wing.wake_ring_vortices
                if current_vortices is None:
                    raise RuntimeError("missing current CUDA wake-vortex grid")
                if (
                    self._max_wake_rows is not None
                    and current_vortices.shape[0] >= self._max_wake_rows
                ):
                    current_vortices = current_vortices[: self._max_wake_rows - 1]
                next_wing.wake_ring_vortices = np.vstack(
                    (
                        np.empty((1, spanwise_points - 1), dtype=object),
                        current_vortices,
                    )
                )
                for chordwise_id in range(chordwise_points - 1):
                    for spanwise_id in range(spanwise_points - 1):
                        front_left = next_points[chordwise_id, spanwise_id]
                        front_right = next_points[chordwise_id, spanwise_id + 1]
                        back_left = next_points[chordwise_id + 1, spanwise_id]
                        back_right = next_points[chordwise_id + 1, spanwise_id + 1]
                        if chordwise_id == 0:
                            panels = this_wing.panels
                            if panels is None:
                                raise RuntimeError("missing current CUDA panel grid")
                            ring = panels[
                                this_wing.num_chordwise_panels - 1, spanwise_id
                            ].ring_vortex
                            if ring is None:
                                raise RuntimeError("missing bound ring vortex")
                            if self.jcfg.joint_tev and self._tev_solved is not None:
                                strength = float(self._tev_solved[spanwise_id].item())
                            else:
                                strength = ring.strength
                        else:
                            old_ring = next_wing.wake_ring_vortices[
                                chordwise_id, spanwise_id
                            ]
                            if not isinstance(
                                old_ring, _vortices.ring_vortex.RingVortex
                            ):
                                raise RuntimeError("invalid wake ring-vortex object")
                            strength = old_ring.strength
                        next_wing.wake_ring_vortices[chordwise_id, spanwise_id] = (
                            _vortices.ring_vortex.RingVortex(
                                Flrvp_GP1_CgP1=front_left,
                                Frrvp_GP1_CgP1=front_right,
                                Blrvp_GP1_CgP1=back_left,
                                Brrvp_GP1_CgP1=back_right,
                                strength=strength,
                            )
                        )

    def _finalize_loads(self) -> None:
        """Skip the unused parent NumPy mean/RMS reduction.

        Per-step loads are already produced by the CUDA backend and are the only
        loads consumed by the validation runners. ``only_final_results=True`` is
        rejected at construction, so silently substituting a host reduction is
        impossible.
        """
        return None

    def _calculate_wing_wing_influences(self) -> None:
        self._assert_supported_step()
        self._require_v5m_scientific_surface_contract()
        d = self.cuda_device
        points = _cuda_tensor(self.stackCpp_GP1_CgP1, d)
        br, fr, fl, bl = self._bound_rings()
        n = br.shape[0]
        expanded = _bound_ring_velocity_expanded(
            points,
            br,
            fr,
            fl,
            bl,
            torch.ones(n, dtype=torch.float64, device=d),
            self._bound_core_radii(),
            torch.zeros(n, dtype=torch.float64, device=d),
            float(self.current_operating_point.nu),
        )
        normals = _cuda_tensor(self.stackUnitNormals_GP1, d)
        aic = torch.sum(expanded * normals.unsqueeze(1), dim=2)
        self._currentGridWingWingInfluences__E = aic.detach().cpu().numpy()
        self._cuda_aic = aic
        self._v5m_author_aic_cuda()
        self.cuda_counters["aic"] += 1

    def _calculate_freestream_wing_influences(self) -> None:
        d = self.cuda_device
        normals = _cuda_tensor(self.stackUnitNormals_GP1, d)
        v_inf = _cuda_tensor(self._currentVInf_GP1__E, d)
        if self._current_step < 1:
            movement = torch.zeros_like(normals)
        else:
            movement = -(
                _cuda_tensor(self.stackCpp_GP1_CgP1, d)
                - _cuda_tensor(self._stackLastCpp_GP1_CgP1, d)
            ) / float(self.delta_time)
        influence = torch.sum(normals * (v_inf.unsqueeze(0) + movement), dim=1)
        self._cuda_freestream = influence
        self._currentStackFreestreamWingInfluences__E = influence.detach().cpu().numpy()

    def _calculate_wake_wing_influences(self) -> None:
        self._assert_supported_step()
        # A Hirato pseudovortex is a current-step bound-sheet extension.  Its
        # circulation is transferred to the next Ptera TE row by Eq. 9, so a
        # previous-step pseudovortex must never re-enter the historical RHS.
        self._current_pseudovortex_rings_cuda = None
        self._current_pseudovortex_strengths_cuda = None
        self._current_pseudovortex_core_cuda = None
        le_now, te_now = self._station_le_te_points()
        if hasattr(self, "_le_points_now"):
            self._le_points_prev = self._le_points_now
            self._te_points_prev = self._te_points_now
        self._le_points_now = le_now
        self._te_points_now = te_now
        d = self.cuda_device
        points = _cuda_tensor(self.stackCpp_GP1_CgP1, d)
        velocity = self._wake_velocity(points)
        if self.lev_pf.n:
            self._advance_dvm_frontier_cuda()
            self.lev_pf.advance_wrk3(
                float(self.delta_time),
                self._particle_external_velocity_cuda,
            )
            velocity = velocity + self.lev_pf.velocity_at_cuda(points)
            self.cuda_counters["particle_advance"] += 1
            self.cuda_counters["particle_velocity"] += 1
        normals = _cuda_tensor(self.stackUnitNormals_GP1, d)
        influence = torch.sum(velocity * normals, dim=1)
        self._cuda_wake = influence
        self._currentStackWakeWingInfluences__E = influence.detach().cpu().numpy()
        self.cuda_counters["wake"] += 1

    def _dvm_node_ribbon_enabled(self) -> bool:
        return self.jcfg.separated_source == "dvm_node_ribbon"

    def _ensure_dvm_source_bank_cuda(
        self,
        leading_edge: torch.Tensor,
        trailing_edge: torch.Tensor,
    ) -> None:
        """Create the cell/node source bank from the first committed geometry."""
        if hasattr(self, "dvm_source_bank"):
            return
        if not self._dvm_node_ribbon_enabled():
            raise RuntimeError("DVM source bank requested outside its explicit mode")
        _, span_count, _ = self._panel_grid()
        if leading_edge.shape != (span_count + 1, 3) or trailing_edge.shape != (
            span_count + 1,
            3,
        ):
            raise ValueError("DVM source bank requires one shared LE/TE node row")
        node_chord = torch.linalg.vector_norm(trailing_edge - leading_edge, dim=1)
        cell_chord = 0.5 * (node_chord[:-1] + node_chord[1:])
        speed = torch.linalg.vector_norm(
            _cuda_tensor(self.current_operating_point.vInf_GP1__E, self.cuda_device)
        )
        if bool((speed <= 0.0).item()) or bool(
            torch.any(torch.cat((cell_chord, node_chord)) <= 0.0).item()
        ):
            raise ValueError("DVM source reference speed/chord must be positive")
        lane_chord = torch.cat((cell_chord, node_chord))
        convective_dt = float(self.delta_time) * speed / lane_chord
        self.dvm_source_bank = CudaLDVMSourceBank(
            batch_size=2 * span_count + 1,
            ndiv=int(self.jcfg.dvm_ndiv),
            naterm=int(self.jcfg.dvm_naterm),
            delta_time_convective=convective_dt,
            lesp_crit=float(self.jcfg.lesp_crit),
            pivot_fraction_chord=float(self.jcfg.dvm_pivot_fraction_chord),
            core_radius_chord=float(self.jcfg.dvm_core_radius_chord),
            max_wake=int(self.jcfg.dvm_max_wake),
            source_parity=True,
            device=str(self.cuda_device),
        )
        self._cuda_dvm_reference_speed = speed.clone()
        self._cuda_dvm_reference_cell_chord = cell_chord.clone()
        self._cuda_dvm_reference_node_chord = node_chord.clone()
        self._cuda_dvm_alpha_previous = None
        self._cuda_dvm_frontier_nodes = leading_edge.clone()
        self._cuda_dvm_frontier_active = torch.zeros(
            span_count + 1, device=self.cuda_device, dtype=torch.bool
        )
        self._cuda_dvm_frontier_ever = torch.zeros_like(self._cuda_dvm_frontier_active)
        self._cuda_dvm_advected_frontier = leading_edge.clone()
        self._cuda_dvm_last_source = None

    def _advance_dvm_frontier_cuda(self) -> None:
        """Convect the live node frontier before the next source handoff."""
        if not self._dvm_node_ribbon_enabled() or not hasattr(
            self, "_cuda_dvm_frontier_active"
        ):
            return
        active = self._cuda_dvm_frontier_active
        if not bool(torch.any(active).item()):
            return
        original = self._cuda_dvm_frontier_nodes.clone()

        def velocity(position: torch.Tensor) -> torch.Tensor:
            result = self._particle_external_velocity_cuda(position)
            if self.lev_pf.n:
                result = result + self.lev_pf.velocity_at_cuda(position)
            return result

        dt = float(self.delta_time)
        u1 = velocity(original)
        u2 = velocity(original + 0.5 * dt * u1)
        u3 = velocity(original + dt * (-u1 + 2.0 * u2))
        advanced = original + (dt / 6.0) * (u1 + 4.0 * u2 + u3)
        if not bool(torch.isfinite(advanced).all().item()):
            raise FloatingPointError("DVM node frontier advance became non-finite")
        self._cuda_dvm_advected_frontier = torch.where(
            active[:, None], advanced, original
        )
        self.cuda_counters["dvm_frontier_advance"] += 1

    def _dvm_source_kinematics_cuda(
        self,
        leading_edge: torch.Tensor,
        trailing_edge: torch.Tensor,
        panels: Any,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Return batched cell/node kinematics and node-local section axes."""
        _, span_count, chord_count = self._panel_grid()
        cell_chord_vector = 0.5 * (
            trailing_edge[:-1]
            + trailing_edge[1:]
            - leading_edge[:-1]
            - leading_edge[1:]
        )
        cell_x = cell_chord_vector / torch.linalg.vector_norm(
            cell_chord_vector, dim=1, keepdim=True
        )
        cell_normal = _cuda_tensor(
            [
                np.mean(
                    [
                        panels[chord, span].unitNormal_GP1
                        for chord in range(chord_count)
                    ],
                    axis=0,
                )
                for span in range(span_count)
            ],
            self.cuda_device,
        )
        cell_normal = cell_normal / torch.linalg.vector_norm(
            cell_normal, dim=1, keepdim=True
        )
        node_x_vector = trailing_edge - leading_edge
        node_x = node_x_vector / torch.linalg.vector_norm(
            node_x_vector, dim=1, keepdim=True
        )
        node_normal = torch.empty(
            (span_count + 1, 3), device=self.cuda_device, dtype=torch.float64
        )
        node_normal[0] = cell_normal[0]
        node_normal[-1] = cell_normal[-1]
        if span_count > 1:
            node_normal[1:-1] = cell_normal[:-1] + cell_normal[1:]
        node_normal = node_normal / torch.linalg.vector_norm(
            node_normal, dim=1, keepdim=True
        )

        v_inf = _cuda_tensor(self.current_operating_point.vInf_GP1__E, self.cuda_device)
        cell_alpha = torch.atan2(
            torch.sum(cell_normal * v_inf, dim=1),
            torch.sum(cell_x * v_inf, dim=1),
        )
        node_alpha = torch.atan2(
            torch.sum(node_normal * v_inf, dim=1),
            torch.sum(node_x * v_inf, dim=1),
        )
        alpha = torch.cat((cell_alpha, node_alpha))
        if self._cuda_dvm_alpha_previous is None:
            alpha_rate = torch.zeros_like(alpha)
        else:
            delta = (
                torch.remainder(
                    alpha - self._cuda_dvm_alpha_previous + math.pi, 2.0 * math.pi
                )
                - math.pi
            )
            alpha_rate = delta / self.dvm_source_bank.dt

        if hasattr(self, "_le_points_prev"):
            previous = _cuda_tensor(self._le_points_prev, self.cuda_device)
            node_velocity = (leading_edge - previous) / float(self.delta_time)
            node_heave = (
                torch.sum(node_velocity * node_normal, dim=1)
                / self._cuda_dvm_reference_speed
            )
        else:
            node_heave = torch.zeros(
                span_count + 1, device=self.cuda_device, dtype=torch.float64
            )
        cell_heave = 0.5 * (node_heave[:-1] + node_heave[1:])
        heave = torch.cat((cell_heave, node_heave))
        return alpha, alpha_rate, heave, node_x, node_normal

    def _evaluate_dvm_node_ribbon_cuda(
        self,
        *,
        leading_edge: torch.Tensor,
        trailing_edge: torch.Tensor,
        panels: Any,
        collocation: torch.Tensor,
        normals: torch.Tensor,
        rhs_without_newborn: torch.Tensor,
        ptera_lesp_scale: torch.Tensor,
        ptera_separated: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Advance the DVM source and deposit one current-time node ribbon."""
        self._ensure_dvm_source_bank_cuda(leading_edge, trailing_edge)
        _, span_count, chord_count = self._panel_grid()
        alpha, alpha_rate, heave, node_x, node_normal = (
            self._dvm_source_kinematics_cuda(leading_edge, trailing_edge, panels)
        )
        result = self.dvm_source_bank.step(
            alpha,
            alpha_rate,
            heave,
            node_topology_from_cell_count=span_count,
        )
        self._cuda_dvm_alpha_previous = alpha.clone()
        cell_active = result["shed_lev"][:span_count]
        if (
            ptera_separated.shape != (span_count,)
            or ptera_separated.dtype is not torch.bool
        ):
            raise ValueError("Ptera separated mask must be one Boolean per strip")
        pin_active = _dvm_ptera_pin_active_mask(cell_active, ptera_separated)
        # A cell lane owns the physical release condition and circulation of
        # one three-dimensional ribbon cell.  Node lanes are kinematic sampling
        # lanes for shared birth coordinates; their raw sectional threshold is
        # diagnostic, while the source bank projects their material history onto
        # the adjacent-cell union before committing the step.
        node_source_active = result["raw_shed_lev"][span_count:]
        node_active = result["shed_lev"][span_count:]
        adjacent_active = torch.empty_like(node_source_active)
        adjacent_active[0] = cell_active[0]
        adjacent_active[-1] = cell_active[-1]
        if span_count > 1:
            adjacent_active[1:-1] = cell_active[:-1] | cell_active[1:]
        if not torch.equal(node_active, adjacent_active):
            raise GateError("DVM node topology differs from adjacent cell union")
        self._cuda_dvm_last_node_source_active = node_source_active.clone()
        self._cuda_dvm_last_node_topology_active = node_active.clone()
        self._cuda_dvm_last_node_activity_mismatch_count = torch.count_nonzero(
            node_source_active != node_active
        )

        dimension_scale = (
            self._cuda_dvm_reference_speed * self._cuda_dvm_reference_cell_chord
        )
        gamma_lev = result["gamma_lev_new"][:span_count] * dimension_scale
        birth_2d = result["lev_birth_position"][span_count:]
        anchor_2d = result["lev_edge_position"][span_count:]
        displacement_2d = birth_2d - anchor_2d
        first_or_restart = node_active & (~self._cuda_dvm_frontier_active)
        continuous = node_active & self._cuda_dvm_frontier_active
        local_birth = leading_edge + self._cuda_dvm_reference_node_chord[:, None] * (
            displacement_2d[:, :1] * node_x + displacement_2d[:, 1:] * node_normal
        )
        continuous_birth = (
            leading_edge + (self._cuda_dvm_advected_frontier - leading_edge) / 3.0
        )
        frontier = leading_edge.clone()
        frontier = torch.where(first_or_restart[:, None], local_birth, frontier)
        frontier = torch.where(continuous[:, None], continuous_birth, frontier)
        if not bool(torch.isfinite(frontier).all().item()):
            raise FloatingPointError("DVM node-owned birth frontier is non-finite")

        particle_start = self.lev_pf.n
        mean_chord = float(torch.mean(self._cuda_dvm_reference_cell_chord).item())
        shed_count = self.lev_pf.add_connected_ribbon_particles(
            leading_edge,
            frontier,
            # The particle helper traverses the reverse of the frozen DVM
            # ribbon orientation, hence the one explicit sign conversion.
            -gamma_lev,
            smoothing_radius=float(self.jcfg.dvm_smoothing_radius_chord) * mean_chord,
            target_spacing=float(self.jcfg.dvm_target_spacing_chord) * mean_chord,
            birth_step=self._current_step,
            connector_source_strips=torch.clamp(
                torch.arange(
                    span_count + 1,
                    device=self.cuda_device,
                    dtype=torch.int64,
                )
                - 1,
                min=0,
                max=span_count - 1,
            ),
        )
        self._cuda_dvm_frontier_nodes = torch.where(
            node_active[:, None], frontier, self._cuda_dvm_frontier_nodes
        )
        self._cuda_dvm_frontier_active = node_active.clone()
        self._cuda_dvm_frontier_ever = self._cuda_dvm_frontier_ever | node_active
        self._cuda_dvm_last_source = {
            key: value.clone() if type(value) is torch.Tensor else value
            for key, value in result.items()
        }
        self.cuda_counters["dvm_source_steps"] += 1
        self.cuda_counters["dvm_ribbon_shed"] += shed_count
        self.cuda_counters["particle_shed"] += shed_count

        if shed_count:
            newborn_velocity = self.lev_pf.velocity_at_cuda(
                collocation,
                source_start=particle_start,
                source_stop=self.lev_pf.n,
            )
            newborn_influence = torch.sum(newborn_velocity * normals, dim=1)
            rhs = rhs_without_newborn - newborn_influence
            self._cuda_wake = self._cuda_wake + newborn_influence
            self._currentStackWakeWingInfluences__E = (
                self._cuda_wake.detach().cpu().numpy()
            )
            self.cuda_counters["particle_velocity"] += 1
        else:
            rhs = rhs_without_newborn
            newborn_influence = torch.zeros_like(rhs_without_newborn)
        self._cuda_dvm_last_newborn_normal_influence = newborn_influence.clone()
        if ptera_lesp_scale.shape != (span_count,) or bool(
            torch.any(ptera_lesp_scale <= 0.0).item()
        ):
            raise ValueError("Ptera LESP scale must be one positive value per strip")
        source_a0 = result["A0"][:span_count]
        target_le_gamma = -source_a0 * ptera_lesp_scale
        leading_indices = torch.arange(
            span_count, device=self.cuda_device, dtype=torch.int64
        )
        # A fixed DVM newborn circulation leaves no extra unknown with which to
        # enforce both the attached leading-edge Neumann row and the separated
        # suction limit.  On active strips the latter is the physical boundary
        # condition, so replace exactly those leading-edge rows by the DVM A0
        # pin.  A source-off event does not reattach a strip while Ptera's own
        # pre-solve suction still exceeds the limit.  Every genuinely attached
        # LE row and every non-LE row retain the original no-penetration equation.
        separated_aic = self._cuda_aic.clone()
        separated_rhs = rhs.clone()
        active_indices = leading_indices[pin_active]
        if active_indices.numel():
            separated_aic[active_indices] = 0.0
            separated_aic[active_indices, active_indices] = 1.0
            separated_rhs[active_indices] = target_le_gamma[pin_active]
        gamma_bound = torch.linalg.solve(separated_aic, separated_rhs)
        ptera_lesp_post = -gamma_bound[leading_indices] / ptera_lesp_scale
        neumann_residual = self._cuda_aic @ gamma_bound - rhs
        retained_rows = torch.ones_like(neumann_residual, dtype=torch.bool)
        retained_rows[active_indices] = False
        retained_residual = torch.max(torch.abs(neumann_residual[retained_rows]))
        retained_scale = torch.clamp(torch.max(torch.abs(rhs[retained_rows])), min=1.0)
        pin_residual = torch.zeros((), device=self.cuda_device, dtype=torch.float64)
        if active_indices.numel():
            pin_residual = torch.max(
                torch.abs(ptera_lesp_post[pin_active] - source_a0[pin_active])
            )
        tolerance = float(self.jcfg.gate_rtol)
        if bool((retained_residual > tolerance * retained_scale).item()):
            raise GateError(
                f"DVM/Ptera retained Neumann rows failed at step {self._current_step}"
            )
        if bool((pin_residual > float(self.jcfg.lesp_rtol)).item()):
            raise GateError(f"DVM/Ptera LESP row failed at step {self._current_step}")
        self._cuda_dvm_last_source_a0 = source_a0.clone()
        self._cuda_dvm_last_ptera_lesp = ptera_lesp_post.clone()
        self._cuda_dvm_last_ptera_pin_active = pin_active.clone()
        self._cuda_dvm_last_retained_neumann_residual = retained_residual.clone()
        self._cuda_dvm_last_lesp_pin_residual = pin_residual.clone()
        rear_indices = torch.arange(
            (chord_count - 1) * span_count,
            chord_count * span_count,
            device=self.cuda_device,
            dtype=torch.int64,
        )
        gamma_tev = gamma_bound[rear_indices] + gamma_lev
        return (
            gamma_bound,
            gamma_tev,
            gamma_lev,
            ptera_lesp_post,
            result["lesp_pre"][:span_count],
            cell_active,
        )

    def _joint_vortex_solve_cuda(
        self,
        gamma_pre: torch.Tensor,
        rhs: torch.Tensor,
        lesp: torch.Tensor,
        active: torch.Tensor,
        allowed: torch.Tensor,
        panels: Any,
        leading_edge: torch.Tensor,
        chords: torch.Tensor,
        reference_velocity: torch.Tensor,
        theta_first: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Solve the Hirato current-step LEV/pseudovortex system on CUDA.

        The current step solves ``[A B; H 0]`` for bound circulation and the
        material-LEV release ``q``.  ``B`` is the combined influence of the
        newborn LEV and Hirato pseudovortex.  Only after loads are evaluated is
        the next material TE row committed as ``Gamma_rear + q`` (Eq. 9).
        Treating that future TE row as a second current-step unknown created
        the observed TEV/LEV cancellation nullspace.
        """

        device = self.cuda_device
        _, span_count, chord_count = self._panel_grid()
        panel_count = self.num_panels
        v_inf = _cuda_tensor(self.current_operating_point.vInf_GP1__E, device)
        rear_right = _cuda_tensor(
            [
                panels[chord_count - 1, span].ring_vortex.Brrvp_GP1_CgP1
                for span in range(span_count)
            ],
            device,
        )
        rear_left = _cuda_tensor(
            [
                panels[chord_count - 1, span].ring_vortex.Blrvp_GP1_CgP1
                for span in range(span_count)
            ],
            device,
        )
        leading_normals = _cuda_tensor(
            [panels[0, span].unitNormal_GP1 for span in range(span_count)],
            device,
        )
        trailing_physical_left = _cuda_tensor(
            [panels[chord_count - 1, span].Blpp_GP1_CgP1 for span in range(span_count)],
            device,
        )
        trailing_physical_right = _cuda_tensor(
            [panels[chord_count - 1, span].Brpp_GP1_CgP1 for span in range(span_count)],
            device,
        )
        chord_vector = 0.5 * (
            trailing_physical_left
            + trailing_physical_right
            - leading_edge[:-1]
            - leading_edge[1:]
        )
        chord_tangent = chord_vector / torch.linalg.vector_norm(
            chord_vector, dim=1, keepdim=True
        )
        local_alpha = torch.atan2(-chord_tangent[:, 2], chord_tangent[:, 0])
        displacement_scale = (
            torch.linalg.vector_norm(v_inf)
            * lesp
            * float(self.delta_time)
            / math.sqrt(2.0)
        )
        displacement = (displacement_scale * torch.sin(local_alpha)).unsqueeze(
            1
        ) * chord_tangent + (displacement_scale * torch.cos(local_alpha)).unsqueeze(
            1
        ) * leading_normals
        lev_rings = torch.empty((span_count, 4, 3), device=device, dtype=torch.float64)
        # This is the Ptera-readable reversed newborn orientation used by the
        # native v5f material-sheet implementation.
        lev_rings[:, 0] = leading_edge[1:]
        lev_rings[:, 1] = leading_edge[:-1]
        lev_rings[:, 2] = leading_edge[:-1] + displacement
        lev_rings[:, 3] = leading_edge[1:] + displacement
        pseudovortex_rings = torch.empty_like(lev_rings)
        pseudovortex_rings[:, 0] = leading_edge[:-1]
        pseudovortex_rings[:, 1] = leading_edge[1:]
        pseudovortex_rings[:, 2] = rear_right
        pseudovortex_rings[:, 3] = rear_left

        collocation = _cuda_tensor(self.stackCpp_GP1_CgP1, device)
        normals = _cuda_tensor(self.stackUnitNormals_GP1, device)
        source_core = self._bound_core_radii()[:span_count]
        lev_velocity = _ptera_ring_velocity_expanded(
            collocation,
            lev_rings,
            source_core,
            float(self.current_operating_point.nu),
        )
        pseudovortex_velocity = _ptera_ring_velocity_expanded(
            collocation,
            pseudovortex_rings,
            source_core,
            float(self.current_operating_point.nu),
        )
        birth_influence = torch.sum(
            normals.unsqueeze(1) * (lev_velocity + pseudovortex_velocity), dim=2
        )

        condensed_count = panel_count + span_count
        base_matrix = torch.zeros(
            (condensed_count, condensed_count),
            device=device,
            dtype=torch.float64,
        )
        base_target = torch.zeros(condensed_count, device=device, dtype=torch.float64)
        base_matrix[:panel_count, :panel_count] = self._cuda_aic
        base_matrix[:panel_count, panel_count:] = birth_influence
        base_target[:panel_count] = rhs
        trailing_panel_indices = torch.empty(
            span_count, device=device, dtype=torch.int64
        )
        for span in range(span_count):
            te_index = (chord_count - 1) * span_count + span
            trailing_panel_indices[span] = te_index

        lesp_denominator = (
            chords * reference_velocity * (theta_first + torch.sin(theta_first))
        )
        pin_sign = torch.sign(lesp)
        active_work = active.clone()
        inactive_limit = float(self.jcfg.lesp_inactive_margin) * float(
            self.jcfg.lesp_crit
        )
        for active_iteration in range(span_count + 1):
            matrix = base_matrix.clone()
            target = base_target.clone()
            caps = (
                -pin_sign * float(self.jcfg.lesp_crit) * lesp_denominator / LESP_FACTOR
            )
            for span in range(span_count):
                constraint_row = panel_count + span
                if bool(active_work[span].item()):
                    matrix[constraint_row, span] = 1.0
                    target[constraint_row] = caps[span]
                else:
                    matrix[constraint_row, panel_count + span] = 1.0

            solution = torch.linalg.solve(matrix, target)
            if not bool(torch.isfinite(solution).all().item()):
                raise GateError(
                    f"G5 non-finite CUDA joint solution step {self._current_step}"
                )
            gamma_bound = solution[:panel_count]
            gamma_lev = solution[panel_count:]
            gamma_tev = gamma_bound[trailing_panel_indices] + gamma_lev
            solved_lesp = (
                -LESP_FACTOR
                * gamma_bound.reshape(chord_count, span_count)[0]
                / lesp_denominator
            )
            newly_active = (~active_work) & (torch.abs(solved_lesp) > inactive_limit)
            if not bool(allowed.item()) or not bool(torch.any(newly_active).item()):
                break
            if active_iteration == span_count:
                raise GateError(
                    f"G3 CUDA active-set closure failed step {self._current_step}"
                )
            pin_sign = torch.where(newly_active, torch.sign(solved_lesp), pin_sign)
            active_work = active_work | newly_active
        active = active_work
        residual = matrix @ solution - target
        scale = torch.clamp(torch.max(torch.abs(gamma_pre)), min=1.0)
        neumann_max = torch.max(torch.abs(residual[:panel_count]))
        kelvin_max = torch.max(
            torch.abs(gamma_tev - gamma_bound[trailing_panel_indices] - gamma_lev)
        )
        if bool((neumann_max > float(self.jcfg.gate_rtol) * scale).item()):
            raise GateError(
                f"G1 CUDA Neumann residual {float(neumann_max.item()):.3e} "
                f"step {self._current_step}"
            )
        if bool((kelvin_max > float(self.jcfg.gate_rtol) * scale).item()):
            raise GateError(f"G2 CUDA Kelvin residual step {self._current_step}")
        if bool(allowed.item()):
            signed_target = pin_sign * float(self.jcfg.lesp_crit)
            pin_error = torch.abs(solved_lesp - signed_target)
            pin_limit = float(self.jcfg.lesp_rtol) * torch.clamp(
                torch.abs(signed_target), min=1.0
            )
            if bool(torch.any(active & (pin_error > pin_limit)).item()):
                condition = torch.linalg.cond(matrix)
                raise GateError(
                    "G3 CUDA LESP pin residual "
                    f"step {self._current_step}: "
                    f"error={float(torch.max(pin_error[active]).item()):.6e}, "
                    f"condition={float(condition.item()):.6e}, "
                    f"gamma_bound={float(torch.max(torch.abs(gamma_bound)).item()):.6e}, "
                    f"gamma_tev={float(torch.max(torch.abs(gamma_tev)).item()):.6e}, "
                    f"gamma_lev={float(torch.max(torch.abs(gamma_lev)).item()):.6e}"
                )
            if bool(
                torch.any((~active) & (torch.abs(solved_lesp) > inactive_limit)).item()
            ):
                raise GateError(
                    f"G3 CUDA inactive LESP residual step {self._current_step}"
                )
        return (
            gamma_bound,
            gamma_tev,
            gamma_lev,
            solved_lesp,
            lev_rings,
            pseudovortex_rings,
            active,
        )

    def _calculate_vortex_strengths(self) -> None:
        d = self.cuda_device
        rhs = -(self._cuda_wake + self._cuda_freestream)
        gamma = torch.linalg.solve(self._cuda_aic, rhs)
        if not bool(torch.isfinite(gamma).all().item()):
            raise RuntimeError("non-finite CUDA bound-vortex solution")
        panels, span_count, chord_count = self._panel_grid()
        gp = gamma.reshape(chord_count, span_count)
        le = _cuda_tensor(self._le_points_now, d)
        te = _cuda_tensor(self._te_points_now, d)
        v_inf = _cuda_tensor(self.current_operating_point.vInf_GP1__E, d)
        if not hasattr(self, "_le_points_prev"):
            v_rel_st = v_inf.unsqueeze(0).expand(span_count + 1, 3).clone()
        else:
            le_prev = _cuda_tensor(self._le_points_prev, d)
            v_rel_st = v_inf.unsqueeze(0) - (le - le_prev) / float(self.delta_time)
        v_ref = 0.5 * (
            torch.linalg.vector_norm(v_rel_st[:-1], dim=1)
            + torch.linalg.vector_norm(v_rel_st[1:], dim=1)
        )
        panel_fl = _cuda_tensor(
            [panels[0, s].Flpp_GP1_CgP1 for s in range(span_count)], d
        )
        panel_fr = _cuda_tensor(
            [panels[0, s].Frpp_GP1_CgP1 for s in range(span_count)], d
        )
        panel_bl_te = _cuda_tensor(
            [panels[chord_count - 1, s].Blpp_GP1_CgP1 for s in range(span_count)],
            d,
        )
        panel_br_te = _cuda_tensor(
            [panels[chord_count - 1, s].Brpp_GP1_CgP1 for s in range(span_count)],
            d,
        )
        panel_bl_le = _cuda_tensor(
            [panels[0, s].Blpp_GP1_CgP1 for s in range(span_count)], d
        )
        panel_br_le = _cuda_tensor(
            [panels[0, s].Brpp_GP1_CgP1 for s in range(span_count)], d
        )
        chords = 0.5 * (
            torch.linalg.vector_norm(panel_bl_te - panel_fl, dim=1)
            + torch.linalg.vector_norm(panel_br_te - panel_fr, dim=1)
        )
        dx_first = 0.5 * (
            torch.linalg.vector_norm(panel_bl_le - panel_fl, dim=1)
            + torch.linalg.vector_norm(panel_br_le - panel_fr, dim=1)
        )
        theta1 = torch.acos(torch.clamp(1.0 - 2.0 * dx_first / chords, -1.0, 1.0))
        lesp = -LESP_FACTOR * gp[0] / (chords * v_ref * (theta1 + torch.sin(theta1)))
        if not bool(torch.isfinite(lesp).all().item()):
            raise GateError(f"non-finite CUDA LESP at step {self._current_step}")

        allowed = torch.as_tensor(
            self.jcfg.enable_lev and self._current_step >= self.jcfg.lev_start_step,
            device=d,
            dtype=torch.bool,
        )
        # Production separated flow is fail-closed and strip-local.  The old
        # reference used a global |LESP|<10/same-sign heuristic that silently
        # switched LEV off for a whole symmetric wing exactly when separation
        # became strongest.  The augmented solver already carries one signed
        # pin per strip, so mixed signs and large pre-solve LESP must activate
        # those rows rather than disable the mechanism.
        active = allowed & (torch.abs(lesp) > float(self.jcfg.lesp_crit))
        dvm_node_ribbon = self._dvm_node_ribbon_enabled()
        if dvm_node_ribbon:
            (
                gamma_bound,
                gamma_tev,
                gamma_lev,
                solved_lesp,
                lesp,
                active,
            ) = self._evaluate_dvm_node_ribbon_cuda(
                leading_edge=le,
                trailing_edge=te,
                panels=panels,
                collocation=_cuda_tensor(self.stackCpp_GP1_CgP1, d),
                normals=_cuda_tensor(self.stackUnitNormals_GP1, d),
                rhs_without_newborn=rhs,
                ptera_lesp_scale=(
                    chords * v_ref * (theta1 + torch.sin(theta1)) / LESP_FACTOR
                ),
                ptera_separated=active,
            )
            self._current_pseudovortex_rings_cuda = None
            self._current_pseudovortex_strengths_cuda = None
            self._current_pseudovortex_core_cuda = None
        elif self.jcfg.joint_tev:
            (
                gamma_bound,
                gamma_tev,
                gamma_lev,
                solved_lesp,
                lev_rings,
                pseudovortex_rings,
                active,
            ) = self._joint_vortex_solve_cuda(
                gamma,
                rhs,
                lesp,
                active,
                allowed,
                panels,
                le,
                chords,
                v_ref,
                theta1,
            )
            self._current_pseudovortex_rings_cuda = pseudovortex_rings.clone()
            self._current_pseudovortex_strengths_cuda = gamma_lev.clone()
            self._current_pseudovortex_core_cuda = self._bound_core_radii()[
                :span_count
            ].clone()
        else:
            ratio = torch.abs(
                float(self.jcfg.lesp_crit)
                / torch.where(lesp != 0.0, lesp, torch.ones_like(lesp))
            )
            capped = gp[0] * ratio
            if self.jcfg.load_mode == "bing":
                gamma_bound = gamma.clone()
                gamma_bound[:span_count] = torch.where(active, capped, gp[0])
                gamma_lev = torch.where(active, capped - gp[0], 0.0)
            elif self.jcfg.load_mode == "v4b3d":
                gamma_bound = gamma
                excess = gp[0] * (1.0 - ratio)
                gamma_lev = torch.where(active, -excess, 0.0)
            else:
                raise ValueError(f"unsupported CUDA load_mode {self.jcfg.load_mode!r}")
            gamma_tev = torch.zeros_like(gamma_lev)
            solved_lesp = (
                -LESP_FACTOR
                * gamma_bound.reshape(chord_count, span_count)[0]
                / (chords * v_ref * (theta1 + torch.sin(theta1)))
            )
            leading_normals = _cuda_tensor(
                [panels[0, s].unitNormal_GP1 for s in range(span_count)], d
            )
            leg_right = (
                torch.sum(
                    v_rel_st[1:] * float(self.delta_time) * leading_normals,
                    dim=1,
                    keepdim=True,
                )
                * leading_normals
            )
            leg_left = (
                torch.sum(
                    v_rel_st[:-1] * float(self.delta_time) * leading_normals,
                    dim=1,
                    keepdim=True,
                )
                * leading_normals
            )
            lev_rings = torch.empty((span_count, 4, 3), device=d, dtype=torch.float64)
            lev_rings[:, 0] = le[1:]
            lev_rings[:, 1] = le[:-1]
            lev_rings[:, 2] = le[:-1] + leg_left
            lev_rings[:, 3] = le[1:] + leg_right
        if not bool(torch.isfinite(gamma_bound).all().item()):
            raise GateError(
                f"non-finite CUDA active-LEV strengths at step {self._current_step}"
            )
        if bool(torch.any(active).item()):
            if not self.jcfg.joint_tev:
                relaxed = torch.abs(solved_lesp[active]) <= (
                    torch.abs(lesp[active]) + self.jcfg.lesp_rtol
                )
                if not bool(torch.all(relaxed).item()):
                    raise GateError(
                        "CUDA active-LEV failed to relax LESP at step "
                        f"{self._current_step}"
                    )

            if not dvm_node_ribbon:
                shed_count = self.lev_pf.add_ring_particles(
                    lev_rings,
                    gamma_lev,
                    sigma_factor=float(self.jcfg.sigma_factor),
                    birth_step=self._current_step,
                    reverse=self.jcfg.joint_tev,
                )
                self.cuda_counters["particle_shed"] += shed_count

        self._cuda_last_bound_strengths_for_loads = (
            self._cuda_bound_strengths.clone()
            if hasattr(self, "_cuda_bound_strengths")
            else torch.zeros_like(gamma_bound)
        )
        self._cuda_bound_strengths = gamma_bound
        # The legacy Hirato cap-and-edit model deliberately convects its
        # particles in the pre-cap attached field.  A DVM node ribbon is a
        # different ownership contract: the newborn sheet has already entered
        # the same-step Neumann solve, so ``gamma_bound`` is the unique physical
        # bound state seen by both the material LEV and its live frontier.
        # Retaining ``gamma`` here made the next transport stage advance in a
        # shadow attached field that no longer matched either the panel objects
        # or the load/RHS state.
        self._cuda_particle_bound_strengths = gamma_bound if dvm_node_ribbon else gamma
        gamma_cpu = gamma_bound.detach().cpu().numpy()
        self._current_bound_vortex_strengths = gamma_cpu
        self._last_bound = (
            self._cuda_particle_bound_strengths.detach().cpu().numpy().copy()
        )
        for index, panel in enumerate(self.panels):
            panel.ring_vortex.strength = float(gamma_cpu[index])

        panel_area_grid = _cuda_tensor(
            [
                [panels[j, s].area for s in range(span_count)]
                for j in range(chord_count)
            ],
            d,
        )
        areas = torch.sum(panel_area_grid, dim=0)
        self.ledger.append(
            {
                "step": self._current_step,
                "lesp": lesp.clone(),
                "chords": chords.clone(),
                "areas": areas.clone(),
                "v_rel_st": v_rel_st.clone(),
                "v_inf": v_inf.clone(),
                "le_now": le.clone(),
                "te_now": te.clone(),
                "le_prev": (
                    None
                    if not hasattr(self, "_le_points_prev")
                    else _cuda_tensor(self._le_points_prev, d)
                ),
                "dt": float(self.delta_time),
            }
        )
        self._tev_solved = gamma_tev.clone() if self.jcfg.joint_tev else None
        self._lev_hist.append(gamma_lev.detach().cpu().numpy())
        self._tev_hist.append(gamma_tev.detach().cpu().numpy())
        self._cuda_lev_history_total = self._cuda_lev_history_total + torch.sum(
            gamma_lev
        )
        self._cuda_tev_history_total = self._cuda_tev_history_total + torch.sum(
            gamma_tev
        )
        self._steps_done += 1
        if self.jcfg.joint_tev:
            rear_bound = gamma_bound.reshape(chord_count, span_count)[-1]
            eq9_max = torch.max(torch.abs(gamma_tev - rear_bound - gamma_lev))
        else:
            eq9_max = torch.zeros((), device=d, dtype=torch.float64)
        circulation_value = float(eq9_max.item())
        if self._circ0 is None:
            self._circ0 = 0.0
        self.diag.append(
            {
                "step": self._current_step,
                "n_particles": self.lev_pf.n,
                "lev_strips": int(torch.count_nonzero(active).item()),
                "lesp_pre_max_abs": float(torch.max(torch.abs(lesp)).item()),
                "lesp_pre_min": float(torch.min(lesp).item()),
                "lesp_pre_max": float(torch.max(lesp).item()),
                "lesp_max": float(torch.max(torch.abs(solved_lesp)).item()),
                "g_tev": float(torch.sum(gamma_tev).item()),
                "g_lev": float(torch.sum(gamma_lev).item()),
                "gamma_pre_max_abs": float(torch.max(torch.abs(gamma)).item()),
                "gamma_bound_max_abs": float(torch.max(torch.abs(gamma_bound)).item()),
                "gamma_tev_max_abs": float(torch.max(torch.abs(gamma_tev)).item()),
                "gamma_lev_max_abs": float(torch.max(torch.abs(gamma_lev)).item()),
                "kelvin_eq9_max_abs": circulation_value,
                "separated_source": self.jcfg.separated_source,
                "dvm_newborn_normal_influence_max_abs": (
                    0.0
                    if not dvm_node_ribbon
                    else float(
                        torch.max(
                            torch.abs(self._cuda_dvm_last_newborn_normal_influence)
                        ).item()
                    )
                ),
                "dvm_ptera_lesp_pin_max_abs": (
                    0.0
                    if not dvm_node_ribbon
                    else float(self._cuda_dvm_last_lesp_pin_residual.item())
                ),
                "dvm_ptera_pin_strips": (
                    0
                    if not dvm_node_ribbon
                    else int(
                        torch.count_nonzero(self._cuda_dvm_last_ptera_pin_active).item()
                    )
                ),
                "dvm_ptera_retained_neumann_max_abs": (
                    0.0
                    if not dvm_node_ribbon
                    else float(self._cuda_dvm_last_retained_neumann_residual.item())
                ),
                "dvm_node_source_active_count": (
                    0
                    if not dvm_node_ribbon
                    else int(
                        torch.count_nonzero(
                            self._cuda_dvm_last_node_source_active
                        ).item()
                    )
                ),
                "dvm_node_topology_active_count": (
                    0
                    if not dvm_node_ribbon
                    else int(
                        torch.count_nonzero(
                            self._cuda_dvm_last_node_topology_active
                        ).item()
                    )
                ),
                "dvm_node_activity_mismatch_count": (
                    0
                    if not dvm_node_ribbon
                    else int(self._cuda_dvm_last_node_activity_mismatch_count.item())
                ),
                # Compatibility key for existing diagnostics.  The previous
                # implementation summed interior panel-ring strengths and
                # called that a global circulation, which is not a valid
                # contour.  It now reports the independently checkable Eq. 9
                # material-contour residual instead.
                "circ_drift": circulation_value,
            }
        )
        self.cuda_counters["solve"] += 1
        self.cuda_counters["ledger"] += 1

    def _external_velocity_cuda(self, points: torch.Tensor) -> torch.Tensor:
        v_inf = _cuda_tensor(self._currentVInf_GP1__E, self.cuda_device)
        return self._bound_velocity(points) + self._wake_velocity(points) + v_inf

    def _particle_external_velocity_cuda(self, points: torch.Tensor) -> torch.Tensor:
        """Evaluate the mode-owned bound/wake field for LEV convection.

        ``hirato_ring`` preserves its frozen pre-cap transport convention;
        ``dvm_node_ribbon`` stores the final physical bound solution here.
        """

        br, fr, fl, bl = self._bound_rings()
        strengths = self._cuda_particle_bound_strengths
        bound = _chunked_ring_velocity_collapsed(
            points,
            br,
            fr,
            fl,
            bl,
            strengths,
            torch.zeros(
                strengths.shape[0], device=self.cuda_device, dtype=torch.float64
            ),
            torch.zeros(
                strengths.shape[0], device=self.cuda_device, dtype=torch.float64
            ),
            float(self.current_operating_point.nu),
        )
        wake = torch.zeros_like(points)
        if self._current_step and len(self._current_wake_vortex_strengths):
            wake_br, wake_fr, wake_fl, wake_bl = self._wake_rings()
            wake_strengths = _cuda_tensor(
                self._current_wake_vortex_strengths, self.cuda_device
            )
            wake = _chunked_ring_velocity_collapsed(
                points,
                wake_br,
                wake_fr,
                wake_fl,
                wake_bl,
                wake_strengths,
                _cuda_tensor(self._currentStackWakeRc0s, self.cuda_device),
                torch.zeros_like(wake_strengths),
                float(self.current_operating_point.nu),
            )
        v_inf = _cuda_tensor(self._currentVInf_GP1__E, self.cuda_device)
        return bound + wake + v_inf

    def _solution_velocity_cuda(self, points: torch.Tensor) -> torch.Tensor:
        velocity = self._external_velocity_cuda(points)
        if self.lev_pf.n:
            velocity = velocity + self.lev_pf.velocity_at_cuda(points)
            self.cuda_counters["particle_velocity"] += 1
        pseudo_rings = getattr(self, "_current_pseudovortex_rings_cuda", None)
        pseudo_strengths = getattr(self, "_current_pseudovortex_strengths_cuda", None)
        pseudo_core = getattr(self, "_current_pseudovortex_core_cuda", None)
        if (
            pseudo_rings is not None
            and pseudo_strengths is not None
            and pseudo_core is not None
        ):
            pseudo_velocity = _ring_velocity_expanded(
                points,
                pseudo_rings[:, 2],
                pseudo_rings[:, 1],
                pseudo_rings[:, 0],
                pseudo_rings[:, 3],
                pseudo_strengths,
                pseudo_core,
                torch.zeros_like(pseudo_strengths),
                float(self.current_operating_point.nu),
            )
            velocity = velocity + torch.sum(pseudo_velocity, dim=1)
        return velocity

    def calculate_solution_velocity(
        self, stackP_GP1_CgP1: Any = None, **_: Any
    ) -> np.ndarray:
        if stackP_GP1_CgP1 is None:
            raise ValueError("stackP_GP1_CgP1 is required")
        points = _cuda_tensor(stackP_GP1_CgP1, self.cuda_device)
        velocity = self._solution_velocity_cuda(points)
        self.cuda_counters["velocity"] += 1
        return velocity.detach().cpu().numpy()

    def _movement_velocity(self, current: Any, previous: Any | None) -> torch.Tensor:
        cur = _cuda_tensor(current, self.cuda_device)
        if self._current_step < 1 or previous is None:
            return torch.zeros_like(cur)
        return -(cur - _cuda_tensor(previous, self.cuda_device)) / float(
            self.delta_time
        )

    def _effective_strengths(self, gamma: torch.Tensor) -> tuple[torch.Tensor, ...]:
        _, span_count, chord_count = self._panel_grid()
        grid = gamma.reshape(chord_count, span_count)
        right = grid.clone()
        right[:, :-1] = 0.5 * (grid[:, :-1] - grid[:, 1:])
        front = grid.clone()
        front[1:, :] = 0.5 * (grid[1:, :] - grid[:-1, :])
        left = grid.clone()
        left[:, 1:] = 0.5 * (grid[:, 1:] - grid[:, :-1])
        back = grid.clone()
        back[:-1, :] = 0.5 * (grid[:-1, :] - grid[1:, :])
        if self._current_step > 0:
            last = self._cuda_last_bound_strengths_for_loads
            back[-1, :] = grid[-1, :] - last.reshape(chord_count, span_count)[-1, :]
        return tuple(x.reshape(-1) for x in (right, front, left, back))

    def _calculate_loads(self) -> None:
        d = self.cuda_device
        gamma = self._cuda_bound_strengths
        right_s, front_s, left_s, back_s = self._effective_strengths(gamma)
        locations = (
            self.stackCblvpr_GP1_CgP1,
            self.stackCblvpf_GP1_CgP1,
            self.stackCblvpl_GP1_CgP1,
            self.stackCblvpb_GP1_CgP1,
        )
        previous = (
            getattr(self, "_lastStackCblvpr_GP1_CgP1", None),
            getattr(self, "_lastStackCblvpf_GP1_CgP1", None),
            getattr(self, "_lastStackCblvpl_GP1_CgP1", None),
            getattr(self, "_lastStackCblvpb_GP1_CgP1", None),
        )
        vectors = (
            self.stackRbrv_GP1,
            self.stackFbrv_GP1,
            self.stackLbrv_GP1,
            self.stackBbrv_GP1,
        )
        strengths = (right_s, front_s, left_s, back_s)
        # All four effective-vortex locations share the same bound, wake and
        # particle sources.  Evaluate them as one 4N target batch so the GPU
        # sees a materially larger parallel problem and source tensors are not
        # traversed four times per load evaluation.
        points = _cuda_tensor(np.stack(locations, axis=0), d)
        panel_count = points.shape[1]
        velocity = self._solution_velocity_cuda(points.reshape(-1, 3)).reshape(
            4, panel_count, 3
        )
        if self._current_step < 1 or any(item is None for item in previous):
            movement = torch.zeros_like(points)
        else:
            previous_points = _cuda_tensor(np.stack(previous, axis=0), d)
            movement = -(points - previous_points) / float(self.delta_time)
        vectors_cuda = _cuda_tensor(np.stack(vectors, axis=0), d)
        strengths_cuda = torch.stack(strengths, dim=0)
        forces_cuda = (
            float(self.current_operating_point.rho)
            * strengths_cuda.unsqueeze(2)
            * torch.linalg.cross(velocity + movement, vectors_cuda, dim=2)
        )
        forces = tuple(forces_cuda[index] for index in range(4))
        last_gamma = self._cuda_last_bound_strengths_for_loads
        unsteady = -(
            float(self.current_operating_point.rho)
            * (gamma - last_gamma).unsqueeze(1)
            * _cuda_tensor(self.panel_areas, d).unsqueeze(1)
            * _cuda_tensor(self.stackUnitNormals_GP1, d)
            / float(self.delta_time)
        )
        panel_forces = sum(forces, start=torch.zeros_like(unsteady)) + unsteady
        if self.diag and self.diag[-1]["step"] == self._current_step:
            self.diag[-1].update(
                kj_panel_force_max_abs=float(
                    torch.max(
                        torch.abs(sum(forces, start=torch.zeros_like(unsteady)))
                    ).item()
                ),
                unsteady_panel_force_max_abs=float(
                    torch.max(torch.abs(unsteady)).item()
                ),
                panel_force_max_abs=float(torch.max(torch.abs(panel_forces)).item()),
            )
        moment_points = tuple(points[index] for index in range(4))
        panel_moments = sum(
            (
                torch.linalg.cross(point, force, dim=1)
                for point, force in zip(moment_points, forces, strict=True)
            ),
            start=torch.zeros_like(unsteady),
        )
        panel_moments = panel_moments + torch.linalg.cross(
            _cuda_tensor(self.stackCpp_GP1_CgP1, d), unsteady, dim=1
        )

        # Force transform to world frame: use Pterra's validated GP1→W matrix.
        # The V5M "scientific frame" is identity to GP1 by construction (see
        # _v5m_gp_to_scientific_rotation_cuda = eye(3)), but computing it
        # through the Pterra transform preserves the exact frozen path for
        # the three-paper chassis reproduction.
        transform = _cuda_tensor(
            self.current_operating_point.T_pas_GP1_CgP1_to_W_CgP1, d
        )
        rotation = transform[:3, :3]
        translation = transform[:3, 3]
        panel_forces_w = (
            torch.einsum("ij,nj->ni", rotation, panel_forces) + translation
        )
        panel_moments_w = (
            torch.einsum("ij,nj->ni", rotation, panel_moments)
            + torch.linalg.cross(
                translation.unsqueeze(0), panel_forces_w, dim=1
            )
        )
        resolved_points_gp = torch.cat(
            (*moment_points, _cuda_tensor(self.stackCpp_GP1_CgP1, d)), dim=0
        )
        resolved_forces_gp = torch.cat((*forces, unsteady), dim=0)
        resolved_points_w = (
            torch.einsum("ij,nj->ni", rotation, resolved_points_gp) + translation
        )
        resolved_forces_w = torch.einsum("ij,nj->ni", rotation, resolved_forces_gp)
        total_force_w = torch.sum(panel_forces_w, dim=0)
        total_moment_w = torch.sum(panel_moments_w, dim=0)
        _, span_count, chord_count = self._panel_grid()
        leading_edge_gp, _ = self._station_le_te_points()
        leading_edge_gp_cuda = _cuda_tensor(leading_edge_gp, d)
        leading_edge_w = (
            torch.einsum("ij,nj->ni", rotation, leading_edge_gp_cuda) + translation
        )
        strip_endpoints_w = torch.stack(
            (leading_edge_w[:-1], leading_edge_w[1:]), dim=1
        )
        if strip_endpoints_w.shape != (span_count, 2, 3) or not bool(
            torch.isfinite(strip_endpoints_w).all().item()
        ):
            raise GateError("spanwise impulse source endpoints are invalid")
        # Every load-bearing step publishes an exact strip-owner record.  Before
        # an impulse history exists this is a physically zero strip force, not
        # a missing optional mechanism: separated LEV remains mandatory and
        # later predictor/corrector code consumes one stable schema.
        self._q16_impulse_strip_force_w = torch.zeros(
            (span_count, 3), dtype=torch.float64, device=d
        )
        self._q16_impulse_strip_le_endpoints_w = strip_endpoints_w.detach().clone()
        impulse_force = torch.zeros(3, device=d, dtype=torch.float64)
        diagnostic_impulse_force = torch.zeros_like(impulse_force)
        diagnostic_strip_impulse_force = torch.zeros(
            (span_count, 3), dtype=torch.float64, device=d
        )
        if self.lev_pf.n or self._last_impulse_cuda is not None:
            particle_positions = self.lev_pf.positions_cuda
            particle_gamma = self.lev_pf.gammas_cuda
            particle_source_strip = self.lev_pf.source_strips_cuda
            if particle_source_strip.shape != (self.lev_pf.n,):
                raise GateError("CUDA LEV source-strip shape drift")
            if self.lev_pf.n and bool(
                torch.any(
                    (particle_source_strip < 0) | (particle_source_strip >= span_count)
                ).item()
            ):
                raise GateError("CUDA LEV particle has no valid source strip")
            position_w = (
                torch.einsum("ij,nj->ni", rotation, particle_positions) + translation
            )
            gamma_w = torch.einsum("ij,nj->ni", rotation, particle_gamma)
            gamma_sum = torch.sum(gamma_w, dim=0)
            free_impulse = (
                0.5
                * float(self.current_operating_point.rho)
                * (
                    torch.sum(torch.linalg.cross(position_w, gamma_w, dim=1), dim=0)
                    + torch.linalg.cross(translation, gamma_sum, dim=0)
                )
            )
            normal_w = torch.einsum(
                "ij,nj->ni",
                rotation,
                _cuda_tensor(self.stackUnitNormals_GP1, d),
            )
            bound_impulse = torch.sum(
                float(self.current_operating_point.rho)
                * gamma.unsqueeze(1)
                * _cuda_tensor(self.panel_areas, d).unsqueeze(1)
                * normal_w,
                dim=0,
            )
            impulse = free_impulse + bound_impulse

            # Preserve the frozen global reduction above and independently
            # decompose the same operands by the span strip that shed each LEV
            # particle.  This is causal source ownership, not a post-hoc area
            # split of the global force.
            free_particle_impulse = (
                0.5
                * float(self.current_operating_point.rho)
                * (
                    torch.linalg.cross(position_w, gamma_w, dim=1)
                    + torch.linalg.cross(translation.expand_as(gamma_w), gamma_w, dim=1)
                )
            )
            free_strip_impulse = torch.zeros(
                (span_count, 3), device=d, dtype=torch.float64
            )
            for strip in range(span_count):
                owned = particle_source_strip == strip
                if bool(torch.any(owned).item()):
                    free_strip_impulse[strip] = torch.sum(
                        free_particle_impulse[owned], dim=0
                    )
            bound_panel_impulse = (
                float(self.current_operating_point.rho)
                * gamma.unsqueeze(1)
                * _cuda_tensor(self.panel_areas, d).unsqueeze(1)
                * normal_w
            )
            bound_strip_impulse = torch.sum(
                bound_panel_impulse.reshape(chord_count, span_count, 3), dim=0
            )
            strip_impulse = free_strip_impulse + bound_strip_impulse
            strip_impulse_sum = torch.sum(strip_impulse, dim=0)
            # A symmetric full wing can cancel large left/right strip terms in
            # the global impulse.  Scale the roundoff gate by the L1 magnitude
            # of the independently accumulated operands, not only by their
            # potentially near-zero resultant.
            impulse_operand_scale = torch.sum(
                torch.abs(free_particle_impulse), dim=0
            ) + torch.sum(torch.abs(bound_panel_impulse), dim=0)
            impulse_scale = max(
                1.0,
                float(torch.max(torch.abs(impulse)).item()),
                float(torch.max(torch.abs(strip_impulse_sum)).item()),
                float(torch.max(impulse_operand_scale).item()),
            )
            if (
                float(torch.max(torch.abs(strip_impulse_sum - impulse)).item())
                > 4096.0 * _EPS * impulse_scale
            ):
                raise GateError("spanwise impulse ledger does not close global impulse")

            strip_impulse_force = torch.zeros_like(strip_impulse)
            force_history_scale = torch.zeros(3, device=d, dtype=torch.float64)
            if self._last_impulse_cuda is not None:
                impulse_force = -(impulse - self._last_impulse_cuda) / float(
                    self.delta_time
                )
                if self._last_impulse_strip_cuda is None:
                    raise GateError("spanwise impulse history is missing")
                if self._last_impulse_strip_cuda.shape != strip_impulse.shape:
                    raise GateError("spanwise impulse history shape drift")
                strip_impulse_force = -(
                    strip_impulse - self._last_impulse_strip_cuda
                ) / float(self.delta_time)
                force_history_scale = torch.sum(
                    torch.abs(strip_impulse) + torch.abs(self._last_impulse_strip_cuda),
                    dim=0,
                ) / float(self.delta_time)
            self._last_impulse_cuda = impulse.clone()
            self._last_impulse_strip_cuda = strip_impulse.clone()
            strip_force_sum = torch.sum(strip_impulse_force, dim=0)
            force_operand_scale = torch.sum(torch.abs(strip_impulse_force), dim=0)
            force_scale = max(
                1.0,
                float(torch.max(torch.abs(impulse_force)).item()),
                float(torch.max(torch.abs(strip_force_sum)).item()),
                float(torch.max(force_operand_scale).item()),
                float(torch.max(force_history_scale).item()),
            )
            if (
                float(torch.max(torch.abs(strip_force_sum - impulse_force)).item())
                > 8192.0 * _EPS * force_scale
            ):
                raise GateError("spanwise impulse force does not close global force")

            diagnostic_impulse_force = impulse_force.detach().clone()
            diagnostic_strip_impulse_force = strip_impulse_force.detach().clone()
            if self._dvm_node_ribbon_enabled():
                # In the physical node-ribbon mode the particles already alter
                # the Neumann RHS, final bound circulation and KJ velocities.
                # Ptera's KJ + dGamma panel forces are therefore the unique
                # aerodynamic load owner.  Adding the derivative of a second
                # free+bound vortex-impulse representation here double counts
                # the same separated circulation (and supplies no matching
                # moment).  Preserve it below as a diagnostic only.
                impulse_force = torch.zeros_like(impulse_force)
            else:
                # Preserve the frozen Hirato/Q16 reduced-load contract until
                # that legacy path is migrated independently.
                self._q16_impulse_strip_force_w = strip_impulse_force.detach().clone()
                self._q16_impulse_strip_le_endpoints_w = (
                    strip_endpoints_w.detach().clone()
                )
                total_force_w = total_force_w + impulse_force
                self.impulse_force.append(impulse_force.detach().cpu().numpy().copy())
                self.cuda_counters["impulse"] += 1
        self._diagnostic_vortex_impulse_force_w = diagnostic_impulse_force
        self._diagnostic_vortex_impulse_strip_force_w = diagnostic_strip_impulse_force
        # Retain the already-computed point-load decomposition for the Q16 FSI
        # boundary.  These are clones so later solver scratch updates cannot
        # mutate an issued packet.  The impulse remains separate because this
        # aerodynamic model does not define its structural application point.
        self._q16_resolved_load_points_w = resolved_points_w.detach().clone()
        self._q16_resolved_load_forces_w = resolved_forces_w.detach().clone()
        self._q16_unresolved_impulse_force_w = impulse_force.detach().clone()
        self._q16_total_force_w = total_force_w.detach().clone()
        self._q16_total_moment_w = total_moment_w.detach().clone()
        if self.diag and self.diag[-1]["step"] == self._current_step:
            self.diag[-1].update(
                load_owner=(
                    "ptera_kj_plus_dgamma"
                    if self._dvm_node_ribbon_enabled()
                    else "ptera_plus_legacy_vortex_impulse"
                ),
                diagnostic_vortex_impulse_force_max_abs=float(
                    torch.max(torch.abs(diagnostic_impulse_force)).item()
                ),
            )
        q_inf = float(self.current_operating_point.qInf__E)
        airplane = self.current_airplanes[0]
        force_coeff = total_force_w / (q_inf * float(airplane.s_ref))
        moment_coeff = total_moment_w / (
            q_inf
            * float(airplane.s_ref)
            * _cuda_tensor([airplane.b_ref, airplane.c_ref, airplane.b_ref], d)
        )

        panel_forces_cpu = panel_forces.detach().cpu().numpy()
        panel_moments_cpu = panel_moments.detach().cpu().numpy()
        panel_forces_w_cpu = panel_forces_w.detach().cpu().numpy()
        panel_moments_w_cpu = panel_moments_w.detach().cpu().numpy()
        for index, panel in enumerate(self.panels):
            panel.forces_GP1 = panel_forces_cpu[index]
            panel.moments_GP1_CgP1 = panel_moments_cpu[index]
            panel.forces_W = panel_forces_w_cpu[index]
            panel.moments_W_CgP1 = panel_moments_w_cpu[index]
        airplane.forces_W = total_force_w.detach().cpu().numpy()
        airplane.moments_W_CgP1 = total_moment_w.detach().cpu().numpy()
        airplane.forceCoefficients_W = force_coeff.detach().cpu().numpy()
        airplane.momentCoefficients_W_CgP1 = moment_coeff.detach().cpu().numpy()

        if self.ledger and len(self.ledger) == self._current_step + 1:
            record = self.ledger[-1]
            _, span_count, chord_count = self._panel_grid()
            force_grid = panel_forces.reshape(chord_count, span_count, 3)
            normal_grid = _cuda_tensor(self.stackUnitNormals_GP1, d).reshape(
                chord_count, span_count, 3
            )
            strip_force = torch.sum(force_grid, dim=0)
            strip_normal = normal_grid[0]
            v_rel = _cuda_tensor(record["v_rel_st"], d)
            v_rel = 0.5 * (v_rel[:-1] + v_rel[1:])
            strip_q = (
                0.5
                * float(self.current_operating_point.rho)
                * torch.sum(v_rel * v_rel, dim=1)
            )
            strip_area = _cuda_tensor(record["areas"], d)
            cn = torch.sum(strip_force * strip_normal, dim=1) / torch.clamp(
                strip_q * strip_area, min=1.0e-30
            )
            record["cn_strip"] = cn.clone()
        self.cuda_counters["loads"] += 1

    def _populate_next_airplanes_wake_vortex_points(self) -> None:
        """Advance prescribed-wake coordinates with CUDA float64 arithmetic.

        Ptera objects remain host-owned, but every coordinate addition and time
        integration operation that determines a later aerodynamic state occurs
        on CUDA.  This deliberately replaces Ptera's NumPy wake advancement.
        """
        if self._current_step >= self.num_steps - 1:
            return
        d = self.cuda_device
        next_airplanes = self.steady_problems[self._current_step + 1].airplanes
        v_inf = _cuda_tensor(self._currentVInf_GP1__E, d).reshape(1, 1, 3)
        delta_time = torch.as_tensor(self.delta_time, device=d, dtype=torch.float64)
        for airplane_id, next_airplane in enumerate(next_airplanes):
            this_airplane = self.current_airplanes[airplane_id]
            for wing_id, next_wing in enumerate(next_airplane.wings):
                this_wing = this_airplane.wings[wing_id]
                span_count = this_wing.num_spanwise_panels
                if span_count is None:
                    raise RuntimeError("spanwise panel count is unavailable")
                chord_index = this_wing.num_chordwise_panels - 1
                next_panels = next_wing.panels
                if next_panels is None:
                    raise RuntimeError("next-step panel grid is unavailable")
                new_row_host = np.empty((1, span_count + 1, 3), dtype=np.float64)
                for span_index in range(span_count):
                    ring = next_panels[chord_index, span_index].ring_vortex
                    if ring is None:
                        raise RuntimeError(
                            "next-step trailing-edge ring is unavailable"
                        )
                    new_left = ring.Blrvp_GP1_CgP1
                    new_right = ring.Brrvp_GP1_CgP1
                    if self.jcfg.full_trailing_edge_bound_ring:
                        new_left = ring.Flrvp_GP1_CgP1 + 2.0 * (
                            new_left - ring.Flrvp_GP1_CgP1
                        )
                        new_right = ring.Frrvp_GP1_CgP1 + 2.0 * (
                            new_right - ring.Frrvp_GP1_CgP1
                        )
                    new_row_host[0, span_index] = new_left
                    if span_index == span_count - 1:
                        new_row_host[0, span_index + 1] = new_right
                new_row = _cuda_tensor(new_row_host, d)
                endpoint_velocity_host = getattr(
                    self, "_q16_incremental_bound_vertex_velocity_gp", None
                )
                if endpoint_velocity_host is not None:
                    endpoint_velocity = _cuda_tensor(endpoint_velocity_host, d)
                    if tuple(endpoint_velocity.shape) != (
                        this_wing.num_chordwise_panels + 1,
                        span_count + 1,
                        3,
                    ):
                        raise RuntimeError("Q16 endpoint wake-anchor velocity drift")
                    new_row_velocity = endpoint_velocity[-1:].clone()
                    if self.jcfg.full_trailing_edge_bound_ring:
                        front_velocity = (
                            0.75 * endpoint_velocity[-2:-1]
                            + 0.25 * endpoint_velocity[-1:]
                        )
                        new_row_velocity = front_velocity + 2.0 * (
                            new_row_velocity - front_velocity
                        )
                else:
                    current_row_host = np.empty_like(new_row_host)
                    current_panels = this_wing.panels
                    if current_panels is None:
                        raise RuntimeError("current trailing panel grid is unavailable")
                    for span_index in range(span_count):
                        ring = current_panels[chord_index, span_index].ring_vortex
                        if ring is None:
                            raise RuntimeError(
                                "current trailing-edge ring is unavailable"
                            )
                        current_left = ring.Blrvp_GP1_CgP1
                        current_right = ring.Brrvp_GP1_CgP1
                        if self.jcfg.full_trailing_edge_bound_ring:
                            current_left = ring.Flrvp_GP1_CgP1 + 2.0 * (
                                current_left - ring.Flrvp_GP1_CgP1
                            )
                            current_right = ring.Frrvp_GP1_CgP1 + 2.0 * (
                                current_right - ring.Frrvp_GP1_CgP1
                            )
                        current_row_host[0, span_index] = current_left
                        if span_index == span_count - 1:
                            current_row_host[0, span_index + 1] = current_right
                    new_row_velocity = (
                        new_row - _cuda_tensor(current_row_host, d)
                    ) / delta_time
                if self._current_step == 0:
                    if self._prescribed_wake:
                        wake_velocity = v_inf.expand_as(new_row)
                    else:
                        wake_velocity = self._solution_velocity_cuda(
                            new_row.reshape(-1, 3)
                        ).reshape_as(new_row)
                    convected = new_row + wake_velocity * delta_time
                    grid = torch.cat((new_row, convected), dim=0)
                    velocity_grid = torch.cat(
                        (new_row_velocity, wake_velocity), dim=0
                    )
                else:
                    current_grid_host = this_wing.gridWrvp_GP1_CgP1
                    if current_grid_host is None:
                        raise RuntimeError(
                            "current prescribed wake grid is unavailable"
                        )
                    current_grid = _cuda_tensor(current_grid_host, d)
                    if self._prescribed_wake:
                        wake_velocity = v_inf.expand_as(current_grid)
                    else:
                        wake_velocity = self._solution_velocity_cuda(
                            current_grid.reshape(-1, 3)
                        ).reshape_as(current_grid)
                    convected = current_grid + wake_velocity * delta_time
                    grid = torch.cat((new_row, convected), dim=0)
                    velocity_grid = torch.cat(
                        (new_row_velocity, wake_velocity), dim=0
                    )
                if self._max_wake_rows is not None:
                    grid = grid[: self._max_wake_rows + 1]
                    velocity_grid = velocity_grid[: self._max_wake_rows + 1]
                self._require_v5m_downstream_wake_contract(grid)
                self._v5m_downstream_wake_contract_step = self._current_step + 1
                next_wing.gridWrvp_GP1_CgP1 = grid.detach().cpu().numpy()
                self._q16_cuda_wake_vertex_velocity_grids_gp[
                    self._current_step + 1
                ] = velocity_grid.detach().clone()
        self.cuda_counters["wake_convection"] += 1


# Backwards-compatible import for the already audited attached-only runners.
CudaAttachedJointLEVTEVSolver = CudaJointLEVTEVSolver

__all__ = ["CudaAttachedJointLEVTEVSolver", "CudaJointLEVTEVSolver"]
