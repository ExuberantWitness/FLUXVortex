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
from bing_joint_solver import GateError, LESP_SANITY_MAX
from pfield_torch_gpu import CudaParticleField

_EPS = float(np.finfo(float).eps)
_TOL = 1.0e-10
_FOUR_LAMB = 5.02572
_SQUIRE = 1.0e-4
_FOUR_PI = 4.0 * math.pi


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
    if not enabled or (_COMPILED_BOUND_RING_VELOCITY is None and interactions < 4096):
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
    if not enabled or (
        _COMPILED_COLLAPSED_RING_VELOCITY is None and interactions < 4096
    ):
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
        }

    def _assert_supported_step(self) -> None:
        if self.current_operating_point.surfaceReflect_T_act_GP1_CgP1 is not None:
            raise ValueError("CUDA attached chassis does not support image surfaces")

    def _bound_rings(self) -> tuple[torch.Tensor, ...]:
        d = self.cuda_device
        return (
            _cuda_tensor(self.stackBrbrvp_GP1_CgP1, d),
            _cuda_tensor(self.stackFrbrvp_GP1_CgP1, d),
            _cuda_tensor(self.stackFlbrvp_GP1_CgP1, d),
            _cuda_tensor(self.stackBlbrvp_GP1_CgP1, d),
        )

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
        return _fused_ring_velocity_collapsed(
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
            _cuda_tensor(self._currentStackBoundRc0s, self.cuda_device),
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
        return _fused_ring_velocity_collapsed(
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
                        next_wing.wake_ring_vortices[
                            chordwise_id, spanwise_id
                        ] = _vortices.ring_vortex.RingVortex(
                            Flrvp_GP1_CgP1=front_left,
                            Frrvp_GP1_CgP1=front_right,
                            Blrvp_GP1_CgP1=back_left,
                            Brrvp_GP1_CgP1=back_right,
                            strength=strength,
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
            _cuda_tensor(self._currentStackBoundRc0s, d),
            torch.zeros(n, dtype=torch.float64, device=d),
            float(self.current_operating_point.nu),
        )
        normals = _cuda_tensor(self.stackUnitNormals_GP1, d)
        aic = torch.sum(expanded * normals.unsqueeze(1), dim=2)
        self._currentGridWingWingInfluences__E = aic.detach().cpu().numpy()
        self._cuda_aic = aic
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

    def _joint_vortex_solve_cuda(
        self,
        gamma_pre: torch.Tensor,
        rhs: torch.Tensor,
        lesp: torch.Tensor,
        active: torch.Tensor,
        allowed: torch.Tensor,
        panels: Any,
        leading_edge: torch.Tensor,
        relative_le_velocity: torch.Tensor,
        chords: torch.Tensor,
        reference_velocity: torch.Tensor,
        theta_first: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,]:
        """Assemble, solve and replay the joint bound/TEV/LEV system on CUDA."""

        device = self.cuda_device
        _, span_count, chord_count = self._panel_grid()
        panel_count = self.num_panels
        v_inf = _cuda_tensor(self.current_operating_point.vInf_GP1__E, device)
        trailing_edge = _cuda_tensor(self._te_points_now, device)
        if not hasattr(self, "_te_points_prev"):
            relative_te_velocity = v_inf.unsqueeze(0).expand(span_count + 1, 3).clone()
        else:
            relative_te_velocity = v_inf.unsqueeze(0) - (
                trailing_edge - _cuda_tensor(self._te_points_prev, device)
            ) / float(self.delta_time)

        te_front_right = _cuda_tensor(
            [
                panels[chord_count - 1, span].ring_vortex.Brrvp_GP1_CgP1
                for span in range(span_count)
            ],
            device,
        )
        te_front_left = _cuda_tensor(
            [
                panels[chord_count - 1, span].ring_vortex.Blrvp_GP1_CgP1
                for span in range(span_count)
            ],
            device,
        )
        tev_rings = torch.empty((span_count, 4, 3), device=device, dtype=torch.float64)
        tev_rings[:, 0] = te_front_right
        tev_rings[:, 1] = te_front_left
        tev_rings[:, 2] = te_front_left + relative_te_velocity[:-1] * float(
            self.delta_time
        )
        tev_rings[:, 3] = te_front_right + relative_te_velocity[1:] * float(
            self.delta_time
        )

        leading_normals = _cuda_tensor(
            [panels[0, span].unitNormal_GP1 for span in range(span_count)],
            device,
        )
        leg_right = (
            torch.sum(
                relative_le_velocity[1:] * float(self.delta_time) * leading_normals,
                dim=1,
                keepdim=True,
            )
            * leading_normals
        )
        leg_left = (
            torch.sum(
                relative_le_velocity[:-1] * float(self.delta_time) * leading_normals,
                dim=1,
                keepdim=True,
            )
            * leading_normals
        )
        lev_rings = torch.empty((span_count, 4, 3), device=device, dtype=torch.float64)
        lev_rings[:, 0] = leading_edge[1:]
        lev_rings[:, 1] = leading_edge[:-1]
        lev_rings[:, 2] = leading_edge[:-1] + leg_left
        lev_rings[:, 3] = leading_edge[1:] + leg_right

        collocation = _cuda_tensor(self.stackCpp_GP1_CgP1, device)
        normals = _cuda_tensor(self.stackUnitNormals_GP1, device)
        tev_velocity = _unregularized_ring_velocity_expanded(collocation, tev_rings)
        lev_velocity = _unregularized_ring_velocity_expanded(collocation, lev_rings)
        a_tev = torch.sum(normals.unsqueeze(1) * tev_velocity, dim=2)
        a_lev = torch.sum(normals.unsqueeze(1) * lev_velocity, dim=2)

        augmented_count = panel_count + 2 * span_count
        matrix = torch.zeros(
            (augmented_count, augmented_count),
            device=device,
            dtype=torch.float64,
        )
        target = torch.zeros(augmented_count, device=device, dtype=torch.float64)
        matrix[:panel_count, :panel_count] = self._cuda_aic
        matrix[:panel_count, panel_count : panel_count + span_count] = -a_tev
        matrix[:panel_count, panel_count + span_count :] = -a_lev
        target[:panel_count] = rhs
        safe_lesp = torch.where(lesp != 0.0, lesp, torch.ones_like(lesp))
        caps = gamma_pre[:span_count] * torch.abs(
            float(self.jcfg.lesp_crit) / safe_lesp
        )
        for span in range(span_count):
            te_index = (chord_count - 1) * span_count + span
            matrix[panel_count + span, te_index] = 1.0
            matrix[panel_count + span, panel_count + span] = -1.0
            matrix[panel_count + span, panel_count + span_count + span] = -1.0
            if bool(active[span].item()):
                matrix[panel_count + span_count + span, span] = 1.0
                target[panel_count + span_count + span] = caps[span]
            else:
                matrix[
                    panel_count + span_count + span,
                    panel_count + span_count + span,
                ] = 1.0

        solution = torch.linalg.solve(matrix, target)
        if not bool(torch.isfinite(solution).all().item()):
            raise GateError(
                f"G5 non-finite CUDA joint solution step {self._current_step}"
            )
        gamma_bound = solution[:panel_count]
        gamma_tev = solution[panel_count : panel_count + span_count]
        gamma_lev = solution[panel_count + span_count :]
        residual = matrix @ solution - target
        scale = torch.clamp(torch.max(torch.abs(gamma_pre)), min=1.0)
        neumann_max = torch.max(torch.abs(residual[:panel_count]))
        kelvin_max = torch.max(
            torch.abs(residual[panel_count : panel_count + span_count])
        )
        if bool((neumann_max > float(self.jcfg.gate_rtol) * scale).item()):
            raise GateError(
                f"G1 CUDA Neumann residual {float(neumann_max.item()):.3e} "
                f"step {self._current_step}"
            )
        if bool((kelvin_max > float(self.jcfg.gate_rtol) * scale).item()):
            raise GateError(f"G2 CUDA Kelvin residual step {self._current_step}")
        solved_lesp = (
            -LESP_FACTOR
            * gamma_bound.reshape(chord_count, span_count)[0]
            / (chords * reference_velocity * (theta_first + torch.sin(theta_first)))
        )
        if bool(allowed.item()):
            signed_target = torch.sign(lesp) * float(self.jcfg.lesp_crit)
            pin_error = torch.abs(solved_lesp - signed_target)
            pin_limit = float(self.jcfg.lesp_rtol) * torch.clamp(
                torch.abs(signed_target), min=1.0
            )
            if bool(torch.any(active & (pin_error > pin_limit)).item()):
                raise GateError(f"G3 CUDA LESP pin residual step {self._current_step}")
            inactive_limit = float(self.jcfg.lesp_inactive_margin) * float(
                self.jcfg.lesp_crit
            )
            if bool(
                torch.any((~active) & (torch.abs(solved_lesp) > inactive_limit)).item()
            ):
                raise GateError(
                    f"G3 CUDA inactive LESP residual step {self._current_step}"
                )
        return gamma_bound, gamma_tev, gamma_lev, solved_lesp, lev_rings

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
        allowed = allowed & (torch.max(torch.abs(lesp)) < LESP_SANITY_MAX)
        allowed = allowed & (torch.max(lesp) * torch.min(lesp) >= 0.0)
        active = allowed & (torch.abs(lesp) > float(self.jcfg.lesp_crit))
        if self.jcfg.joint_tev:
            (
                gamma_bound,
                gamma_tev,
                gamma_lev,
                solved_lesp,
                lev_rings,
            ) = self._joint_vortex_solve_cuda(
                gamma,
                rhs,
                lesp,
                active,
                allowed,
                panels,
                le,
                v_rel_st,
                chords,
                v_ref,
                theta1,
            )
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
            relaxed = torch.abs(solved_lesp[active]) <= (
                torch.abs(lesp[active]) + self.jcfg.lesp_rtol
            )
            if not bool(torch.all(relaxed).item()):
                raise GateError(
                    f"CUDA active-LEV failed to relax LESP at step {self._current_step}"
                )

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
        self._cuda_particle_bound_strengths = gamma
        gamma_cpu = gamma_bound.detach().cpu().numpy()
        self._current_bound_vortex_strengths = gamma_cpu
        self._last_bound = gamma.detach().cpu().numpy().copy()
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
            circulation = (
                torch.sum(gamma_bound)
                + torch.sum(gamma_tev)
                + self._cuda_tev_history_total
                + self._cuda_lev_history_total
            )
        else:
            circulation = torch.sum(gamma_bound) - self._cuda_lev_history_total
        circulation_value = float(circulation.item())
        if self._circ0 is None:
            self._circ0 = circulation_value
        self.diag.append(
            {
                "step": self._current_step,
                "n_particles": self.lev_pf.n,
                "lev_strips": int(torch.count_nonzero(active).item()),
                "lesp_max": float(torch.max(torch.abs(solved_lesp)).item()),
                "g_tev": float(torch.sum(gamma_tev).item()),
                "g_lev": float(torch.sum(gamma_lev).item()),
                "circ_drift": abs(circulation_value - self._circ0),
            }
        )
        self.cuda_counters["solve"] += 1
        self.cuda_counters["ledger"] += 1

    def _external_velocity_cuda(self, points: torch.Tensor) -> torch.Tensor:
        v_inf = _cuda_tensor(self._currentVInf_GP1__E, self.cuda_device)
        return self._bound_velocity(points) + self._wake_velocity(points) + v_inf

    def _particle_external_velocity_cuda(self, points: torch.Tensor) -> torch.Tensor:
        """Mirror the frozen reference's pre-cap bound field for LEV convection."""

        br, fr, fl, bl = self._bound_rings()
        strengths = self._cuda_particle_bound_strengths
        bound = _fused_ring_velocity_collapsed(
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
            wake = _fused_ring_velocity_collapsed(
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

        transform = _cuda_tensor(
            self.current_operating_point.T_pas_GP1_CgP1_to_W_CgP1, d
        )
        rotation = transform[:3, :3]
        translation = transform[:3, 3]
        panel_forces_w = torch.einsum("ij,nj->ni", rotation, panel_forces)
        panel_moments_w = (
            torch.einsum("ij,nj->ni", rotation, panel_moments) + translation
        )
        total_force_w = torch.sum(panel_forces_w, dim=0)
        total_moment_w = torch.sum(panel_moments_w, dim=0)
        if self.lev_pf.n or self._last_impulse_cuda is not None:
            particle_positions = self.lev_pf.positions_cuda
            particle_gamma = self.lev_pf.gammas_cuda
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
            impulse_force = torch.zeros(3, device=d, dtype=torch.float64)
            if self._last_impulse_cuda is not None:
                impulse_force = -(impulse - self._last_impulse_cuda) / float(
                    self.delta_time
                )
            self._last_impulse_cuda = impulse.clone()
            total_force_w = total_force_w + impulse_force
            self.impulse_force.append(impulse_force.detach().cpu().numpy().copy())
            self.cuda_counters["impulse"] += 1
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
                    new_row_host[0, span_index] = ring.Blrvp_GP1_CgP1
                    if span_index == span_count - 1:
                        new_row_host[0, span_index + 1] = ring.Brrvp_GP1_CgP1
                new_row = _cuda_tensor(new_row_host, d)
                if self._current_step == 0:
                    if self._prescribed_wake:
                        wake_velocity = v_inf.expand_as(new_row)
                    else:
                        wake_velocity = self._solution_velocity_cuda(
                            new_row.reshape(-1, 3)
                        ).reshape_as(new_row)
                    convected = new_row + wake_velocity * delta_time
                    grid = torch.cat((new_row, convected), dim=0)
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
                if self._max_wake_rows is not None:
                    grid = grid[: self._max_wake_rows + 1]
                next_wing.gridWrvp_GP1_CgP1 = grid.detach().cpu().numpy()
        self.cuda_counters["wake_convection"] += 1


# Backwards-compatible import for the already audited attached-only runners.
CudaAttachedJointLEVTEVSolver = CudaJointLEVTEVSolver

__all__ = ["CudaAttachedJointLEVTEVSolver", "CudaJointLEVTEVSolver"]
