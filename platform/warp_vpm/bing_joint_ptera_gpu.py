"""Strict CUDA numerical backend for the attached V5M Ptera chassis.

This module intentionally implements only the production configuration used by
the Yang, Izraelevitz--Scherer and Mancini validation cases: one airplane, one
wing, no image surface, and ``JointConfig(enable_lev=False, joint_tev=False)``.
Unsupported configurations fail closed instead of falling back to NumPy/Numba.

Python and Ptera still own geometry construction, object lifecycle and result
serialization.  The aerodynamic numerics -- Biot--Savart evaluation, AIC/RHS,
dense solve, force/moment evaluation, reductions, and the LESP ledger -- execute
as float64 Torch CUDA operations.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
from pterasoftware import _vortices

from bing_joint_ptera import JointConfig, JointLEVTEVSolver, LESP_FACTOR

_EPS = float(np.finfo(float).eps)
_TOL = 1.0e-10
_FOUR_LAMB = 5.02572
_SQUIRE = 1.0e-4
_FOUR_PI = 4.0 * math.pi


def _cuda_tensor(value: Any, device: torch.device) -> torch.Tensor:
    """Copy an orchestration-owned array to CUDA float64."""
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
    out = torch.zeros(
        (points.shape[0], strengths.shape[0], 3),
        dtype=torch.float64,
        device=points.device,
    )
    for start, end in ((br, fr), (fr, fl), (fl, bl), (bl, br)):
        out = out + _line_velocity_expanded(
            points, start, end, strengths, core_radii, ages, nu
        )
    return out


class CudaAttachedJointLEVTEVSolver(JointLEVTEVSolver):
    """Attached-flow V5M chassis with all aerodynamic numerics on CUDA."""

    def __init__(
        self,
        unsteady_problem: Any,
        cfg: JointConfig | None = None,
        *,
        device: str = "cuda:0",
    ) -> None:
        selected = cfg or JointConfig(enable_lev=False)
        if selected.enable_lev or selected.joint_tev:
            raise ValueError(
                "CUDA attached chassis requires LEV and joint_tev disabled"
            )
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
            raise ValueError("CudaAttachedJointLEVTEVSolver requires a CUDA device")
        torch.empty(1, dtype=torch.float64, device=self.cuda_device)
        super().__init__(unsteady_problem, selected)
        if not self._prescribed_wake:
            raise ValueError("CUDA attached chassis requires a prescribed wake")
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
        return _ring_velocity_expanded(
            points,
            br,
            fr,
            fl,
            bl,
            _cuda_tensor(self._current_bound_vortex_strengths, self.cuda_device),
            _cuda_tensor(self._currentStackBoundRc0s, self.cuda_device),
            torch.zeros(n, dtype=torch.float64, device=self.cuda_device),
            float(self.current_operating_point.nu),
        ).sum(dim=1)

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
        return _ring_velocity_expanded(
            points,
            br,
            fr,
            fl,
            bl,
            _cuda_tensor(self._current_wake_vortex_strengths, self.cuda_device),
            _cuda_tensor(self._currentStackWakeRc0s, self.cuda_device),
            ages,
            float(self.current_operating_point.nu),
        ).sum(dim=1)

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
        expanded = _ring_velocity_expanded(
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
        normals = _cuda_tensor(self.stackUnitNormals_GP1, d)
        influence = torch.sum(velocity * normals, dim=1)
        self._cuda_wake = influence
        self._currentStackWakeWingInfluences__E = influence.detach().cpu().numpy()
        self.cuda_counters["wake"] += 1

    def _calculate_vortex_strengths(self) -> None:
        d = self.cuda_device
        rhs = -(self._cuda_wake + self._cuda_freestream)
        gamma = torch.linalg.solve(self._cuda_aic, rhs)
        if not bool(torch.isfinite(gamma).all().item()):
            raise RuntimeError("non-finite CUDA bound-vortex solution")
        gamma_cpu = gamma.detach().cpu().numpy()
        self._current_bound_vortex_strengths = gamma_cpu
        self._last_bound = gamma_cpu.copy()
        for index, panel in enumerate(self.panels):
            panel.ring_vortex.strength = float(gamma_cpu[index])

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
                "lesp": lesp.detach().cpu().numpy(),
                "chords": chords.detach().cpu().numpy(),
                "areas": areas.detach().cpu().numpy(),
                "v_rel_st": v_rel_st.detach().cpu().numpy(),
                "v_inf": v_inf.detach().cpu().numpy(),
                "le_now": le.detach().cpu().numpy(),
                "te_now": te.detach().cpu().numpy(),
                "le_prev": (
                    None
                    if not hasattr(self, "_le_points_prev")
                    else _cuda_tensor(self._le_points_prev, d).detach().cpu().numpy()
                ),
                "dt": float(self.delta_time),
            }
        )
        self._tev_solved = None
        self._lev_hist.append(np.zeros(span_count))
        self._steps_done += 1
        circulation = torch.sum(gamma)
        circulation_value = float(circulation.item())
        if self._circ0 is None:
            self._circ0 = circulation_value
        self.diag.append(
            {
                "step": self._current_step,
                "n_particles": 0,
                "lev_strips": 0,
                "lesp_max": float(torch.max(torch.abs(lesp)).item()),
                "g_tev": 0.0,
                "g_lev": 0.0,
                "circ_drift": abs(circulation_value - self._circ0),
            }
        )
        self.cuda_counters["solve"] += 1
        self.cuda_counters["ledger"] += 1

    def _solution_velocity_cuda(self, points: torch.Tensor) -> torch.Tensor:
        v_inf = _cuda_tensor(self._currentVInf_GP1__E, self.cuda_device)
        return self._bound_velocity(points) + self._wake_velocity(points) + v_inf

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
            last = _cuda_tensor(self._last_bound_vortex_strengths, self.cuda_device)
            back[-1, :] = grid[-1, :] - last.reshape(chord_count, span_count)[-1, :]
        return tuple(x.reshape(-1) for x in (right, front, left, back))

    def _calculate_loads(self) -> None:
        d = self.cuda_device
        gamma = _cuda_tensor(self._current_bound_vortex_strengths, d)
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
        forces: list[torch.Tensor] = []
        for points_cpu, previous_cpu, vector_cpu, strength in zip(
            locations, previous, vectors, strengths, strict=True
        ):
            points = _cuda_tensor(points_cpu, d)
            velocity = self._solution_velocity_cuda(points)
            velocity = velocity + self._movement_velocity(points_cpu, previous_cpu)
            vector = _cuda_tensor(vector_cpu, d)
            force = (
                float(self.current_operating_point.rho)
                * strength.unsqueeze(1)
                * torch.linalg.cross(velocity, vector, dim=1)
            )
            forces.append(force)
        last_gamma = _cuda_tensor(self._last_bound_vortex_strengths, d)
        unsteady = -(
            float(self.current_operating_point.rho)
            * (gamma - last_gamma).unsqueeze(1)
            * _cuda_tensor(self.panel_areas, d).unsqueeze(1)
            * _cuda_tensor(self.stackUnitNormals_GP1, d)
            / float(self.delta_time)
        )
        panel_forces = sum(forces, start=torch.zeros_like(unsteady)) + unsteady
        moment_points = tuple(_cuda_tensor(x, d) for x in locations)
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
            record["cn_strip"] = cn.detach().cpu().numpy()
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
                    convected = new_row + v_inf * delta_time
                    grid = torch.cat((new_row, convected), dim=0)
                else:
                    current_grid_host = this_wing.gridWrvp_GP1_CgP1
                    if current_grid_host is None:
                        raise RuntimeError(
                            "current prescribed wake grid is unavailable"
                        )
                    current_grid = _cuda_tensor(current_grid_host, d)
                    convected = current_grid + v_inf * delta_time
                    grid = torch.cat((new_row, convected), dim=0)
                if self._max_wake_rows is not None:
                    grid = grid[: self._max_wake_rows + 1]
                next_wing.gridWrvp_GP1_CgP1 = grid.detach().cpu().numpy()
        self.cuda_counters["wake_convection"] += 1


__all__ = ["CudaAttachedJointLEVTEVSolver"]
