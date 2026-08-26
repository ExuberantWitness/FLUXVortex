"""CUDA Q16 trial-state to Ptera two-state panel geometry adapter.

This is the narrowest honest bridge between the Q16 predictor variables and
the current Ptera solver lifecycle.  It reconstructs a pristine two-state
branch as ``q_previous = q_trial - dt*dq_trial`` and ``q_current = q_trial``.
It deliberately does not claim a reusable incremental time-step owner; a
solver that has already run is rejected.

Q16 interpolation and W-to-GP transforms execute as Torch/Warp CUDA float64.
Ptera ``Panel`` objects remain host-owned orchestration containers, matching
the numerical boundary already declared by ``CudaJointLEVTEVSolver``.
"""

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
from fluxvortex.q16_work_conjugate_transfer import Q16SurfaceTransferMap
from fluxvortex.warp_fsi import config
from fluxvortex.warp_fsi.kernels_q16_transfer import Q16CudaSurfaceTransfer
from fluxvortex.warp_fsi.q16_mandatory_aero_mode import (
    require_q16_mandatory_aero_mode,
)
from pterasoftware import _panel
from q16_incremental_ptera_owner import (
    Q16CudaIncrementalAeroSession,
    Q16IncrementalAeroLifecycleError,
)

_SCHEMA = "flux-v5m-q16-ptera-two-state-kinematics-v1"
_INCREMENTAL_SCHEMA = "flux-v5m-q16-ptera-incremental-geometry-v1"
_INCREMENTAL_MOTION_SCHEMA = "flux-v5m-q16-ptera-incremental-motion-v1"
_SQRT_TINY = math.sqrt(float(np.finfo(np.float64).tiny))


def _sha256_array(domain: str, value: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(value, dtype=np.float64)
    header = json.dumps(
        {
            "domain": domain,
            "dtype": "float64",
            "shape": list(contiguous.shape),
        },
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(header + b"\0" + contiguous.tobytes(order="C")).hexdigest()


def _warp_sha256(name: str, value: Any) -> str:
    if not isinstance(value, wp.array):
        raise TypeError(f"{name} must be a Warp array")
    if not value.device.is_cuda:
        raise ValueError(f"{name} must reside on CUDA")
    if value.dtype != config.DTYPE:
        raise TypeError(f"{name} must use Warp float64")
    if value.ndim != 2 or value.shape[0] != 1 or value.shape[1] <= 0:
        raise ValueError(f"{name} must have shape (1, positive_dof_count)")
    host = value.numpy()
    if not bool(np.isfinite(host).all()):
        raise FloatingPointError(f"{name} contains non-finite values")
    return _sha256_array(name, host)


@dataclass(frozen=True, slots=True)
class Q16PteraPanelVertexTopology:
    """Fixed structured Ptera vertex order owned by one Q16 surface map."""

    chordwise_panel_count: int
    spanwise_panel_count: int

    def __post_init__(self) -> None:
        for name, value in (
            ("chordwise_panel_count", self.chordwise_panel_count),
            ("spanwise_panel_count", self.spanwise_panel_count),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive exact int")

    @property
    def vertex_count(self) -> int:
        return (self.chordwise_panel_count + 1) * (self.spanwise_panel_count + 1)


@dataclass(frozen=True, slots=True)
class Q16PteraTrialKinematicsEvidence:
    schema_id: str
    vertex_count: int
    q_trial_sha256: str
    dq_trial_sha256: str
    previous_vertices_sha256: str
    current_vertices_sha256: str
    velocity_reconstruction_max_abs_error: float
    evidence_sha256: str


@dataclass(frozen=True, slots=True)
class Q16PteraIncrementalGeometryEvidence:
    schema_id: str
    step_index: int
    vertex_count: int
    q_trial_sha256: str
    current_vertices_sha256: str
    panel_grid_sha256: str
    evidence_sha256: str


@dataclass(frozen=True, slots=True)
class Q16PteraIncrementalMotionEvidence:
    schema_id: str
    step_index: int
    vertex_count: int
    q_trial_sha256: str
    dq_trial_sha256: str
    current_vertices_sha256: str
    vertex_velocity_sha256: str
    panel_grid_sha256: str
    motion_shadow_max_abs_error: float
    evidence_sha256: str


def _panel_grid_is_nondegenerate(
    vertices: torch.Tensor,
    topology: Q16PteraPanelVertexTopology,
) -> None:
    grid = vertices.reshape(
        topology.chordwise_panel_count + 1,
        topology.spanwise_panel_count + 1,
        3,
    )
    front_left = grid[:-1, :-1]
    front_right = grid[:-1, 1:]
    back_left = grid[1:, :-1]
    back_right = grid[1:, 1:]
    first = torch.linalg.cross(front_right - front_left, back_left - front_left, dim=2)
    second = torch.linalg.cross(
        back_right - front_right, back_left - front_right, dim=2
    )
    twice_area = torch.linalg.vector_norm(first, dim=2) + torch.linalg.vector_norm(
        second, dim=2
    )
    scale = max(1.0, float(torch.max(torch.abs(vertices)).item()))
    if bool(torch.any(twice_area <= _SQRT_TINY * scale * scale).item()):
        raise ValueError("Q16 trial creates a degenerate Ptera panel")


def _new_panel(
    *,
    front_right: np.ndarray,
    front_left: np.ndarray,
    back_left: np.ndarray,
    back_right: np.ndarray,
    chordwise_index: int,
    spanwise_index: int,
    topology: Q16PteraPanelVertexTopology,
) -> _panel.Panel:
    values = tuple(
        np.ascontiguousarray(value, dtype=np.float64).copy()
        for value in (front_right, front_left, back_left, back_right)
    )
    panel = _panel.Panel(
        Frpp_G_Cg=values[0].copy(),
        Flpp_G_Cg=values[1].copy(),
        Blpp_G_Cg=values[2].copy(),
        Brpp_G_Cg=values[3].copy(),
        is_leading_edge=chordwise_index == 0,
        is_trailing_edge=chordwise_index == topology.chordwise_panel_count - 1,
    )
    panel.Frpp_GP1_CgP1 = values[0].copy()
    panel.Flpp_GP1_CgP1 = values[1].copy()
    panel.Blpp_GP1_CgP1 = values[2].copy()
    panel.Brpp_GP1_CgP1 = values[3].copy()
    panel.is_left_edge = spanwise_index == 0
    panel.is_right_edge = spanwise_index == topology.spanwise_panel_count - 1
    panel.local_chordwise_position = chordwise_index
    panel.local_spanwise_position = spanwise_index
    return panel


class Q16CudaPteraTwoStateKinematics:
    """Populate one pristine two-state real solver from Q16 q/dq on CUDA."""

    __slots__ = ("surface_transfer", "topology")

    def __init__(
        self,
        transfer_map: Q16SurfaceTransferMap,
        topology: Q16PteraPanelVertexTopology,
        *,
        device: str,
    ) -> None:
        if type(transfer_map) is not Q16SurfaceTransferMap:
            raise TypeError("transfer_map must be an exact Q16SurfaceTransferMap")
        if type(topology) is not Q16PteraPanelVertexTopology:
            raise TypeError("topology must be an exact Q16PteraPanelVertexTopology")
        if transfer_map.point_count != topology.vertex_count:
            raise ValueError("surface-map point count differs from panel vertex count")
        self.surface_transfer = Q16CudaSurfaceTransfer(transfer_map, device=device)
        self.topology = topology

    def _require_pristine_solver(self, solver: Any) -> CudaJointLEVTEVSolver:
        if type(solver) is not CudaJointLEVTEVSolver:
            raise TypeError("solver must be an exact CudaJointLEVTEVSolver")
        require_q16_mandatory_aero_mode(solver)
        if solver.num_steps != 2:
            raise ValueError("Q16 trial requires exactly two aerodynamic states")
        if (
            solver.ran is not False
            or solver._steps_done != 0
            or solver.lev_pf.n != 0
            or len(solver.ledger) != 0
            or solver._last_impulse_cuda is not None
        ):
            raise RuntimeError("Q16 kinematics requires a pristine solver branch")
        for problem in solver.steady_problems:
            if len(problem.airplanes) != 1 or len(problem.airplanes[0].wings) != 1:
                raise ValueError("Q16 trial supports one airplane and one wing")
            wing = problem.airplanes[0].wings[0]
            panels = wing.panels
            if panels is None or panels.shape != (
                self.topology.chordwise_panel_count,
                self.topology.spanwise_panel_count,
            ):
                raise ValueError("Ptera panel topology differs from Q16 panel topology")
        return solver

    def apply(
        self,
        solver: CudaJointLEVTEVSolver,
        q_trial: wp.array,
        dq_trial: wp.array,
    ) -> Q16PteraTrialKinematicsEvidence:
        branch = self._require_pristine_solver(solver)
        q_sha = _warp_sha256("q_trial", q_trial)
        dq_sha = _warp_sha256("dq_trial", dq_trial)
        if q_trial.shape != dq_trial.shape:
            raise ValueError("q_trial and dq_trial shapes differ")
        if q_trial.device.alias != dq_trial.device.alias:
            raise ValueError("q_trial and dq_trial devices differ")

        current_w = wp.to_torch(self.surface_transfer.interpolate(q_trial))[0]
        velocity_w = wp.to_torch(self.surface_transfer.interpolate(dq_trial))[0]
        delta_time = float(branch.delta_time)
        if not math.isfinite(delta_time) or delta_time <= 0.0:
            raise ValueError("aerodynamic delta_time must be finite and positive")
        previous_w = current_w - delta_time * velocity_w
        _panel_grid_is_nondegenerate(previous_w, self.topology)
        _panel_grid_is_nondegenerate(current_w, self.topology)

        scientific_states = (previous_w, current_w)
        gp_states: list[np.ndarray] = []
        for scientific in scientific_states:
            gp = branch._v5m_scientific_points_to_gp_cuda(scientific)
            if not bool(torch.isfinite(gp).all().item()):
                raise FloatingPointError(
                    "V5M-scientific-to-GP panel transform became non-finite"
                )
            gp_states.append(gp.detach().contiguous().cpu().numpy())

        chord_count = self.topology.chordwise_panel_count
        span_count = self.topology.spanwise_panel_count
        for step, flat_vertices in enumerate(gp_states):
            vertices = flat_vertices.reshape(chord_count + 1, span_count + 1, 3)
            panels = np.empty((chord_count, span_count), dtype=object)
            for chord in range(chord_count):
                for span in range(span_count):
                    panels[chord, span] = _new_panel(
                        front_right=vertices[chord, span + 1],
                        front_left=vertices[chord, span],
                        back_left=vertices[chord + 1, span],
                        back_right=vertices[chord + 1, span + 1],
                        chordwise_index=chord,
                        spanwise_index=span,
                        topology=self.topology,
                    )
            wing = branch.steady_problems[step].airplanes[0].wings[0]
            # Ptera intentionally exposes no remeshing setter after construction.
            # This solver is an isolated pickle branch, so replace the single
            # panel-array owner atomically rather than mutating any Panel's
            # set-once coordinates or derived caches in place.
            object.__setattr__(wing, "_panels", panels)
            if wing.panels is not panels or wing.panels.shape != panels.shape:
                raise RuntimeError("Ptera panel owner replacement did not take effect")

        if (
            _warp_sha256("q_trial", q_trial) != q_sha
            or _warp_sha256("dq_trial", dq_trial) != dq_sha
        ):
            raise RuntimeError("Q16 trial state drifted during Ptera reconstruction")
        require_q16_mandatory_aero_mode(branch)

        previous_host = previous_w.detach().contiguous().cpu().numpy()
        current_host = current_w.detach().contiguous().cpu().numpy()
        reconstructed = (current_w - previous_w) / delta_time
        velocity_error = float(torch.max(torch.abs(reconstructed - velocity_w)).item())
        previous_sha = _sha256_array("previous_vertices_w", previous_host)
        current_sha = _sha256_array("current_vertices_w", current_host)
        payload = json.dumps(
            {
                "schema_id": _SCHEMA,
                "vertex_count": self.topology.vertex_count,
                "q_trial_sha256": q_sha,
                "dq_trial_sha256": dq_sha,
                "previous_vertices_sha256": previous_sha,
                "current_vertices_sha256": current_sha,
                "velocity_reconstruction_max_abs_error_hex": velocity_error.hex(),
            },
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
        return Q16PteraTrialKinematicsEvidence(
            schema_id=_SCHEMA,
            vertex_count=self.topology.vertex_count,
            q_trial_sha256=q_sha,
            dq_trial_sha256=dq_sha,
            previous_vertices_sha256=previous_sha,
            current_vertices_sha256=current_sha,
            velocity_reconstruction_max_abs_error=velocity_error,
            evidence_sha256=hashlib.sha256(payload).hexdigest(),
        )


def _installed_world_vertices(
    solver: CudaJointLEVTEVSolver,
    step: int,
    topology: Q16PteraPanelVertexTopology,
) -> np.ndarray:
    problem = solver.steady_problems[step]
    panels = problem.airplanes[0].wings[0].panels
    if panels is None or panels.shape != (
        topology.chordwise_panel_count,
        topology.spanwise_panel_count,
    ):
        raise ValueError("Ptera panel topology differs from Q16 panel topology")
    chord_count, span_count = panels.shape
    vertices = np.empty((chord_count + 1, span_count + 1, 3), dtype=np.float64)
    for chord in range(chord_count):
        for span in range(span_count):
            panel = panels[chord, span]
            vertices[chord, span] = panel.Flpp_GP1_CgP1
            vertices[chord, span + 1] = panel.Frpp_GP1_CgP1
            vertices[chord + 1, span] = panel.Blpp_GP1_CgP1
            vertices[chord + 1, span + 1] = panel.Brpp_GP1_CgP1
    scientific = solver._v5m_gp_points_to_scientific_cuda(
        torch.as_tensor(
            np.ascontiguousarray(vertices.reshape(-1, 3)),
            device=solver.cuda_device,
            dtype=torch.float64,
        )
    )
    if not bool(torch.isfinite(scientific).all().item()):
        raise FloatingPointError("installed V5M scientific vertices are non-finite")
    return np.ascontiguousarray(scientific.detach().cpu().numpy())


class Q16CudaPteraIncrementalGeometry:
    """Bind one trusted incremental branch's next geometry from Q16 q."""

    __slots__ = ("surface_transfer", "topology")

    def __init__(
        self,
        transfer_map: Q16SurfaceTransferMap,
        topology: Q16PteraPanelVertexTopology,
        *,
        device: str,
    ) -> None:
        if type(transfer_map) is not Q16SurfaceTransferMap:
            raise TypeError("transfer_map must be an exact Q16SurfaceTransferMap")
        if type(topology) is not Q16PteraPanelVertexTopology:
            raise TypeError("topology must be an exact Q16PteraPanelVertexTopology")
        if transfer_map.point_count != topology.vertex_count:
            raise ValueError("surface-map point count differs from panel vertex count")
        self.surface_transfer = Q16CudaSurfaceTransfer(transfer_map, device=device)
        self.topology = topology

    def current_vertices_sha256(
        self,
        solver: CudaJointLEVTEVSolver,
        step: int,
    ) -> str:
        if type(solver) is not CudaJointLEVTEVSolver:
            raise TypeError("solver must be an exact CudaJointLEVTEVSolver")
        if type(step) is not int or not 0 <= step < solver.num_steps:
            raise ValueError("step is outside the aerodynamic trajectory")
        return _sha256_array(
            "incremental_current_vertices_w",
            _installed_world_vertices(solver, step, self.topology),
        )

    def bind_next(
        self,
        session: Q16CudaIncrementalAeroSession,
        q_trial: wp.array,
    ) -> Q16PteraIncrementalGeometryEvidence:
        if type(session) is not Q16CudaIncrementalAeroSession:
            raise TypeError("session must be an exact Q16CudaIncrementalAeroSession")
        step = session.next_step
        solver = session.solver
        if step >= solver.num_steps:
            raise Q16IncrementalAeroLifecycleError("no remaining aerodynamic steps")
        q_sha = _warp_sha256("q_trial", q_trial)
        current_w = wp.to_torch(self.surface_transfer.interpolate(q_trial))[0]
        _panel_grid_is_nondegenerate(current_w, self.topology)
        problem = solver.steady_problems[step]
        if len(problem.airplanes) != 1 or len(problem.airplanes[0].wings) != 1:
            raise ValueError("Q16 incremental geometry supports one airplane and wing")
        existing = problem.airplanes[0].wings[0].panels
        expected_shape = (
            self.topology.chordwise_panel_count,
            self.topology.spanwise_panel_count,
        )
        if existing is None or existing.shape != expected_shape:
            raise ValueError("Ptera panel topology differs from Q16 panel topology")
        gp = solver._v5m_scientific_points_to_gp_cuda(current_w)
        if not bool(torch.isfinite(gp).all().item()):
            raise FloatingPointError(
                "V5M-scientific-to-GP panel transform became non-finite"
            )
        vertices = (
            gp.detach()
            .contiguous()
            .cpu()
            .numpy()
            .reshape(
                self.topology.chordwise_panel_count + 1,
                self.topology.spanwise_panel_count + 1,
                3,
            )
        )
        panels = np.empty(expected_shape, dtype=object)
        for chord in range(self.topology.chordwise_panel_count):
            for span in range(self.topology.spanwise_panel_count):
                panels[chord, span] = _new_panel(
                    front_right=vertices[chord, span + 1],
                    front_left=vertices[chord, span],
                    back_left=vertices[chord + 1, span],
                    back_right=vertices[chord + 1, span + 1],
                    chordwise_index=chord,
                    spanwise_index=span,
                    topology=self.topology,
                )
        panel_sha = session.bind_next_panel_grid(panels)
        if _warp_sha256("q_trial", q_trial) != q_sha:
            raise RuntimeError("Q16 trial state drifted during geometry binding")
        current_sha = self.current_vertices_sha256(solver, step)
        payload = json.dumps(
            {
                "schema_id": _INCREMENTAL_SCHEMA,
                "step_index": step,
                "vertex_count": self.topology.vertex_count,
                "q_trial_sha256": q_sha,
                "current_vertices_sha256": current_sha,
                "panel_grid_sha256": panel_sha,
            },
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
        return Q16PteraIncrementalGeometryEvidence(
            schema_id=_INCREMENTAL_SCHEMA,
            step_index=step,
            vertex_count=self.topology.vertex_count,
            q_trial_sha256=q_sha,
            current_vertices_sha256=current_sha,
            panel_grid_sha256=panel_sha,
            evidence_sha256=hashlib.sha256(payload).hexdigest(),
        )

    def bind_next_state(
        self,
        session: Q16CudaIncrementalAeroSession,
        q_trial: wp.array,
        dq_trial: wp.array,
    ) -> Q16PteraIncrementalMotionEvidence:
        """Bind current Q16 position and Newmark endpoint velocity together."""

        if type(session) is not Q16CudaIncrementalAeroSession:
            raise TypeError("session must be an exact Q16CudaIncrementalAeroSession")
        step = session.next_step
        solver = session.solver
        if step >= solver.num_steps:
            raise Q16IncrementalAeroLifecycleError("no remaining aerodynamic steps")
        q_sha = _warp_sha256("q_trial", q_trial)
        dq_sha = _warp_sha256("dq_trial", dq_trial)
        if q_trial.shape != dq_trial.shape:
            raise ValueError("q_trial and dq_trial shapes differ")
        if q_trial.device.alias != dq_trial.device.alias:
            raise ValueError("q_trial and dq_trial devices differ")

        current_w = wp.to_torch(self.surface_transfer.interpolate(q_trial))[0]
        velocity_w = wp.to_torch(self.surface_transfer.interpolate(dq_trial))[0]
        _panel_grid_is_nondegenerate(current_w, self.topology)
        if not bool(torch.isfinite(velocity_w).all().item()):
            raise FloatingPointError("Q16 endpoint surface velocity is non-finite")
        if step == 0 and bool(torch.any(velocity_w != 0.0).item()):
            raise ValueError("initial endpoint surface velocity must be exact zero")
        problem = solver.steady_problems[step]
        if len(problem.airplanes) != 1 or len(problem.airplanes[0].wings) != 1:
            raise ValueError("Q16 incremental motion supports one airplane and wing")
        existing = problem.airplanes[0].wings[0].panels
        expected_shape = (
            self.topology.chordwise_panel_count,
            self.topology.spanwise_panel_count,
        )
        if existing is None or existing.shape != expected_shape:
            raise ValueError("Ptera panel topology differs from Q16 panel topology")
        current_gp = solver._v5m_scientific_points_to_gp_cuda(current_w)
        velocity_gp = solver._v5m_scientific_vectors_to_gp_cuda(velocity_w)
        if not bool(
            torch.isfinite(current_gp).all().item()
            and torch.isfinite(velocity_gp).all().item()
        ):
            raise FloatingPointError("Q16 endpoint GP kinematics became non-finite")
        delta_time = float(solver.delta_time)
        if not math.isfinite(delta_time) or delta_time <= 0.0:
            raise ValueError("aerodynamic delta_time must be finite and positive")
        shadow_gp = current_gp - delta_time * velocity_gp
        reconstructed = (current_gp - shadow_gp) / delta_time
        motion_error = float(torch.max(torch.abs(reconstructed - velocity_gp)).item())
        motion_scale = max(1.0, float(torch.max(torch.abs(velocity_gp)).item()))
        if motion_error > 128.0 * float(np.finfo(np.float64).eps) * motion_scale:
            raise FloatingPointError("endpoint motion shadow reconstruction drift")

        chord_count = self.topology.chordwise_panel_count
        span_count = self.topology.spanwise_panel_count
        vertices = (
            current_gp.detach()
            .contiguous()
            .cpu()
            .numpy()
            .reshape(chord_count + 1, span_count + 1, 3)
        )
        velocity_vertices = (
            velocity_gp.detach()
            .contiguous()
            .cpu()
            .numpy()
            .reshape(chord_count + 1, span_count + 1, 3)
        )
        panels = np.empty(expected_shape, dtype=object)
        for chord in range(chord_count):
            for span in range(span_count):
                panels[chord, span] = _new_panel(
                    front_right=vertices[chord, span + 1],
                    front_left=vertices[chord, span],
                    back_left=vertices[chord + 1, span],
                    back_right=vertices[chord + 1, span + 1],
                    chordwise_index=chord,
                    spanwise_index=span,
                    topology=self.topology,
                )
        panel_sha, velocity_sha = session.bind_next_panel_kinematics(
            panels,
            velocity_vertices,
        )
        if (
            _warp_sha256("q_trial", q_trial) != q_sha
            or _warp_sha256("dq_trial", dq_trial) != dq_sha
        ):
            raise RuntimeError("Q16 endpoint state drifted during motion binding")
        current_sha = self.current_vertices_sha256(solver, step)
        payload = json.dumps(
            {
                "schema_id": _INCREMENTAL_MOTION_SCHEMA,
                "step_index": step,
                "vertex_count": self.topology.vertex_count,
                "q_trial_sha256": q_sha,
                "dq_trial_sha256": dq_sha,
                "current_vertices_sha256": current_sha,
                "vertex_velocity_sha256": velocity_sha,
                "panel_grid_sha256": panel_sha,
                "motion_shadow_max_abs_error_hex": motion_error.hex(),
            },
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
        return Q16PteraIncrementalMotionEvidence(
            schema_id=_INCREMENTAL_MOTION_SCHEMA,
            step_index=step,
            vertex_count=self.topology.vertex_count,
            q_trial_sha256=q_sha,
            dq_trial_sha256=dq_sha,
            current_vertices_sha256=current_sha,
            vertex_velocity_sha256=velocity_sha,
            panel_grid_sha256=panel_sha,
            motion_shadow_max_abs_error=motion_error,
            evidence_sha256=hashlib.sha256(payload).hexdigest(),
        )


__all__ = [
    "Q16CudaPteraIncrementalGeometry",
    "Q16CudaPteraTwoStateKinematics",
    "Q16PteraIncrementalGeometryEvidence",
    "Q16PteraIncrementalMotionEvidence",
    "Q16PteraPanelVertexTopology",
    "Q16PteraTrialKinematicsEvidence",
]
