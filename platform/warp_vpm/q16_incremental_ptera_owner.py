"""Incremental owner for the real CUDA separated-LEV Ptera solver.

The upstream Ptera ``run`` method combines trajectory allocation, vortex
initialization, every numerical step and finalization in one loop.  This module
separates only that orchestration.  Each advance calls the same CUDA AIC, wake,
joint LEV/TEV solve, load and wake-convection methods as the existing solver;
no aerodynamic equation is duplicated here.

The state machine is intentionally Q16-specific and therefore rejects every
reduced aerodynamic mode.  A candidate branch may bind both its next Panel
owner and a detached Q16 endpoint-velocity grid.  The latter supplies one
consistent motion shadow to the collocation, LE/TE, bound-vortex and load-point
finite differences without rewriting the committed previous geometry.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import pickle
from typing import Any

import numpy as np
import torch

from bing_joint_ptera_gpu import CudaJointLEVTEVSolver
from fluxvortex.warp_fsi.q16_aero_load_packet import Q16CudaAerodynamicLoadPacket
from fluxvortex.warp_fsi.q16_mandatory_aero_mode import (
    require_q16_mandatory_aero_mode,
)
from pterasoftware import _panel, _vortices

_RECEIPT_SCHEMA = "flux-v5m-q16-incremental-aero-step-v1"
_NEXT_STEP_ATTRIBUTE = "_q16_incremental_next_step"
_STATE_SHA_ATTRIBUTE = "_q16_incremental_state_sha256"
_BOUND_STEP_ATTRIBUTE = "_q16_incremental_bound_geometry_step"
_BOUND_VELOCITY_ATTRIBUTE = "_q16_incremental_bound_vertex_velocity_gp"
_BOUND_VELOCITY_SHA_ATTRIBUTE = "_q16_incremental_bound_vertex_velocity_sha256"
_MARKER_ATTRIBUTES = (
    _NEXT_STEP_ATTRIBUTE,
    _STATE_SHA_ATTRIBUTE,
    _BOUND_STEP_ATTRIBUTE,
    _BOUND_VELOCITY_ATTRIBUTE,
    _BOUND_VELOCITY_SHA_ATTRIBUTE,
)


class Q16IncrementalAeroLifecycleError(RuntimeError):
    """Invalid or drifted incremental aerodynamic lifecycle."""


def _semantic_solver_sha256(solver: CudaJointLEVTEVSolver) -> str:
    """Hash the exact solver state while excluding this module's own markers."""

    torch.cuda.synchronize(solver.cuda_device)
    markers: dict[str, Any] = {}
    for name in _MARKER_ATTRIBUTES:
        if hasattr(solver, name):
            markers[name] = getattr(solver, name)
            delattr(solver, name)
    try:
        payload = pickle.dumps(solver, protocol=5)
    finally:
        for name, value in markers.items():
            setattr(solver, name, value)
    return hashlib.sha256(payload).hexdigest()


def _update_digest_value(digest: Any, value: Any) -> None:
    if value is None:
        digest.update(b"none\0")
    elif type(value) is bool:
        digest.update(b"bool\0" + (b"1" if value else b"0"))
    elif type(value) is int:
        digest.update(b"int\0" + str(value).encode("ascii") + b"\0")
    elif type(value) is float:
        digest.update(b"float\0" + value.hex().encode("ascii") + b"\0")
    elif type(value) is str:
        encoded = value.encode("utf-8")
        digest.update(b"str\0" + str(len(encoded)).encode("ascii") + b"\0" + encoded)
    elif isinstance(value, np.generic):
        _update_digest_value(digest, value.item())
    elif type(value) is np.ndarray:
        contiguous = np.ascontiguousarray(value)
        digest.update(
            b"numpy\0"
            + str(contiguous.dtype).encode("ascii")
            + b"\0"
            + json.dumps(list(contiguous.shape), separators=(",", ":")).encode("ascii")
            + b"\0"
            + contiguous.tobytes(order="C")
        )
    elif type(value) is torch.Tensor:
        contiguous = value.detach().contiguous().cpu()
        digest.update(
            b"torch\0"
            + str(contiguous.dtype).encode("ascii")
            + b"\0"
            + json.dumps(list(contiguous.shape), separators=(",", ":")).encode("ascii")
            + b"\0"
            + contiguous.numpy().tobytes(order="C")
        )
    elif type(value) in (list, tuple):
        digest.update(b"sequence\0" + str(len(value)).encode("ascii") + b"\0")
        for item in value:
            _update_digest_value(digest, item)
    elif type(value) is dict:
        keys = sorted(value)
        if not all(type(key) is str for key in keys):
            raise TypeError("scientific-state dictionary keys must be strings")
        digest.update(b"dict\0" + str(len(keys)).encode("ascii") + b"\0")
        for key in keys:
            _update_digest_value(digest, key)
            _update_digest_value(digest, value[key])
    else:
        raise TypeError(f"unsupported scientific-state value {type(value)!r}")


def _ring_payload(ring: Any) -> dict[str, Any] | None:
    if ring is None:
        return None
    return {
        "front_left": np.asarray(ring.Flrvp_GP1_CgP1, dtype=np.float64),
        "front_right": np.asarray(ring.Frrvp_GP1_CgP1, dtype=np.float64),
        "back_left": np.asarray(ring.Blrvp_GP1_CgP1, dtype=np.float64),
        "back_right": np.asarray(ring.Brrvp_GP1_CgP1, dtype=np.float64),
        "strength": float(ring.strength),
        "age": float(ring.age),
    }


def _scientific_solver_sha256(solver: CudaJointLEVTEVSolver) -> str:
    """Canonical cross-pickle digest of state that determines later advances."""

    torch.cuda.synchronize(solver.cuda_device)
    particle = solver.lev_pf
    state: dict[str, Any] = {
        "domain": "flux-v5m-q16-incremental-aero-scientific-state-v1",
        "num_steps": solver.num_steps,
        "delta_time": float(solver.delta_time),
        "first_results_step": solver.first_results_step,
        "max_wake_rows": solver._max_wake_rows,
        "current_step": solver._current_step,
        "steps_done": solver._steps_done,
        "prescribed_wake": solver._prescribed_wake,
        "v5m_scientific_frame": {
            "schema": solver.v5m_scientific_frame_schema,
            "gp_to_scientific_rotation": (
                solver._v5m_gp_to_scientific_rotation_cuda
            ),
            "gp_to_scientific_translation": (
                solver._v5m_gp_to_scientific_translation_cuda
            ),
            "verified_step": getattr(
                solver, "_v5m_scientific_surface_contract_step", None
            ),
            "downstream_wake_verified_step": getattr(
                solver, "_v5m_downstream_wake_contract_step", None
            ),
        },
        "config": vars(solver.jcfg).copy(),
        "cuda_counters": solver.cuda_counters.copy(),
        "particle": {
            "capacity": particle.capacity,
            "n": particle.n,
            "source_chunk_size": particle.source_chunk_size,
            "target_chunk_size": particle.target_chunk_size,
            "kernel_calls": particle.kernel_calls,
            "position": particle.pos[: particle.n],
            "gamma": particle.gamma[: particle.n],
            "sigma": particle.sigma[: particle.n],
            "circul": particle.circul[: particle.n],
            "volume": particle.vol[: particle.n],
            "type": particle.ptype[: particle.n],
            "birth_step": particle.birth_step[: particle.n],
            "source_strip": particle.source_strip[: particle.n],
        },
        "lev_history": solver._lev_hist,
        "tev_history": solver._tev_hist,
        "ledger": solver.ledger,
        "diagnostics": solver.diag,
        "circ0": solver._circ0,
        "last_bound": solver._last_bound,
        "last_impulse": solver._last_impulse_cuda,
        "last_impulse_strip": solver._last_impulse_strip_cuda,
        "lev_history_total": solver._cuda_lev_history_total,
        "tev_history_total": solver._cuda_tev_history_total,
        "wake_strength_lists": solver._list_wake_vortex_strengths,
        "wake_age_lists": solver._list_wake_vortex_ages,
        "wake_rc0_lists": solver._list_wake_rc0s,
        "wake_br_lists": solver.listStackBrwrvp_GP1_CgP1,
        "wake_fr_lists": solver.listStackFrwrvp_GP1_CgP1,
        "wake_fl_lists": solver.listStackFlwrvp_GP1_CgP1,
        "wake_bl_lists": solver.listStackBlwrvp_GP1_CgP1,
        "wake_vertex_velocity_grids": (
            solver._q16_cuda_wake_vertex_velocity_grids_gp
        ),
    }
    source_bank = getattr(solver, "dvm_source_bank", None)
    if source_bank is None:
        state["dvm_source_bank"] = None
    else:
        source_state: dict[str, Any] = {}
        for name, value in vars(source_bank).items():
            if name == "cuda_stream":
                continue
            if type(value) in (torch.device, torch.dtype):
                source_state[name] = str(value)
            else:
                source_state[name] = value
        state["dvm_source_bank"] = source_state
    for name in (
        "_cuda_dvm_reference_speed",
        "_cuda_dvm_reference_cell_chord",
        "_cuda_dvm_reference_node_chord",
        "_cuda_dvm_alpha_previous",
        "_cuda_dvm_frontier_nodes",
        "_cuda_dvm_frontier_active",
        "_cuda_dvm_frontier_ever",
        "_cuda_dvm_advected_frontier",
        "_cuda_dvm_last_source",
        "_cuda_dvm_last_newborn_normal_influence",
        "_cuda_dvm_last_source_a0",
        "_cuda_dvm_last_ptera_lesp",
        "_cuda_dvm_last_ptera_pin_active",
        "_cuda_dvm_last_retained_neumann_residual",
        "_cuda_dvm_last_lesp_pin_residual",
    ):
        state[name] = getattr(solver, name, None)
    for name in (
        "_cuda_bound_strengths",
        "_cuda_particle_bound_strengths",
        "_cuda_last_bound_strengths_for_loads",
        "_tev_solved",
        "_q16_resolved_load_points_w",
        "_q16_resolved_load_forces_w",
        "_q16_unresolved_impulse_force_w",
        "_q16_total_force_w",
        "_q16_total_moment_w",
        "_q16_impulse_strip_force_w",
        "_q16_impulse_strip_le_endpoints_w",
        "_diagnostic_vortex_impulse_force_w",
        "_diagnostic_vortex_impulse_strip_force_w",
    ):
        state[name] = getattr(solver, name, None)
    problem_rows: list[dict[str, Any]] = []
    for problem in solver.steady_problems:
        airplane_rows: list[dict[str, Any]] = []
        for airplane in problem.airplanes:
            wing_rows: list[dict[str, Any]] = []
            for wing in airplane.wings:
                panels = wing.panels
                if panels is None:
                    raise ValueError("missing Panel grid in scientific state")
                panel_rows: list[dict[str, Any]] = []
                for panel in panels.flat:
                    panel_rows.append(
                        {
                            "corners": np.stack(
                                (
                                    panel.Frpp_GP1_CgP1,
                                    panel.Flpp_GP1_CgP1,
                                    panel.Blpp_GP1_CgP1,
                                    panel.Brpp_GP1_CgP1,
                                )
                            ),
                            "ring": _ring_payload(panel.ring_vortex),
                        }
                    )
                wake_rows: list[dict[str, Any] | None] = []
                if wing.wake_ring_vortices is not None:
                    wake_rows = [
                        _ring_payload(ring) for ring in wing.wake_ring_vortices.flat
                    ]
                wing_rows.append(
                    {
                        "panels": panel_rows,
                        "wake_grid": wing.gridWrvp_GP1_CgP1,
                        "wake_rings": wake_rows,
                    }
                )
            airplane_rows.append(
                {
                    "wings": wing_rows,
                    "forces": airplane.forces_W,
                    "moments": airplane.moments_W_CgP1,
                    "force_coefficients": airplane.forceCoefficients_W,
                    "moment_coefficients": airplane.momentCoefficients_W_CgP1,
                }
            )
        problem_rows.append(
            {
                "v_inf": problem.operating_point.vInf_GP1__E,
                "transform": problem.operating_point.T_pas_GP1_CgP1_to_W_CgP1,
                "airplanes": airplane_rows,
            }
        )
    state["problems"] = problem_rows
    digest = hashlib.sha256()
    _update_digest_value(digest, state)
    return digest.hexdigest()


def _preallocate_wake_arrays(solver: CudaJointLEVTEVSolver) -> None:
    lists = (
        solver.list_num_wake_vortices,
        solver._list_wake_vortex_strengths,
        solver._list_wake_vortex_ages,
        solver._list_wake_rc0s,
        solver.listStackBrwrvp_GP1_CgP1,
        solver.listStackFrwrvp_GP1_CgP1,
        solver.listStackFlwrvp_GP1_CgP1,
        solver.listStackBlwrvp_GP1_CgP1,
    )
    if any(len(value) != 0 for value in lists):
        raise Q16IncrementalAeroLifecycleError(
            "incremental wake arrays were already initialized"
        )
    for step, problem in enumerate(solver.steady_problems):
        span_count = 0
        for airplane in problem.airplanes:
            for wing in airplane.wings:
                wing_span_count = wing.num_spanwise_panels
                if type(wing_span_count) is not int or wing_span_count <= 0:
                    raise ValueError("invalid incremental wake span topology")
                span_count += wing_span_count
        row_count = step
        if solver._max_wake_rows is not None:
            row_count = min(step, solver._max_wake_rows)
        vortex_count = row_count * span_count
        solver.list_num_wake_vortices.append(vortex_count)
        solver._list_wake_vortex_strengths.append(
            np.zeros(vortex_count, dtype=np.float64)
        )
        solver._list_wake_vortex_ages.append(np.zeros(vortex_count, dtype=np.float64))
        solver._list_wake_rc0s.append(np.zeros(vortex_count, dtype=np.float64))
        for target in (
            solver.listStackBrwrvp_GP1_CgP1,
            solver.listStackFrwrvp_GP1_CgP1,
            solver.listStackFlwrvp_GP1_CgP1,
            solver.listStackBlwrvp_GP1_CgP1,
        ):
            target.append(np.zeros((vortex_count, 3), dtype=np.float64))


def _panel_grid_sha256(panels: np.ndarray) -> str:
    digest = hashlib.sha256(b"flux-v5m-q16-incremental-panel-grid-v1\0")
    digest.update(json.dumps(list(panels.shape), separators=(",", ":")).encode("ascii"))
    for panel in panels.flat:
        for point in (
            panel.Frpp_GP1_CgP1,
            panel.Flpp_GP1_CgP1,
            panel.Blpp_GP1_CgP1,
            panel.Brpp_GP1_CgP1,
        ):
            digest.update(np.ascontiguousarray(point, dtype=np.float64).tobytes())
    return digest.hexdigest()


def _vertex_velocity_sha256(velocity: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(velocity, dtype=np.float64)
    digest = hashlib.sha256(b"flux-v5m-q16-endpoint-vertex-velocity-gp-v1\0")
    digest.update(
        json.dumps(list(contiguous.shape), separators=(",", ":")).encode("ascii")
    )
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def _validate_detached_vertex_velocity(
    velocity: Any,
    expected_panel_shape: tuple[int, int],
) -> np.ndarray:
    chord_count, span_count = expected_panel_shape
    expected_shape = (chord_count + 1, span_count + 1, 3)
    if type(velocity) is not np.ndarray:
        raise TypeError("endpoint vertex velocity must be an exact ndarray")
    if velocity.dtype != np.float64 or velocity.shape != expected_shape:
        raise ValueError("endpoint vertex velocity must be a float64 panel-vertex grid")
    if not bool(np.isfinite(velocity).all()):
        raise FloatingPointError("endpoint vertex velocity contains non-finite values")
    detached = np.ascontiguousarray(velocity).copy()
    detached.flags.writeable = False
    if _vertex_velocity_sha256(detached) != _vertex_velocity_sha256(velocity):
        raise RuntimeError("detached endpoint vertex velocity drift")
    return detached


def _validate_detached_panel_grid(
    panels: Any,
    expected_shape: tuple[int, int],
) -> np.ndarray:
    if type(panels) is not np.ndarray or panels.dtype != object:
        raise TypeError("next Panel grid must be an exact object ndarray")
    if tuple(panels.shape) != expected_shape:
        raise ValueError("next Panel grid differs from aerodynamic panel topology")
    chord_count, span_count = expected_shape
    for chord in range(chord_count):
        for span in range(span_count):
            panel = panels[chord, span]
            if type(panel) is not _panel.Panel:
                raise TypeError("next Panel grid contains a non-exact Panel")
            if panel.ring_vortex is not None:
                raise ValueError("next Panel must not carry a precomputed ring vortex")
            if (
                panel.is_leading_edge is not (chord == 0)
                or panel.is_trailing_edge is not (chord == chord_count - 1)
                or panel.is_left_edge is not (span == 0)
                or panel.is_right_edge is not (span == span_count - 1)
                or panel.local_chordwise_position != chord
                or panel.local_spanwise_position != span
            ):
                raise ValueError("next Panel topology metadata drift")
            points = (
                panel.Frpp_GP1_CgP1,
                panel.Flpp_GP1_CgP1,
                panel.Blpp_GP1_CgP1,
                panel.Brpp_GP1_CgP1,
            )
            if any(
                type(point) is not np.ndarray
                or point.dtype != np.float64
                or point.shape != (3,)
                or not bool(np.isfinite(point).all())
                for point in points
            ):
                raise ValueError(
                    "next Panel coordinates must be finite float64 triples"
                )
            if not np.isfinite(panel.area) or panel.area <= 0.0:
                raise ValueError("next Panel must have finite positive area")
            if span + 1 < span_count:
                right = panels[chord, span + 1]
                if not np.array_equal(
                    panel.Frpp_GP1_CgP1, right.Flpp_GP1_CgP1
                ) or not np.array_equal(panel.Brpp_GP1_CgP1, right.Blpp_GP1_CgP1):
                    raise ValueError("next Panel grid is discontinuous spanwise")
            if chord + 1 < chord_count:
                back = panels[chord + 1, span]
                if not np.array_equal(
                    panel.Blpp_GP1_CgP1, back.Flpp_GP1_CgP1
                ) or not np.array_equal(panel.Brpp_GP1_CgP1, back.Frpp_GP1_CgP1):
                    raise ValueError("next Panel grid is discontinuous chordwise")
    detached = pickle.loads(pickle.dumps(panels, protocol=5))
    if type(detached) is not np.ndarray or detached.dtype != object:
        raise RuntimeError("detached Panel grid changed exact container type")
    if _panel_grid_sha256(detached) != _panel_grid_sha256(panels):
        raise RuntimeError("detached Panel grid content drift")
    return detached


def _initialize_panel_vortices_for_step(
    solver: CudaJointLEVTEVSolver,
    step: int,
    endpoint_vertex_velocity_gp: np.ndarray | None = None,
) -> None:
    """Mirror Ptera's initialization formula for exactly one mutable step."""

    problem = solver.steady_problems[step]
    v_inf = problem.operating_point.vInf_GP1__E
    for airplane_index, airplane in enumerate(problem.airplanes):
        for wing_index, wing in enumerate(airplane.wings):
            span_count = wing.num_spanwise_panels
            if type(span_count) is not int or span_count <= 0:
                raise ValueError("invalid Panel span topology")
            panels = wing.panels
            if panels is None:
                raise ValueError("missing incremental Panel grid")
            for chord in range(wing.num_chordwise_panels):
                for span in range(span_count):
                    panel = panels[chord, span]
                    front_left = panel.Flbvp_GP1_CgP1
                    front_right = panel.Frbvp_GP1_CgP1
                    if front_left is None or front_right is None:
                        raise ValueError("missing bound-vortex front point")
                    if not panel.is_trailing_edge:
                        next_panel = panels[chord + 1, span]
                        back_left = next_panel.Flbvp_GP1_CgP1
                        back_right = next_panel.Frbvp_GP1_CgP1
                    elif step == 0:
                        back_left = (
                            panel.Blpp_GP1_CgP1
                            + v_inf * float(solver.delta_time) * 0.25
                        )
                        back_right = (
                            panel.Brpp_GP1_CgP1
                            + v_inf * float(solver.delta_time) * 0.25
                        )
                    else:
                        if endpoint_vertex_velocity_gp is None:
                            previous_wing = (
                                solver.steady_problems[step - 1]
                                .airplanes[airplane_index]
                                .wings[wing_index]
                            )
                            previous_panels = previous_wing.panels
                            if previous_panels is None:
                                raise ValueError(
                                    "missing previous incremental Panel grid"
                                )
                            previous_panel = previous_panels[chord, span]
                            back_left_velocity = (
                                panel.Blpp_GP1_CgP1 - previous_panel.Blpp_GP1_CgP1
                            ) / float(solver.delta_time)
                            back_right_velocity = (
                                panel.Brpp_GP1_CgP1 - previous_panel.Brpp_GP1_CgP1
                            ) / float(solver.delta_time)
                        else:
                            back_left_velocity = endpoint_vertex_velocity_gp[
                                chord + 1, span
                            ]
                            back_right_velocity = endpoint_vertex_velocity_gp[
                                chord + 1, span + 1
                            ]
                        back_left = (
                            panel.Blpp_GP1_CgP1
                            + (v_inf - back_left_velocity)
                            * float(solver.delta_time)
                            * 0.25
                        )
                        back_right = (
                            panel.Brpp_GP1_CgP1
                            + (v_inf - back_right_velocity)
                            * float(solver.delta_time)
                            * 0.25
                        )
                    values = (front_left, front_right, back_left, back_right)
                    if any(value is None for value in values) or not all(
                        bool(np.isfinite(value).all()) for value in values
                    ):
                        raise FloatingPointError(
                            "incremental bound-vortex geometry is non-finite"
                        )
                    panel.ring_vortex = _vortices.ring_vortex.RingVortex(
                        Flrvp_GP1_CgP1=front_left,
                        Frrvp_GP1_CgP1=front_right,
                        Blrvp_GP1_CgP1=back_left,
                        Brrvp_GP1_CgP1=back_right,
                        strength=1.0,
                    )


def _prepare_step_arrays(solver: CudaJointLEVTEVSolver, step: int) -> None:
    problem = solver.steady_problems[step]
    solver._current_step = step
    solver.current_airplanes = problem.airplanes
    solver.current_operating_point = problem.operating_point
    solver._currentVInf_GP1__E = solver.current_operating_point.vInf_GP1__E
    panel_count = solver.num_panels
    solver._currentStackFreestreamWingInfluences__E = np.zeros(
        panel_count, dtype=np.float64
    )
    solver._currentGridWingWingInfluences__E = np.zeros(
        (panel_count, panel_count), dtype=np.float64
    )
    solver._currentStackWakeWingInfluences__E = np.zeros(panel_count, dtype=np.float64)
    solver._current_bound_vortex_strengths = np.ones(panel_count, dtype=np.float64)
    solver._last_bound_vortex_strengths = np.zeros(panel_count, dtype=np.float64)
    solver.panels = np.empty(panel_count, dtype=object)
    solver.stackUnitNormals_GP1 = np.zeros((panel_count, 3), dtype=np.float64)
    solver.panel_areas = np.zeros(panel_count, dtype=np.float64)
    for name in (
        "stackCpp_GP1_CgP1",
        "_stackLastCpp_GP1_CgP1",
        "stackBrbrvp_GP1_CgP1",
        "stackFrbrvp_GP1_CgP1",
        "stackFlbrvp_GP1_CgP1",
        "stackBlbrvp_GP1_CgP1",
        "_lastStackBrbrvp_GP1_CgP1",
        "_lastStackFrbrvp_GP1_CgP1",
        "_lastStackFlbrvp_GP1_CgP1",
        "_lastStackBlbrvp_GP1_CgP1",
        "stackCblvpr_GP1_CgP1",
        "stackCblvpf_GP1_CgP1",
        "stackCblvpl_GP1_CgP1",
        "stackCblvpb_GP1_CgP1",
        "_lastStackCblvpr_GP1_CgP1",
        "_lastStackCblvpf_GP1_CgP1",
        "_lastStackCblvpl_GP1_CgP1",
        "_lastStackCblvpb_GP1_CgP1",
        "stackRbrv_GP1",
        "stackFbrv_GP1",
        "stackLbrv_GP1",
        "stackBbrv_GP1",
    ):
        setattr(solver, name, np.zeros((panel_count, 3), dtype=np.float64))
    for name in (
        "panel_is_trailing_edge",
        "panel_is_leading_edge",
        "panel_is_left_edge",
        "panel_is_right_edge",
    ):
        setattr(solver, name, np.zeros(panel_count, dtype=bool))
    solver._current_wake_vortex_strengths = solver._list_wake_vortex_strengths[step]
    solver._current_wake_vortex_ages = solver._list_wake_vortex_ages[step]
    solver._currentStackBrwrvp_GP1_CgP1 = solver.listStackBrwrvp_GP1_CgP1[step]
    solver._currentStackFrwrvp_GP1_CgP1 = solver.listStackFrwrvp_GP1_CgP1[step]
    solver._currentStackFlwrvp_GP1_CgP1 = solver.listStackFlwrvp_GP1_CgP1[step]
    solver._currentStackBlwrvp_GP1_CgP1 = solver.listStackBlwrvp_GP1_CgP1[step]
    solver._currentStackBoundRc0s = np.zeros(panel_count, dtype=np.float64)
    solver._currentStackWakeRc0s = solver._list_wake_rc0s[step]
    solver.stackSeedPoints_GP1_CgP1 = np.zeros((0, 3), dtype=np.float64)


def _apply_endpoint_motion_shadow(
    solver: CudaJointLEVTEVSolver,
    step: int,
    endpoint_vertex_velocity_gp: np.ndarray | None,
) -> None:
    """Replace current-step finite-difference operands, never prior geometry.

    Ptera expresses surface motion as ``-(x_now-x_previous)/dt`` in its
    no-penetration and Kutta--Joukowski formulas.  For a Q16 Newmark trial the
    authoritative current velocity is ``dq_trial``.  We therefore create
    private previous-point operands ``x_shadow=x_now-dt*v_endpoint`` after the
    real current geometry has been collapsed.  The committed Panel owner,
    circulation history and wake remain untouched.
    """

    if endpoint_vertex_velocity_gp is None:
        return
    if step == 0:
        if bool(np.any(endpoint_vertex_velocity_gp != 0.0)):
            raise ValueError("initial endpoint vertex velocity must be exact zero")
        return
    dt = float(solver.delta_time)
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("aerodynamic delta_time must be finite and positive")
    chord_count, span_count = solver.current_airplanes[0].wings[0].panels.shape
    expected_shape = (chord_count + 1, span_count + 1, 3)
    if endpoint_vertex_velocity_gp.shape != expected_shape:
        raise ValueError("bound endpoint velocity topology drift")

    panel_count = chord_count * span_count
    cpp_velocity = np.empty((panel_count, 3), dtype=np.float64)
    front_right_velocity = np.empty_like(cpp_velocity)
    front_left_velocity = np.empty_like(cpp_velocity)
    back_left_velocity = np.empty_like(cpp_velocity)
    back_right_velocity = np.empty_like(cpp_velocity)
    right_center_velocity = np.empty_like(cpp_velocity)
    front_center_velocity = np.empty_like(cpp_velocity)
    left_center_velocity = np.empty_like(cpp_velocity)
    back_center_velocity = np.empty_like(cpp_velocity)
    index = 0
    for chord in range(chord_count):
        for span in range(span_count):
            fl = endpoint_vertex_velocity_gp[chord, span]
            fr = endpoint_vertex_velocity_gp[chord, span + 1]
            bl = endpoint_vertex_velocity_gp[chord + 1, span]
            br = endpoint_vertex_velocity_gp[chord + 1, span + 1]
            fl_bound = 0.75 * fl + 0.25 * bl
            fr_bound = 0.75 * fr + 0.25 * br
            if chord + 1 < chord_count:
                next_fl = endpoint_vertex_velocity_gp[chord + 1, span]
                next_fr = endpoint_vertex_velocity_gp[chord + 1, span + 1]
                bl_bound = (
                    0.75 * next_fl + 0.25 * endpoint_vertex_velocity_gp[chord + 2, span]
                )
                br_bound = (
                    0.75 * next_fr
                    + 0.25 * endpoint_vertex_velocity_gp[chord + 2, span + 1]
                )
            else:
                # The trailing bound-ring legs are constructed from the surface
                # back vertices plus a relative-flow offset.  Under the frozen
                # endpoint-velocity contract their material velocity is the
                # corresponding surface-vertex velocity.
                bl_bound = bl
                br_bound = br
            cpp = 0.125 * (fr + fl) + 0.375 * (bl + br)
            cpp_velocity[index] = cpp
            front_right_velocity[index] = fr_bound
            front_left_velocity[index] = fl_bound
            back_left_velocity[index] = bl_bound
            back_right_velocity[index] = br_bound
            right_center_velocity[index] = 0.5 * (fr_bound + br_bound)
            front_center_velocity[index] = 0.5 * (fr_bound + fl_bound)
            left_center_velocity[index] = 0.5 * (fl_bound + bl_bound)
            back_center_velocity[index] = 0.5 * (bl_bound + br_bound)
            index += 1

    if index != panel_count:
        raise RuntimeError("endpoint motion panel count drift")
    shadow_pairs = (
        ("_stackLastCpp_GP1_CgP1", "stackCpp_GP1_CgP1", cpp_velocity),
        ("_lastStackBrbrvp_GP1_CgP1", "stackBrbrvp_GP1_CgP1", back_right_velocity),
        ("_lastStackFrbrvp_GP1_CgP1", "stackFrbrvp_GP1_CgP1", front_right_velocity),
        ("_lastStackFlbrvp_GP1_CgP1", "stackFlbrvp_GP1_CgP1", front_left_velocity),
        ("_lastStackBlbrvp_GP1_CgP1", "stackBlbrvp_GP1_CgP1", back_left_velocity),
        ("_lastStackCblvpr_GP1_CgP1", "stackCblvpr_GP1_CgP1", right_center_velocity),
        ("_lastStackCblvpf_GP1_CgP1", "stackCblvpf_GP1_CgP1", front_center_velocity),
        ("_lastStackCblvpl_GP1_CgP1", "stackCblvpl_GP1_CgP1", left_center_velocity),
        ("_lastStackCblvpb_GP1_CgP1", "stackCblvpb_GP1_CgP1", back_center_velocity),
    )
    for previous_name, current_name, velocity in shadow_pairs:
        current = np.asarray(getattr(solver, current_name), dtype=np.float64)
        shadow = np.ascontiguousarray(current - dt * velocity)
        if not bool(np.isfinite(shadow).all()):
            raise FloatingPointError("endpoint motion shadow became non-finite")
        setattr(solver, previous_name, shadow)

    leading_edge, trailing_edge = solver._station_le_te_points()
    leading_velocity = endpoint_vertex_velocity_gp[0]
    trailing_velocity = endpoint_vertex_velocity_gp[-1]
    solver._le_points_now = np.ascontiguousarray(leading_edge - dt * leading_velocity)
    solver._te_points_now = np.ascontiguousarray(trailing_edge - dt * trailing_velocity)


def _receipt_sha256(
    *,
    step_index: int,
    solver_state_sha256: str,
    load_packet_sha256: str | None,
    lev_particle_count: int,
    wake_ring_count: int,
    wake_convection_count: int,
) -> str:
    payload = json.dumps(
        {
            "schema_id": _RECEIPT_SCHEMA,
            "step_index": step_index,
            "solver_state_sha256": solver_state_sha256,
            "load_packet_sha256": load_packet_sha256,
            "lev_particle_count": lev_particle_count,
            "wake_ring_count": wake_ring_count,
            "wake_convection_count": wake_convection_count,
        },
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True)
class Q16IncrementalAeroStepReceipt:
    schema_id: str
    step_index: int
    solver_state_sha256: str
    load_packet_sha256: str | None
    lev_particle_count: int
    wake_ring_count: int
    wake_convection_count: int
    receipt_sha256: str


class Q16CudaIncrementalAeroSession:
    """External lifecycle owner for one exact real CUDA solver object."""

    __slots__ = (
        "_bound_geometry_step",
        "_bound_vertex_velocity_sha256",
        "_expected_state_sha256",
        "_next_step",
        "_solver",
        "_status",
    )

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        raise TypeError("use Q16CudaIncrementalAeroSession.begin/resume")

    @classmethod
    def _from_state(
        cls,
        solver: CudaJointLEVTEVSolver,
        *,
        next_step: int,
        bound_geometry_step: int | None,
        bound_vertex_velocity_sha256: str | None,
        expected_state_sha256: str,
    ) -> Q16CudaIncrementalAeroSession:
        session = object.__new__(cls)
        session._solver = solver
        session._next_step = next_step
        session._bound_geometry_step = bound_geometry_step
        session._bound_vertex_velocity_sha256 = bound_vertex_velocity_sha256
        session._expected_state_sha256 = expected_state_sha256
        session._status = "open"
        return session

    @classmethod
    def begin(cls, solver: CudaJointLEVTEVSolver) -> Q16CudaIncrementalAeroSession:
        if type(solver) is not CudaJointLEVTEVSolver:
            raise TypeError("solver must be an exact CudaJointLEVTEVSolver")
        require_q16_mandatory_aero_mode(solver)
        if solver.ran is not False:
            raise Q16IncrementalAeroLifecycleError("solver was already run")
        if solver._steps_done != 0 or len(solver.ledger) != 0:
            raise Q16IncrementalAeroLifecycleError("solver is not pristine")
        if any(hasattr(solver, name) for name in _MARKER_ATTRIBUTES):
            raise Q16IncrementalAeroLifecycleError(
                "solver already has an incremental owner"
            )
        _preallocate_wake_arrays(solver)
        setattr(solver, _NEXT_STEP_ATTRIBUTE, 0)
        setattr(solver, _BOUND_STEP_ATTRIBUTE, None)
        setattr(solver, _BOUND_VELOCITY_ATTRIBUTE, None)
        setattr(solver, _BOUND_VELOCITY_SHA_ATTRIBUTE, None)
        state_sha = _semantic_solver_sha256(solver)
        setattr(solver, _STATE_SHA_ATTRIBUTE, state_sha)
        return cls._from_state(
            solver,
            next_step=0,
            bound_geometry_step=None,
            bound_vertex_velocity_sha256=None,
            expected_state_sha256=state_sha,
        )

    @classmethod
    def resume(cls, solver: CudaJointLEVTEVSolver) -> Q16CudaIncrementalAeroSession:
        if type(solver) is not CudaJointLEVTEVSolver:
            raise TypeError("solver must be an exact CudaJointLEVTEVSolver")
        require_q16_mandatory_aero_mode(solver)
        if not all(hasattr(solver, name) for name in _MARKER_ATTRIBUTES):
            raise Q16IncrementalAeroLifecycleError(
                "incremental solver was not initialized"
            )
        next_step = getattr(solver, _NEXT_STEP_ATTRIBUTE)
        bound_step = getattr(solver, _BOUND_STEP_ATTRIBUTE)
        bound_velocity = getattr(solver, _BOUND_VELOCITY_ATTRIBUTE)
        bound_velocity_sha = getattr(solver, _BOUND_VELOCITY_SHA_ATTRIBUTE)
        expected_sha = getattr(solver, _STATE_SHA_ATTRIBUTE)
        if (
            type(next_step) is not int
            or not 0 <= next_step <= solver.num_steps
            or type(expected_sha) is not str
            or len(expected_sha) != 64
            or solver.ran is not False
            or solver._steps_done != next_step
            or (bound_step is not None and bound_step != next_step)
            or (bound_velocity is not None and bound_step is None)
            or (bound_velocity is None and bound_velocity_sha is not None)
            or (
                bound_velocity is not None
                and (
                    type(bound_velocity_sha) is not str
                    or len(bound_velocity_sha) != 64
                    or _vertex_velocity_sha256(bound_velocity) != bound_velocity_sha
                )
            )
        ):
            raise Q16IncrementalAeroLifecycleError("incremental solver state drift")
        if _semantic_solver_sha256(solver) != expected_sha:
            raise Q16IncrementalAeroLifecycleError("incremental solver state drift")
        return cls._from_state(
            solver,
            next_step=next_step,
            bound_geometry_step=bound_step,
            bound_vertex_velocity_sha256=bound_velocity_sha,
            expected_state_sha256=expected_sha,
        )

    def fork(self) -> Q16CudaIncrementalAeroSession:
        """Issue a trusted pickle branch and bind a fresh live-object seal."""

        self._assert_live()
        torch.cuda.synchronize(self._solver.cuda_device)
        clone = pickle.loads(pickle.dumps(self._solver, protocol=5))
        next_step = self._next_step
        bound_step = self._bound_geometry_step
        bound_velocity_sha = self._bound_vertex_velocity_sha256
        setattr(clone, _NEXT_STEP_ATTRIBUTE, next_step)
        setattr(clone, _BOUND_STEP_ATTRIBUTE, bound_step)
        clone_sha = _semantic_solver_sha256(clone)
        setattr(clone, _STATE_SHA_ATTRIBUTE, clone_sha)
        self._assert_live()
        return type(self)._from_state(
            clone,
            next_step=next_step,
            bound_geometry_step=bound_step,
            bound_vertex_velocity_sha256=bound_velocity_sha,
            expected_state_sha256=clone_sha,
        )

    def bind_next_panel_grid(self, panels: np.ndarray) -> str:
        """Detach and bind one candidate Panel owner to the exact next step."""

        self._assert_live()
        solver = self._solver
        step = self._next_step
        if step >= solver.num_steps:
            raise Q16IncrementalAeroLifecycleError("no remaining aerodynamic steps")
        if self._bound_geometry_step is not None:
            raise Q16IncrementalAeroLifecycleError(
                "next aerodynamic geometry is already bound"
            )
        problem = solver.steady_problems[step]
        if len(problem.airplanes) != 1 or len(problem.airplanes[0].wings) != 1:
            raise ValueError("incremental geometry supports one airplane and one wing")
        wing = problem.airplanes[0].wings[0]
        current = wing.panels
        if current is None:
            raise ValueError("missing current aerodynamic Panel topology")
        detached = _validate_detached_panel_grid(panels, tuple(current.shape))
        grid_sha = _panel_grid_sha256(detached)
        object.__setattr__(wing, "_panels", detached)
        if wing.panels is not detached:
            self._status = "failed"
            raise Q16IncrementalAeroLifecycleError(
                "next Panel owner replacement did not take effect"
            )
        self._bound_geometry_step = step
        setattr(solver, _BOUND_STEP_ATTRIBUTE, step)
        live_sha = _semantic_solver_sha256(solver)
        setattr(solver, _STATE_SHA_ATTRIBUTE, live_sha)
        self._expected_state_sha256 = live_sha
        return grid_sha

    def bind_next_panel_kinematics(
        self,
        panels: np.ndarray,
        endpoint_vertex_velocity_gp: np.ndarray,
    ) -> tuple[str, str]:
        """Atomically own the next geometry and its Q16 endpoint velocity."""

        self._assert_live()
        solver = self._solver
        step = self._next_step
        if step >= solver.num_steps:
            raise Q16IncrementalAeroLifecycleError("no remaining aerodynamic steps")
        problem = solver.steady_problems[step]
        if len(problem.airplanes) != 1 or len(problem.airplanes[0].wings) != 1:
            raise ValueError("incremental kinematics supports one airplane and wing")
        wing = problem.airplanes[0].wings[0]
        current = wing.panels
        if current is None:
            raise ValueError("missing current aerodynamic Panel topology")
        detached_velocity = _validate_detached_vertex_velocity(
            endpoint_vertex_velocity_gp,
            tuple(current.shape),
        )
        if step == 0 and bool(np.any(detached_velocity != 0.0)):
            raise ValueError("initial endpoint vertex velocity must be exact zero")
        velocity_sha = _vertex_velocity_sha256(detached_velocity)
        panel_sha = self.bind_next_panel_grid(panels)
        setattr(solver, _BOUND_VELOCITY_ATTRIBUTE, detached_velocity)
        setattr(solver, _BOUND_VELOCITY_SHA_ATTRIBUTE, velocity_sha)
        self._bound_vertex_velocity_sha256 = velocity_sha
        live_sha = _semantic_solver_sha256(solver)
        setattr(solver, _STATE_SHA_ATTRIBUTE, live_sha)
        self._expected_state_sha256 = live_sha
        return panel_sha, velocity_sha

    @property
    def solver(self) -> CudaJointLEVTEVSolver:
        return self._solver

    @property
    def next_step(self) -> int:
        return self._next_step

    @property
    def bound_vertex_velocity_sha256(self) -> str | None:
        return self._bound_vertex_velocity_sha256

    @property
    def status(self) -> str:
        return self._status

    def _ensure_open(self) -> None:
        if self._status != "open":
            raise Q16IncrementalAeroLifecycleError(
                f"incremental session is {self._status}"
            )

    def _assert_live(self) -> None:
        self._ensure_open()
        solver = self._solver
        try:
            require_q16_mandatory_aero_mode(solver)
        except Exception as error:
            self._status = "failed"
            raise Q16IncrementalAeroLifecycleError(
                "aerodynamic mode drift outside session"
            ) from error
        if solver.ran is not False:
            self._status = "failed"
            raise Q16IncrementalAeroLifecycleError("solver advanced outside session")
        bound_velocity = getattr(solver, _BOUND_VELOCITY_ATTRIBUTE, None)
        velocity_binding_drift = self._bound_vertex_velocity_sha256 is not None and (
            type(bound_velocity) is not np.ndarray
            or _vertex_velocity_sha256(bound_velocity)
            != self._bound_vertex_velocity_sha256
        )
        if (
            getattr(solver, _NEXT_STEP_ATTRIBUTE, None) != self._next_step
            or getattr(solver, _STATE_SHA_ATTRIBUTE, None)
            != self._expected_state_sha256
            or getattr(solver, _BOUND_STEP_ATTRIBUTE, None) != self._bound_geometry_step
            or getattr(solver, _BOUND_VELOCITY_SHA_ATTRIBUTE, None)
            != self._bound_vertex_velocity_sha256
            or velocity_binding_drift
            or solver._steps_done != self._next_step
            or _semantic_solver_sha256(solver) != self._expected_state_sha256
        ):
            self._status = "failed"
            raise Q16IncrementalAeroLifecycleError(
                "incremental solver state drift outside session"
            )

    def advance_one_step(self) -> Q16IncrementalAeroStepReceipt:
        self._assert_live()
        solver = self._solver
        step = self._next_step
        if step >= solver.num_steps:
            raise Q16IncrementalAeroLifecycleError("no remaining aerodynamic steps")
        endpoint_velocity = getattr(solver, _BOUND_VELOCITY_ATTRIBUTE)
        try:
            _initialize_panel_vortices_for_step(solver, step, endpoint_velocity)
            # Defer the previous state's wake birth until this trial geometry
            # exists. Ptera anchors the new wake row to the current state's
            # trailing-edge bound rings. This gives predictor/corrector
            # branches causal ownership: each candidate geometry creates its
            # own wake without mutating the committed parent.
            if step > 0:
                if solver._current_step != step - 1:
                    raise Q16IncrementalAeroLifecycleError(
                        "pending wake owner differs from previous step"
                    )
                solver._populate_next_airplanes_wake()
            _prepare_step_arrays(solver, step)
            solver._collapse_geometry()
            _apply_endpoint_motion_shadow(solver, step, endpoint_velocity)
            solver._calculate_wing_wing_influences()
            solver._calculate_freestream_wing_influences()
            solver._calculate_wake_wing_influences()
            solver._calculate_vortex_strengths()
            if step >= solver.first_results_step:
                solver._calculate_loads()
        except Exception:
            self._status = "failed"
            raise
        if solver._steps_done != step + 1 or len(solver.ledger) != step + 1:
            self._status = "failed"
            raise Q16IncrementalAeroLifecycleError(
                "incremental numerical step count drift"
            )
        load_sha: str | None = None
        if step >= solver.first_results_step:
            load_sha = Q16CudaAerodynamicLoadPacket.from_solver(solver).packet_sha256
        self._next_step = step + 1
        setattr(solver, _NEXT_STEP_ATTRIBUTE, self._next_step)
        self._bound_geometry_step = None
        setattr(solver, _BOUND_STEP_ATTRIBUTE, None)
        self._bound_vertex_velocity_sha256 = None
        setattr(solver, _BOUND_VELOCITY_ATTRIBUTE, None)
        setattr(solver, _BOUND_VELOCITY_SHA_ATTRIBUTE, None)
        # Canonical extraction may materialize lazy Ptera presentation fields;
        # do it before sealing the live object so those fields are part of the
        # issued state rather than appearing as post-seal drift.
        scientific_state_sha = _scientific_solver_sha256(solver)
        live_state_sha = _semantic_solver_sha256(solver)
        setattr(solver, _STATE_SHA_ATTRIBUTE, live_state_sha)
        self._expected_state_sha256 = live_state_sha
        wake_count = solver.list_num_wake_vortices[step]
        receipt_sha = _receipt_sha256(
            step_index=step,
            solver_state_sha256=scientific_state_sha,
            load_packet_sha256=load_sha,
            lev_particle_count=solver.lev_pf.n,
            wake_ring_count=wake_count,
            wake_convection_count=solver.cuda_counters["wake_convection"],
        )
        return Q16IncrementalAeroStepReceipt(
            schema_id=_RECEIPT_SCHEMA,
            step_index=step,
            solver_state_sha256=scientific_state_sha,
            load_packet_sha256=load_sha,
            lev_particle_count=solver.lev_pf.n,
            wake_ring_count=wake_count,
            wake_convection_count=solver.cuda_counters["wake_convection"],
            receipt_sha256=receipt_sha,
        )

    def finalize(self) -> CudaJointLEVTEVSolver:
        self._assert_live()
        solver = self._solver
        if self._next_step != solver.num_steps:
            raise Q16IncrementalAeroLifecycleError(
                "all steps must complete before finalization"
            )
        solver._finalize_loads()
        solver.ran = True
        self._status = "finalized"
        return solver


__all__ = [
    "Q16CudaIncrementalAeroSession",
    "Q16IncrementalAeroLifecycleError",
    "Q16IncrementalAeroStepReceipt",
]
