from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pterasoftware as ps
import pytest

from fluxvortex.solver import UVPMHybridSolver
from forward_flight_benchmarks.fluxv_v5f_live_load_adapter import (
    PteraNativeLiveLoadOperands,
    calculate_live_native_surface_release_load_delta,
    calculate_native_load_delta_from_live_operands,
    capture_ptera_native_load_operands,
)
from forward_flight_benchmarks.fluxv_v5f_native_load import LEG_NAMES
from forward_flight_benchmarks.ptera_adapter import build_yang2025_movement


_CENTER_ATTRIBUTES = {
    "right": "stackCblvpr_GP1_CgP1",
    "front": "stackCblvpf_GP1_CgP1",
    "left": "stackCblvpl_GP1_CgP1",
    "back": "stackCblvpb_GP1_CgP1",
}
_VECTOR_ATTRIBUTES = {
    "right": "stackRbrv_GP1",
    "front": "stackFbrv_GP1",
    "left": "stackLbrv_GP1",
    "back": "stackBbrv_GP1",
}
_MOVEMENT_METHODS = {
    "right": "_calculate_current_movement_velocities_at_right_leg_centers",
    "front": "_calculate_current_movement_velocities_at_front_leg_centers",
    "left": "_calculate_current_movement_velocities_at_left_leg_centers",
    "back": "_calculate_current_movement_velocities_at_back_leg_centers",
}


def _array_token(value: Any) -> tuple[Any, ...] | None:
    if value is None:
        return None
    array = np.asarray(value)
    return (array.dtype.str, array.shape, array.tobytes())


def _iter_current_panels(solver: UVPMHybridSolver) -> Iterator[Any]:
    for airplane in solver.current_airplanes:
        for wing in airplane.wings:
            yield from np.ravel(wing.panels)


def _solver_aero_state_token(solver: UVPMHybridSolver) -> tuple[Any, ...]:
    """Bitwise token for state the read-only adapter is forbidden to mutate."""

    panel_loads = tuple(
        (
            _array_token(getattr(panel, "forces_GP1", None)),
            _array_token(getattr(panel, "moments_GP1_CgP1", None)),
            None if panel.ring_vortex is None else float(panel.ring_vortex.strength),
        )
        for panel in _iter_current_panels(solver)
    )
    airplane_loads = tuple(
        (
            _array_token(getattr(airplane, "forces_W", None)),
            _array_token(getattr(airplane, "moments_W_CgP1", None)),
        )
        for airplane in solver.current_airplanes
    )
    solver_arrays = tuple(
        (
            attribute,
            _array_token(getattr(solver, attribute, None)),
        )
        for attribute in (
            "_current_bound_vortex_strengths",
            "_last_bound_vortex_strengths",
        )
    )
    wake_arrays = tuple(
        (
            attribute,
            tuple(_array_token(row) for row in getattr(solver, attribute, ())),
        )
        for attribute in (
            "_list_wake_vortex_strengths",
            "_list_wake_vortex_ages",
            "listStackBrwrvp_GP1_CgP1",
            "listStackFrwrvp_GP1_CgP1",
            "listStackFlwrvp_GP1_CgP1",
            "listStackBlwrvp_GP1_CgP1",
        )
    )
    particle_arrays = tuple(
        (
            attribute,
            _array_token(getattr(solver._vpm_field, attribute)),
        )
        for attribute in ("_pos", "_gamma", "_sigma", "_age")
    )
    return (
        int(solver._current_step),
        panel_loads,
        airplane_loads,
        solver_arrays,
        wake_arrays,
        particle_arrays,
    )


def _actual_wake_ring_count(solver: UVPMHybridSolver) -> int:
    # Ptera preallocates arrays for *all future* steps before solving.  Only the
    # current array represents wake rings that exist at this live step.
    return int(np.asarray(solver._current_wake_vortex_strengths).size)


def _assert_exact_positive_zero(value: Any) -> None:
    array = np.asarray(value, dtype=float)
    np.testing.assert_array_equal(array, np.zeros_like(array))
    assert not np.any(np.signbit(array))


@pytest.fixture(scope="module")
def live_parent_audit() -> dict[str, Any]:
    movement, _ = build_yang2025_movement(15.0, "smoke")
    problem = ps.problems.UnsteadyProblem(
        movement=movement,
        only_final_results=False,
    )
    solver = UVPMHybridSolver(
        problem,
        max_particles=20_000,
        stretch=False,
        free_wake=False,
    )
    audit: dict[str, Any] = {
        "context": "other",
        "velocity_calls": [],
    }

    parent_calculate_solution_velocity = solver.calculate_solution_velocity

    def recording_calculate_solution_velocity(*args: Any, **kwargs: Any) -> np.ndarray:
        points = kwargs.get("stackP_GP1_CgP1")
        if points is None and args:
            points = args[0]
        result = parent_calculate_solution_velocity(*args, **kwargs)
        audit["velocity_calls"].append(
            {
                "context": audit["context"],
                "step": int(solver._current_step),
                "points": np.asarray(points, dtype=float).copy(),
                "solution_velocity": np.asarray(result, dtype=float).copy(),
            }
        )
        return result

    solver.calculate_solution_velocity = recording_calculate_solution_velocity
    parent_calculate_loads = solver._calculate_loads

    def recording_calculate_loads() -> None:
        audit["context"] = "parent_load"
        try:
            parent_calculate_loads()
        finally:
            audit["context"] = "other"
        if solver._current_step == 0:
            assert _actual_wake_ring_count(solver) == 0
            state_before = _solver_aero_state_token(solver)
            audit["context"] = "adapter_step_zero"
            operands = capture_ptera_native_load_operands(solver)
            zero_gamma = np.zeros(operands.panel_count, dtype=float)
            zero_delta = calculate_native_load_delta_from_live_operands(
                operands,
                gamma_lev_current_m2_s=zero_gamma,
                gamma_lev_previous_m2_s=None,
                active_current=False,
            )
            audit["context"] = "other"
            assert _solver_aero_state_token(solver) == state_before
            audit["step_zero_operands"] = operands
            audit["step_zero_delta"] = zero_delta
            audit["step_zero_actual_wake_ring_count"] = _actual_wake_ring_count(solver)

    solver._calculate_loads = recording_calculate_loads
    solver.run(
        prescribed_wake=True,
        calculate_streamlines=False,
        show_progress=False,
    )
    audit["final_state_before_capture"] = _solver_aero_state_token(solver)
    audit["context"] = "adapter_final"
    audit["final_operands"] = capture_ptera_native_load_operands(solver)
    audit["context"] = "other"
    audit["final_state_after_capture"] = _solver_aero_state_token(solver)
    audit["solver"] = solver
    return audit


def test_capture_matches_each_live_parent_load_operand_exactly(
    live_parent_audit: dict[str, Any],
) -> None:
    solver = live_parent_audit["solver"]
    operands = live_parent_audit["final_operands"]
    assert isinstance(operands, PteraNativeLiveLoadOperands)
    assert (
        live_parent_audit["final_state_after_capture"]
        == live_parent_audit["final_state_before_capture"]
    )

    parent_calls = [
        row
        for row in live_parent_audit["velocity_calls"]
        if row["context"] == "parent_load" and row["step"] == operands.current_step
    ]
    assert len(parent_calls) == 4
    for leg, parent_call in zip(LEG_NAMES, parent_calls, strict=True):
        source_center = getattr(solver, _CENTER_ATTRIBUTES[leg])
        source_vector = getattr(solver, _VECTOR_ATTRIBUTES[leg])
        source_movement = getattr(solver, _MOVEMENT_METHODS[leg])()
        np.testing.assert_array_equal(
            operands.leg_centers_gp1_cgp1_m[leg], source_center
        )
        np.testing.assert_array_equal(operands.leg_vectors_gp1_m[leg], source_vector)
        np.testing.assert_array_equal(parent_call["points"], source_center)
        np.testing.assert_array_equal(
            operands.leg_solution_velocities_gp1_m_s[leg],
            parent_call["solution_velocity"],
        )
        np.testing.assert_array_equal(
            operands.leg_movement_velocities_gp1_m_s[leg], source_movement
        )
        np.testing.assert_array_equal(
            operands.leg_velocities_gp1_m_s[leg],
            parent_call["solution_velocity"] + source_movement,
        )

    np.testing.assert_array_equal(operands.panel_area_m2, solver.panel_areas)
    np.testing.assert_array_equal(
        operands.panel_normal_gp1, solver.stackUnitNormals_GP1
    )
    np.testing.assert_array_equal(
        operands.panel_collocation_gp1_cgp1_m, solver.stackCpp_GP1_CgP1
    )
    expected_topology = {
        "airplane_index": [],
        "wing_index": [],
        "chordwise_index": [],
        "spanwise_index": [],
    }
    for airplane_index, airplane in enumerate(solver.current_airplanes):
        for wing_index, wing in enumerate(airplane.wings):
            for panel in np.ravel(wing.panels):
                expected_topology["airplane_index"].append(airplane_index)
                expected_topology["wing_index"].append(wing_index)
                expected_topology["chordwise_index"].append(
                    panel.local_chordwise_position
                )
                expected_topology["spanwise_index"].append(
                    panel.local_spanwise_position
                )
    for name, values in expected_topology.items():
        np.testing.assert_array_equal(operands.topology[name], values)


def test_step_zero_has_no_prior_wake_and_q_zero_is_bitwise_positive_zero(
    live_parent_audit: dict[str, Any],
) -> None:
    operands = live_parent_audit["step_zero_operands"]
    delta = live_parent_audit["step_zero_delta"]
    assert operands.current_step == 0
    assert live_parent_audit["step_zero_actual_wake_ring_count"] == 0
    _assert_exact_positive_zero(delta.gamma_lev_previous_m2_s)
    _assert_exact_positive_zero(delta.surface_release_current_m2_s)
    _assert_exact_positive_zero(delta.wake_release_previous_m2_s)
    for name, value in delta.correction.items():
        if name.startswith("delta_") and isinstance(value, np.ndarray):
            _assert_exact_positive_zero(value)
    for leg in LEG_NAMES:
        ledger = delta.correction["leg_ledger"][leg]
        _assert_exact_positive_zero(ledger["effective_strength_delta_m2_s"])
        _assert_exact_positive_zero(ledger["delta_force_gp1_n"])
        _assert_exact_positive_zero(ledger["delta_moment_gp1_cgp1_nm"])

    with pytest.raises(ValueError, match="must be None at step zero"):
        calculate_native_load_delta_from_live_operands(
            operands,
            gamma_lev_current_m2_s=np.zeros(operands.panel_count),
            gamma_lev_previous_m2_s=np.zeros(operands.panel_count),
            active_current=False,
        )


def test_nonzero_release_returns_only_delta_and_never_mutates_solver(
    live_parent_audit: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    solver = live_parent_audit["solver"]
    operands = live_parent_audit["final_operands"]
    span = np.asarray(operands.topology["spanwise_index"], dtype=float)
    gamma_current = 0.012 * (span + 1.0)
    gamma_previous = 0.007 * (span + 1.0)
    active_by_span = np.asarray([True, False, True, True])
    active = active_by_span[np.asarray(operands.topology["spanwise_index"])]
    state_before = _solver_aero_state_token(solver)

    def forbidden_process_solver_loads(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("the delta-only adapter called process_solver_loads")

    monkeypatch.setattr(
        ps._functions,
        "process_solver_loads",
        forbidden_process_solver_loads,
    )
    live_parent_audit["context"] = "adapter_nonzero"
    try:
        delta = calculate_live_native_surface_release_load_delta(
            solver,
            gamma_lev_current_m2_s=gamma_current,
            gamma_lev_previous_m2_s=gamma_previous,
            active_current=active,
        )
    finally:
        live_parent_audit["context"] = "other"
    state_after = _solver_aero_state_token(solver)

    assert state_after == state_before
    np.testing.assert_array_equal(
        delta.surface_release_current_m2_s,
        np.where(active, gamma_current, 0.0),
    )
    np.testing.assert_array_equal(
        delta.wake_release_previous_m2_s,
        gamma_previous,
    )
    assert delta.call_contract["commit"] == "none_delta_only"
    assert np.max(np.abs(delta.correction["delta_total_force_gp1_n"])) > 0.0
    assert np.max(np.abs(delta.correction["delta_total_moment_gp1_cgp1_nm"])) > 0.0


def test_post_step_zero_requires_an_actual_previous_material_state(
    live_parent_audit: dict[str, Any],
) -> None:
    operands = live_parent_audit["final_operands"]
    assert operands.current_step > 0
    with pytest.raises(ValueError, match="required after step zero"):
        calculate_native_load_delta_from_live_operands(
            operands,
            gamma_lev_current_m2_s=np.zeros(operands.panel_count),
            gamma_lev_previous_m2_s=None,
            active_current=False,
        )


def test_capture_rejects_unsolved_parent_load_state(
    live_parent_audit: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    solver = live_parent_audit["solver"]
    first_panel = next(_iter_current_panels(solver))
    monkeypatch.setattr(first_panel, "forces_GP1", None)
    with pytest.raises(RuntimeError, match="parent load calculation"):
        capture_ptera_native_load_operands(solver)


def test_capture_itself_rejects_a_forged_step_zero_wake(
    live_parent_audit: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    solver = live_parent_audit["solver"]
    monkeypatch.setattr(solver, "_current_step", 0)
    monkeypatch.setattr(solver, "_current_wake_vortex_strengths", np.ones(1))
    with pytest.raises(RuntimeError, match="step zero"):
        capture_ptera_native_load_operands(solver)


def test_adapter_source_contains_no_solver_load_or_wake_commit() -> None:
    source_path = Path(
        "platform/forward_flight_benchmarks/fluxv_v5f_live_load_adapter.py"
    )
    source = source_path.read_text(encoding="utf-8")
    assert "._populate_next_airplanes_wake(" not in source
    assert "._calculate_loads(" not in source
    assert "process_solver_loads(" not in source
