from __future__ import annotations

from dataclasses import fields
from typing import Any

import numpy as np
import pterasoftware as ps
import pytest

import forward_flight_benchmarks.fluxv_v5f_native_solver as native_solver_module
from forward_flight_benchmarks.fluxv_v5f_material_state import (
    PteraMaterialLEVProposal,
    PteraMaterialLEVSnapshot,
)
from forward_flight_benchmarks.fluxv_v5f_native_solver import (
    NativeMaterialLEVStepReport,
    NativeMaterialLEVTimeMarchConfig,
    NativeMaterialLEVTimeMarchSolver,
    make_fluxv_v5f_native_time_march_solver,
)
from forward_flight_benchmarks.ptera_adapter import build_yang2025_movement


class _StopAfterCommittedSteps(RuntimeError):
    pass


class _InjectedLoadFailure(RuntimeError):
    pass


class _InjectedCommitFailure(RuntimeError):
    pass


def _yang_problem() -> ps.problems.UnsteadyProblem:
    movement, _ = build_yang2025_movement(15.0, "smoke")
    return ps.problems.UnsteadyProblem(
        movement=movement,
        only_final_results=False,
    )


def _make_active_solver(
    *, lesp_critical: float = 0.10
) -> NativeMaterialLEVTimeMarchSolver:
    solver = make_fluxv_v5f_native_time_march_solver(
        _yang_problem(),
        config=NativeMaterialLEVTimeMarchConfig(
            enabled=True,
            lesp_critical=lesp_critical,
            core_radius_ratio=0.25,
        ),
        max_particles=20_000,
        stretch=False,
        free_wake=False,
    )
    assert isinstance(solver, NativeMaterialLEVTimeMarchSolver)
    return solver


def _run_until_committed_steps(
    solver: NativeMaterialLEVTimeMarchSolver,
    count: int,
) -> None:
    """Run real Ptera steps, stopping only after ``count`` wake commits."""

    original = solver._calculate_wake_wing_influences

    def stop_at_boundary() -> None:
        if int(solver._current_step) >= count:
            raise _StopAfterCommittedSteps
        original()

    solver._calculate_wake_wing_influences = stop_at_boundary  # type: ignore[method-assign]
    with pytest.raises(_StopAfterCommittedSteps):
        solver.run(
            prescribed_wake=True,
            calculate_streamlines=False,
            show_progress=False,
        )
    assert int(solver._current_step) == count
    assert len(solver.v5f_step_reports) == count


def _assert_snapshot_exact(
    actual: PteraMaterialLEVSnapshot,
    expected: PteraMaterialLEVSnapshot,
) -> None:
    for field in fields(PteraMaterialLEVSnapshot):
        actual_value = getattr(actual, field.name)
        expected_value = getattr(expected, field.name)
        if isinstance(actual_value, np.ndarray):
            np.testing.assert_array_equal(actual_value, expected_value, strict=True)
        else:
            assert actual_value == expected_value


def _rear_bound_strengths_by_strip(problem: Any) -> np.ndarray:
    values: list[float] = []
    for airplane in problem.airplanes:
        for wing in airplane.wings:
            for span_index in range(wing.panels.shape[1]):
                panel = wing.panels[-1, span_index]
                assert panel.ring_vortex is not None
                values.append(float(panel.ring_vortex.strength))
    return np.asarray(values, dtype=float)


def _front_wake_strengths_by_strip(problem: Any) -> np.ndarray:
    values: list[float] = []
    for airplane in problem.airplanes:
        for wing in airplane.wings:
            wake = wing.wake_ring_vortices
            assert wake is not None and wake.shape[0] >= 1
            values.extend(float(vortex.strength) for vortex in wake[0])
    return np.asarray(values, dtype=float)


def _te_panel_mask(topology: Any) -> np.ndarray:
    airplane = np.asarray(topology["airplane_index"], dtype=np.int64)
    wing = np.asarray(topology["wing_index"], dtype=np.int64)
    chord = np.asarray(topology["chordwise_index"], dtype=np.int64)
    result = np.zeros(chord.size, dtype=bool)
    for airplane_index, wing_index in sorted(
        set(zip(airplane.tolist(), wing.tolist(), strict=True))
    ):
        member = (airplane == airplane_index) & (wing == wing_index)
        result[member & (chord == np.max(chord[member]))] = True
    return result


def _panel_strip_values(
    report: NativeMaterialLEVStepReport, values: np.ndarray
) -> np.ndarray:
    """Broadcast Yang's one-wing strip ledger into live Ptera panel order."""

    assert report.load_delta is not None
    topology = report.load_delta.operands.topology
    np.testing.assert_array_equal(
        np.asarray(topology["airplane_index"], dtype=np.int64),
        np.zeros(report.load_delta.operands.panel_count, dtype=np.int64),
    )
    np.testing.assert_array_equal(
        np.asarray(topology["wing_index"], dtype=np.int64),
        np.zeros(report.load_delta.operands.panel_count, dtype=np.int64),
    )
    spanwise = np.asarray(topology["spanwise_index"], dtype=np.int64)
    return np.asarray(values)[spanwise]


def _assert_load_ledger(
    report: NativeMaterialLEVStepReport,
    q_previous_by_strip: np.ndarray,
) -> None:
    delta = report.load_delta
    assert delta is not None
    q_current_panel = _panel_strip_values(report, report.q_by_strip_m2_s)
    q_previous_panel = _panel_strip_values(report, q_previous_by_strip)
    active_panel = _panel_strip_values(report, report.active).astype(bool)
    surface_current = np.where(active_panel, q_current_panel, 0.0)

    np.testing.assert_array_equal(delta.gamma_lev_current_m2_s, q_current_panel)
    np.testing.assert_array_equal(delta.gamma_lev_previous_m2_s, q_previous_panel)
    np.testing.assert_array_equal(delta.active_current, active_panel)
    np.testing.assert_array_equal(delta.surface_release_current_m2_s, surface_current)
    np.testing.assert_array_equal(delta.wake_release_previous_m2_s, q_previous_panel)

    topology = delta.operands.topology
    trailing_edge = _te_panel_mask(topology)
    expected_back = np.zeros_like(q_current_panel)
    expected_back[trailing_edge] = (
        surface_current[trailing_edge] - q_previous_panel[trailing_edge]
    )
    np.testing.assert_array_equal(
        delta.correction["leg_ledger"]["back"]["effective_strength_delta_m2_s"],
        expected_back,
    )

    expected_rate = np.where(
        active_panel,
        (q_current_panel - q_previous_panel) / delta.operands.delta_time_s,
        0.0,
    )
    expected_eq17 = -(
        delta.operands.rho_kg_m3
        * delta.operands.panel_area_m2[:, None]
        * delta.operands.panel_normal_gp1
        * expected_rate[:, None]
    )
    expected_eq17[expected_eq17 == 0.0] = 0.0
    np.testing.assert_allclose(
        delta.correction["delta_eq17_force_gp1_n"],
        expected_eq17,
        atol=2.0e-14,
        rtol=2.0e-14,
    )


def _assert_continuing_eq7_join(
    previous: NativeMaterialLEVStepReport,
    current: NativeMaterialLEVStepReport,
) -> None:
    before = current.material_snapshot_before
    after = current.material_snapshot
    _assert_snapshot_exact(before, previous.material_snapshot)

    old_count = before.gamma_m2_s.size
    assert after.gamma_m2_s[:old_count].tobytes() == before.gamma_m2_s.tobytes()
    np.testing.assert_array_equal(after.ring_id[:old_count], before.ring_id)
    np.testing.assert_array_equal(after.strip[:old_count], before.strip)
    np.testing.assert_array_equal(after.birth_step[:old_count], before.birth_step)
    np.testing.assert_array_equal(after.sheet_id[:old_count], before.sheet_id)
    np.testing.assert_array_equal(
        current.new_sheet,
        current.active & ~previous.active,
    )

    created = np.asarray(current.created_ring_ids, dtype=np.int64)
    assert created.size == int(np.count_nonzero(current.active))
    assert np.intersect1d(created, before.ring_id).size == 0
    for created_id in created:
        newborn_position = int(np.flatnonzero(after.ring_id == created_id)[0])
        strip = int(after.strip[newborn_position])
        assert bool(current.active[strip])
        assert int(after.birth_step[newborn_position]) == int(current.step)
        np.testing.assert_allclose(
            after.gamma_m2_s[newborn_position],
            current.q_by_strip_m2_s[strip],
            atol=2.0e-14,
            rtol=2.0e-14,
        )
        if bool(previous.active[strip]) and not bool(current.new_sheet[strip]):
            old_positions = np.flatnonzero(before.strip == strip)
            assert old_positions.size > 0
            old_position = int(old_positions[-1])
            # Eq. 7 makes the remeshed old front and newborn aft the same
            # material edge.  Eq. 24 then evaluates the coincident vertices
            # with the same live field, so the join remains coincident.
            np.testing.assert_allclose(
                after.shadow_rings_forward_gp1_m[old_position, :2],
                after.shadow_rings_forward_gp1_m[newborn_position, [3, 2]],
                atol=2.0e-13,
                rtol=2.0e-13,
            )


def test_t3_real_yang_three_steps_preserve_material_identity_and_time_ledgers() -> None:
    solver = _make_active_solver(lesp_critical=0.03)
    _run_until_committed_steps(solver, 3)
    step0, step1, step2 = solver.v5f_step_reports[:3]
    assert (step0.step, step1.step, step2.step) == (0, 1, 2)
    assert np.any(step0.active & step1.active)
    assert np.any(step1.active & step2.active)

    _assert_continuing_eq7_join(step0, step1)
    _assert_continuing_eq7_join(step1, step2)

    np.testing.assert_array_equal(
        step1.parent_wake_front_strengths_m2_s,
        step0.next_wake_front_strengths_m2_s,
    )
    np.testing.assert_allclose(
        step1.parent_wake_front_strengths_m2_s,
        step0.rear_bound_strengths_m2_s + step0.q_by_strip_m2_s,
        atol=2.0e-14,
        rtol=2.0e-14,
    )
    np.testing.assert_allclose(
        step1.next_wake_front_strengths_m2_s,
        step1.rear_bound_strengths_m2_s + step1.q_by_strip_m2_s,
        atol=2.0e-14,
        rtol=2.0e-14,
    )
    np.testing.assert_array_equal(
        _front_wake_strengths_by_strip(solver.steady_problems[1]),
        step0.next_wake_front_strengths_m2_s,
    )
    np.testing.assert_array_equal(
        _front_wake_strengths_by_strip(solver.steady_problems[2]),
        step1.next_wake_front_strengths_m2_s,
    )
    _assert_load_ledger(step1, step0.q_by_strip_m2_s)
    _assert_load_ledger(step2, step1.q_by_strip_m2_s)


class _ControlledObserver:
    def __init__(
        self,
        parent: Any,
        *,
        suppress: bool,
        force_first_active: bool,
    ) -> None:
        self._parent = parent
        self._suppress = bool(suppress)
        self._force_first_active = bool(force_first_active)
        self._observe_calls = 0

    def __getattr__(self, name: str) -> Any:
        return getattr(self._parent, name)

    def observe(self, gamma_bound: np.ndarray) -> np.ndarray:
        observed = np.asarray(self._parent.observe(gamma_bound), dtype=float)
        self._observe_calls += 1
        if self._suppress:
            return np.zeros_like(observed)
        if self._force_first_active and self._observe_calls == 1:
            return np.where(observed < 0.0, -0.20, 0.20)
        return observed


def test_t4_deactivate_then_reactivate_has_exact_release_and_new_sheet(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    solver = _make_active_solver()
    original_builder = native_solver_module.build_ptera_native_lesp_observer

    def controlled_builder(live_solver: Any) -> _ControlledObserver:
        step = int(live_solver._current_step)
        return _ControlledObserver(
            original_builder(live_solver),
            suppress=step == 1,
            force_first_active=step == 2,
        )

    monkeypatch.setattr(
        native_solver_module,
        "build_ptera_native_lesp_observer",
        controlled_builder,
    )
    _run_until_committed_steps(solver, 3)
    step0, deactivated, reactivated = solver.v5f_step_reports[:3]

    assert np.all(step0.active)
    assert not np.any(deactivated.active)
    np.testing.assert_array_equal(
        deactivated.q_by_strip_m2_s,
        np.zeros_like(deactivated.q_by_strip_m2_s),
    )
    np.testing.assert_array_equal(
        deactivated.created_ring_ids,
        np.empty(0, dtype=np.int64),
    )
    _assert_snapshot_exact(
        deactivated.material_snapshot_before, step0.material_snapshot
    )
    assert (
        deactivated.material_snapshot.gamma_m2_s.tobytes()
        == step0.material_snapshot.gamma_m2_s.tobytes()
    )
    np.testing.assert_array_equal(
        deactivated.material_snapshot.ring_id,
        step0.material_snapshot.ring_id,
    )
    _assert_load_ledger(deactivated, step0.q_by_strip_m2_s)
    assert deactivated.load_delta is not None
    np.testing.assert_array_equal(
        deactivated.load_delta.correction["delta_eq17_force_gp1_n"],
        np.zeros_like(deactivated.load_delta.correction["delta_eq17_force_gp1_n"]),
    )
    np.testing.assert_allclose(
        deactivated.next_wake_front_strengths_m2_s,
        deactivated.rear_bound_strengths_m2_s,
        atol=2.0e-14,
        rtol=2.0e-14,
    )

    assert np.all(reactivated.active)
    np.testing.assert_array_equal(reactivated.new_sheet, reactivated.active)
    created_positions = np.concatenate(
        [
            np.flatnonzero(reactivated.material_snapshot.ring_id == int(created_id))
            for created_id in reactivated.created_ring_ids
        ]
    )
    old_sheet_ids = set(deactivated.material_snapshot.sheet_id.astype(int).tolist())
    new_sheet_ids = set(
        reactivated.material_snapshot.sheet_id[created_positions].astype(int).tolist()
    )
    assert old_sheet_ids.isdisjoint(new_sheet_ids)
    _assert_load_ledger(reactivated, deactivated.q_by_strip_m2_s)


def test_t6_step_one_load_failure_does_not_publish_material_or_next_te(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    solver = _make_active_solver()
    original = native_solver_module.calculate_live_native_surface_release_load_delta
    owner_at_failure: list[PteraMaterialLEVSnapshot] = []

    def fail_step_one(live_solver: Any, *args: Any, **kwargs: Any) -> Any:
        if live_solver is solver and int(live_solver._current_step) == 1:
            owner_at_failure.append(solver.material_lev_state.snapshot())
            raise _InjectedLoadFailure("injected after solve and before TE population")
        return original(live_solver, *args, **kwargs)

    monkeypatch.setattr(
        native_solver_module,
        "calculate_live_native_surface_release_load_delta",
        fail_step_one,
    )
    with pytest.raises(_InjectedLoadFailure):
        solver.run(
            prescribed_wake=True,
            calculate_streamlines=False,
            show_progress=False,
        )

    assert len(solver.v5f_step_reports) == 1
    assert solver.v5f_poisoned is True
    with pytest.raises(RuntimeError, match="solver is poisoned"):
        solver._calculate_wake_wing_influences()
    assert len(owner_at_failure) == 1
    committed_step0 = solver.v5f_step_reports[0].material_snapshot
    _assert_snapshot_exact(owner_at_failure[0], committed_step0)
    _assert_snapshot_exact(solver.material_lev_state.snapshot(), committed_step0)
    np.testing.assert_array_equal(
        _front_wake_strengths_by_strip(solver.steady_problems[1]),
        solver.v5f_step_reports[0].next_wake_front_strengths_m2_s,
    )
    for airplane in solver.steady_problems[2].airplanes:
        for wing in airplane.wings:
            wake = wing.wake_ring_vortices
            assert wake is None or wake.size == 0


def test_t6_step_one_commit_failure_rolls_back_patched_te_and_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    solver = _make_active_solver()
    original = PteraMaterialLEVProposal.commit_and_convect
    failed_proposals: list[PteraMaterialLEVProposal] = []
    owner_at_failure: list[PteraMaterialLEVSnapshot] = []

    def fail_step_one(
        proposal: PteraMaterialLEVProposal,
        created_strengths_m2_s: Any,
        vertex_velocity_ptera_m_s: Any,
        *,
        dt_s: float,
    ) -> Any:
        if proposal.step == 1:
            failed_proposals.append(proposal)
            owner_at_failure.append(solver.material_lev_state.snapshot())
            raise _InjectedCommitFailure("injected after next-TE patch")
        return original(
            proposal,
            created_strengths_m2_s,
            vertex_velocity_ptera_m_s,
            dt_s=dt_s,
        )

    monkeypatch.setattr(PteraMaterialLEVProposal, "commit_and_convect", fail_step_one)
    with pytest.raises(_InjectedCommitFailure):
        solver.run(
            prescribed_wake=True,
            calculate_streamlines=False,
            show_progress=False,
        )

    assert len(solver.v5f_step_reports) == 1
    assert solver.v5f_poisoned is True
    with pytest.raises(RuntimeError, match="solver is poisoned"):
        solver._calculate_wake_wing_influences()
    assert len(failed_proposals) == 1
    assert failed_proposals[0].is_finalized
    committed_step0 = solver.v5f_step_reports[0].material_snapshot
    _assert_snapshot_exact(owner_at_failure[0], committed_step0)
    _assert_snapshot_exact(solver.material_lev_state.snapshot(), committed_step0)

    # Ptera populated step 2 with the parent rear-bound value before v5f
    # patched it.  The failed atomic publish must restore that exact value.
    np.testing.assert_array_equal(
        _front_wake_strengths_by_strip(solver.steady_problems[2]),
        _rear_bound_strengths_by_strip(solver.steady_problems[1]),
    )
