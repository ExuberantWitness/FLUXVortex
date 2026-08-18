from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pterasoftware as ps
import pytest

from forward_flight_benchmarks.augmented_uvpm import AugmentedUVPMHybridSolver
from forward_flight_benchmarks.cases import IZRAELEVITZ_2017_FIG11
from forward_flight_benchmarks.fluxv_v5e_panel_ledger import (
    PANEL_LEDGER_CLOSURE_ATOL_N,
    PanelLedgerUVPMHybridSolver,
    aggregate_panel_cycle_to_strips,
    extract_last_coherent_panel_cycle,
)
from forward_flight_benchmarks.ptera_adapter import build_izraelevitz_fig11_movement


def _fake_solver(*, record: bool = True) -> PanelLedgerUVPMHybridSolver:
    solver = object.__new__(PanelLedgerUVPMHybridSolver)
    solver._record_panel_ledger = record
    solver.panel_load_ledger = []
    solver._current_step = 0
    solver.delta_time = 0.5
    solver._current_bound_vortex_strengths = np.array([0.30, -0.10, 0.50, 0.25])
    solver._last_bound_vortex_strengths = np.array([0.10, -0.15, 0.45, 0.05])
    solver.panel_areas = np.array([1.0, 2.0, 3.0, 4.0])
    solver.stackUnitNormals_GP1 = np.array(
        [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]]
    )
    solver.stackCpp_GP1_CgP1 = np.array(
        [[0.1, -0.2, 0.3], [0.2, 0.1, 0.4], [0.5, -0.3, 0.2], [0.6, 0.4, 0.1]]
    )
    transform = np.array(
        [
            [0.0, -1.0, 0.0, 2.0],
            [1.0, 0.0, 0.0, -3.0],
            [0.0, 0.0, 1.0, 4.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    solver.current_operating_point = SimpleNamespace(
        rho=2.0,
        T_pas_GP1_CgP1_to_W_CgP1=transform,
    )
    panels = np.empty((2, 2), dtype=object)
    for chord in range(2):
        for span in range(2):
            panels[chord, span] = SimpleNamespace(
                local_chordwise_position=chord,
                local_spanwise_position=span,
                forces_GP1=np.zeros(3),
            )
    solver.current_airplanes = [SimpleNamespace(wings=[SimpleNamespace(panels=panels)])]
    solver._test_parent_total_force_gp1_n = np.array(
        [[2.0, 0.5, -1.0], [-1.0, 3.0, 0.25], [0.2, -0.3, 4.0], [1.1, 2.2, -0.7]]
    )
    return solver


def _parent_force_stub(solver: PanelLedgerUVPMHybridSolver) -> None:
    solver._parent_call_count = getattr(solver, "_parent_call_count", 0) + 1
    panels = solver.current_airplanes[0].wings[0].panels
    for panel, force in zip(
        np.ravel(panels), solver._test_parent_total_force_gp1_n, strict=True
    ):
        panel.forces_GP1 = force.copy()
    solver._parent_result = np.sum(solver._test_parent_total_force_gp1_n, axis=0)


def _record_step(
    solver: PanelLedgerUVPMHybridSolver,
    step: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        AugmentedUVPMHybridSolver, "_calculate_loads", _parent_force_stub
    )
    solver._current_step = step
    solver._current_bound_vortex_strengths = (
        np.array([0.30, -0.10, 0.50, 0.25]) + 0.01 * step
    )
    solver._last_bound_vortex_strengths = (
        np.array([0.10, -0.15, 0.45, 0.05]) + 0.01 * step
    )
    PanelLedgerUVPMHybridSolver._calculate_loads(solver)


def test_record_false_is_the_exact_parent_calculation_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[PanelLedgerUVPMHybridSolver] = []

    def parent(instance: PanelLedgerUVPMHybridSolver) -> None:
        calls.append(instance)
        instance._parent_result = np.array([1.0, 2.0, 3.0])

    monkeypatch.setattr(AugmentedUVPMHybridSolver, "_calculate_loads", parent)
    solver = object.__new__(PanelLedgerUVPMHybridSolver)
    solver._record_panel_ledger = False
    solver.panel_load_ledger = []
    PanelLedgerUVPMHybridSolver._calculate_loads(solver)
    assert calls == [solver]
    np.testing.assert_array_equal(solver._parent_result, [1.0, 2.0, 3.0])
    assert solver.panel_load_ledger == []


def test_panel_ledger_reconstructs_exact_ptera_dgamma_and_kj(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    solver = _fake_solver()
    _record_step(solver, 0, monkeypatch)
    assert solver._parent_call_count == 1
    row = solver.panel_load_ledger[0]
    expected_dgamma = -(
        2.0
        * (
            solver._current_bound_vortex_strengths - solver._last_bound_vortex_strengths
        )[:, None]
        * solver.panel_areas[:, None]
        * solver.stackUnitNormals_GP1
        / solver.delta_time
    )
    np.testing.assert_allclose(row["ptera_dgamma_force_gp1_n"], expected_dgamma)
    np.testing.assert_array_equal(
        row["panel_total_force_gp1_n"], solver._test_parent_total_force_gp1_n
    )
    np.testing.assert_allclose(
        row["kutta_joukowski_force_gp1_n"],
        solver._test_parent_total_force_gp1_n - expected_dgamma,
    )
    assert row["closure"]["all_levels_max_abs_n"] < PANEL_LEDGER_CLOSURE_ATOL_N
    for level in ("per_panel", "per_strip", "per_airplane"):
        assert row["closure"][f"{level}_gp1_max_abs_n"] < 1.0e-12
        assert row["closure"][f"{level}_w_max_abs_n"] < 1.0e-12
    np.testing.assert_array_equal(solver._parent_result, [2.3, 5.4, 2.55])


def test_force_uses_rotation_while_collocation_uses_full_frame_transform(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    solver = _fake_solver()
    _record_step(solver, 0, monkeypatch)
    row = solver.panel_load_ledger[0]
    rotation = row["frame_transform_gp1_to_w"][:3, :3]
    translation = row["frame_transform_gp1_to_w"][:3, 3]
    np.testing.assert_allclose(
        row["panel_total_force_w_n"],
        np.einsum("ij,pj->pi", rotation, row["panel_total_force_gp1_n"]),
    )
    np.testing.assert_allclose(
        row["collocation_w_m"],
        np.einsum("ij,pj->pi", rotation, row["collocation_gp1_m"]) + translation,
    )
    np.testing.assert_allclose(
        row["panel_normal_w"],
        np.einsum("ij,pj->pi", rotation, row["panel_normal_gp1"]),
    )


def test_record_is_a_copy_and_never_mutates_parent_loads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    solver = _fake_solver()
    expected_parent = solver._test_parent_total_force_gp1_n.copy()
    _record_step(solver, 0, monkeypatch)
    row = solver.panel_load_ledger[0]
    for panel, expected in zip(
        np.ravel(solver.current_airplanes[0].wings[0].panels),
        expected_parent,
        strict=True,
    ):
        np.testing.assert_array_equal(panel.forces_GP1, expected)
        panel.forces_GP1[:] = 999.0
    solver._current_bound_vortex_strengths[:] = 888.0
    solver.panel_areas[:] = 777.0
    np.testing.assert_array_equal(row["panel_total_force_gp1_n"], expected_parent)
    np.testing.assert_array_equal(row["gamma_current_m2_s"], [0.30, -0.10, 0.50, 0.25])
    np.testing.assert_array_equal(row["panel_area_m2"], [1.0, 2.0, 3.0, 4.0])


def test_extracts_endpoint_safe_last_complete_cycle_and_topology(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    solver = _fake_solver()
    for step in range(5):
        _record_step(solver, step, monkeypatch)
    cycle = extract_last_coherent_panel_cycle(solver, period_s=2.0, delta_time_s=0.5)
    np.testing.assert_array_equal(cycle["step"], [0, 1, 2, 3])
    np.testing.assert_array_equal(cycle["phase"], [0.0, 0.25, 0.5, 0.75])
    assert cycle["source_cycle_step_range"] == [0, 3]
    assert cycle["samples_per_cycle"] == 4
    assert cycle["panel_total_force_gp1_n"].shape == (4, 4, 3)
    np.testing.assert_array_equal(cycle["topology"]["panel_global_index"], [0, 1, 2, 3])
    np.testing.assert_array_equal(cycle["topology"]["chordwise_index"], [0, 0, 1, 1])
    np.testing.assert_array_equal(cycle["topology"]["spanwise_index"], [0, 1, 0, 1])


def test_strip_aggregation_sums_forces_but_retains_every_chordwise_gamma(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    solver = _fake_solver()
    for step in range(5):
        _record_step(solver, step, monkeypatch)
    cycle = extract_last_coherent_panel_cycle(solver, period_s=2.0, delta_time_s=0.5)
    aggregation = aggregate_panel_cycle_to_strips(cycle)
    assert aggregation["equivalent_gamma_rule"] == (
        "not_selected_chordwise_values_retained"
    )
    assert len(aggregation["strips"]) == 2
    first = aggregation["strips"][0]
    np.testing.assert_array_equal(first["panel_global_indices"], [0, 2])
    np.testing.assert_array_equal(first["chordwise_indices"], [0, 1])
    assert first["gamma_current_by_chord_m2_s"].shape == (4, 2)
    np.testing.assert_allclose(
        first["total_force_gp1_n"],
        cycle["panel_total_force_gp1_n"][:, 0] + cycle["panel_total_force_gp1_n"][:, 2],
    )
    np.testing.assert_allclose(
        first["strip_area_m2"],
        cycle["panel_area_m2"][:, 0] + cycle["panel_area_m2"][:, 2],
    )
    assert aggregation["closure_per_strip_max_abs_n"] < 1.0e-12
    assert aggregation["closure_per_airplane_max_abs_n"] < 1.0e-12
    assert len(aggregation["airplanes"]) == 1
    np.testing.assert_allclose(
        aggregation["airplanes"][0]["total_force_w_n"],
        np.sum(cycle["panel_total_force_w_n"], axis=1),
    )


def test_cycle_extractor_fails_closed_on_gap_or_noninteger_period(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    solver = _fake_solver()
    for step in range(5):
        _record_step(solver, step, monkeypatch)
    solver.panel_load_ledger = [
        row for row in solver.panel_load_ledger if row["step"] != 2
    ]
    with pytest.raises(FloatingPointError, match="coherent cycle"):
        extract_last_coherent_panel_cycle(solver, period_s=2.0, delta_time_s=0.5)
    with pytest.raises(ValueError, match="integer number"):
        extract_last_coherent_panel_cycle(solver, period_s=2.1, delta_time_s=0.5)


def test_extractor_requires_one_time_step_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    solver = _fake_solver()
    _record_step(solver, 0, monkeypatch)
    with pytest.raises(ValueError, match="exactly one"):
        extract_last_coherent_panel_cycle(solver, period_s=2.0)
    with pytest.raises(ValueError, match="exactly one"):
        extract_last_coherent_panel_cycle(
            solver,
            period_s=2.0,
            movement=SimpleNamespace(delta_time=0.5),
            delta_time_s=0.5,
        )


def test_real_ptera_smoke_is_bitwise_load_identical_and_closes() -> None:
    parent_histories: list[np.ndarray] = []
    recorded_solver = None
    recorded_movement = None
    for record_panel_ledger in (False, True):
        movement, _ = build_izraelevitz_fig11_movement("smoke")
        problem = ps.problems.UnsteadyProblem(
            movement=movement, only_final_results=False
        )
        solver = PanelLedgerUVPMHybridSolver(
            problem,
            max_particles=20_000,
            stretch=False,
            free_wake=False,
            record_vpm_particles=False,
            record_panel_ledger=record_panel_ledger,
        )
        solver.run(
            prescribed_wake=True,
            calculate_streamlines=False,
            show_progress=False,
        )
        parent_histories.append(
            np.asarray(
                [
                    row["airplanes"][0]["total_force_w_n"]
                    for row in solver.uvlm_load_ledger
                ]
            )
        )
        if record_panel_ledger:
            recorded_solver = solver
            recorded_movement = movement
        else:
            assert solver.panel_load_ledger == []
    np.testing.assert_array_equal(parent_histories[0], parent_histories[1])
    assert recorded_solver is not None
    assert recorded_movement is not None
    cycle = extract_last_coherent_panel_cycle(
        recorded_solver,
        movement=recorded_movement,
        period_s=IZRAELEVITZ_2017_FIG11.period_s,
    )
    aggregation = aggregate_panel_cycle_to_strips(cycle)
    assert cycle["panel_total_force_gp1_n"].shape == (24, 28, 3)
    assert len(aggregation["strips"]) == 14
    assert aggregation["closure_per_panel_max_abs_n"] < 1.0e-12
    assert aggregation["closure_per_strip_max_abs_n"] < 1.0e-12
    assert aggregation["closure_per_airplane_max_abs_n"] < 1.0e-12
