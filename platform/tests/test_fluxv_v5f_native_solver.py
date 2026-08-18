from __future__ import annotations

from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any

import numpy as np
import pterasoftware as ps
import pytest
from pterasoftware import _functions

from claim_runtime.hirato_shadow import ring_field_velocity_lamb_oseen
from fluxvortex.solver import UVPMHybridSolver
from forward_flight_benchmarks.fluxv_v5f_material_state import (
    ptera_to_shadow_forward_rings,
)
from forward_flight_benchmarks.fluxv_v5f_native_solver import (
    NativeMaterialLEVTimeMarchSolver,
    NativeMaterialLEVTimeMarchConfig,
    eq25_velocity_from_ptera_material_rings,
    make_fluxv_v5f_native_time_march_solver,
)
from forward_flight_benchmarks.ptera_adapter import build_yang2025_movement


def _yang_problem() -> ps.problems.UnsteadyProblem:
    movement, _ = build_yang2025_movement(15.0, "smoke")
    return ps.problems.UnsteadyProblem(
        movement=movement,
        only_final_results=False,
    )


def _make_solver(
    config: NativeMaterialLEVTimeMarchConfig,
) -> UVPMHybridSolver:
    return make_fluxv_v5f_native_time_march_solver(
        _yang_problem(),
        config=config,
        max_particles=20_000,
        stretch=False,
        free_wake=False,
    )


def _run_solver(config: NativeMaterialLEVTimeMarchConfig) -> UVPMHybridSolver:
    solver = _make_solver(config)
    solver.run(
        prescribed_wake=True,
        calculate_streamlines=False,
        show_progress=False,
    )
    return solver


def _iter_panels(problem: Any) -> Iterator[Any]:
    for airplane in problem.airplanes:
        for wing in airplane.wings:
            yield from np.ravel(wing.panels)


def _panel_history(
    solver: UVPMHybridSolver,
    attribute: str,
) -> tuple[np.ndarray, ...]:
    history: list[np.ndarray] = []
    for problem in solver.steady_problems:
        values: list[np.ndarray] = []
        for panel in _iter_panels(problem):
            value = getattr(panel, attribute)
            assert value is not None
            values.append(np.asarray(value, dtype=float).copy())
        history.append(np.asarray(values, dtype=float))
    return tuple(history)


def _bound_gamma_history(solver: UVPMHybridSolver) -> tuple[np.ndarray, ...]:
    history: list[np.ndarray] = []
    for problem in solver.steady_problems:
        values: list[float] = []
        for panel in _iter_panels(problem):
            assert panel.ring_vortex is not None
            values.append(float(panel.ring_vortex.strength))
        history.append(np.asarray(values, dtype=float))
    return tuple(history)


def _airplane_history(
    solver: UVPMHybridSolver,
    attribute: str,
) -> tuple[np.ndarray, ...]:
    history: list[np.ndarray] = []
    for problem in solver.steady_problems:
        values: list[np.ndarray] = []
        for airplane in problem.airplanes:
            value = getattr(airplane, attribute)
            assert value is not None
            values.append(np.asarray(value, dtype=float).copy())
        history.append(np.asarray(values, dtype=float))
    return tuple(history)


def _assert_array_history_exact(
    parent: tuple[np.ndarray, ...],
    candidate: tuple[np.ndarray, ...],
) -> None:
    assert len(parent) == len(candidate)
    for parent_row, candidate_row in zip(parent, candidate, strict=True):
        np.testing.assert_array_equal(candidate_row, parent_row)


@pytest.fixture(scope="module")
def exact_parent_and_inactive_candidate() -> tuple[UVPMHybridSolver, UVPMHybridSolver]:
    parent = _run_solver(
        NativeMaterialLEVTimeMarchConfig(enabled=False, lesp_critical=0.25)
    )
    candidate = _run_solver(
        NativeMaterialLEVTimeMarchConfig(enabled=True, lesp_critical=np.inf)
    )
    return parent, candidate


def test_t0_disabled_factory_is_the_direct_uvpm_parent() -> None:
    solver = _make_solver(
        NativeMaterialLEVTimeMarchConfig(enabled=False, lesp_critical=0.25)
    )
    assert type(solver) is UVPMHybridSolver


def test_t0_infinite_threshold_real_yang_is_bitwise_parent_exact(
    exact_parent_and_inactive_candidate: tuple[
        UVPMHybridSolver,
        UVPMHybridSolver,
    ],
) -> None:
    parent, candidate = exact_parent_and_inactive_candidate
    assert type(parent) is UVPMHybridSolver
    assert type(candidate) is not UVPMHybridSolver

    _assert_array_history_exact(
        _bound_gamma_history(parent),
        _bound_gamma_history(candidate),
    )
    np.testing.assert_array_equal(
        candidate._current_bound_vortex_strengths,
        parent._current_bound_vortex_strengths,
    )

    for attribute in (
        "_list_wake_vortex_strengths",
        "_list_wake_vortex_ages",
        "listStackBrwrvp_GP1_CgP1",
        "listStackFrwrvp_GP1_CgP1",
        "listStackFlwrvp_GP1_CgP1",
        "listStackBlwrvp_GP1_CgP1",
    ):
        _assert_array_history_exact(
            tuple(np.asarray(row) for row in getattr(parent, attribute)),
            tuple(np.asarray(row) for row in getattr(candidate, attribute)),
        )

    for attribute in (
        "forces_GP1",
        "moments_GP1_CgP1",
        "forces_W",
        "moments_W_CgP1",
    ):
        _assert_array_history_exact(
            _panel_history(parent, attribute),
            _panel_history(candidate, attribute),
        )
    for attribute in (
        "forces_W",
        "forceCoefficients_W",
        "moments_W_CgP1",
        "momentCoefficients_W_CgP1",
    ):
        _assert_array_history_exact(
            _airplane_history(parent, attribute),
            _airplane_history(candidate, attribute),
        )

    assert candidate._vpm_field.np == parent._vpm_field.np
    particle_count = parent._vpm_field.np
    for attribute in ("_pos", "_gamma", "_sigma", "_age"):
        np.testing.assert_array_equal(
            getattr(candidate._vpm_field, attribute)[:particle_count],
            getattr(parent._vpm_field, attribute)[:particle_count],
        )

    material = candidate.material_lev_state.snapshot()
    assert material.gamma_m2_s.size == 0


def test_t1_eq25_uses_ptera_orientation_and_per_ring_fixed_cores() -> None:
    shadow_forward_rings = np.array(
        [
            [
                [0.00, -0.30, 0.00],
                [0.00, 0.25, 0.00],
                [0.36, 0.25, 0.08],
                [0.36, -0.30, 0.08],
            ],
            [
                [0.25, 0.20, -0.04],
                [0.25, 0.70, -0.04],
                [0.72, 0.70, 0.03],
                [0.72, 0.20, 0.03],
            ],
        ],
        dtype=float,
    )
    ptera_rings = shadow_forward_rings[:, [1, 0, 3, 2]]
    points = np.array(
        [
            [0.11, 0.03, 0.013],
            [0.43, 0.39, 0.11],
            [0.95, 0.12, -0.08],
        ]
    )
    gamma = np.array([0.37, -0.21])
    core_radius = np.array([0.017, 0.061])

    expected = np.zeros_like(points)
    recovered_forward = ptera_to_shadow_forward_rings(ptera_rings)
    for ring, circulation, core in zip(
        recovered_forward,
        gamma,
        core_radius,
        strict=True,
    ):
        expected += ring_field_velocity_lamb_oseen(
            points,
            ring[None, :, :],
            np.array([circulation]),
            float(core),
        )

    actual = eq25_velocity_from_ptera_material_rings(
        points,
        ptera_rings,
        gamma,
        core_radius,
    )
    np.testing.assert_allclose(actual, expected, atol=3.0e-15, rtol=3.0e-15)

    # A Ptera age/core-growth law would change this result.  The production
    # helper receives and uses only the core frozen for each material ring.
    altered_core = eq25_velocity_from_ptera_material_rings(
        points,
        ptera_rings,
        gamma,
        1.8 * core_radius,
    )
    assert np.max(np.abs(actual - altered_core)) > 1.0e-6


def test_t1_eq25_empty_material_field_is_exact_positive_zero() -> None:
    points = np.array([[0.1, -0.2, 0.3], [1.2, 0.4, -0.7]])
    actual = eq25_velocity_from_ptera_material_rings(
        points,
        np.empty((0, 4, 3)),
        np.empty(0),
        np.empty(0),
    )
    np.testing.assert_array_equal(actual, np.zeros_like(points))
    assert not np.any(np.signbit(actual))


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
            assert wing.wake_ring_vortices.shape[0] >= 1
            for vortex in wing.wake_ring_vortices[0]:
                values.append(float(vortex.strength))
    return np.asarray(values, dtype=float)


def test_registered_core_uses_one_global_nonuniform_grid_scale() -> None:
    solver = object.__new__(NativeMaterialLEVTimeMarchSolver)
    solver.v5f_config = NativeMaterialLEVTimeMarchConfig(
        enabled=True,
        lesp_critical=0.20,
        core_radius_ratio=0.25,
    )
    solver.delta_time = 0.10
    observer = SimpleNamespace(
        leading_edges_gp1_m=np.asarray(
            [
                [[0.0, 0.0, 0.0], [0.0, 0.03, 0.0]],
                [[0.0, 0.03, 0.0], [0.0, 0.11, 0.0]],
                [[0.0, 0.11, 0.0], [0.0, 0.31, 0.0]],
            ],
            dtype=float,
        ),
        freestream_speed_m_s=2.0,
    )

    actual = solver._registered_core_radii(observer)
    birth_displacement = 2.0 * 0.20 * 0.10 / np.sqrt(2.0)
    expected = 0.25 * min(0.03, birth_displacement)
    np.testing.assert_array_equal(actual, np.full(3, expected))


def test_t2_step_zero_closes_solve_load_eq9_and_atomic_material_commit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    solver = _make_solver(
        NativeMaterialLEVTimeMarchConfig(
            enabled=True,
            lesp_critical=0.10,
            core_radius_ratio=0.25,
        )
    )
    process_calls: list[tuple[int, np.ndarray, np.ndarray]] = []
    parent_process_solver_loads = _functions.process_solver_loads

    def recording_process_solver_loads(
        live_solver: UVPMHybridSolver,
        forces_gp1: np.ndarray,
        moments_gp1_cgp1: np.ndarray,
    ) -> None:
        if live_solver is solver:
            process_calls.append(
                (
                    int(live_solver._current_step),
                    np.asarray(forces_gp1, dtype=float).copy(),
                    np.asarray(moments_gp1_cgp1, dtype=float).copy(),
                )
            )
        parent_process_solver_loads(live_solver, forces_gp1, moments_gp1_cgp1)

    monkeypatch.setattr(
        _functions,
        "process_solver_loads",
        recording_process_solver_loads,
    )

    class StopAfterFirstCommit(RuntimeError):
        pass

    original_wake = solver._calculate_wake_wing_influences

    def stop_at_step_one() -> None:
        if int(solver._current_step) >= 1:
            raise StopAfterFirstCommit
        original_wake()

    solver._calculate_wake_wing_influences = stop_at_step_one  # type: ignore[method-assign]

    with pytest.raises(StopAfterFirstCommit):
        solver.run(
            prescribed_wake=True,
            calculate_streamlines=False,
            show_progress=False,
        )

    assert solver._current_step == 1
    assert len(solver.v5f_step_reports) == 1
    report = solver.v5f_step_reports[0]
    assert report.step == 0
    assert report.force_scoring_status == "blocked_mechanical_time_march_unscored"
    assert report.continuation_status == "mechanical_continuation_unscored"
    np.testing.assert_array_equal(report.active, np.ones(4, dtype=bool))
    assert np.all(np.abs(report.a0_pre) > 0.10)
    np.testing.assert_allclose(
        report.a0_post,
        0.10 * np.sign(report.a0_pre),
        atol=1.0e-10,
        rtol=0.0,
    )
    assert report.augmented_solution.used_augmented_system is True
    assert report.augmented_solution.no_penetration_max_abs < 1.0e-10
    assert report.augmented_solution.lesp_constraint_max_abs < 1.0e-10
    assert report.load_delta is not None
    delta_force = report.load_delta.correction["delta_total_force_gp1_n"]
    delta_moment = report.load_delta.correction["delta_total_moment_gp1_cgp1_nm"]
    assert np.all(np.isfinite(delta_force))
    assert np.max(np.abs(delta_force)) > 0.0
    step_zero_process_calls = [row for row in process_calls if row[0] == 0]
    assert len(step_zero_process_calls) == 2
    _, parent_force, parent_moment = step_zero_process_calls[0]
    _, corrected_force, corrected_moment = step_zero_process_calls[1]
    np.testing.assert_array_equal(corrected_force, parent_force + delta_force)
    np.testing.assert_array_equal(corrected_moment, parent_moment + delta_moment)

    q = np.asarray(report.q_by_strip_m2_s, dtype=float)
    active = np.asarray(report.active, dtype=bool)
    assert q.shape == active.shape
    np.testing.assert_array_equal(q[~active], np.zeros(np.count_nonzero(~active)))
    assert np.all(np.isfinite(q[active]))
    assert np.max(np.abs(q[active])) > 0.0
    strip_widths: list[float] = []
    for airplane in solver.steady_problems[0].airplanes:
        for wing in airplane.wings:
            for span_index in range(wing.panels.shape[1]):
                leading_panel = wing.panels[0, span_index]
                strip_widths.append(
                    float(
                        np.linalg.norm(
                            np.asarray(leading_panel.Frpp_GP1_CgP1)
                            - np.asarray(leading_panel.Flpp_GP1_CgP1)
                        )
                    )
                )
    freestream_speed = float(
        np.linalg.norm(
            solver.steady_problems[0].operating_point.vInf_GP1__E,
        )
    )
    birth_displacement = freestream_speed * 0.10 * solver.delta_time / np.sqrt(2.0)
    expected_core = 0.25 * min(min(strip_widths), birth_displacement)
    np.testing.assert_allclose(
        report.core_radius_by_strip_m,
        np.full(len(strip_widths), expected_core),
        atol=2.0e-16,
        rtol=2.0e-15,
    )
    topology = report.load_delta.operands.topology
    panel_q = q[np.asarray(topology["spanwise_index"], dtype=np.int64)]
    np.testing.assert_array_equal(
        report.load_delta.gamma_lev_current_m2_s,
        panel_q,
    )
    np.testing.assert_array_equal(
        report.load_delta.gamma_lev_previous_m2_s,
        np.zeros_like(panel_q),
    )
    np.testing.assert_array_equal(
        report.load_delta.wake_release_previous_m2_s,
        np.zeros_like(panel_q),
    )

    rear_bound = _rear_bound_strengths_by_strip(solver.steady_problems[0])
    next_front_wake = _front_wake_strengths_by_strip(solver.steady_problems[1])
    np.testing.assert_allclose(
        next_front_wake,
        rear_bound + q,
        atol=2.0e-14,
        rtol=2.0e-14,
    )
    np.testing.assert_array_equal(report.rear_bound_strengths_m2_s, rear_bound)
    np.testing.assert_array_equal(
        report.next_wake_front_strengths_m2_s,
        next_front_wake,
    )
    assert report.eq9_max_abs_m2_s <= 1.0e-12

    snapshot = solver.material_lev_state.snapshot()
    assert snapshot.last_committed_step == 0
    assert snapshot.last_convected_step == 0
    assert snapshot.version == 1
    np.testing.assert_array_equal(snapshot.strip, np.flatnonzero(active))
    np.testing.assert_allclose(
        snapshot.gamma_m2_s,
        q[active],
        atol=2.0e-14,
        rtol=2.0e-14,
    )
    assert np.all(np.isfinite(snapshot.ptera_rings_gp1_m))
    assert np.all(np.isfinite(snapshot.previous_vertex_velocity_ptera_m_s))
    np.testing.assert_array_equal(
        report.material_snapshot.ptera_rings_gp1_m,
        snapshot.ptera_rings_gp1_m,
    )
    np.testing.assert_array_equal(
        report.material_snapshot.gamma_m2_s,
        snapshot.gamma_m2_s,
    )
