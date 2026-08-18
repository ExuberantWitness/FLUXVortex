from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pterasoftware as ps
import pytest

from fluxvortex.solver import UVPMHybridSolver
from forward_flight_benchmarks import fluxv_v5f_native_lev as native_lev_module
from forward_flight_benchmarks.fluxv_v5f_native_lev import (
    AUGMENTED_RESIDUAL_ATOL,
    FORCE_SCORING_STATUS,
    NativeMaterialLEVConfig,
    NativeMaterialLEVUVPMHybridSolver,
    build_ptera_native_lesp_observer,
    make_fluxv_v5f_solver,
    native_active_probe,
    ptera_native_unit_ring_velocity_grid,
    solve_ptera_native_augmented_system,
)
from forward_flight_benchmarks.ptera_adapter import build_yang2025_movement


def _assert_array_sequences_equal(
    left: tuple[np.ndarray, ...], right: tuple[np.ndarray, ...]
) -> None:
    assert len(left) == len(right)
    for left_array, right_array in zip(left, right, strict=True):
        np.testing.assert_array_equal(left_array, right_array)


def _solved_panel_history(
    solver: UVPMHybridSolver, attribute: str
) -> tuple[np.ndarray, ...]:
    rows: list[np.ndarray] = []
    for problem in solver.steady_problems:
        values: list[np.ndarray] = []
        for airplane in problem.airplanes:
            for wing in airplane.wings:
                for panel in np.ravel(wing.panels):
                    value = getattr(panel, attribute)
                    assert value is not None
                    values.append(np.asarray(value, dtype=float).copy())
        rows.append(np.asarray(values, dtype=float))
    return tuple(rows)


def _bound_gamma_history(solver: UVPMHybridSolver) -> tuple[np.ndarray, ...]:
    rows: list[np.ndarray] = []
    for problem in solver.steady_problems:
        values: list[float] = []
        for airplane in problem.airplanes:
            for wing in airplane.wings:
                for panel in np.ravel(wing.panels):
                    assert panel.ring_vortex is not None
                    values.append(float(panel.ring_vortex.strength))
        rows.append(np.asarray(values, dtype=float))
    return tuple(rows)


def _airplane_load_history(
    solver: UVPMHybridSolver, attribute: str
) -> tuple[np.ndarray, ...]:
    rows: list[np.ndarray] = []
    for problem in solver.steady_problems:
        values: list[np.ndarray] = []
        for airplane in problem.airplanes:
            value = getattr(airplane, attribute)
            assert value is not None
            values.append(np.asarray(value, dtype=float).copy())
        rows.append(np.asarray(values, dtype=float))
    return tuple(rows)


def _run_real_yang(config: NativeMaterialLEVConfig) -> UVPMHybridSolver:
    movement, _ = build_yang2025_movement(15.0, "smoke")
    problem = ps.problems.UnsteadyProblem(
        movement=movement,
        only_final_results=False,
    )
    solver = make_fluxv_v5f_solver(
        problem,
        config=config,
        max_particles=20_000,
        stretch=False,
        free_wake=False,
    )
    solver.run(
        prescribed_wake=True,
        calculate_streamlines=False,
        show_progress=False,
    )
    return solver


def test_disabled_factory_returns_the_original_solver_type_directly() -> None:
    movement, _ = build_yang2025_movement(15.0, "smoke")
    problem = ps.problems.UnsteadyProblem(movement=movement, only_final_results=False)
    solver = make_fluxv_v5f_solver(
        problem,
        config=NativeMaterialLEVConfig(enabled=False, lesp_critical=0.5),
        max_particles=100,
        stretch=False,
        free_wake=False,
    )
    assert type(solver) is UVPMHybridSolver
    assert not isinstance(solver, NativeMaterialLEVUVPMHybridSolver)


def test_finite_enabled_threshold_fails_closed_before_parent_construction() -> None:
    with pytest.raises(NotImplementedError, match="material wake"):
        make_fluxv_v5f_solver(
            object(),
            config=NativeMaterialLEVConfig(enabled=True, lesp_critical=0.25),
        )


def test_force_scoring_is_explicitly_blocked() -> None:
    assert FORCE_SCORING_STATUS == (
        "blocked_until_material_wake_and_ptera_load_corrections"
    )
    assert (
        NativeMaterialLEVUVPMHybridSolver.force_scoring_status == FORCE_SCORING_STATUS
    )


def test_inactive_augmented_helper_is_the_exact_parent_solve() -> None:
    aic = np.array([[3.0, -0.2, 0.1], [0.4, 2.5, -0.3], [-0.1, 0.2, 1.8]])
    rhs = np.array([0.7, -1.2, 0.4])
    influence = np.array([0.2, -0.5, 0.7])
    operator = np.array([1.0, -0.25, 0.1])
    target = np.array([99.0])
    expected = np.linalg.solve(aic, rhs)
    result = solve_ptera_native_augmented_system(
        aic,
        rhs,
        lev_unit_normal_influence=influence,
        lesp_operator=operator,
        lesp_target=target,
        active=False,
    )
    np.testing.assert_array_equal(result.gamma_bound, expected)
    np.testing.assert_array_equal(result.gamma_lev, [0.0])
    np.testing.assert_array_equal(result.augmented_matrix, aic)
    np.testing.assert_array_equal(result.augmented_rhs, rhs)
    assert result.used_augmented_system is False
    assert result.reduction_reason == "inactive"


def test_zero_lev_parent_feasible_constraint_reduces_exactly() -> None:
    aic = np.array([[2.0, 0.3], [-0.1, 1.5]])
    rhs = np.array([0.8, -0.2])
    influence = np.array([[0.4], [-0.6]])
    operator = np.array([[1.0, -0.2]])
    expected = np.linalg.solve(aic, rhs)
    target = operator @ expected
    result = solve_ptera_native_augmented_system(
        aic,
        rhs,
        lev_unit_normal_influence=influence,
        lesp_operator=operator,
        lesp_target=target,
        active=True,
    )
    np.testing.assert_array_equal(result.gamma_bound, expected)
    np.testing.assert_array_equal(result.gamma_lev, [0.0])
    assert result.used_augmented_system is False
    assert result.reduction_reason == "parent_exactly_satisfies_constraint"


def test_active_augmented_helper_solves_literal_block_without_ridge_or_cap() -> None:
    aic = np.array([[4.0, 1.0, 0.2], [0.3, 3.0, -0.4], [0.1, 0.2, 2.5]])
    influence = np.array([[0.2], [-0.5], [0.7]])
    operator = np.array([[1.0, -0.25, 0.1]])
    expected_bound = np.array([0.4, -0.2, 0.1])
    expected_lev = np.array([17.25])
    rhs = aic @ expected_bound + influence @ expected_lev
    target = operator @ expected_bound
    result = solve_ptera_native_augmented_system(
        aic,
        rhs,
        lev_unit_normal_influence=influence,
        lesp_operator=operator,
        lesp_target=target,
        active=True,
    )
    expected_matrix = np.block([[aic, influence], [operator, np.zeros((1, 1))]])
    np.testing.assert_array_equal(result.augmented_matrix, expected_matrix)
    np.testing.assert_array_equal(result.augmented_rhs, np.concatenate((rhs, target)))
    np.testing.assert_allclose(result.gamma_bound, expected_bound, atol=1.0e-13)
    np.testing.assert_allclose(result.gamma_lev, expected_lev, atol=1.0e-13)
    assert result.gamma_lev[0] > 10.0
    assert result.no_penetration_max_abs < 1.0e-12
    assert result.lesp_constraint_max_abs < 1.0e-12
    assert result.used_augmented_system is True
    assert result.reduction_reason is None


def test_active_augmented_helper_rejects_non_square_constraint_count() -> None:
    with pytest.raises(ValueError, match="one LESP constraint"):
        solve_ptera_native_augmented_system(
            np.eye(2),
            np.ones(2),
            lev_unit_normal_influence=np.ones((2, 2)),
            lesp_operator=np.ones((1, 2)),
            lesp_target=np.ones(1),
            active=True,
        )


def test_active_augmented_helper_fails_closed_on_nonfinite_solution() -> None:
    with np.errstate(over="ignore", invalid="ignore"):
        with pytest.raises(FloatingPointError, match="non-finite"):
            solve_ptera_native_augmented_system(
                np.array([[1.0]]),
                np.array([1.0]),
                lev_unit_normal_influence=np.array([[1.0e-310]]),
                lesp_operator=np.array([[1.0]]),
                lesp_target=np.array([0.0]),
                active=True,
            )


def test_ptera_unit_ring_kernel_matches_analytic_orientation_and_sign() -> None:
    ring = np.array(
        [
            [
                [-1.0, -1.0, 0.0],
                [-1.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
                [1.0, -1.0, 0.0],
            ]
        ]
    )
    point = np.array([[0.0, 0.0, 0.7]])
    actual, singularity_counts = ptera_native_unit_ring_velocity_grid(
        point,
        ring,
        np.array([0.0]),
    )

    # Ptera's positive-strength path is back-right -> front-right ->
    # front-left -> back-left -> back-right.  This is an independent analytic
    # finite-segment Biot--Savart evaluation, not a second production kernel.
    vertices = [ring[0, 2], ring[0, 1], ring[0, 0], ring[0, 3], ring[0, 2]]
    expected = np.zeros(3)
    for start, end in zip(vertices[:-1], vertices[1:], strict=True):
        r_start = point[0] - start
        r_end = point[0] - end
        segment = end - start
        cross = np.cross(r_start, r_end)
        expected += (
            cross
            / (4.0 * np.pi * np.dot(cross, cross))
            * np.dot(
                segment,
                r_start / np.linalg.norm(r_start) - r_end / np.linalg.norm(r_end),
            )
        )
    np.testing.assert_allclose(actual[0, 0], expected, atol=2.0e-15, rtol=2.0e-15)
    assert actual[0, 0, 2] > 0.0
    np.testing.assert_array_equal(singularity_counts, np.zeros(4, dtype=int))


def test_real_ptera_observer_and_ring_mapping_reconstruct_eq6_and_aic() -> None:
    solver = _run_real_yang(NativeMaterialLEVConfig(enabled=False, lesp_critical=0.5))
    observer = build_ptera_native_lesp_observer(solver)
    np.testing.assert_allclose(
        observer.observe(solver._current_bound_vortex_strengths),
        observer.direct_eq6(solver._current_bound_vortex_strengths),
        atol=2.0e-15,
        rtol=2.0e-15,
    )
    np.testing.assert_array_equal(observer.leading_panel_indices, [0, 1, 2, 3])
    np.testing.assert_allclose(observer.chord_m, 0.130, atol=2.0e-15, rtol=0.0)
    np.testing.assert_allclose(
        observer.delta_x_front_m,
        0.065,
        atol=2.0e-15,
        rtol=0.0,
    )

    ptera_bound_rings = np.stack(
        (
            solver.stackFlbrvp_GP1_CgP1,
            solver.stackFrbrvp_GP1_CgP1,
            solver.stackBrbrvp_GP1_CgP1,
            solver.stackBlbrvp_GP1_CgP1,
        ),
        axis=1,
    )
    unit_velocity, singularity_counts = ptera_native_unit_ring_velocity_grid(
        solver.stackCpp_GP1_CgP1,
        ptera_bound_rings,
        solver._currentStackBoundRc0s,
        nu_m2_s=solver.current_operating_point.nu,
    )
    reconstructed_aic = np.einsum(
        "ijk,ik->ij", unit_velocity, solver.stackUnitNormals_GP1
    )
    np.testing.assert_array_equal(
        reconstructed_aic,
        solver._currentGridWingWingInfluences__E,
    )
    np.testing.assert_array_equal(singularity_counts, np.zeros(4, dtype=int))


def _wake_state(solver: UVPMHybridSolver) -> tuple[tuple[np.ndarray, ...], ...]:
    attributes = (
        "_list_wake_vortex_strengths",
        "_list_wake_vortex_ages",
        "listStackBrwrvp_GP1_CgP1",
        "listStackFrwrvp_GP1_CgP1",
        "listStackFlwrvp_GP1_CgP1",
        "listStackBlwrvp_GP1_CgP1",
    )
    return tuple(
        tuple(np.asarray(row).copy() for row in getattr(solver, attribute))
        for attribute in attributes
    )


def _current_load_state(solver: UVPMHybridSolver) -> tuple[np.ndarray, ...]:
    values: list[np.ndarray] = []
    for panel in np.ravel(solver.panels):
        values.extend(
            (
                np.asarray(panel.forces_GP1).copy(),
                np.asarray(panel.moments_GP1_CgP1).copy(),
            )
        )
    for airplane in solver.current_airplanes:
        values.extend(
            (
                np.asarray(airplane.forces_W).copy(),
                np.asarray(airplane.moments_W_CgP1).copy(),
            )
        )
    return tuple(values)


def _assert_nested_array_sequences_equal(
    left: tuple[tuple[np.ndarray, ...], ...],
    right: tuple[tuple[np.ndarray, ...], ...],
) -> None:
    assert len(left) == len(right)
    for left_rows, right_rows in zip(left, right, strict=True):
        _assert_array_sequences_equal(left_rows, right_rows)


def test_real_native_active_probe_closes_without_wake_or_load_commit() -> None:
    solver = _run_real_yang(NativeMaterialLEVConfig(enabled=False, lesp_critical=0.5))
    wake_before = _wake_state(solver)
    loads_before = _current_load_state(solver)
    particle_before = tuple(
        getattr(solver._vpm_field, attribute).copy()
        for attribute in ("_pos", "_gamma", "_sigma", "_age")
    )
    parent_gamma = solver._current_bound_vortex_strengths.copy()

    report = native_active_probe(
        solver,
        lesp_critical=0.03,
        commit_bound_strengths=True,
    )
    np.testing.assert_array_equal(report.active, [True, False, True, True])
    assert report.committed_bound_strengths is True
    assert report.force_scoring_status == FORCE_SCORING_STATUS
    assert report.augmented_solution.used_augmented_system is True
    assert report.augmented_solution.no_penetration_max_abs < 1.0e-10
    assert report.augmented_solution.lesp_constraint_max_abs < 1.0e-10
    assert np.max(np.abs(report.a0_post[report.active])) == pytest.approx(
        0.03, abs=1.0e-14
    )
    assert report.birth_influence.combined_unit_normal_influence.shape == (8, 3)
    newborn = report.birth_influence.newborn_rings_gp1_m
    pseudo = report.birth_influence.pseudovortex_rings_gp1_m
    np.testing.assert_array_equal(
        (newborn[:, 0] - newborn[:, 1]) + (pseudo[:, 0] - pseudo[:, 1]),
        np.zeros_like(newborn[:, 0]),
    )
    active_indices = report.birth_influence.active_strip_indices
    np.testing.assert_array_equal(
        pseudo[:, 2],
        report.observer.rear_bound_aft_edges_gp1_m[active_indices, 0],
    )
    np.testing.assert_array_equal(
        pseudo[:, 3],
        report.observer.rear_bound_aft_edges_gp1_m[active_indices, 1],
    )
    np.testing.assert_array_equal(
        report.birth_influence.singularity_counts,
        np.zeros(4, dtype=int),
    )
    assert not np.array_equal(report.proposed_gamma_bound, parent_gamma)
    np.testing.assert_array_equal(
        solver._current_bound_vortex_strengths,
        report.proposed_gamma_bound,
    )
    np.testing.assert_array_equal(
        [panel.ring_vortex.strength for panel in np.ravel(solver.panels)],
        report.proposed_gamma_bound,
    )

    _assert_nested_array_sequences_equal(wake_before, _wake_state(solver))
    _assert_array_sequences_equal(loads_before, _current_load_state(solver))
    for before, attribute in zip(
        particle_before,
        ("_pos", "_gamma", "_sigma", "_age"),
        strict=True,
    ):
        np.testing.assert_array_equal(before, getattr(solver._vpm_field, attribute))


def test_real_inactive_probe_is_bitwise_parent_and_commits_nothing() -> None:
    solver = _run_real_yang(NativeMaterialLEVConfig(enabled=False, lesp_critical=0.5))
    gamma_before = solver._current_bound_vortex_strengths.copy()
    rings_before = np.asarray(
        [panel.ring_vortex.strength for panel in np.ravel(solver.panels)]
    )
    wake_before = _wake_state(solver)
    loads_before = _current_load_state(solver)
    report = native_active_probe(
        solver,
        lesp_critical=0.1,
        commit_bound_strengths=True,
    )
    assert not np.any(report.active)
    assert report.committed_bound_strengths is False
    np.testing.assert_array_equal(report.proposed_gamma_bound, gamma_before)
    np.testing.assert_array_equal(solver._current_bound_vortex_strengths, gamma_before)
    np.testing.assert_array_equal(
        [panel.ring_vortex.strength for panel in np.ravel(solver.panels)],
        rings_before,
    )
    _assert_nested_array_sequences_equal(wake_before, _wake_state(solver))
    _assert_array_sequences_equal(loads_before, _current_load_state(solver))


def test_failed_real_active_proposal_has_zero_state_side_effects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    solver = _run_real_yang(NativeMaterialLEVConfig(enabled=False, lesp_critical=0.5))
    gamma_before = solver._current_bound_vortex_strengths.copy()
    rings_before = np.asarray(
        [panel.ring_vortex.strength for panel in np.ravel(solver.panels)]
    )
    wake_before = _wake_state(solver)
    loads_before = _current_load_state(solver)

    def zero_influence(
        *args: object, **kwargs: object
    ) -> tuple[np.ndarray, np.ndarray]:
        ring_count = np.asarray(args[1]).shape[0]
        del args, kwargs
        return (
            np.zeros((gamma_before.size, ring_count)),
            np.zeros(4, dtype=np.int64),
        )

    monkeypatch.setattr(
        native_lev_module,
        "_ptera_native_ring_normal_influence",
        zero_influence,
    )
    with pytest.raises(FloatingPointError, match="singular"):
        native_active_probe(
            solver,
            lesp_critical=0.03,
            commit_bound_strengths=True,
        )
    np.testing.assert_array_equal(solver._current_bound_vortex_strengths, gamma_before)
    np.testing.assert_array_equal(
        [panel.ring_vortex.strength for panel in np.ravel(solver.panels)],
        rings_before,
    )
    _assert_nested_array_sequences_equal(wake_before, _wake_state(solver))
    _assert_array_sequences_equal(loads_before, _current_load_state(solver))
    assert AUGMENTED_RESIDUAL_ATOL == 1.0e-10


def test_real_yang_unreachable_threshold_is_bitwise_parent_identical() -> None:
    parent = _run_real_yang(NativeMaterialLEVConfig(enabled=False, lesp_critical=0.5))
    candidate = _run_real_yang(
        NativeMaterialLEVConfig(enabled=True, lesp_critical=np.inf)
    )
    assert type(parent) is UVPMHybridSolver
    assert type(candidate) is NativeMaterialLEVUVPMHybridSolver
    assert candidate.material_lev_history == ()
    assert candidate.force_scoring_status == FORCE_SCORING_STATUS

    _assert_array_sequences_equal(
        _bound_gamma_history(parent), _bound_gamma_history(candidate)
    )
    np.testing.assert_array_equal(
        parent._current_bound_vortex_strengths,
        candidate._current_bound_vortex_strengths,
    )

    for attribute in (
        "_list_wake_vortex_strengths",
        "_list_wake_vortex_ages",
        "listStackBrwrvp_GP1_CgP1",
        "listStackFrwrvp_GP1_CgP1",
        "listStackFlwrvp_GP1_CgP1",
        "listStackBlwrvp_GP1_CgP1",
    ):
        _assert_array_sequences_equal(
            tuple(np.asarray(row) for row in getattr(parent, attribute)),
            tuple(np.asarray(row) for row in getattr(candidate, attribute)),
        )

    for attribute in (
        "forces_GP1",
        "moments_GP1_CgP1",
        "forces_W",
        "moments_W_CgP1",
    ):
        _assert_array_sequences_equal(
            _solved_panel_history(parent, attribute),
            _solved_panel_history(candidate, attribute),
        )
    for attribute in (
        "forces_W",
        "forceCoefficients_W",
        "moments_W_CgP1",
        "momentCoefficients_W_CgP1",
    ):
        _assert_array_sequences_equal(
            _airplane_load_history(parent, attribute),
            _airplane_load_history(candidate, attribute),
        )

    assert parent._vpm_field.np == candidate._vpm_field.np
    particle_count = parent._vpm_field.np
    for attribute in ("_pos", "_gamma", "_sigma", "_age"):
        np.testing.assert_array_equal(
            getattr(parent._vpm_field, attribute)[:particle_count],
            getattr(candidate._vpm_field, attribute)[:particle_count],
        )

    ptera_rhs = -candidate._currentStackWakeWingInfluences__E - (
        candidate._currentStackFreestreamWingInfluences__E
    )
    native_reduction = solve_ptera_native_augmented_system(
        candidate._currentGridWingWingInfluences__E,
        ptera_rhs,
        active=False,
    )
    np.testing.assert_array_equal(
        native_reduction.gamma_bound,
        candidate._current_bound_vortex_strengths,
    )


def test_module_imports_no_alternate_bound_aic_or_force_owner() -> None:
    module_path = (
        Path(__file__).resolve().parents[1]
        / "forward_flight_benchmarks"
        / "fluxv_v5f_native_lev.py"
    )
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    imported_modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported_modules.add(node.module)
    assert not any("v5b" in name for name in imported_modules)
    assert not any("hirato_live_shadow" in name for name in imported_modules)
    assert not any(
        "unified" in name and "pressure" in name for name in imported_modules
    )
