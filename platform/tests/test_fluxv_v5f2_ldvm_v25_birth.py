from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pterasoftware as ps
import pytest

from claim_runtime.hirato_equations import (
    embed_chord_normal_displacement,
    first_lev_displacement_ramesh_2d,
)
from fluxvortex.solver import UVPMHybridSolver
from forward_flight_benchmarks.fluxv_v5f_material_state import (
    PteraNativeMaterialLEVState,
)
from forward_flight_benchmarks.fluxv_v5f_native_solver import (
    PTERA_LDVM_V25_BIRTH_MAPPING_STATUS,
    NativeMaterialLEVLDVMV25BirthTimeMarchSolver,
    NativeMaterialLEVTimeMarchConfig,
    NativeMaterialLEVTimeMarchSolver,
    assemble_ptera_mapped_ldvm_v25_local_edge_velocity,
    make_fluxv_v5f2_ldvm_v25_birth_time_march_solver,
    make_fluxv_v5f_native_time_march_solver,
)
from forward_flight_benchmarks.ptera_adapter import build_yang2025_movement


def _yang_problem() -> ps.problems.UnsteadyProblem:
    movement, _ = build_yang2025_movement(15.0, "smoke")
    return ps.problems.UnsteadyProblem(
        movement=movement,
        only_final_results=False,
    )


def _config(*, enabled: bool = True) -> NativeMaterialLEVTimeMarchConfig:
    return NativeMaterialLEVTimeMarchConfig(
        enabled=enabled,
        lesp_critical=0.10,
        core_radius_ratio=0.25,
    )


def test_local_edge_ledger_sums_only_declared_terms_and_removes_spanwise() -> None:
    shape = (2, 2, 3)
    freestream = np.broadcast_to(np.array([2.0, 0.4, -0.2]), shape).copy()
    parent_te = np.full(shape, 0.1)
    committed_old = np.full(shape, -0.03)
    apparent_movement = np.zeros(shape)
    apparent_movement[..., 2] = 0.5
    span_tangent = np.array([[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]])

    ledger = assemble_ptera_mapped_ldvm_v25_local_edge_velocity(
        step=3,
        new_sheet=np.array([True, False]),
        freestream_m_s=freestream,
        current_parent_te_wake_induced_m_s=parent_te,
        committed_old_material_induced_m_s=committed_old,
        apparent_movement_m_s=apparent_movement,
        span_tangent_gp1=span_tangent,
    )

    raw = freestream + parent_te + committed_old + apparent_movement
    expected = raw.copy()
    expected[..., 1] = 0.0
    np.testing.assert_array_equal(ledger.raw_total_local_edge_velocity_m_s, raw)
    np.testing.assert_allclose(
        ledger.total_local_edge_velocity_m_s,
        expected,
        atol=0.0,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        np.einsum(
            "sej,sj->se",
            ledger.total_local_edge_velocity_m_s,
            span_tangent,
        ),
        0.0,
        atol=2.0e-16,
        rtol=0.0,
    )
    for excluded in (
        ledger.excluded_bound_induced_m_s,
        ledger.excluded_current_newborn_induced_m_s,
        ledger.excluded_pseudovortex_induced_m_s,
    ):
        np.testing.assert_array_equal(excluded, np.zeros(shape))
        assert not np.any(np.signbit(excluded))
    assert ledger.new_sheet.tolist() == [True, False]
    assert ledger.mapping_status == PTERA_LDVM_V25_BIRTH_MAPPING_STATUS
    assert "current-step-provisional-TEV-unavailable" in ledger.mapping_status


def test_legacy_v5f_hook_remains_the_2012_array_path() -> None:
    solver = object.__new__(NativeMaterialLEVTimeMarchSolver)
    solver.delta_time = 0.04
    observer = SimpleNamespace(
        freestream_speed_m_s=3.2,
        leading_edges_gp1_m=np.array(
            [
                [[0.0, -0.5, 0.0], [0.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [0.0, 0.5, 0.0]],
            ]
        ),
        chord_tangent_gp1=np.array([[np.sqrt(0.5), 0.0, -np.sqrt(0.5)]] * 2),
        suction_normal_gp1=np.array([[np.sqrt(0.5), 0.0, np.sqrt(0.5)]] * 2),
    )
    a0 = np.array([0.18, -0.21])
    alpha = np.arctan2(
        -observer.chord_tangent_gp1[:, 2],
        observer.chord_tangent_gp1[:, 0],
    )
    expected = (
        observer.leading_edges_gp1_m
        + embed_chord_normal_displacement(
            first_lev_displacement_ramesh_2d(
                observer.freestream_speed_m_s,
                a0,
                alpha,
                solver.delta_time,
            ),
            observer.chord_tangent_gp1,
            observer.suction_normal_gp1,
        )[:, None, :]
    )

    actual = solver._first_or_restart_aft_edges(
        observer,
        a0,
        np.array([True, False]),
    )
    np.testing.assert_array_equal(actual, expected)


def test_explicit_factory_is_required_and_core_rule_is_inherited() -> None:
    legacy = make_fluxv_v5f_native_time_march_solver(
        _yang_problem(),
        config=_config(),
        max_particles=20_000,
        stretch=False,
        free_wake=False,
    )
    candidate = make_fluxv_v5f2_ldvm_v25_birth_time_march_solver(
        _yang_problem(),
        config=_config(),
        max_particles=20_000,
        stretch=False,
        free_wake=False,
    )
    disabled = make_fluxv_v5f2_ldvm_v25_birth_time_march_solver(
        _yang_problem(),
        config=_config(enabled=False),
        max_particles=20_000,
        stretch=False,
        free_wake=False,
    )

    assert type(legacy) is NativeMaterialLEVTimeMarchSolver
    assert type(candidate) is NativeMaterialLEVLDVMV25BirthTimeMarchSolver
    assert type(disabled) is UVPMHybridSolver
    assert (
        NativeMaterialLEVLDVMV25BirthTimeMarchSolver._registered_core_radii
        is NativeMaterialLEVTimeMarchSolver._registered_core_radii
    )


def test_first_step_uses_half_projected_edge_velocity_with_causal_zero_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    solver = make_fluxv_v5f2_ldvm_v25_birth_time_march_solver(
        _yang_problem(),
        config=_config(),
        max_particles=20_000,
        stretch=False,
        free_wake=False,
    )
    assert isinstance(solver, NativeMaterialLEVLDVMV25BirthTimeMarchSolver)
    proposed: list[dict[str, Any]] = []
    parent_propose = solver.material_lev_state.propose_shed

    def recording_propose(**kwargs: Any):
        proposed.append(
            {
                key: np.asarray(value).copy()
                if isinstance(value, np.ndarray)
                else value
                for key, value in kwargs.items()
            }
        )
        return parent_propose(**kwargs)

    monkeypatch.setattr(solver.material_lev_state, "propose_shed", recording_propose)

    class StopAfterFirstCommit(RuntimeError):
        pass

    parent_wake_hook = solver._calculate_wake_wing_influences

    def stop_at_step_one() -> None:
        if int(solver._current_step) >= 1:
            raise StopAfterFirstCommit
        parent_wake_hook()

    monkeypatch.setattr(solver, "_calculate_wake_wing_influences", stop_at_step_one)
    with pytest.raises(StopAfterFirstCommit):
        solver.run(
            prescribed_wake=True,
            calculate_streamlines=False,
            show_progress=False,
        )

    assert len(solver.v5f2_birth_velocity_ledgers) == 1
    assert len(proposed) == 1
    ledger = solver.v5f2_birth_velocity_ledgers[0]
    np.testing.assert_array_equal(
        ledger.apparent_movement_m_s,
        np.zeros_like(ledger.apparent_movement_m_s),
    )
    np.testing.assert_array_equal(
        ledger.current_parent_te_wake_induced_m_s,
        np.zeros_like(ledger.current_parent_te_wake_induced_m_s),
    )
    np.testing.assert_array_equal(
        ledger.committed_old_material_induced_m_s,
        np.zeros_like(ledger.committed_old_material_induced_m_s),
    )
    expected_aft = (
        proposed[0]["leading_edges_gp1_m"]
        + 0.5 * ledger.total_local_edge_velocity_m_s * solver.delta_time
    )
    np.testing.assert_array_equal(
        proposed[0]["first_aft_edges_gp1_m"],
        expected_aft,
    )
    np.testing.assert_array_equal(ledger.new_sheet, proposed[0]["new_sheet"])


def test_material_continuation_ignores_first_edge_and_keeps_eq7_one_third() -> None:
    state = PteraNativeMaterialLEVState(ns=1)
    leading_0 = np.array([[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])
    first_0 = np.array([[[0.3, 0.0, 0.2], [0.3, 1.0, 0.2]]])
    proposal_0 = state.propose_shed(
        step=0,
        leading_edges_gp1_m=leading_0,
        active=np.array([True]),
        new_sheet=np.array([True]),
        first_aft_edges_gp1_m=first_0,
    )
    proposal_0.commit_and_convect(
        np.array([0.2]),
        np.zeros((1, 4, 3)),
        dt_s=0.1,
    )

    leading_1 = leading_0 + np.array([[[0.1, 0.0, 0.0]]])
    deliberately_wrong_first = np.full((1, 2, 3), 99.0)
    proposal_1 = state.propose_shed(
        step=1,
        leading_edges_gp1_m=leading_1,
        active=np.array([True]),
        new_sheet=np.array([False]),
        first_aft_edges_gp1_m=deliberately_wrong_first,
    )
    newborn = proposal_1.newborn_ptera_rings_gp1_m[0]
    expected_split = (2.0 / 3.0) * leading_1[0] + (1.0 / 3.0) * first_0[0]
    np.testing.assert_allclose(newborn[2], expected_split[0], atol=2.0e-16)
    np.testing.assert_allclose(newborn[3], expected_split[1], atol=2.0e-16)
    assert not np.any(np.isclose(newborn, 99.0))
