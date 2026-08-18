from __future__ import annotations

import os
from typing import Any

os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/fluxv-v5h3-feedback-numba")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/fluxv-v5h3-feedback-mpl")

import numpy as np
import pytest
from pterasoftware import _functions

import fluxvortex.rvpm_reference as rvpm_reference_module
from fluxvortex.rvpm_reference import direct_gaussian_erf_velocity_jacobian
from fluxvortex.solver import UVPMHybridSolver
from forward_flight_benchmarks.fluxv_v5h3_native_feedback import (
    FEEDBACK_INTERFACE_ID,
    FEEDBACK_OWNER,
    FORCE_SCORING_STATUS,
    SURFACE_LOAD_OWNER,
    NativePteraRVPMFeedbackConfig,
    NativePteraRVPMFeedbackSolver,
    make_fluxv_v5h3_native_feedback_solver,
)
from forward_flight_benchmarks.v5h2_dyadic_cumulative_cloud_transport import (
    attest_dyadic_cumulative_ribbon_handoff,
    materialize_dyadic_cumulative_particle_state,
    transport_dyadic_accumulated_particle_cloud,
)
from forward_flight_benchmarks.v5h_dvm_node_ribbon import (
    NodeOwnedDVMRibbonShadow,
)

from test_v5h_cumulative_cloud_transport import (
    DT_S,
    FREESTREAM_GP1_M_PER_S,
    SIGMA_BIRTH_M,
    TARGET_SPACING_M,
    _map_events,
    _map_step,
    _source,
)
from test_v5h_ptera_te_shadow import (
    _generic_non_target_problem,
    _native_state_and_load_bits,
)


class _ExplodingReports:
    def __iter__(self) -> Any:
        raise AssertionError("disabled factory inspected feedback reports")


def _feedback_report(incidence_deg: float = 35.0) -> Any:
    mapper = NodeOwnedDVMRibbonShadow(wing_id="wing", source_family="lev")
    sources = (_source(0), _source(1))
    events = tuple(
        source.step(np.deg2rad(incidence_deg), 0.0, 0.0) for source in sources
    )
    assert all(event.lesp_active for event in events)
    ribbon = _map_events(
        mapper,
        events,  # type: ignore[arg-type]
        source_time_s=0.0,
        previous_report=None,
    )
    handoff = attest_dyadic_cumulative_ribbon_handoff(
        mapper,
        ribbon,
        wing_id="wing",
        source_time_s=0.0,
    )
    report = transport_dyadic_accumulated_particle_cloud(
        handoff,
        smoothing_radius_m=SIGMA_BIRTH_M,
        base_target_spacing_m=TARGET_SPACING_M,
        refinement_level=0,
        transport_end_time_s=DT_S,
        transport_substeps=1,
        freestream_velocity_gp1_m_per_s=FREESTREAM_GP1_M_PER_S,
    )
    assert report.for_source_step_index == 2
    assert report.transport_end_time_s == DT_S
    return report


def _two_feedback_reports() -> tuple[Any, Any]:
    mapper = NodeOwnedDVMRibbonShadow(wing_id="wing", source_family="lev")
    sources = (_source(0), _source(1))
    first_ribbon = _map_step(
        mapper,
        sources,
        active_by_cell=(True, True),
        source_time_s=0.0,
        previous_report=None,
    )
    first_handoff = attest_dyadic_cumulative_ribbon_handoff(
        mapper,
        first_ribbon,
        wing_id="wing",
        source_time_s=0.0,
    )
    first = transport_dyadic_accumulated_particle_cloud(
        first_handoff,
        smoothing_radius_m=SIGMA_BIRTH_M,
        base_target_spacing_m=TARGET_SPACING_M,
        refinement_level=0,
        transport_end_time_s=DT_S,
        transport_substeps=1,
        freestream_velocity_gp1_m_per_s=FREESTREAM_GP1_M_PER_S,
    )
    second_ribbon = _map_step(
        mapper,
        sources,
        active_by_cell=(True, True),
        source_time_s=DT_S,
        previous_report=first,
    )
    second_handoff = attest_dyadic_cumulative_ribbon_handoff(
        mapper,
        second_ribbon,
        wing_id="wing",
        source_time_s=DT_S,
        previous_report=first,
    )
    second = transport_dyadic_accumulated_particle_cloud(
        second_handoff,
        smoothing_radius_m=SIGMA_BIRTH_M,
        base_target_spacing_m=TARGET_SPACING_M,
        refinement_level=0,
        transport_end_time_s=2.0 * DT_S,
        transport_substeps=1,
        freestream_velocity_gp1_m_per_s=FREESTREAM_GP1_M_PER_S,
    )
    return first, second


def _solver(
    config: NativePteraRVPMFeedbackConfig,
    reports: Any = (),
) -> UVPMHybridSolver:
    return make_fluxv_v5h3_native_feedback_solver(
        _generic_non_target_problem(),
        config=config,
        feedback_reports=reports,
        max_particles=100,
        stretch=False,
        free_wake=False,
    )


def _run(
    config: NativePteraRVPMFeedbackConfig,
    reports: Any = (),
) -> UVPMHybridSolver:
    solver = _solver(config, reports)
    solver.run(
        prescribed_wake=True,
        calculate_streamlines=False,
        show_progress=False,
    )
    return solver


def test_disabled_factory_is_exact_parent_and_input_blind() -> None:
    solver = _solver(
        NativePteraRVPMFeedbackConfig(enabled=False),
        _ExplodingReports(),
    )
    assert type(solver) is UVPMHybridSolver


def test_enabled_empty_real_ptera_run_is_bitwise_parent_exact() -> None:
    parent = _run(NativePteraRVPMFeedbackConfig(enabled=False))
    candidate = _run(NativePteraRVPMFeedbackConfig(enabled=True))
    assert type(candidate) is NativePteraRVPMFeedbackSolver
    assert candidate.v5h3_feedback_step_reports == []
    assert _native_state_and_load_bits(candidate) == _native_state_and_load_bits(parent)


@pytest.mark.parametrize("incidence_deg", (-35.0, 35.0))
def test_active_cloud_enters_native_rhs_and_four_load_batches_once(
    incidence_deg: float,
) -> None:
    cloud = _feedback_report(incidence_deg)
    solver = _run(
        NativePteraRVPMFeedbackConfig(enabled=True),
        (cloud,),
    )
    assert isinstance(solver, NativePteraRVPMFeedbackSolver)
    assert not solver.v5h3_poisoned
    assert len(solver.v5h3_feedback_step_reports) == 1
    ledger = solver.v5h3_feedback_step_reports[0]
    assert ledger.interface_id == FEEDBACK_INTERFACE_ID
    assert ledger.ptera_step_index == 1
    assert ledger.dvm_for_source_step_index == 2
    assert ledger.source_time_s == DT_S
    assert ledger.particle_count == cloud.total_particle_count
    assert ledger.collocation_evaluation_count == 1
    assert ledger.load_leg_evaluation_count == 4
    assert ledger.parent_load_call_count == 1
    assert ledger.extension_force_write_count == 0
    assert ledger.extension_moment_write_count == 0
    assert ledger.extension_load_processor_call_count == 0
    assert ledger.feedback_owner == FEEDBACK_OWNER
    assert ledger.surface_load_owner == SURFACE_LOAD_OWNER
    assert ledger.prescribed_wake
    assert ledger.force_scoring_status == FORCE_SCORING_STATUS
    assert ledger.no_penetration_max_abs <= 1.0e-12
    np.testing.assert_array_equal(
        ledger.combined_wake_normal,
        ledger.parent_wake_normal + ledger.feedback_normal,
    )
    np.testing.assert_allclose(
        ledger.no_penetration_residual,
        0.0,
        atol=1.0e-12,
        rtol=0.0,
    )

    state = materialize_dyadic_cumulative_particle_state(cloud)
    for evaluation in (ledger.collocation_evaluation, *ledger.load_evaluations):
        oracle = direct_gaussian_erf_velocity_jacobian(
            state.positions,
            state.gamma,
            state.sigma,
            target_positions=evaluation.target_points_gp1_m,
        ).velocity
        np.testing.assert_allclose(
            evaluation.induced_velocity_gp1_m_per_s,
            oracle,
            atol=1.0e-13,
            rtol=0.0,
        )
    assert [item.channel for item in ledger.load_evaluations] == [
        "load_right",
        "load_front",
        "load_left",
        "load_back",
    ]
    assert any(
        np.any(item.induced_velocity_gp1_m_per_s != 0.0)
        for item in ledger.load_evaluations
    )


def test_active_feedback_changes_native_solution_but_not_prescribed_wake_geometry() -> (
    None
):
    parent = _run(NativePteraRVPMFeedbackConfig(enabled=False))
    active = _run(
        NativePteraRVPMFeedbackConfig(enabled=True),
        (_feedback_report(),),
    )
    assert isinstance(active, NativePteraRVPMFeedbackSolver)
    assert not np.array_equal(
        active.v5h3_feedback_step_reports[0].bound_strengths_m2_s,
        np.asarray(
            [
                panel.ring_vortex.strength
                for panel in np.ravel(
                    parent.steady_problems[1].airplanes[0].wings[0].panels
                )
            ]
        ),
    )
    for name in (
        "listStackBrwrvp_GP1_CgP1",
        "listStackFrwrvp_GP1_CgP1",
        "listStackFlwrvp_GP1_CgP1",
        "listStackBlwrvp_GP1_CgP1",
    ):
        for parent_row, active_row in zip(
            getattr(parent, name), getattr(active, name), strict=True
        ):
            np.testing.assert_array_equal(active_row, parent_row)


def test_wrong_owner_contract_rejects_before_run() -> None:
    report = _feedback_report()
    with pytest.raises(ValueError, match="wing_id"):
        _solver(
            NativePteraRVPMFeedbackConfig(
                enabled=True,
                expected_wing_id="different-wing",
            ),
            (report,),
        )


def test_two_step_live_chain_aligns_one_based_dvm_with_zero_based_ptera() -> None:
    reports = _two_feedback_reports()
    solver = _run(NativePteraRVPMFeedbackConfig(enabled=True), reports)
    assert isinstance(solver, NativePteraRVPMFeedbackSolver)
    assert [item.ptera_step_index for item in solver.v5h3_feedback_step_reports] == [
        1,
        2,
    ]
    assert [
        item.dvm_for_source_step_index for item in solver.v5h3_feedback_step_reports
    ] == [2, 3]
    assert [item.source_time_s for item in solver.v5h3_feedback_step_reports] == [
        DT_S,
        2.0 * DT_S,
    ]
    assert all(
        item.no_penetration_max_abs <= 1.0e-12
        for item in solver.v5h3_feedback_step_reports
    )


def test_cross_run_feedback_parent_chain_is_rejected() -> None:
    # A content-identical fresh parent is intentionally observationally
    # equivalent.  This splice changes the signed physical history, so the
    # child's parent digest must disagree with the supplied first report.
    first_a = _feedback_report(-35.0)
    _, second_b = _two_feedback_reports()
    with pytest.raises(ValueError, match="parent chain"):
        _solver(
            NativePteraRVPMFeedbackConfig(enabled=True),
            (first_a, second_b),
        )


def test_live_report_replay_fails_and_poisoned_solver_cannot_resume() -> None:
    report = _feedback_report()
    first = _run(NativePteraRVPMFeedbackConfig(enabled=True), (report,))
    assert isinstance(first, NativePteraRVPMFeedbackSolver)

    replay = _solver(NativePteraRVPMFeedbackConfig(enabled=True), (report,))
    assert isinstance(replay, NativePteraRVPMFeedbackSolver)
    with pytest.raises(ValueError, match="already consumed"):
        replay.run(
            prescribed_wake=True,
            calculate_streamlines=False,
            show_progress=False,
        )
    assert replay.v5h3_poisoned
    with pytest.raises(RuntimeError, match="poisoned"):
        replay._calculate_wake_wing_influences()


def test_active_path_uses_exactly_one_native_load_processor_call_per_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = _feedback_report()
    calls: list[tuple[Any, np.ndarray, np.ndarray]] = []
    original = _functions.process_solver_loads

    def recording(solver: Any, forces: np.ndarray, moments: np.ndarray) -> Any:
        calls.append((solver, np.asarray(forces).copy(), np.asarray(moments).copy()))
        return original(solver, forces, moments)

    monkeypatch.setattr(_functions, "process_solver_loads", recording)
    solver = _run(NativePteraRVPMFeedbackConfig(enabled=True), (report,))
    assert len(calls) == solver.num_steps == 3
    for step, (owner, forces, moments) in enumerate(calls):
        assert owner is solver
        panels = np.ravel(solver.steady_problems[step].airplanes[0].wings[0].panels)
        np.testing.assert_array_equal(
            forces,
            np.asarray([panel.forces_GP1 for panel in panels]),
        )
        np.testing.assert_array_equal(
            moments,
            np.asarray([panel.moments_GP1_CgP1 for panel in panels]),
        )
    assert solver.v5h3_feedback_step_reports[0].extension_load_processor_call_count == 0


def test_runtime_particle_field_dependency_drift_fails_before_call_and_report_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = _feedback_report()
    calls = 0
    original = rvpm_reference_module.direct_gaussian_erf_velocity_jacobian

    def forwarding(*args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    solver = _solver(NativePteraRVPMFeedbackConfig(enabled=True), (report,))
    assert isinstance(solver, NativePteraRVPMFeedbackSolver)
    with monkeypatch.context() as scoped:
        scoped.setattr(
            rvpm_reference_module,
            "direct_gaussian_erf_velocity_jacobian",
            forwarding,
        )
        with pytest.raises(ValueError, match="callable was replaced"):
            solver.run(
                prescribed_wake=True,
                calculate_streamlines=False,
                show_progress=False,
            )
    assert calls == 0
    assert solver.v5h3_poisoned
    clean = _run(NativePteraRVPMFeedbackConfig(enabled=True), (report,))
    assert isinstance(clean, NativePteraRVPMFeedbackSolver)
    assert len(clean.v5h3_feedback_step_reports) == 1
