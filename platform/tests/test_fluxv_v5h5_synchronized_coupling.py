from __future__ import annotations

from dataclasses import replace
import os
from typing import Any

os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/fluxv-v5h5-sync-numba")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/fluxv-v5h5-sync-mpl")

import numpy as np
import pterasoftware as ps
import pytest

from fluxvortex.solver import UVPMHybridSolver
import forward_flight_benchmarks.fluxv_v5h5_synchronized_coupling as sync_module
from forward_flight_benchmarks.fluxv_v5h5_synchronized_coupling import (
    OBSERVATION_ACCESS,
    SYNCHRONIZED_INTERFACE_ID,
    SYNCHRONIZED_OWNER,
    TARGET_CASE_BRANCH,
    SynchronizedCloudTransportReport,
    SynchronizedPteraRVPMCouplingSolver,
    SynchronizedReleaseLayer,
    make_fluxv_v5h5_synchronized_solver,
    materialize_synchronized_particle_state,
    validate_synchronized_cloud_transport_report,
)
from forward_flight_benchmarks.v5h_dvm_node_placement import (
    DVMNodePlacementCell,
    GP1NodeSectionFact,
    NodeLocalDVMPlacementAdapter,
)
from forward_flight_benchmarks.v5h_dvm_node_ribbon import DVMSpanCellSource

from test_v5h_cumulative_cloud_transport import DT_S, _mapping, _source
from test_v5h_ptera_te_shadow import (
    _generic_non_target_problem,
    _native_state_and_load_bits,
)


class _ExplodingLayers:
    def __iter__(self) -> Any:
        raise AssertionError("disabled factory inspected release layers")

    def __len__(self) -> int:
        raise AssertionError("disabled factory inspected release-layer length")


def _release_layers(num_steps: int = 3) -> tuple[SynchronizedReleaseLayer, ...]:
    node_sources = tuple(_source(index) for index in (10, 11, 12))
    cell_sources = tuple(_source(index) for index in (0, 1))
    placement_adapter = NodeLocalDVMPlacementAdapter(wing_id="wing")
    layers: list[SynchronizedReleaseLayer] = []
    for step in range(1, num_steps + 1):
        source_time = (step - 1) * DT_S
        node_events = tuple(
            source.step(np.deg2rad(35.0), 0.0, 0.0) for source in node_sources
        )
        cell_events = tuple(
            source.step(np.deg2rad(35.0), 0.0, 0.0) for source in cell_sources
        )
        node_facts = tuple(
            _node_fact(index, event, source_time)  # type: ignore[arg-type]
            for index, event in enumerate(node_events)
        )
        placement_cells = tuple(
            DVMNodePlacementCell(
                cell_id=f"cell-{index}",
                left_node_fact=node_facts[index],
                right_node_fact=node_facts[index + 1],
                cell_source_event=event,
            )
            for index, event in enumerate(cell_events)
        )
        placement = placement_adapter.map_step(
            placement_cells,
            delta_time_s=DT_S,
        )
        cells = tuple(
            DVMSpanCellSource(
                cell_id=f"cell-{index}",
                left_node_id=f"node-{index}",
                right_node_id=f"node-{index + 1}",
                event=event,
                plane_to_gp1=_mapping(index),
            )
            for index, event in enumerate(cell_events)
        )
        layers.append(
            SynchronizedReleaseLayer(
                source_step_index=step,
                source_time_s=source_time,
                cells=cells,
                node_placement_result=placement,
            )
        )
    return tuple(layers)


def _one_step_problem() -> ps.problems.UnsteadyProblem:
    parent = _generic_non_target_problem()
    base = parent.movement
    movement = ps.movements.movement.Movement(
        airplane_movements=list(base.airplane_movements),
        operating_point_movement=base.operating_point_movement,
        delta_time=base.delta_time,
        num_steps=1,
        max_wake_rows=base.max_wake_rows,
    )
    return ps.problems.UnsteadyProblem(movement=movement, only_final_results=False)


def _node_fact(index: int, event: Any, source_time_s: float) -> GP1NodeSectionFact:
    x_axis = np.asarray((1.0, 0.0, 0.0))
    z_axis = np.asarray((0.0, 0.0, -1.0))
    lev_2d = np.asarray(
        event.lev_placement.edge_anchor_position_over_chord_backend_world
    )
    tev_2d = np.asarray(
        event.tev_placement.edge_anchor_position_over_chord_backend_world
    )
    lev_gp1 = np.asarray((0.0, float(index), 0.0))
    chord = float(event.provenance.position_scale_chord_m)
    tev_gp1 = lev_gp1 + chord * (
        (tev_2d[0] - lev_2d[0]) * x_axis + (tev_2d[1] - lev_2d[1]) * z_axis
    )
    return GP1NodeSectionFact(
        wing_id="wing",
        node_id=f"node-{index}",
        source_step_index=int(event.lineage.source_step_index),
        source_time_s=source_time_s,
        event=event,
        lev_edge_anchor_gp1_m=tuple(lev_gp1),
        tev_edge_anchor_gp1_m=tuple(tev_gp1),
        reference_chord_m=chord,
        reference_speed_m_per_s=float(
            event.provenance.circulation_scale_u_times_c_m2_per_s / chord
        ),
        dvm_x_axis_gp1=tuple(x_axis),
        dvm_z_axis_gp1=tuple(z_axis),
        positive_span_axis_gp1=(0.0, 1.0, 0.0),
        topology_patch_id="v5h5-main-patch",
        coordinate_frame_id="ptera-gp1-v5h5-wing",
        node_lineage_id=event.lineage.section_lineage_id,
        geometry_token=event.provenance.geometry_hash_sha256,
    )


def _run_parent() -> UVPMHybridSolver:
    solver = UVPMHybridSolver(
        _generic_non_target_problem(),
        max_particles=100,
        stretch=False,
        free_wake=False,
    )
    solver.run(
        prescribed_wake=True,
        calculate_streamlines=False,
        show_progress=False,
    )
    return solver


def _run_candidate() -> SynchronizedPteraRVPMCouplingSolver:
    solver = make_fluxv_v5h5_synchronized_solver(
        _generic_non_target_problem(),
        enabled=True,
        release_layers=_release_layers(),
        wing_id="wing",
        smoothing_radius_m=0.085,
        base_target_spacing_m=0.04,
        refinement_level=0,
        max_particles=100,
        stretch=False,
        free_wake=False,
    )
    assert isinstance(solver, SynchronizedPteraRVPMCouplingSolver)
    solver.run(
        prescribed_wake=True,
        calculate_streamlines=False,
        show_progress=False,
    )
    return solver


def _birth_by_node(report: SynchronizedCloudTransportReport) -> dict[Any, Any]:
    return {birth.node_id: birth for birth in report.prepared_cloud.node_births}


def _fact_by_node(report: SynchronizedCloudTransportReport) -> dict[Any, Any]:
    return {fact.node_id: fact for fact in report.facts}


def test_disabled_factory_is_exact_parent_and_input_blind() -> None:
    solver = make_fluxv_v5h5_synchronized_solver(
        _generic_non_target_problem(),
        enabled=False,
        release_layers=_ExplodingLayers(),
        max_particles=100,
        stretch=False,
        free_wake=False,
    )
    assert type(solver) is UVPMHybridSolver
    solver.run(
        prescribed_wake=True,
        calculate_streamlines=False,
        show_progress=False,
    )
    parent = _run_parent()
    assert _native_state_and_load_bits(solver) == _native_state_and_load_bits(parent)


def test_real_three_step_release_feedback_load_transport_order_closes() -> None:
    solver = _run_candidate()
    reports = solver.v5h5_transport_reports
    feedback = solver.v5h3_feedback_step_reports
    assert len(reports) == len(feedback) == 3
    assert [item.parent_source_step_index for item in reports] == [1, 2, 3]
    assert [item.for_source_step_index for item in reports] == [2, 3, 4]
    assert [item.ptera_step_index for item in feedback] == [0, 1, 2]
    assert [item.dvm_for_source_step_index for item in feedback] == [1, 2, 3]

    previous: SynchronizedCloudTransportReport | None = None
    for index, (report, ledger) in enumerate(zip(reports, feedback, strict=True)):
        assert validate_synchronized_cloud_transport_report(report) is report
        assert report.interface_id == SYNCHRONIZED_INTERFACE_ID
        assert report.parent_report_sha256 == (
            None if previous is None else previous.report_sha256
        )
        assert report.prepared_report_sha256 == report.prepared_cloud.report_sha256
        assert report.previous_particle_count == (
            0 if previous is None else previous.total_particle_count
        )
        assert report.new_particle_count > 0
        assert report.total_particle_count == (
            report.previous_particle_count + report.new_particle_count
        )
        assert len(report.release_slices) == index + 1
        assert report.release_slices[-1].start_index == report.previous_particle_count
        assert report.release_slices[-1].stop_index == report.total_particle_count
        assert report.release_slices[-1].particle_count == report.new_particle_count
        assert report.release_call_count == 1
        assert report.ptera_feedback_enabled
        assert report.combined_transport_enabled
        assert report.ptera_feedback_call_count == 1
        assert report.combined_transport_call_count == 1
        assert report.ptera_center_call_count == 3
        assert report.ptera_finite_difference_call_count == 18
        assert report.frontier_transport_parent_call_count == 3
        assert report.frontier_replay_parent_call_count == 3
        assert report.frontier_stage_replay_max_abs == 0.0
        assert report.exact_append_prefix_max_abs == 0.0
        assert report.parent_state_unchanged
        assert report.feedback_write_count == 0
        assert report.parent_write_count == 0
        assert report.load_write_count == 0
        assert report.observation_access == OBSERVATION_ACCESS
        assert report.target_case_branch == TARGET_CASE_BRANCH
        assert report.facts
        assert all(fact.producer_id == SYNCHRONIZED_OWNER for fact in report.facts)
        assert len(report.transport_result.stages) == 3

        assert ledger is report.ptera_feedback_report
        assert ledger.particle_count == report.total_particle_count
        assert ledger.collocation_evaluation_count == 1
        assert ledger.load_leg_evaluation_count == 4
        assert ledger.parent_load_call_count == 1
        assert ledger.extension_force_write_count == 0
        assert ledger.extension_moment_write_count == 0
        assert ledger.extension_load_processor_call_count == 0
        assert ledger.no_penetration_max_abs <= 1.0e-12
        np.testing.assert_allclose(
            ledger.no_penetration_residual,
            0.0,
            atol=1.0e-12,
            rtol=0.0,
        )

        if previous is not None:
            old_count = previous.total_particle_count
            np.testing.assert_array_equal(
                report.prepared_cloud.positions_gp1_m[:old_count],
                previous.transported_state.positions,
            )
            np.testing.assert_array_equal(
                report.prepared_cloud.gamma_vector_m3_per_s[:old_count],
                previous.transported_state.gamma,
            )
            np.testing.assert_array_equal(
                report.prepared_cloud.sigma_m[:old_count],
                previous.transported_state.sigma,
            )
            assert report.particle_ids[:old_count] == previous.particle_ids
            assert report.lineage[:old_count] == previous.lineage
        previous = report


def test_first_and_continuous_node_births_use_the_declared_time_layers() -> None:
    reports = _run_candidate().v5h5_transport_reports
    first_births = _birth_by_node(reports[0])
    for node in reports[0].prepared_cloud.node_kinematics:
        birth = first_births[node.node_id]
        assert birth.mode == "first"
        expected = np.asarray(node.anchor_position_gp1_m) + 0.5 * DT_S * np.asarray(
            node.edge_velocity_gp1_m_per_s
        )
        np.testing.assert_array_equal(birth.birth_position_gp1_m, expected)

    for previous, current in zip(reports[:-1], reports[1:], strict=True):
        facts = _fact_by_node(previous)
        births = _birth_by_node(current)
        for node in current.prepared_cloud.node_kinematics:
            birth = births[node.node_id]
            assert birth.mode == "continuous"
            anchor = np.asarray(node.anchor_position_gp1_m)
            expected = (
                anchor
                + (np.asarray(facts[node.node_id].advected_position_gp1_m) - anchor)
                / 3.0
            )
            np.testing.assert_array_equal(birth.birth_position_gp1_m, expected)


def test_materialization_is_copy_isolated_and_report_copy_is_rejected() -> None:
    report = _run_candidate().v5h5_transport_reports[-1]
    state = materialize_synchronized_particle_state(report)
    original = report.transported_state.positions.copy()
    state.positions[0, 0] += 1.0
    np.testing.assert_array_equal(report.transported_state.positions, original)
    with pytest.raises(ValueError, match="live producer"):
        validate_synchronized_cloud_transport_report(replace(report))


def test_two_fresh_three_layer_runs_are_deterministic() -> None:
    first = _run_candidate().v5h5_transport_reports
    second = _run_candidate().v5h5_transport_reports
    assert [report.report_sha256 for report in first] == [
        report.report_sha256 for report in second
    ]
    for left, right in zip(first, second, strict=True):
        np.testing.assert_array_equal(
            left.transported_state.positions, right.transported_state.positions
        )
        np.testing.assert_array_equal(
            left.transported_state.gamma, right.transported_state.gamma
        )
        np.testing.assert_array_equal(
            left.transported_state.sigma, right.transported_state.sigma
        )
        assert left.particle_ids == right.particle_ids
        assert left.lineage == right.lineage


def test_same_live_dvm_release_cannot_feed_a_second_solver() -> None:
    layers = _release_layers()
    first = make_fluxv_v5h5_synchronized_solver(
        _generic_non_target_problem(),
        enabled=True,
        release_layers=layers,
        max_particles=100,
        stretch=False,
        free_wake=False,
    )
    first.run(
        prescribed_wake=True,
        calculate_streamlines=False,
        show_progress=False,
    )
    second = make_fluxv_v5h5_synchronized_solver(
        _generic_non_target_problem(),
        enabled=True,
        release_layers=layers,
        max_particles=100,
        stretch=False,
        free_wake=False,
    )
    with pytest.raises(ValueError, match="already consumed"):
        second.run(
            prescribed_wake=True,
            calculate_streamlines=False,
            show_progress=False,
        )
    assert second.v5h3_poisoned


def test_dependency_replacement_fails_before_consumption_and_clean_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layers = _release_layers()
    trusted = sync_module._FROZEN_DEPOSIT

    def forwarded(*args: Any, **kwargs: Any) -> Any:
        return trusted(*args, **kwargs)

    monkeypatch.setattr(sync_module, "_FROZEN_DEPOSIT", forwarded)
    with pytest.raises(ValueError, match="dependency callable"):
        make_fluxv_v5h5_synchronized_solver(
            _generic_non_target_problem(),
            enabled=True,
            release_layers=layers,
            max_particles=100,
            stretch=False,
            free_wake=False,
        )
    monkeypatch.setattr(sync_module, "_FROZEN_DEPOSIT", trusted)
    clean = make_fluxv_v5h5_synchronized_solver(
        _generic_non_target_problem(),
        enabled=True,
        release_layers=layers,
        max_particles=100,
        stretch=False,
        free_wake=False,
    )
    clean.run(
        prescribed_wake=True,
        calculate_streamlines=False,
        show_progress=False,
    )
    assert len(clean.v5h5_transport_reports) == 3


def test_live_nested_frontier_tamper_is_rejected() -> None:
    report = _run_candidate().v5h5_transport_reports[-1]
    object.__setattr__(report.facts[0], "producer_report_sha256", "0" * 64)
    with pytest.raises(ValueError, match="frontier fact"):
        validate_synchronized_cloud_transport_report(report)


def test_feedback_only_reduction_preserves_full_mode_ptera_state_and_loads() -> None:
    full = make_fluxv_v5h5_synchronized_solver(
        _one_step_problem(),
        enabled=True,
        release_layers=_release_layers(1),
        feedback_enabled=True,
        transport_enabled=True,
        max_particles=100,
        stretch=False,
        free_wake=False,
    )
    feedback_only = make_fluxv_v5h5_synchronized_solver(
        _one_step_problem(),
        enabled=True,
        release_layers=_release_layers(1),
        feedback_enabled=True,
        transport_enabled=False,
        max_particles=100,
        stretch=False,
        free_wake=False,
    )
    for solver in (full, feedback_only):
        solver.run(
            prescribed_wake=True,
            calculate_streamlines=False,
            show_progress=False,
        )
    assert _native_state_and_load_bits(full) == _native_state_and_load_bits(
        feedback_only
    )
    assert len(full.v5h5_transport_reports) == 1
    assert feedback_only.v5h5_transport_reports == []
    assert len(full.v5h3_feedback_step_reports) == 1
    assert len(feedback_only.v5h3_feedback_step_reports) == 1


def test_transport_only_reduction_leaves_native_ptera_bitwise_exact() -> None:
    parent = _run_parent()
    candidate = make_fluxv_v5h5_synchronized_solver(
        _generic_non_target_problem(),
        enabled=True,
        release_layers=_release_layers(),
        feedback_enabled=False,
        transport_enabled=True,
        max_particles=100,
        stretch=False,
        free_wake=False,
    )
    candidate.run(
        prescribed_wake=True,
        calculate_streamlines=False,
        show_progress=False,
    )
    assert _native_state_and_load_bits(candidate) == _native_state_and_load_bits(parent)
    assert candidate.v5h3_feedback_step_reports == []
    assert len(candidate.v5h5_transport_reports) == 3
    for report in candidate.v5h5_transport_reports:
        assert not report.ptera_feedback_enabled
        assert report.ptera_feedback_call_count == 0
        assert report.ptera_feedback_report is None
        assert validate_synchronized_cloud_transport_report(report) is report
