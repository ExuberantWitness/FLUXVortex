from __future__ import annotations

from dataclasses import fields, replace
from hashlib import sha256
from inspect import signature
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pytest

import fluxvortex.rvpm_edge_bridge as edge_bridge_module
import fluxvortex.rvpm_transport as transport_module
from fluxvortex.rvpm_edge_bridge import (
    FROZEN_OVERLAP_LAMBDA,
    deposit_edge_graph_prescribed_sigma_and_spacing,
)
from fluxvortex.rvpm_reference import direct_gaussian_erf_velocity_jacobian
from fluxvortex.rvpm_transport import (
    ParticleState,
    lsrk3_step_direct,
    make_particle_state,
)

from forward_flight_benchmarks.ldvm_uvlm_correction import (
    LDVMSectionSettings,
    LESPThreshold,
)
import forward_flight_benchmarks.v5h_cumulative_cloud_transport as cumulative_module
from forward_flight_benchmarks.v5h_cumulative_cloud_transport import (
    MAX_CUMULATIVE_PARTICLES,
    MAX_PARTICLE_COUNT,
    MAX_PARTICLES_PER_RELEASE,
    MAX_TRANSPORT_SUBSTEPS,
    attest_cumulative_ribbon_handoff,
    materialize_cumulative_particle_state,
    transport_accumulated_particle_cloud,
    validate_cumulative_cloud_transport_report,
)
from forward_flight_benchmarks.v5h_dvm_node_ribbon import (
    DVMPlaneToGP1Map,
    DVMRibbonShadowResult,
    DVMSpanCellSource,
    NodeOwnedDVMRibbonShadow,
    SpanNodeKinematics,
)
from forward_flight_benchmarks.v5h_dvm_source import DVMSourceEvent, V5hDVMSource
from forward_flight_benchmarks.v5h_passive_frontier_transport import (
    materialize_transported_particle_state,
    transport_passive_node_frontiers,
)


DT_S = 0.02
REFERENCE_SPEED_M_PER_S = 2.0
REFERENCE_CHORD_M = 0.25
DELTA_TIME_CONVECTIVE = DT_S * REFERENCE_SPEED_M_PER_S / REFERENCE_CHORD_M
SIGMA_BIRTH_M = 0.085
TARGET_SPACING_M = 0.04
FREESTREAM_GP1_M_PER_S = (0.1, 0.0, 0.0)


def _source(strip: int) -> V5hDVMSource:
    return V5hDVMSource(
        physical_section_id=f"cumulative:section:{strip}",
        physical_strip_id=f"cumulative:strip:{strip}",
        geometry_identity="explicit zero-camber flat-plate surrogate",
        reference_speed_m_per_s=REFERENCE_SPEED_M_PER_S,
        reference_chord_m=REFERENCE_CHORD_M,
        zero_camber_surrogate=True,
        delta_time_convective=DELTA_TIME_CONVECTIVE,
        pivot_fraction_chord=0.25,
        threshold=LESPThreshold(
            value=0.18,
            section_family="generic thin flat plate",
            reynolds=30_000.0,
            source="published source input used only by a manufactured gate",
            source_role="published_source_input",
        ),
        settings=LDVMSectionSettings(ndiv=24, naterm=8, max_wake_steps=48),
    )


def _event(source: V5hDVMSource, *, active: bool) -> DVMSourceEvent:
    event = source.step(np.deg2rad(35.0 if active else 5.0), 0.0, 0.0)
    assert event.lesp_active is active
    return event


def _events(
    sources: tuple[V5hDVMSource, V5hDVMSource],
    active_by_cell: tuple[bool, bool],
) -> tuple[DVMSourceEvent, DVMSourceEvent]:
    return tuple(
        _event(source, active=active)
        for source, active in zip(sources, active_by_cell, strict=True)
    )  # type: ignore[return-value]


def _mapping(cell: int) -> DVMPlaneToGP1Map:
    return DVMPlaneToGP1Map(
        origin_gp1_m=np.array([10.0 + cell, 0.0, 0.0]),
        x_axis_gp1=np.array([1.0, 0.0, 0.0]),
        z_axis_gp1=np.array([0.0, 0.0, -1.0]),
        positive_circulation_axis_gp1=np.array([0.0, 1.0, 0.0]),
        circulation_to_ring_traversal_sign=1,
        provenance="manufactured explicit DVM x-z to Ptera GP1 map",
    )


def _nodes() -> tuple[SpanNodeKinematics, ...]:
    return tuple(
        SpanNodeKinematics(
            node_id=index,
            anchor_position_gp1_m=np.array([0.0, float(index), 0.0]),
            edge_velocity_gp1_m_per_s=np.array([float(index + 1), 0.0, 0.0]),
        )
        for index in range(3)
    )


def _cells(
    events: tuple[DVMSourceEvent, DVMSourceEvent],
) -> tuple[DVMSpanCellSource, ...]:
    return (
        DVMSpanCellSource("cell-0", 0, 1, events[0], _mapping(0)),
        DVMSpanCellSource("cell-1", 1, 2, events[1], _mapping(1)),
    )


def _map_events(
    mapper: NodeOwnedDVMRibbonShadow,
    events: tuple[DVMSourceEvent, DVMSourceEvent],
    *,
    source_time_s: float,
    previous_report: object | None,
) -> DVMRibbonShadowResult:
    return mapper.map_step(
        _cells(events),
        _nodes(),
        delta_time_s=DT_S,
        transport_enabled=True,
        source_time_s=source_time_s,
        frontier_transport_report=previous_report,
    )


def _map_step(
    mapper: NodeOwnedDVMRibbonShadow,
    sources: tuple[V5hDVMSource, V5hDVMSource],
    *,
    active_by_cell: tuple[bool, bool],
    source_time_s: float,
    previous_report: object | None,
) -> DVMRibbonShadowResult:
    return _map_events(
        mapper,
        _events(sources, active_by_cell),
        source_time_s=source_time_s,
        previous_report=previous_report,
    )


def _transport(
    mapper: NodeOwnedDVMRibbonShadow,
    ribbon: DVMRibbonShadowResult,
    *,
    source_time_s: float,
    previous_report: object | None,
    transport_end_time_s: float | None = None,
    substeps: int = 1,
) -> Any:
    handoff = attest_cumulative_ribbon_handoff(
        mapper,
        ribbon,
        wing_id="wing",
        source_time_s=source_time_s,
        previous_report=previous_report,
    )
    report = transport_accumulated_particle_cloud(
        handoff,
        smoothing_radius_m=SIGMA_BIRTH_M,
        deposition_target_spacing_m=TARGET_SPACING_M,
        transport_end_time_s=(
            source_time_s + DT_S
            if transport_end_time_s is None
            else transport_end_time_s
        ),
        transport_substeps=substeps,
        freestream_velocity_gp1_m_per_s=FREESTREAM_GP1_M_PER_S,
    )
    assert validate_cumulative_cloud_transport_report(report) is report
    return report


def _deposited(ribbon: DVMRibbonShadowResult) -> Any:
    assert ribbon.edge_graph is not None
    step = ribbon.diagnostics.source_step_index
    assert step is not None
    return deposit_edge_graph_prescribed_sigma_and_spacing(
        ribbon.edge_graph,
        smoothing_radius=SIGMA_BIRTH_M,
        target_spacing=TARGET_SPACING_M,
        step=step,
    )


def _cloud(report: Any) -> Any:
    cloud = report.transported_particle_cloud
    assert len(cloud.particle_ids) == len(cloud.lineage)
    return cloud


def _fact_positions(report: Any) -> np.ndarray:
    ordered = sorted(report.facts, key=lambda fact: int(fact.node_id))
    return np.asarray(
        [fact.advected_position_gp1_m for fact in ordered], dtype=np.float64
    )


def _active_birth_positions(ribbon: DVMRibbonShadowResult) -> np.ndarray:
    births = sorted(
        (birth for birth in ribbon.node_births if birth.active),
        key=lambda birth: int(birth.node_id),
    )
    assert all(birth.birth_position_gp1_m is not None for birth in births)
    return np.asarray(
        [birth.birth_position_gp1_m for birth in births], dtype=np.float64
    )


def _advance_particles_and_frontiers(
    initial: ParticleState,
    frontier_positions: np.ndarray,
    *,
    start_time_s: float,
    end_time_s: float,
    substeps: int,
) -> tuple[ParticleState, np.ndarray]:
    particle_state = make_particle_state(
        initial.positions,
        initial.gamma,
        initial.sigma,
    )
    tracers = np.ascontiguousarray(frontier_positions, dtype=np.float64)
    delta_time = (end_time_s - start_time_s) / substeps
    freestream = np.asarray(FREESTREAM_GP1_M_PER_S, dtype=np.float64)
    for _ in range(substeps):
        next_particle_state, stages = lsrk3_step_direct(
            particle_state,
            delta_time,
            freestream_velocity=freestream,
        )
        tracer_storage = np.zeros_like(tracers)
        for stage in stages:
            field = direct_gaussian_erf_velocity_jacobian(
                stage.pre.positions,
                stage.pre.gamma,
                stage.pre.sigma,
                target_positions=tracers,
            )
            tracer_storage = stage.a * tracer_storage + delta_time * (
                field.velocity + freestream[None, :]
            )
            tracers = tracers + stage.b * tracer_storage
        particle_state = next_particle_state
    return particle_state, tracers


def _concatenate_states(*states: ParticleState) -> ParticleState:
    return make_particle_state(
        np.concatenate([state.positions for state in states], axis=0),
        np.concatenate([state.gamma for state in states], axis=0),
        np.concatenate([state.sigma for state in states], axis=0),
    )


def _snapshot(report: Any) -> tuple[bytes, bytes, bytes, object, object]:
    state = materialize_cumulative_particle_state(report)
    cloud = _cloud(report)
    return (
        state.positions.tobytes(),
        state.gamma.tobytes(),
        state.sigma.tobytes(),
        cloud.particle_ids,
        cloud.lineage,
    )


def _first_active_handoff() -> (
    tuple[
        NodeOwnedDVMRibbonShadow,
        tuple[V5hDVMSource, V5hDVMSource],
        DVMRibbonShadowResult,
        Any,
    ]
):
    mapper = NodeOwnedDVMRibbonShadow(wing_id="wing", source_family="lev")
    sources = (_source(0), _source(1))
    ribbon = _map_step(
        mapper,
        sources,
        active_by_cell=(True, True),
        source_time_s=0.0,
        previous_report=None,
    )
    handoff = attest_cumulative_ribbon_handoff(
        mapper,
        ribbon,
        wing_id="wing",
        source_time_s=0.0,
    )
    return mapper, sources, ribbon, handoff


def _transport_handoff(
    handoff: object,
    *,
    spacing_m: object = TARGET_SPACING_M,
    substeps: object = 1,
    end_time_s: object = DT_S,
) -> Any:
    return transport_accumulated_particle_cloud(
        handoff,
        smoothing_radius_m=SIGMA_BIRTH_M,
        deposition_target_spacing_m=spacing_m,
        transport_end_time_s=end_time_s,
        transport_substeps=substeps,
        freestream_velocity_gp1_m_per_s=FREESTREAM_GP1_M_PER_S,
    )


def _transport_trace_sha256(events: tuple[str, ...]) -> str:
    digest = sha256(b"fluxv-v5h-cumulative-transport-trace-v2\0")
    for event in events:
        digest.update(event.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _assert_release_slices(report: Any, expected_steps: Iterable[int]) -> None:
    cloud = _cloud(report)
    slices = cloud.release_slices
    expected = tuple(expected_steps)
    assert tuple(item.source_step_index for item in slices) == expected
    assert tuple(item.release_index for item in slices) == tuple(
        range(1, len(expected) + 1)
    )
    cursor = 0
    for release, step in zip(slices, expected, strict=True):
        assert release.start_index == cursor
        assert release.stop_index - release.start_index == release.particle_count
        assert release.source_step_index == step
        assert release.smoothing_radius_m == SIGMA_BIRTH_M
        assert release.deposition_target_spacing_m == TARGET_SPACING_M
        release_lineage = cloud.lineage[release.start_index : release.stop_index]
        release_ids = cloud.particle_ids[release.start_index : release.stop_index]
        assert len(release_lineage) == release.particle_count
        assert len(release_ids) == release.particle_count
        assert all(item.step == step for item in release_lineage)
        assert tuple(item.particle_id for item in release_lineage) == release_ids
        cursor = release.stop_index
    assert cursor == len(cloud.particle_ids)
    assert len(set(cloud.particle_ids)) == len(cloud.particle_ids)


class _ExplodingInput:
    def __getattribute__(self, name: str) -> Any:
        if name.startswith("__"):
            return object.__getattribute__(self, name)
        raise AssertionError(f"disabled transport inspected attribute {name!r}")

    def __iter__(self) -> Any:
        raise AssertionError("disabled transport iterated an input")

    def __float__(self) -> float:
        raise AssertionError("disabled transport inspected a numeric input")


def test_public_api_is_output_blind_and_disabled_path_is_input_blind() -> None:
    attest_parameters = signature(attest_cumulative_ribbon_handoff).parameters
    assert tuple(attest_parameters) == (
        "mapper",
        "current_ribbon",
        "wing_id",
        "source_time_s",
        "previous_report",
    )
    transport_parameters = signature(transport_accumulated_particle_cloud).parameters
    assert tuple(transport_parameters) == (
        "handoff",
        "smoothing_radius_m",
        "deposition_target_spacing_m",
        "transport_end_time_s",
        "transport_substeps",
        "freestream_velocity_gp1_m_per_s",
        "enabled",
    )
    assert not {
        "positions",
        "gamma",
        "particle_ids",
        "lineage",
        "node_ids",
        "advected_position_gp1_m",
    }.intersection(transport_parameters)

    exploding = _ExplodingInput()
    disabled = transport_accumulated_particle_cloud(
        exploding,
        smoothing_radius_m=exploding,
        deposition_target_spacing_m=exploding,
        transport_end_time_s=exploding,
        transport_substeps=exploding,
        freestream_velocity_gp1_m_per_s=exploding,
        enabled=False,
    )
    assert validate_cumulative_cloud_transport_report(disabled) is disabled
    assert disabled.enabled is False
    assert disabled.facts == ()
    assert disabled.feedback_call_count == 0
    assert disabled.parent_write_count == 0
    assert disabled.load_write_count == 0
    state = materialize_cumulative_particle_state(disabled)
    assert state.positions.shape == (0, 3)
    assert state.gamma.shape == (0, 3)
    assert state.sigma.shape == (0,)


def test_single_release_v2_is_bitwise_physical_reduction_of_v1() -> None:
    v1_mapper = NodeOwnedDVMRibbonShadow(wing_id="wing", source_family="lev")
    v1_sources = (_source(0), _source(1))
    v1_ribbon = _map_step(
        v1_mapper,
        v1_sources,
        active_by_cell=(True, True),
        source_time_s=0.0,
        previous_report=None,
    )
    v1_deposited = _deposited(v1_ribbon)
    v1 = transport_passive_node_frontiers(
        v1_ribbon,
        v1_deposited,
        wing_id="wing",
        transport_start_time_s=0.0,
        transport_end_time_s=DT_S,
        transport_substeps=1,
        deposition_target_spacing_m=TARGET_SPACING_M,
        freestream_velocity_gp1_m_per_s=FREESTREAM_GP1_M_PER_S,
    )

    v2_mapper = NodeOwnedDVMRibbonShadow(wing_id="wing", source_family="lev")
    v2_sources = (_source(0), _source(1))
    v2_ribbon = _map_step(
        v2_mapper,
        v2_sources,
        active_by_cell=(True, True),
        source_time_s=0.0,
        previous_report=None,
    )
    v2 = _transport(
        v2_mapper,
        v2_ribbon,
        source_time_s=0.0,
        previous_report=None,
    )

    v1_state = materialize_transported_particle_state(v1)
    v2_state = materialize_cumulative_particle_state(v2)
    np.testing.assert_array_equal(v2_state.positions, v1_state.positions)
    np.testing.assert_array_equal(v2_state.gamma, v1_state.gamma)
    np.testing.assert_array_equal(v2_state.sigma, v1_state.sigma)
    assert _cloud(v2).particle_ids == v1.transported_particle_cloud.particle_ids
    assert _cloud(v2).lineage == v1.transported_particle_cloud.lineage
    np.testing.assert_array_equal(_fact_positions(v2), _fact_positions(v1))
    assert tuple(fact.node_id for fact in v2.facts) == tuple(
        fact.node_id for fact in v1.facts
    )
    assert tuple(fact.lineage_epoch for fact in v2.facts) == tuple(
        fact.lineage_epoch for fact in v1.facts
    )
    assert tuple(fact.parent_frontier_id for fact in v2.facts) == tuple(
        fact.parent_frontier_id for fact in v1.facts
    )
    assert tuple(fact.parent_frontier_digest_sha256 for fact in v2.facts) == tuple(
        fact.parent_frontier_digest_sha256 for fact in v1.facts
    )
    _assert_release_slices(v2, (1,))


@pytest.mark.parametrize("attack", ["copy", "reorder", "one_ulp"])
def test_copied_reordered_or_one_ulp_report_is_not_live_attested(
    attack: str,
) -> None:
    mapper = NodeOwnedDVMRibbonShadow(wing_id="wing", source_family="lev")
    sources = (_source(0), _source(1))
    ribbon = _map_step(
        mapper,
        sources,
        active_by_cell=(True, True),
        source_time_s=0.0,
        previous_report=None,
    )
    report = _transport(
        mapper,
        ribbon,
        source_time_s=0.0,
        previous_report=None,
    )
    if attack == "copy":
        forged = replace(report)
    elif attack == "reorder":
        cloud = _cloud(report)
        forged_cloud = replace(
            cloud,
            positions_gp1_m=cloud.positions_gp1_m[::-1],
            gamma_vector_m3_per_s=cloud.gamma_vector_m3_per_s[::-1],
            sigma_m=cloud.sigma_m[::-1],
            particle_ids=cloud.particle_ids[::-1],
            lineage=cloud.lineage[::-1],
        )
        forged = replace(report, transported_particle_cloud=forged_cloud)
    else:
        cloud = _cloud(report)
        positions = [list(row) for row in cloud.positions_gp1_m]
        positions[0][0] = float(np.nextafter(positions[0][0], float("inf")))
        forged = replace(
            report,
            transported_particle_cloud=replace(
                cloud,
                positions_gp1_m=tuple(tuple(row) for row in positions),
            ),
        )
    with pytest.raises(ValueError):
        validate_cumulative_cloud_transport_report(forged)
    assert validate_cumulative_cloud_transport_report(report) is report


def test_handoff_copy_cross_wing_and_failed_transport_roll_back_for_clean_retry() -> (
    None
):
    mapper = NodeOwnedDVMRibbonShadow(wing_id="wing", source_family="lev")
    sources = (_source(0), _source(1))
    ribbon = _map_step(
        mapper,
        sources,
        active_by_cell=(True, True),
        source_time_s=0.0,
        previous_report=None,
    )

    with pytest.raises(ValueError, match="live|directly produced"):
        attest_cumulative_ribbon_handoff(
            mapper,
            replace(ribbon),
            wing_id="wing",
            source_time_s=0.0,
        )
    with pytest.raises(ValueError, match="wing"):
        attest_cumulative_ribbon_handoff(
            mapper,
            ribbon,
            wing_id="other-wing",
            source_time_s=0.0,
        )
    handoff = attest_cumulative_ribbon_handoff(
        mapper,
        ribbon,
        wing_id="wing",
        source_time_s=0.0,
    )
    with pytest.raises(ValueError):
        transport_accumulated_particle_cloud(
            replace(handoff),
            smoothing_radius_m=SIGMA_BIRTH_M,
            deposition_target_spacing_m=TARGET_SPACING_M,
            transport_end_time_s=DT_S,
            transport_substeps=1,
            freestream_velocity_gp1_m_per_s=FREESTREAM_GP1_M_PER_S,
        )
    with pytest.raises(ValueError):
        transport_accumulated_particle_cloud(
            handoff,
            smoothing_radius_m=SIGMA_BIRTH_M,
            deposition_target_spacing_m=np.nextafter(
                SIGMA_BIRTH_M / FROZEN_OVERLAP_LAMBDA,
                float("inf"),
            ),
            transport_end_time_s=DT_S,
            transport_substeps=1,
            freestream_velocity_gp1_m_per_s=FREESTREAM_GP1_M_PER_S,
        )
    clean = transport_accumulated_particle_cloud(
        handoff,
        smoothing_radius_m=SIGMA_BIRTH_M,
        deposition_target_spacing_m=TARGET_SPACING_M,
        transport_end_time_s=DT_S,
        transport_substeps=1,
        freestream_velocity_gp1_m_per_s=FREESTREAM_GP1_M_PER_S,
    )
    assert validate_cumulative_cloud_transport_report(clean) is clean
    with pytest.raises(ValueError, match="replay|consum"):
        transport_accumulated_particle_cloud(
            handoff,
            smoothing_radius_m=SIGMA_BIRTH_M,
            deposition_target_spacing_m=TARGET_SPACING_M,
            transport_end_time_s=DT_S,
            transport_substeps=1,
            freestream_velocity_gp1_m_per_s=FREESTREAM_GP1_M_PER_S,
        )


def test_three_then_four_release_cloud_is_additive_and_old_reports_are_immutable() -> (
    None
):
    mapper = NodeOwnedDVMRibbonShadow(wing_id="wing", source_family="lev")
    sources = (_source(0), _source(1))
    ribbons: list[DVMRibbonShadowResult] = []
    reports: list[Any] = []
    release_counts: list[int] = []
    previous: Any | None = None
    for index in range(4):
        source_time = index * DT_S
        ribbon = _map_step(
            mapper,
            sources,
            active_by_cell=(True, True),
            source_time_s=source_time,
            previous_report=previous,
        )
        deposited = _deposited(ribbon)
        immutable_before = None if previous is None else _snapshot(previous)
        report = _transport(
            mapper,
            ribbon,
            source_time_s=source_time,
            previous_report=previous,
        )
        if previous is not None:
            assert _snapshot(previous) == immutable_before
            previous_cloud = _cloud(previous)
            current_cloud = _cloud(report)
            old_count = len(previous_cloud.particle_ids)
            assert current_cloud.particle_ids[:old_count] == previous_cloud.particle_ids
            assert current_cloud.lineage[:old_count] == previous_cloud.lineage
        release_counts.append(len(deposited.particle_ids))
        assert len(_cloud(report).particle_ids) == sum(release_counts)
        _assert_release_slices(report, range(1, index + 2))
        ribbons.append(ribbon)
        reports.append(report)
        previous = report

    assert len(_cloud(reports[2]).release_slices) == 3
    assert len(_cloud(reports[3]).release_slices) == 4
    assert ribbons[1].diagnostics.continuous_node_count == 3
    assert ribbons[2].diagnostics.continuous_node_count == 3
    assert ribbons[3].diagnostics.continuous_node_count == 3


def test_second_release_uses_one_combined_stage_field_with_independent_replay() -> None:
    mapper = NodeOwnedDVMRibbonShadow(wing_id="wing", source_family="lev")
    sources = (_source(0), _source(1))
    first_ribbon = _map_step(
        mapper,
        sources,
        active_by_cell=(True, True),
        source_time_s=0.0,
        previous_report=None,
    )
    first_report = _transport(
        mapper,
        first_ribbon,
        source_time_s=0.0,
        previous_report=None,
    )
    second_ribbon = _map_step(
        mapper,
        sources,
        active_by_cell=(True, True),
        source_time_s=DT_S,
        previous_report=first_report,
    )
    second_deposited = _deposited(second_ribbon)
    second_report = _transport(
        mapper,
        second_ribbon,
        source_time_s=DT_S,
        previous_report=first_report,
    )

    old_state = materialize_cumulative_particle_state(first_report)
    newborn_state = make_particle_state(
        second_deposited.positions,
        second_deposited.gamma,
        second_deposited.sigma,
    )
    combined_initial = _concatenate_states(old_state, newborn_state)
    replay_state, replay_frontiers = _advance_particles_and_frontiers(
        combined_initial,
        _active_birth_positions(second_ribbon),
        start_time_s=DT_S,
        end_time_s=2.0 * DT_S,
        substeps=1,
    )
    observed = materialize_cumulative_particle_state(second_report)
    np.testing.assert_array_equal(observed.positions, replay_state.positions)
    np.testing.assert_array_equal(observed.gamma, replay_state.gamma)
    np.testing.assert_array_equal(observed.sigma, replay_state.sigma)
    np.testing.assert_array_equal(_fact_positions(second_report), replay_frontiers)

    old_independent, _ = _advance_particles_and_frontiers(
        old_state,
        np.empty((0, 3), dtype=np.float64),
        start_time_s=DT_S,
        end_time_s=2.0 * DT_S,
        substeps=1,
    )
    newborn_independent, _ = _advance_particles_and_frontiers(
        newborn_state,
        np.empty((0, 3), dtype=np.float64),
        start_time_s=DT_S,
        end_time_s=2.0 * DT_S,
        substeps=1,
    )
    split_control = _concatenate_states(old_independent, newborn_independent)
    assert not np.array_equal(observed.positions, split_control.positions)
    assert np.max(np.abs(observed.positions - split_control.positions)) > 0.0


def test_active_inactive_restart_continuous_has_no_phantom_release() -> None:
    mapper = NodeOwnedDVMRibbonShadow(wing_id="wing", source_family="lev")
    sources = (_source(0), _source(1))

    first = _map_step(
        mapper,
        sources,
        active_by_cell=(True, True),
        source_time_s=0.0,
        previous_report=None,
    )
    report1 = _transport(
        mapper,
        first,
        source_time_s=0.0,
        previous_report=None,
    )
    count1 = len(_cloud(report1).particle_ids)

    inactive = _map_step(
        mapper,
        sources,
        active_by_cell=(False, False),
        source_time_s=DT_S,
        previous_report=None,
    )
    assert inactive.edge_graph is None
    assert all(birth.mode == "inactive" for birth in inactive.node_births)
    report2 = _transport(
        mapper,
        inactive,
        source_time_s=DT_S,
        previous_report=report1,
    )
    assert len(_cloud(report2).particle_ids) == count1
    assert report2.facts == ()
    _assert_release_slices(report2, (1,))

    restart = _map_step(
        mapper,
        sources,
        active_by_cell=(True, True),
        source_time_s=2.0 * DT_S,
        previous_report=None,
    )
    assert all(birth.mode == "restart" for birth in restart.node_births)
    restart_count = len(_deposited(restart).particle_ids)
    report3 = _transport(
        mapper,
        restart,
        source_time_s=2.0 * DT_S,
        previous_report=report2,
    )
    assert len(_cloud(report3).particle_ids) == count1 + restart_count
    _assert_release_slices(report3, (1, 3))

    continuous = _map_step(
        mapper,
        sources,
        active_by_cell=(True, True),
        source_time_s=3.0 * DT_S,
        previous_report=report3,
    )
    assert all(birth.mode == "continuous" for birth in continuous.node_births)
    continuous_count = len(_deposited(continuous).particle_ids)
    report4 = _transport(
        mapper,
        continuous,
        source_time_s=3.0 * DT_S,
        previous_report=report3,
    )
    assert len(_cloud(report4).particle_ids) == (
        count1 + restart_count + continuous_count
    )
    _assert_release_slices(report4, (1, 3, 4))


def test_partial_activity_and_cross_wing_report_fail_transactionally() -> None:
    mapper_a = NodeOwnedDVMRibbonShadow(wing_id="wing", source_family="lev")
    sources_a = (_source(0), _source(1))
    first_a = _map_step(
        mapper_a,
        sources_a,
        active_by_cell=(True, False),
        source_time_s=0.0,
        previous_report=None,
    )
    report_a = _transport(
        mapper_a,
        first_a,
        source_time_s=0.0,
        previous_report=None,
    )
    assert len(report_a.facts) == 2

    mapper_b = NodeOwnedDVMRibbonShadow(wing_id="other-wing", source_family="lev")
    sources_b = (_source(100), _source(101))
    first_b = mapper_b.map_step(
        _cells(_events(sources_b, (True, False))),
        _nodes(),
        delta_time_s=DT_S,
        transport_enabled=True,
        source_time_s=0.0,
    )
    handoff_b = attest_cumulative_ribbon_handoff(
        mapper_b,
        first_b,
        wing_id="other-wing",
        source_time_s=0.0,
    )
    report_b = transport_accumulated_particle_cloud(
        handoff_b,
        smoothing_radius_m=SIGMA_BIRTH_M,
        deposition_target_spacing_m=TARGET_SPACING_M,
        transport_end_time_s=DT_S,
        transport_substeps=1,
        freestream_velocity_gp1_m_per_s=FREESTREAM_GP1_M_PER_S,
    )

    second_events_b = _events(sources_b, (True, True))
    before = (
        mapper_b.state_snapshot,
        mapper_b.cell_binding_snapshot,
        mapper_b.cell_history_snapshot,
        mapper_b.transport_handoff_snapshot,
    )
    with pytest.raises(ValueError, match="wing"):
        mapper_b.map_step(
            _cells(second_events_b),
            _nodes(),
            delta_time_s=DT_S,
            transport_enabled=True,
            source_time_s=DT_S,
            frontier_transport_report=report_a,
        )
    assert (
        mapper_b.state_snapshot,
        mapper_b.cell_binding_snapshot,
        mapper_b.cell_history_snapshot,
        mapper_b.transport_handoff_snapshot,
    ) == before
    accepted = mapper_b.map_step(
        _cells(second_events_b),
        _nodes(),
        delta_time_s=DT_S,
        transport_enabled=True,
        source_time_s=DT_S,
        frontier_transport_report=report_b,
    )
    modes = {int(birth.node_id): birth.mode for birth in accepted.node_births}
    assert modes == {0: "continuous", 1: "continuous", 2: "first"}


def test_consumed_report_replay_rolls_back_then_current_report_retries_cleanly() -> (
    None
):
    mapper = NodeOwnedDVMRibbonShadow(wing_id="wing", source_family="lev")
    sources = (_source(0), _source(1))
    first = _map_step(
        mapper,
        sources,
        active_by_cell=(True, True),
        source_time_s=0.0,
        previous_report=None,
    )
    report1 = _transport(
        mapper,
        first,
        source_time_s=0.0,
        previous_report=None,
    )
    second = _map_step(
        mapper,
        sources,
        active_by_cell=(True, True),
        source_time_s=DT_S,
        previous_report=report1,
    )
    report2 = _transport(
        mapper,
        second,
        source_time_s=DT_S,
        previous_report=report1,
    )

    third_events = _events(sources, (True, True))
    before = (
        mapper.state_snapshot,
        mapper.cell_binding_snapshot,
        mapper.cell_history_snapshot,
        mapper.transport_handoff_snapshot,
    )
    with pytest.raises(ValueError, match="replay|stale"):
        _map_events(
            mapper,
            third_events,
            source_time_s=2.0 * DT_S,
            previous_report=report1,
        )
    assert (
        mapper.state_snapshot,
        mapper.cell_binding_snapshot,
        mapper.cell_history_snapshot,
        mapper.transport_handoff_snapshot,
    ) == before
    accepted = _map_events(
        mapper,
        third_events,
        source_time_s=2.0 * DT_S,
        previous_report=report2,
    )
    assert accepted.diagnostics.continuous_node_count == 3


def test_transport_trace_counts_hash_and_replay_residual_are_production_derived() -> (
    None
):
    reports: list[Any] = []
    ribbons: list[DVMRibbonShadowResult] = []
    for substeps in (1, 2):
        _, _, ribbon, handoff = _first_active_handoff()
        report = transport_accumulated_particle_cloud(
            handoff,
            smoothing_radius_m=SIGMA_BIRTH_M,
            deposition_target_spacing_m=TARGET_SPACING_M,
            transport_end_time_s=DT_S,
            transport_substeps=substeps,
            freestream_velocity_gp1_m_per_s=FREESTREAM_GP1_M_PER_S,
        )
        assert validate_cumulative_cloud_transport_report(report) is report
        assert report.lsrk3_call_count == substeps
        assert report.lsrk3_stage_count == 3 * substeps
        assert report.stage_pre_field_call_count == 3 * substeps
        assert report.deposition_call_count == 1
        assert report.predicted_new_particle_count == report.new_particle_count
        assert report.combined_stage_particle_counts == (
            report.total_particle_count,
        ) * (3 * substeps)
        assert report.exact_append_prefix_max_abs == 0.0
        assert np.isfinite(report.stage_pre_replay_max_abs)
        assert report.stage_pre_replay_max_abs >= 0.0
        assert report.transport_trace
        assert all(isinstance(event, str) and event for event in report.transport_trace)
        assert report.transport_trace_sha256 == _transport_trace_sha256(
            report.transport_trace
        )

        deposited = _deposited(ribbon)
        initial = make_particle_state(
            deposited.positions,
            deposited.gamma,
            deposited.sigma,
        )
        replay_state, replay_frontiers = _advance_particles_and_frontiers(
            initial,
            _active_birth_positions(ribbon),
            start_time_s=0.0,
            end_time_s=DT_S,
            substeps=substeps,
        )
        observed = materialize_cumulative_particle_state(report)
        residuals = (
            float(np.max(np.abs(observed.positions - replay_state.positions))),
            float(np.max(np.abs(observed.gamma - replay_state.gamma))),
            float(np.max(np.abs(observed.sigma - replay_state.sigma))),
            float(np.max(np.abs(_fact_positions(report) - replay_frontiers))),
        )
        assert report.stage_pre_replay_max_abs == max(residuals)
        reports.append(report)
        ribbons.append(ribbon)

    assert reports[0].transport_trace != reports[1].transport_trace
    assert reports[0].transport_trace_sha256 != reports[1].transport_trace_sha256
    assert reports[0].lsrk3_stage_count != reports[1].lsrk3_stage_count
    assert ribbons[0] == ribbons[1]


@pytest.mark.parametrize("fake_kind", ["empty_stages", "tampered_final_state"])
def test_fake_lsrk_cannot_emit_a_report_and_failure_leaves_a_clean_retry(
    monkeypatch: pytest.MonkeyPatch,
    fake_kind: str,
) -> None:
    _, _, _, handoff = _first_active_handoff()
    real_lsrk3 = cumulative_module.lsrk3_step_direct
    fake_call_count = 0

    def fake_lsrk3(*args: Any, **kwargs: Any) -> Any:
        nonlocal fake_call_count
        fake_call_count += 1
        if fake_kind == "empty_stages":
            state = args[0]
            return (
                make_particle_state(state.positions, state.gamma, state.sigma),
                (),
            )
        state, stages = real_lsrk3(*args, **kwargs)
        changed = state.positions.copy()
        changed[0, 0] = np.nextafter(changed[0, 0], float("inf"))
        return make_particle_state(changed, state.gamma, state.sigma), stages

    with monkeypatch.context() as patcher:
        patcher.setattr(cumulative_module, "lsrk3_step_direct", fake_lsrk3)
        with pytest.raises(
            (ValueError, RuntimeError),
            match="callable|backend|LSRK|stage|replay|identity",
        ):
            _transport_handoff(handoff)
    assert fake_call_count in (0, 1)

    clean = _transport_handoff(handoff)
    assert validate_cumulative_cloud_transport_report(clean) is clean


def test_runtime_frozen_lsrk_binding_replacement_fails_before_call_and_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, _, handoff = _first_active_handoff()
    real_lsrk3 = cumulative_module._FROZEN_LSRK3_CALLABLE
    fake_call_count = 0

    def legal_three_stage_no_motion(*args: Any, **kwargs: Any) -> Any:
        nonlocal fake_call_count
        fake_call_count += 1
        initial = args[0]
        _, real_stages = real_lsrk3(*args, **kwargs)
        unmoved = make_particle_state(
            initial.positions,
            initial.gamma,
            initial.sigma,
        )
        fake_stages = tuple(
            replace(stage, pre=unmoved, post=unmoved) for stage in real_stages
        )
        return unmoved, fake_stages

    with monkeypatch.context() as patcher:
        patcher.setattr(
            cumulative_module,
            "_FROZEN_LSRK3_CALLABLE",
            legal_three_stage_no_motion,
        )
        with pytest.raises(
            (ValueError, RuntimeError),
            match="lsrk3|LSRK|frozen|callable|binding|replaced|identity",
        ):
            _transport_handoff(handoff)
    assert fake_call_count == 0

    clean = _transport_handoff(handoff)
    assert validate_cumulative_cloud_transport_report(clean) is clean


def test_edge_private_deposition_dependency_replacement_fails_before_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, _, handoff = _first_active_handoff()
    real_helper = edge_bridge_module._deposit_validated_graph
    replacement_call_count = 0

    def equivalent_wrapper(*args: Any, **kwargs: Any) -> Any:
        nonlocal replacement_call_count
        replacement_call_count += 1
        return real_helper(*args, **kwargs)

    with monkeypatch.context() as patcher:
        patcher.setattr(
            edge_bridge_module,
            "_deposit_validated_graph",
            equivalent_wrapper,
        )
        with pytest.raises(
            (ValueError, RuntimeError),
            match="edge_deposition|transitive|dependency|callable|replaced|identity",
        ):
            _transport_handoff(handoff)
    assert replacement_call_count == 0

    clean = _transport_handoff(handoff)
    assert validate_cumulative_cloud_transport_report(clean) is clean


def test_lsrk_transitive_field_replacement_fails_before_call_and_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, _, handoff = _first_active_handoff()
    real_field = transport_module.direct_gaussian_erf_velocity_jacobian
    replacement_call_count = 0

    def zero_field(*args: Any, **kwargs: Any) -> Any:
        nonlocal replacement_call_count
        replacement_call_count += 1
        field = real_field(*args, **kwargs)
        return replace(
            field,
            velocity=np.zeros_like(field.velocity),
            jacobian=np.zeros_like(field.jacobian),
        )

    with monkeypatch.context() as patcher:
        patcher.setattr(
            transport_module,
            "direct_gaussian_erf_velocity_jacobian",
            zero_field,
        )
        with pytest.raises(
            (ValueError, RuntimeError),
            match="lsrk3|transitive|dependency|field|callable|replaced|identity",
        ):
            transport_accumulated_particle_cloud(
                handoff,
                smoothing_radius_m=SIGMA_BIRTH_M,
                deposition_target_spacing_m=TARGET_SPACING_M,
                transport_end_time_s=DT_S,
                transport_substeps=1,
                freestream_velocity_gp1_m_per_s=(0.0, 0.0, 0.0),
            )
    assert replacement_call_count == 0

    clean = _transport_handoff(handoff)
    assert validate_cumulative_cloud_transport_report(clean) is clean


@pytest.mark.parametrize("attack", ["module_source", "imported_callable"])
def test_edge_bridge_source_and_actual_callable_tampering_fail_before_commit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    attack: str,
) -> None:
    _, _, _, handoff = _first_active_handoff()
    if attack == "module_source":
        source_path = Path(edge_bridge_module.__file__).resolve()
        tampered_path = tmp_path / "rvpm_edge_bridge_tampered.py"
        tampered_path.write_bytes(source_path.read_bytes() + b"\n# audit tamper\n")
        with monkeypatch.context() as patcher:
            patcher.setattr(edge_bridge_module, "__file__", str(tampered_path))
            with pytest.raises(
                (ValueError, RuntimeError),
                match=(
                    "edge.*artifact|edge_bridge|module path|source|callable|"
                    "stale|identity"
                ),
            ):
                _transport_handoff(handoff)
    else:
        real_deposit = cumulative_module.deposit_edge_graph_prescribed_sigma_and_spacing
        replacement_call_count = 0

        def replacement(*args: Any, **kwargs: Any) -> Any:
            nonlocal replacement_call_count
            replacement_call_count += 1
            return real_deposit(*args, **kwargs)

        with monkeypatch.context() as patcher:
            patcher.setattr(
                cumulative_module,
                "deposit_edge_graph_prescribed_sigma_and_spacing",
                replacement,
            )
            with pytest.raises(
                (ValueError, RuntimeError),
                match="edge.*callable|callable|stale|identity",
            ):
                _transport_handoff(handoff)
        assert replacement_call_count == 0

    clean = _transport_handoff(handoff)
    assert validate_cumulative_cloud_transport_report(clean) is clean
    assert clean.edge_bridge_artifact_sha256


def test_resource_caps_are_frozen_and_huge_substeps_fail_then_retry() -> None:
    assert MAX_TRANSPORT_SUBSTEPS == 4096
    assert MAX_PARTICLES_PER_RELEASE == 250_000
    assert MAX_CUMULATIVE_PARTICLES == 1_000_000
    assert MAX_PARTICLE_COUNT == MAX_CUMULATIVE_PARTICLES
    _, _, _, handoff = _first_active_handoff()

    with pytest.raises(ValueError, match="substep.*(?:limit|cap|budget|4096)"):
        _transport_handoff(handoff, substeps=10**12)
    clean = _transport_handoff(handoff)
    assert validate_cumulative_cloud_transport_report(clean) is clean


def test_extreme_spacing_is_rejected_from_predicted_count_before_deposition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, _, handoff = _first_active_handoff()
    allocation_call_count = 0

    def allocation_trap(*args: Any, **kwargs: Any) -> Any:
        nonlocal allocation_call_count
        allocation_call_count += 1
        raise AssertionError("edge deposition allocation was reached")

    with monkeypatch.context() as patcher:
        patcher.setattr(
            edge_bridge_module,
            "_deposit_validated_graph",
            allocation_trap,
        )
        with pytest.raises(
            ValueError,
            match="predicted|particle.*(?:limit|cap|budget)",
        ):
            _transport_handoff(handoff, spacing_m=1.0e-12)
    assert allocation_call_count == 0

    clean = _transport_handoff(handoff)
    assert validate_cumulative_cloud_transport_report(clean) is clean


def test_cumulative_count_limit_fails_before_new_release_allocation_then_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mapper, sources, first, first_handoff = _first_active_handoff()
    report1 = _transport_handoff(first_handoff)
    assert first.diagnostics.first_node_count == 3
    second = _map_step(
        mapper,
        sources,
        active_by_cell=(True, True),
        source_time_s=DT_S,
        previous_report=report1,
    )
    second_handoff = attest_cumulative_ribbon_handoff(
        mapper,
        second,
        wing_id="wing",
        source_time_s=DT_S,
        previous_report=report1,
    )
    allocation_call_count = 0

    def allocation_trap(*args: Any, **kwargs: Any) -> Any:
        nonlocal allocation_call_count
        allocation_call_count += 1
        raise AssertionError("new-release deposition allocation was reached")

    with monkeypatch.context() as patcher:
        patcher.setattr(
            cumulative_module,
            "MAX_CUMULATIVE_PARTICLES",
            report1.total_particle_count,
        )
        patcher.setattr(
            cumulative_module,
            "MAX_PARTICLE_COUNT",
            report1.total_particle_count,
        )
        patcher.setattr(
            edge_bridge_module,
            "_deposit_validated_graph",
            allocation_trap,
        )
        with pytest.raises(
            ValueError,
            match="cumulative|particle.*(?:limit|cap|budget)",
        ):
            _transport_handoff(second_handoff, end_time_s=2.0 * DT_S)
    assert allocation_call_count == 0

    clean = _transport_handoff(second_handoff, end_time_s=2.0 * DT_S)
    assert validate_cumulative_cloud_transport_report(clean) is clean


def test_oversized_parent_cloud_fails_before_array_materialization_and_retries() -> (
    None
):
    mapper, sources, _, first_handoff = _first_active_handoff()
    report1 = _transport_handoff(first_handoff)
    original_cloud = report1.transported_particle_cloud
    second = _map_step(
        mapper,
        sources,
        active_by_cell=(True, True),
        source_time_s=DT_S,
        previous_report=report1,
    )
    second_handoff = attest_cumulative_ribbon_handoff(
        mapper,
        second,
        wing_id="wing",
        source_time_s=DT_S,
        previous_report=report1,
    )
    array_call_count = 0

    class OversizedParticleIds:
        def __len__(self) -> int:
            return MAX_CUMULATIVE_PARTICLES + 1

    class AllocationTrap:
        def __array__(self, *args: Any, **kwargs: Any) -> Any:
            nonlocal array_call_count
            array_call_count += 1
            raise AssertionError("particle array materialization was reached")

    trap = AllocationTrap()
    oversized_cloud = replace(
        original_cloud,
        positions_gp1_m=trap,
        gamma_vector_m3_per_s=trap,
        sigma_m=trap,
        particle_ids=OversizedParticleIds(),
    )
    forged_report = replace(report1, transported_particle_cloud=oversized_cloud)
    with pytest.raises(ValueError, match="cloud|cumulative|particle.*(?:cap|limit)"):
        validate_cumulative_cloud_transport_report(forged_report)
    assert array_call_count == 0

    object.__setattr__(report1, "transported_particle_cloud", oversized_cloud)
    try:
        with pytest.raises(
            ValueError,
            match="cloud|cumulative|particle.*(?:cap|limit)",
        ):
            _transport_handoff(second_handoff, end_time_s=2.0 * DT_S)
    finally:
        object.__setattr__(report1, "transported_particle_cloud", original_cloud)
    assert array_call_count == 0

    clean = _transport_handoff(second_handoff, end_time_s=2.0 * DT_S)
    assert validate_cumulative_cloud_transport_report(clean) is clean


def test_report_fact_slice_and_trace_nonfinite_or_time_tampering_is_rejected() -> None:
    mapper = NodeOwnedDVMRibbonShadow(wing_id="wing", source_family="lev")
    sources = (_source(0), _source(1))
    ribbon = _map_step(
        mapper,
        sources,
        active_by_cell=(True, True),
        source_time_s=0.0,
        previous_report=None,
    )
    report = _transport(
        mapper,
        ribbon,
        source_time_s=0.0,
        previous_report=None,
    )
    fact = report.facts[0]
    cloud = _cloud(report)
    release = cloud.release_slices[0]

    def with_fact(**changes: Any) -> Any:
        changed = replace(fact, **changes)
        return replace(report, facts=(changed,) + report.facts[1:])

    def with_slice(**changes: Any) -> Any:
        changed = replace(release, **changes)
        changed_cloud = replace(
            cloud,
            release_slices=(changed,) + cloud.release_slices[1:],
        )
        return replace(report, transported_particle_cloud=changed_cloud)

    attacks: tuple[tuple[str, Any, str], ...] = (
        (
            "report-start-nan",
            replace(report, transport_start_time_s=float("nan")),
            "finite|time",
        ),
        (
            "report-end-inf",
            replace(report, transport_end_time_s=float("inf")),
            "finite|time",
        ),
        (
            "report-end-one-ulp",
            replace(
                report,
                transport_end_time_s=np.nextafter(DT_S, float("inf")),
            ),
            "time|digest",
        ),
        (
            "report-replay-residual-nan",
            replace(report, stage_pre_replay_max_abs=float("nan")),
            "finite|replay",
        ),
        (
            "fact-position-nan",
            with_fact(advected_position_gp1_m=(float("nan"), 0.0, 0.0)),
            "finite|position",
        ),
        (
            "fact-position-inf",
            with_fact(advected_position_gp1_m=(float("inf"), 0.0, 0.0)),
            "finite|position",
        ),
        (
            "fact-start-nan",
            with_fact(transport_start_time_s=float("nan")),
            "finite|time",
        ),
        (
            "fact-end-inf",
            with_fact(transport_end_time_s=float("inf")),
            "finite|time",
        ),
        (
            "fact-end-one-ulp",
            with_fact(transport_end_time_s=np.nextafter(DT_S, float("inf"))),
            "time|digest|disagrees",
        ),
        (
            "slice-time-nan",
            with_slice(source_time_s=float("nan")),
            "finite|time",
        ),
        (
            "slice-time-inf",
            with_slice(source_time_s=float("inf")),
            "finite|time",
        ),
        (
            "slice-time-one-ulp",
            with_slice(source_time_s=np.nextafter(0.0, float("inf"))),
            "time|digest|inconsistent",
        ),
        (
            "slice-sigma-nan",
            with_slice(smoothing_radius_m=float("nan")),
            "finite|smoothing",
        ),
        (
            "slice-spacing-inf",
            with_slice(deposition_target_spacing_m=float("inf")),
            "finite|spacing",
        ),
    )
    for label, forged, message in attacks:
        with pytest.raises(ValueError, match=message):
            validate_cumulative_cloud_transport_report(forged)
        assert label
    assert validate_cumulative_cloud_transport_report(report) is report


def test_report_exposes_only_transport_ownership_and_zero_write_counters() -> None:
    mapper = NodeOwnedDVMRibbonShadow(wing_id="wing", source_family="lev")
    sources = (_source(0), _source(1))
    ribbon = _map_step(
        mapper,
        sources,
        active_by_cell=(True, True),
        source_time_s=0.0,
        previous_report=None,
    )
    report = _transport(
        mapper,
        ribbon,
        source_time_s=0.0,
        previous_report=None,
    )
    assert report.feedback_call_count == 0
    assert report.parent_write_count == 0
    assert report.load_write_count == 0
    assert report.observation_access == "none"
    assert report.target_case_branch == "none"
    assert report.exact_append_passed
    assert report.one_combined_field_passed
    assert report.stage_pre_replay_passed
    report_fields = {item.name for item in fields(report)}
    cloud_fields = {item.name for item in fields(_cloud(report))}
    forbidden = (
        "force",
        "lift",
        "drag",
        "pressure",
        "coefficient",
        "feedback_velocity",
        "load_vector",
        "load_delta",
    )
    assert not any(
        token in name for name in report_fields | cloud_fields for token in forbidden
    )
