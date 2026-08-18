from __future__ import annotations

from dataclasses import fields, replace
import gc
from inspect import signature
from typing import Any
import weakref

import numpy as np
import pytest

from fluxvortex.rvpm_edge_bridge import (
    FROZEN_OVERLAP_LAMBDA,
    canonical_edge_key,
    deposit_edge_graph,
    deposit_edge_graph_fixed_sigma,
    deposit_edge_graph_prescribed_sigma_and_spacing,
)

from forward_flight_benchmarks.ldvm_uvlm_correction import (
    LDVMSectionSettings,
    LESPThreshold,
)
from forward_flight_benchmarks.v5h_cumulative_cloud_transport import (
    attest_cumulative_ribbon_handoff,
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
from forward_flight_benchmarks.v5h_dvm_node_placement import (
    DVMNodePlacementCell,
    DVMNodePlacementResult,
    GP1NodeSectionFact,
    NodeLocalDVMPlacementAdapter,
)
from forward_flight_benchmarks.v5h_passive_frontier_transport import (
    PASSIVE_FRONTIER_FACT_OWNER,
    PASSIVE_FRONTIER_INTERFACE_ID,
    PASSIVE_FRONTIER_PRODUCER_ID,
    TRANSPORT_BACKEND_ID,
    _validate_fixed_sigma_deposition,
    materialize_transported_particle_state,
    ribbon_parent_digest_sha256,
    transport_passive_node_frontiers,
    validate_passive_frontier_transport_report,
)
from forward_flight_benchmarks.v5h_dvm_source import (
    DVMSourceEvent,
    V5hDVMSource,
)


DT_PHYSICAL_S = 0.02 * 0.25**2 / 0.75
FIXED_SIGMA_M = FROZEN_OVERLAP_LAMBDA * 0.0375
PRESCRIBED_TARGET_SPACING_M = FIXED_SIGMA_M / (3.0 * FROZEN_OVERLAP_LAMBDA)
CUMULATIVE_SIGMA_M = 0.085
CUMULATIVE_TARGET_SPACING_M = 0.04


def _source(strip: int) -> V5hDVMSource:
    return V5hDVMSource(
        physical_section_id=f"manufactured:section:{strip}",
        physical_strip_id=f"manufactured:strip:{strip}",
        geometry_identity="explicit zero-camber flat-plate surrogate",
        reference_speed_m_per_s=3.0,
        reference_chord_m=0.25,
        zero_camber_surrogate=True,
        delta_time_convective=0.02,
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


def _manufactured_event(
    source: V5hDVMSource,
    *,
    active: bool,
    gamma_lev: float,
    gamma_tev_solved: float = 0.02,
) -> DVMSourceEvent:
    """Return a directly produced source event; scalar args select test motion."""

    del gamma_tev_solved
    if not active:
        event = source.step(np.deg2rad(5.0), 0.0, 0.0)
        assert not event.lesp_active
        return event
    alpha_deg = 35.0 if abs(gamma_lev) <= 0.1 else 45.0
    event = source.step(np.deg2rad(alpha_deg), 0.0, 0.0)
    assert event.lesp_active
    return event


def _mapping(cell: int, *, sign: int = 1) -> DVMPlaneToGP1Map:
    return DVMPlaneToGP1Map(
        origin_gp1_m=np.array([10.0 + cell, 0.0, 0.0]),
        x_axis_gp1=np.array([1.0, 0.0, 0.0]),
        z_axis_gp1=np.array([0.0, 0.0, -1.0]),
        positive_circulation_axis_gp1=np.array([0.0, 1.0, 0.0]),
        circulation_to_ring_traversal_sign=sign,
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


def _node_placement_for_cell_events(
    events: tuple[DVMSourceEvent, DVMSourceEvent],
    *,
    wing_id: str = "wing",
    topology_patch_id: str = "main-patch",
    coordinate_frame_id: str = "ptera-gp1-wing",
    source_time_s: float = 0.0,
) -> DVMNodePlacementResult:
    node_events = tuple(
        _manufactured_event(_source(100 + index), active=True, gamma_lev=0.1)
        for index in range(3)
    )
    facts: list[GP1NodeSectionFact] = []
    for node_id, event in enumerate(node_events):
        lev_edge = np.asarray(
            event.lev_placement.edge_anchor_position_over_chord_backend_world,
            dtype=float,
        )
        tev_edge = np.asarray(
            event.tev_placement.edge_anchor_position_over_chord_backend_world,
            dtype=float,
        )
        lev_anchor = np.asarray((0.0, float(node_id), 0.0), dtype=float)
        tev_anchor = lev_anchor + 0.25 * (
            (tev_edge[0] - lev_edge[0]) * np.asarray((1.0, 0.0, 0.0))
            + (tev_edge[1] - lev_edge[1]) * np.asarray((0.0, 0.0, -1.0))
        )
        facts.append(
            GP1NodeSectionFact(
                wing_id=wing_id,
                node_id=node_id,
                source_step_index=int(event.lineage.source_step_index),
                source_time_s=source_time_s,
                event=event,
                lev_edge_anchor_gp1_m=tuple(float(item) for item in lev_anchor),
                tev_edge_anchor_gp1_m=tuple(float(item) for item in tev_anchor),
                reference_chord_m=0.25,
                reference_speed_m_per_s=3.0,
                dvm_x_axis_gp1=(1.0, 0.0, 0.0),
                dvm_z_axis_gp1=(0.0, 0.0, -1.0),
                positive_span_axis_gp1=(0.0, 1.0, 0.0),
                topology_patch_id=topology_patch_id,
                coordinate_frame_id=coordinate_frame_id,
                node_lineage_id=event.lineage.section_lineage_id,
                geometry_token=event.provenance.geometry_hash_sha256,
            )
        )
    placement_cells = tuple(
        DVMNodePlacementCell(
            cell_id=f"cell-{index}",
            left_node_fact=facts[index],
            right_node_fact=facts[index + 1],
            cell_source_event=event,
        )
        for index, event in enumerate(events)
    )
    return NodeLocalDVMPlacementAdapter(wing_id=wing_id).map_step(
        placement_cells,
        delta_time_s=DT_PHYSICAL_S,
    )


def _cells(
    events: tuple[DVMSourceEvent, DVMSourceEvent],
    *,
    signs: tuple[int, int] = (1, 1),
) -> tuple[DVMSpanCellSource, ...]:
    return (
        DVMSpanCellSource("cell-0", 0, 1, events[0], _mapping(0, sign=signs[0])),
        DVMSpanCellSource("cell-1", 1, 2, events[1], _mapping(1, sign=signs[1])),
    )


class _ExplodingInput:
    def __iter__(self) -> Any:
        raise AssertionError("disabled mapper iterated an input")

    def __float__(self) -> float:
        raise AssertionError("disabled mapper inspected a numeric input")


def _births_by_id(result: DVMRibbonShadowResult) -> dict[int, Any]:
    return {int(item.node_id): item for item in result.node_births}


def _map_single_event(event: DVMSourceEvent) -> DVMRibbonShadowResult:
    return NodeOwnedDVMRibbonShadow(wing_id="wing", source_family="lev").map_step(
        (DVMSpanCellSource("cell", 0, 1, event, _mapping(0)),),
        _nodes()[:2],
        delta_time_s=DT_PHYSICAL_S,
    )


def test_disabled_is_input_blind_and_state_exact() -> None:
    mapper = NodeOwnedDVMRibbonShadow(wing_id="wing", source_family="lev")
    before = mapper.state_snapshot
    result = mapper.map_step(
        _ExplodingInput(),  # type: ignore[arg-type]
        _ExplodingInput(),  # type: ignore[arg-type]
        delta_time_s=_ExplodingInput(),
        enabled=False,
        transport_enabled=_ExplodingInput(),  # type: ignore[arg-type]
        source_time_s=_ExplodingInput(),
        frontier_transport_report=_ExplodingInput(),  # type: ignore[arg-type]
        node_placement_result=_ExplodingInput(),
    )
    assert result.edge_graph is None
    assert result.feedback_velocity is None
    assert result.diagnostics.canonical_eligible is False
    assert result.diagnostics.feedback_call_count == 0
    assert mapper.state_snapshot == before


def test_live_cell_events_are_global_exact_once_across_ribbon_mappers() -> None:
    sources = (_source(0), _source(1))
    events = (
        _manufactured_event(sources[0], active=True, gamma_lev=0.1),
        _manufactured_event(sources[1], active=True, gamma_lev=0.1),
    )
    first_mapper = NodeOwnedDVMRibbonShadow(wing_id="wing", source_family="lev")
    first_mapper.map_step(_cells(events), _nodes(), delta_time_s=DT_PHYSICAL_S)

    replay_mapper = NodeOwnedDVMRibbonShadow(wing_id="wing", source_family="lev")
    before = _mapper_snapshots(replay_mapper)
    with pytest.raises(ValueError, match="another ribbon mapper"):
        replay_mapper.map_step(
            _cells(events),
            _nodes(),
            delta_time_s=DT_PHYSICAL_S,
        )
    assert _mapper_snapshots(replay_mapper) == before

    # Exact-once is live identity ownership, not a ban on deterministic output.
    # Independent source sessions may produce byte-identical manifests.
    fresh_sources = (_source(0), _source(1))
    fresh_events = (
        _manufactured_event(fresh_sources[0], active=True, gamma_lev=0.1),
        _manufactured_event(fresh_sources[1], active=True, gamma_lev=0.1),
    )
    assert tuple(item.producer_manifest_sha256 for item in fresh_events) == tuple(
        item.producer_manifest_sha256 for item in events
    )
    assert all(
        fresh is not original
        for fresh, original in zip(fresh_events, events, strict=True)
    )
    accepted = replay_mapper.map_step(
        _cells(fresh_events),
        _nodes(),
        delta_time_s=DT_PHYSICAL_S,
    )
    assert accepted.diagnostics.source_step_index == 1


def test_global_exact_once_registry_does_not_retain_source_events() -> None:
    source = _source(0)
    event = _manufactured_event(source, active=True, gamma_lev=0.1)
    event_digest = event.producer_manifest_sha256
    event_reference = weakref.ref(event)
    NodeOwnedDVMRibbonShadow(wing_id="wing", source_family="lev").map_step(
        (DVMSpanCellSource("cell", 0, 1, event, _mapping(0)),),
        _nodes()[:2],
        delta_time_s=DT_PHYSICAL_S,
    )
    del event
    gc.collect()
    assert event_reference() is None

    fresh_event = _manufactured_event(_source(0), active=True, gamma_lev=0.1)
    assert fresh_event.producer_manifest_sha256 == event_digest
    accepted = NodeOwnedDVMRibbonShadow(
        wing_id="wing",
        source_family="lev",
    ).map_step(
        (DVMSpanCellSource("cell", 0, 1, fresh_event, _mapping(0)),),
        _nodes()[:2],
        delta_time_s=DT_PHYSICAL_S,
    )
    assert accepted.diagnostics.source_step_index == 1


@pytest.mark.parametrize(
    ("second_wing", "second_family", "error"),
    [
        ("other-wing", "lev", "another wing"),
        ("wing", "tev_persisted", "another source-family role"),
    ],
)
def test_live_event_consumption_fails_across_wing_and_role_boundaries(
    second_wing: str,
    second_family: str,
    error: str,
) -> None:
    sources = (_source(0), _source(1))
    events = (
        _manufactured_event(sources[0], active=True, gamma_lev=0.1),
        _manufactured_event(sources[1], active=True, gamma_lev=0.1),
    )
    NodeOwnedDVMRibbonShadow(wing_id="wing", source_family="lev").map_step(
        _cells(events),
        _nodes(),
        delta_time_s=DT_PHYSICAL_S,
    )
    candidate = NodeOwnedDVMRibbonShadow(
        wing_id=second_wing,
        source_family=second_family,  # type: ignore[arg-type]
    )
    before = _mapper_snapshots(candidate)
    with pytest.raises(ValueError, match=error):
        candidate.map_step(
            _cells(events),
            _nodes(),
            delta_time_s=DT_PHYSICAL_S,
        )
    assert _mapper_snapshots(candidate) == before


def test_live_event_token_rejects_cross_cell_or_plane_map_rebinding() -> None:
    sources = (_source(0), _source(1))
    events = (
        _manufactured_event(sources[0], active=True, gamma_lev=0.1),
        _manufactured_event(sources[1], active=True, gamma_lev=0.1),
    )
    NodeOwnedDVMRibbonShadow(wing_id="wing", source_family="lev").map_step(
        _cells(events),
        _nodes(),
        delta_time_s=DT_PHYSICAL_S,
    )
    rebound_cells = tuple(
        replace(
            cell,
            cell_id=f"rebound-{index}",
            plane_to_gp1=replace(
                cell.plane_to_gp1,
                provenance="rebound patch/frame provenance",
            ),
        )
        for index, cell in enumerate(_cells(events))
    )
    candidate = NodeOwnedDVMRibbonShadow(wing_id="wing", source_family="lev")
    before = _mapper_snapshots(candidate)
    with pytest.raises(ValueError, match="another patch/cell binding"):
        candidate.map_step(
            rebound_cells,
            _nodes(),
            delta_time_s=DT_PHYSICAL_S,
        )
    assert _mapper_snapshots(candidate) == before


def test_failed_precommit_gate_does_not_pollute_global_live_registry() -> None:
    sources = (_source(0), _source(1))
    events = (
        _manufactured_event(sources[0], active=True, gamma_lev=0.1),
        _manufactured_event(sources[1], active=True, gamma_lev=0.1),
    )
    rejected = NodeOwnedDVMRibbonShadow(wing_id="wing", source_family="lev")
    before = _mapper_snapshots(rejected)
    with pytest.raises(ValueError, match="sign disagrees"):
        rejected.map_step(
            _cells(events, signs=(-1, 1)),
            _nodes(),
            delta_time_s=DT_PHYSICAL_S,
        )
    assert _mapper_snapshots(rejected) == before

    accepted = NodeOwnedDVMRibbonShadow(
        wing_id="wing",
        source_family="lev",
    ).map_step(
        _cells(events),
        _nodes(),
        delta_time_s=DT_PHYSICAL_S,
    )
    assert accepted.diagnostics.source_step_index == 1


def test_live_node_placement_binds_patch_manifest_cells_and_nodes() -> None:
    sources = (_source(0), _source(1))
    events = (
        _manufactured_event(sources[0], active=True, gamma_lev=0.1),
        _manufactured_event(sources[1], active=True, gamma_lev=0.1),
    )
    placement = _node_placement_for_cell_events(
        events,
        topology_patch_id="main-patch",
        coordinate_frame_id="ptera-gp1-main",
    )
    result = NodeOwnedDVMRibbonShadow(
        wing_id="wing",
        source_family="lev",
    ).map_step(
        _cells(events),
        placement.kinematics,
        delta_time_s=DT_PHYSICAL_S,
        source_time_s=0.0,
        node_placement_result=placement,
    )
    diagnostics = result.diagnostics
    assert diagnostics.node_placement_bound
    assert diagnostics.patch_binding_passed
    assert (
        diagnostics.node_placement_manifest_sha256 == placement.producer_manifest_sha256
    )
    assert diagnostics.topology_patch_ids == ("main-patch",)
    assert diagnostics.coordinate_frame_ids == ("ptera-gp1-main",)


def test_node_placement_clone_and_kinematics_clone_fail_then_clean_retry() -> None:
    sources = (_source(0), _source(1))
    events = (
        _manufactured_event(sources[0], active=True, gamma_lev=0.1),
        _manufactured_event(sources[1], active=True, gamma_lev=0.1),
    )
    placement = _node_placement_for_cell_events(events)
    mapper = NodeOwnedDVMRibbonShadow(wing_id="wing", source_family="lev")
    before = _mapper_snapshots(mapper)
    with pytest.raises(ValueError, match="directly produced live placement"):
        mapper.map_step(
            _cells(events),
            placement.kinematics,
            delta_time_s=DT_PHYSICAL_S,
            source_time_s=0.0,
            node_placement_result=replace(placement),
        )
    assert _mapper_snapshots(mapper) == before

    cloned_kinematics = tuple(replace(item) for item in placement.kinematics)
    with pytest.raises(ValueError, match="exact live placement kinematics"):
        mapper.map_step(
            _cells(events),
            cloned_kinematics,
            delta_time_s=DT_PHYSICAL_S,
            source_time_s=0.0,
            node_placement_result=placement,
        )
    assert _mapper_snapshots(mapper) == before

    accepted = mapper.map_step(
        _cells(events),
        placement.kinematics,
        delta_time_s=DT_PHYSICAL_S,
        source_time_s=0.0,
        node_placement_result=placement,
    )
    assert accepted.diagnostics.patch_binding_passed


def test_live_node_placement_replay_with_fresh_equal_events_fails_closed() -> None:
    first_sources = (_source(0), _source(1))
    first_events = (
        _manufactured_event(first_sources[0], active=True, gamma_lev=0.1),
        _manufactured_event(first_sources[1], active=True, gamma_lev=0.1),
    )
    placement = _node_placement_for_cell_events(first_events)
    NodeOwnedDVMRibbonShadow(wing_id="wing", source_family="lev").map_step(
        _cells(first_events),
        placement.kinematics,
        delta_time_s=DT_PHYSICAL_S,
        source_time_s=0.0,
        node_placement_result=placement,
    )

    fresh_sources = (_source(0), _source(1))
    fresh_events = (
        _manufactured_event(fresh_sources[0], active=True, gamma_lev=0.1),
        _manufactured_event(fresh_sources[1], active=True, gamma_lev=0.1),
    )
    assert tuple(item.producer_manifest_sha256 for item in fresh_events) == tuple(
        item.producer_manifest_sha256 for item in first_events
    )
    retry_mapper = NodeOwnedDVMRibbonShadow(wing_id="wing", source_family="lev")
    before = _mapper_snapshots(retry_mapper)
    with pytest.raises(ValueError, match="node placement.*another ribbon mapper"):
        retry_mapper.map_step(
            _cells(fresh_events),
            placement.kinematics,
            delta_time_s=DT_PHYSICAL_S,
            source_time_s=0.0,
            node_placement_result=placement,
        )
    assert _mapper_snapshots(retry_mapper) == before

    fresh_placement = _node_placement_for_cell_events(fresh_events)
    assert (
        fresh_placement.producer_manifest_sha256 == placement.producer_manifest_sha256
    )
    accepted = retry_mapper.map_step(
        _cells(fresh_events),
        fresh_placement.kinematics,
        delta_time_s=DT_PHYSICAL_S,
        source_time_s=0.0,
        node_placement_result=fresh_placement,
    )
    assert accepted.diagnostics.patch_binding_passed


def test_node_placement_cell_event_manifest_mismatch_is_retryable() -> None:
    placement_sources = (_source(0), _source(1))
    placement_events = (
        _manufactured_event(placement_sources[0], active=True, gamma_lev=0.1),
        _manufactured_event(placement_sources[1], active=True, gamma_lev=0.1),
    )
    placement = _node_placement_for_cell_events(placement_events)
    different_sources = (_source(0), _source(1))
    different_events = (
        _manufactured_event(different_sources[0], active=True, gamma_lev=0.2),
        _manufactured_event(different_sources[1], active=True, gamma_lev=0.2),
    )
    mapper = NodeOwnedDVMRibbonShadow(wing_id="wing", source_family="lev")
    before = _mapper_snapshots(mapper)
    with pytest.raises(ValueError, match="another cell source event"):
        mapper.map_step(
            _cells(different_events),
            placement.kinematics,
            delta_time_s=DT_PHYSICAL_S,
            source_time_s=0.0,
            node_placement_result=placement,
        )
    assert _mapper_snapshots(mapper) == before

    accepted = mapper.map_step(
        _cells(placement_events),
        placement.kinematics,
        delta_time_s=DT_PHYSICAL_S,
        source_time_s=0.0,
        node_placement_result=placement,
    )
    assert accepted.diagnostics.patch_binding_passed


def _first_layer_with_transport_report(
    mapper: NodeOwnedDVMRibbonShadow,
    sources: tuple[V5hDVMSource, V5hDVMSource],
    *,
    start_time_s: float = 0.0,
    end_time_s: float = DT_PHYSICAL_S,
) -> tuple[DVMRibbonShadowResult, Any]:
    events = (
        _manufactured_event(sources[0], active=True, gamma_lev=0.1),
        _manufactured_event(sources[1], active=True, gamma_lev=0.1),
    )
    first = mapper.map_step(
        _cells(events),
        _nodes(),
        delta_time_s=DT_PHYSICAL_S,
        transport_enabled=True,
    )
    assert first.edge_graph is not None
    deposited = deposit_edge_graph_fixed_sigma(
        first.edge_graph,
        smoothing_radius=FIXED_SIGMA_M,
        step=1,
    )
    report = transport_passive_node_frontiers(
        first,
        deposited,
        wing_id="wing",
        transport_start_time_s=start_time_s,
        transport_end_time_s=end_time_s,
        transport_substeps=1,
        freestream_velocity_gp1_m_per_s=(0.1, 0.0, 0.0),
    )
    return first, report


def _prescribed_deposition_case(
    *,
    target_spacing_m: float = PRESCRIBED_TARGET_SPACING_M,
) -> tuple[NodeOwnedDVMRibbonShadow, DVMRibbonShadowResult, Any]:
    sources = (_source(0), _source(1))
    mapper = NodeOwnedDVMRibbonShadow(wing_id="wing", source_family="lev")
    events = (
        _manufactured_event(sources[0], active=True, gamma_lev=0.1),
        _manufactured_event(sources[1], active=True, gamma_lev=0.1),
    )
    first = mapper.map_step(
        _cells(events),
        _nodes(),
        delta_time_s=DT_PHYSICAL_S,
        transport_enabled=True,
    )
    assert first.edge_graph is not None
    deposited = deposit_edge_graph_prescribed_sigma_and_spacing(
        first.edge_graph,
        smoothing_radius=FIXED_SIGMA_M,
        target_spacing=target_spacing_m,
        step=1,
    )
    return mapper, first, deposited


def _transport_prescribed(
    first: DVMRibbonShadowResult,
    deposited: Any,
    *,
    target_spacing_m: object = PRESCRIBED_TARGET_SPACING_M,
) -> Any:
    return transport_passive_node_frontiers(
        first,
        deposited,
        wing_id="wing",
        transport_start_time_s=0.0,
        transport_end_time_s=DT_PHYSICAL_S,
        transport_substeps=1,
        deposition_target_spacing_m=target_spacing_m,
        freestream_velocity_gp1_m_per_s=(0.1, 0.0, 0.0),
    )


def _mapper_snapshots(mapper: NodeOwnedDVMRibbonShadow) -> tuple[object, ...]:
    return (
        mapper.state_snapshot,
        mapper.cell_binding_snapshot,
        mapper.cell_history_snapshot,
        mapper.transport_handoff_snapshot,
    )


def _transport_snapshots(
    mapper: NodeOwnedDVMRibbonShadow,
    parent: DVMRibbonShadowResult,
) -> tuple[object, ...]:
    return (*_mapper_snapshots(mapper), ribbon_parent_digest_sha256(parent))


def _next_active_events(
    sources: tuple[V5hDVMSource, V5hDVMSource],
) -> tuple[DVMSourceEvent, DVMSourceEvent]:
    return (
        _manufactured_event(sources[0], active=True, gamma_lev=0.1),
        _manufactured_event(sources[1], active=True, gamma_lev=0.1),
    )


def _map_cumulative_ribbon_step(
    mapper: NodeOwnedDVMRibbonShadow,
    sources: tuple[V5hDVMSource, V5hDVMSource],
    *,
    source_time_s: float,
    previous_report: object | None,
    nodes: tuple[SpanNodeKinematics, ...] | None = None,
) -> DVMRibbonShadowResult:
    return mapper.map_step(
        _cells(_next_active_events(sources)),
        _nodes() if nodes is None else nodes,
        delta_time_s=DT_PHYSICAL_S,
        transport_enabled=True,
        source_time_s=source_time_s,
        frontier_transport_report=previous_report,
    )


def _transport_cumulative_ribbon(
    mapper: NodeOwnedDVMRibbonShadow,
    ribbon: DVMRibbonShadowResult,
    *,
    wing_id: str,
    source_time_s: float,
    previous_report: object | None,
    transport_end_time_s: float | None = None,
) -> Any:
    handoff = attest_cumulative_ribbon_handoff(
        mapper,
        ribbon,
        wing_id=wing_id,
        source_time_s=source_time_s,
        previous_report=previous_report,
    )
    report = transport_accumulated_particle_cloud(
        handoff,
        smoothing_radius_m=CUMULATIVE_SIGMA_M,
        deposition_target_spacing_m=CUMULATIVE_TARGET_SPACING_M,
        transport_end_time_s=(
            source_time_s + DT_PHYSICAL_S
            if transport_end_time_s is None
            else transport_end_time_s
        ),
        transport_substeps=1,
        freestream_velocity_gp1_m_per_s=(0.1, 0.0, 0.0),
    )
    assert validate_cumulative_cloud_transport_report(report) is report
    return report


def _first_cumulative_report(
    *,
    wing_id: str = "wing",
    source_family: str = "lev",
    nodes: tuple[SpanNodeKinematics, ...] | None = None,
    transport_end_time_s: float | None = None,
) -> tuple[
    NodeOwnedDVMRibbonShadow,
    tuple[V5hDVMSource, V5hDVMSource],
    DVMRibbonShadowResult,
    Any,
]:
    sources = (_source(0), _source(1))
    mapper = NodeOwnedDVMRibbonShadow(
        wing_id=wing_id,
        source_family=source_family,  # type: ignore[arg-type]
    )
    ribbon = _map_cumulative_ribbon_step(
        mapper,
        sources,
        source_time_s=0.0,
        previous_report=None,
        nodes=nodes,
    )
    report = _transport_cumulative_ribbon(
        mapper,
        ribbon,
        wing_id=wing_id,
        source_time_s=0.0,
        previous_report=None,
        transport_end_time_s=transport_end_time_s,
    )
    return mapper, sources, ribbon, report


def test_cumulative_v2_report_drives_a_real_step_three_release() -> None:
    mapper, sources, _, report1 = _first_cumulative_report()
    second = _map_cumulative_ribbon_step(
        mapper,
        sources,
        source_time_s=DT_PHYSICAL_S,
        previous_report=report1,
    )
    report2 = _transport_cumulative_ribbon(
        mapper,
        second,
        wing_id="wing",
        source_time_s=DT_PHYSICAL_S,
        previous_report=report1,
    )

    third = _map_cumulative_ribbon_step(
        mapper,
        sources,
        source_time_s=2.0 * DT_PHYSICAL_S,
        previous_report=report2,
    )

    assert report2.parent_source_step_index == 2
    assert report2.for_source_step_index == 3
    assert report2.transport_start_time_s == DT_PHYSICAL_S
    assert report2.transport_end_time_s == 2.0 * DT_PHYSICAL_S
    assert third.diagnostics.source_step_index == 3
    assert third.diagnostics.continuous_node_count == 3
    assert third.diagnostics.transport_advance_count == 3
    facts = {int(fact.node_id): fact for fact in report2.facts}
    for node_id, birth in _births_by_id(third).items():
        anchor = np.asarray(birth.anchor_position_gp1_m)
        advected = np.asarray(facts[node_id].advected_position_gp1_m)
        assert birth.mode == "continuous"
        assert birth.birth_position_gp1_m == pytest.approx(
            anchor + (advected - anchor) / 3.0
        )


@pytest.mark.parametrize("attack", ["copy", "report", "fact"])
def test_cumulative_v2_copy_or_tamper_fails_then_clean_retry(attack: str) -> None:
    mapper, sources, _, report = _first_cumulative_report()
    if attack == "copy":
        candidate = replace(report)
    elif attack == "report":
        candidate = replace(report, current_ribbon_digest_sha256="0" * 64)
    else:
        changed_fact = replace(
            report.facts[0],
            parent_ribbon_digest_sha256="0" * 64,
        )
        candidate = replace(report, facts=(changed_fact,) + report.facts[1:])

    second_events = _next_active_events(sources)
    before = _mapper_snapshots(mapper)
    with pytest.raises(ValueError, match="cumulative"):
        mapper.map_step(
            _cells(second_events),
            _nodes(),
            delta_time_s=DT_PHYSICAL_S,
            transport_enabled=True,
            source_time_s=DT_PHYSICAL_S,
            frontier_transport_report=candidate,
        )
    assert _mapper_snapshots(mapper) == before

    accepted = mapper.map_step(
        _cells(second_events),
        _nodes(),
        delta_time_s=DT_PHYSICAL_S,
        transport_enabled=True,
        source_time_s=DT_PHYSICAL_S,
        frontier_transport_report=report,
    )
    assert accepted.diagnostics.transport_advance_count == 3


@pytest.mark.parametrize("attack", ["time", "wing", "parent"])
def test_live_cumulative_v2_cross_binding_attacks_are_transactional(
    attack: str,
) -> None:
    target_mapper, target_sources, _, correct = _first_cumulative_report()
    if attack == "time":
        _, _, _, candidate = _first_cumulative_report(
            transport_end_time_s=2.0 * DT_PHYSICAL_S,
        )
        assert (
            candidate.current_ribbon_digest_sha256
            == correct.current_ribbon_digest_sha256
        )
    elif attack == "wing":
        _, _, _, candidate = _first_cumulative_report(wing_id="other-wing")
    else:
        shifted_nodes = tuple(
            replace(
                node,
                anchor_position_gp1_m=(
                    np.asarray(node.anchor_position_gp1_m)
                    + np.asarray((0.25, 0.0, 0.0))
                ),
            )
            for node in _nodes()
        )
        _, _, _, candidate = _first_cumulative_report(nodes=shifted_nodes)
        assert candidate.wing_id == correct.wing_id
        assert candidate.source_family == correct.source_family
        assert (
            candidate.current_ribbon_digest_sha256
            != correct.current_ribbon_digest_sha256
        )

    second_events = _next_active_events(target_sources)
    before = _mapper_snapshots(target_mapper)
    with pytest.raises(ValueError):
        target_mapper.map_step(
            _cells(second_events),
            _nodes(),
            delta_time_s=DT_PHYSICAL_S,
            transport_enabled=True,
            source_time_s=DT_PHYSICAL_S,
            frontier_transport_report=candidate,
        )
    assert _mapper_snapshots(target_mapper) == before

    accepted = target_mapper.map_step(
        _cells(second_events),
        _nodes(),
        delta_time_s=DT_PHYSICAL_S,
        transport_enabled=True,
        source_time_s=DT_PHYSICAL_S,
        frontier_transport_report=correct,
    )
    assert accepted.diagnostics.transport_advance_count == 3


def test_live_cumulative_v2_cross_family_attack_is_transactional() -> None:
    _, _, _, lev_report = _first_cumulative_report()
    tev_mapper = NodeOwnedDVMRibbonShadow(
        wing_id="wing",
        source_family="tev_persisted",
    )
    tev_sources = (_source(0), _source(1))
    _map_cumulative_ribbon_step(
        tev_mapper,
        tev_sources,
        source_time_s=0.0,
        previous_report=None,
    )

    second_events = _next_active_events(tev_sources)
    before = _mapper_snapshots(tev_mapper)
    with pytest.raises(ValueError, match="source family"):
        tev_mapper.map_step(
            _cells(second_events),
            _nodes(),
            delta_time_s=DT_PHYSICAL_S,
            transport_enabled=True,
            source_time_s=DT_PHYSICAL_S,
            frontier_transport_report=lev_report,
        )
    assert _mapper_snapshots(tev_mapper) == before

    # The same live source events remain cleanly retryable after rejection.
    accepted = tev_mapper.map_step(
        _cells(second_events),
        _nodes(),
        delta_time_s=DT_PHYSICAL_S,
        transport_enabled=False,
        source_time_s=DT_PHYSICAL_S,
    )
    assert accepted.diagnostics.source_family == "tev_persisted"
    assert accepted.diagnostics.source_step_index == 2


def test_cumulative_v2_replay_rolls_back_and_next_report_retries_cleanly() -> None:
    mapper, sources, _, report1 = _first_cumulative_report()
    second = _map_cumulative_ribbon_step(
        mapper,
        sources,
        source_time_s=DT_PHYSICAL_S,
        previous_report=report1,
    )
    report2 = _transport_cumulative_ribbon(
        mapper,
        second,
        wing_id="wing",
        source_time_s=DT_PHYSICAL_S,
        previous_report=report1,
    )

    third_events = _next_active_events(sources)
    before = _mapper_snapshots(mapper)
    with pytest.raises(ValueError, match="replay"):
        mapper.map_step(
            _cells(third_events),
            _nodes(),
            delta_time_s=DT_PHYSICAL_S,
            transport_enabled=True,
            source_time_s=2.0 * DT_PHYSICAL_S,
            frontier_transport_report=report1,
        )
    assert _mapper_snapshots(mapper) == before

    accepted = mapper.map_step(
        _cells(third_events),
        _nodes(),
        delta_time_s=DT_PHYSICAL_S,
        transport_enabled=True,
        source_time_s=2.0 * DT_PHYSICAL_S,
        frontier_transport_report=report2,
    )
    assert accepted.diagnostics.source_step_index == 3
    assert accepted.diagnostics.transport_advance_count == 3


def test_live_cumulative_v2_report_is_global_exact_once_across_mappers() -> None:
    first_mapper, first_sources, _, shared_report = _first_cumulative_report()
    second_mapper = NodeOwnedDVMRibbonShadow(wing_id="wing", source_family="lev")
    second_sources = (_source(0), _source(1))
    second_first = _map_cumulative_ribbon_step(
        second_mapper,
        second_sources,
        source_time_s=0.0,
        previous_report=None,
    )
    assert (
        ribbon_parent_digest_sha256(second_first)
        == shared_report.current_ribbon_digest_sha256
    )

    _map_cumulative_ribbon_step(
        first_mapper,
        first_sources,
        source_time_s=DT_PHYSICAL_S,
        previous_report=shared_report,
    )
    second_events = _next_active_events(second_sources)
    before = _mapper_snapshots(second_mapper)
    with pytest.raises(ValueError, match="replay"):
        second_mapper.map_step(
            _cells(second_events),
            _nodes(),
            delta_time_s=DT_PHYSICAL_S,
            transport_enabled=True,
            source_time_s=DT_PHYSICAL_S,
            frontier_transport_report=shared_report,
        )
    assert _mapper_snapshots(second_mapper) == before

    clean_report = _transport_cumulative_ribbon(
        second_mapper,
        second_first,
        wing_id="wing",
        source_time_s=0.0,
        previous_report=None,
    )
    assert clean_report.report_sha256 == shared_report.report_sha256
    accepted = second_mapper.map_step(
        _cells(second_events),
        _nodes(),
        delta_time_s=DT_PHYSICAL_S,
        transport_enabled=True,
        source_time_s=DT_PHYSICAL_S,
        frontier_transport_report=clean_report,
    )
    assert accepted.diagnostics.transport_advance_count == 3


def test_real_passive_transport_report_drives_one_continuous_release() -> None:
    sources = (_source(0), _source(1))
    mapper = NodeOwnedDVMRibbonShadow(wing_id="wing", source_family="lev")
    first, report = _first_layer_with_transport_report(mapper, sources)

    assert validate_passive_frontier_transport_report(report) is report
    assert report.interface_id == PASSIVE_FRONTIER_INTERFACE_ID
    assert report.producer_id == PASSIVE_FRONTIER_PRODUCER_ID
    assert report.transport_backend_id == TRANSPORT_BACKEND_ID
    assert report.parent_ribbon_digest_sha256 == mapper.transport_handoff_snapshot[0]
    assert len(report.facts) == 3
    assert {fact.node_id for fact in report.facts} == {0, 1, 2}
    assert all(fact.fact_owner == PASSIVE_FRONTIER_FACT_OWNER for fact in report.facts)
    assert report.feedback_call_count == 0
    assert report.parent_write_count == 0
    assert report.load_write_count == 0
    assert report.observation_access == "none"
    assert report.target_case_branch == "none"

    transported = materialize_transported_particle_state(report)
    cloud = report.transported_particle_cloud
    assert transported.positions.shape == (len(cloud.particle_ids), 3)
    assert transported.gamma.shape == transported.positions.shape
    assert transported.sigma.shape == (transported.positions.shape[0],)
    assert len(cloud.lineage) == len(cloud.particle_ids)
    assert all(
        item.particle_id == particle_id
        for item, particle_id in zip(cloud.lineage, cloud.particle_ids, strict=True)
    )
    assert np.all(np.isfinite(transported.positions))
    assert np.all(np.isfinite(transported.gamma))
    assert np.all(np.isfinite(transported.sigma))
    assert np.all(transported.sigma > 0.0)

    second = mapper.map_step(
        _cells(_next_active_events(sources)),
        _nodes(),
        delta_time_s=DT_PHYSICAL_S,
        transport_enabled=True,
        source_time_s=DT_PHYSICAL_S,
        frontier_transport_report=report,
    )
    assert second.diagnostics.continuous_node_count == 3
    assert second.diagnostics.transport_advance_count == 3
    facts = {int(fact.node_id): fact for fact in report.facts}
    births = _births_by_id(second)
    for node_id, birth in births.items():
        anchor = np.asarray(birth.anchor_position_gp1_m)
        advected = np.asarray(facts[node_id].advected_position_gp1_m)
        assert birth.mode == "continuous"
        assert birth.birth_position_gp1_m == pytest.approx(
            anchor + (advected - anchor) / 3.0
        )
    assert report.report_sha256 in mapper.transport_handoff_snapshot[1]
    assert first.diagnostics.feedback_call_count == 0


def test_prescribed_spacing_cloud_drives_real_transport_without_parent_writes() -> None:
    mapper, first, deposited = _prescribed_deposition_case()
    before = _transport_snapshots(mapper, first)

    report = _transport_prescribed(first, deposited)

    assert validate_passive_frontier_transport_report(report) is report
    transported = materialize_transported_particle_state(report)
    assert transported.positions.shape == deposited.positions.shape
    assert transported.gamma.shape == deposited.gamma.shape
    assert transported.sigma.shape == deposited.sigma.shape
    assert all(
        particle_id[0] == "rvpm-edge-shadow-prescribed-sigma-spacing-v1"
        for particle_id in deposited.particle_ids
    )
    assert report.feedback_call_count == 0
    assert report.parent_write_count == 0
    assert report.load_write_count == 0
    assert _transport_snapshots(mapper, first) == before


def test_legacy_fixed_sigma_none_is_backward_deterministic() -> None:
    sources = (_source(0), _source(1))
    mapper = NodeOwnedDVMRibbonShadow(wing_id="wing", source_family="lev")
    first_events = (
        _manufactured_event(sources[0], active=True, gamma_lev=0.1),
        _manufactured_event(sources[1], active=True, gamma_lev=0.1),
    )
    first = mapper.map_step(_cells(first_events), _nodes(), delta_time_s=DT_PHYSICAL_S)
    assert first.edge_graph is not None
    deposited = deposit_edge_graph_fixed_sigma(
        first.edge_graph,
        smoothing_radius=FIXED_SIGMA_M,
        step=1,
    )
    common = {
        "wing_id": "wing",
        "transport_start_time_s": 0.0,
        "transport_end_time_s": DT_PHYSICAL_S,
        "transport_substeps": 1,
        "freestream_velocity_gp1_m_per_s": (0.1, 0.0, 0.0),
    }

    omitted = transport_passive_node_frontiers(first, deposited, **common)
    explicit_none = transport_passive_node_frontiers(
        first,
        deposited,
        deposition_target_spacing_m=None,
        **common,
    )

    assert omitted == explicit_none
    assert omitted.report_sha256 == explicit_none.report_sha256


def test_fixed_and_prescribed_spacing_schemas_fail_closed_on_keyword_mixing() -> None:
    mapper, first, prescribed = _prescribed_deposition_case()
    assert first.edge_graph is not None
    fixed = deposit_edge_graph_fixed_sigma(
        first.edge_graph,
        smoothing_radius=FIXED_SIGMA_M,
        step=1,
    )
    before = _transport_snapshots(mapper, first)

    with pytest.raises(ValueError, match="prescribed-spacing.*requires"):
        _transport_prescribed(first, prescribed, target_spacing_m=None)
    with pytest.raises(ValueError, match="fixed-sigma.*None"):
        transport_passive_node_frontiers(
            first,
            fixed,
            wing_id="wing",
            transport_start_time_s=0.0,
            transport_end_time_s=DT_PHYSICAL_S,
            transport_substeps=1,
            deposition_target_spacing_m=PRESCRIBED_TARGET_SPACING_M,
        )

    mixed_ids = list(prescribed.particle_ids)
    mixed_ids[0] = (
        "rvpm-edge-shadow-fixed-sigma-v1",
        *mixed_ids[0][1:],
    )
    mixed = replace(prescribed, particle_ids=tuple(mixed_ids))
    with pytest.raises(ValueError, match="mixes particle schemas"):
        _transport_prescribed(first, mixed)
    assert _transport_snapshots(mapper, first) == before

    assert validate_passive_frontier_transport_report(
        _transport_prescribed(first, prescribed)
    )
    fixed_clean = transport_passive_node_frontiers(
        first,
        fixed,
        wing_id="wing",
        transport_start_time_s=0.0,
        transport_end_time_s=DT_PHYSICAL_S,
        transport_substeps=1,
    )
    assert validate_passive_frontier_transport_report(fixed_clean) is fixed_clean
    assert _transport_snapshots(mapper, first) == before


@pytest.mark.parametrize(
    "spacing",
    [0.0, -1.0, float("inf"), float("nan"), True],
)
def test_prescribed_spacing_rejects_invalid_values_then_clean_retry(
    spacing: object,
) -> None:
    mapper, first, deposited = _prescribed_deposition_case()
    before = _transport_snapshots(mapper, first)

    with pytest.raises(ValueError, match="deposition_target_spacing_m"):
        _transport_prescribed(first, deposited, target_spacing_m=spacing)
    assert _transport_snapshots(mapper, first) == before

    clean = _transport_prescribed(first, deposited)
    assert validate_passive_frontier_transport_report(clean) is clean
    assert _transport_snapshots(mapper, first) == before


def test_prescribed_spacing_rejects_underlap_then_clean_retry() -> None:
    mapper, first, deposited = _prescribed_deposition_case()
    before = _transport_snapshots(mapper, first)
    underlapped = np.nextafter(
        FIXED_SIGMA_M / FROZEN_OVERLAP_LAMBDA,
        float("inf"),
    )

    with pytest.raises(ValueError, match="minimum overlap"):
        _transport_prescribed(first, deposited, target_spacing_m=underlapped)
    assert _transport_snapshots(mapper, first) == before

    clean = _transport_prescribed(first, deposited)
    assert validate_passive_frontier_transport_report(clean) is clean
    assert _transport_snapshots(mapper, first) == before


def test_prescribed_wrong_spacing_that_changes_counts_is_rejected() -> None:
    mapper, first, deposited = _prescribed_deposition_case()
    before = _transport_snapshots(mapper, first)
    wrong_spacing = PRESCRIBED_TARGET_SPACING_M / 2.0

    with pytest.raises(ValueError, match="particle count is not reproducible"):
        _transport_prescribed(
            first,
            deposited,
            target_spacing_m=wrong_spacing,
        )
    assert _transport_snapshots(mapper, first) == before

    clean = _transport_prescribed(first, deposited)
    assert validate_passive_frontier_transport_report(clean) is clean
    assert _transport_snapshots(mapper, first) == before


def test_prescribed_one_ulp_spacing_count_change_is_rejected() -> None:
    mapper, first, _ = _prescribed_deposition_case()
    assert first.edge_graph is not None
    selected_edge = max(
        first.edge_graph.retained_edges,
        key=lambda edge: np.linalg.norm(
            np.asarray(edge.end_position) - np.asarray(edge.start_position)
        ),
    )
    edge_length = float(
        np.linalg.norm(
            np.asarray(selected_edge.end_position)
            - np.asarray(selected_edge.start_position)
        )
    )
    minimum_count = int(np.ceil(edge_length * FROZEN_OVERLAP_LAMBDA / FIXED_SIGMA_M))
    boundary_spacing = edge_length / (minimum_count + 1)
    deposited = deposit_edge_graph_prescribed_sigma_and_spacing(
        first.edge_graph,
        smoothing_radius=FIXED_SIGMA_M,
        target_spacing=boundary_spacing,
        step=1,
    )
    one_ulp_smaller = np.nextafter(boundary_spacing, 0.0)
    selected_counts = {
        item.subdivision_count
        for item in deposited.lineage
        if item.source_edge == selected_edge.key
    }
    assert selected_counts == {minimum_count + 1}
    assert int(np.ceil(edge_length / one_ulp_smaller)) != minimum_count + 1
    before = _transport_snapshots(mapper, first)

    with pytest.raises(ValueError, match="particle count is not reproducible"):
        _transport_prescribed(
            first,
            deposited,
            target_spacing_m=one_ulp_smaller,
        )
    assert _transport_snapshots(mapper, first) == before

    clean = _transport_prescribed(
        first,
        deposited,
        target_spacing_m=boundary_spacing,
    )
    assert validate_passive_frontier_transport_report(clean) is clean
    assert _transport_snapshots(mapper, first) == before


def test_prescribed_same_count_spacing_is_observationally_equivalent() -> None:
    mapper, first, deposited = _prescribed_deposition_case()
    assert first.edge_graph is not None
    equivalent_spacing = np.nextafter(
        PRESCRIBED_TARGET_SPACING_M,
        float("inf"),
    )

    def counts(spacing: float) -> tuple[int, ...]:
        return tuple(
            max(
                1,
                int(
                    np.ceil(
                        np.linalg.norm(
                            np.asarray(edge.end_position)
                            - np.asarray(edge.start_position)
                        )
                        / spacing
                    )
                ),
            )
            for edge in first.edge_graph.retained_edges
        )

    assert counts(equivalent_spacing) == counts(PRESCRIBED_TARGET_SPACING_M)
    before = _transport_snapshots(mapper, first)
    report = _transport_prescribed(
        first,
        deposited,
        target_spacing_m=equivalent_spacing,
    )
    assert validate_passive_frontier_transport_report(report) is report
    assert _transport_snapshots(mapper, first) == before


@pytest.mark.parametrize(
    "mutation",
    ["position", "gamma", "sigma", "count", "id", "lineage", "diagnostics"],
)
def test_prescribed_cloud_ledger_attacks_fail_then_clean_retry(
    mutation: str,
) -> None:
    mapper, first, deposited = _prescribed_deposition_case()
    if mutation == "position":
        changed = deposited.positions.copy()
        index = np.unravel_index(np.argmax(np.abs(changed)), changed.shape)
        changed[index] = np.nextafter(changed[index], float("inf"))
        forged = replace(deposited, positions=changed)
    elif mutation == "gamma":
        changed = deposited.gamma.copy()
        index = np.unravel_index(np.argmax(np.abs(changed)), changed.shape)
        changed[index] = np.nextafter(changed[index], float("-inf"))
        forged = replace(deposited, gamma=changed)
    elif mutation == "sigma":
        changed = deposited.sigma.copy()
        changed[0] = np.nextafter(changed[0], float("inf"))
        forged = replace(deposited, sigma=changed)
    elif mutation == "count":
        forged = replace(
            deposited,
            positions=deposited.positions[:-1].copy(),
            gamma=deposited.gamma[:-1].copy(),
            sigma=deposited.sigma[:-1].copy(),
            particle_ids=deposited.particle_ids[:-1],
            lineage=deposited.lineage[:-1],
        )
    elif mutation == "id":
        changed_ids = list(deposited.particle_ids)
        changed_id = list(changed_ids[0])
        changed_id[2] = int(changed_id[2]) + 1_000_000
        changed_ids[0] = tuple(changed_id)
        forged = replace(deposited, particle_ids=tuple(changed_ids))
    elif mutation == "lineage":
        changed_lineage = list(deposited.lineage)
        changed_lineage[0] = replace(
            changed_lineage[0],
            subdivision_count=changed_lineage[0].subdivision_count + 1,
        )
        forged = replace(deposited, lineage=tuple(changed_lineage))
    else:
        forged = replace(
            deposited,
            diagnostics=replace(
                deposited.diagnostics,
                particle_count=deposited.diagnostics.particle_count + 1,
            ),
        )
    before = _transport_snapshots(mapper, first)

    with pytest.raises(ValueError):
        _transport_prescribed(first, forged)
    assert _transport_snapshots(mapper, first) == before

    clean = _transport_prescribed(first, deposited)
    assert validate_passive_frontier_transport_report(clean) is clean
    assert _transport_snapshots(mapper, first) == before


@pytest.mark.parametrize("refinement", [48, 96, 128])
def test_prescribed_high_count_cloud_uses_stable_producer_sums(
    refinement: int,
) -> None:
    _, first, _ = _prescribed_deposition_case()
    assert first.edge_graph is not None
    target_spacing = FIXED_SIGMA_M / FROZEN_OVERLAP_LAMBDA / refinement
    deposited = deposit_edge_graph_prescribed_sigma_and_spacing(
        first.edge_graph,
        smoothing_radius=FIXED_SIGMA_M,
        target_spacing=target_spacing,
        step=1,
    )

    positions, gamma, sigma = _validate_fixed_sigma_deposition(
        first.edge_graph,
        deposited,
        expected_step=1,
        deposition_target_spacing_m=target_spacing,
    )

    assert positions is deposited.positions
    assert gamma is deposited.gamma
    assert sigma is deposited.sigma
    assert deposited.diagnostics.particle_count > 1_000
    assert deposited.diagnostics.max_edge_conservation_abs <= 1.0e-14
    assert deposited.diagnostics.global_conservation_abs <= 1.0e-14


def test_frontier_producer_rejects_legacy_per_edge_sigma_cloud() -> None:
    sources = (_source(0), _source(1))
    mapper = NodeOwnedDVMRibbonShadow(wing_id="wing", source_family="lev")
    first_events = (
        _manufactured_event(sources[0], active=True, gamma_lev=0.1),
        _manufactured_event(sources[1], active=True, gamma_lev=0.1),
    )
    first = mapper.map_step(
        _cells(first_events),
        _nodes(),
        delta_time_s=DT_PHYSICAL_S,
        transport_enabled=True,
    )
    assert first.edge_graph is not None
    legacy = deposit_edge_graph(first.edge_graph, subdivisions=1, step=1)
    with pytest.raises(ValueError, match="uniform fixed sigma|fixed-sigma"):
        transport_passive_node_frontiers(
            first,
            legacy,
            wing_id="wing",
            transport_start_time_s=0.0,
            transport_end_time_s=DT_PHYSICAL_S,
            transport_substeps=1,
        )


@pytest.mark.parametrize("mutation", ["position", "gamma", "particle_id", "lineage"])
def test_frontier_producer_reconstructs_fixed_cloud_ledgers(
    mutation: str,
) -> None:
    sources = (_source(0), _source(1))
    mapper = NodeOwnedDVMRibbonShadow(wing_id="wing", source_family="lev")
    first_events = (
        _manufactured_event(sources[0], active=True, gamma_lev=0.1),
        _manufactured_event(sources[1], active=True, gamma_lev=0.1),
    )
    first = mapper.map_step(_cells(first_events), _nodes(), delta_time_s=DT_PHYSICAL_S)
    assert first.edge_graph is not None
    deposited = deposit_edge_graph_fixed_sigma(
        first.edge_graph, smoothing_radius=FIXED_SIGMA_M, step=1
    )
    if mutation == "position":
        changed = deposited.positions.copy()
        changed[0, 0] += 0.01
        forged = replace(deposited, positions=changed)
    elif mutation == "gamma":
        changed = deposited.gamma.copy()
        changed[0, 1] += 0.01
        forged = replace(deposited, gamma=changed)
    elif mutation == "particle_id":
        ids = list(deposited.particle_ids)
        ids[0] = ("legacy-or-forged",)
        forged = replace(deposited, particle_ids=tuple(ids))
    else:
        lineage = list(deposited.lineage)
        lineage[0] = replace(lineage[0], subdivision_index=99)
        forged = replace(deposited, lineage=tuple(lineage))
    with pytest.raises(
        ValueError,
        match="midpoint|gamma-vector|ID ledger|lineage|particle schemas",
    ):
        transport_passive_node_frontiers(
            first,
            forged,
            wing_id="wing",
            transport_start_time_s=0.0,
            transport_end_time_s=DT_PHYSICAL_S,
            transport_substeps=1,
        )


@pytest.mark.parametrize(
    ("field", "direction"),
    [
        ("position", float("inf")),
        ("position", float("-inf")),
        ("gamma", float("inf")),
        ("gamma", float("-inf")),
    ],
)
def test_frontier_producer_rejects_one_ulp_particle_tampering(
    field: str,
    direction: float,
) -> None:
    sources = (_source(0), _source(1))
    mapper = NodeOwnedDVMRibbonShadow(wing_id="wing", source_family="lev")
    first_events = (
        _manufactured_event(sources[0], active=True, gamma_lev=0.1),
        _manufactured_event(sources[1], active=True, gamma_lev=0.1),
    )
    first = mapper.map_step(_cells(first_events), _nodes(), delta_time_s=DT_PHYSICAL_S)
    assert first.edge_graph is not None
    deposited = deposit_edge_graph_fixed_sigma(
        first.edge_graph, smoothing_radius=FIXED_SIGMA_M, step=1
    )
    before = _transport_snapshots(mapper, first)
    if field == "position":
        changed = deposited.positions.copy()
        index = np.unravel_index(np.argmax(np.abs(changed)), changed.shape)
        changed[index] = np.nextafter(changed[index], direction)
        forged = replace(deposited, positions=changed)
        message = "midpoint ledger"
    else:
        changed = deposited.gamma.copy()
        index = np.unravel_index(np.argmax(np.abs(changed)), changed.shape)
        changed[index] = np.nextafter(changed[index], direction)
        forged = replace(deposited, gamma=changed)
        message = "gamma-vector ledger"
    with pytest.raises(ValueError, match=message):
        transport_passive_node_frontiers(
            first,
            forged,
            wing_id="wing",
            transport_start_time_s=0.0,
            transport_end_time_s=DT_PHYSICAL_S,
            transport_substeps=1,
        )
    assert _transport_snapshots(mapper, first) == before

    clean = transport_passive_node_frontiers(
        first,
        deposited,
        wing_id="wing",
        transport_start_time_s=0.0,
        transport_end_time_s=DT_PHYSICAL_S,
        transport_substeps=1,
    )
    assert validate_passive_frontier_transport_report(clean) is clean
    assert _transport_snapshots(mapper, first) == before


def test_frontier_report_copy_or_missing_duplicate_fact_is_not_attested() -> None:
    sources = (_source(0), _source(1))
    mapper = NodeOwnedDVMRibbonShadow(wing_id="wing", source_family="lev")
    _, report = _first_layer_with_transport_report(mapper, sources)
    with pytest.raises(ValueError, match="directly produced live"):
        validate_passive_frontier_transport_report(replace(report))

    second_events = _next_active_events(sources)
    for facts in (report.facts[:-1], report.facts + (report.facts[0],)):
        forged = replace(report, facts=facts)
        before = _mapper_snapshots(mapper)
        with pytest.raises(ValueError):
            mapper.map_step(
                _cells(second_events),
                _nodes(),
                delta_time_s=DT_PHYSICAL_S,
                transport_enabled=True,
                source_time_s=DT_PHYSICAL_S,
                frontier_transport_report=forged,
            )
        assert _mapper_snapshots(mapper) == before

    accepted = mapper.map_step(
        _cells(second_events),
        _nodes(),
        delta_time_s=DT_PHYSICAL_S,
        transport_enabled=True,
        source_time_s=DT_PHYSICAL_S,
        frontier_transport_report=report,
    )
    assert accepted.diagnostics.transport_advance_count == 3


@pytest.mark.parametrize("attack", ["wing", "time", "parent", "epoch", "node"])
def test_frontier_cross_binding_attacks_fail_transactionally(attack: str) -> None:
    sources = (_source(0), _source(1))
    mapper = NodeOwnedDVMRibbonShadow(wing_id="wing", source_family="lev")
    first, report = _first_layer_with_transport_report(
        mapper,
        sources,
        start_time_s=(DT_PHYSICAL_S if attack == "time" else 0.0),
        end_time_s=(2.0 * DT_PHYSICAL_S if attack == "time" else DT_PHYSICAL_S),
    )
    if attack == "wing":
        assert first.edge_graph is not None
        deposited = deposit_edge_graph_fixed_sigma(
            first.edge_graph, smoothing_radius=FIXED_SIGMA_M, step=1
        )
        candidate = transport_passive_node_frontiers(
            first,
            deposited,
            wing_id="other-wing",
            transport_start_time_s=0.0,
            transport_end_time_s=DT_PHYSICAL_S,
            transport_substeps=1,
        )
    elif attack == "time":
        candidate = report
    else:
        fact = report.facts[0]
        if attack == "parent":
            changed_fact = replace(fact, parent_frontier_digest_sha256="0" * 64)
        elif attack == "epoch":
            changed_fact = replace(fact, lineage_epoch=fact.lineage_epoch + 1)
        else:
            changed_fact = replace(fact, node_id=999)
        candidate = replace(report, facts=(changed_fact,) + report.facts[1:])

    second_events = _next_active_events(sources)
    before = _mapper_snapshots(mapper)
    with pytest.raises(ValueError):
        mapper.map_step(
            _cells(second_events),
            _nodes(),
            delta_time_s=DT_PHYSICAL_S,
            transport_enabled=True,
            source_time_s=DT_PHYSICAL_S,
            frontier_transport_report=candidate,
        )
    assert _mapper_snapshots(mapper) == before


def test_frontier_cross_source_family_report_is_rejected_transactionally() -> None:
    lev_sources = (_source(0), _source(1))
    lev_mapper = NodeOwnedDVMRibbonShadow(wing_id="wing", source_family="lev")
    _, lev_report = _first_layer_with_transport_report(lev_mapper, lev_sources)

    tev_sources = (_source(0), _source(1))
    tev_mapper = NodeOwnedDVMRibbonShadow(wing_id="wing", source_family="tev_persisted")
    first_tev = (
        _manufactured_event(tev_sources[0], active=False, gamma_lev=0.0),
        _manufactured_event(tev_sources[1], active=False, gamma_lev=0.0),
    )
    tev_mapper.map_step(
        _cells(first_tev),
        _nodes(),
        delta_time_s=DT_PHYSICAL_S,
        transport_enabled=True,
    )
    second_tev = (
        _manufactured_event(tev_sources[0], active=False, gamma_lev=0.0),
        _manufactured_event(tev_sources[1], active=False, gamma_lev=0.0),
    )
    before = _mapper_snapshots(tev_mapper)
    with pytest.raises(ValueError, match="source family"):
        tev_mapper.map_step(
            _cells(second_tev),
            _nodes(),
            delta_time_s=DT_PHYSICAL_S,
            transport_enabled=True,
            source_time_s=DT_PHYSICAL_S,
            frontier_transport_report=lev_report,
        )
    assert _mapper_snapshots(tev_mapper) == before


def test_frontier_report_replay_fails_before_a_third_release_commit() -> None:
    sources = (_source(0), _source(1))
    mapper = NodeOwnedDVMRibbonShadow(wing_id="wing", source_family="lev")
    _, report = _first_layer_with_transport_report(mapper, sources)
    mapper.map_step(
        _cells(_next_active_events(sources)),
        _nodes(),
        delta_time_s=DT_PHYSICAL_S,
        transport_enabled=True,
        source_time_s=DT_PHYSICAL_S,
        frontier_transport_report=report,
    )
    third_events = _next_active_events(sources)
    before = _mapper_snapshots(mapper)
    with pytest.raises(ValueError, match="replay"):
        mapper.map_step(
            _cells(third_events),
            _nodes(),
            delta_time_s=DT_PHYSICAL_S,
            transport_enabled=True,
            source_time_s=2.0 * DT_PHYSICAL_S,
            frontier_transport_report=report,
        )
    assert _mapper_snapshots(mapper) == before


def test_first_and_restart_nodes_consume_no_frontier_fact() -> None:
    sources = (_source(0), _source(1))
    mapper = NodeOwnedDVMRibbonShadow(wing_id="wing", source_family="lev")
    first = mapper.map_step(
        _cells(_next_active_events(sources)),
        _nodes(),
        delta_time_s=DT_PHYSICAL_S,
        transport_enabled=True,
    )
    assert first.diagnostics.first_node_count == 3
    assert first.diagnostics.transport_advance_count == 0
    inactive_events = (
        _manufactured_event(sources[0], active=False, gamma_lev=0.0),
        _manufactured_event(sources[1], active=False, gamma_lev=0.0),
    )
    inactive = mapper.map_step(
        _cells(inactive_events),
        _nodes(),
        delta_time_s=DT_PHYSICAL_S,
        transport_enabled=True,
    )
    assert inactive.diagnostics.transport_advance_count == 0
    restarted = mapper.map_step(
        _cells(_next_active_events(sources)),
        _nodes(),
        delta_time_s=DT_PHYSICAL_S,
        transport_enabled=True,
    )
    assert restarted.diagnostics.restart_node_count == 3
    assert restarted.diagnostics.transport_advance_count == 0


def test_inactive_extra_parent_fact_is_validated_but_not_consumed() -> None:
    sources = (_source(0), _source(1))
    mapper = NodeOwnedDVMRibbonShadow(wing_id="wing", source_family="lev")
    _, report = _first_layer_with_transport_report(mapper, sources)
    second_events = (
        _manufactured_event(sources[0], active=True, gamma_lev=0.1),
        _manufactured_event(sources[1], active=False, gamma_lev=0.0),
    )
    second = mapper.map_step(
        _cells(second_events),
        _nodes(),
        delta_time_s=DT_PHYSICAL_S,
        transport_enabled=True,
        source_time_s=DT_PHYSICAL_S,
        frontier_transport_report=report,
    )
    assert len(report.facts) == 3
    assert second.diagnostics.continuous_node_count == 2
    assert second.diagnostics.inactive_node_count == 1
    assert second.diagnostics.transport_advance_count == 2


def test_v1_transport_report_is_explicitly_single_release_only() -> None:
    sources = (_source(0), _source(1))
    mapper = NodeOwnedDVMRibbonShadow(wing_id="wing", source_family="lev")
    _, report = _first_layer_with_transport_report(mapper, sources)
    second = mapper.map_step(
        _cells(_next_active_events(sources)),
        _nodes(),
        delta_time_s=DT_PHYSICAL_S,
        transport_enabled=True,
        source_time_s=DT_PHYSICAL_S,
        frontier_transport_report=report,
    )
    assert second.edge_graph is not None
    second_deposit = deposit_edge_graph_fixed_sigma(
        second.edge_graph,
        smoothing_radius=FIXED_SIGMA_M,
        step=2,
    )
    with pytest.raises(ValueError, match="cumulative-cloud v2"):
        transport_passive_node_frontiers(
            second,
            second_deposit,
            wing_id="wing",
            transport_start_time_s=DT_PHYSICAL_S,
            transport_end_time_s=2.0 * DT_PHYSICAL_S,
            transport_substeps=1,
        )


def test_copied_step2_with_zeroed_transport_count_cannot_bypass_v1() -> None:
    sources = (_source(0), _source(1))
    mapper = NodeOwnedDVMRibbonShadow(wing_id="wing", source_family="lev")
    _, report = _first_layer_with_transport_report(mapper, sources)
    second = mapper.map_step(
        _cells(_next_active_events(sources)),
        _nodes(),
        delta_time_s=DT_PHYSICAL_S,
        transport_enabled=True,
        source_time_s=DT_PHYSICAL_S,
        frontier_transport_report=report,
    )
    assert second.edge_graph is not None
    forged = replace(
        second,
        diagnostics=replace(second.diagnostics, transport_advance_count=0),
    )
    deposited = deposit_edge_graph_fixed_sigma(
        second.edge_graph,
        smoothing_radius=FIXED_SIGMA_M,
        step=2,
    )
    before = _mapper_snapshots(mapper)
    with pytest.raises(ValueError, match="directly produced live mapper"):
        transport_passive_node_frontiers(
            forged,
            deposited,
            wing_id="wing",
            transport_start_time_s=DT_PHYSICAL_S,
            transport_end_time_s=2.0 * DT_PHYSICAL_S,
            transport_substeps=1,
        )
    assert _mapper_snapshots(mapper) == before

    with pytest.raises(ValueError, match="source-step-1 parent"):
        transport_passive_node_frontiers(
            second,
            deposited,
            wing_id="wing",
            transport_start_time_s=DT_PHYSICAL_S,
            transport_end_time_s=2.0 * DT_PHYSICAL_S,
            transport_substeps=1,
        )
    assert _mapper_snapshots(mapper) == before


@pytest.mark.parametrize("forged_value", [float("nan"), -1.0e-300])
def test_fixed_cloud_forged_max_edge_diagnostics_fail_then_clean_retry(
    forged_value: float,
) -> None:
    sources = (_source(0), _source(1))
    mapper = NodeOwnedDVMRibbonShadow(wing_id="wing", source_family="lev")
    first_events = (
        _manufactured_event(sources[0], active=True, gamma_lev=0.1),
        _manufactured_event(sources[1], active=True, gamma_lev=0.1),
    )
    first = mapper.map_step(_cells(first_events), _nodes(), delta_time_s=DT_PHYSICAL_S)
    assert first.edge_graph is not None
    deposited = deposit_edge_graph_fixed_sigma(
        first.edge_graph,
        smoothing_radius=FIXED_SIGMA_M,
        step=1,
    )
    forged = replace(
        deposited,
        diagnostics=replace(
            deposited.diagnostics,
            max_edge_conservation_abs=forged_value,
            max_edge_conservation_rel=forged_value,
        ),
    )
    before = _mapper_snapshots(mapper)
    with pytest.raises(ValueError, match="diagnostics are not reproducible"):
        transport_passive_node_frontiers(
            first,
            forged,
            wing_id="wing",
            transport_start_time_s=0.0,
            transport_end_time_s=DT_PHYSICAL_S,
            transport_substeps=1,
        )
    assert _mapper_snapshots(mapper) == before

    clean = transport_passive_node_frontiers(
        first,
        deposited,
        wing_id="wing",
        transport_start_time_s=0.0,
        transport_end_time_s=DT_PHYSICAL_S,
        transport_substeps=1,
    )
    assert validate_passive_frontier_transport_report(clean) is clean
    assert _mapper_snapshots(mapper) == before


def test_frontier_public_factory_has_no_claimed_output_position_or_subset() -> None:
    parameters = signature(transport_passive_node_frontiers).parameters
    assert "advected_position_gp1_m" not in parameters
    assert "position" not in parameters
    assert "node_ids" not in parameters
    assert {"parent_ribbon", "deposited_shadow"}.issubset(parameters)
    assert parameters["deposition_target_spacing_m"].default is None


def test_shared_node_modes_grow_deactivate_and_restart_without_a_seam() -> None:
    sources = (_source(0), _source(1))
    mapper = NodeOwnedDVMRibbonShadow(wing_id="wing", source_family="lev")

    first = mapper.map_step(
        _cells(
            (
                _manufactured_event(sources[0], active=True, gamma_lev=0.1),
                _manufactured_event(sources[1], active=False, gamma_lev=0.0),
            )
        ),
        _nodes(),
        delta_time_s=DT_PHYSICAL_S,
    )
    first_births = _births_by_id(first)
    assert [first_births[index].mode for index in range(3)] == [
        "first",
        "first",
        "inactive",
    ]
    assert first.diagnostics.active_cell_count == 1

    grown = mapper.map_step(
        _cells(
            (
                _manufactured_event(sources[0], active=True, gamma_lev=0.1),
                _manufactured_event(sources[1], active=True, gamma_lev=0.1),
            )
        ),
        _nodes(),
        delta_time_s=DT_PHYSICAL_S,
    )
    grown_births = _births_by_id(grown)
    assert [grown_births[index].mode for index in range(3)] == [
        "continuous",
        "continuous",
        "first",
    ]
    assert grown.edge_graph is not None
    shared_key = canonical_edge_key(
        grown_births[1].birth_node_id,
        grown_births[1].anchor_node_id,
    )
    shared = {edge.key: edge for edge in grown.edge_graph.edges}[shared_key]
    assert len(shared.incidences) == 2
    grown_gamma = {
        item.cell_id: item.ring_circulation_m2_per_s for item in grown.cell_ledgers
    }
    assert abs(shared.circulation) == pytest.approx(
        abs(grown_gamma["cell-1"] - grown_gamma["cell-0"])
    )
    assert grown.diagnostics.seam_count == 0

    inactive = mapper.map_step(
        _cells(
            (
                _manufactured_event(sources[0], active=False, gamma_lev=0.0),
                _manufactured_event(sources[1], active=False, gamma_lev=0.0),
            )
        ),
        _nodes(),
        delta_time_s=DT_PHYSICAL_S,
    )
    assert inactive.edge_graph is None
    assert all(item.mode == "inactive" for item in inactive.node_births)

    restarted = mapper.map_step(
        _cells(
            (
                _manufactured_event(sources[0], active=False, gamma_lev=0.0),
                _manufactured_event(sources[1], active=True, gamma_lev=0.1),
            )
        ),
        _nodes(),
        delta_time_s=DT_PHYSICAL_S,
    )
    restart_births = _births_by_id(restarted)
    assert [restart_births[index].mode for index in range(3)] == [
        "inactive",
        "restart",
        "restart",
    ]
    assert restart_births[1].lineage_epoch == 1
    assert restart_births[2].lineage_epoch == 1


def test_units_sign_and_full_shared_edge_incidence_are_exact() -> None:
    sources = (_source(0), _source(1))
    mapper = NodeOwnedDVMRibbonShadow(wing_id="wing", source_family="lev")
    result = mapper.map_step(
        _cells(
            (
                _manufactured_event(sources[0], active=True, gamma_lev=0.1),
                _manufactured_event(sources[1], active=True, gamma_lev=0.2),
            )
        ),
        _nodes(),
        delta_time_s=DT_PHYSICAL_S,
    )
    ledgers = {item.cell_id: item for item in result.cell_ledgers}
    expected_0 = result.cell_ledgers[0].gamma_star * 0.75
    expected_1 = result.cell_ledgers[1].gamma_star * 0.75
    assert ledgers["cell-0"].gamma_cell_m2_per_s == pytest.approx(expected_0)
    assert ledgers["cell-1"].gamma_cell_m2_per_s == pytest.approx(expected_1)
    assert ledgers["cell-0"].ring_circulation_m2_per_s == pytest.approx(expected_0)
    births = _births_by_id(result)
    assert result.edge_graph is not None
    shared_key = canonical_edge_key(
        births[1].birth_node_id,
        births[1].anchor_node_id,
    )
    shared = {edge.key: edge for edge in result.edge_graph.edges}[shared_key]
    assert abs(shared.circulation) == pytest.approx(abs(expected_1 - expected_0))
    assert len(shared.incidences) == 2
    assert all(
        item.section_birth_point_used_for_topology is False
        for item in result.cell_ledgers
    )
    assert ledgers["cell-0"].section_birth_audit_point_gp1_m[0] > 9.0
    assert births[0].birth_position_gp1_m[0] < 0.01


def test_input_order_does_not_change_global_edge_identity_or_ledger() -> None:
    forward_sources = (_source(0), _source(1))
    reverse_sources = (_source(0), _source(1))
    forward_events = (
        _manufactured_event(forward_sources[0], active=True, gamma_lev=0.1),
        _manufactured_event(forward_sources[1], active=True, gamma_lev=0.2),
    )
    reverse_events = (
        _manufactured_event(reverse_sources[0], active=True, gamma_lev=0.1),
        _manufactured_event(reverse_sources[1], active=True, gamma_lev=0.2),
    )
    assert tuple(item.producer_manifest_sha256 for item in forward_events) == tuple(
        item.producer_manifest_sha256 for item in reverse_events
    )
    assert all(
        left is not right
        for left, right in zip(forward_events, reverse_events, strict=True)
    )
    forward = NodeOwnedDVMRibbonShadow(wing_id="wing", source_family="lev").map_step(
        _cells(forward_events), _nodes(), delta_time_s=DT_PHYSICAL_S
    )
    reverse = NodeOwnedDVMRibbonShadow(wing_id="wing", source_family="lev").map_step(
        tuple(reversed(_cells(reverse_events))),
        tuple(reversed(_nodes())),
        delta_time_s=DT_PHYSICAL_S,
    )
    assert forward.edge_graph == reverse.edge_graph
    assert forward.node_births == reverse.node_births
    assert forward.cell_ledgers == reverse.cell_ledgers


def test_tev_persisted_family_keeps_first_zero_source_and_then_continues() -> None:
    source = _source(0)
    mapper = NodeOwnedDVMRibbonShadow(wing_id="wing", source_family="tev_persisted")
    first_event = _manufactured_event(source, active=False, gamma_lev=0.0)
    single_nodes = _nodes()[:2]
    first = mapper.map_step(
        (DVMSpanCellSource("cell", 0, 1, first_event, _mapping(0)),),
        single_nodes,
        delta_time_s=DT_PHYSICAL_S,
    )
    assert first.edge_graph is not None
    assert not first.edge_graph.retained_edges
    assert all(item.mode == "first" for item in first.node_births)

    second_event = _manufactured_event(source, active=False, gamma_lev=0.0)
    second = mapper.map_step(
        (DVMSpanCellSource("cell", 0, 1, second_event, _mapping(0)),),
        single_nodes,
        delta_time_s=DT_PHYSICAL_S,
    )
    assert second.edge_graph is not None
    assert second.edge_graph.retained_edges
    assert all(item.mode == "continuous" for item in second.node_births)


def test_fail_closed_frame_time_reuse_and_kelvin_do_not_mutate_mapper() -> None:
    sources = (_source(0), _source(1))
    mapper = NodeOwnedDVMRibbonShadow(wing_id="wing", source_family="lev")
    events = (
        _manufactured_event(sources[0], active=True, gamma_lev=0.1),
        _manufactured_event(sources[1], active=True, gamma_lev=0.1),
    )
    bad_sign_cells = _cells(events, signs=(-1, 1))
    before = mapper.state_snapshot
    with pytest.raises(ValueError, match="sign disagrees"):
        mapper.map_step(bad_sign_cells, _nodes(), delta_time_s=DT_PHYSICAL_S)
    assert mapper.state_snapshot == before

    with pytest.raises(ValueError, match="time steps disagree"):
        mapper.map_step(_cells(events), _nodes(), delta_time_s=2.0 * DT_PHYSICAL_S)
    assert mapper.state_snapshot == before

    good = mapper.map_step(_cells(events), _nodes(), delta_time_s=DT_PHYSICAL_S)
    committed = mapper.state_snapshot
    assert good.edge_graph is not None
    with pytest.raises(ValueError, match="consumed more than once"):
        mapper.map_step(_cells(events), _nodes(), delta_time_s=DT_PHYSICAL_S)
    assert mapper.state_snapshot == committed

    next_events = (
        _manufactured_event(sources[0], active=True, gamma_lev=0.1),
        _manufactured_event(sources[1], active=True, gamma_lev=0.1),
    )
    assert next_events[0].kelvin_ledger is not None
    forged_ledger = replace(
        next_events[0].kelvin_ledger,
        gamma_bound_post=next_events[0].kelvin_ledger.gamma_bound_post + 1.0,
    )
    forged = replace(next_events[0], kelvin_ledger=forged_ledger)
    with pytest.raises(ValueError, match="Kelvin solve ledger"):
        mapper.map_step(
            _cells((forged, next_events[1])),
            _nodes(),
            delta_time_s=DT_PHYSICAL_S,
        )
    assert mapper.state_snapshot == committed


def test_counterexample_active_lesp_residual_cannot_be_large_but_self_consistent() -> (
    None
):
    event = _manufactured_event(_source(0), active=True, gamma_lev=0.1)
    forged = replace(
        event,
        a0_post=0.17,
        lesp_signed_target=0.18,
        lesp_constraint_residual=-0.01,
    )
    with pytest.raises(ValueError, match="active LESP constraint residual"):
        _map_single_event(forged)


def test_counterexample_event_and_provenance_lesp_critical_cannot_diverge() -> None:
    event = _manufactured_event(_source(0), active=True, gamma_lev=0.1)
    with pytest.raises(ValueError, match="LESP critical values disagree"):
        _map_single_event(replace(event, lesp_critical=0.19))


@pytest.mark.parametrize("field", ["units", "source_parity", "interface"])
def test_counterexample_pinned_producer_contract_cannot_be_forged(field: str) -> None:
    event = _manufactured_event(_source(0), active=True, gamma_lev=0.1)
    assert event.kelvin_ledger is not None
    if field == "units":
        provenance = replace(event.provenance, circulation_units="bananas")
        ledger = replace(event.kelvin_ledger, circulation_units="bananas")
        forged = replace(event, provenance=provenance, kelvin_ledger=ledger)
    elif field == "source_parity":
        forged = replace(
            event, provenance=replace(event.provenance, source_parity=False)
        )
    else:
        forged = replace(
            event, provenance=replace(event.provenance, interface_id="forged")
        )
    with pytest.raises(ValueError, match="pinned contract|source_parity"):
        _map_single_event(forged)


@pytest.mark.parametrize("identity", ["section", "strip"])
def test_counterexample_lineage_physical_ids_must_match_provenance(
    identity: str,
) -> None:
    event = _manufactured_event(_source(0), active=True, gamma_lev=0.1)
    if identity == "section":
        lineage = replace(event.lineage, physical_section_id="forged-section")
    else:
        lineage = replace(event.lineage, physical_strip_id="forged-strip")
    with pytest.raises(ValueError, match=f"physical_{identity}_id disagrees"):
        _map_single_event(replace(event, lineage=lineage))


def test_counterexample_fractional_source_step_is_not_silently_truncated() -> None:
    event = _manufactured_event(_source(0), active=True, gamma_lev=0.1)
    forged = replace(event, lineage=replace(event.lineage, source_step_index=1.5))
    with pytest.raises(ValueError, match="source_step_index must be an integer"):
        _map_single_event(forged)


def test_counterexample_first_tev_zeroing_flag_is_lineage_constrained() -> None:
    event = _manufactured_event(_source(0), active=True, gamma_lev=0.1)
    assert event.kelvin_ledger is not None
    forged = replace(
        event,
        kelvin_ledger=replace(event.kelvin_ledger, first_tev_zeroed=False),
    )
    with pytest.raises(ValueError, match="first-TEV persistence flag"):
        _map_single_event(forged)


def test_counterexample_relabelled_duplicate_event_is_not_a_new_strip_source() -> None:
    event = _manufactured_event(_source(0), active=True, gamma_lev=0.1)
    forged_provenance = replace(
        event.provenance,
        physical_section_id="forged:section:1",
        physical_strip_id="forged:strip:1",
    )
    forged_lineage = replace(
        event.lineage,
        physical_section_id="forged:section:1",
        physical_strip_id="forged:strip:1",
        section_lineage_id="dvm-section-forged000000000",
        newborn_tev_source_id="dvm-section-forged000000000:step:1:tev-newborn",
        newborn_lev_source_id="dvm-section-forged000000000:step:1:lev-newborn",
    )
    forged = replace(event, provenance=forged_provenance, lineage=forged_lineage)
    cells = (
        DVMSpanCellSource("cell-0", 0, 1, event, _mapping(0)),
        DVMSpanCellSource("cell-1", 1, 2, forged, _mapping(1)),
    )
    with pytest.raises(ValueError, match="section_lineage_id|directly produced"):
        NodeOwnedDVMRibbonShadow(wing_id="wing", source_family="lev").map_step(
            cells, _nodes(), delta_time_s=DT_PHYSICAL_S
        )


@pytest.mark.parametrize("mutation", ["strip", "endpoints"])
def test_counterexample_first_success_cell_binding_is_frozen(mutation: str) -> None:
    sources = (_source(0), _source(1))
    mapper = NodeOwnedDVMRibbonShadow(wing_id="wing", source_family="lev")
    first_events = (
        _manufactured_event(sources[0], active=True, gamma_lev=0.1),
        _manufactured_event(sources[1], active=True, gamma_lev=0.1),
    )
    mapper.map_step(_cells(first_events), _nodes(), delta_time_s=DT_PHYSICAL_S)
    before_state = mapper.state_snapshot
    before_bindings = mapper.cell_binding_snapshot
    second_events = (
        _manufactured_event(sources[0], active=True, gamma_lev=0.1),
        _manufactured_event(sources[1], active=True, gamma_lev=0.1),
    )
    if mutation == "strip":
        changed_cells = _cells(tuple(reversed(second_events)))
    else:
        changed_cells = (
            DVMSpanCellSource("cell-0", 1, 0, second_events[0], _mapping(0)),
            DVMSpanCellSource("cell-1", 1, 2, second_events[1], _mapping(1)),
        )
    with pytest.raises(ValueError, match="binding changed"):
        mapper.map_step(changed_cells, _nodes(), delta_time_s=DT_PHYSICAL_S)
    assert mapper.state_snapshot == before_state
    assert mapper.cell_binding_snapshot == before_bindings

    # The rejected layer was not consumed or committed.
    accepted = mapper.map_step(
        _cells(second_events), _nodes(), delta_time_s=DT_PHYSICAL_S
    )
    assert accepted.diagnostics.source_step_index == 2


def test_counterexample_cross_producer_history_splice_fails_and_can_retry() -> None:
    source_a = _source(0)
    source_b = _source(0)
    mapper = NodeOwnedDVMRibbonShadow(wing_id="wing", source_family="lev")
    nodes = _nodes()[:2]

    first_a = _manufactured_event(source_a, active=True, gamma_lev=0.2)
    mapper.map_step(
        (DVMSpanCellSource("cell", 0, 1, first_a, _mapping(0)),),
        nodes,
        delta_time_s=DT_PHYSICAL_S,
    )
    before_state = mapper.state_snapshot
    before_bindings = mapper.cell_binding_snapshot
    before_history = mapper.cell_history_snapshot

    # B has the same static provenance and physical IDs, but a different first
    # history.  Its genuine step-2 event must not be spliced after A's step 1.
    _manufactured_event(source_b, active=False, gamma_lev=0.0)
    second_b = _manufactured_event(source_b, active=True, gamma_lev=0.2)
    with pytest.raises(
        ValueError,
        match="parent-event digest|history is discontinuous",
    ):
        mapper.map_step(
            (DVMSpanCellSource("cell", 0, 1, second_b, _mapping(0)),),
            nodes,
            delta_time_s=DT_PHYSICAL_S,
        )
    assert mapper.state_snapshot == before_state
    assert mapper.cell_binding_snapshot == before_bindings
    assert mapper.cell_history_snapshot == before_history

    second_a = _manufactured_event(source_a, active=True, gamma_lev=0.2)
    accepted = mapper.map_step(
        (DVMSpanCellSource("cell", 0, 1, second_a, _mapping(0)),),
        nodes,
        delta_time_s=DT_PHYSICAL_S,
    )
    assert accepted.diagnostics.source_step_index == 2
    assert mapper.cell_history_snapshot != before_history


def test_public_schema_has_no_aerodynamic_result_channel() -> None:
    forbidden = {
        "cl",
        "cd",
        "cn",
        "cs",
        "lift",
        "drag",
        "force",
        "load",
        "moment",
        "suction",
        "impulse",
        "correction",
    }
    names = {field.name.casefold() for field in fields(DVMRibbonShadowResult)}
    assert names.isdisjoint(forbidden)
