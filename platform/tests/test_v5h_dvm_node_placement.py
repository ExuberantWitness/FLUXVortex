from __future__ import annotations

from dataclasses import replace
import os
from typing import Any, Iterable

os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/fluxv-v5h-dvm-numba-cache")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/fluxv-v5h-dvm-mpl-cache")

import numpy as np
import pytest

from forward_flight_benchmarks.ldvm_uvlm_correction import (
    LDVMSectionSettings,
    LESPThreshold,
)
from forward_flight_benchmarks.v5h_dvm_node_placement import (
    DISABLED_STATUS,
    DVMNodePlacementCell,
    EXCLUSIVE_STRENGTH_OWNER,
    EXCLUSIVE_SURFACE_OWNER,
    GP1NodeSectionFact,
    NODE_GEOMETRY_ROLE,
    NodeLocalDVMPlacementAdapter,
    validate_live_dvm_node_placement_result,
)
from forward_flight_benchmarks.v5h_dvm_source import DVMSourceEvent, V5hDVMSource


CHORD_M = 0.25
SPEED_M_PER_S = 3.0
DT_STAR = 0.02
DT_S = DT_STAR * CHORD_M / SPEED_M_PER_S
X_AXIS = (1.0, 0.0, 0.0)
Z_AXIS = (0.0, 0.0, -1.0)
SPAN_AXIS = (0.0, 1.0, 0.0)


def _threshold() -> LESPThreshold:
    return LESPThreshold(
        value=0.18,
        section_family="generic 2-percent thin flat plate",
        reynolds=30_000.0,
        source="Ramesh LDVM v2.5 published source input",
        source_role="published_model_parameter",
    )


def _source(role: str, index: int) -> V5hDVMSource:
    return V5hDVMSource(
        physical_section_id=f"wing:{role}:section:{index}",
        physical_strip_id=f"wing:{role}:strip:{index}",
        geometry_identity="explicit zero-camber flat-plate surrogate",
        reference_speed_m_per_s=SPEED_M_PER_S,
        reference_chord_m=CHORD_M,
        zero_camber_surrogate=True,
        delta_time_convective=DT_STAR,
        pivot_fraction_chord=0.25,
        threshold=_threshold(),
        settings=LDVMSectionSettings(ndiv=24, naterm=8, max_wake_steps=48),
    )


class _Harness:
    def __init__(self, *, node_count: int = 3, cell_count: int | None = None) -> None:
        self.node_sources = tuple(_source("node", index) for index in range(node_count))
        count = node_count - 1 if cell_count is None else cell_count
        self.cell_sources = tuple(_source("cell", index) for index in range(count))

    def layer(
        self,
        *,
        node_angles_deg: float | Iterable[float],
        cell_angles_deg: float | Iterable[float],
        source_time_s: float,
        wing_id: str = "wing-left",
        node_ids: tuple[str, ...] | None = None,
        patches: tuple[str, ...] | None = None,
    ) -> tuple[tuple[GP1NodeSectionFact, ...], tuple[DVMSourceEvent, ...]]:
        node_angles = _expanded(node_angles_deg, len(self.node_sources))
        cell_angles = _expanded(cell_angles_deg, len(self.cell_sources))
        node_events = tuple(
            source.step(np.deg2rad(angle), 0.0, 0.0)
            for source, angle in zip(self.node_sources, node_angles, strict=True)
        )
        cell_events = tuple(
            source.step(np.deg2rad(angle), 0.0, 0.0)
            for source, angle in zip(self.cell_sources, cell_angles, strict=True)
        )
        ids = node_ids or tuple(f"node-{index}" for index in range(len(node_events)))
        patch_values = patches or tuple("main-patch" for _ in node_events)
        facts = tuple(
            _fact(
                node_id=ids[index],
                span_coordinate_m=float(index),
                event=event,
                source_time_s=source_time_s,
                wing_id=wing_id,
                patch_id=patch_values[index],
            )
            for index, event in enumerate(node_events)
        )
        return facts, cell_events


def _expanded(value: float | Iterable[float], count: int) -> tuple[float, ...]:
    if isinstance(value, (int, float)):
        return tuple(float(value) for _ in range(count))
    result = tuple(float(item) for item in value)
    assert len(result) == count
    return result


def _fact(
    *,
    node_id: str,
    span_coordinate_m: float,
    event: DVMSourceEvent,
    source_time_s: float,
    wing_id: str = "wing-left",
    patch_id: str = "main-patch",
    frame_id: str = "ptera-gp1-wing-left",
    x_axis: tuple[float, float, float] = X_AXIS,
    z_axis: tuple[float, float, float] = Z_AXIS,
    span_axis: tuple[float, float, float] = SPAN_AXIS,
) -> GP1NodeSectionFact:
    lev_2d = np.asarray(
        event.lev_placement.edge_anchor_position_over_chord_backend_world,
        dtype=float,
    )
    tev_2d = np.asarray(
        event.tev_placement.edge_anchor_position_over_chord_backend_world,
        dtype=float,
    )
    lev_gp1 = np.asarray((0.0, span_coordinate_m, 0.0), dtype=float)
    tev_gp1 = lev_gp1 + CHORD_M * (
        (tev_2d[0] - lev_2d[0]) * np.asarray(x_axis)
        + (tev_2d[1] - lev_2d[1]) * np.asarray(z_axis)
    )
    return GP1NodeSectionFact(
        wing_id=wing_id,
        node_id=node_id,
        source_step_index=int(event.lineage.source_step_index),
        source_time_s=source_time_s,
        event=event,
        lev_edge_anchor_gp1_m=tuple(float(item) for item in lev_gp1),
        tev_edge_anchor_gp1_m=tuple(float(item) for item in tev_gp1),
        reference_chord_m=CHORD_M,
        reference_speed_m_per_s=SPEED_M_PER_S,
        dvm_x_axis_gp1=x_axis,
        dvm_z_axis_gp1=z_axis,
        positive_span_axis_gp1=span_axis,
        topology_patch_id=patch_id,
        coordinate_frame_id=frame_id,
        node_lineage_id=event.lineage.section_lineage_id,
        geometry_token=event.provenance.geometry_hash_sha256,
    )


def _cells(
    facts: tuple[GP1NodeSectionFact, ...],
    events: tuple[DVMSourceEvent, ...],
) -> tuple[DVMNodePlacementCell, ...]:
    assert len(events) <= len(facts) - 1
    return tuple(
        DVMNodePlacementCell(
            cell_id=f"cell-{index}",
            left_node_fact=facts[index],
            right_node_fact=facts[index + 1],
            cell_source_event=event,
        )
        for index, event in enumerate(events)
    )


def _shift_fact_to_span_coordinate(
    fact: GP1NodeSectionFact, span_coordinate_m: float
) -> GP1NodeSectionFact:
    current = float(np.asarray(fact.lev_edge_anchor_gp1_m)[1])
    shift = np.asarray((0.0, span_coordinate_m - current, 0.0), dtype=float)
    lev = np.asarray(fact.lev_edge_anchor_gp1_m, dtype=float) + shift
    tev = np.asarray(fact.tev_edge_anchor_gp1_m, dtype=float) + shift
    return replace(
        fact,
        lev_edge_anchor_gp1_m=tuple(float(item) for item in lev),
        tev_edge_anchor_gp1_m=tuple(float(item) for item in tev),
    )


def _all_keys(value: Any) -> list[str]:
    keys: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            keys.append(str(key))
            keys.extend(_all_keys(item))
    elif isinstance(value, (list, tuple)):
        for item in value:
            keys.extend(_all_keys(item))
    return keys


class _Bomb:
    def __iter__(self) -> Any:
        raise AssertionError("disabled adapter iterated cells")

    def __float__(self) -> float:
        raise AssertionError("disabled adapter inspected delta_time_s")


def test_first_relative_birth_maps_to_gp1_and_exports_no_strength_or_load() -> None:
    harness = _Harness()
    facts, events = harness.layer(
        node_angles_deg=35.0,
        cell_angles_deg=35.0,
        source_time_s=0.0,
    )
    result = NodeLocalDVMPlacementAdapter(wing_id="wing-left").map_step(
        _cells(facts, events), delta_time_s=DT_S
    )
    assert validate_live_dvm_node_placement_result(result) is result
    assert result.canonical is False and result.feedback_call_count == 0
    assert result.feedback_velocity is None
    assert result.exclusive_strength_owner == EXCLUSIVE_STRENGTH_OWNER
    assert result.exclusive_surface_owner == EXCLUSIVE_SURFACE_OWNER
    assert all(
        ledger.active and ledger.placement_mode == "first"
        for ledger in result.node_ledgers
    )
    assert all(
        ledger.topology_owner == "node_local_dvm_relative_birth"
        and ledger.edge_velocity_used_by_ribbon
        and not ledger.dvm_absolute_birth_used_for_topology
        for ledger in result.node_ledgers
    )
    for fact, ledger, kinematics in zip(
        facts, result.node_ledgers, result.kinematics, strict=True
    ):
        displacement = np.asarray(
            fact.event.lev_placement.birth_displacement_from_edge_over_chord_backend_world
        )
        expected_birth = np.asarray(fact.lev_edge_anchor_gp1_m) + CHORD_M * (
            displacement[0] * np.asarray(X_AXIS) + displacement[1] * np.asarray(Z_AXIS)
        )
        np.testing.assert_allclose(
            ledger.mapped_relative_birth_gp1_m,
            expected_birth,
            rtol=0.0,
            atol=2.0e-15,
        )
        np.testing.assert_allclose(
            expected_birth,
            np.asarray(kinematics.anchor_position_gp1_m)
            + 0.5 * np.asarray(kinematics.edge_velocity_gp1_m_per_s) * DT_S,
            rtol=0.0,
            atol=2.0e-15,
        )
    forbidden = ("gamma", "circulation", "force", "load")
    assert not any(
        token in key.casefold()
        for key in _all_keys(result.manifest())
        for token in forbidden
    )


@pytest.mark.parametrize("angle_deg", [35.0, -35.0])
def test_signed_first_birth_velocity_preserves_positive_and_negative_incidence(
    angle_deg: float,
) -> None:
    harness = _Harness(node_count=2)
    facts, events = harness.layer(
        node_angles_deg=angle_deg,
        cell_angles_deg=angle_deg,
        source_time_s=0.0,
    )
    result = NodeLocalDVMPlacementAdapter(wing_id="wing-left").map_step(
        _cells(facts, events), delta_time_s=DT_S
    )
    z_velocity = result.node_ledgers[0].edge_velocity_gp1_m_per_s[2]
    # Backend z is mapped onto -GP1-z by Z_AXIS.
    assert np.sign(z_velocity) == np.sign(angle_deg)


def test_continuous_dvm_absolute_point_is_not_mapped_or_used_for_topology() -> None:
    harness = _Harness()
    adapter = NodeLocalDVMPlacementAdapter(wing_id="wing-left")
    first_facts, first_events = harness.layer(
        node_angles_deg=35.0,
        cell_angles_deg=35.0,
        source_time_s=0.0,
    )
    adapter.map_step(_cells(first_facts, first_events), delta_time_s=DT_S)
    facts, events = harness.layer(
        node_angles_deg=35.0,
        cell_angles_deg=35.0,
        source_time_s=DT_S,
    )
    assert all(
        fact.event.lev_placement.birth_position_over_chord_backend_world is not None
        for fact in facts
    )
    result = adapter.map_step(_cells(facts, events), delta_time_s=DT_S)
    for ledger, kinematics, fact in zip(
        result.node_ledgers, result.kinematics, facts, strict=True
    ):
        assert ledger.placement_mode == "continuous"
        assert ledger.topology_owner == "attested_rvpm_frontier"
        assert ledger.edge_velocity_used_by_ribbon is False
        assert ledger.dvm_absolute_birth_used_for_topology is False
        assert ledger.mapped_relative_birth_gp1_m is None
        assert ledger.continuous_parent_source_id == (
            fact.event.lev_placement.continuous_parent_source_id
        )
        assert tuple(kinematics.edge_velocity_gp1_m_per_s) == (0.0, 0.0, 0.0)


def test_inactive_and_restart_modes_are_explicit_and_transactional() -> None:
    harness = _Harness()
    adapter = NodeLocalDVMPlacementAdapter(wing_id="wing-left")
    first_facts, first_events = harness.layer(
        node_angles_deg=35.0,
        cell_angles_deg=35.0,
        source_time_s=0.0,
    )
    adapter.map_step(_cells(first_facts, first_events), delta_time_s=DT_S)
    inactive_facts, inactive_events = harness.layer(
        node_angles_deg=0.0,
        cell_angles_deg=0.0,
        source_time_s=DT_S,
    )
    inactive = adapter.map_step(
        _cells(inactive_facts, inactive_events), delta_time_s=DT_S
    )
    assert all(ledger.placement_mode == "inactive" for ledger in inactive.node_ledgers)
    assert all(
        tuple(item.edge_velocity_gp1_m_per_s) == (0.0, 0.0, 0.0)
        for item in inactive.kinematics
    )
    restart_facts, restart_events = harness.layer(
        node_angles_deg=35.0,
        cell_angles_deg=35.0,
        source_time_s=2.0 * DT_S,
    )
    restart = adapter.map_step(_cells(restart_facts, restart_events), delta_time_s=DT_S)
    assert all(ledger.placement_mode == "restart" for ledger in restart.node_ledgers)
    assert all(ledger.edge_velocity_used_by_ribbon for ledger in restart.node_ledgers)


def test_adjacent_cells_share_one_fact_object_and_one_digest() -> None:
    harness = _Harness()
    facts, events = harness.layer(
        node_angles_deg=35.0,
        cell_angles_deg=35.0,
        source_time_s=0.0,
    )
    copied_middle = replace(facts[1])
    copied_cells = (
        DVMNodePlacementCell("cell-0", facts[0], facts[1], events[0]),
        DVMNodePlacementCell("cell-1", copied_middle, facts[2], events[1]),
    )
    with pytest.raises(ValueError, match="exact same live node fact object"):
        NodeLocalDVMPlacementAdapter(wing_id="wing-left").map_step(
            copied_cells, delta_time_s=DT_S
        )
    result = NodeLocalDVMPlacementAdapter(wing_id="wing-left").map_step(
        _cells(facts, events), delta_time_s=DT_S
    )
    assert result.cell_coverage[0].right_node_fact_manifest_sha256 == (
        result.cell_coverage[1].left_node_fact_manifest_sha256
    )
    middle_digest = result.node_ledgers[1].node_fact_manifest_sha256
    assert result.cell_coverage[0].right_node_fact_manifest_sha256 == middle_digest


def test_exact_physical_node_alias_fails_then_clean_shared_fact_retry_commits() -> None:
    harness = _Harness(node_count=4, cell_count=2)
    facts, events = harness.layer(
        node_angles_deg=35.0,
        cell_angles_deg=35.0,
        source_time_s=0.0,
    )
    aliased_right_start = _shift_fact_to_span_coordinate(facts[2], 1.0)
    aliased_cells = (
        DVMNodePlacementCell("cell-0", facts[0], facts[1], events[0]),
        DVMNodePlacementCell("cell-1", aliased_right_start, facts[3], events[1]),
    )
    adapter = NodeLocalDVMPlacementAdapter(wing_id="wing-left")
    before = adapter.state_manifest()
    with pytest.raises(ValueError, match="physical GP1 section.*two node IDs"):
        adapter.map_step(aliased_cells, delta_time_s=DT_S)
    assert adapter.state_manifest() == before

    clean_right = _shift_fact_to_span_coordinate(facts[3], 2.0)
    clean_cells = (
        DVMNodePlacementCell("cell-0", facts[0], facts[1], events[0]),
        DVMNodePlacementCell("cell-1", facts[1], clean_right, events[1]),
    )
    accepted = adapter.map_step(clean_cells, delta_time_s=DT_S)
    assert len(accepted.node_ledgers) == 3
    assert accepted.cell_coverage[0].right_node_fact_manifest_sha256 == (
        accepted.cell_coverage[1].left_node_fact_manifest_sha256
    )


@pytest.mark.parametrize(
    "nearby_coordinate",
    (np.nextafter(1.0, np.inf), 1.0 + 1.0e-12),
)
def test_one_ulp_and_nearby_distinct_nodes_are_not_tolerance_welded(
    nearby_coordinate: float,
) -> None:
    harness = _Harness(node_count=4, cell_count=3)
    facts, events = harness.layer(
        node_angles_deg=35.0,
        cell_angles_deg=35.0,
        source_time_s=0.0,
    )
    nearby = _shift_fact_to_span_coordinate(facts[2], nearby_coordinate)
    final = _shift_fact_to_span_coordinate(facts[3], 2.0)
    chain = (facts[0], facts[1], nearby, final)
    result = NodeLocalDVMPlacementAdapter(wing_id="wing-left").map_step(
        _cells(chain, events), delta_time_s=DT_S
    )
    assert len(result.node_ledgers) == 4
    assert result.node_ledgers[1].node_fact_manifest_sha256 != (
        result.node_ledgers[2].node_fact_manifest_sha256
    )


def test_same_direction_branch_and_reverse_duplicate_fail_without_commit() -> None:
    harness = _Harness()
    facts, events = harness.layer(
        node_angles_deg=35.0,
        cell_angles_deg=35.0,
        source_time_s=0.0,
    )
    for cells, message in (
        (
            (
                DVMNodePlacementCell("cell-0", facts[0], facts[1], events[0]),
                DVMNodePlacementCell("cell-1", facts[0], facts[2], events[1]),
            ),
            "two outgoing",
        ),
        (
            (
                DVMNodePlacementCell("cell-0", facts[0], facts[1], events[0]),
                DVMNodePlacementCell("cell-1", facts[1], facts[0], events[1]),
            ),
            "unordered endpoint pair",
        ),
    ):
        adapter = NodeLocalDVMPlacementAdapter(wing_id="wing-left")
        before = adapter.state_manifest()
        with pytest.raises(ValueError, match=message):
            adapter.map_step(cells, delta_time_s=DT_S)
        assert adapter.state_manifest() == before


def test_each_patch_is_one_connected_ordered_chain() -> None:
    harness = _Harness(node_count=4, cell_count=2)
    facts, events = harness.layer(
        node_angles_deg=35.0,
        cell_angles_deg=35.0,
        source_time_s=0.0,
    )
    disconnected_overlap = (
        DVMNodePlacementCell("cell-0", facts[0], facts[2], events[0]),
        DVMNodePlacementCell("cell-1", facts[1], facts[3], events[1]),
    )
    with pytest.raises(ValueError, match="one connected span chain"):
        NodeLocalDVMPlacementAdapter(wing_id="wing-left").map_step(
            disconnected_overlap, delta_time_s=DT_S
        )


def test_active_cell_with_inactive_endpoint_fails_underresolution_closed() -> None:
    harness = _Harness()
    facts, events = harness.layer(
        node_angles_deg=(35.0, 0.0, 35.0),
        cell_angles_deg=(35.0, 35.0),
        source_time_s=0.0,
    )
    with pytest.raises(ValueError, match="mixed activity/mode"):
        NodeLocalDVMPlacementAdapter(wing_id="wing-left").map_step(
            _cells(facts, events), delta_time_s=DT_S
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda fact: replace(fact, topology_patch_id="other-patch"), "patch/hinge"),
        (lambda fact: replace(fact, coordinate_frame_id="other-frame"), "frames"),
        (
            lambda fact: replace(fact, positive_span_axis_gp1=(0.0, -1.0, 0.0)),
            "right-handed",
        ),
        (
            lambda fact: replace(fact, dvm_x_axis_gp1=(2.0, 0.0, 0.0)),
            "unit length",
        ),
        (
            lambda fact: replace(fact, lev_edge_anchor_gp1_m=(np.nan, 1.0, 0.0)),
            "finite real",
        ),
    ],
)
def test_patch_frame_basis_and_nonfinite_inputs_fail_closed(
    mutation: Any, message: str
) -> None:
    harness = _Harness()
    facts, events = harness.layer(
        node_angles_deg=35.0,
        cell_angles_deg=35.0,
        source_time_s=0.0,
    )
    mutated = (facts[0], mutation(facts[1]), facts[2])
    with pytest.raises(ValueError, match=message):
        NodeLocalDVMPlacementAdapter(wing_id="wing-left").map_step(
            _cells(mutated, events), delta_time_s=DT_S
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda fact: replace(fact, source_step_index=2),
            "source step disagrees",
        ),
        (
            lambda fact: replace(fact, node_lineage_id="forged-lineage"),
            "lineage token disagrees",
        ),
        (
            lambda fact: replace(fact, geometry_token="0" * 64),
            "geometry token disagrees",
        ),
        (
            lambda fact: replace(fact, reference_chord_m=2.0 * CHORD_M),
            "chord disagrees",
        ),
        (
            lambda fact: replace(fact, position_units="ft"),
            "units are not pinned",
        ),
        (
            lambda fact: replace(fact, node_geometry_role="strength-owner"),
            "geometry-only",
        ),
        (
            lambda fact: replace(fact, exclusive_strength_owner="node-local-dvm"),
            "cell-centre strength owner",
        ),
        (
            lambda fact: replace(fact, exclusive_surface_owner="node-local-dvm"),
            "Ptera surface ownership",
        ),
    ],
)
def test_step_lineage_units_and_forbidden_owner_tampering_fails(
    mutation: Any, message: str
) -> None:
    harness = _Harness()
    facts, events = harness.layer(
        node_angles_deg=35.0,
        cell_angles_deg=35.0,
        source_time_s=0.0,
    )
    mutated = (mutation(facts[0]), facts[1], facts[2])
    with pytest.raises(ValueError, match=message):
        NodeLocalDVMPlacementAdapter(wing_id="wing-left").map_step(
            _cells(mutated, events), delta_time_s=DT_S
        )


def test_physical_dt_and_independent_source_identity_are_strict() -> None:
    harness = _Harness()
    facts, events = harness.layer(
        node_angles_deg=35.0,
        cell_angles_deg=35.0,
        source_time_s=0.0,
    )
    with pytest.raises(ValueError, match=r"event_dt\*c/U"):
        NodeLocalDVMPlacementAdapter(wing_id="wing-left").map_step(
            _cells(facts, events), delta_time_s=2.0 * DT_S
        )
    reused = replace(
        facts[1],
        event=facts[0].event,
        node_lineage_id=facts[0].node_lineage_id,
        geometry_token=facts[0].geometry_token,
    )
    with pytest.raises(ValueError, match="independent node-local DVM source"):
        NodeLocalDVMPlacementAdapter(wing_id="wing-left").map_step(
            _cells((facts[0], reused, facts[2]), events), delta_time_s=DT_S
        )


def test_hinge_boundary_requires_split_node_ids_and_facts() -> None:
    harness = _Harness(node_count=4, cell_count=2)
    facts, events = harness.layer(
        node_angles_deg=35.0,
        cell_angles_deg=35.0,
        source_time_s=0.0,
        wing_id="hinged-wing",
        node_ids=("node-0", "hinge-left", "hinge-right", "node-3"),
        patches=("inner", "inner", "outer", "outer"),
    )
    cells = (
        DVMNodePlacementCell("cell-inner", facts[0], facts[1], events[0]),
        DVMNodePlacementCell("cell-outer", facts[2], facts[3], events[1]),
    )
    result = NodeLocalDVMPlacementAdapter(wing_id="hinged-wing").map_step(
        cells, delta_time_s=DT_S
    )
    assert len(result.kinematics) == 4
    assert {item.topology_patch_id for item in result.node_ledgers} == {
        "inner",
        "outer",
    }


def test_disabled_is_input_blind_and_does_not_change_adapter_state() -> None:
    adapter = NodeLocalDVMPlacementAdapter(wing_id="wing-left")
    before = adapter.state_manifest()
    result = adapter.map_step(_Bomb(), delta_time_s=_Bomb(), enabled=False)
    assert result.enabled is False and result.status == DISABLED_STATUS
    assert result.kinematics == result.node_ledgers == result.cell_coverage == ()
    assert validate_live_dvm_node_placement_result(result) is result
    assert adapter.state_manifest() == before


def test_wing_identity_is_in_manifest_digest_and_expected_wing_validation() -> None:
    wing_a = NodeLocalDVMPlacementAdapter(wing_id="wing-A").map_step(
        _Bomb(), delta_time_s=_Bomb(), enabled=False
    )
    wing_b = NodeLocalDVMPlacementAdapter(wing_id="wing-B").map_step(
        _Bomb(), delta_time_s=_Bomb(), enabled=False
    )
    assert wing_a.manifest()["wing_id"] == "wing-A"
    assert wing_b.manifest()["wing_id"] == "wing-B"
    assert wing_a.producer_manifest_sha256 != wing_b.producer_manifest_sha256
    assert (
        validate_live_dvm_node_placement_result(wing_a, expected_wing_id="wing-A")
        is wing_a
    )
    with pytest.raises(ValueError, match="bound to another wing"):
        validate_live_dvm_node_placement_result(wing_a, expected_wing_id="wing-B")


def test_cross_wing_live_event_reuse_is_exact_once_and_transactional() -> None:
    harness = _Harness()
    facts_a, events = harness.layer(
        node_angles_deg=35.0,
        cell_angles_deg=35.0,
        source_time_s=0.0,
        wing_id="wing-A",
    )
    result_a = NodeLocalDVMPlacementAdapter(wing_id="wing-A").map_step(
        _cells(facts_a, events), delta_time_s=DT_S
    )
    assert result_a.wing_id == "wing-A"
    assert all(ledger.wing_id == "wing-A" for ledger in result_a.node_ledgers)
    assert all(row.wing_id == "wing-A" for row in result_a.cell_coverage)

    facts_b = tuple(replace(fact, wing_id="wing-B") for fact in facts_a)
    adapter_b = NodeLocalDVMPlacementAdapter(wing_id="wing-B")
    before = adapter_b.state_manifest()
    with pytest.raises(ValueError, match="already consumed by another wing"):
        adapter_b.map_step(_cells(facts_b, events), delta_time_s=DT_S)
    assert adapter_b.state_manifest() == before


def test_wrong_fact_wing_fails_before_consumption_then_clean_retry_succeeds() -> None:
    harness = _Harness()
    facts, events = harness.layer(
        node_angles_deg=35.0,
        cell_angles_deg=35.0,
        source_time_s=0.0,
        wing_id="wing-A",
    )
    adapter = NodeLocalDVMPlacementAdapter(wing_id="wing-A")
    wrong = (replace(facts[0], wing_id="wing-B"), facts[1], facts[2])
    before = adapter.state_manifest()
    with pytest.raises(ValueError, match="bound to another wing"):
        adapter.map_step(_cells(wrong, events), delta_time_s=DT_S)
    assert adapter.state_manifest() == before
    assert (
        adapter.map_step(_cells(facts, events), delta_time_s=DT_S).wing_id == "wing-A"
    )


def test_causality_failure_rolls_back_and_valid_retry_commits_once() -> None:
    harness = _Harness()
    adapter = NodeLocalDVMPlacementAdapter(wing_id="wing-left")
    first_facts, first_events = harness.layer(
        node_angles_deg=35.0,
        cell_angles_deg=35.0,
        source_time_s=0.0,
    )
    adapter.map_step(_cells(first_facts, first_events), delta_time_s=DT_S)
    second_facts, second_events = harness.layer(
        node_angles_deg=35.0,
        cell_angles_deg=35.0,
        source_time_s=DT_S,
    )
    before = adapter.state_manifest()
    stale = tuple(replace(fact, source_time_s=0.0) for fact in second_facts)
    with pytest.raises(ValueError, match="stale or noncausal"):
        adapter.map_step(_cells(stale, second_events), delta_time_s=DT_S)
    assert adapter.state_manifest() == before
    accepted = adapter.map_step(_cells(second_facts, second_events), delta_time_s=DT_S)
    assert accepted.source_step_index == 2
    committed = adapter.state_manifest()
    with pytest.raises(ValueError, match="consecutive"):
        adapter.map_step(_cells(second_facts, second_events), delta_time_s=DT_S)
    assert adapter.state_manifest() == committed


def test_result_copy_tamper_and_injected_kinematics_field_fail_attestation() -> None:
    harness = _Harness()
    facts, events = harness.layer(
        node_angles_deg=35.0,
        cell_angles_deg=35.0,
        source_time_s=0.0,
    )
    result = NodeLocalDVMPlacementAdapter(wing_id="wing-left").map_step(
        _cells(facts, events), delta_time_s=DT_S
    )
    copied = replace(result)
    with pytest.raises(ValueError, match="not a directly produced live"):
        validate_live_dvm_node_placement_result(copied)
    tampered = replace(result, status="forged")
    with pytest.raises(ValueError, match="status is not pinned"):
        validate_live_dvm_node_placement_result(tampered)

    wrong_wing = replace(result, wing_id="wing-other")
    with pytest.raises(ValueError, match="node ledger is bound to another wing"):
        validate_live_dvm_node_placement_result(wrong_wing)

    object.__setattr__(result.kinematics[0], "load", 1.0)
    with pytest.raises(ValueError, match="injected or missing field"):
        validate_live_dvm_node_placement_result(result)


def test_schema_objects_have_no_instance_dictionary_injection_surface() -> None:
    harness = _Harness(node_count=2)
    facts, events = harness.layer(
        node_angles_deg=35.0,
        cell_angles_deg=35.0,
        source_time_s=0.0,
    )
    result = NodeLocalDVMPlacementAdapter(wing_id="wing-left").map_step(
        _cells(facts, events), delta_time_s=DT_S
    )
    for value in (
        facts[0],
        _cells(facts, events)[0],
        result,
        result.node_ledgers[0],
        result.cell_coverage[0],
    ):
        assert not hasattr(value, "__dict__")
    assert result.node_geometry_role == NODE_GEOMETRY_ROLE
