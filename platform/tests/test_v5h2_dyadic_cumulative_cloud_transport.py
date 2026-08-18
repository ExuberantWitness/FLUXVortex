from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import pytest

import fluxvortex.rvpm_dyadic_edge_bridge as dyadic_bridge_module
from forward_flight_benchmarks import (
    v5h2_dyadic_cumulative_cloud_transport as dyadic_module,
)
from forward_flight_benchmarks.v5h_cumulative_cloud_transport import (
    attest_cumulative_ribbon_handoff,
    transport_accumulated_particle_cloud,
)
from forward_flight_benchmarks.v5h_dvm_node_ribbon import (
    NodeOwnedDVMRibbonShadow,
)
from forward_flight_benchmarks.v5h2_dyadic_cumulative_cloud_transport import (
    MAX_TRANSPORT_SUBSTEPS,
    attest_dyadic_cumulative_ribbon_handoff,
    materialize_dyadic_cumulative_particle_state,
    transport_dyadic_accumulated_particle_cloud,
    validate_dyadic_cumulative_cloud_transport_report,
)

from test_v5h_cumulative_cloud_transport import (
    DT_S,
    FREESTREAM_GP1_M_PER_S,
    SIGMA_BIRTH_M,
    TARGET_SPACING_M,
    _map_step,
    _source,
)


def _first_ribbon() -> tuple[NodeOwnedDVMRibbonShadow, tuple[Any, Any], Any]:
    mapper = NodeOwnedDVMRibbonShadow(wing_id="wing", source_family="lev")
    sources = (_source(0), _source(1))
    ribbon = _map_step(
        mapper,
        sources,
        active_by_cell=(True, True),
        source_time_s=0.0,
        previous_report=None,
    )
    return mapper, sources, ribbon


def _handoff(
    mapper: NodeOwnedDVMRibbonShadow,
    ribbon: object,
    *,
    source_time_s: float,
    previous_report: object | None = None,
) -> object:
    return attest_dyadic_cumulative_ribbon_handoff(
        mapper,
        ribbon,
        wing_id="wing",
        source_time_s=source_time_s,
        previous_report=previous_report,
    )


def _transport(
    handoff: object,
    *,
    source_time_s: float,
    level: int = 0,
    substeps: int = 1,
) -> Any:
    return transport_dyadic_accumulated_particle_cloud(
        handoff,
        smoothing_radius_m=SIGMA_BIRTH_M,
        base_target_spacing_m=TARGET_SPACING_M,
        refinement_level=level,
        transport_end_time_s=source_time_s + DT_S,
        transport_substeps=substeps,
        freestream_velocity_gp1_m_per_s=FREESTREAM_GP1_M_PER_S,
    )


def _fact_positions(report: object) -> np.ndarray:
    facts = sorted(report.facts, key=lambda fact: int(fact.node_id))
    return np.asarray(
        [fact.advected_position_gp1_m for fact in facts], dtype=np.float64
    )


class _Exploding:
    def __getattribute__(self, name: str) -> object:
        if name.startswith("__"):
            return object.__getattribute__(self, name)
        raise AssertionError(f"disabled path inspected {name}")

    def __float__(self) -> float:
        raise AssertionError("disabled path inspected a scalar")


def test_disabled_path_is_input_blind_and_empty() -> None:
    report = transport_dyadic_accumulated_particle_cloud(
        _Exploding(),
        smoothing_radius_m=_Exploding(),
        base_target_spacing_m=_Exploding(),
        refinement_level=_Exploding(),
        transport_end_time_s=_Exploding(),
        transport_substeps=_Exploding(),
        freestream_velocity_gp1_m_per_s=_Exploding(),
        enabled=False,
    )
    assert not report.enabled
    assert report.total_particle_count == 0
    assert report.release_plan_sidecars == ()
    assert validate_dyadic_cumulative_cloud_transport_report(report) is report


def test_level_zero_is_bitwise_v2_cloud_and_frontier_reduction() -> None:
    old_mapper, _, old_ribbon = _first_ribbon()
    old_handoff = attest_cumulative_ribbon_handoff(
        old_mapper, old_ribbon, wing_id="wing", source_time_s=0.0
    )
    old = transport_accumulated_particle_cloud(
        old_handoff,
        smoothing_radius_m=SIGMA_BIRTH_M,
        deposition_target_spacing_m=TARGET_SPACING_M,
        transport_end_time_s=DT_S,
        transport_substeps=1,
        freestream_velocity_gp1_m_per_s=FREESTREAM_GP1_M_PER_S,
    )

    mapper, _, ribbon = _first_ribbon()
    new = _transport(_handoff(mapper, ribbon, source_time_s=0.0), source_time_s=0.0)
    assert new.transported_particle_cloud == old.transported_particle_cloud
    assert np.array_equal(_fact_positions(new), _fact_positions(old))
    assert new.release_plan_sidecars[0].plan.refinement_level == 0
    assert new.release_plan_sidecars[0].plan.predicted_total_particle_count == (
        new.total_particle_count
    )


def test_three_release_level_one_cloud_is_additive_and_drives_ribbon() -> None:
    mapper, sources, ribbon = _first_ribbon()
    previous = _transport(
        _handoff(mapper, ribbon, source_time_s=0.0),
        source_time_s=0.0,
        level=1,
    )
    counts = [previous.total_particle_count]
    for source_step, source_time in ((2, DT_S), (3, 2.0 * DT_S)):
        ribbon = _map_step(
            mapper,
            sources,
            active_by_cell=(True, True),
            source_time_s=source_time,
            previous_report=previous,
        )
        handoff = _handoff(
            mapper,
            ribbon,
            source_time_s=source_time,
            previous_report=previous,
        )
        previous = _transport(handoff, source_time_s=source_time, level=1, substeps=2)
        counts.append(previous.total_particle_count)
        assert previous.for_source_step_index == source_step + 1
    assert counts[0] < counts[1] < counts[2]
    assert len(previous.transported_particle_cloud.release_slices) == 3
    assert len(previous.release_plan_sidecars) == 3
    assert all(
        sidecar.plan.refinement_level == 1 for sidecar in previous.release_plan_sidecars
    )
    assert validate_dyadic_cumulative_cloud_transport_report(previous) is previous


def test_active_inactive_restart_continuous_has_no_phantom_sidecar() -> None:
    mapper, sources, first = _first_ribbon()
    report1 = _transport(
        _handoff(mapper, first, source_time_s=0.0),
        source_time_s=0.0,
        level=1,
    )
    count1 = report1.total_particle_count

    inactive = _map_step(
        mapper,
        sources,
        active_by_cell=(False, False),
        source_time_s=DT_S,
        previous_report=None,
    )
    assert inactive.edge_graph is None
    report2 = _transport(
        _handoff(
            mapper,
            inactive,
            source_time_s=DT_S,
            previous_report=report1,
        ),
        source_time_s=DT_S,
        level=1,
    )
    assert report2.total_particle_count == count1
    assert report2.new_particle_count == 0
    assert report2.facts == ()
    assert len(report2.release_plan_sidecars) == 1

    restart = _map_step(
        mapper,
        sources,
        active_by_cell=(True, True),
        source_time_s=2.0 * DT_S,
        previous_report=None,
    )
    assert all(birth.mode == "restart" for birth in restart.node_births)
    report3 = _transport(
        _handoff(
            mapper,
            restart,
            source_time_s=2.0 * DT_S,
            previous_report=report2,
        ),
        source_time_s=2.0 * DT_S,
        level=1,
    )
    assert report3.total_particle_count == count1 + report3.new_particle_count
    assert len(report3.release_plan_sidecars) == 2

    continuous = _map_step(
        mapper,
        sources,
        active_by_cell=(True, True),
        source_time_s=3.0 * DT_S,
        previous_report=report3,
    )
    report4 = _transport(
        _handoff(
            mapper,
            continuous,
            source_time_s=3.0 * DT_S,
            previous_report=report3,
        ),
        source_time_s=3.0 * DT_S,
        level=1,
    )
    assert report4.total_particle_count == (
        report3.total_particle_count + report4.new_particle_count
    )
    assert tuple(
        release.source_step_index
        for release in report4.transported_particle_cloud.release_slices
    ) == (1, 3, 4)
    assert len(report4.release_plan_sidecars) == 3


def test_failure_does_not_consume_handoff_and_clean_retry_succeeds() -> None:
    mapper, _, ribbon = _first_ribbon()
    handoff = _handoff(mapper, ribbon, source_time_s=0.0)
    with pytest.raises(ValueError, match="substeps"):
        _transport(
            handoff,
            source_time_s=0.0,
            substeps=MAX_TRANSPORT_SUBSTEPS + 1,
        )
    report = _transport(handoff, source_time_s=0.0)
    assert report.enabled
    with pytest.raises(ValueError, match="already consumed"):
        _transport(handoff, source_time_s=0.0)


def test_runtime_bridge_callable_replacement_fails_before_call_and_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mapper, _, ribbon = _first_ribbon()
    handoff = _handoff(mapper, ribbon, source_time_s=0.0)
    calls = 0
    original = dyadic_bridge_module.deposit_edge_graph_prescribed_sigma_dyadic_panels

    def forwarding(*args: object, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    with monkeypatch.context() as scoped:
        scoped.setattr(
            dyadic_bridge_module,
            "deposit_edge_graph_prescribed_sigma_dyadic_panels",
            forwarding,
        )
        with pytest.raises(ValueError, match="callable was replaced"):
            _transport(handoff, source_time_s=0.0)
    assert calls == 0
    report = _transport(handoff, source_time_s=0.0)
    assert report.enabled

    # The imported local binding is independently frozen as well.
    mapper, _, ribbon = _first_ribbon()
    handoff = _handoff(mapper, ribbon, source_time_s=0.0)
    with monkeypatch.context() as scoped:
        scoped.setattr(
            dyadic_module,
            "deposit_edge_graph_prescribed_sigma_dyadic_panels",
            forwarding,
        )
        with pytest.raises(ValueError, match="callable was replaced"):
            _transport(handoff, source_time_s=0.0)
    assert _transport(handoff, source_time_s=0.0).enabled


def test_copy_and_nested_tamper_are_not_live_reports() -> None:
    mapper, _, ribbon = _first_ribbon()
    report = _transport(_handoff(mapper, ribbon, source_time_s=0.0), source_time_s=0.0)
    with pytest.raises(ValueError):
        validate_dyadic_cumulative_cloud_transport_report(replace(report))
    sidecar = report.release_plan_sidecars[0]
    forged_sidecar = replace(sidecar, source_time_s=np.nextafter(0.0, 1.0))
    with pytest.raises(ValueError):
        validate_dyadic_cumulative_cloud_transport_report(
            replace(report, release_plan_sidecars=(forged_sidecar,))
        )


def test_v2_and_dyadic_share_cross_version_live_ribbon_exact_once() -> None:
    mapper, _, ribbon = _first_ribbon()
    _ = attest_cumulative_ribbon_handoff(
        mapper, ribbon, wing_id="wing", source_time_s=0.0
    )
    with pytest.raises(ValueError, match="already consumed"):
        _handoff(mapper, ribbon, source_time_s=0.0)

    mapper, _, ribbon = _first_ribbon()
    _ = _handoff(mapper, ribbon, source_time_s=0.0)
    with pytest.raises(ValueError, match="already consumed"):
        attest_cumulative_ribbon_handoff(
            mapper, ribbon, wing_id="wing", source_time_s=0.0
        )


def test_materialized_state_is_a_copy() -> None:
    mapper, _, ribbon = _first_ribbon()
    report = _transport(_handoff(mapper, ribbon, source_time_s=0.0), source_time_s=0.0)
    first = materialize_dyadic_cumulative_particle_state(report)
    second = materialize_dyadic_cumulative_particle_state(report)
    first.positions[0, 0] += 1.0
    assert not np.array_equal(first.positions, second.positions)
    assert validate_dyadic_cumulative_cloud_transport_report(report) is report
