"""Mechanical contracts for the observation-free FluxV v5h6 gate."""

from __future__ import annotations

from dataclasses import replace
from math import isclose

import numpy as np
import pytest

from forward_flight_benchmarks import run_fluxv_v5h6_geometry_refinement as gate


def _synthetic_row(value: float) -> gate.ConfigurationResult:
    vector = np.full((5, 3), value, dtype=np.float64)
    return gate.ConfigurationResult(
        geometry="straight",
        refinement_level=0,
        delta_time_s=0.02,
        num_steps=3,
        particle_count=1,
        report_sha256=("0" * 64,),
        frontier_displacement_gp1_m=vector.copy(),
        probe_velocity_gp1_m_per_s=vector.copy(),
        final_cloud_sha256="1" * 64,
        max_no_penetration_abs=0.0,
        max_exact_append_abs=0.0,
        max_frontier_replay_abs=0.0,
        mechanics_passed=True,
    )


@pytest.mark.parametrize("geometry", gate.GEOMETRIES)
def test_geometry_nodes_share_one_orthonormal_gp1_law(geometry: str) -> None:
    nodes = gate._geometry_nodes(geometry)
    assert len(nodes) == gate.SPAN_CELLS + 1
    assert [node.leading_edge_gp1_m[1] for node in nodes] == pytest.approx(
        np.linspace(0.0, gate.SPAN_M, gate.SPAN_CELLS + 1)
    )
    for node in nodes:
        x_axis = np.asarray(node.x_axis_gp1)
        z_axis = np.asarray(node.z_axis_gp1)
        span_axis = np.asarray((0.0, 1.0, 0.0))
        assert np.linalg.norm(x_axis) == pytest.approx(1.0)
        assert np.linalg.norm(z_axis) == pytest.approx(1.0)
        assert np.dot(x_axis, z_axis) == pytest.approx(0.0, abs=1.0e-15)
        assert np.cross(span_axis, x_axis) == pytest.approx(z_axis)

    if geometry == "straight":
        assert [node.chord_m for node in nodes] == pytest.approx([0.20] * 5)
        assert [node.twist_deg for node in nodes] == pytest.approx([0.0] * 5)
    elif geometry == "taper":
        assert nodes[0].chord_m == pytest.approx(0.20)
        assert nodes[-1].chord_m == pytest.approx(0.12)
        assert [node.twist_deg for node in nodes] == pytest.approx([0.0] * 5)
    else:
        assert [node.chord_m for node in nodes] == pytest.approx([0.20] * 5)
        assert nodes[0].twist_deg == pytest.approx(0.0)
        assert nodes[-1].twist_deg == pytest.approx(20.0)


def test_ptera_end_sections_match_the_shared_geometry_law() -> None:
    for geometry in gate.GEOMETRIES:
        problem = gate._problem(geometry, delta_time_s=0.02, num_steps=3)
        wing = problem.steady_problems[0].airplanes[0].wings[0]
        root, tip = wing.wing_cross_sections
        expected_root = gate._geometry_nodes(geometry)[0]
        expected_tip = gate._geometry_nodes(geometry)[-1]
        assert root.chord == pytest.approx(expected_root.chord_m)
        assert tip.chord == pytest.approx(expected_tip.chord_m)
        assert tip.angles_Wcsp_to_Wcs_ixyz[1] == pytest.approx(expected_tip.twist_deg)
        assert root.num_spanwise_panels == gate.SPAN_CELLS


@pytest.mark.parametrize("chord_m", (0.12, 0.16, 0.20))
@pytest.mark.parametrize("delta_time_s", gate.TEMPORAL_DT_S)
def test_preregistered_source_protocol_is_active_at_every_layer(
    chord_m: float, delta_time_s: float
) -> None:
    source = gate._source("straight", "activity", 0, chord_m, delta_time_s)
    num_steps = round(gate.PHYSICAL_HORIZON_S / delta_time_s)
    events = tuple(
        source.step(np.deg2rad(gate.SOURCE_INCIDENCE_DEG), 0.0, 0.0)
        for _ in range(num_steps)
    )
    assert all(event.lesp_active for event in events)
    assert events[0].lev_birth_mode == "first"
    assert all(event.lev_birth_mode == "continuous" for event in events[1:])


def test_refinement_contract_has_fifteen_unique_configurations() -> None:
    keys = {
        (geometry, level, 0.02)
        for geometry in gate.GEOMETRIES
        for level in gate.SPATIAL_LEVELS
    }
    keys.update(
        (geometry, 0, delta_time)
        for geometry in gate.GEOMETRIES
        for delta_time in gate.TEMPORAL_DT_S[1:]
    )
    assert len(keys) == 15
    assert all(
        isclose(round(gate.PHYSICAL_HORIZON_S / dt) * dt, gate.PHYSICAL_HORIZON_S)
        for _, _, dt in keys
    )


def test_convergence_gate_accepts_second_order_family_and_rejects_failures() -> None:
    passing = tuple(_synthetic_row(value) for value in (1.04, 1.01, 1.0))
    result = gate._convergence_gate(passing, "frontier_displacement_gp1_m")
    assert result["passed"]
    assert result["middle_to_fine_relative_difference"] == pytest.approx(0.01)
    assert result["difference_ratio"] == pytest.approx(3.0 / 1.01)

    slow = tuple(_synthetic_row(value) for value in (1.03, 1.02, 1.0))
    assert not gate._convergence_gate(slow, "frontier_displacement_gp1_m")["passed"]

    wrong_shape = replace(
        passing[-1], frontier_displacement_gp1_m=np.ones((4, 3), dtype=np.float64)
    )
    assert not gate._convergence_gate(
        (passing[0], passing[1], wrong_shape), "frontier_displacement_gp1_m"
    )["passed"]


def test_minimal_straight_smoke_closes_all_mechanical_ledgers() -> None:
    summary = gate.run_minimal_smoke()
    assert summary["passed"] is True
    row = summary["row"]
    assert row["geometry"] == "straight"
    assert row["num_steps"] == 3
    assert row["particle_count"] == 102
    assert row["max_no_penetration_abs"] <= 1.0e-12
    assert row["max_exact_append_abs"] == 0.0
    assert row["max_frontier_replay_abs"] == 0.0
    assert len(row["report_sha256"]) == 3


def test_strict_json_rejects_nonfinite_values() -> None:
    with pytest.raises(ValueError):
        gate._strict_json({"bad": float("nan")})
