from __future__ import annotations

import json
from pathlib import Path

import pytest

from forward_flight_benchmarks.run_v5h_frontier_vertical_gate import (
    CASE_GEOMETRIES,
    DEFAULT_SPAN_SUBDIVISIONS,
    DEFAULT_TRANSPORT_SUBSTEPS,
    GateConfig,
    run_frontier_vertical_gate,
    write_artifact,
)


@pytest.fixture(scope="module")
def default_result() -> dict[str, object]:
    return run_frontier_vertical_gate()


def test_two_release_gate_covers_all_frozen_non_target_configurations(
    default_result: dict[str, object],
) -> None:
    assert default_result["schema_id"] == (
        "fluxv-v5h-two-release-frontier-vertical-gate-v1"
    )
    assert default_result["scope"] == (
        "non_target_two_release_frontier_vertical_slice_only"
    )
    assert default_result["observation_access"] == "none"
    assert default_result["target_case_branch"] == "none"
    assert default_result["canonical_eligible"] is False
    assert default_result["feedback_call_count"] == 0
    assert default_result["parent_write_count"] == 0
    assert default_result["load_write_count"] == 0

    geometries = default_result["geometries"]
    assert [item["geometry"] for item in geometries] == list(CASE_GEOMETRIES)
    expected_count = len(DEFAULT_SPAN_SUBDIVISIONS) * len(DEFAULT_TRANSPORT_SUBSTEPS)
    for geometry in geometries:
        assert len(geometry["configurations"]) == expected_count
        assert geometry["configuration_passed"] is True
        for row in geometry["configurations"]:
            assert row["source_passed"] is True
            assert row["ribbon_passed"] is True
            assert row["fixed_sigma_bridge_passed"] is True
            assert row["transport_passed"] is True
            assert row["temporal_layer_overlap"]["passed"] is True
            assert row["first_modes"] == ["first"] * 5
            assert row["second_modes"] == ["continuous"] * 5
            assert row["frontier_fact_count"] == 5
            assert row["accepted_frontier_count"] == 5


def test_gate_stops_on_frontier_spatial_nonconvergence_without_hiding_passes(
    default_result: dict[str, object],
) -> None:
    assert default_result["status"] == "failed"
    assert default_result["passed"] is False
    summary = default_result["gate_summary"]
    assert summary == {
        "configuration_mechanics_passed": True,
        "time_refinement_passed": True,
        "spatial_refinement_passed": False,
        "stop_required": True,
    }
    for geometry in default_result["geometries"]:
        assert geometry["time_refinement_passed"] is True
        spatial = geometry["spatial_refinement_at_finest_time"]
        assert spatial["probe_velocity_passed"] is True
        assert spatial["frontier_passed"] is False
        assert spatial["passed"] is False
        frontier = spatial["consecutive_frontier_position_relative_l2"]
        assert len(frontier) == 3
        assert frontier[0] < frontier[1] < frontier[2]
        assert frontier[2] < spatial["fine_relative_l2_limit"]
        probes = spatial["consecutive_probe_velocity_relative_l2"]
        assert probes[0] > probes[1] > probes[2]
        assert probes[2] < spatial["fine_relative_l2_limit"]
    assert (
        "cumulative-cloud exact-once merge v2"
        in default_result["blocked"]["third_release"]
    )


def test_headline_mechanical_bounds_are_recomputable_from_rows(
    default_result: dict[str, object],
) -> None:
    rows = [
        row
        for geometry in default_result["geometries"]
        for row in geometry["configurations"]
    ]
    assert max(row["source_kelvin_max_m2_per_s"] for row in rows) <= 1.0e-10
    assert max(row["particle_invariant_relative_drift_max"] for row in rows) <= 1.0e-6
    assert max(row["global_vector_drift_relative"] for row in rows) <= 1.0e-6
    assert max(row["max_stage_dt_jacobian_frobenius"] for row in rows) <= 1.5
    assert max(row["max_stage_dt_self_speed_over_sigma"] for row in rows) <= 0.5
    assert (
        max(
            row["temporal_layer_overlap"]["max_distance_over_target_spacing"]
            for row in rows
        )
        <= 1.0 + 1.0e-12
    )
    assert min(
        row["temporal_layer_overlap"]["min_two_sided_sigma_over_distance"]
        for row in rows
    ) >= 2.0 * (1.0 - 1.0e-12)


def test_code_hashes_cover_the_executed_frontier_and_transport_chain(
    default_result: dict[str, object],
) -> None:
    required = {
        "runner",
        "manufactured_geometry_provider",
        "dvm_source",
        "dvm_node_ribbon",
        "passive_frontier_transport",
        "ldvm_fourier",
        "ldvm_uvlm_correction",
        "rvpm_edge_bridge",
        "rvpm_reference",
        "rvpm_transport",
    }
    hashes = default_result["code_sha256"]
    assert set(hashes) == required
    assert all(len(value) == 64 for value in hashes.values())


@pytest.mark.parametrize(
    "config",
    (
        GateConfig(span_cells=1),
        GateConfig(span_subdivisions=(2, 4)),
        GateConfig(transport_substeps=(1, 2)),
        GateConfig(span_subdivisions=(2, 2, 4)),
        GateConfig(transport_substeps=(1, 4, 2)),
    ),
)
def test_invalid_refinement_families_fail_closed(config: GateConfig) -> None:
    with pytest.raises(ValueError):
        run_frontier_vertical_gate(config)


def test_artifact_writer_refuses_overwrite(
    default_result: dict[str, object], tmp_path: Path
) -> None:
    output = tmp_path / "frontier.json"
    assert write_artifact(default_result, output) == output
    loaded = json.loads(output.read_text(encoding="utf-8"))
    assert loaded["status"] == "failed"
    assert loaded["gate_summary"]["stop_required"] is True
    serialized = output.read_text(encoding="utf-8")
    assert "Infinity" not in serialized
    assert "NaN" not in serialized
    assert all(
        gate["coarse_to_fine_error_ratio"] is None
        for geometry in loaded["geometries"]
        for gate in geometry["time_refinement_by_span_subdivision"]
    )
    with pytest.raises(FileExistsError):
        write_artifact(default_result, output)

    nonfinite_output = tmp_path / "nonfinite.json"
    with pytest.raises(ValueError, match="Out of range float values"):
        write_artifact(
            {**default_result, "forged_nonfinite": float("nan")}, nonfinite_output
        )
    assert not nonfinite_output.exists()
