from __future__ import annotations

import ast
import builtins
from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any

import numpy as np
import pytest


# Importing the benchmark package also imports Numba-backed Ptera modules.  Keep
# their cache writes outside the source tree and establish the paths before the
# package import occurs.
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/fluxv-v5h-cumulative-numba-cache")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/fluxv-v5h-cumulative-mpl-cache")

import forward_flight_benchmarks.run_v5h_cumulative_cloud_gate as gate


def _trace_sha256(events: list[str]) -> str:
    digest = sha256(b"fluxv-v5h-cumulative-transport-trace-v2\0")
    for event in events:
        digest.update(event.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _fake_release(
    step: int,
    boundaries: tuple[int, int, int],
    digest: str,
    trace_sha256: str,
) -> dict[str, Any]:
    previous_count = 0 if step == 1 else boundaries[step - 2]
    total_count = boundaries[step - 1]
    new_count = total_count - previous_count
    return {
        "source_step_index": step,
        "source_time_s": gate.PHYSICAL_RELEASE_DT_S * (step - 1),
        "incidence_rad": (
            gate.FIRST_ALPHA_RAD if step == 1 else gate.CONTINUOUS_ALPHA_RAD
        ),
        "expected_mode": "first" if step == 1 else "continuous",
        "previous_particle_count": previous_count,
        "new_particle_count": new_count,
        "total_particle_count": total_count,
        "predicted_new_particle_count": new_count,
        "active_release_steps": list(range(1, step + 1)),
        "release_slice_steps": list(range(1, step + 1)),
        "source_passed": True,
        "placement": {"passed": True},
        "placement_birth_passed": True,
        "continuous_one_third_passed": True,
        "ribbon_handoff": {"passed": True},
        "cell_strength_owner_passed": True,
        "ribbon_passed": True,
        "prefix_identity_passed": True,
        "previous_report_immutable_passed": True,
        "new_release_identity_passed": True,
        "release_slice_ledger_passed": True,
        "report_sha256": digest,
        "cloud_sha256": digest,
        "transport_trace_sha256": trace_sha256,
        "counters_passed": True,
        "passed": True,
    }


def _fake_configuration(
    geometry: gate.GateGeometry,
    *,
    span_cells: int,
    target_spacing_m: float,
    transport_substeps: int,
) -> gate._ConfigurationResult:
    space_signal = target_spacing_m**2
    time_signal = 1.0e-7 / transport_substeps**2
    frontier = np.asarray(
        [
            (
                1.0 + 0.1 * index + space_signal + time_signal,
                0.25 * index,
                2.0 - space_signal - time_signal,
            )
            for index in range(5)
        ],
        dtype=np.float64,
    )
    births = np.zeros_like(frontier)
    probes = gate.FIXED_PROBES_GP1_M.copy()
    probes[:, 0] += 0.5 * space_signal + time_signal
    probes[:, 2] -= 0.25 * space_signal + time_signal
    particle_count = int(round(1.0 / target_spacing_m))
    per_release = particle_count // gate.M3_RELEASE_COUNT
    boundaries = (per_release, 2 * per_release, particle_count)
    realized_spacing = 0.9 * target_spacing_m
    digest_seed = f"{geometry}:{target_spacing_m.hex()}:{transport_substeps}".encode(
        "ascii"
    )
    digest = sha256(digest_seed).hexdigest()
    traces: list[list[str]] = []
    for step in range(1, gate.M3_RELEASE_COUNT + 1):
        trace = [f"deposit_prescribed_sigma_spacing:release={step}"]
        for substep in range(1, transport_substeps + 1):
            trace.append(f"combined_lsrk3_call:substep={substep}")
            for stage in range(1, 4):
                trace.append(f"combined_lsrk3_stage:substep={substep}:stage={stage}")
                trace.append(
                    f"frontier_stage_pre_field:substep={substep}:stage={stage}"
                )
        traces.append(trace)
    trace_hashes = [_trace_sha256(trace) for trace in traces]
    releases = [
        _fake_release(step, boundaries, digest, trace_hashes[step - 1])
        for step in range(1, gate.M3_RELEASE_COUNT + 1)
    ]
    summary = {
        "geometry": geometry,
        "span_cells": span_cells,
        "target_spacing_m": target_spacing_m,
        "transport_substeps": transport_substeps,
        "smoothing_radius_m": gate.SIGMA_BIRTH_M,
        "release_count": gate.M3_RELEASE_COUNT,
        "active_release_count": gate.M3_RELEASE_COUNT,
        "active_release_steps": [1, 2, 3],
        "final_particle_count": particle_count,
        "realized_spacing_max_m": realized_spacing,
        "final_cloud_sha256": digest,
        "final_report_sha256": digest,
        "final_state_sha256": digest,
        "releases": releases,
        "passed": True,
    }
    positions = np.column_stack(
        (
            np.linspace(0.0, 1.0, particle_count),
            np.zeros(particle_count),
            np.zeros(particle_count),
        )
    )
    gamma = np.zeros_like(positions)
    sigma = np.full(particle_count, gate.SIGMA_BIRTH_M)

    def slice_records(step: int) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for release_index in range(1, step + 1):
            start = 0 if release_index == 1 else boundaries[release_index - 2]
            stop = boundaries[release_index - 1]
            rows.append(
                {
                    "record_type": "ReleaseSliceLedger",
                    "release_index": release_index,
                    "source_step_index": release_index,
                    "source_time_s": gate.PHYSICAL_RELEASE_DT_S * (release_index - 1),
                    "start_index": start,
                    "stop_index": stop,
                    "particle_count": stop - start,
                    "parent_ribbon_digest_sha256": digest,
                    "deposited_cloud_digest_sha256": digest,
                    "smoothing_radius_m": gate.SIGMA_BIRTH_M,
                    "deposition_target_spacing_m": target_spacing_m,
                    "particle_ids_sha256": digest,
                    "lineage_sha256": digest,
                }
            )
        return rows

    release_raw = []
    for step, total_count in enumerate(boundaries, start=1):
        previous_count = 0 if step == 1 else boundaries[step - 2]
        new_count = total_count - previous_count
        edge_spacing = 0.9 * target_spacing_m
        edge_overlap = gate.SIGMA_BIRTH_M / edge_spacing
        edge_metrics = {
            "active": True,
            "particle_count": new_count,
            "retained_edge_count": 1,
            "edge_subdivision_counts": [new_count],
            "edge_realization": [
                {
                    "subdivision_count": new_count,
                    "realized_spacing_m": edge_spacing,
                    "realized_overlap": edge_overlap,
                }
            ],
            "requested_target_spacing_m": target_spacing_m,
            "smoothing_radius_m": gate.SIGMA_BIRTH_M,
            "fixed_physical_sigma": True,
            "finite": True,
            "realized_spacing_min_m": edge_spacing,
            "realized_spacing_max_m": edge_spacing,
            "realized_overlap_min": edge_overlap,
            "realized_overlap_max": edge_overlap,
            "max_edge_conservation_abs": 0.0,
            "global_conservation_abs": 0.0,
            "incidence_residual": 0.0,
            "edge_reconstruction_residual": 0.0,
            "passed": True,
        }
        transport_invariants = {
            "particle_invariant_relative_drift_max": 0.0,
            "global_vector_change_abs_m3_per_s": 0.0,
            "global_vector_change_relative": 0.0,
            "global_vector_change_gate_eligible": False,
            "positive_finite_sigma": True,
            "passed": True,
        }
        counters = {
            "deposition_call_count": 1,
            "lsrk3_call_count": transport_substeps,
            "lsrk3_stage_count": 3 * transport_substeps,
            "stage_pre_field_call_count": 3 * transport_substeps,
            "combined_stage_particle_counts": [total_count] * (3 * transport_substeps),
            "sort_count": 0,
            "weld_count": 0,
            "delete_count": 0,
            "cancel_count": 0,
            "remesh_count": 0,
            "feedback_call_count": 0,
            "parent_write_count": 0,
            "surface_channel_write_count": 0,
        }
        release_raw.append(
            {
                "source_step_index": step,
                "positions_gp1_m": positions[:total_count].copy(),
                "gamma_vector_m3_per_s": gamma[:total_count].copy(),
                "sigma_m": sigma[:total_count].copy(),
                "frontier_positions_gp1_m": frontier.copy(),
                "latest_birth_positions_gp1_m": births.copy(),
                "frontier_minus_latest_birth_gp1_m": frontier - births,
                "fixed_probe_induced_velocity_gp1_m_per_s": probes.copy(),
                "frontier_fact_identity": [
                    [
                        f"node-{node_index}",
                        f"epoch-{node_index}",
                        digest,
                        digest,
                        step,
                        step + 1,
                    ]
                    for node_index in range(span_cells + 1)
                ],
                "particle_ids": [f"particle-{index}" for index in range(total_count)],
                "lineage": [f"lineage-{index}" for index in range(total_count)],
                "release_slices": slice_records(step),
                "cloud_sha256": digest,
                "report_sha256": digest,
                "current_ribbon_digest_sha256": digest,
                "handoff_sha256": digest,
                "appended_cloud_digest_before_transport_sha256": digest,
                "transported_cloud_digest_after_sha256": digest,
                "edge_bridge_artifact_sha256": digest,
                "transport_trace": traces[step - 1],
                "transport_trace_sha256": trace_hashes[step - 1],
                "source_time_s": gate.PHYSICAL_RELEASE_DT_S * (step - 1),
                "transport_start_time_s": gate.PHYSICAL_RELEASE_DT_S * (step - 1),
                "transport_end_time_s": gate.PHYSICAL_RELEASE_DT_S * step,
                "transport_substeps": transport_substeps,
                "previous_particle_count": previous_count,
                "new_particle_count": new_count,
                "predicted_new_particle_count": new_count,
                "total_particle_count": total_count,
                "exact_append_passed": True,
                "one_combined_field_passed": True,
                "stage_pre_replay_passed": True,
                "exact_append_prefix_max_abs": 0.0,
                "stage_pre_replay_max_abs": 0.0,
                "transport_counters": counters,
                "edge_metrics": edge_metrics,
                "transport_invariants": transport_invariants,
                "measured_mechanical_attestations": {
                    "source_passed": True,
                    "placement_passed": True,
                    "placement_birth_passed": True,
                    "continuous_one_third_passed": True,
                    "ribbon_handoff_passed": True,
                    "cell_strength_owner_passed": True,
                    "ribbon_passed": True,
                    "prefix_identity_passed": True,
                    "previous_report_immutable_passed": True,
                    "new_release_identity_passed": True,
                    "release_slice_ledger_passed": True,
                    "report_exact_append_passed": True,
                    "report_one_combined_field_passed": True,
                    "report_stage_pre_replay_passed": True,
                    "counters_passed": True,
                    "observation_access_none": True,
                    "target_case_branch_none": True,
                },
            }
        )
        releases[step - 1].update(
            {
                "edge_metrics": edge_metrics,
                "transport_invariants": transport_invariants,
                "transport_counters": counters,
            }
        )
    raw = {
        "positions_gp1_m": positions,
        "gamma_vector_m3_per_s": gamma,
        "sigma_m": sigma,
        "frontier_minus_latest_birth_gp1_m": frontier,
        "fixed_probe_induced_velocity_gp1_m_per_s": probes,
        "particle_ids": [f"particle-{index}" for index in range(particle_count)],
        "lineage": [f"lineage-{index}" for index in range(particle_count)],
        "release_slices": slice_records(gate.M3_RELEASE_COUNT),
        "cloud_sha256": digest,
        "report_sha256": digest,
        "transport_trace": traces[-1],
        "transport_trace_sha256": trace_hashes[-1],
        "release_raw": release_raw,
    }
    return gate._ConfigurationResult(
        summary=summary,
        raw=raw,
        frontier_minus_latest_birth=frontier,
        fixed_probe_induced_velocity=probes,
        particle_count=particle_count,
        realized_spacing_max_m=realized_spacing,
    )


def _fake_smoke() -> dict[str, Any]:
    return {
        "schema_id": f"{gate.RUN_SCHEMA_ID}:minimal-smoke",
        "summary": {
            "m0_v1_physical_bitwise_parity": {"passed": True},
            "m1_three_and_four_release": {"passed": True},
            "m2_lifecycle_and_partial_activity": {"passed": True},
            "passed": True,
        },
        "raw_arrays": {
            "m0_v1_physical_bitwise_parity": {
                "state": np.asarray((1.0, 2.0), dtype=np.float64)
            },
            "m1_three_and_four_release": {
                "state": np.asarray((2.0, 3.0), dtype=np.float64)
            },
            "m2_lifecycle": {"state": np.asarray((3.0, 4.0), dtype=np.float64)},
        },
        "passed": True,
    }


def _fake_gate_result(
    monkeypatch: pytest.MonkeyPatch,
    *,
    failing_geometry: str | None = None,
    smoke: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], list[tuple[str, float, int]]]:
    calls: list[tuple[str, float, int]] = []

    def fake_run(
        geometry: gate.GateGeometry,
        *,
        span_cells: int,
        target_spacing_m: float,
        transport_substeps: int,
    ) -> gate._ConfigurationResult:
        calls.append((geometry, target_spacing_m, transport_substeps))
        row = _fake_configuration(
            geometry,
            span_cells=span_cells,
            target_spacing_m=target_spacing_m,
            transport_substeps=transport_substeps,
        )
        if geometry == failing_geometry:
            row.summary["passed"] = False
            row.summary["releases"][0]["passed"] = False
        return row

    monkeypatch.setattr(
        gate,
        "run_minimal_smoke",
        _fake_smoke if smoke is None else lambda: smoke,
    )
    monkeypatch.setattr(gate, "_run_configuration", fake_run)
    return gate.run_cumulative_cloud_gate(), calls


@pytest.fixture(scope="module")
def minimal_smoke() -> dict[str, Any]:
    return gate.run_minimal_smoke()


@pytest.fixture(scope="module")
def straight_configuration() -> gate._ConfigurationResult:
    return gate._run_configuration(
        "straight",
        span_cells=gate.GateConfig().span_cells,
        target_spacing_m=gate.TARGET_SPACINGS_M[0],
        transport_substeps=gate.TRANSPORT_SUBSTEPS[0],
    )


@pytest.fixture(scope="module")
def fake_full_result(minimal_smoke: dict[str, Any]) -> dict[str, Any]:
    patcher = pytest.MonkeyPatch()
    try:
        result, calls = _fake_gate_result(patcher, smoke=minimal_smoke)
    finally:
        patcher.undo()
    assert len(calls) == gate.MAX_CONFIGURATIONS
    return result


@pytest.fixture(scope="module")
def minimal_stop_result(minimal_smoke: dict[str, Any]) -> dict[str, Any]:
    patcher = pytest.MonkeyPatch()
    try:
        patcher.setattr(gate, "run_minimal_smoke", lambda: minimal_smoke)
        result = gate.run_cumulative_cloud_gate(minimal_smoke_only=True)
    finally:
        patcher.undo()
    return result


def _provenance(
    output: Path,
    *,
    run_uuid: str,
    minute: int,
) -> dict[str, Any]:
    return gate.capture_run_provenance(
        output_dir=output,
        numerical_started_utc=f"2026-08-15T00:{minute:02d}:00Z",
        numerical_completed_utc=f"2026-08-15T00:{minute:02d}:01Z",
        run_uuid=run_uuid,
    )


def _write_json(path: Path, payload: object) -> None:
    path.write_text(gate._json_text(payload), encoding="utf-8")


def _first_configuration_pair(
    summary: dict[str, Any],
    raw: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    configuration = summary["summary"]["m3"]["geometries"][0]["configurations"][0]
    record = next(
        item
        for item in raw["m3_configurations"]
        if item["geometry"] == configuration["geometry"]
        and item["target_spacing_m"] == configuration["target_spacing_m"]
        and item["transport_substeps"] == configuration["transport_substeps"]
    )
    return configuration, record


def _rebind_configuration_summary(
    configuration: dict[str, Any],
    raw_record: dict[str, Any],
) -> None:
    raw_record["configuration_summary_sha256"] = gate._json_payload_sha256(
        "fluxv-v5h-cumulative-configuration-summary-v1",
        configuration,
    )


def test_gate_config_grid_release_counts_and_thresholds_are_frozen() -> None:
    config = gate.GateConfig()
    assert config.span_cells == 4
    assert config.target_spacings_m == (0.04, 0.02, 0.01, 0.005)
    assert config.transport_substeps == (1, 2, 4)
    assert config.m3_release_count == 3
    assert (
        len(gate.CASE_GEOMETRIES)
        * len(config.target_spacings_m)
        * len(config.transport_substeps)
        == gate.MAX_CONFIGURATIONS
        == 36
    )
    assert gate.MIN_TIME_REFINEMENT_RATIO == 1.5
    assert gate.MIN_H_REFINEMENT_RATIO == 1.5
    with pytest.raises(FrozenInstanceError):
        config.span_cells = 8  # type: ignore[misc]
    for changed in (
        replace(config, span_cells=3),
        replace(config, target_spacings_m=(0.04, 0.02, 0.01)),
        replace(config, transport_substeps=(1, 2)),
        replace(config, m3_release_count=4),
        replace(config, configuration_cap=35),
    ):
        with pytest.raises(ValueError, match="frozen"):
            gate._validate_config(changed)


def test_refinement_reduction_ratio_boundary_is_exactly_one_point_five() -> None:
    boundary = gate._refinement_family(
        (2.25e-7, 1.5e-7, 1.0e-7),
        fine_limit=1.0e-6,
        minimum_ratio=1.5,
        require_nondegenerate=True,
    )
    below = gate._refinement_family(
        (2.249e-7, 1.5e-7, 1.0e-7),
        fine_limit=1.0e-6,
        minimum_ratio=1.5,
        require_nondegenerate=True,
    )
    assert boundary["coarse_to_fine_ratios"] == [1.5, 1.5]
    assert boundary["passed"] is True
    assert below["passed"] is False


def test_m0_m1_m2_bounded_smoke_closes_all_pre_gates(
    minimal_smoke: dict[str, Any],
) -> None:
    assert minimal_smoke["passed"] is True
    summary = minimal_smoke["summary"]
    m0 = summary["m0_v1_physical_bitwise_parity"]
    assert m0["passed"] is True
    for field in (
        "positions_bitwise_equal",
        "gamma_bitwise_equal",
        "sigma_bitwise_equal",
        "frontier_bitwise_equal",
        "particle_ids_equal",
        "lineage_equal",
        "frontier_fact_identity_equal",
    ):
        assert m0[field] is True

    m1 = summary["m1_three_and_four_release"]
    assert m1["release_count"] == 4
    assert m1["three_release_blocker_closed"] is True
    assert m1["four_release_not_hard_coded"] is True
    assert [row["expected_mode"] for row in m1["releases"]] == [
        "first",
        "continuous",
        "continuous",
        "continuous",
    ]
    assert m1["releases"][-1]["release_slice_steps"] == [1, 2, 3, 4]

    m2 = summary["m2_lifecycle_and_partial_activity"]
    assert m2["sequence"] == ["first", "inactive", "restart", "continuous"]
    assert m2["inactive_old_cloud_advanced_without_phantom_release"] is True
    partial = m2["partial_activity_pre_gates"]
    assert [row["pattern"] for row in partial["patterns"]] == [
        "split",
        "shrink",
        "grow",
    ]
    assert all(row["adapter_state_unchanged"] for row in partial["patterns"])
    assert all(row["passed"] for row in partial["patterns"])


def test_one_real_straight_configuration_has_three_additive_releases(
    straight_configuration: gate._ConfigurationResult,
) -> None:
    row = straight_configuration.summary
    releases = row["releases"]
    assert row["passed"] is True
    assert row["geometry"] == "straight"
    assert row["release_count"] == gate.M3_RELEASE_COUNT == 3
    assert [release["expected_mode"] for release in releases] == [
        "first",
        "continuous",
        "continuous",
    ]
    assert releases[-1]["release_slice_steps"] == [1, 2, 3]
    for release in releases:
        assert release["total_particle_count"] == (
            release["previous_particle_count"] + release["new_particle_count"]
        )
        assert release["source_passed"] is True
        assert release["placement"]["passed"] is True
        assert release["ribbon_handoff"]["passed"] is True
        assert release["cell_strength_owner_passed"] is True
        assert release["release_slice_ledger_passed"] is True
        assert release["counters_passed"] is True
        counters = release["transport_counters"]
        for name in (
            "sort_count",
            "weld_count",
            "delete_count",
            "cancel_count",
            "remesh_count",
            "feedback_call_count",
            "parent_write_count",
            "surface_channel_write_count",
        ):
            assert counters[name] == 0


def test_fake_grid_executes_exactly_36_configurations_and_108_releases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, calls = _fake_gate_result(monkeypatch)
    assert len(calls) == 36
    assert set(calls) == {
        (geometry, spacing, substeps)
        for geometry in gate.CASE_GEOMETRIES
        for spacing in gate.TARGET_SPACINGS_M
        for substeps in gate.TRANSPORT_SUBSTEPS
    }
    assert result["passed"] is True
    assert result["status"] == "go_m3_mechanics_only"
    m3 = result["summary"]["m3"]
    assert m3["configuration_count"] == 36
    assert m3["release_count_per_configuration"] == 3
    assert (
        sum(
            configuration["release_count"]
            for geometry in m3["geometries"]
            for configuration in geometry["configurations"]
        )
        == 108
    )
    assert all(geometry["passed"] for geometry in m3["geometries"])


def test_any_failed_configuration_strictly_stops_the_fake_grid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, calls = _fake_gate_result(monkeypatch, failing_geometry="taper")
    assert len(calls) == 36
    assert result["passed"] is False
    assert result["status"] == "stop"
    assert result["summary"]["passed"] is False
    geometries = result["summary"]["m3"]["geometries"]
    assert (
        next(row for row in geometries if row["geometry"] == "taper")["passed"] is False
    )


def test_minimal_smoke_only_is_a_complete_stop_artifact_without_m3_coverage(
    tmp_path: Path,
    minimal_stop_result: dict[str, Any],
) -> None:
    result = minimal_stop_result
    assert result["status"] == "stop"
    assert result["passed"] is False
    assert result["summary"]["execution_mode"] == ("minimal_smoke_artifact_validation")
    assert result["summary"]["pre_gates"]["passed"] is True
    assert result["summary"]["m3"] is None
    gates = result["summary"]["gate_summary"]
    assert gates["pre_gates_passed"] is True
    assert gates["configuration_coverage_passed"] is False
    assert gates["m3_mechanics_and_refinement_passed"] is False
    assert gates["stop_required"] is True

    output = tmp_path / "minimal-stop"
    provenance = _provenance(
        output,
        run_uuid="00000000-0000-4000-8000-000000000010",
        minute=10,
    )
    assert (
        gate.write_run_artifacts(result, output, run_provenance=provenance)
        == output.resolve()
    )
    assert {path.name for path in output.iterdir()} == gate.ARTIFACT_FILENAMES
    summary = gate._load_strict_json(output / "summary.json")
    raw = gate._load_strict_json(output / gate.RAW_REFINEMENT_ARTIFACT)
    recomputed = gate.recompute_gates_from_artifacts(output)
    assert summary["status"] == "stop"
    assert raw["execution_mode"] == "minimal_smoke_artifact_validation"
    assert isinstance(raw["pre_gates"], dict)
    assert raw["m3_configurations"] == []
    assert recomputed["configuration_count"] == 0
    assert recomputed["configuration_coverage_passed"] is False
    assert recomputed["gate_summary"]["configuration_coverage_passed"] is False
    assert recomputed["reported_values_match_raw_recomputation"] is True
    assert recomputed["passed"] is False
    assert gate.verify_semantic_manifest(output)


def test_artifacts_recompute_from_strict_raw_and_separate_run_provenance(
    tmp_path: Path,
    fake_full_result: dict[str, Any],
) -> None:
    assert fake_full_result["passed"] is True
    runtime = fake_full_result["runtime_boundary_evidence"]
    instrumented = runtime["instrumented_access"]
    assert instrumented["guarded_symbols"]
    assert instrumented["guarded_call_counts"]
    assert not any(instrumented["guarded_call_counts"].values())
    assert instrumented["target_observation_read_count"] == 0
    assert runtime["direct_forbidden_imports"] == []
    assert runtime["surface_quantity_key_paths"] == []
    assert not any(runtime["measured_component_counts"].values())
    assert runtime["passed"] is True
    assert fake_full_result["summary"]["surface_solver_call_count"] == 0
    assert fake_full_result["summary"]["surface_channel_call_count"] == 0
    assert fake_full_result["summary"]["paper_comparison_call_count"] == 0
    left_target = tmp_path / "left"
    right_target = tmp_path / "right"
    left_provenance = _provenance(
        left_target,
        run_uuid="00000000-0000-4000-8000-000000000001",
        minute=1,
    )
    right_provenance = _provenance(
        right_target,
        run_uuid="00000000-0000-4000-8000-000000000002",
        minute=2,
    )
    left = gate.write_run_artifacts(
        fake_full_result,
        left_target,
        run_provenance=left_provenance,
    )
    right = gate.write_run_artifacts(
        fake_full_result,
        right_target,
        run_provenance=right_provenance,
    )
    assert {path.name for path in left.iterdir()} == gate.ARTIFACT_FILENAMES
    assert {path.name for path in right.iterdir()} == gate.ARTIFACT_FILENAMES

    deterministic_files = {
        *gate.SEMANTIC_PAYLOAD_FILENAMES,
        gate.SEMANTIC_MANIFEST_ARTIFACT,
    }
    for name in deterministic_files:
        assert (left / name).read_bytes() == (right / name).read_bytes()
    for name in {"run_manifest.json", "result_manifest.json", "run.log"}:
        assert (left / name).read_bytes() != (right / name).read_bytes()
    assert (left / "SHA256SUMS").read_bytes() != (right / "SHA256SUMS").read_bytes()

    semantic_sha = gate.verify_semantic_manifest(left)
    assert gate.verify_semantic_manifest(right) == semantic_sha
    raw = gate._load_strict_json(left / gate.RAW_REFINEMENT_ARTIFACT)
    assert raw["schema_id"] == gate.RAW_REFINEMENT_SCHEMA_ID
    assert len(raw["m3_configurations"]) == gate.MAX_CONFIGURATIONS
    first_array = raw["m3_configurations"][0]["evidence"]["positions_gp1_m"]
    assert set(first_array) == {"encoding", "dtype", "shape", "values", "sha256"}
    assert first_array["dtype"] == "<f8"
    decoded = gate._decode_raw_value("first", first_array)
    assert isinstance(decoded, np.ndarray)
    assert decoded.shape == tuple(first_array["shape"])

    recomputed = gate.recompute_gates_from_artifacts(left)
    assert recomputed == gate._load_strict_json(left / gate.RECOMPUTED_GATE_ARTIFACT)
    assert recomputed["configuration_count"] == 36
    assert recomputed["configuration_coverage_passed"] is True
    assert recomputed["reported_values_match_raw_recomputation"] is True
    assert recomputed["passed"] is True

    source_manifest = gate._load_strict_json(left / "source_manifest.json")
    declared_paths = {row["path"] for row in source_manifest["files"].values()}
    assert {
        "platform/forward_flight_benchmarks/ldvm_uvlm_correction.py",
        "platform/ldvm_fourier.py",
        "platform/flap_ldvm.py",
    }.issubset(declared_paths)
    assert source_manifest["runtime_import_closure_complete"] is False

    left_run = gate._load_strict_json(left / "run_manifest.json")
    right_run = gate._load_strict_json(right / "run_manifest.json")
    assert left_run["semantic_result_sha256"] == right_run["semantic_result_sha256"]
    assert left_run["provenance"]["run_uuid"] != right_run["provenance"]["run_uuid"]
    assert left_run["provenance"]["output_dir"] == str(left.resolve())
    assert right_run["provenance"]["output_dir"] == str(right.resolve())
    assert left_run["provenance"]["environment"]["packages"]["numpy"] == (
        np.__version__
    )
    serialized = (left / "summary.json").read_text(encoding="utf-8")
    assert "NaN" not in serialized and "Infinity" not in serialized

    hostile = gate._load_strict_json(right / gate.SEMANTIC_MANIFEST_ARTIFACT)
    hostile["files"]["../summary.json"] = "0" * 64
    _write_json(right / gate.SEMANTIC_MANIFEST_ARTIFACT, hostile)
    with pytest.raises(ValueError, match="exact payload set"):
        gate.verify_semantic_manifest(right)

    raw["m3_configurations"][0]["evidence"]["positions_gp1_m"]["values"][0][0] += 1.0
    _write_json(left / gate.RAW_REFINEMENT_ARTIFACT, raw)
    with pytest.raises(ValueError, match="raw array SHA-256 mismatch"):
        gate.recompute_gates_from_artifacts(left)


def test_disk_recomputation_fails_closed_on_summary_and_gate_tampering(
    tmp_path: Path,
    fake_full_result: dict[str, Any],
) -> None:
    base = tmp_path / "base"
    provenance = _provenance(
        base,
        run_uuid="00000000-0000-4000-8000-000000000020",
        minute=20,
    )
    gate.write_run_artifacts(fake_full_result, base, run_provenance=provenance)

    config_attack = tmp_path / "config-attack"
    shutil.copytree(base, config_attack)
    config_summary = gate._load_strict_json(config_attack / "summary.json")
    config_raw = gate._load_strict_json(config_attack / gate.RAW_REFINEMENT_ARTIFACT)
    configuration, raw_record = _first_configuration_pair(config_summary, config_raw)
    configuration["passed"] = False
    _rebind_configuration_summary(configuration, raw_record)
    _write_json(config_attack / "summary.json", config_summary)
    _write_json(config_attack / gate.RAW_REFINEMENT_ARTIFACT, config_raw)
    recomputed = gate.recompute_gates_from_artifacts(config_attack)
    assert recomputed["reported_values_match_raw_recomputation"] is False
    assert recomputed["passed"] is False

    edge_attack = tmp_path / "edge-attack"
    shutil.copytree(base, edge_attack)
    edge_summary = gate._load_strict_json(edge_attack / "summary.json")
    edge_raw = gate._load_strict_json(edge_attack / gate.RAW_REFINEMENT_ARTIFACT)
    configuration, raw_record = _first_configuration_pair(edge_summary, edge_raw)
    configuration["releases"][0]["edge_metrics"]["passed"] = False
    _rebind_configuration_summary(configuration, raw_record)
    _write_json(edge_attack / "summary.json", edge_summary)
    _write_json(edge_attack / gate.RAW_REFINEMENT_ARTIFACT, edge_raw)
    recomputed = gate.recompute_gates_from_artifacts(edge_attack)
    assert recomputed["gate_summary"]["m3_mechanics_and_refinement_passed"] is False
    assert recomputed["reported_values_match_raw_recomputation"] is False
    assert recomputed["passed"] is False

    refinement_attack = tmp_path / "refinement-attack"
    shutil.copytree(base, refinement_attack)
    refinement_summary = gate._load_strict_json(refinement_attack / "summary.json")
    reported_geometry = refinement_summary["summary"]["m3"]["geometries"][0]
    reported_geometry["time_refinement_by_h"][0]["frontier_minus_latest_birth"][
        "passed"
    ] = False
    reported_geometry["h_refinement_at_finest_time"]["fixed_probe_induced_velocity"][
        "passed"
    ] = False
    _write_json(refinement_attack / "summary.json", refinement_summary)
    recomputed = gate.recompute_gates_from_artifacts(refinement_attack)
    assert (
        recomputed["geometries"][0]["time_refinement_by_h"][0][
            "frontier_minus_latest_birth"
        ]["passed"]
        is True
    )
    assert (
        recomputed["geometries"][0]["h_refinement_at_finest_time"][
            "fixed_probe_induced_velocity"
        ]["passed"]
        is True
    )
    assert recomputed["reported_values_match_raw_recomputation"] is False
    assert recomputed["passed"] is False


def test_writer_is_nonoverwriting_transactional_and_rejects_bad_provenance(
    tmp_path: Path,
    fake_full_result: dict[str, Any],
) -> None:
    output = tmp_path / "safe-output"
    provenance = _provenance(
        output,
        run_uuid="00000000-0000-4000-8000-000000000030",
        minute=30,
    )
    gate.write_run_artifacts(fake_full_result, output, run_provenance=provenance)
    with pytest.raises(FileExistsError, match="refusing to reuse"):
        gate.write_run_artifacts(
            fake_full_result,
            output,
            run_provenance=provenance,
        )

    mismatch_target = tmp_path / "mismatched-provenance"
    with pytest.raises(ValueError, match="not the write target"):
        gate.write_run_artifacts(
            fake_full_result,
            mismatch_target,
            run_provenance=provenance,
        )
    assert not mismatch_target.exists()

    nonfinite_target = tmp_path / "nonfinite"
    nonfinite_provenance = _provenance(
        nonfinite_target,
        run_uuid="00000000-0000-4000-8000-000000000031",
        minute=31,
    )
    forged = dict(fake_full_result)
    forged["forged_nonfinite"] = float("nan")
    with pytest.raises(ValueError, match="Out of range float values"):
        gate.write_run_artifacts(
            forged,
            nonfinite_target,
            run_provenance=nonfinite_provenance,
        )
    assert not nonfinite_target.exists()
    assert not list(tmp_path.glob(".nonfinite.tmp-*"))


def test_cli_minimal_smoke_writes_stop_bundle_and_exits_one(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    minimal_stop_result: dict[str, Any],
) -> None:
    output = tmp_path / "cli-stop"
    calls: list[bool] = []

    def fake_run(
        config: gate.GateConfig = gate.GateConfig(),
        *,
        minimal_smoke_only: bool = False,
    ) -> dict[str, Any]:
        assert config == gate.GateConfig()
        calls.append(minimal_smoke_only)
        return minimal_stop_result

    monkeypatch.setattr(gate, "run_cumulative_cloud_gate", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_v5h_cumulative_cloud_gate.py",
            "--output-dir",
            str(output),
            "--minimal-smoke",
        ],
    )
    with pytest.raises(SystemExit) as exit_info:
        gate.main()
    assert exit_info.value.code == 1
    assert calls == [True]
    assert {path.name for path in output.iterdir()} == gate.ARTIFACT_FILENAMES
    assert gate._load_strict_json(output / "summary.json")["status"] == "stop"
    assert gate.recompute_gates_from_artifacts(output)["passed"] is False
    printed = json.loads(capsys.readouterr().out)
    assert printed == {"artifact": str(output.resolve()), "status": "stop"}


def test_runner_source_and_runtime_guard_target_ptera_and_load_boundaries(
    monkeypatch: pytest.MonkeyPatch,
    minimal_smoke: dict[str, Any],
    straight_configuration: gate._ConfigurationResult,
) -> None:
    source = Path(gate.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported_modules = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported_modules.update(
        node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    )
    assert not any(
        token in module.casefold()
        for module in imported_modules
        for token in ("baik", "yang", "izraelevitz", "ptera")
    )
    assert gate._direct_forbidden_imports() == []

    original_open = builtins.open

    def guarded_open(file: object, *args: object, **kwargs: object) -> object:
        folded = str(file).casefold()
        if any(
            token in folded for token in ("baik", "yang", "izraelevitz", "ground_truth")
        ):
            raise AssertionError("target observation access attempted")
        return original_open(file, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", guarded_open)
    assert minimal_smoke["passed"] is True
    assert straight_configuration.summary["passed"] is True
    assert not gate._surface_quantity_key_paths(minimal_smoke)
    for release in straight_configuration.summary["releases"]:
        assert release["transport_counters"]["surface_channel_write_count"] == 0

    import forward_flight_benchmarks as benchmark_package

    original = benchmark_package.build_baik_movement
    instrumentation = gate._RuntimeBoundaryInstrumentation()
    with instrumentation:
        with pytest.raises(RuntimeError, match="forbidden runtime boundary call"):
            benchmark_package.build_baik_movement()
    evidence = instrumentation.evidence()
    assert evidence["guarded_call_counts"]["target_builder"] == 1
    assert evidence["passed"] is False
    assert benchmark_package.build_baik_movement is original


def test_runtime_boundary_accepts_namespace_isolated_zero_guard_path(
    fake_full_result: dict[str, Any],
) -> None:
    isolated = deepcopy(fake_full_result["runtime_boundary_evidence"])
    isolated["package_init_eager_target_or_ptera_definitions_loaded"] = False
    isolated["instrumented_access"]["guarded_symbols"] = []
    isolated["instrumented_access"]["guarded_call_counts"] = {}
    assert gate._runtime_boundary_from_evidence(isolated) == (True, True)

    target_read = deepcopy(isolated)
    target_read["instrumented_access"]["target_observation_read_count"] = 1
    target_read["instrumented_access"]["target_observation_read_paths"] = [
        "/forbidden/baik/ground_truth.json"
    ]
    assert gate._runtime_boundary_from_evidence(target_read) == (False, False)

    feedback_write = deepcopy(isolated)
    feedback_write["measured_component_counts"]["feedback_call_count"] = 1
    assert gate._runtime_boundary_from_evidence(feedback_write) == (False, False)


def test_strict_json_rejects_duplicate_keys_and_nonfinite_tokens(
    tmp_path: Path,
) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"x": 1, "x": 2}', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON key"):
        gate._load_strict_json(duplicate)

    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_text('{"x": NaN}', encoding="utf-8")
    with pytest.raises(ValueError, match="non-finite JSON token"):
        gate._load_strict_json(nonfinite)

    top_level_array = tmp_path / "array.json"
    top_level_array.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="top-level object"):
        gate._load_strict_json(top_level_array)
