from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from fluxvortex.rvpm_edge_bridge import FROZEN_OVERLAP_LAMBDA
from forward_flight_benchmarks.run_v5h_manufactured_shadow_gate import (
    CASE_GEOMETRIES,
    RUN_SCHEMA_ID,
    GateConfig,
    run_mechanical_shadow_gate,
    write_smoke_artifact,
)


@pytest.fixture(scope="module")
def gate_result() -> dict[str, object]:
    return run_mechanical_shadow_gate()


def test_s0_disabled_and_source_attestation_scope_are_exact(
    gate_result: dict[str, object],
) -> None:
    assert gate_result["schema_id"] == RUN_SCHEMA_ID
    assert gate_result["scope"] == "non_target_single_release_mechanical_shadow_only"
    assert gate_result["observation_access"] == "none"
    assert gate_result["target_case_branch"] == "none"
    assert gate_result["source_kind"] == "direct_V5hDVMSource_event"
    assert gate_result["source_canonical_eligible"] is False
    assert gate_result["frozen_overlap_lambda"] == FROZEN_OVERLAP_LAMBDA
    deposition = gate_result["spatial_deposition_contract"]
    assert deposition["status"] == "development_transfer_not_canonical"
    assert deposition["release_clock_decoupled"] is True
    assert deposition["observation_fit"] == "none"
    assert "not a universal rVPM wake-core law" in deposition["overlap_role"]
    stability = gate_result["transport_stability_contract"]
    assert stability["clip_or_limiter"] == "none"
    assert "Jacobian" in stability["primary_explicit_integrator_indicator"]
    assert gate_result["parent_write_count"] == 0
    assert gate_result["load_write_count"] == 0
    assert gate_result["feedback_call_count"] == 0

    disabled = gate_result["disabled_gate"]
    assert isinstance(disabled, dict)
    assert disabled == {
        "input_blind": True,
        "feedback_call_count": 0,
        "transport_advance_count": 0,
        "state_unchanged": True,
        "passed": True,
    }

    hashes = gate_result["code_sha256"]
    assert isinstance(hashes, dict)
    assert set(hashes) == {
        "runner",
        "dvm_source",
        "dvm_node_ribbon",
        "ldvm_fourier",
        "ldvm_uvlm_correction",
        "rvpm_edge_bridge",
        "rvpm_reference",
        "rvpm_transport",
    }
    assert all(isinstance(value, str) and len(value) == 64 for value in hashes.values())


def test_s1_direct_first_release_ribbons_pass_mechanical_gates(
    gate_result: dict[str, object],
) -> None:
    cases = gate_result["cases"]
    assert isinstance(cases, list)
    assert [case["geometry"] for case in cases] == list(CASE_GEOMETRIES)

    for case in cases:
        assert case["source_event_count"] == 4
        assert case["source_step_indices"] == [1, 1, 1, 1]
        assert case["all_direct_first_lev_release"] is True
        assert case["max_kelvin_residual_m2_per_s"] <= 1.0e-10
        assert case["parent_write_count"] == 0
        assert case["load_write_count"] == 0
        assert case["feedback_call_count"] == 0
        assert case["mapping_passed"] is True

        ribbon = case["ribbon"]
        assert ribbon["active_cell_count"] == 4
        assert ribbon["shared_node_count"] == 5
        assert ribbon["first_node_count"] == 5
        assert ribbon["continuous_node_count"] == 0
        assert ribbon["restart_node_count"] == 0
        assert ribbon["incidence_residual"] <= 1.0e-12
        assert ribbon["edge_reconstruction_residual"] <= 1.0e-12
        assert ribbon["seam_count"] == 0
        assert ribbon["nonfinite_count"] == 0
        assert ribbon["source_reuse_count"] == 0
        assert ribbon["feedback_call_count"] == 0
        assert ribbon["transport_advance_count"] == 0


def test_fixed_sigma_spatial_refinement_is_conservative_and_source_aligned(
    gate_result: dict[str, object],
) -> None:
    cases = gate_result["cases"]
    assert isinstance(cases, list)
    for case in cases:
        rows = case["subdivision_refinement"]
        assert [row["span_subdivisions"] for row in rows] == [2, 4, 8, 16]
        counts = [row["particle_count"] for row in rows]
        assert counts[0] < counts[1] < counts[2] < counts[3]
        radii = [row["smoothing_radius_m"] for row in rows]
        assert radii[1] == 0.5 * radii[0]
        assert radii[2] == 0.5 * radii[1]
        assert radii[3] == 0.5 * radii[2]
        errors = [row["analytic_finite_segment_probe_relative_l2"] for row in rows]
        assert errors[-1] <= 0.01
        assert all(
            coarse / fine >= 1.5 for coarse, fine in zip(errors[:-1], errors[1:])
        )
        assert case["spatial_probe_convergence_passed"] is True
        assert len(case["analytic_probe_positions_gp1_m"]) == 5
        for row in rows:
            assert row["passed"] is True
            assert row["finite"] is True
            assert row["overlap_lower_bound_2p125"] is True
            assert row["realized_overlap_min"] >= FROZEN_OVERLAP_LAMBDA * (
                1.0 - 1.0e-14
            )
            assert row["realized_overlap_max"] >= row["realized_overlap_min"]
            assert row["fixed_sigma_across_edges"] is True
            assert row["sigma_min_m"] == row["smoothing_radius_m"]
            assert row["sigma_max_m"] == row["smoothing_radius_m"]
            assert (
                row["smoothing_radius_m"]
                == FROZEN_OVERLAP_LAMBDA * row["target_max_spacing_m"]
            )
            assert row["incidence_residual"] <= 1.0e-12
            assert row["edge_reconstruction_residual"] <= 1.0e-12
            assert row["max_edge_conservation_abs"] <= 1.0e-14
            assert row["max_edge_conservation_rel"] <= 1.0e-12
            assert row["global_conservation_abs"] <= 1.0e-14
            assert row["global_conservation_rel"] <= 1.0e-12
            assert row["sigma_min_m"] > 0.0
            assert row["clip_count"] == 0
            assert row["nonfinite_count"] == 0
            assert row["owner_conflict_count"] == 0
            assert row["feedback_call_count"] == 0


def test_s3_single_cloud_transport_is_finite_and_fail_closed_on_drift(
    gate_result: dict[str, object],
) -> None:
    cases = gate_result["cases"]
    assert isinstance(cases, list)
    for case in cases:
        rows = case["single_cloud_transport_substeps"]
        assert [row["transport_substeps"] for row in rows] == [1, 2, 4]
        assert all(row["finite"] is True for row in rows)
        assert all(row["sigma_min_m"] > 0.0 for row in rows)
        for row in rows:
            expected_pass = bool(
                row["finite"] and row["rvpm_gamma_sigma2_relative_drift_max"] <= 1.0e-6
            )
            assert row["passed"] is expected_pass
            assert np.isfinite(row["max_stage_dt_self_velocity_over_sigma"])
            assert np.isfinite(row["max_stage_dt_velocity_over_sigma"])
            assert np.isfinite(row["max_stage_dt_jacobian_frobenius"])

    expected_overall = bool(
        gate_result["disabled_gate"]["passed"] and all(case["passed"] for case in cases)
    )
    assert gate_result["passed"] is expected_overall
    assert gate_result["status"] == ("passed" if expected_overall else "failed")
    summary = gate_result["gate_summary"]
    assert summary["s0_exact_off_and_attestation_passed"] is True
    assert summary["s1_single_release_ribbon_and_deposition_passed"] is True
    assert summary["s3_single_cloud_transport_passed"] is bool(
        all(case["single_cloud_transport_passed"] for case in cases)
    )
    assert summary["stop_required"] is (not expected_overall)
    assert (
        gate_result["blocked"]["repeated_release_transport"]
        == "requires an audited advected NodeFrontierFact handoff"
    )


def test_custom_controls_validate_and_keep_a_single_source_layer() -> None:
    result = run_mechanical_shadow_gate(
        GateConfig(
            span_cells=2,
            span_subdivisions=(2, 4),
            transport_substeps=(1,),
        )
    )
    for case in result["cases"]:
        assert case["source_event_count"] == 2
        assert case["source_step_indices"] == [1, 1]
        assert len(case["single_cloud_transport_substeps"]) == 1

    with pytest.raises(ValueError, match="must include"):
        run_mechanical_shadow_gate(GateConfig(span_subdivisions=(2, 8)))
    with pytest.raises(ValueError, match="duplicates"):
        run_mechanical_shadow_gate(GateConfig(transport_substeps=(1, 1)))


def test_tmp_smoke_artifact_roundtrips_and_refuses_overwrite(
    tmp_path: Path,
    gate_result: dict[str, object],
) -> None:
    output = tmp_path / "v5h_single_release_shadow.json"
    written = write_smoke_artifact(gate_result, output)
    assert written == output
    loaded = json.loads(output.read_text(encoding="utf-8"))
    assert loaded == gate_result
    assert np.isfinite(loaded["cases"][0]["max_kelvin_residual_m2_per_s"])
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_smoke_artifact(gate_result, output)
