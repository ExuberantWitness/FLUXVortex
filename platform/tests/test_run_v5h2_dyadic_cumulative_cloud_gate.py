from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from forward_flight_benchmarks.run_v5h_cumulative_cloud_gate import (
    _run_configuration as run_v2_configuration,
)
from forward_flight_benchmarks.run_v5h2_dyadic_cumulative_cloud_gate import (
    BASE_TARGET_SPACING_M,
    CASE_GEOMETRIES,
    CONFIGURATION_COUNT,
    DyadicGateConfig,
    REFINEMENT_LEVELS,
    RELEASE_COUNT,
    SIGMA_BIRTH_M,
    SPAN_CELLS,
    TRANSPORT_SUBSTEPS,
    _array_sha256,
    _run_release_sequence,
    recompute_dyadic_gate_from_raw,
    run_minimal_smoke,
    verify_dyadic_gate_artifacts,
    write_dyadic_gate_artifacts,
)


def _fake_raw() -> dict[str, object]:
    records = []
    for geometry_index, geometry in enumerate(CASE_GEOMETRIES):
        for level in REFINEMENT_LEVELS:
            nominal = BASE_TARGET_SPACING_M / (2**level)
            final_count = 30 * (2**level)
            new_count = final_count // RELEASE_COUNT
            for substeps in TRANSPORT_SUBSTEPS:
                offset = (
                    0.1 * nominal**2
                    + 1.0e-6 / (substeps**3)
                    + geometry_index * 1.0e-8
                )
                frontier = np.ones((SPAN_CELLS + 1, 3), dtype=np.float64)
                probes = np.full((5, 3), 2.0, dtype=np.float64)
                frontier[:, 0] += offset
                probes[:, 1] += 0.5 * offset
                positions = np.column_stack(
                    (
                        np.linspace(0.0, 1.0, final_count),
                        np.zeros(final_count),
                        np.zeros(final_count),
                    )
                )
                gamma = np.full((final_count, 3), 0.01, dtype=np.float64)
                sigma = np.full(final_count, SIGMA_BIRTH_M, dtype=np.float64)
                releases = []
                for step in range(1, RELEASE_COUNT + 1):
                    panel_count = 10 * (2**level)
                    releases.append(
                        {
                            "source_step_index": step,
                            "new_particle_count": new_count,
                            "passed": True,
                            "sidecar_passed": True,
                            "cloud_sha256": f"{step:064x}",
                            "report_sha256": f"{step + 10:064x}",
                            "sidecar": {
                                "release_index": step,
                                "source_step_index": step,
                                "refinement_level": level,
                                "refinement_multiplier": 2**level,
                                "predicted_total_particle_count": new_count,
                                "edge_panels": [
                                    {
                                        "base_panel_count": 10,
                                        "panel_count": panel_count,
                                        "refinement_level": level,
                                        "smoothing_radius_m": SIGMA_BIRTH_M,
                                        "realized_spacing_m": nominal * 0.9,
                                    }
                                ],
                            },
                        }
                    )
                records.append(
                    {
                        "geometry": geometry,
                        "span_cells": SPAN_CELLS,
                        "refinement_level": level,
                        "base_target_spacing_m": BASE_TARGET_SPACING_M,
                        "nominal_target_spacing_m": nominal,
                        "transport_substeps": substeps,
                        "smoothing_radius_m": SIGMA_BIRTH_M,
                        "release_count": RELEASE_COUNT,
                        "particle_count": final_count,
                        "realized_spacing_max_m": nominal * 0.9,
                        "frontier_minus_latest_birth_gp1_m": frontier.tolist(),
                        "fixed_probe_induced_velocity_gp1_m_per_s": probes.tolist(),
                        "positions_gp1_m": positions.tolist(),
                        "gamma_vector_m3_per_s": gamma.tolist(),
                        "sigma_m": sigma.tolist(),
                        "state_and_observable_sha256": _array_sha256(
                            positions, gamma, sigma, frontier, probes
                        ),
                        "final_cloud_sha256": "a" * 64,
                        "final_report_sha256": "b" * 64,
                        "releases": releases,
                        "mechanics_passed": True,
                    }
                )
    return {
        "schema_id": "fluxv-v5h2-dyadic-raw-refinement-v1",
        "configuration_count": CONFIGURATION_COUNT,
        "release_count": CONFIGURATION_COUNT * RELEASE_COUNT,
        "configurations": records,
    }


def _fake_result() -> dict[str, object]:
    raw = _fake_raw()
    first = recompute_dyadic_gate_from_raw(raw)
    summary = {
        "schema_id": "fluxv-v5h2-dyadic-cumulative-cloud-gate-v1",
        "evaluation_mode": "simulation_only",
        "target_observation_access": "none",
        "paper_scoring_performed": False,
        "configuration_count": CONFIGURATION_COUNT,
        "release_count": CONFIGURATION_COUNT * RELEASE_COUNT,
        "base_target_spacing_m": BASE_TARGET_SPACING_M,
        "refinement_levels": list(REFINEMENT_LEVELS),
        "nominal_target_spacings_m": [
            BASE_TARGET_SPACING_M / (2**level) for level in REFINEMENT_LEVELS
        ],
        "transport_substeps": list(TRANSPORT_SUBSTEPS),
        "geometries": [
            {
                "geometry": item["geometry"],
                "configuration_mechanics_passed": item[
                    "configuration_mechanics_passed"
                ],
                "time_refinement_by_level": item["time_refinement_by_level"],
                "h_refinement_at_finest_time": item["h_refinement_at_finest_time"],
                "passed": item["passed"],
            }
            for item in first["geometries"]
        ],
        "passed": True,
        "status": "go_v5h2_mechanics_only",
    }
    recomputed = recompute_dyadic_gate_from_raw(raw, summary)
    assert recomputed["passed"]
    return {
        "summary": summary,
        "raw_refinement": raw,
        "recomputed_gates": recomputed,
    }


def test_frozen_grid_is_exactly_36_configurations_and_108_releases() -> None:
    config = DyadicGateConfig()
    assert (
        len(config.geometries)
        * len(config.refinement_levels)
        * len(config.transport_substeps)
        == CONFIGURATION_COUNT
    )
    assert CONFIGURATION_COUNT * config.release_count == 108
    with pytest.raises(ValueError, match="frozen"):
        from forward_flight_benchmarks.run_v5h2_dyadic_cumulative_cloud_gate import (
            _validate_config,
        )

        _validate_config(DyadicGateConfig(span_cells=3))


def test_minimal_smoke_runs_three_real_releases() -> None:
    result = run_minimal_smoke()
    assert result["passed"]
    assert result["configuration"]["release_count"] == RELEASE_COUNT
    assert all(item["passed"] for item in result["configuration"]["releases"])


def test_level_zero_full_geometry_time_matrix_is_bitwise_v2_reduction() -> None:
    for geometry in CASE_GEOMETRIES:
        for substeps in TRANSPORT_SUBSTEPS:
            old = run_v2_configuration(
                geometry,
                span_cells=SPAN_CELLS,
                target_spacing_m=BASE_TARGET_SPACING_M,
                transport_substeps=substeps,
            )
            wing_id = (
                f"cumulative-m3:{geometry}:h={BASE_TARGET_SPACING_M.hex()}:"
                f"substeps={substeps}:wing"
            )
            new = _run_release_sequence(
                geometry,
                refinement_level=0,
                transport_substeps=substeps,
                wing_id=wing_id,
            )
            assert np.array_equal(
                old.raw["positions_gp1_m"], new.raw["positions_gp1_m"]
            )
            assert np.array_equal(
                old.raw["gamma_vector_m3_per_s"],
                new.raw["gamma_vector_m3_per_s"],
            )
            assert np.array_equal(old.raw["sigma_m"], new.raw["sigma_m"])
            assert np.array_equal(
                old.frontier_minus_latest_birth,
                new.frontier_minus_latest_birth,
            )
            assert np.array_equal(
                old.fixed_probe_induced_velocity,
                new.fixed_probe_induced_velocity,
            )
            assert old.raw["particle_ids"] == new.raw["particle_ids"]
            assert old.raw["lineage"] == new.raw["lineage"]
            assert old.raw["release_slices"] == new.raw["release_slices"]
            assert old.particle_count == new.particle_count


def test_disk_recomputation_uses_raw_arrays_and_dyadic_panel_ledgers() -> None:
    result = _fake_result()
    recomputed = recompute_dyadic_gate_from_raw(
        result["raw_refinement"], result["summary"]
    )
    assert recomputed["passed"]
    assert recomputed["configuration_count"] == CONFIGURATION_COUNT
    for geometry in recomputed["geometries"]:
        assert geometry["configuration_mechanics_passed"]
        assert all(gate["passed"] for gate in geometry["time_refinement_by_level"])
        assert geometry["h_refinement_at_finest_time"]["passed"]


def test_raw_array_or_panel_tampering_fails_closed() -> None:
    result = _fake_result()
    raw = deepcopy(result["raw_refinement"])
    raw["configurations"][0]["frontier_minus_latest_birth_gp1_m"][0][0] += 1.0
    with pytest.raises(ValueError, match="digest"):
        recompute_dyadic_gate_from_raw(raw, result["summary"])

    raw = deepcopy(result["raw_refinement"])
    raw["configurations"][0]["releases"][0]["sidecar"]["edge_panels"][0][
        "panel_count"
    ] += 1
    recomputed = recompute_dyadic_gate_from_raw(raw, result["summary"])
    assert not recomputed["passed"]
    assert recomputed["status"] == "stop_v5h2_artifact_recomputation"


def test_artifact_writer_is_strict_and_semantic_payload_is_deterministic(
    tmp_path: Path,
) -> None:
    result = _fake_result()
    left = tmp_path / "left"
    right = tmp_path / "right"
    left_verification = write_dyadic_gate_artifacts(result, left)
    right_verification = write_dyadic_gate_artifacts(result, right)
    assert left_verification["passed"] and right_verification["passed"]
    assert (
        left_verification["semantic_digest_sha256"]
        == right_verification["semantic_digest_sha256"]
    )
    for name in (
        "README.md",
        "summary.json",
        "raw_refinement.json",
        "recomputed_gates.json",
        "source_manifest.json",
        "semantic_manifest.json",
    ):
        assert (left / name).read_bytes() == (right / name).read_bytes()
    assert (left / "run_manifest.json").read_bytes() != (
        right / "run_manifest.json"
    ).read_bytes()
    assert verify_dyadic_gate_artifacts(left)["passed"]
