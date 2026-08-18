from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from fluxvortex.rvpm_reference import (
    direct_gaussian_erf_velocity_jacobian,
    unpack_julia_column_major,
)
from fluxvortex.rvpm_transport import (
    corrected_pedrizzetti,
    lsrk3_step_direct,
    make_particle_state,
)
from tools.v5h_flowvpm_oracle.evaluate_parity import evaluate

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_ROOT = (
    REPOSITORY_ROOT
    / "docs/forward_flight_large_pitch/reproductions"
    / "fluxv_v5c_nextgen_20260814/refine-logs/flowvpm_v5h/runs"
    / "20260814_fluxv_v5h_r0_r1_oracle_direct_v2"
)
RUN_ROOT = Path(os.environ.get("FLUXV_FLOWVPM_ORACLE_RUN_ROOT", DEFAULT_RUN_ROOT))
ORACLE_PATH = RUN_ROOT / "oracle/flowvpm_oracle_v2.json"
ORACLE_HDF5_PATH = RUN_ROOT / "oracle/flowvpm_oracle_v2.h5"
ARTIFACT_MANIFEST_PATH = RUN_ROOT / "artifact_manifest.json"
PROJECT_PATH = REPOSITORY_ROOT / "tools/v5h_flowvpm_oracle/Project.toml"
JULIA_MANIFEST_PATH = REPOSITORY_ROOT / "tools/v5h_flowvpm_oracle/Manifest.toml"
EXPORTER_PATH = REPOSITORY_ROOT / "tools/v5h_flowvpm_oracle/export_oracle.jl"
METRICS_PATH = RUN_ROOT / "metrics.json"
JULIA_J_NAMES = (
    "j11",
    "j21",
    "j31",
    "j12",
    "j22",
    "j32",
    "j13",
    "j23",
    "j33",
)
STATE_KEYS = {
    "x_x",
    "x_y",
    "x_z",
    "gamma_x",
    "gamma_y",
    "gamma_z",
    "sigma",
    *(f"m{index:02d}" for index in range(1, 10)),
}
FIELD_KEYS = {"u_x", "u_y", "u_z", *JULIA_J_NAMES}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _payload() -> dict:
    if not ORACLE_PATH.is_file():
        raise AssertionError(
            f"required frozen FLOWVPM oracle is missing: {ORACLE_PATH}"
        )
    return json.loads(ORACLE_PATH.read_text(encoding="utf-8"))


def _vectors(group: dict, prefix: str) -> np.ndarray:
    return np.column_stack([group[f"{prefix}_{axis}"] for axis in "xyz"]).astype(
        np.float64
    )


def _jacobian(group: dict) -> np.ndarray:
    flat = np.column_stack([group[name] for name in JULIA_J_NAMES]).astype(np.float64)
    return unpack_julia_column_major(flat)


def _storage(group: dict) -> np.ndarray:
    return np.column_stack([group[f"m{index:02d}"] for index in range(1, 10)]).astype(
        np.float64
    )


def _relative_l2(actual: np.ndarray, reference: np.ndarray) -> float:
    denominator = float(np.linalg.norm(reference.ravel()))
    numerator = float(np.linalg.norm((actual - reference).ravel()))
    if denominator == 0.0:
        return numerator
    return numerator / denominator


def test_oracle_identity_and_declared_hashes_are_current() -> None:
    payload = _payload()
    metadata = payload["meta"]
    assert metadata["schema_version"] == "flowvpm_oracle_v2"
    assert metadata["source_commit"] == "4f433fb09f6baad25db65c9905e0d9cbb09663ce"
    assert metadata["source_tree"] == "ecb0fc0b7f7cda244cef695ff06ce23719ad1920"
    assert metadata["fastmultipole_commit"] == "adc4f26"
    assert metadata["fastmultipole_tree"] == "313cf60bed67629b1da6fb94b3b25394bd4f51ec"
    assert metadata["julia_version"] == "1.10.11"
    assert metadata["julia_threads"] == [1]
    assert metadata["blas_threads"] == [1]
    assert metadata["project_sha256"] == _sha256(PROJECT_PATH)
    assert metadata["manifest_sha256"] == _sha256(JULIA_MANIFEST_PATH)
    assert metadata["export_script_sha256"] == _sha256(EXPORTER_PATH)
    assert (
        metadata["stage_state_schema"] == "pre/state_only;rhs/UJ_only;post/state_only"
    )
    assert metadata["rhs_evaluation_state"] == "stage_pre"

    manifest = json.loads(ARTIFACT_MANIFEST_PATH.read_text(encoding="utf-8"))
    declared = manifest["artifacts"]
    for relative_path, path in (
        ("oracle/flowvpm_oracle_v2.json", ORACLE_PATH),
        ("oracle/flowvpm_oracle_v2.h5", ORACLE_HDF5_PATH),
        ("metrics.json", METRICS_PATH),
    ):
        assert declared[relative_path]["sha256"] == _sha256(path)
        assert declared[relative_path]["bytes"] == path.stat().st_size

    for relative_path, expected_hash in manifest["source_hashes"].items():
        assert _sha256(REPOSITORY_ROOT / relative_path) == expected_hash


def test_rk_stage_schema_separates_state_from_rhs_fields() -> None:
    fixtures = _payload()["fixtures"]
    for fixture_name in (
        "rk3_rvpm_direct_gauserf",
        "rk3_timevarying_uinf_direct_gauserf",
    ):
        fixture = fixtures[fixture_name]
        for step in range(1, 3):
            for stage in range(1, 4):
                record = fixture[f"step_{step:02d}"][f"stage_{stage:02d}"]
                assert set(record["pre"]) == STATE_KEYS
                assert set(record["post"]) == STATE_KEYS
                assert set(record["rhs"]) == FIELD_KEYS
                assert FIELD_KEYS.isdisjoint(record["pre"])
                assert FIELD_KEYS.isdisjoint(record["post"])
                assert STATE_KEYS.isdisjoint(record["rhs"])


def test_metrics_file_equals_a_fresh_evaluation_field_for_field() -> None:
    stored_metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    assert evaluate(_payload()) == stored_metrics


def test_mutated_rk_configuration_fails_the_machine_gate() -> None:
    mutated = copy.deepcopy(_payload())
    mutated["fixtures"]["rk3_rvpm_direct_gauserf"]["config"]["rk_a"][1] += 0.01
    metrics = evaluate(mutated)
    assert not metrics["config_contract"]["fixed_rk"]["rk_a_exact"]
    assert not metrics["gates"]["config_contract_all_true"]
    assert not metrics["overall_pass"]


def test_direct_gaussian_erf_velocity_and_jacobian_match_flowvpm() -> None:
    fixture = _payload()["fixtures"]["uj_direct_gauserf"]
    inputs = fixture["input"]
    reference = fixture["output"]
    actual = direct_gaussian_erf_velocity_jacobian(
        _vectors(inputs, "x"),
        _vectors(inputs, "gamma"),
        np.asarray(inputs["sigma"], dtype=np.float64),
    )
    reference_velocity = _vectors(reference, "u")
    reference_jacobian = _jacobian(reference)
    assert _relative_l2(actual.velocity, reference_velocity) <= 1e-12
    assert _relative_l2(actual.jacobian, reference_jacobian) <= 1e-11


def test_probe_only_and_nearfield_sweep_metrics_pass() -> None:
    metrics = evaluate(_payload())
    assert metrics["uj"]["probe_count"] == 3
    assert metrics["uj"]["probe_velocity_relative_l2"] <= 1e-12
    assert metrics["uj"]["probe_jacobian_relative_l2"] <= 1e-11
    assert metrics["nearfield_uj"]["probe_count"] == 9
    assert metrics["nearfield_uj"]["r_over_sigma_min"] == 1e-4
    assert metrics["nearfield_uj"]["r_over_sigma_max"] == 2.0
    assert metrics["nearfield_uj"]["probe_id_contract_exact"]
    assert metrics["gates"]["nearfield_probe_velocity_row_relative_l2_max_le_1e-9"]
    assert metrics["gates"]["nearfield_probe_jacobian_row_relative_l2_max_le_1e-9"]
    assert metrics["gates"]["nearfield_probe_contract_exact"]


def test_two_step_every_stage_rvpm_state_matches_flowvpm() -> None:
    fixture = _payload()["fixtures"]["rk3_rvpm_direct_gauserf"]
    inputs = fixture["input"]
    config = fixture["config"]
    state = make_particle_state(
        _vectors(inputs, "x"),
        _vectors(inputs, "gamma"),
        np.asarray(inputs["sigma"], dtype=np.float64),
    )
    delta_time = float(config["dt"][0])
    freestream = np.array(
        [config[f"uinf_{axis}"][0] for axis in "xyz"],
        dtype=np.float64,
    )

    largest_state_error = 0.0
    largest_rhs_error = 0.0
    for step in range(1, 3):
        state, stages = lsrk3_step_direct(
            state,
            delta_time,
            freestream_velocity=freestream,
        )
        for stage_index, stage in enumerate(stages, start=1):
            reference = fixture[f"step_{step:02d}"][f"stage_{stage_index:02d}"]
            for actual_state, reference_name, storage in (
                (stage.pre, "pre", stage.storage_pre),
                (stage.post, "post", stage.storage_post),
            ):
                reference_state = reference[reference_name]
                largest_state_error = max(
                    largest_state_error,
                    _relative_l2(
                        actual_state.positions, _vectors(reference_state, "x")
                    ),
                    _relative_l2(
                        actual_state.gamma, _vectors(reference_state, "gamma")
                    ),
                    _relative_l2(
                        actual_state.sigma,
                        np.asarray(reference_state["sigma"], dtype=np.float64),
                    ),
                    _relative_l2(storage, _storage(reference_state)),
                )
            reference_rhs = reference["rhs"]
            largest_rhs_error = max(
                largest_rhs_error,
                _relative_l2(stage.rhs.velocity, _vectors(reference_rhs, "u")),
                _relative_l2(stage.rhs.jacobian, _jacobian(reference_rhs)),
            )

        assert np.array_equal(
            stages[0].storage_pre, np.zeros_like(stages[0].storage_pre)
        )
        assert np.array_equal(stages[0].post.positions, stages[1].pre.positions)
        assert np.array_equal(stages[1].post.positions, stages[2].pre.positions)
        assert np.array_equal(stages[0].post.gamma, stages[1].pre.gamma)
        assert np.array_equal(stages[1].post.gamma, stages[2].pre.gamma)
        for stage in stages:
            assert np.array_equal(stage.storage_post[:, 6], np.zeros(state.sigma.shape))
            assert np.array_equal(stage.storage_post[:, 8], np.zeros(state.sigma.shape))

    assert largest_state_error <= 1e-11
    assert largest_rhs_error <= 1e-11


def test_two_step_timevarying_uinf_contract_matches_flowvpm() -> None:
    payload = _payload()
    fixture = payload["fixtures"]["rk3_timevarying_uinf_direct_gauserf"]
    config = fixture["config"]
    base = np.asarray(
        [config[f"uinf_base_{axis}"][0] for axis in "xyz"], dtype=np.float64
    )
    slope = np.asarray(
        [config[f"uinf_slope_{axis}"][0] for axis in "xyz"], dtype=np.float64
    )
    used = []
    for step in range(1, 3):
        step_record = fixture[f"step_{step:02d}"]
        time_before = float(step_record["field_time_before"][0])
        recorded = np.asarray(
            [step_record[f"uinf_used_{axis}"][0] for axis in "xyz"],
            dtype=np.float64,
        )
        assert np.array_equal(recorded, base + slope * time_before)
        used.append(recorded)
    assert not np.array_equal(used[0], used[1])

    metrics = evaluate(payload)
    assert all(metrics["config_contract"]["timevarying_uinf_rk"].values())
    assert metrics["gates"]["timevarying_uinf_rk_state_relative_l2_max_le_1e-11"]
    assert metrics["gates"]["timevarying_uinf_rk_rhs_relative_l2_max_le_1e-11"]
    assert metrics["gates"]["timevarying_uinf_clock_and_contract_le_1e-15"]


def test_corrected_pedrizzetti_cases_match_flowvpm() -> None:
    fixture = _payload()["fixtures"]["corrected_pedrizzetti"]
    largest_error = 0.0
    for case_name in ("case_001", "case_002", "case_003", "case_004"):
        case = fixture[case_name]
        gamma_before = _vectors(case, "gamma_before")
        gamma_reference = _vectors(case, "gamma_after")
        jacobian = unpack_julia_column_major(
            np.asarray(case["j_column_major"], dtype=np.float64)[None, :]
        )
        gamma_actual = corrected_pedrizzetti(
            gamma_before,
            jacobian,
            float(case["alpha"][0]),
        )
        largest_error = max(
            largest_error,
            _relative_l2(gamma_actual, gamma_reference),
        )
        assert abs(np.linalg.norm(gamma_actual) - np.linalg.norm(gamma_before)) <= 1e-14
    assert largest_error <= 1e-12


def test_oracle_and_python_parity_outputs_are_all_finite() -> None:
    payload = _payload()

    def visit(value: object) -> None:
        if isinstance(value, dict):
            for child in value.values():
                visit(child)
        elif isinstance(value, list) and value and isinstance(value[0], (int, float)):
            assert np.all(np.isfinite(np.asarray(value, dtype=np.float64)))

    visit(payload["fixtures"])
