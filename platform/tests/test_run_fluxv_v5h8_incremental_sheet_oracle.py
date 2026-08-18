"""Tests for the isolated FluxV v5h8 incremental-sheet runner."""

from __future__ import annotations

import ast
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import sys

import pytest


RUNNER_PATH = (
    Path(__file__).resolve().parents[1]
    / "forward_flight_benchmarks"
    / "run_fluxv_v5h8_incremental_sheet_oracle.py"
)


def _load_isolated_runner() -> tuple[object, set[str]]:
    before = set(sys.modules)
    runner_directory = str(RUNNER_PATH.parent)
    sys.path.insert(0, runner_directory)
    module_name = "_fluxv_v5h8_incremental_sheet_runner_isolated"
    sys.modules.pop(module_name, None)
    try:
        spec = importlib.util.spec_from_file_location(module_name, RUNNER_PATH)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    finally:
        assert sys.path[0] == runner_directory
        sys.path.pop(0)
    return module, set(sys.modules) - before


gate, RUNNER_IMPORT_DELTA = _load_isolated_runner()


def _strict_load(path: Path) -> object:
    return json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
    )


def test_runner_import_isolated_from_forward_flight_and_ptera_modules() -> None:
    forbidden = {
        name
        for name in RUNNER_IMPORT_DELTA
        if name == "forward_flight_benchmarks"
        or name.startswith("forward_flight_benchmarks.")
        or "pterasoftware" in name.lower()
        or "baik" in name.lower()
        or "yang_plev" in name.lower()
    }
    assert forbidden == set()
    assert gate.sheet.__name__ == "fluxv_v5h8_incremental_sheet"


def test_runner_has_no_package_ptera_target_or_load_import() -> None:
    tree = ast.parse(RUNNER_PATH.read_text(encoding="utf-8"))
    imported_modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported_modules.add(node.module or "")
    assert "forward_flight_benchmarks" not in imported_modules
    assert all("ptera" not in name.lower() for name in imported_modules)
    assert all("baik" not in name.lower() for name in imported_modules)
    assert all("yang" not in name.lower() for name in imported_modules)
    assert all("load" not in name.lower() for name in imported_modules)


def test_frozen_sheet_module_sha256_is_the_executed_source() -> None:
    actual = sha256(gate._sheet_module_path().read_bytes()).hexdigest()
    assert actual == gate.EXPECTED_SHEET_MODULE_SHA256


def test_zero_transport_rows_close_field_jacobian_impulse_and_prefix() -> None:
    rows = gate._zero_transport_rows()
    assert [row["release_count"] for row in rows] == [1, 2, 3, 4]
    assert all(row["passed"] for row in rows)
    for row in rows:
        assert row["max_field_abs_residual"] <= gate.FIELD_IMPULSE_TOLERANCE
        assert row["max_jacobian_abs_residual"] <= gate.JACOBIAN_TOLERANCE
        assert row["max_impulse_abs_residual"] <= gate.FIELD_IMPULSE_TOLERANCE
        assert row["max_analytic_impulse_abs_residual"] <= gate.FIELD_IMPULSE_TOLERANCE
        assert row["append_diagnostics"]["reported_prefix_bitwise_unchanged"]
        assert row["append_diagnostics"]["independent_prefix_bitwise_unchanged"]
        clone_check = row["append_diagnostics"]["independent_clone_check"]
        assert clone_check["old_boundary_set_exact"]
        assert clone_check["appended_clone_set_exact"]
        assert row["incremental_particle_count"] <= gate.PARTICLE_CAP
        assert row["direct_connected_particle_count"] <= gate.PARTICLE_CAP


def test_affine_rows_close_live_basis_and_reject_fresh_redeposit() -> None:
    rows = gate._affine_rows()
    assert [row["release_count"] for row in rows] == [1, 2, 3, 4]
    assert all(row["passed"] for row in rows)
    assert [row["affine_transport_count"] for row in rows] == [0, 1, 2, 3]
    for row in rows:
        assert row["transport_relation"]["passed"]
        assert row["append_diagnostics"]["reported_prefix_bitwise_unchanged"]
        assert row["append_diagnostics"]["independent_prefix_bitwise_unchanged"]
        clone_check = row["append_diagnostics"]["independent_clone_check"]
        assert clone_check["old_boundary_set_exact"]
        assert clone_check["appended_clone_set_exact"]
        assert (
            row["max_live_collapse_field_abs_residual"] <= gate.FIELD_IMPULSE_TOLERANCE
        )
        assert row["max_live_collapse_jacobian_abs_residual"] <= gate.JACOBIAN_TOLERANCE
        assert (
            row["max_live_collapse_impulse_abs_residual"]
            <= gate.FIELD_IMPULSE_TOLERANCE
        )
        assert row["collapse_particle_count_closed"]
        assert row["incremental_particle_count"] <= gate.PARTICLE_CAP
    assert rows[0]["fresh_negative_control_required"] is False
    assert rows[0]["fresh_field_relative_difference"] == pytest.approx(0.0)
    for row in rows[1:]:
        assert row["fresh_negative_control_required"] is True
        assert row["fresh_negative_control_passed"] is True
        assert (
            row["fresh_field_relative_difference"]
            > gate.FRESH_REDEPOSITION_MIN_RELATIVE_DIFFERENCE
        )


def test_full_gate_is_bounded_mechanical_go_but_production_blocked() -> None:
    summary = gate.run_gate()
    assert summary["passed"] is True
    assert summary["status"] == ("go_v5h8_bounded_affine_live_basis_mechanics_only")
    assert summary["scope"]["observation_access"] == "none"
    assert summary["scope"]["forward_flight_package_init_executed"] is False
    assert summary["gates"] == {
        "sheet_module_frozen_sha256_passed": True,
        "all_zero_transport_rows_passed": True,
        "all_affine_live_basis_rows_passed": True,
        "fresh_geometry_negative_control_passed": True,
        "particle_cap_passed": True,
        "cap_failure_rollback_passed": True,
        "target_access_count": 0,
        "ptera_solver_call_count": 0,
        "load_call_count": 0,
    }
    assert summary["rollback_probe"]["passed"] is True
    assert summary["production_decision"]["promotion"] == "blocked"
    assert (
        summary["diagnostics"]["minimum_affine_fresh_field_relative_difference"]
        > gate.FRESH_REDEPOSITION_MIN_RELATIVE_DIFFERENCE
    )
    assert summary["diagnostics"]["maximum_incremental_particle_count"] <= 1_000


def test_gate_is_semantically_deterministic() -> None:
    assert gate.run_gate() == gate.run_gate()


def test_two_fresh_artifacts_are_strict_hash_closed_and_byte_identical(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first_summary = gate.write_artifact(first)
    second_summary = gate.write_artifact(second)
    assert first_summary == second_summary
    assert _strict_load(first / "summary.json") == first_summary
    assert _strict_load(second / "summary.json") == second_summary

    first_files = {path.name for path in first.iterdir()}
    second_files = {path.name for path in second.iterdir()}
    expected_files = {
        "README.md",
        "source_manifest.json",
        "summary.json",
        "result_manifest.json",
        "SHA256SUMS",
    }
    assert first_files == second_files == expected_files
    for name in expected_files:
        assert (first / name).read_bytes() == (second / name).read_bytes()

    source_manifest = _strict_load(first / "source_manifest.json")
    assert source_manifest["frozen_sheet_module_sha256"] == (
        gate.EXPECTED_SHEET_MODULE_SHA256
    )
    assert (
        source_manifest["files"][
            "platform/forward_flight_benchmarks/fluxv_v5h8_incremental_sheet.py"
        ]
        == gate.EXPECTED_SHEET_MODULE_SHA256
    )
    result_manifest = _strict_load(first / "result_manifest.json")
    for name, expected in result_manifest["files"].items():
        assert gate._file_sha256(first / name) == expected
    for line in (first / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        expected, name = line.split("  ", 1)
        assert gate._file_sha256(first / name) == expected


def test_artifact_refuses_to_overwrite_existing_directory(tmp_path: Path) -> None:
    output = tmp_path / "existing"
    output.mkdir()
    with pytest.raises(FileExistsError, match="overwrite"):
        gate.write_artifact(output)
