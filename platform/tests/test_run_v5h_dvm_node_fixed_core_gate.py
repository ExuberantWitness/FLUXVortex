from __future__ import annotations

import ast
import builtins
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

import forward_flight_benchmarks.run_v5h_dvm_node_fixed_core_gate as gate


def _fake_configuration(
    geometry: gate.GateGeometry,
    *,
    span_cells: int,
    target_spacing_m: float,
    smoothing_radius_m: float,
    transport_substeps: int,
) -> gate._ConfigurationResult:
    space_signal = target_spacing_m**2
    time_signal = 1.0e-7 / transport_substeps**2
    state = np.asarray((1.0 + time_signal, 2.0 - time_signal), dtype=float)
    frontier = np.asarray(((1.0 + space_signal, 2.0 - space_signal),), dtype=float)
    probes = np.asarray(((2.0 + 0.5 * space_signal,),), dtype=float)
    particle_count = int(round(1.0 / target_spacing_m))
    summary = {
        "geometry": geometry,
        "span_cells": span_cells,
        "target_spacing_m": target_spacing_m,
        "smoothing_radius_m": smoothing_radius_m,
        "transport_substeps": transport_substeps,
        "first_bridge": {
            "particle_count": particle_count,
            "realized_spacing_max_m": 0.9 * target_spacing_m,
        },
        "node_placement_passed": True,
        "continuous_frontier_only_passed": True,
        "node_handoff_schema_fields": ["kinematics.node_id"],
        "forbidden_node_strength_fields": [],
        "node_circulation_consumed_count": 0,
        "patch_binding_passed": True,
        "global_exact_once_passed": True,
        "isolation_evidence": {
            "first_ribbon_feedback_call_count": 0,
            "second_ribbon_feedback_call_count": 0,
            "transport_feedback_call_count": 0,
            "transport_parent_write_count": 0,
            "transport_surface_channel_write_count": 0,
            "transport_observation_access": "none",
            "transport_target_case_branch": "none",
        },
        "passed": True,
    }
    return gate._ConfigurationResult(
        summary=summary,
        state_vector=state,
        frontier_positions=frontier,
        probe_velocity=probes,
    )


def _fake_smoke() -> dict[str, object]:
    return {
        "passed": True,
        "patch_binding_passed": True,
        "global_exact_once_passed": True,
    }


def _fake_stop_smoke() -> dict[str, object]:
    return {
        "passed": False,
        "patch_binding_passed": False,
        "global_exact_once_passed": False,
        "failure": "injected minimal-smoke stop",
    }


def _fake_restart(*args: object) -> dict[str, object]:
    del args
    return {
        "passed": True,
        "patch_binding_passed": True,
        "global_exact_once_passed": True,
    }


def _fake_core_scan(*args: object, **kwargs: object) -> dict[str, object]:
    del args, kwargs
    return {
        "gate_eligible": False,
        "selection_role": "diagnostic_only_no_selection",
        "patch_binding_passed": True,
        "global_exact_once_passed": True,
    }


@pytest.fixture(scope="module")
def minimal_smoke() -> dict[str, object]:
    return gate.run_minimal_smoke()


@pytest.fixture(scope="module")
def straight_configuration() -> gate._ConfigurationResult:
    return gate._run_configuration(
        "straight",
        span_cells=2,
        target_spacing_m=gate.SOURCE_TRANSFER_SPACING_M,
        smoothing_radius_m=gate.SOURCE_TRANSFER_SIGMA_M,
        transport_substeps=1,
    )


def test_minimal_smoke_is_direct_input_blind_and_sign_symmetric(
    minimal_smoke: dict[str, object],
) -> None:
    assert minimal_smoke["passed"] is True
    assert minimal_smoke["geometry"] == "straight"
    assert minimal_smoke["span_cells"] == 2
    assert minimal_smoke["disabled_input_blind_passed"] is True
    signs = minimal_smoke["incidence_signs"]
    assert [row["incidence_sign"] for row in signs] == [1, -1]
    assert all(row["direct_event_attestation_passed"] for row in signs)
    assert all(row["node_placement_passed"] for row in signs)
    assert all(row["cell_source_coverage_passed"] for row in signs)
    assert all(row["first_half_step_passed"] for row in signs)
    assert minimal_smoke["patch_binding_passed"] is True
    assert minimal_smoke["global_exact_once_passed"] is True
    assert all(row["patch_binding_passed"] for row in signs)
    assert all(row["global_exact_once_passed"] for row in signs)
    assert all(row["global_exact_once_rejection_probe"]["passed"] for row in signs)
    assert all(row["passed"] for row in signs)


def test_minimal_smoke_replay_is_deterministic(
    minimal_smoke: dict[str, object],
) -> None:
    assert gate.run_minimal_smoke() == minimal_smoke


def test_configuration_uses_fixed_sigma_independent_h_and_attested_ownership(
    straight_configuration: gate._ConfigurationResult,
) -> None:
    row = straight_configuration.summary
    assert row["passed"] is True
    assert row["smoothing_radius_m"] == gate.SOURCE_TRANSFER_SIGMA_M
    assert row["target_spacing_m"] == gate.SOURCE_TRANSFER_SPACING_M
    assert row["first_bridge"]["fixed_physical_sigma"] is True
    assert row["second_bridge"]["fixed_physical_sigma"] is True
    assert row["first_bridge"]["realized_spacing_max_m"] <= row["target_spacing_m"]
    assert row["first_bridge"]["realized_overlap_min"] >= 2.125
    assert row["node_source_role"] == "geometry_only"
    assert row["cell_source_role"] == "circulation_only"
    assert row["node_circulation_consumed_count"] == 0
    assert row["patch_binding_passed"] is True
    assert row["global_exact_once_passed"] is True
    assert row["first_ribbon_handoff_integrity"]["passed"] is True
    assert row["second_ribbon_handoff_integrity"]["passed"] is True
    assert row["global_exact_once_rejection_probe"]["passed"] is True
    assert row["cell_source_coverage_passed"] is True
    assert row["first_node_placement"]["active_cell_endpoint_coverage_passed"] is True
    assert row["second_node_placement"]["active_cell_endpoint_coverage_passed"] is True
    assert row["continuous_frontier_only_passed"] is True
    assert row["first_modes"] == ["first"] * 3
    assert row["second_modes"] == ["continuous"] * 3


def test_restart_and_core_scan_bind_live_placement_and_exact_once() -> None:
    restart = gate._run_restart_audit("straight", 2)
    core_scan = gate._run_core_scan(
        "straight",
        span_cells=2,
        target_spacing_m=gate.DEFAULT_TARGET_SPACINGS_M[-1],
    )
    assert restart["passed"] is True
    assert restart["patch_binding_passed"] is True
    assert restart["global_exact_once_passed"] is True
    assert restart["global_exact_once_rejection_probe"]["passed"] is True
    assert all(
        layer["ribbon_handoff_integrity"]["passed"] for layer in restart["layers"]
    )
    assert core_scan["patch_binding_passed"] is True
    assert core_scan["global_exact_once_passed"] is True
    assert core_scan["global_exact_once_rejection_probe"]["passed"] is True
    assert core_scan["ribbon_handoff_integrity"]["passed"] is True


def test_full_loop_covers_all_geometries_h_and_time_without_core_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, float, int]] = []

    def fake_run(
        geometry: gate.GateGeometry,
        *,
        span_cells: int,
        target_spacing_m: float,
        smoothing_radius_m: float,
        transport_substeps: int,
    ) -> gate._ConfigurationResult:
        calls.append((geometry, target_spacing_m, transport_substeps))
        return _fake_configuration(
            geometry,
            span_cells=span_cells,
            target_spacing_m=target_spacing_m,
            smoothing_radius_m=smoothing_radius_m,
            transport_substeps=transport_substeps,
        )

    monkeypatch.setattr(gate, "_run_configuration", fake_run)
    monkeypatch.setattr(gate, "run_minimal_smoke", _fake_smoke)
    monkeypatch.setattr(gate, "_run_restart_audit", _fake_restart)
    monkeypatch.setattr(gate, "_run_core_scan", _fake_core_scan)
    monkeypatch.setattr(
        gate,
        "_source_hashes",
        lambda: {"runner": {"path": "runner.py", "sha256": "0" * 64}},
    )
    result = gate.run_fixed_core_gate()
    expected = {
        (geometry, spacing, substeps)
        for geometry in gate.CASE_GEOMETRIES
        for spacing in gate.DEFAULT_TARGET_SPACINGS_M
        for substeps in gate.DEFAULT_TRANSPORT_SUBSTEPS
    }
    assert set(calls) == expected
    assert len(calls) == len(expected) + 1  # one deterministic replay
    assert result["passed"] is True
    assert result["gate_summary"]["patch_binding_passed"] is True
    assert result["gate_summary"]["global_exact_once_passed"] is True
    assert result["gate_summary"]["core_scan_gate_eligible"] is False
    for geometry in result["geometries"]:
        assert geometry["core_scan_diagnostic"]["gate_eligible"] is False
        quadrature = geometry["fixed_core_quadrature_at_finest_time"]
        assert quadrature["observable_ceil_count_refinement"] is True
        assert quadrature["passed"] is True


@pytest.mark.parametrize(
    "hard_gate", ("patch_binding_passed", "global_exact_once_passed")
)
def test_live_handoff_integrity_components_are_hard_gates(
    hard_gate: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forged_configuration(
        geometry: gate.GateGeometry,
        *,
        span_cells: int,
        target_spacing_m: float,
        smoothing_radius_m: float,
        transport_substeps: int,
    ) -> gate._ConfigurationResult:
        original = _fake_configuration(
            geometry,
            span_cells=span_cells,
            target_spacing_m=target_spacing_m,
            smoothing_radius_m=smoothing_radius_m,
            transport_substeps=transport_substeps,
        )
        summary = dict(original.summary)
        summary[hard_gate] = False
        return replace(original, summary=summary)

    monkeypatch.setattr(gate, "_run_configuration", forged_configuration)
    monkeypatch.setattr(gate, "run_minimal_smoke", _fake_smoke)
    monkeypatch.setattr(gate, "_run_restart_audit", _fake_restart)
    monkeypatch.setattr(gate, "_run_core_scan", _fake_core_scan)
    monkeypatch.setattr(
        gate,
        "_source_hashes",
        lambda: {"runner": {"path": "runner.py", "sha256": "0" * 64}},
    )

    result = gate.run_fixed_core_gate()

    assert result["passed"] is False
    assert result["status"] == "stop"
    assert result["gate_summary"][hard_gate] is False


def test_forged_source_hash_closure_forces_stop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(gate, "_run_configuration", _fake_configuration)
    monkeypatch.setattr(gate, "run_minimal_smoke", _fake_smoke)
    monkeypatch.setattr(gate, "_run_restart_audit", _fake_restart)
    monkeypatch.setattr(gate, "_run_core_scan", _fake_core_scan)
    source_snapshots = iter(
        (
            {"runner": {"path": "runner.py", "sha256": "0" * 64}},
            {"runner": {"path": "runner.py", "sha256": "1" * 64}},
        )
    )
    monkeypatch.setattr(gate, "_source_hashes", lambda: next(source_snapshots))

    result = gate.run_fixed_core_gate()

    assert result["passed"] is False
    assert result["status"] == "stop"
    assert result["gate_summary"]["declared_source_snapshot_stable_passed"] is False
    assert result["gate_summary"]["stop_required"] is True


def test_quadrature_gate_fails_closed_on_nonfinite_or_unobservable_refinement() -> None:
    rows = [
        _fake_configuration(
            "straight",
            span_cells=2,
            target_spacing_m=spacing,
            smoothing_radius_m=gate.SOURCE_TRANSFER_SIGMA_M,
            transport_substeps=4,
        )
        for spacing in gate.DEFAULT_TARGET_SPACINGS_M
    ]
    forged = list(rows)
    forged[-1] = replace(
        forged[-1],
        frontier_positions=np.asarray(((float("nan"), 0.0),), dtype=float),
    )
    forged_gate = gate._quadrature_refinement_gate(forged)
    assert forged_gate["passed"] is False
    assert "NaN" not in gate._json_text(forged_gate)

    wrong_shape = list(rows)
    wrong_shape[-1] = replace(
        wrong_shape[-1],
        frontier_positions=np.asarray((0.0, 1.0), dtype=float),
    )
    assert gate._quadrature_refinement_gate(wrong_shape)["passed"] is False

    repeated_count = list(rows)
    repeated_summary = dict(repeated_count[-1].summary)
    repeated_summary["first_bridge"] = dict(repeated_summary["first_bridge"])
    repeated_summary["first_bridge"]["particle_count"] = repeated_count[-2].summary[
        "first_bridge"
    ]["particle_count"]
    repeated_count[-1] = replace(repeated_count[-1], summary=repeated_summary)
    assert gate._quadrature_refinement_gate(repeated_count)["passed"] is False


@pytest.mark.parametrize(
    "config",
    (
        gate.GateConfig(span_cells=1),
        gate.GateConfig(target_spacings_m=(0.04, 0.02)),
        gate.GateConfig(target_spacings_m=(0.04, 0.02, 0.02)),
        gate.GateConfig(target_spacings_m=(0.08, 0.04, 0.02)),
        gate.GateConfig(transport_substeps=(1, 2)),
        gate.GateConfig(transport_substeps=(1, 4, 2)),
    ),
)
def test_invalid_or_coupled_refinement_inputs_fail_closed(
    config: gate.GateConfig,
) -> None:
    with pytest.raises(ValueError):
        gate._validate_config(config)


def test_output_has_no_surface_quantity_keys_or_target_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(gate, "_run_configuration", _fake_configuration)
    monkeypatch.setattr(gate, "run_minimal_smoke", _fake_smoke)
    monkeypatch.setattr(gate, "_run_restart_audit", _fake_restart)
    monkeypatch.setattr(gate, "_run_core_scan", _fake_core_scan)
    monkeypatch.setattr(
        gate,
        "_source_hashes",
        lambda: {"runner": {"path": "runner.py", "sha256": "0" * 64}},
    )
    result = gate.run_fixed_core_gate()
    forbidden = {"force", "load", "lift", "drag", "pressure", "polar", "score"}

    def walk(value: object) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                tokens = set(str(key).casefold().replace("-", "_").split("_"))
                assert not tokens.intersection(forbidden)
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    walk(result)
    assert result["observation_access"] == "none"
    assert result["target_case_branch"] == "none"
    assert result["ownership"]["node_circulation_consumed_count"] == 0


def test_source_closure_is_code_only_and_observation_isolated() -> None:
    paths = gate._source_paths()
    assert paths
    assert paths["benchmark_package_init"].name == "__init__.py"
    assert paths["ldvm_induction_kernel"].name == "flap_ldvm.py"
    assert all(path.suffix in {".py", ".toml"} for path in paths.values())
    assert all("runs" not in path.parts for path in paths.values())
    assert all("data" not in path.parts for path in paths.values())
    assert not any(
        token in path.name.casefold()
        for path in paths.values()
        for token in ("yang", "izraelevitz", "baik")
    )


def test_runtime_never_calls_target_builders_ground_truth_or_ptera_solver(
    monkeypatch: pytest.MonkeyPatch,
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

    import forward_flight_benchmarks.baik2012 as baik
    import forward_flight_benchmarks.ptera_adapter as ptera_adapter

    def forbidden_call(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("target builder or Ptera solver was called")

    monkeypatch.setattr(baik, "build_baik_movement", forbidden_call)
    monkeypatch.setattr(baik, "run_baik_old_fluxv", forbidden_call)
    monkeypatch.setattr(baik, "run_model", forbidden_call)
    monkeypatch.setattr(ptera_adapter, "run_model", forbidden_call)
    original_open = builtins.open

    def observation_guard(file: object, *args: object, **kwargs: object) -> object:
        path_text = str(file).casefold()
        if any(
            token in path_text
            for token in ("baik", "yang", "izraelevitz", "ground_truth")
        ):
            raise AssertionError("target observation or ground truth was read")
        return original_open(file, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", observation_guard)
    smoke = gate.run_minimal_smoke()
    assert smoke["passed"] is True


def test_runtime_instrumentation_guards_eager_package_aliases() -> None:
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


def test_stop_result_still_writes_a_complete_failure_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(gate, "run_minimal_smoke", _fake_stop_smoke)
    monkeypatch.setattr(
        gate,
        "_source_hashes",
        lambda: {"runner": {"path": "runner.py", "sha256": "0" * 64}},
    )
    result = gate.run_fixed_core_gate()
    assert result["status"] == "stop"
    assert result["geometries"] == []

    output = tmp_path / "stopped"
    provenance = gate.capture_run_provenance(
        output_dir=output,
        numerical_started_utc="2026-08-15T00:00:00Z",
        numerical_completed_utc="2026-08-15T00:00:01Z",
        run_uuid="00000000-0000-4000-8000-000000000010",
    )
    gate.write_run_artifacts(result, output, run_provenance=provenance)

    summary = gate._load_strict_json(output / "summary.json")
    recomputed = gate._load_strict_json(output / "recomputed_gates.json")
    raw = gate._load_strict_json(output / "raw_refinement.json")
    assert summary["status"] == "stop"
    assert recomputed["reported_values_match_raw_recomputation"] is True
    assert recomputed["passed"] is False
    assert raw["configurations"] == []
    assert raw["deterministic_replay"] is None
    assert gate.verify_semantic_manifest(output)


def test_artifact_bundle_is_strict_finite_deterministic_and_nonoverwriting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(gate, "_run_configuration", _fake_configuration)
    monkeypatch.setattr(gate, "run_minimal_smoke", _fake_smoke)
    monkeypatch.setattr(gate, "_run_restart_audit", _fake_restart)
    monkeypatch.setattr(gate, "_run_core_scan", _fake_core_scan)
    monkeypatch.setattr(
        gate,
        "_source_hashes",
        lambda: {"runner": {"path": "runner.py", "sha256": "0" * 64}},
    )
    result = gate.run_fixed_core_gate()
    left_target = tmp_path / "left"
    right_target = tmp_path / "right"
    left_provenance = gate.capture_run_provenance(
        output_dir=left_target,
        numerical_started_utc="2026-08-15T00:00:00Z",
        numerical_completed_utc="2026-08-15T00:00:01Z",
        run_uuid="00000000-0000-4000-8000-000000000001",
    )
    right_provenance = gate.capture_run_provenance(
        output_dir=right_target,
        numerical_started_utc="2026-08-15T00:01:00Z",
        numerical_completed_utc="2026-08-15T00:01:01Z",
        run_uuid="00000000-0000-4000-8000-000000000002",
    )
    left = gate.write_run_artifacts(
        result,
        left_target,
        run_provenance=left_provenance,
    )
    right = gate.write_run_artifacts(
        result,
        right_target,
        run_provenance=right_provenance,
    )
    expected_files = {
        "summary.json",
        "metrics.json",
        "raw_refinement.json",
        "recomputed_gates.json",
        "source_manifest.json",
        "semantic_manifest.json",
        "result_manifest.json",
        "run_manifest.json",
        "run.log",
        "README.md",
        "SHA256SUMS",
    }
    assert {path.name for path in left.iterdir()} == expected_files
    assert {path.name for path in right.iterdir()} == expected_files
    semantic_files = {
        "summary.json",
        "metrics.json",
        "raw_refinement.json",
        "recomputed_gates.json",
        "source_manifest.json",
        "semantic_manifest.json",
        "README.md",
    }
    for name in semantic_files:
        assert (left / name).read_bytes() == (right / name).read_bytes()
    for name in {"result_manifest.json", "run_manifest.json", "run.log"}:
        assert (left / name).read_bytes() != (right / name).read_bytes()
    left_semantic_sha = gate.verify_semantic_manifest(left)
    assert gate.verify_semantic_manifest(right) == left_semantic_sha
    recomputed = gate.recompute_refinement_gates_from_artifacts(left)
    assert recomputed == gate._load_strict_json(left / "recomputed_gates.json")
    assert recomputed["reported_values_match_raw_recomputation"] is True
    assert recomputed["passed"] is True

    reported_summary = gate._load_strict_json(left / "summary.json")
    reported_raw = gate._load_strict_json(left / "raw_refinement.json")
    truncated_summary = json.loads(json.dumps(reported_summary))
    removed_geometry = truncated_summary["geometries"].pop()["geometry"]
    truncated_raw = json.loads(json.dumps(reported_raw))
    truncated_raw["configurations"] = [
        row
        for row in truncated_raw["configurations"]
        if row["geometry"] != removed_geometry
    ]
    with pytest.raises(ValueError, match="exactly cover the frozen config"):
        gate._recompute_refinement_gates(truncated_summary, truncated_raw)
    wrong_replay = json.loads(json.dumps(reported_raw))
    wrong_replay["deterministic_replay"] = wrong_replay["configurations"][-1]
    with pytest.raises(ValueError, match="not the frozen reference"):
        gate._recompute_refinement_gates(reported_summary, wrong_replay)

    source_manifest = gate._load_strict_json(left / "source_manifest.json")
    assert source_manifest["runtime_import_closure_complete"] is False
    assert (
        "not claimed as a complete runtime import closure"
        in source_manifest["closure_scope"]
    )
    run_manifest = gate._load_strict_json(left / "run_manifest.json")
    provenance = run_manifest["provenance"]
    assert provenance["run_uuid"] == "00000000-0000-4000-8000-000000000001"
    assert provenance["output_dir"] == str(left.resolve())
    assert provenance["numerical_started_utc"].endswith("Z")
    assert provenance["process_argv"]
    assert provenance["process_command"]
    assert isinstance(provenance["git"]["available"], bool)
    assert provenance["git"]["dirty"] in {True, False, None}
    assert provenance["environment"]["packages"]["numpy"] == np.__version__

    serialized = (left / "summary.json").read_text(encoding="utf-8")
    assert "NaN" not in serialized and "Infinity" not in serialized
    assert json.loads(serialized)["passed"] is True
    with pytest.raises(FileExistsError):
        gate.write_run_artifacts(result, left, run_provenance=left_provenance)

    forged = dict(result)
    forged["forged_nonfinite"] = float("nan")
    nonfinite_target = tmp_path / "nonfinite"
    nonfinite_provenance = gate.capture_run_provenance(
        output_dir=nonfinite_target,
        numerical_started_utc="2026-08-15T00:02:00Z",
        numerical_completed_utc="2026-08-15T00:02:01Z",
        run_uuid="00000000-0000-4000-8000-000000000003",
    )
    with pytest.raises(ValueError, match="Out of range float values"):
        gate.write_run_artifacts(
            forged,
            nonfinite_target,
            run_provenance=nonfinite_provenance,
        )
    assert not nonfinite_target.exists()

    incomplete_target = tmp_path / "incomplete-provenance"
    incomplete_provenance = dict(left_provenance)
    incomplete_provenance.pop("environment")
    incomplete_provenance["output_dir"] = str(incomplete_target.resolve())
    with pytest.raises(ValueError, match="provenance fields"):
        gate.write_run_artifacts(
            result,
            incomplete_target,
            run_provenance=incomplete_provenance,
        )
    assert not incomplete_target.exists()

    stale_source_target = tmp_path / "stale-source"
    stale_source_provenance = gate.capture_run_provenance(
        output_dir=stale_source_target,
        numerical_started_utc="2026-08-15T00:03:00Z",
        numerical_completed_utc="2026-08-15T00:03:01Z",
        run_uuid="00000000-0000-4000-8000-000000000004",
    )
    monkeypatch.setattr(
        gate,
        "_source_hashes",
        lambda: {"runner": {"path": "runner.py", "sha256": "1" * 64}},
    )
    with pytest.raises(RuntimeError, match="source hashes are stale"):
        gate.write_run_artifacts(
            result,
            stale_source_target,
            run_provenance=stale_source_provenance,
        )
    assert not stale_source_target.exists()

    hostile_manifest = gate._load_strict_json(right / "semantic_manifest.json")
    hostile_manifest["files"]["../summary.json"] = "0" * 64
    (right / "semantic_manifest.json").write_text(
        gate._json_text(hostile_manifest),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="exact payload set"):
        gate.verify_semantic_manifest(right)

    raw = gate._load_strict_json(left / "raw_refinement.json")
    raw["configurations"][0]["state_vector"]["values"][0] += 1.0
    (left / "raw_refinement.json").write_text(
        gate._json_text(raw),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="raw array SHA-256 mismatch"):
        gate.recompute_refinement_gates_from_artifacts(left)


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
