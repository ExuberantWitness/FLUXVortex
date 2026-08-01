"""No-history one-shot definition tests for transport-v2.3.

Every test installs forbidden sentinels on the formal collector, mesh builder,
and time marcher.  Dependency files, markers, and results used below live only
in per-test temporary namespaces; no formal authorization or history exists.
"""
from __future__ import annotations

import copy
from dataclasses import replace
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


PLATFORM = Path(__file__).resolve().parents[1]
if str(PLATFORM) not in sys.path:
    sys.path.insert(0, str(PLATFORM))

import actual_wake_reachable_pressure_obstruction_v2_guard as frozen  # noqa: E402
import actual_wake_reachable_pressure_obstruction_v23_one_shot as v23  # noqa: E402


@pytest.fixture(autouse=True)
def _forbid_formal_history(monkeypatch):
    forbidden = AssertionError(
        "formal collector, mesh builder, or time marcher was called"
    )
    collector = Mock(side_effect=forbidden)
    mesh = Mock(side_effect=forbidden)
    march = Mock(side_effect=forbidden)
    monkeypatch.setattr(frozen, "_collect_frozen_histories", collector)
    monkeypatch.setattr(frozen, "build_canonical_diamond_wing", mesh)
    monkeypatch.setattr(
        frozen,
        "march_actual_boundary_material_wake_explicit_midpoint",
        march,
    )
    yield
    collector.assert_not_called()
    mesh.assert_not_called()
    march.assert_not_called()


def _member(
    path: Path,
    *,
    phase: str,
    requirement: str,
    kind: str = "python_source",
    module: str | None = None,
) -> dict[str, object]:
    metadata = path.stat()
    return {
        "canonical_path": str(path),
        "module_and_distribution_identity": {
            "module": module or path.stem,
            "distribution": "synthetic-definition-only",
            "version": "0",
        },
        "kind": kind,
        "origin_or_package": "synthetic-definition-only",
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "st_dev": int(metadata.st_dev),
        "st_ino": int(metadata.st_ino),
        "st_size": int(metadata.st_size),
        "allowed_phase": phase,
        "required_or_optional": requirement,
        "justification": f"definition control for {path.name}",
    }


def _payload(
    *,
    baseline: list[Path],
    required_lazy: list[Path],
    optional: list[tuple[Path, str]] | None = None,
) -> dict[str, object]:
    members = [
        *[
            _member(
                path,
                phase="baseline_pre_marker",
                requirement="required",
            )
            for path in baseline
        ],
        *[
            _member(
                path,
                phase="declared_formal_path",
                requirement="required",
            )
            for path in required_lazy
        ],
        *[
            _member(path, phase=phase, requirement="optional")
            for path, phase in (optional or [])
        ],
    ]
    return {
        "schema_version": "1.0",
        "artifact": (
            "actual_wake_reachable_pressure_dependency_closure_v23_"
            "20260728_185556"
        ),
        "attempt_id": (
            "S3ai-v2.2-transport-v2.3-successor-20260728_185556"
        ),
        "transport_protocol_version": "S3ai-v2.3",
        "assurance_profile": "RESEARCH_ACCIDENTAL_DRIFT",
        "members": members,
        "B": [str(path) for path in baseline],
        "U": [str(path) for path in [*baseline, *required_lazy]]
        + [str(path) for path, _ in (optional or [])],
        "R": [str(path) for path in required_lazy],
    }


def _state(
    manifest: v23.DependencyManifest,
    paths: list[Path],
) -> dict[str, dict[str, object]]:
    by_path = manifest.member_by_path
    return {
        str(path): {
            **v23._manifest_member_fingerprint(by_path[str(path)]),
            "module_names": [path.stem],
        }
        for path in paths
    }


def _refresh_snapshot_certificate(snapshot: dict[str, object]) -> None:
    without_certificate = dict(snapshot)
    without_certificate.pop("snapshot_sha256", None)
    snapshot["snapshot_sha256"] = v23._sha256_bytes(
        v23._canonical_json_bytes(without_certificate)
    )


def _refresh_closure_certificate(closure: dict[str, object]) -> None:
    without_certificate = dict(closure)
    without_certificate.pop("closure_certificate_sha256", None)
    closure["closure_certificate_sha256"] = v23._sha256_bytes(
        v23._canonical_json_bytes(without_certificate)
    )


def _remove_required_success_and_rehash_ledger(
    closure: dict[str, object],
    required_path: str,
) -> None:
    ledger = closure["import_load_ledger"]
    events = [
        copy.deepcopy(event)
        for event in ledger["events"]
        if not (
            event["event_type"]
            == "checkpoint_confirmed_successful_load"
            and event.get("canonical_path") == required_path
        )
    ]
    previous = v23._ZERO_SHA256
    for sequence, event in enumerate(events):
        event["sequence"] = sequence
        event["previous_event_sha256"] = previous
        event_without_digest = dict(event)
        event_without_digest.pop("event_sha256", None)
        event["event_sha256"] = v23._sha256_bytes(
            v23._canonical_json_bytes(event_without_digest)
        )
        previous = event["event_sha256"]
    ledger["events"] = events
    ledger["event_count"] = len(events)
    ledger["event_chain_sha256"] = previous
    ledger["events_sha256"] = v23._sha256_bytes(
        v23._canonical_json_bytes(events)
    )
    closure["end"]["event_count"] = len(events)
    closure["end"]["event_chain_sha256"] = previous
    _refresh_snapshot_certificate(closure["end"])


def _synthetic_31_case_fixture():
    """Load the frozen v2.2 test's pure-synthetic 31-case observations."""

    fixture_path = (
        PLATFORM
        / "tests"
        / "test_actual_wake_reachable_pressure_obstruction_v22_one_shot.py"
    )
    module_name = "_v23_reused_v22_synthetic_fixture"
    spec = importlib.util.spec_from_file_location(module_name, fixture_path)
    if spec is None or spec.loader is None:
        raise AssertionError("cannot load the frozen synthetic fixture")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
        fixture = module._synthetic_31_case_fixture()
    finally:
        sys.modules.pop(module_name, None)
    assert fixture["forbidden_call_counts"] == {
        "collector": 0,
        "mesh": 0,
        "march": 0,
    }
    return fixture


def _synthetic_source_fingerprints() -> dict[str, str]:
    paths = [
        v23._WRAPPER_RELATIVE,
        *[
            f"platform/synthetic_v23_source_{index:02d}.py"
            for index in range(62)
        ],
    ]
    return {
        path: hashlib.sha256(path.encode("utf-8")).hexdigest()
        for path in paths
    }


def _synthetic_verified(
    *,
    manifest: v23.DependencyManifest,
    source_fingerprints: dict[str, str],
    stable_runtime: dict[str, object],
    definition_sha256: str,
    transport_preregistration_sha256: str,
    registry_manifest_sha256: str = "8" * 64,
) -> v23._VerifiedAuthorization:
    second_audit = {
        "invocation_metadata_path": (
            ".aris/synthetic_v23_invocation_metadata.json"
        ),
        "invocation_metadata_raw_sha256": "9" * 64,
        "invocation_metadata_canonical_sha256": "a" * 64,
        "request_path": ".aris/synthetic_v23_request.json",
        "request_raw_sha256": "1" * 64,
        "request_canonical_sha256": "b" * 64,
        "response_path": ".aris/synthetic_v23_response.json",
        "response_raw_sha256": "2" * 64,
        "response_canonical_sha256": "3" * 64,
        "trace_metadata_path": ".aris/synthetic_v23_trace_metadata.json",
        "trace_metadata_raw_sha256": "c" * 64,
        "clearance_canonical_sha256": "3" * 64,
        "ticket_semantic_projection_sha256": "d" * 64,
    }
    return v23._VerifiedAuthorization(
        payload={
            "authorization": {
                "id": "synthetic-v23-definition-only",
                "decision": (
                    "YES_NEW_ONE_SHOT_31_HISTORY_SUCCESSOR_AFTER_"
                    "TRANSPORT_FAILURE"
                ),
                "formal_execution_allowed": True,
                "execution_limit": 1,
            },
            "scope": {
                "production_activation_allowed": False,
                "force_hp_state_ves_118_fig_allowed": False,
                "malicious_local_writer_or_swap_restore_protection": False,
            },
            "issuance_request": {
                "path": ".aris/synthetic_v23_issuance_request.json",
                "raw_sha256": "e" * 64,
                "canonical_sha256": "f" * 64,
                "ticket_semantic_projection_sha256": "d" * 64,
            },
            "second_bounded_audit": second_audit,
        },
        raw_sha256="4" * 64,
        canonical_sha256="5" * 64,
        token_sha256="6" * 64,
        definition_sha256=definition_sha256,
        transport_preregistration_sha256=(
            transport_preregistration_sha256
        ),
        dependency_manifest_raw_sha256=str(manifest.raw_sha256),
        dependency_manifest_canonical_sha256=manifest.canonical_sha256,
        source_fingerprints=source_fingerprints,
        stable_runtime_identity=stable_runtime,
        audit_files={},
        clearance={
            "verdict": "ACCEPT_SYNTHETIC_V23_DEFINITION_ONLY",
            "review": {
                "actual_independence": "genuine-cross-family",
                "provider": "synthetic-review-provider",
                "model": "synthetic-review-model",
                "model_family": "synthetic-review-family",
                "agent_id": "synthetic-review-agent",
                "trace_id": "synthetic-review-trace",
            },
            "bindings": {
                "ordered_registry_manifest_sha256": (
                    registry_manifest_sha256
                ),
                "frozen_definition_chain_sha256": "0" * 64,
            },
        },
    )


def _install_fake_run_boundary(
    *,
    monkeypatch,
    root: Path,
    fixture,
    violate_after_collector: bool,
):
    """Patch only authorization/identity inputs around the real run chain."""

    root.mkdir(parents=True, exist_ok=True)
    result = root / "result.json"
    marker = root / "result.json.lock"
    wrapper_definition_path = root / "wrapper_definition.yaml"
    transport_path = root / "transport_preregistration.json"
    dependency_path = root / "dependency_manifest.json"
    authorization_path = root / "authorization.yaml"
    for path in (
        wrapper_definition_path,
        transport_path,
        dependency_path,
        authorization_path,
    ):
        path.write_text("synthetic definition-only boundary\n")

    baseline = root / "baseline.py"
    required = root / "required_lazy_v23.py"
    unauthorized = root / "unauthorized_after_collector.py"
    baseline.write_text("BASELINE = True\n")
    required.write_text("REQUIRED_LAZY = True\n")
    unauthorized.write_text("UNAUTHORIZED = True\n")
    manifest = v23._parse_dependency_manifest(
        _payload(baseline=[baseline], required_lazy=[required])
    )
    baseline_state = _state(manifest, [baseline])
    complete_state = _state(manifest, [baseline, required])
    unauthorized_metadata = unauthorized.stat()
    unauthorized_fingerprint = {
        "sha256": hashlib.sha256(unauthorized.read_bytes()).hexdigest(),
        "st_dev": int(unauthorized_metadata.st_dev),
        "st_ino": int(unauthorized_metadata.st_ino),
        "st_size": int(unauthorized_metadata.st_size),
        "kind": "python_source",
        "module_names": [unauthorized.stem],
    }
    state_box = {"value": baseline_state}

    definition, definition_sha = v23._load_wrapper_definition()
    _, transport_sha = v23._load_transport_preregistration()
    source_fingerprints = _synthetic_source_fingerprints()
    stable_runtime = {"synthetic_v23_runtime": True}
    ordered_names = list(fixture["ordered_case_names"])
    _, _, registry_sha = v23._case_registry_identity(fixture["contract"])
    verified = _synthetic_verified(
        manifest=manifest,
        source_fingerprints=source_fingerprints,
        stable_runtime=stable_runtime,
        definition_sha256=definition_sha,
        transport_preregistration_sha256=transport_sha,
        registry_manifest_sha256=registry_sha,
    )
    frozen_files = {"synthetic_frozen.json": "7" * 64}
    stable_input = {
        "frozen_definition_files": frozen_files,
        "synthetic_stable_input": True,
    }

    quarantine_paths = []
    quarantine_hashes = {}
    for index, (_, expected_hash) in enumerate(
        v23._OLD_QUARANTINE_HASHES.items()
    ):
        path = root / f"old_quarantine_{index}.bin"
        payload = f"old-quarantine-{index}\n".encode()
        path.write_bytes(payload)
        quarantine_paths.append(path)
        quarantine_hashes[path] = hashlib.sha256(payload).hexdigest()
    old_result = root / "old_result_must_remain_absent.json"
    old_latest = root / "old_latest_must_remain_absent.json"

    monkeypatch.setattr(v23, "REPO_ROOT", root)
    monkeypatch.setattr(v23, "RESULT_PATH", result)
    monkeypatch.setattr(v23, "ATTEMPT_MARKER_PATH", marker)
    monkeypatch.setattr(
        v23, "WRAPPER_DEFINITION_PATH", wrapper_definition_path
    )
    monkeypatch.setattr(v23, "TRANSPORT_PREREGISTRATION_PATH", transport_path)
    monkeypatch.setattr(v23, "DEPENDENCY_MANIFEST_PATH", dependency_path)
    monkeypatch.setattr(v23, "AUTHORIZATION_PATH", authorization_path)
    monkeypatch.setattr(
        v23, "OLD_CANONICAL_RESULT_PATH", old_result
    )
    monkeypatch.setattr(v23, "LATEST_RESULT_PATH", old_latest)
    monkeypatch.setattr(v23, "_OLD_QUARANTINE_HASHES", quarantine_hashes)
    monkeypatch.setattr(
        v23,
        "_load_transport_preregistration",
        lambda: ({"synthetic": True}, transport_sha),
    )
    monkeypatch.setattr(v23, "_load_dependency_manifest", lambda: manifest)
    monkeypatch.setattr(
        v23,
        "_load_wrapper_definition",
        lambda: (definition, definition_sha),
    )
    monkeypatch.setattr(v23, "_require_source_execution_mode", lambda: None)
    monkeypatch.setattr(
        v23, "_verify_loaded_local_modules_use_source", lambda definition: None
    )
    monkeypatch.setattr(
        v23,
        "_source_fingerprints",
        lambda definition: dict(source_fingerprints),
    )
    monkeypatch.setattr(
        v23, "_stable_runtime_identity", lambda: dict(stable_runtime)
    )
    monkeypatch.setattr(
        v23,
        "_load_and_verify_authorization",
        lambda **kwargs: verified,
    )
    monkeypatch.setattr(
        v23,
        "_verify_ticket_frozen_files_before_loader",
        lambda verified: dict(frozen_files),
    )
    monkeypatch.setattr(
        frozen,
        "_load_frozen_contract",
        lambda: fixture["contract"],
    )
    monkeypatch.setattr(
        v23,
        "_verify_contract_authorization",
        lambda verified, contract: (
            ordered_names,
            {
                case.name: frozen._case_identity_payload(case)["sha256"]
                for case in frozen.frozen_history_cases()
            },
            registry_sha,
        ),
    )
    monkeypatch.setattr(
        v23,
        "_stable_execution_input",
        lambda **kwargs: copy.deepcopy(stable_input),
    )
    monkeypatch.setattr(
        v23,
        "_loaded_dependency_state",
        lambda: copy.deepcopy(state_box["value"]),
    )

    collector_calls = {"count": 0}

    def fake_collector(contract):
        assert contract is fixture["contract"]
        collector_calls["count"] += 1
        sys.path.insert(0, str(root))
        try:
            __import__(required.stem)
        finally:
            sys.path.remove(str(root))
        state_box["value"] = copy.deepcopy(complete_state)
        if violate_after_collector:
            state_box["value"][str(unauthorized)] = dict(
                unauthorized_fingerprint
            )
        return fixture["no_go_observations"]

    return {
        "result": result,
        "marker": marker,
        "old_result": old_result,
        "old_latest": old_latest,
        "quarantine_hashes": quarantine_hashes,
        "manifest": manifest,
        "verified": verified,
        "collector": fake_collector,
        "collector_calls": collector_calls,
        "required_module": required.stem,
        "baseline_path": baseline,
        "required_path": required,
    }


@pytest.fixture
def dependency_files(tmp_path):
    baseline = tmp_path / "baseline.py"
    required = tmp_path / "required_lazy.py"
    optional = tmp_path / "optional.py"
    unauthorized = tmp_path / "same_package_unauthorized.py"
    native = tmp_path / "unauthorized_native.so"
    for path, content in (
        (baseline, b"BASELINE = True\n"),
        (required, b"REQUIRED = True\n"),
        (optional, b"OPTIONAL = True\n"),
        (unauthorized, b"UNAUTHORIZED = True\n"),
        (native, b"not-a-real-elf-definition-control\n"),
    ):
        path.write_bytes(content)
    return {
        "baseline": baseline,
        "required": required,
        "optional": optional,
        "unauthorized": unauthorized,
        "native": native,
    }


def test_manifest_schema_is_exact_per_file_and_rejects_broad_authority(
    dependency_files,
):
    files = dependency_files
    payload = _payload(
        baseline=[files["baseline"]],
        required_lazy=[files["required"]],
    )
    manifest = v23._parse_dependency_manifest(
        json.dumps(payload).encode()
    )
    assert manifest.B == (str(files["baseline"]),)
    assert manifest.R == (str(files["required"]),)
    assert set(manifest.U) == {
        str(files["baseline"]),
        str(files["required"]),
    }

    mutations = []
    wildcard = copy.deepcopy(payload)
    wildcard["members"][0]["canonical_path"] = (
        str(files["baseline"].parent) + "/*.py"
    )
    wildcard["B"][0] = wildcard["members"][0]["canonical_path"]
    wildcard["U"][0] = wildcard["members"][0]["canonical_path"]
    mutations.append(wildcard)

    prefix = copy.deepcopy(payload)
    prefix["members"][0]["canonical_path"] = str(
        files["baseline"].parent
    )
    prefix["B"][0] = prefix["members"][0]["canonical_path"]
    prefix["U"][0] = prefix["members"][0]["canonical_path"]
    mutations.append(prefix)

    version_only = copy.deepcopy(payload)
    version_only["members"][0][
        "module_and_distribution_identity"
    ] = {"version": "1.0"}
    mutations.append(version_only)

    duplicate = copy.deepcopy(payload)
    duplicate["members"].append(copy.deepcopy(duplicate["members"][0]))
    mutations.append(duplicate)

    missing_field = copy.deepcopy(payload)
    missing_field["members"][0].pop("st_ino")
    mutations.append(missing_field)

    bad_profile = copy.deepcopy(payload)
    bad_profile["assurance_profile"] = "ADVERSARIAL_LOCAL_WRITER"
    mutations.append(bad_profile)

    overlapping = copy.deepcopy(payload)
    overlapping["R"].append(overlapping["B"][0])
    mutations.append(overlapping)

    for mutation in mutations:
        with pytest.raises(v23.OneShotAuthorizationError):
            v23._parse_dependency_manifest(mutation)


def test_stable_runtime_identity_has_no_dynamic_dependency_snapshot():
    identity = v23._stable_runtime_identity()
    assert "loaded_native_binary_fingerprints" not in identity
    assert "loaded_python_dependency_fingerprints" not in identity
    assert "dependency_snapshot" not in identity
    assert "dependency_closure_snapshot" not in identity


def test_leggauss_lazy_growth_does_not_change_stable_runtime_identity():
    script = r"""
import json
import sys
from pathlib import Path
from unittest.mock import Mock

sys.path.insert(0, "platform")
import actual_wake_reachable_pressure_obstruction_v23_one_shot as v23

forbidden = AssertionError("formal path called in leggauss regression")
collector = Mock(side_effect=forbidden)
mesh = Mock(side_effect=forbidden)
march = Mock(side_effect=forbidden)
v23.guard._collect_frozen_histories = collector
v23.guard.build_canonical_diamond_wing = mesh
v23.guard.march_actual_boundary_material_wake_explicit_midpoint = march

stable_before = v23._stable_runtime_identity()
modules_before = set(sys.modules)
dependencies_before = v23._loaded_dependency_state()
from numpy.polynomial.legendre import leggauss
nodes, weights = leggauss(8)
modules_after = set(sys.modules)
stable_after = v23._stable_runtime_identity()
dependencies_after = v23._loaded_dependency_state()
before_paths = set(dependencies_before)
after_paths = set(dependencies_after)
added_paths = sorted(after_paths - before_paths)
removed_paths = sorted(before_paths - after_paths)

members = []
for path in sorted(after_paths):
    entry = dependencies_after[path]
    module_names = entry.get("module_names") or []
    members.append({
        "canonical_path": path,
        "module_and_distribution_identity": {
            "module": (
                ",".join(module_names)
                if module_names
                else f"native:{Path(path).name}"
            ),
            "distribution": "observed-runtime-file",
            "version": "definition-control",
        },
        "kind": entry["kind"],
        "origin_or_package": path,
        "sha256": entry["sha256"],
        "st_dev": entry["st_dev"],
        "st_ino": entry["st_ino"],
        "st_size": entry["st_size"],
        "allowed_phase": (
            "baseline_pre_marker"
            if path in before_paths
            else "declared_formal_path"
        ),
        "required_or_optional": "required",
        "justification": "observed exact file in leggauss regression",
    })
manifest = v23._parse_dependency_manifest({
    "schema_version": "1.0",
    "artifact": (
        "actual_wake_reachable_pressure_dependency_closure_v23_"
        "20260728_185556"
    ),
    "attempt_id": (
        "S3ai-v2.2-transport-v2.3-successor-20260728_185556"
    ),
    "transport_protocol_version": "S3ai-v2.3",
    "assurance_profile": "RESEARCH_ACCIDENTAL_DRIFT",
    "members": members,
    "B": sorted(before_paths),
    "U": sorted(after_paths),
    "R": added_paths,
})
ledger = v23._DependencyLedger(manifest)
start = ledger.checkpoint(
    phase="baseline_pre_marker",
    actual_state=dependencies_before,
    require_baseline=True,
)
ledger.checkpoint(
    phase="declared_formal_path",
    actual_state=dependencies_after,
)
end = ledger.checkpoint(
    phase="post_collector_pre_publication",
    actual_state=dependencies_after,
    require_completion=True,
)
closure = v23._dependency_closure_provenance(
    manifest=manifest,
    ledger=ledger,
    start_snapshot=start,
    end_snapshot=end,
)
v23._validate_dependency_closure_provenance(
    closure, expected_manifest=manifest
)
print(json.dumps({
    "stable_equal": stable_before == stable_after,
    "new_modules": sorted(modules_after - modules_before),
    "dependency_delta": closure["delta_paths"],
    "added_paths": added_paths,
    "removed_paths": removed_paths,
    "required_subset_ever_seen": (
        set(manifest.R) <= set(closure["ever_seen_paths"])
    ),
    "node_count": len(nodes),
    "weight_count": len(weights),
    "formal_calls": {
        "collector": collector.call_count,
        "mesh": mesh.call_count,
        "march": march.call_count,
    },
}, sort_keys=True))
"""
    completed = subprocess.run(
        [sys.executable, "-B", "-c", script],
        cwd=PLATFORM.parent,
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    observed = json.loads(completed.stdout)
    assert observed["stable_equal"] is True
    assert observed["node_count"] == observed["weight_count"] == 8
    assert "numpy.polynomial.legendre" in observed["new_modules"]
    assert observed["dependency_delta"] == observed["added_paths"]
    assert observed["added_paths"]
    assert observed["removed_paths"] == []
    assert observed["required_subset_ever_seen"] is True
    assert observed["formal_calls"] == {
        "collector": 0,
        "mesh": 0,
        "march": 0,
    }


def test_public_and_private_entries_fail_closed_before_any_collector(
    monkeypatch,
):
    fake_collector = Mock(
        side_effect=AssertionError("private fake collector was called")
    )
    retired_sha = v23._OLD_AUTHORIZATION_SHA256
    with pytest.raises(v23.OneShotAuthorizationError, match="retired"):
        v23.run_authorized_once(
            expected_authorization_sha256=retired_sha,
            expected_implementation_identity_core_raw_sha256="7" * 64,
            second_audit_token=b"definition-only",
        )
    with pytest.raises(v23.OneShotAuthorizationError, match="retired"):
        v23._run_authorized_once(
            expected_authorization_sha256=retired_sha,
            expected_implementation_identity_core_raw_sha256="7" * 64,
            second_audit_token=b"definition-only",
            collector=frozen._collect_frozen_histories,
        )
    with pytest.raises(v23.OneShotAuthorizationError, match="explicit"):
        v23._run_authorized_once(
            expected_authorization_sha256="0" * 64,
            expected_implementation_identity_core_raw_sha256="7" * 64,
            second_audit_token=b"definition-only",
            collector=None,
        )
    with pytest.raises(
        v23.OneShotAuthorizationError, match="canonical v2.3 output"
    ):
        v23._run_authorized_once(
            expected_authorization_sha256="0" * 64,
            expected_implementation_identity_core_raw_sha256="7" * 64,
            second_audit_token=b"definition-only",
            collector=fake_collector,
        )
    monkeypatch.setattr(
        v23,
        "ATTEMPT_MARKER_PATH",
        v23._FORMAL_ATTEMPT_MARKER_PATH.with_name(
            "synthetic_alternate_marker.lock"
        ),
    )
    with pytest.raises(
        v23.OneShotAuthorizationError, match="canonical v2.3 output"
    ):
        v23._run_authorized_once(
            expected_authorization_sha256="0" * 64,
            expected_implementation_identity_core_raw_sha256="7" * 64,
            second_audit_token=b"definition-only",
            collector=fake_collector,
        )
    fake_collector.assert_not_called()


def test_retired_authorization_id_and_token_commitment_are_rejected(
    dependency_files,
):
    files = dependency_files
    manifest = v23._parse_dependency_manifest(
        _payload(
            baseline=[files["baseline"]],
            required_lazy=[files["required"]],
        )
    )
    sources = _synthetic_source_fingerprints()
    runtime = {"synthetic_v23_runtime": True}
    verified = _synthetic_verified(
        manifest=manifest,
        source_fingerprints=sources,
        stable_runtime=runtime,
        definition_sha256="a" * 64,
        transport_preregistration_sha256="b" * 64,
    )
    old_id_payload = copy.deepcopy(verified.payload)
    old_id_payload["authorization"]["id"] = v23._OLD_AUTHORIZATION_ID
    retired_id = replace(verified, payload=old_id_payload)
    retired_token = replace(
        verified, token_sha256=v23._OLD_TOKEN_SHA256
    )
    for candidate in (retired_id, retired_token):
        with pytest.raises(
            v23.OneShotAuthorizationError, match="exact v2.3 successor"
        ):
            v23._require_verified_v23_identity(
                candidate,
                transport_preregistration_sha256=(
                    verified.transport_preregistration_sha256
                ),
                dependency_manifest=manifest,
                source_fingerprints=sources,
                stable_runtime_identity=runtime,
            )


def test_authorized_lazy_growth_is_ledgered_without_snapshot_equality(
    dependency_files,
):
    files = dependency_files
    manifest = v23._parse_dependency_manifest(
        _payload(
            baseline=[files["baseline"]],
            required_lazy=[files["required"]],
        )
    )
    ledger = v23._DependencyLedger(manifest)
    start = ledger.checkpoint(
        phase="baseline_pre_marker",
        actual_state=_state(manifest, [files["baseline"]]),
        require_baseline=True,
    )
    ledger.observe_import_request(
        "required_lazy",
        phase="declared_formal_path",
        fromlist=(),
        level=0,
    )
    ledger.checkpoint(
        phase="declared_formal_path",
        actual_state=_state(
            manifest, [files["baseline"], files["required"]]
        ),
    )
    end = ledger.checkpoint(
        phase="post_collector_pre_publication",
        actual_state=_state(
            manifest, [files["baseline"], files["required"]]
        ),
        require_completion=True,
    )
    assert start != end
    assert start["loaded_paths"] == [str(files["baseline"])]
    assert end["ever_seen_paths"] == sorted(
        [str(files["baseline"]), str(files["required"])]
    )
    provenance = v23._dependency_closure_provenance(
        manifest=manifest,
        ledger=ledger,
        start_snapshot=start,
        end_snapshot=end,
    )
    v23._validate_dependency_closure_provenance(
        provenance, expected_manifest=manifest
    )
    assert provenance["delta_paths"] == [str(files["required"])]


def test_baseline_extra_or_missing_fails_before_any_marker(dependency_files):
    files = dependency_files
    manifest = v23._parse_dependency_manifest(
        _payload(
            baseline=[files["baseline"]],
            required_lazy=[files["required"]],
        )
    )
    for paths in (
        [],
        [files["baseline"], files["required"]],
    ):
        ledger = v23._DependencyLedger(manifest)
        with pytest.raises(
            v23.OneShotAuthorizationError, match="S0 == B"
        ):
            ledger.checkpoint(
                phase="baseline_pre_marker",
                actual_state=_state(manifest, paths),
                require_baseline=True,
            )


def test_unauthorized_python_same_package_and_native_members_fail(
    dependency_files,
):
    files = dependency_files
    manifest = v23._parse_dependency_manifest(
        _payload(
            baseline=[files["baseline"]],
            required_lazy=[files["required"]],
        )
    )
    ledger = v23._DependencyLedger(manifest)
    ledger.checkpoint(
        phase="baseline_pre_marker",
        actual_state=_state(manifest, [files["baseline"]]),
        require_baseline=True,
    )
    baseline_state = _state(manifest, [files["baseline"]])
    for path, kind in (
        (files["unauthorized"], "python_source"),
        (files["native"], "native_binary"),
    ):
        metadata = path.stat()
        unauthorized = {
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "st_dev": int(metadata.st_dev),
            "st_ino": int(metadata.st_ino),
            "st_size": int(metadata.st_size),
            "kind": kind,
            "module_names": [path.stem],
        }
        with pytest.raises(
            v23.OneShotAuthorizationError, match="unauthorized"
        ):
            ledger.checkpoint(
                phase="declared_formal_path",
                actual_state={
                    **baseline_state,
                    str(path): unauthorized,
                },
            )


def test_registered_member_removal_is_not_erased_from_E(dependency_files):
    files = dependency_files
    manifest = v23._parse_dependency_manifest(
        _payload(
            baseline=[files["baseline"]],
            required_lazy=[files["required"]],
        )
    )
    ledger = v23._DependencyLedger(manifest)
    ledger.checkpoint(
        phase="baseline_pre_marker",
        actual_state=_state(manifest, [files["baseline"]]),
        require_baseline=True,
    )
    ledger.checkpoint(
        phase="declared_formal_path",
        actual_state=_state(
            manifest, [files["baseline"], files["required"]]
        ),
    )
    with pytest.raises(v23.OneShotAuthorizationError, match="removed"):
        ledger.checkpoint(
            phase="post_collector_pre_publication",
            actual_state=_state(manifest, [files["baseline"]]),
            require_completion=True,
        )
    assert str(files["required"]) in ledger.ever_seen
    assert any(
        event["event_type"] == "registered_member_removed"
        for event in ledger.events
    )


def test_missing_R_wrong_phase_and_unloaded_U_drift_fail(
    dependency_files,
):
    files = dependency_files
    payload = _payload(
        baseline=[files["baseline"]],
        required_lazy=[files["required"]],
        optional=[
            (files["optional"], "post_collector_pre_publication")
        ],
    )
    manifest = v23._parse_dependency_manifest(payload)

    missing = v23._DependencyLedger(manifest)
    missing.checkpoint(
        phase="baseline_pre_marker",
        actual_state=_state(manifest, [files["baseline"]]),
        require_baseline=True,
    )
    with pytest.raises(v23.OneShotAuthorizationError, match="mandatory R"):
        missing.checkpoint(
            phase="post_collector_pre_publication",
            actual_state=_state(manifest, [files["baseline"]]),
            require_completion=True,
        )

    wrong_phase = v23._DependencyLedger(manifest)
    wrong_phase.checkpoint(
        phase="baseline_pre_marker",
        actual_state=_state(manifest, [files["baseline"]]),
        require_baseline=True,
    )
    with pytest.raises(v23.OneShotAuthorizationError, match="wrong"):
        wrong_phase.checkpoint(
            phase="post_collector_pre_publication",
            actual_state=_state(
                manifest, [files["baseline"], files["required"]]
            ),
        )

    drift = v23._DependencyLedger(manifest)
    drift.checkpoint(
        phase="baseline_pre_marker",
        actual_state=_state(manifest, [files["baseline"]]),
        require_baseline=True,
    )
    files["optional"].write_bytes(b"DRIFTED_WHILE_UNLOADED = True\n")
    with pytest.raises(v23.OneShotAuthorizationError, match="drifted"):
        drift.checkpoint(
            phase="declared_formal_path",
            actual_state=_state(manifest, [files["baseline"]]),
        )


def test_byte_inode_symlink_and_pyc_substitution_fail(dependency_files):
    files = dependency_files
    payload = _payload(
        baseline=[files["baseline"]],
        required_lazy=[files["required"]],
    )
    manifest = v23._parse_dependency_manifest(payload)
    original = files["required"].read_bytes()
    files["required"].write_bytes(b"CHANGED_BYTES = True\n")
    with pytest.raises(v23.OneShotAuthorizationError, match="drifted"):
        v23._verify_all_manifest_members(manifest)
    files["required"].write_bytes(original)

    replacement = files["required"].with_name("replacement.py")
    replacement.write_bytes(original)
    os.replace(replacement, files["required"])
    with pytest.raises(v23.OneShotAuthorizationError, match="drifted"):
        v23._verify_all_manifest_members(manifest)

    symlink = files["required"].with_name("alias.py")
    symlink.symlink_to(files["required"])
    symlink_payload = copy.deepcopy(payload)
    symlink_payload["members"][1]["canonical_path"] = str(symlink)
    symlink_payload["R"][0] = str(symlink)
    symlink_payload["U"][1] = str(symlink)
    with pytest.raises(v23.OneShotAuthorizationError, match="symlink"):
        v23._parse_dependency_manifest(symlink_payload)

    pyc = files["required"].with_suffix(".pyc")
    pyc.write_bytes(b"not-bytecode")
    pyc_payload = _payload(
        baseline=[files["baseline"]],
        required_lazy=[files["required"]],
    )
    pyc_member = _member(
        pyc,
        phase="declared_formal_path",
        requirement="required",
    )
    pyc_member["canonical_path"] = str(pyc)
    pyc_payload["members"][1] = pyc_member
    pyc_payload["R"][0] = str(pyc)
    pyc_payload["U"][1] = str(pyc)
    with pytest.raises(v23.OneShotAuthorizationError, match="\\.py source"):
        v23._parse_dependency_manifest(pyc_payload)


def test_serialized_closure_fields_and_digests_are_read_only(
    dependency_files,
):
    files = dependency_files
    manifest = v23._parse_dependency_manifest(
        _payload(
            baseline=[files["baseline"]],
            required_lazy=[files["required"]],
        )
    )
    ledger = v23._DependencyLedger(manifest)
    start = ledger.checkpoint(
        phase="baseline_pre_marker",
        actual_state=_state(manifest, [files["baseline"]]),
        require_baseline=True,
    )
    end = ledger.checkpoint(
        phase="declared_formal_path",
        actual_state=_state(
            manifest, [files["baseline"], files["required"]]
        ),
    )
    end = ledger.checkpoint(
        phase="post_collector_pre_publication",
        actual_state=_state(
            manifest, [files["baseline"], files["required"]]
        ),
        require_completion=True,
    )
    provenance = v23._dependency_closure_provenance(
        manifest=manifest,
        ledger=ledger,
        start_snapshot=start,
        end_snapshot=end,
    )
    v23._validate_dependency_closure_provenance(
        provenance, expected_manifest=manifest
    )
    mutations = (
        lambda item: item["manifest"].pop("raw_sha256"),
        lambda item: item["start"].update({"snapshot_sha256": "0" * 64}),
        lambda item: item["end"]["loaded_paths"].pop(),
        lambda item: item["delta_paths"].clear(),
        lambda item: item["import_load_ledger"]["events"].pop(),
        lambda item: item.update(
            {"closure_certificate_sha256": "0" * 64}
        ),
    )
    for mutate in mutations:
        tampered = copy.deepcopy(provenance)
        mutate(tampered)
        with pytest.raises(v23.OneShotAuthorizationError):
            v23._validate_dependency_closure_provenance(
                tampered, expected_manifest=manifest
            )


def test_serialized_closure_is_anchored_to_the_expected_manifest(
    dependency_files,
):
    files = dependency_files
    manifest = v23._parse_dependency_manifest(
        _payload(
            baseline=[files["baseline"]],
            required_lazy=[files["required"]],
        )
    )
    other_manifest = v23._parse_dependency_manifest(
        _payload(
            baseline=[files["baseline"]],
            required_lazy=[],
            optional=[
                (files["optional"], "post_collector_pre_publication")
            ],
        )
    )
    ledger = v23._DependencyLedger(manifest)
    start = ledger.checkpoint(
        phase="baseline_pre_marker",
        actual_state=_state(manifest, [files["baseline"]]),
        require_baseline=True,
    )
    ledger.checkpoint(
        phase="declared_formal_path",
        actual_state=_state(
            manifest, [files["baseline"], files["required"]]
        ),
    )
    end = ledger.checkpoint(
        phase="post_collector_pre_publication",
        actual_state=_state(
            manifest, [files["baseline"], files["required"]]
        ),
        require_completion=True,
    )
    provenance = v23._dependency_closure_provenance(
        manifest=manifest,
        ledger=ledger,
        start_snapshot=start,
        end_snapshot=end,
    )
    with pytest.raises(
        v23.OneShotAuthorizationError, match="authorization-bound"
    ):
        v23._validate_dependency_closure_provenance(
            provenance, expected_manifest=other_manifest
        )


def test_native_loaded_state_requires_proc_maps_identity(
    dependency_files,
    monkeypatch,
):
    native = dependency_files["native"]
    payload = _payload(
        baseline=[],
        required_lazy=[],
        optional=[(native, "declared_formal_path")],
    )
    payload["members"][0]["kind"] = "native_binary"
    manifest = v23._parse_dependency_manifest(payload)
    member = manifest.member_by_path[str(native)]
    incomplete = {
        **v23._manifest_member_fingerprint(member),
        "module_names": ["synthetic_native"],
    }
    ledger = v23._DependencyLedger(manifest)
    ledger.checkpoint(
        phase="baseline_pre_marker",
        actual_state={},
        require_baseline=True,
    )
    with pytest.raises(v23.OneShotAuthorizationError, match="fingerprint"):
        ledger.checkpoint(
            phase="declared_formal_path",
            actual_state={str(native): incomplete},
        )

    fake_module = SimpleNamespace(
        __spec__=SimpleNamespace(origin=str(native)),
        __file__=str(native),
    )
    monkeypatch.setattr(v23.sys, "modules", {"synthetic_native": fake_module})
    monkeypatch.setattr(v23, "_native_mapping_identities", lambda: {})
    with pytest.raises(v23.OneShotAuthorizationError, match="/proc/self/maps"):
        v23._loaded_dependency_state()


def test_atomic_publication_rolls_back_owned_link_on_post_link_failure(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(v23, "REPO_ROOT", tmp_path)
    result = tmp_path / "synthetic_result.json"
    real_fsync = v23.os.fsync
    calls = {"count": 0}

    def fail_once_after_link(fd):
        calls["count"] += 1
        if calls["count"] == 2:
            raise OSError("synthetic post-link fsync failure")
        return real_fsync(fd)

    monkeypatch.setattr(v23.os, "fsync", fail_once_after_link)
    with pytest.raises(
        v23.OneShotAuthorizationError, match="atomic no-replace"
    ):
        v23._atomic_create_json_no_replace(
            result,
            {"stage_decision": "PROTOCOL-NO-GO"},
        )
    assert calls["count"] >= 3
    assert not result.exists()
    assert list(tmp_path.glob(".synthetic_result.json.tmp.*")) == []


def test_atomic_pre_link_gate_failure_never_creates_result(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(v23, "REPO_ROOT", tmp_path)
    result = tmp_path / "synthetic_pre_link_result.json"
    gate = Mock(
        side_effect=v23.OneShotAuthorizationError(
            "synthetic closure drift at pre-link gate"
        )
    )
    with pytest.raises(
        v23.OneShotAuthorizationError, match="synthetic closure drift"
    ):
        v23._atomic_create_json_no_replace(
            result,
            {"stage_decision": "PROTOCOL-NO-GO"},
            pre_link_gate=gate,
        )
    gate.assert_called_once_with()
    assert not result.exists()
    assert list(
        tmp_path.glob(".synthetic_pre_link_result.json.tmp.*")
    ) == []


def test_fake_no_go_with_legal_lazy_growth_publishes_full_result(
    tmp_path,
    monkeypatch,
):
    actual_old_assets = dict(v23._OLD_QUARANTINE_HASHES)
    actual_old_result = v23.OLD_CANONICAL_RESULT_PATH
    actual_latest = v23.LATEST_RESULT_PATH
    actual_old_before = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in actual_old_assets
    }
    assert actual_old_before == {
        str(path): digest for path, digest in actual_old_assets.items()
    }
    assert not actual_old_result.exists()
    assert not actual_latest.exists()

    fixture = _synthetic_31_case_fixture()
    boundary = _install_fake_run_boundary(
        monkeypatch=monkeypatch,
        root=tmp_path / "legal_no_go",
        fixture=fixture,
        violate_after_collector=False,
    )
    quarantine_before = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in boundary["quarantine_hashes"]
    }
    try:
        receipt = v23._run_authorized_once(
            expected_authorization_sha256=boundary["verified"].raw_sha256,
            expected_implementation_identity_core_raw_sha256="7" * 64,
            second_audit_token=b"s" * 32,
            collector=boundary["collector"],
        )
    finally:
        sys.modules.pop(boundary["required_module"], None)

    assert boundary["collector_calls"]["count"] == 1
    assert receipt.stage_decision == "PROTOCOL-NO-GO"
    assert receipt.result_path == boundary["result"]
    assert boundary["result"].is_file()
    assert boundary["marker"].is_file()
    assert not boundary["old_result"].exists()
    assert not boundary["old_latest"].exists()
    assert receipt.result_sha256 == hashlib.sha256(
        boundary["result"].read_bytes()
    ).hexdigest()
    persisted = json.loads(boundary["result"].read_text())
    assert persisted["stage_decision"] == "PROTOCOL-NO-GO"
    assert persisted["production_activation_allowed"] is False
    assert persisted["one_shot_provenance"]["dependency_closure"][
        "delta_paths"
    ] == list(boundary["manifest"].R)
    events = persisted["one_shot_provenance"]["dependency_closure"][
        "import_load_ledger"
    ]["events"]
    assert any(
        event["event_type"] == "observed_import_request"
        and event["requested_name"] == boundary["required_module"]
        for event in events
    )
    assert any(
        event["event_type"] == "checkpoint_confirmed_successful_load"
        and event["canonical_path"] == boundary["manifest"].R[0]
        for event in events
    )
    v23._validate_serialized_result(
        persisted,
        ordered_case_names=fixture["ordered_case_names"],
        expected_dependency_manifest=boundary["manifest"],
        expected_verified=boundary["verified"],
    )

    alternate_manifest = v23._parse_dependency_manifest(
        _payload(
            baseline=[boundary["baseline_path"]],
            required_lazy=[],
            optional=[
                (
                    boundary["required_path"],
                    "declared_formal_path",
                )
            ],
        )
    )

    def replace_manifest(closure):
        closure["manifest"]["raw_sha256"] = (
            alternate_manifest.raw_sha256
        )
        closure["manifest"]["canonical_sha256"] = (
            alternate_manifest.canonical_sha256
        )
        closure["manifest"]["payload"] = alternate_manifest.as_payload()

    def drift_start_phase(closure):
        closure["start"]["phase"] = "declared_formal_path"
        _refresh_snapshot_certificate(closure["start"])

    def drift_end_phase(closure):
        closure["end"]["phase"] = "declared_formal_path"
        _refresh_snapshot_certificate(closure["end"])

    def drift_delta(closure):
        closure["delta_paths"] = []
        closure["delta_paths_sha256"] = v23._sha256_bytes(
            v23._canonical_json_bytes([])
        )

    def drift_ledger(closure):
        _remove_required_success_and_rehash_ledger(
            closure, boundary["manifest"].R[0]
        )

    for mutate in (
        replace_manifest,
        drift_start_phase,
        drift_end_phase,
        drift_delta,
        drift_ledger,
    ):
        tampered = copy.deepcopy(persisted)
        closure = tampered["one_shot_provenance"]["dependency_closure"]
        mutate(closure)
        _refresh_closure_certificate(closure)
        with pytest.raises(v23.OneShotAuthorizationError):
            v23._validate_serialized_result(
                tampered,
                ordered_case_names=fixture["ordered_case_names"],
                expected_dependency_manifest=boundary["manifest"],
                expected_verified=boundary["verified"],
            )

    quarantine_after = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in boundary["quarantine_hashes"]
    }
    assert quarantine_after == quarantine_before
    actual_old_after = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in actual_old_assets
    }
    assert actual_old_after == actual_old_before
    assert not actual_old_result.exists()
    assert not actual_latest.exists()


def test_post_collector_closure_violation_consumes_marker_and_cannot_retry(
    tmp_path,
    monkeypatch,
):
    fixture = _synthetic_31_case_fixture()
    boundary = _install_fake_run_boundary(
        monkeypatch=monkeypatch,
        root=tmp_path / "post_collector_violation",
        fixture=fixture,
        violate_after_collector=True,
    )
    try:
        with pytest.raises(
            v23.OneShotAuthorizationError, match="unauthorized dependency"
        ):
            v23._run_authorized_once(
                expected_authorization_sha256=(
                    boundary["verified"].raw_sha256
                ),
                expected_implementation_identity_core_raw_sha256="7" * 64,
                second_audit_token=b"t" * 32,
                collector=boundary["collector"],
            )
        assert boundary["collector_calls"]["count"] == 1
        assert boundary["marker"].is_file()
        assert not boundary["result"].exists()
        marker_before = boundary["marker"].read_bytes()
        marker_identity = boundary["marker"].stat()
        with pytest.raises(
            v23.OneShotAuthorizationError, match="attempt marker"
        ):
            v23._run_authorized_once(
                expected_authorization_sha256=(
                    boundary["verified"].raw_sha256
                ),
                expected_implementation_identity_core_raw_sha256="7" * 64,
                second_audit_token=b"t" * 32,
                collector=boundary["collector"],
            )
        assert boundary["collector_calls"]["count"] == 1
        assert boundary["marker"].read_bytes() == marker_before
        marker_after = boundary["marker"].stat()
        assert (
            marker_after.st_dev,
            marker_after.st_ino,
            marker_after.st_size,
        ) == (
            marker_identity.st_dev,
            marker_identity.st_ino,
            marker_identity.st_size,
        )
        assert not boundary["result"].exists()
    finally:
        sys.modules.pop(boundary["required_module"], None)


@pytest.mark.parametrize("preexisting_name", ["result", "marker"])
def test_preexisting_result_or_marker_is_never_overwritten(
    preexisting_name,
    tmp_path,
    monkeypatch,
):
    fixture = _synthetic_31_case_fixture()
    boundary = _install_fake_run_boundary(
        monkeypatch=monkeypatch,
        root=tmp_path / f"preexisting_{preexisting_name}",
        fixture=fixture,
        violate_after_collector=False,
    )
    target = boundary[preexisting_name]
    sentinel = f"third-party-{preexisting_name}\n".encode()
    target.write_bytes(sentinel)
    with pytest.raises(v23.OneShotAuthorizationError, match="must be absent"):
        v23._run_authorized_once(
            expected_authorization_sha256=boundary["verified"].raw_sha256,
            expected_implementation_identity_core_raw_sha256="7" * 64,
            second_audit_token=b"u" * 32,
            collector=boundary["collector"],
        )
    assert boundary["collector_calls"]["count"] == 0
    assert target.read_bytes() == sentinel
    other = (
        boundary["marker"]
        if preexisting_name == "result"
        else boundary["result"]
    )
    assert not other.exists()


def test_concurrent_result_publication_is_preserved_without_overwrite(
    tmp_path,
    monkeypatch,
):
    fixture = _synthetic_31_case_fixture()
    boundary = _install_fake_run_boundary(
        monkeypatch=monkeypatch,
        root=tmp_path / "concurrent_result",
        fixture=fixture,
        violate_after_collector=False,
    )
    competitor = b'{"owner":"concurrent-definition-control"}\n'
    real_link = v23.os.link

    def competing_link(*args, **kwargs):
        boundary["result"].write_bytes(competitor)
        return real_link(*args, **kwargs)

    monkeypatch.setattr(v23.os, "link", competing_link)
    try:
        with pytest.raises(
            v23.OneShotAuthorizationError, match="appeared concurrently"
        ):
            v23._run_authorized_once(
                expected_authorization_sha256=(
                    boundary["verified"].raw_sha256
                ),
                expected_implementation_identity_core_raw_sha256="7" * 64,
                second_audit_token=b"v" * 32,
                collector=boundary["collector"],
            )
    finally:
        sys.modules.pop(boundary["required_module"], None)
    assert boundary["collector_calls"]["count"] == 1
    assert boundary["marker"].is_file()
    assert boundary["result"].read_bytes() == competitor


@pytest.mark.parametrize(
    "drift_kind",
    [
        "stable_runtime",
        "local_source",
        "dependency_u",
        "second_audit_binding",
    ],
)
def test_post_marker_stable_boundary_drift_fails_without_result(
    drift_kind,
    tmp_path,
    monkeypatch,
):
    fixture = _synthetic_31_case_fixture()
    boundary = _install_fake_run_boundary(
        monkeypatch=monkeypatch,
        root=tmp_path / f"drift_{drift_kind}",
        fixture=fixture,
        violate_after_collector=False,
    )
    collector = boundary["collector"]

    if drift_kind == "stable_runtime":
        calls = {"count": 0}
        stable = dict(boundary["verified"].stable_runtime_identity)

        def runtime_with_post_marker_drift():
            calls["count"] += 1
            if calls["count"] <= 2:
                return dict(stable)
            return {**stable, "post_marker_drift": True}

        monkeypatch.setattr(
            v23, "_stable_runtime_identity", runtime_with_post_marker_drift
        )
    elif drift_kind == "local_source":
        calls = {"count": 0}
        sources = dict(boundary["verified"].source_fingerprints)

        def sources_with_post_marker_drift(definition):
            calls["count"] += 1
            if calls["count"] == 1:
                return dict(sources)
            drifted = dict(sources)
            first = next(iter(drifted))
            drifted[first] = "f" * 64
            return drifted

        monkeypatch.setattr(
            v23, "_source_fingerprints", sources_with_post_marker_drift
        )
    elif drift_kind == "dependency_u":
        base_collector = collector

        def collector_with_u_drift(contract):
            observations = base_collector(contract)
            boundary["baseline_path"].write_text(
                "BASELINE_DRIFTED_AFTER_MARKER = True\n"
            )
            return observations

        collector = collector_with_u_drift
    else:
        calls = {"count": 0}
        stable_input = v23._stable_execution_input()

        def input_with_audit_binding_drift(**kwargs):
            calls["count"] += 1
            if calls["count"] == 1:
                return copy.deepcopy(stable_input)
            return {
                **copy.deepcopy(stable_input),
                "second_audit_binding_drift": True,
            }

        monkeypatch.setattr(
            v23, "_stable_execution_input", input_with_audit_binding_drift
        )

    try:
        with pytest.raises(v23.OneShotAuthorizationError):
            v23._run_authorized_once(
                expected_authorization_sha256=(
                    boundary["verified"].raw_sha256
                ),
                expected_implementation_identity_core_raw_sha256="7" * 64,
                second_audit_token=b"w" * 32,
                collector=collector,
            )
    finally:
        sys.modules.pop(boundary["required_module"], None)
    assert boundary["collector_calls"]["count"] == 1
    assert boundary["marker"].is_file()
    assert not boundary["result"].exists()
