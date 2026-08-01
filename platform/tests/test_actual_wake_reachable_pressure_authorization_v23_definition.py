"""No-history definition tests for the transport-v2.3 authorization seam.

The fixtures in this module are deliberately confined to
``platform/tests/fixture-<24 lowercase hex>``.  They never create a production
credential, ticket, marker, result, or history.  Cleanup follows an exact
creation ledger; it never recursively removes an inferred directory.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import copy
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import secrets
import stat
import sys
from unittest.mock import Mock

import pytest


PLATFORM = Path(__file__).resolve().parents[1]
if str(PLATFORM) not in sys.path:
    sys.path.insert(0, str(PLATFORM))

import actual_wake_reachable_pressure_obstruction_v2_guard as frozen  # noqa: E402
import actual_wake_reachable_pressure_obstruction_v23_one_shot as v23  # noqa: E402


_FORMAL_PATHS = (
    v23.AUTHORIZATION_PATH,
    v23.RESULT_PATH,
    v23.ATTEMPT_MARKER_PATH,
)


def _path_observation(path: Path) -> tuple[str, str | None]:
    """Return a read-only identity/hash observation for a formal path."""

    try:
        metadata = os.lstat(path)
    except FileNotFoundError:
        return ("absent", None)
    if stat.S_ISREG(metadata.st_mode):
        return ("regular", hashlib.sha256(path.read_bytes()).hexdigest())
    return (
        f"mode:{stat.S_IFMT(metadata.st_mode)}",
        f"{int(metadata.st_dev)}:{int(metadata.st_ino)}",
    )


@pytest.fixture(autouse=True)
def _forbid_formal_history_and_publication(monkeypatch):
    """Make every scientific or formal-publication call a hard test failure."""

    before = {path: _path_observation(path) for path in _FORMAL_PATHS}
    forbidden = AssertionError(
        "authorization definition test reached a formal/history side effect"
    )
    sentinels: list[Mock] = []
    for owner, name in (
        (frozen, "_collect_frozen_histories"),
        (frozen, "build_canonical_diamond_wing"),
        (
            frozen,
            "march_actual_boundary_material_wake_explicit_midpoint",
        ),
        (frozen, "_load_frozen_contract"),
        (frozen, "aggregate_frozen_histories"),
        (v23, "_claim_attempt_once"),
        (v23, "_acquire_output_directory_lease"),
        (v23, "_atomic_create_json_no_replace"),
    ):
        sentinel = Mock(side_effect=forbidden)
        monkeypatch.setattr(owner, name, sentinel)
        sentinels.append(sentinel)

    yield

    for sentinel in sentinels:
        sentinel.assert_not_called()
    after = {path: _path_observation(path) for path in _FORMAL_PATHS}
    assert after == before


@dataclass
class _FixtureLedger:
    """Exact per-test namespace and cleanup ledger."""

    profile: v23._FixtureNamespaceProfileV1
    files: list[Path] = field(default_factory=list)
    directories: list[Path] = field(default_factory=list)

    @property
    def root(self) -> Path:
        return self.profile.isolation_root_lease.absolute_path

    def write_file(self, path: Path, raw: bytes) -> None:
        try:
            path.relative_to(self.root)
        except ValueError as error:
            raise AssertionError("fixture write escaped its root") from error
        if path.parent != self.root and path.parent not in self.directories:
            raise AssertionError("fixture parent is absent from the ledger")
        fd = os.open(
            path,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
            0o600,
        )
        try:
            view = memoryview(raw)
            while view:
                written = os.write(fd, view)
                if written <= 0:
                    raise AssertionError("fixture write made no progress")
                view = view[written:]
            os.fsync(fd)
        finally:
            os.close(fd)
        self.files.append(path)

    def mkdir(self, path: Path) -> None:
        try:
            path.relative_to(self.root)
        except ValueError as error:
            raise AssertionError("fixture directory escaped its root") from error
        if path.parent != self.root and path.parent not in self.directories:
            raise AssertionError("fixture parent directory is not ledgered")
        os.mkdir(path, 0o700)
        self.directories.append(path)

    def symlink(self, path: Path, target: str) -> None:
        if path.parent != self.root:
            raise AssertionError("foundation fixture links only direct files")
        os.symlink(
            target,
            path.name,
            dir_fd=self.profile.isolation_root_lease.fd,
        )
        self.files.append(path)

    def cleanup(self) -> None:
        v23._close_namespace_profile_v1(self.profile)
        for path in reversed(self.files):
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass
        for path in reversed(self.directories):
            try:
                os.rmdir(path)
            except FileNotFoundError:
                pass
        os.rmdir(self.root)


@pytest.fixture
def fixture_namespace() -> _FixtureLedger:
    fixture_id = f"fixture-{secrets.token_hex(12)}"
    ledger = _FixtureLedger(
        profile=v23._create_fixture_namespace_profile_v1(fixture_id)
    )
    try:
        yield ledger
    finally:
        ledger.cleanup()


def _independent_pretty_json(value: object) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _independent_canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _valid_identity_core() -> dict[str, str]:
    """An independently spelled fixture core, not a production credential."""

    return {
        "agent_id": "fixture-implementation-agent",
        "artifact": (
            "actual_wake_reachable_pressure_v23_"
            "implementation_identity_core"
        ),
        "model": "fixture-implementation-model",
        "model_family": "fixture-implementation-family",
        "provider": "fixture-implementation-provider",
        "schema_version": "1.0",
        "source_kind": (
            "external_implementation_orchestrator_trace_identity"
        ),
        "trace_id": "fixture-implementation-trace",
        "version": "S3ai-v2.3-implementation-identity-core-v1",
    }


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _repo_relative(path: Path) -> str:
    return path.relative_to(v23.REPO_ROOT).as_posix()


def _binding(path: Path) -> dict[str, str]:
    return {"path": _repo_relative(path), "sha256": _sha256(path.read_bytes())}


def _minimal_dependency_manifest_raw() -> bytes:
    path = Path(__file__).resolve()
    metadata = path.stat()
    member = {
        "allowed_phase": "baseline_pre_marker",
        "canonical_path": str(path),
        "justification": "authorization definition fixture source",
        "kind": "python_source",
        "module_and_distribution_identity": {
            "distribution": "synthetic-definition-only",
            "module": Path(__file__).stem,
            "version": "0",
        },
        "origin_or_package": "synthetic-definition-only",
        "required_or_optional": "required",
        "sha256": _sha256(path.read_bytes()),
        "st_dev": int(metadata.st_dev),
        "st_ino": int(metadata.st_ino),
        "st_size": int(metadata.st_size),
    }
    return _independent_pretty_json(
        {
            "B": [str(path)],
            "R": [],
            "U": [str(path)],
            "artifact": (
                "actual_wake_reachable_pressure_dependency_closure_"
                "v23_20260728_185556"
            ),
            "assurance_profile": "RESEARCH_ACCIDENTAL_DRIFT",
            "attempt_id": (
                "S3ai-v2.2-transport-v2.3-successor-20260728_185556"
            ),
            "members": [member],
            "schema_version": "1.0",
            "transport_protocol_version": "S3ai-v2.3",
        }
    )


@dataclass(frozen=True)
class _ProjectionFixture:
    token: bytes
    core_sha256: str
    identity: dict[str, object]
    definition: dict[str, object]
    dependency_manifest: v23.DependencyManifest
    source_fingerprints: dict[str, str]
    runtime_identity: dict[str, object]
    bound_artifacts: dict[str, dict[str, str]]
    plan: v23._FrozenTicketSemanticPlanInputsV1
    projection: dict[str, object]


@dataclass(frozen=True)
class _FullAuthorizationFixture:
    projection_fixture: _ProjectionFixture
    q: dict[str, object]
    m: dict[str, object]
    a: dict[str, object]
    c: dict[str, object]
    z: dict[str, object]
    ticket: dict[str, object]
    ticket_raw_sha256: str
    q_path: str
    m_path: str
    a_path: str
    c_path: str
    z_path: str


def _build_projection_fixture(
    ledger: _FixtureLedger,
) -> _ProjectionFixture:
    """Build the complete pre-Q causal input using only fixture credentials."""

    profile = ledger.profile
    core = _valid_identity_core()
    core_raw = _independent_pretty_json(core)
    core_sha = _sha256(core_raw)
    ledger.write_file(profile.implementation_identity_core_path, core_raw)
    seed = v23._build_identity_core_binding_seed_v1(profile, core_sha)

    definition = {
        "frozen_definition_chain": {
            name: {
                "file": f"fixture-{name}.json",
                "sha256": _sha256(name.encode("utf-8")),
            }
            for name in (
                "S3ai-v2",
                "S3ai-v2.1",
                "S3ai-v2.2",
                "implementation_audit",
            )
        }
    }
    dependency_raw = _minimal_dependency_manifest_raw()
    dependency_manifest = v23._parse_dependency_manifest(dependency_raw)

    artifact_raw: dict[str, bytes] = {
        "bootstrap_contract": _independent_pretty_json(
            {"artifact": "fixture-bootstrap-contract", "schema_version": "1.0"}
        ),
        "dependency_capture_protocol": b"fixture dependency capture\n",
        "dependency_manifest": dependency_raw,
        "g0_quarantine_audit": b"fixture G0 audit\n",
        "g2_implementation_diff_audit": b"fixture G2 audit\n",
        "g3_capture_instantiation": b"fixture G3 capture\n",
        "g3_consensus_certificate": _independent_pretty_json(
            {"artifact": "fixture-consensus", "schema_version": "1.0"}
        ),
        "g3_independent_audit": b"fixture G3 independent audit\n",
        "g4_definition_test_audit": b"fixture G4 audit\n",
        "interpretation_contract_json": _independent_pretty_json(
            {"artifact": "fixture-interpretation-contract"}
        ),
        "interpretation_contract_markdown": b"# fixture interpretation\n",
        "transport_preregistration": b"fixture transport preregistration\n",
        "wrapper_definition": _independent_pretty_json(definition),
    }
    artifact_paths: dict[str, Path] = {}
    for role, raw in artifact_raw.items():
        suffix = ".json" if raw.startswith(b"{") else ".bin"
        path = ledger.root / f"bound-{role}{suffix}"
        ledger.write_file(path, raw)
        artifact_paths[role] = path

    source_fingerprints = {
        relative: _sha256((v23.REPO_ROOT / relative).read_bytes())
        for relative in (
            v23._WRAPPER_RELATIVE,
            (
                "platform/"
                "actual_wake_reachable_pressure_obstruction_v2_guard.py"
            ),
        )
    }
    source_map = {
        "g2_implementation_diff_audit": _binding(
            artifact_paths["g2_implementation_diff_audit"]
        ),
        "g4_definition_test_audit": _binding(
            artifact_paths["g4_definition_test_audit"]
        ),
        "wrapper_source": {
            "path": v23._WRAPPER_RELATIVE,
            "sha256": source_fingerprints[v23._WRAPPER_RELATIVE],
        },
    }

    trace_parent = ledger.root
    for component in (".aris", "traces", "research-review"):
        trace_parent = trace_parent / component
        ledger.mkdir(trace_parent)
    trace_path = trace_parent / "fixture-implementation-source.trace.json"
    trace = {
        "agent_id": core["agent_id"],
        "artifact": (
            "actual_wake_reachable_pressure_v23_"
            "implementation_source_trace"
        ),
        "implementation_identity_core_path": seed.path,
        "implementation_identity_core_raw_sha256": (
            seed.observed_raw_sha256
        ),
        "model": core["model"],
        "model_family": core["model_family"],
        "provider": core["provider"],
        "schema_version": "1.0",
        "source_artifact_map_sha256": _sha256(
            _independent_canonical_json(source_map)
        ),
        "source_kind": "implementation_and_definition_audit_trace",
        "trace_id": core["trace_id"],
        "version": "S3ai-v2.3-implementation-source-trace-v1",
    }
    trace_raw = _independent_pretty_json(trace)
    ledger.write_file(trace_path, trace_raw)
    identity: dict[str, object] = {
        name: core[name]
        for name in (
            "provider",
            "model",
            "model_family",
            "agent_id",
            "trace_id",
        )
    }
    identity.update(
        {
            "source_trace_metadata_path": _repo_relative(trace_path),
            "source_trace_metadata_sha256": _sha256(trace_raw),
        }
    )
    anchor_path = ledger.root / "bound-implementation-identity.json"
    ledger.write_file(
        anchor_path,
        _independent_pretty_json(
            {
                "artifact": (
                    "actual_wake_reachable_pressure_v23_"
                    "implementation_identity"
                ),
                "identity": identity,
                "schema_version": "1.0",
                "version": "S3ai-v2.3-implementation-identity-v1",
            }
        ),
    )

    bound_artifacts = {
        role: _binding(path) for role, path in artifact_paths.items()
    }
    bound_artifacts.update(
        {
            "authorization_schema_preregistration": {
                "path": (
                    "platform/docs/diag/"
                    "actual_wake_reachable_pressure_authorization_schema_"
                    "v23_preregistration_20260728_215354.md"
                ),
                "sha256": (
                    "77881f64788bbb0edaa1a2ce43ffd911f59b22a44eaa5b6"
                    "cde0f1723100a921f"
                ),
            },
            "g6_otmpfile_capability_probe": {
                "path": (
                    "platform/docs/diag/"
                    "actual_wake_reachable_pressure_g6_otmpfile_"
                    "capability_probe_20260728_213611.json"
                ),
                "sha256": (
                    "17a363d74ebb244c851ba222b93100415ce60c084c8bee111"
                    "cfda920c3ee7968"
                ),
            },
            "implementation_identity": _binding(anchor_path),
            "implementation_identity_core": {
                "path": seed.path,
                "sha256": seed.observed_raw_sha256,
            },
            "wrapper_source": source_map["wrapper_source"],
        }
    )
    assert set(bound_artifacts) == set(v23._V23_BOUND_ARTIFACT_KEYS)

    token = bytes(range(32))
    runtime_identity: dict[str, object] = {
        "fixture_runtime": True,
        "profile": "no-history-definition",
    }
    plan = v23._FrozenTicketSemanticPlanInputsV1(
        namespace_profile=profile,
        authorization_id="fixture-v23-one-shot-authorization",
        single_use_token_sha256=_sha256(token),
        bound_artifacts=bound_artifacts,
        implementation_identity=identity,
        definition=definition,
        dependency_manifest=dependency_manifest,
        source_fingerprints=source_fingerprints,
        stable_runtime_identity=runtime_identity,
        old_failure_quarantine=v23._v23_old_quarantine_expected(),
    )
    projection = v23._build_projection_v1(plan)
    return _ProjectionFixture(
        token=token,
        core_sha256=core_sha,
        identity=identity,
        definition=definition,
        dependency_manifest=dependency_manifest,
        source_fingerprints=source_fingerprints,
        runtime_identity=runtime_identity,
        bound_artifacts=bound_artifacts,
        plan=plan,
        projection=projection,
    )


def _static_ticket_from_projection(
    ledger: _FixtureLedger,
    projection: dict[str, object],
) -> dict[str, object]:
    zero = "0" * 64
    authorization = copy.deepcopy(projection["authorization_plan"])
    authorization["canonical_sha256"] = zero
    result = copy.deepcopy(projection["result"])
    trace_base = _repo_relative(ledger.root)
    ticket = {
        "artifact": projection["ticket_artifact"],
        "assurance_profile": projection["assurance_profile"],
        "attempt_id": projection["attempt_id"],
        "authorization": authorization,
        "bound_artifacts": copy.deepcopy(projection["bound_artifacts"]),
        "bound_manifests": copy.deepcopy(projection["bound_manifests"]),
        "case_identity_sha256": copy.deepcopy(
            projection["case_identity_sha256"]
        ),
        "definition_chain": copy.deepcopy(projection["definition_chain"]),
        "execution_sources": copy.deepcopy(projection["execution_sources"]),
        "implementation_identity": copy.deepcopy(
            projection["implementation_identity"]
        ),
        "issuance_request": {
            "canonical_sha256": zero,
            "path": _repo_relative(
                ledger.profile.issuance_request_path
            ),
            "raw_sha256": zero,
            "ticket_semantic_projection_sha256": zero,
        },
        "namespace_profile": projection["namespace_profile"],
        "old_failure_quarantine": copy.deepcopy(
            projection["old_failure_quarantine"]
        ),
        "ordered_case_names": copy.deepcopy(
            projection["ordered_case_names"]
        ),
        "ordered_registry_manifest_sha256": projection[
            "ordered_registry_manifest_sha256"
        ],
        "result": result,
        "schema_version": projection["schema_version"],
        "scientific_protocol_version": projection[
            "scientific_protocol_version"
        ],
        "scope": copy.deepcopy(projection["scope"]),
        "second_bounded_audit": {
            "clearance_canonical_sha256": zero,
            "invocation_metadata_canonical_sha256": zero,
            "invocation_metadata_path": f"{trace_base}/m.json",
            "invocation_metadata_raw_sha256": zero,
            "request_canonical_sha256": zero,
            "request_path": f"{trace_base}/a.json",
            "request_raw_sha256": zero,
            "response_canonical_sha256": zero,
            "response_path": f"{trace_base}/c.json",
            "response_raw_sha256": zero,
            "ticket_semantic_projection_sha256": zero,
            "trace_metadata_path": f"{trace_base}/z.json",
            "trace_metadata_raw_sha256": zero,
        },
        "stable_runtime_identity": copy.deepcopy(
            projection["stable_runtime_identity"]
        ),
        "status": "synthetic_definition_fixture_no_production_authority",
        "transport_protocol_version": projection[
            "transport_protocol_version"
        ],
        "version": projection["ticket_version"],
    }
    assert set(ticket) == set(v23._V23_TICKET_FIELDS)
    return ticket


def _all_mapping_keys(value: object) -> list[str]:
    keys: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            keys.append(key)
            keys.extend(_all_mapping_keys(child))
    elif isinstance(value, list):
        for child in value:
            keys.extend(_all_mapping_keys(child))
    return keys


def _write_pretty_artifact(
    ledger: _FixtureLedger,
    path: Path,
    payload: dict[str, object],
) -> tuple[str, str, str]:
    raw = _independent_pretty_json(payload)
    ledger.write_file(path, raw)
    return (
        _repo_relative(path),
        _sha256(raw),
        _sha256(_independent_canonical_json(payload)),
    )


def _independent_clearance_bindings(
    projection: dict[str, object],
) -> dict[str, str]:
    artifacts = projection["bound_artifacts"]
    manifests = projection["bound_manifests"]
    return {
        "authorization_schema_preregistration_raw_sha256": manifests[
            "authorization_schema_preregistration_raw_sha256"
        ],
        "bootstrap_contract_canonical_sha256": manifests[
            "bootstrap_contract_canonical_sha256"
        ],
        "bound_artifact_map_sha256": manifests[
            "bound_artifact_map_sha256"
        ],
        "dependency_B_paths_sha256": manifests[
            "dependency_B_paths_sha256"
        ],
        "dependency_R_paths_sha256": manifests[
            "dependency_R_paths_sha256"
        ],
        "dependency_U_paths_sha256": manifests[
            "dependency_U_paths_sha256"
        ],
        "dependency_capture_protocol_raw_sha256": manifests[
            "dependency_capture_protocol_raw_sha256"
        ],
        "dependency_manifest_canonical_sha256": manifests[
            "dependency_manifest_canonical_sha256"
        ],
        "dependency_manifest_raw_sha256": manifests[
            "dependency_manifest_raw_sha256"
        ],
        "execution_source_map_sha256": manifests[
            "execution_source_map_sha256"
        ],
        "frozen_definition_chain_sha256": manifests[
            "frozen_definition_chain_sha256"
        ],
        "g3_audit_raw_sha256": artifacts["g3_independent_audit"][
            "sha256"
        ],
        "g3_consensus_canonical_sha256": manifests[
            "g3_consensus_canonical_sha256"
        ],
        "g4_audit_raw_sha256": artifacts["g4_definition_test_audit"][
            "sha256"
        ],
        "old_failure_quarantine_canonical_sha256": manifests[
            "old_failure_quarantine_canonical_sha256"
        ],
        "ordered_registry_manifest_sha256": manifests[
            "ordered_registry_manifest_sha256"
        ],
        "reviewed_wrapper_definition_sha256": artifacts[
            "wrapper_definition"
        ]["sha256"],
        "reviewed_wrapper_sha256": artifacts["wrapper_source"]["sha256"],
        "stable_runtime_identity_sha256": manifests[
            "stable_runtime_identity_sha256"
        ],
        "transport_preregistration_raw_sha256": manifests[
            "transport_preregistration_raw_sha256"
        ],
    }


def _independent_ticket_self_digest(
    ticket: dict[str, object],
) -> str:
    normalized = copy.deepcopy(ticket)
    normalized["authorization"]["canonical_sha256"] = "0" * 64
    return _sha256(_independent_canonical_json(normalized))


def _build_full_authorization_fixture(
    ledger: _FixtureLedger,
) -> _FullAuthorizationFixture:
    """Write a complete accepted fixture chain in its normative order."""

    built = _build_projection_fixture(ledger)
    projection = built.projection
    profile = ledger.profile
    projection_sha = _sha256(
        _independent_canonical_json(projection)
    )
    trace_root = profile.allowed_trace_root
    run_root = trace_root / "2026-07-28_run01"
    ledger.mkdir(run_root)
    call_prefix = "fixturecall01"
    m_file = run_root / f"{call_prefix}.invocation.json"
    a_file = run_root / f"{call_prefix}.request.json"
    c_file = run_root / f"{call_prefix}.response.json"
    z_file = run_root / f"{call_prefix}.trace.json"
    m_path = _repo_relative(m_file)
    a_path = _repo_relative(a_file)
    c_path = _repo_relative(c_file)
    z_path = _repo_relative(z_file)
    q_path = _repo_relative(profile.issuance_request_path)

    q: dict[str, object] = {
        "artifact": "actual_wake_reachable_pressure_v23_issuance_request",
        "attempt_id": (
            "S3ai-v2.2-transport-v2.3-successor-20260728_185556"
        ),
        "implementation_identity": copy.deepcopy(built.identity),
        "planned_authorization_path": _repo_relative(profile.ticket_path),
        "planned_invocation": {
            "artifact": (
                "actual_wake_reachable_pressure_v23_review_invocation"
            ),
            "call_prefix": call_prefix,
            "path": m_path,
            "schema_version": "1.0",
            "version": "S3ai-v2.3-review-invocation-v1",
        },
        "planned_marker_path": _repo_relative(profile.marker_path),
        "planned_result_path": _repo_relative(profile.result_path),
        "review_requirements": {
            "acceptance_prefilled": False,
            "blocking_findings_must_be_empty_for_accept": True,
            "rejection_allowed": True,
            "required_independence": "genuine-cross-family",
        },
        "schema_version": "1.0",
        "single_use_token_sha256": _sha256(built.token),
        "status": "REQUEST_INDEPENDENT_REVIEW_NO_TICKET_EXISTS",
        "ticket_semantic_projection": copy.deepcopy(projection),
        "ticket_semantic_projection_sha256": projection_sha,
        "version": "S3ai-v2.3-issuance-request-v2",
    }
    q_observed_path, q_raw_sha, q_canonical_sha = (
        _write_pretty_artifact(
            ledger, profile.issuance_request_path, q
        )
    )
    assert q_observed_path == q_path

    reviewer = {
        "agent_id": "fixture-review-agent",
        "model": "fixture-review-model",
        "model_family": "fixture-review-family",
        "provider": "fixture-review-provider",
        "trace_id": call_prefix,
    }
    m: dict[str, object] = {
        "allowed_trace_root": _repo_relative(trace_root),
        "artifact": (
            "actual_wake_reachable_pressure_v23_review_invocation"
        ),
        "call_prefix": call_prefix,
        "created_at_utc": "2026-07-28T00:00:00.000000+00:00",
        "implementation_identity": copy.deepcopy(built.identity),
        "issuance_request_canonical_sha256": q_canonical_sha,
        "issuance_request_path": q_path,
        "issuance_request_raw_sha256": q_raw_sha,
        "planned_review_request": {
            "artifact": (
                "actual_wake_reachable_pressure_v23_review_request"
            ),
            "call_prefix": call_prefix,
            "path": a_path,
            "schema_version": "1.0",
            "version": "S3ai-v2.3-review-request-v1",
        },
        "requested_reviewer_model": reviewer["model"],
        "requested_reviewer_model_family": reviewer["model_family"],
        "requested_reviewer_provider": reviewer["provider"],
        "schema_version": "1.0",
        "ticket_semantic_projection_sha256": projection_sha,
        "trace_id": reviewer["trace_id"],
        "version": "S3ai-v2.3-review-invocation-v1",
    }
    _, m_raw_sha, m_canonical_sha = _write_pretty_artifact(
        ledger, m_file, m
    )

    a: dict[str, object] = {
        "artifact": "actual_wake_reachable_pressure_v23_review_request",
        "attempt_id": (
            "S3ai-v2.2-transport-v2.3-successor-20260728_185556"
        ),
        "call_prefix": call_prefix,
        "implementation_identity": copy.deepcopy(built.identity),
        "invocation_metadata": {
            "canonical_sha256": m_canonical_sha,
            "path": m_path,
            "raw_sha256": m_raw_sha,
        },
        "issuance_request": {
            "canonical_sha256": q_canonical_sha,
            "path": q_path,
            "raw_sha256": q_raw_sha,
            "ticket_semantic_projection_sha256": projection_sha,
        },
        "required_checks": [
            "G0_QUARANTINE_INTACT",
            "G2_SCIENCE_ZERO_DRIFT",
            "G3_DEPENDENCY_CAPTURE_ACCEPTED",
            "G4_DEFINITION_TESTS_ACCEPTED",
            "TICKET_SEMANTIC_PROJECTION_EXACT",
            "TOKEN_COMMITMENT_BOUND",
            "OLD_FAILURE_QUARANTINE_BOUND",
            "RESULT_NAMESPACE_AND_SCOPE_LOCKED",
            "HARD_NONCLAIMS_PRESERVED",
            "GENUINE_CROSS_FAMILY_TRACE_VERIFIED",
        ],
        "response_contract": {
            "accepted_artifact": (
                "S3ai-v2.3-successor-one-shot-execution-clearance"
            ),
            "accepted_version": 2,
            "post_review_trace_path": z_path,
            "rejected_artifact": (
                "S3ai-v2.3-successor-one-shot-execution-rejection"
            ),
            "rejected_version": 1,
            "response_path": c_path,
        },
        "review_target": {
            "model": reviewer["model"],
            "model_family": reviewer["model_family"],
            "provider": reviewer["provider"],
            "required_independence": "genuine-cross-family",
        },
        "schema_version": "1.0",
        "status": "REQUEST_NEUTRAL_INDEPENDENT_ACCEPT_OR_REJECT",
        "trace_id": reviewer["trace_id"],
        "version": "S3ai-v2.3-review-request-v1",
    }
    _, a_raw_sha, a_canonical_sha = _write_pretty_artifact(
        ledger, a_file, a
    )

    plan = projection["authorization_plan"]
    c: dict[str, object] = {
        "artifact": (
            "S3ai-v2.3-successor-one-shot-execution-clearance"
        ),
        "attempt_id": (
            "S3ai-v2.2-transport-v2.3-successor-20260728_185556"
        ),
        "bindings": _independent_clearance_bindings(projection),
        "blocking_findings": [],
        "grant": {
            "authorization_id": plan["id"],
            "authorization_ticket_creation_allowed": True,
            "decision": plan["decision"],
            "execution_limit": 1,
            "formal_execution_allowed": True,
            "post_marker_retry_allowed": False,
            "single_use_token_sha256": plan[
                "single_use_token_sha256"
            ],
        },
        "review": {
            "actual_independence": "genuine-cross-family",
            "agent_id": reviewer["agent_id"],
            "implementation_model_family": built.identity[
                "model_family"
            ],
            "invocation_metadata_canonical_sha256": m_canonical_sha,
            "invocation_metadata_path": m_path,
            "invocation_metadata_raw_sha256": m_raw_sha,
            "issuance_request_canonical_sha256": q_canonical_sha,
            "issuance_request_raw_sha256": q_raw_sha,
            "model": reviewer["model"],
            "model_family": reviewer["model_family"],
            "provider": reviewer["provider"],
            "request_canonical_sha256": a_canonical_sha,
            "request_raw_sha256": a_raw_sha,
            "ticket_semantic_projection_sha256": projection_sha,
            "trace_id": reviewer["trace_id"],
        },
        "schema_version": "1.0",
        "scope": {
            **copy.deepcopy(projection["scope"]),
            "marker_path": _repo_relative(profile.marker_path),
            "result_path": _repo_relative(profile.result_path),
        },
        "verdict": (
            "ACCEPT_EXACT_V23_SUCCESSOR_ONE_SHOT_31_HISTORY_EXECUTION"
        ),
        "version": 2,
    }
    _, c_raw_sha, c_canonical_sha = _write_pretty_artifact(
        ledger, c_file, c
    )

    z: dict[str, object] = {
        "agent_id": reviewer["agent_id"],
        "artifact": "actual_wake_reachable_pressure_v23_post_review_trace",
        "invocation_metadata_path": m_path,
        "invocation_metadata_raw_sha256": m_raw_sha,
        "model": reviewer["model"],
        "model_family": reviewer["model_family"],
        "provider": reviewer["provider"],
        "request_path": a_path,
        "request_raw_sha256": a_raw_sha,
        "response_path": c_path,
        "response_raw_sha256": c_raw_sha,
        "schema_version": "1.0",
        "trace_id": reviewer["trace_id"],
        "version": "S3ai-v2.3-post-review-trace-v1",
    }
    _, z_raw_sha, _ = _write_pretty_artifact(ledger, z_file, z)

    ticket = _static_ticket_from_projection(ledger, projection)
    ticket["issuance_request"] = {
        "canonical_sha256": q_canonical_sha,
        "path": q_path,
        "raw_sha256": q_raw_sha,
        "ticket_semantic_projection_sha256": projection_sha,
    }
    ticket["second_bounded_audit"] = {
        "clearance_canonical_sha256": c_canonical_sha,
        "invocation_metadata_canonical_sha256": m_canonical_sha,
        "invocation_metadata_path": m_path,
        "invocation_metadata_raw_sha256": m_raw_sha,
        "request_canonical_sha256": a_canonical_sha,
        "request_path": a_path,
        "request_raw_sha256": a_raw_sha,
        "response_canonical_sha256": c_canonical_sha,
        "response_path": c_path,
        "response_raw_sha256": c_raw_sha,
        "ticket_semantic_projection_sha256": projection_sha,
        "trace_metadata_path": z_path,
        "trace_metadata_raw_sha256": z_raw_sha,
    }
    ticket["authorization"]["canonical_sha256"] = (
        _independent_ticket_self_digest(ticket)
    )
    _, ticket_raw_sha, _ = _write_pretty_artifact(
        ledger, profile.ticket_path, ticket
    )
    return _FullAuthorizationFixture(
        projection_fixture=built,
        q=q,
        m=m,
        a=a,
        c=c,
        z=z,
        ticket=ticket,
        ticket_raw_sha256=ticket_raw_sha,
        q_path=q_path,
        m_path=m_path,
        a_path=a_path,
        c_path=c_path,
        z_path=z_path,
    )


def _assert_cloexec_directory_lease(
    lease: v23._DirectoryLeaseV1,
    *,
    owner_created: bool,
) -> None:
    metadata = os.fstat(lease.fd)
    assert stat.S_ISDIR(metadata.st_mode)
    assert (int(metadata.st_dev), int(metadata.st_ino)) == (
        lease.st_dev,
        lease.st_ino,
    )
    assert lease.owner_created is owner_created
    assert fcntl.fcntl(lease.fd, fcntl.F_GETFD) & fcntl.FD_CLOEXEC


def test_fixture_profile_factory_opaque_leases_and_exact_namespace(
    fixture_namespace: _FixtureLedger,
):
    profile = fixture_namespace.profile
    root = fixture_namespace.root

    with pytest.raises(TypeError, match="factory-only"):
        v23._DirectoryLeaseV1()
    with pytest.raises(TypeError, match="factory-only"):
        v23._ProductionNamespaceProfileV1()
    with pytest.raises(TypeError, match="factory-only"):
        v23._FixtureNamespaceProfileV1()

    for invalid in (
        "fixture-",
        "fixture-" + "a" * 23,
        "fixture-" + "a" * 25,
        "fixture-" + "A" * 24,
        "fixture-../../production",
        "other-" + "a" * 24,
    ):
        with pytest.raises(v23.OneShotAuthorizationError):
            v23._create_fixture_namespace_profile_v1(invalid)

    with pytest.raises(
        v23.OneShotAuthorizationError, match="adoption is forbidden"
    ):
        v23._create_fixture_namespace_profile_v1(profile.fixture_id)

    v23._validate_namespace_profile_v1(profile)
    assert profile.mode == "ISOLATED_DEFINITION_FIXTURE_V1"
    assert profile.production_credential_allowed is False
    assert root == PLATFORM / "tests" / profile.fixture_id
    assert profile.implementation_identity_core_path == (
        root / "implementation-identity-core.json"
    )
    assert profile.issuance_request_path == (
        root / "authorization-ticket.json.issuance.json"
    )
    assert profile.ticket_path == root / "authorization-ticket.json"
    assert profile.result_path == root / "result.json"
    assert profile.marker_path == root / "result.json.lock"
    assert profile.allowed_trace_root == (
        root / ".aris" / "traces" / "research-review"
    )

    _assert_cloexec_directory_lease(
        profile.repo_root_lease, owner_created=False
    )
    _assert_cloexec_directory_lease(
        profile.fixture_base_lease, owner_created=False
    )
    _assert_cloexec_directory_lease(
        profile.isolation_root_lease, owner_created=True
    )

    fixture_files = (
        profile.implementation_identity_core_path,
        profile.issuance_request_path,
        profile.ticket_path,
        profile.result_path,
        profile.marker_path,
    )
    for index, left in enumerate(fixture_files):
        assert v23._component_prefix(root, left)
        assert left != root
        for right in fixture_files[index + 1 :]:
            assert not v23._component_prefix(left, right)
            assert not v23._component_prefix(right, left)
    for fixture_path in (root, *fixture_files, profile.allowed_trace_root):
        for production_path in v23._fixed_production_namespace_paths():
            assert not v23._component_prefix(
                fixture_path, production_path
            )
            assert not v23._component_prefix(
                production_path, fixture_path
            )

    production = v23._production_namespace_profile_v1()
    try:
        v23._validate_namespace_profile_v1(production)
        assert production.mode == "PRODUCTION_RESERVED_V23"
        assert production.production_credential_allowed is True
        assert production.ticket_path == v23.AUTHORIZATION_PATH
    finally:
        v23._close_namespace_profile_v1(production)


def test_closed_fixture_lease_fails_validation_without_namespace_adoption(
    fixture_namespace: _FixtureLedger,
):
    profile = fixture_namespace.profile
    os.close(profile.isolation_root_lease.fd)
    with pytest.raises(
        v23.OneShotAuthorizationError, match="fd is not live"
    ):
        v23._validate_namespace_profile_v1(profile)


def test_identity_core_seed_is_external_and_pre_candidate(
    fixture_namespace: _FixtureLedger,
    monkeypatch,
):
    profile = fixture_namespace.profile
    core = _valid_identity_core()
    raw = _independent_pretty_json(core)
    expected = hashlib.sha256(raw).hexdigest()
    fixture_namespace.write_file(
        profile.implementation_identity_core_path, raw
    )

    original_read = v23._read_repo_relative_nofollow_v1
    reads: list[tuple[str, str]] = []

    def observed_read(profile_arg, relative, *, label):
        reads.append((relative, label))
        return original_read(profile_arg, relative, label=label)

    monkeypatch.setattr(
        v23, "_read_repo_relative_nofollow_v1", observed_read
    )
    seed = v23._build_identity_core_binding_seed_v1(profile, expected)
    expected_path = (
        profile.implementation_identity_core_path.relative_to(
            v23.REPO_ROOT
        ).as_posix()
    )
    assert seed.path == expected_path
    assert seed.expected_external_raw_sha256 == expected
    assert seed.observed_raw_sha256 == expected
    assert dict(seed.parsed_exact_core) == core
    assert reads == [(expected_path, "implementation identity core")]

    reads.clear()
    with pytest.raises(
        v23.OneShotAuthorizationError, match="external SHA"
    ):
        v23._build_identity_core_binding_seed_v1(profile, "0" * 64)
    assert reads == [(expected_path, "implementation identity core")]

    reads.clear()
    with pytest.raises(
        v23.OneShotAuthorizationError, match="lowercase SHA256"
    ):
        v23._build_identity_core_binding_seed_v1(profile, "A" * 64)
    assert reads == []


@pytest.mark.parametrize(
    "mutator",
    [
        lambda core: {**core, "future_anchor_sha256": "0" * 64},
        lambda core: {key: value for key, value in core.items() if key != "model"},
        lambda core: {**core, "provider": ""},
        lambda core: {**core, "artifact": "wrong-artifact"},
        lambda core: {**core, "source_kind": "self-reported"},
    ],
    ids=(
        "future-field",
        "missing-field",
        "empty-identity",
        "fixed-artifact",
        "fixed-source-kind",
    ),
)
def test_identity_core_seed_rejects_semantic_schema_drift(
    fixture_namespace: _FixtureLedger,
    mutator,
):
    profile = fixture_namespace.profile
    raw = _independent_pretty_json(mutator(_valid_identity_core()))
    fixture_namespace.write_file(
        profile.implementation_identity_core_path, raw
    )
    with pytest.raises(v23.OneShotAuthorizationError):
        v23._build_identity_core_binding_seed_v1(
            profile, hashlib.sha256(raw).hexdigest()
        )


def test_identity_core_seed_rejects_duplicate_noncanonical_and_symlink(
    fixture_namespace: _FixtureLedger,
):
    profile = fixture_namespace.profile
    path = profile.implementation_identity_core_path
    duplicate = (
        b'{"artifact":"a","artifact":"b","schema_version":"1.0"}\n'
    )
    fixture_namespace.write_file(path, duplicate)
    with pytest.raises(v23.OneShotAuthorizationError):
        v23._build_identity_core_binding_seed_v1(
            profile, hashlib.sha256(duplicate).hexdigest()
        )


def test_identity_core_seed_rejects_symlink_final_component(
    fixture_namespace: _FixtureLedger,
):
    profile = fixture_namespace.profile
    fixture_namespace.symlink(
        profile.implementation_identity_core_path,
        "../test_actual_wake_reachable_pressure_authorization_v23_definition.py",
    )
    with pytest.raises(v23.OneShotAuthorizationError):
        v23._build_identity_core_binding_seed_v1(profile, "0" * 64)


def test_build_projection_extract_roundtrip_and_exclusion_invariance(
    fixture_namespace: _FixtureLedger,
):
    built = _build_projection_fixture(fixture_namespace)
    projection = built.projection
    ticket = _static_ticket_from_projection(
        fixture_namespace, projection
    )

    extracted = v23._ticket_semantic_projection_v1(
        copy.deepcopy(ticket)
    )
    assert extracted == projection
    assert _independent_canonical_json(extracted) == (
        _independent_canonical_json(projection)
    )

    keys = _all_mapping_keys(projection)
    assert "issuance_request" not in keys
    assert "second_bounded_audit" not in keys
    assert "ticket_semantic_projection_sha256" not in keys
    assert "canonical_sha256" not in projection["authorization_plan"]

    excluded_only = copy.deepcopy(ticket)
    excluded_only["authorization"]["canonical_sha256"] = "1" * 64
    for name in (
        "raw_sha256",
        "canonical_sha256",
        "ticket_semantic_projection_sha256",
    ):
        excluded_only["issuance_request"][name] = "2" * 64
    for name in v23._V23_SECOND_AUDIT_FIELDS:
        if not name.endswith("_path"):
            excluded_only["second_bounded_audit"][name] = "3" * 64
    assert v23._ticket_semantic_projection_v1(excluded_only) == projection

    included = copy.deepcopy(ticket)
    included["stable_runtime_identity"]["fixture_runtime"] = False
    changed = v23._ticket_semantic_projection_v1(included)
    assert changed != projection


def test_q_m_a_c_z_fixture_chain_has_exact_schemas_and_bindings(
    fixture_namespace: _FixtureLedger,
):
    built = _build_full_authorization_fixture(fixture_namespace)
    projection = built.projection_fixture.projection

    assert v23._validate_v23_q(copy.deepcopy(built.q)) == built.q
    assert v23._validate_v23_m(copy.deepcopy(built.m)) == built.m
    assert v23._validate_v23_a(copy.deepcopy(built.a)) == built.a
    assert (
        v23._validate_v23_c(copy.deepcopy(built.c), accepted=True)
        == built.c
    )
    assert v23._validate_v23_z(copy.deepcopy(built.z)) == built.z
    assert v23._ticket_semantic_projection_v1(
        copy.deepcopy(built.ticket)
    ) == projection
    assert built.q["ticket_semantic_projection"] == projection
    assert built.q["ticket_semantic_projection_sha256"] == _sha256(
        _independent_canonical_json(projection)
    )
    assert built.ticket["authorization"]["canonical_sha256"] == (
        _independent_ticket_self_digest(built.ticket)
    )
    assert built.c["bindings"] == _independent_clearance_bindings(
        projection
    )
    assert built.c["review"]["model_family"] != (
        built.ticket["implementation_identity"]["model_family"]
    )
    assert built.z["trace_id"] != (
        built.ticket["implementation_identity"]["trace_id"]
    )
    assert not fixture_namespace.profile.result_path.exists()
    assert not fixture_namespace.profile.marker_path.exists()


def test_private_seam_positive_roundtrip_has_three_equal_snapshots(
    fixture_namespace: _FixtureLedger,
    monkeypatch,
):
    built = _build_full_authorization_fixture(fixture_namespace)
    projection_fixture = built.projection_fixture
    original_snapshot = v23._loaded_dependency_state
    snapshots: list[dict[str, object]] = []

    def observed_snapshot():
        state = original_snapshot()
        snapshots.append(copy.deepcopy(state))
        return state

    monkeypatch.setattr(v23, "_loaded_dependency_state", observed_snapshot)
    verified = v23._load_and_verify_authorization_from_ticket_path(
        namespace_profile=fixture_namespace.profile,
        ticket_path=fixture_namespace.profile.ticket_path,
        expected_authorization_sha256=built.ticket_raw_sha256,
        expected_implementation_identity_core_raw_sha256=(
            projection_fixture.core_sha256
        ),
        second_audit_token=projection_fixture.token,
        definition=projection_fixture.definition,
        dependency_manifest=projection_fixture.dependency_manifest,
        source_fingerprints=projection_fixture.source_fingerprints,
        stable_runtime_identity=projection_fixture.runtime_identity,
    )

    assert len(snapshots) == 3
    assert snapshots[0] == snapshots[1] == snapshots[2]
    assert _independent_canonical_json(snapshots[0]) == (
        _independent_canonical_json(snapshots[1])
    ) == _independent_canonical_json(snapshots[2])
    assert verified.payload == built.ticket
    assert verified.raw_sha256 == built.ticket_raw_sha256
    assert verified.canonical_sha256 == built.ticket["authorization"][
        "canonical_sha256"
    ]
    assert verified.token_sha256 == _sha256(projection_fixture.token)
    assert verified.clearance == built.c
    assert dict(verified.source_fingerprints) == (
        projection_fixture.source_fingerprints
    )
    assert dict(verified.stable_runtime_identity) == (
        projection_fixture.runtime_identity
    )

    expected_audit_paths = {
        binding["path"]
        for binding in projection_fixture.bound_artifacts.values()
    }
    expected_audit_paths.update(
        {
            projection_fixture.identity["source_trace_metadata_path"],
            built.q_path,
            built.m_path,
            built.a_path,
            built.c_path,
            built.z_path,
        }
    )
    assert set(verified.audit_files) == expected_audit_paths
    assert len(verified.audit_files) == 24
    assert _repo_relative(fixture_namespace.profile.ticket_path) not in (
        verified.audit_files
    )
    assert not fixture_namespace.profile.result_path.exists()
    assert not fixture_namespace.profile.marker_path.exists()


def test_dynamic_child_read_requires_post_read_profile_validation(
    fixture_namespace: _FixtureLedger,
    monkeypatch,
):
    profile = fixture_namespace.profile
    raw = _independent_pretty_json(_valid_identity_core())
    fixture_namespace.write_file(
        profile.implementation_identity_core_path, raw
    )
    relative = _repo_relative(
        profile.implementation_identity_core_path
    )
    original_validate = v23._validate_namespace_profile_v1
    calls = {"count": 0}

    def fail_only_at_post_read(profile_arg):
        calls["count"] += 1
        original_validate(profile_arg)
        if calls["count"] == 3:
            raise v23.OneShotAuthorizationError(
                "synthetic post-read lease drift"
            )

    monkeypatch.setattr(
        v23, "_validate_namespace_profile_v1", fail_only_at_post_read
    )
    with pytest.raises(
        v23.OneShotAuthorizationError,
        match="synthetic post-read lease drift",
    ):
        v23._read_repo_relative_nofollow_v1(
            profile, relative, label="post-read validation fixture"
        )
    assert calls["count"] == 3


@pytest.mark.parametrize(
    "states",
    [
        ({}, {"added": {"sha256": "1" * 64}}, {"added": {"sha256": "1" * 64}}),
        ({"removed": {"sha256": "2" * 64}}, {}, {}),
        (
            {"changed": {"sha256": "3" * 64}},
            {"changed": {"sha256": "4" * 64}},
            {"changed": {"sha256": "4" * 64}},
        ),
    ],
    ids=("dependency-added", "dependency-removed", "fingerprint-drift"),
)
def test_private_seam_rejects_any_three_snapshot_delta(
    fixture_namespace: _FixtureLedger,
    monkeypatch,
    states,
):
    built = _build_full_authorization_fixture(fixture_namespace)
    observations = [copy.deepcopy(state) for state in states]
    calls = {"count": 0}

    def drifting_snapshot():
        state = observations[calls["count"]]
        calls["count"] += 1
        return copy.deepcopy(state)

    monkeypatch.setattr(
        v23, "_loaded_dependency_state", drifting_snapshot
    )
    with pytest.raises(
        v23.OneShotAuthorizationError,
        match="loaded or changed a file-backed dependency",
    ):
        v23._load_and_verify_authorization_from_ticket_path(
            namespace_profile=fixture_namespace.profile,
            ticket_path=fixture_namespace.profile.ticket_path,
            expected_authorization_sha256=built.ticket_raw_sha256,
            expected_implementation_identity_core_raw_sha256=(
                built.projection_fixture.core_sha256
            ),
            second_audit_token=built.projection_fixture.token,
            definition=built.projection_fixture.definition,
            dependency_manifest=(
                built.projection_fixture.dependency_manifest
            ),
            source_fingerprints=(
                built.projection_fixture.source_fingerprints
            ),
            stable_runtime_identity=(
                built.projection_fixture.runtime_identity
            ),
        )
    assert calls["count"] == 3
    assert not fixture_namespace.profile.result_path.exists()
    assert not fixture_namespace.profile.marker_path.exists()


def test_verified_receipt_rebuilds_identity_and_rejects_memory_tamper(
    fixture_namespace: _FixtureLedger,
):
    built = _build_full_authorization_fixture(fixture_namespace)
    projection_fixture = built.projection_fixture
    verified = v23._load_and_verify_authorization_from_ticket_path(
        namespace_profile=fixture_namespace.profile,
        ticket_path=fixture_namespace.profile.ticket_path,
        expected_authorization_sha256=built.ticket_raw_sha256,
        expected_implementation_identity_core_raw_sha256=(
            projection_fixture.core_sha256
        ),
        second_audit_token=projection_fixture.token,
        definition=projection_fixture.definition,
        dependency_manifest=projection_fixture.dependency_manifest,
        source_fingerprints=projection_fixture.source_fingerprints,
        stable_runtime_identity=projection_fixture.runtime_identity,
    )
    rebuilt = v23._reverify_v23_identity_projection_for_execution(
        verified=verified,
        ticket_raw=fixture_namespace.profile.ticket_path.read_bytes(),
        namespace_profile=fixture_namespace.profile,
    )
    assert _sha256(_independent_canonical_json(rebuilt)) == (
        verified.identity_projection_sha256
    )
    v23._require_verified_v23_identity(
        verified,
        transport_preregistration_sha256=(
            verified.transport_preregistration_sha256
        ),
        dependency_manifest=projection_fixture.dependency_manifest,
        source_fingerprints=projection_fixture.source_fingerprints,
        stable_runtime_identity=projection_fixture.runtime_identity,
    )
    with pytest.raises(TypeError):
        verified.audit_files["forbidden/new/path"] = "0" * 64

    verified.clearance["review"]["model"] = "post-seam-mutated-model"
    with pytest.raises(
        v23.OneShotAuthorizationError,
        match="changed after private-seam acceptance",
    ):
        v23._require_verified_v23_identity(
            verified,
            transport_preregistration_sha256=(
                verified.transport_preregistration_sha256
            ),
            dependency_manifest=projection_fixture.dependency_manifest,
            source_fingerprints=projection_fixture.source_fingerprints,
            stable_runtime_identity=projection_fixture.runtime_identity,
        )
    assert not fixture_namespace.profile.result_path.exists()
    assert not fixture_namespace.profile.marker_path.exists()


def test_strict_bearer_accepts_exact_lowercase_hex_lf_eof():
    token = bytes(range(32))
    raw = token.hex().encode("ascii") + b"\n"
    assert v23._read_strict_second_audit_token(io.BytesIO(raw)) == token


@pytest.mark.parametrize(
    "raw",
    [
        b"0" * 63 + b"\n",
        b"0" * 65 + b"\n",
        b"0" * 64,
        b"0" * 64 + b"\r\n",
        b" " + b"0" * 63 + b"\n",
        b"0" * 63 + b" " + b"\n",
        b"g" + b"0" * 63 + b"\n",
        b"A" * 64 + b"\n",
        b"0" * 64 + b"\n\n",
        b"0" * 64 + b"\nextra",
    ],
    ids=(
        "short",
        "long",
        "missing-lf",
        "crlf",
        "leading-space",
        "trailing-space",
        "invalid-hex",
        "uppercase",
        "extra-lf",
        "extra-bytes",
    ),
)
def test_strict_bearer_rejects_any_other_byte_domain(raw: bytes):
    with pytest.raises(
        v23.OneShotAuthorizationError, match="64 lowercase hex"
    ):
        v23._read_strict_second_audit_token(io.BytesIO(raw))
