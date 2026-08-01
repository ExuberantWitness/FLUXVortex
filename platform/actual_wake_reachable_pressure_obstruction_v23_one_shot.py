"""No-history transport-v2.3 definition for the frozen S3ai-v2.2 run.

This module adds no aerodynamic equation and does not make the frozen public
runner executable.  It defines the finite per-file dependency closure that a
future, independently authorized successor call must satisfy around the
already frozen loader, 31-history collector, and aggregator.  Importing the
module never loads a contract, constructs a mesh, collects a history, or
writes an artifact.  In particular, this definition does not create the
locked v2.3 dependency manifest, authorization, token, marker, or result.

The permanent attempt marker is claimed before the first history.  Therefore
an integrity failure or crash after that point consumes the authorization and
cannot be silently retried.  A scientifically adverse ``PROTOCOL-NO-GO``
result, on the other hand, is a valid observation and is published.
"""
from __future__ import annotations

from contextlib import redirect_stdout
import argparse
import builtins
import ctypes
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import platform as host_platform
import re
import secrets
import stat
import sys
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np
import yaml


PLATFORM = Path(__file__).absolute().parent
REPO_ROOT = PLATFORM.parent
WRAPPER_PATH = Path(__file__).absolute()
WRAPPER_DEFINITION_PATH = (
    PLATFORM
    / "docs"
    / "diag"
    / "actual_wake_reachable_pressure_execution_wrapper_cases_20260728_143034.yaml"
)
TRANSPORT_PREREGISTRATION_PATH = (
    PLATFORM
    / "docs"
    / "diag"
    / "actual_wake_reachable_pressure_transport_v23_preregistration_20260728_185556.json"
)
DEPENDENCY_MANIFEST_PATH = (
    PLATFORM
    / "docs"
    / "diag"
    / "actual_wake_reachable_pressure_dependency_closure_v23_20260728_185556.json"
)
AUTHORIZATION_PATH = (
    PLATFORM
    / "docs"
    / "diag"
    / "actual_wake_reachable_pressure_execution_authorization_v23_20260728_185556.yaml"
)
IMPLEMENTATION_IDENTITY_CORE_PATH = (
    PLATFORM
    / "docs"
    / "diag"
    / "actual_wake_reachable_pressure_implementation_identity_core_v23_20260728_185556.json"
)
RESULT_PATH = (
    PLATFORM
    / "docs"
    / "diag"
    / "actual_wake_reachable_pressure_obstruction_results_s3ai_v22_transport_v23_20260728_185556.json"
)
ATTEMPT_MARKER_PATH = Path(f"{RESULT_PATH}.lock")
_FORMAL_RESULT_PATH = RESULT_PATH
_FORMAL_ATTEMPT_MARKER_PATH = ATTEMPT_MARKER_PATH
LATEST_RESULT_PATH = (
    PLATFORM
    / "docs"
    / "diag"
    / "actual_wake_reachable_pressure_obstruction_results.json"
)
OLD_CANONICAL_RESULT_PATH = (
    PLATFORM
    / "docs"
    / "diag"
    / "actual_wake_reachable_pressure_obstruction_results_20260728_134229.json"
)
OLD_AUTHORIZATION_PATH = (
    PLATFORM
    / "docs"
    / "diag"
    / "actual_wake_reachable_pressure_execution_authorization_20260728_143034.yaml"
)
OLD_ATTEMPT_MARKER_PATH = Path(f"{OLD_CANONICAL_RESULT_PATH}.lock")
OLD_FAILURE_LOG_PATH = (
    PLATFORM
    / "docs"
    / "diag"
    / "actual_wake_reachable_pressure_formal_run_20260728_1616.log"
)
OLD_FORENSICS_MD_PATH = (
    PLATFORM
    / "docs"
    / "diag"
    / "actual_wake_reachable_pressure_formal_failure_forensics_20260728_183314.md"
)
OLD_FORENSICS_JSON_PATH = OLD_FORENSICS_MD_PATH.with_suffix(".json")
OLD_CORRECTION_MD_PATH = (
    PLATFORM
    / "docs"
    / "diag"
    / "actual_wake_reachable_pressure_formal_failure_forensics_correction_20260728_183755.md"
)
OLD_CORRECTION_JSON_PATH = OLD_CORRECTION_MD_PATH.with_suffix(".json")
NO_BYTECODE_CACHE_PATH = REPO_ROOT / ".one_shot_no_bytecode_cache"
_BOOTSTRAP_WRAPPER_SHA256 = globals().get(
    "_BOOTSTRAP_WRAPPER_SHA256"
)

_EXPECTED_ACCOUNTING = {
    "histories": 31,
    "measurement_steps": 380,
    "compatible_presteps": 31,
    "marcher_steps": 411,
    "half_full_solves": 822,
    "observed_stages": 791,
}
_FROZEN_VALUE_HASH_KEYS = {
    "mass_active",
    "stored_step_residuals",
    "direct_step_residuals",
    "stage_times",
    "canonical_material_current_trace",
}
_EXTENDED_VALUE_HASH_KEYS = {
    "mass_active",
    "stage_times",
    "measurement_times",
    "stored_window_residual",
    "direct_window_residual",
    "stored_step_residuals",
    "direct_step_residuals",
    "canonical_material_current_trace",
    "body_cut_trace",
    "canonical_material_release",
    "representation_inventory",
    "stored_weak_pressure",
    "direct_weak_pressure",
    "stage_roles_canonical_JSON",
}
_VALID_DECISIONS = {
    "PROTOCOL-NO-GO",
    "ZEROTH-ORDER NAMED-LAW OBSTRUCTION",
    "FIXED-SPACE REACHABLE FIRST-ORDER OBSTRUCTION WITNESS",
    "NO RESOLVED WITNESS",
}
_ZERO_SHA256 = "0" * 64
_O_DIRECTORY = os.O_DIRECTORY
_O_NOFOLLOW = os.O_NOFOLLOW
_O_CLOEXEC = os.O_CLOEXEC
_ATTEMPT_ID = "S3ai-v2.2-transport-v2.3-successor-20260728_185556"
_TRANSPORT_PROTOCOL_VERSION = "S3ai-v2.3"
_ASSURANCE_PROFILE = "RESEARCH_ACCIDENTAL_DRIFT"
_DEPENDENCY_PHASES = (
    "baseline_pre_marker",
    "declared_formal_path",
    "post_collector_pre_publication",
)
_DEPENDENCY_MEMBER_FIELDS = {
    "canonical_path",
    "module_and_distribution_identity",
    "kind",
    "origin_or_package",
    "sha256",
    "st_dev",
    "st_ino",
    "st_size",
    "allowed_phase",
    "required_or_optional",
    "justification",
}
_OLD_AUTHORIZATION_SHA256 = (
    "39ebcd7b5a51e9ccd400c211cc3025952b9f27be9c09ad296f1dfc1a0bf5a75e"
)
_OLD_AUTHORIZATION_ID = "S3ai-v2.2-one-shot-gemma4-20260728-03"
_OLD_TOKEN_SHA256 = (
    "873efd05485c87936fcea36a360c411c9ecf1bab59cf388ab9711dff4fe214e1"
)
_PRODUCTION_NAMESPACE_MODE = "PRODUCTION_RESERVED_V23"
_FIXTURE_NAMESPACE_MODE = "ISOLATED_DEFINITION_FIXTURE_V1"
_AUTHORIZATION_ARTIFACT = (
    "actual_wake_reachable_pressure_execution_authorization_v23_"
    "20260728_185556"
)
_AUTHORIZATION_VERSION = "S3ai-v2.3-one-shot-authorization-v2"
_AUTHORIZATION_DECISION = (
    "YES_NEW_ONE_SHOT_31_HISTORY_SUCCESSOR_AFTER_TRANSPORT_FAILURE"
)
_IMPLEMENTATION_CORE_ARTIFACT = (
    "actual_wake_reachable_pressure_v23_implementation_identity_core"
)
_IMPLEMENTATION_CORE_VERSION = (
    "S3ai-v2.3-implementation-identity-core-v1"
)
_IMPLEMENTATION_ANCHOR_ARTIFACT = (
    "actual_wake_reachable_pressure_v23_implementation_identity"
)
_IMPLEMENTATION_ANCHOR_VERSION = (
    "S3ai-v2.3-implementation-identity-v1"
)
_IMPLEMENTATION_TRACE_ARTIFACT = (
    "actual_wake_reachable_pressure_v23_implementation_source_trace"
)
_IMPLEMENTATION_TRACE_VERSION = (
    "S3ai-v2.3-implementation-source-trace-v1"
)
_G6_CAPABILITY_PROBE_RELATIVE = (
    "platform/docs/diag/"
    "actual_wake_reachable_pressure_g6_otmpfile_capability_probe_"
    "20260728_213611.json"
)
_G6_CAPABILITY_PROBE_SHA256 = (
    "17a363d74ebb244c851ba222b93100415ce60c084c8bee111cfda920c3ee7968"
)
_AUTHORIZATION_SCHEMA_PREREGISTRATION_RELATIVE = (
    "platform/docs/diag/"
    "actual_wake_reachable_pressure_authorization_schema_v23_"
    "preregistration_20260728_215354.md"
)
_AUTHORIZATION_SCHEMA_PREREGISTRATION_SHA256 = (
    "77881f64788bbb0edaa1a2ce43ffd911f59b22a44eaa5b6cde0f1723100a921f"
)
_V23_REQUIRED_CHECKS = (
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
)
_V23_PROJECTION_FIELDS = frozenset(
    {
        "schema_version",
        "namespace_profile",
        "ticket_artifact",
        "ticket_version",
        "attempt_id",
        "scientific_protocol_version",
        "transport_protocol_version",
        "assurance_profile",
        "authorization_plan",
        "bound_artifacts",
        "implementation_identity",
        "definition_chain",
        "execution_sources",
        "stable_runtime_identity",
        "bound_manifests",
        "result",
        "scope",
        "old_failure_quarantine",
        "ordered_case_names",
        "case_identity_sha256",
        "ordered_registry_manifest_sha256",
    }
)
_V23_AUTHORIZATION_PLAN_FIELDS = frozenset(
    {
        "id",
        "external_raw_sha256_required",
        "single_use_token_sha256",
        "formal_execution_allowed",
        "execution_limit",
        "decision",
        "post_marker_retry_allowed",
    }
)
_V23_TICKET_FIELDS = frozenset(
    {
        "artifact",
        "schema_version",
        "version",
        "status",
        "namespace_profile",
        "attempt_id",
        "scientific_protocol_version",
        "transport_protocol_version",
        "assurance_profile",
        "authorization",
        "issuance_request",
        "bound_artifacts",
        "implementation_identity",
        "definition_chain",
        "execution_sources",
        "stable_runtime_identity",
        "bound_manifests",
        "second_bounded_audit",
        "result",
        "scope",
        "old_failure_quarantine",
        "ordered_case_names",
        "case_identity_sha256",
        "ordered_registry_manifest_sha256",
    }
)
_V23_AUTHORIZATION_FIELDS = frozenset(
    {*_V23_AUTHORIZATION_PLAN_FIELDS, "canonical_sha256"}
)
_V23_BOUND_ARTIFACT_KEYS = frozenset(
    {
        "wrapper_source",
        "wrapper_definition",
        "transport_preregistration",
        "dependency_capture_protocol",
        "authorization_schema_preregistration",
        "bootstrap_contract",
        "implementation_identity_core",
        "implementation_identity",
        "g0_quarantine_audit",
        "g2_implementation_diff_audit",
        "g3_capture_instantiation",
        "dependency_manifest",
        "g3_consensus_certificate",
        "g3_independent_audit",
        "g4_definition_test_audit",
        "g6_otmpfile_capability_probe",
        "interpretation_contract_markdown",
        "interpretation_contract_json",
    }
)
_V23_BOUND_MANIFEST_FIELDS = frozenset(
    {
        "bound_artifact_map_sha256",
        "execution_source_map_sha256",
        "stable_runtime_identity_sha256",
        "ordered_registry_manifest_sha256",
        "frozen_definition_chain_sha256",
        "transport_preregistration_raw_sha256",
        "dependency_capture_protocol_raw_sha256",
        "authorization_schema_preregistration_raw_sha256",
        "dependency_manifest_raw_sha256",
        "dependency_manifest_canonical_sha256",
        "dependency_B_paths_sha256",
        "dependency_U_paths_sha256",
        "dependency_R_paths_sha256",
        "g3_consensus_canonical_sha256",
        "bootstrap_contract_canonical_sha256",
        "old_failure_quarantine_canonical_sha256",
    }
)
_V23_SECOND_AUDIT_FIELDS = frozenset(
    {
        "invocation_metadata_path",
        "invocation_metadata_raw_sha256",
        "invocation_metadata_canonical_sha256",
        "request_path",
        "request_raw_sha256",
        "request_canonical_sha256",
        "response_path",
        "response_raw_sha256",
        "response_canonical_sha256",
        "trace_metadata_path",
        "trace_metadata_raw_sha256",
        "clearance_canonical_sha256",
        "ticket_semantic_projection_sha256",
    }
)
_V23_RESULT_FIELDS = frozenset(
    {
        "path",
        "marker_path",
        "overwrite_allowed",
        "latest_pointer_write_allowed",
        "old_canonical_result_write_allowed",
        "atomic_no_replace_required",
    }
)
_V23_SCOPE_FIELDS = frozenset(
    {
        "one_31_history_successor_execution_only",
        "transport_failure_successor_not_retry",
        "claim_state_change_allowed",
        "production_activation_allowed",
        "force_hp_state_ves_118_fig_allowed",
        "malicious_local_writer_or_swap_restore_protection",
        "premarker_token_reuse_protection_claimed",
    }
)
_V23_IDENTITY_BODY_FIELDS = frozenset(
    {
        "provider",
        "model",
        "model_family",
        "agent_id",
        "trace_id",
        "source_trace_metadata_path",
        "source_trace_metadata_sha256",
    }
)
_V23_SOURCE_ARTIFACT_KEYS = frozenset(
    {
        "wrapper_source",
        "g2_implementation_diff_audit",
        "g4_definition_test_audit",
    }
)
_V23_Q_FIELDS = frozenset(
    {
        "artifact",
        "schema_version",
        "version",
        "status",
        "attempt_id",
        "planned_authorization_path",
        "planned_result_path",
        "planned_marker_path",
        "single_use_token_sha256",
        "ticket_semantic_projection",
        "ticket_semantic_projection_sha256",
        "implementation_identity",
        "planned_invocation",
        "review_requirements",
    }
)
_V23_M_FIELDS = frozenset(
    {
        "artifact",
        "schema_version",
        "version",
        "trace_id",
        "created_at_utc",
        "allowed_trace_root",
        "call_prefix",
        "implementation_identity",
        "requested_reviewer_provider",
        "requested_reviewer_model",
        "requested_reviewer_model_family",
        "issuance_request_path",
        "issuance_request_raw_sha256",
        "issuance_request_canonical_sha256",
        "ticket_semantic_projection_sha256",
        "planned_review_request",
    }
)
_V23_A_FIELDS = frozenset(
    {
        "artifact",
        "schema_version",
        "version",
        "status",
        "attempt_id",
        "trace_id",
        "call_prefix",
        "issuance_request",
        "invocation_metadata",
        "implementation_identity",
        "review_target",
        "required_checks",
        "response_contract",
    }
)
_V23_C_FIELDS = frozenset(
    {
        "artifact",
        "schema_version",
        "version",
        "attempt_id",
        "verdict",
        "review",
        "bindings",
        "grant",
        "scope",
        "blocking_findings",
    }
)
_V23_C_REVIEW_FIELDS = frozenset(
    {
        "actual_independence",
        "provider",
        "model",
        "model_family",
        "agent_id",
        "trace_id",
        "implementation_model_family",
        "invocation_metadata_path",
        "invocation_metadata_raw_sha256",
        "invocation_metadata_canonical_sha256",
        "request_raw_sha256",
        "request_canonical_sha256",
        "issuance_request_raw_sha256",
        "issuance_request_canonical_sha256",
        "ticket_semantic_projection_sha256",
    }
)
_V23_C_BINDING_FIELDS = frozenset(
    {
        "reviewed_wrapper_sha256",
        "reviewed_wrapper_definition_sha256",
        "transport_preregistration_raw_sha256",
        "dependency_capture_protocol_raw_sha256",
        "authorization_schema_preregistration_raw_sha256",
        "bound_artifact_map_sha256",
        "execution_source_map_sha256",
        "stable_runtime_identity_sha256",
        "bootstrap_contract_canonical_sha256",
        "dependency_manifest_raw_sha256",
        "dependency_manifest_canonical_sha256",
        "dependency_B_paths_sha256",
        "dependency_U_paths_sha256",
        "dependency_R_paths_sha256",
        "g3_consensus_canonical_sha256",
        "g3_audit_raw_sha256",
        "g4_audit_raw_sha256",
        "ordered_registry_manifest_sha256",
        "frozen_definition_chain_sha256",
        "old_failure_quarantine_canonical_sha256",
    }
)
_V23_C_GRANT_FIELDS = frozenset(
    {
        "authorization_id",
        "decision",
        "single_use_token_sha256",
        "authorization_ticket_creation_allowed",
        "formal_execution_allowed",
        "execution_limit",
        "post_marker_retry_allowed",
    }
)
_V23_Z_FIELDS = frozenset(
    {
        "artifact",
        "schema_version",
        "version",
        "trace_id",
        "provider",
        "model",
        "model_family",
        "agent_id",
        "request_path",
        "request_raw_sha256",
        "response_path",
        "response_raw_sha256",
        "invocation_metadata_path",
        "invocation_metadata_raw_sha256",
    }
)
_TRACE_ID_RE = re.compile(r"[a-z0-9][a-z0-9_-]{7,127}\Z")
_RUN_DIRECTORY_RE = re.compile(
    r"[0-9]{4}-[0-9]{2}-[0-9]{2}_run[0-9]{2}\Z"
)
_FIXTURE_ID_RE = re.compile(r"fixture-[0-9a-f]{24}\Z")
_UTC_TIMESTAMP_RE = re.compile(
    r"[0-9]{4}-[0-9]{2}-[0-9]{2}T"
    r"[0-9]{2}:[0-9]{2}:[0-9]{2}\.[0-9]{6}\+00:00\Z"
)
_OLD_QUARANTINE_HASHES = {
    OLD_AUTHORIZATION_PATH: _OLD_AUTHORIZATION_SHA256,
    OLD_ATTEMPT_MARKER_PATH: (
        "42f9cb852128b24d2e28330ebf2ad9911764c845fba6370b4220aadbd5d6d778"
    ),
    OLD_FAILURE_LOG_PATH: (
        "61ae80785419e6502c0e4544ba8d9757785c36fbb6e24bb7cac8bba988c8c60d"
    ),
    OLD_FORENSICS_MD_PATH: (
        "b0e05a7114722120da571ad5396ce14370a471512d9bf6daae4a4319f07a632e"
    ),
    OLD_FORENSICS_JSON_PATH: (
        "5ed17ab7d0bb3f793d188af41f7e7acd9f2cc7855f46bd57c14bce5df2219d9c"
    ),
    OLD_CORRECTION_MD_PATH: (
        "657ec4c12a2266e277a8f8e0cc3923f6e8d124942435903ecc7d8780255ef414"
    ),
    OLD_CORRECTION_JSON_PATH: (
        "a4990e4159e8b93c47707329830089e11078f4a2995c32c123dd904c9acc32b5"
    ),
}


class OneShotAuthorizationError(RuntimeError):
    """The one-shot provenance, authorization, or result contract failed."""


@dataclass(frozen=True)
class DependencyMember:
    """One exact file in the finite authorization-bound dependency universe."""

    canonical_path: str
    module_and_distribution_identity: Any
    kind: str
    origin_or_package: str
    sha256: str
    st_dev: int
    st_ino: int
    st_size: int
    allowed_phase: str
    required_or_optional: str
    justification: str

    def as_payload(self) -> dict[str, Any]:
        return {
            name: getattr(self, name)
            for name in (
                "canonical_path",
                "module_and_distribution_identity",
                "kind",
                "origin_or_package",
                "sha256",
                "st_dev",
                "st_ino",
                "st_size",
                "allowed_phase",
                "required_or_optional",
                "justification",
            )
        }


@dataclass(frozen=True)
class DependencyManifest:
    """Validated B/U/R definition, without issuing an authorization."""

    schema_version: str
    artifact: str
    attempt_id: str
    transport_protocol_version: str
    assurance_profile: str
    members: tuple[DependencyMember, ...]
    B: tuple[str, ...]
    U: tuple[str, ...]
    R: tuple[str, ...]
    canonical_sha256: str
    raw_sha256: str | None

    @property
    def member_by_path(self) -> dict[str, DependencyMember]:
        return {member.canonical_path: member for member in self.members}

    def as_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "artifact": self.artifact,
            "attempt_id": self.attempt_id,
            "transport_protocol_version": (
                self.transport_protocol_version
            ),
            "assurance_profile": self.assurance_profile,
            "members": [member.as_payload() for member in self.members],
            "B": list(self.B),
            "U": list(self.U),
            "R": list(self.R),
        }


@dataclass(frozen=True)
class _OutputDirectoryLease:
    """One open output directory shared by marker and result publication."""

    fd: int
    marker_name: str
    result_name: str
    st_dev: int
    st_ino: int


@dataclass(frozen=True)
class _AttemptClaim:
    """Durable marker content and inode identity observed at creation."""

    receipt_sha256: str
    st_dev: int
    st_ino: int
    st_size: int


@dataclass(frozen=True, init=False, slots=True)
class _DirectoryLeaseV1:
    """Opaque inode-pinned directory authority for the v2.3 parser."""

    absolute_path: Path
    fd: int
    st_dev: int
    st_ino: int
    owner_created: bool

    def __new__(cls, *_args: Any, **_kwargs: Any) -> Any:
        raise TypeError("_DirectoryLeaseV1 is factory-only")


@dataclass(frozen=True, init=False, slots=True)
class _ProductionNamespaceProfileV1:
    """Fixed production namespace; construction is factory-only."""

    mode: str
    repo_root_lease: _DirectoryLeaseV1
    implementation_identity_core_path: Path
    issuance_request_path: Path
    ticket_path: Path
    result_path: Path
    marker_path: Path
    allowed_trace_root: Path
    production_credential_allowed: bool

    def __new__(cls, *_args: Any, **_kwargs: Any) -> Any:
        raise TypeError("_ProductionNamespaceProfileV1 is factory-only")


@dataclass(frozen=True, init=False, slots=True)
class _FixtureNamespaceProfileV1:
    """Owned no-history definition namespace beneath platform/tests."""

    mode: str
    repo_root_lease: _DirectoryLeaseV1
    fixture_base_lease: _DirectoryLeaseV1
    isolation_root_lease: _DirectoryLeaseV1
    fixture_id: str
    implementation_identity_core_path: Path
    issuance_request_path: Path
    ticket_path: Path
    result_path: Path
    marker_path: Path
    allowed_trace_root: Path
    production_credential_allowed: bool

    def __new__(cls, *_args: Any, **_kwargs: Any) -> Any:
        raise TypeError("_FixtureNamespaceProfileV1 is factory-only")


NamespaceProfileV1 = (
    _ProductionNamespaceProfileV1 | _FixtureNamespaceProfileV1
)


@dataclass(frozen=True)
class _IdentityCoreBindingSeedV1:
    """Pre-candidate identity path/raw authority."""

    path: str
    expected_external_raw_sha256: str
    observed_raw_sha256: str
    parsed_exact_core: Mapping[str, Any]


@dataclass(frozen=True)
class _FrozenTicketSemanticPlanInputsV1:
    """Only causal pre-ticket inputs accepted by BuildProjectionV1."""

    namespace_profile: NamespaceProfileV1
    authorization_id: str
    single_use_token_sha256: str
    bound_artifacts: Mapping[str, Any]
    implementation_identity: Mapping[str, Any]
    definition: Mapping[str, Any]
    dependency_manifest: DependencyManifest
    source_fingerprints: Mapping[str, str]
    stable_runtime_identity: Mapping[str, Any]
    old_failure_quarantine: Mapping[str, Any]


class _UniqueKeySafeLoader(yaml.SafeLoader):
    """YAML loader that rejects duplicate mapping keys."""


def _construct_unique_mapping(
    loader: _UniqueKeySafeLoader,
    node: yaml.nodes.MappingNode,
    deep: bool = False,
) -> dict[Any, Any]:
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"found duplicate key {key!r}",
                key_node.start_mark,
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeySafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def _source_path_from_module(module: Any) -> Path | None:
    spec = getattr(module, "__spec__", None)
    raw = getattr(spec, "origin", None) or getattr(module, "__file__", None)
    if not raw or raw in {"built-in", "frozen"}:
        return None
    path = Path(raw).absolute()
    if path.suffix in {".pyc", ".pyo"}:
        try:
            from importlib.util import source_from_cache

            path = Path(source_from_cache(str(path))).absolute()
        except (ValueError, NotImplementedError):
            return None
    return path if path.suffix == ".py" else None


def _loaded_local_source_paths() -> tuple[str, ...]:
    paths: set[str] = set()
    for module in tuple(sys.modules.values()):
        path = _source_path_from_module(module)
        if path is None:
            continue
        try:
            relative = path.relative_to(REPO_ROOT)
        except ValueError:
            continue
        paths.add(relative.as_posix())
    return tuple(sorted(paths))


def _early_declared_local_sources() -> tuple[str, ...]:
    raw = WRAPPER_DEFINITION_PATH.read_bytes()
    try:
        payload = yaml.load(
            raw.decode("utf-8"), Loader=_UniqueKeySafeLoader
        )
        sources = payload["wrapper_contract"]["source_closure"][
            "exact_preimplementation_closure"
        ]
    except (
        UnicodeDecodeError,
        yaml.YAMLError,
        KeyError,
        TypeError,
    ) as error:
        raise OneShotAuthorizationError(
            "cannot read the preregistered local-source manifest before "
            "importing the frozen runner"
        ) from error
    if (
        not isinstance(sources, list)
        or len(sources) != 62
        or len(set(sources)) != 62
        or sources != sorted(sources)
        or not all(
            isinstance(path, str)
            and path.startswith("platform/")
            and path.endswith(".py")
            for path in sources
        )
    ):
        raise OneShotAuthorizationError(
            "pre-import local-source manifest is not the exact ordered "
            "62-file set"
        )
    return tuple(sources)


def _early_source_fingerprints(
    relative_paths: Sequence[str],
) -> dict[str, str]:
    return {
        relative: hashlib.sha256(
            (REPO_ROOT / relative).read_bytes()
        ).hexdigest()
        for relative in relative_paths
    }


# This snapshot is deliberately taken before importing the frozen runner.  A
# formal call is accepted only from a fresh process in which the wrapper is the
# sole preloaded repository-local Python source.
_PRE_GUARD_LOCAL_SOURCES = _loaded_local_source_paths()
_EARLY_DECLARED_LOCAL_SOURCES = _early_declared_local_sources()
_LOCAL_SOURCE_BYTES_BEFORE_GUARD = _early_source_fingerprints(
    _EARLY_DECLARED_LOCAL_SOURCES
)
_WRAPPER_BYTES_BEFORE_GUARD = hashlib.sha256(
    WRAPPER_PATH.read_bytes()
).hexdigest()
_DEFINITION_BYTES_BEFORE_GUARD = hashlib.sha256(
    WRAPPER_DEFINITION_PATH.read_bytes()
).hexdigest()
_SOURCE_IMPORT_MODE_AT_GUARD_IMPORT = bool(
    sys.dont_write_bytecode
    and sys.pycache_prefix is not None
    and Path(sys.pycache_prefix).absolute() == NO_BYTECODE_CACHE_PATH
    and not os.path.lexists(NO_BYTECODE_CACHE_PATH)
)

if str(PLATFORM) not in sys.path:
    sys.path.insert(0, str(PLATFORM))

import actual_wake_reachable_pressure_obstruction_v2_guard as guard  # noqa: E402

_POST_GUARD_LOCAL_SOURCES = _loaded_local_source_paths()
_WRAPPER_RELATIVE = WRAPPER_PATH.relative_to(REPO_ROOT).as_posix()
_IMPORT_WAS_CLEAN = set(_PRE_GUARD_LOCAL_SOURCES) <= {_WRAPPER_RELATIVE}
_LOCAL_SOURCE_BYTES_AFTER_GUARD = _early_source_fingerprints(
    _EARLY_DECLARED_LOCAL_SOURCES
)
_WRAPPER_BYTES_AFTER_GUARD = hashlib.sha256(
    WRAPPER_PATH.read_bytes()
).hexdigest()
_DEFINITION_BYTES_AFTER_GUARD = hashlib.sha256(
    WRAPPER_DEFINITION_PATH.read_bytes()
).hexdigest()
_IMPORT_SOURCE_SNAPSHOT_STABLE = bool(
    _LOCAL_SOURCE_BYTES_BEFORE_GUARD == _LOCAL_SOURCE_BYTES_AFTER_GUARD
    and _WRAPPER_BYTES_BEFORE_GUARD == _WRAPPER_BYTES_AFTER_GUARD
    and _DEFINITION_BYTES_BEFORE_GUARD == _DEFINITION_BYTES_AFTER_GUARD
)


@dataclass(frozen=True)
class ExecutionReceipt:
    """Small non-secret receipt returned after durable publication."""

    result_path: Path
    result_sha256: str
    stage_decision: str
    authorization_sha256: str
    attempt_token_sha256: str


@dataclass(frozen=True)
class _VerifiedAuthorization:
    payload: Mapping[str, Any]
    raw_sha256: str
    canonical_sha256: str
    token_sha256: str
    definition_sha256: str
    transport_preregistration_sha256: str
    dependency_manifest_raw_sha256: str
    dependency_manifest_canonical_sha256: str
    source_fingerprints: Mapping[str, str]
    stable_runtime_identity: Mapping[str, Any]
    audit_files: Mapping[str, str]
    clearance: Mapping[str, Any]
    identity_projection_sha256: str = _ZERO_SHA256
    _memory_integrity_sha256: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Detach inputs and seal accidental in-memory authorization drift."""

        payload = _strict_json_value(
            self.payload, label="verified authorization payload"
        )
        source_fingerprints = _strict_json_value(
            self.source_fingerprints,
            label="verified authorization source fingerprints",
        )
        stable_runtime_identity = _strict_json_value(
            self.stable_runtime_identity,
            label="verified authorization stable runtime identity",
        )
        audit_files = _strict_json_value(
            self.audit_files, label="verified authorization audit files"
        )
        clearance = _strict_json_value(
            self.clearance, label="verified authorization clearance"
        )
        if not all(
            isinstance(value, Mapping)
            for value in (
                payload,
                source_fingerprints,
                stable_runtime_identity,
                audit_files,
                clearance,
            )
        ):
            raise OneShotAuthorizationError(
                "verified authorization snapshots must be JSON mappings"
            )
        identity_projection_sha256 = _validate_sha256(
            self.identity_projection_sha256,
            label="verified identity projection SHA256",
        )
        object.__setattr__(self, "payload", payload)
        object.__setattr__(
            self,
            "source_fingerprints",
            MappingProxyType(dict(source_fingerprints)),
        )
        object.__setattr__(
            self,
            "stable_runtime_identity",
            stable_runtime_identity,
        )
        object.__setattr__(
            self, "audit_files", MappingProxyType(dict(audit_files))
        )
        object.__setattr__(self, "clearance", clearance)
        object.__setattr__(
            self,
            "identity_projection_sha256",
            identity_projection_sha256,
        )
        object.__setattr__(
            self,
            "_memory_integrity_sha256",
            _verified_authorization_memory_sha256(self),
        )


@dataclass(frozen=True)
class _VerifiedRejectionV1:
    """No-ticket receipt for one fully verified independent rejection."""

    projection: Mapping[str, Any]
    rejection: Mapping[str, Any]
    audit_files: Mapping[str, str]
    identity_projection_sha256: str

    def __post_init__(self) -> None:
        projection = _strict_json_value(
            self.projection, label="verified rejection projection"
        )
        rejection = _strict_json_value(
            self.rejection, label="verified rejection response"
        )
        audit_files = _strict_json_value(
            self.audit_files, label="verified rejection audit files"
        )
        if not all(
            isinstance(value, Mapping)
            for value in (projection, rejection, audit_files)
        ):
            raise OneShotAuthorizationError(
                "verified rejection snapshots must be JSON mappings"
            )
        object.__setattr__(self, "projection", projection)
        object.__setattr__(self, "rejection", rejection)
        object.__setattr__(
            self, "audit_files", MappingProxyType(dict(audit_files))
        )
        object.__setattr__(
            self,
            "identity_projection_sha256",
            _validate_sha256(
                self.identity_projection_sha256,
                label="verified rejection identity projection SHA256",
            ),
        )


def _validate_sha256(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise OneShotAuthorizationError(f"{label} is not a lowercase SHA256")
    return value


def _verify_second_audit_token(
    expected_sha256: str,
    token: bytes,
) -> str:
    """Verify an out-of-band bearer preimage and return its commitment."""

    expected = _validate_sha256(
        expected_sha256, label="single-use token digest"
    )
    if not isinstance(token, bytes) or len(token) != 32:
        raise OneShotAuthorizationError(
            "second-audit bearer token must contain exactly 32 bytes"
        )
    actual = _sha256_bytes(token)
    if actual != expected:
        raise OneShotAuthorizationError(
            "second-audit bearer token does not match its commitment"
        )
    return actual


def _strict_json_value(value: Any, *, label: str = "value") -> Any:
    if isinstance(value, np.ndarray):
        return _strict_json_value(value.tolist(), label=label)
    if isinstance(value, np.generic):
        return _strict_json_value(value.item(), label=label)
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise OneShotAuthorizationError(
                    f"{label} contains a non-string mapping key"
                )
            normalized[key] = _strict_json_value(
                item, label=f"{label}.{key}"
            )
        return normalized
    if isinstance(value, (list, tuple)):
        return [
            _strict_json_value(item, label=f"{label}[]") for item in value
        ]
    if isinstance(value, float):
        if not np.isfinite(value):
            raise OneShotAuthorizationError(
                f"{label} contains NaN or infinity"
            )
        return float(value)
    if isinstance(value, (bool, int, str)) or value is None:
        return value
    raise OneShotAuthorizationError(
        f"{label} contains unsupported type {type(value).__name__}"
    )


def _canonical_json_bytes(value: Any) -> bytes:
    normalized = _strict_json_value(value)
    return json.dumps(
        normalized,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _pretty_json_bytes(value: Any) -> bytes:
    normalized = _strict_json_value(value)
    return (
        json.dumps(
            normalized,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _verified_authorization_memory_sha256(
    verified: _VerifiedAuthorization,
) -> str:
    """Digest every authorization field subsequently consumed in memory."""

    return _sha256_bytes(
        _canonical_json_bytes(
            {
                "payload": verified.payload,
                "raw_sha256": verified.raw_sha256,
                "canonical_sha256": verified.canonical_sha256,
                "token_sha256": verified.token_sha256,
                "definition_sha256": verified.definition_sha256,
                "transport_preregistration_sha256": (
                    verified.transport_preregistration_sha256
                ),
                "dependency_manifest_raw_sha256": (
                    verified.dependency_manifest_raw_sha256
                ),
                "dependency_manifest_canonical_sha256": (
                    verified.dependency_manifest_canonical_sha256
                ),
                "source_fingerprints": verified.source_fingerprints,
                "stable_runtime_identity": (
                    verified.stable_runtime_identity
                ),
                "audit_files": verified.audit_files,
                "clearance": verified.clearance,
                "identity_projection_sha256": (
                    verified.identity_projection_sha256
                ),
            }
        )
    )


def _assert_verified_authorization_memory_integrity(
    verified: _VerifiedAuthorization,
) -> None:
    if type(verified) is not _VerifiedAuthorization:
        raise OneShotAuthorizationError(
            "authorization consumer requires the exact verified receipt type"
        )
    if (
        _verified_authorization_memory_sha256(verified)
        != verified._memory_integrity_sha256
    ):
        raise OneShotAuthorizationError(
            "verified authorization changed after private-seam acceptance"
        )


def _json_mapping_no_duplicates(raw: bytes, *, label: str) -> dict[str, Any]:
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise OneShotAuthorizationError(
                    f"{label} contains duplicate key {key!r}"
                )
            result[key] = value
        return result

    try:
        payload = json.loads(
            raw.decode("utf-8"), object_pairs_hook=unique
        )
    except OneShotAuthorizationError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise OneShotAuthorizationError(
            f"{label} is not valid UTF-8 JSON"
        ) from error
    if not isinstance(payload, dict):
        raise OneShotAuthorizationError(f"{label} must be a JSON object")
    return _strict_json_value(payload, label=label)


def _factory_frozen_instance(cls: type[Any], **values: Any) -> Any:
    """Construct one init-disabled frozen schema object inside its factory."""

    expected = set(getattr(cls, "__dataclass_fields__", {}))
    if set(values) != expected:
        raise OneShotAuthorizationError(
            f"{cls.__name__} factory fields drifted"
        )
    instance = object.__new__(cls)
    for name, value in values.items():
        object.__setattr__(instance, name, value)
    return instance


def _new_directory_lease_v1(
    *,
    absolute_path: Path,
    fd: int,
    owner_created: bool,
) -> _DirectoryLeaseV1:
    metadata = os.fstat(fd)
    lease = _factory_frozen_instance(
        _DirectoryLeaseV1,
        absolute_path=absolute_path,
        fd=fd,
        st_dev=int(metadata.st_dev),
        st_ino=int(metadata.st_ino),
        owner_created=owner_created,
    )
    _validate_directory_lease_v1(lease)
    return lease


def _validate_directory_lease_v1(lease: _DirectoryLeaseV1) -> None:
    if (
        type(lease) is not _DirectoryLeaseV1
        or not isinstance(lease.absolute_path, Path)
        or not lease.absolute_path.is_absolute()
        or isinstance(lease.fd, bool)
        or not isinstance(lease.fd, int)
        or lease.fd < 0
        or isinstance(lease.st_dev, bool)
        or not isinstance(lease.st_dev, int)
        or lease.st_dev < 0
        or isinstance(lease.st_ino, bool)
        or not isinstance(lease.st_ino, int)
        or lease.st_ino <= 0
        or type(lease.owner_created) is not bool
    ):
        raise OneShotAuthorizationError(
            "directory lease is not the exact opaque v1 type"
        )
    try:
        metadata = os.fstat(lease.fd)
        inheritable = os.get_inheritable(lease.fd)
    except OSError as error:
        raise OneShotAuthorizationError(
            "directory lease fd is not live"
        ) from error
    if (
        inheritable
        or not stat.S_ISDIR(metadata.st_mode)
        or (int(metadata.st_dev), int(metadata.st_ino))
        != (lease.st_dev, lease.st_ino)
    ):
        raise OneShotAuthorizationError(
            "directory lease fd identity or CLOEXEC state drifted"
        )
    try:
        lexical = os.lstat(lease.absolute_path)
        fresh_fd = os.open(
            lease.absolute_path,
            os.O_RDONLY | _O_DIRECTORY | _O_NOFOLLOW | _O_CLOEXEC,
        )
    except OSError as error:
        raise OneShotAuthorizationError(
            "directory lease absolute path cannot be reopened no-follow"
        ) from error
    try:
        fresh = os.fstat(fresh_fd)
        if (
            stat.S_ISLNK(lexical.st_mode)
            or not stat.S_ISDIR(lexical.st_mode)
            or not stat.S_ISDIR(fresh.st_mode)
            or (int(lexical.st_dev), int(lexical.st_ino))
            != (lease.st_dev, lease.st_ino)
            or (int(fresh.st_dev), int(fresh.st_ino))
            != (lease.st_dev, lease.st_ino)
            or os.get_inheritable(fresh_fd)
        ):
            raise OneShotAuthorizationError(
                "directory lease pathname identity drifted"
            )
    finally:
        os.close(fresh_fd)


def _open_directory_lease_v1(
    absolute_path: Path,
    *,
    owner_created: bool,
) -> _DirectoryLeaseV1:
    path = Path(absolute_path)
    if not path.is_absolute() or path != Path(str(path)):
        raise OneShotAuthorizationError(
            "directory lease path must be an exact absolute Path"
        )
    try:
        lexical = os.lstat(path)
        if stat.S_ISLNK(lexical.st_mode) or not stat.S_ISDIR(
            lexical.st_mode
        ):
            raise OneShotAuthorizationError(
                "directory lease path is not a no-follow directory"
            )
        fd = os.open(
            path,
            os.O_RDONLY | _O_DIRECTORY | _O_NOFOLLOW | _O_CLOEXEC,
        )
    except OSError as error:
        raise OneShotAuthorizationError(
            f"cannot open directory lease {path}"
        ) from error
    try:
        lease = _new_directory_lease_v1(
            absolute_path=path,
            fd=fd,
            owner_created=owner_created,
        )
        if (lease.st_dev, lease.st_ino) != (
            int(lexical.st_dev),
            int(lexical.st_ino),
        ):
            raise OneShotAuthorizationError(
                "directory changed while its lease was opened"
            )
        return lease
    except BaseException:
        os.close(fd)
        raise


def _walk_directory_fd_v1(
    start_fd: int,
    components: Sequence[str],
) -> int:
    current = os.dup(start_fd)
    try:
        os.set_inheritable(current, False)
        for component in components:
            if (
                not isinstance(component, str)
                or not component
                or component in {".", ".."}
                or "/" in component
                or "\\" in component
            ):
                raise OneShotAuthorizationError(
                    "directory walk contains a noncanonical component"
                )
            child = os.open(
                component,
                os.O_RDONLY | _O_DIRECTORY | _O_NOFOLLOW | _O_CLOEXEC,
                dir_fd=current,
            )
            os.close(current)
            current = child
        return current
    except BaseException:
        os.close(current)
        raise


def _path_components(path: Path) -> tuple[str, ...]:
    return tuple(path.parts)


def _component_prefix(left: Path, right: Path) -> bool:
    lhs = _path_components(left)
    rhs = _path_components(right)
    return len(lhs) <= len(rhs) and rhs[: len(lhs)] == lhs


def _fixed_production_namespace_paths() -> tuple[Path, ...]:
    return (
        IMPLEMENTATION_IDENTITY_CORE_PATH,
        Path(f"{AUTHORIZATION_PATH}.issuance.json"),
        AUTHORIZATION_PATH,
        RESULT_PATH,
        ATTEMPT_MARKER_PATH,
        REPO_ROOT / ".aris" / "traces" / "research-review",
    )


def _production_namespace_profile_v1() -> _ProductionNamespaceProfileV1:
    root_lease = _open_directory_lease_v1(
        REPO_ROOT, owner_created=False
    )
    try:
        profile = _factory_frozen_instance(
            _ProductionNamespaceProfileV1,
            mode=_PRODUCTION_NAMESPACE_MODE,
            repo_root_lease=root_lease,
            implementation_identity_core_path=(
                IMPLEMENTATION_IDENTITY_CORE_PATH
            ),
            issuance_request_path=Path(
                f"{AUTHORIZATION_PATH}.issuance.json"
            ),
            ticket_path=AUTHORIZATION_PATH,
            result_path=RESULT_PATH,
            marker_path=ATTEMPT_MARKER_PATH,
            allowed_trace_root=(
                REPO_ROOT / ".aris" / "traces" / "research-review"
            ),
            production_credential_allowed=True,
        )
        _validate_namespace_profile_v1(profile)
        return profile
    except BaseException:
        os.close(root_lease.fd)
        raise


def _create_fixture_namespace_profile_v1(
    fixture_id: str,
) -> _FixtureNamespaceProfileV1:
    if not isinstance(fixture_id, str) or not _FIXTURE_ID_RE.fullmatch(
        fixture_id
    ):
        raise OneShotAuthorizationError(
            "fixture id must match fixture-[0-9a-f]{24}"
        )
    root_lease = _open_directory_lease_v1(
        REPO_ROOT, owner_created=False
    )
    base_fd: int | None = None
    isolation_fd: int | None = None
    try:
        base_fd = _walk_directory_fd_v1(
            root_lease.fd, ("platform", "tests")
        )
        base_lease = _new_directory_lease_v1(
            absolute_path=REPO_ROOT / "platform" / "tests",
            fd=base_fd,
            owner_created=False,
        )
        try:
            os.mkdir(fixture_id, 0o700, dir_fd=base_lease.fd)
        except FileExistsError as error:
            raise OneShotAuthorizationError(
                "fixture namespace already exists; adoption is forbidden"
            ) from error
        isolation_fd = os.open(
            fixture_id,
            os.O_RDONLY | _O_DIRECTORY | _O_NOFOLLOW | _O_CLOEXEC,
            dir_fd=base_lease.fd,
        )
        isolation_path = base_lease.absolute_path / fixture_id
        isolation_lease = _new_directory_lease_v1(
            absolute_path=isolation_path,
            fd=isolation_fd,
            owner_created=True,
        )
        named = os.stat(
            fixture_id,
            dir_fd=base_lease.fd,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISDIR(named.st_mode)
            or int(named.st_nlink) < 2
            or (int(named.st_dev), int(named.st_ino))
            != (isolation_lease.st_dev, isolation_lease.st_ino)
        ):
            raise OneShotAuthorizationError(
                "fixture isolation root identity drifted at creation"
            )
        profile = _factory_frozen_instance(
            _FixtureNamespaceProfileV1,
            mode=_FIXTURE_NAMESPACE_MODE,
            repo_root_lease=root_lease,
            fixture_base_lease=base_lease,
            isolation_root_lease=isolation_lease,
            fixture_id=fixture_id,
            implementation_identity_core_path=(
                isolation_path / "implementation-identity-core.json"
            ),
            issuance_request_path=(
                isolation_path / "authorization-ticket.json.issuance.json"
            ),
            ticket_path=isolation_path / "authorization-ticket.json",
            result_path=isolation_path / "result.json",
            marker_path=isolation_path / "result.json.lock",
            allowed_trace_root=(
                isolation_path / ".aris" / "traces" / "research-review"
            ),
            production_credential_allowed=False,
        )
        _validate_namespace_profile_v1(profile)
        return profile
    except BaseException:
        if isolation_fd is not None:
            os.close(isolation_fd)
        if base_fd is not None:
            os.close(base_fd)
        os.close(root_lease.fd)
        raise


def _close_namespace_profile_v1(profile: NamespaceProfileV1) -> None:
    fds: list[int] = [profile.repo_root_lease.fd]
    if type(profile) is _FixtureNamespaceProfileV1:
        fds.extend(
            [
                profile.fixture_base_lease.fd,
                profile.isolation_root_lease.fd,
            ]
        )
    for fd in dict.fromkeys(fds):
        try:
            os.close(fd)
        except OSError:
            pass


def _validate_namespace_profile_v1(
    profile: NamespaceProfileV1,
) -> None:
    if type(profile) not in {
        _ProductionNamespaceProfileV1,
        _FixtureNamespaceProfileV1,
    }:
        raise OneShotAuthorizationError(
            "namespace profile is not an exact typed v1 variant"
        )
    _validate_directory_lease_v1(profile.repo_root_lease)
    if profile.repo_root_lease.absolute_path != REPO_ROOT:
        raise OneShotAuthorizationError(
            "namespace profile repo root is not the fixed repository"
        )
    if type(profile) is _ProductionNamespaceProfileV1:
        expected = _fixed_production_namespace_paths()
        observed = (
            profile.implementation_identity_core_path,
            profile.issuance_request_path,
            profile.ticket_path,
            profile.result_path,
            profile.marker_path,
            profile.allowed_trace_root,
        )
        if (
            profile.mode != _PRODUCTION_NAMESPACE_MODE
            or type(profile.production_credential_allowed) is not bool
            or not profile.production_credential_allowed
            or observed != expected
        ):
            raise OneShotAuthorizationError(
                "production namespace profile fields drifted"
            )
        return

    if (
        profile.mode != _FIXTURE_NAMESPACE_MODE
        or profile.production_credential_allowed is not False
        or not _FIXTURE_ID_RE.fullmatch(profile.fixture_id)
    ):
        raise OneShotAuthorizationError(
            "fixture namespace discriminator or authority drifted"
        )
    _validate_directory_lease_v1(profile.fixture_base_lease)
    _validate_directory_lease_v1(profile.isolation_root_lease)
    if (
        profile.fixture_base_lease.absolute_path
        != REPO_ROOT / "platform" / "tests"
        or profile.fixture_base_lease.owner_created
        or not profile.isolation_root_lease.owner_created
        or profile.isolation_root_lease.absolute_path
        != profile.fixture_base_lease.absolute_path / profile.fixture_id
    ):
        raise OneShotAuthorizationError(
            "fixture directory lease authority drifted"
        )
    base_fd = _walk_directory_fd_v1(
        profile.repo_root_lease.fd, ("platform", "tests")
    )
    try:
        base_now = os.fstat(base_fd)
        named = os.stat(
            profile.fixture_id,
            dir_fd=profile.fixture_base_lease.fd,
            follow_symlinks=False,
        )
    finally:
        os.close(base_fd)
    if (
        (int(base_now.st_dev), int(base_now.st_ino))
        != (
            profile.fixture_base_lease.st_dev,
            profile.fixture_base_lease.st_ino,
        )
        or not stat.S_ISDIR(named.st_mode)
        or (int(named.st_dev), int(named.st_ino))
        != (
            profile.isolation_root_lease.st_dev,
            profile.isolation_root_lease.st_ino,
        )
    ):
        raise OneShotAuthorizationError(
            "fixture lease re-walk identity drifted"
        )
    root = profile.isolation_root_lease.absolute_path
    expected_fixture = (
        root / "implementation-identity-core.json",
        root / "authorization-ticket.json.issuance.json",
        root / "authorization-ticket.json",
        root / "result.json",
        root / "result.json.lock",
        root / ".aris" / "traces" / "research-review",
    )
    observed_fixture = (
        profile.implementation_identity_core_path,
        profile.issuance_request_path,
        profile.ticket_path,
        profile.result_path,
        profile.marker_path,
        profile.allowed_trace_root,
    )
    if observed_fixture != expected_fixture:
        raise OneShotAuthorizationError(
            "fixture namespace paths drifted"
        )
    files = observed_fixture[:5]
    for index, left in enumerate(files):
        if not _component_prefix(root, left) or left == root:
            raise OneShotAuthorizationError(
                "fixture file is not a strict isolation-root descendant"
            )
        for right in files[index + 1 :]:
            if _component_prefix(left, right) or _component_prefix(
                right, left
            ):
                raise OneShotAuthorizationError(
                    "fixture file paths are component-prefix comparable"
                )
    trace_root = profile.allowed_trace_root
    if (
        not _component_prefix(root, trace_root)
        or trace_root == root
        or any(
            _component_prefix(path, trace_root)
            or _component_prefix(trace_root, path)
            for path in files
        )
    ):
        raise OneShotAuthorizationError(
            "fixture trace root overlaps a fixture file path"
        )
    production = _fixed_production_namespace_paths()
    fixture_set = (root, *observed_fixture)
    for fixture_path in fixture_set:
        for production_path in production:
            if _component_prefix(
                fixture_path, production_path
            ) or _component_prefix(production_path, fixture_path):
                raise OneShotAuthorizationError(
                    "fixture and production namespaces overlap"
                )


def _safe_repo_relative_text(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or "\x00" in value
        or "\\" in value
        or any(character in value for character in ("*", "?", "[", "]"))
    ):
        raise OneShotAuthorizationError(
            f"{label} is not a safe repo-relative path"
        )
    pure = PurePosixPath(value)
    if (
        pure.is_absolute()
        or value != pure.as_posix()
        or not pure.parts
        or any(part in {"", ".", ".."} for part in pure.parts)
    ):
        raise OneShotAuthorizationError(
            f"{label} is not lexically canonical"
        )
    return value


def _profile_repo_relative(
    profile: NamespaceProfileV1,
    absolute_path: Path,
) -> str:
    _validate_namespace_profile_v1(profile)
    path = Path(absolute_path)
    if not path.is_absolute():
        raise OneShotAuthorizationError(
            "profile path must be absolute before relativization"
        )
    try:
        relative = path.relative_to(
            profile.repo_root_lease.absolute_path
        ).as_posix()
    except ValueError as error:
        raise OneShotAuthorizationError(
            "profile path lies outside its pinned repository root"
        ) from error
    return _safe_repo_relative_text(relative, label="profile path")


def _open_repo_relative_nofollow_v1(
    profile: NamespaceProfileV1,
    relative: str,
) -> int:
    _validate_namespace_profile_v1(profile)
    safe = _safe_repo_relative_text(relative, label="artifact path")
    parts = PurePosixPath(safe).parts
    start_fd = profile.repo_root_lease.fd
    walk_parts = parts
    if type(profile) is _FixtureNamespaceProfileV1:
        absolute = profile.repo_root_lease.absolute_path.joinpath(*parts)
        isolation = profile.isolation_root_lease.absolute_path
        if _component_prefix(isolation, absolute):
            relative_to_isolation = absolute.relative_to(isolation)
            if not relative_to_isolation.parts:
                raise OneShotAuthorizationError(
                    "fixture artifact cannot name its isolation directory"
                )
            start_fd = profile.isolation_root_lease.fd
            walk_parts = relative_to_isolation.parts
    parent_fd = _walk_directory_fd_v1(start_fd, walk_parts[:-1])
    try:
        fd = os.open(
            walk_parts[-1],
            os.O_RDONLY | _O_NOFOLLOW | _O_CLOEXEC,
            dir_fd=parent_fd,
        )
    finally:
        os.close(parent_fd)
    metadata = os.fstat(fd)
    if not stat.S_ISREG(metadata.st_mode) or os.get_inheritable(fd):
        os.close(fd)
        raise OneShotAuthorizationError(
            "artifact final component is not a regular CLOEXEC file"
        )
    return fd


def _read_repo_relative_nofollow_v1(
    profile: NamespaceProfileV1,
    relative: str,
    *,
    label: str,
) -> bytes:
    _validate_namespace_profile_v1(profile)
    try:
        fd = _open_repo_relative_nofollow_v1(profile, relative)
    except OSError as error:
        raise OneShotAuthorizationError(
            f"cannot open {label} no-follow"
        ) from error
    try:
        chunks: list[bytes] = []
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        raw = b"".join(chunks)
    finally:
        os.close(fd)
        _validate_namespace_profile_v1(profile)
    return raw


def _require_exact_mapping(
    value: Any,
    keys: set[str] | frozenset[str],
    *,
    label: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != set(keys):
        raise OneShotAuthorizationError(
            f"{label} does not have the exact field set"
        )
    return _strict_json_value(value, label=label)


def _require_nonempty_string(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise OneShotAuthorizationError(
            f"{label} must be a nonempty string"
        )
    return value


def _require_exact_bool(value: Any, expected: bool, *, label: str) -> bool:
    if type(value) is not bool or value is not expected:
        raise OneShotAuthorizationError(
            f"{label} must be exactly {expected}"
        )
    return value


def _require_exact_int(value: Any, expected: int, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value != expected:
        raise OneShotAuthorizationError(
            f"{label} must be exactly integer {expected}"
        )
    return value


def _require_unique_array(
    value: Any,
    *,
    label: str,
    nonempty: bool = False,
) -> list[Any]:
    if not isinstance(value, list) or (nonempty and not value):
        raise OneShotAuthorizationError(
            f"{label} must be an exact JSON array"
        )
    signatures: set[bytes] = set()
    normalized: list[Any] = []
    for item in value:
        strict = _strict_json_value(item, label=f"{label}[]")
        signature = _canonical_json_bytes(strict)
        if signature in signatures:
            raise OneShotAuthorizationError(
                f"{label} contains a duplicate item"
            )
        signatures.add(signature)
        normalized.append(strict)
    return normalized


def _strict_pretty_json_object(
    raw: bytes,
    *,
    label: str,
) -> dict[str, Any]:
    payload = _json_mapping_no_duplicates(raw, label=label)

    def check_arrays(value: Any, path: str) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                check_arrays(child, f"{path}.{key}")
        elif isinstance(value, list):
            _require_unique_array(value, label=path)
            for index, child in enumerate(value):
                check_arrays(child, f"{path}[{index}]")

    check_arrays(payload, label)
    if raw != _pretty_json_bytes(payload):
        raise OneShotAuthorizationError(
            f"{label} raw bytes are not canonical pretty JSON plus one LF"
        )
    return payload


def _strict_json_object_any_raw(
    raw: bytes,
    *,
    label: str,
) -> dict[str, Any]:
    payload = _json_mapping_no_duplicates(raw, label=label)

    def check_arrays(value: Any, path: str) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                check_arrays(child, f"{path}.{key}")
        elif isinstance(value, list):
            _require_unique_array(value, label=path)
            for index, child in enumerate(value):
                check_arrays(child, f"{path}[{index}]")

    check_arrays(payload, label)
    return payload


def _read_bound_pretty_json_v1(
    profile: NamespaceProfileV1,
    binding: Mapping[str, Any],
    *,
    label: str,
) -> tuple[dict[str, Any], bytes, str, str]:
    exact = _require_exact_mapping(
        binding, {"path", "sha256"}, label=f"{label} binding"
    )
    path = _safe_repo_relative_text(
        exact["path"], label=f"{label} path"
    )
    expected = _validate_sha256(
        exact["sha256"], label=f"{label} raw SHA256"
    )
    raw = _read_repo_relative_nofollow_v1(
        profile, path, label=label
    )
    observed = _sha256_bytes(raw)
    if observed != expected:
        raise OneShotAuthorizationError(
            f"{label} raw SHA256 drifted"
        )
    payload = _strict_pretty_json_object(raw, label=label)
    return payload, raw, observed, _sha256_bytes(
        _canonical_json_bytes(payload)
    )


def _build_identity_core_binding_seed_v1(
    profile: NamespaceProfileV1,
    expected_implementation_identity_core_raw_sha256: str,
) -> _IdentityCoreBindingSeedV1:
    _validate_namespace_profile_v1(profile)
    expected = _validate_sha256(
        expected_implementation_identity_core_raw_sha256,
        label="externally supplied implementation identity core SHA256",
    )
    path = _profile_repo_relative(
        profile, profile.implementation_identity_core_path
    )
    raw = _read_repo_relative_nofollow_v1(
        profile, path, label="implementation identity core"
    )
    observed = _sha256_bytes(raw)
    if observed != expected:
        raise OneShotAuthorizationError(
            "implementation identity core does not match its external SHA"
        )
    core = _strict_pretty_json_object(
        raw, label="implementation identity core"
    )
    exact = _require_exact_mapping(
        core,
        {
            "artifact",
            "schema_version",
            "version",
            "provider",
            "model",
            "model_family",
            "agent_id",
            "trace_id",
            "source_kind",
        },
        label="implementation identity core",
    )
    if (
        exact["artifact"] != _IMPLEMENTATION_CORE_ARTIFACT
        or exact["schema_version"] != "1.0"
        or exact["version"] != _IMPLEMENTATION_CORE_VERSION
        or exact["source_kind"]
        != "external_implementation_orchestrator_trace_identity"
    ):
        raise OneShotAuthorizationError(
            "implementation identity core fixed identity drifted"
        )
    for name in (
        "provider",
        "model",
        "model_family",
        "agent_id",
        "trace_id",
    ):
        _require_nonempty_string(
            exact[name], label=f"implementation identity core {name}"
        )
    return _IdentityCoreBindingSeedV1(
        path=path,
        expected_external_raw_sha256=expected,
        observed_raw_sha256=observed,
        parsed_exact_core=exact,
    )


def _validate_dependency_canonical_path(value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise OneShotAuthorizationError(
            "dependency canonical_path must be a nonempty string"
        )
    if any(character in value for character in ("*", "?", "[", "]")):
        raise OneShotAuthorizationError(
            "dependency paths cannot use wildcard or glob authorization"
        )
    if value.endswith(("/", os.sep)) or "\x00" in value:
        raise OneShotAuthorizationError(
            "dependency canonical_path is not an exact file path"
        )
    path = Path(value)
    lexical_parts = PurePosixPath(value.replace(os.sep, "/")).parts
    normalized = os.path.abspath(os.path.normpath(value))
    if (
        not path.is_absolute()
        or ".." in lexical_parts
        or value != normalized
    ):
        raise OneShotAuthorizationError(
            "dependency canonical_path must be absolute and lexically "
            "canonical"
        )
    if os.path.lexists(value):
        try:
            if stat.S_ISLNK(os.lstat(value).st_mode):
                raise OneShotAuthorizationError(
                    "dependency canonical_path cannot name a symlink"
                )
            resolved = str(path.resolve(strict=True))
        except OSError as error:
            raise OneShotAuthorizationError(
                f"dependency canonical_path cannot be resolved: {value}"
            ) from error
        if resolved != value:
            raise OneShotAuthorizationError(
                "dependency canonical_path contains a symlink or alternate "
                "origin"
            )
    return value


def _validate_module_distribution_identity(value: Any) -> Any:
    if isinstance(value, str):
        lowered = value.lower()
        if (
            not value.strip()
            or any(character in value for character in ("*", "?", "[", "]"))
            or "module=" not in lowered
            or "distribution=" not in lowered
        ):
            raise OneShotAuthorizationError(
                "dependency identity string must name exact module= and "
                "distribution= identities, not a package version or prefix"
            )
        return value
    if not isinstance(value, Mapping):
        raise OneShotAuthorizationError(
            "module_and_distribution_identity must be an exact string or "
            "mapping"
        )
    identity = _strict_json_value(value, label="dependency identity")
    if set(identity) not in (
        {"module", "distribution"},
        {"module", "distribution", "version"},
    ):
        raise OneShotAuthorizationError(
            "dependency identity mapping must contain module and "
            "distribution; a version-only authorization is forbidden"
        )
    for key, item in identity.items():
        if (
            not isinstance(item, str)
            or not item.strip()
            or any(character in item for character in ("*", "?", "[", "]"))
        ):
            raise OneShotAuthorizationError(
                f"dependency identity {key} must be an exact nonempty string"
            )
    return identity


def _parse_dependency_manifest(
    raw: bytes | str | Mapping[str, Any],
) -> DependencyManifest:
    """Parse the strict, finite, per-file B/U/R schema.

    Parsing is definition-only: it validates structure and canonical path
    spelling but does not create or authorize a manifest.  File fingerprints
    are checked separately at every closure checkpoint.
    """

    raw_sha: str | None
    if isinstance(raw, bytes):
        raw_bytes = raw
        payload = _json_mapping_no_duplicates(
            raw_bytes, label="v2.3 dependency manifest"
        )
        raw_sha = _sha256_bytes(raw_bytes)
    elif isinstance(raw, str):
        raw_bytes = raw.encode("utf-8")
        payload = _json_mapping_no_duplicates(
            raw_bytes, label="v2.3 dependency manifest"
        )
        raw_sha = _sha256_bytes(raw_bytes)
    elif isinstance(raw, Mapping):
        payload = _strict_json_value(
            raw, label="v2.3 dependency manifest"
        )
        if not isinstance(payload, dict):
            raise OneShotAuthorizationError(
                "v2.3 dependency manifest must be a mapping"
            )
        raw_sha = _sha256_bytes(_canonical_json_bytes(payload))
    else:
        raise OneShotAuthorizationError(
            "v2.3 dependency manifest must be JSON bytes, text, or mapping"
        )

    expected_top = {
        "schema_version",
        "artifact",
        "attempt_id",
        "transport_protocol_version",
        "assurance_profile",
        "members",
        "B",
        "U",
        "R",
    }
    if set(payload) != expected_top:
        raise OneShotAuthorizationError(
            "v2.3 dependency manifest top-level schema drifted"
        )
    if (
        payload.get("schema_version") != "1.0"
        or payload.get("artifact")
        != (
            "actual_wake_reachable_pressure_dependency_closure_v23_"
            "20260728_185556"
        )
        or payload.get("attempt_id") != _ATTEMPT_ID
        or payload.get("transport_protocol_version")
        != _TRANSPORT_PROTOCOL_VERSION
        or payload.get("assurance_profile") != _ASSURANCE_PROFILE
    ):
        raise OneShotAuthorizationError(
            "v2.3 dependency manifest identity or assurance profile drifted"
        )

    raw_members = payload.get("members")
    member_items: list[Mapping[str, Any]]
    if isinstance(raw_members, Mapping):
        member_items = []
        for declared_path, member in raw_members.items():
            if not isinstance(member, Mapping):
                raise OneShotAuthorizationError(
                    "each dependency member must be a mapping"
                )
            if member.get("canonical_path") != declared_path:
                raise OneShotAuthorizationError(
                    "dependency member mapping key must equal canonical_path"
                )
            member_items.append(member)
    elif isinstance(raw_members, list):
        member_items = raw_members
    else:
        raise OneShotAuthorizationError(
            "dependency members must be a list or canonical-path mapping"
        )
    if not member_items:
        raise OneShotAuthorizationError(
            "dependency universe cannot be empty"
        )

    members: list[DependencyMember] = []
    seen_paths: set[str] = set()
    for raw_member in member_items:
        if not isinstance(raw_member, Mapping):
            raise OneShotAuthorizationError(
                "each dependency member must be a mapping"
            )
        if set(raw_member) != _DEPENDENCY_MEMBER_FIELDS:
            raise OneShotAuthorizationError(
                "dependency member does not have the exact per-file schema"
            )
        path = _validate_dependency_canonical_path(
            raw_member.get("canonical_path")
        )
        if path in seen_paths:
            raise OneShotAuthorizationError(
                f"duplicate dependency canonical_path: {path}"
            )
        seen_paths.add(path)
        identity = _validate_module_distribution_identity(
            raw_member.get("module_and_distribution_identity")
        )
        kind = raw_member.get("kind")
        if kind not in {"python_source", "native_binary"}:
            raise OneShotAuthorizationError(
                "dependency kind must be python_source or native_binary"
            )
        if kind == "python_source" and Path(path).suffix != ".py":
            raise OneShotAuthorizationError(
                "python_source dependency must name its exact .py source, "
                "not .pyc/.pyo"
            )
        origin = raw_member.get("origin_or_package")
        if (
            not isinstance(origin, str)
            or not origin.strip()
            or any(character in origin for character in ("*", "?", "[", "]"))
        ):
            raise OneShotAuthorizationError(
                "origin_or_package must be an exact nonempty identity"
            )
        digest = _validate_sha256(
            raw_member.get("sha256"),
            label=f"dependency {path}",
        )
        integer_values: dict[str, int] = {}
        for name in ("st_dev", "st_ino", "st_size"):
            value = raw_member.get(name)
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value < 0
            ):
                raise OneShotAuthorizationError(
                    f"dependency {path} {name} must be a nonnegative integer"
                )
            integer_values[name] = value
        phase = raw_member.get("allowed_phase")
        if phase not in _DEPENDENCY_PHASES:
            raise OneShotAuthorizationError(
                f"dependency {path} has an unknown execution phase"
            )
        requirement = raw_member.get("required_or_optional")
        if requirement not in {"required", "optional"}:
            raise OneShotAuthorizationError(
                "required_or_optional must be required or optional"
            )
        justification = raw_member.get("justification")
        if (
            not isinstance(justification, str)
            or not justification.strip()
        ):
            raise OneShotAuthorizationError(
                f"dependency {path} lacks a per-file justification"
            )
        members.append(
            DependencyMember(
                canonical_path=path,
                module_and_distribution_identity=identity,
                kind=kind,
                origin_or_package=origin,
                sha256=digest,
                st_dev=integer_values["st_dev"],
                st_ino=integer_values["st_ino"],
                st_size=integer_values["st_size"],
                allowed_phase=phase,
                required_or_optional=requirement,
                justification=justification,
            )
        )

    sets: dict[str, tuple[str, ...]] = {}
    for name in ("B", "U", "R"):
        values = payload.get(name)
        if (
            not isinstance(values, list)
            or not all(isinstance(item, str) for item in values)
            or len(values) != len(set(values))
        ):
            raise OneShotAuthorizationError(
                f"dependency set {name} must be an explicit unique path list"
            )
        canonical = tuple(
            _validate_dependency_canonical_path(item) for item in values
        )
        sets[name] = canonical
    B, U, R = sets["B"], sets["U"], sets["R"]
    if set(U) != seen_paths:
        raise OneShotAuthorizationError(
            "dependency U must exactly enumerate all per-file members"
        )
    if not set(B) <= set(U) or not set(R) <= set(U):
        raise OneShotAuthorizationError(
            "dependency B and R must both be subsets of U"
        )
    if set(B) & set(R):
        raise OneShotAuthorizationError(
            "baseline B and mandatory lazy R must be disjoint"
        )
    by_path = {member.canonical_path: member for member in members}
    required = {
        path
        for path, member in by_path.items()
        if member.required_or_optional == "required"
    }
    if required != set(B) | set(R):
        raise OneShotAuthorizationError(
            "required members must be exactly B union R"
        )
    if any(
        by_path[path].allowed_phase != "baseline_pre_marker"
        for path in B
    ):
        raise OneShotAuthorizationError(
            "every baseline B member must be allowed only pre-marker"
        )
    if any(
        by_path[path].allowed_phase == "baseline_pre_marker"
        for path in set(U) - set(B)
    ):
        raise OneShotAuthorizationError(
            "non-baseline members cannot first load in the baseline phase"
        )

    normalized_payload = {
        "schema_version": "1.0",
        "artifact": payload["artifact"],
        "attempt_id": _ATTEMPT_ID,
        "transport_protocol_version": _TRANSPORT_PROTOCOL_VERSION,
        "assurance_profile": _ASSURANCE_PROFILE,
        "members": [member.as_payload() for member in members],
        "B": list(B),
        "U": list(U),
        "R": list(R),
    }
    return DependencyManifest(
        schema_version="1.0",
        artifact=str(payload["artifact"]),
        attempt_id=_ATTEMPT_ID,
        transport_protocol_version=_TRANSPORT_PROTOCOL_VERSION,
        assurance_profile=_ASSURANCE_PROFILE,
        members=tuple(members),
        B=B,
        U=U,
        R=R,
        canonical_sha256=_sha256_bytes(
            _canonical_json_bytes(normalized_payload)
        ),
        raw_sha256=raw_sha,
    )


def _load_dependency_manifest() -> DependencyManifest:
    return _parse_dependency_manifest(
        _read_regular_nofollow(
            DEPENDENCY_MANIFEST_PATH,
            label="v2.3 dependency closure manifest",
        )
    )


def _load_transport_preregistration() -> tuple[dict[str, Any], str]:
    raw = _read_regular_nofollow(
        TRANSPORT_PREREGISTRATION_PATH,
        label="transport-v2.3 preregistration",
    )
    payload = _json_mapping_no_duplicates(
        raw, label="transport-v2.3 preregistration"
    )
    reserved = payload.get("reserved_namespace")
    threat = payload.get("threat_boundary")
    if (
        payload.get("artifact")
        != (
            "actual_wake_reachable_pressure_transport_v23_"
            "preregistration_20260728_185556"
        )
        or payload.get("attempt_id") != _ATTEMPT_ID
        or payload.get("scientific_protocol_version") != "S3ai-v2.2"
        or payload.get("transport_protocol_version")
        != _TRANSPORT_PROTOCOL_VERSION
        or payload.get("status")
        != "PREREGISTERED_IMPLEMENTATION_AND_EXECUTION_NOT_AUTHORIZED"
        or not isinstance(reserved, Mapping)
        or reserved.get("wrapper") != _WRAPPER_RELATIVE
        or reserved.get("dependency_manifest")
        != _validated_repo_relative(DEPENDENCY_MANIFEST_PATH).as_posix()
        or reserved.get("authorization")
        != _validated_repo_relative(AUTHORIZATION_PATH).as_posix()
        or reserved.get("result")
        != _validated_repo_relative(RESULT_PATH).as_posix()
        or reserved.get("marker")
        != _validated_repo_relative(ATTEMPT_MARKER_PATH).as_posix()
        or reserved.get("may_write_old_canonical_result_or_latest")
        is not False
        or not isinstance(threat, Mapping)
        or threat.get("selected_assurance_profile")
        != _ASSURANCE_PROFILE
        or threat.get("cryptographic_protection_from_replace_then_restore_writer")
        is not False
    ):
        raise OneShotAuthorizationError(
            "transport-v2.3 preregistration identity, namespace, or "
            "assurance profile drifted"
        )
    return payload, _sha256_bytes(raw)


def _array_sha256(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(repr(array.shape).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _validated_repo_relative(path: Path) -> PurePosixPath:
    absolute = Path(path).absolute()
    try:
        relative = absolute.relative_to(REPO_ROOT)
    except ValueError as error:
        raise OneShotAuthorizationError(
            f"path escapes the fixed repository root: {path}"
        ) from error
    pure = PurePosixPath(relative.as_posix())
    if (
        pure.is_absolute()
        or not pure.parts
        or any(part in {"", ".", ".."} for part in pure.parts)
    ):
        raise OneShotAuthorizationError(f"unsafe repository path: {path}")
    return pure


def _open_parent_dir_nofollow(path: Path) -> tuple[int, str]:
    relative = _validated_repo_relative(path)
    flags = os.O_RDONLY | _O_DIRECTORY | _O_NOFOLLOW
    try:
        current = os.open(REPO_ROOT, flags)
    except OSError as error:
        raise OneShotAuthorizationError(
            f"cannot open fixed repository root: {error}"
        ) from error
    try:
        for part in relative.parts[:-1]:
            following = os.open(part, flags, dir_fd=current)
            os.close(current)
            current = following
    except OSError as error:
        os.close(current)
        raise OneShotAuthorizationError(
            f"unsafe or unavailable parent for {path}: {error}"
        ) from error
    return current, relative.name


def _acquire_output_directory_lease(
    marker_path: Path,
    result_path: Path,
) -> _OutputDirectoryLease:
    marker_parent, marker_name = _open_parent_dir_nofollow(marker_path)
    result_parent = -1
    try:
        result_parent, result_name = _open_parent_dir_nofollow(result_path)
        marker_directory = os.fstat(marker_parent)
        result_directory = os.fstat(result_parent)
        if (
            marker_directory.st_dev,
            marker_directory.st_ino,
        ) != (
            result_directory.st_dev,
            result_directory.st_ino,
        ):
            raise OneShotAuthorizationError(
                "marker and result must share the fixed output directory"
            )
        return _OutputDirectoryLease(
            fd=marker_parent,
            marker_name=marker_name,
            result_name=result_name,
            st_dev=int(marker_directory.st_dev),
            st_ino=int(marker_directory.st_ino),
        )
    except Exception:
        os.close(marker_parent)
        raise
    finally:
        if result_parent >= 0:
            os.close(result_parent)


def _verify_output_directory_lease(
    lease: _OutputDirectoryLease,
    *,
    marker_path: Path,
    result_path: Path,
) -> None:
    """Require both canonical paths to keep naming the leased directory."""

    held = os.fstat(lease.fd)
    if (
        int(held.st_dev),
        int(held.st_ino),
    ) != (lease.st_dev, lease.st_ino):
        raise OneShotAuthorizationError(
            "held output directory identity changed"
        )
    marker_parent, marker_name = _open_parent_dir_nofollow(marker_path)
    result_parent = -1
    try:
        result_parent, result_name = _open_parent_dir_nofollow(result_path)
        marker_directory = os.fstat(marker_parent)
        result_directory = os.fstat(result_parent)
        expected_identity = (lease.st_dev, lease.st_ino)
        if (
            marker_name != lease.marker_name
            or result_name != lease.result_name
            or (
                int(marker_directory.st_dev),
                int(marker_directory.st_ino),
            )
            != expected_identity
            or (
                int(result_directory.st_dev),
                int(result_directory.st_ino),
            )
            != expected_identity
        ):
            raise OneShotAuthorizationError(
                "canonical output directory was replaced during the "
                "one-shot attempt"
            )
    finally:
        if result_parent >= 0:
            os.close(result_parent)
        os.close(marker_parent)


def _entry_lstat(path: Path) -> os.stat_result | None:
    parent_fd, name = _open_parent_dir_nofollow(path)
    try:
        try:
            return os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            return None
    finally:
        os.close(parent_fd)


def _require_absent(path: Path, *, label: str) -> None:
    entry = _entry_lstat(path)
    if entry is not None:
        kind = "symlink" if stat.S_ISLNK(entry.st_mode) else "existing entry"
        raise OneShotAuthorizationError(
            f"{label} must be absent before execution ({kind}: {path})"
        )


def _read_regular_nofollow(path: Path, *, label: str) -> bytes:
    parent_fd, name = _open_parent_dir_nofollow(path)
    fd = -1
    try:
        fd = os.open(name, os.O_RDONLY | _O_NOFOLLOW, dir_fd=parent_fd)
        metadata = os.fstat(fd)
        if not stat.S_ISREG(metadata.st_mode):
            raise OneShotAuthorizationError(
                f"{label} is not a regular file: {path}"
            )
        chunks: list[bytes] = []
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        return b"".join(chunks)
    except FileNotFoundError as error:
        raise OneShotAuthorizationError(
            f"{label} is missing: {path}"
        ) from error
    except OSError as error:
        raise OneShotAuthorizationError(
            f"cannot read {label} without following links: {path}: {error}"
        ) from error
    finally:
        if fd >= 0:
            os.close(fd)
        os.close(parent_fd)


def _parse_yaml_mapping(raw: bytes, *, label: str) -> dict[str, Any]:
    try:
        payload = yaml.load(
            raw.decode("utf-8"), Loader=_UniqueKeySafeLoader
        )
    except (UnicodeDecodeError, yaml.YAMLError) as error:
        raise OneShotAuthorizationError(
            f"{label} is not valid UTF-8 YAML"
        ) from error
    if not isinstance(payload, dict):
        raise OneShotAuthorizationError(f"{label} must be a mapping")
    _strict_json_value(payload, label=label)
    return payload


def _load_wrapper_definition() -> tuple[dict[str, Any], str]:
    raw = _read_regular_nofollow(
        WRAPPER_DEFINITION_PATH, label="wrapper preregistration"
    )
    payload = _parse_yaml_mapping(raw, label="wrapper preregistration")
    if (
        payload.get("artifact")
        != "actual_wake_reachable_pressure_execution_wrapper_preregistration"
        or payload.get("version") != "S3ai-v2.2-one-shot-wrapper-v1"
        or payload.get("status")
        != "preregistered_before_wrapper_implementation"
        or payload.get("role")
        != "execution_transport_only_no_physics_no_result"
    ):
        raise OneShotAuthorizationError(
            "wrapper preregistration identity drifted"
        )
    decision = payload.get("decision", {})
    if (
        not isinstance(decision, Mapping)
        or decision.get("wrapper_implementation_allowed") is not True
        or decision.get("second_bounded_cross_family_audit_required")
        is not True
        or decision.get(
            "authorization_ticket_creation_allowed_before_second_bounded_audit"
        )
        is not False
        or decision.get(
            "authorization_ticket_creation_requires_accepted_second_bounded_audit"
        )
        is not True
        or decision.get("formal_execution_allowed") is not False
        or decision.get("production_activation_allowed") is not False
        or decision.get("no_result_exists_at_preregistration") is not True
    ):
        raise OneShotAuthorizationError(
            "wrapper preregistration must remain execution fail-closed"
        )
    closure = payload.get("wrapper_contract", {}).get("source_closure", {})
    expected = closure.get("exact_preimplementation_closure")
    if (
        not isinstance(expected, list)
        or len(expected) != 62
        or len(set(expected)) != 62
        or expected != sorted(expected)
        or closure.get("frozen_runner_reported_count") != 15
        or closure.get("actual_local_closure_count_before_wrapper") != 62
        or closure.get("required_execution_count_with_wrapper") != 63
    ):
        raise OneShotAuthorizationError(
            "wrapper preregistration source-closure accounting drifted"
        )
    for item in expected:
        if (
            not isinstance(item, str)
            or not item.startswith("platform/")
            or PurePosixPath(item).is_absolute()
            or ".." in PurePosixPath(item).parts
            or not item.endswith(".py")
        ):
            raise OneShotAuthorizationError(
                "wrapper preregistration contains an unsafe source path"
            )
    manifest = "".join(f"{item}\n" for item in expected).encode("utf-8")
    declared_manifest = _validate_sha256(
        closure.get("ordered_path_manifest_sha256_newline_terminated"),
        label="source path manifest",
    )
    if _sha256_bytes(manifest) != declared_manifest:
        raise OneShotAuthorizationError(
            "wrapper preregistration source path manifest drifted"
        )
    return payload, _sha256_bytes(raw)


def _source_fingerprints(
    definition: Mapping[str, Any],
    *,
    require_clean_import: bool = True,
) -> dict[str, str]:
    expected_pre = tuple(
        definition["wrapper_contract"]["source_closure"][
            "exact_preimplementation_closure"
        ]
    )
    expected = tuple(sorted((*expected_pre, _WRAPPER_RELATIVE)))
    observed = _loaded_local_source_paths()
    if require_clean_import and not _IMPORT_WAS_CLEAN:
        raise OneShotAuthorizationError(
            "formal execution requires a fresh process with no preloaded "
            "repository-local modules"
        )
    if observed != expected:
        missing = sorted(set(expected) - set(observed))
        extra = sorted(set(observed) - set(expected))
        raise OneShotAuthorizationError(
            "loaded local source set differs from the audited 63-file set; "
            f"missing={missing}, extra={extra}"
        )
    fingerprints = {
        relative: _sha256_bytes(
            _read_regular_nofollow(
                REPO_ROOT / relative, label=f"execution source {relative}"
            )
        )
        for relative in expected
    }
    if require_clean_import and (
        not _IMPORT_SOURCE_SNAPSHOT_STABLE
        or expected_pre != _EARLY_DECLARED_LOCAL_SOURCES
        or {
            path: fingerprints[path] for path in expected_pre
        }
        != _LOCAL_SOURCE_BYTES_BEFORE_GUARD
        or fingerprints.get(_WRAPPER_RELATIVE)
        != _WRAPPER_BYTES_BEFORE_GUARD
        or _sha256_bytes(
            _read_regular_nofollow(
                WRAPPER_DEFINITION_PATH,
                label="wrapper preregistration",
            )
        )
        != _DEFINITION_BYTES_BEFORE_GUARD
    ):
        raise OneShotAuthorizationError(
            "local source bytes changed before, during, or after the frozen "
            "runner import"
        )
    frozen = guard._code_fingerprints()
    if (
        len(frozen) != 15
        or any(
            relative not in fingerprints
            or fingerprints[relative] != digest
            for relative, digest in frozen.items()
        )
    ):
        raise OneShotAuthorizationError(
            "frozen runner's 15-file inventory is not an exact hash subset "
            "of the audited execution source set"
        )
    return fingerprints


def _require_source_execution_mode() -> None:
    if (
        not _SOURCE_IMPORT_MODE_AT_GUARD_IMPORT
        or not isinstance(_BOOTSTRAP_WRAPPER_SHA256, str)
        or _BOOTSTRAP_WRAPPER_SHA256 != _WRAPPER_BYTES_BEFORE_GUARD
        or not sys.dont_write_bytecode
        or sys.pycache_prefix is None
        or Path(sys.pycache_prefix).absolute() != NO_BYTECODE_CACHE_PATH
        or os.path.lexists(NO_BYTECODE_CACHE_PATH)
    ):
        raise OneShotAuthorizationError(
            "formal execution must start with -B and "
            f"-X pycache_prefix={NO_BYTECODE_CACHE_PATH}, while that fixed "
            "cache path remains absent, and must compile this wrapper from "
            "the same no-follow bytes whose SHA256 is injected by the fixed "
            "stdin bootstrap; this forces all local imports from the audited "
            "source bytes"
        )


def _verify_loaded_local_modules_use_source(
    definition: Mapping[str, Any],
) -> None:
    expected = set(
        definition["wrapper_contract"]["source_closure"][
            "exact_preimplementation_closure"
        ]
    )
    expected.add(_WRAPPER_RELATIVE)
    observed: set[str] = set()
    for module in tuple(sys.modules.values()):
        source = _source_path_from_module(module)
        if source is None:
            continue
        try:
            relative = source.relative_to(REPO_ROOT).as_posix()
        except ValueError:
            continue
        if relative not in expected:
            continue
        spec = getattr(module, "__spec__", None)
        raw_origin = getattr(spec, "origin", None) or getattr(
            module, "__file__", None
        )
        if not raw_origin or Path(raw_origin).suffix != ".py":
            raise OneShotAuthorizationError(
                f"local module {relative} was not loaded from Python source"
            )
        cached = getattr(module, "__cached__", None)
        if cached and os.path.lexists(cached):
            raise OneShotAuthorizationError(
                f"local module {relative} used an existing bytecode cache"
            )
        observed.add(relative)
    if observed != expected:
        raise OneShotAuthorizationError(
            "could not prove that every audited local module was loaded "
            "from the bound source bytes"
        )


def _loaded_native_binary_fingerprints() -> dict[str, dict[str, Any]]:
    paths: set[Path] = set()
    mapped: dict[Path, tuple[str, int]] = {}
    native_suffixes = {".so", ".pyd", ".dll", ".dylib"}
    for module in tuple(sys.modules.values()):
        raw = getattr(module, "__file__", None)
        if not raw:
            continue
        path = Path(raw).absolute()
        if path.suffix in native_suffixes or ".so." in path.name:
            paths.add(path)
    maps = Path("/proc/self/maps")
    if maps.is_file():
        for line in maps.read_text(encoding="utf-8").splitlines():
            fields = line.split(maxsplit=5)
            if len(fields) != 6 or not fields[5].startswith("/"):
                continue
            raw = fields[5]
            if raw.endswith(" (deleted)"):
                raise OneShotAuthorizationError(
                    "a loaded native dependency has been deleted or replaced "
                    f"while mapped: {raw}"
                )
            path = Path(raw).absolute()
            if path.suffix in native_suffixes or ".so." in path.name:
                paths.add(path)
                identity = (fields[3], int(fields[4]))
                previous = mapped.get(path)
                if previous is not None and previous != identity:
                    raise OneShotAuthorizationError(
                        f"native dependency {path} has inconsistent mapped "
                        "device/inode identities"
                    )
                mapped[path] = identity
    fingerprints: dict[str, dict[str, Any]] = {}
    for path in sorted(paths, key=str):
        try:
            metadata = path.stat()
        except OSError as error:
            raise OneShotAuthorizationError(
                f"cannot fingerprint loaded native dependency {path}: {error}"
            ) from error
        if not stat.S_ISREG(metadata.st_mode):
            raise OneShotAuthorizationError(
                f"loaded native dependency is not a regular file: {path}"
            )
        mapped_identity = mapped.get(path)
        if mapped_identity is not None:
            expected_device = (
                f"{os.major(metadata.st_dev):02x}:"
                f"{os.minor(metadata.st_dev):02x}"
            )
            if (
                mapped_identity[0].lower() != expected_device.lower()
                or mapped_identity[1] != metadata.st_ino
            ):
                raise OneShotAuthorizationError(
                    f"loaded native dependency {path} no longer names its "
                    "mapped inode"
                )
        fingerprints[str(path)] = {
            "sha256": _sha256_bytes(path.read_bytes()),
            "st_dev": int(metadata.st_dev),
            "st_ino": int(metadata.st_ino),
            "st_size": int(metadata.st_size),
            "mapped_device": (
                mapped_identity[0] if mapped_identity is not None else None
            ),
            "mapped_inode": (
                mapped_identity[1] if mapped_identity is not None else None
            ),
        }
    if not fingerprints:
        raise OneShotAuthorizationError(
            "runtime identity found no loaded native dependencies"
        )
    return fingerprints


def _loaded_python_dependency_fingerprints() -> dict[str, dict[str, Any]]:
    paths: set[Path] = set()
    for module in tuple(sys.modules.values()):
        path = _source_path_from_module(module)
        if path is None:
            continue
        try:
            path.relative_to(REPO_ROOT)
        except ValueError:
            paths.add(path)
    fingerprints: dict[str, dict[str, Any]] = {}
    for path in sorted(paths, key=str):
        try:
            metadata = path.stat()
            raw = path.read_bytes()
        except OSError as error:
            raise OneShotAuthorizationError(
                f"cannot fingerprint loaded Python dependency {path}: {error}"
            ) from error
        if not stat.S_ISREG(metadata.st_mode):
            raise OneShotAuthorizationError(
                f"loaded Python dependency is not a regular file: {path}"
            )
        fingerprints[str(path)] = {
            "sha256": _sha256_bytes(raw),
            "st_dev": int(metadata.st_dev),
            "st_ino": int(metadata.st_ino),
            "st_size": int(metadata.st_size),
        }
    if not fingerprints:
        raise OneShotAuthorizationError(
            "runtime identity found no loaded Python dependencies"
        )
    return fingerprints


def _fingerprint_dependency_file(
    path_value: str,
    *,
    kind: str,
) -> dict[str, Any]:
    """Read one exact dependency file without following its final symlink."""

    path = Path(_validate_dependency_canonical_path(path_value))
    if kind == "python_source" and path.suffix != ".py":
        raise OneShotAuthorizationError(
            f"authorized Python dependency is not source: {path}"
        )
    try:
        resolved_before = str(path.resolve(strict=True))
    except OSError as error:
        raise OneShotAuthorizationError(
            f"authorized dependency is missing: {path}"
        ) from error
    if resolved_before != str(path):
        raise OneShotAuthorizationError(
            f"authorized dependency uses a symlink/alternate origin: {path}"
        )
    fd = -1
    try:
        fd = os.open(path, os.O_RDONLY | _O_NOFOLLOW)
        metadata = os.fstat(fd)
        if not stat.S_ISREG(metadata.st_mode):
            raise OneShotAuthorizationError(
                f"authorized dependency is not a regular file: {path}"
            )
        chunks: list[bytes] = []
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        linked = os.stat(path, follow_symlinks=False)
        if (
            int(linked.st_dev),
            int(linked.st_ino),
            int(linked.st_size),
        ) != (
            int(metadata.st_dev),
            int(metadata.st_ino),
            int(metadata.st_size),
        ):
            raise OneShotAuthorizationError(
                f"authorized dependency inode changed while read: {path}"
            )
        if str(path.resolve(strict=True)) != str(path):
            raise OneShotAuthorizationError(
                f"authorized dependency origin changed while read: {path}"
            )
        return {
            "sha256": _sha256_bytes(b"".join(chunks)),
            "st_dev": int(metadata.st_dev),
            "st_ino": int(metadata.st_ino),
            "st_size": int(metadata.st_size),
            "kind": kind,
        }
    except OneShotAuthorizationError:
        raise
    except OSError as error:
        raise OneShotAuthorizationError(
            f"cannot fingerprint authorized dependency {path}: {error}"
        ) from error
    finally:
        if fd >= 0:
            os.close(fd)


def _manifest_member_fingerprint(
    member: DependencyMember,
) -> dict[str, Any]:
    return {
        "sha256": member.sha256,
        "st_dev": member.st_dev,
        "st_ino": member.st_ino,
        "st_size": member.st_size,
        "kind": member.kind,
    }


def _expected_loaded_state_fingerprint(
    member: DependencyMember,
) -> dict[str, Any]:
    """Return the checkpoint evidence required for one loaded member.

    A native file fingerprint alone does not prove that the process mapping
    still names that inode.  Bind the `/proc/self/maps` device/inode identity
    for native members while keeping the preregistered 11-field manifest
    schema unchanged.
    """

    fingerprint = _manifest_member_fingerprint(member)
    if member.kind == "native_binary":
        fingerprint["mapped_device"] = (
            f"{os.major(member.st_dev):02x}:"
            f"{os.minor(member.st_dev):02x}"
        ).lower()
        fingerprint["mapped_inode"] = member.st_ino
    return fingerprint


def _verify_all_manifest_members(
    manifest: DependencyManifest,
) -> dict[str, dict[str, Any]]:
    observed: dict[str, dict[str, Any]] = {}
    for member in manifest.members:
        fingerprint = _fingerprint_dependency_file(
            member.canonical_path, kind=member.kind
        )
        if fingerprint != _manifest_member_fingerprint(member):
            raise OneShotAuthorizationError(
                "authorized dependency bytes/device/inode/size drifted: "
                f"{member.canonical_path}"
            )
        observed[member.canonical_path] = fingerprint
    return observed


def _native_mapping_identities() -> dict[str, tuple[str, int]]:
    maps_path = Path("/proc/self/maps")
    if not maps_path.is_file():
        raise OneShotAuthorizationError(
            "dependency closure requires /proc/self/maps for native identity"
        )
    identities: dict[str, tuple[str, int]] = {}
    try:
        lines = maps_path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise OneShotAuthorizationError(
            "cannot read /proc/self/maps for native dependency identity"
        ) from error
    for line in lines:
        fields = line.split(maxsplit=5)
        if len(fields) != 6 or not fields[5].startswith("/"):
            continue
        raw = fields[5]
        if raw.endswith(" (deleted)"):
            raise OneShotAuthorizationError(
                f"loaded native mapping was deleted or replaced: {raw}"
            )
        path = Path(raw)
        if not (
            path.suffix in {".so", ".pyd", ".dll", ".dylib"}
            or ".so." in path.name
        ):
            continue
        absolute = str(path.absolute())
        try:
            canonical = str(path.resolve(strict=True))
        except OSError as error:
            raise OneShotAuthorizationError(
                f"loaded native mapping has no stable origin: {raw}"
            ) from error
        if absolute != canonical:
            raise OneShotAuthorizationError(
                f"loaded native mapping uses an alternate origin: {raw}"
            )
        identity = (fields[3].lower(), int(fields[4]))
        previous = identities.get(canonical)
        if previous is not None and previous != identity:
            raise OneShotAuthorizationError(
                f"native mapping identity is inconsistent: {canonical}"
            )
        identities[canonical] = identity
    return identities


def _loaded_dependency_state() -> dict[str, dict[str, Any]]:
    """Return the checkpoint-confirmed external Python/native loaded set."""

    paths: dict[str, dict[str, Any]] = {}
    native_mappings = _native_mapping_identities()

    def add(path_value: str, kind: str, module_name: str | None) -> None:
        absolute = str(Path(path_value).absolute())
        try:
            resolved = str(Path(absolute).resolve(strict=True))
        except OSError as error:
            raise OneShotAuthorizationError(
                f"loaded dependency origin disappeared: {absolute}"
            ) from error
        if absolute != resolved:
            raise OneShotAuthorizationError(
                f"loaded dependency used a symlink/alternate origin: {absolute}"
            )
        try:
            Path(resolved).relative_to(REPO_ROOT)
            return
        except ValueError:
            pass
        if kind == "python_bytecode":
            raise OneShotAuthorizationError(
                f"loaded Python dependency used bytecode instead of source: "
                f"{resolved}"
            )
        fingerprint = _fingerprint_dependency_file(resolved, kind=kind)
        entry = paths.get(resolved)
        if entry is None:
            entry = dict(fingerprint)
            entry["module_names"] = []
            paths[resolved] = entry
        elif any(
            entry.get(key) != value
            for key, value in fingerprint.items()
        ):
            raise OneShotAuthorizationError(
                f"loaded dependency identity changed during scan: {resolved}"
            )
        if module_name is not None:
            names = entry["module_names"]
            if module_name not in names:
                names.append(module_name)

    for name, module in tuple(sys.modules.items()):
        spec = getattr(module, "__spec__", None)
        raw = getattr(spec, "origin", None) or getattr(
            module, "__file__", None
        )
        if not raw or raw in {"built-in", "frozen"}:
            continue
        path = Path(raw)
        if path.suffix == ".py":
            kind = "python_source"
        elif path.suffix in {".pyc", ".pyo"}:
            kind = "python_bytecode"
        elif (
            path.suffix in {".so", ".pyd", ".dll", ".dylib"}
            or ".so." in path.name
        ):
            kind = "native_binary"
        else:
            continue
        add(str(path), kind, name)

    for path, (mapped_device, mapped_inode) in native_mappings.items():
        add(path, "native_binary", None)
        metadata = os.stat(path, follow_symlinks=False)
        expected_device = (
            f"{os.major(metadata.st_dev):02x}:"
            f"{os.minor(metadata.st_dev):02x}"
        ).lower()
        if (
            mapped_device != expected_device
            or mapped_inode != int(metadata.st_ino)
        ):
            raise OneShotAuthorizationError(
                f"native mapping no longer names its authorized inode: {path}"
            )
        paths[path]["mapped_device"] = mapped_device
        paths[path]["mapped_inode"] = mapped_inode

    for path, entry in paths.items():
        if entry["kind"] == "native_binary" and (
            path not in native_mappings
            or "mapped_device" not in entry
            or "mapped_inode" not in entry
        ):
            raise OneShotAuthorizationError(
                "loaded native dependency has no matching /proc/self/maps "
                f"identity: {path}"
            )
        entry["module_names"] = sorted(entry["module_names"])
    if not paths:
        raise OneShotAuthorizationError(
            "dependency closure observed no external file-backed members"
        )
    return dict(sorted(paths.items()))


def _loaded_state_fingerprint(
    entry: Mapping[str, Any],
) -> dict[str, Any]:
    fingerprint = {
        key: entry.get(key)
        for key in ("sha256", "st_dev", "st_ino", "st_size", "kind")
    }
    if entry.get("kind") == "native_binary":
        fingerprint["mapped_device"] = entry.get("mapped_device")
        fingerprint["mapped_inode"] = entry.get("mapped_inode")
    return fingerprint


class _DependencyLedger:
    """Monotonic research-profile import/load ledger.

    The ledger records observed import requests and checkpoint-confirmed
    successful loads.  It intentionally makes no adversarial claim about
    transient C-level loads or failed/partially initialized imports between
    checkpoints.
    """

    def __init__(self, manifest: DependencyManifest) -> None:
        self.manifest = manifest
        self._events: list[dict[str, Any]] = []
        self._ever_seen: set[str] = set()
        self._last_loaded: set[str] = set()
        self._snapshots: list[dict[str, Any]] = []
        self._last_event_sha256 = _ZERO_SHA256

    @property
    def events(self) -> tuple[dict[str, Any], ...]:
        return tuple(dict(event) for event in self._events)

    @property
    def ever_seen(self) -> tuple[str, ...]:
        return tuple(sorted(self._ever_seen))

    @property
    def snapshots(self) -> tuple[dict[str, Any], ...]:
        return tuple(
            json.loads(_canonical_json_bytes(item).decode("utf-8"))
            for item in self._snapshots
        )

    def _append(self, event_type: str, phase: str, **details: Any) -> None:
        event = {
            "sequence": len(self._events),
            "event_type": event_type,
            "phase": phase,
            **details,
            "previous_event_sha256": self._last_event_sha256,
        }
        event["event_sha256"] = _sha256_bytes(
            _canonical_json_bytes(event)
        )
        self._events.append(event)
        self._last_event_sha256 = event["event_sha256"]

    def observe_import_request(
        self,
        requested_name: str,
        *,
        phase: str,
        fromlist: Sequence[str] | None,
        level: int,
    ) -> None:
        if phase not in _DEPENDENCY_PHASES:
            raise OneShotAuthorizationError(
                "import request used an unknown dependency phase"
            )
        self._append(
            "observed_import_request",
            phase,
            requested_name=str(requested_name),
            fromlist=[str(item) for item in (fromlist or ())],
            level=int(level),
        )

    def checkpoint(
        self,
        *,
        phase: str,
        actual_state: Mapping[str, Mapping[str, Any]] | None = None,
        require_baseline: bool = False,
        require_completion: bool = False,
    ) -> dict[str, Any]:
        if phase not in _DEPENDENCY_PHASES:
            raise OneShotAuthorizationError(
                "dependency checkpoint used an unknown phase"
            )
        _verify_all_manifest_members(self.manifest)
        observed = (
            dict(actual_state)
            if actual_state is not None
            else _loaded_dependency_state()
        )
        loaded = set(observed)
        universe = set(self.manifest.U)
        baseline = set(self.manifest.B)
        required_lazy = set(self.manifest.R)
        if require_baseline and loaded != baseline:
            raise OneShotAuthorizationError(
                "marker precondition requires actual S0 == B exactly; "
                f"missing={sorted(baseline - loaded)}, "
                f"extra={sorted(loaded - baseline)}"
            )
        unauthorized = loaded - universe
        if unauthorized:
            raise OneShotAuthorizationError(
                "checkpoint observed an unauthorized dependency member: "
                f"{sorted(unauthorized)}"
            )
        by_path = self.manifest.member_by_path
        for path in sorted(loaded):
            expected = _expected_loaded_state_fingerprint(by_path[path])
            actual = _loaded_state_fingerprint(observed[path])
            if actual != expected:
                raise OneShotAuthorizationError(
                    f"loaded dependency fingerprint drifted: {path}"
                )
        removed = self._last_loaded - loaded
        for path in sorted(removed):
            self._append(
                "registered_member_removed",
                phase,
                canonical_path=path,
            )
        if removed:
            raise OneShotAuthorizationError(
                "a checkpoint-registered dependency was later removed: "
                f"{sorted(removed)}"
            )
        first_seen = loaded - self._ever_seen
        for path in sorted(first_seen):
            member = by_path[path]
            if member.allowed_phase != phase:
                raise OneShotAuthorizationError(
                    "dependency first appeared in the wrong execution phase: "
                    f"{path}; expected={member.allowed_phase}, actual={phase}"
                )
            self._append(
                "checkpoint_confirmed_successful_load",
                phase,
                canonical_path=path,
                fingerprint=_expected_loaded_state_fingerprint(member),
            )
            self._ever_seen.add(path)
        if not baseline <= self._ever_seen:
            raise OneShotAuthorizationError(
                "dependency ledger lost the mandatory baseline B"
            )
        optional = {
            member.canonical_path
            for member in self.manifest.members
            if member.required_or_optional == "optional"
        }
        if require_completion:
            if not required_lazy <= self._ever_seen:
                raise OneShotAuthorizationError(
                    "declared path did not load every mandatory R member: "
                    f"{sorted(required_lazy - self._ever_seen)}"
                )
            if not optional and self._ever_seen != universe:
                raise OneShotAuthorizationError(
                    "manifest has no audited optional members, so E must "
                    "equal U at completion"
                )
        self._last_loaded = loaded
        snapshot_without_digest = {
            "phase": phase,
            "loaded_paths": sorted(loaded),
            "loaded_members": {
                path: _loaded_state_fingerprint(observed[path])
                for path in sorted(loaded)
            },
            "loaded_paths_sha256": _sha256_bytes(
                _canonical_json_bytes(sorted(loaded))
            ),
            "ever_seen_paths": sorted(self._ever_seen),
            "ever_seen_paths_sha256": _sha256_bytes(
                _canonical_json_bytes(sorted(self._ever_seen))
            ),
            "event_count": len(self._events),
            "event_chain_sha256": self._last_event_sha256,
        }
        snapshot = {
            **snapshot_without_digest,
            "snapshot_sha256": _sha256_bytes(
                _canonical_json_bytes(snapshot_without_digest)
            ),
        }
        self._snapshots.append(snapshot)
        return json.loads(_canonical_json_bytes(snapshot).decode("utf-8"))


class _ImportRequestObserver:
    """Best-effort Python import-request observer for the selected profile."""

    def __init__(self, ledger: _DependencyLedger, phase: str) -> None:
        self.ledger = ledger
        self.phase = phase
        self._original: Any = None

    def __enter__(self) -> "_ImportRequestObserver":
        self._original = builtins.__import__

        def observed_import(
            name: str,
            globals: Mapping[str, Any] | None = None,
            locals: Mapping[str, Any] | None = None,
            fromlist: Sequence[str] = (),
            level: int = 0,
        ) -> Any:
            self.ledger.observe_import_request(
                name,
                phase=self.phase,
                fromlist=fromlist,
                level=level,
            )
            return self._original(name, globals, locals, fromlist, level)

        builtins.__import__ = observed_import
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self._original is not None:
            builtins.__import__ = self._original


def _dependency_closure_provenance(
    *,
    manifest: DependencyManifest,
    ledger: _DependencyLedger,
    start_snapshot: Mapping[str, Any],
    end_snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    start_paths = set(start_snapshot["ever_seen_paths"])
    end_paths = set(end_snapshot["ever_seen_paths"])
    delta = sorted(end_paths - start_paths)
    ever_seen = sorted(end_paths)
    events = [dict(event) for event in ledger.events]
    provenance = {
        "assurance_profile": _ASSURANCE_PROFILE,
        "malicious_local_writer_protection": False,
        "manifest": {
            "path": _validated_repo_relative(
                DEPENDENCY_MANIFEST_PATH
            ).as_posix(),
            "raw_sha256": manifest.raw_sha256,
            "canonical_sha256": manifest.canonical_sha256,
            "payload": manifest.as_payload(),
        },
        "start": dict(start_snapshot),
        "end": dict(end_snapshot),
        "delta_paths": delta,
        "delta_paths_sha256": _sha256_bytes(
            _canonical_json_bytes(delta)
        ),
        "ever_seen_paths": ever_seen,
        "ever_seen_paths_sha256": _sha256_bytes(
            _canonical_json_bytes(ever_seen)
        ),
        "import_load_ledger": {
            "events": events,
            "event_count": len(events),
            "event_chain_sha256": (
                events[-1]["event_sha256"] if events else _ZERO_SHA256
            ),
            "events_sha256": _sha256_bytes(
                _canonical_json_bytes(events)
            ),
        },
    }
    provenance["closure_certificate_sha256"] = _sha256_bytes(
        _canonical_json_bytes(provenance)
    )
    return provenance


def _validate_dependency_snapshot(
    snapshot: Any,
    *,
    manifest: DependencyManifest,
) -> None:
    expected_keys = {
        "phase",
        "loaded_paths",
        "loaded_members",
        "loaded_paths_sha256",
        "ever_seen_paths",
        "ever_seen_paths_sha256",
        "event_count",
        "event_chain_sha256",
        "snapshot_sha256",
    }
    if not isinstance(snapshot, Mapping) or set(snapshot) != expected_keys:
        raise OneShotAuthorizationError(
            "serialized dependency snapshot schema drifted"
        )
    phase = snapshot.get("phase")
    loaded_paths = snapshot.get("loaded_paths")
    loaded_members = snapshot.get("loaded_members")
    ever_seen = snapshot.get("ever_seen_paths")
    if (
        phase not in _DEPENDENCY_PHASES
        or not isinstance(loaded_paths, list)
        or loaded_paths != sorted(loaded_paths)
        or len(loaded_paths) != len(set(loaded_paths))
        or not isinstance(ever_seen, list)
        or ever_seen != sorted(ever_seen)
        or len(ever_seen) != len(set(ever_seen))
        or not isinstance(loaded_members, Mapping)
        or set(loaded_members) != set(loaded_paths)
        or not set(loaded_paths) <= set(ever_seen) <= set(manifest.U)
        or not isinstance(snapshot.get("event_count"), int)
        or isinstance(snapshot.get("event_count"), bool)
        or snapshot["event_count"] < 0
    ):
        raise OneShotAuthorizationError(
            "serialized dependency snapshot membership drifted"
        )
    by_path = manifest.member_by_path
    for path, fingerprint in loaded_members.items():
        expected_fingerprint = _expected_loaded_state_fingerprint(
            by_path[path]
        )
        if (
            not isinstance(fingerprint, Mapping)
            or set(fingerprint) != set(expected_fingerprint)
            or dict(fingerprint) != expected_fingerprint
        ):
            raise OneShotAuthorizationError(
                f"serialized dependency fingerprint drifted: {path}"
            )
    if (
        snapshot.get("loaded_paths_sha256")
        != _sha256_bytes(_canonical_json_bytes(loaded_paths))
        or snapshot.get("ever_seen_paths_sha256")
        != _sha256_bytes(_canonical_json_bytes(ever_seen))
    ):
        raise OneShotAuthorizationError(
            "serialized dependency snapshot set digest drifted"
        )
    without_digest = dict(snapshot)
    declared = without_digest.pop("snapshot_sha256")
    if (
        _validate_sha256(
            declared, label="dependency snapshot certificate"
        )
        != declared
        or declared != _sha256_bytes(_canonical_json_bytes(without_digest))
    ):
        raise OneShotAuthorizationError(
            "serialized dependency snapshot certificate drifted"
        )
    _validate_sha256(
        snapshot.get("event_chain_sha256"),
        label="dependency snapshot event chain",
    )


def _validate_dependency_closure_provenance(
    value: Any,
    *,
    expected_manifest: DependencyManifest,
) -> None:
    expected_keys = {
        "assurance_profile",
        "malicious_local_writer_protection",
        "manifest",
        "start",
        "end",
        "delta_paths",
        "delta_paths_sha256",
        "ever_seen_paths",
        "ever_seen_paths_sha256",
        "import_load_ledger",
        "closure_certificate_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != expected_keys:
        raise OneShotAuthorizationError(
            "serialized dependency closure provenance schema drifted"
        )
    if (
        value.get("assurance_profile") != _ASSURANCE_PROFILE
        or value.get("malicious_local_writer_protection") is not False
    ):
        raise OneShotAuthorizationError(
            "serialized dependency assurance profile drifted"
        )
    manifest_record = value.get("manifest")
    if (
        not isinstance(manifest_record, Mapping)
        or set(manifest_record)
        != {"path", "raw_sha256", "canonical_sha256", "payload"}
        or manifest_record.get("path")
        != _validated_repo_relative(DEPENDENCY_MANIFEST_PATH).as_posix()
    ):
        raise OneShotAuthorizationError(
            "serialized dependency manifest identity drifted"
        )
    manifest = _parse_dependency_manifest(manifest_record.get("payload"))
    if (
        _validate_sha256(
            manifest_record.get("raw_sha256"),
            label="serialized dependency manifest raw digest",
        )
        != manifest_record.get("raw_sha256")
        or manifest_record.get("canonical_sha256")
        != manifest.canonical_sha256
        or expected_manifest.raw_sha256 is None
        or manifest_record.get("raw_sha256")
        != expected_manifest.raw_sha256
        or manifest_record.get("canonical_sha256")
        != expected_manifest.canonical_sha256
        or manifest.as_payload() != expected_manifest.as_payload()
    ):
        raise OneShotAuthorizationError(
            "serialized dependency manifest digest or authorization-bound "
            "payload drifted"
        )
    start = value.get("start")
    end = value.get("end")
    _validate_dependency_snapshot(start, manifest=manifest)
    _validate_dependency_snapshot(end, manifest=manifest)
    if (
        start["phase"] != "baseline_pre_marker"
        or set(start["loaded_paths"]) != set(manifest.B)
        or set(start["ever_seen_paths"]) != set(manifest.B)
        or end["phase"] != "post_collector_pre_publication"
        or set(end["loaded_paths"]) != set(end["ever_seen_paths"])
        or not set(manifest.R) <= set(end["ever_seen_paths"])
        or not set(manifest.B) <= set(end["ever_seen_paths"]) <= set(
            manifest.U
        )
    ):
        raise OneShotAuthorizationError(
            "serialized dependency B/U/R/E closure predicate failed"
        )
    optional = {
        member.canonical_path
        for member in manifest.members
        if member.required_or_optional == "optional"
    }
    if not optional and set(end["ever_seen_paths"]) != set(manifest.U):
        raise OneShotAuthorizationError(
            "serialized dependency E must equal U without optional members"
        )
    expected_delta = sorted(
        set(end["ever_seen_paths"]) - set(start["ever_seen_paths"])
    )
    if (
        value.get("delta_paths") != expected_delta
        or value.get("delta_paths_sha256")
        != _sha256_bytes(_canonical_json_bytes(expected_delta))
        or value.get("ever_seen_paths") != end["ever_seen_paths"]
        or value.get("ever_seen_paths_sha256")
        != _sha256_bytes(_canonical_json_bytes(end["ever_seen_paths"]))
    ):
        raise OneShotAuthorizationError(
            "serialized dependency delta/ever-seen ledger drifted"
        )

    ledger = value.get("import_load_ledger")
    if (
        not isinstance(ledger, Mapping)
        or set(ledger)
        != {
            "events",
            "event_count",
            "event_chain_sha256",
            "events_sha256",
        }
        or not isinstance(ledger.get("events"), list)
        or ledger.get("event_count") != len(ledger["events"])
        or ledger.get("events_sha256")
        != _sha256_bytes(_canonical_json_bytes(ledger["events"]))
    ):
        raise OneShotAuthorizationError(
            "serialized dependency import/load ledger schema drifted"
        )
    previous = _ZERO_SHA256
    loaded_events: set[str] = set()
    prefix_chains: dict[int, str] = {0: _ZERO_SHA256}
    for index, event in enumerate(ledger["events"]):
        if not isinstance(event, Mapping):
            raise OneShotAuthorizationError(
                "serialized dependency ledger event is invalid"
            )
        event_type = event.get("event_type")
        common = {
            "sequence",
            "event_type",
            "phase",
            "previous_event_sha256",
            "event_sha256",
        }
        if event_type == "observed_import_request":
            expected_event_keys = common | {
                "requested_name",
                "fromlist",
                "level",
            }
        elif event_type == "checkpoint_confirmed_successful_load":
            expected_event_keys = common | {
                "canonical_path",
                "fingerprint",
            }
        elif event_type == "registered_member_removed":
            expected_event_keys = common | {"canonical_path"}
        else:
            raise OneShotAuthorizationError(
                "serialized dependency ledger has unknown event type"
            )
        if (
            set(event) != expected_event_keys
            or event.get("sequence") != index
            or event.get("phase") not in _DEPENDENCY_PHASES
            or event.get("previous_event_sha256") != previous
        ):
            raise OneShotAuthorizationError(
                "serialized dependency ledger ordering drifted"
            )
        without_digest = dict(event)
        declared_event = without_digest.pop("event_sha256")
        if declared_event != _sha256_bytes(
            _canonical_json_bytes(without_digest)
        ):
            raise OneShotAuthorizationError(
                "serialized dependency ledger event digest drifted"
            )
        previous = declared_event
        prefix_chains[index + 1] = previous
        if event_type == "checkpoint_confirmed_successful_load":
            path = event.get("canonical_path")
            if (
                path in loaded_events
                or path not in manifest.member_by_path
                or event.get("phase")
                != manifest.member_by_path[path].allowed_phase
                or event.get("fingerprint")
                != _expected_loaded_state_fingerprint(
                    manifest.member_by_path[path]
                )
            ):
                raise OneShotAuthorizationError(
                    "serialized successful-load ledger drifted"
                )
            loaded_events.add(path)
        elif event_type == "registered_member_removed":
            raise OneShotAuthorizationError(
                "a published result cannot contain a registered-member "
                "removal"
            )
    if (
        ledger.get("event_chain_sha256") != previous
        or loaded_events != set(end["ever_seen_paths"])
        or prefix_chains.get(start["event_count"])
        != start["event_chain_sha256"]
        or prefix_chains.get(end["event_count"])
        != end["event_chain_sha256"]
        or end["event_count"] != len(ledger["events"])
    ):
        raise OneShotAuthorizationError(
            "serialized dependency ledger/snapshot cross-binding drifted"
        )
    without_certificate = dict(value)
    declared_certificate = without_certificate.pop(
        "closure_certificate_sha256"
    )
    if declared_certificate != _sha256_bytes(
        _canonical_json_bytes(without_certificate)
    ):
        raise OneShotAuthorizationError(
            "serialized dependency closure certificate drifted"
        )


def _stable_runtime_identity() -> dict[str, Any]:
    executable = Path(sys.executable).absolute()
    executable_metadata = executable.stat()
    proc_executable = Path("/proc/self/exe")
    try:
        mapped_executable_metadata = proc_executable.stat()
        mapped_executable_sha256 = _sha256_bytes(
            proc_executable.read_bytes()
        )
    except OSError as error:
        raise OneShotAuthorizationError(
            "cannot fingerprint the running /proc/self/exe interpreter"
        ) from error
    if (
        executable_metadata.st_dev,
        executable_metadata.st_ino,
    ) != (
        mapped_executable_metadata.st_dev,
        mapped_executable_metadata.st_ino,
    ):
        raise OneShotAuthorizationError(
            "sys.executable no longer names the running interpreter inode"
        )
    numpy_path = Path(np.__file__).absolute()
    yaml_path = Path(yaml.__file__).absolute()
    config_text = io.StringIO()
    with redirect_stdout(config_text):
        np.__config__.show()
    finfo = np.finfo(np.float64)
    repo_metadata = os.stat(REPO_ROOT, follow_symlinks=False)
    platform_metadata = os.stat(PLATFORM, follow_symlinks=False)
    return {
        "sys_executable": str(executable),
        "sys_executable_sha256": _sha256_bytes(executable.read_bytes()),
        "sys_executable_st_dev": int(executable_metadata.st_dev),
        "sys_executable_st_ino": int(executable_metadata.st_ino),
        "mapped_executable_sha256": mapped_executable_sha256,
        "mapped_executable_st_dev": int(
            mapped_executable_metadata.st_dev
        ),
        "mapped_executable_st_ino": int(
            mapped_executable_metadata.st_ino
        ),
        "python_implementation": host_platform.python_implementation(),
        "python_version": sys.version,
        "dont_write_bytecode": sys.dont_write_bytecode,
        "pycache_prefix": sys.pycache_prefix,
        "byteorder": sys.byteorder,
        "float64": {
            "eps_hex": float(finfo.eps).hex(),
            "tiny_hex": float(finfo.tiny).hex(),
            "max_hex": float(finfo.max).hex(),
            "smallest_subnormal_hex": float(
                getattr(finfo, "smallest_subnormal", np.nextafter(0.0, 1.0))
            ).hex(),
        },
        "numpy": {
            "version": np.__version__,
            "path": str(numpy_path),
            "file_sha256": _sha256_bytes(numpy_path.read_bytes()),
            "build_config_sha256": _sha256_bytes(
                config_text.getvalue().encode("utf-8")
            ),
        },
        "pyyaml": {
            "version": yaml.__version__,
            "path": str(yaml_path),
            "file_sha256": _sha256_bytes(yaml_path.read_bytes()),
        },
        "host": {
            "platform": host_platform.platform(),
            "machine": host_platform.machine(),
            "processor": host_platform.processor(),
            "libc": list(host_platform.libc_ver()),
        },
        "workspace": {
            "repo_root": str(REPO_ROOT),
            "repo_root_st_dev": int(repo_metadata.st_dev),
            "repo_root_st_ino": int(repo_metadata.st_ino),
            "platform_root": str(PLATFORM),
            "platform_root_st_dev": int(platform_metadata.st_dev),
            "platform_root_st_ino": int(platform_metadata.st_ino),
        },
        "thread_environment": {
            name: os.environ.get(name)
            for name in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "CUDA_VISIBLE_DEVICES",
                "NPY_DISABLE_CPU_FEATURES",
                "OPENBLAS_CORETYPE",
                "MKL_CBWR",
                "OMP_DYNAMIC",
                "OMP_PROC_BIND",
                "OMP_PLACES",
            )
        },
        "cpu_affinity": (
            sorted(os.sched_getaffinity(0))
            if hasattr(os, "sched_getaffinity")
            else None
        ),
        "floating_point_rounding_mode": int(
            ctypes.CDLL(None).fegetround()
        ),
    }


def _v23_scope_payload() -> dict[str, bool]:
    return {
        "one_31_history_successor_execution_only": True,
        "transport_failure_successor_not_retry": True,
        "claim_state_change_allowed": False,
        "production_activation_allowed": False,
        "force_hp_state_ves_118_fig_allowed": False,
        "malicious_local_writer_or_swap_restore_protection": False,
        "premarker_token_reuse_protection_claimed": False,
    }


def _v23_result_payload(
    profile: NamespaceProfileV1,
) -> dict[str, Any]:
    return {
        "path": _profile_repo_relative(profile, profile.result_path),
        "marker_path": _profile_repo_relative(
            profile, profile.marker_path
        ),
        "overwrite_allowed": False,
        "latest_pointer_write_allowed": False,
        "old_canonical_result_write_allowed": False,
        "atomic_no_replace_required": True,
    }


def _validate_v23_identity_body(
    value: Any,
    *,
    label: str,
) -> dict[str, Any]:
    identity = _require_exact_mapping(
        value, _V23_IDENTITY_BODY_FIELDS, label=label
    )
    for name in (
        "provider",
        "model",
        "model_family",
        "agent_id",
        "trace_id",
    ):
        _require_nonempty_string(
            identity[name], label=f"{label}.{name}"
        )
    identity["source_trace_metadata_path"] = _safe_repo_relative_text(
        identity["source_trace_metadata_path"],
        label=f"{label}.source_trace_metadata_path",
    )
    identity["source_trace_metadata_sha256"] = _validate_sha256(
        identity["source_trace_metadata_sha256"],
        label=f"{label}.source_trace_metadata_sha256",
    )
    return identity


def _v23_definition_chain_projection(
    definition: Mapping[str, Any],
) -> dict[str, Any]:
    frozen = definition.get("frozen_definition_chain")
    if not isinstance(frozen, Mapping):
        raise OneShotAuthorizationError(
            "wrapper definition lacks frozen_definition_chain"
        )
    projected: dict[str, Any] = {}
    for name in (
        "S3ai-v2",
        "S3ai-v2.1",
        "S3ai-v2.2",
        "implementation_audit",
    ):
        entry = _require_exact_mapping(
            frozen.get(name),
            {"file", "sha256"},
            label=f"definition chain {name}",
        )
        _require_nonempty_string(
            entry["file"], label=f"definition chain {name}.file"
        )
        entry["sha256"] = _validate_sha256(
            entry["sha256"],
            label=f"definition chain {name}.sha256",
        )
        projected[name] = entry
    return projected


def _v23_registry_identity() -> tuple[list[str], dict[str, str], str]:
    cases = guard.frozen_history_cases()
    names = [case.name for case in cases]
    if (
        not names
        or len(names) != len(set(names))
        or not all(isinstance(name, str) and name for name in names)
    ):
        raise OneShotAuthorizationError(
            "frozen history registry names drifted"
        )
    identities = {
        case.name: _validate_sha256(
            guard._case_identity_payload(case)["sha256"],
            label=f"case identity {case.name}",
        )
        for case in cases
    }
    manifest = [
        {"name": name, "case_identity_sha256": identities[name]}
        for name in names
    ]
    return (
        names,
        identities,
        _sha256_bytes(_canonical_json_bytes(manifest)),
    )


def _v23_old_quarantine_expected() -> dict[str, Any]:
    expected = {
        _validated_repo_relative(path).as_posix(): digest
        for path, digest in _OLD_QUARANTINE_HASHES.items()
    }
    expected["old_canonical_result_absent"] = True
    expected["old_latest_result_absent"] = True
    return expected


def _validate_v23_old_quarantine(value: Any) -> dict[str, Any]:
    expected = _v23_old_quarantine_expected()
    observed = _require_exact_mapping(
        value, set(expected), label="old_failure_quarantine"
    )
    if observed != expected:
        raise OneShotAuthorizationError(
            "old failure quarantine mapping drifted"
        )
    if _entry_lstat(OLD_CANONICAL_RESULT_PATH) is not None:
        raise OneShotAuthorizationError(
            "retired canonical result must remain absent"
        )
    if _entry_lstat(LATEST_RESULT_PATH) is not None:
        raise OneShotAuthorizationError(
            "retired latest pointer must remain absent"
        )
    for path, digest in _OLD_QUARANTINE_HASHES.items():
        actual = _sha256_bytes(
            _read_regular_nofollow(
                path, label=f"retired asset {path.name}"
            )
        )
        if actual != digest:
            raise OneShotAuthorizationError(
                f"retired failure asset drifted: {path.name}"
            )
    return observed


def _validate_v23_result_scope(
    profile: NamespaceProfileV1,
    result: Any,
    scope: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    exact_result = _require_exact_mapping(
        result, _V23_RESULT_FIELDS, label="ticket result"
    )
    exact_scope = _require_exact_mapping(
        scope, _V23_SCOPE_FIELDS, label="ticket scope"
    )
    if exact_result != _v23_result_payload(profile):
        raise OneShotAuthorizationError(
            "ticket result namespace or no-replace policy drifted"
        )
    if exact_scope != _v23_scope_payload():
        raise OneShotAuthorizationError(
            "ticket scope exceeds the one-shot transport authority"
        )
    return exact_result, exact_scope


def _observe_v23_bound_artifacts(
    profile: NamespaceProfileV1,
    value: Any,
) -> tuple[
    dict[str, Any],
    dict[str, bytes],
    dict[str, str],
]:
    artifacts = _require_exact_mapping(
        value, _V23_BOUND_ARTIFACT_KEYS, label="bound_artifacts"
    )
    normalized: dict[str, Any] = {}
    raw_by_role: dict[str, bytes] = {}
    audit_files: dict[str, str] = {}
    seen_paths: set[str] = set()
    for role in sorted(_V23_BOUND_ARTIFACT_KEYS):
        binding = _require_exact_mapping(
            artifacts[role],
            {"path", "sha256"},
            label=f"bound_artifacts.{role}",
        )
        path = _safe_repo_relative_text(
            binding["path"], label=f"bound_artifacts.{role}.path"
        )
        digest = _validate_sha256(
            binding["sha256"],
            label=f"bound_artifacts.{role}.sha256",
        )
        if path in seen_paths:
            raise OneShotAuthorizationError(
                "two bound artifact roles share one path"
            )
        seen_paths.add(path)
        raw = _read_repo_relative_nofollow_v1(
            profile, path, label=f"bound artifact {role}"
        )
        if _sha256_bytes(raw) != digest:
            raise OneShotAuthorizationError(
                f"bound artifact {role} raw SHA drifted"
            )
        normalized[role] = {"path": path, "sha256": digest}
        raw_by_role[role] = raw
        audit_files[path] = digest
    if normalized["g6_otmpfile_capability_probe"] != {
        "path": _G6_CAPABILITY_PROBE_RELATIVE,
        "sha256": _G6_CAPABILITY_PROBE_SHA256,
    }:
        raise OneShotAuthorizationError(
            "G6 O_TMPFILE capability probe binding drifted"
        )
    if normalized["authorization_schema_preregistration"] != {
        "path": _AUTHORIZATION_SCHEMA_PREREGISTRATION_RELATIVE,
        "sha256": _AUTHORIZATION_SCHEMA_PREREGISTRATION_SHA256,
    }:
        raise OneShotAuthorizationError(
            "authorization schema preregistration binding drifted"
        )
    return normalized, raw_by_role, audit_files


def _validate_v23_identity_chain(
    profile: NamespaceProfileV1,
    *,
    seed: _IdentityCoreBindingSeedV1,
    bound_artifacts: Mapping[str, Any],
    raw_by_role: Mapping[str, bytes],
) -> tuple[dict[str, Any], dict[str, str]]:
    expected_core = {
        "path": seed.path,
        "sha256": seed.observed_raw_sha256,
    }
    if bound_artifacts["implementation_identity_core"] != expected_core:
        raise OneShotAuthorizationError(
            "candidate core binding does not copy the external seed"
        )
    source_map_seed = {
        role: _strict_json_value(bound_artifacts[role])
        for role in sorted(_V23_SOURCE_ARTIFACT_KEYS)
    }
    anchor = _strict_pretty_json_object(
        raw_by_role["implementation_identity"],
        label="implementation identity anchor",
    )
    anchor = _require_exact_mapping(
        anchor,
        {"artifact", "schema_version", "version", "identity"},
        label="implementation identity anchor",
    )
    if (
        anchor["artifact"] != _IMPLEMENTATION_ANCHOR_ARTIFACT
        or anchor["schema_version"] != "1.0"
        or anchor["version"] != _IMPLEMENTATION_ANCHOR_VERSION
    ):
        raise OneShotAuthorizationError(
            "implementation identity anchor fixed identity drifted"
        )
    identity = _validate_v23_identity_body(
        anchor["identity"], label="implementation identity"
    )
    trace_path = identity["source_trace_metadata_path"]
    trace_raw = _read_repo_relative_nofollow_v1(
        profile, trace_path, label="implementation source trace"
    )
    trace_sha = _sha256_bytes(trace_raw)
    if trace_sha != identity["source_trace_metadata_sha256"]:
        raise OneShotAuthorizationError(
            "implementation source trace raw SHA drifted"
        )
    trace = _strict_pretty_json_object(
        trace_raw, label="implementation source trace"
    )
    trace = _require_exact_mapping(
        trace,
        {
            "artifact",
            "schema_version",
            "version",
            "provider",
            "model",
            "model_family",
            "agent_id",
            "trace_id",
            "source_kind",
            "implementation_identity_core_path",
            "implementation_identity_core_raw_sha256",
            "source_artifact_map_sha256",
        },
        label="implementation source trace",
    )
    if (
        trace["artifact"] != _IMPLEMENTATION_TRACE_ARTIFACT
        or trace["schema_version"] != "1.0"
        or trace["version"] != _IMPLEMENTATION_TRACE_VERSION
        or trace["source_kind"]
        != "implementation_and_definition_audit_trace"
        or trace["implementation_identity_core_path"] != seed.path
        or trace["implementation_identity_core_raw_sha256"]
        != seed.observed_raw_sha256
        or trace["source_artifact_map_sha256"]
        != _sha256_bytes(_canonical_json_bytes(source_map_seed))
    ):
        raise OneShotAuthorizationError(
            "implementation source trace binding drifted"
        )
    core = seed.parsed_exact_core
    for name in (
        "provider",
        "model",
        "model_family",
        "agent_id",
        "trace_id",
    ):
        if trace[name] != core[name] or identity[name] != core[name]:
            raise OneShotAuthorizationError(
                "implementation identity content chain drifted"
            )
    return identity, {trace_path: trace_sha}


def _validate_v23_execution_sources(
    source_fingerprints: Mapping[str, str],
) -> dict[str, str]:
    if not isinstance(source_fingerprints, Mapping) or not source_fingerprints:
        raise OneShotAuthorizationError(
            "execution source map must be a nonempty exact mapping"
        )
    normalized: dict[str, str] = {}
    for path, digest in source_fingerprints.items():
        safe = _safe_repo_relative_text(
            path, label="execution source path"
        )
        normalized[safe] = _validate_sha256(
            digest, label=f"execution source {safe}"
        )
    guard_path = (
        "platform/"
        "actual_wake_reachable_pressure_obstruction_v2_guard.py"
    )
    guard_source = _source_path_from_module(guard)
    if (
        guard_source
        != REPO_ROOT
        / "platform"
        / "actual_wake_reachable_pressure_obstruction_v2_guard.py"
        or guard_path not in normalized
        or normalized[guard_path]
        != _sha256_bytes(
            _read_regular_nofollow(
                REPO_ROOT / guard_path, label="bound guard source"
            )
        )
    ):
        raise OneShotAuthorizationError(
            "preloaded guard module/source binding drifted"
        )
    return normalized


def _v23_bound_manifest_projection(
    *,
    profile: NamespaceProfileV1,
    bound_artifacts: Mapping[str, Any],
    raw_by_role: Mapping[str, bytes],
    definition_chain: Mapping[str, Any],
    source_fingerprints: Mapping[str, str],
    stable_runtime_identity: Mapping[str, Any],
    dependency_manifest: DependencyManifest,
    old_failure_quarantine: Mapping[str, Any],
    registry_sha: str,
) -> dict[str, str]:
    if dependency_manifest.raw_sha256 is None:
        raise OneShotAuthorizationError(
            "dependency manifest lacks its externally bound raw SHA"
        )
    if (
        bound_artifacts["dependency_manifest"]["sha256"]
        != dependency_manifest.raw_sha256
    ):
        raise OneShotAuthorizationError(
            "dependency manifest raw binding drifted"
        )
    consensus = _strict_json_object_any_raw(
        raw_by_role["g3_consensus_certificate"],
        label="G3 consensus certificate",
    )
    bootstrap = _strict_json_object_any_raw(
        raw_by_role["bootstrap_contract"],
        label="bootstrap contract",
    )
    return {
        "bound_artifact_map_sha256": _sha256_bytes(
            _canonical_json_bytes(bound_artifacts)
        ),
        "execution_source_map_sha256": _sha256_bytes(
            _canonical_json_bytes(source_fingerprints)
        ),
        "stable_runtime_identity_sha256": _sha256_bytes(
            _canonical_json_bytes(stable_runtime_identity)
        ),
        "ordered_registry_manifest_sha256": registry_sha,
        "frozen_definition_chain_sha256": _sha256_bytes(
            _canonical_json_bytes(definition_chain)
        ),
        "transport_preregistration_raw_sha256": bound_artifacts[
            "transport_preregistration"
        ]["sha256"],
        "dependency_capture_protocol_raw_sha256": bound_artifacts[
            "dependency_capture_protocol"
        ]["sha256"],
        "authorization_schema_preregistration_raw_sha256": (
            bound_artifacts["authorization_schema_preregistration"][
                "sha256"
            ]
        ),
        "dependency_manifest_raw_sha256": dependency_manifest.raw_sha256,
        "dependency_manifest_canonical_sha256": (
            dependency_manifest.canonical_sha256
        ),
        "dependency_B_paths_sha256": _sha256_bytes(
            _canonical_json_bytes(list(dependency_manifest.B))
        ),
        "dependency_U_paths_sha256": _sha256_bytes(
            _canonical_json_bytes(list(dependency_manifest.U))
        ),
        "dependency_R_paths_sha256": _sha256_bytes(
            _canonical_json_bytes(list(dependency_manifest.R))
        ),
        "g3_consensus_canonical_sha256": _sha256_bytes(
            _canonical_json_bytes(consensus)
        ),
        "bootstrap_contract_canonical_sha256": _sha256_bytes(
            _canonical_json_bytes(bootstrap)
        ),
        "old_failure_quarantine_canonical_sha256": _sha256_bytes(
            _canonical_json_bytes(old_failure_quarantine)
        ),
    }


def _validate_v23_authorization_plan(
    value: Any,
    *,
    label: str,
) -> dict[str, Any]:
    plan = _require_exact_mapping(
        value, _V23_AUTHORIZATION_PLAN_FIELDS, label=label
    )
    authorization_id = _require_nonempty_string(
        plan["id"], label=f"{label}.id"
    )
    if authorization_id == _OLD_AUTHORIZATION_ID:
        raise OneShotAuthorizationError(
            "retired authorization id cannot enter v2.3"
        )
    _require_exact_bool(
        plan["external_raw_sha256_required"],
        True,
        label=f"{label}.external_raw_sha256_required",
    )
    token_sha = _validate_sha256(
        plan["single_use_token_sha256"],
        label=f"{label}.single_use_token_sha256",
    )
    if token_sha == _OLD_TOKEN_SHA256:
        raise OneShotAuthorizationError(
            "retired token commitment cannot enter v2.3"
        )
    _require_exact_bool(
        plan["formal_execution_allowed"],
        True,
        label=f"{label}.formal_execution_allowed",
    )
    _require_exact_int(
        plan["execution_limit"],
        1,
        label=f"{label}.execution_limit",
    )
    if plan["decision"] != _AUTHORIZATION_DECISION:
        raise OneShotAuthorizationError(
            f"{label}.decision drifted"
        )
    _require_exact_bool(
        plan["post_marker_retry_allowed"],
        False,
        label=f"{label}.post_marker_retry_allowed",
    )
    return plan


def _validate_v23_projection(
    value: Any,
    *,
    label: str,
) -> dict[str, Any]:
    projection = _require_exact_mapping(
        value, _V23_PROJECTION_FIELDS, label=label
    )
    if (
        projection["schema_version"] != "1.0"
        or projection["namespace_profile"]
        not in {_PRODUCTION_NAMESPACE_MODE, _FIXTURE_NAMESPACE_MODE}
        or projection["ticket_artifact"] != _AUTHORIZATION_ARTIFACT
        or projection["ticket_version"] != _AUTHORIZATION_VERSION
        or projection["attempt_id"] != _ATTEMPT_ID
        or projection["scientific_protocol_version"] != "S3ai-v2.2"
        or projection["transport_protocol_version"] != "S3ai-v2.3"
        or projection["assurance_profile"] != _ASSURANCE_PROFILE
    ):
        raise OneShotAuthorizationError(
            f"{label} fixed identity drifted"
        )
    projection["authorization_plan"] = (
        _validate_v23_authorization_plan(
            projection["authorization_plan"],
            label=f"{label}.authorization_plan",
        )
    )
    projection["bound_artifacts"] = _require_exact_mapping(
        projection["bound_artifacts"],
        _V23_BOUND_ARTIFACT_KEYS,
        label=f"{label}.bound_artifacts",
    )
    for role, binding in projection["bound_artifacts"].items():
        exact = _require_exact_mapping(
            binding,
            {"path", "sha256"},
            label=f"{label}.bound_artifacts.{role}",
        )
        exact["path"] = _safe_repo_relative_text(
            exact["path"],
            label=f"{label}.bound_artifacts.{role}.path",
        )
        exact["sha256"] = _validate_sha256(
            exact["sha256"],
            label=f"{label}.bound_artifacts.{role}.sha256",
        )
        projection["bound_artifacts"][role] = exact
    projection["implementation_identity"] = (
        _validate_v23_identity_body(
            projection["implementation_identity"],
            label=f"{label}.implementation_identity",
        )
    )
    projection["definition_chain"] = _require_exact_mapping(
        projection["definition_chain"],
        {"S3ai-v2", "S3ai-v2.1", "S3ai-v2.2", "implementation_audit"},
        label=f"{label}.definition_chain",
    )
    if not isinstance(projection["execution_sources"], Mapping):
        raise OneShotAuthorizationError(
            f"{label}.execution_sources must be a mapping"
        )
    for path, digest in projection["execution_sources"].items():
        _safe_repo_relative_text(
            path, label=f"{label}.execution_sources path"
        )
        _validate_sha256(
            digest, label=f"{label}.execution_sources digest"
        )
    projection["stable_runtime_identity"] = _strict_json_value(
        projection["stable_runtime_identity"],
        label=f"{label}.stable_runtime_identity",
    )
    manifests = _require_exact_mapping(
        projection["bound_manifests"],
        _V23_BOUND_MANIFEST_FIELDS,
        label=f"{label}.bound_manifests",
    )
    for name, digest in manifests.items():
        manifests[name] = _validate_sha256(
            digest, label=f"{label}.bound_manifests.{name}"
        )
    result = _require_exact_mapping(
        projection["result"],
        _V23_RESULT_FIELDS,
        label=f"{label}.result",
    )
    result["path"] = _safe_repo_relative_text(
        result["path"], label=f"{label}.result.path"
    )
    result["marker_path"] = _safe_repo_relative_text(
        result["marker_path"], label=f"{label}.result.marker_path"
    )
    for name, expected in (
        ("overwrite_allowed", False),
        ("latest_pointer_write_allowed", False),
        ("old_canonical_result_write_allowed", False),
        ("atomic_no_replace_required", True),
    ):
        _require_exact_bool(
            result[name], expected, label=f"{label}.result.{name}"
        )
    if projection["scope"] != _v23_scope_payload():
        raise OneShotAuthorizationError(
            f"{label}.scope drifted"
        )
    if (
        projection["old_failure_quarantine"]
        != _v23_old_quarantine_expected()
    ):
        raise OneShotAuthorizationError(
            f"{label}.old_failure_quarantine drifted"
        )
    names = _require_unique_array(
        projection["ordered_case_names"],
        label=f"{label}.ordered_case_names",
        nonempty=True,
    )
    if not all(isinstance(name, str) and name for name in names):
        raise OneShotAuthorizationError(
            f"{label}.ordered_case_names contains a non-name"
        )
    identities = _require_exact_mapping(
        projection["case_identity_sha256"],
        set(names),
        label=f"{label}.case_identity_sha256",
    )
    for name in names:
        identities[name] = _validate_sha256(
            identities[name], label=f"{label}.case identity {name}"
        )
    projection["ordered_registry_manifest_sha256"] = _validate_sha256(
        projection["ordered_registry_manifest_sha256"],
        label=f"{label}.ordered_registry_manifest_sha256",
    )
    return projection


def _build_projection_v1(
    plan: _FrozenTicketSemanticPlanInputsV1,
) -> dict[str, Any]:
    if type(plan) is not _FrozenTicketSemanticPlanInputsV1:
        raise OneShotAuthorizationError(
            "BuildProjectionV1 requires the exact frozen plan input type"
        )
    profile = plan.namespace_profile
    _validate_namespace_profile_v1(profile)
    authorization_id = _require_nonempty_string(
        plan.authorization_id, label="planned authorization id"
    )
    if authorization_id == _OLD_AUTHORIZATION_ID:
        raise OneShotAuthorizationError(
            "retired authorization id cannot be planned"
        )
    token_sha = _validate_sha256(
        plan.single_use_token_sha256,
        label="planned single-use token SHA256",
    )
    if token_sha == _OLD_TOKEN_SHA256:
        raise OneShotAuthorizationError(
            "retired token commitment cannot be planned"
        )
    artifacts, raw_by_role, _ = _observe_v23_bound_artifacts(
        profile, plan.bound_artifacts
    )
    sources = _validate_v23_execution_sources(
        plan.source_fingerprints
    )
    if (
        artifacts["wrapper_source"]["path"] != _WRAPPER_RELATIVE
        or artifacts["wrapper_source"]["sha256"]
        != sources.get(_WRAPPER_RELATIVE)
    ):
        raise OneShotAuthorizationError(
            "wrapper source artifact and execution source map differ"
        )
    parsed_definition = _parse_yaml_mapping(
        raw_by_role["wrapper_definition"],
        label="bound wrapper definition",
    )
    normalized_definition = _strict_json_value(
        plan.definition, label="frozen wrapper definition"
    )
    if parsed_definition != normalized_definition:
        raise OneShotAuthorizationError(
            "bound wrapper definition differs from the plan input"
        )
    definition_chain = _v23_definition_chain_projection(
        normalized_definition
    )
    identity = _validate_v23_identity_body(
        plan.implementation_identity,
        label="planned implementation identity",
    )
    anchor = _strict_pretty_json_object(
        raw_by_role["implementation_identity"],
        label="bound implementation identity anchor",
    )
    anchor = _require_exact_mapping(
        anchor,
        {"artifact", "schema_version", "version", "identity"},
        label="bound implementation identity anchor",
    )
    if (
        anchor["artifact"] != _IMPLEMENTATION_ANCHOR_ARTIFACT
        or anchor["schema_version"] != "1.0"
        or anchor["version"] != _IMPLEMENTATION_ANCHOR_VERSION
        or _validate_v23_identity_body(
            anchor["identity"], label="bound anchor identity"
        )
        != identity
    ):
        raise OneShotAuthorizationError(
            "planned implementation identity differs from its anchor"
        )
    runtime = _strict_json_value(
        plan.stable_runtime_identity,
        label="planned stable runtime identity",
    )
    quarantine = _validate_v23_old_quarantine(
        plan.old_failure_quarantine
    )
    names, identities, registry_sha = _v23_registry_identity()
    manifests = _v23_bound_manifest_projection(
        profile=profile,
        bound_artifacts=artifacts,
        raw_by_role=raw_by_role,
        definition_chain=definition_chain,
        source_fingerprints=sources,
        stable_runtime_identity=runtime,
        dependency_manifest=plan.dependency_manifest,
        old_failure_quarantine=quarantine,
        registry_sha=registry_sha,
    )
    projection = {
        "schema_version": "1.0",
        "namespace_profile": profile.mode,
        "ticket_artifact": _AUTHORIZATION_ARTIFACT,
        "ticket_version": _AUTHORIZATION_VERSION,
        "attempt_id": _ATTEMPT_ID,
        "scientific_protocol_version": "S3ai-v2.2",
        "transport_protocol_version": "S3ai-v2.3",
        "assurance_profile": _ASSURANCE_PROFILE,
        "authorization_plan": {
            "id": authorization_id,
            "external_raw_sha256_required": True,
            "single_use_token_sha256": token_sha,
            "formal_execution_allowed": True,
            "execution_limit": 1,
            "decision": _AUTHORIZATION_DECISION,
            "post_marker_retry_allowed": False,
        },
        "bound_artifacts": _strict_json_value(artifacts),
        "implementation_identity": _strict_json_value(identity),
        "definition_chain": _strict_json_value(definition_chain),
        "execution_sources": _strict_json_value(sources),
        "stable_runtime_identity": _strict_json_value(runtime),
        "bound_manifests": manifests,
        "result": _v23_result_payload(profile),
        "scope": _v23_scope_payload(),
        "old_failure_quarantine": _strict_json_value(quarantine),
        "ordered_case_names": names,
        "case_identity_sha256": identities,
        "ordered_registry_manifest_sha256": registry_sha,
    }
    return _validate_v23_projection(
        projection, label="BuildProjectionV1 output"
    )


def _validate_v23_ticket_static(value: Any) -> dict[str, Any]:
    ticket = _require_exact_mapping(
        value, _V23_TICKET_FIELDS, label="authorization ticket"
    )
    mode = ticket["namespace_profile"]
    expected_status = {
        _PRODUCTION_NAMESPACE_MODE: (
            "accepted_after_g5_independent_bounded_audit"
        ),
        _FIXTURE_NAMESPACE_MODE: (
            "synthetic_definition_fixture_no_production_authority"
        ),
    }.get(mode)
    if (
        ticket["artifact"] != _AUTHORIZATION_ARTIFACT
        or ticket["schema_version"] != "1.0"
        or ticket["version"] != _AUTHORIZATION_VERSION
        or expected_status is None
        or ticket["status"] != expected_status
        or ticket["attempt_id"] != _ATTEMPT_ID
        or ticket["scientific_protocol_version"] != "S3ai-v2.2"
        or ticket["transport_protocol_version"] != "S3ai-v2.3"
        or ticket["assurance_profile"] != _ASSURANCE_PROFILE
    ):
        raise OneShotAuthorizationError(
            "authorization ticket fixed identity drifted"
        )
    authorization = _require_exact_mapping(
        ticket["authorization"],
        _V23_AUTHORIZATION_FIELDS,
        label="ticket authorization",
    )
    authorization_plan = {
        name: authorization[name]
        for name in _V23_AUTHORIZATION_PLAN_FIELDS
    }
    _validate_v23_authorization_plan(
        authorization_plan, label="ticket authorization plan"
    )
    authorization["canonical_sha256"] = _validate_sha256(
        authorization["canonical_sha256"],
        label="ticket authorization canonical SHA256",
    )
    ticket["authorization"] = authorization
    issuance = _require_exact_mapping(
        ticket["issuance_request"],
        {
            "path",
            "raw_sha256",
            "canonical_sha256",
            "ticket_semantic_projection_sha256",
        },
        label="ticket issuance_request",
    )
    issuance["path"] = _safe_repo_relative_text(
        issuance["path"], label="ticket issuance request path"
    )
    for name in (
        "raw_sha256",
        "canonical_sha256",
        "ticket_semantic_projection_sha256",
    ):
        issuance[name] = _validate_sha256(
            issuance[name], label=f"ticket issuance_request.{name}"
        )
    ticket["bound_artifacts"] = _require_exact_mapping(
        ticket["bound_artifacts"],
        _V23_BOUND_ARTIFACT_KEYS,
        label="ticket bound_artifacts",
    )
    for role, binding in ticket["bound_artifacts"].items():
        exact = _require_exact_mapping(
            binding,
            {"path", "sha256"},
            label=f"ticket bound_artifacts.{role}",
        )
        exact["path"] = _safe_repo_relative_text(
            exact["path"],
            label=f"ticket bound_artifacts.{role}.path",
        )
        exact["sha256"] = _validate_sha256(
            exact["sha256"],
            label=f"ticket bound_artifacts.{role}.sha256",
        )
        ticket["bound_artifacts"][role] = exact
    ticket["implementation_identity"] = _validate_v23_identity_body(
        ticket["implementation_identity"],
        label="ticket implementation_identity",
    )
    ticket["bound_manifests"] = _require_exact_mapping(
        ticket["bound_manifests"],
        _V23_BOUND_MANIFEST_FIELDS,
        label="ticket bound_manifests",
    )
    for name in _V23_BOUND_MANIFEST_FIELDS:
        ticket["bound_manifests"][name] = _validate_sha256(
            ticket["bound_manifests"][name],
            label=f"ticket bound_manifests.{name}",
        )
    second = _require_exact_mapping(
        ticket["second_bounded_audit"],
        _V23_SECOND_AUDIT_FIELDS,
        label="ticket second_bounded_audit",
    )
    for name in _V23_SECOND_AUDIT_FIELDS:
        if name.endswith("_path"):
            second[name] = _safe_repo_relative_text(
                second[name],
                label=f"ticket second_bounded_audit.{name}",
            )
        else:
            second[name] = _validate_sha256(
                second[name],
                label=f"ticket second_bounded_audit.{name}",
            )
    if (
        second["response_canonical_sha256"]
        != second["clearance_canonical_sha256"]
    ):
        raise OneShotAuthorizationError(
            "clearance and response canonical digests differ"
        )
    result = _require_exact_mapping(
        ticket["result"], _V23_RESULT_FIELDS, label="ticket result"
    )
    result["path"] = _safe_repo_relative_text(
        result["path"], label="ticket result path"
    )
    result["marker_path"] = _safe_repo_relative_text(
        result["marker_path"], label="ticket marker path"
    )
    for name, expected in (
        ("overwrite_allowed", False),
        ("latest_pointer_write_allowed", False),
        ("old_canonical_result_write_allowed", False),
        ("atomic_no_replace_required", True),
    ):
        _require_exact_bool(
            result[name], expected, label=f"ticket result.{name}"
        )
    if ticket["scope"] != _v23_scope_payload():
        raise OneShotAuthorizationError("ticket scope drifted")
    _require_exact_mapping(
        ticket["old_failure_quarantine"],
        set(_v23_old_quarantine_expected()),
        label="ticket old_failure_quarantine",
    )
    names = _require_unique_array(
        ticket["ordered_case_names"],
        label="ticket ordered_case_names",
        nonempty=True,
    )
    if not all(isinstance(name, str) and name for name in names):
        raise OneShotAuthorizationError(
            "ticket ordered_case_names contains a non-name"
        )
    identities = _require_exact_mapping(
        ticket["case_identity_sha256"],
        set(names),
        label="ticket case_identity_sha256",
    )
    for name in names:
        identities[name] = _validate_sha256(
            identities[name], label=f"ticket case identity {name}"
        )
    ticket["ordered_registry_manifest_sha256"] = _validate_sha256(
        ticket["ordered_registry_manifest_sha256"],
        label="ticket ordered registry manifest SHA256",
    )
    ticket["definition_chain"] = _strict_json_value(
        ticket["definition_chain"], label="ticket definition_chain"
    )
    ticket["execution_sources"] = _strict_json_value(
        ticket["execution_sources"], label="ticket execution_sources"
    )
    ticket["stable_runtime_identity"] = _strict_json_value(
        ticket["stable_runtime_identity"],
        label="ticket stable_runtime_identity",
    )
    return ticket


def _ticket_semantic_projection_v1(
    ticket: Mapping[str, Any],
) -> dict[str, Any]:
    exact = _validate_v23_ticket_static(ticket)
    authorization = exact["authorization"]
    projection = {
        "schema_version": exact["schema_version"],
        "namespace_profile": exact["namespace_profile"],
        "ticket_artifact": exact["artifact"],
        "ticket_version": exact["version"],
        "attempt_id": exact["attempt_id"],
        "scientific_protocol_version": (
            exact["scientific_protocol_version"]
        ),
        "transport_protocol_version": (
            exact["transport_protocol_version"]
        ),
        "assurance_profile": exact["assurance_profile"],
        "authorization_plan": {
            name: _strict_json_value(authorization[name])
            for name in _V23_AUTHORIZATION_PLAN_FIELDS
        },
        "bound_artifacts": _strict_json_value(
            exact["bound_artifacts"]
        ),
        "implementation_identity": _strict_json_value(
            exact["implementation_identity"]
        ),
        "definition_chain": _strict_json_value(
            exact["definition_chain"]
        ),
        "execution_sources": _strict_json_value(
            exact["execution_sources"]
        ),
        "stable_runtime_identity": _strict_json_value(
            exact["stable_runtime_identity"]
        ),
        "bound_manifests": _strict_json_value(
            exact["bound_manifests"]
        ),
        "result": _strict_json_value(exact["result"]),
        "scope": _strict_json_value(exact["scope"]),
        "old_failure_quarantine": _strict_json_value(
            exact["old_failure_quarantine"]
        ),
        "ordered_case_names": _strict_json_value(
            exact["ordered_case_names"]
        ),
        "case_identity_sha256": _strict_json_value(
            exact["case_identity_sha256"]
        ),
        "ordered_registry_manifest_sha256": exact[
            "ordered_registry_manifest_sha256"
        ],
    }
    return _validate_v23_projection(
        projection, label="TicketSemanticProjectionV1"
    )


def _canonical_authorization_digest(payload: Mapping[str, Any]) -> str:
    normalized = _strict_json_value(payload, label="authorization")
    authorization = normalized.get("authorization")
    if not isinstance(authorization, dict):
        raise OneShotAuthorizationError(
            "authorization document lacks authorization mapping"
        )
    authorization["canonical_sha256"] = _ZERO_SHA256
    return _sha256_bytes(_canonical_json_bytes(normalized))


def _read_pretty_child_v1(
    profile: NamespaceProfileV1,
    path: str,
    *,
    label: str,
    expected_raw_sha256: str | None = None,
    expected_canonical_sha256: str | None = None,
) -> tuple[dict[str, Any], bytes, str, str]:
    safe = _safe_repo_relative_text(path, label=f"{label} path")
    raw = _read_repo_relative_nofollow_v1(
        profile, safe, label=label
    )
    raw_sha = _sha256_bytes(raw)
    if (
        expected_raw_sha256 is not None
        and raw_sha
        != _validate_sha256(
            expected_raw_sha256, label=f"{label} expected raw SHA256"
        )
    ):
        raise OneShotAuthorizationError(
            f"{label} raw SHA256 drifted"
        )
    payload = _strict_pretty_json_object(raw, label=label)
    canonical_sha = _sha256_bytes(_canonical_json_bytes(payload))
    if (
        expected_canonical_sha256 is not None
        and canonical_sha
        != _validate_sha256(
            expected_canonical_sha256,
            label=f"{label} expected canonical SHA256",
        )
    ):
        raise OneShotAuthorizationError(
            f"{label} canonical SHA256 drifted"
        )
    return payload, raw, raw_sha, canonical_sha


def _validate_v23_q(value: Any) -> dict[str, Any]:
    q = _require_exact_mapping(value, _V23_Q_FIELDS, label="issuance Q")
    if (
        q["artifact"]
        != "actual_wake_reachable_pressure_v23_issuance_request"
        or q["schema_version"] != "1.0"
        or q["version"] != "S3ai-v2.3-issuance-request-v2"
        or q["status"]
        != "REQUEST_INDEPENDENT_REVIEW_NO_TICKET_EXISTS"
        or q["attempt_id"] != _ATTEMPT_ID
    ):
        raise OneShotAuthorizationError(
            "issuance Q fixed identity drifted"
        )
    for name in (
        "planned_authorization_path",
        "planned_result_path",
        "planned_marker_path",
    ):
        q[name] = _safe_repo_relative_text(
            q[name], label=f"issuance Q.{name}"
        )
    q["single_use_token_sha256"] = _validate_sha256(
        q["single_use_token_sha256"],
        label="issuance Q token commitment",
    )
    q["ticket_semantic_projection"] = _validate_v23_projection(
        q["ticket_semantic_projection"],
        label="issuance Q projection",
    )
    q["ticket_semantic_projection_sha256"] = _validate_sha256(
        q["ticket_semantic_projection_sha256"],
        label="issuance Q projection SHA256",
    )
    if q["ticket_semantic_projection_sha256"] != _sha256_bytes(
        _canonical_json_bytes(q["ticket_semantic_projection"])
    ):
        raise OneShotAuthorizationError(
            "issuance Q projection digest drifted"
        )
    q["implementation_identity"] = _validate_v23_identity_body(
        q["implementation_identity"],
        label="issuance Q implementation identity",
    )
    invocation = _require_exact_mapping(
        q["planned_invocation"],
        {"path", "artifact", "schema_version", "version", "call_prefix"},
        label="issuance Q planned invocation",
    )
    invocation["path"] = _safe_repo_relative_text(
        invocation["path"], label="issuance Q planned invocation path"
    )
    if (
        invocation["artifact"]
        != "actual_wake_reachable_pressure_v23_review_invocation"
        or invocation["schema_version"] != "1.0"
        or invocation["version"] != "S3ai-v2.3-review-invocation-v1"
        or not isinstance(invocation["call_prefix"], str)
        or not _TRACE_ID_RE.fullmatch(invocation["call_prefix"])
    ):
        raise OneShotAuthorizationError(
            "issuance Q planned invocation identity drifted"
        )
    requirements = _require_exact_mapping(
        q["review_requirements"],
        {
            "acceptance_prefilled",
            "rejection_allowed",
            "required_independence",
            "blocking_findings_must_be_empty_for_accept",
        },
        label="issuance Q review requirements",
    )
    if requirements != {
        "acceptance_prefilled": False,
        "rejection_allowed": True,
        "required_independence": "genuine-cross-family",
        "blocking_findings_must_be_empty_for_accept": True,
    }:
        raise OneShotAuthorizationError(
            "issuance Q review requirements drifted"
        )
    return q


def _validate_v23_m(value: Any) -> dict[str, Any]:
    m = _require_exact_mapping(
        value, _V23_M_FIELDS, label="invocation M"
    )
    if (
        m["artifact"]
        != "actual_wake_reachable_pressure_v23_review_invocation"
        or m["schema_version"] != "1.0"
        or m["version"] != "S3ai-v2.3-review-invocation-v1"
        or not isinstance(m["trace_id"], str)
        or not _TRACE_ID_RE.fullmatch(m["trace_id"])
        or not isinstance(m["call_prefix"], str)
        or not _TRACE_ID_RE.fullmatch(m["call_prefix"])
        or not isinstance(m["created_at_utc"], str)
        or not _UTC_TIMESTAMP_RE.fullmatch(m["created_at_utc"])
    ):
        raise OneShotAuthorizationError(
            "invocation M fixed identity or time grammar drifted"
        )
    m["allowed_trace_root"] = _safe_repo_relative_text(
        m["allowed_trace_root"], label="invocation M allowed trace root"
    )
    m["implementation_identity"] = _validate_v23_identity_body(
        m["implementation_identity"],
        label="invocation M implementation identity",
    )
    for name in (
        "requested_reviewer_provider",
        "requested_reviewer_model",
        "requested_reviewer_model_family",
    ):
        _require_nonempty_string(m[name], label=f"invocation M.{name}")
    m["issuance_request_path"] = _safe_repo_relative_text(
        m["issuance_request_path"],
        label="invocation M issuance request path",
    )
    for name in (
        "issuance_request_raw_sha256",
        "issuance_request_canonical_sha256",
        "ticket_semantic_projection_sha256",
    ):
        m[name] = _validate_sha256(
            m[name], label=f"invocation M.{name}"
        )
    planned = _require_exact_mapping(
        m["planned_review_request"],
        {"path", "artifact", "schema_version", "version", "call_prefix"},
        label="invocation M planned review request",
    )
    planned["path"] = _safe_repo_relative_text(
        planned["path"],
        label="invocation M planned review request path",
    )
    if (
        planned["artifact"]
        != "actual_wake_reachable_pressure_v23_review_request"
        or planned["schema_version"] != "1.0"
        or planned["version"] != "S3ai-v2.3-review-request-v1"
        or planned["call_prefix"] != m["call_prefix"]
    ):
        raise OneShotAuthorizationError(
            "invocation M planned request identity drifted"
        )
    return m


def _validate_v23_prior_binding(
    value: Any,
    *,
    label: str,
    include_projection: bool,
) -> dict[str, Any]:
    fields = {"path", "raw_sha256", "canonical_sha256"}
    if include_projection:
        fields.add("ticket_semantic_projection_sha256")
    binding = _require_exact_mapping(value, fields, label=label)
    binding["path"] = _safe_repo_relative_text(
        binding["path"], label=f"{label}.path"
    )
    for name in fields - {"path"}:
        binding[name] = _validate_sha256(
            binding[name], label=f"{label}.{name}"
        )
    return binding


def _validate_v23_a(value: Any) -> dict[str, Any]:
    a = _require_exact_mapping(
        value, _V23_A_FIELDS, label="review request A"
    )
    if (
        a["artifact"]
        != "actual_wake_reachable_pressure_v23_review_request"
        or a["schema_version"] != "1.0"
        or a["version"] != "S3ai-v2.3-review-request-v1"
        or a["status"]
        != "REQUEST_NEUTRAL_INDEPENDENT_ACCEPT_OR_REJECT"
        or a["attempt_id"] != _ATTEMPT_ID
        or not isinstance(a["trace_id"], str)
        or not _TRACE_ID_RE.fullmatch(a["trace_id"])
        or not isinstance(a["call_prefix"], str)
        or not _TRACE_ID_RE.fullmatch(a["call_prefix"])
    ):
        raise OneShotAuthorizationError(
            "review request A fixed identity drifted"
        )
    a["issuance_request"] = _validate_v23_prior_binding(
        a["issuance_request"],
        label="review request A issuance binding",
        include_projection=True,
    )
    a["invocation_metadata"] = _validate_v23_prior_binding(
        a["invocation_metadata"],
        label="review request A invocation binding",
        include_projection=False,
    )
    a["implementation_identity"] = _validate_v23_identity_body(
        a["implementation_identity"],
        label="review request A implementation identity",
    )
    target = _require_exact_mapping(
        a["review_target"],
        {"provider", "model", "model_family", "required_independence"},
        label="review request A target",
    )
    for name in ("provider", "model", "model_family"):
        _require_nonempty_string(
            target[name], label=f"review request A target.{name}"
        )
    if target["required_independence"] != "genuine-cross-family":
        raise OneShotAuthorizationError(
            "review request A independence requirement drifted"
        )
    checks = _require_unique_array(
        a["required_checks"],
        label="review request A required checks",
        nonempty=True,
    )
    if checks != list(_V23_REQUIRED_CHECKS):
        raise OneShotAuthorizationError(
            "review request A required checks drifted"
        )
    response = _require_exact_mapping(
        a["response_contract"],
        {
            "accepted_artifact",
            "accepted_version",
            "rejected_artifact",
            "rejected_version",
            "response_path",
            "post_review_trace_path",
        },
        label="review request A response contract",
    )
    if (
        response["accepted_artifact"]
        != "S3ai-v2.3-successor-one-shot-execution-clearance"
        or response["accepted_version"] != 2
        or response["rejected_artifact"]
        != "S3ai-v2.3-successor-one-shot-execution-rejection"
        or response["rejected_version"] != 1
    ):
        raise OneShotAuthorizationError(
            "review request A response contract identity drifted"
        )
    response["response_path"] = _safe_repo_relative_text(
        response["response_path"],
        label="review request A response path",
    )
    response["post_review_trace_path"] = _safe_repo_relative_text(
        response["post_review_trace_path"],
        label="review request A post-review trace path",
    )
    return a


def _validate_v23_c(
    value: Any,
    *,
    accepted: bool,
) -> dict[str, Any]:
    c = _require_exact_mapping(
        value, _V23_C_FIELDS, label="review response C"
    )
    expected_artifact = (
        "S3ai-v2.3-successor-one-shot-execution-clearance"
        if accepted
        else "S3ai-v2.3-successor-one-shot-execution-rejection"
    )
    expected_version = 2 if accepted else 1
    expected_verdict = (
        "ACCEPT_EXACT_V23_SUCCESSOR_ONE_SHOT_31_HISTORY_EXECUTION"
        if accepted
        else "REJECT_V23_SUCCESSOR_ONE_SHOT_EXECUTION"
    )
    if (
        c["artifact"] != expected_artifact
        or c["schema_version"] != "1.0"
        or c["version"] != expected_version
        or c["attempt_id"] != _ATTEMPT_ID
        or c["verdict"] != expected_verdict
    ):
        raise OneShotAuthorizationError(
            "review response C fixed identity drifted"
        )
    review = _require_exact_mapping(
        c["review"], _V23_C_REVIEW_FIELDS, label="clearance C review"
    )
    if review["actual_independence"] != "genuine-cross-family":
        raise OneShotAuthorizationError(
            "clearance C independence label drifted"
        )
    for name in (
        "provider",
        "model",
        "model_family",
        "agent_id",
        "trace_id",
        "implementation_model_family",
    ):
        _require_nonempty_string(
            review[name], label=f"clearance C review.{name}"
        )
    review["invocation_metadata_path"] = _safe_repo_relative_text(
        review["invocation_metadata_path"],
        label="clearance C invocation path",
    )
    for name in _V23_C_REVIEW_FIELDS - {
        "actual_independence",
        "provider",
        "model",
        "model_family",
        "agent_id",
        "trace_id",
        "implementation_model_family",
        "invocation_metadata_path",
    }:
        review[name] = _validate_sha256(
            review[name], label=f"clearance C review.{name}"
        )
    bindings = _require_exact_mapping(
        c["bindings"],
        _V23_C_BINDING_FIELDS,
        label="clearance C bindings",
    )
    for name in _V23_C_BINDING_FIELDS:
        bindings[name] = _validate_sha256(
            bindings[name], label=f"clearance C bindings.{name}"
        )
    grant = _require_exact_mapping(
        c["grant"], _V23_C_GRANT_FIELDS, label="clearance C grant"
    )
    _require_nonempty_string(
        grant["authorization_id"],
        label="clearance C grant.authorization_id",
    )
    if grant["decision"] != _AUTHORIZATION_DECISION:
        raise OneShotAuthorizationError(
            "clearance C grant decision drifted"
        )
    grant["single_use_token_sha256"] = _validate_sha256(
        grant["single_use_token_sha256"],
        label="clearance C grant token commitment",
    )
    _require_exact_bool(
        grant["authorization_ticket_creation_allowed"],
        accepted,
        label="clearance C ticket creation grant",
    )
    _require_exact_bool(
        grant["formal_execution_allowed"],
        accepted,
        label="clearance C formal execution grant",
    )
    _require_exact_int(
        grant["execution_limit"],
        1 if accepted else 0,
        label="clearance C execution limit",
    )
    _require_exact_bool(
        grant["post_marker_retry_allowed"],
        False,
        label="clearance C retry policy",
    )
    scope_fields = set(_V23_SCOPE_FIELDS) | {
        "result_path",
        "marker_path",
    }
    scope = _require_exact_mapping(
        c["scope"], scope_fields, label="clearance C scope"
    )
    scope["result_path"] = _safe_repo_relative_text(
        scope["result_path"], label="clearance C result path"
    )
    scope["marker_path"] = _safe_repo_relative_text(
        scope["marker_path"], label="clearance C marker path"
    )
    findings = _require_unique_array(
        c["blocking_findings"],
        label="clearance C blocking findings",
        nonempty=not accepted,
    )
    if accepted and findings:
        raise OneShotAuthorizationError(
            "accepted clearance C has blocking findings"
        )
    if not all(isinstance(item, str) and item for item in findings):
        raise OneShotAuthorizationError(
            "clearance C blocking findings contain a non-string"
        )
    return c


def _validate_v23_z(value: Any) -> dict[str, Any]:
    z = _require_exact_mapping(
        value, _V23_Z_FIELDS, label="post-review trace Z"
    )
    if (
        z["artifact"]
        != "actual_wake_reachable_pressure_v23_post_review_trace"
        or z["schema_version"] != "1.0"
        or z["version"] != "S3ai-v2.3-post-review-trace-v1"
    ):
        raise OneShotAuthorizationError(
            "post-review trace Z fixed identity drifted"
        )
    for name in (
        "trace_id",
        "provider",
        "model",
        "model_family",
        "agent_id",
    ):
        _require_nonempty_string(
            z[name], label=f"post-review trace Z.{name}"
        )
    for name in (
        "request_path",
        "response_path",
        "invocation_metadata_path",
    ):
        z[name] = _safe_repo_relative_text(
            z[name], label=f"post-review trace Z.{name}"
        )
    for name in (
        "request_raw_sha256",
        "response_raw_sha256",
        "invocation_metadata_raw_sha256",
    ):
        z[name] = _validate_sha256(
            z[name], label=f"post-review trace Z.{name}"
        )
    return z


def _trace_path_identity_v1(
    profile: NamespaceProfileV1,
    path: str,
    suffix: str,
) -> tuple[str, str]:
    allowed = PurePosixPath(
        _profile_repo_relative(profile, profile.allowed_trace_root)
    )
    candidate = PurePosixPath(
        _safe_repo_relative_text(path, label="review trace member path")
    )
    if (
        len(candidate.parts) != len(allowed.parts) + 2
        or candidate.parts[: len(allowed.parts)] != allowed.parts
    ):
        raise OneShotAuthorizationError(
            "review artifact path is not one run directory below root"
        )
    run_directory, filename = candidate.parts[-2:]
    if not _RUN_DIRECTORY_RE.fullmatch(run_directory):
        raise OneShotAuthorizationError(
            "review run directory grammar drifted"
        )
    if not filename.endswith(suffix):
        raise OneShotAuthorizationError(
            "review artifact filename suffix drifted"
        )
    prefix = filename[: -len(suffix)]
    if not _TRACE_ID_RE.fullmatch(prefix):
        raise OneShotAuthorizationError(
            "review artifact call prefix grammar drifted"
        )
    return run_directory, prefix


def _expected_v23_clearance_bindings(
    projection: Mapping[str, Any],
) -> dict[str, str]:
    artifacts = projection["bound_artifacts"]
    manifests = projection["bound_manifests"]
    return {
        "reviewed_wrapper_sha256": artifacts["wrapper_source"][
            "sha256"
        ],
        "reviewed_wrapper_definition_sha256": artifacts[
            "wrapper_definition"
        ]["sha256"],
        "transport_preregistration_raw_sha256": manifests[
            "transport_preregistration_raw_sha256"
        ],
        "dependency_capture_protocol_raw_sha256": manifests[
            "dependency_capture_protocol_raw_sha256"
        ],
        "authorization_schema_preregistration_raw_sha256": manifests[
            "authorization_schema_preregistration_raw_sha256"
        ],
        "bound_artifact_map_sha256": manifests[
            "bound_artifact_map_sha256"
        ],
        "execution_source_map_sha256": manifests[
            "execution_source_map_sha256"
        ],
        "stable_runtime_identity_sha256": manifests[
            "stable_runtime_identity_sha256"
        ],
        "bootstrap_contract_canonical_sha256": manifests[
            "bootstrap_contract_canonical_sha256"
        ],
        "dependency_manifest_raw_sha256": manifests[
            "dependency_manifest_raw_sha256"
        ],
        "dependency_manifest_canonical_sha256": manifests[
            "dependency_manifest_canonical_sha256"
        ],
        "dependency_B_paths_sha256": manifests[
            "dependency_B_paths_sha256"
        ],
        "dependency_U_paths_sha256": manifests[
            "dependency_U_paths_sha256"
        ],
        "dependency_R_paths_sha256": manifests[
            "dependency_R_paths_sha256"
        ],
        "g3_consensus_canonical_sha256": manifests[
            "g3_consensus_canonical_sha256"
        ],
        "g3_audit_raw_sha256": artifacts["g3_independent_audit"][
            "sha256"
        ],
        "g4_audit_raw_sha256": artifacts["g4_definition_test_audit"][
            "sha256"
        ],
        "ordered_registry_manifest_sha256": manifests[
            "ordered_registry_manifest_sha256"
        ],
        "frozen_definition_chain_sha256": manifests[
            "frozen_definition_chain_sha256"
        ],
        "old_failure_quarantine_canonical_sha256": manifests[
            "old_failure_quarantine_canonical_sha256"
        ],
    }


def _merge_v23_audit_file(
    target: dict[str, str],
    path: str,
    digest: str,
) -> None:
    safe = _safe_repo_relative_text(path, label="audit file path")
    sha = _validate_sha256(digest, label=f"audit file {safe}")
    previous = target.get(safe)
    if previous is not None:
        raise OneShotAuthorizationError(
            "one audit path appeared in more than one authorization role"
        )
    target[safe] = sha


def _v23_identity_semantic_projection(
    *,
    implementation_identity: Mapping[str, Any],
    clearance: Mapping[str, Any],
    post_review_trace: Mapping[str, Any],
    accepted: bool = True,
) -> dict[str, Any]:
    """Return the exact implementation/reviewer equivalence classes."""

    implementation = _validate_v23_identity_body(
        implementation_identity,
        label="identity semantic projection implementation",
    )
    response = _validate_v23_c(clearance, accepted=accepted)
    trace = _validate_v23_z(post_review_trace)
    review = response["review"]
    reviewer = {
        "actual_independence": review["actual_independence"],
        "provider": review["provider"],
        "model": review["model"],
        "model_family": review["model_family"],
        "agent_id": review["agent_id"],
        "trace_id": review["trace_id"],
    }
    if (
        reviewer["actual_independence"] != "genuine-cross-family"
        or review["implementation_model_family"]
        != implementation["model_family"]
        or any(
            reviewer[name] != trace[name]
            for name in (
                "provider",
                "model",
                "model_family",
                "agent_id",
                "trace_id",
            )
        )
        or reviewer["model_family"] == implementation["model_family"]
        or reviewer["trace_id"] == implementation["trace_id"]
    ):
        raise OneShotAuthorizationError(
            "implementation/reviewer identity equivalence classes drifted"
        )
    return {
        "implementation_identity": implementation,
        "reviewer_identity": reviewer,
    }


def _verify_v23_cross_object_matrix(
    *,
    profile: NamespaceProfileV1,
    ticket: Mapping[str, Any],
    projection_t: Mapping[str, Any],
    projection_b: Mapping[str, Any],
    q: Mapping[str, Any],
    q_path: str,
    q_raw_sha: str,
    q_canonical_sha: str,
    m: Mapping[str, Any],
    m_path: str,
    m_raw_sha: str,
    m_canonical_sha: str,
    a: Mapping[str, Any],
    a_path: str,
    a_raw_sha: str,
    a_canonical_sha: str,
    c: Mapping[str, Any],
    c_path: str,
    c_raw_sha: str,
    c_canonical_sha: str,
    z: Mapping[str, Any],
    z_path: str,
    z_raw_sha: str,
    implementation_identity: Mapping[str, Any],
    token_sha: str,
) -> None:
    ticket_path = _profile_repo_relative(profile, profile.ticket_path)
    result_path = _profile_repo_relative(profile, profile.result_path)
    marker_path = _profile_repo_relative(profile, profile.marker_path)
    projection_sha = _sha256_bytes(
        _canonical_json_bytes(projection_b)
    )
    if (
        projection_t != projection_b
        or q["ticket_semantic_projection"] != projection_b
        or q["ticket_semantic_projection_sha256"] != projection_sha
        or ticket["issuance_request"][
            "ticket_semantic_projection_sha256"
        ]
        != projection_sha
        or ticket["second_bounded_audit"][
            "ticket_semantic_projection_sha256"
        ]
        != projection_sha
        or c["review"]["ticket_semantic_projection_sha256"]
        != projection_sha
    ):
        raise OneShotAuthorizationError(
            "Q/Build/P projection equivalence drifted"
        )
    if (
        q["planned_authorization_path"] != ticket_path
        or ticket["issuance_request"]["path"] != q_path
        or q_path
        != _profile_repo_relative(
            profile, profile.issuance_request_path
        )
        or ticket["issuance_request"]["raw_sha256"] != q_raw_sha
        or ticket["issuance_request"]["canonical_sha256"]
        != q_canonical_sha
    ):
        raise OneShotAuthorizationError(
            "ticket/Q fixed namespace or digest binding drifted"
        )
    if (
        q["planned_result_path"] != result_path
        or q["planned_marker_path"] != marker_path
        or projection_b["result"]["path"] != result_path
        or projection_b["result"]["marker_path"] != marker_path
        or ticket["result"] != projection_b["result"]
        or ticket["scope"] != projection_b["scope"]
        or c["scope"]
        != {
            **projection_b["scope"],
            "result_path": result_path,
            "marker_path": marker_path,
        }
    ):
        raise OneShotAuthorizationError(
            "result/marker/scope equivalence drifted"
        )
    plan = projection_b["authorization_plan"]
    grant = c["grant"]
    if (
        q["single_use_token_sha256"] != token_sha
        or plan["single_use_token_sha256"] != token_sha
        or ticket["authorization"]["single_use_token_sha256"]
        != token_sha
        or grant["single_use_token_sha256"] != token_sha
        or grant["authorization_id"] != plan["id"]
        or ticket["authorization"]["id"] != plan["id"]
        or grant["decision"] != plan["decision"]
        or ticket["authorization"]["decision"] != plan["decision"]
        or grant["formal_execution_allowed"]
        != plan["formal_execution_allowed"]
        or grant["execution_limit"] != plan["execution_limit"]
        or grant["post_marker_retry_allowed"]
        != plan["post_marker_retry_allowed"]
        or grant["authorization_ticket_creation_allowed"] is not True
    ):
        raise OneShotAuthorizationError(
            "authorization plan/bearer/grant equivalence drifted"
        )
    identity_objects = (
        ticket["implementation_identity"],
        projection_t["implementation_identity"],
        projection_b["implementation_identity"],
        q["implementation_identity"],
        m["implementation_identity"],
        a["implementation_identity"],
    )
    if any(
        identity != implementation_identity
        for identity in identity_objects
    ):
        raise OneShotAuthorizationError(
            "implementation identity equivalence class drifted"
        )
    if (
        m["trace_id"] != a["trace_id"]
        or m["trace_id"] != c["review"]["trace_id"]
        or m["trace_id"] != z["trace_id"]
        or m["requested_reviewer_provider"]
        != a["review_target"]["provider"]
        or m["requested_reviewer_provider"] != c["review"]["provider"]
        or m["requested_reviewer_provider"] != z["provider"]
        or m["requested_reviewer_model"]
        != a["review_target"]["model"]
        or m["requested_reviewer_model"] != c["review"]["model"]
        or m["requested_reviewer_model"] != z["model"]
        or m["requested_reviewer_model_family"]
        != a["review_target"]["model_family"]
        or m["requested_reviewer_model_family"]
        != c["review"]["model_family"]
        or m["requested_reviewer_model_family"] != z["model_family"]
        or c["review"]["agent_id"] != z["agent_id"]
        or c["review"]["implementation_model_family"]
        != implementation_identity["model_family"]
        or z["model_family"] == implementation_identity["model_family"]
        or z["trace_id"] == implementation_identity["trace_id"]
    ):
        raise OneShotAuthorizationError(
            "reviewer identity/cross-family evidence drifted"
        )
    if (
        q["planned_invocation"]["path"] != m_path
        or q["planned_invocation"]["artifact"] != m["artifact"]
        or q["planned_invocation"]["schema_version"]
        != m["schema_version"]
        or q["planned_invocation"]["version"] != m["version"]
        or q["planned_invocation"]["call_prefix"] != m["call_prefix"]
        or m["allowed_trace_root"]
        != _profile_repo_relative(profile, profile.allowed_trace_root)
        or m["issuance_request_path"] != q_path
        or m["issuance_request_raw_sha256"] != q_raw_sha
        or m["issuance_request_canonical_sha256"] != q_canonical_sha
        or m["ticket_semantic_projection_sha256"] != projection_sha
        or m["planned_review_request"]["path"] != a_path
        or m["planned_review_request"]["artifact"] != a["artifact"]
        or m["planned_review_request"]["schema_version"]
        != a["schema_version"]
        or m["planned_review_request"]["version"] != a["version"]
        or m["planned_review_request"]["call_prefix"] != a["call_prefix"]
        or a["issuance_request"]
        != {
            "path": q_path,
            "raw_sha256": q_raw_sha,
            "canonical_sha256": q_canonical_sha,
            "ticket_semantic_projection_sha256": projection_sha,
        }
        or a["invocation_metadata"]
        != {
            "path": m_path,
            "raw_sha256": m_raw_sha,
            "canonical_sha256": m_canonical_sha,
        }
    ):
        raise OneShotAuthorizationError(
            "Q→M→A prior-artifact binding drifted"
        )
    second = ticket["second_bounded_audit"]
    expected_second = {
        "invocation_metadata_path": m_path,
        "invocation_metadata_raw_sha256": m_raw_sha,
        "invocation_metadata_canonical_sha256": m_canonical_sha,
        "request_path": a_path,
        "request_raw_sha256": a_raw_sha,
        "request_canonical_sha256": a_canonical_sha,
        "response_path": c_path,
        "response_raw_sha256": c_raw_sha,
        "response_canonical_sha256": c_canonical_sha,
        "trace_metadata_path": z_path,
        "trace_metadata_raw_sha256": z_raw_sha,
        "clearance_canonical_sha256": c_canonical_sha,
        "ticket_semantic_projection_sha256": projection_sha,
    }
    if (
        second != expected_second
        or a["response_contract"]["response_path"] != c_path
        or a["response_contract"]["post_review_trace_path"] != z_path
        or z["request_path"] != a_path
        or z["request_raw_sha256"] != a_raw_sha
        or z["response_path"] != c_path
        or z["response_raw_sha256"] != c_raw_sha
        or z["invocation_metadata_path"] != m_path
        or z["invocation_metadata_raw_sha256"] != m_raw_sha
        or c["review"]["invocation_metadata_path"] != m_path
        or c["review"]["invocation_metadata_raw_sha256"] != m_raw_sha
        or c["review"]["invocation_metadata_canonical_sha256"]
        != m_canonical_sha
        or c["review"]["request_raw_sha256"] != a_raw_sha
        or c["review"]["request_canonical_sha256"] != a_canonical_sha
        or c["review"]["issuance_request_raw_sha256"] != q_raw_sha
        or c["review"]["issuance_request_canonical_sha256"]
        != q_canonical_sha
    ):
        raise OneShotAuthorizationError(
            "A→C→Z/final-ticket audit binding drifted"
        )
    path_identities = (
        _trace_path_identity_v1(profile, m_path, ".invocation.json"),
        _trace_path_identity_v1(profile, a_path, ".request.json"),
        _trace_path_identity_v1(profile, c_path, ".response.json"),
        _trace_path_identity_v1(profile, z_path, ".trace.json"),
    )
    if (
        len(set(path_identities)) != 1
        or path_identities[0][1] != m["call_prefix"]
        or m["call_prefix"] != a["call_prefix"]
    ):
        raise OneShotAuthorizationError(
            "review call-prefix/path derivation drifted"
        )
    expected_bindings = _expected_v23_clearance_bindings(
        projection_b
    )
    if c["bindings"] != expected_bindings:
        raise OneShotAuthorizationError(
            "clearance C bindings are not the Q projection mapping"
        )
    names, identities, registry_sha = _v23_registry_identity()
    if (
        projection_b["ordered_case_names"] != names
        or ticket["ordered_case_names"] != names
        or projection_b["case_identity_sha256"] != identities
        or ticket["case_identity_sha256"] != identities
        or projection_b["ordered_registry_manifest_sha256"]
        != registry_sha
        or ticket["ordered_registry_manifest_sha256"] != registry_sha
        or projection_b["bound_manifests"][
            "ordered_registry_manifest_sha256"
        ]
        != registry_sha
        or c["bindings"]["ordered_registry_manifest_sha256"]
        != registry_sha
    ):
        raise OneShotAuthorizationError(
            "registry/manifest mirrors drifted"
        )


def _review_trace_hashes(
    second_audit: Mapping[str, Any],
) -> tuple[dict[str, str], dict[str, Any]]:
    traces: dict[str, str] = {}
    response_raw: bytes | None = None
    for role in ("request", "response"):
        path_value = second_audit.get(f"{role}_path")
        digest = _validate_sha256(
            second_audit.get(f"{role}_sha256"),
            label=f"second audit {role} SHA256",
        )
        if not isinstance(path_value, str):
            raise OneShotAuthorizationError(
                f"second audit {role} path is missing"
            )
        pure = PurePosixPath(path_value)
        if (
            pure.is_absolute()
            or ".." in pure.parts
            or not pure.parts
            or pure.parts[0] != ".aris"
        ):
            raise OneShotAuthorizationError(
                f"second audit {role} path is outside the traced .aris tree"
            )
        raw = _read_regular_nofollow(
            REPO_ROOT / pure,
            label=f"second bounded audit {role}",
        )
        actual = _sha256_bytes(raw)
        if actual != digest:
            raise OneShotAuthorizationError(
                f"second bounded audit {role} trace hash drifted"
            )
        traces[path_value] = digest
        if role == "response":
            response_raw = raw
    if response_raw is None:
        raise OneShotAuthorizationError(
            "second bounded audit response was not observed"
        )
    try:
        clearance = json.loads(response_raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise OneShotAuthorizationError(
            "second bounded audit response must be an exact UTF-8 JSON "
            "object without Markdown fences"
        ) from error
    if not isinstance(clearance, dict):
        raise OneShotAuthorizationError(
            "second bounded audit clearance must be a JSON object"
        )
    _strict_json_value(clearance, label="second bounded audit clearance")
    declared_clearance = _validate_sha256(
        second_audit.get("clearance_canonical_sha256"),
        label="second bounded audit canonical clearance SHA256",
    )
    actual_clearance = _sha256_bytes(_canonical_json_bytes(clearance))
    if actual_clearance != declared_clearance:
        raise OneShotAuthorizationError(
            "second bounded audit canonical clearance digest drifted"
        )
    return traces, clearance


def _load_and_verify_retired_v22_authorization_reference(
    *,
    expected_authorization_sha256: str,
    second_audit_token: bytes,
    definition: Mapping[str, Any],
    definition_sha256: str,
    source_fingerprints: Mapping[str, str],
    runtime_identity: Mapping[str, Any],
) -> _VerifiedAuthorization:
    expected_raw = _validate_sha256(
        expected_authorization_sha256,
        label="externally supplied authorization SHA256",
    )
    if (
        not isinstance(second_audit_token, bytes)
        or len(second_audit_token) != 32
    ):
        raise OneShotAuthorizationError(
            "second-audit bearer token must contain exactly 32 bytes"
        )
    raw = _read_regular_nofollow(
        AUTHORIZATION_PATH, label="one-shot authorization ticket"
    )
    raw_sha = _sha256_bytes(raw)
    if raw_sha != expected_raw:
        raise OneShotAuthorizationError(
            "authorization ticket does not match the externally supplied "
            "raw SHA256"
        )
    payload = _parse_yaml_mapping(raw, label="one-shot authorization ticket")
    authorization = payload.get("authorization")
    if (
        set(payload)
        != {
            "artifact",
            "version",
            "status",
            "authorization",
            "wrapper_definition",
            "wrapper_source",
            "definition_chain",
            "execution_sources",
            "runtime_identity",
            "bound_manifests",
            "second_bounded_audit",
            "result",
            "scope",
            "ordered_case_names",
            "case_identity_sha256",
            "ordered_registry_manifest_sha256",
        }
        or payload.get("artifact")
        != "actual_wake_reachable_pressure_execution_authorization"
        or payload.get("version") != "S3ai-v2.2-one-shot-authorization-v1"
        or payload.get("status")
        != "accepted_after_second_bounded_audit"
        or not isinstance(authorization, Mapping)
        or not isinstance(authorization.get("id"), str)
        or not authorization.get("id")
        or set(authorization)
        != {
            "id",
            "canonical_sha256",
            "external_raw_sha256_required",
            "single_use_token_sha256",
            "formal_execution_allowed",
            "execution_limit",
            "decision",
        }
    ):
        raise OneShotAuthorizationError(
            "authorization ticket identity or metadata drifted"
        )
    declared_canonical = _validate_sha256(
        authorization.get("canonical_sha256"),
        label="canonical authorization digest",
    )
    canonical = _canonical_authorization_digest(payload)
    if canonical != declared_canonical:
        raise OneShotAuthorizationError(
            "canonical authorization digest drifted"
        )
    token_sha = _verify_second_audit_token(
        authorization.get("single_use_token_sha256"),
        second_audit_token,
    )
    if (
        authorization.get("external_raw_sha256_required") is not True
        or authorization.get("formal_execution_allowed") is not True
        or authorization.get("execution_limit") != 1
        or authorization.get("decision")
        != "YES_ONE_SHOT_31_HISTORY_ONLY"
    ):
        raise OneShotAuthorizationError(
            "authorization ticket does not grant exactly one 31-history run"
        )

    definition_binding = payload.get("wrapper_definition", {})
    if (
        not isinstance(definition_binding, Mapping)
        or set(definition_binding) != {"path", "sha256"}
        or definition_binding.get("path")
        != _validated_repo_relative(WRAPPER_DEFINITION_PATH).as_posix()
        or definition_binding.get("sha256") != definition_sha256
    ):
        raise OneShotAuthorizationError(
            "authorization does not bind the audited wrapper preregistration"
        )
    wrapper_binding = payload.get("wrapper_source", {})
    wrapper_sha = source_fingerprints.get(_WRAPPER_RELATIVE)
    if (
        not isinstance(wrapper_binding, Mapping)
        or set(wrapper_binding) != {"path", "sha256"}
        or wrapper_binding.get("path") != _WRAPPER_RELATIVE
        or wrapper_binding.get("sha256") != wrapper_sha
    ):
        raise OneShotAuthorizationError(
            "authorization does not bind this wrapper source"
        )
    if payload.get("execution_sources") != dict(source_fingerprints):
        raise OneShotAuthorizationError(
            "authorization execution-source map differs from the audited "
            "63-file snapshot"
        )
    if payload.get("runtime_identity") != dict(runtime_identity):
        raise OneShotAuthorizationError(
            "authorization runtime identity differs from this process"
        )
    source_map_sha = _sha256_bytes(
        _canonical_json_bytes(dict(source_fingerprints))
    )
    runtime_sha = _sha256_bytes(_canonical_json_bytes(runtime_identity))
    manifests = payload.get("bound_manifests", {})
    if (
        not isinstance(manifests, Mapping)
        or set(manifests)
        != {
            "execution_source_map_sha256",
            "runtime_identity_sha256",
            "ordered_registry_manifest_sha256",
            "frozen_definition_chain_sha256",
        }
        or manifests.get("execution_source_map_sha256") != source_map_sha
        or manifests.get("runtime_identity_sha256") != runtime_sha
        or not isinstance(
            manifests.get("ordered_registry_manifest_sha256"), str
        )
        or not isinstance(
            manifests.get("frozen_definition_chain_sha256"), str
        )
    ):
        raise OneShotAuthorizationError(
            "authorization source/runtime manifest bindings drifted"
        )
    _validate_sha256(
        manifests.get("ordered_registry_manifest_sha256"),
        label="ordered registry manifest",
    )
    _validate_sha256(
        manifests.get("frozen_definition_chain_sha256"),
        label="frozen definition-chain manifest",
    )

    second_audit = payload.get("second_bounded_audit", {})
    if (
        not isinstance(second_audit, Mapping)
        or set(second_audit)
        != {
            "request_path",
            "request_sha256",
            "response_path",
            "response_sha256",
            "clearance_canonical_sha256",
        }
    ):
        raise OneShotAuthorizationError(
            "second bounded audit metadata is missing"
        )
    audit_files, clearance = _review_trace_hashes(second_audit)
    expected_clearance_keys = {
        "artifact",
        "version",
        "verdict",
        "actual_independence",
        "reviewed_wrapper_sha256",
        "reviewed_prereg_sha256",
        "execution_source_map_sha256",
        "runtime_identity_sha256",
        "ordered_registry_manifest_sha256",
        "frozen_definition_chain_sha256",
        "single_use_token_sha256",
        "authorization_ticket_creation_allowed",
        "formal_execution_allowed",
        "execution_limit",
        "production_activation_allowed",
        "force_hp_state_ves_118_fig_allowed",
        "blocking_findings",
    }
    if (
        set(clearance) != expected_clearance_keys
        or clearance.get("artifact")
        != "S3ai-v2.2-one-shot-execution-clearance"
        or clearance.get("version") != 1
        or clearance.get("verdict")
        != "ACCEPT_EXACT_ONE_SHOT_31_HISTORY_EXECUTION"
        or clearance.get("actual_independence")
        != "genuine-cross-family"
        or clearance.get("reviewed_wrapper_sha256") != wrapper_sha
        or clearance.get("reviewed_prereg_sha256") != definition_sha256
        or clearance.get("execution_source_map_sha256") != source_map_sha
        or clearance.get("runtime_identity_sha256") != runtime_sha
        or clearance.get("ordered_registry_manifest_sha256")
        != manifests.get("ordered_registry_manifest_sha256")
        or clearance.get("frozen_definition_chain_sha256")
        != manifests.get("frozen_definition_chain_sha256")
        or clearance.get("single_use_token_sha256") != token_sha
        or clearance.get("authorization_ticket_creation_allowed")
        is not True
        or clearance.get("formal_execution_allowed") is not True
        or clearance.get("execution_limit") != 1
        or clearance.get("production_activation_allowed") is not False
        or clearance.get("force_hp_state_ves_118_fig_allowed") is not False
        or clearance.get("blocking_findings") != []
    ):
        raise OneShotAuthorizationError(
            "traced structured clearance does not grant the exact bound "
            "one-shot execution"
        )

    result = payload.get("result", {})
    if (
        not isinstance(result, Mapping)
        or set(result)
        != {
            "path",
            "marker_path",
            "overwrite_allowed",
            "latest_pointer_write_allowed",
        }
        or result.get("path")
        != _validated_repo_relative(RESULT_PATH).as_posix()
        or result.get("marker_path")
        != _validated_repo_relative(ATTEMPT_MARKER_PATH).as_posix()
        or result.get("overwrite_allowed") is not False
        or result.get("latest_pointer_write_allowed") is not False
    ):
        raise OneShotAuthorizationError(
            "authorization result/marker contract drifted"
        )
    scope = payload.get("scope", {})
    if (
        not isinstance(scope, Mapping)
        or set(scope)
        != {
            "one_31_history_execution_only",
            "production_activation_allowed",
            "force_hp_state_ves_118_fig_allowed",
        }
        or scope.get("one_31_history_execution_only") is not True
        or scope.get("production_activation_allowed") is not False
        or scope.get("force_hp_state_ves_118_fig_allowed") is not False
    ):
        raise OneShotAuthorizationError(
            "authorization scope exceeds the execution-transport node"
        )
    if (
        definition["decision"][
            "authorization_ticket_creation_allowed_before_second_bounded_audit"
        ]
        is not False
        or definition["decision"][
            "authorization_ticket_creation_requires_accepted_second_bounded_audit"
        ]
        is not True
        or definition["decision"]["formal_execution_allowed"] is not False
    ):
        raise OneShotAuthorizationError(
            "the preregistration definition must remain fail-closed"
        )
    return _VerifiedAuthorization(
        payload=payload,
        raw_sha256=raw_sha,
        canonical_sha256=canonical,
        token_sha256=token_sha,
        definition_sha256=definition_sha256,
        source_fingerprints=dict(source_fingerprints),
        runtime_identity=dict(runtime_identity),
        audit_files=audit_files,
        clearance=clearance,
    )


def _verify_old_failure_quarantine() -> dict[str, str]:
    """Bind all seven retired assets and both required absences."""

    observed: dict[str, str] = {}
    for path, expected in _OLD_QUARANTINE_HASHES.items():
        actual = _sha256_bytes(
            _read_regular_nofollow(path, label=f"retired asset {path.name}")
        )
        if actual != expected:
            raise OneShotAuthorizationError(
                f"retired failure asset drifted: {path.name}"
            )
        observed[_validated_repo_relative(path).as_posix()] = actual
    for path, label in (
        (OLD_CANONICAL_RESULT_PATH, "retired canonical result"),
        (LATEST_RESULT_PATH, "retired latest result pointer"),
    ):
        if _entry_lstat(path) is not None:
            raise OneShotAuthorizationError(
                f"{label} must remain absent for the v2.3 successor"
            )
    return observed


def _reject_retired_authorization_inputs(
    *,
    expected_authorization_sha256: str,
    second_audit_token: bytes,
) -> None:
    expected = _validate_sha256(
        expected_authorization_sha256,
        label="externally supplied v2.3 authorization SHA256",
    )
    if expected == _OLD_AUTHORIZATION_SHA256:
        raise OneShotAuthorizationError(
            "the retired v2.2 authorization SHA cannot authorize v2.3"
        )
    if (
        isinstance(second_audit_token, bytes)
        and _sha256_bytes(second_audit_token) == _OLD_TOKEN_SHA256
    ):
        raise OneShotAuthorizationError(
            "the retired v2.2 bearer token cannot authorize v2.3"
        )


def _load_and_verify_authorization_from_ticket_path(
    *,
    namespace_profile: NamespaceProfileV1,
    ticket_path: Path,
    expected_authorization_sha256: str,
    expected_implementation_identity_core_raw_sha256: str,
    second_audit_token: bytes,
    definition: Mapping[str, Any],
    dependency_manifest: DependencyManifest,
    source_fingerprints: Mapping[str, str],
    stable_runtime_identity: Mapping[str, Any],
) -> _VerifiedAuthorization:
    """Verify the exact v2.3 Q→M→A→C→Z→ticket graph without history."""

    if not isinstance(ticket_path, Path):
        raise OneShotAuthorizationError(
            "private authorization seam requires an explicit Path"
        )
    if ticket_path != namespace_profile.ticket_path:
        raise OneShotAuthorizationError(
            "ticket path differs from the typed namespace profile"
        )
    expected_ticket_sha = _validate_sha256(
        expected_authorization_sha256,
        label="externally supplied v2.3 authorization SHA256",
    )
    if expected_ticket_sha == _OLD_AUTHORIZATION_SHA256:
        raise OneShotAuthorizationError(
            "retired authorization SHA cannot authorize v2.3"
        )
    if (
        not isinstance(second_audit_token, bytes)
        or len(second_audit_token) != 32
    ):
        raise OneShotAuthorizationError(
            "second-audit bearer token must contain exactly 32 bytes"
        )
    token_sha = _sha256_bytes(second_audit_token)
    if token_sha == _OLD_TOKEN_SHA256:
        raise OneShotAuthorizationError(
            "retired bearer token cannot authorize v2.3"
        )
    _validate_namespace_profile_v1(namespace_profile)
    seed = _build_identity_core_binding_seed_v1(
        namespace_profile,
        expected_implementation_identity_core_raw_sha256,
    )

    ticket_relative = _profile_repo_relative(
        namespace_profile, ticket_path
    )
    ticket_fd = _open_repo_relative_nofollow_v1(
        namespace_profile, ticket_relative
    )
    try:
        state_pre = _loaded_dependency_state()
        chunks: list[bytes] = []
        while True:
            chunk = os.read(ticket_fd, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        ticket_raw = b"".join(chunks)
    finally:
        os.close(ticket_fd)
        _validate_namespace_profile_v1(namespace_profile)
    ticket_raw_sha = _sha256_bytes(ticket_raw)
    if (
        ticket_raw_sha != expected_ticket_sha
        or ticket_raw_sha == _OLD_AUTHORIZATION_SHA256
    ):
        raise OneShotAuthorizationError(
            "ticket raw bytes differ from the external authorization SHA"
        )
    ticket = _strict_pretty_json_object(
        ticket_raw, label="v2.3 authorization ticket"
    )
    ticket = _validate_v23_ticket_static(ticket)
    if ticket["namespace_profile"] != namespace_profile.mode:
        raise OneShotAuthorizationError(
            "ticket namespace profile discriminator drifted"
        )
    _validate_v23_result_scope(
        namespace_profile, ticket["result"], ticket["scope"]
    )
    ticket_canonical_sha = _canonical_authorization_digest(ticket)
    if (
        ticket["authorization"]["canonical_sha256"]
        != ticket_canonical_sha
    ):
        raise OneShotAuthorizationError(
            "ticket canonical self digest drifted"
        )
    if ticket["authorization"]["single_use_token_sha256"] != token_sha:
        raise OneShotAuthorizationError(
            "ticket bearer commitment differs from the supplied token"
        )

    q_path = ticket["issuance_request"]["path"]
    if q_path != _profile_repo_relative(
        namespace_profile,
        namespace_profile.issuance_request_path,
    ):
        raise OneShotAuthorizationError(
            "ticket-declared issuance path is not profile-owned"
        )
    q_payload, _, q_raw_sha, q_canonical_sha = _read_pretty_child_v1(
        namespace_profile,
        q_path,
        label="issuance request Q",
        expected_raw_sha256=ticket["issuance_request"]["raw_sha256"],
        expected_canonical_sha256=(
            ticket["issuance_request"]["canonical_sha256"]
        ),
    )
    q = _validate_v23_q(q_payload)

    second = ticket["second_bounded_audit"]
    m_path = q["planned_invocation"]["path"]
    if m_path != second["invocation_metadata_path"]:
        raise OneShotAuthorizationError(
            "Q planned invocation and ticket audit path differ"
        )
    m_payload, _, m_raw_sha, m_canonical_sha = _read_pretty_child_v1(
        namespace_profile,
        m_path,
        label="invocation metadata M",
        expected_raw_sha256=second["invocation_metadata_raw_sha256"],
        expected_canonical_sha256=(
            second["invocation_metadata_canonical_sha256"]
        ),
    )
    m = _validate_v23_m(m_payload)

    a_path = m["planned_review_request"]["path"]
    if a_path != second["request_path"]:
        raise OneShotAuthorizationError(
            "M planned request and ticket audit path differ"
        )
    a_payload, _, a_raw_sha, a_canonical_sha = _read_pretty_child_v1(
        namespace_profile,
        a_path,
        label="review request A",
        expected_raw_sha256=second["request_raw_sha256"],
        expected_canonical_sha256=second["request_canonical_sha256"],
    )
    a = _validate_v23_a(a_payload)

    c_path = second["response_path"]
    z_path = second["trace_metadata_path"]
    if (
        a["response_contract"]["response_path"] != c_path
        or a["response_contract"]["post_review_trace_path"] != z_path
    ):
        raise OneShotAuthorizationError(
            "A response contract and ticket audit paths differ"
        )
    c_payload, _, c_raw_sha, c_canonical_sha = _read_pretty_child_v1(
        namespace_profile,
        c_path,
        label="accepted clearance C",
        expected_raw_sha256=second["response_raw_sha256"],
        expected_canonical_sha256=second["response_canonical_sha256"],
    )
    c = _validate_v23_c(c_payload, accepted=True)
    z_payload, _, z_raw_sha, _ = _read_pretty_child_v1(
        namespace_profile,
        z_path,
        label="post-review trace Z",
        expected_raw_sha256=second["trace_metadata_raw_sha256"],
    )
    z = _validate_v23_z(z_payload)

    artifacts, raw_by_role, audit_files = (
        _observe_v23_bound_artifacts(
            namespace_profile, ticket["bound_artifacts"]
        )
    )
    identity, source_trace_audit = _validate_v23_identity_chain(
        namespace_profile,
        seed=seed,
        bound_artifacts=artifacts,
        raw_by_role=raw_by_role,
    )
    for path, digest in source_trace_audit.items():
        _merge_v23_audit_file(audit_files, path, digest)
    normalized_sources = _validate_v23_execution_sources(
        source_fingerprints
    )
    runtime = _strict_json_value(
        stable_runtime_identity,
        label="private seam stable runtime identity",
    )
    quarantine = _validate_v23_old_quarantine(
        ticket["old_failure_quarantine"]
    )
    plan = _FrozenTicketSemanticPlanInputsV1(
        namespace_profile=namespace_profile,
        authorization_id=q["ticket_semantic_projection"][
            "authorization_plan"
        ]["id"],
        single_use_token_sha256=token_sha,
        bound_artifacts=artifacts,
        implementation_identity=identity,
        definition=definition,
        dependency_manifest=dependency_manifest,
        source_fingerprints=normalized_sources,
        stable_runtime_identity=runtime,
        old_failure_quarantine=quarantine,
    )
    projection_b = _build_projection_v1(plan)
    projection_t = _ticket_semantic_projection_v1(ticket)
    _verify_v23_cross_object_matrix(
        profile=namespace_profile,
        ticket=ticket,
        projection_t=projection_t,
        projection_b=projection_b,
        q=q,
        q_path=q_path,
        q_raw_sha=q_raw_sha,
        q_canonical_sha=q_canonical_sha,
        m=m,
        m_path=m_path,
        m_raw_sha=m_raw_sha,
        m_canonical_sha=m_canonical_sha,
        a=a,
        a_path=a_path,
        a_raw_sha=a_raw_sha,
        a_canonical_sha=a_canonical_sha,
        c=c,
        c_path=c_path,
        c_raw_sha=c_raw_sha,
        c_canonical_sha=c_canonical_sha,
        z=z,
        z_path=z_path,
        z_raw_sha=z_raw_sha,
        implementation_identity=identity,
        token_sha=token_sha,
    )
    identity_projection_sha = _sha256_bytes(
        _canonical_json_bytes(
            _v23_identity_semantic_projection(
                implementation_identity=identity,
                clearance=c,
                post_review_trace=z,
            )
        )
    )
    for path, digest in (
        (q_path, q_raw_sha),
        (m_path, m_raw_sha),
        (a_path, a_raw_sha),
        (c_path, c_raw_sha),
        (z_path, z_raw_sha),
    ):
        _merge_v23_audit_file(audit_files, path, digest)
    _validate_namespace_profile_v1(namespace_profile)
    state_1 = _loaded_dependency_state()
    state_2 = _loaded_dependency_state()
    if (
        state_pre != state_1
        or state_1 != state_2
        or _canonical_json_bytes(state_pre)
        != _canonical_json_bytes(state_1)
        or _canonical_json_bytes(state_1)
        != _canonical_json_bytes(state_2)
    ):
        raise OneShotAuthorizationError(
            "authorization parser loaded or changed a file-backed dependency"
        )
    _validate_namespace_profile_v1(namespace_profile)
    return _VerifiedAuthorization(
        payload=ticket,
        raw_sha256=ticket_raw_sha,
        canonical_sha256=ticket_canonical_sha,
        token_sha256=token_sha,
        definition_sha256=artifacts["wrapper_definition"]["sha256"],
        transport_preregistration_sha256=artifacts[
            "transport_preregistration"
        ]["sha256"],
        dependency_manifest_raw_sha256=str(
            dependency_manifest.raw_sha256
        ),
        dependency_manifest_canonical_sha256=(
            dependency_manifest.canonical_sha256
        ),
        source_fingerprints=normalized_sources,
        stable_runtime_identity=runtime,
        audit_files=dict(sorted(audit_files.items())),
        clearance=c,
        identity_projection_sha256=identity_projection_sha,
    )


def _load_and_verify_authorization(
    *,
    expected_authorization_sha256: str,
    expected_implementation_identity_core_raw_sha256: str,
    second_audit_token: bytes,
    definition: Mapping[str, Any],
    definition_sha256: str,
    transport_preregistration_sha256: str,
    dependency_manifest: DependencyManifest,
    source_fingerprints: Mapping[str, str],
    stable_runtime_identity: Mapping[str, Any],
) -> _VerifiedAuthorization:
    """Production-only adapter over the exact shared private parser."""

    profile = _production_namespace_profile_v1()
    try:
        verified = _load_and_verify_authorization_from_ticket_path(
            namespace_profile=profile,
            ticket_path=AUTHORIZATION_PATH,
            expected_authorization_sha256=(
                expected_authorization_sha256
            ),
            expected_implementation_identity_core_raw_sha256=(
                expected_implementation_identity_core_raw_sha256
            ),
            second_audit_token=second_audit_token,
            definition=definition,
            dependency_manifest=dependency_manifest,
            source_fingerprints=source_fingerprints,
            stable_runtime_identity=stable_runtime_identity,
        )
    finally:
        _close_namespace_profile_v1(profile)
    if (
        verified.definition_sha256 != definition_sha256
        or verified.transport_preregistration_sha256
        != transport_preregistration_sha256
    ):
        raise OneShotAuthorizationError(
            "public adapter definition/transport raw bindings drifted"
        )
    return verified


def _case_registry_identity(
    contract: Mapping[str, Any],
) -> tuple[list[str], dict[str, str], str]:
    cases = guard._validated_registry(contract)
    names = [case.name for case in cases]
    identities = {
        case.name: guard._case_identity_payload(case)["sha256"]
        for case in cases
    }
    manifest = [
        {"name": name, "case_identity_sha256": identities[name]}
        for name in names
    ]
    return names, identities, _sha256_bytes(_canonical_json_bytes(manifest))


def _verify_contract_authorization(
    verified: _VerifiedAuthorization,
    contract: Mapping[str, Any],
) -> tuple[list[str], dict[str, str], str]:
    _assert_verified_authorization_memory_integrity(verified)
    if (
        contract.get("version") != "S3ai-v2.2"
        or contract.get("decision", {}).get("formal_execution_allowed")
        is not False
        or contract.get("decision", {}).get(
            "production_activation_allowed"
        )
        is not False
    ):
        raise OneShotAuthorizationError(
            "frozen contract must remain execution-closed and no-production"
        )
    ticket = verified.payload
    if ticket.get("definition_chain") != contract.get("_definition_chain"):
        raise OneShotAuthorizationError(
            "authorization frozen definition chain drifted"
        )
    names, identities, manifest_sha = _case_registry_identity(contract)
    definition_chain_sha = _sha256_bytes(
        _canonical_json_bytes(contract["_definition_chain"])
    )
    manifests = ticket.get("bound_manifests", {})
    clearance_bindings = verified.clearance.get("bindings", {})
    if (
        ticket.get("ordered_case_names") != names
        or ticket.get("case_identity_sha256") != identities
        or ticket.get("ordered_registry_manifest_sha256") != manifest_sha
        or manifests.get("ordered_registry_manifest_sha256")
        != manifest_sha
        or manifests.get("frozen_definition_chain_sha256")
        != definition_chain_sha
        or not isinstance(clearance_bindings, Mapping)
        or clearance_bindings.get("ordered_registry_manifest_sha256")
        != manifest_sha
        or clearance_bindings.get("frozen_definition_chain_sha256")
        != definition_chain_sha
    ):
        raise OneShotAuthorizationError(
            "authorization/clearance does not bind the exact registry and "
            "frozen definition chain"
        )
    return names, identities, manifest_sha


def _verify_ticket_frozen_files_before_loader(
    verified: _VerifiedAuthorization,
) -> dict[str, str]:
    _assert_verified_authorization_memory_integrity(verified)
    chain = verified.payload.get("definition_chain")
    if (
        not isinstance(chain, Mapping)
        or set(chain)
        != {"S3ai-v2", "S3ai-v2.1", "S3ai-v2.2", "implementation_audit"}
    ):
        raise OneShotAuthorizationError(
            "authorization frozen definition-chain schema drifted"
        )
    chain_sha = _sha256_bytes(_canonical_json_bytes(chain))
    manifests = verified.payload.get("bound_manifests", {})
    clearance_bindings = verified.clearance.get("bindings", {})
    if (
        chain_sha != manifests.get("frozen_definition_chain_sha256")
        or not isinstance(clearance_bindings, Mapping)
        or chain_sha
        != clearance_bindings.get("frozen_definition_chain_sha256")
    ):
        raise OneShotAuthorizationError(
            "authorization/clearance definition-chain digest drifted"
        )
    observed: dict[str, str] = {}
    for role, entry in chain.items():
        if not isinstance(entry, Mapping):
            raise OneShotAuthorizationError(
                f"frozen definition-chain entry {role} is invalid"
            )
        filename = entry.get("file")
        expected = _validate_sha256(
            entry.get("sha256"),
            label=f"frozen definition {role}",
        )
        if (
            not isinstance(filename, str)
            or PurePosixPath(filename).name != filename
            or filename in observed
        ):
            raise OneShotAuthorizationError(
                f"unsafe or duplicate frozen definition filename for {role}"
            )
        actual = _sha256_bytes(
            _read_regular_nofollow(
                PLATFORM / "docs" / "diag" / filename,
                label=f"frozen definition {role}",
            )
        )
        if actual != expected:
            raise OneShotAuthorizationError(
                f"frozen definition {role} changed before its loader"
            )
        observed[filename] = actual
    return observed


def _write_all(fd: int, payload: bytes) -> None:
    offset = 0
    while offset < len(payload):
        written = os.write(fd, payload[offset:])
        if written <= 0:
            raise OSError("short write")
        offset += written


def _pread_exact(fd: int, length: int) -> bytes:
    chunks: list[bytes] = []
    offset = 0
    while offset < length:
        chunk = os.pread(fd, length - offset, offset)
        if not chunk:
            break
        chunks.append(chunk)
        offset += len(chunk)
    return b"".join(chunks)


def _claim_attempt_once(
    *,
    marker_path: Path,
    result_path: Path,
    receipt: Mapping[str, Any],
    output_directory: _OutputDirectoryLease | None = None,
) -> _AttemptClaim:
    lease = output_directory or _acquire_output_directory_lease(
        marker_path, result_path
    )
    owns_lease = output_directory is None
    fd = -1
    try:
        _verify_output_directory_lease(
            lease,
            marker_path=marker_path,
            result_path=result_path,
        )
        for name, label in (
            (lease.result_name, "formal result"),
            (lease.marker_name, "permanent attempt marker"),
        ):
            try:
                existing = os.stat(
                    name, dir_fd=lease.fd, follow_symlinks=False
                )
            except FileNotFoundError:
                continue
            kind = (
                "symlink"
                if stat.S_ISLNK(existing.st_mode)
                else "existing entry"
            )
            raise OneShotAuthorizationError(
                f"{label} must be absent before attempt claim ({kind})"
            )
        payload = _pretty_json_bytes(receipt)
        try:
            fd = os.open(
                lease.marker_name,
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | _O_NOFOLLOW,
                0o444,
                dir_fd=lease.fd,
            )
        except FileExistsError as error:
            raise OneShotAuthorizationError(
                "permanent attempt marker already exists"
            ) from error
        _write_all(fd, payload)
        os.fsync(fd)
        marker_metadata = os.fstat(fd)
        if (
            not stat.S_ISREG(marker_metadata.st_mode)
            or int(marker_metadata.st_size) != len(payload)
        ):
            raise OneShotAuthorizationError(
                "durable attempt marker inode or size is invalid"
            )
        os.close(fd)
        fd = -1
        os.fsync(lease.fd)
        _verify_output_directory_lease(
            lease,
            marker_path=marker_path,
            result_path=result_path,
        )
        canonical_marker = os.stat(
            lease.marker_name,
            dir_fd=lease.fd,
            follow_symlinks=False,
        )
        if (
            int(canonical_marker.st_dev),
            int(canonical_marker.st_ino),
        ) != (
            int(marker_metadata.st_dev),
            int(marker_metadata.st_ino),
        ):
            raise OneShotAuthorizationError(
                "canonical attempt marker does not name its created inode"
            )
        return _AttemptClaim(
            receipt_sha256=_sha256_bytes(payload),
            st_dev=int(marker_metadata.st_dev),
            st_ino=int(marker_metadata.st_ino),
            st_size=int(marker_metadata.st_size),
        )
    except OSError as error:
        raise OneShotAuthorizationError(
            f"could not durably claim the one-shot attempt: {error}"
        ) from error
    finally:
        if fd >= 0:
            os.close(fd)
        if owns_lease:
            os.close(lease.fd)


def _verify_permanent_attempt_marker(
    marker_path: Path,
    expected_sha256: str,
    *,
    output_directory: _OutputDirectoryLease | None = None,
    result_path: Path | None = None,
    expected_identity: _AttemptClaim | None = None,
) -> None:
    expected = _validate_sha256(
        expected_sha256, label="permanent attempt-marker receipt"
    )
    if output_directory is None:
        observed = _sha256_bytes(
            _read_regular_nofollow(
                marker_path, label="permanent attempt marker"
            )
        )
    else:
        if result_path is None:
            raise OneShotAuthorizationError(
                "leased marker verification requires the result path"
            )
        _verify_output_directory_lease(
            output_directory,
            marker_path=marker_path,
            result_path=result_path,
        )
        fd = -1
        try:
            fd = os.open(
                output_directory.marker_name,
                os.O_RDONLY | _O_NOFOLLOW,
                dir_fd=output_directory.fd,
            )
            metadata = os.fstat(fd)
            if not stat.S_ISREG(metadata.st_mode):
                raise OneShotAuthorizationError(
                    "permanent attempt marker is not a regular file"
                )
            if expected_identity is not None and (
                int(metadata.st_dev),
                int(metadata.st_ino),
                int(metadata.st_size),
            ) != (
                expected_identity.st_dev,
                expected_identity.st_ino,
                expected_identity.st_size,
            ):
                raise OneShotAuthorizationError(
                    "permanent attempt marker inode identity changed"
                )
            chunks: list[bytes] = []
            while True:
                chunk = os.read(fd, 1024 * 1024)
                if not chunk:
                    break
                chunks.append(chunk)
            observed = _sha256_bytes(b"".join(chunks))
        except OSError as error:
            raise OneShotAuthorizationError(
                f"cannot read permanent attempt marker: {error}"
            ) from error
        finally:
            if fd >= 0:
                os.close(fd)
    if observed != expected:
        raise OneShotAuthorizationError(
            "permanent attempt marker disappeared or changed after the "
            "authorization was consumed"
        )


def _observation_payload(observation: Any) -> dict[str, Any]:
    stages = observation.stage_arrays
    return {
        "mass_active": np.asarray(observation.mass_active, dtype=float),
        "stage_times": np.asarray(stages.stage_times, dtype=float),
        "measurement_times": np.asarray(
            stages.measurement_times, dtype=float
        ),
        "stored_window_residual": np.asarray(
            observation.stored_window_residual, dtype=float
        ),
        "direct_window_residual": np.asarray(
            observation.direct_window_residual, dtype=float
        ),
        "stored_step_residuals": np.asarray(
            observation.stored_step_residuals, dtype=float
        ),
        "direct_step_residuals": np.asarray(
            observation.direct_step_residuals, dtype=float
        ),
        "canonical_material_current_trace": np.asarray(
            stages.canonical_material_current_trace, dtype=float
        ),
        "body_cut_trace": np.asarray(stages.body_cut_trace, dtype=float),
        "canonical_material_release": np.asarray(
            stages.canonical_material_release, dtype=float
        ),
        "representation_inventory": np.asarray(
            stages.representation_inventory, dtype=float
        ),
        "stored_weak_pressure": np.asarray(
            stages.stored_weak_pressure, dtype=float
        ),
        "direct_weak_pressure": np.asarray(
            stages.direct_weak_pressure, dtype=float
        ),
        "stage_roles": list(stages.stage_roles),
    }


def _extended_observation_hashes(observation: Any) -> dict[str, str]:
    payload = _observation_payload(observation)
    hashes = {
        key: _array_sha256(payload[key])
        for key in (
            "mass_active",
            "stage_times",
            "measurement_times",
            "stored_window_residual",
            "direct_window_residual",
            "stored_step_residuals",
            "direct_step_residuals",
            "canonical_material_current_trace",
            "body_cut_trace",
            "canonical_material_release",
            "representation_inventory",
            "stored_weak_pressure",
            "direct_weak_pressure",
        )
    }
    hashes["stage_roles_canonical_JSON"] = _sha256_bytes(
        _canonical_json_bytes(payload["stage_roles"])
    )
    if set(hashes) != _EXTENDED_VALUE_HASH_KEYS:
        raise OneShotAuthorizationError(
            "extended observation value-hash schema drifted"
        )
    return hashes


def _case_payload_from_serialized(case: Mapping[str, Any]) -> dict[str, Any]:
    typed = case.get("typed_stage_arrays", {})
    if not isinstance(typed, Mapping):
        raise OneShotAuthorizationError(
            "serialized case lacks typed_stage_arrays"
        )
    return {
        "mass_active": case.get("mass_active"),
        "stage_times": case.get("stage_times"),
        "measurement_times": case.get("measurement_times"),
        "stored_window_residual": case.get("stored_window_residual"),
        "direct_window_residual": case.get("direct_window_residual"),
        "stored_step_residuals": case.get("stored_step_residuals"),
        "direct_step_residuals": case.get("direct_step_residuals"),
        "canonical_material_current_trace": typed.get(
            "canonical_material_current_trace"
        ),
        "body_cut_trace": typed.get("body_cut_trace"),
        "canonical_material_release": typed.get(
            "canonical_material_release"
        ),
        "representation_inventory": typed.get(
            "representation_inventory"
        ),
        "stored_weak_pressure": typed.get("stored_weak_pressure"),
        "direct_weak_pressure": typed.get("direct_weak_pressure"),
        "stage_roles": case.get("stage_roles"),
    }


def _hash_serialized_case_payload(
    payload: Mapping[str, Any],
) -> tuple[dict[str, str], str]:
    hashes = {
        key: _array_sha256(payload[key])
        for key in _EXTENDED_VALUE_HASH_KEYS
        if key != "stage_roles_canonical_JSON"
    }
    hashes["stage_roles_canonical_JSON"] = _sha256_bytes(
        _canonical_json_bytes(payload["stage_roles"])
    )
    payload_sha = _sha256_bytes(_canonical_json_bytes(payload))
    return hashes, payload_sha


def _expected_stage_roles(measurement_steps: int) -> list[str]:
    return [
        "entrance_prestep_full",
        *[
            role
            for _ in range(measurement_steps)
            for role in ("measured_midpoint", "measured_full")
        ],
    ]


def _rederive_stage_decision(result: Mapping[str, Any]) -> str:
    checks = result.get("checks")
    if not isinstance(checks, Mapping) or not checks:
        raise OneShotAuthorizationError("frozen result lacks checks")
    if not all(value is True for value in checks.values()):
        return "PROTOCOL-NO-GO"
    try:
        parity = result["v22_span_parity"]
        zero_sum_lower = parity["zero_sum"]["even"]["interval"]["lower"]
        zero_noncancellation_lower = parity["zero_noncancellation"]["even"][
            "interval"
        ]["lower"]
        omega_lower = parity["omega"]["even"]["interval"]["lower"]
    except (KeyError, TypeError) as error:
        raise OneShotAuthorizationError(
            "frozen result lacks v2.2 decision intervals"
        ) from error
    if float(zero_sum_lower or 0.0) > 0.0 or float(
        zero_noncancellation_lower or 0.0
    ) > 0.0:
        return "ZEROTH-ORDER NAMED-LAW OBSTRUCTION"
    if float(omega_lower or 0.0) > 0.0:
        return "FIXED-SPACE REACHABLE FIRST-ORDER OBSTRUCTION WITNESS"
    return "NO RESOLVED WITNESS"


def _validate_matched_stage_ledger(matched: Any) -> None:
    if not isinstance(matched, Mapping):
        raise OneShotAuthorizationError(
            "matched-stage parity ledger must be a mapping"
        )
    entries = matched.get("entries")
    if (
        set(matched) != {"passed", "entry_count", "failed", "entries"}
        or not isinstance(entries, list)
        or matched.get("entry_count") != 393
        or len(entries) != 393
        or not isinstance(matched.get("failed"), list)
        or not isinstance(matched.get("passed"), bool)
    ):
        raise OneShotAuthorizationError(
            "matched-stage parity accounting must contain 393 entries"
        )
    groups = {"signed_positive", "signed_negative", "zero"}
    observable_counts = {
        "canonical_material_current_trace": 33,
        "stored_weak_pressure": 33,
        "direct_weak_pressure": 33,
        "stored_step_residual": 16,
        "direct_step_residual": 16,
    }
    expected_identities = {
        (group, observable, index)
        for group in groups
        for observable, count in observable_counts.items()
        for index in range(count)
    }
    observed_identities: set[tuple[str, str, int]] = set()
    recomputed_failed: list[dict[str, Any]] = []
    for entry in entries:
        if (
            not isinstance(entry, Mapping)
            or set(entry)
            != {
                "group",
                "observable",
                "index",
                "time",
                "continuum_zero_initializer",
                "initializer_zero_target_passed",
                "decision",
            }
            or entry.get("group") not in groups
            or entry.get("observable") not in observable_counts
            or not isinstance(entry.get("index"), int)
            or isinstance(entry.get("index"), bool)
            or not 0
            <= entry["index"]
            < observable_counts[entry["observable"]]
            or not isinstance(entry.get("time"), (int, float))
            or isinstance(entry.get("time"), bool)
            or not np.isfinite(float(entry["time"]))
            or not isinstance(
                entry.get("continuum_zero_initializer"), bool
            )
            or not isinstance(
                entry.get("initializer_zero_target_passed"), bool
            )
            or not isinstance(entry.get("decision"), Mapping)
            or not isinstance(
                entry["decision"].get("protocol_no_go"), bool
            )
            or not isinstance(entry["decision"].get("reasons"), list)
        ):
            raise OneShotAuthorizationError(
                "matched-stage parity entry schema drifted"
            )
        identity = (
            entry["group"],
            entry["observable"],
            entry["index"],
        )
        if identity in observed_identities:
            raise OneShotAuthorizationError(
                "matched-stage parity entry identity is duplicated"
            )
        observed_identities.add(identity)
        initializer = bool(
            entry["observable"]
            == "canonical_material_current_trace"
            and entry["index"] == 0
        )
        if entry["continuum_zero_initializer"] is not initializer:
            raise OneShotAuthorizationError(
                "matched-stage continuum-zero initializer identity drifted"
            )
        if (
            entry["decision"]["protocol_no_go"]
            or not entry["initializer_zero_target_passed"]
        ):
            recomputed_failed.append(
                {
                    "group": entry["group"],
                    "observable": entry["observable"],
                    "index": entry["index"],
                    "time": entry["time"],
                    "reasons": entry["decision"]["reasons"],
                    "initializer_zero_target_passed": entry[
                        "initializer_zero_target_passed"
                    ],
                }
            )
    if observed_identities != expected_identities:
        raise OneShotAuthorizationError(
            "matched-stage parity identity set is incomplete"
        )
    if (
        matched["failed"] != recomputed_failed
        or matched["passed"] is not (not recomputed_failed)
    ):
        raise OneShotAuthorizationError(
            "matched-stage parity failed/pass ledger is inconsistent"
        )


def _validate_frozen_result(
    result: Mapping[str, Any],
    observations: Mapping[str, Any],
    *,
    contract: Mapping[str, Any],
    ordered_case_names: Sequence[str],
    source_fingerprints: Mapping[str, str],
) -> None:
    required_top = {
        "stage",
        "protocol_version",
        "contract_file",
        "definition_chain",
        "generated_at_utc",
        "code_fingerprints",
        "stage_decision",
        "production_activation_allowed",
        "fixed_space_only",
        "checks",
        "execution_accounting",
        "negative_controls",
        "v22_span_parity",
        "omega",
        "zero_sum",
        "zero_noncancellation",
        "cases",
        "nonclaims",
    }
    if set(result) != required_top:
        raise OneShotAuthorizationError(
            "frozen aggregate top-level schema drifted"
        )
    if (
        result.get("protocol_version") != "S3ai-v2.2"
        or result.get("definition_chain") != contract.get(
            "_definition_chain"
        )
        or result.get("production_activation_allowed") is not False
        or result.get("fixed_space_only") is not True
        or result.get("stage_decision") not in _VALID_DECISIONS
        or result.get("code_fingerprints") != dict(source_fingerprints)
        or result.get("execution_accounting") != _EXPECTED_ACCOUNTING
    ):
        raise OneShotAuthorizationError(
            "frozen aggregate identity/accounting/production boundary drifted"
        )
    if result["stage_decision"] != _rederive_stage_decision(result):
        raise OneShotAuthorizationError(
            "frozen aggregate stage decision is inconsistent with its guards"
        )
    cases = result.get("cases")
    if (
        not isinstance(cases, Mapping)
        or list(cases) != list(ordered_case_names)
        or list(observations) != list(ordered_case_names)
    ):
        raise OneShotAuthorizationError(
            "frozen aggregate does not contain the ordered 31-case registry"
        )
    parity = result.get("v22_span_parity", {})
    matched = (
        parity.get("matched_stage_q_families", {})
        if isinstance(parity, Mapping)
        else {}
    )
    _validate_matched_stage_ledger(matched)

    for name in ordered_case_names:
        observation = observations[name]
        case = cases[name]
        if not isinstance(case, Mapping):
            raise OneShotAuthorizationError(f"case {name} is not a mapping")
        observed_case = observation.case
        if (
            observed_case.name != name
            or case.get("configuration") != observed_case.configuration
            or case.get("role") != observed_case.role
            or case.get("epsilon_signed") != observed_case.epsilon_signed
            or case.get("timestep") != observed_case.timestep
            or case.get("quadrature_order")
            != observed_case.quadrature_order
            or case.get("case_identity")
            != guard._case_identity_payload(observed_case)
            or case.get("measurement_steps")
            != observed_case.measurement_steps
            or case.get("observed_stages")
            != observed_case.observed_stages
        ):
            raise OneShotAuthorizationError(
                f"case {name} frozen configuration identity drifted"
            )
        steps = int(case.get("measurement_steps", -1))
        stages = int(case.get("observed_stages", -1))
        payload = _observation_payload(observation)
        expected_shapes = {
            "mass_active": (7, 7),
            "stage_times": (stages,),
            "measurement_times": (steps, 3),
            "stored_window_residual": (7,),
            "direct_window_residual": (7,),
            "stored_step_residuals": (steps, 7),
            "direct_step_residuals": (steps, 7),
            "canonical_material_current_trace": (stages, 9),
            "body_cut_trace": (stages, 9),
            "canonical_material_release": (stages, 9),
            "representation_inventory": (stages, 9),
            "stored_weak_pressure": (stages, 7),
            "direct_weak_pressure": (stages, 7),
        }
        for key, shape in expected_shapes.items():
            array = np.asarray(payload[key], dtype=float)
            if array.shape != shape or not np.all(np.isfinite(array)):
                raise OneShotAuthorizationError(
                    f"case {name} field {key} has invalid shape or value"
                )
        if stages != 1 + 2 * steps or payload["stage_roles"] != (
            _expected_stage_roles(steps)
        ):
            raise OneShotAuthorizationError(
                f"case {name} stage roles/accounting drifted"
            )
        if not np.array_equal(
            payload["stored_window_residual"],
            np.sum(payload["stored_step_residuals"], axis=0),
        ) or not np.array_equal(
            payload["direct_window_residual"],
            np.sum(payload["direct_step_residuals"], axis=0),
        ):
            raise OneShotAuthorizationError(
                f"case {name} window/step residual ledger drifted"
            )
        frozen_hashes = case.get("value_hashes")
        if (
            not isinstance(frozen_hashes, Mapping)
            or set(frozen_hashes) != _FROZEN_VALUE_HASH_KEYS
            or frozen_hashes["mass_active"]
            != _array_sha256(payload["mass_active"])
            or frozen_hashes["stored_step_residuals"]
            != _array_sha256(payload["stored_step_residuals"])
            or frozen_hashes["direct_step_residuals"]
            != _array_sha256(payload["direct_step_residuals"])
            or frozen_hashes["stage_times"]
            != _array_sha256(payload["stage_times"])
            or frozen_hashes["canonical_material_current_trace"]
            != _array_sha256(
                payload["canonical_material_current_trace"]
            )
        ):
            raise OneShotAuthorizationError(
                f"case {name} frozen value hashes drifted"
            )
        diagnostics = case.get("diagnostics")
        if (
            not isinstance(diagnostics, Mapping)
            or "input_mutation_abs_max" not in diagnostics
            or "old_state_mutation_abs_max" not in diagnostics
        ):
            raise OneShotAuthorizationError(
                f"case {name} lacks immutable-input/state diagnostics"
            )


def _augment_result(
    result: dict[str, Any],
    observations: Mapping[str, Any],
    *,
    ordered_case_names: Sequence[str],
    verified: _VerifiedAuthorization,
    registry_manifest_sha256: str,
    source_fingerprints_start: Mapping[str, str],
    source_fingerprints_end: Mapping[str, str],
    stable_runtime_start: Mapping[str, Any],
    stable_runtime_end: Mapping[str, Any],
    stable_execution_input_sha256_start: str,
    stable_execution_input_sha256_end: str,
    dependency_closure: Mapping[str, Any],
    marker_receipt_sha256: str,
) -> dict[str, Any]:
    _assert_verified_authorization_memory_integrity(verified)
    case_payloads: dict[str, Any] = {}
    for name in ordered_case_names:
        observation = observations[name]
        payload = _observation_payload(observation)
        hashes = _extended_observation_hashes(observation)
        payload_sha = _sha256_bytes(_canonical_json_bytes(payload))
        case = result["cases"][name]
        case["mass_active"] = payload["mass_active"]
        case["extended_value_hashes"] = hashes
        case["observation_payload_sha256"] = payload_sha
        case_payloads[name] = payload
    result["ordered_case_names"] = list(ordered_case_names)
    result["observation_payload_sha256"] = _sha256_bytes(
        _canonical_json_bytes(case_payloads)
    )
    second_audit = verified.payload["second_bounded_audit"]
    clearance_review = verified.clearance["review"]
    clearance_bindings = verified.clearance["bindings"]
    if (
        clearance_bindings["ordered_registry_manifest_sha256"]
        != registry_manifest_sha256
    ):
        raise OneShotAuthorizationError(
            "clearance registry binding differs before result augmentation"
        )
    result["one_shot_provenance"] = {
        "scientific_protocol_version": "S3ai-v2.2",
        "transport_protocol_version": _TRANSPORT_PROTOCOL_VERSION,
        "attempt_id": _ATTEMPT_ID,
        "assurance_profile": _ASSURANCE_PROFILE,
        "wrapper_definition": {
            "path": _validated_repo_relative(
                WRAPPER_DEFINITION_PATH
            ).as_posix(),
            "sha256": verified.definition_sha256,
        },
        "transport_preregistration": {
            "path": _validated_repo_relative(
                TRANSPORT_PREREGISTRATION_PATH
            ).as_posix(),
            "sha256": verified.transport_preregistration_sha256,
        },
        "wrapper_source": {
            "path": _WRAPPER_RELATIVE,
            "sha256": source_fingerprints_start[_WRAPPER_RELATIVE],
        },
        "authorization": {
            "id": verified.payload["authorization"]["id"],
            "path": _validated_repo_relative(
                AUTHORIZATION_PATH
            ).as_posix(),
            "raw_sha256": verified.raw_sha256,
            "canonical_sha256": verified.canonical_sha256,
            "single_use_token_sha256": verified.token_sha256,
            "dependency_manifest_raw_sha256": (
                verified.dependency_manifest_raw_sha256
            ),
            "dependency_manifest_canonical_sha256": (
                verified.dependency_manifest_canonical_sha256
            ),
        },
        "second_bounded_audit": {
            "verdict": verified.clearance["verdict"],
            "review": {
                name: clearance_review[name]
                for name in (
                    "actual_independence",
                    "provider",
                    "model",
                    "model_family",
                    "agent_id",
                    "trace_id",
                )
            },
            "issuance_request": {
                name: verified.payload["issuance_request"][name]
                for name in (
                    "path",
                    "raw_sha256",
                    "canonical_sha256",
                    "ticket_semantic_projection_sha256",
                )
            },
            "request": {
                "path": second_audit["request_path"],
                "raw_sha256": second_audit["request_raw_sha256"],
                "canonical_sha256": second_audit[
                    "request_canonical_sha256"
                ],
            },
            "response": {
                "path": second_audit["response_path"],
                "raw_sha256": second_audit["response_raw_sha256"],
                "canonical_sha256": second_audit[
                    "response_canonical_sha256"
                ],
            },
            "invocation_metadata": {
                "path": second_audit["invocation_metadata_path"],
                "raw_sha256": second_audit[
                    "invocation_metadata_raw_sha256"
                ],
                "canonical_sha256": second_audit[
                    "invocation_metadata_canonical_sha256"
                ],
            },
            "trace_metadata": {
                "path": second_audit["trace_metadata_path"],
                "raw_sha256": second_audit[
                    "trace_metadata_raw_sha256"
                ],
            },
            "clearance_canonical_sha256": second_audit[
                "clearance_canonical_sha256"
            ],
        },
        "ordered_registry_manifest_sha256": registry_manifest_sha256,
        "source_fingerprints_start": dict(source_fingerprints_start),
        "source_fingerprints_end": dict(source_fingerprints_end),
        "stable_runtime_identity_start": dict(stable_runtime_start),
        "stable_runtime_identity_end": dict(stable_runtime_end),
        "stable_execution_input_sha256_start": (
            stable_execution_input_sha256_start
        ),
        "stable_execution_input_sha256_end": (
            stable_execution_input_sha256_end
        ),
        "dependency_closure": dict(dependency_closure),
        "attempt_marker": {
            "path": _validated_repo_relative(
                ATTEMPT_MARKER_PATH
            ).as_posix(),
            "receipt_sha256": marker_receipt_sha256,
            "permanent": True,
        },
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "latest_pointer_written": False,
    }
    scientific = dict(result)
    scientific.pop("generated_at_utc", None)
    scientific.pop("code_fingerprints", None)
    scientific.pop("one_shot_provenance", None)
    scientific.pop("scientific_payload_sha256", None)
    result["scientific_payload_sha256"] = _sha256_bytes(
        _canonical_json_bytes(scientific)
    )
    return result


def _validate_serialized_result(
    payload: Mapping[str, Any],
    *,
    ordered_case_names: Sequence[str],
    expected_dependency_manifest: DependencyManifest,
    expected_verified: _VerifiedAuthorization,
) -> None:
    _assert_verified_authorization_memory_integrity(expected_verified)
    frozen_cases = guard.frozen_history_cases()
    frozen_names = [case.name for case in frozen_cases]
    if (
        payload.get("protocol_version") != "S3ai-v2.2"
        or payload.get("production_activation_allowed") is not False
        or payload.get("fixed_space_only") is not True
        or payload.get("execution_accounting") != _EXPECTED_ACCOUNTING
        or payload.get("stage_decision") not in _VALID_DECISIONS
        or payload.get("stage_decision") != _rederive_stage_decision(payload)
        or not isinstance(payload.get("code_fingerprints"), Mapping)
        or len(payload["code_fingerprints"]) != 63
        or payload.get("ordered_case_names") != list(ordered_case_names)
        or list(ordered_case_names) != frozen_names
    ):
        raise OneShotAuthorizationError(
            "serialized result identity, accounting, decision, or explicit "
            "case order drifted"
        )
    cases = payload.get("cases")
    if not isinstance(cases, Mapping) or set(cases) != set(ordered_case_names):
        raise OneShotAuthorizationError(
            "serialized result case registry drifted"
        )
    all_payloads: dict[str, Any] = {}
    for name, frozen_case in zip(
        ordered_case_names, frozen_cases, strict=True
    ):
        case = cases[name]
        if not isinstance(case, Mapping):
            raise OneShotAuthorizationError(
                f"serialized case {name} is invalid"
            )
        if (
            case.get("configuration") != frozen_case.configuration
            or case.get("role") != frozen_case.role
            or case.get("epsilon_signed") != frozen_case.epsilon_signed
            or case.get("timestep") != frozen_case.timestep
            or case.get("quadrature_order")
            != frozen_case.quadrature_order
            or case.get("case_identity")
            != guard._case_identity_payload(frozen_case)
            or case.get("measurement_steps")
            != frozen_case.measurement_steps
            or case.get("observed_stages")
            != frozen_case.observed_stages
        ):
            raise OneShotAuthorizationError(
                f"serialized case {name} frozen configuration drifted"
            )
        raw_payload = _case_payload_from_serialized(case)
        steps = case.get("measurement_steps")
        stages = case.get("observed_stages")
        if (
            not isinstance(steps, int)
            or isinstance(steps, bool)
            or steps not in {4, 8, 16}
            or not isinstance(stages, int)
            or isinstance(stages, bool)
            or stages != 1 + 2 * steps
        ):
            raise OneShotAuthorizationError(
                f"serialized case {name} stage accounting drifted"
            )
        expected_shapes = {
            "mass_active": (7, 7),
            "stage_times": (stages,),
            "measurement_times": (steps, 3),
            "stored_window_residual": (7,),
            "direct_window_residual": (7,),
            "stored_step_residuals": (steps, 7),
            "direct_step_residuals": (steps, 7),
            "canonical_material_current_trace": (stages, 9),
            "body_cut_trace": (stages, 9),
            "canonical_material_release": (stages, 9),
            "representation_inventory": (stages, 9),
            "stored_weak_pressure": (stages, 7),
            "direct_weak_pressure": (stages, 7),
        }
        arrays: dict[str, np.ndarray] = {}
        try:
            for key, expected_shape in expected_shapes.items():
                array = np.asarray(raw_payload[key], dtype=float)
                if (
                    array.shape != expected_shape
                    or not np.all(np.isfinite(array))
                ):
                    raise OneShotAuthorizationError(
                        f"serialized case {name} field {key} has invalid "
                        "shape or nonfinite values"
                    )
                arrays[key] = array
        except (TypeError, ValueError) as error:
            raise OneShotAuthorizationError(
                f"serialized case {name} contains a nonnumeric array"
            ) from error
        try:
            strict_mass = guard._strict_active_mass(arrays["mass_active"])
        except guard.ReachablePressureV2GuardError as error:
            raise OneShotAuthorizationError(
                f"serialized case {name} active mass is invalid"
            ) from error
        if (
            not np.array_equal(strict_mass, arrays["mass_active"])
            or raw_payload.get("stage_roles")
            != _expected_stage_roles(steps)
            or not np.array_equal(
                arrays["stored_window_residual"],
                np.sum(arrays["stored_step_residuals"], axis=0),
            )
            or not np.array_equal(
                arrays["direct_window_residual"],
                np.sum(arrays["direct_step_residuals"], axis=0),
            )
        ):
            raise OneShotAuthorizationError(
                f"serialized case {name} mass, roles, or residual ledger "
                "drifted"
            )
        hashes, payload_sha = _hash_serialized_case_payload(raw_payload)
        frozen_hashes = case.get("value_hashes")
        if (
            case.get("extended_value_hashes") != hashes
            or case.get("observation_payload_sha256") != payload_sha
            or not isinstance(frozen_hashes, Mapping)
            or set(frozen_hashes) != _FROZEN_VALUE_HASH_KEYS
            or frozen_hashes["mass_active"] != hashes["mass_active"]
            or frozen_hashes["stored_step_residuals"]
            != hashes["stored_step_residuals"]
            or frozen_hashes["direct_step_residuals"]
            != hashes["direct_step_residuals"]
            or frozen_hashes["stage_times"] != hashes["stage_times"]
            or frozen_hashes["canonical_material_current_trace"]
            != hashes["canonical_material_current_trace"]
        ):
            raise OneShotAuthorizationError(
                f"serialized case {name} value hashes do not recompute"
            )
        all_payloads[name] = raw_payload
    if payload.get("observation_payload_sha256") != _sha256_bytes(
        _canonical_json_bytes(all_payloads)
    ):
        raise OneShotAuthorizationError(
            "serialized observation payload digest does not recompute"
        )
    provenance = payload.get("one_shot_provenance")
    if not isinstance(provenance, Mapping):
        raise OneShotAuthorizationError(
            "serialized result lacks one-shot provenance"
        )
    expected_provenance_keys = {
        "scientific_protocol_version",
        "transport_protocol_version",
        "attempt_id",
        "assurance_profile",
        "wrapper_definition",
        "transport_preregistration",
        "wrapper_source",
        "authorization",
        "second_bounded_audit",
        "ordered_registry_manifest_sha256",
        "source_fingerprints_start",
        "source_fingerprints_end",
        "stable_runtime_identity_start",
        "stable_runtime_identity_end",
        "stable_execution_input_sha256_start",
        "stable_execution_input_sha256_end",
        "dependency_closure",
        "attempt_marker",
        "completed_at_utc",
        "latest_pointer_written",
    }
    if set(provenance) != expected_provenance_keys:
        raise OneShotAuthorizationError(
            "serialized v2.3 provenance schema drifted"
        )
    sources_start = provenance.get("source_fingerprints_start")
    sources_end = provenance.get("source_fingerprints_end")
    runtime_start = provenance.get("stable_runtime_identity_start")
    runtime_end = provenance.get("stable_runtime_identity_end")
    execution_input_start = provenance.get(
        "stable_execution_input_sha256_start"
    )
    execution_input_end = provenance.get(
        "stable_execution_input_sha256_end"
    )
    wrapper_definition = provenance.get("wrapper_definition", {})
    transport_preregistration = provenance.get(
        "transport_preregistration", {}
    )
    wrapper_source = provenance.get("wrapper_source", {})
    authorization = provenance.get("authorization", {})
    bounded_audit = provenance.get("second_bounded_audit", {})
    marker = provenance.get("attempt_marker", {})
    registry_manifest = _sha256_bytes(
        _canonical_json_bytes(
            [
                {
                    "name": case.name,
                    "case_identity_sha256": guard._case_identity_payload(
                        case
                    )["sha256"],
                }
                for case in frozen_cases
            ]
        )
    )
    _validate_dependency_closure_provenance(
        provenance.get("dependency_closure"),
        expected_manifest=expected_dependency_manifest,
    )
    expected_authorization_raw_sha256 = _validate_sha256(
        expected_verified.raw_sha256,
        label="expected authorization raw digest",
    )
    dependency_record = provenance["dependency_closure"]["manifest"]
    expected_second_audit_source = expected_verified.payload.get(
        "second_bounded_audit"
    )
    if not isinstance(expected_second_audit_source, Mapping):
        raise OneShotAuthorizationError(
            "verified authorization lacks its bounded-audit binding"
        )
    expected_review = expected_verified.clearance["review"]
    expected_bounded_audit = {
        "verdict": expected_verified.clearance["verdict"],
        "review": {
            name: expected_review[name]
            for name in (
                "actual_independence",
                "provider",
                "model",
                "model_family",
                "agent_id",
                "trace_id",
            )
        },
        "issuance_request": {
            name: expected_verified.payload["issuance_request"][name]
            for name in (
                "path",
                "raw_sha256",
                "canonical_sha256",
                "ticket_semantic_projection_sha256",
            )
        },
        "request": {
            "path": expected_second_audit_source["request_path"],
            "raw_sha256": expected_second_audit_source[
                "request_raw_sha256"
            ],
            "canonical_sha256": expected_second_audit_source[
                "request_canonical_sha256"
            ],
        },
        "response": {
            "path": expected_second_audit_source["response_path"],
            "raw_sha256": expected_second_audit_source[
                "response_raw_sha256"
            ],
            "canonical_sha256": expected_second_audit_source[
                "response_canonical_sha256"
            ],
        },
        "invocation_metadata": {
            "path": expected_second_audit_source[
                "invocation_metadata_path"
            ],
            "raw_sha256": expected_second_audit_source[
                "invocation_metadata_raw_sha256"
            ],
            "canonical_sha256": expected_second_audit_source[
                "invocation_metadata_canonical_sha256"
            ],
        },
        "trace_metadata": {
            "path": expected_second_audit_source[
                "trace_metadata_path"
            ],
            "raw_sha256": expected_second_audit_source[
                "trace_metadata_raw_sha256"
            ],
        },
        "clearance_canonical_sha256": expected_second_audit_source[
            "clearance_canonical_sha256"
        ],
    }
    if (
        provenance.get("scientific_protocol_version") != "S3ai-v2.2"
        or provenance.get("transport_protocol_version")
        != _TRANSPORT_PROTOCOL_VERSION
        or provenance.get("attempt_id") != _ATTEMPT_ID
        or provenance.get("assurance_profile") != _ASSURANCE_PROFILE
        or not isinstance(sources_start, Mapping)
        or not isinstance(sources_end, Mapping)
        or len(sources_start) != 63
        or dict(sources_start) != dict(sources_end)
        or dict(sources_start) != dict(payload["code_fingerprints"])
        or not all(
            isinstance(path, str)
            and _validate_sha256(
                digest, label=f"serialized source {path}"
            )
            == digest
            for path, digest in sources_start.items()
        )
        or not isinstance(wrapper_source, Mapping)
        or wrapper_source.get("path") != _WRAPPER_RELATIVE
        or wrapper_source.get("sha256")
        != sources_start.get(_WRAPPER_RELATIVE)
        or not isinstance(wrapper_definition, Mapping)
        or wrapper_definition.get("path")
        != _validated_repo_relative(WRAPPER_DEFINITION_PATH).as_posix()
        or _validate_sha256(
            wrapper_definition.get("sha256"),
            label="serialized frozen local-source definition",
        )
        != wrapper_definition.get("sha256")
        or wrapper_definition.get("sha256")
        != expected_verified.definition_sha256
        or not isinstance(transport_preregistration, Mapping)
        or transport_preregistration.get("path")
        != _validated_repo_relative(
            TRANSPORT_PREREGISTRATION_PATH
        ).as_posix()
        or _validate_sha256(
            transport_preregistration.get("sha256"),
            label="serialized transport preregistration",
        )
        != transport_preregistration.get("sha256")
        or transport_preregistration.get("sha256")
        != expected_verified.transport_preregistration_sha256
        or not isinstance(runtime_start, Mapping)
        or not isinstance(runtime_end, Mapping)
        or dict(runtime_start) != dict(runtime_end)
        or any(
            key in runtime_start
            for key in (
                "loaded_native_binary_fingerprints",
                "loaded_python_dependency_fingerprints",
                "dependency_closure_snapshot",
                "dependency_snapshot",
            )
        )
        or _validate_sha256(
            execution_input_start,
            label="serialized execution input start",
        )
        != execution_input_start
        or execution_input_start != execution_input_end
        or provenance.get("ordered_registry_manifest_sha256")
        != registry_manifest
        or provenance.get("ordered_registry_manifest_sha256")
        != expected_verified.clearance["bindings"][
            "ordered_registry_manifest_sha256"
        ]
        or not isinstance(authorization, Mapping)
        or set(authorization)
        != {
            "id",
            "path",
            "raw_sha256",
            "canonical_sha256",
            "single_use_token_sha256",
            "dependency_manifest_raw_sha256",
            "dependency_manifest_canonical_sha256",
        }
        or not isinstance(authorization.get("id"), str)
        or not authorization.get("id")
        or authorization.get("id") == _OLD_AUTHORIZATION_ID
        or authorization.get("path")
        != _validated_repo_relative(AUTHORIZATION_PATH).as_posix()
        or authorization.get("raw_sha256")
        == _OLD_AUTHORIZATION_SHA256
        or authorization.get("raw_sha256")
        != expected_authorization_raw_sha256
        or authorization.get("id")
        != expected_verified.payload["authorization"]["id"]
        or authorization.get("canonical_sha256")
        != expected_verified.canonical_sha256
        or authorization.get("single_use_token_sha256")
        != expected_verified.token_sha256
        or authorization.get("dependency_manifest_raw_sha256")
        != expected_dependency_manifest.raw_sha256
        or authorization.get("dependency_manifest_canonical_sha256")
        != expected_dependency_manifest.canonical_sha256
        or authorization.get("dependency_manifest_raw_sha256")
        != dependency_record.get("raw_sha256")
        or authorization.get("dependency_manifest_canonical_sha256")
        != dependency_record.get("canonical_sha256")
        or authorization.get("single_use_token_sha256")
        == _OLD_TOKEN_SHA256
        or any(
            _validate_sha256(
                authorization.get(key),
                label=f"serialized authorization {key}",
            )
            != authorization.get(key)
            for key in (
                "raw_sha256",
                "canonical_sha256",
                "single_use_token_sha256",
                "dependency_manifest_raw_sha256",
                "dependency_manifest_canonical_sha256",
            )
        )
        or not isinstance(bounded_audit, Mapping)
        or dict(bounded_audit) != expected_bounded_audit
        or any(
            _validate_sha256(value, label=label) != value
            for label, value in (
                (
                    "serialized issuance request raw SHA",
                    bounded_audit.get("issuance_request", {}).get(
                        "raw_sha256"
                    ),
                ),
                (
                    "serialized issuance request canonical SHA",
                    bounded_audit.get("issuance_request", {}).get(
                        "canonical_sha256"
                    ),
                ),
                (
                    "serialized issuance projection SHA",
                    bounded_audit.get("issuance_request", {}).get(
                        "ticket_semantic_projection_sha256"
                    ),
                ),
                (
                    "serialized request raw SHA",
                    bounded_audit.get("request", {}).get("raw_sha256"),
                ),
                (
                    "serialized request canonical SHA",
                    bounded_audit.get("request", {}).get(
                        "canonical_sha256"
                    ),
                ),
                (
                    "serialized response raw SHA",
                    bounded_audit.get("response", {}).get("raw_sha256"),
                ),
                (
                    "serialized response canonical SHA",
                    bounded_audit.get("response", {}).get(
                        "canonical_sha256"
                    ),
                ),
                (
                    "serialized invocation raw SHA",
                    bounded_audit.get("invocation_metadata", {}).get(
                        "raw_sha256"
                    ),
                ),
                (
                    "serialized invocation canonical SHA",
                    bounded_audit.get("invocation_metadata", {}).get(
                        "canonical_sha256"
                    ),
                ),
                (
                    "serialized trace raw SHA",
                    bounded_audit.get("trace_metadata", {}).get(
                        "raw_sha256"
                    ),
                ),
                (
                    "serialized clearance canonical SHA",
                    bounded_audit.get("clearance_canonical_sha256"),
                ),
            )
        )
        or not isinstance(marker, Mapping)
        or marker.get("path")
        != _validated_repo_relative(ATTEMPT_MARKER_PATH).as_posix()
        or marker.get("permanent") is not True
        or _validate_sha256(
            marker.get("receipt_sha256"),
            label="serialized marker receipt",
        )
        != marker.get("receipt_sha256")
        or provenance.get("latest_pointer_written") is not False
    ):
        raise OneShotAuthorizationError(
            "serialized source/runtime/input/registry provenance is "
            "internally inconsistent"
        )
    scientific = dict(payload)
    declared = scientific.pop("scientific_payload_sha256", None)
    scientific.pop("generated_at_utc", None)
    scientific.pop("code_fingerprints", None)
    scientific.pop("one_shot_provenance", None)
    if declared != _sha256_bytes(_canonical_json_bytes(scientific)):
        raise OneShotAuthorizationError(
            "serialized scientific payload digest does not recompute"
        )
    matched = payload.get("v22_span_parity", {}).get(
        "matched_stage_q_families", {}
    )
    _validate_matched_stage_ledger(matched)


def _atomic_create_json_no_replace(
    path: Path,
    value: Any,
    *,
    output_directory: _OutputDirectoryLease | None = None,
    marker_path: Path | None = None,
    attempt_claim: _AttemptClaim | None = None,
    pre_link_gate: Any = None,
) -> str:
    if pre_link_gate is not None and not callable(pre_link_gate):
        raise OneShotAuthorizationError(
            "atomic publication pre-link gate must be callable"
        )
    payload = _pretty_json_bytes(value)
    try:
        reparsed = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise OneShotAuthorizationError(
            "canonical result failed its JSON round-trip"
        ) from error
    if _canonical_json_bytes(reparsed) != _canonical_json_bytes(value):
        raise OneShotAuthorizationError(
            "canonical result changed across JSON round-trip"
        )
    if output_directory is None:
        parent_fd, name = _open_parent_dir_nofollow(path)
        owns_parent = True
    else:
        if marker_path is None:
            raise OneShotAuthorizationError(
                "leased result publication requires the marker path"
            )
        _verify_output_directory_lease(
            output_directory,
            marker_path=marker_path,
            result_path=path,
        )
        parent_fd = output_directory.fd
        name = output_directory.result_name
        owns_parent = False
    temporary_name = (
        f".{name}.tmp.{os.getpid()}.{secrets.token_hex(16)}"
    )
    fd = -1
    linked = False
    publication_committed = False
    temporary_identity: tuple[int, int, int] | None = None
    result_sha256 = _sha256_bytes(payload)
    try:
        fd = os.open(
            temporary_name,
            os.O_RDWR | os.O_CREAT | os.O_EXCL | _O_NOFOLLOW,
            0o444,
            dir_fd=parent_fd,
        )
        _write_all(fd, payload)
        os.fsync(fd)
        if _pread_exact(fd, len(payload) + 1) != payload:
            raise OneShotAuthorizationError(
                "temporary result bytes changed before publication"
            )
        temporary_metadata = os.fstat(fd)
        temporary_identity = (
            int(temporary_metadata.st_dev),
            int(temporary_metadata.st_ino),
            int(temporary_metadata.st_size),
        )
        if attempt_claim is not None:
            if output_directory is None or marker_path is None:
                raise OneShotAuthorizationError(
                    "attempt identity requires leased result publication"
                )
            _verify_permanent_attempt_marker(
                marker_path,
                attempt_claim.receipt_sha256,
                output_directory=output_directory,
                result_path=path,
                expected_identity=attempt_claim,
            )
        if pre_link_gate is not None:
            pre_link_gate()
        try:
            os.link(
                temporary_name,
                name,
                src_dir_fd=parent_fd,
                dst_dir_fd=parent_fd,
                follow_symlinks=False,
            )
        except FileExistsError as error:
            raise OneShotAuthorizationError(
                "canonical result appeared concurrently; no overwrite "
                "was performed"
            ) from error
        linked = True
        published_metadata = os.stat(
            name, dir_fd=parent_fd, follow_symlinks=False
        )
        if (
            int(published_metadata.st_dev),
            int(published_metadata.st_ino),
            int(published_metadata.st_size),
        ) != temporary_identity:
            raise OneShotAuthorizationError(
                "canonical result does not name the verified temporary inode"
            )
        os.fsync(parent_fd)
        os.unlink(temporary_name, dir_fd=parent_fd)
        os.fsync(parent_fd)
        temporary_name = ""
        if output_directory is not None:
            _verify_output_directory_lease(
                output_directory,
                marker_path=marker_path,
                result_path=path,
            )
        if fd >= 0:
            os.close(fd)
            fd = -1
        publication_committed = True
        return result_sha256
    except OSError as error:
        raise OneShotAuthorizationError(
            f"atomic no-replace result publication failed: {error}"
        ) from error
    finally:
        if fd >= 0:
            try:
                os.close(fd)
            except OSError:
                pass
        if linked and not publication_committed:
            try:
                published = os.stat(
                    name, dir_fd=parent_fd, follow_symlinks=False
                )
                published_identity = (
                    int(published.st_dev),
                    int(published.st_ino),
                    int(published.st_size),
                )
                if (
                    temporary_identity is None
                    or published_identity != temporary_identity
                ):
                    raise OneShotAuthorizationError(
                        "failed publication no longer names the owned "
                        "temporary inode; refusing to delete another result"
                    )
                os.unlink(name, dir_fd=parent_fd)
                os.fsync(parent_fd)
                linked = False
            except FileNotFoundError:
                linked = False
            except OSError as rollback_error:
                raise OneShotAuthorizationError(
                    "failed atomic publication could not roll back its own "
                    f"canonical link: {rollback_error}"
                ) from rollback_error
        if temporary_name:
            try:
                os.unlink(temporary_name, dir_fd=parent_fd)
                os.fsync(parent_fd)
            except FileNotFoundError:
                pass
            except OSError:
                # The permanent marker records the consumed attempt.  Never
                # risk replacing the canonical result while cleaning a temp.
                pass
        if owns_parent:
            try:
                os.close(parent_fd)
            except OSError:
                pass


def _reverify_v23_identity_projection_for_execution(
    *,
    verified: _VerifiedAuthorization,
    ticket_raw: bytes,
    namespace_profile: NamespaceProfileV1 | None = None,
) -> dict[str, Any]:
    """Rebuild both identity classes from their no-follow source bytes."""

    _assert_verified_authorization_memory_integrity(verified)
    ticket = _validate_v23_ticket_static(
        _strict_pretty_json_object(
            ticket_raw, label="execution-snapshot authorization ticket"
        )
    )
    if (
        ticket != verified.payload
        or ticket["authorization"]["canonical_sha256"]
        != verified.canonical_sha256
        or _canonical_authorization_digest(ticket)
        != verified.canonical_sha256
        or ticket["authorization"]["single_use_token_sha256"]
        != verified.token_sha256
    ):
        raise OneShotAuthorizationError(
            "execution-snapshot ticket semantics differ from the accepted "
            "private-seam receipt"
        )
    owns_profile = namespace_profile is None
    profile = (
        _production_namespace_profile_v1()
        if namespace_profile is None
        else namespace_profile
    )
    try:
        _validate_namespace_profile_v1(profile)
        if ticket["namespace_profile"] != profile.mode:
            raise OneShotAuthorizationError(
                "execution-snapshot ticket/profile mode drifted"
            )
        _validate_v23_result_scope(
            profile, ticket["result"], ticket["scope"]
        )
        artifacts, raw_by_role, _ = _observe_v23_bound_artifacts(
            profile, ticket["bound_artifacts"]
        )
        core_binding = artifacts["implementation_identity_core"]
        seed = _build_identity_core_binding_seed_v1(
            profile, core_binding["sha256"]
        )
        implementation, _ = _validate_v23_identity_chain(
            profile,
            seed=seed,
            bound_artifacts=artifacts,
            raw_by_role=raw_by_role,
        )
        second = ticket["second_bounded_audit"]
        clearance_payload, _, _, clearance_canonical_sha = (
            _read_pretty_child_v1(
                profile,
                second["response_path"],
                label="execution-snapshot clearance C",
                expected_raw_sha256=second["response_raw_sha256"],
                expected_canonical_sha256=(
                    second["response_canonical_sha256"]
                ),
            )
        )
        clearance = _validate_v23_c(
            clearance_payload, accepted=True
        )
        trace_payload, _, _, _ = _read_pretty_child_v1(
            profile,
            second["trace_metadata_path"],
            label="execution-snapshot post-review trace Z",
            expected_raw_sha256=second["trace_metadata_raw_sha256"],
        )
        trace = _validate_v23_z(trace_payload)
        projection = _v23_identity_semantic_projection(
            implementation_identity=implementation,
            clearance=clearance,
            post_review_trace=trace,
        )
        if (
            implementation != ticket["implementation_identity"]
            or clearance != verified.clearance
            or clearance_canonical_sha
            != second["clearance_canonical_sha256"]
            or _sha256_bytes(_canonical_json_bytes(projection))
            != verified.identity_projection_sha256
        ):
            raise OneShotAuthorizationError(
                "execution-snapshot implementation/reviewer identity "
                "projection drifted"
            )
        _validate_namespace_profile_v1(profile)
        return projection
    finally:
        if owns_profile:
            _close_namespace_profile_v1(profile)


def _stable_execution_input(
    *,
    verified: _VerifiedAuthorization,
    source_fingerprints: Mapping[str, str],
    contract: Mapping[str, Any],
    registry_manifest_sha256: str,
    dependency_manifest: DependencyManifest,
) -> dict[str, Any]:
    _assert_verified_authorization_memory_integrity(verified)
    ticket_raw_bytes = _read_regular_nofollow(
        AUTHORIZATION_PATH, label="one-shot authorization ticket"
    )
    ticket_raw = _sha256_bytes(ticket_raw_bytes)
    identity_projection = (
        _reverify_v23_identity_projection_for_execution(
            verified=verified,
            ticket_raw=ticket_raw_bytes,
        )
    )
    identity_projection_sha = _sha256_bytes(
        _canonical_json_bytes(identity_projection)
    )
    definition_raw = _sha256_bytes(
        _read_regular_nofollow(
            WRAPPER_DEFINITION_PATH,
            label="frozen v2.2 local-source wrapper definition",
        )
    )
    _, transport_preregistration_raw = _load_transport_preregistration()
    current_manifest = _load_dependency_manifest()
    frozen_files = {
        entry["file"]: _sha256_bytes(
            _read_regular_nofollow(
                PLATFORM / "docs" / "diag" / entry["file"],
                label=f"frozen definition {entry['file']}",
            )
        )
        for entry in contract["_definition_chain"].values()
    }
    audit_files = {
        path: _sha256_bytes(
            _read_regular_nofollow(
                REPO_ROOT / path, label=f"bounded audit trace {path}"
            )
        )
        for path in verified.audit_files
    }
    expected_frozen = {
        entry["file"]: entry["sha256"]
        for entry in contract["_definition_chain"].values()
    }
    if (
        ticket_raw != verified.raw_sha256
        or definition_raw != verified.definition_sha256
        or transport_preregistration_raw
        != verified.transport_preregistration_sha256
        or current_manifest.raw_sha256
        != verified.dependency_manifest_raw_sha256
        or current_manifest.canonical_sha256
        != verified.dependency_manifest_canonical_sha256
        or current_manifest.raw_sha256 != dependency_manifest.raw_sha256
        or current_manifest.canonical_sha256
        != dependency_manifest.canonical_sha256
        or frozen_files != expected_frozen
        or audit_files != dict(verified.audit_files)
        or identity_projection_sha
        != verified.identity_projection_sha256
    ):
        raise OneShotAuthorizationError(
            "an authorization, preregistration, frozen definition, or audit "
            "trace changed between verified use and execution snapshot"
        )
    return {
        "authorization_raw_sha256": ticket_raw,
        "authorization_canonical_sha256": verified.canonical_sha256,
        "frozen_local_source_definition_sha256": definition_raw,
        "transport_preregistration_sha256": (
            transport_preregistration_raw
        ),
        "dependency_manifest_raw_sha256": current_manifest.raw_sha256,
        "dependency_manifest_canonical_sha256": (
            current_manifest.canonical_sha256
        ),
        "source_fingerprints": dict(source_fingerprints),
        "frozen_definition_files": frozen_files,
        "audit_files": audit_files,
        "identity_semantic_projection": identity_projection,
        "identity_semantic_projection_sha256": identity_projection_sha,
        "ordered_registry_manifest_sha256": registry_manifest_sha256,
    }


def _require_verified_v23_identity(
    verified: _VerifiedAuthorization,
    *,
    transport_preregistration_sha256: str,
    dependency_manifest: DependencyManifest,
    source_fingerprints: Mapping[str, str],
    stable_runtime_identity: Mapping[str, Any],
) -> None:
    _assert_verified_authorization_memory_integrity(verified)
    authorization = verified.payload.get("authorization")
    scope = verified.payload.get("scope")
    if (
        not isinstance(authorization, Mapping)
        or not isinstance(authorization.get("id"), str)
        or not authorization.get("id")
        or authorization.get("id") == _OLD_AUTHORIZATION_ID
        or authorization.get("decision")
        != (
            "YES_NEW_ONE_SHOT_31_HISTORY_SUCCESSOR_AFTER_TRANSPORT_"
            "FAILURE"
        )
        or authorization.get("formal_execution_allowed") is not True
        or authorization.get("execution_limit") != 1
        or verified.raw_sha256 == _OLD_AUTHORIZATION_SHA256
        or verified.token_sha256 == _OLD_TOKEN_SHA256
        or verified.transport_preregistration_sha256
        != transport_preregistration_sha256
        or verified.dependency_manifest_raw_sha256
        != dependency_manifest.raw_sha256
        or verified.dependency_manifest_canonical_sha256
        != dependency_manifest.canonical_sha256
        or dict(verified.source_fingerprints)
        != dict(source_fingerprints)
        or dict(verified.stable_runtime_identity)
        != dict(stable_runtime_identity)
        or not isinstance(scope, Mapping)
        or scope.get("production_activation_allowed") is not False
        or scope.get("force_hp_state_ves_118_fig_allowed") is not False
        or scope.get(
            "malicious_local_writer_or_swap_restore_protection"
        )
        is not False
    ):
        raise OneShotAuthorizationError(
            "verified authorization does not bind the exact v2.3 successor, "
            "dependency closure, stable runtime, and hard nonclaims"
        )


def run_authorized_once(
    *,
    expected_authorization_sha256: str,
    expected_implementation_identity_core_raw_sha256: str,
    second_audit_token: bytes,
) -> ExecutionReceipt:
    """Public entry: bind the private runner to the real frozen collector."""

    return _run_authorized_once(
        expected_authorization_sha256=expected_authorization_sha256,
        expected_implementation_identity_core_raw_sha256=(
            expected_implementation_identity_core_raw_sha256
        ),
        second_audit_token=second_audit_token,
        collector=guard._collect_frozen_histories,
    )


def _run_authorized_once(
    *,
    expected_authorization_sha256: str,
    expected_implementation_identity_core_raw_sha256: str,
    second_audit_token: bytes,
    collector: Any,
) -> ExecutionReceipt:
    """Execute one bound transport with an explicitly supplied collector.

    The raw authorization digest and bearer-token preimage are caller inputs;
    neither is inferred from the local ticket or an environment variable.
    Definition tests must pass a fake collector here while patching the real
    collector, mesh builder, and time marcher to forbidden sentinels.
    """

    if not callable(collector):
        raise OneShotAuthorizationError(
            "the v2.3 private transport requires an explicit collector"
        )
    uses_any_formal_output_name = (
        RESULT_PATH == _FORMAL_RESULT_PATH
        or ATTEMPT_MARKER_PATH == _FORMAL_ATTEMPT_MARKER_PATH
    )
    if uses_any_formal_output_name and (
        RESULT_PATH != _FORMAL_RESULT_PATH
        or ATTEMPT_MARKER_PATH != _FORMAL_ATTEMPT_MARKER_PATH
        or collector is not guard._collect_frozen_histories
    ):
        raise OneShotAuthorizationError(
            "either canonical v2.3 output name seals the exact formal "
            "result/marker pair and frozen collector; fake collectors require "
            "a fully isolated test namespace"
        )

    # Every check in this block precedes the durable marker and collector.
    _require_absent(RESULT_PATH, label="formal result")
    _require_absent(
        ATTEMPT_MARKER_PATH, label="permanent attempt marker"
    )
    old_quarantine_start = _verify_old_failure_quarantine()
    _reject_retired_authorization_inputs(
        expected_authorization_sha256=expected_authorization_sha256,
        second_audit_token=second_audit_token,
    )
    _, transport_preregistration_sha = (
        _load_transport_preregistration()
    )
    dependency_manifest = _load_dependency_manifest()
    definition, definition_sha = _load_wrapper_definition()
    _require_source_execution_mode()
    _verify_loaded_local_modules_use_source(definition)
    sources_start = _source_fingerprints(definition)
    stable_runtime_start = _stable_runtime_identity()
    verified = _load_and_verify_authorization(
        expected_authorization_sha256=expected_authorization_sha256,
        expected_implementation_identity_core_raw_sha256=(
            expected_implementation_identity_core_raw_sha256
        ),
        second_audit_token=second_audit_token,
        definition=definition,
        definition_sha256=definition_sha,
        transport_preregistration_sha256=(
            transport_preregistration_sha
        ),
        dependency_manifest=dependency_manifest,
        source_fingerprints=sources_start,
        stable_runtime_identity=stable_runtime_start,
    )
    _require_verified_v23_identity(
        verified,
        transport_preregistration_sha256=transport_preregistration_sha,
        dependency_manifest=dependency_manifest,
        source_fingerprints=sources_start,
        stable_runtime_identity=stable_runtime_start,
    )

    # The original loader continues to observe execution=false.  The new
    # ticket authorizes transport only and never edits the frozen contract.
    frozen_before_loader = _verify_ticket_frozen_files_before_loader(
        verified
    )
    contract = guard._load_frozen_contract()
    ordered_names, _, registry_manifest = _verify_contract_authorization(
        verified, contract
    )
    stable_input_start = _stable_execution_input(
        verified=verified,
        source_fingerprints=sources_start,
        contract=contract,
        registry_manifest_sha256=registry_manifest,
        dependency_manifest=dependency_manifest,
    )
    if (
        stable_input_start["frozen_definition_files"]
        != frozen_before_loader
    ):
        raise OneShotAuthorizationError(
            "frozen definition bytes changed across the frozen loader"
        )
    stable_input_start_sha = _sha256_bytes(
        _canonical_json_bytes(stable_input_start)
    )
    if _stable_runtime_identity() != stable_runtime_start:
        raise OneShotAuthorizationError(
            "stable runtime identity drifted before the marker"
        )

    dependency_ledger = _DependencyLedger(dependency_manifest)
    dependency_start = dependency_ledger.checkpoint(
        phase="baseline_pre_marker",
        require_baseline=True,
    )

    # Recheck both names under fixed no-follow directory descriptors, then
    # durably consume the ticket before the first formal history.
    output_directory = _acquire_output_directory_lease(
        ATTEMPT_MARKER_PATH, RESULT_PATH
    )
    marker_receipt = {
        "artifact": "S3ai-v2.2-transport-v2.3-permanent-one-shot-attempt",
        "attempt_id": _ATTEMPT_ID,
        "scientific_protocol_version": "S3ai-v2.2",
        "transport_protocol_version": _TRANSPORT_PROTOCOL_VERSION,
        "assurance_profile": _ASSURANCE_PROFILE,
        "authorization_id": verified.payload["authorization"]["id"],
        "authorization_raw_sha256": verified.raw_sha256,
        "authorization_canonical_sha256": verified.canonical_sha256,
        "single_use_token_sha256": verified.token_sha256,
        "wrapper_definition_sha256": verified.definition_sha256,
        "transport_preregistration_sha256": (
            transport_preregistration_sha
        ),
        "dependency_manifest_raw_sha256": dependency_manifest.raw_sha256,
        "dependency_manifest_canonical_sha256": (
            dependency_manifest.canonical_sha256
        ),
        "dependency_snapshot_sha256_start": dependency_start[
            "snapshot_sha256"
        ],
        "wrapper_source_sha256": sources_start[_WRAPPER_RELATIVE],
        "stable_execution_input_sha256_start": stable_input_start_sha,
        "ordered_registry_manifest_sha256": registry_manifest,
        "output_directory_st_dev": output_directory.st_dev,
        "output_directory_st_ino": output_directory.st_ino,
        "claimed_at_utc": datetime.now(timezone.utc).isoformat(),
        "retry_allowed": False,
    }
    try:
        attempt_claim = _claim_attempt_once(
            marker_path=ATTEMPT_MARKER_PATH,
            result_path=RESULT_PATH,
            receipt=marker_receipt,
            output_directory=output_directory,
        )

        with _ImportRequestObserver(
            dependency_ledger, "declared_formal_path"
        ):
            observations = collector(contract)
        dependency_ledger.checkpoint(
            phase="declared_formal_path",
        )
        with _ImportRequestObserver(
            dependency_ledger, "post_collector_pre_publication"
        ):
            result = guard.aggregate_frozen_histories(
                observations,
                contract,
                execution_code_fingerprints=sources_start,
            )
        _validate_frozen_result(
            result,
            observations,
            contract=contract,
            ordered_case_names=ordered_names,
            source_fingerprints=sources_start,
        )

        # This distinct post-collector checkpoint makes registered-member
        # removal visible even when the file was valid at the prior checkpoint.
        dependency_end = dependency_ledger.checkpoint(
            phase="post_collector_pre_publication",
            require_completion=True,
        )
        sources_end = _source_fingerprints(definition)
        stable_runtime_end = _stable_runtime_identity()
        stable_input_end = _stable_execution_input(
            verified=verified,
            source_fingerprints=sources_end,
            contract=contract,
            registry_manifest_sha256=registry_manifest,
            dependency_manifest=dependency_manifest,
        )
        stable_input_end_sha = _sha256_bytes(
            _canonical_json_bytes(stable_input_end)
        )
        if (
            sources_end != sources_start
            or stable_runtime_end != stable_runtime_start
            or stable_input_end != stable_input_start
            or stable_input_end_sha != stable_input_start_sha
        ):
            raise OneShotAuthorizationError(
                "local source, stable runtime, or stable execution input "
                "drifted during the consumed attempt; no result was published"
            )
        old_quarantine_end = _verify_old_failure_quarantine()
        if old_quarantine_end != old_quarantine_start:
            raise OneShotAuthorizationError(
                "retired failure quarantine changed during the fake/formal "
                "transport"
            )
        _verify_permanent_attempt_marker(
            ATTEMPT_MARKER_PATH,
            attempt_claim.receipt_sha256,
            output_directory=output_directory,
            result_path=RESULT_PATH,
            expected_identity=attempt_claim,
        )

        dependency_provenance = _dependency_closure_provenance(
            manifest=dependency_manifest,
            ledger=dependency_ledger,
            start_snapshot=dependency_start,
            end_snapshot=dependency_end,
        )
        _validate_dependency_closure_provenance(
            dependency_provenance,
            expected_manifest=dependency_manifest,
        )
        augmented = _augment_result(
            result,
            observations,
            ordered_case_names=ordered_names,
            verified=verified,
            registry_manifest_sha256=registry_manifest,
            source_fingerprints_start=sources_start,
            source_fingerprints_end=sources_end,
            stable_runtime_start=stable_runtime_start,
            stable_runtime_end=stable_runtime_end,
            stable_execution_input_sha256_start=(
                stable_input_start_sha
            ),
            stable_execution_input_sha256_end=stable_input_end_sha,
            dependency_closure=dependency_provenance,
            marker_receipt_sha256=attempt_claim.receipt_sha256,
        )
        # Result augmentation and the read-only validator are themselves in
        # the declared post-collector/pre-publication phase.  Re-check after
        # validation and, if a legal optional member first appeared, rebuild
        # the serialized closure provenance before validating again.  The
        # finite U bounds convergence; an unauthorized or wrong-phase member
        # fails at its first checkpoint.
        for validation_pass in range(len(dependency_manifest.U) + 2):
            serialized = json.loads(
                _pretty_json_bytes(augmented).decode("utf-8")
            )
            if validation_pass == 0:
                with _ImportRequestObserver(
                    dependency_ledger,
                    "post_collector_pre_publication",
                ):
                    _validate_serialized_result(
                        serialized,
                        ordered_case_names=ordered_names,
                        expected_dependency_manifest=dependency_manifest,
                        expected_verified=verified,
                    )
            else:
                _validate_serialized_result(
                    serialized,
                    ordered_case_names=ordered_names,
                    expected_dependency_manifest=dependency_manifest,
                    expected_verified=verified,
                )
            # Validation is executable code and may legally trigger a
            # preregistered optional lazy load.  Re-establish every stable
            # boundary after it, not merely before augmentation.
            boundary_sources = _source_fingerprints(definition)
            boundary_runtime = _stable_runtime_identity()
            boundary_input = _stable_execution_input(
                verified=verified,
                source_fingerprints=boundary_sources,
                contract=contract,
                registry_manifest_sha256=registry_manifest,
                dependency_manifest=dependency_manifest,
            )
            boundary_input_sha = _sha256_bytes(
                _canonical_json_bytes(boundary_input)
            )
            if (
                boundary_sources != sources_start
                or boundary_runtime != stable_runtime_start
                or boundary_input != stable_input_start
                or boundary_input_sha != stable_input_start_sha
            ):
                raise OneShotAuthorizationError(
                    "local source, stable runtime, or stable execution input "
                    "drifted at the atomic publication boundary"
                )
            _verify_all_manifest_members(dependency_manifest)
            if _verify_old_failure_quarantine() != old_quarantine_start:
                raise OneShotAuthorizationError(
                    "retired failure quarantine drifted before publication"
                )
            _verify_permanent_attempt_marker(
                ATTEMPT_MARKER_PATH,
                attempt_claim.receipt_sha256,
                output_directory=output_directory,
                result_path=RESULT_PATH,
                expected_identity=attempt_claim,
            )
            final_dependency_end = dependency_ledger.checkpoint(
                phase="post_collector_pre_publication",
                require_completion=True,
            )
            final_dependency_provenance = (
                _dependency_closure_provenance(
                    manifest=dependency_manifest,
                    ledger=dependency_ledger,
                    start_snapshot=dependency_start,
                    end_snapshot=final_dependency_end,
                )
            )
            if final_dependency_provenance == dependency_provenance:
                break
            dependency_end = final_dependency_end
            dependency_provenance = final_dependency_provenance
            _validate_dependency_closure_provenance(
                dependency_provenance,
                expected_manifest=dependency_manifest,
            )
            augmented["one_shot_provenance"][
                "dependency_closure"
            ] = dependency_provenance
        else:
            raise OneShotAuthorizationError(
                "dependency closure did not stabilize within the finite U "
                "before atomic publication"
            )

        def publication_pre_link_gate() -> None:
            """Re-establish all stable predicates after JSON preparation."""

            gate_sources = _source_fingerprints(definition)
            gate_runtime = _stable_runtime_identity()
            gate_input = _stable_execution_input(
                verified=verified,
                source_fingerprints=gate_sources,
                contract=contract,
                registry_manifest_sha256=registry_manifest,
                dependency_manifest=dependency_manifest,
            )
            gate_input_sha = _sha256_bytes(
                _canonical_json_bytes(gate_input)
            )
            if (
                gate_sources != sources_start
                or gate_runtime != stable_runtime_start
                or gate_input != stable_input_start
                or gate_input_sha != stable_input_start_sha
            ):
                raise OneShotAuthorizationError(
                    "local source, stable runtime, or stable execution input "
                    "drifted at the atomic pre-link gate"
                )
            _verify_all_manifest_members(dependency_manifest)
            if _verify_old_failure_quarantine() != old_quarantine_start:
                raise OneShotAuthorizationError(
                    "retired failure quarantine drifted at the atomic "
                    "pre-link gate"
                )
            _verify_permanent_attempt_marker(
                ATTEMPT_MARKER_PATH,
                attempt_claim.receipt_sha256,
                output_directory=output_directory,
                result_path=RESULT_PATH,
                expected_identity=attempt_claim,
            )
            gate_end = dependency_ledger.checkpoint(
                phase="post_collector_pre_publication",
                require_completion=True,
            )
            gate_provenance = _dependency_closure_provenance(
                manifest=dependency_manifest,
                ledger=dependency_ledger,
                start_snapshot=dependency_start,
                end_snapshot=gate_end,
            )
            _validate_dependency_closure_provenance(
                gate_provenance,
                expected_manifest=dependency_manifest,
            )
            if gate_provenance != dependency_provenance:
                raise OneShotAuthorizationError(
                    "dependency closure changed after serialized validation "
                    "and before the atomic link"
                )

        result_sha = _atomic_create_json_no_replace(
            RESULT_PATH,
            serialized,
            output_directory=output_directory,
            marker_path=ATTEMPT_MARKER_PATH,
            attempt_claim=attempt_claim,
            pre_link_gate=publication_pre_link_gate,
        )
    finally:
        os.close(output_directory.fd)
    return ExecutionReceipt(
        result_path=RESULT_PATH,
        result_sha256=result_sha,
        stage_decision=str(serialized["stage_decision"]),
        authorization_sha256=verified.raw_sha256,
        attempt_token_sha256=verified.token_sha256,
    )


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Execute the externally cleared S3ai-v2.2 one-shot observation"
        )
    )
    parser.add_argument(
        "--expected-authorization-sha256",
        required=True,
        help="out-of-band raw SHA256 of the fixed authorization YAML",
    )
    parser.add_argument(
        "--expected-implementation-identity-core-raw-sha256",
        required=True,
        help=(
            "out-of-band raw SHA256 of the fixed implementation identity "
            "core"
        ),
    )
    return parser


_CLI_PARSER = _build_cli_parser()


def _read_strict_second_audit_token(stream: Any) -> bytes:
    raw = stream.read(65)
    extra = stream.read(1)
    if (
        not isinstance(raw, bytes)
        or len(raw) != 65
        or extra != b""
        or raw[64:] != b"\n"
        or any(
            byte not in b"0123456789abcdef" for byte in raw[:64]
        )
    ):
        raise OneShotAuthorizationError(
            "standard input must be exactly 64 lowercase hex bytes, LF, EOF"
        )
    return bytes.fromhex(raw[:64].decode("ascii"))


def main(argv: Sequence[str] | None = None) -> int:
    """CLI that reads the 32-byte bearer token as hex from standard input."""

    arguments = _CLI_PARSER.parse_args(argv)
    token = _read_strict_second_audit_token(sys.stdin.buffer)
    receipt = run_authorized_once(
        expected_authorization_sha256=(
            arguments.expected_authorization_sha256
        ),
        expected_implementation_identity_core_raw_sha256=(
            arguments.expected_implementation_identity_core_raw_sha256
        ),
        second_audit_token=token,
    )
    print(
        json.dumps(
            {
                "result_path": str(receipt.result_path),
                "result_sha256": receipt.result_sha256,
                "stage_decision": receipt.stage_decision,
                "authorization_sha256": receipt.authorization_sha256,
                "attempt_token_sha256": receipt.attempt_token_sha256,
            },
            allow_nan=False,
            sort_keys=True,
        )
    )
    return 0


# Normalize stable library metadata at import without freezing any loaded-set
# snapshot.  Dynamic dependencies are governed only by the B/U/R/E ledger.
host_platform.platform()
_IMPORT_TIME_CONFIG_WARMUP = io.StringIO()
with redirect_stdout(_IMPORT_TIME_CONFIG_WARMUP):
    np.__config__.show()


__all__ = [
    "DependencyManifest",
    "DependencyMember",
    "ExecutionReceipt",
    "OneShotAuthorizationError",
    "main",
    "run_authorized_once",
]


if __name__ == "__main__":
    raise SystemExit(main())
