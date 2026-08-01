# ruff: noqa: E402
"""Fail-closed bootstrap primitives for the Fig17/18/19 evidence launcher.

This module deliberately contains no selector and no repository-discovery
logic.  The trusted outer bootstrap must supply metadata and exact bytes read
from the Git object database with the command contract defined below.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

from dataclasses import dataclass
import argparse
import hashlib
import importlib
import json
import os
import platform as python_platform
from pathlib import Path, PurePosixPath
import stat
from typing import Any, Iterable, Mapping, Sequence
import unicodedata


ATTESTATION_PAYLOAD_PATH = (
    "platform/docs/diag/fig171819_active_disease_attestation_20260729.json"
)
LAUNCHER_PATH = "platform/fig171819_evidence_launcher.py"
AUTHORIZATION_PATH = (
    "platform/docs/diag/fig171819_active_disease_execution_authorization_20260729.json"
)
EXECUTION_ROOT_RELATIVE = "platform/docs/diag/evidence_exec"
PYTHON_EXECUTABLE_REALPATH = "/home/exuber/anaconda3/envs/fluxvortex/bin/python"
PYTHON_FLAGS = ("-I", "-S", "-B")
LAUNCHER_CONTRACT = "DETACHED_COMMIT_DIRECT_EXEC_V1"

GIT_EXECUTABLE = "/usr/bin/git"
GIT_GLOBAL_OPTIONS = (
    "--no-replace-objects",
    "-c",
    "core.hooksPath=/dev/null",
    "-c",
    "filter.lfs.required=false",
)
GIT_ALLOWED_READ_ONLY_SUBCOMMANDS = frozenset(
    {"--version", "cat-file", "diff-tree", "ls-tree", "rev-parse", "show"}
)
GIT_CLEARED_ENVIRONMENT = frozenset(
    {
        "LD_PRELOAD",
        "LD_LIBRARY_PATH",
        "PYTHONPATH",
        "PYTHONHOME",
        "GIT_DIR",
        "GIT_WORK_TREE",
        "GIT_OBJECT_DIRECTORY",
        "GIT_ALTERNATE_OBJECT_DIRECTORIES",
        "GIT_EXEC_PATH",
    }
)
GIT_REQUIRED_ENVIRONMENT = {
    "GIT_NO_REPLACE_OBJECTS": "1",
    "GIT_CONFIG_NOSYSTEM": "1",
    "GIT_CONFIG_GLOBAL": "/dev/null",
}

_HEX40 = frozenset("0123456789abcdef")
_HEX64 = _HEX40
_REGULAR_GIT_MODES = frozenset({"100644", "100755"})
_RECEIPT_UPSTREAM_KEYS = {
    "outer_preflight": (),
    "inner_launcher": ("H0",),
    "outer_completion": ("H0", "H1"),
    "final": ("H0", "H1", "H2"),
}
ATTRIBUTION_ADAPTER = "FIG171819_CLAIM_ATTRIBUTION_MAIN_V1"
ATTRIBUTION_INPUTS = {
    "select-disease": ("fingerprint", "baseline_receipt"),
    "prepare": ("fingerprint", "disease_spec"),
    "evaluate": (
        "prereg",
        "baseline_receipt",
        "result",
        "manifest",
        "contributions",
        "scorecard",
        "fingerprint",
    ),
}
ATTRIBUTION_INTERMEDIATE_OUTPUTS = {
    "select-disease": ".raw-select-disease-payload.json",
    "prepare": ".raw-prepare-payload.json",
    "evaluate": ".raw-evaluate-payload.json",
}
ATTRIBUTION_SCIENTIFIC_STATUSES = {
    "select-disease": frozenset(
        {
            "ACTIVE_DISEASE_FROZEN",
            "NO_DECISION_NO_PRE_REPLICATION_CANDIDATE",
            "NO_DECISION_NO_INDEPENDENT_REPLICATION",
            "NO_DECISION_DISEASE_TIE",
            "NO_DECISION_VIEW_DISAGREEMENT",
            "NO_DECISION_INSUFFICIENT_RANK_MARGIN",
            "NO_DECISION_LEAVE_ONE_PF_SENSITIVE",
            "INVALID_EVIDENCE",
        }
    ),
    "prepare": frozenset({"PREPARED", "INVALID_EVIDENCE"}),
    "evaluate": frozenset(
        {
            "ACTIVE_N2_WRONG_COMPONENT_HYPOTHESIS",
            "ACTIVE_N3_WRONG_COMPONENT_HYPOTHESIS",
            "NO_DECISION_MULTIPLE_PARENTS",
            "NO_DECISION_NO_PARENT_FULL_COVERAGE",
            "INVALID_EVIDENCE",
        }
    ),
}
SCIENTIFIC_EXECUTION_ENVELOPE_FIELDS = frozenset(
    {
        "schema_version",
        "artifact_type",
        "stage",
        "evidence_commit_sha",
        "attestation_commit_sha",
        "attestation_payload_sha256",
        "authorization_blob_sha256",
        "launcher_blob_sha256",
        "git_no_replace_objects",
        "git_config_nosystem",
        "git_config_global",
        "git_hooks_disabled",
        "checkout_used",
        "raw_blob_materialization",
        "runtime_source_closure_verified",
        "runtime_source_closure_sha256",
        "python_executable_realpath",
        "python_isolated_flag",
        "python_no_site_flag",
        "python_no_bytecode_flag",
        "python_startup_contamination_check",
        "outer_preflight_receipt_sha256",
        "inner_launcher_receipt_sha256",
        "outer_completion_receipt_sha256",
        "outer_preflight_receipt",
        "inner_launcher_receipt",
        "outer_completion_receipt",
        "post_run_source_closure_verified",
        "transport_status",
        "scientific_status",
        "cleanup_status",
        "cleanup_scope",
        "cleanup_target_relative_paths",
        "execution_layout_cleanup_status",
    }
)
ATTRIBUTION_FLAGS = {
    "fingerprint": "--fingerprint",
    "baseline_receipt": "--baseline-receipt",
    "disease_spec": "--disease-spec",
    "prereg": "--prereg",
    "result": "--result",
    "manifest": "--manifest",
    "contributions": "--contributions",
    "scorecard": "--scorecard",
}
H0_RECEIPT_NAME = "H0.outer_preflight.json"
H1_RECEIPT_NAME = "H1.inner_launcher.json"
H2_RECEIPT_NAME = "H2.outer_completion.json"
FINAL_RECEIPT_NAME = "final.evidence_receipt.json"


class EvidenceContractError(ValueError):
    """Raised when any evidence bootstrap contract is not satisfied."""


def sha256_bytes(raw: bytes) -> str:
    """Return the SHA256 of the exact byte sequence supplied."""

    if not isinstance(raw, bytes):
        raise EvidenceContractError("hash input must be exact bytes")
    return hashlib.sha256(raw).hexdigest()


def git_blob_oid(raw: bytes) -> str:
    """Return the SHA-1 Git blob identity for exact bytes."""

    if not isinstance(raw, bytes):
        raise EvidenceContractError("Git blob input must be exact bytes")
    header = f"blob {len(raw)}\0".encode("ascii")
    return hashlib.sha1(header + raw, usedforsecurity=False).hexdigest()


def canonical_json_bytes(value: object) -> bytes:
    """Serialize a receipt/schema object deterministically, with one newline."""

    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
        + b"\n"
    )


def _reject_duplicate_keys(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise EvidenceContractError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_nonfinite_json(token: str) -> None:
    raise EvidenceContractError(f"non-finite JSON number: {token}")


def parse_strict_json_object(raw: bytes, *, label: str) -> dict[str, Any]:
    """Parse strict UTF-8 JSON while rejecting duplicates and non-finite values."""

    if not isinstance(raw, bytes):
        raise EvidenceContractError(f"{label} must be supplied as exact bytes")
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise EvidenceContractError(f"{label} is not strict UTF-8") from error
    try:
        value = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite_json,
        )
    except EvidenceContractError:
        raise
    except (json.JSONDecodeError, TypeError, ValueError) as error:
        raise EvidenceContractError(f"{label} is not valid JSON") from error
    if not isinstance(value, dict):
        raise EvidenceContractError(f"{label} must be a JSON object")
    return value


def _require_exact_keys(
    value: Mapping[str, object], expected: Iterable[str], *, label: str
) -> None:
    expected_set = set(expected)
    actual_set = set(value)
    if actual_set != expected_set:
        missing = sorted(expected_set - actual_set)
        extra = sorted(actual_set - expected_set)
        raise EvidenceContractError(
            f"{label} keys differ; missing={missing}, extra={extra}"
        )


def _require_exact_type(value: object, expected_type: type, *, label: str) -> None:
    if type(value) is not expected_type:
        raise EvidenceContractError(
            f"{label} must have exact type {expected_type.__name__}"
        )


def _validate_lower_hex(value: object, length: int, *, label: str) -> str:
    _require_exact_type(value, str, label=label)
    assert isinstance(value, str)
    alphabet = _HEX40 if length == 40 else _HEX64
    if len(value) != length or any(character not in alphabet for character in value):
        raise EvidenceContractError(
            f"{label} must be {length} lowercase hexadecimal characters"
        )
    return value


def validate_repo_relative_path(value: object, *, label: str) -> str:
    """Validate a canonical, traversal-free, NFC-normalized Git-style path."""

    _require_exact_type(value, str, label=label)
    assert isinstance(value, str)
    if not value or "\0" in value or "\\" in value:
        raise EvidenceContractError(f"{label} is not a safe repository path")
    if unicodedata.normalize("NFC", value) != value:
        raise EvidenceContractError(f"{label} is not NFC-normalized")
    path = PurePosixPath(value)
    if path.is_absolute() or str(path) != value:
        raise EvidenceContractError(f"{label} is not canonical repository-relative")
    if any(part in {"", ".", ".."} for part in path.parts):
        raise EvidenceContractError(f"{label} contains traversal or empty components")
    return value


def _validate_absolute_path(value: object, *, label: str) -> str:
    _require_exact_type(value, str, label=label)
    assert isinstance(value, str)
    if "\0" in value:
        raise EvidenceContractError(f"{label} contains NUL")
    path = Path(value)
    if not path.is_absolute() or os.path.normpath(value) != value:
        raise EvidenceContractError(f"{label} must be a canonical absolute path")
    return value


@dataclass(frozen=True)
class AttestationPayload:
    """The exact v5/v6 external-attestation payload."""

    schema_version: int
    artifact_type: str
    evidence_commit_sha: str
    authorization_path: str
    authorization_blob_sha256: str
    launcher_path: str
    launcher_blob_sha256: str
    attestation_payload_path: str
    fresh_run_id: str
    fresh_status_when_attested: str
    launcher_contract: str


def parse_attestation_payload(raw: bytes) -> AttestationPayload:
    """Parse and validate the closed attestation-payload schema."""

    value = parse_strict_json_object(raw, label="attestation payload")
    fields = tuple(AttestationPayload.__dataclass_fields__)
    _require_exact_keys(value, fields, label="attestation payload")
    _require_exact_type(value["schema_version"], int, label="schema_version")
    if value["schema_version"] != 1:
        raise EvidenceContractError("unsupported attestation schema_version")
    literals = {
        "artifact_type": "fig171819_active_disease_external_attestation",
        "authorization_path": AUTHORIZATION_PATH,
        "launcher_path": LAUNCHER_PATH,
        "attestation_payload_path": ATTESTATION_PAYLOAD_PATH,
        "fresh_run_id": "20260729_135128",
        "fresh_status_when_attested": "running",
        "launcher_contract": LAUNCHER_CONTRACT,
    }
    for key, expected in literals.items():
        if value[key] != expected:
            raise EvidenceContractError(f"attestation {key} is not the frozen literal")
    _validate_lower_hex(value["evidence_commit_sha"], 40, label="evidence_commit_sha")
    _validate_lower_hex(
        value["authorization_blob_sha256"],
        64,
        label="authorization_blob_sha256",
    )
    _validate_lower_hex(value["launcher_blob_sha256"], 64, label="launcher_blob_sha256")
    return AttestationPayload(**value)


@dataclass(frozen=True)
class RawCommitMetadata:
    """Commit identity obtained by no-replace raw-object traversal."""

    commit_sha: str
    tree_oid: str
    parents: tuple[str, ...]

    def __post_init__(self) -> None:
        _validate_lower_hex(self.commit_sha, 40, label="commit_sha")
        _validate_lower_hex(self.tree_oid, 40, label="tree_oid")
        if type(self.parents) is not tuple:
            raise EvidenceContractError("raw commit parents must be an ordered tuple")
        for index, parent in enumerate(self.parents):
            _validate_lower_hex(parent, 40, label=f"parents[{index}]")


@dataclass(frozen=True)
class RawTreeEntry:
    """A flattened raw-tree entry; paths are never discovered heuristically."""

    repo_relative_path: str
    git_mode: str
    git_blob_oid: str
    object_type: str = "blob"

    def __post_init__(self) -> None:
        validate_repo_relative_path(self.repo_relative_path, label="repo_relative_path")
        _require_exact_type(self.git_mode, str, label="git_mode")
        _validate_lower_hex(self.git_blob_oid, 40, label="git_blob_oid")
        if self.object_type != "blob":
            raise EvidenceContractError("raw closure entries must be blob objects")


def _validate_flat_tree(entries: Mapping[str, RawTreeEntry], *, label: str) -> None:
    folded: set[str] = set()
    for path, entry in entries.items():
        if type(path) is not str or not isinstance(entry, RawTreeEntry):
            raise EvidenceContractError(f"{label} is not a RawTreeEntry mapping")
        if path != entry.repo_relative_path:
            raise EvidenceContractError(f"{label} key does not match entry path")
        alias = unicodedata.normalize("NFC", path).casefold()
        if alias in folded:
            raise EvidenceContractError(f"{label} contains a casefold path alias")
        folded.add(alias)


def validate_one_added_payload_delta(
    evidence_tree: Mapping[str, RawTreeEntry],
    attestation_tree: Mapping[str, RawTreeEntry],
    *,
    diff_name_status: Sequence[str],
) -> None:
    """Require an identical flat tree plus exactly the fixed payload blob."""

    _validate_flat_tree(evidence_tree, label="evidence_tree")
    _validate_flat_tree(attestation_tree, label="attestation_tree")
    expected_delta = f"A\t{ATTESTATION_PAYLOAD_PATH}"
    if type(diff_name_status) not in {list, tuple} or list(diff_name_status) != [
        expected_delta
    ]:
        raise EvidenceContractError("tree delta is not exactly one fixed-path add")
    if ATTESTATION_PAYLOAD_PATH in evidence_tree:
        raise EvidenceContractError("payload already exists in evidence tree")
    payload_entry = attestation_tree.get(ATTESTATION_PAYLOAD_PATH)
    if payload_entry is None:
        raise EvidenceContractError("payload is absent from attestation tree")
    if payload_entry.git_mode not in _REGULAR_GIT_MODES:
        raise EvidenceContractError("attestation payload is not a regular Git blob")
    common_expected = dict(attestation_tree)
    del common_expected[ATTESTATION_PAYLOAD_PATH]
    if dict(evidence_tree) != common_expected:
        raise EvidenceContractError(
            "attestation tree has a mode, path, blob, or subtree change"
        )


def _walk_json(value: object) -> Iterable[tuple[str, object]]:
    if isinstance(value, dict):
        for key, child in value.items():
            yield key, child
            yield from _walk_json(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_json(child)


def reject_authorization_commit_identity_fields(value: object) -> None:
    """Reject evidence/attestation SHA fields at any authorization depth."""

    for key, _child in _walk_json(value):
        normalized = key.casefold()
        if ("evidence" in normalized or "attestation" in normalized) and (
            "sha" in normalized or "commit" in normalized
        ):
            raise EvidenceContractError(
                f"authorization contains forbidden commit identity field {key!r}"
            )


@dataclass(frozen=True)
class RuntimeSourceEntry:
    """One exact source blob in the authorized runtime closure."""

    repo_relative_path: str
    git_mode: str
    git_blob_oid: str
    sha256: str
    size_bytes: int

    @classmethod
    def from_json(cls, value: object, *, index: int) -> RuntimeSourceEntry:
        if not isinstance(value, dict):
            raise EvidenceContractError(
                f"runtime_source_closure[{index}] must be an object"
            )
        fields = tuple(cls.__dataclass_fields__)
        _require_exact_keys(value, fields, label=f"runtime_source_closure[{index}]")
        return cls(**value)

    def __post_init__(self) -> None:
        validate_repo_relative_path(self.repo_relative_path, label="repo_relative_path")
        if self.git_mode not in _REGULAR_GIT_MODES:
            raise EvidenceContractError("runtime source mode is not regular")
        _validate_lower_hex(self.git_blob_oid, 40, label="git_blob_oid")
        _validate_lower_hex(self.sha256, 64, label="sha256")
        _require_exact_type(self.size_bytes, int, label="size_bytes")
        if self.size_bytes < 0:
            raise EvidenceContractError("size_bytes must be nonnegative")
        path = PurePosixPath(self.repo_relative_path)
        if any(part == "__pycache__" for part in path.parts) or path.suffix in {
            ".pyc",
            ".pth",
        }:
            raise EvidenceContractError(
                "runtime source closure contains startup/cache contamination"
            )


def validate_runtime_source_closure(
    entries: Sequence[RuntimeSourceEntry],
) -> tuple[RuntimeSourceEntry, ...]:
    """Validate canonical order, uniqueness, and case-insensitive aliases."""

    if type(entries) not in {list, tuple}:
        raise EvidenceContractError("runtime_source_closure must be an array")
    result = tuple(entries)
    paths = [entry.repo_relative_path for entry in result]
    if paths != sorted(paths):
        raise EvidenceContractError(
            "runtime_source_closure must use repository-path lexical order"
        )
    if len(paths) != len(set(paths)):
        raise EvidenceContractError("runtime_source_closure contains duplicate paths")
    aliases = [unicodedata.normalize("NFC", path).casefold() for path in paths]
    if len(aliases) != len(set(aliases)):
        raise EvidenceContractError(
            "runtime_source_closure contains casefold path aliases"
        )
    return result


@dataclass(frozen=True)
class DistributionFile:
    """One exact file in an authorized installed dependency."""

    absolute_path: str
    sha256: str
    size_bytes: int

    @classmethod
    def from_json(cls, value: object, *, label: str) -> DistributionFile:
        if not isinstance(value, dict):
            raise EvidenceContractError(f"{label} must be an object")
        fields = tuple(cls.__dataclass_fields__)
        _require_exact_keys(value, fields, label=label)
        return cls(**value)

    def __post_init__(self) -> None:
        _validate_absolute_path(self.absolute_path, label="absolute_path")
        _validate_lower_hex(self.sha256, 64, label="sha256")
        _require_exact_type(self.size_bytes, int, label="size_bytes")
        if self.size_bytes < 0:
            raise EvidenceContractError("distribution size_bytes must be nonnegative")


@dataclass(frozen=True)
class DistributionRecord:
    """A closed distribution-file inventory."""

    name: str
    version: str
    files: tuple[DistributionFile, ...]

    @classmethod
    def from_json(cls, value: object, *, index: int) -> DistributionRecord:
        label = f"distributions[{index}]"
        if not isinstance(value, dict):
            raise EvidenceContractError(f"{label} must be an object")
        _require_exact_keys(value, ("name", "version", "files"), label=label)
        _require_exact_type(value["name"], str, label=f"{label}.name")
        _require_exact_type(value["version"], str, label=f"{label}.version")
        raw_files = value["files"]
        if not isinstance(raw_files, list):
            raise EvidenceContractError(f"{label}.files must be an array")
        files = tuple(
            DistributionFile.from_json(item, label=f"{label}.files[{file_index}]")
            for file_index, item in enumerate(raw_files)
        )
        paths = [item.absolute_path for item in files]
        if not files or paths != sorted(paths) or len(paths) != len(set(paths)):
            raise EvidenceContractError(
                f"{label}.files must be nonempty, sorted, and unique"
            )
        return cls(name=value["name"], version=value["version"], files=files)

    def __post_init__(self) -> None:
        if not self.name or not self.version:
            raise EvidenceContractError("distribution name/version must be nonempty")


@dataclass(frozen=True)
class RuntimeEnvironmentManifest:
    """Authorized interpreter and dependency identity."""

    python_executable_realpath: str
    python_implementation: str
    python_version: str
    python_flags: tuple[str, ...]
    stdlib_roots: tuple[str, ...]
    site_packages_roots: tuple[str, ...]
    distributions: tuple[DistributionRecord, ...]

    @classmethod
    def from_json(cls, value: object) -> RuntimeEnvironmentManifest:
        if not isinstance(value, dict):
            raise EvidenceContractError(
                "runtime_environment_manifest must be an object"
            )
        _require_exact_keys(
            value,
            tuple(cls.__dataclass_fields__),
            label="runtime_environment_manifest",
        )
        for key in (
            "python_executable_realpath",
            "python_implementation",
            "python_version",
        ):
            _require_exact_type(value[key], str, label=key)
        for key in ("python_flags", "stdlib_roots", "site_packages_roots"):
            if not isinstance(value[key], list) or not all(
                type(item) is str for item in value[key]
            ):
                raise EvidenceContractError(f"{key} must be an array of strings")
        raw_distributions = value["distributions"]
        if not isinstance(raw_distributions, list):
            raise EvidenceContractError("distributions must be an array")
        return cls(
            python_executable_realpath=value["python_executable_realpath"],
            python_implementation=value["python_implementation"],
            python_version=value["python_version"],
            python_flags=tuple(value["python_flags"]),
            stdlib_roots=tuple(value["stdlib_roots"]),
            site_packages_roots=tuple(value["site_packages_roots"]),
            distributions=tuple(
                DistributionRecord.from_json(item, index=index)
                for index, item in enumerate(raw_distributions)
            ),
        )

    def __post_init__(self) -> None:
        if self.python_executable_realpath != PYTHON_EXECUTABLE_REALPATH:
            raise EvidenceContractError("authorization names the wrong interpreter")
        if self.python_implementation != "CPython" or not self.python_version:
            raise EvidenceContractError("authorization has an invalid Python identity")
        if self.python_flags != PYTHON_FLAGS:
            raise EvidenceContractError("authorization must require -I -S -B")
        for label, roots in (
            ("stdlib_roots", self.stdlib_roots),
            ("site_packages_roots", self.site_packages_roots),
        ):
            if not roots:
                raise EvidenceContractError(f"{label} must be nonempty")
            validated = [
                _validate_absolute_path(item, label=f"{label}[]") for item in roots
            ]
            if validated != sorted(validated) or len(validated) != len(set(validated)):
                raise EvidenceContractError(f"{label} must be sorted and unique")
        names = [record.name.casefold() for record in self.distributions]
        if names != sorted(names) or len(names) != len(set(names)):
            raise EvidenceContractError(
                "distributions must be casefold-sorted and unique"
            )


@dataclass(frozen=True)
class ExecutionAuthorization:
    """The closed v7/v8 authorization schema."""

    schema_version: int
    artifact_type: str
    launcher_contract: str
    runtime_source_closure: tuple[RuntimeSourceEntry, ...]
    runtime_environment_manifest: RuntimeEnvironmentManifest
    output_relative_path: str


def parse_execution_authorization(raw: bytes) -> ExecutionAuthorization:
    """Parse and validate the source/runtime authorization without discovery."""

    value = parse_strict_json_object(raw, label="execution authorization")
    reject_authorization_commit_identity_fields(value)
    _require_exact_keys(
        value,
        (
            "schema_version",
            "artifact_type",
            "launcher_contract",
            "runtime_source_closure",
            "runtime_environment_manifest",
            "output_relative_path",
        ),
        label="execution authorization",
    )
    if type(value["schema_version"]) is not int or value["schema_version"] != 1:
        raise EvidenceContractError("unsupported authorization schema_version")
    if value["artifact_type"] != ("fig171819_active_disease_execution_authorization"):
        raise EvidenceContractError("wrong authorization artifact_type")
    if value["launcher_contract"] != LAUNCHER_CONTRACT:
        raise EvidenceContractError("wrong launcher_contract")
    raw_closure = value["runtime_source_closure"]
    if not isinstance(raw_closure, list) or not raw_closure:
        raise EvidenceContractError("runtime_source_closure must be nonempty")
    closure = validate_runtime_source_closure(
        [
            RuntimeSourceEntry.from_json(item, index=index)
            for index, item in enumerate(raw_closure)
        ]
    )
    manifest = RuntimeEnvironmentManifest.from_json(
        value["runtime_environment_manifest"]
    )
    output_relative_path = validate_repo_relative_path(
        value["output_relative_path"], label="output_relative_path"
    )
    if "/" in output_relative_path or output_relative_path != "output":
        raise EvidenceContractError("output_relative_path must be literal 'output'")
    return ExecutionAuthorization(
        schema_version=1,
        artifact_type=value["artifact_type"],
        launcher_contract=value["launcher_contract"],
        runtime_source_closure=closure,
        runtime_environment_manifest=manifest,
        output_relative_path=output_relative_path,
    )


def runtime_source_closure_sha256(
    entries: Sequence[RuntimeSourceEntry],
) -> str:
    """Hash the canonical metadata serialization of the authorized closure."""

    closure = validate_runtime_source_closure(entries)
    serializable = [
        {
            "repo_relative_path": entry.repo_relative_path,
            "git_mode": entry.git_mode,
            "git_blob_oid": entry.git_blob_oid,
            "sha256": entry.sha256,
            "size_bytes": entry.size_bytes,
        }
        for entry in closure
    ]
    return sha256_bytes(canonical_json_bytes(serializable))


def runtime_source_closure_to_json(
    entries: Sequence[RuntimeSourceEntry],
) -> list[dict[str, object]]:
    """Return the closed JSON representation embedded in H0."""

    closure = validate_runtime_source_closure(entries)
    return [
        {
            "repo_relative_path": entry.repo_relative_path,
            "git_mode": entry.git_mode,
            "git_blob_oid": entry.git_blob_oid,
            "sha256": entry.sha256,
            "size_bytes": entry.size_bytes,
        }
        for entry in closure
    ]


def runtime_source_closure_from_json(
    value: object,
) -> tuple[RuntimeSourceEntry, ...]:
    """Parse the H0 runtime closure with the authorization rules."""

    if not isinstance(value, list) or not value:
        raise EvidenceContractError("H0 runtime_source_closure must be nonempty")
    return validate_runtime_source_closure(
        [
            RuntimeSourceEntry.from_json(item, index=index)
            for index, item in enumerate(value)
        ]
    )


@dataclass(frozen=True)
class VerifiedAttestation:
    """Bindings proven by the outer raw-object bootstrap."""

    evidence_commit_sha: str
    attestation_commit_sha: str
    attestation_payload_sha256: str
    authorization_blob_sha256: str
    launcher_blob_sha256: str
    authorization: ExecutionAuthorization


def verify_attestation_bundle(
    *,
    evidence_commit: RawCommitMetadata,
    attestation_commit: RawCommitMetadata,
    evidence_tree: Mapping[str, RawTreeEntry],
    attestation_tree: Mapping[str, RawTreeEntry],
    diff_name_status: Sequence[str],
    payload_blob_bytes: bytes,
    expected_payload_sha256: str,
    authorization_blob_bytes: bytes,
    launcher_blob_bytes: bytes,
) -> VerifiedAttestation:
    """Verify exact commit/tree/blob bindings supplied by the outer bootstrap."""

    _validate_lower_hex(expected_payload_sha256, 64, label="expected_payload_sha256")
    if attestation_commit.parents != (evidence_commit.commit_sha,):
        raise EvidenceContractError(
            "attestation must have exactly the literal evidence parent"
        )
    validate_one_added_payload_delta(
        evidence_tree,
        attestation_tree,
        diff_name_status=diff_name_status,
    )
    payload_hash = sha256_bytes(payload_blob_bytes)
    if payload_hash != expected_payload_sha256:
        raise EvidenceContractError("attestation payload exact-byte hash mismatch")
    payload = parse_attestation_payload(payload_blob_bytes)
    if payload.evidence_commit_sha != evidence_commit.commit_sha:
        raise EvidenceContractError("payload evidence SHA differs from raw parent")

    required_bytes = {
        ATTESTATION_PAYLOAD_PATH: payload_blob_bytes,
        AUTHORIZATION_PATH: authorization_blob_bytes,
        LAUNCHER_PATH: launcher_blob_bytes,
    }
    for path, raw in required_bytes.items():
        entry = attestation_tree.get(path)
        if entry is None or entry.git_mode not in _REGULAR_GIT_MODES:
            raise EvidenceContractError(f"{path} is not a regular attestation blob")
        if entry.git_blob_oid != git_blob_oid(raw):
            raise EvidenceContractError(f"{path} raw Git blob identity mismatch")
    for path in (AUTHORIZATION_PATH, LAUNCHER_PATH):
        if evidence_tree.get(path) != attestation_tree.get(path):
            raise EvidenceContractError(f"{path} changed in attestation commit")

    authorization_hash = sha256_bytes(authorization_blob_bytes)
    launcher_hash = sha256_bytes(launcher_blob_bytes)
    if authorization_hash != payload.authorization_blob_sha256:
        raise EvidenceContractError("authorization exact-byte hash mismatch")
    if launcher_hash != payload.launcher_blob_sha256:
        raise EvidenceContractError("launcher exact-byte hash mismatch")
    authorization = parse_execution_authorization(authorization_blob_bytes)
    return VerifiedAttestation(
        evidence_commit_sha=evidence_commit.commit_sha,
        attestation_commit_sha=attestation_commit.commit_sha,
        attestation_payload_sha256=payload_hash,
        authorization_blob_sha256=authorization_hash,
        launcher_blob_sha256=launcher_hash,
        authorization=authorization,
    )


def sanitized_bootstrap_environment(
    environ: Mapping[str, str],
) -> dict[str, str]:
    """Return the v8 outer-bootstrap environment with contamination removed."""

    if not isinstance(environ, Mapping):
        raise EvidenceContractError("environment must be a string mapping")
    result: dict[str, str] = {}
    for key, value in environ.items():
        if type(key) is not str or type(value) is not str:
            raise EvidenceContractError("environment must contain only strings")
        if key not in GIT_CLEARED_ENVIRONMENT:
            result[key] = value
    result.update(GIT_REQUIRED_ENVIRONMENT)
    return result


def raw_git_argv(repo_root: Path, *arguments: str) -> tuple[str, ...]:
    """Construct one allowed no-replace, hooks-disabled raw Git command."""

    root = Path(repo_root)
    if not root.is_absolute():
        raise EvidenceContractError("Git repository root must be absolute")
    if not arguments or arguments[0] not in GIT_ALLOWED_READ_ONLY_SUBCOMMANDS:
        raise EvidenceContractError("Git command is not an allowed raw read")
    if not all(type(argument) is str and argument for argument in arguments):
        raise EvidenceContractError("Git arguments must be nonempty strings")
    return (
        GIT_EXECUTABLE,
        *GIT_GLOBAL_OPTIONS,
        "-C",
        str(root),
        *arguments,
    )


def validate_raw_git_invocation(
    argv: Sequence[str], environ: Mapping[str, str]
) -> None:
    """Validate an argv/environment record before accepting its metadata."""

    if type(argv) not in {list, tuple}:
        raise EvidenceContractError("Git argv must be an array")
    prefix = (GIT_EXECUTABLE, *GIT_GLOBAL_OPTIONS, "-C")
    if tuple(argv[: len(prefix)]) != prefix:
        raise EvidenceContractError("Git invocation lacks the fixed raw prefix")
    if len(argv) <= len(prefix) + 1:
        raise EvidenceContractError("Git invocation is incomplete")
    repo_root = argv[len(prefix)]
    if type(repo_root) is not str or not Path(repo_root).is_absolute():
        raise EvidenceContractError("Git invocation repository root is not absolute")
    subcommand = argv[len(prefix) + 1]
    if subcommand not in GIT_ALLOWED_READ_ONLY_SUBCOMMANDS:
        raise EvidenceContractError("Git invocation is not read-only")
    for key, expected in GIT_REQUIRED_ENVIRONMENT.items():
        if environ.get(key) != expected:
            raise EvidenceContractError(f"Git environment does not fix {key}")
    for key in GIT_CLEARED_ENVIRONMENT:
        if environ.get(key):
            raise EvidenceContractError(f"Git environment forwards forbidden {key}")


@dataclass(frozen=True)
class BootstrapCommandRecord:
    """One immutable argv/exit-code record required by the outer receipt."""

    argv: tuple[str, ...]
    exit_code: int

    def __post_init__(self) -> None:
        if type(self.argv) is not tuple or not all(
            type(argument) is str for argument in self.argv
        ):
            raise EvidenceContractError("bootstrap command argv must be a string tuple")
        _require_exact_type(self.exit_code, int, label="bootstrap exit_code")

    def to_json(self) -> dict[str, object]:
        return {"argv": list(self.argv), "exit_code": self.exit_code}


def isolated_python_argv(verified_launcher: Path) -> tuple[str, ...]:
    """Build the only authorized launcher invocation."""

    launcher = Path(verified_launcher)
    if not launcher.is_absolute():
        raise EvidenceContractError("verified launcher path must be absolute")
    return (PYTHON_EXECUTABLE_REALPATH, *PYTHON_FLAGS, str(launcher))


def validate_isolated_python_flags() -> None:
    """Require the running interpreter itself to report ``-I -S -B``."""

    if (
        sys.flags.isolated != 1
        or sys.flags.no_site != 1
        or sys.flags.dont_write_bytecode != 1
        or not sys.dont_write_bytecode
    ):
        raise EvidenceContractError(
            "running Python flags are not exactly isolated/no-site/no-bytecode"
        )


def _path_is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def assert_no_symlink_ancestors(
    path: Path, *, allow_missing_leaf: bool = False
) -> None:
    """Reject symlinks in an absolute path and every existing ancestor."""

    candidate = Path(path)
    if not candidate.is_absolute() or os.path.normpath(str(candidate)) != str(
        candidate
    ):
        raise EvidenceContractError("path must be canonical and absolute")
    parts = candidate.parts
    current = Path(parts[0])
    for index, part in enumerate(parts[1:], start=1):
        current = current / part
        try:
            metadata = os.lstat(current)
        except FileNotFoundError:
            if allow_missing_leaf and index == len(parts) - 1:
                return
            raise EvidenceContractError(
                f"path component is missing: {current}"
            ) from None
        if stat.S_ISLNK(metadata.st_mode):
            raise EvidenceContractError(f"path component is a symlink: {current}")
        if index != len(parts) - 1 and not stat.S_ISDIR(metadata.st_mode):
            raise EvidenceContractError(f"path ancestor is not a directory: {current}")


@dataclass(frozen=True)
class ExecutionLayout:
    """Exclusive source/output layout recorded by the launcher."""

    root: Path
    source_root: Path
    output_root: Path
    root_realpath: str
    root_device: int
    root_inode: int


def assert_source_output_separation(source_root: Path, output_root: Path) -> None:
    """Require existing, non-symlink, disjoint source and output roots."""

    source = Path(source_root)
    output = Path(output_root)
    assert_no_symlink_ancestors(source)
    assert_no_symlink_ancestors(output)
    source_real = Path(os.path.realpath(source))
    output_real = Path(os.path.realpath(output))
    if (
        source_real == output_real
        or _path_is_within(source_real, output_real)
        or _path_is_within(output_real, source_real)
    ):
        raise EvidenceContractError("source and output roots are not independent")


def create_execution_layout(
    repo_root: Path,
    evidence_commit_sha: str,
    *,
    allow_temporary_test_root: bool = False,
    execution_label: str | None = None,
) -> ExecutionLayout:
    """Exclusively create the v8 repo-local source/output execution layout."""

    _validate_lower_hex(evidence_commit_sha, 40, label="evidence_commit_sha")
    repository = Path(repo_root)
    if not repository.is_absolute():
        raise EvidenceContractError("repository root must be absolute")
    assert_no_symlink_ancestors(repository)
    repository_realpath = Path(os.path.realpath(repository))
    if (
        _path_is_within(repository_realpath, Path("/tmp"))
        and not allow_temporary_test_root
    ):
        raise EvidenceContractError(
            "production materialization beneath /tmp is forbidden"
        )
    base = repository / EXECUTION_ROOT_RELATIVE
    assert_no_symlink_ancestors(base)
    if execution_label is not None:
        label = validate_repo_relative_path(execution_label, label="execution_label")
        if "/" in label or not all(
            character in "abcdefghijklmnopqrstuvwxyz0123456789-" for character in label
        ):
            raise EvidenceContractError("execution_label is not a safe literal")
        root_name = f"evidence-{evidence_commit_sha}-{label}"
    else:
        root_name = f"evidence-{evidence_commit_sha}"
    root = base / root_name
    assert_no_symlink_ancestors(root, allow_missing_leaf=True)
    created: list[Path] = []
    try:
        os.mkdir(root, 0o700)
        created.append(root)
        source_root = root / "source"
        output_root = root / "output"
        os.mkdir(source_root, 0o700)
        created.append(source_root)
        os.mkdir(output_root, 0o700)
        created.append(output_root)
    except BaseException:
        for item in reversed(created):
            os.rmdir(item)
        raise
    assert_source_output_separation(source_root, output_root)
    metadata = os.lstat(root)
    return ExecutionLayout(
        root=root,
        source_root=source_root,
        output_root=output_root,
        root_realpath=os.path.realpath(root),
        root_device=int(metadata.st_dev),
        root_inode=int(metadata.st_ino),
    )


def _write_all(fd: int, raw: bytes) -> None:
    view = memoryview(raw)
    while view:
        written = os.write(fd, view)
        if written <= 0:
            raise EvidenceContractError("file write made no progress")
        view = view[written:]


def _safe_create_file(path: Path, raw: bytes, mode: int) -> None:
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    fd = os.open(path, flags, mode)
    try:
        _write_all(fd, raw)
        os.fsync(fd)
    finally:
        os.close(fd)


def _expected_directories(
    entries: Sequence[RuntimeSourceEntry],
) -> set[str]:
    expected: set[str] = set()
    for entry in entries:
        parent = PurePosixPath(entry.repo_relative_path).parent
        while str(parent) != ".":
            expected.add(str(parent))
            parent = parent.parent
    return expected


def _enumerate_materialized_tree(
    root: Path,
) -> tuple[dict[str, tuple[int, int, str]], set[str]]:
    files: dict[str, tuple[int, int, str]] = {}
    directories: set[str] = set()
    stack: list[tuple[Path, PurePosixPath]] = [(root, PurePosixPath("."))]
    while stack:
        absolute_directory, relative_directory = stack.pop()
        with os.scandir(absolute_directory) as iterator:
            children = sorted(iterator, key=lambda item: item.name)
        folded: set[str] = set()
        for child in children:
            alias = unicodedata.normalize("NFC", child.name).casefold()
            if alias in folded:
                raise EvidenceContractError("materialized tree has casefold aliases")
            folded.add(alias)
            relative = (
                PurePosixPath(child.name)
                if str(relative_directory) == "."
                else relative_directory / child.name
            )
            relative_text = str(relative)
            metadata = child.stat(follow_symlinks=False)
            if stat.S_ISLNK(metadata.st_mode):
                raise EvidenceContractError("materialized tree contains a symlink")
            if stat.S_ISDIR(metadata.st_mode):
                directories.add(relative_text)
                stack.append((Path(child.path), relative))
            elif stat.S_ISREG(metadata.st_mode):
                raw = Path(child.path).read_bytes()
                files[relative_text] = (
                    len(raw),
                    metadata.st_mode,
                    sha256_bytes(raw),
                )
            else:
                raise EvidenceContractError(
                    "materialized tree contains a non-regular object"
                )
    return files, directories


def verify_materialized_closure(
    source_root: Path,
    entries: Sequence[RuntimeSourceEntry],
    *,
    require_readonly: bool = False,
) -> str:
    """Re-enumerate and exact-hash a source tree, rejecting every extra member."""

    source = Path(source_root)
    assert_no_symlink_ancestors(source)
    closure = validate_runtime_source_closure(entries)
    files, directories = _enumerate_materialized_tree(source)
    expected_paths = {entry.repo_relative_path for entry in closure}
    if set(files) != expected_paths:
        raise EvidenceContractError("materialized source path set differs from closure")
    if directories != _expected_directories(closure):
        raise EvidenceContractError(
            "materialized source directory set differs from closure"
        )
    if require_readonly:
        if os.lstat(source).st_mode & 0o222:
            raise EvidenceContractError("materialized source root is writable")
        for directory in directories:
            directory_path = source.joinpath(*PurePosixPath(directory).parts)
            if os.lstat(directory_path).st_mode & 0o222:
                raise EvidenceContractError(
                    f"materialized source directory is writable: {directory}"
                )
    for entry in closure:
        size, filesystem_mode, digest = files[entry.repo_relative_path]
        if size != entry.size_bytes or digest != entry.sha256:
            raise EvidenceContractError(
                f"materialized source bytes differ: {entry.repo_relative_path}"
            )
        expected_executable = entry.git_mode == "100755"
        actual_executable = bool(filesystem_mode & 0o111)
        if actual_executable != expected_executable:
            raise EvidenceContractError(
                f"materialized executable bit differs: {entry.repo_relative_path}"
            )
        if require_readonly and filesystem_mode & 0o222:
            raise EvidenceContractError(
                f"materialized source is writable: {entry.repo_relative_path}"
            )
    return runtime_source_closure_sha256(closure)


def materialize_runtime_source_closure(
    source_root: Path,
    entries: Sequence[RuntimeSourceEntry],
    exact_blobs_by_oid: Mapping[str, bytes],
) -> tuple[Path, ...]:
    """Write an exact raw-blob closure into a pre-existing empty source root."""

    source = Path(source_root)
    assert_no_symlink_ancestors(source)
    closure = validate_runtime_source_closure(entries)
    required_oids = {entry.git_blob_oid for entry in closure}
    if set(exact_blobs_by_oid) != required_oids:
        raise EvidenceContractError("raw blob map does not exactly cover the closure")
    initial_files, initial_directories = _enumerate_materialized_tree(source)
    if initial_files or initial_directories:
        raise EvidenceContractError("materialized source root is not initially empty")
    for entry in closure:
        raw = exact_blobs_by_oid[entry.git_blob_oid]
        if type(raw) is not bytes:
            raise EvidenceContractError("raw closure object is not exact bytes")
        if (
            len(raw) != entry.size_bytes
            or sha256_bytes(raw) != entry.sha256
            or git_blob_oid(raw) != entry.git_blob_oid
        ):
            raise EvidenceContractError(
                f"raw blob metadata mismatch: {entry.repo_relative_path}"
            )

    created_files: list[Path] = []
    created_directories: list[Path] = []
    try:
        for directory in sorted(
            _expected_directories(closure),
            key=lambda item: (len(PurePosixPath(item).parts), item),
        ):
            path = source.joinpath(*PurePosixPath(directory).parts)
            if not path.exists():
                os.mkdir(path, 0o700)
                created_directories.append(path)
            else:
                metadata = os.lstat(path)
                if not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
                    raise EvidenceContractError(
                        "closure parent is not an owned regular directory"
                    )
        for entry in closure:
            path = source.joinpath(*PurePosixPath(entry.repo_relative_path).parts)
            mode = 0o755 if entry.git_mode == "100755" else 0o644
            _safe_create_file(path, exact_blobs_by_oid[entry.git_blob_oid], mode)
            created_files.append(path)
        verify_materialized_closure(source, closure)
    except BaseException:
        for path in reversed(created_files):
            os.unlink(path)
        for path in reversed(created_directories):
            os.rmdir(path)
        raise
    return tuple(created_files)


def seal_source_closure_readonly(
    source_root: Path, entries: Sequence[RuntimeSourceEntry]
) -> str:
    """Verify then make every source file/directory read-only."""

    source = Path(source_root)
    closure = validate_runtime_source_closure(entries)
    closure_hash = verify_materialized_closure(source, closure)
    for entry in closure:
        path = source.joinpath(*PurePosixPath(entry.repo_relative_path).parts)
        os.chmod(path, 0o555 if entry.git_mode == "100755" else 0o444)
    for directory in sorted(
        _expected_directories(closure),
        key=lambda item: (-len(PurePosixPath(item).parts), item),
    ):
        os.chmod(source.joinpath(*PurePosixPath(directory).parts), 0o555)
    os.chmod(source, 0o555)
    verify_materialized_closure(source, closure, require_readonly=True)
    return closure_hash


def validate_isolated_python_state(
    *,
    current_worktree: Path,
    materialized_source_root: Path,
    sys_path: Sequence[str] | None = None,
    loaded_modules: Mapping[str, object] | None = None,
    current_working_directory: Path | None = None,
    allowed_loaded_module_paths: Sequence[Path] = (),
) -> None:
    """Fail if Python startup contains worktree/site/customization pollution."""

    if not sys.dont_write_bytecode:
        raise EvidenceContractError("sys.dont_write_bytecode is false")
    worktree = Path(os.path.realpath(current_worktree))
    source = Path(os.path.realpath(materialized_source_root))
    cwd = Path(
        os.path.realpath(
            current_working_directory
            if current_working_directory is not None
            else Path.cwd()
        )
    )
    paths = tuple(sys.path if sys_path is None else sys_path)
    for raw_path in paths:
        if type(raw_path) is not str:
            raise EvidenceContractError("sys.path contains a non-string")
        candidate = cwd if raw_path == "" else Path(os.path.realpath(raw_path))
        if candidate == cwd or _path_is_within(candidate, worktree):
            raise EvidenceContractError(
                "current working directory/worktree contaminates sys.path"
            )
        if _path_is_within(candidate, source):
            raise EvidenceContractError(
                "project source was inserted before startup validation"
            )
    modules = sys.modules if loaded_modules is None else loaded_modules
    allowed_module_files = {
        os.path.realpath(path) for path in allowed_loaded_module_paths
    }
    for forbidden in ("sitecustomize", "usercustomize"):
        if forbidden in modules:
            raise EvidenceContractError(f"{forbidden} was imported at startup")
    for name, module in modules.items():
        module_file = getattr(module, "__file__", None)
        if type(module_file) is not str:
            continue
        module_path = Path(os.path.realpath(module_file))
        if str(module_path) in allowed_module_files:
            continue
        if _path_is_within(module_path, worktree) or _path_is_within(
            module_path, source
        ):
            raise EvidenceContractError(
                f"project module was imported before closure verification: {name}"
            )


def insert_authorized_python_roots(
    roots: Sequence[str], *, target_sys_path: list[str] | None = None
) -> tuple[str, ...]:
    """Insert explicit roots directly; never execute ``site.addsitedir`` or .pth."""

    if type(roots) not in {list, tuple} or not roots:
        raise EvidenceContractError("authorized Python roots must be nonempty")
    validated = tuple(
        _validate_absolute_path(root, label="authorized Python root") for root in roots
    )
    if len(validated) != len(set(validated)):
        raise EvidenceContractError("authorized Python roots contain duplicates")
    destination = sys.path if target_sys_path is None else target_sys_path
    for root in reversed(validated):
        destination.insert(0, root)
    return validated


def verify_runtime_environment_manifest(
    manifest: RuntimeEnvironmentManifest,
) -> dict[str, object]:
    """Verify the running interpreter and every authorized dependency file."""

    executable_identity = os.path.normpath(sys.executable)
    if executable_identity != manifest.python_executable_realpath:
        raise EvidenceContractError("running interpreter identity path differs")
    executable_resolved_realpath = os.path.realpath(sys.executable)
    implementation = python_platform.python_implementation()
    version = python_platform.python_version()
    if (
        implementation != manifest.python_implementation
        or version != manifest.python_version
    ):
        raise EvidenceContractError("running Python implementation/version differs")
    for root in (*manifest.stdlib_roots, *manifest.site_packages_roots):
        assert_no_symlink_ancestors(Path(root))
        if not Path(root).is_dir():
            raise EvidenceContractError("authorized Python root is not a directory")
    verified_files: list[dict[str, object]] = []
    for distribution in manifest.distributions:
        for authorized_file in distribution.files:
            path = Path(authorized_file.absolute_path)
            assert_no_symlink_ancestors(path)
            metadata = os.lstat(path)
            if not stat.S_ISREG(metadata.st_mode):
                raise EvidenceContractError("dependency path is not a regular file")
            raw = path.read_bytes()
            if (
                len(raw) != authorized_file.size_bytes
                or sha256_bytes(raw) != authorized_file.sha256
            ):
                raise EvidenceContractError("authorized dependency file drifted")
            verified_files.append(
                {
                    "absolute_path": authorized_file.absolute_path,
                    "sha256": authorized_file.sha256,
                    "size_bytes": authorized_file.size_bytes,
                }
            )
    return {
        "python_executable_realpath": executable_identity,
        "python_executable_resolved_realpath": executable_resolved_realpath,
        "python_implementation": implementation,
        "python_version": version,
        "distribution_files": verified_files,
    }


@dataclass(frozen=True)
class AuthorizedInput:
    """One exact external input authorized by the command envelope."""

    absolute_path: str
    sha256: str
    size_bytes: int

    @classmethod
    def from_json(cls, value: object, *, label: str) -> AuthorizedInput:
        if not isinstance(value, dict):
            raise EvidenceContractError(f"{label} must be an object")
        _require_exact_keys(
            value,
            ("absolute_path", "sha256", "size_bytes"),
            label=label,
        )
        return cls(**value)

    def __post_init__(self) -> None:
        _validate_absolute_path(self.absolute_path, label="input absolute_path")
        _validate_lower_hex(self.sha256, 64, label="input sha256")
        _require_exact_type(self.size_bytes, int, label="input size_bytes")
        if self.size_bytes < 0:
            raise EvidenceContractError("input size_bytes must be nonnegative")


@dataclass(frozen=True)
class AttributionCommandEnvelope:
    """Stable adapter contract for ``fig171819_claim_attribution.main``."""

    schema_version: int
    adapter: str
    command: str
    inputs: tuple[tuple[str, AuthorizedInput], ...]
    output_relative_path: str

    @property
    def receipt_kind(self) -> str:
        if self.command == "select-disease":
            return "selector"
        return self.command


def parse_attribution_command_envelope(
    value: object,
) -> AttributionCommandEnvelope:
    """Parse a no-guess, exact-input adapter command."""

    if not isinstance(value, dict):
        raise EvidenceContractError("attribution command envelope must be an object")
    _require_exact_keys(
        value,
        (
            "schema_version",
            "adapter",
            "command",
            "inputs",
            "output_relative_path",
        ),
        label="attribution command envelope",
    )
    if type(value["schema_version"]) is not int or value["schema_version"] != 1:
        raise EvidenceContractError("unsupported attribution command schema")
    if value["adapter"] != ATTRIBUTION_ADAPTER:
        raise EvidenceContractError("wrong attribution adapter")
    command = value["command"]
    if command not in ATTRIBUTION_INPUTS:
        raise EvidenceContractError("unknown attribution command")
    raw_inputs = value["inputs"]
    if not isinstance(raw_inputs, dict):
        raise EvidenceContractError("attribution inputs must be an object")
    expected_inputs = ATTRIBUTION_INPUTS[command]
    if set(raw_inputs) != set(expected_inputs):
        raise EvidenceContractError(
            "attribution inputs are absent, extra, or command-incompatible"
        )
    inputs = tuple(
        (
            name,
            AuthorizedInput.from_json(
                raw_inputs[name], label=f"attribution inputs.{name}"
            ),
        )
        for name in expected_inputs
    )
    paths = [item.absolute_path for _name, item in inputs]
    if len(paths) != len(set(paths)):
        raise EvidenceContractError("attribution input paths must be unique")
    output_relative_path = validate_repo_relative_path(
        value["output_relative_path"], label="attribution output_relative_path"
    )
    if "/" in output_relative_path:
        raise EvidenceContractError("attribution output must be a direct output member")
    if output_relative_path in {
        H0_RECEIPT_NAME,
        H1_RECEIPT_NAME,
        H2_RECEIPT_NAME,
        FINAL_RECEIPT_NAME,
        *ATTRIBUTION_INTERMEDIATE_OUTPUTS.values(),
    }:
        raise EvidenceContractError("attribution output collides with a receipt")
    return AttributionCommandEnvelope(
        schema_version=1,
        adapter=ATTRIBUTION_ADAPTER,
        command=command,
        inputs=inputs,
        output_relative_path=output_relative_path,
    )


def verify_authorized_input(authorized: AuthorizedInput) -> None:
    """Verify an explicit input without path discovery or fallback."""

    path = Path(authorized.absolute_path)
    assert_no_symlink_ancestors(path)
    metadata = os.lstat(path)
    if not stat.S_ISREG(metadata.st_mode):
        raise EvidenceContractError("authorized input is not a regular file")
    raw = path.read_bytes()
    if len(raw) != authorized.size_bytes or sha256_bytes(raw) != authorized.sha256:
        raise EvidenceContractError("authorized input exact bytes differ")


def attribution_adapter_argv(
    command: AttributionCommandEnvelope, output_root: Path
) -> tuple[str, ...]:
    """Build the frozen ``main(argv)`` forwarding vector."""

    root = Path(output_root)
    assert_no_symlink_ancestors(root)
    output = root / ATTRIBUTION_INTERMEDIATE_OUTPUTS[command.command]
    if not _path_is_within(output, root):
        raise EvidenceContractError("attribution output escapes output root")
    argv: list[str] = [command.command]
    for name, authorized in command.inputs:
        verify_authorized_input(authorized)
        argv.extend((ATTRIBUTION_FLAGS[name], authorized.absolute_path))
    argv.extend(("--output", str(output)))
    return tuple(argv)


def scientific_payload_canonical_sha256(payload: Mapping[str, object]) -> str:
    """Match claim-attribution's exact canonical payload hash."""

    if not isinstance(payload, Mapping):
        raise EvidenceContractError("scientific payload must be an object")
    try:
        raw = json.dumps(
            dict(payload),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise EvidenceContractError(
            "scientific payload is not canonically serializable"
        ) from error
    return sha256_bytes(raw)


def build_scientific_execution_envelope(
    *,
    stage: str,
    bindings: Mapping[str, str],
    runtime_source_closure_sha256_value: str,
    h0: ReceiptEnvelope,
    h1: ReceiptEnvelope,
    h2: ReceiptEnvelope,
) -> dict[str, object]:
    """Build the post-H2 envelope accepted by claim attribution."""

    if stage not in ATTRIBUTION_INPUTS:
        raise EvidenceContractError("scientific execution stage is unknown")
    _require_exact_keys(
        bindings,
        (
            "evidence_commit_sha",
            "attestation_commit_sha",
            "attestation_payload_sha256",
            "authorization_blob_sha256",
            "launcher_blob_sha256",
        ),
        label="scientific execution bindings",
    )
    _validate_lower_hex(
        bindings["evidence_commit_sha"], 40, label="evidence_commit_sha"
    )
    _validate_lower_hex(
        bindings["attestation_commit_sha"],
        40,
        label="attestation_commit_sha",
    )
    for label in (
        "attestation_payload_sha256",
        "authorization_blob_sha256",
        "launcher_blob_sha256",
    ):
        _validate_lower_hex(bindings[label], 64, label=label)
    for label, digest in (
        ("runtime_source_closure_sha256", runtime_source_closure_sha256_value),
        ("outer_preflight_receipt_sha256", h0.sha256),
        ("inner_launcher_receipt_sha256", h1.sha256),
        ("outer_completion_receipt_sha256", h2.sha256),
    ):
        _validate_lower_hex(digest, 64, label=label)
    h0_object = _parse_and_validate_receipt_envelope(h0)
    h1_object = _parse_and_validate_receipt_envelope(h1)
    h2_object = _parse_and_validate_receipt_envelope(h2)
    h2_body = h2_object["body"]
    assert isinstance(h2_body, dict)
    envelope: dict[str, object] = {
        "schema_version": 1,
        "artifact_type": "fig171819_verified_execution_receipt_envelope",
        "stage": stage,
        **dict(bindings),
        "git_no_replace_objects": True,
        "git_config_nosystem": True,
        "git_config_global": "/dev/null",
        "git_hooks_disabled": True,
        "checkout_used": False,
        "raw_blob_materialization": True,
        "runtime_source_closure_verified": True,
        "runtime_source_closure_sha256": (runtime_source_closure_sha256_value),
        "python_executable_realpath": PYTHON_EXECUTABLE_REALPATH,
        "python_isolated_flag": True,
        "python_no_site_flag": True,
        "python_no_bytecode_flag": True,
        "python_startup_contamination_check": "PASS",
        "outer_preflight_receipt_sha256": h0.sha256,
        "inner_launcher_receipt_sha256": h1.sha256,
        "outer_completion_receipt_sha256": h2.sha256,
        "outer_preflight_receipt": h0_object,
        "inner_launcher_receipt": h1_object,
        "outer_completion_receipt": h2_object,
        "post_run_source_closure_verified": True,
        "transport_status": h2_body.get("transport_status"),
        "scientific_status": h2_body.get("scientific_status"),
        "cleanup_status": "PASS",
        "cleanup_scope": "INTERMEDIATE_PAYLOAD_ONLY",
        "cleanup_target_relative_paths": [ATTRIBUTION_INTERMEDIATE_OUTPUTS[stage]],
        "execution_layout_cleanup_status": "NOT_RUN_RETAINED_FOR_AUDIT",
    }
    if set(envelope) != SCIENTIFIC_EXECUTION_ENVELOPE_FIELDS:
        raise EvidenceContractError("scientific execution envelope schema differs")
    validate_scientific_execution_envelope(envelope, stage=stage)
    return envelope


def validate_scientific_execution_envelope(
    value: object, *, stage: str
) -> dict[str, object]:
    """Mirror attribution's exact verified-execution-envelope validator."""

    if stage not in ATTRIBUTION_INPUTS:
        raise EvidenceContractError("scientific execution stage is unknown")
    if not isinstance(value, dict):
        raise EvidenceContractError("scientific execution envelope must be an object")
    if set(value) != SCIENTIFIC_EXECUTION_ENVELOPE_FIELDS:
        raise EvidenceContractError("scientific execution envelope schema differs")
    if (
        type(value["schema_version"]) is not int
        or value["schema_version"] != 1
        or value["artifact_type"] != "fig171819_verified_execution_receipt_envelope"
        or value["stage"] != stage
    ):
        raise EvidenceContractError("scientific execution envelope identity differs")
    for label in ("evidence_commit_sha", "attestation_commit_sha"):
        _validate_lower_hex(value[label], 40, label=label)
    for label in (
        "attestation_payload_sha256",
        "authorization_blob_sha256",
        "launcher_blob_sha256",
        "runtime_source_closure_sha256",
        "outer_preflight_receipt_sha256",
        "inner_launcher_receipt_sha256",
        "outer_completion_receipt_sha256",
    ):
        _validate_lower_hex(value[label], 64, label=label)
    embedded: dict[str, tuple[str, str]] = {
        "outer_preflight_receipt": (
            "outer_preflight",
            "outer_preflight_receipt_sha256",
        ),
        "inner_launcher_receipt": (
            "inner_launcher",
            "inner_launcher_receipt_sha256",
        ),
        "outer_completion_receipt": (
            "outer_completion",
            "outer_completion_receipt_sha256",
        ),
    }
    receipts: dict[str, dict[str, Any]] = {}
    for field, (receipt_stage, hash_field) in embedded.items():
        receipt_object = value[field]
        if not isinstance(receipt_object, dict):
            raise EvidenceContractError(f"{field} must be an exact receipt object")
        exact_bytes = canonical_json_bytes(receipt_object)
        digest = sha256_bytes(exact_bytes)
        if digest != value[hash_field]:
            raise EvidenceContractError(f"{field} canonical SHA256 differs")
        receipt = ReceiptEnvelope(
            stage=receipt_stage,
            exact_bytes=exact_bytes,
            sha256=digest,
        )
        receipts[receipt_stage] = _parse_and_validate_receipt_envelope(receipt)
    h0 = receipts["outer_preflight"]
    h1 = receipts["inner_launcher"]
    h2 = receipts["outer_completion"]
    if h0["upstream_receipts"] != {}:
        raise EvidenceContractError("embedded H0 has an upstream receipt")
    if h1["upstream_receipts"] != {"H0": value["outer_preflight_receipt_sha256"]}:
        raise EvidenceContractError("embedded H1 does not bind H0")
    if h2["upstream_receipts"] != {
        "H0": value["outer_preflight_receipt_sha256"],
        "H1": value["inner_launcher_receipt_sha256"],
    }:
        raise EvidenceContractError("embedded H2 does not bind H0 and H1")
    h0_body = h0["body"]
    h1_body = h1["body"]
    h2_body = h2["body"]
    assert isinstance(h0_body, dict)
    assert isinstance(h1_body, dict)
    assert isinstance(h2_body, dict)
    command = h0_body.get("command_envelope")
    binding_names = (
        "evidence_commit_sha",
        "attestation_commit_sha",
        "attestation_payload_sha256",
        "authorization_blob_sha256",
        "launcher_blob_sha256",
    )
    if (
        h0_body.get("status") != "PASS"
        or h0_body.get("dry_run") is not False
        or h0_body.get("bindings") != {name: value[name] for name in binding_names}
        or h0_body.get("runtime_source_closure_sha256")
        != value["runtime_source_closure_sha256"]
        or h0_body.get("git_no_replace_objects") != value["git_no_replace_objects"]
        or h0_body.get("git_config_nosystem") != value["git_config_nosystem"]
        or h0_body.get("git_config_global") != value["git_config_global"]
        or h0_body.get("git_hooks_disabled") != value["git_hooks_disabled"]
        or h0_body.get("checkout_used") != value["checkout_used"]
        or h0_body.get("raw_blob_materialization") != value["raw_blob_materialization"]
        or not isinstance(command, dict)
        or command.get("command") != stage
    ):
        raise EvidenceContractError("embedded H0 command/binding body differs")
    if (
        h1_body.get("status") != "PASS"
        or h1_body.get("runtime_source_closure_verified") is not True
        or h1_body.get("runtime_source_closure_sha256")
        != value["runtime_source_closure_sha256"]
        or h1_body.get("python_executable_realpath")
        != value["python_executable_realpath"]
        or h1_body.get("python_isolated_flag") is not True
        or h1_body.get("python_no_site_flag") is not True
        or h1_body.get("python_no_bytecode_flag") is not True
        or h1_body.get("python_startup_contamination_check") != "PASS"
        or h1_body.get("runtime_environment_manifest_verified") is not True
    ):
        raise EvidenceContractError("embedded H1 closure/startup body differs")
    if (
        h2_body.get("status") != "PASS"
        or h2_body.get("transport_status") != "PASS"
        or h2_body.get("post_run_source_closure") != "PASS"
        or h2_body.get("post_run_output_inventory") != "PASS"
        or h2_body.get("post_run_errors") != []
        or h2_body.get("intermediate_payload_cleanup_status") != "PASS"
        or h2_body.get("cleanup_status") != "PASS"
        or h2_body.get("cleanup_scope") != "INTERMEDIATE_PAYLOAD_ONLY"
        or h2_body.get("cleanup_target_relative_paths")
        != [ATTRIBUTION_INTERMEDIATE_OUTPUTS[stage]]
        or h2_body.get("execution_layout_cleanup_status")
        != "NOT_RUN_RETAINED_FOR_AUDIT"
    ):
        raise EvidenceContractError("embedded H2 transport/cleanup body differs")
    scientific_status = h2_body.get("scientific_status")
    if (
        scientific_status not in ATTRIBUTION_SCIENTIFIC_STATUSES[stage]
        or value["transport_status"] != "PASS"
        or value["scientific_status"] != scientific_status
        or value["cleanup_scope"] != "INTERMEDIATE_PAYLOAD_ONLY"
        or value["cleanup_target_relative_paths"]
        != [ATTRIBUTION_INTERMEDIATE_OUTPUTS[stage]]
        or value["cleanup_target_relative_paths"]
        != h2_body.get("cleanup_target_relative_paths")
        or value["execution_layout_cleanup_status"] != "NOT_RUN_RETAINED_FOR_AUDIT"
    ):
        raise EvidenceContractError("scientific/cleanup status binding differs")
    expected_exit_code = 2 if scientific_status == "INVALID_EVIDENCE" else 0
    if h2_body.get("inner_exit_code") != expected_exit_code:
        raise EvidenceContractError("scientific status/inner exit code differs")
    for label in (
        "intermediate_payload_exact_sha256",
        "intermediate_payload_canonical_sha256",
    ):
        _validate_lower_hex(h2_body.get(label), 64, label=label)
    for label in (
        "git_no_replace_objects",
        "git_config_nosystem",
        "git_hooks_disabled",
        "raw_blob_materialization",
        "runtime_source_closure_verified",
        "python_isolated_flag",
        "python_no_site_flag",
        "python_no_bytecode_flag",
        "post_run_source_closure_verified",
    ):
        if value[label] is not True:
            raise EvidenceContractError(
                f"scientific execution security gate failed: {label}"
            )
    if (
        value["checkout_used"] is not False
        or value["git_config_global"] != "/dev/null"
        or value["python_executable_realpath"] != PYTHON_EXECUTABLE_REALPATH
        or value["python_startup_contamination_check"] != "PASS"
        or value["cleanup_status"] != "PASS"
    ):
        raise EvidenceContractError("scientific execution runtime contract differs")
    return value


def wrap_scientific_payload(
    *,
    stage: str,
    payload: Mapping[str, object],
    execution_envelope: Mapping[str, object],
) -> dict[str, object]:
    """Wrap a raw payload only after the immutable H2 identity exists."""

    if stage not in ATTRIBUTION_INPUTS:
        raise EvidenceContractError("scientific payload stage is unknown")
    validate_scientific_execution_envelope(dict(execution_envelope), stage=stage)
    normalized_payload = dict(payload)
    return {
        "schema_version": 1,
        "artifact_type": "fig171819_scientific_payload_envelope",
        "stage": stage,
        "payload_sha256": scientific_payload_canonical_sha256(normalized_payload),
        "execution_envelope": dict(execution_envelope),
        "payload": normalized_payload,
        "production_execution_authorized": True,
    }


def validate_wrapped_scientific_payload(
    value: object, *, stage: str
) -> dict[str, object]:
    """Validate the same envelope contract consumed by attribution."""

    if not isinstance(value, dict):
        raise EvidenceContractError("scientific payload envelope must be an object")
    _require_exact_keys(
        value,
        (
            "schema_version",
            "artifact_type",
            "stage",
            "payload_sha256",
            "execution_envelope",
            "payload",
            "production_execution_authorized",
        ),
        label="scientific payload envelope",
    )
    if (
        type(value["schema_version"]) is not int
        or value["schema_version"] != 1
        or value["artifact_type"] != "fig171819_scientific_payload_envelope"
        or value["stage"] != stage
        or value["production_execution_authorized"] is not True
    ):
        raise EvidenceContractError("scientific payload envelope identity differs")
    payload = value["payload"]
    execution = value["execution_envelope"]
    if not isinstance(payload, dict) or not isinstance(execution, dict):
        raise EvidenceContractError("scientific payload envelope is malformed")
    if value["payload_sha256"] != scientific_payload_canonical_sha256(payload):
        raise EvidenceContractError("scientific payload canonical hash differs")
    validate_scientific_execution_envelope(execution, stage=stage)
    h2 = execution["outer_completion_receipt"]
    assert isinstance(h2, dict)
    h2_body = h2["body"]
    assert isinstance(h2_body, dict)
    if payload.get("status") != execution["scientific_status"] or value[
        "payload_sha256"
    ] != h2_body.get("intermediate_payload_canonical_sha256"):
        raise EvidenceContractError(
            "scientific payload status/hash differs from embedded H2"
        )
    return value


def _contains_forbidden_self_hash(value: object) -> bool:
    for key, _child in _walk_json(value):
        normalized = key.casefold()
        if normalized in {"h0", "h1", "h2", "h_self"} or (
            ("self" in normalized or "own" in normalized or "receipt" in normalized)
            and ("sha" in normalized or "hash" in normalized)
        ):
            return True
    return False


@dataclass(frozen=True)
class ReceiptEnvelope:
    """One immutable node in the H0 -> H1 -> H2 -> final receipt DAG."""

    stage: str
    exact_bytes: bytes
    sha256: str


def build_receipt_envelope(
    stage: str,
    *,
    upstream_receipts: Mapping[str, str],
    body: Mapping[str, object],
) -> ReceiptEnvelope:
    """Build a deterministic receipt envelope without a self/downstream hash."""

    if stage not in _RECEIPT_UPSTREAM_KEYS:
        raise EvidenceContractError("unknown receipt stage")
    expected_upstream = _RECEIPT_UPSTREAM_KEYS[stage]
    if tuple(upstream_receipts) != expected_upstream:
        raise EvidenceContractError(
            "receipt upstream hashes are absent, reordered, or downstream"
        )
    for label, digest in upstream_receipts.items():
        _validate_lower_hex(digest, 64, label=label)
    if not isinstance(body, Mapping):
        raise EvidenceContractError("receipt body must be an object")
    if _contains_forbidden_self_hash(body):
        raise EvidenceContractError("receipt body contains a self-hash field")
    if stage == "outer_completion":
        required = {"post_run_source_closure", "cleanup_status"}
        if not required.issubset(body):
            raise EvidenceContractError(
                "outer completion lacks post-run closure or cleanup status"
            )
    if stage == "final" and body.get("receipt_kind") not in {
        "selector",
        "prepare",
        "evaluate",
    }:
        raise EvidenceContractError("final receipt_kind is not frozen")
    value = {
        "schema_version": 1,
        "artifact_type": "fig171819_evidence_receipt_envelope",
        "stage": stage,
        "upstream_receipts": dict(upstream_receipts),
        "body": dict(body),
    }
    raw = canonical_json_bytes(value)
    return ReceiptEnvelope(stage=stage, exact_bytes=raw, sha256=sha256_bytes(raw))


def _parse_and_validate_receipt_envelope(
    envelope: ReceiptEnvelope,
) -> dict[str, Any]:
    if envelope.stage not in _RECEIPT_UPSTREAM_KEYS:
        raise EvidenceContractError("receipt envelope stage is unknown")
    if sha256_bytes(envelope.exact_bytes) != envelope.sha256:
        raise EvidenceContractError("receipt envelope byte hash mismatch")
    value = parse_strict_json_object(
        envelope.exact_bytes, label=f"{envelope.stage} receipt"
    )
    _require_exact_keys(
        value,
        (
            "schema_version",
            "artifact_type",
            "stage",
            "upstream_receipts",
            "body",
        ),
        label=f"{envelope.stage} receipt",
    )
    if (
        type(value["schema_version"]) is not int
        or value["schema_version"] != 1
        or value["artifact_type"] != "fig171819_evidence_receipt_envelope"
        or value["stage"] != envelope.stage
    ):
        raise EvidenceContractError("receipt envelope identity differs")
    upstream = value["upstream_receipts"]
    if not isinstance(upstream, dict):
        raise EvidenceContractError("receipt upstream_receipts must be an object")
    expected_keys = _RECEIPT_UPSTREAM_KEYS[envelope.stage]
    if tuple(upstream) != expected_keys:
        raise EvidenceContractError("receipt upstream schema differs")
    for label, digest in upstream.items():
        _validate_lower_hex(digest, 64, label=label)
    body = value["body"]
    if not isinstance(body, dict) or _contains_forbidden_self_hash(body):
        raise EvidenceContractError("receipt body is invalid or self-referential")
    if envelope.stage == "outer_completion" and not {
        "post_run_source_closure",
        "cleanup_status",
    }.issubset(body):
        raise EvidenceContractError("outer completion body is incomplete")
    if envelope.stage == "final" and body.get("receipt_kind") not in {
        "selector",
        "prepare",
        "evaluate",
    }:
        raise EvidenceContractError("final receipt_kind is not frozen")
    if canonical_json_bytes(value) != envelope.exact_bytes:
        raise EvidenceContractError("receipt envelope is not canonical exact bytes")
    return value


def validate_receipt_chain(
    outer_preflight: ReceiptEnvelope,
    inner_launcher: ReceiptEnvelope,
    outer_completion: ReceiptEnvelope,
    final: ReceiptEnvelope,
) -> tuple[str, str, str]:
    """Validate exact bytes and the acyclic H0/H1/H2 references."""

    envelopes = (
        outer_preflight,
        inner_launcher,
        outer_completion,
        final,
    )
    expected_stages = tuple(_RECEIPT_UPSTREAM_KEYS)
    for envelope, expected_stage in zip(envelopes, expected_stages, strict=True):
        if envelope.stage != expected_stage:
            raise EvidenceContractError("receipt chain stage order differs")
    parsed = [_parse_and_validate_receipt_envelope(item) for item in envelopes]
    h0, h1, h2 = (item.sha256 for item in envelopes[:3])
    expected_refs = ({}, {"H0": h0}, {"H0": h0, "H1": h1})
    for value, expected in zip(parsed[:3], expected_refs, strict=True):
        if value.get("upstream_receipts") != expected:
            raise EvidenceContractError("receipt DAG upstream reference mismatch")
    if parsed[3].get("upstream_receipts") != {
        "H0": h0,
        "H1": h1,
        "H2": h2,
    }:
        raise EvidenceContractError("final receipt does not bind H0, H1, and H2")
    return h0, h1, h2


def write_immutable_bytes(output_root: Path, relative_path: str, raw: bytes) -> str:
    """Create one no-replace file strictly beneath the authorized output root."""

    root = Path(output_root)
    assert_no_symlink_ancestors(root)
    relative = validate_repo_relative_path(relative_path, label="output relative_path")
    destination = root.joinpath(*PurePosixPath(relative).parts)
    if not _path_is_within(destination, root) or destination == root:
        raise EvidenceContractError("immutable output escapes its authorized root")
    assert_no_symlink_ancestors(destination.parent)
    if type(raw) is not bytes:
        raise EvidenceContractError("immutable output must be exact bytes")
    _safe_create_file(destination, raw, 0o600)
    os.chmod(destination, 0o400)
    return sha256_bytes(raw)


def write_immutable_receipt(
    output_root: Path, relative_path: str, envelope: ReceiptEnvelope
) -> str:
    """Write an envelope once; an existing path or symlink fails closed."""

    _parse_and_validate_receipt_envelope(envelope)
    written_hash = write_immutable_bytes(
        output_root, relative_path, envelope.exact_bytes
    )
    if written_hash != envelope.sha256:
        raise EvidenceContractError("immutable receipt write hash mismatch")
    return written_hash


def verify_output_inventory(
    output_root: Path, expected_relative_paths: Sequence[str]
) -> dict[str, str]:
    """Reject symlinks, non-regular objects, and every unregistered output."""

    root = Path(output_root)
    assert_no_symlink_ancestors(root)
    if type(expected_relative_paths) not in {list, tuple}:
        raise EvidenceContractError("expected output inventory must be an array")
    expected = [
        validate_repo_relative_path(path, label="expected output path")
        for path in expected_relative_paths
    ]
    if expected != sorted(expected) or len(expected) != len(set(expected)):
        raise EvidenceContractError("expected output paths must be sorted and unique")
    aliases = [path.casefold() for path in expected]
    if len(aliases) != len(set(aliases)):
        raise EvidenceContractError("expected output paths contain casefold aliases")
    files, directories = _enumerate_materialized_tree(root)
    if set(files) != set(expected):
        raise EvidenceContractError("output path set differs from registered inventory")
    expected_directories: set[str] = set()
    for relative in expected:
        parent = PurePosixPath(relative).parent
        while str(parent) != ".":
            expected_directories.add(str(parent))
            parent = parent.parent
    if directories != expected_directories:
        raise EvidenceContractError(
            "output directory set differs from registered inventory"
        )
    return {path: files[path][2] for path in expected}


def read_immutable_receipt(
    path: Path, *, stage: str, expected_sha256: str | None = None
) -> ReceiptEnvelope:
    """Read one regular read-only receipt and validate its canonical envelope."""

    receipt_path = Path(path)
    assert_no_symlink_ancestors(receipt_path)
    metadata = os.lstat(receipt_path)
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_mode & 0o222:
        raise EvidenceContractError("receipt is not an immutable regular file")
    raw = receipt_path.read_bytes()
    digest = sha256_bytes(raw)
    if expected_sha256 is not None:
        _validate_lower_hex(expected_sha256, 64, label="expected receipt sha256")
        if digest != expected_sha256:
            raise EvidenceContractError("receipt exact-byte hash differs")
    envelope = ReceiptEnvelope(stage=stage, exact_bytes=raw, sha256=digest)
    _parse_and_validate_receipt_envelope(envelope)
    return envelope


def cleanup_execution_layout_exact(
    layout: ExecutionLayout,
    source_entries: Sequence[RuntimeSourceEntry],
    output_relative_paths: Sequence[str],
    *,
    file_handles_closed: bool,
    no_child_processes: bool,
) -> None:
    """Delete only a fully verified, explicitly ledgered execution layout."""

    if not file_handles_closed or not no_child_processes:
        raise EvidenceContractError("cleanup preconditions are not proven")
    assert_no_symlink_ancestors(layout.root)
    root_metadata = os.lstat(layout.root)
    if (
        os.path.realpath(layout.root) != layout.root_realpath
        or int(root_metadata.st_dev) != layout.root_device
        or int(root_metadata.st_ino) != layout.root_inode
    ):
        raise EvidenceContractError("execution layout root identity changed")
    with os.scandir(layout.root) as iterator:
        root_children = sorted(iterator, key=lambda item: item.name)
    if [item.name for item in root_children] != ["output", "source"]:
        raise EvidenceContractError(
            "execution layout root contains an extra or missing member"
        )
    for item in root_children:
        metadata = item.stat(follow_symlinks=False)
        if not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
            raise EvidenceContractError(
                "execution layout root member is not an owned directory"
            )
    closure = validate_runtime_source_closure(source_entries)
    verify_materialized_closure(layout.source_root, closure, require_readonly=True)
    verify_output_inventory(layout.output_root, output_relative_paths)

    os.chmod(layout.source_root, 0o700)
    source_directories = _expected_directories(closure)
    for directory in sorted(
        source_directories,
        key=lambda item: (len(PurePosixPath(item).parts), item),
    ):
        os.chmod(
            layout.source_root.joinpath(*PurePosixPath(directory).parts),
            0o700,
        )
    for entry in closure:
        os.unlink(
            layout.source_root.joinpath(*PurePosixPath(entry.repo_relative_path).parts)
        )
    for directory in sorted(
        source_directories,
        key=lambda item: (-len(PurePosixPath(item).parts), item),
    ):
        os.rmdir(layout.source_root.joinpath(*PurePosixPath(directory).parts))
    os.rmdir(layout.source_root)

    output_directories: set[str] = set()
    for relative_path in output_relative_paths:
        path = validate_repo_relative_path(relative_path, label="cleanup output path")
        os.unlink(layout.output_root.joinpath(*PurePosixPath(path).parts))
        parent = PurePosixPath(path).parent
        while str(parent) != ".":
            output_directories.add(str(parent))
            parent = parent.parent
    for directory in sorted(
        output_directories,
        key=lambda item: (-len(PurePosixPath(item).parts), item),
    ):
        os.rmdir(layout.output_root.joinpath(*PurePosixPath(directory).parts))
    os.rmdir(layout.output_root)
    os.rmdir(layout.root)


_H0_BODY_KEYS = (
    "status",
    "dry_run",
    "repo_root",
    "bindings",
    "authorization",
    "runtime_source_closure",
    "runtime_source_closure_sha256",
    "execution_layout",
    "bootstrap_commands",
    "outer_binary_identities",
    "git_no_replace_objects",
    "git_config_nosystem",
    "git_config_global",
    "git_hooks_disabled",
    "checkout_used",
    "raw_blob_materialization",
    "command_envelope",
)


def _validate_inner_h0(
    *,
    h0: ReceiptEnvelope,
    source_root: Path,
    output_root: Path,
    current_worktree: Path,
    dry_run: bool,
) -> tuple[
    dict[str, Any],
    ExecutionAuthorization,
    tuple[RuntimeSourceEntry, ...],
]:
    parsed = _parse_and_validate_receipt_envelope(h0)
    body = parsed["body"]
    assert isinstance(body, dict)
    _require_exact_keys(body, _H0_BODY_KEYS, label="H0 body")
    if (
        body["status"] != "PASS"
        or type(body["dry_run"]) is not bool
        or body["dry_run"] is not dry_run
    ):
        raise EvidenceContractError("H0 status/dry-run identity differs")
    if body["repo_root"] != str(current_worktree):
        raise EvidenceContractError("H0 repository root differs")
    fixed_controls = {
        "git_no_replace_objects": True,
        "git_config_nosystem": True,
        "git_config_global": "/dev/null",
        "git_hooks_disabled": True,
        "checkout_used": False,
        "raw_blob_materialization": True,
    }
    for key, expected in fixed_controls.items():
        if body[key] != expected:
            raise EvidenceContractError(f"H0 control differs: {key}")
    layout = body["execution_layout"]
    if not isinstance(layout, dict):
        raise EvidenceContractError("H0 execution_layout is not an object")
    _require_exact_keys(
        layout,
        (
            "root",
            "source_root",
            "output_root",
            "root_realpath",
            "root_device",
            "root_inode",
        ),
        label="H0 execution_layout",
    )
    root = source_root.parent
    if (
        output_root.parent != root
        or layout["root"] != str(root)
        or layout["source_root"] != str(source_root)
        or layout["output_root"] != str(output_root)
        or layout["root_realpath"] != os.path.realpath(root)
    ):
        raise EvidenceContractError("H0 execution layout paths differ")
    root_metadata = os.lstat(root)
    if (
        type(layout["root_device"]) is not int
        or type(layout["root_inode"]) is not int
        or layout["root_device"] != int(root_metadata.st_dev)
        or layout["root_inode"] != int(root_metadata.st_ino)
    ):
        raise EvidenceContractError("H0 execution layout identity differs")
    authorization_value = body["authorization"]
    authorization = parse_execution_authorization(
        canonical_json_bytes(authorization_value)
    )
    closure = runtime_source_closure_from_json(body["runtime_source_closure"])
    if closure != authorization.runtime_source_closure:
        raise EvidenceContractError("H0 closure differs from authorization")
    if body["runtime_source_closure_sha256"] != (
        runtime_source_closure_sha256(closure)
    ):
        raise EvidenceContractError("H0 source closure metadata hash differs")
    if dry_run:
        if body["command_envelope"] is not None:
            raise EvidenceContractError("dry-run H0 contains an adapter command")
    else:
        parse_attribution_command_envelope(body["command_envelope"])
    return body, authorization, closure


def _inner_main(arguments: argparse.Namespace) -> int:
    source_root = Path(arguments.source_root)
    output_root = Path(arguments.output_root)
    current_worktree = Path(arguments.current_worktree)
    h0_path = Path(arguments.h0_path)
    for path in (
        source_root,
        output_root,
        current_worktree,
        h0_path,
    ):
        if not path.is_absolute():
            raise EvidenceContractError("inner launcher paths must be absolute")
    assert_source_output_separation(source_root, output_root)
    if h0_path != output_root / H0_RECEIPT_NAME:
        raise EvidenceContractError("inner H0 path is not the fixed output member")
    launcher_path = source_root.joinpath(*PurePosixPath(LAUNCHER_PATH).parts)
    if Path(os.path.realpath(__file__)) != launcher_path:
        raise EvidenceContractError("inner launcher is not the materialized blob")
    h0 = read_immutable_receipt(
        h0_path,
        stage="outer_preflight",
        expected_sha256=arguments.h0_sha256,
    )
    try:
        validate_isolated_python_flags()
        body, authorization, closure = _validate_inner_h0(
            h0=h0,
            source_root=source_root,
            output_root=output_root,
            current_worktree=current_worktree,
            dry_run=arguments.dry_run,
        )
        validate_isolated_python_state(
            current_worktree=current_worktree,
            materialized_source_root=source_root,
            allowed_loaded_module_paths=(launcher_path,),
        )
        closure_hash = verify_materialized_closure(
            source_root, closure, require_readonly=True
        )
        runtime_identity = verify_runtime_environment_manifest(
            authorization.runtime_environment_manifest
        )
    except EvidenceContractError as error:
        failure = build_receipt_envelope(
            "inner_launcher",
            upstream_receipts={"H0": h0.sha256},
            body={
                "status": "INVALID_EVIDENCE",
                "failure_class": type(error).__name__,
                "failure_detail_sha256": sha256_bytes(str(error).encode("utf-8")),
            },
        )
        write_immutable_receipt(output_root, H1_RECEIPT_NAME, failure)
        raise
    h1 = build_receipt_envelope(
        "inner_launcher",
        upstream_receipts={"H0": h0.sha256},
        body={
            "status": "PASS",
            "python_executable_realpath": runtime_identity[
                "python_executable_realpath"
            ],
            "python_executable_resolved_realpath": runtime_identity[
                "python_executable_resolved_realpath"
            ],
            "python_implementation": runtime_identity["python_implementation"],
            "python_version": runtime_identity["python_version"],
            "python_isolated_flag": True,
            "python_no_site_flag": True,
            "python_no_bytecode_flag": True,
            "python_startup_contamination_check": "PASS",
            "runtime_environment_manifest_verified": True,
            "runtime_source_closure_verified": True,
            "runtime_source_closure_sha256": closure_hash,
        },
    )
    write_immutable_receipt(output_root, H1_RECEIPT_NAME, h1)
    if arguments.dry_run:
        return 0

    command = parse_attribution_command_envelope(body["command_envelope"])
    adapter_argv = attribution_adapter_argv(command, output_root)
    manifest = authorization.runtime_environment_manifest
    insert_authorized_python_roots(
        (*manifest.stdlib_roots, *manifest.site_packages_roots)
    )
    materialized_platform = source_root / "platform"
    sys.path.insert(0, str(materialized_platform))
    attribution = importlib.import_module("fig171819_claim_attribution")
    expected_attribution_path = materialized_platform / "fig171819_claim_attribution.py"
    if Path(os.path.realpath(attribution.__file__)) != expected_attribution_path:
        raise EvidenceContractError("attribution adapter imported outside closure")
    main = getattr(attribution, "main", None)
    if not callable(main):
        raise EvidenceContractError("materialized attribution main is absent")
    return_code = main(adapter_argv)
    _require_exact_type(return_code, int, label="attribution return code")
    return return_code


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Verified inner Fig17/18/19 evidence launcher"
    )
    parser.add_argument("--inner", action="store_true", required=True)
    parser.add_argument("--h0-path", required=True)
    parser.add_argument("--h0-sha256", required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--current-worktree", required=True)
    parser.add_argument("--dry-run", action="store_true")
    arguments = parser.parse_args(argv)
    return _inner_main(arguments)


def main(argv: Sequence[str] | None = None) -> int:
    """Public inner entry point used only by the verified materialized blob."""

    try:
        return _main(argv)
    except EvidenceContractError as error:
        print(f"INVALID_EVIDENCE: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
