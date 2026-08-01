# ruff: noqa: E402
"""Minimal trusted outer bootstrap for Fig17/18/19 evidence execution.

The CLI accepts only literal commit/hash identities.  It never searches for a
result, manifest, scorecard, fingerprint, contribution, authorization, or
attestation.  ``--dry-run`` exercises the complete raw-Git and isolated-inner
chain without importing the scientific attribution module.
"""

from __future__ import annotations

import sys
from pathlib import Path, PurePosixPath

sys.dont_write_bytecode = True

# ``-I`` deliberately omits the script directory from ``sys.path``.  The
# external raw-object bootstrap materializes this file and its exact launcher
# sibling together, so add only that verified sibling directory before the
# first project import.
_MATERIALIZED_BOOTSTRAP_DIRECTORY = str(Path(__file__).resolve().parent)
if _MATERIALIZED_BOOTSTRAP_DIRECTORY not in sys.path:
    sys.path.insert(0, _MATERIALIZED_BOOTSTRAP_DIRECTORY)

import argparse
from dataclasses import dataclass
import json
import os
import stat
import subprocess
from typing import Mapping, Sequence

import fig171819_evidence_launcher as launcher


ATTRIBUTION_PATH = "platform/fig171819_claim_attribution.py"


@dataclass
class _RawGitReader:
    repo_root: Path
    environment: dict[str, str]
    records: list[launcher.BootstrapCommandRecord]

    @classmethod
    def create(cls, repo_root: Path) -> _RawGitReader:
        return cls(
            repo_root=repo_root,
            environment=launcher.sanitized_bootstrap_environment(os.environ),
            records=[],
        )

    def run(self, *arguments: str) -> bytes:
        argv = launcher.raw_git_argv(self.repo_root, *arguments)
        launcher.validate_raw_git_invocation(argv, self.environment)
        completed = subprocess.run(
            argv,
            check=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=self.environment,
            cwd="/",
        )
        self.records.append(
            launcher.BootstrapCommandRecord(argv=argv, exit_code=completed.returncode)
        )
        if completed.returncode != 0:
            raise launcher.EvidenceContractError(
                "raw Git command failed: "
                f"{arguments[0]} stderr_sha256="
                f"{launcher.sha256_bytes(completed.stderr)}"
            )
        return completed.stdout


def _parse_raw_commit(commit_sha: str, raw: bytes) -> launcher.RawCommitMetadata:
    launcher._validate_lower_hex(commit_sha, 40, label="literal commit SHA")
    try:
        header = raw.split(b"\n\n", 1)[0].decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise launcher.EvidenceContractError(
            "raw commit header is not strict UTF-8"
        ) from error
    tree_oid: str | None = None
    parents: list[str] = []
    previous_header = False
    for line in header.splitlines():
        if line.startswith(" "):
            if not previous_header:
                raise launcher.EvidenceContractError(
                    "raw commit has an orphan continuation"
                )
            continue
        previous_header = True
        key, separator, value = line.partition(" ")
        if not separator:
            raise launcher.EvidenceContractError("raw commit header is malformed")
        if key == "tree":
            if tree_oid is not None:
                raise launcher.EvidenceContractError(
                    "raw commit has multiple tree headers"
                )
            tree_oid = value
        elif key == "parent":
            parents.append(value)
    if tree_oid is None:
        raise launcher.EvidenceContractError("raw commit tree header is absent")
    return launcher.RawCommitMetadata(
        commit_sha=commit_sha,
        tree_oid=tree_oid,
        parents=tuple(parents),
    )


def _read_commit(git: _RawGitReader, commit_sha: str) -> launcher.RawCommitMetadata:
    object_type = git.run("cat-file", "-t", commit_sha)
    if object_type != b"commit\n":
        raise launcher.EvidenceContractError("literal identity is not a commit")
    return _parse_raw_commit(commit_sha, git.run("cat-file", "commit", commit_sha))


def _read_flat_tree(
    git: _RawGitReader, commit_sha: str
) -> dict[str, launcher.RawTreeEntry]:
    raw = git.run("ls-tree", "-rz", "--full-tree", commit_sha)
    result: dict[str, launcher.RawTreeEntry] = {}
    for record in raw.split(b"\0"):
        if not record:
            continue
        metadata, separator, raw_path = record.partition(b"\t")
        if not separator:
            raise launcher.EvidenceContractError(
                "raw ls-tree record lacks a path separator"
            )
        fields = metadata.split(b" ")
        if len(fields) != 3:
            raise launcher.EvidenceContractError("raw ls-tree metadata is malformed")
        try:
            mode, object_type, oid = (
                field.decode("ascii", errors="strict") for field in fields
            )
            path = raw_path.decode("utf-8", errors="strict")
        except UnicodeDecodeError as error:
            raise launcher.EvidenceContractError(
                "raw ls-tree record is not strict UTF-8/ASCII"
            ) from error
        entry = launcher.RawTreeEntry(
            repo_relative_path=path,
            git_mode=mode,
            git_blob_oid=oid,
            object_type=object_type,
        )
        if path in result:
            raise launcher.EvidenceContractError("raw tree contains duplicate paths")
        result[path] = entry
    launcher._validate_flat_tree(result, label="raw commit tree")
    return result


def _read_blob(
    git: _RawGitReader,
    tree: Mapping[str, launcher.RawTreeEntry],
    path: str,
) -> bytes:
    entry = tree.get(path)
    if entry is None or entry.git_mode not in {"100644", "100755"}:
        raise launcher.EvidenceContractError(
            f"fixed raw blob path is absent or non-regular: {path}"
        )
    raw = git.run("cat-file", "blob", entry.git_blob_oid)
    if launcher.git_blob_oid(raw) != entry.git_blob_oid:
        raise launcher.EvidenceContractError(
            f"raw Git blob bytes differ from OID: {path}"
        )
    return raw


def _read_tree_delta(
    git: _RawGitReader, evidence_sha: str, attestation_sha: str
) -> list[str]:
    raw = git.run(
        "diff-tree",
        "--no-commit-id",
        "--name-status",
        "-r",
        evidence_sha,
        attestation_sha,
    )
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise launcher.EvidenceContractError(
            "raw tree delta is not strict UTF-8"
        ) from error
    return text.splitlines()


def _binary_identity(path: Path) -> dict[str, object]:
    literal = Path(path)
    launcher.assert_no_symlink_ancestors(literal.parent)
    realpath = Path(os.path.realpath(literal))
    launcher.assert_no_symlink_ancestors(realpath)
    metadata = os.lstat(realpath)
    if not stat.S_ISREG(metadata.st_mode):
        raise launcher.EvidenceContractError("TCB executable is not regular")
    raw = realpath.read_bytes()
    return {
        "literal_path": str(literal),
        "realpath": str(realpath),
        "sha256": launcher.sha256_bytes(raw),
        "size_bytes": len(raw),
    }


def _layout_json(layout: launcher.ExecutionLayout) -> dict[str, object]:
    return {
        "root": str(layout.root),
        "source_root": str(layout.source_root),
        "output_root": str(layout.output_root),
        "root_realpath": layout.root_realpath,
        "root_device": layout.root_device,
        "root_inode": layout.root_inode,
    }


def _load_command_envelope(
    path: Path,
) -> tuple[dict[str, object], launcher.AttributionCommandEnvelope]:
    command_path = Path(path)
    if not command_path.is_absolute():
        raise launcher.EvidenceContractError("command envelope path must be absolute")
    launcher.assert_no_symlink_ancestors(command_path)
    metadata = os.lstat(command_path)
    if not stat.S_ISREG(metadata.st_mode):
        raise launcher.EvidenceContractError("command envelope is not a regular file")
    value = launcher.parse_strict_json_object(
        command_path.read_bytes(), label="attribution command envelope"
    )
    command = launcher.parse_attribution_command_envelope(value)
    for _name, authorized_input in command.inputs:
        launcher.verify_authorized_input(authorized_input)
    return value, command


def _closure_blobs(
    git: _RawGitReader,
    evidence_tree: Mapping[str, launcher.RawTreeEntry],
    authorization: launcher.ExecutionAuthorization,
    launcher_blob: bytes,
) -> dict[str, bytes]:
    closure_paths = {
        entry.repo_relative_path for entry in authorization.runtime_source_closure
    }
    required = {launcher.LAUNCHER_PATH, ATTRIBUTION_PATH}
    if not required.issubset(closure_paths):
        raise launcher.EvidenceContractError(
            "runtime closure lacks launcher or attribution adapter"
        )
    blobs: dict[str, bytes] = {}
    for entry in authorization.runtime_source_closure:
        raw_entry = evidence_tree.get(entry.repo_relative_path)
        if (
            raw_entry is None
            or raw_entry.git_mode != entry.git_mode
            or raw_entry.git_blob_oid != entry.git_blob_oid
        ):
            raise launcher.EvidenceContractError(
                "authorization closure differs from raw evidence tree"
            )
        if entry.repo_relative_path == launcher.LAUNCHER_PATH:
            raw = launcher_blob
        else:
            raw = _read_blob(git, evidence_tree, entry.repo_relative_path)
        if (
            len(raw) != entry.size_bytes
            or launcher.sha256_bytes(raw) != entry.sha256
            or launcher.git_blob_oid(raw) != entry.git_blob_oid
        ):
            raise launcher.EvidenceContractError(
                "authorization closure exact bytes differ"
            )
        prior = blobs.setdefault(entry.git_blob_oid, raw)
        if prior != raw:
            raise launcher.EvidenceContractError(
                "one closure OID maps to different bytes"
            )
    return blobs


def _python_inner_environment() -> dict[str, str]:
    return launcher.sanitized_bootstrap_environment(os.environ)


def run_outer_bootstrap(
    *,
    repo_root: Path,
    evidence_commit_sha: str,
    attestation_commit_sha: str,
    attestation_payload_sha256: str,
    dry_run: bool,
    command_envelope_path: Path | None,
    allow_synthetic_temp_repo: bool = False,
) -> tuple[int, dict[str, object]]:
    """Execute the v8 raw-Git -> H0 -> inner -> H1 -> H2 -> final chain."""

    repository = Path(repo_root)
    if not repository.is_absolute():
        raise launcher.EvidenceContractError("repository root must be absolute")
    launcher.assert_no_symlink_ancestors(repository)
    launcher._validate_lower_hex(evidence_commit_sha, 40, label="literal evidence SHA")
    launcher._validate_lower_hex(
        attestation_commit_sha, 40, label="literal attestation SHA"
    )
    launcher._validate_lower_hex(
        attestation_payload_sha256,
        64,
        label="literal payload SHA256",
    )
    if dry_run:
        if command_envelope_path is not None:
            raise launcher.EvidenceContractError(
                "dry-run must not receive a scientific command envelope"
            )
        command_value: dict[str, object] | None = None
        command: launcher.AttributionCommandEnvelope | None = None
    else:
        if command_envelope_path is None:
            raise launcher.EvidenceContractError(
                "non-dry execution requires a command envelope"
            )
        command_value, command = _load_command_envelope(command_envelope_path)

    git = _RawGitReader.create(repository)
    git_version = git.run("--version").decode("ascii", errors="strict").strip()
    attestation_commit = _read_commit(git, attestation_commit_sha)
    if len(attestation_commit.parents) != 1:
        raise launcher.EvidenceContractError(
            "attestation commit does not have exactly one raw parent"
        )
    evidence_sha = evidence_commit_sha
    if attestation_commit.parents != (evidence_sha,):
        raise launcher.EvidenceContractError(
            "attestation raw parent differs from literal evidence SHA"
        )
    evidence_commit = _read_commit(git, evidence_sha)
    evidence_tree = _read_flat_tree(git, evidence_sha)
    attestation_tree = _read_flat_tree(git, attestation_commit_sha)
    diff_name_status = _read_tree_delta(git, evidence_sha, attestation_commit_sha)
    launcher.validate_one_added_payload_delta(
        evidence_tree,
        attestation_tree,
        diff_name_status=diff_name_status,
    )
    payload_blob = _read_blob(git, attestation_tree, launcher.ATTESTATION_PAYLOAD_PATH)
    if launcher.sha256_bytes(payload_blob) != attestation_payload_sha256:
        raise launcher.EvidenceContractError(
            "published payload exact-byte hash differs"
        )
    authorization_blob = _read_blob(git, attestation_tree, launcher.AUTHORIZATION_PATH)
    launcher_blob = _read_blob(git, attestation_tree, launcher.LAUNCHER_PATH)
    verified = launcher.verify_attestation_bundle(
        evidence_commit=evidence_commit,
        attestation_commit=attestation_commit,
        evidence_tree=evidence_tree,
        attestation_tree=attestation_tree,
        diff_name_status=diff_name_status,
        payload_blob_bytes=payload_blob,
        expected_payload_sha256=attestation_payload_sha256,
        authorization_blob_bytes=authorization_blob,
        launcher_blob_bytes=launcher_blob,
    )
    blobs = _closure_blobs(
        git,
        evidence_tree,
        verified.authorization,
        launcher_blob,
    )

    execution_label = "dry-run" if command is None else command.command
    layout = launcher.create_execution_layout(
        repository,
        evidence_sha,
        allow_temporary_test_root=allow_synthetic_temp_repo,
        execution_label=execution_label,
    )
    launcher.materialize_runtime_source_closure(
        layout.source_root,
        verified.authorization.runtime_source_closure,
        blobs,
    )
    closure_hash = launcher.seal_source_closure_readonly(
        layout.source_root, verified.authorization.runtime_source_closure
    )
    authorization_value = launcher.parse_strict_json_object(
        authorization_blob, label="execution authorization"
    )
    bindings = {
        "evidence_commit_sha": verified.evidence_commit_sha,
        "attestation_commit_sha": verified.attestation_commit_sha,
        "attestation_payload_sha256": verified.attestation_payload_sha256,
        "authorization_blob_sha256": verified.authorization_blob_sha256,
        "launcher_blob_sha256": verified.launcher_blob_sha256,
    }
    h0 = launcher.build_receipt_envelope(
        "outer_preflight",
        upstream_receipts={},
        body={
            "status": "PASS",
            "dry_run": dry_run,
            "repo_root": str(repository),
            "bindings": bindings,
            "authorization": authorization_value,
            "runtime_source_closure": launcher.runtime_source_closure_to_json(
                verified.authorization.runtime_source_closure
            ),
            "runtime_source_closure_sha256": closure_hash,
            "execution_layout": _layout_json(layout),
            "bootstrap_commands": [record.to_json() for record in git.records],
            "outer_binary_identities": {
                "git": {
                    **_binary_identity(Path(launcher.GIT_EXECUTABLE)),
                    "version": git_version,
                },
                "python": {
                    **_binary_identity(Path(launcher.PYTHON_EXECUTABLE_REALPATH)),
                    "authorized_version": (
                        verified.authorization.runtime_environment_manifest.python_version
                    ),
                },
            },
            "git_no_replace_objects": True,
            "git_config_nosystem": True,
            "git_config_global": "/dev/null",
            "git_hooks_disabled": True,
            "checkout_used": False,
            "raw_blob_materialization": True,
            "command_envelope": command_value,
        },
    )
    launcher.write_immutable_receipt(layout.output_root, launcher.H0_RECEIPT_NAME, h0)

    materialized_launcher = layout.source_root.joinpath(
        *PurePosixPath(launcher.LAUNCHER_PATH).parts
    )
    python_argv = (
        *launcher.isolated_python_argv(materialized_launcher),
        "--inner",
        "--h0-path",
        str(layout.output_root / launcher.H0_RECEIPT_NAME),
        "--h0-sha256",
        h0.sha256,
        "--source-root",
        str(layout.source_root),
        "--output-root",
        str(layout.output_root),
        "--current-worktree",
        str(repository),
    )
    if dry_run:
        python_argv = (*python_argv, "--dry-run")
    completed = subprocess.run(
        python_argv,
        check=False,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=_python_inner_environment(),
        cwd=layout.output_root,
    )

    h1_path = layout.output_root / launcher.H1_RECEIPT_NAME
    h1 = launcher.read_immutable_receipt(h1_path, stage="inner_launcher")
    if launcher._parse_and_validate_receipt_envelope(h1)["upstream_receipts"] != {
        "H0": h0.sha256
    }:
        raise launcher.EvidenceContractError("inner H1 does not bind literal H0")

    post_source_status = "PASS"
    post_output_status = "PASS"
    post_errors: list[str] = []
    try:
        launcher.verify_materialized_closure(
            layout.source_root,
            verified.authorization.runtime_source_closure,
            require_readonly=True,
        )
    except launcher.EvidenceContractError as error:
        post_source_status = "FAIL"
        post_errors.append(str(error))
    expected_pre_completion = [
        launcher.H0_RECEIPT_NAME,
        launcher.H1_RECEIPT_NAME,
    ]
    if command is not None:
        expected_pre_completion.append(
            launcher.ATTRIBUTION_INTERMEDIATE_OUTPUTS[command.command]
        )
    expected_pre_completion.sort()
    try:
        launcher.verify_output_inventory(layout.output_root, expected_pre_completion)
    except launcher.EvidenceContractError as error:
        post_output_status = "FAIL"
        post_errors.append(str(error))
    raw_scientific_payload: dict[str, object] | None = None
    raw_payload_exact_sha256: str | None = None
    raw_payload_canonical_sha256: str | None = None
    scientific_status = (
        "NOT_RUN_DRY_RUN" if command is None else "UNAVAILABLE_UPSTREAM_FAILURE"
    )
    return_status_match = command is None
    intermediate_payload_cleanup_status = "NOT_APPLICABLE"
    cleanup_status = "PASS"
    if command is not None:
        intermediate_name = launcher.ATTRIBUTION_INTERMEDIATE_OUTPUTS[command.command]
        intermediate_path = layout.output_root / intermediate_name
        intermediate_present = False
        try:
            launcher.assert_no_symlink_ancestors(intermediate_path)
            intermediate_metadata = os.lstat(intermediate_path)
            intermediate_present = True
            if not stat.S_ISREG(intermediate_metadata.st_mode):
                raise launcher.EvidenceContractError(
                    "intermediate scientific payload is not regular"
                )
            if completed.returncode in {0, 2}:
                intermediate_raw = intermediate_path.read_bytes()
                raw_scientific_payload = launcher.parse_strict_json_object(
                    intermediate_raw,
                    label="intermediate scientific payload",
                )
                raw_payload_exact_sha256 = launcher.sha256_bytes(intermediate_raw)
                raw_payload_canonical_sha256 = (
                    launcher.scientific_payload_canonical_sha256(raw_scientific_payload)
                )
                raw_status = raw_scientific_payload.get("status")
                if (
                    type(raw_status) is not str
                    or raw_status
                    not in launcher.ATTRIBUTION_SCIENTIFIC_STATUSES[command.command]
                ):
                    raise launcher.EvidenceContractError(
                        "intermediate scientific status is not authorized"
                    )
                scientific_status = raw_status
                expected_return_code = (
                    2 if scientific_status == "INVALID_EVIDENCE" else 0
                )
                return_status_match = completed.returncode == expected_return_code
                if not return_status_match:
                    post_errors.append("inner exit code/scientific status mismatch")
            else:
                post_errors.append("inner exit code is not transport-authorized")
        except FileNotFoundError:
            intermediate_payload_cleanup_status = "NOT_PRESENT"
        except (OSError, launcher.EvidenceContractError) as error:
            post_errors.append(str(error))
            intermediate_payload_cleanup_status = "FAIL"
        finally:
            if intermediate_present:
                try:
                    os.unlink(intermediate_path)
                    intermediate_payload_cleanup_status = "PASS"
                except OSError as error:
                    post_errors.append(str(error))
                    intermediate_payload_cleanup_status = "FAIL"
            cleanup_status = (
                "PASS"
                if intermediate_payload_cleanup_status in {"PASS", "NOT_PRESENT"}
                else "FAIL"
            )
        expected_pre_completion = sorted(
            [
                launcher.H0_RECEIPT_NAME,
                launcher.H1_RECEIPT_NAME,
            ]
        )
        try:
            launcher.verify_output_inventory(
                layout.output_root,
                expected_pre_completion,
            )
        except launcher.EvidenceContractError as error:
            post_output_status = "FAIL"
            post_errors.append(str(error))
    transport_pass = (
        completed.returncode == 0
        and post_source_status == "PASS"
        and post_output_status == "PASS"
        if command is None
        else (
            completed.returncode in {0, 2}
            and post_source_status == "PASS"
            and post_output_status == "PASS"
            and raw_scientific_payload is not None
            and return_status_match
            and intermediate_payload_cleanup_status == "PASS"
            and cleanup_status == "PASS"
        )
    )
    scientific_invalid = scientific_status == "INVALID_EVIDENCE"
    h2 = launcher.build_receipt_envelope(
        "outer_completion",
        upstream_receipts={"H0": h0.sha256, "H1": h1.sha256},
        body={
            "status": "PASS" if transport_pass else "INVALID_EVIDENCE",
            "transport_status": ("PASS" if transport_pass else "INVALID_EVIDENCE"),
            "scientific_status": scientific_status,
            "inner_python_argv": list(python_argv),
            "inner_exit_code": completed.returncode,
            "inner_stdout_sha256": launcher.sha256_bytes(completed.stdout),
            "inner_stderr_sha256": launcher.sha256_bytes(completed.stderr),
            "post_run_source_closure": post_source_status,
            "post_run_output_inventory": post_output_status,
            "post_run_errors": post_errors,
            "intermediate_payload_exact_sha256": raw_payload_exact_sha256,
            "intermediate_payload_canonical_sha256": (raw_payload_canonical_sha256),
            "intermediate_payload_cleanup_status": (
                intermediate_payload_cleanup_status
            ),
            "cleanup_status": cleanup_status,
            "cleanup_scope": "INTERMEDIATE_PAYLOAD_ONLY",
            "cleanup_target_relative_paths": (
                []
                if command is None
                else [launcher.ATTRIBUTION_INTERMEDIATE_OUTPUTS[command.command]]
            ),
            "execution_layout_cleanup_status": "NOT_RUN_RETAINED_FOR_AUDIT",
        },
    )
    launcher.write_immutable_receipt(layout.output_root, launcher.H2_RECEIPT_NAME, h2)
    wrapped_payload_sha256: str | None = None
    scientific_output_path: str | None = None
    if transport_pass and command is not None:
        assert raw_scientific_payload is not None
        execution_envelope = launcher.build_scientific_execution_envelope(
            stage=command.command,
            bindings=bindings,
            runtime_source_closure_sha256_value=closure_hash,
            h0=h0,
            h1=h1,
            h2=h2,
        )
        wrapped_payload = launcher.wrap_scientific_payload(
            stage=command.command,
            payload=raw_scientific_payload,
            execution_envelope=execution_envelope,
        )
        launcher.validate_wrapped_scientific_payload(
            wrapped_payload, stage=command.command
        )
        wrapped_payload_raw = launcher.canonical_json_bytes(wrapped_payload)
        wrapped_payload_sha256 = launcher.write_immutable_bytes(
            layout.output_root,
            command.output_relative_path,
            wrapped_payload_raw,
        )
        scientific_output_path = str(layout.output_root / command.output_relative_path)
    expected_before_final = [
        *expected_pre_completion,
        launcher.H2_RECEIPT_NAME,
    ]
    if wrapped_payload_sha256 is not None and command is not None:
        expected_before_final.append(command.output_relative_path)
    expected_before_final.sort()
    if transport_pass:
        launcher.verify_output_inventory(
            layout.output_root,
            expected_before_final,
        )
    receipt_kind = "selector" if command is None else command.receipt_kind
    final = launcher.build_receipt_envelope(
        "final",
        upstream_receipts={
            "H0": h0.sha256,
            "H1": h1.sha256,
            "H2": h2.sha256,
        },
        body={
            "receipt_kind": receipt_kind,
            "status": "DRY_RUN_PASS"
            if transport_pass and dry_run
            else (
                "PASS"
                if transport_pass and not scientific_invalid
                else "INVALID_EVIDENCE"
            ),
            "transport_status": ("PASS" if transport_pass else "INVALID_EVIDENCE"),
            "scientific_status": scientific_status,
            "bindings": bindings,
            "scientific_payload_envelope_sha256": wrapped_payload_sha256,
            "scientific_output_path": scientific_output_path,
        },
    )
    launcher.write_immutable_receipt(
        layout.output_root, launcher.FINAL_RECEIPT_NAME, final
    )
    launcher.validate_receipt_chain(h0, h1, h2, final)

    if transport_pass:
        expected_final = sorted(
            [
                *expected_before_final,
                launcher.FINAL_RECEIPT_NAME,
            ]
        )
        launcher.verify_output_inventory(layout.output_root, expected_final)
    result_status = (
        "PASS" if transport_pass and not scientific_invalid else "INVALID_EVIDENCE"
    )
    summary = {
        "status": result_status,
        "transport_status": ("PASS" if transport_pass else "INVALID_EVIDENCE"),
        "scientific_status": scientific_status,
        "dry_run": dry_run,
        "execution_root": str(layout.root),
        "final_receipt": str(layout.output_root / launcher.FINAL_RECEIPT_NAME),
        "final_receipt_sha256": final.sha256,
        "scientific_output": scientific_output_path,
        "scientific_payload_envelope_sha256": wrapped_payload_sha256,
    }
    return (0 if result_status == "PASS" else 2), summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Raw-object Fig17/18/19 evidence bootstrap"
    )
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--evidence-commit", required=True)
    parser.add_argument("--attestation-commit", required=True)
    parser.add_argument("--attestation-payload-sha256", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--command-envelope", type=Path)
    parser.add_argument(
        "--synthetic-temporary-repo",
        action="store_true",
        help="dry-run-only test override for a temporary synthetic repository",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point.  Errors are fail-closed and never trigger discovery."""

    parser = _build_parser()
    arguments = parser.parse_args(argv)
    try:
        if arguments.synthetic_temporary_repo and not arguments.dry_run:
            raise launcher.EvidenceContractError(
                "temporary-repository CLI override is dry-run-only"
            )
        return_code, summary = run_outer_bootstrap(
            repo_root=arguments.repo_root,
            evidence_commit_sha=arguments.evidence_commit,
            attestation_commit_sha=arguments.attestation_commit,
            attestation_payload_sha256=(arguments.attestation_payload_sha256),
            dry_run=arguments.dry_run,
            command_envelope_path=arguments.command_envelope,
            allow_synthetic_temp_repo=arguments.synthetic_temporary_repo,
        )
    except launcher.EvidenceContractError as error:
        print(f"INVALID_EVIDENCE: {error}", file=sys.stderr)
        return 2
    print(json.dumps(summary, sort_keys=True, separators=(",", ":")))
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
