"""Pure, no-live-data tests for the Fig17/18/19 evidence launcher."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
from types import SimpleNamespace

import pytest


PLATFORM = Path(__file__).resolve().parents[1]
if str(PLATFORM) not in sys.path:
    sys.path.insert(0, str(PLATFORM))

import fig171819_evidence_launcher as launcher  # noqa: E402


HEX40_A = "a" * 40
HEX40_B = "b" * 40
HEX40_C = "c" * 40


def _json_bytes(value: object) -> bytes:
    return json.dumps(value, separators=(",", ":")).encode() + b"\n"


def _source_entry(path: str, raw: bytes, mode: str = "100644"):
    return {
        "repo_relative_path": path,
        "git_mode": mode,
        "git_blob_oid": launcher.git_blob_oid(raw),
        "sha256": launcher.sha256_bytes(raw),
        "size_bytes": len(raw),
    }


def _authorization_value(source_raw: bytes = b"VALUE = 1\n"):
    return {
        "schema_version": 1,
        "artifact_type": ("fig171819_active_disease_execution_authorization"),
        "launcher_contract": launcher.LAUNCHER_CONTRACT,
        "runtime_source_closure": [_source_entry("platform/selector.py", source_raw)],
        "runtime_environment_manifest": {
            "python_executable_realpath": launcher.PYTHON_EXECUTABLE_REALPATH,
            "python_implementation": "CPython",
            "python_version": "3.11.fixture",
            "python_flags": ["-I", "-S", "-B"],
            "stdlib_roots": ["/frozen/stdlib"],
            "site_packages_roots": ["/frozen/site-packages"],
            "distributions": [
                {
                    "name": "numpy",
                    "version": "fixture",
                    "files": [
                        {
                            "absolute_path": "/frozen/site-packages/numpy.py",
                            "sha256": "d" * 64,
                            "size_bytes": 7,
                        }
                    ],
                }
            ],
        },
        "output_relative_path": "output",
    }


def _payload_value(authorization_raw: bytes, launcher_raw: bytes):
    return {
        "schema_version": 1,
        "artifact_type": "fig171819_active_disease_external_attestation",
        "evidence_commit_sha": HEX40_A,
        "authorization_path": launcher.AUTHORIZATION_PATH,
        "authorization_blob_sha256": launcher.sha256_bytes(authorization_raw),
        "launcher_path": launcher.LAUNCHER_PATH,
        "launcher_blob_sha256": launcher.sha256_bytes(launcher_raw),
        "attestation_payload_path": launcher.ATTESTATION_PAYLOAD_PATH,
        "fresh_run_id": "20260729_135128",
        "fresh_status_when_attested": "running",
        "launcher_contract": launcher.LAUNCHER_CONTRACT,
    }


def _raw_entry(path: str, raw: bytes, mode: str = "100644"):
    return launcher.RawTreeEntry(
        repo_relative_path=path,
        git_mode=mode,
        git_blob_oid=launcher.git_blob_oid(raw),
    )


def _attestation_fixture():
    launcher_raw = b"import sys\nsys.dont_write_bytecode = True\n"
    authorization_raw = _json_bytes(_authorization_value())
    payload_raw = _json_bytes(_payload_value(authorization_raw, launcher_raw))
    evidence_tree = {
        launcher.AUTHORIZATION_PATH: _raw_entry(
            launcher.AUTHORIZATION_PATH, authorization_raw
        ),
        launcher.LAUNCHER_PATH: _raw_entry(
            launcher.LAUNCHER_PATH, launcher_raw, "100755"
        ),
    }
    attestation_tree = dict(evidence_tree)
    attestation_tree[launcher.ATTESTATION_PAYLOAD_PATH] = _raw_entry(
        launcher.ATTESTATION_PAYLOAD_PATH, payload_raw
    )
    return {
        "evidence_commit": launcher.RawCommitMetadata(
            commit_sha=HEX40_A, tree_oid=HEX40_B, parents=(HEX40_C,)
        ),
        "attestation_commit": launcher.RawCommitMetadata(
            commit_sha=HEX40_B, tree_oid=HEX40_C, parents=(HEX40_A,)
        ),
        "evidence_tree": evidence_tree,
        "attestation_tree": attestation_tree,
        "diff_name_status": [f"A\t{launcher.ATTESTATION_PAYLOAD_PATH}"],
        "payload_blob_bytes": payload_raw,
        "expected_payload_sha256": launcher.sha256_bytes(payload_raw),
        "authorization_blob_bytes": authorization_raw,
        "launcher_blob_bytes": launcher_raw,
    }


def _runtime_entries(*items: tuple[str, bytes, str]):
    return tuple(
        launcher.RuntimeSourceEntry.from_json(
            _source_entry(path, raw, mode), index=index
        )
        for index, (path, raw, mode) in enumerate(items)
    )


def _make_layout_parent(repo: Path) -> None:
    (repo / launcher.EXECUTION_ROOT_RELATIVE).mkdir(parents=True)


def _unseal_for_cleanup(source_root: Path) -> None:
    os.chmod(source_root, 0o700)
    directories = sorted(
        (path for path in source_root.rglob("*") if path.is_dir()),
        key=lambda path: len(path.parts),
    )
    for path in directories:
        os.chmod(path, 0o700)


def test_raw_git_contract_disables_replacement_hooks_filters_and_env(tmp_path):
    argv = launcher.raw_git_argv(tmp_path, "cat-file", "-p", HEX40_A)
    environment = launcher.sanitized_bootstrap_environment(
        {
            "PATH": "/usr/bin",
            "LD_PRELOAD": "hostile.so",
            "PYTHONPATH": "/current/worktree",
            "GIT_DIR": "/wrong/repo",
        }
    )
    launcher.validate_raw_git_invocation(argv, environment)
    assert argv[:7] == (
        "/usr/bin/git",
        "--no-replace-objects",
        "-c",
        "core.hooksPath=/dev/null",
        "-c",
        "filter.lfs.required=false",
        "-C",
    )
    assert environment["GIT_NO_REPLACE_OBJECTS"] == "1"
    assert environment["GIT_CONFIG_NOSYSTEM"] == "1"
    assert environment["GIT_CONFIG_GLOBAL"] == "/dev/null"
    for key in launcher.GIT_CLEARED_ENVIRONMENT:
        assert key not in environment

    with pytest.raises(launcher.EvidenceContractError):
        launcher.raw_git_argv(tmp_path, "checkout", HEX40_A)
    bad_environment = dict(environment, GIT_NO_REPLACE_OBJECTS="0")
    with pytest.raises(launcher.EvidenceContractError):
        launcher.validate_raw_git_invocation(argv, bad_environment)


def test_attestation_exact_one_added_payload_and_blob_bytes_pass():
    fixture = _attestation_fixture()
    verified = launcher.verify_attestation_bundle(**fixture)
    assert verified.evidence_commit_sha == HEX40_A
    assert verified.attestation_commit_sha == HEX40_B
    assert verified.attestation_payload_sha256 == fixture["expected_payload_sha256"]
    assert verified.authorization.runtime_source_closure[0].repo_relative_path == (
        "platform/selector.py"
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "extra_path",
        "changed_launcher_mode",
        "payload_preexists",
        "wrong_parent",
        "rename_delta",
    ],
)
def test_attestation_tree_delta_fail_closed(mutation):
    fixture = _attestation_fixture()
    if mutation == "extra_path":
        extra_raw = b"extra\n"
        fixture["attestation_tree"]["extra.txt"] = _raw_entry("extra.txt", extra_raw)
        fixture["diff_name_status"].append("A\textra.txt")
    elif mutation == "changed_launcher_mode":
        raw = fixture["launcher_blob_bytes"]
        fixture["attestation_tree"][launcher.LAUNCHER_PATH] = _raw_entry(
            launcher.LAUNCHER_PATH, raw, "100644"
        )
    elif mutation == "payload_preexists":
        fixture["evidence_tree"][launcher.ATTESTATION_PAYLOAD_PATH] = fixture[
            "attestation_tree"
        ][launcher.ATTESTATION_PAYLOAD_PATH]
    elif mutation == "wrong_parent":
        fixture["attestation_commit"] = launcher.RawCommitMetadata(
            commit_sha=HEX40_B, tree_oid=HEX40_C, parents=(HEX40_C,)
        )
    else:
        fixture["diff_name_status"] = [
            f"R100\ta.json\t{launcher.ATTESTATION_PAYLOAD_PATH}"
        ]
    with pytest.raises(launcher.EvidenceContractError):
        launcher.verify_attestation_bundle(**fixture)


def test_payload_hash_is_exact_blob_bytes_not_reserialized_json():
    fixture = _attestation_fixture()
    original = fixture["payload_blob_bytes"]
    parsed = json.loads(original)
    respaced = json.dumps(parsed, indent=2).encode() + b"\n"
    assert json.loads(respaced) == parsed
    assert launcher.sha256_bytes(respaced) != launcher.sha256_bytes(original)
    fixture["payload_blob_bytes"] = respaced
    fixture["attestation_tree"][launcher.ATTESTATION_PAYLOAD_PATH] = _raw_entry(
        launcher.ATTESTATION_PAYLOAD_PATH, respaced
    )
    with pytest.raises(
        launcher.EvidenceContractError, match="exact-byte hash mismatch"
    ):
        launcher.verify_attestation_bundle(**fixture)


def test_duplicate_or_unknown_attestation_fields_fail_closed():
    fixture = _attestation_fixture()
    raw = fixture["payload_blob_bytes"]
    duplicate = raw[:-2] + b',"schema_version":1}\\n'
    with pytest.raises(launcher.EvidenceContractError, match="duplicate JSON key"):
        launcher.parse_attestation_payload(duplicate)
    value = json.loads(raw)
    value["unexpected"] = True
    with pytest.raises(launcher.EvidenceContractError, match="keys differ"):
        launcher.parse_attestation_payload(_json_bytes(value))


def test_authorization_schema_rejects_commit_fields_modes_and_path_aliases():
    base = _authorization_value()
    parsed = launcher.parse_execution_authorization(_json_bytes(base))
    assert parsed.runtime_environment_manifest.python_flags == ("-I", "-S", "-B")

    forbidden = copy.deepcopy(base)
    forbidden["runtime_environment_manifest"]["attestation_commit_sha"] = HEX40_A
    with pytest.raises(launcher.EvidenceContractError, match="forbidden"):
        launcher.parse_execution_authorization(_json_bytes(forbidden))

    symlink_mode = copy.deepcopy(base)
    symlink_mode["runtime_source_closure"][0]["git_mode"] = "120000"
    with pytest.raises(launcher.EvidenceContractError, match="mode"):
        launcher.parse_execution_authorization(_json_bytes(symlink_mode))

    aliases = copy.deepcopy(base)
    second = copy.deepcopy(aliases["runtime_source_closure"][0])
    second["repo_relative_path"] = "platform/Selector.py"
    aliases["runtime_source_closure"].append(second)
    aliases["runtime_source_closure"].sort(key=lambda item: item["repo_relative_path"])
    with pytest.raises(launcher.EvidenceContractError, match="casefold"):
        launcher.parse_execution_authorization(_json_bytes(aliases))


@pytest.mark.parametrize(
    "path",
    [
        "../selector.py",
        "/selector.py",
        "platform//selector.py",
        "platform\\selector.py",
        "platform/__pycache__/selector.py",
        "platform/selector.pyc",
        "platform/injection.pth",
    ],
)
def test_runtime_closure_rejects_traversal_and_startup_artifacts(path):
    raw = b"x = 1\n"
    with pytest.raises(launcher.EvidenceContractError):
        launcher.RuntimeSourceEntry.from_json(_source_entry(path, raw), index=0)


def test_materialization_exact_closure_seals_and_detects_extra_file(tmp_path):
    repo = tmp_path / "repo"
    _make_layout_parent(repo)
    layout = launcher.create_execution_layout(
        repo, HEX40_A, allow_temporary_test_root=True
    )
    raw_a = b"VALUE = 1\n"
    raw_b = b"#!/usr/bin/env python\n"
    entries = _runtime_entries(
        ("a.py", raw_a, "100644"),
        ("pkg/run.py", raw_b, "100755"),
    )
    blobs = {
        launcher.git_blob_oid(raw_a): raw_a,
        launcher.git_blob_oid(raw_b): raw_b,
    }
    launcher.materialize_runtime_source_closure(layout.source_root, entries, blobs)
    expected_hash = launcher.runtime_source_closure_sha256(entries)
    assert (
        launcher.verify_materialized_closure(layout.source_root, entries)
        == expected_hash
    )
    assert (
        launcher.seal_source_closure_readonly(layout.source_root, entries)
        == expected_hash
    )
    assert not (os.lstat(layout.source_root / "a.py").st_mode & stat.S_IWUSR)

    _unseal_for_cleanup(layout.source_root)
    (layout.source_root / "extra.py").write_bytes(b"not authorized\n")
    with pytest.raises(launcher.EvidenceContractError, match="path set"):
        launcher.verify_materialized_closure(layout.source_root, entries)


def test_materialization_rejects_symlink_and_wrong_blob(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    (root / "escape.py").symlink_to(tmp_path / "elsewhere")
    raw = b"safe\n"
    entries = _runtime_entries(("escape.py", raw, "100644"))
    blobs = {launcher.git_blob_oid(raw): raw}
    with pytest.raises(launcher.EvidenceContractError):
        launcher.materialize_runtime_source_closure(root, entries, blobs)

    clean = tmp_path / "clean"
    clean.mkdir()
    wrong_blobs = {launcher.git_blob_oid(raw): b"different\n"}
    with pytest.raises(launcher.EvidenceContractError, match="metadata mismatch"):
        launcher.materialize_runtime_source_closure(clean, entries, wrong_blobs)


def test_execution_layout_requires_fixed_repo_local_preexisting_base(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    with pytest.raises(launcher.EvidenceContractError, match="missing"):
        launcher.create_execution_layout(repo, HEX40_A, allow_temporary_test_root=True)
    _make_layout_parent(repo)
    layout = launcher.create_execution_layout(
        repo, HEX40_A, allow_temporary_test_root=True
    )
    assert layout.root.parent == repo / launcher.EXECUTION_ROOT_RELATIVE
    assert HEX40_A in layout.root.name
    assert Path(layout.root_realpath) == layout.root
    assert layout.source_root.parent == layout.output_root.parent
    with pytest.raises(FileExistsError):
        launcher.create_execution_layout(repo, HEX40_A, allow_temporary_test_root=True)


def test_production_materialization_rejects_tmp_without_test_override(tmp_path):
    repo = tmp_path / "repo"
    _make_layout_parent(repo)
    with pytest.raises(launcher.EvidenceContractError, match="/tmp"):
        launcher.create_execution_layout(repo, HEX40_A)


def test_execution_layout_rejects_symlink_ancestor(tmp_path):
    repository = tmp_path / "repo"
    repository.mkdir()
    real_platform = tmp_path / "real-platform"
    (real_platform / "docs" / "diag" / "evidence_exec").mkdir(parents=True)
    (repository / "platform").symlink_to(real_platform)
    with pytest.raises(launcher.EvidenceContractError, match="symlink"):
        launcher.create_execution_layout(
            repository, HEX40_A, allow_temporary_test_root=True
        )


def test_isolated_python_argv_and_state_are_fail_closed(tmp_path):
    verified = tmp_path / "verified-launcher.py"
    argv = launcher.isolated_python_argv(verified)
    assert argv == (
        launcher.PYTHON_EXECUTABLE_REALPATH,
        "-I",
        "-S",
        "-B",
        str(verified),
    )
    with pytest.raises(launcher.EvidenceContractError, match="flags"):
        launcher.validate_isolated_python_flags()
    worktree = tmp_path / "worktree"
    source = tmp_path / "materialized"
    cwd = tmp_path / "cwd"
    worktree.mkdir()
    source.mkdir()
    cwd.mkdir()
    launcher.validate_isolated_python_state(
        current_worktree=worktree,
        materialized_source_root=source,
        sys_path=["/frozen/stdlib", "/frozen/site-packages"],
        loaded_modules={},
        current_working_directory=cwd,
    )
    with pytest.raises(launcher.EvidenceContractError, match="contaminates"):
        launcher.validate_isolated_python_state(
            current_worktree=worktree,
            materialized_source_root=source,
            sys_path=[str(worktree / "platform")],
            loaded_modules={},
            current_working_directory=cwd,
        )
    with pytest.raises(launcher.EvidenceContractError, match="sitecustomize"):
        launcher.validate_isolated_python_state(
            current_worktree=worktree,
            materialized_source_root=source,
            sys_path=["/frozen/stdlib"],
            loaded_modules={"sitecustomize": object()},
            current_working_directory=cwd,
        )
    with pytest.raises(launcher.EvidenceContractError, match="project module"):
        launcher.validate_isolated_python_state(
            current_worktree=worktree,
            materialized_source_root=source,
            sys_path=["/frozen/stdlib"],
            loaded_modules={
                "polluted": SimpleNamespace(
                    __file__=str(worktree / "platform" / "polluted.py")
                )
            },
            current_working_directory=cwd,
        )


def test_authorized_roots_use_direct_ordered_insertion():
    target = ["/existing"]
    inserted = launcher.insert_authorized_python_roots(
        ["/frozen/stdlib", "/frozen/site-packages"],
        target_sys_path=target,
    )
    assert inserted == ("/frozen/stdlib", "/frozen/site-packages")
    assert target == [
        "/frozen/stdlib",
        "/frozen/site-packages",
        "/existing",
    ]


def test_receipt_dag_is_one_way_exact_and_immutable(tmp_path):
    h0_envelope = launcher.build_receipt_envelope(
        "outer_preflight",
        upstream_receipts={},
        body={"bootstrap_commands": [], "status": "PASS"},
    )
    h1_envelope = launcher.build_receipt_envelope(
        "inner_launcher",
        upstream_receipts={"H0": h0_envelope.sha256},
        body={"python_startup_contamination_check": "PASS"},
    )
    h2_envelope = launcher.build_receipt_envelope(
        "outer_completion",
        upstream_receipts={
            "H0": h0_envelope.sha256,
            "H1": h1_envelope.sha256,
        },
        body={
            "post_run_source_closure": "PASS",
            "cleanup_status": "COMPLETE",
        },
    )
    final = launcher.build_receipt_envelope(
        "final",
        upstream_receipts={
            "H0": h0_envelope.sha256,
            "H1": h1_envelope.sha256,
            "H2": h2_envelope.sha256,
        },
        body={"receipt_kind": "selector", "status": "NO_DECISION"},
    )
    assert launcher.validate_receipt_chain(
        h0_envelope, h1_envelope, h2_envelope, final
    ) == (
        h0_envelope.sha256,
        h1_envelope.sha256,
        h2_envelope.sha256,
    )

    receipt_path = tmp_path / "H0.json"
    assert (
        launcher.write_immutable_receipt(tmp_path, "H0.json", h0_envelope)
        == h0_envelope.sha256
    )
    assert hashlib.sha256(receipt_path.read_bytes()).hexdigest() == (h0_envelope.sha256)
    assert not (os.lstat(receipt_path).st_mode & stat.S_IWUSR)
    with pytest.raises(FileExistsError):
        launcher.write_immutable_receipt(tmp_path, "H0.json", h0_envelope)


def test_output_writer_cannot_escape_and_inventory_rejects_extras(tmp_path):
    output_root = tmp_path / "output"
    output_root.mkdir()
    digest = launcher.write_immutable_bytes(
        output_root, "receipt.json", b'{"fixture":true}\n'
    )
    assert digest == launcher.sha256_bytes(b'{"fixture":true}\n')
    assert launcher.verify_output_inventory(output_root, ["receipt.json"]) == {
        "receipt.json": digest
    }
    with pytest.raises(launcher.EvidenceContractError):
        launcher.write_immutable_bytes(output_root, "../escape.json", b"x")
    (output_root / "extra.json").write_bytes(b"extra\n")
    with pytest.raises(launcher.EvidenceContractError, match="path set"):
        launcher.verify_output_inventory(output_root, ["receipt.json"])


def test_receipts_reject_self_downstream_and_incomplete_completion():
    with pytest.raises(launcher.EvidenceContractError, match="upstream"):
        launcher.build_receipt_envelope(
            "inner_launcher",
            upstream_receipts={"H1": "a" * 64},
            body={"status": "PASS"},
        )
    with pytest.raises(launcher.EvidenceContractError, match="self-hash"):
        launcher.build_receipt_envelope(
            "outer_preflight",
            upstream_receipts={},
            body={"receipt_sha256": "a" * 64},
        )
    with pytest.raises(launcher.EvidenceContractError, match="post-run"):
        launcher.build_receipt_envelope(
            "outer_completion",
            upstream_receipts={"H0": "a" * 64, "H1": "b" * 64},
            body={"cleanup_status": "COMPLETE"},
        )


def test_receipt_chain_detects_byte_tampering():
    h0 = launcher.build_receipt_envelope(
        "outer_preflight", upstream_receipts={}, body={"status": "PASS"}
    )
    h1 = launcher.build_receipt_envelope(
        "inner_launcher",
        upstream_receipts={"H0": h0.sha256},
        body={"status": "PASS"},
    )
    h2 = launcher.build_receipt_envelope(
        "outer_completion",
        upstream_receipts={"H0": h0.sha256, "H1": h1.sha256},
        body={
            "post_run_source_closure": "PASS",
            "cleanup_status": "COMPLETE",
        },
    )
    final = launcher.build_receipt_envelope(
        "final",
        upstream_receipts={
            "H0": h0.sha256,
            "H1": h1.sha256,
            "H2": h2.sha256,
        },
        body={"receipt_kind": "evaluate"},
    )
    tampered = launcher.ReceiptEnvelope(
        stage="inner_launcher",
        exact_bytes=h1.exact_bytes + b" ",
        sha256=h1.sha256,
    )
    with pytest.raises(launcher.EvidenceContractError, match="byte hash"):
        launcher.validate_receipt_chain(h0, tampered, h2, final)
