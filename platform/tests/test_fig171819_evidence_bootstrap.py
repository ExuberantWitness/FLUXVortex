"""Synthetic-repository tests for the v8 outer/inner evidence bootstrap."""

from __future__ import annotations

import copy
from dataclasses import dataclass
import json
import os
from pathlib import Path
import stat
import subprocess
import sys

import pytest


PLATFORM = Path(__file__).resolve().parents[1]
if str(PLATFORM) not in sys.path:
    sys.path.insert(0, str(PLATFORM))

import fig171819_evidence_bootstrap as bootstrap  # noqa: E402
import fig171819_evidence_launcher as launcher  # noqa: E402


def test_materialized_cli_starts_under_isolated_python(tmp_path):
    materialized = tmp_path / "materialized"
    materialized.mkdir()
    bootstrap_path = materialized / "fig171819_evidence_bootstrap.py"
    launcher_path = materialized / "fig171819_evidence_launcher.py"
    bootstrap_path.write_bytes(Path(bootstrap.__file__).read_bytes())
    launcher_path.write_bytes(Path(launcher.__file__).read_bytes())

    completed = subprocess.run(
        (
            launcher.PYTHON_EXECUTABLE_REALPATH,
            "-I",
            "-S",
            "-B",
            str(bootstrap_path),
            "--help",
        ),
        check=False,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=launcher.sanitized_bootstrap_environment(os.environ),
        cwd="/",
    )

    assert completed.returncode == 0, completed.stderr.decode(errors="replace")
    assert b"--evidence-commit" in completed.stdout
    assert not any(materialized.rglob("__pycache__"))


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
        + b"\n"
    )


def _git(repo: Path, *arguments: str, env: dict[str, str] | None = None) -> bytes:
    completed = subprocess.run(
        ("/usr/bin/git", "-C", str(repo), *arguments),
        check=False,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )
    if completed.returncode != 0:
        raise AssertionError(
            f"synthetic Git setup failed: {arguments!r}: "
            f"{completed.stderr.decode(errors='replace')}"
        )
    return completed.stdout


def _source_record(path: str, raw: bytes) -> dict[str, object]:
    return {
        "repo_relative_path": path,
        "git_mode": "100644",
        "git_blob_oid": launcher.git_blob_oid(raw),
        "sha256": launcher.sha256_bytes(raw),
        "size_bytes": len(raw),
    }


def _fixed_python_identity() -> dict[str, str]:
    code = (
        "import json,platform,sysconfig;"
        "print(json.dumps({"
        "'implementation':platform.python_implementation(),"
        "'version':platform.python_version(),"
        "'stdlib':sysconfig.get_path('stdlib'),"
        "'purelib':sysconfig.get_path('purelib')}))"
    )
    completed = subprocess.run(
        (
            launcher.PYTHON_EXECUTABLE_REALPATH,
            "-I",
            "-S",
            "-B",
            "-c",
            code,
        ),
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return json.loads(completed.stdout)


def _stub_attribution(behavior: str) -> bytes:
    common = """\
from pathlib import Path
import hashlib
import json
import os

def _canonical_hash(value):
    raw = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()

def _unwrap_previous(path, expected_stage):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    assert value["artifact_type"] == "fig171819_scientific_payload_envelope"
    assert value["stage"] == expected_stage
    assert value["production_execution_authorized"] is True
    assert value["payload_sha256"] == _canonical_hash(value["payload"])
    execution = value["execution_envelope"]
    assert execution["stage"] == expected_stage
    assert execution["cleanup_status"] == "PASS"
    assert len(execution["outer_completion_receipt_sha256"]) == 64
    return value["payload"]

def main(argv=None):
    argv = list(argv or ())
    command = argv[0]
    assert "--execution-envelope" not in argv
    if command == "prepare":
        assert "--baseline-receipt" not in argv
        disease_path = argv[argv.index("--disease-spec") + 1]
        _unwrap_previous(disease_path, "select-disease")
        payload = {
            "status": "PREPARED",
            "synthetic": "PREPARED",
            "saw_baseline_receipt": False,
        }
    else:
        payload = {"status": "ACTIVE_DISEASE_FROZEN", "synthetic": "PASS"}
    output = Path(argv[argv.index("--output") + 1])
"""
    bodies = {
        "success": """\
    output.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\\n",
        encoding="utf-8",
    )
    return 0
""",
        "failure": """\
    return 7
""",
        "scientific_invalid": """\
    payload = {
        "status": "INVALID_EVIDENCE",
        "failed_gates": ["synthetic malformed input"],
        "synthetic": "MALFORMED_INPUT",
    }
    output.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\\n",
        encoding="utf-8",
    )
    return 2
""",
        "invalid_status_exit_zero": """\
    payload = {"status": "INVALID_EVIDENCE", "synthetic": "MISMATCH"}
    output.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\\n",
        encoding="utf-8",
    )
    return 0
""",
        "success_status_exit_two": """\
    output.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\\n",
        encoding="utf-8",
    )
    return 2
""",
        "extra_output": """\
    output.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\\n",
        encoding="utf-8",
    )
    (output.parent / "unauthorized.tmp").write_text("extra\\n", encoding="utf-8")
    return 0
""",
        "pycache": """\
    output.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\\n",
        encoding="utf-8",
    )
    source_dir = Path(__file__).parent
    os.chmod(source_dir, 0o755)
    (source_dir / "__pycache__").mkdir()
    return 0
""",
    }
    return (common + bodies[behavior]).encode("utf-8")


@dataclass(frozen=True)
class SyntheticRepository:
    root: Path
    evidence_sha: str
    attestation_sha: str
    payload_sha256: str


def _build_synthetic_repository(
    tmp_path: Path,
    *,
    behavior: str = "success",
    extra_attestation_path: bool = False,
    source_hash_tamper: bool = False,
    runtime_version_tamper: bool = False,
) -> SyntheticRepository:
    repo = tmp_path / "synthetic-repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.name", "Synthetic Evidence Test")
    _git(repo, "config", "user.email", "synthetic@example.invalid")

    launcher_raw = (PLATFORM / "fig171819_evidence_launcher.py").read_bytes()
    attribution_raw = _stub_attribution(behavior)
    closure = [
        _source_record(bootstrap.ATTRIBUTION_PATH, attribution_raw),
        _source_record(launcher.LAUNCHER_PATH, launcher_raw),
    ]
    if source_hash_tamper:
        closure[0]["sha256"] = "f" * 64
    python_identity = _fixed_python_identity()
    if runtime_version_tamper:
        python_identity["version"] = "0.0.synthetic-tamper"
    authorization = {
        "schema_version": 1,
        "artifact_type": ("fig171819_active_disease_execution_authorization"),
        "launcher_contract": launcher.LAUNCHER_CONTRACT,
        "runtime_source_closure": closure,
        "runtime_environment_manifest": {
            "python_executable_realpath": launcher.PYTHON_EXECUTABLE_REALPATH,
            "python_implementation": python_identity["implementation"],
            "python_version": python_identity["version"],
            "python_flags": ["-I", "-S", "-B"],
            "stdlib_roots": [python_identity["stdlib"]],
            "site_packages_roots": [python_identity["purelib"]],
            "distributions": [],
        },
        "output_relative_path": "output",
    }
    authorization_raw = _json_bytes(authorization)
    paths = {
        launcher.LAUNCHER_PATH: launcher_raw,
        bootstrap.ATTRIBUTION_PATH: attribution_raw,
        launcher.AUTHORIZATION_PATH: authorization_raw,
    }
    for relative, raw in paths.items():
        destination = repo.joinpath(*Path(relative).parts)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(raw)
    _git(repo, "add", *sorted(paths))
    _git(repo, "commit", "-q", "-m", "synthetic evidence")
    evidence_sha = _git(repo, "rev-parse", "HEAD").decode().strip()

    payload = {
        "schema_version": 1,
        "artifact_type": "fig171819_active_disease_external_attestation",
        "evidence_commit_sha": evidence_sha,
        "authorization_path": launcher.AUTHORIZATION_PATH,
        "authorization_blob_sha256": launcher.sha256_bytes(authorization_raw),
        "launcher_path": launcher.LAUNCHER_PATH,
        "launcher_blob_sha256": launcher.sha256_bytes(launcher_raw),
        "attestation_payload_path": launcher.ATTESTATION_PAYLOAD_PATH,
        "fresh_run_id": "20260729_135128",
        "fresh_status_when_attested": "running",
        "launcher_contract": launcher.LAUNCHER_CONTRACT,
    }
    payload_raw = _json_bytes(payload)
    payload_path = repo.joinpath(*Path(launcher.ATTESTATION_PAYLOAD_PATH).parts)
    payload_path.write_bytes(payload_raw)
    additions = [launcher.ATTESTATION_PAYLOAD_PATH]
    if extra_attestation_path:
        extra_path = repo / "unexpected-attestation-member.txt"
        extra_path.write_text("extra\n", encoding="utf-8")
        additions.append("unexpected-attestation-member.txt")
    _git(repo, "add", *additions)
    _git(repo, "commit", "-q", "-m", "synthetic attestation")
    attestation_sha = _git(repo, "rev-parse", "HEAD").decode().strip()

    (repo / launcher.EXECUTION_ROOT_RELATIVE).mkdir(parents=True)
    return SyntheticRepository(
        root=repo,
        evidence_sha=evidence_sha,
        attestation_sha=attestation_sha,
        payload_sha256=launcher.sha256_bytes(payload_raw),
    )


def _input_record(path: Path) -> dict[str, object]:
    raw = path.read_bytes()
    return {
        "absolute_path": str(path),
        "sha256": launcher.sha256_bytes(raw),
        "size_bytes": len(raw),
    }


def _select_command(tmp_path: Path) -> Path:
    fingerprint = tmp_path / "synthetic-fingerprint.json"
    baseline = tmp_path / "synthetic-baseline.json"
    fingerprint.write_text('{"fixture":"fingerprint"}\n', encoding="utf-8")
    baseline.write_text('{"fixture":"baseline"}\n', encoding="utf-8")
    command = {
        "schema_version": 1,
        "adapter": launcher.ATTRIBUTION_ADAPTER,
        "command": "select-disease",
        "inputs": {
            "fingerprint": _input_record(fingerprint),
            "baseline_receipt": _input_record(baseline),
        },
        "output_relative_path": "synthetic-selection.json",
    }
    path = tmp_path / "synthetic-command.json"
    path.write_bytes(_json_bytes(command))
    return path


def _prepare_command(tmp_path: Path, disease_spec: Path) -> Path:
    fingerprint = tmp_path / "synthetic-prepare-fingerprint.json"
    fingerprint.write_text('{"fixture":"fingerprint"}\n', encoding="utf-8")
    command = {
        "schema_version": 1,
        "adapter": launcher.ATTRIBUTION_ADAPTER,
        "command": "prepare",
        "inputs": {
            "fingerprint": _input_record(fingerprint),
            "disease_spec": _input_record(disease_spec),
        },
        "output_relative_path": "synthetic-prepare.json",
    }
    path = tmp_path / "synthetic-prepare-command.json"
    path.write_bytes(_json_bytes(command))
    return path


def _run_dry(repository: SyntheticRepository):
    return bootstrap.run_outer_bootstrap(
        repo_root=repository.root,
        evidence_commit_sha=repository.evidence_sha,
        attestation_commit_sha=repository.attestation_sha,
        attestation_payload_sha256=repository.payload_sha256,
        dry_run=True,
        command_envelope_path=None,
        allow_synthetic_temp_repo=True,
    )


def _execution_output(
    repository: SyntheticRepository,
    execution_label: str = "select-disease",
) -> tuple[Path, Path]:
    root = (
        repository.root
        / launcher.EXECUTION_ROOT_RELATIVE
        / f"evidence-{repository.evidence_sha}-{execution_label}"
    )
    return root, root / "output"


def test_synthetic_dry_run_completes_raw_git_h0_h1_h2_final(tmp_path):
    repository = _build_synthetic_repository(tmp_path)
    return_code, summary = _run_dry(repository)
    assert return_code == 0
    assert summary["status"] == "PASS"
    execution_root, output_root = _execution_output(repository, "dry-run")
    assert summary["execution_root"] == str(execution_root)

    h0 = launcher.read_immutable_receipt(
        output_root / launcher.H0_RECEIPT_NAME,
        stage="outer_preflight",
    )
    h1 = launcher.read_immutable_receipt(
        output_root / launcher.H1_RECEIPT_NAME,
        stage="inner_launcher",
    )
    h2 = launcher.read_immutable_receipt(
        output_root / launcher.H2_RECEIPT_NAME,
        stage="outer_completion",
    )
    final = launcher.read_immutable_receipt(
        output_root / launcher.FINAL_RECEIPT_NAME,
        stage="final",
    )
    assert launcher.validate_receipt_chain(h0, h1, h2, final) == (
        h0.sha256,
        h1.sha256,
        h2.sha256,
    )
    h0_body = json.loads(h0.exact_bytes)["body"]
    for command in h0_body["bootstrap_commands"]:
        assert command["argv"][0] == "/usr/bin/git"
        assert "--no-replace-objects" in command["argv"]
        assert command["exit_code"] == 0
    assert json.loads(final.exact_bytes)["body"]["status"] == "DRY_RUN_PASS"


def test_dry_run_cli_accepts_only_literal_synthetic_identity(tmp_path, capsys):
    repository = _build_synthetic_repository(tmp_path)
    return_code = bootstrap.main(
        [
            "--repo-root",
            str(repository.root),
            "--evidence-commit",
            repository.evidence_sha,
            "--attestation-commit",
            repository.attestation_sha,
            "--attestation-payload-sha256",
            repository.payload_sha256,
            "--dry-run",
            "--synthetic-temporary-repo",
        ]
    )
    assert return_code == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["status"] == "PASS"
    assert summary["dry_run"] is True


def test_git_replace_ref_cannot_redirect_literal_attestation(tmp_path):
    repository = _build_synthetic_repository(tmp_path)
    # commit-tree needs a message on stdin; use a direct setup subprocess.
    completed = subprocess.run(
        (
            "/usr/bin/git",
            "-C",
            str(repository.root),
            "commit-tree",
            f"{repository.evidence_sha}^{{tree}}",
            "-p",
            repository.evidence_sha,
        ),
        input=b"replacement\n",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    replacement_sha = completed.stdout.decode().strip()
    _git(
        repository.root,
        "replace",
        repository.attestation_sha,
        replacement_sha,
    )
    return_code, summary = _run_dry(repository)
    assert return_code == 0
    assert summary["status"] == "PASS"


def test_literal_evidence_sha_must_equal_raw_attestation_parent(tmp_path):
    repository = _build_synthetic_repository(tmp_path)
    wrong_literal = "0" * 40 if repository.evidence_sha != "0" * 40 else "1" * 40
    with pytest.raises(launcher.EvidenceContractError, match="literal evidence"):
        bootstrap.run_outer_bootstrap(
            repo_root=repository.root,
            evidence_commit_sha=wrong_literal,
            attestation_commit_sha=repository.attestation_sha,
            attestation_payload_sha256=repository.payload_sha256,
            dry_run=True,
            command_envelope_path=None,
            allow_synthetic_temp_repo=True,
        )


@pytest.mark.parametrize(
    ("fixture_kwargs", "payload_override", "match"),
    [
        ({"extra_attestation_path": True}, None, "tree delta"),
        ({}, "0" * 64, "payload exact-byte hash"),
        ({"source_hash_tamper": True}, None, "closure exact bytes"),
    ],
)
def test_tree_payload_and_source_tamper_fail_closed(
    tmp_path, fixture_kwargs, payload_override, match
):
    repository = _build_synthetic_repository(tmp_path, **fixture_kwargs)
    with pytest.raises(launcher.EvidenceContractError, match=match):
        bootstrap.run_outer_bootstrap(
            repo_root=repository.root,
            evidence_commit_sha=repository.evidence_sha,
            attestation_commit_sha=repository.attestation_sha,
            attestation_payload_sha256=(payload_override or repository.payload_sha256),
            dry_run=True,
            command_envelope_path=None,
            allow_synthetic_temp_repo=True,
        )


def test_inner_preflight_failure_still_closes_h1_h2_final_dag(tmp_path):
    repository = _build_synthetic_repository(tmp_path, runtime_version_tamper=True)
    return_code, summary = _run_dry(repository)
    assert return_code == 2
    assert summary["status"] == "INVALID_EVIDENCE"
    _root, output_root = _execution_output(repository, "dry-run")
    envelopes = (
        launcher.read_immutable_receipt(
            output_root / launcher.H0_RECEIPT_NAME,
            stage="outer_preflight",
        ),
        launcher.read_immutable_receipt(
            output_root / launcher.H1_RECEIPT_NAME,
            stage="inner_launcher",
        ),
        launcher.read_immutable_receipt(
            output_root / launcher.H2_RECEIPT_NAME,
            stage="outer_completion",
        ),
        launcher.read_immutable_receipt(
            output_root / launcher.FINAL_RECEIPT_NAME,
            stage="final",
        ),
    )
    assert json.loads(envelopes[1].exact_bytes)["body"]["status"] == (
        "INVALID_EVIDENCE"
    )
    launcher.validate_receipt_chain(*envelopes)


@pytest.mark.parametrize(
    ("behavior", "expected_source", "expected_output"),
    [
        ("failure", "PASS", "FAIL"),
        ("extra_output", "PASS", "FAIL"),
        ("pycache", "FAIL", "PASS"),
    ],
)
def test_inner_failure_extra_output_and_pycache_are_invalid(
    tmp_path, behavior, expected_source, expected_output
):
    repository = _build_synthetic_repository(tmp_path, behavior=behavior)
    command_path = _select_command(tmp_path)
    return_code, summary = bootstrap.run_outer_bootstrap(
        repo_root=repository.root,
        evidence_commit_sha=repository.evidence_sha,
        attestation_commit_sha=repository.attestation_sha,
        attestation_payload_sha256=repository.payload_sha256,
        dry_run=False,
        command_envelope_path=command_path,
        allow_synthetic_temp_repo=True,
    )
    assert return_code == 2
    assert summary["status"] == "INVALID_EVIDENCE"
    _execution_root, output_root = _execution_output(repository)
    h2 = launcher.read_immutable_receipt(
        output_root / launcher.H2_RECEIPT_NAME,
        stage="outer_completion",
    )
    body = json.loads(h2.exact_bytes)["body"]
    assert body["status"] == "INVALID_EVIDENCE"
    assert body["post_run_source_closure"] == expected_source
    assert body["post_run_output_inventory"] == expected_output


def test_synthetic_adapter_success_forwards_only_authorized_paths(tmp_path):
    repository = _build_synthetic_repository(tmp_path)
    command_path = _select_command(tmp_path)
    return_code, summary = bootstrap.run_outer_bootstrap(
        repo_root=repository.root,
        evidence_commit_sha=repository.evidence_sha,
        attestation_commit_sha=repository.attestation_sha,
        attestation_payload_sha256=repository.payload_sha256,
        dry_run=False,
        command_envelope_path=command_path,
        allow_synthetic_temp_repo=True,
    )
    assert return_code == 0
    assert summary["status"] == "PASS"
    _root, output_root = _execution_output(repository)
    wrapped = launcher.parse_strict_json_object(
        (output_root / "synthetic-selection.json").read_bytes(),
        label="synthetic wrapped selection",
    )
    launcher.validate_wrapped_scientific_payload(wrapped, stage="select-disease")
    assert wrapped["payload"] == {
        "status": "ACTIVE_DISEASE_FROZEN",
        "synthetic": "PASS",
    }
    h2 = launcher.read_immutable_receipt(
        output_root / launcher.H2_RECEIPT_NAME,
        stage="outer_completion",
    )
    assert wrapped["execution_envelope"]["outer_completion_receipt_sha256"] == h2.sha256
    assert wrapped["execution_envelope"]["cleanup_status"] == "PASS"
    assert wrapped["execution_envelope"]["cleanup_target_relative_paths"] == [
        launcher.ATTRIBUTION_INTERMEDIATE_OUTPUTS["select-disease"]
    ]
    final = launcher.read_immutable_receipt(
        output_root / launcher.FINAL_RECEIPT_NAME,
        stage="final",
    )
    assert json.loads(final.exact_bytes)["body"][
        "scientific_payload_envelope_sha256"
    ] == launcher.sha256_bytes((output_root / "synthetic-selection.json").read_bytes())
    assert not (
        output_root / launcher.ATTRIBUTION_INTERMEDIATE_OUTPUTS["select-disease"]
    ).exists()
    assert launcher.verify_output_inventory(
        output_root,
        sorted(
            [
                launcher.H0_RECEIPT_NAME,
                launcher.H1_RECEIPT_NAME,
                launcher.H2_RECEIPT_NAME,
                launcher.FINAL_RECEIPT_NAME,
                "synthetic-selection.json",
            ]
        ),
    )
    fake_digest = copy.deepcopy(wrapped["execution_envelope"])
    fake_digest["outer_preflight_receipt_sha256"] = "0" * 64
    with pytest.raises(launcher.EvidenceContractError, match="canonical SHA256"):
        launcher.validate_scientific_execution_envelope(
            fake_digest,
            stage="select-disease",
        )
    semantic_forgery = copy.deepcopy(wrapped["execution_envelope"])
    semantic_forgery["outer_completion_receipt"]["body"]["cleanup_scope"] = (
        "ALL_OUTPUTS"
    )
    semantic_h2_raw = launcher.canonical_json_bytes(
        semantic_forgery["outer_completion_receipt"]
    )
    semantic_forgery["outer_completion_receipt_sha256"] = launcher.sha256_bytes(
        semantic_h2_raw
    )
    with pytest.raises(
        launcher.EvidenceContractError,
        match="transport/cleanup",
    ):
        launcher.validate_scientific_execution_envelope(
            semantic_forgery,
            stage="select-disease",
        )
    cleanup_target_forgery = copy.deepcopy(wrapped["execution_envelope"])
    cleanup_target_forgery["outer_completion_receipt"]["body"][
        "cleanup_target_relative_paths"
    ] = [launcher.ATTRIBUTION_INTERMEDIATE_OUTPUTS["prepare"]]
    cleanup_target_h2_raw = launcher.canonical_json_bytes(
        cleanup_target_forgery["outer_completion_receipt"]
    )
    cleanup_target_forgery["outer_completion_receipt_sha256"] = launcher.sha256_bytes(
        cleanup_target_h2_raw
    )
    with pytest.raises(
        launcher.EvidenceContractError,
        match="transport/cleanup",
    ):
        launcher.validate_scientific_execution_envelope(
            cleanup_target_forgery,
            stage="select-disease",
        )


def test_synthetic_malformed_input_is_wrapped_as_authorized_scientific_invalid(
    tmp_path,
):
    repository = _build_synthetic_repository(
        tmp_path,
        behavior="scientific_invalid",
    )
    command_path = _select_command(tmp_path)
    return_code, summary = bootstrap.run_outer_bootstrap(
        repo_root=repository.root,
        evidence_commit_sha=repository.evidence_sha,
        attestation_commit_sha=repository.attestation_sha,
        attestation_payload_sha256=repository.payload_sha256,
        dry_run=False,
        command_envelope_path=command_path,
        allow_synthetic_temp_repo=True,
    )
    assert return_code == 2
    assert summary["status"] == "INVALID_EVIDENCE"
    assert summary["transport_status"] == "PASS"
    assert summary["scientific_status"] == "INVALID_EVIDENCE"
    _root, output_root = _execution_output(repository)
    wrapped_path = output_root / "synthetic-selection.json"
    wrapped = launcher.parse_strict_json_object(
        wrapped_path.read_bytes(),
        label="synthetic wrapped invalid selection",
    )
    launcher.validate_wrapped_scientific_payload(
        wrapped,
        stage="select-disease",
    )
    assert wrapped["production_execution_authorized"] is True
    assert wrapped["payload"] == {
        "failed_gates": ["synthetic malformed input"],
        "status": "INVALID_EVIDENCE",
        "synthetic": "MALFORMED_INPUT",
    }
    h0 = launcher.read_immutable_receipt(
        output_root / launcher.H0_RECEIPT_NAME,
        stage="outer_preflight",
    )
    h1 = launcher.read_immutable_receipt(
        output_root / launcher.H1_RECEIPT_NAME,
        stage="inner_launcher",
    )
    h2 = launcher.read_immutable_receipt(
        output_root / launcher.H2_RECEIPT_NAME,
        stage="outer_completion",
    )
    final = launcher.read_immutable_receipt(
        output_root / launcher.FINAL_RECEIPT_NAME,
        stage="final",
    )
    launcher.validate_receipt_chain(h0, h1, h2, final)
    h2_body = json.loads(h2.exact_bytes)["body"]
    assert h2_body["status"] == "PASS"
    assert h2_body["transport_status"] == "PASS"
    assert h2_body["scientific_status"] == "INVALID_EVIDENCE"
    assert h2_body["inner_exit_code"] == 2
    assert h2_body["cleanup_status"] == "PASS"
    assert h2_body["cleanup_scope"] == "INTERMEDIATE_PAYLOAD_ONLY"
    assert h2_body["cleanup_target_relative_paths"] == [
        launcher.ATTRIBUTION_INTERMEDIATE_OUTPUTS["select-disease"]
    ]
    assert h2_body["execution_layout_cleanup_status"] == "NOT_RUN_RETAINED_FOR_AUDIT"
    assert h2_body["intermediate_payload_canonical_sha256"] == wrapped["payload_sha256"]
    final_body = json.loads(final.exact_bytes)["body"]
    assert final_body["status"] == "INVALID_EVIDENCE"
    assert final_body["transport_status"] == "PASS"
    assert final_body["scientific_status"] == "INVALID_EVIDENCE"
    assert not (
        output_root / launcher.ATTRIBUTION_INTERMEDIATE_OUTPUTS["select-disease"]
    ).exists()


@pytest.mark.parametrize(
    "behavior",
    ["invalid_status_exit_zero", "success_status_exit_two"],
)
def test_inner_exit_and_scientific_status_mismatch_fails_transport(
    tmp_path,
    behavior,
):
    repository = _build_synthetic_repository(tmp_path, behavior=behavior)
    command_path = _select_command(tmp_path)
    return_code, summary = bootstrap.run_outer_bootstrap(
        repo_root=repository.root,
        evidence_commit_sha=repository.evidence_sha,
        attestation_commit_sha=repository.attestation_sha,
        attestation_payload_sha256=repository.payload_sha256,
        dry_run=False,
        command_envelope_path=command_path,
        allow_synthetic_temp_repo=True,
    )
    assert return_code == 2
    assert summary["status"] == "INVALID_EVIDENCE"
    assert summary["transport_status"] == "INVALID_EVIDENCE"
    assert summary["scientific_output"] is None
    _root, output_root = _execution_output(repository)
    h2 = launcher.read_immutable_receipt(
        output_root / launcher.H2_RECEIPT_NAME,
        stage="outer_completion",
    )
    h2_body = json.loads(h2.exact_bytes)["body"]
    assert h2_body["status"] == "INVALID_EVIDENCE"
    assert "inner exit code/scientific status mismatch" in h2_body["post_run_errors"]
    assert h2_body["intermediate_payload_cleanup_status"] == "PASS"
    assert not (
        output_root / launcher.ATTRIBUTION_INTERMEDIATE_OUTPUTS["select-disease"]
    ).exists()


def test_synthetic_select_then_contribution_blind_prepare_accepts_wrapper(
    tmp_path,
):
    repository = _build_synthetic_repository(tmp_path)
    select_command = _select_command(tmp_path)
    select_return, _select_summary = bootstrap.run_outer_bootstrap(
        repo_root=repository.root,
        evidence_commit_sha=repository.evidence_sha,
        attestation_commit_sha=repository.attestation_sha,
        attestation_payload_sha256=repository.payload_sha256,
        dry_run=False,
        command_envelope_path=select_command,
        allow_synthetic_temp_repo=True,
    )
    assert select_return == 0
    _select_root, select_output_root = _execution_output(repository, "select-disease")
    disease_spec = select_output_root / "synthetic-selection.json"

    prepare_command = _prepare_command(tmp_path, disease_spec)
    prepare_return, prepare_summary = bootstrap.run_outer_bootstrap(
        repo_root=repository.root,
        evidence_commit_sha=repository.evidence_sha,
        attestation_commit_sha=repository.attestation_sha,
        attestation_payload_sha256=repository.payload_sha256,
        dry_run=False,
        command_envelope_path=prepare_command,
        allow_synthetic_temp_repo=True,
    )
    assert prepare_return == 0
    assert prepare_summary["status"] == "PASS"
    _prepare_root, prepare_output_root = _execution_output(repository, "prepare")
    wrapped_prepare = launcher.parse_strict_json_object(
        (prepare_output_root / "synthetic-prepare.json").read_bytes(),
        label="synthetic wrapped prepare",
    )
    launcher.validate_wrapped_scientific_payload(wrapped_prepare, stage="prepare")
    assert wrapped_prepare["payload"] == {
        "saw_baseline_receipt": False,
        "status": "PREPARED",
        "synthetic": "PREPARED",
    }
    assert not (
        prepare_output_root / launcher.ATTRIBUTION_INTERMEDIATE_OUTPUTS["prepare"]
    ).exists()


def test_execution_envelope_cannot_be_supplied_before_h2(tmp_path):
    repository = _build_synthetic_repository(tmp_path)
    command_path = _select_command(tmp_path)
    command = json.loads(command_path.read_bytes())
    command["execution_envelope"] = {"outer_completion_receipt_sha256": "0" * 64}
    command_path.write_bytes(_json_bytes(command))
    with pytest.raises(launcher.EvidenceContractError, match="keys differ"):
        bootstrap.run_outer_bootstrap(
            repo_root=repository.root,
            evidence_commit_sha=repository.evidence_sha,
            attestation_commit_sha=repository.attestation_sha,
            attestation_payload_sha256=repository.payload_sha256,
            dry_run=False,
            command_envelope_path=command_path,
            allow_synthetic_temp_repo=True,
        )


def test_command_input_hash_and_output_escape_fail_before_inner(tmp_path):
    repository = _build_synthetic_repository(tmp_path)
    command_path = _select_command(tmp_path)
    command = json.loads(command_path.read_bytes())
    command["inputs"]["fingerprint"]["sha256"] = "0" * 64
    command_path.write_bytes(_json_bytes(command))
    with pytest.raises(launcher.EvidenceContractError, match="input exact bytes"):
        bootstrap.run_outer_bootstrap(
            repo_root=repository.root,
            evidence_commit_sha=repository.evidence_sha,
            attestation_commit_sha=repository.attestation_sha,
            attestation_payload_sha256=repository.payload_sha256,
            dry_run=False,
            command_envelope_path=command_path,
            allow_synthetic_temp_repo=True,
        )

    command["inputs"]["fingerprint"] = _input_record(
        tmp_path / "synthetic-fingerprint.json"
    )
    command["output_relative_path"] = "../escape.json"
    command_path.write_bytes(_json_bytes(command))
    with pytest.raises(launcher.EvidenceContractError):
        bootstrap.run_outer_bootstrap(
            repo_root=repository.root,
            evidence_commit_sha=repository.evidence_sha,
            attestation_commit_sha=repository.attestation_sha,
            attestation_payload_sha256=repository.payload_sha256,
            dry_run=False,
            command_envelope_path=command_path,
            allow_synthetic_temp_repo=True,
        )


@pytest.mark.parametrize(
    ("command_name", "input_names"),
    [
        (
            "prepare",
            ("fingerprint", "disease_spec"),
        ),
        (
            "evaluate",
            (
                "prereg",
                "baseline_receipt",
                "result",
                "manifest",
                "contributions",
                "scorecard",
                "fingerprint",
            ),
        ),
    ],
)
def test_prepare_evaluate_adapter_vectors_are_frozen(
    tmp_path, command_name, input_names
):
    raw_inputs = {}
    for name in input_names:
        path = tmp_path / f"{name}.json"
        path.write_text(f'{{"fixture":"{name}"}}\n', encoding="utf-8")
        raw_inputs[name] = _input_record(path)
    command = launcher.parse_attribution_command_envelope(
        {
            "schema_version": 1,
            "adapter": launcher.ATTRIBUTION_ADAPTER,
            "command": command_name,
            "inputs": raw_inputs,
            "output_relative_path": f"{command_name}.json",
        }
    )
    output_root = tmp_path / "output"
    output_root.mkdir()
    argv = launcher.attribution_adapter_argv(command, output_root)
    assert argv[0] == command_name
    assert argv[-2:] == (
        "--output",
        str(output_root / launcher.ATTRIBUTION_INTERMEDIATE_OUTPUTS[command_name]),
    )
    if command_name == "prepare":
        assert "--contributions" not in argv
        assert "--baseline-receipt" not in argv
    assert "--execution-envelope" not in argv


def test_cleanup_refuses_unproven_preconditions_and_extra_output(tmp_path):
    repository = _build_synthetic_repository(tmp_path)
    return_code, _summary = _run_dry(repository)
    assert return_code == 0
    execution_root, output_root = _execution_output(repository, "dry-run")
    root_metadata = os.lstat(execution_root)
    layout = launcher.ExecutionLayout(
        root=execution_root,
        source_root=execution_root / "source",
        output_root=output_root,
        root_realpath=os.path.realpath(execution_root),
        root_device=int(root_metadata.st_dev),
        root_inode=int(root_metadata.st_ino),
    )
    h0 = json.loads((output_root / launcher.H0_RECEIPT_NAME).read_bytes())
    entries = launcher.runtime_source_closure_from_json(
        h0["body"]["runtime_source_closure"]
    )
    expected = sorted(
        [
            launcher.H0_RECEIPT_NAME,
            launcher.H1_RECEIPT_NAME,
            launcher.H2_RECEIPT_NAME,
            launcher.FINAL_RECEIPT_NAME,
        ]
    )
    with pytest.raises(launcher.EvidenceContractError, match="preconditions"):
        launcher.cleanup_execution_layout_exact(
            layout,
            entries,
            expected,
            file_handles_closed=False,
            no_child_processes=True,
        )
    (output_root / "extra.tmp").write_text("extra\n", encoding="utf-8")
    with pytest.raises(launcher.EvidenceContractError, match="path set"):
        launcher.cleanup_execution_layout_exact(
            layout,
            entries,
            expected,
            file_handles_closed=True,
            no_child_processes=True,
        )
    assert execution_root.exists()


def test_receipts_and_sources_are_read_only_and_no_pyc_is_created(tmp_path):
    repository = _build_synthetic_repository(tmp_path)
    return_code, _summary = _run_dry(repository)
    assert return_code == 0
    execution_root, output_root = _execution_output(repository, "dry-run")
    for receipt_name in (
        launcher.H0_RECEIPT_NAME,
        launcher.H1_RECEIPT_NAME,
        launcher.H2_RECEIPT_NAME,
        launcher.FINAL_RECEIPT_NAME,
    ):
        assert not (os.lstat(output_root / receipt_name).st_mode & stat.S_IWUSR)
    source_files = list((execution_root / "source").rglob("*"))
    assert not any(
        path.name == "__pycache__" or path.suffix == ".pyc" for path in source_files
    )
