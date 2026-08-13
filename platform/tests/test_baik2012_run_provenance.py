"""Cheap tests for Baik runner provenance and output safety helpers."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from forward_flight_benchmarks.run_provenance import (
    baik_transfer_dependency_paths,
    collect_git_state,
    collect_run_provenance,
    collect_source_hashes,
    prepare_output_directory,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_baik_dependency_closure_includes_ldvm_and_uvpm_particle_sources() -> None:
    relative = {
        str(path.relative_to(REPO_ROOT))
        for path in baik_transfer_dependency_paths(REPO_ROOT)
    }
    assert "platform/flap_ldvm.py" in relative
    assert "platform/ldvm_fourier.py" in relative
    assert "src/fluxvortex/solver.py" in relative
    assert "src/fluxvortex/particles.py" in relative
    assert "src/fluxvortex/kernel.py" in relative
    assert "pyproject.toml" in relative
    assert all((REPO_ROOT / path).is_file() for path in relative)


def test_collect_source_hashes_uses_repository_relative_keys(tmp_path: Path) -> None:
    source = tmp_path / "source.txt"
    source.write_text("Baik\n", encoding="utf-8")
    hashes = collect_source_hashes(tmp_path, (source,))
    assert hashes == {"source.txt": hashlib.sha256(b"Baik\n").hexdigest()}
    with pytest.raises(ValueError, match="outside repository"):
        collect_source_hashes(tmp_path, (REPO_ROOT / "pyproject.toml",))


def test_prepare_output_directory_refuses_nonempty_without_opt_in(
    tmp_path: Path,
) -> None:
    output = tmp_path / "result"
    prepared = prepare_output_directory(output)
    assert prepared == output.resolve()
    (output / "keep.txt").write_text("do not overwrite", encoding="utf-8")
    with pytest.raises(FileExistsError, match="--allow-existing-output"):
        prepare_output_directory(output)
    assert prepare_output_directory(output, allow_existing=True) == output.resolve()
    assert (output / "keep.txt").read_text(encoding="utf-8") == "do not overwrite"


def test_collect_run_provenance_records_args_packages_and_git() -> None:
    provenance = collect_run_provenance(
        REPO_ROOT,
        {"quality": "smoke", "output_dir": Path("relative/result")},
        argv=("runner", "--quality", "smoke"),
    )
    assert provenance["invocation"]["argv"] == [
        "runner",
        "--quality",
        "smoke",
    ]
    assert provenance["invocation"]["parsed_arguments"]["output_dir"] == (
        "relative/result"
    )
    assert provenance["environment"]["package_versions"]["numpy"] is not None
    git_state = provenance["git"]
    assert git_state["available"] is True
    assert len(git_state["revision"]) == 40
    assert git_state["dirty"] is True
    assert git_state["status_entry_count"] == len(git_state["status_porcelain"])
    assert len(git_state["status_porcelain_sha256"]) == 64


def test_collect_git_state_handles_non_repository(tmp_path: Path) -> None:
    state = collect_git_state(tmp_path)
    assert state["available"] is False
    assert "error" in state
