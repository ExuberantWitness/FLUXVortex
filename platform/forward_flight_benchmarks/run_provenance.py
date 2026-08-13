"""Small, dependency-free provenance helpers for benchmark runners.

The helpers deliberately avoid importing the numerical model.  They can be
unit-tested cheaply, and they collect repository state before a runner creates
new result files inside the worktree.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import platform
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


PROVENANCE_PACKAGES = (
    "fluxvortex",
    "matplotlib",
    "numba",
    "numpy",
    "pterasoftware",
    "scipy",
)


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of one file without loading it all at once."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def collect_source_hashes(repo_root: Path, paths: Iterable[Path]) -> dict[str, str]:
    """Hash declared direct sources and fail closed on missing/outside paths."""

    root = repo_root.resolve()
    hashes: dict[str, str] = {}
    for source in paths:
        resolved = source.resolve()
        try:
            relative = resolved.relative_to(root)
        except ValueError as error:
            raise ValueError(f"source is outside repository: {resolved}") from error
        if not resolved.is_file():
            raise FileNotFoundError(f"declared source does not exist: {resolved}")
        hashes[str(relative)] = sha256_file(resolved)
    return hashes


def baik_transfer_dependency_paths(repo_root: Path) -> tuple[Path, ...]:
    """Return code paths that directly determine the Baik FluxV/v4b histories."""

    root = repo_root.resolve()
    benchmark = root / "platform/forward_flight_benchmarks"
    return (
        benchmark / "baik2012.py",
        benchmark / "ldvm_uvlm_correction.py",
        benchmark / "causal_incidence_owner.py",
        benchmark / "uvlm_polar_correction.py",
        benchmark / "ptera_adapter.py",
        benchmark / "cases.py",
        root / "platform/ldvm_fourier.py",
        root / "platform/flap_ldvm.py",
        root / "src/fluxvortex/solver.py",
        root / "src/fluxvortex/particles.py",
        root / "src/fluxvortex/kernel.py",
        root / "pyproject.toml",
    )


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def collect_package_versions(
    package_names: Sequence[str] = PROVENANCE_PACKAGES,
) -> dict[str, str | None]:
    """Collect installed distribution versions without importing packages."""

    versions: dict[str, str | None] = {}
    for name in package_names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _git(repo_root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ("git", *arguments),
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.rstrip("\n")


def collect_git_state(repo_root: Path) -> dict[str, Any]:
    """Collect revision, branch and an exact hash/list of pre-run status lines."""

    try:
        revision = _git(repo_root, "rev-parse", "HEAD")
        branch = _git(repo_root, "branch", "--show-current") or None
        status_text = _git(
            repo_root, "status", "--porcelain=v1", "--untracked-files=all"
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        return {
            "available": False,
            "error": f"{type(error).__name__}: {error}",
        }
    status_lines = status_text.splitlines() if status_text else []
    return {
        "available": True,
        "revision": revision,
        "branch": branch,
        "dirty": bool(status_lines),
        "status_entry_count": len(status_lines),
        "status_porcelain_sha256": hashlib.sha256(
            status_text.encode("utf-8")
        ).hexdigest(),
        "status_porcelain": status_lines,
    }


def collect_run_provenance(
    repo_root: Path,
    parsed_arguments: Mapping[str, Any],
    *,
    argv: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Collect invocation/environment/repository provenance before a run."""

    effective_argv = list(sys.argv if argv is None else argv)
    return {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "invocation": {
            "executable": sys.executable,
            "argv": effective_argv,
            "parsed_arguments": _jsonable(dict(parsed_arguments)),
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "package_versions": collect_package_versions(),
        },
        "git": collect_git_state(repo_root),
    }


def prepare_output_directory(
    output: Path,
    *,
    allow_existing: bool = False,
) -> Path:
    """Create an output directory, refusing a non-empty target by default.

    ``allow_existing`` is intentionally explicit.  It permits writers to
    replace files with the same names, but this helper never deletes unrelated
    files from the target.
    """

    resolved = output.resolve()
    if resolved == Path(resolved.anchor):
        raise ValueError("refusing to use a filesystem root as output directory")
    if resolved.exists() and not resolved.is_dir():
        raise NotADirectoryError(f"output path is not a directory: {resolved}")
    if resolved.exists() and any(resolved.iterdir()) and not allow_existing:
        raise FileExistsError(
            f"output directory is non-empty: {resolved}; choose a new --output-dir "
            "or pass --allow-existing-output explicitly"
        )
    resolved.mkdir(parents=True, exist_ok=True)
    probe = resolved / f".fluxv-write-probe-{uuid.uuid4().hex}"
    try:
        probe.touch(exist_ok=False)
        probe.unlink()
    except OSError as error:
        raise PermissionError(
            f"output directory is not writable: {resolved}"
        ) from error
    return resolved


__all__ = [
    "baik_transfer_dependency_paths",
    "collect_git_state",
    "collect_package_versions",
    "collect_run_provenance",
    "collect_source_hashes",
    "prepare_output_directory",
    "sha256_file",
]
