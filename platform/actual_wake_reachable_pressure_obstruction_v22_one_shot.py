"""Externally authorized one-shot transport for the frozen S3ai-v2.2 run.

This module adds no aerodynamic equation and does not make the frozen public
runner executable.  It only transports one independently authorized call to
the already frozen loader, 31-history collector, and aggregator.  Importing
the module never loads a contract, constructs a mesh, collects a history, or
writes an artifact.

The permanent attempt marker is claimed before the first history.  Therefore
an integrity failure or crash after that point consumes the authorization and
cannot be silently retried.  A scientifically adverse ``PROTOCOL-NO-GO``
result, on the other hand, is a valid observation and is published.
"""
from __future__ import annotations

from contextlib import redirect_stdout
import argparse
import ctypes
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import platform as host_platform
import secrets
import stat
import sys
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
AUTHORIZATION_PATH = (
    PLATFORM
    / "docs"
    / "diag"
    / "actual_wake_reachable_pressure_execution_authorization_20260728_143034.yaml"
)
RESULT_PATH = (
    PLATFORM
    / "docs"
    / "diag"
    / "actual_wake_reachable_pressure_obstruction_results_20260728_134229.json"
)
ATTEMPT_MARKER_PATH = Path(f"{RESULT_PATH}.lock")
LATEST_RESULT_PATH = (
    PLATFORM
    / "docs"
    / "diag"
    / "actual_wake_reachable_pressure_obstruction_results.json"
)
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
_O_DIRECTORY = getattr(os, "O_DIRECTORY", 0)
_O_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_IMPORT_TIME_PYTHON_DEPENDENCY_FINGERPRINTS: (
    dict[str, dict[str, Any]] | None
) = None


class OneShotAuthorizationError(RuntimeError):
    """The one-shot provenance, authorization, or result contract failed."""


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
    source_fingerprints: Mapping[str, str]
    runtime_identity: Mapping[str, Any]
    audit_files: Mapping[str, str]
    clearance: Mapping[str, Any]


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


def _runtime_identity() -> dict[str, Any]:
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
    python_dependency_fingerprints = (
        _loaded_python_dependency_fingerprints()
    )
    if (
        _IMPORT_TIME_PYTHON_DEPENDENCY_FINGERPRINTS is not None
        and python_dependency_fingerprints
        != _IMPORT_TIME_PYTHON_DEPENDENCY_FINGERPRINTS
    ):
        raise OneShotAuthorizationError(
            "loaded Python dependency source set or bytes changed after "
            "wrapper import"
        )
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
        "loaded_native_binary_fingerprints": (
            _loaded_native_binary_fingerprints()
        ),
        "loaded_python_dependency_fingerprints": (
            python_dependency_fingerprints
        ),
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


def _canonical_authorization_digest(payload: Mapping[str, Any]) -> str:
    normalized = _strict_json_value(payload, label="authorization")
    authorization = normalized.get("authorization")
    if not isinstance(authorization, dict):
        raise OneShotAuthorizationError(
            "authorization document lacks authorization mapping"
        )
    authorization["canonical_sha256"] = _ZERO_SHA256
    return _sha256_bytes(_canonical_json_bytes(normalized))


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


def _load_and_verify_authorization(
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
    if (
        ticket.get("ordered_case_names") != names
        or ticket.get("case_identity_sha256") != identities
        or ticket.get("ordered_registry_manifest_sha256") != manifest_sha
        or manifests.get("ordered_registry_manifest_sha256")
        != manifest_sha
        or manifests.get("frozen_definition_chain_sha256")
        != definition_chain_sha
        or verified.clearance.get("ordered_registry_manifest_sha256")
        != manifest_sha
        or verified.clearance.get("frozen_definition_chain_sha256")
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
    if (
        chain_sha != manifests.get("frozen_definition_chain_sha256")
        or chain_sha
        != verified.clearance.get("frozen_definition_chain_sha256")
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
    runtime_start: Mapping[str, Any],
    runtime_end: Mapping[str, Any],
    execution_input_sha256_start: str,
    execution_input_sha256_end: str,
    marker_receipt_sha256: str,
) -> dict[str, Any]:
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
    result["one_shot_provenance"] = {
        "wrapper_definition": {
            "path": _validated_repo_relative(
                WRAPPER_DEFINITION_PATH
            ).as_posix(),
            "sha256": verified.definition_sha256,
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
        },
        "second_bounded_audit": {
            "verdict": verified.clearance["verdict"],
            "actual_independence": verified.clearance[
                "actual_independence"
            ],
            "request_path": second_audit["request_path"],
            "request_sha256": second_audit["request_sha256"],
            "response_path": second_audit["response_path"],
            "response_sha256": second_audit["response_sha256"],
            "clearance_canonical_sha256": second_audit[
                "clearance_canonical_sha256"
            ],
        },
        "ordered_registry_manifest_sha256": registry_manifest_sha256,
        "source_fingerprints_start": dict(source_fingerprints_start),
        "source_fingerprints_end": dict(source_fingerprints_end),
        "runtime_identity_start": dict(runtime_start),
        "runtime_identity_end": dict(runtime_end),
        "execution_input_sha256_start": execution_input_sha256_start,
        "execution_input_sha256_end": execution_input_sha256_end,
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
) -> None:
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
    sources_start = provenance.get("source_fingerprints_start")
    sources_end = provenance.get("source_fingerprints_end")
    runtime_start = provenance.get("runtime_identity_start")
    runtime_end = provenance.get("runtime_identity_end")
    execution_input_start = provenance.get(
        "execution_input_sha256_start"
    )
    execution_input_end = provenance.get("execution_input_sha256_end")
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
    if (
        not isinstance(sources_start, Mapping)
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
        or not isinstance(runtime_start, Mapping)
        or not isinstance(runtime_end, Mapping)
        or dict(runtime_start) != dict(runtime_end)
        or _validate_sha256(
            execution_input_start,
            label="serialized execution input start",
        )
        != execution_input_start
        or execution_input_start != execution_input_end
        or provenance.get("ordered_registry_manifest_sha256")
        != registry_manifest
        or not isinstance(authorization, Mapping)
        or not isinstance(authorization.get("id"), str)
        or not authorization.get("id")
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
            )
        )
        or not isinstance(bounded_audit, Mapping)
        or bounded_audit.get("verdict")
        != "ACCEPT_EXACT_ONE_SHOT_31_HISTORY_EXECUTION"
        or bounded_audit.get("actual_independence")
        != "genuine-cross-family"
        or any(
            _validate_sha256(
                bounded_audit.get(key),
                label=f"serialized bounded audit {key}",
            )
            != bounded_audit.get(key)
            for key in (
                "request_sha256",
                "response_sha256",
                "clearance_canonical_sha256",
            )
        )
        or not isinstance(marker, Mapping)
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
) -> str:
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
        ) != (
            int(temporary_metadata.st_dev),
            int(temporary_metadata.st_ino),
            len(payload),
        ):
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
        return _sha256_bytes(payload)
    except OSError as error:
        raise OneShotAuthorizationError(
            f"atomic no-replace result publication failed: {error}"
        ) from error
    finally:
        if fd >= 0:
            os.close(fd)
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
        if linked:
            try:
                published = os.stat(
                    name, dir_fd=parent_fd, follow_symlinks=False
                )
            except FileNotFoundError as error:
                raise OneShotAuthorizationError(
                    "published result disappeared before receipt construction"
                ) from error
            if not stat.S_ISREG(published.st_mode):
                raise OneShotAuthorizationError(
                    "published result is not a regular file"
                )
        if owns_parent:
            os.close(parent_fd)


def _execution_input_snapshot(
    *,
    verified: _VerifiedAuthorization,
    source_fingerprints: Mapping[str, str],
    runtime_identity: Mapping[str, Any],
    contract: Mapping[str, Any],
    registry_manifest_sha256: str,
) -> dict[str, Any]:
    ticket_raw = _sha256_bytes(
        _read_regular_nofollow(
            AUTHORIZATION_PATH, label="one-shot authorization ticket"
        )
    )
    definition_raw = _sha256_bytes(
        _read_regular_nofollow(
            WRAPPER_DEFINITION_PATH, label="wrapper preregistration"
        )
    )
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
        or frozen_files != expected_frozen
        or audit_files != dict(verified.audit_files)
    ):
        raise OneShotAuthorizationError(
            "an authorization, preregistration, frozen definition, or audit "
            "trace changed between verified use and execution snapshot"
        )
    return {
        "authorization_raw_sha256": ticket_raw,
        "authorization_canonical_sha256": verified.canonical_sha256,
        "wrapper_definition_sha256": definition_raw,
        "source_fingerprints": dict(source_fingerprints),
        "runtime_identity": dict(runtime_identity),
        "frozen_definition_files": frozen_files,
        "audit_files": audit_files,
        "ordered_registry_manifest_sha256": registry_manifest_sha256,
    }


def run_authorized_once(
    *,
    expected_authorization_sha256: str,
    second_audit_token: bytes,
) -> ExecutionReceipt:
    """Execute and publish the single externally authorized 31-history run.

    The raw authorization digest and bearer-token preimage are caller inputs;
    neither is inferred from the local ticket or an environment variable.
    """

    # These checks precede the frozen loader and every mesh/solver call.
    _require_absent(RESULT_PATH, label="formal result")
    _require_absent(
        ATTEMPT_MARKER_PATH, label="permanent attempt marker"
    )
    if _entry_lstat(LATEST_RESULT_PATH) is not None:
        # A historical pointer may exist in another campaign, but this
        # wrapper is preregistered to never mutate it.  Requiring absence also
        # prevents a misleading pointer from being mistaken for this result.
        raise OneShotAuthorizationError(
            "latest result pointer must remain absent for the one-shot run"
        )
    definition, definition_sha = _load_wrapper_definition()
    _require_source_execution_mode()
    _verify_loaded_local_modules_use_source(definition)
    sources_start = _source_fingerprints(definition)
    runtime_start = _runtime_identity()
    verified = _load_and_verify_authorization(
        expected_authorization_sha256=expected_authorization_sha256,
        second_audit_token=second_audit_token,
        definition=definition,
        definition_sha256=definition_sha,
        source_fingerprints=sources_start,
        runtime_identity=runtime_start,
    )

    # The original frozen loader must continue to observe false.  The external
    # ticket authorizes only this wrapper; it never mutates the contract.
    frozen_before_loader = _verify_ticket_frozen_files_before_loader(
        verified
    )
    contract = guard._load_frozen_contract()
    ordered_names, _, registry_manifest = _verify_contract_authorization(
        verified, contract
    )
    start_snapshot = _execution_input_snapshot(
        verified=verified,
        source_fingerprints=sources_start,
        runtime_identity=runtime_start,
        contract=contract,
        registry_manifest_sha256=registry_manifest,
    )
    if start_snapshot["frozen_definition_files"] != frozen_before_loader:
        raise OneShotAuthorizationError(
            "frozen definition bytes changed across the frozen loader"
        )
    start_snapshot_sha = _sha256_bytes(
        _canonical_json_bytes(start_snapshot)
    )

    # Recheck both names under fixed no-follow directory descriptors, then
    # durably consume the ticket before the first formal history.
    output_directory = _acquire_output_directory_lease(
        ATTEMPT_MARKER_PATH, RESULT_PATH
    )
    marker_receipt = {
        "artifact": "S3ai-v2.2-permanent-one-shot-attempt",
        "authorization_id": verified.payload["authorization"]["id"],
        "authorization_raw_sha256": verified.raw_sha256,
        "authorization_canonical_sha256": verified.canonical_sha256,
        "single_use_token_sha256": verified.token_sha256,
        "wrapper_definition_sha256": verified.definition_sha256,
        "wrapper_source_sha256": sources_start[_WRAPPER_RELATIVE],
        "execution_input_sha256_start": start_snapshot_sha,
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

        observations = guard._collect_frozen_histories(contract)
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

        sources_end = _source_fingerprints(definition)
        runtime_end = _runtime_identity()
        end_snapshot = _execution_input_snapshot(
            verified=verified,
            source_fingerprints=sources_end,
            runtime_identity=runtime_end,
            contract=contract,
            registry_manifest_sha256=registry_manifest,
        )
        end_snapshot_sha = _sha256_bytes(
            _canonical_json_bytes(end_snapshot)
        )
        if (
            sources_end != sources_start
            or runtime_end != runtime_start
            or end_snapshot != start_snapshot
            or end_snapshot_sha != start_snapshot_sha
        ):
            raise OneShotAuthorizationError(
                "execution source, runtime, or bound input drifted during "
                "the consumed formal attempt; no result was published"
            )
        _verify_permanent_attempt_marker(
            ATTEMPT_MARKER_PATH,
            attempt_claim.receipt_sha256,
            output_directory=output_directory,
            result_path=RESULT_PATH,
            expected_identity=attempt_claim,
        )

        augmented = _augment_result(
            result,
            observations,
            ordered_case_names=ordered_names,
            verified=verified,
            registry_manifest_sha256=registry_manifest,
            source_fingerprints_start=sources_start,
            source_fingerprints_end=sources_end,
            runtime_start=runtime_start,
            runtime_end=runtime_end,
            execution_input_sha256_start=start_snapshot_sha,
            execution_input_sha256_end=end_snapshot_sha,
            marker_receipt_sha256=attempt_claim.receipt_sha256,
        )
        serialized = json.loads(
            _pretty_json_bytes(augmented).decode("utf-8")
        )
        _validate_serialized_result(
            serialized, ordered_case_names=ordered_names
        )
        result_sha = _atomic_create_json_no_replace(
            RESULT_PATH,
            serialized,
            output_directory=output_directory,
            marker_path=ATTEMPT_MARKER_PATH,
            attempt_claim=attempt_claim,
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
    return parser


_CLI_PARSER = _build_cli_parser()


def main(argv: Sequence[str] | None = None) -> int:
    """CLI that reads the 32-byte bearer token as hex from standard input."""

    arguments = _CLI_PARSER.parse_args(argv)
    token_hex = sys.stdin.buffer.readline().strip()
    try:
        token = bytes.fromhex(token_hex.decode("ascii"))
    except (UnicodeDecodeError, ValueError) as error:
        raise OneShotAuthorizationError(
            "standard input must contain exactly one 64-hex-character token"
        ) from error
    receipt = run_authorized_once(
        expected_authorization_sha256=(
            arguments.expected_authorization_sha256
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


# Complete any documented lazy imports before freezing the external
# pure-Python dependency snapshot used by the formal runtime identity.
host_platform.platform()
_IMPORT_TIME_CONFIG_WARMUP = io.StringIO()
with redirect_stdout(_IMPORT_TIME_CONFIG_WARMUP):
    np.__config__.show()
_IMPORT_TIME_PYTHON_DEPENDENCY_FINGERPRINTS = (
    _loaded_python_dependency_fingerprints()
)


__all__ = [
    "ExecutionReceipt",
    "OneShotAuthorizationError",
    "main",
    "run_authorized_once",
]


if __name__ == "__main__":
    raise SystemExit(main())
