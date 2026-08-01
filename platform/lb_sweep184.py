"""Complete the frozen V4.1 Fig. 17/18/19 benchmark from 118 to 184 cases.

This runner is deliberately separate from ``lb_sweep118.py``:

* the frozen 118-point source is never overwritten;
* a timestamped checkpoint is seeded from those 118 points;
* only the 66 missing Fig. 18(c,d) U=6/10 twist conditions are run;
* a fresh-process cold call is excluded, then a warm formal anchor must
  reproduce the frozen source within 0.15 N before the missing cases run;
* fixed-name ``latest`` artifacts are published only after all 184 conditions
  are finite and the 50-curve benchmark contract is complete.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import inspect
import json
import math
import os
import platform as py_platform
import re
import shutil
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence


PLATFORM = Path(__file__).resolve().parent
ROOT = PLATFORM.parent
DOCS = PLATFORM / "docs"
DIAG = DOCS / "diag"
SCORECARDS = DOCS / "scorecards"
for path in (ROOT, PLATFORM):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from fig171819_benchmark import (  # noqa: E402
    CONDITIONS,
    CURVES,
    DEFAULT_REPRO,
    condition_key,
    coverage,
    scorecard,
)
from lb_sweep118 import BASE, CONDS as FROZEN_118_CONDITIONS, key_of, spc_of  # noqa: E402


_IMPORT_BOUND_SOURCE_HASHES = {
    str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
    for path in (
        Path(__file__).resolve(),
        (PLATFORM / "fig171819_benchmark.py").resolve(),
        (PLATFORM / "lb_sweep118.py").resolve(),
    )
}


FROZEN_EXPECTED_HASHES = {
    # 2026-08-01: v41 预设修正(E1/E2 闭环, lb_hybrid 1.0->0.0 + lb_cla3d=True),
    # 三点复现缓存最大差 0.100N < 0.15N 门。旧哈希 eb516d6a 为修正前版本。
    "platform/_v2_robo.py": "e13feca0187c6980113e91a604c7a160b27a9fae032fb3bddd498d247d94379f",
    "platform/_v2_repro_nc12.py": "880cacb1e7844341255e06d8e464274932aa9fcfa7fbd13679d6983d216548ba",
    "platform/lb_sweep118.py": "059add02d0d3d448c632956ead2a0a83a6307ea0673ae83c9a2c20937200f325",
    "platform/lb_dyn.py": "11b7e81acc7b8b43a4df74954d44653334156a16bc507485423aad6e6e8445b1",
    "platform/lb_static.py": "b8c161bf6775ccb837a6934b97c5b51f6dd33470c29cf9ebcb27ddcf6c547524",
    "platform/cd_table.py": "1d7a19cba0ce8aeb9fe118672be043ae8fca80cb0f22887431fdce62c4e20ac9",
    "platform/diff_uvlm_unsteady_gpu.py": "1262d54344c5c740375f5d33c8474e90f15fbbc38d5fcfa632d738d8a7d58ac0",
    "platform/diff_uvlm_unsteady.py": "83081be5882f710d6592aedd806e72f9f6a818d12bf638bacfcbed37f6c16b31",
    "platform/diff_coupled_unsteady.py": "0e2026ae93c1e48d29fd67bd96df53f279ea9ab865574e2c67bc861a406f6c22",
    "platform/flap_flight_validate.py": "2ae23eefc2c2616f8419e6d4576d45509181e142760fc1ad37159fb67a758efa",
    "platform/_v2_robogeom.py": "2de57d9062e61cdeafcf4bced2647917e50e7adcfa114f1be8c71eeef6b69e98",
    "platform/airfoil_geometry.py": "da4894e7e6415c3fb1db5c55e3c9e7d72fc6e203de0808e6c32bc203da9f9f3f",
    "src/fluxvortex/warp_fsi/config.py": "cefba419b9b8602b9b6b6a4f19f7cd953c97d47be56d259753ab5733d4e4093c",
    "src/fluxvortex/warp_fsi/batched_solver.py": "64e9b6a96b7941c54e9399c7de6bf3d9a300f893b605c094472f8db1afa2aaaa",
    "researchpaper/uiuc_polars/SD7003.DRG": "22a00c18fba12402f7f9f782ac33e77f935a457a0043607ca4068c307ccefb8a",
    "platform/docs/s6_sweep_v41.json": "965da388863dc57b390d58b49fe3b8978bdc77c3603b4a3276c97d4d17f94c73",
    "platform/docs/repro_data.json": "808ffeed36be0071850e954231417fa7007167c59eb5730cd8cb6829ff18101c",
    "platform/docs/data.md": "ca4274e0b5c4af4f8fa526f2403579ee5eda1a1cdceabfac5682286f8c3cf3a1",
}
SEED = DOCS / "s6_sweep_v41.json"
FIXED_RESULT = DOCS / "s6_sweep_v41_full184.json"
FIXED_MANIFEST = DIAG / "fig171819_v41_baseline_manifest.json"
FIXED_SCORECARD = SCORECARDS / "scorecard_v41_full184.json"
RUN_LOCK = DIAG / "fig171819_v41_baseline.lock"
ANCHOR = (8.0, 2.6, 0.0, 5.0)
ANCHOR_TOLERANCE_N = 0.15
MANIFEST_SCHEMA_VERSION = 4
GOVERNED_SOURCE_GLOBS = (
    "platform/claim_nodes/*.y*ml",
    "platform/claim_runtime/**/*.py",
)
AUTHORITATIVE_SOURCE_GLOBS = (
    "platform/*.py",
    "platform/claim_runtime/**/*.py",
    "platform/claim_nodes/*.y*ml",
    "src/fluxvortex/**/*.py",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise RuntimeError(f"{path}: expected a JSON object")
    return value


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, partial_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".partial", dir=path.parent
    )
    partial = Path(partial_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(partial, path)
        _fsync_directory(path.parent)
    finally:
        if partial.exists():
            partial.unlink()


def _copy_atomic(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, partial_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".partial", dir=destination.parent
    )
    partial = Path(partial_name)
    try:
        with os.fdopen(descriptor, "wb") as dst:
            with source.open("rb") as src:
                shutil.copyfileobj(src, dst)
                dst.flush()
                os.fsync(dst.fileno())
        os.replace(partial, destination)
        _fsync_directory(destination.parent)
    finally:
        if partial.exists():
            partial.unlink()


@contextmanager
def _exclusive_campaign_lock(path: Path = RUN_LOCK) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(
                f"another Fig17/18/19 V4.1 baseline process holds {path}"
            ) from exc
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _validate_frozen_identity() -> dict[str, str]:
    actual: dict[str, str] = {}
    for relative, expected in FROZEN_EXPECTED_HASHES.items():
        path = ROOT / relative
        digest = _sha256_file(path)
        actual[relative] = digest
        if digest != expected:
            raise RuntimeError(
                f"frozen V4.1 identity drift for {relative}: "
                f"{digest}, expected {expected}"
            )
    return actual


def _snapshot_static_run_sources() -> dict[str, str]:
    paths = {
        Path(__file__).resolve(),
        (PLATFORM / "fig171819_benchmark.py").resolve(),
        (PLATFORM / "lb_sweep118.py").resolve(),
    }
    paths.update((ROOT / relative).resolve() for relative in FROZEN_EXPECTED_HASHES)
    for members in _snapshot_governed_source_members().values():
        paths.update((ROOT / relative).resolve() for relative in members)
    return {
        str(path.relative_to(ROOT)): _sha256_file(path)
        for path in sorted(paths)
    }


def _snapshot_governed_source_members() -> dict[str, list[str]]:
    return {
        pattern: sorted(
            str(path.resolve().relative_to(ROOT))
            for path in ROOT.glob(pattern)
            if path.is_file()
        )
        for pattern in GOVERNED_SOURCE_GLOBS
    }


def _authoritative_source_paths() -> set[Path]:
    paths = {
        path.resolve()
        for pattern in AUTHORITATIVE_SOURCE_GLOBS
        for path in ROOT.glob(pattern)
        if path.is_file()
    }
    paths.update(
        (ROOT / relative).resolve()
        for relative in FROZEN_EXPECTED_HASHES
    )
    return paths


def _snapshot_authoritative_sources() -> dict[str, str]:
    return {
        str(path.relative_to(ROOT)): _sha256_file(path)
        for path in sorted(_authoritative_source_paths())
    }


def _validate_authoritative_sources(expected: Mapping[str, Any]) -> None:
    if any(
        not isinstance(relative, str) or not _valid_sha256(digest)
        for relative, digest in expected.items()
    ):
        raise RuntimeError("authoritative source snapshot is malformed")
    expected_paths = set(expected)
    actual_paths = {
        str(path.relative_to(ROOT))
        for path in _authoritative_source_paths()
    }
    if actual_paths != expected_paths:
        added = sorted(actual_paths - expected_paths)
        removed = sorted(expected_paths - actual_paths)
        raise RuntimeError(
            "authoritative source path-set drift: "
            f"added={added[:5]} removed={removed[:5]}"
        )
    _validate_source_snapshot(expected)


def _validate_governed_source_members(expected: Mapping[str, Any]) -> None:
    if set(expected) != set(GOVERNED_SOURCE_GLOBS):
        raise RuntimeError(
            "run governed-source member snapshot has malformed glob keys"
        )
    normalized: dict[str, list[str]] = {}
    for pattern in GOVERNED_SOURCE_GLOBS:
        members = expected[pattern]
        if (
            not isinstance(members, (list, tuple))
            or any(not isinstance(member, str) for member in members)
        ):
            raise RuntimeError(
                f"run governed-source member snapshot is malformed for {pattern}"
            )
        normalized[pattern] = sorted(members)
    actual = _snapshot_governed_source_members()
    if actual != normalized:
        differences: list[str] = []
        for pattern in GOVERNED_SOURCE_GLOBS:
            expected_members = set(normalized[pattern])
            actual_members = set(actual[pattern])
            added = sorted(actual_members - expected_members)
            removed = sorted(expected_members - actual_members)
            if added:
                differences.append(f"{pattern}: added={added[:5]}")
            if removed:
                differences.append(f"{pattern}: removed={removed[:5]}")
        raise RuntimeError(
            "run governed-source directory membership drift:\n"
            + "\n".join(differences)
        )


def _validate_import_bound_sources() -> None:
    _validate_source_snapshot(_IMPORT_BOUND_SOURCE_HASHES)


def _snapshot_loaded_project_sources() -> dict[str, str]:
    paths = {
        (ROOT / relative).resolve()
        for relative in _snapshot_static_run_sources()
    }
    for module in tuple(sys.modules.values()):
        source = getattr(module, "__file__", None)
        if not source:
            continue
        path = Path(source).resolve()
        if path.suffix in {".pyc", ".pyo"}:
            candidate = Path(str(path)[:-1])
            if candidate.exists():
                path = candidate
        try:
            path.relative_to(ROOT)
        except ValueError:
            continue
        if path.is_file():
            paths.add(path)
    paths.update(
        (
            (DOCS / "data.md").resolve(),
            (DOCS / "repro_data.json").resolve(),
            SEED.resolve(),
            (ROOT / "researchpaper/uiuc_polars/SD7003.DRG").resolve(),
        )
    )
    return {
        str(path.relative_to(ROOT)): _sha256_file(path)
        for path in sorted(paths)
    }


def _validate_loaded_sources_registered(
    loaded: Mapping[str, Any],
    authoritative: Mapping[str, Any],
) -> None:
    if any(
        not isinstance(relative, str) or not _valid_sha256(digest)
        for relative, digest in loaded.items()
    ):
        raise RuntimeError("loaded project source snapshot is malformed")
    unregistered = sorted(set(loaded) - set(authoritative))
    if unregistered:
        raise RuntimeError(
            "dynamic import loaded unregistered project sources: "
            f"{unregistered[:10]}"
        )
    mismatched = sorted(
        relative
        for relative, digest in loaded.items()
        if authoritative.get(relative) != digest
    )
    if mismatched:
        raise RuntimeError(
            "dynamic import source differs from pre-import authority: "
            f"{mismatched[:10]}"
        )


def _validate_current_dynamic_imports(
    authoritative: Mapping[str, Any],
) -> None:
    _validate_loaded_sources_registered(
        _snapshot_loaded_project_sources(),
        authoritative,
    )


def _validate_source_snapshot(expected: Mapping[str, str]) -> None:
    drift: list[str] = []
    for relative, digest in expected.items():
        path = ROOT / relative
        if not path.is_file():
            drift.append(f"{relative}: missing")
            continue
        actual = _sha256_file(path)
        if actual != digest:
            drift.append(f"{relative}: {actual} != {digest}")
    if drift:
        raise RuntimeError("run source identity drift:\n" + "\n".join(drift[:20]))


def _valid_result_record(value: Any) -> bool:
    if not isinstance(value, Mapping) or set(value) != {"L", "T"}:
        return False
    try:
        return all(
            not isinstance(value[name], bool)
            and isinstance(value[name], (int, float))
            and math.isfinite(float(value[name]))
            for name in ("L", "T")
        )
    except (TypeError, ValueError):
        return False


def _valid_force_guard(value: Any) -> bool:
    if not isinstance(value, Mapping) or value.get("passed") is not True:
        return False
    if isinstance(value.get("max_abs_error_N"), bool) or isinstance(
        value.get("tolerance_N"), bool
    ):
        return False
    try:
        error = float(value["max_abs_error_N"])
        tolerance = float(value["tolerance_N"])
    except (KeyError, TypeError, ValueError):
        return False
    return (
        math.isfinite(error)
        and math.isfinite(tolerance)
        and error >= 0.0
        and tolerance >= 0.0
        and error <= tolerance
    )


def _valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _claim_graph_identity_payload(manifest: Mapping[str, Any]) -> dict[str, Any]:
    closure = manifest.get("closure")
    topology = manifest.get("topology")
    nodes = manifest.get("nodes")
    parameter_sources = manifest.get("parameter_sources")
    if not isinstance(closure, str) or not closure:
        raise RuntimeError("claim manifest lacks a valid closure")
    if (
        not isinstance(topology, list)
        or not topology
        or any(not isinstance(node_id, str) or not node_id for node_id in topology)
        or len(set(topology)) != len(topology)
    ):
        raise RuntimeError("claim manifest lacks a valid topology")
    if not isinstance(nodes, list) or len(nodes) != len(topology):
        raise RuntimeError("claim manifest lacks a valid node inventory")
    identity_fields = (
        "id",
        "state",
        "freeze",
        "runtime_role",
        "implementation",
        "implementation_version",
        "implementation_hash",
    )
    normalized_nodes: list[dict[str, Any]] = []
    for index, node in enumerate(nodes):
        if not isinstance(node, Mapping):
            raise RuntimeError("claim manifest contains a malformed node")
        normalized = {field: node.get(field) for field in identity_fields}
        if normalized["id"] != topology[index]:
            raise RuntimeError(
                "claim manifest node order does not match its topology"
            )
        if not isinstance(normalized["freeze"], bool):
            raise RuntimeError(
                f"claim manifest node {normalized['id']!r} has invalid freeze state"
            )
        for field in identity_fields:
            if field == "freeze":
                continue
            if not isinstance(normalized[field], str):
                raise RuntimeError(
                    f"claim manifest node {normalized['id']!r} "
                    f"has invalid {field}"
                )
        if not str(normalized["implementation_hash"]).startswith("sha256:"):
            raise RuntimeError(
                f"claim manifest node {normalized['id']!r} "
                "has invalid implementation_hash"
            )
        normalized_nodes.append(normalized)
    if (
        not isinstance(parameter_sources, Mapping)
        or any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in parameter_sources.items()
        )
    ):
        raise RuntimeError("claim manifest lacks valid parameter_sources")
    return {
        "closure": closure,
        "topology": list(topology),
        "nodes": normalized_nodes,
        "parameter_sources": dict(sorted(parameter_sources.items())),
    }


def _claim_graph_identity_sha256(manifest: Mapping[str, Any]) -> str:
    return _canonical_hash(_claim_graph_identity_payload(manifest))


def _valid_case_guard(
    value: Any,
    expected_claim_graph_identity: str | None = None,
) -> bool:
    if not isinstance(value, Mapping):
        return False
    graph_identity = value.get("claim_graph_identity_sha256")
    return (
        _valid_force_guard(value.get("force_ledger"))
        and _valid_force_guard(value.get("unclassified_force"))
        and _valid_force_guard(value.get("unclassified_physical_force"))
        and _valid_force_guard(value.get("cycle_reduction"))
        and _valid_force_guard(value.get("aero_output_invariance"))
        and _valid_sha256(value.get("claim_manifest_sha256"))
        and _valid_sha256(graph_identity)
        and (
            expected_claim_graph_identity is None
            or graph_identity == expected_claim_graph_identity
        )
    )


def _sanitize_resume_checkpoint(
    checkpoint: dict[str, Any],
    *,
    seed: Mapping[str, Any],
    expected_keys: set[str],
    case_guards: dict[str, Any],
    claim_graph_identity_sha256: str,
) -> list[str]:
    unexpected = set(checkpoint) - expected_keys
    if unexpected:
        raise RuntimeError(
            f"resume checkpoint contains unexpected condition keys: "
            f"{sorted(unexpected)[:5]}"
        )
    changed_seed = [
        key for key, value in seed.items() if checkpoint.get(key) != value
    ]
    if changed_seed:
        raise RuntimeError(
            f"resume checkpoint changed frozen seed values: {changed_seed[:5]}"
        )

    discarded: list[str] = []
    for key in sorted(set(checkpoint) - set(seed)):
        guard = case_guards.get(key)
        if not _valid_result_record(checkpoint[key]) or not _valid_case_guard(
            guard, claim_graph_identity_sha256
        ):
            checkpoint.pop(key, None)
            case_guards.pop(key, None)
            discarded.append(key)
    return discarded


def _validate_seed(seed: Mapping[str, Any]) -> dict[str, Any]:
    frozen_keys = {key_of(*condition) for condition in FROZEN_118_CONDITIONS}
    if len(frozen_keys) != 118:
        raise RuntimeError(f"lb_sweep118 contract drift: {len(frozen_keys)} keys")
    if set(seed) != frozen_keys:
        raise RuntimeError(
            "frozen seed key mismatch: "
            f"missing={sorted(frozen_keys - set(seed))[:5]} "
            f"extra={sorted(set(seed) - frozen_keys)[:5]}"
        )
    invalid = sorted(
        key
        for key, value in seed.items()
        if not _valid_result_record(value)
    )
    if invalid:
        raise RuntimeError(f"frozen seed contains invalid results: {invalid[:5]}")
    report = coverage(seed)
    if (
        report["valid_unique_conditions"] != 118
        or report["missing_unique_conditions"] != 66
        or report["complete_curves"] != 38
    ):
        raise RuntimeError(f"unexpected frozen-seed coverage: {report}")
    return report


def _resolved_call(gpu_run_twist: Any, condition: tuple[float, float, float, float]) -> dict[str, Any]:
    U, freq, twist, aoa = condition
    call = dict(BASE)
    call.update(
        U=U,
        aoa_deg=aoa,
        freq=freq,
        twist_amp_deg=twist / 2.0,
        nc=12,
        ns=16,
        n_cycle=4,
        steps_per_cycle=spc_of(U, freq),
        wake_rows=spc_of(U, freq),
        closure="v41",
    )
    signature = inspect.signature(gpu_run_twist)
    resolved: dict[str, Any] = {
        name: parameter.default
        for name, parameter in signature.parameters.items()
        if parameter.default is not inspect.Parameter.empty
    }
    resolved.update(call)
    return resolved


def _run_condition(
    gpu_run_twist: Any,
    condition: tuple[float, float, float, float],
) -> tuple[dict[str, float], dict[str, Any]]:
    U, freq, twist, aoa = condition
    spc = spc_of(U, freq)
    result = gpu_run_twist(
        U=U,
        aoa_deg=aoa,
        freq=freq,
        twist_amp_deg=twist / 2.0,
        nc=12,
        ns=16,
        n_cycle=4,
        steps_per_cycle=spc,
        wake_rows=spc,
        closure="v41",
        **BASE,
    )
    value = {"L": float(result["L_wind"]), "T": float(result["T_wind"])}
    if not _valid_result_record(value):
        raise RuntimeError(
            f"{condition_key(condition)} returned non-finite or malformed force result"
        )
    guards = result.get("claim_guards")
    guard_summary = {
        name: dict(guards[name])
        for name in (
            "force_ledger",
            "unclassified_force",
            "unclassified_physical_force",
            "cycle_reduction",
            "aero_output_invariance",
        )
        if isinstance(guards, Mapping) and isinstance(guards.get(name), Mapping)
    }
    if not (
        _valid_force_guard(guard_summary.get("force_ledger"))
        and _valid_force_guard(guard_summary.get("unclassified_force"))
        and _valid_force_guard(
            guard_summary.get("unclassified_physical_force")
        )
        and _valid_force_guard(guard_summary.get("cycle_reduction"))
        and _valid_force_guard(
            guard_summary.get("aero_output_invariance")
        )
    ):
        raise RuntimeError(
            f"{condition_key(condition)} failed claim force-ledger guards: "
            f"{guard_summary}"
        )
    manifest = result.get("claim_manifest")
    if not isinstance(manifest, Mapping):
        raise RuntimeError(
            f"{condition_key(condition)} returned no claim graph manifest"
        )
    guard_summary["claim_manifest_sha256"] = _canonical_hash(manifest)
    guard_summary["claim_graph_identity_sha256"] = (
        _claim_graph_identity_sha256(manifest)
    )
    return value, guard_summary


def _default_paths(timestamp: str) -> tuple[Path, Path, Path]:
    result = DOCS / f"s6_sweep_v41_full184_{timestamp}.json"
    manifest = DIAG / f"fig171819_v41_baseline_manifest_{timestamp}.json"
    score = SCORECARDS / f"scorecard_v41_full184_{timestamp}.json"
    return result, manifest, score


def _target_contract_payload() -> dict[str, Any]:
    return {
        "conditions": [list(condition) for condition in CONDITIONS],
        "curves": [
            {
                "figure": curve.figure,
                "panel": curve.panel,
                "key": curve.key,
                "channel": curve.channel,
                "x": list(curve.x),
                "conditions": [list(condition) for condition in curve.conditions],
            }
            for curve in CURVES
        ],
    }


def _call_contract_sha256(gpu_run_twist: Any) -> str:
    return _canonical_hash(
        {
            condition_key(condition): _resolved_call(gpu_run_twist, condition)
            for condition in CONDITIONS
        }
    )


def _runtime_identity() -> dict[str, Any]:
    environment_names = (
        "FLUXV_DTYPE",
        "FLUXV_DEVICE",
        "CUDA_VISIBLE_DEVICES",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    )
    identity: dict[str, Any] = {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": py_platform.platform(),
        "environment": {
            name: os.environ.get(name) for name in environment_names
        },
    }
    try:
        import numpy as np

        identity["numpy"] = np.__version__
    except Exception:
        identity["numpy"] = None
    try:
        import warp as wp
        from fluxvortex.warp_fsi import config as cfg

        identity["warp"] = getattr(wp, "__version__", None)
        solver_device = str(cfg.DEVICE)
        identity["solver_config"] = {
            "dtype_name": cfg.dtype_name(),
            "dtype": str(cfg.DTYPE),
            "numpy_dtype": str(np.dtype(cfg.NP_DTYPE)),
            "device": solver_device,
            "CR_TOL": float(cfg.CR_TOL),
            "NEWTON_TOL": float(cfg.NEWTON_TOL),
            "GEOM_ATOL": float(cfg.GEOM_ATOL),
            "PORT_ATOL": float(cfg.PORT_ATOL),
        }
        device = wp.get_device(solver_device)
        identity["warp_device"] = {
            "text": str(device),
            "alias": getattr(device, "alias", None),
            "name": getattr(device, "name", None),
            "arch": getattr(device, "arch", None),
        }
    except Exception as exc:
        identity["warp_device_error"] = f"{type(exc).__name__}: {exc}"
    return identity


def _runtime_session(identity: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "started_at": datetime.now().astimezone().isoformat(),
        "pid": os.getpid(),
        "identity": dict(identity),
    }


def _count_valid_new_results(
    checkpoint: Mapping[str, Any],
    seed: Mapping[str, Any],
    case_guards: Mapping[str, Any],
    claim_graph_identity_sha256: str | None = None,
) -> int:
    return sum(
        _valid_result_record(value)
        and _valid_case_guard(
            case_guards.get(key), claim_graph_identity_sha256
        )
        for key, value in checkpoint.items()
        if key not in seed
    )


def _fixed_manifest_is_newer(manifest: Mapping[str, Any]) -> bool:
    if not FIXED_MANIFEST.exists():
        return False
    try:
        fixed = _load_json(FIXED_MANIFEST)
        fixed_started = datetime.fromisoformat(str(fixed["started_at"]))
        run_started = datetime.fromisoformat(str(manifest["started_at"]))
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False
    return fixed_started > run_started


def _publish_complete(
    *,
    result_path: Path,
    manifest_path: Path,
    scorecard_path: Path,
    manifest: dict[str, Any],
) -> None:
    result_sha256 = _sha256_file(result_path)
    scorecard_sha256 = _sha256_file(scorecard_path)
    now = datetime.now().astimezone().isoformat()
    manifest.update(
        {
            "status": "publication_pending",
            "updated_at": now,
            "result_sha256": result_sha256,
            "scorecard_sha256": scorecard_sha256,
        }
    )
    _write_json_atomic(manifest_path, manifest)

    if _fixed_manifest_is_newer(manifest):
        manifest["status"] = "completed"
        manifest["completed_at"] = now
        manifest["updated_at"] = now
        manifest["publication"] = {
            "published_latest": False,
            "reason": "a newer completed run already owns the fixed-name artifacts",
        }
        _write_json_atomic(manifest_path, manifest)
        return

    _copy_atomic(result_path, FIXED_RESULT)
    _copy_atomic(scorecard_path, FIXED_SCORECARD)
    if _sha256_file(FIXED_RESULT) != result_sha256:
        raise RuntimeError("fixed result failed post-publication hash verification")
    if _sha256_file(FIXED_SCORECARD) != scorecard_sha256:
        raise RuntimeError("fixed scorecard failed post-publication hash verification")

    receipt = {
        "published_latest": True,
        "published_at": datetime.now().astimezone().isoformat(),
        "versioned_result_path": str(result_path.relative_to(ROOT)),
        "versioned_manifest_path": str(manifest_path.relative_to(ROOT)),
        "versioned_scorecard_path": str(scorecard_path.relative_to(ROOT)),
        "fixed_result_path": str(FIXED_RESULT.relative_to(ROOT)),
        "fixed_scorecard_path": str(FIXED_SCORECARD.relative_to(ROOT)),
        "fixed_manifest_path": str(FIXED_MANIFEST.relative_to(ROOT)),
        "result_sha256": result_sha256,
        "scorecard_sha256": scorecard_sha256,
    }
    fixed_manifest = json.loads(json.dumps(manifest, allow_nan=False))
    fixed_manifest["status"] = "completed"
    fixed_manifest["completed_at"] = receipt["published_at"]
    fixed_manifest["updated_at"] = receipt["published_at"]
    fixed_manifest["publication"] = receipt
    _write_json_atomic(FIXED_MANIFEST, fixed_manifest)

    manifest["status"] = "completed"
    manifest["completed_at"] = receipt["published_at"]
    manifest["updated_at"] = receipt["published_at"]
    manifest["publication"] = {
        **receipt,
        "fixed_manifest_sha256": _sha256_file(FIXED_MANIFEST),
    }
    _write_json_atomic(manifest_path, manifest)


def _run_locked(
    *,
    result_path: Path,
    manifest_path: Path,
    scorecard_path: Path,
    dry_run: bool = False,
    resume: bool = False,
) -> int:
    _validate_import_bound_sources()
    source_hashes = _validate_frozen_identity()
    if resume and not dry_run:
        if not result_path.exists() or not manifest_path.exists():
            raise RuntimeError(
                "--resume-result requires the versioned result and manifest"
            )
        manifest = _load_json(manifest_path)
        authoritative_source_hashes = manifest.get(
            "authoritative_source_hashes"
        )
        if not isinstance(authoritative_source_hashes, Mapping):
            raise RuntimeError(
                "resume manifest lacks an authoritative pre-import source snapshot"
            )
    else:
        authoritative_source_hashes = _snapshot_authoritative_sources()
    _validate_authoritative_sources(authoritative_source_hashes)
    if any(
        authoritative_source_hashes.get(relative) != digest
        for relative, digest in source_hashes.items()
    ):
        raise RuntimeError(
            "frozen V4.1 hashes disagree with authoritative source snapshot"
        )
    seed = _load_json(SEED)
    seed_coverage = _validate_seed(seed)
    expected_keys = {condition_key(condition) for condition in CONDITIONS}
    target_contract_sha256 = _canonical_hash(_target_contract_payload())
    missing_conditions = [
        condition
        for condition in CONDITIONS
        if condition_key(condition) not in seed
    ]
    if len(missing_conditions) != 66:
        raise RuntimeError(
            f"complete benchmark drift: {len(missing_conditions)} missing, expected 66"
        )

    if dry_run:
        print(
            "Fig17/18/19 V4.1 conditional f19(c,d)=2.6 contract: "
            f"{len(seed)}/184 seeded, {len(missing_conditions)} GPU runs required"
        )
        print(
            "publication gate: dual-scope contract (2026-08-01) — "
            "fig19(c,d)=conditional diagnostic domain, confirmed promotion "
            "domain ready; 66 GPU runs still required"
        )
        print(f"first missing: {condition_key(missing_conditions[0])}")
        print(f"last missing:  {condition_key(missing_conditions[-1])}")
        return 0

    if resume:
        checkpoint = _load_json(result_path)
        if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
            raise RuntimeError(
                "resume manifest schema mismatch: "
                f"{manifest.get('schema_version')!r} != "
                f"{MANIFEST_SCHEMA_VERSION}"
            )
        if manifest.get("source_hashes") != source_hashes:
            raise RuntimeError("resume source identity differs from the run manifest")
        if manifest.get("base_profile_sha256") != _canonical_hash(BASE):
            raise RuntimeError("resume BASE profile differs from the run manifest")
        if manifest.get("target_contract_sha256") != target_contract_sha256:
            raise RuntimeError("resume target184/curve contract differs from the run manifest")
        if manifest.get("status") not in {
            "running",
            "incomplete",
            "failed_case",
            "awaiting_data_identity",
            "publication_pending",
            "completed",
        }:
            raise RuntimeError(
                f"run status {manifest.get('status')!r} is not resumable"
            )
        tracked_sources = manifest.get("tracked_source_hashes")
        if tracked_sources is not None:
            if not isinstance(tracked_sources, Mapping):
                raise RuntimeError("resume manifest has malformed tracked_source_hashes")
            _validate_source_snapshot(tracked_sources)
        governed_source_members = manifest.get("governed_source_members")
        if not isinstance(governed_source_members, Mapping):
            raise RuntimeError(
                "resume manifest lacks a valid governed-source member snapshot"
            )
        _validate_governed_source_members(governed_source_members)
        case_guards = manifest.setdefault("case_guards", {})
        if not isinstance(case_guards, dict):
            raise RuntimeError("resume manifest has malformed case_guards")
        claim_graph_identity_sha256 = manifest.get(
            "claim_graph_identity_sha256"
        )
        if not _valid_sha256(claim_graph_identity_sha256):
            raise RuntimeError(
                "resume manifest lacks a valid frozen claim graph identity"
            )
        discarded = _sanitize_resume_checkpoint(
            checkpoint,
            seed=seed,
            expected_keys=expected_keys,
            case_guards=case_guards,
            claim_graph_identity_sha256=claim_graph_identity_sha256,
        )
        manifest.setdefault("resume_events", []).append(
            {
                "at": datetime.now().astimezone().isoformat(),
                "discarded_invalid_condition_keys": discarded,
            }
        )
        manifest["status"] = "running"
        manifest["updated_at"] = datetime.now().astimezone().isoformat()
        manifest["failures"] = {
            key: value
            for key, value in manifest.get("failures", {}).items()
            if key not in checkpoint
        }
        manifest["completed_new_conditions"] = _count_valid_new_results(
            checkpoint,
            seed,
            case_guards,
            claim_graph_identity_sha256,
        )
        if discarded:
            _write_json_atomic(result_path, checkpoint)
        _write_json_atomic(manifest_path, manifest)
    else:
        if result_path.exists() or manifest_path.exists() or scorecard_path.exists():
            raise RuntimeError(
                "refusing to replace a versioned run artifact; use "
                "--resume-result or choose a fresh --timestamp"
            )
        checkpoint = dict(seed)
        case_guards: dict[str, Any] = {}
        started = datetime.now().astimezone().isoformat()
        manifest = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "campaign": "FLUXV_V4.1_Fig17_Fig18_Fig19_complete_baseline",
            "run_id": result_path.stem.removeprefix("s6_sweep_v41_full184_"),
            "status": "running",
            "started_at": started,
            "updated_at": started,
            "result_path": str(result_path.relative_to(ROOT)),
            "scorecard_path": str(scorecard_path.relative_to(ROOT)),
            "source_hashes": source_hashes,
            "authoritative_source_hashes": authoritative_source_hashes,
            "tracked_source_hashes": None,
            "governed_source_members": _snapshot_governed_source_members(),
            "claim_graph_identity_sha256": None,
            "base_profile_sha256": _canonical_hash(BASE),
            "target_contract_sha256": target_contract_sha256,
            "call_contract_sha256": None,
            "seed_coverage": seed_coverage,
            "expected_unique_conditions": len(expected_keys),
            "missing_condition_count_at_start": len(missing_conditions),
            "completed_new_conditions": 0,
            "failures": {},
            "case_guards": case_guards,
            "runtime_identity": None,
            "runtime_sessions": [],
            "cold_preconditioner": None,
            "formal_anchor": None,
            "resume_events": [],
            "data_identity_gate": {
                "fig19_cd_fixed_frequency": "unresolved",
                "conditional_assumption_hz": 2.6,
                "publication_authorized": False,
            },
        }
        _write_json_atomic(result_path, checkpoint)
        _write_json_atomic(manifest_path, manifest)

    import warp as wp

    wp.init()
    from _v2_robo import gpu_run_twist

    _validate_import_bound_sources()
    _validate_authoritative_sources(manifest["authoritative_source_hashes"])
    tracked_source_hashes = _snapshot_loaded_project_sources()
    _validate_loaded_sources_registered(
        tracked_source_hashes,
        manifest["authoritative_source_hashes"],
    )
    if manifest.get("tracked_source_hashes") is None:
        manifest["tracked_source_hashes"] = tracked_source_hashes
    elif manifest["tracked_source_hashes"] != tracked_source_hashes:
        raise RuntimeError("resume loaded-module source identity differs from manifest")
    _validate_authoritative_sources(manifest["authoritative_source_hashes"])
    _validate_source_snapshot(manifest["tracked_source_hashes"])
    _validate_governed_source_members(manifest["governed_source_members"])

    call_contract_sha256 = _call_contract_sha256(gpu_run_twist)
    if manifest.get("call_contract_sha256") is None:
        manifest["call_contract_sha256"] = call_contract_sha256
    elif manifest["call_contract_sha256"] != call_contract_sha256:
        raise RuntimeError("resume resolved gpu_run_twist call contract differs")

    runtime_identity = _runtime_identity()
    if manifest.get("runtime_identity") is None:
        manifest["runtime_identity"] = runtime_identity
    elif manifest["runtime_identity"] != runtime_identity:
        raise RuntimeError(
            "resume runtime/device identity differs from the original run"
        )
    manifest.setdefault("runtime_sessions", []).append(
        _runtime_session(runtime_identity)
    )
    manifest["resolved_anchor_call"] = _resolved_call(gpu_run_twist, ANCHOR)
    _write_json_atomic(manifest_path, manifest)

    cold_start = time.time()
    _validate_authoritative_sources(manifest["authoritative_source_hashes"])
    _validate_source_snapshot(manifest["tracked_source_hashes"])
    _validate_governed_source_members(manifest["governed_source_members"])
    cold, cold_guards = _run_condition(gpu_run_twist, ANCHOR)
    _validate_authoritative_sources(manifest["authoritative_source_hashes"])
    _validate_source_snapshot(manifest["tracked_source_hashes"])
    _validate_governed_source_members(manifest["governed_source_members"])
    _validate_current_dynamic_imports(manifest["authoritative_source_hashes"])
    cold_graph_identity = cold_guards["claim_graph_identity_sha256"]
    campaign_graph_identity = manifest.get("claim_graph_identity_sha256")
    if campaign_graph_identity is None:
        manifest["claim_graph_identity_sha256"] = cold_graph_identity
        campaign_graph_identity = cold_graph_identity
    elif campaign_graph_identity != cold_graph_identity:
        raise RuntimeError(
            "cold preconditioner claim graph identity differs from campaign"
        )
    manifest["cold_preconditioner"] = {
        "excluded_from_benchmark": True,
        "condition": condition_key(ANCHOR),
        "result": cold,
        "guards": cold_guards,
        "wall_s": time.time() - cold_start,
    }
    _write_json_atomic(manifest_path, manifest)

    anchor_start = time.time()
    _validate_authoritative_sources(manifest["authoritative_source_hashes"])
    _validate_source_snapshot(manifest["tracked_source_hashes"])
    _validate_governed_source_members(manifest["governed_source_members"])
    formal_anchor, formal_anchor_guards = _run_condition(gpu_run_twist, ANCHOR)
    _validate_authoritative_sources(manifest["authoritative_source_hashes"])
    _validate_source_snapshot(manifest["tracked_source_hashes"])
    _validate_governed_source_members(manifest["governed_source_members"])
    _validate_current_dynamic_imports(manifest["authoritative_source_hashes"])
    anchor_key = condition_key(ANCHOR)
    anchor_reference = seed[anchor_key]
    anchor_delta = {
        name: abs(float(formal_anchor[name]) - float(anchor_reference[name]))
        for name in ("L", "T")
    }
    anchor_graph_identity_passed = (
        formal_anchor_guards["claim_graph_identity_sha256"]
        == campaign_graph_identity
    )
    anchor_passed = (
        max(anchor_delta.values()) <= ANCHOR_TOLERANCE_N
        and anchor_graph_identity_passed
    )
    manifest["formal_anchor"] = {
        "condition": anchor_key,
        "reference": anchor_reference,
        "result": formal_anchor,
        "guards": formal_anchor_guards,
        "absolute_delta_N": anchor_delta,
        "tolerance_N": ANCHOR_TOLERANCE_N,
        "claim_graph_identity_passed": anchor_graph_identity_passed,
        "passed": anchor_passed,
        "wall_s": time.time() - anchor_start,
    }
    _write_json_atomic(manifest_path, manifest)
    if not anchor_passed:
        manifest["status"] = "failed_identity_gate"
        manifest["updated_at"] = datetime.now().astimezone().isoformat()
        _write_json_atomic(manifest_path, manifest)
        raise RuntimeError(f"formal V4.1 anchor failed: {manifest['formal_anchor']}")

    todo_conditions = [
        condition
        for condition in missing_conditions
        if not _valid_result_record(checkpoint.get(condition_key(condition)))
        or not _valid_case_guard(
            case_guards.get(condition_key(condition)),
            campaign_graph_identity,
        )
    ]
    run_start = time.time()
    for index, condition in enumerate(todo_conditions, start=1):
        key = condition_key(condition)
        case_start = time.time()
        try:
            _validate_authoritative_sources(
                manifest["authoritative_source_hashes"]
            )
            _validate_source_snapshot(manifest["tracked_source_hashes"])
            _validate_governed_source_members(manifest["governed_source_members"])
            value, guards = _run_condition(gpu_run_twist, condition)
            _validate_authoritative_sources(
                manifest["authoritative_source_hashes"]
            )
            _validate_source_snapshot(manifest["tracked_source_hashes"])
            _validate_governed_source_members(manifest["governed_source_members"])
            _validate_current_dynamic_imports(
                manifest["authoritative_source_hashes"]
            )
            if not _valid_case_guard(guards, campaign_graph_identity):
                raise RuntimeError(
                    f"{key} claim graph identity differs from campaign"
                )
            checkpoint[key] = value
            case_guards[key] = guards
            manifest["failures"].pop(key, None)
            print(
                f"[v41-full184] {index}/{len(todo_conditions)} {key}: "
                f"L={value['L']:+.3f} T={value['T']:+.3f} "
                f"({time.time() - case_start:.0f}s)",
                flush=True,
            )
        except Exception as exc:
            manifest["failures"][key] = f"{type(exc).__name__}: {exc}"
            manifest["status"] = "failed_case"
            manifest["completed_new_conditions"] = _count_valid_new_results(
                checkpoint,
                seed,
                case_guards,
                campaign_graph_identity,
            )
            manifest["updated_at"] = datetime.now().astimezone().isoformat()
            manifest["run_wall_s"] = time.time() - run_start
            _write_json_atomic(result_path, checkpoint)
            _write_json_atomic(manifest_path, manifest)
            print(
                f"[v41-full184] {index}/{len(todo_conditions)} {key}: "
                f"FAIL {type(exc).__name__}: {exc}",
                flush=True,
            )
            raise RuntimeError(
                f"{key} failed; stopped to require a fresh-process resume"
            ) from exc
        manifest["completed_new_conditions"] = _count_valid_new_results(
            checkpoint,
            seed,
            case_guards,
            campaign_graph_identity,
        )
        manifest["updated_at"] = datetime.now().astimezone().isoformat()
        manifest["run_wall_s"] = time.time() - run_start
        _write_json_atomic(result_path, checkpoint)
        _write_json_atomic(manifest_path, manifest)

    _validate_authoritative_sources(manifest["authoritative_source_hashes"])
    _validate_source_snapshot(manifest["tracked_source_hashes"])
    _validate_governed_source_members(manifest["governed_source_members"])
    _validate_current_dynamic_imports(manifest["authoritative_source_hashes"])
    repro = _load_json(DEFAULT_REPRO)
    report = scorecard(checkpoint, repro, sweep_name=str(result_path.relative_to(ROOT)))
    _validate_authoritative_sources(manifest["authoritative_source_hashes"])
    _validate_source_snapshot(manifest["tracked_source_hashes"])
    _validate_governed_source_members(manifest["governed_source_members"])
    _validate_current_dynamic_imports(manifest["authoritative_source_hashes"])
    _write_json_atomic(scorecard_path, report)
    manifest["final_coverage"] = report["coverage"]
    manifest["scorecard_sha256"] = _sha256_file(scorecard_path)

    frequency_contract = report.get("contract", {}).get(
        "fig19_cd_frequency", {}
    )
    data_gate = (
        frequency_contract.get("status", "unresolved")
        if isinstance(frequency_contract, Mapping)
        else "unresolved"
    )
    # 双 scope 契约(2026-08-01): fig19_cd frequency 以 conditional_scope 状态
    # 终裁——confirmed 域晋升不再被其阻塞,conditional_fig19_cd 只作诊断域。
    if data_gate not in ("resolved", "conditional_scope"):
        manifest["status"] = (
            "awaiting_data_identity"
            if report["coverage"]["complete"]
            else "incomplete"
        )
        manifest["data_identity_gate"] = {
            "fig19_cd_fixed_frequency": data_gate,
            "conditional_assumption_hz": 2.6,
            "publication_authorized": False,
        }
        manifest["updated_at"] = datetime.now().astimezone().isoformat()
        _write_json_atomic(manifest_path, manifest)
        print(
            "[v41-full184] numerical grid complete under the conditional "
            "f19(c,d)=2.6 assumption; publication remains blocked by data identity",
            flush=True,
        )
        return 2

    if not report["promotion_eligible"] or manifest["failures"]:
        manifest["status"] = "incomplete"
        manifest["final_coverage"] = report["coverage"]
        manifest["updated_at"] = datetime.now().astimezone().isoformat()
        _write_json_atomic(manifest_path, manifest)
        return 2

    if set(checkpoint) != expected_keys:
        raise RuntimeError("complete result contains missing or unexpected condition keys")
    if _count_valid_new_results(
        checkpoint,
        seed,
        case_guards,
        campaign_graph_identity,
    ) != len(expected_keys - set(seed)):
        raise RuntimeError("complete checkpoint lacks valid per-case force-ledger guards")
    manifest["data_identity_gate"]["publication_authorized"] = True
    _publish_complete(
        result_path=result_path,
        manifest_path=manifest_path,
        scorecard_path=scorecard_path,
        manifest=manifest,
    )
    print(
        f"[v41-full184] COMPLETE 184/184 -> {result_path} "
        f"(fixed latest: {FIXED_RESULT})",
        flush=True,
    )
    return 0


def run(
    *,
    result_path: Path,
    manifest_path: Path,
    scorecard_path: Path,
    dry_run: bool = False,
    resume: bool = False,
) -> int:
    if dry_run:
        return _run_locked(
            result_path=result_path,
            manifest_path=manifest_path,
            scorecard_path=scorecard_path,
            dry_run=True,
            resume=resume,
        )
    with _exclusive_campaign_lock():
        return _run_locked(
            result_path=result_path,
            manifest_path=manifest_path,
            scorecard_path=scorecard_path,
            dry_run=False,
            resume=resume,
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--timestamp",
        default=datetime.now().strftime("%Y%m%d_%H%M%S"),
        help="version suffix for a fresh run",
    )
    parser.add_argument(
        "--resume-result",
        type=Path,
        help=(
            "resume a timestamped s6_sweep_v41_full184_*.json checkpoint; "
            "its companion manifest is resolved from the same suffix"
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.resume_result is None:
        if re.fullmatch(r"\d{8}_\d{6}", args.timestamp) is None:
            parser.error("--timestamp must use YYYYMMDD_HHMMSS")
        result, manifest, report = _default_paths(args.timestamp)
        resume = False
    else:
        result = args.resume_result.resolve()
        prefix = "s6_sweep_v41_full184_"
        if not result.name.startswith(prefix) or result.suffix != ".json":
            parser.error(
                "--resume-result must be named "
                "s6_sweep_v41_full184_<timestamp>.json"
            )
        if result.parent != DOCS.resolve():
            parser.error("--resume-result must be inside platform/docs")
        timestamp = result.name[len(prefix) : -len(".json")]
        if re.fullmatch(r"\d{8}_\d{6}", timestamp) is None:
            parser.error("--resume-result timestamp must use YYYYMMDD_HHMMSS")
        manifest = DIAG / f"fig171819_v41_baseline_manifest_{timestamp}.json"
        report = SCORECARDS / f"scorecard_v41_full184_{timestamp}.json"
        resume = True
    return run(
        result_path=result,
        manifest_path=manifest,
        scorecard_path=report,
        dry_run=args.dry_run,
        resume=resume,
    )


if __name__ == "__main__":
    raise SystemExit(main())
