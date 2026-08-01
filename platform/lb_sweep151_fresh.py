"""Build a zero-seed-reuse V4.1 baseline for the confirmed 151 conditions."""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import platform as py_platform
import sys
import tempfile
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np


PLATFORM = Path(__file__).resolve().parent
ROOT = PLATFORM.parent
DOCS = PLATFORM / "docs"
DIAG = DOCS / "diag"
for path in (ROOT, PLATFORM):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_claim_witnesses as witness  # noqa: E402
from fig171819_benchmark import (  # noqa: E402
    CONDITIONS_BY_EVIDENCE_SCOPE,
    EVIDENCE_CONFIRMED,
    condition_key,
    coverage,
)
from lb_sweep118 import BASE, spc_of  # noqa: E402


PREREG = DIAG / "v41_confirmed151_fresh_prereg_20260729.md"
INTEGRITY_ADDENDUM = (
    DIAG / "v41_confirmed151_resume_integrity_addendum_20260729.md"
)
LAUNCH_AUTHORIZATION = (
    DIAG / "v41_confirmed151_launch_authorization_20260729.md"
)
OLD_MERGED_BASELINE = DOCS / "s6_sweep_v41_full184_20260729_105013.json"
OLD_SEED = DOCS / "s6_sweep_v41.json"
ANCHOR = (8.0, 2.6, 0.0, 5.0)
REPRO_TOLERANCE_N = 0.15
LEDGER_TOLERANCE_N = 1.0e-9
CHECKPOINT_SCHEMA_VERSION = 2
EXPECTED_CONTRIBUTION_NODES = frozenset(("N1", "N2", "N3", "N4", "N6", "R0"))
EXPECTED_CONTRIBUTION_INVENTORY = {
    "N1": (
        ("uvlm", "physics"),
        ("vortex_impulse", "physics"),
        ("leading_edge_suction", "physics"),
        ("uvlm_remainder", "physics"),
    ),
    "N2": (
        ("separation", "physics"),
        ("profile_drag", "physics"),
    ),
    "N3": (
        ("ds_vortex", "physics"),
        ("vortex_normal", "physics"),
    ),
    "N4": (("ct_consistency", "diagnostic"),),
    "N6": (("rig_drag", "necessary_physics"),),
    "R0": (("numerical_cycle_reduction", "diagnostic"),),
}
RUNTIME_ENVIRONMENT_NAMES = (
    "FLUXV_DTYPE",
    "FLUXV_DEVICE",
    "CUDA_VISIBLE_DEVICES",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
CONDITIONS = CONDITIONS_BY_EVIDENCE_SCOPE[EVIDENCE_CONFIRMED]
EXPECTED_KEYS = {condition_key(condition) for condition in CONDITIONS}
CONDITION_BY_KEY = {
    condition_key(condition): condition
    for condition in CONDITIONS
}
RUN_LOCK = DIAG / "fig171819_v41_confirmed151_fresh.lock"
if len(CONDITIONS) != 151 or len(EXPECTED_KEYS) != 151:
    raise AssertionError("fresh confirmed condition contract drift")


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
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
        prefix=f".{path.name}.",
        suffix=".partial",
        dir=path.parent,
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


@contextmanager
def _exclusive_lock() -> Iterator[None]:
    RUN_LOCK.parent.mkdir(parents=True, exist_ok=True)
    with RUN_LOCK.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("another fresh151 campaign holds the lock") from exc
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _tracked_control_paths() -> tuple[Path, ...]:
    return (
        Path(__file__).resolve(),
        PREREG.resolve(),
        INTEGRITY_ADDENDUM.resolve(),
        LAUNCH_AUTHORIZATION.resolve(),
        (PLATFORM / "run_claim_witnesses.py").resolve(),
        (PLATFORM / "lb_sweep184.py").resolve(),
        (PLATFORM / "fig171819_benchmark.py").resolve(),
        (PLATFORM / "lb_sweep118.py").resolve(),
        OLD_MERGED_BASELINE.resolve(),
        OLD_SEED.resolve(),
        (DOCS / "data.md").resolve(),
    )


def _snapshot_control_sources() -> dict[str, str]:
    return {
        str(path.relative_to(ROOT)): witness._sha256_file(path)
        for path in _tracked_control_paths()
    }


def _validate_control_sources(expected: Mapping[str, str]) -> None:
    actual = _snapshot_control_sources()
    if actual != dict(expected):
        changed = sorted(
            key
            for key in set(actual) | set(expected)
            if actual.get(key) != expected.get(key)
        )
        raise RuntimeError(f"fresh151 control source drift: {changed}")


def _validate_solver_sources(
    expected: Mapping[str, str],
    baseline_manifest: Mapping[str, Any],
) -> None:
    actual = witness._validate_solver_sources(baseline_manifest)
    if actual != dict(expected):
        changed = sorted(
            key
            for key in set(actual) | set(expected)
            if actual.get(key) != expected.get(key)
        )
        raise RuntimeError(f"fresh151 solver source-set drift: {changed[:10]}")


def _validate_governed_solver_members(
    baseline_manifest: Mapping[str, Any],
) -> None:
    expected = baseline_manifest.get("governed_source_members")
    if not isinstance(expected, Mapping):
        raise RuntimeError("baseline manifest lacks governed solver members")
    actual = {
        "platform/claim_nodes/*.y*ml": sorted(
            str(path.resolve().relative_to(ROOT))
            for path in ROOT.glob("platform/claim_nodes/*.y*ml")
            if path.is_file()
        ),
        "platform/claim_runtime/**/*.py": sorted(
            str(path.resolve().relative_to(ROOT))
            for path in ROOT.glob("platform/claim_runtime/**/*.py")
            if path.is_file()
        ),
    }
    if not _canonical_equal(expected, actual):
        differences = {
            pattern: {
                "added": sorted(set(actual[pattern]) - set(expected.get(pattern, []))),
                "removed": sorted(set(expected.get(pattern, [])) - set(actual[pattern])),
            }
            for pattern in actual
        }
        raise RuntimeError(
            f"fresh151 governed solver member drift: {differences}"
        )


def _snapshot_loaded_project_sources() -> dict[str, str]:
    paths: set[Path] = set()
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
    return {
        str(path.relative_to(ROOT)): witness._sha256_file(path)
        for path in sorted(paths)
    }


def _validate_loaded_source_closure(
    solver_sources: Mapping[str, str],
    control_sources: Mapping[str, str],
) -> None:
    expected = {**solver_sources, **control_sources}
    loaded = _snapshot_loaded_project_sources()
    unregistered = sorted(set(loaded) - set(expected))
    mismatched = sorted(
        relative
        for relative, digest in loaded.items()
        if expected.get(relative) != digest
    )
    if unregistered or mismatched:
        raise RuntimeError(
            "fresh151 loaded source closure drift: "
            f"unregistered={unregistered[:10]} mismatched={mismatched[:10]}"
        )


def _validate_execution_sources(
    *,
    control_sources: Mapping[str, str],
    solver_sources: Mapping[str, str],
    baseline_manifest: Mapping[str, Any],
) -> None:
    _validate_control_sources(control_sources)
    _validate_solver_sources(solver_sources, baseline_manifest)
    _validate_governed_solver_members(baseline_manifest)
    _validate_loaded_source_closure(solver_sources, control_sources)


def _canonical_equal(left: Any, right: Any) -> bool:
    try:
        return witness._canonical_hash(left) == witness._canonical_hash(right)
    except (TypeError, ValueError):
        return False


def _finite_float(value: Any) -> float | None:
    if isinstance(value, (bool, str)) or not isinstance(
        value,
        (int, float, np.integer, np.floating),
    ):
        return None
    try:
        normalized = float(value)
    except (TypeError, ValueError):
        return None
    return normalized if math.isfinite(normalized) else None


def _valid_guards_strict(guards: Any) -> bool:
    names = (
        "force_ledger",
        "unclassified_force",
        "unclassified_physical_force",
        "cycle_reduction",
        "aero_output_invariance",
    )
    if not isinstance(guards, Mapping) or set(guards) != set(names):
        return False
    for name in names:
        guard = guards.get(name)
        if not isinstance(guard, Mapping) or guard.get("passed") is not True:
            return False
        error = _finite_float(guard.get("max_abs_error_N"))
        tolerance = _finite_float(guard.get("tolerance_N"))
        if (
            error is None
            or tolerance is None
            or error < 0.0
            or tolerance < 0.0
            or error > tolerance
        ):
            return False
        if "body_force_N" in guard:
            body_force = guard["body_force_N"]
            if (
                not isinstance(body_force, (list, tuple))
                or len(body_force) != 3
                or any(_finite_float(component) is None for component in body_force)
            ):
                return False
    return True


def _condition_record(
    condition: tuple[float, float, float, float],
) -> dict[str, float]:
    U, frequency, twist, aoa = condition
    return {
        "U_m_s": U,
        "frequency_Hz": frequency,
        "nominal_twist_deg": twist,
        "solver_twist_amplitude_deg": twist / 2.0,
        "aoa_deg": aoa,
    }


def _claim_manifest_static_payload(manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in manifest.items()
        if key != "guards"
    }


def _case_claim_manifest_hash(
    common_claim_manifest: Mapping[str, Any],
    guards: Mapping[str, Any],
) -> str:
    case_manifest = dict(common_claim_manifest)
    case_manifest["guards"] = dict(guards)
    return witness._canonical_hash(case_manifest)


def _runtime_identity(wp_module: Any) -> dict[str, Any]:
    try:
        from fluxvortex.warp_fsi import config as cfg

        solver_device = str(cfg.DEVICE)
        device = wp_module.get_device(solver_device)
    except Exception as exc:  # pragma: no cover - defensive runtime evidence
        raise RuntimeError(
            f"cannot resolve solver dtype/device identity: {exc}"
        ) from exc
    return {
        "python_executable": str(Path(sys.executable).resolve()),
        "python_version": sys.version,
        "python_implementation": py_platform.python_implementation(),
        "platform": py_platform.platform(),
        "numpy_version": np.__version__,
        "warp_version": str(getattr(wp_module, "__version__", "unknown")),
        "solver_config": {
            "dtype_name": cfg.dtype_name(),
            "dtype": str(cfg.DTYPE),
            "numpy_dtype": str(np.dtype(cfg.NP_DTYPE)),
            "device": solver_device,
            "CR_TOL": float(cfg.CR_TOL),
            "NEWTON_TOL": float(cfg.NEWTON_TOL),
            "GEOM_ATOL": float(cfg.GEOM_ATOL),
            "PORT_ATOL": float(cfg.PORT_ATOL),
        },
        "warp_device": {
            "text": str(device),
            "alias": getattr(device, "alias", None),
            "name": getattr(device, "name", None),
            "arch": getattr(device, "arch", None),
        },
        "environment": {
            name: os.environ.get(name)
            for name in RUNTIME_ENVIRONMENT_NAMES
        },
    }


def _resolved_call_contract(
    gpu_run_twist: Any,
) -> tuple[dict[str, dict[str, Any]], str]:
    calls = {
        key: witness._resolved_call(gpu_run_twist, CONDITION_BY_KEY[key])
        for key in sorted(EXPECTED_KEYS)
    }
    return calls, witness._canonical_hash(calls)


def _validate_runtime_and_call_contract(
    manifest: Mapping[str, Any],
    *,
    runtime_identity: Mapping[str, Any],
    call_contract_sha256: str,
) -> None:
    if manifest.get("resolved_call_contract_sha256") != call_contract_sha256:
        raise RuntimeError("fresh151 resume resolved-call contract drift")
    if not _canonical_equal(manifest.get("runtime_identity"), runtime_identity):
        raise RuntimeError("fresh151 resume runtime identity drift")


def _case_validation_error(
    *,
    key: str,
    condition: tuple[float, float, float, float],
    result: Any,
    evidence: Any,
    manifest_guard: Any,
    expected_call: Mapping[str, Any],
    graph_identity: str | None,
    common_claim_manifest: Any,
    old_baseline: Mapping[str, Any],
) -> str | None:
    if not isinstance(result, Mapping) or set(result) != {"L", "T"}:
        return "result must contain exactly finite L/T"
    lift = _finite_float(result.get("L"))
    thrust = _finite_float(result.get("T"))
    if lift is None or thrust is None:
        return "result contains non-finite L/T"
    if not isinstance(evidence, Mapping):
        return "case evidence is not a mapping"
    if key != condition_key(condition):
        return "outer condition key does not match expected condition"
    if evidence.get("condition_key") != key:
        return "evidence condition_key mismatch"
    if not _canonical_equal(evidence.get("condition"), _condition_record(condition)):
        return "evidence condition mismatch"
    if not _canonical_equal(evidence.get("resolved_call"), expected_call):
        return "resolved-call contract mismatch"

    reference = old_baseline.get(key)
    if not isinstance(reference, Mapping):
        return "old diagnostic baseline missing"
    try:
        expected_reference = {
            "L_N": float(reference["L"]),
            "T_N": float(reference["T"]),
        }
    except (KeyError, TypeError, ValueError):
        return "old diagnostic baseline malformed"
    if not _canonical_equal(evidence.get("old_baseline"), expected_reference):
        return "old-baseline evidence mismatch"
    expected_delta = {
        "L_N": lift - expected_reference["L_N"],
        "T_N": thrust - expected_reference["T_N"],
    }
    if not _canonical_equal(
        evidence.get("signed_old_baseline_delta_N"),
        expected_delta,
    ):
        return "old-baseline delta mismatch"

    guards = evidence.get("claim_guards")
    if not _valid_guards_strict(guards):
        return "claim guards missing or failed"
    if not _canonical_equal(guards, manifest_guard):
        return "manifest/evidence guard mismatch"
    if not isinstance(common_claim_manifest, Mapping):
        return "common claim manifest missing"
    try:
        manifest_identity = witness._claim_graph_identity_sha256(
            common_claim_manifest
        )
    except (KeyError, RuntimeError, TypeError, ValueError) as exc:
        return f"common claim manifest malformed: {exc}"
    if not isinstance(graph_identity, str) or manifest_identity != graph_identity:
        return "common claim graph identity mismatch"
    if evidence.get("claim_graph_identity_sha256") != graph_identity:
        return "case claim graph identity mismatch"
    if evidence.get("claim_manifest_sha256") != _case_claim_manifest_hash(
        common_claim_manifest,
        guards,
    ):
        return "case claim manifest hash mismatch"

    contributions = evidence.get("claim_contributions")
    if not isinstance(contributions, Mapping):
        return "raw claim contributions missing"
    if set(contributions) != EXPECTED_CONTRIBUTION_NODES:
        return "raw claim contribution node set mismatch"
    if any(
        not isinstance(contributions[node_id], list)
        or not contributions[node_id]
        for node_id in EXPECTED_CONTRIBUTION_NODES
    ):
        return "raw claim contribution list missing or empty"
    for node_id, expected_inventory in EXPECTED_CONTRIBUTION_INVENTORY.items():
        for item in contributions[node_id]:
            if not isinstance(item, Mapping) or set(item) != {
                "body_force",
                "channel",
                "metadata",
                "role",
            }:
                return f"{node_id}: contribution record schema mismatch"
            body_force = item.get("body_force")
            if (
                not isinstance(body_force, (list, tuple))
                or len(body_force) != 3
                or any(_finite_float(component) is None for component in body_force)
            ):
                return f"{node_id}: contribution body force is not strict finite 3-vector"
            if not isinstance(item.get("metadata"), Mapping):
                return f"{node_id}: contribution metadata malformed"
        actual_inventory = tuple(
            (item.get("channel"), item.get("role"))
            if isinstance(item, Mapping)
            else (None, None)
            for item in contributions[node_id]
        )
        if actual_inventory != expected_inventory:
            return f"{node_id}: contribution channel/role inventory mismatch"
    try:
        summary, ledger_total = witness._contribution_summary(
            contributions,
            aoa_deg=condition[3],
        )
    except (RuntimeError, TypeError, ValueError) as exc:
        return f"raw claim contribution malformed: {exc}"
    if set(summary) != EXPECTED_CONTRIBUTION_NODES:
        return "recomputed contribution summary node set mismatch"
    if not _canonical_equal(evidence.get("contribution_summary"), summary):
        return "stored contribution summary mismatch"

    wind_total = witness._wind_force(ledger_total, condition[3])
    wind_error = max(
        abs(wind_total["L_N"] - lift),
        abs(wind_total["T_N"] - thrust),
    )
    angle = math.radians(condition[3])
    target_body = np.asarray(
        [
            -thrust * math.cos(angle) - lift * math.sin(angle),
            0.0,
            lift * math.cos(angle) - thrust * math.sin(angle),
        ],
        dtype=float,
    )
    body_error = float(np.max(np.abs(ledger_total - target_body)))
    if (
        not math.isfinite(body_error)
        or not math.isfinite(wind_error)
        or body_error > LEDGER_TOLERANCE_N
        or wind_error > LEDGER_TOLERANCE_N
    ):
        return (
            "raw contribution ledger does not close to result "
            f"(body={body_error}, wind={wind_error})"
        )

    stored_ledger = evidence.get("recomputed_ledger")
    if not isinstance(stored_ledger, Mapping):
        return "stored recomputed ledger missing"
    if not _canonical_equal(
        stored_ledger.get("total_body_force_N"),
        ledger_total.tolist(),
    ):
        return "stored body ledger total mismatch"
    if not _canonical_equal(
        stored_ledger.get("total_wind_force"),
        wind_total,
    ):
        return "stored wind ledger total mismatch"
    for name in ("max_body_error_N", "max_wind_error_N"):
        stored_error = _finite_float(stored_ledger.get(name))
        if stored_error is None or stored_error > LEDGER_TOLERANCE_N:
            return f"stored ledger diagnostic {name} invalid"
    wall_s = _finite_float(evidence.get("wall_s"))
    if wall_s is None or wall_s < 0.0:
        return "case wall time invalid"
    return None


def _valid_saved_case(
    *,
    key: str,
    condition: tuple[float, float, float, float],
    result: Any,
    evidence: Any,
    manifest_guard: Any,
    expected_call: Mapping[str, Any],
    graph_identity: str | None,
    common_claim_manifest: Any,
    old_baseline: Mapping[str, Any],
) -> bool:
    return (
        _case_validation_error(
            key=key,
            condition=condition,
            result=result,
            evidence=evidence,
            manifest_guard=manifest_guard,
            expected_call=expected_call,
            graph_identity=graph_identity,
            common_claim_manifest=common_claim_manifest,
            old_baseline=old_baseline,
        )
        is None
    )


def _run_case(
    gpu_run_twist: Any,
    condition: tuple[float, float, float, float],
    old_baseline: Mapping[str, Any],
) -> tuple[dict[str, float], dict[str, Any], dict[str, Any]]:
    U, frequency, twist, aoa = condition
    key = condition_key(condition)
    started = time.time()
    result = gpu_run_twist(
        U=U,
        aoa_deg=aoa,
        freq=frequency,
        twist_amp_deg=twist / 2.0,
        nc=12,
        ns=16,
        n_cycle=4,
        steps_per_cycle=spc_of(U, frequency),
        wake_rows=spc_of(U, frequency),
        closure="v41",
        **BASE,
    )
    value = {
        "L": float(result["L_wind"]),
        "T": float(result["T_wind"]),
    }
    if not all(math.isfinite(item) for item in value.values()):
        raise RuntimeError(f"{key}: non-finite force")
    guards = result.get("claim_guards")
    manifest = result.get("claim_manifest")
    contributions = result.get("claim_contributions")
    if (
        not _valid_guards_strict(guards)
        or not isinstance(manifest, Mapping)
        or not isinstance(contributions, Mapping)
    ):
        raise RuntimeError(f"{key}: incomplete claim graph evidence")
    summary, ledger_total = witness._contribution_summary(
        contributions,
        aoa_deg=aoa,
    )
    target = np.asarray(
        [float(result["Fx_body"]), 0.0, float(result["Fz_body"])],
        dtype=float,
    )
    body_error = float(np.max(np.abs(ledger_total - target)))
    wind_total = witness._wind_force(ledger_total, aoa)
    wind_error = max(
        abs(wind_total["L_N"] - value["L"]),
        abs(wind_total["T_N"] - value["T"]),
    )
    if body_error > LEDGER_TOLERANCE_N or wind_error > LEDGER_TOLERANCE_N:
        raise RuntimeError(
            f"{key}: claim ledger mismatch body={body_error} wind={wind_error}"
        )
    reference = old_baseline.get(key)
    if not isinstance(reference, Mapping):
        raise RuntimeError(f"{key}: missing old diagnostic baseline")
    evidence = {
        "condition_key": key,
        "condition": _condition_record(condition),
        "resolved_call": witness._resolved_call(gpu_run_twist, condition),
        "old_baseline": {
            "L_N": float(reference["L"]),
            "T_N": float(reference["T"]),
        },
        "signed_old_baseline_delta_N": {
            "L_N": value["L"] - float(reference["L"]),
            "T_N": value["T"] - float(reference["T"]),
        },
        "claim_graph_identity_sha256": (
            witness._claim_graph_identity_sha256(manifest)
        ),
        "claim_manifest_sha256": witness._canonical_hash(manifest),
        "claim_guards": dict(guards),
        "claim_contributions": dict(contributions),
        "contribution_summary": summary,
        "recomputed_ledger": {
            "total_body_force_N": ledger_total.tolist(),
            "total_wind_force": wind_total,
            "max_body_error_N": body_error,
            "max_wind_error_N": wind_error,
        },
        "wall_s": time.time() - started,
    }
    return value, evidence, dict(manifest)


def _paths(timestamp: str) -> tuple[Path, Path, Path]:
    result = DOCS / f"s6_sweep_v41_confirmed151_fresh_{timestamp}.json"
    manifest = (
        DIAG / f"fig171819_v41_confirmed151_fresh_manifest_{timestamp}.json"
    )
    contributions = (
        DIAG / f"fig171819_v41_confirmed151_contributions_{timestamp}.json"
    )
    return result, manifest, contributions


def _drift_summary(
    results: Mapping[str, Any],
    old_baseline: Mapping[str, Any],
    old_seed: Mapping[str, Any],
) -> dict[str, Any]:
    summaries: dict[str, Any] = {}
    groups = {
        "all_confirmed151": set(results),
        "old_seed_confirmed85": set(results) & set(old_seed),
        "former_new66": set(results) - set(old_seed),
    }
    for name, keys in groups.items():
        l_delta = np.asarray(
            [float(results[key]["L"]) - float(old_baseline[key]["L"]) for key in keys]
        )
        t_delta = np.asarray(
            [float(results[key]["T"]) - float(old_baseline[key]["T"]) for key in keys]
        )
        summaries[name] = {
            "n_conditions": len(keys),
            "L": {
                "mean_signed_N": float(np.mean(l_delta)),
                "max_absolute_N": float(np.max(np.abs(l_delta))),
                "n_over_0.15N": int(np.sum(np.abs(l_delta) > REPRO_TOLERANCE_N)),
            },
            "T": {
                "mean_signed_N": float(np.mean(t_delta)),
                "max_absolute_N": float(np.max(np.abs(t_delta))),
                "n_over_0.15N": int(np.sum(np.abs(t_delta) > REPRO_TOLERANCE_N)),
            },
        }
    return summaries


def _checkpoint(
    *,
    result_path: Path,
    manifest_path: Path,
    contributions_path: Path,
    results: Mapping[str, Any],
    manifest: dict[str, Any],
    contribution_file: dict[str, Any],
    case_evidence: Mapping[str, Any],
) -> None:
    contribution_file["cases"] = dict(case_evidence)
    manifest["completed_condition_count"] = len(results)
    _write_json_atomic(result_path, dict(results))
    _write_json_atomic(contributions_path, contribution_file)
    _write_json_atomic(manifest_path, manifest)


def _validate_resume_containers(
    *,
    timestamp: str,
    result_path: Path,
    manifest_path: Path,
    contributions_path: Path,
    results: Mapping[str, Any],
    manifest: Mapping[str, Any],
    contribution_file: Mapping[str, Any],
    solver_sources: Mapping[str, str],
    control_sources: Mapping[str, str],
) -> None:
    if manifest.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise RuntimeError("fresh151 resume manifest schema mismatch")
    if contribution_file.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise RuntimeError("fresh151 resume contribution schema mismatch")
    if manifest.get("run_id") != timestamp or contribution_file.get("run_id") != timestamp:
        raise RuntimeError("fresh151 resume run-id mismatch")
    expected_paths = {
        "result_path": str(result_path.relative_to(ROOT)),
        "manifest_path": str(manifest_path.relative_to(ROOT)),
        "contributions_path": str(contributions_path.relative_to(ROOT)),
    }
    for name, expected in expected_paths.items():
        if manifest.get(name) != expected:
            raise RuntimeError(f"fresh151 resume manifest {name} mismatch")
        if contribution_file.get(name) != expected:
            raise RuntimeError(f"fresh151 resume contribution {name} mismatch")
    if manifest.get("status") not in {"running", "failed", "complete"}:
        raise RuntimeError("fresh151 resume status malformed")
    if (
        manifest.get("expected_condition_keys") != sorted(EXPECTED_KEYS)
        or manifest.get("expected_condition_count") != len(EXPECTED_KEYS)
        or manifest.get("solver_source_hashes") != dict(solver_sources)
        or manifest.get("control_source_hashes") != dict(control_sources)
    ):
        raise RuntimeError("fresh151 resume contract drift")
    if not isinstance(results, Mapping):
        raise RuntimeError("fresh151 result checkpoint malformed")
    for field in (
        "case_guards",
        "failures",
    ):
        if not isinstance(manifest.get(field), dict):
            raise RuntimeError(f"fresh151 resume manifest {field} malformed")
    if not isinstance(manifest.get("runtime_sessions"), list):
        raise RuntimeError("fresh151 resume runtime session log malformed")
    if not isinstance(contribution_file.get("cases"), dict):
        raise RuntimeError("fresh151 contribution checkpoint malformed")


def _sanitize_resume_checkpoint(
    *,
    results: dict[str, Any],
    case_evidence: dict[str, Any],
    case_guards: dict[str, Any],
    expected_calls: Mapping[str, Mapping[str, Any]],
    graph_identity: str | None,
    common_claim_manifest: Any,
    old_baseline: Mapping[str, Any],
) -> dict[str, str]:
    discarded: dict[str, str] = {}
    all_keys = set(results) | set(case_evidence) | set(case_guards)
    for key in sorted(all_keys):
        if key not in EXPECTED_KEYS:
            error = "unexpected condition key"
        elif (
            key not in results
            or key not in case_evidence
            or key not in case_guards
        ):
            error = "result/evidence/guard orphan"
        else:
            error = _case_validation_error(
                key=key,
                condition=CONDITION_BY_KEY[key],
                result=results[key],
                evidence=case_evidence[key],
                manifest_guard=case_guards[key],
                expected_call=expected_calls[key],
                graph_identity=graph_identity,
                common_claim_manifest=common_claim_manifest,
                old_baseline=old_baseline,
            )
        if error is not None:
            results.pop(key, None)
            case_evidence.pop(key, None)
            case_guards.pop(key, None)
            discarded[key] = error
    return discarded


def _validate_complete_checkpoint(
    *,
    results: Mapping[str, Any],
    case_evidence: Mapping[str, Any],
    case_guards: Mapping[str, Any],
    expected_calls: Mapping[str, Mapping[str, Any]],
    graph_identity: str | None,
    common_claim_manifest: Any,
    old_baseline: Mapping[str, Any],
) -> None:
    key_sets = (set(results), set(case_evidence), set(case_guards))
    if any(keys != EXPECTED_KEYS for keys in key_sets):
        raise RuntimeError(
            "fresh151 completion three-file key-set mismatch: "
            f"result={len(key_sets[0])} evidence={len(key_sets[1])} "
            f"guards={len(key_sets[2])}"
        )
    invalid: dict[str, str] = {}
    for key in sorted(EXPECTED_KEYS):
        error = _case_validation_error(
            key=key,
            condition=CONDITION_BY_KEY[key],
            result=results[key],
            evidence=case_evidence[key],
            manifest_guard=case_guards[key],
            expected_call=expected_calls[key],
            graph_identity=graph_identity,
            common_claim_manifest=common_claim_manifest,
            old_baseline=old_baseline,
        )
        if error is not None:
            invalid[key] = error
    if invalid:
        preview = list(invalid.items())[:5]
        raise RuntimeError(f"fresh151 completion evidence invalid: {preview}")


def _record_anchor_failure(
    *,
    manifest: dict[str, Any],
    error: BaseException,
    cold_value: Any,
    cold_evidence: Any,
    warm_value: Any,
    warm_evidence: Any,
) -> None:
    manifest["status"] = "failed"
    manifest.setdefault("failures", {})["__session_anchor__"] = (
        f"{type(error).__name__}: {error}"
    )
    manifest["anchor_nogo"] = {
        "at": datetime.now().astimezone().isoformat(),
        "cold_value": cold_value,
        "cold_evidence": cold_evidence,
        "warm_value": warm_value,
        "warm_evidence": warm_evidence,
    }
    manifest["updated_at"] = datetime.now().astimezone().isoformat()


def _run_case_with_source_guards(
    *,
    gpu_run_twist: Any,
    condition: tuple[float, float, float, float],
    old_baseline: Mapping[str, Any],
    control_sources: Mapping[str, str],
    solver_sources: Mapping[str, str],
    baseline_manifest: Mapping[str, Any],
) -> tuple[dict[str, float], dict[str, Any], dict[str, Any]]:
    _validate_execution_sources(
        control_sources=control_sources,
        solver_sources=solver_sources,
        baseline_manifest=baseline_manifest,
    )
    try:
        return _run_case(gpu_run_twist, condition, old_baseline)
    finally:
        _validate_execution_sources(
            control_sources=control_sources,
            solver_sources=solver_sources,
            baseline_manifest=baseline_manifest,
        )


def _run(
    *,
    timestamp: str,
    resume: bool,
) -> int:
    result_path, manifest_path, contributions_path = _paths(timestamp)
    if (
        witness._sha256_file(OLD_MERGED_BASELINE)
        != witness.BASELINE_RESULT_SHA256
    ):
        raise RuntimeError("old merged baseline hash drift")
    if (
        witness._sha256_file(witness.BASELINE_MANIFEST)
        != witness.BASELINE_MANIFEST_SHA256
    ):
        raise RuntimeError("old baseline manifest hash drift")
    existing = [
        path.exists()
        for path in (result_path, manifest_path, contributions_path)
    ]
    if any(existing) and not all(existing):
        raise RuntimeError("partial fresh151 artifact set; manual audit required")
    if any(existing) and not resume:
        raise FileExistsError(
            "fresh151 artifacts exist; use a new timestamp or explicit --resume"
        )

    old_baseline = _load_json(OLD_MERGED_BASELINE)
    old_seed = _load_json(OLD_SEED)
    baseline_manifest = witness._load_json(witness.BASELINE_MANIFEST)
    solver_sources = witness._validate_solver_sources(baseline_manifest)
    control_sources = _snapshot_control_sources()

    if resume:
        results = _load_json(result_path)
        manifest = _load_json(manifest_path)
        contribution_file = _load_json(contributions_path)
        case_evidence = contribution_file.get("cases")
        _validate_resume_containers(
            timestamp=timestamp,
            result_path=result_path,
            manifest_path=manifest_path,
            contributions_path=contributions_path,
            results=results,
            manifest=manifest,
            contribution_file=contribution_file,
            solver_sources=solver_sources,
            control_sources=control_sources,
        )
        if not isinstance(case_evidence, dict):  # narrowed by validator
            raise AssertionError("unreachable malformed case evidence")
    else:
        results: dict[str, Any] = {}
        case_evidence: dict[str, Any] = {}
        manifest = {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "status": "running",
            "run_id": timestamp,
            "started_at": datetime.now().astimezone().isoformat(),
            "preregistration": {
                "path": str(PREREG.relative_to(ROOT)),
                "sha256": witness._sha256_file(PREREG),
            },
            "integrity_addendum": {
                "path": str(INTEGRITY_ADDENDUM.relative_to(ROOT)),
                "sha256": witness._sha256_file(INTEGRITY_ADDENDUM),
            },
            "launch_authorization": {
                "path": str(LAUNCH_AUTHORIZATION.relative_to(ROOT)),
                "sha256": witness._sha256_file(LAUNCH_AUTHORIZATION),
            },
            "runner": {
                "path": str(Path(__file__).resolve().relative_to(ROOT)),
                "sha256": witness._sha256_file(Path(__file__).resolve()),
            },
            "solver_source_hashes": solver_sources,
            "control_source_hashes": control_sources,
            "expected_condition_keys": sorted(EXPECTED_KEYS),
            "expected_condition_count": 151,
            "resolved_call_contract_sha256": None,
            "runtime_identity": None,
            "cold_preconditioner": None,
            "formal_anchor": None,
            "runtime_sessions": [],
            "claim_graph_identity_sha256": None,
            "common_claim_manifest": None,
            "case_guards": {},
            "failures": {},
            "completed_condition_count": 0,
            "result_path": str(result_path.relative_to(ROOT)),
            "manifest_path": str(manifest_path.relative_to(ROOT)),
            "contributions_path": str(contributions_path.relative_to(ROOT)),
        }
        contribution_file = {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "run_id": timestamp,
            "result_path": str(result_path.relative_to(ROOT)),
            "manifest_path": str(manifest_path.relative_to(ROOT)),
            "contributions_path": str(contributions_path.relative_to(ROOT)),
            "cases": case_evidence,
        }

    import warp as wp

    wp.init()
    from _v2_robo import gpu_run_twist

    _validate_execution_sources(
        control_sources=control_sources,
        solver_sources=solver_sources,
        baseline_manifest=baseline_manifest,
    )
    expected_calls, call_contract_sha256 = _resolved_call_contract(gpu_run_twist)
    runtime_identity = _runtime_identity(wp)
    if resume:
        _validate_runtime_and_call_contract(
            manifest,
            runtime_identity=runtime_identity,
            call_contract_sha256=call_contract_sha256,
        )
    else:
        manifest["resolved_call_contract_sha256"] = call_contract_sha256
        manifest["runtime_identity"] = runtime_identity
        _checkpoint(
            result_path=result_path,
            manifest_path=manifest_path,
            contributions_path=contributions_path,
            results=results,
            manifest=manifest,
            contribution_file=contribution_file,
            case_evidence=case_evidence,
        )

    graph_identity = manifest.get("claim_graph_identity_sha256")
    common_claim_manifest = manifest.get("common_claim_manifest")
    case_guards = manifest["case_guards"]
    if manifest.get("status") == "complete":
        if (
            manifest.get("result_sha256") != witness._sha256_file(result_path)
            or manifest.get("contributions_sha256")
            != witness._sha256_file(contributions_path)
        ):
            raise RuntimeError("fresh151 complete artifact hash mismatch")
        _validate_complete_checkpoint(
            results=results,
            case_evidence=case_evidence,
            case_guards=case_guards,
            expected_calls=expected_calls,
            graph_identity=graph_identity,
            common_claim_manifest=common_claim_manifest,
            old_baseline=old_baseline,
        )
        return 0

    discarded = _sanitize_resume_checkpoint(
        results=results,
        case_evidence=case_evidence,
        case_guards=case_guards,
        expected_calls=expected_calls,
        graph_identity=graph_identity,
        common_claim_manifest=common_claim_manifest,
        old_baseline=old_baseline,
    )
    if discarded:
        manifest["status"] = "running"
        if condition_key(ANCHOR) in discarded:
            manifest["formal_anchor"] = None
        manifest.setdefault("resume_discarded", []).append(
            {
                "at": datetime.now().astimezone().isoformat(),
                "cases": discarded,
            }
        )
        _checkpoint(
            result_path=result_path,
            manifest_path=manifest_path,
            contributions_path=contributions_path,
            results=results,
            manifest=manifest,
            contribution_file=contribution_file,
            case_evidence=case_evidence,
        )

    anchor_key = condition_key(ANCHOR)
    cold_value: Any = None
    cold_evidence: Any = None
    value: Any = None
    evidence: Any = None
    claim_manifest: Any = None
    try:
        cold_value, cold_evidence, _ = _run_case_with_source_guards(
            gpu_run_twist=gpu_run_twist,
            condition=ANCHOR,
            old_baseline=old_baseline,
            control_sources=control_sources,
            solver_sources=solver_sources,
            baseline_manifest=baseline_manifest,
        )
        value, evidence, claim_manifest = _run_case_with_source_guards(
            gpu_run_twist=gpu_run_twist,
            condition=ANCHOR,
            old_baseline=old_baseline,
            control_sources=control_sources,
            solver_sources=solver_sources,
            baseline_manifest=baseline_manifest,
        )
        warm_anchor_max_delta = max(
            abs(delta)
            for delta in evidence["signed_old_baseline_delta_N"].values()
        )
        if warm_anchor_max_delta > REPRO_TOLERANCE_N:
            raise RuntimeError(
                "fresh151 warm anchor failed frozen 0.15 N identity gate"
            )
        session_identity = evidence["claim_graph_identity_sha256"]
        if cold_evidence["claim_graph_identity_sha256"] != session_identity:
            raise RuntimeError("fresh151 cold/warm graph identity drift")
        if graph_identity is not None and session_identity != graph_identity:
            raise RuntimeError("fresh151 resume graph identity drift")
        if (
            common_claim_manifest is not None
            and not _canonical_equal(
                _claim_manifest_static_payload(common_claim_manifest),
                _claim_manifest_static_payload(claim_manifest),
            )
        ):
            raise RuntimeError("fresh151 resume common claim manifest drift")
    except Exception as exc:
        _record_anchor_failure(
            manifest=manifest,
            error=exc,
            cold_value=cold_value,
            cold_evidence=cold_evidence,
            warm_value=value,
            warm_evidence=evidence,
        )
        _checkpoint(
            result_path=result_path,
            manifest_path=manifest_path,
            contributions_path=contributions_path,
            results=results,
            manifest=manifest,
            contribution_file=contribution_file,
            case_evidence=case_evidence,
        )
        raise

    session_identity = evidence["claim_graph_identity_sha256"]
    graph_identity = session_identity
    common_claim_manifest = claim_manifest
    manifest["claim_graph_identity_sha256"] = graph_identity
    manifest["common_claim_manifest"] = claim_manifest
    manifest["cold_preconditioner"] = {
        "value": cold_value,
        "claim_graph_identity_sha256": cold_evidence[
            "claim_graph_identity_sha256"
        ],
        "guards": cold_evidence["claim_guards"],
        "discarded_from_results": True,
    }
    manifest.setdefault("runtime_sessions", []).append(
        {
            "started_at": datetime.now().astimezone().isoformat(),
            "cold_preconditioner": {
                "value": cold_value,
                "graph_identity": cold_evidence[
                    "claim_graph_identity_sha256"
                ],
            },
            "warm_anchor": {
                "value": value,
                "old_baseline_delta_N": evidence[
                    "signed_old_baseline_delta_N"
                ],
                "graph_identity": graph_identity,
            },
        }
    )
    if anchor_key not in results:
        manifest["formal_anchor"] = {
            "condition_key": anchor_key,
            "value": value,
            "old_baseline_delta_N": evidence[
                "signed_old_baseline_delta_N"
            ],
            "guards": evidence["claim_guards"],
        }
        results[anchor_key] = value
        case_evidence[anchor_key] = evidence
        case_guards[anchor_key] = evidence["claim_guards"]
    manifest["status"] = "running"
    manifest["failures"].pop("__session_anchor__", None)
    anchor_error = _case_validation_error(
        key=anchor_key,
        condition=ANCHOR,
        result=results[anchor_key],
        evidence=case_evidence[anchor_key],
        manifest_guard=case_guards[anchor_key],
        expected_call=expected_calls[anchor_key],
        graph_identity=graph_identity,
        common_claim_manifest=common_claim_manifest,
        old_baseline=old_baseline,
    )
    if anchor_error is not None:
        error = RuntimeError(f"fresh151 formal anchor evidence invalid: {anchor_error}")
        _record_anchor_failure(
            manifest=manifest,
            error=error,
            cold_value=cold_value,
            cold_evidence=cold_evidence,
            warm_value=value,
            warm_evidence=evidence,
        )
        _checkpoint(
            result_path=result_path,
            manifest_path=manifest_path,
            contributions_path=contributions_path,
            results=results,
            manifest=manifest,
            contribution_file=contribution_file,
            case_evidence=case_evidence,
        )
        raise error
    _checkpoint(
        result_path=result_path,
        manifest_path=manifest_path,
        contributions_path=contributions_path,
        results=results,
        manifest=manifest,
        contribution_file=contribution_file,
        case_evidence=case_evidence,
    )

    ordered = sorted(
        CONDITIONS,
        key=lambda condition: tuple(float(value) for value in condition),
    )
    run_started = time.time()
    for index, condition in enumerate(ordered, start=1):
        key = condition_key(condition)
        if key in results:
            continue
        try:
            value, evidence, claim_manifest = _run_case_with_source_guards(
                gpu_run_twist=gpu_run_twist,
                condition=condition,
                old_baseline=old_baseline,
                control_sources=control_sources,
                solver_sources=solver_sources,
                baseline_manifest=baseline_manifest,
            )
            if evidence["claim_graph_identity_sha256"] != graph_identity:
                raise RuntimeError(f"{key}: claim graph identity drift")
            if (
                witness._claim_graph_identity_sha256(claim_manifest)
                != graph_identity
            ):
                raise RuntimeError(f"{key}: malformed claim manifest identity")
            if not _canonical_equal(
                _claim_manifest_static_payload(claim_manifest),
                _claim_manifest_static_payload(common_claim_manifest),
            ):
                raise RuntimeError(f"{key}: common claim manifest drift")
            if not _canonical_equal(
                claim_manifest.get("guards"),
                evidence["claim_guards"],
            ):
                raise RuntimeError(f"{key}: claim manifest guard mismatch")
            validation_error = _case_validation_error(
                key=key,
                condition=condition,
                result=value,
                evidence=evidence,
                manifest_guard=evidence["claim_guards"],
                expected_call=expected_calls[key],
                graph_identity=graph_identity,
                common_claim_manifest=common_claim_manifest,
                old_baseline=old_baseline,
            )
            if validation_error is not None:
                raise RuntimeError(
                    f"{key}: fresh case evidence invalid: {validation_error}"
                )
            results[key] = value
            case_evidence[key] = evidence
            case_guards[key] = evidence["claim_guards"]
            manifest["failures"].pop(key, None)
            print(
                f"[v41-fresh151] {len(results)}/151 {key}: "
                f"L={value['L']:+.3f} T={value['T']:+.3f} "
                f"({evidence['wall_s']:.0f}s)",
                flush=True,
            )
        except Exception as exc:
            manifest["status"] = "failed"
            manifest["failures"][key] = f"{type(exc).__name__}: {exc}"
            manifest["updated_at"] = datetime.now().astimezone().isoformat()
            manifest["run_wall_s"] = time.time() - run_started
            _checkpoint(
                result_path=result_path,
                manifest_path=manifest_path,
                contributions_path=contributions_path,
                results=results,
                manifest=manifest,
                contribution_file=contribution_file,
                case_evidence=case_evidence,
            )
            raise
        manifest["updated_at"] = datetime.now().astimezone().isoformat()
        manifest["run_wall_s"] = time.time() - run_started
        _checkpoint(
            result_path=result_path,
            manifest_path=manifest_path,
            contributions_path=contributions_path,
            results=results,
            manifest=manifest,
            contribution_file=contribution_file,
            case_evidence=case_evidence,
        )

    _validate_execution_sources(
        control_sources=control_sources,
        solver_sources=solver_sources,
        baseline_manifest=baseline_manifest,
    )
    _validate_complete_checkpoint(
        results=results,
        case_evidence=case_evidence,
        case_guards=case_guards,
        expected_calls=expected_calls,
        graph_identity=graph_identity,
        common_claim_manifest=common_claim_manifest,
        old_baseline=old_baseline,
    )
    scoped_coverage = coverage(
        results,
        evidence_scope=EVIDENCE_CONFIRMED,
    )
    if not scoped_coverage["complete"]:
        raise RuntimeError(f"fresh151 confirmed coverage incomplete: {scoped_coverage}")
    manifest["status"] = "running"
    manifest["final_confirmed_coverage"] = scoped_coverage
    manifest["drift_from_old_merged"] = _drift_summary(
        results,
        old_baseline,
        old_seed,
    )
    manifest["updated_at"] = datetime.now().astimezone().isoformat()
    _checkpoint(
        result_path=result_path,
        manifest_path=manifest_path,
        contributions_path=contributions_path,
        results=results,
        manifest=manifest,
        contribution_file=contribution_file,
        case_evidence=case_evidence,
    )
    manifest["result_sha256"] = witness._sha256_file(result_path)
    manifest["contributions_sha256"] = witness._sha256_file(contributions_path)
    manifest["status"] = "complete"
    manifest["updated_at"] = datetime.now().astimezone().isoformat()
    _write_json_atomic(manifest_path, manifest)

    reopened_results = _load_json(result_path)
    reopened_contributions = _load_json(contributions_path)
    reopened_manifest = _load_json(manifest_path)
    reopened_cases = reopened_contributions.get("cases")
    reopened_guards = reopened_manifest.get("case_guards")
    if not isinstance(reopened_cases, Mapping) or not isinstance(
        reopened_guards, Mapping
    ):
        raise RuntimeError("fresh151 completed artifact reopen malformed")
    _validate_complete_checkpoint(
        results=reopened_results,
        case_evidence=reopened_cases,
        case_guards=reopened_guards,
        expected_calls=expected_calls,
        graph_identity=reopened_manifest.get("claim_graph_identity_sha256"),
        common_claim_manifest=reopened_manifest.get("common_claim_manifest"),
        old_baseline=old_baseline,
    )
    if (
        reopened_manifest.get("status") != "complete"
        or reopened_manifest.get("result_sha256")
        != witness._sha256_file(result_path)
        or reopened_manifest.get("contributions_sha256")
        != witness._sha256_file(contributions_path)
    ):
        raise RuntimeError("fresh151 completed artifact receipt mismatch")
    print(
        f"[v41-fresh151] COMPLETE 151/151 -> {result_path}",
        flush=True,
    )
    return 0


def run(*, timestamp: str, resume: bool = False) -> int:
    with _exclusive_lock():
        return _run(timestamp=timestamp, resume=resume)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timestamp", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if not args.timestamp.replace("_", "").isdigit():
        parser.error("--timestamp must contain only digits and underscores")
    return run(timestamp=args.timestamp, resume=args.resume)


if __name__ == "__main__":
    raise SystemExit(main())
