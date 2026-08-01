"""Run the preregistered six-point V4.1 claim-contribution witness set."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


PLATFORM = Path(__file__).resolve().parent
ROOT = PLATFORM.parent
DOCS = PLATFORM / "docs"
DIAG = DOCS / "diag"
for path in (ROOT, PLATFORM):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from fig171819_benchmark import condition_key  # noqa: E402
from lb_sweep118 import BASE, spc_of  # noqa: E402
from lb_sweep184 import (  # noqa: E402
    _canonical_hash,
    _claim_graph_identity_sha256,
)


BASELINE_RESULT = DOCS / "s6_sweep_v41_full184_20260729_105013.json"
BASELINE_MANIFEST = (
    DIAG / "fig171819_v41_baseline_manifest_20260729_105013.json"
)
PREREG = DIAG / "v41_confirmed42_primary_thrust_witness_prereg_20260729.md"
BASELINE_RESULT_SHA256 = (
    "e8e903b8760f760f2a379032d2d2b3e814ba789f8e3e2fafa6de65ea851b8ea6"
)
BASELINE_MANIFEST_SHA256 = (
    "5f0f02b4346bef03c8bab012088c1a65401fa463040f34caaa7b0b4df859bc82"
)
ANCHOR = (8.0, 2.6, 0.0, 5.0)
REPRO_TOLERANCE_N = 0.15
WITNESSES = (
    (6.0, 2.6, 22.5, 5.0),
    (10.0, 2.6, 22.5, 5.0),
    (10.0, 2.6, 0.0, 5.0),
    (10.0, 2.6, 45.0, 5.0),
    (10.0, 1.4, 22.5, 5.0),
    (8.0, 2.6, 22.5, 5.0),
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"{path}: expected a JSON object")
    return value


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    with partial.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(partial, path)


def _solver_source_paths() -> list[Path]:
    direct = (
        "_v2_robo.py",
        "_v2_repro_nc12.py",
        "_v2_robogeom.py",
        "airfoil_geometry.py",
        "lb_sweep118.py",
        "lb_dyn.py",
        "lb_static.py",
        "cd_table.py",
        "diff_coupled_fsi.py",
        "diff_uvlm_unsteady_gpu.py",
        "diff_uvlm_unsteady.py",
        "diff_coupled_unsteady.py",
        "diff_struct_design.py",
        "diff_vlm.py",
        "flap_flight_validate.py",
    )
    paths = [PLATFORM / name for name in direct]
    missing = [str(path.relative_to(ROOT)) for path in paths if not path.is_file()]
    if missing:
        raise RuntimeError(f"mandatory aerodynamic solver sources missing: {missing}")
    paths.extend((PLATFORM / "claim_runtime").rglob("*.py"))
    paths.extend((PLATFORM / "claim_nodes").glob("*.yaml"))
    paths.extend((PLATFORM / "claim_nodes").glob("*.yml"))
    paths.extend((ROOT / "src" / "fluxvortex").rglob("*.py"))
    paths.extend(
        (
            ROOT / "researchpaper" / "uiuc_polars" / "SD7003.DRG",
            DOCS / "s6_sweep_v41.json",
        )
    )
    return sorted({path.resolve() for path in paths if path.is_file()})


def _validate_solver_sources(
    baseline_manifest: Mapping[str, Any],
) -> dict[str, str]:
    authority = baseline_manifest.get("authoritative_source_hashes")
    if not isinstance(authority, Mapping):
        raise RuntimeError("baseline manifest lacks authoritative source hashes")
    current: dict[str, str] = {}
    drift: list[str] = []
    for path in _solver_source_paths():
        relative = str(path.relative_to(ROOT))
        digest = _sha256_file(path)
        current[relative] = digest
        if authority.get(relative) != digest:
            drift.append(relative)
    if drift:
        raise RuntimeError(f"aerodynamic solver source drift: {drift[:10]}")
    return current


def _resolved_call(
    gpu_run_twist: Any,
    condition: tuple[float, float, float, float],
) -> dict[str, Any]:
    U, frequency, twist, aoa = condition
    explicit = dict(BASE)
    explicit.update(
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
    )
    resolved = {
        name: parameter.default
        for name, parameter in inspect.signature(
            gpu_run_twist
        ).parameters.items()
        if parameter.default is not inspect.Parameter.empty
    }
    resolved.update(explicit)
    return resolved


def _wind_force(body_force: Sequence[float], aoa_deg: float) -> dict[str, float]:
    force = np.asarray(body_force, dtype=float)
    angle = math.radians(aoa_deg)
    cosine = math.cos(angle)
    sine = math.sin(angle)
    return {
        "L_N": float(force[2] * cosine - force[0] * sine),
        "T_N": float(-(force[0] * cosine + force[2] * sine)),
    }


def _contribution_summary(
    contributions: Mapping[str, Any],
    *,
    aoa_deg: float,
) -> tuple[dict[str, Any], np.ndarray]:
    summary: dict[str, Any] = {}
    ledger_total = np.zeros(3, dtype=float)
    for node_id, raw_items in contributions.items():
        if not isinstance(raw_items, list):
            raise RuntimeError(f"{node_id}: malformed contribution list")
        node_total = np.zeros(3, dtype=float)
        items: list[dict[str, Any]] = []
        for raw in raw_items:
            if not isinstance(raw, Mapping):
                raise RuntimeError(f"{node_id}: malformed contribution")
            body = np.asarray(raw.get("body_force"), dtype=float)
            if body.shape != (3,) or not np.all(np.isfinite(body)):
                raise RuntimeError(f"{node_id}: invalid contribution force")
            node_total += body
            ledger_total += body
            items.append(
                {
                    "channel": raw.get("channel"),
                    "role": raw.get("role"),
                    "body_force_N": body.tolist(),
                    "wind_force": _wind_force(body, aoa_deg),
                    "metadata": dict(raw.get("metadata") or {}),
                }
            )
        summary[node_id] = {
            "items": items,
            "total_body_force_N": node_total.tolist(),
            "total_wind_force": _wind_force(node_total, aoa_deg),
        }
    return summary, ledger_total


def _valid_guards(guards: Any) -> bool:
    names = (
        "force_ledger",
        "unclassified_force",
        "unclassified_physical_force",
        "cycle_reduction",
        "aero_output_invariance",
    )
    return bool(
        isinstance(guards, Mapping)
        and all(
            isinstance(guards.get(name), Mapping)
            and guards[name].get("passed") is True
            for name in names
        )
    )


def _run_case(
    gpu_run_twist: Any,
    condition: tuple[float, float, float, float],
    baseline: Mapping[str, Any],
) -> dict[str, Any]:
    U, frequency, twist, aoa = condition
    key = condition_key(condition)
    call = _resolved_call(gpu_run_twist, condition)
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
    reported = {
        "L_N": float(result["L_wind"]),
        "T_N": float(result["T_wind"]),
    }
    reference = baseline.get(key)
    if not isinstance(reference, Mapping):
        raise RuntimeError(f"{key}: missing frozen baseline")
    delta = {
        "L_N": abs(reported["L_N"] - float(reference["L"])),
        "T_N": abs(reported["T_N"] - float(reference["T"])),
    }
    if max(delta.values()) > REPRO_TOLERANCE_N:
        raise RuntimeError(
            f"{key}: baseline reproduction failed {delta} > "
            f"{REPRO_TOLERANCE_N} N"
        )
    guards = result.get("claim_guards")
    if not _valid_guards(guards):
        raise RuntimeError(f"{key}: claim guards failed: {guards}")
    manifest = result.get("claim_manifest")
    contributions = result.get("claim_contributions")
    if not isinstance(manifest, Mapping) or not isinstance(
        contributions, Mapping
    ):
        raise RuntimeError(f"{key}: missing claim graph evidence")
    summary, ledger_total = _contribution_summary(
        contributions,
        aoa_deg=aoa,
    )
    target = np.asarray(
        [float(result["Fx_body"]), 0.0, float(result["Fz_body"])],
        dtype=float,
    )
    ledger_error = float(np.max(np.abs(ledger_total - target)))
    wind_total = _wind_force(ledger_total, aoa)
    wind_error = max(
        abs(wind_total["L_N"] - reported["L_N"]),
        abs(wind_total["T_N"] - reported["T_N"]),
    )
    if ledger_error > 1.0e-9 or wind_error > 1.0e-9:
        raise RuntimeError(
            f"{key}: recomputed ledger mismatch body={ledger_error} "
            f"wind={wind_error}"
        )
    graph_identity = _claim_graph_identity_sha256(manifest)
    return {
        "condition_key": key,
        "condition": {
            "U_m_s": U,
            "frequency_Hz": frequency,
            "nominal_twist_deg": twist,
            "solver_twist_amplitude_deg": twist / 2.0,
            "aoa_deg": aoa,
        },
        "resolved_call": call,
        "reported_wind_force": reported,
        "frozen_baseline": {
            "L_N": float(reference["L"]),
            "T_N": float(reference["T"]),
        },
        "absolute_reproduction_delta_N": delta,
        "claim_graph_identity_sha256": graph_identity,
        "claim_manifest_sha256": _canonical_hash(manifest),
        "claim_manifest": dict(manifest),
        "claim_guards": dict(guards),
        "claim_contributions": dict(contributions),
        "contribution_summary": summary,
        "recomputed_ledger": {
            "total_body_force_N": ledger_total.tolist(),
            "total_wind_force": wind_total,
            "max_body_error_N": ledger_error,
            "max_wind_error_N": wind_error,
        },
        "wall_s": time.time() - started,
    }


def run(output: Path, *, resume: bool = False) -> int:
    if _sha256_file(BASELINE_RESULT) != BASELINE_RESULT_SHA256:
        raise RuntimeError("frozen baseline result hash drift")
    if _sha256_file(BASELINE_MANIFEST) != BASELINE_MANIFEST_SHA256:
        raise RuntimeError("frozen baseline manifest hash drift")
    baseline = _load_json(BASELINE_RESULT)
    baseline_manifest = _load_json(BASELINE_MANIFEST)
    solver_sources = _validate_solver_sources(baseline_manifest)
    expected_keys = [condition_key(condition) for condition in WITNESSES]

    if output.exists():
        if not resume:
            raise FileExistsError(
                f"{output} exists; use a new path or explicit --resume"
            )
        campaign = _load_json(output)
        if campaign.get("expected_condition_keys") != expected_keys:
            raise RuntimeError("resume witness contract mismatch")
        if campaign.get("solver_source_hashes") != solver_sources:
            raise RuntimeError("resume solver source snapshot drift")
    else:
        campaign = {
            "schema_version": 1,
            "status": "running",
            "started_at": datetime.now().astimezone().isoformat(),
            "preregistration": {
                "path": str(PREREG.relative_to(ROOT)),
                "sha256": _sha256_file(PREREG),
            },
            "baseline_result": {
                "path": str(BASELINE_RESULT.relative_to(ROOT)),
                "sha256": BASELINE_RESULT_SHA256,
            },
            "baseline_manifest": {
                "path": str(BASELINE_MANIFEST.relative_to(ROOT)),
                "sha256": BASELINE_MANIFEST_SHA256,
            },
            "runner": {
                "path": str(Path(__file__).resolve().relative_to(ROOT)),
                "sha256": _sha256_file(Path(__file__).resolve()),
            },
            "solver_source_hashes": solver_sources,
            "expected_condition_keys": expected_keys,
            "formal_anchor": None,
            "cases": {},
            "failures": {},
        }
        _write_json_atomic(output, campaign)

    import warp as wp

    wp.init()
    from _v2_robo import gpu_run_twist

    if campaign.get("formal_anchor") is None:
        print("[claim-witness] cold preconditioner (discarded)", flush=True)
        _run_case(gpu_run_twist, ANCHOR, baseline)
        print("[claim-witness] formal warm anchor", flush=True)
        anchor = _run_case(gpu_run_twist, ANCHOR, baseline)
        campaign["formal_anchor"] = anchor
        campaign["claim_graph_identity_sha256"] = anchor[
            "claim_graph_identity_sha256"
        ]
        _write_json_atomic(output, campaign)

    graph_identity = campaign["claim_graph_identity_sha256"]
    for index, condition in enumerate(WITNESSES, start=1):
        key = condition_key(condition)
        if key in campaign["cases"]:
            continue
        try:
            case = _run_case(gpu_run_twist, condition, baseline)
            if case["claim_graph_identity_sha256"] != graph_identity:
                raise RuntimeError(f"{key}: claim graph identity drift")
            campaign["cases"][key] = case
            campaign["failures"].pop(key, None)
            print(
                f"[claim-witness] {index}/{len(WITNESSES)} {key}: "
                f"L={case['reported_wind_force']['L_N']:+.3f} "
                f"T={case['reported_wind_force']['T_N']:+.3f} "
                f"({case['wall_s']:.0f}s)",
                flush=True,
            )
        except Exception as exc:
            campaign["status"] = "failed"
            campaign["failures"][key] = f"{type(exc).__name__}: {exc}"
            campaign["updated_at"] = datetime.now().astimezone().isoformat()
            _write_json_atomic(output, campaign)
            raise
        campaign["updated_at"] = datetime.now().astimezone().isoformat()
        _write_json_atomic(output, campaign)

    campaign["status"] = "complete"
    campaign["completed_condition_count"] = len(campaign["cases"])
    campaign["max_reproduction_delta_N"] = max(
        max(case["absolute_reproduction_delta_N"].values())
        for case in campaign["cases"].values()
    )
    campaign["updated_at"] = datetime.now().astimezone().isoformat()
    _write_json_atomic(output, campaign)
    print(f"[claim-witness] COMPLETE -> {output}", flush=True)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    return run(args.output, resume=args.resume)


if __name__ == "__main__":
    raise SystemExit(main())
