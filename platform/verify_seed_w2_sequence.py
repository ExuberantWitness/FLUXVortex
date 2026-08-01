"""Sequence-controlled reproduction audit for the frozen U10 W2 seed point."""

from __future__ import annotations

import argparse
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

import run_claim_witnesses as witness  # noqa: E402
from lb_sweep118 import BASE, spc_of  # noqa: E402


PREREG = DIAG / "v41_seed_w2_sequence_repro_prereg_20260729.md"
TOLERANCE_N = 0.15
SEQUENCE = (
    ("A0_cold", (8.0, 2.6, 0.0, 5.0)),
    ("A1_warm", (8.0, 2.6, 0.0, 5.0)),
    ("X1", (10.0, 2.6, 22.5, 5.0)),
    ("X2", (10.0, 2.6, 22.5, 5.0)),
    ("B20", (10.0, 2.6, 20.0, 5.0)),
    ("B25", (10.0, 2.6, 25.0, 5.0)),
    ("X3", (10.0, 2.6, 22.5, 5.0)),
)


def _run_raw(
    gpu_run_twist: Any,
    label: str,
    condition: tuple[float, float, float, float],
    baseline: Mapping[str, Any],
) -> dict[str, Any]:
    U, frequency, twist, aoa = condition
    key = witness.condition_key(condition)
    reference = baseline.get(key)
    if not isinstance(reference, Mapping):
        raise RuntimeError(f"{key}: missing frozen baseline")
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
    if not all(math.isfinite(value) for value in reported.values()):
        raise RuntimeError(f"{label}: non-finite reported force")
    guards = result.get("claim_guards")
    if not witness._valid_guards(guards):
        raise RuntimeError(f"{label}: claim guards failed")
    manifest = result.get("claim_manifest")
    contributions = result.get("claim_contributions")
    if not isinstance(manifest, Mapping) or not isinstance(
        contributions, Mapping
    ):
        raise RuntimeError(f"{label}: missing claim graph evidence")
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
        abs(wind_total["L_N"] - reported["L_N"]),
        abs(wind_total["T_N"] - reported["T_N"]),
    )
    if body_error > 1.0e-9 or wind_error > 1.0e-9:
        raise RuntimeError(
            f"{label}: ledger mismatch body={body_error} wind={wind_error}"
        )
    signed_delta = {
        "L_N": reported["L_N"] - float(reference["L"]),
        "T_N": reported["T_N"] - float(reference["T"]),
    }
    return {
        "label": label,
        "condition_key": key,
        "condition": {
            "U_m_s": U,
            "frequency_Hz": frequency,
            "nominal_twist_deg": twist,
            "solver_twist_amplitude_deg": twist / 2.0,
            "aoa_deg": aoa,
        },
        "resolved_call": witness._resolved_call(gpu_run_twist, condition),
        "reported_wind_force": reported,
        "frozen_baseline": {
            "L_N": float(reference["L"]),
            "T_N": float(reference["T"]),
        },
        "signed_cache_delta_N": signed_delta,
        "absolute_cache_delta_N": {
            channel: abs(value) for channel, value in signed_delta.items()
        },
        "within_cache_band": max(abs(value) for value in signed_delta.values())
        <= TOLERANCE_N,
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


def _classify(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_label = {record["label"]: record for record in records}
    repeated = [by_label[label]["reported_wind_force"] for label in ("X1", "X2", "X3")]
    statistics: dict[str, Any] = {}
    run_variability = False
    sequence_dependence = False
    cache_drift = False
    for channel in ("L_N", "T_N"):
        values = np.asarray([item[channel] for item in repeated], dtype=float)
        first_pair_mean = float(np.mean(values[:2]))
        repeat_mean = float(np.mean(values))
        repeat_spread = float(np.ptp(values))
        sequence_shift = abs(float(values[2]) - first_pair_mean)
        cache_value = float(by_label["X1"]["frozen_baseline"][channel])
        cache_delta = repeat_mean - cache_value
        midpoint = 0.5 * (
            float(by_label["B20"]["reported_wind_force"][channel])
            + float(by_label["B25"]["reported_wind_force"][channel])
        )
        statistics[channel] = {
            "X_values_N": values.tolist(),
            "repeat_mean_N": repeat_mean,
            "repeat_spread_N": repeat_spread,
            "sequence_shift_N": sequence_shift,
            "frozen_cache_N": cache_value,
            "signed_mean_cache_delta_N": cache_delta,
            "absolute_mean_cache_delta_N": abs(cache_delta),
            "neighbor_midpoint_N": midpoint,
            "signed_midpoint_delta_N": repeat_mean - midpoint,
            "absolute_midpoint_delta_N": abs(repeat_mean - midpoint),
        }
        run_variability = run_variability or repeat_spread > TOLERANCE_N
        sequence_dependence = sequence_dependence or sequence_shift > TOLERANCE_N
        cache_drift = cache_drift or (
            repeat_spread <= TOLERANCE_N
            and sequence_shift <= TOLERANCE_N
            and abs(cache_delta) > TOLERANCE_N
        )

    flags = {
        "RUN_VARIABILITY": run_variability,
        "SEQUENCE_DEPENDENCE": sequence_dependence,
        "CACHE_DRIFT": cache_drift,
    }
    if run_variability:
        primary = "RUN_VARIABILITY"
    elif sequence_dependence:
        primary = "SEQUENCE_DEPENDENCE"
    elif cache_drift:
        primary = "CACHE_DRIFT"
    else:
        primary = "CACHE_COMPATIBLE_BUT_PRIOR_OUTLIER_UNRESOLVED"
    return {
        "primary_classification": primary,
        "flags": flags,
        "statistics": statistics,
        "fresh_confirmed151_required": primary
        in {"RUN_VARIABILITY", "SEQUENCE_DEPENDENCE", "CACHE_DRIFT"},
        "old_seed_authoritative": False,
    }


def run(output: Path) -> int:
    if output.exists():
        raise FileExistsError(f"{output} exists; use a new versioned path")
    if (
        witness._sha256_file(witness.BASELINE_RESULT)
        != witness.BASELINE_RESULT_SHA256
    ):
        raise RuntimeError("frozen baseline result hash drift")
    if (
        witness._sha256_file(witness.BASELINE_MANIFEST)
        != witness.BASELINE_MANIFEST_SHA256
    ):
        raise RuntimeError("frozen baseline manifest hash drift")
    baseline = witness._load_json(witness.BASELINE_RESULT)
    baseline_manifest = witness._load_json(witness.BASELINE_MANIFEST)
    solver_sources = witness._validate_solver_sources(baseline_manifest)
    campaign: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "started_at": datetime.now().astimezone().isoformat(),
        "preregistration": {
            "path": str(PREREG.relative_to(ROOT)),
            "sha256": witness._sha256_file(PREREG),
        },
        "runner": {
            "path": str(Path(__file__).resolve().relative_to(ROOT)),
            "sha256": witness._sha256_file(Path(__file__).resolve()),
        },
        "frozen_baseline": {
            "path": str(witness.BASELINE_RESULT.relative_to(ROOT)),
            "sha256": witness.BASELINE_RESULT_SHA256,
        },
        "solver_source_hashes": solver_sources,
        "tolerance_N": TOLERANCE_N,
        "sequence": [
            {"label": label, "condition_key": witness.condition_key(condition)}
            for label, condition in SEQUENCE
        ],
        "records": [],
    }
    witness._write_json_atomic(output, campaign)

    import warp as wp

    wp.init()
    from _v2_robo import gpu_run_twist

    graph_identity: str | None = None
    for index, (label, condition) in enumerate(SEQUENCE, start=1):
        try:
            record = _run_raw(gpu_run_twist, label, condition, baseline)
            if graph_identity is None:
                graph_identity = record["claim_graph_identity_sha256"]
            elif record["claim_graph_identity_sha256"] != graph_identity:
                raise RuntimeError(f"{label}: claim graph identity drift")
            campaign["records"].append(record)
            campaign["updated_at"] = datetime.now().astimezone().isoformat()
            witness._write_json_atomic(output, campaign)
            print(
                f"[seed-sequence] {index}/{len(SEQUENCE)} {label} "
                f"L={record['reported_wind_force']['L_N']:+.6f} "
                f"T={record['reported_wind_force']['T_N']:+.6f} "
                f"dL={record['signed_cache_delta_N']['L_N']:+.6f} "
                f"dT={record['signed_cache_delta_N']['T_N']:+.6f}",
                flush=True,
            )
        except Exception as exc:
            campaign["status"] = "failed"
            campaign["failure"] = f"{type(exc).__name__}: {exc}"
            campaign["updated_at"] = datetime.now().astimezone().isoformat()
            witness._write_json_atomic(output, campaign)
            raise

    warm_anchor = next(
        record for record in campaign["records"] if record["label"] == "A1_warm"
    )
    if not warm_anchor["within_cache_band"]:
        raise RuntimeError("warm formal anchor failed frozen 0.15 N band")
    campaign["classification"] = _classify(campaign["records"])
    campaign["claim_graph_identity_sha256"] = graph_identity
    campaign["status"] = "complete"
    campaign["updated_at"] = datetime.now().astimezone().isoformat()
    witness._write_json_atomic(output, campaign)
    print(
        "[seed-sequence] COMPLETE: "
        f"{campaign['classification']['primary_classification']} -> {output}",
        flush=True,
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    return run(args.output)


if __name__ == "__main__":
    raise SystemExit(main())
