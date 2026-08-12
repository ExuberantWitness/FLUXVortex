"""Causal Ramesh-LDVM diagnostic for the Figure-14 Scherer experiment.

The finite-wing UVLM remains the baseline.  A clean-room two-dimensional
LDVM is run twice for each motion: once with an explicit LESP threshold and
once with shedding disabled.  Their coefficient difference is multiplied by
the classical finite-wing lift-slope ratio and added to the frozen UVLM mean.

This script is a mechanism diagnostic, not yet the final three-dimensional
LEV coupling.  It deliberately evaluates source-derived threshold hypotheses
without selecting a value by the Figure-14 observations.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from forward_flight_benchmarks.cases import IZRAELEVITZ_2017_FIG14_SCHERER
from forward_flight_benchmarks.ldvm_uvlm_correction import (
    LDVMSectionSettings,
    LESPThreshold,
    run_ldvm_separation_pair,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
BASE_ROOT = (
    REPO_ROOT
    / "docs/forward_flight_large_pitch/reproductions"
    / "unified_fluxv_upgrade_20260812"
)
SOURCE_CSV = BASE_ROOT / "source_data/izraelevitz2017_fig14_digitized.csv"
BASE_CSV = (
    BASE_ROOT / "runs/20260812_scherer_fig14_experiment_full/mean_thrust_vs_phase.csv"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "docs/forward_flight_large_pitch/reproductions"
    / "unified_fluxv_v4_ldvm_stevens_20260812/runs/fig14_ldvm_diagnostic"
)


THRESHOLDS = (
    LESPThreshold(
        value=0.18,
        section_family="SD7003",
        reynolds=30_000.0,
        source="Ramesh LDVM v2.5 bundled SD7003 reference",
    ),
    LESPThreshold(
        value=0.239,
        section_family="NACA 63A015 static-polar estimate",
        reynolds=309_677.0,
        source=(
            "Scherer static CLa=0.065/deg and CLmax=0.90 imply "
            "alpha_stall=13.85 deg; Lcrit proxy=sin(alpha_stall)"
        ),
        source_role="geometry/static-polar hypothesis; not force-fit",
    ),
    LESPThreshold(
        value=0.29,
        section_family="NACA 0012",
        reynolds=10_000.0,
        source="Martinez-Carmena et al. SVDVM Case II published Lcrit",
    ),
)


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write empty CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _metrics(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    error = predicted - observed
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "bias": float(np.mean(error)),
        "max_abs": float(np.max(np.abs(error))),
    }


def _condition_delta_ct(
    theta_max_deg: float,
    phase_offset_deg: float,
    threshold: LESPThreshold,
    *,
    steps_per_cycle: int,
    cycles: int,
) -> tuple[float, dict[str, Any]]:
    case = IZRAELEVITZ_2017_FIG14_SCHERER
    omega_star = np.pi * case.strouhal / case.heave_to_chord
    period_star = 2.0 * np.pi / omega_star
    delta_time = period_star / steps_per_cycle
    time = np.arange(cycles * steps_per_cycle, dtype=float) * delta_time
    angle_amplitude = np.deg2rad(theta_max_deg)
    phase = np.deg2rad(phase_offset_deg)
    alpha = angle_amplitude * np.cos(omega_star * time + phase)
    alpha_rate = -angle_amplitude * omega_star * np.sin(omega_star * time + phase)
    heave_rate = -case.heave_to_chord * omega_star * np.sin(omega_star * time)
    pair = run_ldvm_separation_pair(
        alpha_rad=alpha,
        alpha_rate_per_convective_time=alpha_rate,
        heave_rate_over_u=heave_rate,
        delta_time_convective=delta_time,
        pivot_fraction_chord=case.pivot_fraction_chord,
        threshold=threshold,
        settings=LDVMSectionSettings(
            ndiv=50,
            naterm=24,
            core_radius_time_step_ratio=1.3,
            max_wake_steps=cycles * steps_per_cycle,
        ),
    )
    selected = slice((cycles - 1) * steps_per_cycle, cycles * steps_per_cycle)
    finite_wing_gain = 1.0 / (1.0 + 2.0 / case.aspect_ratio)
    delta_cd = np.asarray(pair["delta"]["CDf"])[selected]
    delta_ct = -finite_wing_gain * float(np.mean(delta_cd))
    return delta_ct, {
        "finite_wing_gain": finite_wing_gain,
        "lev_count_final": int(pair["lev_count"][-1]),
        "max_abs_pre_cap_lesp": float(
            np.max(np.abs(np.asarray(pair["separated"]["lesp"])[selected]))
        ),
        "max_abs_post_cap_a0": float(
            np.max(np.abs(np.asarray(pair["separated"]["A0"])[selected]))
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--steps-per-cycle", type=int, default=96)
    parser.add_argument("--cycles", type=int, default=4)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    source_rows = [
        row for row in _rows(SOURCE_CSV) if row["series"] == "scherer_1968_experiment"
    ]
    old_by_condition = {
        (float(row["theta_max_deg"]), float(row["phase_offset_deg"])): float(row["CT"])
        for row in _rows(BASE_CSV)
        if row["series"] == "fluxv_uvpm"
    }
    conditions = sorted(
        {
            (float(row["theta_max_deg"]), float(row["phase_offset_deg"]))
            for row in source_rows
        }
    )

    prediction_rows: list[dict[str, Any]] = []
    condition_audit: dict[str, Any] = {}
    for threshold in THRESHOLDS:
        for theta_max, phase_offset in conditions:
            delta_ct, audit = _condition_delta_ct(
                theta_max,
                phase_offset,
                threshold,
                steps_per_cycle=args.steps_per_cycle,
                cycles=args.cycles,
            )
            old_ct = old_by_condition[(theta_max, phase_offset)]
            prediction_rows.append(
                {
                    "model": f"fluxv_uvlm_ldvm_Lcrit_{threshold.value:g}",
                    "theta_max_deg": theta_max,
                    "phase_offset_deg": phase_offset,
                    "old_fluxv_CT": old_ct,
                    "ldvm_delta_CT": delta_ct,
                    "CT": old_ct + delta_ct,
                    "lesp_critical": threshold.value,
                    "threshold_section_family": threshold.section_family,
                    "threshold_source_role": threshold.source_role,
                }
            )
            condition_audit[
                f"L{threshold.value:g}_theta{theta_max:g}_psi{phase_offset:g}"
            ] = audit

    metric_rows: list[dict[str, Any]] = []
    observed = np.asarray([float(row["ct"]) for row in source_rows])
    old_prediction = np.asarray(
        [
            old_by_condition[
                (float(row["theta_max_deg"]), float(row["phase_offset_deg"]))
            ]
            for row in source_rows
        ]
    )
    metric_rows.append({"model": "fluxv_uvpm", **_metrics(observed, old_prediction)})
    for threshold in THRESHOLDS:
        lookup = {
            (row["theta_max_deg"], row["phase_offset_deg"]): row["CT"]
            for row in prediction_rows
            if np.isclose(float(row["lesp_critical"]), threshold.value)
        }
        prediction = np.asarray(
            [
                lookup[(float(row["theta_max_deg"]), float(row["phase_offset_deg"]))]
                for row in source_rows
            ]
        )
        metric_rows.append(
            {
                "model": f"fluxv_uvlm_ldvm_Lcrit_{threshold.value:g}",
                **_metrics(observed, prediction),
            }
        )

    predictions_path = output / "fig14_ldvm_predictions.csv"
    metrics_path = output / "fig14_ldvm_metrics.csv"
    _write_csv(predictions_path, prediction_rows)
    _write_csv(metrics_path, metric_rows)
    manifest = {
        "run_id": output.name,
        "status": "exploratory_mechanism_diagnostic",
        "model_semantics": (
            "frozen finite-wing UVLM mean plus clean-room Ramesh LDVM "
            "separated-minus-attached section correction"
        ),
        "not_a_claim": (
            "not a three-dimensional material-LEV UVLM coupling and not a "
            "threshold selection by Figure-14 score"
        ),
        "thresholds": [threshold.manifest() for threshold in THRESHOLDS],
        "steps_per_cycle": args.steps_per_cycle,
        "cycles": args.cycles,
        "source_hashes": {
            str(SOURCE_CSV.relative_to(REPO_ROOT)): _sha256(SOURCE_CSV),
            str(BASE_CSV.relative_to(REPO_ROOT)): _sha256(BASE_CSV),
        },
        "result_hashes": {
            predictions_path.name: _sha256(predictions_path),
            metrics_path.name: _sha256(metrics_path),
        },
        "condition_audit": condition_audit,
    }
    (output / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(metric_rows, indent=2))


if __name__ == "__main__":
    main()
