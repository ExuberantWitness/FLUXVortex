"""Run old FluxV and the frozen v4b transfer on Razak/Lambert experiments."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from .razak2014 import (
    RAZAK_2014_CASES,
    apply_frozen_v4b_transfer,
    run_razak_old_fluxv,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DOC_ROOT = (
    REPO_ROOT
    / "docs/forward_flight_large_pitch/reproductions/fluxv_v4b_newcases_20260813"
)
SOURCE_CSV = DOC_ROOT / "source_data/razak_lambert_experiment_errorbar_centres.csv"
DEFAULT_OUTPUT = DOC_ROOT / "runs/20260813_razak2014_smoke"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty CSV")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _metrics(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    error = predicted - observed
    span = float(np.ptp(observed))
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "bias": float(np.mean(error)),
        "max_abs": float(np.max(np.abs(error))),
        "range_nrmse": float(np.sqrt(np.mean(error**2)) / span)
        if span > 0
        else np.nan,
    }


def _score_channel(
    observations: list[dict[str, str]], history: dict[str, Any], quantity: str
) -> tuple[dict[str, float], np.ndarray]:
    phase = np.asarray([float(row["phase"]) for row in observations])
    observed = np.asarray([float(row["coefficient"]) for row in observations])
    predicted = np.interp(
        phase,
        np.asarray(history["phase"], dtype=float),
        np.asarray(history[quantity], dtype=float),
        period=1.0,
    )
    return _metrics(observed, predicted), predicted


def _plot(
    source: list[dict[str, str]],
    histories: dict[tuple[int, str], dict[str, Any]],
    output: Path,
) -> list[Path]:
    figures = sorted({figure for figure, _model in histories})
    fig, axes = plt.subplots(
        len(figures),
        2,
        figsize=(10.8, 2.35 * len(figures)),
        squeeze=False,
    )
    for row_index, figure in enumerate(figures):
        for column_index, quantity in enumerate(("CL", "CD")):
            axis = axes[row_index, column_index]
            observed = [
                row
                for row in source
                if int(row["figure"]) == figure and row["quantity"] == quantity
            ]
            axis.errorbar(
                [float(row["phase"]) for row in observed],
                [float(row["coefficient"]) for row in observed],
                yerr=[float(row["std"]) for row in observed],
                fmt="o",
                markersize=1.8,
                linestyle="none",
                color="black",
                elinewidth=0.45,
                capsize=1.1,
                label="Experiment ±1 cycle SD",
                zorder=3,
            )
            for model, style, color, label in (
                ("old", "--", "tab:blue", "FluxV old"),
                ("v4b", "-", "tab:red", "FluxV v4b transfer"),
            ):
                history = histories[(figure, model)]
                axis.plot(
                    history["phase"],
                    history[quantity],
                    style,
                    color=color,
                    linewidth=1.25,
                    label=label,
                )
            case = RAZAK_2014_CASES[figure]
            axis.set_title(
                f"Fig. {figure}: {case.motion_family.replace('_', ' ')}, "
                f"U={case.freestream_m_s:g} m/s, f={case.frequency_hz:g} Hz",
                fontsize=8.5,
            )
            axis.set_ylabel(quantity)
            axis.grid(alpha=0.25)
            if row_index == len(figures) - 1:
                axis.set_xlabel("Displayed-cycle phase (no phase fit)")
            if row_index == 0:
                axis.legend(frameon=False, fontsize=7.5)
    fig.suptitle(
        "Razak–Dimitriadis / Lambert experiments: nominal-motion diagnostic",
        y=1.0,
    )
    fig.tight_layout()
    png = output / "razak2014_old_vs_v4b_phase_loads.png"
    pdf = output / "razak2014_old_vs_v4b_phase_loads.pdf"
    fig.savefig(png, dpi=220, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return [png, pdf]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quality", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--ldvm-steps-per-cycle", type=int, default=512)
    parser.add_argument("--ldvm-max-wake-steps", type=int, default=256)
    parser.add_argument(
        "--figures",
        type=int,
        nargs="+",
        choices=tuple(sorted(RAZAK_2014_CASES)),
        default=tuple(sorted(RAZAK_2014_CASES)),
    )
    args = parser.parse_args()

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    source = _rows(SOURCE_CSV)
    histories: dict[tuple[int, str], dict[str, Any]] = {}
    metric_rows: list[dict[str, Any]] = []
    phase_rows: list[dict[str, Any]] = []
    case_manifests: dict[str, Any] = {}

    for figure in args.figures:
        case = RAZAK_2014_CASES[figure]
        print(f"running Figure {figure} old FluxV", flush=True)
        old, movement, movement_metadata = run_razak_old_fluxv(
            case, args.quality, output_samples=128
        )
        print(f"running Figure {figure} frozen v4b transfer", flush=True)
        v4b = apply_frozen_v4b_transfer(
            case,
            old,
            movement,
            output_samples=128,
            ldvm_steps_per_cycle=args.ldvm_steps_per_cycle,
            ldvm_max_wake_steps=args.ldvm_max_wake_steps,
        )
        histories[(figure, "old")] = old
        histories[(figure, "v4b")] = v4b
        print(f"completed Figure {figure}", flush=True)
        case_manifests[str(figure)] = {
            "case": case.manifest(),
            "movement": movement_metadata,
            "old_runtime_s": old["runtime_s"],
            "source_cycle_step_range": old["source_cycle_step_range"],
            "v4b_ldvm_steps_per_cycle": v4b["ldvm_steps_per_cycle"],
            "v4b_ldvm_delta_time_convective": v4b["ldvm_delta_time_convective"],
            "v4b_ldvm_max_wake_steps": v4b["ldvm_max_wake_steps"],
            "v4b_lesp": v4b["lesp_provenance"],
            "mean_persistence": float(np.mean(v4b["persistence"])),
            "mean_ldvm_shedding": float(np.mean(v4b["ldvm_shedding"])),
        }
        for model, history in (("fluxv_old", old), ("fluxv_v4b", v4b)):
            for quantity in ("CL", "CD"):
                observed = [
                    row
                    for row in source
                    if int(row["figure"]) == figure and row["quantity"] == quantity
                ]
                scores, prediction = _score_channel(observed, history, quantity)
                metric_rows.append(
                    {
                        "figure": figure,
                        "case_id": case.case_id,
                        "model": model,
                        "quantity": quantity,
                        "observation_count": len(observed),
                        **scores,
                    }
                )
                for observation, value in zip(observed, prediction, strict=True):
                    phase_rows.append(
                        {
                            "figure": figure,
                            "case_id": case.case_id,
                            "model": model,
                            "quantity": quantity,
                            "phase": observation["phase"],
                            "experiment": observation["coefficient"],
                            "experiment_std": observation["std"],
                            "prediction": float(value),
                        }
                    )

    metrics_path = output / "accuracy_metrics.csv"
    phases_path = output / "scored_phase_samples.csv"
    histories_path = output / "model_phase_histories.csv"
    _write_csv(metrics_path, metric_rows)
    _write_csv(phases_path, phase_rows)
    full_histories: list[dict[str, Any]] = []
    for (figure, model), history in histories.items():
        for index, phase in enumerate(np.asarray(history["phase"])):
            full_histories.append(
                {
                    "figure": figure,
                    "case_id": RAZAK_2014_CASES[figure].case_id,
                    "model": model,
                    "phase": float(phase),
                    "CL": float(history["CL"][index]),
                    "CD": float(history["CD"][index]),
                    "persistence": (
                        float(history["persistence"][index])
                        if "persistence" in history
                        else ""
                    ),
                    "ldvm_delta_CL": (
                        float(history["ldvm_delta_CL"][index])
                        if "ldvm_delta_CL" in history
                        else ""
                    ),
                    "ldvm_delta_CD": (
                        float(history["ldvm_delta_CD"][index])
                        if "ldvm_delta_CD" in history
                        else ""
                    ),
                    "ldvm_shedding": (
                        float(history["ldvm_shedding"][index])
                        if "ldvm_shedding" in history
                        else ""
                    ),
                }
            )
    _write_csv(histories_path, full_histories)
    figures = _plot(source, histories, output)

    aggregate: dict[str, dict[str, float]] = {}
    for model in ("fluxv_old", "fluxv_v4b"):
        aggregate[model] = {}
        for quantity in ("CL", "CD"):
            selected = [
                row
                for row in metric_rows
                if row["model"] == model and row["quantity"] == quantity
            ]
            aggregate[model][f"macro_case_rmse_{quantity}"] = float(
                np.mean([row["rmse"] for row in selected])
            )
            aggregate[model][f"macro_case_mae_{quantity}"] = float(
                np.mean([row["mae"] for row in selected])
            )

    result_files = (metrics_path, phases_path, histories_path, *figures)
    summary = {
        "run_id": output.name,
        "quality": args.quality,
        "status": "nominal_motion_transfer_diagnostic",
        "aggregate": aggregate,
        "case_manifests": case_manifests,
        "metric_contract": (
            "raw displayed-cycle phase; no phase/amplitude fit; per-case then "
            "equal-case macro average; visible error-bar centres only"
        ),
        "limitations": [
            "measured 64-point flap/pitch input histories are not public",
            "quarter-chord pitch axis is a declared reconstruction assumption",
            "v4b uses independent 2-D LDVM strip discrepancies",
            "smoke is not a mesh/time/wake convergence demonstration",
        ],
        "source_hashes": {
            str(path.relative_to(REPO_ROOT)): _sha256(path)
            for path in (
                Path(__file__).resolve(),
                Path(__file__).with_name("razak2014.py").resolve(),
                Path(__file__).with_name("ldvm_uvlm_correction.py").resolve(),
                Path(__file__).with_name("causal_incidence_owner.py").resolve(),
                Path(__file__).with_name("uvlm_polar_correction.py").resolve(),
                REPO_ROOT / "platform/ldvm_fourier.py",
                SOURCE_CSV,
            )
        },
        "result_hashes": {path.name: _sha256(path) for path in result_files},
    }
    summary_path = output / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
