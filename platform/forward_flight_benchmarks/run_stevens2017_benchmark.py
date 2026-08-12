"""Run the lift-only Stevens & Babinsky (2017) benchmark.

The benchmark freezes two predictions before loading the Figure-21 force
observations: the untouched FluxV UVLM load channel and the existing 2-D LDVM
strip diagnostic.  Experimental lift is then used only for scoring.  Stevens
& Babinsky did not publish drag, so predicted drag is exported as a diagnostic
but is never assigned an experimental error metric.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform as system_platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from .stevens2017 import (
    STEVENS_2017,
    STEVENS_EXPERIMENTAL_OBSERVABLES,
    STEVENS_EXPERIMENT_HAS_DRAG,
    STEVENS_FIG21_EXPERIMENT_CSV,
    load_stevens_fig21_experiment,
    run_stevens_fluxv_baseline,
    run_stevens_fluxv_ldvm_increment,
    run_stevens_ldvm_strips,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DOC_ROOT = (
    REPO_ROOT / "docs/forward_flight_large_pitch/reproductions/"
    "unified_fluxv_v4_ldvm_stevens_20260812"
)
LDVM_DIAGNOSTIC_LESP_CRITICAL = 0.11
LDVM_STEPS_PER_CHORD = {"smoke": 24, "full": 96}
SCORING_WINDOWS = (
    ("primary_0_to_5", 0.0, 5.0, True),
    ("startup_sensitivity_0p05_to_5", 0.05, 5.0, False),
)
AXES = (
    ("leading_edge_axis", 0.0, "CL_leading_edge_axis"),
    ("mid_chord_axis", 0.5, "CL_mid_chord_axis"),
)
MODEL_LABELS = {
    "experiment": "Stevens 2017 experiment",
    "fluxv_old": "FluxV old (UVLM load channel)",
    "fluxv_v4_ldvm_increment": "FluxV v4 (UVLM + LDVM increment)",
    "ldvm_diagnostic": "LDVM 2-D strip diagnostic",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT).as_posix()


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Path):
        return _relative(value)
    raise TypeError(type(value).__name__)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _lift_metrics(
    distance: np.ndarray,
    observation: np.ndarray,
    prediction: np.ndarray,
    *,
    start: float,
    end: float,
) -> dict[str, float | int]:
    distance = np.asarray(distance, dtype=float)
    observation = np.asarray(observation, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    if not (distance.shape == observation.shape == prediction.shape):
        raise ValueError("metric arrays must have identical shape")
    mask = (distance >= start - 1.0e-12) & (distance <= end + 1.0e-12)
    if np.count_nonzero(mask) < 3:
        raise ValueError("metric window has fewer than three samples")
    x = distance[mask]
    truth = observation[mask]
    model = prediction[mask]
    if not np.all(np.isfinite(truth + model)):
        raise FloatingPointError("non-finite lift history in scoring window")

    error = model - truth
    truth_range = float(np.ptp(truth))
    if truth_range <= 0.0:
        raise ValueError("experimental lift range must be positive")
    correlation = float(np.corrcoef(truth, model)[0, 1])
    truth_peak = int(np.argmax(truth))
    model_peak = int(np.argmax(model))
    return {
        "n_samples": int(x.size),
        "lift_rmse": float(np.sqrt(np.mean(error**2))),
        "lift_mae": float(np.mean(np.abs(error))),
        "lift_bias": float(np.mean(error)),
        "lift_range_nrmse": float(np.sqrt(np.mean(error**2)) / truth_range),
        "lift_correlation": correlation,
        "experiment_peak_CL": float(truth[truth_peak]),
        "model_peak_CL": float(model[model_peak]),
        "peak_magnitude_error_CL": float(model[model_peak] - truth[truth_peak]),
        "experiment_peak_s_over_c": float(x[truth_peak]),
        "model_peak_s_over_c": float(x[model_peak]),
        "peak_phase_error_s_over_c": float(x[model_peak] - x[truth_peak]),
    }


def _assert_common_grid(history: dict[str, Any], target: np.ndarray) -> None:
    grid = np.asarray(history["s_over_c"], dtype=float)
    if not np.allclose(grid, target, rtol=0.0, atol=1.0e-12):
        raise ValueError("model history does not use the frozen Figure-21 grid")


def _prediction_rows(
    predictions: dict[str, dict[str, dict[str, Any]]],
    experiment: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    distance = np.asarray(experiment["s_over_c"], dtype=float)
    for axis_name, pivot, observation_key in AXES:
        old = predictions[axis_name]["fluxv_old"]
        v4 = predictions[axis_name]["fluxv_v4_ldvm_increment"]
        ldvm = predictions[axis_name]["ldvm_diagnostic"]
        _assert_common_grid(old, distance)
        _assert_common_grid(v4, distance)
        _assert_common_grid(ldvm, distance)
        observation = np.asarray(experiment[observation_key], dtype=float)
        for index, location in enumerate(distance):
            rows.append(
                {
                    "axis": axis_name,
                    "pivot_fraction_chord": pivot,
                    "s_over_c": location,
                    "pitch_deg": float(old["pitch_deg"][index]),
                    "experiment_CL": float(observation[index]),
                    "fluxv_old_CL": float(old["CL"][index]),
                    "fluxv_old_CD_diagnostic_only": float(old["CD"][index]),
                    "fluxv_v4_ldvm_increment_CL": float(v4["CL"][index]),
                    "fluxv_v4_ldvm_increment_CD_diagnostic_only": float(
                        v4["CD"][index]
                    ),
                    "ldvm_diagnostic_CL": float(ldvm["CL"][index]),
                    "ldvm_diagnostic_CD_diagnostic_only": float(ldvm["CD"][index]),
                    "source_figure": 21,
                    "experimental_drag_available": False,
                }
            )
    return rows


def _metric_rows(
    predictions: dict[str, dict[str, dict[str, Any]]],
    experiment: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    distance = np.asarray(experiment["s_over_c"], dtype=float)
    for axis_name, pivot, observation_key in AXES:
        observation = np.asarray(experiment[observation_key], dtype=float)
        for model in (
            "fluxv_old",
            "fluxv_v4_ldvm_increment",
            "ldvm_diagnostic",
        ):
            prediction = np.asarray(predictions[axis_name][model]["CL"], dtype=float)
            for window, start, end, primary in SCORING_WINDOWS:
                rows.append(
                    {
                        "axis": axis_name,
                        "pivot_fraction_chord": pivot,
                        "model": model,
                        "model_label": MODEL_LABELS[model],
                        "window": window,
                        "window_start_s_over_c": start,
                        "window_end_s_over_c": end,
                        "primary_metric": primary,
                        **_lift_metrics(
                            distance,
                            observation,
                            prediction,
                            start=start,
                            end=end,
                        ),
                        "drag_metric_status": (
                            "not_scored_no_published_experimental_drag"
                        ),
                    }
                )
    return rows


def _plot(
    path: Path,
    predictions: dict[str, dict[str, dict[str, Any]]],
    experiment: dict[str, Any],
) -> list[Path]:
    plt.rcParams.update(
        {
            "font.size": 9.0,
            "axes.labelsize": 9.5,
            "axes.titlesize": 10.0,
            "legend.fontsize": 8.0,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure, axes = plt.subplots(1, 2, figsize=(8.2, 3.45), sharex=True)
    distance = np.asarray(experiment["s_over_c"], dtype=float)
    for panel, (axis_name, pivot, observation_key) in zip(axes, AXES):
        panel.axvspan(0.0, 1.0, color="#E7E1D6", alpha=0.58, zorder=0)
        panel.plot(
            distance,
            experiment[observation_key],
            color="black",
            linewidth=1.7,
            linestyle=(0, (5, 2)),
            label=MODEL_LABELS["experiment"],
        )
        panel.plot(
            distance,
            predictions[axis_name]["fluxv_old"]["CL"],
            color="#0072B2",
            linewidth=1.45,
            label=MODEL_LABELS["fluxv_old"],
        )
        panel.plot(
            distance,
            predictions[axis_name]["fluxv_v4_ldvm_increment"]["CL"],
            color="#009E73",
            linewidth=1.6,
            label=MODEL_LABELS["fluxv_v4_ldvm_increment"],
        )
        panel.plot(
            distance,
            predictions[axis_name]["ldvm_diagnostic"]["CL"],
            color="#D55E00",
            linewidth=1.25,
            linestyle="-.",
            label=MODEL_LABELS["ldvm_diagnostic"],
        )
        panel.axvline(0.05, color="#8A9199", linewidth=0.8, linestyle=":")
        panel.set_title(
            "Leading-edge pitch axis" if pivot == 0.0 else "Mid-chord pitch axis"
        )
        panel.set_xlabel(r"Convected distance $s/c$")
        panel.set_ylabel(r"Lift coefficient $C_L$")
        panel.set_xlim(0.0, 5.0)
        panel.legend(loc="best", frameon=False)
    figure.text(
        0.5,
        0.012,
        (
            "Shading: 0--1c pitch ramp; dotted line: 0.05c sensitivity-window "
            "start. Stevens published lift only; model drag is not scored."
        ),
        ha="center",
        va="bottom",
        fontsize=7.5,
        color="#4D4D4D",
    )
    figure.tight_layout(rect=(0.0, 0.065, 1.0, 1.0))
    path.parent.mkdir(parents=True, exist_ok=True)
    png = path.with_suffix(".png")
    pdf = path.with_suffix(".pdf")
    figure.savefig(png, dpi=220, bbox_inches="tight")
    figure.savefig(pdf, bbox_inches="tight")
    plt.close(figure)
    return [png, pdf]


def _run_predictions(
    quality: str,
) -> tuple[dict[str, dict[str, dict[str, Any]]], dict[str, Any]]:
    """Run both model channels before the experiment CSV is loaded."""

    predictions: dict[str, dict[str, dict[str, Any]]] = {}
    audit: dict[str, Any] = {}
    for axis_name, pivot, _ in AXES:
        axis_start = time.perf_counter()
        old, movement_metadata = run_stevens_fluxv_baseline(
            pivot,
            quality=quality,
            output_samples=501,
        )
        old_runtime = time.perf_counter() - axis_start

        ldvm_start = time.perf_counter()
        ldvm = run_stevens_ldvm_strips(
            pivot,
            lesp_critical=LDVM_DIAGNOSTIC_LESP_CRITICAL,
            steps_per_chord=LDVM_STEPS_PER_CHORD[quality],
            output_samples=501,
        )
        v4 = run_stevens_fluxv_ldvm_increment(
            pivot,
            old,
            lesp_critical=LDVM_DIAGNOSTIC_LESP_CRITICAL,
            threshold_source=(
                "pre-existing thin-flat-plate LDVM diagnostic threshold; "
                "same 0.11 value frozen before the v4 Stevens increment run"
            ),
            steps_per_chord=LDVM_STEPS_PER_CHORD[quality],
        )
        ldvm_runtime = time.perf_counter() - ldvm_start
        predictions[axis_name] = {
            "fluxv_old": old,
            "fluxv_v4_ldvm_increment": v4,
            "ldvm_diagnostic": ldvm,
        }
        audit[axis_name] = {
            "pivot_fraction_chord": pivot,
            "movement": movement_metadata,
            "fluxv_old_runtime_s": old_runtime,
            "ldvm_diagnostic_runtime_s": ldvm_runtime,
            "fluxv_old_particle_count": old["particle_count"],
        }
    return predictions, audit


def run(
    *,
    quality: str,
    output: Path,
    source: Path = STEVENS_FIG21_EXPERIMENT_CSV,
) -> dict[str, Any]:
    if quality not in LDVM_STEPS_PER_CHORD:
        raise ValueError("quality must be 'smoke' or 'full'")
    started_at = datetime.now(timezone.utc)
    wall_start = time.perf_counter()
    output.mkdir(parents=True, exist_ok=True)

    # Deliberately run all predictions before loading the held-out force trace.
    predictions, prediction_audit = _run_predictions(quality)
    experiment = load_stevens_fig21_experiment(source)
    if experiment["drag_available"] or STEVENS_EXPERIMENT_HAS_DRAG:
        raise AssertionError("Stevens experiment must remain lift-only")

    histories_path = output / "stevens2017_lift_histories.csv"
    metrics_path = output / "stevens2017_lift_metrics.csv"
    _write_csv(histories_path, _prediction_rows(predictions, experiment))
    metric_rows = _metric_rows(predictions, experiment)
    _write_csv(metrics_path, metric_rows)
    figure_paths = _plot(
        output / "figures/stevens2017_lift_comparison",
        predictions,
        experiment,
    )

    result_paths = [histories_path, metrics_path, *figure_paths]
    manifest = {
        "run_id": output.name,
        "status": "complete",
        "quality": quality,
        "started_at_utc": started_at.isoformat(),
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_s": time.perf_counter() - wall_start,
        "canonical_command": (
            "PYTHONPATH=platform:src NUMBA_DISABLE_JIT=1 "
            "python -m forward_flight_benchmarks.run_stevens2017_benchmark "
            f"--quality {quality} --output {_relative(output)}"
        ),
        "argv": sys.argv,
        "environment": {
            "python": sys.version,
            "platform": system_platform.platform(),
            "randomness": "none",
        },
        "case": STEVENS_2017.manifest(),
        "axes": [
            {"axis": axis_name, "pivot_fraction_chord": pivot}
            for axis_name, pivot, _ in AXES
        ],
        "experimental_contract": {
            "source": _relative(source),
            "source_figure": 21,
            "observed_channels": STEVENS_EXPERIMENTAL_OBSERVABLES,
            "experimental_drag_available": False,
            "drag_metrics_computed": False,
            "digitization_sha256": experiment["digitization_sha256"],
            "sample_count_per_axis": 501,
            "statistical_independence_claimed": False,
        },
        "scoring_windows": [
            {
                "name": name,
                "start_s_over_c": start,
                "end_s_over_c": end,
                "primary": primary,
            }
            for name, start, end, primary in SCORING_WINDOWS
        ],
        "no_experiment_parameter_tuning": {
            "enforced_for_threshold": True,
            "execution_order": "both model predictions completed before CSV load",
            "ldvm_threshold_selection": (
                "single frozen lesp_critical=0.11 inherited from the pre-existing "
                "ldvm_fourier.py diagnostic default; no Stevens sweep or fit"
            ),
            "development_disclosure": (
                "the additive separated-minus-attached architecture was evaluated "
                "after the first old/LDVM smoke comparison; Stevens is therefore "
                "a development transfer test, not an untouched confirmation set"
            ),
        },
        "models": {
            "fluxv_old": {
                "role": "baseline_prediction",
                "semantics": predictions["leading_edge_axis"]["fluxv_old"][
                    "model_semantics"
                ],
            },
            "ldvm_diagnostic": {
                "role": "diagnostic_only",
                "lesp_critical": LDVM_DIAGNOSTIC_LESP_CRITICAL,
                "steps_per_chord": LDVM_STEPS_PER_CHORD[quality],
                "semantics": predictions["leading_edge_axis"]["ldvm_diagnostic"][
                    "model_semantics"
                ],
                "not_claimed": (
                    "three-dimensional LDVM or UVLM-coupled production model"
                ),
            },
            "fluxv_v4_ldvm_increment": {
                "role": "development_prediction",
                "lesp_critical": LDVM_DIAGNOSTIC_LESP_CRITICAL,
                "semantics": predictions["leading_edge_axis"][
                    "fluxv_v4_ldvm_increment"
                ]["model_semantics"],
                "uvlm_retained": True,
                "not_claimed": (
                    "fully coupled three-dimensional material-LEV wake or an "
                    "independent held-out confirmation"
                ),
            },
        },
        "prediction_audit": prediction_audit,
        "limitations": [
            "Figure 21 publishes lift only; predicted CD has no experimental target",
            "501 curve samples are interpolated and correlated, not independent repeats",
            "the LE-axis startup impulse is digitization/filter sensitive",
            "LDVM diagnostic uses independent 2-D sections and a Prandtl slope ratio",
            "UVLM mean-surface geometry omits the exact rounded-leading-edge section",
        ],
        "source_hashes": {
            _relative(Path(__file__)): _sha256(Path(__file__)),
            _relative(
                REPO_ROOT / "platform/forward_flight_benchmarks/stevens2017.py"
            ): _sha256(REPO_ROOT / "platform/forward_flight_benchmarks/stevens2017.py"),
            _relative(REPO_ROOT / "platform/ldvm_fourier.py"): _sha256(
                REPO_ROOT / "platform/ldvm_fourier.py"
            ),
            _relative(
                REPO_ROOT / "platform/forward_flight_benchmarks/ldvm_uvlm_correction.py"
            ): _sha256(
                REPO_ROOT / "platform/forward_flight_benchmarks/ldvm_uvlm_correction.py"
            ),
            _relative(source): _sha256(source),
        },
        "result_hashes": {_relative(path): _sha256(path) for path in result_paths},
        "headline_primary_lift_rmse": {
            f"{row['axis']}:{row['model']}": row["lift_rmse"]
            for row in metric_rows
            if row["primary_metric"]
        },
    }
    manifest_path = output / "run_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest["headline_primary_lift_rmse"], indent=2))
    print(f"wrote {_relative(output)}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quality", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--source", type=Path, default=STEVENS_FIG21_EXPERIMENT_CSV)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    output = arguments.output
    if output is None:
        output = (
            DOC_ROOT
            / "runs"
            / f"20260812_stevens2017_old_fluxv_ldvm_{arguments.quality}"
        )
    run(quality=arguments.quality, output=output, source=arguments.source)


if __name__ == "__main__":
    main()
