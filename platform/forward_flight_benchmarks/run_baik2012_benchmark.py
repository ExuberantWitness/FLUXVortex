"""Run Baik 2012 W1--W4 against corrected-total experimental loads."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from .baik2012 import (
    BAIK_2012_CASES,
    apply_declared_v4b_transfer,
    run_baik_old_fluxv,
    sharp_fourier_lowpass,
)
from .run_provenance import (
    baik_transfer_dependency_paths,
    collect_run_provenance,
    collect_source_hashes,
    prepare_output_directory,
    sha256_file,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DOC_ROOT = REPO_ROOT / "docs/forward_flight_large_pitch/reproductions/baik2012_w1_w4"
SOURCE_DIR = DOC_ROOT / "source_data"
EXPERIMENT_CSV = SOURCE_DIR / "baik2012_w1_w4_corrected_total_cl_cd.csv"
THEODORSEN_CSV = SOURCE_DIR / "baik2012_fig528_531_theodorsen_lift_markers.csv"
DEFAULT_OUTPUT = DOC_ROOT / "runs/20260813_baik2012_w1_w4_smoke"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _cyclic_phase_error(predicted: float, observed: float) -> float:
    return float((predicted - observed + 0.5) % 1.0 - 0.5)


def _metrics(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    truth = np.asarray(observed, dtype=float)
    estimate = np.asarray(predicted, dtype=float)
    if truth.shape != estimate.shape or truth.ndim != 1 or truth.size < 8:
        raise ValueError("metric histories must be aligned one-dimensional cycles")
    error = estimate - truth
    span = float(np.ptp(truth))
    correlation = float(np.corrcoef(truth, estimate)[0, 1])
    observed_max = int(np.argmax(truth))
    predicted_max = int(np.argmax(estimate))
    observed_min = int(np.argmin(truth))
    predicted_min = int(np.argmin(estimate))
    n = truth.size
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "bias": float(np.mean(error)),
        "max_abs_error": float(np.max(np.abs(error))),
        "range_nrmse": float(np.sqrt(np.mean(error**2)) / span)
        if span > 0.0
        else np.nan,
        "correlation": correlation,
        "observed_mean": float(np.mean(truth)),
        "predicted_mean": float(np.mean(estimate)),
        "observed_max": float(truth[observed_max]),
        "predicted_max": float(estimate[predicted_max]),
        "max_phase_error_cycle": _cyclic_phase_error(
            predicted_max / n, observed_max / n
        ),
        "observed_min": float(truth[observed_min]),
        "predicted_min": float(estimate[predicted_min]),
        "min_phase_error_cycle": _cyclic_phase_error(
            predicted_min / n, observed_min / n
        ),
    }


def _experiment_by_case(
    rows: list[dict[str, str]], case_id: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    selected = [row for row in rows if row["case"] == case_id]
    phase = np.asarray([float(row["phase_t_over_T"]) for row in selected])
    cl = np.asarray([float(row["cl"]) for row in selected])
    cd = np.asarray([float(row["cd"]) for row in selected])
    if selected and np.isclose(phase[-1], 1.0) and np.isclose(phase[0], 0.0):
        # The endpoint is retained in source_data for plotting and trapezoidal
        # mean QA, but is removed from cycle metrics so phase zero is not
        # double weighted.
        phase, cl, cd = phase[:-1], cl[:-1], cd[:-1]
    if phase.size != 400 or not np.allclose(np.diff(phase), 1.0 / 400.0):
        raise ValueError(f"unexpected experimental phase grid for {case_id}")
    return phase, cl, cd


def _theodorsen_history(
    rows: list[dict[str, str]], case_id: str, phase: np.ndarray
) -> np.ndarray:
    selected = [row for row in rows if row["case"] == case_id]
    source_phase = np.asarray([float(row["phase_t_over_T"]) for row in selected])
    source_cl = np.asarray([float(row["cl"]) for row in selected])
    if source_phase.size not in (28, 29):
        raise ValueError(
            f"expected 28 or 29 unique standard-Theodorsen markers for {case_id}"
        )
    order = np.argsort(source_phase)
    return np.interp(phase, source_phase[order], source_cl[order], period=1.0)


def _filter_history(history: dict[str, Any], maximum_harmonic: int) -> dict[str, Any]:
    # This is a coefficient-only scoring view.  Copying the whole force
    # dictionary would leave lift_n/drag_n and mean_* inconsistent with the
    # filtered CL/CD histories.
    filtered: dict[str, Any] = {
        "phase": np.asarray(history["phase"], dtype=float).copy()
    }
    for quantity in ("CL", "CD"):
        filtered[quantity] = sharp_fourier_lowpass(
            np.asarray(history[quantity], dtype=float),
            maximum_harmonic=maximum_harmonic,
        )
    filtered["filter_maximum_harmonic"] = maximum_harmonic
    filtered[
        "filter_semantics"
    ] = "ideal one-cycle Fourier low-pass matching 1 Hz source processing"
    return filtered


def _score_model(
    case_id: str,
    model: str,
    phase: np.ndarray,
    experiment_cl: np.ndarray,
    experiment_cd: np.ndarray,
    history: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    metric_rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    model_phase = np.asarray(history["phase"], dtype=float)
    for quantity, observed in (("CL", experiment_cl), ("CD", experiment_cd)):
        predicted = np.interp(
            phase,
            model_phase,
            np.asarray(history[quantity], dtype=float),
            period=1.0,
        )
        metric_rows.append(
            {
                "case_id": case_id,
                "model": model,
                "quantity": quantity,
                "observation_count": observed.size,
                **_metrics(observed, predicted),
            }
        )
        sample_rows.extend(
            {
                "case_id": case_id,
                "model": model,
                "quantity": quantity,
                "phase": float(this_phase),
                "experiment": float(this_observed),
                "prediction": float(this_predicted),
            }
            for this_phase, this_observed, this_predicted in zip(
                phase, observed, predicted, strict=True
            )
        )
    return metric_rows, sample_rows


def _plot_main(
    experiments: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    histories: dict[tuple[str, str], dict[str, Any]],
    theodorsen: dict[str, np.ndarray],
    output: Path,
) -> list[Path]:
    case_ids = tuple(experiments)
    fig, axes = plt.subplots(
        len(case_ids),
        2,
        figsize=(11.2, 2.9 * len(case_ids)),
        sharex=True,
        squeeze=False,
    )
    for row_index, case_id in enumerate(case_ids):
        phase, experiment_cl, experiment_cd = experiments[case_id]
        case = BAIK_2012_CASES[case_id]
        for column_index, (quantity, observed) in enumerate(
            (("CL", experiment_cl), ("CD", experiment_cd))
        ):
            axis = axes[row_index, column_index]
            axis.fill_between(
                phase,
                observed - 0.02,
                observed + 0.02,
                color="0.75",
                alpha=0.30,
                linewidth=0.0,
                label="Experiment ±0.02" if row_index == 0 else None,
            )
            axis.plot(phase, observed, color="black", linewidth=1.5, label="Experiment")
            for model, style, color, label in (
                ("fluxv_old_filtered", "--", "tab:blue", "FluxV old (1 Hz)"),
                ("fluxv_v4b_filtered", "-", "tab:red", "FluxV v4b (1 Hz)"),
            ):
                history = histories[(case_id, model)]
                axis.plot(
                    history["phase"],
                    history[quantity],
                    style,
                    color=color,
                    linewidth=1.3,
                    label=label,
                )
            if quantity == "CL":
                axis.plot(
                    phase,
                    theodorsen[case_id],
                    color="tab:green",
                    linestyle=":",
                    linewidth=1.0,
                    label="Published standard Theodorsen",
                )
            axis.axhline(0.0, color="0.5", linewidth=0.45)
            axis.grid(alpha=0.22)
            axis.set_ylabel(rf"${quantity[0]}_{{{quantity[1]}}}$")
            axis.set_title(
                f"{case_id}: k={case.reduced_frequency:g}, h₀/c={case.heave_to_chord:g}, "
                f"θ₀={case.pitch_amplitude_deg:g}°",
                fontsize=9,
            )
            if row_index == len(case_ids) - 1:
                axis.set_xlabel("t/T (no phase fit)")
            if row_index == 0:
                axis.legend(frameon=False, fontsize=7.2, ncol=2)
    fig.suptitle(
        "Baik W1–W4 corrected-total loads: experiment vs FluxV old/v4b",
        y=0.998,
    )
    fig.tight_layout()
    paths = [
        output / "baik2012_w1_w4_filtered_old_v4b.png",
        output / "baik2012_w1_w4_filtered_old_v4b.pdf",
    ]
    fig.savefig(paths[0], dpi=220, bbox_inches="tight")
    fig.savefig(paths[1], bbox_inches="tight")
    plt.close(fig)
    return paths


def _plot_filter_diagnostic(
    histories: dict[tuple[str, str], dict[str, Any]], output: Path
) -> list[Path]:
    case_ids = tuple(dict.fromkeys(case_id for case_id, _model in histories))
    fig, axes = plt.subplots(
        len(case_ids),
        2,
        figsize=(11.0, 2.7 * len(case_ids)),
        sharex=True,
        squeeze=False,
    )
    for row_index, case_id in enumerate(case_ids):
        for column_index, quantity in enumerate(("CL", "CD")):
            axis = axes[row_index, column_index]
            for model, color, label in (
                ("fluxv_old", "tab:blue", "FluxV old"),
                ("fluxv_v4b", "tab:red", "FluxV v4b"),
            ):
                raw = histories[(case_id, model)]
                filtered = histories[(case_id, f"{model}_filtered")]
                axis.plot(
                    raw["phase"],
                    raw[quantity],
                    color=color,
                    alpha=0.32,
                    linewidth=0.8,
                    label=f"{label} raw",
                )
                axis.plot(
                    filtered["phase"],
                    filtered[quantity],
                    color=color,
                    linewidth=1.3,
                    label=f"{label} 1 Hz",
                )
            axis.grid(alpha=0.2)
            axis.set_title(f"{case_id} {quantity}", fontsize=9)
            if row_index == len(case_ids) - 1:
                axis.set_xlabel("t/T")
            if column_index == 0:
                axis.set_ylabel(quantity)
            if row_index == 0:
                axis.legend(frameon=False, fontsize=7.0, ncol=2)
    fig.suptitle("Raw numerical histories and source-matched 1 Hz filtering", y=0.998)
    fig.tight_layout()
    paths = [
        output / "baik2012_w1_w4_filter_diagnostic.png",
        output / "baik2012_w1_w4_filter_diagnostic.pdf",
    ]
    fig.savefig(paths[0], dpi=220, bbox_inches="tight")
    fig.savefig(paths[1], bbox_inches="tight")
    plt.close(fig)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quality", choices=("smoke", "full"), default="smoke")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help=(
            "new/empty result directory; required for full-quality and subset runs "
            "to prevent a misleading smoke/W1-W4 default label"
        ),
    )
    parser.add_argument(
        "--allow-existing-output",
        action="store_true",
        help="explicitly permit replacing same-name files in a non-empty output directory",
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        choices=tuple(BAIK_2012_CASES),
        default=tuple(BAIK_2012_CASES),
    )
    parser.add_argument("--ldvm-steps-per-cycle", type=int)
    parser.add_argument("--ldvm-max-wake-steps", type=int)
    parser.add_argument("--skip-lesp-source-sensitivity", action="store_true")
    args = parser.parse_args()

    all_cases = tuple(BAIK_2012_CASES)
    if args.output_dir is None:
        if args.quality != "smoke" or tuple(args.cases) != all_cases:
            parser.error("--output-dir is required for --quality full or a case subset")
        requested_output = DEFAULT_OUTPUT
    else:
        requested_output = args.output_dir
    output = requested_output.resolve()
    parsed_arguments = vars(args) | {"resolved_output_dir": output}
    provenance = collect_run_provenance(REPO_ROOT, parsed_arguments)
    output = prepare_output_directory(output, allow_existing=args.allow_existing_output)
    experiment_rows = _read_csv(EXPERIMENT_CSV)
    theodorsen_rows = _read_csv(THEODORSEN_CSV)
    ldvm_steps = args.ldvm_steps_per_cycle or (128 if args.quality == "smoke" else 512)
    wake_steps = args.ldvm_max_wake_steps or (64 if args.quality == "smoke" else 256)

    experiments: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    histories: dict[tuple[str, str], dict[str, Any]] = {}
    theodorsen: dict[str, np.ndarray] = {}
    metric_rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    history_rows: list[dict[str, Any]] = []
    case_manifests: dict[str, Any] = {}

    for case_id in args.cases:
        case = BAIK_2012_CASES[case_id]
        print(f"{case_id}: running old FluxV ({args.quality})", flush=True)
        old, movement, movement_metadata = run_baik_old_fluxv(
            case, args.quality, output_samples=128
        )
        print(f"{case_id}: applying declared v4b, Lcrit=0.11", flush=True)
        v4b = apply_declared_v4b_transfer(
            case,
            old,
            movement,
            output_samples=128,
            ldvm_steps_per_cycle=ldvm_steps,
            ldvm_max_wake_steps=wake_steps,
            lesp_critical=0.11,
        )
        source_conflict = None
        if not args.skip_lesp_source_sensitivity:
            print(f"{case_id}: applying Table-4.1 Lcrit=0.19 sensitivity", flush=True)
            source_conflict = apply_declared_v4b_transfer(
                case,
                old,
                movement,
                output_samples=128,
                ldvm_steps_per_cycle=ldvm_steps,
                ldvm_max_wake_steps=wake_steps,
                lesp_critical=0.19,
            )

        maximum_harmonic = case.experimental_filter_harmonic
        old_filtered = _filter_history(old, maximum_harmonic)
        v4b_filtered = _filter_history(v4b, maximum_harmonic)
        histories[(case_id, "fluxv_old")] = old
        histories[(case_id, "fluxv_old_filtered")] = old_filtered
        histories[(case_id, "fluxv_v4b")] = v4b
        histories[(case_id, "fluxv_v4b_filtered")] = v4b_filtered
        if source_conflict is not None:
            histories[(case_id, "fluxv_v4b_lcrit019")] = source_conflict
            histories[(case_id, "fluxv_v4b_lcrit019_filtered")] = _filter_history(
                source_conflict, maximum_harmonic
            )

        phase, experiment_cl, experiment_cd = _experiment_by_case(
            experiment_rows, case_id
        )
        experiments[case_id] = (phase, experiment_cl, experiment_cd)
        theodorsen[case_id] = _theodorsen_history(theodorsen_rows, case_id, phase)
        for model in (
            "fluxv_old",
            "fluxv_old_filtered",
            "fluxv_v4b",
            "fluxv_v4b_filtered",
        ):
            metrics, samples = _score_model(
                case_id,
                model,
                phase,
                experiment_cl,
                experiment_cd,
                histories[(case_id, model)],
            )
            metric_rows.extend(metrics)
            sample_rows.extend(samples)
        if source_conflict is not None:
            for model in ("fluxv_v4b_lcrit019", "fluxv_v4b_lcrit019_filtered"):
                metrics, samples = _score_model(
                    case_id,
                    model,
                    phase,
                    experiment_cl,
                    experiment_cd,
                    histories[(case_id, model)],
                )
                metric_rows.extend(metrics)
                sample_rows.extend(samples)

        metric_rows.append(
            {
                "case_id": case_id,
                "model": "published_standard_theodorsen_digitized",
                "quantity": "CL",
                "observation_count": phase.size,
                **_metrics(experiment_cl, theodorsen[case_id]),
            }
        )
        sample_rows.extend(
            {
                "case_id": case_id,
                "model": "published_standard_theodorsen_digitized",
                "quantity": "CL",
                "phase": float(this_phase),
                "experiment": float(this_observed),
                "prediction": float(this_predicted),
            }
            for this_phase, this_observed, this_predicted in zip(
                phase,
                experiment_cl,
                theodorsen[case_id],
                strict=True,
            )
        )
        case_manifests[case_id] = {
            "case": case.manifest(),
            "movement": movement_metadata,
            "old_runtime_s": old["runtime_s"],
            "source_cycle_step_range": old["source_cycle_step_range"],
            "filter_maximum_harmonic": maximum_harmonic,
            "v4b_primary_lesp": v4b["lesp_provenance"],
            "v4b_ldvm_steps_per_cycle": ldvm_steps,
            "v4b_ldvm_max_wake_steps": wake_steps,
            "v4b_ldvm_settings": {
                "integration_steps_per_cycle": v4b["ldvm_steps_per_cycle"],
                "integration_cycles": 3,
                "delta_time_convective": v4b["ldvm_delta_time_convective"],
                "section": v4b["ldvm_settings"],
            },
            "v4b_mean_persistence": float(np.mean(v4b["persistence"])),
            "v4b_mean_shedding_fraction": float(np.mean(v4b["ldvm_shedding"])),
            "source_conflict_sensitivity": source_conflict["lesp_provenance"]
            if source_conflict is not None
            else None,
        }
        print(f"{case_id}: complete", flush=True)

    for (case_id, model), history in histories.items():
        for index, phase_value in enumerate(np.asarray(history["phase"], dtype=float)):
            history_rows.append(
                {
                    "case_id": case_id,
                    "model": model,
                    "phase": float(phase_value),
                    "CL": float(history["CL"][index]),
                    "CD": float(history["CD"][index]),
                    "persistence": float(history["persistence"][index])
                    if "persistence" in history
                    else "",
                    "ldvm_delta_CL": float(history["ldvm_delta_CL"][index])
                    if "ldvm_delta_CL" in history
                    else "",
                    "ldvm_delta_CD": float(history["ldvm_delta_CD"][index])
                    if "ldvm_delta_CD" in history
                    else "",
                    "ldvm_shedding": float(history["ldvm_shedding"][index])
                    if "ldvm_shedding" in history
                    else "",
                }
            )

    metrics_path = output / "accuracy_metrics.csv"
    samples_path = output / "scored_phase_samples.csv"
    histories_path = output / "model_phase_histories.csv"
    _write_csv(metrics_path, metric_rows)
    _write_csv(samples_path, sample_rows)
    _write_csv(histories_path, history_rows)
    figure_paths = _plot_main(experiments, histories, theodorsen, output)
    figure_paths.extend(_plot_filter_diagnostic(histories, output))

    primary_models = (
        "fluxv_old_filtered",
        "fluxv_v4b_filtered",
        "published_standard_theodorsen_digitized",
    )
    aggregate: dict[str, dict[str, float]] = {}
    for model in primary_models:
        aggregate[model] = {}
        for quantity in ("CL", "CD"):
            selected = [
                row
                for row in metric_rows
                if row["model"] == model and row["quantity"] == quantity
            ]
            if not selected:
                continue
            aggregate[model][f"macro_case_rmse_{quantity}"] = float(
                np.mean([row["rmse"] for row in selected])
            )
            aggregate[model][f"macro_case_mae_{quantity}"] = float(
                np.mean([row["mae"] for row in selected])
            )
            aggregate[model][f"macro_case_bias_{quantity}"] = float(
                np.mean([row["bias"] for row in selected])
            )

    result_paths = (metrics_path, samples_path, histories_path, *figure_paths)
    direct_sources = (
        Path(__file__).resolve(),
        Path(__file__).with_name("run_provenance.py").resolve(),
        *baik_transfer_dependency_paths(REPO_ROOT),
        EXPERIMENT_CSV,
        THEODORSEN_CSV,
        SOURCE_DIR / "DIGITIZATION_AND_PROVENANCE.md",
    )
    summary = {
        "run_id": output.name,
        "quality": args.quality,
        "status": "development_transfer_validation_not_held_out",
        "cases": list(args.cases),
        "aggregate": aggregate,
        "case_manifests": case_manifests,
        "metric_contract": (
            "400 unique equally-spaced experimental phase samples; no phase/amplitude/"
            "offset fit; model filtered with the source's ideal 1 Hz Fourier cutoff; "
            "case metrics followed by equal-case macro average"
        ),
        "ground_truth": (
            "Baik dissertation Figures 5.24--5.27 corrected-total direct-force histories"
        ),
        "scientific_scope": [
            "W1--W4 are quasi-2D wall/endplate water-channel cases at Re=5000",
            "the UVLM adapter is a physical-span free-tip zero-camber mean surface",
            "6.25% thickness, rounded edges, viscosity and wall images are unresolved",
            "Lcrit=0.11 is a cross-Re/thickness source transfer, not a Baik fit",
            "the Lcrit=0.19 Table-4.1 conflict is reported only as sensitivity",
            "this benchmark was inspected during model development and is not held out",
        ],
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "package_versions": provenance["environment"]["package_versions"],
        },
        "provenance": {
            **provenance,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        },
        "source_hashes": collect_source_hashes(REPO_ROOT, direct_sources),
        "result_hashes": {path.name: sha256_file(path) for path in result_paths},
    }
    summary_path = output / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(aggregate, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
