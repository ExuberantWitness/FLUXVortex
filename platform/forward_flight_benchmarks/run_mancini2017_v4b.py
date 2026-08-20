"""Run the frozen FluxV-v4b transfer on Mancini (2017) Figure 4.13b."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
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
from matplotlib import font_manager
from matplotlib.patches import Rectangle

from .mancini2017 import (
    FROZEN_V4B_LESP_CRITICAL,
    MANCINI_2017_CASES,
    MANCINI_EXPERIMENT_HAS_SPANWISE_LOADS,
    MANCINI_FIG4_13B_CSV,
    apply_frozen_mancini_v4b,
    load_mancini_fig4_13b_experiment,
    run_mancini_fluxv_baseline,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DOC_ROOT = (
    REPO_ROOT / "docs/forward_flight_large_pitch/reproductions/mancini2017_v4b_20260820"
)
DEFAULT_OUTPUT = DOC_ROOT / "runs/20260820_mancini2017_v4b_smoke"
V4B_STEPS_PER_CHORD = {"smoke": 24, "full": 96}
CASE_LABELS_ZH = {"fast_pitch": "快速俯仰", "slow_pitch": "慢速俯仰"}
CHINESE_FONT_CANDIDATES = (
    Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
    Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT).as_posix()


def _configure_chinese_font() -> str:
    for font_path in CHINESE_FONT_CANDIDATES:
        if not font_path.is_file():
            continue
        font_manager.fontManager.addfont(font_path)
        family = font_manager.FontProperties(fname=font_path).get_name()
        plt.rcParams["font.family"] = family
        plt.rcParams["font.sans-serif"] = [family, "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
        return family
    raise RuntimeError("缺少可用于中文证据图的 Noto Sans CJK 字体")


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
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _metrics(
    t_star: np.ndarray,
    experiment: np.ndarray,
    prediction: np.ndarray,
) -> dict[str, float]:
    if not (t_star.shape == experiment.shape == prediction.shape):
        raise ValueError("metric arrays must be aligned")
    if not np.all(np.isfinite(t_star + experiment + prediction)):
        raise ValueError("metric arrays must be finite")
    error = prediction - experiment
    experiment_peak = int(np.argmax(experiment))
    prediction_peak = int(np.argmax(prediction))
    correlation = float(np.corrcoef(experiment, prediction)[0, 1])
    return {
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "bias": float(np.mean(error)),
        "correlation": correlation,
        "experiment_peak_CL": float(experiment[experiment_peak]),
        "experiment_peak_t_star": float(t_star[experiment_peak]),
        "prediction_peak_CL": float(prediction[prediction_peak]),
        "prediction_peak_t_star": float(t_star[prediction_peak]),
    }


def _run_predictions(
    quality: str,
    case_ids: tuple[str, ...],
) -> tuple[dict[str, dict[str, dict[str, Any]]], dict[str, Any], list[str]]:
    predictions: dict[str, dict[str, dict[str, Any]]] = {}
    audit: dict[str, Any] = {}
    events: list[str] = []
    for case_id in case_ids:
        case = MANCINI_2017_CASES[case_id]
        events.append(f"START {case_id} FluxV UVLM")
        started = time.perf_counter()
        baseline, movement = run_mancini_fluxv_baseline(case, quality=quality)
        old_runtime = time.perf_counter() - started
        events.append(f"END {case_id} FluxV UVLM runtime_s={old_runtime:.6f}")

        events.append(f"START {case_id} frozen v4b")
        started = time.perf_counter()
        v4b = apply_frozen_mancini_v4b(
            case,
            baseline,
            steps_per_chord=V4B_STEPS_PER_CHORD[quality],
        )
        v4b_runtime = time.perf_counter() - started
        events.append(f"END {case_id} frozen v4b runtime_s={v4b_runtime:.6f}")
        predictions[case_id] = {"fluxv_uvlm": baseline, "fluxv_v4b": v4b}
        audit[case_id] = {
            "case": case.manifest(),
            "movement": movement,
            "fluxv_uvlm_runtime_s": old_runtime,
            "fluxv_v4b_runtime_s": v4b_runtime,
            "v4b_steps_per_chord": V4B_STEPS_PER_CHORD[quality],
            "v4b_lesp_critical": FROZEN_V4B_LESP_CRITICAL,
        }
    return predictions, audit, events


def _rows_and_metrics(
    predictions: dict[str, dict[str, dict[str, Any]]],
    experiment: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    full_t = np.asarray(experiment["t_star"], dtype=float)
    observed_by_case = {
        "fast_pitch": np.asarray(experiment["CL_fast_pitch"], dtype=float),
        "slow_pitch": np.asarray(experiment["CL_slow_pitch"], dtype=float),
    }
    full_rows: list[dict[str, Any]] = []
    scored_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    primary = full_t <= 5.0 + 1.0e-12
    for case_id, observation in observed_by_case.items():
        for index, t_star in enumerate(full_t):
            full_rows.append(
                {
                    "case_id": case_id,
                    "t_star": float(t_star),
                    "experiment_CL": float(observation[index]),
                    "data_role": experiment["data_role"],
                }
            )
        if case_id not in predictions:
            continue
        target = np.asarray(predictions[case_id]["fluxv_uvlm"]["t_star"], dtype=float)
        observed = np.interp(target, full_t, observation)
        for model, history in predictions[case_id].items():
            predicted = np.asarray(history["CL"], dtype=float)
            metrics = _metrics(target, observed, predicted)
            metric_rows.append(
                {
                    "case_id": case_id,
                    "case_label_zh": CASE_LABELS_ZH[case_id],
                    "model": model,
                    "window": "0<=t_star<=5",
                    "observation_count": int(np.count_nonzero(primary)),
                    **metrics,
                }
            )
            for index, t_star in enumerate(target):
                scored_rows.append(
                    {
                        "case_id": case_id,
                        "model": model,
                        "t_star": float(t_star),
                        "pitch_deg": float(history["pitch_deg"][index]),
                        "experiment_CL": float(observed[index]),
                        "prediction_CL": float(predicted[index]),
                        "prediction_CD_diagnostic_only": float(history["CD"][index]),
                        "ldvm_delta_CL": (
                            float(history["delta_CL"][index])
                            if "delta_CL" in history
                            else 0.0
                        ),
                        "shed_lev": (
                            bool(history["shed_lev"][index])
                            if "shed_lev" in history
                            else False
                        ),
                    }
                )
    for case_id in predictions:
        selected = [row for row in metric_rows if row["case_id"] == case_id]
        by_model = {row["model"]: row for row in selected}
        old_rmse = float(by_model["fluxv_uvlm"]["rmse"])
        v4b_rmse = float(by_model["fluxv_v4b"]["rmse"])
        for row in selected:
            row["v4b_rmse_change_vs_uvlm_percent"] = (
                100.0 * (v4b_rmse - old_rmse) / old_rmse
            )
            row["v4b_improves_uvlm"] = v4b_rmse < old_rmse
    return full_rows, scored_rows, metric_rows


def _plot_lift(
    output: Path,
    predictions: dict[str, dict[str, dict[str, Any]]],
    experiment: dict[str, Any],
) -> list[Path]:
    _configure_chinese_font()
    fig, axes = plt.subplots(1, len(predictions), figsize=(6.0 * len(predictions), 4.0))
    if len(predictions) == 1:
        axes = np.asarray([axes])
    full_t = np.asarray(experiment["t_star"], dtype=float)
    for axis, case_id in zip(axes, predictions, strict=True):
        observation = np.asarray(experiment[f"CL_{case_id}"], dtype=float)
        mask = full_t <= 5.0 + 1.0e-12
        axis.plot(
            full_t[mask],
            observation[mask],
            color="black",
            linewidth=2.0,
            label="实验（Figure 4.13b 图线提取）",
        )
        styles = (
            ("fluxv_uvlm", "--", "#0072B2", "未增强有限翼 UVLM"),
            ("fluxv_v4b", "-", "#D55E00", "冻结 FluxV v4b"),
        )
        for model, linestyle, color, label in styles:
            history = predictions[case_id][model]
            axis.plot(
                history["t_star"],
                history["CL"],
                linestyle,
                color=color,
                linewidth=1.7,
                label=label,
            )
        case = MANCINI_2017_CASES[case_id]
        axis.axvspan(
            0.0,
            min(case.acceleration_distance_chords, 5.0),
            color="#F0E442",
            alpha=0.12,
            label="俯仰运动区间" if case_id == next(iter(predictions)) else None,
        )
        axis.set_title(
            f"{CASE_LABELS_ZH[case_id]}：$s_a/c={case.acceleration_distance_chords:g}$，"
            f"$k={case.reduced_pitch_rate:g}$"
        )
        axis.set_xlabel(r"对流时间 $t^*=tU_\infty/c$（弦向尺度）")
        axis.set_ylabel(r"整翼升力系数 $C_L$")
        axis.set_xlim(0.0, 5.0)
        axis.grid(alpha=0.25)
        axis.legend(frameon=False, fontsize=8)
    fig.suptitle("Mancini 2017：AR=4 有限翼快速机动的 v4b 直接迁移")
    fig.tight_layout()
    png = output / "figures/mancini2017_lift_comparison_zh.png"
    pdf = output / "figures/mancini2017_lift_comparison_zh.pdf"
    png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png, dpi=220, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return [png, pdf]


def _plot_geometry(output: Path) -> list[Path]:
    _configure_chinese_font()
    case = MANCINI_2017_CASES["fast_pitch"]
    fig, axis = plt.subplots(figsize=(6.7, 4.2))
    axis.add_patch(
        Rectangle((0.0, -0.5), 1.0, 1.0, facecolor="#D9EAF7", edgecolor="black")
    )
    axis.axhline(0.375, color="#CC79A7", linestyle="--", linewidth=1.5)
    axis.text(
        0.52,
        0.39,
        "PIV/染料平面：论文称 3/4 展向位置",
        color="#8A3B71",
        fontsize=9,
    )
    axis.annotate(
        "",
        xy=(0.92, -0.38),
        xytext=(0.15, -0.38),
        arrowprops={"arrowstyle": "->", "lw": 1.5},
        va="center",
    )
    axis.annotate(
        "",
        xy=(0.12, 0.45),
        xytext=(0.12, -0.38),
        arrowprops={"arrowstyle": "->", "lw": 1.5},
        ha="center",
    )
    axis.text(0.50, -0.35, "弦向 $x/c$", ha="center", va="bottom", fontsize=10)
    axis.text(
        0.15,
        0.02,
        "展向 $y/b$",
        ha="left",
        va="center",
        rotation=90,
        fontsize=10,
    )
    axis.scatter([0.0], [0.0], color="#D55E00", zorder=3)
    axis.text(0.02, 0.02, "前缘俯仰轴", color="#D55E00", fontsize=9)
    axis.set_xlim(-0.08, 1.08)
    axis.set_ylim(-0.62, 0.62)
    axis.set_xlabel(r"弦向坐标 $x/c$")
    axis.set_ylabel(r"展向坐标 $y/b$")
    axis.set_title(
        f"实验有限翼平面形：$c={case.chord_m * 1000:.1f}$ mm，"
        f"$b={case.span_m * 1000:.1f}$ mm，AR={case.aspect_ratio:g}"
    )
    axis.set_aspect(0.75)
    axis.grid(alpha=0.18)
    fig.tight_layout()
    png = output / "figures/mancini2017_geometry_chord_span_zh.png"
    pdf = output / "figures/mancini2017_geometry_chord_span_zh.pdf"
    png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png, dpi=220, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return [png, pdf]


def run(*, quality: str, case_ids: tuple[str, ...], output: Path) -> dict[str, Any]:
    if quality not in V4B_STEPS_PER_CHORD:
        raise ValueError("quality must be smoke or full")
    if not case_ids or any(case_id not in MANCINI_2017_CASES for case_id in case_ids):
        raise ValueError("unknown or empty Mancini case selection")
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(timezone.utc)
    wall_start = time.perf_counter()

    # Freeze predictions before loading the experimental force curves.
    predictions, prediction_audit, events = _run_predictions(quality, case_ids)
    experiment = load_mancini_fig4_13b_experiment()
    if experiment["spanwise_loads_available"] or MANCINI_EXPERIMENT_HAS_SPANWISE_LOADS:
        raise AssertionError("Mancini Figure 4.13 must remain whole-wing lift only")

    experiment_rows, scored_rows, metric_rows = _rows_and_metrics(
        predictions, experiment
    )
    experiment_path = output / "digitized_experiment.csv"
    scored_path = output / "scored_samples.csv"
    metrics_path = output / "accuracy_metrics.csv"
    _write_csv(experiment_path, experiment_rows)
    _write_csv(scored_path, scored_rows)
    _write_csv(metrics_path, metric_rows)
    figures = [*_plot_lift(output, predictions, experiment), *_plot_geometry(output)]

    log_path = output / "run.log"
    log_path.write_text("\n".join(events) + "\n", encoding="utf-8")
    result_paths = [experiment_path, scored_path, metrics_path, log_path, *figures]
    source_paths = (
        Path(__file__).resolve(),
        Path(__file__).with_name("mancini2017.py").resolve(),
        Path(__file__).with_name("digitize_mancini2017.py").resolve(),
        Path(__file__).with_name("ldvm_uvlm_correction.py").resolve(),
        REPO_ROOT / "platform/warp_vpm/bing_v4b_refined.py",
        MANCINI_FIG4_13B_CSV,
    )
    by_case_model = {f"{row['case_id']}:{row['model']}": row for row in metric_rows}
    headline = {}
    for case_id in case_ids:
        old = by_case_model[f"{case_id}:fluxv_uvlm"]
        v4b = by_case_model[f"{case_id}:fluxv_v4b"]
        headline[case_id] = {
            "fluxv_uvlm_rmse_CL": old["rmse"],
            "fluxv_v4b_rmse_CL": v4b["rmse"],
            "v4b_rmse_change_vs_uvlm_percent": v4b["v4b_rmse_change_vs_uvlm_percent"],
            "v4b_improves_uvlm": v4b["v4b_improves_uvlm"],
        }
    summary = {
        "run_id": output.name,
        "status": "complete_direct_transfer_diagnostic",
        "quality": quality,
        "case_ids": list(case_ids),
        "headline": headline,
        "experimental_contract": {
            "paper": "Mancini 2017 dissertation",
            "source_figure": "4.13b",
            "source_csv": _relative(MANCINI_FIG4_13B_CSV),
            "digitization_sha256": experiment["digitization_sha256"],
            "observable": "whole-wing CL(t*)",
            "spanwise_load_distribution_available": False,
            "digitization_uncertainty_note": (
                "raster curve extraction; adjacent 0.01 samples are correlated and "
                "no pointwise experimental confidence interval is synthesized"
            ),
        },
        "interpretation_gate": (
            "can-run and improves-UVLM are separate; no threshold/phase/amplitude fit; "
            "a whole-wing CL match does not validate spanwise load distribution"
        ),
        "prediction_audit": prediction_audit,
        "source_hashes": {_relative(path): _sha256(path) for path in source_paths},
        "result_hashes": {_relative(path): _sha256(path) for path in result_paths},
    }
    summary_path = output / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "run_id": output.name,
        "status": summary["status"],
        "started_at_utc": started_at.isoformat(),
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_s": time.perf_counter() - wall_start,
        "canonical_command": (
            "PYTHONPATH=platform:src NUMBA_CACHE_DIR=/tmp/numba-mancini "
            "MPLCONFIGDIR=/tmp/mpl-mancini "
            + (
                f"NUMBA_DISABLE_JIT={os.environ['NUMBA_DISABLE_JIT']} "
                if "NUMBA_DISABLE_JIT" in os.environ
                else ""
            )
            + "python -m "
            "forward_flight_benchmarks.run_mancini2017_v4b "
            f"--quality {quality} --cases {' '.join(case_ids)} --output {_relative(output)}"
        ),
        "argv": sys.argv,
        "environment": {
            "python": sys.version,
            "platform": system_platform.platform(),
            "randomness": "none",
            "numba_disable_jit": os.environ.get("NUMBA_DISABLE_JIT", "unset"),
            "experiment_skill_governance": (
                "fallback terminal used because bash_exec/artifact/memory tools are unavailable"
            ),
        },
        "summary_sha256": _sha256(summary_path),
        "result_files": sorted(
            [_relative(path) for path in [*result_paths, summary_path]]
        ),
    }
    manifest_path = output / "run_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {"summary": summary, "manifest": manifest}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quality", choices=("smoke", "full"), default="smoke")
    parser.add_argument(
        "--cases",
        nargs="+",
        choices=tuple(MANCINI_2017_CASES),
        default=("fast_pitch", "slow_pitch"),
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    result = run(
        quality=arguments.quality,
        case_ids=tuple(arguments.cases),
        output=arguments.output,
    )
    print(json.dumps(result["summary"]["headline"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
