"""Merge independently executed Razak benchmark cases without recomputation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from .razak2014 import RAZAK_2014_CASES
from .run_razak2014_benchmark import (
    DEFAULT_OUTPUT,
    REPO_ROOT,
    SOURCE_CSV,
    _rows,
    _sha256,
    _write_csv,
)


def _repo_path(path: Path) -> str:
    return str(path.resolve().relative_to(REPO_ROOT))


def _history(rows: list[dict[str, str]]) -> dict[str, Any]:
    return {
        "phase": np.asarray([float(row["phase"]) for row in rows]),
        "CL": np.asarray([float(row["CL"]) for row in rows]),
        "CD": np.asarray([float(row["CD"]) for row in rows]),
    }


def _plot_merged(
    source: list[dict[str, str]],
    histories: dict[tuple[int, str], dict[str, Any]],
    output: Path,
) -> list[Path]:
    figures = sorted({figure for figure, _model in histories})
    fig, axes = plt.subplots(
        len(figures), 2, figsize=(10.8, 2.35 * len(figures)), squeeze=False
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
                markersize=1.6,
                linestyle="none",
                color="0.2",
                alpha=0.55,
                elinewidth=0.4,
                capsize=1.0,
                label="Experiment ±1 cycle SD",
                zorder=2,
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
                    linewidth=1.55,
                    label=label,
                    zorder=4,
                )
            case = RAZAK_2014_CASES[figure]
            axis.set_title(
                f"Fig. {figure}: {case.motion_family.replace('_', ' ')}, "
                f"U={case.freestream_m_s:g} m/s, f={case.frequency_hz:g} Hz",
                fontsize=8.5,
            )
            axis.set_ylabel(quantity)
            axis.grid(alpha=0.22)
            if row_index == len(figures) - 1:
                axis.set_xlabel("Displayed-cycle phase (no phase fit)")
            if row_index == 0:
                axis.legend(frameon=False, fontsize=7.5)
    fig.suptitle(
        "Razak–Dimitriadis / Lambert experiments: nominal-motion diagnostic", y=1.0
    )
    fig.tight_layout()
    png = output / "razak2014_old_vs_v4b_phase_loads.png"
    pdf = output / "razak2014_old_vs_v4b_phase_loads.pdf"
    fig.savefig(png, dpi=220, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return [png, pdf]


def _plot_metrics(metric_rows: list[dict[str, str]], output: Path) -> list[Path]:
    figures = sorted({int(row["figure"]) for row in metric_rows})
    x = np.arange(len(figures), dtype=float)
    width = 0.36
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.5), squeeze=False)
    for axis, quantity in zip(axes[0], ("CL", "CD"), strict=True):
        for offset, model, color, label in (
            (-0.5 * width, "fluxv_old", "tab:blue", "FluxV old"),
            (+0.5 * width, "fluxv_v4b", "tab:red", "FluxV v4b transfer"),
        ):
            values = [
                float(
                    next(
                        row["rmse"]
                        for row in metric_rows
                        if int(row["figure"]) == figure
                        and row["quantity"] == quantity
                        and row["model"] == model
                    )
                )
                for figure in figures
            ]
            axis.bar(x + offset, values, width=width, color=color, label=label)
        axis.set_xticks(x, [str(figure) for figure in figures])
        axis.set_xlabel("Source figure")
        axis.set_ylabel(f"{quantity} raw-phase RMSE")
        axis.grid(axis="y", alpha=0.25)
        axis.legend(frameon=False, fontsize=8)
    fig.suptitle("Razak six-case accuracy: improvements and regressions are both shown")
    fig.tight_layout()
    png = output / "razak2014_rmse_by_figure.png"
    pdf = output / "razak2014_rmse_by_figure.pdf"
    fig.savefig(png, dpi=220, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return [png, pdf]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, default=DEFAULT_OUTPUT.parent)
    parser.add_argument("--quality", choices=("smoke", "full"), default="smoke")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT.parent / "20260813_razak2014_sixcase_smoke",
    )
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
    input_files: list[Path] = []
    metric_rows: list[dict[str, str]] = []
    phase_rows: list[dict[str, str]] = []
    history_rows: list[dict[str, str]] = []
    histories: dict[tuple[int, str], dict[str, Any]] = {}
    case_summaries: dict[str, Any] = {}

    for figure in args.figures:
        case_dir = (
            args.input_root.resolve() / f"20260813_razak2014_fig{figure}_{args.quality}"
        )
        metrics_path = case_dir / "accuracy_metrics.csv"
        phases_path = case_dir / "scored_phase_samples.csv"
        histories_path = case_dir / "model_phase_histories.csv"
        summary_path = case_dir / "summary.json"
        required = (metrics_path, phases_path, histories_path, summary_path)
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"missing per-case artifacts: {missing}")
        input_files.extend(required)
        metric_rows.extend(_rows(metrics_path))
        phase_rows.extend(_rows(phases_path))
        rows = _rows(histories_path)
        history_rows.extend(rows)
        for model in ("old", "v4b"):
            selected = [row for row in rows if row["model"] == model]
            if not selected:
                raise ValueError(f"Figure {figure} has no {model} history rows")
            histories[(figure, model)] = _history(selected)
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        case_summaries[str(figure)] = {
            "run_id": summary["run_id"],
            "summary_sha256": _sha256(summary_path),
            "case_manifest": summary["case_manifests"][str(figure)],
        }

    metrics_path = output / "accuracy_metrics.csv"
    phases_path = output / "scored_phase_samples.csv"
    histories_path = output / "model_phase_histories.csv"
    _write_csv(metrics_path, metric_rows)
    _write_csv(phases_path, phase_rows)
    _write_csv(histories_path, history_rows)
    figure_paths = [
        *_plot_merged(_rows(SOURCE_CSV), histories, output),
        *_plot_metrics(metric_rows, output),
    ]

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
                np.mean([float(row["rmse"]) for row in selected])
            )
            aggregate[model][f"macro_case_mae_{quantity}"] = float(
                np.mean([float(row["mae"]) for row in selected])
            )

    result_files = (metrics_path, phases_path, histories_path, *figure_paths)
    summary = {
        "run_id": output.name,
        "quality": args.quality,
        "status": "merged_nominal_motion_transfer_diagnostic",
        "figures": list(args.figures),
        "aggregate": aggregate,
        "case_summaries": case_summaries,
        "metric_contract": (
            "raw displayed-cycle phase; no phase/amplitude fit; visible error-bar "
            "centres; equal-case macro average"
        ),
        "limitations": [
            "merged from isolated case processes to avoid process-level resource exit",
            "measured 64-point flap/pitch histories are not public",
            "nominal sinusoidal kinematics and quarter-chord pitch axis are assumptions",
            "v4b LDVM uses 512 internal steps and a 256-step wake retention window",
            "direct 128-step LDVM transfer was singular for Figure 13 and is rejected",
            f"this {args.quality} matrix is not by itself a convergence demonstration",
        ],
        "input_hashes": {_repo_path(path): _sha256(path) for path in input_files},
        "source_hashes": {_repo_path(Path(__file__)): _sha256(Path(__file__))},
        "result_hashes": {path.name: _sha256(path) for path in result_files},
    }
    summary_path = output / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
