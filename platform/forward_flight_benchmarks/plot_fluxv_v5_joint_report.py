"""Plot the fail-closed FluxV v5a/v5b joint stage summary.

The left panel compares the frozen development-only v5a cache smoke with the
frozen v4b values stored in the same ``case_metrics.csv``.  The right panel
reports only v5b G0--G2 mechanical gate counts.  Because the current v5b
shared-wake smoke has no pressure/force coupling, this script deliberately has
no code path that plots v5b Yang/Figure-14/Baik accuracy.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[2]
DOC_ROOT = (
    REPO_ROOT
    / "docs/forward_flight_large_pitch/reproductions/fluxv_v5_nextgen_20260814"
)
DEFAULT_V5A_RUN = DOC_ROOT / "runs/20260814_fluxv_v5a_cache_smoke_frozen"
DEFAULT_V5B_RUN = DOC_ROOT / "runs/20260814_fluxv_v5b_no_force_smoke_frozen"
DEFAULT_OUTPUT = DOC_ROOT / "runs/20260814_fluxv_v5_joint_report_skeleton"
ARTIFACT_NAMES = (
    "fluxv_v5_joint_stage_summary.pdf",
    "fluxv_v5_joint_stage_summary.png",
    "plot_data.csv",
    "figure_manifest.json",
    "latex_includes.tex",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _single_row(
    rows: list[dict[str, str]],
    *,
    benchmark: str,
    case_id: str,
    model: str,
    quantity: str,
    view: str,
) -> dict[str, str]:
    matches = [
        row
        for row in rows
        if row["benchmark"] == benchmark
        and row["case_id"] == case_id
        and row["model"] == model
        and row["quantity"] == quantity
        and row["view"] == view
    ]
    if len(matches) != 1:
        raise ValueError(
            "expected exactly one metric row for "
            f"{benchmark}/{case_id}/{model}/{quantity}/{view}; got {len(matches)}"
        )
    return matches[0]


def _extract_v5a_comparison(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    """Extract five preregistered headline channels from the frozen metric table."""

    specifications = (
        ("Yang lift", "yang2025", "six_aoa", "lift_gf", "cycle_mean", "mae"),
        ("Yang drag", "yang2025", "six_aoa", "drag_gf", "cycle_mean", "mae"),
        (
            "Figure 14 thrust",
            "izraelevitz2017_fig14",
            "all14",
            "CT",
            "cycle_mean",
            "rmse",
        ),
    )
    extracted: list[dict[str, Any]] = []
    for label, benchmark, case_id, quantity, view, metric in specifications:
        model_values: dict[str, float] = {}
        observation_count = 0
        for model in ("fluxv_v4b", "fluxv_v5a"):
            row = _single_row(
                rows,
                benchmark=benchmark,
                case_id=case_id,
                model=model,
                quantity=quantity,
                view=view,
            )
            model_values[model] = float(row[metric])
            observation_count = int(row["observation_count"])
        extracted.append(
            {
                "panel": "v5a_stop",
                "label": label,
                "benchmark": benchmark,
                "quantity": quantity,
                "error_metric": metric,
                "observation_scope": str(observation_count),
                "v4b_error": model_values["fluxv_v4b"],
                "v5a_error": model_values["fluxv_v5a"],
                "v5a_over_v4b": model_values["fluxv_v5a"] / model_values["fluxv_v4b"],
                "lower_is_better": True,
            }
        )

    for quantity in ("CL", "CD"):
        model_values: dict[str, list[float]] = defaultdict(list)
        sample_count = 0
        for case_id in ("W1", "W2", "W3", "W4"):
            for model in ("fluxv_v4b", "fluxv_v5a"):
                row = _single_row(
                    rows,
                    benchmark="baik2012",
                    case_id=case_id,
                    model=model,
                    quantity=quantity,
                    view="filtered_1hz",
                )
                model_values[model].append(float(row["rmse"]))
                sample_count += int(row["observation_count"])
        v4b_error = sum(model_values["fluxv_v4b"]) / 4.0
        v5a_error = sum(model_values["fluxv_v5a"]) / 4.0
        extracted.append(
            {
                "panel": "v5a_stop",
                "label": f"Baik {quantity}",
                "benchmark": "baik2012",
                "quantity": quantity,
                "error_metric": "macro_rmse_filtered_1hz",
                "observation_scope": f"4 waveforms x {sample_count // 8} samples",
                "v4b_error": v4b_error,
                "v5a_error": v5a_error,
                "v5a_over_v4b": v5a_error / v4b_error,
                "lower_is_better": True,
            }
        )
    return extracted


def _validate_v5a_summary(summary: dict[str, Any]) -> None:
    expected = {
        "status": "partial_development_only",
        "canonical_eligible": False,
        "incidence_source": "kinematic_proxy",
        "aggregation_scope": "projected_integrated_proxy",
    }
    for key, value in expected.items():
        if summary.get(key) != value:
            raise ValueError(
                f"v5a joint plot expects frozen development-only contract "
                f"{key}={value!r}; got {summary.get(key)!r}"
            )
    gates = summary.get("gate_counts", {})
    if gates.get("promotion_pass") != 0:
        raise ValueError("v5a stop plot requires zero promotion gates passed")


def _validate_no_force_v5b(summary: dict[str, Any]) -> None:
    expected = {
        "force_coupling": "not_implemented",
        "crosspaper_performance_status": "blocked_not_scored",
        "evidence_role": "topology_and_conservation_only_no_force",
    }
    for key, value in expected.items():
        if summary.get(key) != value:
            raise ValueError(
                f"no-force v5b plot requires {key}={value!r}; "
                f"got {summary.get(key)!r}"
            )
    prohibited_metric_keys = {
        "accuracy_metrics",
        "crosspaper_metrics",
        "headline",
        "load_metrics",
        "paper_metrics",
    }
    present = prohibited_metric_keys.intersection(summary)
    if present:
        raise ValueError(
            "refusing to plot a no-force summary containing cross-paper metric keys: "
            + ", ".join(sorted(present))
        )


def _extract_mechanical_gates(summary: dict[str, Any]) -> list[dict[str, Any]]:
    _validate_no_force_v5b(summary)
    by_level = summary.get("gate_counts", {}).get("by_level", {})
    expected_levels = ("G0", "G1", "G2")
    if tuple(by_level) != expected_levels:
        raise ValueError(
            f"expected ordered G0--G2 gate counts; got {tuple(by_level)!r}"
        )
    rows: list[dict[str, Any]] = []
    for level in expected_levels:
        passed = int(by_level[level]["passed"])
        total = int(by_level[level]["total"])
        if passed < 0 or total <= 0 or passed > total:
            raise ValueError(f"invalid {level} gate count: {passed}/{total}")
        rows.append(
            {
                "panel": "v5b_mechanical_gates",
                "label": level,
                "passed": passed,
                "failed": total - passed,
                "total": total,
                "force_evidence": False,
                "crosspaper_performance_status": "blocked_not_scored",
            }
        )
    return rows


def _write_plot_data(
    path: Path,
    comparison: list[dict[str, Any]],
    gates: list[dict[str, Any]],
) -> None:
    fieldnames = (
        "panel",
        "label",
        "benchmark",
        "quantity",
        "error_metric",
        "observation_scope",
        "v4b_error",
        "v5a_error",
        "v5a_over_v4b",
        "lower_is_better",
        "passed",
        "failed",
        "total",
        "force_evidence",
        "crosspaper_performance_status",
    )
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in comparison + gates:
            writer.writerow(row)


def _render(
    comparison: list[dict[str, Any]],
    gates: list[dict[str, Any]],
    output: Path,
) -> tuple[Path, Path]:
    plt.rcParams.update(
        {
            "font.size": 9.0,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "axes.labelsize": 9.0,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "legend.fontsize": 8.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    colors = {
        "v4b": "#7A7A7A",
        "v5a": "#D55E00",
        "pass": "#009E73",
        "fail": "#CC79A7",
        "blocked": "#E69F00",
    }
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(7.25, 3.15),
        gridspec_kw={"width_ratios": [1.45, 1.0], "wspace": 0.33},
    )

    left = axes[0]
    positions = list(range(len(comparison)))
    width = 0.36
    v4b = left.bar(
        [position - width / 2 for position in positions],
        [1.0] * len(comparison),
        width,
        color=colors["v4b"],
        edgecolor="black",
        linewidth=0.5,
        label="FluxV v4b",
    )
    v5a_values = [float(row["v5a_over_v4b"]) for row in comparison]
    v5a = left.bar(
        [position + width / 2 for position in positions],
        v5a_values,
        width,
        color=colors["v5a"],
        edgecolor="black",
        linewidth=0.5,
        hatch="///",
        label="FluxV v5a (dev proxy)",
    )
    left.axhline(1.0, color="black", linewidth=0.8, linestyle="--", zorder=0)
    left.set_ylabel("Error / FluxV v4b error")
    left.set_xticks(positions)
    left.set_xticklabels(
        [
            "Yang\nlift",
            "Yang\ndrag",
            "Fig. 14\nthrust",
            "Baik\n$C_L$",
            "Baik\n$C_D$",
        ]
    )
    left.set_ylim(0.0, max(4.15, max(v5a_values) * 1.12))
    left.grid(axis="y", color="#D9D9D9", linewidth=0.5, zorder=0)
    left.legend(frameon=False, loc="upper left")
    left.text(-0.16, 1.02, "(a)", transform=left.transAxes, fontweight="bold")
    for bar, value in zip(v5a, v5a_values, strict=True):
        left.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.06,
            f"{value:.2f}x",
            ha="center",
            va="bottom",
            fontsize=7.5,
        )
    for bar in v4b:
        bar.set_zorder(2)

    right = axes[1]
    levels = [row["label"] for row in gates]
    passed = [int(row["passed"]) for row in gates]
    failed = [int(row["failed"]) for row in gates]
    gate_positions = list(range(len(gates)))
    right.bar(
        gate_positions,
        passed,
        color=colors["pass"],
        edgecolor="black",
        linewidth=0.5,
        label="Passed mechanical gates",
    )
    right.bar(
        gate_positions,
        failed,
        bottom=passed,
        color=colors["fail"],
        edgecolor="black",
        linewidth=0.5,
        hatch="xx",
        label="Failed mechanical gates",
    )
    right.set_ylabel("v5b mechanical-only gate count")
    right.set_xticks(gate_positions)
    right.set_xticklabels(levels)
    right.set_ylim(0.0, max(row["total"] for row in gates) * 1.42)
    right.grid(axis="y", color="#D9D9D9", linewidth=0.5, zorder=0)
    right.text(-0.22, 1.02, "(b)", transform=right.transAxes, fontweight="bold")
    for position, row in zip(gate_positions, gates, strict=True):
        right.text(
            position,
            row["total"] + 0.25,
            f"{row['passed']}/{row['total']}",
            ha="center",
            va="bottom",
            fontsize=8.0,
        )
    right.text(
        0.98,
        0.96,
        "Cross-paper loads:\nBLOCKED / NOT SCORED\n(no force coupling)",
        transform=right.transAxes,
        ha="right",
        va="top",
        fontsize=7.7,
        color="#7A4A00",
        bbox={
            "boxstyle": "round,pad=0.3",
            "facecolor": "white",
            "edgecolor": colors["blocked"],
            "linewidth": 0.9,
        },
    )

    png = output / "fluxv_v5_joint_stage_summary.png"
    pdf = output / "fluxv_v5_joint_stage_summary.pdf"
    figure.savefig(pdf, bbox_inches="tight", pad_inches=0.04)
    figure.savefig(png, dpi=300, bbox_inches="tight", pad_inches=0.04)
    plt.close(figure)
    return pdf, png


def generate_joint_figure(
    v5a_run: Path,
    v5b_run: Path,
    output: Path,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    existing = [output / name for name in ARTIFACT_NAMES if (output / name).exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "refusing to overwrite joint-report artifacts: "
            + ", ".join(str(path) for path in existing)
        )
    output.mkdir(parents=True, exist_ok=True)

    v5a_summary_path = v5a_run / "summary.json"
    v5a_metrics_path = v5a_run / "case_metrics.csv"
    v5b_summary_path = v5b_run / "summary.json"
    v5a_summary = _read_json(v5a_summary_path)
    v5b_summary = _read_json(v5b_summary_path)
    _validate_v5a_summary(v5a_summary)
    comparison = _extract_v5a_comparison(_read_csv(v5a_metrics_path))
    gates = _extract_mechanical_gates(v5b_summary)

    plot_data = output / "plot_data.csv"
    _write_plot_data(plot_data, comparison, gates)
    pdf, png = _render(comparison, gates, output)
    latex = output / "latex_includes.tex"
    latex.write_text(
        "% FluxV v5a/v5b joint stage summary (fail-closed)\n"
        "\\begin{figure}[t]\n"
        "  \\centering\n"
        "  \\includegraphics[width=0.95\\textwidth]{"
        "fluxv_v5_joint_stage_summary.pdf}\n"
        "  \\caption{FluxV v5 staged evidence. (a) Frozen development-only "
        "v5a error normalized by v4b error; lower is better. Metrics are Yang "
        "cycle-mean MAE, Figure 14 all-14 CT RMSE, and Baik 1-Hz-filtered "
        "waveform macro RMSE. (b) v5b G0--G2 topology/conservation gate "
        "counts. The v5b shared-wake smoke has no force coupling, so "
        "cross-paper load accuracy is blocked and not scored. Sample scopes "
        "are Yang: 6 conditions; Figure 14: 14 points; Baik: 4 waveforms with "
        "400 unique phase samples each.}\n"
        "  \\label{fig:fluxv-v5-joint-stage}\n"
        "\\end{figure}\n",
        encoding="utf-8",
    )

    manifest = {
        "run_id": output.name,
        "status": "joint_report_skeleton_no_v5b_crosspaper_score",
        "v5a_status": v5a_summary["status"],
        "v5a_canonical_eligible": v5a_summary["canonical_eligible"],
        "v5a_incidence_source": v5a_summary["incidence_source"],
        "v5a_promotion_gates_passed": v5a_summary["gate_counts"]["promotion_pass"],
        "v5b_status": v5b_summary["status"],
        "v5b_force_coupling": v5b_summary["force_coupling"],
        "v5b_crosspaper_performance_status": v5b_summary[
            "crosspaper_performance_status"
        ],
        "v5b_blocked_reason": v5b_summary["blocked_reason"],
        "v5b_sequence_crosspaper": {
            "status": "pending_not_supplied",
            "metrics": None,
            "note": "must remain empty until a force-coupled sequence run is supplied",
        },
        "plotted_v5a_rows": comparison,
        "plotted_v5b_gate_rows": gates,
        "source_hashes": {
            str(Path(__file__).resolve().relative_to(REPO_ROOT)): _sha256(
                Path(__file__).resolve()
            ),
            str(v5a_summary_path.relative_to(REPO_ROOT)): _sha256(v5a_summary_path),
            str(v5a_metrics_path.relative_to(REPO_ROOT)): _sha256(v5a_metrics_path),
            str(v5b_summary_path.relative_to(REPO_ROOT)): _sha256(v5b_summary_path),
        },
        "artifact_hashes": {
            pdf.name: _sha256(pdf),
            png.name: _sha256(png),
            plot_data.name: _sha256(plot_data),
            latex.name: _sha256(latex),
        },
    }
    manifest_path = output / "figure_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v5a-run", type=Path, default=DEFAULT_V5A_RUN)
    parser.add_argument("--v5b-run", type=Path, default=DEFAULT_V5B_RUN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="overwrite only this script's five named artifacts",
    )
    return parser


def main() -> None:
    arguments = _parser().parse_args()
    manifest = generate_joint_figure(
        arguments.v5a_run.resolve(),
        arguments.v5b_run.resolve(),
        arguments.output_dir.resolve(),
        overwrite=arguments.overwrite,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
