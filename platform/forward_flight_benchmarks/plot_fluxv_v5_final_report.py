"""Render the final, fail-closed FluxV v5a/v5b development outcome."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from .plot_fluxv_v5_joint_report import (
    _extract_v5a_comparison,
    _read_csv,
    _read_json,
    _validate_v5a_summary,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DOC_ROOT = (
    REPO_ROOT
    / "docs/forward_flight_large_pitch/reproductions/fluxv_v5_nextgen_20260814"
)
DEFAULT_V5A = DOC_ROOT / "runs/20260814_fluxv_v5a_cache_smoke_frozen"
DEFAULT_V5B = DOC_ROOT / "runs/20260814_fluxv_v5b_force_gate_reproducible"
DEFAULT_OUTPUT = DOC_ROOT / "runs/20260814_fluxv_v5_joint_final_verified"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_force_gate(summary: dict[str, Any]) -> None:
    expected = {
        "status": "no_go_before_crosspaper",
        "crosspaper_performance_status": "blocked_not_scored",
        "promotion_passed": False,
        "paper_results": None,
    }
    for key, value in expected.items():
        if summary.get(key) != value:
            raise ValueError(
                f"final report expects force-gate {key}={value!r}; "
                f"got {summary.get(key)!r}"
            )
    names = [row["gate"] for row in summary.get("gates", [])]
    required = {
        "G1_current_FluxV_no_LEV_exact_reduction",
        "G4_single_surface_pressure_force_owner",
        "G5_smooth_birth_limit",
        "G6_Ramesh_high_AR_force_parity",
    }
    if set(names) != required:
        raise ValueError(
            "force-gate summary does not contain the frozen G1/G4/G5/G6 set"
        )


def _style() -> None:
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


def _save(figure: Any, output: Path, stem: str) -> tuple[Path, Path]:
    png = output / f"{stem}.png"
    pdf = output / f"{stem}.pdf"
    figure.savefig(
        pdf,
        bbox_inches="tight",
        pad_inches=0.04,
        metadata={
            "Creator": "FluxV v5 benchmark plotter",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    figure.savefig(png, dpi=300, bbox_inches="tight", pad_inches=0.04)
    plt.close(figure)
    return png, pdf


def _plot_outcome(
    comparison: list[dict[str, Any]],
    force_summary: dict[str, Any],
    output: Path,
) -> tuple[Path, Path]:
    colors = {
        "v4b": "#7A7A7A",
        "v5a": "#D55E00",
        "pass": "#009E73",
        "fail": "#CC79A7",
    }
    figure, axes = plt.subplots(1, 2, figsize=(7.25, 3.15), layout="constrained")
    left, right = axes
    position = np.arange(len(comparison))
    width = 0.36
    left.bar(
        position - width / 2,
        np.ones(len(comparison)),
        width,
        color=colors["v4b"],
        edgecolor="black",
        linewidth=0.5,
        label="FluxV v4b",
    )
    ratio = np.asarray([row["v5a_over_v4b"] for row in comparison])
    bars = left.bar(
        position + width / 2,
        ratio,
        width,
        color=colors["v5a"],
        edgecolor="black",
        linewidth=0.5,
        hatch="///",
        label="FluxV v5a (dev proxy)",
    )
    left.axhline(1.0, color="black", linestyle="--", linewidth=0.8)
    left.set_ylabel("Error / FluxV v4b error")
    left.set_xticks(position)
    left.set_xticklabels(
        ["Yang\nlift", "Yang\ndrag", "Fig. 14\nthrust", "Baik\n$C_L$", "Baik\n$C_D$"]
    )
    left.set_ylim(0.0, max(4.15, float(np.max(ratio)) * 1.13))
    left.grid(axis="y", color="#D9D9D9", linewidth=0.5)
    left.legend(frameon=False, loc="upper left")
    left.text(-0.14, 1.02, "(a)", transform=left.transAxes, fontweight="bold")
    for bar, value in zip(bars, ratio, strict=True):
        left.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.06,
            f"{value:.2f}x",
            ha="center",
            va="bottom",
            fontsize=7.5,
        )

    gates = force_summary["gates"]
    labels = [row["gate"].split("_")[0] for row in gates]
    passed = np.asarray([bool(row["passed"]) for row in gates])
    gate_colors = [
        "#A6A6A6"
        if labels[index] == "G6"
        else colors["pass"]
        if value
        else colors["fail"]
        for index, value in enumerate(passed)
    ]
    right.bar(
        np.arange(len(gates)),
        np.ones(len(gates)),
        color=gate_colors,
        edgecolor="black",
        linewidth=0.5,
    )
    right.set_xticks(np.arange(len(gates)))
    right.set_xticklabels(labels)
    right.set_ylim(0.0, 1.45)
    right.set_yticks([0.0, 1.0])
    right.set_yticklabels(["", "Gate evaluated"])
    right.set_ylabel("FluxV v5b promotion gate")
    right.grid(axis="y", color="#D9D9D9", linewidth=0.5)
    right.text(-0.16, 1.02, "(b)", transform=right.transAxes, fontweight="bold")
    annotations = {
        "G1": "FAIL",
        "G4": "INTERNAL\nPASS*",
        "G5": "DEV\nPASS*",
        "G6": "NOT RUN",
    }
    for index, value in enumerate(passed):
        right.text(
            index,
            1.06,
            annotations[labels[index]],
            ha="center",
            va="bottom",
            fontsize=7.5,
            fontweight="bold",
        )
    right.set_title("Promotion gates (*qualified diagnostics)")
    right.text(
        0.98,
        0.04,
        "Three-paper v5b loads:\nBLOCKED / NOT SCORED",
        transform=right.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.0,
        color="#7A1F5C",
        bbox={
            "boxstyle": "round,pad=0.3",
            "facecolor": "white",
            "edgecolor": colors["fail"],
        },
    )
    return _save(figure, output, "fluxv_v5_joint_final_outcome")


def _plot_no_lev_reduction(
    rows: list[dict[str, str]], output: Path
) -> tuple[Path, Path]:
    phase = np.asarray([float(row["phase"]) for row in rows])
    figure, axes = plt.subplots(
        2, 1, figsize=(6.2, 4.5), sharex=True, layout="constrained"
    )
    channels = (
        ("CL", "Lift coefficient, $C_L$"),
        ("CD", "Drag coefficient, $C_D$"),
    )
    for axis, (channel, ylabel) in zip(axes, channels, strict=True):
        old = np.asarray([float(row[f"current_fluxv_{channel}"]) for row in rows])
        v5b = np.asarray([float(row[f"v5b_no_lev_{channel}"]) for row in rows])
        axis.plot(phase, old, color="#0072B2", linewidth=1.8, label="Current FluxV")
        axis.plot(
            phase,
            v5b,
            color="#CC79A7",
            linewidth=1.6,
            linestyle="--",
            label="Standalone v5b, LEV disabled",
        )
        axis.set_ylabel(ylabel)
        axis.grid(color="#D9D9D9", linewidth=0.5)
        axis.legend(frameon=False, loc="best")
    axes[-1].set_xlabel("Cycle phase")
    axes[0].set_title("No-LEV reduction probe: Yang 2025, 15° nominal-four-bar smoke")
    return _save(figure, output, "fluxv_v5b_no_lev_reduction_failure")


def _plot_birth(rows: list[dict[str, str]], output: Path) -> tuple[Path, Path]:
    dt = np.asarray([float(row["dt_s"]) for row in rows])
    gamma = np.asarray([float(row["birth_gamma_max_abs_m2_s"]) for row in rows])
    order = np.argsort(dt)
    dt, gamma = dt[order], gamma[order]
    slope, intercept = np.polyfit(np.log(dt), np.log(gamma), 1)
    fitted = np.exp(intercept) * dt**slope
    reference = gamma[0] * (dt / dt[0])
    figure, axis = plt.subplots(figsize=(4.4, 3.35), layout="constrained")
    axis.loglog(dt, gamma, "o-", color="#009E73", label="Smooth-onset v5b")
    axis.loglog(dt, fitted, "--", color="#0072B2", label=f"Fit: $p={slope:.3f}$")
    axis.loglog(dt, reference, ":", color="#666666", label="$O(\\Delta t)$ reference")
    axis.set_xlabel("Time step, $\\Delta t$ [s]")
    axis.set_ylabel("First LE-ring coefficient magnitude [m$^2$/s]")
    axis.grid(which="both", color="#D9D9D9", linewidth=0.5)
    axis.legend(frameon=False)
    axis.set_title("Smooth LESP-threshold crossing (post-hoc development diagnostic)")
    return _save(figure, output, "fluxv_v5b_smooth_birth_limit")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v5a-run", type=Path, default=DEFAULT_V5A)
    parser.add_argument("--v5b-run", type=Path, default=DEFAULT_V5B)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error("output directory exists and is non-empty; use a new directory")
    output.mkdir(parents=True, exist_ok=True)

    v5a_summary_path = args.v5a_run.resolve() / "summary.json"
    v5a_metrics_path = args.v5a_run.resolve() / "case_metrics.csv"
    v5b_summary_path = args.v5b_run.resolve() / "summary.json"
    reduction_path = args.v5b_run.resolve() / "no_lev_current_fluxv_comparison.csv"
    birth_path = args.v5b_run.resolve() / "smooth_birth_refinement.csv"
    v5a_summary = _read_json(v5a_summary_path)
    v5b_summary = _read_json(v5b_summary_path)
    _validate_v5a_summary(v5a_summary)
    _validate_force_gate(v5b_summary)
    comparison = _extract_v5a_comparison(_read_csv(v5a_metrics_path))
    reduction = _read_csv(reduction_path)
    birth = _read_csv(birth_path)
    _style()
    outputs = [
        *_plot_outcome(comparison, v5b_summary, output),
        *_plot_no_lev_reduction(reduction, output),
        *_plot_birth(birth, output),
    ]
    inputs = [
        v5a_summary_path,
        v5a_metrics_path,
        v5b_summary_path,
        reduction_path,
        birth_path,
        Path(__file__).resolve(),
    ]
    latex = output / "latex_includes.tex"
    latex.write_text(
        "\\includegraphics[width=\\linewidth]{fluxv_v5_joint_final_outcome.pdf}\n"
        "\\includegraphics[width=\\linewidth]{fluxv_v5b_no_lev_reduction_failure.pdf}\n"
        "\\includegraphics[width=0.72\\linewidth]{fluxv_v5b_smooth_birth_limit.pdf}\n",
        encoding="utf-8",
    )
    manifest = {
        "manifest_version": 2,
        "run_id": output.name,
        "status": "v5a_rejected_v5b_no_go_before_crosspaper",
        "crosspaper_v5b_metrics": None,
        "input_hashes": {
            str(path.resolve().relative_to(REPO_ROOT)): _sha256(path) for path in inputs
        },
        "figure_hashes": {path.name: _sha256(path) for path in outputs},
        "auxiliary_hashes": {latex.name: _sha256(latex)},
        "v5a_error_ratios": {row["label"]: row["v5a_over_v4b"] for row in comparison},
        "v5b_gates": v5b_summary["gates"],
    }
    manifest_path = output / "figure_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {"output": str(output), "figures": [path.name for path in outputs]},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
