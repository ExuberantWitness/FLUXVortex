"""Publication plots for the three-gate periodic-v3 regression."""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DOC_ROOT = (
    REPO_ROOT / "docs/forward_flight_large_pitch/reproductions/"
    "unified_fluxv_upgrade_20260812"
)
DEFAULT_RUN = DOC_ROOT / "runs/20260812_periodic_v3_persistent_full"
DEFAULT_BASE = DOC_ROOT / "runs/20260812_periodic_v2_ullt_full"
DEFAULT_FIG14_BASE = DOC_ROOT / "runs/20260812_scherer_fig14_experiment_full"

# Okabe--Ito: colorblind-safe, with line style/marker redundancy for print.
COLORS = {
    "reference": "#000000",
    "author": "#009E73",
    "author_alt": "#CC79A7",
    "old": "#0072B2",
    "v1": "#E69F00",
    "v2": "#D55E00",
    "v3": "#7B2CBF",
    "ablation": "#666666",
}
STYLES: dict[str, dict[str, Any]] = {
    "scherer_1968_experiment": dict(
        color=COLORS["reference"], marker="s", ls="none", ms=4.5
    ),
    "authors_6state_ullt": dict(color=COLORS["author_alt"], ls=(0, (6, 2)), lw=1.4),
    "authors_1state_ullt": dict(color=COLORS["author"], ls="--", lw=1.6),
    "wind_tunnel_test": dict(color=COLORS["reference"], marker="o", lw=1.8, ms=4.5),
    "authors_proposed_modified_uvlm": dict(
        color=COLORS["author"], marker="s", ls="--", lw=1.5, ms=4.0
    ),
    "paper_uvlm": dict(color=COLORS["reference"], lw=2.0),
    "1_state": dict(color=COLORS["author"], ls="--", lw=1.5),
    "fluxv_uvpm": dict(color=COLORS["old"], ls="-.", lw=1.6),
    "fluxv_periodic_v1": dict(color=COLORS["v1"], ls=":", lw=1.8),
    "fluxv_periodic_v2": dict(color=COLORS["v2"], ls=(0, (5, 2)), lw=1.8),
    "fluxv_periodic_v3_persistent": dict(color=COLORS["v3"], lw=2.3),
    "fluxv_periodic_v3_mean_passthrough": dict(
        color=COLORS["ablation"], ls=(0, (2, 1)), lw=1.6, marker="^", ms=3.2
    ),
}
LABELS = {
    "scherer_1968_experiment": "Scherer experiment",
    "authors_6state_ullt": "Authors' 6-state ULLT",
    "authors_1state_ullt": "Authors' 1-state ULLT",
    "wind_tunnel_test": "Wind-tunnel test",
    "authors_proposed_modified_uvlm": "Authors' modified UVLM",
    "paper_uvlm": "Authors' numerical UVLM reference",
    "1_state": "Authors' 1-state ULLT",
    "fluxv_uvpm": "FluxV old",
    "fluxv_periodic_v1": "FluxV v1",
    "fluxv_periodic_v2": "FluxV v2",
    "fluxv_periodic_v3_persistent": "FluxV v3 persistent owner",
    "fluxv_periodic_v3_mean_passthrough": ("Old mean + 3/4c $C_{d0}$ (ablation)"),
}


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
            "font.size": 9,
            "axes.labelsize": 10,
            "legend.fontsize": 7.7,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "lines.solid_capstyle": "round",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 300,
        }
    )


def _finish(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.04)
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def _panel_labels(axes: np.ndarray | list[plt.Axes]) -> None:
    for index, axis in enumerate(np.asarray(axes, dtype=object).ravel()):
        axis.text(
            0.015,
            0.975,
            f"({chr(ord('a') + index)})",
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontweight="bold",
        )


def plot_fig14(run: Path, base: Path, output: Path) -> None:
    rows = _rows(base / "mean_thrust_vs_phase.csv")
    v3_rows = _rows(run / "izraelevitz2017_fig14_v3_mean_thrust.csv")
    models = (
        "authors_6state_ullt",
        "authors_1state_ullt",
        "fluxv_periodic_v3_mean_passthrough",
        "fluxv_periodic_v3_persistent",
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.25, 3.45), sharex=True, sharey=True)
    legend: dict[str, Any] = {}
    for axis, theta in zip(axes, (15.0, 25.0)):
        observations = [
            row
            for row in rows
            if row["data_role"] == "experimental_observation"
            and np.isclose(float(row["theta_max_deg"]), theta)
        ]
        artist = axis.errorbar(
            [float(row["phase_offset_deg"]) for row in observations],
            [float(row["CT"]) for row in observations],
            yerr=np.asarray(
                [
                    [float(row["CT_error_minus"]) for row in observations],
                    [float(row["CT_error_plus"]) for row in observations],
                ]
            ),
            mfc="white",
            mec=COLORS["reference"],
            mew=0.9,
            ecolor=COLORS["reference"],
            elinewidth=0.8,
            capsize=2.0,
            zorder=10,
            **STYLES["scherer_1968_experiment"],
        )
        legend.setdefault(LABELS["scherer_1968_experiment"], artist)
        for model in models:
            source = v3_rows if model.startswith("fluxv_periodic_v3_") else rows
            key = "model" if source is v3_rows else "series"
            selected = sorted(
                (
                    row
                    for row in source
                    if row[key] == model
                    and np.isclose(float(row["theta_max_deg"]), theta)
                ),
                key=lambda row: float(row["phase_offset_deg"]),
            )
            if not selected:
                continue
            line = axis.plot(
                [float(row["phase_offset_deg"]) for row in selected],
                [float(row["CT"]) for row in selected],
                **STYLES[model],
            )[0]
            legend.setdefault(LABELS[model], line)
        axis.axhline(0.0, color="#AAAAAA", lw=0.7, zorder=0)
        axis.set_xlabel(r"Pitch--heave phase offset $\psi$ [deg]")
        axis.set_xlim(10.0, 110.0)
        axis.set_xticks(np.arange(15.0, 106.0, 15.0))
        axis.grid(color="#D9D9D9", lw=0.5, alpha=0.8)
        axis.text(
            0.98,
            0.97,
            rf"$\theta_{{\max}}={theta:.0f}^\circ$",
            transform=axis.transAxes,
            ha="right",
            va="top",
        )
    axes[0].set_ylabel(r"Cycle-mean thrust coefficient $\overline{C_T}$ [-]")
    axes[0].set_ylim(-0.28, 0.34)
    _panel_labels(axes)
    fig.legend(
        legend.values(),
        legend.keys(),
        loc="upper center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 1.02),
        columnspacing=0.8,
        handlelength=2.2,
    )
    fig.text(
        0.5,
        0.02,
        r"All local FluxV curves use the same $3/4c$ $C_{d0}=0.057$ ledger",
        ha="center",
        fontsize=7,
    )
    fig.subplots_adjust(top=0.78, bottom=0.20, left=0.095, right=0.99, wspace=0.10)
    _finish(fig, output / "izraelevitz2017_fig14_v3_experiment_comparison")


def plot_yang(run: Path, base: Path, output: Path) -> None:
    rows = _rows(base / "yang2025_mean_characteristics.csv")
    v3_rows = _rows(run / "yang2025_v3_mean_characteristics.csv")
    models = (
        "wind_tunnel_test",
        "authors_proposed_modified_uvlm",
        "fluxv_uvpm",
        "fluxv_periodic_v2",
        "fluxv_periodic_v3_persistent",
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.25, 3.35), sharex=True)
    for model in models:
        source = v3_rows if model == "fluxv_periodic_v3_persistent" else rows
        selected = sorted(
            (row for row in source if row["model"] == model),
            key=lambda row: float(row["aoa_deg"]),
        )
        for axis, field in zip(axes, ("mean_lift_gf", "mean_drag_gf")):
            x = [float(row["aoa_deg"]) for row in selected]
            y = [float(row[field]) for row in selected]
            if model in ("wind_tunnel_test", "authors_proposed_modified_uvlm"):
                axis.errorbar(
                    x,
                    y,
                    yerr=[
                        float(row["digitization_uncertainty_gf"]) for row in selected
                    ],
                    capsize=2.0,
                    elinewidth=0.8,
                    label=LABELS[model],
                    **STYLES[model],
                )
            else:
                axis.plot(x, y, label=LABELS[model], **STYLES[model])
    axes[0].set_ylabel("Cycle-mean lift [gf]")
    axes[1].set_ylabel(r"Cycle-mean drag $D=-T$ [gf]")
    for axis in axes:
        axis.set_xlabel(r"Installation angle $\alpha_0$ [deg]")
        axis.set_xticks([0, 5, 10, 15, 20, 25])
        axis.axhline(0.0, color="#AAAAAA", lw=0.7, zorder=0)
        axis.grid(color="#D9D9D9", lw=0.5, alpha=0.8)
    _panel_labels(axes)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 1.02),
        columnspacing=1.1,
    )
    fig.text(
        0.5,
        0.02,
        r"Error bars: $\pm0.4$ gf digitization uncertainty; not a statistical CI",
        ha="center",
        fontsize=7,
    )
    fig.subplots_adjust(top=0.78, bottom=0.20, left=0.095, right=0.98, wspace=0.27)
    _finish(fig, output / "yang2025_v3_mean_lift_drag_vs_aoa")


def plot_fig11(run: Path, base: Path, output: Path) -> None:
    rows = _rows(base / "izraelevitz2017_fig11_phase_histories.csv")
    v3_rows = _rows(run / "izraelevitz2017_fig11_v3_phase_histories.csv")
    models = (
        "paper_uvlm",
        "1_state",
        "fluxv_uvpm",
        "fluxv_periodic_v2",
        "fluxv_periodic_v3_persistent",
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.7, 3.5), sharex=True)
    for model in models:
        source = v3_rows if model == "fluxv_periodic_v3_persistent" else rows
        selected = sorted(
            (row for row in source if row["model"] == model),
            key=lambda row: float(row["phase"]),
        )
        for axis, field in zip(axes, ("CL_alpha", "CD_alpha")):
            axis.plot(
                [float(row["phase"]) for row in selected],
                [float(row[field]) for row in selected],
                label=LABELS[model],
                **STYLES[model],
            )
    axes[0].set_ylabel(r"Paper-scaled lift $C_{L\alpha}$ [-]")
    axes[1].set_ylabel(r"Paper-scaled drag $C_{D\alpha}$ [-]")
    for axis in axes:
        axis.set_xlabel(r"Cycle phase $t/T$ [-]")
        axis.set_xlim(0.0, 1.0)
        axis.axhline(0.0, color="#AAAAAA", lw=0.7, zorder=0)
        axis.grid(color="#D9D9D9", lw=0.5, alpha=0.8)
    _panel_labels(axes)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 1.02),
        columnspacing=1.1,
    )
    fig.text(
        0.5,
        0.02,
        "128 plotted phase samples; the paper UVLM curve is numerical, not experimental",
        ha="center",
        fontsize=7,
    )
    fig.subplots_adjust(top=0.78, bottom=0.20, left=0.095, right=0.98, wspace=0.27)
    _finish(fig, output / "izraelevitz2017_fig11_v3_lift_drag_phase")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--base-run", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--fig14-base-run", type=Path, default=DEFAULT_FIG14_BASE)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    base = args.base_run.resolve()
    fig14_base = args.fig14_base_run.resolve()
    output = (
        args.output_dir.resolve() if args.output_dir is not None else run / "figures"
    )
    _style()
    plot_fig14(run, fig14_base, output)
    plot_yang(run, base, output)
    plot_fig11(run, base, output)
    for stem in (
        "izraelevitz2017_fig14_v3_experiment_comparison",
        "yang2025_v3_mean_lift_drag_vs_aoa",
        "izraelevitz2017_fig11_v3_lift_drag_phase",
    ):
        print(output / f"{stem}.png")
        print(output / f"{stem}.pdf")
    hash_path = output / "figure_hashes.csv"
    figure_paths = sorted(output.glob("*.png")) + sorted(output.glob("*.pdf"))
    with hash_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("file", "sha256"))
        writer.writeheader()
        for path in figure_paths:
            writer.writerow({"file": path.name, "sha256": _sha256(path)})
    print(hash_path)
    latex_path = output / "latex_includes.tex"
    latex_path.write_text(
        r"""% Generated by forward_flight_benchmarks.plot_periodic_v3_regression.
\begin{figure}[t]
  \centering
  \includegraphics[width=\linewidth]{izraelevitz2017_fig14_v3_experiment_comparison.pdf}
  \caption{Izraelevitz et al. Figure 14 / Scherer experimental cycle-mean
  thrust. All local FluxV curves use the same three-quarter-chord
  $C_{d0}=0.057$ load ledger.}
\end{figure}

\begin{figure}[t]
  \centering
  \includegraphics[width=\linewidth]{yang2025_v3_mean_lift_drag_vs_aoa.pdf}
  \caption{Yang rigid-wing cycle-mean lift and drag versus installation angle.
  Error bars are digitization uncertainty, not statistical confidence intervals.}
\end{figure}

\begin{figure}[t]
  \centering
  \includegraphics[width=\linewidth]{izraelevitz2017_fig11_v3_lift_drag_phase.pdf}
  \caption{Izraelevitz Figure 11 phase loads. The paper UVLM trace is a
  numerical reference rather than experimental data; 128 phase samples are shown.}
\end{figure}
""",
        encoding="utf-8",
    )
    print(latex_path)


if __name__ == "__main__":
    main()
