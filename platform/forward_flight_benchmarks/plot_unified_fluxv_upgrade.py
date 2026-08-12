"""Publication figures for the cross-paper FluxV upgrade experiment."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN = (
    ROOT
    / "docs/forward_flight_large_pitch/reproductions/"
    "unified_fluxv_upgrade_20260812/runs/20260812_periodic_v2_ullt_full"
)

COLORS = {
    "reference": "#111111",
    "author": "#009E73",
    "old": "#0072B2",
    "v1": "#E69F00",
    "v2": "#D55E00",
    "ullt": "#7B2CBF",
    "free": "#7F7F7F",
    "qs": "#CC79A7",
}

STYLES: dict[str, dict[str, Any]] = {
    "wind_tunnel_test": dict(color=COLORS["reference"], marker="o", lw=1.8, ms=4.5),
    "authors_proposed_modified_uvlm": dict(color=COLORS["author"], marker="s", lw=1.6, ls="--", ms=4.0),
    "fluxv_uvpm": dict(color=COLORS["old"], lw=1.8, ls="--"),
    "fluxv_periodic_v1": dict(color=COLORS["v1"], lw=1.3, ls=":"),
    "fluxv_periodic_v2": dict(color=COLORS["v2"], lw=2.2),
    "one_state_ullt_local": dict(color=COLORS["ullt"], lw=1.5, ls="-."),
    "ptera_free_wake_uvlm": dict(color=COLORS["free"], lw=1.2, ls=(0, (4, 2))),
    "paper_uvlm": dict(color=COLORS["reference"], lw=2.1, zorder=10),
    "6_state": dict(color=COLORS["author"], lw=1.2, ls=(0, (6, 2))),
    "1_state": dict(color=COLORS["author"], lw=1.5, ls="--"),
    "qs_added_mass": dict(color=COLORS["qs"], lw=1.2, ls=":"),
}

LABELS = {
    "wind_tunnel_test": "Wind-tunnel test",
    "authors_proposed_modified_uvlm": "Authors' modified UVLM",
    "fluxv_uvpm": "FluxV old",
    "fluxv_periodic_v1": "FluxV v1",
    "fluxv_periodic_v2": "FluxV v2",
    "one_state_ullt_local": "Local 1-state ULLT",
    "ptera_free_wake_uvlm": "Ptera free-wake UVLM",
    "paper_uvlm": "Authors' UVLM reference",
    "6_state": "Authors' 6-State ULLT",
    "1_state": "Authors' 1-State ULLT",
    "qs_added_mass": "QS + added mass",
}


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _style() -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.labelsize": 10,
            "legend.fontsize": 8,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.linewidth": 0.8,
            "lines.solid_capstyle": "round",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _finish(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def _panel_labels(axes: np.ndarray | list[plt.Axes]) -> None:
    flat = np.asarray(axes, dtype=object).ravel()
    for index, axis in enumerate(flat):
        axis.text(
            0.015,
            0.975,
            f"({chr(ord('a') + index)})",
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontweight="bold",
        )


def plot_yang_means(run: Path, figures: Path) -> None:
    rows = _rows(run / "yang2025_mean_characteristics.csv")
    models = (
        "wind_tunnel_test",
        "authors_proposed_modified_uvlm",
        "fluxv_uvpm",
        "fluxv_periodic_v2",
        "one_state_ullt_local",
        "ptera_free_wake_uvlm",
    )
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.0), sharex=True)
    for model in models:
        selected = sorted(
            (row for row in rows if row["model"] == model),
            key=lambda row: float(row["aoa_deg"]),
        )
        x = [float(row["aoa_deg"]) for row in selected]
        for axis, field in zip(axes, ("mean_lift_gf", "mean_drag_gf")):
            axis.plot(
                x,
                [float(row[field]) for row in selected],
                label=LABELS[model],
                **STYLES[model],
            )
    axes[0].set_ylabel("Cycle-mean lift [gf]")
    axes[1].set_ylabel(r"Cycle-mean drag $D=-T$ [gf]")
    for axis in axes:
        axis.set_xlabel(r"Installation angle $\alpha_0$ [deg]")
        axis.set_xticks([0, 5, 10, 15, 20, 25])
        axis.axhline(0.0, color="#BBBBBB", lw=0.7, zorder=0)
        axis.grid(color="#D9D9D9", lw=0.55, alpha=0.8)
    _panel_labels(axes)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.04))
    fig.subplots_adjust(top=0.79, wspace=0.25)
    _finish(fig, figures / "yang2025_mean_lift_drag_vs_aoa")


def plot_yang_phase(run: Path, figures: Path, aoa: float = 15.0) -> None:
    rows = _rows(run / "yang2025_phase_histories.csv")
    models = (
        "fluxv_uvpm",
        "fluxv_periodic_v1",
        "fluxv_periodic_v2",
        "one_state_ullt_local",
        "ptera_free_wake_uvlm",
    )
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.0), sharex=True)
    for model in models:
        selected = sorted(
            (
                row
                for row in rows
                if row["model"] == model
                and np.isclose(float(row["aoa_deg"]), aoa)
            ),
            key=lambda row: float(row["phase"]),
        )
        phase = [float(row["phase"]) for row in selected]
        for axis, field in zip(axes, ("lift_n", "drag_n")):
            axis.plot(
                phase,
                [float(row[field]) for row in selected],
                label=LABELS[model],
                **STYLES[model],
            )
    axes[0].set_ylabel("Instantaneous lift [N]")
    axes[1].set_ylabel(r"Instantaneous drag $D=-T$ [N]")
    for axis in axes:
        axis.set_xlabel(r"Cycle phase $t/T$")
        axis.set_xlim(0.0, 1.0)
        axis.axhline(0.0, color="#BBBBBB", lw=0.7, zorder=0)
        axis.grid(color="#D9D9D9", lw=0.55, alpha=0.8)
    _panel_labels(axes)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.04))
    fig.subplots_adjust(top=0.79, wspace=0.25)
    _finish(fig, figures / "yang2025_15deg_phase_lift_drag")


def plot_izraelevitz_phase(run: Path, figures: Path) -> None:
    rows = _rows(run / "izraelevitz2017_fig11_phase_histories.csv")
    models = (
        "paper_uvlm",
        "6_state",
        "1_state",
        "qs_added_mass",
        "one_state_ullt_local",
        "fluxv_uvpm",
        "fluxv_periodic_v2",
    )
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2), sharex=True)
    for model in models:
        selected = sorted(
            (row for row in rows if row["model"] == model),
            key=lambda row: float(row["phase"]),
        )
        phase = [float(row["phase"]) for row in selected]
        for axis, field in zip(axes, ("CL_alpha", "CD_alpha")):
            kwargs = dict(STYLES[model])
            if model == "one_state_ullt_local":
                kwargs.update(marker="x", markevery=12, ms=3.5)
            axis.plot(
                phase,
                [float(row[field]) for row in selected],
                label=LABELS[model],
                **kwargs,
            )
    axes[0].set_ylabel(r"Paper-scaled lift $C_{L\alpha}$ [-]")
    axes[1].set_ylabel(r"Paper-scaled drag $C_{D\alpha}$ [-]")
    for axis in axes:
        axis.set_xlabel(r"Cycle phase $t/T$")
        axis.set_xlim(0.0, 1.0)
        axis.axhline(0.0, color="#BBBBBB", lw=0.7, zorder=0)
        axis.grid(color="#D9D9D9", lw=0.55, alpha=0.8)
    _panel_labels(axes)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.055))
    fig.subplots_adjust(top=0.77, wspace=0.24)
    _finish(fig, figures / "izraelevitz2017_fig11_lift_drag_phase")


def plot_accuracy(run: Path, figures: Path) -> None:
    rows = _rows(run / "accuracy_metrics.csv")
    yang_models = (
        "authors_proposed_modified_uvlm",
        "fluxv_uvpm",
        "fluxv_periodic_v2",
        "one_state_ullt_local",
        "ptera_free_wake_uvlm",
    )
    iz_models = (
        "1_state",
        "one_state_ullt_local",
        "fluxv_uvpm",
        "fluxv_periodic_v2",
        "ptera_free_wake_uvlm",
    )
    color_map = {
        "authors_proposed_modified_uvlm": COLORS["author"],
        "1_state": COLORS["author"],
        "fluxv_uvpm": COLORS["old"],
        "fluxv_periodic_v2": COLORS["v2"],
        "one_state_ullt_local": COLORS["ullt"],
        "ptera_free_wake_uvlm": COLORS["free"],
    }
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.3))
    for axis, paper, models, metric, ylabel in (
        (axes[0], "yang2025", yang_models, "mae", "MAE versus wind tunnel [gf]"),
        (
            axes[1],
            "izraelevitz2017_fig11",
            iz_models,
            "rmse",
            "Raw-phase RMSE in paper-scaled\ncoefficient [-] versus authors' UVLM",
        ),
    ):
        x = np.arange(len(models), dtype=float)
        width = 0.35
        for offset, channel, hatch in ((-width / 2, "lift", ""), (width / 2, "drag", "//")):
            values = []
            for model in models:
                row = next(
                    item
                    for item in rows
                    if item["paper"] == paper
                    and item["model"] == model
                    and item["channel"] == channel
                )
                values.append(float(row[metric]))
            bars = axis.bar(
                x + offset,
                values,
                width,
                label=channel.capitalize(),
                color=[color_map[model] for model in models],
                hatch=hatch,
                edgecolor="#333333",
                linewidth=0.45,
                alpha=0.88,
            )
            for bar, value in zip(bars, values):
                axis.text(
                    bar.get_x() + bar.get_width() / 2,
                    value,
                    f"{value:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    rotation=90,
                )
        axis.set_xticks(x)
        axis.set_xticklabels(
            [LABELS[model] for model in models], rotation=24, ha="right"
        )
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", color="#D9D9D9", lw=0.55, alpha=0.8)
        axis.set_axisbelow(True)
        axis.margins(y=0.18)
    handles = [
        Patch(facecolor="#BDBDBD", edgecolor="#333333", label="Lift"),
        Patch(
            facecolor="#BDBDBD",
            edgecolor="#333333",
            hatch="//",
            label="Drag",
        ),
    ]
    labels = ["Lift", "Drag"]
    fig.legend(
        handles,
        labels,
        frameon=False,
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
    )
    _panel_labels(axes)
    fig.subplots_adjust(bottom=0.30, top=0.84, wspace=0.28)
    _finish(fig, figures / "crosspaper_accuracy_summary")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    figures = run / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    _style()
    plot_yang_means(run, figures)
    plot_yang_phase(run, figures)
    plot_izraelevitz_phase(run, figures)
    plot_accuracy(run, figures)
    captions = """# Figure captions and interpretation limits

Across all local-model figures, plotted drag follows `D=-T=-Fx_W`: positive
drag is resistance and negative drag denotes net thrust.

## yang2025_mean_lift_drag_vs_aoa

Cycle-mean single-wing lift and drag versus installation angle for the Yang
2025 rigid-wing wind-tunnel cases.  The wind-tunnel and authors' modified-UVLM
points are vector/raster-digitized cycle means.  The assigned ±0.4 gf source
uncertainty is digitization uncertainty only, not an experimental error bar,
and it is not drawn as an error bar.  FluxV v2 is exploratory periodic
two-pass output.

## yang2025_15deg_phase_lift_drag

Local-model diagnostic at installation angle `alpha_0=15 deg`.  Yang et al. did not publish
phase-resolved test or proposed-model loads, so this figure compares predicted
shape only and is not a phase-accuracy validation.  Shared-gate transition
features are not smoothed or hidden; the movement derivative is closed on the
selected coherent cycle rather than across a duplicated movement endpoint.

## izraelevitz2017_fig11_lift_drag_phase

Paper-scaled lift and drag for Figure 11.  Author curves were recovered from
the source PDF vector paths without amplitude fitting or phase alignment.  The
authors' UVLM is a numerical reference rather than experimental ground truth.
The dimensionless scaling is `C_Lalpha=C_L/sin(15 deg)` and
`C_Dalpha=C_D/sin(15 deg)`; negative `C_Dalpha` denotes net thrust.

## crosspaper_accuracy_summary

Yang bars are six-angle MAE in gf against wind-tunnel cycle means.
Izraelevitz bars are raw-phase RMSE in paper-scaled coefficients against the
authors' UVLM; no optimal cyclic phase shift is applied.  Units therefore
differ between panels and values must not be pooled into a single score.
"""
    (run / "FIGURE_CAPTIONS.md").write_text(captions, encoding="utf-8")
    print(figures)


if __name__ == "__main__":
    main()
