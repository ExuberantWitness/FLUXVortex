"""Plot all Izraelevitz/Scherer Figure 14 mean-thrust conditions."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .fluxv_v5_all_conditions import OUTPUT_ROOT
from .fluxv_v5_all_conditions_style import (
    MODEL_STYLES,
    configure_matplotlib,
    panel_label,
    save_figure,
)


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _theta(row: dict[str, str]) -> int:
    return int(float(row["case_id"].split("_")[1]))


def _plot(ax, rows, theta, model_id, label):
    selected = [
        row
        for row in rows
        if row["paper"] == "izraelevitz2017_fig14"
        and row["model_id"] == model_id
        and _theta(row) == theta
    ]
    selected.sort(key=lambda row: (float(row["x_value"]), row["replicate"]))
    x = np.asarray([float(row["x_value"]) for row in selected])
    y = np.asarray([float(row["value"]) for row in selected])
    style = MODEL_STYLES[model_id]
    if model_id == "experiment":
        lower = np.asarray([float(row["uncertainty_minus"]) for row in selected])
        upper = np.asarray([float(row["uncertainty_plus"]) for row in selected])
        ax.errorbar(
            x,
            y,
            yerr=np.vstack((lower, upper)),
            label=label,
            color=style["color"],
            marker=style["marker"],
            linestyle="none",
            markersize=4.3,
            capsize=2.0,
            elinewidth=0.75,
            zorder=style["zorder"],
        )
    elif selected:
        ax.plot(x, y, label=label, **style)


def make_figure(curve_path: Path, output_stem: Path) -> tuple[Path, Path]:
    configure_matplotlib()
    rows = _read(curve_path)
    fig, axes = plt.subplots(2, 2, figsize=(7.25, 5.6), sharex="col")
    reference = (
        ("experiment", "Scherer experiment"),
        ("authors_1state_ullt", "Authors' one-state ULLT"),
        ("authors_6state_ullt", "Authors' six-state ULLT"),
        ("authors_qs_added_mass", "QS + added mass"),
        ("one_state_ullt_local", "Local one-state ULLT"),
    )
    fluxv = (
        ("experiment", "Scherer experiment"),
        ("fluxv_old", "FluxV old"),
        ("fluxv_v1_v2", "FluxV v1/v2"),
        ("fluxv_v3", "FluxV v3"),
        ("fluxv_v4b", "FluxV v4b"),
        ("fluxv_v5a", "FluxV v5a development proxy"),
    )
    for column, theta in enumerate((15, 25)):
        for model_id, label in reference:
            _plot(axes[0, column], rows, theta, model_id, label)
        for model_id, label in fluxv:
            _plot(axes[1, column], rows, theta, model_id, label)
        for row in range(2):
            axes[row, column].axhline(0.0, color="#777777", linewidth=0.65, zorder=0)
            axes[row, column].set_ylabel(r"$\overline{C_T}$")
            axes[row, column].set_xticks(np.arange(15, 106, 15))
        axes[1, column].set_xlabel(r"Pitch--flap phase offset, $\psi$ [deg]")
        axes[0, column].text(
            0.98,
            0.95,
            rf"$\theta_{{\max}}={theta}^\circ$",
            transform=axes[0, column].transAxes,
            ha="right",
            va="top",
        )
        axes[1, column].text(
            0.98,
            0.95,
            rf"$\theta_{{\max}}={theta}^\circ$",
            transform=axes[1, column].transAxes,
            ha="right",
            va="top",
        )
    for ax, label in zip(axes.flat, ("(a)", "(b)", "(c)", "(d)")):
        panel_label(ax, label)
    reference_handles, reference_labels = axes[0, 0].get_legend_handles_labels()
    fluxv_handles, fluxv_labels = axes[1, 0].get_legend_handles_labels()
    fig.legend(
        reference_handles,
        reference_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        frameon=False,
        ncol=5,
        columnspacing=1.0,
    )
    fig.legend(
        fluxv_handles,
        fluxv_labels,
        loc="center",
        bbox_to_anchor=(0.5, 0.505),
        frameon=False,
        ncol=6,
        columnspacing=0.9,
    )
    fig.subplots_adjust(
        left=0.09, right=0.985, bottom=0.085, top=0.92, wspace=0.20, hspace=0.42
    )
    paths = save_figure(fig, output_stem)
    plt.close(fig)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=OUTPUT_ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT / "figures")
    args = parser.parse_args()
    make_figure(
        args.data_dir / "all_conditions_curves.csv",
        args.output_dir / "fig02_izraelevitz_fig14_all_conditions",
    )


if __name__ == "__main__":
    main()
