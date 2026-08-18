"""Plot every frozen Yang 2025 installation-angle condition."""

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


def _plot_model(ax, rows, model_id, observable, label):
    selected = [
        row
        for row in rows
        if row["paper"] == "yang2025"
        and row["observable"] == observable
        and row["model_id"] == model_id
    ]
    selected.sort(key=lambda row: float(row["x_value"]))
    x = np.asarray([float(row["x_value"]) for row in selected])
    y = np.asarray([float(row["value"]) for row in selected])
    style = MODEL_STYLES[model_id]
    if model_id == "experiment":
        err = np.asarray([float(row["uncertainty_plus"]) for row in selected])
        ax.errorbar(
            x,
            y,
            yerr=err,
            label=label,
            color=style["color"],
            marker=style["marker"],
            linestyle="none",
            markersize=4.5,
            capsize=2.0,
            elinewidth=0.8,
            zorder=style["zorder"],
        )
    else:
        ax.plot(x, y, label=label, **style)


def make_figure(curve_path: Path, output_stem: Path) -> tuple[Path, Path]:
    configure_matplotlib()
    rows = _read(curve_path)
    fig, axes = plt.subplots(2, 2, figsize=(7.25, 5.6), sharex=True)
    external = (
        ("experiment", "Experiment (±0.4 gf digitization)"),
        ("authors_proposed_modified_uvlm", "Authors' modified UVLM"),
        ("ptera_free_wake_uvlm", "Ptera free-wake UVLM"),
        ("one_state_ullt_local", "Local one-state ULLT"),
    )
    fluxv = (
        ("experiment", "Experiment (±0.4 gf digitization)"),
        ("fluxv_old", "FluxV old"),
        ("fluxv_v3", "FluxV v3"),
        ("fluxv_v4b", "FluxV v4b"),
        ("fluxv_v5a", "FluxV v5a proxy (= v1/v2 means)"),
    )
    for column, observable in enumerate(("lift", "drag")):
        for model_id, label in external:
            _plot_model(axes[0, column], rows, model_id, observable, label)
        for model_id, label in fluxv:
            _plot_model(axes[1, column], rows, model_id, observable, label)
        axes[0, column].axhline(0.0, color="#777777", linewidth=0.65, zorder=0)
        axes[1, column].axhline(0.0, color="#777777", linewidth=0.65, zorder=0)
        axes[1, column].set_xlabel(r"Installation angle, $\alpha_0$ [deg]")
        axes[0, column].set_ylabel(
            r"$\overline{L}$ [gf]" if observable == "lift" else r"$\overline{D}$ [gf]"
        )
        axes[1, column].set_ylabel(
            r"$\overline{L}$ [gf]" if observable == "lift" else r"$\overline{D}$ [gf]"
        )
        axes[0, column].set_xticks(np.arange(0, 26, 5))
        axes[1, column].set_xticks(np.arange(0, 26, 5))
    for ax, label in zip(axes.flat, ("(a)", "(b)", "(c)", "(d)")):
        panel_label(ax, label)
    external_handles, external_labels = axes[0, 0].get_legend_handles_labels()
    fluxv_handles, fluxv_labels = axes[1, 0].get_legend_handles_labels()
    fig.legend(
        external_handles,
        external_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        frameon=False,
        ncol=4,
        columnspacing=1.1,
    )
    fig.legend(
        fluxv_handles,
        fluxv_labels,
        loc="center",
        bbox_to_anchor=(0.5, 0.505),
        frameon=False,
        ncol=5,
        columnspacing=1.0,
    )
    fig.subplots_adjust(
        left=0.09, right=0.985, bottom=0.085, top=0.92, wspace=0.23, hspace=0.42
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
        args.output_dir / "fig01_yang_all_conditions",
    )


if __name__ == "__main__":
    main()
