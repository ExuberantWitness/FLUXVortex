"""Shared plotting helper for the paired Baik phase-history figures."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .fluxv_v5_all_conditions_style import (
    MODEL_STYLES,
    configure_matplotlib,
    panel_label,
    save_figure,
)


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def plot_baik_grid(
    curve_path: Path,
    output_stem: Path,
    *,
    view: str,
    include_theodorsen: bool,
) -> tuple[Path, Path]:
    configure_matplotlib()
    rows = _read(curve_path)
    fig, axes = plt.subplots(4, 2, figsize=(7.25, 7.65), sharex=True)
    models = [
        ("experiment", "Corrected-total experiment"),
        ("fluxv_old", "FluxV old"),
        ("fluxv_v4b", "FluxV v4b"),
        ("fluxv_v5a", "FluxV v5a development proxy"),
    ]
    for row_index, case_id in enumerate(("W1", "W2", "W3", "W4")):
        for column, observable in enumerate(("CL", "CD")):
            ax = axes[row_index, column]
            panel_models = list(models)
            if observable == "CL" and include_theodorsen:
                panel_models.insert(1, ("theodorsen", "Published standard Theodorsen"))
            for model_id, label in panel_models:
                selected = [
                    row
                    for row in rows
                    if row["paper"] == "baik2012"
                    and row["case_id"] == case_id
                    and row["view"] == view
                    and row["observable"] == observable
                    and row["model_id"] == model_id
                ]
                selected.sort(key=lambda row: float(row["x_value"]))
                if not selected:
                    continue
                x = np.asarray([float(row["x_value"]) for row in selected])
                y = np.asarray([float(row["value"]) for row in selected])
                style = MODEL_STYLES[model_id]
                if model_id == "experiment":
                    ax.plot(
                        x,
                        y,
                        label=label,
                        color=style["color"],
                        linewidth=1.55,
                        zorder=8,
                    )
                else:
                    line_style = {
                        key: value
                        for key, value in style.items()
                        if key not in ("marker", "markersize")
                    }
                    ax.plot(x, y, label=label, **line_style)
            ax.axhline(0.0, color="#777777", linewidth=0.6, zorder=0)
            ax.set_ylabel(r"$C_L$" if observable == "CL" else r"$C_D$")
            ax.text(
                0.98,
                0.94,
                case_id,
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontweight="bold",
            )
            ax.set_xlim(0.0, 1.0)
            ax.set_xticks(np.linspace(0.0, 1.0, 5))
        axes[row_index, 1].text(
            0.98,
            0.06,
            "$C_D<0$: thrust",
            transform=axes[row_index, 1].transAxes,
            ha="right",
            va="bottom",
            fontsize=6.8,
            color="#555555",
        )
    axes[-1, 0].set_xlabel(r"Cycle phase, $t/T$")
    axes[-1, 1].set_xlabel(r"Cycle phase, $t/T$")
    for index, ax in enumerate(axes.flat):
        panel_label(ax, f"({chr(ord('a') + index)})")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.997),
        frameon=False,
        ncol=len(labels),
        columnspacing=1.25,
        handlelength=2.1,
    )
    fig.subplots_adjust(
        left=0.085, right=0.99, bottom=0.065, top=0.955, wspace=0.20, hspace=0.14
    )
    paths = save_figure(fig, output_stem)
    plt.close(fig)
    return paths
