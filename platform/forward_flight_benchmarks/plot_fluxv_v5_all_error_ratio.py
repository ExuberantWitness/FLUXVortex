"""Plot per-condition v5a/v4b error ratios without mixing physical units."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np

from .fluxv_v5_all_conditions import OUTPUT_ROOT
from .fluxv_v5_all_conditions_style import (
    configure_matplotlib,
    save_figure,
)


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _annotate(ax, ratio):
    for i in range(ratio.shape[0]):
        for j in range(ratio.shape[1]):
            value = ratio[i, j]
            if np.isfinite(value):
                ax.text(j, i, f"{value:.2f}×", ha="center", va="center", fontsize=6.0)
            else:
                ax.text(
                    j, i, "n/a", ha="center", va="center", fontsize=6.0, color="#666666"
                )


def _draw_heatmap(ax, ratio, *, cmap, norm):
    rows, columns = ratio.shape
    mesh = ax.pcolormesh(
        np.arange(columns + 1) - 0.5,
        np.arange(rows + 1) - 0.5,
        np.log2(ratio),
        cmap=cmap,
        norm=norm,
        shading="flat",
        rasterized=False,
    )
    ax.set_xlim(-0.5, columns - 0.5)
    ax.set_ylim(rows - 0.5, -0.5)
    return mesh


def make_figure(
    curve_path: Path, metric_path: Path, output_stem: Path
) -> tuple[Path, Path]:
    configure_matplotlib()
    curves = _read(curve_path)
    metrics = _read(metric_path)
    norm = colors.TwoSlopeNorm(vmin=-2.0, vcenter=0.0, vmax=2.0)
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#DDDDDD")
    fig = plt.figure(figsize=(7.25, 4.75))
    grid = fig.add_gridspec(1, 3, width_ratios=(1.35, 0.9, 2.0), wspace=0.48)
    axes = [fig.add_subplot(grid[0, i]) for i in range(3)]

    # Yang: 6 conditions x lift/drag.
    rows = [row for row in curves if row["paper"] == "yang2025"]
    values = defaultdict(dict)
    for row in rows:
        values[(row["case_id"], row["observable"])][row["model_id"]] = float(
            row["value"]
        )
    yang = np.full((6, 2), np.nan)
    for i, aoa in enumerate((0, 5, 10, 15, 20, 25)):
        for j, observable in enumerate(("lift", "drag")):
            item = values[(f"aoa_{aoa}", observable)]
            denom = abs(item["fluxv_v4b"] - item["experiment"])
            numer = abs(item["fluxv_v5a"] - item["experiment"])
            yang[i, j] = numer / denom if denom > 1.0e-12 else np.nan
    image = _draw_heatmap(axes[0], yang, cmap=cmap, norm=norm)
    _annotate(axes[0], yang)
    axes[0].set_xticks((0, 1), ("Lift", "Drag"))
    axes[0].set_yticks(range(6), [f"{x}°" for x in (0, 5, 10, 15, 20, 25)])
    axes[0].set_ylabel("Yang installation angle")

    # Figure 14: 12 unique conditions.
    rows = [row for row in curves if row["paper"] == "izraelevitz2017_fig14"]
    truth = defaultdict(list)
    pred = defaultdict(dict)
    for row in rows:
        if row["model_id"] == "experiment":
            truth[row["case_id"]].append(float(row["value"]))
        else:
            pred[row["case_id"]][row["model_id"]] = float(row["value"])
    keys = sorted(
        truth, key=lambda key: (int(key.split("_")[1]), int(key.split("_")[3]))
    )
    fig14 = np.full((len(keys), 1), np.nan)
    for i, key in enumerate(keys):
        observed = float(np.mean(truth[key]))
        denom = abs(pred[key]["fluxv_v4b"] - observed)
        numer = abs(pred[key]["fluxv_v5a"] - observed)
        fig14[i, 0] = numer / denom if denom > 1.0e-12 else np.nan
    _draw_heatmap(axes[1], fig14, cmap=cmap, norm=norm)
    _annotate(axes[1], fig14)
    axes[1].set_xticks((0,), (r"$C_T$",))
    axes[1].set_yticks(
        range(len(keys)),
        [f"{key.split('_')[1]}°/{key.split('_')[3]}°" for key in keys],
    )
    axes[1].set_ylabel(r"Figure 14: $\theta_{\max}/\psi$")

    # Baik: four cases x raw/filtered lift/drag phase RMSE.
    baik = np.full((4, 4), np.nan)
    for i, case_id in enumerate(("W1", "W2", "W3", "W4")):
        for j, (view, observable) in enumerate(
            (
                ("raw_numeric_diagnostic", "CL"),
                ("raw_numeric_diagnostic", "CD"),
                ("filtered_1hz", "CL"),
                ("filtered_1hz", "CD"),
            )
        ):
            selected = [
                row
                for row in metrics
                if row["paper"] == "baik2012"
                and row["scope"] == "case_phase_history"
                and row["case_id"] == case_id
                and row["view"] == view
                and row["observable"] == observable
            ]
            by_model = {row["model_id"]: float(row["rmse"]) for row in selected}
            baik[i, j] = by_model["fluxv_v5a"] / by_model["fluxv_v4b"]
    _draw_heatmap(axes[2], baik, cmap=cmap, norm=norm)
    _annotate(axes[2], baik)
    axes[2].set_xticks(range(4), ("CL\nraw", "CD\nraw", "CL\n1 Hz", "CD\n1 Hz"))
    axes[2].set_yticks(range(4), ("W1", "W2", "W3", "W4"))
    axes[2].set_ylabel("Baik case")

    for ax, label in zip(axes, ("(a)", "(b)", "(c)")):
        ax.text(
            -0.14,
            1.035,
            label,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontweight="bold",
            fontsize=9.0,
            clip_on=False,
        )
        ax.tick_params(top=False, right=False)
    colorbar = fig.colorbar(
        image, ax=axes, orientation="horizontal", fraction=0.055, pad=0.12, aspect=35
    )
    colorbar.solids.set_rasterized(False)
    colorbar.set_label(
        r"$\log_2(E_{\mathrm{v5a}}/E_{\mathrm{v4b}})$; blue improves, red regresses"
    )
    colorbar.set_ticks((-2, -1, 0, 1, 2))
    colorbar.set_ticklabels(("0.25×", "0.5×", "1×", "2×", "4×"))
    fig.subplots_adjust(left=0.09, right=0.99, bottom=0.20, top=0.985)
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
        args.data_dir / "all_conditions_metrics.csv",
        args.output_dir / "fig04_all_condition_error_ratio",
    )


if __name__ == "__main__":
    main()
