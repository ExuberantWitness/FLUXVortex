"""Plot the v5b no-LEV exact-reduction failure that blocked paper scoring."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .fluxv_v5_all_conditions import INPUTS, OUTPUT_ROOT
from .fluxv_v5_all_conditions_style import (
    MODEL_STYLES,
    configure_matplotlib,
    panel_label,
    save_figure,
)


def make_figure(input_path: Path, output_stem: Path) -> tuple[Path, Path]:
    configure_matplotlib()
    with input_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    phase = np.asarray([float(row["phase"]) for row in rows])
    fig, axes = plt.subplots(1, 2, figsize=(7.25, 2.75), sharex=True)
    for ax, observable in zip(axes, ("CL", "CD")):
        current = np.asarray(
            [float(row[f"current_fluxv_{observable}"]) for row in rows]
        )
        v5b = np.asarray([float(row[f"v5b_no_lev_{observable}"]) for row in rows])
        ax.plot(phase, current, label="Current FluxV", **MODEL_STYLES["fluxv_old"])
        ax.plot(phase, v5b, label="Standalone v5b, LEV disabled", **MODEL_STYLES["v5b"])
        ax.set_xlabel("Cycle phase")
        ax.set_ylabel(r"$C_L$" if observable == "CL" else r"$C_D$")
        ax.axhline(0.0, color="#777777", linewidth=0.6, zorder=0)
        ax.set_xlim(0.0, 0.95)
    panel_label(axes[0], "(a)")
    panel_label(axes[1], "(b)")
    axes[0].legend(loc="lower left", frameon=False)
    axes[1].text(
        0.98,
        0.06,
        r"max $|\Delta C_L|=0.556$" "\n" r"max $|\Delta C_D|=0.529$",
        transform=axes[1].transAxes,
        ha="right",
        va="bottom",
        color="#7A1E1E",
    )
    fig.subplots_adjust(left=0.09, right=0.99, bottom=0.18, top=0.98, wspace=0.20)
    paths = save_figure(fig, output_stem)
    plt.close(fig)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=INPUTS["v5b_gate"])
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT / "figures")
    args = parser.parse_args()
    make_figure(args.input, args.output_dir / "fig05_v5b_no_lev_gate")


if __name__ == "__main__":
    main()
