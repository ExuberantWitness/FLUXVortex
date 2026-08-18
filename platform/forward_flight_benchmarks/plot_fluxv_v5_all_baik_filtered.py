"""Plot the source-matched 1 Hz Baik W1--W4 comparison."""

from __future__ import annotations

import argparse
from pathlib import Path

from .fluxv_v5_all_conditions import OUTPUT_ROOT
from .fluxv_v5_all_conditions_plot_helpers import plot_baik_grid


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=OUTPUT_ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT / "figures")
    args = parser.parse_args()
    plot_baik_grid(
        args.data_dir / "all_conditions_curves.csv",
        args.output_dir / "fig03_baik_w1_w4_filtered",
        view="filtered_1hz",
        include_theodorsen=True,
    )


if __name__ == "__main__":
    main()
