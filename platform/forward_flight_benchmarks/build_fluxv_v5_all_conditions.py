"""CLI for building the FluxV all-condition comparison data products."""

from __future__ import annotations

import argparse
from pathlib import Path

from .fluxv_v5_all_conditions import OUTPUT_ROOT, build_all


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    manifest = build_all(args.output_dir)
    print(f"wrote {manifest['row_counts']['curves']} curve rows")
    print(f"wrote {manifest['row_counts']['metrics']} metric rows")


if __name__ == "__main__":
    main()
