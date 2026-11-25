"""Flapping wing (Izraelevitz-Scherer Fig. 14): several runs next to each other and to the experiment.

Every argument is a folder that holds a `metrics.json` and a `predictions.csv` as
reproduce_izraelevitz2017_fig14_v5m_mandatory.py writes them.  For the 14 experimental markers
the table gives the measured thrust coefficient, each run's value and its signed error (run minus
experiment); below it the 14-marker MAE, mean bias, RMSE, largest error and the number of markers
inside the experimental error bar.  Pure post-processing: no solver, no GPU.

Usage
-----
    PYTHONPATH=src python3 platform/warp_vpm/compare_flapping_runs.py LABEL=DIR [LABEL=DIR ...]
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


def load(folder: Path) -> dict:
    metrics = json.loads((folder / "metrics.json").read_text())
    points = {
        (row["case_id"], row["replicate"]): row
        for row in csv.DictReader(open(folder / "predictions.csv"))
    }
    return {"metrics": metrics, "points": points}


def main(argv: list[str]) -> int:
    runs = []
    for argument in argv:
        label, _, folder = argument.partition("=")
        if not folder or not (Path(folder) / "metrics.json").exists():
            print(f"not a run folder (no metrics.json): {argument}")
            return 2
        runs.append((label, load(Path(folder))))
    if not runs:
        print(__doc__)
        return 2
    first = runs[0][1]["points"]
    order = sorted(first, key=lambda k: (float(first[k]["theta_max_deg"]), float(first[k]["phase_offset_deg"]), k[1]))
    print("Izraelevitz-Scherer Fig. 14, thrust coefficient: experiment, then each run and its error (run - experiment)")
    print(f"{'point':<16s}{'experiment':>11s}" + "".join(f"{label:>22s}" for label, _ in runs))
    for key in order:
        row = first[key]
        name = key[0] + (" (r2)" if key[1] != "1" else "")
        cells = []
        for _, run in runs:
            point = run["points"].get(key)
            cells.append(
                f"{float(point['ct_prediction']):11.4f}{float(point['signed_error']):+11.4f}" if point else f"{'-':>22s}"
            )
        print(f"{name:<16s}{float(row['ct_experiment']):11.4f}" + "".join(cells))
    print()
    print(f"{'run':<16s}{'MAE':>9s}{'bias':>9s}{'RMSE':>9s}{'max|err|':>10s}{'at':>14s}{'in bar':>8s}")
    for label, run in runs:
        m = run["metrics"]
        condition = m["max_abs_error_condition"]
        print(f"{label:<16s}{m['mae_ct']:9.4f}{m['bias_ct']:+9.4f}{m['rmse_ct']:9.4f}"
              f"{m['max_abs_error_ct']:10.4f}{str(condition):>14s}{m['markers_within_error_bar']:>5d}/{m['n_markers']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
