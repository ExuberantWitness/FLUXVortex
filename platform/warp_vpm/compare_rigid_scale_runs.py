"""Gate G-Scale (plan V3, section 6.2): the same rigid-plate run in three sets of units.

The plan asks for the rigid plate at alpha = 5 deg with (c, b, U dt) scaled together by 1, 10 and
100, and for Cn, the spanwise loading and the wake shape to agree to 1e-10.  Every run is
written by diagnose_rigid_plate_coefficients.py with --scale and --dump-state; this script reads
the three results and prints the differences.  Lengths are divided by the chord of their own run,
circulations by the chord and the freestream, before anything is compared.

Usage
-----
    python3 platform/warp_vpm/compare_rigid_scale_runs.py A.json B.json [C.json ...]

Each .json must have a sibling .npz of the same stem (the --dump-state file).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

LIMIT = 1.0e-10


def load(path: Path) -> dict:
    result = json.loads(path.read_text())
    state = np.load(path.with_suffix(".npz"))
    chord, speed = float(state["chord"]), float(state["speed"])
    ns = int(state["ns"])
    return {
        "name": path.stem,
        "chord": chord,
        "cn": float(result["cn_normal"]["mean"]),
        "kj": float(result["kutta_joukowski_check"]),
        "series": np.asarray(result["cn_series_every_100"], dtype=float),
        "te_row": np.asarray(result["trailing_edge_row"]["values"], dtype=float) / (chord * speed),
        "gamma": np.asarray(state["gamma_bound"], dtype=float) / (chord * speed),
        "wake": np.asarray(state["wake_rings"], dtype=float).reshape(-1, ns, 4, 3) / chord,
        "wake_gamma": np.asarray(state["wake_gamma"], dtype=float) / (chord * speed),
        "bound": np.asarray(state["bound_rings"], dtype=float).reshape(-1, ns, 4, 3) / chord,
    }


def relative(value: float, reference: float) -> float:
    return abs(value - reference) / max(abs(reference), 1e-300)


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 2
    runs = [load(Path(name)) for name in argv]
    base = runs[0]
    print(f"reference: {base['name']} (chord {base['chord']:g} m), {len(runs) - 1} comparison(s); limit {LIMIT:.0e}")
    worst = 0.0
    for run in runs[1:]:
        print(f"\n--- {run['name']} (chord {run['chord']:g} m) against {base['name']}")
        shape = [run["wake"].shape, base["wake"].shape]
        rows = []
        rows.append(("Cn, window mean", relative(run["cn"], base["cn"])))
        rows.append(("Kutta-Joukowski on the trailing-edge row", relative(run["kj"], base["kj"])))
        rows.append(("Cn every 100 steps, worst", float(np.max(np.abs(run["series"] - base["series"]) / np.abs(base["series"])))))
        rows.append(("spanwise loading (trailing-edge row / U c), worst", float(np.max(np.abs(run["te_row"] - base["te_row"])) / np.max(np.abs(base["te_row"])))))
        rows.append(("bound circulation / U c, worst", float(np.max(np.abs(run["gamma"] - base["gamma"])) / np.max(np.abs(base["gamma"])))))
        rows.append(("wake circulation / U c, worst", float(np.max(np.abs(run["wake_gamma"] - base["wake_gamma"])) / np.max(np.abs(base["wake_gamma"])))))
        rows.append(("bound ring corners / c, worst", float(np.max(np.abs(run["bound"] - base["bound"])))))
        if shape[0] == shape[1]:
            corners = float(np.max(np.abs(run["wake"] - base["wake"])))
            rows.append(("wake corners / c, worst", corners))
            rows.append(("wake corners, worst, in row spacings (0.01 c)", corners / 0.01))
        else:
            print(f"  wake shape differs: {shape[0]} against {shape[1]}")
        for label, value in rows:
            worst = max(worst, value) if np.isfinite(value) else worst
            print(f"  {label:52s} {value:9.1e}  {'pass' if value <= LIMIT else 'ABOVE'}")
    print(f"\nworst difference over all comparisons: {worst:.1e}  ->  {'PASS' if worst <= LIMIT else 'ABOVE THE LIMIT'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
