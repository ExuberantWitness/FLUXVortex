"""Summarise the flapping-wing benchmark (Izraelevitz-Scherer Fig. 14) across the ring kernel constant.

Reads every `izraelevitz_fig14_eps<value>/` folder under the sweep directory that has a
`metrics.json`, and reports, per value of the constant, the thrust-coefficient error of
the production solver against the 14 digitised experimental markers, plus the model's
thrust for each condition.  Writes `izraelevitz_mae_vs_constant.png`.

Pure post-processing: no solver, no GPU.

Usage
-----
    PYTHONPATH=src python3 platform/warp_vpm/summarize_izraelevitz_eps_sweep.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SWEEP_ROOT = REPO_ROOT / "artifacts/experiment/20260824_rojratsirikul2011_q16_fsi/eps_sweep"


def load_all() -> list[dict]:
    found = []
    for folder in sorted(SWEEP_ROOT.glob("izraelevitz_fig14_eps*")):
        metrics_file = folder / "metrics.json"
        if not metrics_file.exists():
            continue
        metrics = json.loads(metrics_file.read_text())
        manifest_file = folder / "manifest.json"
        manifest = json.loads(manifest_file.read_text()) if manifest_file.exists() else {}
        predictions_file = folder / "predictions.csv"
        predictions = (
            {row["case_id"]: row for row in csv.DictReader(open(predictions_file))}
            if predictions_file.exists() else {}
        )
        found.append({
            "eps_text": folder.name.split("_eps", 1)[1],
            "eps": float(folder.name.split("_eps", 1)[1]),
            "metrics": metrics,
            "gate": manifest.get("metrics", {}).get("gate", {}),
            "predictions": predictions,
        })
    return sorted(found, key=lambda item: item["eps"])


def main() -> int:
    runs = load_all()
    if not runs:
        print(f"no flapping results under {SWEEP_ROOT}")
        return 1

    print("Izraelevitz-Scherer Fig. 14, flapping wing, 14 experimental markers, production solver")
    print(f"{'eps (m^4)':>10} {'MAE CT':>8} {'bias':>8} {'RMSE':>8} {'max|err|':>9} {'markers':>8} "
          f"{'within bar':>11} {'gate limit':>11} {'gate':>6}")
    for run in runs:
        m = run["metrics"]
        gate = run["gate"]
        limit = gate.get("limit")
        print(f"{run['eps_text']:>10} {m['mae_ct']:8.4f} {m['bias_ct']:+8.4f} {m['rmse_ct']:8.4f} "
              f"{m['max_abs_error_ct']:9.4f} {m['n_markers']:8d} {m['markers_within_error_bar']:11d} "
              f"{(f'{limit:.4f}' if isinstance(limit, float) else '-'):>11} {str(gate.get('passed')):>6}")

    if len(runs) >= 2:
        print("\nthrust coefficient CT of the model per condition (last-cycle mean), by constant; "
              "experiment last")
        print(f"{'condition':13s} {'theta':>5s} {'psi':>4s} | " + " | ".join(f"{r['eps_text']:>8s}" for r in runs)
              + " | {'experiment':>10s}")
        reference = runs[0]["predictions"]
        for case_id, row in reference.items():
            cells = []
            for run in runs:
                other = run["predictions"].get(case_id)
                cells.append(f"{float(other['ct_prediction']):+8.4f}" if other else f"{'-':>8s}")
            print(f"{case_id:13s} {float(row['theta_max_deg']):5.0f} {float(row['phase_offset_deg']):4.0f} | "
                  + " | ".join(cells) + f" | {float(row['ct_experiment']):+10.4f}")

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        x = [run["eps"] for run in runs]
        figure, axis = plt.subplots(figsize=(7.6, 4.6))
        axis.plot(x, [r["metrics"]["mae_ct"] for r in runs], "o-", label="mean absolute error of CT")
        axis.plot(x, [abs(r["metrics"]["bias_ct"]) for r in runs], "s--", label="|mean signed error of CT|")
        axis.plot(x, [r["metrics"]["max_abs_error_ct"] for r in runs], "^:", label="largest |error| of CT")
        limits = [r["gate"].get("limit") for r in runs if isinstance(r["gate"].get("limit"), float)]
        if limits:
            axis.axhline(limits[0], color="grey", linestyle=":", label=f"accuracy gate {limits[0]:.4f}")
        axis.axvline(1.0e-9, color="grey", linestyle="--", linewidth=1.0)
        axis.annotate("current constant 1e-9", xy=(1.0e-9, axis.get_ylim()[1]), xytext=(4, -12),
                      textcoords="offset points", fontsize=8, color="grey", va="top")
        axis.set_xscale("log")
        axis.set_xlabel("ring kernel denominator constant eps (m^4);  smaller = closer to Biot-Savart")
        axis.set_ylabel("error in thrust coefficient CT")
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize=8)
        axis.set_title("Flapping wing (Izraelevitz-Scherer Fig. 14, 14 markers): "
                       "thrust error vs the ring kernel constant", fontsize=10)
        figure.tight_layout()
        out = SWEEP_ROOT / "izraelevitz_mae_vs_constant.png"
        figure.savefig(out, dpi=150)
        print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
