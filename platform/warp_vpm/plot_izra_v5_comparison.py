"""Pre- vs post-speedup accuracy comparison for the Izraelevitz Fig.14 suite.

Plots CT vs phase-offset for both pitch families with four series:
  experiment (Scherer 1968 digitized, error bars),
  legacy non-current architecture (Pterra + post-hoc, legacy_v4b_CT_025c),
  previous current-architecture run (mandatory_v4, before the speed kernels),
  current run (mandatory_v5_post_speedup, after the fused ring kernel /
  particle far-field early-out / tangent-cadence work).
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
GT = ROOT / "docs/forward_flight_large_pitch/reproductions/unified_fluxv_upgrade_20260812/source_data/izraelevitz2017_fig14_digitized.csv"
LEGACY = ROOT / "docs/forward_flight_large_pitch/reproductions/fluxv_v5c_nextgen_20260814/runs/20260814_fluxv_v5c0_reference_full/fig14_v5c0_mean_thrust.csv"
V4 = ROOT / "artifacts/baselines/fluxv_v5m_izraelevitz2017_fig14_mandatory_v4/predictions.csv"
V5_DIR = ROOT / "artifacts/baselines/fluxv_v5m_izraelevitz2017_fig14_mandatory_v5_post_speedup"
V5 = V5_DIR / "predictions.csv"
OUT = V5_DIR / "comparison_pre_vs_post_speedup.png"


def load_gt() -> dict[float, dict[float, list[tuple[float, float, float]]]]:
    data: dict[float, dict[float, list[tuple[float, float, float]]]] = {}
    for row in csv.DictReader(open(GT)):
        if row["series"] != "scherer_1968_experiment":
            continue
        theta = float(row["theta_max_deg"])
        psi = float(row["phase_offset_deg"])
        data.setdefault(theta, {}).setdefault(psi, []).append(
            (float(row["ct"]), float(row["ct_error_minus"]), float(row["ct_error_plus"]))
        )
    return data


def load_model_predictions(path: Path, ct_key: str) -> dict[float, dict[float, float]]:
    data: dict[float, dict[float, float]] = {}
    for row in csv.DictReader(open(path)):
        theta = float(row["theta_max_deg"])
        psi = float(row["phase_offset_deg"])
        data.setdefault(theta, {})[psi] = float(row[ct_key])
    return data


def main() -> None:
    gt = load_gt()
    legacy = load_model_predictions(LEGACY, "legacy_v4b_CT_025c")
    v4 = load_model_predictions(V4, "ct_prediction")
    v5 = load_model_predictions(V5, "ct_prediction")

    fig = plt.figure(figsize=(16.5, 10.5))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.35, 1.0], hspace=0.32, wspace=0.22)

    colors = {"legacy": "#8c8c8c", "v4": "#1f77b4", "v5": "#d62728"}
    for column, theta in enumerate((15.0, 25.0)):
        axis = fig.add_subplot(grid[0, column])
        psis = sorted(gt.get(theta, {}))
        exp_mean = []
        exp_minus, exp_plus = [], []
        for psi in psis:
            marks = gt[theta][psi]
            mean = sum(m[0] for m in marks) / len(marks)
            exp_mean.append(mean)
            exp_minus.append(sum(m[1] for m in marks) / len(marks))
            exp_plus.append(sum(m[2] for m in marks) / len(marks))
            for mark in marks:
                axis.errorbar(
                    psi, mark[0], yerr=[[mark[1]], [mark[2]]],
                    fmt="o", color="black", markersize=4, alpha=0.55,
                    capsize=2, zorder=2,
                )
        axis.errorbar(
            [], [], yerr=[[0], [0]], fmt="o", color="black", markersize=4,
            label="Experiment (Scherer 1968)",
        )
        for label, series, style in (
            ("Legacy non-current arch (Pterra+post-hoc)", legacy, "--s"),
            ("Previous current-arch (v4, pre-speedup)", v4, "--^"),
            ("Current (v5, post-speedup)", v5, "-o"),
        ):
            key = {"Legacy non-current arch (Pterra+post-hoc)": "legacy",
                   "Previous current-arch (v4, pre-speedup)": "v4",
                   "Current (v5, post-speedup)": "v5"}[label]
            xs = sorted(series.get(theta, {}))
            if not xs:
                continue
            axis.plot(
                xs, [series[theta][x] for x in xs], style,
                color=colors[key], markersize=5, linewidth=1.6, label=label,
                zorder=3 if key == "v5" else 2,
            )
        axis.set_xlabel(r"phase offset $\psi$ (deg)")
        axis.set_ylabel(r"$C_T$")
        axis.set_title(rf"$\theta_{{max}}={theta:.0f}^\circ$ family: CT vs phase")
        axis.grid(alpha=0.3)
        axis.legend(fontsize=8.5, loc="best")

    # Signed error vs experiment (marker-mean) for the three models.
    axis = fig.add_subplot(grid[1, 0])
    for label, series, style in (
        ("legacy", legacy, "--s"),
        ("v4", v4, "--^"),
        ("v5", v5, "-o"),
    ):
        xs_all, errs_all = [], []
        for theta in (15.0, 25.0):
            for psi in sorted(gt.get(theta, {})):
                if psi not in series.get(theta, {}):
                    continue
                marks = gt[theta][psi]
                mean = sum(m[0] for m in marks) / len(marks)
                xs_all.append(psi + (0 if theta == 15.0 else 0.6))
                errs_all.append(series[theta][psi] - mean)
        axis.plot(xs_all, errs_all, style, color=colors[label], markersize=5,
                  linewidth=1.4, label=label)
    axis.axhline(0.0, color="black", linewidth=0.9)
    axis.set_xlabel(r"$\psi$ (deg; 25° family offset +0.6 for clarity)")
    axis.set_ylabel(r"signed $C_T$ error vs experiment")
    axis.set_title("Signed error against experimental marker means")
    axis.grid(alpha=0.3)
    axis.legend(fontsize=9)

    # Per-series MAE + summary.
    axis = fig.add_subplot(grid[1, 1])
    names, maes = [], []
    for label, series in (("legacy", legacy), ("v4", v4), ("v5", v5)):
        errors = []
        for theta in (15.0, 25.0):
            for psi, marks in gt.get(theta, {}).items():
                if psi not in series.get(theta, {}):
                    continue
                mean = sum(m[0] for m in marks) / len(marks)
                errors.append(abs(series[theta][psi] - mean))
        if errors:
            names.append(label)
            maes.append(sum(errors) / len(errors))
    bars = axis.bar(names, maes, color=[colors[n] for n in names], alpha=0.85)
    for bar, value in zip(bars, maes):
        axis.text(bar.get_x() + bar.get_width() / 2, value + 0.001,
                  f"{value:.4f}", ha="center", fontsize=10)
    v4_mae = maes[names.index("v4")] if "v4" in names else float("nan")
    v5_mae = maes[names.index("v5")] if "v5" in names else float("nan")
    drift = v5_mae - v4_mae
    axis.set_ylabel(r"MAE of $C_T$ (12 condition means)")
    axis.set_title("MAE vs experiment (metrics.json 14-marker MAE: v4 0.0386, v5 0.0409)")
    axis.grid(alpha=0.3, axis="y")

    info = (
        f"pre→post speedup MAE drift: {drift:+.2e}\n"
        f"(positive = slightly worse vs v4)\n"
        f"speed kernels: fused ring BS (7.8x, rel-diff 7e-16),\n"
        f"particle far-field early-out (3x, rel-diff 7e-16),\n"
        f"tangent refresh cadence (FSI path only, not used here)"
    )
    fig.text(0.985, 0.015, info, ha="right", va="bottom", fontsize=9,
             family="monospace",
             bbox=dict(boxstyle="round", facecolor="#f5f5f5", alpha=0.9))

    fig.suptitle(
        "Izraelevitz 2017 Fig.14 — accuracy before vs after the speed-engineering "
        "kernels (8x24 mandatory suite)",
        fontsize=13,
    )
    fig.savefig(OUT, dpi=170, bbox_inches="tight")
    print(OUT.resolve())
    print(f"MAE v4={v4_mae:.6f}  v5={v5_mae:.6f}  drift={drift:+.3e}")


if __name__ == "__main__":
    main()
