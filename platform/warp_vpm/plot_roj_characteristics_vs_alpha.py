"""Rojratsirikul 2011: characteristics vs angle of attack.

Experiment truth (digitized Figures 6/9/10/11) against the current
unified-architecture runs (A16, A17-MODE).  A10/A23 were cut from the run
scope, so the current-architecture series has two points; they are drawn as
a dashed guide, not a fitted curve.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path("artifacts/baselines/fluxv_v5m_rojratsirikul2011_unified_current")
FIG = OUT / "roj_characteristics_vs_alpha.png"

# Digitized experiment truth (frozen observations CSV / handoff §4.2).
EXP_ALPHA = [10.0, 16.0, 17.0, 23.0]
EXP_ZMAX = [0.032, 0.043, 0.0445, 0.0475]
EXP_CN_LO = [0.50, 0.92, 0.97, 0.98]
EXP_CN_HI = [0.52, 0.95, 0.97, 1.02]
EXP_ST = [1.10, None, 0.85, 0.83]           # A16 has no St oracle in the paper
EXP_PEAKS_CH = [3, None, 2, 2]
EXP_PEAKS_SP = [3, None, 0, None]           # A23 spanwise: not digitized


def load(tag: str) -> dict:
    return json.loads((OUT / f"{tag}.json").read_text())


def main() -> None:
    a16 = load("ROJ11_A16_FULL")
    a17 = load("ROJ11_A17_MODE_FULL")
    sim_alpha = [16.0, 17.0]
    sim = [a16, a17]

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.5))
    exp_color, sim_color = "black", "#d62728"

    # -- Cn(alpha) ---------------------------------------------------------
    axis = axes[0, 0]
    mid = [(lo + hi) / 2 for lo, hi in zip(EXP_CN_LO, EXP_CN_HI)]
    half = [(hi - lo) / 2 for lo, hi in zip(EXP_CN_LO, EXP_CN_HI)]
    axis.errorbar(EXP_ALPHA, mid, yerr=half, fmt="o-", color=exp_color,
                  markersize=7, capsize=4, linewidth=1.6,
                  label="experiment (digitized Fig.9)")
    axis.plot(sim_alpha, [p["mean_Cn"] for p in sim], "*--", color=sim_color,
              markersize=17, linewidth=1.4,
              label="current architecture (unified Q16–V5M)")
    for alpha, p in zip(sim_alpha, sim):
        offset = (EXP_CN_LO[EXP_ALPHA.index(alpha)] + EXP_CN_HI[EXP_ALPHA.index(alpha)]) / 2
        axis.annotate(f"+{(p['mean_Cn'] / offset - 1) * 100:.0f}%",
                      (alpha, p["mean_Cn"]), textcoords="offset points",
                      xytext=(8, 2), color=sim_color, fontsize=11)
    axis.set_xlabel(r"$\alpha$ (deg)")
    axis.set_ylabel(r"mean $C_n$ (stationary window)")
    axis.set_title(r"Normal force: $C_n(\alpha)$ — +33/36% potential-flow offset")
    axis.grid(alpha=0.3); axis.legend(fontsize=9)

    # -- zmax/c(alpha) -----------------------------------------------------
    axis = axes[0, 1]
    axis.errorbar(EXP_ALPHA, EXP_ZMAX, yerr=0.005, fmt="o-", color=exp_color,
                  markersize=7, capsize=4, linewidth=1.6,
                  label="experiment (digitized Fig.6)")
    axis.plot(sim_alpha, [p["mean_zmax_over_c"] for p in sim], "*--",
              color=sim_color, markersize=17, linewidth=1.4,
              label="current architecture")
    for alpha, p in zip(sim_alpha, sim):
        exp_v = EXP_ZMAX[EXP_ALPHA.index(alpha)]
        axis.annotate(f"{(p['mean_zmax_over_c'] / exp_v - 1) * 100:+.1f}%",
                      (alpha, p["mean_zmax_over_c"]), textcoords="offset points",
                      xytext=(8, 2), color=sim_color, fontsize=11)
    axis.set_xlabel(r"$\alpha$ (deg)")
    axis.set_ylabel(r"$\max(\overline{z})/c$")
    axis.set_title(r"Mean camber: $z_{max}/c(\alpha)$ — both inside ±0.005 gate")
    axis.grid(alpha=0.3); axis.legend(fontsize=9)

    # -- St(alpha) ---------------------------------------------------------
    axis = axes[1, 0]
    exp_a = [a for a, s in zip(EXP_ALPHA, EXP_ST) if s is not None]
    exp_s = [s for s in EXP_ST if s is not None]
    axis.plot(exp_a, exp_s, "o-", color=exp_color, markersize=7, linewidth=1.6,
              label="experiment (Fig.11)")
    sim_st = [p["dominant_St"] for p in sim]
    axis.plot(sim_alpha, sim_st, "*--", color=sim_color, markersize=17,
              linewidth=1.4, label="current architecture")
    axis.annotate("A17: fallback window\n(not stationary)", (17.0, sim_st[1]),
                  textcoords="offset points", xytext=(10, -34),
                  color=sim_color, fontsize=8.5)
    axis.set_xlabel(r"$\alpha$ (deg)")
    axis.set_ylabel(r"dominant $St$ at max-zsd station")
    axis.set_title(r"Vibration frequency: $St(\alpha)$ — model misses the experimental lock-in band")
    axis.grid(alpha=0.3); axis.legend(fontsize=9)

    # -- peak counts(alpha) ------------------------------------------------
    axis = axes[1, 1]
    width = 0.55
    ch_exp = [(a, v) for a, v in zip(EXP_ALPHA, EXP_PEAKS_CH) if v is not None]
    sp_exp = [(a, v) for a, v in zip(EXP_ALPHA, EXP_PEAKS_SP) if v is not None]
    axis.bar([a - width / 2 for a, _ in ch_exp], [v for _, v in ch_exp],
             width=width, color="#888888", label="experiment chordwise (Fig.10)")
    axis.bar([a + width / 2 for a, _ in sp_exp], [v for _, v in sp_exp],
             width=width, color="#cccccc", label="experiment spanwise")
    axis.plot(sim_alpha, [p["chordwise_peak_count"] for p in sim], "*",
              color=sim_color, markersize=17, label="current chordwise")
    axis.plot(sim_alpha, [p["spanwise_peak_count"] for p in sim], "o",
              color="#9467bd", markersize=10, label="current spanwise")
    axis.set_xlabel(r"$\alpha$ (deg)")
    axis.set_ylabel("zsd-map interior peak count")
    axis.set_title("Vibration spatial modes — chordwise 2nd mode not excited")
    axis.set_yticks(range(0, 5))
    axis.grid(alpha=0.3, axis="y"); axis.legend(fontsize=8.5)

    fig.suptitle(
        "Rojratsirikul 2011 — characteristics vs angle of attack: "
        "experiment truth vs current unified architecture "
        "(A16 + A17-MODE; A10/A23 not run)",
        fontsize=12.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(FIG, dpi=170, bbox_inches="tight")
    print(FIG.resolve())


if __name__ == "__main__":
    main()
