"""Rojratsirikul 2011 unified-framework reproduction summary figure.

A16 (2100 steps, t*=21) + A17-MODE (1300 steps, t*=13) vs the digitized
experiment: mean Cn and max(mean z)/c against angle of attack, the time
series with the statistics windows, mean/zsd maps and the vibration
spectrum at the max-zsd station.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path("artifacts/baselines/fluxv_v5m_rojratsirikul2011_unified_current")
FIG = OUT / "roj_unified_vs_experiment_summary.png"

# Digitized experiment (handoff §4.2 / frozen observations CSV).
EXP = {
    10.0: {"zmax": 0.032, "cn": (0.50, 0.52)},
    16.0: {"zmax": 0.043, "cn": (0.92, 0.95)},
    17.0: {"zmax": 0.0445, "cn": (0.97, 0.97)},
    23.0: {"zmax": 0.0475, "cn": (0.98, 1.02)},
}


def load(tag: str) -> tuple[dict, np.ndarray]:
    payload = json.loads((OUT / f"{tag}.json").read_text())
    archive = np.load(OUT / f"{tag}.z_history.npz")
    return payload, archive["z_history_over_c"], archive["time_star"]


def main() -> None:
    a16, z16, t16 = load("ROJ11_A16_FULL")
    a17, z17, t17 = load("ROJ11_A17_MODE_FULL")

    fig = plt.figure(figsize=(16.5, 10.0))
    grid = fig.add_gridspec(2, 3, hspace=0.34, wspace=0.27)

    # -- Cn vs alpha -------------------------------------------------------
    axis = fig.add_subplot(grid[0, 0])
    alphas = sorted(EXP)
    cn_lo = [EXP[a]["cn"][0] for a in alphas]
    cn_hi = [EXP[a]["cn"][1] for a in alphas]
    cn_mid = [(lo + hi) / 2 for lo, hi in zip(cn_lo, cn_hi)]
    axis.fill_between(alphas, cn_lo, cn_hi, color="#bbbbbb", alpha=0.5,
                      label="experiment (digitized band)")
    axis.plot(alphas, cn_mid, "o-", color="black", markersize=5, linewidth=1.2)
    for payload, color, label in (
        (a16, "#d62728", "unified A16 (t*=3–21 window)"),
        (a17, "#1f77b4", "unified A17-MODE (fallback window)"),
    ):
        alpha = float(payload["alpha_deg"])
        axis.plot([alpha], [payload["mean_Cn"]], "*", color=color, markersize=15,
                  label=label)
        axis.annotate(
            f"+{(payload['mean_Cn'] / ((EXP[alpha]['cn'][0] + EXP[alpha]['cn'][1]) / 2) - 1) * 100:.0f}%",
            (alpha, payload["mean_Cn"]), textcoords="offset points",
            xytext=(6, -12), color=color, fontsize=10,
        )
    axis.set_xlabel(r"angle of attack $\alpha$ (deg)")
    axis.set_ylabel(r"mean $C_n$")
    axis.set_title("Normal-force coefficient: +33–36% potential-flow offset")
    axis.grid(alpha=0.3)
    axis.legend(fontsize=8.5)

    # -- zmax/c vs alpha ---------------------------------------------------
    axis = fig.add_subplot(grid[0, 1])
    axis.fill_between(
        alphas,
        [EXP[a]["zmax"] - 0.005 for a in alphas],
        [EXP[a]["zmax"] + 0.005 for a in alphas],
        color="#bbbbbb", alpha=0.5, label="experiment ± project tolerance",
    )
    axis.plot(alphas, [EXP[a]["zmax"] for a in alphas], "o-", color="black",
              markersize=5, linewidth=1.2)
    for payload, color in ((a16, "#d62728"), (a17, "#1f77b4")):
        axis.plot([float(payload["alpha_deg"])], [payload["mean_zmax_over_c"]],
                  "*", color=color, markersize=15)
    axis.set_xlabel(r"$\alpha$ (deg)")
    axis.set_ylabel(r"$\max(\overline{z})/c$")
    axis.set_title("Mean camber: both cases INSIDE the ±0.005 gate")
    axis.grid(alpha=0.3)
    axis.legend(fontsize=9)

    # -- time series -------------------------------------------------------
    axis = fig.add_subplot(grid[0, 2])
    dt = 0.0688 / 5.0
    for payload, zhist, tstar, color, tag in (
        (a16, z16, t16, "#d62728", "A16"),
        (a17, z17, t17, "#1f77b4", "A17"),
    ):
        cn = np.array([r["cn"] for r in payload["records"]])
        axis.plot(tstar, cn, color=color, linewidth=0.9, label=f"{tag} $C_n$(t)")
        if payload["window_selection"]["stationary_window_found"]:
            start, stop = payload["window_selection"]["window"]
            axis.axvspan(tstar[start], tstar[stop - 1], color=color, alpha=0.10)
    axis.axhline(0.92, color="black", linestyle=":", linewidth=1)
    axis.axhline(0.95, color="black", linestyle=":", linewidth=1)
    axis.text(0.4, 0.935, "experiment A16 band", fontsize=8)
    axis.set_xlabel(r"$t^*$")
    axis.set_ylabel(r"$C_n$")
    axis.set_title("Time series (shaded = stationary statistics window)")
    axis.grid(alpha=0.3)
    axis.legend(fontsize=9)

    # -- maps + spectrum ---------------------------------------------------
    for column, (payload, zhist, color, tag) in enumerate(
        ((a16, z16, "#d62728", "A16"), (a17, z17, "#1f77b4", "A17-MODE"))
    ):
        axis = fig.add_subplot(grid[1, column])
        window = payload["window_selection"]["window"]
        span = zhist[window[0]:window[1]]
        mean_map = span.mean(axis=0)
        zsd_map = span.std(axis=0)
        chord_x = np.linspace(0, 1, mean_map.shape[0])
        axis.plot(chord_x, mean_map[:, mean_map.shape[1] // 2], color=color,
                  linewidth=2, label=r"mean $z/c$ (mid-span)")
        axis.fill_between(
            chord_x,
            (mean_map - 2 * zsd_map)[:, mean_map.shape[1] // 2],
            (mean_map + 2 * zsd_map)[:, mean_map.shape[1] // 2],
            color=color, alpha=0.18, label=r"±2 zsd",
        )
        ci, si = np.unravel_index(np.argmax(zsd_map), zsd_map.shape)
        spectrum = np.abs(np.fft.rfft(
            (span[:, ci, si] - span[:, ci, si].mean()) * np.hanning(span.shape[0])
        ))
        freqs = np.fft.rfftfreq(span.shape[0], d=dt)
        strouhal = freqs * 0.0688 / 5.0
        peak = 1 + spectrum[1:].argmax()
        axis2 = axis.twinx()
        axis2.plot(strouhal[1:], spectrum[1:] / spectrum[peak], color="#2ca02c",
                   linewidth=0.9, alpha=0.75)
        axis2.set_ylabel("spectrum (norm)", color="#2ca02c")
        axis2.axvline(0.85, color="#2ca02c", linestyle=":", linewidth=1)
        payload_st = payload["dominant_St"]
        axis.set_xlabel("x/c (chordwise, mid-span)")
        axis.set_ylabel(r"$z/c$")
        axis.set_title(
            f"{tag}: mean camber + vibration   St={payload_st:.3f}"
            if payload_st is not None else f"{tag}: mean camber + vibration"
        )
        axis.grid(alpha=0.3)
        axis.legend(fontsize=8.5, loc="lower right")

    axis = fig.add_subplot(grid[1, 2])
    axis.axis("off")
    rows = [
        ("", "A16 (α=16°)", "A17-MODE (α=17°)"),
        ("max(mean z)/c", f"{a16['mean_zmax_over_c']:.4f}", f"{a17['mean_zmax_over_c']:.4f}"),
        ("experiment", "0.043 ± 0.005", "0.0445 ± 0.005"),
        ("verdict", "PASS (+4.9%)", "PASS (−1.9%)"),
        ("", "", ""),
        ("mean Cn", f"{a16['mean_Cn']:.4f}", f"{a17['mean_Cn']:.4f}"),
        ("experiment", "0.92–0.95", "0.97"),
        ("verdict", "FAIL (+33%)", "FAIL (+36%)"),
        ("", "", ""),
        ("stationary window", "t*=3.0–21 (6.0 per.)", "NOT FOUND in t*≤13"),
        ("chordwise peaks", "1 (not gated)", "1 vs 2 → FAIL"),
        ("spanwise peaks", "1 (not gated)", "1 vs 0 → FAIL"),
        ("dominant St", "0.44 (not gated)", f"{a17['dominant_St']:.2f} vs 0.85 → FAIL"),
        ("", "", ""),
        ("physics/transfer/retention", "PASS", "PASS"),
        ("reproduction", "PARTIAL (exit 2)", "PARTIAL (exit 2)"),
    ]
    table = axis.table(
        cellText=rows[1:], colLabels=rows[0], loc="center", cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.6)
    table.scale(1, 1.35)
    axis.set_title("Gate summary", fontsize=11)

    fig.suptitle(
        "Rojratsirikul 2011 unified Q16–FLUX-V5M reproduction — mean quantities "
        "reproduced, Cn carries the +33% potential-flow class boundary",
        fontsize=12.5,
    )
    fig.savefig(FIG, dpi=170, bbox_inches="tight")
    print(FIG.resolve())


if __name__ == "__main__":
    main()
