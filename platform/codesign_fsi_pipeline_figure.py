"""Figure for the end-to-end real-coupled-FSI co-design pipeline (4090 run).

Left  — co-design score vs real-FSI evaluation (the search improving on true physics).
Right — the (lift, feather) the search visited, colored by tip MASS: shows the over-flex
        lift blow-up being tamed as the 质量 distribution is co-designed against the 刚柔,
        with the best design marked. Real coupled-FSI numbers throughout (≈70 s/eval).
"""
from __future__ import annotations

import os
import sys

import numpy as np

_FLUXV = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for p in (_FLUXV, os.path.dirname(__file__)):
    if p not in sys.path:
        sys.path.insert(0, p)

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt                                   # noqa: E402


def main():
    d = np.load(os.path.join(_FLUXV, "docs", "codesign_fsi_pipeline.npz"))
    ev = d["evals"]; best = d["best"]
    # columns: stiff_root, stiff_tip, mass_root, mass_tip, feather, lift, bend, score
    s_root, s_tip, m_root, m_tip, feather, lift, bend, score = ev.T
    finite = score > -1e5
    print(f"{len(ev)} real-FSI evals; best score={best[7]:+.2f} "
          f"刚柔=({best[0]:.1f},{best[1]:.1f}) 质量=({best[2]:.2f},{best[3]:.2f})", flush=True)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.5, 5.2))
    run_best = np.maximum.accumulate(np.where(finite, score, -1e6))
    axL.plot(np.arange(len(score)), np.where(finite, score, np.nan), "o-", color="#3060c0",
             ms=5, lw=1.2, label="per-eval score")
    axL.plot(np.arange(len(score)), run_best, "-", color="#c03030", lw=2, label="running best")
    axL.set_xlabel("real coupled-FSI evaluation #")
    axL.set_ylabel("co-design score (feather − over-flex/bend penalties)")
    axL.set_title("End-to-end co-design on the REAL coupled FSI (4090)\n"
                  "(刚柔 + 质量 search, ≈70 s/eval)", fontsize=11)
    axL.grid(alpha=0.3); axL.legend(fontsize=9, loc="lower right")

    XL = 200.0
    onscale = finite & (np.abs(lift) <= XL)
    offscale = int((finite & (np.abs(lift) > XL)).sum()) + int((~finite).sum())
    sc = axR.scatter(lift[onscale], feather[onscale], c=m_root[onscale], cmap="viridis", s=80,
                     edgecolor="k", linewidth=0.4)
    axR.scatter([best[5]], [best[4]], marker="*", s=340, c="red", edgecolor="k", zorder=5,
                label="best design")
    axR.axvspan(-XL * 1.5, -60, color="red", alpha=0.07)
    axR.axvspan(60, XL * 1.5, color="red", alpha=0.07)
    axR.set_xlim(-XL, XL)
    axR.set_xlabel("mean lift (N)   |   shaded = over-flex zone (|lift|>60 N)")
    axR.set_ylabel("passive feather amplitude (°)  — 气弹推进")
    axR.set_title("质量 distribution tames the over-flex lift blow-up\n"
                  f"(color = ROOT mass; {offscale} designs blew off-scale → rejected)", fontsize=11)
    cb = plt.colorbar(sc, ax=axR); cb.set_label("ROOT MASS scale  (← light | heavy →)")
    axR.grid(alpha=0.3); axR.legend(fontsize=9, loc="upper right")

    plt.tight_layout()
    out = os.path.join(_FLUXV, "docs", "codesign_fsi_pipeline.png")
    plt.savefig(out, dpi=110)
    print(f"saved -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
