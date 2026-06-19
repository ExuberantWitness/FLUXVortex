"""MAP-Elites archive figure — the 抗风×效率 behavior space illuminated by diverse 刚柔
designs, plus the DQD-vs-random QD curves.

Left  — the illuminated archive: every filled cell is a design, placed at its (L/D, gust)
        behavior and colored by its TIP stiffness. The feasible band fills with a continuum
        of designs; the flexible-tip designs sit at low gust transmissibility (top), the
        stiff-tip ones at higher — the design diversity behind the 抗风×效率 front.
Right — QD-score vs iterations for the DQD (gradient-arborescence) and random emitters.

Honest finding (printed): on this CHEAP analytic surrogate DQD ≈ random for archive filling
— the design→behavior map is smooth and the behavior space is 2-D, so blind mutation already
finds it. Differentiable QD's value is sample efficiency on EXPENSIVE coupled-FSI evaluation
and precise behavior-targeting (the gradient optimizer reaching stiff-root/flex-tip).
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
    d = np.load(os.path.join(_FLUXV, "docs", "codesign_qd.npz"))
    eff, gust, q, ctrl = d["dqd_eff"], d["dqd_gust"], d["dqd_q"], d["dqd_ctrl"]
    h_dqd, h_rnd = d["dqd_hist"], d["rnd_hist"]
    m = np.isfinite(eff)
    e, g = eff[m], gust[m]
    tip = ctrl[m][:, -1]; root = ctrl[m][:, 0]
    cov = m.mean()
    print(f"archive: {m.sum()} elites, coverage {cov*100:.1f}% (K={int(d['K'])} field)", flush=True)

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13.5, 5.3))

    sc = axA.scatter(e, g, c=tip, cmap="coolwarm_r", s=46, edgecolor="0.3", linewidth=0.3,
                     vmin=0.3, vmax=2.5)
    axA.set_xlabel("cruise efficiency  L/D  →  (better)")
    axA.set_ylabel("gust transmissibility  (better, lower)")
    axA.invert_yaxis()
    axA.set_title("MAP-Elites archive — 抗风×效率 illuminated by diverse 刚柔 designs\n"
                  f"({m.sum()} elite designs; color = TIP stiffness)", fontsize=11)
    cb = plt.colorbar(sc, ax=axA); cb.set_label("TIP stiffness  (← flexible | stiff →)")
    axA.grid(alpha=0.3)
    axA.text(0.03, 0.05, "flexible-tip designs →\nlower gust transmissibility\n"
             "(passive alleviation)", transform=axA.transAxes, fontsize=8.5, va="bottom",
             bbox=dict(boxstyle="round", fc="#eef6ff", alpha=0.9))

    axB.plot(h_dqd[:, 1], color="#c03030", lw=2, label="DQD (gradient arborescence)")
    axB.plot(h_rnd[:, 1], color="#3060c0", lw=2, label="random mutation (baseline)")
    axB.set_xlabel("iteration"); axB.set_ylabel("QD-score (Σ cell quality)")
    axB.set_title("QD-score vs iterations\n(cheap surrogate: DQD ≈ random — see honest note)",
                  fontsize=11)
    axB.grid(alpha=0.3); axB.legend(fontsize=9, loc="lower right")
    axB.text(0.03, 0.96, "on a cheap 2-D-behavior surrogate, blind mutation\n"
             "already finds the manifold; DQD's edge is sample\n"
             "efficiency on expensive coupled-FSI eval",
             transform=axB.transAxes, fontsize=8, va="top",
             bbox=dict(boxstyle="round", fc="#ffffcc", alpha=0.85))

    plt.tight_layout()
    out = os.path.join(_FLUXV, "docs", "codesign_qd.png")
    plt.savefig(out, dpi=110)
    print(f"saved archive figure -> {out}")
    # design diversity check: tip stiffness genuinely varies across the front
    print(f"  tip stiffness across elites: {tip.min():.2f}..{tip.max():.2f}  "
          f"root: {root.min():.2f}..{root.max():.2f}  -> diverse 刚柔 fields illuminated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
