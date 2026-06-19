"""Capstone figure for the GRADIENT-DRIVEN differentiable co-design loop.

Left  — the spanwise 刚柔 FIELD evolving under Adam ascent on the validated analytic
        gradients: from a uniform start to the discovered stiff-root / flexible-tip shape.
Right — the 抗风×效率 Pareto front traced DIRECTLY by the gradient (one optimized design
        per loss weight λ/α), with the single-run optimization trajectory overlaid.

Everything is driven by the two Warp-tape design gradients validated bit-exact vs FD:
∂efficiency/∂ctrl (DQD) and ∂gust_factor/∂ctrl (SHAC). No grid search, no policy in the loop.
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

import warp as wp                                                 # noqa: E402
from codesign_optimize import optimize, K_CTRL                    # noqa: E402


def main():
    wp.init()
    xi = np.linspace(0.0, 1.0, K_CTRL)                            # root(0) -> tip(1)

    # 1) single balanced optimization, recording the field evolution
    ctrl0 = np.full((4, K_CTRL), 1.2)
    ctrl, hist, snaps = optimize(ctrl0, alpha=20.0, lam=1.0, iters=200, snap_every=20)
    print(f"discovered field: root={ctrl[:,0].mean():.2f} tip={ctrl[:,-1].mean():.2f}", flush=True)

    # 2) λ-sweep -> gradient-traced Pareto front (each weight -> one optimized design)
    lams = [0.15, 0.3, 0.6, 1.0, 2.0, 4.0, 8.0]
    pf_gust, pf_eff, pf_root, pf_tip = [], [], [], []
    for lam in lams:
        c, h = optimize(np.full((4, K_CTRL), 1.2), alpha=20.0, lam=lam, iters=160)
        pf_gust.append(h[-1, 1]); pf_eff.append(h[-1, 2])
        pf_root.append(c[:, 0].mean()); pf_tip.append(c[:, -1].mean())
        print(f"  λ/α={lam/20:.3f}: gust_factor={h[-1,1]:.3f} L/D={h[-1,2]:.2f}", flush=True)
    pf_gust = np.array(pf_gust); pf_eff = np.array(pf_eff)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.5, 5.3))

    # LEFT: field evolution (spanwise stiffness profile)
    cmap = plt.cm.viridis
    for k, (it, field) in enumerate(snaps):
        axL.plot(xi, field, "-o", color=cmap(k / max(1, len(snaps) - 1)),
                 lw=1.6, ms=4, alpha=0.85, label=f"iter {it}" if it in (0, snaps[-1][0]) else None)
    axL.plot(xi, snaps[0][1], "-o", color=cmap(0.0), lw=2.4, ms=6, label="start (uniform)")
    axL.plot(xi, snaps[-1][1], "-o", color=cmap(1.0), lw=2.4, ms=6,
             label="optimized (stiff-root/flex-tip)")
    axL.set_xlabel("span fraction  ξ   (root → tip)")
    axL.set_ylabel("stiffness scale  s(ξ)")
    axL.set_title("刚柔 FIELD discovered by gradient ascent\n"
                  "(uniform → stiff root, flexible tip)", fontsize=11)
    axL.grid(alpha=0.3); axL.legend(fontsize=8.5, loc="upper right")
    axL.annotate("stiff root → L/D", (0.02, snaps[-1][1][0]), fontsize=8.5, color="#b00000",
                 xytext=(0.12, snaps[-1][1][0] + 0.15),
                 arrowprops=dict(arrowstyle="->", color="#b00000"))
    axL.annotate("flexible tip →\npassive gust alleviation", (0.98, snaps[-1][1][-1]),
                 fontsize=8.5, color="#0050b0", ha="right",
                 xytext=(0.62, snaps[-1][1][-1] + 0.5),
                 arrowprops=dict(arrowstyle="->", color="#0050b0"))

    # RIGHT: gradient-traced Pareto front + single-run trajectory
    oi = np.argsort(pf_eff)
    axR.plot(pf_eff[oi], pf_gust[oi], "k--", alpha=0.5, zorder=2)
    sc = axR.scatter(pf_eff, pf_gust, c=np.log10([l / 20 for l in lams]), cmap="coolwarm",
                     s=130, edgecolor="k", zorder=4, label="gradient-traced Pareto (per λ/α)")
    axR.plot(hist[:, 2], hist[:, 1], "-", color="0.4", lw=1.4, alpha=0.7, zorder=3,
             label="single-run ascent trajectory")
    axR.scatter([hist[0, 2]], [hist[0, 1]], marker="*", s=180, c="lime", edgecolor="k",
                zorder=5, label="start (uniform s=1.2)")
    axR.scatter([hist[-1, 2]], [hist[-1, 1]], marker="*", s=180, c="red", edgecolor="k",
                zorder=5, label="optimized")
    axR.set_xlabel("cruise efficiency  L/D  →  (better)")
    axR.set_ylabel("gust transmissibility  (better, lower)")
    axR.invert_yaxis()
    axR.set_title("抗风 × 效率 Pareto — traced by analytic gradients\n"
                  "(DQD ∂L/D/∂ctrl + SHAC ∂gust/∂ctrl, no grid search)", fontsize=11)
    cb = plt.colorbar(sc, ax=axR); cb.set_label("log10(λ/α)  (← 抗风 weight | 效率 weight →)")
    axR.grid(alpha=0.3); axR.legend(fontsize=8, loc="upper left")

    plt.tight_layout()
    out = os.path.join(_FLUXV, "docs", "codesign_gradient.png")
    plt.savefig(out, dpi=110)
    np.savez(os.path.join(_FLUXV, "docs", "codesign_gradient.npz"),
             xi=xi, field_start=snaps[0][1], field_opt=snaps[-1][1],
             traj_eff=hist[:, 2], traj_gust=hist[:, 1],
             pareto_eff=pf_eff, pareto_gust=pf_gust, pareto_root=np.array(pf_root),
             pareto_tip=np.array(pf_tip), lams=np.array(lams))
    print(f"\nsaved capstone figure -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
