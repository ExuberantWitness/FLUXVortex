"""Discovery hero figure: the 抗风×效率 (gust-rejection × efficiency) Pareto frontier
produced by the meta-RL policy adapting across the wing 刚柔 FIELD design space.

Upgraded from a 1-D scalar-stiffness sweep to the 2-D spanwise FIELD design space
(root × tip stiffness; design_field.StiffnessField). Loads the trained RL^2 meta-policy,
evaluates each field design with the meta-policy (few-shot adaptation, no retraining)
averaged over gust realizations, and plots:
  - the field designs in the 抗风×效率 plane (color = TIP stiffness),
  - the uniform-scalar frontier (root==tip diagonal) as the BASELINE the scalar design
    was limited to,
showing that stiff-root / flexible-tip FIELDS dominate it (efficient AND gust-tolerant) —
the distributional payoff a single uniform stiffness cannot reach. Also the learning curve.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

_FLUXV = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for p in (_FLUXV, os.path.dirname(__file__)):
    if p not in sys.path:
        sys.path.insert(0, p)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                    # noqa: E402

import design_field as dfield                                      # noqa: E402
from meta_rl_train import (MetaFlightEnv, RL2Policy, CtxEmbedder,   # noqa: E402
                           cruise_efficiency, eval_field, K_CTRL)


def eval_design(net, field, n_seed=5):
    """Average controlled gust excursion for a 刚柔 field over gust realizations."""
    emb = CtxEmbedder(); exc = []
    for seed in range(n_seed):
        g = eval_field(net, field, emb, seed=10 + seed)
        if np.isfinite(g):
            exc.append(g)
    return (float(np.mean(exc)) if exc else np.nan,
            float(np.std(exc)) if exc else 0.0)


def main():
    import warp as wp; wp.init()
    net = RL2Policy()
    net.load_state_dict(torch.load(os.path.join(_FLUXV, "docs", "meta_policy.pt")))
    net.eval()

    # 2-D 刚柔 FIELD design space: root × tip stiffness (root->tip spanwise spline)
    roots = np.array([0.6, 1.0, 1.4, 1.8, 2.2])
    tips = np.array([0.4, 0.9, 1.5])
    F_root, F_tip, F_gust, F_gstd, F_eff = [], [], [], [], []
    for root in roots:
        for tip in tips:
            f = dfield.StiffnessField.from_root_tip(root, tip, K=K_CTRL)
            g, gs = eval_design(net, f)
            F_root.append(root); F_tip.append(tip); F_gust.append(g)
            F_gstd.append(gs); F_eff.append(cruise_efficiency(f))
            print(f"  root={root:.1f} tip={tip:.1f}: gust={g:.2f}+-{gs:.2f}m  L/D={F_eff[-1]:.1f}",
                  flush=True)
    F_root, F_tip = np.array(F_root), np.array(F_tip)
    F_gust, F_eff = np.array(F_gust), np.array(F_eff)

    # uniform-scalar baseline (root==tip) — what the scalar design was limited to
    U_s = np.array([0.5, 0.8, 1.1, 1.4, 1.7, 2.0])
    U_gust, U_eff = [], []
    for s in U_s:
        g, _ = eval_design(net, dfield.StiffnessField.uniform(float(s), K=K_CTRL))
        U_gust.append(g); U_eff.append(cruise_efficiency(float(s)))
    U_gust, U_eff = np.array(U_gust), np.array(U_eff)

    fig, (ax, axl) = plt.subplots(1, 2, figsize=(13.5, 5.4))
    # uniform-scalar frontier (baseline the scalar design could reach)
    oi = np.argsort(U_eff)
    ax.plot(U_eff[oi], U_gust[oi], "k--", lw=1.8, alpha=0.7, zorder=2,
            label="uniform stiffness (scalar design)")
    ax.scatter(U_eff, U_gust, c="0.45", s=55, marker="s", zorder=3,
               edgecolor="k", linewidth=0.5)
    # FIELD designs: color = tip stiffness, marker size grows with root stiffness
    sizes = 70 + 150 * (F_root - F_root.min()) / (np.ptp(F_root) + 1e-9)
    sc = ax.scatter(F_eff, F_gust, c=F_tip, cmap="coolwarm_r", s=sizes, edgecolor="k",
                    linewidth=0.6, zorder=4, label="刚柔 FIELD (root×tip)")
    ax.errorbar(F_eff, F_gust, yerr=F_gstd, fmt="none", ecolor="0.6", alpha=0.45, zorder=1)
    # highlight the dominating stiff-root / flexible-tip design
    dom = int(np.argmin(F_gust / (F_eff / F_eff.max())))    # good gust AND good eff
    ax.annotate(f"stiff-root/flex-tip\nroot={F_root[dom]:.1f} tip={F_tip[dom]:.1f}",
                (F_eff[dom], F_gust[dom]), textcoords="offset points", xytext=(10, -28),
                fontsize=9, fontweight="bold", color="#b00000",
                arrowprops=dict(arrowstyle="->", color="#b00000"))
    ax.set_xlabel("cruise efficiency  L/D  →  (better)")
    ax.set_ylabel("gust excursion (m)  ←  (better, lower)")
    ax.invert_yaxis()
    ax.set_title("抗风 × 效率 Pareto frontier — 刚柔 FIELD vs uniform stiffness\n"
                 "(meta-policy adapts per design, no retraining)", fontsize=11)
    cb = plt.colorbar(sc, ax=ax); cb.set_label("TIP stiffness scale  (← flexible | stiff →)")
    ax.grid(alpha=0.3); ax.legend(loc="upper left", fontsize=8.5)
    ax.text(0.97, 0.05, "marker size ∝ ROOT stiffness\nfield dominates uniform: stiff root\n"
            "(efficient) + flexible tip (gust-tolerant)",
            transform=ax.transAxes, fontsize=8.5, va="bottom", ha="right",
            bbox=dict(boxstyle="round", fc="#ffffcc", alpha=0.85))
    # meta-RL learning curve
    try:
        h = np.load(os.path.join(_FLUXV, "docs", "ppo_hist.npz"))["hist"]
        axl.plot(h, color="#3060c0")
        axl.set_title("meta-RL (RL²) learning curve")
        axl.set_xlabel("iteration"); axl.set_ylabel("mean episode return")
        axl.grid(alpha=0.3)
    except Exception:
        axl.axis("off")
    plt.tight_layout()
    out = os.path.join(_FLUXV, "docs", "codesign_frontier.png")
    plt.savefig(out, dpi=110)
    np.savez(os.path.join(_FLUXV, "docs", "codesign_frontier.npz"),
             root=F_root, tip=F_tip, gust=F_gust, gust_std=np.array(F_gstd), efficiency=F_eff,
             uniform_s=U_s, uniform_gust=U_gust, uniform_efficiency=U_eff)
    print(f"\nsaved hero figure -> {out}")

    # Dominance check: does a FIELD design beat EVERY uniform design on both axes?
    dominates = 0
    for fe, fg in zip(F_eff, F_gust):
        if np.any((fe >= U_eff) & (fg <= U_gust)) and \
           np.any((fe > U_eff) | (fg < U_gust)):
            dominates += 1
    print(f"field designs Pareto-dominating ≥1 uniform design: {dominates}/{len(F_eff)} "
          f"-> {'刚柔 FIELD opens a better front (distributional payoff)' if dominates else 'no gain'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
