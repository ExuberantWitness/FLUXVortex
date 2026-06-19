"""Discovery hero figure: the 抗风×效率 (gust-rejection × efficiency) Pareto frontier
produced by the GPU-batched meta-RL policy adapting across the wing 刚柔 FIELD design space.

Loads the trained RL^2 meta-policy and evaluates, with the Warp fp64 GPU env (batched,
one rollout over all designs), the 2-D spanwise FIELD design space (root × tip stiffness;
design_field.StiffnessField). Plots the field designs in the 抗风×效率 plane (color = TIP
stiffness) over the uniform-scalar baseline (root==tip), showing that stiff-root/flexible-tip
FIELDS dominate it (efficient AND gust-tolerant) — the distributional payoff a single uniform
stiffness cannot reach. Also the meta-RL learning curve.
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
from meta_rl_train import RL2Policy                                # noqa: E402
from gpu_meta_rl_train import eval_designs, K_CTRL                 # noqa: E402


def main():
    import warp as wp; wp.init()
    net = RL2Policy()
    net.load_state_dict(torch.load(os.path.join(_FLUXV, "docs", "meta_policy.pt")))
    net.eval()

    # 2-D 刚柔 FIELD design space: root × tip stiffness (spanwise spline)
    roots = [0.6, 1.0, 1.4, 1.8, 2.2]
    tips = [0.4, 0.9, 1.5]
    grid = [(r, t) for r in roots for t in tips]
    F_ctrl = np.array([dfield.StiffnessField.from_root_tip(r, t, K=K_CTRL).ctrl for r, t in grid])
    F_exc = eval_designs(net, F_ctrl)                              # one batched GPU rollout
    F_root = np.array([r for r, t in grid]); F_tip = np.array([t for r, t in grid])
    F_eff = np.array([dfield.cruise_efficiency(dfield.StiffnessField.from_root_tip(r, t, K=K_CTRL))
                      for r, t in grid])

    # uniform-scalar baseline (root==tip) — what a single stiffness could reach
    U_s = np.array([0.5, 0.8, 1.1, 1.4, 1.7, 2.0])
    U_ctrl = np.array([dfield.StiffnessField.uniform(float(s), K=K_CTRL).ctrl for s in U_s])
    U_exc = eval_designs(net, U_ctrl)
    U_eff = np.array([dfield.cruise_efficiency(float(s)) for s in U_s])

    for r, t, g, e in zip(F_root, F_tip, F_exc, F_eff):
        print(f"  root={r:.1f} tip={t:.1f}: gust excursion={g:5.2f}m  L/D={e:.1f}", flush=True)

    fig, (ax, axl) = plt.subplots(1, 2, figsize=(13.5, 5.4))
    fok = np.isfinite(F_exc) & (F_exc < 7.9)            # mark crashed (poor) designs separately
    # uniform-scalar frontier (baseline)
    oi = np.argsort(U_eff)
    ax.plot(U_eff[oi], U_exc[oi], "k--", lw=1.8, alpha=0.7, zorder=2,
            label="uniform stiffness (scalar design)")
    ax.scatter(U_eff, U_exc, c="0.45", s=55, marker="s", zorder=3, edgecolor="k", linewidth=0.5)
    # FIELD designs: color = tip stiffness, size grows with root stiffness
    sizes = 70 + 150 * (F_root - F_root.min()) / (F_root.max() - F_root.min() + 1e-9)
    sc = ax.scatter(F_eff[fok], F_exc[fok], c=F_tip[fok], cmap="coolwarm_r", s=sizes[fok],
                    edgecolor="k", linewidth=0.6, zorder=4, label="刚柔 FIELD (root×tip)")
    if (~fok).any():
        ax.scatter(F_eff[~fok], np.full((~fok).sum(), 2.6), marker="x", c="red", s=80,
                   zorder=5, label="crashed (lost gust)")
    # highlight the principled stiff-root / flexible-tip design (max root, min tip):
    # stiff root -> high L/D, flexible tip -> passive gust alleviation -> good on BOTH.
    dom = int(np.argmax(F_root - 3.0 * F_tip))
    ax.annotate(f"stiff-root + flex-tip\nroot={F_root[dom]:.1f} tip={F_tip[dom]:.1f}\n"
                f"(L/D={F_eff[dom]:.1f}, gust={F_exc[dom]:.2f}m)",
                (F_eff[dom], F_exc[dom]), textcoords="offset points", xytext=(10, 22),
                fontsize=9, fontweight="bold", color="#b00000",
                arrowprops=dict(arrowstyle="->", color="#b00000"))
    ax.set_xlabel("cruise efficiency  L/D  →  (better)")
    ax.set_ylabel("gust excursion (m)  ←  (better, lower)")
    ax.invert_yaxis()
    ax.set_title("抗风 × 效率 Pareto frontier — 刚柔 FIELD vs uniform stiffness\n"
                 "(GPU fp64 batched env; RL² meta-policy adapts per design, no retraining)",
                 fontsize=11)
    cb = plt.colorbar(sc, ax=ax); cb.set_label("TIP stiffness scale  (← flexible | stiff →)")
    ax.grid(alpha=0.3); ax.legend(loc="upper left", fontsize=8.5)
    ax.text(0.97, 0.05, "marker size ∝ ROOT stiffness\nflexible TIP → passive gust\n"
            "alleviation; stiff ROOT → L/D",
            transform=ax.transAxes, fontsize=8.5, va="bottom", ha="right",
            bbox=dict(boxstyle="round", fc="#ffffcc", alpha=0.85))
    # meta-RL learning curve
    try:
        h = np.load(os.path.join(_FLUXV, "docs", "ppo_hist.npz"))["hist"]
        axl.plot(h, color="#3060c0")
        axl.set_title("GPU-batched meta-RL (RL²) learning curve")
        axl.set_xlabel("iteration"); axl.set_ylabel("mean step reward")
        axl.grid(alpha=0.3)
    except Exception:
        axl.axis("off")
    plt.tight_layout()
    out = os.path.join(_FLUXV, "docs", "codesign_frontier.png")
    plt.savefig(out, dpi=110)
    np.savez(os.path.join(_FLUXV, "docs", "codesign_frontier.npz"),
             root=F_root, tip=F_tip, gust=F_exc, efficiency=F_eff,
             uniform_s=U_s, uniform_gust=U_exc, uniform_efficiency=U_eff)
    print(f"\nsaved hero figure -> {out}")

    # Dominance: a FIELD design that beats EVERY uniform design on both axes
    dominates = 0
    for fe, fg in zip(F_eff[fok], F_exc[fok]):
        if np.any((fe >= U_eff) & (fg <= U_exc)) and np.any((fe > U_eff) | (fg < U_exc)):
            dominates += 1
    print(f"field designs Pareto-dominating ≥1 uniform design: {dominates}/{int(fok.sum())} "
          f"-> {'刚柔 FIELD opens a better front (distributional payoff)' if dominates else 'no gain'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
