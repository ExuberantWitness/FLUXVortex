"""Discovery hero figure: the 抗风×效率 (gust-rejection x efficiency) Pareto frontier
produced by the meta-RL policy adapting across the wing-design distribution.

Loads the trained RL^2 meta-policy, sweeps wing stiffness designs, evaluates each with
the meta-policy (few-shot adaptation, no retraining) averaged over gust realizations,
and plots the Pareto frontier — the plan's killer visual: diverse designs spread along
the gust x efficiency front. Also plots the meta-RL learning curve.
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

from meta_rl_train import (MetaFlightEnv, RL2Policy, CtxEmbedder,   # noqa: E402
                           cruise_efficiency)


def eval_design(net, s, n_seed=5):
    """Average controlled gust excursion for design s over gust realizations."""
    emb = CtxEmbedder(); exc = []
    for seed in range(n_seed):
        env = MetaFlightEnv(seed=10 + seed)
        obs = env.reset(design=s); emb.reset(); gz = []
        for k in range(env.horizon):
            with torch.no_grad():
                mu, _ = net(emb.push(obs).unsqueeze(0))
            obs, r, d, info = env.step(mu[0].numpy()); emb.record(mu[0].numpy(), r)
            if env.gust["t0"] <= env.t < env.gust["t0"] + env.gust["dur"] + 0.6:
                gz.append(env.x[2])
            if d:
                break
        if gz:
            exc.append(max(gz) - min(gz))
    return float(np.mean(exc)) if exc else np.nan, float(np.std(exc)) if exc else 0.0


def main():
    import warp as wp; wp.init()
    net = RL2Policy()
    net.load_state_dict(torch.load(os.path.join(_FLUXV, "docs", "meta_policy.pt")))
    net.eval()
    designs = np.array([0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 2.0])
    gust, gstd, eff = [], [], []
    for s in designs:
        g, gs = eval_design(net, float(s))
        gust.append(g); gstd.append(gs); eff.append(cruise_efficiency(float(s)))
        print(f"  s={s:.2f}: gust excursion={g:.2f}+-{gs:.2f}m  L/D={eff[-1]:.1f}", flush=True)
    gust, eff = np.array(gust), np.array(eff)

    fig, (ax, axl) = plt.subplots(1, 2, figsize=(13, 5.2))
    # Pareto frontier: efficiency (x, higher better) vs gust rejection (y = -excursion)
    sc = ax.scatter(eff, gust, c=designs, cmap="viridis", s=140, edgecolor="k", zorder=3)
    ax.plot(eff, gust, "k--", alpha=0.4, zorder=2)
    for s, e, g in zip(designs, eff, gust):
        ax.annotate(f"s={s:.1f}", (e, g), textcoords="offset points", xytext=(6, 5),
                    fontsize=8)
    ax.errorbar(eff, gust, yerr=gstd, fmt="none", ecolor="0.5", alpha=0.5, zorder=1)
    ax.set_xlabel("cruise efficiency  L/D  →  (better)")
    ax.set_ylabel("gust excursion (m)  ←  (better, lower)")
    ax.invert_yaxis()
    ax.set_title("抗风 × 效率 Pareto frontier\n(meta-policy adapts per design, no retraining)",
                 fontsize=11)
    cb = plt.colorbar(sc, ax=ax); cb.set_label("wing stiffness scale s")
    ax.grid(alpha=0.3)
    ax.text(0.04, 0.06, "flexible: gust-tolerant, lower L/D\nstiff: efficient, more gust",
            transform=ax.transAxes, fontsize=9, va="bottom",
            bbox=dict(boxstyle="round", fc="#ffffcc", alpha=0.8))
    # meta-RL learning curve
    try:
        h = np.load(os.path.join(_FLUXV, "docs", "ppo_hist.npz"))["hist"]
        axl.plot(h, color="#3060c0")
        axl.set_title("PPO/meta-RL learning curve")
        axl.set_xlabel("iteration"); axl.set_ylabel("mean episode return")
        axl.grid(alpha=0.3)
    except Exception:
        axl.axis("off")
    plt.tight_layout()
    out = os.path.join(_FLUXV, "docs", "codesign_frontier.png")
    plt.savefig(out, dpi=110)
    np.savez(os.path.join(_FLUXV, "docs", "codesign_frontier.npz"),
             stiffness=designs, gust=gust, gust_std=np.array(gstd), efficiency=eff)
    print(f"\nsaved hero figure -> {out}")
    # Pareto check: is there a real trade-off (best-gust != best-efficiency)?
    bg, be = int(np.nanargmin(gust)), int(np.argmax(eff))
    print(f"best gust rejection: s={designs[bg]:.1f}; best efficiency: s={designs[be]:.1f} "
          f"-> {'REAL TRADE-OFF (抗风×效率)' if bg != be else 'no trade-off'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
