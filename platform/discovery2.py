"""Discovery run #2 — the structure-control SYNERGY (passive vs controlled).

The headline co-design question (plan §0): does closing the control loop change
which wing design is best? Passively, flexibility helps gust rejection (discovery
#1). But with an active gust-rejection policy in the loop, the design that wins may
shift — a stiffer wing may have more control authority, or a flexible wing may
over-respond. Where the controlled ranking differs from the passive one is the
structure-control synergy a decoupled optimizer cannot see.

For each stiffness we run the REAL coupled FSI under the same 1-cosine gust, once
passive and once with the Takens-embedding PD policy (control_eval), and compare
the gust-induced peak excursion and the design ranking.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

_FLUXV = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for p in (_FLUXV, os.path.join(_FLUXV, "src"), os.path.join(_FLUXV, "tests"),
          os.path.dirname(__file__)):
    if p not in sys.path:
        sys.path.insert(0, p)

import warp as wp                                                # noqa: E402
from fluxvortex.warp_fsi import config as cfg                   # noqa: E402
from design_map import DesignMap                                # noqa: E402
from control_eval import _run as controlled_run, TakensPolicy   # noqa: E402

STIFF = [0.5, 0.8, 1.1, 1.4, 1.7, 2.0]


def run():
    from run_standalone_yamano import yamano_params, build_yamano_shell
    params = yamano_params()
    shell0, _, _, _ = build_yamano_shell(params, nx=15, ny=10)
    dmap = DesignMap(shell0.nodes, shell0.quads, shell0.h, shell0.rho,
                     shell0.nu_xy, shell0.Ex, shell0._bc_dofs)
    passive, controlled = [], []
    t0 = time.time()
    for s in STIFF:
        p = controlled_run(dmap, [s, 1.0], None, n_base=1, n_gust=3, n_recover=2,
                           gust_w=2.5)
        c = controlled_run(dmap, [s, 1.0], TakensPolicy(n_embed=20), n_base=1,
                           n_gust=3, n_recover=2, gust_w=2.5)
        passive.append(p); controlled.append(c)
        print(f"  s={s:.2f}: passive={p:.4e}  controlled={c:.4e}  "
              f"reduction={100*(1-c/p):.0f}%  ({time.time()-t0:.0f}s)", flush=True)
    return np.array(STIFF), np.array(passive), np.array(controlled)


def analyze(stiff, passive, controlled):
    bp, bc = int(np.argmin(passive)), int(np.argmin(controlled))
    print("\n=== passive vs controlled gust rejection ===")
    print("  stiffness : " + "  ".join(f"{s:.2f}" for s in stiff))
    print("  passive   : " + "  ".join(f"{v:.2e}" for v in passive))
    print("  controlled: " + "  ".join(f"{v:.2e}" for v in controlled))
    print("  reduction : " + "  ".join(f"{100*(1-c/p):.0f}%"
                                       for p, c in zip(passive, controlled)))
    print("\n=== FINDINGS ===")
    print(f"  best PASSIVE design:    stiffness={stiff[bp]:.2f}")
    print(f"  best CONTROLLED design: stiffness={stiff[bc]:.2f}")
    synergy = bp != bc
    # does control reduce MORE on stiffer wings (control authority grows w/ stiffness)?
    red = 1 - controlled / passive
    auth_trend = float(np.corrcoef(stiff, red)[0, 1])
    if synergy:
        print(f"  -> SYNERGY: control SHIFTS the optimal design "
              f"({stiff[bp]:.2f} passive -> {stiff[bc]:.2f} controlled) — "
              f"structure-control co-design beats decoupled.")
    else:
        print(f"  -> optimal design unchanged by control (best={stiff[bp]:.2f}); "
              f"control reduces gust {np.mean(red)*100:.0f}% on average.")
    print(f"  control authority vs stiffness (corr of reduction with stiffness) = "
          f"{auth_trend:+.2f}  "
          f"({'stiffer wings control better' if auth_trend > 0.2 else 'flexible control better' if auth_trend < -0.2 else 'authority ~stiffness-independent'})")
    return synergy, auth_trend


if __name__ == "__main__":
    wp.init()
    print(cfg.summary())
    print(f"discovery #2: {len(STIFF)} stiffnesses x (passive, controlled), real coupled FSI")
    s, p, c = run()
    np.savez(os.path.join(_FLUXV, "docs", "discovery2.npz"), stiff=s, passive=p, controlled=c)
    analyze(s, p, c)
