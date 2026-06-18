"""Discovery run #5 — the REAL (gust x aerodynamic-efficiency) trade-off.

Fixes defect D1 (docs/fix_plan.md): prior runs scored "efficiency" as a force-norm
proxy (#1/#2) or an analytical resonance COT (#3). This run sweeps wing stiffness and,
for each design, runs the validated coupled FSI at a cruise AoA to measure BOTH
co-design objectives from the SAME real rollout:

  gust_rejection = peak tip excursion under a 1-cosine gust  (smaller better)
  L/D            = cruise aerodynamic efficiency (induced from UVLM + profile strip)

The scientific question this settles, on real physics: does aerodynamic efficiency
favor a DIFFERENT design than gust rejection? If a flexible wing rejects gusts (F1)
but its load-induced washout/twist degrades L/D, that is a genuine competing
constraint — the precondition for the headline ranking-inversion synergy that the
single-objective single-wing setup (#1/#2) lacked.
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
from power_probe import evaluate_real                           # noqa: E402

STIFF = [0.5, 0.8, 1.1, 1.4, 1.7, 2.0]
AOA_DEG = 6.0


def run():
    from run_standalone_yamano import yamano_params, build_yamano_shell
    params = yamano_params()
    shell0, _, _, _ = build_yamano_shell(params, nx=15, ny=10)
    dmap = DesignMap(shell0.nodes, shell0.quads, shell0.h, shell0.rho,
                     shell0.nu_xy, shell0.Ex, shell0._bc_dofs)
    gust, ld, ldt, lift, drag = [], [], [], [], []
    t0 = time.time()
    for s in STIFF:
        r = evaluate_real(dmap, [s, 1.0], aoa_deg=AOA_DEG, n_base=3, n_gust=3,
                          n_recover=3, gust_w=2.5)
        gust.append(r["gust_rejection"]); ld.append(r["L_over_D_induced"])
        ldt.append(r["L_over_D"]); lift.append(r["lift"]); drag.append(r["drag_induced"])
        print(f"  s={s:.2f}: gust={r['gust_rejection']:.4e}  L={r['lift']:+.3e}N  "
              f"D_ind={r['drag_induced']:+.3e}N  L/D_ind={r['L_over_D_induced']:+.2f}  "
              f"L/D_tot={r['L_over_D']:+.2f}  ({time.time()-t0:.0f}s)", flush=True)
    return (np.array(STIFF), np.array(gust), np.array(ld),
            np.array(lift), np.array(drag))


def analyze(stiff, gust, ld):
    bg = int(np.argmin(gust))         # best gust rejection (smaller better)
    be = int(np.argmax(ld))           # best efficiency (larger L/D better)
    print("\n=== REAL (gust x L/D) landscape, cruise AoA, coupled FSI ===")
    print("  stiffness : " + "  ".join(f"{s:.2f}" for s in stiff))
    print("  gust      : " + "  ".join(f"{v:.2e}" for v in gust))
    print("  L/D       : " + "  ".join(f"{v:+.2f}" for v in ld))
    print("\n=== FINDING ===")
    print(f"  best GUST rejection : stiffness={stiff[bg]:.2f}")
    print(f"  best EFFICIENCY L/D : stiffness={stiff[be]:.2f}")
    tradeoff = bg != be
    if tradeoff:
        print(f"  -> REAL TRADE-OFF: gust favors {stiff[bg]:.2f}, efficiency favors "
              f"{stiff[be]:.2f} — competing constraint confirmed on real physics.")
    else:
        print(f"  -> no trade-off: stiffness {stiff[bg]:.2f} wins both "
              f"(flexibility still dominates; need the flapping/resonance axis).")
    return tradeoff


if __name__ == "__main__":
    wp.init()
    print(cfg.summary())
    print(f"discovery #5: {len(STIFF)} stiffnesses, real (gust, L/D) from coupled FSI "
          f"at AoA={AOA_DEG} deg")
    stiff, gust, ld, lift, drag = run()
    np.savez(os.path.join(_FLUXV, "docs", "discovery5.npz"),
             stiff=stiff, gust=gust, ld=ld, lift=lift, drag=drag)
    raise SystemExit(0 if analyze(stiff, gust, ld) else 1)
