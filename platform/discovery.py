"""Discovery run — characterize the gust-rejection landscape from REAL coupled FSI.

A focused scientific experiment on the 4090: sweep the wing design space
(stiffness scale x chord/span orthotropy) and, for each design, run the *real*
predictor-corrector coupled FSI under a 1-cosine vertical gust to measure the
gust-induced peak tip excursion. Then characterize the landscape and test the
plan's hypothesized non-intuitive structure:

  H1: passive gust rejection is NON-monotone in stiffness (a transient gust
      excites the structure, so the dynamic response depends on stiffness AND the
      modal/damping interaction — not "stiffer is always better").
  H2: orthotropy (chordwise vs spanwise stiffening) moves the gust response
      independently of the overall stiffness scale.

This is the real-physics version of codesign's objective (no analytical proxy).
Saves the full landscape to discovery.npz for the (gust x efficiency) frontier
and the discovery-paper figure.
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
from codesign_eval import evaluate as coupled_eval              # noqa: E402

STIFF = [0.5, 0.7, 0.9, 1.1, 1.4, 1.7, 2.0]
ORTHO = [0.8, 1.2]


def run():
    from run_standalone_yamano import yamano_params, build_yamano_shell
    params = yamano_params()
    shell0, _, _, _ = build_yamano_shell(params, nx=15, ny=10)   # coupled-FSI geom cache
    dmap = DesignMap(shell0.nodes, shell0.quads, shell0.h, shell0.rho,
                     shell0.nu_xy, shell0.Ex, shell0._bc_dofs)
    grid = np.full((len(ORTHO), len(STIFF)), np.nan)
    t0 = time.time()
    for j, r in enumerate(ORTHO):
        for i, s in enumerate(STIFF):
            res = coupled_eval(dmap, [s, r], n_base=1, n_gust=3, n_recover=2,
                               gust_w=2.5)
            grid[j, i] = res["gust_rejection"]
            print(f"  s={s:.2f} r={r:.1f}: gust={grid[j,i]:.4e}  "
                  f"({time.time()-t0:.0f}s)", flush=True)
    np.savez(os.path.join(_FLUXV, "docs", "discovery.npz"),
             stiff=np.array(STIFF), ortho=np.array(ORTHO), gust=grid)
    return np.array(STIFF), np.array(ORTHO), grid


def analyze(stiff, ortho, grid):
    print("\n=== gust-rejection landscape (real coupled FSI + gust) ===")
    print("  stiffness:   " + "  ".join(f"{s:.2f}" for s in stiff))
    findings = []
    for j, r in enumerate(ortho):
        row = grid[j]
        imin = int(np.nanargmin(row))
        monotone_inc = bool(np.all(np.diff(row) > 0))
        monotone_dec = bool(np.all(np.diff(row) < 0))
        interior = 0 < imin < len(stiff) - 1
        print(f"  ortho={r:.1f}: " + "  ".join(f"{v:.2e}" for v in row))
        print(f"            best gust @ stiffness={stiff[imin]:.2f} "
              f"(interior_optimum={interior}, monotone_inc={monotone_inc}, "
              f"monotone_dec={monotone_dec})")
        if interior:
            findings.append(f"ortho={r:.1f}: NON-monotone — interior optimal "
                            f"stiffness {stiff[imin]:.2f} for gust rejection (H1 ✓)")
        elif not (monotone_inc or monotone_dec):
            findings.append(f"ortho={r:.1f}: non-monotone gust(stiffness) (H1 ✓)")
    # H2: does orthotropy shift the response at fixed stiffness?
    if grid.shape[0] >= 2:
        shift = np.nanmean(np.abs(grid[0] - grid[1])) / (np.nanmean(grid) + 1e-30)
        findings.append(f"orthotropy shifts gust response by {shift:.0%} at fixed "
                        f"stiffness (H2 {'✓' if shift > 0.02 else '~'})")
    print("\n=== FINDINGS ===")
    for f in findings:
        print("  • " + f)
    return findings


if __name__ == "__main__":
    wp.init()
    print(cfg.summary())
    print(f"discovery sweep: {len(STIFF)}x{len(ORTHO)} designs, real coupled FSI + gust")
    stiff, ortho, grid = run()
    analyze(stiff, ortho, grid)
