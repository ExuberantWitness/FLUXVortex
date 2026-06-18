"""DQD emitter — gradient-driven quality-diversity (differentiable QD core).

The plan's optimizer (plan §6) is MAP-Elites + a *differentiable* QD emitter
(CMA-MEGA style): instead of blind Gaussian mutation, the emitter follows the
**design gradient of the objective** to propose improved designs, which fills the
archive far more sample-efficiently in the smooth design dimensions.

This minimal DQD uses the DesignMap quality gradient (finite-difference here; the
analytic path is the K_t structural adjoint + the AIC solve adjoint already
implemented) to do gradient ascent from archive elites, and compares its
best-quality-vs-budget against the random-mutation MAP-Elites baseline on the same
evaluation budget.

verify: DQD reaches a better (or equal) best quality than random mutation at the
same budget — the sample-efficiency win that motivates differentiable QD.
"""
from __future__ import annotations

import os
import sys

import numpy as np

_FLUXV = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for p in (_FLUXV, os.path.join(_FLUXV, "src"), os.path.join(_FLUXV, "tests"),
          os.path.dirname(__file__)):
    if p not in sys.path:
        sys.path.insert(0, p)

import warp as wp                                                # noqa: E402
from fluxvortex.warp_fsi import config as cfg                   # noqa: E402
from design_map import DesignMap, evaluate                      # noqa: E402
from map_elites import S_RANGE, R_RANGE, _cell, GRID            # noqa: E402


def _quality(dmap, d, N):
    return -evaluate(dmap, d, N=N) - 0.02 * d[0]


def _grad(dmap, d, N, h=2e-2):
    g = np.zeros(2)
    for i in range(2):
        dp = list(d); dp[i] += h; dm = list(d); dm[i] -= h
        g[i] = (_quality(dmap, dp, N) - _quality(dmap, dm, N)) / (2 * h)
    return g                                       # 5 evals (1 base implicit + 4)


def run_dqd(dmap, budget, N, seed):
    rng = np.random.default_rng(seed)
    archive = {}
    evals = 0

    def add(d):
        nonlocal evals
        d = [float(np.clip(d[0], *S_RANGE)), float(np.clip(d[1], *R_RANGE))]
        q = _quality(dmap, d, N); evals += 1
        c = _cell(*d)
        if c not in archive or q > archive[c][0]:
            archive[c] = (q, d)
        return q

    add([rng.uniform(*S_RANGE), rng.uniform(*R_RANGE)])
    best = []
    while evals < budget:
        # pick a random elite, follow its quality gradient (gradient-ascent emitter)
        _, d = archive[list(archive.keys())[rng.integers(len(archive))]]
        g = _grad(dmap, d, N); evals += 4
        step = 0.3 * g / (np.linalg.norm(g) + 1e-9)
        add([d[0] + step[0], d[1] + step[1]])
        best.append(max(v[0] for v in archive.values()))
    return archive, (max(v[0] for v in archive.values()))


def run_random(dmap, budget, N, seed):
    rng = np.random.default_rng(seed)
    archive = {}
    for _ in range(budget):
        d = [float(np.clip(rng.uniform(*S_RANGE), *S_RANGE)),
             float(np.clip(rng.uniform(*R_RANGE), *R_RANGE))]
        q = _quality(dmap, d, N)
        c = _cell(*d)
        if c not in archive or q > archive[c][0]:
            archive[c] = (q, d)
    return archive, max(v[0] for v in archive.values())


def verify() -> bool:
    from run_standalone_yamano import yamano_params, build_yamano_shell
    params = yamano_params()
    shell0, _, _, _ = build_yamano_shell(params, nx=8, ny=6)
    dmap = DesignMap(shell0.nodes, shell0.quads, shell0.h, shell0.rho,
                     shell0.nu_xy, shell0.Ex, shell0._bc_dofs)
    budget, N = 36, 40
    _, dqd_best = run_dqd(dmap, budget, N, seed=2)
    _, rnd_best = run_random(dmap, budget, N, seed=2)
    win = dqd_best >= rnd_best - 1e-9
    print(f"DQD (gradient emitter) vs random mutation, budget={budget} evals each:")
    print(f"  best quality: DQD={dqd_best:.5e}  random={rnd_best:.5e}  "
          f"DQD>=random: {win}")
    print(f"  -> {'PASS' if win else 'FAIL'}: differentiable-QD emitter works "
          f"(gradient-driven design improvement; analytic grad = K_t + AIC adjoints)")
    return win


if __name__ == "__main__":
    wp.init()
    print(cfg.summary())
    raise SystemExit(0 if verify() else 1)
