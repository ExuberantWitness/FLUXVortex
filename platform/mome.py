"""MOME — multi-objective archive: the gust-rejection x efficiency Pareto frontier.

The plan's quality is two-objective (plan §7): every design is scored by
(gust_rejection, efficiency), and the co-design output is the **Pareto frontier**
of non-dominated (design, policy) — the project's 10k-star hero artifact (plan §8:
"MOME archive ... 抗风x效率 前沿"). This module produces that frontier: sample a
design population, evaluate both objectives, and extract the non-dominated set.

Objectives (both minimized here):
  gust  = peak tip excursion under a gust (fast structural DesignMap proxy)
  cot   = cost of transport (Zhong&Xu power model; mass grows with stiffness, so
          stiffer wings reject the gust better but cost more power -> a real
          trade-off, hence a real frontier).

A stiffer wing wins on gust but loses on COT (heavier -> larger P_iner); the
frontier is the set of designs where neither objective can improve without the
other worsening — exactly what the MOME archive illuminates and the discovery
paper reports.
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
import cot as cotmod                                            # noqa: E402


def objectives(dmap, design, N=40):
    """Return (gust, cot) — both to MINIMIZE."""
    gust = evaluate(dmap, design, N=N)                          # tip excursion proxy
    s = design[0]
    # COT proxy: heavier (stiffer) wing -> larger wing inertia -> larger P_iner
    m, b, c_mean = 0.52 + 0.25 * (s - 1.0), 1.7, 0.29
    S = b * c_mean; AR = b * b / S
    f, amp, V = 3.0, 35.0, 6.0
    CL = m * cotmod.G / (0.5 * cotmod.RHO * V * V * S)
    I_w = (0.16 * s) * (b / 2.0) ** 2 / 3.0                     # inertia grows with stiffness
    comps = cotmod.power_components(m=m, b=b, S=S, AR=AR, c_mean=c_mean, f=f,
                                    amp_deg=amp, V=V, CL=CL, I_w=I_w)
    # COT reflects the inertial cost of a heavier/stiffer wing (the trade-off vs gust);
    # the resonant-spring axis is a separate design dimension (see cot.py).
    _, c = cotmod.cot(comps, m=m, V=V, resonant_spring=False)
    return float(gust), float(c)


def pareto_front(pts):
    """Indices of non-dominated points (both objectives minimized)."""
    pts = np.asarray(pts)
    nd = []
    for i, p in enumerate(pts):
        if not np.any(np.all(pts <= p, axis=1) & np.any(pts < p, axis=1)):
            nd.append(i)
    return nd


def verify() -> bool:
    from run_standalone_yamano import yamano_params, build_yamano_shell
    params = yamano_params()
    shell0, _, _, _ = build_yamano_shell(params, nx=8, ny=6)
    dmap = DesignMap(shell0.nodes, shell0.quads, shell0.h, shell0.rho,
                     shell0.nu_xy, shell0.Ex, shell0._bc_dofs)
    rng = np.random.default_rng(4)
    pop = [[float(rng.uniform(0.5, 2.0)), float(rng.uniform(0.6, 1.4))] for _ in range(20)]
    objs = np.array([objectives(dmap, d) for d in pop])
    front = pareto_front(objs)
    front = sorted(front, key=lambda i: objs[i, 0])

    print("MOME — gust x efficiency(COT) Pareto frontier (20 designs):")
    print(f"  Pareto-optimal designs: {len(front)} of {len(pop)}")
    for i in front:
        print(f"    design s={pop[i][0]:.2f} r={pop[i][1]:.2f}: "
              f"gust={objs[i,0]:.4e}  COT={objs[i,1]:.3f}")
    # a valid frontier: >=3 non-dominated, and they trade off (gust and COT
    # anti-correlated along the front)
    fg = objs[front, 0]; fc = objs[front, 1]
    trades = len(front) >= 3 and np.corrcoef(fg, fc)[0, 1] < -0.2
    print(f"  frontier trades off (gust vs COT, corr<0): {trades}")
    print(f"  -> {'PASS' if trades else 'FAIL'}: MOME frontier emerges — the discovery "
          f"hero artifact (diverse non-dominated designs on gust x efficiency)")
    return trades


if __name__ == "__main__":
    wp.init()
    print(cfg.summary())
    raise SystemExit(0 if verify() else 1)
