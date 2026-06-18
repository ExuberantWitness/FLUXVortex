"""codesign — the one-command co-design entry point (config-as-code + run).

Usable iteration-1 co-design (plan Layer-1/Layer-0): define a design space and the
two objectives, run a MOME quality-diversity search, and get the gust-rejection x
efficiency Pareto frontier + archive.

  from codesign import CoDesign
  cd = CoDesign(mode="fast")            # or "full" (real coupled FSI + gust)
  result = cd.run(budget=40)
  result.report()                       # prints the frontier; result.save("out.npz")

CLI:
  cd FLUXV/src
  FLUXV_DEVICE=cuda:0 python ../platform/codesign.py                 # fast, budget 40
  FLUXV_DEVICE=cuda:0 python ../platform/codesign.py --full --budget 8
  FLUXV_DEVICE=cuda:0 python ../platform/codesign.py --budget 60 --out frontier.npz

Design vector d = [stiffness_scale in [0.5,2.0], orthotropy_ratio in [0.6,1.4]].
Objectives (both MINIMIZED): gust = peak tip excursion under a 1-cosine gust;
cot = cost of transport (Zhong&Xu power model). The optimizer is MAP-Elites
(random init + Gaussian mutation; swap in the DQD gradient emitter from dqd.py)
maintaining a behavior grid; the reported result is the non-dominated frontier.
"""
from __future__ import annotations

import argparse
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
from design_map import DesignMap                                # noqa: E402
from mome import objectives as fast_objectives, pareto_front    # noqa: E402

S_RANGE, R_RANGE, GRID = (0.5, 2.0), (0.6, 1.4), 8


class Result:
    def __init__(self, designs, objs, front):
        self.designs, self.objs, self.front = designs, objs, front

    def report(self):
        print(f"\n=== co-design result: {len(self.front)} Pareto-optimal of "
              f"{len(self.designs)} evaluated ===")
        print(f"{'stiffness':>10} {'orthotropy':>11} {'gust':>11} {'COT':>8}")
        for i in sorted(self.front, key=lambda j: self.objs[j, 0]):
            d = self.designs[i]
            print(f"{d[0]:>10.3f} {d[1]:>11.3f} {self.objs[i,0]:>11.4e} "
                  f"{self.objs[i,1]:>8.3f}")

    def save(self, path):
        np.savez(path, designs=np.array(self.designs), objs=self.objs,
                 front=np.array(self.front))
        print(f"saved -> {path}")


class CoDesign:
    def __init__(self, mode="fast", nx=None, ny=None, seed=0, device=None):
        self.mode = mode
        self.device = device or cfg.DEVICE
        # full mode runs the coupled FSI, which needs the 15x10 geometry cache;
        # fast mode (structural proxy) can use a smaller/quicker mesh.
        if nx is None:
            nx, ny = (15, 10) if mode == "full" else (8, 6)
        from run_standalone_yamano import yamano_params, build_yamano_shell
        params = yamano_params()
        shell0, _, _, _ = build_yamano_shell(params, nx=nx, ny=ny)
        self.dmap = DesignMap(shell0.nodes, shell0.quads, shell0.h, shell0.rho,
                              shell0.nu_xy, shell0.Ex, shell0._bc_dofs)
        self.rng = np.random.default_rng(seed)

    def _eval(self, d):
        if self.mode == "full":
            from codesign_eval import evaluate as coupled_eval
            r = coupled_eval(self.dmap, d, device=self.device)
            # gust from the coupled FSI; efficiency objective from the COT model
            _, c = fast_objectives(self.dmap, d, N=20)
            return float(r["gust_rejection"]), float(c)
        return fast_objectives(self.dmap, d, N=40)               # fast structural proxy

    def run(self, budget=40, emitter="random"):
        """emitter: 'random' (Gaussian mutation) or 'dqd' (gradient-driven, plan §6).
        The DQD emitter follows the FD gradient of a scalarized objective
        (-gust - 0.01*COT) to propose sample-efficient improvements in the smooth
        design dimensions; the analytic gradient is the committed K_t/AIC adjoints."""
        archive, designs, objs = {}, [], []

        def add(d):
            d = [float(np.clip(d[0], *S_RANGE)), float(np.clip(d[1], *R_RANGE))]
            o = self._eval(d); designs.append(d); objs.append(o)
            cs = int(np.clip((d[0] - S_RANGE[0]) / (S_RANGE[1] - S_RANGE[0]) * GRID, 0, GRID - 1))
            cr = int(np.clip((d[1] - R_RANGE[0]) / (R_RANGE[1] - R_RANGE[0]) * GRID, 0, GRID - 1))
            archive.setdefault((cs, cr), []).append((d, o))
            if (len(designs)) % max(1, budget // 6) == 0:
                print(f"  evaluated {len(designs)}/{budget}", flush=True)
            return o

        def fitness(o):
            return -o[0] - 0.01 * o[1]               # scalarization for the emitter

        n_init = max(4, budget // 3)
        for _ in range(n_init):
            add([self.rng.uniform(*S_RANGE), self.rng.uniform(*R_RANGE)])
        while len(designs) < budget:
            keys = list(archive.keys())
            (d, o) = archive[keys[self.rng.integers(len(keys))]][-1]
            if emitter == "dqd" and len(designs) + 4 < budget:
                g = np.zeros(2); base = fitness(o)
                for i in range(2):                    # FD gradient of the fitness
                    dp = list(d); dp[i] += 0.06
                    g[i] = (fitness(add(dp)) - base) / 0.06
                stepdir = 0.3 * g / (np.linalg.norm(g) + 1e-9)
                add([d[0] + stepdir[0], d[1] + stepdir[1]])
            else:
                add([d[0] + 0.25 * self.rng.standard_normal(),
                     d[1] + 0.12 * self.rng.standard_normal()])
        objs = np.array(objs)
        return Result(designs, objs, pareto_front(objs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="use the real coupled FSI+gust eval")
    ap.add_argument("--dqd", action="store_true", help="DQD gradient emitter (vs random)")
    ap.add_argument("--budget", type=int, default=40)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    wp.init()
    print(cfg.summary())
    cd = CoDesign(mode="full" if args.full else "fast", seed=1)
    emitter = "dqd" if args.dqd else "random"
    print(f"running co-design: mode={'full' if args.full else 'fast'}, "
          f"emitter={emitter}, budget={args.budget}")
    res = cd.run(budget=args.budget, emitter=emitter)
    res.report()
    if args.out:
        res.save(args.out)
    return 0 if len(res.front) >= 3 else 1


if __name__ == "__main__":
    raise SystemExit(main())
