"""Newton demo: flexible plate in wind — lagged (ZOH) vs two-pass coupling.

A VBD cloth plate, clamped on one edge, in a uniform wind modeled by a
ring-vortex lifting surface (quasi-steady UVLM). The SAME arena is run with:
  - mode="lagged":   one aero solve per window, force held constant across
                     substeps (the zero-order-hold semantics of existing
                     coupled-solver frameworks)
  - mode="two-pass": predictor pass -> aero solve at predicted state ->
                     rewind -> corrector with per-substep force interpolation

plus a small-window reference (windows of 1 substep == per-step coupling) to
quantify which mode tracks the tightly-coupled limit better.

Run: python newton_pc/examples/plate_in_wind.py [--windows 40] [--substeps 16]
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import warp as wp

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _ROOT)
os.chdir(_ROOT)

import newton  # noqa: E402
from newton.solvers import SolverVBD  # noqa: E402

from newton_pc import WindowPredictorCorrector  # noqa: E402
from newton_pc.adapters.newton_vbd import (ParticleForceSet, UVLMProvider,  # noqa: E402
                                           VBDEntry)

NX = NY = 8
DT = 5e-4


def build_arena():
    b = newton.ModelBuilder()
    b.add_cloth_grid(pos=wp.vec3(0.0, 0.0, 0.0), rot=wp.quat_identity(),
                     vel=wp.vec3(0.0, 0.0, 0.0),
                     dim_x=NX, dim_y=NY, cell_x=0.1, cell_y=0.1, mass=0.005,
                     fix_left=True, tri_ke=5e2, tri_ka=5e2, tri_kd=1e-4,
                     edge_ke=5.0, edge_kd=1e-4)
    b.color()
    b.gravity = 0.0  # wind-only arena (builder.gravity is a scalar magnitude)
    model = b.finalize()
    solver = SolverVBD(model, iterations=8)
    entry = VBDEntry(model, solver, model.state(), model.state(),
                     model.control())
    provider = UVLMProvider(NX, NY, V_inf=[6.0, 0.0, 1.0], rho=1.225)
    n_part = (NX + 1) * (NY + 1)
    zero = ParticleForceSet(np.zeros((n_part, 3)))
    return entry, provider, zero


def run(mode: str, substeps: int, windows: int) -> np.ndarray:
    entry, provider, zero = build_arena()
    pc = WindowPredictorCorrector(entry=entry, provider=provider,
                                  substeps=substeps, dt=DT, mode=mode)
    pc.initialize(zero)
    tip_idx = (NX + 1) * (NY + 1) - 1
    tips = []
    pc.advance(n_substeps=1)
    for _ in range(windows):
        pc.advance()
        tips.append(entry.state()[tip_idx].copy())
    print(f"[{mode:>8} sub={substeps:3d}] tip(end)="
          f"{tips[-1]} solves={provider.n_solves}", flush=True)
    return np.array(tips)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", type=int, default=40)
    ap.add_argument("--substeps", type=int, default=16)
    args = ap.parse_args()

    # tightly-coupled reference: 1-substep windows, same total time
    n_total = args.windows * args.substeps
    ref = run("two-pass", 1, n_total)
    lag = run("lagged", args.substeps, args.windows)
    two = run("two-pass", args.substeps, args.windows)

    # compare at common times (window ends)
    ref_at = ref[args.substeps - 1::args.substeps][:args.windows]
    err_lag = np.linalg.norm(lag - ref_at, axis=1).max()
    err_two = np.linalg.norm(two - ref_at, axis=1).max()
    print(f"\nmax tip deviation vs per-step-coupled reference:")
    print(f"  lagged (ZOH) : {err_lag:.3e}")
    print(f"  two-pass     : {err_two:.3e}")
    print(f"  improvement  : {err_lag / max(err_two, 1e-300):.1f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
