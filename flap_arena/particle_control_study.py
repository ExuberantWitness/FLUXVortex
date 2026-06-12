"""Full-scale particle population control study (100+ flapping cycles).

Three-way: A unbounded reference | B drop-oldest (status quo) | C pairwise
moment-conserving merging with at-wing error threshold (per IDEA_REPORT_EFF).

Per-window records: particle count, wall ms, lift, circulation ledger.
Usage:
  python flap_arena/particle_control_study.py --scheme none  --cycles 100 --tag A
  python flap_arena/particle_control_study.py --scheme drop  --cycles 100 --tag B
  python flap_arena/particle_control_study.py --scheme merge --eps 1e-3 --cycles 100 --tag C3
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _ROOT)
os.chdir(_ROOT)

from newton_pc import WindowPredictorCorrector  # noqa: E402
from newton_pc.adapters.flap import (FlapEntry, FlapKinematics,  # noqa: E402
                                     FlapUVLMProvider, NodalForceSet)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scheme", required=True, choices=["none", "drop", "merge"])
    ap.add_argument("--eps", type=float, default=1e-3)
    ap.add_argument("--cycles", type=int, default=100)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--cap", type=int, default=20000)
    ap.add_argument("--protect_dist", type=float, default=0.0)
    args = ap.parse_args()

    base = np.load("flap_arena/out/ptera_baseline5.npz")
    chord, span = float(base["chord"]), float(base["span"])
    nc, ns = int(base["nc"]), int(base["ns"])
    amp = np.deg2rad(float(base["amp_deg"]))
    period = float(base["period"])
    V, alpha = float(base["v_inf"]), np.deg2rad(float(base["alpha"]))
    rho = float(base["rho"])
    dtw = float(base["dt_free"])
    n_windows = int(round(args.cycles * period / dtw))

    kin = FlapKinematics(amp, period)
    entry = FlapEntry(chord, span, nc, ns, kin, mode="kinematic")
    V_vec = V * np.array([np.cos(alpha), 0.0, np.sin(alpha)])
    provider = FlapUVLMProvider(
        V_vec, rho, dtw, K=args.K, chord=chord, particles=True,
        max_particles=(10**6 if args.scheme != "drop" else args.cap),
        pop_scheme=args.scheme, merge_eps=args.eps)
    if args.protect_dist > 0:
        provider.merge_protect_dist = args.protect_dist
    pc = WindowPredictorCorrector(entry=entry, provider=provider,
                                  substeps=8, dt=dtw / 8, mode="two-pass")
    pc.initialize(NodalForceSet(np.zeros(entry.shell.ndof)))

    lift = np.empty(n_windows)
    npart = np.empty(n_windows, np.int64)
    wall = np.empty(n_windows)
    ledger = np.empty((n_windows, 3))
    t0 = time.time()
    pc.advance(n_substeps=1)
    for w in range(n_windows):
        tw = time.time()
        pc.advance()
        F = pc._F_cur.payload["f_panel"].sum(axis=(0, 1))
        lift[w] = -F[0] * np.sin(alpha) + F[2] * np.cos(alpha)
        npart[w] = len(provider.p_pos)
        wall[w] = time.time() - tw
        ledger[w] = provider.circulation_ledger()
        if not np.isfinite(lift[w]):
            raise FloatingPointError(f"diverged at window {w}")
        if w % 200 == 0 or w == n_windows - 1:
            print(f"[{args.tag}] w={w:5d} t={pc._t:7.2f}s n_p={npart[w]:7d} "
                  f"L={lift[w]:+9.2f}N {wall[w]*1000:6.0f}ms/win "
                  f"merged={provider.stats['n_merged']:8d} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    np.savez(f"flap_arena/out/pcs_{args.tag}.npz", lift=lift, npart=npart,
             wall=wall, ledger=ledger, dtw=dtw, scheme=args.scheme,
             eps=args.eps, n_merged=provider.stats["n_merged"])
    print(f"[{args.tag}] DONE n_p(end)={npart[-1]} wall={time.time()-t0:.0f}s "
          f"mean_ms/win={wall.mean()*1000:.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
