"""W3/W4 — flapping-wing run: our coupler vs the PteraSoftware baseline.

Levels:
  L1  kinematic rigid wing (validates aero + wake + kinematics chain)
  L2  elastic shell, stiffness scaled by --kscale (quasi-rigid limit check)

Usage:
  python flap_arena/run_flap.py --level L1 [--K 8] [--cycles 3] [--substeps 8]
  python flap_arena/run_flap.py --level L2 --kscale 100
Compares lift history vs flap_arena/out/ptera_baseline.npz (free wake) over
the LAST cycle: correlation + amplitude ratio. Saves out/flap_<tag>.npz.
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
    ap.add_argument("--level", default="L1", choices=["L1", "L2"])
    ap.add_argument("--kscale", type=float, default=100.0)
    ap.add_argument("--hscale", type=float, default=1.0)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--cycles", type=int, default=3)
    ap.add_argument("--substeps", type=int, default=8)
    ap.add_argument("--mode", default="two-pass", choices=["two-pass", "lagged"])
    ap.add_argument("--tag", default=None)
    ap.add_argument("--ampsign", type=float, default=1.0)
    ap.add_argument("--ampzero", action="store_true")
    ap.add_argument("--baseline", default="flap_arena/out/ptera_baseline.npz")
    args = ap.parse_args()

    base = np.load(args.baseline)
    chord, span = float(base["chord"]), float(base["span"])
    nc, ns = int(base["nc"]), int(base["ns"])
    amp = np.deg2rad(float(base["amp_deg"])) * args.ampsign
    if args.ampzero:
        amp = 0.0
    period = float(base["period"])
    V, alpha = float(base["v_inf"]), np.deg2rad(float(base["alpha"]))
    rho = float(base["rho"])
    dtw = float(base["dt_free"])          # window = Ptera step (same cadence)
    n_windows = int(round(args.cycles * period / dtw))

    kin = FlapKinematics(amp, period)
    entry = FlapEntry(chord, span, nc, ns, kin,
                      mode=("kinematic" if args.level == "L1" else "elastic"),
                      kscale=args.kscale, hscale=args.hscale)
    V_vec = V * np.array([np.cos(alpha), 0.0, np.sin(alpha)])
    provider = FlapUVLMProvider(V_vec, rho, dtw, K=args.K)
    ndof = entry.shell.ndof
    zero = NodalForceSet(np.zeros(ndof))

    pc = WindowPredictorCorrector(entry=entry, provider=provider,
                                  substeps=args.substeps, dt=dtw / args.substeps,
                                  mode=args.mode)
    pc.initialize(zero)
    tag = args.tag or f"{args.level}_K{args.K}_{args.mode}"

    lift = []
    t0 = time.time()
    pc.advance(n_substeps=1)
    for w in range(n_windows):
        pc.advance()
        F = pc._F_cur.payload["f_panel"].sum(axis=(0, 1))
        # lift = force component perpendicular to freestream in x-z plane
        L = -F[0] * np.sin(alpha) + F[2] * np.cos(alpha)
        lift.append(L)
        if w % 20 == 0 or w == n_windows - 1:
            npart = len(provider.p_pos)
            print(f"[{tag}] w={w:4d} t={pc._t:7.3f}s L={L:+9.3f} N "
                  f"rings={len(provider.wake_v)} parts={npart} "
                  f"({time.time() - t0:.0f}s)", flush=True)
    lift = np.array(lift)
    np.savez(f"flap_arena/out/flap_{tag}.npz", lift=lift, dtw=dtw,
             n_windows=n_windows)

    # ---- compare last cycle vs Ptera free-wake ----
    Cf = base["C_free"]
    lift_p = -base["F_free"][:, 2]        # Ptera lift (N), wind axes
    per_win = int(round(period / dtw))
    ours = lift[-per_win:]
    n_avail = min(len(lift_p), per_win)
    theirs = lift_p[-n_avail:]
    ours_c = ours[-n_avail:]
    corr = np.corrcoef(ours_c, theirs)[0, 1]
    amp_ratio = (ours_c.max() - ours_c.min()) / (theirs.max() - theirs.min())
    mean_ratio = ours_c.mean() / theirs.mean() if abs(theirs.mean()) > 1e-9 else np.nan
    print(f"\n[{tag}] last-cycle vs Ptera(free wake): corr={corr:.4f} "
          f"amp_ratio={amp_ratio:.3f} mean_ratio={mean_ratio:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
