"""Goland-wing flutter via the newton_pc two-pass predictor-corrector.

Validates the newton_pc coupler on the classic Goland & Luke (1948) flutter
benchmark, driving the bending-torsion beam (beam_fe) + ring-UVLM with the
two-pass window predictor-corrector (vs the legacy lagged staggered scheme in
tests/benchmark_goland.py). Flutter = freestream speed where the tip-heave
envelope growth rate crosses zero.

Usage: python flap_arena/goland_newtonpc.py [--vmin 110 --vmax 170 --dv 10]
       python flap_arena/goland_newtonpc.py --single 160   # one speed
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
from newton_pc.adapters.beam import (BeamForceSet, BeamUVLMProvider,  # noqa: E402
                                     GolandBeamEntry)

CHORD, SPAN, NC, NS = 1.8288, 6.096, 4, 8
ALPHA = 2.0
BEAM = dict(length=SPAN, EI=9.773e6, GJ=0.988e6, m_per_length=35.72,
            Ip=35.72 * CHORD ** 2 / 24.0, x_ea_cg=0.10 * CHORD,
            structural_damping=0.005)


def envelope_growth(sig, dt):
    a = np.abs(sig)
    pk = [(i * dt, a[i]) for i in range(1, len(a) - 1)
          if a[i] > a[i - 1] and a[i] > a[i + 1]]
    if len(pk) < 3:
        return 0.0
    t = np.array([p[0] for p in pk])
    la = np.log(np.maximum([p[1] for p in pk], 1e-15))
    if len(t) > 4:
        t, la = t[1:], la[1:]
    return float(np.polyfit(t, la, 1)[0]) if len(t) >= 2 else 0.0


def run_at_velocity(V, n_chords=100, substeps=6, rho=1.225):
    entry = GolandBeamEntry(CHORD, SPAN, NC, NS, BEAM, alpha_deg=ALPHA)
    dtw = (CHORD / NC) / V                     # convective window
    provider = BeamUVLMProvider(
        V * np.array([np.cos(np.deg2rad(ALPHA)), 0.0, np.sin(np.deg2rad(ALPHA))]),
        rho, dtw, K=8, chord=CHORD, particles=True, max_particles=10**6,
        pop_scheme="merge", merge_eps=1e-4).bind(entry)
    provider.merge_protect_dist = 3.0
    pc = WindowPredictorCorrector(entry=entry, provider=provider,
                                  substeps=substeps, dt=dtw / substeps,
                                  mode="two-pass")
    pc.initialize(BeamForceSet(np.zeros(entry.beam.ndof)))
    entry.perturb(w_tip=0.05, theta_tip_deg=2.0)
    n_windows = int(round(n_chords * (CHORD / V) / dtw))   # n_chords travel
    tip = entry.beam.nnodes - 1
    tip_w, tip_th = [], []
    t0 = time.time()
    pc.advance(n_substeps=1)
    for w in range(n_windows):
        pc.advance()
        tip_w.append(entry.beam.d[3 * tip])
        tip_th.append(entry.beam.d[3 * tip + 2])
    tip_w = np.array(tip_w)
    sig_w = envelope_growth(tip_w, dtw)
    sig_th = envelope_growth(np.array(tip_th), dtw)
    return dict(V=V, sigma_w=sig_w, sigma_th=sig_th,
                max_w=float(np.max(np.abs(tip_w))), wall=time.time() - t0,
                n_windows=n_windows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vmin", type=int, default=110)
    ap.add_argument("--vmax", type=int, default=170)
    ap.add_argument("--dv", type=int, default=10)
    ap.add_argument("--nchords", type=int, default=100)
    ap.add_argument("--single", type=float, default=None)
    args = ap.parse_args()

    if args.single is not None:
        r = run_at_velocity(args.single, n_chords=args.nchords)
        print(f"V={r['V']:.0f}: sigma_w={r['sigma_w']:+.4f} "
              f"sigma_th={r['sigma_th']:+.4f} max_w={r['max_w']:.4f} "
              f"({r['wall']:.0f}s, {r['n_windows']} win)", flush=True)
        return 0

    vs = list(range(args.vmin, args.vmax + 1, args.dv))
    print(f"=== Goland flutter via newton_pc two-pass PC ===", flush=True)
    print(f"  EI={BEAM['EI']:.2e} GJ={BEAM['GJ']:.2e} m={BEAM['m_per_length']}"
          f" Ip={BEAM['Ip']:.2f} x_ea_cg={BEAM['x_ea_cg']:.3f}", flush=True)
    res = []
    for V in vs:
        r = run_at_velocity(V, n_chords=args.nchords)
        st = "FLUTTER" if r["sigma_w"] > 0 else "stable"
        print(f"  V={V:3d}  sigma_w={r['sigma_w']:+.4f}  "
              f"sigma_th={r['sigma_th']:+.4f}  {st}  ({r['wall']:.0f}s)",
              flush=True)
        res.append(r)
        np.save("flap_arena/out/goland_npc_lastV.npy",
                np.array([V, r["sigma_w"], r["sigma_th"]]))
    # interpolate flutter crossing
    Vf = None
    for i in range(len(res) - 1):
        if res[i]["sigma_w"] < 0 < res[i + 1]["sigma_w"]:
            s0, s1 = res[i]["sigma_w"], res[i + 1]["sigma_w"]
            v0, v1 = res[i]["V"], res[i + 1]["V"]
            Vf = v0 - s0 * (v1 - v0) / (s1 - s0)
            break
    print(f"\n  newton_pc two-pass flutter speed: "
          f"{Vf:.1f} m/s" if Vf else "\n  no crossing in range", flush=True)
    print(f"  Reference (Goland & Luke 1948): ~137 m/s; legacy lagged: 140.2",
          flush=True)
    if Vf:
        print(f"  error vs 137: {abs(Vf-137)/137*100:.1f}%", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
