"""Long-horizon stability run: generic coupler vs MATLAB ground truth.

Usage: python newton_pc/examples/fluxv_longrun.py --tstar 3 [--mode lagged]
Validates per-checkpoint against fixtures_traj_long (t*<=3) or
fixtures_traj_xlong (t*<=6) h_X_vec, reports tip ratio + bounded-error +
NaN guard. ``--mode lagged`` runs the zero-order-hold baseline (the #2848
substep semantics) in the same arena — the stability contrast is the core
evidence for the predictor-corrector contribution.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _ROOT)
os.chdir(_ROOT)

from newton_pc import WindowPredictorCorrector  # noqa: E402
from newton_pc.adapters.fluxv import make_fluxv_pair  # noqa: E402

ZDOF = 9 * 175 + 2


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tstar", type=float, default=3.0)
    ap.add_argument("--mode", default="two-pass", choices=["two-pass", "lagged"])
    ap.add_argument("--interp", default="linear", choices=["linear", "quad"])
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    entry, provider, zero, P = make_fluxv_pair(mscale=1.0)
    truth_path = ("FSI_by_FEM_and_UVLM/single_sheet/fixtures_traj_xlong/"
                  "fixture_step5_t6.0000.mat" if args.tstar > 3.01 else
                  "FSI_by_FEM_and_UVLM/single_sheet/fixtures_traj_long/"
                  "fixture_step5_t3.0000.mat")
    from scipy.io import loadmat
    hX = np.asarray(loadmat(truth_path, squeeze_me=True,
                            struct_as_record=False)["h_X_vec"])

    pc = WindowPredictorCorrector(entry=entry, provider=provider,
                                  substeps=34, dt=P.d_t,
                                  mode=args.mode, interp=args.interp)
    pc.initialize(zero)
    n_windows = int(round(args.tstar / (34 * P.d_t)))
    tag = args.tag or f"{args.mode}-{args.interp}"

    t0 = time.time()
    pc.advance(n_substeps=1)
    b = 1
    worst_amp = 0.0
    amp_scale = 1e-3  # running amplitude floor for relative metrics
    for w in range(n_windows):
        pc.advance()
        b += 34
        X = entry.state()
        if not np.all(np.isfinite(X)):
            print(f"[{tag}] b={b} NaN/Inf -> UNSTABLE", flush=True)
            return 1
        tip = X[ZDOF]
        amp_scale = max(amp_scale, abs(tip))
        if b < hX.shape[1]:
            ml = hX[:, b]
            tip_ml = ml[ZDOF]
            ratio = tip / tip_ml if abs(tip_ml) > 1e-12 else float("nan")
            amp_err = abs(tip - tip_ml) / amp_scale
            worst_amp = max(worst_amp, amp_err)
            if w % 4 == 0 or w == n_windows - 1:
                print(f"[{tag}] b={b:5d} t*={b * P.d_t:6.3f} "
                      f"tip={tip:+.6e} ml={tip_ml:+.6e} ratio={ratio:8.6f} "
                      f"amp_err={amp_err:.2e} ({time.time() - t0:.0f}s)",
                      flush=True)
        else:
            if w % 4 == 0:
                print(f"[{tag}] b={b:5d} t*={b * P.d_t:6.3f} tip={tip:+.6e} "
                      f"(beyond truth) ({time.time() - t0:.0f}s)", flush=True)
        np.save(f"/tmp/npc_{tag}_b{b}.npy", X)
    print(f"[{tag}] DONE worst_amp_err={worst_amp:.3e} "
          f"solves={provider.n_solves} wall={time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
