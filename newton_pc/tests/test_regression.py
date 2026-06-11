"""P1 red line: the generic coupler must reproduce the validated chain.

Drives the FLUXV adapters through WindowPredictorCorrector for 5 windows
(boot + 4 full) and compares every boundary against the MATLAB corrected
trajectory (fixture step4 h_X_vec): tip ratio must print 1.000000 and the
full-state error must stay within the validated chain's envelope (<=2e-5).

Run:  python -m newton_pc.tests.test_regression
"""
from __future__ import annotations

import os
import sys

import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _ROOT)
os.chdir(_ROOT)

from newton_pc import WindowPredictorCorrector  # noqa: E402
from newton_pc.adapters.fluxv import make_fluxv_pair  # noqa: E402

ZDOF = 9 * 175 + 2
N_WINDOWS_FULL = 4          # boot(1 substep) + 4 x 34
STATE_TOL = 2.0e-5
RATIO_TOL = 5.0e-6


def main() -> int:
    entry, provider, zero, P = make_fluxv_pair(mscale=1.0)
    from scipy.io import loadmat
    f4 = loadmat("FSI_by_FEM_and_UVLM/single_sheet/fixtures_traj/"
                 "fixture_step4_t0.4000.mat",
                 squeeze_me=True, struct_as_record=False)
    hX = np.asarray(f4["h_X_vec"])  # corrected cols for earlier blocks

    pc = WindowPredictorCorrector(entry=entry, provider=provider,
                                  substeps=34, dt=P.d_t)
    pc.initialize(zero)

    ok = True
    boundaries = [1] + [1 + 34 * (k + 1) for k in range(N_WINDOWS_FULL)]
    pc.advance(n_substeps=1)  # boot window
    for b in boundaries[1:]:
        pc.advance()
        X = entry.state()
        truth = hX[:, b]  # MATLAB state at i_time=b+1 (0-based col b)
        err = np.abs(X - truth).max()
        tip, tip_ml = X[ZDOF], truth[ZDOF]
        ratio = tip / tip_ml if abs(tip_ml) > 1e-12 else float("nan")
        line_ok = err <= STATE_TOL and abs(ratio - 1.0) <= RATIO_TOL
        ok &= line_ok
        print(f"  b={b:4d} t*={b * P.d_t:.3f}  max|X-ml|={err:.3e}  "
              f"tip={tip:+.6e} ml={tip_ml:+.6e} ratio={ratio:.6f}  "
              f"{'PASS' if line_ok else 'FAIL'}", flush=True)
    print(f"RED LINE: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
