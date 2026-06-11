"""Time-aligned comparison: interpolate Ptera lift onto our window-end times.

Our lift[w] timestamp: t = dtw/substeps (boot) + (w+1)*dtw.
Ptera force[k] timestamp: t = k*dt (state at step k; first valid index varies).
Reports corr/amp on the last cycle with exact time alignment, plus a small
fractional-offset scan to expose any residual convention mismatch.

Usage: python flap_arena/compare_aligned.py flap_arena/out/flap_<tag>.npz \
           flap_arena/out/ptera_baseline5.npz [substeps]
"""
from __future__ import annotations

import sys

import numpy as np


def main() -> int:
    ours_npz = np.load(sys.argv[1])
    base = np.load(sys.argv[2])
    substeps = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    lift = ours_npz["lift"]
    dtw = float(ours_npz["dtw"])
    t_ours = dtw / substeps + (np.arange(len(lift)) + 1) * dtw

    Lp = -base["F_free"][:, 2]
    valid = np.isfinite(Lp)
    kk = np.arange(len(Lp))[valid]
    t_p = kk * dtw
    Lp = Lp[valid]

    per = int(round(1.0 / dtw))
    sel = t_ours >= t_ours[-1] - per * dtw + 1e-9   # last cycle of ours
    to = t_ours[sel]
    o = lift[sel]

    print(f"{'offset(dt frac)':>16} {'corr':>8} {'amp_ratio':>10} {'mean_ratio':>10}")
    best = None
    for frac in np.arange(-1.5, 1.51, 0.25):
        tt = to + frac * dtw
        th = np.interp(tt, t_p, Lp)
        c = np.corrcoef(o, th)[0, 1]
        a = (o.max() - o.min()) / (th.max() - th.min())
        m = o.mean() / th.mean() if abs(th.mean()) > 1e-9 else np.nan
        tagb = ""
        if best is None or c > best[1]:
            best = (frac, c, a, m)
            tagb = "  <-"
        print(f"{frac:16.2f} {c:8.4f} {a:10.3f} {m:10.3f}{tagb}")
    print(f"\nBEST: offset={best[0]:+.2f}dt  corr={best[1]:.4f}  "
          f"amp={best[2]:.3f}  mean={best[3]:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
