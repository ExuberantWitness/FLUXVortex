"""Multi-metric comparison of flapping lift histories vs PteraSoftware.

Metrics (last cycle, fixed time-convention offset — NOT fitted per case):
  NRMSE       : RMS(ours-ref)/RMS(ref)            (waveform error norm)
  rel MAE     : mean|ours-ref| / mean|ref|
  peak errors : relative error at max / min lift  (extreme fidelity)
  mean error  : cycle-averaged lift (net lift) relative error
  amp error   : (peak-to-peak ratio) - 1
  phase error : fundamental-harmonic phase difference (deg)
  harmonics   : amplitude ratio of 1st/2nd/3rd FFT harmonics

Usage: python flap_arena/compare_metrics.py <flap_npz> <baseline_npz> [offset_dt]
"""
from __future__ import annotations

import sys

import numpy as np

SUBSTEPS = 8
OFFSET_DT = 2.125     # determined ONCE from the 5-deg K=130 alignment scan


def harmonics(x: np.ndarray, n: int = 3):
    X = np.fft.rfft(x - x.mean())
    amp = np.abs(X) * 2 / len(x)
    ph = np.angle(X)
    return amp[1:n + 1], ph[1:n + 1]


def main() -> int:
    ours_npz = np.load(sys.argv[1])
    base = np.load(sys.argv[2])
    off = float(sys.argv[3]) if len(sys.argv) > 3 else OFFSET_DT
    lift = ours_npz["lift"]
    dtw = float(ours_npz["dtw"])
    t_ours = dtw / SUBSTEPS + (np.arange(len(lift)) + 1) * dtw

    Lp = -base["F_free"][:, 2]
    ok = np.isfinite(Lp)
    t_p = np.arange(len(Lp))[ok] * dtw
    Lp = Lp[ok]

    per = int(round(1.0 / dtw))
    sel = t_ours >= t_ours[-1] - per * dtw + 1e-9
    to, o = t_ours[sel], lift[sel]
    th = np.interp(to + off * dtw, t_p, Lp)

    err = o - th
    nrmse = np.sqrt(np.mean(err ** 2)) / np.sqrt(np.mean(th ** 2))
    relmae = np.mean(np.abs(err)) / np.mean(np.abs(th))
    pk_hi = (o.max() - th.max()) / abs(th.max())
    pk_lo = (o.min() - th.min()) / abs(th.min())
    mean_err = (o.mean() - th.mean()) / (abs(th).max())
    amp_err = (o.max() - o.min()) / (th.max() - th.min()) - 1.0

    a_o, p_o = harmonics(o)
    a_t, p_t = harmonics(th)
    dphase = np.rad2deg((p_o[0] - p_t[0] + np.pi) % (2 * np.pi) - np.pi)

    print(f"case: {sys.argv[1].split('/')[-1]}  (offset fixed {off}dt)")
    print(f"  NRMSE          : {nrmse * 100:6.2f} %")
    print(f"  rel MAE        : {relmae * 100:6.2f} %")
    print(f"  amp (p2p) err  : {amp_err * 100:+6.2f} %")
    print(f"  peak-high err  : {pk_hi * 100:+6.2f} %")
    print(f"  peak-low  err  : {pk_lo * 100:+6.2f} %")
    print(f"  cycle-mean err : {mean_err * 100:+6.2f} %  (of |L|max)")
    print(f"  phase err (H1) : {dphase:+6.2f} deg")
    print(f"  harmonic amp ratios H1/H2/H3: "
          + " ".join(f"{a / b:5.3f}" if b > 1e-9 else "n/a"
                     for a, b in zip(a_o, a_t)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
