"""[工具箱备用 2026-07-19:非生产路径,主线 = v4 + 记分卡 GAP 闭环]
v5 strip-LDVM DELTA channel on the GPU batch engine (ldvm_gpu.LDVMBatch).
Same math/conventions as strip_ldvm_v5.py (CPU reference), lanes = strips x {on, off}.
Converged-gate settings now affordable: ns=16, n_cycle=5, no wake cap, dt* refinement.
~5-10 s/condition (vs 16-68 min CPU)."""
import sys
import time

import numpy as np
import warp as wp

sys.path.insert(0, "platform")
from ldvm_gpu import LDVMBatch
from strip_ldvm_v5 import chord_at


def strip_delta_gpu(U=8.0, aoa_deg=5.0, freq=2.3, flap_amp_deg=22.5, twist_amp_deg=11.25,
                    twist_phase_deg=90.0, half_span=0.80, chord=0.287, ns=16, n_cycle=5,
                    lesp_crit=0.14, ndiv=70, naterm=35, rho=1.225, dtstar=0.015):
    Om = 2.0 * np.pi * freq
    A_f = np.radians(flap_amp_deg); A_t = np.radians(twist_amp_deg)
    phi = np.radians(twist_phase_deg); a_b = np.radians(aoa_deg)
    edges = np.linspace(0.0, half_span, ns + 1)
    ys = 0.5 * (edges[:-1] + edges[1:]); widths = np.diff(edges)
    chords = np.array([chord_at(y, chord, half_span) for y in ys])
    keep = chords > 0.02
    ys, widths, chords = ys[keep], widths[keep], chords[keep]
    nsr = len(ys)
    dt = dtstar * np.max(chords) / U
    spc = int(round((1.0 / freq) / dt)); dt = (1.0 / freq) / spc
    # lanes: [on strips..., off strips...]
    lane_c = np.concatenate([chords, chords])
    lane_crit = np.concatenate([np.full(nsr, lesp_crit), np.full(nsr, 99.0)])
    wmax = 2 * n_cycle * spc + 16
    eng = LDVMBatch(chords=lane_c, lesp_crit=lane_crit, U=U, dt=dt, ndiv=ndiv,
                    naterm=naterm, rho=rho, camber_m=0.02, camber_p=0.40, wmax=wmax)
    q = 0.5 * rho * U * U
    N = n_cycle * spc
    dLh = np.zeros(spc); dTh = np.zeros(spc)
    t0 = time.time()
    for it in range(N):
        t = it * dt
        th = A_f * np.sin(Om * t); thd = A_f * Om * np.cos(Om * t)
        cth = np.cos(th)
        psi = A_t * (ys / half_span) * np.sin(Om * t + phi)
        psid = A_t * (ys / half_span) * Om * np.cos(Om * t + phi)
        alpha = a_b + psi; hdot = ys * thd
        r = eng.step(np.concatenate([alpha, alpha]),
                     np.concatenate([psid, psid]),
                     np.concatenate([hdot, hdot]))
        if it >= (n_cycle - 1) * spc:
            j = it - (n_cycle - 1) * spc
            dCl = r["CLf"][:nsr] - r["CLf"][nsr:]
            dCd = r["CDf"][:nsr] - r["CDf"][nsr:]
            w = q * chords * widths
            dLh[j] = np.sum(dCl * cth * w)
            dTh[j] = np.sum(-dCd * w)
    return dict(dL=2.0 * float(np.mean(dLh)), dT=2.0 * float(np.mean(dTh)),
                sec=time.time() - t0, spc=spc)


if __name__ == "__main__":
    wp.init()
    crit = float(sys.argv[1]) if len(sys.argv) > 1 else 0.14
    dts = float(sys.argv[2]) if len(sys.argv) > 2 else 0.015
    CONDS = [(8.0, 0.0, 2.0, 22.5), (8.0, 5.0, 1.4, 22.5), (8.0, 5.0, 2.6, 22.5),
             (8.0, 10.0, 1.4, 22.5), (8.0, 10.0, 2.0, 22.5), (8.0, 10.0, 2.6, 22.5)]
    res = {}
    for U, aoa, f, tw in CONDS:
        r = strip_delta_gpu(U=U, aoa_deg=aoa, freq=f, twist_amp_deg=0.5 * tw,
                            lesp_crit=crit, dtstar=dts)
        res[(aoa, f)] = r
        print(f"[gpu crit={crit} dt*={dts}] aoa{aoa:g}/f{f}: dL={r['dL']:+.3f} "
              f"dT={r['dT']:+.3f}  ({r['sec']:.0f}s)", flush=True)
    s10 = np.polyfit([1.4, 2.0, 2.6], [res[(10.0, f)]["dL"] for f in (1.4, 2.0, 2.6)], 1)[0]
    s5 = (res[(5.0, 2.6)]["dL"] - res[(5.0, 1.4)]["dL"]) / 1.2
    print(f"== d(dL)/df @aoa10 = {s10:+.2f} N/Hz (L2 target +1.5-1.9) | @aoa5 = {s5:+.2f} "
          f"| aoa0 dL = {res[(0.0, 2.0)]['dL']:+.2f}", flush=True)
