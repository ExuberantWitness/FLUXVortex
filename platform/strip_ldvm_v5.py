"""v5 candidate closure: strip-wise LDVM DELTA channel (2026-07-19, user-approved).

Architecture: the 3D UVLM attached base stays untouched; per spanwise strip run the
S2-faithful LDVM2D (ldvm_fourier.py — Fourier bound layer, pre-LEV rate semantics,
wake term: the published force channel that carries pinned-A0 lift) TWICE — LEV on
(literature crit) and LEV off (crit=99, attached) — and integrate only the DIFFERENCE
dL(t), dT(t). At subcritical strips the delta vanishes by construction; the attached
physics is counted once (3D base), the LEV increment once (2D delta). This supplies
exactly the channel E5/E6 proved missing from v4 (gap_s3_force.md: pinned-rate
apparent-mass + wake-term lift).

Kinematics per strip (legacy _v2_flap_strip decomposition, conventions re-audited):
alpha = aoa + twist(y,t); hdot = y*thetadot UP-positive fed to LDVM2D's +h*ca BC
(S2-corrected convention — the legacy pair fed up-positive into a down-positive BC;
cycle means are inversion-immune, dL/df is not). Lift projected by cos(theta), x2 wings.
Amplitude conventions = production replay (flap half-amp 22.5, twist tip amp = tw/2).

Known modeling notes (recorded): strip theory lacks 3D downwash — the DELTA is more-2D
than the attached part (LEV local); raw delta first, finite-wing correction only if the
gate demands. dt* = 0.015 (S2-verified); ns=8 strips; max_wake cap (far-wake truncation).
Zero-fit: crit in {0.14, 0.18} literature anchors (SD7003 family; section Re~1.5e5)."""
import sys
import time

import numpy as np

sys.path.insert(0, "platform")
from ldvm_fourier import LDVM2D


def chord_at(y, chord=0.287, half_span=0.80):
    r = chord / 2.0
    y_round = half_span - r
    if y <= y_round:
        return chord
    d = y - y_round
    return 2.0 * np.sqrt(max(r * r - d * d, 0.0))


def strip_delta(U=8.0, aoa_deg=5.0, freq=2.3, flap_amp_deg=22.5, twist_amp_deg=11.25,
                twist_phase_deg=90.0, half_span=0.80, chord=0.287, ns=8, n_cycle=3,
                lesp_crit=0.14, max_wake=1500, ndiv=70, naterm=35, rho=1.225,
                dtstar=0.015, verbose=False):
    """Cycle-mean (last cycle) LEV DELTA forces (both wings): dict(dL, dT, per-strip)."""
    Om = 2.0 * np.pi * freq
    A_f = np.radians(flap_amp_deg); A_t = np.radians(twist_amp_deg)
    phi = np.radians(twist_phase_deg); a_b = np.radians(aoa_deg)
    edges = np.linspace(0.0, half_span, ns + 1)
    ys = 0.5 * (edges[:-1] + edges[1:]); widths = np.diff(edges)
    chords = np.array([chord_at(y, chord, half_span) for y in ys])
    keep = chords > 0.02
    ys, widths, chords = ys[keep], widths[keep], chords[keep]
    # per-strip dt from dt* on the STRIP chord, then snapped to a common step count/cycle
    dt = dtstar * np.max(chords) / U
    spc = int(round((1.0 / freq) / dt)); dt = (1.0 / freq) / spc
    mk = lambda crit, c: LDVM2D(U=U, c=c, ndiv=ndiv, naterm=naterm, dt=dt, rho=rho,
                                lesp_crit=crit, camber_m=0.02, camber_p=0.40,
                                max_wake=max_wake)
    on = [mk(lesp_crit, c) for c in chords]
    off = [mk(99.0, c) for c in chords]
    N = n_cycle * spc
    dL_h = np.zeros(spc); dT_h = np.zeros(spc)
    dL_strip = np.zeros(len(ys))
    q = 0.5 * rho * U * U
    t0 = time.time()
    for it in range(N):
        t = it * dt
        th = A_f * np.sin(Om * t); thd = A_f * Om * np.cos(Om * t)
        cth = np.cos(th)
        acc_L = 0.0
        for k, y in enumerate(ys):
            psi = A_t * (y / half_span) * np.sin(Om * t + phi)
            psid = A_t * (y / half_span) * Om * np.cos(Om * t + phi)
            alpha = a_b + psi
            hdot = y * thd                                   # UP-positive (LDVM2D convention)
            r1 = on[k].step(alpha, psid, hdot)
            r0 = off[k].step(alpha, psid, hdot)
            dCl = r1["CLf"] - r0["CLf"]; dCd = r1["CDf"] - r0["CDf"]
            w = q * chords[k] * widths[k]
            if it >= (n_cycle - 1) * spc:
                j = it - (n_cycle - 1) * spc
                dL_h[j] += dCl * cth * w
                dT_h[j] += -dCd * w
                dL_strip[k] += dCl * cth * w / spc
        if verbose and it % max(spc // 4, 1) == 0:
            print(f"  t*={it/spc:.2f}cyc  np_on0={on[len(ys)//2].tx.__len__()}  "
                  f"({time.time()-t0:.0f}s)", flush=True)
    return dict(dL=2.0 * float(np.mean(dL_h)), dT=2.0 * float(np.mean(dT_h)),
                dL_strip=(2.0 * dL_strip).tolist(),
                sec=time.time() - t0)


if __name__ == "__main__":
    # L2-fingerprint gate: d(dL)/df @ aoa10 target +1.5-1.9 N/Hz, alpha-gated (small @aoa0)
    args = sys.argv[1:]
    crit = float(args[0]) if args else 0.14
    CONDS = [(8.0, 0.0, 2.0, 22.5), (8.0, 5.0, 1.4, 22.5), (8.0, 5.0, 2.6, 22.5),
             (8.0, 10.0, 1.4, 22.5), (8.0, 10.0, 2.0, 22.5), (8.0, 10.0, 2.6, 22.5)]
    res = {}
    for U, aoa, f, tw in CONDS:
        r = strip_delta(U=U, aoa_deg=aoa, freq=f, twist_amp_deg=0.5 * tw, lesp_crit=crit)
        res[(aoa, f)] = r
        print(f"[crit={crit}] U{U:g}/aoa{aoa:g}/f{f}/tw{tw:g}: dL={r['dL']:+.3f} "
              f"dT={r['dT']:+.3f}  ({r['sec']:.0f}s)", flush=True)
    s10 = np.polyfit([1.4, 2.0, 2.6], [res[(10.0, f)]["dL"] for f in (1.4, 2.0, 2.6)], 1)[0]
    s5 = (res[(5.0, 2.6)]["dL"] - res[(5.0, 1.4)]["dL"]) / 1.2
    print(f"== d(dL)/df @aoa10 = {s10:+.2f} N/Hz (L2 target +1.5-1.9) | @aoa5 = {s5:+.2f} "
          f"| aoa0 dL = {res[(0.0, 2.0)]['dL']:+.2f} (symmetry ~0)", flush=True)
