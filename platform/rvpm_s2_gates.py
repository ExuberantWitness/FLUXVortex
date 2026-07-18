"""rVPM S2 gates — 2D strip testbed (PROJECT_rvpm S2; case definitions verified from
K. Ramesh NCSU thesis 2013 (Ch.4 == JFM 751:500-538) Table 4.1 + Appendix C + UNSflow source,
see docs/diag/research_rvpm_s2cases.md).

Testbed = platform/flap_ldvm.py in S2 mode: shed_rule='ramesh' (shed every supercritical step,
A0 pinned at +-crit), placement='third' (Ansari 1/3 rule; first particle at 0.5*v_edge*dt),
core='vatistas2' with r_c = 0.02c HARDCODED (UNSflow keeps 0.02c even at refined dt),
circulation frozen after the shed-step solve. Force = Ramesh Fourier channel CLf/CDf
(thesis eqs 4.26-4.32: A0..A2 + rates + nonlinear wake term). 2D has no stretching, so the
rVPM transport upgrade (S1) is inert here — S2 validates FEEDING rules + force channel.

G3a — Eldredge family, all pitch about the LE, dt* per UNSflow find_tstep:
  A_flat45 (=thesis Case 5B): flat plate Re=1e3, up 0->45deg, K=0.4, a=11, hold to t*=9,
      LESP_crit=0.11, dt*=0.0075. Targets: LOM CL spike ~6 @t*~1.5 (CFD ~5.5), drop to ~2
      at ramp end, von-Karman hold oscillation ~1.5-3.
  B_sd25 (=thesis Case 1): SD7003 Re=3e4, ramp-hold-return 25deg, K=0.11, a=11,
      LESP_crit=0.18, dt*=0.015. Targets: peak 2.7-2.8 @t*2.5-3.0, return-start dip to
      1.0-1.2 @t*~4, undershoot ~-0.5 @t*~6, settle ~0.1.
  C_flat90 (=thesis Case 5A): flat plate Re=1e3, up 0->90deg, K=0.2, a=11, to t*=5,
      LESP_crit=0.11, dt*=0.015. Targets: first peak ~2.4 @t*~1.2, main ~4.0 @t*~2.5,
      ~1.5 @t*=5 (LOM ~= couplevpm CFD).
G3b — SD7003 plunge, Re=6e4 (A3-C6, Visbal-family case): omega*c/U=0.5, h0/c=0.5,
      alpha0=8deg, LESP_crit=0.14 (SD7003 family value @1e5; 0.18@3e4 — Re=6e4 sits between,
      0.18 as sensitivity). Targets: CL peak 2.3-2.4 @alpha_eff~21deg (psi~100-120deg),
      CD peak ~0.35, CL_min ~ -0.15.

Zero-fit: all constants literature-anchored. SD7003 camber approximated by the NACA 4-digit
camber machinery (m=1.46%, p=0.353) — recorded assumption (~<0.1 CL zero-lift offset).
"""
import sys
import time

import numpy as np

sys.path.insert(0, "platform")
from ldvm_fourier import LDVM2D

OUT_PNG = "platform/docs/diag/rvpm_s2_gates.png"


# ---------------------------------------------------------------- Eldredge kinematics
def eldredge_up(t, amax, K, a_s, t1):
    """Pitch-up(-hold) (UNSflow EldUpDef, kinem.jl L18-29): alpha ends at amax and holds."""
    t2 = t1 + amax / (2.0 * K)
    al = (K / a_s) * (np.log(np.cosh(a_s * (t - t1))) - np.log(np.cosh(a_s * (t - t2)))) \
        + 0.5 * amax
    dal = K * (np.tanh(a_s * (t - t1)) - np.tanh(a_s * (t - t2)))
    return al, dal


def eldredge_rhr(t, amax, K, a_s, t1):
    """Ramp-hold-return (thesis Appendix C / UNSflow EldRampReturnDef):
    t2=t1+A/2K, t3=t2+piA/4K-A/2K, t4=t3+A/2K; alpha = A*G/maxG."""
    t2 = t1 + amax / (2.0 * K)
    t3 = t2 + np.pi * amax / (4.0 * K) - amax / (2.0 * K)
    t4 = t3 + amax / (2.0 * K)

    def G(tt):
        return (np.log(np.cosh(a_s * (tt - t1))) + np.log(np.cosh(a_s * (tt - t4)))
                - np.log(np.cosh(a_s * (tt - t2))) - np.log(np.cosh(a_s * (tt - t3))))

    gmax = np.max(G(np.linspace(t1 - 1.0, t4 + 1.0, 8000)))
    al = amax * G(t) / gmax
    dal = amax * a_s * (np.tanh(a_s * (t - t1)) + np.tanh(a_s * (t - t4))
                        - np.tanh(a_s * (t - t2)) - np.tanh(a_s * (t - t3))) / gmax
    return al, dal


ELDREDGE_CASES = {
    "A_flat45": dict(kind="up", amax=np.radians(45.0), K=0.4, a_s=11.0, t1=1.0,
                     lesp_crit=0.11, dtstar=0.0075, tmax=9.0, camber=(0.0, 0.4),
                     targets=[(6.0, "g"), (2.0, "b")]),
    "B_sd25": dict(kind="rhr", amax=np.radians(25.0), K=0.11, a_s=11.0, t1=1.0,
                   lesp_crit=0.18, dtstar=0.015, tmax=8.0, camber=(0.0146, 0.353),
                   targets=[(2.75, "g"), (-0.5, "r")]),
    "C_flat90": dict(kind="up", amax=np.radians(90.0), K=0.2, a_s=11.0, t1=1.0,
                     lesp_crit=0.11, dtstar=0.015, tmax=5.0, camber=(0.0, 0.4),
                     targets=[(4.0, "g"), (2.4, "b"), (1.5, "r")]),
}


def run_eldredge(name, case, ndiv=70, naterm=35, max_wake=100000, dt=None):
    dt = dt or case["dtstar"]
    cm, cp = case.get("camber", (0.0, 0.4))
    m = LDVM2D(U=1.0, c=1.0, ndiv=ndiv, naterm=naterm, dt=dt, rho=1.0,
               lesp_crit=case["lesp_crit"], max_wake=max_wake, pivot_xc=0.0,
               camber_m=cm, camber_p=cp)
    ts = np.arange(0.0, case["tmax"], dt)
    fkin = eldredge_up if case["kind"] == "up" else eldredge_rhr
    al, dal = fkin(ts, case["amax"], case["K"], case["a_s"], case["t1"])
    rec = []
    t0 = time.time()
    for i, t in enumerate(ts):
        r = m.step(float(al[i]), float(dal[i]), 0.0)
        rec.append((t, np.degrees(al[i]), r["CLf"], r["CDf"], r["gamb"], r["lesp"],
                    r["n_lev"], r["n_tev"]))
    rec = np.array(rec)
    print(f"[{name}] {len(ts)} steps  {time.time()-t0:.0f}s  "
          f"LEV={int(rec[-1,6])} TEV={int(rec[-1,7])}", flush=True)
    return rec


def eldredge_gate(name, rec):
    t, al, clf = rec[:, 0], rec[:, 1], rec[:, 2]
    kpk = np.argmax(clf)
    print(f"== G3a {name} ==  CLf peak {clf[kpk]:+.2f} @ t*={t[kpk]:.2f}", flush=True)
    if name == "A_flat45":
        hold = clf[t > 3.5]
        print(f"  ramp-end value {clf[np.argmin(np.abs(t-3.0)):][0]:+.2f};"
              f" hold range [{hold.min():+.2f},{hold.max():+.2f}] mean {hold.mean():+.2f}"
              f"   targets: spike~6@1.5, drop~2, hold osc 1.5-3", flush=True)
    if name == "B_sd25":
        i4 = np.argmin(np.abs(t - 4.2)); i6 = np.argmin(np.abs(t - 6.0))
        print(f"  @t*~4.2 {clf[i4]:+.2f} (dip target 1.0-1.2);"
              f" min after t*=5 {clf[t > 5.0].min():+.2f} (target ~-0.5);"
              f" end {clf[-1]:+.2f} (target ~0.1)   peak target 2.7-2.8 @2.5-3.0", flush=True)
    if name == "C_flat90":
        i12 = np.argmin(np.abs(t - 1.2))
        print(f"  @t*~1.2 {clf[i12]:+.2f} (target ~2.4); end {clf[-1]:+.2f} (target ~1.5)"
              f"   main peak target ~4.0 @2.5", flush=True)


# ---------------------------------------------------------------- G3b SD7003 plunge
def run_plunge(dtstar=0.015, ndiv=70, naterm=35, ncyc=4, max_wake=100000,
               omstar=0.5, h0=0.5, alpha0_deg=8.0, lesp_crit=0.14):
    """h(t) = h0*cos(om t) (start at top; downstroke first half-cycle);
    hdot = -h0*om*sin(om t);  alpha_eff = alpha0 + atan(-hdot/U) peaks 22deg at om*t=90deg."""
    dt = dtstar
    m = LDVM2D(U=1.0, c=1.0, ndiv=ndiv, naterm=naterm, dt=dt, rho=1.0,
               lesp_crit=lesp_crit, max_wake=max_wake,
               camber_m=0.0146, camber_p=0.353)       # SD7003 camber via NACA machinery (approx)
    a0 = np.radians(alpha0_deg)
    Tstar = 2.0 * np.pi / omstar
    steps = int(round(ncyc * Tstar / dt))
    rec = []
    t0 = time.time()
    for i in range(steps):
        t = i * dt
        hdot = -h0 * omstar * np.sin(omstar * t)
        r = m.step(a0, 0.0, hdot)
        aeff = alpha0_deg + np.degrees(np.arctan(-hdot / 1.0))
        phase = np.degrees(omstar * t) % 360.0
        rec.append((t, phase, aeff, r["CLf"], r["CDf"], r["gamb"], r["lesp"],
                    r["n_lev"], r["n_tev"]))
        if i % 400 == 0:
            print(f"  plunge step {i}/{steps}  ({time.time()-t0:.0f}s)", flush=True)
    rec = np.array(rec)
    print(f"[SD7003 plunge] {steps} steps  {time.time()-t0:.0f}s  "
          f"LEV={int(rec[-1,7])} TEV={int(rec[-1,8])}", flush=True)
    return rec


def plunge_gate(rec, ncyc=4):
    """Phase-average the last 2 cycles; report CL peak/phase, CD peak, CL_min vs targets."""
    Tstar = 2.0 * np.pi / 0.5
    t = rec[:, 0]
    mask = t >= (ncyc - 2) * Tstar
    ph = rec[mask, 1]; aeff = rec[mask, 2]; cl = rec[mask, 3]; cd = rec[mask, 4]
    bins = np.linspace(0.0, 360.0, 73)
    idx = np.digitize(ph, bins) - 1
    clb = np.array([np.mean(cl[idx == k]) if np.any(idx == k) else np.nan for k in range(72)])
    cdb = np.array([np.mean(cd[idx == k]) if np.any(idx == k) else np.nan for k in range(72)])
    aeb = np.array([np.mean(aeff[idx == k]) if np.any(idx == k) else np.nan for k in range(72)])
    pc = 0.5 * (bins[:-1] + bins[1:])
    kpk = np.nanargmax(clb)
    print("== G3b SD7003 plunge (phase-avg last 2 cycles, CLf channel) ==", flush=True)
    print(f"  CL peak {np.nanmax(clb):+.2f} @ psi={pc[kpk]:.0f}deg (a_eff there {aeb[kpk]:.1f}deg)"
          f"   target 2.3-2.4 @ psi 100-120 (a_eff~21)", flush=True)
    print(f"  CL min  {np.nanmin(clb):+.2f}   target ~ -0.15", flush=True)
    print(f"  CD peak {np.nanmax(cdb):+.2f}   target ~ 0.35", flush=True)
    return pc, clb, cdb, aeb


def plot_all():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import os
    fig, axs = plt.subplots(2, 2, figsize=(11, 8))
    ax = axs[0, 0]
    try:
        rec = np.load("platform/docs/diag/rvpm_s2_plunge.npy")
        pc, clb, cdb, aeb = plunge_gate(rec)
        ax.plot(pc, clb, "b-", lw=1.5, label="CLf (crit=0.14)")
        ax.plot(pc, cdb, "r-", lw=1.0, label="CDf")
        try:
            r18 = np.load("platform/docs/diag/rvpm_s2_plunge_crit018.npy")
            _, c18, d18, _ = plunge_gate(r18, )
            ax.plot(pc, c18, "b--", lw=1.0, label="CLf (crit=0.18)")
            ax.plot(pc, d18, "r--", lw=0.7)
        except FileNotFoundError:
            pass
        ax.axhspan(2.3, 2.4, color="g", alpha=0.2, label="CL peak target")
        ax.axvspan(100, 120, color="g", alpha=0.1)
        ax.axhline(-0.15, color="k", ls=":", lw=0.8)
        ax.axhline(0.35, color="r", ls=":", lw=0.8)
        ax.set_xlabel("phase psi (deg)"); ax.set_title("G3b SD7003 plunge  om*c/U=0.5 h0/c=0.5 a0=8deg")
        ax.legend(fontsize=7)
    except FileNotFoundError:
        ax.set_title("G3b plunge: no data")
    for k, nm in enumerate(list(ELDREDGE_CASES.keys())[:3]):
        ax = axs.flat[k + 1]
        f = f"platform/docs/diag/rvpm_s2_eld_{nm}.npy"
        if not os.path.exists(f):
            ax.set_title(f"G3a {nm}: no data"); continue
        rec = np.load(f)
        ax.plot(rec[:, 0], rec[:, 2], "b-", lw=1.5, label="CLf")
        ax.plot(rec[:, 0], rec[:, 3], "r-", lw=0.8, label="CDf")
        ax.plot(rec[:, 0], rec[:, 1] / 10.0, "k--", lw=0.8, label="alpha/10 (deg)")
        for y, c in ELDREDGE_CASES[nm].get("targets", []):
            ax.axhline(y, color=c, ls=":", lw=0.8)
        ax.set_xlabel("t* = tU/c"); ax.set_title(f"G3a Eldredge {nm}")
        ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(OUT_PNG, dpi=130)
    print(f"saved {OUT_PNG}", flush=True)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "eldredge"
    if which == "plunge":
        rec = run_plunge()
        np.save("platform/docs/diag/rvpm_s2_plunge.npy", rec)
        plunge_gate(rec)
    elif which == "plot":
        plot_all()
    else:
        for nm, case in ELDREDGE_CASES.items():
            rec = run_eldredge(nm, case)
            np.save(f"platform/docs/diag/rvpm_s2_eld_{nm}.npy", rec)
            eldredge_gate(nm, rec)
    print("GATES DONE", flush=True)
