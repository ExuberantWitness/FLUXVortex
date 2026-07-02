"""Reproduce Hirato et al. 2019 (J.Aircraft) Fig.15 CASE 1: SD7003 RECTANGULAR wing, AR=6, pure PITCH
RAMP 0->45deg about c/4 at reduced rate K=adot*c/(2U)=0.3, Re=20000, LESP_crit=0.27.
Compares LEV-ON (faithful vortex-sheet, lev_place='wake') vs LEV-OFF (attached UVLM).
Expected (paper Fig.15a): apparent-mass C_L spike at t*~1 (ramp start), C_L rises with alpha; LEV onset ~t*=1.6
REDUCES the lift-growth rate vs LEV-off (which keeps climbing); peak C_L ~4.

Usage: python _hirato_case1.py [nc] [ns] [spc] [shed_mode]
"""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import warp as wp; wp.init()
import _v2_robo as R

# --- Case-1 geometry & flow (AR=6 rect; Re=20000) ---
CHORD = 0.10; HALF_SPAN = 0.30            # AR = 2*half_span/chord = 6
NU = 1.5e-5; RE = 20000.0
U = RE * NU / CHORD                        # = 3.0 m/s  -> Re=20000
AR = 2 * HALF_SPAN / CHORD
LESP_CRIT = 0.27; TC = 0.085               # SD7003
NC = int(sys.argv[1]) if len(sys.argv) > 1 else 6
NS = int(sys.argv[2]) if len(sys.argv) > 2 else 12
SPC = int(sys.argv[3]) if len(sys.argv) > 3 else 200
SHED = sys.argv[4] if len(sys.argv) > 4 else 'kelvin'
# time: t* = U t / c goes 0..3.  t_max = 3 c/U.  freq=10,n_cycle=1 -> T=t_max=0.1s only if 3c/U=0.1 => here
# 3*c/U = 3*0.1/3 = 0.1 s.  So freq=10, n_cycle=1 => T=0.1s, dt=T/spc, N=spc.
FREQ = 10.0; NCYC = 1
T = 1.0 / FREQ; dt = T / SPC; N = NCYC * SPC
tstar = np.arange(N) * dt * U / CHORD      # nondimensional time per step

COMMON = dict(U=U, aoa_deg=0.0, freq=FREQ, n_cycle=NCYC, steps_per_cycle=SPC, wake_rows=min(SPC, 220),
              nc=NC, ns=NS, chord=CHORD, half_span=HALF_SPAN, flap_amp_deg=0.0, twist_amp_deg=0.0,
              real_geom=False, sym=True, tc_thick=TC,       # sym=True: root symmetry plane -> full AR=6 wing (paper Case 1)
              pitch_ramp=True, pitch_max=45.0, pitch_K=0.3, pitch_t0star=1.0)

def alpha_of_tstar(ts):
    """the ramp alpha(t*) the code applies (canonical Eldredge/Granlund smoothed ramp-hold, K=0.3)."""
    pmax = np.radians(45.0); t1s = 1.0; t2s = t1s + pmax / (2.0 * 0.3); a_sm = 6.0
    lnc = lambda z: np.logaddexp(z, -z) - np.log(2.0)
    f = lnc(a_sm * (ts - t1s)) - lnc(a_sm * (ts - t2s)); D = a_sm * (t2s - t1s)
    return np.degrees(pmax * np.clip((f + D) / (2.0 * D), 0.0, 1.0))

def cl_hist(res):
    """C_L(t) history following paper Eq.22  C_L = C_N cos(a) + C_S sin(a): the unsteady-Bernoulli NORMAL force
    (Lh_bern, = C_N cos a projection) PLUS the leading-edge SUCTION lift component (Lh_les, = C_S sin a). aoa=0
    so body-z == wind-z (lift _|_ freestream). Normalized by half-wing S (the 2x for full wing cancels in q*S)."""
    S = HALF_SPAN * CHORD; q = 0.5 * R.ug.RHO * U * U
    Lb = np.asarray(res['Lh_bern']); Lles = np.asarray(res['Lh_les'])
    cl_full = (Lb + Lles) / (q * S + 1e-12)
    return cl_full, cl_full

def run(label, **kw):
    k = dict(COMMON); k.update(kw)
    r = R.gpu_run_twist(**k)
    clkj, clb = cl_hist(r)
    print(f"[{label}]  L_wind={r['L_wind']:.3f}N  max|Lh_bern|={np.abs(r['Lh_bern']).max():.2f}N  "
          f"CL_bern(final)={clb[-1]:.2f}  CL_bern(max)={clb.max():.2f}  (blow-up if |Lh_bern|>>50)", flush=True)
    return clkj, clb, r

if __name__ == '__main__':
    print(f"HIRATO CASE 1: rect AR={AR:.1f}, U={U:.3f} m/s (Re={RE:.0f}), chord={CHORD}, half_span={HALF_SPAN}", flush=True)
    print(f"pitch ramp 0->45deg K=0.3; nc={NC} ns={NS} spc={SPC} N={N} shed={SHED}; LESP_crit={LESP_CRIT}", flush=True)
    print(f"alpha(t*): t*=1.0->{alpha_of_tstar(1.0):.1f}deg  1.6->{alpha_of_tstar(1.6):.1f}  2.0->{alpha_of_tstar(2.0):.1f}  3.0->{alpha_of_tstar(3.0):.1f}", flush=True)
    off_kj, off_b, off_r = run('LEV-OFF', lev_shed_mode='none', les_suction=True, les_eta=1.0, a0_crit=LESP_CRIT,
                               lesp_crit_deg=90.0)   # no LEV -> FULL attached LE suction (no cap), like paper 'UVLM without LEV'
    on_kj, on_b, on_r = run('LEV-ON(wake)', lev_shed_mode=SHED, lev_place='wake', lev_sheet=True,
                             les_suction=True, les_eta=1.0, a0_crit=LESP_CRIT, lev_sign=1.0)
    # save curves
    np.savez(os.path.join('docs', 'hirato_case1.npz'), tstar=tstar, alpha=alpha_of_tstar(tstar),
             off_b=off_b, on_b=on_b, off_kj=off_kj, on_kj=on_kj)
    # quick text comparison at key t*
    for ts in (1.0, 1.6, 2.0, 2.5, 3.0):
        i = int(np.argmin(np.abs(tstar - ts)))
        print(f"  t*={ts:.1f} a={alpha_of_tstar(ts):5.1f}deg : CL_off={off_b[i]:+.2f}  CL_on={on_b[i]:+.2f}  "
              f"(LEV effect {on_b[i]-off_b[i]:+.2f})", flush=True)
    print("DONE", flush=True)
