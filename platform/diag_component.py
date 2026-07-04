"""P1 deep-stall overshoot attribution: per-channel force decomposition (WIND frame) for H4 at the
three representative blowup conditions:

  10/5/2.6/45   worst thrust overshoot   (exp T=-0.11, H4 -15.39)
  8/15/2.6/0    lift overshoot, NO twist (exp L=13.33, H4 +17.99)
  8/15/2.6/45   double-blowup corner     (exp T=-4.18/L=12.55, H4 -15.90/+22.85)

Channels (sum == Fzb_tot/Fxb_tot): bern (Bernoulli base) + les (Garrick LE suction) + vtx (hirato
vnf vortex normal force) + pd (faure attached drag) + vis (friction) + stall (Fix1) + d_para (rig).

  python diag_component.py [--cfg H4] [--conds 10_5_2.6_45,...]
"""
import os, json, argparse, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE)
OD = os.path.join(HERE, 'docs', 'diag'); os.makedirs(OD, exist_ok=True)
from _v2_repro_nc12 import CFG_PRESETS, cond_of

NC, NS, NCYC = 4, 16, 3
CONDS = [(10.0, 5.0, 2.6, 45.0), (8.0, 15.0, 2.6, 0.0), (8.0, 15.0, 2.6, 45.0)]
CHANNELS = [('bern', 'Lh_bern', 'Xh_bern'), ('les', 'Lh_les', 'Xh_les'), ('vtx', 'Lh_vtx', 'Xh_vtx'),
            ('pd', 'Lh_pd', 'Xh_pd'), ('vis', 'Lh_vis', 'Xh_vis'), ('stall', 'Lh_stall', None)]


def robmean(a, last):                       # same winsorized cycle-mean as production (_v2_robo)
    a = np.asarray(a, float)[last]
    m = np.median(a); mad = np.median(np.abs(a - m)) + 1e-12
    return 2.0 * np.mean(np.clip(a, m - 8 * 1.4826 * mad, m + 8 * 1.4826 * mad))


def exp_of(U, aoa, f, tw):                  # measured L/T at this condition (any curve containing it)
    R = json.load(open(os.path.join(HERE, 'docs', 'repro_data.json'))); out = {'L': [], 'T': []}
    for key in R:
        for xi, e in zip(R[key]['x'], R[key]['exp']):
            if cond_of(key, xi) == (U, aoa, round(f / 0.05) * 0.05, round(tw / 0.5) * 0.5):
                out[R[key]['kind']].append(e)
    return {k: (np.mean(v) if v else np.nan) for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg', default='H4', choices=sorted(CFG_PRESETS))
    ap.add_argument('--conds', help='comma list U_aoa_f_tw (default: the 3 blowup reps)')
    a = ap.parse_args()
    conds = [tuple(float(x) for x in c.split('_')) for c in a.conds.split(',')] if a.conds else CONDS
    import warp as wp; wp.init()
    from _v2_robo import gpu_run_twist
    kw = dict(CFG_PRESETS[a.cfg]); results = {}
    for (U, aoa, f, tw) in conds:
        spc = int(round(15.0 * U * NC / f / 60.0)) * 60
        r = gpu_run_twist(U=U, aoa_deg=aoa, freq=f, twist_amp_deg=tw, twist_phase_deg=90.0,
                          nc=NC, ns=NS, n_cycle=NCYC, steps_per_cycle=spc, wake_rows=spc, **kw)
        last = slice((NCYC - 1) * spc, NCYC * spc)
        ca, sa = np.cos(np.radians(aoa)), np.sin(np.radians(aoa))
        exp = exp_of(U, aoa, f, tw)
        print(f"\n=== {a.cfg} {U:g}/{aoa:g}/{f:g}/tw{tw:g}  ->  L_wind={r['L_wind']:+.2f} T_wind={r['T_wind']:+.2f}"
              f"   (exp L={exp['L']:.2f} T={exp['T']:.2f})")
        print(f"{'channel':>8} {'Fz_body':>8} {'Fx_body':>8} {'L_wind':>8} {'T_wind':>8}   [robust cycle-mean, N]")
        rows = {}; sz = sx = 0.0
        for name, lk, xk in CHANNELS:
            fz = robmean(r[lk], last) if lk in r else 0.0
            fx = robmean(r[xk], last) if xk and xk in r else 0.0
            Lw = fz * ca - fx * sa; Tw = -(fx * ca + fz * sa)
            sz += fz; sx += fx; rows[name] = dict(Fz=fz, Fx=fx, L=Lw, T=Tw)
            print(f"{name:>8} {fz:>8.2f} {fx:>8.2f} {Lw:>8.2f} {Tw:>8.2f}")
        dpar = kw.get('d_para', 0.0) * (U / 8.0) ** 2
        rows['d_para'] = dict(Fz=dpar * sa, Fx=dpar * ca, L=0.0, T=-dpar)
        print(f"{'d_para':>8} {dpar * sa:>8.2f} {dpar * ca:>8.2f} {0.0:>8.2f} {-dpar:>8.2f}")
        sz += dpar * sa; sx += dpar * ca
        print(f"{'SUM':>8} {sz:>8.2f} {sx:>8.2f} {sz * ca - sx * sa:>8.2f} {-(sx * ca + sz * sa):>8.2f}"
              f"   (prod totals L={r['L_wind']:+.2f} T={r['T_wind']:+.2f}; SUM uses per-channel winsorize -> small diff ok)")
        results[f"{U:g}_{aoa:g}_{f:g}_{tw:g}"] = dict(cfg=a.cfg, rows=rows, exp=exp,
                                                      L_wind=float(r['L_wind']), T_wind=float(r['T_wind']))
    jp = os.path.join(OD, f"components_{a.cfg}.json"); json.dump(results, open(jp, 'w'), indent=1)
    print(f"\nsaved {jp}", flush=True)


if __name__ == '__main__':
    main()
