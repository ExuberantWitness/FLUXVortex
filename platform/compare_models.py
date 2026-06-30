"""MULTI-MODEL comparison harness (2026-06-27), PARALLEL-per-model.

  python compare_models.py --model M4   # compute ONE model -> docs/models_cmp/M4.json  (run 5 in parallel)
  python compare_models.py --merge      # combine all models_cmp/*.json -> repro_compare.json + MODEL_SCORECARD.md

Candidate matrix (orthogonal MODE switches in _v2_robo.gpu_run_twist):
  M0 attached-UVLM floor | M1 Hirato kelvin LEV | M3 varA0+hold | M4 varA0+hold_detach (rec.) | ML legacy anchor.
  (M2 Modulation-varA0 == M1 and M5 kelvin+hold_detach == M4 in this ring framework -- the Kelvin-conservative
   cap makes varA0 and kelvin coincide whenever supercritical; omitted to halve cost, noted in the scorecard.)

Scorecard ranks by the USER PRIORITY LADDER: trend > sign > >50%-err count > <20%-err count > MAE.
"""
import sys, os, json, argparse, numpy as np

DOCS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'docs')
JP = os.path.join(DOCS, 'repro_data.json')
CMP = os.path.join(DOCS, 'models_cmp')
OUT = os.path.join(DOCS, 'repro_compare.json')
SCORE = os.path.join(DOCS, 'MODEL_SCORECARD.md')
os.makedirs(CMP, exist_ok=True)

GRID = dict(nc=4, ns=8, n_cycle=2, steps_per_cycle=60, wake_rows=60)
BASE = dict(real_geom=True, sym=True, les_suction=True, les_eta=1.0, d_para=3.0, a0_crit=0.23, **GRID)
MODELS = {
    'M0': dict(les_suction=False, lev_shed_mode='none',      lev_hold_mode='inviscid',    attached_drag='none'),
    'M1': dict(                    lev_shed_mode='kelvin',    lev_hold_mode='inviscid',    attached_drag='faure'),
    'M3': dict(                    lev_shed_mode='varA0',     lev_hold_mode='hold',        attached_drag='faure'),
    'M4': dict(                    lev_shed_mode='varA0',     lev_hold_mode='hold_detach', attached_drag='faure'),
    'ML': dict(                    lev_shed_mode='kinematic', lev_hold_mode='inviscid',    attached_drag='legacy',
               visc=True, prof_drag=True),
}
DESC = {'M0': 'attached UVLM floor', 'M1': 'Hirato kelvin LEV', 'M3': 'varA0 + hold',
        'M4': 'varA0 + hold_detach (rec.)', 'ML': 'legacy fp/sectional anchor'}


def cond_of(key, xi):
    fig, sub, param = key.split('|')
    if fig == '17':                 U, aoa, freq, tw = 8., 5., float(param), xi
    elif sub in ('a', 'b'):         U, aoa, freq, tw = float(param), 5., xi, 0.
    else:                           w, f = eval(param); U, aoa, freq, tw = float(w), 5., float(f), xi
    return U, aoa, freq, tw


def run_one(name):
    import warp as wp; wp.init()
    from _v2_robo import gpu_run_twist
    R = json.load(open(JP)); keys = sorted(R.keys())
    cfg = {**BASE, **MODELS[name]}; cache = {}; pred = {}
    for i, key in enumerate(keys):
        e = R[key]; kind = e['kind']; line = []
        for xi in e['x']:
            U, aoa, freq, tw = cond_of(key, xi)
            ck = (round(U, 1), round(aoa, 1), round(freq, 3), round(tw, 3))
            if ck not in cache:
                try:
                    r = gpu_run_twist(U=U, aoa_deg=aoa, freq=freq, twist_amp_deg=tw, twist_phase_deg=90., **cfg)
                    cache[ck] = (r['L_wind'], r['T_wind'])
                except Exception as ex:
                    print(f"  ERR {name} {ck}: {ex}", flush=True); cache[ck] = (float('nan'), float('nan'))
            L, T = cache[ck]; v = L if kind == 'L' else T
            line.append(None if (v != v) else float(v))
        pred[key] = line
        if (i + 1) % 10 == 0: print(f"[{name}] {i+1}/{len(keys)} ({len(cache)} runs)", flush=True)
    json.dump(pred, open(os.path.join(CMP, f"{name}.json"), 'w'))
    print(f"[{name}] DONE {len(keys)} keys, {len(cache)} unique GPU runs -> {name}.json", flush=True)


def slopes_sign(y):
    y = np.asarray([np.nan if v is None else v for v in y], float)
    return np.sign(np.diff(y))


def merge():
    R = json.load(open(JP)); keys = sorted(R.keys())
    have = [n for n in MODELS if os.path.exists(os.path.join(CMP, f"{n}.json"))]
    for n in have:
        pj = json.load(open(os.path.join(CMP, f"{n}.json")))
        for k in keys:
            R[k].setdefault('models', {})[n] = pj.get(k)
    json.dump(R, open(OUT, 'w')); print("saved", OUT, "| models:", have, flush=True)

    def score(name):
        th = tt = sh = st = big = small = 0; ae = []
        for k in keys:
            e = R[k]; exp = np.asarray(e['exp'], float); pr = e['models'].get(name)
            if pr is None: continue
            pr = np.asarray([np.nan if v is None else v for v in pr], float)
            m = np.isfinite(exp) & np.isfinite(pr)
            if m.sum() == 0: continue
            if m.sum() >= 2:
                se, sp = slopes_sign(exp[m]), slopes_sign(pr[m]); mm = np.isfinite(se) & np.isfinite(sp)
                if mm.sum() > 0: th += (float(np.mean(se[mm] == sp[mm])) >= 0.5); tt += 1
            sh += int(np.sum(np.sign(exp[m]) == np.sign(pr[m]))); st += int(m.sum())
            rel = np.abs(pr[m] - exp[m]) / (np.abs(exp[m]) + 1e-6)
            big += int(np.sum(rel > 0.5)); small += int(np.sum(rel < 0.2)); ae.extend(list(np.abs(pr[m] - exp[m])))
        return dict(trend=th / max(tt, 1), sign=sh / max(st, 1), big=big, small=small,
                    mae=float(np.mean(ae)) if ae else np.nan, rmse=float(np.sqrt(np.mean(np.square(ae)))) if ae else np.nan,
                    npts=st)
    sc = {n: score(n) for n in have}
    order = sorted(have, key=lambda n: (-sc[n]['trend'], -sc[n]['sign'], sc[n]['big'], -sc[n]['small'], sc[n]['mae']))
    L = ["# MODEL SCORECARD — candidate LESP-LEV models vs RoboEagle Fig17/18 measured", "",
         f"grid {GRID} | a0_crit=0.23 | phase +90 | both wings | {len(keys)} keys / {sc[order[0]]['npts']} pts",
         "", "Ranked by USER LADDER: **trend > sign > >50%err > <20%err > MAE**. "
         "(M2==M1, M5==M4 in this framework — omitted.)", "",
         "| rank | model | trend↑ | sign↑ | >50%err↓ | <20%err↑ | MAE(N)↓ | RMSE | description |",
         "|---|---|---|---|---|---|---|---|---|"]
    for i, n in enumerate(order):
        s = sc[n]
        L.append(f"| {i+1} | **{n}** | {s['trend']*100:.0f}% | {s['sign']*100:.0f}% | {s['big']} | {s['small']} "
                 f"| {s['mae']:.2f} | {s['rmse']:.2f} | {DESC[n]} |")
    L += ["", f"**Winner: {order[0]}** ({DESC[order[0]]}) by the priority ladder.", "",
          "Honest note: absolute under/over-prediction of held-LEV lift is the inviscid-LEV limit (Li JFM); "
          "trends/signs are the primary acceptance criteria. M0/ML bound the comparison (floor / legacy anchor)."]
    open(SCORE, 'w').write("\n".join(L)); print("\n".join(L), flush=True); print("\nsaved", SCORE, flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--model'); ap.add_argument('--merge', action='store_true')
    a = ap.parse_args()
    if a.merge: merge()
    elif a.model: run_one(a.model)
    else: print("use --model <name> or --merge")
