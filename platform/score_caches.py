"""Ladder scorecard over condition-keyed prediction caches vs the FULL measured set (Fig17+18+19).

Reads docs/repro_data.json (extended with Fig19 by extend_repro_data.py) and one cache_*.json per model
(produced by _v2_repro_nc12.py --cfg <name>), scores each model by the USER PRIORITY LADDER
(trend > sign > #(>50% rel-err) > #(<20% rel-err) > MAE), broken down by kind (L/T) and figure (17/18/19).
Same metric definitions as compare_models.py (slopes_sign per curve).

  python score_caches.py --nc 4 --ncyc 3 --cfgs K0,H2,H3       # -> docs/SCORECARD_full.md (+ stdout)
"""
import os, json, argparse, numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DOCS = os.path.join(HERE, 'docs'); OD = os.path.join(DOCS, 'repro_nc12')
JP = os.path.join(DOCS, 'repro_data.json')

import importlib
_r = importlib.import_module('_v2_repro_nc12')          # reuse cond_of/ckey (incl. Fig19 mapping)


def slopes_sign(v):
    v = np.asarray(v, float)
    return np.sign(np.diff(v))


def cache_file(cfg, nc, ncyc):
    if cfg == 'K0':
        return os.path.join(OD, f"cache_nc{nc}_cyc{ncyc}_fix.json")
    return os.path.join(OD, f"cache_nc{nc}_cyc{ncyc}_{cfg}.json")


def pred_line(R, key, cache):
    kind = R[key]['kind']; out = []
    for xi in R[key]['x']:
        U, aoa, freq, tw = _r.cond_of(key, xi)
        v = cache.get(_r.ckey(U, aoa, freq, tw))
        out.append(np.nan if (v is None or not np.isfinite(v[0])) else v[0 if kind == 'L' else 1])
    return np.asarray(out, float)


def score(R, cache, keys):
    th = tt = sh = st = big = small = 0; ae = []
    for k in keys:
        exp = np.asarray(R[k]['exp'], float); pr = pred_line(R, k, cache)
        m = np.isfinite(exp) & np.isfinite(pr)
        if m.sum() == 0: continue
        if m.sum() >= 2:
            se, sp = slopes_sign(exp[m]), slopes_sign(pr[m])
            mm = (se != 0) & (sp != 0)
            if mm.sum() > 0: th += (float(np.mean(se[mm] == sp[mm])) >= 0.5); tt += 1
        sh += int(np.sum(np.sign(exp[m]) == np.sign(pr[m]))); st += int(m.sum())
        rel = np.abs(pr[m] - exp[m]) / (np.abs(exp[m]) + 1e-6)
        big += int(np.sum(rel > 0.5)); small += int(np.sum(rel < 0.2)); ae.extend(list(np.abs(pr[m] - exp[m])))
    return dict(trend=th / max(tt, 1), ncurves=tt, sign=sh / max(st, 1), big=big, small=small,
                mae=float(np.mean(ae)) if ae else np.nan, npts=st)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--nc', type=int, default=4); ap.add_argument('--ncyc', type=int, default=3)
    ap.add_argument('--cfgs', default='K0,H2,H3'); ap.add_argument('--out', default='SCORECARD_full.md')
    a = ap.parse_args()
    R = json.load(open(JP)); cfgs = a.cfgs.split(',')
    caches = {}
    for c in cfgs:
        f = cache_file(c, a.nc, a.ncyc)
        caches[c] = json.load(open(f)) if os.path.exists(f) else {}
        print(f"[{c}] cache {os.path.basename(f)}: {len(caches[c])} conds")
    groups = {'ALL': sorted(R), 'Fig17': [k for k in R if k.startswith('17')],
              'Fig18': [k for k in R if k.startswith('18')], 'Fig19': [k for k in R if k.startswith('19')],
              'L(all)': [k for k in R if R[k]['kind'] == 'L'], 'T(all)': [k for k in R if R[k]['kind'] == 'T']}
    L = [f"# SCORECARD — full data.md (Fig17+18+19) | nc={a.nc} ncyc={a.ncyc} | ladder: trend>sign>>50%err><20%err>MAE", ""]
    overall = {}
    for gname, keys in groups.items():
        L += [f"## {gname} ({len(keys)} curves)", "",
              "| model | trend↑ | sign↑ | >50%err↓ | <20%err↑ | MAE(N)↓ | pts |", "|---|---|---|---|---|---|---|"]
        for c in cfgs:
            s = score(R, caches[c], keys)
            if gname == 'ALL': overall[c] = s
            L.append(f"| {c} | {s['trend']*100:.0f}% ({s['ncurves']}) | {s['sign']*100:.0f}% | {s['big']} | {s['small']} "
                     f"| {s['mae']:.2f} | {s['npts']} |")
        L.append("")
    order = sorted([c for c in cfgs if overall.get(c, {}).get('npts', 0) > 0],
                   key=lambda c: (-overall[c]['trend'], -overall[c]['sign'], overall[c]['big'],
                                  -overall[c]['small'], overall[c]['mae']))
    if order: L += [f"**Ladder winner (ALL): {order[0]}**", ""]
    txt = "\n".join(L)
    open(os.path.join(DOCS, a.out), 'w').write(txt); print(txt)
    print("saved", os.path.join('docs', a.out))


if __name__ == '__main__':
    main()
