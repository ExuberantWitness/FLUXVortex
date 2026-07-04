"""P2 A/B comparison table: exp vs model caches on a condition set (default: the worst-15 P2 set).

  python ab_table.py                          # worst-15, models fix(K0),H4,H5,H6
  python ab_table.py --models fix,H4,H5       # subset
  python ab_table.py --conds 10_5_2.6_45,...  # custom set
"""
import os, json, argparse, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE)
from _v2_repro_nc12 import cond_of, ckey

WORST15 = ["10_5_2.6_45", "10_5_2.3_45", "10_5_2.0_45", "8_15_2.6_45", "8_5_2.6_45", "8_15_2.6_40",
           "10_5_2.6_30", "8_10_2.6_45", "8_0_2.6_45", "8_5_2.3_45",                       # worst-10 thrust
           "8_15_2.6_0", "8_15_2.6_22.5",                                                  # 19|d|15 lift trio (45 dup above)
           "6_5_2.0_0", "6_5_2.6_0", "8_5_2.0_15"]                                         # tw<30 sign-region guards
GUARDS = {"6.0_5.0_2.000_0.000", "6.0_5.0_2.600_0.000", "8.0_5.0_2.000_15.000"}


def exp_map():
    R = json.load(open('docs/repro_data.json')); m = {}
    for key in R:
        kind = R[key]['kind']
        for xi, e in zip(R[key]['x'], R[key]['exp']):
            ck = ckey(*cond_of(key, xi)); m.setdefault(ck, {}).setdefault(kind, []).append(e)
    return {ck: {k: float(np.mean(v)) for k, v in d.items()} for ck, d in m.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', default='fix,H4,H5,H6')
    ap.add_argument('--conds', default=','.join(WORST15))
    a = ap.parse_args()
    models = a.models.split(',')
    conds = [ckey(*[float(x) for x in c.split('_')]) for c in a.conds.split(',')]
    caches = {}
    for m in models:
        p = f'docs/repro_nc12/cache_nc4_cyc3_{m}.json'
        caches[m] = json.load(open(p)) if os.path.exists(p) else {}
    E = exp_map()
    hdr = f"{'cond':>22} {'expL':>6} {'expT':>6}" + ''.join(f" | {m}:L{'':>4} {m}:T{'':>4}" for m in models)
    print(hdr); print('-' * len(hdr))
    agg = {m: {'dT_w10': [], 'dL_19d15': [], 'dT_guard': [], 'dL_all': [], 'dT_all': []} for m in models}
    for ck in conds:
        e = E.get(ck, {}); eL, eT = e.get('L', np.nan), e.get('T', np.nan)
        row = f"{ck:>22} {eL:>6.2f} {eT:>6.2f}"
        for m in models:
            v = caches[m].get(ck)
            if v is None: row += f" | {'--':>7} {'--':>7}"; continue
            L, T = v
            row += f" | {L:>7.2f} {T:>7.2f}"
            tw = float(ck.split('_')[3]); guard = ck in GUARDS
            if not np.isnan(eT):
                agg[m]['dT_all'].append(abs(T - eT))
                if tw >= 30 and not guard: agg[m]['dT_w10'].append(abs(T - eT))
                if guard: agg[m]['dT_guard'].append(abs(T - eT))
            if not np.isnan(eL):
                agg[m]['dL_all'].append(abs(L - eL))
                if ck.startswith('8.0_15.0_2.600'): agg[m]['dL_19d15'].append(abs(L - eL))
        print(row)
    print(f"\n{'model':>6} {'MAE_T(w10 tw>=30)':>18} {'MAE_L(19d15)':>13} {'MAE_T(guard)':>13} {'MAE_T(set)':>11} {'MAE_L(set)':>11}")
    for m in models:
        g = agg[m]; f = lambda k: (f"{np.mean(g[k]):.2f}" if g[k] else '--')
        print(f"{m:>6} {f('dT_w10'):>18} {f('dL_19d15'):>13} {f('dT_guard'):>13} {f('dT_all'):>11} {f('dL_all'):>11}")


if __name__ == '__main__':
    main()
