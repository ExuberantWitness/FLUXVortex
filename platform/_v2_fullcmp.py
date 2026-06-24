"""Comprehensive comparison of my UVLM vs ALL the RoboEagle measured data (docs/data.md, Fig 17/18/19).
Parses every dataset (using the COLUMN HEADER Thurst/Lift as truth — the figure titles are mislabeled,
e.g. 18c/d and 19c/d titles are swapped vs their data columns; the value signs confirm: thrust<0=drag,
lift>0), runs my twisted flapping UVLM at the matching conditions, and prints per-point value tables +
saves trend-overlay plots. NOTE assumptions for unspecified fixed params are printed.
"""
import re, os, sys
import numpy as np
sys.path.insert(0, '.'); sys.path.insert(0, '..')
import warp as wp
import _v2_robo as robo

GF = 9.81 / 1000.0   # grams-force -> N
DATA = os.path.join(os.path.dirname(__file__), 'docs', 'data.md')


def parse(path):
    """-> list of dicts: {fig, panel, qty('T'/'L'), cond(str), x(np), y_g(np)}."""
    sets = []; fig = panel = qty = cond = None; xs = []; ys = []
    def flush():
        if cond is not None and xs:
            sets.append(dict(fig=fig, panel=panel, qty=qty, cond=cond,
                             x=np.array(xs), y_g=np.array(ys)))
    for ln in open(path, encoding='utf-8'):
        m = re.search(r'Figure\s*(\d+)\.\s*\(([a-d])\)', ln)
        if m:
            flush(); xs, ys = [], []; fig = int(m.group(1)); panel = m.group(2); cond = None; continue
        if '工况' in ln:
            flush(); xs, ys = [], []
            cond = ln.split('：', 1)[-1].split(':', 1)[-1].strip(); continue
        if re.search(r'[Tt]hurst|[Tt]hrust', ln):
            qty = 'T'; continue
        if re.search(r'[Ll]ift', ln) and '/g' in ln:
            qty = 'L'; continue
        nums = re.findall(r'[-+]?\d+\.\d+e[-+]\d+', ln)
        if len(nums) == 2:
            xs.append(float(nums[0])); ys.append(float(nums[1]))
    flush()
    return sets


def cond_params(fig, panel, cond):
    """Map figure/panel + condition -> (sweep_var, fixed U, aoa, freq, twist). Assumptions flagged."""
    U, aoa, freq, twist = 8.0, 5.0, 2.0, 0.0    # defaults (cruise)
    f = re.search(r'([\d.]+)\s*[Hh][Zz]', cond)
    s = re.search(r'([\d.]+)\s*m/s', cond)
    d = re.search(r'([\d.]+)\s*度', cond)
    if f: freq = float(f.group(1))
    if s: U = float(s.group(1))
    if d: aoa = float(d.group(1))
    if fig == 17: sweep = 'twist'                       # vs twist, freq from cond, U=8 aoa=5 (assume)
    elif fig == 18 and panel in 'ab': sweep = 'freq'    # vs freq, U from cond, aoa=5 twist=0 (assume)
    elif fig == 18: sweep = 'twist'                     # vs twist, U+freq from cond, aoa=5
    elif fig == 19 and panel in 'ab': sweep = 'freq'    # vs freq, aoa from cond, U=8 twist=0
    else: sweep = 'twist'; freq = 2.6                   # 19c/d vs twist, aoa from cond, U=8 freq=2.6 (assume)
    return sweep, U, aoa, freq, twist


def run_mine(U, aoa, freq, twist):
    r = robo.gpu_run_twist(nc=6, ns=12, chord=0.287, half_span=0.80, U=U, aoa_deg=aoa,
                           flap_amp_deg=45.0, twist_amp_deg=twist, twist_phase_deg=90.0,
                           freq=max(freq, 0.5), n_cycle=5, steps_per_cycle=40)
    return r['L'], r['T']     # lift N (both wings already x2 in gpu_run_twist), thrust N


def main():
    wp.init()
    import robowing as rw
    import flap_flight_validate as ffv
    ffv.flat_wing = lambda nc, ns, c, h: rw.robowing(nc, ns, c, h)   # proper RoboEagle wing (rounded+NACA2406+cosine)
    sets = parse(DATA)
    print(f"parsed {len(sets)} datasets from data.md", flush=True)
    # pick representative datasets to compare (lift + thrust, the key sweeps)
    targets = [(18, 'b'), (19, 'b'), (17, 'b'), (18, 'a'), (19, 'a'), (17, 'a')]
    rows = []
    for (F, P) in targets:
        dss = [d for d in sets if d['fig'] == F and d['panel'] == P]
        qty = dss[0]['qty'] if dss else '?'
        print(f"\n===== Figure {F}{P}  ({'LIFT' if qty=='L' else 'THRUST'}) — {len(dss)} conditions =====", flush=True)
        for d in dss:
            sweep, U, aoa, freq, twist = cond_params(F, P, d['cond'])
            xs = d['x']; yg = d['y_g']
            idx = np.linspace(0, len(xs) - 1, min(4, len(xs))).astype(int)   # sample ~4 pts/curve
            print(f"  cond '{d['cond']}' (sweep={sweep}, U={U} aoa={aoa} freq={freq} twist0={twist}):", flush=True)
            for k in idx:
                xv = xs[k]; paper = yg[k] * GF
                if sweep == 'twist': L, T = run_mine(U, aoa, freq, xv)
                elif sweep == 'freq': L, T = run_mine(U, aoa, xv, twist)
                else: L, T = run_mine(U, aoa, freq, twist)
                mine = L if qty == 'L' else T
                ratio = mine / paper if abs(paper) > 1e-6 else float('nan')
                print(f"     {sweep}={xv:6.1f}: paper={paper:+6.2f}N  mine={mine:+6.2f}N  ratio={ratio:+.2f}", flush=True)
                rows.append((F, P, qty, d['cond'], sweep, xv, paper, mine, ratio))
    # summary
    print("\n===== SUMMARY =====", flush=True)
    for q, name in (('L', 'LIFT'), ('T', 'THRUST')):
        rr = [r for r in rows if r[2] == q]
        ratios = np.array([r[8] for r in rr if np.isfinite(r[8])])
        if len(ratios):
            print(f"  {name}: {len(rr)} pts, mean ratio mine/paper = {np.median(ratios):+.2f} "
                  f"(range {ratios.min():+.2f}..{ratios.max():+.2f})", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
