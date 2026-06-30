"""FULL data.md sweep at the GRID-INDEPENDENT nc=12 production config (Ansari-LEV), RESUMABLE.
Each unique (U,aoa,freq,twist) is run with spc matched to hold the validated U*dt/chord ratio
(spc = 15*U*nc/freq, rounded to 60; = 720 at the U8/2Hz/nc12 validation point). Results cached to
disk incrementally so a multi-day run survives interruption (just re-launch to resume).

  python _v2_repro_nc12.py --dry    # list unique conditions + spc + cost estimate (no GPU)
  python _v2_repro_nc12.py --run    # run all (resumable); saves docs/repro_nc12/cache.json incrementally
  python _v2_repro_nc12.py --plot   # docs/repro_fig17.png + repro_fig18.png (exp vs nc12 model) + MAE
"""
import sys, os, json, time, argparse, numpy as np
DOCS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'docs')
JP = os.path.join(DOCS, 'repro_data.json')
OD = os.path.join(DOCS, 'repro_nc12'); os.makedirs(OD, exist_ok=True)

NC = 12; NS = 16; NCYC = 4; WINDS = None; FIXMODE = False
PROD = dict(real_geom=True, sym=True, les_suction=True, les_eta=1.0, d_para=3.0, a0_crit=0.23,
            lev_shed_mode='kelvin', lev_hold_mode='inviscid', attached_drag='faure',
            lev_sheet=True, lev_place='ansari', lev_sign=1.0)
# 2026-06-30 Fix1 ONLY (lift): geometric quasi-steady stall, physics-anchored (NACA-2406 alpha_ss/width).
# Fix2 (friction) DROPPED: skin friction is freq-independent (~U^2, chordwise-flow-dominated), so it can NOT
# explain the thrust freq-SLOPE error (model -3.2->+2.2 vs meas -1.1->+0.7, ~2.9x too steep). Thrust = open item.
FIX = dict(geo_stall=True, geo_stall_deg=12.0, geo_stall_width=16.0)

def cache_path():     # grid+fix-tagged so different grids / model variants never collide
    return os.path.join(OD, f"cache_nc{NC}_cyc{NCYC}{'_fix' if FIXMODE else ''}.json")


def cond_of(key, xi):
    fig, sub, param = key.split('|')
    if fig == '17':                 U, aoa, freq, tw = 8., 5., float(param), xi
    elif sub in ('a', 'b'):         U, aoa, freq, tw = float(param), 5., xi, 0.
    else:                           w, f = eval(param); U, aoa, freq, tw = float(w), 5., float(f), xi
    # quantize away parse-noise duplicates (physically identical points run once): tw->0.5deg, freq->0.05Hz
    U = round(U, 1); aoa = round(aoa, 1); freq = round(freq / 0.05) * 0.05; tw = round(tw / 0.5) * 0.5
    return U, aoa, freq, tw


def spc_of(U, freq):
    return int(round(15.0 * U * NC / freq / 60.0)) * 60     # holds U*dt/chord = validated ratio


def unique_conds():
    R = json.load(open(JP)); seen = {}
    for key in sorted(R.keys()):
        for xi in R[key]['x']:
            U, aoa, freq, tw = cond_of(key, xi)
            if WINDS is not None and round(U, 1) not in WINDS: continue
            ck = f"{U:.1f}_{aoa:.1f}_{freq:.3f}_{tw:.3f}"
            seen[ck] = (U, aoa, freq, tw)
    return seen


def ckey(U, aoa, freq, tw): return f"{U:.1f}_{aoa:.1f}_{freq:.3f}_{tw:.3f}"


def dry():
    conds = unique_conds()
    tot = 0.0
    print(f"{'cond':>26} {'spc':>5} {'est_min':>7}")
    for ck, (U, aoa, freq, tw) in sorted(conds.items()):
        spc = spc_of(U, freq); est = 2110.0 * (spc / 720.0) ** 2 / 60.0   # ~35min at spc720 baseline (n_cycle4)
        tot += est
        print(f"{ck:>26} {spc:>5} {est:>7.0f}")
    print(f"\n{len(conds)} unique nc12 runs | total est ~{tot/60:.1f} h ~{tot/60/24:.1f} days", flush=True)


def run():
    import warp as wp; wp.init()
    from _v2_robo import gpu_run_twist
    conds = unique_conds(); CACHE = cache_path(); kw = dict(PROD, **(FIX if FIXMODE else {}))
    cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
    todo = [(ck, c) for ck, c in sorted(conds.items()) if ck not in cache]
    print(f"resume[{os.path.basename(CACHE)}]: {len(cache)} done, {len(todo)} to run (of {len(conds)})", flush=True)
    for i, (ck, (U, aoa, freq, tw)) in enumerate(todo):
        spc = spc_of(U, freq); t0 = time.time()
        try:
            r = gpu_run_twist(U=U, aoa_deg=aoa, freq=freq, twist_amp_deg=tw, twist_phase_deg=90.0,
                              nc=NC, ns=NS, n_cycle=NCYC, steps_per_cycle=spc, wake_rows=spc, **kw)
            cache[ck] = [float(r['L_wind']), float(r['T_wind'])]
        except Exception as ex:
            print(f"  ERR {ck}: {ex}", flush=True); cache[ck] = [float('nan'), float('nan')]
        json.dump(cache, open(CACHE, 'w'))                          # INCREMENTAL save (resumable)
        L, T = cache[ck]
        print(f"[{i+1}/{len(todo)}] {ck} spc={spc} -> L={L:.2f} T={T:.2f} ({time.time()-t0:.0f}s)", flush=True)
    print(f"DONE all {len(conds)} conditions cached", flush=True)


def _pred_lines():
    R = json.load(open(JP)); cache = json.load(open(cache_path()))
    pred = {}
    for key in sorted(R.keys()):
        kind = R[key]['kind']; line = []
        for xi in R[key]['x']:
            U, aoa, freq, tw = cond_of(key, xi); v = cache.get(ckey(U, aoa, freq, tw))
            if v is None: line.append(None)
            else: line.append(None if np.isnan(v[0 if kind == 'L' else 1]) else v[0 if kind == 'L' else 1])
        pred[key] = line
    return R, pred


def plot():
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    R, pred = _pred_lines(); CMAP = plt.cm.viridis
    nan = lambda L: [np.nan if v is None else v for v in (L or [])]
    ae = []
    def line(ax, key, c, lab):
        if key not in R: return
        d = R[key]; x = d['x']
        ax.plot(x, d['exp'], '-o', color=c, ms=5, lw=2.2, label=f"exp {lab}")
        p = nan(pred.get(key)); ax.plot(x, p, '--x', color=c, ms=5, lw=1.5, alpha=0.85, label=f"nc12 {lab}")
        e = np.asarray(d['exp'], float); pp = np.asarray(p, float); m = np.isfinite(e) & np.isfinite(pp)
        ae.extend(list(np.abs(pp[m] - e[m])))
    # Fig17: thrust(a)+lift(b) vs twist, 5 freqs
    FREQS = [1.4, 1.7, 2.0, 2.3, 2.6]
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    for i, f in enumerate(FREQS):
        c = CMAP(i / 4); line(ax[0], f"17|a|{f}", c, f"{f}Hz"); line(ax[1], f"17|b|{f}", c, f"{f}Hz")
    ax[0].set_title("Fig17a net thrust vs twist"); ax[1].set_title("Fig17b lift vs twist")
    for a in ax: a.set_xlabel("twist (deg)"); a.set_ylabel("force (N)"); a.grid(alpha=0.3); a.legend(fontsize=6, ncol=2)
    fig.suptitle("Fig17 exp(-o) vs nc12 Ansari-LEV(--x)  [8m/s AoA5, per freq]", fontsize=12)
    fig.tight_layout(); fig.savefig(os.path.join(DOCS, "repro_fig17.png"), dpi=110); plt.close(fig)
    # Fig18: thrust(a)+lift(b) vs freq, 3 winds
    WINDS = [6.0, 8.0, 10.0]
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    for i, U in enumerate(WINDS):
        c = CMAP(i / 2); line(ax[0], f"18|a|{U}", c, f"{U:.0f}m/s"); line(ax[1], f"18|b|{U}", c, f"{U:.0f}m/s")
    ax[0].set_title("Fig18a net thrust vs freq"); ax[1].set_title("Fig18b lift vs freq")
    for a in ax: a.set_xlabel("freq (Hz)"); a.set_ylabel("force (N)"); a.grid(alpha=0.3); a.legend(fontsize=7, ncol=2)
    fig.suptitle("Fig18 exp(-o) vs nc12 Ansari-LEV(--x)  [twist0, per wind]", fontsize=12)
    fig.tight_layout(); fig.savefig(os.path.join(DOCS, "repro_fig18.png"), dpi=110); plt.close(fig)
    done = sum(1 for v in json.load(open(cache_path())).values() if not np.isnan(v[0]))
    print(f"saved repro_fig17.png + repro_fig18.png | {done} conds done | MAE={np.mean(ae) if ae else float('nan'):.2f}N "
          f"over {len(ae)} matched pts", flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry', action='store_true'); ap.add_argument('--run', action='store_true'); ap.add_argument('--plot', action='store_true')
    ap.add_argument('--nc', type=int); ap.add_argument('--ncyc', type=int); ap.add_argument('--winds'); ap.add_argument('--fix', action='store_true')
    a = ap.parse_args()
    if a.nc: NC = a.nc
    if a.ncyc: NCYC = a.ncyc
    if a.winds: WINDS = [round(float(x), 1) for x in a.winds.split(',')]
    FIXMODE = a.fix
    if a.dry: dry()
    elif a.run: run()
    elif a.plot: plot()
    else: print("use --dry | --run | --plot")
