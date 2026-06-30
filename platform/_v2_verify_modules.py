"""Per-condition BEFORE/AFTER verification of the two new modules vs RoboEagle measured data.
Runs THREE levels per condition -> baseline | +Fix1(geo_stall) | +Fix1+Fix2(fric_drag) -> shows each
module's contribution. Resumable (results cached to disk). Plots (a) 4-panel measured-vs-3-levels and
(b) per-condition |error| bars (baseline vs +Fix1 vs +both) for lift and thrust -> did adding each module
improve or worsen each condition?

  python _v2_verify_modules.py [nc] [winds]    # default nc=4, winds=10
"""
import sys, os, json, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import warp as wp; wp.init()
from _v2_robo import gpu_run_twist
from grid_indep import MODEL
import importlib.util
spec = importlib.util.spec_from_file_location("rn", "_v2_repro_nc12.py"); rn = importlib.util.module_from_spec(spec); spec.loader.exec_module(rn)
DOCS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'docs')

NC = int(sys.argv[1]) if len(sys.argv) > 1 else 4
WINDS = [round(float(x), 1) for x in sys.argv[2].split(',')] if len(sys.argv) > 2 else [10.0]
SPC = 60 * NC; BASE = {k: v for k, v in MODEL.items() if k != 'n_cycle'}
LEVELS = {'base': {},
          'f1':   dict(geo_stall=True, geo_stall_deg=12.0, geo_stall_width=16.0),
          'both': dict(geo_stall=True, geo_stall_deg=12.0, geo_stall_width=16.0, fric_drag=True, cf_mode='turbulent')}
COL = {'base': 'tab:gray', 'f1': 'tab:blue', 'both': 'tab:red'}
LAB = {'base': 'baseline', 'f1': '+Fix1 stall', 'both': '+Fix1+Fix2'}
CACHE = os.path.join(DOCS, 'repro_nc12', f'verify3_nc{NC}_U{"-".join(f"{w:.0f}" for w in WINDS)}.json')
cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
R = json.load(open(os.path.join(DOCS, 'repro_data.json')))

def run(U, aoa, freq, tw, lvl):
    ck = f"{lvl}|{U:.1f}_{aoa:.1f}_{freq:.3f}_{tw:.3f}"
    if ck not in cache:
        r = gpu_run_twist(U=U, aoa_deg=aoa, freq=freq, twist_amp_deg=tw, twist_phase_deg=90.0,
                          nc=NC, ns=16, n_cycle=3, steps_per_cycle=SPC, wake_rows=SPC, **BASE, **LEVELS[lvl])
        cache[ck] = [float(r['L_wind']), float(r['T_wind'])]
        json.dump(cache, open(CACHE, 'w'))                      # incremental (resumable)
    return cache[ck]

# ---- compute all (resumable) ----
conds = []
for key in sorted(R.keys()):
    e = R[key]; kind = e['kind']
    for xi, ev in zip(e['x'], e['exp']):
        U, aoa, freq, tw = rn.cond_of(key, xi)
        if U not in WINDS or ev != ev: continue
        conds.append((key, xi, kind, ev, U, aoa, freq, tw))
done = 0
for (key, xi, kind, ev, U, aoa, freq, tw) in conds:
    for lvl in LEVELS:
        run(U, aoa, freq, tw, lvl)
    done += 1
    if done % 5 == 0: print(f"  {done}/{len(conds)} conds done", flush=True)
print(f"all {len(conds)} conds x 3 levels cached -> {os.path.basename(CACHE)}", flush=True)

# ---- (a) per-condition |error| change table + MAE ----
def err(kind, lvl):
    es = []
    for (key, xi, k, ev, U, aoa, freq, tw) in conds:
        if k != kind: continue
        L, T = run(U, aoa, freq, tw, lvl); es.append(abs((L if kind == 'L' else T) - ev))
    return np.array(es)
print("\n=== MAE per level ===")
for kind, nm in [('L', 'LIFT'), ('T', 'THRUST')]:
    print(f"  {nm}: baseline {err(kind,'base').mean():.2f}N -> +Fix1 {err(kind,'f1').mean():.2f}N -> +both {err(kind,'both').mean():.2f}N", flush=True)

# ---- (b) plots ----
def ckey(U, aoa, freq, tw): return f"{U:.1f}_{aoa:.1f}_{freq:.3f}_{tw:.3f}"
def modline(key, lvl):
    e = R[key]; kind = e['kind']; out = []
    for xi in e['x']:
        U, aoa, freq, tw = rn.cond_of(key, xi)
        if U not in WINDS: out.append(np.nan); continue
        L, T = run(U, aoa, freq, tw, lvl); out.append(L if kind == 'L' else T)
    return e['x'], e['exp'], out
W0 = WINDS[0]; CM = plt.cm.viridis
# 4-panel measured vs 3 levels
fig, ax = plt.subplots(2, 2, figsize=(16, 10))
for key, a, ttl, yl in [(f'18|b|{W0}', ax[0, 0], f'LIFT vs freq (U={W0:.0f}, twist0)', 'lift (N)'),
                        (f'18|a|{W0}', ax[0, 1], f'THRUST vs freq (U={W0:.0f}, twist0)', 'thrust (N)')]:
    if key not in R: continue
    x, ex, _ = modline(key, 'base'); a.plot(x, ex, '-o', c='k', ms=6, lw=2.4, label='measured', zorder=5)
    for lvl in LEVELS:
        _, _, m = modline(key, lvl); a.plot(x, m, '--x', c=COL[lvl], ms=6, lw=1.6, label=LAB[lvl])
    a.set_title(ttl); a.set_xlabel('freq (Hz)'); a.set_ylabel(yl); a.grid(alpha=0.3); a.legend(fontsize=8)
for prefix, a, ttl, yl in [('18|d|', ax[1, 0], f'LIFT vs twist (U={W0:.0f}, 3 freqs)', 'lift (N)'),
                           ('18|c|', ax[1, 1], f'THRUST vs twist (U={W0:.0f}, 3 freqs)', 'thrust (N)')]:
    for key in sorted(R.keys()):
        if not key.startswith(prefix) or f'{W0}' not in key: continue
        f = eval(key.split('|')[2])[1]
        x, ex, _ = modline(key, 'base'); a.plot(x, ex, '-o', c='k', ms=5, lw=2.0, alpha=0.5)
        for lvl, ls in [('base', ':'), ('f1', '--'), ('both', '-.')]:
            _, _, m = modline(key, lvl); a.plot(x, m, ls, c=COL[lvl], lw=1.4, alpha=0.8)
    a.plot([], [], '-o', c='k', label='measured'); [a.plot([], [], '-', c=COL[l], label=LAB[l]) for l in LEVELS]
    a.set_title(ttl); a.set_xlabel('twist (deg)'); a.set_ylabel(yl); a.grid(alpha=0.3); a.legend(fontsize=8)
fig.suptitle(f'BEFORE/AFTER vs measured (U={W0:.0f}, nc={NC}): baseline vs +Fix1(geo_stall) vs +Fix1+Fix2(fric)', fontsize=13)
fig.tight_layout(); fig.savefig(os.path.join(DOCS, f'verify_modules_U{W0:.0f}.png'), dpi=110); plt.close(fig)
# per-condition |error| bars
fig, ax = plt.subplots(2, 1, figsize=(16, 9))
for axi, kind, nm in [(ax[0], 'L', 'LIFT'), (ax[1], 'T', 'THRUST')]:
    rows = [(f"{tw:.0f}d/{freq:.1f}Hz", U, aoa, freq, tw, ev) for (key, xi, k, ev, U, aoa, freq, tw) in conds if k == kind]
    labels = [r[0] for r in rows]; x = np.arange(len(rows)); w = 0.27
    for i, lvl in enumerate(LEVELS):
        es = [abs((run(U, aoa, freq, tw, lvl)[0 if kind == 'L' else 1]) - ev) for (_, U, aoa, freq, tw, ev) in rows]
        axi.bar(x + (i - 1) * w, es, w, color=COL[lvl], label=LAB[lvl])
    axi.set_xticks(x); axi.set_xticklabels(labels, rotation=90, fontsize=6); axi.set_ylabel(f'|{nm} error| (N)')
    axi.set_title(f'{nm}: per-condition |model-measured|  (lower=better)'); axi.legend(); axi.grid(alpha=0.3, axis='y')
fig.suptitle(f'Did the modules IMPROVE each condition? per-condition |error| (U={W0:.0f}, nc={NC})', fontsize=13)
fig.tight_layout(); fig.savefig(os.path.join(DOCS, f'verify_modules_err_U{W0:.0f}.png'), dpi=110); plt.close(fig)
print(f"saved docs/verify_modules_U{W0:.0f}.png + verify_modules_err_U{W0:.0f}.png", flush=True); print("DONE", flush=True)
