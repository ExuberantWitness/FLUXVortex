"""Verify the two additive corrections (geo_stall + fric_drag) on the U=10 conditions at nc=4 (grid-equivalent
to nc12, fast). Runs baseline vs +fixes, scores lift/thrust MAE vs measured, redraws the 4-panel U=10 figure.
Physics-anchored constants (NACA-2406 alpha_ss=12deg, separation width 16deg, turbulent Cf) — NO RoboEagle fitting.

  python _v2_verify_fix.py [nc] [winds]    # default nc=4, winds=10
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
WINDS = [float(x) for x in sys.argv[2].split(',')] if len(sys.argv) > 2 else [10.0]
SPC = 60 * NC; BASE = {k: v for k, v in MODEL.items() if k != 'n_cycle'}
FIX = dict(geo_stall=True, geo_stall_deg=12.0, geo_stall_width=16.0)   # Fix1 only (lift); friction dropped (wrong mechanism)

R = json.load(open(os.path.join(DOCS, 'repro_data.json')))
cache = {}
def run(U, aoa, freq, tw, fix):
    key = (round(U, 1), round(aoa, 1), round(freq, 3), round(tw, 3), fix)
    if key not in cache:
        kw = dict(FIX) if fix else {}
        r = gpu_run_twist(U=U, aoa_deg=aoa, freq=freq, twist_amp_deg=tw, twist_phase_deg=90.0,
                          nc=NC, ns=16, n_cycle=3, steps_per_cycle=SPC, wake_rows=SPC, **BASE, **kw)
        cache[key] = (float(r['L_wind']), float(r['T_wind']))
    return cache[key]

# score MAE on all matched U=10 points, baseline vs fixed
ae = {'base': {'L': [], 'T': []}, 'fix': {'L': [], 'T': []}}
for key in sorted(R.keys()):
    e = R[key]; kind = e['kind']
    for xi, ev in zip(e['x'], e['exp']):
        U, aoa, freq, tw = rn.cond_of(key, xi)
        if U not in WINDS or ev != ev: continue
        for tag, fix in [('base', False), ('fix', True)]:
            L, T = run(U, aoa, freq, tw, fix); mv = L if kind == 'L' else T
            ae[tag][kind].append(abs(mv - ev))
print(f"=== MAE (nc={NC}, U={WINDS}) baseline -> +fixes ===")
for k in ('L', 'T'):
    b = np.mean(ae['base'][k]); f = np.mean(ae['fix'][k])
    print(f"  {k}: {b:.2f}N -> {f:.2f}N  ({len(ae['fix'][k])} pts)", flush=True)

# redraw the 4-panel U=10 figure (measured -o, baseline --., +fixes --x)
def modline(key, fix):
    e = R[key]; kind = e['kind']; out = []
    for xi in e['x']:
        U, aoa, freq, tw = rn.cond_of(key, xi)
        if U not in WINDS: out.append(np.nan); continue
        L, T = run(U, aoa, freq, tw, fix); out.append(L if kind == 'L' else T)
    return e['x'], e['exp'], out
fig, ax = plt.subplots(2, 2, figsize=(15, 10)); CM = plt.cm.viridis
W0 = WINDS[0]
for key, a, ttl, yl in [(f'18|b|{W0}', ax[0, 0], f'LIFT vs freq (U={W0:.0f}, twist0)', 'lift (N)'),
                        (f'18|a|{W0}', ax[0, 1], f'THRUST vs freq (U={W0:.0f}, twist0)', 'thrust (N)')]:
    if key not in R: continue
    x, ex, mb = modline(key, False); _, _, mf = modline(key, True)
    a.plot(x, ex, '-o', c='tab:blue', ms=6, lw=2.3, label='measured')
    a.plot(x, mb, '--.', c='tab:gray', ms=8, lw=1.4, label='baseline')
    a.plot(x, mf, '--x', c='tab:red', ms=8, lw=1.8, label='+fixes')
    a.set_title(ttl); a.set_xlabel('freq (Hz)'); a.set_ylabel(yl); a.grid(alpha=0.3); a.legend()
for prefix, a, ttl, yl in [('18|d|', ax[1, 0], f'LIFT vs twist (U={W0:.0f})', 'lift (N)'),
                           ('18|c|', ax[1, 1], f'THRUST vs twist (U={W0:.0f})', 'thrust (N)')]:
    i = 0
    for key in sorted(R.keys()):
        if not key.startswith(prefix) or f'{W0}' not in key: continue
        f = eval(key.split('|')[2])[1]; c = CM(i / 2.5); i += 1
        x, ex, mb = modline(key, False); _, _, mf = modline(key, True)
        a.plot(x, ex, '-o', c=c, ms=6, lw=2.3, label=f'meas {f}Hz')
        a.plot(x, mf, '--x', c=c, ms=7, lw=1.8, alpha=0.85, label=f'+fix {f}Hz')
    a.set_title(ttl); a.set_xlabel('twist (deg)'); a.set_ylabel(yl); a.grid(alpha=0.3); a.legend(fontsize=8, ncol=3)
fig.suptitle(f'Fix verify @nc{NC} U={W0:.0f}: geo_stall(12deg/16deg) + fric_drag(turbulent) — measured(-o) baseline(--.) +fixes(--x)', fontsize=12)
fig.tight_layout(); fig.savefig(os.path.join(DOCS, f'verify_fix_U{W0:.0f}.png'), dpi=110)
print(f"saved docs/verify_fix_U{W0:.0f}.png", flush=True); print("DONE", flush=True)
