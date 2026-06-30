"""COMPARISON RE-PLOTTER (no GPU): reads docs/repro_compare.json (exp + per-model predicted lines from
compare_models.py) and overlays all candidate models (M0..M5, ML) on each condition. Instant re-run.
Data keys: "<fig>|<sub>|<ident>"; each value has {x, exp, kind('L'/'T'), models:{M0:[...],...}}."""
import json, os, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
DOCS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'docs')
R = json.load(open(os.path.join(DOCS, 'repro_compare.json')))
# stable color/style per model (winner highlighted in the scorecard, not here)
STYLE = {'M0': ('tab:gray', ':', 'o'), 'M1': ('tab:blue', '-', 's'), 'M2': ('tab:cyan', '--', '^'),
         'M3': ('tab:green', '-', 'D'), 'M4': ('tab:red', '-', 'v'), 'M5': ('tab:purple', '--', 'P'),
         'ML': ('tab:brown', ':', 'x')}
nan = lambda L: [np.nan if v is None else v for v in (L or [])]


def draw(ax, d, xlab):
    kn = 'Lift' if d['kind'] == 'L' else 'Net thrust'
    ax.plot(d['x'], d['exp'], '-o', color='k', ms=6, lw=3.0, label='exp', zorder=10)
    for nm in ['M0', 'M1', 'M2', 'M3', 'M4', 'M5', 'ML']:
        if nm in d.get('models', {}):
            c, ls, mk = STYLE[nm]
            ax.plot(d['x'], nan(d['models'][nm]), ls=ls, marker=mk, color=c, ms=4, lw=1.6, alpha=0.85, label=nm)
    ax.set_title(f"{kn} vs {xlab.split()[0]}", fontsize=10)
    ax.set_xlabel(xlab, fontsize=9); ax.set_ylabel('force (N)', fontsize=9)
    ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.3)


def twofig(kT, kL, xlab, title, fname):
    if kT not in R or kL not in R:
        print(f"  MISSING {kT} or {kL} -> skip {fname}"); return
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    draw(ax[0], R[kT], xlab); draw(ax[1], R[kL], xlab)
    fig.suptitle(title, fontsize=11); fig.tight_layout()
    fig.savefig(os.path.join(DOCS, fname), dpi=110); plt.close(fig); print("  saved", fname)


if __name__ == '__main__':
    print("available keys:", sorted(R.keys()))
    for f in [1.4, 1.7, 2.0, 2.3, 2.6]:                                   # Fig17 per frequency
        twofig(f"17|a|{f}", f"17|b|{f}", 'twist (deg)',
               f"Fig17  8m/s AoA5  {f}Hz  (model comparison)", f"cmp_fig17_{int(round(f*10))}.png")
    for U in [6.0, 8.0, 10.0]:                                            # Fig18 a/b per wind
        twofig(f"18|a|{U}", f"18|b|{U}", 'freq (Hz)',
               f"Fig18ab  {U:.0f}m/s AoA5 twist0  (model comparison)", f"cmp_fig18ab_u{int(U)}.png")
    for U in [6.0, 8.0, 10.0]:                                            # Fig18 c/d per (wind, freq)
        for f in [2.0, 2.3, 2.6]:
            twofig(f"18|c|({U}, {f})", f"18|d|({U}, {f})", 'twist (deg)',
                   f"Fig18cd  {U:.0f}m/s AoA5  {f}Hz  (model comparison)", f"cmp_fig18cd_u{int(U)}_f{int(round(f*10))}.png")
    print("DONE (pure matplotlib, no GPU)")
