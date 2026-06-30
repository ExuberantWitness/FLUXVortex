"""Standalone RE-PLOTTER (no GPU): reads docs/repro_data.json (exp/PREV/CURR per condition, produced by the
model-run script) and draws the per-condition comparison figures. Edit the LAYOUT section freely and re-run
instantly -- no model evaluation. Data keys: "<fig>|<sub>|<ident>" where ident = freq (Fig17), wind (Fig18ab),
or "(wind, freq)" (Fig18cd); each value = {x, exp, prev, curr, kind('L'/'T')}.
"""
import json, os, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
DOCS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'docs')
R = json.load(open(os.path.join(DOCS, 'repro_data.json')))

def draw(ax, d, xlab):
    kn = 'Lift' if d['kind'] == 'L' else 'Net thrust'
    nan = lambda L: [v if v is not None else np.nan for v in L]
    ax.plot(d['x'], d['exp'], '-o', color='k', ms=5, lw=2.5, label='exp')
    ax.plot(d['x'], nan(d['curr']), ':^', color='tab:blue', ms=6, mew=1.5, alpha=0.75, label='OLD (lev_merge, −90)')
    if d.get('new') is not None:
        ax.plot(d['x'], nan(d['new']), '-s', color='tab:green', ms=6, lw=2.2, label='NEW (fp-LEV, +90)')
    ax.set_title(f"{kn} vs {xlab.split()[0]}", fontsize=10)
    ax.set_xlabel(xlab, fontsize=9); ax.set_ylabel('force (N)', fontsize=9)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

def twofig(kT, kL, xlab, title, fname):
    if kT not in R or kL not in R:
        print(f"  MISSING {kT} or {kL} -> skip {fname}"); return
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
    draw(ax[0], R[kT], xlab); draw(ax[1], R[kL], xlab)
    fig.suptitle(title, fontsize=11); fig.tight_layout()
    fig.savefig(os.path.join(DOCS, fname), dpi=110); plt.close(fig); print("  saved", fname)

# ----------------------- LAYOUT (edit freely; re-run = instant) -----------------------
if __name__ == '__main__':
    print("available keys:", sorted(R.keys()))
    for f in [1.4, 1.7, 2.0, 2.3, 2.6]:                                   # Fig17 per frequency
        twofig(f"17|a|{f}", f"17|b|{f}", 'twist (deg)',
               f"Fig17  8m/s AoA5  {f}Hz", f"repro_fig17_{int(round(f*10))}.png")
    for U in [6.0, 8.0, 10.0]:                                            # Fig18 a/b per wind
        twofig(f"18|a|{U}", f"18|b|{U}", 'freq (Hz)',
               f"Fig18ab  {U:.0f}m/s AoA5 twist0", f"repro_fig18ab_u{int(U)}.png")
    for U in [6.0, 8.0, 10.0]:                                            # Fig18 c/d per (wind, freq)
        for f in [2.0, 2.3, 2.6]:
            twofig(f"18|c|({U}, {f})", f"18|d|({U}, {f})", 'twist (deg)',
                   f"Fig18cd  {U:.0f}m/s AoA5  {f}Hz", f"repro_fig18cd_u{int(U)}_f{int(round(f*10))}.png")
    print("DONE (pure matplotlib, no GPU)")
