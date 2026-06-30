"""OVERVIEW re-plotter (no GPU): regenerates the all-in-one docs/repro_fig17.png & repro_fig18.png
from docs/repro_compare.json with the NEW candidate models. exp = solid -o, winner M4 = dashed,
Hirato M1 = dotted; one color per frequency/wind. Overwrites the stale (prev-session) overview files."""
import json, os, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
DOCS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'docs')
R = json.load(open(os.path.join(DOCS, 'repro_compare.json')))
nan = lambda L: [np.nan if v is None else v for v in (L or [])]
CMAP = plt.cm.viridis


def line(ax, key, color, lab_prefix, show_models=('M4', 'M1')):
    if key not in R: return
    d = R[key]; x = d['x']
    ax.plot(x, d['exp'], '-o', color=color, ms=5, lw=2.2, label=f"exp {lab_prefix}")
    sty = {'M4': ('--', 'x'), 'M1': (':', '^'), 'M0': (':', '.'), 'M3': ('-.', 'd'), 'ML': ('--', '+')}
    for nm in show_models:
        if nm in d.get('models', {}):
            ls, mk = sty.get(nm, ('--', '.'))
            ax.plot(x, nan(d['models'][nm]), ls=ls, marker=mk, color=color, ms=4, lw=1.4, alpha=0.8,
                    label=f"{nm} {lab_prefix}")


# ---------------- Fig 17: thrust(a) + lift(b) vs twist, all 5 frequencies overlaid ----------------
FREQS = [1.4, 1.7, 2.0, 2.3, 2.6]
fig, ax = plt.subplots(1, 2, figsize=(15, 6))
for i, f in enumerate(FREQS):
    c = CMAP(i / (len(FREQS) - 1))
    line(ax[0], f"17|a|{f}", c, f"{f}Hz"); line(ax[1], f"17|b|{f}", c, f"{f}Hz")
ax[0].set_title("Fig17a  Net thrust vs twist  [data=T]"); ax[1].set_title("Fig17b  Lift vs twist  [data=L]")
for a in ax:
    a.set_xlabel("twist (deg)"); a.set_ylabel("force (N)"); a.grid(alpha=0.3); a.legend(fontsize=6, ncol=3)
fig.suptitle("Fig17  exp(-o)  vs  M4 hold_detach(--x)  vs  M1 Hirato(:^)   [8m/s AoA5, per frequency]", fontsize=12)
fig.tight_layout(); fig.savefig(os.path.join(DOCS, "repro_fig17.png"), dpi=110); plt.close(fig)
print("saved repro_fig17.png")

# ---------------- Fig 18ab: thrust(a) + lift(b) vs frequency, 3 winds overlaid ----------------
WINDS = [6.0, 8.0, 10.0]
fig, ax = plt.subplots(1, 2, figsize=(15, 6))
for i, U in enumerate(WINDS):
    c = CMAP(i / (len(WINDS) - 1))
    line(ax[0], f"18|a|{U}", c, f"{U:.0f}m/s"); line(ax[1], f"18|b|{U}", c, f"{U:.0f}m/s")
ax[0].set_title("Fig18a  Net thrust vs freq  [data=T]"); ax[1].set_title("Fig18b  Lift vs freq  [data=L]")
for a in ax:
    a.set_xlabel("freq (Hz)"); a.set_ylabel("force (N)"); a.grid(alpha=0.3); a.legend(fontsize=7, ncol=3)
fig.suptitle("Fig18ab  exp(-o)  vs  M4 hold_detach(--x)  vs  M1 Hirato(:^)   [twist0, per wind]", fontsize=12)
fig.tight_layout(); fig.savefig(os.path.join(DOCS, "repro_fig18.png"), dpi=110); plt.close(fig)
print("saved repro_fig18.png")
print("DONE")
