"""P2-S6: flexible vs rigid vs MEASURED overlay from s6_results.json.

Panels (whatever conditions exist in the results file):
  (a) Fig17-style: L and T vs twist amplitude at U=8, f=2.3
  (b) Fig18ab-style: L and T vs frequency at tw=0 (one line per U)
Measured lines from docs/repro_data.json (paper digitized data).

Run: cd FLUXV && python platform/p2_s6_plot.py
"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager, rcParams

for f in ("Noto Sans CJK SC", "WenQuanYi Zen Hei", "AR PL UKai CN", "SimHei"):
    try:
        font_manager.findfont(f, fallback_to_default=False)
        rcParams["font.sans-serif"] = [f]
        break
    except Exception:
        pass
rcParams["axes.unicode_minus"] = False

_HERE = os.path.dirname(os.path.abspath(__file__))
DOCS = os.path.join(_HERE, "docs")


def main():
    res = json.load(open(os.path.join(DOCS, "s6_results.json")))
    meas = json.load(open(os.path.join(DOCS, "repro_data.json")))
    pts = {}
    for k, v in res.items():
        if "flex" not in v:
            continue
        U, f, tw = (float(x) for x in k.split("_"))
        pts[(U, f, tw)] = v

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    # (a) L,T vs twist @ U8 f2.3
    tws = sorted(tw for (U, f, tw) in pts if U == 8.0 and f == 2.3)
    for ax, comp, idx in ((axes[0, 0], "L", 0), (axes[0, 1], "T", 1)):
        if tws:
            ax.plot(tws, [pts[(8.0, 2.3, t)]["flex"][idx] for t in tws],
                    "-o", color="#b3382e", label="柔性(耦合回放)")
            ax.plot(tws, [pts[(8.0, 2.3, t)]["rigid"][idx] for t in tws],
                    "--s", color="#2b2f36", label="刚性(同闭合)")
        mk = f"17|{'a' if comp == 'L' else 'b'}|2.3"
        if mk in meas:
            ax.plot(meas[mk]["x"], meas[mk]["exp"], "k-x", lw=1.2,
                    label="实测(Fig17)")
        ax.set_xlabel("扭转幅值 (deg)"); ax.set_ylabel(f"{comp} (N)")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
        ax.set_title(f"Fig17 型:{comp} vs 扭转 @U8/f2.3", fontsize=10)
    # (b) L,T vs freq @ tw0
    Us = sorted({U for (U, f, tw) in pts if tw == 0.0})
    cols = {6.0: "#3d7a6e", 8.0: "#b3382e", 10.0: "#5a626e"}
    for ax, comp, idx in ((axes[1, 0], "L", 0), (axes[1, 1], "T", 1)):
        for U in Us:
            fs = sorted(f for (UU, f, tw) in pts if UU == U and tw == 0.0)
            if fs:
                ax.plot(fs, [pts[(U, f, 0.0)]["flex"][idx] for f in fs],
                        "-o", color=cols.get(U, "k"), label=f"柔性 U{U:g}")
                ax.plot(fs, [pts[(U, f, 0.0)]["rigid"][idx] for f in fs],
                        "--s", color=cols.get(U, "k"), alpha=0.5,
                        label=f"刚性 U{U:g}")
            mk = f"18|{'a' if comp == 'L' else 'b'}|{U}"
            if mk in meas:
                ax.plot(meas[mk]["x"], meas[mk]["exp"], "-x", color=cols.get(U, "k"),
                        lw=1.0, alpha=0.8)
        ax.set_xlabel("扑动频率 (Hz)"); ax.set_ylabel(f"{comp} (N)")
        ax.legend(fontsize=7); ax.grid(alpha=0.3)
        ax.set_title(f"Fig18 型:{comp} vs 频率 @tw0(×= 实测)", fontsize=10)
    fig.suptitle("P2-S6 柔性翼 Fig17/18 复现(K0 闭合回放)vs 刚性 vs 实测")
    fig.tight_layout()
    out = os.path.join(DOCS, "p2_s6_fig1718.png")
    fig.savefig(out, dpi=150)
    print(f"saved {out} ({len(pts)} conditions)")


if __name__ == "__main__":
    main()
