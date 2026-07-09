"""STRICT paper-format Fig17/18/19 reproduction: measured curve families +
model points from MULTIPLE replay closures (K0, H16, and les_sep-corrected
when available), flexible (filled) vs rigid (open), with per-point errors and
per-family MAE summaries. Outputs docs/p2_s6_fig17.png / fig18.png / fig19.png.

Measured lines: docs/repro_data.json (paper digitization; kind field routes
L/T). Model: docs/s6_results.json (K0), docs/s6_results_h16.json (H16),
docs/s6_results_lessep.json (les_sep corrected closures, e.g. H16P).

Run: cd FLUXV && python platform/p2_s6_fig_strict.py
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager, rcParams
from matplotlib.lines import Line2D

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
MEAS = json.load(open(os.path.join(DOCS, "repro_data.json")))


def _load(fn, prefix=""):
    p = os.path.join(DOCS, fn)
    pts = {}
    if not os.path.exists(p):
        return pts
    for k, v in json.load(open(p)).items():
        if "flex" not in v:
            continue
        kk = k.split("|")[-1]                     # lessep keys are "CFG|U_f_tw"
        cfg = v.get("cfg", k.split("|")[0] if "|" in k else prefix)
        U, f, tw = (float(x) for x in kk.split("_"))
        pts.setdefault(cfg, {})[(U, f, tw)] = v
    return pts

# FAMILIES[name] = (points, flex marker, rigid marker, annotation dy)
_ALL = {}
_ALL.update(_load("s6_results.json", "K0"))
_ALL.update(_load("s6_results_h16.json", "H16"))
_ALL.update(_load("s6_results_lessep.json"))
FAM_STYLE = {                                     # (marker, ann_offset_y, draw_rigid)
    "K0":   ("o", 6, True),
    "H16":  ("^", -11, True),
    "H16P": ("*", 15, False),
    "H16Z": ("P", 15, False),
    "K0P":  ("X", -20, False),
}
FAMILIES = {k: v for k, v in _ALL.items() if k in FAM_STYLE}

FREQS = [1.4, 1.7, 2.0, 2.3, 2.6]
FCOL = {1.4: "#4477aa", 1.7: "#66ccee", 2.0: "#228833", 2.3: "#cc3311",
        2.6: "#aa3377"}
UCOL = {6.0: "#228833", 8.0: "#cc3311", 10.0: "#4477aa"}
ACOL = {0: "#4477aa", 5: "#cc3311", 10: "#228833", 15: "#aa3377"}
ERRS = {}


def _clear():
    ERRS.clear()


def _mae(idx):
    out = []
    for fam in FAMILIES:
        for kind in ("flex", "rigid"):
            e = ERRS.get((fam, kind, idx))
            if e:
                out.append(f"{fam}{'柔' if kind=='flex' else '刚'} {np.mean(e):.2f}({len(e)})")
    return " / ".join(out) if out else "—"


def meas_at(key, x):
    d = MEAS.get(key)
    if d is None:
        return None
    return float(np.interp(x, d["x"], d["exp"]))


def draw_meas(ax, key, color, label=None):
    d = MEAS.get(key)
    if d is None:
        return
    ax.plot(d["x"], d["exp"], "-x", color=color, lw=1.3, ms=5, alpha=0.9,
            label=label)


def draw_model(ax, x, cond, idx, mkey, color):
    """Plot every closure family's flex (filled) + rigid (open) points."""
    mv = meas_at(mkey, x)
    for fam, pts in FAMILIES.items():
        v = pts.get(cond)
        if v is None:
            continue
        mk, dy, do_rigid = FAM_STYLE[fam]
        fl = v["flex"][idx]
        ax.plot(x, fl, mk, ms=10 if mk == "*" else 8, color=color,
                mec="black", mew=0.8, zorder=6)
        if do_rigid:
            ax.plot(x, v["rigid"][idx], mk, ms=7, mfc="white", mec=color,
                    mew=1.6, zorder=5, alpha=0.8)
        if mv is not None:
            ERRS.setdefault((fam, "flex", idx), []).append(abs(fl - mv))
            if do_rigid:
                ERRS.setdefault((fam, "rigid", idx), []).append(
                    abs(v["rigid"][idx] - mv))
            ax.annotate(f"{fl - mv:+.1f}", (x, fl),
                        textcoords="offset points", xytext=(6, dy),
                        fontsize=7, color=color)


def legend_common(ax):
    h = [Line2D([], [], marker="x", ls="-", color="gray", label="实测(线)")]
    for fam in FAMILIES:
        mk, _, do_rigid = FAM_STYLE[fam]
        h.append(Line2D([], [], marker=mk, ls="", ms=9, mfc="gray",
                        mec="black", label=f"{fam} 柔性(标注=误差N)"))
        if do_rigid:
            h.append(Line2D([], [], marker=mk, ls="", ms=7, mfc="white",
                            mec="gray", mew=1.6, label=f"{fam} 刚性"))
    ax.legend(handles=h + ax.get_legend_handles_labels()[0], fontsize=6.5,
              ncol=2, loc="best")


def fig17():
    _clear()
    fig, (axT, axL) = plt.subplots(1, 2, figsize=(13.5, 5.8))
    for fr in FREQS:
        draw_meas(axT, f"17|a|{fr}", FCOL[fr], f"实测 f={fr}")
        draw_meas(axL, f"17|b|{fr}", FCOL[fr], f"实测 f={fr}")
    for tw in (0.0, 15.0, 22.5, 30.0, 45.0):
        draw_model(axT, tw, (8.0, 2.3, tw), 1, "17|a|2.3", FCOL[2.3])
        draw_model(axL, tw, (8.0, 2.3, tw), 0, "17|b|2.3", FCOL[2.3])
    for ax, ttl in ((axT, "(a) 推力 T vs 扭转幅值"),
                    (axL, "(b) 升力 L vs 扭转幅值")):
        ax.set_xlabel("扭转幅值 (deg)"); ax.set_ylabel("N")
        ax.grid(alpha=0.3); ax.set_title(ttl)
        legend_common(ax)
    fig.suptitle("Fig17 严格复现 — U=8 m/s, AoA=5°(模型点=f2.3 行; tw45 未收敛未画)\n"
                 f"升力 MAE: {_mae(0)}\n推力 MAE: {_mae(1)}", fontsize=9.5)
    fig.tight_layout()
    fig.savefig(os.path.join(DOCS, "p2_s6_fig17.png"), dpi=150)
    plt.close(fig)


def fig18():
    _clear()
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 10.5))
    axT, axL, axCT, axDL = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]
    for U in (6.0, 8.0, 10.0):
        draw_meas(axT, f"18|a|{U}", UCOL[U], f"实测 U={U:g}")
        draw_meas(axL, f"18|b|{U}", UCOL[U], f"实测 U={U:g}")
        for fr in FREQS:
            draw_model(axT, fr, (U, fr, 0.0), 1, f"18|a|{U}", UCOL[U])
            draw_model(axL, fr, (U, fr, 0.0), 0, f"18|b|{U}", UCOL[U])
    for ax, ttl in ((axT, "(a) T vs 频率 @tw0"), (axL, "(b) L vs 频率 @tw0")):
        ax.set_xlabel("扑动频率 (Hz)"); ax.set_ylabel("N")
        ax.grid(alpha=0.3); ax.set_title(ttl)
        legend_common(ax)
    draw_meas(axCT, "18|c|(8.0, 2.3)", UCOL[8.0], "实测 (U8,f2.3)")
    draw_meas(axDL, "18|d|(8.0, 2.3)", UCOL[8.0], "实测 (U8,f2.3)")
    for tw in (0.0, 15.0, 22.5, 30.0, 45.0):
        draw_model(axCT, tw, (8.0, 2.3, tw), 1, "18|c|(8.0, 2.3)", UCOL[8.0])
        draw_model(axDL, tw, (8.0, 2.3, tw), 0, "18|d|(8.0, 2.3)", UCOL[8.0])
    for ax, ttl in ((axCT, "(c) T vs 扭转 @(U8,f2.3)"),
                    (axDL, "(d) L vs 扭转 @(U8,f2.3)")):
        ax.set_xlabel("扭转幅值 (deg)"); ax.set_ylabel("N")
        ax.grid(alpha=0.3); ax.set_title(ttl)
        legend_common(ax)
    fig.suptitle("Fig18 严格复现 — AoA=5°(U10 未收敛未画)\n"
                 f"升力 MAE: {_mae(0)}\n推力 MAE: {_mae(1)}", fontsize=9.5)
    fig.tight_layout()
    fig.savefig(os.path.join(DOCS, "p2_s6_fig18.png"), dpi=150)
    plt.close(fig)


def fig19():
    _clear()
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 10.5))
    axT, axL, axCT, axDL = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]
    for a in (0, 5, 10, 15):
        draw_meas(axT, f"19|a|{a}", ACOL[a], f"实测 AoA={a}")
        draw_meas(axL, f"19|b|{a}", ACOL[a], f"实测 AoA={a}")
    for fr in FREQS:                               # model exists only at AoA=5
        draw_model(axT, fr, (8.0, fr, 0.0), 1, "19|a|5", ACOL[5])
        draw_model(axL, fr, (8.0, fr, 0.0), 0, "19|b|5", ACOL[5])
    for ax, ttl in ((axT, "(a) T vs 频率(各攻角)"),
                    (axL, "(b) L vs 频率(各攻角)")):
        ax.set_xlabel("扑动频率 (Hz)"); ax.set_ylabel("N")
        ax.grid(alpha=0.3); ax.set_title(ttl)
        legend_common(ax)
    for a in (0, 5, 10, 15):
        draw_meas(axCT, f"19|c|{a}", ACOL[a], f"实测 AoA={a}")
        draw_meas(axDL, f"19|d|{a}", ACOL[a], f"实测 AoA={a}")
    for ax, ttl in ((axCT, "(c) T vs 扭转 @f2.6(模型工况未算)"),
                    (axDL, "(d) L vs 扭转 @f2.6(模型工况未算)")):
        ax.set_xlabel("扭转幅值 (deg)"); ax.set_ylabel("N")
        ax.grid(alpha=0.3); ax.set_title(ttl)
        ax.legend(fontsize=7)
    fig.suptitle("Fig19 严格复现 — 模型仅 AoA=5° 线(其余攻角/f2.6 扭转行未算,已拍板搁置)\n"
                 f"升力 MAE: {_mae(0)}\n推力 MAE: {_mae(1)}", fontsize=9.5)
    fig.tight_layout()
    fig.savefig(os.path.join(DOCS, "p2_s6_fig19.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    print("families:", {k: len(v) for k, v in FAMILIES.items()})
    fig17(); fig18(); fig19()
    print("saved p2_s6_fig17/18/19.png (分族 MAE 在各图标题)")
