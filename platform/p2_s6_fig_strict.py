"""STRICT paper-format Fig17/18/19 reproduction: measured curve families +
flexible (coupled replay) + rigid (same closure) model points, with per-point
errors and MAE summaries. Outputs docs/p2_s6_fig17.png / fig18.png / fig19.png.

Measured lines: docs/repro_data.json (paper digitization; kind field routes
L/T). Model: docs/s6_results.json (K0 closure replay; flexible = coupled
deformation replay, rigid = same production closure on rigid kinematics).

Run: cd FLUXV && python platform/p2_s6_fig_strict.py
"""
import json
import os
import sys

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
RES = json.load(open(os.path.join(DOCS, "s6_results.json")))

PTS = {}
for k, v in RES.items():
    if "flex" in v:
        U, f, tw = (float(x) for x in k.split("_"))
        PTS[(U, f, tw)] = v

FREQS = [1.4, 1.7, 2.0, 2.3, 2.6]
FCOL = {1.4: "#4477aa", 1.7: "#66ccee", 2.0: "#228833", 2.3: "#cc3311",
        2.6: "#aa3377"}
UCOL = {6.0: "#228833", 8.0: "#cc3311", 10.0: "#4477aa"}
ACOL = {0: "#4477aa", 5: "#cc3311", 10: "#228833", 15: "#aa3377"}
ERRS = {("flex", 0): [], ("rigid", 0): [], ("flex", 1): [], ("rigid", 1): []}


def _clear():
    for v in ERRS.values():
        v.clear()


def _mae(idx):
    f_ = ERRS[("flex", idx)]; r_ = ERRS[("rigid", idx)]
    return (np.mean(f_) if f_ else float("nan"),
            np.mean(r_) if r_ else float("nan"), len(f_))


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
    """Plot flex (filled) + rigid (open) model points; collect errors."""
    v = PTS.get(cond)
    if v is None:
        return
    mv = meas_at(mkey, x)
    fl, rg = v["flex"][idx], v["rigid"][idx]
    ax.plot(x, fl, "o", ms=9, color=color, mec="black", mew=0.8, zorder=6)
    ax.plot(x, rg, "s", ms=8, mfc="white", mec=color, mew=1.8, zorder=5)
    if mv is not None:
        ERRS[("flex", idx)].append(abs(fl - mv))
        ERRS[("rigid", idx)].append(abs(rg - mv))
        ax.annotate(f"{fl - mv:+.1f}", (x, fl), textcoords="offset points",
                    xytext=(6, 6), fontsize=7.5, color=color)


def legend_common(ax):
    h = [Line2D([], [], marker="o", ls="", ms=9, mfc="gray", mec="black",
                label="柔性(耦合回放,标注=误差N)"),
         Line2D([], [], marker="s", ls="", ms=8, mfc="white", mec="gray",
                mew=1.8, label="刚性(同闭合)"),
         Line2D([], [], marker="x", ls="-", color="gray", label="实测(线)")]
    ax.legend(handles=h + ax.get_legend_handles_labels()[0], fontsize=7,
              ncol=2, loc="best")


def fig17():
    _clear()
    fig, (axT, axL) = plt.subplots(1, 2, figsize=(13, 5.5))
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
    mLf, mLr, nL = _mae(0); mTf, mTr, nT = _mae(1)
    fig.suptitle(f"Fig17 严格复现 — U=8 m/s, AoA=5° (模型点=f2.3 行; tw45 未收敛未画) [K0 闭合回放]\n"
                 f"升力 MAE({nL}点): 柔性 {mLf:.2f} / 刚性 {mLr:.2f} N   |   "
                 f"推力 MAE({nT}点): 柔性 {mTf:.2f} / 刚性 {mTr:.2f} N(K0 推力口径已知偏差)")
    fig.tight_layout()
    fig.savefig(os.path.join(DOCS, "p2_s6_fig17.png"), dpi=150)
    plt.close(fig)


def fig18():
    _clear()
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
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
    # (c,d): vs twist for the (8, 2.3) pair (the only coupled twist row)
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
    mLf, mLr, nL = _mae(0); mTf, mTr, nT = _mae(1)
    fig.suptitle(f"Fig18 严格复现 — AoA=5°(U10 未收敛未画) [K0 闭合回放]\n"
                 f"升力 MAE({nL}点): 柔性 {mLf:.2f} / 刚性 {mLr:.2f} N   |   "
                 f"推力 MAE({nT}点): 柔性 {mTf:.2f} / 刚性 {mTr:.2f} N(K0 推力口径已知偏差)")
    fig.tight_layout()
    fig.savefig(os.path.join(DOCS, "p2_s6_fig18.png"), dpi=150)
    plt.close(fig)


def fig19():
    _clear()
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
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
    mLf, mLr, nL = _mae(0); mTf, mTr, nT = _mae(1)
    fig.suptitle(f"Fig19 严格复现 — 模型仅 AoA=5° 线(其余攻角/f2.6 扭转行未算,已拍板搁置) [K0 闭合回放]\n"
                 f"升力 MAE({nL}点): 柔性 {mLf:.2f} / 刚性 {mLr:.2f} N   |   "
                 f"推力 MAE({nT}点): 柔性 {mTf:.2f} / 刚性 {mTr:.2f} N(K0 推力口径已知偏差)")
    fig.tight_layout()
    fig.savefig(os.path.join(DOCS, "p2_s6_fig19.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    fig17(); fig18(); fig19()
    print("saved p2_s6_fig17/18/19.png (L/T MAE 分列在各图标题)")
