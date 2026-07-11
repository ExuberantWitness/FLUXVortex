"""Fig17/18/19 — 强耦合 FSI 对比版:柔性(耦合回放,GPU 结构求解录音)vs 刚性
(同 H16KT 闭合、刚性运动学),连线 + δ(柔-刚)标注,实测为参照线。
Run: cd FLUXV && python platform/p2_s6_fig_fsi.py
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
CFG = "H16KTv"

PTS = {}
for k, v in json.load(open(os.path.join(DOCS, "s6_results_lessep.json"))).items():
    if v.get("cfg") == CFG and "flex" in v:
        U, f, tw = (float(x) for x in k.split("|")[-1].split("_"))
        PTS[(U, f, tw)] = v

FREQS = [1.4, 1.7, 2.0, 2.3, 2.6]
FCOL = {1.4: "#4477aa", 1.7: "#66ccee", 2.0: "#228833", 2.3: "#cc3311",
        2.6: "#aa3377"}
UCOL = {6.0: "#228833", 8.0: "#cc3311", 10.0: "#4477aa"}
ACOL = {0: "#4477aa", 5: "#cc3311", 10: "#228833", 15: "#aa3377"}
DELTAS = {}


def _clear():
    DELTAS.clear()


def meas_at(key, x):
    d = MEAS.get(key)
    return None if d is None else float(np.interp(x, d["x"], d["exp"]))


def draw_meas(ax, key, color, label=None):
    d = MEAS.get(key)
    if d is None:
        return
    ax.plot(d["x"], d["exp"], "-x", color=color, lw=1.2, ms=4, alpha=0.75,
            label=label)


def draw_pair(ax, x, cond, idx, mkey, color):
    v = PTS.get(cond)
    if v is None:
        return
    fl, rg = v["flex"][idx], v["rigid"][idx]
    ax.plot([x, x], [rg, fl], "-", color="gray", lw=1.0, alpha=0.8, zorder=4)
    ax.plot(x, rg, "s", ms=7, mfc="white", mec=color, mew=1.6, zorder=5)
    ax.plot(x, fl, "o", ms=9, color=color, mec="black", mew=0.8, zorder=6)
    d = fl - rg
    DELTAS.setdefault(idx, []).append(d)
    mv = meas_at(mkey, x)
    to_meas = "" if mv is None else ("→实测" if (mv - rg) * d > 0 else "←实测")
    ax.annotate(f"δ{d:+.2f}{to_meas}", (x, 0.5 * (fl + rg)),
                textcoords="offset points", xytext=(7, 0), fontsize=7,
                color="dimgray")


def legend_common(ax):
    h = [Line2D([], [], marker="o", ls="", ms=9, mfc="gray", mec="black",
                label="柔性 = 强耦合 FSI(GPU 结构求解)"),
         Line2D([], [], marker="s", ls="", ms=7, mfc="white", mec="gray",
                mew=1.6, label="刚性(同闭合,刚性运动学)"),
         Line2D([], [], ls="-", color="gray", label="δ = 柔-刚(耦合效应)"),
         Line2D([], [], marker="x", ls="-", color="gray", label="实测(线)")]
    ax.legend(handles=h + ax.get_legend_handles_labels()[0], fontsize=6.5,
              ncol=2, loc="best")


def _dsum(idx):
    d = np.array(DELTAS.get(idx, []))
    if not len(d):
        return "—"
    return f"均值 {d.mean():+.2f} / 幅 {np.abs(d).max():.2f} N({len(d)}点)"


def fig17():
    _clear()
    fig, (axT, axL) = plt.subplots(1, 2, figsize=(13, 5.5))
    for fr in FREQS:
        draw_meas(axT, f"17|a|{fr}", FCOL[fr], f"实测 f={fr}")
        draw_meas(axL, f"17|b|{fr}", FCOL[fr], f"实测 f={fr}")
    for tw in (0.0, 15.0, 22.5, 30.0):
        draw_pair(axT, tw, (8.0, 2.3, tw), 1, "17|a|2.3", FCOL[2.3])
        draw_pair(axL, tw, (8.0, 2.3, tw), 0, "17|b|2.3", FCOL[2.3])
    for ax, ttl in ((axT, "(a) 推力 T vs 扭转幅值"),
                    (axL, "(b) 升力 L vs 扭转幅值")):
        ax.set_xlabel("扭转幅值 (deg)"); ax.set_ylabel("N")
        ax.grid(alpha=0.3); ax.set_title(ttl)
        legend_common(ax)
    fig.suptitle(f"Fig17 柔性/刚性耦合对比 — U=8, AoA=5°, f2.3 行 [{CFG}]\n"
                 f"耦合效应 δ:升力 {_dsum(0)}   |   推力 {_dsum(1)}")
    fig.tight_layout()
    fig.savefig(os.path.join(DOCS, "p2_s6_fig17_fsi.png"), dpi=150)
    plt.close(fig)


def fig18():
    _clear()
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    axT, axL, axCT, axDL = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]
    for U in (6.0, 8.0, 10.0):
        draw_meas(axT, f"18|a|{U}", UCOL[U], f"实测 U={U:g}")
        draw_meas(axL, f"18|b|{U}", UCOL[U], f"实测 U={U:g}")
        for fr in FREQS:
            draw_pair(axT, fr, (U, fr, 0.0), 1, f"18|a|{U}", UCOL[U])
            draw_pair(axL, fr, (U, fr, 0.0), 0, f"18|b|{U}", UCOL[U])
    for ax, ttl in ((axT, "(a) T vs 频率 @tw0"), (axL, "(b) L vs 频率 @tw0")):
        ax.set_xlabel("扑动频率 (Hz)"); ax.set_ylabel("N")
        ax.grid(alpha=0.3); ax.set_title(ttl)
        legend_common(ax)
    draw_meas(axCT, "18|c|(8.0, 2.3)", UCOL[8.0], "实测 (U8,f2.3)")
    draw_meas(axDL, "18|d|(8.0, 2.3)", UCOL[8.0], "实测 (U8,f2.3)")
    for tw in (0.0, 15.0, 22.5, 30.0):
        draw_pair(axCT, tw, (8.0, 2.3, tw), 1, "18|c|(8.0, 2.3)", UCOL[8.0])
        draw_pair(axDL, tw, (8.0, 2.3, tw), 0, "18|d|(8.0, 2.3)", UCOL[8.0])
    for ax, ttl in ((axCT, "(c) T vs 扭转 @(U8,f2.3)"),
                    (axDL, "(d) L vs 扭转 @(U8,f2.3)")):
        ax.set_xlabel("扭转幅值 (deg)"); ax.set_ylabel("N")
        ax.grid(alpha=0.3); ax.set_title(ttl)
        legend_common(ax)
    fig.suptitle(f"Fig18 柔性/刚性耦合对比 — AoA=5° [{CFG}](U10 录音失败未画)\n"
                 f"耦合效应 δ:升力 {_dsum(0)}   |   推力 {_dsum(1)}")
    fig.tight_layout()
    fig.savefig(os.path.join(DOCS, "p2_s6_fig18_fsi.png"), dpi=150)
    plt.close(fig)


def fig19():
    _clear()
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    axT, axL, axCT, axDL = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]
    for a in (0, 5, 10, 15):
        draw_meas(axT, f"19|a|{a}", ACOL[a], f"实测 AoA={a}")
        draw_meas(axL, f"19|b|{a}", ACOL[a], f"实测 AoA={a}")
    for fr in FREQS:
        draw_pair(axT, fr, (8.0, fr, 0.0), 1, "19|a|5", ACOL[5])
        draw_pair(axL, fr, (8.0, fr, 0.0), 0, "19|b|5", ACOL[5])
    for ax, ttl in ((axT, "(a) T vs 频率(各攻角)"),
                    (axL, "(b) L vs 频率(各攻角)")):
        ax.set_xlabel("扑动频率 (Hz)"); ax.set_ylabel("N")
        ax.grid(alpha=0.3); ax.set_title(ttl)
        legend_common(ax)
    for a in (0, 5, 10, 15):
        draw_meas(axCT, f"19|c|{a}", ACOL[a], f"实测 AoA={a}")
        draw_meas(axDL, f"19|d|{a}", ACOL[a], f"实测 AoA={a}")
    for tw in (0.0, 15.0, 22.5, 30.0):
        draw_pair(axCT, tw, (8.0, 2.6, tw), 1, "19|c|5", ACOL[5])
        draw_pair(axDL, tw, (8.0, 2.6, tw), 0, "19|d|5", ACOL[5])
    for ax, ttl in ((axCT, "(c) T vs 扭转 @f2.6"),
                    (axDL, "(d) L vs 扭转 @f2.6")):
        ax.set_xlabel("扭转幅值 (deg)"); ax.set_ylabel("N")
        ax.grid(alpha=0.3); ax.set_title(ttl)
        legend_common(ax)
    fig.suptitle(f"Fig19 柔性/刚性耦合对比 — 模型=AoA5° 线 [{CFG}]"
                 "(f2.6 tw45 录音失败;AoA 0/10/15 行搁置)\n"
                 f"耦合效应 δ:升力 {_dsum(0)}   |   推力 {_dsum(1)}")
    fig.tight_layout()
    fig.savefig(os.path.join(DOCS, "p2_s6_fig19_fsi.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    print(f"{CFG} pairs: {len(PTS)}")
    fig17(); fig18(); fig19()
    print("saved p2_s6_fig17/18/19_fsi.png")
