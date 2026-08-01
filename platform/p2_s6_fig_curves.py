"""全工况逐条曲线对比:Fig17/18/19 每条实测曲线配同色模型曲线(H16pltL2 刚性全扫
s6_sweep_v2.json),柔性回放点(H16pltL2 20 点)叠加为大圆点。每条曲线标注工况与
MAE;缺失/发散点如实断线。输出 p2_s6_fig17_curves.png 等 + 逐曲线 MAE 表(stdout)。
运动学口径:扑±22.5°,tw 标签=峰-峰(kinematics_audit.md);Fig18/19 频率线=tw22.5。"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager, rcParams
from matplotlib.lines import Line2D

for fname in ("Noto Sans CJK SC", "WenQuanYi Zen Hei", "SimHei"):
    try:
        font_manager.findfont(fname, fallback_to_default=False)
        rcParams["font.sans-serif"] = [fname]
        break
    except Exception:
        pass
rcParams["axes.unicode_minus"] = False

_HERE = os.path.dirname(os.path.abspath(__file__))
DOCS = os.path.join(_HERE, "docs")
MEAS = json.load(open(os.path.join(DOCS, "repro_data.json")))
SW = {}
_p = os.path.join(DOCS, "s6_sweep_v2.json")
if os.path.exists(_p):
    for k, v in json.load(open(_p)).items():
        if "L" in v:
            SW[(v["U"], v["f"], v["tw"], v["aoa"])] = (v["L"], v["T"])
FLEX = {}
for k, v in json.load(open(os.path.join(DOCS, "s6_results_lessep.json"))).items():
    if k.startswith("H16pltL2|") and "flex" in v and abs(v["flex"][0]) < 100:
        parts = k.split("|")[-1].split("_")
        aoa = 5.0
        if parts[-1].startswith("aoa"):
            aoa = float(parts.pop()[3:])
        U, f, tw = (float(x) for x in parts)
        FLEX[(U, f, tw, aoa)] = v["flex"]

TWS = [0.0, 5.0, 10.0, 15.0, 20.0, 22.5, 25.0, 27.5, 30.0, 35.0, 40.0, 45.0]
FS = [1.4, 1.7, 2.0, 2.3, 2.6]
FCOL = {1.4: "#4477aa", 1.7: "#66ccee", 2.0: "#228833", 2.3: "#cc3311", 2.6: "#aa3377"}
UCOL = {6.0: "#228833", 8.0: "#cc3311", 10.0: "#4477aa"}
ACOL = {0: "#4477aa", 5: "#cc3311", 10: "#228833", 15: "#aa3377"}
MAE_LINES = []


def meas_curve(key):
    d = MEAS.get(key)
    return (None, None) if d is None else (np.asarray(d["x"]), np.asarray(d["exp"]))


def model_curve(xs, cond_of, idx):
    out_x, out_y = [], []
    for x in xs:
        c = cond_of(x)
        if c in SW:
            out_x.append(x)
            out_y.append(SW[c][idx])
    return np.asarray(out_x), np.asarray(out_y)


def draw_pair(ax, mkey, xs, cond_of, idx, color, tag):
    mx, my = meas_curve(mkey)
    if mx is None:
        return
    ax.plot(mx, my, "-x", color=color, lw=1.4, ms=4, alpha=0.85)
    sx, sy = model_curve(xs, cond_of, idx)
    if len(sx):
        ax.plot(sx, sy, "--o", color=color, lw=1.6, ms=5, mfc="white", mew=1.4)
        mi = np.interp(sx, mx, my)
        mae = float(np.mean(np.abs(sy - mi)))
        MAE_LINES.append(f"{tag:>26}: 模型 {len(sx)} 点, MAE {mae:.2f} N")
    for x in xs:
        c = cond_of(x)
        if c in FLEX:
            ax.plot(x, FLEX[c][idx], "o", ms=9, color=color, mec="black", mew=0.9,
                    zorder=6)


def style(ax, xlabel, title):
    ax.set_xlabel(xlabel)
    ax.set_ylabel("N")
    ax.grid(alpha=0.3)
    ax.set_title(title)


def legend_kinds(ax):
    h = [Line2D([], [], ls="-", marker="x", color="gray", label="实测"),
         Line2D([], [], ls="--", marker="o", mfc="white", color="gray",
                label="模型(刚性,同色同工况)"),
         Line2D([], [], ls="", marker="o", ms=9, color="gray", mec="black",
                label="模型(柔性 FSI 回放)")]
    ax.legend(handles=h + ax.get_legend_handles_labels()[0], fontsize=7, ncol=2)


def fig17():
    fig, (axT, axL) = plt.subplots(1, 2, figsize=(14, 5.8))
    for f in FS:
        cond = lambda tw, f=f: (8.0, f, tw, 5.0)
        draw_pair(axT, f"17|a|{f}", TWS, cond, 1, FCOL[f], f"Fig17a 推力 f={f}")
        draw_pair(axL, f"17|b|{f}", TWS, cond, 0, FCOL[f], f"Fig17b 升力 f={f}")
        axT.plot([], [], "-", color=FCOL[f], label=f"f={f} Hz")
    style(axT, "扭转幅值标签 (deg, 峰-峰)", "(a) 推力 T vs 扭转 — U8, AoA5°, 5 条频率线")
    style(axL, "扭转幅值标签 (deg, 峰-峰)", "(b) 升力 L vs 扭转 — U8, AoA5°, 5 条频率线")
    legend_kinds(axT)
    axL.legend(handles=[Line2D([], [], color=FCOL[f], label=f"f={f} Hz") for f in FS],
               fontsize=7)
    fig.suptitle("Fig17 全工况逐曲线对比 [H16pltL2, 正确运动学口径] "
                 "(实线=实测, 虚线空心=模型刚性, 实心大点=模型柔性)")
    fig.tight_layout()
    fig.savefig(os.path.join(DOCS, "p2_s6_fig17_curves.png"), dpi=150)
    plt.close(fig)


def fig18():
    fig, (axT, axL) = plt.subplots(1, 2, figsize=(14, 5.8))
    for U in (6.0, 8.0, 10.0):
        cond = lambda f, U=U: (U, f, 22.5, 5.0)
        draw_pair(axT, f"18|a|{U}", FS, cond, 1, UCOL[U], f"Fig18a 推力 U={U:g}")
        draw_pair(axL, f"18|b|{U}", FS, cond, 0, UCOL[U], f"Fig18b 升力 U={U:g}")
        axT.plot([], [], "-", color=UCOL[U], label=f"U={U:g} m/s")
    style(axT, "扑动频率 (Hz)", "(a) T vs 频率 — tw22.5, AoA5°, 3 条速度线")
    style(axL, "扑动频率 (Hz)", "(b) L vs 频率 — tw22.5, AoA5°, 3 条速度线")
    legend_kinds(axT)
    axL.legend(handles=[Line2D([], [], color=UCOL[U], label=f"U={U:g}") for U in
                        (6.0, 8.0, 10.0)], fontsize=7)
    fig.suptitle("Fig18(a,b) 全工况逐曲线对比 [H16pltL2] (c,d 面板=Fig17 的 f2.3 线)")
    fig.tight_layout()
    fig.savefig(os.path.join(DOCS, "p2_s6_fig18_curves.png"), dpi=150)
    plt.close(fig)


def fig19():
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    (axT, axL), (axCT, axDL) = axes
    for a in (0, 5, 10, 15):
        cond = lambda f, a=a: (8.0, f, 22.5, float(a))
        draw_pair(axT, f"19|a|{a}", FS, cond, 1, ACOL[a], f"Fig19a 推力 AoA={a}")
        draw_pair(axL, f"19|b|{a}", FS, cond, 0, ACOL[a], f"Fig19b 升力 AoA={a}")
        cond2 = lambda tw, a=a: (8.0, 2.6, tw, float(a))
        draw_pair(axCT, f"19|c|{a}", TWS, cond2, 1, ACOL[a], f"Fig19c 推力 AoA={a}")
        draw_pair(axDL, f"19|d|{a}", TWS, cond2, 0, ACOL[a], f"Fig19d 升力 AoA={a}")
        axT.plot([], [], "-", color=ACOL[a], label=f"AoA={a}°")
    style(axT, "扑动频率 (Hz)", "(a) T vs 频率 — U8, tw22.5, 4 条攻角线")
    style(axL, "扑动频率 (Hz)", "(b) L vs 频率 — U8, tw22.5, 4 条攻角线")
    style(axCT, "扭转幅值标签 (deg)", "(c) T vs 扭转 — U8, f2.6(原文标注存疑)")
    style(axDL, "扭转幅值标签 (deg)", "(d) L vs 扭转 — U8, f2.6(原文标注存疑)")
    legend_kinds(axT)
    axL.legend(handles=[Line2D([], [], color=ACOL[a], label=f"AoA={a}°") for a in
                        (0, 5, 10, 15)], fontsize=7)
    fig.suptitle("Fig19 全工况逐曲线对比 [H16pltL2]")
    fig.tight_layout()
    fig.savefig(os.path.join(DOCS, "p2_s6_fig19_curves.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    print(f"扫描点: {len(SW)} | 柔性点: {len(FLEX)}")
    fig17()
    fig18()
    fig19()
    print("saved p2_s6_fig1{7,8,9}_curves.png\n逐曲线 MAE:")
    for ln in MAE_LINES:
        print(ln)
