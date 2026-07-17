"""STRICT paper-format Fig17/18/19: measured families + the CURRENT BEST closure
only (H16P = H16 + les_sep='polhamus', nc12 flexible replay; rigid = same closure).
Per-point error annotations + MAE; unfilled conditions annotated honestly.

Data: docs/repro_data.json (measured), docs/s6_results_lessep.json (H16P).
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
CFG = "H16v4"    # v2 baseline: correct kinematics (flap ±22.5, tw label=peak-to-peak)
                    # + correct pairing (Fig18/19 freq lines measured at tw22.5, kinematics_audit.md)

PTS = {}
_p = os.path.join(DOCS, "s6_results_lessep.json")
if os.path.exists(_p):
    for k, v in json.load(open(_p)).items():
        if v.get("cfg") == CFG and "flex" in v:
            parts = k.split("|")[-1].split("_")
            aoa = 5.0
            if parts and parts[-1].startswith("aoa"):
                aoa = float(parts.pop()[3:])
            U, f, tw = (float(x) for x in parts)
            PTS[(U, f, tw, aoa)] = v

FREQS = [1.4, 1.7, 2.0, 2.3, 2.6]
FCOL = {1.4: "#4477aa", 1.7: "#66ccee", 2.0: "#228833", 2.3: "#cc3311",
        2.6: "#aa3377"}
UCOL = {6.0: "#228833", 8.0: "#cc3311", 10.0: "#4477aa"}
ACOL = {0: "#4477aa", 5: "#cc3311", 10: "#228833", 15: "#aa3377"}
ERRS = {}


def _clear():
    ERRS.clear()


def _mae(idx):
    fl, rg = ERRS.get(("flex", idx), []), ERRS.get(("rigid", idx), [])
    if not fl:
        return "—"
    return (f"柔性 {np.mean(fl):.2f} / 刚性 {np.mean(rg):.2f} N({len(fl)}点)")


def meas_at(key, x):
    d = MEAS.get(key)
    return None if d is None else float(np.interp(x, d["x"], d["exp"]))


def draw_meas(ax, key, color, label=None):
    d = MEAS.get(key)
    if d is None:
        return
    ax.plot(d["x"], d["exp"], "-x", color=color, lw=1.3, ms=5, alpha=0.9,
            label=label)


def draw_model(ax, x, cond, idx, mkey, color):
    if len(cond) == 3:
        cond = cond + (5.0,)
    v = PTS.get(cond)
    if v is None:
        return
    mv = meas_at(mkey, x)
    fl, rg = v["flex"][idx], v["rigid"][idx]
    ax.plot(x, fl, "o", ms=9, color=color, mec="black", mew=0.8, zorder=6)
    ax.plot(x, rg, "s", ms=8, mfc="white", mec=color, mew=1.8, zorder=5)
    if mv is not None:
        ERRS.setdefault(("flex", idx), []).append(abs(fl - mv))
        ERRS.setdefault(("rigid", idx), []).append(abs(rg - mv))
        ax.annotate(f"{fl - mv:+.1f}", (x, fl), textcoords="offset points",
                    xytext=(6, 6), fontsize=7.5, color=color)


def legend_common(ax):
    h = [Line2D([], [], marker="o", ls="", ms=9, mfc="gray", mec="black",
                label=f"{CFG} 柔性(标注=误差N)"),
         Line2D([], [], marker="s", ls="", ms=8, mfc="white", mec="gray",
                mew=1.8, label=f"{CFG} 刚性(同闭合)"),
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
    fig.suptitle(f"Fig17 严格复现 v2 — U=8, AoA=5°, f2.3 行 [{CFG}] "
                 "(正确口径: 扑±22.5°, tw标签=峰-峰; tw45 已解锁)\n"
                 f"升力 MAE: {_mae(0)}   |   推力 MAE: {_mae(1)}")
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
            draw_model(axT, fr, (U, fr, 22.5), 1, f"18|a|{U}", UCOL[U])
            draw_model(axL, fr, (U, fr, 22.5), 0, f"18|b|{U}", UCOL[U])
    for ax, ttl in ((axT, "(a) T vs 频率 @tw22.5"), (axL, "(b) L vs 频率 @tw22.5")):
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
    fig.suptitle(f"Fig18 严格复现 v2 — AoA=5° [{CFG}] "
                 "(频率线实测条件=tw22.5, 模型同配对; kinematics_audit)\n"
                 f"升力 MAE: {_mae(0)}   |   推力 MAE: {_mae(1)}")
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
    for a in (0, 5, 10, 15):                       # 全 AoA 族(B线解锁)
        for fr in FREQS:
            draw_model(axT, fr, (8.0, fr, 22.5, float(a)), 1, f"19|a|{a}", ACOL[a])
            draw_model(axL, fr, (8.0, fr, 22.5, float(a)), 0, f"19|b|{a}", ACOL[a])
    for ax, ttl in ((axT, "(a) T vs 频率(各攻角)"),
                    (axL, "(b) L vs 频率(各攻角)")):
        ax.set_xlabel("扑动频率 (Hz)"); ax.set_ylabel("N")
        ax.grid(alpha=0.3); ax.set_title(ttl)
        legend_common(ax)
    for a in (0, 5, 10, 15):
        draw_meas(axCT, f"19|c|{a}", ACOL[a], f"实测 AoA={a}")
        draw_meas(axDL, f"19|d|{a}", ACOL[a], f"实测 AoA={a}")
    for tw in (0.0, 15.0, 22.5, 30.0, 45.0):       # f2.6 twist row (AoA=5)
        draw_model(axCT, tw, (8.0, 2.6, tw), 1, "19|c|5", ACOL[5])
        draw_model(axDL, tw, (8.0, 2.6, tw), 0, "19|d|5", ACOL[5])
    for ax, ttl in ((axCT, "(c) T vs 扭转 @f2.6"),
                    (axDL, "(d) L vs 扭转 @f2.6")):
        ax.set_xlabel("扭转幅值 (deg)"); ax.set_ylabel("N")
        ax.grid(alpha=0.3); ax.set_title(ttl)
        legend_common(ax)
    fig.suptitle(f"Fig19 严格复现 v2 — [{CFG}] "
                 "(频率线=tw22.5 正确配对; c/d 面板原文标注不自洽仅供参考)\n"
                 f"升力 MAE: {_mae(0)}   |   推力 MAE: {_mae(1)}")
    fig.tight_layout()
    fig.savefig(os.path.join(DOCS, "p2_s6_fig19.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    print(f"{CFG} points: {len(PTS)}")
    fig17(); fig18(); fig19()
    print("saved p2_s6_fig17/18/19.png")
