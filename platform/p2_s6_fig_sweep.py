"""全工况逐曲线对比图(用户指令 2026-07-15):118 工况刚性模型扫(s6_sweep_v3eff.json,
H16pltL2fn+d_para物理(v3闭合)/le/nc12)vs 实测(repro_data.json),25+ 条曲线全画、逐曲线 MAE。
输出 p2_s6_fig17_sweep.png / fig18_sweep / fig19_sweep + 控制台逐曲线 MAE 表。"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager, rcParams

for fn in ("Noto Sans CJK SC", "WenQuanYi Zen Hei", "AR PL UKai CN", "SimHei"):
    try:
        font_manager.findfont(fn, fallback_to_default=False)
        rcParams["font.sans-serif"] = [fn]
        break
    except Exception:
        pass
rcParams["axes.unicode_minus"] = False

DOCS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "docs")
MEAS = json.load(open(os.path.join(DOCS, "repro_data.json")))
SW = json.load(open(os.path.join(DOCS, "s6_sweep_v3eff.json")))

TWS = [0.0, 5.0, 10.0, 15.0, 20.0, 22.5, 25.0, 27.5, 30.0, 35.0, 40.0, 45.0]
FS = [1.4, 1.7, 2.0, 2.3, 2.6]
FCOL = {1.4: "#4477aa", 1.7: "#66ccee", 2.0: "#228833", 2.3: "#cc3311", 2.6: "#aa3377"}
UCOL = {6.0: "#228833", 8.0: "#cc3311", 10.0: "#4477aa"}
ACOL = {0.0: "#4477aa", 5.0: "#cc3311", 10.0: "#228833", 15.0: "#aa3377"}
TABLE = []


def model(U, f, tw, aoa):
    v = SW.get(f"{U:g}_{f:g}_{tw:g}_{aoa:g}")
    return None if (v is None or "fail" in v) else (v["L"], v["T"])


def curve(ax, mkey, xs, vals, color, lbl):
    """一条模型曲线 vs 一条实测曲线;逐曲线 MAE 进图例与全局表。"""
    d = MEAS.get(mkey)
    if d is not None:
        ax.plot(d["x"], d["exp"], "-x", color=color, lw=1.2, ms=4, alpha=0.85)
    ok = [(x, v) for x, v in zip(xs, vals) if v is not None]
    if not ok:
        return
    xs2, vs2 = zip(*ok)
    mae = None
    if d is not None:
        mv = np.interp(xs2, d["x"], d["exp"])
        mae = float(np.mean(np.abs(np.array(vs2) - mv)))
        TABLE.append((mkey, lbl, len(xs2), mae))
    ax.plot(xs2, vs2, "--o", color=color, lw=1.4, ms=5, mfc="white", mew=1.3,
            label=f"{lbl}" + (f"  MAE {mae:.2f}N" if mae is not None else ""))


def fig17():
    fig, (axT, axL) = plt.subplots(1, 2, figsize=(15, 6))
    for f in FS:
        Ls, Ts = zip(*[(model(8, f, tw, 5) or (None, None)) for tw in TWS])
        curve(axT, f"17|a|{f:.1f}", TWS, Ts, FCOL[f], f"f={f:g}")
        curve(axL, f"17|b|{f:.1f}", TWS, Ls, FCOL[f], f"f={f:g}")
    for ax, ttl in ((axT, "(a) 推力 vs 扭转幅值(标称,峰-峰)"),
                    (axL, "(b) 升力 vs 扭转幅值")):
        ax.set_xlabel("扭转幅值标称 (deg)")
        ax.set_ylabel("N")
        ax.grid(alpha=0.3)
        ax.set_title(ttl)
        ax.legend(fontsize=8, ncol=2, title="虚线圆点=模型  实线×=实测")
    fig.suptitle("Fig17 全工况逐曲线对比 — U=8, AoA=5°(5 频率 × 12 扭转 = 60 工况)\n[推力含有效系统阻力 d_para_eff=3.0×(U/8)²(显式去皮项;分解:0.5 物理+~2.5@U8 未解=T3b 开放案);U6/U10 绝对零位含实测基线带]"
                 "[H16pltL2 刚性, le/nc12, v2 运动学口径]")
    fig.tight_layout()
    fig.savefig(os.path.join(DOCS, "p2_s6_fig17_sweep.png"), dpi=150)
    plt.close(fig)


def fig18():
    fig, (axT, axL) = plt.subplots(1, 2, figsize=(15, 6))
    for U in (6.0, 8.0, 10.0):
        Ls, Ts = zip(*[(model(U, f, 22.5, 5) or (None, None)) for f in FS])
        curve(axT, f"18|a|{U}", FS, Ts, UCOL[U], f"U={U:g}")
        curve(axL, f"18|b|{U}", FS, Ls, UCOL[U], f"U={U:g}")
    for ax, ttl in ((axT, "(a) 推力 vs 频率 @tw22.5"), (axL, "(b) 升力 vs 频率 @tw22.5")):
        ax.set_xlabel("扑动频率 (Hz)")
        ax.set_ylabel("N")
        ax.grid(alpha=0.3)
        ax.set_title(ttl)
        ax.legend(fontsize=8, title="虚线圆点=模型  实线×=实测")
    fig.suptitle("Fig18 全工况逐曲线对比 — AoA=5°, tw=22.5(3 速度 × 5 频率 = 15 工况)\n[推力含有效系统阻力 d_para_eff=3.0×(U/8)²(显式去皮项;分解:0.5 物理+~2.5@U8 未解=T3b 开放案);U6/U10 绝对零位含实测基线带]"
                 "[H16pltL2 刚性; 实测频率线条件=tw22.5, kinematics_audit]")
    fig.tight_layout()
    fig.savefig(os.path.join(DOCS, "p2_s6_fig18_sweep.png"), dpi=150)
    plt.close(fig)


def fig19():
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    axT, axL, axCT, axDL = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]
    for a in (0.0, 5.0, 10.0, 15.0):
        Ls, Ts = zip(*[(model(8, f, 22.5, a) or (None, None)) for f in FS])
        curve(axT, f"19|a|{a:g}", FS, Ts, ACOL[a], f"AoA={a:g}")
        curve(axL, f"19|b|{a:g}", FS, Ls, ACOL[a], f"AoA={a:g}")
    for ax, ttl in ((axT, "(a) 推力 vs 频率(各攻角)@tw22.5"),
                    (axL, "(b) 升力 vs 频率(各攻角)@tw22.5")):
        ax.set_xlabel("扑动频率 (Hz)")
        ax.set_ylabel("N")
        ax.grid(alpha=0.3)
        ax.set_title(ttl)
        ax.legend(fontsize=8, ncol=2, title="虚线圆点=模型  实线×=实测")
    for a in (0.0, 5.0, 10.0, 15.0):
        Ls, Ts = zip(*[(model(8, 2.6, tw, a) or (None, None)) for tw in TWS])
        curve(axCT, f"19|c|{a:g}", TWS, Ts, ACOL[a], f"AoA={a:g}")
        curve(axDL, f"19|d|{a:g}", TWS, Ls, ACOL[a], f"AoA={a:g}")
    for ax, ttl in ((axCT, "(c) 推力 vs 扭转 @f2.6(原文频率标注不自洽,存疑)"),
                    (axDL, "(d) 升力 vs 扭转 @f2.6(同上)")):
        ax.set_xlabel("扭转幅值标称 (deg)")
        ax.set_ylabel("N")
        ax.grid(alpha=0.3)
        ax.set_title(ttl)
        ax.legend(fontsize=8, ncol=2, title="虚线圆点=模型  实线×=实测")
    fig.suptitle("Fig19 全工况逐曲线对比 — 4 攻角 ×(5 频率 + 12 扭转)= 63 工况\n[推力含有效系统阻力 d_para_eff=3.0×(U/8)²(显式去皮项;分解:0.5 物理+~2.5@U8 未解=T3b 开放案);U6/U10 绝对零位含实测基线带]"
                 "[H16pltL2 刚性; a/b 配对 tw22.5]")
    fig.tight_layout()
    fig.savefig(os.path.join(DOCS, "p2_s6_fig19_sweep.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    n_ok = sum(1 for v in SW.values() if "fail" not in v)
    print(f"sweep 工况: {n_ok}/{len(SW)}")
    fig17()
    fig18()
    fig19()
    print("saved p2_s6_fig1{7,8,9}_sweep.png\n")
    print(f"{'实测曲线':<12} {'模型曲线':<10} {'点数':>4} {'MAE(N)':>8}")
    for mkey, lbl, n, mae in TABLE:
        print(f"{mkey:<12} {lbl:<10} {n:>4} {mae:>8.2f}")
    Ts = [m for k, l, n, m in TABLE if "|a|" in k or "|c|" in k]
    Lsr = [m for k, l, n, m in TABLE if "|b|" in k or "|d|" in k]
    print(f"\n推力曲线均值 MAE {np.mean(Ts):.2f}N ({len(Ts)}条) | "
          f"升力曲线均值 MAE {np.mean(Lsr):.2f}N ({len(Lsr)}条)")
