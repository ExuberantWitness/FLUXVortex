"""Regenerate the P2 assembly figures FROM THE ACTUAL WingModel (figure==code).

Replaces the hand-drawn design-intent versions of A/D/F in docs/ (B/C/E are
conceptual element/BC sketches and stay with the HTML source). Everything
plotted here is read off the built model: mesh nodes/tris, beam chains, mass
program output, strip added mass.

Run: cd FLUXV && python platform/p2_assembly_fig.py
"""
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
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
for p in (_ROOT, _HERE, os.path.join(_ROOT, "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

from wing_system import WingModel                             # noqa: E402
import _v2_robogeom as rg                                     # noqa: E402

DOCS = os.path.join(_HERE, "docs")
D_STAR = 3.77e-3
C_CARBON, C_CARBON2, C_PLY, C_MEM, C_ALU, C_AXIS = (
    "#2b2f36", "#5a626e", "#b07a28", "#3d7a6e", "#7d4a9c", "#b3382e")


def _mm(a):
    return np.asarray(a) * 1e3


def fig_A(m):
    fig, ax = plt.subplots(figsize=(11.5, 5.2))
    nodes = _mm(m.nodes)
    ax.triplot(nodes[:, 1], nodes[:, 0], m.mesh["tris"],
               color="#c9cec4", lw=0.5, zorder=1)
    ch = m.mesh["chains"]
    for r in ch["ribs"]:
        ax.plot(nodes[r, 1], nodes[r, 0], color=C_PLY, lw=2.8, zorder=3)
    ax.plot(nodes[ch["aux"], 1], nodes[ch["aux"], 0], color=C_CARBON2,
            lw=2.6, ls=(0, (6, 3)), zorder=4)
    ax.plot(nodes[ch["main"], 1], nodes[ch["main"], 0], color=C_CARBON,
            lw=3.6, zorder=4)
    ax.plot(nodes[ch["le"], 1], nodes[ch["le"], 0], color=C_CARBON,
            lw=3.0, zorder=4)
    for name in ("le", "main", "aux"):
        e = nodes[ch[name][-1]]
        ax.plot(e[1], e[0], "o", ms=7, color=C_CARBON, mec="white", zorder=6)
    root = nodes[[j * (m.nc + 1) for j in [0]][0]:m.nc + 1]   # j=0 row ids 0..nc
    ax.plot(root[:, 1], root[:, 0], color=C_ALU, lw=5, zorder=5)
    ax.plot([0, 800], [96.89, 0], color=C_AXIS, lw=1.4, ls="--", zorder=2)
    ys = np.linspace(0, 0.8, 200)
    ax.plot(_mm(ys), _mm(rg.chord_at(ys)), color=C_MEM, lw=1.6,
            ls=":", zorder=2)
    yme, yae = m.mesh["y_main"] * 1e3, m.mesh["y_aux"] * 1e3
    ax.annotate(f"主梁 Ø10×1 直线 x=88.6 → TE弧 ({yme:.1f})", (400, 78),
                color=C_CARBON, fontsize=10)
    ax.annotate(f"辅梁 Ø6×1 直线 x=198.4 → TE弧 ({yae:.1f})", (320, 212),
                color=C_CARBON2, fontsize=10)
    ax.annotate("LE 梁 Ø8 实心 → 翼尖 (0,800)", (520, -12), color=C_CARBON,
                fontsize=10)
    ax.annotate("扑动/扭转轴(实测斜掠)", (430, 45), color=C_AXIS, fontsize=9)
    ax.annotate("肋 ×7 航空层板 3×%.2fmm" % (m.rib_depth * 1e3), (95, 262),
                color=C_PLY, fontsize=9)
    ax.annotate("根肋(铝)= prescribed 扑动框", (5, 300), color=C_ALU, fontsize=9)
    ax.set_xlim(-25, 860); ax.set_ylim(320, -35)
    ax.set_xlabel("展向 y (mm)"); ax.set_ylabel("弦向 x (mm)")
    ax.set_title(f"图A 平面布置(由 WingModel 实例生成)— 结构平铺 z=0,"
                 f"{m.nn} 节点 / {m.MemC.ne} CST / {len(m.BeamC.edofs_np)} 梁元 / ndof {m.ndof}")
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(os.path.join(DOCS, "p2_assembly_A_planform.png"), dpi=160)
    plt.close(fig)


def fig_D(m):
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    fig = plt.figure(figsize=(11.5, 5.2))
    ax = fig.add_subplot(111, projection="3d")
    nodes = _mm(m.nodes)
    tris = m.mesh["tris"]
    ax.plot_trisurf(nodes[:, 1], nodes[:, 0], nodes[:, 2],
                    triangles=tris, color=(0.24, 0.48, 0.43, 0.18),
                    edgecolor="#b9beb4", linewidth=0.3)
    ch = m.mesh["chains"]
    # aero camber surface (grid + offset, at rest normals = +z), dashed wire
    g = nodes[m.nid_grid.ravel()].reshape(m.ns + 1, m.nc + 1, 3).copy()
    g[..., 2] += _mm(m.aero_off)
    for j in range(0, m.ns + 1, 4):
        ax.plot(g[j, :, 1], g[j, :, 0], g[j, :, 2], color=C_AXIS, lw=0.8,
                ls="--", alpha=0.7)
    for r in ch["ribs"]:
        ax.plot(nodes[r, 1], nodes[r, 0], nodes[r, 2], color=C_PLY, lw=2.2)
    ax.plot(nodes[ch["aux"], 1], nodes[ch["aux"], 0], nodes[ch["aux"], 2],
            color=C_CARBON2, lw=2.2, ls="--")
    ax.plot(nodes[ch["main"], 1], nodes[ch["main"], 0], nodes[ch["main"], 2],
            color=C_CARBON, lw=3.2)
    ax.plot(nodes[ch["le"], 1], nodes[ch["le"], 0], nodes[ch["le"], 2],
            color=C_CARBON, lw=2.6)
    rid = list(range(m.nc + 1))
    ax.plot(nodes[rid, 1], nodes[rid, 0], nodes[rid, 2], color=C_ALU, lw=4)
    ax.set_title("图D 三维几何(实模型)— 结构平铺 z=0(实),气动拱度面(红虚,NACA2406 偏置)")
    ax.set_xlabel("y (mm)"); ax.set_ylabel("x (mm)"); ax.set_zlabel("z")
    ax.set_box_aspect((800, 290, 80))
    ax.set_zticks([0, 5])
    ax.view_init(elev=32, azim=-65)
    ax.set_xlim(0, 800); ax.set_ylim(300, -10)
    fig.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)
    fig.savefig(os.path.join(DOCS, "p2_assembly_D_geometry3d.png"), dpi=160)
    plt.close(fig)


def fig_F(m):
    md = m.mass_detail
    m_node = md["m_node"] * 1e3                      # g
    m_add = m.m_added_node * 1e3                     # g
    nodes = _mm(m.nodes)
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.4),
                             gridspec_kw=dict(height_ratios=[1.35, 1]))
    for ax, val, col, ttl in ((axes[0, 0], m_node, C_CARBON,
                               f"结构节点质量 Σ={m_node.sum():.1f} g(质量程序 wing_mass)"),
                              (axes[0, 1], m_add, C_MEM,
                               f"条带附加质量 Σ={m_add.sum():.1f} g(ρπc²/4·A_trib/c)")):
        ax.triplot(nodes[:, 1], nodes[:, 0], m.mesh["tris"],
                   color="#d8d6cc", lw=0.4)
        ax.scatter(nodes[:, 1], nodes[:, 0], s=110 * val / max(val.max(), 1e-9),
                   color=col, alpha=0.6, edgecolor=col)
        ax.set_title(ttl, fontsize=10)
        ax.set_aspect("equal"); ax.invert_yaxis()
        ax.set_xlabel("y (mm)"); ax.set_ylabel("x (mm)")
    ax = axes[1, 0]
    yb = np.linspace(0, 800, 17)
    idx = np.clip((nodes[:, 1] // 50).astype(int), 0, 16)
    row_str = np.bincount(idx, weights=m_node, minlength=17)
    row_add = np.bincount(idx, weights=m_add, minlength=17)
    ax.plot(yb, row_str, "-o", color=C_CARBON, ms=3, label="结构 g/站位")
    ax.plot(yb, row_add, "--s", color=C_MEM, ms=3, label="附加质量 g/站位")
    ax.set_xlabel("展向 y (mm)"); ax.set_ylabel("g / 50mm 站位")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    ax = axes[1, 1]
    tt = md["totals"]
    names = ["le_spar", "main_spar", "aux_spar", "membrane"]
    vals = [tt[n] * 1e3 for n in names] + [
        sum(v for k, v in tt.items() if k.startswith("rib")) * 1e3]
    labels = ["LE Ø8", "主 Ø10×1", "辅 Ø6×1", "膜 0.05", f"肋×7 d*={m.rib_depth*1e3:.2f}"]
    ax.bar(labels, vals, color=[C_CARBON, C_CARBON, C_CARBON2, C_MEM, C_PLY])
    for i, v in enumerate(vals):
        ax.text(i, v + 0.5, f"{v:.1f}", ha="center", fontsize=9)
    ax.set_ylabel("g"); ax.set_title(f"构件质量分解,合计 {sum(vals):.1f} g", fontsize=10)
    fig.suptitle("图F 翼面量分布(由 WingModel 实例 + wing_mass 程序生成)")
    fig.tight_layout()
    fig.savefig(os.path.join(DOCS, "p2_assembly_F_distributions.png"), dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    m = WingModel(rib_depth=D_STAR)
    fig_A(m); fig_D(m); fig_F(m)
    print("regenerated p2_assembly_{A,D,F} from the actual WingModel "
          f"(d*={D_STAR*1e3:.2f}mm, mass {m.mass_detail['total']*1e3:.1f}g)")
