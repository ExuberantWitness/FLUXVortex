"""Contour plots of per-panel load distribution (Baik W2, interpolated to a
fine grid via cubic interpolation on panel centers)."""
import sys
from dataclasses import replace as _replace
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.sans-serif"] = ["WenQuanYi Micro Hei", "Noto Sans CJK SC", "SimHei", "AR PL UMing CN", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

repo = Path("/tmp/fluxv-v5-nextgen")
sys.path.insert(0, str(repo / "src"))
sys.path.insert(0, str(repo / "platform"))
sys.path.insert(0, str(repo / "platform/warp_vpm"))

import pterasoftware
from forward_flight_benchmarks import baik2012 as baik
from bing_joint_ptera import JointLEVTEVSolver, JointConfig
from bing_baik_runner import build_movement_refined, executor

SPC = 128
case = baik.BAIK_2012_CASES["W2"]
executor._heave_spacing_samples.cache_clear()
executor.W2_CASE = _replace(
    executor.W2_CASE, strouhal=case.strouhal,
    reduced_frequency=case.reduced_frequency,
    heave_to_chord=case.heave_to_chord, period_s=case.period_s)
movement = build_movement_refined(pterasoftware, 8, 8, "cosine",
                                  steps_per_cycle=SPC, cycles=3)
problem = pterasoftware.problems.UnsteadyProblem(
    movement=movement, only_final_results=False)
solver = JointLEVTEVSolver(problem, JointConfig(enable_lev=False))
solver.run(prescribed_wake=True, calculate_streamlines=False,
               show_progress=False)

U = np.pi * (1.0 / case.period_s) * case.chord_m / case.reduced_frequency
q = 0.5 * case.rho_kg_m3 * U**2
cl = np.array([-float(np.asarray(sp_.airplanes[0].forces_W)[2])
               / (q * case.area_m2) for sp_ in solver.steady_problems])
last0 = len(cl) - SPC
peak = int(np.argmax(cl[last0:])) + last0
trough = int(np.argmin(cl[last0:])) + last0
mid = (peak + SPC // 4) % len(cl)

fig, axes = plt.subplots(3, 2, figsize=(13, 12),
                         gridspec_kw={"hspace": 0.32, "wspace": 0.22})
levels = None
for row, (label, step) in enumerate(
        [("最大CL相位（前缘吸力峰）", peak),
         ("过零相位（尾迹记忆）", mid),
         ("最小CL相位（反向载荷）", trough)]):
    sp_ = solver.steady_problems[step]
    panels = sp_.airplanes[0].wings[0].panels
    C, S = panels.shape
    pts, vals = [], []
    for jc in range(C):
        for sc in range(S):
            p = panels[jc, sc]
            f = np.asarray(p.forces_W)
            n = np.asarray(p.unitNormal_GP1)
            cn = float(f @ n) / (q * p.area)
            cpp = np.asarray(p.Cpp_GP1_CgP1)
            pts.append([cpp[0] / case.chord_m, cpp[1] / case.chord_m])
            vals.append(cn)
    pts = np.array(pts)
    vals = np.array(vals)
    gx, gy = np.meshgrid(np.linspace(pts[:, 0].min() - 0.05,
                                     pts[:, 0].max() + 0.05, 200),
                         np.linspace(pts[:, 1].min() - 0.1,
                                     pts[:, 1].max() + 0.1, 200))
    gz = griddata(pts, vals, (gx, gy), method="cubic")
    if levels is None:
        vmax = np.nanmax(np.abs(vals))
        levels = np.linspace(-vmax, vmax, 21)

    ax = axes[row, 0]
    cf = ax.contourf(gx, gy, gz, levels=levels, cmap="coolwarm",
                     extend="both")
    cs = ax.contour(gx, gy, gz, levels=levels[::2], colors="k",
                    linewidths=0.4, alpha=0.5)
    ax.clabel(cs, fontsize=6, fmt="%.1f")
    ax.plot(pts[:, 0], pts[:, 1], "k.", ms=2, alpha=0.35)
    ax.set_ylabel("展向 y/c")
    ax.set_title(f"{label}  (phase {step / SPC % 1:.2f})", fontsize=10)
    plt.colorbar(cf, ax=ax, label="面板法向力系数 $C_n$")
    ax.grid(alpha=0.15)

    ax2 = axes[row, 1]
    ax2.plot(pts[:, 1][np.argsort(pts[:, 1])],
             vals[np.argsort(pts[:, 1])], ".", alpha=0.15, color="gray")
    # strip mean
    strips = {}
    for (x, y), v in zip(pts, vals):
        strips.setdefault(round(y, 2), []).append(v)
    ys = sorted(strips)
    ax2.errorbar(ys, [np.mean(strips[y]) for y in ys],
                 yerr=[np.std(strips[y]) for y in ys], fmt="o-",
                 color="crimson", lw=1.6, capsize=3, label="均值 ± 标准差")
    ax2.axhline(0, color="gray", lw=0.8)
    ax2.set_xlabel("展向 y/c")
    if row == 0:
        ax2.set_ylabel("面板法向力系数 $C_n$")
        ax2.legend(fontsize=8)
    ax2.set_title(f"{label} — 展向分布", fontsize=10)
    ax2.grid(alpha=0.3)

fig.suptitle("Baik W2 — 翼面逐单元载荷等高线图（8×8余弦网格机架，瞬时未滤波）", fontsize=14)
fig.savefig("/tmp/v5h15-paper/panel_contours.png", dpi=150,
            bbox_inches="tight")
print("SAVED /tmp/v5h15-paper/panel_contours.png")
