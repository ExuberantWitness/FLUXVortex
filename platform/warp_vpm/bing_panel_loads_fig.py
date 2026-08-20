"""Per-panel load distribution on the wing (Baik W2, chassis 8x8cos).

Three representative phases of the last cycle: chordwise x spanwise map of
the per-panel normal-force coefficient Cn_j = (F_j . n_j)/(q A_j), plus the
spanwise strip sums. Raw (unfiltered) instantaneous panel forces.
"""
import sys
from dataclasses import replace as _replace
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
# strip CL history for phase selection
cl = np.array([-float(np.asarray(sp_.airplanes[0].forces_W)[2])
               / (q * case.area_m2) for sp_ in solver.steady_problems])
ph_all = np.arange(len(cl)) / SPC
peak = np.argmax(cl[-SPC:]) + len(cl) - SPC          # max-lift step
phases = {"min CL": np.argmin(cl[-SPC:]) + len(cl) - SPC,
          "mid (zero-cross)": int(peak + SPC / 4) % (len(cl)) ,
          "max CL": peak}

fig, axes = plt.subplots(2, 3, figsize=(15, 8.6),
                         gridspec_kw={"height_ratios": [1.5, 1],
                                      "hspace": 0.3, "wspace": 0.28})
cmap = "coolwarm"
vmax = None
snap = {}
for j, (label, step) in enumerate(phases.items()):
    sp_ = solver.steady_problems[step]
    panels = sp_.airplanes[0].wings[0].panels
    C, S = panels.shape
    cn = np.zeros((C, S))
    xg = np.zeros((C, S))
    yg = np.zeros((C, S))
    for jc in range(C):
        for sc in range(S):
            p = panels[jc, sc]
            f = p.forces_W
            n = np.asarray(p.unitNormal_GP1)
            # normal force coefficient per panel (world-frame normal ~ GP1)
            from pterasoftware import _transformations as tr
            T = sp_.airplanes[0].Cg_GP1_CgP1  # placeholder, use GP1 normal
            cn[jc, sc] = float(np.asarray(f) @ n) / (q * p.area)
            xg[jc, sc] = float(p.Cpp_GP1_CgP1[0])
            yg[jc, sc] = float(p.Cpp_GP1_CgP1[1])
    snap[label] = (xg, yg, cn)
    vmax = max(vmax or 0.0, np.abs(cn).max())

for j, (label, (xg, yg, cn)) in enumerate(snap.items()):
    ax = axes[0, j]
    sc_ = ax.scatter(xg / case.chord_m, yg / case.chord_m, c=cn, s=210,
                     cmap=cmap, vmin=-vmax, vmax=vmax, edgecolors="k",
                     linewidths=0.4)
    ax.set_xlabel("x/c")
    if j == 0:
        ax.set_ylabel("y/c (span)")
    ax.set_title(f"Baik W2 @ {label}\n(phase {ph_all[list(snap)[j] if False else list(snap.keys()).index(label)] % 1.0:.2f})",
                 fontsize=10)
    ax.grid(alpha=0.2)
    plt.colorbar(sc_, ax=ax, label="panel $C_n$")

    ax2 = axes[1, j]
    strip = cn.mean(axis=0)                     # chordwise-mean per strip
    ys = yg.mean(axis=0) / case.chord_m
    ax2.plot(ys, strip, "o-", color="crimson", lw=1.6)
    ax2.axhline(0, color="gray", lw=0.8)
    ax2.set_xlabel("y/c")
    if j == 0:
        ax2.set_ylabel("strip-mean $C_n$")
    ax2.grid(alpha=0.3)

fig.suptitle("Baik W2 per-panel load distribution (chassis 8x8 cosine, "
             "instantaneous, unfiltered)", fontsize=13)
fig.savefig("/tmp/v5h15-paper/panel_loads.png", dpi=150, bbox_inches="tight")
print("SAVED /tmp/v5h15-paper/panel_loads.png")
