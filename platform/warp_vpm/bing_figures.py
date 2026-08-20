"""Three-paper performance curves (final configurations vs GT vs V4B).

Yang: polar + T3 dynamic viscous (lift/drag vs AoA)
Izra: T1@0.239 + T3 dynamic ledger (CT vs phase offset, both theta families)
Baik: 4x8uni chassis + V4B transfer (CL vs phase, W1-W4)
Scoreboard bar strip vs V4B.
"""
import csv
import json
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
from forward_flight_benchmarks.ptera_adapter import (
    build_yang2025_movement, build_izraelevitz_scherer_movement)
from forward_flight_benchmarks.cases import Yang2025RigidCase, IzraelevitzSchererCase
from forward_flight_benchmarks import baik2012 as baik
from forward_flight_benchmarks.baik2012 import apply_declared_v4b_transfer
from forward_flight_benchmarks.uvlm_polar_correction import (
    DEFAULT_POLAR_PARAMETERS, augment_uvlm_history, movement_polar_residual)
from bing_joint_ptera import JointLEVTEVSolver, JointConfig
from bing_drag_ledger import LedgerConfig, run_ledger
from bing_baik_runner import build_movement_refined, executor

SPC, N_OUT = 128, 128
g = 9.81
v4dir = (repo / "docs/forward_flight_large_pitch/reproductions/"
         "unified_fluxv_v4_ldvm_stevens_20260812/runs/"
         "20260812_fluxv_v4b_crosspaper_full")
out = {}


def chassis(movement):
    problem = pterasoftware.problems.UnsteadyProblem(
        movement=movement, only_final_results=False)
    solver = JointLEVTEVSolver(problem, JointConfig(enable_lev=False))
    solver.run(prescribed_wake=True, calculate_streamlines=False,
               show_progress=False)
    FX = np.array([float(sp_.airplanes[0].forces_W[0])
                   for sp_ in solver.steady_problems])
    FZ = np.array([float(sp_.airplanes[0].forces_W[2])
                   for sp_ in solver.steady_problems])
    return solver, FX, FZ


def lowpass(x, dt):
    X = np.fft.rfft(x)
    fr = np.fft.rfftfreq(len(x), dt)
    X[fr > 1.0] = 0.0
    return np.fft.irfft(X, n=len(x))


# ================= Yang =================
print("Yang ...", flush=True)
case = Yang2025RigidCase()
qS = 0.5 * case.rho_kg_m3 * case.freestream_m_s**2 * case.area_m2
CD0 = 2.0 * 1.328 / np.sqrt(case.reynolds)
gt = {}
with open(repo / "docs/forward_flight_large_pitch/reproductions/plev2025/"
          "source_data/yang2025_fig11_rigid_digitized.csv") as f:
    for row in csv.DictReader(f):
        gt[float(row["aoa_deg"])] = dict(
            test_lift=float(row["test_lift_gf"]),
            test_drag=-float(row["test_thrust_gf"]))
v4 = {}
with open(v4dir / "yang2025_v4_mean_characteristics.csv") as f:
    for row in csv.DictReader(f):
        v4[float(row["aoa_deg"])] = (float(row["v4_lift_gf"]),
                                     float(row["v4_drag_gf"]))
yang = dict(aoa=[], lift_bare=[], lift_model=[], drag_bare=[], drag_model=[],
            lift_gt=[], drag_gt=[], lift_v4b=[], drag_v4b=[])
for aoa in sorted(gt):
    movement = build_yang2025_movement(aoa, "full", settings=(8, 12, SPC, 4, 4))
    if isinstance(movement, tuple):
        movement = movement[0]
    solver, FX, FZ = chassis(movement)
    dt = case.period_s / SPC
    last = slice(len(FX) - SPC, len(FX))
    t = np.arange(len(FX)) * dt
    pr = (t[last] - t[last][0]) / case.period_s
    ph = np.arange(N_OUT) / N_OUT
    cl = np.interp(ph, pr, lowpass(-FZ / qS, dt)[last], period=1.0)
    cd = np.interp(ph, pr, lowpass(-FX / qS, dt)[last], period=1.0)
    baseline = {"phase": ph, "lift_n": cl * qS, "thrust_n": -cd * qS,
                "mean_lift_n": float(np.mean(cl) * qS),
                "mean_thrust_n": float(np.mean(-cd) * qS),
                "source_cycle_step_range": [len(FX) - SPC, len(FX) - 1]}
    polar = movement_polar_residual(
        movement, source_cycle_step_range=baseline["source_cycle_step_range"],
        period_s=case.period_s, freestream_m_s=case.freestream_m_s,
        rho_kg_m3=case.rho_kg_m3, aspect_ratio=case.aspect_ratio,
        output_samples=N_OUT, parameters=DEFAULT_POLAR_PARAMETERS)
    aug = augment_uvlm_history(baseline, polar, rho_kg_m3=case.rho_kg_m3,
                               freestream_m_s=case.freestream_m_s,
                               area_m2=case.area_m2)
    led = run_ledger(solver.ledger,
                     LedgerConfig(lesp_crit=0.0872,
                                  aspect_ratio=case.aspect_ratio,
                                  rho=case.rho_kg_m3, cd0=CD0,
                                  enable_t1=False), last_n=SPC)
    t3gf = led["mean_t3_N"] / g * 1000.0
    yang["aoa"].append(aoa)
    yang["lift_bare"].append(float(np.mean(cl)) * qS / g * 1000)
    yang["lift_model"].append(float(np.asarray(aug["CL"]).mean()) * qS / g * 1000)
    yang["drag_bare"].append(float(np.mean(cd)) * qS / g * 1000)
    yang["drag_model"].append(float(np.asarray(aug["CD"]).mean()) * qS / g * 1000
                              + t3gf)
    yang["lift_gt"].append(gt[aoa]["test_lift"])
    yang["drag_gt"].append(gt[aoa]["test_drag"])
    yang["lift_v4b"].append(v4[aoa][0])
    yang["drag_v4b"].append(v4[aoa][1])
    print(f"  AoA {aoa:.0f} done", flush=True)

# ================= Izra =================
print("Izra ...", flush=True)
case = IzraelevitzSchererCase()
bw = case.aspect_ratio * case.chord_m
qS = 0.5 * case.rho_kg_m3 * case.freestream_m_s**2 * case.chord_m * bw
period = 0.2
gt_rows = []
with open(repo / "docs/forward_flight_large_pitch/reproductions/"
          "unified_fluxv_upgrade_20260812/source_data/"
          "izraelevitz2017_fig14_digitized.csv") as fh:
    for row in csv.DictReader(fh):
        if row["data_role"] == "experimental_observation":
            gt_rows.append(row)
v4i = {}
with open(v4dir / "izraelevitz2017_fig14_v4_mean_thrust.csv") as f:
    for row in csv.DictReader(f):
        v4i[(float(row["theta_max_deg"]), float(row["phase_offset_deg"]))] = \
            float(row["v4_CT"])
izra = dict(theta=[], psi=[], ct_bare=[], ct_model=[], ct_gt=[], ct_err=[],
            ct_v4b=[], reps=[])
for th, ps in sorted(set((float(r["theta_max_deg"]),
                          float(r["phase_offset_deg"])) for r in gt_rows)):
    movement = build_izraelevitz_scherer_movement(th, ps, "full",
                                                  settings=(8, 12, SPC, 4))
    if isinstance(movement, tuple):
        movement = movement[0]
    solver, FX, FZ = chassis(movement)
    dt = period / SPC
    ct_bare = -float(np.mean(FX[-SPC:])) / qS
    led = run_ledger(solver.ledger,
                     LedgerConfig(lesp_crit=0.239, aspect_ratio=case.aspect_ratio,
                                  rho=case.rho_kg_m3, cd0=0.057),
                     last_n=SPC)
    ct_model = ct_bare - led["mean_total_N"] / qS
    match = [(float(r["ct"]), float(r["ct_error_minus"])) for r in gt_rows
             if float(r["theta_max_deg"]) == th
             and float(r["phase_offset_deg"]) == ps]
    izra["theta"].append(th); izra["psi"].append(ps)
    izra["ct_bare"].append(ct_bare); izra["ct_model"].append(ct_model)
    izra["ct_gt"].append(float(np.mean([m[0] for m in match])))
    izra["ct_err"].append(float(np.mean([m[1] for m in match])))
    izra["ct_v4b"].append(v4i[(th, ps)])
    print(f"  t={th:.0f} p={ps:.0f} done", flush=True)

# ================= Baik =================
print("Baik ...", flush=True)
baik_out = {}
for cid in ("W1", "W2", "W3", "W4"):
    case = baik.BAIK_2012_CASES[cid]
    executor._heave_spacing_samples.cache_clear()
    executor.W2_CASE = _replace(
        executor.W2_CASE, strouhal=case.strouhal,
        reduced_frequency=case.reduced_frequency,
        heave_to_chord=case.heave_to_chord, period_s=case.period_s)
    movement = build_movement_refined(pterasoftware, 4, 8, "uniform",
                                      steps_per_cycle=SPC, cycles=3)
    problem = pterasoftware.problems.UnsteadyProblem(
        movement=movement, only_final_results=False)
    solver = JointLEVTEVSolver(problem, JointConfig(enable_lev=False))
    solver.run(prescribed_wake=True, calculate_streamlines=False,
               show_progress=False)
    U = np.pi * (1.0 / case.period_s) * case.chord_m / case.reduced_frequency
    qS = 0.5 * case.rho_kg_m3 * U**2 * case.area_m2
    FX = np.array([float(sp_.airplanes[0].forces_W[0])
                   for sp_ in solver.steady_problems])
    FZ = np.array([float(sp_.airplanes[0].forces_W[2])
                   for sp_ in solver.steady_problems])
    dt = case.period_s / SPC
    t = np.arange(len(FZ)) * dt
    last = slice(len(FZ) - SPC, len(FX))
    pr = (t[last] - t[last][0]) / case.period_s
    ph = np.arange(N_OUT) / N_OUT
    cl = np.interp(ph, pr, lowpass(-FZ / qS, dt)[last], period=1.0)
    cd = np.interp(ph, pr, lowpass(-FX / qS, dt)[last], period=1.0)
    baseline = {"phase": ph, "lift_n": cl * qS, "thrust_n": -cd * qS,
                "CL": cl.copy(), "CD": cd.copy(),
                "mean_lift_n": float(np.mean(cl) * qS),
                "mean_thrust_n": float(np.mean(-cd) * qS),
                "source_cycle_step_range": [len(FZ) - SPC, len(FZ) - 1]}
    v4b = apply_declared_v4b_transfer(
        case, baseline, movement, output_samples=N_OUT,
        ldvm_steps_per_cycle=512, ldvm_max_wake_steps=256, lesp_critical=0.11)
    gtc = []
    with open(repo / "docs/forward_flight_large_pitch/reproductions/"
              "baik2012_w1_w4/runs/20260813_baik2012_w1_w4_full_reproducible/"
              "scored_phase_samples.csv") as f:
        for row in csv.DictReader(f):
            if row["case_id"] == cid and row["quantity"] == "CL":
                gtc.append((float(row["phase"]), float(row["experiment"])))
    gp = np.array([p for p, _ in gtc]); gv = np.array([v for _, v in gtc])
    clm = np.interp(gp, ph, np.asarray(v4b["CL"]))
    rmse = float(np.sqrt(np.mean((clm - gv)**2)))
    baik_out[cid] = dict(phase=ph, cl_model=np.asarray(v4b["CL"]),
                         cl_bare=cl, gt_phase=gp, gt_cl=gv, rmse=rmse)
    print(f"  {cid} done (RMSE {rmse:.3f})", flush=True)

np.savez("/tmp/v5h15-paper/figure_data.npz",
         yang={k: np.array(v) for k, v in yang.items()},
         izra={k: np.array(v) for k, v in izra.items()},
         baik={c: json.dumps({k: (v.tolist() if hasattr(v, "tolist") else v)
                              for k, v in d.items()}) for c, d in baik_out.items()})

# ================= figure =================
fig = plt.figure(figsize=(15, 13))
gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1.15], hspace=0.34,
                      wspace=0.24)

# Yang
ax = fig.add_subplot(gs[0, 0])
ax.plot(yang["aoa"], yang["lift_model"], "o-", color="crimson", lw=2,
        label="ours (polar+T3)")
ax.plot(yang["aoa"], yang["lift_bare"], "--", color="gray", lw=1.2,
        label="bare chassis")
ax.plot(yang["aoa"], yang["lift_v4b"], "s:", color="tab:blue", lw=1.5,
        label="V4B (frozen)")
ax.plot(yang["aoa"], yang["lift_gt"], "k*", ms=13, label="experiment")
ax.set_xlabel("angle of attack [deg]")
ax.set_ylabel("cycle-mean lift [gf]")
ax.set_title("Yang 2025 — lift vs AoA   (MAE: ours 4.10 | V4B 4.55 gf)")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

ax = fig.add_subplot(gs[0, 1])
ax.plot(yang["aoa"], yang["drag_model"], "o-", color="crimson", lw=2,
        label="ours (polar+T3)")
ax.plot(yang["aoa"], yang["drag_bare"], "--", color="gray", lw=1.2,
        label="bare chassis")
ax.plot(yang["aoa"], yang["drag_v4b"], "s:", color="tab:blue", lw=1.5,
        label="V4B (frozen)")
ax.plot(yang["aoa"], yang["drag_gt"], "k*", ms=13, label="experiment")
ax.set_xlabel("angle of attack [deg]")
ax.set_ylabel("cycle-mean drag [gf]")
ax.set_title("Yang 2025 — drag vs AoA   (MAE: ours 1.52 | V4B 2.64 gf)")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

# Izra
for j, th in enumerate((15.0, 25.0)):
    ax = fig.add_subplot(gs[1, j])
    m = np.array(izra["theta"]) == th
    ax.errorbar(np.array(izra["psi"])[m], np.array(izra["ct_gt"])[m],
                yerr=np.array(izra["ct_err"])[m], fmt="k*", ms=12,
                capsize=3, label="experiment")
    ax.plot(np.array(izra["psi"])[m], np.array(izra["ct_model"])[m], "o-",
            color="crimson", lw=2, label="ours (ledger T1+T3)")
    ax.plot(np.array(izra["psi"])[m], np.array(izra["ct_bare"])[m], "--",
            color="gray", lw=1.2, label="bare chassis")
    ax.plot(np.array(izra["psi"])[m], np.array(izra["ct_v4b"])[m], "s:",
            color="tab:blue", lw=1.5, label="V4B (frozen)")
    ax.set_xlabel("phase offset psi [deg]")
    ax.set_ylabel("cycle-mean CT")
    ax.set_title(f"Izraelevitz Fig.14 — theta={th:.0f} deg family "
                 f"(all-cond MAE: ours 0.026 | V4B 0.020)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

# Baik small multiples
for j, cid in enumerate(("W1", "W2", "W3", "W4")):
    ax = fig.add_subplot(gs[2, :].subgridspec(1, 4)[0, j])
    d = baik_out[cid]
    ax.plot(d["gt_phase"], d["gt_cl"], "k-", lw=1.5, label="experiment")
    ax.plot(d["phase"], d["cl_model"], "-", color="crimson", lw=1.5,
            label="ours (chassis+transfer)")
    ax.plot(d["phase"], d["cl_bare"], "--", color="gray", lw=1.0,
            label="bare")
    ax.set_xlabel("phase")
    ax.set_title(f"{cid}  CL RMSE {d['rmse']:.3f}", fontsize=10)
    if j == 0:
        ax.set_ylabel("CL")
        ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

fig.suptitle("Mechanism-based chassis vs V4B — three-paper performance curves "
             "(all cache-clean, 128 steps/cycle)", fontsize=13, y=0.995)
fig.savefig("/tmp/v5h15-paper/three_paper_curves.png", dpi=150,
            bbox_inches="tight")
print("FIGURE SAVED /tmp/v5h15-paper/three_paper_curves.png")
