"""Baik final: canonical pipeline (raw last cycle -> transfer -> per-cycle
sharp Fourier filter -> score). Fixes the multi-cycle rfft endpoint artifact
that cost W1/W2 ~0.05 RMSE."""
import csv
import sys
from pathlib import Path

import numpy as np

repo = Path("/tmp/fluxv-v5-nextgen")
sys.path.insert(0, str(repo / "src"))
sys.path.insert(0, str(repo / "platform"))
sys.path.insert(0, str(repo / "platform/warp_vpm"))

import pterasoftware
from forward_flight_benchmarks import baik2012 as baik
from forward_flight_benchmarks.baik2012 import (
    apply_declared_v4b_transfer, build_baik_movement, sharp_fourier_lowpass)
from bing_joint_ptera import JointLEVTEVSolver, JointConfig

SPC = 128
N_OUT = 128
macro = []
print("canonical pipeline: raw -> transfer -> sharp Fourier (declared harm.)")
for cid in ("W1", "W2", "W3", "W4"):
    case = baik.BAIK_2012_CASES[cid]
    movement, _ = build_baik_movement(case, "full", settings=(4, 8, 128, 3))
    problem = pterasoftware.problems.UnsteadyProblem(
        movement=movement, only_final_results=False)
    solver = JointLEVTEVSolver(problem, JointConfig(enable_lev=False))
    solver.run(prescribed_wake=True, calculate_streamlines=False,
               show_progress=False)
    U = np.pi * (1.0 / case.period_s) * case.chord_m / case.reduced_frequency
    qS = 0.5 * case.rho_kg_m3 * U**2 * case.area_m2
    FZ = np.array([float(sp_.airplanes[0].forces_W[2])
                   for sp_ in solver.steady_problems])
    FX = np.array([float(sp_.airplanes[0].forces_W[0])
                   for sp_ in solver.steady_problems])
    steps = len(FZ)
    t = np.arange(steps) * (case.period_s / SPC)
    last = slice(steps - SPC, steps)
    phase_raw = (t[last] - t[last][0]) / case.period_s
    ph = np.arange(N_OUT) / N_OUT
    cl = np.interp(ph, phase_raw, (-FZ / qS)[last], period=1.0)
    cd = np.interp(ph, phase_raw, (-FX / qS)[last], period=1.0)

    baseline = {"phase": ph, "lift_n": cl * qS, "thrust_n": -cd * qS,
                "CL": cl.copy(), "CD": cd.copy(),
                "mean_lift_n": float(np.mean(cl) * qS),
                "mean_thrust_n": float(np.mean(-cd) * qS),
                "source_cycle_step_range": [steps - SPC, steps - 1]}
    v4b = apply_declared_v4b_transfer(
        case, baseline, movement, output_samples=N_OUT,
        ldvm_steps_per_cycle=512, ldvm_max_wake_steps=256,
        lesp_critical=0.11)
    harm = case.experimental_filter_harmonic
    cl_f = sharp_fourier_lowpass(np.asarray(v4b["CL"]),
                                 maximum_harmonic=harm)
    cd_f = sharp_fourier_lowpass(np.asarray(v4b["CD"]),
                                 maximum_harmonic=harm)

    gt = {"CL": [], "CD": []}
    with open(repo / "docs/forward_flight_large_pitch/reproductions/"
              "baik2012_w1_w4/runs/20260813_baik2012_w1_w4_full_reproducible/"
              "scored_phase_samples.csv") as f:
        for row in csv.DictReader(f):
            if row["case_id"] == cid:
                gt[row["quantity"]].append((float(row["phase"]),
                                            float(row["experiment"])))
    pts = sorted(gt["CL"])
    gp = np.array([p for p, _ in pts])
    gv = np.array([v for _, v in pts])
    e = np.interp(gp, ph, cl_f) - gv
    rmse = float(np.sqrt(np.mean(e ** 2)))
    pts_d = sorted(gt["CD"])
    gpd = np.array([p for p, _ in pts_d])
    gvd = np.array([v for _, v in pts_d])
    ed = np.interp(gpd, ph, cd_f) - gvd
    rmse_d = float(np.sqrt(np.mean(ed ** 2)))
    macro.append(rmse)
    print(f"  {cid} (harm {harm}): CL RMSE {rmse:.4f} | CD RMSE {rmse_d:.4f}",
          flush=True)
    np.savez(f"/tmp/v5h15-paper/baik_final_{cid}.npz", phase=ph, cl=cl_f,
             cd=cd_f, cl_raw=cl, gt_phase=gp, gt_cl=gv)

print(f"MACRO CL: {np.mean(macro):.4f} (historical v4b 0.6575, "
      f"my prior 0.670)")
