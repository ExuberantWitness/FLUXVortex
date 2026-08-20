"""Izraelevitz Fig14 + Yang 2025 on the new machine-precision chassis.

Bare chassis (LEV off), 8 chordwise x 12 spanwise, 128 steps/cycle.
Outputs raw and Cd0-corrected CT (Izra) and gf lift/thrust (Yang).
"""
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

repo = Path("/tmp/fluxv-v5-nextgen")
sys.path.insert(0, str(repo / "src"))
sys.path.insert(0, str(repo / "platform"))
sys.path.insert(0, str(repo / "platform/warp_vpm"))

import pterasoftware
from forward_flight_benchmarks.ptera_adapter import (
    build_izraelevitz_scherer_movement, build_yang2025_movement)
from forward_flight_benchmarks.cases import IzraelevitzSchererCase, Yang2025RigidCase
from bing_joint_ptera import JointLEVTEVSolver, JointConfig

CD0_PRIMARY = 0.057   # frozen: Figure 14 convention
CD0_SENS = 0.027      # source-conflict sensitivity


def run_solver(movement):
    problem = pterasoftware.problems.UnsteadyProblem(
        movement=movement, only_final_results=False)
    solver = JointLEVTEVSolver(problem, JointConfig(enable_lev=False))
    solver.run(prescribed_wake=True, calculate_streamlines=False,
               show_progress=False)
    FX, FZ = [], []
    for sp in solver.steady_problems:
        f = sp.airplanes[0].forces_W
        if f is not None:
            FX.append(float(f[0])); FZ.append(float(f[2]))
    return np.array(FX), np.array(FZ)


def izra():
    case = IzraelevitzSchererCase()
    f = 5.0
    period = 1.0 / f
    U = case.freestream_m_s
    c = case.chord_m
    b = case.aspect_ratio * c
    qS = 0.5 * case.rho_kg_m3 * U**2 * c * b

    gt_rows = []
    with open(repo / "docs/forward_flight_large_pitch/reproductions/"
              "unified_fluxv_upgrade_20260812/source_data/"
              "izraelevitz2017_fig14_digitized.csv") as fh:
        for row in csv.DictReader(fh):
            if row["data_role"] == "experimental_observation":
                gt_rows.append(row)
    conditions = sorted(set((float(r["theta_max_deg"]), float(r["phase_offset_deg"]))
                            for r in gt_rows))
    print(f"Izra: {len(conditions)} conditions", flush=True)

    results = []
    for theta_max, psi in conditions:
        t0 = time.perf_counter()
        movement = build_izraelevitz_scherer_movement(
            theta_max, psi, "full", settings=(8, 12, 128, 4))
        if isinstance(movement, tuple):
            movement = movement[0]
        FX, FZ = run_solver(movement)
        spc = 128
        ct_raw = float(np.mean(FX[-spc:])) / qS
        ct_corr = ct_raw - CD0_PRIMARY
        ct_sens = ct_raw - CD0_SENS

        match = [r for r in gt_rows
                 if float(r["theta_max_deg"]) == theta_max
                 and float(r["phase_offset_deg"]) == psi]
        for j, r in enumerate(match):
            gct = float(r["ct"])
            gerr = float(r["ct_error_minus"])
            results.append(dict(theta_max=theta_max, psi=psi, replicate=j + 1,
                                ct_raw=ct_raw, ct_corr=ct_corr, ct_gt=gct,
                                abs_err_corr=abs(ct_corr - gct),
                                within_bar=abs(ct_corr - gct) <= gerr))
        g0 = float(match[0]["ct"])
        print(f"  t={theta_max:.0f} p={psi:.0f}: CT_raw={ct_raw:+.4f} "
              f"CT_corr={ct_corr:+.4f} GT={g0:+.4f} "
              f"|d|={abs(ct_corr-g0):.4f} [{time.perf_counter()-t0:.1f}s]",
              flush=True)

    errs = [r["abs_err_corr"] for r in results]
    within = sum(r["within_bar"] for r in results)
    mae_corr = float(np.mean(errs))
    raw_errs = [abs(r["ct_raw"] - r["ct_gt"]) for r in results]
    sens_errs = [abs(r["ct_raw"] - CD0_SENS - r["ct_gt"]) for r in results]
    print(f"\n=== Izra Fig14 @new chassis 8x12/128 ===")
    print(f"MAE corrected(Cd0=0.057): {mae_corr:.4f} | raw: "
          f"{float(np.mean(raw_errs)):.4f} | sens(0.027): "
          f"{float(np.mean(sens_errs)):.4f}")
    print(f"RMSE corrected: {float(np.sqrt(np.mean(np.array(errs)**2))):.4f} | "
          f"within bars {within}/{len(results)}")
    Path("/tmp/v5h15-paper/bing_izra_results.json").write_text(
        json.dumps(dict(results=results, mae_corr=mae_corr), indent=2,
                   default=float))
    return mae_corr


def yang():
    case = Yang2025RigidCase()
    g = 9.81
    gt = {}
    with open(repo / "docs/forward_flight_large_pitch/reproductions/plev2025/"
              "source_data/yang2025_fig11_rigid_digitized.csv") as f:
        for row in csv.DictReader(f):
            gt[float(row["aoa_deg"])] = dict(
                test_lift=float(row["test_lift_gf"]),
                test_thrust=float(row["test_thrust_gf"]),
                unc=float(row["digitization_uncertainty_gf"]))
    print(f"Yang: {len(gt)} AoA", flush=True)

    results = []
    for aoa in sorted(gt):
        t0 = time.perf_counter()
        movement = build_yang2025_movement(aoa, "full",
                                           settings=(8, 12, 128, 4, 4))
        if isinstance(movement, tuple):
            movement = movement[0]
        FX, FZ = run_solver(movement)
        spc = 128
        lift_gf = -float(np.mean(FZ[-spc:])) / g * 1000.0   # Baik convention: lift = -FZ
        thrust_gf = float(np.mean(FX[-spc:])) / g * 1000.0
        d = gt[aoa]
        results.append(dict(aoa=aoa, lift=lift_gf, thrust=thrust_gf,
                            lift_gt=d["test_lift"], thrust_gt=d["test_thrust"],
                            lift_err=abs(lift_gf - d["test_lift"]),
                            thrust_err=abs(thrust_gf - d["test_thrust"])))
        print(f"  AoA={aoa:.0f}: lift={lift_gf:+.1f}gf (GT {d['test_lift']:+.1f}) "
              f"thrust={thrust_gf:+.1f}gf (GT {d['test_thrust']:+.1f}) "
              f"[{time.perf_counter()-t0:.1f}s]", flush=True)

    lift_mae = float(np.mean([r["lift_err"] for r in results]))
    thrust_mae = float(np.mean([r["thrust_err"] for r in results]))
    first4_lift = float(np.mean([r["lift_err"] for r in results if r["aoa"] <= 15]))
    print(f"\n=== Yang 2025 @new chassis 8x12/128 ===")
    print(f"Lift MAE: {lift_mae:.1f} gf (first-4-AoA {first4_lift:.1f}) | "
          f"Thrust MAE: {thrust_mae:.1f} gf")
    Path("/tmp/v5h15-paper/bing_yang_results.json").write_text(
        json.dumps(dict(results=results, lift_mae=lift_mae,
                        thrust_mae=thrust_mae), indent=2, default=float))
    return lift_mae, thrust_mae


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    if which in ("izra", "both"):
        izra()
    if which in ("yang", "both"):
        yang()
    print("BING IZRA/YANG DONE")
