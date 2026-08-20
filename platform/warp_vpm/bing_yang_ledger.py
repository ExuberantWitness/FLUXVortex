"""Yang 2025 drag-ledger validation (plan P1/P2).

Baseline: chassis LEV-off (drag MAE 13.0 gf). Add T1+T3 ledger, Cd0 declared
Blasius, lesp_crit = frozen Yang declared 0.0872. Expect drag MAE <= 6 gf,
lift unchanged.
"""
import csv
import sys
import time
from pathlib import Path

import numpy as np

repo = Path("/tmp/fluxv-v5-nextgen")
sys.path.insert(0, str(repo / "src"))
sys.path.insert(0, str(repo / "platform"))
sys.path.insert(0, str(repo / "platform/warp_vpm"))

import pterasoftware
from forward_flight_benchmarks.ptera_adapter import build_yang2025_movement
from forward_flight_benchmarks.cases import Yang2025RigidCase
from bing_joint_ptera import JointLEVTEVSolver, JointConfig
from bing_drag_ledger import LedgerConfig, run_ledger

case = Yang2025RigidCase()
g = 9.81
SPC = 128
CD0 = 2.0 * 1.328 / np.sqrt(case.reynolds)   # Blasius, both sides
print(f"declared Cd0 = {CD0:.5f}, AR = {case.aspect_ratio:.3f}")

gt = {}
with open(repo / "docs/forward_flight_large_pitch/reproductions/plev2025/"
          "source_data/yang2025_fig11_rigid_digitized.csv") as f:
    for row in csv.DictReader(f):
        gt[float(row["aoa_deg"])] = dict(
            test_lift=float(row["test_lift_gf"]),
            test_drag=-float(row["test_thrust_gf"]))

cfg_led = LedgerConfig(
    lesp_crit=0.0872, aspect_ratio=case.aspect_ratio,
    rho=case.rho_kg_m3, cd0=CD0, enable_t2=False)

print(f"{'AoA':>4} {'lift':>7} {'GT':>6} | {'drag_raw':>8} {'T1':>6} {'T3':>6} "
      f"{'drag_corr':>9} {'GT':>6}")
lift_errs, drag_errs = [], []
for aoa in sorted(gt):
    t0 = time.perf_counter()
    movement = build_yang2025_movement(aoa, "full", settings=(8, 12, SPC, 4, 4))
    if isinstance(movement, tuple):
        movement = movement[0]
    problem = pterasoftware.problems.UnsteadyProblem(
        movement=movement, only_final_results=False)
    solver = JointLEVTEVSolver(problem, JointConfig(enable_lev=False))
    solver.run(prescribed_wake=True, calculate_streamlines=False,
               show_progress=False)
    FX = np.array([float(sp_.airplanes[0].forces_W[0])
                   for sp_ in solver.steady_problems])
    FZ = np.array([float(sp_.airplanes[0].forces_W[2])
                   for sp_ in solver.steady_problems])
    led = run_ledger(solver.ledger, cfg_led, last_n=SPC)
    lift_gf = -float(np.mean(FZ[-SPC:])) / g * 1000.0 \
        + led["mean_lift2_N"] / g * 1000.0
    drag_raw = -float(np.mean(FX[-SPC:])) / g * 1000.0
    drag_corr = drag_raw + led["mean_total_N"] / g * 1000.0
    d = gt[aoa]
    lift_errs.append(abs(lift_gf - d["test_lift"]))
    drag_errs.append(abs(drag_corr - d["test_drag"]))
    t1gf = led["mean_t1_N"] / g * 1000.0
    t3gf = led["mean_t3_N"] / g * 1000.0
    t2gf = led["mean_t2_N"] / g * 1000.0
    print(f"{aoa:4.0f} {lift_gf:7.1f} {d['test_lift']:6.1f} | {drag_raw:8.1f} "
          f"{t1gf:6.1f} {t3gf:6.1f} {drag_corr:9.1f} {d['test_drag']:6.1f} "
          f"[{time.perf_counter()-t0:.0f}s]", flush=True)

print(f"\nlift MAE: {np.mean(lift_errs):.2f} gf (baseline 6.8, must hold) | "
      f"drag MAE: {np.mean(drag_errs):.2f} gf (baseline 13.0, target <=6, "
      f"v4b 2.64)")
