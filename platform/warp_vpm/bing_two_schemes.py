"""Two-scheme test (2026-08-20):

Scheme 1 (T2 memory via physical LEV): Yang with the LEV-ON chassis (bing
mode) + ledger. The LEV particle field carries the separation-state memory
physically; the self-suppressed LESP then gates T2 without over-firing.
Variants: (a) LEV-on + T1+T3, (b) LEV-on + T1+T2+T3.

Scheme 2 (Baik rounded-LE crit): Baik W2/W4 CD with ledger lesp_crit
0.11 (sharp-family) vs 0.239 (rounded-family, Izra-declared convention).
"""
import csv
import sys
import time
from dataclasses import replace as _replace
from pathlib import Path

import numpy as np

repo = Path("/tmp/fluxv-v5-nextgen")
sys.path.insert(0, str(repo / "src"))
sys.path.insert(0, str(repo / "platform"))
sys.path.insert(0, str(repo / "platform/warp_vpm"))

import pterasoftware
from forward_flight_benchmarks.ptera_adapter import build_yang2025_movement
from forward_flight_benchmarks.cases import Yang2025RigidCase
from forward_flight_benchmarks import baik2012 as baik
from bing_joint_ptera import JointLEVTEVSolver, JointConfig
from bing_drag_ledger import LedgerConfig, run_ledger
from bing_baik_runner import build_movement_refined, executor

g = 9.81
SPC = 128


def harvest(solver):
    FX, FZ = [], []
    for sp in solver.steady_problems:
        f = sp.airplanes[0].forces_W
        if f is not None:
            FX.append(float(f[0])); FZ.append(float(f[2]))
    return np.array(FX), np.array(FZ)


# ================= Scheme 1: Yang LEV-on + ledger =================
case = Yang2025RigidCase()
CD0 = 2.0 * 1.328 / np.sqrt(case.reynolds)
gt = {}
with open(repo / "docs/forward_flight_large_pitch/reproductions/plev2025/"
          "source_data/yang2025_fig11_rigid_digitized.csv") as f:
    for row in csv.DictReader(f):
        gt[float(row["aoa_deg"])] = dict(
            test_lift=float(row["test_lift_gf"]),
            test_drag=-float(row["test_thrust_gf"]))

print("=== Scheme 1: Yang, LEV-ON chassis + ledger ===")
print(f"{'AoA':>4} {'a:lift':>7} {'a:drag':>7} | {'b:lift':>7} {'b:drag':>7} "
      f"| {'GT lift':>7} {'GT drag':>7}  {'T2(b)gf':>8}")
res_a, res_b = [], []
for aoa in sorted(gt):
    t0 = time.perf_counter()
    movement = build_yang2025_movement(aoa, "full",
                                       settings=(8, 12, SPC, 2, 2))
    if isinstance(movement, tuple):
        movement = movement[0]
    problem = pterasoftware.problems.UnsteadyProblem(
        movement=movement, only_final_results=False)
    solver = JointLEVTEVSolver(
        problem, JointConfig(enable_lev=True, load_mode="bing",
                             lesp_crit=0.0872, lev_start_step=SPC + 4))
    solver.run(prescribed_wake=True, calculate_streamlines=False,
               show_progress=False)
    FX, FZ = harvest(solver)
    base_lift = -float(np.mean(FZ[-SPC:])) / g * 1000.0
    base_drag = -float(np.mean(FX[-SPC:])) / g * 1000.0

    cfg_a = LedgerConfig(lesp_crit=0.0872, aspect_ratio=case.aspect_ratio,
                         rho=case.rho_kg_m3, cd0=CD0, enable_t2=False)
    cfg_b = LedgerConfig(lesp_crit=0.0872, aspect_ratio=case.aspect_ratio,
                         rho=case.rho_kg_m3, cd0=CD0, enable_t2=True)
    led_a = run_ledger(solver.ledger, cfg_a, last_n=SPC)
    led_b = run_ledger(solver.ledger, cfg_b, last_n=SPC)
    lift_a = base_lift + led_a["mean_lift2_N"] / g * 1000.0  # 0 for T2-off
    drag_a = base_drag + led_a["mean_total_N"] / g * 1000.0
    lift_b = base_lift + led_b["mean_lift2_N"] / g * 1000.0
    drag_b = base_drag + led_b["mean_total_N"] / g * 1000.0
    t2gf = led_b["mean_t2_N"] / g * 1000.0
    d = gt[aoa]
    res_a.append((abs(lift_a - d["test_lift"]), abs(drag_a - d["test_drag"])))
    res_b.append((abs(lift_b - d["test_lift"]), abs(drag_b - d["test_drag"])))
    print(f"{aoa:4.0f} {lift_a:7.1f} {drag_a:7.1f} | {lift_b:7.1f} "
          f"{drag_b:7.1f} | {d['test_lift']:7.1f} {d['test_drag']:7.1f}  "
          f"{t2gf:8.1f} [{time.perf_counter()-t0:.0f}s]", flush=True)

la = np.mean([r[0] for r in res_a]); da = np.mean([r[1] for r in res_a])
lb = np.mean([r[0] for r in res_b]); db = np.mean([r[1] for r in res_b])
print(f"\n(a) LEV-on + T1T3 : lift {la:.2f} drag {da:.2f} gf")
print(f"(b) LEV-on + T1T2T3: lift {lb:.2f} drag {db:.2f} gf")
print(f"refs: LEV-off base lift 6.82 drag 13.0 | LEV-off+T1T3 drag 9.67 "
      f"| v4b lift 4.55 drag 2.64")

# ================= Scheme 2: Baik W2/W4 rounded crit =================
print("\n=== Scheme 2: Baik W2/W4 ledger crit 0.11 vs 0.239 ===")
cd0_baik = 2.0 * 1.328 / np.sqrt(5000.0)
for cid in ("W1", "W2", "W3", "W4"):
    case = baik.BAIK_2012_CASES[cid]
    executor._heave_spacing_samples.cache_clear()
    executor.W2_CASE = _replace(
        executor.W2_CASE, strouhal=case.strouhal,
        reduced_frequency=case.reduced_frequency,
        heave_to_chord=case.heave_to_chord, period_s=case.period_s)
    movement = build_movement_refined(pterasoftware, 8, 8, "cosine",
                                      steps_per_cycle=128, cycles=3)
    problem = pterasoftware.problems.UnsteadyProblem(
        movement=movement, only_final_results=False)
    solver = JointLEVTEVSolver(problem, JointConfig(enable_lev=False))
    solver.run(prescribed_wake=True, calculate_streamlines=False,
               show_progress=False)
    U = np.pi * (1.0 / case.period_s) * case.chord_m / case.reduced_frequency
    qS = 0.5 * case.rho_kg_m3 * U**2 * case.area_m2
    FX, _ = harvest(solver)
    t = np.arange(len(FX)) * (case.period_s / SPC)
    cd_raw = -FX / qS

    def lowpass(x):
        X = np.fft.rfft(x)
        fr = np.fft.rfftfreq(len(x), t[1] - t[0])
        X[fr > 1.0] = 0.0
        return np.fft.irfft(X, n=len(x))

    gt_cd = []
    with open(repo / "docs/forward_flight_large_pitch/reproductions/"
              "baik2012_w1_w4/runs/20260813_baik2012_w1_w4_full_reproducible/"
              "scored_phase_samples.csv") as f:
        for row in csv.DictReader(f):
            if row["case_id"] == cid and row["quantity"] == "CD":
                gt_cd.append((float(row["phase"]), float(row["experiment"])))
    gp = np.array([p for p, _ in gt_cd]); gv = np.array([v for _, v in gt_cd])
    last = slice(len(FX) - SPC, len(FX))
    phase_sim = (t[last] - t[last][0]) / case.period_s
    line = f"  {cid}: raw {np.sqrt(np.mean((np.interp(gp, phase_sim, lowpass(cd_raw)[last]) - gv)**2)):.3f}"
    for crit in (0.11, 0.239):
        cfg = LedgerConfig(lesp_crit=crit,
                           aspect_ratio=case.span_m / case.chord_m,
                           rho=case.rho_kg_m3, cd0=cd0_baik)
        cd_led = cd_raw.copy()
        for k, rec in enumerate(solver.ledger):
            cd_led[k] += run_ledger([rec], cfg, last_n=1)["mean_total_N"] / qS
        e = np.interp(gp, phase_sim, lowpass(cd_led)[last]) - gv
        line += f" | crit {crit}: {np.sqrt(np.mean(e**2)):.3f}"
    print(line, flush=True)
print("(prior: W2 0.688 raw -> 0.708 @0.11; W4 0.301 -> 0.343 @0.11; "
      "v4b macro 0.3452)")
