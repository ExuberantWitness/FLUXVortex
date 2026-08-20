"""P3: drag ledger on Izra Fig14 (dynamic viscous replaces constant Cd0)
and Baik W1-W4 (CD macro).

Izra: lesp_crit 0.239 / cd0 0.057 (both frozen crosspaper-declared), AR=3.
Baik: lesp_crit 0.11 (frozen 2D chain), cd0 = Blasius 2*1.328/sqrt(5000),
AR = span/chord.
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
from forward_flight_benchmarks.ptera_adapter import (
    build_izraelevitz_scherer_movement)
from forward_flight_benchmarks.cases import IzraelevitzSchererCase
from forward_flight_benchmarks import baik2012 as baik
from bing_joint_ptera import JointLEVTEVSolver, JointConfig
from bing_drag_ledger import LedgerConfig, run_ledger
from bing_baik_runner import build_movement_refined, executor


def harvest(solver):
    FX, FZ = [], []
    for sp in solver.steady_problems:
        f = sp.airplanes[0].forces_W
        if f is not None:
            FX.append(float(f[0])); FZ.append(float(f[2]))
    return np.array(FX), np.array(FZ)


# ---------------- Izra ----------------
case = IzraelevitzSchererCase()
U = case.freestream_m_s
b = case.aspect_ratio * case.chord_m
qS = 0.5 * case.rho_kg_m3 * U**2 * case.chord_m * b
cfg_led = LedgerConfig(lesp_crit=0.239, aspect_ratio=case.aspect_ratio,
                       rho=case.rho_kg_m3, cd0=0.057)

gt_rows = []
with open(repo / "docs/forward_flight_large_pitch/reproductions/"
          "unified_fluxv_upgrade_20260812/source_data/"
          "izraelevitz2017_fig14_digitized.csv") as fh:
    for row in csv.DictReader(fh):
        if row["data_role"] == "experimental_observation":
            gt_rows.append(row)
conditions = sorted(set((float(r["theta_max_deg"]), float(r["phase_offset_deg"]))
                        for r in gt_rows))
errs_dyn, errs_const = [], []
print("=== Izra: dynamic viscous vs constant 0.057 ===")
for th, ps in conditions:
    movement = build_izraelevitz_scherer_movement(th, ps, "full",
                                                  settings=(8, 12, 128, 4))
    if isinstance(movement, tuple):
        movement = movement[0]
    problem = pterasoftware.problems.UnsteadyProblem(
        movement=movement, only_final_results=False)
    solver = JointLEVTEVSolver(problem, JointConfig(enable_lev=False))
    solver.run(prescribed_wake=True, calculate_streamlines=False,
               show_progress=False)
    FX, _ = harvest(solver)
    spc = 128
    ct_raw = float(np.mean(FX[-spc:])) / qS
    led = run_ledger(solver.ledger, cfg_led, last_n=spc)
    ct_dyn = ct_raw - led["mean_total_N"] / qS
    ct_const = ct_raw - 0.057
    for r in gt_rows:
        if float(r["theta_max_deg"]) == th and float(r["phase_offset_deg"]) == ps:
            errs_dyn.append(abs(ct_dyn - float(r["ct"])))
            errs_const.append(abs(ct_const - float(r["ct"])))
    print(f"  t={th:.0f} p={ps:.0f}: raw {ct_raw:+.4f} dyn {ct_dyn:+.4f} "
          f"const {ct_const:+.4f}  T1 {led['mean_t1_N']/qS:+.4f} "
          f"T3dyn {led['mean_t3_N']/qS:+.4f}", flush=True)
print(f"Izra MAE dynamic: {np.mean(errs_dyn):.4f} | constant: "
      f"{np.mean(errs_const):.4f} (prior run 0.0386)")

# ---------------- Baik ----------------
print("\n=== Baik W1-W4: CD with T1+T3 ===")
cd0_baik = 2.0 * 1.328 / np.sqrt(5000.0)
macro_cd_raw, macro_cd_led = [], []
macro_cl = []
for cid in ("W1", "W2", "W3", "W4"):
    case = baik.BAIK_2012_CASES[cid]
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
    cfg = LedgerConfig(lesp_crit=0.11, aspect_ratio=case.span_m / case.chord_m,
                       rho=case.rho_kg_m3, cd0=cd0_baik)
    FX, FZ = harvest(solver)
    spc = 128
    t = np.arange(len(FX)) * (case.period_s / spc)
    cd_raw = -FX / qS

    def lowpass(x):
        X = np.fft.rfft(x)
        fr = np.fft.rfftfreq(len(x), t[1] - t[0])
        X[fr > 1.0] = 0.0
        return np.fft.irfft(X, n=len(x))

    cd_led = cd_raw.copy()
    for k, rec in enumerate(solver.ledger):
        out_led = run_ledger([rec], cfg, last_n=1)
        cd_led[k] += out_led["mean_total_N"] / qS
    last = slice(len(FX) - spc, len(FX))
    phase_sim = (t[last] - t[last][0]) / case.period_s
    gt = []
    with open(repo / "docs/forward_flight_large_pitch/reproductions/"
              "baik2012_w1_w4/runs/20260813_baik2012_w1_w4_full_reproducible/"
              "scored_phase_samples.csv") as f:
        for row in csv.DictReader(f):
            if row["case_id"] == cid and row["quantity"] == "CD":
                gt.append((float(row["phase"]), float(row["experiment"])))
    gp = np.array([p for p, _ in gt]); gv = np.array([v for _, v in gt])
    e_raw = np.interp(gp, phase_sim, lowpass(cd_raw)[last]) - gv
    e_led = np.interp(gp, phase_sim, lowpass(cd_led)[last]) - gv
    macro_cd_raw.append(np.sqrt(np.mean(e_raw**2)))
    macro_cd_led.append(np.sqrt(np.mean(e_led**2)))
    print(f"  {cid}: CD RMSE raw {macro_cd_raw[-1]:.3f} -> ledger "
          f"{macro_cd_led[-1]:.3f}", flush=True)
print(f"Baik CD macro: raw {np.mean(macro_cd_raw):.3f} -> ledger "
      f"{np.mean(macro_cd_led):.3f} (v4b 0.3452, prior run 0.379)")
