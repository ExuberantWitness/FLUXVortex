"""V4B transfer on the refined-mesh chassis (8 chordwise cosine x 8 spanwise).

For each Baik case: run the chassis bare (machine-precision bare core) at the
refined mesh, build the v4b baseline dict, apply apply_declared_v4b_transfer,
score against the digitized GT. Compare with bare@2x8 (0.793 macro) and
V4B@4x8 (0.594 macro).
"""
import csv
import importlib.util
import sys
from dataclasses import replace as _replace
from pathlib import Path

import numpy as np

repo = Path("/tmp/fluxv-v5-nextgen")
sys.path.insert(0, str(repo / "src"))
sys.path.insert(0, str(repo / "platform"))
sys.path.insert(0, str(repo / "platform/warp_vpm"))


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


executor = load("baik_ex", str(
    repo / "platform/forward_flight_benchmarks/fluxv_v5h15_baik_w2_executor.py"))
import pterasoftware
from forward_flight_benchmarks import baik2012 as baik
from forward_flight_benchmarks.baik2012 import apply_declared_v4b_transfer
from bing_joint_ptera import JointLEVTEVSolver, JointConfig
import bing_baik_runner as _runner
# single shared executor module instance (avoid double importlib load)
executor = _runner.executor
build_movement_refined = _runner.build_movement_refined

N_OUT = 128
SPC = int(__import__("os").environ.get("SPC", "128"))


def score(phase, sim, case_id, quantity):
    gt = []
    with open(repo / "docs/forward_flight_large_pitch/reproductions/"
              "baik2012_w1_w4/runs/20260813_baik2012_w1_w4_full_reproducible/"
              "scored_phase_samples.csv") as f:
        for row in csv.DictReader(f):
            if row["case_id"] == case_id and row["quantity"] == quantity:
                gt.append((float(row["phase"]), float(row["experiment"])))
    pts = sorted(gt)
    gp = np.array([p for p, _ in pts])
    gv = np.array([v for _, v in pts])
    pred = np.interp(gp, phase, sim)
    err = pred - gv
    return float(np.sqrt(np.mean(err**2))), float(np.mean(err))


macro = {"bare": [], "v4b": []}
for case_id in ("W1", "W2", "W3", "W4"):
    case = baik.BAIK_2012_CASES[case_id]
    executor.W2_CASE = _replace(
        executor.W2_CASE,
        strouhal=case.strouhal,
        reduced_frequency=case.reduced_frequency,
        heave_to_chord=case.heave_to_chord,
        period_s=case.period_s,
    )
    movement = build_movement_refined(pterasoftware, 8, 8, "cosine",
                                        steps_per_cycle=SPC, cycles=3)
    problem = pterasoftware.problems.UnsteadyProblem(
        movement=movement, only_final_results=False)
    solver = JointLEVTEVSolver(problem, JointConfig(enable_lev=False))
    solver.run(prescribed_wake=True, calculate_streamlines=False,
               show_progress=False)

    U = np.pi * (1.0 / case.period_s) * case.chord_m / case.reduced_frequency
    qS = 0.5 * case.rho_kg_m3 * U**2 * case.area_m2
    steps_per_cycle = SPC
    FX = np.array([float(sp_.airplanes[0].forces_W[0])
                   for sp_ in solver.steady_problems
                   if sp_.airplanes[0].forces_W is not None])
    FZ = np.array([float(sp_.airplanes[0].forces_W[2])
                   for sp_ in solver.steady_problems
                   if sp_.airplanes[0].forces_W is not None])
    steps = len(FZ)
    t = np.arange(steps) * (case.period_s / steps_per_cycle)
    CL = -FZ / qS
    CD = -FX / qS

    def lowpass(x, cutoff=1.0):
        X = np.fft.rfft(x)
        fr = np.fft.rfftfreq(len(x), t[1] - t[0])
        X[fr > cutoff] = 0.0
        return np.fft.irfft(X, n=len(x))

    last = slice(steps - steps_per_cycle, steps)
    phase_raw = (t[last] - t[last][0]) / case.period_s
    phase = np.arange(N_OUT) / N_OUT
    cl128 = np.interp(phase, phase_raw, lowpass(CL)[last], period=1.0)
    cd128 = np.interp(phase, phase_raw, lowpass(CD)[last], period=1.0)

    baseline = {
        "phase": phase,
        "lift_n": cl128 * qS,
        "thrust_n": -cd128 * qS,
        "CL": cl128.copy(),
        "CD": cd128.copy(),
        "mean_lift_n": float(np.mean(cl128 * qS)),
        "mean_thrust_n": float(np.mean(-cd128 * qS)),
        "source_cycle_step_range": [steps - steps_per_cycle, steps - 1],
    }
    v4b = apply_declared_v4b_transfer(
        case, baseline, movement,
        output_samples=N_OUT, ldvm_steps_per_cycle=512,
        ldvm_max_wake_steps=256, lesp_critical=0.11)

    r_bare_cl = score(phase, cl128, case_id, "CL")
    r_bare_cd = score(phase, cd128, case_id, "CD")
    r_v4b_cl = score(phase, np.asarray(v4b["CL"]), case_id, "CL")
    r_v4b_cd = score(phase, np.asarray(v4b["CD"]), case_id, "CD")
    macro["bare"].append((r_bare_cl[0], r_bare_cd[0]))
    macro["v4b"].append((r_v4b_cl[0], r_v4b_cd[0]))
    print(f"{case_id} bare@8x8cos: CL {r_bare_cl[0]:.3f} CD {r_bare_cd[0]:.3f} | "
          f"V4B@8x8cos: CL {r_v4b_cl[0]:.3f} CD {r_v4b_cd[0]:.3f}", flush=True)

bare_cl = float(np.mean([m[0] for m in macro["bare"]]))
bare_cd = float(np.mean([m[1] for m in macro["bare"]]))
v4b_cl = float(np.mean([m[0] for m in macro["v4b"]]))
v4b_cd = float(np.mean([m[1] for m in macro["v4b"]]))
print(f"\nMACRO bare@8x8cos: CL {bare_cl:.3f} CD {bare_cd:.3f}")
print(f"MACRO  V4B@8x8cos: CL {v4b_cl:.3f} CD {v4b_cd:.3f}")
print(f"reference: bare@2x8 CL 0.793 | V4B@4x8 CL 0.594")
