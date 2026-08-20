"""P1+P5: cache-clean Baik CL macro, two meshes x {bare, +V4B transfer}.

Resolves whether the historical V4B 0.594-class advantage lives in the mesh
(4x8 uniform, the historical baseline mesh) or in a different load pipeline.
"""
import csv
import sys
from dataclasses import replace as _replace
from pathlib import Path

import numpy as np

repo = Path("/tmp/fluxv-v5-nextgen")
sys.path.insert(0, str(repo / "src"))
sys.path.insert(0, str(repo / "platform"))
sys.path.insert(0, str(repo / "platform/warp_vpm"))

import pterasoftware
from forward_flight_benchmarks import baik2012 as baik
from forward_flight_benchmarks.baik2012 import apply_declared_v4b_transfer
from bing_joint_ptera import JointLEVTEVSolver, JointConfig
from bing_baik_runner import build_movement_refined, executor

SPC = 128
N_OUT = 128


def lowpass(x, dt):
    X = np.fft.rfft(x)
    fr = np.fft.rfftfreq(len(x), dt)
    X[fr > 1.0] = 0.0
    return np.fft.irfft(X, n=len(x))


rows_out = {}
for mesh_name, (nch, nsp, spacing) in {"8x8cos": (8, 8, "cosine"),
                                       "4x8uni": (4, 8, "uniform")}.items():
    per = {c: {"bare": None, "v4b": None} for c in ("W1", "W2", "W3", "W4")}
    for cid in ("W1", "W2", "W3", "W4"):
        case = baik.BAIK_2012_CASES[cid]
        executor._heave_spacing_samples.cache_clear()
        executor.W2_CASE = _replace(
            executor.W2_CASE, strouhal=case.strouhal,
            reduced_frequency=case.reduced_frequency,
            heave_to_chord=case.heave_to_chord, period_s=case.period_s)
        movement = build_movement_refined(pterasoftware, nch, nsp, spacing,
                                          steps_per_cycle=SPC, cycles=3)
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
        dt = case.period_s / SPC
        t = np.arange(len(FZ)) * dt
        cl = lowpass(-FZ / qS, dt)
        last = slice(len(FZ) - SPC, len(FZ))
        phase_raw = (t[last] - t[last][0]) / case.period_s
        phase = np.arange(N_OUT) / N_OUT
        cl128 = np.interp(phase, phase_raw, cl[last], period=1.0)
        cd128 = np.interp(phase, phase_raw,
                          lowpass(-FX / qS, dt)[last], period=1.0)

        gt = {"CL": [], "CD": []}
        with open(repo / "docs/forward_flight_large_pitch/reproductions/"
                  "baik2012_w1_w4/runs/20260813_baik2012_w1_w4_full_reproducible/"
                  "scored_phase_samples.csv") as f:
            for row in csv.DictReader(f):
                if row["case_id"] == cid:
                    gt[row["quantity"]].append(
                        (float(row["phase"]), float(row["experiment"])))
        pts = sorted(gt["CL"])
        gp = np.array([p for p, _ in pts]); gv = np.array([v for _, v in pts])

        def rmse(pred):
            return float(np.sqrt(np.mean((np.interp(gp, phase, pred) - gv)**2)))

        per[cid]["bare"] = rmse(cl128)
        baseline = {
            "phase": phase, "lift_n": cl128 * qS, "thrust_n": -cd128 * qS,
            "CL": cl128.copy(), "CD": cd128.copy(),
            "mean_lift_n": float(np.mean(cl128 * qS)),
            "mean_thrust_n": float(np.mean(-cd128 * qS)),
            "source_cycle_step_range": [len(FZ) - SPC, len(FZ) - 1],
        }
        v4b = apply_declared_v4b_transfer(
            case, baseline, movement, output_samples=N_OUT,
            ldvm_steps_per_cycle=512, ldvm_max_wake_steps=256,
            lesp_critical=0.11)
        per[cid]["v4b"] = rmse(np.asarray(v4b["CL"]))
        print(f"[{mesh_name}] {cid}: bare CL {per[cid]['bare']:.3f} "
              f"+transfer {per[cid]['v4b']:.3f}", flush=True)
    mb = np.mean([per[c]["bare"] for c in per])
    mv = np.mean([per[c]["v4b"] for c in per])
    rows_out[mesh_name] = dict(per_case=per, macro_bare=float(mb),
                               macro_v4b=float(mv))
    print(f"[{mesh_name}] MACRO bare {mb:.3f} | +transfer {mv:.3f}\n",
          flush=True)

import json
Path("/tmp/v5h15-paper/p1_p5_clean.json").write_text(json.dumps(
    rows_out, indent=2, default=float))
print("P1+P5 DONE (historical refs: bare@2x8 0.793, v4b@4x8 0.6575)")
