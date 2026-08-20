"""Clean Scheme-2 rerun: Baik W1-W4, ledger crit 0.11 vs 0.239, with the
heave-spacing lru_cache cleared per case (fixes the stale-cache bug that
polluted the earlier multi-case macro)."""
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
from bing_joint_ptera import JointLEVTEVSolver, JointConfig
from bing_drag_ledger import LedgerConfig, run_ledger
from bing_baik_runner import build_movement_refined, executor

SPC = 128
cd0_baik = 2.0 * 1.328 / np.sqrt(5000.0)
res = {0.11: [], 0.239: []}
raws = []
for cid in ("W1", "W2", "W3", "W4"):
    case = baik.BAIK_2012_CASES[cid]
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
    qS = 0.5 * case.rho_kg_m3 * U**2 * case.area_m2
    FX = np.array([float(sp_.airplanes[0].forces_W[0])
                   for sp_ in solver.steady_problems])
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
    gp = np.array([p for p, _ in gt_cd])
    gv = np.array([v for _, v in gt_cd])
    last = slice(len(FX) - SPC, len(FX))
    phase_sim = (t[last] - t[last][0]) / case.period_s
    e_raw = np.interp(gp, phase_sim, lowpass(cd_raw)[last]) - gv
    raws.append(float(np.sqrt(np.mean(e_raw**2))))
    line = f"  {cid}: raw {raws[-1]:.3f}"
    for crit in (0.11, 0.239):
        cfg = LedgerConfig(lesp_crit=crit,
                           aspect_ratio=case.span_m / case.chord_m,
                           rho=case.rho_kg_m3, cd0=cd0_baik)
        cd_led = cd_raw.copy()
        for k, rec in enumerate(solver.ledger):
            cd_led[k] += run_ledger([rec], cfg, last_n=1)["mean_total_N"] / qS
        e = np.interp(gp, phase_sim, lowpass(cd_led)[last]) - gv
        res[crit].append(float(np.sqrt(np.mean(e**2))))
        line += f" | {crit}: {res[crit][-1]:.3f}"
    print(line, flush=True)

print(f"\nmacro CD raw {np.mean(raws):.3f} | crit0.11 {np.mean(res[0.11]):.3f} "
      f"| crit0.239 {np.mean(res[0.239]):.3f} (v4b 0.3452)")
