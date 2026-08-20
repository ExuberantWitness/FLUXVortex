"""Fresh Baik W1--W4 scorer at the handoff's frozen LESPcrit=0.19."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from forward_flight_benchmarks.baik2012 import (
    BAIK_2012_CASES,
    baik_kinematics,
    sharp_fourier_lowpass,
)
from ldvm_fourier import LDVM2D


REPO = Path(__file__).resolve().parents[3]
OUTPUT = Path(__file__).resolve().parent / "fresh_results/baik_019"
GT = (
    REPO
    / "docs/forward_flight_large_pitch/reproductions/baik2012_w1_w4/runs/"
    "20260813_baik2012_w1_w4_full_reproducible/scored_phase_samples.csv"
)
LESP_CRIT = 0.19
STEPS_PER_CYCLE = 512
CYCLES = 3
N_OUTPUT = 128


def derivative(values: np.ndarray, step: float) -> np.ndarray:
    return (np.roll(values, -1) - np.roll(values, 1)) / (2.0 * step)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


samples: dict[str, dict[str, list[tuple[float, float]]]] = {
    case_id: {"CL": [], "CD": []} for case_id in BAIK_2012_CASES
}
with GT.open(newline="") as stream:
    for row in csv.DictReader(stream):
        samples[row["case_id"]][row["quantity"]].append(
            (float(row["phase"]), float(row["experiment"]))
        )

OUTPUT.mkdir(parents=True, exist_ok=False)
started = time.perf_counter()
rows = []
for case_id in ("W1", "W2", "W3", "W4"):
    case = BAIK_2012_CASES[case_id]
    fine_phase = np.arange(STEPS_PER_CYCLE, dtype=float) / STEPS_PER_CYCLE
    motion = baik_kinematics(fine_phase, case)
    alpha = np.deg2rad(motion["geometric_alpha_deg"])
    heave = np.asarray(motion["heave_rate_over_u"], dtype=float)
    delta_time = (
        case.freestream_m_s * case.period_s / case.chord_m / STEPS_PER_CYCLE
    )
    alpha_rate = derivative(alpha, delta_time)
    solver = LDVM2D(
        U=1.0,
        c=1.0,
        ndiv=32,
        naterm=14,
        dt=float(delta_time),
        rho=1.0,
        camber_m=0.0,
        pivot_xc=case.pivot_fraction_chord,
        core_rc=0.02,
        lesp_crit=LESP_CRIT,
        max_wake=256,
    )
    cl_history = []
    cd_history = []
    for step in range(CYCLES * STEPS_PER_CYCLE):
        index = step % STEPS_PER_CYCLE
        result = solver.step(alpha[index], alpha_rate[index], heave[index])
        cl_history.append(result["CLf"])
        cd_history.append(result["CDf"])
    output_phase = np.arange(N_OUTPUT, dtype=float) / N_OUTPUT
    cl = np.interp(
        output_phase,
        fine_phase,
        np.asarray(cl_history[-STEPS_PER_CYCLE:]),
        period=1.0,
    )
    cd = np.interp(
        output_phase,
        fine_phase,
        np.asarray(cd_history[-STEPS_PER_CYCLE:]),
        period=1.0,
    )
    harmonic = case.experimental_filter_harmonic
    cl = sharp_fourier_lowpass(cl, maximum_harmonic=harmonic)
    cd = sharp_fourier_lowpass(cd, maximum_harmonic=harmonic)
    rmse = {}
    for quantity, prediction in (("CL", cl), ("CD", cd)):
        points = sorted(samples[case_id][quantity])
        phase_gt = np.asarray([point[0] for point in points])
        value_gt = np.asarray([point[1] for point in points])
        error = np.interp(phase_gt, output_phase, prediction) - value_gt
        rmse[quantity] = float(np.sqrt(np.mean(error**2)))
    path = OUTPUT / f"{case_id}.npz"
    np.savez_compressed(path, phase=output_phase, cl=cl, cd=cd)
    row = {
        "case_id": case_id,
        "cl_rmse": rmse["CL"],
        "cd_rmse": rmse["CD"],
        "npz_sha256": digest(path),
    }
    rows.append(row)
    print(json.dumps(row, sort_keys=True), flush=True)

metrics = {
    "schema": "fluxv-v5m-baik-019-fresh-v1",
    "lesp_crit": LESP_CRIT,
    "cases": rows,
    "baik_cl_macro_rmse": float(np.mean([row["cl_rmse"] for row in rows])),
    "baik_cd_macro_rmse": float(np.mean([row["cd_rmse"] for row in rows])),
    "v4b_cl_macro_rmse": 0.658,
    "v4b_cd_macro_rmse": 0.345,
    "gt_sha256": digest(GT),
    "execution_class": "cpu-only-ldvm2d",
    "elapsed_seconds": time.perf_counter() - started,
}
(OUTPUT / "metrics.json").write_text(
    json.dumps(metrics, indent=2, sort_keys=True, allow_nan=False) + "\n",
    encoding="utf-8",
)
print(json.dumps(metrics, sort_keys=True), flush=True)

