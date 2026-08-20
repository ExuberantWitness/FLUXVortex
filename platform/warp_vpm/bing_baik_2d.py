"""E1: 2D LDVM (validated, Ramesh-parity) scored directly against Baik GT.

The experiment is wall-to-wall end-plated quasi-2D (SOURCE_AUDIT). The 2D
LDVM with LEV + added-mass (CNnc) + unsteady wake is that regime. Driving
pattern copied verbatim from the frozen baik2012 v4b-transfer driver:
alpha = geometric alpha, alpha_rate = periodic derivative in convective
time, heave_rate = heave_rate_over_u; 3 cycles, last cycle scored with the
canonical per-cycle sharp Fourier filter.
"""
import csv
import sys
from pathlib import Path

import numpy as np

repo = Path("/tmp/fluxv-v5-nextgen")
sys.path.insert(0, str(repo / "src"))
sys.path.insert(0, str(repo / "platform"))
sys.path.insert(0, str(repo / "platform/warp_vpm"))

from forward_flight_benchmarks.baik2012 import (
    BAIK_2012_CASES, baik_kinematics, sharp_fourier_lowpass)
from forward_flight_benchmarks.ldvm_uvlm_correction import (
    LESPThreshold, LDVMSectionSettings)
from ldvm_fourier import LDVM2D

N_OUT = 128
LDVM_SPC = 512     # convective-resolution steps per cycle for the 2D solve
CYCLES = 3


def periodic_derivative(values, step):
    return (np.roll(values, -1) - np.roll(values, 1)) / (2.0 * step)


macro_cl, macro_cd = [], []
for cid in ("W1", "W2", "W3", "W4"):
    case = BAIK_2012_CASES[cid]
    phase = np.arange(LDVM_SPC, dtype=float) / LDVM_SPC
    kin = baik_kinematics(phase, case)
    alpha = np.deg2rad(kin["geometric_alpha_deg"])
    heave = np.asarray(kin["heave_rate_over_u"], dtype=float)
    # convective time: t* = t*U/c; dt* = (U/c)*(T/SPC)
    dt_star = case.freestream_m_s * case.period_s / case.chord_m / LDVM_SPC
    alpha_rate = periodic_derivative(alpha, dt_star)

    threshold = LESPThreshold(
        value=0.11, section_family="rounded flat plate",
        reynolds=case.reynolds,
        source="Ramesh 2013 flat-plate Re=1000 declared primary",
        source_role="published transfer hypothesis")
    ldvm = LDVM2D(U=1.0, c=1.0, ndiv=32, naterm=14,
                  dt=float(dt_star), rho=1.0, camber_m=0.0,
                  pivot_xc=case.pivot_fraction_chord, core_rc=0.02,
                  lesp_crit=threshold.value, max_wake=256)
    cl_hist, cd_hist = [], []
    for k in range(CYCLES * LDVM_SPC):
        i = k % LDVM_SPC
        out = ldvm.step(alpha[i], alpha_rate[i], heave[i])
        cl_hist.append(out["CLf"])
        cd_hist.append(out["CDf"])
    cl = np.asarray(cl_hist)[-LDVM_SPC:]
    cd = np.asarray(cd_hist)[-LDVM_SPC:]
    ph = np.arange(N_OUT, dtype=float) / N_OUT
    ph_fine = np.arange(LDVM_SPC, dtype=float) / LDVM_SPC
    cl128 = np.interp(ph, ph_fine, cl, period=1.0)
    cd128 = np.interp(ph, ph_fine, cd, period=1.0)
    harm = case.experimental_filter_harmonic
    cl_f = sharp_fourier_lowpass(cl128, maximum_harmonic=harm)
    cd_f = sharp_fourier_lowpass(cd128, maximum_harmonic=harm)

    gt = {"CL": [], "CD": []}
    with open(repo / "docs/forward_flight_large_pitch/reproductions/"
              "baik2012_w1_w4/runs/20260813_baik2012_w1_w4_full_reproducible/"
              "scored_phase_samples.csv") as f:
        for row in csv.DictReader(f):
            if row["case_id"] == cid:
                gt[row["quantity"]].append((float(row["phase"]),
                                            float(row["experiment"])))
    r = {}
    for q, sim in (("CL", cl_f), ("CD", cd_f)):
        pts = sorted(gt[q])
        gp = np.array([p for p, _ in pts])
        gv = np.array([v for _, v in pts])
        e = np.interp(gp, ph, sim) - gv
        r[q] = float(np.sqrt(np.mean(e ** 2)))
    macro_cl.append(r["CL"]); macro_cd.append(r["CD"])
    print(f"{cid}: 2D LDVM CL RMSE {r['CL']:.4f} | CD RMSE {r['CD']:.4f} "
          f"(chassis+transfer: CL "
          f"{{'W1': 0.516, 'W2': 1.033, 'W3': 0.374, 'W4': 0.708}}[cid])",
          flush=True)
    np.savez(f"/tmp/v5h15-paper/baik_2dldvm_{cid}.npz", phase=ph, cl=cl_f,
             cd=cd_f)

print(f"\n2D LDVM macro: CL {np.mean(macro_cl):.4f} | CD {np.mean(macro_cd):.4f}")
print(f"chassis+transfer macro: CL 0.6577 | CD 0.345 | v4b 0.6575/0.3452")
