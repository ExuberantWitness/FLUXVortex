"""Izra fix: frozen V4B LDVM-delta recipe applied to OUR chassis baseline.

Recipe (run_v4_crosspaper.py:_fig14_phase_diagnostic, verbatim):
  alpha = theta*cos(phi+psi); alpha_dot = -theta*omega* sin(phi+psi)
  heave_rate = -(h/c)*omega*sin(phi);  omega* = pi*St/(h/c)
  LESP threshold = sin(CLmax/CLalpha) = sin(0.90/0.065) = 0.2393 (Scherer
  static polar, declared)
  LDVM pair ndiv=50 naterm=24, 4 cycles, last cycle; project AR=3;
  delta_CT = -delta_CD.
Our CT = chassis_raw - 0.057 (declared Cd0) + delta_CT. T1 dropped (the
pair's CSf ledger already contains the suction loss; keeping both would
double-count).
"""
import csv
import sys
from pathlib import Path

import numpy as np

repo = Path("/tmp/fluxv-v5-nextgen")
sys.path.insert(0, str(repo / "src"))
sys.path.insert(0, str(repo / "platform"))
sys.path.insert(0, str(repo / "platform/warp_vpm"))

import pterasoftware
from forward_flight_benchmarks.ptera_adapter import build_izraelevitz_scherer_movement
from forward_flight_benchmarks.cases import IzraelevitzSchererCase
from forward_flight_benchmarks.ldvm_uvlm_correction import (
    LESPThreshold, LDVMSectionSettings, run_ldvm_separation_pair,
    project_ldvm_delta_to_finite_wing)
from bing_joint_ptera import JointLEVTEVSolver, JointConfig

case = IzraelevitzSchererCase()
SPC = 128
bw = case.aspect_ratio * case.chord_m
qS = 0.5 * case.rho_kg_m3 * case.freestream_m_s**2 * case.chord_m * bw
omega_star = np.pi * case.strouhal / case.heave_to_chord
period_star = 2.0 * np.pi / omega_star

alpha_stall = np.deg2rad(0.90 / 0.065)
threshold = LESPThreshold(
    value=float(np.sin(alpha_stall)), section_family=case.section_name,
    reynolds=case.freestream_m_s * case.chord_m / case.nu_m2_s,
    source="Scherer static CLa=0.065/deg CLmax=0.90; Lcrit=sin(CLmax/CLa)",
    source_role="static-polar-derived hypothesis; no Figure-14 force fit")

gt_rows = []
with open(repo / "docs/forward_flight_large_pitch/reproductions/"
          "unified_fluxv_upgrade_20260812/source_data/"
          "izraelevitz2017_fig14_digitized.csv") as fh:
    for row in csv.DictReader(fh):
        if row["data_role"] == "experimental_observation":
            gt_rows.append(row)
v4 = {}
v4dir = (repo / "docs/forward_flight_large_pitch/reproductions/"
         "unified_fluxv_v4_ldvm_stevens_20260812/runs/"
         "20260812_fluxv_v4b_crosspaper_full")
with open(v4dir / "izraelevitz2017_fig14_v4_mean_thrust.csv") as f:
    for row in csv.DictReader(f):
        v4[(float(row["theta_max_deg"]), float(row["phase_offset_deg"]))] = \
            float(row["v4_CT"])

errs_new, errs_ledger = [], []
print(f"{'cond':>10} {'raw':>8} {'delta':>8} {'final':>8} {'GT':>8} {'v4b':>8} "
      f"{'d_ours':>7} {'d_v4b':>7}")
results = {}
for th, ps in sorted(set((float(r["theta_max_deg"]),
                          float(r["phase_offset_deg"])) for r in gt_rows)):
    movement = build_izraelevitz_scherer_movement(th, ps, "full",
                                                  settings=(8, 12, SPC, 4))
    if isinstance(movement, tuple):
        movement = movement[0]
    problem = pterasoftware.problems.UnsteadyProblem(
        movement=movement, only_final_results=False)
    solver = JointLEVTEVSolver(problem, JointConfig(enable_lev=False))
    solver.run(prescribed_wake=True, calculate_streamlines=False,
               show_progress=False)
    FX = np.array([float(sp_.airplanes[0].forces_W[0])
                   for sp_ in solver.steady_problems])
    ct_raw = float(np.mean(FX[-SPC:])) / qS

    # frozen LDVM-delta recipe (verbatim from _fig14_phase_diagnostic)
    phase = np.arange(4 * SPC) * 2.0 * np.pi / SPC
    alpha = np.deg2rad(th) * np.cos(phase + np.deg2rad(ps))
    alpha_rate = -np.deg2rad(th) * omega_star * np.sin(phase + np.deg2rad(ps))
    heave_rate = -case.heave_to_chord * omega_star * np.sin(phase)
    pair = run_ldvm_separation_pair(
        alpha_rad=alpha, alpha_rate_per_convective_time=alpha_rate,
        heave_rate_over_u=heave_rate,
        delta_time_convective=period_star / SPC,
        pivot_fraction_chord=case.pivot_fraction_chord,
        threshold=threshold,
        settings=LDVMSectionSettings(ndiv=50, naterm=24,
                                     max_wake_steps=4 * SPC))
    sel = slice(3 * SPC, 4 * SPC)
    proj = project_ldvm_delta_to_finite_wing(
        np.asarray(pair["delta"]["CNc"])[sel],
        np.asarray(pair["delta"]["CNnc"])[sel],
        np.asarray(pair["delta"]["CNnonl"])[sel],
        np.asarray(pair["delta"]["CSf"])[sel],
        alpha[sel], aspect_ratio=case.aspect_ratio)
    delta_ct = -float(np.mean(proj["delta_CD"]))

    ct_final = ct_raw - 0.057 + delta_ct
    v4ct = v4[(th, ps)]
    match = [(float(r["ct"])) for r in gt_rows
             if float(r["theta_max_deg"]) == th
             and float(r["phase_offset_deg"]) == ps]
    for gct in match:
        errs_new.append(abs(ct_final - gct))
        errs_ledger.append(abs(v4ct - gct))
    results[(th, ps)] = dict(raw=ct_raw, delta=delta_ct, final=ct_final)
    print(f"{th:4.0f}/{ps:3.0f} {ct_raw:+8.4f} {delta_ct:+8.4f} "
          f"{ct_final:+8.4f} {match[0]:+8.4f} {v4ct:+8.4f} "
          f"{abs(ct_final-match[0]):7.4f} {abs(v4ct-match[0]):7.4f}",
          flush=True)

print(f"\nIzra CT MAE: ours(chassis+frozen delta) {np.mean(errs_new):.4f} | "
      f"V4B {np.mean(errs_ledger):.4f} | prior best (ledger) 0.0260")
import json
Path("/tmp/v5h15-paper/izra_v2.json").write_text(json.dumps(
    {f"{k[0]}/{k[1]}": v for k, v in results.items()}, indent=2))
