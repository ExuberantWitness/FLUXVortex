"""Yang 2025: declared LDVM load correction on the new chassis + LEV-delta test.

Experiment A (viscous/separation term): extract the mid-span section
kinematics from the chassis geometry, drive the frozen generic LDVM pair
(run_ldvm_separation_pair) with the declared Yang threshold, project the
4-ledger delta to finite wing, add to the chassis cycle-mean loads.

Experiment B (can the current LEV approximate it?): run the chassis with
BING-LEV (bing mode) at the same threshold, compute the LEV-induced load
delta, compare sign/magnitude with the LDVM delta and V4B's frozen values.
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
from forward_flight_benchmarks.ldvm_uvlm_correction import (
    LESPThreshold, LDVMSectionSettings, run_ldvm_separation_pair,
    project_ldvm_delta_to_finite_wing)
from bing_joint_ptera import JointLEVTEVSolver, JointConfig

case = Yang2025RigidCase()
g = 9.81
qS = 0.5 * case.rho_kg_m3 * case.freestream_m_s**2 * case.area_m2
U = case.freestream_m_s
c = case.chord_m
period = case.period_s
SPC = 128
LESP_YANG = 0.0872   # frozen crosspaper declared value for Yang

gt = {}
with open(repo / "docs/forward_flight_large_pitch/reproductions/plev2025/"
          "source_data/yang2025_fig11_rigid_digitized.csv") as f:
    for row in csv.DictReader(f):
        gt[float(row["aoa_deg"])] = dict(
            test_lift=float(row["test_lift_gf"]),
            test_drag=-float(row["test_thrust_gf"]))

v4 = {}
v4dir = (repo / "docs/forward_flight_large_pitch/reproductions/"
         "unified_fluxv_v4_ldvm_stevens_20260812/runs/"
         "20260812_fluxv_v4b_crosspaper_full")
with open(v4dir / "yang2025_v4_mean_characteristics.csv") as f:
    for row in csv.DictReader(f):
        v4[float(row["aoa_deg"])] = dict(
            ldvm_dcl=float(row["ldvm_delta_CL"]),
            ldvm_dcd=float(row["ldvm_delta_CD"]))


def extract_kinematics(solver, dt):
    """Mid-span strip geometric alpha and plunge rate from chassis geometry."""
    alphas, heaves = [], []
    prev_mid = None
    for sp_ in solver.steady_problems:
        panels = sp_.airplanes[0].wings[0].panels
        nch, nsp = panels.shape
        s = nsp // 2
        le = np.asarray(panels[0, s].Flpp_GP1_CgP1)
        te = np.asarray(panels[nch - 1, s].Blpp_GP1_CgP1)
        mid = 0.5 * (le + te)
        alphas.append(float(np.arctan2(float(le[2]) - float(te[2]),
                                       float(te[0]) - float(le[0]))))
        heaves.append(0.0 if prev_mid is None
                      else float((mid[2] - prev_mid[2]) / dt))
        prev_mid = mid
    return np.array(alphas), np.array(heaves)


def run_case(aoa, enable_lev, lesp_crit):
    cycles = 2 if enable_lev else 4   # LEV-on: shorter run, O(N^2) tail costs
    movement = build_yang2025_movement(
        aoa, "full", settings=(8, 12, SPC, cycles, cycles))
    if isinstance(movement, tuple):
        movement = movement[0]
    problem = pterasoftware.problems.UnsteadyProblem(
        movement=movement, only_final_results=False)
    cfg = JointConfig(enable_lev=enable_lev, load_mode="bing",
                      lesp_crit=lesp_crit, lev_start_step=SPC + 4)
    solver = JointLEVTEVSolver(problem, cfg)
    solver.run(prescribed_wake=True, calculate_streamlines=False,
               show_progress=False)
    FX = np.array([float(sp_.airplanes[0].forces_W[0])
                   for sp_ in solver.steady_problems])
    FZ = np.array([float(sp_.airplanes[0].forces_W[2])
                   for sp_ in solver.steady_problems])
    alpha, heave = extract_kinematics(solver, movement.delta_time)
    return FX, FZ, alpha, heave


print(f"{'AoA':>4} {'raw lift':>8} {'+LDVM':>7} {'GT':>6} | "
      f"{'raw drag':>8} {'+LDVM':>7} {'GT':>6} | {'dCL_mine':>8} {'dCL_v4b':>8} "
      f"| {'dCD_mine':>8} {'dCD_v4b':>8} | {'LEV dL':>7} {'LEV dD':>7}")
rows_out = []
for aoa in sorted(gt):
    t0 = time.perf_counter()
    print(f"AoA {aoa}: LEV-off run...", flush=True)
    FX, FZ, alpha, heave = run_case(aoa, False, LESP_YANG)
    print(f"AoA {aoa}: LDVM pair + LEV-on run...", flush=True)
    spc = SPC
    cl_raw = float(-np.mean(FZ[-spc:])) / qS
    cd_raw = float(-np.mean(FX[-spc:])) / qS  # drag = -thrust

    # last cycle kinematics, periodic
    a1 = alpha[-spc:].copy()
    h1 = heave[-spc:].copy()
    dt_conv = U * period / c / spc
    a_rate = np.gradient(a1, dt_conv, edge_order=2)
    threshold = LESPThreshold(
        value=LESP_YANG, section_family="rounded flat plate",
        reynolds=case.reynolds,
        source="frozen crosspaper Yang declared value",
        source_role="published transfer")
    settings = LDVMSectionSettings(ndiv=32, naterm=14, max_wake_steps=256,
                                   core_radius_chord=0.02)
    pair = run_ldvm_separation_pair(
        alpha_rad=np.tile(a1, 3), alpha_rate_per_convective_time=np.tile(a_rate, 3),
        heave_rate_over_u=np.tile(h1, 3), delta_time_convective=dt_conv,
        pivot_fraction_chord=0.25, threshold=threshold, settings=settings)
    sel = slice(2 * spc, 3 * spc)
    proj = project_ldvm_delta_to_finite_wing(
        np.asarray(pair["delta"]["CNc"])[sel], np.asarray(pair["delta"]["CNnc"])[sel],
        np.asarray(pair["delta"]["CNnonl"])[sel], np.asarray(pair["delta"]["CSf"])[sel],
        a1, aspect_ratio=case.aspect_ratio)
    d_cl = float(np.mean(proj["delta_CL"]))
    d_cd = float(np.mean(proj["delta_CD"]))

    lift_corr = (cl_raw + d_cl) * qS / g * 1000.0
    drag_corr = (cd_raw + d_cd) * qS / g * 1000.0

    # Experiment B: LEV-on run
    FX2, FZ2, _, _ = run_case(aoa, True, LESP_YANG)
    lev_dlift = float(-np.mean(FZ2[-spc:]) + np.mean(FZ[-spc:])) / g * 1000.0
    lev_ddrag = float(-np.mean(FX2[-spc:]) + np.mean(FX[-spc:])) / g * 1000.0

    d = gt[aoa]
    rows_out.append(dict(aoa=aoa, lift_raw=-np.mean(FZ[-spc:]) / g * 1000,
                         lift_corr=lift_corr, drag_corr=drag_corr,
                         d_cl=d_cl, d_cd=d_cd, lev_dlift=lev_dlift,
                         lev_ddrag=lev_ddrag))
    print(f"{aoa:4.0f} {-np.mean(FZ[-spc:])/g*1000:8.1f} {lift_corr:7.1f} "
          f"{d['test_lift']:6.1f} | {-np.mean(FX[-spc:])/g*1000:8.1f} "
          f"{drag_corr:7.1f} {d['test_drag']:6.1f} | {d_cl:8.4f} "
          f"{v4[aoa]['ldvm_dcl']:8.4f} | {d_cd:8.4f} {v4[aoa]['ldvm_dcd']:8.4f} "
          f"| {lev_dlift:7.1f} {lev_ddrag:7.1f} [{time.perf_counter()-t0:.0f}s]",
          flush=True)

lm = [abs(r["lift_corr"] - gt[r["aoa"]]["test_lift"]) for r in rows_out]
dm = [abs(r["drag_corr"] - gt[r["aoa"]]["test_drag"]) for r in rows_out]
print(f"\ncorrected lift MAE: {np.mean(lm):.2f} gf (was 6.8, v4b 4.55) | "
      f"drag MAE: {np.mean(dm):.2f} gf (was 12.95, v4b 2.64)")
print(f"LEV-delta vs LDVM-delta: see columns above (same-sign = approximable)")
