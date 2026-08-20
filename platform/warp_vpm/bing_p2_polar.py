"""P2: frozen full-angle polar residual on the chassis (Yang + Izra).

movement_polar_residual is movement-generic; augment adds the geometry-derived
full-angle polar delta to the chassis baseline. Reports bare vs +polar vs
+polar+T3(dynamic viscous) for attribution.
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
from forward_flight_benchmarks.ptera_adapter import (
    build_yang2025_movement, build_izraelevitz_scherer_movement)
from forward_flight_benchmarks.cases import Yang2025RigidCase, IzraelevitzSchererCase
from forward_flight_benchmarks.uvlm_polar_correction import (
    DEFAULT_POLAR_PARAMETERS, augment_uvlm_history, movement_polar_residual)
from bing_joint_ptera import JointLEVTEVSolver, JointConfig
from bing_drag_ledger import LedgerConfig, run_ledger

SPC = 128
N_OUT = 128
g = 9.81


def chassis_cycle(movement):
    problem = pterasoftware.problems.UnsteadyProblem(
        movement=movement, only_final_results=False)
    solver = JointLEVTEVSolver(problem, JointConfig(enable_lev=False))
    solver.run(prescribed_wake=True, calculate_streamlines=False,
               show_progress=False)
    FX = np.array([float(sp_.airplanes[0].forces_W[0])
                   for sp_ in solver.steady_problems])
    FZ = np.array([float(sp_.airplanes[0].forces_W[2])
                   for sp_ in solver.steady_problems])
    return solver, FX, FZ


def phase_hist(FX, FZ, qS, period, ledger=None, cfg_led=None):
    dt = period / SPC
    t = np.arange(len(FX)) * dt

    def lowpass(x):
        X = np.fft.rfft(x)
        fr = np.fft.rfftfreq(len(x), dt)
        X[fr > 1.0] = 0.0
        return np.fft.irfft(X, n=len(x))

    last = slice(len(FX) - SPC, len(FX))
    phase_raw = (t[last] - t[last][0]) / period
    phase = np.arange(N_OUT) / N_OUT
    cl = np.interp(phase, phase_raw, lowpass(-FZ / qS)[last], period=1.0)
    cd = np.interp(phase, phase_raw, lowpass(-FX / qS)[last], period=1.0)
    led_extra = 0.0
    if ledger is not None and cfg_led is not None:
        led = run_ledger(ledger, cfg_led, last_n=SPC)
        led_extra = led["mean_t3_N"] / qS   # T3 only (dynamic viscous)
    return phase, cl, cd, led_extra


# ---------------- Yang ----------------
print("=== P2 Yang: bare vs +polar vs +polar+T3 ===")
case = Yang2025RigidCase()
qS = 0.5 * case.rho_kg_m3 * case.freestream_m_s**2 * case.area_m2
CD0 = 2.0 * 1.328 / np.sqrt(case.reynolds)
gt = {}
with open(repo / "docs/forward_flight_large_pitch/reproductions/plev2025/"
          "source_data/yang2025_fig11_rigid_digitized.csv") as f:
    for row in csv.DictReader(f):
        gt[float(row["aoa_deg"])] = dict(
            test_lift=float(row["test_lift_gf"]),
            test_drag=-float(row["test_thrust_gf"]))

errs = {k: [] for k in ("lift_bare", "lift_pol", "drag_bare", "drag_pol",
                        "drag_pol_t3")}
for aoa in sorted(gt):
    movement = build_yang2025_movement(aoa, "full", settings=(8, 12, SPC, 4, 4))
    if isinstance(movement, tuple):
        movement = movement[0]
    solver, FX, FZ = chassis_cycle(movement)
    phase, cl, cd, t3 = phase_hist(
        FX, FZ, qS, case.period_s, solver.ledger,
        LedgerConfig(lesp_crit=0.0872, aspect_ratio=case.aspect_ratio,
                     rho=case.rho_kg_m3, cd0=CD0, enable_t1=False))
    baseline = {
        "phase": phase, "lift_n": cl * qS, "thrust_n": -cd * qS,
        "mean_lift_n": float(np.mean(cl * qS)),
        "mean_thrust_n": float(np.mean(-cd * qS)),
        "source_cycle_step_range": [len(FX) - SPC, len(FX) - 1],
    }
    polar = movement_polar_residual(
        movement, source_cycle_step_range=baseline["source_cycle_step_range"],
        period_s=case.period_s, freestream_m_s=case.freestream_m_s,
        rho_kg_m3=case.rho_kg_m3, aspect_ratio=case.aspect_ratio,
        output_samples=N_OUT, parameters=DEFAULT_POLAR_PARAMETERS)
    aug = augment_uvlm_history(baseline, polar, rho_kg_m3=case.rho_kg_m3,
                               freestream_m_s=case.freestream_m_s,
                               area_m2=case.area_m2)
    lift_bare = float(np.mean(cl)) * qS / g * 1000
    lift_pol = float(np.asarray(aug["CL"]).mean()) * qS / g * 1000
    drag_bare = float(np.mean(cd)) * qS / g * 1000
    drag_pol = float(np.asarray(aug["CD"]).mean()) * qS / g * 1000
    d = gt[aoa]
    errs["lift_bare"].append(abs(lift_bare - d["test_lift"]))
    errs["lift_pol"].append(abs(lift_pol - d["test_lift"]))
    errs["drag_bare"].append(abs(drag_bare - d["test_drag"]))
    errs["drag_pol"].append(abs(drag_pol - d["test_drag"]))
    errs["drag_pol_t3"].append(abs(drag_pol + t3 * qS / g * 1000 - d["test_drag"]))
    print(f"  AoA {aoa:.0f}: lift {lift_bare:6.1f}->{lift_pol:6.1f} "
          f"(GT {d['test_lift']:5.1f}) | drag {drag_bare:6.1f}->"
          f"{drag_pol:6.1f}+T3 {t3*qS/g*1000:.1f} (GT {d['test_drag']:5.1f})",
          flush=True)
print(f"MAE lift: bare {np.mean(errs['lift_bare']):.2f} -> polar "
      f"{np.mean(errs['lift_pol']):.2f} (v4b 4.55)")
print(f"MAE drag: bare {np.mean(errs['drag_bare']):.2f} -> polar "
      f"{np.mean(errs['drag_pol']):.2f} -> +T3 {np.mean(errs['drag_pol_t3']):.2f} "
      f"(v4b 2.64)\n")

# ---------------- Izra ----------------
print("=== P2 Izra: bare vs +polar vs +polar+T3 (CT) ===")
case = IzraelevitzSchererCase()
b = case.aspect_ratio * case.chord_m
qS = 0.5 * case.rho_kg_m3 * case.freestream_m_s**2 * case.chord_m * b
gt_rows = []
with open(repo / "docs/forward_flight_large_pitch/reproductions/"
          "unified_fluxv_upgrade_20260812/source_data/"
          "izraelevitz2017_fig14_digitized.csv") as fh:
    for row in csv.DictReader(fh):
        if row["data_role"] == "experimental_observation":
            gt_rows.append(row)
conditions = sorted(set((float(r["theta_max_deg"]), float(r["phase_offset_deg"]))
                        for r in gt_rows))
e_bare, e_pol, e_pol_t3 = [], [], []
for th, ps in conditions:
    movement = build_izraelevitz_scherer_movement(th, ps, "full",
                                                  settings=(8, 12, SPC, 4))
    if isinstance(movement, tuple):
        movement = movement[0]
    solver, FX, FZ = chassis_cycle(movement)
    period = 1.0 / 5.0
    phase, cl, cd, t3 = phase_hist(
        FX, FZ, qS, period, solver.ledger,
        LedgerConfig(lesp_crit=0.239, aspect_ratio=case.aspect_ratio,
                     rho=case.rho_kg_m3, cd0=0.057, enable_t1=False))
    baseline = {
        "phase": phase, "lift_n": cl * qS, "thrust_n": -cd * qS,
        "mean_lift_n": float(np.mean(cl * qS)),
        "mean_thrust_n": float(np.mean(-cd * qS)),
        "source_cycle_step_range": [len(FX) - SPC, len(FX) - 1],
    }
    polar = movement_polar_residual(
        movement, source_cycle_step_range=baseline["source_cycle_step_range"],
        period_s=period, freestream_m_s=case.freestream_m_s,
        rho_kg_m3=case.rho_kg_m3, aspect_ratio=case.aspect_ratio,
        output_samples=N_OUT, parameters=DEFAULT_POLAR_PARAMETERS)
    aug = augment_uvlm_history(baseline, polar, rho_kg_m3=case.rho_kg_m3,
                               freestream_m_s=case.freestream_m_s,
                               area_m2=case.chord_m * b)
    ct_bare = -float(np.mean(cd))
    ct_pol = -float(np.asarray(aug["CD"]).mean())
    match = [r for r in gt_rows if float(r["theta_max_deg"]) == th
             and float(r["phase_offset_deg"]) == ps]
    for r in match:
        gct = float(r["ct"])
        e_bare.append(abs(ct_bare - gct))
        e_pol.append(abs(ct_pol - gct))
        e_pol_t3.append(abs(ct_pol + t3 - gct))
    print(f"  t={th:.0f} p={ps:.0f}: CT {ct_bare:+.4f} -> polar {ct_pol:+.4f} "
          f"+T3 {t3:+.4f} (GT {float(match[0]['ct']):+.4f})", flush=True)
print(f"Izra CT MAE: bare {np.mean(e_bare):.4f} -> polar {np.mean(e_pol):.4f} "
      f"-> +T3 {np.mean(e_pol_t3):.4f} (v4b 0.0198, ledger-T1T3 0.0260)")
