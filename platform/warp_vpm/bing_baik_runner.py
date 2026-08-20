"""Baik 2012 W1-W4 runner for the joint LEV solver (bing_joint_ptera).

Drives the same frozen W2 movement builder as baik_runner (bare core), swaps
in JointLEVTEVSolver. Scores the digitized GT (400 phase points per case).

Usage: python bing_baik_runner.py [W1|W2|W3|W4] [lev=on|off]
"""
import csv
import importlib.util
import sys
import time
from dataclasses import replace as _replace
from pathlib import Path

import numpy as np

repo = Path("/tmp/fluxv-v5-nextgen")
sys.path.insert(0, str(repo / "src"))
sys.path.insert(0, str(repo / "platform"))
sys.path.insert(0, str(repo / "platform/warp_vpm"))

CASE_ID = sys.argv[1] if len(sys.argv) > 1 else "W2"
LEV_ON = (sys.argv[2].lower() != "off") if len(sys.argv) > 2 else True
N_CH = int(sys.argv[3]) if len(sys.argv) > 3 else 2
N_SP = int(sys.argv[4]) if len(sys.argv) > 4 else 8
CH_SPACING = sys.argv[5] if len(sys.argv) > 5 else "uniform"


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
from bing_joint_ptera import JointLEVTEVSolver, JointConfig

case = baik.BAIK_2012_CASES[CASE_ID]
executor.W2_CASE = _replace(
    executor.W2_CASE,
    strouhal=case.strouhal,
    reduced_frequency=case.reduced_frequency,
    heave_to_chord=case.heave_to_chord,
    period_s=case.period_s,
)
def build_movement_refined(ps, n_ch, n_sp, ch_spacing, steps_per_cycle=32,
                           cycles=2):
    """Mirror of executor.build_w2_movement with panel counts as arguments."""
    w2 = executor.W2_CASE
    outline = np.asarray(
        [[1.0, 1.0e-4], [0.5, 1.0e-4], [0.0, 0.0], [0.5, -1.0e-4], [1.0, -1.0e-4]])
    airfoil = ps.geometry.airfoil.Airfoil(
        name="Baik-2012-zero-camber-mean-surface-adapter",
        outline_A_lp=outline, resample=False)
    root = ps.geometry.wing_cross_section.WingCrossSection(
        airfoil=airfoil, chord=w2.chord_m, num_spanwise_panels=n_sp,
        spanwise_spacing="cosine")
    tip = ps.geometry.wing_cross_section.WingCrossSection(
        airfoil=airfoil, chord=w2.chord_m,
        Lp_Wcsp_Lpp=(0.0, w2.span_m, 0.0),
        num_spanwise_panels=None, spanwise_spacing=None)
    pivot = w2.pivot_fraction_chord * w2.chord_m
    wing = ps.geometry.wing.Wing(
        name="Baik adapter", wing_cross_sections=[root, tip],
        Ler_Gs_Cgs=(-pivot, 0.0, 0.0),
        angles_Gs_to_Wn_ixyz=(0.0, w2.mean_alpha_deg, 0.0),
        symmetric=False, num_chordwise_panels=n_ch,
        chordwise_spacing=ch_spacing)
    airplane = ps.geometry.airplane.Airplane(
        wings=[wing], name="Baik", s_ref=w2.area_m2, c_ref=w2.chord_m,
        b_ref=w2.span_m)
    secs = [ps.movements.wing_cross_section_movement.WingCrossSectionMovement(
        base_wing_cross_section=s) for s in (root, tip)]
    wm = ps.movements.wing_movement.WingMovement(
        base_wing=wing, wing_cross_section_movements=secs,
        ampLer_Gs_Cgs=(0.0, 0.0, w2.heave_to_chord * w2.chord_m),
        periodLer_Gs_Cgs=(0.0, 0.0, w2.period_s),
        spacingLer_Gs_Cgs=("sine", "sine", executor._w2_heave_spacing),
        phaseLer_Gs_Cgs=(0.0, 0.0, 90.0),
        ampAngles_Gs_to_Wn_ixyz=(0.0, w2.implemented_pitch_amplitude_deg, 0.0),
        periodAngles_Gs_to_Wn_ixyz=(0.0, w2.period_s, 0.0),
        phaseAngles_Gs_to_Wn_ixyz=(0.0, 180.0, 0.0),
        rotationPointOffset_Gs_Ler=(pivot, 0.0, 0.0))
    am = ps.movements.airplane_movement.AirplaneMovement(
        base_airplane=airplane, wing_movements=[wm])
    op = ps.operating_point.OperatingPoint(
        rho=w2.rho_kg_m3, vCg__E=w2.freestream_m_s, alpha=0.0, beta=0.0,
        nu=w2.nu_m2_s)
    opm = ps.movements.operating_point_movement.OperatingPointMovement(
        base_operating_point=op)
    return ps.movements.movement.Movement(
        airplane_movements=[am], operating_point_movement=opm,
        delta_time=w2.period_s / steps_per_cycle, num_cycles=cycles,
        max_wake_cycles=cycles)


if __name__ == "__main__":
    movement = build_movement_refined(pterasoftware, N_CH, N_SP, CH_SPACING)
    problem = pterasoftware.problems.UnsteadyProblem(
        movement=movement, only_final_results=False)

    U = case.reduced_frequency and (
        np.pi * (1.0 / case.period_s) * case.chord_m / case.reduced_frequency)
    qS = 0.5 * case.rho_kg_m3 * U**2 * case.chord_m * case.span_m
    steps_per_cycle = 32
    n_cycles = 2
    period = case.period_s
    dt = period / steps_per_cycle
    n_steps = steps_per_cycle * n_cycles

    cfg = JointConfig(lesp_crit=0.11, lev_start_step=3, enable_lev=LEV_ON)
    solver = JointLEVTEVSolver(problem, cfg)
    t0 = time.perf_counter()
    print(f"Solving {CASE_ID} (LEV {'on' if LEV_ON else 'off'}), "
          f"{n_steps} steps, T={period}s, U={U:.4f} m/s", flush=True)
    solver.run(prescribed_wake=True, calculate_streamlines=False, show_progress=False)
    t1 = time.perf_counter()
    print(f"Solve complete in {t1-t0:.1f}s", flush=True)

    FX, FZ = [], []
    for sp in solver.steady_problems:
        f = sp.airplanes[0].forces_W
        if f is not None:
            FX.append(float(f[0])); FZ.append(float(f[2]))
    FX, FZ = np.array(FX), np.array(FZ)
    steps = len(FZ)
    t = np.arange(steps) * dt
    CL = -FZ / qS
    CD = -FX / qS


    def lowpass(x, cutoff=1.0):
        X = np.fft.rfft(x)
        fr = np.fft.rfftfreq(len(x), t[1] - t[0])
        X[fr > cutoff] = 0.0
        return np.fft.irfft(X, n=len(x))


    CLf, CDf = lowpass(CL), lowpass(CD)
    last = slice(steps - steps_per_cycle, steps)
    phase_sim = (t[last] - t[last][0]) / period

    gt = {"CL": [], "CD": []}
    with open(repo / "docs/forward_flight_large_pitch/reproductions/"
           "baik2012_w1_w4/runs/20260813_baik2012_w1_w4_full_reproducible/"
           "scored_phase_samples.csv") as f:
        for row in csv.DictReader(f):
            if row["case_id"] == CASE_ID:
                gt[row["quantity"]].append((float(row["phase"]), float(row["experiment"])))

    results = {"case": CASE_ID, "lev": LEV_ON, "solve_time_s": t1 - t0}
    for q, sim in (("CL", CLf[last]), ("CD", CDf[last])):
        pts = sorted(gt[q])
        gp = np.array([p for p, _ in pts])
        gv = np.array([v for _, v in pts])
        pred = np.interp(gp, phase_sim, sim)
        err = pred - gv
        results[q] = dict(rmse=float(np.sqrt(np.mean(err**2))),
                          mae=float(np.mean(np.abs(err))),
                          bias=float(np.mean(err)),
                          corr=float(np.corrcoef(pred, gv)[0, 1]))
        print(f"{CASE_ID} {'LEV-on ' if LEV_ON else 'LEV-off'} {q}: "
              f"RMSE={results[q]['rmse']:.3f} MAE={results[q]['mae']:.3f} "
              f"bias={results[q]['bias']:+.3f} corr={results[q]['corr']:+.3f}",
              flush=True)

    d = solver.diag
    print(f"LEV particles: {d[-1]['n_particles']}, "
          f"active strips (max): {max(x['lev_strips'] for x in d)}, "
          f"max|LESP| over run: {max(x['lesp_max'] for x in d):.3f}, "
          f"G_lev total: {sum(x['g_lev'] for x in d):+.2f}")

    np.savez(f"/tmp/v5h15-paper/bing_baik_{CASE_ID}_{'lev' if LEV_ON else 'off'}.npz",
             t=t, CL=CL, CD=CD, CL_f=CLf, CD_f=CDf, phase_sim=phase_sim,
             lesp=np.array([x["lesp_max"] for x in d]),
             g_lev=np.array([x["g_lev"] for x in d]))
    print(f"BING BAIK {CASE_ID} DONE", flush=True)
