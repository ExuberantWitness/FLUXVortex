"""Mancini 2017 with the latest mechanism-based algorithm.

Our chassis (machine-precision bare pterasoftware, 8×12 cosine mesh,
128 steps/chord) vs the frozen v4b baseline (4×12 uniform, 64 steps/chord).

Three variants:
  A) chassis bare (high-resolution baseline)
  B) chassis + frozen LDVM delta @ Lcrit=0.11 (same as v4b)
  C) chassis + frozen LDVM delta @ Lcrit=0.239 (rounded-family rule)
"""
import sys
import time
from pathlib import Path

import numpy as np

repo = Path("/tmp/fluxv-v5-nextgen")
sys.path.insert(0, str(repo / "src"))
sys.path.insert(0, str(repo / "platform"))
sys.path.insert(0, str(repo / "platform/warp_vpm"))

import pterasoftware
from forward_flight_benchmarks.mancini2017 import (
    MANCINI_2017_CASES, mancini_pitch_angle_deg,
    load_mancini_fig4_13b_experiment)
from forward_flight_benchmarks.ldvm_uvlm_correction import (
    LESPThreshold, LDVMSectionSettings, run_ldvm_separation_pair,
    project_ldvm_delta_to_finite_wing)
from bing_joint_ptera import JointLEVTEVSolver, JointConfig

N_OUT = 501
LDVM_SPC = 96   # steps per chord for the 2D LDVM (same as v4b)


def build_chassis_movement(case, nc=8, ns=12, spc=128):
    """Our high-resolution movement, mirroring the frozen builder."""
    airfoil = ps_airfoil = pterasoftware.geometry.airfoil.Airfoil(
        name="naca0012")
    root = pterasoftware.geometry.wing_cross_section.WingCrossSection(
        airfoil=ps_airfoil, chord=case.chord_m, num_spanwise_panels=ns,
        spanwise_spacing="cosine",
        control_surface_symmetry_type="symmetric")
    tip = pterasoftware.geometry.wing_cross_section.WingCrossSection(
        airfoil=ps_airfoil, chord=case.chord_m,
        Lp_Wcsp_Lpp=(0.0, case.semispan_m, 0.0), num_spanwise_panels=None,
        spanwise_spacing=None, control_surface_symmetry_type="symmetric")

    from forward_flight_benchmarks.mancini2017 import (
        mancini_periodic_pitch_spacing)
    wing = pterasoftware.geometry.wing.Wing(
        name="Mancini chassis", wing_cross_sections=[root, tip],
        symmetric=True, symmetryNormal_G=(0.0, 1.0, 0.0),
        symmetryPoint_G_Cg=(0.0, 0.0, 0.0),
        num_chordwise_panels=nc, chordwise_spacing="cosine")
    airplane = pterasoftware.geometry.airplane.Airplane(
        wings=[wing], name="Mancini chassis",
        s_ref=case.area_m2, c_ref=case.chord_m, b_ref=case.span_m)
    secs = [pterasoftware.movements.wing_cross_section_movement
            .WingCrossSectionMovement(base_wing_cross_section=s)
            for s in (root, tip)]
    wm = pterasoftware.movements.wing_movement.WingMovement(
        base_wing=wing, wing_cross_section_movements=secs,
        ampAngles_Gs_to_Wn_ixyz=(0.0, case.maximum_pitch_deg, 0.0),
        periodAngles_Gs_to_Wn_ixyz=(0.0, case.waveform_period_s, 0.0),
        spacingAngles_Gs_to_Wn_ixyz=(
            "sine", lambda ph: mancini_periodic_pitch_spacing(ph, case),
            "sine"),
        phaseAngles_Gs_to_Wn_ixyz=(0.0, 0.0, 0.0),
        rotationPointOffset_Gs_Ler=(0.0, 0.0, 0.0))
    am = pterasoftware.movements.airplane_movement.AirplaneMovement(
        base_airplane=airplane, wing_movements=[wm])
    op = pterasoftware.operating_point.OperatingPoint(
        rho=case.rho_kg_m3, vCg__E=case.freestream_m_s, alpha=0.0, beta=0.0,
        nu=case.nu_m2_s)
    opm = pterasoftware.movements.operating_point_movement.\
        OperatingPointMovement(base_operating_point=op)
    dt = case.convective_time_s / spc
    steps = int(np.ceil((case.warmup_chords + case.observation_chords) * spc)) + 1
    return pterasoftware.movements.movement.Movement(
        airplane_movements=[am], operating_point_movement=opm,
        delta_time=dt, num_steps=steps,
        max_wake_rows=int(np.ceil(6.0 * spc)))


def run_case(case, exp_cl):
    t0 = time.perf_counter()
    movement = build_chassis_movement(case)
    problem = pterasoftware.problems.UnsteadyProblem(
        movement=movement, only_final_results=False)
    solver = JointLEVTEVSolver(problem, JointConfig(enable_lev=False))
    solver.run(prescribed_wake=True, calculate_streamlines=False,
               show_progress=False)

    q_area = 0.5 * case.rho_kg_m3 * case.freestream_m_s**2 * case.area_m2
    FZ = np.array([float(sp_.airplanes[0].forces_W[2])
                   for sp_ in solver.steady_problems])
    FX = np.array([float(sp_.airplanes[0].forces_W[0])
                   for sp_ in solver.steady_problems])
    dt = movement.delta_time
    t_star_all = np.arange(len(FZ)) * dt / case.convective_time_s \
        - case.warmup_chords
    valid = (t_star_all >= -1e-10) & (t_star_all <= case.observation_chords + 1e-10)
    target = np.linspace(0, case.observation_chords, N_OUT)
    cl_bare = np.interp(target, t_star_all[valid], (-FZ / q_area)[valid])

    # LDVM delta at two crit values
    results = {"bare": cl_bare}
    for crit, tag in ((0.11, "v4b_crit"), (0.239, "rounded_crit")):
        dtc = 1.0 / LDVM_SPC
        total_chords = case.warmup_chords + case.observation_chords
        ts = np.arange(0.0, total_chords + 0.5 * dtc, dtc) - case.warmup_chords
        alpha = np.deg2rad(mancini_pitch_angle_deg(ts, case))
        arate = np.gradient(alpha, dtc, edge_order=2)
        thr = LESPThreshold(
            value=crit, section_family="5% rounded plate",
            reynolds=case.reynolds,
            source=f"declared Lcrit={crit}",
            source_role="published transfer hypothesis")
        pair = run_ldvm_separation_pair(
            alpha_rad=alpha, alpha_rate_per_convective_time=arate,
            heave_rate_over_u=np.zeros_like(alpha),
            delta_time_convective=dtc,
            pivot_fraction_chord=case.pivot_fraction_chord,
            threshold=thr,
            settings=LDVMSectionSettings(
                ndiv=50, naterm=24, core_radius_time_step_ratio=1.3,
                max_wake_steps=ts.size))
        v = (ts >= -1e-10) & (ts <= case.observation_chords + 1e-10)
        angle = np.interp(target, ts[v], alpha[v])

        def sample(name):
            return np.interp(target, ts[v],
                             np.asarray(pair["delta"][name], dtype=float)[v])
        proj = project_ldvm_delta_to_finite_wing(
            sample("CNc"), sample("CNnc"), sample("CNnonl"), sample("CSf"),
            angle, aspect_ratio=case.aspect_ratio)
        results[tag] = cl_bare + np.asarray(proj["delta_CL"])

    # score
    t_exp = np.linspace(0, 13.0, 1301)
    mask = t_exp <= case.observation_chords
    gt = np.interp(target, t_exp[mask], exp_cl[mask])
    scores = {}
    for k, sim in results.items():
        scores[k] = float(np.sqrt(np.mean((sim - gt) ** 2)))
    print(f"  bare RMSE {scores['bare']:.4f} | "
          f"+LDVM@0.11 {scores['v4b_crit']:.4f} | "
          f"+LDVM@0.239 {scores['rounded_crit']:.4f} "
          f"[{time.perf_counter()-t0:.0f}s]", flush=True)
    return results, scores, target, gt


exp = load_mancini_fig4_13b_experiment()
print("=== Mancini 2017 — chassis 8x12cos/128 + LDVM delta ===")
all_scores = {}
for cid in ("fast_pitch", "slow_pitch"):
    case = MANCINI_2017_CASES[cid]
    print(f"\n{cid} (s_a/c={case.acceleration_distance_chords}, "
          f"k={case.reduced_pitch_rate}):")
    exp_cl = exp[f"CL_{cid}"]
    results, scores, target, gt = run_case(case, exp_cl)
    all_scores[cid] = scores
    np.savez(f"/tmp/v5h15-paper/mancini_{cid}.npz",
             t_star=target, gt=gt, **results)

print("\n=== Summary ===")
print(f"{'case':>12} {'bare':>8} {'@0.11':>8} {'@0.239':>8} "
      f"| v4b(frozen) | uvlm(frozen)")
print(f"{'fast_pitch':>12} {all_scores['fast_pitch']['bare']:8.4f} "
      f"{all_scores['fast_pitch']['v4b_crit']:8.4f} "
      f"{all_scores['fast_pitch']['rounded_crit']:8.4f} "
      f"|    1.2184    |    1.4004")
print(f"{'slow_pitch':>12} {all_scores['slow_pitch']['bare']:8.4f} "
      f"{all_scores['slow_pitch']['v4b_crit']:8.4f} "
      f"{all_scores['slow_pitch']['rounded_crit']:8.4f} "
      f"|    0.2908    |    0.3778")
