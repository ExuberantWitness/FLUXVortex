"""
Flapping wing validation: VPM-wake vs Ring-wake vs Theory.

Tests the UVPMHybridSolver against:
  1. PteraSoftware ring-wake UVLM (3D baseline)
  2. Theodorsen theory (2D analytical, high-AR approximation)

Cases:
  - Plunging wing at reduced frequencies k = 0.1, 0.2, 0.5
  - Static wing at AoA = 5° (baseline)
"""
import sys
import os
import time
import numpy as np
from scipy.special import hankel2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

if not hasattr(np, 'trapezoid'):
    np.trapezoid = np.trapz

import pterasoftware as ps

ps.set_up_logging(level="Warning")

from fluxvortex.solver import UVPMHybridSolver


def theodorsen_function(k):
    """C(k) = H1^(2)(k) / (H1^(2)(k) + i*H0^(2)(k))"""
    if k < 1e-10:
        return 1.0 + 0j
    H0 = hankel2(0, k)
    H1 = hankel2(1, k)
    return H1 / (H1 + 1j * H0)


def theodorsen_cl_plunge(k, h0_over_c, omega, t, chord=1.0):
    """
    CL(t) for pure plunging h(t) = h0*sin(wt), h positive UP.

    CL = πcḧ/(2U²) - 2π C(k) ḣ/U

    Non-circulatory (apparent mass): πcḧ/(2U²)
    Circulatory (Theodorsen): -2π C(k) ḣ/U
      When ḣ>0 (wing moving UP), effective AoA is negative → negative CL.
    """
    U = omega * chord / (2 * k) if k > 1e-10 else 1e10
    h0 = h0_over_c * chord
    h_dot = h0 * omega * np.cos(omega * t)
    h_ddot = -h0 * omega**2 * np.sin(omega * t)
    C = theodorsen_function(k)
    CL = np.pi * h_ddot * chord / (2 * U**2) - 2 * np.pi * C * h_dot / U
    return np.real(CL)


def make_wing(chord=1.0, half_span=5.0, nc=10, ns=6):
    """Create NACA 0012 rectangular wing."""
    return ps.geometry.wing.Wing(
        name="Wing", symmetric=True,
        symmetryNormal_G=(0.0, 1.0, 0.0), symmetryPoint_G_Cg=(0.0, 0.0, 0.0),
        num_chordwise_panels=nc, chordwise_spacing="uniform",
        wing_cross_sections=[
            ps.geometry.wing_cross_section.WingCrossSection(
                num_spanwise_panels=ns, spanwise_spacing="uniform", chord=chord,
                Lp_Wcsp_Lpp=(0, 0, 0), airfoil=ps.geometry.airfoil.Airfoil(name="naca0012"),
                control_surface_symmetry_type="symmetric",
                control_surface_hinge_point=0.75, control_surface_deflection=0.0),
            ps.geometry.wing_cross_section.WingCrossSection(
                num_spanwise_panels=None, spanwise_spacing=None, chord=chord,
                Lp_Wcsp_Lpp=(0, half_span, 0), airfoil=ps.geometry.airfoil.Airfoil(name="naca0012"),
                control_surface_symmetry_type="symmetric",
                control_surface_hinge_point=0.75, control_surface_deflection=0.0),
        ])


def make_plunge_movement(wing, h0, period, V_inf=10.0):
    """Create sinusoidal plunging movement."""
    airplane = ps.geometry.airplane.Airplane(wings=[wing], name="P")
    op = ps.operating_point.OperatingPoint(rho=1.225, vCg__E=V_inf, alpha=0.0, beta=0.0)
    wcs_mov = [ps.movements.wing_cross_section_movement.WingCrossSectionMovement(
        base_wing_cross_section=wcs) for wcs in wing.wing_cross_sections]
    wm = ps.movements.wing_movement.WingMovement(
        base_wing=wing, wing_cross_section_movements=wcs_mov,
        ampLer_Gs_Cgs=(0, 0, h0), periodLer_Gs_Cgs=(0, 0, period),
        spacingLer_Gs_Cgs=("sine", "sine", "sine"), phaseLer_Gs_Cgs=(0, 0, 0))
    am = ps.movements.airplane_movement.AirplaneMovement(
        base_airplane=airplane, wing_movements=[wm])
    opm = ps.movements.operating_point_movement.OperatingPointMovement(base_operating_point=op)
    return airplane, ps.movements.movement.Movement(
        airplane_movements=[am], operating_point_movement=opm,
        num_cycles=3, delta_time=period / 50)


def extract_cl(solver, movement):
    """Extract CL time history."""
    num_steps = solver.num_steps
    first = solver.unsteady_problem.first_results_step
    dt = movement.delta_time
    times, cl = [], []
    for step in range(first, num_steps):
        for airplane in solver.steady_problems[step].airplanes:
            c = airplane.forceCoefficients_W
            if c is not None:
                times.append(step * dt)
                cl.append(-c[2])
    return np.array(times), np.array(cl)


def run_case(label, k, h0_over_c=0.1, V_inf=10.0, chord=1.0):
    """Run ring-wake + VPM-wake for one reduced frequency."""
    omega = 2 * k * V_inf / chord
    period = 2 * np.pi / omega
    h0 = h0_over_c * chord

    print(f"\n--- {label} (k={k:.2f}) ---")

    # Ring-wake
    wing1 = make_wing(chord=chord)
    airplane1, mv1 = make_plunge_movement(wing1, h0, period, V_inf)
    print(f"  Steps: {mv1.num_steps}, dt: {mv1.delta_time:.5f}s, Period: {period:.3f}s")

    t0 = time.perf_counter()
    prob1 = ps.problems.UnsteadyProblem(movement=mv1, only_final_results=False)
    sol1 = ps.unsteady_ring_vortex_lattice_method.UnsteadyRingVortexLatticeMethodSolver(unsteady_problem=prob1)
    sol1.run(prescribed_wake=True, calculate_streamlines=False, show_progress=False)
    ring_time = time.perf_counter() - t0
    t_r, cl_r = extract_cl(sol1, mv1)

    # VPM-wake
    wing2 = make_wing(chord=chord)
    airplane2, mv2 = make_plunge_movement(wing2, h0, period, V_inf)

    t0 = time.perf_counter()
    prob2 = ps.problems.UnsteadyProblem(movement=mv2, only_final_results=False)
    sol2 = UVPMHybridSolver(unsteady_problem=prob2, max_particles=50000, nu=0.0, rlxf=0.3)
    sol2.run(prescribed_wake=False, calculate_streamlines=False, show_progress=False)
    vpm_time = time.perf_counter() - t0
    t_v, cl_v = extract_cl(sol2, mv2)

    # Theodorsen theory
    cl_theo_r = theodorsen_cl_plunge(k, h0_over_c, omega, t_r, chord)
    cl_theo_v = theodorsen_cl_plunge(k, h0_over_c, omega, t_v, chord)

    # Discard first 2 cycles
    t_trans = 2 * period
    mr = t_r > t_trans
    mv = t_v > t_trans
    cl_rs, t_rs = cl_r[mr], t_r[mr]
    cl_vs, t_vs = cl_v[mv], t_v[mv]
    cl_tr = cl_theo_r[mr]
    cl_tv = cl_theo_v[mv]

    if len(cl_rs) < 5 or len(cl_vs) < 5:
        print(f"  Not enough data (ring={len(cl_rs)}, vpm={len(cl_vs)})")
        return None

    # Amplitudes
    ring_amp = (np.max(cl_rs) - np.min(cl_rs)) / 2
    vpm_amp = (np.max(cl_vs) - np.min(cl_vs)) / 2
    theo_amp = (np.max(cl_tr) - np.min(cl_tr)) / 2

    # Ring vs Theodorsen
    r_theo_ratio = ring_amp / theo_amp if theo_amp > 1e-10 else 0
    r_theo_corr = np.corrcoef(cl_rs, cl_tr)[0, 1]

    # VPM vs Ring
    n = min(len(cl_rs), len(cl_vs))
    vr_corr = np.corrcoef(cl_rs[:n], cl_vs[:n])[0, 1]
    vr_amp_ratio = vpm_amp / ring_amp if ring_amp > 1e-10 else 0
    vr_rmse = np.sqrt(np.mean((cl_rs[:n] - cl_vs[:n]) ** 2))

    # VPM vs Theodorsen
    v_theo_ratio = vpm_amp / theo_amp if theo_amp > 1e-10 else 0
    v_theo_corr = np.corrcoef(cl_vs, cl_tv)[0, 1]

    C_k = theodorsen_function(k)
    print(f"  Ring: {ring_time:.1f}s | amp={ring_amp:.4f} | vs Theo: ratio={r_theo_ratio:.3f} corr={r_theo_corr:.4f}")
    print(f"  VPM:  {vpm_time:.1f}s | amp={vpm_amp:.4f} | vs Ring: amp={vr_amp_ratio:.1%} corr={vr_corr:.4f} RMSE={vr_rmse:.4f} np={sol2._vpm_field.np}")
    print(f"  Theo: amp={theo_amp:.4f} | C(k)=|{abs(C_k):.4f}|∠{np.degrees(np.angle(C_k)):.1f}°")

    return {
        'k': k, 'ring_amp': ring_amp, 'vpm_amp': vpm_amp, 'theo_amp': theo_amp,
        'r_theo_ratio': r_theo_ratio, 'r_theo_corr': r_theo_corr,
        'vr_amp_ratio': vr_amp_ratio, 'vr_corr': vr_corr, 'vr_rmse': vr_rmse,
        'v_theo_ratio': v_theo_ratio, 'v_theo_corr': v_theo_corr,
        'C_k': abs(C_k), 'ring_time': ring_time, 'vpm_time': vpm_time,
    }


if __name__ == '__main__':
    print("=" * 70)
    print("Flapping Wing Validation: VPM-wake vs Ring-wake vs Theodorsen")
    print("NACA 0012, AR=10, h0/c=0.1, nc=10, ns=6")
    print("=" * 70)

    results = []
    for k in [0.5, 0.2, 0.1]:
        r = run_case(f"k={k:.2f}", k)
        if r:
            results.append(r)

    # Summary
    print(f"\n{'='*70}")
    print("  Summary: Flapping Wing Validation")
    print(f"{'='*70}")
    print(f"  {'k':>5s} | {'Ring/Theo':>9s} {'r':>6s} | {'VPM/Ring':>8s} {'r':>6s} {'RMSE':>7s} | {'VPM/Theo':>9s} {'r':>6s}")
    print(f"  {'-'*5}-+-{'-'*9}-+-{'-'*8}-+-{'-'*9}")
    for r in results:
        print(f"  {r['k']:5.2f} | {r['r_theo_ratio']:9.3f} {r['r_theo_corr']:6.3f} | "
              f"{r['vr_amp_ratio']:8.1%} {r['vr_corr']:6.3f} {r['vr_rmse']:7.4f} | "
              f"{r['v_theo_ratio']:9.3f} {r['v_theo_corr']:6.3f}")

    # Pass criteria: VPM/Ring amplitude > 85%, correlation > 0.85
    all_pass = all(r['vr_amp_ratio'] > 0.85 and r['vr_corr'] > 0.85 for r in results)
    print(f"\n  VPM vs Ring: {'PASS' if all_pass else 'FAIL'}")
    print(f"{'='*70}")
