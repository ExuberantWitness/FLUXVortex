"""
FLUXVortex vs PteraSoftware: Side-by-side accuracy comparison.

Case 1: Steady-state convergence (static wing, unsteady solver → steady limit)
  - NACA 2412, chord=2.0, semi-span=5.0, alpha=5°
  - XFLR5 reference: CL=0.485, CDi=0.015, Cm=-0.166
  - PteraSoftware UVLM → steady CL/CDi convergence
  - FLUXVortex HybridSolver → same, with VPM far-field wake

Case 2: Plunging wing (Theodorsen analytical reference)
  - NACA 0012, chord=1.0, half-span=5.0, h0/c=0.1
  - k = 0.5, 0.2, 0.1
  - Compare: PteraSoftware pure ring-wake vs HybridSolver N=10 FREE

Case 3: Flapping wing (Yeo et al. 2011 style)
  - NACA 0012, smaller wing, flapping motion
  - Compare force history shapes
"""
import sys, os, time
import numpy as np
from scipy.special import hankel2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
np.trapezoid = np.trapz

import pterasoftware as ps
ps.set_up_logging(level="Warning")
from fluxvortex.particles import VortexParticleField

from experiment_hybrid_panel_particle import (
    HybridSolver, make_wing, make_plunge, extract_cl, get_Vvec, theo_Ck, theo_cl
)


# ═══════════════════════════════════════════════════════════════
# Case 1: Static wing — Unsteady solver → Steady convergence
# ═══════════════════════════════════════════════════════════════

def make_static_wing_naca2412(chord=2.0, half_span=5.0, nc=7, ns=18):
    """NACA 2412 rectangular wing (matches PteraSoftware test case 1D)."""
    return ps.geometry.wing.Wing(
        name="Wing", symmetric=True,
        symmetryNormal_G=(0, 1, 0), symmetryPoint_G_Cg=(0, 0, 0),
        num_chordwise_panels=nc, chordwise_spacing="cosine",
        wing_cross_sections=[
            ps.geometry.wing_cross_section.WingCrossSection(
                num_spanwise_panels=ns, spanwise_spacing="cosine", chord=chord,
                Lp_Wcsp_Lpp=(0, 0, 0),
                airfoil=ps.geometry.airfoil.Airfoil(name="naca2412"),
                control_surface_symmetry_type="symmetric",
                control_surface_hinge_point=0.75,
                control_surface_deflection=0.0),
            ps.geometry.wing_cross_section.WingCrossSection(
                num_spanwise_panels=None, spanwise_spacing=None, chord=chord,
                Lp_Wcsp_Lpp=(0, half_span, 0),
                airfoil=ps.geometry.airfoil.Airfoil(name="naca2412"),
                control_surface_symmetry_type="symmetric",
                control_surface_hinge_point=0.75,
                control_surface_deflection=0.0)])


def make_static_problem(wing, V=10.0, alpha=5.0, num_chords=6):
    """Static wing unsteady problem (no motion, converges to steady state)."""
    ap = ps.geometry.airplane.Airplane(wings=[wing], name="Static")
    op = ps.operating_point.OperatingPoint(rho=1.225, vCg__E=V, alpha=alpha, beta=0)
    am = ps.movements.airplane_movement.AirplaneMovement(
        base_airplane=ap,
        wing_movements=[ps.movements.wing_movement.WingMovement(
            base_wing=wing,
            wing_cross_section_movements=[
                ps.movements.wing_cross_section_movement.WingCrossSectionMovement(
                    base_wing_cross_section=wcs)
                for wcs in wing.wing_cross_sections])])
    opm = ps.movements.operating_point_movement.OperatingPointMovement(
        base_operating_point=op)
    return ap, ps.movements.movement.Movement(
        airplane_movements=[am], operating_point_movement=opm,
        num_chords=num_chords), op


def extract_time_history(solver, mv):
    """Extract CL/CDi time history."""
    first = solver.unsteady_problem.first_results_step
    dt = mv.delta_time
    ts, cls, cdis = [], [], []
    for step in range(first, solver.num_steps):
        for ap in solver.steady_problems[step].airplanes:
            c = ap.forceCoefficients_W
            if c is not None:
                ts.append(step * dt)
                cls.append(-c[2])   # CL (z-axis, positive up)
                cdis.append(-c[0])  # CDi (x-axis, positive backward)
    return np.array(ts), np.array(cls), np.array(cdis)


def run_static_case():
    """Case 1: Static wing convergence comparison."""
    print("=" * 70)
    print("Case 1: Static Wing — Unsteady Solver → Steady Convergence")
    print("NACA 2412, chord=2.0, semi-span=5.0, alpha=5°, nc=7, ns=18")
    print("XFLR5 Reference: CL=0.485, CDi=0.015, Cm=-0.166")
    print("=" * 70)

    V, alpha = 10.0, 5.0
    ref_CL, ref_CDi, ref_Cm = 0.485, 0.015, -0.166

    # --- PteraSoftware pure ring-wake ---
    print(f"\n  [PteraSoftware] Pure ring-wake UVLM ... ", end="", flush=True)
    t0 = time.time()
    w1 = make_static_wing_naca2412()
    ap1, mv1, op1 = make_static_problem(w1, V, alpha, num_chords=6)
    prob1 = ps.problems.UnsteadyProblem(movement=mv1, only_final_results=False)
    sol1 = ps.unsteady_ring_vortex_lattice_method.UnsteadyRingVortexLatticeMethodSolver(prob1)
    sol1.run(prescribed_wake=True, calculate_streamlines=False, show_progress=False)
    t_r, cl_r, cdi_r = extract_time_history(sol1, mv1)
    t1_ps = time.time() - t0
    # Final converged values
    ps_CL_final = cl_r[-1]
    ps_CDi_final = cdi_r[-1]
    print(f"done ({t1_ps:.1f}s)")
    print(f"    CL={ps_CL_final:.4f} (ref={ref_CL:.3f}, err={abs(ps_CL_final-ref_CL)/ref_CL:.1%})")
    print(f"    CDi={ps_CDi_final:.6f} (ref={ref_CDi:.3f}, err={abs(ps_CDi_final-ref_CDi)/ref_CDi:.1%})")

    # --- FLUXVortex HybridSolver ---
    results_hybrid = {}
    for n_keep in [5, 10, 20]:
        label = f"Hybrid N={n_keep}"
        print(f"\n  [FLUXVortex] {label} ... ", end="", flush=True)
        t0 = time.time()
        w2 = make_static_wing_naca2412()
        ap2, mv2, op2 = make_static_problem(w2, V, alpha, num_chords=6)
        prob2 = ps.problems.UnsteadyProblem(movement=mv2, only_final_results=False)
        sol2 = HybridSolver(prob2, n_keep=n_keep, free_vpm=False)
        sol2.run(prescribed_wake=True, calculate_streamlines=False, show_progress=False)
        t_h, cl_h, cdi_h = extract_time_history(sol2, mv2)
        t1_h = time.time() - t0
        h_CL_final = cl_h[-1]
        h_CDi_final = cdi_h[-1]
        np_count = sol2._vpm.np
        print(f"done ({t1_h:.1f}s, np={np_count})")
        print(f"    CL={h_CL_final:.4f} (ref={ref_CL:.3f}, err={abs(h_CL_final-ref_CL)/ref_CL:.1%})")
        print(f"    CDi={h_CDi_final:.6f} (ref={ref_CDi:.3f}, err={abs(h_CDi_final-ref_CDi)/ref_CDi:.1%})")
        results_hybrid[n_keep] = {
            'CL': h_CL_final, 'CDi': h_CDi_final,
            'time': t1_h, 'np': np_count,
            't': t_h, 'cl': cl_h, 'cdi': cdi_h
        }

    return {
        'ps': {'CL': ps_CL_final, 'CDi': ps_CDi_final, 'time': t1_ps,
               't': t_r, 'cl': cl_r, 'cdi': cdi_r},
        'hybrid': results_hybrid,
        'ref': {'CL': ref_CL, 'CDi': ref_CDi, 'Cm': ref_Cm}
    }


# ═══════════════════════════════════════════════════════════════
# Case 2: Plunging wing — Theodorsen comparison
# ═══════════════════════════════════════════════════════════════

def run_plunging_case():
    """Case 2: Plunging wing at multiple reduced frequencies."""
    print("\n" + "=" * 70)
    print("Case 2: Plunging Wing — Theodorsen Analytical Reference")
    print("NACA 0012, chord=1.0, half-span=5.0 (AR=10), h0/c=0.1, nc=10, ns=6")
    print("=" * 70)

    V, chord, h0c = 10.0, 1.0, 0.1
    results = {}

    for k in [0.5, 0.2, 0.1]:
        omega = 2 * k * V / chord
        period = 2 * np.pi / omega
        h0 = h0c * chord
        Ck = theo_Ck(k)
        print(f"\n  k={k:.2f}  period={period:.4f}s  C(k)=|{abs(Ck):.4f}|∠{np.degrees(np.angle(Ck)):.1f}°")

        # PteraSoftware pure ring-wake
        print(f"    [PteraSoftware] Ring-wake ... ", end="", flush=True)
        t0 = time.time()
        w1 = make_wing(chord)
        _, mv1 = make_plunge(w1, h0, period, V)
        prob1 = ps.problems.UnsteadyProblem(movement=mv1, only_final_results=False)
        sol1 = ps.unsteady_ring_vortex_lattice_method.UnsteadyRingVortexLatticeMethodSolver(prob1)
        sol1.run(prescribed_wake=True, calculate_streamlines=False, show_progress=False)
        t_r, cl_r = extract_cl(sol1, mv1)
        t1_ps = time.time() - t0

        t_trans = 2 * period
        mask = t_r > t_trans
        ring_amp = (np.max(cl_r[mask]) - np.min(cl_r[mask])) / 2

        # Theodorsen
        cl_th = theo_cl(k, h0c, omega, t_r, chord)
        theo_amp = (np.max(cl_th[mask]) - np.min(cl_th[mask])) / 2
        ring_ratio = ring_amp / theo_amp
        ring_corr = np.corrcoef(cl_r[mask], cl_th[mask])[0, 1]
        print(f"done ({t1_ps:.1f}s)")
        print(f"      amp={ring_amp:.4f}, Theo={theo_amp:.4f}, ratio={ring_ratio:.3f}, corr={ring_corr:.3f}")

        # FLUXVortex HybridSolver N=10 FREE
        print(f"    [FLUXVortex] Hybrid N=10 FREE ... ", end="", flush=True)
        t0 = time.time()
        w2 = make_wing(chord)
        _, mv2 = make_plunge(w2, h0, period, V)
        prob2 = ps.problems.UnsteadyProblem(movement=mv2, only_final_results=False)
        sol2 = HybridSolver(prob2, n_keep=10, free_vpm=True)
        sol2.run(prescribed_wake=True, calculate_streamlines=False, show_progress=False)
        t_v, cl_v = extract_cl(sol2, mv2)
        t1_h = time.time() - t0

        mask_v = t_v > t_trans
        hybrid_amp = (np.max(cl_v[mask_v]) - np.min(cl_v[mask_v])) / 2
        hybrid_ratio = hybrid_amp / theo_amp
        hybrid_corr = np.corrcoef(cl_v[mask_v], cl_th[:len(cl_v[mask_v])])[0, 1] if len(cl_v[mask_v]) == len(cl_th) else 0
        np_count = sol2._vpm.np
        print(f"done ({t1_h:.1f}s, np={np_count})")
        print(f"      amp={hybrid_amp:.4f}, Theo={theo_amp:.4f}, ratio={hybrid_ratio:.3f}, corr={hybrid_corr:.3f}")

        # FLUXVortex HybridSolver N=20 FREE
        print(f"    [FLUXVortex] Hybrid N=20 FREE ... ", end="", flush=True)
        t0 = time.time()
        w3 = make_wing(chord)
        _, mv3 = make_plunge(w3, h0, period, V)
        prob3 = ps.problems.UnsteadyProblem(movement=mv3, only_final_results=False)
        sol3 = HybridSolver(prob3, n_keep=20, free_vpm=True)
        sol3.run(prescribed_wake=True, calculate_streamlines=False, show_progress=False)
        t_v3, cl_v3 = extract_cl(sol3, mv3)
        t1_h3 = time.time() - t0

        mask_v3 = t_v3 > t_trans
        hybrid3_amp = (np.max(cl_v3[mask_v3]) - np.min(cl_v3[mask_v3])) / 2
        hybrid3_ratio = hybrid3_amp / theo_amp
        hybrid3_corr = np.corrcoef(cl_v3[mask_v3], cl_th[:len(cl_v3[mask_v3])])[0, 1] if len(cl_v3[mask_v3]) == len(cl_th) else 0
        np_count3 = sol3._vpm.np
        print(f"done ({t1_h3:.1f}s, np={np_count3})")
        print(f"      amp={hybrid3_amp:.4f}, Theo={theo_amp:.4f}, ratio={hybrid3_ratio:.3f}, corr={hybrid3_corr:.3f}")

        results[k] = {
            'theo_amp': theo_amp,
            'ring': {'amp': ring_amp, 'ratio': ring_ratio, 'corr': ring_corr, 'time': t1_ps},
            'hybrid10': {'amp': hybrid_amp, 'ratio': hybrid_ratio, 'corr': hybrid_corr, 'time': t1_h, 'np': np_count},
            'hybrid20': {'amp': hybrid3_amp, 'ratio': hybrid3_ratio, 'corr': hybrid3_corr, 'time': t1_h3, 'np': np_count3},
        }

    return results


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("=" * 70)
    print("FLUXVortex vs PteraSoftware: Accuracy Comparison")
    print("=" * 70)

    # Case 1: Static wing
    case1 = run_static_case()

    # Case 2: Plunging wing
    case2 = run_plunging_case()

    # ═══ Summary ═══
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)

    # Case 1 summary
    ref = case1['ref']
    ps_r = case1['ps']
    print(f"\n  Case 1: Static Wing (XFLR5 reference)")
    print(f"  {'Method':<25s} {'CL':>8s} {'CL err':>8s} {'CDi':>10s} {'CDi err':>8s} {'Time':>8s}")
    print(f"  {'─'*25} {'─'*8} {'─'*8} {'─'*10} {'─'*8} {'─'*8}")
    print(f"  {'XFLR5 (reference)':<25s} {ref['CL']:8.3f} {'—':>8s} {ref['CDi']:10.6f} {'—':>8s} {'—':>8s}")
    ps_cl_err = abs(ps_r['CL'] - ref['CL']) / ref['CL']
    ps_cdi_err = abs(ps_r['CDi'] - ref['CDi']) / ref['CDi']
    print(f"  {'PteraSoftware (ring)':<25s} {ps_r['CL']:8.4f} {ps_cl_err:7.1%} {ps_r['CDi']:10.6f} {ps_cdi_err:7.1%} {ps_r['time']:7.1f}s")
    for nk, hr in case1['hybrid'].items():
        h_cl_err = abs(hr['CL'] - ref['CL']) / ref['CL']
        h_cdi_err = abs(hr['CDi'] - ref['CDi']) / ref['CDi']
        print(f"  {'FLUXVortex N='+str(nk):<25s} {hr['CL']:8.4f} {h_cl_err:7.1%} {hr['CDi']:10.6f} {h_cdi_err:7.1%} {hr['time']:7.1f}s")

    # Case 2 summary
    print(f"\n  Case 2: Plunging Wing (Theodorsen reference)")
    print(f"  {'k':>5s}  {'Method':<25s} {'Amp':>8s} {'vs Theo':>8s} {'Corr':>7s} {'Time':>8s}")
    print(f"  {'─'*5}  {'─'*25} {'─'*8} {'─'*8} {'─'*7} {'─'*8}")
    for k, r in case2.items():
        print(f"  {k:5.2f}  {'Theodorsen':<25s} {r['theo_amp']:8.4f}")
        print(f"  {'':5s}  {'PteraSoftware (ring)':<25s} {r['ring']['amp']:8.4f} {r['ring']['ratio']:8.3f} {r['ring']['corr']:7.3f} {r['ring']['time']:7.1f}s")
        print(f"  {'':5s}  {'FLUXVortex N=10 FREE':<25s} {r['hybrid10']['amp']:8.4f} {r['hybrid10']['ratio']:8.3f} {r['hybrid10']['corr']:7.3f} {r['hybrid10']['time']:7.1f}s")
        print(f"  {'':5s}  {'FLUXVortex N=20 FREE':<25s} {r['hybrid20']['amp']:8.4f} {r['hybrid20']['ratio']:8.3f} {r['hybrid20']['corr']:7.3f} {r['hybrid20']['time']:7.1f}s")

    # Highlight improvements
    print(f"\n  ─── Key Findings ───")
    for k, r in case2.items():
        delta = r['hybrid10']['ratio'] - r['ring']['ratio']
        sign = '+' if delta > 0 else ''
        print(f"  k={k:.2f}: Hybrid N=10 FREE vs PteraSoftware ring: {sign}{delta:.3f} ({sign}{delta/r['ring']['ratio']*100:.1f}% change)")

    print(f"\n{'='*70}")
