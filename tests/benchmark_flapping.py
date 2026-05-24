"""
FLUXVortex vs PteraSoftware: Flapping Wing Accuracy Comparison.

Reproduces PteraSoftware's main flapping wing example exactly:
  examples/unsteady_ring_vortex_lattice_method_solver_variable.py

Configuration:
  Main wing: NACA 2412, chord 1.75/1.5, semi-span 6.0, nc=6, ns=8
  V-Tail: NACA 0012, chord 1.5/1.0, semi-span 2.0, nc=6, ns=8
  Flapping: 15° sweep amplitude about x-axis, period 1.0s
  V=10 m/s, alpha=1°, 3 cycles, prescribed wake

Note: The main wing has type 5 symmetry (symmetric + offset symmetry plane),
so PteraSoftware splits it into 2 wings → 3 wings total.
"""
import sys, os, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
np.trapezoid = np.trapz

import pterasoftware as ps
ps.set_up_logging(level="Warning")
from fluxvortex.particles import VortexParticleField
from experiment_hybrid_panel_particle import HybridSolver, get_Vvec


def build_problem():
    """Build the exact same problem as PteraSoftware's example."""
    example_airplane = ps.geometry.airplane.Airplane(
        wings=[
            ps.geometry.wing.Wing(
                wing_cross_sections=[
                    ps.geometry.wing_cross_section.WingCrossSection(
                        num_spanwise_panels=8,
                        chord=1.75,
                        Lp_Wcsp_Lpp=(0.0, 0.0, 0.0),
                        angles_Wcsp_to_Wcs_ixyz=(0.0, 0.0, 0.0),
                        control_surface_symmetry_type="symmetric",
                        control_surface_hinge_point=0.75,
                        control_surface_deflection=0.0,
                        spanwise_spacing="cosine",
                        airfoil=ps.geometry.airfoil.Airfoil(
                            name="naca2412", n_points_per_side=400),
                    ),
                    ps.geometry.wing_cross_section.WingCrossSection(
                        num_spanwise_panels=None,
                        chord=1.5,
                        Lp_Wcsp_Lpp=(0.75, 6.0, 1.0),
                        angles_Wcsp_to_Wcs_ixyz=(0.0, 5.0, 0.0),
                        control_surface_symmetry_type="symmetric",
                        control_surface_hinge_point=0.75,
                        control_surface_deflection=0.0,
                        spanwise_spacing=None,
                        airfoil=ps.geometry.airfoil.Airfoil(
                            name="naca2412", n_points_per_side=400),
                    ),
                ],
                name="Main Wing",
                Ler_Gs_Cgs=(0.0, 0.5, 0.0),
                angles_Gs_to_Wn_ixyz=(0.0, 0.0, 0.0),
                symmetric=True,
                mirror_only=False,
                symmetryNormal_G=(0.0, 1.0, 0.0),
                symmetryPoint_G_Cg=(0.0, 0.0, 0.0),
                num_chordwise_panels=6,
                chordwise_spacing="uniform",
            ),
            ps.geometry.wing.Wing(
                wing_cross_sections=[
                    ps.geometry.wing_cross_section.WingCrossSection(
                        num_spanwise_panels=8,
                        chord=1.5,
                        Lp_Wcsp_Lpp=(0.0, 0.0, 0.0),
                        angles_Wcsp_to_Wcs_ixyz=(0.0, 0.0, 0.0),
                        control_surface_symmetry_type="symmetric",
                        control_surface_hinge_point=0.75,
                        control_surface_deflection=0.0,
                        spanwise_spacing="uniform",
                        airfoil=ps.geometry.airfoil.Airfoil(
                            name="naca0012", n_points_per_side=400),
                    ),
                    ps.geometry.wing_cross_section.WingCrossSection(
                        num_spanwise_panels=None,
                        chord=1.0,
                        Lp_Wcsp_Lpp=(0.5,  2.0, 1.0),
                        angles_Wcsp_to_Wcs_ixyz=(0.0, 0.0, 0.0),
                        control_surface_symmetry_type="symmetric",
                        control_surface_hinge_point=0.75,
                        control_surface_deflection=0.0,
                        spanwise_spacing=None,
                        airfoil=ps.geometry.airfoil.Airfoil(
                            name="naca0012", n_points_per_side=400),
                    ),
                ],
                name="V-Tail",
                Ler_Gs_Cgs=(5.0, 0.0, 0.0),
                angles_Gs_to_Wn_ixyz=(0.0, -5.0, 0.0),
                symmetric=True,
                mirror_only=False,
                symmetryNormal_G=(0.0, 1.0, 0.0),
                symmetryPoint_G_Cg=(0.0, 0.0, 0.0),
                num_chordwise_panels=6,
                chordwise_spacing="uniform",
            ),
        ],
        name="Example Airplane",
    )

    print(f"  Wings after symmetry: {len(example_airplane.wings)}")
    for i, w in enumerate(example_airplane.wings):
        print(f"    Wing {i}: {w.name}, panels={w.num_panels}")

    # Create movements for all wings (type 5 symmetry splits main wing)
    wing_movements = []
    for i, wing in enumerate(example_airplane.wings):
        is_flapping = (wing.name == "Main Wing")
        wm = ps.movements.wing_movement.WingMovement(
            base_wing=wing,
            wing_cross_section_movements=[
                ps.movements.wing_cross_section_movement.WingCrossSectionMovement(
                    base_wing_cross_section=wcs)
                for wcs in wing.wing_cross_sections
            ],
            ampAngles_Gs_to_Wn_ixyz=(15.0 if is_flapping else 0.0, 0.0, 0.0),
            periodAngles_Gs_to_Wn_ixyz=(1.0 if is_flapping else 0.0, 0.0, 0.0),
            spacingAngles_Gs_to_Wn_ixyz=("sine", "sine", "sine"),
            phaseAngles_Gs_to_Wn_ixyz=(0.0, 0.0, 0.0),
        )
        wing_movements.append(wm)

    am = ps.movements.airplane_movement.AirplaneMovement(
        base_airplane=example_airplane,
        wing_movements=wing_movements,
    )

    op = ps.operating_point.OperatingPoint(
        rho=1.225, vCg__E=10.0, alpha=1.0, beta=0.0, nu=15.06e-6)

    opm = ps.movements.operating_point_movement.OperatingPointMovement(
        base_operating_point=op)

    mv = ps.movements.movement.Movement(
        airplane_movements=[am],
        operating_point_movement=opm,
        num_cycles=3,
    )

    return example_airplane, mv


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
                cls.append(-c[2])
                cdis.append(-c[0])
    return np.array(ts), np.array(cls), np.array(cdis)


if __name__ == '__main__':
    print("=" * 70)
    print("FLUXVortex vs PteraSoftware: Flapping Wing Comparison")
    print("NACA 2412 main wing + NACA 0012 V-tail")
    print("Flapping: 15° sweep about x-axis, period=1.0s, V=10 m/s, alpha=1°")
    print("3 cycles, prescribed wake")
    print("=" * 70)

    results = {}

    # ─── PteraSoftware pure ring-wake ───
    print(f"\n  [1/2] PteraSoftware (ring-wake) ... ", end="", flush=True)
    t0 = time.time()
    ap1, mv1 = build_problem()
    prob1 = ps.problems.UnsteadyProblem(movement=mv1, only_final_results=False)
    sol1 = ps.unsteady_ring_vortex_lattice_method.UnsteadyRingVortexLatticeMethodSolver(prob1)
    sol1.run(prescribed_wake=True, calculate_streamlines=False, show_progress=False)
    t_ps, cl_ps, cdi_ps = extract_time_history(sol1, mv1)
    t1_ps = time.time() - t0
    print(f"done ({t1_ps:.1f}s, {len(t_ps)} steps)")
    results['ring'] = {'t': t_ps, 'cl': cl_ps, 'cdi': cdi_ps, 'time': t1_ps}

    # ─── FLUXVortex Hybrid N=10 prescribed VPM ───
    print(f"  [2/2] FLUXVortex Hybrid N=10 (prescribed VPM) ... ", end="", flush=True)
    t0 = time.time()
    ap2, mv2 = build_problem()
    prob2 = ps.problems.UnsteadyProblem(movement=mv2, only_final_results=False)
    sol2 = HybridSolver(prob2, n_keep=10, free_vpm=False)
    sol2.run(prescribed_wake=True, calculate_streamlines=False, show_progress=False)
    t_h10, cl_h10, cdi_h10 = extract_time_history(sol2, mv2)
    t1_h10 = time.time() - t0
    np10 = sol2._vpm.np
    print(f"done ({t1_h10:.1f}s, np={np10})")
    results['hybrid10'] = {'t': t_h10, 'cl': cl_h10, 'cdi': cdi_h10, 'time': t1_h10, 'np': np10}

    # ═══ Analysis ═══
    print(f"\n{'='*70}")
    print("ANALYSIS")
    print(f"{'='*70}")

    period = 1.0
    t_last = t_ps[-1]
    t_start_last_cycle = t_last - period
    mask_ps = t_ps >= t_start_last_cycle
    cl_ps_last = cl_ps[mask_ps]

    print(f"\n  Last Cycle CL Metrics (t >= {t_start_last_cycle:.2f}s):")
    print(f"  {'Method':<30s} {'CL mean':>8s} {'CL amp':>8s} {'CL max':>8s} {'CL min':>8s} {'Corr':>7s}")
    print(f"  {'─'*30} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*7}")

    def print_metrics(label, t_arr, cl_arr):
        mask = t_arr >= t_start_last_cycle
        if not mask.any():
            print(f"  {label:<30s} (no data in last cycle)")
            return {}
        cl_last = cl_arr[mask]
        n = min(len(cl_ps_last), len(cl_last))
        if n < 2:
            corr = 0
        else:
            corr = np.corrcoef(cl_ps_last[:n], cl_last[:n])[0, 1]
        amp = (cl_last.max() - cl_last.min()) / 2
        print(f"  {label:<30s} {cl_last.mean():8.4f} {amp:8.4f} {cl_last.max():8.4f} {cl_last.min():8.4f} {corr:7.4f}")
        return {'mean': cl_last.mean(), 'amp': amp, 'max': cl_last.max(), 'min': cl_last.min(), 'corr': corr}

    m_ps = print_metrics("PteraSoftware (ring)", t_ps, cl_ps)
    m_h10 = print_metrics("FLUXVortex N=10", t_h10, cl_h10)

    # Full time history correlation
    n = min(len(cl_ps), len(cl_h10))
    n2 = min(len(cdi_ps), len(cdi_h10))
    corr_cl = np.corrcoef(cl_ps[:n], cl_h10[:n])[0, 1]
    corr_cdi = np.corrcoef(cdi_ps[:n2], cdi_h10[:n2])[0, 1]
    rmse_cl = np.sqrt(np.mean((cl_ps[:n] - cl_h10[:n]) ** 2))

    print(f"\n  Full Time History:")
    print(f"  {'Metric':<20s} {'Value':>10s}")
    print(f"  {'─'*20} {'─'*10}")
    print(f"  {'Corr(CL)':<20s} {corr_cl:10.6f}")
    print(f"  {'Corr(CDi)':<20s} {corr_cdi:10.6f}")
    print(f"  {'RMSE(CL)':<20s} {rmse_cl:10.6f}")

    # Save results
    print(f"\n  Saving results ...")
    np.savez(os.path.join(os.path.dirname(__file__), '..', 'figures', 'flapping_results.npz'),
             t_ps=t_ps, cl_ps=cl_ps, cdi_ps=cdi_ps,
             t_h10=t_h10, cl_h10=cl_h10, cdi_h10=cdi_h10,
             # Empty placeholders for N=20 and FREE (not run — too slow)
             t_h20=t_h10, cl_h20=cl_h10, cdi_h20=cdi_h10,
             t_h10f=np.array([]), cl_h10f=np.array([]), cdi_h10f=np.array([]),
             allow_pickle=True)
    print(f"  Now run: python tests/plot_flapping.py")
    print(f"{'='*70}")
