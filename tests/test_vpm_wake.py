"""
Test: ring wake (PteraSoftware) vs particle wake (UVPM Hybrid).

Runs the same static wing case twice and compares CL/CD correlation.
"""
import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

if not hasattr(np, 'trapezoid'):
    np.trapezoid = np.trapz

import pterasoftware as ps
ps.set_up_logging(level="Warning")

from fluxvortex.solver import UVPMHybridSolver


def make_case():
    """NACA 0012 rectangular wing, AoA=5 deg, V=10 m/s."""
    num_chordwise = 5
    num_spanwise = 10
    chord = 1.0
    half_span = 2.0

    wing = ps.geometry.wing.Wing(
        name="Main Wing",
        symmetric=True,
        symmetryNormal_G=(0.0, 1.0, 0.0),
        symmetryPoint_G_Cg=(0.0, 0.0, 0.0),
        num_chordwise_panels=num_chordwise,
        chordwise_spacing="uniform",
        wing_cross_sections=[
            ps.geometry.wing_cross_section.WingCrossSection(
                num_spanwise_panels=num_spanwise,
                spanwise_spacing="uniform",
                chord=chord,
                Lp_Wcsp_Lpp=(0.0, 0.0, 0.0),
                airfoil=ps.geometry.airfoil.Airfoil(name="naca0012"),
                control_surface_symmetry_type="symmetric",
                control_surface_hinge_point=0.75,
                control_surface_deflection=0.0,
            ),
            ps.geometry.wing_cross_section.WingCrossSection(
                num_spanwise_panels=None,
                spanwise_spacing=None,
                chord=chord,
                Lp_Wcsp_Lpp=(0.0, half_span, 0.0),
                airfoil=ps.geometry.airfoil.Airfoil(name="naca0012"),
                control_surface_symmetry_type="symmetric",
                control_surface_hinge_point=0.75,
                control_surface_deflection=0.0,
            ),
        ],
    )
    airplane = ps.geometry.airplane.Airplane(wings=[wing], name="Test")

    op = ps.operating_point.OperatingPoint(rho=1.225, vCg__E=10.0, alpha=5.0, beta=0.0)

    am = ps.movements.airplane_movement.AirplaneMovement(
        base_airplane=airplane,
        wing_movements=[
            ps.movements.wing_movement.WingMovement(
                base_wing=wing,
                wing_cross_section_movements=[
                    ps.movements.wing_cross_section_movement.WingCrossSectionMovement(
                        base_wing_cross_section=wcs
                    )
                    for wcs in wing.wing_cross_sections
                ],
            )
        ],
    )
    opm = ps.movements.operating_point_movement.OperatingPointMovement(
        base_operating_point=op
    )

    movement = ps.movements.movement.Movement(
        airplane_movements=[am],
        operating_point_movement=opm,
        num_chords=10,
    )
    return airplane, movement


def extract_cl_cd(solver, movement):
    """Extract CL/CD from solver results."""
    num_steps = solver.num_steps
    first_results = solver.unsteady_problem.first_results_step
    dt = movement.delta_time

    times, cl_list, cd_list = [], [], []
    for step in range(first_results, num_steps):
        airplanes = solver.steady_problems[step].airplanes
        for airplane in airplanes:
            coeffs = airplane.forceCoefficients_W
            if coeffs is not None:
                times.append(step * dt)
                cl_list.append(-coeffs[2])
                cd_list.append(-coeffs[0])

    return np.array(times), np.array(cl_list), np.array(cd_list)


if __name__ == '__main__':
    print("=" * 60)
    print("VPM Wake Test: Ring Wake vs Particle Wake (FLOWVLM-style shedding)")
    print("=" * 60)

    # ── 1. Ring vortex wake (PteraSoftware original) ──
    airplane1, movement1 = make_case()
    print(f"\nPanels: {airplane1.num_panels}, Steps: {movement1.num_steps}, dt: {movement1.delta_time:.4f}s")

    problem1 = ps.problems.UnsteadyProblem(movement=movement1, only_final_results=False)
    solver1 = ps.unsteady_ring_vortex_lattice_method.UnsteadyRingVortexLatticeMethodSolver(
        unsteady_problem=problem1
    )
    t0 = time.perf_counter()
    solver1.run(prescribed_wake=True, calculate_streamlines=False, show_progress=False)
    ring_time = time.perf_counter() - t0
    t_ring, cl_ring, cd_ring = extract_cl_cd(solver1, movement1)
    print(f"Ring wake: {ring_time:.2f}s | CL=[{cl_ring.min():.6f}, {cl_ring.max():.6f}] | mean={np.mean(cl_ring):.6f}")

    # ── 2. Vortex particle wake (UVPM Hybrid) ──
    airplane2, movement2 = make_case()

    problem2 = ps.problems.UnsteadyProblem(movement=movement2, only_final_results=False)
    solver2 = UVPMHybridSolver(
        unsteady_problem=problem2,
        max_particles=50000,
        nu=0.0,
        rlxf=0.3,
    )
    t0 = time.perf_counter()
    solver2.run(prescribed_wake=False, calculate_streamlines=False, show_progress=False)
    vpm_time = time.perf_counter() - t0
    t_vpm, cl_vpm, cd_vpm = extract_cl_cd(solver2, movement2)
    print(f"VPM wake:  {vpm_time:.2f}s | CL=[{cl_vpm.min():.6f}, {cl_vpm.max():.6f}] | mean={np.mean(cl_vpm):.6f}")
    print(f"Particles at end: {solver2._vpm_field.np}")

    # ── 3. Compare ──
    n_compare = min(len(cl_ring), len(cl_vpm))
    if n_compare < 2:
        print("Not enough data points for comparison")
        sys.exit(1)

    cl_corr = np.corrcoef(cl_ring[:n_compare], cl_vpm[:n_compare])[0, 1]
    cl_rmse = np.sqrt(np.mean((cl_ring[:n_compare] - cl_vpm[:n_compare]) ** 2))

    print(f"\n{'='*60}")
    print(f"  CL correlation:  {cl_corr:.4f}")
    print(f"  CL RMSE:         {cl_rmse:.6f}")
    print(f"  CL ring mean:    {np.mean(cl_ring):.6f}")
    print(f"  CL VPM mean:     {np.mean(cl_vpm):.6f}")
    print(f"  CL mean diff:    {abs(np.mean(cl_ring) - np.mean(cl_vpm)):.6f}")
    print(f"{'='*60}")

    # ── 4. Plot ──
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 1, figsize=(12, 8))

        ax = axes[0]
        ax.plot(t_ring, cl_ring, 'b-o', ms=3, label='Ring Wake (PteraSoftware)')
        ax.plot(t_vpm, cl_vpm, 'r--x', ms=3, label='Particle Wake (rVPM)')
        ax.set_ylabel('$C_L$')
        ax.set_title(f'CL Comparison (corr={cl_corr:.4f})')
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[1]
        ax.plot(t_ring, cd_ring, 'b-o', ms=3, label='Ring Wake (PteraSoftware)')
        ax.plot(t_vpm, cd_vpm, 'r--x', ms=3, label='Particle Wake (rVPM)')
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('$C_{D_i}$')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.suptitle('FLUXVortex: Ring Wake vs Particle Wake (NACA 0012, AoA=5 deg)', fontsize=12)
        plt.tight_layout()

        plot_dir = os.path.join(os.path.dirname(__file__), '..', 'figures')
        os.makedirs(plot_dir, exist_ok=True)
        plot_path = os.path.join(plot_dir, 'vpm_wake_comparison.png')
        fig.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"\nPlot saved to: {plot_path}")
    except Exception as e:
        print(f"Plotting skipped: {e}")
