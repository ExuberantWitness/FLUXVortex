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


def make_case(alpha=5.0, V=10.0, chord=1.0, half_span=2.0):
    """NACA 0012 rectangular wing."""
    num_chordwise = 5
    num_spanwise = 10

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

    op = ps.operating_point.OperatingPoint(rho=1.225, vCg__E=V, alpha=alpha, beta=0.0)

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


def run_comparison(label, alpha, V):
    """Run ring vs VPM comparison for a given configuration."""
    print(f"\n{'='*60}")
    print(f"  {label} (AoA={alpha}°, V={V} m/s)")
    print(f"{'='*60}")

    # Ring wake
    airplane1, movement1 = make_case(alpha=alpha, V=V)
    problem1 = ps.problems.UnsteadyProblem(movement=movement1, only_final_results=False)
    solver1 = ps.unsteady_ring_vortex_lattice_method.UnsteadyRingVortexLatticeMethodSolver(
        unsteady_problem=problem1
    )
    t0 = time.perf_counter()
    solver1.run(prescribed_wake=True, calculate_streamlines=False, show_progress=False)
    ring_time = time.perf_counter() - t0
    t_ring, cl_ring, cd_ring = extract_cl_cd(solver1, movement1)
    print(f"  Ring: {ring_time:.2f}s | CL mean={np.mean(cl_ring):.6f}")

    # VPM wake
    airplane2, movement2 = make_case(alpha=alpha, V=V)
    problem2 = ps.problems.UnsteadyProblem(movement=movement2, only_final_results=False)
    solver2 = UVPMHybridSolver(unsteady_problem=problem2, max_particles=50000, nu=0.0, rlxf=0.3)
    t0 = time.perf_counter()
    solver2.run(prescribed_wake=False, calculate_streamlines=False, show_progress=False)
    vpm_time = time.perf_counter() - t0
    t_vpm, cl_vpm, cd_vpm = extract_cl_cd(solver2, movement2)
    print(f"  VPM:  {vpm_time:.2f}s | CL mean={np.mean(cl_vpm):.6f} | particles={solver2._vpm_field.np}")

    n_compare = min(len(cl_ring), len(cl_vpm))
    cl_corr = np.corrcoef(cl_ring[:n_compare], cl_vpm[:n_compare])[0, 1]
    cl_rmse = np.sqrt(np.mean((cl_ring[:n_compare] - cl_vpm[:n_compare]) ** 2))
    amp_ratio = np.mean(cl_vpm) / np.mean(cl_ring) if np.mean(cl_ring) != 0 else 0

    print(f"  Correlation: {cl_corr:.4f} | RMSE: {cl_rmse:.6f} | Amplitude: {amp_ratio:.1%}")

    return cl_corr > 0.85 and abs(amp_ratio - 1.0) < 0.15


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

    cases = [
        ('AoA=5 V=10', 5.0, 10.0),
        ('AoA=5 V=15', 5.0, 15.0),
        ('AoA=10 V=10', 10.0, 10.0),
    ]

    all_pass = True
    all_results = []

    for label, alpha, V in cases:
        ok = run_comparison(label, alpha, V)
        all_results.append((label, alpha, V, ok))
        if not ok:
            all_pass = False

    print(f"\n{'='*60}")
    print("  Summary")
    print(f"{'='*60}")
    for label, alpha, V, ok in all_results:
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {label}")
    print(f"{'='*60}")

    if not all_pass:
        sys.exit(1)
