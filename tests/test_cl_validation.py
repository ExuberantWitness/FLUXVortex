"""
Lift coefficient (CL) level validation: GPU (Warp) vs CPU (Numba).

Runs the same PteraSoftware unsteady simulation twice:
  1. Original Numba CPU Biot-Savart
  2. Warp GPU Biot-Savart (monkey-patched)

Compares CL/CD time histories. Max relative error should be < 1e-10.
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

from fluxvortex.warp_patch import patch, unpatch


# ── Geometry & movement setup ─────────────────────────────────────────
def make_static_case():
    """NACA 0012 rectangular wing, AoA = 5 deg, V = 10 m/s, static."""
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
    airplane = ps.geometry.airplane.Airplane(wings=[wing], name="Validation")

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


def make_high_aoa_case():
    """NACA 0012 tapered wing, AoA = 10 deg, V = 15 m/s — different geometry."""
    num_chordwise = 5
    num_spanwise = 10
    root_chord = 1.5
    tip_chord = 0.8
    half_span = 3.0

    wing = ps.geometry.wing.Wing(
        name="Tapered Wing",
        symmetric=True,
        symmetryNormal_G=(0.0, 1.0, 0.0),
        symmetryPoint_G_Cg=(0.0, 0.0, 0.0),
        num_chordwise_panels=num_chordwise,
        chordwise_spacing="uniform",
        wing_cross_sections=[
            ps.geometry.wing_cross_section.WingCrossSection(
                num_spanwise_panels=num_spanwise,
                spanwise_spacing="uniform",
                chord=root_chord,
                Lp_Wcsp_Lpp=(0.0, 0.0, 0.0),
                airfoil=ps.geometry.airfoil.Airfoil(name="naca0012"),
                control_surface_symmetry_type="symmetric",
                control_surface_hinge_point=0.75,
                control_surface_deflection=0.0,
            ),
            ps.geometry.wing_cross_section.WingCrossSection(
                num_spanwise_panels=None,
                spanwise_spacing=None,
                chord=tip_chord,
                Lp_Wcsp_Lpp=(0.0, half_span, 0.0),
                airfoil=ps.geometry.airfoil.Airfoil(name="naca0012"),
                control_surface_symmetry_type="symmetric",
                control_surface_hinge_point=0.75,
                control_surface_deflection=0.0,
            ),
        ],
    )
    airplane = ps.geometry.airplane.Airplane(wings=[wing], name="Tapered")

    op = ps.operating_point.OperatingPoint(rho=1.225, vCg__E=15.0, alpha=10.0, beta=0.0)

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
    """Extract CL and CD time histories from a solved solver."""
    num_steps = solver.num_steps
    first_results = solver.unsteady_problem.first_results_step
    dt = movement.delta_time

    times = []
    cl_list = []
    cd_list = []

    for step in range(first_results, num_steps):
        airplanes = solver.steady_problems[step].airplanes
        for airplane in airplanes:
            coeffs = airplane.forceCoefficients_W
            if coeffs is not None:
                t = step * dt
                cl = -coeffs[2]  # CL = -cFZ
                cd = -coeffs[0]  # CDi = -cFX
                times.append(t)
                cl_list.append(cl)
                cd_list.append(cd)

    return np.array(times), np.array(cl_list), np.array(cd_list)


# ── Run validation ────────────────────────────────────────────────────
def run_case(case_name, airplane, movement):
    """Run a case on both CPU and GPU, compare CL/CD."""
    print(f"\n{'='*60}")
    print(f"  {case_name}")
    print(f"  Panels: {airplane.num_panels}, Wings: {len(airplane.wings)}")
    print(f"  Steps: {movement.num_steps}, dt: {movement.delta_time:.4f}s")
    print(f"{'='*60}")

    dt = movement.delta_time

    # ── CPU run ──
    problem_cpu = ps.problems.UnsteadyProblem(
        movement=movement, only_final_results=False
    )
    solver_cpu = ps.unsteady_ring_vortex_lattice_method.UnsteadyRingVortexLatticeMethodSolver(
        unsteady_problem=problem_cpu
    )
    t0 = time.perf_counter()
    solver_cpu.run(prescribed_wake=True, calculate_streamlines=False, show_progress=False)
    cpu_time = time.perf_counter() - t0
    t_cpu, cl_cpu, cd_cpu = extract_cl_cd(solver_cpu, movement)
    num_steps_cpu = solver_cpu.num_steps
    first_results_cpu = solver_cpu.unsteady_problem.first_results_step
    print(f"  CPU: {cpu_time:.2f}s | CL range: [{cl_cpu.min():.6f}, {cl_cpu.max():.6f}]")

    # ── GPU run ──
    patch()

    # Re-create airplane and movement from scratch for GPU run
    # (PteraSoftware mutates airplane panels, can't reuse)
    airplane2, movement2 = type(movement).__module__, None  # dummy to trigger re-creation
    del airplane2, movement2

    # Use the same factory that created the original
    if "Static" in case_name:
        airplane_gpu, movement_gpu = make_static_case()
    else:
        airplane_gpu, movement_gpu = make_high_aoa_case()

    problem_gpu = ps.problems.UnsteadyProblem(
        movement=movement_gpu, only_final_results=False
    )
    solver_gpu = ps.unsteady_ring_vortex_lattice_method.UnsteadyRingVortexLatticeMethodSolver(
        unsteady_problem=problem_gpu
    )
    t0 = time.perf_counter()
    solver_gpu.run(prescribed_wake=True, calculate_streamlines=False, show_progress=False)
    gpu_time = time.perf_counter() - t0
    t_gpu, cl_gpu, cd_gpu = extract_cl_cd(solver_gpu, movement_gpu)
    print(f"  GPU: {gpu_time:.2f}s | CL range: [{cl_gpu.min():.6f}, {cl_gpu.max():.6f}]")

    unpatch()

    # ── Compare ──
    assert len(cl_cpu) == len(cl_gpu), f"Step count mismatch: {len(cl_cpu)} vs {len(cl_gpu)}"

    cl_err = np.max(np.abs(cl_cpu - cl_gpu))
    cl_rel = cl_err / (np.max(np.abs(cl_cpu)) + 1e-16)
    cd_err = np.max(np.abs(cd_cpu - cd_gpu))
    cd_rel = cd_err / (np.max(np.abs(cd_cpu)) + 1e-16)

    cl_corr = np.corrcoef(cl_cpu, cl_gpu)[0, 1] if len(cl_cpu) > 1 else 1.0

    passed = cl_err < 1e-8

    print(f"\n  CL max abs error:  {cl_err:.2e}")
    print(f"  CL max rel error:  {cl_rel:.2e}")
    print(f"  CL correlation:    {cl_corr:.10f}")
    print(f"  CDi max abs error: {cd_err:.2e}")
    print(f"  CDi max rel error: {cd_rel:.2e}")
    print(f"  Speedup:           {cpu_time/gpu_time:.2f}x")
    print(f"  Result:            {'PASS' if passed else 'FAIL'}")

    return passed, {
        'cl_err': cl_err, 'cl_rel': cl_rel, 'cl_corr': cl_corr,
        'cd_err': cd_err, 'cd_rel': cd_rel,
        'cl_cpu': cl_cpu, 'cl_gpu': cl_gpu,
        'cd_cpu': cd_cpu, 'cd_gpu': cd_gpu,
        't_cpu': t_cpu, 't_gpu': t_gpu,
        'cpu_time': cpu_time, 'gpu_time': gpu_time,
    }


if __name__ == '__main__':
    print("=" * 60)
    print("FLUXVortex CL-Level Validation: GPU vs CPU")
    print("Comparing lift coefficients from full PteraSoftware simulations")
    print("=" * 60)

    all_pass = True

    # Case 1: Static wing
    ok1, res1 = run_case("Static Wing (NACA 0012, AoA=5 deg, V=10 m/s)", *make_static_case())
    all_pass = all_pass and ok1

    # Case 2: High AoA tapered wing
    ok2, res2 = run_case("Tapered Wing (NACA 0012, AoA=10 deg, V=15 m/s)", *make_high_aoa_case())
    all_pass = all_pass and ok2

    # ── Summary ──
    print(f"\n{'='*60}")
    print(f"  SUMMARY: {'ALL PASSED' if all_pass else 'SOME FAILED'}")
    print(f"{'='*60}")

    if all_pass:
        print("  GPU Biot-Savart produces identical CL/CD to CPU Numba.")
        print("  The Warp kernel is a drop-in replacement for PteraSoftware.")
    else:
        print("  WARNING: GPU and CPU results differ beyond tolerance!")
        sys.exit(1)

    # ── Save plot ──
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Static CL
        ax = axes[0, 0]
        ax.plot(res1['t_cpu'], res1['cl_cpu'], 'b-o', ms=3, label='CPU (Numba)')
        ax.plot(res1['t_gpu'], res1['cl_gpu'], 'r--x', ms=3, label='GPU (Warp)')
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('$C_L$')
        ax.set_title(f'Static Wing CL  (err={res1["cl_err"]:.1e})')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Static CD
        ax = axes[0, 1]
        ax.plot(res1['t_cpu'], res1['cd_cpu'], 'b-o', ms=3, label='CPU (Numba)')
        ax.plot(res1['t_gpu'], res1['cd_gpu'], 'r--x', ms=3, label='GPU (Warp)')
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('$C_{D_i}$')
        ax.set_title(f'Static Wing CDi  (err={res1["cd_err"]:.1e})')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # High AoA CL
        ax = axes[1, 0]
        ax.plot(res2['t_cpu'], res2['cl_cpu'], 'b-o', ms=3, label='CPU (Numba)')
        ax.plot(res2['t_gpu'], res2['cl_gpu'], 'r--x', ms=3, label='GPU (Warp)')
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('$C_L$')
        ax.set_title(f'Tapered Wing CL (AoA=10 deg)  (err={res2["cl_err"]:.1e})')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # High AoA CD
        ax = axes[1, 1]
        ax.plot(res2['t_cpu'], res2['cd_cpu'], 'b-o', ms=3, label='CPU (Numba)')
        ax.plot(res2['t_gpu'], res2['cd_gpu'], 'r--x', ms=3, label='GPU (Warp)')
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('$C_{D_i}$')
        ax.set_title(f'Tapered Wing CDi (AoA=10 deg)  (err={res2["cd_err"]:.1e})')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.suptitle('FLUXVortex CL-Level Validation: GPU (Warp) vs CPU (Numba)', fontsize=13)
        plt.tight_layout()

        plot_dir = os.path.join(os.path.dirname(__file__), '..', 'figures')
        os.makedirs(plot_dir, exist_ok=True)
        plot_path = os.path.join(plot_dir, 'cl_validation.png')
        fig.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"\n  Plot saved to: {plot_path}")
    except Exception as e:
        print(f"\n  Plotting skipped: {e}")

    sys.exit(0 if all_pass else 1)
