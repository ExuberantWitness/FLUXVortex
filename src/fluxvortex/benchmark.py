"""
Benchmark: compare PteraSoftware UVLM (ring wake) vs UVPM Hybrid (particle wake)
on a rectangular wing with dihedral flapping.
"""
import numpy as np
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# numpy compat
if not hasattr(np, 'trapezoid'):
    np.trapezoid = np.trapz

import pterasoftware as ps
ps.set_up_logging(level="Warning")

from fluxvortex.solver import UVPMHybridSolver


def make_airplane():
    """Create rectangular wing airplane."""
    num_chordwise_panels = 5
    num_spanwise_panels = 8
    chord = 1.0
    half_span = 2.0

    airplane = ps.geometry.airplane.Airplane(
        wings=[
            ps.geometry.wing.Wing(
                wing_cross_sections=[
                    ps.geometry.wing_cross_section.WingCrossSection(
                        num_spanwise_panels=num_spanwise_panels,
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
                name="Main Wing",
                Ler_Gs_Cgs=(0.0, 0.005, 0.0),
                angles_Gs_to_Wn_ixyz=(0.0, 0.0, 0.0),
                symmetric=True,
                mirror_only=False,
                symmetryNormal_G=(0.0, 1.0, 0.0),
                symmetryPoint_G_Cg=(0.0, 0.0, 0.0),
                num_chordwise_panels=num_chordwise_panels,
                chordwise_spacing="uniform",
            ),
        ],
        name="Benchmark Airplane",
    )
    return airplane


def make_movement(airplane):
    """Create flapping movement."""
    freq = 1.0
    wing0 = airplane.wings[0]
    wing1 = airplane.wings[1]

    movements = []
    for wing in [wing0, wing1]:
        wcs_movements = [
            ps.movements.wing_cross_section_movement.WingCrossSectionMovement(
                base_wing_cross_section=wing.wing_cross_sections[0],
            ),
            ps.movements.wing_cross_section_movement.WingCrossSectionMovement(
                base_wing_cross_section=wing.wing_cross_sections[1],
            ),
        ]
        movements.append(
            ps.movements.wing_movement.WingMovement(
                base_wing=wing,
                wing_cross_section_movements=wcs_movements,
                ampAngles_Gs_to_Wn_ixyz=(20.0, 0.0, 0.0),
                periodAngles_Gs_to_Wn_ixyz=(1.0 / freq, 0.0, 0.0),
                spacingAngles_Gs_to_Wn_ixyz=("sine", "sine", "sine"),
            )
        )

    airplane_movement = ps.movements.airplane_movement.AirplaneMovement(
        base_airplane=airplane,
        wing_movements=movements,
    )

    op = ps.operating_point.OperatingPoint(
        rho=1.225, vCg__E=10.0, alpha=5.0, beta=0.0
    )
    op_movement = ps.movements.operating_point_movement.OperatingPointMovement(
        base_operating_point=op
    )

    movement = ps.movements.movement.Movement(
        airplane_movements=[airplane_movement],
        operating_point_movement=op_movement,
        num_cycles=3,
    )
    return movement


def extract_results(solver, movement):
    """Extract CL/CD from solver results."""
    num_steps = movement.num_steps
    dt = movement.delta_time
    times = np.linspace(0, num_steps * dt, num_steps, endpoint=False)

    cl_list = []
    cd_list = []
    for step in range(num_steps):
        airplane = solver.steady_problems[step].airplanes[0]
        cl = -airplane.forceCoefficients_W[2]
        cd = -airplane.forceCoefficients_W[0]
        cl_list.append(float(cl))
        cd_list.append(float(cd))

    return times.tolist(), cl_list, cd_list


# ── Run benchmark ──────────────────────────────────────────────────
print("=" * 60)
print("Benchmark: PteraSoftware UVLM vs UVPM Hybrid (VPM Particle Wake)")
print("=" * 60)

airplane = make_airplane()
movement = make_movement(airplane)

# 1. Original PteraSoftware UVLM with free wake
print("\n[1/2] Running PteraSoftware UVLM (free wake ring vortices)...")
problem_ring = ps.problems.UnsteadyProblem(movement=movement)
solver_ring = ps.unsteady_ring_vortex_lattice_method.UnsteadyRingVortexLatticeMethodSolver(
    unsteady_problem=problem_ring
)
solver_ring.run(prescribed_wake=False, calculate_streamlines=False, show_progress=True)
t_ring, cl_ring, cd_ring = extract_results(solver_ring, movement)
print(f"  CL range: [{min(cl_ring):.4f}, {max(cl_ring):.4f}]  Mean: {np.mean(cl_ring):.4f}")

# Reset movement for second run
airplane2 = make_airplane()
movement2 = make_movement(airplane2)

# 2. UVPM Hybrid with vortex particle wake
print("\n[2/2] Running UVPM Hybrid (vortex particle wake + rVPM)...")
problem_vpm = ps.problems.UnsteadyProblem(movement=movement2)
solver_vpm = UVPMHybridSolver(
    unsteady_problem=problem_vpm,
    max_particles=50000,
    nu=0.0,
    rlxf=0.3,
)
solver_vpm.run(prescribed_wake=False, calculate_streamlines=False, show_progress=True)
t_vpm, cl_vpm, cd_vpm = extract_results(solver_vpm, movement2)
print(f"  CL range: [{min(cl_vpm):.4f}, {max(cl_vpm):.4f}]  Mean: {np.mean(cl_vpm):.4f}")

# 3. Compare and plot
print("\n" + "=" * 60)
print("Comparison Results")
print("=" * 60)

# Load FLOWVLM results if available
flowvlm_path = os.path.join(os.path.dirname(__file__), "..", "flowvlm_results.json")
fvlm_cl = None
fvlm_t = None
if os.path.exists(flowvlm_path):
    with open(flowvlm_path) as f:
        fvlm_data = eval(f.read())
    fvlm_t = fvlm_data["times"]
    fvlm_cl = fvlm_data["CL"]
    print(f"  FLOWVLM CL range: [{min(fvlm_cl):.4f}, {max(fvlm_cl):.4f}]  Mean: {np.mean(fvlm_cl):.4f}")

# Statistics
cl_vpm_arr = np.array(cl_vpm)
cl_ring_arr = np.array(cl_ring)

# Interpolate for correlation
cl_vpm_interp = np.interp(t_ring, t_vpm, cl_vpm)
cl_corr = np.corrcoef(cl_ring_arr, cl_vpm_interp)[0, 1]
cl_rmse = np.sqrt(np.mean((cl_ring_arr - cl_vpm_interp) ** 2))

print(f"\n  CL correlation (ring vs particle wake): {cl_corr:.4f}")
print(f"  CL RMSE: {cl_rmse:.4f}")
print(f"  CL mean difference: {abs(np.mean(cl_ring) - np.mean(cl_vpm)):.4f}")

# Save results
results = {
    "ring_wake": {"times": t_ring, "CL": cl_ring, "CD": cd_ring},
    "vpm_wake": {"times": t_vpm, "CL": cl_vpm, "CD": cd_vpm},
}
if fvlm_cl:
    results["flowvlm"] = {"times": fvlm_t, "CL": fvlm_cl}

results_path = os.path.join(os.path.dirname(__file__), "..", "three_solver_comparison.json")
with open(results_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to: {results_path}")

# Generate plot
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    ax = axes[0]
    ax.plot(t_ring, cl_ring, 'b-', linewidth=1.5, label='PteraSoftware UVLM (ring wake)')
    ax.plot(t_vpm, cl_vpm, 'r--', linewidth=1.5, label='UVPM Hybrid (VPM particle wake)')
    if fvlm_cl:
        ax.plot(fvlm_t, fvlm_cl, 'g:', linewidth=1.5, label='FLOWVLM (horseshoe + VPM)')
    ax.set_ylabel('$C_L$', fontsize=12)
    ax.set_title('Flapping Wing CL: Ring Wake vs VPM Particle Wake', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(t_ring, cd_ring, 'b-', linewidth=1.5, label='PteraSoftware UVLM (ring wake)')
    ax.plot(t_vpm, cd_vpm, 'r--', linewidth=1.5, label='UVPM Hybrid (VPM particle wake)')
    ax.set_xlabel('Time [s]', fontsize=12)
    ax.set_ylabel('$C_D$', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(os.path.dirname(__file__), "..", "vpm_comparison_plot.png")
    fig.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Plot saved to: {plot_path}")
except Exception as e:
    print(f"Plotting failed: {e}")
