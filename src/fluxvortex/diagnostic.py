"""Quick diagnostic: check particle shedding and wake influence."""
import numpy as np
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if not hasattr(np, 'trapezoid'):
    np.trapezoid = np.trapz

import pterasoftware as ps
ps.set_up_logging(level="Warning")

from fluxvortex.solver import UVPMHybridSolver
from fluxvortex.benchmark import make_airplane, make_movement

airplane = make_airplane()
movement = make_movement(airplane)

problem = ps.problems.UnsteadyProblem(movement=movement)
solver = UVPMHybridSolver(unsteady_problem=problem, max_particles=50000, nu=0.0, rlxf=0.3)

# Run 5 steps manually by calling run with limited scope
# Actually let's just monkey-patch run to stop early
import types

original_run = solver.run

def limited_run(prescribed_wake=False, calculate_streamlines=False, show_progress=False):
    """Run just first 5 steps with diagnostics."""
    # Replicate key parts of the parent run() method
    solver.num_steps = min(10, solver.num_steps)
    solver.initialize_panel_vortices()

    for step in range(solver.num_steps):
        solver.current_step = step
        solver.current_airplanes = solver.steady_problems[step].airplanes
        solver.current_operating_point = solver.steady_problems[step].operating_point
        solver.current_freestream_velocity_geometry_axes = (
            solver.current_operating_point.calculate_freestream_velocity_geometry_axes()
        )

        solver.num_panels = 0
        for airplane_obj in solver.current_airplanes:
            solver.num_panels += airplane_obj.num_panels

        solver.current_wing_wing_influences = np.zeros((solver.num_panels, solver.num_panels))
        solver.current_freestream_velocity_geometry_axes = (
            solver.current_operating_point.calculate_freestream_velocity_geometry_axes()
        )
        solver.current_freestream_wing_influences = np.zeros(solver.num_panels)
        solver.current_wake_wing_influences = np.zeros(solver.num_panels)
        solver.current_vortex_strengths = np.ones(solver.num_panels)

        solver.panels = np.empty(solver.num_panels, dtype=object)
        solver.panel_normal_directions = np.zeros((solver.num_panels, 3))
        solver.panel_areas = np.zeros(solver.num_panels)
        solver.panel_collocation_points = np.zeros((solver.num_panels, 3))
        solver.panel_back_right_vortex_vertices = np.zeros((solver.num_panels, 3))
        solver.panel_front_right_vortex_vertices = np.zeros((solver.num_panels, 3))
        solver.panel_front_left_vortex_vertices = np.zeros((solver.num_panels, 3))
        solver.panel_back_left_vortex_vertices = np.zeros((solver.num_panels, 3))
        solver.panel_right_vortex_centers = np.zeros((solver.num_panels, 3))
        solver.panel_right_vortex_vectors = np.zeros((solver.num_panels, 3))
        solver.panel_front_vortex_centers = np.zeros((solver.num_panels, 3))
        solver.panel_front_vortex_vectors = np.zeros((solver.num_panels, 3))
        solver.panel_left_vortex_centers = np.zeros((solver.num_panels, 3))
        solver.panel_left_vortex_vectors = np.zeros((solver.num_panels, 3))
        solver.panel_back_vortex_centers = np.zeros((solver.num_panels, 3))
        solver.panel_back_vortex_vectors = np.zeros((solver.num_panels, 3))
        solver.seed_points = np.zeros((0, 3))
        solver.panel_moment_references = np.zeros((solver.num_panels, 3))

        solver.panel_is_trailing_edge = np.zeros(solver.num_panels, dtype=bool)
        solver.panel_is_leading_edge = np.zeros(solver.num_panels, dtype=bool)
        solver.panel_is_left_edge = np.zeros(solver.num_panels, dtype=bool)
        solver.panel_is_right_edge = np.zeros(solver.num_panels, dtype=bool)

        solver.last_panel_collocation_points = np.zeros((solver.num_panels, 3))
        solver.last_panel_vortex_strengths = np.zeros(solver.num_panels)
        solver.last_panel_back_right_vortex_vertices = np.zeros((solver.num_panels, 3))
        solver.last_panel_front_right_vortex_vertices = np.zeros((solver.num_panels, 3))
        solver.last_panel_front_left_vortex_vertices = np.zeros((solver.num_panels, 3))
        solver.last_panel_back_left_vortex_vertices = np.zeros((solver.num_panels, 3))
        solver.last_panel_right_vortex_centers = np.zeros((solver.num_panels, 3))
        solver.last_panel_front_vortex_centers = np.zeros((solver.num_panels, 3))
        solver.last_panel_left_vortex_centers = np.zeros((solver.num_panels, 3))
        solver.last_panel_back_vortex_centers = np.zeros((solver.num_panels, 3))

        # Pre-allocated wake arrays for this step
        solver.current_wake_ring_vortex_strengths = solver.wake_ring_vortex_strengths_list[step]
        solver.current_wake_ring_vortex_ages = solver.wake_ring_vortex_ages_list[step]
        solver.current_wake_ring_vortex_front_right_vertices = solver.wake_ring_vortex_front_right_vertices_list[step]
        solver.current_wake_ring_vortex_front_left_vertices = solver.wake_ring_vortex_front_left_vertices_list[step]
        solver.current_wake_ring_vortex_back_left_vertices = solver.wake_ring_vortex_back_left_vertices_list[step]
        solver.current_wake_ring_vortex_back_right_vertices = solver.wake_ring_vortex_back_right_vertices_list[step]

        solver.collapse_geometry()
        solver.calculate_wing_wing_influences()
        solver.calculate_freestream_wing_influences()

        # Our override
        print(f"\n=== Step {step} ===")
        print(f"  Particle field np = {solver._vpm_field.np}")
        solver.calculate_wake_wing_influences()
        print(f"  Wake influence norm = {np.linalg.norm(solver.current_wake_wing_influences):.8f}")
        print(f"  Wake influence max  = {np.max(np.abs(solver.current_wake_wing_influences)):.8f}")

        solver.calculate_vortex_strengths()
        print(f"  Vortex strengths: min={np.min(solver.current_vortex_strengths):.6f}, max={np.max(solver.current_vortex_strengths):.6f}")

        if step >= 1:
            solver.calculate_near_field_forces_and_moments()
            # Print CL
            for ap in solver.current_airplanes:
                print(f"  CL = {-ap.forceCoefficients_W[2]:.6f}")

        solver.populate_next_airplanes_wake(prescribed_wake=prescribed_wake)
        print(f"  After populate_next_airplanes_wake: particle np = {solver._vpm_field.np}")
        if solver._vpm_field.np > 0:
            n = solver._vpm_field.np
            gam_norms = np.linalg.norm(solver._vpm_field._gamma[:n], axis=1)
            print(f"  Particle |Gamma|: min={gam_norms.min():.6f}, max={gam_norms.max():.6f}, mean={gam_norms.mean():.6f}")
            print(f"  Particle sigma: min={solver._vpm_field._sigma[:n].min():.6f}, max={solver._vpm_field._sigma[:n].max():.6f}")

solver.run = types.MethodType(limited_run, solver)
solver.run(prescribed_wake=False, calculate_streamlines=False, show_progress=False)
