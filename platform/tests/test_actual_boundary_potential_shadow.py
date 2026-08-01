import sys
import unittest
from pathlib import Path

import numpy as np

PLATFORM = Path(__file__).resolve().parents[1]
if str(PLATFORM) not in sys.path:
    sys.path.insert(0, str(PLATFORM))

from actual_boundary_potential_shadow_guard import (  # noqa: E402
    run as run_guard,
)
from claim_runtime.actual_boundary_potential_shadow import (  # noqa: E402
    solve_actual_boundary_potential,
)
from claim_runtime.thick_body_neumann_shadow import (  # noqa: E402
    ThickBodyNeumannError,
    closed_triangular_mesh,
)
from thick_body_neumann_shadow_guard import icosphere  # noqa: E402


class ActualBoundaryPotentialShadowTests(unittest.TestCase):
    def test_inside_outside_doublet_jump_is_identity(self):
        vertices, faces = icosphere(0)
        mesh = closed_triangular_mesh(vertices, faces)
        incident = np.broadcast_to(
            [1.0, 0.0, 0.0], mesh.centroids.shape
        ).copy()
        solution = solve_actual_boundary_potential(
            mesh, incident_velocity=incident
        )
        np.testing.assert_allclose(
            solution.interior_doublet_matrix
            - solution.exterior_doublet_matrix,
            np.eye(len(mesh.faces)),
            rtol=0.0,
            atol=2.0e-15,
        )
        self.assertLess(
            solution.relative_internal_potential_residual, 1.0e-14
        )
        self.assertLess(
            solution.relative_exterior_surface_identity_residual,
            1.0e-14,
        )

    def test_solver_does_not_mutate_inputs_and_source_is_prescribed(self):
        vertices, faces = icosphere(0)
        mesh = closed_triangular_mesh(vertices, faces)
        incident = np.broadcast_to(
            [0.8, -0.1, 0.2], mesh.centroids.shape
        ).copy()
        wall = np.broadcast_to(
            [0.1, 0.05, -0.02], mesh.centroids.shape
        ).copy()
        incident_before = incident.copy()
        wall_before = wall.copy()
        solution = solve_actual_boundary_potential(
            mesh,
            incident_velocity=incident,
            wall_velocity=wall,
        )
        np.testing.assert_array_equal(incident, incident_before)
        np.testing.assert_array_equal(wall, wall_before)
        np.testing.assert_allclose(
            solution.source_strength,
            np.einsum(
                "ij,ij->i", incident - wall, mesh.normals
            ),
            rtol=0.0,
            atol=0.0,
        )

    def test_evaluation_on_active_perimeter_filament_fails(self):
        vertices, faces = icosphere(0)
        mesh = closed_triangular_mesh(vertices, faces)
        incident = np.broadcast_to(
            [1.0, 0.0, 0.0], mesh.centroids.shape
        ).copy()
        solution = solve_actual_boundary_potential(
            mesh, incident_velocity=incident
        )
        edge_midpoint = 0.5 * (
            mesh.vertices[mesh.faces[0, 0]]
            + mesh.vertices[mesh.faces[0, 1]]
        )
        with self.assertRaises(ThickBodyNeumannError):
            solution.evaluate(edge_midpoint[None, :])

    def test_preregistered_S0_falsifies_direct_surface_pressure(self):
        result = run_guard()
        self.assertEqual(result["equation_oracle_gate"], "GO")
        self.assertEqual(result["surface_pressure_gate"], "NO-GO")
        self.assertEqual(result["stage_decision"], "NO-GO")
        self.assertFalse(result["production_activation_allowed"])


if __name__ == "__main__":
    unittest.main()
