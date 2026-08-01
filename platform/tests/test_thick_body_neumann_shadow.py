import sys
import unittest
from pathlib import Path

import numpy as np
import yaml

PLATFORM = Path(__file__).resolve().parents[1]
if str(PLATFORM) not in sys.path:
    sys.path.insert(0, str(PLATFORM))

from claim_runtime.thick_body_neumann_shadow import (  # noqa: E402
    ThickBodyNeumannError,
    close_roboeagle_dual_surface_shell,
    closed_triangular_mesh,
    constant_source_polygon_influence,
    solve_conditioned_neumann_source,
)
from claim_runtime.viscous_shell_geometry import (  # noqa: E402
    naca4_dual_surface_shell,
)
from thick_body_neumann_shadow_guard import (  # noqa: E402
    CASES,
    execute_g1d,
    icosphere,
    triangle_quadrature_oracle,
)


class ThickBodyNeumannShadowTests(unittest.TestCase):
    def test_constant_triangle_matches_independent_duffy_oracle(self):
        triangle = np.array(
            [[0.0, 0.0, 0.0], [1.1, 0.0, 0.0], [0.2, 0.8, 0.0]]
        )
        targets = np.array(
            [[0.31, 0.17, 0.73], [-0.21, 0.12, 0.41]]
        )
        result = constant_source_polygon_influence(triangle, targets)
        potential, velocity = triangle_quadrature_oracle(
            triangle, targets
        )
        np.testing.assert_allclose(
            result.potential, potential, rtol=2.0e-12, atol=2.0e-14
        )
        np.testing.assert_allclose(
            result.velocity, velocity, rtol=2.0e-11, atol=2.0e-14
        )

    def test_source_normal_jump_is_explicit_not_offset_fitted(self):
        triangle = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.2, 0.8, 0.0]]
        )
        centroid = np.mean(triangle, axis=0, keepdims=True)
        exterior = constant_source_polygon_influence(
            triangle, centroid, on_surface_side="exterior"
        )
        interior = constant_source_polygon_influence(
            triangle, centroid, on_surface_side="interior"
        )
        np.testing.assert_allclose(
            exterior.velocity[0, 2], -0.5, atol=0.0
        )
        np.testing.assert_allclose(
            interior.velocity[0, 2], 0.5, atol=0.0
        )
        np.testing.assert_allclose(
            exterior.potential, interior.potential, atol=0.0
        )

    def test_closed_sphere_neumann_solution_and_flux(self):
        vertices, faces = icosphere(1)
        mesh = closed_triangular_mesh(vertices, faces)
        incident = np.tile([1.0, 0.0, 0.0], (len(faces), 1))
        solution = solve_conditioned_neumann_source(
            mesh, incident_velocity=incident
        )
        self.assertLess(
            solution.relative_no_penetration_residual, 2.0e-13
        )
        self.assertLess(solution.relative_source_flux, 2.0e-13)
        self.assertLess(solution.condition_number, 10.0)
        expected = 1.5 * (
            incident
            - np.einsum(
                "ij,ij->i", incident, mesh.normals
            )[:, None]
            * mesh.normals
        )
        error = np.sqrt(
            np.mean(
                np.sum((solution.total_velocity - expected) ** 2, axis=1)
            )
        ) / 1.5
        self.assertLess(error, 0.02)

    def test_open_or_reversed_mesh_fails(self):
        vertices, faces = icosphere(0)
        with self.assertRaises(ThickBodyNeumannError):
            closed_triangular_mesh(vertices, faces[:-1])
        with self.assertRaises(ThickBodyNeumannError):
            closed_triangular_mesh(vertices, faces[:, ::-1])

    def test_solver_does_not_mutate_conditioning_inputs(self):
        vertices, faces = icosphere(1)
        mesh = closed_triangular_mesh(vertices, faces)
        incident = np.tile([0.4, -0.1, 0.2], (len(faces), 1))
        incident_before = incident.copy()
        vertices_before = mesh.vertices.copy()
        solve_conditioned_neumann_source(
            mesh, incident_velocity=incident
        )
        np.testing.assert_array_equal(incident, incident_before)
        np.testing.assert_array_equal(mesh.vertices, vertices_before)

    def test_unsteady_objectivity_stage_passes_frozen_thresholds(self):
        contract = yaml.safe_load(CASES.read_text(encoding="utf-8"))
        result = execute_g1d(contract)
        self.assertEqual(result["status"], "passed")
        self.assertTrue(all(result["checks"].values()))

    def test_roboeagle_dual_surface_is_closed_without_moving_mean(self):
        chord_fraction = 0.5 * (
            1.0 - np.cos(np.linspace(0.0, np.pi, 33))
        )
        span = np.linspace(0.0, 0.8, 7)
        chord = np.linspace(0.287, 0.12, len(span))
        shell = naca4_dual_surface_shell(
            chord_fraction, span, chord
        )
        mean_before = shell.mean_surface.copy()
        result = close_roboeagle_dual_surface_shell(shell)
        self.assertEqual(result.mesh.boundary_edge_count, 0)
        self.assertEqual(result.mesh.nonmanifold_edge_count, 0)
        self.assertEqual(result.mesh.orientation_mismatch_count, 0)
        self.assertGreater(result.mesh.signed_volume, 0.0)
        self.assertEqual(result.leading_edge_weld_count, len(span))
        self.assertEqual(result.trailing_edge_weld_count, 0)
        self.assertEqual(
            set(result.face_roles),
            {
                "upper",
                "lower",
                "root_cap",
                "tip_cap",
                "trailing_base",
            },
        )
        np.testing.assert_array_equal(shell.mean_surface, mean_before)


if __name__ == "__main__":
    unittest.main()
