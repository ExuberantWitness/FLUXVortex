import sys
import unittest
from pathlib import Path

import numpy as np

PLATFORM = Path(__file__).resolve().parents[1]
if str(PLATFORM) not in sys.path:
    sys.path.insert(0, str(PLATFORM))

from actual_boundary_p2_galerkin_guard import run as run_guard  # noqa: E402
from claim_runtime.actual_boundary_p2_galerkin import (  # noqa: E402
    closed_p2_topology,
    element_basis_doublet_potential_line_reduced,
    paired_p2_triangle_integral,
    solve_actual_boundary_p2_galerkin,
)
from claim_runtime.distributed_doublet import (  # noqa: E402
    QuadraticDoubletElement,
)
from claim_runtime.doublet_potential import (  # noqa: E402
    element_doublet_potential,
)
from claim_runtime.thick_body_neumann_shadow import (  # noqa: E402
    closed_triangular_mesh,
)
from thick_body_neumann_shadow_guard import icosphere  # noqa: E402


class ActualBoundaryP2GalerkinTests(unittest.TestCase):
    def test_closed_icosahedron_has_shared_vertex_and_edge_dofs(self):
        vertices, faces = icosphere(0)
        mesh = closed_triangular_mesh(vertices, faces)
        topology = closed_p2_topology(mesh)
        self.assertEqual(topology.vertex_dof_count, 12)
        self.assertEqual(topology.edge_dof_count, 30)
        self.assertEqual(topology.dof_count, 42)
        self.assertEqual(
            len(np.unique(topology.local_to_global)), topology.dof_count
        )

    def test_solution_is_weakly_closed_and_trace_continuous(self):
        vertices, faces = icosphere(0)
        mesh = closed_triangular_mesh(vertices, faces)
        incident = np.broadcast_to(
            [1.0, 0.0, 0.0], mesh.centroids.shape
        ).copy()
        solution = solve_actual_boundary_p2_galerkin(
            mesh,
            incident_velocity=incident,
            target_quadrature_order=6,
            source_quadrature_order=6,
        )
        self.assertLess(solution.relative_weak_residual, 1.0e-13)
        self.assertEqual(solution.continuity_residual, 0.0)
        self.assertLess(solution.relative_source_flux, 1.0e-14)
        self.assertEqual(
            solution.surface.continuity_report().boundary_edges, 0
        )

    def test_line_reduced_element_potential_matches_area_oracle(self):
        triangle = np.array(
            [[0.0, 0.0, 0.0], [1.2, 0.1, 0.0], [0.15, 0.95, 0.0]]
        )
        points = np.array(
            [[0.2, 0.3, 0.7], [1.4, -0.5, 1.1], [-0.3, 0.5, -0.6]]
        )
        actual = element_basis_doublet_potential_line_reduced(
            triangle, points, line_quadrature_order=24
        )
        expected = np.column_stack(
            [
                element_doublet_potential(
                    QuadraticDoubletElement(
                        triangle, np.eye(6)[index]
                    ),
                    points,
                    quadrature_order=48,
                )
                for index in range(6)
            ]
        )
        np.testing.assert_allclose(
            actual, expected, rtol=2.0e-12, atol=2.0e-14
        )

    def test_paired_partition_preserves_area_product(self):
        target = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        )
        source = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
        )
        pair = paired_p2_triangle_integral(
            target,
            source,
            target_vertex_ids=[0, 1, 2],
            source_vertex_ids=[0, 1, 3],
            quadrature_order=6,
        )
        self.assertEqual(pair.common_vertex_count, 2)
        self.assertAlmostEqual(pair.partition_measure, 0.25, places=14)

    def test_paired_solver_assembles_all_intersecting_topologies(self):
        vertices, faces = icosphere(0)
        mesh = closed_triangular_mesh(vertices, faces)
        incident = np.broadcast_to(
            [1.0, 0.0, 0.0], mesh.centroids.shape
        ).copy()
        solution = solve_actual_boundary_p2_galerkin(
            mesh,
            incident_velocity=incident,
            target_quadrature_order=4,
            source_quadrature_order=4,
            potential_operator="paired_singular",
        )
        self.assertLess(solution.relative_weak_residual, 1.0e-13)
        self.assertEqual(
            solution.paired_topology_counts,
            {
                "common_triangle": 20,
                "common_edge": 60,
                "common_vertex": 120,
            },
        )

    def test_preregistered_S1_reports_accuracy_without_promotion(self):
        result = run_guard()
        self.assertFalse(result["production_activation_allowed"])
        self.assertTrue(result["checks"]["weak_residual"])
        self.assertTrue(result["checks"]["continuity"])
        self.assertEqual(result["stage_decision"], "NO-GO")


if __name__ == "__main__":
    unittest.main()
