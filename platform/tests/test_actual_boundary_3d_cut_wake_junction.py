import sys
import unittest
from pathlib import Path

import numpy as np

PLATFORM = Path(__file__).resolve().parents[1]
if str(PLATFORM) not in sys.path:
    sys.path.insert(0, str(PLATFORM))

from actual_boundary_3d_cut_wake_junction_guard import (  # noqa: E402
    build_canonical_diamond_wing,
    run as run_guard,
)
from claim_runtime.classified_p2_cut_topology import (  # noqa: E402
    ClassifiedP2CutError,
    classified_p2_cut_topology,
)


class ActualBoundary3DCutWakeJunctionTests(unittest.TestCase):
    @staticmethod
    def topology():
        mesh, upper, lower, cut_edges, endpoints = (
            build_canonical_diamond_wing()
        )
        topology = classified_p2_cut_topology(
            mesh,
            upper_face_indices=upper,
            lower_face_indices=lower,
            cut_edges=cut_edges,
            zero_jump_end_vertices=endpoints,
        )
        return mesh, topology

    def test_geometry_stays_watertight_while_only_cut_trace_splits(self):
        mesh, topology = self.topology()
        self.assertEqual(mesh.boundary_edge_count, 0)
        self.assertEqual(mesh.nonmanifold_edge_count, 0)
        self.assertEqual(mesh.orientation_mismatch_count, 0)
        self.assertEqual(topology.duplicated_vertex_count, 3)
        self.assertEqual(topology.duplicated_edge_midpoint_count, 4)
        self.assertEqual(topology.duplicated_dof_count, 7)
        self.assertEqual(
            topology.noncut_trace_dof_mismatch_count,
            0,
        )
        self.assertEqual(
            topology.maximum_cut_coordinate_pair_gap,
            0.0,
        )

    def test_manufactured_quadratic_jump_is_exact_and_gauge_invariant(self):
        _, topology = self.topology()
        span = topology.cut_node_coordinates[:, 1]
        expected = 1.0 - span**2
        values = np.zeros(topology.dof_count)
        values[topology.upper_cut_dofs] = 0.5 * expected
        values[topology.lower_cut_dofs] = -0.5 * expected
        np.testing.assert_allclose(
            topology.cut_jump(values),
            expected,
            rtol=0.0,
            atol=1.0e-15,
        )
        np.testing.assert_allclose(
            topology.cut_jump(values + 3.7),
            expected,
            rtol=0.0,
            atol=1.0e-15,
        )
        self.assertEqual(
            topology.upper_cut_dofs[0],
            topology.lower_cut_dofs[0],
        )
        self.assertEqual(
            topology.upper_cut_dofs[-1],
            topology.lower_cut_dofs[-1],
        )

    def test_unclassified_or_branched_cut_is_rejected(self):
        mesh, upper, lower, cut_edges, endpoints = (
            build_canonical_diamond_wing()
        )
        first_cut = set(cut_edges[0])
        incident_lower = next(
            face_index
            for face_index in lower
            if first_cut.issubset(set(mesh.faces[face_index]))
        )
        with self.assertRaises(ClassifiedP2CutError):
            classified_p2_cut_topology(
                mesh,
                upper_face_indices=upper,
                lower_face_indices=[
                    face_index
                    for face_index in lower
                    if face_index != incident_lower
                ],
                cut_edges=cut_edges,
                zero_jump_end_vertices=endpoints,
            )
        with self.assertRaises(ClassifiedP2CutError):
            classified_p2_cut_topology(
                mesh,
                upper_face_indices=upper,
                lower_face_indices=lower,
                cut_edges=cut_edges[:-1],
                zero_jump_end_vertices=endpoints,
            )

    def test_preregistered_guard_passes_without_pressure_or_force(self):
        result = run_guard()
        self.assertEqual(result["stage_decision"], "GO")
        self.assertTrue(all(result["checks"].values()))
        self.assertFalse(result["production_activation_allowed"])


if __name__ == "__main__":
    unittest.main()
