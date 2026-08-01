import sys
import unittest
from pathlib import Path

import numpy as np

PLATFORM = Path(__file__).resolve().parents[1]
if str(PLATFORM) not in sys.path:
    sys.path.insert(0, str(PLATFORM))

from actual_boundary_3d_cut_wake_junction_guard import (  # noqa: E402
    build_canonical_diamond_wing,
)
from actual_boundary_body_wake_coupled_guard import (  # noqa: E402
    run as run_guard,
)
from claim_runtime.actual_boundary_body_wake import (  # noqa: E402
    solve_actual_boundary_body_wake_p2,
)
from claim_runtime.classified_p2_cut_topology import (  # noqa: E402
    classified_p2_cut_topology,
)
from claim_runtime.thick_body_neumann_shadow import (  # noqa: E402
    ThickBodyNeumannError,
)


class ActualBoundaryBodyWakeTests(unittest.TestCase):
    @staticmethod
    def canonical():
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
        alpha = np.deg2rad(5.0)
        incident = np.repeat(
            np.array(((np.cos(alpha), 0.0, np.sin(alpha)),)),
            len(mesh.faces),
            axis=0,
        )
        return mesh, topology, incident

    def test_wake_is_eliminated_into_one_full_rank_body_system(self):
        mesh, topology, incident = self.canonical()
        solution = solve_actual_boundary_body_wake_p2(
            mesh,
            topology,
            incident_velocity=incident,
            downstream_edge_x=8.0,
            target_quadrature_order=3,
        )
        self.assertEqual(solution.rank, topology.dof_count)
        self.assertEqual(solution.matrix.shape, (81, 81))
        self.assertEqual(solution.independent_wake_unknown_count, 0)
        self.assertLess(solution.relative_weak_residual, 1.0e-12)
        self.assertEqual(solution.wake_attachment_error, 0.0)

    def test_invalid_upstream_or_coincident_downstream_edge_fails(self):
        mesh, topology, incident = self.canonical()
        with self.assertRaises(ThickBodyNeumannError):
            solve_actual_boundary_body_wake_p2(
                mesh,
                topology,
                incident_velocity=incident,
                downstream_edge_x=1.0,
                target_quadrature_order=3,
            )

    def test_preregistered_gate_records_no_go_not_false_promotion(self):
        result = run_guard()
        self.assertEqual(result["stage_decision"], "NO-GO")
        self.assertFalse(
            result["checks"]["paired_quadrature_converges"]
        )
        self.assertFalse(result["production_activation_allowed"])


if __name__ == "__main__":
    unittest.main()
