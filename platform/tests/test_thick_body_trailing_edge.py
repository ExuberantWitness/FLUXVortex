import sys
import unittest
from pathlib import Path

import numpy as np

PLATFORM = Path(__file__).resolve().parents[1]
if str(PLATFORM) not in sys.path:
    sys.path.insert(0, str(PLATFORM))

from claim_runtime.n1_thick_shell_adapter import (  # noqa: E402
    N1IncidentVelocityLedger,
)
from claim_runtime.thick_body_neumann_shadow import (  # noqa: E402
    closed_triangular_mesh,
)
from claim_runtime.thick_body_trailing_edge import (  # noqa: E402
    _audit_filament_channel,
    closed_surface_flux_ledger,
)
from thick_body_neumann_shadow_guard import icosphere  # noqa: E402


class ThickBodyTrailingEdgeTests(unittest.TestCase):
    def test_flux_ledger_is_channel_resolved_and_reconstructs_totals(self):
        vertices, faces = icosphere(0)
        mesh = closed_triangular_mesh(vertices, faces)
        shape = (len(mesh.faces), 3)
        freestream = np.broadcast_to([1.0, -0.2, 0.3], shape).copy()
        bound_direct = np.broadcast_to([0.1, 0.0, 0.2], shape).copy()
        bound_image = np.broadcast_to([-0.03, 0.02, 0.0], shape).copy()
        wake_direct = np.broadcast_to([0.0, 0.04, -0.01], shape).copy()
        wake_image = np.broadcast_to([0.02, 0.01, 0.0], shape).copy()
        production = (
            freestream + bound_direct + bound_image + wake_direct
        )
        ledger = N1IncidentVelocityLedger(
            freestream=freestream,
            bound_direct=bound_direct,
            bound_image=bound_image,
            wake_direct=wake_direct,
            wake_image_candidate=wake_image,
            production_total=production,
            physical_symmetry_candidate_total=production + wake_image,
        )
        result = closed_surface_flux_ledger(
            mesh, ledger, wall_velocity=np.zeros(shape)
        )
        self.assertLess(abs(result.production_total), 2.0e-15)
        self.assertLess(
            abs(result.physical_symmetry_candidate_total), 2.0e-15
        )
        self.assertLess(abs(result.production_reconstruction_error), 1e-15)
        self.assertLess(
            abs(result.physical_symmetry_reconstruction_error), 1e-15
        )

    def test_active_ring_piercing_closed_shell_is_not_hidden(self):
        vertices, faces = icosphere(0)
        mesh = closed_triangular_mesh(vertices, faces)
        rings = np.array(
            [[
                [-0.1, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [2.0, 2.0, 0.0],
                [-0.1, 2.0, 0.0],
            ]]
        )
        result = _audit_filament_channel(
            "manufactured", rings, np.array([1.0]), mesh
        )
        self.assertEqual(result.raw_segment_count, 4)
        self.assertEqual(result.active_unique_segment_count, 4)
        self.assertEqual(result.inside_outside_segment_count, 2)
        self.assertEqual(result.proper_shell_piercing_segment_count, 2)
        self.assertEqual(result.proper_shell_intersection_count, 2)
        self.assertEqual(result.minimum_active_segment_to_shell_distance, 0.0)
        self.assertAlmostEqual(
            result.circulation_weighted_piercing_fraction, 0.5
        )

    def test_opposite_coincident_rings_cancel_before_topology_claim(self):
        vertices, faces = icosphere(0)
        mesh = closed_triangular_mesh(vertices, faces)
        ring = np.array(
            [
                [-0.1, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [2.0, 2.0, 0.0],
                [-0.1, 2.0, 0.0],
            ]
        )
        result = _audit_filament_channel(
            "manufactured",
            np.stack((ring, ring)),
            np.array([1.0, -1.0]),
            mesh,
        )
        self.assertEqual(result.unique_segment_count, 4)
        self.assertEqual(result.cancelled_unique_segment_count, 4)
        self.assertEqual(result.active_unique_segment_count, 0)
        self.assertEqual(result.proper_shell_piercing_segment_count, 0)
        self.assertIsNone(result.minimum_active_segment_to_shell_distance)


if __name__ == "__main__":
    unittest.main()
