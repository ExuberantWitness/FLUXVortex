import sys
import unittest
from pathlib import Path

PLATFORM = Path(__file__).resolve().parents[1]
if str(PLATFORM) not in sys.path:
    sys.path.insert(0, str(PLATFORM))

from actual_boundary_blunt_base_topology_guard import (  # noqa: E402
    run as run_guard,
)
from claim_runtime.blunt_base_topology import (  # noqa: E402
    BluntBaseTopologyError,
    naca4_blunt_base_topology,
)


class ActualBoundaryBluntBaseTopologyTests(unittest.TestCase):
    def test_standard_open_naca2406_has_two_distinct_origins(self):
        state = naca4_blunt_base_topology(base_fraction=1.0)
        self.assertAlmostEqual(state.base_thickness, 0.00126)
        self.assertAlmostEqual(
            state.optimal_single_origin_residual,
            0.5 * state.base_thickness,
        )
        self.assertFalse(
            state.single_junction_topologically_admissible
        )
        self.assertEqual(state.two_origin_attachment_residual, 0.0)

    def test_closed_endpoint_recovers_one_finite_angle_junction(self):
        state = naca4_blunt_base_topology(base_fraction=0.0)
        self.assertEqual(state.base_thickness, 0.0)
        self.assertTrue(
            state.single_junction_topologically_admissible
        )
        self.assertGreater(state.tangent_gap_angle_rad, 0.0)

    def test_invalid_continuation_fraction_is_rejected(self):
        with self.assertRaises(BluntBaseTopologyError):
            naca4_blunt_base_topology(base_fraction=1.1)

    def test_preregistered_guard_passes_without_pressure_or_production(self):
        result = run_guard()
        self.assertEqual(result["stage_decision"], "GO")
        self.assertTrue(all(result["checks"].values()))
        self.assertFalse(result["production_activation_allowed"])


if __name__ == "__main__":
    unittest.main()
