import sys
import unittest
from pathlib import Path

import numpy as np

PLATFORM = Path(__file__).resolve().parents[1]
if str(PLATFORM) not in sys.path:
    sys.path.insert(0, str(PLATFORM))

from actual_boundary_circulation_cut_guard import run as run_guard  # noqa: E402
from claim_runtime.circulation_cut_oracle import (  # noqa: E402
    CirculationCutError,
    circular_p2_trace,
)


class ActualBoundaryCirculationCutTests(unittest.TestCase):
    def test_closed_trace_gradient_telescopes_to_zero(self):
        case = circular_p2_trace(
            panel_count=8,
            topology="closed",
            prescribed_circulation=0.8,
        )
        self.assertLess(abs(case.circulation), 1.0e-13)
        self.assertLess(abs(case.telescoping_circulation), 1.0e-13)
        self.assertEqual(case.potential_jump, 0.0)

    def test_cut_jump_and_gradient_integral_equal_circulation(self):
        case = circular_p2_trace(
            panel_count=8,
            topology="cut",
            prescribed_circulation=0.8,
        )
        self.assertAlmostEqual(case.potential_jump, 0.8, places=14)
        self.assertAlmostEqual(case.circulation, 0.8, places=14)
        self.assertAlmostEqual(
            case.telescoping_circulation, 0.8, places=14
        )

    def test_constant_potential_gauge_changes_neither_velocity_nor_force(self):
        base = circular_p2_trace(panel_count=32, topology="cut")
        shifted = circular_p2_trace(
            panel_count=32,
            topology="cut",
            potential_gauge=7.25,
        )
        np.testing.assert_allclose(
            shifted.tangential_velocity,
            base.tangential_velocity,
            rtol=0.0,
            atol=1.0e-13,
        )
        np.testing.assert_allclose(
            shifted.pressure_force,
            base.pressure_force,
            rtol=0.0,
            atol=1.0e-13,
        )

    def test_preregistered_guard_passes_without_production_activation(self):
        result = run_guard()
        self.assertEqual(result["stage_decision"], "GO")
        self.assertTrue(all(result["checks"].values()))
        self.assertFalse(result["production_activation_allowed"])

    def test_invalid_topology_is_rejected(self):
        with self.assertRaises(CirculationCutError):
            circular_p2_trace(panel_count=8, topology="periodic")


if __name__ == "__main__":
    unittest.main()
