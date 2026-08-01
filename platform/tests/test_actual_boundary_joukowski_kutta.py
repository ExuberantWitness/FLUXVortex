import sys
import unittest
from pathlib import Path

import numpy as np

PLATFORM = Path(__file__).resolve().parents[1]
if str(PLATFORM) not in sys.path:
    sys.path.insert(0, str(PLATFORM))

from actual_boundary_joukowski_kutta_guard import run as run_guard  # noqa: E402
from claim_runtime.joukowski_kutta_oracle import (  # noqa: E402
    joukowski_p2_kutta_trace,
)


class ActualBoundaryJoukowskiKuttaTests(unittest.TestCase):
    def test_kutta_condition_selects_cut_jump_and_circulation(self):
        case = joukowski_p2_kutta_trace(panel_count=32)
        self.assertLess(case.kutta_numerator_residual, 1.0e-13)
        self.assertAlmostEqual(
            case.potential_jump, case.kutta_circulation, places=13
        )
        self.assertAlmostEqual(
            case.circulation, case.kutta_circulation, places=13
        )

    def test_constant_gauge_preserves_velocity_and_pressure_force(self):
        base = joukowski_p2_kutta_trace(panel_count=64)
        shifted = joukowski_p2_kutta_trace(
            panel_count=64, potential_gauge=7.25
        )
        np.testing.assert_allclose(
            shifted.tangential_velocity,
            base.tangential_velocity,
            rtol=0.0,
            atol=1.0e-12,
        )
        np.testing.assert_allclose(
            shifted.pressure_force,
            base.pressure_force,
            rtol=0.0,
            atol=1.0e-12,
        )

    def test_preregistered_guard_passes_without_production_activation(self):
        result = run_guard()
        self.assertEqual(result["stage_decision"], "GO")
        self.assertTrue(all(result["checks"].values()))
        self.assertFalse(result["production_activation_allowed"])


if __name__ == "__main__":
    unittest.main()
