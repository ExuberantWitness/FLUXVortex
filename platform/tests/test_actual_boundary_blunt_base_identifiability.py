import sys
import unittest
from pathlib import Path

PLATFORM = Path(__file__).resolve().parents[1]
if str(PLATFORM) not in sys.path:
    sys.path.insert(0, str(PLATFORM))

from actual_boundary_blunt_base_identifiability_guard import (  # noqa: E402
    run as run_guard,
)
from claim_runtime.blunt_base_identifiability import (  # noqa: E402
    blunt_base_corner_witnesses,
)


class ActualBoundaryBluntBaseIdentifiabilityTests(
    unittest.TestCase
):
    def test_same_outer_input_admits_distinct_conservative_states(self):
        family = blunt_base_corner_witnesses(
            base_speed_ratios=(0.0, 0.25, 0.5, 0.75)
        )
        self.assertEqual(
            {member.observed_outer_speed for member in family},
            {1.0},
        )
        angle_spread = (
            family[-1].upper.delta_theta1
            - family[0].upper.delta_theta1
        )
        self.assertGreater(angle_spread, 0.5)
        for member in family:
            self.assertLess(
                member.upper.normalized_momentum_residual, 1.0e-13
            )
            self.assertGreaterEqual(member.upper.u_g_plus, 0.0)
            self.assertGreaterEqual(member.upper.u_g_minus, 0.0)

    def test_ratios_are_witness_inputs_not_an_implicit_selector(self):
        with self.assertRaises(ValueError):
            blunt_base_corner_witnesses(
                base_speed_ratios=(0.5, 0.25)
            )
        with self.assertRaises(ValueError):
            blunt_base_corner_witnesses(
                base_speed_ratios=(0.0, 1.0)
            )

    def test_preregistered_guard_falsifies_outer_only_closure(self):
        result = run_guard()
        self.assertEqual(result["stage_decision"], "GO")
        self.assertTrue(all(result["checks"].values()))
        self.assertFalse(result["outer_only_closure_identifiable"])
        self.assertFalse(result["production_activation_allowed"])


if __name__ == "__main__":
    unittest.main()
