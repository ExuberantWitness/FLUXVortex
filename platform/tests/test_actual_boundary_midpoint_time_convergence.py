import sys
import unittest
from pathlib import Path

PLATFORM = Path(__file__).resolve().parents[1]
if str(PLATFORM) not in sys.path:
    sys.path.insert(0, str(PLATFORM))

from actual_boundary_midpoint_time_convergence_guard import (  # noqa: E402
    run as run_guard,
)


class ActualBoundaryMidpointTimeConvergenceTests(unittest.TestCase):
    def test_preregistered_fixed_body_dual_cauchy_gate_passes(self):
        result = run_guard()
        self.assertEqual(result["stage_decision"], "GO")
        self.assertTrue(all(result["checks"].values()))
        self.assertEqual(
            result["aggregate_metrics"][
                "old_state_mutation_abs_max"
            ],
            0.0,
        )
        self.assertFalse(result["production_activation_allowed"])


if __name__ == "__main__":
    unittest.main()
