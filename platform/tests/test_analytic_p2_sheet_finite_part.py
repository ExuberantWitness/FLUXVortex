import sys
import unittest
from pathlib import Path

PLATFORM = Path(__file__).resolve().parents[1]
if str(PLATFORM) not in sys.path:
    sys.path.insert(0, str(PLATFORM))

from analytic_p2_sheet_finite_part_guard import run as run_guard  # noqa: E402


class AnalyticP2SheetFinitePartTests(unittest.TestCase):
    def test_preregistered_analytic_finite_part_gate_passes(self):
        result = run_guard()
        self.assertEqual(result["stage_decision"], "GO")
        self.assertTrue(all(result["checks"].values()))
        metrics = result["aggregate_metrics"]
        self.assertEqual(
            metrics["manufactured_order_invariance_abs_max"],
            0.0,
        )
        self.assertLessEqual(
            metrics["actual_wake_last_abs_change"],
            1.0e-8,
        )
        self.assertLessEqual(
            metrics["rigid_channel_abs_max"],
            2.0e-9,
        )
        self.assertFalse(result["production_activation_allowed"])


if __name__ == "__main__":
    unittest.main()
