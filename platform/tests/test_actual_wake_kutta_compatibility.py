"""Regression test for the S3ag Kutta compatibility counterexample gate."""
from __future__ import annotations

from pathlib import Path
import sys
import unittest


PLATFORM = Path(__file__).resolve().parents[1]
if str(PLATFORM) not in sys.path:
    sys.path.insert(0, str(PLATFORM))

from actual_wake_kutta_compatibility_guard import run  # noqa: E402


class ActualWakeKuttaCompatibilityTest(unittest.TestCase):
    def test_residual_only_closures_reproduce_physical_no_go(self) -> None:
        result = run()
        self.assertEqual(
            result["stage_decision"],
            "COUNTEREXAMPLE-GO / PHYSICAL-COMPATIBILITY-NO-GO",
        )
        self.assertTrue(all(result["checks"].values()))
        self.assertEqual(
            result["counterexample_decision"],
            "GO",
        )
        self.assertEqual(
            result["physical_compatibility_decision"],
            "NO-GO",
        )
        self.assertFalse(
            result["production_activation_allowed"]
        )

        birth = result["birth_counterexample"]
        self.assertLess(
            birth["edge_compatibility_defect_dt_order_abs"],
            result["thresholds"][
                "birth_edge_defect_dt_order_abs_max"
            ],
        )
        self.assertGreater(
            birth["edge_compatibility_defect_abs_min"],
            result["thresholds"]["birth_edge_defect_abs_min"],
        )

        pressure = result["pressure_counterexample"]
        self.assertEqual(len(pressure["roots"]), 2)
        self.assertGreater(
            pressure["observation_root_difference_abs_max"],
            result["thresholds"][
                "pressure_observation_root_difference_abs_min"
            ],
        )
        self.assertLess(
            pressure[
                "analytic_jacobian_directional_abs_error_max"
            ],
            result["thresholds"][
                "pressure_jacobian_directional_abs_error_max"
            ],
        )


if __name__ == "__main__":
    unittest.main()
