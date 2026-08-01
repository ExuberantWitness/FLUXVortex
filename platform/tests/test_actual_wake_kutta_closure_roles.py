"""Regression test for the S3af Kutta-closure equation-role gate."""
from __future__ import annotations

from pathlib import Path
import sys
import unittest


PLATFORM = Path(__file__).resolve().parents[1]
if str(PLATFORM) not in sys.path:
    sys.path.insert(0, str(PLATFORM))

from actual_wake_kutta_closure_roles_guard import run  # noqa: E402


class ActualWakeKuttaClosureRolesTest(unittest.TestCase):
    def test_algebraic_role_go_and_quotient_no_go_are_reproducible(
        self,
    ) -> None:
        result = run()
        self.assertEqual(
            result["stage_decision"],
            "ALGEBRAIC-ROLE-GO / QUOTIENT-NO-GO",
        )
        self.assertTrue(
            all(result["algebraic_checks"].values())
        )
        self.assertTrue(
            all(
                result[
                    "quotient_counterexample_checks"
                ].values()
            )
        )
        self.assertFalse(
            result["production_activation_allowed"]
        )


if __name__ == "__main__":
    unittest.main()

