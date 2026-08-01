import sys
import unittest
from pathlib import Path

PLATFORM = Path(__file__).resolve().parents[1]
if str(PLATFORM) not in sys.path:
    sys.path.insert(0, str(PLATFORM))

from actual_body_wake_velocity_ledger_guard import run as run_guard  # noqa: E402
from claim_runtime.actual_body_wake_velocity import (  # noqa: E402
    VALIDATED_EDGE_QUADRATURE,
)


class ActualBodyWakeVelocityLedgerTests(unittest.TestCase):
    def test_validated_operator_runs_full_preregistered_ledger(self):
        self.assertEqual(
            VALIDATED_EDGE_QUADRATURE,
            "target_sinh_analytic_sheet",
        )
        result = run_guard(
            edge_quadrature=VALIDATED_EDGE_QUADRATURE,
            results_path=(
                PLATFORM / "docs" / "diag"
                / "actual_body_wake_velocity_ledger_analytic_results.json"
            ),
        )
        self.assertEqual(result["stage_decision"], "GO")
        self.assertTrue(all(result["checks"].values()))
        self.assertEqual(
            result["aggregate_metrics"]["ledger_closure_abs_max"],
            0.0,
        )
        self.assertFalse(result["production_activation_allowed"])


if __name__ == "__main__":
    unittest.main()
