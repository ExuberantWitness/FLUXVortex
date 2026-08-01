import sys
import unittest
from pathlib import Path

PLATFORM = Path(__file__).resolve().parents[1]
if str(PLATFORM) not in sys.path:
    sys.path.insert(0, str(PLATFORM))

from p2_wake_inflow_trace_guard import run as run_guard  # noqa: E402


class P2WakeInflowTraceTests(unittest.TestCase):
    def test_preregistered_typed_inflow_gate_passes(self):
        result = run_guard()
        self.assertEqual(result["stage_decision"], "GO")
        self.assertTrue(all(result["checks"].values()))
        metrics = result["aggregate_metrics"]
        self.assertEqual(
            metrics["boundary_role_partition_failure_count"],
            0,
        )
        self.assertEqual(metrics["constrained_nonbody_dof_count"], 0)
        self.assertGreaterEqual(
            metrics["clamp_rate_injection_normalized_residual"],
            0.99,
        )
        self.assertGreaterEqual(
            metrics["propagation_time_cauchy_ratio"],
            2.5,
        )
        self.assertEqual(metrics["invalid_role_failure_count"], 3)
        self.assertFalse(result["production_activation_allowed"])


if __name__ == "__main__":
    unittest.main()
