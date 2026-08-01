import sys
import unittest
from pathlib import Path

PLATFORM = Path(__file__).resolve().parents[1]
if str(PLATFORM) not in sys.path:
    sys.path.insert(0, str(PLATFORM))

from p2_wake_gauge_transport_guard import run as run_guard  # noqa: E402


class P2WakeGaugeTransportTests(unittest.TestCase):
    def test_preregistered_semidiscrete_spatial_gate_passes(self):
        result = run_guard()
        self.assertEqual(result["stage_decision"], "GO")
        self.assertTrue(all(result["checks"].values()))
        metrics = result["aggregate_metrics"]
        self.assertEqual(metrics["mass_rank_deficiency_max"], 0)
        self.assertEqual(metrics["shared_trace_jump_abs_max"], 0.0)
        self.assertGreaterEqual(
            metrics["minimum_relative_L2_cauchy_ratio"],
            3.5,
        )
        self.assertFalse(result["production_activation_allowed"])


if __name__ == "__main__":
    unittest.main()
