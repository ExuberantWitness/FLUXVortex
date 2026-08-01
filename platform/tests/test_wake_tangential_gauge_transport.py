import sys
import unittest
from pathlib import Path

PLATFORM = Path(__file__).resolve().parents[1]
if str(PLATFORM) not in sys.path:
    sys.path.insert(0, str(PLATFORM))

from wake_tangential_gauge_transport_guard import (  # noqa: E402
    run as run_guard,
)


class WakeTangentialGaugeTransportTests(unittest.TestCase):
    def test_preregistered_continuum_identity_gate_passes(self):
        result = run_guard()
        self.assertEqual(result["stage_decision"], "GO")
        self.assertTrue(all(result["checks"].values()))
        metrics = result["aggregate_metrics"]
        self.assertLessEqual(
            metrics["ale_transport_residual_abs_max"],
            5.0e-14,
        )
        self.assertGreaterEqual(
            metrics["naive_frozen_mu_abs_error"],
            0.05,
        )
        self.assertFalse(result["production_activation_allowed"])


if __name__ == "__main__":
    unittest.main()
