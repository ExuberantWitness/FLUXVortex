import sys
import unittest
from pathlib import Path

PLATFORM = Path(__file__).resolve().parents[1]
if str(PLATFORM) not in sys.path:
    sys.path.insert(0, str(PLATFORM))

from constrained_material_wake_advection_guard import (  # noqa: E402
    run as run_guard,
)


class ConstrainedMaterialWakeAdvectionTests(unittest.TestCase):
    def test_preregistered_constrained_heun_gate_passes(self):
        result = run_guard()
        self.assertEqual(result["stage_decision"], "GO")
        self.assertTrue(all(result["checks"].values()))
        metrics = result["aggregate_metrics"]
        self.assertEqual(metrics["attached_edge_abs_max"], 0.0)
        self.assertEqual(metrics["history_seam_abs_max"], 0.0)
        self.assertEqual(
            metrics["material_strength_mutation_abs_max"],
            0.0,
        )
        self.assertGreaterEqual(
            metrics["free_vertex_time_cauchy_ratio"],
            3.5,
        )
        self.assertEqual(metrics["mismatch_failure_count"], 1)
        self.assertFalse(result["production_activation_allowed"])


if __name__ == "__main__":
    unittest.main()
