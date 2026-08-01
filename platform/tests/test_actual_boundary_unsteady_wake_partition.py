import sys
import unittest
from pathlib import Path

PLATFORM = Path(__file__).resolve().parents[1]
if str(PLATFORM) not in sys.path:
    sys.path.insert(0, str(PLATFORM))

from actual_boundary_unsteady_wake_partition_guard import (  # noqa: E402
    run as run_guard,
)


class ActualBoundaryUnsteadyWakePartitionTests(unittest.TestCase):
    def test_preregistered_affine_kelvin_ledger_gate_passes(self):
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
