import sys
import unittest
from pathlib import Path

PLATFORM = Path(__file__).resolve().parents[1]
if str(PLATFORM) not in sys.path:
    sys.path.insert(0, str(PLATFORM))

from actual_boundary_material_wake_history_guard import (  # noqa: E402
    run as run_guard,
)


class ActualBoundaryMaterialWakeHistoryTests(unittest.TestCase):
    def test_preregistered_shape_regular_history_gate_passes(self):
        result = run_guard()
        self.assertEqual(result["stage_decision"], "GO")
        self.assertTrue(all(result["checks"].values()))
        self.assertEqual(
            result["aggregate_metrics"][
                "independent_wake_unknown_count_max"
            ],
            0,
        )
        self.assertEqual(
            result["aggregate_metrics"][
                "history_geometry_gap_abs_max"
            ],
            0.0,
        )
        self.assertFalse(result["production_activation_allowed"])


if __name__ == "__main__":
    unittest.main()
