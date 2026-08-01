import sys
import unittest
from pathlib import Path

PLATFORM = Path(__file__).resolve().parents[1]
if str(PLATFORM) not in sys.path:
    sys.path.insert(0, str(PLATFORM))

from actual_boundary_oriented_wake_attachment_guard import (  # noqa: E402
    run as run_guard,
)


class ActualBoundaryOrientedWakeAttachmentTests(unittest.TestCase):
    def test_preregistered_oriented_attachment_gate_passes(self):
        result = run_guard()
        self.assertEqual(result["stage_decision"], "GO")
        self.assertTrue(all(result["checks"].values()))
        self.assertTrue(
            result["aggregate_metrics"][
                "rigid_coordinate_cut_order_reversed"
            ]
        )
        self.assertEqual(
            result["aggregate_metrics"][
                "invalid_identity_failure_count"
            ],
            3,
        )
        self.assertFalse(result["production_activation_allowed"])


if __name__ == "__main__":
    unittest.main()
