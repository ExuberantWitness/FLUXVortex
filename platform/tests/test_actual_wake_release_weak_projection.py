"""Regression test for the S3z weak release-velocity decision gate."""
from __future__ import annotations

from pathlib import Path
import sys
import unittest


PLATFORM = Path(__file__).resolve().parents[1]
if str(PLATFORM) not in sys.path:
    sys.path.insert(0, str(PLATFORM))

from actual_wake_release_weak_projection_guard import run  # noqa: E402


class ActualWakeReleaseWeakProjectionTest(unittest.TestCase):
    def test_preregistered_decision_remains_weak_go(self) -> None:
        result = run()
        self.assertEqual(result["stage_decision"], "WEAK-GO")
        self.assertEqual(
            result["candidate_verdicts"]["point_trace"],
            "FALSIFIED",
        )
        self.assertEqual(
            result["candidate_verdicts"]["global_weak_P1"],
            "GO",
        )
        self.assertTrue(all(result["checks"].values()))


if __name__ == "__main__":
    unittest.main()
