"""Regression test for the S3y actual-wake newborn transition gate."""
from __future__ import annotations

from pathlib import Path
import sys
import unittest


PLATFORM = Path(__file__).resolve().parents[1]
if str(PLATFORM) not in sys.path:
    sys.path.insert(0, str(PLATFORM))

from actual_wake_newborn_transition_guard import run  # noqa: E402


class ActualWakeNewbornTransitionTest(unittest.TestCase):
    def test_preregistered_transition_gate_remains_go(self) -> None:
        result = run()
        self.assertEqual(result["stage_decision"], "GO")
        self.assertTrue(all(result["checks"].values()))


if __name__ == "__main__":
    unittest.main()
