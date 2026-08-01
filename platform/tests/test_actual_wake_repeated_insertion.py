"""Regression test for the S3ac actual repeated-insertion gate."""
from __future__ import annotations

from pathlib import Path
import sys
import unittest


PLATFORM = Path(__file__).resolve().parents[1]
if str(PLATFORM) not in sys.path:
    sys.path.insert(0, str(PLATFORM))

from actual_wake_repeated_insertion_guard import run  # noqa: E402


class ActualWakeRepeatedInsertionTest(unittest.TestCase):
    def test_preregistered_actual_repeated_insertion_gate(self) -> None:
        result = run()
        self.assertEqual(result["stage_decision"], "GO")
        self.assertTrue(all(result["checks"].values()))


if __name__ == "__main__":
    unittest.main()
