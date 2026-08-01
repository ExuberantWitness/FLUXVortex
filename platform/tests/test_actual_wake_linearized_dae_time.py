"""Regression test for the S3v actual linearized-DAE time gate."""
from __future__ import annotations

from pathlib import Path
import sys
import unittest


PLATFORM = Path(__file__).resolve().parents[1]
if str(PLATFORM) not in sys.path:
    sys.path.insert(0, str(PLATFORM))

from actual_wake_linearized_dae_time_guard import run  # noqa: E402


class ActualWakeLinearizedDAETimeTest(unittest.TestCase):
    def test_preregistered_local_gate_remains_explicit_go(self) -> None:
        result = run()
        self.assertEqual(result["stage_decision"], "EXPLICIT-GO")
        self.assertTrue(all(result["checks"].values()))


if __name__ == "__main__":
    unittest.main()
