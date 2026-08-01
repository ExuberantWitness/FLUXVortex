"""Regression test for the S3u actual-wake DAE coupling audit."""
from __future__ import annotations

from pathlib import Path
import sys
import unittest


PLATFORM = Path(__file__).resolve().parents[1]
if str(PLATFORM) not in sys.path:
    sys.path.insert(0, str(PLATFORM))

from actual_wake_time_coupling_audit import run  # noqa: E402


class ActualWakeTimeCouplingAuditTest(unittest.TestCase):
    def test_preregistered_actual_stage_remains_coupled(self) -> None:
        result = run()
        self.assertEqual(result["stage_decision"], "COUPLED")
        self.assertTrue(all(result["checks"].values()))


if __name__ == "__main__":
    unittest.main()
