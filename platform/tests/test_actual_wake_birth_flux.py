"""Regression test for the S3ae material birth-flux gate."""
from __future__ import annotations

from pathlib import Path
import sys
import unittest


PLATFORM = Path(__file__).resolve().parents[1]
if str(PLATFORM) not in sys.path:
    sys.path.insert(0, str(PLATFORM))

from actual_wake_birth_flux_guard import run  # noqa: E402


class ActualWakeBirthFluxTest(unittest.TestCase):
    def test_preregistered_material_birth_flux_gate_remains_go(
        self,
    ) -> None:
        result = run()
        self.assertEqual(result["stage_decision"], "GO")
        self.assertTrue(all(result["checks"].values()))


if __name__ == "__main__":
    unittest.main()
