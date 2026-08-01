"""Unit tests for S3ai read-only primitives; no preregistered run occurs."""
from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


PLATFORM = Path(__file__).resolve().parents[1]
if str(PLATFORM) not in sys.path:
    sys.path.insert(0, str(PLATFORM))

from actual_wake_reachable_pressure_obstruction_guard import (  # noqa: E402
    AUDIT,
    CASES,
    ReachablePressureGuardError,
    _load_frozen_contract,
    run_preregistered_observation,
)
from claim_runtime.actual_wake_reachable_pressure import (  # noqa: E402
    centered_tangent,
    dual_mass_norm,
    weak_pressure_step_residual,
)


class ActualWakeReachablePressureObstructionTests(unittest.TestCase):
    def test_frozen_contract_is_the_timestamped_preregistration(self) -> None:
        contract = _load_frozen_contract()
        self.assertTrue(CASES.name.endswith("_20260728_125228.yaml"))
        self.assertEqual(contract["status"], "preregistered_before_any_formal_execution")
        self.assertFalse(contract["decision"]["production_activation_allowed"])
        self.assertTrue(AUDIT.is_file())
        with self.assertRaisesRegex(
            ReachablePressureGuardError,
            "aborted before formal execution",
        ):
            run_preregistered_observation()

    def test_weak_residual_and_centered_dual_mass_observation(self) -> None:
        mass = np.array(((2.0, 0.5), (0.5, 1.5)))
        residual = weak_pressure_step_residual(
            mass,
            np.array((1.0, -2.0)),
            np.array((1.5, -1.0)),
            0.25,
            np.array((4.0, -8.0)),
        )
        self.assertTrue(np.allclose(residual, np.array((2.5, -0.25))))
        self.assertAlmostEqual(
            dual_mass_norm(residual, mass) ** 2,
            float(residual @ np.linalg.solve(mass, residual)),
        )
        tangent = centered_tangent(
            np.array((3.0, -1.0)), np.array((-1.0, 3.0)), 0.5
        )
        self.assertTrue(np.array_equal(tangent, np.array((4.0, -4.0))))


if __name__ == "__main__":
    unittest.main()
