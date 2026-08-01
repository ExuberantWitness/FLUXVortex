"""Definition tests for the frozen 31-history S3ai-v2 enumeration."""
from __future__ import annotations

from pathlib import Path
import sys
import unittest


PLATFORM = Path(__file__).resolve().parents[1]
if str(PLATFORM) not in sys.path:
    sys.path.insert(0, str(PLATFORM))

from claim_runtime.reachable_pressure_cases import (  # noqa: E402
    frozen_execution_accounting,
    frozen_history_cases,
    frozen_tangent_configurations,
    mixed_cube_configuration_names,
)


class ReachablePressureCaseTests(unittest.TestCase):
    def test_frozen_history_count_and_solver_accounting(self) -> None:
        accounting = frozen_execution_accounting()
        self.assertEqual(accounting["histories"], 31)
        self.assertEqual(accounting["nominal_signed_histories"], 22)
        self.assertEqual(accounting["zero_histories"], 6)
        self.assertEqual(accounting["fresh_repeat_histories"], 3)
        self.assertEqual(accounting["measurement_steps"], 380)
        self.assertEqual(accounting["compatible_presteps"], 31)
        self.assertEqual(accounting["marcher_steps"], 411)
        self.assertEqual(accounting["half_full_solves"], 822)
        self.assertEqual(accounting["observed_stages"], 791)
        self.assertEqual(
            accounting["histories_by_timestep"],
            {"0.0625": 18, "0.125": 10, "0.25": 3},
        )

    def test_every_history_has_exactly_one_compatible_prestep(self) -> None:
        for case in frozen_history_cases():
            self.assertEqual(
                case.marcher_steps,
                case.measurement_steps + 1,
            )
            self.assertEqual(
                case.half_full_solves,
                2 * case.marcher_steps,
            )

    def test_axis_and_complete_mixed_cube_labels_are_unambiguous(self) -> None:
        configurations = frozen_tangent_configurations()
        cube = mixed_cube_configuration_names()
        self.assertEqual(set(cube), {f"{index:03b}" for index in range(8)})
        self.assertEqual(set(cube.values()), {
            "A",
            "E2",
            "DT2",
            "Q10",
            "E2_DT2",
            "E2_Q10",
            "DT2_Q10",
            "E2_DT2_Q10",
        })
        self.assertEqual(configurations["A"].epsilon, 0.0025)
        self.assertEqual(configurations["A"].timestep, 0.0625)
        self.assertEqual(configurations["A"].quadrature_order, 12)


if __name__ == "__main__":
    unittest.main()
