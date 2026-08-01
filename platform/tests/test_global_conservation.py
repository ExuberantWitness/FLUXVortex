import sys
import unittest
from pathlib import Path

import numpy as np

PLATFORM = Path(__file__).resolve().parents[1]
if str(PLATFORM) not in sys.path:
    sys.path.insert(0, str(PLATFORM))

from claim_runtime.global_conservation import (  # noqa: E402
    GlobalConservationError,
    GlobalConservationSystem,
)


class GlobalConservationTests(unittest.TestCase):
    def test_manufactured_named_blocks_recover_one_global_state(self):
        truth = np.array([0.18, -0.07, 0.11, -0.04])
        system = GlobalConservationSystem(
            ["Gamma_b0", "Gamma_b1", "mu_new0", "mu_new1"],
            velocity_reference=8.0,
            length_reference=0.287,
        )
        blocks = {
            "no_penetration": np.array(
                [[1.2, 0.1, 0.5, -0.2], [0.2, 1.1, -0.1, 0.4]]
            ),
            "trace_continuity": np.array([[0.0, 0.0, 1.0, -1.0]]),
            "vorticity_compatibility": np.array(
                [[0.0, 0.0, 2.0, 1.0]]
            ),
            "material_kelvin": np.array([[1.0, 1.0, 1.0, 1.0]]),
            "kutta_interface": np.array([[1.0, 0.0, -1.0, 0.0]]),
            "mirror_symmetry": np.array([[1.0, -1.0, 0.0, 0.0]]),
            "free_edge": np.array([[0.0, 0.0, 0.0, 1.0]]),
        }
        for name, matrix in blocks.items():
            system.add_block(name, matrix, matrix @ truth)
        result = system.solve(normalized_tolerance=1.0e-12)
        self.assertTrue(result.passed)
        self.assertEqual(result.rank, len(truth))
        self.assertGreater(result.equation_count, len(truth))
        np.testing.assert_allclose(result.values, truth, atol=2.0e-15)
        self.assertEqual(
            set(result.block_reports),
            set(blocks),
        )

    def test_lesp_cannot_be_added_as_amplitude_equation(self):
        system = GlobalConservationSystem(
            ["Gamma_b", "mu_new"],
            velocity_reference=1.0,
            length_reference=1.0,
        )
        with self.assertRaises(GlobalConservationError):
            system.add_block(
                "lesp_amplitude",
                np.eye(2),
                np.zeros(2),
            )

    def test_rank_deficiency_and_duplicate_blocks_fail(self):
        system = GlobalConservationSystem(
            ["Gamma_b", "mu_new"],
            velocity_reference=1.0,
            length_reference=1.0,
        )
        system.add_block("material_kelvin", [[1.0, 0.0]], [0.2])
        with self.assertRaises(GlobalConservationError):
            system.add_block("material_kelvin", [[0.0, 1.0]], [0.1])
        with self.assertRaises(GlobalConservationError):
            system.solve()


if __name__ == "__main__":
    unittest.main()

