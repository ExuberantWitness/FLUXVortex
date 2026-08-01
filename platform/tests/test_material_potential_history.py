import sys
import unittest
from pathlib import Path

import numpy as np

PLATFORM = Path(__file__).resolve().parents[1]
if str(PLATFORM) not in sys.path:
    sys.path.insert(0, str(PLATFORM))

from claim_runtime.distributed_doublet import (  # noqa: E402
    QuadraticDoubletSurface,
)
from claim_runtime.material_potential_history import (  # noqa: E402
    MaterialPotentialHistoryError,
    material_potential_history_rate,
    three_stage_lagrange_derivative_weights,
)


class MaterialPotentialHistoryTests(unittest.TestCase):
    def setUp(self):
        self.vertices = np.asarray(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        )
        self.faces = np.asarray([[0, 1, 2]])
        self.mu = np.asarray([[0.0, 0.2, -0.1, 0.1, 0.05, -0.05]])

    def surface(self, shift=0.0, mu=None):
        return QuadraticDoubletSurface(
            self.vertices + np.asarray([shift, 0.0, 0.0]),
            self.faces,
            self.mu if mu is None else mu,
        )

    def test_three_stage_weights_differentiate_quadratic_exactly(self):
        times = np.asarray([-0.3, 0.0, 0.2])
        for target_index in range(3):
            weights = three_stage_lagrange_derivative_weights(
                times,
                target_index=target_index,
            )
            values = 1.7 * times**2 - 0.4 * times + 2.0
            derivative = float(weights @ values)
            expected = 3.4 * times[target_index] - 0.4
            self.assertAlmostEqual(derivative, expected, places=14)

    def test_explicit_geometry_history_returns_finite_rate(self):
        surfaces = [
            self.surface(-0.02),
            self.surface(0.0),
            self.surface(0.02),
        ]
        point = np.asarray([[0.3, 0.2, 0.8]])
        wall_history = np.repeat(point[None, :, :], 3, axis=0)
        result = material_potential_history_rate(
            surfaces,
            wall_history,
            [-0.1, 0.0, 0.1],
            target_index=2,
            quadrature_order=24,
        )
        self.assertTrue(np.all(np.isfinite(result.wall_material_rate)))
        self.assertEqual(result.potential_by_stage.shape, (3, 1))
        self.assertEqual(result.max_material_mu_residual, 0.0)
        self.assertTrue(result.topology_equal)

    def test_missing_middle_stage_is_rejected(self):
        with self.assertRaises(MaterialPotentialHistoryError):
            material_potential_history_rate(
                [self.surface(), self.surface()],
                np.zeros((2, 1, 3)),
                [0.0, 0.1],
            )

    def test_material_strength_change_is_rejected(self):
        changed = self.mu.copy()
        changed[0, 3] += 1.0e-8
        with self.assertRaises(MaterialPotentialHistoryError):
            material_potential_history_rate(
                [self.surface(), self.surface(), self.surface(mu=changed)],
                np.repeat(
                    np.asarray([[[0.3, 0.2, 0.8]]]),
                    3,
                    axis=0,
                ),
                [-0.1, 0.0, 0.1],
            )


if __name__ == "__main__":
    unittest.main()
