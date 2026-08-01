import sys
import unittest
from pathlib import Path

import numpy as np

PLATFORM = Path(__file__).resolve().parents[1]
if str(PLATFORM) not in sys.path:
    sys.path.insert(0, str(PLATFORM))

from claim_runtime.distributed_doublet import (  # noqa: E402
    DistributedDoubletError,
    QuadraticDoubletSurface,
)
from claim_runtime.doublet_potential import (  # noqa: E402
    dde_potential_side_limits,
    surface_doublet_potential,
    surface_sheet_average_potential,
)


class DoubletPotentialTests(unittest.TestCase):
    def setUp(self):
        self.vertices = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.2, 0.1, 0.0],
                [0.15, 0.95, 0.0],
            ]
        )
        self.faces = np.array([[0, 1, 2]])
        self.mu = np.array(
            [[0.2, -0.1, 0.5, 0.35, 0.15, -0.05]]
        )
        self.surface = QuadraticDoubletSurface(
            self.vertices, self.faces, self.mu
        )

    def test_potential_gradient_matches_existing_doublet_velocity(self):
        points = np.array(
            [
                [0.25, 0.22, 0.7],
                [1.4, -0.4, 0.8],
                [-0.3, 0.5, -0.6],
            ]
        )
        step = 2.0e-6
        gradient = np.zeros_like(points)
        for axis in range(3):
            offset = np.zeros(3)
            offset[axis] = step
            high = surface_doublet_potential(
                self.surface,
                points+offset,
                quadrature_order=48,
            )
            low = surface_doublet_potential(
                self.surface,
                points-offset,
                quadrature_order=48,
            )
            gradient[:, axis] = (high-low)/(2.0*step)
        velocity = self.surface.induced_velocity(
            points, quadrature_order=48
        )
        np.testing.assert_allclose(
            gradient, velocity, rtol=2.0e-8, atol=2.0e-10
        )

    def test_planar_owned_principal_value_is_zero(self):
        barycentric = np.array(
            [
                [0.2, 0.3, 0.5],
                [1.0/3.0, 1.0/3.0, 1.0/3.0],
            ]
        )
        mean = surface_sheet_average_potential(
            self.surface,
            np.zeros(2, dtype=int),
            barycentric,
        )
        np.testing.assert_allclose(mean, 0.0, atol=0.0)

    def test_aligned_plemelj_jump_is_n1_chi_minus_mu(self):
        mean = np.array([0.2, -0.7, 1.1])
        dde_mu = np.array([0.4, -0.3, 0.9])
        limits = dde_potential_side_limits(mean, dde_mu)
        np.testing.assert_allclose(
            limits.potential_plus-limits.potential_minus,
            -dde_mu,
            atol=1.0e-16,
        )
        np.testing.assert_allclose(
            0.5*(limits.potential_plus+limits.potential_minus),
            mean,
            atol=1.0e-16,
        )
        self.assertLessEqual(limits.max_jump_residual, 1.0e-15)

    def test_owner_and_shape_errors_fail(self):
        with self.assertRaises(DistributedDoubletError):
            surface_sheet_average_potential(
                self.surface,
                [1],
                [[0.2, 0.3, 0.5]],
            )
        with self.assertRaises(DistributedDoubletError):
            dde_potential_side_limits([0.0, 1.0], [0.0])


if __name__ == "__main__":
    unittest.main()
