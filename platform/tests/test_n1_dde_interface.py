import sys
import unittest
from pathlib import Path

import numpy as np

PLATFORM = Path(__file__).resolve().parents[1]
if str(PLATFORM) not in sys.path:
    sys.path.insert(0, str(PLATFORM))

from claim_runtime.n1_dde_interface import (  # noqa: E402
    N1DDEInterfaceError,
    dde_mu_to_n1_gamma,
    n1_gamma_to_dde_mu,
)
from claim_runtime.coupled_lesp_dde import mirror_halfwing_surface  # noqa: E402
from claim_runtime.distributed_doublet import (  # noqa: E402
    QuadraticDoubletSurface,
)
from claim_runtime.hirato_shadow import mirrored_ring_field  # noqa: E402
from diff_uvlm_unsteady import _ring_vel  # noqa: E402


class N1DDEInterfaceTests(unittest.TestCase):
    def test_aligned_normal_requires_negative_dde_strength(self):
        gamma = np.array([0.2, -0.4])
        normal = np.array([[0.0, 0.0, 1.0]] * 2)
        mu = n1_gamma_to_dde_mu(
            gamma,
            n1_normal=normal,
            dde_normal=normal,
        )
        np.testing.assert_allclose(mu, -gamma)
        np.testing.assert_allclose(
            dde_mu_to_n1_gamma(
                mu,
                n1_normal=normal,
                dde_normal=normal,
            ),
            gamma,
        )

    def test_reversed_dde_normal_reverses_adapter_sign_once(self):
        gamma = np.array([0.2, -0.4])
        n1 = np.array([[0.0, 0.0, 1.0]] * 2)
        dde = -n1
        np.testing.assert_allclose(
            n1_gamma_to_dde_mu(
                gamma,
                n1_normal=n1,
                dde_normal=dde,
            ),
            gamma,
        )

    def test_oblique_or_zero_normals_fail(self):
        with self.assertRaises(N1DDEInterfaceError):
            n1_gamma_to_dde_mu(
                [0.2],
                n1_normal=[[0.0, 0.0, 1.0]],
                dde_normal=[[0.0, 1.0, 0.0]],
            )
        with self.assertRaises(N1DDEInterfaceError):
            n1_gamma_to_dde_mu(
                [0.2],
                n1_normal=[[0.0, 0.0, 0.0]],
                dde_normal=[[0.0, 0.0, 1.0]],
            )

    def test_halfwing_image_preserves_n1_dde_adapter_and_vector_symmetry(self):
        gamma = 0.37
        qfl = np.array([0.0, 0.2, 0.0])
        qfr = np.array([0.0, 1.2, 0.0])
        qbr = np.array([1.0, 1.2, 0.0])
        qbl = np.array([1.0, 0.2, 0.0])
        ring = np.array([[qfl, qfr, qbr, qbl]])
        surface = QuadraticDoubletSurface(
            np.array([qfl, qbl, qbr, qfr]),
            [[0, 1, 2], [0, 2, 3]],
            np.full((2, 6), -gamma),
        )
        points = np.array(
            [[0.2, 0.4, 0.8], [1.8, 0.7, 1.2], [-0.6, 0.3, 0.5]]
        )
        mirror_points = points.copy()
        mirror_points[:, 1] *= -1.0
        original = np.asarray(
            [gamma * _ring_vel(point, ring[0]) for point in points]
        )
        n1_image = np.asarray(
            [
                gamma * _ring_vel(point, mirrored_ring_field(ring)[0])
                for point in mirror_points
            ]
        )
        dde_image = mirror_halfwing_surface(
            surface
        ).induced_velocity_line_reduced(
            mirror_points,
            quadrature_order=96,
        )
        np.testing.assert_allclose(
            n1_image,
            original * np.array([1.0, -1.0, 1.0]),
            rtol=2.0e-10,
            atol=2.0e-11,
        )
        np.testing.assert_allclose(
            dde_image,
            n1_image,
            rtol=2.0e-9,
            atol=2.0e-11,
        )


if __name__ == "__main__":
    unittest.main()
