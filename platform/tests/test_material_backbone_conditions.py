import sys
import unittest
from pathlib import Path

import numpy as np

PLATFORM = Path(__file__).resolve().parents[1]
if str(PLATFORM) not in sys.path:
    sys.path.insert(0, str(PLATFORM))

from claim_runtime.material_backbone_conditions import (  # noqa: E402
    MaterialBackboneConditionError,
    material_backbone_condition_diagnostics,
)


def grid(nu=81, nv=41):
    u = np.linspace(-1.0, 1.0, nu)
    v = np.linspace(-0.8, 0.8, nv)
    return u, v, np.meshgrid(u, v, indexing="ij")


def directions(shape, value=(1.0, 0.0)):
    result = np.empty(shape+(2,), dtype=float)
    result[...] = value
    return result


def diagnostics(
    curvature,
    direction,
    u,
    v,
    *,
    stationarity=1.0e-10,
    negative=1.0e-4,
    gap=1.0e-8,
):
    return material_backbone_condition_diagnostics(
        curvature,
        direction,
        u=u,
        v=v,
        stationarity_tolerance=stationarity,
        negative_curvature_tolerance=negative,
        eigenvalue_gap_tolerance=gap,
    )


class MaterialBackboneConditionTests(unittest.TestCase):
    def test_two_dimensional_gaussian_ridge_is_detected(self):
        u, v, (uu, _) = grid()
        sigma = 0.35
        largest = 2.0*np.exp(-(uu/sigma)**2)
        curvature = np.stack((np.zeros_like(largest), largest), axis=-1)
        result = diagnostics(
            curvature,
            directions(largest.shape),
            u,
            v,
        )
        centre = len(u)//2
        self.assertTrue(np.all(
            result.two_dimensional_candidate[centre]
        ))
        self.assertFalse(np.any(
            result.one_dimensional_candidate[centre]
        ))
        self.assertFalse(np.any(
            result.two_dimensional_candidate[
                np.arange(len(u)) != centre
            ]
        ))
        self.assertTrue(np.all(
            result.directional_second_derivative[centre] < 0.0
        ))

    def test_one_dimensional_isotropic_maximum_is_detected(self):
        u, v, (uu, vv) = grid()
        largest = 1.7*np.exp(
            -(uu/0.33)**2-(vv/0.27)**2
        )
        curvature = np.stack((largest, largest), axis=-1)
        result = diagnostics(
            curvature,
            directions(largest.shape),
            u,
            v,
        )
        centre = (len(u)//2, len(v)//2)
        self.assertTrue(result.one_dimensional_candidate[centre])
        self.assertFalse(result.two_dimensional_candidate[centre])
        self.assertTrue(np.all(
            result.hessian_eigenvalues[centre] < 0.0
        ))
        self.assertEqual(
            np.count_nonzero(result.one_dimensional_candidate),
            1,
        )

    def test_downwelling_and_directional_minimum_are_rejected(self):
        u, v, (uu, vv) = grid()
        downwelling = -np.exp(-uu**2-vv**2)
        result = diagnostics(
            np.stack((downwelling, downwelling), axis=-1),
            directions(downwelling.shape),
            u,
            v,
        )
        centre = (len(u)//2, len(v)//2)
        self.assertFalse(result.positive_upwelling[centre])
        self.assertFalse(result.candidate[centre])

        minimum = 1.0+uu**2
        result = diagnostics(
            np.stack((np.zeros_like(minimum), minimum), axis=-1),
            directions(minimum.shape),
            u,
            v,
        )
        self.assertAlmostEqual(
            result.directional_derivative[centre],
            0.0,
            places=12,
        )
        self.assertGreater(
            result.directional_second_derivative[centre],
            0.0,
        )
        self.assertFalse(result.candidate[centre])

    def test_derivative_fields_converge_at_second_order(self):
        errors = []
        for count in (41, 81, 161):
            u, v, (uu, _) = grid(nu=count, nv=31)
            sigma = 0.43
            largest = np.exp(-(uu/sigma)**2)
            curvature = np.stack(
                (np.zeros_like(largest), largest),
                axis=-1,
            )
            result = diagnostics(
                curvature,
                directions(largest.shape),
                u,
                v,
            )
            exact_first = (
                -2.0*uu/sigma**2*largest
            )
            exact_second = (
                (4.0*uu**2/sigma**4-2.0/sigma**2)
                *largest
            )
            interior = np.s_[3:-3, 3:-3]
            errors.append(max(
                float(np.max(np.abs(
                    result.directional_derivative[interior]
                    -exact_first[interior]
                ))),
                float(np.max(np.abs(
                    result.directional_second_derivative[interior]
                    -exact_second[interior]
                ))),
            ))
        self.assertGreater(errors[0]/errors[1], 3.7)
        self.assertGreater(errors[1]/errors[2], 3.7)

    def test_linear_material_reparametrization_preserves_candidates(self):
        u, v, (uu, _) = grid()
        largest = 2.0*np.exp(-(uu/0.35)**2)
        curvature = np.stack((np.zeros_like(largest), largest), axis=-1)
        baseline = diagnostics(
            curvature,
            directions(largest.shape),
            u,
            v,
        )
        rescaled = diagnostics(
            curvature,
            directions(largest.shape),
            3.2*u+0.4,
            0.7*v-0.2,
        )
        np.testing.assert_array_equal(
            rescaled.candidate,
            baseline.candidate,
        )

    def test_invalid_fields_and_tolerances_fail(self):
        u, v, (uu, _) = grid()
        largest = np.exp(-uu**2)
        curvature = np.stack((np.zeros_like(largest), largest), axis=-1)
        with self.assertRaisesRegex(
            MaterialBackboneConditionError,
            "must have",
        ):
            diagnostics(
                curvature,
                np.zeros((3, 2)),
                u,
                v,
            )
        invalid = curvature.copy()
        invalid[3, 4, 1] = np.nan
        with self.assertRaisesRegex(
            MaterialBackboneConditionError,
            "non-finite",
        ):
            diagnostics(
                invalid,
                directions(largest.shape),
                u,
                v,
            )
        with self.assertRaisesRegex(
            MaterialBackboneConditionError,
            "finite and positive",
        ):
            diagnostics(
                curvature,
                directions(largest.shape),
                u,
                v,
                stationarity=0.0,
            )


if __name__ == "__main__":
    unittest.main()
