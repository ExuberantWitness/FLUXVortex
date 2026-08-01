import sys
import unittest
from pathlib import Path

import numpy as np

PLATFORM = Path(__file__).resolve().parents[1]
if str(PLATFORM) not in sys.path:
    sys.path.insert(0, str(PLATFORM))

from claim_runtime.work_conjugate_transfer import (  # noqa: E402
    LoadTransferError,
    resultant,
    rigid_body_jacobian,
    rigid_resultant_report,
    transfer_generalized,
    transfer_linear_nodal,
)


class WorkConjugateTransferTests(unittest.TestCase):
    def setUp(self):
        self.origin = np.array([0.2, -0.1, 0.3])
        self.points = np.array(
            [
                [0.4, 0.2, 0.1],
                [1.1, -0.5, 0.7],
                [-0.2, 0.8, 0.5],
                [0.6, 0.3, -0.4],
            ]
        )
        self.force = np.array(
            [
                [1.2, -0.4, 2.1],
                [-0.7, 1.5, 0.2],
                [0.3, 0.8, -1.1],
                [1.6, -0.2, 0.9],
            ]
        )

    def test_rigid_columns_recover_exact_force_and_moment(self):
        jacobian = rigid_body_jacobian(self.points, origin=self.origin)
        load = transfer_generalized(self.force, jacobian)
        force, moment = resultant(
            self.points,
            self.force,
            origin=self.origin,
        )
        np.testing.assert_allclose(load.values[:3], force, atol=2e-15)
        np.testing.assert_allclose(load.values[3:], moment, atol=2e-15)
        report = rigid_resultant_report(
            self.points,
            self.force,
            load.values,
            origin=self.origin,
        )
        self.assertTrue(report.passed)

    def test_virtual_work_holds_for_arbitrary_nonlinear_map_jacobian(self):
        rng = np.random.default_rng(20260727)
        jacobian = rng.normal(size=(len(self.points), 3, 11))
        load = transfer_generalized(self.force, jacobian)
        variations = rng.normal(size=(17, 11))
        report = load.virtual_work_report(
            variations,
            absolute_tolerance=5e-14,
            relative_tolerance=5e-14,
        )
        self.assertTrue(report.passed)
        self.assertLess(report.max_absolute_residual, 5e-14)

    def test_linear_H_transpose_is_same_generalized_operation(self):
        rng = np.random.default_rng(8)
        mapping = rng.normal(size=(self.force.size, 9))
        jacobian = mapping.reshape(len(self.force), 3, 9)
        expected = transfer_generalized(self.force, jacobian).values
        actual = transfer_linear_nodal(mapping, self.force)
        np.testing.assert_allclose(actual, expected, atol=2e-15)

    def test_bad_shapes_and_nonfinite_values_fail(self):
        with self.assertRaises(LoadTransferError):
            transfer_generalized(self.force, np.zeros((4, 2, 6)))
        with self.assertRaises(LoadTransferError):
            transfer_generalized(
                np.array([[np.nan, 0.0, 0.0]]),
                np.zeros((1, 3, 1)),
            )


if __name__ == "__main__":
    unittest.main()
