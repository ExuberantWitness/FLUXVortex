import sys
import unittest
from pathlib import Path

import numpy as np

PLATFORM = Path(__file__).resolve().parents[1]
if str(PLATFORM) not in sys.path:
    sys.path.insert(0, str(PLATFORM))

from claim_runtime.dual_side_bernoulli import (  # noqa: E402
    DualSideBernoulliError,
    dual_side_moving_bernoulli,
    paired_thin_sheet_source_report,
)


class DualSideBernoulliTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(20260727)
        self.count = 19
        self.rho = 1.225
        self.mean_rate = rng.normal(size=self.count)
        self.jump_rate = rng.normal(size=self.count)
        self.mean_velocity = rng.normal(size=(self.count, 3))
        self.gradient = rng.normal(size=(self.count, 3))
        self.wall_velocity = rng.normal(size=(self.count, 3))

    def evaluate(self, gauge=0.0):
        return dual_side_moving_bernoulli(
            density=self.rho,
            mean_potential_wall_rate=self.mean_rate,
            potential_jump_wall_rate=self.jump_rate,
            mean_velocity=self.mean_velocity,
            potential_jump_surface_gradient=self.gradient,
            wall_velocity=self.wall_velocity,
            bernoulli_gauge=gauge,
        )

    def test_dual_side_difference_is_unified_pressure_panel_by_panel(self):
        result = self.evaluate()
        expected = self.rho * (
            self.jump_rate
            + np.einsum(
                "ij,ij->i",
                self.mean_velocity - self.wall_velocity,
                self.gradient,
            )
        )
        np.testing.assert_allclose(
            result.pressure_jump,
            expected,
            rtol=0.0,
            atol=2.0e-15,
        )
        self.assertLessEqual(result.max_jump_identity_residual, 2.0e-15)

    def test_common_time_dependent_gauge_changes_sides_not_jump(self):
        gauge = np.linspace(-3.0, 4.0, self.count)
        reference = self.evaluate()
        shifted = self.evaluate(gauge)
        np.testing.assert_allclose(
            shifted.pressure_plus - reference.pressure_plus,
            self.rho * gauge,
            atol=2.0e-15,
        )
        np.testing.assert_allclose(
            shifted.pressure_minus - reference.pressure_minus,
            self.rho * gauge,
            atol=2.0e-15,
        )
        np.testing.assert_allclose(
            shifted.pressure_jump,
            reference.pressure_jump,
            atol=2.0e-15,
        )

    def test_comoving_uniform_jump_has_zero_pressure_jump(self):
        result = dual_side_moving_bernoulli(
            density=self.rho,
            mean_potential_wall_rate=np.linspace(-1.0, 1.0, self.count),
            potential_jump_wall_rate=np.zeros(self.count),
            mean_velocity=self.wall_velocity,
            potential_jump_surface_gradient=np.zeros((self.count, 3)),
            wall_velocity=self.wall_velocity,
        )
        np.testing.assert_allclose(result.pressure_jump, 0.0, atol=0.0)

    def test_paired_wall_source_cancels_acceleration_and_matches_jump(self):
        rng = np.random.default_rng(91)
        normal = rng.normal(size=(self.count, 3))
        normal /= np.linalg.norm(normal, axis=1)[:, None]
        acceleration = rng.normal(size=(self.count, 3))
        gradient_plus = rng.normal(size=(self.count, 3))
        gradient_minus = rng.normal(size=(self.count, 3))
        report = paired_thin_sheet_source_report(
            normal_plus=normal,
            wall_acceleration=acceleration,
            specific_pressure_gradient_plus=gradient_plus,
            specific_pressure_gradient_minus=gradient_minus,
            specific_pressure_jump_gradient=gradient_minus-gradient_plus,
            tolerance=2.0e-14,
        )
        self.assertTrue(report.passed)
        self.assertLessEqual(report.max_identity_residual, 2.0e-14)

    def test_bad_shapes_and_nonunit_normals_fail(self):
        with self.assertRaises(DualSideBernoulliError):
            dual_side_moving_bernoulli(
                density=self.rho,
                mean_potential_wall_rate=self.mean_rate,
                potential_jump_wall_rate=self.jump_rate,
                mean_velocity=self.mean_velocity,
                potential_jump_surface_gradient=self.gradient[:-1],
                wall_velocity=self.wall_velocity,
            )
        with self.assertRaises(DualSideBernoulliError):
            paired_thin_sheet_source_report(
                normal_plus=np.ones((self.count, 3)),
                wall_acceleration=np.zeros((self.count, 3)),
                specific_pressure_gradient_plus=np.zeros((self.count, 3)),
                specific_pressure_gradient_minus=np.zeros((self.count, 3)),
                specific_pressure_jump_gradient=np.zeros((self.count, 3)),
            )


if __name__ == "__main__":
    unittest.main()
