import sys
import unittest
from pathlib import Path

import numpy as np

PLATFORM = Path(__file__).resolve().parents[1]
if str(PLATFORM) not in sys.path:
    sys.path.insert(0, str(PLATFORM))

from claim_runtime.svi_dw_pressure_2d import (  # noqa: E402
    BernoulliPressureInput2D,
    evaluate_moving_body_bernoulli_pressure,
    first_order_backward_time_derivative,
    integrate_surface_traction_once,
)
from claim_runtime.svi_dw_types import (  # noqa: E402
    NACA4SectionConfig,
    SVIDWValidationError,
    build_naca4_actual_surface,
)


def surface():
    return build_naca4_actual_surface(
        NACA4SectionConfig(
            maximum_camber=0.0,
            camber_location=0.4,
            thickness_ratio=0.15,
            chord=1.0,
        ),
        panels_per_side=32,
    )


class SVIDWPressure2DTests(unittest.TestCase):
    def test_stationary_uniform_flow_has_zero_gauge_pressure(self):
        velocity = np.tile((8.0, 0.0), (5, 1))
        result = evaluate_moving_body_bernoulli_pressure(
            BernoulliPressureInput2D(
                density=1.2,
                freestream_velocity_inertial=(8.0, 0.0),
                relative_surface_velocity=velocity,
                body_surface_velocity=np.zeros((5, 2)),
                phi_minus_phi_infinity_current=np.zeros(5),
                phi_minus_phi_infinity_previous=np.zeros(5),
                time_step=0.02,
                inside_separated_bubble=np.zeros(5, dtype=bool),
                total_pressure_deficit=0.0,
            )
        )
        np.testing.assert_allclose(result.pressure_difference, 0.0, rtol=0.0, atol=0.0)

    def test_bubble_delta_h_is_not_an_extra_force_channel(self):
        count = 4
        common = dict(
            density=1.25,
            freestream_velocity_inertial=(7.0, 0.0),
            relative_surface_velocity=np.tile((6.0, 0.0), (count, 1)),
            body_surface_velocity=np.zeros((count, 2)),
            phi_minus_phi_infinity_current=np.zeros(count),
            phi_minus_phi_infinity_previous=np.zeros(count),
            time_step=0.01,
            total_pressure_deficit=3.2,
        )
        result = evaluate_moving_body_bernoulli_pressure(
            BernoulliPressureInput2D(
                **common,
                inside_separated_bubble=(False, True, False, True),
            )
        )
        outside = 1.25 * (0.5 * 7.0**2 - 0.5 * 6.0**2)
        expected = np.array(
            (outside, outside - 1.25 * 3.2, outside, outside - 1.25 * 3.2)
        )
        np.testing.assert_allclose(
            result.pressure_difference, expected, rtol=0.0, atol=1.0e-14
        )
        np.testing.assert_allclose(result.bubble_pressure_deficit, (0.0, 3.2, 0.0, 3.2))

    def test_phi_minus_phi_infinity_derivative_has_source_sign(self):
        result = evaluate_moving_body_bernoulli_pressure(
            BernoulliPressureInput2D(
                density=2.0,
                freestream_velocity_inertial=(0.0, 0.0),
                relative_surface_velocity=np.zeros((2, 2)),
                body_surface_velocity=np.zeros((2, 2)),
                phi_minus_phi_infinity_current=(0.3, -0.1),
                phi_minus_phi_infinity_previous=(0.1, -0.2),
                time_step=0.05,
                inside_separated_bubble=(False, False),
                total_pressure_deficit=0.0,
            )
        )
        np.testing.assert_allclose(result.potential_time_derivative, (4.0, 2.0))
        np.testing.assert_allclose(result.pressure_difference, (-8.0, -4.0))

    def test_constant_pressure_on_closed_surface_integrates_to_zero(self):
        actual_surface = surface()
        ledger = integrate_surface_traction_once(
            actual_surface,
            pressure_difference=np.full(actual_surface.panel_count, 37.0),
            signed_wall_shear=np.zeros(actual_surface.panel_count),
            reference_point_body=(0.25, 0.0),
        )
        np.testing.assert_allclose(ledger.total_force, 0.0, rtol=0.0, atol=2.0e-14)
        self.assertAlmostEqual(ledger.total_moment, 0.0, places=14)

    def test_panel_ledger_closes_force_and_moment_once(self):
        actual_surface = surface()
        count = actual_surface.panel_count
        pressure = np.linspace(-4.0, 9.0, count)
        shear = np.linspace(0.2, -0.1, count)
        reference = np.array((0.31, -0.02))
        ledger = integrate_surface_traction_once(
            actual_surface,
            pressure_difference=pressure,
            signed_wall_shear=shear,
            reference_point_body=reference,
        )
        np.testing.assert_allclose(
            ledger.total_force,
            np.sum(
                ledger.pressure_force_per_panel + ledger.shear_force_per_panel,
                axis=0,
            ),
            rtol=0.0,
            atol=1.0e-15,
        )
        arm = actual_surface.panel_midpoints - reference
        independent_moment = np.sum(
            arm[:, 0] * ledger.total_force_per_panel[:, 1]
            - arm[:, 1] * ledger.total_force_per_panel[:, 0]
        )
        self.assertAlmostEqual(ledger.total_moment, independent_moment, places=15)

    def test_invalid_pressure_inputs_fail_closed(self):
        with self.assertRaises(SVIDWValidationError):
            first_order_backward_time_derivative(
                np.zeros(2), np.zeros(3), time_step=0.1
            )
        with self.assertRaisesRegex(SVIDWValidationError, "must be finite"):
            BernoulliPressureInput2D(
                density=1.2,
                freestream_velocity_inertial=(8.0, 0.0),
                relative_surface_velocity=np.tile((8.0, 0.0), (2, 1)),
                body_surface_velocity=np.zeros((2, 2)),
                phi_minus_phi_infinity_current=np.zeros(2),
                phi_minus_phi_infinity_previous=np.zeros(2),
                time_step=0.02,
                inside_separated_bubble=(False, False),
                total_pressure_deficit=np.nan,
            )


if __name__ == "__main__":
    unittest.main()
