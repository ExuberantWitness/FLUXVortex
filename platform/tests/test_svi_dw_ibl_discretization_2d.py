import sys
import unittest
from pathlib import Path

import numpy as np

PLATFORM = Path(__file__).resolve().parents[1]
if str(PLATFORM) not in sys.path:
    sys.path.insert(0, str(PLATFORM))

from claim_runtime.svi_dw_ibl_discretization_2d import (  # noqa: E402
    IBLIntervalClosure2D,
    IBLIntervalEndpoint2D,
    IBLIntervalTemporalDerivatives2D,
    bdf2_time_derivative,
    evaluate_riziotis_interval_residuals,
)
from claim_runtime.svi_dw_types import SVIDWValidationError  # noqa: E402


def endpoint(**overrides):
    values = {
        "arc_length_from_stagnation": 0.2,
        "edge_density": 1.1,
        "edge_tangential_velocity": 7.0,
        "displacement_thickness": 0.004,
        "momentum_thickness": 0.0015,
        "kinetic_energy_shape_factor": 1.8,
        "density_shape_factor": 0.15,
    }
    values.update(overrides)
    return IBLIntervalEndpoint2D(**values)


def temporal(**overrides):
    values = {
        "dt_rho_edge_tangential_velocity_displacement": 0.0,
        "dt_rho_edge_tangential_velocity_squared_momentum": 0.0,
        "dt_rho_displacement_thickness": 0.0,
        "dt_edge_tangential_velocity": 0.0,
    }
    values.update(overrides)
    return IBLIntervalTemporalDerivatives2D(**values)


class SVIDWIBLDiscretization2DTests(unittest.TestCase):
    def test_bdf2_is_exact_for_quadratic_at_current_time(self):
        dt = 0.03
        current_time = 0.6

        def polynomial(time):
            return 2.3 * time**2 - 0.7 * time + 1.1

        result = bdf2_time_derivative(
            polynomial(current_time),
            polynomial(current_time - dt),
            polynomial(current_time - 2.0 * dt),
            time_step=dt,
        )
        expected = 2.0 * 2.3 * current_time - 0.7
        self.assertAlmostEqual(result, expected, places=13)

    def test_bdf2_preserves_array_shape_and_is_read_only(self):
        current = np.array((1.0, 4.0))
        previous = np.array((0.5, 3.0))
        older = np.array((0.25, 2.0))
        result = bdf2_time_derivative(current, previous, older, time_step=0.5)
        np.testing.assert_allclose(
            result,
            (3.0 * current - 4.0 * previous + older),
            rtol=0.0,
            atol=0.0,
        )
        self.assertFalse(result.flags.writeable)

    def test_uniform_interval_without_sources_is_zero(self):
        upstream = endpoint()
        downstream = endpoint(arc_length_from_stagnation=0.4)
        result = evaluate_riziotis_interval_residuals(
            upstream,
            downstream,
            temporal(),
            IBLIntervalClosure2D(
                skin_friction_coefficient=0.0,
                dissipation_coefficient=0.0,
            ),
        )
        self.assertEqual(result.momentum_residual, 0.0)
        self.assertEqual(result.energy_residual, 0.0)

    def test_every_eq7111_7112_term_matches_independent_arithmetic(self):
        upstream = endpoint()
        downstream = endpoint(
            arc_length_from_stagnation=0.31,
            edge_density=1.14,
            edge_tangential_velocity=6.6,
            displacement_thickness=0.0045,
            momentum_thickness=0.0017,
            kinetic_energy_shape_factor=1.86,
            density_shape_factor=0.17,
        )
        time_terms = temporal(
            dt_rho_edge_tangential_velocity_displacement=0.021,
            dt_rho_edge_tangential_velocity_squared_momentum=-0.08,
            dt_rho_displacement_thickness=0.003,
            dt_edge_tangential_velocity=-0.25,
        )
        closure = IBLIntervalClosure2D(
            skin_friction_coefficient=-0.004,
            dissipation_coefficient=0.0032,
            angular_velocity=1.3,
            local_flow_acceleration=-0.8,
        )
        result = evaluate_riziotis_interval_residuals(
            upstream, downstream, time_terms, closure
        )

        arc = 0.5 * (0.2 + 0.31)
        rho = 0.5 * (1.1 + 1.14)
        velocity = 0.5 * (7.0 + 6.6)
        delta = 0.5 * (0.004 + 0.0045)
        theta = 0.5 * (0.0015 + 0.0017)
        shape = 0.5 * (0.004 / 0.0015 + 0.0045 / 0.0017)
        shape_star = 0.5 * (1.8 + 1.86)
        shape_starstar = 0.5 * (0.15 + 0.17)
        dlns = np.log(0.31) - np.log(0.2)
        dlntheta = np.log(0.0017) - np.log(0.0015)
        dlnu = np.log(6.6) - np.log(7.0)
        dlnrho = np.log(1.14) - np.log(1.1)
        dlnhstar = np.log(1.86) - np.log(1.8)
        dlndelta = np.log(0.0045) - np.log(0.004)

        momentum_terms = (
            arc / (rho * velocity**2 * theta) * dlns * 0.021,
            dlntheta,
            (2.0 + shape) * dlnu,
            dlnrho,
            -(-0.004) * arc / (2.0 * theta) * dlns,
        )
        energy_terms = (
            arc / (rho * velocity**3 * shape_star * theta) * dlns * -0.08,
            arc / (rho * velocity * shape_star * theta) * dlns * 0.003,
            2.0 * shape_starstar * arc / (velocity**2 * shape_star) * dlns * -0.25,
            -arc / (rho * velocity**2 * theta) * dlns * 0.021,
            dlnhstar,
            (2.0 * shape_starstar / shape_star + 1.0 - shape) * dlnu,
            -4.0 * 1.3 / velocity * (theta + delta) * shape / shape_star * dlndelta,
            -0.004 * arc / (2.0 * theta) * dlns,
            -2.0 * 0.0032 * arc / (theta * shape_star) * dlns,
            -2.0 * -0.8 * shape * arc / (velocity**2 * shape_star) * dlns,
        )
        self.assertAlmostEqual(result.momentum_residual, sum(momentum_terms), places=14)
        self.assertAlmostEqual(result.energy_residual, sum(energy_terms), places=14)
        self.assertAlmostEqual(result.energy_rotation_east, energy_terms[6], places=15)

    def test_rotation_retains_signed_east_gradient(self):
        base = endpoint()
        increasing = evaluate_riziotis_interval_residuals(
            base,
            endpoint(
                arc_length_from_stagnation=0.3,
                displacement_thickness=0.0048,
            ),
            temporal(),
            IBLIntervalClosure2D(
                skin_friction_coefficient=0.0,
                dissipation_coefficient=0.0,
                angular_velocity=2.0,
            ),
        )
        decreasing = evaluate_riziotis_interval_residuals(
            endpoint(displacement_thickness=0.0048),
            endpoint(
                arc_length_from_stagnation=0.3,
                displacement_thickness=0.004,
            ),
            temporal(),
            IBLIntervalClosure2D(
                skin_friction_coefficient=0.0,
                dissipation_coefficient=0.0,
                angular_velocity=2.0,
            ),
        )
        self.assertLess(increasing.energy_rotation_east, 0.0)
        self.assertGreater(decreasing.energy_rotation_east, 0.0)

    def test_fail_closed_outside_log_domain_or_order(self):
        with self.assertRaises(SVIDWValidationError):
            endpoint(arc_length_from_stagnation=0.0)
        with self.assertRaises(SVIDWValidationError):
            endpoint(kinetic_energy_shape_factor=0.0)
        with self.assertRaisesRegex(SVIDWValidationError, "must increase downstream"):
            evaluate_riziotis_interval_residuals(
                endpoint(arc_length_from_stagnation=0.3),
                endpoint(arc_length_from_stagnation=0.2),
                temporal(),
                IBLIntervalClosure2D(
                    skin_friction_coefficient=0.0,
                    dissipation_coefficient=0.0,
                ),
            )
        with self.assertRaisesRegex(SVIDWValidationError, "identical shapes"):
            bdf2_time_derivative(np.zeros(2), np.zeros(3), np.zeros(2), time_step=0.1)


if __name__ == "__main__":
    unittest.main()
