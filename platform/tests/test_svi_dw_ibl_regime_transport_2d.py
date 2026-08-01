import math
import sys
import unittest
from pathlib import Path

PLATFORM = Path(__file__).resolve().parents[1]
if str(PLATFORM) not in sys.path:
    sys.path.insert(0, str(PLATFORM))

from claim_runtime.svi_dw_ibl_regime_transport_2d import (  # noqa: E402
    TransitionInterpolationEndpoint2D,
    TurbulentShearEndpoint2D,
    arithmetic_midpoint,
    evaluate_laminar_en_interval_residual,
    evaluate_turbulent_shear_interval_residual,
    initialize_transition_sqrt_shear_coefficient,
    interpolate_n9_transition_state,
    n9_transition_event,
    n9_transition_fraction,
)
from claim_runtime.svi_dw_types import SVIDWValidationError  # noqa: E402


def turbulent_endpoint(**overrides):
    values = {
        "arc_length_from_stagnation": 0.20,
        "edge_tangential_velocity": 8.0,
        "boundary_layer_thickness": 0.020,
        "displacement_thickness": 0.004,
        "kinematic_shape_factor": 2.0,
        "skin_friction_coefficient": 0.004,
        "sqrt_shear_coefficient": 0.05,
        "equilibrium_sqrt_shear_coefficient": 0.06,
    }
    values.update(overrides)
    return TurbulentShearEndpoint2D(**values)


class SVIDWIBLRegimeTransport2DTests(unittest.TestCase):
    def test_eq7120_matches_independent_hand_calculation(self):
        result = evaluate_laminar_en_interval_residual(
            arc_length_upstream=0.20,
            arc_length_downstream=0.32,
            amplification_upstream=1.1,
            amplification_downstream=1.7,
            growth_per_momentum_reynolds=0.012,
            velocity_gradient_parameter=-0.25,
            reynolds_length_factor=180.0,
            momentum_thickness=0.003,
        )
        expected_gradient = (1.7 - 1.1) / (0.32 - 0.20)
        expected_source = 0.012 * 0.5 * (-0.25 + 1.0) * 180.0 / 0.003
        self.assertAlmostEqual(
            result.amplification_gradient,
            expected_gradient,
            places=14,
        )
        self.assertAlmostEqual(
            result.amplification_source,
            expected_source,
            places=14,
        )
        self.assertAlmostEqual(
            result.residual,
            expected_gradient - expected_source,
            places=14,
        )

    def test_eq7120_keeps_signed_velocity_gradient_parameter(self):
        common = {
            "arc_length_upstream": 0.1,
            "arc_length_downstream": 0.2,
            "amplification_upstream": 2.0,
            "amplification_downstream": 2.0,
            "growth_per_momentum_reynolds": 0.01,
            "reynolds_length_factor": 20.0,
            "momentum_thickness": 0.002,
        }
        accelerating = evaluate_laminar_en_interval_residual(
            **common,
            velocity_gradient_parameter=0.5,
        )
        adverse = evaluate_laminar_en_interval_residual(
            **common,
            velocity_gradient_parameter=-1.5,
        )
        self.assertLess(accelerating.amplification_source, 100.0)
        self.assertGreater(accelerating.amplification_source, 0.0)
        self.assertLess(adverse.amplification_source, 0.0)
        self.assertLess(accelerating.residual, 0.0)
        self.assertGreater(adverse.residual, 0.0)

    def test_eq7120_constant_solution_is_zero_when_source_is_zero(self):
        result = evaluate_laminar_en_interval_residual(
            arc_length_upstream=0.1,
            arc_length_downstream=0.3,
            amplification_upstream=4.2,
            amplification_downstream=4.2,
            growth_per_momentum_reynolds=0.01,
            velocity_gradient_parameter=-1.0,
            reynolds_length_factor=100.0,
            momentum_thickness=0.002,
        )
        self.assertEqual(result.amplification_gradient, 0.0)
        self.assertEqual(result.amplification_source, 0.0)
        self.assertEqual(result.residual, 0.0)

    def test_eq7121_matches_independent_hand_calculation(self):
        upstream = turbulent_endpoint()
        downstream = turbulent_endpoint(
            arc_length_from_stagnation=0.35,
            edge_tangential_velocity=7.2,
            boundary_layer_thickness=0.024,
            displacement_thickness=0.005,
            kinematic_shape_factor=2.4,
            skin_friction_coefficient=-0.002,
            sqrt_shear_coefficient=0.08,
            equilibrium_sqrt_shear_coefficient=0.09,
        )
        result = evaluate_turbulent_shear_interval_residual(
            upstream,
            downstream,
        )

        ds = 0.35 - 0.20
        delta = 0.5 * (0.020 + 0.024)
        delta_star = 0.5 * (0.004 + 0.005)
        sqrt_shear = 0.5 * (0.05 + 0.08)
        equilibrium_sqrt_shear = 0.5 * (0.06 + 0.09)
        h_k = 0.5 * (2.0 + 2.4)
        skin_friction = 0.5 * (0.004 - 0.002)
        dlog_sqrt_shear = math.log(0.08) - math.log(0.05)
        dlog_velocity = math.log(7.2) - math.log(8.0)
        lhs = 2.0 * delta * dlog_sqrt_shear
        equilibrium = 5.6 * (
            equilibrium_sqrt_shear - sqrt_shear
        ) * ds
        wall = (
            2.0
            * delta
            * 4.0
            / (3.0 * delta_star)
            * (
                skin_friction / 2.0
                - ((h_k - 1.0) / (6.7 * h_k)) ** 2
            )
            * ds
        )
        velocity = -2.0 * delta * dlog_velocity
        expected = lhs - equilibrium - wall - velocity

        self.assertEqual(result.sqrt_shear_midpoint, sqrt_shear)
        self.assertEqual(result.kinematic_shape_midpoint, h_k)
        self.assertAlmostEqual(
            result.log_sqrt_shear_increment,
            dlog_sqrt_shear,
            places=15,
        )
        self.assertAlmostEqual(
            result.log_edge_velocity_increment,
            dlog_velocity,
            places=15,
        )
        self.assertAlmostEqual(result.log_shear_transport, lhs, places=15)
        self.assertAlmostEqual(
            result.equilibrium_relaxation,
            equilibrium,
            places=15,
        )
        self.assertAlmostEqual(result.wall_shear_drive, wall, places=15)
        self.assertAlmostEqual(
            result.signed_edge_velocity_transport,
            velocity,
            places=15,
        )
        self.assertAlmostEqual(result.residual, expected, places=15)

    def test_eq7121_constant_solution_is_zero(self):
        h_k = 2.1
        balanced_skin_friction = 2.0 * (
            (h_k - 1.0) / (6.7 * h_k)
        ) ** 2
        upstream = turbulent_endpoint(
            kinematic_shape_factor=h_k,
            skin_friction_coefficient=balanced_skin_friction,
            sqrt_shear_coefficient=0.07,
            equilibrium_sqrt_shear_coefficient=0.07,
        )
        downstream = turbulent_endpoint(
            arc_length_from_stagnation=0.4,
            kinematic_shape_factor=h_k,
            skin_friction_coefficient=balanced_skin_friction,
            sqrt_shear_coefficient=0.07,
            equilibrium_sqrt_shear_coefficient=0.07,
        )
        result = evaluate_turbulent_shear_interval_residual(
            upstream,
            downstream,
        )
        self.assertEqual(result.log_shear_transport, 0.0)
        self.assertEqual(result.equilibrium_relaxation, 0.0)
        self.assertEqual(result.wall_shear_drive, 0.0)
        self.assertEqual(result.signed_edge_velocity_transport, 0.0)
        self.assertEqual(result.residual, 0.0)

    def test_eq7121_uses_log_of_sqrt_ctau_not_log_of_ctau(self):
        h_k = 2.0
        balanced_skin_friction = 2.0 * (
            (h_k - 1.0) / (6.7 * h_k)
        ) ** 2
        upstream = turbulent_endpoint(
            edge_tangential_velocity=5.0,
            kinematic_shape_factor=h_k,
            skin_friction_coefficient=balanced_skin_friction,
            sqrt_shear_coefficient=0.02,
            equilibrium_sqrt_shear_coefficient=0.03,
        )
        downstream = turbulent_endpoint(
            arc_length_from_stagnation=0.3,
            edge_tangential_velocity=5.0,
            kinematic_shape_factor=h_k,
            skin_friction_coefficient=balanced_skin_friction,
            sqrt_shear_coefficient=0.04,
            equilibrium_sqrt_shear_coefficient=0.03,
        )
        result = evaluate_turbulent_shear_interval_residual(
            upstream,
            downstream,
        )
        self.assertAlmostEqual(
            result.log_sqrt_shear_increment,
            math.log(2.0),
            places=15,
        )
        self.assertAlmostEqual(
            result.log_shear_transport,
            2.0 * 0.020 * math.log(2.0),
            places=15,
        )

    def test_eq7121_keeps_signed_edge_velocity_increment(self):
        h_k = 2.0
        balanced_skin_friction = 2.0 * (
            (h_k - 1.0) / (6.7 * h_k)
        ) ** 2

        def result(edge_velocity):
            return evaluate_turbulent_shear_interval_residual(
                turbulent_endpoint(
                    edge_tangential_velocity=8.0,
                    kinematic_shape_factor=h_k,
                    skin_friction_coefficient=balanced_skin_friction,
                    sqrt_shear_coefficient=0.06,
                    equilibrium_sqrt_shear_coefficient=0.06,
                ),
                turbulent_endpoint(
                    arc_length_from_stagnation=0.3,
                    edge_tangential_velocity=edge_velocity,
                    kinematic_shape_factor=h_k,
                    skin_friction_coefficient=balanced_skin_friction,
                    sqrt_shear_coefficient=0.06,
                    equilibrium_sqrt_shear_coefficient=0.06,
                ),
            )

        accelerating = result(10.0)
        decelerating = result(6.0)
        self.assertLess(accelerating.signed_edge_velocity_transport, 0.0)
        self.assertGreater(decelerating.signed_edge_velocity_transport, 0.0)
        self.assertGreater(accelerating.residual, 0.0)
        self.assertLess(decelerating.residual, 0.0)

    def test_arithmetic_midpoint_is_not_geometric_midpoint(self):
        self.assertEqual(arithmetic_midpoint(1.0, 9.0), 5.0)
        self.assertNotEqual(arithmetic_midpoint(1.0, 9.0), 3.0)

    def test_n9_event_fraction_and_linear_state_interpolation(self):
        self.assertEqual(n9_transition_event(8.25), -0.75)
        self.assertEqual(n9_transition_event(9.0), 0.0)
        self.assertEqual(n9_transition_event(10.5), 1.5)
        self.assertEqual(n9_transition_fraction(8.0, 12.0), 0.25)

        transition = interpolate_n9_transition_state(
            TransitionInterpolationEndpoint2D(
                arc_length_from_stagnation=0.2,
                amplification=8.0,
                displacement_thickness=0.004,
                momentum_thickness=0.002,
            ),
            TransitionInterpolationEndpoint2D(
                arc_length_from_stagnation=0.6,
                amplification=12.0,
                displacement_thickness=0.008,
                momentum_thickness=0.004,
            ),
        )
        self.assertEqual(transition.interval_fraction, 0.25)
        self.assertEqual(transition.arc_length_from_stagnation, 0.3)
        self.assertEqual(transition.amplification, 9.0)
        self.assertEqual(transition.displacement_thickness, 0.005)
        self.assertEqual(transition.momentum_thickness, 0.0025)

    def test_transition_shear_initialization_acts_on_sqrt_state(self):
        initialized = initialize_transition_sqrt_shear_coefficient(0.2)
        self.assertAlmostEqual(initialized, 0.7 * 0.2, places=15)
        self.assertNotAlmostEqual(initialized, 0.49 * 0.2, places=15)

    def test_fail_closed_on_invalid_eq7120_inputs(self):
        common = {
            "arc_length_upstream": 0.1,
            "arc_length_downstream": 0.2,
            "amplification_upstream": 1.0,
            "amplification_downstream": 2.0,
            "growth_per_momentum_reynolds": 0.01,
            "velocity_gradient_parameter": 0.0,
            "reynolds_length_factor": 100.0,
            "momentum_thickness": 0.002,
        }
        invalid_overrides = (
            {"arc_length_downstream": 0.1},
            {"growth_per_momentum_reynolds": 0.0},
            {"reynolds_length_factor": -1.0},
            {"momentum_thickness": 0.0},
            {"velocity_gradient_parameter": math.nan},
        )
        for override in invalid_overrides:
            with self.subTest(override=override):
                with self.assertRaises(SVIDWValidationError):
                    evaluate_laminar_en_interval_residual(
                        **(common | override)
                    )

    def test_fail_closed_on_invalid_eq7121_or_transition_inputs(self):
        invalid_endpoint_overrides = (
            {"edge_tangential_velocity": 0.0},
            {"boundary_layer_thickness": 0.003},
            {"displacement_thickness": 0.0},
            {"kinematic_shape_factor": 1.0},
            {"sqrt_shear_coefficient": 0.0},
            {"equilibrium_sqrt_shear_coefficient": -0.01},
            {"skin_friction_coefficient": math.inf},
        )
        for override in invalid_endpoint_overrides:
            with self.subTest(override=override):
                with self.assertRaises(SVIDWValidationError):
                    turbulent_endpoint(**override)

        with self.assertRaisesRegex(
            SVIDWValidationError,
            "must increase downstream",
        ):
            evaluate_turbulent_shear_interval_residual(
                turbulent_endpoint(arc_length_from_stagnation=0.3),
                turbulent_endpoint(arc_length_from_stagnation=0.2),
            )
        for amplification_pair in ((9.1, 10.0), (7.0, 8.0), (9.0, 9.0)):
            with self.subTest(amplification_pair=amplification_pair):
                with self.assertRaises(SVIDWValidationError):
                    n9_transition_fraction(*amplification_pair)
        with self.assertRaises(SVIDWValidationError):
            initialize_transition_sqrt_shear_coefficient(0.0)


if __name__ == "__main__":
    unittest.main()
