import inspect
import math
import sys
import unittest
from pathlib import Path

PLATFORM = Path(__file__).resolve().parents[1]
if str(PLATFORM) not in sys.path:
    sys.path.insert(0, str(PLATFORM))

from claim_runtime.svi_dw_ibl_closures_2d import (  # noqa: E402
    RIZIOTIS_CTAU_TRANSITION_SQRT_RATIO,
    RIZIOTIS_N_CRIT,
    body_tangential_acceleration,
    density_shape_factor,
    east_normal_momentum_thickness,
    en_growth_per_momentum_reynolds,
    en_onset_momentum_reynolds,
    en_similar_flow_amplification,
    en_spatial_growth,
    kinematic_shape_factor,
    laminar_dissipation_coefficient,
    laminar_dissipation_ratio,
    laminar_energy_shape_factor,
    laminar_skin_friction_coefficient,
    transition_amplification_threshold,
    transition_shear_coefficient,
    turbulent_dissipation_coefficient,
    turbulent_energy_shape_factor,
    turbulent_equilibrium_shear_coefficient,
    turbulent_h0,
    turbulent_nominal_thickness,
    turbulent_skin_friction_coefficient,
    turbulent_slip_velocity_ratio,
)
from claim_runtime.svi_dw_types import SVIDWValidationError  # noqa: E402


class ShapeFactorClosureTests(unittest.TestCase):
    def test_shape_transforms_match_independent_values(self):
        # (2.2 - 0.290*0.12**2) / (1 + 0.113*0.12**2)
        self.assertAlmostEqual(
            kinematic_shape_factor(2.2, 0.12),
            2.1922567598004528,
            places=15,
        )
        # (0.064/(2.0 - 0.8) + 0.251)*0.12**2
        self.assertAlmostEqual(
            density_shape_factor(2.0, 0.12),
            0.0043824,
            places=15,
        )

    def test_incompressible_source_limit_is_exact(self):
        self.assertEqual(kinematic_shape_factor(2.2, 0.0), 2.2)
        self.assertEqual(density_shape_factor(2.2, 0.0), 0.0)

    def test_shape_transform_domains_fail_closed(self):
        with self.assertRaises(SVIDWValidationError):
            kinematic_shape_factor(0.1, 4.0)
        with self.assertRaises(SVIDWValidationError):
            kinematic_shape_factor(2.0, -0.1)
        with self.assertRaises(SVIDWValidationError):
            density_shape_factor(0.8, 0.0)
        with self.assertRaises(SVIDWValidationError):
            density_shape_factor(0.7, 0.2)


class LaminarClosureTests(unittest.TestCase):
    def test_laminar_closure_matches_independent_hand_values(self):
        self.assertAlmostEqual(
            laminar_energy_shape_factor(2.5),
            1.5834,
            places=15,
        )
        self.assertAlmostEqual(
            laminar_skin_friction_coefficient(2.5, 850.0),
            0.0005869454117647061,
            places=15,
        )
        self.assertAlmostEqual(
            laminar_dissipation_ratio(2.5, 850.0),
            0.0002659598035913132,
            places=15,
        )
        self.assertAlmostEqual(
            laminar_dissipation_coefficient(2.5, 850.0),
            0.00021056037650324264,
            places=15,
        )

    def test_laminar_piecewise_boundaries_are_continuous(self):
        epsilon = 1.0e-9
        self.assertEqual(laminar_energy_shape_factor(4.0), 1.515)
        self.assertAlmostEqual(
            laminar_energy_shape_factor(4.0 - epsilon),
            laminar_energy_shape_factor(4.0 + epsilon),
            places=9,
        )
        self.assertEqual(
            laminar_dissipation_ratio(4.0, 1000.0),
            0.000207,
        )
        self.assertAlmostEqual(
            laminar_dissipation_ratio(4.0 - epsilon, 1000.0),
            laminar_dissipation_ratio(4.0 + epsilon, 1000.0),
            places=12,
        )
        self.assertEqual(
            laminar_skin_friction_coefficient(7.4, 1000.0),
            -0.000134,
        )
        self.assertAlmostEqual(
            laminar_skin_friction_coefficient(
                7.4 - epsilon, 1000.0
            ),
            laminar_skin_friction_coefficient(
                7.4 + epsilon, 1000.0
            ),
            places=12,
        )

    def test_thesis_separated_cd_coefficient_is_not_xfoil_replacement(self):
        thesis_value = laminar_dissipation_ratio(5.0, 1000.0)
        self.assertAlmostEqual(
            thesis_value,
            0.00020405882352941175,
            places=15,
        )
        xfoil_newer_value = (
            0.207 - 0.0016 / 1.02
        ) / 1000.0
        self.assertNotAlmostEqual(
            thesis_value,
            xfoil_newer_value,
            places=10,
        )

    def test_laminar_friction_preserves_separation_sign(self):
        self.assertLess(
            laminar_skin_friction_coefficient(7.4, 1000.0),
            0.0,
        )


class TurbulentClosureTests(unittest.TestCase):
    def test_turbulent_closure_matches_independent_hand_values(self):
        skin_friction = turbulent_skin_friction_coefficient(
            2.2, 1200.0, 0.12
        )
        energy_shape = turbulent_energy_shape_factor(2.2, 1200.0)
        slip = turbulent_slip_velocity_ratio(
            energy_shape, 2.2, 2.25
        )
        nominal_delta = turbulent_nominal_thickness(
            0.0012, 0.0027, 2.2
        )
        equilibrium_shear = (
            turbulent_equilibrium_shear_coefficient(
                energy_shape, slip, 2.2, 2.25
            )
        )
        dissipation = turbulent_dissipation_coefficient(
            skin_friction,
            slip,
            0.8 * equilibrium_shear,
        )

        self.assertAlmostEqual(
            skin_friction, 0.0010436185031307203, places=15
        )
        self.assertAlmostEqual(
            energy_shape, 1.574312952843335, places=15
        )
        self.assertAlmostEqual(
            slip, 0.22740075985514838, places=15
        )
        self.assertAlmostEqual(
            nominal_delta, 0.0082, places=15
        )
        self.assertAlmostEqual(
            equilibrium_shear, 0.004850024242945934, places=15
        )
        self.assertAlmostEqual(
            dissipation, 0.0031163598561327194, places=15
        )

    def test_h0_erratum_is_unique_and_continuous(self):
        self.assertEqual(turbulent_h0(399.0), 4.0)
        self.assertEqual(turbulent_h0(400.0), 4.0)
        self.assertAlmostEqual(
            turbulent_h0(400.0 + 1.0e-9),
            4.0,
            places=11,
        )
        self.assertEqual(turbulent_h0(1000.0), 3.4)
        self.assertNotEqual(turbulent_h0(1000.0), 3.0 + 4.0 / 1000.0)

    def test_old_thesis_energy_shape_has_no_xfoil_reynolds_clamp(self):
        # Direct old Eq. (7.62) at Re_theta=100.  A newer-XFOIL RTZ floor
        # would silently evaluate a different Reynolds number.
        self.assertAlmostEqual(
            turbulent_energy_shape_factor(2.0, 100.0),
            1.552578582832552,
            places=15,
        )
        self.assertNotAlmostEqual(
            turbulent_energy_shape_factor(2.0, 100.0),
            turbulent_energy_shape_factor(2.0, 200.0),
            places=10,
        )

    def test_energy_shape_branches_meet_at_h0(self):
        reynolds = 1000.0
        h_0 = turbulent_h0(reynolds)
        expected = 1.505 + 4.0 / reynolds
        self.assertEqual(
            turbulent_energy_shape_factor(h_0, reynolds),
            expected,
        )
        epsilon = 1.0e-8
        self.assertAlmostEqual(
            turbulent_energy_shape_factor(
                h_0 - epsilon, reynolds
            ),
            turbulent_energy_shape_factor(
                h_0 + epsilon, reynolds
            ),
            places=8,
        )

    def test_turbulent_singular_domains_fail_closed_without_clamps(self):
        with self.assertRaises(SVIDWValidationError):
            turbulent_skin_friction_coefficient(2.0, 1.0, 0.0)
        with self.assertRaises(SVIDWValidationError):
            turbulent_skin_friction_coefficient(2.0, 0.5, 0.0)
        with self.assertRaises(SVIDWValidationError):
            turbulent_energy_shape_factor(5.0, 1.0)
        # H_k == H0 belongs to the separated Eq. (7.62) branch, whose
        # logarithmic term is undefined at Re_theta == 1.
        with self.assertRaises(SVIDWValidationError):
            turbulent_energy_shape_factor(4.0, 1.0)
        with self.assertRaises(SVIDWValidationError):
            turbulent_energy_shape_factor(5.0, math.exp(-4.0))
        with self.assertRaises(SVIDWValidationError):
            turbulent_equilibrium_shear_coefficient(
                1.6, 1.0, 2.0, 2.0
            )
        with self.assertRaises(SVIDWValidationError):
            turbulent_equilibrium_shear_coefficient(
                1.6, 1.1, 2.0, 2.0
            )

    def test_negative_dissipation_is_rejected_not_clipped(self):
        with self.assertRaises(SVIDWValidationError):
            turbulent_dissipation_coefficient(0.02, -1.0, 0.0)


class MovingFrameAndTransitionClosureTests(unittest.TestCase):
    def test_east_theta_n_preserves_gradient_sign(self):
        positive = east_normal_momentum_thickness(
            0.0012, 0.0027, 0.08
        )
        negative = east_normal_momentum_thickness(
            0.0012, 0.0027, -0.08
        )
        self.assertEqual(positive, 0.000312)
        self.assertEqual(negative, -0.000312)
        self.assertEqual(positive, -negative)

    def test_body_acceleration_keeps_all_coordinate_signs(self):
        # 2**2*0.4 + (-3)*(-0.2) - 1.1 = 1.1
        self.assertAlmostEqual(
            body_tangential_acceleration(
                angular_velocity=2.0,
                angular_acceleration=-3.0,
                point_from_origin_tangent=0.4,
                point_from_origin_normal=-0.2,
                origin_tangential_acceleration=1.1,
            ),
            1.1,
            places=15,
        )
        self.assertAlmostEqual(
            body_tangential_acceleration(
                angular_velocity=-2.0,
                angular_acceleration=3.0,
                point_from_origin_tangent=0.4,
                point_from_origin_normal=-0.2,
                origin_tangential_acceleration=1.1,
            ),
            -0.1,
            places=15,
        )

    def test_en_relations_match_independent_hand_values(self):
        self.assertAlmostEqual(
            en_growth_per_momentum_reynolds(2.2),
            0.007849752948995677,
            places=15,
        )
        self.assertAlmostEqual(
            en_onset_momentum_reynolds(2.2),
            7503.467098683781,
            places=11,
        )
        self.assertAlmostEqual(
            en_similar_flow_amplification(2.2, 1200.0),
            -49.480659446790234,
            places=12,
        )
        self.assertAlmostEqual(
            en_spatial_growth(2.2, -35.0),
            -0.2747413532148487,
            places=15,
        )

    def test_transition_constants_are_source_fixed_not_call_parameters(self):
        self.assertEqual(RIZIOTIS_N_CRIT, 9.0)
        self.assertEqual(transition_amplification_threshold(), 9.0)
        self.assertEqual(
            RIZIOTIS_CTAU_TRANSITION_SQRT_RATIO,
            0.7,
        )
        transitioned = transition_shear_coefficient(0.02)
        self.assertAlmostEqual(transitioned, 0.0098, places=15)
        self.assertAlmostEqual(
            math.sqrt(transitioned / 0.02),
            0.7,
            places=15,
        )
        self.assertEqual(
            len(inspect.signature(
                transition_amplification_threshold
            ).parameters),
            0,
        )
        self.assertEqual(
            tuple(inspect.signature(
                transition_shear_coefficient
            ).parameters),
            ("equilibrium_shear_coefficient",),
        )


class ClosureValidationTests(unittest.TestCase):
    def test_nonfinite_inputs_fail_closed(self):
        calls = (
            lambda: kinematic_shape_factor(math.nan, 0.0),
            lambda: density_shape_factor(2.0, math.inf),
            lambda: laminar_energy_shape_factor(math.nan),
            lambda: laminar_skin_friction_coefficient(2.0, math.inf),
            lambda: laminar_dissipation_ratio(math.inf, 1000.0),
            lambda: turbulent_skin_friction_coefficient(
                2.0, 1000.0, math.nan
            ),
            lambda: turbulent_energy_shape_factor(math.inf, 1000.0),
            lambda: turbulent_slip_velocity_ratio(
                1.5, 2.0, math.nan
            ),
            lambda: turbulent_nominal_thickness(
                0.001, math.inf, 2.0
            ),
            lambda: turbulent_equilibrium_shear_coefficient(
                1.5, math.nan, 2.0, 2.0
            ),
            lambda: turbulent_dissipation_coefficient(
                0.01, 0.2, math.inf
            ),
            lambda: east_normal_momentum_thickness(
                0.001, 0.002, math.nan
            ),
            lambda: body_tangential_acceleration(
                math.inf, 0.0, 0.0, 0.0, 0.0
            ),
            lambda: en_growth_per_momentum_reynolds(math.nan),
            lambda: en_onset_momentum_reynolds(math.inf),
            lambda: en_similar_flow_amplification(2.0, math.nan),
            lambda: en_spatial_growth(2.0, math.inf),
            lambda: transition_shear_coefficient(math.nan),
        )
        for call in calls:
            with self.subTest(call=call):
                with self.assertRaises(SVIDWValidationError):
                    call()

    def test_physical_denominators_and_positive_inventories_fail_closed(self):
        calls = (
            lambda: laminar_energy_shape_factor(1.0),
            lambda: laminar_skin_friction_coefficient(2.0, 0.0),
            lambda: turbulent_h0(0.0),
            lambda: turbulent_slip_velocity_ratio(0.0, 2.0, 2.0),
            lambda: turbulent_nominal_thickness(0.0, 0.002, 2.0),
            lambda: east_normal_momentum_thickness(
                -0.001, 0.002, 0.1
            ),
            lambda: en_growth_per_momentum_reynolds(1.0),
            lambda: en_onset_momentum_reynolds(1.0),
            lambda: transition_shear_coefficient(-0.01),
        )
        for call in calls:
            with self.subTest(call=call):
                with self.assertRaises(SVIDWValidationError):
                    call()


if __name__ == "__main__":
    unittest.main()
