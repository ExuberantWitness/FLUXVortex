import sys
import unittest
from pathlib import Path

PLATFORM = Path(__file__).resolve().parents[1]
if str(PLATFORM) not in sys.path:
    sys.path.insert(0, str(PLATFORM))

from claim_runtime.svi_dw_ibl_equations_2d import (  # noqa: E402
    IBLClosureTerms2D,
    IBLStationDerivatives2D,
    IBLStationState2D,
    evaluate_riziotis_ibl_residuals,
)
from claim_runtime.svi_dw_types import SVIDWValidationError  # noqa: E402


def zero_derivatives(**overrides):
    values = {
        "dt_rho_edge_tangential_velocity_displacement": 0.0,
        "ds_momentum_thickness": 0.0,
        "ds_edge_tangential_velocity": 0.0,
        "ds_edge_density": 0.0,
        "dt_rho_edge_tangential_velocity_squared_momentum": 0.0,
        "dt_rho_displacement_thickness": 0.0,
        "dt_edge_tangential_velocity": 0.0,
        "ds_kinetic_energy_shape_factor": 0.0,
    }
    values.update(overrides)
    return IBLStationDerivatives2D(**values)


class SVIDWIBLEquationTests(unittest.TestCase):
    def test_quiescent_uniform_equations_have_zero_residual(self):
        result = evaluate_riziotis_ibl_residuals(
            IBLStationState2D(
                edge_density=1.2,
                edge_tangential_velocity=8.0,
                edge_speed=8.0,
                displacement_thickness=0.003,
                momentum_thickness=0.001,
                kinetic_energy_thickness=0.0017,
            ),
            zero_derivatives(),
            IBLClosureTerms2D(
                skin_friction_coefficient=0.0,
                dissipation_coefficient=0.0,
            ),
        )
        self.assertEqual(result.momentum_residual, 0.0)
        self.assertEqual(result.energy_residual, 0.0)

    def test_every_published_term_matches_independent_arithmetic(self):
        state = IBLStationState2D(
            edge_density=1.1,
            edge_tangential_velocity=7.0,
            edge_speed=7.0,
            displacement_thickness=0.004,
            momentum_thickness=0.0015,
            kinetic_energy_thickness=0.0027,
            density_thickness=0.0003,
            normal_momentum_transport=-0.0002,
        )
        derivatives = IBLStationDerivatives2D(
            dt_rho_edge_tangential_velocity_displacement=0.08,
            ds_momentum_thickness=0.004,
            ds_edge_tangential_velocity=-1.3,
            ds_edge_density=0.02,
            dt_rho_edge_tangential_velocity_squared_momentum=-0.12,
            dt_rho_displacement_thickness=0.006,
            dt_edge_tangential_velocity=0.7,
            ds_kinetic_energy_shape_factor=-0.11,
        )
        closure = IBLClosureTerms2D(
            skin_friction_coefficient=-0.006,
            dissipation_coefficient=0.004,
            angular_velocity=1.7,
            local_flow_acceleration=-0.9,
        )
        result = evaluate_riziotis_ibl_residuals(
            state, derivatives, closure
        )

        rho = 1.1
        velocity = 7.0
        delta = 0.004
        theta = 0.0015
        shape = delta / theta
        shape_star = 0.0027 / theta
        shape_starstar = 0.0003 / theta
        expected_momentum = (
            0.08 / (rho * velocity**2)
            + 0.004
            + (2.0 + shape) * theta / velocity * -1.3
            + theta / rho * 0.02
            - (-0.006 / 2.0)
        )
        expected_energy = (
            -0.12 / (rho * velocity**3)
            + 0.006 / (rho * velocity)
            + 2.0 * shape_starstar * theta / velocity**2 * 0.7
            - shape_star / (rho * velocity**2) * 0.08
            - 4.0 * 1.7 / velocity * -0.0002
            + theta * -0.11
            + (
                2.0 * shape_starstar
                + shape_star * (1.0 - shape)
            )
            * theta
            / velocity
            * -1.3
            - 2.0 * 0.004
            - 2.0 * -0.9 / velocity**2 * delta
            + 0.5 * shape_star * -0.006
        )
        self.assertAlmostEqual(
            result.momentum_residual, expected_momentum, places=15
        )
        self.assertAlmostEqual(
            result.energy_residual, expected_energy, places=15
        )

    def test_local_acceleration_uses_displacement_not_density_thickness(self):
        common = dict(
            edge_density=1.2,
            edge_tangential_velocity=6.0,
            edge_speed=6.0,
            displacement_thickness=0.005,
            momentum_thickness=0.002,
            kinetic_energy_thickness=0.003,
        )
        closure = IBLClosureTerms2D(
            skin_friction_coefficient=0.0,
            dissipation_coefficient=0.0,
            local_flow_acceleration=2.0,
        )
        incompressible = evaluate_riziotis_ibl_residuals(
            IBLStationState2D(**common, density_thickness=0.0),
            zero_derivatives(),
            closure,
        )
        compressible = evaluate_riziotis_ibl_residuals(
            IBLStationState2D(**common, density_thickness=0.001),
            zero_derivatives(),
            closure,
        )
        self.assertEqual(
            incompressible.energy_local_acceleration_rhs,
            compressible.energy_local_acceleration_rhs,
        )
        self.assertAlmostEqual(
            incompressible.energy_local_acceleration_rhs,
            2.0 * 2.0 * 0.005 / 6.0**2,
        )

    def test_eq10_distinguishes_tangential_velocity_from_speed_magnitude(self):
        result = evaluate_riziotis_ibl_residuals(
            IBLStationState2D(
                edge_density=1.2,
                edge_tangential_velocity=6.0,
                edge_speed=10.0,
                displacement_thickness=0.005,
                momentum_thickness=0.002,
                kinetic_energy_thickness=0.003,
            ),
            zero_derivatives(
                dt_rho_displacement_thickness=0.012,
            ),
            IBLClosureTerms2D(
                skin_friction_coefficient=0.0,
                dissipation_coefficient=0.0,
                local_flow_acceleration=2.0,
            ),
        )
        self.assertAlmostEqual(
            result.energy_displacement_storage,
            0.012 / (1.2 * 6.0),
        )
        self.assertAlmostEqual(
            result.energy_local_acceleration_rhs,
            2.0 * 2.0 * 0.005 / 10.0**2,
        )

    def test_second_energy_storage_is_displacement_not_density_thickness(self):
        derivatives = zero_derivatives(
            dt_rho_displacement_thickness=0.012
        )
        closure = IBLClosureTerms2D(
            skin_friction_coefficient=0.0,
            dissipation_coefficient=0.0,
        )
        common = dict(
            edge_density=1.2,
            edge_tangential_velocity=6.0,
            edge_speed=6.0,
            displacement_thickness=0.005,
            momentum_thickness=0.002,
            kinetic_energy_thickness=0.003,
        )
        zero_density_thickness = evaluate_riziotis_ibl_residuals(
            IBLStationState2D(**common, density_thickness=0.0),
            derivatives,
            closure,
        )
        nonzero_density_thickness = evaluate_riziotis_ibl_residuals(
            IBLStationState2D(**common, density_thickness=0.001),
            derivatives,
            closure,
        )
        expected = 0.012 / (1.2 * 6.0)
        self.assertAlmostEqual(
            zero_density_thickness.energy_displacement_storage,
            expected,
        )
        self.assertEqual(
            zero_density_thickness.energy_displacement_storage,
            nonzero_density_thickness.energy_displacement_storage,
        )

    def test_invalid_or_inconsistent_thickness_state_fails_closed(self):
        invalid_cases = (
            {"edge_tangential_velocity": 0.0},
            {"edge_speed": 0.0},
            {"edge_density": -1.0},
            {
                "displacement_thickness": 0.0005,
                "momentum_thickness": 0.001,
            },
            {
                "kinetic_energy_thickness": 0.0005,
                "momentum_thickness": 0.001,
            },
        )
        defaults = {
            "edge_density": 1.2,
            "edge_tangential_velocity": 8.0,
            "edge_speed": 8.0,
            "displacement_thickness": 0.003,
            "momentum_thickness": 0.001,
            "kinetic_energy_thickness": 0.0017,
        }
        for override in invalid_cases:
            with self.subTest(override=override):
                values = defaults | override
                with self.assertRaises(SVIDWValidationError):
                    IBLStationState2D(**values)


if __name__ == "__main__":
    unittest.main()
