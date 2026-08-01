import sys
import unittest
from pathlib import Path

import numpy as np

PLATFORM = Path(__file__).resolve().parents[1]
if str(PLATFORM) not in sys.path:
    sys.path.insert(0, str(PLATFORM))

from claim_runtime.svi_dw_outer_2d import (  # noqa: E402
    solve_svi_dw_outer_foundation,
)
from claim_runtime.svi_dw_types import (  # noqa: E402
    ActualSurface2D,
    DoubleWakeState2D,
    DualSideIBLState,
    IBLRegime,
    NACA4SectionConfig,
    SVIDWFoundationConfig,
    SVIDWFoundationScopeError,
    SVIDWValidationError,
    WakeBranch2D,
)


def symmetric_config(
    panels_per_side: int,
    *,
    angle_of_attack_deg: float,
) -> SVIDWFoundationConfig:
    return SVIDWFoundationConfig(
        section=NACA4SectionConfig(
            maximum_camber=0.0,
            camber_location=0.4,
            thickness_ratio=0.15,
            chord=1.0,
        ),
        panels_per_side=panels_per_side,
        freestream_speed=11.0,
        angle_of_attack_deg=angle_of_attack_deg,
        density=1.2,
    )


def circular_surface(
    panels_per_side: int,
) -> tuple[NACA4SectionConfig, ActualSurface2D]:
    """Independent unit-circle fixture represented by a regular polygon."""
    n_side = panels_per_side
    section = NACA4SectionConfig(
        maximum_camber=0.0,
        camber_location=0.4,
        thickness_ratio=0.15,
        chord=2.0,
    )
    lower_angle = np.linspace(np.pi, 2.0 * np.pi, n_side + 1)
    upper_angle = np.linspace(np.pi, 0.0, n_side + 1)
    lower = np.column_stack((
        1.0 + np.cos(lower_angle),
        np.sin(lower_angle),
    ))
    upper = np.column_stack((
        1.0 + np.cos(upper_angle),
        np.sin(upper_angle),
    ))
    lower[0] = upper[0] = (0.0, 0.0)
    lower[-1] = upper[-1] = (2.0, 0.0)
    contour = np.vstack((lower[::-1], upper[1:]))
    segment = np.diff(contour, axis=0)
    length = np.linalg.norm(segment, axis=1)
    tangent = segment / length[:, None]
    normal = np.column_stack((-tangent[:, 1], tangent[:, 0]))
    midpoint = 0.5 * (contour[:-1] + contour[1:])
    area = 0.5 * float(np.sum(
        contour[:-1, 0] * contour[1:, 1]
        - contour[1:, 0] * contour[:-1, 1]
    ))
    surface = ActualSurface2D(
        section=section,
        upper_nodes=upper,
        lower_nodes=lower,
        contour_nodes=contour,
        panel_midpoints=midpoint,
        panel_tangents=tangent,
        panel_outward_normals=normal,
        panel_lengths=length,
        panel_side=np.array(
            ["lower"] * n_side + ["upper"] * n_side
        ),
        panel_chord_fraction=np.clip(
            midpoint[:, 0] / section.chord, 0.0, 1.0
        ),
        signed_area=area,
    )
    return section, surface


class SVIDWOuter2DTests(unittest.TestCase):
    def test_circle_matches_independent_analytic_pressure(self):
        section, surface = circular_surface(40)
        config = SVIDWFoundationConfig(
            section=section,
            panels_per_side=40,
            freestream_speed=3.0,
            angle_of_attack_deg=0.0,
            density=1.2,
        )
        result = solve_svi_dw_outer_foundation(
            config, surface=surface
        )
        radial = (
            surface.panel_midpoints - np.array([1.0, 0.0])
        )
        polar_angle = np.arctan2(radial[:, 1], radial[:, 0])
        analytic_cp = 1.0 - 4.0 * np.sin(polar_angle) ** 2
        np.testing.assert_allclose(
            result.pressure_coefficient,
            analytic_cp,
            rtol=0.0,
            atol=3.0e-13,
        )
        self.assertLess(abs(result.drag_coefficient), 3.0e-13)
        self.assertLess(abs(result.lift_coefficient), 3.0e-13)

    def test_symmetric_naca0015_has_zero_lift_at_zero_angle(self):
        result = solve_svi_dw_outer_foundation(
            symmetric_config(48, angle_of_attack_deg=0.0)
        )
        self.assertLess(abs(result.lift_coefficient), 2.0e-12)
        self.assertLess(abs(result.bound_circulation), 2.0e-12)
        self.assertLess(result.maximum_relative_normal_residual, 2.0e-12)
        self.assertLess(result.relative_kutta_residual, 2.0e-12)

    def test_small_positive_angle_has_positive_convergent_lift(self):
        coarse = solve_svi_dw_outer_foundation(
            symmetric_config(32, angle_of_attack_deg=4.0)
        )
        medium = solve_svi_dw_outer_foundation(
            symmetric_config(64, angle_of_attack_deg=4.0)
        )
        fine = solve_svi_dw_outer_foundation(
            symmetric_config(96, angle_of_attack_deg=4.0)
        )
        self.assertGreater(coarse.lift_coefficient, 0.0)
        self.assertGreater(medium.lift_coefficient, 0.0)
        self.assertGreater(fine.lift_coefficient, 0.0)
        self.assertLess(
            abs(fine.lift_coefficient - medium.lift_coefficient),
            abs(medium.lift_coefficient - coarse.lift_coefficient),
        )
        self.assertLess(
            abs(fine.lift_coefficient - medium.lift_coefficient),
            0.02,
        )
        for result in (coarse, medium, fine):
            self.assertLess(
                result.maximum_relative_normal_residual, 2.0e-12
            )
            self.assertLess(result.relative_kutta_residual, 2.0e-12)
        # Constant-strength collocation panels conserve the pointwise
        # boundary equation to roundoff; their integrated source and
        # circulation ledgers converge under surface refinement.
        self.assertLess(
            medium.relative_source_flux, coarse.relative_source_flux
        )
        self.assertLess(
            fine.relative_source_flux, medium.relative_source_flux
        )
        self.assertLess(fine.relative_source_flux, 1.0e-3)
        self.assertLess(
            abs(medium.circulation_ledger_residual),
            abs(coarse.circulation_ledger_residual),
        )
        self.assertLess(
            abs(fine.circulation_ledger_residual),
            abs(medium.circulation_ledger_residual),
        )
        self.assertLess(
            abs(fine.circulation_ledger_residual), 4.0e-3
        )

    def test_pressure_lift_converges_to_kutta_joukowski_ledger(self):
        result = solve_svi_dw_outer_foundation(
            symmetric_config(128, angle_of_attack_deg=4.0)
        )
        expected_lift = (
            2.0
            * result.bound_circulation
            / (
                result.config.freestream_speed
                * result.config.section.chord
            )
        )
        self.assertLess(
            abs(result.lift_coefficient - expected_lift), 2.0e-3
        )

    def test_inviscid_pressure_drag_converges_towards_zero(self):
        coarse = solve_svi_dw_outer_foundation(
            symmetric_config(32, angle_of_attack_deg=4.0)
        )
        medium = solve_svi_dw_outer_foundation(
            symmetric_config(64, angle_of_attack_deg=4.0)
        )
        fine = solve_svi_dw_outer_foundation(
            symmetric_config(96, angle_of_attack_deg=4.0)
        )
        self.assertGreater(
            abs(coarse.drag_coefficient),
            abs(medium.drag_coefficient),
        )
        self.assertGreater(
            abs(medium.drag_coefficient),
            abs(fine.drag_coefficient),
        )
        self.assertLess(abs(fine.drag_coefficient), 3.0e-4)

    def test_symmetric_section_obeys_angle_reversal(self):
        positive = solve_svi_dw_outer_foundation(
            symmetric_config(64, angle_of_attack_deg=4.0)
        )
        negative = solve_svi_dw_outer_foundation(
            symmetric_config(64, angle_of_attack_deg=-4.0)
        )
        self.assertLess(
            abs(
                positive.lift_coefficient
                + negative.lift_coefficient
            ),
            2.0e-12,
        )
        self.assertLess(
            abs(
                positive.drag_coefficient
                - negative.drag_coefficient
            ),
            2.0e-12,
        )
        self.assertLess(
            abs(
                positive.bound_circulation
                + negative.bound_circulation
            ),
            2.0e-12,
        )

    def test_zero_deficit_and_empty_double_wake_are_same_inviscid_baseline(self):
        config = symmetric_config(40, angle_of_attack_deg=3.0)
        baseline = solve_svi_dw_outer_foundation(config)
        explicit_zero = solve_svi_dw_outer_foundation(
            config,
            ibl_state=DualSideIBLState.quiescent(
                config.panels_per_side
            ),
            double_wake_state=DoubleWakeState2D.quiescent(),
            transpiration_velocity=np.zeros(
                2 * config.panels_per_side
            ),
        )
        np.testing.assert_array_equal(
            explicit_zero.source_strength,
            baseline.source_strength,
        )
        np.testing.assert_array_equal(
            explicit_zero.pressure_coefficient,
            baseline.pressure_coefficient,
        )
        np.testing.assert_array_equal(
            explicit_zero.force_coefficient_xy,
            baseline.force_coefficient_xy,
        )
        edge_velocity_only = DualSideIBLState(
            displacement_thickness=np.zeros(
                (2, config.panels_per_side)
            ),
            momentum_thickness=np.zeros(
                (2, config.panels_per_side)
            ),
            kinetic_energy_thickness=np.zeros(
                (2, config.panels_per_side)
            ),
            edge_velocity=np.ones(
                (2, config.panels_per_side)
            ),
            skin_friction_coefficient=np.zeros(
                (2, config.panels_per_side)
            ),
            transpiration_velocity=np.zeros(
                (2, config.panels_per_side)
            ),
        )
        edge_observable = solve_svi_dw_outer_foundation(
            config, ibl_state=edge_velocity_only
        )
        np.testing.assert_array_equal(
            edge_observable.pressure_coefficient,
            baseline.pressure_coefficient,
        )

    def test_nonzero_ibl_or_transpiration_is_not_claimed_by_s0(self):
        config = symmetric_config(16, angle_of_attack_deg=2.0)
        nonzero = np.zeros((2, config.panels_per_side))
        nonzero[0, 4] = 1.0e-4
        active_ibl = DualSideIBLState(
            displacement_thickness=nonzero,
            momentum_thickness=0.5 * nonzero,
            kinetic_energy_thickness=0.75 * nonzero,
            edge_velocity=np.zeros_like(nonzero),
            skin_friction_coefficient=np.zeros_like(nonzero),
            transpiration_velocity=np.zeros_like(nonzero),
            regime=np.where(
                nonzero > 0.0,
                IBLRegime.LAMINAR.value,
                IBLRegime.INACTIVE.value,
            ),
        )
        with self.assertRaises(SVIDWFoundationScopeError):
            solve_svi_dw_outer_foundation(
                config, ibl_state=active_ibl
            )
        transpiration = np.zeros(2 * config.panels_per_side)
        transpiration[3] = 1.0e-3
        with self.assertRaises(SVIDWFoundationScopeError):
            solve_svi_dw_outer_foundation(
                config, transpiration_velocity=transpiration
            )
        zero_geometry_wake = DoubleWakeState2D(
            trailing_edge=WakeBranch2D.empty("trailing_edge"),
            separation=WakeBranch2D(
                role="separation",
                nodes=np.array([[1.0, 0.0], [2.0, 0.0]]),
                segment_circulation=np.zeros(1),
            ),
        )
        solve_svi_dw_outer_foundation(
            config, double_wake_state=zero_geometry_wake
        )
        active_wake = DoubleWakeState2D(
            trailing_edge=WakeBranch2D.empty("trailing_edge"),
            separation=WakeBranch2D(
                role="separation",
                nodes=np.array([[1.0, 0.0], [2.0, 0.0]]),
                segment_circulation=np.array([1.0e-3]),
            ),
        )
        with self.assertRaises(SVIDWFoundationScopeError):
            solve_svi_dw_outer_foundation(
                config, double_wake_state=active_wake
            )

    def test_near_coincident_two_sided_wall_fails_condition_guard(self):
        config = SVIDWFoundationConfig(
            section=NACA4SectionConfig(thickness_ratio=1.0e-10),
            panels_per_side=40,
            freestream_speed=1.0,
            angle_of_attack_deg=4.0,
        )
        with self.assertRaisesRegex(
            SVIDWValidationError, "thin-sheet"
        ):
            solve_svi_dw_outer_foundation(config)


if __name__ == "__main__":
    unittest.main()
