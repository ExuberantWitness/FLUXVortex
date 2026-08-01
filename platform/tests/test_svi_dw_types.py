import sys
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

PLATFORM = Path(__file__).resolve().parents[1]
if str(PLATFORM) not in sys.path:
    sys.path.insert(0, str(PLATFORM))

from claim_runtime.svi_dw_types import (  # noqa: E402
    DoubleWakeState2D,
    DualSideIBLState,
    IBLRegime,
    NACA4SectionConfig,
    SVIDWFoundationConfig,
    SVIDWValidationError,
    SurfaceTractionState2D,
    WakeBranch2D,
    build_naca4_actual_surface,
)


class SVIDWTypesTests(unittest.TestCase):
    def test_actual_naca_wall_is_closed_clockwise_and_two_sided(self):
        section = NACA4SectionConfig(
            maximum_camber=0.02,
            camber_location=0.4,
            thickness_ratio=0.06,
        )
        wall = build_naca4_actual_surface(
            section, panels_per_side=32
        )
        np.testing.assert_array_equal(
            wall.contour_nodes[0], wall.contour_nodes[-1]
        )
        np.testing.assert_array_equal(
            wall.upper_nodes[0], wall.lower_nodes[0]
        )
        np.testing.assert_array_equal(
            wall.upper_nodes[-1], wall.lower_nodes[-1]
        )
        self.assertLess(wall.signed_area, 0.0)
        self.assertEqual(wall.panel_count, 64)
        self.assertTrue(
            np.all(wall.panel_side[:32] == "lower")
        )
        self.assertTrue(
            np.all(wall.panel_side[32:] == "upper")
        )
        self.assertGreater(
            float(np.max(wall.upper_nodes[:, 1])), 0.0
        )
        self.assertLess(
            float(np.min(wall.lower_nodes[:, 1])), 0.0
        )

    def test_actual_surface_rejects_forged_derived_geometry(self):
        wall = build_naca4_actual_surface(
            NACA4SectionConfig(), panels_per_side=16
        )
        invalid_changes = (
            {
                "panel_outward_normals":
                    -wall.panel_outward_normals
            },
            {"panel_tangents": -wall.panel_tangents},
            {
                "panel_midpoints":
                    wall.panel_midpoints + np.array([0.01, 0.0])
            },
            {"panel_lengths": 1.01 * wall.panel_lengths},
            {"signed_area": -999.0},
            {
                "upper_nodes":
                    wall.upper_nodes + np.array([0.0, 0.01])
            },
        )
        for change in invalid_changes:
            with self.subTest(field=next(iter(change))):
                with self.assertRaises(SVIDWValidationError):
                    replace(wall, **change)

    def test_canonical_ibl_map_is_explicit_and_round_trips(self):
        wall = build_naca4_actual_surface(
            NACA4SectionConfig(), panels_per_side=8
        )
        mapping = wall.canonical_ibl_map
        upper = 100.0 + np.arange(8, dtype=float)
        lower = 200.0 + np.arange(8, dtype=float)
        canonical = np.vstack((upper, lower))

        scalar_contour = mapping.scalar_to_contour(canonical)
        np.testing.assert_array_equal(
            scalar_contour[:8], lower[::-1]
        )
        np.testing.assert_array_equal(
            scalar_contour[8:], upper
        )
        np.testing.assert_array_equal(
            mapping.scalar_from_contour(scalar_contour), canonical
        )

        tangent_contour = mapping.tangential_to_contour(canonical)
        np.testing.assert_array_equal(
            tangent_contour[:8], -lower[::-1]
        )
        np.testing.assert_array_equal(
            tangent_contour[8:], upper
        )
        np.testing.assert_array_equal(
            mapping.tangential_from_contour(tangent_contour),
            canonical,
        )

    def test_configuration_rejects_unknown_target_and_multiplier_fields(self):
        with self.assertRaises(SVIDWValidationError):
            SVIDWFoundationConfig.from_mapping(
                {"target_lift": 2.0}
            )
        with self.assertRaises(SVIDWValidationError):
            SVIDWFoundationConfig.from_mapping(
                {"force_multiplier": 1.1}
            )
        with self.assertRaises(SVIDWValidationError):
            SVIDWFoundationConfig.from_mapping(
                {"unregistered_switch": True}
            )
        with self.assertRaises(SVIDWValidationError):
            SVIDWFoundationConfig(panels_per_side=7)
        with self.assertRaises(SVIDWValidationError):
            NACA4SectionConfig(
                thickness_ratio=0.15,
                closed_trailing_edge=False,
            )

    def test_ibl_double_wake_and_traction_states_are_strongly_typed(self):
        wall = build_naca4_actual_surface(
            NACA4SectionConfig(), panels_per_side=16
        )
        ibl = DualSideIBLState.quiescent(16)
        self.assertTrue(ibl.is_quiescent)
        self.assertFalse(ibl.has_viscous_coupling)
        edge_only = DualSideIBLState(
            displacement_thickness=np.zeros((2, 16)),
            momentum_thickness=np.zeros((2, 16)),
            kinetic_energy_thickness=np.zeros((2, 16)),
            edge_velocity=np.ones((2, 16)),
            skin_friction_coefficient=np.zeros((2, 16)),
            transpiration_velocity=np.zeros((2, 16)),
        )
        self.assertFalse(edge_only.has_viscous_coupling)
        self.assertTrue(edge_only.is_quiescent)
        with self.assertRaises(SVIDWValidationError):
            DualSideIBLState(
                displacement_thickness=-np.ones((2, 16)),
                momentum_thickness=np.zeros((2, 16)),
                kinetic_energy_thickness=np.zeros((2, 16)),
                edge_velocity=np.zeros((2, 16)),
                skin_friction_coefficient=np.zeros((2, 16)),
                transpiration_velocity=np.zeros((2, 16)),
            )
        delta = np.full((2, 16), 2.0e-3)
        theta = np.full((2, 16), 1.0e-3)
        theta_k = np.full((2, 16), 1.5e-3)
        laminar = DualSideIBLState(
            displacement_thickness=delta,
            momentum_thickness=theta,
            kinetic_energy_thickness=theta_k,
            edge_velocity=np.ones((2, 16)),
            skin_friction_coefficient=np.zeros((2, 16)),
            transpiration_velocity=np.zeros((2, 16)),
            regime=IBLRegime.LAMINAR,
            transition_amplification=np.ones((2, 16)),
        )
        self.assertTrue(laminar.has_viscous_coupling)
        np.testing.assert_allclose(laminar.shape_factor, 2.0)
        np.testing.assert_allclose(
            laminar.kinetic_energy_shape_factor, 1.5
        )
        with self.assertRaises(SVIDWValidationError):
            replace(
                laminar,
                maximum_shear_stress_coefficient=np.ones((2, 16)),
            )
        turbulent_regime = np.full(
            (2, 16), IBLRegime.TURBULENT.value
        )
        with self.assertRaises(SVIDWValidationError):
            replace(
                laminar,
                regime=turbulent_regime,
                transition_amplification=np.ones((2, 16)),
                maximum_shear_stress_coefficient=np.ones((2, 16)),
            )
        wakes = DoubleWakeState2D.quiescent()
        self.assertTrue(wakes.is_quiescent)
        self.assertFalse(wakes.has_induction)
        with self.assertRaises(SVIDWValidationError):
            WakeBranch2D(
                role="separation",
                nodes=np.zeros((2, 2)),
                segment_circulation=np.zeros(2),
            )
        with self.assertRaises(SVIDWValidationError):
            WakeBranch2D(
                role="separation",
                nodes=np.zeros((2, 2)),
                segment_circulation=np.zeros(1),
            )
        zero_wake = WakeBranch2D(
            role="separation",
            nodes=np.array([[0.0, 0.0], [1.0, 0.0]]),
            segment_circulation=np.zeros(1),
        )
        self.assertFalse(zero_wake.has_induction)
        active_wake = WakeBranch2D(
            role="separation",
            nodes=np.array([[0.0, 0.0], [1.0, 0.0]]),
            segment_circulation=np.array([0.25]),
        )
        self.assertTrue(active_wake.has_induction)
        self.assertEqual(active_wake.total_circulation, 0.25)
        pressure = np.linspace(-1.0, 1.0, wall.panel_count)
        shear = np.zeros(wall.panel_count)
        traction = SurfaceTractionState2D.from_pressure_and_shear(
            wall, pressure=pressure, wall_shear=shear
        )
        expected = (
            -pressure[:, None]
            * wall.panel_outward_normals
            * wall.panel_lengths[:, None]
        )
        np.testing.assert_allclose(
            traction.panel_force, expected, rtol=0.0, atol=0.0
        )
        expected_moment = (
            wall.panel_midpoints[:, 0] * expected[:, 1]
            - wall.panel_midpoints[:, 1] * expected[:, 0]
        )
        np.testing.assert_allclose(
            traction.panel_moment,
            expected_moment,
            rtol=0.0,
            atol=0.0,
        )
        self.assertAlmostEqual(
            traction.resultant_moment,
            float(np.sum(expected_moment)),
        )
        with self.assertRaises(SVIDWValidationError):
            SurfaceTractionState2D(
                surface=wall,
                pressure=pressure,
                wall_shear=shear,
                traction=traction.traction,
                panel_force=traction.panel_force + 1.0,
            )

    def test_ibl_directed_shear_maps_to_force_and_reference_moment(self):
        wall = build_naca4_actual_surface(
            NACA4SectionConfig(), panels_per_side=8
        )
        ibl_shear = np.vstack((
            np.linspace(1.0, 2.0, 8),
            np.linspace(3.0, 4.0, 8),
        ))
        pressure = np.zeros(wall.panel_count)
        origin = SurfaceTractionState2D.from_pressure_and_ibl_shear(
            wall,
            pressure=pressure,
            ibl_wall_shear=ibl_shear,
        )
        expected_contour_shear = np.concatenate((
            -ibl_shear[1, ::-1],
            ibl_shear[0],
        ))
        np.testing.assert_array_equal(
            origin.wall_shear, expected_contour_shear
        )

        reference = np.array([0.25, -0.1])
        shifted = SurfaceTractionState2D.from_pressure_and_ibl_shear(
            wall,
            pressure=pressure,
            ibl_wall_shear=ibl_shear,
            reference_point=reference,
        )
        force = origin.resultant_force
        expected_shift = (
            origin.resultant_moment
            - reference[0] * force[1]
            + reference[1] * force[0]
        )
        self.assertAlmostEqual(
            shifted.resultant_moment, expected_shift
        )


if __name__ == "__main__":
    unittest.main()
