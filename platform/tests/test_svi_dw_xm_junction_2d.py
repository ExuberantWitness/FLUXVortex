import sys
import unittest
from pathlib import Path

import numpy as np

PLATFORM = Path(__file__).resolve().parents[1]
if str(PLATFORM) not in sys.path:
    sys.path.insert(0, str(PLATFORM))

from claim_runtime.svi_dw_types import (  # noqa: E402
    NACA4SectionConfig,
    SVIDWValidationError,
    build_naca4_actual_surface,
)
from claim_runtime.svi_dw_unsteady_outer_2d import (  # noqa: E402
    TENearWakeSegment2D,
    constant_vortex_segment_velocity_body,
)
from claim_runtime import svi_dw_xm_junction_2d as xm_junction  # noqa: E402
from claim_runtime.svi_dw_xm_junction_2d import (  # noqa: E402
    build_xia_ccw_geometry,
    initialize_xm_bound_state,
    linear_vortex_panel_velocity_basis,
    solve_xm_forming_step,
)


def naca0015(panels_per_side=32):
    return build_naca4_actual_surface(
        NACA4SectionConfig(
            maximum_camber=0.0,
            camber_location=0.4,
            thickness_ratio=0.15,
            chord=1.0,
        ),
        panels_per_side=panels_per_side,
    )


def freestream(alpha_deg, speed=1.0):
    alpha = np.deg2rad(alpha_deg)
    return speed * np.array((np.cos(alpha), np.sin(alpha)))


class LinearVortexPanelBasisTests(unittest.TestCase):
    def test_preregistered_closed_form_point_values(self):
        basis = linear_vortex_panel_velocity_basis(
            np.array(((0.6, 0.8),)),
            np.array(((0.0, 0.0),)),
            np.array(((2.0, 0.0),)),
        )
        expected_start = np.array(
            (-0.1584393245247987, -0.00198777112016646)
        )
        expected_end = np.array(
            (-0.11135238755548409, -0.07404941370833168)
        )
        np.testing.assert_allclose(
            basis.start_node_velocity[0, 0],
            expected_start,
            rtol=0.0,
            atol=2.0e-16,
        )
        np.testing.assert_allclose(
            basis.end_node_velocity[0, 0],
            expected_end,
            rtol=0.0,
            atol=2.0e-16,
        )
        combined = (
            1.3 * basis.start_node_velocity[0, 0]
            - 0.4 * basis.end_node_velocity[0, 0]
        )
        np.testing.assert_allclose(
            combined,
            np.array(
                (-0.16143016686004469, 0.027035663027116272)
            ),
            rtol=0.0,
            atol=3.0e-16,
        )

    def test_analytic_basis_matches_gauss_legendre_line_integral(self):
        starts = np.array(((-0.4, 0.2), (0.7, -0.3)))
        ends = np.array(((0.8, 0.55), (0.15, 0.9)))
        points = np.array(
            ((1.7, 1.3), (-1.1, 0.8), (0.2, -1.4), (2.0, -0.5))
        )
        analytic = linear_vortex_panel_velocity_basis(
            points, starts, ends
        )
        abscissa, weights = np.polynomial.legendre.leggauss(64)
        for panel_index, (start, end) in enumerate(zip(starts, ends)):
            segment = end - start
            length = float(np.linalg.norm(segment))
            tangent = segment / length
            coordinate = 0.5 * length * (abscissa + 1.0)
            quadrature_weight = 0.5 * length * weights
            sources = start + coordinate[:, None] * tangent
            for target_index, point in enumerate(points):
                relative = point - sources
                kernel = np.column_stack(
                    (-relative[:, 1], relative[:, 0])
                ) / (
                    2.0
                    * np.pi
                    * np.einsum("ij,ij->i", relative, relative)[:, None]
                )
                start_numeric = np.sum(
                    quadrature_weight[:, None]
                    * (1.0 - coordinate / length)[:, None]
                    * kernel,
                    axis=0,
                )
                end_numeric = np.sum(
                    quadrature_weight[:, None]
                    * (coordinate / length)[:, None]
                    * kernel,
                    axis=0,
                )
                scale = max(
                    float(np.linalg.norm(start_numeric)),
                    float(np.linalg.norm(end_numeric)),
                    1.0 / (2.0 * np.pi),
                )
                self.assertLess(
                    np.linalg.norm(
                        analytic.start_node_velocity[
                            target_index, panel_index
                        ]
                        - start_numeric
                    )
                    / scale,
                    1.0e-11,
                )
                self.assertLess(
                    np.linalg.norm(
                        analytic.end_node_velocity[
                            target_index, panel_index
                        ]
                        - end_numeric
                    )
                    / scale,
                    1.0e-11,
                )

    def test_equal_nodes_reduce_to_existing_constant_vortex_segment(self):
        start = np.array((-0.35, 0.15))
        end = np.array((0.85, 0.62))
        length = float(np.linalg.norm(end - start))
        points = np.array(
            ((1.4, 1.1), (-0.8, 0.9), (0.1, -0.7), (1.8, -0.4))
        )
        linear = linear_vortex_panel_velocity_basis(
            points, start[None, :], end[None, :]
        )
        constant = constant_vortex_segment_velocity_body(
            points,
            TENearWakeSegment2D(
                start_body=start,
                end_body=end,
                orientation_side="upper",
                mean_emission_speed=length,
                time_step=1.0,
            ),
        )
        np.testing.assert_allclose(
            linear.start_node_velocity[:, 0]
            + linear.end_node_velocity[:, 0],
            constant,
            rtol=0.0,
            atol=2.0e-15,
        )

    def test_endpoint_exchange_and_rotation_are_covariant(self):
        start = np.array((-0.2, 0.4))
        end = np.array((0.9, -0.1))
        points = np.array(((1.7, 0.8), (-0.6, -1.0), (0.2, 1.4)))
        original = linear_vortex_panel_velocity_basis(
            points, start[None], end[None]
        )
        reversed_panel = linear_vortex_panel_velocity_basis(
            points, end[None], start[None]
        )
        np.testing.assert_allclose(
            reversed_panel.start_node_velocity[:, 0],
            original.end_node_velocity[:, 0],
            rtol=0.0,
            atol=3.0e-16,
        )
        np.testing.assert_allclose(
            reversed_panel.end_node_velocity[:, 0],
            original.start_node_velocity[:, 0],
            rtol=0.0,
            atol=3.0e-16,
        )

        angle = 0.73
        rotation = np.array(
            (
                (np.cos(angle), -np.sin(angle)),
                (np.sin(angle), np.cos(angle)),
            )
        )
        rotated = linear_vortex_panel_velocity_basis(
            points @ rotation.T,
            (start @ rotation.T)[None],
            (end @ rotation.T)[None],
        )
        np.testing.assert_allclose(
            rotated.start_node_velocity[:, 0],
            original.start_node_velocity[:, 0] @ rotation.T,
            rtol=0.0,
            atol=4.0e-16,
        )
        np.testing.assert_allclose(
            rotated.end_node_velocity[:, 0],
            original.end_node_velocity[:, 0] @ rotation.T,
            rtol=0.0,
            atol=4.0e-16,
        )

    def test_nodal_strength_superposition_is_exactly_linear(self):
        starts = np.array(((-0.5, 0.1), (0.2, -0.6)))
        ends = np.array(((0.4, 0.7), (1.0, -0.2)))
        points = np.array(((-1.0, 0.9), (1.6, 1.1), (0.5, -1.2)))
        basis = linear_vortex_panel_velocity_basis(
            points, starts, ends
        )
        start_a = np.array((0.7, -1.3))
        end_a = np.array((-0.2, 0.9))
        start_b = np.array((-1.1, 0.4))
        end_b = np.array((0.6, -0.8))
        combined = (
            basis.start_node_velocity
            * (start_a + start_b)[None, :, None]
            + basis.end_node_velocity
            * (end_a + end_b)[None, :, None]
        )
        split = (
            basis.start_node_velocity * start_a[None, :, None]
            + basis.end_node_velocity * end_a[None, :, None]
            + basis.start_node_velocity * start_b[None, :, None]
            + basis.end_node_velocity * end_b[None, :, None]
        )
        np.testing.assert_allclose(
            combined,
            split,
            rtol=0.0,
            atol=2.0e-16,
        )

    def test_endpoint_target_is_rejected(self):
        with self.assertRaises(SVIDWValidationError):
            linear_vortex_panel_velocity_basis(
                np.array(((0.0, 0.0),)),
                np.array(((0.0, 0.0),)),
                np.array(((1.0, 0.0),)),
            )


class XMCoupledJunctionTests(unittest.TestCase):
    def geometry(self, panels_per_side=32, epsilon_ratio=1.0 / 8.0):
        return build_xia_ccw_geometry(
            naca0015(panels_per_side),
            epsilon_over_te_panel=epsilon_ratio,
        )

    def assert_residual_gate(self, state):
        self.assertLess(
            state.residuals.maximum_relative_normal_residual, 1.0e-10
        )
        self.assertLess(
            state.residuals.normalized_kelvin_residual, 1.0e-10
        )
        self.assertLess(
            state.residuals.maximum_normalized_linear_system_residual,
            1.0e-10,
        )
        self.assertTrue(
            np.isfinite(state.residuals.system_condition_number)
        )
        self.assertLess(
            state.residuals.system_condition_number, 1.0e12
        )
        if state.residuals.normalized_kutta_residual is not None:
            self.assertLess(
                state.residuals.normalized_kutta_residual, 1.0e-10
            )

    def test_actual_surface_is_ccw_cropped_and_has_expected_finite_wedge(self):
        geometry = self.geometry()
        closed_for_area = np.vstack(
            (geometry.contour_nodes_ccw, geometry.contour_nodes_ccw[0])
        )
        signed_area = 0.5 * np.sum(
            closed_for_area[:-1, 0] * closed_for_area[1:, 1]
            - closed_for_area[1:, 0] * closed_for_area[:-1, 1]
        )
        self.assertGreater(signed_area, 0.0)
        self.assertAlmostEqual(
            geometry.epsilon,
            geometry.epsilon_over_te_panel
            * geometry.trailing_edge_reference_panel_length,
        )
        self.assertAlmostEqual(
            geometry.trailing_edge_wedge_angle_deg,
            20.563717743195166,
            delta=1.0e-12,
        )
        np.testing.assert_allclose(
            np.linalg.norm(geometry.panel_tangents, axis=1),
            1.0,
            rtol=0.0,
            atol=2.0e-15,
        )

    def test_all_preregistered_wedge_levels_and_self_pv_are_locked(self):
        expected_wedge = {
            32: 20.563717743195166,
            64: 20.58731250734261,
            128: 20.5932247966579,
        }
        for panels, expected in expected_wedge.items():
            with self.subTest(panels=panels):
                geometry = self.geometry(panels_per_side=panels)
                self.assertAlmostEqual(
                    geometry.trailing_edge_wedge_angle_deg,
                    expected,
                    delta=2.0e-12,
                )

        geometry = self.geometry(panels_per_side=32)
        node_velocity, _ = (
            xm_junction._bound_velocity_basis_at_collocation(geometry)
        )
        raw = linear_vortex_panel_velocity_basis(
            geometry.panel_midpoints,
            geometry.contour_nodes_ccw[:-1],
            geometry.contour_nodes_ccw[1:],
        )
        panel = geometry.panel_count // 3
        self_start = (
            node_velocity[panel, panel]
            - raw.end_node_velocity[panel, panel - 1]
        )
        self_end = (
            node_velocity[panel, panel + 1]
            - raw.start_node_velocity[panel, panel + 1]
        )
        tangent = geometry.panel_tangents[panel]
        outward = geometry.panel_outward_normals[panel]
        self.assertAlmostEqual(
            float(self_start @ tangent), 0.0, delta=2.0e-15
        )
        self.assertAlmostEqual(
            float(self_end @ tangent), 0.0, delta=2.0e-15
        )
        self.assertAlmostEqual(
            float(self_start @ outward),
            -1.0 / (2.0 * np.pi),
            delta=2.0e-15,
        )
        self.assertAlmostEqual(
            float(self_end @ outward),
            1.0 / (2.0 * np.pi),
            delta=2.0e-15,
        )
        np.testing.assert_allclose(
            np.einsum(
                "ij,ij->i",
                geometry.panel_tangents,
                geometry.panel_outward_normals,
            ),
            0.0,
            rtol=0.0,
            atol=2.0e-15,
        )

    def test_no_g_initialization_closes_normal_and_kelvin_equations(self):
        state = initialize_xm_bound_state(
            self.geometry(),
            freestream_velocity_body=freestream(6.0),
        )
        self.assertEqual(state.stage, "initialization")
        self.assertFalse(state.no_birth)
        self.assertIsNone(state.formation)
        self.assertEqual(state.forming_length, 0.0)
        self.assertEqual(state.forming_sheet_strength_ccw, 0.0)
        self.assertEqual(
            state.bound_node_strength_ccw.size,
            state.geometry.panel_count + 1,
        )
        self.assertAlmostEqual(
            state.gamma1_upper_physical,
            state.bound_node_strength_ccw[0],
        )
        self.assertAlmostEqual(
            state.gamma2_lower_physical,
            state.bound_node_strength_ccw[-1],
        )
        self.assertAlmostEqual(
            state.u1_plus, state.gamma1_upper_physical
        )
        self.assertAlmostEqual(
            state.u2_minus, -state.gamma2_lower_physical
        )
        self.assert_residual_gate(state)

    def test_alpha_zero_initialization_is_explicit_no_birth(self):
        state = initialize_xm_bound_state(
            self.geometry(),
            freestream_velocity_body=freestream(0.0),
        )
        self.assertEqual(state.stage, "no_birth")
        self.assertTrue(state.no_birth)
        self.assertIsNone(state.formation)
        self.assertIsNone(state.forming_direction_body)
        self.assertIsNone(state.forming_start_body)
        self.assertIsNone(state.forming_end_body)
        self.assertEqual(state.forming_length, 0.0)
        self.assertEqual(state.forming_sheet_strength_ccw, 0.0)
        self.assert_residual_gate(state)

    def test_zero_and_equal_incident_states_do_not_fabricate_a_sheet(self):
        geometry = self.geometry()
        for previous in ((0.0, 0.0), (-1.0, -1.0)):
            with self.subTest(previous=previous):
                state = solve_xm_forming_step(
                    geometry,
                    freestream_velocity_body=freestream(0.0),
                    previous_u1_plus=previous[0],
                    previous_u2_minus=previous[1],
                    time_step=0.01,
                )
                self.assertEqual(state.stage, "no_birth")
                self.assertTrue(state.no_birth)
                self.assertIsNone(state.formation)
                self.assertIsNone(state.forming_direction_body)
                self.assertEqual(state.forming_length, 0.0)
                self.assertEqual(state.forming_sheet_strength_ccw, 0.0)
                self.assertEqual(state.previous_u1_plus, previous[0])
                self.assertEqual(state.previous_u2_minus, previous[1])
                self.assert_residual_gate(state)

    def test_coupled_step_closes_no_through_kelvin_and_published_eq35(self):
        state = solve_xm_forming_step(
            self.geometry(),
            freestream_velocity_body=freestream(6.0),
            previous_u1_plus=-2.0,
            previous_u2_minus=-1.0,
            time_step=0.01,
        )
        self.assertEqual(state.stage, "forming")
        self.assertFalse(state.no_birth)
        self.assertIsNotNone(state.formation)
        self.assertGreater(state.forming_length, 0.0)
        self.assertAlmostEqual(
            state.forming_length,
            state.formation.relative_velocity * 0.01,
            delta=2.0e-16,
        )
        equation35 = (
            state.gamma1_upper_physical
            * np.cos(state.formation.delta_theta1)
            + state.gamma2_lower_physical
            * np.cos(state.formation.delta_theta2)
            - state.forming_sheet_strength_ccw
        )
        self.assertLess(abs(equation35), 1.0e-12)
        self.assertAlmostEqual(
            state.circulation.bound_circulation_ccw
            + state.circulation.forming_circulation_ccw,
            state.circulation.total_circulation_ccw,
        )
        self.assertLess(
            state.formation.normalized_angle_sum_residual, 1.0e-12
        )
        self.assertLess(
            state.formation.normalized_direction_residual, 1.0e-12
        )
        self.assertLess(
            state.formation.normalized_kutta_strength_residual, 1.0e-12
        )
        self.assertLess(
            state.formation.normalized_circulation_rate_residual, 1.0e-12
        )
        self.assertLess(
            state.formation.normalized_momentum_residual, 1.0e-12
        )
        self.assert_residual_gate(state)

    def test_plus_minus_six_degree_canonical_states_are_exact_mirrors(self):
        geometry = self.geometry()
        positive = solve_xm_forming_step(
            geometry,
            freestream_velocity_body=freestream(6.0),
            previous_u1_plus=-2.0,
            previous_u2_minus=-1.0,
            time_step=0.01,
        )
        negative = solve_xm_forming_step(
            geometry,
            freestream_velocity_body=freestream(-6.0),
            previous_u1_plus=-1.0,
            previous_u2_minus=-2.0,
            time_step=0.01,
        )
        np.testing.assert_allclose(
            positive.bound_node_strength_ccw,
            -negative.bound_node_strength_ccw[::-1],
            rtol=0.0,
            atol=2.0e-11,
        )
        self.assertAlmostEqual(
            positive.forming_sheet_strength_ccw,
            -negative.forming_sheet_strength_ccw,
            delta=2.0e-12,
        )
        self.assertAlmostEqual(
            positive.circulation.bound_circulation_ccw,
            -negative.circulation.bound_circulation_ccw,
            delta=2.0e-12,
        )
        self.assertAlmostEqual(
            positive.forming_length,
            negative.forming_length,
            delta=2.0e-16,
        )
        np.testing.assert_allclose(
            positive.forming_direction_body,
            negative.forming_direction_body * np.array((1.0, -1.0)),
            rtol=0.0,
            atol=2.0e-14,
        )
        self.assertAlmostEqual(
            positive.formation.delta_theta1,
            negative.formation.delta_theta2,
            delta=2.0e-15,
        )
        self.assertAlmostEqual(
            positive.formation.delta_theta2,
            negative.formation.delta_theta1,
            delta=2.0e-15,
        )
        self.assert_residual_gate(positive)
        self.assert_residual_gate(negative)


if __name__ == "__main__":
    unittest.main()
