import sys
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

PLATFORM = Path(__file__).resolve().parents[1]
if str(PLATFORM) not in sys.path:
    sys.path.insert(0, str(PLATFORM))

from claim_runtime.svi_dw_types import (  # noqa: E402
    NACA4SectionConfig,
    SVIDWFoundationConfig,
    SVIDWValidationError,
    build_naca4_actual_surface,
)
from claim_runtime.svi_dw_outer_2d import (  # noqa: E402
    solve_svi_dw_outer_foundation,
)
from claim_runtime.svi_dw_unsteady_outer_2d import (  # noqa: E402
    AttachedOuterHistory2D,
    AttachedOuterStepInput2D,
    MaterialVortexBlob2D,
    RigidKinematics2D,
    SVIUnsteadyOuterConvergenceError,
    SVIUnsteadyOuterScopeError,
    TENearWakeSegment2D,
    _select_physical_orientation_branch,
    _solve_orientation_branch,
    build_te_near_wake_segment,
    complete_attached_outer_history,
    completed_step_to_material_blob,
    constant_vortex_segment_velocity_body,
    convect_attached_outer_history_explicit_euler,
    convect_material_vortex_blobs,
    march_attached_unsteady_outer_explicit_euler,
    material_blob_velocity_inertial,
    solve_attached_unsteady_outer_step,
)


def naca0015(panels_per_side: int = 32):
    return build_naca4_actual_surface(
        NACA4SectionConfig(
            maximum_camber=0.0,
            camber_location=0.4,
            thickness_ratio=0.15,
            chord=1.0,
        ),
        panels_per_side=panels_per_side,
    )


def step_input(
    *,
    freestream=(9.0, 0.0),
    kinematics=None,
    previous_bound=0.0,
    predicted_change=0.0,
    old_blobs=(),
    transpiration=None,
):
    return AttachedOuterStepInput2D(
        surface=naca0015(),
        kinematics=kinematics or RigidKinematics2D(),
        freestream_velocity_inertial=freestream,
        time_step=0.008,
        previous_bound_circulation_ccw=previous_bound,
        predicted_bound_circulation_change_ccw=predicted_change,
        old_blobs=old_blobs,
        wall_transpiration_velocity=transpiration,
    )


def empty_history(stage_time=0.0):
    return AttachedOuterHistory2D(
        bound_circulation_ccw=0.0,
        material_blobs=(),
        kelvin_reference_total_ccw=0.0,
        stage_time=stage_time,
    )


class SVIUnsteadyOuter2DTests(unittest.TestCase):
    def assert_equation_residuals(self, result):
        velocity_scale = max(
            float(
                np.linalg.norm(
                    result.inputs.freestream_velocity_inertial
                    - result.inputs.kinematics.translation_velocity_inertial
                )
            ),
            1.0,
        )
        circulation_scale = max(
            abs(result.kelvin_initial_total_ccw),
            abs(result.kelvin_final_total_ccw),
            velocity_scale * result.inputs.surface.section.chord,
        )
        self.assertLess(
            result.maximum_normal_boundary_residual,
            2.0e-11 * velocity_scale,
        )
        self.assertLess(
            abs(result.emission_residual),
            2.0e-11 * velocity_scale,
        )
        self.assertLess(
            abs(result.kelvin_residual),
            2.0e-11 * circulation_scale,
        )
        self.assertLess(
            float(np.max(np.abs(result.linear_system_residual), initial=0.0)),
            2.0e-11 * velocity_scale,
        )
        self.assertLess(
            abs(result.eq7_length_residual),
            5.0e-11 * result.near_wake_segment.length,
        )

    def test_stationary_symmetric_zero_angle_has_zero_vorticity(self):
        result = solve_attached_unsteady_outer_step(step_input(predicted_change=1.0))
        self.assert_equation_residuals(result)
        self.assertLess(abs(result.bound_circulation_ccw), 2.0e-12)
        self.assertLess(abs(result.newborn_circulation_ccw), 2.0e-12)
        self.assertLess(abs(result.newborn_sheet_strength_ccw), 2.0e-12)
        self.assertEqual(result.near_wake_segment.orientation_side, "lower")

    def test_static_kutta_state_is_exact_no_birth_limit_of_s1(self):
        """S1 must reproduce S0 when the Kelvin history is already steady."""
        speed = 9.0
        alpha_deg = 6.0
        static = solve_svi_dw_outer_foundation(
            SVIDWFoundationConfig(
                section=NACA4SectionConfig(
                    maximum_camber=0.0,
                    camber_location=0.4,
                    thickness_ratio=0.15,
                    chord=1.0,
                ),
                panels_per_side=32,
                freestream_speed=speed,
                angle_of_attack_deg=alpha_deg,
            )
        )
        # The S0 ``bound_circulation`` field uses the conventional
        # lift-positive aerodynamic sign.  S1's Kelvin ledger instead uses
        # physical CCW circulation, which is gamma_B times perimeter.
        previous_ccw = static.circulation_sheet_strength * static.surface.perimeter
        alpha = np.deg2rad(alpha_deg)

        for time_step in (0.1, 0.01, 0.001):
            with self.subTest(time_step=time_step):
                result = solve_attached_unsteady_outer_step(
                    AttachedOuterStepInput2D(
                        surface=static.surface,
                        kinematics=RigidKinematics2D(),
                        freestream_velocity_inertial=speed
                        * np.array((np.cos(alpha), np.sin(alpha))),
                        time_step=time_step,
                        previous_bound_circulation_ccw=previous_ccw,
                        predicted_bound_circulation_change_ccw=0.0,
                    )
                )
                self.assert_equation_residuals(result)
                self.assertAlmostEqual(
                    result.bound_circulation_ccw,
                    previous_ccw,
                    delta=2.0e-13,
                )
                self.assertLess(
                    abs(result.newborn_circulation_ccw),
                    5.0e-13,
                )
                np.testing.assert_allclose(
                    result.source_strength,
                    static.source_strength,
                    rtol=0.0,
                    atol=1.0e-12,
                )
                np.testing.assert_allclose(
                    result.relative_tangential_velocity,
                    static.tangential_velocity,
                    rtol=0.0,
                    atol=2.0e-12,
                )

    def test_moving_wall_and_prescribed_neumann_port_are_enforced(self):
        surface = naca0015()
        transpiration = 0.004 * np.sin(
            np.linspace(0.0, 2.0 * np.pi, surface.panel_count)
        )
        motion = RigidKinematics2D(
            pivot_body=(0.31, 0.015),
            pivot_inertial=(1.2, -0.7),
            angle_rad=0.23,
            translation_velocity_inertial=(1.1, -0.35),
            angular_velocity_rad_s=0.8,
        )
        inputs = AttachedOuterStepInput2D(
            surface=surface,
            kinematics=motion,
            freestream_velocity_inertial=(8.4, 0.6),
            time_step=0.006,
            previous_bound_circulation_ccw=-0.2,
            predicted_bound_circulation_change_ccw=0.05,
            wall_transpiration_velocity=transpiration,
        )
        result = solve_attached_unsteady_outer_step(inputs)
        self.assert_equation_residuals(result)
        reconstructed_normal = np.einsum(
            "ij,ij->i",
            result.relative_surface_velocity_body,
            surface.panel_outward_normals,
        )
        np.testing.assert_allclose(
            reconstructed_normal,
            transpiration,
            rtol=0.0,
            atol=2.0e-11,
        )
        self.assertGreater(float(np.linalg.norm(result.wall_velocity_body)), 0.0)

    def test_starting_vortex_closes_kelvin_and_emission_ledgers(self):
        alpha = np.deg2rad(6.0)
        result = solve_attached_unsteady_outer_step(
            step_input(
                freestream=9.0 * np.array((np.cos(alpha), np.sin(alpha))),
                previous_bound=0.0,
                predicted_change=-0.3,
            )
        )
        self.assert_equation_residuals(result)
        self.assertLess(result.bound_circulation_ccw, 0.0)
        self.assertGreater(result.newborn_circulation_ccw, 0.0)
        self.assertAlmostEqual(
            result.bound_circulation_ccw + result.newborn_circulation_ccw,
            0.0,
            delta=2.0e-12,
        )
        self.assertAlmostEqual(
            result.emission_jump_ccw,
            result.newborn_sheet_strength_ccw,
            delta=2.0e-11,
        )
        self.assertEqual(result.near_wake_segment.orientation_side, "lower")

    def test_wrong_predictor_orders_but_does_not_select_the_te_branch(self):
        alpha = np.deg2rad(6.0)
        result = solve_attached_unsteady_outer_step(
            step_input(
                freestream=9.0 * np.array((np.cos(alpha), np.sin(alpha))),
                # This predicts the upper branch, while the solved physical
                # bound-circulation change is negative/lower.
                predicted_change=0.3,
            )
        )
        self.assert_equation_residuals(result)
        self.assertEqual(result.near_wake_segment.orientation_side, "lower")
        self.assertLess(result.bound_circulation_ccw, 0.0)

    def test_ambiguous_and_zero_consistent_branch_sets_fail_closed(self):
        alpha = np.deg2rad(6.0)
        inputs = step_input(
            freestream=9.0 * np.array((np.cos(alpha), np.sin(alpha))),
            predicted_change=-0.3,
        )
        lower = _solve_orientation_branch(inputs, "lower")
        upper = _solve_orientation_branch(inputs, "upper")
        tolerance = min(lower.sign_tolerance, upper.sign_tolerance)
        ambiguous = {
            "lower": replace(
                lower,
                bound_circulation_change_ccw=-0.1,
                sign_tolerance=tolerance,
            ),
            "upper": replace(
                upper,
                bound_circulation_change_ccw=0.1,
                sign_tolerance=tolerance,
            ),
        }
        with self.assertRaisesRegex(SVIUnsteadyOuterConvergenceError, "both nonzero"):
            _select_physical_orientation_branch(ambiguous, {})

        no_consistent = {
            "lower": replace(
                lower,
                bound_circulation_change_ccw=0.1,
                sign_tolerance=tolerance,
            ),
            "upper": replace(
                upper,
                bound_circulation_change_ccw=-0.1,
                sign_tolerance=tolerance,
            ),
        }
        with self.assertRaisesRegex(SVIUnsteadyOuterConvergenceError, "zero nonzero"):
            _select_physical_orientation_branch(no_consistent, {})

    def test_common_te_diagnostic_exposes_both_adjacent_panel_traces(self):
        alpha = np.deg2rad(6.0)
        result = solve_attached_unsteady_outer_step(
            step_input(
                freestream=9.0 * np.array((np.cos(alpha), np.sin(alpha))),
                predicted_change=-0.3,
            )
        )
        diagnostic = result.common_te_diagnostic
        surface = result.inputs.surface
        self.assertEqual(diagnostic.lower_panel_index, 0)
        self.assertEqual(diagnostic.upper_panel_index, surface.panel_count - 1)
        np.testing.assert_array_equal(
            diagnostic.lower_panel_center_body,
            surface.panel_midpoints[0],
        )
        np.testing.assert_array_equal(
            diagnostic.upper_panel_center_body,
            surface.panel_midpoints[-1],
        )
        self.assertGreater(diagnostic.lower_downstream_trace, 0.0)
        self.assertGreater(diagnostic.upper_downstream_trace, 0.0)
        self.assertAlmostEqual(
            diagnostic.mean_downstream_trace,
            result.mean_emission_speed,
        )
        self.assertAlmostEqual(diagnostic.jump_ccw, result.emission_jump_ccw)
        self.assertAlmostEqual(
            diagnostic.lower_te_distance_over_panel_length,
            0.5,
            delta=2.0e-13,
        )
        self.assertAlmostEqual(
            diagnostic.upper_te_distance_over_panel_length,
            0.5,
            delta=2.0e-13,
        )

    def test_impulsive_start_respects_the_attached_trace_gate(self):
        """Eq. 8 is accepted only while both adjacent traces stay downstream."""
        alpha = np.deg2rad(6.0)
        previous_bound_magnitude = None
        for time_step in (1.0e-1, 1.0e-2):
            with self.subTest(time_step=time_step):
                inputs = step_input(
                    freestream=9.0 * np.array((np.cos(alpha), np.sin(alpha))),
                    predicted_change=-0.3,
                )
                inputs = AttachedOuterStepInput2D(
                    surface=inputs.surface,
                    kinematics=inputs.kinematics,
                    freestream_velocity_inertial=(inputs.freestream_velocity_inertial),
                    time_step=time_step,
                    previous_bound_circulation_ccw=0.0,
                    predicted_bound_circulation_change_ccw=-0.3,
                )
                result = solve_attached_unsteady_outer_step(inputs)
                self.assert_equation_residuals(result)
                self.assertEqual(
                    result.near_wake_segment.orientation_side,
                    "lower",
                )
                self.assertLess(result.bound_circulation_ccw, 0.0)
                self.assertGreater(result.newborn_circulation_ccw, 0.0)
                self.assertGreater(result.newborn_sheet_strength_ccw, 0.0)
                magnitude = abs(result.bound_circulation_ccw)
                if previous_bound_magnitude is not None:
                    self.assertLess(magnitude, previous_bound_magnitude)
                previous_bound_magnitude = magnitude

        # At fixed spatial resolution, an arbitrarily short newborn segment
        # can drive one adjacent-centre trace upstream.  S1 exposes that
        # space/time-resolution limit instead of accepting a positive mean.
        inputs = AttachedOuterStepInput2D(
            surface=naca0015(),
            kinematics=RigidKinematics2D(),
            freestream_velocity_inertial=9.0 * np.array((np.cos(alpha), np.sin(alpha))),
            time_step=1.0e-4,
            previous_bound_circulation_ccw=0.0,
            predicted_bound_circulation_change_ccw=-0.3,
        )
        with self.assertRaisesRegex(
            SVIUnsteadyOuterConvergenceError,
            "downstream-positive",
        ):
            solve_attached_unsteady_outer_step(inputs)

    def test_angle_reversal_reverses_ccw_circulation_and_orientation(self):
        alpha = np.deg2rad(6.0)
        positive = solve_attached_unsteady_outer_step(
            step_input(
                freestream=9.0 * np.array((np.cos(alpha), np.sin(alpha))),
                predicted_change=-0.3,
            )
        )
        negative = solve_attached_unsteady_outer_step(
            step_input(
                freestream=9.0 * np.array((np.cos(alpha), -np.sin(alpha))),
                predicted_change=0.3,
            )
        )
        self.assertEqual(positive.near_wake_segment.orientation_side, "lower")
        self.assertEqual(negative.near_wake_segment.orientation_side, "upper")
        self.assertAlmostEqual(
            positive.bound_circulation_ccw,
            -negative.bound_circulation_ccw,
            delta=3.0e-13,
        )
        self.assertAlmostEqual(
            positive.newborn_circulation_ccw,
            -negative.newborn_circulation_ccw,
            delta=3.0e-13,
        )
        self.assertAlmostEqual(
            positive.near_wake_segment.length,
            negative.near_wake_segment.length,
            delta=3.0e-15,
        )

    def test_galilean_boost_leaves_relative_solution_unchanged(self):
        motion = RigidKinematics2D(
            pivot_body=(0.25, 0.0),
            pivot_inertial=(0.7, -0.2),
            angle_rad=-0.12,
            translation_velocity_inertial=(0.8, -0.3),
            angular_velocity_rad_s=0.45,
        )
        base = step_input(
            freestream=(8.2, -0.1),
            kinematics=motion,
            previous_bound=0.11,
            predicted_change=-0.05,
        )
        boost = np.array((3.7, -2.1))
        boosted_motion = RigidKinematics2D(
            pivot_body=motion.pivot_body,
            pivot_inertial=motion.pivot_inertial,
            angle_rad=motion.angle_rad,
            translation_velocity_inertial=(
                motion.translation_velocity_inertial + boost
            ),
            angular_velocity_rad_s=motion.angular_velocity_rad_s,
        )
        boosted = step_input(
            freestream=base.freestream_velocity_inertial + boost,
            kinematics=boosted_motion,
            previous_bound=base.previous_bound_circulation_ccw,
            predicted_change=base.predicted_bound_circulation_change_ccw,
        )
        result = solve_attached_unsteady_outer_step(base)
        shifted = solve_attached_unsteady_outer_step(boosted)
        self.assert_equation_residuals(result)
        self.assert_equation_residuals(shifted)
        np.testing.assert_allclose(
            shifted.source_strength,
            result.source_strength,
            rtol=0.0,
            atol=3.0e-13,
        )
        np.testing.assert_allclose(
            shifted.relative_surface_velocity_body,
            result.relative_surface_velocity_body,
            rtol=0.0,
            atol=3.0e-13,
        )
        self.assertAlmostEqual(
            shifted.bound_circulation_ccw,
            result.bound_circulation_ccw,
            delta=3.0e-13,
        )
        self.assertAlmostEqual(
            shifted.newborn_circulation_ccw,
            result.newborn_circulation_ccw,
            delta=3.0e-13,
        )
        self.assertAlmostEqual(
            shifted.near_wake_segment.length,
            result.near_wake_segment.length,
            delta=3.0e-15,
        )

    def test_old_blobs_convect_but_circulation_and_core_are_immutable(self):
        blobs = (
            MaterialVortexBlob2D(
                position_inertial=(1.2, 0.2),
                circulation_ccw=0.17,
                core_radius=0.03,
            ),
            MaterialVortexBlob2D(
                position_inertial=(1.8, -0.1),
                circulation_ccw=-0.09,
                core_radius=0.05,
            ),
        )
        velocity = np.array(((2.0, -0.5), (1.1, 0.2)))
        advanced = convect_material_vortex_blobs(
            blobs, velocity_inertial=velocity, time_step=0.02
        )
        for index, (before, after) in enumerate(zip(blobs, advanced)):
            np.testing.assert_allclose(
                after.position_inertial,
                before.position_inertial + 0.02 * velocity[index],
                rtol=0.0,
                atol=0.0,
            )
            self.assertEqual(after.circulation_ccw, before.circulation_ccw)
            self.assertEqual(after.core_radius, before.core_radius)
        self.assertEqual(
            sum(item.circulation_ccw for item in advanced),
            sum(item.circulation_ccw for item in blobs),
        )

        result = solve_attached_unsteady_outer_step(
            step_input(
                old_blobs=blobs,
                previous_bound=-0.15,
                predicted_change=0.04,
            )
        )
        self.assert_equation_residuals(result)
        self.assertAlmostEqual(result.old_blob_circulation_ccw, 0.08)
        self.assertAlmostEqual(
            result.kelvin_initial_total_ccw,
            -0.15 + 0.08,
        )

    def test_positive_blob_circulation_is_counter_clockwise(self):
        blob = MaterialVortexBlob2D(
            position_inertial=(0.0, 0.0),
            circulation_ccw=0.4,
            core_radius=0.01,
        )
        velocity = material_blob_velocity_inertial(
            np.array(((1.0, 0.0), (0.0, 1.0))),
            (blob,),
        )
        self.assertGreater(velocity[0, 1], 0.0)
        self.assertLess(velocity[1, 0], 0.0)
        self.assertAlmostEqual(velocity[0, 0], 0.0)
        self.assertAlmostEqual(velocity[1, 1], 0.0)

    def test_positive_ccw_segment_has_lower_minus_upper_unit_jump(self):
        segment = TENearWakeSegment2D(
            start_body=(0.0, 0.0),
            end_body=(1.0, 0.0),
            orientation_side="lower",
            mean_emission_speed=1.0,
            time_step=1.0,
        )
        epsilon = 1.0e-8
        velocity = constant_vortex_segment_velocity_body(
            np.array(((0.5, epsilon), (0.5, -epsilon))),
            segment,
        )
        upper_trace = velocity[0, 0]
        lower_trace = velocity[1, 0]
        self.assertAlmostEqual(upper_trace, -0.5, delta=1.0e-7)
        self.assertAlmostEqual(lower_trace, 0.5, delta=1.0e-7)
        self.assertAlmostEqual(
            lower_trace - upper_trace,
            1.0,
            delta=2.0e-7,
        )

    def test_solved_segment_is_only_blob_birth_and_closes_next_step_kelvin(self):
        alpha = np.deg2rad(6.0)
        freestream = 9.0 * np.array((np.cos(alpha), np.sin(alpha)))
        first_march = march_attached_unsteady_outer_explicit_euler(
            surface=naca0015(),
            kinematics=RigidKinematics2D(),
            freestream_velocity_inertial=freestream,
            time_step=0.008,
            history=empty_history(),
            # Deliberately wrong: the march still finds the lower branch.
            predicted_bound_circulation_change_ccw=0.3,
            newborn_core_radius=0.02,
        )
        first = first_march.solution
        blob = completed_step_to_material_blob(first, core_radius=0.02)
        self.assertEqual(
            blob.circulation_ccw,
            first.newborn_circulation_ccw,
        )
        self.assertEqual(blob.core_radius, 0.02)
        np.testing.assert_allclose(
            blob.position_inertial,
            0.5
            * (first.near_wake_segment.start_body + first.near_wake_segment.end_body),
            rtol=0.0,
            atol=0.0,
        )
        born = first_march.history_at_stage_after_birth
        history = first_march.history_next
        self.assertAlmostEqual(
            born.kelvin_reference_total_ccw,
            first.kelvin_initial_total_ccw,
        )
        self.assertEqual(
            history.stage_time,
            first.inputs.stage_time + first.inputs.time_step,
        )
        np.testing.assert_allclose(
            history.material_blobs[0].position_inertial,
            born.material_blobs[0].position_inertial
            + first.inputs.time_step * first_march.advection.total_velocity_inertial[0],
            rtol=0.0,
            atol=2.0e-15,
        )
        second_march = march_attached_unsteady_outer_explicit_euler(
            surface=first.inputs.surface,
            kinematics=RigidKinematics2D(),
            freestream_velocity_inertial=freestream,
            time_step=first.inputs.time_step,
            history=history,
            predicted_bound_circulation_change_ccw=0.1,
            newborn_core_radius=0.02,
        )
        second = second_march.solution
        self.assert_equation_residuals(second)
        self.assertAlmostEqual(
            second.kelvin_initial_total_ccw,
            first.kelvin_final_total_ccw,
            delta=2.0e-12,
        )
        self.assertAlmostEqual(
            second.kelvin_final_total_ccw,
            first.kelvin_final_total_ccw,
            delta=2.0e-12,
        )
        self.assertAlmostEqual(
            second_march.history_next.kelvin_reference_total_ccw,
            first.kelvin_initial_total_ccw,
            delta=2.0e-12,
        )

        forged = MaterialVortexBlob2D(
            position_inertial=blob.position_inertial,
            circulation_ccw=-blob.circulation_ccw,
            core_radius=blob.core_radius,
        )
        with self.assertRaisesRegex(SVIDWValidationError, "Kelvin reference"):
            AttachedOuterHistory2D(
                bound_circulation_ccw=first.bound_circulation_ccw,
                material_blobs=(forged,),
                kelvin_reference_total_ccw=(first.kelvin_initial_total_ccw),
                stage_time=first.inputs.stage_time,
            )

        drifted = replace(
            first,
            kelvin_final_total_ccw=(first.kelvin_final_total_ccw + 1.0e-6),
        )
        with self.assertRaisesRegex(SVIDWValidationError, "cannot be absorbed"):
            complete_attached_outer_history(
                drifted,
                newborn_core_radius=0.02,
            )

    def test_caller_velocity_history_path_is_retired_fail_closed(self):
        with self.assertRaisesRegex(
            SVIUnsteadyOuterScopeError,
            "caller-supplied material velocities are forbidden",
        ):
            convect_attached_outer_history_explicit_euler(
                empty_history(),
                velocity_inertial=np.empty((0, 2)),
                time_step=0.01,
            )

    def test_march_excludes_blob_and_near_wake_self_induction(self):
        alpha = np.deg2rad(6.0)
        freestream = 9.0 * np.array((np.cos(alpha), np.sin(alpha)))
        common = dict(
            surface=naca0015(),
            kinematics=RigidKinematics2D(),
            freestream_velocity_inertial=freestream,
            time_step=0.008,
            history=empty_history(),
            predicted_bound_circulation_change_ccw=-0.3,
        )
        compact = march_attached_unsteady_outer_explicit_euler(
            **common, newborn_core_radius=0.005
        )
        diffuse = march_attached_unsteady_outer_explicit_euler(
            **common, newborn_core_radius=0.5
        )
        newborn = compact.advection.newborn_index
        np.testing.assert_array_equal(
            compact.advection.current_near_wake_velocity_inertial[newborn],
            np.zeros(2),
        )
        np.testing.assert_allclose(
            compact.advection.total_velocity_inertial[newborn],
            diffuse.advection.total_velocity_inertial[newborn],
            rtol=0.0,
            atol=0.0,
        )
        np.testing.assert_allclose(
            compact.history_next.material_blobs[newborn].position_inertial,
            diffuse.history_next.material_blobs[newborn].position_inertial,
            rtol=0.0,
            atol=0.0,
        )

        # On the second march the first old blob sees the current segment but
        # not its own Rosenhead field; the newborn sees the old blob but not
        # its own segment.
        second = march_attached_unsteady_outer_explicit_euler(
            surface=common["surface"],
            kinematics=RigidKinematics2D(),
            freestream_velocity_inertial=freestream,
            time_step=0.008,
            history=compact.history_next,
            predicted_bound_circulation_change_ccw=-0.1,
            newborn_core_radius=0.02,
        )
        np.testing.assert_array_equal(
            second.advection.other_material_blob_velocity_inertial[0],
            np.zeros(2),
        )
        self.assertGreater(
            float(
                np.linalg.norm(second.advection.current_near_wake_velocity_inertial[0])
            ),
            0.0,
        )
        np.testing.assert_array_equal(
            second.advection.current_near_wake_velocity_inertial[-1],
            np.zeros(2),
        )
        self.assertGreater(
            float(
                np.linalg.norm(
                    second.advection.other_material_blob_velocity_inertial[-1]
                )
            ),
            0.0,
        )

    def test_owned_march_is_galilean_covariant(self):
        alpha = np.deg2rad(5.0)
        freestream = 8.5 * np.array((np.cos(alpha), np.sin(alpha)))
        motion = RigidKinematics2D(
            pivot_body=(0.25, 0.0),
            pivot_inertial=(0.7, -0.2),
            angle_rad=-0.12,
            translation_velocity_inertial=(0.8, -0.3),
            angular_velocity_rad_s=0.45,
        )
        boost = np.array((3.7, -2.1))
        boosted_motion = RigidKinematics2D(
            pivot_body=motion.pivot_body,
            pivot_inertial=motion.pivot_inertial,
            angle_rad=motion.angle_rad,
            translation_velocity_inertial=(
                motion.translation_velocity_inertial + boost
            ),
            angular_velocity_rad_s=motion.angular_velocity_rad_s,
        )
        shared = dict(
            surface=naca0015(),
            time_step=0.008,
            history=empty_history(stage_time=0.3),
            predicted_bound_circulation_change_ccw=0.3,
            newborn_core_radius=0.02,
        )
        base = march_attached_unsteady_outer_explicit_euler(
            **shared,
            kinematics=motion,
            freestream_velocity_inertial=freestream,
        )
        shifted = march_attached_unsteady_outer_explicit_euler(
            **shared,
            kinematics=boosted_motion,
            freestream_velocity_inertial=freestream + boost,
        )
        np.testing.assert_allclose(
            shifted.advection.total_velocity_inertial,
            base.advection.total_velocity_inertial + boost,
            rtol=0.0,
            atol=2.0e-12,
        )
        for base_blob, shifted_blob in zip(
            base.history_next.material_blobs,
            shifted.history_next.material_blobs,
        ):
            np.testing.assert_allclose(
                shifted_blob.position_inertial,
                base_blob.position_inertial + shared["time_step"] * boost,
                rtol=0.0,
                atol=2.0e-14,
            )
            self.assertAlmostEqual(
                shifted_blob.circulation_ccw,
                base_blob.circulation_ccw,
                delta=2.0e-15,
            )

    def test_owned_march_time_step_refinement_smoke(self):
        surface = naca0015(panels_per_side=24)
        alpha = np.deg2rad(4.0)
        freestream = 9.0 * np.array((np.cos(alpha), np.sin(alpha)))
        terminal_bound = []
        for time_step in (0.02, 0.01, 0.005):
            history = empty_history()
            for _ in range(round(0.02 / time_step)):
                march = march_attached_unsteady_outer_explicit_euler(
                    surface=surface,
                    kinematics=RigidKinematics2D(),
                    freestream_velocity_inertial=freestream,
                    time_step=time_step,
                    history=history,
                    predicted_bound_circulation_change_ccw=0.1,
                    newborn_core_radius=0.02,
                )
                history = march.history_next
                self.assertAlmostEqual(
                    history.bound_circulation_ccw
                    + sum(blob.circulation_ccw for blob in history.material_blobs),
                    history.kelvin_reference_total_ccw,
                    delta=2.0e-12,
                )
            terminal_bound.append(history.bound_circulation_ccw)
        self.assertTrue(np.all(np.isfinite(terminal_bound)))
        self.assertLess(
            abs(terminal_bound[2] - terminal_bound[1]),
            abs(terminal_bound[1] - terminal_bound[0]),
        )

    def test_invalid_core_and_zero_length_segment_are_rejected(self):
        for invalid_core in (0.0, -0.1, np.nan):
            with self.subTest(core=invalid_core):
                with self.assertRaises(SVIDWValidationError):
                    MaterialVortexBlob2D(
                        position_inertial=(1.0, 0.0),
                        circulation_ccw=0.1,
                        core_radius=invalid_core,
                    )
        with self.assertRaises(SVIDWValidationError):
            TENearWakeSegment2D(
                start_body=(1.0, 0.0),
                end_body=(1.0, 0.0),
                orientation_side="upper",
                mean_emission_speed=1.0,
                time_step=0.01,
            )
        with self.assertRaises(SVIDWValidationError):
            TENearWakeSegment2D(
                start_body=(1.0, 0.0),
                end_body=(1.1, 0.0),
                orientation_side="upper",
                mean_emission_speed=1.0,
                time_step=0.01,
            )


if __name__ == "__main__":
    unittest.main()
