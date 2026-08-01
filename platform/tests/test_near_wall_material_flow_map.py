import sys
import unittest
from pathlib import Path

import numpy as np

PLATFORM = Path(__file__).resolve().parents[1]
if str(PLATFORM) not in sys.path:
    sys.path.insert(0, str(PLATFORM))

from claim_runtime.material_spike_geometry import (  # noqa: E402
    material_curvature_change,
)
from claim_runtime.near_wall_material_flow_map import (  # noqa: E402
    NearWallFlowMapError,
    integrate_near_wall_material_flow_map,
)
from tests.test_material_spike_geometry import plane  # noqa: E402


def rotation_z(angle):
    return np.array([
        [np.cos(angle), -np.sin(angle), 0.0],
        [np.sin(angle), np.cos(angle), 0.0],
        [0.0, 0.0, 1.0],
    ])


class NearWallMaterialFlowMapTests(unittest.TestCase):
    def test_uniform_translation_is_exact_and_labels_are_preserved(self):
        u, v, initial = plane(nu=13, nv=9)
        velocity = np.array([0.4, -0.2, 0.1])
        result = integrate_near_wall_material_flow_map(
            initial,
            initial_time=0.2,
            final_time=0.7,
            maximum_step=0.07,
            velocity_field=lambda points, time: np.broadcast_to(
                velocity,
                points.shape,
            ),
        )
        expected = initial+0.5*velocity
        np.testing.assert_allclose(
            result.final_position,
            expected,
            atol=3.0e-16,
        )
        self.assertEqual(result.initial_shape, initial.shape)
        self.assertEqual(result.position.shape[1:], initial.shape)
        np.testing.assert_array_equal(result.initial_position, initial)
        np.testing.assert_allclose(result.time[[0, -1]], [0.2, 0.7])
        self.assertLessEqual(result.step_size, 0.07)

    def test_rigid_rotation_has_fourth_order_convergence(self):
        _, _, initial = plane(nu=13, nv=9)
        angular_speed = 0.9
        duration = 0.8
        exact = initial@rotation_z(angular_speed*duration).T

        def velocity(points, time):
            del time
            result = np.empty_like(points)
            result[..., 0] = -angular_speed*points[..., 1]
            result[..., 1] = angular_speed*points[..., 0]
            result[..., 2] = 0.0
            return result

        errors = []
        for steps in (2, 4, 8):
            result = integrate_near_wall_material_flow_map(
                initial,
                initial_time=0.0,
                final_time=duration,
                maximum_step=duration/steps,
                velocity_field=velocity,
            )
            errors.append(float(np.max(np.linalg.norm(
                result.final_position-exact,
                axis=-1,
            ))))
        self.assertGreater(errors[0]/errors[1], 14.0)
        self.assertGreater(errors[1]/errors[2], 14.0)

    def test_time_dependent_observer_of_stationary_flow(self):
        _, _, initial = plane(nu=13, nv=9)
        angular_speed = 0.8
        final_time = 0.6

        def translation(time):
            return np.array([
                0.1*time*time,
                -0.2*np.sin(time),
                0.05*time,
            ])

        def translation_rate(time):
            return np.array([
                0.2*time,
                -0.2*np.cos(time),
                0.05,
            ])

        spin = np.array([
            [0.0, -angular_speed, 0.0],
            [angular_speed, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ])

        def transformed_velocity(points, time):
            offset = translation(time)
            return (points-offset)@spin.T+translation_rate(time)

        result = integrate_near_wall_material_flow_map(
            initial+translation(0.0),
            initial_time=0.0,
            final_time=final_time,
            maximum_step=0.005,
            velocity_field=transformed_velocity,
        )
        exact = (
            initial@rotation_z(angular_speed*final_time).T
            +translation(final_time)
        )
        np.testing.assert_allclose(
            result.final_position,
            exact,
            atol=2.0e-11,
        )

    def test_no_slip_layers_follow_exact_nonlinear_fold(self):
        u, v, wall = plane(nu=31, nv=15)
        layers = []
        for height in (0.0, 0.01, 0.025):
            layer = wall.copy()
            layer[..., 2] = height
            layers.append(layer)
        initial = np.stack(layers)
        fold_rate = 1.7
        final_time = 0.4

        def velocity(points, time):
            del time
            result = np.zeros_like(points)
            result[..., 2] = (
                fold_rate*points[..., 0]**2*points[..., 2]
            )
            return result

        flow_map = integrate_near_wall_material_flow_map(
            initial,
            initial_time=0.0,
            final_time=final_time,
            maximum_step=0.01,
            velocity_field=velocity,
        )
        exact = initial.copy()
        exact[..., 2] = (
            initial[..., 2]
            *np.exp(fold_rate*initial[..., 0]**2*final_time)
        )
        np.testing.assert_allclose(
            flow_map.final_position,
            exact,
            atol=2.0e-13,
        )
        np.testing.assert_array_equal(
            flow_map.final_position[0],
            initial[0],
        )

        numerical_change = material_curvature_change(
            initial[2],
            flow_map.final_position[2],
            u=u,
            v=v,
            initial_time=0.0,
            final_time=final_time,
        )
        exact_change = material_curvature_change(
            initial[2],
            exact[2],
            u=u,
            v=v,
            initial_time=0.0,
            final_time=final_time,
        )
        np.testing.assert_allclose(
            numerical_change.largest_principal_curvature_change,
            exact_change.largest_principal_curvature_change,
            atol=3.0e-11,
        )

    def test_invalid_velocity_and_interval_fail(self):
        _, _, initial = plane(nu=7, nv=5)
        with self.assertRaisesRegex(
            NearWallFlowMapError,
            "material-position shape",
        ):
            integrate_near_wall_material_flow_map(
                initial,
                initial_time=0.0,
                final_time=0.1,
                maximum_step=0.01,
                velocity_field=lambda points, time: np.zeros((2, 3)),
            )
        with self.assertRaisesRegex(
            NearWallFlowMapError,
            "non-finite",
        ):
            integrate_near_wall_material_flow_map(
                initial,
                initial_time=0.0,
                final_time=0.1,
                maximum_step=0.01,
                velocity_field=lambda points, time: (
                    np.full_like(points, np.nan)
                ),
            )
        with self.assertRaisesRegex(
            NearWallFlowMapError,
            "greater than initial_time",
        ):
            integrate_near_wall_material_flow_map(
                initial,
                initial_time=0.1,
                final_time=0.1,
                maximum_step=0.01,
                velocity_field=lambda points, time: np.zeros_like(points),
            )


if __name__ == "__main__":
    unittest.main()
