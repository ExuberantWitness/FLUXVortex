import sys
import unittest
from pathlib import Path

import numpy as np

PLATFORM = Path(__file__).resolve().parents[1]
if str(PLATFORM) not in sys.path:
    sys.path.insert(0, str(PLATFORM))

from claim_runtime.deforming_shell_kinematics import (  # noqa: E402
    DeformingShellError,
    deforming_dual_surface_kinematics,
    dual_surface_snapshot,
    rigidly_transform_surface_geometry,
    structured_surface_geometry,
)


def curved_patch():
    xi = np.linspace(0.0, 1.0, 17)
    eta = np.linspace(-0.6, 0.6, 15)
    x, y = np.meshgrid(xi, eta, indexing="ij")
    points = np.stack(
        (x, y, 0.08*x*x+0.03*x*y-0.02*y*y),
        axis=2,
    )
    return xi, eta, points


class DeformingShellKinematicsTests(unittest.TestCase):
    def test_dual_side_pairing_and_thickness(self):
        xi, eta, points = curved_patch()
        geometry = structured_surface_geometry(points, xi=xi, eta=eta)
        thickness = 0.01+0.002*np.outer(xi, np.ones_like(eta))
        shell = dual_surface_snapshot(
            geometry,
            half_thickness=thickness,
        )
        np.testing.assert_allclose(
            0.5*(shell.upper_surface+shell.lower_surface),
            points,
            atol=2.0e-16,
        )
        np.testing.assert_allclose(
            0.5*np.linalg.norm(
                shell.upper_surface-shell.lower_surface,
                axis=2,
            ),
            thickness,
            atol=2.0e-16,
        )
        np.testing.assert_allclose(
            np.linalg.norm(shell.director, axis=2),
            1.0,
            atol=2.0e-16,
        )

    def test_proper_rigid_transform_preserves_intrinsic_geometry(self):
        xi, eta, points = curved_patch()
        reference = structured_surface_geometry(points, xi=xi, eta=eta)
        axis = np.array([0.4, -0.7, 1.1])
        axis /= np.linalg.norm(axis)
        angle = 0.83
        cross = np.array([
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ])
        rotation = (
            np.eye(3)*np.cos(angle)
            +(1.0-np.cos(angle))*np.outer(axis, axis)
            +np.sin(angle)*cross
        )
        moved = rigidly_transform_surface_geometry(
            reference,
            rotation=rotation,
            translation=[0.3, -0.2, 0.7],
            xi=xi,
            eta=eta,
        )
        np.testing.assert_allclose(
            moved.first_fundamental_form,
            reference.first_fundamental_form,
            atol=2.0e-14,
        )
        np.testing.assert_allclose(
            moved.area_jacobian,
            reference.area_jacobian,
            atol=2.0e-14,
        )
        np.testing.assert_allclose(
            moved.mean_curvature,
            reference.mean_curvature,
            atol=2.0e-12,
        )
        np.testing.assert_allclose(
            moved.gaussian_curvature,
            reference.gaussian_curvature,
            atol=2.0e-12,
        )
        np.testing.assert_allclose(
            moved.director,
            np.einsum("ij,...j->...i", rotation, reference.director),
            atol=2.0e-14,
        )

    def test_constant_translation_velocity_and_acceleration(self):
        xi, eta, points = curved_patch()
        dt = 0.125
        velocity = np.array([0.5, -1.0, 0.25])
        thickness = np.full(points.shape[:2], 0.012)
        result = deforming_dual_surface_kinematics(
            previous_mean_surface=points-dt*velocity,
            current_mean_surface=points,
            next_mean_surface=points+dt*velocity,
            xi=xi,
            eta=eta,
            half_thickness=thickness,
            dt=dt,
        )
        expected = np.broadcast_to(velocity, points.shape)
        for actual in (
            result.mean_velocity,
            result.upper_velocity,
            result.lower_velocity,
        ):
            np.testing.assert_allclose(actual, expected, atol=1.0e-12)
        for actual in (
            result.mean_acceleration,
            result.upper_acceleration,
            result.lower_acceleration,
        ):
            np.testing.assert_allclose(actual, 0.0, atol=1.0e-12)

    def test_cylindrical_bending_recovers_area_and_curvature(self):
        radius = 0.8
        xi = np.linspace(-0.55, 0.55, 101)
        eta = np.linspace(-0.4, 0.4, 17)
        theta, span = np.meshgrid(xi, eta, indexing="ij")
        points = np.stack(
            (
                radius*np.sin(theta),
                span,
                radius*(1.0-np.cos(theta)),
            ),
            axis=2,
        )
        geometry = structured_surface_geometry(points, xi=xi, eta=eta)
        interior = np.s_[2:-2, 2:-2]
        np.testing.assert_allclose(
            geometry.area_jacobian[interior],
            radius,
            atol=5.0e-5,
        )
        np.testing.assert_allclose(
            np.abs(geometry.mean_curvature[interior]),
            0.5/radius,
            atol=8.0e-5,
        )
        np.testing.assert_allclose(
            geometry.gaussian_curvature[interior],
            0.0,
            atol=2.0e-12,
        )

    def test_degenerate_and_reversed_material_orientation_fail(self):
        xi, eta, points = curved_patch()
        reference = structured_surface_geometry(points, xi=xi, eta=eta)
        collapsed = points.copy()
        collapsed[8, :, :] = collapsed[7, :, :]
        collapsed[9, :, :] = collapsed[7, :, :]
        with self.assertRaises(DeformingShellError):
            structured_surface_geometry(
                collapsed,
                xi=xi,
                eta=eta,
            )
        reversed_points = points[:, ::-1, :]
        with self.assertRaises(DeformingShellError):
            structured_surface_geometry(
                reversed_points,
                xi=xi,
                eta=eta,
                reference_director=reference.director,
            )


if __name__ == "__main__":
    unittest.main()
