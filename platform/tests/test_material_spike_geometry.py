import sys
import unittest
from pathlib import Path

import numpy as np

PLATFORM = Path(__file__).resolve().parents[1]
if str(PLATFORM) not in sys.path:
    sys.path.insert(0, str(PLATFORM))

from claim_runtime.material_spike_geometry import (  # noqa: E402
    MaterialSpikeGeometryError,
    material_curvature_change,
    material_surface_curvature,
    proper_euclidean_observer_transform,
)


def rotation_matrix(axis, angle):
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    cross = np.array([
        [0.0, -axis[2], axis[1]],
        [axis[2], 0.0, -axis[0]],
        [-axis[1], axis[0], 0.0],
    ])
    return (
        np.eye(3)*np.cos(angle)
        +(1.0-np.cos(angle))*np.outer(axis, axis)
        +np.sin(angle)*cross
    )


def plane(nu=31, nv=17):
    u = np.linspace(-0.7, 0.7, nu)
    v = np.linspace(-0.4, 0.4, nv)
    uu, vv = np.meshgrid(u, v, indexing="ij")
    points = np.stack((uu, vv, np.zeros_like(uu)), axis=-1)
    return u, v, points


def cylinder(radius, nu):
    u = np.linspace(-0.65, 0.65, nu)
    v = np.linspace(-0.4, 0.4, 17)
    uu, vv = np.meshgrid(u, v, indexing="ij")
    points = np.stack((
        radius*np.sin(uu),
        vv,
        radius*(1.0-np.cos(uu)),
    ), axis=-1)
    return u, v, points


class MaterialSpikeGeometryTests(unittest.TestCase):
    def test_plane_has_zero_weingarten_map(self):
        u, v, points = plane()
        result = material_surface_curvature(points, u=u, v=v)
        expected_metric = np.broadcast_to(
            np.eye(2),
            result.first_fundamental_form.shape,
        )
        np.testing.assert_allclose(
            result.first_fundamental_form,
            expected_metric,
            atol=2.0e-14,
        )
        np.testing.assert_allclose(
            result.second_fundamental_form,
            0.0,
            atol=2.0e-14,
        )
        np.testing.assert_allclose(
            result.principal_curvatures,
            0.0,
            atol=2.0e-14,
        )

    def test_cylinder_curvature_converges_under_refinement(self):
        radius = 0.8
        errors = []
        for count in (31, 121):
            u, v, points = cylinder(radius, count)
            result = material_surface_curvature(points, u=u, v=v)
            errors.append(float(np.max(np.abs(
                result.principal_curvatures
                -np.array([0.0, 1.0/radius])
            ))))
            np.testing.assert_allclose(
                result.gaussian_curvature[2:-2, 2:-2],
                0.0,
                atol=2.0e-12,
            )
        self.assertLess(errors[1], 0.08*errors[0])
        self.assertLess(errors[1], 2.0e-4)

    def test_static_surface_has_zero_curvature_change(self):
        u, v, points = cylinder(0.8, 51)
        change = material_curvature_change(
            points,
            points.copy(),
            u=u,
            v=v,
            initial_time=0.0,
            final_time=0.25,
        )
        np.testing.assert_allclose(
            change.weingarten_change,
            0.0,
            atol=0.0,
        )
        np.testing.assert_allclose(
            change.principal_curvature_changes,
            0.0,
            atol=0.0,
        )

    def test_parabolic_fold_recovers_positive_origin_change(self):
        u, v, initial = plane(nu=41, nv=21)
        uu, _ = np.meshgrid(u, v, indexing="ij")
        rate = 2.4
        final = initial.copy()
        final[..., 2] = 0.5*rate*uu*uu
        change = material_curvature_change(
            initial,
            final,
            u=u,
            v=v,
            initial_time=0.0,
            final_time=0.1,
        )
        centre = (len(u)//2, len(v)//2)
        np.testing.assert_allclose(
            change.principal_curvature_changes[centre],
            [0.0, rate],
            atol=2.0e-12,
        )
        self.assertGreater(
            change.largest_principal_curvature_change[centre],
            0.0,
        )

    def test_time_dependent_proper_observer_change_is_objective(self):
        u, v, initial = plane(nu=41, nv=21)
        uu, _ = np.meshgrid(u, v, indexing="ij")
        final = initial.copy()
        final[..., 2] = 0.9*uu*uu
        reference = material_curvature_change(
            initial,
            final,
            u=u,
            v=v,
            initial_time=0.0,
            final_time=0.2,
        )
        moved_initial = proper_euclidean_observer_transform(
            initial,
            rotation=rotation_matrix([0.3, -0.2, 0.9], 0.51),
            translation=[0.4, -0.7, 0.2],
        )
        moved_final = proper_euclidean_observer_transform(
            final,
            rotation=rotation_matrix([-0.4, 0.8, 0.1], -0.73),
            translation=[-0.3, 0.1, 0.9],
        )
        moved = material_curvature_change(
            moved_initial,
            moved_final,
            u=u,
            v=v,
            initial_time=0.0,
            final_time=0.2,
        )
        np.testing.assert_allclose(
            moved.principal_curvature_changes,
            reference.principal_curvature_changes,
            atol=2.0e-12,
        )
        np.testing.assert_allclose(
            moved.mean_curvature_change,
            reference.mean_curvature_change,
            atol=2.0e-12,
        )
        np.testing.assert_allclose(
            moved.gaussian_curvature_change,
            reference.gaussian_curvature_change,
            atol=2.0e-12,
        )

    def test_linear_material_reparametrization_preserves_curvature(self):
        u, v, points = cylinder(0.8, 81)
        baseline = material_surface_curvature(points, u=u, v=v)
        reparameterized = material_surface_curvature(
            points,
            u=3.7*u+0.2,
            v=0.6*v-0.4,
        )
        np.testing.assert_allclose(
            reparameterized.principal_curvatures[2:-2, 2:-2],
            baseline.principal_curvatures[2:-2, 2:-2],
            atol=2.0e-11,
        )

    def test_invalid_observer_and_material_orientation_fail(self):
        u, v, points = plane()
        with self.assertRaisesRegex(
            MaterialSpikeGeometryError,
            "proper orthogonal",
        ):
            proper_euclidean_observer_transform(
                points,
                rotation=np.diag([1.0, 1.0, -1.0]),
                translation=[0.0, 0.0, 0.0],
            )
        reference = np.zeros_like(points)
        reference[..., 2] = 1.0
        with self.assertRaisesRegex(
            MaterialSpikeGeometryError,
            "orientation is reversed",
        ):
            material_surface_curvature(
                points[:, ::-1],
                u=u,
                v=v,
                reference_normal=reference,
            )

    def test_degenerate_surface_and_time_order_fail(self):
        u, v, points = plane()
        collapsed = points.copy()
        collapsed[14:18] = collapsed[14]
        with self.assertRaisesRegex(
            MaterialSpikeGeometryError,
            "degenerate",
        ):
            material_surface_curvature(collapsed, u=u, v=v)
        with self.assertRaisesRegex(
            MaterialSpikeGeometryError,
            "greater than initial_time",
        ):
            material_curvature_change(
                points,
                points,
                u=u,
                v=v,
                initial_time=1.0,
                final_time=1.0,
            )


if __name__ == "__main__":
    unittest.main()
