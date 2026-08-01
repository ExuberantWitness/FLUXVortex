import sys
import unittest
from pathlib import Path

import numpy as np

PLATFORM = Path(__file__).resolve().parents[1]
if str(PLATFORM) not in sys.path:
    sys.path.insert(0, str(PLATFORM))

from claim_runtime.distributed_doublet import (  # noqa: E402
    DistributedDoubletError,
    MaterialWakeHistory,
    QuadraticDoubletAssembly,
    QuadraticDoubletElement,
    QuadraticDoubletPatch,
    QuadraticDoubletSurface,
    newborn_material_wake_band,
)


def polynomial(points):
    x = points[:, 0]
    y = points[:, 1]
    return 0.7 + 1.1 * x - 0.4 * y + 0.3 * x**2 + 0.2 * x * y - 0.6 * y**2


def polynomial_gradient(points):
    x = points[:, 0]
    y = points[:, 1]
    return np.column_stack(
        (
            1.1 + 0.6 * x + 0.2 * y,
            -0.4 + 0.2 * x - 1.2 * y,
            np.zeros_like(x),
        )
    )


class QuadraticElementTests(unittest.TestCase):
    def setUp(self):
        self.vertices = np.array(
            [
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [0.2, 1.5, 0.0],
            ]
        )
        nodes = np.vstack(
            (
                self.vertices,
                0.5 * (self.vertices[0] + self.vertices[1]),
                0.5 * (self.vertices[1] + self.vertices[2]),
                0.5 * (self.vertices[2] + self.vertices[0]),
            )
        )
        self.element = QuadraticDoubletElement(
            self.vertices,
            polynomial(nodes),
        )

    def test_p2_reproduces_quadratic_value_and_gradient(self):
        bary = np.array(
            [
                [0.2, 0.3, 0.5],
                [0.6, 0.1, 0.3],
                [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
            ]
        )
        points = bary @ self.vertices
        np.testing.assert_allclose(
            self.element.evaluate(points),
            polynomial(points),
            atol=2e-15,
        )
        np.testing.assert_allclose(
            self.element.surface_gradient(points),
            polynomial_gradient(points),
            atol=2e-15,
        )

    def test_dve_span_polynomial_gives_streamwise_linear_sheet(self):
        vertices = np.array(
            [
                [0.0, -1.0, 0.0],
                [1.0, -1.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        )
        nodes = np.vstack(
            (
                vertices,
                0.5 * (vertices[0] + vertices[1]),
                0.5 * (vertices[1] + vertices[2]),
                0.5 * (vertices[2] + vertices[0]),
            )
        )
        a, b, c = 0.2, -0.7, 0.35
        mu = a + b * nodes[:, 1] + c * nodes[:, 1] ** 2
        element = QuadraticDoubletElement(vertices, mu)
        bary = np.array([[0.2, 0.3, 0.5], [0.6, 0.2, 0.2]])
        points = bary @ vertices
        expected = np.column_stack(
            (
                b + 2.0 * c * points[:, 1],
                np.zeros(len(points)),
                np.zeros(len(points)),
            )
        )
        np.testing.assert_allclose(
            element.sheet_vorticity_barycentric(bary),
            expected,
            atol=2e-15,
        )

    def test_off_plane_and_degenerate_geometry_fail(self):
        with self.assertRaises(DistributedDoubletError):
            self.element.evaluate(np.array([[0.2, 0.2, 1e-3]]))
        with self.assertRaises(DistributedDoubletError):
            QuadraticDoubletElement(
                np.array([[0.0, 0.0, 0.0]] * 3),
                np.zeros(6),
            )


class QuadraticSurfaceTests(unittest.TestCase):
    def setUp(self):
        self.vertices = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        )
        self.faces = np.array([[0, 1, 2], [0, 2, 3]])
        self.face_mu = []
        for face in self.faces:
            triangle = self.vertices[face]
            nodes = np.vstack(
                (
                    triangle,
                    0.5 * (triangle[0] + triangle[1]),
                    0.5 * (triangle[1] + triangle[2]),
                    0.5 * (triangle[2] + triangle[0]),
                )
            )
            self.face_mu.append(polynomial(nodes))
        self.surface = QuadraticDoubletSurface(
            self.vertices,
            self.faces,
            np.asarray(self.face_mu),
        )

    def test_shared_edge_trace_is_exactly_continuous(self):
        report = self.surface.continuity_report()
        self.assertTrue(report.compatible)
        self.assertEqual(report.internal_edges, 1)
        self.assertEqual(report.boundary_edges, 4)
        self.assertLess(report.max_trace_jump, 2e-15)

    def test_midpoint_vorticity_constraint_is_independent_of_trace(self):
        broken_gradient = np.asarray(self.face_mu).copy()
        # Face 1 local vertex 2 is opposite the shared diagonal.  Changing it
        # leaves all three shared-edge trace DOFs untouched but changes the
        # cross-edge derivative at the midpoint.
        broken_gradient[1, 2] += 0.1
        surface = QuadraticDoubletSurface(
            self.vertices,
            self.faces,
            broken_gradient,
        )
        trace = surface.continuity_report()
        vorticity = surface.coplanar_midpoint_vorticity_report()
        self.assertTrue(trace.compatible)
        self.assertTrue(vorticity.coplanar)
        self.assertFalse(vorticity.compatible)
        self.assertGreater(vorticity.max_midpoint_vorticity_jump, 0.0)

    def test_discontinuous_internal_midpoint_is_detected(self):
        broken = np.asarray(self.face_mu).copy()
        # Shared edge is face-0 e20 and face-1 e01.
        broken[1, 3] += 0.01
        report = QuadraticDoubletSurface(
            self.vertices,
            self.faces,
            broken,
        ).continuity_report()
        self.assertFalse(report.compatible)
        self.assertAlmostEqual(report.max_trace_node_jump, 0.01)

    def test_isolated_nonzero_boundary_is_rejected(self):
        report = self.surface.boundary_report()
        self.assertFalse(report.compatible)
        self.assertGreater(report.max_boundary_trace, 0.0)
        zero = QuadraticDoubletSurface(
            self.vertices,
            self.faces,
            np.zeros((2, 6)),
        ).boundary_report()
        self.assertTrue(zero.compatible)

    def test_global_assembly_accepts_zero_edges_and_continuous_seam(self):
        upper_vertices = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        )
        lower_vertices = np.array(
            [
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        )
        # The shared diagonal has trace (0, 0.4, 0); every physical outer
        # edge has a zero P2 trace.
        upper = QuadraticDoubletSurface(
            upper_vertices,
            [[0, 1, 2]],
            [[0.0, 0.0, 0.0, 0.0, 0.4, 0.0]],
        )
        lower = QuadraticDoubletSurface(
            lower_vertices,
            [[0, 1, 2]],
            [[0.0, 0.0, 0.0, 0.0, 0.0, 0.4]],
        )
        patch_a = QuadraticDoubletPatch(
            "bound",
            upper,
            {
                (0, 1): "zero",
                (0, 2): "zero",
                (1, 2): "interface:bound-wake",
            },
        )
        patch_b = QuadraticDoubletPatch(
            "wake",
            lower,
            {
                (0, 1): "zero",
                (1, 2): "zero",
                (0, 2): "interface:bound-wake",
            },
        )
        report = QuadraticDoubletAssembly(
            [patch_a, patch_b]
        ).topology_report()
        self.assertTrue(report.compatible)
        self.assertEqual(report.zero_boundaries, 4)
        self.assertEqual(report.coupled_interfaces, 1)
        self.assertEqual(report.max_zero_trace, 0.0)
        self.assertEqual(report.max_interface_trace_jump, 0.0)

    def test_interface_matching_is_independent_of_local_edge_direction(self):
        surface = QuadraticDoubletSurface(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0]],
            [[0, 1, 2]],
            [[0, 0, 0, 0, 0.4, 0]],
        )
        roles = {
            (0, 1): "zero",
            (0, 2): "zero",
            (1, 2): "interface:same-order",
        }
        report = QuadraticDoubletAssembly(
            [
                QuadraticDoubletPatch("first", surface, roles),
                QuadraticDoubletPatch("second", surface, roles),
            ]
        ).topology_report()
        self.assertTrue(report.compatible)
        self.assertEqual(report.max_interface_geometry_gap, 0.0)
        self.assertEqual(report.max_interface_trace_jump, 0.0)

    def test_global_assembly_rejects_seam_strength_or_geometry_gap(self):
        first = QuadraticDoubletSurface(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0]],
            [[0, 1, 2]],
            [[0, 0, 0, 0, 0.4, 0]],
        )
        second = QuadraticDoubletSurface(
            [[1, 0, 0], [1, 1, 0], [0, 1.01, 0]],
            [[0, 1, 2]],
            [[0, 0, 0, 0, 0, 0.3]],
        )
        report = QuadraticDoubletAssembly(
            [
                QuadraticDoubletPatch(
                    "a",
                    first,
                    {
                        (0, 1): "zero",
                        (0, 2): "zero",
                        (1, 2): "interface:seam",
                    },
                ),
                QuadraticDoubletPatch(
                    "b",
                    second,
                    {
                        (0, 1): "zero",
                        (1, 2): "zero",
                        (0, 2): "interface:seam",
                    },
                ),
            ]
        ).topology_report()
        self.assertFalse(report.compatible)
        self.assertGreater(report.max_interface_trace_jump, 0.0)
        self.assertGreater(report.max_interface_geometry_gap, 0.0)

    def test_assembled_sheet_average_equals_single_continuous_surface(self):
        upper_vertices = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        )
        lower_vertices = np.array(
            [
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        )
        upper_mu = np.array([[0.0, 0.0, 0.0, 0.0, 0.4, 0.0]])
        lower_mu = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.4]])
        upper = QuadraticDoubletSurface(
            upper_vertices, [[0, 1, 2]], upper_mu
        )
        lower = QuadraticDoubletSurface(
            lower_vertices, [[0, 1, 2]], lower_mu
        )
        assembly = QuadraticDoubletAssembly(
            [
                QuadraticDoubletPatch(
                    "bound",
                    upper,
                    {
                        (0, 1): "zero",
                        (0, 2): "zero",
                        (1, 2): "interface:bound-wake",
                    },
                ),
                QuadraticDoubletPatch(
                    "wake",
                    lower,
                    {
                        (0, 1): "zero",
                        (1, 2): "zero",
                        (0, 2): "interface:bound-wake",
                    },
                ),
            ]
        )
        points, patch_owner, face_owner, barycentric = (
            assembly.interior_collocation_points()
        )
        actual, report = assembly.induced_velocity_sheet_average_converged(
            patch_owner,
            face_owner,
            barycentric,
            orders=(48, 80, 128, 192, 256),
            absolute_tolerance=2.0e-10,
            relative_tolerance=2.0e-9,
        )
        self.assertTrue(report.converged, report)

        merged = QuadraticDoubletSurface(
            np.array(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [1.0, 1.0, 0.0],
                    [0.0, 1.0, 0.0],
                ]
            ),
            [[0, 1, 3], [1, 2, 3]],
            np.vstack((upper_mu, lower_mu)),
        )
        merged_owner = patch_owner.copy()
        expected = merged.induced_velocity_sheet_average(
            merged_owner,
            barycentric,
            quadrature_order=report.quadrature_order,
        )
        np.testing.assert_allclose(actual, expected, rtol=2e-11, atol=2e-12)
        expected_points = np.empty_like(points)
        for point_index, owner in enumerate(merged_owner):
            expected_points[point_index] = (
                barycentric[point_index]
                @ merged.vertices[merged.faces[owner]]
            )
        np.testing.assert_allclose(points, expected_points, atol=0.0)

    def test_every_patch_boundary_requires_a_declared_role(self):
        with self.assertRaisesRegex(
            DistributedDoubletError,
            "boundary classification mismatch",
        ):
            QuadraticDoubletPatch(
                "unclassified",
                self.surface,
                {},
            )

    def test_affine_material_motion_preserves_kelvin_strengths(self):
        transform = np.array(
            [
                [1.3, 0.2, 0.0],
                [-0.1, 0.8, 0.15],
                [0.05, 0.0, 1.1],
            ]
        )
        moved_vertices = self.vertices @ transform.T + np.array([0.2, -0.3, 0.4])
        moved = self.surface.material_update(moved_vertices)
        kelvin = self.surface.kelvin_report(moved)
        self.assertTrue(kelvin.passed)
        self.assertEqual(kelvin.max_material_mu_residual, 0.0)
        self.assertTrue(moved.continuity_report().compatible)

    def test_nonmanifold_edge_fails_explicitly(self):
        vertices = np.vstack((self.vertices, [[0.5, -0.5, 0.0]]))
        faces = np.array([[0, 1, 2], [1, 0, 3], [0, 1, 4]])
        mu = np.zeros((3, 6))
        surface = QuadraticDoubletSurface(vertices, faces, mu)
        with self.assertRaises(DistributedDoubletError):
            surface.continuity_report()

    @staticmethod
    def _segment_velocity(points, start, end, gamma):
        r1 = points - start
        r2 = points - end
        filament = end - start
        cross = np.cross(r1, r2)
        denominator = np.einsum("ij,ij->i", cross, cross)
        direction = (
            r1 / np.linalg.norm(r1, axis=1)[:, None]
            - r2 / np.linalg.norm(r2, axis=1)[:, None]
        )
        coefficient = direction @ filament
        return (
            gamma
            * cross
            * coefficient[:, None]
            / (4.0 * np.pi * denominator[:, None])
        )

    def test_constant_doublet_reduces_to_oriented_triangle_ring(self):
        triangle = self.vertices[self.faces[0]]
        gamma = 0.37
        surface = QuadraticDoubletSurface(
            triangle,
            np.array([[0, 1, 2]]),
            np.full((1, 6), gamma),
        )
        points = np.array(
            [
                [0.25, 0.30, 0.8],
                [1.8, -0.7, 1.2],
                [-0.6, 0.2, 0.5],
            ]
        )
        expected = np.zeros_like(points)
        for start, end in ((0, 1), (1, 2), (2, 0)):
            expected += self._segment_velocity(
                points,
                triangle[start],
                triangle[end],
                gamma,
            )
        actual = surface.induced_velocity(points, quadrature_order=48)
        np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=2e-13)

    def test_quadratic_field_has_registered_quadrature_convergence(self):
        points = np.array(
            [
                [0.2, 0.3, 0.7],
                [0.8, 0.1, -0.9],
                [1.7, 0.5, 0.4],
            ]
        )
        velocity, report = self.surface.induced_velocity_converged(
            points,
            orders=(8, 12, 18, 26),
            absolute_tolerance=2e-11,
            relative_tolerance=2e-10,
        )
        self.assertTrue(report.converged)
        self.assertTrue(np.all(np.isfinite(velocity)))
        reference = self.surface.induced_velocity(
            points,
            quadrature_order=40,
        )
        np.testing.assert_allclose(velocity, reference, rtol=2e-10, atol=2e-11)

    def test_line_reduced_off_sheet_operator_matches_area_oracle(self):
        points = np.array(
            [
                [0.2, 0.3, 0.7],
                [0.8, 0.1, -0.9],
                [1.7, 0.5, 0.4],
                [-0.3, 1.2, -0.6],
            ]
        )
        expected = self.surface.induced_velocity(
            points,
            quadrature_order=80,
            singular_tolerance=1.0e-13,
        )
        actual = self.surface.induced_velocity_line_reduced(
            points,
            quadrature_order=96,
            plane_tolerance=1.0e-13,
        )
        np.testing.assert_allclose(
            actual,
            expected,
            rtol=2.0e-10,
            atol=2.0e-11,
        )
        converged, report = (
            self.surface.induced_velocity_line_reduced_converged(
                points,
                orders=(12, 18, 28, 42, 64),
                absolute_tolerance=2.0e-10,
                relative_tolerance=2.0e-9,
                plane_tolerance=1.0e-13,
            )
        )
        self.assertTrue(report.converged, report)
        np.testing.assert_allclose(
            converged,
            actual,
            rtol=2.0e-9,
            atol=2.0e-10,
        )

    def test_line_reduced_symmetric_limit_matches_sheet_average(self):
        triangle = self.vertices[self.faces[0]]
        surface = QuadraticDoubletSurface(
            triangle,
            [[0, 1, 2]],
            np.asarray(self.face_mu)[0:1],
        )
        barycentric = np.array([[0.27, 0.31, 0.42]])
        point = barycentric @ triangle
        normal = surface.element(0).normal
        target = surface.induced_velocity_sheet_average(
            [0],
            barycentric,
            quadrature_order=192,
        )[0]
        paired = []
        for height in (0.04, 0.02):
            points = np.vstack(
                (point[0] + height * normal, point[0] - height * normal)
            )
            velocity = surface.induced_velocity_line_reduced(
                points,
                quadrature_order=192,
                plane_tolerance=1.0e-14,
            )
            paired.append(0.5 * (velocity[0] + velocity[1]))
        richardson = (4.0 * paired[1] - paired[0]) / 3.0
        np.testing.assert_allclose(
            richardson,
            target,
            rtol=3.0e-3,
            atol=1.0e-3,
        )

    def test_on_sheet_velocity_is_rejected_until_principal_value_exists(self):
        with self.assertRaisesRegex(
            DistributedDoubletError,
            "principal-value",
        ):
            self.surface.induced_velocity(
                np.array([[0.25, 0.25, 0.0]]),
            )

    def test_krebs_collocation_points_are_strictly_inside(self):
        points, owners, barycentric = self.surface.interior_collocation_points()
        self.assertEqual(points.shape, (8, 3))
        self.assertEqual(owners.tolist(), [0, 0, 0, 0, 1, 1, 1, 1])
        self.assertTrue(np.all(barycentric > 0.0))
        np.testing.assert_allclose(
            barycentric.sum(axis=1),
            1.0,
            atol=2e-16,
        )
        np.testing.assert_allclose(
            barycentric[0],
            np.full(3, 1.0 / 3.0),
        )
        self.assertAlmostEqual(float(np.max(barycentric[1])), 14.0 / 15.0)

    def test_constant_doublet_sheet_average_is_exact_boundary_ring(self):
        triangle = self.vertices[self.faces[0]]
        gamma = 0.37
        surface = QuadraticDoubletSurface(
            triangle,
            [[0, 1, 2]],
            np.full((1, 6), gamma),
        )
        barycentric = np.array(
            [[0.2, 0.3, 0.5], [0.6, 0.2, 0.2]]
        )
        points = barycentric @ triangle
        expected = np.zeros_like(points)
        for start, end in ((0, 1), (1, 2), (2, 0)):
            expected += self._segment_velocity(
                points,
                triangle[start],
                triangle[end],
                gamma,
            )
        actual = surface.induced_velocity_sheet_average(
            [0, 0],
            barycentric,
            quadrature_order=80,
        )
        np.testing.assert_allclose(actual, expected, rtol=2e-11, atol=2e-12)

    def test_quadratic_sheet_average_has_no_offset_and_converges(self):
        triangle = self.vertices[self.faces[0]]
        face_mu = np.asarray(self.face_mu)[0:1]
        surface = QuadraticDoubletSurface(
            triangle,
            [[0, 1, 2]],
            face_mu,
        )
        barycentric = np.array(
            [[0.25, 0.35, 0.40], [0.50, 0.20, 0.30]]
        )
        velocity, report = surface.induced_velocity_sheet_average_converged(
            [0, 0],
            barycentric,
            orders=(24, 36, 48, 64, 80),
            absolute_tolerance=2e-10,
            relative_tolerance=2e-9,
        )
        self.assertTrue(report.converged)
        reference = surface.induced_velocity_sheet_average(
            [0, 0],
            barycentric,
            quadrature_order=100,
        )
        np.testing.assert_allclose(
            velocity,
            reference,
            rtol=2e-9,
            atol=2e-10,
        )
        self.assertTrue(np.all(np.isfinite(velocity)))

    def test_sheet_average_matches_symmetric_off_sheet_limit(self):
        triangle = self.vertices[self.faces[0]]
        surface = QuadraticDoubletSurface(
            triangle,
            [[0, 1, 2]],
            np.asarray(self.face_mu)[0:1],
        )
        barycentric = np.array([[0.3, 0.3, 0.4]])
        point = barycentric @ triangle
        normal = surface.element(0).normal
        target = surface.induced_velocity_sheet_average(
            [0],
            barycentric,
            quadrature_order=60,
        )[0]
        paired = []
        for height in (0.08, 0.04):
            field_points = np.vstack(
                (point[0] + height * normal, point[0] - height * normal)
            )
            value = surface.induced_velocity(
                field_points,
                quadrature_order=80,
                singular_tolerance=1.0e-13,
            )
            paired.append(0.5 * (value[0] + value[1]))
        richardson = (4.0 * paired[1] - paired[0]) / 3.0
        np.testing.assert_allclose(
            richardson,
            target,
            rtol=2e-2,
            atol=8e-3,
        )

    def test_coplanar_exterior_finite_part_matches_off_sheet_limit(self):
        triangle = self.vertices[self.faces[0]]
        surface = QuadraticDoubletSurface(
            triangle,
            [[0, 1, 2]],
            np.asarray(self.face_mu)[0:1],
        )
        point = np.array([[-0.18, 0.52, 0.0]])
        target = surface.induced_velocity_nonowner_sheet_points(
            point,
            quadrature_order=256,
        )[0]
        normal = surface.element(0).normal
        paired = []
        for height in (0.08, 0.04):
            field_points = np.vstack(
                (point[0] + height * normal, point[0] - height * normal)
            )
            value = surface.induced_velocity(
                field_points,
                quadrature_order=100,
                singular_tolerance=1.0e-13,
            )
            paired.append(0.5 * (value[0] + value[1]))
        richardson = (4.0 * paired[1] - paired[0]) / 3.0
        np.testing.assert_allclose(
            richardson,
            target,
            rtol=2.0e-2,
            atol=8.0e-3,
        )

    def test_sheet_average_rejects_edge_points(self):
        with self.assertRaisesRegex(
            DistributedDoubletError,
            "strict element-interior",
        ):
            self.surface.induced_velocity_sheet_average(
                [0],
                [[0.5, 0.5, 0.0]],
            )


class MaterialWakeBandTests(unittest.TestCase):
    @staticmethod
    def strength(tau, span):
        return (
            0.2
            + 0.7 * tau
            - 0.4 * span
            + 0.3 * tau**2
            + 0.5 * tau * span
            - 0.2 * span**2
        )

    def band(self, *, family="TEV"):
        span_vertex = np.linspace(0.0, 1.0, 3)
        span_p2 = np.linspace(0.0, 1.0, 5)
        previous = np.column_stack(
            (np.zeros(3), span_vertex, np.zeros(3))
        )
        current = np.column_stack(
            (np.ones(3), span_vertex, 0.1 * span_vertex)
        )
        temporal = np.array([0.0, 0.5, 1.0])
        rows = self.strength(temporal[:, None], span_p2[None, :])
        return newborn_material_wake_band(
            sheet_id=f"{family.lower()}-row-7",
            vortex_family=family,
            previous_edge=previous,
            current_edge=current,
            time_nodes=np.array([0.2, 0.25, 0.3]),
            potential_jump_rows=rows,
        )

    def test_newborn_band_exactly_reproduces_tensor_p2_material_field(self):
        band = self.band()
        self.assertEqual(len(band.surface), 4)
        self.assertTrue(band.surface.continuity_report().compatible)
        for face_index in range(len(band.surface)):
            element = band.surface.element(face_index)
            nodes = element.material_nodes
            expected = self.strength(nodes[:, 0], nodes[:, 1])
            np.testing.assert_allclose(
                element.material_mu,
                expected,
                atol=2e-15,
            )

    def test_middle_shedding_row_is_mandatory_not_inferred(self):
        span_vertex = np.linspace(0.0, 1.0, 3)
        edge = np.column_stack(
            (np.zeros(3), span_vertex, np.zeros(3))
        )
        with self.assertRaisesRegex(
            DistributedDoubletError,
            "potential_jump_rows",
        ):
            newborn_material_wake_band(
                sheet_id="missing-midpoint",
                vortex_family="TEV",
                previous_edge=edge,
                current_edge=edge + [1.0, 0.0, 0.0],
                time_nodes=[0.0, 0.5, 1.0],
                potential_jump_rows=np.zeros((2, 5)),
            )

    def test_mirror_symmetric_diagonals_preserve_continuous_p2_trace(self):
        span_vertex = np.linspace(-1.0, 1.0, 5)
        span_p2 = np.linspace(-1.0, 1.0, 9)
        previous = np.column_stack(
            (np.ones(5), span_vertex, np.zeros(5))
        )
        current = previous.copy()
        current[:, 0] = 0.0
        rows = np.repeat((1.0 - span_p2**2)[None, :], 3, axis=0)
        band = newborn_material_wake_band(
            sheet_id="mirror-symmetric-band",
            vortex_family="TEV",
            previous_edge=previous,
            current_edge=current,
            time_nodes=[0.0, 0.5, 1.0],
            potential_jump_rows=rows,
            span_diagonal_pattern="mirror_symmetric",
        )
        self.assertTrue(band.surface.continuity_report().compatible)
        for face_index in range(len(band.surface)):
            element = band.surface.element(face_index)
            expected = 1.0 - element.material_nodes[:, 1] ** 2
            np.testing.assert_allclose(
                element.material_mu,
                expected,
                rtol=0.0,
                atol=2.0e-15,
            )

    def test_material_motion_preserves_strength_and_family_identity(self):
        band = self.band(family="LEV_SUCTION")
        moved_vertices = band.surface.vertices + np.column_stack(
            (
                np.zeros(len(band.surface.vertices)),
                np.zeros(len(band.surface.vertices)),
                0.05 * band.surface.vertices[:, 0],
            )
        )
        moved = band.material_update(moved_vertices)
        self.assertEqual(moved.vortex_family, "LEV_SUCTION")
        self.assertEqual(moved.sheet_id, band.sheet_id)
        self.assertTrue(band.surface.kelvin_report(moved.surface).passed)

    def test_tev_and_lev_are_distinct_topological_families(self):
        self.assertEqual(self.band(family="TEV").vortex_family, "TEV")
        self.assertEqual(
            self.band(family="LEV_PRESSURE").vortex_family,
            "LEV_PRESSURE",
        )
        with self.assertRaisesRegex(
            DistributedDoubletError,
            "vortex_family",
        ):
            self.band(family="MIXED_LEV_TEV")

    def test_patch_roles_make_every_band_boundary_explicit(self):
        band = self.band()
        patch = band.as_patch(
            upstream_interface="bound-te",
            downstream_interface="next-row",
        )
        self.assertEqual(len(patch.boundary_roles), 6)
        self.assertEqual(
            patch.boundary_roles[(0, 1)],
            "interface:bound-te:0",
        )
        self.assertEqual(
            patch.boundary_roles[(3, 4)],
            "interface:next-row:0",
        )
        self.assertEqual(patch.boundary_roles[(0, 3)], "zero")

    def closed_history(self):
        span_vertex = np.linspace(0.0, 1.0, 3)
        span_p2 = np.linspace(0.0, 1.0, 5)
        bands = []
        for band_index in range(3):
            previous = np.column_stack(
                (
                    np.full(3, float(band_index)),
                    span_vertex,
                    np.zeros(3),
                )
            )
            current = np.column_stack(
                (
                    np.full(3, float(band_index + 1)),
                    span_vertex,
                    np.zeros(3),
                )
            )
            times = np.array(
                [
                    float(band_index),
                    band_index + 0.5,
                    float(band_index + 1),
                ]
            )
            rows = (
                times[:, None]
                * (3.0 - times[:, None])
                * span_p2[None, :]
                * (1.0 - span_p2[None, :])
            )
            bands.append(
                newborn_material_wake_band(
                    sheet_id=f"tev-band-{band_index}",
                    vortex_family="TEV",
                    previous_edge=previous,
                    current_edge=current,
                    time_nodes=times,
                    potential_jump_rows=rows,
                )
            )
        return MaterialWakeHistory("tev-history", tuple(bands))

    def test_multistep_history_is_continuous_and_globally_bounded(self):
        history = self.closed_history()
        report = history.continuity_report()
        self.assertTrue(report.compatible)
        self.assertEqual(report.internal_seams, 2)
        self.assertEqual(report.max_time_gap, 0.0)
        self.assertEqual(report.max_geometry_gap, 0.0)
        self.assertEqual(report.max_trace_jump, 0.0)
        assembly = QuadraticDoubletAssembly(
            history.as_patches(
                oldest_role="zero",
                newest_role="zero",
            )
        )
        topology = assembly.topology_report()
        self.assertTrue(topology.compatible)
        self.assertEqual(topology.coupled_interfaces, 4)

    def test_history_rejects_strength_or_geometry_break_on_append(self):
        history = self.closed_history()
        last = history.bands[-1]
        broken_mu = last.potential_jump_rows.copy()
        broken_mu[0, 2] += 0.02
        broken = newborn_material_wake_band(
            sheet_id="broken-band",
            vortex_family="TEV",
            previous_edge=last.surface.vertices[: last.span_nodes],
            current_edge=last.surface.vertices[last.span_nodes :] + [1, 0, 0],
            time_nodes=[2.0, 2.5, 3.0],
            potential_jump_rows=broken_mu,
        )
        with self.assertRaisesRegex(
            DistributedDoubletError,
            "breaks history",
        ):
            MaterialWakeHistory(
                "prefix",
                history.bands[:2],
            ).append(broken)


if __name__ == "__main__":
    unittest.main()
