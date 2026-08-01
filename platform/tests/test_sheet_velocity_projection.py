import sys
import unittest
from pathlib import Path

import numpy as np

PLATFORM = Path(__file__).resolve().parents[1]
if str(PLATFORM) not in sys.path:
    sys.path.insert(0, str(PLATFORM))

from claim_runtime.distributed_doublet import (  # noqa: E402
    DistributedDoubletError,
    QuadraticDoubletAssembly,
    QuadraticDoubletPatch,
    QuadraticDoubletSurface,
)
from claim_runtime.sheet_velocity_projection import (  # noqa: E402
    project_assembly_sheet_average_velocity,
    project_assembly_vertex_star_normal_geometry_velocity,
    project_sheet_average_velocity,
    project_sheet_normal_geometry_velocity,
    project_vertex_star_normal_geometry_velocity,
)


def velocity_field(points):
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    return np.column_stack(
        (
            1.0 + 0.2 * x - 0.1 * y + 0.3 * z,
            -0.4 + 0.5 * x + 0.2 * y - 0.1 * z,
            0.7 - 0.3 * x + 0.4 * y + 0.2 * z,
        )
    )


def surface():
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.5, 0.0],
            [0.5, 0.5, 0.0],
            [1.0, 0.5, 0.0],
            [0.0, 1.0, 0.0],
            [0.5, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ]
    )
    faces = np.array(
        [
            [0, 1, 4],
            [0, 4, 3],
            [1, 2, 5],
            [1, 5, 4],
            [3, 4, 7],
            [3, 7, 6],
            [4, 5, 8],
            [4, 8, 7],
        ]
    )
    return QuadraticDoubletSurface(
        vertices,
        faces,
        np.zeros((len(faces), 6)),
    )


class SheetVelocityProjectionTests(unittest.TestCase):
    def test_global_affine_field_is_recovered_exactly(self):
        state = surface()
        points, _, _ = state.interior_collocation_points()
        projection = project_sheet_average_velocity(
            state,
            velocity_field(points),
        )
        np.testing.assert_allclose(
            projection.dof_velocity,
            velocity_field(projection.dof_positions),
            rtol=2.0e-14,
            atol=2.0e-14,
        )
        np.testing.assert_allclose(
            projection.vertex_velocity,
            velocity_field(state.vertices),
            rtol=2.0e-14,
            atol=2.0e-14,
        )
        self.assertTrue(projection.report.full_rank)
        self.assertEqual(projection.report.gauge, "sheet_average")
        self.assertLess(projection.report.max_abs_residual, 2.0e-14)

    def test_face_evaluation_preserves_manufactured_field(self):
        state = surface()
        points, _, _ = state.interior_collocation_points()
        projection = project_sheet_average_velocity(
            state,
            velocity_field(points),
        )
        barycentric = np.array([[0.21, 0.33, 0.46]])
        element = state.element(3)
        expected = velocity_field(barycentric @ element.vertices)
        actual = projection.evaluate_face(3, barycentric)
        np.testing.assert_allclose(actual, expected, atol=2.0e-14)

    def test_quadratic_content_remains_visible_in_projection_residual(self):
        state = surface()
        points, _, _ = state.interior_collocation_points()
        velocity = velocity_field(points)
        velocity[:, 2] += 0.6 * points[:, 0] * points[:, 1]
        projection = project_sheet_average_velocity(state, velocity)
        self.assertGreater(projection.report.max_abs_residual, 1.0e-3)

    def test_normal_projection_ignores_tangential_jump_content(self):
        state = surface()
        points, _, _ = state.interior_collocation_points()
        normal_speed = 0.3 + 0.2 * points[:, 0] - 0.1 * points[:, 1]
        velocity = np.column_stack(
            (
                points[:, 0] * points[:, 1],
                points[:, 0] ** 2 - points[:, 1] ** 2,
                normal_speed,
            )
        )
        projection = project_sheet_normal_geometry_velocity(
            state,
            velocity,
        )
        expected_speed = (
            0.3
            + 0.2 * state.vertices[:, 0]
            - 0.1 * state.vertices[:, 1]
        )
        np.testing.assert_allclose(
            projection.vertex_normal_speed,
            expected_speed,
            atol=2.0e-14,
        )
        np.testing.assert_allclose(
            projection.vertex_velocity[:, :2],
            0.0,
            atol=2.0e-14,
        )
        self.assertLess(
            projection.report.max_abs_residual_fraction,
            2.0e-14,
        )
        star = project_vertex_star_normal_geometry_velocity(
            state,
            velocity,
        )
        np.testing.assert_allclose(
            star.vertex_normal_speed,
            expected_speed,
            atol=2.0e-14,
        )
        self.assertEqual(
            star.report.gauge,
            "sheet_average_normal_vertex_star",
        )

    def test_assembly_welds_interface_vertices_before_projection(self):
        first = QuadraticDoubletSurface(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0]],
            [[0, 1, 2]],
            [[0, 0, 0, 0, 0.4, 0]],
        )
        second = QuadraticDoubletSurface(
            [[1, 0, 0], [1, 1, 0], [0, 1, 0]],
            [[0, 1, 2]],
            [[0, 0, 0, 0, 0, 0.4]],
        )
        assembly = QuadraticDoubletAssembly(
            [
                QuadraticDoubletPatch(
                    "first",
                    first,
                    {
                        (0, 1): "zero",
                        (0, 2): "zero",
                        (1, 2): "interface:diagonal",
                    },
                ),
                QuadraticDoubletPatch(
                    "second",
                    second,
                    {
                        (0, 1): "zero",
                        (1, 2): "zero",
                        (0, 2): "interface:diagonal",
                    },
                ),
            ]
        )
        points, _, _, _ = assembly.interior_collocation_points()
        projection = project_assembly_sheet_average_velocity(
            assembly,
            velocity_field(points),
        )
        self.assertEqual(len(projection.dof_positions), 4)
        np.testing.assert_allclose(
            projection.vertex_velocity(0),
            velocity_field(first.vertices),
            atol=2.0e-14,
        )
        np.testing.assert_allclose(
            projection.vertex_velocity(1),
            velocity_field(second.vertices),
            atol=2.0e-14,
        )
        normal_projection = (
            project_assembly_vertex_star_normal_geometry_velocity(
                assembly,
                velocity_field(points),
            )
        )
        self.assertEqual(
            normal_projection.report.gauge,
            "sheet_average_normal_vertex_star_assembly",
        )
        np.testing.assert_allclose(
            normal_projection.vertex_velocity(0)[:, 2],
            velocity_field(first.vertices)[:, 2],
            atol=2.0e-14,
        )
        np.testing.assert_allclose(
            normal_projection.vertex_velocity(1)[:, 2],
            velocity_field(second.vertices)[:, 2],
            atol=2.0e-14,
        )

    def test_unreferenced_geometry_vertex_is_rejected_as_rank_deficient(self):
        state = QuadraticDoubletSurface(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [2, 2, 0]],
            [[0, 1, 2]],
            np.zeros((1, 6)),
        )
        points, _, _ = state.interior_collocation_points()
        with self.assertRaisesRegex(
            DistributedDoubletError,
            "rank deficient",
        ):
            project_sheet_average_velocity(
                state,
                velocity_field(points),
            )

    def test_shape_and_nonfinite_inputs_fail(self):
        state = surface()
        with self.assertRaisesRegex(
            DistributedDoubletError,
            "shape",
        ):
            project_sheet_average_velocity(
                state,
                np.zeros((3, 3)),
            )
        bad = np.zeros((4 * len(state), 3))
        bad[0, 0] = np.nan
        with self.assertRaisesRegex(
            DistributedDoubletError,
            "non-finite",
        ):
            project_sheet_average_velocity(state, bad)


if __name__ == "__main__":
    unittest.main()
