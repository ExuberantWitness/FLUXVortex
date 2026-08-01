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
from claim_runtime.material_sheet_advection import (  # noqa: E402
    advance_assembly_normal_geometry_heun,
    advance_material_surface_heun,
    advance_surface_normal_geometry_heun,
    self_induced_assembly_vertex_star_normal_velocity,
    self_induced_geometry_velocity,
    self_induced_normal_geometry_velocity,
    self_induced_vertex_star_normal_velocity,
)
from claim_runtime.sheet_velocity_projection import (  # noqa: E402
    project_assembly_vertex_star_normal_geometry_velocity,
    project_sheet_average_velocity,
    project_sheet_normal_geometry_velocity,
)


def zero_boundary_surface():
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
    face_mu = []
    for face in faces:
        triangle = vertices[face]
        nodes = np.vstack(
            (
                triangle,
                0.5 * (triangle[0] + triangle[1]),
                0.5 * (triangle[1] + triangle[2]),
                0.5 * (triangle[2] + triangle[0]),
            )
        )
        x = nodes[:, 0]
        y = nodes[:, 1]
        face_mu.append(x * (1.0 - x) * y * (1.0 - y))
    return QuadraticDoubletSurface(
        vertices,
        faces,
        np.asarray(face_mu),
    )


def rotation_provider(angular_speed):
    def provider(surface):
        points, _, _ = surface.interior_collocation_points()
        velocity = np.column_stack(
            (
                -angular_speed * points[:, 1],
                angular_speed * points[:, 0],
                np.zeros(len(points)),
            )
        )
        return project_sheet_average_velocity(surface, velocity)

    return provider


def split_square_assembly():
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
    monolithic = QuadraticDoubletSurface(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]],
        [[0, 1, 2], [1, 3, 2]],
        [
            [0, 0, 0, 0, 0.4, 0],
            [0, 0, 0, 0, 0, 0.4],
        ],
    )
    return assembly, monolithic


def integrate_rotation(surface, *, angular_speed, final_time, steps):
    dt = final_time / steps
    state = surface
    provider = rotation_provider(angular_speed)
    for _ in range(steps):
        state = advance_material_surface_heun(
            state,
            dt=dt,
            velocity_provider=provider,
        ).surface
    return state


class MaterialSheetAdvectionTests(unittest.TestCase):
    def test_heun_has_second_order_rigid_rotation_convergence(self):
        initial = zero_boundary_surface()
        angular_speed = 0.7
        final_time = 0.4
        angle = angular_speed * final_time
        rotation = np.array(
            [
                [np.cos(angle), -np.sin(angle), 0.0],
                [np.sin(angle), np.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        exact = initial.vertices @ rotation.T
        errors = []
        for steps in (4, 8, 16):
            numerical = integrate_rotation(
                initial,
                angular_speed=angular_speed,
                final_time=final_time,
                steps=steps,
            )
            errors.append(
                float(np.max(np.linalg.norm(numerical.vertices - exact, axis=1)))
            )
            self.assertTrue(
                initial.kelvin_report(numerical).passed
            )
            self.assertTrue(numerical.continuity_report().compatible)
            self.assertTrue(numerical.boundary_report().compatible)
        self.assertGreater(errors[0] / errors[1], 3.8)
        self.assertGreater(errors[1] / errors[2], 3.8)

    def test_self_induced_step_is_finite_and_preserves_material_identity(self):
        initial = zero_boundary_surface()
        step = advance_material_surface_heun(
            initial,
            dt=0.01,
            velocity_provider=lambda state: self_induced_geometry_velocity(
                state,
                quadrature_order=64,
            ),
        )
        self.assertTrue(step.report.passed)
        self.assertTrue(np.all(np.isfinite(step.surface.vertices)))
        self.assertGreater(
            float(
                np.max(
                    np.linalg.norm(
                        step.surface.vertices - initial.vertices,
                        axis=1,
                    )
                )
            ),
            0.0,
        )

    def test_normal_geometry_gauge_exactly_advects_uniform_normal_motion(self):
        initial = zero_boundary_surface()

        def provider(state):
            points, _, _ = state.interior_collocation_points()
            velocity = np.tile([0.0, 0.0, 0.2], (len(points), 1))
            return project_sheet_normal_geometry_velocity(
                state,
                velocity,
            )

        step = advance_surface_normal_geometry_heun(
            initial,
            dt=0.05,
            velocity_provider=provider,
        )
        expected = initial.vertices + [0.0, 0.0, 0.01]
        np.testing.assert_allclose(
            step.surface.vertices,
            expected,
            atol=2.0e-15,
        )
        self.assertTrue(step.report.passed)

    def test_self_induced_normal_geometry_step_is_finite(self):
        initial = zero_boundary_surface()
        step = advance_surface_normal_geometry_heun(
            initial,
            dt=0.01,
            velocity_provider=lambda state: (
                self_induced_normal_geometry_velocity(
                    state,
                    quadrature_order=64,
                )
            ),
        )
        self.assertTrue(step.report.passed)
        self.assertTrue(np.all(np.isfinite(step.surface.vertices)))

    def test_self_induced_vertex_star_step_is_finite(self):
        initial = zero_boundary_surface()
        step = advance_surface_normal_geometry_heun(
            initial,
            dt=0.01,
            velocity_provider=lambda state: (
                self_induced_vertex_star_normal_velocity(
                    state,
                    quadrature_order=64,
                )
            ),
        )
        self.assertTrue(step.report.passed)
        self.assertTrue(np.all(np.isfinite(step.surface.vertices)))

    def test_assembly_heun_keeps_declared_seam_and_material_strength(self):
        assembly, _ = split_square_assembly()

        def provider(state):
            points, _, _, _ = state.interior_collocation_points()
            velocity = np.tile([0.0, 0.0, 0.2], (len(points), 1))
            return project_assembly_vertex_star_normal_geometry_velocity(
                state,
                velocity,
            )

        step = advance_assembly_normal_geometry_heun(
            assembly,
            dt=0.05,
            velocity_provider=provider,
        )
        self.assertTrue(step.report.passed)
        self.assertEqual(step.report.topology.coupled_interfaces, 1)
        self.assertLess(
            step.report.topology.max_interface_geometry_gap,
            2.0e-15,
        )
        for original, advanced in zip(
            assembly.patches,
            step.assembly.patches,
        ):
            np.testing.assert_allclose(
                advanced.surface.vertices,
                original.surface.vertices + [0.0, 0.0, 0.01],
                atol=2.0e-15,
            )
            np.testing.assert_array_equal(
                advanced.surface.face_mu,
                original.surface.face_mu,
            )

    def test_split_and_monolithic_self_induced_velocity_are_equivalent(self):
        assembly, monolithic = split_square_assembly()
        assembly_velocity = (
            self_induced_assembly_vertex_star_normal_velocity(
                assembly,
                quadrature_order=96,
            )
        )
        monolithic_velocity = self_induced_vertex_star_normal_velocity(
            monolithic,
            quadrature_order=96,
        )
        patch_velocity = np.vstack(
            (
                assembly_velocity.vertex_velocity(0),
                assembly_velocity.vertex_velocity(1)[1:2],
            )
        )
        np.testing.assert_allclose(
            patch_velocity,
            monolithic_velocity.vertex_velocity,
            atol=2.0e-12,
            rtol=2.0e-12,
        )

    def test_invalid_step_fails(self):
        initial = zero_boundary_surface()
        with self.assertRaisesRegex(
            DistributedDoubletError,
            "dt",
        ):
            advance_material_surface_heun(
                initial,
                dt=0.0,
                velocity_provider=rotation_provider(1.0),
            )


if __name__ == "__main__":
    unittest.main()
