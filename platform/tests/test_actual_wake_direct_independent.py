"""Regression tests for direct independent current-wake assembly."""
from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


PLATFORM = Path(__file__).resolve().parents[1]
if str(PLATFORM) not in sys.path:
    sys.path.insert(0, str(PLATFORM))

from actual_boundary_3d_cut_wake_junction_guard import (  # noqa: E402
    build_canonical_diamond_wing,
)
from claim_runtime.actual_boundary_body_wake import (  # noqa: E402
    MaterialWakeCutAttachment,
    solve_actual_boundary_body_wake_p2,
)
from claim_runtime.actual_wake_direct_independent import (  # noqa: E402
    assemble_direct_independent_wake_matrix,
)
from claim_runtime.classified_p2_cut_topology import (  # noqa: E402
    classified_p2_cut_topology,
)
from claim_runtime.distributed_doublet import (  # noqa: E402
    MaterialWakeHistory,
    newborn_material_wake_band,
)
from claim_runtime.thick_body_neumann_shadow import (  # noqa: E402
    closed_triangular_mesh,
)


def _history(mesh, topology) -> MaterialWakeHistory:
    body_edge = mesh.vertices[
        topology.ordered_cut_vertex_indices
    ].copy()
    span = body_edge[:, 1]
    vertex_shape = 1.0 - span**2
    far_edge = body_edge.copy()
    far_edge[:, 0] = 2.0
    far_edge[:, 2] += 0.20 * vertex_shape
    seam_edge = body_edge.copy()
    seam_edge[:, 0] = 1.5
    seam_edge[:, 2] += 0.12 * vertex_shape

    cut_span = topology.cut_node_coordinates[:, 1]
    cut_shape = 1.0 - cut_span**2
    old = newborn_material_wake_band(
        sheet_id="direct-W-old",
        vortex_family="TEV",
        previous_edge=far_edge,
        current_edge=seam_edge,
        time_nodes=np.array((0.0, 0.5, 1.0)),
        potential_jump_rows=np.array(
            (0.20 * cut_shape, 0.30 * cut_shape, 0.40 * cut_shape)
        ),
        span_diagonal_pattern="mirror_symmetric",
    )
    newest = newborn_material_wake_band(
        sheet_id="direct-W-newest",
        vortex_family="TEV",
        previous_edge=seam_edge,
        current_edge=body_edge,
        time_nodes=np.array((1.0, 1.5, 2.0)),
        potential_jump_rows=np.array(
            (0.40 * cut_shape, 0.10 * cut_shape, np.zeros_like(cut_shape))
        ),
        span_diagonal_pattern="mirror_symmetric",
    )
    return MaterialWakeHistory(
        "direct-W-prescribed-history",
        (old, newest),
    )


def _rotation_matrix(axis, angle) -> np.ndarray:
    direction = np.asarray(axis, dtype=float)
    direction /= np.linalg.norm(direction)
    x, y, z = direction
    cross = np.array(
        ((0.0, -z, y), (z, 0.0, -x), (-y, x, 0.0))
    )
    return (
        np.cos(angle) * np.eye(3)
        + (1.0 - np.cos(angle))
        * np.outer(direction, direction)
        + np.sin(angle) * cross
    )


def _transform_history(
    history: MaterialWakeHistory,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> MaterialWakeHistory:
    bands = []
    for band in history.bands:
        count = band.span_nodes
        bands.append(
            newborn_material_wake_band(
                sheet_id=band.sheet_id,
                vortex_family=band.vortex_family,
                previous_edge=(
                    band.surface.vertices[:count] @ rotation.T
                    + translation
                ),
                current_edge=(
                    band.surface.vertices[count:] @ rotation.T
                    + translation
                ),
                time_nodes=band.time_nodes,
                potential_jump_rows=band.potential_jump_rows,
                span_diagonal_pattern="mirror_symmetric",
            )
        )
    return MaterialWakeHistory(history.history_id, tuple(bands))


def _reverse_parameterization(
    history: MaterialWakeHistory,
) -> MaterialWakeHistory:
    bands = []
    for band in history.bands:
        count = band.span_nodes
        bands.append(
            newborn_material_wake_band(
                sheet_id=band.sheet_id,
                vortex_family=band.vortex_family,
                previous_edge=(
                    band.surface.vertices[:count][::-1]
                ),
                current_edge=(
                    band.surface.vertices[count:][::-1]
                ),
                time_nodes=band.time_nodes,
                potential_jump_rows=(
                    -band.potential_jump_rows[:, ::-1]
                ),
                span_diagonal_pattern="mirror_symmetric",
            )
        )
    return MaterialWakeHistory(history.history_id, tuple(bands))


class ActualWakeDirectIndependentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        (
            cls.mesh,
            cls.upper,
            cls.lower,
            cls.cut_edges,
            cls.endpoints,
        ) = build_canonical_diamond_wing()
        cls.topology = classified_p2_cut_topology(
            cls.mesh,
            upper_face_indices=cls.upper,
            lower_face_indices=cls.lower,
            cut_edges=cls.cut_edges,
            zero_jump_end_vertices=cls.endpoints,
        )
        cls.history = _history(cls.mesh, cls.topology)
        cls.attachment = MaterialWakeCutAttachment(
            cls.topology.ordered_cut_vertex_indices,
            1,
        )
        angle = np.deg2rad(5.0)
        cls.incident = np.repeat(
            np.array(((np.cos(angle), 0.0, np.sin(angle)),)),
            len(cls.mesh.faces),
            axis=0,
        )
        cls.records = {}
        for order in (5, 10):
            solution = solve_actual_boundary_body_wake_p2(
                cls.mesh,
                cls.topology,
                incident_velocity=cls.incident,
                downstream_edge_x=None,
                prescribed_wake_history=cls.history,
                prescribed_wake_attachment=cls.attachment,
                target_quadrature_order=order,
                source_quadrature_order=order,
            )
            direct = assemble_direct_independent_wake_matrix(
                cls.mesh,
                cls.topology,
                cls.history,
                prescribed_wake_attachment=cls.attachment,
                target_quadrature_order=order,
                source_quadrature_order=order,
            )
            cls.records[order] = (solution, direct)

    def test_q5_q10_direct_factorization_matches_eliminated_block(
        self,
    ) -> None:
        for order, (solution, direct) in self.records.items():
            with self.subTest(order=order):
                self.assertEqual(direct.matrix.shape, (81, 7))
                self.assertEqual(direct.active_jump.shape, (7, 81))
                self.assertEqual(direct.rank, 7)
                self.assertLessEqual(
                    float(
                        np.max(
                            np.abs(
                                direct.eliminated_wake_matrix
                                - solution.wake_matrix
                            ),
                            initial=0.0,
                        )
                    ),
                    5.0e-13,
                )
                self.assertEqual(
                    direct.body_wake_paired_topology_counts,
                    solution.body_wake_paired_topology_counts,
                )
                self.assertEqual(
                    direct.body_wake_paired_topology_counts[
                        "common_edge"
                    ],
                    8,
                )

    def test_proper_rigid_transform_is_covariant_even_if_cut_order_reverses(
        self,
    ) -> None:
        rotation = _rotation_matrix(
            np.array((0.3, -0.4, 0.5)),
            np.deg2rad(37.0),
        )
        translation = np.array((0.2, -0.1, 0.4))
        moved_mesh = closed_triangular_mesh(
            self.mesh.vertices @ rotation.T + translation,
            self.mesh.faces,
        )
        moved_topology = classified_p2_cut_topology(
            moved_mesh,
            upper_face_indices=self.upper,
            lower_face_indices=self.lower,
            cut_edges=self.cut_edges,
            zero_jump_end_vertices=self.endpoints,
        )
        self.assertTrue(
            np.array_equal(
                moved_topology.ordered_cut_vertex_indices,
                self.topology.ordered_cut_vertex_indices[::-1],
            )
        )
        moved = assemble_direct_independent_wake_matrix(
            moved_mesh,
            moved_topology,
            _transform_history(
                self.history,
                rotation,
                translation,
            ),
            prescribed_wake_attachment=self.attachment,
            target_quadrature_order=10,
            source_quadrature_order=10,
        )
        reference = self.records[10][1]
        active_reversal = np.eye(
            reference.matrix.shape[1]
        )[:, ::-1]
        self.assertLessEqual(
            float(
                np.max(
                    np.abs(
                        moved.matrix @ active_reversal
                        - reference.matrix
                    ),
                    initial=0.0,
                )
            ),
            5.0e-12,
        )
        self.assertLessEqual(
            float(
                np.max(
                    np.abs(
                        moved.eliminated_wake_matrix
                        - reference.eliminated_wake_matrix
                    ),
                    initial=0.0,
                )
            ),
            5.0e-12,
        )

    def test_span_mirror_parameterization_is_covariant(self) -> None:
        reversed_history = _reverse_parameterization(self.history)
        reverse_attachment = MaterialWakeCutAttachment(
            self.topology.ordered_cut_vertex_indices[::-1],
            -1,
        )
        mirrored = assemble_direct_independent_wake_matrix(
            self.mesh,
            self.topology,
            reversed_history,
            prescribed_wake_attachment=reverse_attachment,
            target_quadrature_order=10,
            source_quadrature_order=10,
        )
        reference = self.records[10][1]
        self.assertEqual(mirrored.rank, 7)
        self.assertLessEqual(
            float(
                np.max(
                    np.abs(mirrored.matrix - reference.matrix),
                    initial=0.0,
                )
            ),
            5.0e-11,
        )
        self.assertLessEqual(
            float(
                np.max(
                    np.abs(
                        mirrored.eliminated_wake_matrix
                        - reference.eliminated_wake_matrix
                    ),
                    initial=0.0,
                )
            ),
            5.0e-11,
        )

    def test_assembly_does_not_mutate_prescribed_history(self) -> None:
        geometry = tuple(
            band.surface.vertices.copy()
            for band in self.history.bands
        )
        strength = tuple(
            band.potential_jump_rows.copy()
            for band in self.history.bands
        )
        assemble_direct_independent_wake_matrix(
            self.mesh,
            self.topology,
            self.history,
            prescribed_wake_attachment=self.attachment,
            target_quadrature_order=5,
            source_quadrature_order=5,
        )
        for band, expected_geometry, expected_strength in zip(
            self.history.bands,
            geometry,
            strength,
            strict=True,
        ):
            np.testing.assert_array_equal(
                band.surface.vertices,
                expected_geometry,
            )
            np.testing.assert_array_equal(
                band.potential_jump_rows,
                expected_strength,
            )


if __name__ == "__main__":
    unittest.main()
