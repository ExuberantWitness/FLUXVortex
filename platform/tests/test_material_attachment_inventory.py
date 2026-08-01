"""Tests for the S3ai-v2 face_mu-only material inventory."""
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
)
from claim_runtime.classified_p2_cut_topology import (  # noqa: E402
    classified_p2_cut_topology,
)
from claim_runtime.distributed_doublet import (  # noqa: E402
    MaterialWakeBand,
    MaterialWakeHistory,
    QuadraticDoubletSurface,
    newborn_material_wake_band,
)
from claim_runtime.material_attachment_inventory import (  # noqa: E402
    MaterialAttachmentInventoryError,
    extract_surface_boundary_trace,
    material_inventory_increment,
    observe_material_attachment_inventory,
    observe_material_history_surface,
)


def _body_potential(topology, cut_trace: np.ndarray) -> np.ndarray:
    result = np.zeros(topology.dof_count)
    for upper, lower, value in zip(
        topology.upper_cut_dofs,
        topology.lower_cut_dofs,
        cut_trace,
    ):
        if int(upper) == int(lower):
            if value != 0.0:
                raise AssertionError("shared tip DOF requires zero trace")
            continue
        result[int(upper)] = 0.5 * value
        result[int(lower)] = -0.5 * value
    np.testing.assert_allclose(topology.cut_jump(result), cut_trace)
    return result


def _history(
    body_edge: np.ndarray,
    rows: tuple[np.ndarray, ...],
    *,
    reverse: bool = False,
) -> MaterialWakeHistory:
    count = len(rows)
    x_nodes = np.linspace(2.0, 1.0, count + 1)
    edges = []
    for x in x_nodes:
        edge = body_edge.copy()
        edge[:, 0] = x
        edge[:, 2] += 0.03 * (2.0 - x) * (
            1.0 - body_edge[:, 1] ** 2
        )
        edges.append(edge[::-1] if reverse else edge)
    bands = []
    for index, (previous, current) in enumerate(zip(rows[:-1], rows[1:])):
        values = np.stack(
            (
                previous,
                0.5 * (previous + current),
                current,
            )
        )
        if reverse:
            values = -values[:, ::-1]
        bands.append(
            newborn_material_wake_band(
                sheet_id=f"inventory-band-{index}",
                vortex_family="TEV",
                previous_edge=edges[index],
                current_edge=edges[index + 1],
                time_nodes=np.array(
                    (float(index), index + 0.5, float(index + 1))
                ),
                potential_jump_rows=values,
                span_diagonal_pattern="mirror_symmetric",
            )
        )
    return MaterialWakeHistory("inventory-history", tuple(bands))


def _replace_band(
    band: MaterialWakeBand,
    *,
    face_mu: np.ndarray | None = None,
    rows: np.ndarray | None = None,
) -> MaterialWakeBand:
    surface = QuadraticDoubletSurface(
        band.surface.vertices,
        band.surface.faces,
        band.surface.face_mu if face_mu is None else face_mu,
    )
    return MaterialWakeBand(
        sheet_id=band.sheet_id,
        vortex_family=band.vortex_family,
        time_nodes=band.time_nodes,
        span_nodes=band.span_nodes,
        surface=surface,
        potential_jump_rows=(
            band.potential_jump_rows if rows is None else rows
        ),
    )


def _change_boundary_midpoint(
    band: MaterialWakeBand,
    *,
    boundary: str,
    segment: int,
    delta: float,
) -> MaterialWakeBand:
    offset = 0 if boundary == "previous" else band.span_nodes
    first = offset + segment
    second = first + 1
    face_mu = band.surface.face_mu.copy()
    matches = []
    midpoint_by_edge = {
        frozenset((0, 1)): 3,
        frozenset((1, 2)): 4,
        frozenset((0, 2)): 5,
    }
    for face_index, face in enumerate(band.surface.faces):
        local_first = np.flatnonzero(face == first)
        local_second = np.flatnonzero(face == second)
        if len(local_first) == 1 and len(local_second) == 1:
            midpoint = midpoint_by_edge[
                frozenset((int(local_first[0]), int(local_second[0])))
            ]
            matches.append((face_index, midpoint))
    if len(matches) != 1:
        raise AssertionError("test boundary edge must be unique")
    face_mu[matches[0]] += delta
    return _replace_band(band, face_mu=face_mu)


class MaterialAttachmentInventoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        mesh, upper, lower, cut_edges, endpoints = (
            build_canonical_diamond_wing()
        )
        cls.mesh = mesh
        cls.topology = classified_p2_cut_topology(
            mesh,
            upper_face_indices=upper,
            lower_face_indices=lower,
            cut_edges=cut_edges,
            zero_jump_end_vertices=endpoints,
        )
        cls.body_edge = mesh.vertices[
            cls.topology.ordered_cut_vertex_indices
        ].copy()
        cls.trace = np.array(
            (0.0, 0.13, -0.08, 0.31, -0.17, 0.22, 0.07, -0.11, 0.0)
        )
        cls.zero = np.zeros_like(cls.trace)
        cls.attachment = MaterialWakeCutAttachment(
            cls.topology.ordered_cut_vertex_indices,
            1,
        )

    def _observe(
        self,
        history: MaterialWakeHistory,
        trace: np.ndarray,
        *,
        attachment: MaterialWakeCutAttachment | None = None,
        birth_sign: int = 1,
    ):
        return observe_material_attachment_inventory(
            self.topology,
            history,
            global_body_potential=_body_potential(
                self.topology, trace
            ),
            attachment=(
                self.attachment if attachment is None else attachment
            ),
            birth_sign=birth_sign,
        )

    def test_boundary_traces_and_history_release_come_from_surface(self):
        history = _history(
            self.body_edge,
            (self.zero, 0.37 * self.trace, self.trace),
        )
        observation = observe_material_history_surface(history)
        np.testing.assert_allclose(
            observation.material_release, self.trace, atol=2.0e-16
        )
        self.assertLessEqual(
            observation.maximum_history_seam_error, 2.0e-16
        )
        self.assertLessEqual(
            observation.maximum_surface_internal_trace_error, 2.0e-16
        )
        for band in history.bands:
            np.testing.assert_allclose(
                extract_surface_boundary_trace(band, "previous"),
                band.potential_jump_rows[0],
                atol=2.0e-16,
            )
            np.testing.assert_allclose(
                extract_surface_boundary_trace(band, "current"),
                band.potential_jump_rows[2],
                atol=2.0e-16,
            )

    def test_row_and_surface_sources_are_independent(self):
        history = _history(
            self.body_edge,
            (self.zero, 0.37 * self.trace, self.trace),
        )
        baseline = self._observe(history, self.trace)
        band = history.bands[-1]

        changed_rows = band.potential_jump_rows.copy()
        changed_rows[0, 3] += 0.4
        changed_rows[1:, 3] += np.array((-0.2, 0.1))
        row_mutated = MaterialWakeHistory(
            history.history_id,
            (
                history.bands[0],
                _replace_band(band, rows=changed_rows),
            ),
        )
        row_observation = self._observe(row_mutated, self.trace)
        np.testing.assert_array_equal(
            row_observation.material_release,
            baseline.material_release,
        )
        np.testing.assert_array_equal(
            row_observation.inventory,
            baseline.inventory,
        )
        self.assertGreater(
            row_observation.history.maximum_row_surface_cache_error,
            0.39,
        )
        self.assertLessEqual(
            row_observation.history.maximum_history_seam_error,
            2.0e-16,
        )

        surface_band = _change_boundary_midpoint(
            band, boundary="current", segment=1, delta=0.2
        )
        surface_mutated = MaterialWakeHistory(
            history.history_id, (history.bands[0], surface_band)
        )
        surface_observation = self._observe(
            surface_mutated, self.trace
        )
        self.assertGreater(
            abs(
                surface_observation.material_release[3]
                - baseline.material_release[3]
            ),
            0.19,
        )
        self.assertGreater(
            surface_observation.history.maximum_row_surface_cache_error,
            0.19,
        )

    def test_rigid_material_update_preserves_inventory(self):
        history = _history(
            self.body_edge,
            (self.zero, 0.37 * self.trace, self.trace),
        )
        baseline = self._observe(history, self.trace)
        axis = np.array((0.2, -0.4, 0.7))
        axis /= np.linalg.norm(axis)
        cross = np.array(
            (
                (0.0, -axis[2], axis[1]),
                (axis[2], 0.0, -axis[0]),
                (-axis[1], axis[0], 0.0),
            )
        )
        angle = 0.63
        rotation = (
            np.cos(angle) * np.eye(3)
            + (1.0 - np.cos(angle)) * np.outer(axis, axis)
            + np.sin(angle) * cross
        )
        translation = np.array((0.3, -0.2, 0.4))
        moved = MaterialWakeHistory(
            history.history_id,
            tuple(
                band.material_update(
                    band.surface.vertices @ rotation.T + translation
                )
                for band in history.bands
            ),
        )
        observation = self._observe(moved, self.trace)
        np.testing.assert_array_equal(
            observation.material_release,
            baseline.material_release,
        )
        np.testing.assert_array_equal(
            observation.canonical_release,
            baseline.canonical_release,
        )
        np.testing.assert_array_equal(
            observation.inventory, baseline.inventory
        )

    def test_reversed_negative_attachment_roundtrip_is_asymmetric_safe(self):
        forward = _history(
            self.body_edge,
            (self.zero, 0.37 * self.trace, self.trace),
        )
        reverse = _history(
            self.body_edge,
            (self.zero, 0.37 * self.trace, self.trace),
            reverse=True,
        )
        reversed_attachment = MaterialWakeCutAttachment(
            self.topology.ordered_cut_vertex_indices[::-1],
            -1,
        )
        forward_observation = self._observe(forward, self.trace)
        reverse_observation = self._observe(
            reverse,
            self.trace,
            attachment=reversed_attachment,
        )
        np.testing.assert_allclose(
            reverse_observation.canonical_release,
            forward_observation.canonical_release,
            atol=2.0e-16,
        )
        np.testing.assert_allclose(
            reverse_observation.inventory,
            forward_observation.inventory,
            atol=2.0e-16,
        )

    def test_wrong_attachment_sign_breaks_material_body_compatibility(self):
        history = _history(self.body_edge, (self.zero, self.trace))
        wrong_attachment = MaterialWakeCutAttachment(
            self.topology.ordered_cut_vertex_indices,
            -1,
        )
        observation = self._observe(
            history,
            self.trace,
            attachment=wrong_attachment,
        )
        np.testing.assert_allclose(
            observation.canonical_release,
            -self.trace,
            atol=2.0e-16,
        )
        np.testing.assert_allclose(
            observation.inventory,
            -2.0 * self.trace,
            atol=3.0e-16,
        )
        self.assertGreater(np.max(np.abs(observation.inventory)), 0.6)

    def test_wrong_birth_sign_separates_nonzero_stage_increment(self):
        previous_trace = 0.37 * self.trace
        previous_history = _history(
            self.body_edge, (self.zero, previous_trace)
        )
        current_history = _history(
            self.body_edge, (self.zero, self.trace)
        )
        correct_previous = self._observe(
            previous_history, previous_trace, birth_sign=1
        )
        correct_current = self._observe(
            current_history, self.trace, birth_sign=1
        )
        wrong_previous = self._observe(
            previous_history, previous_trace, birth_sign=-1
        )
        wrong_current = self._observe(
            current_history, self.trace, birth_sign=-1
        )
        np.testing.assert_allclose(
            material_inventory_increment(
                correct_previous, correct_current
            ),
            0.0,
            atol=2.0e-16,
        )
        wrong_increment = material_inventory_increment(
            wrong_previous, wrong_current
        )
        np.testing.assert_allclose(
            wrong_increment,
            -2.0 * (self.trace - previous_trace),
            atol=3.0e-16,
        )
        self.assertGreater(np.max(np.abs(wrong_increment)), 0.1)

    def test_antisymmetric_local_defect_cannot_hide_in_scalar_sum(self):
        history = _history(self.body_edge, (self.zero, self.trace))
        first = _change_boundary_midpoint(
            history.bands[0],
            boundary="current",
            segment=0,
            delta=0.15,
        )
        second = _change_boundary_midpoint(
            first,
            boundary="current",
            segment=2,
            delta=-0.15,
        )
        defective = self._observe(
            MaterialWakeHistory(history.history_id, (second,)),
            self.trace,
        )
        self.assertAlmostEqual(float(np.sum(defective.inventory)), 0.0)
        self.assertGreater(
            float(np.max(np.abs(defective.inventory))), 0.14
        )

    def test_missing_attachment_fails_closed(self):
        history = _history(self.body_edge, (self.zero, self.trace))
        with self.assertRaisesRegex(
            MaterialAttachmentInventoryError,
            "explicit MaterialWakeCutAttachment",
        ):
            observe_material_attachment_inventory(
                self.topology,
                history,
                global_body_potential=_body_potential(
                    self.topology, self.trace
                ),
                attachment=None,
            )

    def test_nonidentity_nonreversal_permutation_fails_closed(self):
        history = _history(self.body_edge, (self.zero, self.trace))
        invalid_ids = self.topology.ordered_cut_vertex_indices.copy()
        invalid_ids[[1, 2]] = invalid_ids[[2, 1]]
        invalid = MaterialWakeCutAttachment(invalid_ids, 1)
        with self.assertRaisesRegex(
            MaterialAttachmentInventoryError,
            "incompatible with the body cut",
        ):
            self._observe(
                history,
                self.trace,
                attachment=invalid,
            )

    def test_boundary_duplicate_and_surface_seam_are_observable(self):
        history = _history(
            self.body_edge,
            (self.zero, 0.37 * self.trace, self.trace),
        )
        first = history.bands[0]
        face_mu = first.surface.face_mu.copy()
        shared_vertex = first.span_nodes + 1
        boundary_owners = []
        for neighbour in (
            first.span_nodes,
            first.span_nodes + 2,
        ):
            matches = []
            for face_index, face in enumerate(first.surface.faces):
                if shared_vertex in face and neighbour in face:
                    local = int(
                        np.flatnonzero(face == shared_vertex)[0]
                    )
                    matches.append((face_index, local))
            self.assertEqual(len(matches), 1)
            boundary_owners.append(matches[0])
        self.assertNotEqual(
            boundary_owners[0][0],
            boundary_owners[1][0],
        )
        face_mu[boundary_owners[0]] += 0.02
        duplicate_band = _replace_band(first, face_mu=face_mu)
        with self.assertRaisesRegex(
            MaterialAttachmentInventoryError,
            "duplicate mismatch",
        ):
            extract_surface_boundary_trace(
                duplicate_band,
                "current",
                duplicate_tolerance=2.0e-12,
            )

        newer = _change_boundary_midpoint(
            history.bands[1],
            boundary="previous",
            segment=1,
            delta=0.03,
        )
        seam = observe_material_history_surface(
            MaterialWakeHistory(
                history.history_id,
                (history.bands[0], newer),
            )
        )
        self.assertGreater(seam.maximum_history_seam_error, 0.029)

    def test_observation_does_not_mutate_inputs(self):
        history = _history(self.body_edge, (self.zero, self.trace))
        body = _body_potential(self.topology, self.trace)
        body_before = body.copy()
        faces_before = tuple(
            band.surface.face_mu.copy() for band in history.bands
        )
        rows_before = tuple(
            band.potential_jump_rows.copy() for band in history.bands
        )
        self._observe(history, self.trace)
        np.testing.assert_array_equal(body, body_before)
        for band, face_mu, rows in zip(
            history.bands,
            faces_before,
            rows_before,
            strict=True,
        ):
            np.testing.assert_array_equal(
                band.surface.face_mu, face_mu
            )
            np.testing.assert_array_equal(
                band.potential_jump_rows, rows
            )


if __name__ == "__main__":
    unittest.main()
