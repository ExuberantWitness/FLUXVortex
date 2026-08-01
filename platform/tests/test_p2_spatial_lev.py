import sys
import unittest
from pathlib import Path

import numpy as np

PLATFORM = Path(__file__).resolve().parents[1]
if str(PLATFORM) not in sys.path:
    sys.path.insert(0, str(PLATFORM))

from claim_runtime.coupled_lesp_dde import (  # noqa: E402
    mirror_halfwing_surface,
)
from claim_runtime.distributed_doublet import (  # noqa: E402
    DistributedDoubletError,
    QuadraticDoubletSurface,
)
from claim_runtime.p2_spatial_lev import (  # noqa: E402
    P2LEVHistory,
    causal_band,
    vectorized_induced_velocity,
)
from claim_runtime.p2_spatial_candidate import (  # noqa: E402
    P2SpatialLEVCandidate,
)


class VectorizedP2VelocityTests(unittest.TestCase):
    def setUp(self):
        vertices = np.array(
            [
                [0.0, 0.1, 0.0],
                [1.0, 0.1, 0.1],
                [0.9, 1.0, 0.2],
                [-0.1, 0.9, 0.05],
            ]
        )
        faces = np.array([[0, 1, 2], [0, 2, 3]])
        # Matching P2 trace on the shared 0--2 edge.
        face_mu = np.array(
            [
                [0.2, -0.1, 0.4, 0.05, 0.3, 0.17],
                [0.2, 0.4, -0.2, 0.17, 0.08, -0.03],
            ]
        )
        self.surface = QuadraticDoubletSurface(
            vertices, faces, face_mu
        )
        self.points = np.array(
            [
                [0.2, 0.3, 0.8],
                [0.7, 0.4, -0.7],
                [1.3, 0.6, 0.5],
            ]
        )

    def test_vectorized_q24_matches_face_target_reference(self):
        expected = self.surface.induced_velocity_line_reduced(
            self.points,
            quadrature_order=24,
            plane_tolerance=1.0e-13,
        )
        actual = vectorized_induced_velocity(
            self.surface,
            self.points,
            quadrature_order=24,
            plane_tolerance=1.0e-13,
        )
        np.testing.assert_allclose(
            actual, expected, rtol=0.0, atol=1.0e-11
        )

    def test_multiple_surfaces_and_mirror_halfwing(self):
        shifted = self.surface.material_update(
            self.surface.vertices + [0.0, 0.0, 0.35]
        )
        surfaces = (self.surface, shifted)
        expected = sum(
            (
                surface.induced_velocity_line_reduced(
                    self.points,
                    quadrature_order=24,
                    plane_tolerance=1.0e-13,
                )
                + mirror_halfwing_surface(
                    surface
                ).induced_velocity_line_reduced(
                    self.points,
                    quadrature_order=24,
                    plane_tolerance=1.0e-13,
                )
            )
            for surface in surfaces
        )
        actual = vectorized_induced_velocity(
            surfaces,
            self.points,
            quadrature_order=24,
            plane_tolerance=1.0e-13,
            mirror_halfwing=True,
        )
        np.testing.assert_allclose(
            actual, expected, rtol=0.0, atol=1.0e-11
        )

    def test_face_batching_is_the_same_face_sum(self):
        surfaces = tuple(
            self.surface.material_update(
                self.surface.vertices + [0.0, 0.0, 0.17 * index]
            )
            for index in range(5)
        )
        unbatched = vectorized_induced_velocity(
            surfaces,
            self.points,
            quadrature_order=16,
            mirror_halfwing=True,
            face_batch_size=10_000,
        )
        batched = vectorized_induced_velocity(
            surfaces,
            self.points,
            quadrature_order=16,
            mirror_halfwing=True,
            face_batch_size=3,
        )
        np.testing.assert_allclose(
            batched, unbatched, rtol=2.0e-14, atol=2.0e-14
        )

    def test_nonfinite_and_on_sheet_targets_fail(self):
        with self.assertRaisesRegex(
            DistributedDoubletError, "finite"
        ):
            vectorized_induced_velocity(
                self.surface, [[np.nan, 0.0, 1.0]]
            )
        point = np.mean(self.surface.vertices[self.surface.faces[0]], axis=0)
        with self.assertRaisesRegex(
            DistributedDoubletError, "on-sheet"
        ):
            vectorized_induced_velocity(
                self.surface,
                [point],
                quadrature_order=24,
            )


class P2LEVHistoryTests(unittest.TestCase):
    def setUp(self):
        self.span_edges = np.linspace(0.0, 1.0, 5)
        self.edge0 = np.column_stack(
            (
                np.zeros(len(self.span_edges)),
                self.span_edges,
                np.zeros(len(self.span_edges)),
            )
        )
        self.midpoints = 0.5 * (
            self.span_edges[:-1] + self.span_edges[1:]
        )

    def make_band(self, index):
        previous = self.edge0 + [0.1 * index, 0.0, 0.02 * index]
        current = self.edge0 + [
            0.1 * (index + 1),
            0.0,
            0.02 * (index + 1),
        ]
        q_prev = (1.0 + 0.1 * index) * (1.0 - self.midpoints**2)
        q_now = (1.0 + 0.1 * (index + 1)) * (
            1.0 - self.midpoints**2
        )
        q_mid = 0.5 * (q_prev + q_now)
        return causal_band(
            sheet_id=f"lev-{index}",
            previous_edge=previous,
            current_edge=current,
            span_edges=self.span_edges,
            time_nodes=[
                0.1 * index,
                0.1 * index + 0.05,
                0.1 * (index + 1),
            ],
            q_prev=q_prev,
            q_mid=q_mid,
            q_now=q_now,
        )

    def test_causal_band_is_continuous_p2_and_history_seams_match(self):
        first = self.make_band(0)
        second = self.make_band(1)
        self.assertTrue(first.surface.continuity_report().compatible)
        history = P2LEVHistory("near-lev", 3)
        self.assertIsNone(history.current_edge)
        self.assertIsNone(history.previous_strip_strength)
        history.append(first)
        history.append(second)
        np.testing.assert_allclose(
            history.bands[0].surface.vertices[
                history.bands[0].span_nodes :
            ],
            history.bands[1].surface.vertices[
                : history.bands[1].span_nodes
            ],
            rtol=0.0,
            atol=0.0,
        )
        np.testing.assert_allclose(
            history.previous_strip_strength,
            second.potential_jump_rows[2, 1::2],
            rtol=0.0,
            atol=0.0,
        )

    def test_heun_moves_geometry_and_freezes_material_strength(self):
        history = P2LEVHistory(
            "near-lev", 3, bands=(self.make_band(0), self.make_band(1))
        )
        before_geometry = [band.surface.vertices.copy() for band in history.bands]
        before_mu = [band.surface.face_mu.copy() for band in history.bands]

        def velocity(points):
            return np.column_stack(
                (
                    0.2 + 0.1 * points[:, 2],
                    np.zeros(len(points)),
                    -0.05 * points[:, 0],
                )
            )

        history.convect_heun(velocity, 0.02)
        self.assertFalse(
            np.array_equal(before_geometry[0], history.bands[0].surface.vertices)
        )
        for expected, band in zip(before_mu, history.bands):
            np.testing.assert_array_equal(expected, band.surface.face_mu)
        np.testing.assert_allclose(
            history.bands[0].surface.vertices[
                history.bands[0].span_nodes :
            ],
            history.bands[1].surface.vertices[
                : history.bands[1].span_nodes
            ],
            rtol=0.0,
            atol=0.0,
        )

    def test_max_bands_records_integral_outflow(self):
        history = P2LEVHistory("near-lev", 1)
        history.append(self.make_band(0))
        history.append(self.make_band(1))
        self.assertEqual(len(history.bands), 1)
        self.assertEqual(len(history.outflow), 1)
        self.assertEqual(history.outflow[0].sheet_id, "lev-0")
        self.assertGreater(history.outflow[0].area, 0.0)
        self.assertTrue(
            np.isfinite(
                history.outflow[0].potential_jump_area_integral
            )
        )

    def test_candidate_window_is_global_across_release_events(self):
        candidate = P2SpatialLEVCandidate(
            ns=4,
            span_edges=self.span_edges,
            u_infinity=8.0,
            dt=0.1,
            lesp_crit=0.2,
            quadrature_order=8,
            max_bands=2,
            mirror_halfwing=True,
        )
        first = candidate._new_history()
        candidate._append_band(first, self.make_band(0))
        candidate.active_history = None
        second = candidate._new_history()
        candidate._append_band(second, self.make_band(1))
        candidate._append_band(second, self.make_band(2))
        self.assertEqual(candidate.band_count, 2)
        self.assertEqual(candidate.outflow_count, 1)
        self.assertEqual(len(first.bands), 0)
        self.assertEqual(first.outflow[0].sheet_id, "lev-0")
        self.assertEqual(
            [band.sheet_id for band in second.bands],
            ["lev-1", "lev-2"],
        )

    def test_broken_strength_seam_fails(self):
        first = self.make_band(0)
        second = self.make_band(1)
        broken = causal_band(
            sheet_id="broken",
            previous_edge=second.surface.vertices[: second.span_nodes],
            current_edge=second.surface.vertices[second.span_nodes :],
            span_edges=self.span_edges,
            time_nodes=second.time_nodes,
            q_prev=second.potential_jump_rows[0, 1::2] + 0.1,
            q_mid=second.potential_jump_rows[1, 1::2],
            q_now=second.potential_jump_rows[2, 1::2],
        )
        history = P2LEVHistory("near-lev", 3, bands=(first,))
        with self.assertRaisesRegex(
            DistributedDoubletError, "breaks history"
        ):
            history.append(broken)


if __name__ == "__main__":
    unittest.main()
