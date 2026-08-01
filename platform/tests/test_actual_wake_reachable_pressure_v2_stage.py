"""One-path S3e integration tests for S3ai-v2; no 31-history run occurs."""
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
from claim_runtime.actual_wake_reachable_pressure import (  # noqa: E402
    observe_direct_independent_stage,
)
from claim_runtime.classified_p2_cut_topology import (  # noqa: E402
    classified_p2_cut_topology,
)
from claim_runtime.distributed_doublet import (  # noqa: E402
    MaterialWakeBand,
    MaterialWakeHistory,
    QuadraticDoubletSurface,
)
from claim_runtime.material_attachment_inventory import (  # noqa: E402
    material_inventory_increment,
    observe_material_attachment_inventory,
    observe_material_band_surface,
    observe_material_history_surface,
)
from claim_runtime.material_wake_time_march import (  # noqa: E402
    march_actual_boundary_material_wake_explicit_midpoint,
)


def _maximum_abs(value) -> float:
    return float(np.max(np.abs(np.asarray(value, dtype=float)), initial=0.0))


class ActualWakeReachablePressureV2StageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.mesh, cls.upper, cls.lower, cut_edges, endpoints = (
            build_canonical_diamond_wing()
        )
        cls.topology = classified_p2_cut_topology(
            cls.mesh,
            upper_face_indices=cls.upper,
            lower_face_indices=cls.lower,
            cut_edges=cut_edges,
            zero_jump_end_vertices=endpoints,
        )
        cls.attachment = MaterialWakeCutAttachment(
            cls.topology.ordered_cut_vertex_indices,
            1,
        )
        cls.dt = 0.25
        cls.epsilon = 0.0025

        def incident(time: float) -> np.ndarray:
            alpha = (
                0.0
                if time <= 0.0
                else cls.epsilon * np.sin(np.pi * time)
            )
            vector = np.array(
                (np.cos(alpha), 0.0, np.sin(alpha)),
                dtype=float,
            )
            return np.repeat(
                vector[None, :],
                len(cls.mesh.faces),
                axis=0,
            )

        cls.march = (
            march_actual_boundary_material_wake_explicit_midpoint(
                cls.mesh,
                cls.topology,
                incident_velocity_at_time=incident,
                initial_body_cut_jump=np.zeros(
                    len(cls.topology.cut_node_coordinates)
                ),
                time_start=-cls.dt,
                time_end=cls.dt,
                timestep=cls.dt,
                trailing_edge_x=1.0,
                convection_speed=1.0,
                target_quadrature_order=3,
                source_quadrature_order=3,
            )
        )
        cls.prestep = cls.march.steps[0]
        cls.measured = cls.march.steps[1]

    def _inventory(self, solution, birth_sign: int = 1):
        return observe_material_attachment_inventory(
            self.topology,
            solution.wake_history,
            global_body_potential=solution.global_body_potential,
            attachment=self.attachment,
            birth_sign=birth_sign,
        )

    def test_compatible_prestep_and_actual_two_stage_inventory(self) -> None:
        self.assertEqual(self.prestep.time_previous, -self.dt)
        self.assertEqual(self.prestep.time_current, 0.0)
        self.assertEqual(self.measured.time_previous, 0.0)
        previous = self._inventory(self.prestep.full_stage, 1)
        midpoint = self._inventory(self.measured.half_stage, 1)
        current = self._inventory(self.measured.full_stage, 1)
        wrong_previous = self._inventory(
            self.prestep.full_stage, -1
        )
        wrong_midpoint = self._inventory(
            self.measured.half_stage, -1
        )
        wrong_current = self._inventory(
            self.measured.full_stage, -1
        )
        first = material_inventory_increment(previous, midpoint)
        second = material_inventory_increment(midpoint, current)
        wrong_first = material_inventory_increment(
            wrong_previous, wrong_midpoint
        )
        wrong_second = material_inventory_increment(
            wrong_midpoint, wrong_current
        )
        correct = max(_maximum_abs(first), _maximum_abs(second))
        wrong = max(
            _maximum_abs(wrong_first),
            _maximum_abs(wrong_second),
        )
        self.assertLessEqual(correct, 2.0e-11)
        self.assertGreater(
            wrong,
            1.0e6 * max(correct, np.finfo(float).eps),
        )
        self.assertLessEqual(
            _maximum_abs(previous.inventory),
            2.0e-11,
        )

    def test_stage_pressure_uses_seven_dimensional_exact_spd_mass(self) -> None:
        observation = observe_direct_independent_stage(
            self.measured.half_stage,
            attachment=self.attachment,
            upper_face_indices=self.upper,
            lower_face_indices=self.lower,
            body_and_direct_w_quadrature_order=3,
            pressure_line_quadrature_order=12,
        )
        mass = observation.active_mass
        self.assertEqual(observation.active_trace.shape, (7,))
        self.assertEqual(mass.shape, (7, 7))
        self.assertTrue(np.array_equal(mass, mass.T))
        self.assertGreater(float(np.linalg.eigvalsh(mass)[0]), 0.0)
        self.assertLessEqual(
            observation.stored_bie_backward_error,
            2.0e-11,
        )
        self.assertLessEqual(
            observation.stored_material_body_trace_abs_residual,
            2.0e-11,
        )

    def test_observers_leave_actual_stage_and_old_history_immutable(self) -> None:
        solution = self.measured.full_stage
        phi = solution.global_body_potential.copy()
        histories = tuple(
            (
                band.surface.vertices.copy(),
                band.surface.face_mu.copy(),
                band.potential_jump_rows.copy(),
            )
            for band in solution.wake_history.bands
        )
        self._inventory(solution)
        observe_direct_independent_stage(
            solution,
            attachment=self.attachment,
            upper_face_indices=self.upper,
            lower_face_indices=self.lower,
            body_and_direct_w_quadrature_order=3,
            pressure_line_quadrature_order=12,
        )
        np.testing.assert_array_equal(
            solution.global_body_potential, phi
        )
        for band, (vertices, face_mu, rows) in zip(
            solution.wake_history.bands,
            histories,
            strict=True,
        ):
            np.testing.assert_array_equal(
                band.surface.vertices, vertices
            )
            np.testing.assert_array_equal(
                band.surface.face_mu, face_mu
            )
            np.testing.assert_array_equal(
                band.potential_jump_rows, rows
            )

    def test_internal_trace_time_and_geometry_defects_are_observable(self) -> None:
        history = self.measured.full_stage.wake_history
        band = history.bands[-1]
        face_mu = band.surface.face_mu.copy()
        internal_edge = next(
            (
                edge
                for edge, records in band.surface._edge_records().items()
                if len(records) == 2
            ),
            None,
        )
        self.assertIsNotNone(internal_edge)
        records = band.surface._edge_records()[internal_edge]
        owner = records[0][0]
        face = band.surface.faces[owner]
        first_local = int(
            np.flatnonzero(face == internal_edge[0])[0]
        )
        second_local = int(
            np.flatnonzero(face == internal_edge[1])[0]
        )
        midpoint = {
            frozenset((0, 1)): 3,
            frozenset((1, 2)): 4,
            frozenset((0, 2)): 5,
        }[frozenset((first_local, second_local))]
        face_mu[owner, midpoint] += 0.01
        defective_surface = QuadraticDoubletSurface(
            band.surface.vertices,
            band.surface.faces,
            face_mu,
        )
        defective_band = MaterialWakeBand(
            sheet_id=band.sheet_id,
            vortex_family=band.vortex_family,
            time_nodes=band.time_nodes,
            span_nodes=band.span_nodes,
            surface=defective_surface,
            potential_jump_rows=band.potential_jump_rows,
        )
        self.assertGreater(
            observe_material_band_surface(
                defective_band
            ).surface_internal_trace_abs_error,
            0.009,
        )

        if len(history.bands) < 2:
            self.skipTest("one measured step did not create two bands")
        older, newer = history.bands[-2:]
        shifted_time = MaterialWakeBand(
            sheet_id=newer.sheet_id,
            vortex_family=newer.vortex_family,
            time_nodes=newer.time_nodes + 0.125,
            span_nodes=newer.span_nodes,
            surface=newer.surface,
            potential_jump_rows=newer.potential_jump_rows,
        )
        time_report = observe_material_history_surface(
            MaterialWakeHistory(
                "time-gap-history",
                (older, shifted_time),
            )
        )
        self.assertGreater(time_report.maximum_time_gap, 0.124)

        moved_vertices = newer.surface.vertices.copy()
        moved_vertices[: newer.span_nodes, 2] += 0.02
        moved = newer.material_update(moved_vertices)
        geometry_report = observe_material_history_surface(
            MaterialWakeHistory(
                "geometry-gap-history",
                (older, moved),
            )
        )
        self.assertGreater(
            geometry_report.maximum_geometry_gap,
            0.019,
        )


if __name__ == "__main__":
    unittest.main()
