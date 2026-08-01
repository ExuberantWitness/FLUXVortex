"""Unit tests for S3ai-v2 stored-phi read-only pressure primitives.

These tests exercise one compatible stage only.  They do not execute the
frozen 31-history protocol or make a physical obstruction classification.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np


PLATFORM = Path(__file__).resolve().parents[1]
if str(PLATFORM) not in sys.path:
    sys.path.insert(0, str(PLATFORM))

from actual_boundary_3d_cut_wake_junction_guard import (  # noqa: E402
    build_canonical_diamond_wing,
)
from claim_runtime import actual_wake_reachable_pressure as pressure_runtime  # noqa: E402
from claim_runtime import actual_wake_kutta_compatibility as kutta  # noqa: E402
from claim_runtime.actual_boundary_body_wake import (  # noqa: E402
    MaterialWakeCutAttachment,
    solve_actual_boundary_body_wake_p2,
)
from claim_runtime.actual_wake_reachable_pressure import (  # noqa: E402
    canonical_material_trace,
    material_current_trace_from_surface,
    observe_direct_independent_stage,
)
from claim_runtime.classified_p2_cut_topology import (  # noqa: E402
    classified_p2_cut_topology,
)
from claim_runtime.actual_wake_kutta_closure_roles import (  # noqa: E402
    cut_role_operators,
)
from claim_runtime.distributed_doublet import (  # noqa: E402
    MaterialWakeBand,
    MaterialWakeHistory,
    newborn_material_wake_band,
)
from claim_runtime.reachable_pressure_uncertainty import (  # noqa: E402
    dual_mass_norm,
    floating_plateau_floor,
)


class ActualWakeReachablePressureV2PrimitiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        (
            cls.mesh,
            cls.upper,
            cls.lower,
            cut_edges,
            endpoints,
        ) = build_canonical_diamond_wing()
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
        current_edge = cls.mesh.vertices[
            cls.topology.ordered_cut_vertex_indices
        ].copy()
        previous_edge = current_edge.copy()
        previous_edge[:, 0] += 0.5
        span = cls.topology.cut_node_coordinates[:, 1]
        shape = 1.0 - span**2
        prescribed = newborn_material_wake_band(
            sheet_id="s3ai-v2-unit-stage",
            vortex_family="TEV",
            previous_edge=previous_edge,
            current_edge=current_edge,
            time_nodes=np.array((0.0, 0.5, 1.0)),
            potential_jump_rows=np.array(
                (0.10 * shape, 0.15 * shape, np.zeros_like(shape))
            ),
            span_diagonal_pattern="mirror_symmetric",
        )
        history = MaterialWakeHistory(
            "s3ai-v2-unit-history",
            (prescribed,),
        )
        angle = np.deg2rad(4.0)
        incident = np.repeat(
            np.array(((np.cos(angle), 0.0, np.sin(angle)),)),
            len(cls.mesh.faces),
            axis=0,
        )
        cls.order = 3
        cls.solution = solve_actual_boundary_body_wake_p2(
            cls.mesh,
            cls.topology,
            incident_velocity=incident,
            downstream_edge_x=None,
            prescribed_wake_history=history,
            prescribed_wake_attachment=cls.attachment,
            target_quadrature_order=cls.order,
            source_quadrature_order=cls.order,
        )
        cls.material_trace = material_current_trace_from_surface(
            cls.solution,
            cls.attachment,
        )
        with (
            patch.object(
                kutta,
                "build_actual_pressure_kutta_model",
                side_effect=AssertionError(
                    "stored pressure must not build a solve-backed model"
                ),
            ),
            patch.object(
                kutta,
                "solve_independent_wake_body_state",
                side_effect=AssertionError(
                    "stored pressure must not call the independent state solver"
                ),
            ),
            patch.object(
                kutta,
                "evaluate_pressure_kutta",
                side_effect=AssertionError(
                    "stored pressure must not call the solve-backed evaluator"
                ),
            ),
        ):
            cls.observation = observe_direct_independent_stage(
                cls.solution,
                attachment=cls.attachment,
                upper_face_indices=cls.upper,
                lower_face_indices=cls.lower,
                body_and_direct_w_quadrature_order=cls.order,
                pressure_line_quadrature_order=3,
            )

    def test_surface_material_trace_and_stored_phi_are_primary(self) -> None:
        expected_canonical = canonical_material_trace(
            self.solution,
            self.attachment,
            self.material_trace,
        )
        observation = self.observation
        self.assertTrue(
            np.array_equal(
                observation.material_current_trace,
                self.material_trace,
            )
        )
        self.assertTrue(
            np.array_equal(
                observation.canonical_material_trace,
                expected_canonical,
            )
        )
        self.assertTrue(
            np.array_equal(
                observation.stored_state.body_potential,
                self.solution.global_body_potential,
            )
        )
        self.assertLessEqual(
            observation.stored_bie_backward_error,
            2.0e-11,
        )
        self.assertLessEqual(
            observation.direct_bie_backward_error,
            2.0e-11,
        )
        self.assertLessEqual(
            observation.direct_w_factorization_abs_residual,
            5.0e-11,
        )
        self.assertEqual(observation.direct_w_rank_deficiency, 0)
        self.assertEqual(
            observation.body_and_direct_w_quadrature_order,
            self.order,
        )
        self.assertEqual(
            observation.direct_assembly.target_quadrature_order,
            self.order,
        )
        self.assertEqual(
            observation.direct_assembly.source_quadrature_order,
            self.order,
        )
        self.assertEqual(
            observation.body_matrix_condition_norm,
            "spectral_2",
        )

    def test_value_only_stored_pressure_remains_solve_free(self) -> None:
        observation = self.observation
        operators = cut_role_operators(self.topology)
        model = pressure_runtime._value_only_pressure_model(
            self.solution,
            observation.direct_system,
            operators,
            upper_face_indices=self.upper,
            lower_face_indices=self.lower,
            line_quadrature_order=3,
        )
        state = pressure_runtime._typed_body_state(
            model,
            observation.active_trace,
            self.solution.global_body_potential,
            observation.stored_bie_residual,
        )
        with patch.object(
            np.linalg,
            "solve",
            side_effect=AssertionError(
                "value-only stored pressure must remain solve-free"
            ),
        ):
            pressure = pressure_runtime._weak_active_pressure_from_state(
                model,
                state,
            )
        self.assertTrue(np.array_equal(pressure, observation.weak_pressure))

    def test_surface_extractor_does_not_read_material_row_cache(self) -> None:
        newest = self.solution.wake_history.bands[-1]
        altered_rows = newest.potential_jump_rows.copy()
        altered_rows[-1] += np.linspace(
            0.0,
            0.25,
            altered_rows.shape[1],
        )
        altered_band = MaterialWakeBand(
            sheet_id=newest.sheet_id,
            vortex_family=newest.vortex_family,
            time_nodes=newest.time_nodes,
            span_nodes=newest.span_nodes,
            surface=newest.surface,
            potential_jump_rows=altered_rows,
        )
        altered_history = MaterialWakeHistory(
            self.solution.wake_history.history_id,
            (altered_band,),
        )
        altered_solution = replace(
            self.solution,
            wake=altered_band,
            wake_history=altered_history,
            body_cut_jump=(
                self.solution.body_cut_jump
                + np.linspace(1.0, 2.0, len(self.solution.body_cut_jump))
            ),
        )
        extracted = material_current_trace_from_surface(
            altered_solution,
            self.attachment,
        )
        self.assertTrue(np.array_equal(extracted, self.material_trace))
        self.assertFalse(
            np.array_equal(extracted, altered_band.potential_jump_rows[-1])
        )
        self.assertFalse(
            np.array_equal(extracted, altered_solution.body_cut_jump)
        )

    def test_stored_phi_negative_control_separates_primary_and_audit(self) -> None:
        corrupted_phi = self.solution.global_body_potential.copy()
        corrupted_dof = int(self.topology.upper_cut_dofs[3])
        corrupted_phi[corrupted_dof] += 2.0**-10
        corrupted_solution = replace(
            self.solution,
            global_body_potential=corrupted_phi,
        )
        with (
            patch.object(
                kutta,
                "build_actual_pressure_kutta_model",
                side_effect=AssertionError("forbidden solve-backed model"),
            ),
            patch.object(
                kutta,
                "solve_independent_wake_body_state",
                side_effect=AssertionError("forbidden solve-backed observer"),
            ),
            patch.object(
                kutta,
                "evaluate_pressure_kutta",
                side_effect=AssertionError("forbidden solve-backed observer"),
            ),
        ):
            corrupted = observe_direct_independent_stage(
                corrupted_solution,
                attachment=self.attachment,
                upper_face_indices=self.upper,
                lower_face_indices=self.lower,
                body_and_direct_w_quadrature_order=self.order,
                pressure_line_quadrature_order=3,
                material_current_trace=self.material_trace,
            )

        baseline = self.observation
        self.assertGreater(
            corrupted.stored_bie_backward_error,
            2.0e-11,
        )
        self.assertGreaterEqual(
            corrupted.stored_bie_backward_error,
            1.0e3 * baseline.stored_bie_backward_error,
        )
        pressure_change = (
            corrupted.weak_pressure - baseline.weak_pressure
        )
        pressure_round = floating_plateau_floor(
            [baseline.weak_pressure, corrupted.weak_pressure],
            baseline.active_mass,
        )
        self.assertGreater(
            dual_mass_norm(pressure_change, baseline.active_mass),
            1.0e6 * pressure_round,
        )
        self.assertTrue(
            np.array_equal(
                corrupted.direct_state.body_potential,
                baseline.direct_state.body_potential,
            )
        )
        self.assertTrue(
            np.array_equal(
                corrupted.direct_bie_residual,
                baseline.direct_bie_residual,
            )
        )
        self.assertEqual(
            corrupted.direct_bie_backward_error,
            baseline.direct_bie_backward_error,
        )
        self.assertTrue(
            np.array_equal(
                corrupted.direct_weak_pressure,
                baseline.direct_weak_pressure,
            )
        )
        self.assertGreater(
            float(
                np.max(
                    np.abs(corrupted.pressure_cross_observer_difference),
                    initial=0.0,
                )
            ),
            1.0e-10,
        )


if __name__ == "__main__":
    unittest.main()
