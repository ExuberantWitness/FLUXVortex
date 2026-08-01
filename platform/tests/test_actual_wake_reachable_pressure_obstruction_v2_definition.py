"""Definition tests for the fail-closed S3ai-v2.2 31-history runner.

No test in this file executes a physical S3e march.  Marcher and observer
spies verify path freshness and typed quadrature propagation.
"""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np


PLATFORM = Path(__file__).resolve().parents[1]
if str(PLATFORM) not in sys.path:
    sys.path.insert(0, str(PLATFORM))

import actual_wake_reachable_pressure_obstruction_v2_guard as guard  # noqa: E402


class ActualWakeReachablePressureObstructionV2DefinitionTests(
    unittest.TestCase
):
    @classmethod
    def setUpClass(cls) -> None:
        cls.contract = guard._load_frozen_contract()

    def test_frozen_registry_has_exact_preregistered_accounting(self) -> None:
        report = guard.frozen_registry_report()
        accounting = report["accounting"]
        self.assertEqual(report["version"], "S3ai-v2.2")
        self.assertEqual(
            report["definition_chain"]["S3ai-v2"]["sha256"],
            "8345035356d300d4154ac276fc54b8ab27b6e83704cb4444436dbc0c2c59c75b",
        )
        self.assertEqual(
            report["definition_chain"]["S3ai-v2.1"]["sha256"],
            "e751381942ec7c0cac8ea055c6aad1a5756643b85992bc04f4bee88339b563b4",
        )
        self.assertEqual(
            report["definition_chain"]["S3ai-v2.2"]["sha256"],
            "3d662a69c1da80a1452b6b05c67107b188070871bb1b594977467ab7384e0b27",
        )
        self.assertEqual(len(report["case_names"]), 31)
        self.assertEqual(len(set(report["case_names"])), 31)
        self.assertEqual(accounting["histories"], 31)
        self.assertEqual(accounting["measurement_steps"], 380)
        self.assertEqual(accounting["compatible_presteps"], 31)
        self.assertEqual(accounting["marcher_steps"], 411)
        self.assertEqual(accounting["half_full_solves"], 822)
        self.assertEqual(accounting["observed_stages"], 791)

    def test_public_run_fails_before_mesh_or_solver(self) -> None:
        self.assertFalse(
            self.contract["decision"]["formal_execution_allowed"]
        )
        with (
            patch.object(
                guard,
                "build_canonical_diamond_wing",
                side_effect=AssertionError(
                    "fail-closed runner must not construct a mesh"
                ),
            ),
            patch.object(
                guard,
                "march_actual_boundary_material_wake_explicit_midpoint",
                side_effect=AssertionError(
                    "fail-closed runner must not invoke S3e"
                ),
            ),
            self.assertRaisesRegex(
                guard.ReachablePressureV2GuardError,
                "formal_execution_allowed=false",
            ),
        ):
            guard.run_preregistered_observation()

    def test_all_paths_are_fresh_and_q_is_forwarded_to_both_layers(
        self,
    ) -> None:
        meshes = []
        topologies = []
        marches = []
        observations = []

        def fake_march(mesh, topology, **kwargs):
            record = {
                "mesh": mesh,
                "topology": topology,
                "kwargs": kwargs,
            }
            meshes.append(mesh)
            topologies.append(topology)
            marches.append(record)
            return record

        def fake_observer(march, **kwargs):
            observations.append({"march": march, "kwargs": kwargs})
            return kwargs["case"].name

        def execute(case, contract):
            return guard._execute_history_case(
                case,
                contract,
                _marcher=fake_march,
                _observer=fake_observer,
            )

        collected = guard._collect_frozen_histories(
            self.contract,
            case_executor=execute,
        )
        self.assertEqual(len(collected), 31)
        self.assertEqual(len(meshes), 31)
        self.assertEqual(len({id(mesh) for mesh in meshes}), 31)
        self.assertEqual(
            len({id(topology) for topology in topologies}), 31
        )
        for case, march, observation in zip(
            guard.frozen_history_cases(),
            marches,
            observations,
            strict=True,
        ):
            q = case.quadrature_order
            kwargs = march["kwargs"]
            self.assertEqual(kwargs["target_quadrature_order"], q)
            self.assertEqual(kwargs["source_quadrature_order"], q)
            self.assertEqual(kwargs["time_start"], -case.timestep)
            self.assertEqual(kwargs["time_end"], 1.0)
            self.assertEqual(kwargs["timestep"], case.timestep)
            self.assertTrue(
                np.array_equal(
                    kwargs["initial_body_cut_jump"],
                    np.zeros_like(kwargs["initial_body_cut_jump"]),
                )
            )
            observer_kwargs = observation["kwargs"]
            self.assertEqual(
                observer_kwargs[
                    "body_and_direct_w_quadrature_order"
                ],
                q,
            )
            self.assertEqual(
                observer_kwargs["pressure_line_quadrature_order"],
                12,
            )
            self.assertIs(observation["march"], march)

    def test_incident_prestep_is_exactly_zero_and_physical_sign_is_kept(
        self,
    ) -> None:
        negative = guard._incident_history(
            -0.125, epsilon_signed=0.01, face_count=2
        )
        zero = guard._incident_history(
            0.0, epsilon_signed=-0.01, face_count=2
        )
        positive = guard._incident_history(
            0.5, epsilon_signed=0.01, face_count=2
        )
        minus = guard._incident_history(
            0.5, epsilon_signed=-0.01, face_count=2
        )
        expected_zero = np.repeat(
            np.array(((1.0, 0.0, 0.0),)), 2, axis=0
        )
        np.testing.assert_array_equal(negative, expected_zero)
        np.testing.assert_array_equal(zero, expected_zero)
        self.assertGreater(positive[0, 2], 0.0)
        self.assertLess(minus[0, 2], 0.0)

    def test_synthetic_31_case_aggregate_is_complete_json_dict(
        self,
    ) -> None:
        mass = np.eye(7)
        base = np.array((0.2, 0.3, 0.5, 0.8, 0.5, 0.3, 0.2))
        epsilon_shape = np.array(
            (0.1, 0.2, 0.4, 0.7, 0.4, 0.2, 0.1)
        )
        timestep_shape = np.array(
            (0.7, 0.4, 0.2, 0.1, 0.2, 0.4, 0.7)
        )
        quadrature_shape = np.array(
            (-0.3, -0.2, 0.1, 0.3, 0.1, -0.2, -0.3)
        )
        diagnostics = {
            "material_inventory_increment_abs_max": 1.0e-15,
            "material_inventory_abs_max": 1.0e-15,
            "wrong_birth_inventory_increment_abs_max": 1.0e-2,
            "wrong_attachment_inventory_increment_abs_max": 1.0e-2,
            "wrong_attachment_inventory_abs_max": 1.0e-2,
        }
        observations = {}
        for case in guard.frozen_history_cases():
            if case.epsilon_signed == 0.0:
                window = np.zeros(7)
            else:
                tangent = (
                    base
                    + case.epsilon_signed**2 * epsilon_shape
                    + case.timestep**2 * timestep_shape
                    + case.quadrature_order ** (-4)
                    * quadrature_shape
                )
                window = case.epsilon_signed * tangent
            steps = np.zeros((case.measurement_steps, 7))
            steps[0] = window
            stage_count = case.observed_stages
            stage_times = np.concatenate(
                (
                    np.array((0.0,)),
                    np.repeat(
                        np.arange(
                            0.5 * case.timestep,
                            1.0 + 0.25 * case.timestep,
                            0.5 * case.timestep,
                        ),
                        1,
                    )[: stage_count - 1],
                )
            )
            measurement_times = np.array(
                [
                    (
                        index * case.timestep,
                        (index + 0.5) * case.timestep,
                        (index + 1.0) * case.timestep,
                    )
                    for index in range(case.measurement_steps)
                ]
            )
            stage_arrays = guard.TypedStageArrays(
                stage_times=stage_times,
                stage_roles=(
                    "entrance_prestep_full",
                    *(
                        role
                        for _ in range(case.measurement_steps)
                        for role in (
                            "measured_midpoint",
                            "measured_full",
                        )
                    ),
                ),
                canonical_material_current_trace=np.zeros(
                    (stage_count, 9)
                ),
                body_cut_trace=np.zeros((stage_count, 9)),
                canonical_material_release=np.zeros((stage_count, 9)),
                representation_inventory=np.zeros((stage_count, 9)),
                stored_weak_pressure=np.zeros((stage_count, 7)),
                direct_weak_pressure=np.zeros((stage_count, 7)),
                measurement_times=measurement_times,
            )
            observations[case.name] = guard.CompatibleHistoryObservation(
                case=case,
                stored_window_residual=window,
                direct_window_residual=window,
                stored_step_residuals=steps,
                direct_step_residuals=steps,
                mass_active=mass,
                stage_arrays=stage_arrays,
                diagnostics=diagnostics,
                checks={"synthetic_definition_guard": True},
                observed_stage_count=case.observed_stages,
            )

        result = guard.aggregate_frozen_histories(
            observations, self.contract
        )
        self.assertEqual(len(result["cases"]), 31)
        self.assertEqual(
            result["execution_accounting"]["measurement_steps"], 380
        )
        self.assertTrue(all(result["checks"].values()))
        self.assertEqual(
            result["stage_decision"],
            "FIXED-SPACE REACHABLE FIRST-ORDER OBSTRUCTION WITNESS",
        )
        encoded = json.dumps(
            result, allow_nan=False, sort_keys=True
        )
        self.assertIn('"production_activation_allowed": false', encoded)

        # A q-contracting initializer may still converge to a nonzero even
        # trace.  v2.2 requires the complete zero-target interval, not only
        # a generic parity-convergence pass.
        bad = dict(observations)
        even = np.array((1.0, -2.0, 3.0, 4.0, 3.0, -2.0, 1.0))
        odd = np.array((1.0, -2.0, 3.0, 0.0, -3.0, 2.0, -1.0))
        for name, factor in (
            ("Q8_plus", 4.0),
            ("Q10_plus", 2.0),
            ("A_plus", 1.0),
        ):
            observation = bad[name]
            traces = (
                observation.stage_arrays
                .canonical_material_current_trace.copy()
            )
            traces[0, 1:-1] = even + 0.125 * factor * odd
            stages = replace(
                observation.stage_arrays,
                canonical_material_current_trace=traces,
            )
            bad[name] = replace(observation, stage_arrays=stages)
        bad_result = guard.aggregate_frozen_histories(
            bad, self.contract
        )
        self.assertFalse(
            bad_result["checks"][
                "matched_stage_projected_q_families"
            ]
        )
        self.assertEqual(
            bad_result["stage_decision"], "PROTOCOL-NO-GO"
        )


if __name__ == "__main__":
    unittest.main()
