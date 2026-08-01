import importlib.util
import math
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


PLATFORM = Path(__file__).resolve().parents[1]
RUNNER = PLATFORM / "run_n26e1b2_xm_coupled_junction_gate.py"
SPEC = importlib.util.spec_from_file_location(
    "run_n26e1b2_xm_coupled_junction_gate",
    RUNNER,
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class XMCoupledJunctionGateRunnerTests(unittest.TestCase):
    def test_frozen_matrix_is_exactly_three_by_three_by_three(self):
        cases = MODULE.frozen_cases()
        self.assertEqual(len(cases), 27)
        self.assertEqual(len({case.case_id for case in cases}), 27)
        self.assertEqual(
            sorted({case.panels_per_side for case in cases}),
            [32, 64, 128],
        )
        self.assertEqual(
            sorted(
                {case.epsilon_over_te_panel for case in cases},
                reverse=True,
            ),
            [1.0 / 4.0, 1.0 / 8.0, 1.0 / 16.0],
        )
        self.assertEqual(
            {case.canonical_state_id for case in cases},
            {
                "side1_dominant",
                "mirror_side2_dominant",
                "symmetric_no_birth",
            },
        )
        first = cases[0]
        self.assertEqual(first.panels_per_side, 32)
        self.assertEqual(first.epsilon_over_te_panel, 1.0 / 4.0)
        self.assertEqual(first.canonical_state_id, "side1_dominant")
        self.assertEqual(
            first.case_id,
            "side1_dominant__p32__eps1of4",
        )

        for invalid in (16, 33, 256, 32.5):
            with self.subTest(panels=invalid):
                with self.assertRaises(ValueError):
                    MODULE.JunctionGateCase(
                        invalid, 1.0 / 4.0, "side1_dominant"
                    )
        with self.assertRaises(ValueError):
            MODULE.JunctionGateCase(
                32, 1.0 / 3.0, "side1_dominant"
            )
        with self.assertRaises(ValueError):
            MODULE.JunctionGateCase(32, 1.0 / 4.0, "other")

    def test_unit_scaling_and_default_artifact_contract_are_frozen(self):
        self.assertEqual(MODULE.CHORD, 1.0)
        self.assertEqual(MODULE.FREESTREAM_SPEED, 1.0)
        self.assertEqual(MODULE.TIME_STEP, 0.01)
        self.assertEqual(
            MODULE.TIME_STEP
            * MODULE.FREESTREAM_SPEED
            / MODULE.CHORD,
            0.01,
        )
        self.assertEqual(
            set(MODULE.METRIC_SCALES),
            {
                "gamma1",
                "gamma2",
                "gamma_g",
                "Gamma_g",
                "Gamma_bound",
                "u_g",
                "delta1",
                "delta2",
                "absolute_forming_angle",
            },
        )
        self.assertTrue(
            str(MODULE.DEFAULT_JSON).endswith(
                "n26e1b2_xm_coupled_junction_result_20260730.json"
            )
        )
        self.assertTrue(
            str(MODULE.DEFAULT_MARKDOWN).endswith(
                "n26e1b2_xm_coupled_junction_result_20260730.md"
            )
        )

    def test_cauchy_score_preserves_sign_and_wraps_only_angle_difference(
        self,
    ):
        signed = MODULE.convergence_score(
            -1.03, -1.0, 1.0
        )
        self.assertEqual(signed["coarse"], -1.03)
        self.assertEqual(signed["fine"], -1.0)
        self.assertAlmostEqual(signed["signed_change"], 0.03)
        self.assertAlmostEqual(signed["score"], 0.03)
        self.assertFalse(signed["angular_difference_wrapped"])

        angular = MODULE.convergence_score(
            math.pi - 0.01,
            -math.pi + 0.01,
            1.0,
            angular=True,
        )
        self.assertAlmostEqual(
            angular["signed_change"], 0.02, delta=2.0e-15
        )
        self.assertLess(angular["score"], 0.01)
        self.assertTrue(angular["angular_difference_wrapped"])

        floor = MODULE.convergence_score(0.001, 0.0, 1.0)
        self.assertEqual(floor["denominator"], 0.02)
        self.assertAlmostEqual(floor["score"], 0.05)
        for args in (
            (math.nan, 1.0, 1.0),
            (1.0, math.inf, 1.0),
            (1.0, 1.0, 0.0),
        ):
            with self.subTest(args=args):
                with self.assertRaises(ValueError):
                    MODULE.convergence_score(*args)

    @staticmethod
    def _synthetic_completed(case):
        panel_term = {32: 8.0e-4, 64: 3.0e-4, 128: 1.0e-4}[
            case.panels_per_side
        ]
        epsilon_term = {
            1.0 / 4.0: 8.0e-4,
            1.0 / 8.0: 3.0e-4,
            1.0 / 16.0: 1.0e-4,
        }[case.epsilon_over_te_panel]
        shift = panel_term + epsilon_term
        n_node = 2 * case.panels_per_side + 1

        if case.canonical_state_id == "side1_dominant":
            angle = 0.05 + 0.1 * shift
            metrics = {
                "gamma1": 1.1 + shift,
                "gamma2": 0.6 + 0.5 * shift,
                "gamma_g": 0.8 + 0.2 * shift,
                "Gamma_g": 0.03 + 0.1 * shift,
                "Gamma_bound": -0.03 - 0.1 * shift,
                "u_g": 1.4 + 0.1 * shift,
                "delta1": 0.1 + 0.1 * shift,
                "delta2": 0.25 - 0.1 * shift,
                "absolute_forming_angle": angle,
            }
            strengths = np.linspace(-0.3, 0.4, n_node) + shift
            direction = [math.cos(angle), math.sin(angle)]
            stage = "forming"
            no_birth = False
            identifiable = True
        elif case.canonical_state_id == "mirror_side2_dominant":
            angle = 0.05 + 0.1 * shift
            metrics = {
                "gamma1": -0.6 - 0.5 * shift,
                "gamma2": -1.1 - shift,
                "gamma_g": -0.8 - 0.2 * shift,
                "Gamma_g": -0.03 - 0.1 * shift,
                "Gamma_bound": 0.03 + 0.1 * shift,
                "u_g": 1.4 + 0.1 * shift,
                "delta1": 0.25 - 0.1 * shift,
                "delta2": 0.1 + 0.1 * shift,
                "absolute_forming_angle": -angle,
            }
            plus = np.linspace(-0.3, 0.4, n_node) + shift
            strengths = -plus[::-1]
            direction = [math.cos(angle), -math.sin(angle)]
            stage = "forming"
            no_birth = False
            identifiable = True
        else:
            metrics = {
                "gamma1": -0.1,
                "gamma2": 0.1,
                "gamma_g": 0.0,
                "Gamma_g": 0.0,
                "Gamma_bound": 0.0,
                "u_g": None,
                "delta1": None,
                "delta2": None,
                "absolute_forming_angle": None,
            }
            strengths = np.linspace(-0.1, 0.1, n_node)
            direction = None
            stage = "no_birth"
            no_birth = True
            identifiable = False

        return {
            **case.as_dict(),
            "status": "completed",
            "stage": stage,
            "no_birth": no_birth,
            "geometry": {
                "panel_count": 2 * case.panels_per_side,
                "trailing_edge_wedge_angle_deg": 20.59,
                "h_TE_over_c": 0.01,
                "epsilon_over_c": (
                    0.01 * case.epsilon_over_te_panel
                ),
                "forming_length_over_c": (
                    0.0 if no_birth else 0.014
                ),
                "forming_direction_body": direction,
                "forming_start_over_c": (
                    None if no_birth else [1.0, 0.0]
                ),
                "forming_end_over_c": (
                    None if no_birth else [1.014, 0.0]
                ),
            },
            "bound_node_strength_over_U": strengths.tolist(),
            "dimensionless_metrics": metrics,
            "current_diagnostics": {
                "maximum_normal_residual_over_U": 1.0e-13,
                "kelvin_residual_over_Uc": 2.0e-13,
                "kutta_residual_over_U": (
                    None if no_birth else 3.0e-13
                ),
                "linear_system_infinity_residual_scaled": 4.0e-13,
                "scaled_system_condition_number_2": 1.0e5,
            },
            "previous_formation_oracle": {
                "state_identifiable": identifiable,
                "delta1_rad": 0.1,
                "delta2_rad": 0.25,
                "sheet_strength_over_U": (
                    0.0 if no_birth else 0.8
                ),
                "circulation_rate_over_U2": (
                    0.0 if no_birth else -1.5
                ),
                "relative_velocity_over_U": (
                    None if no_birth else 1.4
                ),
                "normalized_residuals": {
                    "angle_sum": 1.0e-14,
                    "direction": 1.0e-14,
                    "kutta_strength": 1.0e-14,
                    "circulation_rate": (
                        None if no_birth else 1.0e-14
                    ),
                    "momentum": (
                        None if no_birth else 1.0e-14
                    ),
                },
                "maximum_applicable_normalized_residual": 1.0e-14,
            },
        }

    @classmethod
    def _synthetic_matrix(cls):
        return {
            case.case_id: cls._synthetic_completed(case)
            for case in MODULE.frozen_cases()
        }

    def test_cauchy_evaluator_requires_exact_matrix_and_both_axes(self):
        matrix = self._synthetic_matrix()
        evaluated = MODULE.evaluate_cauchy(matrix)
        self.assertTrue(evaluated["exact_27_case_matrix"])
        self.assertTrue(evaluated["all_cases_completed"])
        self.assertTrue(evaluated["all_scores_finite"])
        self.assertTrue(
            evaluated["all_final_scores_within_tolerance"]
        )
        self.assertTrue(evaluated["all_scores_nonincreasing"])
        self.assertTrue(evaluated["pass"])
        epsilon_entries = [
            entry
            for levels in evaluated["epsilon_axis"].values()
            for metrics in levels.values()
            for entry in metrics.values()
        ]
        panel_entries = [
            entry
            for metrics in evaluated[
                "panel_axis_at_epsilon_1of16"
            ].values()
            for entry in metrics.values()
        ]
        self.assertEqual(len(epsilon_entries), 2 * 3 * 9)
        self.assertEqual(len(panel_entries), 2 * 9)

        missing = dict(matrix)
        missing.pop(next(iter(missing)))
        incomplete = MODULE.evaluate_cauchy(missing)
        self.assertFalse(incomplete["exact_27_case_matrix"])
        self.assertFalse(incomplete["pass"])
        self.assertEqual(len(incomplete["missing_case_ids"]), 1)

        extra = dict(matrix)
        extra["unregistered"] = {"status": "completed"}
        unexpected = MODULE.evaluate_cauchy(extra)
        self.assertFalse(unexpected["exact_27_case_matrix"])
        self.assertEqual(
            unexpected["unexpected_case_ids"], ["unregistered"]
        )

    def test_cauchy_evaluator_fails_two_percent_and_monotonic_gates(self):
        matrix = self._synthetic_matrix()
        final_case = MODULE.JunctionGateCase(
            128, 1.0 / 16.0, "side1_dominant"
        )
        matrix[final_case.case_id]["dimensionless_metrics"][
            "gamma1"
        ] = 1.2
        evaluated = MODULE.evaluate_cauchy(matrix)
        entry = evaluated["panel_axis_at_epsilon_1of16"][
            "side1_dominant"
        ]["gamma1"]
        self.assertGreater(
            entry["middle_to_fine"]["score"],
            MODULE.FINAL_SCORE_TOLERANCE,
        )
        self.assertFalse(entry["pass"])
        self.assertFalse(evaluated["pass"])

        matrix = self._synthetic_matrix()
        case32 = MODULE.JunctionGateCase(
            32, 1.0 / 16.0, "side1_dominant"
        )
        case64 = MODULE.JunctionGateCase(
            64, 1.0 / 16.0, "side1_dominant"
        )
        case128 = MODULE.JunctionGateCase(
            128, 1.0 / 16.0, "side1_dominant"
        )
        for case, value in (
            (case32, 1.0),
            (case64, 1.001),
            (case128, 1.003),
        ):
            matrix[case.case_id]["dimensionless_metrics"][
                "gamma2"
            ] = value
        evaluated = MODULE.evaluate_cauchy(matrix)
        entry = evaluated["panel_axis_at_epsilon_1of16"][
            "side1_dominant"
        ]["gamma2"]
        self.assertLessEqual(
            entry["middle_to_fine"]["score"],
            MODULE.FINAL_SCORE_TOLERANCE,
        )
        self.assertFalse(entry["score_nonincreasing"])
        self.assertFalse(entry["pass"])

    def test_mirror_and_no_birth_gates_are_recomputed_from_records(self):
        matrix = self._synthetic_matrix()
        mirror = MODULE.evaluate_mirror_contract(matrix)
        no_birth = MODULE.evaluate_no_birth_contract(matrix)
        self.assertTrue(mirror["all_nine_pairs_pass"])
        self.assertTrue(no_birth["all_nine_explicit_no_birth"])

        negative = MODULE.JunctionGateCase(
            64, 1.0 / 8.0, "mirror_side2_dominant"
        )
        matrix[negative.case_id]["bound_node_strength_over_U"][3] += (
            1.0e-4
        )
        mirror = MODULE.evaluate_mirror_contract(matrix)
        self.assertFalse(mirror["pass"])
        self.assertFalse(
            mirror["entries"]["p64__eps1of8"]["pass"]
        )

        matrix = self._synthetic_matrix()
        symmetric = MODULE.JunctionGateCase(
            32, 1.0 / 4.0, "symmetric_no_birth"
        )
        matrix[symmetric.case_id]["geometry"][
            "forming_direction_body"
        ] = [1.0, 0.0]
        no_birth = MODULE.evaluate_no_birth_contract(matrix)
        self.assertFalse(no_birth["pass"])
        self.assertFalse(
            no_birth["entries"][symmetric.case_id]["pass"]
        )

    def test_algebra_gate_recomputes_thresholds_and_condition_number(self):
        case = MODULE.JunctionGateCase(
            32, 1.0 / 8.0, "side1_dominant"
        )
        record = self._synthetic_completed(case)
        self.assertTrue(MODULE._completed_case_algebra_pass(record))
        record["current_diagnostics"][
            "maximum_normal_residual_over_U"
        ] = 2.0e-10
        self.assertFalse(MODULE._completed_case_algebra_pass(record))

        record = self._synthetic_completed(case)
        record["current_diagnostics"][
            "scaled_system_condition_number_2"
        ] = 1.0e13
        self.assertFalse(MODULE._completed_case_algebra_pass(record))

        record = self._synthetic_completed(case)
        record["previous_formation_oracle"][
            "normalized_residuals"
        ]["momentum"] = 2.0e-12
        self.assertFalse(MODULE._completed_case_algebra_pass(record))

    def test_execute_gate_preserves_a_failed_cell_and_cannot_drop_it(self):
        failed_id = MODULE.JunctionGateCase(
            128, 1.0 / 16.0, "side1_dominant"
        ).case_id

        def fake_run(case):
            if case.case_id == failed_id:
                raise RuntimeError("synthetic singular junction")
            return self._synthetic_completed(case)

        with patch.object(MODULE, "run_gate_case", side_effect=fake_run):
            with patch("builtins.print"):
                result = MODULE.execute_gate()
        self.assertTrue(result["gate"]["exact_27_case_matrix"])
        self.assertFalse(result["gate"]["all_cases_completed"])
        self.assertEqual(result["gate"]["verdict"], "NO-GO")
        self.assertEqual(result["cases"][failed_id]["status"], "failed")
        self.assertEqual(
            result["cases"][failed_id]["error_type"], "RuntimeError"
        )
        self.assertIn(
            "singular junction", result["cases"][failed_id]["error"]
        )

    def test_real_single_cells_expose_dimensionless_and_no_birth_contracts(
        self,
    ):
        forming = MODULE.run_gate_case(
            MODULE.JunctionGateCase(
                32, 1.0 / 8.0, "side1_dominant"
            )
        )
        self.assertEqual(forming["status"], "completed")
        self.assertEqual(forming["stage"], "forming")
        self.assertFalse(forming["no_birth"])
        self.assertEqual(
            set(forming["dimensionless_metrics"]),
            set(MODULE.METRIC_SCALES),
        )
        self.assertEqual(
            len(forming["bound_node_strength_over_U"]), 65
        )
        self.assertTrue(
            MODULE._completed_case_algebra_pass(forming)
        )

        symmetric = MODULE.run_gate_case(
            MODULE.JunctionGateCase(
                32, 1.0 / 8.0, "symmetric_no_birth"
            )
        )
        self.assertEqual(symmetric["stage"], "no_birth")
        self.assertTrue(symmetric["no_birth"])
        self.assertIsNone(
            symmetric["geometry"]["forming_direction_body"]
        )
        self.assertIsNone(
            symmetric["dimensionless_metrics"]["u_g"]
        )
        self.assertTrue(
            MODULE._completed_case_algebra_pass(symmetric)
        )

    def test_runner_has_no_downstream_reader_import(self):
        source = RUNNER.read_text(encoding="utf-8")
        for forbidden in (
            "from claim_runtime.svi_dw_pressure",
            "import _v2_robo",
            "s6_sweep_v41",
            "score_riziotis_fig12",
            "plot_3way",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
