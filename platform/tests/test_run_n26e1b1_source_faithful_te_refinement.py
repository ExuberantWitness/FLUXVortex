import importlib.util
import math
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np


PLATFORM = Path(__file__).resolve().parents[1]
RUNNER = PLATFORM / "run_n26e1b1_source_faithful_te_refinement.py"
SPEC = importlib.util.spec_from_file_location(
    "run_n26e1b1_source_faithful_te_refinement",
    RUNNER,
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class SourceFaithfulTERefinementRunnerTests(unittest.TestCase):
    def test_only_preregistered_spatial_levels_are_constructible(self):
        cases = MODULE.frozen_cases()
        self.assertEqual(
            [case.panels_per_side for case in cases],
            [64, 128, 256],
        )
        self.assertEqual(
            [case.ramp_steps for case in cases],
            [32, 32, 32],
        )
        self.assertEqual(
            [case.core_radius_over_c for case in cases],
            [0.02, 0.02, 0.02],
        )
        self.assertEqual(
            [case.case_id for case in cases],
            [
                "p64_n32_core0.02",
                "p128_n32_core0.02",
                "p256_n32_core0.02",
            ],
        )
        for invalid in (32, 65, 512, 64.5):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    MODULE.RefinementCase(invalid)

    def test_motion_time_and_core_are_frozen(self):
        alpha_0, rate_0 = MODULE.half_cosine_pitch(0.0)
        alpha_mid, rate_mid = MODULE.half_cosine_pitch(
            0.5 * MODULE.RAMP_DURATION
        )
        alpha_end, rate_end = MODULE.half_cosine_pitch(
            MODULE.RAMP_DURATION
        )
        self.assertEqual(alpha_0, 0.0)
        self.assertAlmostEqual(rate_0, 0.0, delta=1.0e-15)
        self.assertAlmostEqual(
            alpha_mid,
            0.5 * MODULE.FINAL_ALPHA_RAD,
            delta=1.0e-15,
        )
        self.assertGreater(rate_mid, 0.0)
        self.assertAlmostEqual(
            alpha_end,
            MODULE.FINAL_ALPHA_RAD,
            delta=1.0e-15,
        )
        self.assertAlmostEqual(rate_end, 0.0, delta=1.0e-15)
        motion = MODULE.pitching_kinematics(
            0.5 * MODULE.RAMP_DURATION
        )
        self.assertAlmostEqual(motion.angle_rad, -alpha_mid)
        self.assertAlmostEqual(
            motion.angular_velocity_rad_s,
            -rate_mid,
        )
        self.assertEqual(MODULE.RAMP_STEPS, 32)
        self.assertEqual(MODULE.CORE_RADIUS_OVER_C, 0.02)
        self.assertAlmostEqual(
            MODULE.RAMP_DURATION / MODULE.RAMP_STEPS,
            0.0125,
        )

    def test_score_uses_exact_preregistered_denominator_floor(self):
        relative = MODULE.convergence_score(
            middle_value=1.019,
            fine_value=1.0,
            physical_scale=10.0,
        )
        self.assertEqual(relative["mode"], "relative_to_fine")
        self.assertEqual(relative["denominator"], 1.0)
        self.assertAlmostEqual(relative["score"], 0.019)

        near_zero = MODULE.convergence_score(
            middle_value=0.104,
            fine_value=0.1,
            physical_scale=10.0,
        )
        self.assertEqual(near_zero["mode"], "physical_scale_floor")
        self.assertEqual(near_zero["denominator_floor"], 0.2)
        self.assertEqual(near_zero["denominator"], 0.2)
        self.assertAlmostEqual(near_zero["score"], 0.02)

        for invalid in (
            (math.nan, 1.0, 1.0),
            (1.0, math.inf, 1.0),
            (1.0, 1.0, 0.0),
        ):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    MODULE.convergence_score(*invalid)

    def test_metric_contract_covers_every_preregistered_quantity(self):
        self.assertEqual(len(MODULE.LOCAL_BIRTH_SCALES), 9)
        self.assertEqual(len(MODULE.GLOBAL_WAKE_SCALES), 6)
        self.assertEqual(len(MODULE.ALGEBRAIC_SCALES), 5)
        self.assertEqual(
            MODULE.LOCAL_BIRTH_SCALES[
                "newborn_sheet_strength_ccw"
            ],
            MODULE.FREESTREAM_SPEED,
        )
        self.assertEqual(
            MODULE.LOCAL_BIRTH_SCALES[
                "newborn_circulation_ccw"
            ],
            MODULE.FREESTREAM_SPEED * MODULE.CHORD,
        )
        self.assertEqual(
            MODULE.GLOBAL_WAKE_SCALES["wake_first_moment_x"],
            MODULE.FREESTREAM_SPEED * MODULE.CHORD**2,
        )

    def test_birth_and_wake_extractor_matches_independent_ledger(self):
        class IdentityKinematics:
            @staticmethod
            def points_body_to_inertial(points):
                return np.asarray(points, dtype=float)

        diagnostic = SimpleNamespace(
            lower_downstream_trace=8.0,
            upper_downstream_trace=6.0,
            mean_downstream_trace=7.0,
            jump_ccw=2.0,
        )
        segment = SimpleNamespace(
            start_body=np.array((0.0, 0.0)),
            end_body=np.array((0.2, 0.0)),
            length=0.2,
        )
        old_blob = SimpleNamespace(
            circulation_ccw=0.2,
            position_inertial=np.array((1.0, 2.0)),
        )
        solution = SimpleNamespace(
            common_te_diagnostic=diagnostic,
            near_wake_segment=segment,
            newborn_sheet_strength_ccw=1.5,
            newborn_circulation_ccw=0.3,
            bound_circulation_ccw=-0.5,
            inputs=SimpleNamespace(
                old_blobs=(old_blob,),
                kinematics=IdentityKinematics(),
            ),
        )
        local, global_wake = MODULE._birth_and_wake_metrics(solution)
        self.assertEqual(set(local), set(MODULE.LOCAL_BIRTH_SCALES))
        self.assertEqual(
            set(global_wake),
            set(MODULE.GLOBAL_WAKE_SCALES),
        )
        self.assertEqual(local["te_mean_downstream_trace"], 7.0)
        self.assertEqual(local["te_emission_jump_ccw"], 2.0)
        self.assertEqual(local["newborn_segment_length"], 0.2)
        self.assertEqual(local["newborn_circulation_ccw"], 0.3)
        self.assertEqual(global_wake["wake_circulation_ccw"], 0.5)
        # First moment: 0.2*(1,2) + 0.3*(0.1,0) = (0.23,0.4).
        self.assertAlmostEqual(
            global_wake["wake_first_moment_x"],
            0.23,
        )
        self.assertAlmostEqual(
            global_wake["wake_first_moment_y"],
            0.4,
        )
        self.assertAlmostEqual(
            global_wake["wake_signed_centroid_x"],
            0.46,
        )
        self.assertAlmostEqual(
            global_wake["wake_signed_centroid_y"],
            0.8,
        )

    @staticmethod
    def _completed_case(case, index):
        physical_multiplier = (0.5, 0.51, 0.515)[index]
        metrics = {
            group_name: {
                name: physical_multiplier * scale
                for name, scale in scales.items()
            }
            for group_name, scales in MODULE.METRIC_GROUP_SCALES.items()
            if group_name != "algebraic"
        }
        residual_level = (8.0e-10, 7.0e-10, 6.5e-10)[index]
        metrics["algebraic"] = {
            name: residual_level for name in MODULE.ALGEBRAIC_KEYS
        }
        return {
            **case.as_dict(),
            "status": "completed",
            "branch_unambiguous": True,
            "final_orientation_side": "lower",
            "metrics": metrics,
            "algebraic_gate_pass": True,
        }

    @classmethod
    def _passing_cases(cls):
        return {
            case.case_id: cls._completed_case(case, index)
            for index, case in enumerate(MODULE.frozen_cases())
        }

    def test_evaluator_applies_final_two_percent_and_monotonic_gates(self):
        comparison = MODULE.evaluate_refinement(self._passing_cases())
        self.assertTrue(comparison["all_scores_finite"])
        self.assertTrue(
            comparison["all_final_scores_within_tolerance"]
        )
        self.assertTrue(comparison["all_scores_nonincreasing"])
        self.assertTrue(comparison["pass"])
        entries = [
            entry
            for group in comparison["metrics"].values()
            for entry in group.values()
        ]
        self.assertEqual(len(entries), 20)
        self.assertTrue(all(entry["pass"] for entry in entries))

    def test_evaluator_fails_a_final_score_above_two_percent(self):
        cases = self._passing_cases()
        fine_id = MODULE.RefinementCase(256).case_id
        scale = MODULE.LOCAL_BIRTH_SCALES[
            "te_lower_downstream_trace"
        ]
        cases[fine_id]["metrics"]["local_birth"][
            "te_lower_downstream_trace"
        ] = 0.55 * scale
        comparison = MODULE.evaluate_refinement(cases)
        entry = comparison["metrics"]["local_birth"][
            "te_lower_downstream_trace"
        ]
        self.assertGreater(
            entry["middle_to_fine"]["score"],
            MODULE.FINAL_SCORE_TOLERANCE,
        )
        self.assertFalse(entry["final_score_within_tolerance"])
        self.assertFalse(comparison["pass"])

    def test_evaluator_fails_nonmonotonic_even_if_final_is_below_two_percent(
        self,
    ):
        cases = self._passing_cases()
        ids = [case.case_id for case in MODULE.frozen_cases()]
        scale = MODULE.GLOBAL_WAKE_SCALES["bound_circulation_ccw"]
        values = (0.5 * scale, 0.502 * scale, 0.507 * scale)
        for case_id, value in zip(ids, values):
            cases[case_id]["metrics"]["global_wake"][
                "bound_circulation_ccw"
            ] = value
        comparison = MODULE.evaluate_refinement(cases)
        entry = comparison["metrics"]["global_wake"][
            "bound_circulation_ccw"
        ]
        self.assertLessEqual(
            entry["middle_to_fine"]["score"],
            MODULE.FINAL_SCORE_TOLERANCE,
        )
        self.assertFalse(entry["score_nonincreasing"])
        self.assertFalse(entry["pass"])
        self.assertFalse(comparison["pass"])

    def test_execute_gate_preserves_failures_and_rejects_partial_matrix(self):
        def fake_case(case):
            if case.panels_per_side == 256:
                raise RuntimeError("synthetic branch ambiguity")
            index = MODULE.PANEL_LEVELS.index(case.panels_per_side)
            return self._completed_case(case, index)

        with patch.object(
            MODULE,
            "run_refinement_case",
            side_effect=fake_case,
        ):
            result = MODULE.execute_gate()
        self.assertEqual(result["gate"]["verdict"], "NO-GO")
        self.assertTrue(result["gate"]["all_three_unique_cases_present"])
        self.assertFalse(result["gate"]["all_cases_completed"])
        failed = result["cases"]["p256_n32_core0.02"]
        self.assertEqual(failed["status"], "failed")
        self.assertEqual(failed["error_type"], "RuntimeError")
        self.assertIn("branch ambiguity", failed["error"])

    def test_execute_gate_recomputes_branch_and_algebra_guards(self):
        def ambiguous_case(case):
            index = MODULE.PANEL_LEVELS.index(case.panels_per_side)
            result = self._completed_case(case, index)
            if case.panels_per_side == 128:
                result["branch_unambiguous"] = False
            return result

        with patch.object(
            MODULE,
            "run_refinement_case",
            side_effect=ambiguous_case,
        ):
            ambiguous = MODULE.execute_gate()
        self.assertFalse(ambiguous["gate"]["all_branches_unambiguous"])
        self.assertFalse(ambiguous["gate"]["overall_pass"])

        def bad_algebra_case(case):
            index = MODULE.PANEL_LEVELS.index(case.panels_per_side)
            result = self._completed_case(case, index)
            if case.panels_per_side == 256:
                result["metrics"]["algebraic"][
                    "maximum_eq7_residual_over_c"
                ] = 2.0e-9
                result["algebraic_gate_pass"] = True
            return result

        with patch.object(
            MODULE,
            "run_refinement_case",
            side_effect=bad_algebra_case,
        ):
            bad_algebra = MODULE.execute_gate()
        self.assertFalse(
            bad_algebra["gate"]["all_algebraic_residuals_pass"]
        )
        self.assertFalse(bad_algebra["gate"]["overall_pass"])

    def test_result_writer_is_strict_json_and_reports_same_verdict(self):
        def fake_case(case):
            index = MODULE.PANEL_LEVELS.index(case.panels_per_side)
            return self._completed_case(case, index)

        with patch.object(
            MODULE,
            "run_refinement_case",
            side_effect=fake_case,
        ):
            result = MODULE.execute_gate()
        self.assertEqual(result["gate"]["verdict"], "GO")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            json_path = root / "result.json"
            markdown_path = root / "result.md"
            MODULE.write_results(
                result,
                json_path=json_path,
                markdown_path=markdown_path,
            )
            text = json_path.read_text(encoding="utf-8")
            self.assertNotIn("NaN", text)
            markdown = markdown_path.read_text(encoding="utf-8")
            self.assertIn("Verdict: **GO**", markdown)
            self.assertIn("128->256 score <= 2%", markdown)
            self.assertIn("spatial asymptotics only", markdown)

    def test_main_exit_code_is_zero_only_for_overall_pass(self):
        for overall_pass, expected in ((True, 0), (False, 1)):
            with self.subTest(overall_pass=overall_pass):
                fake = {
                    "gate": {
                        "overall_pass": overall_pass,
                        "verdict": "GO" if overall_pass else "NO-GO",
                    }
                }
                with (
                    patch.object(MODULE, "execute_gate", return_value=fake),
                    patch.object(MODULE, "write_results"),
                ):
                    code = MODULE.main(
                        ["--json", "unused.json", "--markdown", "unused.md"]
                    )
                self.assertEqual(code, expected)

    def test_runner_has_no_response_dataset_or_old_runner_dependency(self):
        text = RUNNER.read_text(encoding="utf-8").lower()
        forbidden = (
            "s6_sweep",
            "fig12_digitized",
            "fig17",
            "fig18",
            "fig19",
            "roboeagle",
            "_v2_robo",
            "run_n26e1b_attached_outer_refinement",
        )
        for token in forbidden:
            with self.subTest(token=token):
                self.assertNotIn(token, text)


if __name__ == "__main__":
    unittest.main()
