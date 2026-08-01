import importlib.util
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


PLATFORM = Path(__file__).resolve().parents[1]
RUNNER = PLATFORM / "run_n26e1b_attached_outer_refinement.py"
SPEC = importlib.util.spec_from_file_location(
    "run_n26e1b_attached_outer_refinement",
    RUNNER,
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class AttachedOuterRefinementRunnerTests(unittest.TestCase):
    def test_frozen_matrix_has_seven_unique_one_axis_cases(self):
        families = MODULE.frozen_case_families()
        self.assertEqual(tuple(families), ("panel", "time", "core"))
        self.assertEqual(
            [case.panels_per_side for case in families["panel"]],
            [16, 32, 64],
        )
        self.assertEqual(
            [case.ramp_steps for case in families["time"]],
            [16, 32, 64],
        )
        self.assertEqual(
            [case.core_radius_over_c for case in families["core"]],
            [0.04, 0.02, 0.01],
        )
        self.assertEqual(len(MODULE.frozen_unique_cases()), 7)
        middle = MODULE.RefinementCase(32, 32, 0.02)
        self.assertEqual(
            sum(middle in family for family in families.values()),
            3,
        )

    def test_half_cosine_and_clockwise_runtime_sign_are_frozen(self):
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
        motion = MODULE.pitching_kinematics(0.5 * MODULE.RAMP_DURATION)
        self.assertAlmostEqual(motion.angle_rad, -alpha_mid)
        self.assertAlmostEqual(motion.angular_velocity_rad_s, -rate_mid)
        self.assertAlmostEqual(motion.pivot_body[0], 0.25)
        self.assertAlmostEqual(motion.pivot_inertial[0], 0.25)

    def test_change_gate_switches_to_absolute_only_inside_frozen_near_zero(self):
        relative = MODULE.convergence_change(
            middle_value=1.019,
            fine_value=1.0,
            physical_scale=10.0,
        )
        self.assertEqual(relative["mode"], "relative_to_fine")
        self.assertAlmostEqual(relative["score"], 0.019)
        self.assertTrue(relative["pass"])

        near_zero = MODULE.convergence_change(
            middle_value=0.21,
            fine_value=0.19,
            physical_scale=10.0,
        )
        self.assertEqual(
            near_zero["mode"],
            "normalized_absolute_near_zero",
        )
        self.assertAlmostEqual(near_zero["score"], 0.002)
        self.assertTrue(near_zero["pass"])

        failed = MODULE.convergence_change(
            middle_value=0.5,
            fine_value=0.1,
            physical_scale=10.0,
        )
        self.assertFalse(failed["pass"])

    @staticmethod
    def _completed_case(case, offset=0.0):
        observables = {
            name: 0.5 * scale + offset
            for name, scale in MODULE.OBSERVABLE_SCALES.items()
        }
        residuals = {
            key: 1.0e-12 for key in MODULE.ALGEBRAIC_KEYS
        }
        return {
            **case.as_dict(),
            "status": "completed",
            "observables": observables,
            "algebraic_residuals": residuals,
            "algebraic_gate_pass": True,
        }

    def test_evaluator_uses_only_middle_and_fine_levels(self):
        results = {}
        for case in MODULE.frozen_unique_cases():
            results[case.case_id] = self._completed_case(case)
        # Deliberately corrupt only the coarse levels; the preregistered
        # decision must remain a middle-to-fine comparison.
        for family in MODULE.frozen_case_families().values():
            coarse = family[0]
            if coarse.case_id != MODULE.RefinementCase(32, 32, 0.02).case_id:
                results[coarse.case_id]["observables"] = {
                    name: 1000.0 for name in MODULE.OBSERVABLE_SCALES
                }
        comparison = MODULE.evaluate_refinement(results)
        self.assertTrue(all(item["pass"] for item in comparison.values()))

    def test_execute_gate_records_failures_and_cannot_pass_a_partial_matrix(self):
        def fake_case(case):
            if case == MODULE.RefinementCase(64, 32, 0.02):
                raise RuntimeError("synthetic branch failure")
            return self._completed_case(case)

        with patch.object(MODULE, "run_refinement_case", side_effect=fake_case):
            result = MODULE.execute_gate()
        self.assertEqual(result["gate"]["verdict"], "NO-GO")
        self.assertFalse(result["gate"]["all_cases_completed"])
        failed = result["cases"]["p64_n32_core0.02"]
        self.assertEqual(failed["status"], "failed")
        self.assertEqual(failed["error_type"], "RuntimeError")
        self.assertIn("synthetic branch failure", failed["error"])

    def test_result_writer_is_strict_json_and_reports_same_verdict(self):
        cases = {
            case.case_id: self._completed_case(case)
            for case in MODULE.frozen_unique_cases()
        }
        comparison = MODULE.evaluate_refinement(cases)
        result = {
            "run_id": MODULE.RUN_ID,
            "generated_at_utc": "2026-07-30T00:00:00+00:00",
            "gate": {
                "verdict": "GO",
                "overall_pass": True,
                "all_seven_unique_cases_present": True,
                "all_cases_completed": True,
                "all_algebraic_residuals_pass": True,
                "all_final_two_level_changes_pass": True,
            },
            "cases": cases,
            "comparisons": comparison,
        }
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
            self.assertIn("numerical admissibility only", markdown)

    def test_runner_has_no_response_dataset_import(self):
        text = RUNNER.read_text(encoding="utf-8").lower()
        forbidden = (
            "s6_sweep",
            "fig12_digitized",
            "fig17",
            "fig18",
            "fig19",
            "roboeagle",
            "_v2_robo",
        )
        for token in forbidden:
            with self.subTest(token=token):
                self.assertNotIn(token, text)


if __name__ == "__main__":
    unittest.main()
