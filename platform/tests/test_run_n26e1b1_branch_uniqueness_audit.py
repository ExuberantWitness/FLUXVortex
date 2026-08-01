import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


PLATFORM = Path(__file__).resolve().parents[1]
AUDITOR = PLATFORM / "run_n26e1b1_branch_uniqueness_audit.py"
SPEC = importlib.util.spec_from_file_location(
    "run_n26e1b1_branch_uniqueness_audit",
    AUDITOR,
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class FakeBranch:
    def __init__(
        self,
        side,
        delta_gamma,
        *,
        tolerance=1.0e-12,
    ):
        self.orientation_side = side
        self.bound_circulation_change_ccw = delta_gamma
        self.sign_tolerance = tolerance

    @property
    def is_roundoff_no_birth(self):
        return (
            abs(self.bound_circulation_change_ccw)
            <= self.sign_tolerance
        )

    @property
    def is_nonzero_sign_consistent(self):
        if self.orientation_side == "lower":
            return (
                self.bound_circulation_change_ccw
                < -self.sign_tolerance
            )
        return (
            self.bound_circulation_change_ccw
            > self.sign_tolerance
        )


class BranchUniquenessAuditTests(unittest.TestCase):
    def test_wrapper_returns_exact_original_object_and_records_both_sides(self):
        lower = FakeBranch("lower", -1.0)
        upper = FakeBranch("upper", -0.5)
        calls = []

        def selector(branches, errors):
            calls.append((branches, errors))
            return branches["lower"]

        recorder = MODULE.BranchSelectorRecorder(
            "case",
            selector,
            lambda _lower, _upper: False,
        )
        returned = recorder(
            {"lower": lower, "upper": upper},
            {},
        )
        self.assertIs(returned, lower)
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(recorder.records), 1)
        record = recorder.records[0]
        self.assertTrue(record["audit"]["both_branches_succeeded"])
        self.assertEqual(record["audit"]["consistent_sides"], ["lower"])
        self.assertTrue(record["audit"]["selection_rule_pass"])
        self.assertTrue(record["audit"]["pass"])

    def test_one_success_one_error_fails_even_if_original_selects_success(self):
        lower = FakeBranch("lower", -1.0)

        def selector(branches, _errors):
            return branches["lower"]

        recorder = MODULE.BranchSelectorRecorder(
            "case",
            selector,
            lambda _lower, _upper: False,
        )
        returned = recorder(
            {"lower": lower},
            {"upper": RuntimeError("did not converge")},
        )
        self.assertIs(returned, lower)
        record = recorder.records[0]
        self.assertEqual(record["candidates"]["upper"]["status"], "error")
        self.assertFalse(record["audit"]["both_branches_succeeded"])
        self.assertFalse(record["audit"]["pass"])

    def test_sign_consistency_is_recomputed_from_delta_and_tolerance(self):
        class LyingBranch(FakeBranch):
            @property
            def is_nonzero_sign_consistent(self):
                return False

        lower = LyingBranch("lower", -1.0)
        upper = FakeBranch("upper", -0.5)
        recorder = MODULE.BranchSelectorRecorder(
            "case",
            lambda branches, _errors: branches["lower"],
            lambda _lower, _upper: False,
        )
        recorder({"lower": lower, "upper": upper}, {})
        snapshot = recorder.records[0]["candidates"]["lower"]
        self.assertTrue(snapshot["nonzero_sign_consistent"])
        self.assertFalse(
            snapshot["solver_reported_nonzero_sign_consistent"]
        )
        self.assertFalse(snapshot["facts_complete"])
        self.assertFalse(recorder.records[0]["audit"]["pass"])

    def test_original_selector_exception_is_re_raised_unchanged(self):
        sentinel = RuntimeError("ambiguous")

        def selector(_branches, _errors):
            raise sentinel

        recorder = MODULE.BranchSelectorRecorder(
            "case",
            selector,
            lambda _lower, _upper: False,
        )
        try:
            recorder(
                {
                    "lower": FakeBranch("lower", -1.0),
                    "upper": FakeBranch("upper", 1.0),
                },
                {},
            )
        except RuntimeError as caught:
            self.assertIs(caught, sentinel)
        else:
            self.fail("selector exception was not re-raised")
        self.assertEqual(
            recorder.records[0]["selector_error"]["message"],
            "ambiguous",
        )
        self.assertFalse(recorder.records[0]["audit"]["pass"])

    def test_roundoff_no_birth_requires_agreement_and_lower_selection(self):
        lower = FakeBranch("lower", 0.0)
        upper = FakeBranch("upper", 0.0)

        def selector(branches, _errors):
            return branches["lower"]

        passing = MODULE.BranchSelectorRecorder(
            "case",
            selector,
            lambda _lower, _upper: True,
        )
        passing({"lower": lower, "upper": upper}, {})
        self.assertTrue(
            passing.records[0]["audit"]["common_no_birth_rule_pass"]
        )

        failing = MODULE.BranchSelectorRecorder(
            "case",
            selector,
            lambda _lower, _upper: False,
        )
        failing({"lower": lower, "upper": upper}, {})
        self.assertFalse(
            failing.records[0]["audit"]["common_no_birth_rule_pass"]
        )
        self.assertFalse(failing.records[0]["audit"]["pass"])

    def test_case_summary_requires_exactly_33_passing_calls(self):
        passing_call = {
            "audit": {
                "both_branches_succeeded": True,
                "selection_rule_pass": True,
            }
        }
        complete = MODULE.summarize_calls(
            [passing_call] * 33,
            expected_count=33,
        )
        short = MODULE.summarize_calls(
            [passing_call] * 32,
            expected_count=33,
        )
        self.assertTrue(complete["pass"])
        self.assertFalse(short["selector_call_count_pass"])
        self.assertFalse(short["pass"])

    def test_metric_reproduction_is_ieee754_bitwise_not_numeric_equality(self):
        equal = MODULE.compare_metrics_bitwise(
            {"group": {"metric": 1.25}},
            {"group": {"metric": 1.25}},
        )
        signed_zero = MODULE.compare_metrics_bitwise(
            {"group": {"metric": 0.0}},
            {"group": {"metric": -0.0}},
        )
        self.assertTrue(equal["bitwise_equal"])
        self.assertFalse(signed_zero["bitwise_equal"])
        self.assertEqual(len(signed_zero["value_mismatches"]), 1)

    def test_run_audited_case_patches_production_module_and_fails_short_log(self):
        case = MODULE.formal_runner.RefinementCase(64)
        lower = FakeBranch("lower", -1.0)
        upper = FakeBranch("upper", -0.5)

        def fake_case(_case):
            selected = (
                MODULE.outer_solver._select_physical_orientation_branch(
                    {"lower": lower, "upper": upper},
                    {},
                )
            )
            self.assertIs(selected, lower)
            return {
                **case.as_dict(),
                "status": "completed",
                "final_orientation_side": "lower",
                "metrics": {"group": {"metric": 1.0}},
            }

        original = MODULE.outer_solver._select_physical_orientation_branch
        with patch.object(
            MODULE.formal_runner,
            "run_refinement_case",
            side_effect=fake_case,
        ):
            result = MODULE.run_audited_case(case)
        self.assertIs(
            MODULE.outer_solver._select_physical_orientation_branch,
            original,
        )
        self.assertEqual(
            result["selector_summary"]["observed_selector_calls"],
            1,
        )
        self.assertFalse(result["branch_evidence_pass"])

    def test_writer_is_strict_json_and_main_exit_follows_audit_gate(self):
        result = {
            "generated_at_utc": "2026-07-30T00:00:00+00:00",
            "cases": {},
            "gate": {
                "verdict": "FAIL",
                "exact_three_case_matrix": False,
                "all_cases_completed": False,
                "all_99_calls_have_two_branch_evidence": False,
                "all_formal_case_metrics_bitwise_reproduced": False,
                "formal_reference_and_source_hashes_valid": False,
                "branch_audit_pass": False,
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            MODULE.write_results(
                result,
                json_path=root / "result.json",
                markdown_path=root / "result.md",
            )
            self.assertNotIn(
                "NaN",
                (root / "result.json").read_text(encoding="utf-8"),
            )
            self.assertIn(
                "Audit verdict: **FAIL**",
                (root / "result.md").read_text(encoding="utf-8"),
            )

        with (
            patch.object(MODULE, "execute_audit", return_value=result),
            patch.object(MODULE, "write_results"),
        ):
            self.assertEqual(
                MODULE.main(
                    ["--json", "unused.json", "--markdown", "unused.md"]
                ),
                1,
            )


if __name__ == "__main__":
    unittest.main()
