#!/usr/bin/env python3
"""Independently audit every N2.6e1b1 two-branch TE selection.

This script does not modify the formal runner or solver.  It temporarily wraps
the solver's existing branch selector, calls the unchanged formal case runner,
and records the two candidate branches presented to every selector call.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping
from unittest.mock import patch

PLATFORM_DIR = Path(__file__).resolve().parent
if str(PLATFORM_DIR) not in sys.path:
    sys.path.insert(0, str(PLATFORM_DIR))

import run_n26e1b1_source_faithful_te_refinement as formal_runner  # noqa: E402
from claim_runtime import svi_dw_unsteady_outer_2d as outer_solver  # noqa: E402


RUN_ID = "n26e1b1_branch_uniqueness_audit_20260730"
AUDIT_PREREGISTRATION = (
    PLATFORM_DIR
    / "docs"
    / "diag"
    / "n26e1b1_branch_uniqueness_audit_prereg_20260730.md"
)
DEFAULT_FORMAL_RESULT = (
    PLATFORM_DIR
    / "docs"
    / "diag"
    / "n26e1b1_source_faithful_te_refinement_result_20260730.json"
)
DEFAULT_JSON = (
    PLATFORM_DIR
    / "docs"
    / "diag"
    / "n26e1b1_branch_uniqueness_audit_result_20260730.json"
)
DEFAULT_MARKDOWN = (
    PLATFORM_DIR
    / "docs"
    / "diag"
    / "n26e1b1_branch_uniqueness_audit_result_20260730.md"
)
EXPECTED_CALLS_PER_CASE = formal_runner.RAMP_STEPS + 1
EXPECTED_TOTAL_CALLS = len(formal_runner.PANEL_LEVELS) * EXPECTED_CALLS_PER_CASE


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _error_payload(error: BaseException | None) -> dict[str, str] | None:
    if error is None:
        return None
    try:
        message = str(error)
    except BaseException:
        message = "<error message unavailable>"
    return {"type": type(error).__name__, "message": message}


def _finite_float(value: Any) -> float | None:
    try:
        converted = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return converted if math.isfinite(converted) else None


def _branch_snapshot(
    side: str,
    branches: Mapping[str, Any],
    branch_errors: Mapping[str, BaseException],
) -> dict[str, Any]:
    branch = branches.get(side)
    error = branch_errors.get(side)
    if branch is not None and error is not None:
        status = "invalid_success_and_error"
    elif branch is not None:
        status = "success"
    elif error is not None:
        status = "error"
    else:
        status = "missing"
    snapshot: dict[str, Any] = {
        "status": status,
        "error": _error_payload(error),
        "delta_gamma_bound_ccw": None,
        "sign_tolerance": None,
        "roundoff_no_birth": None,
        "nonzero_sign_consistent": None,
        "solver_reported_roundoff_no_birth": None,
        "solver_reported_nonzero_sign_consistent": None,
        "orientation_side": None,
        "facts_complete": False,
    }
    if branch is None:
        return snapshot
    try:
        delta_gamma = _finite_float(
            branch.bound_circulation_change_ccw
        )
        sign_tolerance = _finite_float(branch.sign_tolerance)
        solver_roundoff = bool(branch.is_roundoff_no_birth)
        solver_consistent = bool(branch.is_nonzero_sign_consistent)
        independent_roundoff = (
            None
            if delta_gamma is None or sign_tolerance is None
            else abs(delta_gamma) <= sign_tolerance
        )
        independent_consistent = (
            None
            if delta_gamma is None or sign_tolerance is None
            else (
                delta_gamma < -sign_tolerance
                if side == "lower"
                else delta_gamma > sign_tolerance
            )
        )
        snapshot.update(
            {
                "delta_gamma_bound_ccw": delta_gamma,
                "sign_tolerance": sign_tolerance,
                "roundoff_no_birth": independent_roundoff,
                "nonzero_sign_consistent": independent_consistent,
                "solver_reported_roundoff_no_birth": solver_roundoff,
                "solver_reported_nonzero_sign_consistent": (
                    solver_consistent
                ),
                "orientation_side": str(branch.orientation_side),
            }
        )
    except BaseException as capture_error:
        snapshot["capture_error"] = _error_payload(capture_error)
        return snapshot
    snapshot["facts_complete"] = (
        snapshot["delta_gamma_bound_ccw"] is not None
        and snapshot["sign_tolerance"] is not None
        and snapshot["sign_tolerance"] >= 0.0
        and snapshot["orientation_side"] == side
        and snapshot["roundoff_no_birth"]
        == snapshot["solver_reported_roundoff_no_birth"]
        and snapshot["nonzero_sign_consistent"]
        == snapshot["solver_reported_nonzero_sign_consistent"]
    )
    return snapshot


def _evaluate_call(record: dict[str, Any]) -> None:
    candidates = record["candidates"]
    exact_success_set = (
        set(record["input_success_sides"]) == {"lower", "upper"}
        and not record["input_error_sides"]
        and not record["unexpected_input_sides"]
    )
    both_succeeded = exact_success_set and all(
        candidates[side]["status"] == "success"
        and candidates[side]["facts_complete"] is True
        for side in ("lower", "upper")
    )
    consistent_sides = [
        side
        for side in ("lower", "upper")
        if candidates[side]["nonzero_sign_consistent"] is True
    ]
    selected_side = record["selected_side"]
    selected_is_input = record["selected_object_is_input_branch"] is True
    selector_returned = record["selector_error"] is None

    unique_nonzero_pass = (
        both_succeeded
        and len(consistent_sides) == 1
        and selector_returned
        and selected_is_input
        and selected_side == consistent_sides[0]
    )
    no_birth_pass = (
        both_succeeded
        and not consistent_sides
        and all(
            candidates[side]["roundoff_no_birth"] is True
            for side in ("lower", "upper")
        )
        and record["no_birth_fields_agree"] is True
        and selector_returned
        and selected_is_input
        and selected_side == "lower"
    )
    selection_rule_pass = unique_nonzero_pass or no_birth_pass

    reasons: list[str] = []
    if not both_succeeded:
        reasons.append("lower and upper did not both return complete branches")
    if not selector_returned:
        reasons.append("original selector raised")
    if both_succeeded and not selection_rule_pass:
        reasons.append(
            "neither the unique-nonzero nor common-no-birth rule passed"
        )
    record["audit"] = {
        "both_branches_succeeded": both_succeeded,
        "consistent_sides": consistent_sides,
        "unique_nonzero_rule_pass": unique_nonzero_pass,
        "common_no_birth_rule_pass": no_birth_pass,
        "selection_rule_pass": selection_rule_pass,
        "pass": both_succeeded and selection_rule_pass,
        "failure_reasons": reasons,
    }


class BranchSelectorRecorder:
    """Transparent callable wrapper around the production selector."""

    def __init__(
        self,
        case_id: str,
        original_selector: Any,
        no_birth_agreement: Any,
    ) -> None:
        self.case_id = case_id
        self.original_selector = original_selector
        self.no_birth_agreement = no_birth_agreement
        self.records: list[dict[str, Any]] = []

    def __call__(
        self,
        branches: dict[str, Any],
        branch_errors: dict[str, BaseException],
    ) -> Any:
        selected = None
        selector_error: BaseException | None = None
        try:
            selected = self.original_selector(branches, branch_errors)
        except BaseException as error:
            selector_error = error
            self._append_record(
                branches, branch_errors, selected, selector_error
            )
            raise
        self._append_record(branches, branch_errors, selected, None)
        return selected

    def _append_record(
        self,
        branches: Mapping[str, Any],
        branch_errors: Mapping[str, BaseException],
        selected: Any,
        selector_error: BaseException | None,
    ) -> None:
        call_index = len(self.records)
        try:
            candidates = {
                side: _branch_snapshot(side, branches, branch_errors)
                for side in ("lower", "upper")
            }
            for side in ("lower", "upper"):
                candidates[side]["selected"] = (
                    selected is not None and selected is branches.get(side)
                )
            selected_side = (
                None
                if selected is None
                else str(getattr(selected, "orientation_side", "unknown"))
            )
            input_sides = set(branches) | set(branch_errors)
            record: dict[str, Any] = {
                "case_id": self.case_id,
                "call_index": call_index,
                "stage": (
                    "march"
                    if call_index < formal_runner.RAMP_STEPS
                    else (
                        "final_observation"
                        if call_index == formal_runner.RAMP_STEPS
                        else "unexpected_extra"
                    )
                ),
                "expected_time_s": (
                    call_index
                    * formal_runner.RAMP_DURATION
                    / formal_runner.RAMP_STEPS
                    if call_index < formal_runner.RAMP_STEPS
                    else formal_runner.RAMP_DURATION
                ),
                "input_success_sides": sorted(str(key) for key in branches),
                "input_error_sides": sorted(
                    str(key) for key in branch_errors
                ),
                "unexpected_input_sides": sorted(
                    str(key)
                    for key in input_sides
                    if key not in {"lower", "upper"}
                ),
                "candidates": candidates,
                "selected_side": selected_side,
                "selected_object_is_input_branch": (
                    selected is not None
                    and selected is branches.get(selected_side)
                ),
                "selector_error": _error_payload(selector_error),
                "no_birth_fields_agree": None,
                "no_birth_agreement_error": None,
            }
            if all(
                candidates[side]["status"] == "success"
                and candidates[side]["roundoff_no_birth"] is True
                for side in ("lower", "upper")
            ):
                try:
                    record["no_birth_fields_agree"] = bool(
                        self.no_birth_agreement(
                            branches["lower"], branches["upper"]
                        )
                    )
                except BaseException as agreement_error:
                    record["no_birth_agreement_error"] = _error_payload(
                        agreement_error
                    )
            _evaluate_call(record)
        except BaseException as capture_error:
            record = {
                "case_id": self.case_id,
                "call_index": call_index,
                "stage": "capture_failed",
                "capture_error": _error_payload(capture_error),
                "audit": {
                    "both_branches_succeeded": False,
                    "selection_rule_pass": False,
                    "pass": False,
                    "failure_reasons": ["audit capture failed"],
                },
            }
        self.records.append(record)


def summarize_calls(
    records: Iterable[Mapping[str, Any]],
    *,
    expected_count: int = EXPECTED_CALLS_PER_CASE,
) -> dict[str, Any]:
    calls = list(records)
    count_pass = len(calls) == expected_count
    both_succeeded = count_pass and all(
        call.get("audit", {}).get("both_branches_succeeded") is True
        for call in calls
    )
    selections_pass = count_pass and all(
        call.get("audit", {}).get("selection_rule_pass") is True
        for call in calls
    )
    return {
        "expected_selector_calls": expected_count,
        "observed_selector_calls": len(calls),
        "selector_call_count_pass": count_pass,
        "every_call_has_two_successful_branches": both_succeeded,
        "every_call_has_one_admissible_selection": selections_pass,
        "pass": count_pass and both_succeeded and selections_pass,
    }


def run_audited_case(case: formal_runner.RefinementCase) -> dict[str, Any]:
    original = outer_solver._select_physical_orientation_branch
    recorder = BranchSelectorRecorder(
        case.case_id,
        original,
        outer_solver._no_birth_branch_solutions_agree,
    )
    case_result: dict[str, Any] | None = None
    case_error: BaseException | None = None
    case_traceback: str | None = None
    try:
        with patch.object(
            outer_solver,
            "_select_physical_orientation_branch",
            new=recorder,
        ):
            case_result = formal_runner.run_refinement_case(case)
    except Exception as error:
        case_error = error
        case_traceback = traceback.format_exc()
    summary = summarize_calls(recorder.records)
    completed = (
        case_error is None
        and case_result is not None
        and case_result.get("status") == "completed"
    )
    return {
        **case.as_dict(),
        "status": "completed" if completed else "failed",
        "case_error": _error_payload(case_error),
        "case_traceback": case_traceback,
        "selector_summary": summary,
        "selector_calls": recorder.records,
        "recomputed_case": case_result,
        "branch_evidence_pass": completed and summary["pass"],
    }


def _numeric_metric_leaves(
    value: Any,
    *,
    path: str = "",
) -> tuple[dict[str, float], list[str]]:
    leaves: dict[str, float] = {}
    errors: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_path = f"{path}.{key}" if path else str(key)
            child_leaves, child_errors = _numeric_metric_leaves(
                child, path=child_path
            )
            leaves.update(child_leaves)
            errors.extend(child_errors)
        return leaves, errors
    if isinstance(value, bool):
        errors.append(f"{path}: boolean is not a numeric metric")
        return leaves, errors
    converted = _finite_float(value)
    if converted is None:
        errors.append(f"{path}: metric is non-finite or non-numeric")
    else:
        leaves[path] = converted
    return leaves, errors


def compare_metrics_bitwise(
    actual: Any,
    reference: Any,
) -> dict[str, Any]:
    actual_leaves, actual_errors = _numeric_metric_leaves(actual)
    reference_leaves, reference_errors = _numeric_metric_leaves(reference)
    actual_keys = set(actual_leaves)
    reference_keys = set(reference_leaves)
    mismatches: list[dict[str, Any]] = []
    for key in sorted(actual_keys & reference_keys):
        actual_bits = struct.pack("!d", actual_leaves[key]).hex()
        reference_bits = struct.pack("!d", reference_leaves[key]).hex()
        if actual_bits != reference_bits:
            mismatches.append(
                {
                    "metric": key,
                    "actual": actual_leaves[key],
                    "reference": reference_leaves[key],
                    "actual_ieee754_hex": actual_bits,
                    "reference_ieee754_hex": reference_bits,
                }
            )
    return {
        "actual_errors": actual_errors,
        "reference_errors": reference_errors,
        "missing_metrics": sorted(reference_keys - actual_keys),
        "unexpected_metrics": sorted(actual_keys - reference_keys),
        "value_mismatches": mismatches,
        "bitwise_equal": not (
            actual_errors
            or reference_errors
            or actual_keys != reference_keys
            or mismatches
        ),
    }


def _source_hash_audit(formal_result: Mapping[str, Any]) -> dict[str, Any]:
    environment = formal_result.get("environment", {})
    runtime_sources = environment.get("runtime_sources", {})
    items = {
        "formal_runner": (
            Path(formal_runner.__file__).resolve(),
            environment.get("runner_sha256"),
        ),
        "unsteady_outer_solver": (
            Path(outer_solver.__file__).resolve(),
            runtime_sources.get("unsteady_outer", {}).get("sha256"),
        ),
        "runtime_types": (
            formal_runner.RUNTIME_SOURCES["types"],
            runtime_sources.get("types", {}).get("sha256"),
        ),
        "formal_preregistration": (
            formal_runner.PREREGISTRATION,
            environment.get("preregistration_sha256"),
        ),
    }
    entries: dict[str, Any] = {}
    for name, (path, expected) in items.items():
        actual = _sha256(path)
        entries[name] = {
            "path": str(path.relative_to(PLATFORM_DIR)),
            "formal_result_sha256": expected,
            "current_sha256": actual,
            "match": isinstance(expected, str) and actual == expected,
        }
    return {
        "entries": entries,
        "all_match": all(entry["match"] for entry in entries.values()),
    }


def execute_audit(
    *,
    cases: Iterable[formal_runner.RefinementCase] | None = None,
    formal_result_path: Path = DEFAULT_FORMAL_RESULT,
) -> dict[str, Any]:
    selected = (
        tuple(cases) if cases is not None else formal_runner.frozen_cases()
    )
    if not all(
        isinstance(case, formal_runner.RefinementCase) for case in selected
    ):
        raise TypeError("cases must contain only RefinementCase values")
    if len({case.case_id for case in selected}) != len(selected):
        raise ValueError("duplicate refinement cases are not allowed")

    audited_cases: dict[str, dict[str, Any]] = {}
    for case in selected:
        print(f"[{RUN_ID}] auditing {case.case_id}", flush=True)
        audited_cases[case.case_id] = run_audited_case(case)

    formal_result: Mapping[str, Any] | None = None
    formal_error: BaseException | None = None
    try:
        loaded = json.loads(formal_result_path.read_text(encoding="utf-8"))
        if not isinstance(loaded, Mapping):
            raise TypeError("formal result root must be an object")
        formal_result = loaded
    except Exception as error:
        formal_error = error

    source_hashes: dict[str, Any] = {"entries": {}, "all_match": False}
    formal_cases: Mapping[str, Any] = {}
    formal_verdict: str | None = None
    if formal_result is not None:
        source_hashes = _source_hash_audit(formal_result)
        candidate_cases = formal_result.get("cases", {})
        if isinstance(candidate_cases, Mapping):
            formal_cases = candidate_cases
        gate = formal_result.get("gate", {})
        if isinstance(gate, Mapping):
            value = gate.get("verdict")
            formal_verdict = value if isinstance(value, str) else None

    expected_ids = {
        case.case_id for case in formal_runner.frozen_cases()
    }
    for case_id, audited in audited_cases.items():
        recomputed = audited.get("recomputed_case")
        reference = formal_cases.get(case_id)
        if (
            isinstance(recomputed, Mapping)
            and isinstance(reference, Mapping)
        ):
            comparison = compare_metrics_bitwise(
                recomputed.get("metrics"),
                reference.get("metrics"),
            )
            orientation_equal = (
                recomputed.get("final_orientation_side")
                == reference.get("final_orientation_side")
            )
        else:
            comparison = {
                "actual_errors": ["recomputed case unavailable"],
                "reference_errors": ["formal reference case unavailable"],
                "missing_metrics": [],
                "unexpected_metrics": [],
                "value_mismatches": [],
                "bitwise_equal": False,
            }
            orientation_equal = False
        audited["formal_reproduction"] = {
            "metrics": comparison,
            "final_orientation_equal": orientation_equal,
            "pass": comparison["bitwise_equal"] and orientation_equal,
        }
        audited["pass"] = (
            audited["branch_evidence_pass"]
            and audited["formal_reproduction"]["pass"]
        )

    exact_case_matrix = set(audited_cases) == expected_ids
    all_completed = exact_case_matrix and all(
        case["status"] == "completed" for case in audited_cases.values()
    )
    total_calls = sum(
        case["selector_summary"]["observed_selector_calls"]
        for case in audited_cases.values()
    )
    all_branch_evidence = all_completed and total_calls == EXPECTED_TOTAL_CALLS
    all_branch_evidence = all_branch_evidence and all(
        case["branch_evidence_pass"] is True
        for case in audited_cases.values()
    )
    all_reproduced = all_completed and all(
        case["formal_reproduction"]["pass"] is True
        for case in audited_cases.values()
    )
    formal_reference_valid = (
        formal_error is None
        and formal_result is not None
        and formal_result.get("run_id") == formal_runner.RUN_ID
        and formal_verdict == "NO-GO"
        and source_hashes["all_match"] is True
    )
    overall_pass = (
        exact_case_matrix
        and all_branch_evidence
        and all_reproduced
        and formal_reference_valid
    )
    return {
        "schema_version": 1,
        "run_id": RUN_ID,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "audit_preregistration": str(
            AUDIT_PREREGISTRATION.relative_to(PLATFORM_DIR)
        ),
        "audit_environment": {
            "auditor_sha256": _sha256(Path(__file__).resolve()),
            "audit_preregistration_sha256": _sha256(
                AUDIT_PREREGISTRATION
            ),
            "formal_reference_loaded_only_after_solver_runs": True,
        },
        "formal_result": {
            "path": str(formal_result_path),
            "sha256": (
                _sha256(formal_result_path)
                if formal_result_path.is_file()
                else None
            ),
            "load_error": _error_payload(formal_error),
            "run_id": (
                formal_result.get("run_id")
                if formal_result is not None
                else None
            ),
            "verdict": formal_verdict,
            "source_hashes": source_hashes,
        },
        "frozen_contract": {
            "panels_per_side": list(formal_runner.PANEL_LEVELS),
            "ramp_steps": formal_runner.RAMP_STEPS,
            "selector_calls_per_case": EXPECTED_CALLS_PER_CASE,
            "selector_calls_total": EXPECTED_TOTAL_CALLS,
            "rule": (
                "both branches succeed and exactly one is nonzero "
                "sign-consistent, or both share the original roundoff "
                "no-birth limit"
            ),
        },
        "cases": audited_cases,
        "gate": {
            "exact_three_case_matrix": exact_case_matrix,
            "all_cases_completed": all_completed,
            "observed_selector_calls_total": total_calls,
            "all_99_calls_have_two_branch_evidence": all_branch_evidence,
            "all_formal_case_metrics_bitwise_reproduced": all_reproduced,
            "formal_reference_and_source_hashes_valid": (
                formal_reference_valid
            ),
            "branch_audit_pass": overall_pass,
            "verdict": "PASS" if overall_pass else "FAIL",
            "formal_n26e1b1_verdict_preserved": "NO-GO",
        },
    }


def render_markdown(result: Mapping[str, Any]) -> str:
    gate = result["gate"]
    lines = [
        "# N2.6e1b1 independent branch-uniqueness audit",
        "",
        f"- Audit verdict: **{gate['verdict']}**",
        "- Formal N2.6e1b1 verdict remains: **NO-GO**",
        f"- Generated (UTC): `{result['generated_at_utc']}`",
        "",
        "## Audit gates",
        "",
        "| gate | pass |",
        "|---|---:|",
        f"| exact 64/128/256 matrix | {gate['exact_three_case_matrix']} |",
        f"| all cases completed | {gate['all_cases_completed']} |",
        (
            "| all 99 calls contain successful lower and upper evidence | "
            f"{gate['all_99_calls_have_two_branch_evidence']} |"
        ),
        (
            "| formal case metrics reproduced bitwise | "
            f"{gate['all_formal_case_metrics_bitwise_reproduced']} |"
        ),
        (
            "| formal reference/source hashes valid | "
            f"{gate['formal_reference_and_source_hashes_valid']} |"
        ),
        "",
        "## Cases",
        "",
        "| case | calls | two branches every call | selection every call | "
        "formal metrics bitwise | pass |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for case_id in sorted(result["cases"]):
        case = result["cases"][case_id]
        summary = case["selector_summary"]
        lines.append(
            f"| `{case_id}` | {summary['observed_selector_calls']} | "
            f"{summary['every_call_has_two_successful_branches']} | "
            f"{summary['every_call_has_one_admissible_selection']} | "
            f"{case['formal_reproduction']['metrics']['bitwise_equal']} | "
            f"{case['pass']} |"
        )
    failures = [
        call
        for case in result["cases"].values()
        for call in case["selector_calls"]
        if call.get("audit", {}).get("pass") is not True
    ]
    lines.extend(["", "## Failed selector calls", ""])
    if not failures:
        lines.append("None.")
    else:
        lines.extend(
            [
                "| case | call | stage | reasons |",
                "|---|---:|---|---|",
            ]
        )
        for call in failures:
            reasons = "; ".join(
                call.get("audit", {}).get(
                    "failure_reasons", ["unknown audit failure"]
                )
            ).replace("|", "\\|")
            lines.append(
                f"| `{call['case_id']}` | {call['call_index']} | "
                f"{call.get('stage', 'unknown')} | {reasons} |"
            )
    lines.extend(
        [
            "",
            "This audit is diagnostic only. It cannot reverse the formal "
            "N2.6e1b1 NO-GO.",
            "",
        ]
    )
    return "\n".join(lines)


def write_results(
    result: Mapping[str, Any],
    *,
    json_path: Path,
    markdown_path: Path,
) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(
            result,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(
        render_markdown(result),
        encoding="utf-8",
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--formal-result",
        type=Path,
        default=DEFAULT_FORMAL_RESULT,
        help="unchanged formal N2.6e1b1 JSON to reproduce",
    )
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = execute_audit(formal_result_path=args.formal_result)
    write_results(
        result,
        json_path=args.json,
        markdown_path=args.markdown,
    )
    print(
        f"[{RUN_ID}] verdict={result['gate']['verdict']} "
        f"json={args.json} markdown={args.markdown}",
        flush=True,
    )
    return 0 if result["gate"]["branch_audit_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
