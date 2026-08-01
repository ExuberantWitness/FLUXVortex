#!/usr/bin/env python3
"""Run the preregistered N2.6e1b1 source-faithful TE refinement gate.

Only the source-specified nearest trailing-edge control-point discretization
is exercised.  The runner has no force metric or target-response reader and
does not alter the outer-flow implementation.  Its sole independent variable
is the frozen cosine-clustered panel level.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform as python_platform
import sys
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

PLATFORM_DIR = Path(__file__).resolve().parent
if str(PLATFORM_DIR) not in sys.path:
    sys.path.insert(0, str(PLATFORM_DIR))

from claim_runtime.svi_dw_types import (  # noqa: E402
    NACA4SectionConfig,
    build_naca4_actual_surface,
)
from claim_runtime.svi_dw_unsteady_outer_2d import (  # noqa: E402
    AttachedOuterHistory2D,
    AttachedOuterStepInput2D,
    RigidKinematics2D,
    march_attached_unsteady_outer_explicit_euler,
    solve_attached_unsteady_outer_step,
)


RUN_ID = "n26e1b1_source_faithful_te_refinement_20260730"
PREREGISTRATION = (
    PLATFORM_DIR
    / "docs"
    / "diag"
    / "n26e1b1_source_faithful_te_refinement_prereg_20260730.md"
)
DEFAULT_JSON = (
    PLATFORM_DIR
    / "docs"
    / "diag"
    / "n26e1b1_source_faithful_te_refinement_result_20260730.json"
)
DEFAULT_MARKDOWN = (
    PLATFORM_DIR
    / "docs"
    / "diag"
    / "n26e1b1_source_faithful_te_refinement_result_20260730.md"
)
RUNTIME_SOURCES = {
    "types": PLATFORM_DIR / "claim_runtime" / "svi_dw_types.py",
    "unsteady_outer": (
        PLATFORM_DIR / "claim_runtime" / "svi_dw_unsteady_outer_2d.py"
    ),
}

CHORD = 1.0
FREESTREAM_SPEED = 9.0
FINAL_ALPHA_RAD = math.radians(6.0)
RAMP_DURATION = 0.4
PANEL_LEVELS = (64, 128, 256)
RAMP_STEPS = 32
CORE_RADIUS_OVER_C = 0.02

SCORE_FLOOR_FRACTION = 0.02
FINAL_SCORE_TOLERANCE = 0.02
MONOTONIC_SCORE_SLACK = 1.0e-12
ALGEBRAIC_TOLERANCE = 1.0e-9

LOCAL_BIRTH_SCALES = {
    "te_lower_downstream_trace": FREESTREAM_SPEED,
    "te_upper_downstream_trace": FREESTREAM_SPEED,
    "te_mean_downstream_trace": FREESTREAM_SPEED,
    "te_emission_jump_ccw": FREESTREAM_SPEED,
    "newborn_segment_length": CHORD,
    "newborn_sheet_strength_ccw": FREESTREAM_SPEED,
    "newborn_circulation_ccw": FREESTREAM_SPEED * CHORD,
    "newborn_endpoint_body_x": CHORD,
    "newborn_endpoint_body_y": CHORD,
}
GLOBAL_WAKE_SCALES = {
    "bound_circulation_ccw": FREESTREAM_SPEED * CHORD,
    "wake_circulation_ccw": FREESTREAM_SPEED * CHORD,
    "wake_signed_centroid_x": CHORD,
    "wake_signed_centroid_y": CHORD,
    "wake_first_moment_x": FREESTREAM_SPEED * CHORD**2,
    "wake_first_moment_y": FREESTREAM_SPEED * CHORD**2,
}
ALGEBRAIC_SCALES = {
    "maximum_kelvin_residual_over_Uc": 1.0,
    "maximum_normal_bc_residual_over_U": 1.0,
    "maximum_eq7_residual_over_c": 1.0,
    "maximum_eq8_residual_over_U": 1.0,
    "maximum_linear_system_residual_over_U": 1.0,
}
METRIC_GROUP_SCALES = {
    "local_birth": LOCAL_BIRTH_SCALES,
    "global_wake": GLOBAL_WAKE_SCALES,
    "algebraic": ALGEBRAIC_SCALES,
}
ALGEBRAIC_KEYS = tuple(ALGEBRAIC_SCALES)


@dataclass(frozen=True, order=True)
class RefinementCase:
    """One of the three immutable spatial refinement levels."""

    panels_per_side: int

    def __post_init__(self) -> None:
        panels = int(self.panels_per_side)
        if panels != self.panels_per_side or panels not in PANEL_LEVELS:
            raise ValueError(
                f"panels_per_side must be one of {PANEL_LEVELS}"
            )
        object.__setattr__(self, "panels_per_side", panels)

    @property
    def ramp_steps(self) -> int:
        return RAMP_STEPS

    @property
    def core_radius_over_c(self) -> float:
        return CORE_RADIUS_OVER_C

    @property
    def case_id(self) -> str:
        core = f"{self.core_radius_over_c:.3f}".rstrip("0").rstrip(".")
        return (
            f"p{self.panels_per_side}_n{self.ramp_steps}_core{core}"
        )

    def as_dict(self) -> dict[str, int | float | str]:
        return {
            "case_id": self.case_id,
            "panels_per_side": self.panels_per_side,
            "ramp_steps": self.ramp_steps,
            "core_radius_over_c": self.core_radius_over_c,
        }


def frozen_cases() -> tuple[RefinementCase, ...]:
    return tuple(RefinementCase(level) for level in PANEL_LEVELS)


def half_cosine_pitch(time_s: float) -> tuple[float, float]:
    """Return positive aerodynamic alpha and alpha-dot on the frozen ramp."""
    time_value = float(time_s)
    if not math.isfinite(time_value) or not 0.0 <= time_value <= RAMP_DURATION:
        raise ValueError(
            "time_s must be finite and lie in [0, RAMP_DURATION]"
        )
    phase = math.pi * time_value / RAMP_DURATION
    alpha = 0.5 * FINAL_ALPHA_RAD * (1.0 - math.cos(phase))
    alpha_dot = (
        0.5
        * FINAL_ALPHA_RAD
        * math.pi
        / RAMP_DURATION
        * math.sin(phase)
    )
    return alpha, alpha_dot


def pitching_kinematics(time_s: float) -> RigidKinematics2D:
    alpha, alpha_dot = half_cosine_pitch(time_s)
    pivot = (0.25 * CHORD, 0.0)
    return RigidKinematics2D(
        pivot_body=pivot,
        pivot_inertial=pivot,
        angle_rad=-alpha,
        translation_velocity_inertial=(0.0, 0.0),
        angular_velocity_rad_s=-alpha_dot,
    )


def _initial_history() -> AttachedOuterHistory2D:
    return AttachedOuterHistory2D(
        bound_circulation_ccw=0.0,
        material_blobs=(),
        kelvin_reference_total_ccw=0.0,
        stage_time=0.0,
    )


def _analytic_change_predictor(time_s: float, time_step: float) -> float:
    """Dimensional sign predictor only; branch acceptance remains internal."""
    alpha_now, _ = half_cosine_pitch(time_s)
    alpha_next, _ = half_cosine_pitch(
        min(time_s + time_step, RAMP_DURATION)
    )
    return -FREESTREAM_SPEED * CHORD * (alpha_next - alpha_now)


def _solution_residuals(solution: Any) -> dict[str, float]:
    return {
        "maximum_kelvin_residual_over_Uc": (
            abs(float(solution.kelvin_residual))
            / (FREESTREAM_SPEED * CHORD)
        ),
        "maximum_normal_bc_residual_over_U": (
            float(solution.maximum_normal_boundary_residual)
            / FREESTREAM_SPEED
        ),
        "maximum_eq7_residual_over_c": (
            abs(float(solution.eq7_length_residual)) / CHORD
        ),
        "maximum_eq8_residual_over_U": (
            abs(float(solution.emission_residual)) / FREESTREAM_SPEED
        ),
        "maximum_linear_system_residual_over_U": (
            float(
                np.max(
                    np.abs(solution.linear_system_residual),
                    initial=0.0,
                )
            )
            / FREESTREAM_SPEED
        ),
    }


def _update_residual_maxima(
    maxima: dict[str, float],
    values: Mapping[str, float],
) -> None:
    for key in ALGEBRAIC_KEYS:
        value = float(values[key])
        if not math.isfinite(value):
            raise FloatingPointError(
                f"non-finite algebraic residual: {key}"
            )
        maxima[key] = max(maxima[key], value)


def _birth_and_wake_metrics(
    final_solution: Any,
) -> tuple[dict[str, float], dict[str, float]]:
    """Extract every preregistered local-birth and global-wake observable."""
    diagnostic = final_solution.common_te_diagnostic
    segment = final_solution.near_wake_segment
    endpoint_body = np.asarray(segment.end_body, dtype=float)
    if endpoint_body.shape != (2,) or not np.all(np.isfinite(endpoint_body)):
        raise FloatingPointError("newborn endpoint is not a finite vector")

    local_birth = {
        "te_lower_downstream_trace": float(
            diagnostic.lower_downstream_trace
        ),
        "te_upper_downstream_trace": float(
            diagnostic.upper_downstream_trace
        ),
        "te_mean_downstream_trace": float(
            diagnostic.mean_downstream_trace
        ),
        "te_emission_jump_ccw": float(diagnostic.jump_ccw),
        "newborn_segment_length": float(segment.length),
        "newborn_sheet_strength_ccw": float(
            final_solution.newborn_sheet_strength_ccw
        ),
        "newborn_circulation_ccw": float(
            final_solution.newborn_circulation_ccw
        ),
        "newborn_endpoint_body_x": float(endpoint_body[0]),
        "newborn_endpoint_body_y": float(endpoint_body[1]),
    }

    circulations = [
        float(blob.circulation_ccw)
        for blob in final_solution.inputs.old_blobs
    ]
    positions = [
        np.asarray(blob.position_inertial, dtype=float)
        for blob in final_solution.inputs.old_blobs
    ]
    midpoint_body = 0.5 * (
        np.asarray(segment.start_body, dtype=float) + endpoint_body
    )
    midpoint_inertial = (
        final_solution.inputs.kinematics.points_body_to_inertial(
            midpoint_body[None, :]
        )[0]
    )
    circulations.append(float(final_solution.newborn_circulation_ccw))
    positions.append(np.asarray(midpoint_inertial, dtype=float))

    gamma = np.asarray(circulations, dtype=float)
    position = np.asarray(positions, dtype=float)
    if (
        gamma.ndim != 1
        or position.shape != (gamma.size, 2)
        or not np.all(np.isfinite(gamma))
        or not np.all(np.isfinite(position))
    ):
        raise FloatingPointError(
            "wake inventory is not finite and shape-consistent"
        )
    wake_circulation = float(np.sum(gamma))
    first_moment = np.sum(gamma[:, None] * position, axis=0)
    zero_tolerance = (
        4096.0
        * np.finfo(float).eps
        * max(
            float(np.sum(np.abs(gamma))),
            FREESTREAM_SPEED * CHORD,
        )
    )
    if abs(wake_circulation) <= zero_tolerance:
        raise FloatingPointError(
            "final wake circulation is numerically zero; signed centroid "
            "is undefined"
        )
    signed_centroid = first_moment / wake_circulation
    global_wake = {
        "bound_circulation_ccw": float(
            final_solution.bound_circulation_ccw
        ),
        "wake_circulation_ccw": wake_circulation,
        "wake_signed_centroid_x": float(signed_centroid[0]),
        "wake_signed_centroid_y": float(signed_centroid[1]),
        "wake_first_moment_x": float(first_moment[0]),
        "wake_first_moment_y": float(first_moment[1]),
    }

    for group_name, metrics, scales in (
        ("local_birth", local_birth, LOCAL_BIRTH_SCALES),
        ("global_wake", global_wake, GLOBAL_WAKE_SCALES),
    ):
        if set(metrics) != set(scales):
            raise RuntimeError(f"{group_name} metric contract drifted")
        if not all(math.isfinite(value) for value in metrics.values()):
            raise FloatingPointError(
                f"{group_name} contains a non-finite metric"
            )
    return local_birth, global_wake


def run_refinement_case(case: RefinementCase) -> dict[str, Any]:
    if not isinstance(case, RefinementCase):
        raise TypeError("case must be RefinementCase")
    surface = build_naca4_actual_surface(
        NACA4SectionConfig(
            maximum_camber=0.0,
            camber_location=0.4,
            thickness_ratio=0.15,
            chord=CHORD,
        ),
        panels_per_side=case.panels_per_side,
    )
    time_step = RAMP_DURATION / RAMP_STEPS
    history = _initial_history()
    maxima = {key: 0.0 for key in ALGEBRAIC_KEYS}
    maximum_condition_number = 0.0
    maximum_geometry_iterations = 0

    for step_index in range(RAMP_STEPS):
        time_s = step_index * time_step
        march = march_attached_unsteady_outer_explicit_euler(
            surface=surface,
            kinematics=pitching_kinematics(time_s),
            freestream_velocity_inertial=(FREESTREAM_SPEED, 0.0),
            time_step=time_step,
            history=history,
            predicted_bound_circulation_change_ccw=(
                _analytic_change_predictor(time_s, time_step)
            ),
            newborn_core_radius=CORE_RADIUS_OVER_C * CHORD,
        )
        _update_residual_maxima(
            maxima,
            _solution_residuals(march.solution),
        )
        maximum_condition_number = max(
            maximum_condition_number,
            float(march.solution.linear_system_condition_number),
        )
        maximum_geometry_iterations = max(
            maximum_geometry_iterations,
            int(march.solution.geometry_iterations),
        )
        history = march.history_next

    final_inputs = AttachedOuterStepInput2D.from_history(
        surface=surface,
        kinematics=pitching_kinematics(RAMP_DURATION),
        freestream_velocity_inertial=(FREESTREAM_SPEED, 0.0),
        time_step=time_step,
        history=history,
        predicted_bound_circulation_change_ccw=0.0,
    )
    final_solution = solve_attached_unsteady_outer_step(final_inputs)
    _update_residual_maxima(maxima, _solution_residuals(final_solution))
    maximum_condition_number = max(
        maximum_condition_number,
        float(final_solution.linear_system_condition_number),
    )
    maximum_geometry_iterations = max(
        maximum_geometry_iterations,
        int(final_solution.geometry_iterations),
    )
    local_birth, global_wake = _birth_and_wake_metrics(final_solution)
    if not all(math.isfinite(value) for value in maxima.values()):
        raise FloatingPointError("case produced a non-finite residual")
    return {
        **case.as_dict(),
        "time_step_s": time_step,
        "stage_solves": RAMP_STEPS + 1,
        "final_time_s": RAMP_DURATION,
        "branch_unambiguous": True,
        "final_orientation_side": (
            final_solution.near_wake_segment.orientation_side
        ),
        "metrics": {
            "local_birth": local_birth,
            "global_wake": global_wake,
            "algebraic": maxima,
        },
        "maximum_linear_system_condition_number": (
            maximum_condition_number
        ),
        "maximum_geometry_iterations": maximum_geometry_iterations,
        "algebraic_gate_pass": all(
            maxima[key] <= ALGEBRAIC_TOLERANCE
            for key in ALGEBRAIC_KEYS
        ),
        "status": "completed",
    }


def convergence_score(
    middle_value: float,
    fine_value: float,
    physical_scale: float,
) -> dict[str, float | str | bool]:
    """Apply the exact preregistered floored relative score."""
    middle = float(middle_value)
    fine = float(fine_value)
    scale = float(physical_scale)
    if (
        not math.isfinite(middle)
        or not math.isfinite(fine)
        or not math.isfinite(scale)
        or scale <= 0.0
    ):
        raise ValueError(
            "score values must be finite and physical_scale positive"
        )
    floor = SCORE_FLOOR_FRACTION * scale
    denominator = max(abs(fine), floor)
    score = abs(fine - middle) / denominator
    if not math.isfinite(score):
        raise FloatingPointError("convergence score is non-finite")
    return {
        "middle": middle,
        "fine": fine,
        "physical_scale": scale,
        "denominator_floor": floor,
        "denominator": denominator,
        "mode": (
            "relative_to_fine"
            if abs(fine) >= floor
            else "physical_scale_floor"
        ),
        "absolute_change": abs(fine - middle),
        "score": score,
        "score_finite": True,
    }


def _metric_value(
    result: Mapping[str, Any],
    group_name: str,
    metric_name: str,
) -> float:
    value = float(result["metrics"][group_name][metric_name])
    if not math.isfinite(value):
        raise FloatingPointError(
            f"{group_name}.{metric_name} is non-finite"
        )
    return value


def evaluate_refinement(
    case_results: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Score both frozen refinement intervals and their asymptotic trend."""
    cases = frozen_cases()
    case_ids = [case.case_id for case in cases]
    available = all(
        case_id in case_results
        and case_results[case_id].get("status") == "completed"
        for case_id in case_ids
    )
    output: dict[str, Any] = {
        "case_ids": {
            "coarse": case_ids[0],
            "middle": case_ids[1],
            "fine": case_ids[2],
        },
        "metrics": {
            group_name: {} for group_name in METRIC_GROUP_SCALES
        },
        "all_scores_finite": False,
        "all_final_scores_within_tolerance": False,
        "all_scores_nonincreasing": False,
        "pass": False,
    }
    if not available:
        output["failure"] = (
            "all three frozen cases must complete before scoring"
        )
        return output

    score_entries: list[dict[str, Any]] = []
    for group_name, scales in METRIC_GROUP_SCALES.items():
        for metric_name, scale in scales.items():
            entry: dict[str, Any] = {
                "physical_scale": scale,
                "final_score_threshold": FINAL_SCORE_TOLERANCE,
                "monotonic_slack": MONOTONIC_SCORE_SLACK,
                "pass": False,
            }
            try:
                coarse = _metric_value(
                    case_results[case_ids[0]],
                    group_name,
                    metric_name,
                )
                middle = _metric_value(
                    case_results[case_ids[1]],
                    group_name,
                    metric_name,
                )
                fine = _metric_value(
                    case_results[case_ids[2]],
                    group_name,
                    metric_name,
                )
                first = convergence_score(coarse, middle, scale)
                final = convergence_score(middle, fine, scale)
                final_pass = (
                    float(final["score"]) <= FINAL_SCORE_TOLERANCE
                )
                monotonic_pass = (
                    float(final["score"])
                    <= float(first["score"]) + MONOTONIC_SCORE_SLACK
                )
                entry.update(
                    {
                        "coarse_to_middle": first,
                        "middle_to_fine": final,
                        "scores_finite": True,
                        "final_score_within_tolerance": final_pass,
                        "score_nonincreasing": monotonic_pass,
                        "pass": final_pass and monotonic_pass,
                    }
                )
            except Exception as error:
                entry.update(
                    {
                        "scores_finite": False,
                        "final_score_within_tolerance": False,
                        "score_nonincreasing": False,
                        "error_type": type(error).__name__,
                        "error": str(error),
                    }
                )
            output["metrics"][group_name][metric_name] = entry
            score_entries.append(entry)

    output["all_scores_finite"] = all(
        entry["scores_finite"] is True for entry in score_entries
    )
    output["all_final_scores_within_tolerance"] = all(
        entry["final_score_within_tolerance"] is True
        for entry in score_entries
    )
    output["all_scores_nonincreasing"] = all(
        entry["score_nonincreasing"] is True
        for entry in score_entries
    )
    output["pass"] = all(
        (
            output["all_scores_finite"],
            output["all_final_scores_within_tolerance"],
            output["all_scores_nonincreasing"],
        )
    )
    return output


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _completed_case_algebra_pass(result: Mapping[str, Any]) -> bool:
    if result.get("status") != "completed":
        return False
    try:
        residuals = result["metrics"]["algebraic"]
        return all(
            math.isfinite(float(residuals[key]))
            and float(residuals[key]) <= ALGEBRAIC_TOLERANCE
            for key in ALGEBRAIC_KEYS
        )
    except (KeyError, TypeError, ValueError):
        return False


def execute_gate(
    *,
    cases: Iterable[RefinementCase] | None = None,
) -> dict[str, Any]:
    selected = tuple(cases) if cases is not None else frozen_cases()
    if not all(isinstance(case, RefinementCase) for case in selected):
        raise TypeError("cases must contain only RefinementCase values")
    if len({case.case_id for case in selected}) != len(selected):
        raise ValueError("duplicate refinement cases are not allowed")

    result_by_id: dict[str, dict[str, Any]] = {}
    for case in selected:
        print(f"[{RUN_ID}] running {case.case_id}", flush=True)
        try:
            result_by_id[case.case_id] = run_refinement_case(case)
        except Exception as error:
            result_by_id[case.case_id] = {
                **case.as_dict(),
                "status": "failed",
                "branch_unambiguous": False,
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
                "algebraic_gate_pass": False,
            }
            print(
                f"[{RUN_ID}] {case.case_id} failed: "
                f"{type(error).__name__}: {error}",
                flush=True,
            )

    comparisons = evaluate_refinement(result_by_id)
    expected_ids = {case.case_id for case in frozen_cases()}
    all_cases_present = set(result_by_id) == expected_ids
    all_cases_completed = all_cases_present and all(
        result_by_id[case_id].get("status") == "completed"
        for case_id in expected_ids
    )
    all_branches_unambiguous = all_cases_completed and all(
        result_by_id[case_id].get("branch_unambiguous") is True
        for case_id in expected_ids
    )
    algebraic_pass = all_cases_completed and all(
        _completed_case_algebra_pass(result_by_id[case_id])
        for case_id in expected_ids
    )
    score_pass = comparisons.get("pass") is True
    overall_pass = all(
        (
            all_cases_completed,
            all_branches_unambiguous,
            algebraic_pass,
            score_pass,
        )
    )
    return {
        "schema_version": 1,
        "run_id": RUN_ID,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "preregistration": str(
            PREREGISTRATION.relative_to(PLATFORM_DIR)
        ),
        "problem": {
            "section": "NACA0015",
            "chord_m": CHORD,
            "freestream_speed_m_s": FREESTREAM_SPEED,
            "pivot_x_over_c": 0.25,
            "final_alpha_deg": math.degrees(FINAL_ALPHA_RAD),
            "ramp_duration_s": RAMP_DURATION,
            "pitch_schedule": "half_cosine",
            "wall_transpiration": 0.0,
            "initial_total_circulation_ccw": 0.0,
        },
        "frozen_levels": {
            "panels_per_side": list(PANEL_LEVELS),
            "ramp_steps": RAMP_STEPS,
            "core_radius_over_c": CORE_RADIUS_OVER_C,
        },
        "thresholds": {
            "score_denominator_floor_fraction": SCORE_FLOOR_FRACTION,
            "final_score": FINAL_SCORE_TOLERANCE,
            "score_nonincrease_slack": MONOTONIC_SCORE_SLACK,
            "normalized_algebraic_residual": ALGEBRAIC_TOLERANCE,
        },
        "metric_physical_scales": METRIC_GROUP_SCALES,
        "environment": {
            "python": sys.version,
            "python_platform": python_platform.platform(),
            "numpy": np.__version__,
            "runtime_sources": {
                name: {
                    "path": str(path.relative_to(PLATFORM_DIR)),
                    "sha256": _sha256(path),
                }
                for name, path in RUNTIME_SOURCES.items()
            },
            "runner_sha256": _sha256(Path(__file__).resolve()),
            "preregistration_sha256": _sha256(PREREGISTRATION),
        },
        "cases": result_by_id,
        "comparisons": comparisons,
        "gate": {
            "all_three_unique_cases_present": all_cases_present,
            "all_cases_completed": all_cases_completed,
            "all_branches_unambiguous": all_branches_unambiguous,
            "all_algebraic_residuals_pass": algebraic_pass,
            "all_scores_finite": (
                comparisons.get("all_scores_finite") is True
            ),
            "all_final_scores_within_two_percent": (
                comparisons.get(
                    "all_final_scores_within_tolerance"
                )
                is True
            ),
            "all_scores_nonincreasing": (
                comparisons.get("all_scores_nonincreasing") is True
            ),
            "overall_pass": overall_pass,
            "verdict": "GO" if overall_pass else "NO-GO",
        },
    }


def render_markdown(result: Mapping[str, Any]) -> str:
    gate = result["gate"]
    lines = [
        "# N2.6e1b1 source-faithful TE refinement result",
        "",
        f"- Run: `{result['run_id']}`",
        f"- Verdict: **{gate['verdict']}**",
        f"- Generated (UTC): `{result['generated_at_utc']}`",
        (
            "- Scope: nearest-control-point TE spatial asymptotics only; "
            "no IBL, pressure, or force validation."
        ),
        "",
        "## Gate summary",
        "",
        "| gate | pass |",
        "|---|---:|",
        (
            "| three frozen spatial levels present | "
            f"{gate['all_three_unique_cases_present']} |"
        ),
        f"| all cases completed | {gate['all_cases_completed']} |",
        (
            "| no branch ambiguity | "
            f"{gate['all_branches_unambiguous']} |"
        ),
        (
            "| all normalized algebraic residuals <= 1e-9 | "
            f"{gate['all_algebraic_residuals_pass']} |"
        ),
        f"| every score finite | {gate['all_scores_finite']} |",
        (
            "| every 128->256 score <= 2% | "
            f"{gate['all_final_scores_within_two_percent']} |"
        ),
        (
            "| every final score nonincreasing | "
            f"{gate['all_scores_nonincreasing']} |"
        ),
        "",
        "## Cases",
        "",
        (
            "| case | status | side | lower/U | upper/U | mean/U | "
            "jump/U | length/c | Gamma_new/(Uc) | max residual |"
        ),
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for case_id in sorted(result["cases"]):
        case = result["cases"][case_id]
        if case["status"] != "completed":
            error = (
                f"{case.get('error_type', 'Error')}: "
                f"{case.get('error', 'unknown failure')}"
            ).replace("|", "\\|")
            lines.append(
                f"| `{case_id}` | failed: {error} | — | — | — | — | "
                "— | — | — | — |"
            )
            continue
        local = case["metrics"]["local_birth"]
        algebraic = case["metrics"]["algebraic"]
        maximum_residual = max(float(value) for value in algebraic.values())
        lines.append(
            f"| `{case_id}` | completed | "
            f"{case['final_orientation_side']} | "
            f"{local['te_lower_downstream_trace'] / FREESTREAM_SPEED:.9g} | "
            f"{local['te_upper_downstream_trace'] / FREESTREAM_SPEED:.9g} | "
            f"{local['te_mean_downstream_trace'] / FREESTREAM_SPEED:.9g} | "
            f"{local['te_emission_jump_ccw'] / FREESTREAM_SPEED:.9g} | "
            f"{local['newborn_segment_length'] / CHORD:.9g} | "
            f"{local['newborn_circulation_ccw'] / (FREESTREAM_SPEED * CHORD):.9g} | "
            f"{maximum_residual:.3e} |"
        )

    lines.extend(
        [
            "",
            "## Preregistered metric scores",
            "",
            (
                "| group.metric | 64->128 | 128->256 | final <=2% | "
                "nonincreasing | pass |"
            ),
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for group_name, metrics in result["comparisons"]["metrics"].items():
        for metric_name, entry in metrics.items():
            if entry.get("scores_finite") is not True:
                error = (
                    f"{entry.get('error_type', 'Error')}: "
                    f"{entry.get('error', 'unavailable')}"
                ).replace("|", "\\|")
                lines.append(
                    f"| `{group_name}.{metric_name}` | {error} | — | "
                    "False | False | False |"
                )
                continue
            first = float(entry["coarse_to_middle"]["score"])
            final = float(entry["middle_to_fine"]["score"])
            lines.append(
                f"| `{group_name}.{metric_name}` | {first:.9g} | "
                f"{final:.9g} | "
                f"{entry['final_score_within_tolerance']} | "
                f"{entry['score_nonincreasing']} | {entry['pass']} |"
            )

    lines.extend(["", "## Decision", ""])
    if gate["overall_pass"]:
        lines.append(
            "GO only for retaining the source-specified nearest-control-point "
            "TE discretization and preregistering a separate time-convergence "
            "gate. This result does not authorize downstream physics."
        )
    else:
        lines.append(
            "NO-GO for N2.6e1b1. The source-specified fixed-grid spatial "
            "refinement path failed at least one frozen completion, algebra, "
            "2%, or monotonicity condition; its thresholds remain unchanged."
        )
    lines.extend(
        [
            "",
            "Machine-readable values, both interval scores, source hashes, "
            "and any failure tracebacks are in the companion JSON.",
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
        "--json",
        type=Path,
        default=DEFAULT_JSON,
        help="machine-readable result path",
    )
    parser.add_argument(
        "--markdown",
        type=Path,
        default=DEFAULT_MARKDOWN,
        help="human-readable result path",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = execute_gate()
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
    return 0 if result["gate"]["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
