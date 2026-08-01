#!/usr/bin/env python3
"""Run the preregistered N2.6e1b attached-outer refinement gate.

This runner intentionally imports only the N2.6e1b geometry and attached
outer-flow runtime.  It contains no target-response reader and no force
metric.  See the frozen preregistration beside the result artifacts.
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


RUN_ID = "n26e1b_attached_outer_refinement_20260730"
PREREGISTRATION = (
    PLATFORM_DIR
    / "docs"
    / "diag"
    / "n26e1b_attached_outer_refinement_prereg_20260730.md"
)
DEFAULT_JSON = (
    PLATFORM_DIR
    / "docs"
    / "diag"
    / "n26e1b_attached_outer_refinement_result_20260730.json"
)
DEFAULT_MARKDOWN = (
    PLATFORM_DIR
    / "docs"
    / "diag"
    / "n26e1b_attached_outer_refinement_result_20260730.md"
)
RUNTIME_SOURCE = (
    PLATFORM_DIR / "claim_runtime" / "svi_dw_unsteady_outer_2d.py"
)

CHORD = 1.0
FREESTREAM_SPEED = 9.0
FINAL_ALPHA_RAD = math.radians(6.0)
RAMP_DURATION = 0.4
REFINEMENT_TOLERANCE = 0.02
ALGEBRAIC_TOLERANCE = 1.0e-9

PANEL_LEVELS = (16, 32, 64)
TIME_LEVELS = (16, 32, 64)
CORE_LEVELS = (0.04, 0.02, 0.01)
MIDDLE_PANELS = 32
MIDDLE_STEPS = 32
MIDDLE_CORE = 0.02

OBSERVABLE_SCALES = {
    "bound_circulation_ccw": FREESTREAM_SPEED * CHORD,
    "wake_circulation_ccw": FREESTREAM_SPEED * CHORD,
    "wake_centroid_x": CHORD,
    "wake_centroid_y": CHORD,
    "wake_first_moment_x": FREESTREAM_SPEED * CHORD**2,
    "wake_first_moment_y": FREESTREAM_SPEED * CHORD**2,
    "te_lower_downstream_trace": FREESTREAM_SPEED,
    "te_upper_downstream_trace": FREESTREAM_SPEED,
}
ALGEBRAIC_KEYS = (
    "maximum_kelvin_residual_over_Uc",
    "maximum_normal_bc_residual_over_U",
    "maximum_eq7_residual_over_c",
    "maximum_linear_system_residual_over_U",
    "maximum_eq8_residual_over_U",
)


@dataclass(frozen=True, order=True)
class RefinementCase:
    panels_per_side: int
    ramp_steps: int
    core_radius_over_c: float

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


def frozen_case_families() -> dict[str, tuple[RefinementCase, ...]]:
    return {
        "panel": tuple(
            RefinementCase(level, MIDDLE_STEPS, MIDDLE_CORE)
            for level in PANEL_LEVELS
        ),
        "time": tuple(
            RefinementCase(MIDDLE_PANELS, level, MIDDLE_CORE)
            for level in TIME_LEVELS
        ),
        "core": tuple(
            RefinementCase(MIDDLE_PANELS, MIDDLE_STEPS, level)
            for level in CORE_LEVELS
        ),
    }


def frozen_unique_cases() -> tuple[RefinementCase, ...]:
    cases = {
        case
        for family in frozen_case_families().values()
        for case in family
    }
    return tuple(sorted(cases))


def half_cosine_pitch(time_s: float) -> tuple[float, float]:
    """Return positive aerodynamic alpha and alpha-dot on the frozen ramp."""
    time_value = float(time_s)
    if not math.isfinite(time_value) or not 0.0 <= time_value <= RAMP_DURATION:
        raise ValueError("time_s must be finite and lie in [0, RAMP_DURATION]")
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
    """Dimensional sign predictor only; branch acceptance is internal."""
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
        "maximum_linear_system_residual_over_U": (
            float(
                np.max(
                    np.abs(solution.linear_system_residual),
                    initial=0.0,
                )
            )
            / FREESTREAM_SPEED
        ),
        "maximum_eq8_residual_over_U": (
            abs(float(solution.emission_residual)) / FREESTREAM_SPEED
        ),
    }


def _update_maxima(maxima: dict[str, float], values: Mapping[str, float]) -> None:
    for key in ALGEBRAIC_KEYS:
        value = float(values[key])
        if not math.isfinite(value):
            raise FloatingPointError(f"non-finite algebraic residual: {key}")
        maxima[key] = max(maxima[key], value)


def _wake_observables(final_solution: Any) -> dict[str, float]:
    blobs = final_solution.inputs.old_blobs
    circulations = [float(blob.circulation_ccw) for blob in blobs]
    positions = [
        np.asarray(blob.position_inertial, dtype=float) for blob in blobs
    ]

    segment = final_solution.near_wake_segment
    segment_midpoint_body = 0.5 * (
        segment.start_body + segment.end_body
    )
    segment_midpoint_inertial = (
        final_solution.inputs.kinematics.points_body_to_inertial(
            segment_midpoint_body[None, :]
        )[0]
    )
    circulations.append(float(final_solution.newborn_circulation_ccw))
    positions.append(np.asarray(segment_midpoint_inertial, dtype=float))

    gamma = np.asarray(circulations, dtype=float)
    position = np.asarray(positions, dtype=float)
    wake_circulation = float(np.sum(gamma))
    first_moment = np.sum(gamma[:, None] * position, axis=0)
    circulation_zero_tolerance = (
        4096.0
        * np.finfo(float).eps
        * max(
            float(np.sum(np.abs(gamma))),
            FREESTREAM_SPEED * CHORD,
        )
    )
    if abs(wake_circulation) <= circulation_zero_tolerance:
        raise FloatingPointError(
            "final wake circulation is numerically zero; signed centroid "
            "is undefined"
        )
    centroid = first_moment / wake_circulation
    diagnostic = final_solution.common_te_diagnostic
    return {
        "bound_circulation_ccw": float(
            final_solution.bound_circulation_ccw
        ),
        "wake_circulation_ccw": wake_circulation,
        "wake_centroid_x": float(centroid[0]),
        "wake_centroid_y": float(centroid[1]),
        "wake_first_moment_x": float(first_moment[0]),
        "wake_first_moment_y": float(first_moment[1]),
        "te_lower_downstream_trace": float(
            diagnostic.lower_downstream_trace
        ),
        "te_upper_downstream_trace": float(
            diagnostic.upper_downstream_trace
        ),
    }


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
    time_step = RAMP_DURATION / case.ramp_steps
    history = _initial_history()
    maxima = {key: 0.0 for key in ALGEBRAIC_KEYS}
    maximum_condition_number = 0.0
    maximum_geometry_iterations = 0

    for step_index in range(case.ramp_steps):
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
            newborn_core_radius=case.core_radius_over_c * CHORD,
        )
        _update_maxima(maxima, _solution_residuals(march.solution))
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
    _update_maxima(maxima, _solution_residuals(final_solution))
    maximum_condition_number = max(
        maximum_condition_number,
        float(final_solution.linear_system_condition_number),
    )
    maximum_geometry_iterations = max(
        maximum_geometry_iterations,
        int(final_solution.geometry_iterations),
    )
    observables = _wake_observables(final_solution)
    all_values = tuple(observables.values()) + tuple(maxima.values())
    if not all(math.isfinite(value) for value in all_values):
        raise FloatingPointError("case produced a non-finite metric")
    return {
        **case.as_dict(),
        "time_step_s": time_step,
        "stage_solves": case.ramp_steps + 1,
        "final_time_s": RAMP_DURATION,
        "observables": observables,
        "algebraic_residuals": maxima,
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


def convergence_change(
    middle_value: float,
    fine_value: float,
    physical_scale: float,
) -> dict[str, float | str | bool]:
    middle = float(middle_value)
    fine = float(fine_value)
    scale = float(physical_scale)
    if (
        not math.isfinite(middle)
        or not math.isfinite(fine)
        or not math.isfinite(scale)
        or scale <= 0.0
    ):
        raise ValueError("convergence inputs and scale must be finite")
    absolute_change = abs(fine - middle)
    if abs(fine) > REFINEMENT_TOLERANCE * scale:
        mode = "relative_to_fine"
        score = absolute_change / abs(fine)
    else:
        mode = "normalized_absolute_near_zero"
        score = absolute_change / scale
    return {
        "middle": middle,
        "fine": fine,
        "physical_scale": scale,
        "absolute_change": absolute_change,
        "mode": mode,
        "score": score,
        "threshold": REFINEMENT_TOLERANCE,
        "pass": score <= REFINEMENT_TOLERANCE,
    }


def evaluate_refinement(
    case_results: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    families = frozen_case_families()
    output: dict[str, Any] = {}
    for family_name, cases in families.items():
        middle_case = cases[-2]
        fine_case = cases[-1]
        middle_result = case_results.get(middle_case.case_id)
        fine_result = case_results.get(fine_case.case_id)
        family: dict[str, Any] = {
            "middle_case_id": middle_case.case_id,
            "fine_case_id": fine_case.case_id,
            "observables": {},
            "pass": False,
        }
        if (
            middle_result is None
            or fine_result is None
            or middle_result.get("status") != "completed"
            or fine_result.get("status") != "completed"
        ):
            family["failure"] = "middle or fine case did not complete"
            output[family_name] = family
            continue
        for name, scale in OBSERVABLE_SCALES.items():
            family["observables"][name] = convergence_change(
                middle_result["observables"][name],
                fine_result["observables"][name],
                scale,
            )
        family["pass"] = all(
            item["pass"] for item in family["observables"].values()
        )
        output[family_name] = family
    return output


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def execute_gate(
    *,
    cases: Iterable[RefinementCase] | None = None,
) -> dict[str, Any]:
    selected = tuple(cases) if cases is not None else frozen_unique_cases()
    result_by_id: dict[str, dict[str, Any]] = {}
    for case in selected:
        print(f"[{RUN_ID}] running {case.case_id}", flush=True)
        try:
            result_by_id[case.case_id] = run_refinement_case(case)
        except Exception as error:  # fail closed and preserve the other cases
            result_by_id[case.case_id] = {
                **case.as_dict(),
                "status": "failed",
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
    expected_ids = {case.case_id for case in frozen_unique_cases()}
    all_cases_present = set(result_by_id) == expected_ids
    all_cases_completed = all_cases_present and all(
        result_by_id[case_id].get("status") == "completed"
        for case_id in expected_ids
    )
    algebraic_pass = all_cases_completed and all(
        result_by_id[case_id].get("algebraic_gate_pass") is True
        for case_id in expected_ids
    )
    refinement_pass = all(
        family.get("pass") is True for family in comparisons.values()
    )
    overall_pass = all_cases_completed and algebraic_pass and refinement_pass
    return {
        "schema_version": 1,
        "run_id": RUN_ID,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "preregistration": str(PREREGISTRATION.relative_to(PLATFORM_DIR)),
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
            "ramp_steps": list(TIME_LEVELS),
            "core_radius_over_c": list(CORE_LEVELS),
            "middle": {
                "panels_per_side": MIDDLE_PANELS,
                "ramp_steps": MIDDLE_STEPS,
                "core_radius_over_c": MIDDLE_CORE,
            },
        },
        "thresholds": {
            "final_two_level_change": REFINEMENT_TOLERANCE,
            "normalized_algebraic_residual": ALGEBRAIC_TOLERANCE,
        },
        "environment": {
            "python": sys.version,
            "python_platform": python_platform.platform(),
            "numpy": np.__version__,
            "runtime_source": str(RUNTIME_SOURCE.relative_to(PLATFORM_DIR)),
            "runtime_source_sha256": _sha256(RUNTIME_SOURCE),
            "runner_sha256": _sha256(Path(__file__).resolve()),
            "preregistration_sha256": _sha256(PREREGISTRATION),
        },
        "cases": result_by_id,
        "comparisons": comparisons,
        "gate": {
            "all_seven_unique_cases_present": all_cases_present,
            "all_cases_completed": all_cases_completed,
            "all_algebraic_residuals_pass": algebraic_pass,
            "all_final_two_level_changes_pass": refinement_pass,
            "overall_pass": overall_pass,
            "verdict": "GO" if overall_pass else "NO-GO",
        },
    }


def render_markdown(result: Mapping[str, Any]) -> str:
    gate = result["gate"]
    lines = [
        "# N2.6e1b attached-outer numerical refinement result",
        "",
        f"- Run: `{result['run_id']}`",
        f"- Verdict: **{gate['verdict']}**",
        f"- Generated (UTC): `{result['generated_at_utc']}`",
        (
            "- Scope: attached outer-flow numerical admissibility only; "
            "no target-response or force validation."
        ),
        "",
        "## Gate summary",
        "",
        "| gate | pass |",
        "|---|---:|",
        (
            "| seven frozen unique cases present | "
            f"{gate['all_seven_unique_cases_present']} |"
        ),
        f"| all cases completed | {gate['all_cases_completed']} |",
        (
            "| all normalized algebraic residuals <= 1e-9 | "
            f"{gate['all_algebraic_residuals_pass']} |"
        ),
        (
            "| all final-two-level observable changes <= 2% | "
            f"{gate['all_final_two_level_changes_pass']} |"
        ),
        "",
        "## Cases",
        "",
        (
            "| case | status | Gamma_B | Gamma_W | x_Gamma/c | "
            "y_Gamma/c | max Kelvin | max BC | max Eq7 |"
        ),
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for case_id in sorted(result["cases"]):
        case = result["cases"][case_id]
        if case["status"] != "completed":
            error = (
                f"{case.get('error_type', 'Error')}: "
                f"{case.get('error', 'unknown failure')}"
            ).replace("|", "\\|")
            lines.append(
                f"| `{case_id}` | failed: {error} | — | — | — | — | — | — | — |"
            )
            continue
        observable = case["observables"]
        residual = case["algebraic_residuals"]
        lines.append(
            f"| `{case_id}` | completed | "
            f"{observable['bound_circulation_ccw']:.9g} | "
            f"{observable['wake_circulation_ccw']:.9g} | "
            f"{observable['wake_centroid_x'] / CHORD:.9g} | "
            f"{observable['wake_centroid_y'] / CHORD:.9g} | "
            f"{residual['maximum_kelvin_residual_over_Uc']:.3e} | "
            f"{residual['maximum_normal_bc_residual_over_U']:.3e} | "
            f"{residual['maximum_eq7_residual_over_c']:.3e} |"
        )

    lines.extend(
        [
            "",
            "## Final-two-level comparisons",
            "",
            "| family | middle -> fine | worst score | pass |",
            "|---|---|---:|---:|",
        ]
    )
    for name in ("panel", "time", "core"):
        family = result["comparisons"][name]
        scores = [
            float(item["score"])
            for item in family.get("observables", {}).values()
        ]
        worst = max(scores) if scores else math.nan
        lines.append(
            f"| {name} | `{family['middle_case_id']}` -> "
            f"`{family['fine_case_id']}` | "
            f"{worst:.6g} | {family['pass']} |"
        )
        if "failure" in family:
            lines.append(
                f"\nFailure: {family['failure']}."
            )

    lines.extend(
        [
            "",
            "## Decision",
            "",
        ]
    )
    if gate["overall_pass"]:
        lines.append(
            "GO for the narrow N2.6e1b attached-outer numerical gate. "
            "This does not authorize IBL, separation, pressure, force, or "
            "target-response claims."
        )
    else:
        lines.append(
            "NO-GO for the current N2.6e1b attached-outer numerical gate. "
            "The frozen levels and thresholds were not changed; the failed "
            "case or observable must be diagnosed before downstream use."
        )
    lines.extend(
        [
            "",
            "Machine-readable details, including every observable score and "
            "traceback, are in the companion JSON.",
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
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(render_markdown(result), encoding="utf-8")


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
    write_results(result, json_path=args.json, markdown_path=args.markdown)
    print(
        f"[{RUN_ID}] verdict={result['gate']['verdict']} "
        f"json={args.json} markdown={args.markdown}",
        flush=True,
    )
    return 0 if result["gate"]["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
