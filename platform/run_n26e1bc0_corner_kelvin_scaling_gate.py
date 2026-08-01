#!/usr/bin/env python3
"""Run the frozen N2.6e1bc0 regular-corner / Kelvin scaling gate.

This shadow diagnostic reuses the source-faithful attached-outer runner and
runtime.  It reads only geometry, velocity traces, circulation ledgers,
branch identity, and algebraic residuals.  It is not a candidate solver and
cannot promote the parent claim.
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

import run_n26e1b1_source_faithful_te_refinement as source_gate  # noqa: E402
from claim_runtime.svi_dw_types import (  # noqa: E402
    NACA4SectionConfig,
    build_naca4_actual_surface,
)
from claim_runtime.svi_dw_unsteady_outer_2d import (  # noqa: E402
    AttachedOuterStepInput2D,
    march_attached_unsteady_outer_explicit_euler,
    solve_attached_unsteady_outer_step,
)


RUN_ID = "n26e1bc0_corner_kelvin_scaling_20260730"
PREREGISTRATION = (
    PLATFORM_DIR
    / "docs"
    / "diag"
    / "n26e1bc0_corner_kelvin_scaling_prereg_20260730.md"
)
DEFAULT_JSON = (
    PLATFORM_DIR
    / "docs"
    / "diag"
    / "n26e1bc0_corner_kelvin_scaling_result_20260730.json"
)
DEFAULT_MARKDOWN = (
    PLATFORM_DIR
    / "docs"
    / "diag"
    / "n26e1bc0_corner_kelvin_scaling_result_20260730.md"
)
EXPECTED_PREREGISTRATION_SHA256 = (
    "86c972bc49391b3ecae26ab11191b27213cd0911a89519fc6b97294f2e03383e"
)

CHORD = 1.0
FREESTREAM_SPEED = 9.0
RAMP_DURATION = 0.4
OBSERVATION_TIME = 0.2
CORE_RADIUS_OVER_C = 0.02
REGULAR_VELOCITY_EXPONENT = 0.06068033385533589
REGULAR_ONLY_BIRTH_EXPONENT = (
    (1.0 + REGULAR_VELOCITY_EXPONENT)
    / (1.0 - REGULAR_VELOCITY_EXPONENT)
)

SPATIAL_LEVELS = (64, 128, 256)
SPATIAL_TIME_STEP = 0.00625
TEMPORAL_PANEL_LEVEL = 256
TEMPORAL_TIME_STEPS = (0.025, 0.0125, 0.00625, 0.003125)
TEMPORAL_RAMP_STEPS = (16, 32, 64, 128)
COMMON_ANCHOR = (256, 0.003125)

ALGEBRAIC_TOLERANCE = 1.0e-9
GENERIC_BIRTH_FLOOR_OVER_UC = 1.0e-10
MODAL_CHANGE_TOLERANCE = 0.02
OLS_EXPONENT_TOLERANCE = 0.03
FINE_LOCAL_ORDER_TOLERANCE = 0.05
TIME_ALIGNMENT_TOLERANCE = 64.0 * np.finfo(float).eps

ALGEBRAIC_KEYS = tuple(source_gate.ALGEBRAIC_KEYS)
TRACE_NAMES = {
    "lower": "lower_downstream_trace",
    "upper": "upper_downstream_trace",
    "mean": "mean_downstream_trace",
}
SOURCE_FILES = {
    "runner": Path(__file__).resolve(),
    "source_faithful_runner": (
        PLATFORM_DIR / "run_n26e1b1_source_faithful_te_refinement.py"
    ),
    "types": PLATFORM_DIR / "claim_runtime" / "svi_dw_types.py",
    "outer": PLATFORM_DIR / "claim_runtime" / "svi_dw_outer_2d.py",
    "unsteady_outer": (
        PLATFORM_DIR / "claim_runtime" / "svi_dw_unsteady_outer_2d.py"
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _time_token(time_step: float) -> str:
    return f"{time_step:.6f}".rstrip("0").rstrip(".").replace(".", "p")


@dataclass(frozen=True, order=True)
class ScalingCase:
    """One immutable point on either frozen independent axis."""

    panels_per_side: int
    time_step_s: float

    def __post_init__(self) -> None:
        panels = int(self.panels_per_side)
        time_step = float(self.time_step_s)
        allowed = {
            *((level, SPATIAL_TIME_STEP) for level in SPATIAL_LEVELS),
            *((TEMPORAL_PANEL_LEVEL, step) for step in TEMPORAL_TIME_STEPS),
        }
        if (
            panels != self.panels_per_side
            or not math.isfinite(time_step)
            or (panels, time_step) not in allowed
        ):
            raise ValueError("case is not on either frozen preregistered axis")
        ratio = RAMP_DURATION / time_step
        ramp_steps = round(ratio)
        if abs(ratio - ramp_steps) > TIME_ALIGNMENT_TOLERANCE:
            raise ValueError("time step does not divide the frozen ramp")
        observation_ratio = OBSERVATION_TIME / time_step
        observation_step = round(observation_ratio)
        if (
            abs(observation_ratio - observation_step)
            > TIME_ALIGNMENT_TOLERANCE
        ):
            raise ValueError("t*=0.2 s does not lie on the time grid")
        object.__setattr__(self, "panels_per_side", panels)
        object.__setattr__(self, "time_step_s", time_step)

    @property
    def case_id(self) -> str:
        return (
            f"p{self.panels_per_side}_dt{_time_token(self.time_step_s)}"
        )

    @property
    def ramp_steps(self) -> int:
        return round(RAMP_DURATION / self.time_step_s)

    @property
    def observation_step(self) -> int:
        return round(OBSERVATION_TIME / self.time_step_s)

    def as_dict(self) -> dict[str, int | float | str]:
        return {
            "case_id": self.case_id,
            "panels_per_side": self.panels_per_side,
            "time_step_s": self.time_step_s,
            "ramp_steps": self.ramp_steps,
            "observation_step": self.observation_step,
            "observation_time_s": OBSERVATION_TIME,
        }


def spatial_cases() -> tuple[ScalingCase, ...]:
    return tuple(
        ScalingCase(level, SPATIAL_TIME_STEP)
        for level in SPATIAL_LEVELS
    )


def temporal_cases() -> tuple[ScalingCase, ...]:
    return tuple(
        ScalingCase(TEMPORAL_PANEL_LEVEL, time_step)
        for time_step in TEMPORAL_TIME_STEPS
    )


def frozen_cases() -> tuple[ScalingCase, ...]:
    unique: dict[str, ScalingCase] = {}
    for case in (*spatial_cases(), *temporal_cases()):
        unique.setdefault(case.case_id, case)
    return tuple(unique.values())


def _protocol_manifest() -> dict[str, Any]:
    return {
        "claim": "N2.6e1bc0-FCR-KELVIN-SCALING",
        "section": {
            "name": "closed NACA0015",
            "chord_m": CHORD,
        },
        "motion": {
            "freestream_speed_m_s": FREESTREAM_SPEED,
            "quarter_chord_pivot_x_over_c": 0.25,
            "schedule": "alpha=3deg*(1-cos(pi*t/0.4))",
            "ramp_duration_s": RAMP_DURATION,
            "observation_time_s": OBSERVATION_TIME,
        },
        "wake": {
            "topology": "single_TE_material_wake",
            "core_radius_over_c": CORE_RADIUS_OVER_C,
            "wall_transpiration": 0.0,
        },
        "axes": {
            "spatial": [case.as_dict() for case in spatial_cases()],
            "temporal": [case.as_dict() for case in temporal_cases()],
            "shared_axis_case_id": ScalingCase(
                256, SPATIAL_TIME_STEP
            ).case_id,
            "common_anchor_case_id": ScalingCase(*COMMON_ANCHOR).case_id,
        },
        "geometry_exponents": {
            "beta": REGULAR_VELOCITY_EXPONENT,
            "p_star": REGULAR_ONLY_BIRTH_EXPONENT,
        },
        "thresholds": {
            "generic_birth_floor_over_Uc": GENERIC_BIRTH_FLOOR_OVER_UC,
            "maximum_algebraic_residual": ALGEBRAIC_TOLERANCE,
            "modal_change_128_to_256": MODAL_CHANGE_TOLERANCE,
            "absolute_ols_exponent_difference": OLS_EXPONENT_TOLERANCE,
            "absolute_fine_local_order_difference": (
                FINE_LOCAL_ORDER_TOLERANCE
            ),
        },
        "observables": {
            "traces": list(TRACE_NAMES),
            "terminal_radius": "symmetric terminal panel midpoint",
            "birth": "-(Gamma_b(t*)-Gamma_b(t*-dt))",
            "algebraic_residuals": list(ALGEBRAIC_KEYS),
            "branch_identity": "near_wake_segment.orientation_side",
        },
    }


def _terminal_midpoint_radius(surface: Any) -> dict[str, float]:
    trailing_edge = np.asarray(surface.upper_nodes[-1], dtype=float)
    lower = float(
        np.linalg.norm(surface.panel_midpoints[0] - trailing_edge)
    )
    upper = float(
        np.linalg.norm(surface.panel_midpoints[-1] - trailing_edge)
    )
    scale = max(lower, upper, CHORD)
    tolerance = 4096.0 * np.finfo(float).eps * scale
    if (
        not math.isfinite(lower)
        or not math.isfinite(upper)
        or lower <= 0.0
        or upper <= 0.0
        or abs(lower - upper) > tolerance
    ):
        raise ValueError(
            "closed symmetric NACA0015 terminal midpoint radii disagree"
        )
    return {
        "lower_m": lower,
        "upper_m": upper,
        "canonical_m": 0.5 * (lower + upper),
        "symmetry_error_m": abs(lower - upper),
    }


def _update_residual_maxima(
    maxima: dict[str, float],
    solution: Any,
) -> None:
    residuals = source_gate._solution_residuals(solution)
    source_gate._update_residual_maxima(maxima, residuals)


def _stage_observables(
    solution: Any,
    *,
    previous_bound_circulation_ccw: float,
    terminal_radius: Mapping[str, float],
) -> dict[str, Any]:
    diagnostic = solution.common_te_diagnostic
    radius = float(terminal_radius["canonical_m"])
    traces = {
        short_name: float(getattr(diagnostic, attribute))
        for short_name, attribute in TRACE_NAMES.items()
    }
    modal = {
        name: value / radius**REGULAR_VELOCITY_EXPONENT
        for name, value in traces.items()
    }
    delta_bound = (
        float(solution.bound_circulation_ccw)
        - float(previous_bound_circulation_ccw)
    )
    actual_birth = -delta_bound
    solver_birth = float(solution.newborn_circulation_ccw)
    identity_residual = actual_birth - solver_birth
    values = (
        radius,
        *traces.values(),
        *modal.values(),
        delta_bound,
        actual_birth,
        solver_birth,
        identity_residual,
    )
    if not all(math.isfinite(value) for value in values):
        raise FloatingPointError("stage observables contain a non-finite value")
    return {
        "terminal_midpoint_radius": dict(terminal_radius),
        "terminal_traces_m_s": traces,
        "modal_A_u_over_r_beta": modal,
        "previous_bound_circulation_ccw_m2_s": (
            float(previous_bound_circulation_ccw)
        ),
        "bound_circulation_ccw_m2_s": (
            float(solution.bound_circulation_ccw)
        ),
        "delta_bound_circulation_ccw_m2_s": delta_bound,
        "actual_newborn_circulation_ccw_m2_s": actual_birth,
        "solver_newborn_circulation_ccw_m2_s": solver_birth,
        "birth_identity_residual_m2_s": identity_residual,
        "birth_magnitude_over_Uc": (
            abs(actual_birth) / (FREESTREAM_SPEED * CHORD)
        ),
        "branch_identity": solution.near_wake_segment.orientation_side,
    }


def run_scaling_case(case: ScalingCase) -> dict[str, Any]:
    if not isinstance(case, ScalingCase):
        raise TypeError("case must be ScalingCase")
    surface = build_naca4_actual_surface(
        NACA4SectionConfig(
            maximum_camber=0.0,
            camber_location=0.4,
            thickness_ratio=0.15,
            chord=CHORD,
            closed_trailing_edge=True,
        ),
        panels_per_side=case.panels_per_side,
    )
    terminal_radius = _terminal_midpoint_radius(surface)
    history = source_gate._initial_history()
    maxima = {key: 0.0 for key in ALGEBRAIC_KEYS}
    branch_stages: list[dict[str, Any]] = []
    maximum_condition_number = 0.0
    maximum_geometry_iterations = 0

    for stage_index in range(case.observation_step):
        stage_time = stage_index * case.time_step_s
        march = march_attached_unsteady_outer_explicit_euler(
            surface=surface,
            kinematics=source_gate.pitching_kinematics(stage_time),
            freestream_velocity_inertial=(FREESTREAM_SPEED, 0.0),
            time_step=case.time_step_s,
            history=history,
            predicted_bound_circulation_change_ccw=(
                source_gate._analytic_change_predictor(
                    stage_time,
                    case.time_step_s,
                )
            ),
            newborn_core_radius=CORE_RADIUS_OVER_C * CHORD,
        )
        solution = march.solution
        _update_residual_maxima(maxima, solution)
        branch_stages.append(
            {
                "stage_index": stage_index,
                "stage_time_s": stage_time,
                "identity": solution.near_wake_segment.orientation_side,
            }
        )
        maximum_condition_number = max(
            maximum_condition_number,
            float(solution.linear_system_condition_number),
        )
        maximum_geometry_iterations = max(
            maximum_geometry_iterations,
            int(solution.geometry_iterations),
        )
        history = march.history_next

    time_error = history.stage_time - OBSERVATION_TIME
    if abs(time_error) > TIME_ALIGNMENT_TOLERANCE:
        raise ValueError("history did not land exactly on t*=0.2 s")
    final_inputs = AttachedOuterStepInput2D.from_history(
        surface=surface,
        kinematics=source_gate.pitching_kinematics(OBSERVATION_TIME),
        freestream_velocity_inertial=(FREESTREAM_SPEED, 0.0),
        time_step=case.time_step_s,
        history=history,
        predicted_bound_circulation_change_ccw=(
            source_gate._analytic_change_predictor(
                OBSERVATION_TIME,
                case.time_step_s,
            )
        ),
    )
    final_solution = solve_attached_unsteady_outer_step(final_inputs)
    _update_residual_maxima(maxima, final_solution)
    branch_stages.append(
        {
            "stage_index": case.observation_step,
            "stage_time_s": OBSERVATION_TIME,
            "identity": (
                final_solution.near_wake_segment.orientation_side
            ),
        }
    )
    maximum_condition_number = max(
        maximum_condition_number,
        float(final_solution.linear_system_condition_number),
    )
    maximum_geometry_iterations = max(
        maximum_geometry_iterations,
        int(final_solution.geometry_iterations),
    )
    observables = _stage_observables(
        final_solution,
        previous_bound_circulation_ccw=history.bound_circulation_ccw,
        terminal_radius=terminal_radius,
    )
    identities = [item["identity"] for item in branch_stages]
    unique_identities = sorted(set(identities))
    branch_identity_sha256 = _canonical_sha256(branch_stages)
    finite_health = all(
        math.isfinite(value)
        for value in (
            *maxima.values(),
            maximum_condition_number,
            float(maximum_geometry_iterations),
        )
    )
    return {
        **case.as_dict(),
        "status": "completed",
        "stage_solves": case.observation_step + 1,
        "history_time_alignment_error_s": time_error,
        "observables_at_t_star": observables,
        "maximum_algebraic_residuals": maxima,
        "maximum_linear_system_condition_number": (
            maximum_condition_number
        ),
        "maximum_geometry_iterations": maximum_geometry_iterations,
        "branch_audit": {
            "identity_by_stage": branch_stages,
            "identity_sequence_sha256": branch_identity_sha256,
            "unique_identities": unique_identities,
            "internally_consistent": len(unique_identities) == 1,
            "final_identity": identities[-1],
        },
        "finite_health": finite_health,
        "algebraic_gate_pass": all(
            maxima[key] <= ALGEBRAIC_TOLERANCE
            for key in ALGEBRAIC_KEYS
        ),
        "generic_birth_gate_pass": (
            observables["birth_magnitude_over_Uc"]
            > GENERIC_BIRTH_FLOOR_OVER_UC
        ),
    }


def _completed_case(
    case_results: Mapping[str, Mapping[str, Any]],
    case: ScalingCase,
) -> Mapping[str, Any]:
    result = case_results.get(case.case_id)
    if not isinstance(result, Mapping) or result.get("status") != "completed":
        raise ValueError(f"missing completed case {case.case_id}")
    return result


def _relative_change(middle: float, fine: float) -> float:
    middle_value = float(middle)
    fine_value = float(fine)
    if (
        not math.isfinite(middle_value)
        or not math.isfinite(fine_value)
        or fine_value == 0.0
    ):
        raise ValueError("relative change requires finite values and nonzero fine")
    return abs(fine_value - middle_value) / abs(fine_value)


def four_point_log_ols(
    time_steps: Iterable[float],
    birth_magnitudes: Iterable[float],
) -> dict[str, Any]:
    dt = np.asarray(tuple(time_steps), dtype=float)
    gamma = np.asarray(tuple(birth_magnitudes), dtype=float)
    if (
        dt.shape != (4,)
        or gamma.shape != (4,)
        or not np.all(np.isfinite(dt))
        or not np.all(np.isfinite(gamma))
        or np.any(dt <= 0.0)
        or np.any(gamma <= 0.0)
        or len(set(dt.tolist())) != 4
    ):
        raise ValueError("four positive finite distinct dt and birth values required")
    x = np.log(dt)
    y = np.log(gamma)
    design = np.column_stack((x, np.ones(4, dtype=float)))
    coefficients, _, rank, singular_values = np.linalg.lstsq(
        design,
        y,
        rcond=None,
    )
    if rank != 2:
        raise ValueError("four-point log OLS design is rank deficient")
    slope = float(coefficients[0])
    intercept = float(coefficients[1])
    fitted = design @ coefficients
    residual = y - fitted
    total = y - float(np.mean(y))
    residual_sum_squares = float(residual @ residual)
    total_sum_squares = float(total @ total)
    r_squared = (
        1.0 - residual_sum_squares / total_sum_squares
        if total_sum_squares > 0.0
        else 1.0
    )
    local_orders = [
        math.log(gamma[index + 1] / gamma[index])
        / math.log(dt[index + 1] / dt[index])
        for index in range(3)
    ]
    values = (
        slope,
        intercept,
        residual_sum_squares,
        r_squared,
        *local_orders,
        *singular_values,
    )
    if not all(math.isfinite(float(value)) for value in values):
        raise FloatingPointError("log OLS produced a non-finite value")
    return {
        "method": "four_point_ordinary_least_squares_log_abs_birth_on_log_dt",
        "time_steps_s_in_preregistered_order": dt.tolist(),
        "birth_magnitudes_m2_s_in_preregistered_order": gamma.tolist(),
        "slope_p_K": slope,
        "intercept": intercept,
        "residual_sum_squares": residual_sum_squares,
        "r_squared": r_squared,
        "design_rank": int(rank),
        "design_singular_values": singular_values.tolist(),
        "local_orders_consecutive_pairs": local_orders,
        "fine_pair_local_order": local_orders[-1],
    }


def evaluate_gate(
    case_results: Mapping[str, Mapping[str, Any]],
    *,
    preregistration_hash_matches: bool,
) -> dict[str, Any]:
    expected_cases = frozen_cases()
    expected_ids = {case.case_id for case in expected_cases}
    present_ids = set(case_results)
    exact_case_set = present_ids == expected_ids
    completed = exact_case_set and all(
        case_results[case_id].get("status") == "completed"
        for case_id in expected_ids
    )

    health = {
        "preregistration_hash_matches": preregistration_hash_matches,
        "exact_frozen_case_set": exact_case_set,
        "all_cases_completed": completed,
        "all_values_finite": False,
        "all_times_aligned": False,
        "all_branches_internally_consistent": False,
        "all_branch_identities_equal": False,
        "all_births_generic": False,
        "all_five_algebraic_residuals_pass": False,
        "birth_identity_residuals_pass": False,
    }
    physics: dict[str, Any] = {
        "available": False,
        "modal_changes_128_to_256": {},
        "all_modal_changes_within_two_percent": False,
        "time_scaling": None,
        "ols_exponent_matches_p_star": False,
        "fine_local_order_matches_p_star": False,
        "necessary_scaling_gate_pass": False,
        "status": "open",
    }
    if completed:
        completed_results = [
            case_results[case.case_id] for case in expected_cases
        ]
        health["all_values_finite"] = all(
            result.get("finite_health") is True
            for result in completed_results
        )
        health["all_times_aligned"] = all(
            abs(float(result["history_time_alignment_error_s"]))
            <= TIME_ALIGNMENT_TOLERANCE
            for result in completed_results
        )
        health["all_branches_internally_consistent"] = all(
            result["branch_audit"]["internally_consistent"] is True
            for result in completed_results
        )
        all_unique_identities = {
            identity
            for result in completed_results
            for identity in result["branch_audit"]["unique_identities"]
        }
        health["all_branch_identities_equal"] = (
            len(all_unique_identities) == 1
        )
        health["all_births_generic"] = all(
            result.get("generic_birth_gate_pass") is True
            for result in completed_results
        )
        health["all_five_algebraic_residuals_pass"] = all(
            result.get("algebraic_gate_pass") is True
            and set(result["maximum_algebraic_residuals"])
            == set(ALGEBRAIC_KEYS)
            for result in completed_results
        )
        identity_tolerance = (
            ALGEBRAIC_TOLERANCE * FREESTREAM_SPEED * CHORD
        )
        health["birth_identity_residuals_pass"] = all(
            abs(
                float(
                    result["observables_at_t_star"][
                        "birth_identity_residual_m2_s"
                    ]
                )
            )
            <= identity_tolerance
            for result in completed_results
        )

    protocol_pass = all(health.values())
    if protocol_pass:
        middle_case = _completed_case(
            case_results,
            ScalingCase(128, SPATIAL_TIME_STEP),
        )
        fine_case = _completed_case(
            case_results,
            ScalingCase(256, SPATIAL_TIME_STEP),
        )
        modal_changes = {
            name: _relative_change(
                middle_case["observables_at_t_star"][
                    "modal_A_u_over_r_beta"
                ][name],
                fine_case["observables_at_t_star"][
                    "modal_A_u_over_r_beta"
                ][name],
            )
            for name in TRACE_NAMES
        }
        time_results = [
            _completed_case(case_results, case)
            for case in temporal_cases()
        ]
        scaling = four_point_log_ols(
            TEMPORAL_TIME_STEPS,
            (
                abs(
                    float(
                        result["observables_at_t_star"][
                            "actual_newborn_circulation_ccw_m2_s"
                        ]
                    )
                )
                for result in time_results
            ),
        )
        modal_pass = all(
            change <= MODAL_CHANGE_TOLERANCE
            for change in modal_changes.values()
        )
        ols_difference = abs(
            float(scaling["slope_p_K"])
            - REGULAR_ONLY_BIRTH_EXPONENT
        )
        local_difference = abs(
            float(scaling["fine_pair_local_order"])
            - REGULAR_ONLY_BIRTH_EXPONENT
        )
        ols_pass = ols_difference <= OLS_EXPONENT_TOLERANCE
        local_pass = local_difference <= FINE_LOCAL_ORDER_TOLERANCE
        necessary_pass = modal_pass and ols_pass and local_pass
        physics.update(
            {
                "available": True,
                "modal_changes_128_to_256": modal_changes,
                "all_modal_changes_within_two_percent": modal_pass,
                "time_scaling": scaling,
                "absolute_ols_difference_from_p_star": ols_difference,
                "ols_exponent_matches_p_star": ols_pass,
                "absolute_fine_local_order_difference_from_p_star": (
                    local_difference
                ),
                "fine_local_order_matches_p_star": local_pass,
                "necessary_scaling_gate_pass": necessary_pass,
                "status": (
                    "REPRESENTATION-GO"
                    if necessary_pass
                    else "PHYSICS-NO-GO"
                ),
            }
        )

    if not protocol_pass:
        verdict = "PROTOCOL-NO-GO"
        claim_state = "open"
    elif physics["necessary_scaling_gate_pass"]:
        verdict = "REPRESENTATION-GO"
        claim_state = "parent_open"
    else:
        verdict = "PHYSICS-NO-GO"
        claim_state = "N2.6e1bc0_regular_corner_only_falsified_frozen"
    return {
        "protocol": {
            "health": health,
            "pass": protocol_pass,
            "status": (
                "PROTOCOL-PASS" if protocol_pass else "PROTOCOL-NO-GO"
            ),
        },
        "physics": physics,
        "verdict": verdict,
        "claim_state": claim_state,
    }


def execute_gate(
    *,
    cases: Iterable[ScalingCase] | None = None,
) -> dict[str, Any]:
    preregistration_sha256 = _sha256(PREREGISTRATION)
    preregistration_hash_matches = (
        preregistration_sha256 == EXPECTED_PREREGISTRATION_SHA256
    )
    selected = tuple(cases) if cases is not None else frozen_cases()
    if not all(isinstance(case, ScalingCase) for case in selected):
        raise TypeError("cases must contain only ScalingCase values")
    if len({case.case_id for case in selected}) != len(selected):
        raise ValueError("duplicate cases are not allowed")

    result_by_id: dict[str, dict[str, Any]] = {}
    if preregistration_hash_matches:
        for case in selected:
            print(f"[{RUN_ID}] running {case.case_id}", flush=True)
            try:
                result_by_id[case.case_id] = run_scaling_case(case)
            except Exception as error:
                result_by_id[case.case_id] = {
                    **case.as_dict(),
                    "status": "failed",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                    "finite_health": False,
                    "algebraic_gate_pass": False,
                    "generic_birth_gate_pass": False,
                }
                print(
                    f"[{RUN_ID}] {case.case_id} failed: "
                    f"{type(error).__name__}: {error}",
                    flush=True,
                )

    decision = evaluate_gate(
        result_by_id,
        preregistration_hash_matches=preregistration_hash_matches,
    )
    protocol_manifest = _protocol_manifest()
    source_hashes = {
        name: {
            "path": str(path.relative_to(PLATFORM_DIR)),
            "sha256": _sha256(path),
        }
        for name, path in SOURCE_FILES.items()
    }
    return {
        "schema_version": 1,
        "run_id": RUN_ID,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "role": "preregistered_shadow_scaling_diagnostic",
        "candidate_solver": False,
        "candidate_promotion_allowed": False,
        "protocol_manifest": protocol_manifest,
        "reproducibility": {
            "preregistration": {
                "path": str(PREREGISTRATION.relative_to(PLATFORM_DIR)),
                "expected_sha256_frozen_before_implementation": (
                    EXPECTED_PREREGISTRATION_SHA256
                ),
                "actual_sha256": preregistration_sha256,
                "matches": preregistration_hash_matches,
            },
            "input_manifest_sha256": _canonical_sha256(
                protocol_manifest
            ),
            "source_files": source_hashes,
            "source_manifest_sha256": _canonical_sha256(source_hashes),
            "environment": {
                "python": sys.version,
                "python_platform": python_platform.platform(),
                "numpy": np.__version__,
            },
        },
        "cases": result_by_id,
        "decision": decision,
    }


def render_markdown(result: Mapping[str, Any]) -> str:
    decision = result["decision"]
    protocol = decision["protocol"]
    physics = decision["physics"]
    lines = [
        "# N2.6e1bc0 regular-corner / Kelvin scaling result",
        "",
        f"- Run: `{result['run_id']}`",
        f"- Verdict: **{decision['verdict']}**",
        f"- Protocol: **{protocol['status']}**",
        f"- Physics state: **{physics['status']}**",
        f"- Generated (UTC): `{result['generated_at_utc']}`",
        (
            "- Scope: frozen shadow scaling diagnostic only; no parent-claim "
            "promotion."
        ),
        "",
        "## Protocol health",
        "",
        "| check | pass |",
        "|---|---:|",
    ]
    for name, passed in protocol["health"].items():
        lines.append(f"| `{name}` | {passed} |")

    lines.extend(
        [
            "",
            "## Frozen cases at t*=0.2 s",
            "",
            (
                "| case | status | branch | r/c | A_lower | A_upper | "
                "A_mean | actual Gamma_birth/(Uc) | max residual |"
            ),
            "|---|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    ordered_ids = [case.case_id for case in frozen_cases()]
    for case_id in ordered_ids:
        case = result["cases"].get(case_id)
        if not isinstance(case, Mapping):
            lines.append(
                f"| `{case_id}` | missing | — | — | — | — | — | — | — |"
            )
            continue
        if case.get("status") != "completed":
            error = (
                f"{case.get('error_type', 'Error')}: "
                f"{case.get('error', 'unknown failure')}"
            ).replace("|", "\\|")
            lines.append(
                f"| `{case_id}` | failed: {error} | — | — | — | — | "
                "— | — | — |"
            )
            continue
        observable = case["observables_at_t_star"]
        modal = observable["modal_A_u_over_r_beta"]
        maximum_residual = max(
            float(value)
            for value in case["maximum_algebraic_residuals"].values()
        )
        lines.append(
            f"| `{case_id}` | completed | "
            f"{observable['branch_identity']} | "
            f"{observable['terminal_midpoint_radius']['canonical_m'] / CHORD:.10g} | "
            f"{modal['lower']:.10g} | {modal['upper']:.10g} | "
            f"{modal['mean']:.10g} | "
            f"{observable['actual_newborn_circulation_ccw_m2_s'] / (FREESTREAM_SPEED * CHORD):.10g} | "
            f"{maximum_residual:.3e} |"
        )

    lines.extend(["", "## Physics scaling decision", ""])
    if not physics["available"]:
        lines.append(
            "Physics verdict unavailable because the protocol/implementation "
            "health gate failed; the subclaim remains open."
        )
    else:
        scaling = physics["time_scaling"]
        lines.extend(
            [
                (
                    f"- regular-only predicted exponent `p* = "
                    f"{REGULAR_ONLY_BIRTH_EXPONENT:.10f}`"
                ),
                (
                    f"- four-point log OLS `p_K = "
                    f"{scaling['slope_p_K']:.10f}`"
                ),
                (
                    "- local orders (coarse to fine): `"
                    + ", ".join(
                        f"{value:.10f}"
                        for value in scaling[
                            "local_orders_consecutive_pairs"
                        ]
                    )
                    + "`"
                ),
                (
                    "- modal changes 128->256: "
                    + ", ".join(
                        f"`{name}={100.0 * value:.6f}%`"
                        for name, value in physics[
                            "modal_changes_128_to_256"
                        ].items()
                    )
                ),
                "",
                (
                    "The necessary regular-corner-only scaling gate "
                    + (
                        "passed. This is only REPRESENTATION-GO; the parent "
                        "claim remains open."
                        if physics["necessary_scaling_gate_pass"]
                        else (
                            "failed. N2.6e1bc0 regular-corner-only is "
                            "falsified/frozen; broader moving-interface "
                            "circulation theories are not adjudicated."
                        )
                    )
                ),
            ]
        )
    lines.extend(
        [
            "",
            "The companion JSON contains every stage branch identity, all five "
            "residual maxima, actual-versus-solver birth ledgers, the fixed "
            "four-point fit, and complete input/source hashes.",
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
            allow_nan=False,
            indent=2,
            sort_keys=True,
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
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument(
        "--markdown",
        type=Path,
        default=DEFAULT_MARKDOWN,
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    arguments = parse_args(argv)
    result = execute_gate()
    write_results(
        result,
        json_path=arguments.json,
        markdown_path=arguments.markdown,
    )
    print(
        f"[{RUN_ID}] verdict={result['decision']['verdict']} "
        f"json={arguments.json} markdown={arguments.markdown}",
        flush=True,
    )
    return (
        2
        if result["decision"]["verdict"] == "PROTOCOL-NO-GO"
        else 0
    )


if __name__ == "__main__":
    raise SystemExit(main())
