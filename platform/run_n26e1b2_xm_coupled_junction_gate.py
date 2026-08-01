#!/usr/bin/env python3
"""Run the preregistered N2.6e1b2 Xia--Mohseni junction spatial gate.

The gate is deliberately restricted to the fixed-wing, unit-scaled
canonical in the v2 preregistration.  It exercises the actual NACA0015
surface, the coupled no-through/Kelvin/finite-angle-Kutta system, and the
frozen panel--epsilon matrix.  It has no pressure, force, target-response,
or production-solver reader.
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

from claim_runtime.finite_angle_sheet_formation import (  # noqa: E402
    finite_angle_sheet_formation,
)
from claim_runtime.svi_dw_types import (  # noqa: E402
    NACA4SectionConfig,
    build_naca4_actual_surface,
)
from claim_runtime.svi_dw_xm_junction_2d import (  # noqa: E402
    build_xia_ccw_geometry,
    solve_xm_forming_step,
)


RUN_ID = "n26e1b2_xm_coupled_junction_gate_20260730"
PREREGISTRATION = (
    PLATFORM_DIR
    / "docs"
    / "diag"
    / "n26e1b2_xm_coupled_junction_prereg_v2_20260730.md"
)
DEFAULT_JSON = (
    PLATFORM_DIR
    / "docs"
    / "diag"
    / "n26e1b2_xm_coupled_junction_result_20260730.json"
)
DEFAULT_MARKDOWN = (
    PLATFORM_DIR
    / "docs"
    / "diag"
    / "n26e1b2_xm_coupled_junction_result_20260730.md"
)
RUNTIME_SOURCES = {
    "types": PLATFORM_DIR / "claim_runtime" / "svi_dw_types.py",
    "formation_oracle": (
        PLATFORM_DIR
        / "claim_runtime"
        / "finite_angle_sheet_formation.py"
    ),
    "xm_junction": (
        PLATFORM_DIR / "claim_runtime" / "svi_dw_xm_junction_2d.py"
    ),
}

CHORD = 1.0
FREESTREAM_SPEED = 1.0
TIME_STEP = 0.01
PANEL_LEVELS = (32, 64, 128)
EPSILON_RATIOS = (1.0 / 4.0, 1.0 / 8.0, 1.0 / 16.0)

SCORE_FLOOR_FRACTION = 0.02
FINAL_SCORE_TOLERANCE = 0.02
MONOTONIC_SCORE_SLACK = 1.0e-12
ALGEBRAIC_TOLERANCE = 1.0e-10
FORMATION_TOLERANCE = 1.0e-12
CONDITION_NUMBER_LIMIT = 1.0e12
# The preregistration states the mirror identities as exact numerical
# identities.  Use the already-frozen algebraic tolerance rather than
# introducing another adjustable threshold.
MIRROR_TOLERANCE = ALGEBRAIC_TOLERANCE

METRIC_SCALES = {
    "gamma1": FREESTREAM_SPEED,
    "gamma2": FREESTREAM_SPEED,
    "gamma_g": FREESTREAM_SPEED,
    "Gamma_g": FREESTREAM_SPEED * CHORD,
    "Gamma_bound": FREESTREAM_SPEED * CHORD,
    "u_g": FREESTREAM_SPEED,
    "delta1": 1.0,
    "delta2": 1.0,
    "absolute_forming_angle": 1.0,
}
ANGLE_METRICS = frozenset(
    {"delta1", "delta2", "absolute_forming_angle"}
)
CAUCHY_CASE_IDS = ("side1_dominant", "mirror_side2_dominant")


@dataclass(frozen=True)
class CanonicalState:
    identifier: str
    alpha_deg: float
    previous_u1_plus_over_U: float
    previous_u2_minus_over_U: float
    expected_no_birth: bool


CANONICAL_STATES = (
    CanonicalState(
        identifier="side1_dominant",
        alpha_deg=6.0,
        previous_u1_plus_over_U=-2.0,
        previous_u2_minus_over_U=-1.0,
        expected_no_birth=False,
    ),
    CanonicalState(
        identifier="mirror_side2_dominant",
        alpha_deg=-6.0,
        previous_u1_plus_over_U=-1.0,
        previous_u2_minus_over_U=-2.0,
        expected_no_birth=False,
    ),
    CanonicalState(
        identifier="symmetric_no_birth",
        alpha_deg=0.0,
        previous_u1_plus_over_U=-1.0,
        previous_u2_minus_over_U=-1.0,
        expected_no_birth=True,
    ),
)
CANONICAL_BY_ID = {case.identifier: case for case in CANONICAL_STATES}


def _epsilon_label(value: float) -> str:
    mapping = {
        EPSILON_RATIOS[0]: "1of4",
        EPSILON_RATIOS[1]: "1of8",
        EPSILON_RATIOS[2]: "1of16",
    }
    try:
        return mapping[float(value)]
    except KeyError as error:
        raise ValueError(
            f"epsilon ratio must be one of {EPSILON_RATIOS}"
        ) from error


@dataclass(frozen=True, order=True)
class JunctionGateCase:
    """One member of the immutable 3 x 3 x 3 canonical matrix."""

    panels_per_side: int
    epsilon_over_te_panel: float
    canonical_state_id: str

    def __post_init__(self) -> None:
        panels = int(self.panels_per_side)
        if panels != self.panels_per_side or panels not in PANEL_LEVELS:
            raise ValueError(
                f"panels_per_side must be one of {PANEL_LEVELS}"
            )
        ratio = float(self.epsilon_over_te_panel)
        if not math.isfinite(ratio) or ratio not in EPSILON_RATIOS:
            raise ValueError(
                f"epsilon_over_te_panel must be one of {EPSILON_RATIOS}"
            )
        if self.canonical_state_id not in CANONICAL_BY_ID:
            raise ValueError(
                "canonical_state_id must be one of "
                f"{tuple(CANONICAL_BY_ID)}"
            )
        object.__setattr__(self, "panels_per_side", panels)
        object.__setattr__(self, "epsilon_over_te_panel", ratio)

    @property
    def specification(self) -> CanonicalState:
        return CANONICAL_BY_ID[self.canonical_state_id]

    @property
    def case_id(self) -> str:
        return (
            f"{self.canonical_state_id}__p{self.panels_per_side}"
            f"__eps{_epsilon_label(self.epsilon_over_te_panel)}"
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "canonical_state_id": self.canonical_state_id,
            "panels_per_side": self.panels_per_side,
            "epsilon_over_te_panel": self.epsilon_over_te_panel,
            "alpha_deg": self.specification.alpha_deg,
            "previous_u1_plus_over_U": (
                self.specification.previous_u1_plus_over_U
            ),
            "previous_u2_minus_over_U": (
                self.specification.previous_u2_minus_over_U
            ),
            "expected_no_birth": self.specification.expected_no_birth,
        }


def frozen_cases() -> tuple[JunctionGateCase, ...]:
    """Return all 27 cases in the preregistered deterministic order."""
    return tuple(
        JunctionGateCase(
            panels_per_side=panels,
            epsilon_over_te_panel=ratio,
            canonical_state_id=canonical.identifier,
        )
        for panels in PANEL_LEVELS
        for ratio in EPSILON_RATIOS
        for canonical in CANONICAL_STATES
    )


def _freestream(alpha_deg: float) -> np.ndarray:
    angle = math.radians(float(alpha_deg))
    return FREESTREAM_SPEED * np.array(
        (math.cos(angle), math.sin(angle)), dtype=float
    )


def _formation_record(formation: Any) -> dict[str, Any]:
    residuals = {
        "angle_sum": float(formation.normalized_angle_sum_residual),
        "direction": float(formation.normalized_direction_residual),
        "kutta_strength": float(
            formation.normalized_kutta_strength_residual
        ),
        "circulation_rate": (
            None
            if formation.normalized_circulation_rate_residual is None
            else float(
                formation.normalized_circulation_rate_residual
            )
        ),
        "momentum": (
            None
            if formation.normalized_momentum_residual is None
            else float(formation.normalized_momentum_residual)
        ),
    }
    finite_residuals = [
        value for value in residuals.values() if value is not None
    ]
    return {
        "state_identifiable": bool(formation.state_identifiable),
        "delta1_rad": float(formation.delta_theta1),
        "delta2_rad": float(formation.delta_theta2),
        "sheet_strength_over_U": (
            float(formation.sheet_strength) / FREESTREAM_SPEED
        ),
        "circulation_rate_over_U2": (
            float(formation.circulation_rate) / FREESTREAM_SPEED**2
        ),
        "relative_velocity_over_U": (
            None
            if formation.relative_velocity is None
            else float(formation.relative_velocity) / FREESTREAM_SPEED
        ),
        "normalized_residuals": residuals,
        "maximum_applicable_normalized_residual": max(
            finite_residuals, default=0.0
        ),
    }


def _current_diagnostics(state: Any) -> dict[str, float | None]:
    residuals = state.residuals
    return {
        "maximum_normal_residual_over_U": float(
            residuals.maximum_relative_normal_residual
        ),
        "kelvin_residual_over_Uc": float(
            residuals.normalized_kelvin_residual
        ),
        "kutta_residual_over_U": (
            None
            if residuals.normalized_kutta_residual is None
            else float(residuals.normalized_kutta_residual)
        ),
        "linear_system_infinity_residual_scaled": float(
            residuals.maximum_normalized_linear_system_residual
        ),
        "scaled_system_condition_number_2": float(
            residuals.system_condition_number
        ),
    }


def _dimensionless_metrics(state: Any) -> dict[str, float | None]:
    if state.no_birth:
        return {
            "gamma1": float(state.gamma1_upper_physical)
            / FREESTREAM_SPEED,
            "gamma2": float(state.gamma2_lower_physical)
            / FREESTREAM_SPEED,
            "gamma_g": 0.0,
            "Gamma_g": 0.0,
            "Gamma_bound": float(
                state.circulation.bound_circulation_ccw
            )
            / (FREESTREAM_SPEED * CHORD),
            "u_g": None,
            "delta1": None,
            "delta2": None,
            "absolute_forming_angle": None,
        }
    if (
        state.formation is None
        or state.formation.relative_velocity is None
        or state.forming_direction_body is None
    ):
        raise RuntimeError(
            "a non-no-birth state is missing its formation state"
        )
    direction = np.asarray(state.forming_direction_body, dtype=float)
    return {
        "gamma1": float(state.gamma1_upper_physical)
        / FREESTREAM_SPEED,
        "gamma2": float(state.gamma2_lower_physical)
        / FREESTREAM_SPEED,
        "gamma_g": float(state.forming_sheet_strength_ccw)
        / FREESTREAM_SPEED,
        "Gamma_g": float(
            state.circulation.forming_circulation_ccw
        )
        / (FREESTREAM_SPEED * CHORD),
        "Gamma_bound": float(
            state.circulation.bound_circulation_ccw
        )
        / (FREESTREAM_SPEED * CHORD),
        "u_g": float(state.formation.relative_velocity)
        / FREESTREAM_SPEED,
        "delta1": float(state.formation.delta_theta1),
        "delta2": float(state.formation.delta_theta2),
        "absolute_forming_angle": float(
            math.atan2(direction[1], direction[0])
        ),
    }


def run_gate_case(case: JunctionGateCase) -> dict[str, Any]:
    """Execute one frozen case without reading any downstream observable."""
    if not isinstance(case, JunctionGateCase):
        raise TypeError("case must be JunctionGateCase")
    surface = build_naca4_actual_surface(
        NACA4SectionConfig(
            maximum_camber=0.0,
            camber_location=0.4,
            thickness_ratio=0.15,
            chord=CHORD,
        ),
        panels_per_side=case.panels_per_side,
    )
    geometry = build_xia_ccw_geometry(
        surface,
        epsilon_over_te_panel=case.epsilon_over_te_panel,
    )
    specification = case.specification
    previous_first = (
        specification.previous_u1_plus_over_U * FREESTREAM_SPEED
    )
    previous_second = (
        specification.previous_u2_minus_over_U * FREESTREAM_SPEED
    )
    oracle = finite_angle_sheet_formation(
        u1_plus=previous_first,
        u2_minus=previous_second,
        wedge_angle_deg=geometry.trailing_edge_wedge_angle_deg,
    )
    state = solve_xm_forming_step(
        geometry,
        freestream_velocity_body=_freestream(
            specification.alpha_deg
        ),
        previous_u1_plus=previous_first,
        previous_u2_minus=previous_second,
        time_step=TIME_STEP,
    )

    strengths = (
        np.asarray(state.bound_node_strength_ccw, dtype=float)
        / FREESTREAM_SPEED
    )
    if strengths.shape != (2 * case.panels_per_side + 1,):
        raise RuntimeError("bound nodal-strength contract drifted")
    if not np.all(np.isfinite(strengths)):
        raise FloatingPointError(
            "bound nodal strengths contain non-finite values"
        )
    metrics = _dimensionless_metrics(state)
    if set(metrics) != set(METRIC_SCALES):
        raise RuntimeError("dimensionless metric contract drifted")
    finite_metrics = [
        value for value in metrics.values() if value is not None
    ]
    if not all(math.isfinite(float(value)) for value in finite_metrics):
        raise FloatingPointError(
            "dimensionless metrics contain non-finite values"
        )

    return {
        **case.as_dict(),
        "status": "completed",
        "stage": state.stage,
        "no_birth": bool(state.no_birth),
        "geometry": {
            "panel_count": int(geometry.panel_count),
            "trailing_edge_wedge_angle_deg": (
                geometry.trailing_edge_wedge_angle_deg
            ),
            "h_TE_over_c": (
                geometry.trailing_edge_reference_panel_length / CHORD
            ),
            "epsilon_over_c": geometry.epsilon / CHORD,
            "forming_length_over_c": state.forming_length / CHORD,
            "forming_direction_body": (
                None
                if state.forming_direction_body is None
                else np.asarray(
                    state.forming_direction_body, dtype=float
                ).tolist()
            ),
            "forming_start_over_c": (
                None
                if state.forming_start_body is None
                else (
                    np.asarray(state.forming_start_body, dtype=float)
                    / CHORD
                ).tolist()
            ),
            "forming_end_over_c": (
                None
                if state.forming_end_body is None
                else (
                    np.asarray(state.forming_end_body, dtype=float)
                    / CHORD
                ).tolist()
            ),
        },
        "bound_node_strength_over_U": strengths.tolist(),
        "dimensionless_metrics": metrics,
        "current_diagnostics": _current_diagnostics(state),
        "previous_formation_oracle": _formation_record(oracle),
    }


def wrap_to_pi(angle_rad: float) -> float:
    angle = float(angle_rad)
    if not math.isfinite(angle):
        raise ValueError("angle_rad must be finite")
    wrapped = (angle + math.pi) % (2.0 * math.pi) - math.pi
    # Make the branch deterministic at the otherwise equivalent endpoint.
    if wrapped == -math.pi and angle > 0.0:
        return math.pi
    return wrapped


def convergence_score(
    coarse_value: float,
    fine_value: float,
    physical_scale: float,
    *,
    angular: bool = False,
) -> dict[str, float | str | bool]:
    """Return the frozen signed-value Cauchy score.

    Values themselves retain their sign.  Only the numerator is made
    absolute; angular differences are wrapped before that operation.
    """
    coarse = float(coarse_value)
    fine = float(fine_value)
    scale = float(physical_scale)
    if (
        not math.isfinite(coarse)
        or not math.isfinite(fine)
        or not math.isfinite(scale)
        or scale <= 0.0
    ):
        raise ValueError(
            "score values must be finite and physical_scale positive"
        )
    signed_change = fine - coarse
    if angular:
        signed_change = wrap_to_pi(signed_change)
    floor = SCORE_FLOOR_FRACTION * scale
    denominator = max(abs(fine), floor)
    score = abs(signed_change) / denominator
    if not math.isfinite(score):
        raise FloatingPointError("convergence score is non-finite")
    return {
        "coarse": coarse,
        "fine": fine,
        "angular_difference_wrapped": bool(angular),
        "signed_change": signed_change,
        "absolute_change": abs(signed_change),
        "physical_scale": scale,
        "denominator_floor": floor,
        "denominator": denominator,
        "mode": (
            "relative_to_fine"
            if abs(fine) >= floor
            else "physical_scale_floor"
        ),
        "score": score,
        "score_finite": True,
    }


def _required_case(
    results: Mapping[str, Mapping[str, Any]],
    *,
    panels: int,
    epsilon: float,
    canonical: str,
) -> Mapping[str, Any]:
    case = JunctionGateCase(
        panels_per_side=panels,
        epsilon_over_te_panel=epsilon,
        canonical_state_id=canonical,
    )
    result = results[case.case_id]
    if result.get("status") != "completed":
        raise RuntimeError(f"{case.case_id} did not complete")
    return result


def _metric(
    result: Mapping[str, Any],
    metric_name: str,
) -> float:
    value = result["dimensionless_metrics"][metric_name]
    if value is None:
        raise ValueError(f"{metric_name} is undefined")
    scalar = float(value)
    if not math.isfinite(scalar):
        raise FloatingPointError(f"{metric_name} is non-finite")
    return scalar


def _score_three(
    first: Mapping[str, Any],
    second: Mapping[str, Any],
    third: Mapping[str, Any],
    *,
    metric_name: str,
) -> dict[str, Any]:
    scale = METRIC_SCALES[metric_name]
    angular = metric_name in ANGLE_METRICS
    coarse_to_middle = convergence_score(
        _metric(first, metric_name),
        _metric(second, metric_name),
        scale,
        angular=angular,
    )
    middle_to_fine = convergence_score(
        _metric(second, metric_name),
        _metric(third, metric_name),
        scale,
        angular=angular,
    )
    final_pass = (
        float(middle_to_fine["score"])
        <= FINAL_SCORE_TOLERANCE
    )
    monotonic_pass = (
        float(middle_to_fine["score"])
        <= float(coarse_to_middle["score"])
        + MONOTONIC_SCORE_SLACK
    )
    return {
        "coarse_to_middle": coarse_to_middle,
        "middle_to_fine": middle_to_fine,
        "final_score_threshold": FINAL_SCORE_TOLERANCE,
        "monotonic_slack": MONOTONIC_SCORE_SLACK,
        "scores_finite": True,
        "final_score_within_tolerance": final_pass,
        "score_nonincreasing": monotonic_pass,
        "pass": final_pass and monotonic_pass,
    }


def evaluate_cauchy(
    case_results: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Apply both frozen Cauchy axes to the two identifiable states."""
    expected_ids = {case.case_id for case in frozen_cases()}
    exact_matrix = set(case_results) == expected_ids
    all_completed = exact_matrix and all(
        case_results[case_id].get("status") == "completed"
        for case_id in expected_ids
    )
    output: dict[str, Any] = {
        "exact_27_case_matrix": exact_matrix,
        "all_cases_completed": all_completed,
        "epsilon_axis": {},
        "panel_axis_at_epsilon_1of16": {},
        "all_scores_finite": False,
        "all_final_scores_within_tolerance": False,
        "all_scores_nonincreasing": False,
        "pass": False,
    }
    if not all_completed:
        output["missing_case_ids"] = sorted(
            expected_ids - set(case_results)
        )
        output["unexpected_case_ids"] = sorted(
            set(case_results) - expected_ids
        )
        output["failure"] = (
            "all 27 frozen cases must complete before Cauchy scoring"
        )
        return output

    entries: list[dict[str, Any]] = []
    for canonical in CAUCHY_CASE_IDS:
        epsilon_case: dict[str, Any] = {}
        for panels in PANEL_LEVELS:
            panel_entry: dict[str, Any] = {}
            trio = [
                _required_case(
                    case_results,
                    panels=panels,
                    epsilon=ratio,
                    canonical=canonical,
                )
                for ratio in EPSILON_RATIOS
            ]
            for metric_name in METRIC_SCALES:
                try:
                    score = _score_three(
                        trio[0],
                        trio[1],
                        trio[2],
                        metric_name=metric_name,
                    )
                except Exception as error:
                    score = {
                        "scores_finite": False,
                        "final_score_within_tolerance": False,
                        "score_nonincreasing": False,
                        "pass": False,
                        "error_type": type(error).__name__,
                        "error": str(error),
                    }
                panel_entry[metric_name] = score
                entries.append(score)
            epsilon_case[f"p{panels}"] = panel_entry
        output["epsilon_axis"][canonical] = epsilon_case

        panel_entry = {}
        trio = [
            _required_case(
                case_results,
                panels=panels,
                epsilon=EPSILON_RATIOS[-1],
                canonical=canonical,
            )
            for panels in PANEL_LEVELS
        ]
        for metric_name in METRIC_SCALES:
            try:
                score = _score_three(
                    trio[0],
                    trio[1],
                    trio[2],
                    metric_name=metric_name,
                )
            except Exception as error:
                score = {
                    "scores_finite": False,
                    "final_score_within_tolerance": False,
                    "score_nonincreasing": False,
                    "pass": False,
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            panel_entry[metric_name] = score
            entries.append(score)
        output["panel_axis_at_epsilon_1of16"][canonical] = (
            panel_entry
        )

    output["all_scores_finite"] = all(
        entry.get("scores_finite") is True for entry in entries
    )
    output["all_final_scores_within_tolerance"] = all(
        entry.get("final_score_within_tolerance") is True
        for entry in entries
    )
    output["all_scores_nonincreasing"] = all(
        entry.get("score_nonincreasing") is True
        for entry in entries
    )
    output["pass"] = all(
        (
            output["all_scores_finite"],
            output["all_final_scores_within_tolerance"],
            output["all_scores_nonincreasing"],
        )
    )
    return output


def _completed_case_algebra_pass(
    result: Mapping[str, Any],
) -> bool:
    if result.get("status") != "completed":
        return False
    try:
        diagnostic = result["current_diagnostics"]
        normal = float(diagnostic["maximum_normal_residual_over_U"])
        kelvin = float(diagnostic["kelvin_residual_over_Uc"])
        kutta_raw = diagnostic["kutta_residual_over_U"]
        linear = float(
            diagnostic["linear_system_infinity_residual_scaled"]
        )
        condition = float(
            diagnostic["scaled_system_condition_number_2"]
        )
        no_birth = bool(result["no_birth"])
        kutta_pass = (
            kutta_raw is None
            if no_birth
            else (
                kutta_raw is not None
                and math.isfinite(float(kutta_raw))
                and float(kutta_raw) <= ALGEBRAIC_TOLERANCE
            )
        )
        current_pass = all(
            math.isfinite(value)
            and value <= ALGEBRAIC_TOLERANCE
            for value in (normal, kelvin, linear)
        ) and kutta_pass
        condition_pass = (
            math.isfinite(condition)
            and condition <= CONDITION_NUMBER_LIMIT
        )

        oracle = result["previous_formation_oracle"]
        residuals = oracle["normalized_residuals"]
        applicable = [
            float(value)
            for value in residuals.values()
            if value is not None
        ]
        oracle_pass = bool(applicable) and all(
            math.isfinite(value)
            and value <= FORMATION_TOLERANCE
            for value in applicable
        )
        expected_no_birth = bool(result["expected_no_birth"])
        oracle_identity_pass = (
            bool(oracle["state_identifiable"])
            is (not expected_no_birth)
        )
        if expected_no_birth:
            oracle_identity_pass = oracle_identity_pass and (
                float(oracle["sheet_strength_over_U"]) == 0.0
                and float(oracle["circulation_rate_over_U2"]) == 0.0
                and oracle["relative_velocity_over_U"] is None
            )
        return (
            current_pass
            and condition_pass
            and oracle_pass
            and oracle_identity_pass
        )
    except (KeyError, TypeError, ValueError):
        return False


def evaluate_no_birth_contract(
    case_results: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Require explicit no-birth at every symmetric matrix point."""
    entries: dict[str, Any] = {}
    for panels in PANEL_LEVELS:
        for ratio in EPSILON_RATIOS:
            case = JunctionGateCase(
                panels_per_side=panels,
                epsilon_over_te_panel=ratio,
                canonical_state_id="symmetric_no_birth",
            )
            result = case_results.get(case.case_id)
            passed = False
            reason = "case missing or failed"
            if result is not None and result.get("status") == "completed":
                try:
                    geometry = result["geometry"]
                    metrics = result["dimensionless_metrics"]
                    strengths = np.asarray(
                        result["bound_node_strength_over_U"],
                        dtype=float,
                    )
                    symmetry_residual = float(
                        np.max(
                            np.abs(strengths + strengths[::-1]),
                            initial=0.0,
                        )
                    )
                    passed = all(
                        (
                            result["stage"] == "no_birth",
                            result["no_birth"] is True,
                            geometry["forming_direction_body"] is None,
                            geometry["forming_start_over_c"] is None,
                            geometry["forming_end_over_c"] is None,
                            float(
                                geometry["forming_length_over_c"]
                            )
                            == 0.0,
                            float(metrics["gamma_g"]) == 0.0,
                            float(metrics["Gamma_g"]) == 0.0,
                            metrics["u_g"] is None,
                            metrics["delta1"] is None,
                            metrics["delta2"] is None,
                            metrics["absolute_forming_angle"] is None,
                            symmetry_residual <= MIRROR_TOLERANCE,
                        )
                    )
                    reason = (
                        "explicit no-birth contract satisfied"
                        if passed
                        else "a no-birth invariant failed"
                    )
                except (KeyError, TypeError, ValueError):
                    passed = False
                    reason = "malformed no-birth record"
                    symmetry_residual = math.inf
                entries[case.case_id] = {
                    "pass": passed,
                    "reason": reason,
                    "self_mirror_bound_strength_residual": (
                        symmetry_residual
                    ),
                }
                continue
            entries[case.case_id] = {
                "pass": False,
                "reason": reason,
            }
    return {
        "entries": entries,
        "all_nine_explicit_no_birth": (
            len(entries) == 9
            and all(entry["pass"] for entry in entries.values())
        ),
        "pass": (
            len(entries) == 9
            and all(entry["pass"] for entry in entries.values())
        ),
    }


def evaluate_mirror_contract(
    case_results: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Recompute every preregistered positive/negative mirror identity."""
    entries: dict[str, Any] = {}
    for panels in PANEL_LEVELS:
        for ratio in EPSILON_RATIOS:
            positive_case = JunctionGateCase(
                panels_per_side=panels,
                epsilon_over_te_panel=ratio,
                canonical_state_id="side1_dominant",
            )
            negative_case = JunctionGateCase(
                panels_per_side=panels,
                epsilon_over_te_panel=ratio,
                canonical_state_id="mirror_side2_dominant",
            )
            pair_id = (
                f"p{panels}__eps"
                f"{_epsilon_label(ratio)}"
            )
            try:
                positive = case_results[positive_case.case_id]
                negative = case_results[negative_case.case_id]
                if (
                    positive.get("status") != "completed"
                    or negative.get("status") != "completed"
                ):
                    raise RuntimeError("mirror pair did not complete")
                plus_bound = np.asarray(
                    positive["bound_node_strength_over_U"],
                    dtype=float,
                )
                minus_bound = np.asarray(
                    negative["bound_node_strength_over_U"],
                    dtype=float,
                )
                if (
                    plus_bound.shape != minus_bound.shape
                    or plus_bound.shape
                    != (2 * panels + 1,)
                    or not np.all(np.isfinite(plus_bound))
                    or not np.all(np.isfinite(minus_bound))
                ):
                    raise ValueError(
                        "mirror bound vectors are shape-invalid"
                    )
                plus = positive["dimensionless_metrics"]
                minus = negative["dimensionless_metrics"]
                plus_direction = np.asarray(
                    positive["geometry"]["forming_direction_body"],
                    dtype=float,
                )
                minus_direction = np.asarray(
                    negative["geometry"]["forming_direction_body"],
                    dtype=float,
                )
                residuals = {
                    "bound_nodes": float(
                        np.max(
                            np.abs(
                                minus_bound + plus_bound[::-1]
                            ),
                            initial=0.0,
                        )
                    ),
                    "gamma1_swap": abs(
                        float(minus["gamma1"])
                        + float(plus["gamma2"])
                    ),
                    "gamma2_swap": abs(
                        float(minus["gamma2"])
                        + float(plus["gamma1"])
                    ),
                    "gamma_g_sign": abs(
                        float(minus["gamma_g"])
                        + float(plus["gamma_g"])
                    ),
                    "Gamma_g_sign": abs(
                        float(minus["Gamma_g"])
                        + float(plus["Gamma_g"])
                    ),
                    "Gamma_bound_sign": abs(
                        float(minus["Gamma_bound"])
                        + float(plus["Gamma_bound"])
                    ),
                    "u_g_equal": abs(
                        float(minus["u_g"])
                        - float(plus["u_g"])
                    ),
                    "delta1_delta2_swap": abs(
                        wrap_to_pi(
                            float(minus["delta1"])
                            - float(plus["delta2"])
                        )
                    ),
                    "delta2_delta1_swap": abs(
                        wrap_to_pi(
                            float(minus["delta2"])
                            - float(plus["delta1"])
                        )
                    ),
                    "direction_x_equal": abs(
                        float(minus_direction[0])
                        - float(plus_direction[0])
                    ),
                    "direction_y_sign": abs(
                        float(minus_direction[1])
                        + float(plus_direction[1])
                    ),
                    "absolute_angle_sign": abs(
                        wrap_to_pi(
                            float(
                                minus[
                                    "absolute_forming_angle"
                                ]
                            )
                            + float(
                                plus[
                                    "absolute_forming_angle"
                                ]
                            )
                        )
                    ),
                }
                maximum = max(residuals.values())
                passed = (
                    all(math.isfinite(value) for value in residuals.values())
                    and maximum <= MIRROR_TOLERANCE
                )
                entries[pair_id] = {
                    "normalized_residuals": residuals,
                    "maximum_normalized_residual": maximum,
                    "threshold": MIRROR_TOLERANCE,
                    "pass": passed,
                }
            except Exception as error:
                entries[pair_id] = {
                    "pass": False,
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
    return {
        "entries": entries,
        "all_nine_pairs_pass": (
            len(entries) == 9
            and all(entry["pass"] for entry in entries.values())
        ),
        "pass": (
            len(entries) == 9
            and all(entry["pass"] for entry in entries.values())
        ),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def execute_gate(
    *,
    cases: Iterable[JunctionGateCase] | None = None,
) -> dict[str, Any]:
    """Execute and judge the exact preregistered matrix without elision."""
    selected = tuple(cases) if cases is not None else frozen_cases()
    if not all(
        isinstance(case, JunctionGateCase) for case in selected
    ):
        raise TypeError(
            "cases must contain only JunctionGateCase values"
        )
    if len({case.case_id for case in selected}) != len(selected):
        raise ValueError("duplicate gate cases are not allowed")

    results: dict[str, dict[str, Any]] = {}
    for case in selected:
        print(f"[{RUN_ID}] running {case.case_id}", flush=True)
        try:
            results[case.case_id] = run_gate_case(case)
        except Exception as error:
            results[case.case_id] = {
                **case.as_dict(),
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            }
            print(
                f"[{RUN_ID}] {case.case_id} failed: "
                f"{type(error).__name__}: {error}",
                flush=True,
            )

    expected_ids = {case.case_id for case in frozen_cases()}
    exact_matrix = set(results) == expected_ids
    all_completed = exact_matrix and all(
        results[case_id].get("status") == "completed"
        for case_id in expected_ids
    )
    algebra_pass = all_completed and all(
        _completed_case_algebra_pass(results[case_id])
        for case_id in expected_ids
    )
    no_birth = evaluate_no_birth_contract(results)
    mirror = evaluate_mirror_contract(results)
    cauchy = evaluate_cauchy(results)
    overall_pass = all(
        (
            exact_matrix,
            all_completed,
            algebra_pass,
            no_birth["pass"],
            mirror["pass"],
            cauchy["pass"],
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
            "chord": CHORD,
            "freestream_speed": FREESTREAM_SPEED,
            "time_step_U_over_c": (
                TIME_STEP * FREESTREAM_SPEED / CHORD
            ),
            "reference_frame": "fixed_body_uniform_freestream",
            "historical_wake_circulation": 0.0,
        },
        "frozen_matrix": {
            "panels_per_side": list(PANEL_LEVELS),
            "epsilon_over_te_panel": list(EPSILON_RATIOS),
            "canonical_states": [
                {
                    "id": state.identifier,
                    "alpha_deg": state.alpha_deg,
                    "previous_u1_plus_over_U": (
                        state.previous_u1_plus_over_U
                    ),
                    "previous_u2_minus_over_U": (
                        state.previous_u2_minus_over_U
                    ),
                    "expected_no_birth": state.expected_no_birth,
                }
                for state in CANONICAL_STATES
            ],
            "case_count": 27,
        },
        "thresholds": {
            "score_denominator_floor_fraction": (
                SCORE_FLOOR_FRACTION
            ),
            "final_cauchy_score": FINAL_SCORE_TOLERANCE,
            "score_nonincrease_slack": MONOTONIC_SCORE_SLACK,
            "normalized_algebraic_residual": (
                ALGEBRAIC_TOLERANCE
            ),
            "normalized_formation_residual": (
                FORMATION_TOLERANCE
            ),
            "scaled_condition_number_2": (
                CONDITION_NUMBER_LIMIT
            ),
            "normalized_mirror_identity": MIRROR_TOLERANCE,
        },
        "metric_physical_scales": METRIC_SCALES,
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
        "cases": results,
        "no_birth_contract": no_birth,
        "mirror_contract": mirror,
        "cauchy": cauchy,
        "gate": {
            "exact_27_case_matrix": exact_matrix,
            "all_cases_completed": all_completed,
            "all_algebraic_and_condition_gates_pass": algebra_pass,
            "all_symmetric_cases_explicit_no_birth": no_birth["pass"],
            "all_mirror_pairs_pass": mirror["pass"],
            "all_cauchy_scores_finite": (
                cauchy["all_scores_finite"]
            ),
            "all_final_cauchy_scores_within_two_percent": (
                cauchy[
                    "all_final_scores_within_tolerance"
                ]
            ),
            "all_cauchy_scores_nonincreasing": (
                cauchy["all_scores_nonincreasing"]
            ),
            "overall_pass": overall_pass,
            "verdict": "GO" if overall_pass else "NO-GO",
            "failure_interpretation": (
                None
                if overall_pass
                else (
                    "not converged within preregistered budget; "
                    "this does not falsify the continuous "
                    "Xia--Mohseni mechanism"
                )
            ),
        },
    }


def render_markdown(result: Mapping[str, Any]) -> str:
    gate = result["gate"]
    lines = [
        "# N2.6e1b2 Xia--Mohseni coupled junction gate result",
        "",
        f"- Run: `{result['run_id']}`",
        f"- Verdict: **{gate['verdict']}**",
        f"- Generated (UTC): `{result['generated_at_utc']}`",
        (
            "- Scope: fixed-wing canonical spatial operator only; "
            "no pressure, force, target response, or production coupling."
        ),
        "",
        "## Gate summary",
        "",
        "| gate | pass |",
        "|---|---:|",
        (
            "| exact 27-case matrix | "
            f"{gate['exact_27_case_matrix']} |"
        ),
        f"| all cases completed | {gate['all_cases_completed']} |",
        (
            "| algebra, formation and condition gates | "
            f"{gate['all_algebraic_and_condition_gates_pass']} |"
        ),
        (
            "| all nine symmetric cases explicit no-birth | "
            f"{gate['all_symmetric_cases_explicit_no_birth']} |"
        ),
        (
            "| all nine mirror pairs | "
            f"{gate['all_mirror_pairs_pass']} |"
        ),
        (
            "| every Cauchy score finite | "
            f"{gate['all_cauchy_scores_finite']} |"
        ),
        (
            "| every final Cauchy score <= 2% | "
            f"{gate['all_final_cauchy_scores_within_two_percent']} |"
        ),
        (
            "| every final Cauchy score nonincreasing | "
            f"{gate['all_cauchy_scores_nonincreasing']} |"
        ),
        "",
        "## Cases",
        "",
        (
            "| case | status | stage | cond2 | normal/U | "
            "Kelvin/(Uc) | Kutta/U |"
        ),
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for case in frozen_cases():
        record = result["cases"].get(case.case_id)
        if record is None:
            lines.append(
                f"| `{case.case_id}` | missing | — | — | — | — | — |"
            )
            continue
        if record.get("status") != "completed":
            error = (
                f"{record.get('error_type', 'Error')}: "
                f"{record.get('error', 'unknown failure')}"
            ).replace("|", "\\|")
            lines.append(
                f"| `{case.case_id}` | failed: {error} | — | — | "
                "— | — | — |"
            )
            continue
        diagnostic = record["current_diagnostics"]
        kutta = diagnostic["kutta_residual_over_U"]
        kutta_text = "—" if kutta is None else f"{float(kutta):.3e}"
        lines.append(
            f"| `{case.case_id}` | completed | {record['stage']} | "
            f"{float(diagnostic['scaled_system_condition_number_2']):.6g} | "
            f"{float(diagnostic['maximum_normal_residual_over_U']):.3e} | "
            f"{float(diagnostic['kelvin_residual_over_Uc']):.3e} | "
            f"{kutta_text} |"
        )

    score_entries: list[tuple[str, Mapping[str, Any]]] = []
    for canonical, levels in result["cauchy"]["epsilon_axis"].items():
        for level, metrics in levels.items():
            for metric, entry in metrics.items():
                score_entries.append(
                    (f"epsilon.{canonical}.{level}.{metric}", entry)
                )
    for canonical, metrics in result["cauchy"][
        "panel_axis_at_epsilon_1of16"
    ].items():
        for metric, entry in metrics.items():
            score_entries.append(
                (f"panel.{canonical}.{metric}", entry)
            )
    finite_scores = [
        (
            name,
            float(entry["middle_to_fine"]["score"]),
            bool(entry["pass"]),
        )
        for name, entry in score_entries
        if entry.get("scores_finite") is True
    ]
    finite_scores.sort(key=lambda item: item[1], reverse=True)
    lines.extend(
        [
            "",
            "## Largest final Cauchy scores",
            "",
            "| axis.case.metric | final score | pass |",
            "|---|---:|---:|",
        ]
    )
    for name, score, passed in finite_scores[:20]:
        lines.append(f"| `{name}` | {score:.9g} | {passed} |")
    if not finite_scores:
        lines.append("| unavailable | — | False |")

    lines.extend(["", "## Decision", ""])
    if gate["overall_pass"]:
        lines.append(
            "GO only for the fixed canonical spatial discretization. "
            "A separate preregistration is still required for moving-wall "
            "and forming-to-material-wake time convergence."
        )
    else:
        lines.append(
            "NO-GO for this fixed-budget N2.6e1b2 implementation: not "
            "converged within the preregistered budget. This result does "
            "not falsify the continuous Xia--Mohseni mechanism."
        )
    lines.extend(
        [
            "",
            "All signed values, wrapped angular differences, both Cauchy "
            "intervals, exact matrix records, and source hashes are in "
            "the companion JSON.",
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
