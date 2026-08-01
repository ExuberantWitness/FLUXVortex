"""Run the preregistered S2c finite-angle sheet-formation oracle."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from claim_runtime.finite_angle_sheet_formation import (  # noqa: E402
    FiniteAngleSheetFormation,
    finite_angle_sheet_formation,
)


CASES = (
    HERE / "docs" / "diag"
    / "actual_boundary_finite_angle_sheet_cases.yaml"
)
RESULTS = (
    HERE / "docs" / "diag"
    / "actual_boundary_finite_angle_sheet_results.json"
)


def _optional(value: float | None) -> float | None:
    return None if value is None else float(value)


def _record(case: FiniteAngleSheetFormation) -> dict:
    return {
        "u1_plus": case.u1_plus,
        "u2_minus": case.u2_minus,
        "wedge_angle_rad": case.wedge_angle_rad,
        "delta_theta1": case.delta_theta1,
        "delta_theta2": case.delta_theta2,
        "sheet_strength": case.sheet_strength,
        "circulation_rate": case.circulation_rate,
        "relative_velocity": _optional(case.relative_velocity),
        "u_g_plus": _optional(case.u_g_plus),
        "u_g_minus": _optional(case.u_g_minus),
        "state_identifiable": case.state_identifiable,
        "normalized_residuals": {
            "angle_sum": case.normalized_angle_sum_residual,
            "direction": case.normalized_direction_residual,
            "kutta_strength": (
                case.normalized_kutta_strength_residual
            ),
            "circulation_rate": _optional(
                case.normalized_circulation_rate_residual
            ),
            "momentum": _optional(
                case.normalized_momentum_residual
            ),
        },
    }


def run() -> dict:
    contract = yaml.safe_load(CASES.read_text(encoding="utf-8"))
    canonical = contract["canonical_cases"]
    thresholds = contract["thresholds"]
    wedge = float(canonical["wedge_angle_deg"])
    named: dict[str, FiniteAngleSheetFormation] = {}
    roles: dict[str, str] = {}
    for specification in canonical["cases"]:
        identifier = specification["id"]
        roles[identifier] = specification["expected_role"]
        named[identifier] = finite_angle_sheet_formation(
            u1_plus=float(specification["u1_plus"]),
            u2_minus=float(specification["u2_minus"]),
            wedge_angle_deg=wedge,
        )

    nondegenerate = [
        case for case in named.values() if case.state_identifiable
    ]
    max_angle = max(
        case.normalized_angle_sum_residual
        for case in named.values()
    )
    max_direction = max(
        case.normalized_direction_residual
        for case in named.values()
    )
    max_kutta = max(
        case.normalized_kutta_strength_residual
        for case in named.values()
    )
    max_circulation = max(
        case.normalized_circulation_rate_residual
        for case in nondegenerate
    )
    max_momentum = max(
        case.normalized_momentum_residual for case in nondegenerate
    )
    min_sheet_side_velocity = min(
        min(case.u_g_plus, case.u_g_minus)
        for case in nondegenerate
    )

    first = named["side1_dominant"]
    mirror = named["side2_dominant_mirror"]
    mirror_residual = max(
        abs(first.delta_theta1 - mirror.delta_theta2),
        abs(first.delta_theta2 - mirror.delta_theta1),
        abs(first.sheet_strength + mirror.sheet_strength),
        abs(first.relative_velocity - mirror.relative_velocity),
        abs(first.u_g_plus - mirror.u_g_minus),
        abs(first.u_g_minus - mirror.u_g_plus),
    )
    scaled = named["side1_dominant_scaled"]
    scale_residual = max(
        abs(scaled.delta_theta1 - first.delta_theta1),
        abs(scaled.delta_theta2 - first.delta_theta2),
        abs(scaled.sheet_strength - 2.0 * first.sheet_strength)
        / max(abs(scaled.sheet_strength), np.finfo(float).tiny),
        abs(
            scaled.relative_velocity
            - 2.0 * first.relative_velocity
        ) / max(abs(scaled.relative_velocity), np.finfo(float).tiny),
        abs(scaled.circulation_rate - 4.0 * first.circulation_rate)
        / max(abs(scaled.circulation_rate), np.finfo(float).tiny),
    )
    symmetric = named["symmetric_bisector"]
    bisector_residual = max(
        abs(
            symmetric.delta_theta1
            - 0.5 * symmetric.wedge_angle_rad
        ),
        abs(
            symmetric.delta_theta2
            - 0.5 * symmetric.wedge_angle_rad
        ),
        abs(symmetric.sheet_strength),
        abs(symmetric.circulation_rate),
    )
    side2_zero = named["side2_stagnant_tangent_limit"]
    side1_zero = named["side1_stagnant_tangent_limit"]
    tangent_residual = max(
        abs(side2_zero.delta_theta1),
        abs(
            side2_zero.delta_theta2 - side2_zero.wedge_angle_rad
        ),
        abs(
            side1_zero.delta_theta1 - side1_zero.wedge_angle_rad
        ),
        abs(side1_zero.delta_theta2),
    )

    checks = {
        "angle_sum": max_angle
        <= float(thresholds["normalized_angle_sum_residual_max"]),
        "direction": max_direction
        <= float(thresholds["normalized_direction_residual_max"]),
        "kutta_strength": max_kutta
        <= float(
            thresholds["normalized_kutta_strength_residual_max"]
        ),
        "circulation_rate": max_circulation
        <= float(
            thresholds["normalized_circulation_rate_residual_max"]
        ),
        "momentum": max_momentum
        <= float(thresholds["normalized_momentum_residual_max"]),
        "sheet_side_velocities_nonnegative": min_sheet_side_velocity
        >= -float(
            thresholds["sheet_side_velocity_negative_tolerance"]
        ),
        "mirror": mirror_residual
        <= float(thresholds["mirror_residual_max"]),
        "scale_covariance": scale_residual
        <= float(thresholds["scale_covariance_residual_max"]),
        "symmetric_bisector": (
            not symmetric.state_identifiable
            and symmetric.relative_velocity is None
            and bisector_residual
            <= float(thresholds["symmetric_bisector_residual_max"])
        ),
        "tangent_limits": tangent_residual
        <= float(thresholds["tangent_limit_residual_max"]),
    }
    result = {
        "artifact": "finite_angle_sheet_formation_oracle",
        "claim_node": "N3.1j3b6d",
        "stage": "S2c_finite_angle_sharp_edge_sheet_formation",
        "canonical": canonical,
        "cases": {
            identifier: {
                "expected_role": roles[identifier],
                **_record(case),
            }
            for identifier, case in named.items()
        },
        "aggregate_metrics": {
            "normalized_angle_sum_residual_max": max_angle,
            "normalized_direction_residual_max": max_direction,
            "normalized_kutta_strength_residual_max": max_kutta,
            "normalized_circulation_rate_residual_max": (
                max_circulation
            ),
            "normalized_momentum_residual_max": max_momentum,
            "minimum_sheet_side_velocity": min_sheet_side_velocity,
            "mirror_residual_max": mirror_residual,
            "scale_covariance_residual_max": scale_residual,
            "symmetric_bisector_residual_max": bisector_residual,
            "tangent_limit_residual_max": tangent_residual,
        },
        "thresholds": thresholds,
        "checks": checks,
        "stage_decision": "GO" if all(checks.values()) else "NO-GO",
        "production_activation_allowed": False,
        "interpretation": (
            "Passing validates only the parameter-free high-Re "
            "finite-angle sharp-edge formation identities. It does not "
            "qualify a blunt base, viscous sheet inventory, 3-D junction, "
            "material wake evolution or production loads."
        ),
    }
    RESULTS.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, sort_keys=True))
