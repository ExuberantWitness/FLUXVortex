"""Run the preregistered S2b Joukowski Kutta-pressure oracle."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from claim_runtime.joukowski_kutta_oracle import (  # noqa: E402
    JoukowskiP2Evaluation,
    joukowski_p2_kutta_trace,
)


CASES = (
    HERE / "docs" / "diag"
    / "actual_boundary_joukowski_kutta_cases.yaml"
)
RESULTS = (
    HERE / "docs" / "diag"
    / "actual_boundary_joukowski_kutta_results.json"
)


def _relative_error(actual: float, expected: float) -> float:
    return float(
        abs(actual - expected)
        / max(abs(expected), np.finfo(float).tiny)
    )


def _nonincreasing(values: list[float]) -> bool:
    return all(
        following <= previous
        for previous, following in zip(values, values[1:])
    )


def _record(case: JoukowskiP2Evaluation) -> dict:
    return {
        "panel_count": case.panel_count,
        "quadrature_order": case.quadrature_order,
        "kutta_circulation": case.kutta_circulation,
        "potential_jump": case.potential_jump,
        "circulation": case.circulation,
        "kutta_numerator_residual": case.kutta_numerator_residual,
        "surface_velocity_rms_error": (
            case.surface_velocity_rms_error
        ),
        "surface_Cp_rms_error": case.surface_cp_rms_error,
        "pressure_force_xy": case.pressure_force.tolist(),
        "drag": case.drag,
        "lift": case.lift,
        "expected_lift": case.expected_lift,
        "lift_relative_error": case.lift_relative_error,
        "trailing_side_Cp_mismatch": (
            case.trailing_side_cp_mismatch
        ),
    }


def run() -> dict:
    contract = yaml.safe_load(CASES.read_text(encoding="utf-8"))
    canonical = contract["canonical"]
    thresholds = contract["thresholds"]
    common = {
        "mapping_parameter_b": float(
            canonical["mapping_parameter_b"]
        ),
        "circle_center": tuple(canonical["circle_center"]),
        "freestream_speed": float(canonical["freestream_speed"]),
        "density": float(canonical["density"]),
        "angle_of_attack_deg": float(
            canonical["angle_of_attack_deg"]
        ),
        "quadrature_order": int(canonical["quadrature_order"]),
    }
    counts = [int(value) for value in canonical["p2_panel_counts"]]
    cases = [
        joukowski_p2_kutta_trace(panel_count=count, **common)
        for count in counts
    ]
    finest = cases[-1]
    shifted = joukowski_p2_kutta_trace(
        panel_count=counts[-1],
        potential_gauge=float(canonical["gauge_shift"]),
        **common,
    )
    gamma = finest.kutta_circulation
    jump_error = max(
        _relative_error(case.potential_jump, gamma)
        for case in cases
    )
    circulation_error = max(
        _relative_error(case.circulation, gamma)
        for case in cases
    )
    gauge_velocity_change = float(np.max(
        np.abs(
            shifted.tangential_velocity
            - finest.tangential_velocity
        ),
        initial=0.0,
    ))
    gauge_force_change = float(np.max(
        np.abs(shifted.pressure_force - finest.pressure_force),
        initial=0.0,
    ))
    velocity_errors = [
        case.surface_velocity_rms_error for case in cases
    ]
    cp_errors = [case.surface_cp_rms_error for case in cases]
    lift_errors = [case.lift_relative_error for case in cases]

    checks = {
        "cut_jump": jump_error
        <= float(thresholds["cut_jump_relative_error_max"]),
        "circulation": circulation_error
        <= float(thresholds["circulation_relative_error_max"]),
        "analytic_kutta": finest.kutta_numerator_residual
        <= float(
            thresholds["analytic_kutta_numerator_absolute_max"]
        ),
        "finest_velocity": finest.surface_velocity_rms_error
        <= float(
            thresholds["finest_surface_velocity_rms_error_max"]
        ),
        "finest_Cp": finest.surface_cp_rms_error
        <= float(thresholds["finest_surface_Cp_rms_error_max"]),
        "finest_lift": finest.lift_relative_error
        <= float(thresholds["finest_lift_relative_error_max"]),
        "finest_drag": abs(finest.drag)
        <= float(thresholds["finest_drag_absolute_max"]),
        "trailing_side_Cp": finest.trailing_side_cp_mismatch
        <= float(
            thresholds["finest_trailing_side_Cp_mismatch_max"]
        ),
        "gauge_velocity": gauge_velocity_change
        <= float(
            thresholds["gauge_velocity_absolute_change_max"]
        ),
        "gauge_force": gauge_force_change
        <= float(thresholds["gauge_force_absolute_change_max"]),
        "velocity_errors_monotone": _nonincreasing(velocity_errors),
        "Cp_errors_monotone": _nonincreasing(cp_errors),
        "lift_errors_monotone": _nonincreasing(lift_errors),
    }
    go = all(checks.values())
    result = {
        "artifact": "actual_boundary_joukowski_kutta_oracle",
        "claim_node": "N3.1j3b6d",
        "stage": "S2b_sharp_trailing_edge_kutta_pressure",
        "canonical": canonical,
        "cases": [_record(case) for case in cases],
        "aggregate_metrics": {
            "cut_jump_relative_error_max": jump_error,
            "circulation_relative_error_max": circulation_error,
            "gauge_velocity_absolute_change_max": (
                gauge_velocity_change
            ),
            "gauge_force_absolute_change_max": gauge_force_change,
        },
        "thresholds": thresholds,
        "checks": checks,
        "stage_decision": "GO" if go else "NO-GO",
        "production_activation_allowed": False,
        "interpretation": (
            "Passing qualifies only the steady sharp-cusp Kutta selection, "
            "curved-P2 trace differentiation and one pressure integral. It "
            "does not qualify finite-angle/base shedding, material wake "
            "history, three-dimensional flow or production loads."
        ),
    }
    RESULTS.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, sort_keys=True))
