"""Run the preregistered circular-cylinder circulation-cut oracle."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from claim_runtime.circulation_cut_oracle import (  # noqa: E402
    CircularP2TraceEvaluation,
    circular_p2_trace,
)


CASES = (
    HERE / "docs" / "diag" / "actual_boundary_circulation_cut_cases.yaml"
)
RESULTS = (
    HERE / "docs" / "diag"
    / "actual_boundary_circulation_cut_results.json"
)


def _relative_error(actual: float, expected: float) -> float:
    return float(
        abs(actual - expected)
        / max(abs(expected), np.finfo(float).tiny)
    )


def _case_record(case: CircularP2TraceEvaluation) -> dict:
    return {
        "topology": case.topology,
        "panel_count": case.panel_count,
        "quadrature_order": case.quadrature_order,
        "potential_jump": case.potential_jump,
        "circulation": case.circulation,
        "telescoping_circulation": case.telescoping_circulation,
        "tangential_velocity_rms_error": (
            case.tangential_velocity_rms_error
        ),
        "pressure_force": case.pressure_force.tolist(),
        "expected_pressure_force": (
            case.expected_pressure_force.tolist()
        ),
        "lift_relative_error": case.lift_relative_error,
        "drag_absolute": abs(case.drag),
    }


def _nonincreasing(values: list[float]) -> bool:
    return all(
        following <= previous
        for previous, following in zip(values, values[1:])
    )


def run() -> dict:
    contract = yaml.safe_load(CASES.read_text(encoding="utf-8"))
    canonical = contract["canonical"]
    thresholds = contract["thresholds"]
    common = {
        "radius": float(canonical["radius"]),
        "freestream_speed": float(canonical["freestream_speed"]),
        "density": float(canonical["density"]),
        "prescribed_circulation": float(
            canonical["prescribed_circulation"]
        ),
        "quadrature_order": int(canonical["quadrature_order"]),
    }
    counts = [int(value) for value in canonical["p2_panel_counts"]]
    cut_cases = [
        circular_p2_trace(
            panel_count=count,
            topology="cut",
            **common,
        )
        for count in counts
    ]
    closed_cases = [
        circular_p2_trace(
            panel_count=count,
            topology="closed",
            **common,
        )
        for count in counts
    ]
    gauge_shift = 7.25
    gauge_case = circular_p2_trace(
        panel_count=counts[-1],
        topology="cut",
        potential_gauge=gauge_shift,
        **common,
    )
    finest = cut_cases[-1]
    gamma = common["prescribed_circulation"]

    closed_circulation_max = float(max(
        max(abs(case.circulation), abs(case.telescoping_circulation))
        for case in closed_cases
    ))
    cut_circulation_relative_error_max = float(max(
        max(
            _relative_error(case.circulation, gamma),
            _relative_error(case.telescoping_circulation, gamma),
        )
        for case in cut_cases
    ))
    cut_jump_relative_error_max = float(max(
        _relative_error(case.potential_jump, gamma)
        for case in cut_cases
    ))
    gauge_velocity_change = float(np.max(
        np.abs(
            gauge_case.tangential_velocity
            - finest.tangential_velocity
        ),
        initial=0.0,
    ))
    gauge_force_change = float(np.max(
        np.abs(gauge_case.pressure_force - finest.pressure_force),
        initial=0.0,
    ))
    velocity_errors = [
        case.tangential_velocity_rms_error for case in cut_cases
    ]
    lift_errors = [case.lift_relative_error for case in cut_cases]

    checks = {
        "closed_trace_telescopes_to_zero": (
            closed_circulation_max
            <= float(
                thresholds["closed_trace_circulation_absolute_max"]
            )
        ),
        "cut_trace_recovers_circulation": (
            cut_circulation_relative_error_max
            <= float(
                thresholds[
                    "cut_trace_circulation_relative_error_max"
                ]
            )
        ),
        "cut_jump_equals_circulation": (
            cut_jump_relative_error_max
            <= float(thresholds["cut_jump_relative_error_max"])
        ),
        "finest_velocity_accuracy": (
            finest.tangential_velocity_rms_error
            <= float(
                thresholds[
                    "finest_tangential_velocity_rms_error_max"
                ]
            )
        ),
        "finest_kutta_joukowski_lift": (
            finest.lift_relative_error
            <= float(thresholds["finest_lift_relative_error_max"])
        ),
        "finest_zero_drag": (
            abs(finest.drag)
            <= float(thresholds["finest_drag_absolute_max"])
        ),
        "gauge_velocity_invariance": (
            gauge_velocity_change
            <= float(
                thresholds["gauge_velocity_absolute_change_max"]
            )
        ),
        "gauge_force_invariance": (
            gauge_force_change
            <= float(thresholds["gauge_force_absolute_change_max"])
        ),
        "velocity_errors_monotone": _nonincreasing(velocity_errors),
        "force_errors_monotone": _nonincreasing(lift_errors),
    }
    go = all(checks.values())
    result = {
        "artifact": "actual_boundary_circulation_cut_oracle",
        "claim_node": "N3.1j3b6d",
        "stage": "S2a_circulation_cut_topology",
        "canonical": canonical,
        "closed_cases": [_case_record(case) for case in closed_cases],
        "cut_cases": [_case_record(case) for case in cut_cases],
        "aggregate_metrics": {
            "closed_circulation_absolute_max": (
                closed_circulation_max
            ),
            "cut_circulation_relative_error_max": (
                cut_circulation_relative_error_max
            ),
            "cut_jump_relative_error_max": (
                cut_jump_relative_error_max
            ),
            "gauge_shift": gauge_shift,
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
            "Passing validates only that nonzero circulation requires a "
            "classified potential cut/wake jump and that the same curved "
            "P2 trace recovers the attached two-dimensional pressure force. "
            "It does not validate a finite-wing wake, Kutta/base closure, "
            "unsteady pressure, separated flow, or production loads."
        ),
    }
    RESULTS.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, sort_keys=True))
