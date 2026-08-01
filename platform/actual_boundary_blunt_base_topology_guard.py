"""Run the preregistered S2d finite-blunt-base topology oracle."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from claim_runtime.blunt_base_topology import (  # noqa: E402
    BluntBaseTopology,
    naca4_blunt_base_topology,
)


CASES = (
    HERE / "docs" / "diag"
    / "actual_boundary_blunt_base_topology_cases.yaml"
)
RESULTS = (
    HERE / "docs" / "diag"
    / "actual_boundary_blunt_base_topology_results.json"
)


def _record(case: BluntBaseTopology) -> dict:
    return {
        "thickness_coefficient": case.thickness_coefficient,
        "upper_corner": case.upper_corner.tolist(),
        "lower_corner": case.lower_corner.tolist(),
        "upper_tangent": case.upper_tangent.tolist(),
        "lower_tangent": case.lower_tangent.tolist(),
        "half_base_thickness": case.half_base_thickness,
        "base_thickness": case.base_thickness,
        "geometry_identity_residual": (
            case.geometry_identity_residual
        ),
        "optimal_single_origin": (
            case.optimal_single_origin.tolist()
        ),
        "optimal_single_origin_residual": (
            case.optimal_single_origin_residual
        ),
        "normalized_single_origin_residual": (
            case.normalized_single_origin_residual
        ),
        "two_origin_attachment_residual": (
            case.two_origin_attachment_residual
        ),
        "tangent_gap_angle_deg": float(
            np.rad2deg(case.tangent_gap_angle_rad)
        ),
        "optimal_B2_direction": (
            case.optimal_B2_direction.tolist()
        ),
        "optimal_B2_direction_mismatch_deg": float(
            np.rad2deg(case.optimal_B2_direction_mismatch_rad)
        ),
        "single_junction_topologically_admissible": (
            case.single_junction_topologically_admissible
        ),
    }


def run() -> dict:
    contract = yaml.safe_load(CASES.read_text(encoding="utf-8"))
    geometry = contract["geometry_continuation"]
    thresholds = contract["thresholds"]
    fractions = [
        float(value) for value in geometry["base_fractions"]
    ]
    states = [
        naca4_blunt_base_topology(
            base_fraction=fraction,
            maximum_camber=float(geometry["maximum_camber"]),
            camber_location=float(geometry["camber_location"]),
            thickness_ratio=float(geometry["thickness_ratio"]),
            chord=float(geometry["chord"]),
        )
        for fraction in fractions
    ]
    nonzero = states[:-1]
    closed = states[-1]
    bases = np.array([state.base_thickness for state in states])
    origins = np.array(
        [state.optimal_single_origin_residual for state in states]
    )
    geometry_error = max(
        abs(state.geometry_identity_residual) for state in states
    )
    normalized_identity_error = max(
        abs(state.normalized_single_origin_residual - 0.5)
        for state in nonzero
    )
    two_origin_error = max(
        state.two_origin_attachment_residual for state in states
    )
    open_base_error = abs(states[0].base_thickness - 0.00126)
    monotonic_tolerance = float(
        thresholds["monotonic_abs_tolerance"]
    )
    checks = {
        "geometry_identity": geometry_error
        <= float(thresholds["geometry_identity_abs_max"]),
        "standard_open_base": open_base_error
        <= float(
            thresholds["open_base_thickness_abs_error_max"]
        ),
        "finite_bases_positive": all(
            state.base_thickness
            > float(thresholds["positive_base_floor"])
            for state in nonzero
        ),
        "single_origin_identity": normalized_identity_error
        <= float(
            thresholds[
                "single_origin_normalized_identity_abs_error_max"
            ]
        ),
        "single_origin_nonzero_for_finite_base": all(
            state.optimal_single_origin_residual
            > float(thresholds["positive_base_floor"])
            and not state.single_junction_topologically_admissible
            for state in nonzero
        ),
        "two_origins_attach_exactly": two_origin_error
        <= float(
            thresholds["two_origin_attachment_abs_max"]
        ),
        "monotonic_sharp_limit": (
            bool(np.all(np.diff(bases) <= monotonic_tolerance))
            and bool(
                np.all(np.diff(origins) <= monotonic_tolerance)
            )
        ),
        "coincident_finite_angle_limit": (
            closed.base_thickness
            <= float(thresholds["coincident_corner_abs_max"])
            and closed.single_junction_topologically_admissible
            and np.rad2deg(closed.tangent_gap_angle_rad)
            > float(thresholds["finite_angle_floor_deg"])
        ),
        "B2_comparator_remains_separate": all(
            state.optimal_B2_direction_mismatch_rad > 0.0
            for state in states
        ),
    }
    checks = {name: bool(value) for name, value in checks.items()}
    result = {
        "artifact": "actual_boundary_blunt_base_topology_oracle",
        "claim_node": contract["claim_node"],
        "stage": contract["stage"],
        "canonical": geometry,
        "cases": {
            str(state.base_fraction): _record(state)
            for state in states
        },
        "aggregate_metrics": {
            "geometry_identity_abs_max": geometry_error,
            "open_base_thickness_abs_error": open_base_error,
            "single_origin_normalized_identity_abs_error_max": (
                normalized_identity_error
            ),
            "two_origin_attachment_abs_max": two_origin_error,
            "minimum_finite_base": min(
                state.base_thickness for state in nonzero
            ),
            "closed_base_thickness": closed.base_thickness,
            "closed_finite_angle_deg": float(
                np.rad2deg(closed.tangent_gap_angle_rad)
            ),
        },
        "thresholds": thresholds,
        "checks": checks,
        "stage_decision": (
            "GO" if all(checks.values()) else "NO-GO"
        ),
        "production_activation_allowed": False,
        "interpretation": (
            "Passing proves only that a finite two-corner base cannot be "
            "represented by one exact material sheet origin.  The next "
            "state must retain two fronts or an equivalent finite-width "
            "interface plus a base/confluence region; its dynamics and "
            "pressure remain unresolved."
        ),
    }
    RESULTS.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, sort_keys=True))
