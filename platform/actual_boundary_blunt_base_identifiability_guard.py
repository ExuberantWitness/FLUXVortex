"""Run the preregistered S2e blunt-base identifiability oracle."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from claim_runtime.blunt_base_identifiability import (  # noqa: E402
    BluntBaseCornerWitness,
    blunt_base_corner_witnesses,
)
from claim_runtime.blunt_base_topology import (  # noqa: E402
    naca4_blunt_base_topology,
)


CASES = (
    HERE / "docs" / "diag"
    / "actual_boundary_blunt_base_identifiability_cases.yaml"
)
RESULTS = (
    HERE / "docs" / "diag"
    / "actual_boundary_blunt_base_identifiability_results.json"
)


def _formation_record(state) -> dict:
    return {
        "forming_angle_deg": float(
            np.rad2deg(state.delta_theta1)
        ),
        "sheet_strength": state.sheet_strength,
        "circulation_rate": state.circulation_rate,
        "relative_velocity": state.relative_velocity,
        "u_g_plus": state.u_g_plus,
        "u_g_minus": state.u_g_minus,
        "normalized_residuals": {
            "angle_sum": state.normalized_angle_sum_residual,
            "direction": state.normalized_direction_residual,
            "kutta_strength": (
                state.normalized_kutta_strength_residual
            ),
            "circulation_rate": (
                state.normalized_circulation_rate_residual
            ),
            "momentum": state.normalized_momentum_residual,
        },
    }


def _record(witness: BluntBaseCornerWitness) -> dict:
    return {
        "observed_outer_speed": witness.observed_outer_speed,
        "base_speed_ratio": witness.base_speed_ratio,
        "upper_wedge_angle_deg": witness.upper_wedge_angle_deg,
        "lower_wedge_angle_deg": witness.lower_wedge_angle_deg,
        "upper": _formation_record(witness.upper),
        "lower": _formation_record(witness.lower),
    }


def run() -> dict:
    contract = yaml.safe_load(CASES.read_text(encoding="utf-8"))
    canonical = contract["canonical"]
    geometry_spec = canonical["geometry"]
    thresholds = contract["thresholds"]
    topology = naca4_blunt_base_topology(
        base_fraction=float(geometry_spec["base_fraction"]),
        maximum_camber=float(geometry_spec["maximum_camber"]),
        camber_location=float(geometry_spec["camber_location"]),
        thickness_ratio=float(geometry_spec["thickness_ratio"]),
        chord=float(geometry_spec["chord"]),
    )
    witnesses = blunt_base_corner_witnesses(
        base_speed_ratios=tuple(
            float(value)
            for value in canonical["unobserved_base_speed_ratios"]
        ),
        observed_outer_speed=float(
            canonical["fixed_observed_outer_speed"]
        ),
        geometry=topology,
    )
    formations = [
        state
        for witness in witnesses
        for state in (witness.upper, witness.lower)
    ]
    residual_values = [
        value
        for state in formations
        for value in (
            state.normalized_angle_sum_residual,
            state.normalized_direction_residual,
            state.normalized_kutta_strength_residual,
            state.normalized_circulation_rate_residual,
            state.normalized_momentum_residual,
        )
        if value is not None
    ]
    residual_max = max(residual_values)
    minimum_side_velocity = min(
        min(state.u_g_plus, state.u_g_minus)
        for state in formations
    )
    wedge_mirror = max(
        abs(
            witness.upper_wedge_angle_deg
            - witness.lower_wedge_angle_deg
        )
        for witness in witnesses
    )
    family_mirror = max(
        max(
            abs(
                witness.upper.delta_theta1
                - witness.lower.delta_theta1
            ),
            abs(
                witness.upper.sheet_strength
                - witness.lower.sheet_strength
            ),
            abs(
                witness.upper.relative_velocity
                - witness.lower.relative_velocity
            ),
            abs(witness.upper.u_g_plus - witness.lower.u_g_plus),
            abs(witness.upper.u_g_minus - witness.lower.u_g_minus),
        )
        for witness in witnesses
    )
    upper = [witness.upper for witness in witnesses]
    angle_spread = float(
        np.rad2deg(
            max(state.delta_theta1 for state in upper)
            - min(state.delta_theta1 for state in upper)
        )
    )
    strength_spread = (
        max(state.sheet_strength for state in upper)
        - min(state.sheet_strength for state in upper)
    )
    relative_speed_spread = (
        max(state.relative_velocity for state in upper)
        - min(state.relative_velocity for state in upper)
    )
    outer_identity = max(
        abs(
            witness.observed_outer_speed
            - float(canonical["fixed_observed_outer_speed"])
        )
        for witness in witnesses
    )
    checks = {
        "all_local_conservation_identities": residual_max
        <= float(thresholds["local_normalized_residual_max"]),
        "all_sheet_side_velocities_nonnegative": minimum_side_velocity
        >= -float(
            thresholds["sheet_side_velocity_negative_tolerance"]
        ),
        "corner_wedge_mirror": wedge_mirror
        <= float(thresholds["corner_wedge_mirror_abs_deg_max"]),
        "corner_family_mirror": family_mirror
        <= float(thresholds["corner_family_mirror_abs_max"]),
        "forming_angle_nonunique": angle_spread
        >= float(thresholds["forming_angle_spread_min_deg"]),
        "sheet_strength_nonunique": strength_spread
        >= float(thresholds["sheet_strength_spread_min"]),
        "relative_velocity_nonunique": relative_speed_spread
        >= float(thresholds["relative_velocity_spread_min"]),
        "observed_outer_input_identical": outer_identity
        <= float(thresholds["outer_input_identity_abs_max"]),
    }
    checks = {name: bool(value) for name, value in checks.items()}
    result = {
        "artifact": (
            "actual_boundary_blunt_base_identifiability_oracle"
        ),
        "claim_node": contract["claim_node"],
        "stage": contract["stage"],
        "canonical": canonical,
        "cases": {
            str(witness.base_speed_ratio): _record(witness)
            for witness in witnesses
        },
        "aggregate_metrics": {
            "local_normalized_residual_max": residual_max,
            "minimum_sheet_side_velocity": minimum_side_velocity,
            "corner_wedge_mirror_abs_deg": wedge_mirror,
            "corner_family_mirror_abs_max": family_mirror,
            "forming_angle_spread_deg": angle_spread,
            "sheet_strength_spread": strength_spread,
            "relative_velocity_spread": relative_speed_spread,
            "observed_outer_input_identity_abs_max": outer_identity,
        },
        "thresholds": thresholds,
        "checks": checks,
        "stage_decision": (
            "GO" if all(checks.values()) else "NO-GO"
        ),
        "outer_only_closure_identifiable": False,
        "production_activation_allowed": False,
        "interpretation": (
            "Passing falsifies outer-only two-front closure: identical "
            "outer speed and geometry admit multiple locally conservative "
            "states when the unobserved base-side flow changes.  It does "
            "not select any witness as physical."
        ),
    }
    RESULTS.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, sort_keys=True))
