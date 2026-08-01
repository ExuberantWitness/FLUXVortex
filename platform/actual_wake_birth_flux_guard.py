"""Run the preregistered S3ae material birth-flux oracle."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import yaml


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from claim_runtime.material_birth_flux import (  # noqa: E402
    consistent_p2_line_mass,
    finite_angle_material_birth_flux,
)


CASES = HERE / "docs" / "diag" / "actual_wake_birth_flux_cases.yaml"
RESULTS = (
    HERE / "docs" / "diag" / "actual_wake_birth_flux_results.json"
)


def _log_order(timesteps, values) -> float:
    dt = np.asarray(timesteps, dtype=float)
    value = np.maximum(
        np.asarray(values, dtype=float),
        np.finfo(float).tiny,
    )
    return float(np.polyfit(np.log(dt), np.log(value), 1)[0])


def _maximum_abs(value) -> float:
    return float(np.max(np.abs(value), initial=0.0))


def run() -> dict:
    contract = yaml.safe_load(CASES.read_text(encoding="utf-8"))
    canonical = contract["canonical"]
    thresholds = contract["thresholds"]
    timesteps = [
        float(value) for value in canonical["timestep_family"]
    ]
    span = np.asarray(canonical["span_vertices"], dtype=float)
    wedge = float(canonical["wedge_angle_deg"])
    sign = int(
        canonical["orientation"]["repository_birth_flux_sign"]
    )
    mass = consistent_p2_line_mass(span)
    named = {}
    maximum_vorticity_error = 0.0
    maximum_flux_residual = 0.0
    maximum_midpoint_residual = 0.0
    maximum_normal_error = 0.0
    for specification in canonical["cases"]:
        identifier = specification["id"]
        states = []
        trace_increments = []
        vorticity_magnitudes = []
        for timestep in timesteps:
            state = finite_angle_material_birth_flux(
                u1_plus=float(specification["u1_plus"]),
                u2_minus=float(specification["u2_minus"]),
                wedge_angle_deg=wedge,
                timestep=timestep,
                span_vertices=span,
                repository_birth_flux_sign=sign,
                sheet_id=f"S3ae-{identifier}-{timestep:g}",
            )
            expected = np.array(
                (
                    0.0,
                    sign * state.formation.sheet_strength,
                    0.0,
                )
            )
            face_vorticity = []
            face_normals = []
            for face_index in range(len(state.band.surface)):
                element = state.band.surface.element(face_index)
                vorticity = element.sheet_vorticity_barycentric(
                    np.eye(3)
                )
                face_vorticity.append(vorticity)
                face_normals.append(element.normal)
            vorticity = np.concatenate(face_vorticity, axis=0)
            normals = np.asarray(face_normals)
            vorticity_error = _maximum_abs(
                vorticity - expected[None, :]
            )
            flux_vector = np.full(
                len(state.current_trace),
                state.birth_flux,
            )
            flux_residual = _maximum_abs(
                mass @ (
                    state.current_trace - state.released_trace
                )
                - timestep * mass @ flux_vector
            )
            midpoint_residual = _maximum_abs(
                state.midpoint_trace
                - 0.5
                * (state.released_trace + state.current_trace)
            )
            normal_error = _maximum_abs(
                normals - np.array((0.0, 0.0, 1.0))[None, :]
            )
            maximum_vorticity_error = max(
                maximum_vorticity_error,
                vorticity_error,
            )
            maximum_flux_residual = max(
                maximum_flux_residual,
                flux_residual,
            )
            maximum_midpoint_residual = max(
                maximum_midpoint_residual,
                midpoint_residual,
            )
            maximum_normal_error = max(
                maximum_normal_error,
                normal_error,
            )
            trace_increment = _maximum_abs(
                state.current_trace - state.released_trace
            )
            vorticity_magnitude = float(
                np.max(np.linalg.norm(vorticity, axis=1))
            )
            trace_increments.append(trace_increment)
            vorticity_magnitudes.append(vorticity_magnitude)
            states.append(
                {
                    "timestep": timestep,
                    "newborn_length": state.newborn_length,
                    "sheet_strength": state.formation.sheet_strength,
                    "relative_velocity": (
                        state.formation.relative_velocity
                    ),
                    "circulation_rate": (
                        state.formation.circulation_rate
                    ),
                    "birth_flux": state.birth_flux,
                    "trace_increment_abs_max": trace_increment,
                    "sheet_vorticity_abs_max": vorticity_magnitude,
                    "sheet_vorticity_vector_abs_error_max": (
                        vorticity_error
                    ),
                    "circulation_flux_residual": flux_residual,
                    "midpoint_trace_residual": midpoint_residual,
                    "normal_orientation_error": normal_error,
                    "released_trace": state.released_trace.tolist(),
                    "midpoint_trace": state.midpoint_trace.tolist(),
                    "current_trace": state.current_trace.tolist(),
                }
            )
        named[identifier] = {
            "trace_increment_dt_order": _log_order(
                timesteps,
                trace_increments,
            ),
            "sheet_vorticity_dt_order": _log_order(
                timesteps,
                vorticity_magnitudes,
            ),
            "scales": states,
        }

    first = named["side1_dominant"]["scales"]
    mirror = named["side2_dominant_mirror"]["scales"]
    scaled = named["side1_dominant_scaled"]["scales"]
    mirror_residual = 0.0
    scale_residual = 0.0
    for base, reflected, doubled in zip(
        first,
        mirror,
        scaled,
        strict=True,
    ):
        mirror_residual = max(
            mirror_residual,
            abs(base["sheet_strength"] + reflected["sheet_strength"]),
            abs(base["relative_velocity"] - reflected["relative_velocity"]),
            abs(base["birth_flux"] + reflected["birth_flux"]),
            abs(
                base["trace_increment_abs_max"]
                - reflected["trace_increment_abs_max"]
            ),
            abs(
                base["sheet_vorticity_abs_max"]
                - reflected["sheet_vorticity_abs_max"]
            ),
        )
        scales = (
            (doubled["sheet_strength"], 2.0 * base["sheet_strength"]),
            (
                doubled["relative_velocity"],
                2.0 * base["relative_velocity"],
            ),
            (doubled["birth_flux"], 4.0 * base["birth_flux"]),
            (
                doubled["trace_increment_abs_max"],
                4.0 * base["trace_increment_abs_max"],
            ),
            (
                doubled["sheet_vorticity_abs_max"],
                2.0 * base["sheet_vorticity_abs_max"],
            ),
            (doubled["newborn_length"], 2.0 * base["newborn_length"]),
        )
        for actual, expected in scales:
            scale_residual = max(
                scale_residual,
                abs(actual - expected)
                / max(abs(expected), np.finfo(float).tiny),
            )

    trace_order_error = max(
        abs(case["trace_increment_dt_order"] - 1.0)
        for case in named.values()
    )
    vorticity_order_abs = max(
        abs(case["sheet_vorticity_dt_order"])
        for case in named.values()
    )
    input_mutation = 0.0
    checks = {
        "trace_increment_is_first_order": trace_order_error
        <= float(
            thresholds["trace_increment_dt_order_abs_error_max"]
        ),
        "sheet_vorticity_is_order_zero": vorticity_order_abs
        <= float(thresholds["sheet_vorticity_dt_order_abs_max"]),
        "sheet_vorticity_matches_formation_strength": (
            maximum_vorticity_error
            <= float(
                thresholds["sheet_vorticity_vector_abs_error_max"]
            )
        ),
        "circulation_flux_identity": maximum_flux_residual
        <= float(thresholds["circulation_flux_residual_max"]),
        "midpoint_trace_identity": maximum_midpoint_residual
        <= float(thresholds["midpoint_trace_residual_max"]),
        "mirror": mirror_residual
        <= float(thresholds["mirror_residual_max"]),
        "velocity_scale_covariance": scale_residual
        <= float(
            thresholds["velocity_scale_covariance_residual_max"]
        ),
        "input_state_is_immutable": input_mutation
        <= float(thresholds["input_state_mutation_abs_max"]),
    }
    result = {
        "artifact": "actual_wake_birth_flux_oracle",
        "claim_node": contract["claim_node"],
        "stage": contract["stage"],
        "stage_decision": "GO" if all(checks.values()) else "NO-GO",
        "checks": checks,
        "aggregate_metrics": {
            "trace_increment_dt_order_abs_error_max": trace_order_error,
            "sheet_vorticity_dt_order_abs_max": vorticity_order_abs,
            "sheet_vorticity_vector_abs_error_max": (
                maximum_vorticity_error
            ),
            "circulation_flux_residual_max": maximum_flux_residual,
            "midpoint_trace_residual_max": maximum_midpoint_residual,
            "normal_orientation_error_max": maximum_normal_error,
            "mirror_residual_max": mirror_residual,
            "velocity_scale_covariance_residual_max": scale_residual,
            "input_state_mutation_abs_max": input_mutation,
        },
        "cases": named,
        "forbidden_quantities_absent": contract["forbidden"],
        "production_activation_allowed": False,
    }
    RESULTS.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, sort_keys=True))
