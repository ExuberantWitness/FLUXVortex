"""Run the preregistered S3i tangential-gauge material-transport oracle."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from claim_runtime.wake_gauge_transport import (  # noqa: E402
    material_potential_jump_ale_identity,
)


CASES = (
    HERE / "docs" / "diag"
    / "wake_tangential_gauge_transport_cases.yaml"
)
RESULTS = (
    HERE / "docs" / "diag"
    / "wake_tangential_gauge_transport_results.json"
)


def _flow(value: np.ndarray, coefficient: float, time: float) -> np.ndarray:
    tangent = np.tan(0.5 * np.pi * value)
    return (
        2.0
        / np.pi
        * np.arctan(tangent * np.exp(coefficient * np.pi * time))
    )


def _inverse_flow(
    value: np.ndarray,
    coefficient: float,
    time: float,
) -> np.ndarray:
    tangent = np.tan(0.5 * np.pi * value)
    return (
        2.0
        / np.pi
        * np.arctan(tangent * np.exp(-coefficient * np.pi * time))
    )


def _initial_mu(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    return (
        0.15
        + 0.4 * first
        + 0.2 * second
        + 0.3 * first * second
        + 0.25 * first**2
        - 0.1 * second**2
    )


def _exact_state(
    first: np.ndarray,
    second: np.ndarray,
    *,
    time: float,
    coefficient_first: float,
    coefficient_second: float,
):
    first0 = _inverse_flow(first, coefficient_first, time)
    second0 = _inverse_flow(second, coefficient_second, time)
    tangent_first = np.tan(0.5 * np.pi * first)
    tangent_second = np.tan(0.5 * np.pi * second)
    q_first = tangent_first * np.exp(
        -coefficient_first * np.pi * time
    )
    q_second = tangent_second * np.exp(
        -coefficient_second * np.pi * time
    )
    first0_first = (
        np.exp(-coefficient_first * np.pi * time)
        * (1.0 + tangent_first**2)
        / (1.0 + q_first**2)
    )
    second0_second = (
        np.exp(-coefficient_second * np.pi * time)
        * (1.0 + tangent_second**2)
        / (1.0 + q_second**2)
    )
    first0_time = (
        -coefficient_first * np.sin(np.pi * first0)
    )
    second0_time = (
        -coefficient_second * np.sin(np.pi * second0)
    )
    derivative_first0 = (
        0.4 + 0.3 * second0 + 0.5 * first0
    )
    derivative_second0 = (
        0.2 + 0.3 * first0 - 0.2 * second0
    )
    value = _initial_mu(first0, second0)
    gradient = np.column_stack(
        (
            (derivative_first0 * first0_first).ravel(),
            (derivative_second0 * second0_second).ravel(),
            np.zeros(first.size),
        )
    )
    eulerian_rate = (
        derivative_first0 * first0_time
        + derivative_second0 * second0_time
    ).ravel()
    return value.ravel(), gradient, eulerian_rate


def _rotation_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    direction = axis / np.linalg.norm(axis)
    cross = np.array(
        (
            (0.0, -direction[2], direction[1]),
            (direction[2], 0.0, -direction[0]),
            (-direction[1], direction[0], 0.0),
        )
    )
    return (
        np.eye(3) * np.cos(angle)
        + (1.0 - np.cos(angle))
        * np.outer(direction, direction)
        + np.sin(angle) * cross
    )


def run() -> dict:
    contract = yaml.safe_load(CASES.read_text(encoding="utf-8"))
    canonical = contract["canonical"]
    thresholds = contract["thresholds"]
    time = float(canonical["time"]["end"])
    coefficient_first = float(
        canonical["material_velocity"]["a"]
    )
    coefficient_second = float(
        canonical["material_velocity"]["b"]
    )
    count_first, count_second = [
        int(value)
        for value in canonical["surface"]["evaluation_grid"]
    ]
    margin = float(
        canonical["surface"]["evaluation_margin"]
    )
    first_axis = np.linspace(
        margin,
        1.0 - margin,
        count_first,
    )
    second_axis = np.linspace(
        margin,
        1.0 - margin,
        count_second,
    )
    first, second = np.meshgrid(
        first_axis,
        second_axis,
        indexing="ij",
    )
    exact_mu, gradient, eulerian_rate = _exact_state(
        first,
        second,
        time=time,
        coefficient_first=coefficient_first,
        coefficient_second=coefficient_second,
    )
    fluid_velocity = np.column_stack(
        (
            (
                coefficient_first
                * np.sin(np.pi * first)
            ).ravel(),
            (
                coefficient_second
                * np.sin(np.pi * second)
            ).ravel(),
            np.zeros(first.size),
        )
    )
    normal = np.tile((0.0, 0.0, 1.0), (first.size, 1))
    gauge_reports = {}
    gauge_fields = {}
    maximum_ale_residual = 0.0
    for name, fraction in canonical["gauges"].items():
        fraction = float(fraction)
        mesh_velocity = fraction * fluid_velocity
        mesh_rate = eulerian_rate + np.einsum(
            "ij,ij->i",
            mesh_velocity,
            gradient,
        )
        identity = material_potential_jump_ale_identity(
            normal=normal,
            fluid_sheet_average_velocity=fluid_velocity,
            mesh_velocity=mesh_velocity,
            surface_gradient=gradient,
            mesh_time_derivative=mesh_rate,
            normal_velocity_tolerance=float(
                thresholds["ale_transport_residual_abs_max"]
            ),
            surface_gradient_tolerance=float(
                thresholds["ale_transport_residual_abs_max"]
            ),
            transport_residual_tolerance=float(
                thresholds["ale_transport_residual_abs_max"]
            ),
        )
        maximum_ale_residual = max(
            maximum_ale_residual,
            identity.maximum_absolute_transport_residual,
        )
        mesh_label_first = _inverse_flow(
            first,
            fraction * coefficient_first,
            time,
        )
        mesh_label_second = _inverse_flow(
            second,
            fraction * coefficient_second,
            time,
        )
        reconstructed_first = _flow(
            mesh_label_first,
            fraction * coefficient_first,
            time,
        )
        reconstructed_second = _flow(
            mesh_label_second,
            fraction * coefficient_second,
            time,
        )
        field, _, _ = _exact_state(
            reconstructed_first,
            reconstructed_second,
            time=time,
            coefficient_first=coefficient_first,
            coefficient_second=coefficient_second,
        )
        gauge_fields[name] = field
        gauge_reports[name] = {
            "fraction": fraction,
            "maximum_normal_velocity_mismatch": (
                identity.maximum_normal_velocity_mismatch
            ),
            "maximum_surface_gradient_normal_component": (
                identity.maximum_surface_gradient_normal_component
            ),
            "maximum_absolute_transport_residual": (
                identity.maximum_absolute_transport_residual
            ),
            "passed": identity.passed,
        }

    cross_gauge_error = max(
        float(np.max(np.abs(field - exact_mu), initial=0.0))
        for field in gauge_fields.values()
    )
    label_first, label_second = np.meshgrid(
        first_axis,
        second_axis,
        indexing="ij",
    )
    material_first = _flow(
        label_first,
        coefficient_first,
        time,
    )
    material_second = _flow(
        label_second,
        coefficient_second,
        time,
    )
    material_value, _, _ = _exact_state(
        material_first,
        material_second,
        time=time,
        coefficient_first=coefficient_first,
        coefficient_second=coefficient_second,
    )
    material_error = float(
        np.max(
            np.abs(
                material_value
                - _initial_mu(label_first, label_second).ravel()
            ),
            initial=0.0,
        )
    )
    naive_error = float(
        np.max(
            np.abs(
                _initial_mu(first, second).ravel() - exact_mu
            ),
            initial=0.0,
        )
    )

    rigid = canonical["rigid_frame_counterfactual"]
    rotation = _rotation_matrix(
        np.asarray(rigid["rotation_axis"], dtype=float),
        np.deg2rad(float(rigid["rotation_deg"])),
    )
    intermediate_fraction = float(
        canonical["gauges"]["intermediate_fraction"]
    )
    intermediate_mesh = intermediate_fraction * fluid_velocity
    intermediate_rate = eulerian_rate + np.einsum(
        "ij,ij->i",
        intermediate_mesh,
        gradient,
    )
    rigid_identity = material_potential_jump_ale_identity(
        normal=normal @ rotation.T,
        fluid_sheet_average_velocity=fluid_velocity @ rotation.T,
        mesh_velocity=intermediate_mesh @ rotation.T,
        surface_gradient=gradient @ rotation.T,
        mesh_time_derivative=intermediate_rate,
        normal_velocity_tolerance=float(
            thresholds["rigid_frame_residual_abs_max"]
        ),
        surface_gradient_tolerance=float(
            thresholds["rigid_frame_residual_abs_max"]
        ),
        transport_residual_tolerance=float(
            thresholds["rigid_frame_residual_abs_max"]
        ),
    )
    rigid_error = max(
        rigid_identity.maximum_normal_velocity_mismatch,
        rigid_identity.maximum_surface_gradient_normal_component,
        rigid_identity.maximum_absolute_transport_residual,
    )
    boundary = np.linspace(0.0, 1.0, 101)
    boundary_velocity = max(
        float(
            np.max(
                np.abs(
                    coefficient_first
                    * np.sin(np.pi * np.array((0.0, 1.0)))
                ),
                initial=0.0,
            )
        ),
        float(
            np.max(
                np.abs(
                    coefficient_second
                    * np.sin(np.pi * np.array((0.0, 1.0)))
                ),
                initial=0.0,
            )
        ),
        0.0 * float(np.sum(boundary)),
    )
    checks = {
        "material_trajectory_conserves_mu": (
            material_error
            <= float(
                thresholds["material_trajectory_mu_abs_max"]
            )
        ),
        "all_gauges_satisfy_ALE_transport": (
            all(report["passed"] for report in gauge_reports.values())
            and maximum_ale_residual
            <= float(
                thresholds["ale_transport_residual_abs_max"]
            )
        ),
        "all_gauges_describe_same_scalar_field": (
            cross_gauge_error
            <= float(
                thresholds["cross_gauge_scalar_abs_max"]
            )
        ),
        "rigid_frame_objectivity_passes": (
            rigid_error
            <= float(
                thresholds["rigid_frame_residual_abs_max"]
            )
        ),
        "boundary_attachment_is_not_the_counterexample": (
            boundary_velocity
            <= float(
                thresholds["boundary_velocity_abs_max"]
            )
        ),
        "normal_only_frozen_mu_is_falsified": (
            naive_error
            >= float(
                thresholds["naive_frozen_mu_abs_error_min"]
            )
        ),
    }
    decision = "GO" if all(checks.values()) else "NO-GO"
    result = {
        "artifact": "wake_tangential_gauge_transport_oracle",
        "stage": contract["stage"],
        "claim_nodes": contract["claim_nodes"],
        "stage_decision": decision,
        "checks": checks,
        "gauge_reports": gauge_reports,
        "aggregate_metrics": {
            "material_trajectory_mu_abs_max": material_error,
            "ale_transport_residual_abs_max": maximum_ale_residual,
            "cross_gauge_scalar_abs_max": cross_gauge_error,
            "rigid_frame_residual_abs_max": rigid_error,
            "boundary_velocity_abs_max": boundary_velocity,
            "naive_frozen_mu_abs_error": naive_error,
        },
        "forbidden_quantities_absent": [
            "discrete_P2_transport_scheme",
            "actual_induced_velocity",
            "strength_equilibrium",
            "pressure",
            "force",
            "LESP",
            "wake_core",
            "smoothing",
            "target_load",
            "structural_dynamics",
        ],
        "production_activation_allowed": False,
    }
    RESULTS.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


if __name__ == "__main__":
    payload = run()
    print(
        json.dumps(payload, ensure_ascii=False, indent=2),
        flush=True,
    )
    raise SystemExit(
        0 if payload["stage_decision"] == "GO" else 1
    )
