"""Run the preregistered S3ab characteristic space-time birth gate."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import yaml


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from claim_runtime.characteristic_birth_slab import (  # noqa: E402
    REFERENCE_NODES,
    solve_p2_characteristic_birth_slab,
)


CASES = (
    HERE
    / "docs"
    / "diag"
    / "actual_wake_space_time_birth_cases.yaml"
)
RESULTS = (
    HERE
    / "docs"
    / "diag"
    / "actual_wake_space_time_birth_results.json"
)


def _polynomial(coefficients: np.ndarray, value: np.ndarray) -> np.ndarray:
    return (
        coefficients[0]
        + coefficients[1] * value
        + coefficients[2] * value * value
    )


def _trace(time: np.ndarray | float) -> np.ndarray:
    value = np.asarray(time, dtype=float)
    return np.exp(0.7 * value) + 0.15 * np.sin(2.3 * value)


def _piecewise_p2(
    query: np.ndarray,
    *,
    final_time: float,
    step_count: int,
) -> np.ndarray:
    time = np.asarray(query, dtype=float)
    timestep = final_time / step_count
    index = np.minimum(
        np.floor(time / timestep).astype(np.int64),
        step_count - 1,
    )
    start = index * timestep
    local = (time - start) / timestep
    g0 = _trace(start)
    gm = _trace(start + 0.5 * timestep)
    g1 = _trace(start + timestep)
    l0 = 2.0 * (local - 0.5) * (local - 1.0)
    lm = 4.0 * local * (1.0 - local)
    l1 = 2.0 * local * (local - 0.5)
    return l0 * g0 + lm * gm + l1 * g1


def _l2_norm(
    function,
    *,
    final_time: float,
    partitions: int,
    quadrature_order: int = 16,
) -> float:
    abscissa, weight = np.polynomial.legendre.leggauss(
        quadrature_order
    )
    result = 0.0
    for index in range(partitions):
        left = final_time * index / partitions
        right = final_time * (index + 1) / partitions
        points = (
            0.5 * (left + right)
            + 0.5 * (right - left) * abscissa
        )
        values = np.asarray(function(points), dtype=float)
        result += (
            0.5
            * (right - left)
            * float(np.dot(weight, values * values))
        )
    return float(np.sqrt(result))


def _march(step_count: int, final_time: float) -> dict:
    timestep = final_time / step_count
    rows = []
    maximum_weak = 0.0
    maximum_mass = 0.0
    maximum_endpoint = 0.0
    maximum_mutation = 0.0
    for index in range(step_count):
        start = index * timestep
        inflow = _trace(
            np.array((start, start + 0.5 * timestep, start + timestep))
        )
        slab = solve_p2_characteristic_birth_slab(
            inflow,
            timestep=timestep,
            convection_speed=1.0,
            quadrature_order=8,
        )
        rows.append(slab.endpoint_chronological_rows.copy())
        maximum_weak = max(
            maximum_weak,
            slab.report.weak_residual_abs_max,
        )
        maximum_mass = max(
            maximum_mass,
            slab.report.newborn_mass_balance_abs,
        )
        maximum_endpoint = max(
            maximum_endpoint,
            slab.report.endpoint_trace_identity_abs_max,
        )
        maximum_mutation = max(
            maximum_mutation,
            slab.report.input_state_mutation_abs,
        )
    seam = max(
        (
            float(abs(rows[index][2] - rows[index + 1][0]))
            for index in range(len(rows) - 1)
        ),
        default=0.0,
    )
    exact_error = _l2_norm(
        lambda time: (
            _piecewise_p2(
                time,
                final_time=final_time,
                step_count=step_count,
            )
            - _trace(time)
        ),
        final_time=final_time,
        partitions=step_count,
    )
    return {
        "step_count": int(step_count),
        "timestep": float(timestep),
        "rows": rows,
        "chronological_seam_abs_max": seam,
        "weak_residual_abs_max": maximum_weak,
        "mass_balance_abs_max": maximum_mass,
        "endpoint_identity_abs_max": maximum_endpoint,
        "input_state_mutation_abs_max": maximum_mutation,
        "exact_L2_error": exact_error,
    }


def run() -> dict:
    contract = yaml.safe_load(CASES.read_text(encoding="utf-8"))
    canonical = contract["canonical"]
    thresholds = contract["thresholds"]
    tau = REFERENCE_NODES[:, 0] - REFERENCE_NODES[:, 1]
    polynomial_error = 0.0
    weak_residual = 0.0
    endpoint_error = 0.0
    mass_balance = 0.0
    maximum_condition = 0.0
    maximum_rank_deficiency = 0
    maximum_mutation = 0.0
    reference_values = []
    for record in canonical["polynomial_oracles"]:
        coefficients = np.asarray(record["coefficients"], dtype=float)
        inflow_tau = np.array((0.0, 0.5, 1.0), dtype=float)
        inflow = _polynomial(coefficients, inflow_tau)
        slab = solve_p2_characteristic_birth_slab(
            inflow,
            timestep=0.031,
            convection_speed=1.4,
            quadrature_order=int(
                canonical["quadrature"]["order"]
            ),
        )
        exact = _polynomial(coefficients, tau)
        polynomial_error = max(
            polynomial_error,
            float(
                np.max(
                    np.abs(slab.reference_node_values - exact),
                    initial=0.0,
                )
            ),
        )
        weak_residual = max(
            weak_residual,
            slab.report.weak_residual_abs_max,
        )
        endpoint_error = max(
            endpoint_error,
            slab.report.endpoint_trace_identity_abs_max,
        )
        mass_balance = max(
            mass_balance,
            slab.report.newborn_mass_balance_abs,
        )
        maximum_condition = max(
            maximum_condition,
            slab.report.free_condition_number,
        )
        maximum_rank_deficiency = max(
            maximum_rank_deficiency,
            slab.report.free_rank_deficiency,
        )
        maximum_mutation = max(
            maximum_mutation,
            slab.report.input_state_mutation_abs,
        )
        reference_values.append(slab.reference_node_values)

    scale_invariance = 0.0
    scale_reference = None
    scale_count = 0
    scale_inflow = np.array((0.3, -0.2, 0.8), dtype=float)
    for timestep in canonical["scale_family"]["timesteps"]:
        for speed in canonical["scale_family"][
            "convection_speeds"
        ]:
            slab = solve_p2_characteristic_birth_slab(
                scale_inflow,
                timestep=float(timestep),
                convection_speed=float(speed),
                quadrature_order=int(
                    canonical["quadrature"]["order"]
                ),
            )
            if scale_reference is None:
                scale_reference = slab.reference_node_values
            scale_invariance = max(
                scale_invariance,
                float(
                    np.max(
                        np.abs(
                            slab.reference_node_values
                            - scale_reference
                        ),
                        initial=0.0,
                    )
                ),
            )
            scale_count += 1

    repeated = canonical["repeated_insertion"]
    final_time = float(repeated["final_time"])
    step_counts = [
        int(value) for value in repeated["step_counts"]
    ]
    marches = [
        _march(step_count, final_time)
        for step_count in step_counts
    ]
    maximum_seam = max(
        march["chronological_seam_abs_max"]
        for march in marches
    )
    maximum_weak = max(
        weak_residual,
        *(march["weak_residual_abs_max"] for march in marches),
    )
    maximum_mass = max(
        mass_balance,
        *(march["mass_balance_abs_max"] for march in marches),
    )
    maximum_endpoint = max(
        endpoint_error,
        *(march["endpoint_identity_abs_max"] for march in marches),
    )
    maximum_mutation = max(
        maximum_mutation,
        *(march["input_state_mutation_abs_max"] for march in marches),
    )
    cauchy_differences = []
    for coarse, fine in zip(step_counts[:-1], step_counts[1:]):
        cauchy_differences.append(
            _l2_norm(
                lambda time, coarse=coarse, fine=fine: (
                    _piecewise_p2(
                        time,
                        final_time=final_time,
                        step_count=coarse,
                    )
                    - _piecewise_p2(
                        time,
                        final_time=final_time,
                        step_count=fine,
                    )
                ),
                final_time=final_time,
                partitions=fine,
            )
        )
    contractions = [
        cauchy_differences[index]
        / cauchy_differences[index + 1]
        for index in range(len(cauchy_differences) - 1)
    ]
    minimum_contraction = min(contractions)

    first = solve_p2_characteristic_birth_slab(
        np.array((1.0, 1.0, 1.0)),
        timestep=0.02,
        convection_speed=1.0,
        quadrature_order=int(canonical["quadrature"]["order"]),
    )
    checks = {
        "three_inflow_values_are_the_only_prescribed_data": (
            first.report.prescribed_dof_count
            == int(thresholds["prescribed_dof_count_expected"])
            and first.report.initial_newborn_scalar_count
            == int(
                thresholds[
                    "initial_newborn_scalar_count_expected"
                ]
            )
        ),
        "three_noninflow_values_are_solved_by_a_full_rank_block": (
            first.report.solved_dof_count
            == int(thresholds["solved_dof_count_expected"])
            and maximum_rank_deficiency
            <= int(thresholds["free_rank_deficiency_max"])
            and maximum_condition
            <= float(thresholds["free_condition_number_max"])
        ),
        "constant_linear_and_quadratic_characteristics_are_exact": (
            polynomial_error
            <= float(thresholds["polynomial_recovery_abs_max"])
        ),
        "all_test_weak_residual_closes": (
            maximum_weak
            <= float(thresholds["weak_residual_abs_max"])
        ),
        "endpoint_rows_retain_release_time_identity": (
            maximum_endpoint
            <= float(
                thresholds["endpoint_trace_identity_abs_max"]
            )
        ),
        "newborn_mass_equals_time_integrated_inflow": (
            maximum_mass
            <= float(thresholds["newborn_mass_balance_abs_max"])
        ),
        "positive_dt_and_speed_scaling_is_invariant": (
            scale_invariance
            <= float(thresholds["scale_invariance_abs_max"])
        ),
        "repeated_births_preserve_chronology_and_seams": (
            min(step_counts)
            >= int(thresholds["repeated_insertion_count_min"])
            and maximum_seam
            <= float(thresholds["chronological_seam_abs_max"])
        ),
        "smooth_trace_has_third_order_L2_cauchy_contraction": (
            minimum_contraction
            >= float(thresholds["L2_cauchy_contraction_min"])
        ),
        "inputs_are_immutable_and_no_newborn_initial_state_enters": (
            maximum_mutation
            <= float(thresholds["input_state_mutation_abs_max"])
            and first.report.initial_newborn_scalar_count == 0
        ),
    }
    decision = "GO" if all(checks.values()) else "NO-GO"
    result = {
        "artifact": "actual_wake_space_time_birth_oracle",
        "claim_node": contract["claim_node"],
        "stage": contract["stage"],
        "stage_decision": decision,
        "checks": checks,
        "aggregate_metrics": {
            "prescribed_dof_count": first.report.prescribed_dof_count,
            "solved_dof_count": first.report.solved_dof_count,
            "initial_newborn_scalar_count": (
                first.report.initial_newborn_scalar_count
            ),
            "free_rank_deficiency_max": maximum_rank_deficiency,
            "free_condition_number_max": maximum_condition,
            "polynomial_recovery_abs_max": polynomial_error,
            "weak_residual_abs_max": maximum_weak,
            "endpoint_trace_identity_abs_max": maximum_endpoint,
            "newborn_mass_balance_abs_max": maximum_mass,
            "scale_invariance_abs_max": scale_invariance,
            "scale_case_count": scale_count,
            "chronological_seam_abs_max": maximum_seam,
            "repeated_step_counts": step_counts,
            "repeated_exact_L2_errors": [
                march["exact_L2_error"] for march in marches
            ],
            "L2_cauchy_differences": cauchy_differences,
            "L2_cauchy_contractions": contractions,
            "L2_cauchy_contraction_min": minimum_contraction,
            "input_state_mutation_abs_max": maximum_mutation,
        },
        "forbidden_quantities_absent": [
            "newborn_initial_state",
            "old_state_remap",
            "copy",
            "average",
            "clamp",
            "blob_radius",
            "epsilon_offset",
            "smoothing",
            "damping",
            "pressure",
            "force",
            "LESP",
            "target_load",
            "structural_dynamics",
        ],
        "production_activation_allowed": False,
    }
    RESULTS.write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> int:
    result = run()
    print(json.dumps(result, indent=2))
    return 0 if result["stage_decision"] == "GO" else 1


if __name__ == "__main__":
    raise SystemExit(main())
