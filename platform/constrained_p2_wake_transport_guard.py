"""Run the preregistered S3l essential-P2-trace transport oracle."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from claim_runtime.distributed_doublet import DistributedDoubletError  # noqa: E402
from claim_runtime.p2_surface_material_transport import (  # noqa: E402
    P2EssentialScalarTrace,
    p2_essential_trace_transport_rate,
)
from moving_p2_wake_transport_guard import (  # noqa: E402
    _operator,
    _relative_mass_error,
    _rotation_matrix,
    _trace_and_boundary,
)


CASES = (
    HERE / "docs" / "diag"
    / "constrained_p2_wake_transport_cases.yaml"
)
RESULTS = (
    HERE / "docs" / "diag"
    / "constrained_p2_wake_transport_results.json"
)


def _flow_inverse(value: np.ndarray, coefficient: float, time: float) -> np.ndarray:
    return (
        2.0
        / np.pi
        * np.arctan(
            np.tan(0.5 * np.pi * value)
            * np.exp(-coefficient * np.pi * time)
        )
    )


def _case_state(
    first: np.ndarray,
    second: np.ndarray,
    time: float,
    *,
    case_name: str,
    coefficient_first: float,
    coefficient_second: float,
) -> tuple[np.ndarray, np.ndarray]:
    first0 = _flow_inverse(first, coefficient_first, time)
    second0 = _flow_inverse(second, coefficient_second, time)
    first0_rate = -coefficient_first * np.sin(np.pi * first0)
    second0_rate = -coefficient_second * np.sin(np.pi * second0)
    if case_name == "zero_all_boundaries":
        value = (
            first0
            * (1.0 - first0)
            * second0
            * (1.0 - second0)
        )
        derivative_first = (
            (1.0 - 2.0 * first0)
            * second0
            * (1.0 - second0)
        )
        derivative_second = (
            first0
            * (1.0 - first0)
            * (1.0 - 2.0 * second0)
        )
    elif case_name == "nonzero_body_attachment":
        value = first0 * second0 * (1.0 - second0)
        derivative_first = second0 * (1.0 - second0)
        derivative_second = first0 * (1.0 - 2.0 * second0)
    else:
        raise ValueError(f"unknown manufactured case {case_name!r}")
    rate = (
        derivative_first * first0_rate
        + derivative_second * second0_rate
    )
    return value, rate


def _base_coordinates(
    operator,
    *,
    rotation: np.ndarray | None,
    translation: np.ndarray | None,
) -> np.ndarray:
    coordinates = (
        operator.operator.topology.degree_of_freedom_coordinates
    )
    if rotation is not None:
        coordinates = (coordinates - translation) @ rotation
    # The first two coordinates are exact manufactured material labels in
    # [0, 1].  An inverse rigid transform can move an endpoint a few ulps
    # outside that closed interval; feeding 1+eps to tan(pi*x/2) selects
    # the wrong analytic branch.  Fail on a material-label mismatch and
    # remove roundoff only.  This guard-only operation is not runtime
    # coordinate inference or a state cleanup.
    material = coordinates[:, :2]
    tolerance = 2.0e-12
    if (
        float(np.min(material)) < -tolerance
        or float(np.max(material)) > 1.0 + tolerance
    ):
        raise ValueError(
            "inverse rigid transform changed manufactured material labels"
        )
    coordinates = coordinates.copy()
    coordinates[:, :2] = np.clip(material, 0.0, 1.0)
    return coordinates


def _boundary_indices(coordinates: np.ndarray) -> np.ndarray:
    return np.flatnonzero(
        np.isclose(coordinates[:, 0], 0.0)
        | np.isclose(coordinates[:, 0], 1.0)
        | np.isclose(coordinates[:, 1], 0.0)
        | np.isclose(coordinates[:, 1], 1.0)
    )


def _integrate(
    contract: dict,
    *,
    case_name: str,
    steps: int,
    constrained: bool,
    rotation: np.ndarray | None = None,
    translation: np.ndarray | None = None,
):
    canonical = contract["canonical"]
    start = float(canonical["time"]["start"])
    end = float(canonical["time"]["end"])
    dt = (end - start) / steps
    rates = canonical["geometry_and_mesh"]
    del rates
    base_rates = yaml.safe_load(
        (
            HERE / "docs" / "diag"
            / "moving_p2_wake_transport_cases.yaml"
        ).read_text(encoding="utf-8")
    )["canonical"]["relative_material_velocity"]["parameter_rates"]
    coefficient_first = float(base_rates["a"])
    coefficient_second = float(base_rates["b"])
    _, patch0, _ = _operator(
        start,
        yaml.safe_load(
            (
                HERE / "docs" / "diag"
                / "moving_p2_wake_transport_cases.yaml"
            ).read_text(encoding="utf-8")
        ),
        rotation=rotation,
        translation=translation,
    )
    coordinates = _base_coordinates(
        patch0,
        rotation=rotation,
        translation=translation,
    )
    boundary = _boundary_indices(coordinates)
    trace = P2EssentialScalarTrace(
        boundary,
        f"{case_name}-explicit-material-ids",
    )
    state, _ = _case_state(
        coordinates[:, 0],
        coordinates[:, 1],
        start,
        case_name=case_name,
        coefficient_first=coefficient_first,
        coefficient_second=coefficient_second,
    )
    maximum_trace_error = 0.0
    maximum_rank_deficiency = 0
    partition_failures = 0
    moving_contract = yaml.safe_load(
        (
            HERE / "docs" / "diag"
            / "moving_p2_wake_transport_cases.yaml"
        ).read_text(encoding="utf-8")
    )
    last_patch = patch0
    for step in range(steps):
        time = start + step * dt
        _, patch, _ = _operator(
            time,
            moving_contract,
            rotation=rotation,
            translation=translation,
        )
        _, next_patch, _ = _operator(
            time + dt,
            moving_contract,
            rotation=rotation,
            translation=translation,
        )
        coordinates = _base_coordinates(
            patch,
            rotation=rotation,
            translation=translation,
        )
        next_coordinates = _base_coordinates(
            next_patch,
            rotation=rotation,
            translation=translation,
        )
        value0, rate0_boundary = _case_state(
            coordinates[boundary, 0],
            coordinates[boundary, 1],
            time,
            case_name=case_name,
            coefficient_first=coefficient_first,
            coefficient_second=coefficient_second,
        )
        value1, rate1_boundary = _case_state(
            next_coordinates[boundary, 0],
            next_coordinates[boundary, 1],
            time + dt,
            case_name=case_name,
            coefficient_first=coefficient_first,
            coefficient_second=coefficient_second,
        )
        state[boundary] = value0
        if constrained:
            first_rate = p2_essential_trace_transport_rate(
                patch.operator,
                state,
                essential_trace=trace,
                prescribed_value=value0,
                prescribed_time_derivative=rate0_boundary,
            )
            predictor = state + dt * first_rate.rate
            predictor[boundary] = value1
            second_rate = p2_essential_trace_transport_rate(
                next_patch.operator,
                predictor,
                essential_trace=trace,
                prescribed_value=value1,
                prescribed_time_derivative=rate1_boundary,
            )
            state = state + 0.5 * dt * (
                first_rate.rate + second_rate.rate
            )
            state[boundary] = value1
            maximum_rank_deficiency = max(
                maximum_rank_deficiency,
                len(first_rate.free_dof_indices)
                - first_rate.free_mass_rank,
                len(second_rate.free_dof_indices)
                - second_rate.free_mass_rank,
            )
            free = first_rate.free_dof_indices
            if (
                len(np.intersect1d(free, boundary)) != 0
                or len(free) + len(boundary) != len(state)
            ):
                partition_failures += 1
        else:
            first_rate = patch.rate(state)
            predictor = state + dt * first_rate
            predictor[boundary] = value1
            second_rate = next_patch.rate(predictor)
            state = state + 0.5 * dt * (
                first_rate + second_rate
            )
            state[boundary] = value1
            free = trace.free_indices(len(state))
        maximum_trace_error = max(
            maximum_trace_error,
            float(
                np.max(
                    np.abs(state[boundary] - value1),
                    initial=0.0,
                )
            ),
        )
        last_patch = next_patch
    coordinates = _base_coordinates(
        last_patch,
        rotation=rotation,
        translation=translation,
    )
    exact, _ = _case_state(
        coordinates[:, 0],
        coordinates[:, 1],
        end,
        case_name=case_name,
        coefficient_first=coefficient_first,
        coefficient_second=coefficient_second,
    )
    trace_jump, _ = _trace_and_boundary(
        last_patch,
        state,
        rotation=rotation,
        translation=translation,
    )
    return {
        "state": state,
        "exact": exact,
        "relative_l2_error": _relative_mass_error(
            last_patch.operator,
            state,
            exact,
        ),
        "trace_error": maximum_trace_error,
        "shared_trace_jump": trace_jump,
        "rank_deficiency": maximum_rank_deficiency,
        "partition_failures": partition_failures,
        "free": free,
        "final_patch": last_patch,
    }


def _time_family(contract: dict, case_name: str):
    steps = [
        int(value)
        for value in contract["canonical"]["time"]["step_families"]
    ]
    results = [
        _integrate(
            contract,
            case_name=case_name,
            steps=value,
            constrained=True,
        )
        for value in steps
    ]
    coarse = float(
        np.max(
            np.abs(results[1]["state"] - results[0]["state"]),
            initial=0.0,
        )
    )
    fine = float(
        np.max(
            np.abs(results[2]["state"] - results[1]["state"]),
            initial=0.0,
        )
    )
    return results, coarse / max(fine, np.finfo(float).tiny)


def run() -> dict:
    contract = yaml.safe_load(CASES.read_text(encoding="utf-8"))
    thresholds = contract["thresholds"]
    zero_results, zero_ratio = _time_family(
        contract,
        "zero_all_boundaries",
    )
    nonzero_results, nonzero_ratio = _time_family(
        contract,
        "nonzero_body_attachment",
    )
    zero = zero_results[-1]
    nonzero = nonzero_results[-1]
    clamp = _integrate(
        contract,
        case_name="nonzero_body_attachment",
        steps=int(contract["canonical"]["time"]["step_families"][-1]),
        constrained=False,
    )
    clamp_error = float(
        np.max(
            np.abs(
                clamp["state"][nonzero["free"]]
                - nonzero["state"][nonzero["free"]]
            ),
            initial=0.0,
        )
    )
    rigid = yaml.safe_load(
        (
            HERE / "docs" / "diag"
            / "moving_p2_wake_transport_cases.yaml"
        ).read_text(encoding="utf-8")
    )["canonical"]["rigid_frame_counterfactual"]
    rotation = _rotation_matrix(
        np.asarray(rigid["rotation_axis"], dtype=float),
        np.deg2rad(float(rigid["rotation_deg"])),
    )
    translation = np.asarray(rigid["translation"], dtype=float)
    moved = _integrate(
        contract,
        case_name="nonzero_body_attachment",
        steps=int(contract["canonical"]["time"]["step_families"][-1]),
        constrained=True,
        rotation=rotation,
        translation=translation,
    )
    rigid_error = float(
        np.max(
            np.abs(moved["state"] - nonzero["state"]),
            initial=0.0,
        )
    )
    invalid_failures = 0
    for invalid in (
        np.array((0, 0)),
        np.array((-1, 2)),
    ):
        try:
            P2EssentialScalarTrace(invalid, "invalid")
        except DistributedDoubletError:
            invalid_failures += 1
    try:
        invalid = P2EssentialScalarTrace(
            np.array((len(nonzero["state"]) + 1,)),
            "out-of-range",
        )
        invalid.free_indices(len(nonzero["state"]))
    except DistributedDoubletError:
        invalid_failures += 1

    trace_error = max(
        item["trace_error"]
        for item in zero_results + nonzero_results + [moved]
    )
    trace_jump = max(
        item["shared_trace_jump"]
        for item in zero_results + nonzero_results + [moved]
    )
    rank_deficiency = max(
        item["rank_deficiency"]
        for item in zero_results + nonzero_results + [moved]
    )
    partition_failures = sum(
        item["partition_failures"]
        for item in zero_results + nonzero_results + [moved]
    )
    checks = {
        "typed_partition_is_complete": (
            partition_failures
            <= int(thresholds["partition_failure_count_max"])
        ),
        "free_mass_blocks_are_full_rank": (
            rank_deficiency
            <= int(thresholds["free_mass_rank_deficiency_max"])
        ),
        "prescribed_stage_traces_are_exact": (
            trace_error
            <= float(thresholds["prescribed_trace_abs_max"])
        ),
        "shared_patch_traces_are_exact": (
            trace_jump
            <= float(thresholds["shared_trace_jump_abs_max"])
        ),
        "zero_case_time_Cauchy_passes": (
            zero_ratio
            >= float(
                thresholds["zero_case_time_cauchy_ratio_min"]
            )
        ),
        "nonzero_case_time_Cauchy_passes": (
            nonzero_ratio
            >= float(
                thresholds["nonzero_case_time_cauchy_ratio_min"]
            )
        ),
        "zero_case_accuracy_passes": (
            zero["relative_l2_error"]
            <= float(
                thresholds["zero_case_finest_relative_l2_error_max"]
            )
        ),
        "nonzero_case_accuracy_passes": (
            nonzero["relative_l2_error"]
            <= float(
                thresholds[
                    "nonzero_case_finest_relative_l2_error_max"
                ]
            )
        ),
        "rigid_scalar_is_objective": (
            rigid_error
            <= float(
                thresholds["rigid_final_scalar_abs_difference_max"]
            )
        ),
        "clamp_only_counterfactual_is_visible": (
            clamp_error
            >= float(
                thresholds[
                    "clamp_only_free_interior_abs_error_min"
                ]
            )
        ),
        "invalid_constraints_fail_closed": (
            invalid_failures
            >= int(
                thresholds["invalid_constraint_failure_count_min"]
            )
        ),
    }
    result = {
        "artifact": "constrained_p2_wake_transport_oracle",
        "stage": contract["stage"],
        "claim_node": contract["claim_node"],
        "stage_decision": (
            "GO" if all(checks.values()) else "NO-GO"
        ),
        "checks": checks,
        "aggregate_metrics": {
            "partition_failure_count": partition_failures,
            "free_mass_rank_deficiency_max": rank_deficiency,
            "prescribed_trace_abs_max": trace_error,
            "shared_trace_jump_abs_max": trace_jump,
            "zero_case_time_cauchy_ratio": zero_ratio,
            "nonzero_case_time_cauchy_ratio": nonzero_ratio,
            "zero_case_finest_relative_l2_error": (
                zero["relative_l2_error"]
            ),
            "nonzero_case_finest_relative_l2_error": (
                nonzero["relative_l2_error"]
            ),
            "rigid_final_scalar_abs_difference": rigid_error,
            "clamp_only_free_interior_abs_error": clamp_error,
            "invalid_constraint_failure_count": invalid_failures,
        },
        "forbidden_quantities_absent": [
            "coordinate_runtime_inference",
            "post_hoc_cleanup",
            "mass_lumping",
            "upwind",
            "artificial_diffusion",
            "actual_induced_velocity",
            "pressure",
            "force",
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
