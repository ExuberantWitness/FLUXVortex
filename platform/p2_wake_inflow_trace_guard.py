"""Run the preregistered S3m typed inflow/outflow P2 transport gate."""
from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from claim_runtime.distributed_doublet import DistributedDoubletError  # noqa: E402
from claim_runtime.p2_surface_material_transport import (  # noqa: E402
    P2TransportBoundaryRoles,
    assemble_p2_patch_material_transport,
    p2_essential_trace_transport_rate,
)
from moving_p2_wake_transport_guard import (  # noqa: E402
    _patch_assembly,
    _relative_mass_error,
    _rotation_matrix,
    _trace_and_boundary,
)


CASES = HERE / "docs" / "diag" / "p2_wake_inflow_trace_cases.yaml"
MOVING_CASES = (
    HERE / "docs" / "diag" / "moving_p2_wake_transport_cases.yaml"
)
RESULTS = (
    HERE / "docs" / "diag" / "p2_wake_inflow_trace_results.json"
)


def _material_parameters(
    patch,
    *,
    rotation: np.ndarray | None,
    translation: np.ndarray | None,
) -> np.ndarray:
    coordinates = (
        patch.operator.topology.degree_of_freedom_coordinates.copy()
    )
    if rotation is not None:
        coordinates = (coordinates - translation) @ rotation
    parameters = coordinates[:, :2]
    tolerance = 2.0e-12
    if (
        float(np.min(parameters)) < -tolerance
        or float(np.max(parameters)) > 1.0 + tolerance
    ):
        raise DistributedDoubletError(
            "manufactured material parameters left their closed domain"
        )
    return np.clip(parameters, 0.0, 1.0)


def _typed_roles(
    patch,
    *,
    rotation: np.ndarray | None,
    translation: np.ndarray | None,
) -> tuple[P2TransportBoundaryRoles, np.ndarray]:
    parameters = _material_parameters(
        patch,
        rotation=rotation,
        translation=translation,
    )
    first = parameters[:, 0]
    second = parameters[:, 1]
    body_mask = np.isclose(first, 1.0)
    old_mask = np.isclose(first, 0.0)
    lower_mask = (
        np.isclose(second, 0.0) & ~body_mask & ~old_mask
    )
    upper_mask = (
        np.isclose(second, 1.0) & ~body_mask & ~old_mask
    )
    boundary = np.flatnonzero(
        body_mask | old_mask | lower_mask | upper_mask
    )
    roles = P2TransportBoundaryRoles(
        body_inflow_dof_indices=np.flatnonzero(body_mask),
        old_outflow_dof_indices=np.flatnonzero(old_mask),
        lower_characteristic_dof_indices=np.flatnonzero(lower_mask),
        upper_characteristic_dof_indices=np.flatnonzero(upper_mask),
        role_id="manufactured-open-wake",
    )
    roles.validate(
        patch.operator.topology.degree_of_freedom_count,
        declared_boundary_dof_indices=boundary,
    )
    return roles, boundary


def _operator(
    time: float,
    contract: dict[str, Any],
    moving: dict[str, Any],
    *,
    rotation: np.ndarray | None = None,
    translation: np.ndarray | None = None,
):
    geometry = moving["canonical"]["geometry"]
    cells = contract["canonical"]["geometry_and_mesh"]
    rates = contract["canonical"]["relative_material_velocity"][
        "parameter_rates"
    ]
    epsilon = float(geometry["epsilon"])
    omega = float(geometry["omega"])
    first_rate = float(rates["ds_dt"])
    second_rate = float(rates["dr_dt"])
    assembly = _patch_assembly(
        time,
        streamwise_cells=int(
            cells["streamwise_chronological_strips"]
        ),
        spanwise_cells=int(cells["spanwise_cells"]),
        epsilon=epsilon,
        omega=omega,
        rotation=rotation,
        translation=translation,
    )

    def relative_velocity(points: np.ndarray) -> np.ndarray:
        if rotation is None:
            base = points
        else:
            base = (points - translation) @ rotation
        first = base[:, 0]
        second = base[:, 1]
        slope_first = (
            epsilon
            * np.pi
            * np.cos(np.pi * first)
            * np.sin(np.pi * second)
            * np.sin(omega * time)
        )
        slope_second = (
            epsilon
            * np.pi
            * np.sin(np.pi * first)
            * np.cos(np.pi * second)
            * np.sin(omega * time)
        )
        value = np.column_stack(
            (
                np.full_like(first, first_rate),
                np.full_like(second, second_rate),
                slope_first * first_rate
                + slope_second * second_rate,
            )
        )
        return value if rotation is None else value @ rotation.T

    return assemble_p2_patch_material_transport(
        assembly,
        relative_velocity_provider=relative_velocity,
        quadrature_order=int(contract["canonical"]["quadrature_order"]),
    )


def _trace_shape(second: np.ndarray) -> np.ndarray:
    return second * (1.0 - second)


def _block_terms(
    operator,
    state: np.ndarray,
    rate: np.ndarray,
    roles: P2TransportBoundaryRoles,
    prescribed_value: np.ndarray,
    prescribed_rate: np.ndarray,
) -> tuple[float, float]:
    body = roles.body_inflow_dof_indices
    free = roles.body_essential_trace().free_indices(len(state))
    mass = operator.mass_matrix
    advection = operator.advection_matrix
    terms = (
        mass[np.ix_(free, free)] @ rate[free],
        mass[np.ix_(free, body)] @ prescribed_rate,
        advection[np.ix_(free, free)] @ state[free],
        advection[np.ix_(free, body)] @ prescribed_value,
    )
    residual = sum(terms)
    scale = max(
        (
            float(np.max(np.abs(term), initial=0.0))
            for term in terms
        ),
        default=0.0,
    )
    normalized = float(
        np.max(np.abs(residual), initial=0.0)
        / max(scale, np.finfo(float).tiny)
    )
    return normalized, scale


def _algebraic_probe(contract: dict[str, Any], moving: dict[str, Any]):
    patch = _operator(0.0, contract, moving)
    roles, _ = _typed_roles(
        patch,
        rotation=None,
        translation=None,
    )
    operator = patch.operator
    count = operator.topology.degree_of_freedom_count
    body = roles.body_inflow_dof_indices
    parameters = _material_parameters(
        patch,
        rotation=None,
        translation=None,
    )
    shape = _trace_shape(parameters[body, 1])
    trace = roles.body_essential_trace()

    zero_state = np.zeros(count)
    zero_value = np.zeros(len(body))
    rate_injection = p2_essential_trace_transport_rate(
        operator,
        zero_state,
        essential_trace=trace,
        prescribed_value=zero_value,
        prescribed_time_derivative=shape,
    )
    rate_residual, rate_scale = _block_terms(
        operator,
        zero_state,
        rate_injection.rate,
        roles,
        zero_value,
        shape,
    )
    clamp_rate = operator.rate(zero_state)
    clamp_rate[body] = shape
    clamp_residual, _ = _block_terms(
        operator,
        zero_state,
        clamp_rate,
        roles,
        zero_value,
        shape,
    )

    value_state = np.zeros(count)
    value_state[body] = shape
    value_rate = p2_essential_trace_transport_rate(
        operator,
        value_state,
        essential_trace=trace,
        prescribed_value=shape,
        prescribed_time_derivative=np.zeros(len(body)),
    )
    value_residual, value_scale = _block_terms(
        operator,
        value_state,
        value_rate.rate,
        roles,
        shape,
        np.zeros(len(body)),
    )
    return {
        "rate_normalized_residual": rate_residual,
        "value_normalized_residual": value_residual,
        "clamp_rate_normalized_residual": clamp_residual,
        "rate_injection_free_response": float(
            np.max(
                np.abs(
                    rate_injection.rate[
                        rate_injection.free_dof_indices
                    ]
                ),
                initial=0.0,
            )
        ),
        "rate_boundary_coupling_scale": rate_scale,
        "value_boundary_coupling_scale": value_scale,
        "free_mass_rank_deficiency": (
            len(rate_injection.free_dof_indices)
            - rate_injection.free_mass_rank
        ),
        "roles": roles,
        "patch": patch,
    }


def _exact_solution(
    parameters: np.ndarray,
    time: float,
    *,
    advection_speed: float,
) -> np.ndarray:
    delay = (1.0 - parameters[:, 0]) / advection_speed
    age = np.maximum(time - delay, 0.0)
    return age * age * _trace_shape(parameters[:, 1])


def _integrate(
    contract: dict[str, Any],
    moving: dict[str, Any],
    *,
    steps: int,
    constrained: bool,
    rotation: np.ndarray | None = None,
    translation: np.ndarray | None = None,
) -> dict[str, Any]:
    propagation = contract["canonical"]["propagation_case"]
    start = float(propagation["start"])
    end = float(propagation["end"])
    dt = (end - start) / steps
    patch0 = _operator(
        start,
        contract,
        moving,
        rotation=rotation,
        translation=translation,
    )
    roles, declared_boundary = _typed_roles(
        patch0,
        rotation=rotation,
        translation=translation,
    )
    trace = roles.body_essential_trace()
    body = roles.body_inflow_dof_indices
    state = np.zeros(
        patch0.operator.topology.degree_of_freedom_count
    )
    maximum_trace_error = 0.0
    maximum_rank_deficiency = 0
    partition_failures = 0
    last_patch = patch0
    for step in range(steps):
        time = start + step * dt
        patch = _operator(
            time,
            contract,
            moving,
            rotation=rotation,
            translation=translation,
        )
        next_patch = _operator(
            time + dt,
            contract,
            moving,
            rotation=rotation,
            translation=translation,
        )
        try:
            roles.validate(
                patch.operator.topology.degree_of_freedom_count,
                declared_boundary_dof_indices=declared_boundary,
            )
        except DistributedDoubletError:
            partition_failures += 1
        parameters = _material_parameters(
            patch,
            rotation=rotation,
            translation=translation,
        )
        next_parameters = _material_parameters(
            next_patch,
            rotation=rotation,
            translation=translation,
        )
        shape0 = _trace_shape(parameters[body, 1])
        shape1 = _trace_shape(next_parameters[body, 1])
        value0 = time * time * shape0
        derivative0 = 2.0 * time * shape0
        value1 = (time + dt) ** 2 * shape1
        derivative1 = 2.0 * (time + dt) * shape1
        state[body] = value0
        if constrained:
            first = p2_essential_trace_transport_rate(
                patch.operator,
                state,
                essential_trace=trace,
                prescribed_value=value0,
                prescribed_time_derivative=derivative0,
            )
            predictor = state + dt * first.rate
            predictor[body] = value1
            second = p2_essential_trace_transport_rate(
                next_patch.operator,
                predictor,
                essential_trace=trace,
                prescribed_value=value1,
                prescribed_time_derivative=derivative1,
            )
            state = state + 0.5 * dt * (
                first.rate + second.rate
            )
            maximum_rank_deficiency = max(
                maximum_rank_deficiency,
                len(first.free_dof_indices) - first.free_mass_rank,
                len(second.free_dof_indices) - second.free_mass_rank,
            )
        else:
            first_rate = patch.rate(state)
            predictor = state + dt * first_rate
            predictor[body] = value1
            second_rate = next_patch.rate(predictor)
            state = state + 0.5 * dt * (
                first_rate + second_rate
            )
        state[body] = value1
        maximum_trace_error = max(
            maximum_trace_error,
            float(
                np.max(
                    np.abs(state[body] - value1),
                    initial=0.0,
                )
            ),
        )
        last_patch = next_patch
    parameters = _material_parameters(
        last_patch,
        rotation=rotation,
        translation=translation,
    )
    speed = abs(
        float(
            contract["canonical"]["relative_material_velocity"][
                "parameter_rates"
            ]["ds_dt"]
        )
    )
    exact = _exact_solution(
        parameters,
        end,
        advection_speed=speed,
    )
    shared_trace, _ = _trace_and_boundary(
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
        "shared_trace_jump": shared_trace,
        "rank_deficiency": maximum_rank_deficiency,
        "partition_failures": partition_failures,
        "roles": roles,
        "final_patch": last_patch,
    }


def _time_family(contract: dict[str, Any], moving: dict[str, Any]):
    steps = [
        int(value)
        for value in contract["canonical"]["propagation_case"][
            "step_families"
        ]
    ]
    results = [
        _integrate(
            contract,
            moving,
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


def _invalid_role_failures(
    roles: P2TransportBoundaryRoles,
    dof_count: int,
) -> int:
    failures = 0
    try:
        P2TransportBoundaryRoles(
            roles.body_inflow_dof_indices,
            roles.body_inflow_dof_indices,
            roles.lower_characteristic_dof_indices,
            roles.upper_characteristic_dof_indices,
            "overlap",
        )
    except DistributedDoubletError:
        failures += 1
    try:
        roles.validate(
            dof_count,
            declared_boundary_dof_indices=(
                roles.all_boundary_dof_indices[:-1]
            ),
        )
    except DistributedDoubletError:
        failures += 1
    try:
        invalid = P2TransportBoundaryRoles(
            np.array((dof_count + 1,)),
            roles.old_outflow_dof_indices,
            roles.lower_characteristic_dof_indices,
            roles.upper_characteristic_dof_indices,
            "out-of-range",
        )
        invalid.validate(
            dof_count,
            declared_boundary_dof_indices=(
                invalid.all_boundary_dof_indices
            ),
        )
    except DistributedDoubletError:
        failures += 1
    return failures


def run() -> dict[str, Any]:
    contract = yaml.safe_load(CASES.read_text(encoding="utf-8"))
    moving = yaml.safe_load(MOVING_CASES.read_text(encoding="utf-8"))
    thresholds = contract["thresholds"]
    algebraic = _algebraic_probe(contract, moving)
    integrations, time_ratio = _time_family(contract, moving)
    finest = integrations[-1]
    steps = int(
        contract["canonical"]["propagation_case"]["step_families"][-1]
    )
    clamp = _integrate(
        contract,
        moving,
        steps=steps,
        constrained=False,
    )
    free = finest["roles"].body_essential_trace().free_indices(
        len(finest["state"])
    )
    clamp_error = float(
        np.max(
            np.abs(
                clamp["state"][free] - finest["state"][free]
            ),
            initial=0.0,
        )
    )

    rigid = moving["canonical"]["rigid_frame_counterfactual"]
    rotation = _rotation_matrix(
        np.asarray(rigid["rotation_axis"], dtype=float),
        np.deg2rad(float(rigid["rotation_deg"])),
    )
    translation = np.asarray(rigid["translation"], dtype=float)
    moved = _integrate(
        contract,
        moving,
        steps=steps,
        constrained=True,
        rotation=rotation,
        translation=translation,
    )
    rigid_error = float(
        np.max(
            np.abs(moved["state"] - finest["state"]),
            initial=0.0,
        )
    )

    rates = contract["canonical"]["relative_material_velocity"][
        "parameter_rates"
    ]
    signed = contract["canonical"]["relative_material_velocity"]
    body_flux_error = abs(
        float(rates["ds_dt"])
        - float(signed["body_edge_signed_flux"])
    )
    old_flux_error = abs(
        -float(rates["ds_dt"])
        - float(signed["old_edge_signed_flux"])
    )
    tip_flux = abs(float(rates["dr_dt"]))
    roles = finest["roles"]
    constrained_nonbody = len(
        np.intersect1d(
            roles.body_essential_trace().global_dof_indices,
            np.concatenate(
                (
                    roles.old_outflow_dof_indices,
                    roles.lower_characteristic_dof_indices,
                    roles.upper_characteristic_dof_indices,
                )
            ),
        )
    )
    partition_failures = sum(
        item["partition_failures"]
        for item in integrations + [moved]
    )
    trace_error = max(
        item["trace_error"]
        for item in integrations + [moved]
    )
    shared_trace = max(
        item["shared_trace_jump"]
        for item in integrations + [moved]
    )
    rank_deficiency = max(
        [
            algebraic["free_mass_rank_deficiency"],
            *[
                item["rank_deficiency"]
                for item in integrations + [moved]
            ],
        ]
    )
    invalid_failures = _invalid_role_failures(
        roles,
        len(finest["state"]),
    )
    correct_residual = max(
        algebraic["rate_normalized_residual"],
        algebraic["value_normalized_residual"],
    )
    checks = {
        "typed_boundary_roles_partition": (
            partition_failures
            <= int(
                thresholds[
                    "boundary_role_partition_failure_count_max"
                ]
            )
        ),
        "only_body_inflow_is_constrained": (
            constrained_nonbody
            <= int(thresholds["constrained_nonbody_dof_count_max"])
        ),
        "boundary_flux_roles_are_correct": (
            body_flux_error
            <= float(thresholds["body_flux_abs_error_max"])
            and old_flux_error
            <= float(thresholds["old_flux_abs_error_max"])
            and tip_flux <= float(thresholds["tip_flux_abs_max"])
        ),
        "free_mass_blocks_are_full_rank": (
            rank_deficiency
            <= int(thresholds["free_mass_rank_deficiency_max"])
        ),
        "correct_block_residual_is_exact": (
            correct_residual
            <= float(
                thresholds[
                    "correct_algebraic_normalized_residual_max"
                ]
            )
        ),
        "rate_injection_reaches_free_state": (
            algebraic["rate_injection_free_response"]
            >= float(
                thresholds["rate_injection_free_response_abs_min"]
            )
        ),
        "clamp_rate_injection_is_rejected": (
            algebraic["clamp_rate_normalized_residual"]
            >= float(
                thresholds[
                    "clamp_rate_injection_normalized_residual_min"
                ]
            )
        ),
        "prescribed_body_trace_is_exact": (
            trace_error
            <= float(thresholds["prescribed_trace_abs_max"])
        ),
        "shared_patch_trace_is_exact": (
            shared_trace
            <= float(
                thresholds["shared_patch_trace_jump_abs_max"]
            )
        ),
        "propagation_time_converges": (
            time_ratio
            >= float(
                thresholds["propagation_time_cauchy_ratio_min"]
            )
        ),
        "propagation_matches_inflow_solution": (
            finest["relative_l2_error"]
            <= float(
                thresholds[
                    "propagation_finest_relative_l2_error_max"
                ]
            )
        ),
        "clamp_only_final_state_is_rejected": (
            clamp_error
            >= float(
                thresholds[
                    "clamp_only_free_interior_abs_error_min"
                ]
            )
        ),
        "rigid_scalar_is_objective": (
            rigid_error
            <= float(
                thresholds["rigid_final_scalar_abs_difference_max"]
            )
        ),
        "invalid_roles_fail_closed": (
            invalid_failures
            >= int(thresholds["invalid_role_failure_count_min"])
        ),
    }
    result = {
        "artifact": "p2_wake_inflow_trace_oracle",
        "stage": contract["stage"],
        "claim_node": contract["claim_node"],
        "stage_decision": (
            "GO" if all(checks.values()) else "NO-GO"
        ),
        "checks": checks,
        "aggregate_metrics": {
            "boundary_role_partition_failure_count": (
                partition_failures
            ),
            "constrained_nonbody_dof_count": constrained_nonbody,
            "body_flux_abs_error": body_flux_error,
            "old_flux_abs_error": old_flux_error,
            "tip_flux_abs_max": tip_flux,
            "free_mass_rank_deficiency_max": rank_deficiency,
            "correct_algebraic_normalized_residual_max": (
                correct_residual
            ),
            "rate_injection_free_response_abs_max": (
                algebraic["rate_injection_free_response"]
            ),
            "rate_boundary_coupling_scale": (
                algebraic["rate_boundary_coupling_scale"]
            ),
            "value_boundary_coupling_scale": (
                algebraic["value_boundary_coupling_scale"]
            ),
            "clamp_rate_injection_normalized_residual": (
                algebraic["clamp_rate_normalized_residual"]
            ),
            "prescribed_trace_abs_max": trace_error,
            "shared_patch_trace_jump_abs_max": shared_trace,
            "propagation_time_cauchy_ratio": time_ratio,
            "propagation_finest_relative_l2_error": (
                finest["relative_l2_error"]
            ),
            "clamp_only_free_interior_abs_error": clamp_error,
            "rigid_final_scalar_abs_difference": rigid_error,
            "invalid_role_failure_count": invalid_failures,
            "body_inflow_dof_count": len(
                roles.body_inflow_dof_indices
            ),
            "old_outflow_dof_count": len(
                roles.old_outflow_dof_indices
            ),
            "characteristic_tip_dof_count": (
                len(roles.lower_characteristic_dof_indices)
                + len(roles.upper_characteristic_dof_indices)
            ),
        },
        "forbidden_quantities_absent": [
            "coordinate_runtime_inference",
            "post_hoc_threshold_change",
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
