"""Run the preregistered S3ag Kutta compatibility counterexample oracle."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import yaml


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from actual_boundary_3d_cut_wake_junction_guard import (  # noqa: E402
    build_canonical_diamond_wing,
)
from claim_runtime.actual_boundary_body_wake import (  # noqa: E402
    solve_actual_boundary_body_wake_p2,
)
from claim_runtime.actual_wake_kutta_closure_roles import (  # noqa: E402
    cut_role_operators,
    independent_wake_system,
)
from claim_runtime.actual_wake_kutta_compatibility import (  # noqa: E402
    build_actual_pressure_kutta_model,
    evaluate_pressure_kutta,
    solve_independent_wake_body_state,
    solve_pressure_kutta_newton,
)
from claim_runtime.classified_p2_cut_topology import (  # noqa: E402
    classified_p2_cut_topology,
)
from claim_runtime.finite_angle_sheet_formation import (  # noqa: E402
    finite_angle_sheet_formation,
)
from claim_runtime.material_birth_flux import (  # noqa: E402
    consistent_p2_line_mass,
)


CASES = (
    HERE
    / "docs"
    / "diag"
    / "actual_wake_kutta_compatibility_cases.yaml"
)
RESULTS = (
    HERE
    / "docs"
    / "diag"
    / "actual_wake_kutta_compatibility_results.json"
)


def _maximum_abs(value) -> float:
    return float(np.max(np.abs(value), initial=0.0))


def _log_order(scales, values) -> float:
    scale = np.asarray(scales, dtype=float)
    magnitude = np.maximum(
        np.asarray(values, dtype=float),
        np.finfo(float).tiny,
    )
    return float(np.polyfit(np.log(scale), np.log(magnitude), 1)[0])


def _normalized_full_bie_residual(state, system) -> float:
    scale = max(
        float(np.linalg.norm(system.right_hand_side)),
        np.finfo(float).tiny,
    )
    return float(np.linalg.norm(state.full_bie_residual) / scale)


def _full_bie_backward_error(state, system) -> float:
    body_term = system.body_matrix @ state.body_potential
    wake_term = system.independent_wake_matrix @ state.wake_jump
    right_hand_side = system.right_hand_side
    scale = max(
        float(np.linalg.norm(body_term))
        + float(np.linalg.norm(wake_term))
        + float(np.linalg.norm(right_hand_side)),
        np.finfo(float).tiny,
    )
    return float(np.linalg.norm(state.full_bie_residual) / scale)


def _directional_jacobian_error(model, wake_jump, observation_map) -> float:
    wake = np.asarray(wake_jump, dtype=float)
    direction = np.linspace(-1.0, 1.0, len(wake))
    direction /= np.linalg.norm(direction)
    step = 1.0e-6
    center = evaluate_pressure_kutta(
        model,
        wake,
        observation_map=observation_map,
    )
    plus = evaluate_pressure_kutta(
        model,
        wake + step * direction,
        observation_map=observation_map,
    )
    minus = evaluate_pressure_kutta(
        model,
        wake - step * direction,
        observation_map=observation_map,
    )
    finite_difference = (plus.residual - minus.residual) / (2.0 * step)
    analytic = center.jacobian @ direction
    return _maximum_abs(finite_difference - analytic)


def run() -> dict:
    contract = yaml.safe_load(CASES.read_text(encoding="utf-8"))
    canonical = contract["canonical"]
    thresholds = contract["thresholds"]
    (
        mesh,
        upper_faces,
        lower_faces,
        cut_edges,
        zero_endpoints,
    ) = build_canonical_diamond_wing()
    topology = classified_p2_cut_topology(
        mesh,
        upper_face_indices=upper_faces,
        lower_face_indices=lower_faces,
        cut_edges=cut_edges,
        zero_jump_end_vertices=zero_endpoints,
    )
    speed = float(canonical["freestream"]["speed"])
    alpha = np.deg2rad(
        float(canonical["freestream"]["alpha_deg"])
    )
    incident = np.repeat(
        np.array(
            (
                (
                    speed * np.cos(alpha),
                    0.0,
                    speed * np.sin(alpha),
                ),
            )
        ),
        len(mesh.faces),
        axis=0,
    )
    boundary = canonical["actual_boundary"]
    target_order = int(boundary["target_quadrature_order"])
    source_order = int(boundary["source_quadrature_order"])
    downstream = float(boundary["downstream_edge_x"])
    pressure_specification = canonical["pressure_counterexample"]
    line_order = int(pressure_specification["line_quadrature_order"])

    base_solution = solve_actual_boundary_body_wake_p2(
        mesh,
        topology,
        incident_velocity=incident,
        downstream_edge_x=downstream,
        target_quadrature_order=target_order,
        source_quadrature_order=source_order,
    )
    immutable_snapshots = {
        "mesh_vertices": mesh.vertices.copy(),
        "mesh_faces": mesh.faces.copy(),
        "upper_cut_dofs": topology.upper_cut_dofs.copy(),
        "lower_cut_dofs": topology.lower_cut_dofs.copy(),
        "body_matrix": base_solution.body_matrix.copy(),
        "wake_matrix": base_solution.wake_matrix.copy(),
        "right_hand_side": base_solution.right_hand_side.copy(),
        "body_potential": base_solution.global_body_potential.copy(),
        "body_cut_jump": base_solution.body_cut_jump.copy(),
    }
    base_operators = cut_role_operators(topology)
    base_system = independent_wake_system(
        base_solution,
        base_operators,
    )
    base_model = build_actual_pressure_kutta_model(
        base_solution,
        base_system,
        base_operators,
        upper_face_indices=upper_faces,
        lower_face_indices=lower_faces,
        line_quadrature_order=line_order,
    )

    full_old_trace = base_solution.body_cut_jump.copy()
    active_rows = base_operators.active_row_indices
    initial_active_trace = full_old_trace[active_rows]
    initial_trace_scale = max(
        _maximum_abs(initial_active_trace),
        np.finfo(float).tiny,
    )
    span_coordinate = topology.cut_node_coordinates[:, 1]
    span_vertices = topology.cut_node_coordinates[::2, 1]
    line_mass = consistent_p2_line_mass(span_vertices)

    birth_specification = canonical["birth_counterexample"]
    formation = finite_angle_sheet_formation(
        u1_plus=float(birth_specification["u1_plus"]),
        u2_minus=float(birth_specification["u2_minus"]),
        wedge_angle_deg=float(
            birth_specification["wedge_angle_deg"]
        ),
    )
    repository_sign = int(
        birth_specification["repository_birth_flux_sign"]
    )
    prescribed_rate = (
        repository_sign
        * formation.circulation_rate
        * (1.0 - span_coordinate**2)
    )
    timesteps = [
        float(value)
        for value in birth_specification["timestep_family"]
    ]
    birth_records = []
    trace_increments = []
    edge_defects = []
    maximum_birth_bie_residual = 0.0
    maximum_birth_bie_backward_error = 0.0
    maximum_birth_flux_residual = 0.0
    for timestep in timesteps:
        newborn_length = float(
            formation.relative_velocity * timestep
        )
        midpoint_trace = (
            full_old_trace + 0.5 * timestep * prescribed_rate
        )
        current_trace = (
            full_old_trace + timestep * prescribed_rate
        )
        edge_nodes = np.array(
            (1.0, 1.0 + newborn_length, downstream),
            dtype=float,
        )
        old_rows = np.array(
            ((full_old_trace, full_old_trace, full_old_trace),),
            dtype=float,
        )
        active_known_rows = np.vstack(
            (full_old_trace, midpoint_trace)
        )
        unsteady_solution = solve_actual_boundary_body_wake_p2(
            mesh,
            topology,
            incident_velocity=incident,
            downstream_edge_x=downstream,
            wake_edge_x_nodes=edge_nodes,
            fixed_old_wake_rows=old_rows,
            active_known_rows=active_known_rows,
            target_quadrature_order=target_order,
            source_quadrature_order=source_order,
        )
        operators = cut_role_operators(topology)
        system = independent_wake_system(
            unsteady_solution,
            operators,
        )
        model = build_actual_pressure_kutta_model(
            unsteady_solution,
            system,
            operators,
            upper_face_indices=upper_faces,
            lower_face_indices=lower_faces,
            line_quadrature_order=line_order,
        )
        state = solve_independent_wake_body_state(
            model,
            current_trace[operators.active_row_indices],
        )
        bie_residual = _normalized_full_bie_residual(
            state,
            system,
        )
        bie_backward_error = _full_bie_backward_error(
            state,
            system,
        )
        trace_increment = _maximum_abs(
            current_trace - full_old_trace
        )
        edge_defect = _maximum_abs(
            state.edge_compatibility_defect
        )
        flux_residual = _maximum_abs(
            line_mass @ (current_trace - full_old_trace)
            - timestep * line_mass @ prescribed_rate
        )
        maximum_birth_bie_residual = max(
            maximum_birth_bie_residual,
            bie_residual,
        )
        maximum_birth_bie_backward_error = max(
            maximum_birth_bie_backward_error,
            bie_backward_error,
        )
        maximum_birth_flux_residual = max(
            maximum_birth_flux_residual,
            flux_residual,
        )
        trace_increments.append(trace_increment)
        edge_defects.append(edge_defect)
        birth_records.append(
            {
                "timestep": timestep,
                "newborn_length": newborn_length,
                "prescribed_rate_abs_max": _maximum_abs(
                    prescribed_rate
                ),
                "trace_increment_abs_max": trace_increment,
                "circulation_flux_residual_abs_max": flux_residual,
                "normalized_full_bie_residual": bie_residual,
                "full_bie_backward_error": bie_backward_error,
                "edge_compatibility_defect_abs_max": edge_defect,
                "active_wake_jump_abs_max": _maximum_abs(
                    state.wake_jump
                ),
                "body_cut_jump_abs_max": _maximum_abs(
                    operators.active_jump @ state.body_potential
                ),
            }
        )
    trace_order = _log_order(timesteps, trace_increments)
    defect_order = _log_order(timesteps, edge_defects)
    trace_order_error = abs(trace_order - 1.0)
    minimum_birth_edge_defect = min(edge_defects)

    pressure_roots = {}
    maximum_pressure_bie_residual = 0.0
    maximum_pressure_bie_backward_error = 0.0
    maximum_pressure_closure_residual = 0.0
    maximum_pressure_jacobian_error = 0.0
    minimum_pressure_amplification = float("inf")
    minimum_pressure_edge_defect = float("inf")
    for observation_map in pressure_specification[
        "observation_maps"
    ]:
        root = solve_pressure_kutta_newton(
            base_model,
            initial_active_trace,
            observation_map=observation_map,
            residual_tolerance=float(
                pressure_specification["residual_tolerance"]
            ),
            maximum_iterations=int(
                pressure_specification["maximum_iterations"]
            ),
            maximum_backtracking_steps=int(
                pressure_specification["maximum_backtracking_steps"]
            ),
        )
        evaluation = root.evaluation
        closure_residual = _maximum_abs(evaluation.residual)
        bie_residual = _normalized_full_bie_residual(
            evaluation.state,
            base_system,
        )
        bie_backward_error = _full_bie_backward_error(
            evaluation.state,
            base_system,
        )
        jacobian_error = _directional_jacobian_error(
            base_model,
            evaluation.state.wake_jump,
            observation_map,
        )
        wake_amplitude = _maximum_abs(
            evaluation.state.wake_jump
        )
        amplification = wake_amplitude / initial_trace_scale
        edge_defect = _maximum_abs(
            evaluation.state.edge_compatibility_defect
        )
        dense_pressure = _maximum_abs(
            evaluation.dense_pressure_jump
        )
        maximum_pressure_bie_residual = max(
            maximum_pressure_bie_residual,
            bie_residual,
        )
        maximum_pressure_bie_backward_error = max(
            maximum_pressure_bie_backward_error,
            bie_backward_error,
        )
        maximum_pressure_closure_residual = max(
            maximum_pressure_closure_residual,
            closure_residual,
        )
        maximum_pressure_jacobian_error = max(
            maximum_pressure_jacobian_error,
            jacobian_error,
        )
        minimum_pressure_amplification = min(
            minimum_pressure_amplification,
            amplification,
        )
        minimum_pressure_edge_defect = min(
            minimum_pressure_edge_defect,
            edge_defect,
        )
        pressure_roots[observation_map] = {
            "converged": root.converged,
            "iterations": root.iterations,
            "pressure_closure_residual_abs_max": closure_residual,
            "normalized_full_bie_residual": bie_residual,
            "full_bie_backward_error": bie_backward_error,
            "wake_jump": evaluation.state.wake_jump.tolist(),
            "wake_jump_abs_max": wake_amplitude,
            "wake_to_morino_amplification": amplification,
            "edge_compatibility_defect_abs_max": edge_defect,
            "dense_pressure_jump_abs_max": dense_pressure,
            "reduced_jacobian_rank": evaluation.jacobian_rank,
            "reduced_jacobian_condition_number": (
                evaluation.jacobian_condition_number
            ),
            "maximum_jacobian_condition_number": (
                root.maximum_jacobian_condition_number
            ),
            "analytic_jacobian_directional_abs_error": (
                jacobian_error
            ),
            "residual_history": root.residual_history.tolist(),
            "accepted_step_lengths": (
                root.accepted_step_lengths.tolist()
            ),
        }
    observation_maps = list(
        pressure_specification["observation_maps"]
    )
    first_root = np.asarray(
        pressure_roots[observation_maps[0]]["wake_jump"],
        dtype=float,
    )
    second_root = np.asarray(
        pressure_roots[observation_maps[1]]["wake_jump"],
        dtype=float,
    )
    pressure_root_difference = _maximum_abs(
        first_root - second_root
    )
    weak_dense_pressure = pressure_roots[
        "weak_active_p2_line_mass"
    ]["dense_pressure_jump_abs_max"]

    maximum_full_bie_residual = max(
        maximum_birth_bie_residual,
        maximum_pressure_bie_residual,
    )
    input_mutation = max(
        _maximum_abs(mesh.vertices - immutable_snapshots["mesh_vertices"]),
        _maximum_abs(mesh.faces - immutable_snapshots["mesh_faces"]),
        _maximum_abs(
            topology.upper_cut_dofs
            - immutable_snapshots["upper_cut_dofs"]
        ),
        _maximum_abs(
            topology.lower_cut_dofs
            - immutable_snapshots["lower_cut_dofs"]
        ),
        _maximum_abs(
            base_solution.body_matrix
            - immutable_snapshots["body_matrix"]
        ),
        _maximum_abs(
            base_solution.wake_matrix
            - immutable_snapshots["wake_matrix"]
        ),
        _maximum_abs(
            base_solution.right_hand_side
            - immutable_snapshots["right_hand_side"]
        ),
        _maximum_abs(
            base_solution.global_body_potential
            - immutable_snapshots["body_potential"]
        ),
        _maximum_abs(
            base_solution.body_cut_jump
            - immutable_snapshots["body_cut_jump"]
        ),
    )
    input_bitwise_equal = all(
        (
            np.array_equal(mesh.vertices, immutable_snapshots["mesh_vertices"]),
            np.array_equal(mesh.faces, immutable_snapshots["mesh_faces"]),
            np.array_equal(
                topology.upper_cut_dofs,
                immutable_snapshots["upper_cut_dofs"],
            ),
            np.array_equal(
                topology.lower_cut_dofs,
                immutable_snapshots["lower_cut_dofs"],
            ),
            np.array_equal(
                base_solution.body_matrix,
                immutable_snapshots["body_matrix"],
            ),
            np.array_equal(
                base_solution.wake_matrix,
                immutable_snapshots["wake_matrix"],
            ),
            np.array_equal(
                base_solution.right_hand_side,
                immutable_snapshots["right_hand_side"],
            ),
            np.array_equal(
                base_solution.global_body_potential,
                immutable_snapshots["body_potential"],
            ),
            np.array_equal(
                base_solution.body_cut_jump,
                immutable_snapshots["body_cut_jump"],
            ),
        )
    )

    checks = {
        "all_full_body_bie_rows_are_satisfied": (
            maximum_full_bie_residual
            <= float(
                thresholds["normalized_full_bie_residual_max"]
            )
        ),
        "prescribed_birth_flux_is_exact": (
            maximum_birth_flux_residual
            <= float(
                thresholds["normalized_full_bie_residual_max"]
            )
        ),
        "birth_trace_has_first_order_increment": (
            trace_order_error
            <= float(
                thresholds[
                    "birth_trace_dt_order_abs_error_max"
                ]
            )
        ),
        "birth_edge_defect_remains_nonzero": (
            minimum_birth_edge_defect
            >= float(thresholds["birth_edge_defect_abs_min"])
        ),
        "birth_edge_defect_is_order_zero_in_dt": (
            abs(defect_order)
            <= float(
                thresholds[
                    "birth_edge_defect_dt_order_abs_max"
                ]
            )
        ),
        "both_pressure_observations_reach_a_root": (
            all(
                record["converged"]
                for record in pressure_roots.values()
            )
            and maximum_pressure_closure_residual
            <= float(
                thresholds[
                    "pressure_closure_residual_abs_max"
                ]
            )
        ),
        "pressure_roots_are_strongly_amplified": (
            minimum_pressure_amplification
            >= float(
                thresholds[
                    "pressure_root_wake_to_morino_amplification_min"
                ]
            )
        ),
        "pressure_analytic_jacobian_matches_directional_difference": (
            maximum_pressure_jacobian_error
            <= float(
                thresholds[
                    "pressure_jacobian_directional_abs_error_max"
                ]
            )
        ),
        "pressure_roots_leave_large_edge_defects": (
            minimum_pressure_edge_defect
            >= float(
                thresholds["pressure_root_edge_defect_abs_min"]
            )
        ),
        "pressure_observation_changes_the_root": (
            pressure_root_difference
            >= float(
                thresholds[
                    "pressure_observation_root_difference_abs_min"
                ]
            )
        ),
        "weak_root_leaves_dense_pressure_jump": (
            weak_dense_pressure
            >= float(
                thresholds[
                    "weak_root_dense_pressure_jump_abs_min"
                ]
            )
        ),
        "input_state_is_immutable": (
            input_mutation
            <= float(
                thresholds["input_state_mutation_abs_max"]
            )
            and input_bitwise_equal
        ),
    }
    checks = {name: bool(value) for name, value in checks.items()}
    counterexample_go = all(checks.values())
    result = {
        "artifact": "actual_wake_kutta_compatibility_oracle",
        "claim_node": contract["claim_node"],
        "stage": contract["stage"],
        "canonical": canonical,
        "initial_morino_state": {
            "active_wake_jump": initial_active_trace.tolist(),
            "active_wake_jump_abs_max": initial_trace_scale,
            "normalized_full_bie_residual": (
                base_solution.relative_weak_residual
            ),
        },
        "birth_counterexample": {
            "formation": {
                "relative_velocity": formation.relative_velocity,
                "sheet_strength": formation.sheet_strength,
                "circulation_rate": formation.circulation_rate,
                "momentum_residual": formation.momentum_residual,
            },
            "prescribed_span_rate": prescribed_rate.tolist(),
            "trace_increment_dt_order": trace_order,
            "trace_increment_dt_order_abs_error": trace_order_error,
            "edge_compatibility_defect_dt_order": defect_order,
            "edge_compatibility_defect_dt_order_abs": abs(
                defect_order
            ),
            "edge_compatibility_defect_abs_min": (
                minimum_birth_edge_defect
            ),
            "circulation_flux_residual_abs_max": (
                maximum_birth_flux_residual
            ),
            "normalized_full_bie_residual_max": (
                maximum_birth_bie_residual
            ),
            "full_bie_backward_error_max": (
                maximum_birth_bie_backward_error
            ),
            "scales": birth_records,
        },
        "pressure_counterexample": {
            "initial_active_wake_jump_abs_max": (
                initial_trace_scale
            ),
            "roots": pressure_roots,
            "pressure_closure_residual_abs_max": (
                maximum_pressure_closure_residual
            ),
            "normalized_full_bie_residual_max": (
                maximum_pressure_bie_residual
            ),
            "full_bie_backward_error_max": (
                maximum_pressure_bie_backward_error
            ),
            "analytic_jacobian_directional_abs_error_max": (
                maximum_pressure_jacobian_error
            ),
            "wake_to_morino_amplification_min": (
                minimum_pressure_amplification
            ),
            "edge_compatibility_defect_abs_min": (
                minimum_pressure_edge_defect
            ),
            "observation_root_difference_abs_max": (
                pressure_root_difference
            ),
            "weak_root_dense_pressure_jump_abs_max": (
                weak_dense_pressure
            ),
        },
        "aggregate_metrics": {
            "normalized_full_bie_residual_max": (
                maximum_full_bie_residual
            ),
            "input_state_mutation_abs_max": input_mutation,
            "input_state_bitwise_equal": input_bitwise_equal,
        },
        "thresholds": thresholds,
        "checks": checks,
        "counterexample_decision": (
            "GO" if counterexample_go else "NO-GO"
        ),
        "physical_compatibility_decision": (
            "NO-GO" if counterexample_go else "UNRESOLVED"
        ),
        "stage_decision": (
            "COUNTEREXAMPLE-GO / PHYSICAL-COMPATIBILITY-NO-GO"
            if counterexample_go
            else "NO-GO"
        ),
        "production_activation_allowed": False,
        "interpretation": (
            "The prescribed stripwise birth rate and both pressure "
            "observations satisfy every retained algebraic equation. "
            "Nevertheless the birth edge defect has an order-zero dt "
            "plateau, while the two pressure observations select different "
            "strongly amplified roots with large body-to-wake defects. "
            "This falsifies residual-only sufficiency, not the fully coupled "
            "birth or unsteady-pressure Kutta mechanisms."
        ),
    }
    RESULTS.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


if __name__ == "__main__":
    output = run()
    print(json.dumps(output, indent=2, sort_keys=True))
    if output["stage_decision"] != (
        "COUNTEREXAMPLE-GO / PHYSICAL-COMPATIBILITY-NO-GO"
    ):
        raise SystemExit(1)
