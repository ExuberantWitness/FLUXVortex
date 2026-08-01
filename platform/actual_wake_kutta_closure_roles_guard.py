"""Run the preregistered S3af Kutta-closure equation-role oracle."""
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
    consistent_body_surface_mass,
    cut_role_operators,
    independent_wake_system,
    orthonormal_nullspace,
    solve_morino_block,
    solve_quotient_with_attachment,
)
from claim_runtime.classified_p2_cut_topology import (  # noqa: E402
    classified_p2_cut_topology,
)


CASES = (
    HERE
    / "docs"
    / "diag"
    / "actual_wake_kutta_closure_roles_cases.yaml"
)
RESULTS = (
    HERE
    / "docs"
    / "diag"
    / "actual_wake_kutta_closure_roles_results.json"
)


def _maximum_abs(value) -> float:
    return float(np.max(np.abs(value), initial=0.0))


def _wake_coordinate_maps(size: int) -> dict[str, np.ndarray]:
    reverse = np.eye(size)[:, ::-1]
    diagonal = np.diag(np.linspace(0.5, 2.0, size))
    triangular = np.eye(size)
    triangular[np.arange(size - 1), np.arange(1, size)] = 0.2
    return {
        "identity": np.eye(size),
        "reverse_permutation": reverse,
        "positive_diagonal_scaling": diagonal,
        "unit_upper_triangular": triangular,
    }


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
    wake_specification = canonical["material_wake"]
    solution = solve_actual_boundary_body_wake_p2(
        mesh,
        topology,
        incident_velocity=incident,
        downstream_edge_x=float(
            wake_specification["downstream_edge_x"]
        ),
        target_quadrature_order=int(
            wake_specification["target_quadrature_order"]
        ),
        source_quadrature_order=int(
            wake_specification["source_quadrature_order"]
        ),
    )
    immutable_snapshots = {
        "body_matrix": solution.body_matrix.copy(),
        "wake_matrix": solution.wake_matrix.copy(),
        "right_hand_side": solution.right_hand_side.copy(),
        "body_potential": solution.global_body_potential.copy(),
        "upper_cut_dofs": topology.upper_cut_dofs.copy(),
        "lower_cut_dofs": topology.lower_cut_dofs.copy(),
    }

    operators = cut_role_operators(topology)
    system = independent_wake_system(solution, operators)
    jump = operators.active_jump
    prolongation = operators.continuous_prolongation
    injection = operators.paired_jump_injection
    body_count = solution.body_unknown_count
    jump_count = operators.independent_jump_count
    base_count = topology.base_topology.dof_count

    topology_identity_error = max(
        _maximum_abs(jump @ prolongation),
        _maximum_abs(
            jump @ injection - np.eye(jump_count)
        ),
        _maximum_abs(prolongation.T @ injection),
    )
    topology_basis_rank = int(
        np.linalg.matrix_rank(
            np.column_stack((prolongation, injection))
        )
    )
    zero_endpoint_error = _maximum_abs(
        operators.full_jump[operators.zero_row_indices]
    )
    active_singular_values = np.linalg.svd(
        jump,
        compute_uv=False,
    )

    body_rank = int(np.linalg.matrix_rank(system.body_matrix))
    augmented = np.column_stack(
        (
            system.body_matrix,
            system.independent_wake_matrix,
        )
    )
    augmented_rank = int(np.linalg.matrix_rank(augmented))
    augmented_nullity = int(
        augmented.shape[1] - augmented_rank
    )
    eliminated_identity_error = _maximum_abs(
        system.body_matrix
        + system.eliminated_wake_matrix
        - system.eliminated_matrix
    )

    transformed_solutions = {}
    reference_morino = None
    basis_covariance_error = 0.0
    maximum_block_solution_error = 0.0
    maximum_full_residual = 0.0
    maximum_closure_residual = 0.0
    block_rank_deficiency = 0
    block_condition_number = 0.0
    expected_wake_jump = (
        jump @ solution.global_body_potential
    )
    residual_scale = max(
        float(np.linalg.norm(system.right_hand_side)),
        np.finfo(float).tiny,
    )
    for name, coordinate_map in _wake_coordinate_maps(
        jump_count
    ).items():
        solved = solve_morino_block(
            system,
            wake_coordinate_map=coordinate_map,
        )
        if reference_morino is None:
            reference_morino = solved
        body_error = _maximum_abs(
            solved.body_potential
            - solution.global_body_potential
        )
        wake_error = _maximum_abs(
            solved.wake_jump - expected_wake_jump
        )
        maximum_block_solution_error = max(
            maximum_block_solution_error,
            body_error,
            wake_error,
        )
        basis_covariance_error = max(
            basis_covariance_error,
            _maximum_abs(
                solved.body_potential
                - reference_morino.body_potential
            ),
            _maximum_abs(
                solved.wake_jump - reference_morino.wake_jump
            ),
        )
        full_residual = float(
            np.linalg.norm(solved.full_bie_residual)
            / residual_scale
        )
        closure_residual = _maximum_abs(
            solved.closure_residual
        )
        maximum_full_residual = max(
            maximum_full_residual,
            full_residual,
        )
        maximum_closure_residual = max(
            maximum_closure_residual,
            closure_residual,
        )
        block_rank_deficiency = max(
            block_rank_deficiency,
            solved.block_matrix.shape[0] - solved.block_rank,
        )
        block_condition_number = max(
            block_condition_number,
            solved.block_condition_number,
        )
        transformed_solutions[name] = {
            "coordinate_map_condition_number": float(
                np.linalg.cond(coordinate_map)
            ),
            "block_rank": solved.block_rank,
            "block_condition_number": (
                solved.block_condition_number
            ),
            "body_solution_abs_error": body_error,
            "wake_solution_abs_error": wake_error,
            "normalized_full_bie_residual": full_residual,
            "morino_closure_abs_residual": closure_residual,
        }

    mass = consistent_body_surface_mass(
        solution,
        quadrature_order=int(
            wake_specification["target_quadrature_order"]
        ),
    )
    quotient_euclidean = orthonormal_nullspace(injection.T)
    quotient_surface_l2 = orthonormal_nullspace(
        injection.T @ mass
    )
    projector_euclidean = (
        quotient_euclidean @ quotient_euclidean.T
    )
    projector_surface_l2 = (
        quotient_surface_l2 @ quotient_surface_l2.T
    )
    projector_difference = (
        projector_euclidean - projector_surface_l2
    )
    projector_operator_difference = float(
        np.linalg.norm(projector_difference, ord=2)
    )
    projector_frobenius_difference = float(
        np.linalg.norm(projector_difference, ord="fro")
    )
    active_y = topology.cut_node_coordinates[
        operators.active_row_indices,
        1,
    ]
    prescribed_wake_jump = 0.1 * (1.0 - active_y**2)
    quotient_solutions = {
        "euclidean": solve_quotient_with_attachment(
            system,
            quotient_basis=quotient_euclidean,
            prescribed_wake_jump=prescribed_wake_jump,
        ),
        "surface_l2": solve_quotient_with_attachment(
            system,
            quotient_basis=quotient_surface_l2,
            prescribed_wake_jump=prescribed_wake_jump,
        ),
    }
    quotient_solution_difference = _maximum_abs(
        quotient_solutions["euclidean"].body_potential
        - quotient_solutions["surface_l2"].body_potential
    )
    quotient_records = {}
    maximum_projected_residual = 0.0
    maximum_attachment_residual = 0.0
    minimum_full_bie_relative_residual = float("inf")
    manufactured_scale = max(
        float(
            np.linalg.norm(
                system.right_hand_side
                - system.independent_wake_matrix
                @ prescribed_wake_jump
            )
        ),
        np.finfo(float).tiny,
    )
    for name, quotient_solution in quotient_solutions.items():
        projected_residual = _maximum_abs(
            quotient_solution.projected_residual
        )
        attachment_residual = _maximum_abs(
            quotient_solution.attachment_residual
        )
        full_bie_relative_residual = float(
            np.linalg.norm(
                quotient_solution.body_bie_residual
            )
            / manufactured_scale
        )
        maximum_projected_residual = max(
            maximum_projected_residual,
            projected_residual,
        )
        maximum_attachment_residual = max(
            maximum_attachment_residual,
            attachment_residual,
        )
        minimum_full_bie_relative_residual = min(
            minimum_full_bie_relative_residual,
            full_bie_relative_residual,
        )
        quotient_records[name] = {
            "basis_shape": list(
                (
                    quotient_euclidean
                    if name == "euclidean"
                    else quotient_surface_l2
                ).shape
            ),
            "system_rank": quotient_solution.system_rank,
            "system_condition_number": (
                quotient_solution.system_condition_number
            ),
            "projected_residual_abs_max": projected_residual,
            "attachment_residual_abs_max": attachment_residual,
            "full_bie_relative_residual": (
                full_bie_relative_residual
            ),
        }

    input_mutation = max(
        _maximum_abs(
            solution.body_matrix
            - immutable_snapshots["body_matrix"]
        ),
        _maximum_abs(
            solution.wake_matrix
            - immutable_snapshots["wake_matrix"]
        ),
        _maximum_abs(
            solution.right_hand_side
            - immutable_snapshots["right_hand_side"]
        ),
        _maximum_abs(
            solution.global_body_potential
            - immutable_snapshots["body_potential"]
        ),
        _maximum_abs(
            topology.upper_cut_dofs
            - immutable_snapshots["upper_cut_dofs"]
        ),
        _maximum_abs(
            topology.lower_cut_dofs
            - immutable_snapshots["lower_cut_dofs"]
        ),
    )

    algebraic_checks = {
        "full_cut_node_count": (
            operators.full_cut_node_count
            == int(thresholds["expected_full_cut_node_count"])
        ),
        "independent_cut_rank": (
            jump_count
            == int(thresholds["expected_independent_cut_rank"])
            and int(np.linalg.matrix_rank(jump)) == jump_count
        ),
        "continuous_base_dof_count": (
            base_count
            == int(
                thresholds[
                    "expected_continuous_base_dof_count"
                ]
            )
        ),
        "classified_body_dof_count": (
            body_count
            == int(
                thresholds[
                    "expected_classified_body_dof_count"
                ]
            )
        ),
        "zero_endpoint_rows": (
            len(operators.zero_row_indices) == 2
            and zero_endpoint_error
            <= float(thresholds["zero_endpoint_row_abs_max"])
        ),
        "topology_split_identities": (
            topology_identity_error
            <= float(thresholds["topology_identity_abs_max"])
            and topology_basis_rank == body_count
        ),
        "body_matrix_is_full_rank": (
            body_count - body_rank
            <= int(thresholds["rank_deficiency_max"])
        ),
        "independent_wake_has_r_closure_directions": (
            augmented_rank == body_count
            and augmented_nullity == jump_count
        ),
        "eliminated_wake_factorization": (
            system.wake_factorization_error
            <= float(
                thresholds["wake_factorization_abs_max"]
            )
            and eliminated_identity_error
            <= float(
                thresholds["wake_factorization_abs_max"]
            )
        ),
        "morino_block_is_full_rank": (
            block_rank_deficiency
            <= int(thresholds["rank_deficiency_max"])
        ),
        "morino_block_matches_eliminated_solve": (
            maximum_block_solution_error
            <= float(
                thresholds["morino_block_solution_abs_max"]
            )
        ),
        "full_body_bie_residual": (
            maximum_full_residual
            <= float(
                thresholds["normalized_full_bie_residual_max"]
            )
        ),
        "morino_closure_residual": (
            maximum_closure_residual
            <= float(thresholds["morino_closure_abs_max"])
        ),
        "wake_basis_covariance": (
            basis_covariance_error
            <= float(
                thresholds["wake_basis_covariance_abs_max"]
            )
        ),
        "input_state_is_immutable": (
            input_mutation
            <= float(
                thresholds["input_state_mutation_abs_max"]
            )
        ),
    }
    quotient_counterexample_checks = {
        "metric_changes_quotient_projector": (
            projector_operator_difference
            >= float(
                thresholds[
                    "quotient_projector_operator_difference_min"
                ]
            )
        ),
        "metric_changes_quotient_solution": (
            quotient_solution_difference
            >= float(
                thresholds[
                    "quotient_solution_abs_difference_min"
                ]
            )
        ),
        "each_metric_satisfies_its_projected_equation": (
            maximum_projected_residual
            <= float(
                thresholds["quotient_projected_residual_max"]
            )
        ),
        "each_metric_satisfies_attachment": (
            maximum_attachment_residual
            <= float(
                thresholds["quotient_attachment_abs_max"]
            )
        ),
        "each_metric_discards_nonzero_full_bie_residual": (
            minimum_full_bie_relative_residual
            >= float(
                thresholds[
                    "quotient_full_bie_relative_residual_min"
                ]
            )
        ),
    }
    algebraic_checks = {
        name: bool(value)
        for name, value in algebraic_checks.items()
    }
    quotient_counterexample_checks = {
        name: bool(value)
        for name, value in quotient_counterexample_checks.items()
    }
    algebraic_go = all(algebraic_checks.values())
    quotient_no_go = all(
        quotient_counterexample_checks.values()
    )
    result = {
        "artifact": "actual_wake_kutta_closure_roles_oracle",
        "claim_node": contract["claim_node"],
        "stage": contract["stage"],
        "canonical": canonical,
        "topology": {
            "full_cut_node_count": (
                operators.full_cut_node_count
            ),
            "active_row_indices": (
                operators.active_row_indices.tolist()
            ),
            "zero_row_indices": (
                operators.zero_row_indices.tolist()
            ),
            "independent_cut_rank": jump_count,
            "continuous_base_dof_count": base_count,
            "classified_body_dof_count": body_count,
            "duplicated_dof_count": (
                topology.duplicated_dof_count
            ),
            "upper_cut_dofs": (
                topology.upper_cut_dofs.tolist()
            ),
            "lower_cut_dofs": (
                topology.lower_cut_dofs.tolist()
            ),
            "active_jump_singular_values": (
                active_singular_values.tolist()
            ),
            "zero_endpoint_row_abs_max": zero_endpoint_error,
            "topology_identity_abs_max": (
                topology_identity_error
            ),
            "topology_basis_rank": topology_basis_rank,
        },
        "independent_wake_equation": {
            "body_matrix_rank": body_rank,
            "body_matrix_condition_number": float(
                np.linalg.cond(system.body_matrix)
            ),
            "augmented_shape": list(augmented.shape),
            "augmented_rank": augmented_rank,
            "augmented_nullity": augmented_nullity,
            "eliminated_matrix_condition_number": float(
                np.linalg.cond(system.eliminated_matrix)
            ),
            "wake_factorization_abs_error_max": (
                system.wake_factorization_error
            ),
            "eliminated_matrix_identity_abs_error_max": (
                eliminated_identity_error
            ),
        },
        "morino_block": {
            "shape": [body_count + jump_count] * 2,
            "rank_deficiency_max": block_rank_deficiency,
            "condition_number_max": block_condition_number,
            "solution_abs_error_max": (
                maximum_block_solution_error
            ),
            "normalized_full_bie_residual_max": (
                maximum_full_residual
            ),
            "closure_abs_residual_max": (
                maximum_closure_residual
            ),
            "wake_basis_covariance_abs_error_max": (
                basis_covariance_error
            ),
            "coordinate_cases": transformed_solutions,
        },
        "quotient_counterexample": {
            "prescribed_wake_jump": (
                prescribed_wake_jump.tolist()
            ),
            "projector_operator_difference": (
                projector_operator_difference
            ),
            "projector_frobenius_difference": (
                projector_frobenius_difference
            ),
            "body_solution_abs_difference": (
                quotient_solution_difference
            ),
            "projected_residual_abs_max": (
                maximum_projected_residual
            ),
            "attachment_residual_abs_max": (
                maximum_attachment_residual
            ),
            "full_bie_relative_residual_min": (
                minimum_full_bie_relative_residual
            ),
            "metrics": quotient_records,
        },
        "input_state_mutation_abs_max": input_mutation,
        "thresholds": thresholds,
        "algebraic_checks": algebraic_checks,
        "quotient_counterexample_checks": (
            quotient_counterexample_checks
        ),
        "algebraic_role_decision": (
            "GO" if algebraic_go else "NO-GO"
        ),
        "quotient_claim_decision": (
            "NO-GO" if quotient_no_go else "UNRESOLVED"
        ),
        "stage_decision": (
            "ALGEBRAIC-ROLE-GO / QUOTIENT-NO-GO"
            if algebraic_go and quotient_no_go
            else "NO-GO"
        ),
        "production_activation_allowed": False,
        "interpretation": (
            "All body BIE rows remain independent. The natural seven "
            "closure directions arise only after the current wake jump is "
            "made independent. Morino is one closure and a physical "
            "birth/unsteady-pressure Kutta law must replace it, not be "
            "appended beside it. The tested quotient is metric dependent "
            "and succeeds only by discarding nonzero body-BIE residual."
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
        "ALGEBRAIC-ROLE-GO / QUOTIENT-NO-GO"
    ):
        raise SystemExit(1)

