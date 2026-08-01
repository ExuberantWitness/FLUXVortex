"""Run the preregistered S3d old-known/new-current wake partition oracle."""
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
    ActualBoundaryBodyWakeSolution,
    solve_actual_boundary_body_wake_p2,
)
from claim_runtime.classified_p2_cut_topology import (  # noqa: E402
    classified_p2_cut_topology,
)


CASES = (
    HERE / "docs" / "diag"
    / "actual_boundary_unsteady_wake_partition_cases.yaml"
)
RESULTS = (
    HERE / "docs" / "diag"
    / "actual_boundary_unsteady_wake_partition_results.json"
)


def run() -> dict:
    contract = yaml.safe_load(CASES.read_text(encoding="utf-8"))
    canonical = contract["canonical"]
    thresholds = contract["thresholds"]
    mesh, upper, lower, cut_edges, endpoints = (
        build_canonical_diamond_wing()
    )
    topology = classified_p2_cut_topology(
        mesh,
        upper_face_indices=upper,
        lower_face_indices=lower,
        cut_edges=cut_edges,
        zero_jump_end_vertices=endpoints,
    )
    y = topology.cut_node_coordinates[:, 1]
    g = 1.0 - y**2
    normal_fixed = np.array(
        ((0.20 * g, 0.30 * g, 0.40 * g),),
        dtype=float,
    )
    normal_active = np.array((0.40 * g, 0.10 * g))
    reverse_fixed = -normal_fixed
    reverse_active = -normal_active
    zero_fixed = np.zeros_like(normal_fixed)
    zero_active = np.zeros_like(normal_active)
    immutable_inputs = [
        normal_fixed.copy(),
        normal_active.copy(),
        reverse_fixed.copy(),
        reverse_active.copy(),
        zero_fixed.copy(),
        zero_active.copy(),
    ]
    alpha = np.deg2rad(
        float(canonical["freestream"]["alpha_deg"])
    )
    speed = float(canonical["freestream"]["speed"])
    incident = np.repeat(
        np.array(
            ((speed * np.cos(alpha), 0.0, speed * np.sin(alpha)),)
        ),
        len(mesh.faces),
        axis=0,
    )
    zero_incident = np.zeros_like(incident)
    order = int(canonical["quadrature_order"])
    edge_x = np.array((1.0, 1.5, 2.0))

    def solve(
        velocity: np.ndarray,
        fixed: np.ndarray,
        active: np.ndarray,
    ) -> ActualBoundaryBodyWakeSolution:
        return solve_actual_boundary_body_wake_p2(
            mesh,
            topology,
            incident_velocity=velocity,
            downstream_edge_x=2.0,
            wake_edge_x_nodes=edge_x,
            fixed_old_wake_rows=fixed,
            active_known_rows=active,
            target_quadrature_order=order,
            source_quadrature_order=order,
        )

    normal = solve(incident, normal_fixed, normal_active)
    reverse = solve(incident, reverse_fixed, reverse_active)
    zero_history = solve(incident, zero_fixed, zero_active)
    history_only = solve(
        zero_incident,
        normal_fixed,
        normal_active,
    )
    solutions = (normal, reverse, zero_history, history_only)
    unchanged = [
        normal_fixed,
        normal_active,
        reverse_fixed,
        reverse_active,
        zero_fixed,
        zero_active,
    ]
    mutation = max(
        float(np.max(np.abs(after - before), initial=0.0))
        for before, after in zip(immutable_inputs, unchanged)
    )
    interface_error = max(
        float(
            np.max(
                np.abs(
                    solution.wake_history.bands[0]
                    .potential_jump_rows[-1]
                    - solution.wake_history.bands[1]
                    .potential_jump_rows[0]
                ),
                initial=0.0,
            )
        )
        for solution in solutions
    )
    attachment_error = max(
        float(
            np.max(
                np.abs(
                    solution.wake_history.bands[-1]
                    .potential_jump_rows[-1]
                    - solution.body_cut_jump
                ),
                initial=0.0,
            )
        )
        for solution in solutions
    )
    counterfactual_difference = float(
        np.max(
            np.abs(normal.body_cut_jump - reverse.body_cut_jump),
            initial=0.0,
        )
    )
    affine_error = float(
        np.max(
            np.abs(
                normal.global_body_potential
                - zero_history.global_body_potential
                - history_only.global_body_potential
            ),
            initial=0.0,
        )
    )
    rank_deficiency = max(
        solution.body_unknown_count - solution.rank
        for solution in solutions
    )
    weak_residual = max(
        solution.relative_weak_residual for solution in solutions
    )
    condition_number = max(
        solution.condition_number for solution in solutions
    )
    old_unknowns = max(
        solution.independent_old_wake_unknown_count
        for solution in solutions
    )
    tip_jump = max(
        max(
            abs(solution.body_cut_jump[0]),
            abs(solution.body_cut_jump[-1]),
        )
        for solution in solutions
    )
    checks = {
        "old_material_state_is_immutable": (
            mutation
            <= float(thresholds["old_state_mutation_abs_max"])
        ),
        "old_to_active_interface_is_exact": (
            interface_error
            <= float(thresholds["history_interface_abs_max"])
        ),
        "active_current_attaches_to_body": (
            attachment_error
            <= float(thresholds["active_attachment_abs_max"])
        ),
        "old_wake_is_known_rhs_not_unknown": (
            old_unknowns
            <= int(
                thresholds[
                    "independent_old_wake_unknown_count_max"
                ]
            )
            and all(
                np.linalg.norm(
                    solution.known_wake_right_hand_side
                ) > 0.0
                for solution in (normal, reverse, history_only)
            )
        ),
        "full_rank_weak_system": (
            rank_deficiency
            <= int(thresholds["rank_deficiency_max"])
            and weak_residual
            <= float(thresholds["normalized_weak_residual_max"])
            and condition_number
            <= float(thresholds["condition_number_max"])
        ),
        "current_solution_consumes_old_history": (
            counterfactual_difference
            >= float(
                thresholds[
                    "counterfactual_current_jump_difference_min"
                ]
            )
        ),
        "incident_history_affine_superposition": (
            affine_error
            <= float(
                thresholds["affine_superposition_abs_max"]
            )
        ),
        "tip_jump_is_zero": (
            tip_jump <= float(thresholds["tip_jump_abs_max"])
        ),
    }
    checks = {name: bool(value) for name, value in checks.items()}
    result = {
        "artifact": "actual_boundary_unsteady_wake_partition_oracle",
        "claim_node": contract["claim_node"],
        "stage": contract["stage"],
        "canonical": canonical,
        "aggregate_metrics": {
            "old_state_mutation_abs_max": mutation,
            "history_interface_abs_max": interface_error,
            "active_attachment_abs_max": attachment_error,
            "independent_old_wake_unknown_count_max": old_unknowns,
            "rank_deficiency_max": rank_deficiency,
            "normalized_weak_residual_max": weak_residual,
            "condition_number_max": condition_number,
            "counterfactual_current_jump_difference": (
                counterfactual_difference
            ),
            "affine_superposition_abs_max": affine_error,
            "tip_jump_abs_max": tip_jump,
            "normal_current_jump": normal.body_cut_jump.tolist(),
            "reverse_current_jump": reverse.body_cut_jump.tolist(),
            "zero_history_current_jump": (
                zero_history.body_cut_jump.tolist()
            ),
            "known_wake_rhs_norm": float(
                np.linalg.norm(normal.known_wake_right_hand_side)
            ),
        },
        "thresholds": thresholds,
        "checks": checks,
        "stage_decision": (
            "GO" if all(checks.values()) else "NO-GO"
        ),
        "production_activation_allowed": False,
        "interpretation": (
            "Passing validates only the affine old-known/new-current "
            "material ledger. The explicit previous/middle rows remain "
            "canonical inputs; no time integrator, pressure or force is "
            "validated."
        ),
    }
    RESULTS.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, sort_keys=True))
