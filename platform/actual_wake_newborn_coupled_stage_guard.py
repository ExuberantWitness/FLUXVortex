"""Run the preregistered S3aa coupled newborn half/full trace gate."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import yaml


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from actual_body_wake_velocity_ledger_guard import (  # noqa: E402
    _canonical_state,
)
from actual_wake_owned_stage_velocity_guard import (  # noqa: E402
    _actual_ledger,
)
from claim_runtime.actual_wake_newborn_coupled_stage import (  # noqa: E402
    solve_actual_wake_coupled_newborn_trace,
)
from claim_runtime.actual_wake_newborn_transition import (  # noqa: E402
    augment_actual_wake_with_newborn_band,
)
from claim_runtime.actual_wake_stage_topology import (  # noqa: E402
    actual_wake_stage_topology,
)
from claim_runtime.actual_wake_stage_velocity import (  # noqa: E402
    actual_wake_owned_quadrature,
)
from claim_runtime.actual_wake_weak_geometry_velocity import (  # noqa: E402
    project_actual_wake_global_weak_normal_velocity,
)


CASES = (
    HERE
    / "docs"
    / "diag"
    / "actual_wake_newborn_coupled_stage_cases.yaml"
)
S3T_CASES = (
    HERE
    / "docs"
    / "diag"
    / "actual_wake_owned_stage_velocity_cases.yaml"
)
S3Z_RESULTS = (
    HERE
    / "docs"
    / "diag"
    / "actual_wake_release_weak_projection_results.json"
)
RESULTS = (
    HERE
    / "docs"
    / "diag"
    / "actual_wake_newborn_coupled_stage_results.json"
)


def run() -> dict:
    contract = yaml.safe_load(CASES.read_text(encoding="utf-8"))
    s3t_contract = yaml.safe_load(
        S3T_CASES.read_text(encoding="utf-8")
    )
    s3z_result = json.loads(
        S3Z_RESULTS.read_text(encoding="utf-8")
    )
    canonical = contract["canonical"]
    thresholds = contract["thresholds"]
    (
        mesh,
        body_topology,
        _upper,
        _lower,
        _cut_edges,
        _endpoints,
        _pre_solve_history,
        attachment,
        solution,
    ) = _canonical_state()
    history = solution.wake_history
    topology = actual_wake_stage_topology(
        history,
        body_attachment_id="canonical-body-cut",
    )
    geometry_snapshot = tuple(
        band.surface.vertices.copy() for band in history.bands
    )
    scalar_snapshot = tuple(
        band.potential_jump_rows.copy() for band in history.bands
    )
    weak_quadrature = actual_wake_owned_quadrature(
        topology,
        history,
        quadrature_order=7,
        query_id="S3aa-weak-release-q7",
    )
    weak_ledger = _actual_ledger(
        solution,
        weak_quadrature.query,
        s3t_contract,
    )
    weak_projection = (
        project_actual_wake_global_weak_normal_velocity(
            topology,
            history,
            weak_quadrature,
            weak_ledger.total,
        )
    )
    body_p1 = topology.boundary_roles.body_attachment_p1_dofs
    body_edge = topology.p1_vertices[body_p1].copy()
    release_velocity = weak_projection.dof_velocity[body_p1]
    old_trace = topology.chronological_rows(history)[-1]
    timestep = float(
        canonical["release_geometry"]["timestep"]
    )
    start_time = float(history.bands[-1].time_nodes[-1])
    half_fraction = float(
        canonical["release_geometry"]["half_fraction"]
    )
    full_fraction = float(
        canonical["release_geometry"]["full_fraction"]
    )
    half_transition = augment_actual_wake_with_newborn_band(
        history,
        topology,
        released_edge=(
            body_edge
            + half_fraction * timestep * release_velocity
        ),
        current_body_edge=body_edge,
        time_nodes=np.array(
            (
                start_time,
                start_time
                + 0.5 * half_fraction * timestep,
                start_time + half_fraction * timestep,
            )
        ),
        midpoint_trace=old_trace,
        current_trace=old_trace,
        sheet_id="S3aa-newborn-half",
    )
    half = solve_actual_wake_coupled_newborn_trace(
        mesh,
        body_topology,
        half_transition,
        attachment,
        incident_velocity=solution.incident_velocity,
        wall_velocity=solution.wall_velocity,
        copy_counterfactual_trace=old_trace,
        boundary_quadrature_order=int(
            canonical["algebraic_solver"][
                "boundary_quadrature_order"
            ]
        ),
    )
    full_transition = augment_actual_wake_with_newborn_band(
        history,
        topology,
        released_edge=(
            body_edge
            + full_fraction * timestep * release_velocity
        ),
        current_body_edge=body_edge,
        time_nodes=np.array(
            (
                start_time,
                start_time + 0.5 * full_fraction * timestep,
                start_time + full_fraction * timestep,
            )
        ),
        midpoint_trace=half.solved_trace,
        current_trace=half.solved_trace,
        sheet_id="S3aa-newborn-full",
    )
    full = solve_actual_wake_coupled_newborn_trace(
        mesh,
        body_topology,
        full_transition,
        attachment,
        incident_velocity=solution.incident_velocity,
        wall_velocity=solution.wall_velocity,
        copy_counterfactual_trace=half.solved_trace,
        boundary_quadrature_order=int(
            canonical["algebraic_solver"][
                "boundary_quadrature_order"
            ]
        ),
    )
    stages = (half, full)
    transitions = (half_transition, full_transition)
    maximum_rank_deficiency = max(
        value.report.rank_deficiency for value in stages
    )
    maximum_condition = max(
        value.report.condition_number for value in stages
    )
    maximum_algebraic = max(
        value.report.algebraic_trace_residual
        for value in stages
    )
    maximum_free = max(
        value.report.free_state_preservation_error
        for value in stages
    )
    maximum_weak = max(
        value.report.maximum_actual_boundary_relative_weak_residual
        for value in stages
    )
    maximum_attachment = max(
        value.report.maximum_wake_attachment_error
        for value in stages
    )
    minimum_copy = min(
        value.report.copy_counterfactual_residual
        for value in stages
    )
    maximum_seam = 0.0
    maximum_roundtrip = 0.0
    for stage in stages:
        report = stage.solution.wake_history.continuity_report()
        maximum_seam = max(
            maximum_seam,
            report.max_geometry_gap,
            report.max_trace_jump,
        )
        roundtrip = stage.topology.global_p2_state(
            stage.topology.rebuild_history(
                stage.solution.wake_history,
                stage.global_p2_state,
            )
        )
        maximum_roundtrip = max(
            maximum_roundtrip,
            float(
                np.max(
                    np.abs(
                        roundtrip - stage.global_p2_state
                    ),
                    initial=0.0,
                )
            ),
        )
    half_change = float(
        np.max(
            np.abs(half.solved_trace - old_trace),
            initial=0.0,
        )
    )
    full_change = float(
        np.max(
            np.abs(full.solved_trace - half.solved_trace),
            initial=0.0,
        )
    )
    mutation = max(
        *(value.report.input_state_mutation for value in stages),
        *(
            float(
                np.max(
                    np.abs(band.surface.vertices - geometry),
                    initial=0.0,
                )
            )
            for band, geometry in zip(
                history.bands,
                geometry_snapshot,
                strict=True,
            )
        ),
        *(
            float(
                np.max(
                    np.abs(
                        band.potential_jump_rows - scalar
                    ),
                    initial=0.0,
                )
            )
            for band, scalar in zip(
                history.bands,
                scalar_snapshot,
                strict=True,
            )
        ),
    )
    inferred = max(
        value.report.inferred_scalar_count for value in stages
    )
    geometry_iterations = max(
        value.report.geometry_iteration_count for value in stages
    )
    weak_relative_change = float(
        s3z_result["aggregate_metrics"][
            "weak_finest_release_relative_change"
        ]
    )
    checks = {
        "validated_weak_release_is_the_only_geometry_input": (
            s3z_result["stage_decision"] == "WEAK-GO"
            and weak_relative_change
            <= float(
                thresholds[
                    "weak_release_q7_relative_change_max"
                ]
            )
        ),
        "half_and_full_augmented_geometries_are_valid": (
            half_transition.report.minimum_newborn_face_area
            >= float(
                thresholds["minimum_half_stage_face_area_min"]
            )
            and full_transition.report.minimum_newborn_face_area
            >= float(
                thresholds["minimum_full_stage_face_area_min"]
            )
            and all(
                value.report.augmented_p2_dof_count
                == int(
                    thresholds[
                        "augmented_P2_dof_count_expected"
                    ]
                )
                for value in transitions
            )
        ),
        "half_and_full_affine_trace_systems_are_valid": (
            maximum_rank_deficiency
            <= int(
                thresholds["algebraic_rank_deficiency_max"]
            )
            and maximum_condition
            <= float(
                thresholds["algebraic_condition_number_max"]
            )
        ),
        "independent_algebraic_trace_residuals_close": (
            maximum_algebraic
            <= float(
                thresholds["algebraic_trace_residual_abs_max"]
            )
        ),
        "all_nonbody_P2_values_are_preserved": (
            maximum_free
            <= float(
                thresholds["free_state_preservation_abs_max"]
            )
        ),
        "actual_boundary_and_attachment_residuals_close": (
            maximum_weak
            <= float(
                thresholds[
                    "actual_boundary_relative_weak_residual_max"
                ]
            )
            and maximum_attachment
            <= float(thresholds["wake_attachment_abs_max"])
        ),
        "newborn_rows_seams_and_P2_roundtrip_hold": (
            maximum_seam
            <= float(thresholds["chronological_seam_abs_max"])
            and maximum_roundtrip
            <= float(thresholds["P2_roundtrip_abs_max"])
        ),
        "solved_half_and_full_traces_are_nontrivial": (
            half_change
            >= float(
                thresholds[
                    "solved_half_trace_change_abs_min"
                ]
            )
            and full_change
            >= float(
                thresholds[
                    "solved_full_trace_change_abs_min"
                ]
            )
        ),
        "copy_counterfactuals_fail_the_stage_constraint": (
            minimum_copy
            >= float(
                thresholds[
                    "copy_counterfactual_residual_abs_min"
                ]
            )
        ),
        "no_inference_iteration_or_input_mutation_enters": (
            inferred
            <= int(thresholds["inferred_scalar_count_max"])
            and geometry_iterations
            <= int(thresholds["geometry_iteration_count_max"])
            and mutation
            <= float(
                thresholds["input_state_mutation_abs_max"]
            )
        ),
    }
    checks = {name: bool(value) for name, value in checks.items()}
    result = {
        "artifact": "actual_wake_newborn_coupled_stage_oracle",
        "claim_node": contract["claim_node"],
        "stage": contract["stage"],
        "stage_decision": (
            "GO" if all(checks.values()) else "NO-GO"
        ),
        "checks": checks,
        "aggregate_metrics": {
            "weak_release_q7_relative_change": (
                weak_relative_change
            ),
            "weak_release_speed_abs_max": float(
                np.max(
                    np.linalg.norm(release_velocity, axis=1),
                    initial=0.0,
                )
            ),
            "minimum_half_stage_face_area": (
                half_transition.report.minimum_newborn_face_area
            ),
            "minimum_full_stage_face_area": (
                full_transition.report.minimum_newborn_face_area
            ),
            "augmented_P2_dof_counts": [
                value.report.augmented_p2_dof_count
                for value in transitions
            ],
            "algebraic_rank_deficiency_max": (
                maximum_rank_deficiency
            ),
            "algebraic_condition_number_max": (
                maximum_condition
            ),
            "algebraic_trace_residual_abs_max": (
                maximum_algebraic
            ),
            "free_state_preservation_abs_max": maximum_free,
            "actual_boundary_relative_weak_residual_max": (
                maximum_weak
            ),
            "wake_attachment_abs_max": maximum_attachment,
            "chronological_seam_abs_max": maximum_seam,
            "P2_roundtrip_abs_max": maximum_roundtrip,
            "solved_half_trace_change_abs_max": half_change,
            "solved_full_trace_change_abs_max": full_change,
            "half_copy_counterfactual_residual_abs_max": (
                half.report.copy_counterfactual_residual
            ),
            "full_copy_counterfactual_residual_abs_max": (
                full.report.copy_counterfactual_residual
            ),
            "minimum_copy_counterfactual_residual": minimum_copy,
            "half_basis_solve_count": (
                half.report.basis_solve_count
            ),
            "full_basis_solve_count": (
                full.report.basis_solve_count
            ),
            "input_state_mutation_abs_max": mutation,
            "inferred_scalar_count": inferred,
            "geometry_iteration_count": geometry_iterations,
        },
        "forbidden_quantities_absent": [
            "point_trace",
            "scalar_copy_in_solution",
            "scalar_average",
            "scalar_clamp",
            "strength_inference",
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
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


if __name__ == "__main__":
    payload = run()
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
    raise SystemExit(
        0 if payload["stage_decision"] == "GO" else 1
    )
