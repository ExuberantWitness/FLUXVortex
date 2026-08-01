"""Run the preregistered S3y actual-wake newborn transition gate."""
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
from actual_boundary_explicit_wake_geometry_guard import (  # noqa: E402
    _rotation_matrix,
    _transform_history,
)
from actual_wake_owned_stage_velocity_guard import (  # noqa: E402
    _actual_ledger,
)
from claim_runtime.actual_body_wake_velocity import (  # noqa: E402
    WakeSheetQuery,
    wake_sheet_interior_query,
)
from claim_runtime.actual_wake_newborn_transition import (  # noqa: E402
    augment_actual_wake_with_newborn_band,
    project_released_attachment_normal_velocity,
)
from claim_runtime.actual_wake_stage_topology import (  # noqa: E402
    ActualWakeStageTopologyError,
    actual_wake_stage_topology,
)
from claim_runtime.distributed_doublet import (  # noqa: E402
    DistributedDoubletError,
    newborn_material_wake_band,
)


CASES = (
    HERE
    / "docs"
    / "diag"
    / "actual_wake_newborn_transition_cases.yaml"
)
S3T_CASES = (
    HERE
    / "docs"
    / "diag"
    / "actual_wake_owned_stage_velocity_cases.yaml"
)
RESULTS = (
    HERE
    / "docs"
    / "diag"
    / "actual_wake_newborn_transition_results.json"
)


def _counterfactual_rows(
    upstream: np.ndarray,
    fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    coordinate = np.linspace(-1.0, 1.0, len(upstream))
    envelope = 1.0 - coordinate**2
    midpoint = (
        upstream
        + fraction
        * envelope
        * (0.013 + 0.004 * coordinate)
    )
    current = (
        upstream
        + fraction
        * envelope
        * (0.021 - 0.003 * coordinate)
    )
    return midpoint, current


def _rotated_query(
    query: WakeSheetQuery,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> WakeSheetQuery:
    return WakeSheetQuery(
        points=query.points @ rotation.T + translation,
        patch_indices=query.patch_indices,
        face_indices=query.face_indices,
        barycentric=query.barycentric,
        query_id="S3y-rigid-release-query",
    )


def run() -> dict:
    contract = yaml.safe_load(CASES.read_text(encoding="utf-8"))
    s3t_contract = yaml.safe_load(
        S3T_CASES.read_text(encoding="utf-8")
    )
    canonical = contract["canonical"]
    thresholds = contract["thresholds"]
    (
        _mesh,
        _body_topology,
        _upper,
        _lower,
        _cut_edges,
        _endpoints,
        _pre_solve_history,
        _attachment,
        solution,
    ) = _canonical_state()
    history = solution.wake_history
    topology = actual_wake_stage_topology(
        history,
        body_attachment_id="canonical-body-cut",
    )
    query = wake_sheet_interior_query(history)
    ledger = _actual_ledger(solution, query, s3t_contract)
    release = project_released_attachment_normal_velocity(
        topology,
        history,
        query,
        ledger.total,
    )
    timestep = float(canonical["release"]["timestep"])
    body_dofs = topology.boundary_roles.body_attachment_p1_dofs
    body_edge = topology.p1_vertices[body_dofs].copy()
    upstream = topology.chronological_rows(history)[-1]
    start_time = float(history.bands[-1].time_nodes[-1])
    transitions = {}
    for fraction in canonical["release"]["stage_fractions"]:
        value = float(fraction)
        midpoint, current = _counterfactual_rows(upstream, value)
        transitions[value] = augment_actual_wake_with_newborn_band(
            history,
            topology,
            released_edge=(
                body_edge + value * timestep * release.velocity
            ),
            current_body_edge=body_edge,
            time_nodes=np.array(
                (
                    start_time,
                    start_time + 0.5 * value * timestep,
                    start_time + value * timestep,
                )
            ),
            midpoint_trace=midpoint,
            current_trace=current,
            sheet_id=f"S3y-newborn-{value:g}",
        )
    half = transitions[0.5]
    full = transitions[1.0]

    naive_failure = 0
    try:
        newborn_material_wake_band(
            sheet_id="S3y-naive-coincident",
            vortex_family=history.vortex_family,
            previous_edge=body_edge,
            current_edge=body_edge,
            time_nodes=np.array(
                (
                    start_time,
                    start_time + 0.5 * timestep,
                    start_time + timestep,
                )
            ),
            potential_jump_rows=np.repeat(
                upstream[None, :],
                3,
                axis=0,
            ),
            span_diagonal_pattern="mirror_symmetric",
        )
    except DistributedDoubletError:
        naive_failure += 1

    rigid = canonical["rigid_frame_counterfactual"]
    rotation = _rotation_matrix(
        np.asarray(rigid["rotation_axis"], dtype=float),
        np.deg2rad(float(rigid["rotation_deg"])),
    )
    translation = np.asarray(rigid["translation"], dtype=float)
    moved_history = _transform_history(
        history,
        rotation,
        translation,
    )
    moved_topology = actual_wake_stage_topology(
        moved_history,
        body_attachment_id="canonical-body-cut",
    )
    moved_query = _rotated_query(query, rotation, translation)
    moved_release = project_released_attachment_normal_velocity(
        moved_topology,
        moved_history,
        moved_query,
        ledger.total @ rotation.T,
    )
    rigid_release_error = float(
        np.max(
            np.abs(
                moved_release.velocity
                - release.velocity @ rotation.T
            ),
            initial=0.0,
        )
    )
    midpoint, current = _counterfactual_rows(upstream, 1.0)
    moved_full = augment_actual_wake_with_newborn_band(
        moved_history,
        moved_topology,
        released_edge=(
            (body_edge + timestep * release.velocity)
            @ rotation.T
            + translation
        ),
        current_body_edge=body_edge @ rotation.T + translation,
        time_nodes=np.array(
            (
                start_time,
                start_time + 0.5 * timestep,
                start_time + timestep,
            )
        ),
        midpoint_trace=midpoint,
        current_trace=current,
        sheet_id="S3y-newborn-1-rigid",
    )
    rigid_geometry_error = float(
        np.max(
            np.abs(
                moved_full.augmented_topology.p1_vertices
                - (
                    full.augmented_topology.p1_vertices
                    @ rotation.T
                    + translation
                )
            ),
            initial=0.0,
        )
    )
    rigid_topology_mismatches = sum(
        (
            int(
                not np.array_equal(
                    moved_full.augmented_topology.p1_faces,
                    full.augmented_topology.p1_faces,
                )
            ),
            int(
                not np.array_equal(
                    moved_full.old_p1_to_augmented,
                    full.old_p1_to_augmented,
                )
            ),
            int(
                not np.array_equal(
                    moved_full.old_p2_to_augmented,
                    full.old_p2_to_augmented,
                )
            ),
            int(
                not np.array_equal(
                    moved_full.augmented_topology
                    .p2_dof_to_chronological,
                    full.augmented_topology
                    .p2_dof_to_chronological,
                )
            ),
        )
    )

    invalid_failures = 0
    try:
        augment_actual_wake_with_newborn_band(
            history,
            topology,
            released_edge=body_edge,
            current_body_edge=body_edge,
            time_nodes=np.array(
                (
                    start_time,
                    start_time + 0.5 * timestep,
                    start_time + timestep,
                )
            ),
            midpoint_trace=midpoint,
            current_trace=current,
            sheet_id="invalid-coincident",
        )
    except (ActualWakeStageTopologyError, DistributedDoubletError):
        invalid_failures += 1
    try:
        augment_actual_wake_with_newborn_band(
            history,
            topology,
            released_edge=(
                body_edge + timestep * release.velocity
            ),
            current_body_edge=body_edge,
            time_nodes=np.array(
                (
                    start_time,
                    start_time + 0.5 * timestep,
                    start_time + timestep,
                )
            ),
            midpoint_trace=midpoint[:-1],
            current_trace=current,
            sheet_id="invalid-trace",
        )
    except (ActualWakeStageTopologyError, DistributedDoubletError):
        invalid_failures += 1
    try:
        augment_actual_wake_with_newborn_band(
            history,
            full.augmented_topology,
            released_edge=(
                body_edge + timestep * release.velocity
            ),
            current_body_edge=body_edge,
            time_nodes=np.array(
                (
                    start_time,
                    start_time + 0.5 * timestep,
                    start_time + timestep,
                )
            ),
            midpoint_trace=midpoint,
            current_trace=current,
            sheet_id="invalid-topology",
        )
    except (ActualWakeStageTopologyError, DistributedDoubletError):
        invalid_failures += 1

    expected = canonical["augmented_counts"]
    report = full.report
    count_mismatches = sum(
        (
            int(report.augmented_band_count != expected["band_count"]),
            int(
                report.augmented_p1_dof_count
                != expected["global_P1_dof_count"]
            ),
            int(
                report.augmented_p2_dof_count
                != expected["global_P2_dof_count"]
            ),
            int(
                len(full.old_p1_to_augmented)
                != expected["old_P1_injected_dof_count"]
            ),
            int(
                len(full.old_p2_to_augmented)
                != expected["old_P2_injected_dof_count"]
            ),
            int(
                report.augmented_p2_dof_count
                - report.initial_p2_dof_count
                != expected["newborn_P2_dof_count"]
            ),
        )
    )
    maximum_injection = max(
        value.report.old_p1_injection_error
        for value in transitions.values()
    )
    maximum_p2_injection = max(
        value.report.old_p2_injection_error
        for value in transitions.values()
    )
    maximum_trace_error = max(
        max(
            value.report.newborn_upstream_trace_error,
            value.report.newborn_midpoint_trace_error,
            value.report.newborn_current_trace_error,
            value.report.chronological_geometry_seam_error,
            value.report.chronological_scalar_seam_error,
        )
        for value in transitions.values()
    )
    maximum_roundtrip = max(
        value.report.p2_roundtrip_error
        for value in transitions.values()
    )
    maximum_overlap = max(
        value.report.boundary_role_overlap_count
        for value in transitions.values()
    )
    maximum_mutation = max(
        value.report.input_state_mutation
        for value in transitions.values()
    )
    checks = {
        "actual_retiring_attachment_velocity_is_nontrivial": (
            release.report.maximum_speed
            >= float(thresholds["release_speed_abs_min"])
        ),
        "release_velocity_is_normal_and_full_rank": (
            release.report.maximum_tangential_speed
            <= float(
                thresholds[
                    "release_tangential_velocity_abs_max"
                ]
            )
            and release.report.rank_deficiency
            <= int(
                thresholds[
                    "release_projection_rank_deficiency_max"
                ]
            )
            and release.report.maximum_condition_number
            <= float(
                thresholds[
                    "release_projection_condition_number_max"
                ]
            )
        ),
        "naive_coincident_append_fails_closed": (
            naive_failure
            >= int(
                thresholds[
                    "naive_coincident_append_failure_count_min"
                ]
            )
        ),
        "half_and_full_newborn_areas_are_finite": (
            half.report.minimum_newborn_face_area
            >= float(
                thresholds["minimum_half_stage_face_area_min"]
            )
            and full.report.minimum_newborn_face_area
            >= float(
                thresholds["minimum_full_stage_face_area_min"]
            )
        ),
        "augmented_counts_and_integer_maps_are_exact": (
            count_mismatches == 0
            and maximum_injection
            <= float(thresholds["P1_injection_abs_max"])
            and maximum_p2_injection
            <= float(thresholds["P2_injection_abs_max"])
        ),
        "release_and_new_body_roles_are_disjoint": (
            maximum_overlap
            <= int(
                thresholds["boundary_role_overlap_count_max"]
            )
        ),
        "explicit_newborn_rows_and_seams_roundtrip": (
            maximum_trace_error
            <= float(thresholds["chronological_seam_abs_max"])
            and maximum_roundtrip
            <= float(thresholds["P2_roundtrip_abs_max"])
        ),
        "transition_is_rigid_frame_covariant": (
            rigid_geometry_error
            <= float(thresholds["rigid_geometry_abs_max"])
            and rigid_release_error
            <= float(
                thresholds["rigid_release_velocity_abs_max"]
            )
            and rigid_topology_mismatches
            <= int(
                thresholds[
                    "rigid_topology_mismatch_count_max"
                ]
            )
        ),
        "invalid_transitions_fail_closed": (
            invalid_failures
            >= int(
                thresholds[
                    "invalid_transition_failure_count_min"
                ]
            )
        ),
        "no_inference_offset_or_input_mutation_enters": (
            report.inferred_scalar_count
            <= int(thresholds["inferred_scalar_count_max"])
            and report.epsilon_offset
            <= float(thresholds["epsilon_offset_abs_max"])
            and maximum_mutation
            <= float(
                thresholds["input_state_mutation_abs_max"]
            )
        ),
    }
    checks = {name: bool(value) for name, value in checks.items()}
    result = {
        "artifact": "actual_wake_newborn_transition_oracle",
        "claim_node": contract["claim_node"],
        "stage": contract["stage"],
        "stage_decision": (
            "GO" if all(checks.values()) else "NO-GO"
        ),
        "checks": checks,
        "aggregate_metrics": {
            "initial_band_count": report.initial_band_count,
            "augmented_band_count": report.augmented_band_count,
            "initial_P1_dof_count": report.initial_p1_dof_count,
            "augmented_P1_dof_count": (
                report.augmented_p1_dof_count
            ),
            "initial_P2_dof_count": report.initial_p2_dof_count,
            "augmented_P2_dof_count": (
                report.augmented_p2_dof_count
            ),
            "release_speed_abs_min": (
                release.report.minimum_speed
            ),
            "release_speed_abs_max": (
                release.report.maximum_speed
            ),
            "release_tangential_velocity_abs_max": (
                release.report.maximum_tangential_speed
            ),
            "release_projection_condition_number_max": (
                release.report.maximum_condition_number
            ),
            "release_projection_fit_residual_fraction": (
                release.report.maximum_relative_fit_residual
            ),
            "naive_coincident_append_failure_count": (
                naive_failure
            ),
            "minimum_half_stage_face_area": (
                half.report.minimum_newborn_face_area
            ),
            "minimum_full_stage_face_area": (
                full.report.minimum_newborn_face_area
            ),
            "P1_injection_abs_max": maximum_injection,
            "P2_injection_abs_max": maximum_p2_injection,
            "chronological_seam_abs_max": maximum_trace_error,
            "P2_roundtrip_abs_max": maximum_roundtrip,
            "boundary_role_overlap_count": maximum_overlap,
            "rigid_geometry_abs_max": rigid_geometry_error,
            "rigid_release_velocity_abs_max": rigid_release_error,
            "rigid_topology_mismatch_count": (
                rigid_topology_mismatches
            ),
            "invalid_transition_failure_count": invalid_failures,
            "inferred_scalar_count": report.inferred_scalar_count,
            "epsilon_offset_abs_max": report.epsilon_offset,
            "input_state_mutation_abs_max": maximum_mutation,
        },
        "forbidden_quantities_absent": [
            "newborn_strength_inference",
            "scalar_average",
            "coordinate_welding",
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
