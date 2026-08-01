"""Run the preregistered S3g oriented material-attachment oracle."""
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
from actual_boundary_explicit_wake_geometry_guard import (  # noqa: E402
    _history,
    _max_abs,
    _rotation_matrix,
    _transform_history,
)
from claim_runtime.actual_boundary_body_wake import (  # noqa: E402
    MaterialWakeCutAttachment,
    solve_actual_boundary_body_wake_p2,
)
from claim_runtime.classified_p2_cut_topology import (  # noqa: E402
    ClassifiedP2CutError,
    classified_p2_cut_topology,
)
from claim_runtime.distributed_doublet import (  # noqa: E402
    MaterialWakeHistory,
    newborn_material_wake_band,
)
from claim_runtime.thick_body_neumann_shadow import (  # noqa: E402
    closed_triangular_mesh,
)


CASES = (
    HERE / "docs" / "diag"
    / "actual_boundary_oriented_wake_attachment_cases.yaml"
)
RESULTS = (
    HERE / "docs" / "diag"
    / "actual_boundary_oriented_wake_attachment_results.json"
)


def _reverse_parameterization(
    history: MaterialWakeHistory,
) -> MaterialWakeHistory:
    bands = []
    for band in history.bands:
        count = band.span_nodes
        bands.append(
            newborn_material_wake_band(
                sheet_id=band.sheet_id,
                vortex_family=band.vortex_family,
                previous_edge=band.surface.vertices[:count][::-1],
                current_edge=band.surface.vertices[count:][::-1],
                time_nodes=band.time_nodes,
                potential_jump_rows=(
                    -band.potential_jump_rows[:, ::-1]
                ),
                span_diagonal_pattern="mirror_symmetric",
            )
        )
    return MaterialWakeHistory(history.history_id, tuple(bands))


def _oriented_attachment_error(
    solution,
    attachment: MaterialWakeCutAttachment,
) -> float:
    permutation = attachment.p2_trace_permutation(
        solution.topology
    )
    expected = (
        attachment.wake_jump_from_body_cut_sign
        * solution.body_cut_jump[permutation]
    )
    return _max_abs(
        solution.wake.potential_jump_rows[-1],
        expected,
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
    angle = np.deg2rad(5.0)
    incident = np.repeat(
        np.array(((np.cos(angle), 0.0, np.sin(angle)),)),
        len(mesh.faces),
        axis=0,
    )
    order = 10
    straight_history = _history(mesh, topology, curved=False)
    curved_history = _history(mesh, topology, curved=True)
    material_ids = np.asarray(
        canonical["forward_attachment"][
            "ordered_body_cut_vertex_indices"
        ],
        dtype=np.int64,
    )
    forward_attachment = MaterialWakeCutAttachment(
        material_ids,
        int(
            canonical["forward_attachment"][
                "wake_jump_from_body_cut_sign"
            ]
        ),
    )
    legacy = solve_actual_boundary_body_wake_p2(
        mesh,
        topology,
        incident_velocity=incident,
        downstream_edge_x=2.0,
        wake_edge_x_nodes=np.array((1.0, 1.5, 2.0)),
        fixed_old_wake_rows=np.array(
            (straight_history.bands[0].potential_jump_rows,)
        ),
        active_known_rows=(
            straight_history.bands[1].potential_jump_rows[:2]
        ),
        target_quadrature_order=order,
        source_quadrature_order=order,
    )
    straight = solve_actual_boundary_body_wake_p2(
        mesh,
        topology,
        incident_velocity=incident,
        downstream_edge_x=None,
        prescribed_wake_history=straight_history,
        prescribed_wake_attachment=forward_attachment,
        target_quadrature_order=order,
        source_quadrature_order=order,
    )
    curved_snapshot = [
        (
            band.surface.vertices.copy(),
            band.potential_jump_rows.copy(),
        )
        for band in curved_history.bands
    ]
    curved = solve_actual_boundary_body_wake_p2(
        mesh,
        topology,
        incident_velocity=incident,
        downstream_edge_x=None,
        prescribed_wake_history=curved_history,
        prescribed_wake_attachment=forward_attachment,
        target_quadrature_order=order,
        source_quadrature_order=order,
    )

    rigid_spec = canonical["generic_rigid_transform"]
    rotation = _rotation_matrix(
        np.asarray(rigid_spec["rotation_axis"], dtype=float),
        np.deg2rad(float(rigid_spec["rotation_deg"])),
    )
    translation = np.asarray(
        rigid_spec["translation"],
        dtype=float,
    )
    moved_mesh = closed_triangular_mesh(
        mesh.vertices @ rotation.T + translation,
        mesh.faces,
    )
    moved_topology = classified_p2_cut_topology(
        moved_mesh,
        upper_face_indices=upper,
        lower_face_indices=lower,
        cut_edges=cut_edges,
        zero_jump_end_vertices=endpoints,
    )
    coordinate_order_reversed = np.array_equal(
        moved_topology.ordered_cut_vertex_indices,
        topology.ordered_cut_vertex_indices[::-1],
    )
    moved_history = _transform_history(
        curved_history,
        rotation,
        translation,
        reverse_span=False,
    )
    moved = solve_actual_boundary_body_wake_p2(
        moved_mesh,
        moved_topology,
        incident_velocity=incident @ rotation.T,
        downstream_edge_x=None,
        prescribed_wake_history=moved_history,
        prescribed_wake_attachment=forward_attachment,
        target_quadrature_order=order,
        source_quadrature_order=order,
    )

    reversed_history = _reverse_parameterization(curved_history)
    reverse_spec = canonical["reverse_parameterization"]
    reverse_attachment = MaterialWakeCutAttachment(
        np.asarray(
            reverse_spec["ordered_body_cut_vertex_indices"],
            dtype=np.int64,
        ),
        int(reverse_spec["wake_jump_from_body_cut_sign"]),
    )
    reversed_solution = solve_actual_boundary_body_wake_p2(
        mesh,
        topology,
        incident_velocity=incident,
        downstream_edge_x=None,
        prescribed_wake_history=reversed_history,
        prescribed_wake_attachment=reverse_attachment,
        target_quadrature_order=order,
        source_quadrature_order=order,
    )

    invalid_failures = []
    invalid_builders = (
        lambda: MaterialWakeCutAttachment(
            np.array((2, 2, 10, 14, 18)),
            1,
        ),
        lambda: MaterialWakeCutAttachment(material_ids, 0),
    )
    for builder in invalid_builders:
        try:
            builder()
        except ClassifiedP2CutError:
            invalid_failures.append(True)
        else:
            invalid_failures.append(False)
    try:
        invalid_chain = MaterialWakeCutAttachment(
            np.array((2, 6, 14, 10, 18)),
            1,
        )
        solve_actual_boundary_body_wake_p2(
            mesh,
            topology,
            incident_velocity=incident,
            downstream_edge_x=None,
            prescribed_wake_history=curved_history,
            prescribed_wake_attachment=invalid_chain,
            target_quadrature_order=order,
            source_quadrature_order=order,
        )
    except ClassifiedP2CutError:
        invalid_failures.append(True)
    else:
        invalid_failures.append(False)

    geometry_mutation = max(
        _max_abs(band.surface.vertices, vertices)
        for band, (vertices, _) in zip(
            curved_history.bands,
            curved_snapshot,
        )
    )
    strength_mutation = max(
        _max_abs(band.potential_jump_rows, rows)
        for band, (_, rows) in zip(
            curved_history.bands,
            curved_snapshot,
        )
    )
    straight_matrix = _max_abs(legacy.matrix, straight.matrix)
    straight_rhs = _max_abs(
        legacy.right_hand_side,
        straight.right_hand_side,
    )
    straight_jump = _max_abs(
        legacy.body_cut_jump,
        straight.body_cut_jump,
    )
    curved_difference = _max_abs(
        curved.body_cut_jump,
        straight.body_cut_jump,
    )
    rigid_matrix = _max_abs(curved.matrix, moved.matrix)
    rigid_rhs = _max_abs(
        curved.right_hand_side,
        moved.right_hand_side,
    )
    moved_jump_in_material_order = moved.body_cut_jump[
        forward_attachment.p2_trace_permutation(moved_topology)
    ]
    curved_jump_in_material_order = curved.body_cut_jump[
        forward_attachment.p2_trace_permutation(topology)
    ]
    rigid_jump = _max_abs(
        curved_jump_in_material_order,
        moved_jump_in_material_order,
    )
    reverse_matrix = _max_abs(
        curved.matrix,
        reversed_solution.matrix,
    )
    reverse_rhs = _max_abs(
        curved.right_hand_side,
        reversed_solution.right_hand_side,
    )
    reverse_jump = _max_abs(
        curved.body_cut_jump,
        reversed_solution.body_cut_jump,
    )
    solutions = (straight, curved, moved, reversed_solution)
    attachments = (
        forward_attachment,
        forward_attachment,
        forward_attachment,
        reverse_attachment,
    )
    oriented_attachment = max(
        _oriented_attachment_error(solution, attachment)
        for solution, attachment in zip(solutions, attachments)
    )
    reports = [
        solution.wake_history.continuity_report()
        for solution in solutions
    ]
    interface_error = max(
        max(
            report.max_time_gap,
            report.max_geometry_gap,
            report.max_trace_jump,
        )
        for report in reports
    )
    common_edges = {
        solution.body_wake_paired_topology_counts["common_edge"]
        for solution in solutions
    }
    old_unknowns = max(
        solution.independent_old_wake_unknown_count
        for solution in solutions
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
    tip_jump = max(
        max(
            abs(float(solution.body_cut_jump[0])),
            abs(float(solution.body_cut_jump[-1])),
        )
        for solution in solutions
    )
    checks = {
        "forward_attachment_preserves_legacy": (
            straight_matrix
            <= float(
                thresholds[
                    "straight_legacy_matrix_max_abs_difference"
                ]
            )
            and straight_rhs
            <= float(
                thresholds[
                    "straight_legacy_rhs_max_abs_difference"
                ]
            )
            and straight_jump
            <= float(
                thresholds[
                    "straight_legacy_cut_jump_max_abs_difference"
                ]
            )
        ),
        "curved_geometry_is_consumed_without_mutation": (
            geometry_mutation
            <= float(
                thresholds["curved_geometry_mutation_abs_max"]
            )
            and strength_mutation
            <= float(
                thresholds["curved_strength_mutation_abs_max"]
            )
            and curved_difference
            >= float(
                thresholds[
                    "curved_vs_straight_cut_jump_difference_min"
                ]
            )
        ),
        "generic_rigid_objectivity_passes": (
            coordinate_order_reversed
            is bool(
                rigid_spec[
                    "expected_coordinate_cut_order_reversal"
                ]
            )
            and rigid_matrix
            <= float(
                thresholds["rigid_matrix_max_abs_difference"]
            )
            and rigid_rhs
            <= float(thresholds["rigid_rhs_max_abs_difference"])
            and rigid_jump
            <= float(
                thresholds["rigid_cut_jump_max_abs_difference"]
            )
        ),
        "reverse_parameterization_gauge_passes": (
            reverse_matrix
            <= float(
                thresholds[
                    "reverse_gauge_matrix_max_abs_difference"
                ]
            )
            and reverse_rhs
            <= float(
                thresholds[
                    "reverse_gauge_rhs_max_abs_difference"
                ]
            )
            and reverse_jump
            <= float(
                thresholds[
                    "reverse_gauge_cut_jump_max_abs_difference"
                ]
            )
        ),
        "oriented_attachment_identity_is_exact": (
            oriented_attachment
            <= float(
                thresholds[
                    "active_oriented_attachment_abs_max"
                ]
            )
        ),
        "material_interfaces_are_exact": (
            all(report.compatible for report in reports)
            and interface_error
            <= float(thresholds["history_interface_abs_max"])
        ),
        "only_newest_band_has_body_common_edges": (
            common_edges
            == {
                int(
                    thresholds[
                        "body_wake_common_edge_pair_count"
                    ]
                )
            }
        ),
        "all_systems_are_full_rank": (
            old_unknowns
            <= int(
                thresholds[
                    "independent_old_wake_unknown_count_max"
                ]
            )
            and rank_deficiency
            <= int(thresholds["rank_deficiency_max"])
            and weak_residual
            <= float(thresholds["normalized_weak_residual_max"])
            and condition_number
            <= float(thresholds["condition_number_max"])
        ),
        "tip_jump_is_zero": (
            tip_jump <= float(thresholds["tip_jump_abs_max"])
        ),
        "invalid_material_identities_fail_closed": all(
            invalid_failures
        ),
    }
    checks = {name: bool(value) for name, value in checks.items()}
    result = {
        "artifact": "actual_boundary_oriented_wake_attachment_oracle",
        "claim_node": contract["claim_node"],
        "stage": contract["stage"],
        "aggregate_metrics": {
            "straight_legacy_matrix_max_abs_difference": (
                straight_matrix
            ),
            "straight_legacy_rhs_max_abs_difference": straight_rhs,
            "straight_legacy_cut_jump_max_abs_difference": (
                straight_jump
            ),
            "curved_geometry_mutation_abs_max": geometry_mutation,
            "curved_strength_mutation_abs_max": strength_mutation,
            "curved_vs_straight_cut_jump_difference": (
                curved_difference
            ),
            "rigid_coordinate_cut_order_reversed": (
                coordinate_order_reversed
            ),
            "rigid_matrix_max_abs_difference": rigid_matrix,
            "rigid_rhs_max_abs_difference": rigid_rhs,
            "rigid_cut_jump_max_abs_difference": rigid_jump,
            "reverse_gauge_matrix_max_abs_difference": reverse_matrix,
            "reverse_gauge_rhs_max_abs_difference": reverse_rhs,
            "reverse_gauge_cut_jump_max_abs_difference": reverse_jump,
            "active_oriented_attachment_abs_max": (
                oriented_attachment
            ),
            "history_interface_abs_max": interface_error,
            "body_wake_common_edge_pair_counts": sorted(common_edges),
            "independent_old_wake_unknown_count_max": old_unknowns,
            "rank_deficiency_max": rank_deficiency,
            "normalized_weak_residual_max": weak_residual,
            "condition_number_max": condition_number,
            "tip_jump_abs_max": tip_jump,
            "invalid_identity_failure_count": sum(
                invalid_failures
            ),
        },
        "checks": checks,
        "stage_decision": (
            "GO" if all(checks.values()) else "NO-GO"
        ),
        "production_activation_allowed": False,
        "forbidden_quantities_absent": [
            "coordinate_sort_attachment",
            "proximity_welding",
            "post_solve_flip",
            "smoothing",
            "wake_core",
            "pressure",
            "force",
            "LESP",
            "target_load",
        ],
    }
    RESULTS.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, sort_keys=True))

