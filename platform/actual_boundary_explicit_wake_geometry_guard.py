"""Run the preregistered S3f explicit 3-D wake-geometry interface oracle."""
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
from claim_runtime.distributed_doublet import (  # noqa: E402
    MaterialWakeHistory,
    newborn_material_wake_band,
)
from claim_runtime.thick_body_neumann_shadow import (  # noqa: E402
    closed_triangular_mesh,
)


CASES = (
    HERE / "docs" / "diag"
    / "actual_boundary_explicit_wake_geometry_cases.yaml"
)
RESULTS = (
    HERE / "docs" / "diag"
    / "actual_boundary_explicit_wake_geometry_results.json"
)


def _rotation_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    direction = axis / np.linalg.norm(axis)
    x, y, z = direction
    cross = np.array(
        ((0.0, -z, y), (z, 0.0, -x), (-y, x, 0.0))
    )
    return (
        np.eye(3) * np.cos(angle)
        + (1.0 - np.cos(angle)) * np.outer(direction, direction)
        + np.sin(angle) * cross
    )


def _history(
    mesh,
    topology,
    *,
    curved: bool,
) -> MaterialWakeHistory:
    body_edge = mesh.vertices[
        topology.ordered_cut_vertex_indices
    ].copy()
    vertex_y = body_edge[:, 1]
    vertex_shape = 1.0 - vertex_y**2
    far_edge = body_edge.copy()
    far_edge[:, 0] = 2.0
    seam_edge = body_edge.copy()
    seam_edge[:, 0] = 1.5
    if curved:
        far_edge[:, 2] += 0.20 * vertex_shape
        seam_edge[:, 2] += 0.12 * vertex_shape
    cut_y = topology.cut_node_coordinates[:, 1]
    shape = 1.0 - cut_y**2
    old_rows = np.array(
        (0.20 * shape, 0.30 * shape, 0.40 * shape)
    )
    active_rows = np.array(
        (0.40 * shape, 0.10 * shape, np.zeros_like(shape))
    )
    old = newborn_material_wake_band(
        sheet_id="explicit-old-TEV",
        vortex_family="TEV",
        previous_edge=far_edge,
        current_edge=seam_edge,
        time_nodes=np.array((0.0, 0.5, 1.0)),
        potential_jump_rows=old_rows,
        span_diagonal_pattern="mirror_symmetric",
    )
    active = newborn_material_wake_band(
        sheet_id="explicit-active-TEV",
        vortex_family="TEV",
        previous_edge=seam_edge,
        current_edge=body_edge,
        time_nodes=np.array((1.0, 1.5, 2.0)),
        potential_jump_rows=active_rows,
        span_diagonal_pattern="mirror_symmetric",
    )
    return MaterialWakeHistory(
        "explicit-two-band-TEV-history",
        (old, active),
    )


def _transform_history(
    history: MaterialWakeHistory,
    rotation: np.ndarray,
    translation: np.ndarray,
    *,
    reverse_span: bool = False,
) -> MaterialWakeHistory:
    bands = []
    for band in history.bands:
        span_nodes = band.span_nodes
        previous = (
            band.surface.vertices[:span_nodes] @ rotation.T
            + translation
        )
        current = (
            band.surface.vertices[span_nodes:] @ rotation.T
            + translation
        )
        rows = band.potential_jump_rows.copy()
        if reverse_span:
            previous = previous[::-1]
            current = current[::-1]
            rows = rows[:, ::-1]
        bands.append(
            newborn_material_wake_band(
                sheet_id=band.sheet_id,
                vortex_family=band.vortex_family,
                previous_edge=previous,
                current_edge=current,
                time_nodes=band.time_nodes,
                potential_jump_rows=rows,
                span_diagonal_pattern="mirror_symmetric",
            )
        )
    return MaterialWakeHistory(history.history_id, tuple(bands))


def _max_abs(first, second) -> float:
    return float(
        np.max(
            np.abs(np.asarray(first) - np.asarray(second)),
            initial=0.0,
        )
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
    speed = float(canonical["freestream"]["speed"])
    alpha = np.deg2rad(
        float(canonical["freestream"]["alpha_deg"])
    )
    incident = np.repeat(
        np.array(
            ((speed * np.cos(alpha), 0.0, speed * np.sin(alpha)),)
        ),
        len(mesh.faces),
        axis=0,
    )
    order = int(canonical["spatial_quadrature_order"])
    straight_history = _history(mesh, topology, curved=False)
    curved_history = _history(mesh, topology, curved=True)
    straight_snapshot = [
        (
            band.surface.vertices.copy(),
            band.surface.faces.copy(),
            band.potential_jump_rows.copy(),
            band.time_nodes.copy(),
        )
        for band in straight_history.bands
    ]
    curved_snapshot = [
        (
            band.surface.vertices.copy(),
            band.surface.faces.copy(),
            band.potential_jump_rows.copy(),
            band.time_nodes.copy(),
        )
        for band in curved_history.bands
    ]
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
    explicit_straight = solve_actual_boundary_body_wake_p2(
        mesh,
        topology,
        incident_velocity=incident,
        downstream_edge_x=None,
        prescribed_wake_history=straight_history,
        target_quadrature_order=order,
        source_quadrature_order=order,
    )
    explicit_curved = solve_actual_boundary_body_wake_p2(
        mesh,
        topology,
        incident_velocity=incident,
        downstream_edge_x=None,
        prescribed_wake_history=curved_history,
        target_quadrature_order=order,
        source_quadrature_order=order,
    )

    rigid = canonical["rigid_frame_counterfactual"]
    rotation = _rotation_matrix(
        np.asarray(rigid["rotation_axis"], dtype=float),
        np.deg2rad(float(rigid["rotation_deg"])),
    )
    translation = np.asarray(rigid["translation"], dtype=float)
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
    original_cut_ids = topology.ordered_cut_vertex_indices
    moved_cut_ids = moved_topology.ordered_cut_vertex_indices
    if np.array_equal(moved_cut_ids, original_cut_ids):
        reverse_span = False
    elif np.array_equal(moved_cut_ids, original_cut_ids[::-1]):
        reverse_span = True
    else:
        raise RuntimeError(
            "rigid transform changed classified cut material identity"
        )
    moved_history = _transform_history(
        curved_history,
        rotation,
        translation,
        reverse_span=reverse_span,
    )
    moved_incident = incident @ rotation.T
    rigid_solution = solve_actual_boundary_body_wake_p2(
        moved_mesh,
        moved_topology,
        incident_velocity=moved_incident,
        downstream_edge_x=None,
        prescribed_wake_history=moved_history,
        target_quadrature_order=order,
        source_quadrature_order=order,
    )

    geometry_mutation = 0.0
    strength_mutation = 0.0
    for history, snapshots in (
        (straight_history, straight_snapshot),
        (curved_history, curved_snapshot),
    ):
        for band, (vertices, faces, rows, times) in zip(
            history.bands,
            snapshots,
        ):
            geometry_mutation = max(
                geometry_mutation,
                _max_abs(band.surface.vertices, vertices),
                float(
                    np.max(
                        np.abs(
                            band.surface.faces.astype(float)
                            - faces.astype(float)
                        ),
                        initial=0.0,
                    )
                ),
                _max_abs(band.time_nodes, times),
            )
            strength_mutation = max(
                strength_mutation,
                _max_abs(band.potential_jump_rows, rows),
            )
    output_geometry_difference = max(
        _max_abs(
            supplied.surface.vertices,
            solved.surface.vertices,
        )
        for supplied, solved in zip(
            curved_history.bands,
            explicit_curved.wake_history.bands,
        )
    )
    output_old_strength_difference = _max_abs(
        curved_history.bands[0].potential_jump_rows,
        explicit_curved.wake_history.bands[0].potential_jump_rows,
    )
    output_active_known_difference = _max_abs(
        curved_history.bands[1].potential_jump_rows[:2],
        explicit_curved.wake_history.bands[1].potential_jump_rows[:2],
    )
    geometry_mutation = max(
        geometry_mutation,
        output_geometry_difference,
    )
    strength_mutation = max(
        strength_mutation,
        output_old_strength_difference,
        output_active_known_difference,
    )

    solutions: tuple[ActualBoundaryBodyWakeSolution, ...] = (
        legacy,
        explicit_straight,
        explicit_curved,
        rigid_solution,
    )
    reports = [
        solution.wake_history.continuity_report()
        for solution in solutions
    ]
    time_gap = max(report.max_time_gap for report in reports)
    geometry_gap = max(
        report.max_geometry_gap for report in reports
    )
    trace_jump = max(report.max_trace_jump for report in reports)
    common_edges = {
        solution.body_wake_paired_topology_counts["common_edge"]
        for solution in solutions
    }
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
    attachment = max(
        solution.wake_attachment_error for solution in solutions
    )
    tip_jump = max(
        max(
            abs(float(solution.body_cut_jump[0])),
            abs(float(solution.body_cut_jump[-1])),
        )
        for solution in solutions
    )
    straight_matrix = _max_abs(
        legacy.matrix,
        explicit_straight.matrix,
    )
    straight_rhs = _max_abs(
        legacy.right_hand_side,
        explicit_straight.right_hand_side,
    )
    straight_potential = _max_abs(
        legacy.global_body_potential,
        explicit_straight.global_body_potential,
    )
    straight_jump = _max_abs(
        legacy.body_cut_jump,
        explicit_straight.body_cut_jump,
    )
    curved_difference = _max_abs(
        explicit_curved.body_cut_jump,
        explicit_straight.body_cut_jump,
    )
    rigid_matrix = _max_abs(
        explicit_curved.matrix,
        rigid_solution.matrix,
    )
    rigid_rhs = _max_abs(
        explicit_curved.right_hand_side,
        rigid_solution.right_hand_side,
    )
    rigid_jump = _max_abs(
        explicit_curved.body_cut_jump,
        (
            rigid_solution.body_cut_jump[::-1]
            if reverse_span
            else rigid_solution.body_cut_jump
        ),
    )
    checks = {
        "legacy_straight_path_is_preserved": (
            straight_matrix
            <= float(
                thresholds[
                    "straight_matrix_max_abs_difference"
                ]
            )
            and straight_rhs
            <= float(
                thresholds["straight_rhs_max_abs_difference"]
            )
            and straight_potential
            <= float(
                thresholds[
                    "straight_body_potential_max_abs_difference"
                ]
            )
            and straight_jump
            <= float(
                thresholds[
                    "straight_cut_jump_max_abs_difference"
                ]
            )
        ),
        "explicit_geometry_and_old_strength_are_immutable": (
            geometry_mutation
            <= float(
                thresholds["explicit_geometry_mutation_abs_max"]
            )
            and strength_mutation
            <= float(
                thresholds["explicit_strength_mutation_abs_max"]
            )
        ),
        "material_interfaces_are_exact": (
            all(report.compatible for report in reports)
            and time_gap
            <= float(thresholds["history_time_gap_abs_max"])
            and geometry_gap
            <= float(
                thresholds["history_geometry_gap_abs_max"]
            )
            and trace_jump
            <= float(thresholds["history_trace_jump_abs_max"])
        ),
        "declared_curved_geometry_is_consumed": (
            curved_difference
            >= float(
                thresholds[
                    "curved_vs_straight_cut_jump_difference_min"
                ]
            )
        ),
        "rigid_frame_objectivity_passes": (
            rigid_matrix
            <= float(
                thresholds["rigid_matrix_max_abs_difference"]
            )
            and rigid_rhs
            <= float(
                thresholds["rigid_rhs_max_abs_difference"]
            )
            and rigid_jump
            <= float(
                thresholds["rigid_cut_jump_max_abs_difference"]
            )
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
        "old_wake_is_known_not_unknown": (
            old_unknowns
            <= int(
                thresholds[
                    "independent_old_wake_unknown_count_max"
                ]
            )
        ),
        "all_systems_are_full_rank": (
            rank_deficiency
            <= int(thresholds["rank_deficiency_max"])
            and weak_residual
            <= float(thresholds["normalized_weak_residual_max"])
            and condition_number
            <= float(thresholds["condition_number_max"])
        ),
        "attachment_and_tip_identity_pass": (
            attachment
            <= float(thresholds["active_attachment_abs_max"])
            and tip_jump
            <= float(thresholds["tip_jump_abs_max"])
        ),
    }
    checks = {name: bool(value) for name, value in checks.items()}
    result = {
        "artifact": "actual_boundary_explicit_wake_geometry_oracle",
        "claim_node": contract["claim_node"],
        "stage": contract["stage"],
        "aggregate_metrics": {
            "straight_matrix_max_abs_difference": straight_matrix,
            "straight_rhs_max_abs_difference": straight_rhs,
            "straight_body_potential_max_abs_difference": (
                straight_potential
            ),
            "straight_cut_jump_max_abs_difference": straight_jump,
            "explicit_geometry_mutation_abs_max": geometry_mutation,
            "explicit_strength_mutation_abs_max": strength_mutation,
            "history_time_gap_abs_max": time_gap,
            "history_geometry_gap_abs_max": geometry_gap,
            "history_trace_jump_abs_max": trace_jump,
            "curved_vs_straight_cut_jump_difference": curved_difference,
            "rigid_matrix_max_abs_difference": rigid_matrix,
            "rigid_rhs_max_abs_difference": rigid_rhs,
            "rigid_cut_jump_max_abs_difference": rigid_jump,
            "rigid_cut_span_orientation_reversed": reverse_span,
            "body_wake_common_edge_pair_counts": sorted(common_edges),
            "independent_old_wake_unknown_count_max": old_unknowns,
            "rank_deficiency_max": rank_deficiency,
            "normalized_weak_residual_max": weak_residual,
            "condition_number_max": condition_number,
            "active_attachment_abs_max": attachment,
            "tip_jump_abs_max": tip_jump,
        },
        "checks": checks,
        "stage_decision": (
            "GO" if all(checks.values()) else "NO-GO"
        ),
        "production_activation_allowed": False,
        "forbidden_quantities_absent": [
            "wake_core",
            "smoothing",
            "pressure",
            "force",
            "LESP",
            "prescribed_current_circulation",
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
