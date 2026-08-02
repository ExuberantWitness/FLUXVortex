"""Run the preregistered S3o target-centered edge-quadrature gate."""
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
from claim_runtime.actual_body_wake_velocity import (  # noqa: E402
    ExternalIncidentField,
    evaluate_actual_body_wake_sheet_velocity,
    material_wake_assembly,
    wake_sheet_interior_query,
)
from claim_runtime.actual_boundary_body_wake import (  # noqa: E402
    MaterialWakeCutAttachment,
    solve_actual_boundary_body_wake_p2,
)
from claim_runtime.classified_p2_cut_topology import (  # noqa: E402
    classified_p2_cut_topology,
)
from claim_runtime.distributed_doublet import (  # noqa: E402
    DistributedDoubletError,
    QuadraticDoubletSurface,
    _target_sinh_line_quadrature,
)
from claim_runtime.thick_body_neumann_shadow import (  # noqa: E402
    closed_triangular_mesh,
)


CASES = (
    HERE / "docs" / "diag"
    / "near_singular_p2_edge_quadrature_cases.yaml"
)
RESULTS = (
    HERE / "docs" / "diag"
    / "near_singular_p2_edge_quadrature_results.json"
)


def _segment_velocity(
    points: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
    strength: float,
) -> np.ndarray:
    r1 = points - start
    r2 = points - end
    filament = end - start
    cross = np.cross(r1, r2)
    denominator = np.einsum("ij,ij->i", cross, cross)
    direction = (
        r1 / np.linalg.norm(r1, axis=1)[:, None]
        - r2 / np.linalg.norm(r2, axis=1)[:, None]
    )
    coefficient = direction @ filament
    return (
        strength
        * cross
        * coefficient[:, None]
        / (4.0 * np.pi * denominator[:, None])
    )


def _ring_velocity(
    triangle: np.ndarray,
    points: np.ndarray,
    strength: float,
) -> np.ndarray:
    result = np.zeros_like(points)
    for start, end in ((0, 1), (1, 2), (2, 0)):
        result += _segment_velocity(
            points,
            triangle[start],
            triangle[end],
            strength,
        )
    return result


def _last_change(
    values: list[np.ndarray],
    absolute_floor: float,
) -> tuple[float, float]:
    difference = np.linalg.norm(values[-1] - values[-2], axis=1)
    scale = np.maximum(
        np.linalg.norm(values[-1], axis=1),
        absolute_floor,
    )
    return (
        float(np.max(difference, initial=0.0)),
        float(np.max(difference / scale, initial=0.0)),
    )


def _field(vector: np.ndarray) -> ExternalIncidentField:
    return ExternalIncidentField(
        "canonical-external",
        ("uniform_freestream",),
        lambda points: np.repeat(vector[None, :], len(points), axis=0),
    )


def _candidate_ledger(solution, contract, vector):
    query = wake_sheet_interior_query(solution.wake_history)
    snapshot = contract["canonical"]["actual_snapshot_orders"]
    ledger = evaluate_actual_body_wake_sheet_velocity(
        solution,
        query,
        external_incident=_field(vector),
        body_doublet_orders=tuple(snapshot["body"]),
        wake_sheet_average_orders=tuple(snapshot["wake"]),
        absolute_tolerance=1.0e-8,
        relative_tolerance=1.0e-6,
        edge_quadrature="target_sinh",
    )
    return query, ledger


def _maximum_rotation_error(base, moved, rotation) -> float:
    return max(
        float(
            np.max(
                np.abs(
                    getattr(moved, name)
                    - getattr(base, name) @ rotation.T
                ),
                initial=0.0,
            )
        )
        for name in (
            "external_incident",
            "body_source",
            "body_doublet",
            "wake_sheet_average",
            "total",
        )
    )


def run() -> dict:
    contract = yaml.safe_load(CASES.read_text(encoding="utf-8"))
    canonical = contract["canonical"]
    thresholds = contract["thresholds"]
    manufactured = canonical["manufactured_triangle"]
    triangle = np.asarray(manufactured["vertices"], dtype=float)
    barycentric = np.asarray(
        manufactured["owner_barycentric"],
        dtype=float,
    )
    face = np.array([[0, 1, 2]], dtype=np.int64)
    quadratic = QuadraticDoubletSurface(
        triangle,
        face,
        np.asarray(
            manufactured["quadratic_mu_nodes"],
            dtype=float,
        )[None, :],
    )
    constant_strength = float(manufactured["constant_mu"])
    constant = QuadraticDoubletSurface(
        triangle,
        face,
        np.full((1, 6), constant_strength),
    )

    measure_error = 0.0
    for target in (
        np.array((0.3, 0.01, 0.0)),
        np.array((1.01, 0.001, 0.0)),
        np.array((0.5, 0.5, 0.2)),
    ):
        coordinate, weights = _target_sinh_line_quadrature(
            triangle[0],
            triangle[1],
            target,
            order=24,
        )
        measure_error = max(
            measure_error,
            abs(float(np.sum(weights)) - 1.0),
            abs(float(coordinate @ weights) - 0.5),
        )

    owner_points = barycentric @ triangle
    height = np.asarray(
        manufactured["off_sheet_height_over_sqrt_2_area"],
        dtype=float,
    )
    off_points = owner_points[: len(height)].copy()
    off_points[:, 2] += height
    constant_owner = constant.induced_velocity_sheet_average(
        np.zeros(len(owner_points), dtype=np.int64),
        barycentric,
        quadrature_order=24,
        edge_quadrature="target_sinh",
    )
    constant_off = constant.induced_velocity_line_reduced(
        off_points,
        quadrature_order=24,
        edge_quadrature="target_sinh",
    )
    constant_ring_error = max(
        float(
            np.max(
                np.abs(
                    constant_owner
                    - _ring_velocity(
                        triangle,
                        owner_points,
                        constant_strength,
                    )
                ),
                initial=0.0,
            )
        ),
        float(
            np.max(
                np.abs(
                    constant_off
                    - _ring_velocity(
                        triangle,
                        off_points,
                        constant_strength,
                    )
                ),
                initial=0.0,
            )
        ),
    )

    candidate_orders = tuple(canonical["candidate_orders"])
    manufactured_candidate = [
        quadratic.induced_velocity_sheet_average(
            np.zeros(len(owner_points), dtype=np.int64),
            barycentric,
            quadrature_order=order,
            edge_quadrature="target_sinh",
        )
        for order in candidate_orders
    ]
    manufactured_candidate_abs, manufactured_candidate_rel = (
        _last_change(manufactured_candidate, 1.0e-8)
    )
    reference_orders = tuple(
        canonical["independent_reference_orders"]
    )
    manufactured_reference = [
        quadratic.induced_velocity_sheet_average(
            np.zeros(len(owner_points), dtype=np.int64),
            barycentric,
            quadrature_order=order,
            edge_quadrature="standard",
        )
        for order in reference_orders
    ]
    manufactured_reference_abs, _ = _last_change(
        manufactured_reference,
        1.0e-8,
    )
    manufactured_candidate_reference_error = float(
        np.max(
            np.abs(
                manufactured_candidate[-1]
                - manufactured_reference[-1]
            ),
            initial=0.0,
        )
    )
    regular_point = np.array(((0.2, 0.3, 0.8),))
    regular_equivalence_error = float(
        np.max(
            np.abs(
                quadratic.induced_velocity_line_reduced(
                    regular_point,
                    quadrature_order=24,
                    edge_quadrature="target_sinh",
                )
                - quadratic.induced_velocity_line_reduced(
                    regular_point,
                    quadrature_order=64,
                    edge_quadrature="standard",
                )
            ),
            initial=0.0,
        )
    )

    (
        mesh,
        topology,
        upper,
        lower,
        cut_edges,
        endpoints,
        curved,
        attachment,
        solution,
    ) = _canonical_state()
    query = wake_sheet_interior_query(solution.wake_history)
    body = QuadraticDoubletSurface(
        solution.mesh.vertices,
        solution.mesh.faces,
        solution.body_face_potential,
    )
    assembly = material_wake_assembly(solution.wake_history)
    actual_candidate_body = [
        body.induced_velocity_line_reduced(
            query.points,
            quadrature_order=order,
            edge_quadrature="target_sinh",
        )
        for order in candidate_orders
    ]
    actual_candidate_wake = [
        assembly.induced_velocity_sheet_average(
            query.patch_indices,
            query.face_indices,
            query.barycentric,
            quadrature_order=order,
            edge_quadrature="target_sinh",
        )
        for order in candidate_orders
    ]
    actual_candidate_body_abs, actual_candidate_body_rel = (
        _last_change(actual_candidate_body, 1.0e-8)
    )
    actual_candidate_wake_abs, actual_candidate_wake_rel = (
        _last_change(actual_candidate_wake, 1.0e-8)
    )
    vector = np.array(
        (np.cos(np.deg2rad(5.0)), 0.0, np.sin(np.deg2rad(5.0)))
    )
    _, ledger = _candidate_ledger(solution, contract, vector)

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
    moved_history = _transform_history(
        curved,
        rotation,
        translation,
    )
    moved_attachment = MaterialWakeCutAttachment(
        topology.ordered_cut_vertex_indices,
        1,
    )
    moved_vector = vector @ rotation.T
    moved_solution = solve_actual_boundary_body_wake_p2(
        moved_mesh,
        moved_topology,
        incident_velocity=np.repeat(
            moved_vector[None, :],
            len(moved_mesh.faces),
            axis=0,
        ),
        downstream_edge_x=None,
        prescribed_wake_history=moved_history,
        prescribed_wake_attachment=moved_attachment,
        target_quadrature_order=10,
        source_quadrature_order=10,
    )
    _, moved_ledger = _candidate_ledger(
        moved_solution,
        contract,
        moved_vector,
    )
    rigid_error = _maximum_rotation_error(
        ledger,
        moved_ledger,
        rotation,
    )

    on_edge_failures = 0
    for target in (
        np.array((0.5, 0.0, 0.0)),
        np.array((0.0, 0.0, 0.0)),
    ):
        try:
            _target_sinh_line_quadrature(
                triangle[0],
                triangle[1],
                target,
                order=12,
            )
        except DistributedDoubletError:
            on_edge_failures += 1
    invalid_geometry_failures = 0
    for start, end, target in (
        (triangle[0], triangle[0], np.array((0.2, 0.3, 0.1))),
        (triangle[0], triangle[1], np.array((np.nan, 0.3, 0.1))),
    ):
        try:
            _target_sinh_line_quadrature(
                start,
                end,
                target,
                order=12,
            )
        except DistributedDoubletError:
            invalid_geometry_failures += 1

    candidate_abs = max(
        manufactured_candidate_abs,
        actual_candidate_body_abs,
        actual_candidate_wake_abs,
    )
    candidate_rel = max(
        manufactured_candidate_rel,
        actual_candidate_body_rel,
        actual_candidate_wake_rel,
    )
    checks = {
        "transformed_measure_is_exact": (
            measure_error
            <= float(thresholds["transformed_measure_abs_max"])
        ),
        "constant_doublet_matches_analytic_ring": (
            constant_ring_error
            <= float(thresholds["constant_ring_abs_max"])
        ),
        "candidate_reaches_finite_order_cauchy": (
            candidate_abs
            <= float(thresholds["candidate_last_abs_change_max"])
            or candidate_rel
            <= float(thresholds["candidate_last_rel_change_max"])
        ),
        "independent_reference_is_resolved": (
            manufactured_reference_abs
            <= float(
                thresholds[
                    "independent_reference_cauchy_abs_max"
                ]
            )
        ),
        "candidate_matches_independent_reference": (
            manufactured_candidate_reference_error
            <= float(
                thresholds["candidate_vs_reference_abs_max"]
            )
        ),
        "regular_field_is_unchanged": (
            regular_equivalence_error
            <= float(
                thresholds["regular_field_equivalence_abs_max"]
            )
        ),
        "actual_snapshot_frozen_orders_converge": (
            ledger.body_doublet_report.converged
            and ledger.wake_sheet_average_report.converged
            and ledger.body_doublet_report.max_abs_change
            <= float(
                thresholds["actual_body_last_abs_change_max"]
            )
            and ledger.wake_sheet_average_report.max_abs_change
            <= float(
                thresholds["actual_wake_last_abs_change_max"]
            )
        ),
        "actual_ledger_remains_exact": (
            ledger.closure_error() <= 2.0e-12
            and ledger.wake_representation_error <= 2.0e-12
        ),
        "candidate_is_rigid_objective": (
            rigid_error
            <= float(thresholds["rigid_channel_abs_max"])
        ),
        "edge_and_geometry_fail_closed": (
            on_edge_failures
            >= int(thresholds["on_edge_failure_count_min"])
            and invalid_geometry_failures
            >= int(
                thresholds["invalid_geometry_failure_count_min"]
            )
        ),
    }
    checks = {key: bool(value) for key, value in checks.items()}
    result = {
        "artifact": "near_singular_p2_edge_quadrature_oracle",
        "stage": contract["stage"],
        "claim_node": contract["claim_node"],
        "stage_decision": (
            "GO" if all(checks.values()) else "NO-GO"
        ),
        "checks": checks,
        "aggregate_metrics": {
            "transformed_measure_abs_max": measure_error,
            "constant_ring_abs_max": constant_ring_error,
            "candidate_last_abs_change_max": candidate_abs,
            "candidate_last_rel_change_max": candidate_rel,
            "manufactured_candidate_last_abs_change": (
                manufactured_candidate_abs
            ),
            "actual_body_candidate_last_abs_change": (
                actual_candidate_body_abs
            ),
            "actual_wake_candidate_last_abs_change": (
                actual_candidate_wake_abs
            ),
            "independent_reference_cauchy_abs_max": (
                manufactured_reference_abs
            ),
            "candidate_vs_reference_abs_max": (
                manufactured_candidate_reference_error
            ),
            "regular_field_equivalence_abs_max": (
                regular_equivalence_error
            ),
            "actual_body_frozen_order_abs_change": (
                ledger.body_doublet_report.max_abs_change
            ),
            "actual_wake_frozen_order_abs_change": (
                ledger.wake_sheet_average_report.max_abs_change
            ),
            "rigid_channel_abs_max": rigid_error,
            "on_edge_failure_count": on_edge_failures,
            "invalid_geometry_failure_count": (
                invalid_geometry_failures
            ),
        },
        "forbidden_quantities_absent": contract["forbidden"],
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
