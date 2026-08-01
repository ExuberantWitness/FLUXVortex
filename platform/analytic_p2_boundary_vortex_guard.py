"""Run the preregistered S3p analytic P2 boundary-vortex gate."""
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
    _analytic_quadratic_boundary_vortex_velocity,
)
from claim_runtime.thick_body_neumann_shadow import (  # noqa: E402
    closed_triangular_mesh,
)
from near_singular_p2_edge_quadrature_guard import (  # noqa: E402
    _last_change,
    _ring_velocity,
)


CASES = (
    HERE / "docs" / "diag"
    / "analytic_p2_boundary_vortex_cases.yaml"
)
PARENT_CASES = (
    HERE / "docs" / "diag"
    / "near_singular_p2_edge_quadrature_cases.yaml"
)
RESULTS = (
    HERE / "docs" / "diag"
    / "analytic_p2_boundary_vortex_results.json"
)
MODE = "target_sinh_analytic_boundary"


def _field(vector: np.ndarray) -> ExternalIncidentField:
    return ExternalIncidentField(
        "canonical-external",
        ("uniform_freestream",),
        lambda points: np.repeat(vector[None, :], len(points), axis=0),
    )


def _ledger(solution, contract, vector):
    query = wake_sheet_interior_query(solution.wake_history)
    orders = contract["canonical"]["actual_snapshot_orders"]
    ledger = evaluate_actual_body_wake_sheet_velocity(
        solution,
        query,
        external_incident=_field(vector),
        body_doublet_orders=tuple(orders["body"]),
        wake_sheet_average_orders=tuple(orders["wake"]),
        absolute_tolerance=1.0e-8,
        relative_tolerance=0.0,
        edge_quadrature=MODE,
    )
    return query, ledger


def _rotation_error(base, moved, rotation) -> float:
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
    parent = yaml.safe_load(
        PARENT_CASES.read_text(encoding="utf-8")
    )
    thresholds = contract["thresholds"]
    manufactured = parent["canonical"]["manufactured_triangle"]
    triangle = np.asarray(manufactured["vertices"], dtype=float)
    faces = np.array(((0, 1, 2),), dtype=np.int64)
    barycentric = np.asarray(
        manufactured["owner_barycentric"],
        dtype=float,
    )
    owner_points = barycentric @ triangle
    quadratic = QuadraticDoubletSurface(
        triangle,
        faces,
        np.asarray(
            manufactured["quadratic_mu_nodes"],
            dtype=float,
        )[None, :],
    )
    constant_strength = float(manufactured["constant_mu"])
    constant = QuadraticDoubletSurface(
        triangle,
        faces,
        np.full((1, 6), constant_strength),
    )

    element = quadratic.element(0)
    reconstruction_error = 0.0
    for start_index, end_index in ((0, 1), (1, 2), (2, 0)):
        start = triangle[start_index]
        end = triangle[end_index]
        nodes = np.vstack((start, 0.5 * (start + end), end))
        value_start, value_mid, value_end = element.evaluate(nodes)
        coefficient_a = 2.0 * (
            value_end + value_start - 2.0 * value_mid
        )
        coefficient_b = (
            4.0 * value_mid - 3.0 * value_start - value_end
        )
        coordinate = np.array((0.13, 0.37, 0.71, 0.91))
        points = (
            (1.0 - coordinate)[:, None] * start
            + coordinate[:, None] * end
        )
        reconstructed = (
            coefficient_a * coordinate**2
            + coefficient_b * coordinate
            + value_start
        )
        reconstruction_error = max(
            reconstruction_error,
            float(
                np.max(
                    np.abs(reconstructed - element.evaluate(points)),
                    initial=0.0,
                )
            ),
        )

    constant_actual = constant.induced_velocity_sheet_average(
        np.zeros(len(owner_points), dtype=np.int64),
        barycentric,
        quadrature_order=8,
        edge_quadrature=MODE,
    )
    constant_ring_error = float(
        np.max(
            np.abs(
                constant_actual
                - _ring_velocity(
                    triangle,
                    owner_points,
                    constant_strength,
                )
            ),
            initial=0.0,
        )
    )
    candidate_orders = tuple(contract["canonical"]["candidate_orders"])
    manufactured_values = [
        quadratic.induced_velocity_sheet_average(
            np.zeros(len(owner_points), dtype=np.int64),
            barycentric,
            quadrature_order=order,
            edge_quadrature=MODE,
        )
        for order in candidate_orders
    ]
    manufactured_abs, manufactured_rel = _last_change(
        manufactured_values,
        1.0e-8,
    )

    regular_point = np.array(((0.2, 0.3, 0.8),))
    regular_error = float(
        np.max(
            np.abs(
                quadratic.induced_velocity_line_reduced(
                    regular_point,
                    quadrature_order=24,
                    edge_quadrature=MODE,
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
    actual_body_values = [
        body.induced_velocity_line_reduced(
            query.points,
            quadrature_order=order,
            edge_quadrature=MODE,
        )
        for order in candidate_orders
    ]
    actual_wake_values = [
        assembly.induced_velocity_sheet_average(
            query.patch_indices,
            query.face_indices,
            query.barycentric,
            quadrature_order=order,
            edge_quadrature=MODE,
        )
        for order in candidate_orders
    ]
    actual_body_abs, actual_body_rel = _last_change(
        actual_body_values,
        1.0e-8,
    )
    actual_wake_abs, actual_wake_rel = _last_change(
        actual_wake_values,
        1.0e-8,
    )
    vector = np.array(
        (np.cos(np.deg2rad(5.0)), 0.0, np.sin(np.deg2rad(5.0)))
    )
    _, ledger = _ledger(solution, contract, vector)

    rigid = parent["canonical"]["rigid_frame_counterfactual"]
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
    _, moved_ledger = _ledger(
        moved_solution,
        contract,
        moved_vector,
    )
    rigid_error = _rotation_error(ledger, moved_ledger, rotation)

    edge_failures = 0
    for point in (
        np.array(((0.5, 0.0, 0.0),)),
        np.array(((0.0, 0.0, 0.0),)),
    ):
        try:
            _analytic_quadratic_boundary_vortex_velocity(
                element,
                0,
                1,
                point,
            )
        except DistributedDoubletError:
            edge_failures += 1

    candidate_abs = max(
        manufactured_abs,
        actual_body_abs,
        actual_wake_abs,
    )
    candidate_rel = max(
        manufactured_rel,
        actual_body_rel,
        actual_wake_rel,
    )
    checks = {
        "edge_polynomial_is_reconstructed_exactly": (
            reconstruction_error
            <= float(
                thresholds[
                    "edge_polynomial_reconstruction_abs_max"
                ]
            )
        ),
        "constant_strength_is_exact_ring": (
            constant_ring_error
            <= float(thresholds["constant_ring_abs_max"])
        ),
        "manufactured_near_vertex_reaches_cauchy": (
            candidate_abs
            <= float(thresholds["candidate_last_abs_change_max"])
            or candidate_rel
            <= float(thresholds["candidate_last_rel_change_max"])
        ),
        "regular_field_is_unchanged": (
            regular_error
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
        "edge_targets_fail_closed": (
            edge_failures
            >= int(thresholds["edge_failure_count_min"])
        ),
    }
    checks = {name: bool(value) for name, value in checks.items()}
    result = {
        "artifact": "analytic_p2_boundary_vortex_oracle",
        "stage": contract["stage"],
        "claim_node": contract["claim_node"],
        "stage_decision": (
            "GO" if all(checks.values()) else "NO-GO"
        ),
        "checks": checks,
        "aggregate_metrics": {
            "edge_polynomial_reconstruction_abs_max": (
                reconstruction_error
            ),
            "constant_ring_abs_max": constant_ring_error,
            "candidate_last_abs_change_max": candidate_abs,
            "candidate_last_rel_change_max": candidate_rel,
            "manufactured_candidate_last_abs_change": (
                manufactured_abs
            ),
            "actual_body_candidate_last_abs_change": (
                actual_body_abs
            ),
            "actual_wake_candidate_last_abs_change": (
                actual_wake_abs
            ),
            "regular_field_equivalence_abs_max": regular_error,
            "actual_body_frozen_order_abs_change": (
                ledger.body_doublet_report.max_abs_change
            ),
            "actual_wake_frozen_order_abs_change": (
                ledger.wake_sheet_average_report.max_abs_change
            ),
            "rigid_channel_abs_max": rigid_error,
            "edge_failure_count": edge_failures,
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
