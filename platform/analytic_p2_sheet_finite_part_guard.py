"""Run the preregistered S3q analytic sheet finite-part gate."""
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
    _finite_part_edge_moments,
)
from claim_runtime.thick_body_neumann_shadow import (  # noqa: E402
    closed_triangular_mesh,
)
from near_singular_p2_edge_quadrature_guard import (  # noqa: E402
    _ring_velocity,
)


CASES = (
    HERE / "docs" / "diag"
    / "analytic_p2_sheet_finite_part_cases.yaml"
)
PARENT_CASES = (
    HERE / "docs" / "diag"
    / "near_singular_p2_edge_quadrature_cases.yaml"
)
RESULTS = (
    HERE / "docs" / "diag"
    / "analytic_p2_sheet_finite_part_results.json"
)
MODE = "target_sinh_analytic_sheet"


def _field(vector: np.ndarray) -> ExternalIncidentField:
    return ExternalIncidentField(
        "canonical-external",
        ("uniform_freestream",),
        lambda points: np.repeat(vector[None, :], len(points), axis=0),
    )


def _ledger(solution, contract, vector):
    query = __import__(
        "claim_runtime.actual_body_wake_velocity",
        fromlist=["wake_sheet_interior_query"],
    ).wake_sheet_interior_query(solution.wake_history)
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
    return ledger


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


def _moment_derivative_relative_error() -> float:
    distance = 0.2
    length_reference = 0.8
    lower = -1.3
    step = 2.0e-6
    maximum = 0.0
    for value in (-0.7, -0.1, 0.4, 0.9):
        plus = np.asarray(
            _finite_part_edge_moments(
                lower,
                value + step,
                distance,
                length_reference,
            )
        )
        minus = np.asarray(
            _finite_part_edge_moments(
                lower,
                value - step,
                distance,
                length_reference,
            )
        )
        derivative = (plus - minus) / (2.0 * step)
        radius = np.hypot(value, distance)
        expected = np.array(
            (
                1.0 / radius**3,
                value / radius**3,
                value**2 / radius**3,
                np.log(radius / length_reference) / radius**3,
                value
                * np.log(radius / length_reference)
                / radius**3,
            )
        )
        maximum = max(
            maximum,
            float(
                np.max(
                    np.abs(derivative - expected)
                    / np.maximum(np.abs(expected), 1.0e-10),
                    initial=0.0,
                )
            ),
        )
    return maximum


def run() -> dict:
    contract = yaml.safe_load(CASES.read_text(encoding="utf-8"))
    parent = yaml.safe_load(
        PARENT_CASES.read_text(encoding="utf-8")
    )
    thresholds = contract["thresholds"]
    manufactured = parent["canonical"]["manufactured_triangle"]
    triangle = np.asarray(manufactured["vertices"], dtype=float)
    faces = np.array(((0, 1, 2),), dtype=np.int64)
    face_mu = np.asarray(
        manufactured["quadratic_mu_nodes"],
        dtype=float,
    )[None, :]
    quadratic = QuadraticDoubletSurface(
        triangle,
        faces,
        face_mu,
    )
    constant_strength = float(manufactured["constant_mu"])
    constant = QuadraticDoubletSurface(
        triangle,
        faces,
        np.full((1, 6), constant_strength),
    )
    barycentric = np.asarray(
        manufactured["owner_barycentric"],
        dtype=float,
    )
    points = barycentric @ triangle
    constant_actual = constant.induced_velocity_sheet_average(
        np.zeros(len(points), dtype=np.int64),
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
                    points,
                    constant_strength,
                )
            ),
            initial=0.0,
        )
    )
    low = quadratic.induced_velocity_sheet_average(
        np.zeros(len(points), dtype=np.int64),
        barycentric,
        quadrature_order=8,
        edge_quadrature=MODE,
    )
    high = quadratic.induced_velocity_sheet_average(
        np.zeros(len(points), dtype=np.int64),
        barycentric,
        quadrature_order=24,
        edge_quadrature=MODE,
    )
    manufactured_invariance = float(
        np.max(np.abs(high - low), initial=0.0)
    )
    moderate = np.asarray(
        contract["canonical"]["moderate_owner_barycentric"],
        dtype=float,
    )
    analytic_moderate = quadratic.induced_velocity_sheet_average(
        np.zeros(len(moderate), dtype=np.int64),
        moderate,
        quadrature_order=8,
        edge_quadrature=MODE,
    )
    standard_moderate = quadratic.induced_velocity_sheet_average(
        np.zeros(len(moderate), dtype=np.int64),
        moderate,
        quadrature_order=int(
            contract["canonical"][
                "independent_standard_reference_order"
            ]
        ),
        edge_quadrature="standard",
    )
    moderate_error = float(
        np.max(
            np.abs(analytic_moderate - standard_moderate),
            initial=0.0,
        )
    )
    moment_error = _moment_derivative_relative_error()

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
    vector = np.array(
        (np.cos(np.deg2rad(5.0)), 0.0, np.sin(np.deg2rad(5.0)))
    )
    ledger = _ledger(solution, contract, vector)

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
    moved_ledger = _ledger(
        moved_solution,
        contract,
        moved_vector,
    )
    rigid_error = _rotation_error(ledger, moved_ledger, rotation)

    edge_failures = 0
    for bad_barycentric in (
        np.array(((0.5, 0.5, 0.0),)),
        np.array(((1.0, 0.0, 0.0),)),
    ):
        try:
            quadratic.induced_velocity_sheet_average(
                [0],
                bad_barycentric,
                quadrature_order=8,
                edge_quadrature=MODE,
            )
        except DistributedDoubletError:
            edge_failures += 1

    checks = {
        "moment_derivatives_match_integrands": (
            moment_error
            <= float(thresholds["moment_derivative_relative_max"])
        ),
        "analytic_matches_independent_moderate_reference": (
            moderate_error
            <= float(
                thresholds[
                    "analytic_vs_standard_moderate_abs_max"
                ]
            )
        ),
        "manufactured_near_vertex_is_order_invariant": (
            manufactured_invariance
            <= float(
                thresholds[
                    "manufactured_order_invariance_abs_max"
                ]
            )
        ),
        "constant_strength_is_exact_ring": (
            constant_ring_error
            <= float(thresholds["constant_ring_abs_max"])
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
        "ledger_is_exact": (
            ledger.closure_error()
            <= float(thresholds["ledger_closure_abs_max"])
            and ledger.wake_representation_error
            <= float(thresholds["ledger_closure_abs_max"])
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
        "artifact": "analytic_p2_sheet_finite_part_oracle",
        "stage": contract["stage"],
        "claim_node": contract["claim_node"],
        "stage_decision": (
            "GO" if all(checks.values()) else "NO-GO"
        ),
        "checks": checks,
        "aggregate_metrics": {
            "moment_derivative_relative_max": moment_error,
            "analytic_vs_standard_moderate_abs_max": moderate_error,
            "manufactured_order_invariance_abs_max": (
                manufactured_invariance
            ),
            "constant_ring_abs_max": constant_ring_error,
            "actual_body_last_abs_change": (
                ledger.body_doublet_report.max_abs_change
            ),
            "actual_wake_last_abs_change": (
                ledger.wake_sheet_average_report.max_abs_change
            ),
            "ledger_closure_abs_max": ledger.closure_error(),
            "full_wake_representation_abs_max": (
                ledger.wake_representation_error
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
