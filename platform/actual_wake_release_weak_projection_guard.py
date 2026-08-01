"""Run the preregistered S3z point-trace versus weak-release gate."""
from __future__ import annotations

from collections import defaultdict
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
)
from claim_runtime.actual_wake_stage_topology import (  # noqa: E402
    actual_wake_stage_topology,
)
from claim_runtime.actual_wake_stage_velocity import (  # noqa: E402
    actual_wake_owned_quadrature,
)
from claim_runtime.actual_wake_weak_geometry_velocity import (  # noqa: E402
    project_actual_wake_global_weak_normal_velocity,
    weak_normal_collocation_velocity,
)
from claim_runtime.distributed_doublet import (  # noqa: E402
    QuadraticDoubletElement,
)


CASES = (
    HERE
    / "docs"
    / "diag"
    / "actual_wake_release_weak_projection_cases.yaml"
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
    / "actual_wake_release_weak_projection_results.json"
)


def _point_approach_query(history, topology, epsilons):
    band = history.bands[-1]
    patch = len(history.bands) - 1
    span_nodes = band.span_nodes
    normal_sum = np.zeros_like(topology.p1_vertices)
    for face in topology.p1_faces:
        element = QuadraticDoubletElement(
            topology.p1_vertices[face],
            np.zeros(6),
        )
        for dof in face:
            normal_sum[int(dof)] += element.area_vector
    normals = normal_sum / np.linalg.norm(
        normal_sum,
        axis=1,
    )[:, None]
    body = topology.boundary_roles.body_attachment_p1_dofs
    records = []
    for span_index, local_vertex in enumerate(
        range(span_nodes, 2 * span_nodes)
    ):
        for face_index, face in enumerate(band.surface.faces):
            location = np.flatnonzero(face == local_vertex)
            if len(location) == 0:
                continue
            local = int(location[0])
            for epsilon in epsilons:
                barycentric = np.full(3, epsilon, dtype=float)
                barycentric[local] = 1.0 - 2.0 * epsilon
                records.append(
                    (
                        "vertex",
                        span_index,
                        face_index,
                        float(epsilon),
                        barycentric,
                        normals[body[span_index]],
                    )
                )
    for span_index, edge in enumerate(band.downstream_edges):
        for face_index, face in enumerate(band.surface.faces):
            if not set(edge).issubset(set(face)):
                continue
            local_edge = [
                int(np.flatnonzero(face == vertex)[0])
                for vertex in edge
            ]
            opposite = next(
                value
                for value in range(3)
                if value not in local_edge
            )
            element = QuadraticDoubletElement(
                band.surface.vertices[face],
                np.zeros(6),
            )
            for epsilon in epsilons:
                barycentric = np.zeros(3, dtype=float)
                barycentric[opposite] = epsilon
                barycentric[local_edge] = 0.5 * (1.0 - epsilon)
                records.append(
                    (
                        "edge",
                        span_index,
                        face_index,
                        float(epsilon),
                        barycentric,
                        element.normal,
                    )
                )
            break
    points = np.vstack(
        [
            record[4]
            @ band.surface.vertices[
                band.surface.faces[record[2]]
            ]
            for record in records
        ]
    )
    query = WakeSheetQuery(
        points=points,
        patch_indices=np.full(
            len(records),
            patch,
            dtype=np.int64,
        ),
        face_indices=np.asarray(
            [record[2] for record in records],
            dtype=np.int64,
        ),
        barycentric=np.vstack(
            [record[4] for record in records]
        ),
        query_id="S3z-unseen-attachment-approach",
    )
    return query, tuple(records)


def _point_metrics(records, velocity, span_nodes):
    sequences = defaultdict(list)
    for record, value in zip(records, velocity, strict=True):
        kind, node, face, epsilon, _barycentric, normal = record
        sequences[(kind, node, face)].append(
            (epsilon, float(value @ normal))
        )
    differences = {}
    contractions = {}
    for key, sequence in sequences.items():
        ordered = sorted(sequence, reverse=True)
        values = [item[1] for item in ordered]
        delta = [
            abs(values[index] - values[index + 1])
            for index in range(len(values) - 1)
        ]
        differences[key] = delta
        contractions[key] = (
            delta[-2] / max(delta[-1], np.finfo(float).tiny)
        )
    tip_keys = [
        key
        for key in sequences
        if key[0] == "vertex"
        and key[1] in (0, span_nodes - 1)
    ]
    outer_edge_keys = [
        key
        for key in sequences
        if key[0] == "edge"
        and key[1] in (0, span_nodes - 2)
    ]
    tip_finest = max(differences[key][-1] for key in tip_keys)
    tip_contraction = min(contractions[key] for key in tip_keys)
    edge_finest = max(
        differences[key][-1] for key in outer_edge_keys
    )
    edge_contraction = min(
        contractions[key] for key in outer_edge_keys
    )
    finest_epsilon = min(
        record[3] for record in records
    )
    direction_spread = 0.0
    for node in range(1, span_nodes - 1):
        values = []
        for key, sequence in sequences.items():
            if key[0] != "vertex" or key[1] != node:
                continue
            values.extend(
                value
                for epsilon, value in sequence
                if epsilon == finest_epsilon
            )
        direction_spread = max(
            direction_spread,
            max(values) - min(values),
        )
    return {
        "tip_finest_change_abs_max": tip_finest,
        "tip_last_difference_contraction_min": tip_contraction,
        "outer_edge_finest_change_abs_max": edge_finest,
        "outer_edge_last_difference_contraction_min": (
            edge_contraction
        ),
        "incident_direction_spread_abs_max": direction_spread,
        "tip_sequences": {
            str(key): sorted(sequences[key], reverse=True)
            for key in tip_keys
        },
        "outer_edge_sequences": {
            str(key): sorted(sequences[key], reverse=True)
            for key in outer_edge_keys
        },
    }


def run() -> dict:
    contract = yaml.safe_load(CASES.read_text(encoding="utf-8"))
    s3t_contract = yaml.safe_load(
        S3T_CASES.read_text(encoding="utf-8")
    )
    thresholds = contract["thresholds"]
    canonical = contract["canonical"]
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
    geometry_snapshot = tuple(
        band.surface.vertices.copy() for band in history.bands
    )
    scalar_snapshot = tuple(
        band.potential_jump_rows.copy() for band in history.bands
    )

    epsilons = [
        float(value)
        for value in contract["candidates"]["point_trace"][
            "approach_epsilons"
        ]
    ]
    point_query, point_records = _point_approach_query(
        history,
        topology,
        epsilons,
    )
    point_ledger = _actual_ledger(
        solution,
        point_query,
        s3t_contract,
    )
    point = _point_metrics(
        point_records,
        point_ledger.total,
        history.span_nodes,
    )
    point_checks = {
        "tip_values_converge": (
            point["tip_finest_change_abs_max"]
            <= float(
                thresholds["point_tip_finest_change_abs_max"]
            )
            and point["tip_last_difference_contraction_min"]
            >= float(
                thresholds[
                    "point_last_difference_contraction_min"
                ]
            )
        ),
        "outer_edge_values_converge": (
            point["outer_edge_finest_change_abs_max"]
            <= float(
                thresholds[
                    "point_outer_edge_finest_change_abs_max"
                ]
            )
            and point[
                "outer_edge_last_difference_contraction_min"
            ]
            >= float(
                thresholds[
                    "point_last_difference_contraction_min"
                ]
            )
        ),
        "incident_face_directions_agree": (
            point["incident_direction_spread_abs_max"]
            <= float(
                thresholds[
                    "point_incident_direction_spread_abs_max"
                ]
            )
        ),
    }
    point_checks = {
        key: bool(value) for key, value in point_checks.items()
    }
    point_pass = all(point_checks.values())

    orders = [
        int(value)
        for value in contract["candidates"]["weak_projection"][
            "quadrature_orders"
        ]
    ]
    projections = []
    ledgers = []
    manufactured_errors = []
    for order in orders:
        quadrature = actual_wake_owned_quadrature(
            topology,
            history,
            quadrature_order=order,
            query_id=f"S3z-weak-q{order}",
        )
        ledger = _actual_ledger(
            solution,
            quadrature.query,
            s3t_contract,
        )
        projection = (
            project_actual_wake_global_weak_normal_velocity(
                topology,
                history,
                quadrature,
                ledger.total,
            )
        )
        coordinate = np.linspace(
            -1.0,
            1.0,
            len(topology.p1_vertices),
        )
        manufactured_scalar = (
            0.17
            + 0.09 * coordinate
            - 0.04 * coordinate**2
        )
        manufactured_velocity = (
            weak_normal_collocation_velocity(
                topology,
                history,
                quadrature,
                manufactured_scalar,
            )
        )
        manufactured = (
            project_actual_wake_global_weak_normal_velocity(
                topology,
                history,
                quadrature,
                manufactured_velocity,
            )
        )
        manufactured_errors.append(
            float(
                np.max(
                    np.abs(
                        manufactured.scalar_normal_speed
                        - manufactured_scalar
                    ),
                    initial=0.0,
                )
            )
        )
        projections.append(projection)
        ledgers.append(ledger)
    body = topology.boundary_roles.body_attachment_p1_dofs
    release_fields = [
        value.dof_velocity[body] for value in projections
    ]

    def maximum_vector_change(left, right):
        return float(
            np.max(
                np.linalg.norm(right - left, axis=1),
                initial=0.0,
            )
        )

    coarse_medium = maximum_vector_change(
        release_fields[0],
        release_fields[1],
    )
    medium_fine = maximum_vector_change(
        release_fields[1],
        release_fields[2],
    )
    weak_contraction = coarse_medium / max(
        medium_fine,
        np.finfo(float).tiny,
    )
    fine_scale = max(
        float(
            np.max(
                np.linalg.norm(release_fields[-1], axis=1),
                initial=0.0,
            )
        ),
        np.finfo(float).tiny,
    )
    weak_finest_relative = medium_fine / fine_scale

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
    moved_quadrature = actual_wake_owned_quadrature(
        moved_topology,
        moved_history,
        quadrature_order=orders[-1],
        query_id="S3z-weak-rigid",
    )
    moved_projection = project_actual_wake_global_weak_normal_velocity(
        moved_topology,
        moved_history,
        moved_quadrature,
        ledgers[-1].total @ rotation.T,
    )
    rigid_error = float(
        np.max(
            np.abs(
                moved_projection.dof_velocity
                - projections[-1].dof_velocity @ rotation.T
            ),
            initial=0.0,
        )
    )
    maximum_rank_deficiency = max(
        value.report.rank_deficiency for value in projections
    )
    maximum_condition = max(
        value.report.condition_number for value in projections
    )
    maximum_orthogonality = max(
        value.report.weak_orthogonality_relative_residual
        for value in projections
    )
    maximum_tangent = max(
        value.report.maximum_tangential_nodal_velocity
        for value in projections
    )
    maximum_ledger = max(
        point_ledger.closure_error(),
        *(value.closure_error() for value in ledgers),
    )
    maximum_manufactured = max(manufactured_errors)
    release_speed = float(
        np.max(
            np.linalg.norm(release_fields[-1], axis=1),
            initial=0.0,
        )
    )
    mutation = 0.0
    for band, geometry, scalar in zip(
        history.bands,
        geometry_snapshot,
        scalar_snapshot,
        strict=True,
    ):
        mutation = max(
            mutation,
            float(
                np.max(
                    np.abs(band.surface.vertices - geometry),
                    initial=0.0,
                )
            ),
            float(
                np.max(
                    np.abs(band.potential_jump_rows - scalar),
                    initial=0.0,
                )
            ),
        )
    weak_checks = {
        "weak_gram_systems_are_valid": (
            maximum_rank_deficiency
            <= int(thresholds["weak_mass_rank_deficiency_max"])
            and maximum_condition
            <= float(
                thresholds["weak_mass_condition_number_max"]
            )
        ),
        "manufactured_P1_field_is_recovered": (
            maximum_manufactured
            <= float(
                thresholds[
                    "weak_manufactured_recovery_abs_max"
                ]
            )
        ),
        "physical_projection_is_weakly_orthogonal": (
            maximum_orthogonality
            <= float(
                thresholds[
                    "weak_orthogonality_relative_residual_max"
                ]
            )
        ),
        "weak_release_is_nontrivial_and_normal": (
            release_speed
            >= float(thresholds["weak_release_speed_abs_min"])
            and maximum_tangent
            <= float(
                thresholds[
                    "weak_tangential_velocity_abs_max"
                ]
            )
        ),
        "weak_release_converges_with_quadrature": (
            weak_contraction
            >= float(
                thresholds[
                    "weak_quadrature_change_contraction_min"
                ]
            )
            and weak_finest_relative
            <= float(
                thresholds[
                    "weak_finest_release_relative_change_max"
                ]
            )
        ),
        "weak_release_is_rigid_frame_covariant": (
            rigid_error
            <= float(
                thresholds["rigid_release_velocity_abs_max"]
            )
        ),
        "actual_ledgers_close_and_inputs_are_immutable": (
            maximum_ledger
            <= float(
                thresholds[
                    "actual_velocity_ledger_closure_abs_max"
                ]
            )
            and mutation
            <= float(
                thresholds["input_state_mutation_abs_max"]
            )
        ),
    }
    weak_checks = {
        key: bool(value) for key, value in weak_checks.items()
    }
    weak_pass = all(weak_checks.values())
    if not point_pass and weak_pass:
        stage_decision = "WEAK-GO"
    elif point_pass and not weak_pass:
        stage_decision = "POINT-GO"
    elif point_pass and weak_pass:
        stage_decision = "AMBIGUOUS"
    else:
        stage_decision = "NO-GO"
    checks = {
        "point_trace_fails_preregistered_convergence": (
            not point_pass
        ),
        **weak_checks,
    }
    result = {
        "artifact": "actual_wake_release_weak_projection_oracle",
        "claim_node": contract["claim_node"],
        "stage": contract["stage"],
        "stage_decision": stage_decision,
        "checks": checks,
        "candidate_verdicts": {
            "point_trace": (
                "GO" if point_pass else "FALSIFIED"
            ),
            "global_weak_P1": (
                "GO" if weak_pass else "NO-GO"
            ),
            "point_checks": point_checks,
            "weak_checks": weak_checks,
        },
        "aggregate_metrics": {
            **point,
            "weak_quadrature_orders": orders,
            "weak_mass_rank_deficiency_max": (
                maximum_rank_deficiency
            ),
            "weak_mass_condition_number_max": maximum_condition,
            "weak_manufactured_recovery_abs_max": (
                maximum_manufactured
            ),
            "weak_orthogonality_relative_residual_max": (
                maximum_orthogonality
            ),
            "weak_surface_L2_residuals": [
                value.report.relative_surface_L2_residual
                for value in projections
            ],
            "weak_release_coarse_medium_change": (
                coarse_medium
            ),
            "weak_release_medium_fine_change": medium_fine,
            "weak_quadrature_change_contraction": (
                weak_contraction
            ),
            "weak_finest_release_relative_change": (
                weak_finest_relative
            ),
            "weak_release_speed_abs_max": release_speed,
            "weak_tangential_velocity_abs_max": maximum_tangent,
            "rigid_release_velocity_abs_max": rigid_error,
            "actual_velocity_ledger_closure_abs_max": (
                maximum_ledger
            ),
            "input_state_mutation_abs_max": mutation,
        },
        "forbidden_quantities_absent": [
            "point_trace_in_weak_candidate",
            "blob_radius",
            "core_radius",
            "epsilon_crop",
            "filter",
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
        0 if payload["stage_decision"] == "WEAK-GO" else 1
    )
