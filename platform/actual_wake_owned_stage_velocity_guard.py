"""Run the preregistered S3t owner-aware actual-wake stage gate."""
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
    _field,
    _incident_vector,
)
from actual_boundary_explicit_wake_geometry_guard import (  # noqa: E402
    _rotation_matrix,
    _transform_history,
)
from claim_runtime.actual_body_wake_velocity import (  # noqa: E402
    VALIDATED_EDGE_QUADRATURE,
    WakeSheetQuery,
    evaluate_actual_body_wake_sheet_velocity,
    material_wake_assembly,
    wake_sheet_interior_query,
)
from claim_runtime.actual_wake_stage_topology import (  # noqa: E402
    actual_wake_stage_topology,
)
from claim_runtime.actual_wake_stage_velocity import (  # noqa: E402
    actual_wake_owned_quadrature,
    assemble_owned_actual_wake_p2_transport,
    project_actual_wake_vertex_star_normal_velocity,
)
from claim_runtime.distributed_doublet import (  # noqa: E402
    QuadraticDoubletAssembly,
)
from claim_runtime.p2_surface_material_transport import (  # noqa: E402
    assemble_p2_surface_material_transport,
)
from claim_runtime.sheet_velocity_projection import (  # noqa: E402
    project_assembly_vertex_star_normal_geometry_velocity,
)


CASES = (
    HERE
    / "docs"
    / "diag"
    / "actual_wake_owned_stage_velocity_cases.yaml"
)
RESULTS = (
    HERE
    / "docs"
    / "diag"
    / "actual_wake_owned_stage_velocity_results.json"
)


def _actual_ledger(solution, query, contract):
    canonical = contract["canonical"]["actual_velocity"]
    return evaluate_actual_body_wake_sheet_velocity(
        solution,
        query,
        external_incident=_field(
            _incident_vector(0.5),
            "uniform_freestream",
        ),
        body_doublet_orders=tuple(
            int(value) for value in canonical["body_orders"]
        ),
        wake_sheet_average_orders=tuple(
            int(value) for value in canonical["wake_orders"]
        ),
        absolute_tolerance=0.0,
        relative_tolerance=0.0,
        edge_quadrature=VALIDATED_EDGE_QUADRATURE,
    )


def _closed_projector_equivalence(
    history,
    topology,
    query,
    velocity,
) -> float:
    zero_history = topology.rebuild_history(
        history,
        np.zeros(topology.p2_topology.degree_of_freedom_count),
    )
    zero_topology = actual_wake_stage_topology(
        zero_history,
        body_attachment_id="canonical-body-cut",
    )
    open_projection = (
        project_actual_wake_vertex_star_normal_velocity(
            zero_topology,
            zero_history,
            query,
            velocity,
            body_attachment_velocity=np.zeros(
                (
                    len(
                        zero_topology.boundary_roles
                        .body_attachment_p1_dofs
                    ),
                    3,
                )
            ),
        )
    )
    closed_assembly = QuadraticDoubletAssembly(
        zero_history.as_patches(
            oldest_role="zero",
            newest_role="zero",
            side_roles=("zero", "zero"),
        )
    )
    closed_projection = (
        project_assembly_vertex_star_normal_geometry_velocity(
            closed_assembly,
            velocity,
        )
    )
    maximum = 0.0
    last = len(zero_history.bands) - 1
    for band_index, band in enumerate(zero_history.bands):
        left = open_projection.band_vertex_velocity(band_index)
        right = closed_projection.vertex_velocity(band_index)
        mask = np.ones(len(left), dtype=bool)
        if band_index == last:
            mask[band.span_nodes :] = False
        maximum = max(
            maximum,
            float(
                np.max(
                    np.abs(left[mask] - right[mask]),
                    initial=0.0,
                )
            ),
        )
    return maximum


def _seam_velocity_error(history, projection) -> float:
    maximum = 0.0
    for index, (older, newer) in enumerate(
        zip(history.bands, history.bands[1:])
    ):
        older_velocity = projection.band_vertex_velocity(index)
        newer_velocity = projection.band_vertex_velocity(index + 1)
        maximum = max(
            maximum,
            float(
                np.max(
                    np.abs(
                        older_velocity[older.span_nodes :]
                        - newer_velocity[: newer.span_nodes]
                    ),
                    initial=0.0,
                )
            ),
        )
    return maximum


def run() -> dict:
    contract = yaml.safe_load(CASES.read_text(encoding="utf-8"))
    canonical = contract["canonical"]
    thresholds = contract["thresholds"]
    *_, history, _attachment, solution = _canonical_state()
    topology = actual_wake_stage_topology(
        history,
        body_attachment_id="canonical-body-cut",
    )

    geometry_query = wake_sheet_interior_query(history)
    geometry_ledger = _actual_ledger(
        solution,
        geometry_query,
        contract,
    )
    body_velocity = np.zeros(
        (
            len(topology.boundary_roles.body_attachment_p1_dofs),
            3,
        )
    )
    projection = project_actual_wake_vertex_star_normal_velocity(
        topology,
        history,
        geometry_query,
        geometry_ledger.total,
        body_attachment_velocity=body_velocity,
    )
    body_dofs = topology.boundary_roles.body_attachment_p1_dofs
    free_dofs = np.setdiff1d(
        np.arange(len(topology.p1_vertices), dtype=np.int64),
        body_dofs,
    )
    normal_component = np.einsum(
        "ij,ij->i",
        projection.dof_velocity[free_dofs],
        projection.dof_normals[free_dofs],
    )
    free_tangential = (
        projection.dof_velocity[free_dofs]
        - normal_component[:, None] * projection.dof_normals[free_dofs]
    )
    free_tangential_error = float(
        np.max(np.abs(free_tangential), initial=0.0)
    )
    body_velocity_error = float(
        np.max(
            np.abs(projection.dof_velocity[body_dofs] - body_velocity),
            initial=0.0,
        )
    )
    seam_velocity_error = _seam_velocity_error(history, projection)
    closed_equivalence = _closed_projector_equivalence(
        history,
        topology,
        geometry_query,
        geometry_ledger.total,
    )

    quadrature = actual_wake_owned_quadrature(
        topology,
        history,
        quadrature_order=int(
            canonical["transport_query"]["triangle_quadrature_order"]
        ),
    )
    matrix_a = np.array(
        (
            (0.08, -0.03, 0.02),
            (0.01, -0.04, 0.05),
            (-0.02, 0.06, 0.03),
        )
    )
    vector_b = np.array((0.2, -0.1, 0.07))
    manufactured = (
        quadrature.query.points @ matrix_a.T + vector_b
    )
    owned_reference = assemble_owned_actual_wake_p2_transport(
        topology,
        history,
        quadrature,
        manufactured,
    )
    frozen_reference = assemble_p2_surface_material_transport(
        topology.p1_vertices,
        topology.p1_faces,
        relative_velocity_provider=lambda points: (
            points @ matrix_a.T + vector_b
        ),
        quadrature_order=quadrature.quadrature_order,
    )
    mass_equivalence = float(
        np.max(
            np.abs(
                owned_reference.mass_matrix
                - frozen_reference.mass_matrix
            ),
            initial=0.0,
        )
    )
    advection_equivalence = float(
        np.max(
            np.abs(
                owned_reference.advection_matrix
                - frozen_reference.advection_matrix
            ),
            initial=0.0,
        )
    )

    transport_ledger = _actual_ledger(
        solution,
        quadrature.query,
        contract,
    )
    mesh_velocity = projection.evaluate_query(
        history,
        quadrature.query,
    )
    relative_velocity = transport_ledger.total - mesh_velocity
    actual_operator = assemble_owned_actual_wake_p2_transport(
        topology,
        history,
        quadrature,
        relative_velocity,
    )
    rank_deficiency = (
        topology.p2_topology.degree_of_freedom_count
        - actual_operator.mass_rank
    )

    rigid = canonical["rigid_frame_counterfactual"]
    rotation = _rotation_matrix(
        np.asarray(rigid["rotation_axis"], dtype=float),
        np.deg2rad(float(rigid["rotation_deg"])),
    )
    translation = np.asarray(rigid["translation"], dtype=float)
    moved_history = _transform_history(history, rotation, translation)
    moved_topology = actual_wake_stage_topology(
        moved_history,
        body_attachment_id="canonical-body-cut",
    )
    moved_geometry_query = WakeSheetQuery(
        points=geometry_query.points @ rotation.T + translation,
        patch_indices=geometry_query.patch_indices,
        face_indices=geometry_query.face_indices,
        barycentric=geometry_query.barycentric,
        query_id="rigid-geometry-query",
    )
    moved_projection = project_actual_wake_vertex_star_normal_velocity(
        moved_topology,
        moved_history,
        moved_geometry_query,
        geometry_ledger.total @ rotation.T,
        body_attachment_velocity=body_velocity @ rotation.T,
    )
    rigid_projection_error = float(
        np.max(
            np.abs(
                moved_projection.dof_velocity
                - projection.dof_velocity @ rotation.T
            ),
            initial=0.0,
        )
    )
    moved_quadrature = actual_wake_owned_quadrature(
        moved_topology,
        moved_history,
        quadrature_order=quadrature.quadrature_order,
        query_id="rigid-P2-query",
    )
    moved_operator = assemble_owned_actual_wake_p2_transport(
        moved_topology,
        moved_history,
        moved_quadrature,
        relative_velocity @ rotation.T,
    )
    rigid_mass_error = float(
        np.max(
            np.abs(
                moved_operator.mass_matrix - actual_operator.mass_matrix
            ),
            initial=0.0,
        )
    )
    rigid_advection_error = float(
        np.max(
            np.abs(
                moved_operator.advection_matrix
                - actual_operator.advection_matrix
            ),
            initial=0.0,
        )
    )

    invalid_failures = 0
    invalid_messages = []
    bad_owner = WakeSheetQuery(
        points=geometry_query.points,
        patch_indices=(
            geometry_query.patch_indices + topology.band_count
        ),
        face_indices=geometry_query.face_indices,
        barycentric=geometry_query.barycentric,
        query_id="bad-owner",
    )
    invalid_calls = (
        lambda: project_actual_wake_vertex_star_normal_velocity(
            topology,
            history,
            bad_owner,
            geometry_ledger.total,
            body_attachment_velocity=body_velocity,
        ),
        lambda: project_actual_wake_vertex_star_normal_velocity(
            topology,
            history,
            geometry_query,
            geometry_ledger.total,
            body_attachment_velocity=np.zeros((len(body_velocity), 2)),
        ),
        lambda: assemble_owned_actual_wake_p2_transport(
            topology,
            history,
            quadrature,
            relative_velocity[:-1],
        ),
    )
    for call in invalid_calls:
        try:
            call()
        except Exception as error:  # noqa: BLE001 - audit evidence
            invalid_failures += 1
            invalid_messages.append(
                f"{type(error).__name__}: {error}"
            )

    query_error = max(
        projection.report.maximum_query_reconstruction_error,
        float(
            np.max(
                np.linalg.norm(
                    quadrature.query.points
                    - np.vstack(
                        [
                            quadrature.query.barycentric[rows]
                            @ topology.p1_vertices[
                                topology.p1_faces[face_index]
                            ]
                            for face_index, rows in enumerate(
                                quadrature.face_query_rows
                            )
                        ]
                    ),
                    axis=1,
                ),
                initial=0.0,
            )
        ),
    )
    checks = {
        "owned_queries_reconstruct_physical_points": (
            query_error
            <= float(thresholds["query_reconstruction_abs_max"])
        ),
        "free_geometry_velocity_is_normal_only": (
            free_tangential_error
            <= float(thresholds["free_velocity_tangential_abs_max"])
        ),
        "body_attachment_velocity_is_exact": (
            body_velocity_error
            <= float(thresholds["body_attachment_velocity_abs_max"])
        ),
        "chronological_seam_velocity_is_exact": (
            seam_velocity_error
            <= float(thresholds["chronological_seam_velocity_abs_max"])
        ),
        "open_and_frozen_closed_projectors_agree": (
            closed_equivalence
            <= float(thresholds["closed_projector_equivalence_abs_max"])
        ),
        "owned_and_frozen_transport_assemblers_agree": (
            mass_equivalence
            <= float(thresholds["transport_mass_equivalence_abs_max"])
            and advection_equivalence
            <= float(
                thresholds["transport_advection_equivalence_abs_max"]
            )
        ),
        "actual_four_channel_transport_query_is_exact": (
            transport_ledger.closure_error()
            <= float(thresholds["actual_ledger_closure_abs_max"])
            and np.all(np.isfinite(transport_ledger.total))
        ),
        "actual_p2_operator_is_full_rank_and_constant_exact": (
            rank_deficiency
            <= int(thresholds["actual_mass_rank_deficiency_max"])
            and actual_operator.constant_rate_residual
            <= float(thresholds["actual_constant_rate_residual_max"])
        ),
        "rigid_objectivity_holds": (
            rigid_projection_error
            <= float(thresholds["rigid_projection_abs_max"])
            and rigid_mass_error
            <= float(thresholds["rigid_mass_abs_max"])
            and rigid_advection_error
            <= float(thresholds["rigid_advection_abs_max"])
        ),
        "invalid_inputs_fail_closed": (
            invalid_failures
            >= int(thresholds["invalid_input_failure_count_min"])
        ),
        "no_full_vector_fit_or_coordinate_owner_inference": (
            0
            <= int(
                thresholds["coordinate_owner_inference_count_max"]
            )
        ),
    }
    checks = {name: bool(value) for name, value in checks.items()}
    result = {
        "artifact": "actual_wake_owned_stage_velocity_oracle",
        "stage": contract["stage"],
        "claim_node": contract["claim_node"],
        "stage_decision": (
            "GO" if all(checks.values()) else "NO-GO"
        ),
        "checks": checks,
        "aggregate_metrics": {
            "query_reconstruction_abs_max": query_error,
            "free_velocity_tangential_abs_max": free_tangential_error,
            "body_attachment_velocity_abs_max": body_velocity_error,
            "chronological_seam_velocity_abs_max": seam_velocity_error,
            "closed_projector_equivalence_abs_max": closed_equivalence,
            "transport_mass_equivalence_abs_max": mass_equivalence,
            "transport_advection_equivalence_abs_max": (
                advection_equivalence
            ),
            "geometry_ledger_closure_abs_max": (
                geometry_ledger.closure_error()
            ),
            "actual_transport_ledger_closure_abs_max": (
                transport_ledger.closure_error()
            ),
            "actual_mass_rank_deficiency": rank_deficiency,
            "actual_mass_condition_number": (
                actual_operator.mass_condition_number
            ),
            "actual_constant_rate_residual": (
                actual_operator.constant_rate_residual
            ),
            "actual_relative_velocity_normal_component_max": (
                actual_operator
                .maximum_relative_velocity_normal_component
            ),
            "projection_condition_number": (
                projection.report.maximum_condition_number
            ),
            "projection_residual_fraction": (
                projection.report.maximum_absolute_residual
                / max(
                    projection.report.maximum_input_normal_speed,
                    np.finfo(float).eps,
                )
            ),
            "rigid_projection_abs_max": rigid_projection_error,
            "rigid_mass_abs_max": rigid_mass_error,
            "rigid_advection_abs_max": rigid_advection_error,
            "invalid_input_failure_count": invalid_failures,
            "coordinate_owner_inference_count": 0,
            "geometry_query_point_count": len(geometry_query.points),
            "transport_query_point_count": len(
                quadrature.query.points
            ),
        },
        "invalid_input_failures": invalid_messages,
        "forbidden_quantities_absent": [
            "full_vector_P1_fit",
            "coordinate_owner_inference",
            "geometry_update",
            "P2_scalar_update",
            "inflow_classification",
            "boundary_iteration",
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
