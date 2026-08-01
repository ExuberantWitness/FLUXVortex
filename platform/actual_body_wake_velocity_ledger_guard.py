"""Run the preregistered S3n actual body-wake velocity-ledger gate."""
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
    _rotation_matrix,
    _transform_history,
)
from claim_runtime.actual_body_wake_velocity import (  # noqa: E402
    ActualBodyWakeVelocityError,
    ExternalIncidentField,
    WakeSheetQuery,
    evaluate_actual_body_wake_sheet_velocity,
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
    MaterialWakeHistory,
)
from claim_runtime.material_wake_time_march import (  # noqa: E402
    march_actual_boundary_material_wake_explicit_midpoint,
)
from claim_runtime.thick_body_neumann_shadow import (  # noqa: E402
    closed_triangular_mesh,
)


CASES = (
    HERE / "docs" / "diag"
    / "actual_body_wake_velocity_ledger_cases.yaml"
)
RESULTS = (
    HERE / "docs" / "diag"
    / "actual_body_wake_velocity_ledger_results.json"
)


def _incident_vector(time: float) -> np.ndarray:
    alpha = np.deg2rad(5.0) * np.sin(np.pi * time)
    return np.array((np.cos(alpha), 0.0, np.sin(alpha)))


def _bend_history(history: MaterialWakeHistory) -> MaterialWakeHistory:
    bands = []
    for band in history.bands:
        vertices = band.surface.vertices.copy()
        vertices[:, 2] += (
            0.12
            * (vertices[:, 0] - 1.0)
            * (1.0 - vertices[:, 1] ** 2)
        )
        bands.append(band.material_update(vertices))
    candidate = MaterialWakeHistory(history.history_id, tuple(bands))
    if not candidate.continuity_report().compatible:
        raise RuntimeError("deterministic curved history broke its seams")
    return candidate


def _canonical_state():
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

    def incident_at_body(time: float) -> np.ndarray:
        return np.repeat(
            _incident_vector(time)[None, :],
            len(mesh.faces),
            axis=0,
        )

    march = march_actual_boundary_material_wake_explicit_midpoint(
        mesh,
        topology,
        incident_velocity_at_time=incident_at_body,
        initial_body_cut_jump=np.zeros(
            len(topology.cut_node_coordinates)
        ),
        time_start=0.0,
        time_end=0.5,
        timestep=0.25,
        trailing_edge_x=1.0,
        convection_speed=1.0,
        target_quadrature_order=10,
        source_quadrature_order=10,
    )
    curved = _bend_history(march.final_history)
    attachment = MaterialWakeCutAttachment(
        topology.ordered_cut_vertex_indices,
        1,
    )
    incident = incident_at_body(0.5)
    solution = solve_actual_boundary_body_wake_p2(
        mesh,
        topology,
        incident_velocity=incident,
        downstream_edge_x=None,
        prescribed_wake_history=curved,
        prescribed_wake_attachment=attachment,
        target_quadrature_order=10,
        source_quadrature_order=10,
    )
    return (
        mesh,
        topology,
        upper,
        lower,
        cut_edges,
        endpoints,
        curved,
        attachment,
        solution,
    )


def _field(vector: np.ndarray, *source_ids: str) -> ExternalIncidentField:
    return ExternalIncidentField(
        field_id="canonical-external",
        included_source_ids=tuple(source_ids),
        velocity_provider=lambda points: np.repeat(
            vector[None, :],
            len(points),
            axis=0,
        ),
    )


def _ledger(
    solution,
    contract,
    vector,
    *,
    edge_quadrature: str = "standard",
):
    query = wake_sheet_interior_query(solution.wake_history)
    canonical = contract["canonical"]
    tolerances = canonical["quadrature_tolerances"]
    ledger = evaluate_actual_body_wake_sheet_velocity(
        solution,
        query,
        external_incident=_field(vector, "uniform_freestream"),
        body_doublet_orders=tuple(
            int(value)
            for value in canonical["body_doublet_orders"]
        ),
        wake_sheet_average_orders=tuple(
            int(value)
            for value in canonical["wake_sheet_average_orders"]
        ),
        absolute_tolerance=float(tolerances["absolute"]),
        relative_tolerance=float(tolerances["relative"]),
        edge_quadrature=edge_quadrature,
    )
    return query, ledger


def _maximum_channel_rotation_error(base, moved, rotation) -> float:
    names = (
        "external_incident",
        "body_source",
        "body_doublet",
        "wake_sheet_average",
        "total",
    )
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
        for name in names
    )


def run(
    *,
    edge_quadrature: str = "standard",
    results_path: Path = RESULTS,
) -> dict:
    contract = yaml.safe_load(CASES.read_text(encoding="utf-8"))
    thresholds = contract["thresholds"]
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
    vector = _incident_vector(0.5)
    query, ledger = _ledger(
        solution,
        contract,
        vector,
        edge_quadrature=edge_quadrature,
    )

    history_report = solution.wake_history.continuity_report()
    oldest_trace = float(
        np.max(
            np.abs(
                solution.wake_history.bands[0].potential_jump_rows[0]
            ),
            initial=0.0,
        )
    )
    tip_trace = max(
        float(
            np.max(
                np.abs(
                    band.potential_jump_rows[:, (0, -1)]
                ),
                initial=0.0,
            )
        )
        for band in solution.wake_history.bands
    )
    attachment_error = float(
        np.max(
            np.abs(
                solution.wake.potential_jump_rows[-1]
                - solution.body_cut_jump[
                    attachment.p2_trace_permutation(topology)
                ]
            ),
            initial=0.0,
        )
    )
    channel_norms = {
        name: float(
            np.max(
                np.linalg.norm(getattr(ledger, name), axis=1),
                initial=0.0,
            )
        )
        for name in (
            "external_incident",
            "body_source",
            "body_doublet",
            "wake_sheet_average",
        )
    }

    rigid = contract["canonical"]["rigid_frame_counterfactual"]
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
        edge_quadrature=edge_quadrature,
    )
    rigid_error = _maximum_channel_rotation_error(
        ledger,
        moved_ledger,
        rotation,
    )

    duplicate_failures = 0
    for duplicate in (
        "actual_body_source",
        "current_material_wake",
    ):
        try:
            evaluate_actual_body_wake_sheet_velocity(
                solution,
                query,
                external_incident=_field(
                    vector,
                    "uniform_freestream",
                    duplicate,
                ),
            )
        except ActualBodyWakeVelocityError:
            duplicate_failures += 1

    invalid_failures = 0
    try:
        bad_owner = WakeSheetQuery(
            points=query.points,
            patch_indices=(
                query.patch_indices + len(solution.wake_history.bands)
            ),
            face_indices=query.face_indices,
            barycentric=query.barycentric,
            query_id="bad-owner",
        )
        evaluate_actual_body_wake_sheet_velocity(
            solution,
            bad_owner,
            external_incident=_field(vector, "uniform_freestream"),
        )
    except ActualBodyWakeVelocityError:
        invalid_failures += 1
    try:
        bad_provider = ExternalIncidentField(
            "bad-shape",
            ("uniform_freestream",),
            lambda points: np.zeros((len(points), 2)),
        )
        evaluate_actual_body_wake_sheet_velocity(
            solution,
            query,
            external_incident=bad_provider,
        )
    except ActualBodyWakeVelocityError:
        invalid_failures += 1

    checks = {
        "query_owner_identity_is_exact": (
            ledger.query_reconstruction_error
            <= float(thresholds["query_reconstruction_abs_max"])
        ),
        "history_and_attachment_are_exact": (
            history_report.compatible
            and history_report.max_geometry_gap
            <= float(thresholds["history_geometry_gap_abs_max"])
            and history_report.max_trace_jump
            <= float(thresholds["history_trace_jump_abs_max"])
            and oldest_trace
            <= float(thresholds["oldest_trace_abs_max"])
            and tip_trace <= float(thresholds["tip_trace_abs_max"])
            and attachment_error
            <= float(thresholds["body_attachment_abs_max"])
        ),
        "four_named_channels_are_unique": (
            len(ledger.channel_source_ids)
            == len(set(ledger.channel_source_ids))
            and set(ledger.channel_source_ids)
            == {
                "uniform_freestream",
                "actual_body_source",
                "actual_body_doublet",
                "current_material_wake",
            }
        ),
        "every_channel_is_finite_and_nonzero": (
            all(
                np.all(np.isfinite(getattr(ledger, name)))
                for name in (
                    "external_incident",
                    "body_source",
                    "body_doublet",
                    "wake_sheet_average",
                )
            )
            and min(channel_norms.values())
            >= float(thresholds["channel_norm_min"])
        ),
        "ledger_sum_is_exact": (
            ledger.closure_error()
            <= float(thresholds["ledger_closure_abs_max"])
        ),
        "body_and_wake_quadratures_converge": (
            ledger.body_doublet_report.converged
            and ledger.wake_sheet_average_report.converged
        ),
        "full_wake_equals_explicit_band_sum": (
            ledger.wake_representation_error
            <= float(
                thresholds["full_wake_vs_explicit_sum_abs_max"]
            )
        ),
        "every_vector_channel_is_rigid_objective": (
            rigid_error
            <= float(thresholds["rigid_channel_abs_max"])
        ),
        "duplicate_sources_fail_closed": (
            duplicate_failures
            >= int(
                thresholds["duplicate_source_failure_count_min"]
            )
        ),
        "invalid_query_and_provider_fail_closed": (
            invalid_failures
            >= int(thresholds["invalid_query_failure_count_min"])
        ),
    }
    checks = {name: bool(value) for name, value in checks.items()}
    result = {
        "artifact": "actual_body_wake_velocity_ledger_oracle",
        "stage": contract["stage"],
        "claim_node": contract["claim_node"],
        "edge_quadrature": edge_quadrature,
        "stage_decision": (
            "GO" if all(checks.values()) else "NO-GO"
        ),
        "checks": checks,
        "aggregate_metrics": {
            "query_reconstruction_abs_max": (
                ledger.query_reconstruction_error
            ),
            "history_geometry_gap_abs_max": (
                history_report.max_geometry_gap
            ),
            "history_trace_jump_abs_max": (
                history_report.max_trace_jump
            ),
            "oldest_trace_abs_max": oldest_trace,
            "tip_trace_abs_max": tip_trace,
            "body_attachment_abs_max": attachment_error,
            "ledger_closure_abs_max": ledger.closure_error(),
            "channel_norm_max": channel_norms,
            "body_doublet_quadrature_order": (
                ledger.body_doublet_report.quadrature_order
            ),
            "body_doublet_quadrature_abs_change": (
                ledger.body_doublet_report.max_abs_change
            ),
            "body_doublet_quadrature_rel_change": (
                ledger.body_doublet_report.max_rel_change
            ),
            "wake_sheet_quadrature_order": (
                ledger.wake_sheet_average_report.quadrature_order
            ),
            "wake_sheet_quadrature_abs_change": (
                ledger.wake_sheet_average_report.max_abs_change
            ),
            "wake_sheet_quadrature_rel_change": (
                ledger.wake_sheet_average_report.max_rel_change
            ),
            "full_wake_vs_explicit_sum_abs_max": (
                ledger.wake_representation_error
            ),
            "rigid_channel_abs_max": rigid_error,
            "duplicate_source_failure_count": duplicate_failures,
            "invalid_query_failure_count": invalid_failures,
            "query_point_count": len(query.points),
        },
        "forbidden_quantities_absent": [
            "geometry_update",
            "P2_mu_transport",
            "relaxation_iteration",
            "vortex_core",
            "pressure",
            "force",
            "LESP",
            "target_load",
            "structural_dynamics",
        ],
        "production_activation_allowed": False,
    }
    results_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


if __name__ == "__main__":
    payload = run()
    print(
        json.dumps(payload, ensure_ascii=False, indent=2),
        flush=True,
    )
    raise SystemExit(
        0 if payload["stage_decision"] == "GO" else 1
    )
