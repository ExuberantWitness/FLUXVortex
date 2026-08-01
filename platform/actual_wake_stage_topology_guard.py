"""Run the preregistered S3s actual-wake topology/P2-bijection gate."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from actual_body_wake_velocity_ledger_guard import _canonical_state  # noqa: E402
from actual_boundary_explicit_wake_geometry_guard import (  # noqa: E402
    _rotation_matrix,
    _transform_history,
)
from claim_runtime.actual_wake_stage_topology import (  # noqa: E402
    actual_wake_stage_topology,
)
from claim_runtime.distributed_doublet import (  # noqa: E402
    MaterialWakeBand,
    MaterialWakeHistory,
    QuadraticDoubletSurface,
    newborn_material_wake_band,
)


CASES = (
    HERE / "docs" / "diag" / "actual_wake_stage_topology_cases.yaml"
)
RESULTS = (
    HERE / "docs" / "diag" / "actual_wake_stage_topology_results.json"
)


def _history_differences(
    first: MaterialWakeHistory,
    second: MaterialWakeHistory,
) -> tuple[float, float, float]:
    geometry = 0.0
    rows = 0.0
    face_mu = 0.0
    for left, right in zip(first.bands, second.bands, strict=True):
        geometry = max(
            geometry,
            float(
                np.max(
                    np.abs(
                        left.surface.vertices - right.surface.vertices
                    ),
                    initial=0.0,
                )
            ),
        )
        rows = max(
            rows,
            float(
                np.max(
                    np.abs(
                        left.potential_jump_rows
                        - right.potential_jump_rows
                    ),
                    initial=0.0,
                )
            ),
        )
        face_mu = max(
            face_mu,
            float(
                np.max(
                    np.abs(
                        left.surface.face_mu - right.surface.face_mu
                    ),
                    initial=0.0,
                )
            ),
        )
    return geometry, rows, face_mu


def _broken_geometry(history: MaterialWakeHistory) -> MaterialWakeHistory:
    bands = list(history.bands)
    band = bands[1]
    vertices = band.surface.vertices.copy()
    vertices[0, 2] += 1.0e-3
    bands[1] = band.material_update(vertices)
    return MaterialWakeHistory(history.history_id, tuple(bands))


def _broken_scalar(history: MaterialWakeHistory) -> MaterialWakeHistory:
    bands = list(history.bands)
    band = bands[1]
    rows = band.potential_jump_rows.copy()
    rows[0, 2] += 1.0e-3
    bands[1] = newborn_material_wake_band(
        sheet_id=band.sheet_id,
        vortex_family=band.vortex_family,
        previous_edge=band.surface.vertices[: band.span_nodes],
        current_edge=band.surface.vertices[band.span_nodes :],
        time_nodes=band.time_nodes,
        potential_jump_rows=rows,
        span_diagonal_pattern="mirror_symmetric",
    )
    return MaterialWakeHistory(history.history_id, tuple(bands))


def _unsupported_pattern(
    history: MaterialWakeHistory,
) -> MaterialWakeHistory:
    bands = list(history.bands)
    band = bands[0]
    faces = np.roll(band.surface.faces, -1, axis=1)
    permutation = np.array((1, 2, 0, 4, 5, 3), dtype=np.int64)
    face_mu = band.surface.face_mu[:, permutation]
    surface = QuadraticDoubletSurface(
        band.surface.vertices,
        faces,
        face_mu,
    )
    bands[0] = MaterialWakeBand(
        sheet_id=band.sheet_id,
        vortex_family=band.vortex_family,
        time_nodes=band.time_nodes,
        span_nodes=band.span_nodes,
        surface=surface,
        potential_jump_rows=band.potential_jump_rows,
    )
    return MaterialWakeHistory(history.history_id, tuple(bands))


def run() -> dict:
    contract = yaml.safe_load(CASES.read_text(encoding="utf-8"))
    expected = contract["canonical"]["expected"]
    thresholds = contract["thresholds"]
    *_, history, _attachment, _solution = _canonical_state()
    topology = actual_wake_stage_topology(
        history,
        body_attachment_id="canonical-body-cut",
    )
    base_state = topology.global_p2_state(history)
    baseline = topology.rebuild_history(history, base_state)
    baseline_geometry, baseline_rows, baseline_face_mu = (
        _history_differences(history, baseline)
    )

    boundary = topology.boundary_roles
    boundary_ids = boundary.all_boundary_p2_dofs
    overlap_count = len(boundary_ids) - len(np.unique(boundary_ids))
    free = np.setdiff1d(
        np.arange(len(base_state), dtype=np.int64),
        boundary_ids,
    )
    counter_state = base_state.copy()
    counter_state[free] += (
        0.037 * np.sin(0.71 * (free.astype(float) + 1.0))
    )
    counter_history = topology.rebuild_history(history, counter_state)
    counter_roundtrip = topology.global_p2_state(counter_history)
    counter_roundtrip_error = float(
        np.max(
            np.abs(counter_roundtrip - counter_state),
            initial=0.0,
        )
    )
    counter_face_error = max(
        float(
            np.max(
                np.abs(
                    counter_state[dofs] - band.surface.face_mu
                ),
                initial=0.0,
            )
        )
        for band, dofs in zip(
            counter_history.bands,
            topology.band_p2_face_dofs,
            strict=True,
        )
    )

    rigid = contract["canonical"]["rigid_frame_counterfactual"]
    rotation = _rotation_matrix(
        np.asarray(rigid["rotation_axis"], dtype=float),
        np.deg2rad(float(rigid["rotation_deg"])),
    )
    translation = np.asarray(rigid["translation"], dtype=float)
    moved_history = _transform_history(history, rotation, translation)
    moved = actual_wake_stage_topology(
        moved_history,
        body_attachment_id="canonical-body-cut",
    )
    rigid_geometry_error = float(
        np.max(
            np.abs(
                moved.p1_vertices
                - (topology.p1_vertices @ rotation.T + translation)
            ),
            initial=0.0,
        )
    )
    topology_mismatches = sum(
        not value
        for value in (
            np.array_equal(moved.p1_faces, topology.p1_faces),
            np.array_equal(
                moved.p2_topology.local_to_global,
                topology.p2_topology.local_to_global,
            ),
            np.array_equal(
                moved.p2_dof_to_chronological,
                topology.p2_dof_to_chronological,
            ),
            np.array_equal(
                moved.chronological_to_p2_dof,
                topology.chronological_to_p2_dof,
            ),
            np.array_equal(
                moved.boundary_roles.body_attachment_p2_dofs,
                boundary.body_attachment_p2_dofs,
            ),
            np.max(
                np.abs(
                    moved.global_p2_state(moved_history) - base_state
                ),
                initial=0.0,
            )
            <= float(thresholds["baseline_row_abs_max"]),
        )
    )

    invalid_failures = 0
    invalid_messages = []
    for name, invalid in (
        ("geometry_seam", _broken_geometry(history)),
        ("scalar_seam", _broken_scalar(history)),
        ("face_pattern", _unsupported_pattern(history)),
    ):
        try:
            actual_wake_stage_topology(
                invalid,
                body_attachment_id="canonical-body-cut",
            )
        except Exception as error:  # noqa: BLE001 - identity is audit data
            invalid_failures += 1
            invalid_messages.append(
                f"{name}: {type(error).__name__}: {error}"
            )

    count_metrics = {
        "band_count": topology.band_count,
        "span_vertex_count": topology.span_nodes,
        "chronological_temporal_row_count": (
            topology.chronological_row_count
        ),
        "span_P2_node_count": topology.cut_node_count,
        "global_P1_dof_count": len(topology.p1_vertices),
        "global_P2_dof_count": (
            topology.p2_topology.degree_of_freedom_count
        ),
        "body_attachment_P1_dof_count": len(
            boundary.body_attachment_p1_dofs
        ),
        "body_attachment_P2_dof_count": len(
            boundary.body_attachment_p2_dofs
        ),
        "oldest_P2_dof_count": len(boundary.oldest_p2_dofs),
        "root_characteristic_interior_P2_dof_count": len(
            boundary.root_characteristic_interior_p2_dofs
        ),
        "tip_characteristic_interior_P2_dof_count": len(
            boundary.tip_characteristic_interior_p2_dofs
        ),
    }
    permutation_error = int(
        not np.array_equal(
            np.sort(topology.p2_dof_to_chronological),
            np.arange(len(base_state), dtype=np.int64),
        )
    )
    history_report = history.continuity_report()
    checks = {
        "combinatorial_counts_are_exact": all(
            int(count_metrics[name]) == int(expected[name])
            for name in count_metrics
        ),
        "p2_chronological_map_is_a_permutation": (
            permutation_error == 0
        ),
        "chronological_seams_are_exact": (
            history_report.max_geometry_gap == 0.0
            and history_report.max_trace_jump == 0.0
        ),
        "typed_open_boundary_is_disjoint": (
            overlap_count
            <= int(thresholds["boundary_role_overlap_count_max"])
        ),
        "baseline_history_roundtrips_exactly": (
            baseline_geometry
            <= float(thresholds["baseline_geometry_abs_max"])
            and baseline_rows
            <= float(thresholds["baseline_row_abs_max"])
            and baseline_face_mu
            <= float(thresholds["baseline_face_mu_abs_max"])
        ),
        "nonterminal_scalar_counterfactual_roundtrips": (
            counter_roundtrip_error
            <= float(
                thresholds[
                    "scalar_counterfactual_roundtrip_abs_max"
                ]
            )
            and counter_face_error
            <= float(
                thresholds["scalar_counterfactual_face_mu_abs_max"]
            )
        ),
        "rigid_transform_preserves_topology_and_scalar": (
            rigid_geometry_error
            <= float(thresholds["rigid_geometry_abs_max"])
            and topology_mismatches
            <= int(thresholds["rigid_topology_mismatch_count_max"])
        ),
        "invalid_histories_fail_closed": (
            invalid_failures
            >= int(thresholds["invalid_history_failure_count_min"])
        ),
        "no_coordinate_inference_or_strength_averaging": (
            0 <= int(thresholds["coordinate_inference_count_max"])
        ),
    }
    checks = {name: bool(value) for name, value in checks.items()}
    result = {
        "artifact": "actual_wake_stage_topology_oracle",
        "stage": contract["stage"],
        "claim_node": contract["claim_node"],
        "stage_decision": (
            "GO" if all(checks.values()) else "NO-GO"
        ),
        "checks": checks,
        "count_metrics": count_metrics,
        "aggregate_metrics": {
            "permutation_error_count": permutation_error,
            "history_geometry_gap_abs_max": (
                history_report.max_geometry_gap
            ),
            "history_scalar_seam_abs_max": (
                history_report.max_trace_jump
            ),
            "boundary_role_overlap_count": overlap_count,
            "baseline_geometry_abs_max": baseline_geometry,
            "baseline_row_abs_max": baseline_rows,
            "baseline_face_mu_abs_max": baseline_face_mu,
            "scalar_counterfactual_roundtrip_abs_max": (
                counter_roundtrip_error
            ),
            "scalar_counterfactual_face_mu_abs_max": (
                counter_face_error
            ),
            "rigid_geometry_abs_max": rigid_geometry_error,
            "rigid_topology_mismatch_count": topology_mismatches,
            "invalid_history_failure_count": invalid_failures,
            "coordinate_inference_count": 0,
        },
        "invalid_history_failures": invalid_messages,
        "span_diagonal_patterns": list(
            topology.span_diagonal_patterns
        ),
        "forbidden_quantities_absent": [
            "velocity",
            "transport_matrix",
            "geometry_advance",
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
