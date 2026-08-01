"""Run the preregistered S3h body-attached material-wake Heun oracle."""
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
from claim_runtime.classified_p2_cut_topology import (  # noqa: E402
    classified_p2_cut_topology,
)
from claim_runtime.constrained_material_wake_advection import (  # noqa: E402
    advance_constrained_material_wake_heun,
)
from claim_runtime.distributed_doublet import (  # noqa: E402
    DistributedDoubletError,
    MaterialWakeHistory,
)


CASES = (
    HERE / "docs" / "diag"
    / "constrained_material_wake_advection_cases.yaml"
)
RESULTS = (
    HERE / "docs" / "diag"
    / "constrained_material_wake_advection_results.json"
)


def _base_translation(time: float) -> np.ndarray:
    return np.array(
        (
            0.04 * np.sin(2.0 * time),
            0.02 * (1.0 - np.cos(time)),
            0.03 * time,
        )
    )


def _base_offset(time: float) -> np.ndarray:
    return np.array(
        (
            0.03 * np.cos(time),
            -0.02 * np.sin(time),
            0.04,
        )
    )


def _velocity_provider(
    matrix: np.ndarray,
    offset,
):
    def provider(history: MaterialWakeHistory, time: float):
        value = np.asarray(offset(time), dtype=float)
        return tuple(
            band.surface.vertices @ matrix.T + value
            for band in history.bands
        )

    return provider


def _integrate(
    initial: MaterialWakeHistory,
    *,
    start: float,
    end: float,
    steps: int,
    velocity_provider,
    attached_edge_provider,
    minimum_area_ratio: float,
):
    history = initial
    reports = []
    dt = (end - start) / steps
    for index in range(steps):
        result = advance_constrained_material_wake_heun(
            history,
            time=start + index * dt,
            dt=dt,
            velocity_provider=velocity_provider,
            attached_edge_provider=attached_edge_provider,
            seam_velocity_tolerance=1.0e-14,
            attachment_tolerance=1.0e-14,
            strength_tolerance=0.0,
            minimum_face_area_ratio=minimum_area_ratio,
        )
        history = result.history
        reports.append(result.report)
    return history, tuple(reports)


def _free_vertices(history: MaterialWakeHistory) -> np.ndarray:
    values = []
    last = len(history.bands) - 1
    for index, band in enumerate(history.bands):
        if index == last:
            values.append(
                band.surface.vertices[: band.span_nodes]
            )
        else:
            values.append(band.surface.vertices)
    return np.vstack(values)


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
    initial = _history(mesh, topology, curved=True)
    snapshots = [
        (
            band.surface.vertices.copy(),
            band.potential_jump_rows.copy(),
            band.surface.face_mu.copy(),
        )
        for band in initial.bands
    ]
    matrix = np.asarray(
        canonical["manufactured_free_velocity"]["A"],
        dtype=float,
    )
    time_spec = canonical["time"]
    start = float(time_spec["start"])
    end = float(time_spec["end"])
    step_families = [
        int(value) for value in time_spec["step_families"]
    ]
    body_edge0 = initial.bands[-1].surface.vertices[
        initial.bands[-1].span_nodes :
    ].copy()

    def attached(time: float) -> np.ndarray:
        return body_edge0 + _base_translation(time)

    provider = _velocity_provider(matrix, _base_offset)
    integrations = [
        _integrate(
            initial,
            start=start,
            end=end,
            steps=steps,
            velocity_provider=provider,
            attached_edge_provider=attached,
            minimum_area_ratio=float(
                thresholds["minimum_face_area_ratio_min"]
            ),
        )
        for steps in step_families
    ]
    histories = [item[0] for item in integrations]
    report_sets = [item[1] for item in integrations]
    free = [_free_vertices(history) for history in histories]
    coarse_change = float(
        np.max(
            np.linalg.norm(free[1] - free[0], axis=1),
            initial=0.0,
        )
    )
    fine_change = float(
        np.max(
            np.linalg.norm(free[2] - free[1], axis=1),
            initial=0.0,
        )
    )
    time_ratio = coarse_change / max(
        fine_change,
        np.finfo(float).tiny,
    )
    displacement_scale = max(
        float(
            np.max(
                np.linalg.norm(
                    free[2] - _free_vertices(initial),
                    axis=1,
                ),
                initial=0.0,
            )
        ),
        np.finfo(float).tiny,
    )
    finest_relative = fine_change / displacement_scale

    rigid_spec = canonical["rigid_frame_counterfactual"]
    rotation = _rotation_matrix(
        np.asarray(rigid_spec["rotation_axis"], dtype=float),
        np.deg2rad(float(rigid_spec["rotation_deg"])),
    )
    translation = np.asarray(
        rigid_spec["translation"],
        dtype=float,
    )
    moved_initial = _transform_history(
        initial,
        rotation,
        translation,
        reverse_span=False,
    )
    moved_matrix = rotation @ matrix @ rotation.T

    def moved_offset(time: float) -> np.ndarray:
        return (
            rotation @ _base_offset(time)
            - moved_matrix @ translation
        )

    moved_provider = _velocity_provider(
        moved_matrix,
        moved_offset,
    )

    def moved_attached(time: float) -> np.ndarray:
        return attached(time) @ rotation.T + translation

    moved_final, moved_reports = _integrate(
        moved_initial,
        start=start,
        end=end,
        steps=step_families[-1],
        velocity_provider=moved_provider,
        attached_edge_provider=moved_attached,
        minimum_area_ratio=float(
            thresholds["minimum_face_area_ratio_min"]
        ),
    )
    rigid_error = max(
        _max_abs(
            moved_band.surface.vertices,
            base_band.surface.vertices @ rotation.T + translation,
        )
        for moved_band, base_band in zip(
            moved_final.bands,
            histories[-1].bands,
        )
    )

    mismatch_failures = 0

    def mismatched_provider(
        history: MaterialWakeHistory,
        time: float,
    ):
        velocities = [
            value.copy()
            for value in provider(history, time)
        ]
        velocities[1][0, 0] += float(
            canonical["mismatch_counterfactual"][
                "perturbation"
            ].split()[1]
        )
        return tuple(velocities)

    try:
        advance_constrained_material_wake_heun(
            initial,
            time=start,
            dt=(end - start) / step_families[0],
            velocity_provider=mismatched_provider,
            attached_edge_provider=attached,
            seam_velocity_tolerance=float(
                thresholds["seam_velocity_abs_max"]
            ),
        )
    except DistributedDoubletError:
        mismatch_failures += 1

    input_geometry_mutation = max(
        _max_abs(band.surface.vertices, vertices)
        for band, (vertices, _, _) in zip(initial.bands, snapshots)
    )
    material_strength_mutation = max(
        max(
            _max_abs(band.potential_jump_rows, rows),
            _max_abs(band.surface.face_mu, face_mu),
        )
        for band, (_, rows, face_mu) in zip(initial.bands, snapshots)
    )
    all_reports = [
        report
        for reports in report_sets + [moved_reports]
        for report in reports
    ]
    attachment_error = max(
        report.attached_edge_error for report in all_reports
    )
    seam_position_error = max(
        max(
            report.history.max_geometry_gap,
            report.history.max_trace_jump,
            report.history.max_time_gap,
        )
        for report in all_reports
    )
    seam_velocity_error = max(
        max(
            report.stage0_seam_velocity_error,
            report.stage1_seam_velocity_error,
        )
        for report in all_reports
    )
    strength_residual = max(
        report.material_strength_residual
        for report in all_reports
    )
    minimum_area_ratio = min(
        report.minimum_face_area_ratio for report in all_reports
    )
    checks = {
        "input_and_material_strength_are_immutable": (
            input_geometry_mutation
            <= float(
                thresholds["input_geometry_mutation_abs_max"]
            )
            and material_strength_mutation
            <= float(
                thresholds[
                    "material_strength_mutation_abs_max"
                ]
            )
            and strength_residual
            <= float(
                thresholds[
                    "material_strength_mutation_abs_max"
                ]
            )
        ),
        "body_attachment_is_exact": (
            attachment_error
            <= float(thresholds["attached_edge_abs_max"])
        ),
        "chronological_seams_are_exact": (
            seam_position_error
            <= float(thresholds["history_seam_abs_max"])
            and seam_velocity_error
            <= float(thresholds["seam_velocity_abs_max"])
        ),
        "free_vertex_heun_time_order_passes": (
            time_ratio
            >= float(
                thresholds["free_vertex_time_cauchy_ratio_min"]
            )
            and finest_relative
            <= float(
                thresholds[
                    "free_vertex_finest_relative_change_max"
                ]
            )
        ),
        "rigid_frame_objectivity_passes": (
            rigid_error
            <= float(
                thresholds["rigid_frame_max_abs_difference"]
            )
        ),
        "face_areas_remain_valid": (
            minimum_area_ratio
            >= float(
                thresholds["minimum_face_area_ratio_min"]
            )
            and all(report.passed for report in all_reports)
        ),
        "seam_velocity_mismatch_fails_closed": (
            mismatch_failures
            >= int(
                thresholds["mismatch_failure_count_min"]
            )
        ),
    }
    checks = {name: bool(value) for name, value in checks.items()}
    result = {
        "artifact": "constrained_material_wake_advection_oracle",
        "claim_node": contract["claim_node"],
        "stage": contract["stage"],
        "aggregate_metrics": {
            "input_geometry_mutation_abs_max": (
                input_geometry_mutation
            ),
            "material_strength_mutation_abs_max": max(
                material_strength_mutation,
                strength_residual,
            ),
            "attached_edge_abs_max": attachment_error,
            "history_seam_abs_max": seam_position_error,
            "seam_velocity_abs_max": seam_velocity_error,
            "free_vertex_coarse_to_medium_abs_change": (
                coarse_change
            ),
            "free_vertex_medium_to_fine_abs_change": fine_change,
            "free_vertex_time_cauchy_ratio": time_ratio,
            "free_vertex_finest_relative_change": finest_relative,
            "rigid_frame_max_abs_difference": rigid_error,
            "minimum_face_area_ratio": minimum_area_ratio,
            "mismatch_failure_count": mismatch_failures,
        },
        "checks": checks,
        "stage_decision": (
            "GO" if all(checks.values()) else "NO-GO"
        ),
        "production_activation_allowed": False,
        "forbidden_quantities_absent": [
            "actual_boundary_strength_update",
            "pressure",
            "force",
            "LESP",
            "wake_core",
            "smoothing",
            "target_load",
            "structural_dynamics",
        ],
    }
    RESULTS.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, sort_keys=True))

