"""Run the preregistered S3e explicit-midpoint material-wake oracle."""
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
from claim_runtime.classified_p2_cut_topology import (  # noqa: E402
    classified_p2_cut_topology,
)
from claim_runtime.material_wake_time_march import (  # noqa: E402
    ExplicitMidpointWakeMarch,
    march_actual_boundary_material_wake_explicit_midpoint,
)


CASES = (
    HERE / "docs" / "diag"
    / "actual_boundary_midpoint_time_convergence_cases.yaml"
)
RESULTS = (
    HERE / "docs" / "diag"
    / "actual_boundary_midpoint_time_convergence_results.json"
)


def _maximum_band_aspect_ratio(
    march: ExplicitMidpointWakeMarch,
) -> float:
    maximum = 0.0
    for band in march.final_history.bands:
        previous = band.surface.vertices[: band.span_nodes]
        current = band.surface.vertices[band.span_nodes :]
        chord = float(
            np.max(np.linalg.norm(previous - current, axis=1))
        )
        span = float(
            np.min(
                np.linalg.norm(np.diff(previous, axis=0), axis=1)
            )
        )
        maximum = max(maximum, chord / span)
    return maximum


def _wake_probe_velocity(
    march: ExplicitMidpointWakeMarch,
    probes: np.ndarray,
    quadrature_order: int,
) -> np.ndarray:
    velocity = np.zeros_like(probes)
    for band in march.final_history.bands:
        velocity += band.surface.induced_velocity_line_reduced(
            probes,
            quadrature_order=quadrature_order,
        )
    return velocity


def _maximum_difference(first: np.ndarray, second: np.ndarray) -> float:
    return float(
        np.max(np.abs(np.asarray(first) - np.asarray(second)), initial=0.0)
    )


def _contraction(
    coarse: np.ndarray,
    medium: np.ndarray,
    fine: np.ndarray,
) -> tuple[float, float, float, float]:
    coarse_change = _maximum_difference(coarse, medium)
    fine_change = _maximum_difference(medium, fine)
    contraction = coarse_change / max(
        fine_change,
        np.finfo(float).tiny,
    )
    relative_fine = fine_change / max(
        float(np.max(np.abs(fine), initial=0.0)),
        np.finfo(float).tiny,
    )
    return coarse_change, fine_change, contraction, relative_fine


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
    start = float(canonical["physical_time"]["start"])
    end = float(canonical["physical_time"]["end"])
    timesteps = [
        float(value)
        for value in canonical["physical_time"][
            "timestep_families"
        ]
    ]
    convection_speed = float(
        canonical["wake"]["uniform_convection_speed"]
    )
    order = int(canonical["spatial_quadrature_order"])
    velocity_order = int(
        canonical["off_sheet_velocity_quadrature_order"]
    )
    probes = np.asarray(
        canonical["off_sheet_probe_points"],
        dtype=float,
    )
    amplitude = np.deg2rad(5.0)

    def incident(time: float) -> np.ndarray:
        alpha = amplitude * np.sin(
            np.pi * (time - start) / (end - start)
        )
        return np.repeat(
            np.array(
                ((speed * np.cos(alpha), 0.0, speed * np.sin(alpha)),)
            ),
            len(mesh.faces),
            axis=0,
        )

    initial_jump = np.zeros(
        len(topology.cut_node_coordinates),
        dtype=float,
    )
    marches = [
        march_actual_boundary_material_wake_explicit_midpoint(
            mesh,
            topology,
            incident_velocity_at_time=incident,
            initial_body_cut_jump=initial_jump,
            time_start=start,
            time_end=end,
            timestep=timestep,
            trailing_edge_x=1.0,
            convection_speed=convection_speed,
            target_quadrature_order=order,
            source_quadrature_order=order,
        )
        for timestep in timesteps
    ]
    all_steps = [
        step for march in marches for step in march.steps
    ]
    all_stage_solutions = [
        solution
        for step in all_steps
        for solution in (step.half_stage, step.full_stage)
    ]
    reports = [
        step.history_after.continuity_report()
        for step in all_steps
    ]
    old_mutation = max(
        step.old_strength_mutation for step in all_steps
    )
    old_geometry_error = max(
        step.old_geometry_convection_error for step in all_steps
    )
    time_gap = max(report.max_time_gap for report in reports)
    geometry_gap = max(
        report.max_geometry_gap for report in reports
    )
    trace_jump = max(report.max_trace_jump for report in reports)
    midpoint_identity = max(
        step.midpoint_row_identity_error for step in all_steps
    )
    attachment = max(
        step.current_attachment_error for step in all_steps
    )
    endpoint_average_difference = max(
        _maximum_difference(
            step.body_jump_midpoint,
            0.5
            * (
                step.body_jump_previous
                + step.body_jump_current
            ),
        )
        for step in all_steps
    )
    old_unknowns = max(
        solution.independent_old_wake_unknown_count
        for solution in all_stage_solutions
    )
    rank_deficiency = max(
        solution.body_unknown_count - solution.rank
        for solution in all_stage_solutions
    )
    weak_residual = max(
        solution.relative_weak_residual
        for solution in all_stage_solutions
    )
    condition_number = max(
        solution.condition_number
        for solution in all_stage_solutions
    )
    known_history_rhs_norms = [
        float(np.linalg.norm(solution.known_wake_right_hand_side))
        for step in all_steps
        if step.step_index > 0
        for solution in (step.half_stage, step.full_stage)
    ]
    tip_jump = max(
        max(
            abs(float(jump[0])),
            abs(float(jump[-1])),
        )
        for step in all_steps
        for jump in (
            step.body_jump_previous,
            step.body_jump_midpoint,
            step.body_jump_current,
        )
    )
    aspect_ratio = max(
        _maximum_band_aspect_ratio(march)
        for march in marches
    )
    body_metrics = _contraction(
        marches[0].final_body_cut_jump,
        marches[1].final_body_cut_jump,
        marches[2].final_body_cut_jump,
    )
    probe_velocities = [
        _wake_probe_velocity(march, probes, velocity_order)
        for march in marches
    ]
    probe_metrics = _contraction(*probe_velocities)

    checks = {
        "old_material_strength_is_immutable": (
            old_mutation
            <= float(thresholds["old_state_mutation_abs_max"])
        ),
        "old_geometry_uses_prescribed_convection": (
            old_geometry_error
            <= float(
                thresholds["history_geometry_gap_abs_max"]
            )
        ),
        "chronological_history_interfaces_are_exact": (
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
        "midpoint_row_comes_from_half_stage": (
            midpoint_identity
            <= float(
                thresholds["stage_midpoint_identity_abs_max"]
            )
        ),
        "midpoint_is_not_endpoint_average": (
            endpoint_average_difference
            >= float(
                thresholds[
                    "midpoint_vs_endpoint_average_difference_min"
                ]
            )
        ),
        "current_row_attaches_to_body": (
            attachment
            <= float(thresholds["active_attachment_abs_max"])
        ),
        "old_wake_is_known_rhs_not_unknown": (
            old_unknowns
            <= int(
                thresholds[
                    "independent_old_wake_unknown_count_max"
                ]
            )
            and bool(known_history_rhs_norms)
            and min(known_history_rhs_norms) > 0.0
        ),
        "all_stage_systems_are_full_rank": (
            rank_deficiency
            <= int(thresholds["rank_deficiency_max"])
            and weak_residual
            <= float(thresholds["normalized_weak_residual_max"])
            and condition_number
            <= float(thresholds["condition_number_max"])
        ),
        "tip_jump_is_zero": (
            tip_jump <= float(thresholds["tip_jump_abs_max"])
        ),
        "bands_are_shape_regular": (
            aspect_ratio
            <= float(
                thresholds["maximum_band_aspect_ratio_max"]
            )
        ),
        "body_jump_time_cauchy_passes": (
            body_metrics[2]
            >= float(
                thresholds["body_jump_cauchy_contraction_min"]
            )
            and body_metrics[3]
            <= float(
                thresholds[
                    "body_jump_finest_relative_change_max"
                ]
            )
        ),
        "wake_probe_time_cauchy_passes": (
            probe_metrics[2]
            >= float(
                thresholds["wake_probe_cauchy_contraction_min"]
            )
            and probe_metrics[3]
            <= float(
                thresholds[
                    "wake_probe_finest_relative_change_max"
                ]
            )
        ),
    }
    checks = {name: bool(value) for name, value in checks.items()}
    result = {
        "artifact": "actual_boundary_midpoint_time_convergence_oracle",
        "claim_node": contract["claim_node"],
        "stage": contract["stage"],
        "canonical": canonical,
        "aggregate_metrics": {
            "old_state_mutation_abs_max": old_mutation,
            "old_geometry_convection_error_abs_max": old_geometry_error,
            "history_time_gap_abs_max": time_gap,
            "history_geometry_gap_abs_max": geometry_gap,
            "history_trace_jump_abs_max": trace_jump,
            "stage_midpoint_identity_abs_max": midpoint_identity,
            "midpoint_vs_endpoint_average_difference_max": (
                endpoint_average_difference
            ),
            "active_attachment_abs_max": attachment,
            "independent_old_wake_unknown_count_max": old_unknowns,
            "known_history_rhs_norm_min": min(
                known_history_rhs_norms
            ),
            "rank_deficiency_max": rank_deficiency,
            "normalized_weak_residual_max": weak_residual,
            "condition_number_max": condition_number,
            "tip_jump_abs_max": tip_jump,
            "maximum_band_aspect_ratio": aspect_ratio,
            "body_jump_coarse_to_medium_abs_change": body_metrics[0],
            "body_jump_medium_to_fine_abs_change": body_metrics[1],
            "body_jump_cauchy_contraction": body_metrics[2],
            "body_jump_finest_relative_change": body_metrics[3],
            "wake_probe_coarse_to_medium_abs_change": probe_metrics[0],
            "wake_probe_medium_to_fine_abs_change": probe_metrics[1],
            "wake_probe_cauchy_contraction": probe_metrics[2],
            "wake_probe_finest_relative_change": probe_metrics[3],
            "final_body_cut_jumps": {
                str(timestep): march.final_body_cut_jump.tolist()
                for timestep, march in zip(timesteps, marches)
            },
            "final_probe_velocities": {
                str(timestep): velocity.tolist()
                for timestep, velocity in zip(
                    timesteps,
                    probe_velocities,
                )
            },
        },
        "checks": checks,
        "stage_decision": (
            "GO" if all(checks.values()) else "NO-GO"
        ),
        "production_activation_allowed": False,
        "forbidden_quantities_absent": [
            "pressure",
            "force",
            "LESP",
            "prescribed_current_circulation",
            "regularizer",
            "endpoint_fit",
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

