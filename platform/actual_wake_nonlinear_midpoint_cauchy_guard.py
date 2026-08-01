"""Run the preregistered S3x nonlinear actual-wake time-Cauchy gate."""
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
from actual_wake_nonlinear_midpoint_step_guard import (  # noqa: E402
    _evaluate_stage,
)
from claim_runtime.actual_wake_nonlinear_midpoint import (  # noqa: E402
    advance_actual_wake_previous_time_midpoint,
)


CASES = (
    HERE
    / "docs"
    / "diag"
    / "actual_wake_nonlinear_midpoint_cauchy_cases.yaml"
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
    / "actual_wake_nonlinear_midpoint_cauchy_results.json"
)


def _geometry_change(left: np.ndarray, right: np.ndarray) -> float:
    return float(
        np.max(
            np.linalg.norm(right - left, axis=1),
            initial=0.0,
        )
    )


def _scalar_change(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.max(np.abs(right - left), initial=0.0))


def _cauchy(
    values,
    change,
) -> tuple[float, float, float]:
    coarse_medium = change(values[0], values[1])
    medium_fine = change(values[1], values[2])
    ratio = coarse_medium / max(
        medium_fine,
        np.finfo(float).tiny,
    )
    return coarse_medium, medium_fine, ratio


def run() -> dict:
    contract = yaml.safe_load(CASES.read_text(encoding="utf-8"))
    s3t_contract = yaml.safe_load(S3T_CASES.read_text(encoding="utf-8"))
    canonical = contract["canonical"]
    thresholds = contract["thresholds"]
    (
        mesh,
        body_topology,
        _upper,
        _lower,
        _cut_edges,
        _endpoints,
        _pre_solve_history,
        attachment,
        solution,
    ) = _canonical_state()
    quadrature_order = int(
        canonical["physical_velocity"]["P2_quadrature_order"]
    )

    def evaluator(value):
        return _evaluate_stage(
            value,
            s3t_contract,
            quadrature_order=quadrature_order,
        )

    initial_stage = evaluator(solution)
    initial_topology = initial_stage.topology
    initial_geometry = initial_topology.p1_vertices.copy()
    initial_scalar = initial_topology.global_p2_state(
        solution.wake_history
    )
    initial_body = (
        initial_topology.boundary_roles.body_attachment_p2_dofs
    )
    initial_trace = initial_scalar[initial_body].copy()
    initial_history_geometry = tuple(
        band.surface.vertices.copy()
        for band in solution.wake_history.bands
    )
    initial_history_scalar = initial_scalar.copy()

    time = canonical["time"]
    duration = float(time["end"]) - float(time["start"])
    step_families = [
        int(value) for value in time["step_families"]
    ]
    trajectories = []
    for step_count in step_families:
        current = initial_stage
        step_reports = []
        midpoint_stages = []
        timestep = duration / step_count
        for _ in range(step_count):
            step = advance_actual_wake_previous_time_midpoint(
                mesh,
                body_topology,
                current,
                attachment,
                timestep=timestep,
                stage_evaluator=evaluator,
                boundary_quadrature_order=10,
            )
            step_reports.append(step.report)
            midpoint_stages.append(step.midpoint_stage)
            current = evaluator(step.endpoint_solution)
        final_scalar = current.topology.global_p2_state(
            current.solution.wake_history
        )
        trajectories.append(
            {
                "steps": step_count,
                "final_stage": current,
                "reports": tuple(step_reports),
                "midpoint_stages": tuple(midpoint_stages),
                "geometry": current.topology.p1_vertices.copy(),
                "scalar": final_scalar,
                "trace": final_scalar[
                    current.topology.boundary_roles
                    .body_attachment_p2_dofs
                ].copy(),
            }
        )

    geometries = [value["geometry"] for value in trajectories]
    scalars = [value["scalar"] for value in trajectories]
    traces = [value["trace"] for value in trajectories]
    geometry_metrics = _cauchy(geometries, _geometry_change)
    scalar_metrics = _cauchy(scalars, _scalar_change)
    trace_metrics = _cauchy(traces, _scalar_change)
    geometry_evolution = _geometry_change(
        initial_geometry,
        geometries[-1],
    )
    scalar_evolution = _scalar_change(
        initial_scalar,
        scalars[-1],
    )
    trace_evolution = _scalar_change(
        initial_trace,
        traces[-1],
    )
    geometry_finest = geometry_metrics[1] / max(
        geometry_evolution,
        np.finfo(float).tiny,
    )
    scalar_finest = scalar_metrics[1] / max(
        scalar_evolution,
        np.finfo(float).tiny,
    )
    trace_finest = trace_metrics[1] / max(
        trace_evolution,
        np.finfo(float).tiny,
    )

    all_reports = [
        report
        for trajectory in trajectories
        for report in trajectory["reports"]
    ]
    all_affine = [
        stage
        for report in all_reports
        for stage in (report.half_stage, report.endpoint_stage)
    ]
    all_physical_stages = [
        initial_stage,
        *[
            stage
            for trajectory in trajectories
            for stage in (
                *trajectory["midpoint_stages"],
                trajectory["final_stage"],
            )
        ],
    ]
    boundary_weak = max(
        [
            *(float(stage.solution.relative_weak_residual)
              for stage in all_physical_stages),
            *(stage.maximum_actual_boundary_relative_weak_residual
              for stage in all_affine),
        ]
    )
    algebraic_residual = max(
        stage.algebraic_trace_residual for stage in all_affine
    )
    scalar_residual = max(
        max(
            report.half_scalar_normalized_residual,
            report.full_scalar_normalized_residual,
        )
        for report in all_reports
    )
    identity_residual = max(
        [
            *(max(
                report.body_geometry_attachment_error,
                report.chronological_seam_error,
                report.p2_roundtrip_error,
                report.midpoint_geometry_identity_error,
                report.endpoint_geometry_identity_error,
            ) for report in all_reports),
            *(max(
                stage.geometry_velocity_ledger_closure,
                stage.transport_velocity_ledger_closure,
            ) for stage in all_physical_stages),
        ]
    )
    minimum_area = min(
        report.minimum_face_area_ratio for report in all_reports
    )
    input_mutation = max(
        max(report.input_state_mutation for report in all_reports),
        _scalar_change(
            initial_history_scalar,
            initial_topology.global_p2_state(solution.wake_history),
        ),
        max(
            float(
                np.max(
                    np.abs(band.surface.vertices - before),
                    initial=0.0,
                )
            )
            for band, before in zip(
                solution.wake_history.bands,
                initial_history_geometry,
                strict=True,
            )
        ),
    )
    geometry_iterations = max(
        report.geometry_iteration_count for report in all_reports
    )
    topology_mismatches = 0
    for trajectory in trajectories:
        topology = trajectory["final_stage"].topology
        if (
            not np.array_equal(
                topology.p2_dof_to_chronological,
                initial_topology.p2_dof_to_chronological,
            )
            or not np.array_equal(
                topology.boundary_roles.body_attachment_p2_dofs,
                initial_body,
            )
        ):
            topology_mismatches += 1

    terminal_projection = [
        value["final_stage"].geometry_projection_residual_fraction
        for value in trajectories
    ]
    terminal_normal = [
        value["final_stage"].relative_velocity_normal_component_max
        for value in trajectories
    ]
    projection_changes = (
        abs(terminal_projection[1] - terminal_projection[0]),
        abs(terminal_projection[2] - terminal_projection[1]),
    )
    normal_changes = (
        abs(terminal_normal[1] - terminal_normal[0]),
        abs(terminal_normal[2] - terminal_normal[1]),
    )
    projection_contraction = projection_changes[0] / max(
        projection_changes[1],
        np.finfo(float).tiny,
    )
    normal_contraction = normal_changes[0] / max(
        normal_changes[1],
        np.finfo(float).tiny,
    )
    diagnostic_contraction = min(
        projection_contraction,
        normal_contraction,
    )

    checks = {
        "geometry_time_cauchy_is_second_order": (
            geometry_metrics[2]
            >= float(thresholds["geometry_time_cauchy_ratio_min"])
        ),
        "scalar_time_cauchy_is_second_order": (
            scalar_metrics[2]
            >= float(thresholds["scalar_time_cauchy_ratio_min"])
        ),
        "body_trace_time_cauchy_is_second_order": (
            trace_metrics[2]
            >= float(
                thresholds["body_trace_time_cauchy_ratio_min"]
            )
        ),
        "finest_state_changes_are_small": (
            geometry_finest
            <= float(
                thresholds[
                    "geometry_finest_change_over_evolution_max"
                ]
            )
            and scalar_finest
            <= float(
                thresholds[
                    "scalar_finest_change_over_evolution_max"
                ]
            )
            and trace_finest
            <= float(
                thresholds[
                    "body_trace_finest_change_over_evolution_max"
                ]
            )
        ),
        "all_named_stage_residuals_close": (
            boundary_weak
            <= float(
                thresholds[
                    "actual_boundary_relative_weak_residual_max"
                ]
            )
            and algebraic_residual
            <= float(
                thresholds["algebraic_trace_residual_abs_max"]
            )
            and scalar_residual
            <= float(
                thresholds[
                    "scalar_increment_normalized_residual_max"
                ]
            )
        ),
        "attachment_seams_P2_and_ledgers_close": (
            identity_residual
            <= float(
                thresholds[
                    "attachment_seam_P2_ledger_abs_max"
                ]
            )
        ),
        "terminal_stage_diagnostics_contract": (
            diagnostic_contraction
            >= float(
                thresholds[
                    "diagnostic_refinement_contraction_min"
                ]
            )
        ),
        "terminal_topologies_preserve_chronology": (
            topology_mismatches
            <= int(thresholds["topology_mismatch_count_max"])
        ),
        "mesh_and_shared_initial_state_remain_valid": (
            minimum_area
            >= float(thresholds["minimum_face_area_ratio_min"])
            and input_mutation
            <= float(thresholds["input_state_mutation_abs_max"])
        ),
        "no_geometry_iteration_or_scalar_clamp_enters": (
            geometry_iterations
            <= int(thresholds["geometry_iteration_count_max"])
        ),
    }
    checks = {name: bool(value) for name, value in checks.items()}
    result = {
        "artifact": "actual_wake_nonlinear_midpoint_cauchy_oracle",
        "claim_node": contract["claim_node"],
        "stage": contract["stage"],
        "stage_decision": (
            "GO" if all(checks.values()) else "NO-GO"
        ),
        "checks": checks,
        "aggregate_metrics": {
            "step_families": step_families,
            "geometry_coarse_medium_change": geometry_metrics[0],
            "geometry_medium_fine_change": geometry_metrics[1],
            "geometry_time_cauchy_ratio": geometry_metrics[2],
            "scalar_coarse_medium_change": scalar_metrics[0],
            "scalar_medium_fine_change": scalar_metrics[1],
            "scalar_time_cauchy_ratio": scalar_metrics[2],
            "body_trace_coarse_medium_change": trace_metrics[0],
            "body_trace_medium_fine_change": trace_metrics[1],
            "body_trace_time_cauchy_ratio": trace_metrics[2],
            "geometry_finest_change_over_evolution": geometry_finest,
            "scalar_finest_change_over_evolution": scalar_finest,
            "body_trace_finest_change_over_evolution": trace_finest,
            "actual_boundary_relative_weak_residual_max": boundary_weak,
            "algebraic_trace_residual_abs_max": algebraic_residual,
            "scalar_increment_normalized_residual_max": scalar_residual,
            "attachment_seam_P2_ledger_abs_max": identity_residual,
            "terminal_projection_residual_fractions": (
                terminal_projection
            ),
            "projection_diagnostic_refinement_contraction": (
                projection_contraction
            ),
            "terminal_relative_normal_component_maxima": terminal_normal,
            "relative_normal_diagnostic_refinement_contraction": (
                normal_contraction
            ),
            "topology_mismatch_count": topology_mismatches,
            "minimum_face_area_ratio": minimum_area,
            "input_state_mutation_abs_max": input_mutation,
            "geometry_iteration_count": geometry_iterations,
            "nonlinear_step_count_total": sum(step_families),
            "physical_stage_evaluation_count": (
                1 + 2 * sum(step_families)
            ),
        },
        "forbidden_quantities_absent": [
            "geometry_fixed_point",
            "scalar_endpoint_clamp",
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
        0 if payload["stage_decision"] == "GO" else 1
    )
