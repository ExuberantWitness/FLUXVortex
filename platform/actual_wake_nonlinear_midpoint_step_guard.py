"""Run the preregistered S3w nonlinear actual-wake midpoint step gate."""
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
from actual_wake_owned_stage_velocity_guard import (  # noqa: E402
    _actual_ledger,
)
from claim_runtime.actual_body_wake_velocity import (  # noqa: E402
    wake_sheet_interior_query,
)
from claim_runtime.actual_wake_nonlinear_midpoint import (  # noqa: E402
    ActualWakeEvaluatedStage,
    advance_actual_wake_previous_time_midpoint,
)
from claim_runtime.actual_wake_stage_topology import (  # noqa: E402
    actual_wake_stage_topology,
)
from claim_runtime.actual_wake_stage_velocity import (  # noqa: E402
    actual_wake_owned_quadrature,
    assemble_owned_actual_wake_p2_transport,
    project_actual_wake_vertex_star_normal_velocity,
)


CASES = (
    HERE
    / "docs"
    / "diag"
    / "actual_wake_nonlinear_midpoint_step_cases.yaml"
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
    / "actual_wake_nonlinear_midpoint_step_results.json"
)


def _evaluate_stage(
    solution,
    s3t_contract,
    *,
    quadrature_order: int,
) -> ActualWakeEvaluatedStage:
    history = solution.wake_history
    topology = actual_wake_stage_topology(
        history,
        body_attachment_id="canonical-body-cut",
    )
    geometry_query = wake_sheet_interior_query(history)
    geometry_ledger = _actual_ledger(
        solution,
        geometry_query,
        s3t_contract,
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
    quadrature = actual_wake_owned_quadrature(
        topology,
        history,
        quadrature_order=int(quadrature_order),
        query_id="S3w-actual-P2-stage",
    )
    transport_ledger = _actual_ledger(
        solution,
        quadrature.query,
        s3t_contract,
    )
    mesh_velocity = projection.evaluate_query(
        history,
        quadrature.query,
    )
    transport = assemble_owned_actual_wake_p2_transport(
        topology,
        history,
        quadrature,
        transport_ledger.total - mesh_velocity,
    )
    return ActualWakeEvaluatedStage(
        solution=solution,
        topology=topology,
        p1_geometry_velocity=projection.dof_velocity,
        transport_operator=transport,
        geometry_velocity_ledger_closure=(
            geometry_ledger.closure_error()
        ),
        transport_velocity_ledger_closure=(
            transport_ledger.closure_error()
        ),
        geometry_projection_residual_fraction=(
            projection.report.maximum_absolute_residual
            / max(
                projection.report.maximum_input_normal_speed,
                np.finfo(float).eps,
            )
        ),
        relative_velocity_normal_component_max=(
            transport.maximum_relative_velocity_normal_component
        ),
    )


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
    step = advance_actual_wake_previous_time_midpoint(
        mesh,
        body_topology,
        initial_stage,
        attachment,
        timestep=float(canonical["timestep"]),
        stage_evaluator=evaluator,
        boundary_quadrature_order=10,
    )
    report = step.report
    stage_reports = (
        report.half_stage,
        report.endpoint_stage,
    )
    boundary_weak = max(
        float(initial_stage.solution.relative_weak_residual),
        float(step.midpoint_stage.solution.relative_weak_residual),
        float(step.endpoint_solution.relative_weak_residual),
        *(value.maximum_actual_boundary_relative_weak_residual
          for value in stage_reports),
    )
    rank_deficiency = max(
        len(
            step.initial_stage.topology.boundary_roles
            .body_attachment_p2_dofs
        )
        - value.rank
        for value in stage_reports
    )
    stage_condition = max(
        value.condition_number for value in stage_reports
    )
    algebraic_residual = max(
        value.algebraic_trace_residual for value in stage_reports
    )
    free_preservation = max(
        value.free_state_preservation_error for value in stage_reports
    )
    scalar_residual = max(
        report.half_scalar_normalized_residual,
        report.full_scalar_normalized_residual,
    )
    geometry_identity = max(
        report.midpoint_geometry_identity_error,
        report.endpoint_geometry_identity_error,
    )
    ledger_closure = max(
        initial_stage.geometry_velocity_ledger_closure,
        initial_stage.transport_velocity_ledger_closure,
        step.midpoint_stage.geometry_velocity_ledger_closure,
        step.midpoint_stage.transport_velocity_ledger_closure,
    )
    checks = {
        "actual_boundary_systems_remain_valid": (
            boundary_weak
            <= float(
                thresholds[
                    "actual_boundary_relative_weak_residual_max"
                ]
            )
        ),
        "affine_stage_trace_systems_are_valid": (
            rank_deficiency
            <= int(
                thresholds["algebraic_stage_rank_deficiency_max"]
            )
            and stage_condition
            <= float(
                thresholds["algebraic_stage_condition_number_max"]
            )
        ),
        "midpoint_and_endpoint_traces_close": (
            algebraic_residual
            <= float(
                thresholds["algebraic_trace_residual_abs_max"]
            )
            and free_preservation
            <= float(thresholds["P2_roundtrip_abs_max"])
        ),
        "half_and_full_scalar_increments_close": (
            scalar_residual
            <= float(
                thresholds[
                    "scalar_increment_normalized_residual_max"
                ]
            )
        ),
        "geometry_stage_identities_and_attachment_hold": (
            geometry_identity
            <= float(thresholds["chronological_seam_abs_max"])
            and report.body_geometry_attachment_error
            <= float(
                thresholds["body_geometry_attachment_abs_max"]
            )
        ),
        "chronology_and_P2_roundtrip_hold": (
            report.chronological_seam_error
            <= float(thresholds["chronological_seam_abs_max"])
            and report.p2_roundtrip_error
            <= float(thresholds["P2_roundtrip_abs_max"])
        ),
        "actual_four_channel_stage_ledgers_close": (
            ledger_closure
            <= float(
                thresholds["velocity_ledger_closure_abs_max"]
            )
        ),
        "geometry_and_scalar_updates_are_nontrivial": (
            report.free_geometry_change
            >= float(thresholds["free_geometry_change_abs_min"])
            and report.free_scalar_change
            >= float(thresholds["free_scalar_change_abs_min"])
        ),
        "mesh_and_input_state_remain_valid": (
            report.minimum_face_area_ratio
            >= float(thresholds["minimum_face_area_ratio_min"])
            and report.input_state_mutation
            <= float(thresholds["input_state_mutation_abs_max"])
        ),
        "no_geometry_iteration_or_scalar_clamp_enters": (
            report.geometry_iteration_count
            <= int(thresholds["geometry_iteration_count_max"])
        ),
    }
    checks = {name: bool(value) for name, value in checks.items()}
    result = {
        "artifact": "actual_wake_nonlinear_midpoint_step_oracle",
        "claim_node": contract["claim_node"],
        "stage": contract["stage"],
        "stage_decision": (
            "GO" if all(checks.values()) else "NO-GO"
        ),
        "checks": checks,
        "aggregate_metrics": {
            "actual_boundary_relative_weak_residual_max": boundary_weak,
            "algebraic_stage_rank_deficiency_max": rank_deficiency,
            "algebraic_stage_condition_number_max": stage_condition,
            "algebraic_trace_residual_abs_max": algebraic_residual,
            "free_state_preservation_abs_max": free_preservation,
            "half_scalar_normalized_residual": (
                report.half_scalar_normalized_residual
            ),
            "full_scalar_normalized_residual": (
                report.full_scalar_normalized_residual
            ),
            "geometry_stage_identity_abs_max": geometry_identity,
            "body_geometry_attachment_abs_max": (
                report.body_geometry_attachment_error
            ),
            "chronological_seam_abs_max": (
                report.chronological_seam_error
            ),
            "P2_roundtrip_abs_max": report.p2_roundtrip_error,
            "velocity_ledger_closure_abs_max": ledger_closure,
            "free_geometry_change_abs_max": report.free_geometry_change,
            "free_scalar_change_abs_max": report.free_scalar_change,
            "minimum_face_area_ratio": report.minimum_face_area_ratio,
            "input_state_mutation_abs_max": report.input_state_mutation,
            "geometry_iteration_count": report.geometry_iteration_count,
            "half_stage_basis_solve_count": (
                report.half_stage.basis_solve_count
            ),
            "endpoint_basis_solve_count": (
                report.endpoint_stage.basis_solve_count
            ),
            "base_geometry_projection_residual_fraction": (
                initial_stage.geometry_projection_residual_fraction
            ),
            "midpoint_geometry_projection_residual_fraction": (
                step.midpoint_stage
                .geometry_projection_residual_fraction
            ),
            "base_relative_velocity_normal_component_max": (
                initial_stage.relative_velocity_normal_component_max
            ),
            "midpoint_relative_velocity_normal_component_max": (
                step.midpoint_stage
                .relative_velocity_normal_component_max
            ),
        },
        "forbidden_quantities_absent": [
            "geometry_fixed_point",
            "scalar_endpoint_clamp",
            "damping",
            "stabilization",
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
