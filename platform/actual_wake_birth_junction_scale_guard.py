"""Run the preregistered S3ad birth-junction scale diagnosis."""
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
from actual_wake_owned_stage_velocity_guard import _actual_ledger  # noqa: E402
from claim_runtime.actual_body_wake_velocity import (  # noqa: E402
    WakeSheetQuery,
    wake_sheet_interior_query,
)
from claim_runtime.actual_wake_direct_trace import (  # noqa: E402
    solve_actual_wake_coupled_newborn_trace_direct,
)
from claim_runtime.actual_wake_newborn_transition import (  # noqa: E402
    augment_actual_wake_with_newborn_band,
)
from claim_runtime.actual_wake_stage_topology import (  # noqa: E402
    actual_wake_stage_topology,
)
from claim_runtime.actual_wake_stage_velocity import (  # noqa: E402
    actual_wake_owned_quadrature,
)
from claim_runtime.actual_wake_weak_geometry_velocity import (  # noqa: E402
    project_actual_wake_global_weak_normal_velocity,
)


CASES = (
    HERE
    / "docs"
    / "diag"
    / "actual_wake_birth_junction_scale_cases.yaml"
)
S3T_CASES = (
    HERE / "docs" / "diag" / "actual_wake_owned_stage_velocity_cases.yaml"
)
RESULTS = (
    HERE
    / "docs"
    / "diag"
    / "actual_wake_birth_junction_scale_results.json"
)


def _maximum_abs(array: np.ndarray) -> float:
    return float(np.max(np.abs(array), initial=0.0))


def _maximum_norm(array: np.ndarray) -> float:
    return float(np.max(np.linalg.norm(array, axis=1), initial=0.0))


def _newest_owned_query(history, query_id: str) -> WakeSheetQuery:
    full = wake_sheet_interior_query(history, query_id=query_id)
    newest = len(history.bands) - 1
    selected = full.patch_indices == newest
    if not np.any(selected):
        raise RuntimeError("newborn query contains no newest-band points")
    return WakeSheetQuery(
        points=full.points[selected],
        patch_indices=full.patch_indices[selected],
        face_indices=full.face_indices[selected],
        barycentric=full.barycentric[selected],
        query_id=query_id,
    )


def _newborn_vorticity_abs_max(history) -> float:
    surface = history.bands[-1].surface
    vertices = np.eye(3)
    maximum = 0.0
    for face_index in range(len(surface)):
        vorticity = surface.element(
            face_index
        ).sheet_vorticity_barycentric(vertices)
        maximum = max(maximum, _maximum_norm(vorticity))
    return maximum


def _fit_log_order(timesteps, values) -> tuple[float, list[float]]:
    dt = np.asarray(timesteps, dtype=float)
    value = np.maximum(
        np.asarray(values, dtype=float),
        np.finfo(float).tiny,
    )
    order = float(np.polyfit(np.log(dt), np.log(value), 1)[0])
    pairwise = [
        float(
            np.log(value[index + 1] / value[index])
            / np.log(dt[index + 1] / dt[index])
        )
        for index in range(len(dt) - 1)
    ]
    return order, pairwise


def run() -> dict:
    contract = yaml.safe_load(CASES.read_text(encoding="utf-8"))
    s3t_contract = yaml.safe_load(
        S3T_CASES.read_text(encoding="utf-8")
    )
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
    weak_quadrature = actual_wake_owned_quadrature(
        topology,
        history,
        quadrature_order=int(
            canonical["transport_quadrature_order"]
        ),
        query_id="S3ad-initial-release",
    )
    initial_ledger = _actual_ledger(
        solution,
        weak_quadrature.query,
        s3t_contract,
    )
    weak_projection = project_actual_wake_global_weak_normal_velocity(
        topology,
        history,
        weak_quadrature,
        initial_ledger.total,
    )
    body_p1 = topology.boundary_roles.body_attachment_p1_dofs
    body_edge = topology.p1_vertices[body_p1].copy()
    release_velocity = weak_projection.dof_velocity[body_p1]
    old_trace = topology.chronological_rows(history)[-1]
    start_time = float(history.bands[-1].time_nodes[-1])
    fraction = float(canonical["geometry_fraction"])
    timesteps = [
        float(value) for value in canonical["timestep_family"]
    ]
    cases = []
    for index, timestep in enumerate(timesteps):
        print(
            f"S3ad scale {index + 1}/{len(timesteps)} dt={timestep:g}",
            flush=True,
        )
        released_edge = (
            body_edge + fraction * timestep * release_velocity
        )
        transition = augment_actual_wake_with_newborn_band(
            history,
            topology,
            released_edge=released_edge,
            current_body_edge=body_edge,
            time_nodes=np.array(
                (
                    start_time,
                    start_time + 0.5 * fraction * timestep,
                    start_time + fraction * timestep,
                )
            ),
            midpoint_trace=old_trace,
            current_trace=old_trace,
            sheet_id=f"S3ad-dt-{index}",
        )
        stage = solve_actual_wake_coupled_newborn_trace_direct(
            mesh,
            body_topology,
            transition,
            attachment,
            incident_velocity=solution.incident_velocity,
            wall_velocity=solution.wall_velocity,
            copy_counterfactual_trace=old_trace,
            boundary_quadrature_order=int(
                canonical["actual_boundary_quadrature_order"]
            ),
        )
        newborn_query = _newest_owned_query(
            stage.solution.wake_history,
            query_id=f"S3ad-newborn-dt-{index}",
        )
        ledger = _actual_ledger(
            stage.solution,
            newborn_query,
            s3t_contract,
        )
        displacement = released_edge - body_edge
        cases.append(
            {
                "timestep": timestep,
                "newborn_thickness_abs_max": _maximum_norm(
                    displacement
                ),
                "minimum_newborn_face_area": float(
                    transition.report.minimum_newborn_face_area
                ),
                "trace_increment_abs_max": _maximum_abs(
                    stage.solved_trace - old_trace
                ),
                "newborn_sheet_vorticity_abs_max": (
                    _newborn_vorticity_abs_max(
                        stage.solution.wake_history
                    )
                ),
                "actual_matrix_condition_number": float(
                    stage.solution.condition_number
                ),
                "actual_relative_weak_residual": float(
                    stage.solution.relative_weak_residual
                ),
                "wake_attachment_error": float(
                    stage.solution.wake_attachment_error
                ),
                "body_doublet_velocity_abs_max": _maximum_abs(
                    ledger.body_doublet
                ),
                "wake_sheet_velocity_abs_max": _maximum_abs(
                    ledger.wake_sheet_average
                ),
                "total_velocity_abs_max": _maximum_abs(
                    ledger.total
                ),
                "body_doublet_quadrature_change_abs_max": float(
                    ledger.body_doublet_report.max_abs_change
                ),
                "wake_quadrature_change_abs_max": float(
                    ledger.wake_sheet_average_report.max_abs_change
                ),
                "wake_representation_error": float(
                    ledger.wake_representation_error
                ),
                "query_reconstruction_error": float(
                    ledger.query_reconstruction_error
                ),
            }
        )
        partial = {
            "artifact": "actual_wake_birth_junction_scale_oracle",
            "claim_node": contract["claim_node"],
            "stage": contract["stage"],
            "stage_decision": "RUNNING",
            "cases": cases,
            "production_activation_allowed": False,
        }
        RESULTS.write_text(
            json.dumps(partial, indent=2) + "\n",
            encoding="utf-8",
        )

    trace_order, trace_pairwise = _fit_log_order(
        timesteps,
        [value["trace_increment_abs_max"] for value in cases],
    )
    vorticity_order, vorticity_pairwise = _fit_log_order(
        timesteps,
        [
            value["newborn_sheet_vorticity_abs_max"]
            for value in cases
        ],
    )
    velocity_order, velocity_pairwise = _fit_log_order(
        timesteps,
        [value["total_velocity_abs_max"] for value in cases],
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
            _maximum_abs(band.surface.vertices - geometry),
            _maximum_abs(band.potential_jump_rows - scalar),
        )
    aggregate = {
        "trace_increment_dt_order": trace_order,
        "trace_increment_pairwise_orders": trace_pairwise,
        "sheet_vorticity_dt_order": vorticity_order,
        "sheet_vorticity_pairwise_orders": vorticity_pairwise,
        "sheet_vorticity_divergence_order": max(
            0.0, -vorticity_order
        ),
        "total_velocity_dt_order": velocity_order,
        "total_velocity_pairwise_orders": velocity_pairwise,
        "total_velocity_divergence_order": max(
            0.0, -velocity_order
        ),
        "actual_matrix_condition_number_max": max(
            value["actual_matrix_condition_number"] for value in cases
        ),
        "actual_relative_weak_residual_max": max(
            value["actual_relative_weak_residual"] for value in cases
        ),
        "wake_attachment_error_max": max(
            value["wake_attachment_error"] for value in cases
        ),
        "input_state_mutation_abs_max": mutation,
    }
    checks = {
        "actual_algebraic_system_remains_resolved": (
            aggregate["actual_matrix_condition_number_max"]
            <= float(
                thresholds["actual_matrix_condition_number_max"]
            )
            and aggregate["actual_relative_weak_residual_max"]
            <= float(
                thresholds["actual_relative_weak_residual_max"]
            )
            and aggregate["wake_attachment_error_max"]
            <= float(thresholds["wake_attachment_error_max"])
        ),
        "trace_increment_is_at_least_first_order": (
            trace_order
            >= float(thresholds["trace_increment_dt_order_min"])
        ),
        "newborn_sheet_vorticity_is_bounded": (
            aggregate["sheet_vorticity_divergence_order"]
            <= float(
                thresholds[
                    "sheet_vorticity_divergence_order_max"
                ]
            )
        ),
        "junction_total_velocity_is_bounded": (
            aggregate["total_velocity_divergence_order"]
            <= float(
                thresholds[
                    "total_velocity_divergence_order_max"
                ]
            )
        ),
        "input_state_is_immutable": (
            mutation
            <= float(thresholds["input_state_mutation_abs_max"])
        ),
    }
    checks = {name: bool(value) for name, value in checks.items()}
    result = {
        "artifact": "actual_wake_birth_junction_scale_oracle",
        "claim_node": contract["claim_node"],
        "stage": contract["stage"],
        "stage_decision": (
            "GO" if all(checks.values()) else "NO-GO"
        ),
        "checks": checks,
        "aggregate_metrics": aggregate,
        "cases": cases,
        "forbidden_quantities_absent": [
            "wake_panel_length_rule",
            "vortex_core",
            "epsilon_offset",
            "damping",
            "clamp",
            "pressure",
            "force",
            "LESP",
            "target_load",
            "structural_dynamics",
        ],
        "production_activation_allowed": False,
    }
    RESULTS.write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> int:
    result = run()
    print(json.dumps(result, indent=2))
    return 0 if result["stage_decision"] == "GO" else 1


if __name__ == "__main__":
    raise SystemExit(main())
