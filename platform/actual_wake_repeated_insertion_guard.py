"""Run the preregistered S3ac actual repeated-insertion Cauchy gate."""
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
from claim_runtime.actual_wake_repeated_insertion import (  # noqa: E402
    advance_actual_wake_repeated_insertion_midpoint,
)
from claim_runtime.actual_wake_stage_topology import (  # noqa: E402
    actual_wake_stage_topology,
)


CASES = (
    HERE
    / "docs"
    / "diag"
    / "actual_wake_repeated_insertion_cases.yaml"
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
    / "actual_wake_repeated_insertion_results.json"
)


def _max_geometry_change(left: np.ndarray, right: np.ndarray) -> float:
    return float(
        np.max(np.linalg.norm(right - left, axis=1), initial=0.0)
    )


def _max_scalar_change(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.max(np.abs(right - left), initial=0.0))


def _cauchy(values, change) -> tuple[float, float, float]:
    coarse_medium = change(values[0], values[1])
    medium_fine = change(values[1], values[2])
    return (
        coarse_medium,
        medium_fine,
        coarse_medium
        / max(medium_fine, np.finfo(float).tiny),
    )


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
    initial_topology = actual_wake_stage_topology(
        solution.wake_history,
        body_attachment_id="canonical-body-cut",
    )
    initial_p1_count = len(initial_topology.p1_vertices)
    initial_p2_count = (
        initial_topology.p2_topology.degree_of_freedom_count
    )
    initial_geometry = initial_topology.p1_vertices.copy()
    initial_scalar = initial_topology.global_p2_state(
        solution.wake_history
    )
    initial_history_geometry = tuple(
        band.surface.vertices.copy()
        for band in solution.wake_history.bands
    )
    initial_history_scalar = tuple(
        band.potential_jump_rows.copy()
        for band in solution.wake_history.bands
    )

    def provider(actual_solution, query):
        return _actual_ledger(
            actual_solution,
            query,
            s3t_contract,
        ).total

    time = canonical["time"]
    duration = float(time["end"]) - float(time["start"])
    step_families = [
        int(value) for value in time["step_families"]
    ]
    trajectories = []
    for step_count in step_families:
        current = solution
        reports = []
        timestep = duration / step_count
        for step_index in range(step_count):
            step = advance_actual_wake_repeated_insertion_midpoint(
                mesh,
                body_topology,
                current,
                attachment,
                timestep=timestep,
                physical_velocity_provider=provider,
                transport_quadrature_order=int(
                    canonical["quadrature"][
                        "geometry_and_transport_order"
                    ]
                ),
                boundary_quadrature_order=int(
                    canonical["quadrature"][
                        "actual_boundary_order"
                    ]
                ),
                step_index=step_index,
            )
            reports.append(step.report)
            current = step.endpoint_stage.solution
        final_topology = actual_wake_stage_topology(
            current.wake_history,
            body_attachment_id="canonical-body-cut",
        )
        final_state = final_topology.global_p2_state(
            current.wake_history
        )
        original_p2 = (
            final_topology.chronological_to_p2_dof[
                initial_topology.p2_dof_to_chronological
            ]
        )
        body = (
            final_topology.boundary_roles.body_attachment_p2_dofs
        )
        trajectories.append(
            {
                "steps": step_count,
                "solution": current,
                "topology": final_topology,
                "reports": tuple(reports),
                "original_geometry": (
                    final_topology.p1_vertices[:initial_p1_count].copy()
                ),
                "original_scalar": final_state[original_p2].copy(),
                "body_trace": final_state[body].copy(),
            }
        )

    geometry_cauchy = _cauchy(
        [value["original_geometry"] for value in trajectories],
        _max_geometry_change,
    )
    scalar_cauchy = _cauchy(
        [value["original_scalar"] for value in trajectories],
        _max_scalar_change,
    )
    trace_cauchy = _cauchy(
        [value["body_trace"] for value in trajectories],
        _max_scalar_change,
    )
    geometry_evolution = _max_geometry_change(
        initial_geometry,
        trajectories[-1]["original_geometry"],
    )
    scalar_evolution = _max_scalar_change(
        initial_scalar,
        trajectories[-1]["original_scalar"],
    )
    trace_initial = initial_scalar[
        initial_topology.boundary_roles.body_attachment_p2_dofs
    ]
    trace_evolution = _max_scalar_change(
        trace_initial,
        trajectories[-1]["body_trace"],
    )
    finest_ratios = (
        geometry_cauchy[1]
        / max(geometry_evolution, np.finfo(float).tiny),
        scalar_cauchy[1]
        / max(scalar_evolution, np.finfo(float).tiny),
        trace_cauchy[1]
        / max(trace_evolution, np.finfo(float).tiny),
    )
    reports = [
        report
        for trajectory in trajectories
        for report in trajectory["reports"]
    ]
    minimum_insertions = min(
        len(trajectory["reports"]) for trajectory in trajectories
    )
    band_mismatch = max(
        report.band_increment_mismatch for report in reports
    )
    transport_residual = max(
        max(
            report.half_old_transport_normalized_residual,
            report.full_old_transport_normalized_residual,
        )
        for report in reports
    )
    mass_rank_deficiency = max(
        report.old_mass_rank_deficiency_max for report in reports
    )
    mass_condition = max(
        report.old_mass_condition_number_max for report in reports
    )
    boundary_residual = max(
        report.actual_boundary_relative_weak_residual_max
        for report in reports
    )
    algebraic_residual = max(
        report.algebraic_trace_residual_abs_max for report in reports
    )
    preservation = max(
        report.free_state_preservation_abs_max for report in reports
    )
    birth_identity = max(
        report.characteristic_birth_identity_abs_max
        for report in reports
    )
    seam = max(
        report.chronological_seam_abs_max for report in reports
    )
    roundtrip = max(
        report.p2_roundtrip_abs_max for report in reports
    )
    geometry_identity = max(
        max(
            report.half_geometry_identity_abs_max,
            report.full_geometry_identity_abs_max,
        )
        for report in reports
    )
    minimum_area = min(
        report.minimum_newborn_face_area for report in reports
    )
    mutation = max(
        max(report.input_state_mutation_abs_max for report in reports),
        *(
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
        *(
            float(
                np.max(
                    np.abs(band.potential_jump_rows - before),
                    initial=0.0,
                )
            )
            for band, before in zip(
                solution.wake_history.bands,
                initial_history_scalar,
                strict=True,
            )
        ),
    )
    inferred = max(
        report.inferred_scalar_count for report in reports
    )
    geometry_iterations = max(
        report.geometry_iteration_count for report in reports
    )
    identity_count_mismatch = max(
        max(
            abs(report.old_p1_identity_count - (
                initial_p1_count + (
                    report.initial_band_count
                    - initial_topology.band_count
                ) * initial_topology.span_nodes
            )),
            abs(report.old_p2_identity_count - (
                initial_p2_count + (
                    report.initial_band_count
                    - initial_topology.band_count
                ) * 2 * initial_topology.cut_node_count
            )),
        )
        for report in reports
    )

    checks = {
        "every_trajectory_has_at_least_three_insertions": (
            minimum_insertions
            >= int(thresholds["minimum_consecutive_insertions"])
        ),
        "band_growth_and_material_identity_maps_are_exact": (
            band_mismatch
            <= int(thresholds["band_increment_mismatch_max"])
            and identity_count_mismatch == 0
        ),
        "old_half_and_full_material_balances_close": (
            transport_residual
            <= float(
                thresholds[
                    "old_transport_normalized_residual_max"
                ]
            )
            and mass_rank_deficiency
            <= int(thresholds["old_mass_rank_deficiency_max"])
            and mass_condition
            <= float(thresholds["old_mass_condition_number_max"])
        ),
        "actual_half_and_full_algebraic_constraints_close": (
            boundary_residual
            <= float(
                thresholds[
                    "actual_boundary_relative_weak_residual_max"
                ]
            )
            and algebraic_residual
            <= float(
                thresholds["algebraic_trace_residual_abs_max"]
            )
            and preservation
            <= float(thresholds["free_state_preservation_abs_max"])
        ),
        "characteristic_birth_identity_closes": (
            birth_identity
            <= float(
                thresholds[
                    "characteristic_birth_identity_abs_max"
                ]
            )
        ),
        "geometry_chronology_and_P2_roundtrip_hold": (
            geometry_identity
            <= float(thresholds["chronological_seam_abs_max"])
            and seam
            <= float(thresholds["chronological_seam_abs_max"])
            and roundtrip
            <= float(thresholds["P2_roundtrip_abs_max"])
            and minimum_area
            >= float(thresholds["minimum_newborn_face_area_min"])
        ),
        "original_material_geometry_is_second_order_cauchy": (
            geometry_cauchy[2]
            >= float(
                thresholds["original_geometry_cauchy_ratio_min"]
            )
        ),
        "original_material_scalar_is_second_order_cauchy": (
            scalar_cauchy[2]
            >= float(
                thresholds["original_scalar_cauchy_ratio_min"]
            )
        ),
        "body_trace_is_second_order_cauchy": (
            trace_cauchy[2]
            >= float(thresholds["body_trace_cauchy_ratio_min"])
            and max(finest_ratios)
            <= float(
                thresholds["finest_change_over_evolution_max"]
            )
        ),
        "no_inference_iteration_or_input_mutation_enters": (
            mutation
            <= float(thresholds["input_state_mutation_abs_max"])
            and inferred
            <= int(thresholds["inferred_scalar_count_max"])
            and geometry_iterations
            <= int(thresholds["geometry_iteration_count_max"])
        ),
    }
    checks = {name: bool(value) for name, value in checks.items()}
    result = {
        "artifact": "actual_wake_repeated_insertion_oracle",
        "claim_node": contract["claim_node"],
        "stage": contract["stage"],
        "stage_decision": (
            "GO" if all(checks.values()) else "NO-GO"
        ),
        "checks": checks,
        "aggregate_metrics": {
            "step_families": step_families,
            "minimum_consecutive_insertions": minimum_insertions,
            "band_increment_mismatch_max": band_mismatch,
            "material_identity_count_mismatch_max": (
                identity_count_mismatch
            ),
            "old_transport_normalized_residual_max": (
                transport_residual
            ),
            "old_mass_rank_deficiency_max": mass_rank_deficiency,
            "old_mass_condition_number_max": mass_condition,
            "actual_boundary_relative_weak_residual_max": (
                boundary_residual
            ),
            "algebraic_trace_residual_abs_max": (
                algebraic_residual
            ),
            "free_state_preservation_abs_max": preservation,
            "characteristic_birth_identity_abs_max": birth_identity,
            "chronological_seam_abs_max": seam,
            "P2_roundtrip_abs_max": roundtrip,
            "geometry_identity_abs_max": geometry_identity,
            "minimum_newborn_face_area": minimum_area,
            "original_geometry_cauchy": geometry_cauchy,
            "original_scalar_cauchy": scalar_cauchy,
            "body_trace_cauchy": trace_cauchy,
            "finest_change_over_evolution": finest_ratios,
            "input_state_mutation_abs_max": mutation,
            "inferred_scalar_count": inferred,
            "geometry_iteration_count": geometry_iterations,
        },
        "forbidden_quantities_absent": [
            "newborn_initial_state",
            "old_state_remap",
            "scalar_copy",
            "scalar_average",
            "scalar_clamp",
            "blob_radius",
            "epsilon_offset",
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
