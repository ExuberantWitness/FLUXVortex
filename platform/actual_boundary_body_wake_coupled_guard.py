"""Run the preregistered S3b coupled actual-boundary/body-wake oracle."""
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
from claim_runtime.actual_boundary_body_wake import (  # noqa: E402
    ActualBoundaryBodyWakeSolution,
    solve_actual_boundary_body_wake_p2,
)
from claim_runtime.classified_p2_cut_topology import (  # noqa: E402
    classified_p2_cut_topology,
)


CASES = (
    HERE / "docs" / "diag"
    / "actual_boundary_body_wake_coupled_cases.yaml"
)
RESULTS = (
    HERE / "docs" / "diag"
    / "actual_boundary_body_wake_coupled_results.json"
)


def _relative_change(first: float, second: float) -> float:
    return float(
        abs(second - first)
        / max(abs(second), np.finfo(float).tiny)
    )


def _case_record(
    solution: ActualBoundaryBodyWakeSolution,
    *,
    alpha_deg: float,
    downstream_edge_x: float,
    quadrature_order: int,
) -> dict:
    continuity = solution.wake.surface.continuity_report()
    return {
        "alpha_deg": alpha_deg,
        "downstream_edge_x": downstream_edge_x,
        "quadrature_order": quadrature_order,
        "rank": solution.rank,
        "body_unknown_count": solution.body_unknown_count,
        "independent_wake_unknown_count": (
            solution.independent_wake_unknown_count
        ),
        "condition_number": solution.condition_number,
        "relative_weak_residual": solution.relative_weak_residual,
        "wake_attachment_error": solution.wake_attachment_error,
        "body_cut_jump": solution.body_cut_jump.tolist(),
        "wake_internal_trace_jump": max(
            continuity.max_trace_node_jump,
            continuity.max_trace_jump,
        ),
        "body_paired_topology_counts": (
            solution.body_paired_topology_counts
        ),
        "body_wake_paired_topology_counts": (
            solution.body_wake_paired_topology_counts
        ),
    }


def run() -> dict:
    contract = yaml.safe_load(CASES.read_text(encoding="utf-8"))
    canonical = contract["canonical"]
    thresholds = contract["thresholds"]
    (
        mesh,
        upper_faces,
        lower_faces,
        cut_edges,
        zero_endpoints,
    ) = build_canonical_diamond_wing()
    topology = classified_p2_cut_topology(
        mesh,
        upper_face_indices=upper_faces,
        lower_face_indices=lower_faces,
        cut_edges=cut_edges,
        zero_jump_end_vertices=zero_endpoints,
    )
    speed = float(canonical["freestream"]["speed"])
    downstream_values = [
        float(value)
        for value in canonical["steady_material_wake"][
            "downstream_edge_x"
        ]
    ]
    orders = [
        int(value)
        for value in canonical[
            "paired_singular_quadrature_orders"
        ]
    ]
    cache: dict[tuple[float, float, int], ActualBoundaryBodyWakeSolution] = {}

    def solve(
        alpha_deg: float,
        downstream_edge_x: float,
        order: int,
    ) -> ActualBoundaryBodyWakeSolution:
        key = (alpha_deg, downstream_edge_x, order)
        if key not in cache:
            alpha = np.deg2rad(alpha_deg)
            incident = np.repeat(
                np.array(
                    ((speed * np.cos(alpha), 0.0, speed * np.sin(alpha)),)
                ),
                len(mesh.faces),
                axis=0,
            )
            cache[key] = solve_actual_boundary_body_wake_p2(
                mesh,
                topology,
                incident_velocity=incident,
                downstream_edge_x=downstream_edge_x,
                target_quadrature_order=order,
                source_quadrature_order=order,
            )
        return cache[key]

    reference_x = float(
        canonical["reference_case"]["downstream_edge_x"]
    )
    reference_q = int(
        canonical["reference_case"]["quadrature_order"]
    )
    positive = solve(5.0, reference_x, reference_q)
    negative = solve(-5.0, reference_x, reference_q)
    zero = solve(0.0, reference_x, reference_q)
    length_cases = [
        solve(5.0, downstream, reference_q)
        for downstream in downstream_values
    ]
    order_cases = [
        solve(5.0, reference_x, order)
        for order in orders
    ]
    all_cases = list({
        id(solution): solution
        for solution in (
            [positive, negative, zero]
            + length_cases
            + order_cases
        )
    }.values())

    rank_deficiency = max(
        solution.body_unknown_count - solution.rank
        for solution in all_cases
    )
    weak_residual = max(
        solution.relative_weak_residual
        for solution in all_cases
    )
    condition_number = max(
        solution.condition_number for solution in all_cases
    )
    wake_attachment = max(
        solution.wake_attachment_error for solution in all_cases
    )
    zero_jump = float(
        np.max(np.abs(zero.body_cut_jump), initial=0.0)
    )
    antisymmetry = float(
        np.max(
            np.abs(positive.body_cut_jump + negative.body_cut_jump),
            initial=0.0,
        )
    )
    mirror = float(
        np.max(
            np.abs(
                positive.body_cut_jump
                - positive.body_cut_jump[::-1]
            ),
            initial=0.0,
        )
    )
    tip_jump = float(max(
        abs(positive.body_cut_jump[0]),
        abs(positive.body_cut_jump[-1]),
    ))
    root_index = len(positive.body_cut_jump) // 2
    reference_root = float(positive.body_cut_jump[root_index])
    far_wake_change = _relative_change(
        float(length_cases[-2].body_cut_jump[root_index]),
        float(length_cases[-1].body_cut_jump[root_index]),
    )
    quadrature_change = _relative_change(
        float(order_cases[-2].body_cut_jump[root_index]),
        float(order_cases[-1].body_cut_jump[root_index]),
    )
    wake_row_change = max(
        float(
            np.max(
                np.abs(
                    solution.wake.potential_jump_rows
                    - solution.wake.potential_jump_rows[0]
                ),
                initial=0.0,
            )
        )
        for solution in all_cases
    )
    wake_internal_jump = max(
        max(
            solution.wake.surface.continuity_report().max_trace_node_jump,
            solution.wake.surface.continuity_report().max_trace_jump,
        )
        for solution in all_cases
    )

    checks = {
        "square_full_rank_body_system": (
            rank_deficiency
            <= int(thresholds["rank_deficiency_max"])
            and all(
                solution.matrix.shape
                == (
                    solution.body_unknown_count,
                    solution.body_unknown_count,
                )
                for solution in all_cases
            )
        ),
        "no_independent_wake_amplitude": all(
            solution.independent_wake_unknown_count == 0
            for solution in all_cases
        ),
        "weak_equation_residual": (
            weak_residual
            <= float(thresholds["normalized_weak_residual_max"])
        ),
        "condition_number": (
            condition_number
            <= float(thresholds["condition_number_max"])
        ),
        "wake_attaches_to_body_jump": (
            wake_attachment
            <= float(thresholds["wake_attachment_abs_error_max"])
        ),
        "zero_incidence_has_zero_jump": (
            zero_jump
            <= float(thresholds["zero_alpha_jump_abs_max"])
        ),
        "incidence_antisymmetry": (
            antisymmetry
            <= float(
                thresholds["incidence_antisymmetry_abs_max"]
            )
        ),
        "full_wing_span_mirror": (
            mirror <= float(thresholds["span_mirror_abs_max"])
        ),
        "tip_jump_is_zero": (
            tip_jump <= float(thresholds["tip_jump_abs_max"])
        ),
        "reference_circulation_is_nonzero": (
            abs(reference_root)
            >= float(thresholds["reference_root_jump_abs_min"])
        ),
        "far_wake_refinement_converges": (
            far_wake_change
            <= float(
                thresholds["far_wake_root_relative_change_max"]
            )
        ),
        "paired_quadrature_converges": (
            quadrature_change
            <= float(
                thresholds[
                    "quadrature_root_relative_change_max"
                ]
            )
        ),
        "steady_material_rows_are_identical": (
            wake_row_change
            <= float(
                thresholds["wake_material_row_change_abs_max"]
            )
        ),
        "wake_is_internally_p2_continuous": (
            wake_internal_jump
            <= float(
                thresholds["wake_internal_trace_jump_abs_max"]
            )
        ),
    }
    checks = {name: bool(value) for name, value in checks.items()}
    result = {
        "artifact": "actual_boundary_body_wake_coupled_oracle",
        "claim_node": contract["claim_node"],
        "stage": contract["stage"],
        "canonical": canonical,
        "cases": [
            _case_record(
                solution,
                alpha_deg=key[0],
                downstream_edge_x=key[1],
                quadrature_order=key[2],
            )
            for key, solution in cache.items()
        ],
        "aggregate_metrics": {
            "rank_deficiency_max": rank_deficiency,
            "normalized_weak_residual_max": weak_residual,
            "condition_number_max": condition_number,
            "wake_attachment_abs_error_max": wake_attachment,
            "zero_alpha_jump_abs_max": zero_jump,
            "incidence_antisymmetry_abs_max": antisymmetry,
            "span_mirror_abs_max": mirror,
            "tip_jump_abs_max": tip_jump,
            "reference_root_jump": reference_root,
            "far_wake_root_relative_change": far_wake_change,
            "quadrature_root_relative_change": quadrature_change,
            "wake_material_row_change_abs_max": wake_row_change,
            "wake_internal_trace_jump_abs_max": wake_internal_jump,
        },
        "thresholds": thresholds,
        "checks": checks,
        "stage_decision": (
            "GO" if all(checks.values()) else "NO-GO"
        ),
        "production_activation_allowed": False,
        "interpretation": (
            "A full-rank algebraic solve is not sufficient evidence. "
            "Failure of the frozen quadrature or symmetry gates means the "
            "pairwise body/wake-edge singular assembly cannot be promoted; "
            "the equation remains a diagnostic shadow with no pressure or "
            "force authority."
        ),
    }
    RESULTS.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, sort_keys=True))
