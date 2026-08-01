"""Run the preregistered S3c shape-regular material-wake history oracle."""
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
    / "actual_boundary_material_wake_history_cases.yaml"
)
RESULTS = (
    HERE / "docs" / "diag"
    / "actual_boundary_material_wake_history_results.json"
)


def _relative_change(first: float, second: float) -> float:
    return float(
        abs(second - first)
        / max(abs(second), np.finfo(float).tiny)
    )


def _nonincreasing(values: list[float]) -> bool:
    return all(
        following <= previous
        for previous, following in zip(values, values[1:])
    )


def _maximum_band_aspect_ratio(
    solution: ActualBoundaryBodyWakeSolution,
) -> float:
    ratios = []
    for band in solution.wake_history.bands:
        previous = band.surface.vertices[: band.span_nodes]
        current = band.surface.vertices[band.span_nodes :]
        chord = float(
            np.max(np.linalg.norm(previous - current, axis=1))
        )
        span = float(
            np.min(
                np.linalg.norm(
                    np.diff(previous, axis=0),
                    axis=1,
                )
            )
        )
        ratios.append(chord / span)
    return max(ratios)


def run() -> dict:
    contract = yaml.safe_load(CASES.read_text(encoding="utf-8"))
    canonical = contract["canonical"]
    thresholds = contract["thresholds"]
    (
        mesh,
        upper_faces,
        lower_faces,
        cut_edges,
        endpoints,
    ) = build_canonical_diamond_wing()
    topology = classified_p2_cut_topology(
        mesh,
        upper_face_indices=upper_faces,
        lower_face_indices=lower_faces,
        cut_edges=cut_edges,
        zero_jump_end_vertices=endpoints,
    )
    speed = float(canonical["freestream"]["speed"])
    band_chord = float(canonical["wake"]["band_chord"])
    cutoffs = [
        float(value) for value in canonical["wake"]["cutoff_x"]
    ]
    orders = [
        int(value)
        for value in canonical[
            "paired_singular_quadrature_orders"
        ]
    ]
    reference_cutoff = float(canonical["reference"]["cutoff_x"])
    reference_order = int(
        canonical["reference"]["quadrature_order"]
    )
    cache: dict[tuple[float, float, int], ActualBoundaryBodyWakeSolution] = {}

    def solve(
        alpha_deg: float,
        cutoff_x: float,
        order: int,
    ) -> ActualBoundaryBodyWakeSolution:
        key = (alpha_deg, cutoff_x, order)
        if key not in cache:
            alpha = np.deg2rad(alpha_deg)
            incident = np.repeat(
                np.array(
                    ((speed * np.cos(alpha), 0.0, speed * np.sin(alpha)),)
                ),
                len(mesh.faces),
                axis=0,
            )
            edge_x = np.arange(
                1.0,
                cutoff_x + 0.5 * band_chord,
                band_chord,
            )
            edge_x[-1] = cutoff_x
            cache[key] = solve_actual_boundary_body_wake_p2(
                mesh,
                topology,
                incident_velocity=incident,
                downstream_edge_x=cutoff_x,
                wake_edge_x_nodes=edge_x,
                target_quadrature_order=order,
                source_quadrature_order=order,
            )
        return cache[key]

    order_triplets = [
        (
            solve(5.0, reference_cutoff, order),
            solve(-5.0, reference_cutoff, order),
            solve(0.0, reference_cutoff, order),
        )
        for order in orders
    ]
    length_cases = [
        solve(5.0, cutoff, reference_order)
        for cutoff in cutoffs
    ]
    all_cases = list({
        id(solution): solution
        for triplet in order_triplets
        for solution in triplet
    }.values())
    all_cases.extend(
        solution
        for solution in length_cases
        if all(solution is not existing for existing in all_cases)
    )
    reference = order_triplets[-1][0]
    scale = max(
        abs(float(reference.body_cut_jump[
            len(reference.body_cut_jump) // 2
        ])),
        np.finfo(float).tiny,
    )

    zero_errors = [
        float(np.max(np.abs(zero.body_cut_jump), initial=0.0))
        / scale
        for _, _, zero in order_triplets
    ]
    antisymmetry_errors = [
        float(
            np.max(
                np.abs(
                    positive.body_cut_jump + negative.body_cut_jump
                ),
                initial=0.0,
            )
        )
        / scale
        for positive, negative, _ in order_triplets
    ]
    mirror_errors = [
        float(
            np.max(
                np.abs(
                    positive.body_cut_jump
                    - positive.body_cut_jump[::-1]
                ),
                initial=0.0,
            )
        )
        / scale
        for positive, _, _ in order_triplets
    ]
    root_index = len(reference.body_cut_jump) // 2
    quadrature_change = _relative_change(
        float(order_triplets[-2][0].body_cut_jump[root_index]),
        float(order_triplets[-1][0].body_cut_jump[root_index]),
    )
    far_wake_change = _relative_change(
        float(length_cases[-2].body_cut_jump[root_index]),
        float(length_cases[-1].body_cut_jump[root_index]),
    )
    aspect_ratio = max(
        _maximum_band_aspect_ratio(solution)
        for solution in all_cases
    )
    history_reports = [
        solution.wake_history.continuity_report()
        for solution in all_cases
    ]
    time_gap = max(report.max_time_gap for report in history_reports)
    geometry_gap = max(
        report.max_geometry_gap for report in history_reports
    )
    trace_jump = max(
        report.max_trace_jump for report in history_reports
    )
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
    independent_wake_unknowns = max(
        solution.independent_wake_unknown_count
        for solution in all_cases
    )
    common_edge_pairs = {
        solution.body_wake_paired_topology_counts["common_edge"]
        for solution in all_cases
    }
    attachment_error = max(
        solution.wake_attachment_error for solution in all_cases
    )
    tip_jump = max(
        max(
            abs(solution.body_cut_jump[0]),
            abs(solution.body_cut_jump[-1]),
        )
        for solution in all_cases
    )

    checks = {
        "bands_are_shape_regular": (
            aspect_ratio
            <= float(
                thresholds["maximum_band_aspect_ratio_max"]
            )
        ),
        "history_interfaces_are_exact": (
            all(report.compatible for report in history_reports)
            and time_gap
            <= float(thresholds["history_time_gap_abs_max"])
            and geometry_gap
            <= float(
                thresholds["history_geometry_gap_abs_max"]
            )
            and trace_jump
            <= float(thresholds["history_trace_jump_abs_max"])
        ),
        "only_newest_band_has_body_common_edges": (
            common_edge_pairs
            == {
                int(
                    thresholds[
                        "body_wake_common_edge_pair_count"
                    ]
                )
            }
        ),
        "one_circulation_ledger": (
            independent_wake_unknowns
            <= int(
                thresholds[
                    "independent_wake_unknown_count_max"
                ]
            )
        ),
        "full_rank_weak_system": (
            rank_deficiency
            <= int(thresholds["rank_deficiency_max"])
            and weak_residual
            <= float(thresholds["normalized_weak_residual_max"])
            and condition_number
            <= float(thresholds["condition_number_max"])
        ),
        "symmetry_errors_decrease_with_quadrature": (
            _nonincreasing(zero_errors)
            and _nonincreasing(antisymmetry_errors)
            and _nonincreasing(mirror_errors)
        ),
        "reference_zero_alpha": (
            zero_errors[-1]
            <= float(
                thresholds["normalized_zero_alpha_jump_max"]
            )
        ),
        "reference_incidence_antisymmetry": (
            antisymmetry_errors[-1]
            <= float(
                thresholds[
                    "normalized_incidence_antisymmetry_max"
                ]
            )
        ),
        "reference_span_mirror": (
            mirror_errors[-1]
            <= float(
                thresholds["normalized_span_mirror_max"]
            )
        ),
        "quadrature_refinement_converges": (
            quadrature_change
            <= float(
                thresholds[
                    "quadrature_root_relative_change_max"
                ]
            )
        ),
        "far_wake_refinement_converges": (
            far_wake_change
            <= float(
                thresholds["far_wake_root_relative_change_max"]
            )
        ),
        "tip_and_attachment_are_exact": (
            tip_jump <= float(thresholds["tip_jump_abs_max"])
            and attachment_error
            <= float(
                thresholds["wake_attachment_abs_error_max"]
            )
        ),
    }
    checks = {name: bool(value) for name, value in checks.items()}
    result = {
        "artifact": "actual_boundary_material_wake_history_oracle",
        "claim_node": contract["claim_node"],
        "stage": contract["stage"],
        "canonical": canonical,
        "case_summary": [
            {
                "alpha_deg": key[0],
                "cutoff_x": key[1],
                "quadrature_order": key[2],
                "bands": len(solution.wake_history.bands),
                "root_jump": float(
                    solution.body_cut_jump[root_index]
                ),
                "rank": solution.rank,
                "condition_number": solution.condition_number,
                "relative_weak_residual": (
                    solution.relative_weak_residual
                ),
            }
            for key, solution in cache.items()
        ],
        "aggregate_metrics": {
            "maximum_band_aspect_ratio": aspect_ratio,
            "history_time_gap_abs_max": time_gap,
            "history_geometry_gap_abs_max": geometry_gap,
            "history_trace_jump_abs_max": trace_jump,
            "body_wake_common_edge_pair_counts": sorted(
                common_edge_pairs
            ),
            "independent_wake_unknown_count_max": (
                independent_wake_unknowns
            ),
            "rank_deficiency_max": rank_deficiency,
            "normalized_weak_residual_max": weak_residual,
            "condition_number_max": condition_number,
            "normalized_zero_alpha_by_order": dict(
                zip(map(str, orders), zero_errors)
            ),
            "normalized_incidence_antisymmetry_by_order": dict(
                zip(map(str, orders), antisymmetry_errors)
            ),
            "normalized_span_mirror_by_order": dict(
                zip(map(str, orders), mirror_errors)
            ),
            "quadrature_root_relative_change": quadrature_change,
            "far_wake_root_relative_change": far_wake_change,
            "tip_jump_abs_max": tip_jump,
            "wake_attachment_abs_error_max": attachment_error,
        },
        "thresholds": thresholds,
        "checks": checks,
        "stage_decision": (
            "GO" if all(checks.values()) else "NO-GO"
        ),
        "production_activation_allowed": False,
        "interpretation": (
            "Passing would validate only the shape-regular steady "
            "multi-band equation. It would not validate unsteady Kelvin "
            "history, pressure, force or production."
        ),
    }
    RESULTS.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, sort_keys=True))
