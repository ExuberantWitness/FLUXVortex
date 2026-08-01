"""Run preregistered S1d paired singular Galerkin gates."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from actual_boundary_p2_galerkin_guard import _case  # noqa: E402
from claim_runtime.actual_boundary_p2_galerkin import (  # noqa: E402
    paired_p2_triangle_integral,
)
from claim_runtime.distributed_doublet import (  # noqa: E402
    _triangle_quadrature,
    p2_shape_values,
)


CASES = (
    HERE / "docs" / "diag" / "actual_boundary_p2_galerkin_cases.yaml"
)
RESULTS = (
    HERE / "docs" / "diag"
    / "actual_boundary_p2_paired_results.json"
)


def _product_pair(
    target: np.ndarray,
    source: np.ndarray,
    order: int,
) -> tuple[np.ndarray, np.ndarray]:
    barycentric, reference_weight = _triangle_quadrature(order)
    target_points = barycentric @ target
    source_points = barycentric @ source
    target_shape = p2_shape_values(barycentric)
    source_shape = p2_shape_values(barycentric)
    target_area_vector = np.cross(
        target[1] - target[0], target[2] - target[0]
    )
    source_area_vector = np.cross(
        source[1] - source[0], source[2] - source[0]
    )
    target_weight = (
        reference_weight * np.linalg.norm(target_area_vector)
    )
    source_weight = (
        reference_weight * np.linalg.norm(source_area_vector)
    )
    separation = (
        target_points[:, None, :] - source_points[None, :, :]
    )
    radius_square = np.einsum(
        "tsj,tsj->ts", separation, separation
    )
    radius = np.sqrt(radius_square)
    source_normal = source_area_vector / np.linalg.norm(
        source_area_vector
    )
    doublet_kernel = -(
        separation @ source_normal
    ) / (4.0 * np.pi * radius_square * radius)
    source_kernel = 1.0 / (4.0 * np.pi * radius)
    pair_weight = target_weight[:, None] * source_weight[None, :]
    doublet_block = np.einsum(
        "ti,ts,sj->ij",
        target_shape,
        pair_weight * doublet_kernel,
        source_shape,
    )
    source_vector = np.einsum(
        "ti,ts->i",
        target_shape,
        pair_weight * source_kernel,
    )
    return doublet_block, source_vector


def _relative_change(first: np.ndarray, second: np.ndarray) -> float:
    scale = max(
        float(np.max(np.abs(second), initial=0.0)),
        np.finfo(float).tiny,
    )
    return float(
        np.max(np.abs(second - first), initial=0.0) / scale
    )


def _pair_canonical(thresholds: dict) -> dict:
    target = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    )
    geometries = {
        "common_edge": (
            np.array(
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
            ),
            np.array([0, 1, 3]),
        ),
        "common_vertex": (
            np.array(
                [[0.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
            ),
            np.array([0, 3, 4]),
        ),
        "common_triangle": (target.copy(), np.array([0, 1, 2])),
    }
    target_ids = np.array([0, 1, 2])
    results: dict[str, list[dict]] = {}
    area_errors = []
    for label, (source, source_ids) in geometries.items():
        sequence = []
        for order in (4, 6, 8):
            integral = paired_p2_triangle_integral(
                target,
                source,
                target_vertex_ids=target_ids,
                source_vertex_ids=source_ids,
                quadrature_order=order,
            )
            exact_area_product = 0.25
            area_error = abs(
                integral.partition_measure - exact_area_product
            ) / exact_area_product
            area_errors.append(area_error)
            sequence.append({
                "order": order,
                "area_partition_relative_error": area_error,
                "doublet_block": integral.doublet_block.tolist(),
                "source_vector": integral.source_vector.tolist(),
            })
        sequence[-1]["finest_doublet_relative_change"] = (
            _relative_change(
                np.asarray(sequence[-2]["doublet_block"]),
                np.asarray(sequence[-1]["doublet_block"]),
            )
        )
        sequence[-1]["finest_source_relative_change"] = (
            _relative_change(
                np.asarray(sequence[-2]["source_vector"]),
                np.asarray(sequence[-1]["source_vector"]),
            )
        )
        results[label] = sequence

    separated_source = np.array(
        [[2.0, -0.2, 0.8], [3.1, 0.1, 0.7], [2.2, 0.9, 1.0]]
    )
    separated12 = _product_pair(target, separated_source, 12)
    separated24 = _product_pair(target, separated_source, 24)
    separated_doublet_error = _relative_change(
        separated12[0], separated24[0]
    )
    separated_source_error = _relative_change(
        separated12[1], separated24[1]
    )
    edge_finest = results["common_edge"][-1]
    vertex_finest = results["common_vertex"][-1]
    identical_finest = results["common_triangle"][-1]
    checks = {
        "area_partition": max(area_errors)
        <= float(thresholds["area_partition_relative_error_max"]),
        "separated_doublet": separated_doublet_error
        <= float(
            thresholds[
                "separated_doublet_block_relative_error_max"
            ]
        ),
        "separated_source": separated_source_error
        <= float(
            thresholds[
                "separated_source_vector_relative_error_max"
            ]
        ),
        "common_edge_doublet_cauchy": (
            edge_finest["finest_doublet_relative_change"]
            <= float(
                thresholds[
                    "common_edge_finest_block_relative_change_max"
                ]
            )
        ),
        "common_edge_source_cauchy": (
            edge_finest["finest_source_relative_change"]
            <= float(
                thresholds[
                    "common_edge_finest_block_relative_change_max"
                ]
            )
        ),
        "common_vertex_doublet_cauchy": (
            vertex_finest["finest_doublet_relative_change"]
            <= float(
                thresholds[
                    "common_vertex_finest_block_relative_change_max"
                ]
            )
        ),
        "common_vertex_source_cauchy": (
            vertex_finest["finest_source_relative_change"]
            <= float(
                thresholds[
                    "common_vertex_finest_block_relative_change_max"
                ]
            )
        ),
        "identical_source_cauchy": (
            identical_finest["finest_source_relative_change"]
            <= float(
                thresholds[
                    "identical_source_finest_relative_change_max"
                ]
            )
        ),
    }
    return {
        "topology_sequences": results,
        "separated_product_oracle": {
            "order12_to_order24_doublet_relative_change":
                separated_doublet_error,
            "order12_to_order24_source_relative_change":
                separated_source_error,
        },
        "checks": checks,
    }


def run() -> dict:
    contract = yaml.safe_load(CASES.read_text(encoding="utf-8"))
    specification = contract[
        "S1d_paired_singular_galerkin_preregistered_after_S1c_before_implementation"
    ]
    thresholds = specification["thresholds"]
    pair = _pair_canonical(thresholds)
    offbody = np.asarray(
        contract["canonical"]["offbody_points"], dtype=float
    )
    analytic = (
        0.5 * offbody[:, 0] / np.linalg.norm(offbody, axis=1) ** 3
    )
    potential_scale = max(
        float(np.max(np.abs(analytic), initial=0.0)),
        np.finfo(float).tiny,
    )
    level0 = []
    level0_potential = []
    for order in (4, 6, 8):
        metrics, potential = _case(
            0, order, offbody, potential_operator="paired_singular"
        )
        metrics["offbody_potential_relative_error"] = float(
            np.max(np.abs(potential - analytic), initial=0.0)
            / potential_scale
        )
        level0.append(metrics)
        level0_potential.append(potential)
    level1, level1_potential = _case(
        1, 8, offbody, potential_operator="paired_singular"
    )
    level1["offbody_potential_relative_error"] = float(
        np.max(np.abs(level1_potential - analytic), initial=0.0)
        / potential_scale
    )
    level0_changes = {
        "offbody_potential": float(
            np.max(
                np.abs(level0_potential[-1] - level0_potential[-2]),
                initial=0.0,
            )
            / potential_scale
        ),
        "surface_velocity": abs(
            level0[-1]["surface_velocity_rms_error"]
            - level0[-2]["surface_velocity_rms_error"]
        ),
        "surface_Cp": abs(
            level0[-1]["surface_Cp_rms_error"]
            - level0[-2]["surface_Cp_rms_error"]
        ),
    }
    sphere_checks = {
        "weak_residual": max(
            item["relative_weak_residual"] for item in level0 + [level1]
        ) <= float(thresholds["weak_relative_residual_max"]),
        "continuity": max(
            item["continuity_residual"] for item in level0 + [level1]
        ) <= float(thresholds["continuity_residual_max"]),
        "level0_potential_cauchy": (
            level0_changes["offbody_potential"] <= 0.005
        ),
        "level0_velocity_cauchy": (
            level0_changes["surface_velocity"] <= 0.005
        ),
        "level0_Cp_cauchy": level0_changes["surface_Cp"] <= 0.005,
        "level1_offbody_potential": (
            level1["offbody_potential_relative_error"]
            <= float(
                thresholds[
                    "finest_offbody_potential_relative_error_max"
                ]
            )
        ),
        "level1_surface_velocity": (
            level1["surface_velocity_rms_error"]
            <= float(
                thresholds[
                    "finest_surface_velocity_rms_error_max"
                ]
            )
        ),
        "level1_surface_Cp": (
            level1["surface_Cp_rms_error"]
            <= float(thresholds["finest_surface_Cp_rms_error_max"])
        ),
    }
    checks = {**pair["checks"], **sphere_checks}
    result = {
        "artifact": "actual_boundary_p2_paired_singular_galerkin",
        "claim_node": "N3.1j3b6d3",
        "stage": "S1d_paired_singular_galerkin",
        "pair_canonical": pair,
        "level0_order_cauchy": level0,
        "level0_order6_to_order8_changes": level0_changes,
        "level1": level1,
        "sphere_checks": sphere_checks,
        "checks": checks,
        "stage_decision": "GO" if all(checks.values()) else "NO-GO",
        "production_activation_allowed": False,
        "interpretation": (
            "A failed pair canonical localizes the numerical component to "
            "that topology/kernel. Passing pair canonicals but failing the "
            "sphere would move attribution to geometry or surface gradient; "
            "no aerodynamic constant is available in this stage."
        ),
    }
    RESULTS.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, sort_keys=True))
