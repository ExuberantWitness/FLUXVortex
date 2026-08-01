"""Run preregistered S1c analytic-radial potential attribution."""
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
    element_basis_doublet_potential_line_reduced,
)
from claim_runtime.distributed_doublet import (  # noqa: E402
    QuadraticDoubletElement,
)
from claim_runtime.doublet_potential import (  # noqa: E402
    element_doublet_potential,
)
from claim_runtime.thick_body_neumann_shadow import (  # noqa: E402
    constant_source_polygon_influence,
)


CASES = (
    HERE / "docs" / "diag" / "actual_boundary_p2_galerkin_cases.yaml"
)
RESULTS = (
    HERE / "docs" / "diag"
    / "actual_boundary_p2_line_reduced_results.json"
)


def _element_gates(thresholds: dict) -> dict:
    triangle = np.array(
        [[0.0, 0.0, 0.0], [1.2, 0.1, 0.0], [0.15, 0.95, 0.0]]
    )
    points = np.array(
        [[0.2, 0.3, 0.7], [1.4, -0.5, 1.1], [-0.3, 0.5, -0.6]]
    )
    line = element_basis_doublet_potential_line_reduced(
        triangle, points, line_quadrature_order=32
    )
    area = np.column_stack(
        [
            element_doublet_potential(
                QuadraticDoubletElement(triangle, np.eye(6)[index]),
                points,
                quadrature_order=48,
            )
            for index in range(6)
        ]
    )
    area_scale = max(
        float(np.max(np.abs(area), initial=0.0)),
        np.finfo(float).tiny,
    )
    separated_error = float(
        np.max(np.abs(line - area), initial=0.0) / area_scale
    )
    constant_line = np.sum(line, axis=1)
    exact_constant = (
        constant_source_polygon_influence(
            triangle, points, on_surface_side="principal"
        ).velocity
        @ (
            np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
            / np.linalg.norm(
                np.cross(
                    triangle[1] - triangle[0],
                    triangle[2] - triangle[0],
                )
            )
        )
    )
    constant_scale = max(
        float(np.max(np.abs(exact_constant), initial=0.0)),
        np.finfo(float).tiny,
    )
    constant_error = float(
        np.max(
            np.abs(constant_line - exact_constant), initial=0.0
        )
        / constant_scale
    )

    neighbor_target = np.array(
        [
            [0.5, 1.0e-4, 1.0e-4],
            [0.5, 1.0e-6, 1.0e-6],
            [0.5, 1.0e-8, 1.0e-8],
        ]
    )
    near_values = [
        element_basis_doublet_potential_line_reduced(
            triangle,
            neighbor_target,
            line_quadrature_order=order,
        )
        for order in (8, 16, 32)
    ]
    near_scale = max(
        float(np.max(np.abs(near_values[-1]), initial=0.0)),
        np.finfo(float).tiny,
    )
    finest_change = float(
        np.max(
            np.abs(near_values[-1] - near_values[-2]), initial=0.0
        )
        / near_scale
    )
    return {
        "separated_area_oracle_relative_error": separated_error,
        "constant_panel_relative_error": constant_error,
        "near_edge_finest_relative_change": finest_change,
        "checks": {
            "separated_area_oracle": separated_error
            <= float(
                thresholds[
                    "separated_area_oracle_relative_error_max"
                ]
            ),
            "constant_panel": constant_error
            <= float(thresholds["constant_panel_relative_error_max"]),
            "near_edge_cauchy": finest_change
            <= float(
                thresholds[
                    "near_edge_cauchy_finest_relative_change_max"
                ]
            ),
        },
    }


def run() -> dict:
    contract = yaml.safe_load(CASES.read_text(encoding="utf-8"))
    specification = contract[
        "S1c_line_reduced_potential_preregistered_after_S1b_before_implementation"
    ]
    thresholds = specification["thresholds"]
    element = _element_gates(thresholds)
    offbody = np.asarray(
        contract["canonical"]["offbody_points"], dtype=float
    )
    analytic = (
        0.5
        * offbody[:, 0]
        / np.linalg.norm(offbody, axis=1) ** 3
    )
    scale = max(
        float(np.max(np.abs(analytic), initial=0.0)),
        np.finfo(float).tiny,
    )
    level0 = []
    level0_potential = []
    for order in (6, 10, 16):
        metrics, potential = _case(
            0, order, offbody, potential_operator="line_reduced"
        )
        metrics["offbody_potential_relative_error"] = float(
            np.max(np.abs(potential - analytic), initial=0.0) / scale
        )
        level0.append(metrics)
        level0_potential.append(potential)
    level1, level1_potential = _case(
        1, 16, offbody, potential_operator="line_reduced"
    )
    level1["offbody_potential_relative_error"] = float(
        np.max(np.abs(level1_potential - analytic), initial=0.0)
        / scale
    )
    closed_checks = {
        "weak_residual": max(
            item["relative_weak_residual"] for item in level0 + [level1]
        ) <= float(thresholds["weak_relative_residual_max"]),
        "continuity": max(
            item["continuity_residual"] for item in level0 + [level1]
        ) <= float(thresholds["continuity_residual_max"]),
        "offbody_potential": (
            level1["offbody_potential_relative_error"]
            <= float(
                thresholds[
                    "finest_offbody_potential_relative_error_max"
                ]
            )
        ),
        "surface_velocity": (
            level1["surface_velocity_rms_error"]
            <= float(
                thresholds[
                    "finest_surface_velocity_rms_error_max"
                ]
            )
        ),
        "surface_Cp": (
            level1["surface_Cp_rms_error"]
            <= float(
                thresholds["finest_surface_Cp_rms_error_max"]
            )
        ),
    }
    result = {
        "artifact": "actual_boundary_p2_line_reduced_potential",
        "claim_node": "N3.1j3b6d3",
        "stage": "S1c_line_reduced_potential",
        "element_gates": element,
        "level0_line_order_cauchy": level0,
        "level0_finest_potential_change": float(
            np.max(
                np.abs(level0_potential[-1] - level0_potential[-2]),
                initial=0.0,
            )
            / scale
        ),
        "level1": level1,
        "closed_surface_checks": closed_checks,
        "element_decision": (
            "GO" if all(element["checks"].values()) else "NO-GO"
        ),
        "closed_surface_decision": (
            "GO" if all(closed_checks.values()) else "NO-GO"
        ),
        "stage_decision": (
            "GO"
            if all(element["checks"].values())
            and all(closed_checks.values())
            else "NO-GO"
        ),
        "diagnosis": (
            "Exact radial reduction qualifies the pointwise element "
            "potential but does not remove the outer target/source-pair "
            "weak singularity. A paired singular Galerkin quadrature is "
            "still required."
        ),
        "production_activation_allowed": False,
    }
    RESULTS.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, sort_keys=True))
