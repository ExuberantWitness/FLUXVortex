"""Run preregistered S1b fixed-mesh quadrature attribution."""
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


CASES = (
    HERE / "docs" / "diag" / "actual_boundary_p2_galerkin_cases.yaml"
)
REFINEMENT = (
    HERE / "docs" / "diag"
    / "actual_boundary_p2_refinement_results.json"
)
RESULTS = (
    HERE / "docs" / "diag"
    / "actual_boundary_p2_quadrature_results.json"
)


def run() -> dict:
    contract = yaml.safe_load(CASES.read_text(encoding="utf-8"))
    previous = json.loads(REFINEMENT.read_text(encoding="utf-8"))
    order6 = previous["formal_sequence_level1_to_level3"][1]
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
    order10, potential = _case(2, 10, offbody)
    order10["offbody_potential_relative_error"] = float(
        np.max(np.abs(potential - analytic), initial=0.0) / scale
    )
    changes = {
        name: abs(order10[name] - order6[name])
        for name in (
            "offbody_potential_relative_error",
            "surface_velocity_rms_error",
            "surface_Cp_rms_error",
        )
    }
    equation_checks = {
        "weak_residual": order10["relative_weak_residual"]
        <= float(contract["thresholds"]["weak_relative_residual_max"]),
        "continuity": order10["continuity_residual"]
        <= float(contract["thresholds"]["continuity_residual_max"]),
        "source_flux": order10["relative_source_flux"]
        <= float(
            contract["thresholds"][
                "prescribed_source_flux_relative_max"
            ]
        ),
        "condition_number": order10["condition_number"]
        <= float(contract["thresholds"]["matrix_condition_number_max"]),
    }
    geometry_limited = (
        changes["surface_Cp_rms_error"] <= 0.005
        and all(equation_checks.values())
    )
    result = {
        "artifact": "actual_boundary_p2_fixed_mesh_quadrature_attribution",
        "claim_node": "N3.1j3b6d3",
        "stage": "S1b_quadrature_attribution",
        "level2_order6": order6,
        "level2_order10": order10,
        "absolute_metric_changes": changes,
        "equation_checks": equation_checks,
        "diagnosis": (
            "geometry_or_resolution_limited"
            if geometry_limited
            else "weak_quadrature_limited"
        ),
        "geometry_limited_gate": geometry_limited,
        "production_activation_allowed": False,
    }
    RESULTS.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, sort_keys=True))
