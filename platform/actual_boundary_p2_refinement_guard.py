"""Run preregistered S1a flat-facet refinement attribution."""
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
S1_RESULTS = (
    HERE / "docs" / "diag" / "actual_boundary_p2_galerkin_results.json"
)
RESULTS = (
    HERE / "docs" / "diag"
    / "actual_boundary_p2_refinement_results.json"
)


def run() -> dict:
    contract = yaml.safe_load(CASES.read_text(encoding="utf-8"))
    extension = contract[
        "S1a_refinement_extension_preregistered_after_S1_no_go_before_execution"
    ]
    formal = json.loads(S1_RESULTS.read_text(encoding="utf-8"))
    offbody = np.asarray(
        contract["canonical"]["offbody_points"], dtype=float
    )
    analytic = (
        0.5
        * offbody[:, 0]
        / np.linalg.norm(offbody, axis=1) ** 3
    )
    potential_scale = max(
        float(np.max(np.abs(analytic), initial=0.0)),
        np.finfo(float).tiny,
    )

    level0_order6, potential0_order6 = _case(0, 6, offbody)
    level0_order6["offbody_potential_relative_error"] = float(
        np.max(np.abs(potential0_order6 - analytic), initial=0.0)
        / potential_scale
    )
    level0_order16 = formal["level0_quadrature_cauchy"][-1]
    quadrature_bias = {
        "offbody_potential_error_change": abs(
            level0_order6["offbody_potential_relative_error"]
            - level0_order16["offbody_potential_relative_error"]
        ),
        "surface_velocity_rms_error_change": abs(
            level0_order6["surface_velocity_rms_error"]
            - level0_order16["surface_velocity_rms_error"]
        ),
        "surface_Cp_rms_error_change": abs(
            level0_order6["surface_Cp_rms_error"]
            - level0_order16["surface_Cp_rms_error"]
        ),
    }
    extension_cases = []
    for level in map(int, extension["formal_mesh_levels"]):
        metrics, potential = _case(
            level, int(extension["quadrature_order"]), offbody
        )
        metrics["offbody_potential_relative_error"] = float(
            np.max(np.abs(potential - analytic), initial=0.0)
            / potential_scale
        )
        extension_cases.append(metrics)

    level1 = formal["mesh_refinement"][-1]
    sequence = [level1] + extension_cases
    error_names = (
        "offbody_potential_relative_error",
        "surface_velocity_rms_error",
        "surface_Cp_rms_error",
    )
    monotone = {
        name: all(
            later[name] < earlier[name]
            for earlier, later in zip(
                sequence[:-1], sequence[1:], strict=True
            )
        )
        for name in error_names
    }
    finest = extension_cases[-1]
    accuracy = extension["unchanged_accuracy_thresholds"]
    checks = {
        "quadrature_bias_potential": (
            quadrature_bias["offbody_potential_error_change"] <= 0.005
        ),
        "quadrature_bias_velocity": (
            quadrature_bias["surface_velocity_rms_error_change"] <= 0.01
        ),
        "quadrature_bias_Cp": (
            quadrature_bias["surface_Cp_rms_error_change"] <= 0.02
        ),
        "potential_monotone": monotone[
            "offbody_potential_relative_error"
        ],
        "velocity_monotone": monotone[
            "surface_velocity_rms_error"
        ],
        "Cp_monotone": monotone["surface_Cp_rms_error"],
        "potential_accuracy": (
            finest["offbody_potential_relative_error"]
            <= float(accuracy["offbody_potential_relative_error_max"])
        ),
        "velocity_accuracy": (
            finest["surface_velocity_rms_error"]
            <= float(accuracy["surface_velocity_rms_error_max"])
        ),
        "Cp_accuracy": (
            finest["surface_Cp_rms_error"]
            <= float(accuracy["surface_Cp_rms_error_max"])
        ),
        "weak_residual": max(
            item["relative_weak_residual"] for item in extension_cases
        ) <= float(contract["thresholds"]["weak_relative_residual_max"]),
        "continuity": max(
            item["continuity_residual"] for item in extension_cases
        ) <= float(contract["thresholds"]["continuity_residual_max"]),
        "source_flux": max(
            item["relative_source_flux"] for item in extension_cases
        ) <= float(
            contract["thresholds"][
                "prescribed_source_flux_relative_max"
            ]
        ),
        "condition_number": max(
            item["condition_number"] for item in extension_cases
        ) <= float(
            contract["thresholds"]["matrix_condition_number_max"]
        ),
    }
    result = {
        "artifact": "actual_boundary_p2_flat_geometry_refinement",
        "claim_node": "N3.1j3b6d3",
        "stage": "S1a_refinement_attribution",
        "level0_quadrature_bias": quadrature_bias,
        "formal_sequence_level1_to_level3": sequence,
        "monotone": monotone,
        "unchanged_accuracy_thresholds": accuracy,
        "checks": checks,
        "stage_decision": (
            "GO" if all(checks.values()) else "NO-GO"
        ),
        "scope": (
            "attached sphere flat-facet resolution only; original S1 "
            "20/80-face NO-GO remains recorded"
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
