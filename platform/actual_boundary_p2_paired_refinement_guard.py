"""Run preregistered S1e paired-operator mesh attribution."""
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
S1D_RESULTS = (
    HERE / "docs" / "diag" / "actual_boundary_p2_paired_results.json"
)
RESULTS = (
    HERE / "docs" / "diag"
    / "actual_boundary_p2_paired_refinement_results.json"
)


def run() -> dict:
    contract = yaml.safe_load(CASES.read_text(encoding="utf-8"))
    specification = contract[
        "S1e_paired_mesh_attribution_preregistered_after_S1d_before_execution"
    ]
    previous = json.loads(S1D_RESULTS.read_text(encoding="utf-8"))
    level1 = previous["level1"]
    level = int(specification["formal_mesh_level"])
    order = int(specification["target_and_pair_quadrature_order"])
    offbody = np.asarray(
        contract["canonical"]["offbody_points"], dtype=float
    )
    analytic = (
        0.5 * offbody[:, 0] / np.linalg.norm(offbody, axis=1) ** 3
    )
    scale = max(
        float(np.max(np.abs(analytic), initial=0.0)),
        np.finfo(float).tiny,
    )
    level2, potential = _case(
        level,
        order,
        offbody,
        potential_operator="paired_singular",
    )
    level2["offbody_potential_relative_error"] = float(
        np.max(np.abs(potential - analytic), initial=0.0) / scale
    )
    error_names = (
        "offbody_potential_relative_error",
        "surface_velocity_rms_error",
        "surface_Cp_rms_error",
    )
    monotone = {
        name: level2[name] < level1[name] for name in error_names
    }
    thresholds = specification["unchanged_thresholds"]
    expected_topology_counts = {
        "common_triangle": level2["face_count"],
        "common_edge": 3 * level2["face_count"],
    }
    topology = level2["paired_topology_counts"]
    checks = {
        "potential_monotone": monotone[
            "offbody_potential_relative_error"
        ],
        "velocity_monotone": monotone["surface_velocity_rms_error"],
        "Cp_monotone": monotone["surface_Cp_rms_error"],
        "potential_accuracy": (
            level2["offbody_potential_relative_error"]
            <= float(
                thresholds["offbody_potential_relative_error_max"]
            )
        ),
        "velocity_accuracy": (
            level2["surface_velocity_rms_error"]
            <= float(thresholds["surface_velocity_rms_error_max"])
        ),
        "Cp_accuracy": (
            level2["surface_Cp_rms_error"]
            <= float(thresholds["surface_Cp_rms_error_max"])
        ),
        "weak_residual": level2["relative_weak_residual"]
        <= float(contract["thresholds"]["weak_relative_residual_max"]),
        "continuity": level2["continuity_residual"]
        <= float(contract["thresholds"]["continuity_residual_max"]),
        "source_flux": level2["relative_source_flux"]
        <= float(
            contract["thresholds"][
                "prescribed_source_flux_relative_max"
            ]
        ),
        "condition_number": level2["condition_number"]
        <= float(contract["thresholds"]["matrix_condition_number_max"]),
        "topology_common_triangle": (
            topology["common_triangle"]
            == expected_topology_counts["common_triangle"]
        ),
        "topology_common_edge": (
            topology["common_edge"]
            == expected_topology_counts["common_edge"]
        ),
        "topology_common_vertex_nonzero": (
            topology["common_vertex"] > 0
        ),
    }
    result = {
        "artifact": "actual_boundary_p2_paired_mesh_attribution",
        "claim_node": "N3.1j3b6d3",
        "stage": "S1e_paired_mesh_attribution",
        "level1": level1,
        "level2": level2,
        "monotone": monotone,
        "unchanged_thresholds": thresholds,
        "checks": checks,
        "stage_decision": "GO" if all(checks.values()) else "NO-GO",
        "production_activation_allowed": False,
        "interpretation": (
            "A monotone but insufficient result preserves the paired weak "
            "operator and opens the preregistered level-3/isoparametric "
            "geometry decision. A nonmonotone result reopens the surface "
            "gradient representation."
        ),
    }
    RESULTS.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, sort_keys=True))
