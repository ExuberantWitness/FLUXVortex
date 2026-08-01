"""Run preregistered S1f qualified level-3 continuation."""
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
S1E_RESULTS = (
    HERE / "docs" / "diag"
    / "actual_boundary_p2_paired_refinement_results.json"
)
RESULTS = (
    HERE / "docs" / "diag"
    / "actual_boundary_p2_paired_level3_results.json"
)


def _with_potential_error(
    level: int,
    order: int,
    offbody: np.ndarray,
    analytic: np.ndarray,
    scale: float,
) -> dict:
    metrics, potential = _case(
        level,
        order,
        offbody,
        potential_operator="paired_singular",
    )
    metrics["offbody_potential_relative_error"] = float(
        np.max(np.abs(potential - analytic), initial=0.0) / scale
    )
    return metrics


def run() -> dict:
    contract = yaml.safe_load(CASES.read_text(encoding="utf-8"))
    specification = contract[
        "S1f_paired_level3_continuation_preregistered_after_S1e_before_execution"
    ]
    previous = json.loads(S1E_RESULTS.read_text(encoding="utf-8"))
    level2_order8 = previous["level2"]
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
    order = int(
        specification["formal"]["target_source_and_pair_order"]
    )
    level2_order6 = _with_potential_error(
        2, order, offbody, analytic, scale
    )
    error_names = (
        "offbody_potential_relative_error",
        "surface_velocity_rms_error",
        "surface_Cp_rms_error",
    )
    order_changes = {
        name: abs(level2_order6[name] - level2_order8[name])
        for name in error_names
    }
    order_checks = {
        name: value <= 0.002 for name, value in order_changes.items()
    }
    result = {
        "artifact": "actual_boundary_p2_paired_level3_continuation",
        "claim_node": "N3.1j3b6d3",
        "stage": "S1f_paired_level3_continuation",
        "level2_order8": level2_order8,
        "level2_order6": level2_order6,
        "level2_order_absolute_changes": order_changes,
        "order_qualification_checks": order_checks,
        "level3": None,
        "checks": dict(order_checks),
        "stage_decision": "NO-GO",
        "production_activation_allowed": False,
    }
    if not all(order_checks.values()):
        result["interpretation"] = (
            "The preregistered order qualification failed, so level 3 was "
            "not executed and no mesh-resolution claim is admissible."
        )
        RESULTS.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return result

    level = int(specification["formal"]["mesh_level"])
    level3 = _with_potential_error(
        level, order, offbody, analytic, scale
    )
    thresholds = specification["unchanged_thresholds"]
    expected_counts = {
        "common_triangle": level3["face_count"],
        "common_edge": 3 * level3["face_count"],
    }
    topology = level3["paired_topology_counts"]
    final_checks = {
        "potential_accuracy": (
            level3["offbody_potential_relative_error"]
            <= float(
                thresholds["offbody_potential_relative_error_max"]
            )
        ),
        "velocity_accuracy": (
            level3["surface_velocity_rms_error"]
            <= float(thresholds["surface_velocity_rms_error_max"])
        ),
        "Cp_accuracy": (
            level3["surface_Cp_rms_error"]
            <= float(thresholds["surface_Cp_rms_error_max"])
        ),
        "weak_residual": level3["relative_weak_residual"]
        <= float(contract["thresholds"]["weak_relative_residual_max"]),
        "continuity": level3["continuity_residual"]
        <= float(contract["thresholds"]["continuity_residual_max"]),
        "source_flux": level3["relative_source_flux"]
        <= float(
            contract["thresholds"][
                "prescribed_source_flux_relative_max"
            ]
        ),
        "condition_number": level3["condition_number"]
        <= float(contract["thresholds"]["matrix_condition_number_max"]),
        "topology_common_triangle": (
            topology["common_triangle"]
            == expected_counts["common_triangle"]
        ),
        "topology_common_edge": (
            topology["common_edge"]
            == expected_counts["common_edge"]
        ),
        "topology_common_vertex_nonzero": (
            topology["common_vertex"] > 0
        ),
    }
    checks = {**order_checks, **final_checks}
    result.update({
        "level3": level3,
        "unchanged_thresholds": thresholds,
        "checks": checks,
        "stage_decision": "GO" if all(checks.values()) else "NO-GO",
        "interpretation": (
            "Passing qualifies only the attached unit-sphere continuous-P2 "
            "weak potential and surface-gradient oracle at this demonstrated "
            "flat-mesh resolution. It does not qualify circulation, wake, "
            "unsteady pressure or production force."
        ),
    })
    RESULTS.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, sort_keys=True))
