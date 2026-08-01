"""Execute preregistered S0 actual-boundary potential equation gates."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from claim_runtime.actual_boundary_potential_shadow import (  # noqa: E402
    solve_actual_boundary_potential,
)
from claim_runtime.thick_body_neumann_shadow import (  # noqa: E402
    closed_triangular_mesh,
)
from thick_body_neumann_shadow_guard import icosphere  # noqa: E402


CASES = (
    HERE / "docs" / "diag"
    / "actual_boundary_circulation_representation_cases.yaml"
)
RESULTS = (
    HERE / "docs" / "diag"
    / "actual_boundary_potential_shadow_results.json"
)


def _relative_max(actual: np.ndarray, expected: np.ndarray) -> float:
    return float(
        np.max(np.abs(actual - expected), initial=0.0)
        / max(
            float(np.max(np.abs(expected), initial=0.0)),
            np.finfo(float).tiny,
        )
    )


def run() -> dict:
    contract = yaml.safe_load(CASES.read_text(encoding="utf-8"))
    thresholds = {
        name: value
        for name, value in contract[
            "S0_canonical_specification"
        ]["thresholds"].items()
    }
    offbody_points = np.array(
        [
            [2.0, 0.0, 0.0],
            [-2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 2.0],
            [np.sqrt(2.0), np.sqrt(2.0), 0.0],
        ]
    )
    offbody_expected = (
        0.5
        * offbody_points[:, 0]
        / np.linalg.norm(offbody_points, axis=1) ** 3
    )
    refinement = []
    for level in (0, 1, 2):
        vertices, faces = icosphere(level)
        mesh = closed_triangular_mesh(vertices, faces)
        incident = np.broadcast_to(
            np.array([1.0, 0.0, 0.0]), mesh.centroids.shape
        ).copy()
        solution = solve_actual_boundary_potential(
            mesh, incident_velocity=incident
        )
        evaluation = solution.evaluate(offbody_points)
        analytic_surface_velocity = 1.5 * (
            incident
            - np.einsum(
                "ij,ij->i", incident, mesh.normals
            )[:, None]
            * mesh.normals
        )
        computed_cp = 1.0 - np.einsum(
            "ij,ij->i", solution.total_velocity, solution.total_velocity
        )
        analytic_cp = 1.0 - np.einsum(
            "ij,ij->i",
            analytic_surface_velocity,
            analytic_surface_velocity,
        )
        matrix_jump = (
            solution.interior_doublet_matrix
            - solution.exterior_doublet_matrix
        )
        refinement.append(
            {
                "icosphere_level": level,
                "panel_count": len(mesh.faces),
                "condition_number": solution.condition_number,
                "relative_internal_potential_residual":
                    solution.relative_internal_potential_residual,
                "relative_exterior_surface_identity_residual":
                    solution.relative_exterior_surface_identity_residual,
                "relative_source_flux": solution.relative_source_flux,
                "offbody_potential_relative_error": _relative_max(
                    evaluation.perturbation_potential,
                    offbody_expected,
                ),
                "surface_normal_velocity_relative_residual":
                    solution.relative_normal_velocity_residual,
                "surface_Cp_rms_error": float(
                    np.sqrt(np.mean((computed_cp - analytic_cp) ** 2))
                ),
                "surface_velocity_rms_error": float(
                    np.sqrt(
                        np.mean(
                            np.einsum(
                                "ij,ij->i",
                                solution.total_velocity
                                - analytic_surface_velocity,
                                solution.total_velocity
                                - analytic_surface_velocity,
                            )
                        )
                    )
                ),
                "interior_exterior_jump_identity_max_error": float(
                    np.max(
                        np.abs(matrix_jump - np.eye(len(mesh.faces))),
                        initial=0.0,
                    )
                ),
            }
        )
    finest = refinement[-1]
    potential_errors = [
        item["offbody_potential_relative_error"]
        for item in refinement
    ]
    checks = {
        "internal_potential": (
            finest["relative_internal_potential_residual"]
            <= float(
                thresholds[
                    "internal_potential_relative_residual_max"
                ]
            )
        ),
        "prescribed_source_flux": (
            finest["relative_source_flux"]
            <= float(
                thresholds["prescribed_source_flux_relative_max"]
            )
        ),
        "exterior_surface_identity": (
            finest["relative_exterior_surface_identity_residual"]
            <= float(
                thresholds[
                    "exterior_surface_identity_relative_residual_max"
                ]
            )
        ),
        "offbody_potential_finest": (
            finest["offbody_potential_relative_error"]
            <= float(
                thresholds[
                    "finest_offbody_potential_relative_error_max"
                ]
            )
        ),
        "offbody_potential_monotone": all(
            later < earlier
            for earlier, later in zip(
                potential_errors[:-1], potential_errors[1:], strict=True
            )
        ),
        "surface_normal_velocity": (
            finest["surface_normal_velocity_relative_residual"]
            <= float(
                thresholds[
                    "finest_surface_normal_velocity_relative_residual_max"
                ]
            )
        ),
        "surface_Cp": (
            finest["surface_Cp_rms_error"]
            <= float(
                thresholds["finest_surface_Cp_rms_error_max"]
            )
        ),
        "condition_number": (
            max(item["condition_number"] for item in refinement)
            <= float(thresholds["matrix_condition_number_max"])
        ),
    }
    equation_checks = (
        "internal_potential",
        "prescribed_source_flux",
        "exterior_surface_identity",
        "offbody_potential_finest",
        "offbody_potential_monotone",
        "condition_number",
    )
    pressure_checks = ("surface_normal_velocity", "surface_Cp")
    result = {
        "artifact": "actual_boundary_constant_panel_equation_oracle",
        "claim_node": "N3.1j3b6d",
        "stage": "S0_low_order_equation_oracle",
        "refinement": refinement,
        "thresholds": thresholds,
        "checks": checks,
        "equation_oracle_gate": (
            "GO" if all(checks[name] for name in equation_checks)
            else "NO-GO"
        ),
        "surface_pressure_gate": (
            "GO" if all(checks[name] for name in pressure_checks)
            else "NO-GO"
        ),
        "stage_decision": (
            "GO" if all(checks.values()) else "NO-GO"
        ),
        "diagnosis": (
            "The same-boundary Morino potential equation and off-body field "
            "are qualified, but direct perimeter-ring velocity from "
            "piecewise-constant doublets is not a qualified surface-pressure "
            "operator. Continue to preregistered continuous P2/Galerkin "
            "actual-boundary representation; do not loosen thresholds."
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
