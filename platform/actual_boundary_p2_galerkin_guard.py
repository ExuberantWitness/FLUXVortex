"""Execute preregistered continuous-P2 actual-boundary Galerkin gates."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from claim_runtime.actual_boundary_p2_galerkin import (  # noqa: E402
    solve_actual_boundary_p2_galerkin,
)
from claim_runtime.thick_body_neumann_shadow import (  # noqa: E402
    closed_triangular_mesh,
)
from thick_body_neumann_shadow_guard import icosphere  # noqa: E402


CASES = (
    HERE / "docs" / "diag" / "actual_boundary_p2_galerkin_cases.yaml"
)
RESULTS = (
    HERE / "docs" / "diag" / "actual_boundary_p2_galerkin_results.json"
)


def _weighted_rms(values: np.ndarray, weights: np.ndarray) -> float:
    return float(
        np.sqrt(np.sum(weights * values**2) / np.sum(weights))
    )


def _case(
    level: int,
    order: int,
    offbody: np.ndarray,
    *,
    potential_operator: str = "tensor_duffy",
) -> tuple[dict, np.ndarray]:
    vertices, faces = icosphere(level)
    mesh = closed_triangular_mesh(vertices, faces)
    incident = np.broadcast_to(
        np.array([1.0, 0.0, 0.0]), mesh.centroids.shape
    ).copy()
    solution = solve_actual_boundary_p2_galerkin(
        mesh,
        incident_velocity=incident,
        target_quadrature_order=order,
        source_quadrature_order=order,
        potential_operator=potential_operator,
    )
    potential = solution.evaluate_potential(offbody)
    points = solution.quadrature_points
    radial_normal = points / np.linalg.norm(points, axis=1)[:, None]
    uniform = np.broadcast_to(
        np.array([1.0, 0.0, 0.0]), points.shape
    )
    analytic_velocity = 1.5 * (
        uniform
        - radial_normal[:, 0, None] * radial_normal
    )
    velocity_error = np.linalg.norm(
        solution.quadrature_total_velocity - analytic_velocity,
        axis=1,
    )
    computed_cp = 1.0 - np.einsum(
        "ij,ij->i",
        solution.quadrature_total_velocity,
        solution.quadrature_total_velocity,
    )
    analytic_cp = 1.0 - np.einsum(
        "ij,ij->i", analytic_velocity, analytic_velocity
    )
    pointwise_scale = max(
        float(
            np.sqrt(
                np.sum(
                    solution.quadrature_weights
                    * solution.quadrature_mu**2
                )
                / np.sum(solution.quadrature_weights)
            )
        ),
        np.finfo(float).tiny,
    )
    result = {
        "icosphere_level": level,
        "face_count": len(mesh.faces),
        "p2_dof_count": solution.topology.dof_count,
        "quadrature_order": order,
        "condition_number": solution.condition_number,
        "relative_weak_residual": solution.relative_weak_residual,
        "relative_source_flux": solution.relative_source_flux,
        "continuity_residual": solution.continuity_residual,
        "pointwise_interior_potential_rms_relative": (
            _weighted_rms(
                solution.quadrature_interior_potential_residual,
                solution.quadrature_weights,
            )
            / pointwise_scale
        ),
        "surface_velocity_rms_error": _weighted_rms(
            velocity_error, solution.quadrature_weights
        ),
        "surface_Cp_rms_error": _weighted_rms(
            computed_cp - analytic_cp,
            solution.quadrature_weights,
        ),
        "paired_quadrature_order": solution.paired_quadrature_order,
        "paired_topology_counts": solution.paired_topology_counts,
    }
    return result, potential


def run() -> dict:
    contract = yaml.safe_load(CASES.read_text(encoding="utf-8"))
    thresholds = {
        name: value for name, value in contract["thresholds"].items()
    }
    offbody = np.asarray(
        contract["canonical"]["offbody_points"], dtype=float
    )
    analytic_potential = (
        0.5
        * offbody[:, 0]
        / np.linalg.norm(offbody, axis=1) ** 3
    )
    potential_scale = max(
        float(np.max(np.abs(analytic_potential), initial=0.0)),
        np.finfo(float).tiny,
    )
    cauchy = []
    cauchy_potential = []
    for order in contract["quadrature"][
        "source_and_target_orders_for_level0_cauchy"
    ]:
        metrics, potential = _case(0, int(order), offbody)
        metrics["offbody_potential_relative_error"] = float(
            np.max(np.abs(potential - analytic_potential), initial=0.0)
            / potential_scale
        )
        cauchy.append(metrics)
        cauchy_potential.append(potential)
    production_order = int(
        contract["quadrature"]["production_oracle_order"]
    )
    coarse = cauchy[-1]
    fine, fine_potential = _case(1, production_order, offbody)
    fine["offbody_potential_relative_error"] = float(
        np.max(
            np.abs(fine_potential - analytic_potential), initial=0.0
        )
        / potential_scale
    )
    quadrature_change = float(
        np.max(
            np.abs(cauchy_potential[-1] - cauchy_potential[-2]),
            initial=0.0,
        )
        / potential_scale
    )
    checks = {
        "weak_residual": max(
            item["relative_weak_residual"] for item in cauchy + [fine]
        ) <= float(thresholds["weak_relative_residual_max"]),
        "continuity": max(
            item["continuity_residual"] for item in cauchy + [fine]
        ) <= float(thresholds["continuity_residual_max"]),
        "source_flux": max(
            item["relative_source_flux"] for item in cauchy + [fine]
        ) <= float(
            thresholds["prescribed_source_flux_relative_max"]
        ),
        "level0_quadrature_change": quadrature_change <= float(
            thresholds[
                "level0_quadrature_finest_offbody_change_max"
            ]
        ),
        "offbody_mesh_monotone": (
            fine["offbody_potential_relative_error"]
            < coarse["offbody_potential_relative_error"]
        ),
        "offbody_finest": (
            fine["offbody_potential_relative_error"]
            <= float(
                thresholds[
                    "finest_offbody_potential_relative_error_max"
                ]
            )
        ),
        "surface_velocity": (
            fine["surface_velocity_rms_error"]
            <= float(
                thresholds[
                    "finest_surface_velocity_rms_error_max"
                ]
            )
        ),
        "surface_Cp": (
            fine["surface_Cp_rms_error"]
            <= float(
                thresholds["finest_surface_Cp_rms_error_max"]
            )
        ),
        "condition_number": max(
            item["condition_number"] for item in cauchy + [fine]
        ) <= float(thresholds["matrix_condition_number_max"]),
    }
    result = {
        "artifact": "actual_boundary_continuous_p2_galerkin",
        "claim_node": "N3.1j3b6d3",
        "stage": "S1_continuous_actual_boundary",
        "level0_quadrature_cauchy": cauchy,
        "level0_quadrature_finest_offbody_change":
            quadrature_change,
        "mesh_refinement": [coarse, fine],
        "thresholds": thresholds,
        "checks": checks,
        "stage_decision": (
            "GO" if all(checks.values()) else "NO-GO"
        ),
        "production_activation_allowed": False,
        "interpretation": (
            "Continuous topology, weak equation and prescribed-source "
            "conservation are separated from exterior-field and surface-"
            "pressure accuracy. Any failed accuracy gate requires a "
            "geometry/weak-operator diagnosis; no threshold is loosened."
        ),
    }
    RESULTS.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, sort_keys=True))
