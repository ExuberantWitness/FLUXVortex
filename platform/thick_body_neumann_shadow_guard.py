#!/usr/bin/env python3
"""Execute the implemented subset of the N3.1j3b6c canonical contract."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from claim_runtime.thick_body_neumann_shadow import (
    close_roboeagle_dual_surface_shell,
    closed_triangular_mesh,
    constant_source_polygon_influence,
    solve_conditioned_neumann_source,
)
from claim_runtime.viscous_shell_geometry import naca4_dual_surface_shell


PLATFORM = Path(__file__).resolve().parent
CASES = PLATFORM / "docs" / "diag" / "thick_body_neumann_shadow_cases.yaml"
OUTPUT = (
    PLATFORM
    / "docs"
    / "diag"
    / "thick_body_neumann_shadow_results.json"
)


def icosphere(subdivisions: int) -> tuple[np.ndarray, np.ndarray]:
    """Return a deterministic outward-oriented unit icosphere."""
    if not isinstance(subdivisions, int) or subdivisions < 0:
        raise ValueError("subdivisions must be a non-negative integer")
    golden = 0.5 * (1.0 + np.sqrt(5.0))
    vertices = np.array(
        [
            [-1, golden, 0],
            [1, golden, 0],
            [-1, -golden, 0],
            [1, -golden, 0],
            [0, -1, golden],
            [0, 1, golden],
            [0, -1, -golden],
            [0, 1, -golden],
            [golden, 0, -1],
            [golden, 0, 1],
            [-golden, 0, -1],
            [-golden, 0, 1],
        ],
        dtype=float,
    )
    vertices /= np.linalg.norm(vertices, axis=1)[:, None]
    faces = np.array(
        [
            [0, 11, 5],
            [0, 5, 1],
            [0, 1, 7],
            [0, 7, 10],
            [0, 10, 11],
            [1, 5, 9],
            [5, 11, 4],
            [11, 10, 2],
            [10, 7, 6],
            [7, 1, 8],
            [3, 9, 4],
            [3, 4, 2],
            [3, 2, 6],
            [3, 6, 8],
            [3, 8, 9],
            [4, 9, 5],
            [2, 4, 11],
            [6, 2, 10],
            [8, 6, 7],
            [9, 8, 1],
        ],
        dtype=int,
    )
    for _ in range(subdivisions):
        new_vertices = vertices.tolist()
        midpoint_cache: dict[tuple[int, int], int] = {}

        def midpoint(first: int, second: int) -> int:
            key = tuple(sorted((int(first), int(second))))
            if key not in midpoint_cache:
                point = 0.5 * (vertices[first] + vertices[second])
                point /= np.linalg.norm(point)
                midpoint_cache[key] = len(new_vertices)
                new_vertices.append(point.tolist())
            return midpoint_cache[key]

        new_faces: list[list[int]] = []
        for first, second, third in faces:
            edge_12 = midpoint(first, second)
            edge_23 = midpoint(second, third)
            edge_31 = midpoint(third, first)
            new_faces.extend(
                [
                    [first, edge_12, edge_31],
                    [second, edge_23, edge_12],
                    [third, edge_31, edge_23],
                    [edge_12, edge_23, edge_31],
                ]
            )
        vertices = np.asarray(new_vertices, dtype=float)
        faces = np.asarray(new_faces, dtype=int)
    return vertices, faces


def triangle_quadrature_oracle(
    vertices: np.ndarray,
    targets: np.ndarray,
    *,
    order: int = 48,
) -> tuple[np.ndarray, np.ndarray]:
    """Independent tensor Gauss/Duffy oracle for off-surface triangles."""
    nodes, weights = np.polynomial.legendre.leggauss(order)
    first = 0.5 * (nodes + 1.0)
    first_weight = 0.5 * weights
    second = first.copy()
    second_weight = first_weight.copy()
    edge_1 = vertices[1] - vertices[0]
    edge_2 = vertices[2] - vertices[0]
    jacobian = float(np.linalg.norm(np.cross(edge_1, edge_2)))
    potential = np.zeros(len(targets), dtype=float)
    velocity = np.zeros((len(targets), 3), dtype=float)
    for first_index, coordinate_1 in enumerate(first):
        for second_index, coordinate_2 in enumerate(second):
            bary_1 = coordinate_1
            bary_2 = (1.0 - coordinate_1) * coordinate_2
            point = (
                vertices[0] + bary_1 * edge_1 + bary_2 * edge_2
            )
            weight = (
                first_weight[first_index]
                * second_weight[second_index]
                * (1.0 - coordinate_1)
                * jacobian
            )
            displacement = targets - point
            distance = np.linalg.norm(displacement, axis=1)
            potential += weight / (4.0 * np.pi * distance)
            velocity -= (
                weight
                * displacement
                / (4.0 * np.pi * distance[:, None] ** 3)
            )
    return potential, velocity


def _relative_error(actual: np.ndarray, expected: np.ndarray) -> float:
    denominator = max(
        float(np.max(np.abs(expected), initial=0.0)),
        np.finfo(float).tiny,
    )
    return float(
        np.max(np.abs(actual - expected), initial=0.0) / denominator
    )


def execute_g1a(contract: dict[str, Any]) -> dict[str, Any]:
    thresholds = contract["stages"]["G1a_kernel"]["thresholds"]
    triangle = np.array(
        [[0.0, 0.0, 0.0], [1.1, 0.0, 0.0], [0.2, 0.8, 0.0]]
    )
    targets = np.array(
        [[0.31, 0.17, 0.73], [-0.21, 0.12, 0.41], [0.7, 0.4, -0.62]]
    )
    analytic = constant_source_polygon_influence(triangle, targets)
    oracle_potential, oracle_velocity = triangle_quadrature_oracle(
        triangle, targets
    )
    potential_error = _relative_error(
        analytic.potential, oracle_potential
    )
    velocity_error = _relative_error(
        analytic.velocity, oracle_velocity
    )
    centroid = np.mean(triangle, axis=0, keepdims=True)
    exterior = constant_source_polygon_influence(
        triangle, centroid, on_surface_side="exterior"
    )
    interior = constant_source_polygon_influence(
        triangle, centroid, on_surface_side="interior"
    )
    normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
    normal /= np.linalg.norm(normal)
    jump = float((interior.velocity[0] - exterior.velocity[0]) @ normal)
    jump_error = abs(jump - 1.0)
    checks = {
        "potential_oracle": (
            potential_error
            <= float(thresholds["max_potential_oracle_relative_error"])
        ),
        "velocity_oracle": (
            velocity_error
            <= float(thresholds["max_velocity_oracle_relative_error"])
        ),
        "normal_velocity_jump": (
            jump_error <= float(thresholds["max_normal_jump_error"])
        ),
    }
    return {
        "status": "passed" if all(checks.values()) else "failed",
        "metrics": {
            "max_potential_oracle_relative_error": potential_error,
            "max_velocity_oracle_relative_error": velocity_error,
            "normal_velocity_jump_error": jump_error,
        },
        "checks": checks,
    }


def _sphere_analytic_surface_velocity(
    incident: np.ndarray, normals: np.ndarray
) -> np.ndarray:
    return 1.5 * (
        incident
        - np.einsum("ij,ij->i", incident, normals)[:, None] * normals
    )


def execute_g1b(contract: dict[str, Any]) -> dict[str, Any]:
    stage = contract["stages"]["G1b_closed_body"]
    thresholds = stage["thresholds"]
    refinements: list[dict[str, Any]] = []
    velocity_errors: list[float] = []
    maximum_residual = 0.0
    maximum_flux = 0.0
    maximum_condition = 0.0
    finest_pressure_error = np.inf
    finest_force_norm = np.inf
    for subdivision in stage["refinements"]["subdivisions"]:
        vertices, faces = icosphere(int(subdivision))
        mesh = closed_triangular_mesh(vertices, faces)
        incident = np.tile([1.0, 0.0, 0.0], (len(faces), 1))
        solution = solve_conditioned_neumann_source(
            mesh, incident_velocity=incident
        )
        analytic_velocity = _sphere_analytic_surface_velocity(
            incident, mesh.normals
        )
        velocity_error = float(
            np.sqrt(
                np.mean(
                    np.sum(
                        (solution.total_velocity - analytic_velocity) ** 2,
                        axis=1,
                    )
                )
            )
            / 1.5
        )
        pressure = 1.0 - np.sum(solution.total_velocity**2, axis=1)
        analytic_pressure = 1.0 - np.sum(analytic_velocity**2, axis=1)
        pressure_error = float(
            np.sqrt(np.mean((pressure - analytic_pressure) ** 2))
        )
        force_coefficient = (
            -np.einsum(
                "i,ij,i->j", pressure, mesh.normals, mesh.areas
            )
            / np.pi
        )
        force_norm = float(np.linalg.norm(force_coefficient))
        velocity_errors.append(velocity_error)
        maximum_residual = max(
            maximum_residual, solution.relative_no_penetration_residual
        )
        maximum_flux = max(
            maximum_flux, solution.relative_source_flux
        )
        maximum_condition = max(
            maximum_condition, solution.condition_number
        )
        finest_pressure_error = pressure_error
        finest_force_norm = force_norm
        refinements.append(
            {
                "subdivision": int(subdivision),
                "panel_count": int(len(faces)),
                "relative_no_penetration": (
                    solution.relative_no_penetration_residual
                ),
                "relative_source_flux": solution.relative_source_flux,
                "condition_number": solution.condition_number,
                "surface_velocity_rms_relative": velocity_error,
                "pressure_rms_absolute": pressure_error,
                "force_coefficient": force_coefficient.tolist(),
            }
        )
    decreases = [
        1.0 - current / previous
        for previous, current in zip(velocity_errors, velocity_errors[1:])
    ]
    minimum_decrease = min(decreases, default=0.0)
    checks = {
        "watertight_manifold": all(
            item == 0
            for item in (
                mesh.boundary_edge_count,
                mesh.nonmanifold_edge_count,
                mesh.orientation_mismatch_count,
            )
        ),
        "collocation_no_penetration": (
            maximum_residual
            <= float(
                thresholds["max_collocation_no_penetration_relative"]
            )
        ),
        "source_flux_compatibility": (
            maximum_flux
            <= float(thresholds["max_source_flux_relative"])
        ),
        "surface_velocity": (
            velocity_errors[-1]
            <= float(thresholds["finest_surface_velocity_rms_relative"])
        ),
        "surface_pressure": (
            finest_pressure_error
            <= float(thresholds["finest_pressure_rms_absolute"])
        ),
        "dalembert_force": (
            finest_force_norm
            <= float(thresholds["finest_force_coefficient_norm"])
        ),
        "panel_refinement": (
            minimum_decrease
            >= float(
                thresholds[
                    "required_velocity_error_decrease_fraction"
                ]
            )
        ),
        "condition_number": (
            maximum_condition
            <= float(thresholds["max_condition_number"])
        ),
    }
    return {
        "status": "passed" if all(checks.values()) else "failed",
        "refinements": refinements,
        "metrics": {
            "max_relative_no_penetration": maximum_residual,
            "max_relative_source_flux": maximum_flux,
            "max_condition_number": maximum_condition,
            "finest_surface_velocity_rms_relative": velocity_errors[-1],
            "finest_pressure_rms_absolute": finest_pressure_error,
            "finest_force_coefficient_norm": finest_force_norm,
            "minimum_velocity_error_decrease_fraction": minimum_decrease,
        },
        "checks": checks,
    }


def _internal_dipole_velocity(
    points: np.ndarray, moment: np.ndarray
) -> np.ndarray:
    radius = np.linalg.norm(points, axis=1)
    projection = points @ moment
    return (
        moment[None, :] / radius[:, None] ** 3
        - 3.0 * projection[:, None] * points / radius[:, None] ** 5
    ) / (4.0 * np.pi)


def execute_g1c(contract: dict[str, Any]) -> dict[str, Any]:
    thresholds = contract["stages"]["G1c_conditioned_incident_state"][
        "thresholds"
    ]
    vertices, faces = icosphere(2)
    mesh = closed_triangular_mesh(vertices, faces)
    base_moment = np.array([0.7, -0.2, 0.4])
    incident = _internal_dipole_velocity(mesh.centroids, base_moment)
    incident_before = incident.copy()
    vertices_before = mesh.vertices.copy()
    faces_before = mesh.faces.copy()
    first = solve_conditioned_neumann_source(
        mesh, incident_velocity=incident
    )
    changed_incident = _internal_dipole_velocity(
        mesh.centroids, 1.2 * base_moment
    )
    second = solve_conditioned_neumann_source(
        mesh, incident_velocity=changed_incident
    )
    source_change = float(
        np.linalg.norm(second.source_strength - first.source_strength)
        / max(
            float(np.linalg.norm(first.source_strength)),
            np.finfo(float).tiny,
        )
    )
    potential_jump = 0.0
    sample_count = min(24, len(mesh.faces))
    for panel_index in range(sample_count):
        face = mesh.faces[panel_index]
        centroid = mesh.centroids[panel_index : panel_index + 1]
        exterior = constant_source_polygon_influence(
            mesh.vertices[face],
            centroid,
            strength=float(first.source_strength[panel_index]),
            on_surface_side="exterior",
        )
        interior = constant_source_polygon_influence(
            mesh.vertices[face],
            centroid,
            strength=float(first.source_strength[panel_index]),
            on_surface_side="interior",
        )
        potential_jump = max(
            potential_jump,
            float(
                np.max(
                    np.abs(exterior.potential - interior.potential),
                    initial=0.0,
                )
            ),
        )
    frozen_change = max(
        float(np.max(np.abs(incident - incident_before), initial=0.0)),
        float(
            np.max(
                np.abs(mesh.vertices - vertices_before), initial=0.0
            )
        ),
        float(np.max(np.abs(mesh.faces - faces_before), initial=0)),
    )
    checks = {
        "conditioned_no_penetration": (
            first.relative_no_penetration_residual
            <= float(thresholds["max_no_penetration_relative"])
        ),
        "source_potential_single_valued": (
            potential_jump
            <= float(thresholds["max_source_potential_jump"])
        ),
        "incident_change_requires_resolve": (
            source_change
            >= float(
                thresholds["minimum_source_solution_change_relative"]
            )
        ),
        "frozen_input_immutability": (
            frozen_change
            <= float(thresholds["max_frozen_input_change"])
        ),
    }
    return {
        "status": "passed" if all(checks.values()) else "failed",
        "metrics": {
            "relative_no_penetration": (
                first.relative_no_penetration_residual
            ),
            "source_potential_jump": potential_jump,
            "source_solution_change_relative": source_change,
            "frozen_input_change": frozen_change,
        },
        "checks": checks,
    }


def _pressure_force(
    bernoulli: np.ndarray,
    normals: np.ndarray,
    areas: np.ndarray,
    *,
    density: float,
) -> np.ndarray:
    """Integrate body force for ``p/rho = gauge - bernoulli``."""
    return density * np.einsum(
        "i,ij,i->j", bernoulli, normals, areas
    )


def execute_g1d(contract: dict[str, Any]) -> dict[str, Any]:
    thresholds = contract["stages"]["G1d_unsteady_objectivity"][
        "thresholds"
    ]
    vertices, faces = icosphere(2)
    base_mesh = closed_triangular_mesh(vertices, faces)

    # Constant translation in two Galilean frames.  The relative flow and
    # source field are identical, while the Bernoulli scalar may change only
    # by a spatially uniform gauge.
    body_velocity = np.array([0.37, -0.21, 0.14])
    frame_velocity = np.array([-0.46, 0.33, 0.18])
    wall_base = np.tile(body_velocity, (len(faces), 1))
    incident_base = np.zeros_like(wall_base)
    base = solve_conditioned_neumann_source(
        base_mesh,
        incident_velocity=incident_base,
        wall_velocity=wall_base,
    )
    bernoulli_base = (
        -np.einsum(
            "ij,ij->i", wall_base, base.total_velocity
        )
        + 0.5 * np.sum(base.total_velocity**2, axis=1)
    )
    force_base = _pressure_force(
        bernoulli_base,
        base_mesh.normals,
        base_mesh.areas,
        density=1.0,
    )

    wall_shifted = np.tile(
        body_velocity + frame_velocity, (len(faces), 1)
    )
    incident_shifted = np.tile(frame_velocity, (len(faces), 1))
    shifted = solve_conditioned_neumann_source(
        base_mesh,
        incident_velocity=incident_shifted,
        wall_velocity=wall_shifted,
    )
    farfield_material_rate = np.einsum(
        "ij,ij->i", incident_shifted, wall_shifted
    )
    bernoulli_shifted = (
        farfield_material_rate
        - np.einsum(
            "ij,ij->i", wall_shifted, shifted.total_velocity
        )
        + 0.5 * np.sum(shifted.total_velocity**2, axis=1)
    )
    force_shifted = _pressure_force(
        bernoulli_shifted,
        base_mesh.normals,
        base_mesh.areas,
        density=1.0,
    )
    force_scale = max(
        float(np.linalg.norm(force_base)),
        float(np.linalg.norm(force_shifted)),
        1.0,
    )
    galilean_force_change = float(
        np.linalg.norm(force_shifted - force_base) / force_scale
    )

    # At the central stage the translating sphere has zero speed and finite
    # acceleration.  Differentiating the source potential at the same
    # material wall points gives the inertial unsteady Bernoulli term without
    # a convective ambiguity.
    acceleration = np.array([0.8, -0.3, 0.5])
    time_step = 2.0e-3
    stage_potential: list[np.ndarray] = []
    for stage_time in (-time_step, 0.0, time_step):
        stage_position = 0.5 * acceleration * stage_time**2
        stage_velocity = acceleration * stage_time
        stage_mesh = closed_triangular_mesh(
            vertices + stage_position, faces
        )
        stage_solution = solve_conditioned_neumann_source(
            stage_mesh,
            incident_velocity=np.zeros((len(faces), 3)),
            wall_velocity=np.tile(stage_velocity, (len(faces), 1)),
        )
        stage_potential.append(stage_solution.source_potential)
    material_rate = (
        stage_potential[2] - stage_potential[0]
    ) / (2.0 * time_step)
    # Isolate the time-history error from the constant-panel spatial error:
    # the discrete Neumann operator is linear in wall velocity, so its exact
    # material-time derivative at W=0 is the source potential obtained with
    # unit-time velocity derivative A on the same frozen mesh.  The separate
    # added-mass comparison below retains the continuum spatial-accuracy gate.
    discrete_linear_rate = solve_conditioned_neumann_source(
        base_mesh,
        incident_velocity=np.zeros((len(faces), 3)),
        wall_velocity=np.tile(acceleration, (len(faces), 1)),
    ).source_potential
    material_rate_error = float(
        np.sqrt(
            np.mean(
                (material_rate - discrete_linear_rate) ** 2
            )
        )
        / max(
            float(
                np.sqrt(np.mean(discrete_linear_rate**2))
            ),
            np.finfo(float).tiny,
        )
    )
    unsteady_force = _pressure_force(
        material_rate,
        base_mesh.normals,
        base_mesh.areas,
        density=1.0,
    )
    # B = Dphi/Dt at zero instantaneous speed; the pressure force is
    # rho*integral(B n dS), equal to minus the classical added-mass load.
    displaced_mass = 4.0 * np.pi / 3.0
    analytic_added_mass_force = -0.5 * displaced_mass * acceleration
    added_mass_error = float(
        np.linalg.norm(
            unsteady_force - analytic_added_mass_force
        )
        / np.linalg.norm(analytic_added_mass_force)
    )
    checks = {
        "translating_sphere_galilean_invariance": (
            galilean_force_change
            <= float(
                thresholds[
                    "max_galilean_pressure_force_change_relative"
                ]
            )
        ),
        "accelerating_sphere_added_mass": (
            added_mass_error
            <= float(thresholds["max_added_mass_relative_error"])
        ),
        "three_stage_material_potential_history": (
            material_rate_error
            <= float(thresholds["max_material_rate_relative_error"])
        ),
    }
    return {
        "status": "passed" if all(checks.values()) else "failed",
        "metrics": {
            "galilean_pressure_force_change_relative": (
                galilean_force_change
            ),
            "material_rate_relative_error": material_rate_error,
            "added_mass_relative_error": added_mass_error,
            "computed_added_mass_force": unsteady_force.tolist(),
            "analytic_added_mass_force": (
                analytic_added_mass_force.tolist()
            ),
        },
        "checks": checks,
    }


def execute_g1e(contract: dict[str, Any]) -> dict[str, Any]:
    thresholds = contract["stages"]["G1e_roboeagle_geometry_adapter"][
        "thresholds"
    ]
    chord_fraction = 0.5 * (
        1.0 - np.cos(np.linspace(0.0, np.pi, 81))
    )
    span = np.linspace(0.0, 0.8, 9)
    chord = np.linspace(0.287, 0.12, len(span))
    shell = naca4_dual_surface_shell(
        chord_fraction,
        span,
        chord,
        closed_trailing_edge=False,
    )
    mean_before = shell.mean_surface.copy()
    closed = close_roboeagle_dual_surface_shell(shell)
    n1_change = float(
        np.max(
            np.abs(shell.mean_surface - mean_before), initial=0.0
        )
    )
    role_counts = {
        role: int(np.count_nonzero(closed.face_roles == role))
        for role in np.unique(closed.face_roles)
    }
    checks = {
        "upper_lower_material_pairing": (
            closed.maximum_material_pairing_error
            <= float(thresholds["max_material_pairing_error"])
        ),
        "watertight_boundary": (
            closed.mesh.boundary_edge_count
            <= int(thresholds["max_boundary_edges"])
        ),
        "manifold_boundary": (
            closed.mesh.nonmanifold_edge_count
            <= int(thresholds["max_nonmanifold_edges"])
            and closed.mesh.orientation_mismatch_count == 0
        ),
        "outward_positive_volume": closed.mesh.signed_volume > 0.0,
        "n1_mean_surface_unchanged": (
            n1_change <= float(thresholds["max_n1_geometry_change"])
            and closed.maximum_mean_surface_change
            <= float(thresholds["max_n1_geometry_change"])
        ),
        "explicit_boundary_roles": (
            set(role_counts)
            == {
                "upper",
                "lower",
                "root_cap",
                "tip_cap",
                "trailing_base",
            }
        ),
        "leading_edge_weld_only": (
            closed.leading_edge_weld_count == len(span)
            and closed.trailing_edge_weld_count == 0
        ),
    }
    return {
        "status": "passed" if all(checks.values()) else "failed",
        "metrics": {
            "vertex_count": int(len(closed.mesh.vertices)),
            "panel_count": int(len(closed.mesh.faces)),
            "signed_volume": closed.mesh.signed_volume,
            "boundary_edge_count": closed.mesh.boundary_edge_count,
            "nonmanifold_edge_count": (
                closed.mesh.nonmanifold_edge_count
            ),
            "orientation_mismatch_count": (
                closed.mesh.orientation_mismatch_count
            ),
            "maximum_material_pairing_error": (
                closed.maximum_material_pairing_error
            ),
            "maximum_n1_mean_surface_change": max(
                n1_change, closed.maximum_mean_surface_change
            ),
            "leading_edge_weld_count": closed.leading_edge_weld_count,
            "trailing_edge_weld_count": closed.trailing_edge_weld_count,
            "face_role_counts": role_counts,
        },
        "checks": checks,
    }


def evaluate_contract(contract: dict[str, Any]) -> dict[str, Any]:
    stages = {
        "G1a_kernel": execute_g1a(contract),
        "G1b_closed_body": execute_g1b(contract),
        "G1c_conditioned_incident_state": execute_g1c(contract),
        "G1d_unsteady_objectivity": execute_g1d(contract),
        "G1e_roboeagle_geometry_adapter": execute_g1e(contract),
    }
    executed = [
        stage
        for stage in stages.values()
        if stage["status"] != "not_executed"
    ]
    canonical_complete = all(
        stage["status"] == "passed" for stage in stages.values()
    )
    return {
        "version": 1,
        "claim": contract["claim"],
        "implementation_role": contract["implementation_role"],
        "stages": stages,
        "executed_stage_checks_pass": all(
            stage["status"] == "passed" for stage in executed
        ),
        "canonical_complete": canonical_complete,
        "model_comparison_executed": False,
        "production_activation_allowed": False,
        "promotion_gate": "NO-GO",
        "production_formula_changed": False,
        "frozen_n1_changed": False,
        "remaining_required_stages": [
            name
            for name, stage in stages.items()
            if stage["status"] != "passed"
        ],
    }


def main() -> int:
    contract = yaml.safe_load(CASES.read_text(encoding="utf-8"))
    result = evaluate_contract(contract)
    OUTPUT.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["executed_stage_checks_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
