"""Low-order actual-boundary source/doublet equation oracle.

This module is stage S0 of claim ``N3.1j3b6d``.  It verifies the Morino
interior-Dirichlet representation on a validated closed triangular body:

* the source strength is prescribed by the moving-wall normal condition,
  ``sigma = (u_incident-u_wall).n``;
* the doublet strength is solved on the *same actual boundary* from
  ``D_inside mu + S sigma = 0``;
* exterior velocity is evaluated from the source panels and the oriented
  perimeter vortices exactly equivalent to each constant doublet panel.

Piecewise-constant doublets are intentionally only a low-order equation
oracle.  They do not provide the continuous P2 state, circulation/Kutta
closure, wake, material pressure history, force, or production activation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .thick_body_neumann_shadow import (
    ClosedTriangularMesh,
    ThickBodyNeumannError,
    constant_source_polygon_influence,
)


@dataclass(frozen=True)
class ActualBoundaryPotentialEvaluation:
    points: np.ndarray
    source_potential: np.ndarray
    doublet_potential: np.ndarray
    perturbation_potential: np.ndarray
    source_velocity: np.ndarray
    doublet_velocity: np.ndarray
    perturbation_velocity: np.ndarray


@dataclass(frozen=True)
class ActualBoundaryPotentialSolution:
    mesh: ClosedTriangularMesh
    incident_velocity: np.ndarray
    wall_velocity: np.ndarray
    source_strength: np.ndarray
    doublet_strength: np.ndarray
    source_potential_matrix: np.ndarray
    interior_doublet_matrix: np.ndarray
    exterior_doublet_matrix: np.ndarray
    interior_potential: np.ndarray
    exterior_perturbation_potential: np.ndarray
    source_velocity: np.ndarray
    doublet_velocity: np.ndarray
    perturbation_velocity: np.ndarray
    total_velocity: np.ndarray
    source_flux: float
    relative_source_flux: float
    maximum_internal_potential_residual: float
    relative_internal_potential_residual: float
    maximum_exterior_surface_identity_residual: float
    relative_exterior_surface_identity_residual: float
    maximum_normal_velocity_residual: float
    relative_normal_velocity_residual: float
    condition_number: float

    def evaluate(self, points: Any) -> ActualBoundaryPotentialEvaluation:
        return evaluate_actual_boundary_potential(self, points)


def _finite_points(name: str, value: Any) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if (
        array.ndim != 2
        or array.shape[1] != 3
        or not np.all(np.isfinite(array))
    ):
        raise ThickBodyNeumannError(
            f"{name} must be a finite array with shape (n,3)"
        )
    return array


def actual_boundary_potential_matrices(
    mesh: ClosedTriangularMesh,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return source, inside-doublet and outside-doublet potential matrices."""
    if not isinstance(mesh, ClosedTriangularMesh):
        raise ThickBodyNeumannError(
            "mesh must be validated by closed_triangular_mesh"
        )
    panel_count = len(mesh.faces)
    source = np.empty((panel_count, panel_count), dtype=float)
    inside = np.empty_like(source)
    outside = np.empty_like(source)
    for source_index, face in enumerate(mesh.faces):
        polygon = mesh.vertices[face]
        interior_influence = constant_source_polygon_influence(
            polygon,
            mesh.centroids,
            on_surface_side="interior",
        )
        exterior_influence = constant_source_polygon_influence(
            polygon,
            mesh.centroids,
            on_surface_side="exterior",
        )
        source[:, source_index] = interior_influence.potential
        panel_normal = mesh.normals[source_index]
        inside[:, source_index] = (
            interior_influence.velocity @ panel_normal
        )
        outside[:, source_index] = (
            exterior_influence.velocity @ panel_normal
        )
    return source, inside, outside


def _segment_velocity(
    points: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
    strength: float,
) -> np.ndarray:
    r1 = points - start
    r2 = points - end
    filament = end - start
    cross = np.cross(r1, r2)
    denominator = np.einsum("ij,ij->i", cross, cross)
    norm1 = np.linalg.norm(r1, axis=1)
    norm2 = np.linalg.norm(r2, axis=1)
    scale = max(float(np.linalg.norm(filament)), 1.0)
    floor = 128.0 * np.finfo(float).eps * scale
    if (
        np.any(norm1 <= floor)
        or np.any(norm2 <= floor)
        or np.any(denominator <= floor**4)
    ):
        raise ThickBodyNeumannError(
            "doublet field target lies on an active perimeter filament"
        )
    direction = r1 / norm1[:, None] - r2 / norm2[:, None]
    coefficient = direction @ filament
    return (
        float(strength)
        * cross
        * coefficient[:, None]
        / (4.0 * np.pi * denominator[:, None])
    )


def _doublet_ring_velocity(
    mesh: ClosedTriangularMesh,
    strength: np.ndarray,
    points: np.ndarray,
) -> np.ndarray:
    velocity = np.zeros_like(points)
    for face, value in zip(mesh.faces, strength, strict=True):
        polygon = mesh.vertices[face]
        for first, second in ((0, 1), (1, 2), (2, 0)):
            velocity += _segment_velocity(
                points,
                polygon[first],
                polygon[second],
                float(value),
            )
    return velocity


def solve_actual_boundary_potential(
    mesh: ClosedTriangularMesh,
    *,
    incident_velocity: Any,
    wall_velocity: Any | None = None,
) -> ActualBoundaryPotentialSolution:
    """Solve the S0 constant-panel interior-Dirichlet equation."""
    if not isinstance(mesh, ClosedTriangularMesh):
        raise ThickBodyNeumannError(
            "mesh must be validated by closed_triangular_mesh"
        )
    incident = _finite_points("incident_velocity", incident_velocity)
    expected = mesh.centroids.shape
    if incident.shape != expected:
        raise ThickBodyNeumannError(
            f"incident_velocity must have shape {expected}"
        )
    if wall_velocity is None:
        wall = np.zeros_like(incident)
    else:
        wall = _finite_points("wall_velocity", wall_velocity)
        if wall.shape != expected:
            raise ThickBodyNeumannError(
                f"wall_velocity must have shape {expected}"
            )
    source_strength = np.einsum(
        "ij,ij->i", incident - wall, mesh.normals
    )
    source_matrix, inside_matrix, outside_matrix = (
        actual_boundary_potential_matrices(mesh)
    )
    condition_number = float(np.linalg.cond(inside_matrix))
    if not np.isfinite(condition_number):
        raise ThickBodyNeumannError(
            "interior doublet matrix is singular or non-finite"
        )
    source_potential = source_matrix @ source_strength
    try:
        doublet_strength = np.linalg.solve(
            inside_matrix, -source_potential
        )
    except np.linalg.LinAlgError as error:
        raise ThickBodyNeumannError(
            "actual-boundary doublet solve failed"
        ) from error
    interior_potential = (
        source_potential + inside_matrix @ doublet_strength
    )
    exterior_potential = (
        source_potential + outside_matrix @ doublet_strength
    )
    exterior_identity = exterior_potential + doublet_strength
    potential_scale = max(
        float(np.max(np.abs(source_potential), initial=0.0)),
        float(np.max(np.abs(doublet_strength), initial=0.0)),
        np.finfo(float).tiny,
    )

    source_velocity = np.zeros_like(incident)
    for source_index, face in enumerate(mesh.faces):
        source_velocity += constant_source_polygon_influence(
            mesh.vertices[face],
            mesh.centroids,
            strength=float(source_strength[source_index]),
            on_surface_side="exterior",
        ).velocity
    doublet_velocity = _doublet_ring_velocity(
        mesh, doublet_strength, mesh.centroids
    )
    perturbation_velocity = source_velocity + doublet_velocity
    total_velocity = incident + perturbation_velocity
    normal_residual = np.einsum(
        "ij,ij->i", total_velocity - wall, mesh.normals
    )
    velocity_scale = max(
        float(
            np.max(
                np.linalg.norm(incident - wall, axis=1),
                initial=0.0,
            )
        ),
        np.finfo(float).tiny,
    )
    source_flux = float(np.dot(source_strength, mesh.areas))
    source_flux_scale = max(
        float(np.dot(np.abs(source_strength), mesh.areas)),
        np.finfo(float).tiny,
    )
    return ActualBoundaryPotentialSolution(
        mesh=mesh,
        incident_velocity=incident.copy(),
        wall_velocity=wall.copy(),
        source_strength=source_strength,
        doublet_strength=doublet_strength,
        source_potential_matrix=source_matrix,
        interior_doublet_matrix=inside_matrix,
        exterior_doublet_matrix=outside_matrix,
        interior_potential=interior_potential,
        exterior_perturbation_potential=exterior_potential,
        source_velocity=source_velocity,
        doublet_velocity=doublet_velocity,
        perturbation_velocity=perturbation_velocity,
        total_velocity=total_velocity,
        source_flux=source_flux,
        relative_source_flux=abs(source_flux) / source_flux_scale,
        maximum_internal_potential_residual=float(
            np.max(np.abs(interior_potential), initial=0.0)
        ),
        relative_internal_potential_residual=float(
            np.max(np.abs(interior_potential), initial=0.0)
            / potential_scale
        ),
        maximum_exterior_surface_identity_residual=float(
            np.max(np.abs(exterior_identity), initial=0.0)
        ),
        relative_exterior_surface_identity_residual=float(
            np.max(np.abs(exterior_identity), initial=0.0)
            / potential_scale
        ),
        maximum_normal_velocity_residual=float(
            np.max(np.abs(normal_residual), initial=0.0)
        ),
        relative_normal_velocity_residual=float(
            np.max(np.abs(normal_residual), initial=0.0)
            / velocity_scale
        ),
        condition_number=condition_number,
    )


def evaluate_actual_boundary_potential(
    solution: ActualBoundaryPotentialSolution,
    points: Any,
) -> ActualBoundaryPotentialEvaluation:
    """Evaluate the solved perturbation state away from active filaments."""
    if not isinstance(solution, ActualBoundaryPotentialSolution):
        raise ThickBodyNeumannError(
            "solution must be an ActualBoundaryPotentialSolution"
        )
    targets = _finite_points("points", points)
    source_potential = np.zeros(len(targets), dtype=float)
    source_velocity = np.zeros_like(targets)
    doublet_potential = np.zeros(len(targets), dtype=float)
    for index, face in enumerate(solution.mesh.faces):
        influence = constant_source_polygon_influence(
            solution.mesh.vertices[face],
            targets,
            strength=float(solution.source_strength[index]),
            on_surface_side="principal",
        )
        source_potential += influence.potential
        source_velocity += influence.velocity
        unit_source = constant_source_polygon_influence(
            solution.mesh.vertices[face],
            targets,
            on_surface_side="principal",
        )
        doublet_potential += (
            float(solution.doublet_strength[index])
            * (unit_source.velocity @ solution.mesh.normals[index])
        )
    doublet_velocity = _doublet_ring_velocity(
        solution.mesh, solution.doublet_strength, targets
    )
    return ActualBoundaryPotentialEvaluation(
        points=targets.copy(),
        source_potential=source_potential,
        doublet_potential=doublet_potential,
        perturbation_potential=source_potential + doublet_potential,
        source_velocity=source_velocity,
        doublet_velocity=doublet_velocity,
        perturbation_velocity=source_velocity + doublet_velocity,
    )
