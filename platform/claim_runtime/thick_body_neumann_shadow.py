"""Actual-thickness constant-source Neumann shadow.

This module is a diagnostic boundary-element operator for claim
``N3.1j3b6c``.  It consumes a complete incident velocity on a closed,
outward-oriented body and solves only the single-valued source potential
needed to cancel the remaining wall-normal velocity.  It does not solve or
modify circulation, a wake, N1 forces, viscous pressure, or a Kutta
condition.

The constant-source polygon influence follows the Hess--Smith formulation
documented in NASA TP-2995.  The Green-function convention used here is

    phi_sigma(x) = integral_S sigma(y)/(4*pi*|x-y|) dS_y,

so the exterior normal-velocity jump of one flat panel is ``-sigma/2``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .viscous_shell_geometry import DualSurfaceShell


class ThickBodyNeumannError(ValueError):
    """Invalid panel, closed body, boundary condition, or source solution."""


def _finite(name: str, value: Any, *, ndim: int | None = None) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if ndim is not None and array.ndim != ndim:
        raise ThickBodyNeumannError(
            f"{name} must have ndim={ndim}, got {array.shape}"
        )
    if not np.all(np.isfinite(array)):
        raise ThickBodyNeumannError(f"{name} contains non-finite values")
    return array


@dataclass(frozen=True)
class SourcePanelInfluence:
    potential: np.ndarray
    velocity: np.ndarray


@dataclass(frozen=True)
class ClosedTriangularMesh:
    vertices: np.ndarray
    faces: np.ndarray
    centroids: np.ndarray
    normals: np.ndarray
    areas: np.ndarray
    signed_volume: float
    boundary_edge_count: int
    nonmanifold_edge_count: int
    orientation_mismatch_count: int


@dataclass(frozen=True)
class NeumannSourceSolution:
    mesh: ClosedTriangularMesh
    incident_velocity: np.ndarray
    wall_velocity: np.ndarray
    source_strength: np.ndarray
    source_potential: np.ndarray
    source_velocity: np.ndarray
    total_velocity: np.ndarray
    influence_matrix: np.ndarray
    right_hand_side: np.ndarray
    max_no_penetration_residual: float
    relative_no_penetration_residual: float
    source_flux: float
    relative_source_flux: float
    condition_number: float


@dataclass(frozen=True)
class RoboEagleClosedShell:
    mesh: ClosedTriangularMesh
    upper_vertex_indices: np.ndarray
    lower_vertex_indices: np.ndarray
    face_roles: np.ndarray
    maximum_material_pairing_error: float
    maximum_mean_surface_change: float
    leading_edge_weld_count: int
    trailing_edge_weld_count: int


def _polygon_frame(vertices: np.ndarray) -> tuple[np.ndarray, ...]:
    edge = vertices[1] - vertices[0]
    edge_norm = float(np.linalg.norm(edge))
    raw_normal = np.cross(edge, vertices[2] - vertices[0])
    normal_norm = float(np.linalg.norm(raw_normal))
    scale = float(
        np.max(
            np.linalg.norm(vertices - vertices[0], axis=1),
            initial=0.0,
        )
    )
    if edge_norm <= 0.0 or normal_norm <= 0.0 or scale <= 0.0:
        raise ThickBodyNeumannError("source polygon is degenerate")
    tangent = edge / edge_norm
    normal = raw_normal / normal_norm
    # Hess--Smith uses a left-handed local frame.  This makes an outward-CCW
    # global polygon clockwise in local (xi,eta), matching the analytic
    # edge formula without silently reversing the physical normal.
    oblique = np.cross(tangent, normal)
    local = np.column_stack(
        (
            (vertices - vertices[0]) @ tangent,
            (vertices - vertices[0]) @ oblique,
            (vertices - vertices[0]) @ normal,
        )
    )
    planarity = float(np.max(np.abs(local[:, 2]), initial=0.0))
    if planarity > 2.0e-12 * scale:
        raise ThickBodyNeumannError(
            f"source polygon is not planar: residual={planarity}"
        )
    return tangent, oblique, normal, local[:, :2], scale


def constant_source_polygon_influence(
    vertices: Any,
    targets: Any,
    *,
    strength: float = 1.0,
    on_surface_side: str = "exterior",
) -> SourcePanelInfluence:
    """Evaluate one constant-source polygon at one or more targets.

    ``vertices`` must be ordered counterclockwise when viewed from the
    physical outward normal.  Coplanar targets inside the polygon use the
    explicit exterior or interior limiting normal velocity; no collocation
    offset is fitted or inferred.
    """
    polygon = _finite("vertices", vertices, ndim=2)
    points = _finite("targets", targets, ndim=2)
    if (
        polygon.shape[1] != 3
        or len(polygon) < 3
        or points.shape[1] != 3
        or not np.isfinite(strength)
        or on_surface_side not in {"exterior", "interior", "principal"}
    ):
        raise ThickBodyNeumannError(
            "invalid polygon, target, strength, or on-surface side"
        )
    tangent, oblique, normal, local_vertices, scale = _polygon_frame(
        polygon
    )
    relative = points - polygon[0]
    x = relative @ tangent
    y = relative @ oblique
    z = relative @ normal
    absolute_z = np.abs(z)
    count = len(points)
    documented_potential = np.zeros(count, dtype=float)
    documented_velocity = np.zeros((count, 3), dtype=float)
    inside = np.ones(count, dtype=bool)
    zero_R_count = np.zeros(count, dtype=int)
    numerical_floor = 64.0 * np.finfo(float).eps * max(scale, 1.0)

    for edge_index in range(len(local_vertices)):
        next_index = (edge_index + 1) % len(local_vertices)
        xi, yi = local_vertices[edge_index]
        xj, yj = local_vertices[next_index]
        edge_length = float(np.hypot(xj - xi, yj - yi))
        if edge_length <= numerical_floor:
            raise ThickBodyNeumannError("source polygon has a zero edge")
        ri = np.sqrt((x - xi) ** 2 + (y - yi) ** 2 + z**2)
        rj = np.sqrt((x - xj) ** 2 + (y - yj) ** 2 + z**2)
        numerator = ri + rj + edge_length
        denominator = ri + rj - edge_length
        if np.any(denominator < -numerical_floor):
            raise ThickBodyNeumannError(
                "invalid Hess-Smith logarithm geometry"
            )
        denominator = np.maximum(denominator, numerical_floor)
        edge_log = np.log(numerator / denominator)
        sine = (yj - yi) / edge_length
        cosine = (xj - xi) / edge_length
        si = (xi - x) * cosine + (yi - y) * sine
        sj = (xj - x) * cosine + (yj - y) * sine
        perpendicular = (
            (x - xi) * sine - (y - yi) * cosine
        )
        angle = np.arctan2(
            perpendicular
            * absolute_z
            * (ri * sj - rj * si),
            ri * rj * perpendicular**2 + z**2 * sj * si,
        )
        documented_velocity[:, 0] -= sine * edge_log
        documented_velocity[:, 1] += cosine * edge_log
        documented_velocity[:, 2] -= angle
        documented_potential -= (
            perpendicular * edge_log + absolute_z * angle
        )
        inside &= perpendicular >= -numerical_floor
        zero_R_count += np.abs(perpendicular) <= numerical_floor

    solid_angle = np.where(inside, 2.0 * np.pi, 0.0)
    documented_velocity[:, 2] += solid_angle
    documented_velocity[:, 2] *= np.sign(z)
    documented_velocity[zero_R_count > 1, 2] = 0.0
    documented_potential += absolute_z * solid_angle
    documented_velocity /= 4.0 * np.pi
    documented_potential /= 4.0 * np.pi

    # The documented local formula above evaluates
    # -integral(1/(4*pi*r)); negate it for this module's Green convention.
    local_velocity = -documented_velocity
    potential = -documented_potential
    coplanar_inside = inside & (
        absolute_z <= 2.0e-13 * max(scale, 1.0)
    )
    if on_surface_side == "exterior":
        local_velocity[coplanar_inside, 2] = -0.5
    elif on_surface_side == "interior":
        local_velocity[coplanar_inside, 2] = 0.5
    else:
        local_velocity[coplanar_inside, 2] = 0.0

    velocity = (
        local_velocity[:, 0, None] * tangent
        + local_velocity[:, 1, None] * oblique
        + local_velocity[:, 2, None] * normal
    )
    return SourcePanelInfluence(
        potential=float(strength) * potential,
        velocity=float(strength) * velocity,
    )


def closed_triangular_mesh(vertices: Any, faces: Any) -> ClosedTriangularMesh:
    """Validate and freeze one outward-oriented watertight triangle mesh."""
    points = _finite("vertices", vertices, ndim=2)
    topology = np.asarray(faces)
    if (
        points.shape[1] != 3
        or topology.ndim != 2
        or topology.shape[1] != 3
        or not np.issubdtype(topology.dtype, np.integer)
        or len(points) < 4
        or len(topology) < 4
    ):
        raise ThickBodyNeumannError(
            "vertices/faces must describe a triangular closed body"
        )
    topology = topology.astype(int, copy=True)
    if np.any(topology < 0) or np.any(topology >= len(points)):
        raise ThickBodyNeumannError("face index is outside the vertex array")
    triangles = points[topology]
    raw_normals = np.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
    )
    double_area = np.linalg.norm(raw_normals, axis=1)
    scale = float(np.ptp(points, axis=0).max(initial=0.0))
    if scale <= 0.0 or np.any(double_area <= 1.0e-14 * scale**2):
        raise ThickBodyNeumannError("mesh contains a degenerate triangle")
    normals = raw_normals / double_area[:, None]
    areas = 0.5 * double_area
    centroids = np.mean(triangles, axis=1)
    signed_volume = float(
        np.sum(
            np.einsum(
                "ij,ij->i",
                triangles[:, 0],
                np.cross(triangles[:, 1], triangles[:, 2]),
            )
        )
        / 6.0
    )

    edge_occurrences: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for face in topology:
        for start, end in (
            (int(face[0]), int(face[1])),
            (int(face[1]), int(face[2])),
            (int(face[2]), int(face[0])),
        ):
            key = (min(start, end), max(start, end))
            edge_occurrences.setdefault(key, []).append((start, end))
    boundary = sum(
        len(occurrences) == 1
        for occurrences in edge_occurrences.values()
    )
    nonmanifold = sum(
        len(occurrences) != 2
        for occurrences in edge_occurrences.values()
        if len(occurrences) != 1
    )
    orientation_mismatch = sum(
        len(occurrences) == 2
        and occurrences[0] == occurrences[1]
        for occurrences in edge_occurrences.values()
    )
    if (
        boundary != 0
        or nonmanifold != 0
        or orientation_mismatch != 0
        or signed_volume <= 0.0
    ):
        raise ThickBodyNeumannError(
            "mesh must be watertight, manifold, consistently outward, "
            f"and positive-volume; boundary={boundary}, "
            f"nonmanifold={nonmanifold}, orientation={orientation_mismatch}, "
            f"volume={signed_volume}"
        )
    return ClosedTriangularMesh(
        vertices=points.copy(),
        faces=topology,
        centroids=centroids,
        normals=normals,
        areas=areas,
        signed_volume=signed_volume,
        boundary_edge_count=boundary,
        nonmanifold_edge_count=nonmanifold,
        orientation_mismatch_count=orientation_mismatch,
    )


def _append_triangle_if_nondegenerate(
    faces: list[list[int]],
    roles: list[str],
    vertices: list[list[float]],
    triangle: tuple[int, int, int],
    role: str,
    *,
    area_floor: float,
) -> None:
    if len(set(triangle)) < 3:
        return
    points = np.asarray(
        [vertices[index] for index in triangle], dtype=float
    )
    double_area = float(
        np.linalg.norm(
            np.cross(points[1] - points[0], points[2] - points[0])
        )
    )
    if double_area <= area_floor:
        return
    faces.append(list(triangle))
    roles.append(role)


def close_roboeagle_dual_surface_shell(
    shell: DualSurfaceShell,
) -> RoboEagleClosedShell:
    """Triangulate and cap one semi-wing dual surface without moving N1.

    Upper and lower material points are welded only when their frozen
    coordinates already coincide (normally the NACA leading edge).  The
    remaining boundaries are closed by explicit root, tip, and finite
    trailing-base panels.  No aerodynamic target, panel offset, or fitted
    geometry tolerance is used.
    """
    if not isinstance(shell, DualSurfaceShell):
        raise ThickBodyNeumannError(
            "shell must be a validated DualSurfaceShell"
        )
    upper = _finite("upper_surface", shell.upper_surface, ndim=3)
    lower = _finite("lower_surface", shell.lower_surface, ndim=3)
    mean = _finite("mean_surface", shell.mean_surface, ndim=3)
    if (
        upper.shape != lower.shape
        or upper.shape != mean.shape
        or upper.shape[2] != 3
        or upper.shape[0] < 2
        or upper.shape[1] < 2
    ):
        raise ThickBodyNeumannError(
            "dual shell surfaces must share shape (nxi,neta,3)"
        )
    n_chord, n_span, _ = upper.shape
    geometry_before = mean.copy()
    reconstructed_mean = 0.5 * (upper + lower)
    pairing_error = float(
        np.max(
            np.abs(reconstructed_mean - mean), initial=0.0
        )
    )
    scale = float(
        np.ptp(
            np.concatenate(
                (upper.reshape(-1, 3), lower.reshape(-1, 3)), axis=0
            ),
            axis=0,
        ).max(initial=0.0)
    )
    if scale <= 0.0:
        raise ThickBodyNeumannError("dual shell has zero geometric scale")
    weld_tolerance = 128.0 * np.finfo(float).eps * max(scale, 1.0)
    area_floor = 128.0 * np.finfo(float).eps * scale**2
    vertices: list[list[float]] = []
    upper_indices = np.empty((n_chord, n_span), dtype=int)
    lower_indices = np.empty((n_chord, n_span), dtype=int)
    for chord_index in range(n_chord):
        for span_index in range(n_span):
            upper_indices[chord_index, span_index] = len(vertices)
            vertices.append(
                upper[chord_index, span_index].tolist()
            )
    leading_weld = 0
    trailing_weld = 0
    for chord_index in range(n_chord):
        for span_index in range(n_span):
            separation = float(
                np.linalg.norm(
                    upper[chord_index, span_index]
                    - lower[chord_index, span_index]
                )
            )
            if separation <= weld_tolerance:
                lower_indices[chord_index, span_index] = (
                    upper_indices[chord_index, span_index]
                )
                if chord_index == 0:
                    leading_weld += 1
                if chord_index == n_chord - 1:
                    trailing_weld += 1
            else:
                lower_indices[chord_index, span_index] = len(vertices)
                vertices.append(
                    lower[chord_index, span_index].tolist()
                )

    faces: list[list[int]] = []
    roles: list[str] = []
    for chord_index in range(n_chord - 1):
        for span_index in range(n_span - 1):
            u00 = int(upper_indices[chord_index, span_index])
            u10 = int(upper_indices[chord_index + 1, span_index])
            u11 = int(
                upper_indices[chord_index + 1, span_index + 1]
            )
            u01 = int(upper_indices[chord_index, span_index + 1])
            for triangle in ((u00, u10, u11), (u00, u11, u01)):
                _append_triangle_if_nondegenerate(
                    faces,
                    roles,
                    vertices,
                    triangle,
                    "upper",
                    area_floor=area_floor,
                )

            l00 = int(lower_indices[chord_index, span_index])
            l10 = int(lower_indices[chord_index + 1, span_index])
            l11 = int(
                lower_indices[chord_index + 1, span_index + 1]
            )
            l01 = int(lower_indices[chord_index, span_index + 1])
            for triangle in ((l00, l11, l10), (l00, l01, l11)):
                _append_triangle_if_nondegenerate(
                    faces,
                    roles,
                    vertices,
                    triangle,
                    "lower",
                    area_floor=area_floor,
                )

    # Root cap: outward direction is negative span.
    root = 0
    for chord_index in range(n_chord - 1):
        u0 = int(upper_indices[chord_index, root])
        u1 = int(upper_indices[chord_index + 1, root])
        l0 = int(lower_indices[chord_index, root])
        l1 = int(lower_indices[chord_index + 1, root])
        for triangle in ((u0, l0, l1), (u0, l1, u1)):
            _append_triangle_if_nondegenerate(
                faces,
                roles,
                vertices,
                triangle,
                "root_cap",
                area_floor=area_floor,
            )

    # Tip cap: outward direction is positive span.
    tip = n_span - 1
    for chord_index in range(n_chord - 1):
        u0 = int(upper_indices[chord_index, tip])
        u1 = int(upper_indices[chord_index + 1, tip])
        l0 = int(lower_indices[chord_index, tip])
        l1 = int(lower_indices[chord_index + 1, tip])
        for triangle in ((u0, u1, l1), (u0, l1, l0)):
            _append_triangle_if_nondegenerate(
                faces,
                roles,
                vertices,
                triangle,
                "tip_cap",
                area_floor=area_floor,
            )

    # Finite trailing-edge base: outward direction is downstream.
    trailing = n_chord - 1
    for span_index in range(n_span - 1):
        u0 = int(upper_indices[trailing, span_index])
        u1 = int(upper_indices[trailing, span_index + 1])
        l0 = int(lower_indices[trailing, span_index])
        l1 = int(lower_indices[trailing, span_index + 1])
        for triangle in ((u0, l0, l1), (u0, l1, u1)):
            _append_triangle_if_nondegenerate(
                faces,
                roles,
                vertices,
                triangle,
                "trailing_base",
                area_floor=area_floor,
            )

    mesh = closed_triangular_mesh(
        np.asarray(vertices, dtype=float),
        np.asarray(faces, dtype=int),
    )
    mean_change = float(
        np.max(np.abs(mean - geometry_before), initial=0.0)
    )
    return RoboEagleClosedShell(
        mesh=mesh,
        upper_vertex_indices=upper_indices,
        lower_vertex_indices=lower_indices,
        face_roles=np.asarray(roles, dtype="U24"),
        maximum_material_pairing_error=pairing_error,
        maximum_mean_surface_change=mean_change,
        leading_edge_weld_count=leading_weld,
        trailing_edge_weld_count=trailing_weld,
    )


def source_influence_matrix(
    mesh: ClosedTriangularMesh,
) -> tuple[np.ndarray, np.ndarray]:
    """Return exterior source velocity and normal influence at centroids."""
    if not isinstance(mesh, ClosedTriangularMesh):
        raise ThickBodyNeumannError(
            "mesh must be validated by closed_triangular_mesh"
        )
    panel_count = len(mesh.faces)
    velocity = np.empty((panel_count, panel_count, 3), dtype=float)
    for source_index, face in enumerate(mesh.faces):
        influence = constant_source_polygon_influence(
            mesh.vertices[face],
            mesh.centroids,
            on_surface_side="exterior",
        )
        velocity[:, source_index, :] = influence.velocity
    normal_matrix = np.einsum(
        "ijk,ik->ij", velocity, mesh.normals
    )
    return velocity, normal_matrix


def solve_conditioned_neumann_source(
    mesh: ClosedTriangularMesh,
    *,
    incident_velocity: Any,
    wall_velocity: Any | None = None,
) -> NeumannSourceSolution:
    """Solve the source field that cancels total incident wall-normal flow."""
    if not isinstance(mesh, ClosedTriangularMesh):
        raise ThickBodyNeumannError(
            "mesh must be validated by closed_triangular_mesh"
        )
    incident = _finite("incident_velocity", incident_velocity, ndim=2)
    if incident.shape != mesh.centroids.shape:
        raise ThickBodyNeumannError(
            "incident_velocity must match mesh panel centroids"
        )
    if wall_velocity is None:
        wall = np.zeros_like(incident)
    else:
        wall = _finite("wall_velocity", wall_velocity, ndim=2)
        if wall.shape != incident.shape:
            raise ThickBodyNeumannError(
                "wall_velocity must match incident_velocity"
            )
    velocity_influence, matrix = source_influence_matrix(mesh)
    right_hand_side = np.einsum(
        "ij,ij->i", wall - incident, mesh.normals
    )
    condition_number = float(np.linalg.cond(matrix))
    if not np.isfinite(condition_number):
        raise ThickBodyNeumannError(
            "source influence matrix is singular or non-finite"
        )
    try:
        source_strength = np.linalg.solve(matrix, right_hand_side)
    except np.linalg.LinAlgError as exc:
        raise ThickBodyNeumannError(
            "source influence solve failed"
        ) from exc
    source_velocity = np.einsum(
        "ijk,j->ik", velocity_influence, source_strength
    )
    source_potential = np.zeros(len(mesh.faces), dtype=float)
    for source_index, face in enumerate(mesh.faces):
        source_potential += constant_source_polygon_influence(
            mesh.vertices[face],
            mesh.centroids,
            strength=float(source_strength[source_index]),
            on_surface_side="principal",
        ).potential
    total_velocity = incident + source_velocity
    residual = np.einsum(
        "ij,ij->i", total_velocity - wall, mesh.normals
    )
    residual_scale = max(
        float(
            np.max(
                np.linalg.norm(incident - wall, axis=1),
                initial=0.0,
            )
        ),
        np.finfo(float).tiny,
    )
    max_residual = float(np.max(np.abs(residual), initial=0.0))
    source_flux = float(np.dot(source_strength, mesh.areas))
    flux_scale = max(
        float(np.dot(np.abs(source_strength), mesh.areas)),
        np.finfo(float).tiny,
    )
    return NeumannSourceSolution(
        mesh=mesh,
        incident_velocity=incident.copy(),
        wall_velocity=wall.copy(),
        source_strength=source_strength,
        source_potential=source_potential,
        source_velocity=source_velocity,
        total_velocity=total_velocity,
        influence_matrix=matrix,
        right_hand_side=right_hand_side,
        max_no_penetration_residual=max_residual,
        relative_no_penetration_residual=max_residual / residual_scale,
        source_flux=source_flux,
        relative_source_flux=abs(source_flux) / flux_scale,
        condition_number=condition_number,
    )
