"""Objective differential geometry for the deforming N2.6 dual-side shell.

This module maps structured material mean-surface snapshots to dual-side wall
positions and finite-difference kinematics.  It supplies geometry only: no
pressure, force, boundary-layer state or separation information is accepted.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class DeformingShellError(ValueError):
    """Invalid or inverted structured material surface."""


def _finite(name: str, value, ndim: int) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim != ndim or not np.all(np.isfinite(array)):
        raise DeformingShellError(
            f"{name} must be a finite {ndim}D array, got {array.shape}"
        )
    return array.copy()


def _coordinates(name: str, value, count: int) -> np.ndarray:
    coordinate = _finite(name, value, 1)
    if len(coordinate) != count or len(coordinate) < 3:
        raise DeformingShellError(
            f"{name} must contain at least three surface nodes"
        )
    if np.any(np.diff(coordinate) <= 0.0):
        raise DeformingShellError(f"{name} must be strictly increasing")
    return coordinate


@dataclass(frozen=True)
class StructuredSurfaceGeometry:
    position: np.ndarray
    tangent_xi: np.ndarray
    tangent_eta: np.ndarray
    director: np.ndarray
    area_jacobian: np.ndarray
    first_fundamental_form: np.ndarray
    second_fundamental_form: np.ndarray
    shape_operator: np.ndarray
    mean_curvature: np.ndarray
    gaussian_curvature: np.ndarray


@dataclass(frozen=True)
class DualSurfaceSnapshot:
    mean_surface: np.ndarray
    upper_surface: np.ndarray
    lower_surface: np.ndarray
    director: np.ndarray
    half_thickness: np.ndarray


@dataclass(frozen=True)
class DeformingDualSurfaceKinematics:
    geometry: StructuredSurfaceGeometry
    current: DualSurfaceSnapshot
    mean_velocity: np.ndarray
    mean_acceleration: np.ndarray
    upper_velocity: np.ndarray
    upper_acceleration: np.ndarray
    lower_velocity: np.ndarray
    lower_acceleration: np.ndarray
    director_velocity: np.ndarray
    director_acceleration: np.ndarray


def structured_surface_geometry(
    position,
    *,
    xi,
    eta,
    reference_director=None,
    minimum_area_jacobian: float = 1.0e-12,
) -> StructuredSurfaceGeometry:
    """Compute first/second forms and curvature in material coordinates."""
    points = _finite("position", position, 3)
    if points.shape[2] != 3:
        raise DeformingShellError(
            f"position must have shape (nxi,neta,3), got {points.shape}"
        )
    xi_coordinate = _coordinates("xi", xi, points.shape[0])
    eta_coordinate = _coordinates("eta", eta, points.shape[1])
    if (
        not np.isfinite(minimum_area_jacobian)
        or minimum_area_jacobian <= 0.0
    ):
        raise DeformingShellError(
            "minimum_area_jacobian must be positive and finite"
        )
    tangent_xi = np.gradient(
        points,
        xi_coordinate,
        axis=0,
        edge_order=2,
    )
    tangent_eta = np.gradient(
        points,
        eta_coordinate,
        axis=1,
        edge_order=2,
    )
    area_vector = np.cross(tangent_xi, tangent_eta)
    area = np.linalg.norm(area_vector, axis=2)
    if np.any(area <= minimum_area_jacobian):
        raise DeformingShellError(
            "surface is degenerate at one or more material nodes"
        )
    director = area_vector/area[..., None]
    if reference_director is not None:
        reference = _finite("reference_director", reference_director, 3)
        if reference.shape != points.shape:
            raise DeformingShellError(
                "reference_director must match the surface shape"
            )
        reference_norm = np.linalg.norm(reference, axis=2)
        if np.max(np.abs(reference_norm-1.0), initial=0.0) > 1.0e-11:
            raise DeformingShellError(
                "reference_director must contain unit vectors"
            )
        orientation = np.einsum("...i,...i->...", director, reference)
        if np.any(orientation <= 0.0):
            raise DeformingShellError(
                "surface director flipped relative to its material reference"
            )

    metric = np.empty(points.shape[:2]+(2, 2), dtype=float)
    metric[..., 0, 0] = np.einsum(
        "...i,...i->...",
        tangent_xi,
        tangent_xi,
    )
    metric[..., 0, 1] = np.einsum(
        "...i,...i->...",
        tangent_xi,
        tangent_eta,
    )
    metric[..., 1, 0] = metric[..., 0, 1]
    metric[..., 1, 1] = np.einsum(
        "...i,...i->...",
        tangent_eta,
        tangent_eta,
    )

    second_xi = np.gradient(
        tangent_xi,
        xi_coordinate,
        axis=0,
        edge_order=2,
    )
    mixed = 0.5*(
        np.gradient(
            tangent_xi,
            eta_coordinate,
            axis=1,
            edge_order=2,
        )
        + np.gradient(
            tangent_eta,
            xi_coordinate,
            axis=0,
            edge_order=2,
        )
    )
    second_eta = np.gradient(
        tangent_eta,
        eta_coordinate,
        axis=1,
        edge_order=2,
    )
    second_form = np.empty_like(metric)
    second_form[..., 0, 0] = np.einsum(
        "...i,...i->...",
        director,
        second_xi,
    )
    second_form[..., 0, 1] = np.einsum(
        "...i,...i->...",
        director,
        mixed,
    )
    second_form[..., 1, 0] = second_form[..., 0, 1]
    second_form[..., 1, 1] = np.einsum(
        "...i,...i->...",
        director,
        second_eta,
    )
    try:
        inverse_metric = np.linalg.inv(metric)
    except np.linalg.LinAlgError as error:
        raise DeformingShellError("surface metric is singular") from error
    shape_operator = np.einsum(
        "...ij,...jk->...ik",
        inverse_metric,
        second_form,
    )
    mean_curvature = 0.5*np.trace(
        shape_operator,
        axis1=2,
        axis2=3,
    )
    gaussian_curvature = np.linalg.det(shape_operator)
    return StructuredSurfaceGeometry(
        position=points,
        tangent_xi=tangent_xi,
        tangent_eta=tangent_eta,
        director=director,
        area_jacobian=area,
        first_fundamental_form=metric,
        second_fundamental_form=second_form,
        shape_operator=shape_operator,
        mean_curvature=mean_curvature,
        gaussian_curvature=gaussian_curvature,
    )


def dual_surface_snapshot(
    geometry: StructuredSurfaceGeometry,
    *,
    half_thickness,
) -> DualSurfaceSnapshot:
    """Offset one material mean surface along its objective director."""
    if not isinstance(geometry, StructuredSurfaceGeometry):
        raise DeformingShellError(
            "geometry must be StructuredSurfaceGeometry"
        )
    thickness = _finite("half_thickness", half_thickness, 2)
    if thickness.shape != geometry.position.shape[:2]:
        raise DeformingShellError(
            "half_thickness must match the material surface"
        )
    if np.any(thickness < 0.0):
        raise DeformingShellError("half_thickness cannot be negative")
    offset = thickness[..., None]*geometry.director
    return DualSurfaceSnapshot(
        mean_surface=geometry.position.copy(),
        upper_surface=geometry.position+offset,
        lower_surface=geometry.position-offset,
        director=geometry.director.copy(),
        half_thickness=thickness,
    )


def deforming_dual_surface_kinematics(
    *,
    previous_mean_surface,
    current_mean_surface,
    next_mean_surface,
    xi,
    eta,
    half_thickness,
    dt: float,
    reference_director=None,
) -> DeformingDualSurfaceKinematics:
    """Build dual-side shell snapshots and centred position kinematics."""
    if not np.isfinite(dt) or dt <= 0.0:
        raise DeformingShellError("dt must be positive and finite")
    current_geometry = structured_surface_geometry(
        current_mean_surface,
        xi=xi,
        eta=eta,
        reference_director=reference_director,
    )
    previous_geometry = structured_surface_geometry(
        previous_mean_surface,
        xi=xi,
        eta=eta,
    )
    next_geometry = structured_surface_geometry(
        next_mean_surface,
        xi=xi,
        eta=eta,
    )
    current = dual_surface_snapshot(
        current_geometry,
        half_thickness=half_thickness,
    )
    previous = dual_surface_snapshot(
        previous_geometry,
        half_thickness=half_thickness,
    )
    following = dual_surface_snapshot(
        next_geometry,
        half_thickness=half_thickness,
    )

    def velocity(before, after):
        return (after-before)/(2.0*dt)

    def acceleration(before, present, after):
        return (after-2.0*present+before)/(dt*dt)

    return DeformingDualSurfaceKinematics(
        geometry=current_geometry,
        current=current,
        mean_velocity=velocity(
            previous.mean_surface,
            following.mean_surface,
        ),
        mean_acceleration=acceleration(
            previous.mean_surface,
            current.mean_surface,
            following.mean_surface,
        ),
        upper_velocity=velocity(
            previous.upper_surface,
            following.upper_surface,
        ),
        upper_acceleration=acceleration(
            previous.upper_surface,
            current.upper_surface,
            following.upper_surface,
        ),
        lower_velocity=velocity(
            previous.lower_surface,
            following.lower_surface,
        ),
        lower_acceleration=acceleration(
            previous.lower_surface,
            current.lower_surface,
            following.lower_surface,
        ),
        director_velocity=velocity(
            previous.director,
            following.director,
        ),
        director_acceleration=acceleration(
            previous.director,
            current.director,
            following.director,
        ),
    )


def rigidly_transform_surface_geometry(
    geometry: StructuredSurfaceGeometry,
    *,
    rotation,
    translation,
    xi,
    eta,
) -> StructuredSurfaceGeometry:
    """Recompute geometry after one proper rigid transform."""
    if not isinstance(geometry, StructuredSurfaceGeometry):
        raise DeformingShellError(
            "geometry must be StructuredSurfaceGeometry"
        )
    matrix = _finite("rotation", rotation, 2)
    shift = _finite("translation", translation, 1)
    if matrix.shape != (3, 3) or shift.shape != (3,):
        raise DeformingShellError(
            "rotation and translation must have shapes (3,3) and (3,)"
        )
    if (
        np.max(np.abs(matrix.T@matrix-np.eye(3)), initial=0.0) > 1.0e-12
        or abs(float(np.linalg.det(matrix))-1.0) > 1.0e-12
    ):
        raise DeformingShellError("rotation must be proper orthogonal")
    transformed_position = np.einsum(
        "ij,...j->...i",
        matrix,
        geometry.position,
    )+shift
    transformed_director = np.einsum(
        "ij,...j->...i",
        matrix,
        geometry.director,
    )
    return structured_surface_geometry(
        transformed_position,
        xi=xi,
        eta=eta,
        reference_director=transformed_director,
    )

