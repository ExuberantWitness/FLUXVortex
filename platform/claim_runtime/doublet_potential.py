"""Scalar-potential observer for the continuous P2 doublet state.

The existing DDE velocity oracle is the gradient of

    phi(P) = -1/(4*pi) integral_S mu(Q)
             n(Q).(P-Q) / |P-Q|^3 dS_Q.

For the aligned FLUXV interface ``chi=-mu_DDE``.  At a smooth owned sheet
point the Cauchy principal-value potential is the side average and the
Plemelj limits are

    phi_plus  = phi_bar + chi/2 = phi_bar - mu_DDE/2
    phi_minus = phi_bar - chi/2 = phi_bar + mu_DDE/2.

This module is a CPU equation oracle.  It computes no pressure or force and
contains no core, offset, regularization, or target-load information.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .distributed_doublet import (
    DistributedDoubletError,
    QuadraticDoubletElement,
    QuadraticDoubletSurface,
    _triangle_quadrature,
)


@dataclass(frozen=True)
class DoubletPotentialLimits:
    mean_potential: np.ndarray
    physical_potential_jump: np.ndarray
    potential_plus: np.ndarray
    potential_minus: np.ndarray
    max_jump_residual: float


def _points(value) -> np.ndarray:
    points = np.asarray(value, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise DistributedDoubletError(
            f"points must have shape (n,3), got {points.shape}"
        )
    if not np.all(np.isfinite(points)):
        raise DistributedDoubletError("points contain non-finite values")
    return points


def element_doublet_potential(
    element: QuadraticDoubletElement,
    points,
    *,
    quadrature_order: int = 32,
    plane_tolerance: float = 128.0 * np.finfo(float).eps,
) -> np.ndarray:
    """Evaluate the scalar double-layer potential or its coplanar PV.

    On the element plane the ordinary kernel is zero away from its
    singularity and its Cauchy principal value is zero.  The missing local
    ``+/- mu/2`` term is deliberately added only by
    :func:`dde_potential_side_limits`.
    """
    if not isinstance(element, QuadraticDoubletElement):
        raise DistributedDoubletError(
            "element must be a QuadraticDoubletElement"
        )
    if plane_tolerance < 0.0 or not np.isfinite(plane_tolerance):
        raise DistributedDoubletError(
            "plane_tolerance must be finite and non-negative"
        )
    point_array = _points(points)
    signed_distance = (
        point_array - element.vertices[0]
    ) @ element.normal
    length_scale = max(np.sqrt(2.0 * element.area), 1.0)
    coplanar = (
        np.abs(signed_distance) <= plane_tolerance * length_scale
    )
    result = np.zeros(len(point_array), dtype=float)
    if np.all(coplanar):
        return result

    barycentric, reference_weight = _triangle_quadrature(
        int(quadrature_order)
    )
    source = barycentric @ element.vertices
    strength = element.evaluate_barycentric(barycentric)
    physical_weight = (
        reference_weight * np.linalg.norm(element.area_vector)
    )
    separation = (
        point_array[~coplanar, None, :] - source[None, :, :]
    )
    radius_squared = np.einsum(
        "pqj,pqj->pq", separation, separation
    )
    if np.any(radius_squared <= np.finfo(float).tiny):
        raise DistributedDoubletError(
            "field point coincides with a potential quadrature point"
        )
    kernel = (
        separation @ element.normal
    ) / radius_squared**1.5
    result[~coplanar] = -np.einsum(
        "pq,q,q->p",
        kernel,
        strength,
        physical_weight,
    ) / (4.0 * np.pi)
    if not np.all(np.isfinite(result)):
        raise DistributedDoubletError(
            "doublet potential contains non-finite values"
        )
    return result


def surface_doublet_potential(
    surface: QuadraticDoubletSurface,
    points,
    *,
    quadrature_order: int = 32,
    plane_tolerance: float = 128.0 * np.finfo(float).eps,
) -> np.ndarray:
    """Evaluate the summed off-sheet/PV potential of one DDE surface."""
    if not isinstance(surface, QuadraticDoubletSurface):
        raise DistributedDoubletError(
            "surface must be a QuadraticDoubletSurface"
        )
    point_array = _points(points)
    result = np.zeros(len(point_array), dtype=float)
    for face_index in range(len(surface)):
        result += element_doublet_potential(
            surface.element(face_index),
            point_array,
            quadrature_order=quadrature_order,
            plane_tolerance=plane_tolerance,
        )
    return result


def surface_sheet_average_potential(
    surface: QuadraticDoubletSurface,
    face_indices,
    barycentric,
    *,
    quadrature_order: int = 32,
    plane_tolerance: float = 128.0 * np.finfo(float).eps,
) -> np.ndarray:
    """Evaluate the Cauchy principal-value potential at owned sheet points."""
    if not isinstance(surface, QuadraticDoubletSurface):
        raise DistributedDoubletError(
            "surface must be a QuadraticDoubletSurface"
        )
    owner = np.asarray(face_indices, dtype=np.int64)
    lam = np.asarray(barycentric, dtype=float)
    if (
        owner.ndim != 1
        or lam.shape != (len(owner), 3)
        or not np.all(np.isfinite(lam))
    ):
        raise DistributedDoubletError(
            "face_indices and barycentric must have shapes (n,) and (n,3)"
        )
    if np.any(owner < 0) or np.any(owner >= len(surface)):
        raise DistributedDoubletError(
            "face_indices contain invalid owners"
        )
    margin = 128.0 * np.finfo(float).eps
    if np.any(lam <= margin) or np.any(lam >= 1.0-margin):
        raise DistributedDoubletError(
            "sheet-average potential requires strict owner interior points"
        )
    if np.max(
        np.abs(np.sum(lam, axis=1)-1.0), initial=0.0
    ) > 1.0e-12:
        raise DistributedDoubletError(
            "barycentric coordinates must sum to one"
        )
    points = np.empty((len(owner), 3), dtype=float)
    for point_index, face_index in enumerate(owner):
        points[point_index] = (
            lam[point_index]
            @ surface.vertices[surface.faces[face_index]]
        )

    potential = np.zeros(len(points), dtype=float)
    for source_face in range(len(surface)):
        element = surface.element(source_face)
        delta = points-element.vertices[0]
        distance = delta @ element.normal
        length_scale = max(np.sqrt(2.0*element.area), 1.0)
        coplanar = (
            np.abs(distance) <= plane_tolerance*length_scale
        )
        nonowner_coplanar = coplanar & (owner != source_face)
        if np.any(nonowner_coplanar):
            projected = element.barycentric_coordinates(
                points[nonowner_coplanar],
                plane_tolerance=4.0*plane_tolerance*length_scale,
            )
            inside = np.all(projected >= -margin, axis=1) & np.all(
                projected <= 1.0+margin, axis=1
            )
            if np.any(inside):
                raise DistributedDoubletError(
                    "sheet point lies inside a non-owner element"
                )
        potential += element_doublet_potential(
            element,
            points,
            quadrature_order=quadrature_order,
            plane_tolerance=plane_tolerance,
        )
    return potential


def dde_potential_side_limits(
    mean_potential,
    dde_mu,
) -> DoubletPotentialLimits:
    """Apply the aligned N1/DDE Plemelj jump ``chi=-mu_DDE``."""
    mean = np.asarray(mean_potential, dtype=float)
    mu = np.asarray(dde_mu, dtype=float)
    if mean.ndim != 1 or mu.shape != mean.shape:
        raise DistributedDoubletError(
            "mean_potential and dde_mu must have the same shape (n,)"
        )
    if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(mu)):
        raise DistributedDoubletError(
            "mean_potential and dde_mu must be finite"
        )
    jump = -mu
    plus = mean + 0.5*jump
    minus = mean - 0.5*jump
    residual = float(
        np.max(np.abs((plus-minus)-jump), initial=0.0)
    )
    return DoubletPotentialLimits(
        mean_potential=mean.copy(),
        physical_potential_jump=jump,
        potential_plus=plus,
        potential_minus=minus,
        max_jump_residual=residual,
    )
