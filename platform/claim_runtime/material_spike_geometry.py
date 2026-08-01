"""Geometric identity oracle for the N2.6c1 material-spike claim.

Santhosh et al. (JFM 969, A25, 2023) define finite-time material folding with
the change in the Weingarten map of the *same fluid material surface*,

    W_bar(t0, t) = W(t) - W(t0).

This module evaluates that geometric identity on structured material
coordinates.  It deliberately does not:

* advect material surfaces from a velocity field;
* extract a ridge or declare a separation location;
* accept wall-shear, LESP, force, pressure, or structural channels; or
* select the material time interval.

Those omissions are claim boundaries, not implementation TODOs hidden behind
defaults.  Physical promotion still requires independent near-wall field data,
flow-map integration, and a converged positive-ridge extraction.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class MaterialSpikeGeometryError(ValueError):
    """Invalid material surface, observer map, or curvature spectrum."""


def _finite_array(name: str, value, *, ndim: int) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim != ndim:
        raise MaterialSpikeGeometryError(
            f"{name} must have {ndim} dimensions, got {array.shape}"
        )
    if not np.all(np.isfinite(array)):
        raise MaterialSpikeGeometryError(f"{name} contains non-finite values")
    return array.copy()


def _material_coordinate(name: str, value, count: int) -> np.ndarray:
    coordinate = _finite_array(name, value, ndim=1)
    if len(coordinate) != count or count < 3:
        raise MaterialSpikeGeometryError(
            f"{name} must contain one value for each of at least three nodes"
        )
    if np.any(np.diff(coordinate) <= 0.0):
        raise MaterialSpikeGeometryError(
            f"{name} must be strictly increasing"
        )
    return coordinate


def _proper_rotation(rotation, *, tolerance: float) -> np.ndarray:
    result = _finite_array("rotation", rotation, ndim=2)
    if result.shape != (3, 3):
        raise MaterialSpikeGeometryError(
            f"rotation must have shape (3,3), got {result.shape}"
        )
    orthogonality = np.linalg.norm(result.T@result-np.eye(3), ord=np.inf)
    determinant = float(np.linalg.det(result))
    if orthogonality > tolerance or abs(determinant-1.0) > tolerance:
        raise MaterialSpikeGeometryError(
            "rotation must be a proper orthogonal map (Q.T Q=I, det(Q)=+1)"
        )
    return result


def _real_ordered_eigensystem(
    matrices: np.ndarray,
    *,
    imaginary_tolerance: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ascending real eigenvalues and matching column eigenvectors."""
    leading_shape = matrices.shape[:-2]
    flat = matrices.reshape((-1, 2, 2))
    values = np.empty((len(flat), 2), dtype=float)
    vectors = np.empty((len(flat), 2, 2), dtype=float)
    for index, matrix in enumerate(flat):
        eigenvalues, eigenvectors = np.linalg.eig(matrix)
        imaginary = max(
            float(np.max(np.abs(eigenvalues.imag))),
            float(np.max(np.abs(eigenvectors.imag))),
        )
        if imaginary > imaginary_tolerance:
            raise MaterialSpikeGeometryError(
                "Weingarten map has a non-real numerical eigensystem; "
                "check surface resolution and orientation"
            )
        order = np.argsort(eigenvalues.real)
        values[index] = eigenvalues.real[order]
        selected = eigenvectors.real[:, order]
        norms = np.linalg.norm(selected, axis=0)
        if np.any(norms <= np.finfo(float).eps):
            raise MaterialSpikeGeometryError(
                "principal direction is numerically singular"
            )
        vectors[index] = selected/norms[None, :]
    return (
        values.reshape(leading_shape+(2,)),
        vectors.reshape(leading_shape+(2, 2)),
    )


@dataclass(frozen=True)
class MaterialSurfaceCurvature:
    """Intrinsic and extrinsic geometry of one fluid material surface."""

    position: np.ndarray
    tangent_u: np.ndarray
    tangent_v: np.ndarray
    normal: np.ndarray
    area_jacobian: np.ndarray
    first_fundamental_form: np.ndarray
    second_fundamental_form: np.ndarray
    weingarten_map: np.ndarray
    principal_curvatures: np.ndarray
    principal_directions_material: np.ndarray
    principal_directions_spatial: np.ndarray
    mean_curvature: np.ndarray
    gaussian_curvature: np.ndarray


@dataclass(frozen=True)
class MaterialCurvatureChange:
    """Finite-time folding of the same material surface over [t0, t]."""

    initial_time: float
    final_time: float
    initial: MaterialSurfaceCurvature
    final: MaterialSurfaceCurvature
    weingarten_change: np.ndarray
    principal_curvature_changes: np.ndarray
    principal_change_directions_material: np.ndarray
    largest_principal_curvature_change: np.ndarray
    mean_curvature_change: np.ndarray
    gaussian_curvature_change: np.ndarray


def material_surface_curvature(
    position,
    *,
    u,
    v,
    reference_normal=None,
    minimum_area_jacobian: float = 1.0e-12,
    imaginary_tolerance: float = 1.0e-10,
) -> MaterialSurfaceCurvature:
    """Evaluate Eq. (3.3) of Santhosh et al. on a structured surface.

    ``position[i,j]`` must retain material identity: the same ``(u[i],v[j])``
    labels are used at all times.  The returned normal follows
    ``cross(dposition/du, dposition/dv)``.
    """
    points = _finite_array("position", position, ndim=3)
    if points.shape[-1] != 3:
        raise MaterialSpikeGeometryError(
            f"position must have shape (nu,nv,3), got {points.shape}"
        )
    u_coordinate = _material_coordinate("u", u, points.shape[0])
    v_coordinate = _material_coordinate("v", v, points.shape[1])
    minimum_area = float(minimum_area_jacobian)
    if not np.isfinite(minimum_area) or minimum_area <= 0.0:
        raise MaterialSpikeGeometryError(
            "minimum_area_jacobian must be finite and positive"
        )
    imaginary_limit = float(imaginary_tolerance)
    if not np.isfinite(imaginary_limit) or imaginary_limit <= 0.0:
        raise MaterialSpikeGeometryError(
            "imaginary_tolerance must be finite and positive"
        )

    tangent_u = np.gradient(
        points,
        u_coordinate,
        axis=0,
        edge_order=2,
    )
    tangent_v = np.gradient(
        points,
        v_coordinate,
        axis=1,
        edge_order=2,
    )
    area_vector = np.cross(tangent_u, tangent_v)
    area_jacobian = np.linalg.norm(area_vector, axis=-1)
    if np.any(area_jacobian <= minimum_area):
        raise MaterialSpikeGeometryError(
            "material surface metric is degenerate at one or more nodes"
        )
    normal = area_vector/area_jacobian[..., None]

    if reference_normal is not None:
        reference = _finite_array(
            "reference_normal",
            reference_normal,
            ndim=3,
        )
        if reference.shape != points.shape:
            raise MaterialSpikeGeometryError(
                "reference_normal must match position"
            )
        reference_norm = np.linalg.norm(reference, axis=-1)
        if np.max(np.abs(reference_norm-1.0), initial=0.0) > 1.0e-10:
            raise MaterialSpikeGeometryError(
                "reference_normal must contain unit vectors"
            )
        if np.any(np.einsum("...i,...i->...", normal, reference) <= 0.0):
            raise MaterialSpikeGeometryError(
                "material orientation is reversed relative to reference_normal"
            )

    first_form = np.empty(points.shape[:2]+(2, 2), dtype=float)
    first_form[..., 0, 0] = np.einsum(
        "...i,...i->...",
        tangent_u,
        tangent_u,
    )
    first_form[..., 0, 1] = np.einsum(
        "...i,...i->...",
        tangent_u,
        tangent_v,
    )
    first_form[..., 1, 0] = first_form[..., 0, 1]
    first_form[..., 1, 1] = np.einsum(
        "...i,...i->...",
        tangent_v,
        tangent_v,
    )

    second_u = np.gradient(
        tangent_u,
        u_coordinate,
        axis=0,
        edge_order=2,
    )
    mixed = 0.5*(
        np.gradient(
            tangent_u,
            v_coordinate,
            axis=1,
            edge_order=2,
        )
        +np.gradient(
            tangent_v,
            u_coordinate,
            axis=0,
            edge_order=2,
        )
    )
    second_v = np.gradient(
        tangent_v,
        v_coordinate,
        axis=1,
        edge_order=2,
    )
    second_form = np.empty_like(first_form)
    second_form[..., 0, 0] = np.einsum(
        "...i,...i->...",
        normal,
        second_u,
    )
    second_form[..., 0, 1] = np.einsum(
        "...i,...i->...",
        normal,
        mixed,
    )
    second_form[..., 1, 0] = second_form[..., 0, 1]
    second_form[..., 1, 1] = np.einsum(
        "...i,...i->...",
        normal,
        second_v,
    )

    determinant = np.linalg.det(first_form)
    if np.any(determinant <= minimum_area*minimum_area):
        raise MaterialSpikeGeometryError(
            "first fundamental form is singular"
        )
    try:
        weingarten = np.linalg.solve(first_form, second_form)
    except np.linalg.LinAlgError as error:
        raise MaterialSpikeGeometryError(
            "first fundamental form is singular"
        ) from error

    principal_curvatures, principal_material = _real_ordered_eigensystem(
        weingarten,
        imaginary_tolerance=imaginary_limit,
    )
    tangent_basis = np.stack((tangent_u, tangent_v), axis=-2)
    principal_spatial = np.einsum(
        "...ak,...ac->...kc",
        principal_material,
        tangent_basis,
    )
    direction_norm = np.linalg.norm(principal_spatial, axis=-1)
    principal_spatial = principal_spatial/direction_norm[..., None]

    return MaterialSurfaceCurvature(
        position=points,
        tangent_u=tangent_u,
        tangent_v=tangent_v,
        normal=normal,
        area_jacobian=area_jacobian,
        first_fundamental_form=first_form,
        second_fundamental_form=second_form,
        weingarten_map=weingarten,
        principal_curvatures=principal_curvatures,
        principal_directions_material=principal_material,
        principal_directions_spatial=principal_spatial,
        mean_curvature=0.5*np.trace(
            weingarten,
            axis1=-2,
            axis2=-1,
        ),
        gaussian_curvature=np.linalg.det(weingarten),
    )


def material_curvature_change(
    initial_position,
    final_position,
    *,
    u,
    v,
    initial_time: float,
    final_time: float,
    minimum_area_jacobian: float = 1.0e-12,
    imaginary_tolerance: float = 1.0e-10,
) -> MaterialCurvatureChange:
    """Evaluate the finite-time folding measure in Eq. (4.1a).

    This function requires material correspondence but does not infer it.  A
    pair of unrelated Eulerian surfaces is therefore invalid evidence even if
    the arrays have matching shapes.
    """
    t0 = float(initial_time)
    t1 = float(final_time)
    if (
        not np.isfinite(t0)
        or not np.isfinite(t1)
        or t1 <= t0
    ):
        raise MaterialSpikeGeometryError(
            "final_time must be finite and greater than initial_time"
        )
    initial = material_surface_curvature(
        initial_position,
        u=u,
        v=v,
        minimum_area_jacobian=minimum_area_jacobian,
        imaginary_tolerance=imaginary_tolerance,
    )
    final = material_surface_curvature(
        final_position,
        u=u,
        v=v,
        minimum_area_jacobian=minimum_area_jacobian,
        imaginary_tolerance=imaginary_tolerance,
    )
    if initial.position.shape != final.position.shape:
        raise MaterialSpikeGeometryError(
            "initial and final surfaces must share material topology"
        )
    change = final.weingarten_map-initial.weingarten_map
    principal_changes, directions = _real_ordered_eigensystem(
        change,
        imaginary_tolerance=imaginary_tolerance,
    )
    return MaterialCurvatureChange(
        initial_time=t0,
        final_time=t1,
        initial=initial,
        final=final,
        weingarten_change=change,
        principal_curvature_changes=principal_changes,
        principal_change_directions_material=directions,
        largest_principal_curvature_change=principal_changes[..., 1],
        mean_curvature_change=0.5*np.trace(
            change,
            axis1=-2,
            axis2=-1,
        ),
        gaussian_curvature_change=np.linalg.det(change),
    )


def proper_euclidean_observer_transform(
    position,
    *,
    rotation,
    translation,
    tolerance: float = 1.0e-11,
) -> np.ndarray:
    """Apply x_tilde=Qx+b after validating Q is in SO(3)."""
    points = _finite_array("position", position, ndim=3)
    if points.shape[-1] != 3:
        raise MaterialSpikeGeometryError(
            f"position must have shape (nu,nv,3), got {points.shape}"
        )
    limit = float(tolerance)
    if not np.isfinite(limit) or limit <= 0.0:
        raise MaterialSpikeGeometryError(
            "tolerance must be finite and positive"
        )
    proper = _proper_rotation(rotation, tolerance=limit)
    offset = _finite_array("translation", translation, ndim=1)
    if offset.shape != (3,):
        raise MaterialSpikeGeometryError(
            f"translation must have shape (3,), got {offset.shape}"
        )
    return points@proper.T+offset
