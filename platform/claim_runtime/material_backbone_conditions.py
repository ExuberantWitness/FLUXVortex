"""Differential-condition oracle for N2.6c1 material backbones.

For each near-wall material layer, Definition 1 and Proposition 2 of
Santhosh et al. (JFM 969, A25, 2023) require more than a large value of the
largest principal-curvature change:

* upwelling: kappa_bar_2 > 0;
* a 1-D backbone: a positive local maximum with zero gradient and
  negative-definite Hessian when the two curvature changes are degenerate;
* a 2-D backbone: stationarity and negative second derivative along the
  largest-principal direction when the eigenvalues are distinct.

This module evaluates those local conditions.  It does not connect candidates
across grid cells or wall-normal layers and therefore does not itself return a
physical spiking curve or separation surface.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class MaterialBackboneConditionError(ValueError):
    """Invalid curvature field, material coordinates, or error band."""


def _finite(name: str, value, *, ndim: int) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.ndim != ndim:
        raise MaterialBackboneConditionError(
            f"{name} must have {ndim} dimensions, got {result.shape}"
        )
    if not np.all(np.isfinite(result)):
        raise MaterialBackboneConditionError(
            f"{name} contains non-finite values"
        )
    return result.copy()


def _coordinate(name: str, value, count: int) -> np.ndarray:
    result = _finite(name, value, ndim=1)
    if len(result) != count or count < 5:
        raise MaterialBackboneConditionError(
            f"{name} must contain one value for each of at least five nodes"
        )
    if np.any(np.diff(result) <= 0.0):
        raise MaterialBackboneConditionError(
            f"{name} must be strictly increasing"
        )
    return result


def _positive_tolerance(name: str, value: float) -> float:
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise MaterialBackboneConditionError(
            f"{name} must be finite and positive"
        )
    return result


@dataclass(frozen=True)
class MaterialBackboneConditionDiagnostics:
    """Local Proposition-2 fields and numerical candidate masks."""

    largest_curvature_change: np.ndarray
    eigenvalue_gap: np.ndarray
    largest_direction_material: np.ndarray
    gradient_material: np.ndarray
    hessian_material: np.ndarray
    directional_derivative: np.ndarray
    directional_second_derivative: np.ndarray
    hessian_eigenvalues: np.ndarray
    positive_upwelling: np.ndarray
    degenerate_curvature_change: np.ndarray
    one_dimensional_candidate: np.ndarray
    two_dimensional_candidate: np.ndarray
    candidate: np.ndarray
    stationarity_tolerance: float
    negative_curvature_tolerance: float
    eigenvalue_gap_tolerance: float


def material_backbone_condition_diagnostics(
    principal_curvature_changes,
    largest_direction_material,
    *,
    u,
    v,
    stationarity_tolerance: float,
    negative_curvature_tolerance: float,
    eigenvalue_gap_tolerance: float,
) -> MaterialBackboneConditionDiagnostics:
    """Evaluate local 1-D/2-D backbone conditions on one material layer.

    Tolerances are mandatory numerical error bands.  Their convergence must be
    audited externally; this function does not infer them from a field or a
    load target.
    """
    curvature = _finite(
        "principal_curvature_changes",
        principal_curvature_changes,
        ndim=3,
    )
    if curvature.shape[-1] != 2:
        raise MaterialBackboneConditionError(
            "principal_curvature_changes must have shape (nu,nv,2)"
        )
    direction = _finite(
        "largest_direction_material",
        largest_direction_material,
        ndim=3,
    )
    if direction.shape != curvature.shape:
        raise MaterialBackboneConditionError(
            "largest_direction_material must have shape (nu,nv,2)"
        )
    u_coordinate = _coordinate("u", u, curvature.shape[0])
    v_coordinate = _coordinate("v", v, curvature.shape[1])
    stationarity = _positive_tolerance(
        "stationarity_tolerance",
        stationarity_tolerance,
    )
    negative_curvature = _positive_tolerance(
        "negative_curvature_tolerance",
        negative_curvature_tolerance,
    )
    gap_tolerance = _positive_tolerance(
        "eigenvalue_gap_tolerance",
        eigenvalue_gap_tolerance,
    )
    ordering_error = np.max(
        curvature[..., 0]-curvature[..., 1],
        initial=0.0,
    )
    if ordering_error > gap_tolerance:
        raise MaterialBackboneConditionError(
            "principal_curvature_changes must be sorted ascending"
        )

    direction_norm = np.linalg.norm(direction, axis=-1)
    if np.any(direction_norm <= np.finfo(float).eps):
        raise MaterialBackboneConditionError(
            "largest principal direction cannot vanish"
        )
    direction = direction/direction_norm[..., None]

    largest = curvature[..., 1]
    derivative_u = np.gradient(
        largest,
        u_coordinate,
        axis=0,
        edge_order=2,
    )
    derivative_v = np.gradient(
        largest,
        v_coordinate,
        axis=1,
        edge_order=2,
    )
    gradient = np.stack((derivative_u, derivative_v), axis=-1)

    second_uu = np.gradient(
        derivative_u,
        u_coordinate,
        axis=0,
        edge_order=2,
    )
    second_uv = 0.5*(
        np.gradient(
            derivative_u,
            v_coordinate,
            axis=1,
            edge_order=2,
        )
        +np.gradient(
            derivative_v,
            u_coordinate,
            axis=0,
            edge_order=2,
        )
    )
    second_vv = np.gradient(
        derivative_v,
        v_coordinate,
        axis=1,
        edge_order=2,
    )
    hessian = np.empty(curvature.shape[:2]+(2, 2), dtype=float)
    hessian[..., 0, 0] = second_uu
    hessian[..., 0, 1] = second_uv
    hessian[..., 1, 0] = second_uv
    hessian[..., 1, 1] = second_vv

    directional_derivative = np.einsum(
        "...i,...i->...",
        gradient,
        direction,
    )
    directional_second_derivative = np.einsum(
        "...i,...ij,...j->...",
        direction,
        hessian,
        direction,
    )
    hessian_eigenvalues = np.linalg.eigvalsh(hessian)

    gap = curvature[..., 1]-curvature[..., 0]
    degenerate = gap <= gap_tolerance
    positive = largest > 0.0
    gradient_norm = np.linalg.norm(gradient, axis=-1)
    one_dimensional = (
        degenerate
        &positive
        &(gradient_norm <= stationarity)
        &(hessian_eigenvalues[..., 1] < -negative_curvature)
    )
    two_dimensional = (
        (~degenerate)
        &positive
        &(np.abs(directional_derivative) <= stationarity)
        &(directional_second_derivative < -negative_curvature)
    )
    return MaterialBackboneConditionDiagnostics(
        largest_curvature_change=largest,
        eigenvalue_gap=gap,
        largest_direction_material=direction,
        gradient_material=gradient,
        hessian_material=hessian,
        directional_derivative=directional_derivative,
        directional_second_derivative=directional_second_derivative,
        hessian_eigenvalues=hessian_eigenvalues,
        positive_upwelling=positive,
        degenerate_curvature_change=degenerate,
        one_dimensional_candidate=one_dimensional,
        two_dimensional_candidate=two_dimensional,
        candidate=one_dimensional|two_dimensional,
        stationarity_tolerance=stationarity,
        negative_curvature_tolerance=negative_curvature,
        eigenvalue_gap_tolerance=gap_tolerance,
    )
