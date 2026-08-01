"""Explicit material-time history for the frozen P2 doublet potential.

This module supplies one missing input to the dual-side Bernoulli observer:
the wall-material derivative of the scalar potential induced by a moving
free doublet sheet.  It requires three real geometry/field stages and uses
the derivative of their quadratic Lagrange interpolant.  It never invents a
midpoint, smooths the potential history, or computes pressure or force.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .distributed_doublet import (
    DistributedDoubletError,
    QuadraticDoubletSurface,
)
from .doublet_potential import surface_doublet_potential


class MaterialPotentialHistoryError(ValueError):
    """Invalid topology, material identity, time stages, or wall history."""


@dataclass(frozen=True)
class MaterialPotentialRate:
    times: np.ndarray
    target_index: int
    derivative_weights: np.ndarray
    potential_by_stage: np.ndarray
    wall_material_rate: np.ndarray
    max_material_mu_residual: float
    topology_equal: bool


def three_stage_lagrange_derivative_weights(
    times,
    *,
    target_index: int,
) -> np.ndarray:
    """Derivative weights for three explicit, noncoincident time stages."""
    stage_times = np.asarray(times, dtype=float)
    if stage_times.shape != (3,) or not np.all(np.isfinite(stage_times)):
        raise MaterialPotentialHistoryError(
            "times must contain exactly three finite stages"
        )
    if np.any(np.diff(stage_times) <= 0.0):
        raise MaterialPotentialHistoryError(
            "times must be strictly increasing"
        )
    if target_index not in (0, 1, 2):
        raise MaterialPotentialHistoryError(
            "target_index must identify one of the three explicit stages"
        )
    target = stage_times[target_index]
    weights = np.zeros(3, dtype=float)
    for node in range(3):
        for differentiated_factor in range(3):
            if differentiated_factor == node:
                continue
            term = 1.0 / (
                stage_times[node]
                - stage_times[differentiated_factor]
            )
            for factor in range(3):
                if factor in (node, differentiated_factor):
                    continue
                term *= (
                    target - stage_times[factor]
                ) / (
                    stage_times[node] - stage_times[factor]
                )
            weights[node] += term
    return weights


def material_potential_history_rate(
    surfaces: Sequence[QuadraticDoubletSurface],
    wall_points_by_stage,
    times,
    *,
    target_index: int = 2,
    quadrature_order: int = 32,
) -> MaterialPotentialRate:
    """Differentiate free-sheet potential along explicit wall trajectories.

    Each row of ``wall_points_by_stage`` contains the same material wall
    points at one of the three time stages.  Free-sheet topology and P2
    material ``mu`` must remain identical across the stages; geometry may
    move or deform.
    """
    if len(surfaces) != 3 or not all(
        isinstance(surface, QuadraticDoubletSurface)
        for surface in surfaces
    ):
        raise MaterialPotentialHistoryError(
            "surfaces must contain exactly three P2 DDE stages"
        )
    weights = three_stage_lagrange_derivative_weights(
        times,
        target_index=target_index,
    )
    wall_points = np.asarray(wall_points_by_stage, dtype=float)
    if (
        wall_points.ndim != 3
        or wall_points.shape[0] != 3
        or wall_points.shape[2] != 3
        or not np.all(np.isfinite(wall_points))
    ):
        raise MaterialPotentialHistoryError(
            "wall_points_by_stage must have finite shape (3,npoint,3)"
        )
    reference = surfaces[0]
    topology_equal = all(
        surface.vertices.shape == reference.vertices.shape
        and np.array_equal(surface.faces, reference.faces)
        for surface in surfaces[1:]
    )
    if not topology_equal:
        raise MaterialPotentialHistoryError(
            "free-sheet topology changed across material time stages"
        )
    mu_residual = float(
        max(
            (
                np.max(
                    np.abs(surface.face_mu - reference.face_mu),
                    initial=0.0,
                )
                for surface in surfaces[1:]
            ),
            default=0.0,
        )
    )
    if mu_residual != 0.0:
        raise MaterialPotentialHistoryError(
            "P2 material mu changed across time stages"
        )
    try:
        potential = np.stack(
            [
                surface_doublet_potential(
                    surface,
                    wall_points[stage],
                    quadrature_order=quadrature_order,
                )
                for stage, surface in enumerate(surfaces)
            ]
        )
    except DistributedDoubletError as exc:
        raise MaterialPotentialHistoryError(str(exc)) from exc
    rate = np.einsum("s,sn->n", weights, potential)
    if not np.all(np.isfinite(rate)):
        raise MaterialPotentialHistoryError(
            "material potential rate contains non-finite values"
        )
    return MaterialPotentialRate(
        times=np.asarray(times, dtype=float).copy(),
        target_index=target_index,
        derivative_weights=weights,
        potential_by_stage=potential,
        wall_material_rate=rate,
        max_material_mu_residual=mu_residual,
        topology_equal=topology_equal,
    )
