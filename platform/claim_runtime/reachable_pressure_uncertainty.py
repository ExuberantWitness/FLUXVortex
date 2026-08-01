"""Same-space numerical uncertainty primitives for the S3ai-v2 observer.

The functions in this module know nothing about a body, wake, pressure, or
force.  They operate only on already observed vectors in one fixed finite
element dual space.  This separation prevents BIE, compatibility, or
attachment residuals with different units from being folded into the
pressure-law uncertainty ball.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


class ReachablePressureUncertaintyError(ValueError):
    """A same-space convergence or uncertainty definition is invalid."""


@dataclass(frozen=True)
class RichardsonReport:
    """Second-order three-level Richardson observation."""

    fine_extrapolated: np.ndarray
    medium_extrapolated: np.ndarray
    coarse_medium_change: float
    medium_fine_change: float
    contraction_ratio: float
    floating_floor: float
    plateau: bool
    allowance: float
    passed: bool


@dataclass(frozen=True)
class QuadratureTailReport:
    """Order-free three-level contracting-tail observation."""

    coarse_medium_change: float
    medium_fine_change: float
    contraction_ratio: float
    floating_floor: float
    plateau: bool
    allowance: float
    passed: bool


@dataclass(frozen=True)
class MixedCubeReport:
    """Pairwise and three-way Möbius differences on one adjacent cube."""

    epsilon_timestep: np.ndarray
    epsilon_quadrature: np.ndarray
    timestep_quadrature: np.ndarray
    epsilon_timestep_quadrature: np.ndarray
    component_norms: dict[str, float]
    allowance: float


@dataclass(frozen=True)
class ResolvedInterval:
    """Triangle-inequality interval for the norm of one observation."""

    observed_norm: float
    uncertainty: float
    lower: float
    upper: float


def _finite_vector(name: str, value: Any) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if (
        array.ndim != 1
        or array.size < 1
        or not np.all(np.isfinite(array))
    ):
        raise ReachablePressureUncertaintyError(
            f"{name} must be a finite one-dimensional vector"
        )
    return array.copy()


def _common_vectors(**values: Any) -> dict[str, np.ndarray]:
    vectors = {
        name: _finite_vector(name, value)
        for name, value in values.items()
    }
    shapes = {value.shape for value in vectors.values()}
    if len(shapes) != 1:
        raise ReachablePressureUncertaintyError(
            "all observations must belong to one vector space"
        )
    return vectors


def _mass_matrix(value: Any, size: int) -> np.ndarray:
    mass = np.asarray(value, dtype=float)
    if (
        mass.shape != (size, size)
        or not np.all(np.isfinite(mass))
        or not np.allclose(mass, mass.T, rtol=0.0, atol=0.0)
    ):
        raise ReachablePressureUncertaintyError(
            "mass must be a finite exactly symmetric square matrix"
        )
    try:
        np.linalg.cholesky(mass)
    except np.linalg.LinAlgError as error:
        raise ReachablePressureUncertaintyError(
            "mass must be positive definite"
        ) from error
    return mass


def dual_mass_norm(value: Any, mass: Any) -> float:
    """Return ``sqrt(x.T M^-1 x)`` in one fixed dual space."""

    vector = _finite_vector("value", value)
    metric = _mass_matrix(mass, len(vector))
    solved = np.linalg.solve(metric, vector)
    square = float(vector @ solved)
    if square < -64.0 * np.finfo(float).eps:
        raise ReachablePressureUncertaintyError(
            "dual-mass norm received a non-positive metric"
        )
    return float(np.sqrt(max(square, 0.0)))


def floating_plateau_floor(
    values: list[Any] | tuple[Any, ...],
    mass: Any,
    *,
    factor: float = 4096.0,
) -> float:
    """Return the preregistered relative float64 plateau in the same norm."""

    if not np.isfinite(factor) or factor <= 0.0:
        raise ReachablePressureUncertaintyError(
            "floating plateau factor must be finite and positive"
        )
    if not values:
        raise ReachablePressureUncertaintyError(
            "at least one observation is required"
        )
    vectors = [_finite_vector(f"value_{index}", value) for index, value in enumerate(values)]
    size = len(vectors[0])
    if any(len(vector) != size for vector in vectors):
        raise ReachablePressureUncertaintyError(
            "floating plateau observations must share one vector space"
        )
    metric = _mass_matrix(mass, size)
    scale = max((dual_mass_norm(vector, metric) for vector in vectors), default=0.0)
    return float(factor * np.finfo(float).eps * scale)


def second_order_richardson(
    *,
    fine: Any,
    medium: Any,
    coarse: Any,
    mass: Any,
    contraction_ratio_min: float = 3.2,
    floating_factor: float = 4096.0,
) -> RichardsonReport:
    """Evaluate the frozen three-level second-order family."""

    vectors = _common_vectors(fine=fine, medium=medium, coarse=coarse)
    metric = _mass_matrix(mass, len(vectors["fine"]))
    ratio_min = float(contraction_ratio_min)
    if not np.isfinite(ratio_min) or ratio_min <= 1.0:
        raise ReachablePressureUncertaintyError(
            "contraction_ratio_min must exceed one"
        )
    fine_extrapolated = (
        4.0 * vectors["fine"] - vectors["medium"]
    ) / 3.0
    medium_extrapolated = (
        4.0 * vectors["medium"] - vectors["coarse"]
    ) / 3.0
    coarse_medium = dual_mass_norm(
        vectors["coarse"] - vectors["medium"],
        metric,
    )
    medium_fine = dual_mass_norm(
        vectors["medium"] - vectors["fine"],
        metric,
    )
    floor = floating_plateau_floor(
        list(vectors.values()),
        metric,
        factor=floating_factor,
    )
    plateau = max(coarse_medium, medium_fine) <= floor
    if medium_fine == 0.0:
        ratio = np.inf if coarse_medium > 0.0 else np.nan
    else:
        ratio = coarse_medium / medium_fine
    passed = bool(
        plateau
        or (
            coarse_medium > medium_fine
            and ratio >= ratio_min
        )
    )
    allowance = (
        dual_mass_norm(fine_extrapolated - vectors["fine"], metric)
        + dual_mass_norm(
            fine_extrapolated - medium_extrapolated,
            metric,
        )
    )
    return RichardsonReport(
        fine_extrapolated=fine_extrapolated,
        medium_extrapolated=medium_extrapolated,
        coarse_medium_change=coarse_medium,
        medium_fine_change=medium_fine,
        contraction_ratio=float(ratio),
        floating_floor=floor,
        plateau=plateau,
        allowance=float(allowance),
        passed=passed,
    )


def contracting_quadrature_tail(
    *,
    coarse: Any,
    medium: Any,
    fine: Any,
    mass: Any,
    floating_factor: float = 4096.0,
) -> QuadratureTailReport:
    """Evaluate a geometric tail only when the observed differences contract."""

    vectors = _common_vectors(coarse=coarse, medium=medium, fine=fine)
    metric = _mass_matrix(mass, len(vectors["fine"]))
    delta1 = dual_mass_norm(
        vectors["medium"] - vectors["coarse"],
        metric,
    )
    delta2 = dual_mass_norm(
        vectors["fine"] - vectors["medium"],
        metric,
    )
    floor = floating_plateau_floor(
        list(vectors.values()),
        metric,
        factor=floating_factor,
    )
    plateau = max(delta1, delta2) <= floor
    if plateau:
        ratio = np.nan if delta1 == 0.0 else delta2 / delta1
        allowance = delta2
        passed = True
    elif delta1 <= floor:
        ratio = np.inf
        allowance = np.inf
        passed = False
    else:
        ratio = delta2 / delta1
        passed = bool(ratio < 1.0)
        allowance = (
            delta2 / (1.0 - ratio)
            if passed
            else np.inf
        )
    return QuadratureTailReport(
        coarse_medium_change=delta1,
        medium_fine_change=delta2,
        contraction_ratio=float(ratio),
        floating_floor=floor,
        plateau=plateau,
        allowance=float(allowance),
        passed=passed,
    )


def adjacent_mixed_cube(
    values: Mapping[str, Any],
    mass: Any,
) -> MixedCubeReport:
    """Return all non-additive differences of a complete adjacent cube.

    Keys use bit order ``epsilon,timestep,quadrature`` and must be exactly
    ``000`` through ``111``.
    """

    expected = {f"{index:03b}" for index in range(8)}
    if set(values) != expected:
        raise ReachablePressureUncertaintyError(
            "mixed cube requires exactly keys 000 through 111"
        )
    vectors = _common_vectors(
        **{f"v_{key}": value for key, value in values.items()}
    )
    cube = {key: vectors[f"v_{key}"] for key in expected}
    metric = _mass_matrix(mass, len(cube["000"]))
    epsilon_timestep = (
        cube["110"] - cube["100"] - cube["010"] + cube["000"]
    )
    epsilon_quadrature = (
        cube["101"] - cube["100"] - cube["001"] + cube["000"]
    )
    timestep_quadrature = (
        cube["011"] - cube["010"] - cube["001"] + cube["000"]
    )
    epsilon_timestep_quadrature = (
        cube["111"]
        - cube["110"]
        - cube["101"]
        - cube["011"]
        + cube["100"]
        + cube["010"]
        + cube["001"]
        - cube["000"]
    )
    components = {
        "epsilon_timestep": dual_mass_norm(
            epsilon_timestep,
            metric,
        ),
        "epsilon_quadrature": dual_mass_norm(
            epsilon_quadrature,
            metric,
        ),
        "timestep_quadrature": dual_mass_norm(
            timestep_quadrature,
            metric,
        ),
        "epsilon_timestep_quadrature": dual_mass_norm(
            epsilon_timestep_quadrature,
            metric,
        ),
    }
    return MixedCubeReport(
        epsilon_timestep=epsilon_timestep,
        epsilon_quadrature=epsilon_quadrature,
        timestep_quadrature=timestep_quadrature,
        epsilon_timestep_quadrature=epsilon_timestep_quadrature,
        component_norms=components,
        allowance=float(sum(components.values())),
    )


def resolved_norm_interval(
    observation: Any,
    mass: Any,
    uncertainty: float,
) -> ResolvedInterval:
    """Return the non-negative norm interval after a triangle bound."""

    vector = _finite_vector("observation", observation)
    metric = _mass_matrix(mass, len(vector))
    error = float(uncertainty)
    if not np.isfinite(error) or error < 0.0:
        raise ReachablePressureUncertaintyError(
            "uncertainty must be finite and non-negative"
        )
    norm = dual_mass_norm(vector, metric)
    return ResolvedInterval(
        observed_norm=norm,
        uncertainty=error,
        lower=max(0.0, norm - error),
        upper=norm + error,
    )
