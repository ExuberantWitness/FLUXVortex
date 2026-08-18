"""Source-faithful direct Gaussian-erf interactions for FluxV v5h.

This module is intentionally isolated from the legacy particle implementation.
It contains no clipping, NaN repair, FMM approximation, viscosity, SFS model, or
aerodynamic force path.  The equations and tensor layout match FLOWVPM.jl
v4.0.4 at commit ``4f433fb09f6baad25db65c9905e0d9cbb09663ce``.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import pi, sqrt

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import erf

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class DirectField:
    r"""Velocity and Jacobian evaluated at target points.

    ``jacobian[:, i, j]`` is :math:`\partial u_i/\partial x_j`.
    """

    velocity: FloatArray
    jacobian: FloatArray


def _finite_array(name: str, value: ArrayLike, *, ndim: int) -> FloatArray:
    original = np.asarray(value)
    if original.dtype.kind not in "iuf":
        raise ValueError(f"{name} must use a real numeric dtype")
    array = np.asarray(original, dtype=np.float64)
    if array.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions; got {array.ndim}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return np.ascontiguousarray(array)


def pack_julia_column_major(jacobian: ArrayLike) -> FloatArray:
    """Pack ``(n, 3, 3)`` Jacobians as Julia's column-major nine-vector."""

    array = _finite_array("jacobian", jacobian, ndim=3)
    if array.shape[1:] != (3, 3):
        raise ValueError("jacobian must have shape (n, 3, 3)")
    return np.ascontiguousarray(array.transpose(0, 2, 1).reshape(array.shape[0], 9))


def unpack_julia_column_major(flat_jacobian: ArrayLike) -> FloatArray:
    """Unpack ``j11,j21,j31,...,j33`` into ``J[n,u_component,x]``."""

    array = _finite_array("flat_jacobian", flat_jacobian, ndim=2)
    if array.shape[1:] != (9,):
        raise ValueError("flat_jacobian must have shape (n, 9)")
    return np.ascontiguousarray(array.reshape(array.shape[0], 3, 3).transpose(0, 2, 1))


def validate_particle_state(
    positions: ArrayLike,
    gamma: ArrayLike,
    sigma: ArrayLike,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Return validated Float64 particle arrays without changing their values."""

    positions_array = _finite_array("positions", positions, ndim=2)
    gamma_array = _finite_array("gamma", gamma, ndim=2)
    sigma_array = _finite_array("sigma", sigma, ndim=1)
    if positions_array.shape[1:] != (3,):
        raise ValueError("positions must have shape (n, 3)")
    if gamma_array.shape != positions_array.shape:
        raise ValueError("gamma must have the same (n, 3) shape as positions")
    if sigma_array.shape != (positions_array.shape[0],):
        raise ValueError("sigma must have shape (n,)")
    if np.any(sigma_array <= 0.0):
        raise ValueError("sigma must be strictly positive")
    return positions_array, gamma_array, sigma_array


def direct_gaussian_erf_velocity_jacobian(
    source_positions: ArrayLike,
    source_gamma: ArrayLike,
    source_sigma: ArrayLike,
    *,
    target_positions: ArrayLike | None = None,
) -> DirectField:
    """Evaluate direct Gaussian-erf particle velocity and analytic Jacobian.

    Interactions at exactly coincident coordinates are omitted, matching
    FLOWVPM's ``if !iszero(r2)`` direct-kernel contract.  If
    ``target_positions`` is omitted, this gives the particle field's self
    interaction with each particle's own contribution excluded.
    """

    positions, gamma, sigma = validate_particle_state(
        source_positions,
        source_gamma,
        source_sigma,
    )
    if target_positions is None:
        targets = positions
    else:
        targets = _finite_array("target_positions", target_positions, ndim=2)
        if targets.shape[1:] != (3,):
            raise ValueError("target_positions must have shape (m, 3)")

    velocity = np.zeros((targets.shape[0], 3), dtype=np.float64)
    jacobian = np.zeros((targets.shape[0], 3, 3), dtype=np.float64)
    minus_one_over_four_pi = -1.0 / (4.0 * pi)
    sqrt_two_over_pi = sqrt(2.0 / pi)
    sqrt_two = sqrt(2.0)

    # FLOWVPM accumulates source outer / target inner.  Preserving that order
    # minimizes floating-point differences in the oracle parity test.
    for source_index in range(positions.shape[0]):
        dx_all = targets - positions[source_index]
        r2_all = np.einsum("ij,ij->i", dx_all, dx_all)
        active = r2_all != 0.0
        if not np.any(active):
            continue

        dx = dx_all[active]
        r2 = r2_all[active]
        radius = np.sqrt(r2)
        radius_over_sigma = radius / sigma[source_index]
        exponential = np.exp(-0.5 * radius_over_sigma**2)
        auxiliary = sqrt_two_over_pi * radius_over_sigma * exponential
        regularization = erf(radius_over_sigma / sqrt_two) - auxiliary
        regularization_derivative = radius_over_sigma * auxiliary

        radius_cubed_inverse = 1.0 / (r2 * radius)
        cross = np.cross(dx, gamma[source_index])
        kernel_cross_gamma = (
            minus_one_over_four_pi * radius_cubed_inverse[:, None] * cross
        )
        velocity[active] += regularization[:, None] * kernel_cross_gamma

        gradient_radial = (
            regularization_derivative / (sigma[source_index] * radius)
            - 3.0 * regularization / r2
        )
        kronecker = minus_one_over_four_pi * regularization * radius_cubed_inverse
        source_strength = gamma[source_index]
        contribution = np.empty((dx.shape[0], 3, 3), dtype=np.float64)

        contribution[:, 0, 0] = gradient_radial * kernel_cross_gamma[:, 0] * dx[:, 0]
        contribution[:, 1, 0] = (
            gradient_radial * kernel_cross_gamma[:, 1] * dx[:, 0]
            - kronecker * source_strength[2]
        )
        contribution[:, 2, 0] = (
            gradient_radial * kernel_cross_gamma[:, 2] * dx[:, 0]
            + kronecker * source_strength[1]
        )

        contribution[:, 0, 1] = (
            gradient_radial * kernel_cross_gamma[:, 0] * dx[:, 1]
            + kronecker * source_strength[2]
        )
        contribution[:, 1, 1] = gradient_radial * kernel_cross_gamma[:, 1] * dx[:, 1]
        contribution[:, 2, 1] = (
            gradient_radial * kernel_cross_gamma[:, 2] * dx[:, 1]
            - kronecker * source_strength[0]
        )

        contribution[:, 0, 2] = (
            gradient_radial * kernel_cross_gamma[:, 0] * dx[:, 2]
            - kronecker * source_strength[1]
        )
        contribution[:, 1, 2] = (
            gradient_radial * kernel_cross_gamma[:, 1] * dx[:, 2]
            + kronecker * source_strength[0]
        )
        contribution[:, 2, 2] = gradient_radial * kernel_cross_gamma[:, 2] * dx[:, 2]
        jacobian[active] += contribution

    if not np.all(np.isfinite(velocity)) or not np.all(np.isfinite(jacobian)):
        raise FloatingPointError(
            "direct Gaussian-erf evaluation produced non-finite values"
        )
    return DirectField(velocity=velocity, jacobian=jacobian)


__all__ = [
    "DirectField",
    "direct_gaussian_erf_velocity_jacobian",
    "pack_julia_column_major",
    "unpack_julia_column_major",
    "validate_particle_state",
]
