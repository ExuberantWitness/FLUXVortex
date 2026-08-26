"""Independent total-Lagrangian Q16 continuum element oracle.

This module supplies the unprojected three-dimensional continuum baseline used
to verify CUDA residual, consistent-mass and tangent-vector operators.  It uses
the fixed Q16 position/director interpolation and 6x6x3 quadrature.  Locking
control is deliberately not hidden here: ANS/EAS projection remains a separate
validation gate before thin-shell production claims.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .q16_ancf_shell import (
    Q16_DOF_PER_ELEMENT,
    Q16_NODE_COUNT,
    Q16ReferenceElement,
    q16_shape,
)

_FLOAT64 = np.dtype(np.float64)
Q16_IN_PLANE_QUADRATURE_ORDER = 6
Q16_THICKNESS_QUADRATURE_ORDER = 3
Q16_QUADRATURE_POINT_COUNT = (
    Q16_IN_PLANE_QUADRATURE_ORDER**2 * Q16_THICKNESS_QUADRATURE_ORDER
)


def _readonly(value: np.ndarray) -> np.ndarray:
    contiguous = np.ascontiguousarray(value, dtype=np.float64)
    frozen = np.frombuffer(contiguous.tobytes(order="C"), dtype=np.float64)
    return frozen.reshape(contiguous.shape)


def _positive_scalar(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise TypeError(f"{name} must be a real scalar")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _exact_state(name: str, value: np.ndarray) -> np.ndarray:
    if type(value) is not np.ndarray:
        raise TypeError(f"{name} must be an exact numpy.ndarray")
    if value.dtype != _FLOAT64:
        raise TypeError(f"{name} must use float64")
    if value.shape != (Q16_DOF_PER_ELEMENT,):
        raise ValueError(f"{name} must have shape ({Q16_DOF_PER_ELEMENT},)")
    if not value.flags.c_contiguous:
        raise ValueError(f"{name} must be C-contiguous")
    if not bool(np.isfinite(value).all()):
        raise ValueError(f"{name} must contain only finite values")
    return value.reshape(Q16_NODE_COUNT, 6)


@dataclass(frozen=True, slots=True, eq=False)
class Q16QuadratureData:
    shape_values: np.ndarray
    zeta: np.ndarray
    position_gradients: np.ndarray
    director_gradients: np.ndarray
    reference_weights: np.ndarray

    @property
    def point_count(self) -> int:
        return int(self.reference_weights.size)


def make_q16_quadrature(reference: Q16ReferenceElement) -> Q16QuadratureData:
    """Precompute physical reference gradients for fixed 6x6x3 quadrature."""

    if type(reference) is not Q16ReferenceElement:
        raise TypeError("reference must be an exact Q16ReferenceElement")
    plane_points, plane_weights = np.polynomial.legendre.leggauss(
        Q16_IN_PLANE_QUADRATURE_ORDER
    )
    thickness_points, thickness_weights = np.polynomial.legendre.leggauss(
        Q16_THICKNESS_QUADRATURE_ORDER
    )
    shapes = np.empty((Q16_QUADRATURE_POINT_COUNT, 16), dtype=np.float64)
    zeta_values = np.empty(Q16_QUADRATURE_POINT_COUNT, dtype=np.float64)
    position_gradients = np.empty((Q16_QUADRATURE_POINT_COUNT, 16, 3), dtype=np.float64)
    director_gradients = np.empty_like(position_gradients)
    weights = np.empty(Q16_QUADRATURE_POINT_COUNT, dtype=np.float64)
    point = 0
    for xi, wx in zip(plane_points, plane_weights, strict=True):
        for eta, wy in zip(plane_points, plane_weights, strict=True):
            shape, dxi, deta = q16_shape(float(xi), float(eta))
            for zeta, wz in zip(thickness_points, thickness_weights, strict=True):
                bases = reference.covariant_bases(
                    reference.reference_state,
                    float(xi),
                    float(eta),
                    float(zeta),
                )
                jacobian = np.column_stack(bases)
                determinant = float(np.linalg.det(jacobian))
                if not math.isfinite(determinant) or determinant <= 0.0:
                    raise ValueError("reference quadrature Jacobian is non-positive")
                inverse_transpose = np.linalg.inv(jacobian).T
                shapes[point] = shape
                zeta_values[point] = float(zeta)
                for node in range(Q16_NODE_COUNT):
                    position_gradients[point, node] = inverse_transpose @ np.array(
                        [dxi[node], deta[node], 0.0], dtype=np.float64
                    )
                    director_gradients[point, node] = inverse_transpose @ np.array(
                        [
                            float(zeta) * dxi[node],
                            float(zeta) * deta[node],
                            shape[node],
                        ],
                        dtype=np.float64,
                    )
                weights[point] = float(wx) * float(wy) * float(wz) * determinant
                point += 1
    if point != Q16_QUADRATURE_POINT_COUNT:
        raise AssertionError("fixed Q16 quadrature count drifted")
    return Q16QuadratureData(
        shape_values=_readonly(shapes),
        zeta=_readonly(zeta_values),
        position_gradients=_readonly(position_gradients),
        director_gradients=_readonly(director_gradients),
        reference_weights=_readonly(weights),
    )


@dataclass(frozen=True, slots=True, init=False, eq=False)
class Q16ContinuumElement:
    """Fixed Q16 StVK continuum baseline with analytic residual and Jv."""

    reference: Q16ReferenceElement
    young_modulus: float
    poisson_ratio: float
    density: float
    lame_lambda: float
    lame_mu: float
    quadrature: Q16QuadratureData
    reference_volume: float

    def __init__(
        self,
        reference: Q16ReferenceElement,
        *,
        young_modulus: float,
        poisson_ratio: float,
        density: float,
    ) -> None:
        if type(reference) is not Q16ReferenceElement:
            raise TypeError("reference must be an exact Q16ReferenceElement")
        young = _positive_scalar("young_modulus", young_modulus)
        rho = _positive_scalar("density", density)
        if isinstance(poisson_ratio, bool) or not isinstance(
            poisson_ratio, (int, float, np.integer, np.floating)
        ):
            raise TypeError("poisson_ratio must be a real scalar")
        nu = float(poisson_ratio)
        if not math.isfinite(nu) or not (-1.0 < nu < 0.5):
            raise ValueError("poisson_ratio must be finite and lie in (-1, 0.5)")
        lame_lambda = young * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
        lame_mu = young / (2.0 * (1.0 + nu))
        quadrature = make_q16_quadrature(reference)
        object.__setattr__(self, "reference", reference)
        object.__setattr__(self, "young_modulus", young)
        object.__setattr__(self, "poisson_ratio", nu)
        object.__setattr__(self, "density", rho)
        object.__setattr__(self, "lame_lambda", lame_lambda)
        object.__setattr__(self, "lame_mu", lame_mu)
        object.__setattr__(self, "quadrature", quadrature)
        object.__setattr__(
            self, "reference_volume", float(np.sum(quadrature.reference_weights))
        )

    def _deformation_gradient(self, rows: np.ndarray, point: int) -> np.ndarray:
        return (
            rows[:, :3].T @ self.quadrature.position_gradients[point]
            + rows[:, 3:].T @ self.quadrature.director_gradients[point]
        )

    def _stress(
        self, deformation_gradient: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        identity = np.eye(3, dtype=np.float64)
        strain = 0.5 * (deformation_gradient.T @ deformation_gradient - identity)
        stress = (
            self.lame_lambda * float(np.trace(strain)) * identity
            + 2.0 * self.lame_mu * strain
        )
        return strain, stress

    def strain_energy(self, state: np.ndarray) -> float:
        rows = _exact_state("state", state)
        energy = 0.0
        for point in range(self.quadrature.point_count):
            deformation_gradient = self._deformation_gradient(rows, point)
            strain, _ = self._stress(deformation_gradient)
            density = 0.5 * self.lame_lambda * float(np.trace(strain)) ** 2
            density += self.lame_mu * float(np.sum(strain * strain))
            energy += self.quadrature.reference_weights[point] * density
        if not math.isfinite(energy):
            raise FloatingPointError("Q16 strain energy became non-finite")
        return float(energy)

    def internal_force(self, state: np.ndarray) -> np.ndarray:
        rows = _exact_state("state", state)
        force = np.zeros((Q16_NODE_COUNT, 6), dtype=np.float64)
        for point in range(self.quadrature.point_count):
            deformation_gradient = self._deformation_gradient(rows, point)
            _, second_piola = self._stress(deformation_gradient)
            first_piola = deformation_gradient @ second_piola
            weight = self.quadrature.reference_weights[point]
            for node in range(Q16_NODE_COUNT):
                force[node, :3] += weight * (
                    first_piola @ self.quadrature.position_gradients[point, node]
                )
                force[node, 3:] += weight * (
                    first_piola @ self.quadrature.director_gradients[point, node]
                )
        if not bool(np.isfinite(force).all()):
            raise FloatingPointError("Q16 internal force became non-finite")
        return _readonly(force.reshape(Q16_DOF_PER_ELEMENT))

    def tangent_action(self, state: np.ndarray, direction: np.ndarray) -> np.ndarray:
        rows = _exact_state("state", state)
        delta_rows = _exact_state("direction", direction)
        action = np.zeros((Q16_NODE_COUNT, 6), dtype=np.float64)
        identity = np.eye(3, dtype=np.float64)
        for point in range(self.quadrature.point_count):
            deformation_gradient = self._deformation_gradient(rows, point)
            delta_gradient = self._deformation_gradient(delta_rows, point)
            _, second_piola = self._stress(deformation_gradient)
            delta_strain = 0.5 * (
                delta_gradient.T @ deformation_gradient
                + deformation_gradient.T @ delta_gradient
            )
            delta_second_piola = (
                self.lame_lambda * float(np.trace(delta_strain)) * identity
                + 2.0 * self.lame_mu * delta_strain
            )
            delta_first_piola = (
                delta_gradient @ second_piola
                + deformation_gradient @ delta_second_piola
            )
            weight = self.quadrature.reference_weights[point]
            for node in range(Q16_NODE_COUNT):
                action[node, :3] += weight * (
                    delta_first_piola @ self.quadrature.position_gradients[point, node]
                )
                action[node, 3:] += weight * (
                    delta_first_piola @ self.quadrature.director_gradients[point, node]
                )
        if not bool(np.isfinite(action).all()):
            raise FloatingPointError("Q16 tangent action became non-finite")
        return _readonly(action.reshape(Q16_DOF_PER_ELEMENT))

    def mass_action(self, acceleration: np.ndarray) -> np.ndarray:
        rows = _exact_state("acceleration", acceleration)
        action = np.zeros((Q16_NODE_COUNT, 6), dtype=np.float64)
        for point in range(self.quadrature.point_count):
            shape = self.quadrature.shape_values[point]
            zeta = self.quadrature.zeta[point]
            value = shape @ (rows[:, :3] + zeta * rows[:, 3:])
            factor = self.density * self.quadrature.reference_weights[point]
            for node in range(Q16_NODE_COUNT):
                action[node, :3] += factor * shape[node] * value
                action[node, 3:] += factor * zeta * shape[node] * value
        if not bool(np.isfinite(action).all()):
            raise FloatingPointError("Q16 mass action became non-finite")
        return _readonly(action.reshape(Q16_DOF_PER_ELEMENT))


__all__ = [
    "Q16ContinuumElement",
    "Q16QuadratureData",
    "Q16_IN_PLANE_QUADRATURE_ORDER",
    "Q16_QUADRATURE_POINT_COUNT",
    "Q16_THICKNESS_QUADRATURE_ORDER",
    "make_q16_quadrature",
]
