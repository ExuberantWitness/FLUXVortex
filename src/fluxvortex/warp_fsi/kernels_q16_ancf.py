"""Matrix-free CUDA float64 operators for the fixed Q16 continuum baseline."""

from __future__ import annotations

from typing import Any

import numpy as np
import warp as wp

from fluxvortex.q16_ancf_continuum import (
    Q16ContinuumElement,
    Q16_QUADRATURE_POINT_COUNT,
)
from fluxvortex.q16_ancf_shell import Q16_DOF_PER_ELEMENT, Q16_NODE_COUNT

from . import config

DTYPE = config.DTYPE
VEC3 = config.VEC3
MAT33 = config.MAT33


@wp.func
def _outer3(left: VEC3, right: VEC3):
    return MAT33(
        left[0] * right[0],
        left[0] * right[1],
        left[0] * right[2],
        left[1] * right[0],
        left[1] * right[1],
        left[1] * right[2],
        left[2] * right[0],
        left[2] * right[1],
        left[2] * right[2],
    )


@wp.func
def _identity3():
    return MAT33(
        DTYPE(1.0),
        DTYPE(0.0),
        DTYPE(0.0),
        DTYPE(0.0),
        DTYPE(1.0),
        DTYPE(0.0),
        DTYPE(0.0),
        DTYPE(0.0),
        DTYPE(1.0),
    )


@wp.kernel
def _q16_scalar3_nonfinite_kernel(
    value: wp.array(dtype=DTYPE, ndim=3),
    flag: wp.array(dtype=wp.int32, ndim=1),
):
    batch, element, coordinate = wp.tid()
    if not wp.isfinite(value[batch, element, coordinate]):
        wp.atomic_max(flag, 0, 1)


@wp.kernel
def q16_stvk_response_kernel(
    state: wp.array(dtype=DTYPE, ndim=3),
    position_gradients: wp.array(dtype=VEC3, ndim=3),
    director_gradients: wp.array(dtype=VEC3, ndim=3),
    lame_lambda: wp.array(dtype=DTYPE, ndim=1),
    lame_mu: wp.array(dtype=DTYPE, ndim=1),
    deformation_gradient: wp.array(dtype=MAT33, ndim=3),
    second_piola: wp.array(dtype=MAT33, ndim=3),
    first_piola: wp.array(dtype=MAT33, ndim=3),
    invalid: wp.array(dtype=wp.int32, ndim=1),
):
    batch, element, point = wp.tid()
    gradient = MAT33(
        DTYPE(0.0),
        DTYPE(0.0),
        DTYPE(0.0),
        DTYPE(0.0),
        DTYPE(0.0),
        DTYPE(0.0),
        DTYPE(0.0),
        DTYPE(0.0),
        DTYPE(0.0),
    )
    for node in range(Q16_NODE_COUNT):
        base = node * 6
        position = VEC3(
            state[batch, element, base],
            state[batch, element, base + 1],
            state[batch, element, base + 2],
        )
        director = VEC3(
            state[batch, element, base + 3],
            state[batch, element, base + 4],
            state[batch, element, base + 5],
        )
        gradient += _outer3(position, position_gradients[element, point, node])
        gradient += _outer3(director, director_gradients[element, point, node])
    determinant = wp.determinant(gradient)
    if not wp.isfinite(determinant) or determinant <= DTYPE(0.0):
        wp.atomic_max(invalid, 0, 1)
    identity = _identity3()
    strain = DTYPE(0.5) * (wp.transpose(gradient) * gradient - identity)
    trace = strain[0, 0] + strain[1, 1] + strain[2, 2]
    stress = (
        lame_lambda[element] * trace * identity + DTYPE(2.0) * lame_mu[element] * strain
    )
    deformation_gradient[batch, element, point] = gradient
    second_piola[batch, element, point] = stress
    first_piola[batch, element, point] = gradient * stress


@wp.kernel
def q16_stvk_tangent_response_kernel(
    direction: wp.array(dtype=DTYPE, ndim=3),
    position_gradients: wp.array(dtype=VEC3, ndim=3),
    director_gradients: wp.array(dtype=VEC3, ndim=3),
    lame_lambda: wp.array(dtype=DTYPE, ndim=1),
    lame_mu: wp.array(dtype=DTYPE, ndim=1),
    deformation_gradient: wp.array(dtype=MAT33, ndim=3),
    second_piola: wp.array(dtype=MAT33, ndim=3),
    delta_first_piola: wp.array(dtype=MAT33, ndim=3),
):
    batch, element, point = wp.tid()
    delta_gradient = MAT33(
        DTYPE(0.0),
        DTYPE(0.0),
        DTYPE(0.0),
        DTYPE(0.0),
        DTYPE(0.0),
        DTYPE(0.0),
        DTYPE(0.0),
        DTYPE(0.0),
        DTYPE(0.0),
    )
    for node in range(Q16_NODE_COUNT):
        base = node * 6
        delta_position = VEC3(
            direction[batch, element, base],
            direction[batch, element, base + 1],
            direction[batch, element, base + 2],
        )
        delta_director = VEC3(
            direction[batch, element, base + 3],
            direction[batch, element, base + 4],
            direction[batch, element, base + 5],
        )
        delta_gradient += _outer3(
            delta_position, position_gradients[element, point, node]
        )
        delta_gradient += _outer3(
            delta_director, director_gradients[element, point, node]
        )
    gradient = deformation_gradient[batch, element, point]
    delta_strain = DTYPE(0.5) * (
        wp.transpose(delta_gradient) * gradient
        + wp.transpose(gradient) * delta_gradient
    )
    trace = delta_strain[0, 0] + delta_strain[1, 1] + delta_strain[2, 2]
    delta_stress = lame_lambda[element] * trace * _identity3() + DTYPE(2.0) * (
        lame_mu[element] * delta_strain
    )
    delta_first_piola[batch, element, point] = (
        delta_gradient * second_piola[batch, element, point] + gradient * delta_stress
    )


@wp.kernel
def q16_piola_gather_kernel(
    piola: wp.array(dtype=MAT33, ndim=3),
    position_gradients: wp.array(dtype=VEC3, ndim=3),
    director_gradients: wp.array(dtype=VEC3, ndim=3),
    reference_weights: wp.array(dtype=DTYPE, ndim=2),
    output: wp.array(dtype=DTYPE, ndim=3),
):
    batch, element, node, coordinate = wp.tid()
    value = DTYPE(0.0)
    for point in range(Q16_QUADRATURE_POINT_COUNT):
        gradient = position_gradients[element, point, node]
        if coordinate >= 3:
            gradient = director_gradients[element, point, node]
        vector = piola[batch, element, point] * gradient
        value += reference_weights[element, point] * vector[coordinate % 3]
    output[batch, element, node * 6 + coordinate] = value


@wp.kernel
def q16_mass_interpolate_kernel(
    acceleration: wp.array(dtype=DTYPE, ndim=3),
    shape_values: wp.array(dtype=DTYPE, ndim=2),
    zeta: wp.array(dtype=DTYPE, ndim=1),
    value: wp.array(dtype=VEC3, ndim=3),
):
    batch, element, point = wp.tid()
    result = VEC3(DTYPE(0.0), DTYPE(0.0), DTYPE(0.0))
    for node in range(Q16_NODE_COUNT):
        base = node * 6
        weight = shape_values[point, node]
        director_weight = zeta[point] * weight
        result += VEC3(
            weight * acceleration[batch, element, base]
            + director_weight * acceleration[batch, element, base + 3],
            weight * acceleration[batch, element, base + 1]
            + director_weight * acceleration[batch, element, base + 4],
            weight * acceleration[batch, element, base + 2]
            + director_weight * acceleration[batch, element, base + 5],
        )
    value[batch, element, point] = result


@wp.kernel
def q16_mass_gather_kernel(
    value: wp.array(dtype=VEC3, ndim=3),
    shape_values: wp.array(dtype=DTYPE, ndim=2),
    zeta: wp.array(dtype=DTYPE, ndim=1),
    reference_weights: wp.array(dtype=DTYPE, ndim=2),
    density: wp.array(dtype=DTYPE, ndim=1),
    output: wp.array(dtype=DTYPE, ndim=3),
):
    batch, element, node, coordinate = wp.tid()
    result = DTYPE(0.0)
    for point in range(Q16_QUADRATURE_POINT_COUNT):
        weight = shape_values[point, node]
        if coordinate >= 3:
            weight *= zeta[point]
        result += (
            density[element]
            * reference_weights[element, point]
            * weight
            * value[batch, element, point][coordinate % 3]
        )
    output[batch, element, node * 6 + coordinate] = result


@wp.kernel
def q16_mass_diagonal_kernel(
    shape_values: wp.array(dtype=DTYPE, ndim=2),
    zeta: wp.array(dtype=DTYPE, ndim=1),
    reference_weights: wp.array(dtype=DTYPE, ndim=2),
    density: wp.array(dtype=DTYPE, ndim=1),
    diagonal: wp.array(dtype=DTYPE, ndim=3),
):
    batch, element, node, coordinate = wp.tid()
    result = DTYPE(0.0)
    for point in range(Q16_QUADRATURE_POINT_COUNT):
        interpolation = shape_values[point, node]
        if coordinate >= 3:
            interpolation *= zeta[point]
        result += (
            density[element]
            * reference_weights[element, point]
            * interpolation
            * interpolation
        )
    diagonal[batch, element, node * 6 + coordinate] = result


def _require_state(name: str, value: Any, *, device: str, element_count: int) -> None:
    if not isinstance(value, wp.array):
        raise TypeError(f"{name} must be a Warp array")
    if not value.device.is_cuda or value.device.alias != device:
        raise ValueError(f"{name} must reside on CUDA device {device}")
    if value.dtype != DTYPE:
        raise TypeError(f"{name} must use the frozen float64 Warp dtype")
    if value.ndim != 3 or value.shape[0] <= 0:
        raise ValueError(f"{name} must have three dimensions and a positive batch")
    if value.shape[1:] != (element_count, Q16_DOF_PER_ELEMENT):
        raise ValueError(
            f"{name} must end with shape ({element_count}, {Q16_DOF_PER_ELEMENT})"
        )


def _assert_finite(name: str, value: wp.array, *, device: str) -> None:
    flag = wp.zeros(1, dtype=wp.int32, device=device)
    wp.launch(
        _q16_scalar3_nonfinite_kernel,
        dim=value.shape,
        inputs=[value],
        outputs=[flag],
        device=device,
    )
    wp.synchronize_device(device)
    if int(flag.numpy()[0]) != 0:
        raise FloatingPointError(f"{name} contains non-finite values")


class Q16CudaContinuumOperator:
    """Deterministic batched CUDA residual, mass action and analytic Jv."""

    __slots__ = (
        "_density",
        "_director_gradients",
        "_lame_lambda",
        "_lame_mu",
        "_position_gradients",
        "_reference_weights",
        "_shape_values",
        "_zeta",
        "device",
        "element_count",
    )

    def __init__(
        self, elements: tuple[Q16ContinuumElement, ...], *, device: str
    ) -> None:
        if type(elements) is not tuple or not elements:
            raise ValueError("elements must be a non-empty exact tuple")
        if any(type(element) is not Q16ContinuumElement for element in elements):
            raise TypeError("every element must be an exact Q16ContinuumElement")
        if config.dtype_name() != "float64" or DTYPE != wp.float64:
            raise RuntimeError("Q16 continuum production operator requires float64")
        selected = wp.get_device(device)
        if not selected.is_cuda:
            raise ValueError("Q16 continuum production operator requires CUDA")
        self.device = selected.alias
        self.element_count = len(elements)
        position_gradients = np.ascontiguousarray(
            np.stack([element.quadrature.position_gradients for element in elements])
        )
        director_gradients = np.ascontiguousarray(
            np.stack([element.quadrature.director_gradients for element in elements])
        )
        weights = np.ascontiguousarray(
            np.stack([element.quadrature.reference_weights for element in elements])
        )
        first_quadrature = elements[0].quadrature
        self._position_gradients = wp.array(
            position_gradients, dtype=VEC3, device=self.device
        )
        self._director_gradients = wp.array(
            director_gradients, dtype=VEC3, device=self.device
        )
        self._reference_weights = wp.array(weights, dtype=DTYPE, device=self.device)
        self._shape_values = wp.array(
            np.ascontiguousarray(first_quadrature.shape_values),
            dtype=DTYPE,
            device=self.device,
        )
        self._zeta = wp.array(
            np.ascontiguousarray(first_quadrature.zeta),
            dtype=DTYPE,
            device=self.device,
        )
        self._lame_lambda = wp.array(
            np.asarray([element.lame_lambda for element in elements]),
            dtype=DTYPE,
            device=self.device,
        )
        self._lame_mu = wp.array(
            np.asarray([element.lame_mu for element in elements]),
            dtype=DTYPE,
            device=self.device,
        )
        self._density = wp.array(
            np.asarray([element.density for element in elements]),
            dtype=DTYPE,
            device=self.device,
        )

    def _validate(self, name: str, value: Any) -> None:
        _require_state(
            name, value, device=self.device, element_count=self.element_count
        )
        _assert_finite(name, value, device=self.device)

    def _response(self, state: wp.array) -> tuple[wp.array, wp.array, wp.array]:
        batch = state.shape[0]
        shape = (batch, self.element_count, Q16_QUADRATURE_POINT_COUNT)
        deformation_gradient = wp.zeros(shape, dtype=MAT33, device=self.device)
        second_piola = wp.zeros(shape, dtype=MAT33, device=self.device)
        first_piola = wp.zeros(shape, dtype=MAT33, device=self.device)
        invalid = wp.zeros(1, dtype=wp.int32, device=self.device)
        wp.launch(
            q16_stvk_response_kernel,
            dim=shape,
            inputs=[
                state,
                self._position_gradients,
                self._director_gradients,
                self._lame_lambda,
                self._lame_mu,
            ],
            outputs=[
                deformation_gradient,
                second_piola,
                first_piola,
                invalid,
            ],
            device=self.device,
        )
        wp.synchronize_device(self.device)
        if int(invalid.numpy()[0]) != 0:
            raise FloatingPointError(
                "Q16 deformation gradient is non-finite or orientation reversing"
            )
        return deformation_gradient, second_piola, first_piola

    def _gather(self, piola: wp.array, batch: int) -> wp.array:
        output = wp.zeros(
            (batch, self.element_count, Q16_DOF_PER_ELEMENT),
            dtype=DTYPE,
            device=self.device,
        )
        wp.launch(
            q16_piola_gather_kernel,
            dim=(batch, self.element_count, Q16_NODE_COUNT, 6),
            inputs=[
                piola,
                self._position_gradients,
                self._director_gradients,
                self._reference_weights,
            ],
            outputs=[output],
            device=self.device,
        )
        _assert_finite("Q16 operator output", output, device=self.device)
        return output

    def internal_force(self, state: Any) -> wp.array:
        self._validate("state", state)
        _, _, first_piola = self._response(state)
        return self._gather(first_piola, state.shape[0])

    def tangent_action(self, state: Any, direction: Any) -> wp.array:
        self._validate("state", state)
        self._validate("direction", direction)
        if direction.shape != state.shape:
            raise ValueError("direction shape differs from state")
        deformation_gradient, second_piola, _ = self._response(state)
        delta_first_piola = wp.zeros(
            (state.shape[0], self.element_count, Q16_QUADRATURE_POINT_COUNT),
            dtype=MAT33,
            device=self.device,
        )
        wp.launch(
            q16_stvk_tangent_response_kernel,
            dim=(state.shape[0], self.element_count, Q16_QUADRATURE_POINT_COUNT),
            inputs=[
                direction,
                self._position_gradients,
                self._director_gradients,
                self._lame_lambda,
                self._lame_mu,
                deformation_gradient,
                second_piola,
            ],
            outputs=[delta_first_piola],
            device=self.device,
        )
        return self._gather(delta_first_piola, state.shape[0])

    def mass_action(self, acceleration: Any) -> wp.array:
        self._validate("acceleration", acceleration)
        values = wp.zeros(
            (
                acceleration.shape[0],
                self.element_count,
                Q16_QUADRATURE_POINT_COUNT,
            ),
            dtype=VEC3,
            device=self.device,
        )
        wp.launch(
            q16_mass_interpolate_kernel,
            dim=values.shape,
            inputs=[acceleration, self._shape_values, self._zeta],
            outputs=[values],
            device=self.device,
        )
        output = wp.zeros(
            (acceleration.shape[0], self.element_count, Q16_DOF_PER_ELEMENT),
            dtype=DTYPE,
            device=self.device,
        )
        wp.launch(
            q16_mass_gather_kernel,
            dim=(acceleration.shape[0], self.element_count, Q16_NODE_COUNT, 6),
            inputs=[
                values,
                self._shape_values,
                self._zeta,
                self._reference_weights,
                self._density,
            ],
            outputs=[output],
            device=self.device,
        )
        _assert_finite("Q16 mass output", output, device=self.device)
        return output

    def _mass_diagonal_prechecked(self, batch: int) -> wp.array:
        diagonal = wp.zeros(
            (batch, self.element_count, Q16_DOF_PER_ELEMENT),
            dtype=DTYPE,
            device=self.device,
        )
        wp.launch(
            q16_mass_diagonal_kernel,
            dim=(batch, self.element_count, Q16_NODE_COUNT, 6),
            inputs=[
                self._shape_values,
                self._zeta,
                self._reference_weights,
                self._density,
            ],
            outputs=[diagonal],
            device=self.device,
        )
        _assert_finite("Q16 mass diagonal", diagonal, device=self.device)
        return diagonal


__all__ = [
    "Q16CudaContinuumOperator",
    "q16_mass_gather_kernel",
    "q16_mass_diagonal_kernel",
    "q16_mass_interpolate_kernel",
    "q16_piola_gather_kernel",
    "q16_stvk_response_kernel",
    "q16_stvk_tangent_response_kernel",
]
