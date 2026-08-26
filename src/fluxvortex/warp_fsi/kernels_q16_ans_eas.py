"""Parallel CUDA float64 MITC16+ANS/EAS operators for fixed Q16 elements.

The implementation evaluates every tying sample once, projects the compatible
strain and strain-displacement operator in separate kernels, condenses the
single enhanced thickness parameter per element, and performs deterministic
quadrature gathers for residual and analytic Jv.  No floating-point atomic
scatter or host numerical fallback is used.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import warp as wp
from warp.types import vector

from fluxvortex.q16_ancf_shell import (
    Q16_DOF_PER_ELEMENT,
    Q16_NODE_COUNT,
    Q16_PARAMETRIC_NODES,
    q16_shape,
)
from fluxvortex.q16_ans_eas_continuum import (
    Q16MITC16EASContinuumElement,
    Q16_EAS_QUADRATURE_POINT_COUNT,
)
from fluxvortex.q16_mitc16_projection import (
    MITC16_TYING_POINTS_3,
    MITC16_TYING_POINTS_4,
)

from . import config

DTYPE = config.DTYPE
VEC3 = config.VEC3
VEC6 = vector(length=6, dtype=DTYPE)
MAX_PROJECTION_SAMPLES = 16
PROJECTION_COMPONENT_COUNT = 6
PROJECTION_SAMPLE_SLOT_COUNT = PROJECTION_COMPONENT_COUNT * MAX_PROJECTION_SAMPLES
PROJECTED_B_SLOT_COUNT = PROJECTION_COMPONENT_COUNT * Q16_DOF_PER_ELEMENT


def _lagrange_values(nodes: np.ndarray, coordinate: float) -> np.ndarray:
    values = np.empty(nodes.size, dtype=np.float64)
    for active in range(nodes.size):
        value = 1.0
        for other in range(nodes.size):
            if other != active:
                value *= (coordinate - nodes[other]) / (nodes[active] - nodes[other])
        values[active] = value
    return values


def _projection_stencil(
    coordinates: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    point_count = int(coordinates.shape[0])
    shape = np.zeros(
        (point_count, PROJECTION_SAMPLE_SLOT_COUNT, Q16_NODE_COUNT),
        dtype=np.float64,
    )
    dxi = np.zeros_like(shape)
    deta = np.zeros_like(shape)
    zeta = np.zeros((point_count, PROJECTION_SAMPLE_SLOT_COUNT), dtype=np.float64)
    weight = np.zeros_like(zeta)
    for point, (xi_query, eta_query, zeta_query) in enumerate(coordinates):
        xi_three = _lagrange_values(MITC16_TYING_POINTS_3, float(xi_query))
        eta_three = _lagrange_values(MITC16_TYING_POINTS_3, float(eta_query))
        xi_four = _lagrange_values(MITC16_TYING_POINTS_4, float(xi_query))
        eta_four = _lagrange_values(MITC16_TYING_POINTS_4, float(eta_query))
        samples: dict[int, list[tuple[float, float, float, float]]] = {
            0: [
                (float(xi), float(eta), float(zeta_query), float(wx * wy))
                for eta, wy in zip(MITC16_TYING_POINTS_4, eta_four, strict=True)
                for xi, wx in zip(MITC16_TYING_POINTS_3, xi_three, strict=True)
            ],
            1: [
                (float(xi), float(eta), float(zeta_query), float(wx * wy))
                for eta, wy in zip(MITC16_TYING_POINTS_3, eta_three, strict=True)
                for xi, wx in zip(MITC16_TYING_POINTS_4, xi_four, strict=True)
            ],
            3: [
                (float(xi), float(eta), float(zeta_query), float(wx * wy))
                for eta, wy in zip(MITC16_TYING_POINTS_3, eta_three, strict=True)
                for xi, wx in zip(MITC16_TYING_POINTS_3, xi_three, strict=True)
            ],
            4: [
                (float(xi), float(eta), 0.0, float(wx * wy))
                for eta, wy in zip(MITC16_TYING_POINTS_3, eta_three, strict=True)
                for xi, wx in zip(MITC16_TYING_POINTS_4, xi_four, strict=True)
            ],
            5: [
                (float(xi), float(eta), 0.0, float(wx * wy))
                for eta, wy in zip(MITC16_TYING_POINTS_4, eta_four, strict=True)
                for xi, wx in zip(MITC16_TYING_POINTS_3, xi_three, strict=True)
            ],
        }
        q16_values, _, _ = q16_shape(float(xi_query), float(eta_query))
        samples[2] = [
            (float(xi), float(eta), 0.0, float(node_weight))
            for (xi, eta), node_weight in zip(
                Q16_PARAMETRIC_NODES, q16_values, strict=True
            )
        ]
        for component in range(PROJECTION_COMPONENT_COUNT):
            for sample, (xi, eta, sample_zeta, sample_weight) in enumerate(
                samples[component]
            ):
                slot = component * MAX_PROJECTION_SAMPLES + sample
                values, derivatives_xi, derivatives_eta = q16_shape(xi, eta)
                shape[point, slot] = values
                dxi[point, slot] = derivatives_xi
                deta[point, slot] = derivatives_eta
                zeta[point, slot] = sample_zeta
                weight[point, slot] = sample_weight
    return tuple(
        np.ascontiguousarray(value) for value in (shape, dxi, deta, zeta, weight)
    )


def _measure(first: np.ndarray, second: np.ndarray, third: np.ndarray, component: int):
    if component == 0:
        return 0.5 * float(first @ first)
    if component == 1:
        return 0.5 * float(second @ second)
    if component == 2:
        return 0.5 * float(third @ third)
    if component == 3:
        return float(first @ second)
    if component == 4:
        return float(second @ third)
    return float(first @ third)


def _reference_measures(
    elements: tuple[Q16MITC16EASContinuumElement, ...],
    shape: np.ndarray,
    dxi: np.ndarray,
    deta: np.ndarray,
    zeta: np.ndarray,
    weight: np.ndarray,
) -> np.ndarray:
    output = np.zeros(
        (
            len(elements),
            Q16_EAS_QUADRATURE_POINT_COUNT,
            PROJECTION_SAMPLE_SLOT_COUNT,
        ),
        dtype=np.float64,
    )
    for element_index, element in enumerate(elements):
        rows = element.reference.reference_rows
        for point in range(Q16_EAS_QUADRATURE_POINT_COUNT):
            for slot in range(PROJECTION_SAMPLE_SLOT_COUNT):
                if weight[point, slot] == 0.0:
                    continue
                component = slot // MAX_PROJECTION_SAMPLES
                first = dxi[point, slot] @ rows[:, :3] + zeta[point, slot] * (
                    dxi[point, slot] @ rows[:, 3:]
                )
                second = deta[point, slot] @ rows[:, :3] + zeta[point, slot] * (
                    deta[point, slot] @ rows[:, 3:]
                )
                third = shape[point, slot] @ rows[:, 3:]
                output[element_index, point, slot] = _measure(
                    first, second, third, component
                )
    return np.ascontiguousarray(output)


def _query_stencil(coordinates: np.ndarray) -> tuple[np.ndarray, ...]:
    shape = np.empty((coordinates.shape[0], Q16_NODE_COUNT), dtype=np.float64)
    dxi = np.empty_like(shape)
    deta = np.empty_like(shape)
    zeta = np.ascontiguousarray(coordinates[:, 2], dtype=np.float64)
    for point, (xi, eta, _) in enumerate(coordinates):
        values, derivatives_xi, derivatives_eta = q16_shape(float(xi), float(eta))
        shape[point] = values
        dxi[point] = derivatives_xi
        deta[point] = derivatives_eta
    return tuple(np.ascontiguousarray(value) for value in (shape, dxi, deta, zeta))


@wp.func
def _strain_measure(first: VEC3, second: VEC3, third: VEC3, component: int):
    result = DTYPE(0.0)
    if component == 0:
        result = DTYPE(0.5) * wp.dot(first, first)
    elif component == 1:
        result = DTYPE(0.5) * wp.dot(second, second)
    elif component == 2:
        result = DTYPE(0.5) * wp.dot(third, third)
    elif component == 3:
        result = wp.dot(first, second)
    elif component == 4:
        result = wp.dot(second, third)
    else:
        result = wp.dot(first, third)
    return result


@wp.func
def _b_entry(
    first: VEC3,
    second: VEC3,
    third: VEC3,
    component: int,
    coordinate: int,
    first_coefficient: DTYPE,
    second_coefficient: DTYPE,
    third_coefficient: DTYPE,
):
    result = DTYPE(0.0)
    if component == 0:
        result = first_coefficient * first[coordinate]
    elif component == 1:
        result = second_coefficient * second[coordinate]
    elif component == 2:
        result = third_coefficient * third[coordinate]
    elif component == 3:
        result = (
            first_coefficient * second[coordinate]
            + second_coefficient * first[coordinate]
        )
    elif component == 4:
        result = (
            second_coefficient * third[coordinate]
            + third_coefficient * second[coordinate]
        )
    else:
        result = (
            first_coefficient * third[coordinate]
            + third_coefficient * first[coordinate]
        )
    return result


@wp.func
def _stress(strain: VEC6, lame_lambda: DTYPE, lame_mu: DTYPE):
    trace = strain[0] + strain[1] + strain[2]
    return VEC6(
        lame_lambda * trace + DTYPE(2.0) * lame_mu * strain[0],
        lame_lambda * trace + DTYPE(2.0) * lame_mu * strain[1],
        lame_lambda * trace + DTYPE(2.0) * lame_mu * strain[2],
        lame_mu * strain[3],
        lame_mu * strain[4],
        lame_mu * strain[5],
    )


@wp.func
def _load6(value: wp.array(dtype=DTYPE, ndim=4), i: int, j: int, k: int):
    return VEC6(
        value[i, j, k, 0],
        value[i, j, k, 1],
        value[i, j, k, 2],
        value[i, j, k, 3],
        value[i, j, k, 4],
        value[i, j, k, 5],
    )


@wp.func
def _load_mode(value: wp.array(dtype=DTYPE, ndim=3), i: int, j: int):
    return VEC6(
        value[i, j, 0],
        value[i, j, 1],
        value[i, j, 2],
        value[i, j, 3],
        value[i, j, 4],
        value[i, j, 5],
    )


@wp.func
def _dot6(first: VEC6, second: VEC6):
    result = DTYPE(0.0)
    for component in range(6):
        result += first[component] * second[component]
    return result


@wp.func
def _projected_unit_strain_entry(
    projected_b: wp.array(dtype=DTYPE, ndim=4),
    transform: wp.array(dtype=DTYPE, ndim=4),
    batch: int,
    element: int,
    point: int,
    row: int,
    dof: int,
):
    result = DTYPE(0.0)
    for component in range(PROJECTION_COMPONENT_COUNT):
        result += (
            transform[element, point, row, component]
            * projected_b[
                batch,
                element,
                point,
                component * Q16_DOF_PER_ELEMENT + dof,
            ]
        )
    return result


@wp.kernel
def q16_sample_basis_measure_kernel(
    state: wp.array(dtype=DTYPE, ndim=3),
    sample_shape: wp.array(dtype=DTYPE, ndim=3),
    sample_dxi: wp.array(dtype=DTYPE, ndim=3),
    sample_deta: wp.array(dtype=DTYPE, ndim=3),
    sample_zeta: wp.array(dtype=DTYPE, ndim=2),
    sample_weight: wp.array(dtype=DTYPE, ndim=2),
    reference_measure: wp.array(dtype=DTYPE, ndim=3),
    first_basis: wp.array(dtype=VEC3, ndim=4),
    second_basis: wp.array(dtype=VEC3, ndim=4),
    third_basis: wp.array(dtype=VEC3, ndim=4),
    measure: wp.array(dtype=DTYPE, ndim=4),
):
    batch, element, point, slot = wp.tid()
    first = VEC3(DTYPE(0.0), DTYPE(0.0), DTYPE(0.0))
    second = VEC3(DTYPE(0.0), DTYPE(0.0), DTYPE(0.0))
    third = VEC3(DTYPE(0.0), DTYPE(0.0), DTYPE(0.0))
    if sample_weight[point, slot] != DTYPE(0.0):
        thickness = sample_zeta[point, slot]
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
            first += sample_dxi[point, slot, node] * (position + thickness * director)
            second += sample_deta[point, slot, node] * (position + thickness * director)
            third += sample_shape[point, slot, node] * director
    first_basis[batch, element, point, slot] = first
    second_basis[batch, element, point, slot] = second
    third_basis[batch, element, point, slot] = third
    component = slot // MAX_PROJECTION_SAMPLES
    measure[batch, element, point, slot] = (
        _strain_measure(first, second, third, component)
        - reference_measure[element, point, slot]
    )


@wp.kernel
def q16_projected_strain_kernel(
    measure: wp.array(dtype=DTYPE, ndim=4),
    sample_weight: wp.array(dtype=DTYPE, ndim=2),
    transform: wp.array(dtype=DTYPE, ndim=4),
    strain: wp.array(dtype=DTYPE, ndim=4),
):
    batch, element, point, row = wp.tid()
    result = DTYPE(0.0)
    for component in range(PROJECTION_COMPONENT_COUNT):
        projected = DTYPE(0.0)
        base = component * MAX_PROJECTION_SAMPLES
        for sample in range(MAX_PROJECTION_SAMPLES):
            slot = base + sample
            projected += (
                sample_weight[point, slot] * measure[batch, element, point, slot]
            )
        result += transform[element, point, row, component] * projected
    strain[batch, element, point, row] = result


@wp.kernel
def q16_projected_b_kernel(
    first_basis: wp.array(dtype=VEC3, ndim=4),
    second_basis: wp.array(dtype=VEC3, ndim=4),
    third_basis: wp.array(dtype=VEC3, ndim=4),
    sample_shape: wp.array(dtype=DTYPE, ndim=3),
    sample_dxi: wp.array(dtype=DTYPE, ndim=3),
    sample_deta: wp.array(dtype=DTYPE, ndim=3),
    sample_zeta: wp.array(dtype=DTYPE, ndim=2),
    sample_weight: wp.array(dtype=DTYPE, ndim=2),
    projected_b: wp.array(dtype=DTYPE, ndim=4),
):
    batch, element, point, flat = wp.tid()
    component = flat // Q16_DOF_PER_ELEMENT
    dof = flat - component * Q16_DOF_PER_ELEMENT
    node = dof // 6
    local = dof - node * 6
    coordinate = local
    is_director = 0
    if local >= 3:
        coordinate = local - 3
        is_director = 1
    result = DTYPE(0.0)
    base = component * MAX_PROJECTION_SAMPLES
    for sample in range(MAX_PROJECTION_SAMPLES):
        slot = base + sample
        weight = sample_weight[point, slot]
        if weight != DTYPE(0.0):
            first_coefficient = sample_dxi[point, slot, node]
            second_coefficient = sample_deta[point, slot, node]
            third_coefficient = DTYPE(0.0)
            if is_director == 1:
                thickness = sample_zeta[point, slot]
                first_coefficient *= thickness
                second_coefficient *= thickness
                third_coefficient = sample_shape[point, slot, node]
            result += weight * _b_entry(
                first_basis[batch, element, point, slot],
                second_basis[batch, element, point, slot],
                third_basis[batch, element, point, slot],
                component,
                coordinate,
                first_coefficient,
                second_coefficient,
                third_coefficient,
            )
    projected_b[batch, element, point, flat] = result


@wp.kernel
def q16_geometry_gate_kernel(
    state: wp.array(dtype=DTYPE, ndim=3),
    query_shape: wp.array(dtype=DTYPE, ndim=2),
    query_dxi: wp.array(dtype=DTYPE, ndim=2),
    query_deta: wp.array(dtype=DTYPE, ndim=2),
    query_zeta: wp.array(dtype=DTYPE, ndim=1),
    invalid: wp.array(dtype=wp.int32, ndim=1),
):
    batch, element, point = wp.tid()
    first = VEC3(DTYPE(0.0), DTYPE(0.0), DTYPE(0.0))
    second = VEC3(DTYPE(0.0), DTYPE(0.0), DTYPE(0.0))
    third = VEC3(DTYPE(0.0), DTYPE(0.0), DTYPE(0.0))
    thickness = query_zeta[point]
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
        first += query_dxi[point, node] * (position + thickness * director)
        second += query_deta[point, node] * (position + thickness * director)
        third += query_shape[point, node] * director
    determinant = wp.dot(first, wp.cross(second, third))
    scale = wp.length(first) * wp.length(second) * wp.length(third)
    if (
        not wp.isfinite(determinant)
        or determinant <= DTYPE(1.1368683772161603e-13) * scale
    ):
        wp.atomic_max(invalid, 0, 1)


@wp.kernel
def q16_projected_direction_kernel(
    projected_b: wp.array(dtype=DTYPE, ndim=4),
    direction: wp.array(dtype=DTYPE, ndim=3),
    transform: wp.array(dtype=DTYPE, ndim=4),
    delta_strain: wp.array(dtype=DTYPE, ndim=4),
):
    batch, element, point, row = wp.tid()
    result = DTYPE(0.0)
    for component in range(PROJECTION_COMPONENT_COUNT):
        covariant = DTYPE(0.0)
        base = component * Q16_DOF_PER_ELEMENT
        for dof in range(Q16_DOF_PER_ELEMENT):
            covariant += (
                projected_b[batch, element, point, base + dof]
                * direction[batch, element, dof]
            )
        result += transform[element, point, row, component] * covariant
    delta_strain[batch, element, point, row] = result


@wp.kernel
def q16_eas_parameter_kernel(
    strain: wp.array(dtype=DTYPE, ndim=4),
    enhanced_mode: wp.array(dtype=DTYPE, ndim=3),
    reference_weight: wp.array(dtype=DTYPE, ndim=2),
    lame_lambda: wp.array(dtype=DTYPE, ndim=1),
    lame_mu: wp.array(dtype=DTYPE, ndim=1),
    eas_stiffness: wp.array(dtype=DTYPE, ndim=1),
    alpha: wp.array(dtype=DTYPE, ndim=2),
):
    batch, element = wp.tid()
    residual = DTYPE(0.0)
    for point in range(Q16_EAS_QUADRATURE_POINT_COUNT):
        compatible = _load6(strain, batch, element, point)
        mode = _load_mode(enhanced_mode, element, point)
        stress = _stress(compatible, lame_lambda[element], lame_mu[element])
        residual += reference_weight[element, point] * _dot6(mode, stress)
    alpha[batch, element] = -residual / eas_stiffness[element]


@wp.kernel
def q16_eas_delta_parameter_kernel(
    delta_strain: wp.array(dtype=DTYPE, ndim=4),
    enhanced_mode: wp.array(dtype=DTYPE, ndim=3),
    reference_weight: wp.array(dtype=DTYPE, ndim=2),
    lame_lambda: wp.array(dtype=DTYPE, ndim=1),
    lame_mu: wp.array(dtype=DTYPE, ndim=1),
    eas_stiffness: wp.array(dtype=DTYPE, ndim=1),
    delta_alpha: wp.array(dtype=DTYPE, ndim=2),
):
    batch, element = wp.tid()
    residual = DTYPE(0.0)
    for point in range(Q16_EAS_QUADRATURE_POINT_COUNT):
        variation = _load6(delta_strain, batch, element, point)
        mode = _load_mode(enhanced_mode, element, point)
        delta_stress = _stress(variation, lame_lambda[element], lame_mu[element])
        residual += reference_weight[element, point] * _dot6(mode, delta_stress)
    delta_alpha[batch, element] = -residual / eas_stiffness[element]


@wp.kernel
def q16_ans_eas_force_kernel(
    projected_b: wp.array(dtype=DTYPE, ndim=4),
    strain: wp.array(dtype=DTYPE, ndim=4),
    alpha: wp.array(dtype=DTYPE, ndim=2),
    transform: wp.array(dtype=DTYPE, ndim=4),
    enhanced_mode: wp.array(dtype=DTYPE, ndim=3),
    reference_weight: wp.array(dtype=DTYPE, ndim=2),
    lame_lambda: wp.array(dtype=DTYPE, ndim=1),
    lame_mu: wp.array(dtype=DTYPE, ndim=1),
    force: wp.array(dtype=DTYPE, ndim=3),
):
    batch, element, dof = wp.tid()
    result = DTYPE(0.0)
    for point in range(Q16_EAS_QUADRATURE_POINT_COUNT):
        mode = _load_mode(enhanced_mode, element, point)
        total_strain = _load6(strain, batch, element, point) + (
            alpha[batch, element] * mode
        )
        stress = _stress(total_strain, lame_lambda[element], lame_mu[element])
        for component in range(PROJECTION_COMPONENT_COUNT):
            dual = DTYPE(0.0)
            for row in range(6):
                dual += transform[element, point, row, component] * stress[row]
            result += reference_weight[element, point] * (
                projected_b[
                    batch,
                    element,
                    point,
                    component * Q16_DOF_PER_ELEMENT + dof,
                ]
                * dual
            )
    force[batch, element, dof] = result


@wp.kernel
def q16_ans_eas_tangent_kernel(
    projected_b: wp.array(dtype=DTYPE, ndim=4),
    delta_projected_b: wp.array(dtype=DTYPE, ndim=4),
    strain: wp.array(dtype=DTYPE, ndim=4),
    delta_strain: wp.array(dtype=DTYPE, ndim=4),
    alpha: wp.array(dtype=DTYPE, ndim=2),
    delta_alpha: wp.array(dtype=DTYPE, ndim=2),
    transform: wp.array(dtype=DTYPE, ndim=4),
    enhanced_mode: wp.array(dtype=DTYPE, ndim=3),
    reference_weight: wp.array(dtype=DTYPE, ndim=2),
    lame_lambda: wp.array(dtype=DTYPE, ndim=1),
    lame_mu: wp.array(dtype=DTYPE, ndim=1),
    action: wp.array(dtype=DTYPE, ndim=3),
):
    batch, element, dof = wp.tid()
    result = DTYPE(0.0)
    for point in range(Q16_EAS_QUADRATURE_POINT_COUNT):
        mode = _load_mode(enhanced_mode, element, point)
        total_strain = _load6(strain, batch, element, point) + (
            alpha[batch, element] * mode
        )
        total_delta_strain = _load6(delta_strain, batch, element, point) + (
            delta_alpha[batch, element] * mode
        )
        stress = _stress(total_strain, lame_lambda[element], lame_mu[element])
        delta_stress = _stress(
            total_delta_strain, lame_lambda[element], lame_mu[element]
        )
        for component in range(PROJECTION_COMPONENT_COUNT):
            dual = DTYPE(0.0)
            delta_dual = DTYPE(0.0)
            for row in range(6):
                transform_value = transform[element, point, row, component]
                dual += transform_value * stress[row]
                delta_dual += transform_value * delta_stress[row]
            slot = component * Q16_DOF_PER_ELEMENT + dof
            result += reference_weight[element, point] * (
                delta_projected_b[batch, element, point, slot] * dual
                + projected_b[batch, element, point, slot] * delta_dual
            )
    action[batch, element, dof] = result


@wp.kernel
def q16_condensed_material_diagonal_kernel(
    projected_b: wp.array(dtype=DTYPE, ndim=4),
    transform: wp.array(dtype=DTYPE, ndim=4),
    enhanced_mode: wp.array(dtype=DTYPE, ndim=3),
    reference_weight: wp.array(dtype=DTYPE, ndim=2),
    lame_lambda: wp.array(dtype=DTYPE, ndim=1),
    lame_mu: wp.array(dtype=DTYPE, ndim=1),
    eas_stiffness: wp.array(dtype=DTYPE, ndim=1),
    diagonal: wp.array(dtype=DTYPE, ndim=3),
):
    """Positive condensed material diagonal at one frozen Q16 state.

    The nonlinear geometric part remains in every tangent action.  This
    kernel extracts the positive material/EAS-condensed part used only as a
    Jacobi preconditioner, so it cannot alter the governing Newton equation.
    """

    batch, element, dof = wp.tid()
    alpha_residual = DTYPE(0.0)
    for point in range(Q16_EAS_QUADRATURE_POINT_COUNT):
        variation = VEC6(
            _projected_unit_strain_entry(
                projected_b, transform, batch, element, point, 0, dof
            ),
            _projected_unit_strain_entry(
                projected_b, transform, batch, element, point, 1, dof
            ),
            _projected_unit_strain_entry(
                projected_b, transform, batch, element, point, 2, dof
            ),
            _projected_unit_strain_entry(
                projected_b, transform, batch, element, point, 3, dof
            ),
            _projected_unit_strain_entry(
                projected_b, transform, batch, element, point, 4, dof
            ),
            _projected_unit_strain_entry(
                projected_b, transform, batch, element, point, 5, dof
            ),
        )
        mode = _load_mode(enhanced_mode, element, point)
        stress = _stress(variation, lame_lambda[element], lame_mu[element])
        alpha_residual += reference_weight[element, point] * _dot6(mode, stress)
    delta_alpha = -alpha_residual / eas_stiffness[element]

    result = DTYPE(0.0)
    for point in range(Q16_EAS_QUADRATURE_POINT_COUNT):
        variation = VEC6(
            _projected_unit_strain_entry(
                projected_b, transform, batch, element, point, 0, dof
            ),
            _projected_unit_strain_entry(
                projected_b, transform, batch, element, point, 1, dof
            ),
            _projected_unit_strain_entry(
                projected_b, transform, batch, element, point, 2, dof
            ),
            _projected_unit_strain_entry(
                projected_b, transform, batch, element, point, 3, dof
            ),
            _projected_unit_strain_entry(
                projected_b, transform, batch, element, point, 4, dof
            ),
            _projected_unit_strain_entry(
                projected_b, transform, batch, element, point, 5, dof
            ),
        )
        mode = _load_mode(enhanced_mode, element, point)
        delta_stress = _stress(
            variation + delta_alpha * mode,
            lame_lambda[element],
            lame_mu[element],
        )
        for component in range(PROJECTION_COMPONENT_COUNT):
            dual = DTYPE(0.0)
            for row in range(PROJECTION_COMPONENT_COUNT):
                dual += transform[element, point, row, component] * delta_stress[row]
            result += reference_weight[element, point] * (
                projected_b[
                    batch,
                    element,
                    point,
                    component * Q16_DOF_PER_ELEMENT + dof,
                ]
                * dual
            )
    diagonal[batch, element, dof] = result


@wp.kernel
def _nonfinite3_kernel(
    value: wp.array(dtype=DTYPE, ndim=3),
    flag: wp.array(dtype=wp.int32, ndim=1),
):
    i, j, k = wp.tid()
    if not wp.isfinite(value[i, j, k]):
        wp.atomic_max(flag, 0, 1)


@wp.kernel
def _nonfinite2_kernel(
    value: wp.array(dtype=DTYPE, ndim=2),
    flag: wp.array(dtype=wp.int32, ndim=1),
):
    i, j = wp.tid()
    if not wp.isfinite(value[i, j]):
        wp.atomic_max(flag, 0, 1)


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


def _assert_finite3(name: str, value: wp.array, *, device: str) -> None:
    flag = wp.zeros(1, dtype=wp.int32, device=device)
    wp.launch(
        _nonfinite3_kernel,
        dim=value.shape,
        inputs=[value],
        outputs=[flag],
        device=device,
    )
    wp.synchronize_device(device)
    if int(flag.numpy()[0]) != 0:
        raise FloatingPointError(f"{name} contains non-finite values")


def _assert_finite2(name: str, value: wp.array, *, device: str) -> None:
    flag = wp.zeros(1, dtype=wp.int32, device=device)
    wp.launch(
        _nonfinite2_kernel,
        dim=value.shape,
        inputs=[value],
        outputs=[flag],
        device=device,
    )
    wp.synchronize_device(device)
    if int(flag.numpy()[0]) != 0:
        raise FloatingPointError(f"{name} contains non-finite values")


class Q16CudaMITC16EASOperator:
    """Resident CUDA operator for the fixed projected and condensed Q16 model."""

    __slots__ = (
        "_eas_stiffness",
        "_enhanced_mode",
        "_lame_lambda",
        "_lame_mu",
        "_query_deta",
        "_query_dxi",
        "_query_shape",
        "_query_zeta",
        "_reference_measure",
        "_reference_weight",
        "_sample_deta",
        "_sample_dxi",
        "_sample_shape",
        "_sample_weight",
        "_sample_zeta",
        "_transform",
        "device",
        "element_count",
    )

    def __init__(
        self, elements: tuple[Q16MITC16EASContinuumElement, ...], *, device: str
    ) -> None:
        if type(elements) is not tuple or not elements:
            raise ValueError("elements must be a non-empty exact tuple")
        if any(
            type(element) is not Q16MITC16EASContinuumElement for element in elements
        ):
            raise TypeError(
                "every element must be an exact Q16MITC16EASContinuumElement"
            )
        if config.dtype_name() != "float64" or DTYPE != wp.float64:
            raise RuntimeError("Q16 ANS/EAS production operator requires float64")
        selected = wp.get_device(device)
        if not selected.is_cuda:
            raise ValueError("Q16 ANS/EAS production operator requires CUDA")
        self.device = selected.alias
        self.element_count = len(elements)
        coordinates = np.ascontiguousarray(elements[0].quadrature.coordinates)
        if any(
            not np.array_equal(element.quadrature.coordinates, coordinates)
            for element in elements
        ):
            raise ValueError("Q16 ANS/EAS quadrature coordinates differ")
        shape, dxi, deta, zeta, weight = _projection_stencil(coordinates)
        reference_measure = _reference_measures(
            elements, shape, dxi, deta, zeta, weight
        )
        query_shape, query_dxi, query_deta, query_zeta = _query_stencil(coordinates)
        transforms = np.ascontiguousarray(
            np.stack(
                [element.quadrature.covariant_to_cartesian for element in elements]
            )
        )
        modes = np.ascontiguousarray(
            np.stack([element.quadrature.enhanced_modes for element in elements])
        )
        weights = np.ascontiguousarray(
            np.stack([element.quadrature.reference_weights for element in elements])
        )
        eas_stiffness = []
        for element in elements:
            stiffness = sum(
                element.quadrature.reference_weights[point]
                * float(
                    element.quadrature.enhanced_modes[point]
                    @ element.constitutive
                    @ element.quadrature.enhanced_modes[point]
                )
                for point in range(Q16_EAS_QUADRATURE_POINT_COUNT)
            )
            if not np.isfinite(stiffness) or stiffness <= 0.0:
                raise ValueError("Q16 EAS local stiffness is non-positive")
            eas_stiffness.append(stiffness)
        self._sample_shape = wp.array(shape, dtype=DTYPE, device=self.device)
        self._sample_dxi = wp.array(dxi, dtype=DTYPE, device=self.device)
        self._sample_deta = wp.array(deta, dtype=DTYPE, device=self.device)
        self._sample_zeta = wp.array(zeta, dtype=DTYPE, device=self.device)
        self._sample_weight = wp.array(weight, dtype=DTYPE, device=self.device)
        self._reference_measure = wp.array(
            reference_measure, dtype=DTYPE, device=self.device
        )
        self._query_shape = wp.array(query_shape, dtype=DTYPE, device=self.device)
        self._query_dxi = wp.array(query_dxi, dtype=DTYPE, device=self.device)
        self._query_deta = wp.array(query_deta, dtype=DTYPE, device=self.device)
        self._query_zeta = wp.array(query_zeta, dtype=DTYPE, device=self.device)
        self._transform = wp.array(transforms, dtype=DTYPE, device=self.device)
        self._enhanced_mode = wp.array(modes, dtype=DTYPE, device=self.device)
        self._reference_weight = wp.array(weights, dtype=DTYPE, device=self.device)
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
        self._eas_stiffness = wp.array(
            np.asarray(eas_stiffness), dtype=DTYPE, device=self.device
        )

    def _validate(self, name: str, value: Any) -> None:
        _require_state(
            name, value, device=self.device, element_count=self.element_count
        )
        _assert_finite3(name, value, device=self.device)

    def _sample_fields(self, value: wp.array):
        shape = (
            value.shape[0],
            self.element_count,
            Q16_EAS_QUADRATURE_POINT_COUNT,
            PROJECTION_SAMPLE_SLOT_COUNT,
        )
        first = wp.zeros(shape, dtype=VEC3, device=self.device)
        second = wp.zeros(shape, dtype=VEC3, device=self.device)
        third = wp.zeros(shape, dtype=VEC3, device=self.device)
        measure = wp.zeros(shape, dtype=DTYPE, device=self.device)
        wp.launch(
            q16_sample_basis_measure_kernel,
            dim=shape,
            inputs=[
                value,
                self._sample_shape,
                self._sample_dxi,
                self._sample_deta,
                self._sample_zeta,
                self._sample_weight,
                self._reference_measure,
            ],
            outputs=[first, second, third, measure],
            device=self.device,
        )
        return first, second, third, measure

    def _projected_b(self, first, second, third, batch: int):
        projected_b = wp.zeros(
            (
                batch,
                self.element_count,
                Q16_EAS_QUADRATURE_POINT_COUNT,
                PROJECTED_B_SLOT_COUNT,
            ),
            dtype=DTYPE,
            device=self.device,
        )
        wp.launch(
            q16_projected_b_kernel,
            dim=projected_b.shape,
            inputs=[
                first,
                second,
                third,
                self._sample_shape,
                self._sample_dxi,
                self._sample_deta,
                self._sample_zeta,
                self._sample_weight,
            ],
            outputs=[projected_b],
            device=self.device,
        )
        return projected_b

    def _strain(self, measure: wp.array, state: wp.array):
        strain = wp.zeros(
            (
                state.shape[0],
                self.element_count,
                Q16_EAS_QUADRATURE_POINT_COUNT,
                6,
            ),
            dtype=DTYPE,
            device=self.device,
        )
        wp.launch(
            q16_projected_strain_kernel,
            dim=strain.shape,
            inputs=[measure, self._sample_weight, self._transform],
            outputs=[strain],
            device=self.device,
        )
        invalid = wp.zeros(1, dtype=wp.int32, device=self.device)
        wp.launch(
            q16_geometry_gate_kernel,
            dim=(
                state.shape[0],
                self.element_count,
                Q16_EAS_QUADRATURE_POINT_COUNT,
            ),
            inputs=[
                state,
                self._query_shape,
                self._query_dxi,
                self._query_deta,
                self._query_zeta,
            ],
            outputs=[invalid],
            device=self.device,
        )
        wp.synchronize_device(self.device)
        if int(invalid.numpy()[0]) != 0:
            raise FloatingPointError(
                "Q16 current geometry is orientation reversing or singular"
            )
        return strain

    def _alpha(self, strain: wp.array, batch: int):
        alpha = wp.zeros((batch, self.element_count), dtype=DTYPE, device=self.device)
        wp.launch(
            q16_eas_parameter_kernel,
            dim=alpha.shape,
            inputs=[
                strain,
                self._enhanced_mode,
                self._reference_weight,
                self._lame_lambda,
                self._lame_mu,
                self._eas_stiffness,
            ],
            outputs=[alpha],
            device=self.device,
        )
        _assert_finite2("Q16 EAS parameter", alpha, device=self.device)
        return alpha

    def _current_fields(self, state: wp.array):
        first, second, third, measure = self._sample_fields(state)
        strain = self._strain(measure, state)
        projected_b = self._projected_b(first, second, third, state.shape[0])
        alpha = self._alpha(strain, state.shape[0])
        return strain, projected_b, alpha

    def _linearization_prechecked(self, state: wp.array):
        """Freeze one state-dependent CUDA linearization for a Krylov solve."""

        return self._current_fields(state)

    def enhanced_parameter(self, state: Any):
        self._validate("state", state)
        _, _, _, measure = self._sample_fields(state)
        strain = self._strain(measure, state)
        return self._alpha(strain, state.shape[0])

    def internal_force(self, state: Any):
        self._validate("state", state)
        return self._internal_force_prechecked(state)

    def _internal_force_prechecked(self, state: wp.array):
        """Evaluate after the shared-mesh owner validated shape/device/dtype."""

        return self._internal_force_linearized_prechecked(
            self._linearization_prechecked(state)
        )

    def _internal_force_linearized_prechecked(self, linearization):
        strain, projected_b, alpha = linearization
        force = wp.zeros(
            (strain.shape[0], self.element_count, Q16_DOF_PER_ELEMENT),
            dtype=DTYPE,
            device=self.device,
        )
        wp.launch(
            q16_ans_eas_force_kernel,
            dim=force.shape,
            inputs=[
                projected_b,
                strain,
                alpha,
                self._transform,
                self._enhanced_mode,
                self._reference_weight,
                self._lame_lambda,
                self._lame_mu,
            ],
            outputs=[force],
            device=self.device,
        )
        _assert_finite3("Q16 ANS/EAS force", force, device=self.device)
        return force

    def tangent_action(self, state: Any, direction: Any):
        self._validate("state", state)
        self._validate("direction", direction)
        if direction.shape != state.shape:
            raise ValueError("direction shape differs from state")
        return self._tangent_action_prechecked(state, direction)

    def _tangent_action_prechecked(
        self, state: wp.array, direction: wp.array
    ) -> wp.array:
        """Evaluate after the shared-mesh owner validated both input arrays."""

        return self._tangent_action_linearized_prechecked(
            direction,
            self._linearization_prechecked(state),
        )

    def _tangent_action_linearized_prechecked(
        self, direction: wp.array, linearization
    ) -> wp.array:
        """Apply a previously frozen CUDA tangent without rebuilding it."""

        strain, projected_b, alpha = linearization
        delta_first, delta_second, delta_third, _ = self._sample_fields(direction)
        delta_projected_b = self._projected_b(
            delta_first, delta_second, delta_third, direction.shape[0]
        )
        delta_strain = wp.zeros_like(strain)
        wp.launch(
            q16_projected_direction_kernel,
            dim=delta_strain.shape,
            inputs=[projected_b, direction, self._transform],
            outputs=[delta_strain],
            device=self.device,
        )
        delta_alpha = wp.zeros_like(alpha)
        wp.launch(
            q16_eas_delta_parameter_kernel,
            dim=delta_alpha.shape,
            inputs=[
                delta_strain,
                self._enhanced_mode,
                self._reference_weight,
                self._lame_lambda,
                self._lame_mu,
                self._eas_stiffness,
            ],
            outputs=[delta_alpha],
            device=self.device,
        )
        action = wp.zeros_like(direction)
        wp.launch(
            q16_ans_eas_tangent_kernel,
            dim=direction.shape,
            inputs=[
                projected_b,
                delta_projected_b,
                strain,
                delta_strain,
                alpha,
                delta_alpha,
                self._transform,
                self._enhanced_mode,
                self._reference_weight,
                self._lame_lambda,
                self._lame_mu,
            ],
            outputs=[action],
            device=self.device,
        )
        _assert_finite3("Q16 ANS/EAS tangent action", action, device=self.device)
        return action

    def _material_diagonal_linearized_prechecked(self, linearization) -> wp.array:
        """Return the CUDA condensed-material Jacobi diagonal."""

        strain, projected_b, _ = linearization
        diagonal = wp.zeros(
            (strain.shape[0], self.element_count, Q16_DOF_PER_ELEMENT),
            dtype=DTYPE,
            device=self.device,
        )
        wp.launch(
            q16_condensed_material_diagonal_kernel,
            dim=diagonal.shape,
            inputs=[
                projected_b,
                self._transform,
                self._enhanced_mode,
                self._reference_weight,
                self._lame_lambda,
                self._lame_mu,
                self._eas_stiffness,
            ],
            outputs=[diagonal],
            device=self.device,
        )
        _assert_finite3("Q16 condensed material diagonal", diagonal, device=self.device)
        return diagonal


__all__ = [
    "Q16CudaMITC16EASOperator",
    "q16_ans_eas_force_kernel",
    "q16_ans_eas_tangent_kernel",
    "q16_condensed_material_diagonal_kernel",
    "q16_eas_delta_parameter_kernel",
    "q16_eas_parameter_kernel",
    "q16_geometry_gate_kernel",
    "q16_projected_b_kernel",
    "q16_projected_direction_kernel",
    "q16_projected_strain_kernel",
    "q16_sample_basis_measure_kernel",
]
