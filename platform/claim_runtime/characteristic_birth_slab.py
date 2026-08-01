"""P2 characteristic space-time birth boundary for a growing wake strip.

The physical birth wedge is

    0 < t < dt,  0 < x < c t,
    partial_t(mu) + c partial_x(mu) = 0.

After ``r=t/dt`` and ``q=x/(c*dt)`` it is the reference triangle with
vertices ``(0,0), (1,0), (1,1)``.  Only the P2 values on the inflow edge
``q=0`` are prescribed.  The other three P2 values are obtained from the
weak characteristic equation.

There is intentionally no newborn initial-state argument: the lower
time-face of the birth wedge is one point and has zero surface measure.
This module does not remap an old state or compute pressure, force, LESP,
regularization, smoothing, or structural quantities.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


class CharacteristicBirthSlabError(ValueError):
    """Raised when a birth-slab input or weak system is invalid."""


REFERENCE_NODES = np.array(
    (
        (0.0, 0.0),  # A
        (1.0, 0.0),  # B
        (1.0, 1.0),  # C
        (0.5, 0.0),  # AB
        (1.0, 0.5),  # BC
        (0.5, 0.5),  # CA
    ),
    dtype=float,
)
INFLOW_NODE_INDICES = np.array((0, 3, 1), dtype=np.int64)
SOLVED_NODE_INDICES = np.array((2, 4, 5), dtype=np.int64)
ENDPOINT_CHRONOLOGICAL_INDICES = np.array((2, 4, 1), dtype=np.int64)


@dataclass(frozen=True)
class CharacteristicBirthSlabReport:
    prescribed_dof_count: int
    solved_dof_count: int
    initial_newborn_scalar_count: int
    free_rank: int
    free_rank_deficiency: int
    free_condition_number: float
    weak_residual_abs_max: float
    inflow_identity_abs_max: float
    endpoint_trace_identity_abs_max: float
    newborn_mass_balance_abs: float
    input_state_mutation_abs: float


@dataclass(frozen=True)
class CharacteristicBirthSlab:
    timestep: float
    convection_speed: float
    inflow_trace: np.ndarray
    reference_node_values: np.ndarray
    endpoint_chronological_rows: np.ndarray
    weak_matrix: np.ndarray
    report: CharacteristicBirthSlabReport


def _p2_basis_and_gradient(
    r: float,
    q: float,
) -> tuple[np.ndarray, np.ndarray]:
    barycentric = np.array((1.0 - r, r - q, q), dtype=float)
    barycentric_gradient = np.array(
        ((-1.0, 0.0), (1.0, -1.0), (0.0, 1.0)),
        dtype=float,
    )
    values = np.empty(6, dtype=float)
    gradient = np.empty((6, 2), dtype=float)
    values[:3] = barycentric * (2.0 * barycentric - 1.0)
    for index in range(3):
        gradient[index] = (
            (4.0 * barycentric[index] - 1.0)
            * barycentric_gradient[index]
        )
    edge_pairs = ((0, 1), (1, 2), (2, 0))
    for local, (left, right) in enumerate(edge_pairs, start=3):
        values[local] = (
            4.0 * barycentric[left] * barycentric[right]
        )
        gradient[local] = 4.0 * (
            barycentric_gradient[left] * barycentric[right]
            + barycentric[left] * barycentric_gradient[right]
        )
    return values, gradient


def _weak_characteristic_matrix(
    quadrature_order: int,
) -> np.ndarray:
    if (
        not isinstance(quadrature_order, (int, np.integer))
        or quadrature_order < 3
    ):
        raise CharacteristicBirthSlabError(
            "quadrature_order must be an integer of at least three"
        )
    abscissa, weight = np.polynomial.legendre.leggauss(
        int(quadrature_order)
    )
    unit = 0.5 * (abscissa + 1.0)
    unit_weight = 0.5 * weight
    matrix = np.zeros((6, 6), dtype=float)
    for outer, r in enumerate(unit):
        for inner, eta in enumerate(unit):
            q = r * eta
            jacobian = r
            basis, gradient = _p2_basis_and_gradient(
                float(r),
                float(q),
            )
            characteristic_derivative = (
                gradient[:, 0] + gradient[:, 1]
            )
            matrix += (
                unit_weight[outer]
                * unit_weight[inner]
                * jacobian
                * np.outer(basis, characteristic_derivative)
            )
    return matrix


def _trace_array(value: Any) -> tuple[np.ndarray, bool]:
    trace = np.asarray(value, dtype=float)
    was_vector = trace.ndim == 1
    if was_vector:
        trace = trace[:, None]
    if (
        trace.ndim != 2
        or trace.shape[0] != 3
        or trace.shape[1] < 1
        or not np.all(np.isfinite(trace))
    ):
        raise CharacteristicBirthSlabError(
            "inflow_trace must be finite with shape (3,) or (3, n)"
        )
    return trace.copy(), was_vector


def solve_p2_characteristic_birth_slab(
    inflow_trace: Any,
    *,
    timestep: float,
    convection_speed: float,
    quadrature_order: int = 8,
) -> CharacteristicBirthSlab:
    """Solve the three non-inflow P2 values on one birth wedge."""
    if not np.isfinite(timestep) or timestep <= 0.0:
        raise CharacteristicBirthSlabError(
            "timestep must be finite and strictly positive"
        )
    if not np.isfinite(convection_speed) or convection_speed <= 0.0:
        raise CharacteristicBirthSlabError(
            "convection_speed must be finite and strictly positive"
        )
    trace, was_vector = _trace_array(inflow_trace)
    trace_snapshot = trace.copy()
    matrix = _weak_characteristic_matrix(quadrature_order)
    free_matrix = matrix[
        np.ix_(SOLVED_NODE_INDICES, SOLVED_NODE_INDICES)
    ]
    coupling = matrix[
        np.ix_(SOLVED_NODE_INDICES, INFLOW_NODE_INDICES)
    ]
    rank = int(np.linalg.matrix_rank(free_matrix))
    condition = float(np.linalg.cond(free_matrix))
    if rank != len(SOLVED_NODE_INDICES) or not np.isfinite(condition):
        raise CharacteristicBirthSlabError(
            "characteristic birth free block is singular or non-finite"
        )
    node_values = np.empty((6, trace.shape[1]), dtype=float)
    node_values[INFLOW_NODE_INDICES] = trace
    node_values[SOLVED_NODE_INDICES] = np.linalg.solve(
        free_matrix,
        -(coupling @ trace),
    )
    weak_residual = matrix @ node_values
    endpoint = node_values[ENDPOINT_CHRONOLOGICAL_INDICES]
    endpoint_expected = trace.copy()
    line_weights = np.array((1.0, 4.0, 1.0), dtype=float) / 6.0
    endpoint_integral = (
        float(convection_speed)
        * float(timestep)
        * (line_weights @ endpoint)
    )
    inflow_integral = (
        float(convection_speed)
        * float(timestep)
        * (line_weights @ trace)
    )
    mutation = float(
        np.max(np.abs(trace - trace_snapshot), initial=0.0)
    )
    report = CharacteristicBirthSlabReport(
        prescribed_dof_count=len(INFLOW_NODE_INDICES),
        solved_dof_count=len(SOLVED_NODE_INDICES),
        initial_newborn_scalar_count=0,
        free_rank=rank,
        free_rank_deficiency=len(SOLVED_NODE_INDICES) - rank,
        free_condition_number=condition,
        weak_residual_abs_max=float(
            np.max(np.abs(weak_residual), initial=0.0)
        ),
        inflow_identity_abs_max=float(
            np.max(
                np.abs(
                    node_values[INFLOW_NODE_INDICES] - trace
                ),
                initial=0.0,
            )
        ),
        endpoint_trace_identity_abs_max=float(
            np.max(np.abs(endpoint - endpoint_expected), initial=0.0)
        ),
        newborn_mass_balance_abs=float(
            np.max(
                np.abs(endpoint_integral - inflow_integral),
                initial=0.0,
            )
        ),
        input_state_mutation_abs=mutation,
    )
    if was_vector:
        stored_trace = trace[:, 0].copy()
        stored_nodes = node_values[:, 0].copy()
        stored_endpoint = endpoint[:, 0].copy()
    else:
        stored_trace = trace.copy()
        stored_nodes = node_values.copy()
        stored_endpoint = endpoint.copy()
    return CharacteristicBirthSlab(
        timestep=float(timestep),
        convection_speed=float(convection_speed),
        inflow_trace=stored_trace,
        reference_node_values=stored_nodes,
        endpoint_chronological_rows=stored_endpoint,
        weak_matrix=matrix.copy(),
        report=report,
    )
