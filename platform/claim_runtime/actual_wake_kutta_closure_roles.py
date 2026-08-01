"""No-force algebra for independent wake traces and Kutta closures.

This module does not introduce a new aerodynamic closure.  It exposes the
equation roles hidden by the current Morino elimination:

    B phi + W g = b,
    g - C phi = 0.

It also supplies an adversarial quotient solve used to test whether deleting
body Galerkin equations is topology invariant.  Pressure, force, LESP,
regularization and production activation are deliberately absent.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .actual_boundary_body_wake import (
    ActualBoundaryBodyWakeSolution,
    _cut_jump_matrix,
)
from .actual_boundary_p2_galerkin import (
    _global_shape_matrix,
    _surface_quadrature,
)
from .classified_p2_cut_topology import ClassifiedP2CutTopology


class KuttaClosureRoleError(ValueError):
    """Invalid topology or algebra in the no-force closure-role oracle."""


@dataclass(frozen=True)
class CutRoleOperators:
    """Independent and continuous spaces induced by one classified cut."""

    full_jump: np.ndarray
    active_jump: np.ndarray
    active_row_indices: np.ndarray
    zero_row_indices: np.ndarray
    continuous_prolongation: np.ndarray
    paired_jump_injection: np.ndarray

    @property
    def full_cut_node_count(self) -> int:
        return int(self.full_jump.shape[0])

    @property
    def independent_jump_count(self) -> int:
        return int(self.active_jump.shape[0])


@dataclass(frozen=True)
class IndependentWakeSystem:
    """Full body equation with the active current wake trace uneliminated."""

    body_matrix: np.ndarray
    independent_wake_matrix: np.ndarray
    active_jump: np.ndarray
    right_hand_side: np.ndarray
    eliminated_wake_matrix: np.ndarray
    eliminated_matrix: np.ndarray
    wake_factorization_error: float


@dataclass(frozen=True)
class MorinoBlockSolution:
    """Solution of the full-body-equation plus one Morino closure."""

    body_potential: np.ndarray
    wake_jump: np.ndarray
    full_bie_residual: np.ndarray
    closure_residual: np.ndarray
    block_matrix: np.ndarray
    block_rank: int
    block_condition_number: float


@dataclass(frozen=True)
class QuotientSolution:
    """One metric-dependent row-deletion counterexample solution."""

    body_potential: np.ndarray
    body_bie_residual: np.ndarray
    projected_residual: np.ndarray
    attachment_residual: np.ndarray
    system_rank: int
    system_condition_number: float


def cut_role_operators(
    topology: ClassifiedP2CutTopology,
) -> CutRoleOperators:
    """Build exact active-jump, continuous and paired-jump operators."""
    if not isinstance(topology, ClassifiedP2CutTopology):
        raise KuttaClosureRoleError(
            "topology must be a ClassifiedP2CutTopology"
        )
    full = _cut_jump_matrix(topology)
    row_norm = np.linalg.norm(full, axis=1)
    active_rows = np.flatnonzero(row_norm > 0.0)
    zero_rows = np.flatnonzero(row_norm == 0.0)
    active = full[active_rows].copy()
    rank = int(np.linalg.matrix_rank(active))
    if rank != len(active_rows):
        raise KuttaClosureRoleError(
            "nonzero cut-jump rows are not independent"
        )

    base_count = int(topology.base_topology.dof_count)
    classified_count = int(topology.dof_count)
    prolongation = np.zeros(
        (classified_count, base_count),
        dtype=float,
    )
    prolongation[:base_count] = np.eye(base_count)
    for upper, lower in zip(
        topology.upper_cut_dofs,
        topology.lower_cut_dofs,
        strict=True,
    ):
        upper_index = int(upper)
        lower_index = int(lower)
        if upper_index == lower_index:
            continue
        if upper_index >= base_count or lower_index < base_count:
            raise KuttaClosureRoleError(
                "classified cut duplicates do not follow base/append layout"
            )
        prolongation[lower_index, upper_index] = 1.0

    injection = 0.5 * active.T
    return CutRoleOperators(
        full_jump=full,
        active_jump=active,
        active_row_indices=active_rows.astype(np.int64),
        zero_row_indices=zero_rows.astype(np.int64),
        continuous_prolongation=prolongation,
        paired_jump_injection=injection,
    )


def independent_wake_system(
    solution: ActualBoundaryBodyWakeSolution,
    operators: CutRoleOperators,
) -> IndependentWakeSystem:
    """Factor the eliminated current-wake operator through active jump DOFs.

    The right inverse is exact for the disjoint paired cut rows.  This
    factorization is a read-only equation-role audit; a production solver
    must assemble the independent wake basis directly.
    """
    if not isinstance(solution, ActualBoundaryBodyWakeSolution):
        raise KuttaClosureRoleError(
            "solution must be an ActualBoundaryBodyWakeSolution"
        )
    active = np.asarray(operators.active_jump, dtype=float)
    gram = active @ active.T
    try:
        right_inverse = active.T @ np.linalg.solve(
            gram,
            np.eye(len(active)),
        )
    except np.linalg.LinAlgError as error:
        raise KuttaClosureRoleError(
            "active cut-jump map has no exact right inverse"
        ) from error
    eliminated_wake = np.asarray(solution.wake_matrix, dtype=float)
    independent_wake = eliminated_wake @ right_inverse
    factorization_error = float(
        np.max(
            np.abs(
                independent_wake @ active - eliminated_wake
            ),
            initial=0.0,
        )
    )
    return IndependentWakeSystem(
        body_matrix=np.asarray(
            solution.body_matrix,
            dtype=float,
        ).copy(),
        independent_wake_matrix=independent_wake,
        active_jump=active.copy(),
        right_hand_side=np.asarray(
            solution.right_hand_side,
            dtype=float,
        ).copy(),
        eliminated_wake_matrix=eliminated_wake.copy(),
        eliminated_matrix=np.asarray(
            solution.matrix,
            dtype=float,
        ).copy(),
        wake_factorization_error=factorization_error,
    )


def solve_morino_block(
    system: IndependentWakeSystem,
    *,
    wake_coordinate_map: Any | None = None,
) -> MorinoBlockSolution:
    """Solve all body BIE rows plus one Morino closure in any wake basis."""
    body = np.asarray(system.body_matrix, dtype=float)
    wake = np.asarray(system.independent_wake_matrix, dtype=float)
    jump = np.asarray(system.active_jump, dtype=float)
    right_hand_side = np.asarray(system.right_hand_side, dtype=float)
    independent_count = wake.shape[1]
    if wake_coordinate_map is None:
        coordinate_map = np.eye(independent_count)
    else:
        coordinate_map = np.asarray(
            wake_coordinate_map,
            dtype=float,
        )
    if (
        coordinate_map.shape
        != (independent_count, independent_count)
        or not np.all(np.isfinite(coordinate_map))
        or np.linalg.matrix_rank(coordinate_map) != independent_count
    ):
        raise KuttaClosureRoleError(
            "wake_coordinate_map must be finite, square and invertible"
        )

    block = np.block(
        [
            [body, wake @ coordinate_map],
            [-jump, coordinate_map],
        ]
    )
    block_right_hand_side = np.concatenate(
        (right_hand_side, np.zeros(independent_count))
    )
    try:
        unknown = np.linalg.solve(block, block_right_hand_side)
    except np.linalg.LinAlgError as error:
        raise KuttaClosureRoleError(
            "Morino closure block is singular"
        ) from error
    body_potential = unknown[: body.shape[0]]
    wake_coordinates = unknown[body.shape[0] :]
    wake_jump = coordinate_map @ wake_coordinates
    full_residual = body @ body_potential + wake @ wake_jump
    full_residual -= right_hand_side
    closure_residual = wake_jump - jump @ body_potential
    return MorinoBlockSolution(
        body_potential=body_potential,
        wake_jump=wake_jump,
        full_bie_residual=full_residual,
        closure_residual=closure_residual,
        block_matrix=block,
        block_rank=int(np.linalg.matrix_rank(block)),
        block_condition_number=float(np.linalg.cond(block)),
    )


def consistent_body_surface_mass(
    solution: ActualBoundaryBodyWakeSolution,
    *,
    quadrature_order: int,
) -> np.ndarray:
    """Return the consistent classified-P2 surface mass matrix."""
    if (
        not isinstance(quadrature_order, (int, np.integer))
        or int(quadrature_order) < 2
    ):
        raise KuttaClosureRoleError(
            "quadrature_order must be an integer >=2"
        )
    _points, weights, owners, barycentric = _surface_quadrature(
        solution.mesh,
        int(quadrature_order),
    )
    shape = _global_shape_matrix(
        solution.topology,
        owners,
        barycentric,
    )
    mass = shape.T @ (weights[:, None] * shape)
    if (
        not np.all(np.isfinite(mass))
        or np.linalg.matrix_rank(mass) != solution.body_unknown_count
    ):
        raise KuttaClosureRoleError(
            "consistent body surface mass is not finite positive rank"
        )
    return mass


def orthonormal_nullspace(
    row_operator: Any,
) -> np.ndarray:
    """Return an orthonormal Euclidean basis for a row nullspace."""
    operator = np.asarray(row_operator, dtype=float)
    if operator.ndim != 2 or not np.all(np.isfinite(operator)):
        raise KuttaClosureRoleError(
            "row_operator must be a finite matrix"
        )
    _left, singular_values, right_transpose = np.linalg.svd(
        operator,
        full_matrices=True,
    )
    if singular_values.size:
        tolerance = (
            np.finfo(float).eps
            * max(operator.shape)
            * singular_values[0]
        )
        rank = int(np.sum(singular_values > tolerance))
    else:
        rank = 0
    return right_transpose[rank:].T.copy()


def solve_quotient_with_attachment(
    system: IndependentWakeSystem,
    *,
    quotient_basis: Any,
    prescribed_wake_jump: Any,
) -> QuotientSolution:
    """Delete body rows by a quotient and impose attachment instead."""
    body = np.asarray(system.body_matrix, dtype=float)
    wake = np.asarray(system.independent_wake_matrix, dtype=float)
    jump = np.asarray(system.active_jump, dtype=float)
    right_hand_side = np.asarray(system.right_hand_side, dtype=float)
    quotient = np.asarray(quotient_basis, dtype=float)
    prescribed = np.asarray(prescribed_wake_jump, dtype=float)
    expected_columns = body.shape[0] - jump.shape[0]
    if (
        quotient.shape != (body.shape[0], expected_columns)
        or prescribed.shape != (jump.shape[0],)
        or not np.all(np.isfinite(quotient))
        or not np.all(np.isfinite(prescribed))
    ):
        raise KuttaClosureRoleError(
            "quotient basis or prescribed wake jump has invalid shape"
        )
    reduced = np.vstack((quotient.T @ body, jump))
    reduced_right_hand_side = np.concatenate(
        (
            quotient.T
            @ (right_hand_side - wake @ prescribed),
            prescribed,
        )
    )
    try:
        body_potential = np.linalg.solve(
            reduced,
            reduced_right_hand_side,
        )
    except np.linalg.LinAlgError as error:
        raise KuttaClosureRoleError(
            "quotient-plus-attachment system is singular"
        ) from error
    full_residual = body @ body_potential + wake @ prescribed
    full_residual -= right_hand_side
    return QuotientSolution(
        body_potential=body_potential,
        body_bie_residual=full_residual,
        projected_residual=quotient.T @ full_residual,
        attachment_residual=jump @ body_potential - prescribed,
        system_rank=int(np.linalg.matrix_rank(reduced)),
        system_condition_number=float(np.linalg.cond(reduced)),
    )
