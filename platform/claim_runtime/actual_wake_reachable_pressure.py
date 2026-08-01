"""Read-only S3ai pressure-rate observations on the S3e material path.

This module contains no pressure solve and no force integration.  It only
turns an already solved actual-boundary/material-wake stage into the frozen
active-P2 weak pressure observation required by S3ai.  In particular, the
current-wake operator is assembled by
``assemble_direct_independent_wake_matrix``; the eliminated operator is used
only as an equality audit and never as the primary operator.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from .actual_boundary_body_wake import (
    ActualBoundaryBodyWakeSolution,
    MaterialWakeCutAttachment,
)
from .actual_wake_direct_independent import (
    DirectIndependentWakeAssembly,
    assemble_direct_independent_wake_matrix,
)
from .actual_wake_kutta_closure_roles import (
    CutRoleOperators,
    IndependentWakeSystem,
    cut_role_operators,
)
from .actual_wake_kutta_compatibility import (
    ActualPressureKuttaModel,
    IndependentWakeBodyState,
    _line_p2_shape,
    _segment_pressure_jump,
    trailing_edge_face_pairs,
)
from .material_birth_flux import consistent_p2_line_mass
from .material_attachment_inventory import extract_surface_boundary_trace


class ReachablePressureObservationError(ValueError):
    """An S3ai read-only pressure observation is not well defined."""


@dataclass(frozen=True)
class DirectIndependentStageObservation:
    """Stored-state primary observation plus an independent direct audit.

    ``weak_pressure`` is evaluated from the S3e stored body potential.
    ``direct_weak_pressure`` is a cross-observer evaluated from an
    independently solved body potential.  The latter never replaces the
    primary state or the material trace.
    """

    material_current_trace: np.ndarray
    canonical_material_trace: np.ndarray
    active_trace: np.ndarray
    weak_pressure: np.ndarray
    direct_weak_pressure: np.ndarray
    pressure_cross_observer_difference: np.ndarray
    active_mass: np.ndarray
    stored_state: IndependentWakeBodyState
    direct_state: IndependentWakeBodyState
    direct_system: IndependentWakeSystem
    direct_assembly: DirectIndependentWakeAssembly
    stored_bie_residual: np.ndarray
    direct_bie_residual: np.ndarray
    stored_bie_backward_error: float
    direct_bie_backward_error: float
    body_matrix_condition_number: float
    body_matrix_condition_norm: str
    stored_condition_times_backward_error: float
    direct_condition_times_backward_error: float
    direct_minus_stored_body_potential: np.ndarray
    body_potential_difference_abs: float
    stored_material_body_trace_abs_residual: float
    stored_compatibility_abs_residual: float
    direct_full_bie_relative_residual: float
    compatibility_abs_residual: float
    zero_tip_abs_residual: float
    direct_w_factorization_abs_residual: float
    direct_w_rank_deficiency: int
    body_and_direct_w_quadrature_order: int


def _finite_vector(name: str, value: Any, size: int) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != (size,) or not np.all(np.isfinite(array)):
        raise ReachablePressureObservationError(
            f"{name} must be finite with shape {(size,)}, got {array.shape}"
        )
    return array.copy()


def active_p2_line_mass(
    solution: ActualBoundaryBodyWakeSolution,
    operators: CutRoleOperators | None = None,
) -> np.ndarray:
    """Return the active-active block of the complete consistent P2 mass.

    The complete trace uses the classified cut order, exactly the order used
    by the weak P2 pressure map.  Arc length rather than a coordinate sort is
    used so the material/trace orientation is never altered.
    """
    if not isinstance(solution, ActualBoundaryBodyWakeSolution):
        raise ReachablePressureObservationError(
            "solution must be an ActualBoundaryBodyWakeSolution"
        )
    roles = cut_role_operators(solution.topology) if operators is None else operators
    if not isinstance(roles, CutRoleOperators):
        raise ReachablePressureObservationError("operators must be CutRoleOperators")
    vertices = solution.mesh.vertices[
        solution.topology.ordered_cut_vertex_indices
    ]
    arclength = np.concatenate(
        (
            np.array((0.0,)),
            np.cumsum(np.linalg.norm(np.diff(vertices, axis=0), axis=1)),
        )
    )
    full_mass = consistent_p2_line_mass(arclength)
    active = roles.active_row_indices
    mass = full_mass[np.ix_(active, active)]
    if (
        mass.shape != (roles.independent_jump_count,) * 2
        or not np.all(np.isfinite(mass))
        or np.linalg.eigvalsh(mass)[0] <= 0.0
    ):
        raise ReachablePressureObservationError(
            "active P2 line mass is not finite positive definite"
        )
    return mass


def material_current_trace_from_surface(
    solution: ActualBoundaryBodyWakeSolution,
    attachment: MaterialWakeCutAttachment,
) -> np.ndarray:
    """Extract the newest material current row without reading row caches.

    The returned vector is in the material wake-chain order declared by
    ``attachment``.  No body-cut trace, coordinate sort, or nearest-neighbour
    inference participates.
    """

    if not isinstance(solution, ActualBoundaryBodyWakeSolution):
        raise ReachablePressureObservationError(
            "solution must be an ActualBoundaryBodyWakeSolution"
        )
    if not isinstance(attachment, MaterialWakeCutAttachment):
        raise ReachablePressureObservationError(
            "attachment must be a MaterialWakeCutAttachment"
        )
    newest = solution.wake_history.bands[-1]
    current_vertices = newest.surface.vertices[newest.span_nodes :]
    declared_vertices = solution.mesh.vertices[
        attachment.ordered_body_cut_vertex_indices
    ]
    if (
        current_vertices.shape != declared_vertices.shape
        or np.max(
            np.linalg.norm(current_vertices - declared_vertices, axis=1),
            initial=0.0,
        )
        > 1.0e-12
    ):
        raise ReachablePressureObservationError(
            "newest material edge is not attached through the typed "
            "body-cut identity"
        )
    return extract_surface_boundary_trace(newest, "current")


def canonical_material_trace(
    solution: ActualBoundaryBodyWakeSolution,
    attachment: MaterialWakeCutAttachment,
    material_trace: Any,
) -> np.ndarray:
    """Map ``w=s*P*c`` from material order back to canonical cut order."""

    if not isinstance(solution, ActualBoundaryBodyWakeSolution):
        raise ReachablePressureObservationError(
            "solution must be an ActualBoundaryBodyWakeSolution"
        )
    if not isinstance(attachment, MaterialWakeCutAttachment):
        raise ReachablePressureObservationError(
            "attachment must be a MaterialWakeCutAttachment"
        )
    full_count = len(solution.topology.cut_node_coordinates)
    material = _finite_vector(
        "material_trace",
        material_trace,
        full_count,
    )
    permutation = attachment.p2_trace_permutation(solution.topology)
    canonical = np.empty_like(material)
    canonical[permutation] = (
        attachment.wake_jump_from_body_cut_sign * material
    )
    return canonical


def direct_independent_system(
    solution: ActualBoundaryBodyWakeSolution,
    *,
    attachment: MaterialWakeCutAttachment,
    body_and_direct_w_quadrature_order: int,
) -> tuple[IndependentWakeSystem, DirectIndependentWakeAssembly, CutRoleOperators]:
    """Build ``B phi + W_direct g = b`` at one externally typed order."""
    if not isinstance(solution, ActualBoundaryBodyWakeSolution):
        raise ReachablePressureObservationError(
            "solution must be an ActualBoundaryBodyWakeSolution"
        )
    if not isinstance(attachment, MaterialWakeCutAttachment):
        raise ReachablePressureObservationError(
            "attachment must be a MaterialWakeCutAttachment"
        )
    if (
        not isinstance(
            body_and_direct_w_quadrature_order,
            (int, np.integer),
        )
        or int(body_and_direct_w_quadrature_order) < 2
    ):
        raise ReachablePressureObservationError(
            "body/direct-W quadrature order must be an integer >=2"
        )
    order = int(body_and_direct_w_quadrature_order)
    operators = cut_role_operators(solution.topology)
    direct = assemble_direct_independent_wake_matrix(
        solution.mesh,
        solution.topology,
        solution.wake_history,
        prescribed_wake_attachment=attachment,
        target_quadrature_order=order,
        source_quadrature_order=order,
    )
    system = IndependentWakeSystem(
        body_matrix=np.asarray(solution.body_matrix, dtype=float).copy(),
        independent_wake_matrix=direct.matrix.copy(),
        active_jump=operators.active_jump.copy(),
        right_hand_side=np.asarray(solution.right_hand_side, dtype=float).copy(),
        eliminated_wake_matrix=np.asarray(solution.wake_matrix, dtype=float).copy(),
        eliminated_matrix=np.asarray(solution.matrix, dtype=float).copy(),
        wake_factorization_error=float(
            np.max(
                np.abs(direct.eliminated_wake_matrix - solution.wake_matrix),
                initial=0.0,
            )
        ),
    )
    return system, direct, operators


def _componentwise_bie_backward_error(
    body: np.ndarray,
    wake: np.ndarray,
    body_potential: np.ndarray,
    active_trace: np.ndarray,
    right_hand_side: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Return residual and max componentwise backward error for ``Bφ+Wg=b``."""

    residual = body @ body_potential + wake @ active_trace - right_hand_side
    denominator = (
        np.abs(body) @ np.abs(body_potential)
        + np.abs(wake) @ np.abs(active_trace)
        + np.abs(right_hand_side)
    )
    ratios = np.zeros_like(residual)
    nonzero = denominator > 0.0
    ratios[nonzero] = np.abs(residual[nonzero]) / denominator[nonzero]
    ratios[~nonzero & (residual != 0.0)] = np.inf
    return residual, float(np.max(ratios, initial=0.0))


def _value_only_pressure_model(
    solution: ActualBoundaryBodyWakeSolution,
    system: IndependentWakeSystem,
    operators: CutRoleOperators,
    *,
    upper_face_indices: Iterable[int],
    lower_face_indices: Iterable[int],
    line_quadrature_order: int,
) -> ActualPressureKuttaModel:
    """Build pressure geometry/rule only, without any linear body solve."""

    if (
        not isinstance(line_quadrature_order, (int, np.integer))
        or int(line_quadrature_order) < 2
    ):
        raise ReachablePressureObservationError(
            "pressure line quadrature order must be an integer >=2"
        )
    if np.max(np.abs(solution.wall_velocity), initial=0.0) > 0.0:
        raise ReachablePressureObservationError(
            "stored pressure value requires a fixed wall in S3ai-v2"
        )
    abscissa, weights = np.polynomial.legendre.leggauss(
        int(line_quadrature_order)
    )
    coordinate = 0.5 * (abscissa + 1.0)
    return ActualPressureKuttaModel(
        solution=solution,
        system=system,
        operators=operators,
        face_pairs=trailing_edge_face_pairs(
            solution,
            upper_face_indices=upper_face_indices,
            lower_face_indices=lower_face_indices,
        ),
        line_coordinates=coordinate,
        line_weights=0.5 * weights,
        # S3ai-v2 observes pressure values, not a reduced pressure
        # Jacobian.  A zero sensitivity keeps the legacy typed state usable
        # by the value-only segment evaluator without solving B^{-1}W.
        body_potential_wake_jacobian=np.zeros(
            (
                solution.body_unknown_count,
                operators.independent_jump_count,
            ),
            dtype=float,
        ),
    )


def _typed_body_state(
    model: ActualPressureKuttaModel,
    active_trace: np.ndarray,
    body_potential: np.ndarray,
    bie_residual: np.ndarray,
) -> IndependentWakeBodyState:
    """Construct an observation state from an already available body trace."""

    return IndependentWakeBodyState(
        wake_jump=active_trace.copy(),
        body_potential=body_potential.copy(),
        body_potential_wake_jacobian=(
            model.body_potential_wake_jacobian.copy()
        ),
        full_bie_residual=bie_residual.copy(),
        edge_compatibility_defect=(
            active_trace
            - model.operators.active_jump @ body_potential
        ),
    )


def _weak_active_pressure_from_state(
    model: ActualPressureKuttaModel,
    state: IndependentWakeBodyState,
) -> np.ndarray:
    """Evaluate weak active-P2 pressure from a supplied typed body state.

    This is intentionally the value-only part of the frozen observation.  It
    never solves for ``state.body_potential``.
    """

    full = np.zeros(model.operators.full_cut_node_count, dtype=float)
    shape = _line_p2_shape(model.line_coordinates)
    for pair in model.face_pairs:
        pressure, _pressure_jacobian = _segment_pressure_jump(
            model,
            state,
            pair,
            model.line_coordinates,
        )
        measure = pair.segment_length * model.line_weights
        rows = np.array(
            (
                2 * pair.segment_index,
                2 * pair.segment_index + 1,
                2 * pair.segment_index + 2,
            ),
            dtype=np.int64,
        )
        full[rows] += shape.T @ (measure * pressure)
    active = full[model.operators.active_row_indices]
    if not np.all(np.isfinite(active)):
        raise ReachablePressureObservationError(
            "weak stored-state pressure contains non-finite values"
        )
    return active


def observe_direct_independent_stage(
    solution: ActualBoundaryBodyWakeSolution,
    *,
    attachment: MaterialWakeCutAttachment,
    upper_face_indices: Iterable[int],
    lower_face_indices: Iterable[int],
    body_and_direct_w_quadrature_order: int,
    pressure_line_quadrature_order: int,
    material_current_trace: Any | None = None,
) -> DirectIndependentStageObservation:
    """Observe one S3e stage without re-solving its primary pressure state.

    If ``material_current_trace`` is omitted, it is extracted exclusively
    from the newest ``surface.face_mu`` boundary.  If supplied, it must be
    the complete trace in the typed material wake-chain order.
    """
    system, direct, operators = direct_independent_system(
        solution,
        attachment=attachment,
        body_and_direct_w_quadrature_order=(
            body_and_direct_w_quadrature_order
        ),
    )
    material = (
        material_current_trace_from_surface(solution, attachment)
        if material_current_trace is None
        else _finite_vector(
            "material_current_trace",
            material_current_trace,
            operators.full_cut_node_count,
        )
    )
    canonical = canonical_material_trace(solution, attachment, material)
    active_trace = canonical[operators.active_row_indices].copy()
    model = _value_only_pressure_model(
        solution,
        system,
        operators,
        upper_face_indices=upper_face_indices,
        lower_face_indices=lower_face_indices,
        line_quadrature_order=pressure_line_quadrature_order,
    )
    body = np.asarray(system.body_matrix, dtype=float)
    wake = np.asarray(system.independent_wake_matrix, dtype=float)
    right_hand_side = np.asarray(system.right_hand_side, dtype=float)
    stored_phi = _finite_vector(
        "solution.global_body_potential",
        solution.global_body_potential,
        solution.body_unknown_count,
    )
    stored_residual, stored_backward_error = (
        _componentwise_bie_backward_error(
            body,
            wake,
            stored_phi,
            active_trace,
            right_hand_side,
        )
    )
    stored_state = _typed_body_state(
        model,
        active_trace,
        stored_phi,
        stored_residual,
    )
    # The primary pressure is complete before the independent audit solve.
    stored_pressure = _weak_active_pressure_from_state(model, stored_state)
    try:
        direct_phi = np.linalg.solve(
            body,
            right_hand_side - wake @ active_trace,
        )
    except np.linalg.LinAlgError as error:
        raise ReachablePressureObservationError(
            "independent direct body audit solve failed"
        ) from error
    direct_residual, direct_backward_error = (
        _componentwise_bie_backward_error(
            body,
            wake,
            direct_phi,
            active_trace,
            right_hand_side,
        )
    )
    direct_state = _typed_body_state(
        model,
        active_trace,
        direct_phi,
        direct_residual,
    )
    direct_pressure = _weak_active_pressure_from_state(model, direct_state)
    cut_from_stored = solution.topology.cut_jump(stored_phi)
    body_condition = float(np.linalg.cond(body, p=2))
    if not np.isfinite(body_condition):
        raise ReachablePressureObservationError(
            "direct body matrix condition number is non-finite"
        )
    difference = direct_phi - stored_phi
    denominator = max(
        float(np.max(np.abs(right_hand_side), initial=0.0)),
        1.0,
    )
    return DirectIndependentStageObservation(
        material_current_trace=material.copy(),
        canonical_material_trace=canonical.copy(),
        active_trace=active_trace,
        weak_pressure=stored_pressure,
        direct_weak_pressure=direct_pressure,
        pressure_cross_observer_difference=(
            direct_pressure - stored_pressure
        ),
        active_mass=active_p2_line_mass(solution, operators),
        stored_state=stored_state,
        direct_state=direct_state,
        direct_system=system,
        direct_assembly=direct,
        stored_bie_residual=stored_residual,
        direct_bie_residual=direct_residual,
        stored_bie_backward_error=stored_backward_error,
        direct_bie_backward_error=direct_backward_error,
        body_matrix_condition_number=body_condition,
        body_matrix_condition_norm="spectral_2",
        stored_condition_times_backward_error=(
            body_condition * stored_backward_error
        ),
        direct_condition_times_backward_error=(
            body_condition * direct_backward_error
        ),
        direct_minus_stored_body_potential=difference,
        body_potential_difference_abs=float(
            np.max(np.abs(difference), initial=0.0)
        ),
        stored_material_body_trace_abs_residual=float(
            np.max(
                np.abs(canonical - cut_from_stored),
                initial=0.0,
            )
        ),
        stored_compatibility_abs_residual=float(
            np.max(
                np.abs(stored_state.edge_compatibility_defect),
                initial=0.0,
            )
        ),
        direct_full_bie_relative_residual=float(
            np.max(np.abs(direct_residual), initial=0.0) / denominator
        ),
        compatibility_abs_residual=float(
            np.max(
                np.abs(direct_state.edge_compatibility_defect),
                initial=0.0,
            )
        ),
        zero_tip_abs_residual=float(
            np.max(
                np.abs(cut_from_stored[operators.zero_row_indices]),
                initial=0.0,
            )
        ),
        direct_w_factorization_abs_residual=system.wake_factorization_error,
        direct_w_rank_deficiency=(
            operators.independent_jump_count - int(direct.rank)
        ),
        body_and_direct_w_quadrature_order=int(
            body_and_direct_w_quadrature_order
        ),
    )


def weak_pressure_step_residual(
    mass_active: Any,
    g_previous: Any,
    g_current: Any,
    timestep: float,
    pressure_midpoint: Any,
) -> np.ndarray:
    """Return the frozen S3ai ``M(g1-g0) + dt P_mid`` observation."""
    mass = np.asarray(mass_active, dtype=float)
    if (
        mass.ndim != 2
        or mass.shape[0] != mass.shape[1]
        or not np.all(np.isfinite(mass))
    ):
        raise ReachablePressureObservationError("mass_active must be finite square")
    size = mass.shape[0]
    previous = _finite_vector("g_previous", g_previous, size)
    current = _finite_vector("g_current", g_current, size)
    pressure = _finite_vector("pressure_midpoint", pressure_midpoint, size)
    dt = float(timestep)
    if not np.isfinite(dt) or dt <= 0.0:
        raise ReachablePressureObservationError("timestep must be finite and positive")
    return mass @ (current - previous) + dt * pressure


def dual_mass_norm(residual: Any, mass_active: Any) -> float:
    """Return ``sqrt(R.T M_active^-1 R)`` without forming an inverse."""
    mass = np.asarray(mass_active, dtype=float)
    if (
        mass.ndim != 2
        or mass.shape[0] != mass.shape[1]
        or not np.all(np.isfinite(mass))
    ):
        raise ReachablePressureObservationError(
            "mass_active must be a finite square matrix"
        )
    vector = _finite_vector("residual", residual, mass.shape[0])
    try:
        value = float(vector @ np.linalg.solve(mass, vector))
    except np.linalg.LinAlgError as error:
        raise ReachablePressureObservationError("mass_active must be nonsingular") from error
    if value < -64.0 * np.finfo(float).eps:
        raise ReachablePressureObservationError("mass_active is not positive definite")
    return float(np.sqrt(max(value, 0.0)))


def centered_tangent(positive: Any, negative: Any, epsilon: float) -> np.ndarray:
    """Return the frozen centered physical-incidence tangent."""
    plus = np.asarray(positive, dtype=float)
    minus = np.asarray(negative, dtype=float)
    eps = float(epsilon)
    if (
        plus.shape != minus.shape
        or plus.ndim != 1
        or not np.all(np.isfinite(plus))
        or not np.all(np.isfinite(minus))
        or not np.isfinite(eps)
        or eps <= 0.0
    ):
        raise ReachablePressureObservationError(
            "centered tangent requires same finite vectors and positive epsilon"
        )
    return (plus - minus) / (2.0 * eps)
