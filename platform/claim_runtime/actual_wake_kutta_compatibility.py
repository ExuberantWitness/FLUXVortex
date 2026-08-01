"""No-force compatibility observations for independent-wake Kutta closures.

The S3af equation-role audit exposes the current wake trace ``g`` as an
independent active P2 field:

    B phi + W g = b.

This module deliberately does not choose a production closure.  It provides
two reproducible observations needed by the S3ag counterexample gate:

* the body-to-wake edge defect ``g - C phi`` after solving every body BIE;
* a steady, gauge-free pressure-Kutta residual evaluated from the two
  incident P2 surface-gradient limits at the classified trailing-edge cut.

The pressure residual can be represented either by a consistent weak P2 line
test or by midpoint/two-sided-vertex-average collocation.  Both use an
analytic reduced Jacobian through ``d phi / d g = -B^{-1} W``.  No pressure is
integrated into force, and no regularizer, least-squares solve, amplitude
clamp, core, smoothing, or production state is present.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from .actual_boundary_body_wake import ActualBoundaryBodyWakeSolution
from .actual_wake_kutta_closure_roles import (
    CutRoleOperators,
    IndependentWakeSystem,
)
from .distributed_doublet import QuadraticDoubletElement


class KuttaCompatibilityError(ValueError):
    """Invalid topology, observation, or nonlinear closure state."""


@dataclass(frozen=True)
class TrailingEdgeFacePair:
    """The upper/lower incident body faces for one ordered cut segment."""

    segment_index: int
    start_vertex_index: int
    end_vertex_index: int
    upper_face_index: int
    lower_face_index: int
    segment_length: float


@dataclass(frozen=True)
class IndependentWakeBodyState:
    """Body solution and exact sensitivity for one prescribed active wake."""

    wake_jump: np.ndarray
    body_potential: np.ndarray
    body_potential_wake_jacobian: np.ndarray
    full_bie_residual: np.ndarray
    edge_compatibility_defect: np.ndarray


@dataclass(frozen=True)
class PressureKuttaEvaluation:
    """One reduced pressure observation at a prescribed active wake trace."""

    observation_map: str
    state: IndependentWakeBodyState
    residual: np.ndarray
    jacobian: np.ndarray
    dense_pressure_jump: np.ndarray
    dense_pressure_jump_jacobian: np.ndarray
    jacobian_rank: int
    jacobian_condition_number: float


@dataclass(frozen=True)
class PressureKuttaRoot:
    """A Newton root of one pressure observation map."""

    observation_map: str
    converged: bool
    iterations: int
    evaluation: PressureKuttaEvaluation
    residual_history: np.ndarray
    accepted_step_lengths: np.ndarray
    maximum_jacobian_condition_number: float


@dataclass(frozen=True)
class ActualPressureKuttaModel:
    """Fixed-geometry actual-boundary pressure observation model."""

    solution: ActualBoundaryBodyWakeSolution
    system: IndependentWakeSystem
    operators: CutRoleOperators
    face_pairs: tuple[TrailingEdgeFacePair, ...]
    line_coordinates: np.ndarray
    line_weights: np.ndarray
    body_potential_wake_jacobian: np.ndarray

    @property
    def independent_wake_count(self) -> int:
        return int(self.system.independent_wake_matrix.shape[1])


_LOCAL_EDGES = ((0, 1), (1, 2), (2, 0))
_OBSERVATION_MAPS = {
    "weak_active_p2_line_mass",
    "midpoint_and_two_sided_vertex_average_collocation",
}


def _finite_vector(name: str, value: Any, *, size: int) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != (size,) or not np.all(np.isfinite(array)):
        raise KuttaCompatibilityError(
            f"{name} must be finite with shape {(size,)}, got {array.shape}"
        )
    return array


def _finite_index_set(
    name: str,
    values: Iterable[int],
    *,
    upper_bound: int,
) -> set[int]:
    try:
        result = {int(value) for value in values}
    except (TypeError, ValueError) as error:
        raise KuttaCompatibilityError(
            f"{name} must contain integer face indices"
        ) from error
    if (
        not result
        or any(index < 0 or index >= upper_bound for index in result)
    ):
        raise KuttaCompatibilityError(
            f"{name} contains an invalid face index"
        )
    return result


def trailing_edge_face_pairs(
    solution: ActualBoundaryBodyWakeSolution,
    *,
    upper_face_indices: Iterable[int],
    lower_face_indices: Iterable[int],
) -> tuple[TrailingEdgeFacePair, ...]:
    """Resolve exactly one upper/lower incident face per ordered cut edge."""
    if not isinstance(solution, ActualBoundaryBodyWakeSolution):
        raise KuttaCompatibilityError(
            "solution must be an ActualBoundaryBodyWakeSolution"
        )
    mesh = solution.mesh
    topology = solution.topology
    upper = _finite_index_set(
        "upper_face_indices",
        upper_face_indices,
        upper_bound=len(mesh.faces),
    )
    lower = _finite_index_set(
        "lower_face_indices",
        lower_face_indices,
        upper_bound=len(mesh.faces),
    )
    if upper & lower:
        raise KuttaCompatibilityError(
            "upper/lower face classifications must be disjoint"
        )

    pairs: list[TrailingEdgeFacePair] = []
    ordered_vertices = topology.ordered_cut_vertex_indices
    for segment_index, (start, end) in enumerate(
        zip(ordered_vertices[:-1], ordered_vertices[1:], strict=True)
    ):
        start_index = int(start)
        end_index = int(end)
        edge_vertices = {start_index, end_index}
        incident = [
            face_index
            for face_index, face in enumerate(mesh.faces)
            if edge_vertices.issubset(set(map(int, face)))
        ]
        upper_incident = [
            face_index for face_index in incident if face_index in upper
        ]
        lower_incident = [
            face_index for face_index in incident if face_index in lower
        ]
        if len(incident) != 2 or len(upper_incident) != 1 or len(
            lower_incident
        ) != 1:
            raise KuttaCompatibilityError(
                "each ordered cut edge must have exactly one classified "
                "upper and one classified lower incident face"
            )

        upper_face = upper_incident[0]
        lower_face = lower_incident[0]
        expected_upper = {
            int(topology.upper_cut_dofs[2 * segment_index]),
            int(topology.upper_cut_dofs[2 * segment_index + 1]),
            int(topology.upper_cut_dofs[2 * segment_index + 2]),
        }
        expected_lower = {
            int(topology.lower_cut_dofs[2 * segment_index]),
            int(topology.lower_cut_dofs[2 * segment_index + 1]),
            int(topology.lower_cut_dofs[2 * segment_index + 2]),
        }

        def edge_trace_dofs(face_index: int) -> set[int]:
            face = mesh.faces[face_index]
            local_vertices = {
                int(np.flatnonzero(face == start_index)[0]),
                int(np.flatnonzero(face == end_index)[0]),
            }
            local_edge = next(
                (
                    local_index
                    for local_index, endpoints in enumerate(_LOCAL_EDGES)
                    if set(endpoints) == local_vertices
                ),
                None,
            )
            if local_edge is None:
                raise KuttaCompatibilityError(
                    "cut edge is not a local triangle edge"
                )
            return {
                int(solution.topology.local_to_global[face_index, local])
                for local in (*local_vertices, 3 + local_edge)
            }

        if (
            edge_trace_dofs(upper_face) != expected_upper
            or edge_trace_dofs(lower_face) != expected_lower
        ):
            raise KuttaCompatibilityError(
                "incident face trace does not match classified cut DOFs"
            )
        length = float(
            np.linalg.norm(
                mesh.vertices[end_index] - mesh.vertices[start_index]
            )
        )
        if not np.isfinite(length) or length <= np.finfo(float).tiny:
            raise KuttaCompatibilityError(
                "cut segment must be finite and nondegenerate"
            )
        pairs.append(
            TrailingEdgeFacePair(
                segment_index=segment_index,
                start_vertex_index=start_index,
                end_vertex_index=end_index,
                upper_face_index=upper_face,
                lower_face_index=lower_face,
                segment_length=length,
            )
        )
    return tuple(pairs)


def build_actual_pressure_kutta_model(
    solution: ActualBoundaryBodyWakeSolution,
    system: IndependentWakeSystem,
    operators: CutRoleOperators,
    *,
    upper_face_indices: Iterable[int],
    lower_face_indices: Iterable[int],
    line_quadrature_order: int,
) -> ActualPressureKuttaModel:
    """Build the fixed actual-boundary observation and reduced sensitivity."""
    if not isinstance(system, IndependentWakeSystem):
        raise KuttaCompatibilityError(
            "system must be an IndependentWakeSystem"
        )
    if not isinstance(operators, CutRoleOperators):
        raise KuttaCompatibilityError(
            "operators must be CutRoleOperators"
        )
    if (
        not isinstance(line_quadrature_order, (int, np.integer))
        or int(line_quadrature_order) < 2
    ):
        raise KuttaCompatibilityError(
            "line_quadrature_order must be an integer >=2"
        )
    if np.max(
        np.abs(solution.wall_velocity),
        initial=0.0,
    ) > 0.0:
        raise KuttaCompatibilityError(
            "the S3ag pressure observation is steady-only; a moving wall "
            "requires the two-sided unsteady potential-rate term"
        )
    body = np.asarray(system.body_matrix, dtype=float)
    wake = np.asarray(system.independent_wake_matrix, dtype=float)
    if (
        body.shape
        != (solution.body_unknown_count, solution.body_unknown_count)
        or wake.shape[0] != solution.body_unknown_count
        or wake.shape[1] != operators.independent_jump_count
        or not np.all(np.isfinite(body))
        or not np.all(np.isfinite(wake))
    ):
        raise KuttaCompatibilityError(
            "independent wake system does not match the actual body solution"
        )
    try:
        body_wake_jacobian = -np.linalg.solve(body, wake)
    except np.linalg.LinAlgError as error:
        raise KuttaCompatibilityError(
            "body matrix cannot form the reduced wake sensitivity"
        ) from error
    abscissa, weights = np.polynomial.legendre.leggauss(
        int(line_quadrature_order)
    )
    coordinate = 0.5 * (abscissa + 1.0)
    line_weights = 0.5 * weights
    pairs = trailing_edge_face_pairs(
        solution,
        upper_face_indices=upper_face_indices,
        lower_face_indices=lower_face_indices,
    )
    return ActualPressureKuttaModel(
        solution=solution,
        system=system,
        operators=operators,
        face_pairs=pairs,
        line_coordinates=coordinate,
        line_weights=line_weights,
        body_potential_wake_jacobian=body_wake_jacobian,
    )


def solve_independent_wake_body_state(
    model: ActualPressureKuttaModel,
    wake_jump: Any,
) -> IndependentWakeBodyState:
    """Solve all body BIE rows for a prescribed active current-wake trace."""
    if not isinstance(model, ActualPressureKuttaModel):
        raise KuttaCompatibilityError(
            "model must be an ActualPressureKuttaModel"
        )
    wake_state = _finite_vector(
        "wake_jump",
        wake_jump,
        size=model.independent_wake_count,
    )
    body = np.asarray(model.system.body_matrix, dtype=float)
    wake = np.asarray(model.system.independent_wake_matrix, dtype=float)
    right_hand_side = np.asarray(
        model.system.right_hand_side,
        dtype=float,
    )
    try:
        body_potential = np.linalg.solve(
            body,
            right_hand_side - wake @ wake_state,
        )
    except np.linalg.LinAlgError as error:
        raise KuttaCompatibilityError(
            "full body BIE solve failed for the prescribed wake trace"
        ) from error
    residual = body @ body_potential + wake @ wake_state
    residual -= right_hand_side
    defect = wake_state - model.operators.active_jump @ body_potential
    return IndependentWakeBodyState(
        wake_jump=wake_state.copy(),
        body_potential=body_potential,
        body_potential_wake_jacobian=(
            model.body_potential_wake_jacobian.copy()
        ),
        full_bie_residual=residual,
        edge_compatibility_defect=defect,
    )


def _line_p2_shape(coordinate: np.ndarray) -> np.ndarray:
    t = np.asarray(coordinate, dtype=float)
    return np.column_stack(
        (
            (1.0 - t) * (1.0 - 2.0 * t),
            4.0 * t * (1.0 - t),
            t * (2.0 * t - 1.0),
        )
    )


def _face_velocity_and_jacobian(
    model: ActualPressureKuttaModel,
    state: IndependentWakeBodyState,
    *,
    face_index: int,
    points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    solution = model.solution
    face = solution.mesh.faces[face_index]
    local_dofs = solution.topology.local_to_global[face_index]
    element = QuadraticDoubletElement(
        solution.mesh.vertices[face],
        state.body_potential[local_dofs],
    )
    barycentric = element.barycentric_coordinates(
        points,
        plane_tolerance=2.0e-12,
    )
    gradient = element.surface_gradient_barycentric(barycentric)
    shape_gradient = element.shape_gradients(barycentric)
    gradient_jacobian = np.einsum(
        "qic,ij->qcj",
        shape_gradient,
        state.body_potential_wake_jacobian[local_dofs],
    )
    normal = solution.mesh.normals[face_index]
    incident = solution.incident_velocity[face_index]
    wall = solution.wall_velocity[face_index]
    tangent_incident = incident - float(incident @ normal) * normal
    wall_normal = float(wall @ normal) * normal
    velocity = tangent_incident[None, :] - gradient + wall_normal[None, :]
    velocity_jacobian = -gradient_jacobian
    return velocity, velocity_jacobian


def _segment_pressure_jump(
    model: ActualPressureKuttaModel,
    state: IndependentWakeBodyState,
    pair: TrailingEdgeFacePair,
    coordinate: Any,
) -> tuple[np.ndarray, np.ndarray]:
    t = np.atleast_1d(np.asarray(coordinate, dtype=float))
    if (
        t.ndim != 1
        or not np.all(np.isfinite(t))
        or np.any(t < 0.0)
        or np.any(t > 1.0)
    ):
        raise KuttaCompatibilityError(
            "segment coordinate must lie in [0,1]"
        )
    start = model.solution.mesh.vertices[pair.start_vertex_index]
    end = model.solution.mesh.vertices[pair.end_vertex_index]
    points = (1.0 - t[:, None]) * start + t[:, None] * end
    upper_velocity, upper_jacobian = _face_velocity_and_jacobian(
        model,
        state,
        face_index=pair.upper_face_index,
        points=points,
    )
    lower_velocity, lower_jacobian = _face_velocity_and_jacobian(
        model,
        state,
        face_index=pair.lower_face_index,
        points=points,
    )
    # The sign is upper-minus-lower specific dynamic pressure.  Only its
    # zero set is used in this no-force closure observation.
    pressure_jump = 0.5 * (
        np.einsum("qi,qi->q", upper_velocity, upper_velocity)
        - np.einsum("qi,qi->q", lower_velocity, lower_velocity)
    )
    pressure_jacobian = (
        np.einsum("qi,qij->qj", upper_velocity, upper_jacobian)
        - np.einsum("qi,qij->qj", lower_velocity, lower_jacobian)
    )
    return pressure_jump, pressure_jacobian


def evaluate_pressure_kutta(
    model: ActualPressureKuttaModel,
    wake_jump: Any,
    *,
    observation_map: str,
) -> PressureKuttaEvaluation:
    """Evaluate one pressure residual and its analytic reduced Jacobian."""
    if observation_map not in _OBSERVATION_MAPS:
        raise KuttaCompatibilityError(
            f"unknown pressure observation map {observation_map!r}"
        )
    state = solve_independent_wake_body_state(model, wake_jump)
    full_cut_count = model.operators.full_cut_node_count
    active_rows = model.operators.active_row_indices
    active_count = model.independent_wake_count
    dense_values: list[np.ndarray] = []
    dense_jacobians: list[np.ndarray] = []

    if observation_map == "weak_active_p2_line_mass":
        full_residual = np.zeros(full_cut_count, dtype=float)
        full_jacobian = np.zeros(
            (full_cut_count, active_count),
            dtype=float,
        )
        shape = _line_p2_shape(model.line_coordinates)
        for pair in model.face_pairs:
            pressure, pressure_jacobian = _segment_pressure_jump(
                model,
                state,
                pair,
                model.line_coordinates,
            )
            measure = pair.segment_length * model.line_weights
            local_rows = np.array(
                (
                    2 * pair.segment_index,
                    2 * pair.segment_index + 1,
                    2 * pair.segment_index + 2,
                ),
                dtype=np.int64,
            )
            full_residual[local_rows] += shape.T @ (measure * pressure)
            full_jacobian[local_rows] += shape.T @ (
                measure[:, None] * pressure_jacobian
            )
            dense_values.append(pressure)
            dense_jacobians.append(pressure_jacobian)
        residual = full_residual[active_rows]
        jacobian = full_jacobian[active_rows]
    else:
        residual = np.empty(active_count, dtype=float)
        jacobian = np.empty((active_count, active_count), dtype=float)
        for active_index, full_index in enumerate(active_rows):
            full_node = int(full_index)
            if full_node % 2 == 1:
                segment = (full_node - 1) // 2
                value, derivative = _segment_pressure_jump(
                    model,
                    state,
                    model.face_pairs[segment],
                    np.array((0.5,)),
                )
                residual[active_index] = value[0]
                jacobian[active_index] = derivative[0]
            else:
                vertex = full_node // 2
                left_value, left_derivative = _segment_pressure_jump(
                    model,
                    state,
                    model.face_pairs[vertex - 1],
                    np.array((1.0,)),
                )
                right_value, right_derivative = _segment_pressure_jump(
                    model,
                    state,
                    model.face_pairs[vertex],
                    np.array((0.0,)),
                )
                residual[active_index] = 0.5 * (
                    left_value[0] + right_value[0]
                )
                jacobian[active_index] = 0.5 * (
                    left_derivative[0] + right_derivative[0]
                )

        # Dense pressure is always evaluated with the preregistered line
        # rule, not only at the collocation locations.
        for pair in model.face_pairs:
            pressure, pressure_jacobian = _segment_pressure_jump(
                model,
                state,
                pair,
                model.line_coordinates,
            )
            dense_values.append(pressure)
            dense_jacobians.append(pressure_jacobian)

    dense = np.concatenate(dense_values)
    dense_jacobian = np.concatenate(dense_jacobians, axis=0)
    rank = int(np.linalg.matrix_rank(jacobian))
    condition = float(np.linalg.cond(jacobian))
    if (
        residual.shape != (active_count,)
        or jacobian.shape != (active_count, active_count)
        or not np.all(np.isfinite(residual))
        or not np.all(np.isfinite(jacobian))
        or not np.all(np.isfinite(dense))
        or not np.all(np.isfinite(dense_jacobian))
    ):
        raise KuttaCompatibilityError(
            "pressure observation produced a non-finite or invalid output"
        )
    return PressureKuttaEvaluation(
        observation_map=observation_map,
        state=state,
        residual=residual,
        jacobian=jacobian,
        dense_pressure_jump=dense,
        dense_pressure_jump_jacobian=dense_jacobian,
        jacobian_rank=rank,
        jacobian_condition_number=condition,
    )


def solve_pressure_kutta_newton(
    model: ActualPressureKuttaModel,
    initial_wake_jump: Any,
    *,
    observation_map: str,
    residual_tolerance: float,
    maximum_iterations: int,
    maximum_backtracking_steps: int = 24,
) -> PressureKuttaRoot:
    """Solve one pressure observation with analytic Newton/backtracking.

    Backtracking accepts only a strict decrease of the infinity norm.  It is
    a deterministic globalization of the Newton step, not an amplitude
    penalty or a physical branch-selection rule.
    """
    tolerance = float(residual_tolerance)
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise KuttaCompatibilityError(
            "residual_tolerance must be finite and positive"
        )
    if (
        not isinstance(maximum_iterations, (int, np.integer))
        or int(maximum_iterations) < 1
        or not isinstance(maximum_backtracking_steps, (int, np.integer))
        or int(maximum_backtracking_steps) < 1
    ):
        raise KuttaCompatibilityError(
            "iteration and backtracking limits must be positive integers"
        )
    wake = _finite_vector(
        "initial_wake_jump",
        initial_wake_jump,
        size=model.independent_wake_count,
    ).copy()
    residual_history: list[float] = []
    accepted_steps: list[float] = []
    maximum_condition = 0.0
    converged = False
    evaluation: PressureKuttaEvaluation | None = None

    for iteration in range(int(maximum_iterations) + 1):
        evaluation = evaluate_pressure_kutta(
            model,
            wake,
            observation_map=observation_map,
        )
        residual_norm = float(
            np.max(np.abs(evaluation.residual), initial=0.0)
        )
        residual_history.append(residual_norm)
        maximum_condition = max(
            maximum_condition,
            evaluation.jacobian_condition_number,
        )
        if residual_norm <= tolerance:
            converged = True
            break
        if iteration == int(maximum_iterations):
            break
        if evaluation.jacobian_rank != model.independent_wake_count:
            raise KuttaCompatibilityError(
                f"{observation_map} reduced Jacobian is rank deficient"
            )
        try:
            newton_step = np.linalg.solve(
                evaluation.jacobian,
                -evaluation.residual,
            )
        except np.linalg.LinAlgError as error:
            raise KuttaCompatibilityError(
                f"{observation_map} Newton Jacobian solve failed"
            ) from error

        accepted = False
        step_length = 1.0
        for _backtrack in range(int(maximum_backtracking_steps)):
            candidate = wake + step_length * newton_step
            candidate_evaluation = evaluate_pressure_kutta(
                model,
                candidate,
                observation_map=observation_map,
            )
            candidate_norm = float(
                np.max(
                    np.abs(candidate_evaluation.residual),
                    initial=0.0,
                )
            )
            if candidate_norm < residual_norm:
                wake = candidate
                accepted_steps.append(step_length)
                accepted = True
                break
            step_length *= 0.5
        if not accepted:
            raise KuttaCompatibilityError(
                f"{observation_map} Newton backtracking found no "
                "strict residual decrease"
            )

    if evaluation is None:
        raise KuttaCompatibilityError(
            "pressure Newton produced no evaluation"
        )
    return PressureKuttaRoot(
        observation_map=observation_map,
        converged=converged,
        iterations=len(accepted_steps),
        evaluation=evaluation,
        residual_history=np.asarray(residual_history, dtype=float),
        accepted_step_lengths=np.asarray(accepted_steps, dtype=float),
        maximum_jacobian_condition_number=maximum_condition,
    )
