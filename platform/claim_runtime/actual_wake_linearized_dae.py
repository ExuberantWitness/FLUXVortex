"""Fixed-geometry linearized actual-wake differential-algebraic oracle.

For one algebraically consistent actual body/wake solution, this module
builds the exact affine map from the free global-P2 wake state ``y`` to the
body-attachment trace ``g`` by unit-basis superposition:

    g = G y + c.

Combining that map with one already assembled actual ALE transport operator
gives the reduced local DAE

    (M_ff + M_fb G) y_dot
        = -(C_ff + C_fb G)y - C_fb c.

The implementation contains no finite-difference Jacobian, geometry update,
pressure, force, target, damping or production activation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.linalg import expm

from .actual_boundary_body_wake import (
    ActualBoundaryBodyWakeSolution,
    MaterialWakeCutAttachment,
    solve_actual_boundary_body_wake_p2,
)
from .actual_wake_stage_topology import (
    ActualWakeStageTopology,
    ActualWakeStageTopologyError,
)
from .p2_surface_material_transport import (
    P2SurfaceMaterialTransportOperator,
)
from .thick_body_neumann_shadow import ClosedTriangularMesh


@dataclass(frozen=True)
class ActualWakeAffineTraceReport:
    basis_solve_count: int
    maximum_relative_weak_residual: float
    base_reconstruction_error: float
    effective_mass_rank: int
    effective_mass_condition_number: float


@dataclass(frozen=True)
class ActualWakeLinearizedDAE:
    """Exact local affine DAE on the free actual-wake P2 state."""

    topology: ActualWakeStageTopology
    free_dof_indices: np.ndarray
    body_dof_indices: np.ndarray
    base_global_state: np.ndarray
    trace_matrix: np.ndarray
    trace_offset: np.ndarray
    mass_ff: np.ndarray
    mass_fb: np.ndarray
    advection_ff: np.ndarray
    advection_fb: np.ndarray
    effective_mass: np.ndarray
    reduced_matrix: np.ndarray
    reduced_offset: np.ndarray
    report: ActualWakeAffineTraceReport

    def body_trace(self, free_state: Any) -> np.ndarray:
        state = self._free_state(free_state)
        return self.trace_matrix @ state + self.trace_offset

    def full_state(self, free_state: Any) -> np.ndarray:
        state = self._free_state(free_state)
        full = np.empty_like(self.base_global_state)
        full[self.free_dof_indices] = state
        full[self.body_dof_indices] = self.body_trace(state)
        return full

    def rate(self, free_state: Any) -> np.ndarray:
        state = self._free_state(free_state)
        return self.reduced_matrix @ state + self.reduced_offset

    def free_weak_residual(
        self,
        free_state: Any,
        free_rate: Any,
    ) -> np.ndarray:
        state = self._free_state(free_state)
        rate = np.asarray(free_rate, dtype=float)
        if rate.shape != state.shape or not np.all(np.isfinite(rate)):
            raise ActualWakeStageTopologyError(
                "free_rate has incompatible shape or values"
            )
        trace = self.body_trace(state)
        trace_rate = self.trace_matrix @ rate
        return (
            self.mass_ff @ rate
            + self.mass_fb @ trace_rate
            + self.advection_ff @ state
            + self.advection_fb @ trace
        )

    def exact_affine_step(
        self,
        initial_free_state: Any,
        duration: float,
    ) -> np.ndarray:
        initial = self._free_state(initial_free_state)
        if duration < 0.0 or not np.isfinite(duration):
            raise ActualWakeStageTopologyError(
                "duration must be finite and non-negative"
            )
        count = len(initial)
        augmented = np.zeros((count + 1, count + 1), dtype=float)
        augmented[:count, :count] = self.reduced_matrix
        augmented[:count, count] = self.reduced_offset
        initial_augmented = np.concatenate((initial, np.ones(1)))
        return (expm(float(duration) * augmented) @ initial_augmented)[:count]

    def explicit_midpoint(
        self,
        initial_free_state: Any,
        *,
        duration: float,
        steps: int,
    ) -> tuple[np.ndarray, float, float]:
        state = self._free_state(initial_free_state).copy()
        dt = self._step(duration, steps)
        weak = 0.0
        algebraic = 0.0
        for _ in range(int(steps)):
            first = self.rate(state)
            middle = state + 0.5 * dt * first
            middle_trace = self.body_trace(middle)
            middle_full = self.full_state(middle)
            algebraic = max(
                algebraic,
                float(
                    np.max(
                        np.abs(
                            middle_full[self.body_dof_indices]
                            - middle_trace
                        ),
                        initial=0.0,
                    )
                ),
            )
            second = self.rate(middle)
            residual = self.free_weak_residual(middle, second)
            weak = max(
                weak,
                self._normalized_weak(
                    middle,
                    second,
                    residual,
                ),
            )
            state = state + dt * second
            endpoint_trace = self.body_trace(state)
            endpoint_full = self.full_state(state)
            algebraic = max(
                algebraic,
                float(
                    np.max(
                        np.abs(
                            endpoint_full[self.body_dof_indices]
                            - endpoint_trace
                        ),
                        initial=0.0,
                    )
                ),
            )
        return state, weak, algebraic

    def implicit_trapezoidal(
        self,
        initial_free_state: Any,
        *,
        duration: float,
        steps: int,
    ) -> tuple[np.ndarray, float, float]:
        state = self._free_state(initial_free_state).copy()
        dt = self._step(duration, steps)
        count = len(state)
        identity = np.eye(count)
        left = identity - 0.5 * dt * self.reduced_matrix
        right_matrix = identity + 0.5 * dt * self.reduced_matrix
        weak = 0.0
        algebraic = 0.0
        for _ in range(int(steps)):
            state = np.linalg.solve(
                left,
                right_matrix @ state + dt * self.reduced_offset,
            )
            rate = self.rate(state)
            residual = self.free_weak_residual(state, rate)
            weak = max(
                weak,
                self._normalized_weak(state, rate, residual),
            )
            trace = self.body_trace(state)
            full = self.full_state(state)
            algebraic = max(
                algebraic,
                float(
                    np.max(
                        np.abs(full[self.body_dof_indices] - trace),
                        initial=0.0,
                    )
                ),
            )
        return state, weak, algebraic

    def explicit_polynomial_defect(self, timestep: float) -> float:
        if timestep <= 0.0 or not np.isfinite(timestep):
            raise ActualWakeStageTopologyError(
                "timestep must be finite and positive"
            )
        eigenvalues = np.linalg.eigvals(self.reduced_matrix)
        scaled = float(timestep) * eigenvalues
        polynomial = 1.0 + scaled + 0.5 * scaled**2
        exact = np.exp(scaled)
        return float(
            np.max(
                np.abs(polynomial - exact)
                / np.maximum(np.abs(exact), np.finfo(float).tiny),
                initial=0.0,
            )
        )

    def _free_state(self, value: Any) -> np.ndarray:
        state = np.asarray(value, dtype=float)
        expected = (len(self.free_dof_indices),)
        if state.shape != expected or not np.all(np.isfinite(state)):
            raise ActualWakeStageTopologyError(
                f"free_state must be finite with shape {expected}"
            )
        return state

    @staticmethod
    def _step(duration: float, steps: int) -> float:
        if (
            duration <= 0.0
            or not np.isfinite(duration)
            or not isinstance(steps, (int, np.integer))
            or int(steps) < 1
        ):
            raise ActualWakeStageTopologyError(
                "duration/steps must be finite positive values"
            )
        return float(duration) / int(steps)

    def _normalized_weak(
        self,
        state: np.ndarray,
        rate: np.ndarray,
        residual: np.ndarray,
    ) -> float:
        trace = self.body_trace(state)
        trace_rate = self.trace_matrix @ rate
        terms = (
            self.mass_ff @ rate,
            self.mass_fb @ trace_rate,
            self.advection_ff @ state,
            self.advection_fb @ trace,
        )
        scale = max(
            float(np.linalg.norm(term, ord=np.inf))
            for term in terms
        )
        return float(
            np.linalg.norm(residual, ord=np.inf)
            / max(scale, np.finfo(float).tiny)
        )


def _solution_global_state(
    topology: ActualWakeStageTopology,
    solution: ActualBoundaryBodyWakeSolution,
) -> np.ndarray:
    return topology.global_p2_state(solution.wake_history)


def solve_actual_wake_affine_counterfactual(
    mesh: ClosedTriangularMesh,
    body_topology,
    stage_topology: ActualWakeStageTopology,
    base_solution: ActualBoundaryBodyWakeSolution,
    attachment: MaterialWakeCutAttachment,
    global_p2_state: Any,
    *,
    quadrature_order: int,
) -> ActualBoundaryBodyWakeSolution:
    """Re-solve one exact affine scalar counterfactual on fixed geometry."""
    state = np.asarray(global_p2_state, dtype=float)
    history = stage_topology.rebuild_history(
        base_solution.wake_history,
        state,
    )
    return solve_actual_boundary_body_wake_p2(
        mesh,
        body_topology,
        incident_velocity=base_solution.incident_velocity,
        wall_velocity=base_solution.wall_velocity,
        downstream_edge_x=None,
        prescribed_wake_history=history,
        prescribed_wake_attachment=attachment,
        target_quadrature_order=int(quadrature_order),
        source_quadrature_order=int(quadrature_order),
    )


def build_actual_wake_linearized_dae(
    mesh: ClosedTriangularMesh,
    body_topology,
    stage_topology: ActualWakeStageTopology,
    base_solution: ActualBoundaryBodyWakeSolution,
    attachment: MaterialWakeCutAttachment,
    transport_operator: P2SurfaceMaterialTransportOperator,
    *,
    boundary_quadrature_order: int = 10,
) -> ActualWakeLinearizedDAE:
    """Build the exact unit-basis affine body-trace map and local DAE."""
    if not isinstance(stage_topology, ActualWakeStageTopology):
        raise ActualWakeStageTopologyError(
            "stage_topology must be ActualWakeStageTopology"
        )
    if not isinstance(base_solution, ActualBoundaryBodyWakeSolution):
        raise ActualWakeStageTopologyError(
            "base_solution must be ActualBoundaryBodyWakeSolution"
        )
    if not isinstance(
        transport_operator,
        P2SurfaceMaterialTransportOperator,
    ):
        raise ActualWakeStageTopologyError(
            "transport_operator has incompatible type"
        )
    if (
        transport_operator.topology.degree_of_freedom_count
        != stage_topology.p2_topology.degree_of_freedom_count
        or not np.array_equal(
            transport_operator.topology.local_to_global,
            stage_topology.p2_topology.local_to_global,
        )
    ):
        raise ActualWakeStageTopologyError(
            "transport and stage P2 topologies differ"
        )
    base = _solution_global_state(stage_topology, base_solution)
    body = stage_topology.boundary_roles.body_attachment_p2_dofs
    free = np.setdiff1d(
        np.arange(len(base), dtype=np.int64),
        body,
    )
    base_trace = base[body]
    matrix = np.empty((len(body), len(free)), dtype=float)
    weak = float(base_solution.relative_weak_residual)
    for column, dof in enumerate(free):
        state = base.copy()
        state[int(dof)] += 1.0
        solution = solve_actual_wake_affine_counterfactual(
            mesh,
            body_topology,
            stage_topology,
            base_solution,
            attachment,
            state,
            quadrature_order=boundary_quadrature_order,
        )
        solved = _solution_global_state(stage_topology, solution)
        matrix[:, column] = solved[body] - base_trace
        weak = max(weak, float(solution.relative_weak_residual))
    offset = base_trace - matrix @ base[free]
    base_error = float(
        np.max(
            np.abs(matrix @ base[free] + offset - base_trace),
            initial=0.0,
        )
    )
    mass = transport_operator.mass_matrix
    advection = transport_operator.advection_matrix
    mass_ff = mass[np.ix_(free, free)]
    mass_fb = mass[np.ix_(free, body)]
    advection_ff = advection[np.ix_(free, free)]
    advection_fb = advection[np.ix_(free, body)]
    effective_mass = mass_ff + mass_fb @ matrix
    rank = int(np.linalg.matrix_rank(effective_mass))
    condition = float(np.linalg.cond(effective_mass))
    if rank != len(free) or not np.isfinite(condition):
        raise ActualWakeStageTopologyError(
            "actual-wake effective differential mass is singular"
        )
    reduced_matrix = np.linalg.solve(
        effective_mass,
        -(advection_ff + advection_fb @ matrix),
    )
    reduced_offset = np.linalg.solve(
        effective_mass,
        -(advection_fb @ offset),
    )
    return ActualWakeLinearizedDAE(
        topology=stage_topology,
        free_dof_indices=free,
        body_dof_indices=body.copy(),
        base_global_state=base,
        trace_matrix=matrix,
        trace_offset=offset,
        mass_ff=mass_ff,
        mass_fb=mass_fb,
        advection_ff=advection_ff,
        advection_fb=advection_fb,
        effective_mass=effective_mass,
        reduced_matrix=reduced_matrix,
        reduced_offset=reduced_offset,
        report=ActualWakeAffineTraceReport(
            basis_solve_count=len(free),
            maximum_relative_weak_residual=weak,
            base_reconstruction_error=base_error,
            effective_mass_rank=rank,
            effective_mass_condition_number=condition,
        ),
    )
