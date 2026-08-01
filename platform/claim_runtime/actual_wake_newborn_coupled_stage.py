"""Coupled algebraic body/newborn trace solve on a fixed augmented stage.

The augmented geometry and every non-body P2 value are fixed inputs.  The
current newborn/body attachment row is obtained from the exact affine
residual between a proposed trace and the actual body-wake boundary solve.

This module does not advance old material state, infer rows, copy/clamp a
trace, iterate geometry, regularize a kernel, or compute pressure/force.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .actual_boundary_body_wake import (
    ActualBoundaryBodyWakeSolution,
    MaterialWakeCutAttachment,
    solve_actual_boundary_body_wake_p2,
)
from .actual_wake_newborn_transition import (
    ActualWakeNewbornTransition,
)
from .actual_wake_stage_topology import (
    ActualWakeStageTopology,
    ActualWakeStageTopologyError,
)
from .thick_body_neumann_shadow import ClosedTriangularMesh


@dataclass(frozen=True)
class ActualWakeNewbornTraceSolveReport:
    unknown_count: int
    free_count: int
    basis_solve_count: int
    rank: int
    rank_deficiency: int
    condition_number: float
    algebraic_trace_residual: float
    free_state_preservation_error: float
    maximum_actual_boundary_relative_weak_residual: float
    maximum_wake_attachment_error: float
    copy_counterfactual_residual: float
    input_state_mutation: float
    inferred_scalar_count: int
    geometry_iteration_count: int


@dataclass(frozen=True)
class ActualWakeCoupledNewbornStage:
    topology: ActualWakeStageTopology
    solution: ActualBoundaryBodyWakeSolution
    global_p2_state: np.ndarray
    solved_trace: np.ndarray
    report: ActualWakeNewbornTraceSolveReport


def solve_actual_wake_coupled_newborn_trace(
    mesh: ClosedTriangularMesh,
    body_topology,
    transition: ActualWakeNewbornTransition,
    attachment: MaterialWakeCutAttachment,
    *,
    incident_velocity: Any,
    wall_velocity: Any,
    copy_counterfactual_trace: Any,
    boundary_quadrature_order: int = 10,
) -> ActualWakeCoupledNewbornStage:
    """Solve one fixed-geometry nine-dimensional newborn trace stage."""
    if not isinstance(transition, ActualWakeNewbornTransition):
        raise ActualWakeStageTopologyError(
            "newborn stage requires ActualWakeNewbornTransition"
        )
    topology = transition.augmented_topology
    history = transition.augmented_history
    topology._validate_history(history, tolerance=2.0e-12)
    body = topology.boundary_roles.body_attachment_p2_dofs
    free = np.setdiff1d(
        np.arange(
            topology.p2_topology.degree_of_freedom_count,
            dtype=np.int64,
        ),
        body,
    )
    initial_state = topology.global_p2_state(history)
    geometry_snapshot = tuple(
        band.surface.vertices.copy() for band in history.bands
    )
    scalar_snapshot = tuple(
        band.potential_jump_rows.copy() for band in history.bands
    )
    incident = np.asarray(incident_velocity, dtype=float)
    wall = np.asarray(wall_velocity, dtype=float)
    if (
        incident.shape != (len(mesh.faces), 3)
        or wall.shape != incident.shape
        or not np.all(np.isfinite(incident))
        or not np.all(np.isfinite(wall))
    ):
        raise ActualWakeStageTopologyError(
            "newborn stage incident/wall velocities are incompatible"
        )

    def evaluate(trace_value):
        trace = np.asarray(trace_value, dtype=float)
        if (
            trace.shape != (len(body),)
            or not np.all(np.isfinite(trace))
        ):
            raise ActualWakeStageTopologyError(
                "newborn trace has incompatible shape or values"
            )
        state = initial_state.copy()
        state[body] = trace
        candidate = topology.rebuild_history(history, state)
        solution = solve_actual_boundary_body_wake_p2(
            mesh,
            body_topology,
            incident_velocity=incident,
            wall_velocity=wall,
            downstream_edge_x=None,
            prescribed_wake_history=candidate,
            prescribed_wake_attachment=attachment,
            target_quadrature_order=int(
                boundary_quadrature_order
            ),
            source_quadrature_order=int(
                boundary_quadrature_order
            ),
        )
        solved = topology.global_p2_state(
            solution.wake_history
        )
        residual = trace - solved[body]
        free_error = float(
            np.max(
                np.abs(solved[free] - initial_state[free]),
                initial=0.0,
            )
        )
        return residual, solution, solved, free_error

    zero = np.zeros(len(body), dtype=float)
    base = evaluate(zero)
    matrix = np.empty((len(body), len(body)), dtype=float)
    maximum_weak = float(base[1].relative_weak_residual)
    maximum_attachment = float(base[1].wake_attachment_error)
    maximum_free = base[3]
    for column in range(len(body)):
        basis = zero.copy()
        basis[column] = 1.0
        record = evaluate(basis)
        matrix[:, column] = record[0] - base[0]
        maximum_weak = max(
            maximum_weak,
            float(record[1].relative_weak_residual),
        )
        maximum_attachment = max(
            maximum_attachment,
            float(record[1].wake_attachment_error),
        )
        maximum_free = max(maximum_free, record[3])
    rank = int(np.linalg.matrix_rank(matrix))
    condition = float(np.linalg.cond(matrix))
    if rank != len(body) or not np.isfinite(condition):
        raise ActualWakeStageTopologyError(
            "newborn affine trace system is singular or non-finite"
        )
    solved_trace = np.linalg.solve(matrix, -base[0])
    verified = evaluate(solved_trace)
    maximum_weak = max(
        maximum_weak,
        float(verified[1].relative_weak_residual),
    )
    maximum_attachment = max(
        maximum_attachment,
        float(verified[1].wake_attachment_error),
    )
    maximum_free = max(maximum_free, verified[3])
    counterfactual = evaluate(copy_counterfactual_trace)
    maximum_weak = max(
        maximum_weak,
        float(counterfactual[1].relative_weak_residual),
    )
    maximum_attachment = max(
        maximum_attachment,
        float(counterfactual[1].wake_attachment_error),
    )
    maximum_free = max(maximum_free, counterfactual[3])
    mutation = 0.0
    for band, geometry, scalar in zip(
        history.bands,
        geometry_snapshot,
        scalar_snapshot,
        strict=True,
    ):
        mutation = max(
            mutation,
            float(
                np.max(
                    np.abs(band.surface.vertices - geometry),
                    initial=0.0,
                )
            ),
            float(
                np.max(
                    np.abs(band.potential_jump_rows - scalar),
                    initial=0.0,
                )
            ),
        )
    return ActualWakeCoupledNewbornStage(
        topology=topology,
        solution=verified[1],
        global_p2_state=verified[2].copy(),
        solved_trace=solved_trace.copy(),
        report=ActualWakeNewbornTraceSolveReport(
            unknown_count=len(body),
            free_count=len(free),
            basis_solve_count=len(body) + 3,
            rank=rank,
            rank_deficiency=len(body) - rank,
            condition_number=condition,
            algebraic_trace_residual=float(
                np.max(
                    np.abs(verified[0]),
                    initial=0.0,
                )
            ),
            free_state_preservation_error=maximum_free,
            maximum_actual_boundary_relative_weak_residual=(
                maximum_weak
            ),
            maximum_wake_attachment_error=maximum_attachment,
            copy_counterfactual_residual=float(
                np.max(
                    np.abs(counterfactual[0]),
                    initial=0.0,
                )
            ),
            input_state_mutation=mutation,
            inferred_scalar_count=0,
            geometry_iteration_count=0,
        ),
    )
