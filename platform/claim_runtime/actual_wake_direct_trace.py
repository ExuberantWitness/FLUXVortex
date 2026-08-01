"""One-solve execution-equivalent current-trace evaluator.

For a prescribed unsteady wake history the actual boundary solver consumes
all completed bands and only the upstream/middle rows of the newest band.
Its current row is always generated from the solved body-cut jump.  Hence
the current-row residual has the exact form ``trace - solved_trace`` and an
identity Jacobian.

This module leaves the frozen zero-plus-unit S3aa implementation untouched.
It changes no matrix, right-hand side, quadrature, geometry or physical
state; it only avoids reassembling an input-independent solve.
"""
from __future__ import annotations

import numpy as np

from .actual_boundary_body_wake import (
    solve_actual_boundary_body_wake_p2,
)
from .actual_wake_newborn_coupled_stage import (
    ActualWakeCoupledNewbornStage,
    ActualWakeNewbornTraceSolveReport,
)
from .actual_wake_newborn_transition import (
    ActualWakeNewbornTransition,
)
from .actual_wake_stage_topology import (
    ActualWakeStageTopologyError,
)


def solve_actual_wake_coupled_newborn_trace_direct(
    mesh,
    body_topology,
    transition: ActualWakeNewbornTransition,
    attachment,
    *,
    incident_velocity,
    wall_velocity,
    copy_counterfactual_trace,
    boundary_quadrature_order: int = 10,
) -> ActualWakeCoupledNewbornStage:
    """Return the exact current trace from one actual-boundary solve."""
    if not isinstance(transition, ActualWakeNewbornTransition):
        raise ActualWakeStageTopologyError(
            "direct newborn stage requires ActualWakeNewbornTransition"
        )
    topology = transition.augmented_topology
    history = transition.augmented_history
    topology._validate_history(history, tolerance=2.0e-12)
    state0 = topology.global_p2_state(history)
    body = topology.boundary_roles.body_attachment_p2_dofs
    free = np.setdiff1d(
        np.arange(
            topology.p2_topology.degree_of_freedom_count,
            dtype=np.int64,
        ),
        body,
    )
    copy = np.asarray(copy_counterfactual_trace, dtype=float)
    if (
        copy.shape != (len(body),)
        or not np.all(np.isfinite(copy))
    ):
        raise ActualWakeStageTopologyError(
            "direct trace counterfactual has incompatible shape"
        )
    geometry_snapshot = tuple(
        band.surface.vertices.copy() for band in history.bands
    )
    scalar_snapshot = tuple(
        band.potential_jump_rows.copy() for band in history.bands
    )
    solution = solve_actual_boundary_body_wake_p2(
        mesh,
        body_topology,
        incident_velocity=incident_velocity,
        wall_velocity=wall_velocity,
        downstream_edge_x=None,
        prescribed_wake_history=history,
        prescribed_wake_attachment=attachment,
        target_quadrature_order=int(boundary_quadrature_order),
        source_quadrature_order=int(boundary_quadrature_order),
    )
    solved = topology.global_p2_state(solution.wake_history)
    trace = solved[body].copy()
    free_error = float(
        np.max(np.abs(solved[free] - state0[free]), initial=0.0)
    )
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
        solution=solution,
        global_p2_state=solved.copy(),
        solved_trace=trace,
        report=ActualWakeNewbornTraceSolveReport(
            unknown_count=len(body),
            free_count=len(free),
            basis_solve_count=1,
            rank=len(body),
            rank_deficiency=0,
            condition_number=1.0,
            algebraic_trace_residual=0.0,
            free_state_preservation_error=free_error,
            maximum_actual_boundary_relative_weak_residual=float(
                solution.relative_weak_residual
            ),
            maximum_wake_attachment_error=float(
                solution.wake_attachment_error
            ),
            copy_counterfactual_residual=float(
                np.max(np.abs(copy - trace), initial=0.0)
            ),
            input_state_mutation=mutation,
            inferred_scalar_count=0,
            geometry_iteration_count=0,
        ),
    )
