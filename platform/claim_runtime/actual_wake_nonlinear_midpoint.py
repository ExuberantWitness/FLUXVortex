"""Constraint-consistent nonlinear previous-time actual-wake midpoint step.

Geometry is predicted explicitly from the already evaluated stage velocity.
At each fixed midpoint/endpoint geometry, the nine body-attachment P2 values
are solved together with the free P2 increment by exact affine
superposition.  Geometry itself is never iterated.

The module contains no pressure, force, LESP, damping, stabilization, target
load or structural state.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from .actual_boundary_body_wake import (
    ActualBoundaryBodyWakeSolution,
    MaterialWakeCutAttachment,
    solve_actual_boundary_body_wake_p2,
)
from .actual_wake_stage_topology import (
    ActualWakeStageTopology,
    ActualWakeStageTopologyError,
    actual_wake_stage_topology,
)
from .distributed_doublet import MaterialWakeHistory
from .p2_surface_material_transport import (
    P2SurfaceMaterialTransportOperator,
)
from .thick_body_neumann_shadow import ClosedTriangularMesh


@dataclass(frozen=True)
class ActualWakeEvaluatedStage:
    """One algebraically consistent stage plus its physical velocities."""

    solution: ActualBoundaryBodyWakeSolution
    topology: ActualWakeStageTopology
    p1_geometry_velocity: np.ndarray
    transport_operator: P2SurfaceMaterialTransportOperator
    geometry_velocity_ledger_closure: float
    transport_velocity_ledger_closure: float
    geometry_projection_residual_fraction: float
    relative_velocity_normal_component_max: float

    def __post_init__(self) -> None:
        velocity = np.asarray(self.p1_geometry_velocity, dtype=float)
        if (
            not isinstance(self.solution, ActualBoundaryBodyWakeSolution)
            or not isinstance(self.topology, ActualWakeStageTopology)
            or not isinstance(
                self.transport_operator,
                P2SurfaceMaterialTransportOperator,
            )
            or velocity.shape != self.topology.p1_vertices.shape
            or not np.all(np.isfinite(velocity))
        ):
            raise ActualWakeStageTopologyError(
                "evaluated actual-wake stage has incompatible state"
            )
        scalars = (
            self.geometry_velocity_ledger_closure,
            self.transport_velocity_ledger_closure,
            self.geometry_projection_residual_fraction,
            self.relative_velocity_normal_component_max,
        )
        if not all(np.isfinite(value) and value >= 0.0 for value in scalars):
            raise ActualWakeStageTopologyError(
                "evaluated actual-wake diagnostics must be finite/nonnegative"
            )
        if (
            self.transport_operator.topology.degree_of_freedom_count
            != self.topology.p2_topology.degree_of_freedom_count
            or not np.array_equal(
                self.transport_operator.topology.local_to_global,
                self.topology.p2_topology.local_to_global,
            )
        ):
            raise ActualWakeStageTopologyError(
                "stage transport and actual-wake P2 topology differ"
            )
        object.__setattr__(
            self,
            "p1_geometry_velocity",
            velocity.copy(),
        )


@dataclass(frozen=True)
class ActualWakeAffineStageReport:
    basis_solve_count: int
    rank: int
    condition_number: float
    algebraic_trace_residual: float
    free_state_preservation_error: float
    maximum_actual_boundary_relative_weak_residual: float


@dataclass(frozen=True)
class ActualWakeNonlinearMidpointReport:
    timestep: float
    half_scalar_normalized_residual: float
    full_scalar_normalized_residual: float
    midpoint_geometry_identity_error: float
    endpoint_geometry_identity_error: float
    body_geometry_attachment_error: float
    chronological_seam_error: float
    p2_roundtrip_error: float
    free_geometry_change: float
    free_scalar_change: float
    minimum_face_area_ratio: float
    input_state_mutation: float
    geometry_iteration_count: int
    half_stage: ActualWakeAffineStageReport
    endpoint_stage: ActualWakeAffineStageReport


@dataclass(frozen=True)
class ActualWakeNonlinearMidpointStep:
    initial_stage: ActualWakeEvaluatedStage
    midpoint_stage: ActualWakeEvaluatedStage
    endpoint_solution: ActualBoundaryBodyWakeSolution
    endpoint_topology: ActualWakeStageTopology
    report: ActualWakeNonlinearMidpointReport


StageEvaluator = Callable[
    [ActualBoundaryBodyWakeSolution],
    ActualWakeEvaluatedStage,
]


def _geometry_history(
    history: MaterialWakeHistory,
    topology: ActualWakeStageTopology,
    global_vertices: np.ndarray,
) -> MaterialWakeHistory:
    vertices = np.asarray(global_vertices, dtype=float)
    if (
        vertices.shape != topology.p1_vertices.shape
        or not np.all(np.isfinite(vertices))
    ):
        raise ActualWakeStageTopologyError(
            "global wake geometry has incompatible shape or values"
        )
    bands = []
    for band, dofs in zip(
        history.bands,
        topology.band_p1_vertex_dofs,
        strict=True,
    ):
        bands.append(band.material_update(vertices[dofs]))
    candidate = MaterialWakeHistory(history.history_id, tuple(bands))
    report = candidate.continuity_report()
    if not report.compatible:
        raise ActualWakeStageTopologyError(
            f"global wake geometry broke chronology: {report}"
        )
    return candidate


def _global_state(
    topology: ActualWakeStageTopology,
    solution: ActualBoundaryBodyWakeSolution,
) -> np.ndarray:
    return topology.global_p2_state(solution.wake_history)


def _normalized_sum_residual(
    residual: np.ndarray,
    *terms: np.ndarray,
) -> float:
    scale = max(
        (float(np.linalg.norm(term, ord=np.inf)) for term in terms),
        default=0.0,
    )
    return float(
        np.linalg.norm(residual, ord=np.inf)
        / max(scale, np.finfo(float).tiny)
    )


def _solve_stage_trace(
    mesh: ClosedTriangularMesh,
    body_topology,
    geometry_history: MaterialWakeHistory,
    attachment: MaterialWakeCutAttachment,
    incident_velocity: np.ndarray,
    wall_velocity: np.ndarray,
    free_state_from_trace: Callable[[np.ndarray], np.ndarray],
    *,
    boundary_quadrature_order: int,
) -> tuple[
    ActualBoundaryBodyWakeSolution,
    ActualWakeStageTopology,
    np.ndarray,
    np.ndarray,
    ActualWakeAffineStageReport,
]:
    topology = actual_wake_stage_topology(
        geometry_history,
        body_attachment_id="canonical-body-cut",
    )
    body = topology.boundary_roles.body_attachment_p2_dofs
    free = np.setdiff1d(
        np.arange(
            topology.p2_topology.degree_of_freedom_count,
            dtype=np.int64,
        ),
        body,
    )

    def evaluate(trace: np.ndarray):
        value = np.asarray(trace, dtype=float)
        if value.shape != (len(body),) or not np.all(np.isfinite(value)):
            raise ActualWakeStageTopologyError(
                "stage body trace has incompatible shape or values"
            )
        free_state = np.asarray(
            free_state_from_trace(value),
            dtype=float,
        )
        if (
            free_state.shape != (len(free),)
            or not np.all(np.isfinite(free_state))
        ):
            raise ActualWakeStageTopologyError(
                "stage free-state map returned incompatible values"
            )
        global_state = np.empty(
            topology.p2_topology.degree_of_freedom_count,
            dtype=float,
        )
        global_state[free] = free_state
        global_state[body] = value
        candidate = topology.rebuild_history(
            geometry_history,
            global_state,
        )
        solution = solve_actual_boundary_body_wake_p2(
            mesh,
            body_topology,
            incident_velocity=incident_velocity,
            wall_velocity=wall_velocity,
            downstream_edge_x=None,
            prescribed_wake_history=candidate,
            prescribed_wake_attachment=attachment,
            target_quadrature_order=int(boundary_quadrature_order),
            source_quadrature_order=int(boundary_quadrature_order),
        )
        solved = topology.global_p2_state(solution.wake_history)
        residual = value - solved[body]
        free_error = float(
            np.max(
                np.abs(solved[free] - free_state),
                initial=0.0,
            )
        )
        return residual, solution, free_state, solved, free_error

    zero = np.zeros(len(body), dtype=float)
    base = evaluate(zero)
    matrix = np.empty((len(body), len(body)), dtype=float)
    maximum_weak = float(base[1].relative_weak_residual)
    maximum_free_error = base[4]
    for column in range(len(body)):
        basis = zero.copy()
        basis[column] = 1.0
        record = evaluate(basis)
        matrix[:, column] = record[0] - base[0]
        maximum_weak = max(
            maximum_weak,
            float(record[1].relative_weak_residual),
        )
        maximum_free_error = max(maximum_free_error, record[4])
    rank = int(np.linalg.matrix_rank(matrix))
    condition = float(np.linalg.cond(matrix))
    if rank != len(body) or not np.isfinite(condition):
        raise ActualWakeStageTopologyError(
            "actual stage body-trace affine system is singular"
        )
    trace = np.linalg.solve(matrix, -base[0])
    verified = evaluate(trace)
    maximum_weak = max(
        maximum_weak,
        float(verified[1].relative_weak_residual),
    )
    maximum_free_error = max(maximum_free_error, verified[4])
    algebraic = float(
        np.max(np.abs(verified[0]), initial=0.0)
    )
    return (
        verified[1],
        topology,
        verified[2],
        trace,
        ActualWakeAffineStageReport(
            basis_solve_count=len(body) + 2,
            rank=rank,
            condition_number=condition,
            algebraic_trace_residual=algebraic,
            free_state_preservation_error=maximum_free_error,
            maximum_actual_boundary_relative_weak_residual=maximum_weak,
        ),
    )


def _operator_blocks(
    stage: ActualWakeEvaluatedStage,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    body = stage.topology.boundary_roles.body_attachment_p2_dofs
    free = np.setdiff1d(
        np.arange(
            stage.topology.p2_topology.degree_of_freedom_count,
            dtype=np.int64,
        ),
        body,
    )
    mass = stage.transport_operator.mass_matrix
    advection = stage.transport_operator.advection_matrix
    return (
        free,
        body,
        mass[np.ix_(free, free)],
        mass[np.ix_(free, body)],
        advection[np.ix_(free, free)],
        advection[np.ix_(free, body)],
    )


def _minimum_area_ratio(
    before: MaterialWakeHistory,
    after: MaterialWakeHistory,
) -> float:
    minimum = np.inf
    for old_band, new_band in zip(
        before.bands,
        after.bands,
        strict=True,
    ):
        for face_index in range(len(old_band.surface)):
            minimum = min(
                minimum,
                new_band.surface.element(face_index).area
                / old_band.surface.element(face_index).area,
            )
    return float(minimum)


def advance_actual_wake_previous_time_midpoint(
    mesh: ClosedTriangularMesh,
    body_topology,
    initial_stage: ActualWakeEvaluatedStage,
    attachment: MaterialWakeCutAttachment,
    *,
    timestep: float,
    stage_evaluator: StageEvaluator,
    boundary_quadrature_order: int = 10,
) -> ActualWakeNonlinearMidpointStep:
    """Advance one nonlinear actual-wake step without geometry iteration."""
    if timestep <= 0.0 or not np.isfinite(timestep):
        raise ActualWakeStageTopologyError(
            "timestep must be finite and positive"
        )
    if not callable(stage_evaluator):
        raise ActualWakeStageTopologyError(
            "stage_evaluator must be callable"
        )
    initial_history = initial_stage.solution.wake_history
    topology0 = initial_stage.topology
    state0 = topology0.global_p2_state(initial_history)
    geometry_snapshot = tuple(
        band.surface.vertices.copy() for band in initial_history.bands
    )
    state_snapshot = state0.copy()
    free0, body0, mff0, mfb0, cff0, cfb0 = _operator_blocks(
        initial_stage
    )
    y0 = state0[free0]
    g0 = state0[body0]

    x0 = topology0.p1_vertices
    xmid = x0 + 0.5 * float(timestep) * (
        initial_stage.p1_geometry_velocity
    )
    geometry_mid = _geometry_history(
        initial_history,
        topology0,
        xmid,
    )
    half_flux = 0.5 * float(timestep) * (
        cff0 @ y0 + cfb0 @ g0
    )

    def half_free(trace: np.ndarray) -> np.ndarray:
        return y0 - np.linalg.solve(
            mff0,
            mfb0 @ (trace - g0) + half_flux,
        )

    (
        midpoint_solution,
        midpoint_topology,
        ymid,
        gmid,
        half_report,
    ) = _solve_stage_trace(
        mesh,
        body_topology,
        geometry_mid,
        attachment,
        initial_stage.solution.incident_velocity,
        initial_stage.solution.wall_velocity,
        half_free,
        boundary_quadrature_order=boundary_quadrature_order,
    )
    midpoint_stage = stage_evaluator(midpoint_solution)
    if (
        not np.array_equal(
            midpoint_stage.topology.p2_dof_to_chronological,
            topology0.p2_dof_to_chronological,
        )
        or not np.array_equal(
            midpoint_stage.topology.boundary_roles.body_attachment_p2_dofs,
            body0,
        )
    ):
        raise ActualWakeStageTopologyError(
            "midpoint stage changed chronological P2 identity"
        )
    free_mid, body_mid, mff_mid, mfb_mid, cff_mid, cfb_mid = (
        _operator_blocks(midpoint_stage)
    )
    if not np.array_equal(free_mid, free0) or not np.array_equal(
        body_mid,
        body0,
    ):
        raise ActualWakeStageTopologyError(
            "midpoint stage changed free/body P2 partition"
        )

    x1 = x0 + float(timestep) * midpoint_stage.p1_geometry_velocity
    geometry_end = _geometry_history(
        initial_history,
        topology0,
        x1,
    )
    full_flux = float(timestep) * (
        cff_mid @ ymid + cfb_mid @ gmid
    )

    def endpoint_free(trace: np.ndarray) -> np.ndarray:
        return y0 - np.linalg.solve(
            mff_mid,
            mfb_mid @ (trace - g0) + full_flux,
        )

    (
        endpoint_solution,
        endpoint_topology,
        y1,
        g1,
        endpoint_report,
    ) = _solve_stage_trace(
        mesh,
        body_topology,
        geometry_end,
        attachment,
        initial_stage.solution.incident_velocity,
        initial_stage.solution.wall_velocity,
        endpoint_free,
        boundary_quadrature_order=boundary_quadrature_order,
    )
    if (
        not np.array_equal(
            endpoint_topology.p2_dof_to_chronological,
            topology0.p2_dof_to_chronological,
        )
        or not np.array_equal(
            endpoint_topology.boundary_roles.body_attachment_p2_dofs,
            body0,
        )
    ):
        raise ActualWakeStageTopologyError(
            "endpoint stage changed chronological P2 identity"
        )

    half_terms = (
        mff0 @ (ymid - y0),
        mfb0 @ (gmid - g0),
        half_flux,
    )
    half_residual = sum(half_terms)
    half_normalized = _normalized_sum_residual(
        half_residual,
        *half_terms,
    )
    full_terms = (
        mff_mid @ (y1 - y0),
        mfb_mid @ (g1 - g0),
        full_flux,
    )
    full_residual = sum(full_terms)
    full_normalized = _normalized_sum_residual(
        full_residual,
        *full_terms,
    )
    midpoint_geometry_error = float(
        np.max(
            np.abs(
                midpoint_topology.p1_vertices - xmid
            ),
            initial=0.0,
        )
    )
    endpoint_geometry_error = float(
        np.max(
            np.abs(endpoint_topology.p1_vertices - x1),
            initial=0.0,
        )
    )
    attachment_error = max(
        float(
            np.max(
                np.abs(
                    stage_topology.p1_vertices[
                        stage_topology.boundary_roles
                        .body_attachment_p1_dofs
                    ]
                    - x0[
                        topology0.boundary_roles
                        .body_attachment_p1_dofs
                    ]
                ),
                initial=0.0,
            )
        )
        for stage_topology in (midpoint_topology, endpoint_topology)
    )
    seam_error = max(
        max(
            report.max_geometry_gap,
            report.max_trace_jump,
            report.max_time_gap,
        )
        for report in (
            midpoint_solution.wake_history.continuity_report(),
            endpoint_solution.wake_history.continuity_report(),
        )
    )
    endpoint_state = endpoint_topology.global_p2_state(
        endpoint_solution.wake_history
    )
    expected_endpoint = np.empty_like(endpoint_state)
    expected_endpoint[free0] = y1
    expected_endpoint[body0] = g1
    roundtrip = float(
        np.max(
            np.abs(endpoint_state - expected_endpoint),
            initial=0.0,
        )
    )
    free_p1 = np.setdiff1d(
        np.arange(len(x0), dtype=np.int64),
        topology0.boundary_roles.body_attachment_p1_dofs,
    )
    free_geometry_change = float(
        np.max(
            np.linalg.norm(
                endpoint_topology.p1_vertices[free_p1] - x0[free_p1],
                axis=1,
            ),
            initial=0.0,
        )
    )
    free_scalar_change = float(
        np.max(np.abs(y1 - y0), initial=0.0)
    )
    mutation = max(
        float(
            np.max(
                np.abs(
                    topology0.global_p2_state(initial_history)
                    - state_snapshot
                ),
                initial=0.0,
            )
        ),
        max(
            float(
                np.max(
                    np.abs(band.surface.vertices - snapshot),
                    initial=0.0,
                )
            )
            for band, snapshot in zip(
                initial_history.bands,
                geometry_snapshot,
                strict=True,
            )
        ),
    )
    return ActualWakeNonlinearMidpointStep(
        initial_stage=initial_stage,
        midpoint_stage=midpoint_stage,
        endpoint_solution=endpoint_solution,
        endpoint_topology=endpoint_topology,
        report=ActualWakeNonlinearMidpointReport(
            timestep=float(timestep),
            half_scalar_normalized_residual=half_normalized,
            full_scalar_normalized_residual=full_normalized,
            midpoint_geometry_identity_error=midpoint_geometry_error,
            endpoint_geometry_identity_error=endpoint_geometry_error,
            body_geometry_attachment_error=attachment_error,
            chronological_seam_error=seam_error,
            p2_roundtrip_error=roundtrip,
            free_geometry_change=free_geometry_change,
            free_scalar_change=free_scalar_change,
            minimum_face_area_ratio=_minimum_area_ratio(
                initial_history,
                endpoint_solution.wake_history,
            ),
            input_state_mutation=mutation,
            geometry_iteration_count=0,
            half_stage=half_report,
            endpoint_stage=endpoint_report,
        ),
    )
