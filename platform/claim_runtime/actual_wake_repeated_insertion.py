"""Actual old-material/newborn explicit-midpoint insertion composition.

The old wake is advanced as its own material subdomain with the validated
consistent-P2 ALE balance.  Its former body row becomes a released material
seam.  A new characteristic band is then attached, while its current body
trace is solved by the actual body-wake algebraic residual at the half and
full stages.

No 63-DOF initial vector is formed.  No newborn scalar is inferred, copied,
averaged, clamped or remapped.  The module contains no pressure, force,
LESP, regularization, target load, or structural state.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .actual_boundary_body_wake import (
    ActualBoundaryBodyWakeSolution,
    MaterialWakeCutAttachment,
)
from .actual_body_wake_velocity import WakeSheetQuery
from .actual_wake_newborn_coupled_stage import (
    ActualWakeCoupledNewbornStage,
)
from .actual_wake_direct_trace import (
    solve_actual_wake_coupled_newborn_trace_direct,
)
from .actual_wake_newborn_transition import (
    augment_actual_wake_with_newborn_band,
)
from .actual_wake_stage_topology import (
    ActualWakeStageTopology,
    ActualWakeStageTopologyError,
    actual_wake_stage_topology,
)
from .actual_wake_stage_velocity import (
    actual_wake_owned_quadrature,
    assemble_owned_actual_wake_p2_transport,
)
from .actual_wake_weak_geometry_velocity import (
    project_actual_wake_global_weak_normal_velocity,
    weak_normal_collocation_velocity,
)
from .characteristic_birth_slab import (
    solve_p2_characteristic_birth_slab,
)
from .distributed_doublet import MaterialWakeHistory
from .p2_surface_material_transport import (
    P2SurfaceMaterialTransportOperator,
)
from .thick_body_neumann_shadow import ClosedTriangularMesh


PhysicalVelocityProvider = Callable[
    [ActualBoundaryBodyWakeSolution, WakeSheetQuery],
    np.ndarray,
]


@dataclass(frozen=True)
class ReleasedOldMaterialStage:
    history: MaterialWakeHistory
    topology: ActualWakeStageTopology
    p1_geometry_velocity: np.ndarray
    transport_operator: P2SurfaceMaterialTransportOperator
    weak_rank_deficiency: int
    weak_condition_number: float
    weak_orthogonality_relative_residual: float
    weak_surface_L2_residual: float
    transport_relative_normal_component_max: float


@dataclass(frozen=True)
class ActualWakeRepeatedInsertionStepReport:
    initial_band_count: int
    final_band_count: int
    band_increment_mismatch: int
    old_p1_identity_count: int
    old_p2_identity_count: int
    half_old_transport_normalized_residual: float
    full_old_transport_normalized_residual: float
    old_mass_rank_deficiency_max: int
    old_mass_condition_number_max: float
    actual_boundary_relative_weak_residual_max: float
    algebraic_trace_residual_abs_max: float
    free_state_preservation_abs_max: float
    characteristic_birth_identity_abs_max: float
    chronological_seam_abs_max: float
    p2_roundtrip_abs_max: float
    half_geometry_identity_abs_max: float
    full_geometry_identity_abs_max: float
    minimum_newborn_face_area: float
    old_geometry_change_abs_max: float
    old_scalar_change_abs_max: float
    input_state_mutation_abs_max: float
    inferred_scalar_count: int
    geometry_iteration_count: int


@dataclass(frozen=True)
class ActualWakeRepeatedInsertionStep:
    half_stage: ActualWakeCoupledNewbornStage
    endpoint_stage: ActualWakeCoupledNewbornStage
    report: ActualWakeRepeatedInsertionStepReport


def _normalized_balance(
    residual: np.ndarray,
    *terms: np.ndarray,
) -> float:
    scale = max(
        (
            float(np.linalg.norm(term, ord=np.inf))
            for term in terms
        ),
        default=0.0,
    )
    return float(
        np.linalg.norm(residual, ord=np.inf)
        / max(scale, np.finfo(float).tiny)
    )


def _history_prefix(
    solution: ActualBoundaryBodyWakeSolution,
    band_count: int,
) -> MaterialWakeHistory:
    if band_count < 1 or band_count > len(solution.wake_history.bands):
        raise ActualWakeStageTopologyError(
            "old material band count is invalid"
        )
    return MaterialWakeHistory(
        solution.wake_history.history_id,
        tuple(solution.wake_history.bands[:band_count]),
    )


def _rebuild_old_material(
    history: MaterialWakeHistory,
    topology: ActualWakeStageTopology,
    p1_geometry: np.ndarray,
    p2_state: np.ndarray,
) -> MaterialWakeHistory:
    geometry = np.asarray(p1_geometry, dtype=float)
    state = np.asarray(p2_state, dtype=float)
    if (
        geometry.shape != topology.p1_vertices.shape
        or state.shape
        != (topology.p2_topology.degree_of_freedom_count,)
        or not np.all(np.isfinite(geometry))
        or not np.all(np.isfinite(state))
    ):
        raise ActualWakeStageTopologyError(
            "old material geometry/state has incompatible shape"
        )
    moved = []
    for band, dofs in zip(
        history.bands,
        topology.band_p1_vertex_dofs,
        strict=True,
    ):
        moved.append(band.material_update(geometry[dofs]))
    geometry_history = MaterialWakeHistory(
        history.history_id,
        tuple(moved),
    )
    return topology.rebuild_history(geometry_history, state)


def _released_old_material_stage(
    solution: ActualBoundaryBodyWakeSolution,
    old_history: MaterialWakeHistory,
    *,
    physical_velocity_provider: PhysicalVelocityProvider,
    quadrature_order: int,
) -> ReleasedOldMaterialStage:
    if not callable(physical_velocity_provider):
        raise ActualWakeStageTopologyError(
            "physical_velocity_provider must be callable"
        )
    topology = actual_wake_stage_topology(
        old_history,
        body_attachment_id="released-material-seam",
    )
    quadrature = actual_wake_owned_quadrature(
        topology,
        old_history,
        quadrature_order=int(quadrature_order),
        query_id="S3ac-released-old-material",
    )
    physical = np.asarray(
        physical_velocity_provider(solution, quadrature.query),
        dtype=float,
    )
    if (
        physical.shape != quadrature.query.points.shape
        or not np.all(np.isfinite(physical))
    ):
        raise ActualWakeStageTopologyError(
            "physical velocity provider returned incompatible values"
        )
    projection = project_actual_wake_global_weak_normal_velocity(
        topology,
        old_history,
        quadrature,
        physical,
    )
    mesh_velocity = weak_normal_collocation_velocity(
        topology,
        old_history,
        quadrature,
        projection.scalar_normal_speed,
    )
    transport = assemble_owned_actual_wake_p2_transport(
        topology,
        old_history,
        quadrature,
        physical - mesh_velocity,
    )
    return ReleasedOldMaterialStage(
        history=old_history,
        topology=topology,
        p1_geometry_velocity=projection.dof_velocity.copy(),
        transport_operator=transport,
        weak_rank_deficiency=projection.report.rank_deficiency,
        weak_condition_number=projection.report.condition_number,
        weak_orthogonality_relative_residual=(
            projection.report.weak_orthogonality_relative_residual
        ),
        weak_surface_L2_residual=(
            projection.report.relative_surface_L2_residual
        ),
        transport_relative_normal_component_max=(
            transport.maximum_relative_velocity_normal_component
        ),
    )


def _maximum_roundtrip(
    stage: ActualWakeCoupledNewbornStage,
) -> float:
    state = stage.topology.global_p2_state(
        stage.solution.wake_history
    )
    rebuilt = stage.topology.rebuild_history(
        stage.solution.wake_history,
        state,
    )
    return float(
        np.max(
            np.abs(stage.topology.global_p2_state(rebuilt) - state),
            initial=0.0,
        )
    )


def advance_actual_wake_repeated_insertion_midpoint(
    mesh: ClosedTriangularMesh,
    body_topology,
    initial_solution: ActualBoundaryBodyWakeSolution,
    attachment: MaterialWakeCutAttachment,
    *,
    timestep: float,
    physical_velocity_provider: PhysicalVelocityProvider,
    transport_quadrature_order: int = 7,
    boundary_quadrature_order: int = 10,
    step_index: int = 0,
) -> ActualWakeRepeatedInsertionStep:
    """Advance one actual topology-growth step without newborn initial data."""
    if not isinstance(initial_solution, ActualBoundaryBodyWakeSolution):
        raise ActualWakeStageTopologyError(
            "initial_solution must be an actual body-wake solution"
        )
    if not np.isfinite(timestep) or timestep <= 0.0:
        raise ActualWakeStageTopologyError(
            "timestep must be finite and strictly positive"
        )
    if not isinstance(step_index, (int, np.integer)) or step_index < 0:
        raise ActualWakeStageTopologyError(
            "step_index must be a non-negative integer"
        )
    history0 = initial_solution.wake_history
    band_count = len(history0.bands)
    topology0 = actual_wake_stage_topology(
        history0,
        body_attachment_id="canonical-body-cut",
    )
    geometry_snapshot = tuple(
        band.surface.vertices.copy() for band in history0.bands
    )
    scalar_snapshot = tuple(
        band.potential_jump_rows.copy() for band in history0.bands
    )
    released0 = _released_old_material_stage(
        initial_solution,
        history0,
        physical_velocity_provider=physical_velocity_provider,
        quadrature_order=int(transport_quadrature_order),
    )
    geometry0 = topology0.p1_vertices.copy()
    state0 = topology0.global_p2_state(history0)
    body_p1 = topology0.boundary_roles.body_attachment_p1_dofs
    body_edge = geometry0[body_p1].copy()
    start_time = float(history0.bands[-1].time_nodes[-1])

    rate0 = released0.transport_operator.rate(state0)
    state_half_target = state0 + 0.5 * float(timestep) * rate0
    geometry_half_target = (
        geometry0
        + 0.5
        * float(timestep)
        * released0.p1_geometry_velocity
    )
    old_half_history = _rebuild_old_material(
        history0,
        topology0,
        geometry_half_target,
        state_half_target,
    )
    old_half_topology = actual_wake_stage_topology(
        old_half_history,
        body_attachment_id="released-material-seam",
    )
    seam_half = old_half_topology.chronological_rows(
        old_half_history
    )[-1]
    half_transition = augment_actual_wake_with_newborn_band(
        old_half_history,
        old_half_topology,
        released_edge=geometry_half_target[body_p1],
        current_body_edge=body_edge,
        time_nodes=np.array(
            (
                start_time,
                start_time + 0.25 * float(timestep),
                start_time + 0.5 * float(timestep),
            )
        ),
        midpoint_trace=seam_half,
        current_trace=seam_half,
        sheet_id=f"S3ac-half-{int(step_index)}",
    )
    half_stage = solve_actual_wake_coupled_newborn_trace_direct(
        mesh,
        body_topology,
        half_transition,
        attachment,
        incident_velocity=initial_solution.incident_velocity,
        wall_velocity=initial_solution.wall_velocity,
        copy_counterfactual_trace=seam_half,
        boundary_quadrature_order=int(boundary_quadrature_order),
    )

    old_mid_history = _history_prefix(
        half_stage.solution,
        band_count,
    )
    released_mid = _released_old_material_stage(
        half_stage.solution,
        old_mid_history,
        physical_velocity_provider=physical_velocity_provider,
        quadrature_order=int(transport_quadrature_order),
    )
    state_mid = released_mid.topology.global_p2_state(
        old_mid_history
    )
    geometry_mid = released_mid.topology.p1_vertices
    full_rate = released_mid.transport_operator.rate(state_mid)
    state_end_target = state0 + float(timestep) * full_rate
    geometry_end_target = (
        geometry0
        + float(timestep)
        * released_mid.p1_geometry_velocity
    )
    old_end_history = _rebuild_old_material(
        history0,
        topology0,
        geometry_end_target,
        state_end_target,
    )
    old_end_topology = actual_wake_stage_topology(
        old_end_history,
        body_attachment_id="released-material-seam",
    )
    seam_end = old_end_topology.chronological_rows(
        old_end_history
    )[-1]
    full_transition = augment_actual_wake_with_newborn_band(
        old_end_history,
        old_end_topology,
        released_edge=geometry_end_target[body_p1],
        current_body_edge=body_edge,
        time_nodes=np.array(
            (
                start_time,
                start_time + 0.5 * float(timestep),
                start_time + float(timestep),
            )
        ),
        midpoint_trace=half_stage.solved_trace,
        current_trace=half_stage.solved_trace,
        sheet_id=f"S3ac-full-{int(step_index)}",
    )
    endpoint_stage = solve_actual_wake_coupled_newborn_trace_direct(
        mesh,
        body_topology,
        full_transition,
        attachment,
        incident_velocity=initial_solution.incident_velocity,
        wall_velocity=initial_solution.wall_velocity,
        copy_counterfactual_trace=half_stage.solved_trace,
        boundary_quadrature_order=int(boundary_quadrature_order),
    )

    endpoint_old_history = _history_prefix(
        endpoint_stage.solution,
        band_count,
    )
    endpoint_old_topology = actual_wake_stage_topology(
        endpoint_old_history,
        body_attachment_id="released-material-seam",
    )
    state_end = endpoint_old_topology.global_p2_state(
        endpoint_old_history
    )
    geometry_end = endpoint_old_topology.p1_vertices
    half_mass = released0.transport_operator.mass_matrix
    half_advection = released0.transport_operator.advection_matrix
    half_terms = (
        half_mass @ (state_mid - state0),
        0.5 * float(timestep) * (half_advection @ state0),
    )
    full_mass = released_mid.transport_operator.mass_matrix
    full_advection = released_mid.transport_operator.advection_matrix
    full_terms = (
        full_mass @ (state_end - state0),
        float(timestep) * (full_advection @ state_mid),
    )
    birth = solve_p2_characteristic_birth_slab(
        np.vstack(
            (
                seam_end,
                half_stage.solved_trace,
                endpoint_stage.solved_trace,
            )
        ),
        timestep=float(timestep),
        convection_speed=1.0,
    )
    expected_birth = np.vstack(
        (
            seam_end,
            half_stage.solved_trace,
            endpoint_stage.solved_trace,
        )
    )
    birth_identity = float(
        np.max(
            np.abs(
                birth.endpoint_chronological_rows - expected_birth
            ),
            initial=0.0,
        )
    )
    stage_reports = (
        half_stage.report,
        endpoint_stage.report,
    )
    histories = (
        half_stage.solution.wake_history,
        endpoint_stage.solution.wake_history,
    )
    seam = max(
        max(
            report.max_geometry_gap,
            report.max_trace_jump,
            report.max_time_gap,
        )
        for report in (
            history.continuity_report() for history in histories
        )
    )
    mutation = 0.0
    for band, geometry, scalar in zip(
        history0.bands,
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
    return ActualWakeRepeatedInsertionStep(
        half_stage=half_stage,
        endpoint_stage=endpoint_stage,
        report=ActualWakeRepeatedInsertionStepReport(
            initial_band_count=band_count,
            final_band_count=len(
                endpoint_stage.solution.wake_history.bands
            ),
            band_increment_mismatch=abs(
                len(endpoint_stage.solution.wake_history.bands)
                - (band_count + 1)
            ),
            old_p1_identity_count=len(geometry0),
            old_p2_identity_count=len(state0),
            half_old_transport_normalized_residual=(
                _normalized_balance(sum(half_terms), *half_terms)
            ),
            full_old_transport_normalized_residual=(
                _normalized_balance(sum(full_terms), *full_terms)
            ),
            old_mass_rank_deficiency_max=max(
                len(state0)
                - released0.transport_operator.mass_rank,
                len(state_mid)
                - released_mid.transport_operator.mass_rank,
            ),
            old_mass_condition_number_max=max(
                released0.transport_operator.mass_condition_number,
                released_mid.transport_operator.mass_condition_number,
            ),
            actual_boundary_relative_weak_residual_max=max(
                float(half_stage.solution.relative_weak_residual),
                float(
                    endpoint_stage.solution.relative_weak_residual
                ),
                *(
                    report.maximum_actual_boundary_relative_weak_residual
                    for report in stage_reports
                ),
            ),
            algebraic_trace_residual_abs_max=max(
                report.algebraic_trace_residual
                for report in stage_reports
            ),
            free_state_preservation_abs_max=max(
                report.free_state_preservation_error
                for report in stage_reports
            ),
            characteristic_birth_identity_abs_max=birth_identity,
            chronological_seam_abs_max=seam,
            p2_roundtrip_abs_max=max(
                _maximum_roundtrip(half_stage),
                _maximum_roundtrip(endpoint_stage),
            ),
            half_geometry_identity_abs_max=float(
                np.max(
                    np.abs(geometry_mid - geometry_half_target),
                    initial=0.0,
                )
            ),
            full_geometry_identity_abs_max=float(
                np.max(
                    np.abs(geometry_end - geometry_end_target),
                    initial=0.0,
                )
            ),
            minimum_newborn_face_area=min(
                half_transition.report.minimum_newborn_face_area,
                full_transition.report.minimum_newborn_face_area,
            ),
            old_geometry_change_abs_max=float(
                np.max(
                    np.linalg.norm(
                        geometry_end - geometry0,
                        axis=1,
                    ),
                    initial=0.0,
                )
            ),
            old_scalar_change_abs_max=float(
                np.max(np.abs(state_end - state0), initial=0.0)
            ),
            input_state_mutation_abs_max=mutation,
            inferred_scalar_count=0,
            geometry_iteration_count=0,
        ),
    )
