"""Typed release/newborn transition for an actual chronological wake.

The retiring body-attachment row is evaluated with the one-sided physical
sheet-normal limit and becomes an internal material seam.  A new current
body row and two explicitly supplied temporal P2 rows are then appended.

This module changes topology and state dimension only.  It does not infer a
newborn strength, solve a stage equation, add an offset, or compute pressure,
force, LESP, damping, stabilization, targets, or structural state.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .actual_body_wake_velocity import WakeSheetQuery
from .actual_wake_stage_topology import (
    ActualWakeStageTopology,
    ActualWakeStageTopologyError,
    actual_wake_stage_topology,
)
from .distributed_doublet import (
    MaterialWakeHistory,
    QuadraticDoubletElement,
    newborn_material_wake_band,
)


@dataclass(frozen=True)
class ActualWakeAttachmentReleaseReport:
    released_p1_dof_count: int
    minimum_speed: float
    maximum_speed: float
    maximum_tangential_speed: float
    maximum_condition_number: float
    maximum_absolute_fit_residual: float
    maximum_relative_fit_residual: float
    rank_deficiency: int
    query_reconstruction_error: float
    gauge: str = "one_sided_vertex_star_normal"


@dataclass(frozen=True)
class ActualWakeAttachmentRelease:
    p1_dofs: np.ndarray
    velocity: np.ndarray
    normals: np.ndarray
    report: ActualWakeAttachmentReleaseReport

    def __post_init__(self) -> None:
        dofs = np.asarray(self.p1_dofs, dtype=np.int64)
        velocity = np.asarray(self.velocity, dtype=float)
        normals = np.asarray(self.normals, dtype=float)
        if (
            dofs.ndim != 1
            or len(dofs) == 0
            or len(np.unique(dofs)) != len(dofs)
            or velocity.shape != (len(dofs), 3)
            or normals.shape != velocity.shape
            or not np.all(np.isfinite(velocity))
            or not np.all(np.isfinite(normals))
        ):
            raise ActualWakeStageTopologyError(
                "attachment release has incompatible DOFs or vectors"
            )
        object.__setattr__(self, "p1_dofs", dofs.copy())
        object.__setattr__(self, "velocity", velocity.copy())
        object.__setattr__(self, "normals", normals.copy())


@dataclass(frozen=True)
class ActualWakeNewbornTransitionReport:
    initial_band_count: int
    augmented_band_count: int
    initial_p1_dof_count: int
    augmented_p1_dof_count: int
    initial_p2_dof_count: int
    augmented_p2_dof_count: int
    old_p1_injection_error: float
    old_p2_injection_error: float
    newborn_upstream_trace_error: float
    newborn_midpoint_trace_error: float
    newborn_current_trace_error: float
    chronological_geometry_seam_error: float
    chronological_scalar_seam_error: float
    p2_roundtrip_error: float
    boundary_role_overlap_count: int
    minimum_newborn_face_area: float
    input_state_mutation: float
    inferred_scalar_count: int
    epsilon_offset: float


@dataclass(frozen=True)
class ActualWakeNewbornTransition:
    initial_topology: ActualWakeStageTopology
    augmented_topology: ActualWakeStageTopology
    augmented_history: MaterialWakeHistory
    old_p1_to_augmented: np.ndarray
    old_p2_to_augmented: np.ndarray
    released_p1_dofs: np.ndarray
    released_p2_dofs: np.ndarray
    new_body_p1_dofs: np.ndarray
    new_body_p2_dofs: np.ndarray
    report: ActualWakeNewbornTransitionReport


def _owned_query_reconstruction_error(
    topology: ActualWakeStageTopology,
    history: MaterialWakeHistory,
    query: WakeSheetQuery,
) -> float:
    topology._validate_history(history, tolerance=2.0e-12)
    if not isinstance(query, WakeSheetQuery):
        raise ActualWakeStageTopologyError(
            "release query must be a WakeSheetQuery"
        )
    reconstructed = np.empty_like(query.points)
    for row, (patch, face, barycentric) in enumerate(
        zip(
            query.patch_indices,
            query.face_indices,
            query.barycentric,
            strict=True,
        )
    ):
        patch_index = int(patch)
        face_index = int(face)
        if (
            patch_index < 0
            or patch_index >= len(history.bands)
            or face_index < 0
            or face_index
            >= len(history.bands[patch_index].surface.faces)
        ):
            raise ActualWakeStageTopologyError(
                "release query owner is outside the wake history"
            )
        band = history.bands[patch_index]
        reconstructed[row] = (
            barycentric
            @ band.surface.vertices[band.surface.faces[face_index]]
        )
    error = float(
        np.max(
            np.linalg.norm(reconstructed - query.points, axis=1),
            initial=0.0,
        )
    )
    if error > 2.0e-12:
        raise ActualWakeStageTopologyError(
            "release query does not reconstruct from owner identities"
        )
    return error


def project_released_attachment_normal_velocity(
    topology: ActualWakeStageTopology,
    history: MaterialWakeHistory,
    query: WakeSheetQuery,
    collocation_velocity: Any,
    *,
    relative_rank_tolerance: float | None = None,
) -> ActualWakeAttachmentRelease:
    """Extrapolate the one-sided normal velocity at the retiring body row."""
    reconstruction = _owned_query_reconstruction_error(
        topology,
        history,
        query,
    )
    physical_velocity = np.asarray(collocation_velocity, dtype=float)
    if (
        physical_velocity.shape != query.points.shape
        or not np.all(np.isfinite(physical_velocity))
    ):
        raise ActualWakeStageTopologyError(
            "release collocation_velocity must match the owned query"
        )
    if relative_rank_tolerance is None:
        relative_rank_tolerance = (
            64.0
            * np.finfo(float).eps
            * max(len(query.points), 3)
        )
    if (
        relative_rank_tolerance < 0.0
        or not np.isfinite(relative_rank_tolerance)
    ):
        raise ActualWakeStageTopologyError(
            "release rank tolerance must be finite and non-negative"
        )

    dof_count = len(topology.p1_vertices)
    normal_sum = np.zeros_like(topology.p1_vertices)
    incident_faces: list[list[int]] = [[] for _ in range(dof_count)]
    for face_index, face in enumerate(topology.p1_faces):
        element = QuadraticDoubletElement(
            topology.p1_vertices[face],
            np.zeros(6),
        )
        for dof in face:
            normal_sum[int(dof)] += element.area_vector
            incident_faces[int(dof)].append(face_index)
    normal_norm = np.linalg.norm(normal_sum, axis=1)
    if np.any(normal_norm <= np.finfo(float).eps):
        raise ActualWakeStageTopologyError(
            "release vertex normal is undefined"
        )
    dof_normals = normal_sum / normal_norm[:, None]

    offsets = []
    offset = 0
    for band in history.bands:
        offsets.append(offset)
        offset += len(band.surface.faces)
    query_rows_by_face: list[list[int]] = [
        [] for _ in topology.p1_faces
    ]
    for query_row, (patch, face) in enumerate(
        zip(query.patch_indices, query.face_indices, strict=True)
    ):
        global_face = offsets[int(patch)] + int(face)
        query_rows_by_face[global_face].append(query_row)
    if any(not rows for rows in query_rows_by_face):
        raise ActualWakeStageTopologyError(
            "release query omitted a wake face"
        )

    release_dofs = (
        topology.boundary_roles.body_attachment_p1_dofs.copy()
    )
    release_velocity = np.empty((len(release_dofs), 3), dtype=float)
    release_normals = dof_normals[release_dofs].copy()
    maximum_condition = 0.0
    maximum_absolute = 0.0
    maximum_relative = 0.0
    rank_deficiency = 0
    for local_index, dof_value in enumerate(release_dofs):
        dof = int(dof_value)
        records = incident_faces[dof]
        if not records:
            raise ActualWakeStageTopologyError(
                "retiring attachment DOF has no incident wake face"
            )
        neighbors = sorted(
            {
                int(other)
                for face_index in records
                for other in topology.p1_faces[face_index]
                if int(other) != dof
            }
        )
        normal = dof_normals[dof]
        tangent1 = None
        for neighbor in neighbors:
            candidate = (
                topology.p1_vertices[neighbor]
                - topology.p1_vertices[dof]
            )
            candidate -= np.dot(candidate, normal) * normal
            norm = np.linalg.norm(candidate)
            if norm > 64.0 * np.finfo(float).eps:
                tangent1 = candidate / norm
                break
        if tangent1 is None:
            raise ActualWakeStageTopologyError(
                "retiring attachment has no local tangent direction"
            )
        tangent2 = np.cross(normal, tangent1)
        rows = np.concatenate(
            [
                np.asarray(query_rows_by_face[face], dtype=np.int64)
                for face in records
            ]
        )
        delta = query.points[rows] - topology.p1_vertices[dof]
        design = np.column_stack(
            (
                np.ones(len(rows)),
                delta @ tangent1,
                delta @ tangent2,
            )
        )
        samples = []
        for face_index in records:
            face_rows = np.asarray(
                query_rows_by_face[face_index],
                dtype=np.int64,
            )
            element = QuadraticDoubletElement(
                topology.p1_vertices[
                    topology.p1_faces[face_index]
                ],
                np.zeros(6),
            )
            samples.append(
                physical_velocity[face_rows] @ element.normal
            )
        sample_speed = np.concatenate(samples)
        coefficients, _, rank, singular = np.linalg.lstsq(
            design,
            sample_speed,
            rcond=relative_rank_tolerance,
        )
        rank_deficiency = max(rank_deficiency, 3 - int(rank))
        if int(rank) != 3 or singular[-1] <= 0.0:
            raise ActualWakeStageTopologyError(
                "retiring attachment one-sided fit is rank deficient"
            )
        residual = np.abs(design @ coefficients - sample_speed)
        scale = np.maximum(np.abs(sample_speed), np.finfo(float).eps)
        release_velocity[local_index] = coefficients[0] * normal
        maximum_condition = max(
            maximum_condition,
            float(singular[0] / singular[-1]),
        )
        maximum_absolute = max(
            maximum_absolute,
            float(np.max(residual, initial=0.0)),
        )
        maximum_relative = max(
            maximum_relative,
            float(np.max(residual / scale, initial=0.0)),
        )
    normal_speed = np.einsum(
        "ij,ij->i",
        release_velocity,
        release_normals,
    )
    tangent = (
        release_velocity - normal_speed[:, None] * release_normals
    )
    speed = np.linalg.norm(release_velocity, axis=1)
    return ActualWakeAttachmentRelease(
        p1_dofs=release_dofs,
        velocity=release_velocity,
        normals=release_normals,
        report=ActualWakeAttachmentReleaseReport(
            released_p1_dof_count=len(release_dofs),
            minimum_speed=float(np.min(speed)),
            maximum_speed=float(np.max(speed)),
            maximum_tangential_speed=float(
                np.max(np.linalg.norm(tangent, axis=1), initial=0.0)
            ),
            maximum_condition_number=maximum_condition,
            maximum_absolute_fit_residual=maximum_absolute,
            maximum_relative_fit_residual=maximum_relative,
            rank_deficiency=rank_deficiency,
            query_reconstruction_error=reconstruction,
        ),
    )


def augment_actual_wake_with_newborn_band(
    history: MaterialWakeHistory,
    topology: ActualWakeStageTopology,
    *,
    released_edge: Any,
    current_body_edge: Any,
    time_nodes: Any,
    midpoint_trace: Any,
    current_trace: Any,
    sheet_id: str,
    span_diagonal_pattern: str = "mirror_symmetric",
) -> ActualWakeNewbornTransition:
    """Release the old body row and append one explicitly valued band."""
    topology._validate_history(history, tolerance=1.0e-12)
    if topology.band_count != len(history.bands):
        raise ActualWakeStageTopologyError(
            "transition topology and history band counts differ"
        )
    if not isinstance(sheet_id, str) or not sheet_id:
        raise ActualWakeStageTopologyError(
            "newborn sheet_id must be nonempty"
        )
    span_nodes = topology.span_nodes
    cut_nodes = topology.cut_node_count
    released = np.asarray(released_edge, dtype=float)
    body_edge = np.asarray(current_body_edge, dtype=float)
    midpoint = np.asarray(midpoint_trace, dtype=float)
    current = np.asarray(current_trace, dtype=float)
    if (
        released.shape != (span_nodes, 3)
        or body_edge.shape != released.shape
        or not np.all(np.isfinite(released))
        or not np.all(np.isfinite(body_edge))
    ):
        raise ActualWakeStageTopologyError(
            "released/current edges must be finite span-node coordinates"
        )
    if (
        midpoint.shape != (cut_nodes,)
        or current.shape != (cut_nodes,)
        or not np.all(np.isfinite(midpoint))
        or not np.all(np.isfinite(current))
    ):
        raise ActualWakeStageTopologyError(
            "newborn midpoint/current traces have incompatible shape"
        )

    geometry_snapshot = tuple(
        band.surface.vertices.copy() for band in history.bands
    )
    scalar_snapshot = tuple(
        band.potential_jump_rows.copy() for band in history.bands
    )
    old_state = topology.global_p2_state(history)
    old_rows = topology.chronological_rows(history)
    upstream = old_rows[-1].copy()
    old_body_p1 = (
        topology.boundary_roles.body_attachment_p1_dofs
    )
    old_body_p2 = (
        topology.boundary_roles.body_attachment_p2_dofs
    )

    bands = list(history.bands)
    last = bands[-1]
    last_vertices = last.surface.vertices.copy()
    last_vertices[last.span_nodes :] = released
    bands[-1] = last.material_update(last_vertices)
    released_history = MaterialWakeHistory(
        history.history_id,
        tuple(bands),
    )
    released_report = released_history.continuity_report()
    if not released_report.compatible:
        raise ActualWakeStageTopologyError(
            "releasing the old body row broke wake chronology"
        )
    newborn = newborn_material_wake_band(
        sheet_id=sheet_id,
        vortex_family=history.vortex_family,
        previous_edge=released,
        current_edge=body_edge,
        time_nodes=time_nodes,
        potential_jump_rows=np.vstack(
            (upstream, midpoint, current)
        ),
        span_diagonal_pattern=span_diagonal_pattern,
    )
    augmented_history = released_history.append(newborn)
    augmented_topology = actual_wake_stage_topology(
        augmented_history,
        body_attachment_id=(
            topology.boundary_roles.body_attachment_id
        ),
    )
    augmented_state = augmented_topology.global_p2_state(
        augmented_history
    )
    augmented_rows = augmented_topology.chronological_rows(
        augmented_history
    )

    old_p1_to_augmented = np.arange(
        len(topology.p1_vertices),
        dtype=np.int64,
    )
    old_p2_to_augmented = (
        augmented_topology.chronological_to_p2_dof[
            topology.p2_dof_to_chronological
        ]
    )
    expected_old_geometry = topology.p1_vertices.copy()
    expected_old_geometry[old_body_p1] = released
    p1_error = float(
        np.max(
            np.abs(
                augmented_topology.p1_vertices[
                    old_p1_to_augmented
                ]
                - expected_old_geometry
            ),
            initial=0.0,
        )
    )
    p2_error = float(
        np.max(
            np.abs(
                augmented_state[old_p2_to_augmented] - old_state
            ),
            initial=0.0,
        )
    )
    rebuilt = augmented_topology.rebuild_history(
        augmented_history,
        augmented_state,
    )
    roundtrip = augmented_topology.global_p2_state(rebuilt)
    p2_roundtrip = float(
        np.max(
            np.abs(roundtrip - augmented_state),
            initial=0.0,
        )
    )
    history_report = augmented_history.continuity_report()
    released_p2 = old_p2_to_augmented[old_body_p2]
    new_body_p1 = (
        augmented_topology.boundary_roles
        .body_attachment_p1_dofs
    )
    new_body_p2 = (
        augmented_topology.boundary_roles
        .body_attachment_p2_dofs
    )
    role_overlap = (
        len(np.intersect1d(old_body_p1, new_body_p1))
        + len(np.intersect1d(released_p2, new_body_p2))
    )
    minimum_area = min(
        newborn.surface.element(index).area
        for index in range(len(newborn.surface))
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
    return ActualWakeNewbornTransition(
        initial_topology=topology,
        augmented_topology=augmented_topology,
        augmented_history=augmented_history,
        old_p1_to_augmented=old_p1_to_augmented,
        old_p2_to_augmented=old_p2_to_augmented,
        released_p1_dofs=old_body_p1.copy(),
        released_p2_dofs=released_p2.copy(),
        new_body_p1_dofs=new_body_p1.copy(),
        new_body_p2_dofs=new_body_p2.copy(),
        report=ActualWakeNewbornTransitionReport(
            initial_band_count=topology.band_count,
            augmented_band_count=augmented_topology.band_count,
            initial_p1_dof_count=len(topology.p1_vertices),
            augmented_p1_dof_count=len(
                augmented_topology.p1_vertices
            ),
            initial_p2_dof_count=(
                topology.p2_topology.degree_of_freedom_count
            ),
            augmented_p2_dof_count=(
                augmented_topology.p2_topology
                .degree_of_freedom_count
            ),
            old_p1_injection_error=p1_error,
            old_p2_injection_error=p2_error,
            newborn_upstream_trace_error=float(
                np.max(
                    np.abs(augmented_rows[-3] - upstream),
                    initial=0.0,
                )
            ),
            newborn_midpoint_trace_error=float(
                np.max(
                    np.abs(augmented_rows[-2] - midpoint),
                    initial=0.0,
                )
            ),
            newborn_current_trace_error=float(
                np.max(
                    np.abs(augmented_rows[-1] - current),
                    initial=0.0,
                )
            ),
            chronological_geometry_seam_error=(
                history_report.max_geometry_gap
            ),
            chronological_scalar_seam_error=(
                history_report.max_trace_jump
            ),
            p2_roundtrip_error=p2_roundtrip,
            boundary_role_overlap_count=role_overlap,
            minimum_newborn_face_area=float(minimum_area),
            input_state_mutation=mutation,
            inferred_scalar_count=0,
            epsilon_offset=0.0,
        ),
    )
