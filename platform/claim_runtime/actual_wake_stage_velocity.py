"""Owner-aware fixed-stage velocity discretization for an actual wake.

The module consumes an already evaluated physical sheet velocity.  Free P1
geometry DOFs receive only the local vertex-star normal limit; newest body
attachment DOFs receive prescribed body kinematics.  A separate owned
quadrature feeds the unchanged consistent-P2 ALE mass/advection form.

There is no field closure, time advance, scalar update, pressure or force.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .actual_body_wake_velocity import WakeSheetQuery
from .actual_wake_stage_topology import (
    ActualWakeStageTopology,
    ActualWakeStageTopologyError,
)
from .distributed_doublet import (
    MaterialWakeHistory,
    QuadraticDoubletElement,
    _triangle_quadrature,
    p2_shape_values,
)
from .p2_surface_material_transport import (
    P2SurfaceMaterialTransportOperator,
)


@dataclass(frozen=True)
class OwnedWakeQuadrature:
    """One explicit owner query and its per-global-face integration rows."""

    query: WakeSheetQuery
    quadrature_order: int
    reference_weights: np.ndarray
    face_query_rows: tuple[np.ndarray, ...]

    def __post_init__(self) -> None:
        weights = np.asarray(self.reference_weights, dtype=float)
        if (
            not isinstance(self.query, WakeSheetQuery)
            or not isinstance(self.quadrature_order, (int, np.integer))
            or int(self.quadrature_order) < 2
            or weights.ndim != 1
            or len(weights) == 0
            or not np.all(np.isfinite(weights))
            or np.any(weights <= 0.0)
        ):
            raise ActualWakeStageTopologyError(
                "owned wake quadrature has invalid query/order/weights"
            )
        rows = tuple(
            np.asarray(value, dtype=np.int64).copy()
            for value in self.face_query_rows
        )
        if (
            not rows
            or any(
                value.ndim != 1
                or len(value) != len(weights)
                or len(np.unique(value)) != len(value)
                or np.any(value < 0)
                or np.any(value >= len(self.query.points))
                for value in rows
            )
        ):
            raise ActualWakeStageTopologyError(
                "owned wake face-query rows are invalid"
            )
        joined = np.concatenate(rows)
        if not np.array_equal(
            np.sort(joined),
            np.arange(len(self.query.points), dtype=np.int64),
        ):
            raise ActualWakeStageTopologyError(
                "owned wake face-query rows do not partition the query"
            )
        object.__setattr__(self, "quadrature_order", int(self.quadrature_order))
        object.__setattr__(self, "reference_weights", weights.copy())
        object.__setattr__(self, "face_query_rows", rows)


@dataclass(frozen=True)
class ActualWakeNormalProjectionReport:
    free_p1_dof_count: int
    prescribed_body_p1_dof_count: int
    maximum_condition_number: float
    maximum_absolute_residual: float
    maximum_relative_residual: float
    maximum_input_normal_speed: float
    maximum_query_reconstruction_error: float
    full_rank: bool
    gauge: str = "free_vertex_star_normal_plus_body_essential"


@dataclass(frozen=True)
class ActualWakeNormalGeometryVelocity:
    topology: ActualWakeStageTopology
    dof_velocity: np.ndarray
    dof_normals: np.ndarray
    report: ActualWakeNormalProjectionReport

    def band_vertex_velocity(self, band_index: int) -> np.ndarray:
        if band_index < 0 or band_index >= self.topology.band_count:
            raise ActualWakeStageTopologyError("invalid material band index")
        return self.dof_velocity[
            self.topology.band_p1_vertex_dofs[band_index]
        ].copy()

    def evaluate_query(
        self,
        history: MaterialWakeHistory,
        query: WakeSheetQuery,
        *,
        tolerance: float = 2.0e-12,
    ) -> np.ndarray:
        _validate_owned_query(
            self.topology,
            history,
            query,
            tolerance=tolerance,
        )
        result = np.empty_like(query.points)
        offsets = _band_face_offsets(history)
        for row, (patch, face, barycentric) in enumerate(
            zip(
                query.patch_indices,
                query.face_indices,
                query.barycentric,
                strict=True,
            )
        ):
            global_face = offsets[int(patch)] + int(face)
            result[row] = (
                barycentric
                @ self.dof_velocity[
                    self.topology.p1_faces[global_face]
                ]
            )
        return result


def _band_face_offsets(
    history: MaterialWakeHistory,
) -> tuple[int, ...]:
    offsets = []
    offset = 0
    for band in history.bands:
        offsets.append(offset)
        offset += len(band.surface.faces)
    return tuple(offsets)


def _validate_owned_query(
    topology: ActualWakeStageTopology,
    history: MaterialWakeHistory,
    query: WakeSheetQuery,
    *,
    tolerance: float,
) -> float:
    topology._validate_history(history, tolerance=tolerance)
    if not isinstance(query, WakeSheetQuery):
        raise ActualWakeStageTopologyError(
            "query must be WakeSheetQuery"
        )
    reconstructed = np.empty_like(query.points)
    for row, (patch_index, face_index, barycentric) in enumerate(
        zip(
            query.patch_indices,
            query.face_indices,
            query.barycentric,
            strict=True,
        )
    ):
        patch = int(patch_index)
        face = int(face_index)
        if (
            patch >= len(history.bands)
            or face >= len(history.bands[patch].surface.faces)
        ):
            raise ActualWakeStageTopologyError(
                "owned query patch/face is out of range"
            )
        band = history.bands[patch]
        reconstructed[row] = (
            barycentric
            @ band.surface.vertices[band.surface.faces[face]]
        )
    error = float(
        np.max(
            np.linalg.norm(reconstructed - query.points, axis=1),
            initial=0.0,
        )
    )
    if error > tolerance:
        raise ActualWakeStageTopologyError(
            "owned query points disagree with patch/face/barycentric identity"
        )
    return error


def actual_wake_owned_quadrature(
    topology: ActualWakeStageTopology,
    history: MaterialWakeHistory,
    *,
    quadrature_order: int,
    query_id: str = "actual-wake-owned-P2-quadrature",
    tolerance: float = 1.0e-12,
) -> OwnedWakeQuadrature:
    """Build an owner-preserving query at every triangle quadrature point."""
    topology._validate_history(history, tolerance=tolerance)
    barycentric, weights = _triangle_quadrature(int(quadrature_order))
    points = []
    patch_indices = []
    face_indices = []
    barycentric_rows = []
    face_rows = []
    row = 0
    for patch_index, band in enumerate(history.bands):
        for face_index, face in enumerate(band.surface.faces):
            count = len(barycentric)
            points.append(
                barycentric @ band.surface.vertices[face]
            )
            patch_indices.extend([patch_index] * count)
            face_indices.extend([face_index] * count)
            barycentric_rows.append(barycentric)
            face_rows.append(
                np.arange(row, row + count, dtype=np.int64)
            )
            row += count
    query = WakeSheetQuery(
        points=np.vstack(points),
        patch_indices=np.asarray(patch_indices, dtype=np.int64),
        face_indices=np.asarray(face_indices, dtype=np.int64),
        barycentric=np.vstack(barycentric_rows),
        query_id=query_id,
    )
    _validate_owned_query(
        topology,
        history,
        query,
        tolerance=tolerance,
    )
    return OwnedWakeQuadrature(
        query=query,
        quadrature_order=int(quadrature_order),
        reference_weights=weights,
        face_query_rows=tuple(face_rows),
    )


def project_actual_wake_vertex_star_normal_velocity(
    topology: ActualWakeStageTopology,
    history: MaterialWakeHistory,
    query: WakeSheetQuery,
    collocation_velocity: Any,
    *,
    body_attachment_velocity: Any,
    query_tolerance: float = 2.0e-12,
    relative_rank_tolerance: float | None = None,
) -> ActualWakeNormalGeometryVelocity:
    """Project free normal geometry speed and prescribe the body edge."""
    reconstruction = _validate_owned_query(
        topology,
        history,
        query,
        tolerance=query_tolerance,
    )
    velocity = np.asarray(collocation_velocity, dtype=float)
    if velocity.shape != query.points.shape or not np.all(
        np.isfinite(velocity)
    ):
        raise ActualWakeStageTopologyError(
            "collocation_velocity must match the owned query"
        )
    body_dofs = topology.boundary_roles.body_attachment_p1_dofs
    body_velocity = np.asarray(body_attachment_velocity, dtype=float)
    if body_velocity.shape != (len(body_dofs), 3) or not np.all(
        np.isfinite(body_velocity)
    ):
        raise ActualWakeStageTopologyError(
            "body_attachment_velocity has incompatible shape or values"
        )
    dof_count = len(topology.p1_vertices)
    body_mask = np.zeros(dof_count, dtype=bool)
    body_mask[body_dofs] = True
    free_dofs = np.flatnonzero(~body_mask)
    dof_normal_sum = np.zeros_like(topology.p1_vertices)
    incident_faces: list[list[int]] = [[] for _ in range(dof_count)]
    for face_index, face in enumerate(topology.p1_faces):
        element = QuadraticDoubletElement(
            topology.p1_vertices[face],
            np.zeros(6),
        )
        for dof in face:
            dof_normal_sum[int(dof)] += element.area_vector
            incident_faces[int(dof)].append(face_index)
    normal_norm = np.linalg.norm(dof_normal_sum, axis=1)
    if np.any(normal_norm <= np.finfo(float).eps):
        raise ActualWakeStageTopologyError(
            "actual-wake P1 vertex normal is undefined"
        )
    dof_normals = dof_normal_sum / normal_norm[:, None]
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
            "relative_rank_tolerance must be finite and non-negative"
        )
    offsets = _band_face_offsets(history)
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
            "owned geometry query omitted a wake face"
        )

    dof_velocity = np.empty_like(topology.p1_vertices)
    dof_velocity[body_dofs] = body_velocity
    maximum_condition = 0.0
    maximum_abs_residual = 0.0
    maximum_rel_residual = 0.0
    maximum_input = 0.0
    for dof in free_dofs:
        records = incident_faces[int(dof)]
        neighbor_dofs = sorted(
            {
                int(other)
                for face_index in records
                for other in topology.p1_faces[face_index]
                if int(other) != int(dof)
            }
        )
        normal = dof_normals[dof]
        tangent1 = None
        for neighbor in neighbor_dofs:
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
                f"actual-wake dof {dof} has no tangent direction"
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
        sample_speed_parts = []
        for face in records:
            face_rows = np.asarray(
                query_rows_by_face[face],
                dtype=np.int64,
            )
            element = QuadraticDoubletElement(
                topology.p1_vertices[topology.p1_faces[face]],
                np.zeros(6),
            )
            sample_speed_parts.append(
                velocity[face_rows] @ element.normal
            )
        sample_speed = np.concatenate(sample_speed_parts)
        singular = np.linalg.svd(
            design,
            compute_uv=False,
            full_matrices=False,
        )
        rank = int(
            np.count_nonzero(
                singular
                > relative_rank_tolerance * singular[0]
            )
        )
        if rank != 3:
            raise ActualWakeStageTopologyError(
                "actual-wake vertex-star extrapolation is rank deficient: "
                f"dof={dof}, rank={rank}"
            )
        coefficients, _, solved_rank, solved_singular = np.linalg.lstsq(
            design,
            sample_speed,
            rcond=relative_rank_tolerance,
        )
        if int(solved_rank) != 3:
            raise ActualWakeStageTopologyError(
                "actual-wake vertex-star solve lost rank"
            )
        residual = np.abs(design @ coefficients - sample_speed)
        scale = np.maximum(np.abs(sample_speed), np.finfo(float).eps)
        dof_velocity[dof] = coefficients[0] * normal
        maximum_condition = max(
            maximum_condition,
            float(solved_singular[0] / solved_singular[-1]),
        )
        maximum_abs_residual = max(
            maximum_abs_residual,
            float(np.max(residual, initial=0.0)),
        )
        maximum_rel_residual = max(
            maximum_rel_residual,
            float(np.max(residual / scale, initial=0.0)),
        )
        maximum_input = max(
            maximum_input,
            float(np.max(np.abs(sample_speed), initial=0.0)),
        )
    return ActualWakeNormalGeometryVelocity(
        topology=topology,
        dof_velocity=dof_velocity,
        dof_normals=dof_normals,
        report=ActualWakeNormalProjectionReport(
            free_p1_dof_count=len(free_dofs),
            prescribed_body_p1_dof_count=len(body_dofs),
            maximum_condition_number=maximum_condition,
            maximum_absolute_residual=maximum_abs_residual,
            maximum_relative_residual=maximum_rel_residual,
            maximum_input_normal_speed=maximum_input,
            maximum_query_reconstruction_error=reconstruction,
            full_rank=True,
        ),
    )


def assemble_owned_actual_wake_p2_transport(
    topology: ActualWakeStageTopology,
    history: MaterialWakeHistory,
    quadrature: OwnedWakeQuadrature,
    relative_velocity: Any,
    *,
    query_tolerance: float = 2.0e-12,
) -> P2SurfaceMaterialTransportOperator:
    """Assemble the frozen consistent-P2 form from owned stage velocities."""
    if not isinstance(quadrature, OwnedWakeQuadrature):
        raise ActualWakeStageTopologyError(
            "quadrature must be OwnedWakeQuadrature"
        )
    _validate_owned_query(
        topology,
        history,
        quadrature.query,
        tolerance=query_tolerance,
    )
    if len(quadrature.face_query_rows) != len(topology.p1_faces):
        raise ActualWakeStageTopologyError(
            "owned quadrature face count does not match topology"
        )
    velocity = np.asarray(relative_velocity, dtype=float)
    if velocity.shape != quadrature.query.points.shape or not np.all(
        np.isfinite(velocity)
    ):
        raise ActualWakeStageTopologyError(
            "relative_velocity must match the owned quadrature query"
        )
    count = topology.p2_topology.degree_of_freedom_count
    mass = np.zeros((count, count), dtype=float)
    advection = np.zeros_like(mass)
    maximum_normal = 0.0
    for face_index, rows in enumerate(quadrature.face_query_rows):
        face = topology.p1_faces[face_index]
        element = QuadraticDoubletElement(
            topology.p1_vertices[face],
            np.zeros(6),
        )
        barycentric = quadrature.query.barycentric[rows]
        points = barycentric @ element.vertices
        if (
            np.max(
                np.linalg.norm(
                    points - quadrature.query.points[rows],
                    axis=1,
                ),
                initial=0.0,
            )
            > query_tolerance
        ):
            raise ActualWakeStageTopologyError(
                "owned quadrature rows do not match their global face"
            )
        face_velocity = velocity[rows]
        normal_component = face_velocity @ element.normal
        maximum_normal = max(
            maximum_normal,
            float(
                np.max(
                    np.abs(normal_component),
                    initial=0.0,
                )
            ),
        )
        tangential_velocity = (
            face_velocity
            - normal_component[:, None] * element.normal
        )
        shape = p2_shape_values(barycentric)
        gradients = element.shape_gradients(barycentric)
        advective_derivative = np.einsum(
            "qi,qji->qj",
            tangential_velocity,
            gradients,
        )
        weights = (
            quadrature.reference_weights
            * np.linalg.norm(element.area_vector)
        )
        local_mass = np.einsum(
            "qi,q,qj->ij",
            shape,
            weights,
            shape,
        )
        local_advection = np.einsum(
            "qi,q,qj->ij",
            shape,
            weights,
            advective_derivative,
        )
        dofs = topology.p2_topology.local_to_global[face_index]
        mass[np.ix_(dofs, dofs)] += local_mass
        advection[np.ix_(dofs, dofs)] += local_advection
    rank = int(np.linalg.matrix_rank(mass))
    condition = float(np.linalg.cond(mass))
    if rank != count or not np.isfinite(condition):
        raise ActualWakeStageTopologyError(
            "owned actual-wake continuous-P2 mass matrix is singular"
        )
    constant_rate = np.linalg.solve(
        mass,
        -(advection @ np.ones(count)),
    )
    return P2SurfaceMaterialTransportOperator(
        topology=topology.p2_topology,
        mass_matrix=mass,
        advection_matrix=advection,
        mass_rank=rank,
        mass_condition_number=condition,
        maximum_relative_velocity_normal_component=maximum_normal,
        constant_rate_residual=float(
            np.max(np.abs(constant_rate), initial=0.0)
        ),
    )
