"""Typed no-force velocity ledger on an actual-boundary material wake.

The evaluator composes exactly four physical channels at explicit owned
strict-interior wake points:

    external incident + body source + body P2 doublet
    + full-wake sheet-average.

It does not move geometry, transport potential jump, iterate a coupled
solution, compute pressure/force, add a core or inspect a target load.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from .actual_boundary_body_wake import (
    ActualBoundaryBodyWakeSolution,
)
from .distributed_doublet import (
    DoubletVelocityReport,
    MaterialWakeHistory,
    QuadraticDoubletAssembly,
    QuadraticDoubletSurface,
    SheetAverageVelocityReport,
)
from .thick_body_neumann_shadow import (
    constant_source_polygon_influence,
)


class ActualBodyWakeVelocityError(ValueError):
    """Invalid ownership, source identity or field contribution."""


VALIDATED_EDGE_QUADRATURE = "target_sinh_analytic_sheet"


@dataclass(frozen=True)
class WakeSheetQuery:
    """Explicit owner identity for one set of on-sheet field points."""

    points: np.ndarray
    patch_indices: np.ndarray
    face_indices: np.ndarray
    barycentric: np.ndarray
    query_id: str

    def __post_init__(self) -> None:
        points = np.asarray(self.points, dtype=float)
        patch = np.asarray(self.patch_indices, dtype=np.int64)
        face = np.asarray(self.face_indices, dtype=np.int64)
        barycentric = np.asarray(self.barycentric, dtype=float)
        if (
            points.ndim != 2
            or points.shape[1] != 3
            or patch.shape != (len(points),)
            or face.shape != (len(points),)
            or barycentric.shape != (len(points), 3)
            or not np.all(np.isfinite(points))
            or not np.all(np.isfinite(barycentric))
            or np.any(patch < 0)
            or np.any(face < 0)
        ):
            raise ActualBodyWakeVelocityError(
                "wake query arrays have incompatible shape or values"
            )
        margin = 128.0 * np.finfo(float).eps
        if (
            np.any(barycentric <= margin)
            or np.any(barycentric >= 1.0 - margin)
            or np.max(
                np.abs(np.sum(barycentric, axis=1) - 1.0),
                initial=0.0,
            )
            > 64.0 * np.finfo(float).eps
        ):
            raise ActualBodyWakeVelocityError(
                "wake query barycentric rows must be strict interior "
                "coordinates summing to one"
            )
        if not isinstance(self.query_id, str) or not self.query_id:
            raise ActualBodyWakeVelocityError(
                "wake query_id must be nonempty"
            )
        object.__setattr__(self, "points", points.copy())
        object.__setattr__(self, "patch_indices", patch.copy())
        object.__setattr__(self, "face_indices", face.copy())
        object.__setattr__(self, "barycentric", barycentric.copy())


@dataclass(frozen=True)
class ExternalIncidentField:
    """External velocity provider with declared included source identities."""

    field_id: str
    included_source_ids: tuple[str, ...]
    velocity_provider: Callable[[np.ndarray], Any]

    def __post_init__(self) -> None:
        source_ids = tuple(self.included_source_ids)
        if (
            not isinstance(self.field_id, str)
            or not self.field_id
            or not source_ids
            or any(
                not isinstance(value, str) or not value
                for value in source_ids
            )
            or len(set(source_ids)) != len(source_ids)
            or not callable(self.velocity_provider)
        ):
            raise ActualBodyWakeVelocityError(
                "external field requires a name, unique source ids and "
                "a callable provider"
            )
        object.__setattr__(self, "included_source_ids", source_ids)

    def evaluate(self, points: np.ndarray) -> np.ndarray:
        value = np.asarray(self.velocity_provider(points.copy()), dtype=float)
        if value.shape != points.shape or not np.all(np.isfinite(value)):
            raise ActualBodyWakeVelocityError(
                "external incident provider returned incompatible data"
            )
        return value


@dataclass(frozen=True)
class ActualBodyWakeVelocityLedger:
    query: WakeSheetQuery
    external_incident: np.ndarray
    body_source: np.ndarray
    body_doublet: np.ndarray
    wake_sheet_average: np.ndarray
    total: np.ndarray
    body_doublet_report: DoubletVelocityReport
    wake_sheet_average_report: SheetAverageVelocityReport
    query_reconstruction_error: float
    wake_representation_error: float
    channel_source_ids: tuple[str, ...]

    def channel_sum(self) -> np.ndarray:
        return (
            self.external_incident
            + self.body_source
            + self.body_doublet
            + self.wake_sheet_average
        )

    def closure_error(self) -> float:
        return float(
            np.max(
                np.abs(self.total - self.channel_sum()),
                initial=0.0,
            )
        )


def material_wake_assembly(
    history: MaterialWakeHistory,
) -> QuadraticDoubletAssembly:
    """Expose one history for field evaluation with explicit open terminals."""
    if not isinstance(history, MaterialWakeHistory):
        raise ActualBodyWakeVelocityError(
            "history must be MaterialWakeHistory"
        )
    patches = history.as_patches(
        oldest_role="zero",
        newest_role="interface:external-body-cut",
        side_roles=("zero", "zero"),
    )
    return QuadraticDoubletAssembly(patches)


def wake_sheet_interior_query(
    history: MaterialWakeHistory,
    *,
    query_id: str = "all-strict-interior-wake-points",
) -> WakeSheetQuery:
    assembly = material_wake_assembly(history)
    points, patch, face, barycentric = (
        assembly.interior_collocation_points()
    )
    return WakeSheetQuery(
        points=points,
        patch_indices=patch,
        face_indices=face,
        barycentric=barycentric,
        query_id=query_id,
    )


def _validate_query(
    assembly: QuadraticDoubletAssembly,
    query: WakeSheetQuery,
) -> float:
    if not isinstance(query, WakeSheetQuery):
        raise ActualBodyWakeVelocityError(
            "query must be WakeSheetQuery"
        )
    reconstructed = np.empty_like(query.points)
    for point_index, (patch_index, face_index) in enumerate(
        zip(query.patch_indices, query.face_indices, strict=True)
    ):
        if patch_index >= len(assembly.patches):
            raise ActualBodyWakeVelocityError(
                "wake query patch owner is out of range"
            )
        surface = assembly.patches[int(patch_index)].surface
        if face_index >= len(surface.faces):
            raise ActualBodyWakeVelocityError(
                "wake query face owner is out of range"
            )
        reconstructed[point_index] = (
            query.barycentric[point_index]
            @ surface.vertices[surface.faces[int(face_index)]]
        )
    return float(
        np.max(
            np.linalg.norm(reconstructed - query.points, axis=1),
            initial=0.0,
        )
    )


def _explicit_wake_sum(
    assembly: QuadraticDoubletAssembly,
    query: WakeSheetQuery,
    *,
    quadrature_order: int,
    edge_quadrature: str,
) -> np.ndarray:
    velocity = np.zeros_like(query.points)
    for patch_index, patch in enumerate(assembly.patches):
        owner = query.patch_indices == patch_index
        if np.any(owner):
            velocity[owner] += (
                patch.surface.induced_velocity_sheet_average(
                    query.face_indices[owner],
                    query.barycentric[owner],
                    quadrature_order=quadrature_order,
                    edge_quadrature=edge_quadrature,
                )
            )
        if np.any(~owner):
            velocity[~owner] += (
                patch.surface.induced_velocity_nonowner_sheet_points(
                    query.points[~owner],
                    quadrature_order=quadrature_order,
                    edge_quadrature=edge_quadrature,
                )
            )
    return velocity


def evaluate_actual_body_wake_sheet_velocity(
    solution: ActualBoundaryBodyWakeSolution,
    query: WakeSheetQuery,
    *,
    external_incident: ExternalIncidentField,
    body_doublet_orders: tuple[int, ...] = (16, 24, 36, 52),
    wake_sheet_average_orders: tuple[int, ...] = (24, 36, 48, 64),
    absolute_tolerance: float = 1.0e-8,
    relative_tolerance: float = 1.0e-6,
    query_tolerance: float = 2.0e-12,
    edge_quadrature: str = VALIDATED_EDGE_QUADRATURE,
) -> ActualBodyWakeVelocityLedger:
    """Evaluate the four-channel velocity ledger without changing state."""
    if not isinstance(solution, ActualBoundaryBodyWakeSolution):
        raise ActualBodyWakeVelocityError(
            "solution must be ActualBoundaryBodyWakeSolution"
        )
    if not isinstance(external_incident, ExternalIncidentField):
        raise ActualBodyWakeVelocityError(
            "external_incident must be ExternalIncidentField"
        )
    if (
        not np.isfinite(query_tolerance)
        or query_tolerance < 0.0
        or not np.isfinite(absolute_tolerance)
        or absolute_tolerance < 0.0
        or not np.isfinite(relative_tolerance)
        or relative_tolerance < 0.0
    ):
        raise ActualBodyWakeVelocityError(
            "velocity ledger tolerances must be finite and non-negative"
        )
    reserved = (
        "actual_body_source",
        "actual_body_doublet",
        "current_material_wake",
    )
    overlap = sorted(
        set(external_incident.included_source_ids) & set(reserved)
    )
    if overlap:
        raise ActualBodyWakeVelocityError(
            f"external incident duplicates reserved sources: {overlap}"
        )
    assembly = material_wake_assembly(solution.wake_history)
    reconstruction_error = _validate_query(assembly, query)
    if reconstruction_error > query_tolerance:
        raise ActualBodyWakeVelocityError(
            "wake query points disagree with explicit owner identity"
        )

    external = external_incident.evaluate(query.points)
    source = np.zeros_like(query.points)
    for face_index, face in enumerate(solution.mesh.faces):
        source += constant_source_polygon_influence(
            solution.mesh.vertices[face],
            query.points,
            strength=float(solution.source_strength[face_index]),
            on_surface_side="principal",
        ).velocity

    body_surface = QuadraticDoubletSurface(
        solution.mesh.vertices,
        solution.mesh.faces,
        solution.body_face_potential,
    )
    # N3.1j4d freezes the analytic-radial/edge-quadrature operator as the
    # actual P2 off-sheet field.  The tensor-area path is only an independent
    # smooth-field oracle and is not the runtime body-field identity.
    body_doublet, body_report = (
        body_surface.induced_velocity_line_reduced_converged(
            query.points,
            orders=body_doublet_orders,
            absolute_tolerance=absolute_tolerance,
            relative_tolerance=relative_tolerance,
            edge_quadrature=edge_quadrature,
        )
    )
    wake, wake_report = (
        assembly.induced_velocity_sheet_average_converged(
            query.patch_indices,
            query.face_indices,
            query.barycentric,
            orders=wake_sheet_average_orders,
            absolute_tolerance=absolute_tolerance,
            relative_tolerance=relative_tolerance,
            edge_quadrature=edge_quadrature,
        )
    )
    explicit = _explicit_wake_sum(
        assembly,
        query,
        quadrature_order=wake_report.quadrature_order,
        edge_quadrature=edge_quadrature,
    )
    representation_error = float(
        np.max(np.abs(explicit - wake), initial=0.0)
    )
    total = external + source + body_doublet + wake
    return ActualBodyWakeVelocityLedger(
        query=query,
        external_incident=external,
        body_source=source,
        body_doublet=body_doublet,
        wake_sheet_average=wake,
        total=total,
        body_doublet_report=body_report,
        wake_sheet_average_report=wake_report,
        query_reconstruction_error=reconstruction_error,
        wake_representation_error=representation_error,
        channel_source_ids=(
            *external_incident.included_source_ids,
            *reserved,
        ),
    )
