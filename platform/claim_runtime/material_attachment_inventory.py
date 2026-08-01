"""Read-only material attachment/orientation inventory for S3ai-v2.

The physical traces in this module are extracted only from the face-local
P2 coefficients stored in ``MaterialWakeBand.surface.face_mu``.  The cached
``potential_jump_rows`` are inspected by a separate consistency diagnostic;
they never define a boundary trace, a material release, or an inventory.

This is a representation guard, not an independent Kelvin equation.  It
does not solve a body equation, choose circulation, evaluate pressure, or
contribute force.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from .actual_boundary_body_wake import MaterialWakeCutAttachment
from .classified_p2_cut_topology import ClassifiedP2CutTopology
from .distributed_doublet import (
    MaterialWakeBand,
    MaterialWakeHistory,
    newborn_material_wake_band,
)


class MaterialAttachmentInventoryError(ValueError):
    """A surface material trace or typed attachment is not well defined."""


_LOCAL_EDGE_MIDPOINT = {
    frozenset((0, 1)): 3,
    frozenset((1, 2)): 4,
    frozenset((0, 2)): 5,
}


def _finite_vector(name: str, value: Any, size: int | None = None) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if (
        array.ndim != 1
        or array.size < 1
        or (size is not None and array.shape != (size,))
        or not np.all(np.isfinite(array))
    ):
        expected = "(n,)" if size is None else str((size,))
        raise MaterialAttachmentInventoryError(
            f"{name} must be finite with shape {expected}, got {array.shape}"
        )
    return array.copy()


def _nonnegative_tolerance(name: str, value: Any) -> float:
    tolerance = float(value)
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise MaterialAttachmentInventoryError(
            f"{name} must be finite and non-negative"
        )
    return tolerance


@dataclass(frozen=True)
class MaterialBandSurfaceObservation:
    """Face-derived previous/current traces for one material wake band."""

    sheet_id: str
    previous_trace: np.ndarray
    current_trace: np.ndarray
    release: np.ndarray
    boundary_duplicate_abs_error: float
    surface_internal_trace_abs_error: float
    row_surface_cache_abs_error: float

    def __post_init__(self) -> None:
        previous = _finite_vector("previous_trace", self.previous_trace)
        current = _finite_vector(
            "current_trace", self.current_trace, len(previous)
        )
        release = _finite_vector("release", self.release, len(previous))
        if not np.array_equal(release, current - previous):
            raise MaterialAttachmentInventoryError(
                "release must be exactly current_trace-previous_trace"
            )
        scalars = (
            self.boundary_duplicate_abs_error,
            self.surface_internal_trace_abs_error,
            self.row_surface_cache_abs_error,
        )
        if any(not np.isfinite(value) or value < 0.0 for value in scalars):
            raise MaterialAttachmentInventoryError(
                "surface diagnostic errors must be finite and non-negative"
            )
        object.__setattr__(self, "previous_trace", previous)
        object.__setattr__(self, "current_trace", current)
        object.__setattr__(self, "release", release)


@dataclass(frozen=True)
class MaterialHistorySurfaceObservation:
    """Surface-derived release and seam diagnostics for one history."""

    bands: tuple[MaterialBandSurfaceObservation, ...]
    material_release: np.ndarray
    seam_residuals: tuple[np.ndarray, ...]
    maximum_surface_internal_trace_error: float
    maximum_history_seam_error: float
    maximum_row_surface_cache_error: float
    maximum_time_gap: float
    maximum_geometry_gap: float

    def __post_init__(self) -> None:
        bands = tuple(self.bands)
        if not bands:
            raise MaterialAttachmentInventoryError(
                "a material history observation needs at least one band"
            )
        size = len(bands[0].release)
        if any(len(band.release) != size for band in bands):
            raise MaterialAttachmentInventoryError(
                "all observed bands must use the same P2 trace size"
            )
        release = _finite_vector(
            "material_release", self.material_release, size
        )
        expected = np.sum(
            np.stack([band.release for band in bands], axis=0),
            axis=0,
        )
        if not np.array_equal(release, expected):
            raise MaterialAttachmentInventoryError(
                "material_release must be the exact sum of band releases"
            )
        seams = tuple(
            _finite_vector("seam_residual", residual, size)
            for residual in self.seam_residuals
        )
        if len(seams) != max(len(bands) - 1, 0):
            raise MaterialAttachmentInventoryError(
                "seam residual count does not match the band history"
            )
        diagnostics = (
            self.maximum_surface_internal_trace_error,
            self.maximum_history_seam_error,
            self.maximum_row_surface_cache_error,
            self.maximum_time_gap,
            self.maximum_geometry_gap,
        )
        if any(not np.isfinite(value) or value < 0.0 for value in diagnostics):
            raise MaterialAttachmentInventoryError(
                "history diagnostic errors must be finite and non-negative"
            )
        object.__setattr__(self, "bands", bands)
        object.__setattr__(self, "material_release", release)
        object.__setattr__(self, "seam_residuals", seams)


@dataclass(frozen=True)
class MaterialAttachmentInventory:
    """One complete-trace representation inventory ``I_beta``."""

    body_cut_trace: np.ndarray
    material_release: np.ndarray
    canonical_release: np.ndarray
    inventory: np.ndarray
    attachment_permutation: np.ndarray
    attachment_sign: int
    birth_sign: int
    history: MaterialHistorySurfaceObservation

    def __post_init__(self) -> None:
        body = _finite_vector("body_cut_trace", self.body_cut_trace)
        material = _finite_vector(
            "material_release", self.material_release, len(body)
        )
        canonical = _finite_vector(
            "canonical_release", self.canonical_release, len(body)
        )
        inventory = _finite_vector("inventory", self.inventory, len(body))
        permutation = np.asarray(
            self.attachment_permutation, dtype=np.int64
        )
        if (
            permutation.shape != (len(body),)
            or not np.array_equal(
                np.sort(permutation),
                np.arange(len(body), dtype=np.int64),
            )
        ):
            raise MaterialAttachmentInventoryError(
                "attachment_permutation must be a complete P2 permutation"
            )
        if self.attachment_sign not in (-1, 1):
            raise MaterialAttachmentInventoryError(
                "attachment_sign must be exactly -1 or +1"
            )
        if self.birth_sign not in (-1, 1):
            raise MaterialAttachmentInventoryError(
                "birth_sign must be exactly -1 or +1"
            )
        expected_canonical = np.empty_like(material)
        expected_canonical[permutation] = (
            self.attachment_sign * material
        )
        if not np.array_equal(canonical, expected_canonical):
            raise MaterialAttachmentInventoryError(
                "canonical_release does not equal s*P^T*material_release"
            )
        if not np.array_equal(
            inventory,
            -body + self.birth_sign * canonical,
        ):
            raise MaterialAttachmentInventoryError(
                "inventory does not equal -c+beta*R_w_body"
            )
        object.__setattr__(self, "body_cut_trace", body)
        object.__setattr__(self, "material_release", material)
        object.__setattr__(self, "canonical_release", canonical)
        object.__setattr__(self, "inventory", inventory)
        object.__setattr__(
            self, "attachment_permutation", permutation.copy()
        )


def _chain_edge_trace(
    band: MaterialWakeBand,
    *,
    boundary: Literal["previous", "current"],
) -> tuple[np.ndarray, float]:
    """Extract one ordered P2 boundary chain from ``surface.face_mu``."""

    if not isinstance(band, MaterialWakeBand):
        raise MaterialAttachmentInventoryError(
            "band must be a MaterialWakeBand"
        )
    if boundary == "previous":
        chain = np.arange(band.span_nodes, dtype=np.int64)
    elif boundary == "current":
        chain = np.arange(
            band.span_nodes,
            2 * band.span_nodes,
            dtype=np.int64,
        )
    else:
        raise MaterialAttachmentInventoryError(
            "boundary must be exactly 'previous' or 'current'"
        )

    edge_traces: list[np.ndarray] = []
    for first, second in zip(chain[:-1], chain[1:]):
        records: list[np.ndarray] = []
        for face_index, face in enumerate(band.surface.faces):
            first_local = np.flatnonzero(face == first)
            second_local = np.flatnonzero(face == second)
            if len(first_local) != 1 or len(second_local) != 1:
                continue
            local_first = int(first_local[0])
            local_second = int(second_local[0])
            midpoint = _LOCAL_EDGE_MIDPOINT.get(
                frozenset((local_first, local_second))
            )
            if midpoint is None:
                raise MaterialAttachmentInventoryError(
                    "adjacent chain vertices are not one triangle edge"
                )
            coefficients = band.surface.face_mu[face_index]
            records.append(
                np.array(
                    (
                        coefficients[local_first],
                        coefficients[midpoint],
                        coefficients[local_second],
                    ),
                    dtype=float,
                )
            )
        if len(records) != 1:
            raise MaterialAttachmentInventoryError(
                f"{boundary} material chain edge "
                f"({int(first)},{int(second)}) must have exactly one "
                f"incident boundary triangle, got {len(records)}"
            )
        edge_traces.append(records[0])

    duplicate_error = 0.0
    for left, right in zip(edge_traces[:-1], edge_traces[1:]):
        duplicate_error = max(
            duplicate_error,
            abs(float(left[2] - right[0])),
        )
    trace = np.empty(2 * band.span_nodes - 1, dtype=float)
    for index, edge_trace in enumerate(edge_traces):
        trace[2 * index] = edge_trace[0]
        trace[2 * index + 1] = edge_trace[1]
    trace[-1] = edge_traces[-1][2]
    if not np.all(np.isfinite(trace)):
        raise MaterialAttachmentInventoryError(
            f"{boundary} surface trace contains non-finite values"
        )
    return trace, duplicate_error


def extract_surface_boundary_trace(
    band: MaterialWakeBand,
    boundary: Literal["previous", "current"],
    *,
    duplicate_tolerance: float = 2.0e-12,
) -> np.ndarray:
    """Return an ordered boundary trace derived only from ``face_mu``.

    Every adjacent material-vertex pair must be a unique topological
    boundary edge.  The value at a shared chain vertex is read independently
    from its two incident boundary faces and must agree within the declared
    tolerance.
    """

    tolerance = _nonnegative_tolerance(
        "duplicate_tolerance", duplicate_tolerance
    )
    trace, duplicate_error = _chain_edge_trace(
        band, boundary=boundary
    )
    if duplicate_error > tolerance:
        raise MaterialAttachmentInventoryError(
            f"{boundary} boundary duplicate mismatch "
            f"{duplicate_error:.6e} exceeds {tolerance:.6e}"
        )
    return trace


def surface_row_cache_abs_error(band: MaterialWakeBand) -> float:
    """Compare ``face_mu`` with the row cache without defining a trace.

    This diagnostic is deliberately separate from
    :func:`extract_surface_boundary_trace`.  Altering the row cache changes
    this error but cannot change the surface-derived material functional.
    """

    if not isinstance(band, MaterialWakeBand):
        raise MaterialAttachmentInventoryError(
            "band must be a MaterialWakeBand"
        )
    previous = band.surface.vertices[: band.span_nodes]
    current = band.surface.vertices[band.span_nodes :]
    matches: list[float] = []
    for pattern in ("forward", "mirror_symmetric"):
        try:
            candidate = newborn_material_wake_band(
                sheet_id=f"{band.sheet_id}-row-cache-audit",
                vortex_family=band.vortex_family,
                previous_edge=previous,
                current_edge=current,
                time_nodes=band.time_nodes,
                potential_jump_rows=band.potential_jump_rows,
                span_diagonal_pattern=pattern,
            )
        except ValueError:
            continue
        if (
            np.array_equal(candidate.surface.vertices, band.surface.vertices)
            and np.array_equal(candidate.surface.faces, band.surface.faces)
        ):
            matches.append(
                float(
                    np.max(
                        np.abs(
                            candidate.surface.face_mu
                            - band.surface.face_mu
                        ),
                        initial=0.0,
                    )
                )
            )
    if len(matches) != 1:
        raise MaterialAttachmentInventoryError(
            "material band does not have one uniquely supported P2 "
            "row-cache topology"
        )
    return matches[0]


def observe_material_band_surface(
    band: MaterialWakeBand,
    *,
    duplicate_tolerance: float = 2.0e-12,
) -> MaterialBandSurfaceObservation:
    """Observe one band's two material boundaries without using row cache."""

    tolerance = _nonnegative_tolerance(
        "duplicate_tolerance", duplicate_tolerance
    )
    previous, previous_duplicate = _chain_edge_trace(
        band, boundary="previous"
    )
    current, current_duplicate = _chain_edge_trace(
        band, boundary="current"
    )
    duplicate = max(previous_duplicate, current_duplicate)
    if duplicate > tolerance:
        raise MaterialAttachmentInventoryError(
            "surface boundary duplicate mismatch "
            f"{duplicate:.6e} exceeds {tolerance:.6e}"
        )
    continuity = band.surface.continuity_report(tolerance=tolerance)
    internal_error = max(
        float(continuity.max_trace_node_jump),
        float(continuity.max_trace_jump),
    )
    return MaterialBandSurfaceObservation(
        sheet_id=band.sheet_id,
        previous_trace=previous,
        current_trace=current,
        release=current - previous,
        boundary_duplicate_abs_error=duplicate,
        surface_internal_trace_abs_error=internal_error,
        row_surface_cache_abs_error=surface_row_cache_abs_error(band),
    )


def observe_material_history_surface(
    history: MaterialWakeHistory,
    *,
    duplicate_tolerance: float = 2.0e-12,
) -> MaterialHistorySurfaceObservation:
    """Sum face-derived band releases and audit face-derived history seams."""

    if not isinstance(history, MaterialWakeHistory):
        raise MaterialAttachmentInventoryError(
            "history must be a MaterialWakeHistory"
        )
    tolerance = _nonnegative_tolerance(
        "duplicate_tolerance", duplicate_tolerance
    )
    bands = tuple(
        observe_material_band_surface(
            band, duplicate_tolerance=tolerance
        )
        for band in history.bands
    )
    span_nodes = history.bands[0].span_nodes
    if any(band.span_nodes != span_nodes for band in history.bands):
        raise MaterialAttachmentInventoryError(
            "all material history bands must share span_nodes"
        )
    if any(
        band.vortex_family != history.bands[0].vortex_family
        for band in history.bands
    ):
        raise MaterialAttachmentInventoryError(
            "all material history bands must share vortex_family"
        )

    seam_residuals = tuple(
        newer.previous_trace - older.current_trace
        for older, newer in zip(bands, bands[1:])
    )
    maximum_seam = max(
        (
            float(np.max(np.abs(residual), initial=0.0))
            for residual in seam_residuals
        ),
        default=0.0,
    )
    maximum_time_gap = 0.0
    maximum_geometry_gap = 0.0
    for older, newer in zip(history.bands, history.bands[1:]):
        maximum_time_gap = max(
            maximum_time_gap,
            abs(float(older.time_nodes[2] - newer.time_nodes[0])),
        )
        maximum_geometry_gap = max(
            maximum_geometry_gap,
            float(
                np.max(
                    np.linalg.norm(
                        older.surface.vertices[older.span_nodes :]
                        - newer.surface.vertices[: newer.span_nodes],
                        axis=1,
                    ),
                    initial=0.0,
                )
            ),
        )
    return MaterialHistorySurfaceObservation(
        bands=bands,
        material_release=np.sum(
            np.stack([band.release for band in bands], axis=0),
            axis=0,
        ),
        seam_residuals=seam_residuals,
        maximum_surface_internal_trace_error=max(
            band.surface_internal_trace_abs_error for band in bands
        ),
        maximum_history_seam_error=maximum_seam,
        maximum_row_surface_cache_error=max(
            band.row_surface_cache_abs_error for band in bands
        ),
        maximum_time_gap=maximum_time_gap,
        maximum_geometry_gap=maximum_geometry_gap,
    )


def canonicalize_material_release(
    material_release: Any,
    topology: ClassifiedP2CutTopology,
    attachment: MaterialWakeCutAttachment,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the unique typed inverse ``s*P^T`` to a material trace."""

    if not isinstance(topology, ClassifiedP2CutTopology):
        raise MaterialAttachmentInventoryError(
            "topology must be a ClassifiedP2CutTopology"
        )
    if not isinstance(attachment, MaterialWakeCutAttachment):
        raise MaterialAttachmentInventoryError(
            "attachment must be an explicit MaterialWakeCutAttachment"
        )
    size = len(topology.cut_node_coordinates)
    material = _finite_vector(
        "material_release", material_release, size
    )
    try:
        permutation = attachment.p2_trace_permutation(topology)
    except ValueError as exc:
        raise MaterialAttachmentInventoryError(
            "typed attachment is incompatible with the body cut"
        ) from exc
    canonical = np.empty_like(material)
    canonical[permutation] = (
        attachment.wake_jump_from_body_cut_sign * material
    )
    return canonical, permutation.copy()


def observe_material_attachment_inventory(
    topology: ClassifiedP2CutTopology,
    history: MaterialWakeHistory,
    *,
    global_body_potential: Any,
    attachment: MaterialWakeCutAttachment,
    birth_sign: int = 1,
    duplicate_tolerance: float = 2.0e-12,
) -> MaterialAttachmentInventory:
    """Observe ``I_beta=-c+beta*s*P^T*R_w_material`` read-only."""

    if not isinstance(topology, ClassifiedP2CutTopology):
        raise MaterialAttachmentInventoryError(
            "topology must be a ClassifiedP2CutTopology"
        )
    if not isinstance(attachment, MaterialWakeCutAttachment):
        raise MaterialAttachmentInventoryError(
            "attachment must be an explicit MaterialWakeCutAttachment"
        )
    if (
        not isinstance(birth_sign, (int, np.integer))
        or int(birth_sign) not in (-1, 1)
    ):
        raise MaterialAttachmentInventoryError(
            "birth_sign must be exactly -1 or +1"
        )
    surface_history = observe_material_history_surface(
        history, duplicate_tolerance=duplicate_tolerance
    )
    size = len(topology.cut_node_coordinates)
    if len(surface_history.material_release) != size:
        raise MaterialAttachmentInventoryError(
            "material history trace size does not match the body cut"
        )
    body_potential = _finite_vector(
        "global_body_potential",
        global_body_potential,
        topology.dof_count,
    )
    body_trace = topology.cut_jump(body_potential)
    canonical, permutation = canonicalize_material_release(
        surface_history.material_release,
        topology,
        attachment,
    )
    beta = int(birth_sign)
    return MaterialAttachmentInventory(
        body_cut_trace=body_trace,
        material_release=surface_history.material_release,
        canonical_release=canonical,
        inventory=-body_trace + beta * canonical,
        attachment_permutation=permutation,
        attachment_sign=attachment.wake_jump_from_body_cut_sign,
        birth_sign=beta,
        history=surface_history,
    )


def material_inventory_increment(
    previous: MaterialAttachmentInventory,
    current: MaterialAttachmentInventory,
) -> np.ndarray:
    """Return one complete-trace inventory increment."""

    if not isinstance(previous, MaterialAttachmentInventory) or not isinstance(
        current, MaterialAttachmentInventory
    ):
        raise MaterialAttachmentInventoryError(
            "previous and current must be MaterialAttachmentInventory"
        )
    if (
        previous.inventory.shape != current.inventory.shape
        or previous.birth_sign != current.birth_sign
        or previous.attachment_sign != current.attachment_sign
        or not np.array_equal(
            previous.attachment_permutation,
            current.attachment_permutation,
        )
    ):
        raise MaterialAttachmentInventoryError(
            "inventory states do not share one typed trace convention"
        )
    return current.inventory - previous.inventory
