"""Evidence-constrained spanwise strength reconstruction for newborn sheets.

The N1/LESP interface supplies one circulation-like value per spanwise strip,
whereas a continuous P2 material row has vertex and midside values.  A
parameter-free half-wing reconstruction is obtained by combining:

* exact interpolation of every strip-centre value at the P2 midside;
* continuity of the spanwise derivative at every internal strip boundary;
* zero derivative at the mirror-symmetry root;
* zero potential jump at the free tip.

These are algebraic boundary/compatibility conditions, not a smoothing
functional.  Three temporal rows remain mandatory and are never inferred.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .distributed_doublet import (
    DistributedDoubletError,
    MaterialWakeBand,
    newborn_material_wake_band,
)


@dataclass(frozen=True)
class SpanTraceReport:
    strip_count: int
    degrees_of_freedom: int
    rank: int
    condition_number: float
    max_midpoint_residual: float
    root_derivative_residual: float
    tip_value_residual: float
    max_internal_derivative_jump: float
    passed: bool


@dataclass(frozen=True)
class ReconstructedSpanTrace:
    p2_values: np.ndarray
    span_p2_coordinates: np.ndarray
    report: SpanTraceReport

    def evaluate(self, coordinates) -> np.ndarray:
        """Evaluate the piecewise P2 trace at physical span coordinates."""
        query = np.asarray(coordinates, dtype=float)
        if query.ndim != 1 or not np.all(np.isfinite(query)):
            raise DistributedDoubletError(
                "coordinates must be a finite one-dimensional array"
            )
        edges = self.span_p2_coordinates[0::2]
        tolerance = (
            64.0
            * np.finfo(float).eps
            * max(abs(float(edges[-1] - edges[0])), 1.0)
        )
        if np.any(query < edges[0] - tolerance) or np.any(
            query > edges[-1] + tolerance
        ):
            raise DistributedDoubletError(
                "trace evaluation coordinate is outside the span"
            )
        clipped = np.clip(query, edges[0], edges[-1])
        strip = np.searchsorted(edges, clipped, side="right") - 1
        strip = np.clip(strip, 0, len(edges) - 2)
        width = edges[strip + 1] - edges[strip]
        local = (clipped - edges[strip]) / width
        basis = np.column_stack(
            (
                2.0 * (local - 0.5) * (local - 1.0),
                4.0 * local * (1.0 - local),
                2.0 * local * (local - 0.5),
            )
        )
        indices = np.column_stack(
            (2 * strip, 2 * strip + 1, 2 * strip + 2)
        )
        return np.einsum(
            "ij,ij->i",
            basis,
            self.p2_values[indices],
        )


@dataclass(frozen=True)
class NewbornSheddingBand:
    band: MaterialWakeBand
    trace_reports: tuple[SpanTraceReport, SpanTraceReport, SpanTraceReport]


def _finite_vector(name: str, value) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim != 1 or not np.all(np.isfinite(array)):
        raise DistributedDoubletError(
            f"{name} must be a finite one-dimensional array"
        )
    return array


def halfwing_p2_trace_matrix(
    span_edges,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the square C1 reconstruction matrix and P2 coordinates."""
    edges = _finite_vector("span_edges", span_edges)
    if len(edges) < 2 or np.any(np.diff(edges) <= 0.0):
        raise DistributedDoubletError(
            "span_edges must contain at least two strictly increasing values"
        )
    strip_count = len(edges) - 1
    coordinates = np.empty(2 * strip_count + 1)
    coordinates[0::2] = edges
    coordinates[1::2] = 0.5 * (edges[:-1] + edges[1:])
    matrix = np.zeros((2 * strip_count + 1, 2 * strip_count + 1))
    row = 0

    # One observable from N1/LESP at each strip centre.
    for strip in range(strip_count):
        matrix[row, 2 * strip + 1] = 1.0
        row += 1

    # Mirror-root derivative: (-3 q0 + 4 qm - q1) / h = 0.
    first_width = edges[1] - edges[0]
    matrix[row, 0:3] = np.array([-3.0, 4.0, -1.0]) / first_width
    row += 1

    # Free-tip potential jump.
    matrix[row, -1] = 1.0
    row += 1

    # C1 derivative compatibility at each internal strip boundary.
    for strip in range(strip_count - 1):
        left_width = edges[strip + 1] - edges[strip]
        right_width = edges[strip + 2] - edges[strip + 1]
        left = np.array([1.0, -4.0, 3.0]) / left_width
        right = np.array([-3.0, 4.0, -1.0]) / right_width
        matrix[row, 2 * strip : 2 * strip + 3] += left
        matrix[row, 2 * strip + 2 : 2 * strip + 5] -= right
        row += 1
    if row != len(matrix):
        raise DistributedDoubletError(
            "internal error while assembling the span-trace system"
        )
    return matrix, coordinates


def reconstruct_halfwing_p2_trace(
    strip_midpoint_values,
    span_edges,
    *,
    residual_tolerance: float = 1.0e-11,
) -> ReconstructedSpanTrace:
    """Lift strip-centre values into a unique half-wing C1 P2 trace."""
    midpoint = _finite_vector(
        "strip_midpoint_values",
        strip_midpoint_values,
    )
    matrix, coordinates = halfwing_p2_trace_matrix(span_edges)
    strip_count = (len(matrix) - 1) // 2
    if midpoint.shape != (strip_count,):
        raise DistributedDoubletError(
            "strip_midpoint_values count must match span_edges"
        )
    if residual_tolerance < 0.0 or not np.isfinite(residual_tolerance):
        raise DistributedDoubletError(
            "residual_tolerance must be finite and non-negative"
        )
    rhs = np.zeros(len(matrix))
    rhs[:strip_count] = midpoint
    singular_values = np.linalg.svd(
        matrix,
        compute_uv=False,
        full_matrices=False,
    )
    rank = int(np.linalg.matrix_rank(matrix))
    if rank != len(matrix):
        raise DistributedDoubletError(
            "half-wing P2 shedding trace is rank deficient: "
            f"rank={rank}, dofs={len(matrix)}"
        )
    values = np.linalg.solve(matrix, rhs)
    residual = matrix @ values - rhs
    derivative_jump = residual[strip_count + 2 :]
    report = SpanTraceReport(
        strip_count=strip_count,
        degrees_of_freedom=len(matrix),
        rank=rank,
        condition_number=float(singular_values[0] / singular_values[-1]),
        max_midpoint_residual=float(
            np.max(np.abs(residual[:strip_count]), initial=0.0)
        ),
        root_derivative_residual=float(abs(residual[strip_count])),
        tip_value_residual=float(abs(residual[strip_count + 1])),
        max_internal_derivative_jump=float(
            np.max(np.abs(derivative_jump), initial=0.0)
        ),
        passed=bool(
            np.max(np.abs(residual), initial=0.0)
            <= residual_tolerance
        ),
    )
    return ReconstructedSpanTrace(
        p2_values=values,
        span_p2_coordinates=coordinates,
        report=report,
    )


def newborn_halfwing_shedding_band(
    *,
    sheet_id: str,
    vortex_family: str,
    previous_edge,
    current_edge,
    span_edges,
    time_nodes,
    strip_strength_rows,
    residual_tolerance: float = 1.0e-11,
) -> NewbornSheddingBand:
    """Create a material band from three explicitly solved strip rows."""
    rows = np.asarray(strip_strength_rows, dtype=float)
    edge_coordinates = _finite_vector("span_edges", span_edges)
    strip_count = len(edge_coordinates) - 1
    if (
        rows.shape != (3, strip_count)
        or not np.all(np.isfinite(rows))
    ):
        raise DistributedDoubletError(
            "strip_strength_rows must have shape "
            f"{(3, strip_count)} and contain finite values"
        )
    reconstructions = tuple(
        reconstruct_halfwing_p2_trace(
            row,
            edge_coordinates,
            residual_tolerance=residual_tolerance,
        )
        for row in rows
    )
    band = newborn_material_wake_band(
        sheet_id=sheet_id,
        vortex_family=vortex_family,
        previous_edge=previous_edge,
        current_edge=current_edge,
        time_nodes=time_nodes,
        potential_jump_rows=np.stack(
            [item.p2_values for item in reconstructions]
        ),
    )
    return NewbornSheddingBand(
        band=band,
        trace_reports=tuple(
            item.report for item in reconstructions
        ),
    )
