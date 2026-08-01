"""Typed open-boundary topology for an actual chronological material wake.

The existing :class:`QuadraticDoubletAssembly` correctly validates closed
patch assemblies whose interfaces have exactly two sides.  An actual wake
also has a one-sided newest edge attached to an external body cut.  This
module leaves the closed-assembly implementation unchanged and builds the
separate combinatorial topology required by that actual-wake role.

No coordinate matching is used.  Consecutive material bands define shared
P1 rows by chronological identity.  Continuous-P2 vertex and edge DOFs are
then in an exact permutation with the stored chronological potential-jump
rows.  The module contains no velocity, time integration, pressure or force.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .distributed_doublet import (
    DistributedDoubletError,
    MaterialWakeHistory,
    newborn_material_wake_band,
)
from .p2_surface_material_transport import (
    ContinuousP2ScalarTopology,
    continuous_p2_scalar_topology,
)


class ActualWakeStageTopologyError(DistributedDoubletError):
    """Invalid chronological identity or scalar-state representation."""


@dataclass(frozen=True)
class ActualWakeStageBoundaryRoles:
    """Disjoint typed entities on the one-sided actual-wake boundary."""

    body_attachment_p1_dofs: np.ndarray
    oldest_p1_dofs: np.ndarray
    body_attachment_p2_dofs: np.ndarray
    oldest_p2_dofs: np.ndarray
    root_characteristic_interior_p2_dofs: np.ndarray
    tip_characteristic_interior_p2_dofs: np.ndarray
    body_attachment_id: str

    def __post_init__(self) -> None:
        names = (
            "body_attachment_p1_dofs",
            "oldest_p1_dofs",
            "body_attachment_p2_dofs",
            "oldest_p2_dofs",
            "root_characteristic_interior_p2_dofs",
            "tip_characteristic_interior_p2_dofs",
        )
        for name in names:
            value = np.asarray(getattr(self, name), dtype=np.int64)
            if (
                value.ndim != 1
                or len(value) == 0
                or len(np.unique(value)) != len(value)
                or np.any(value < 0)
            ):
                raise ActualWakeStageTopologyError(
                    f"{name} must be a nonempty unique non-negative "
                    "integer array"
                )
            object.__setattr__(self, name, value.copy())
        if (
            not isinstance(self.body_attachment_id, str)
            or not self.body_attachment_id
        ):
            raise ActualWakeStageTopologyError(
                "body_attachment_id must be nonempty"
            )
        p2_roles = (
            self.body_attachment_p2_dofs,
            self.oldest_p2_dofs,
            self.root_characteristic_interior_p2_dofs,
            self.tip_characteristic_interior_p2_dofs,
        )
        joined = np.concatenate(p2_roles)
        if len(np.unique(joined)) != len(joined):
            raise ActualWakeStageTopologyError(
                "actual-wake P2 boundary roles must be disjoint"
            )

    @property
    def all_boundary_p2_dofs(self) -> np.ndarray:
        return np.concatenate(
            (
                self.body_attachment_p2_dofs,
                self.oldest_p2_dofs,
                self.root_characteristic_interior_p2_dofs,
                self.tip_characteristic_interior_p2_dofs,
            )
        )


@dataclass(frozen=True)
class ActualWakeStageTopology:
    """Combinatorial P1/P2 topology and chronological scalar bijection."""

    history_id: str
    band_count: int
    span_nodes: int
    cut_node_count: int
    p1_vertices: np.ndarray
    p1_faces: np.ndarray
    band_p1_vertex_dofs: tuple[np.ndarray, ...]
    p2_topology: ContinuousP2ScalarTopology
    band_p2_face_dofs: tuple[np.ndarray, ...]
    p2_dof_to_chronological: np.ndarray
    chronological_to_p2_dof: np.ndarray
    span_diagonal_patterns: tuple[str, ...]
    boundary_roles: ActualWakeStageBoundaryRoles

    @property
    def chronological_row_count(self) -> int:
        return 2 * self.band_count + 1

    @property
    def chronological_scalar_count(self) -> int:
        return self.chronological_row_count * self.cut_node_count

    def _validate_history(
        self,
        history: MaterialWakeHistory,
        *,
        tolerance: float,
    ) -> None:
        if not isinstance(history, MaterialWakeHistory):
            raise ActualWakeStageTopologyError(
                "history must be MaterialWakeHistory"
            )
        if tolerance < 0.0 or not np.isfinite(tolerance):
            raise ActualWakeStageTopologyError(
                "tolerance must be finite and non-negative"
            )
        if (
            history.history_id != self.history_id
            or len(history.bands) != self.band_count
            or history.span_nodes != self.span_nodes
        ):
            raise ActualWakeStageTopologyError(
                "history identity/count does not match stage topology"
            )
        report = history.continuity_report(
            geometry_tolerance=tolerance,
            strength_tolerance=tolerance,
        )
        if not report.compatible:
            raise ActualWakeStageTopologyError(
                f"material history is incompatible: {report}"
            )
        for band, expected_pattern in zip(
            history.bands,
            self.span_diagonal_patterns,
            strict=True,
        ):
            pattern = _band_diagonal_pattern(band)
            if pattern != expected_pattern:
                raise ActualWakeStageTopologyError(
                    "material band face pattern changed"
                )

    def chronological_rows(
        self,
        history: MaterialWakeHistory,
        *,
        tolerance: float = 1.0e-12,
    ) -> np.ndarray:
        """Return the unique ``(2B+1, 2n-1)`` material-row state."""
        self._validate_history(history, tolerance=tolerance)
        rows = np.empty(
            (self.chronological_row_count, self.cut_node_count),
            dtype=float,
        )
        assigned = np.zeros(self.chronological_row_count, dtype=bool)
        for band_index, band in enumerate(history.bands):
            for local_row in range(3):
                row = 2 * band_index + local_row
                value = band.potential_jump_rows[local_row]
                if assigned[row]:
                    if (
                        np.max(
                            np.abs(rows[row] - value),
                            initial=0.0,
                        )
                        > tolerance
                    ):
                        raise ActualWakeStageTopologyError(
                            "chronological scalar seam is inconsistent"
                        )
                else:
                    rows[row] = value
                    assigned[row] = True
        if not np.all(assigned):
            raise ActualWakeStageTopologyError(
                "chronological scalar rows are incomplete"
            )
        return rows

    def global_p2_state(
        self,
        history: MaterialWakeHistory,
        *,
        tolerance: float = 1.0e-12,
    ) -> np.ndarray:
        """Extract one continuous-P2 vector and verify every face value."""
        rows = self.chronological_rows(history, tolerance=tolerance)
        state = rows.ravel()[self.p2_dof_to_chronological]
        for band, face_dofs in zip(
            history.bands,
            self.band_p2_face_dofs,
            strict=True,
        ):
            error = float(
                np.max(
                    np.abs(state[face_dofs] - band.surface.face_mu),
                    initial=0.0,
                )
            )
            if error > tolerance:
                raise ActualWakeStageTopologyError(
                    "chronological rows and band face_mu disagree"
                )
        return state

    def rebuild_history(
        self,
        geometry_history: MaterialWakeHistory,
        global_p2_state: Any,
        *,
        tolerance: float = 1.0e-12,
    ) -> MaterialWakeHistory:
        """Rebuild all band rows/face values from one global P2 vector."""
        self._validate_history(geometry_history, tolerance=tolerance)
        state = np.asarray(global_p2_state, dtype=float)
        expected = (self.p2_topology.degree_of_freedom_count,)
        if state.shape != expected or not np.all(np.isfinite(state)):
            raise ActualWakeStageTopologyError(
                f"global_p2_state must have shape {expected}"
            )
        chronological = np.empty(self.chronological_scalar_count)
        chronological[self.p2_dof_to_chronological] = state
        rows = chronological.reshape(
            self.chronological_row_count,
            self.cut_node_count,
        )
        rebuilt = []
        for band_index, (band, pattern) in enumerate(
            zip(
                geometry_history.bands,
                self.span_diagonal_patterns,
                strict=True,
            )
        ):
            rebuilt.append(
                newborn_material_wake_band(
                    sheet_id=band.sheet_id,
                    vortex_family=band.vortex_family,
                    previous_edge=band.surface.vertices[
                        : self.span_nodes
                    ],
                    current_edge=band.surface.vertices[
                        self.span_nodes :
                    ],
                    time_nodes=band.time_nodes,
                    potential_jump_rows=rows[
                        2 * band_index : 2 * band_index + 3
                    ],
                    span_diagonal_pattern=pattern,
                )
            )
        candidate = MaterialWakeHistory(
            geometry_history.history_id,
            tuple(rebuilt),
        )
        report = candidate.continuity_report(
            geometry_tolerance=tolerance,
            strength_tolerance=tolerance,
        )
        if not report.compatible:
            raise ActualWakeStageTopologyError(
                f"rebuilt material history is incompatible: {report}"
            )
        roundtrip = self.global_p2_state(
            candidate,
            tolerance=tolerance,
        )
        if (
            np.max(np.abs(roundtrip - state), initial=0.0)
            > tolerance
        ):
            raise ActualWakeStageTopologyError(
                "rebuilt material P2 state does not round-trip"
            )
        return candidate


def _band_diagonal_pattern(band) -> str:
    matches = []
    zero = np.zeros_like(band.potential_jump_rows)
    previous = band.surface.vertices[: band.span_nodes]
    current = band.surface.vertices[band.span_nodes :]
    for pattern in ("forward", "mirror_symmetric"):
        try:
            candidate = newborn_material_wake_band(
                sheet_id=band.sheet_id,
                vortex_family=band.vortex_family,
                previous_edge=previous,
                current_edge=current,
                time_nodes=band.time_nodes,
                potential_jump_rows=zero,
                span_diagonal_pattern=pattern,
            )
        except DistributedDoubletError:
            continue
        if np.array_equal(candidate.surface.faces, band.surface.faces):
            matches.append(pattern)
    if len(matches) != 1:
        raise ActualWakeStageTopologyError(
            "material band does not have one supported explicit face pattern"
        )
    return matches[0]


def _global_p1_mesh(
    history: MaterialWakeHistory,
) -> tuple[np.ndarray, np.ndarray, tuple[np.ndarray, ...]]:
    band_count = len(history.bands)
    span_nodes = history.span_nodes
    rows = [
        history.bands[0].surface.vertices[:span_nodes].copy(),
        *(
            band.surface.vertices[span_nodes:].copy()
            for band in history.bands
        ),
    ]
    vertices = np.vstack(rows)
    faces = []
    band_vertex_dofs = []
    for band_index, band in enumerate(history.bands):
        previous = np.arange(
            band_index * span_nodes,
            (band_index + 1) * span_nodes,
            dtype=np.int64,
        )
        current = np.arange(
            (band_index + 1) * span_nodes,
            (band_index + 2) * span_nodes,
            dtype=np.int64,
        )
        local_to_global = np.concatenate((previous, current))
        band_vertex_dofs.append(local_to_global)
        faces.append(local_to_global[band.surface.faces])
    expected_vertices = (band_count + 1) * span_nodes
    if len(vertices) != expected_vertices:
        raise ActualWakeStageTopologyError(
            "chronological P1 vertex count is inconsistent"
        )
    return vertices, np.vstack(faces), tuple(band_vertex_dofs)


def _p2_chronological_permutation(
    topology: ContinuousP2ScalarTopology,
    *,
    band_count: int,
    span_nodes: int,
) -> tuple[np.ndarray, np.ndarray]:
    cut_nodes = 2 * span_nodes - 1
    temporal_rows = 2 * band_count + 1
    expected = temporal_rows * cut_nodes
    if topology.degree_of_freedom_count != expected:
        raise ActualWakeStageTopologyError(
            "continuous-P2 count does not match chronological rows"
        )
    p2_to_chronological = np.empty(expected, dtype=np.int64)
    p1_count = len(topology.vertices)
    for vertex in range(p1_count):
        time_index, span_index = divmod(vertex, span_nodes)
        if time_index > band_count:
            raise ActualWakeStageTopologyError(
                "P1 vertex lies outside chronological rows"
            )
        p2_to_chronological[vertex] = (
            2 * time_index * cut_nodes + 2 * span_index
        )
    for edge_index, (first, second) in enumerate(
        topology.edge_vertices
    ):
        first_time, first_span = divmod(int(first), span_nodes)
        second_time, second_span = divmod(int(second), span_nodes)
        time_delta = abs(second_time - first_time)
        span_delta = abs(second_span - first_span)
        if time_delta == 0 and span_delta == 1:
            row = 2 * first_time
            cut = 2 * min(first_span, second_span) + 1
        elif time_delta == 1 and span_delta == 0:
            row = 2 * min(first_time, second_time) + 1
            cut = 2 * first_span
        elif time_delta == 1 and span_delta == 1:
            row = 2 * min(first_time, second_time) + 1
            cut = 2 * min(first_span, second_span) + 1
        else:
            raise ActualWakeStageTopologyError(
                "P1 edge is not a span/time/cell-diagonal entity"
            )
        p2_to_chronological[p1_count + edge_index] = (
            row * cut_nodes + cut
        )
    if not np.array_equal(
        np.sort(p2_to_chronological),
        np.arange(expected, dtype=np.int64),
    ):
        raise ActualWakeStageTopologyError(
            "P2-to-chronological map is not a complete permutation"
        )
    chronological_to_p2 = np.empty(expected, dtype=np.int64)
    chronological_to_p2[p2_to_chronological] = np.arange(
        expected,
        dtype=np.int64,
    )
    return p2_to_chronological, chronological_to_p2


def actual_wake_stage_topology(
    history: MaterialWakeHistory,
    *,
    body_attachment_id: str,
    tolerance: float = 1.0e-12,
) -> ActualWakeStageTopology:
    """Build the typed actual-wake topology from chronological identities."""
    if not isinstance(history, MaterialWakeHistory):
        raise ActualWakeStageTopologyError(
            "history must be MaterialWakeHistory"
        )
    if tolerance < 0.0 or not np.isfinite(tolerance):
        raise ActualWakeStageTopologyError(
            "tolerance must be finite and non-negative"
        )
    report = history.continuity_report(
        geometry_tolerance=tolerance,
        strength_tolerance=tolerance,
    )
    if not report.compatible:
        raise ActualWakeStageTopologyError(
            f"material history is incompatible: {report}"
        )
    if not isinstance(body_attachment_id, str) or not body_attachment_id:
        raise ActualWakeStageTopologyError(
            "body_attachment_id must be nonempty"
        )
    band_count = len(history.bands)
    span_nodes = history.span_nodes
    cut_nodes = 2 * span_nodes - 1
    patterns = tuple(_band_diagonal_pattern(band) for band in history.bands)
    vertices, faces, band_p1_dofs = _global_p1_mesh(history)
    p2 = continuous_p2_scalar_topology(vertices, faces)
    p2_to_chronological, chronological_to_p2 = (
        _p2_chronological_permutation(
            p2,
            band_count=band_count,
            span_nodes=span_nodes,
        )
    )
    band_face_dofs = []
    offset = 0
    for band in history.bands:
        count = len(band.surface.faces)
        band_face_dofs.append(
            p2.local_to_global[offset : offset + count].copy()
        )
        offset += count

    def p2_dof(row: int, cut: int) -> int:
        return int(
            chronological_to_p2[row * cut_nodes + cut]
        )

    roles = ActualWakeStageBoundaryRoles(
        body_attachment_p1_dofs=np.arange(
            band_count * span_nodes,
            (band_count + 1) * span_nodes,
            dtype=np.int64,
        ),
        oldest_p1_dofs=np.arange(span_nodes, dtype=np.int64),
        body_attachment_p2_dofs=np.array(
            [p2_dof(2 * band_count, cut) for cut in range(cut_nodes)],
            dtype=np.int64,
        ),
        oldest_p2_dofs=np.array(
            [p2_dof(0, cut) for cut in range(cut_nodes)],
            dtype=np.int64,
        ),
        root_characteristic_interior_p2_dofs=np.array(
            [
                p2_dof(row, 0)
                for row in range(1, 2 * band_count)
            ],
            dtype=np.int64,
        ),
        tip_characteristic_interior_p2_dofs=np.array(
            [
                p2_dof(row, cut_nodes - 1)
                for row in range(1, 2 * band_count)
            ],
            dtype=np.int64,
        ),
        body_attachment_id=body_attachment_id,
    )
    topology = ActualWakeStageTopology(
        history_id=history.history_id,
        band_count=band_count,
        span_nodes=span_nodes,
        cut_node_count=cut_nodes,
        p1_vertices=vertices,
        p1_faces=faces,
        band_p1_vertex_dofs=band_p1_dofs,
        p2_topology=p2,
        band_p2_face_dofs=tuple(band_face_dofs),
        p2_dof_to_chronological=p2_to_chronological,
        chronological_to_p2_dof=chronological_to_p2,
        span_diagonal_patterns=patterns,
        boundary_roles=roles,
    )
    topology.global_p2_state(history, tolerance=tolerance)
    return topology
