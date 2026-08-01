"""Classified P2 potential cut on a watertight triangular body.

The physical body mesh remains single-valued and closed.  Only the scalar
quadratic potential trace is duplicated across a prescribed lifting cut:

* internal cut vertices receive one upper and one lower potential degree of
  freedom;
* every cut-edge midpoint receives one upper and one lower potential degree
  of freedom;
* zero-jump cut endpoints retain one shared degree of freedom;
* every non-cut edge retains one shared P2 trace.

This module is a topology oracle.  It does not solve a boundary-integral
equation, choose circulation, create pressure, or contribute force.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .actual_boundary_p2_galerkin import (
    ClosedP2Topology,
    closed_p2_topology,
)
from .thick_body_neumann_shadow import (
    ClosedTriangularMesh,
    ThickBodyNeumannError,
)


class ClassifiedP2CutError(ValueError):
    """Invalid lifting-cut classification or trace topology."""


_LOCAL_EDGES = ((0, 1), (1, 2), (2, 0))


def _edge(first: int, second: int) -> tuple[int, int]:
    return min(int(first), int(second)), max(int(first), int(second))


@dataclass(frozen=True)
class ClassifiedP2CutTopology:
    """P2 trace topology with paired upper/lower cut degrees of freedom."""

    base_topology: ClosedP2Topology
    local_to_global: np.ndarray
    degree_of_freedom_coordinates: np.ndarray
    cut_edges: np.ndarray
    ordered_cut_vertex_indices: np.ndarray
    cut_node_coordinates: np.ndarray
    upper_cut_dofs: np.ndarray
    lower_cut_dofs: np.ndarray
    duplicated_vertex_count: int
    duplicated_edge_midpoint_count: int
    noncut_trace_dof_mismatch_count: int
    maximum_cut_coordinate_pair_gap: float

    @property
    def dof_count(self) -> int:
        return int(len(self.degree_of_freedom_coordinates))

    @property
    def duplicated_dof_count(self) -> int:
        return (
            int(self.duplicated_vertex_count)
            + int(self.duplicated_edge_midpoint_count)
        )

    def cut_jump(self, trace_values) -> np.ndarray:
        """Return upper-minus-lower trace at ordered cut P2 nodes."""
        values = np.asarray(trace_values, dtype=float)
        if (
            values.shape != (self.dof_count,)
            or not np.all(np.isfinite(values))
        ):
            raise ClassifiedP2CutError(
                "trace_values must be finite with shape (dof_count,)"
            )
        return (
            values[self.upper_cut_dofs]
            - values[self.lower_cut_dofs]
        )


def _face_edge_occurrences(
    mesh: ClosedTriangularMesh,
) -> dict[tuple[int, int], list[tuple[int, int]]]:
    occurrences: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for face_index, face in enumerate(mesh.faces):
        for local_edge, (first, second) in enumerate(_LOCAL_EDGES):
            key = _edge(face[first], face[second])
            occurrences.setdefault(key, []).append(
                (int(face_index), int(local_edge))
            )
    return occurrences


def _validated_indices(
    name: str,
    values: Iterable[int],
    *,
    upper_bound: int,
) -> set[int]:
    try:
        result = {int(value) for value in values}
    except (TypeError, ValueError) as exc:
        raise ClassifiedP2CutError(
            f"{name} must contain integer indices"
        ) from exc
    if any(value < 0 or value >= upper_bound for value in result):
        raise ClassifiedP2CutError(
            f"{name} contains an index outside [0,{upper_bound})"
        )
    return result


def _ordered_cut_vertices(
    mesh: ClosedTriangularMesh,
    cut_edges: set[tuple[int, int]],
    zero_jump_end_vertices: set[int],
) -> list[int]:
    adjacency: dict[int, set[int]] = {}
    for first, second in cut_edges:
        adjacency.setdefault(first, set()).add(second)
        adjacency.setdefault(second, set()).add(first)
    endpoints = {
        vertex for vertex, neighbours in adjacency.items()
        if len(neighbours) == 1
    }
    internal = {
        vertex for vertex, neighbours in adjacency.items()
        if len(neighbours) == 2
    }
    if (
        len(endpoints) != 2
        or len(endpoints) + len(internal) != len(adjacency)
        or endpoints != zero_jump_end_vertices
    ):
        raise ClassifiedP2CutError(
            "cut_edges must form one unbranched open chain whose two "
            "endpoints are exactly zero_jump_end_vertices"
        )
    start = min(
        endpoints,
        key=lambda index: tuple(mesh.vertices[index].tolist()),
    )
    ordered = [start]
    previous: int | None = None
    current = start
    while True:
        candidates = adjacency[current] - (
            set() if previous is None else {previous}
        )
        if not candidates:
            break
        if len(candidates) != 1:
            raise ClassifiedP2CutError(
                "cut traversal encountered a branch"
            )
        following = next(iter(candidates))
        ordered.append(following)
        previous, current = current, following
    if len(ordered) != len(adjacency):
        raise ClassifiedP2CutError(
            "cut_edges contain disconnected components"
        )
    return ordered


def classified_p2_cut_topology(
    mesh: ClosedTriangularMesh,
    *,
    upper_face_indices: Iterable[int],
    lower_face_indices: Iterable[int],
    cut_edges: Iterable[tuple[int, int]],
    zero_jump_end_vertices: Iterable[int],
) -> ClassifiedP2CutTopology:
    """Duplicate only scalar P2 trace DOFs across one classified body cut.

    Each cut edge must have exactly one incident upper face and one incident
    lower face.  Geometry is never copied or modified.  The returned ordered
    cut nodes alternate vertex and edge midpoint along the unbranched cut.
    """
    if not isinstance(mesh, ClosedTriangularMesh):
        raise ThickBodyNeumannError(
            "mesh must be validated by closed_triangular_mesh"
        )
    upper = _validated_indices(
        "upper_face_indices",
        upper_face_indices,
        upper_bound=len(mesh.faces),
    )
    lower = _validated_indices(
        "lower_face_indices",
        lower_face_indices,
        upper_bound=len(mesh.faces),
    )
    if not upper or not lower or upper & lower:
        raise ClassifiedP2CutError(
            "upper and lower face classifications must be non-empty "
            "and disjoint"
        )
    try:
        classified_edges = {
            _edge(first, second) for first, second in cut_edges
        }
    except (TypeError, ValueError) as exc:
        raise ClassifiedP2CutError(
            "cut_edges must contain pairs of vertex indices"
        ) from exc
    if not classified_edges:
        raise ClassifiedP2CutError("at least one cut edge is required")
    if any(
        first < 0
        or second >= len(mesh.vertices)
        or first == second
        for first, second in classified_edges
    ):
        raise ClassifiedP2CutError(
            "cut_edges contain invalid body vertex indices"
        )
    zero_endpoints = _validated_indices(
        "zero_jump_end_vertices",
        zero_jump_end_vertices,
        upper_bound=len(mesh.vertices),
    )
    ordered_vertices = _ordered_cut_vertices(
        mesh,
        classified_edges,
        zero_endpoints,
    )

    base = closed_p2_topology(mesh)
    occurrences = _face_edge_occurrences(mesh)
    base_edge_dof = {
        tuple(map(int, edge_vertices)): (
            base.vertex_dof_count + edge_index
        )
        for edge_index, edge_vertices in enumerate(base.edge_vertices)
    }
    for edge in classified_edges:
        incident = occurrences.get(edge, ())
        incident_faces = {face for face, _ in incident}
        if (
            len(incident) != 2
            or len(incident_faces & upper) != 1
            or len(incident_faces & lower) != 1
            or len(incident_faces) != 2
        ):
            raise ClassifiedP2CutError(
                f"cut edge {edge} must have exactly one upper and one "
                "lower incident face"
            )

    local = base.local_to_global.copy()
    coordinates = [
        coordinate.copy()
        for coordinate in base.degree_of_freedom_coordinates
    ]
    cut_vertex_set = set(ordered_vertices)
    internal_vertices = cut_vertex_set - zero_endpoints
    lower_vertex_dof: dict[int, int] = {}
    for vertex in sorted(internal_vertices):
        lower_vertex_dof[vertex] = len(coordinates)
        coordinates.append(mesh.vertices[vertex].copy())
    lower_edge_dof: dict[tuple[int, int], int] = {}
    for edge in sorted(classified_edges):
        lower_edge_dof[edge] = len(coordinates)
        coordinates.append(
            0.5 * (
                mesh.vertices[edge[0]] + mesh.vertices[edge[1]]
            )
        )

    for face_index in lower:
        face = mesh.faces[face_index]
        for local_vertex, vertex in enumerate(face):
            vertex_index = int(vertex)
            if vertex_index in lower_vertex_dof:
                local[face_index, local_vertex] = (
                    lower_vertex_dof[vertex_index]
                )
        for local_edge, (first, second) in enumerate(_LOCAL_EDGES):
            edge = _edge(face[first], face[second])
            if edge in lower_edge_dof:
                local[face_index, 3 + local_edge] = (
                    lower_edge_dof[edge]
                )

    noncut_mismatches = 0
    for edge, incident in occurrences.items():
        if edge in classified_edges:
            continue
        traces = []
        for face_index, local_edge in incident:
            face = mesh.faces[face_index]
            local_first, local_second = _LOCAL_EDGES[local_edge]
            endpoint_dofs = {
                int(face[local_first]): int(
                    local[face_index, local_first]
                ),
                int(face[local_second]): int(
                    local[face_index, local_second]
                ),
            }
            traces.append(
                (
                    endpoint_dofs[edge[0]],
                    int(local[face_index, 3 + local_edge]),
                    endpoint_dofs[edge[1]],
                )
            )
        if len(incident) != 2 or traces[0] != traces[1]:
            noncut_mismatches += 1

    upper_nodes: list[int] = []
    lower_nodes: list[int] = []
    cut_coordinates: list[np.ndarray] = []
    for segment_index, (first, second) in enumerate(
        zip(ordered_vertices, ordered_vertices[1:])
    ):
        if segment_index == 0:
            upper_nodes.append(int(first))
            lower_nodes.append(
                lower_vertex_dof.get(int(first), int(first))
            )
            cut_coordinates.append(mesh.vertices[first].copy())
        edge = _edge(first, second)
        upper_nodes.append(int(base_edge_dof[edge]))
        lower_nodes.append(int(lower_edge_dof[edge]))
        cut_coordinates.append(
            0.5 * (mesh.vertices[first] + mesh.vertices[second])
        )
        upper_nodes.append(int(second))
        lower_nodes.append(
            lower_vertex_dof.get(int(second), int(second))
        )
        cut_coordinates.append(mesh.vertices[second].copy())

    coordinate_array = np.asarray(coordinates, dtype=float)
    upper_array = np.asarray(upper_nodes, dtype=np.int64)
    lower_array = np.asarray(lower_nodes, dtype=np.int64)
    cut_coordinate_array = np.asarray(cut_coordinates, dtype=float)
    coordinate_gap = float(
        np.max(
            np.linalg.norm(
                coordinate_array[upper_array]
                - coordinate_array[lower_array],
                axis=1,
            ),
            initial=0.0,
        )
    )
    return ClassifiedP2CutTopology(
        base_topology=base,
        local_to_global=local,
        degree_of_freedom_coordinates=coordinate_array,
        cut_edges=np.asarray(sorted(classified_edges), dtype=np.int64),
        ordered_cut_vertex_indices=np.asarray(
            ordered_vertices,
            dtype=np.int64,
        ),
        cut_node_coordinates=cut_coordinate_array,
        upper_cut_dofs=upper_array,
        lower_cut_dofs=lower_array,
        duplicated_vertex_count=len(lower_vertex_dof),
        duplicated_edge_midpoint_count=len(lower_edge_dof),
        noncut_trace_dof_mismatch_count=noncut_mismatches,
        maximum_cut_coordinate_pair_gap=coordinate_gap,
    )
