"""Steady P2 actual-boundary/body-wake coupled potential equation.

The unknown vector contains only the classified body-potential trace.  The
steady material-wake strength is the upper-minus-lower body trace at the
classified trailing-edge cut and is eliminated exactly into the body
Galerkin matrix.  There is no independent circulation, wake-amplitude,
pressure, force, regularization, or target-load equation.

This CPU implementation is an equation/identifiability oracle for
``N3.1j3b6d16``.  It is not a production aerodynamic solver.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .actual_boundary_p2_galerkin import (
    _element_basis_doublet_potential,
    _global_shape_matrix,
    _surface_quadrature,
    paired_p2_triangle_integral,
)
from .classified_p2_cut_topology import (
    ClassifiedP2CutError,
    ClassifiedP2CutTopology,
)
from .distributed_doublet import (
    DistributedDoubletError,
    MaterialWakeBand,
    MaterialWakeHistory,
    newborn_material_wake_band,
)
from .thick_body_neumann_shadow import (
    ClosedTriangularMesh,
    ThickBodyNeumannError,
    constant_source_polygon_influence,
)


@dataclass(frozen=True)
class ActualBoundaryBodyWakeSolution:
    """One no-force coupled body/wake equation solution."""

    mesh: ClosedTriangularMesh
    topology: ClassifiedP2CutTopology
    incident_velocity: np.ndarray
    wall_velocity: np.ndarray
    source_strength: np.ndarray
    global_body_potential: np.ndarray
    body_face_potential: np.ndarray
    body_cut_jump: np.ndarray
    wake: MaterialWakeBand
    wake_history: MaterialWakeHistory
    matrix: np.ndarray
    body_matrix: np.ndarray
    wake_matrix: np.ndarray
    known_wake_right_hand_side: np.ndarray
    right_hand_side: np.ndarray
    weak_residual: np.ndarray
    relative_weak_residual: float
    rank: int
    condition_number: float
    body_unknown_count: int
    independent_wake_unknown_count: int
    independent_old_wake_unknown_count: int
    wake_attachment_error: float
    body_paired_topology_counts: dict[str, int]
    body_wake_paired_topology_counts: dict[str, int]


@dataclass(frozen=True)
class MaterialWakeCutAttachment:
    """Oriented material identity from the body cut to the newest wake edge.

    Vertex ids are declared in the span order used by every wake row.  The
    sign defines ``wake_mu = sign * P * (mu_upper - mu_lower)``, where ``P``
    is the exact P2 identity/reversal induced by the material vertex ids.
    No coordinate sorting or proximity matching is permitted.
    """

    ordered_body_cut_vertex_indices: np.ndarray
    wake_jump_from_body_cut_sign: int

    def __post_init__(self) -> None:
        indices = np.asarray(
            self.ordered_body_cut_vertex_indices,
            dtype=np.int64,
        )
        if (
            indices.ndim != 1
            or len(indices) < 2
            or len(np.unique(indices)) != len(indices)
        ):
            raise ClassifiedP2CutError(
                "ordered body-cut material ids must be a unique 1-D chain"
            )
        sign = self.wake_jump_from_body_cut_sign
        if (
            not isinstance(sign, (int, np.integer))
            or int(sign) not in (-1, 1)
        ):
            raise ClassifiedP2CutError(
                "wake_jump_from_body_cut_sign must be exactly -1 or +1"
            )
        object.__setattr__(
            self,
            "ordered_body_cut_vertex_indices",
            indices.copy(),
        )
        object.__setattr__(
            self,
            "wake_jump_from_body_cut_sign",
            int(sign),
        )

    def p2_trace_permutation(
        self,
        topology: ClassifiedP2CutTopology,
    ) -> np.ndarray:
        canonical = topology.ordered_cut_vertex_indices
        declared = self.ordered_body_cut_vertex_indices
        if np.array_equal(declared, canonical):
            return np.arange(
                len(topology.cut_node_coordinates),
                dtype=np.int64,
            )
        if np.array_equal(declared, canonical[::-1]):
            return np.arange(
                len(topology.cut_node_coordinates) - 1,
                -1,
                -1,
                dtype=np.int64,
            )
        raise ClassifiedP2CutError(
            "declared body-cut material ids must equal the classified "
            "cut chain in its forward or reverse orientation"
        )


def _finite_points(name: str, value: Any) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if (
        array.ndim != 2
        or array.shape[1] != 3
        or not np.all(np.isfinite(array))
    ):
        raise ThickBodyNeumannError(
            f"{name} must be a finite array with shape (n,3)"
        )
    return array


def _cut_jump_matrix(
    topology: ClassifiedP2CutTopology,
) -> np.ndarray:
    rows = len(topology.upper_cut_dofs)
    matrix = np.zeros((rows, topology.dof_count), dtype=float)
    for row, (upper, lower) in enumerate(
        zip(topology.upper_cut_dofs, topology.lower_cut_dofs)
    ):
        matrix[row, int(upper)] += 1.0
        matrix[row, int(lower)] -= 1.0
    return matrix


def _wake_face_from_temporal_cut_map(
    *,
    previous_edge: np.ndarray,
    current_edge: np.ndarray,
    cut_node_count: int,
    time_nodes: Any = (0.0, 0.5, 1.0),
    span_diagonal_pattern: str = "mirror_symmetric",
) -> tuple[MaterialWakeBand, np.ndarray]:
    zero_rows = np.zeros((3, cut_node_count), dtype=float)
    temporal_nodes = np.asarray(time_nodes, dtype=float)
    template = newborn_material_wake_band(
        sheet_id="coupled-body-wake-template",
        vortex_family="TEV",
        previous_edge=previous_edge,
        current_edge=current_edge,
        time_nodes=temporal_nodes,
        potential_jump_rows=zero_rows,
        span_diagonal_pattern=span_diagonal_pattern,
    )
    face_from_temporal_cut = np.empty(
        (len(template.surface), 6, 3, cut_node_count),
        dtype=float,
    )
    for temporal_index in range(3):
        for cut_index in range(cut_node_count):
            rows = np.zeros_like(zero_rows)
            rows[temporal_index, cut_index] = 1.0
            basis = newborn_material_wake_band(
                sheet_id=(
                    "coupled-body-wake-basis-"
                    f"{temporal_index}-{cut_index}"
                ),
                vortex_family="TEV",
                previous_edge=previous_edge,
                current_edge=current_edge,
                time_nodes=temporal_nodes,
                potential_jump_rows=rows,
                span_diagonal_pattern=span_diagonal_pattern,
            )
            face_from_temporal_cut[
                :, :, temporal_index, cut_index
            ] = basis.surface.face_mu
    return template, face_from_temporal_cut


def solve_actual_boundary_body_wake_p2(
    mesh: ClosedTriangularMesh,
    topology: ClassifiedP2CutTopology,
    *,
    incident_velocity: Any,
    wall_velocity: Any | None = None,
    downstream_edge_x: float | None,
    wake_edge_x_nodes: Any | None = None,
    fixed_old_wake_rows: Any | None = None,
    active_known_rows: Any | None = None,
    prescribed_wake_history: MaterialWakeHistory | None = None,
    prescribed_wake_attachment: MaterialWakeCutAttachment | None = None,
    target_quadrature_order: int = 5,
    source_quadrature_order: int | None = None,
) -> ActualBoundaryBodyWakeSolution:
    """Solve the steady body/wake interior-Dirichlet Galerkin equation."""
    if not isinstance(mesh, ClosedTriangularMesh):
        raise ThickBodyNeumannError(
            "mesh must be validated by closed_triangular_mesh"
        )
    if not isinstance(topology, ClassifiedP2CutTopology):
        raise ClassifiedP2CutError(
            "topology must be a ClassifiedP2CutTopology"
        )
    if topology.base_topology.local_to_global.shape[0] != len(mesh.faces):
        raise ClassifiedP2CutError(
            "classified topology does not belong to this body mesh"
        )
    incident = _finite_points("incident_velocity", incident_velocity)
    if incident.shape != mesh.centroids.shape:
        raise ThickBodyNeumannError(
            "incident_velocity must match body face centroids"
        )
    if wall_velocity is None:
        wall = np.zeros_like(incident)
    else:
        wall = _finite_points("wall_velocity", wall_velocity)
        if wall.shape != mesh.centroids.shape:
            raise ThickBodyNeumannError(
                "wall_velocity must match body face centroids"
            )
    if source_quadrature_order is None:
        source_order = int(target_quadrature_order)
    else:
        source_order = int(source_quadrature_order)
    if (
        not isinstance(target_quadrature_order, (int, np.integer))
        or int(target_quadrature_order) < 2
        or source_order < 2
    ):
        raise ThickBodyNeumannError(
            "target/source quadrature orders must be integers >=2"
        )
    current_edge = mesh.vertices[
        topology.ordered_cut_vertex_indices
    ].copy()
    if np.max(
        np.linalg.norm(
            current_edge
            - topology.cut_node_coordinates[::2],
            axis=1,
        ),
        initial=0.0,
    ) > 1.0e-14:
        raise ClassifiedP2CutError(
            "ordered cut vertices and P2 cut trace are inconsistent"
        )
    cut_map = _cut_jump_matrix(topology)
    active_cut_map = cut_map
    cut_node_count = len(topology.cut_node_coordinates)
    wake_specs = []
    wake_time_nodes = []
    wake_sheet_ids = []
    wake_patterns = []
    wake_attached_vertex_ids = []
    if prescribed_wake_history is not None:
        if not isinstance(
            prescribed_wake_history,
            MaterialWakeHistory,
        ):
            raise ThickBodyNeumannError(
                "prescribed_wake_history must be MaterialWakeHistory"
            )
        if (
            wake_edge_x_nodes is not None
            or fixed_old_wake_rows is not None
            or active_known_rows is not None
        ):
            raise ThickBodyNeumannError(
                "prescribed_wake_history is mutually exclusive with "
                "wake_edge_x_nodes and explicit wake-row arrays"
            )
        history_report = prescribed_wake_history.continuity_report()
        if not history_report.compatible:
            raise ThickBodyNeumannError(
                "prescribed material wake history is discontinuous"
            )
        if prescribed_wake_history.span_nodes != len(current_edge):
            raise ThickBodyNeumannError(
                "prescribed wake span topology does not match body cut"
            )
        if prescribed_wake_attachment is None:
            attachment_vertex_ids = (
                topology.ordered_cut_vertex_indices.copy()
            )
        else:
            if not isinstance(
                prescribed_wake_attachment,
                MaterialWakeCutAttachment,
            ):
                raise ThickBodyNeumannError(
                    "prescribed_wake_attachment must be "
                    "MaterialWakeCutAttachment"
                )
            permutation = (
                prescribed_wake_attachment.p2_trace_permutation(
                    topology
                )
            )
            attachment_vertex_ids = (
                prescribed_wake_attachment
                .ordered_body_cut_vertex_indices.copy()
            )
            active_cut_map = (
                prescribed_wake_attachment
                .wake_jump_from_body_cut_sign
                * cut_map[permutation]
            )
        newest = prescribed_wake_history.bands[-1]
        newest_current = newest.surface.vertices[
            newest.span_nodes :
        ]
        if np.max(
            np.linalg.norm(
                newest_current
                - mesh.vertices[attachment_vertex_ids],
                axis=1,
            ),
            initial=0.0,
        ) > 1.0e-12:
            raise ThickBodyNeumannError(
                "newest prescribed wake edge must attach to body cut"
            )
        for band_index, band in enumerate(
            prescribed_wake_history.bands
        ):
            previous_edge = band.surface.vertices[
                : band.span_nodes
            ].copy()
            band_current_edge = band.surface.vertices[
                band.span_nodes :
            ].copy()
            matches = []
            for pattern in ("forward", "mirror_symmetric"):
                try:
                    candidate, candidate_map = (
                        _wake_face_from_temporal_cut_map(
                            previous_edge=previous_edge,
                            current_edge=band_current_edge,
                            cut_node_count=cut_node_count,
                            time_nodes=band.time_nodes,
                            span_diagonal_pattern=pattern,
                        )
                    )
                except DistributedDoubletError:
                    continue
                if (
                    np.array_equal(
                        candidate.surface.faces,
                        band.surface.faces,
                    )
                    and np.array_equal(
                        candidate.surface.vertices,
                        band.surface.vertices,
                    )
                ):
                    reconstructed_mu = np.einsum(
                        "fitk,tk->fi",
                        candidate_map,
                        band.potential_jump_rows,
                    )
                    if np.max(
                        np.abs(
                            reconstructed_mu - band.surface.face_mu
                        ),
                        initial=0.0,
                    ) <= 1.0e-12:
                        matches.append(
                            (pattern, candidate, candidate_map)
                        )
            if len(matches) != 1:
                raise ThickBodyNeumannError(
                    "prescribed wake band topology/trace is not a "
                    "supported explicit P2 material band"
                )
            pattern, wake_template, face_from_temporal_cut = (
                matches[0]
            )
            wake_specs.append(
                (
                    wake_template,
                    face_from_temporal_cut,
                    previous_edge,
                    band_current_edge,
                    band_index
                    == len(prescribed_wake_history.bands) - 1,
                )
            )
            wake_time_nodes.append(band.time_nodes.copy())
            wake_sheet_ids.append(band.sheet_id)
            wake_patterns.append(pattern)
            wake_attached_vertex_ids.append(
                attachment_vertex_ids.copy()
                if band_index
                == len(prescribed_wake_history.bands) - 1
                else np.empty(0, dtype=np.int64)
            )
        fixed_rows = np.stack(
            [
                band.potential_jump_rows
                for band in prescribed_wake_history.bands[:-1]
            ],
            axis=0,
        ) if len(prescribed_wake_history.bands) > 1 else np.empty(
            (0, 3, cut_node_count),
            dtype=float,
        )
        active_rows = newest.potential_jump_rows[:2].copy()
        unsteady_partition = True
        wake_history_id = prescribed_wake_history.history_id
    else:
        trailing_edge_x = float(
            np.max(topology.cut_node_coordinates[:, 0])
        )
        if (
            not np.isfinite(downstream_edge_x)
            or downstream_edge_x <= trailing_edge_x
        ):
            raise ThickBodyNeumannError(
                "downstream_edge_x must lie strictly downstream of the cut"
            )
        if wake_edge_x_nodes is None:
            edge_x = np.array(
                (trailing_edge_x, float(downstream_edge_x)),
                dtype=float,
            )
        else:
            edge_x = np.asarray(wake_edge_x_nodes, dtype=float)
            if (
                edge_x.ndim != 1
                or len(edge_x) < 2
                or not np.all(np.isfinite(edge_x))
                or abs(edge_x[0] - trailing_edge_x) > 1.0e-14
                or abs(edge_x[-1] - downstream_edge_x) > 1.0e-14
                or np.any(np.diff(edge_x) <= 0.0)
            ):
                raise ThickBodyNeumannError(
                    "wake_edge_x_nodes must be a strictly increasing line "
                    "from the trailing edge to downstream_edge_x"
                )
        for interval in reversed(range(len(edge_x) - 1)):
            previous_edge = current_edge.copy()
            previous_edge[:, 0] = edge_x[interval + 1]
            band_current_edge = current_edge.copy()
            band_current_edge[:, 0] = edge_x[interval]
            wake_template, face_from_temporal_cut = (
                _wake_face_from_temporal_cut_map(
                    previous_edge=previous_edge,
                    current_edge=band_current_edge,
                    cut_node_count=cut_node_count,
                )
            )
            wake_specs.append(
                (
                    wake_template,
                    face_from_temporal_cut,
                    previous_edge,
                    band_current_edge,
                    interval == 0,
                )
            )
        wake_time_nodes = [
            np.array(
                (
                    float(index),
                    float(index) + 0.5,
                    float(index) + 1.0,
                )
            )
            for index in range(len(wake_specs))
        ]
        wake_sheet_ids = [
            f"coupled-steady-TEV-{index}"
            for index in range(len(wake_specs))
        ]
        wake_patterns = [
            "mirror_symmetric" for _ in wake_specs
        ]
        wake_attached_vertex_ids = [
            (
                topology.ordered_cut_vertex_indices.copy()
                if index == len(wake_specs) - 1
                else np.empty(0, dtype=np.int64)
            )
            for index in range(len(wake_specs))
        ]
        wake_history_id = "coupled-steady-TEV-history"
        unsteady_partition = (
            fixed_old_wake_rows is not None
            or active_known_rows is not None
        )
        if unsteady_partition:
            if (
                fixed_old_wake_rows is None
                or active_known_rows is None
            ):
                raise ThickBodyNeumannError(
                    "fixed_old_wake_rows and active_known_rows must be "
                    "provided together"
                )
            fixed_rows = np.asarray(
                fixed_old_wake_rows,
                dtype=float,
            )
            active_rows = np.asarray(
                active_known_rows,
                dtype=float,
            )
            expected_fixed_shape = (
                max(len(wake_specs) - 1, 0),
                3,
                cut_node_count,
            )
            if (
                fixed_rows.shape != expected_fixed_shape
                or active_rows.shape != (2, cut_node_count)
                or not np.all(np.isfinite(fixed_rows))
                or not np.all(np.isfinite(active_rows))
            ):
                raise ThickBodyNeumannError(
                    "unsteady wake rows have incompatible shapes: "
                    f"fixed expected {expected_fixed_shape}, "
                    f"active expected {(2, cut_node_count)}"
                )
            fixed_rows = fixed_rows.copy()
            active_rows = active_rows.copy()
        else:
            fixed_rows = np.empty(
                (0, 3, cut_node_count),
                dtype=float,
            )
            active_rows = np.empty(
                (0, cut_node_count),
                dtype=float,
            )

    affine_wake_specs = []
    for band_index, (
        wake_template,
        face_from_temporal_cut,
        previous_edge,
        band_current_edge,
        attached_to_body,
    ) in enumerate(wake_specs):
        if unsteady_partition:
            if band_index < len(wake_specs) - 1:
                rows = fixed_rows[band_index]
                face_constant = np.einsum(
                    "fitk,tk->fi",
                    face_from_temporal_cut,
                    rows,
                )
                face_linear = np.zeros(
                    (
                        len(wake_template.surface),
                        6,
                        topology.dof_count,
                    ),
                    dtype=float,
                )
            else:
                known_rows = np.vstack(
                    (
                        active_rows,
                        np.zeros((1, cut_node_count)),
                    )
                )
                face_constant = np.einsum(
                    "fitk,tk->fi",
                    face_from_temporal_cut,
                    known_rows,
                )
                face_linear = np.einsum(
                    "fik,kj->fij",
                    face_from_temporal_cut[:, :, 2, :],
                    active_cut_map,
                )
        else:
            face_constant = np.zeros(
                (len(wake_template.surface), 6),
                dtype=float,
            )
            face_linear = np.einsum(
                "fitk,kj->fij",
                face_from_temporal_cut,
                cut_map,
            )
        affine_wake_specs.append(
            (
                wake_template,
                face_linear,
                face_constant,
                previous_edge,
                band_current_edge,
                attached_to_body,
            )
        )

    points, weights, owner, barycentric = _surface_quadrature(
        mesh,
        int(target_quadrature_order),
    )
    test = _global_shape_matrix(topology, owner, barycentric)
    weighted_test = test.T * weights[None, :]
    body_face_sets = [set(map(int, face)) for face in mesh.faces]
    body_common = np.zeros(
        (len(mesh.faces), len(mesh.faces)),
        dtype=np.int8,
    )
    for target_face in range(len(mesh.faces)):
        for source_face in range(len(mesh.faces)):
            body_common[target_face, source_face] = len(
                body_face_sets[target_face]
                & body_face_sets[source_face]
            )

    body_pointwise = np.zeros_like(test)
    for source_face, face in enumerate(mesh.faces):
        rows = body_common[owner, source_face] == 0
        local_influence = np.zeros((len(points), 6), dtype=float)
        local_influence[rows] = _element_basis_doublet_potential(
            mesh.vertices[face],
            points[rows],
            source_order=source_order,
        )
        body_pointwise[
            :, topology.local_to_global[source_face]
        ] += local_influence
    body_matrix = weighted_test @ (0.5 * test + body_pointwise)

    source_strength = np.einsum(
        "ij,ij->i",
        incident - wall,
        mesh.normals,
    )
    weak_source_potential = np.zeros(len(points), dtype=float)
    for source_face, face in enumerate(mesh.faces):
        contribution = constant_source_polygon_influence(
            mesh.vertices[face],
            points,
            strength=float(source_strength[source_face]),
            on_surface_side="principal",
        ).potential
        rows = body_common[owner, source_face] == 0
        weak_source_potential[rows] += contribution[rows]
    weak_source_integral = weighted_test @ weak_source_potential

    body_counts = {
        "common_triangle": 0,
        "common_edge": 0,
        "common_vertex": 0,
    }
    labels = {
        3: "common_triangle",
        2: "common_edge",
        1: "common_vertex",
    }
    for target_face, target_ids in enumerate(mesh.faces):
        target_global = topology.local_to_global[target_face]
        for source_face, source_ids in enumerate(mesh.faces):
            count = int(body_common[target_face, source_face])
            if count == 0:
                continue
            pair = paired_p2_triangle_integral(
                mesh.vertices[target_ids],
                mesh.vertices[source_ids],
                target_vertex_ids=target_ids,
                source_vertex_ids=source_ids,
                quadrature_order=source_order,
            )
            source_global = topology.local_to_global[source_face]
            body_matrix[np.ix_(target_global, source_global)] += (
                pair.doublet_block
            )
            weak_source_integral[target_global] += (
                float(source_strength[source_face])
                * pair.source_vector
            )
            body_counts[labels[count]] += 1

    wake_matrix = np.zeros_like(body_matrix)
    known_wake_integral = np.zeros(
        topology.dof_count,
        dtype=float,
    )
    body_wake_counts = {
        "common_edge": 0,
        "common_vertex": 0,
    }
    next_wake_vertex_id = len(mesh.vertices)
    for wake_index, (
        wake_template,
        wake_face_from_body,
        wake_face_constant,
        _previous_edge,
        _band_current_edge,
        attached_to_body,
    ) in enumerate(affine_wake_specs):
        span_nodes = wake_template.span_nodes
        previous_vertex_ids = (
            next_wake_vertex_id + np.arange(span_nodes, dtype=int)
        )
        next_wake_vertex_id += span_nodes
        if attached_to_body:
            band_current_vertex_ids = (
                wake_attached_vertex_ids[wake_index]
            )
        else:
            band_current_vertex_ids = (
                next_wake_vertex_id + np.arange(span_nodes, dtype=int)
            )
            next_wake_vertex_id += span_nodes
        wake_vertex_ids = np.concatenate(
            (previous_vertex_ids, band_current_vertex_ids)
        )
        wake_face_ids = wake_vertex_ids[
            wake_template.surface.faces
        ]
        body_wake_common = np.zeros(
            (len(mesh.faces), len(wake_template.surface.faces)),
            dtype=np.int8,
        )
        for target_face in range(len(mesh.faces)):
            for source_face in range(
                len(wake_template.surface.faces)
            ):
                body_wake_common[target_face, source_face] = len(
                    body_face_sets[target_face]
                    & set(map(int, wake_face_ids[source_face]))
                )

        wake_pointwise = np.zeros(
            (len(points), topology.dof_count),
            dtype=float,
        )
        known_wake_pointwise = np.zeros(len(points), dtype=float)
        for source_face, face in enumerate(
            wake_template.surface.faces
        ):
            rows = body_wake_common[owner, source_face] == 0
            local_influence = np.zeros((len(points), 6), dtype=float)
            local_influence[rows] = (
                _element_basis_doublet_potential(
                    wake_template.surface.vertices[face],
                    points[rows],
                    source_order=source_order,
                )
            )
            wake_pointwise += (
                local_influence
                @ wake_face_from_body[source_face]
            )
            known_wake_pointwise += (
                local_influence
                @ wake_face_constant[source_face]
            )
        wake_matrix += weighted_test @ wake_pointwise
        known_wake_integral += (
            weighted_test @ known_wake_pointwise
        )

        for target_face, target_ids in enumerate(mesh.faces):
            target_global = topology.local_to_global[target_face]
            for source_face, source_ids in enumerate(wake_face_ids):
                count = int(
                    body_wake_common[target_face, source_face]
                )
                if count == 0:
                    continue
                if count not in (1, 2):
                    raise ThickBodyNeumannError(
                        "body/wake pair has invalid shared topology"
                    )
                pair = paired_p2_triangle_integral(
                    mesh.vertices[target_ids],
                    wake_template.surface.vertices[
                        wake_template.surface.faces[source_face]
                    ],
                    target_vertex_ids=target_ids,
                    source_vertex_ids=source_ids,
                    quadrature_order=source_order,
                )
                wake_matrix[target_global] += (
                    pair.doublet_block
                    @ wake_face_from_body[source_face]
                )
                known_wake_integral[target_global] += (
                    pair.doublet_block
                    @ wake_face_constant[source_face]
                )
                body_wake_counts[labels[count]] += 1

    matrix = body_matrix + wake_matrix
    known_wake_right_hand_side = -known_wake_integral
    right_hand_side = (
        -weak_source_integral + known_wake_right_hand_side
    )
    rank = int(np.linalg.matrix_rank(matrix))
    condition_number = float(np.linalg.cond(matrix))
    if rank != topology.dof_count or not np.isfinite(condition_number):
        raise ThickBodyNeumannError(
            "coupled body/wake matrix is rank-deficient or non-finite: "
            f"rank={rank}, unknowns={topology.dof_count}, "
            f"condition={condition_number}"
        )
    try:
        body_potential = np.linalg.solve(matrix, right_hand_side)
    except np.linalg.LinAlgError as error:
        raise ThickBodyNeumannError(
            "coupled body/wake solve failed"
        ) from error
    weak_residual = matrix @ body_potential - right_hand_side
    residual_scale = max(
        float(np.linalg.norm(right_hand_side)),
        np.finfo(float).tiny,
    )
    cut_jump = cut_map @ body_potential
    active_wake_jump = active_cut_map @ body_potential
    solved_bands = []
    for band_index, (
        _wake_template,
        _face_from_temporal_cut,
        previous_edge,
        band_current_edge,
        _attached_to_body,
    ) in enumerate(wake_specs):
        if unsteady_partition:
            if band_index < len(wake_specs) - 1:
                wake_rows = fixed_rows[band_index]
            else:
                wake_rows = np.vstack(
                    (active_rows, active_wake_jump)
                )
        else:
            wake_rows = np.repeat(cut_jump[None, :], 3, axis=0)
        solved_bands.append(
            newborn_material_wake_band(
                sheet_id=wake_sheet_ids[band_index],
                vortex_family="TEV",
                previous_edge=previous_edge,
                current_edge=band_current_edge,
                time_nodes=wake_time_nodes[band_index],
                potential_jump_rows=wake_rows,
                span_diagonal_pattern=wake_patterns[band_index],
            )
        )
    wake_history = MaterialWakeHistory(
        wake_history_id,
        tuple(solved_bands),
    )
    history_report = wake_history.continuity_report()
    if not history_report.compatible:
        raise ThickBodyNeumannError(
            f"constructed wake history is discontinuous: {history_report}"
        )
    wake = wake_history.bands[-1]
    attachment_error = float(
        np.max(
            np.abs(
                wake.potential_jump_rows[-1]
                - active_wake_jump
            ),
            initial=0.0,
        )
    )
    return ActualBoundaryBodyWakeSolution(
        mesh=mesh,
        topology=topology,
        incident_velocity=incident.copy(),
        wall_velocity=wall.copy(),
        source_strength=source_strength,
        global_body_potential=body_potential,
        body_face_potential=body_potential[topology.local_to_global],
        body_cut_jump=cut_jump,
        wake=wake,
        wake_history=wake_history,
        matrix=matrix,
        body_matrix=body_matrix,
        wake_matrix=wake_matrix,
        known_wake_right_hand_side=known_wake_right_hand_side,
        right_hand_side=right_hand_side,
        weak_residual=weak_residual,
        relative_weak_residual=float(
            np.linalg.norm(weak_residual) / residual_scale
        ),
        rank=rank,
        condition_number=condition_number,
        body_unknown_count=topology.dof_count,
        independent_wake_unknown_count=0,
        independent_old_wake_unknown_count=0,
        wake_attachment_error=attachment_error,
        body_paired_topology_counts=body_counts,
        body_wake_paired_topology_counts=body_wake_counts,
    )
