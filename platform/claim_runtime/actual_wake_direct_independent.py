"""Direct independent-current-wake Galerkin assembly.

For a prescribed actual material-wake history, every completed band and the
first two temporal rows of the newest band are known right-hand-side data.
Only the newest current row is an independent active P2 trace ``g``.  This
module assembles its body-boundary influence directly:

    B phi + W g = b.

The implementation deliberately does not recover ``W`` from an eliminated
matrix or a right inverse.  It applies the active trace to the newest wake
faces, evaluates ordinary body/wake pairs pointwise, and uses the same paired
P2 triangle integral for body/wake common-edge and common-vertex pairs.

There is no pressure, force, Kutta law, load target, core, smoothing,
regularization, or production activation in this module.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .actual_boundary_body_wake import (
    MaterialWakeCutAttachment,
    _wake_face_from_temporal_cut_map,
)
from .actual_boundary_p2_galerkin import (
    _element_basis_doublet_potential,
    _global_shape_matrix,
    _surface_quadrature,
    paired_p2_triangle_integral,
)
from .actual_wake_kutta_closure_roles import (
    CutRoleOperators,
    cut_role_operators,
)
from .classified_p2_cut_topology import ClassifiedP2CutTopology
from .distributed_doublet import (
    DistributedDoubletError,
    MaterialWakeBand,
    MaterialWakeHistory,
)
from .thick_body_neumann_shadow import ClosedTriangularMesh


class DirectIndependentWakeError(ValueError):
    """Invalid prescribed geometry or direct independent-wake assembly."""


@dataclass(frozen=True)
class DirectIndependentWakeAssembly:
    """One direct ``W`` operator in canonical active body-cut coordinates."""

    matrix: np.ndarray
    active_jump: np.ndarray
    full_current_trace_from_active: np.ndarray
    current_face_potential_from_active: np.ndarray
    active_row_indices: np.ndarray
    zero_row_indices: np.ndarray
    span_diagonal_pattern: str
    target_quadrature_order: int
    source_quadrature_order: int
    rank: int
    body_wake_paired_topology_counts: dict[str, int]

    @property
    def eliminated_wake_matrix(self) -> np.ndarray:
        """Return the corresponding body-trace-eliminated wake block."""

        return self.matrix @ self.active_jump


def _supported_newest_band_map(
    band: MaterialWakeBand,
    *,
    cut_node_count: int,
) -> tuple[MaterialWakeBand, np.ndarray, str]:
    """Recover the explicit temporal/cut interpolation of one wake band."""

    previous_edge = band.surface.vertices[: band.span_nodes]
    current_edge = band.surface.vertices[band.span_nodes :]
    matches: list[tuple[MaterialWakeBand, np.ndarray, str]] = []
    for pattern in ("forward", "mirror_symmetric"):
        try:
            template, face_from_temporal_cut = (
                _wake_face_from_temporal_cut_map(
                    previous_edge=previous_edge,
                    current_edge=current_edge,
                    cut_node_count=cut_node_count,
                    time_nodes=band.time_nodes,
                    span_diagonal_pattern=pattern,
                )
            )
        except DistributedDoubletError:
            continue
        if (
            not np.array_equal(
                template.surface.faces,
                band.surface.faces,
            )
            or not np.array_equal(
                template.surface.vertices,
                band.surface.vertices,
            )
        ):
            continue
        reconstructed = np.einsum(
            "fitk,tk->fi",
            face_from_temporal_cut,
            band.potential_jump_rows,
        )
        if (
            np.max(
                np.abs(reconstructed - band.surface.face_mu),
                initial=0.0,
            )
            <= 1.0e-12
        ):
            matches.append(
                (template, face_from_temporal_cut, pattern)
            )
    if len(matches) != 1:
        raise DirectIndependentWakeError(
            "newest prescribed wake band is not one uniquely supported "
            "explicit P2 material-band topology"
        )
    return matches[0]


def _full_current_trace_from_active(
    topology: ClassifiedP2CutTopology,
    operators: CutRoleOperators,
    attachment: MaterialWakeCutAttachment | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Map canonical active body-cut values into the newest wake row."""

    full_count = operators.full_cut_node_count
    active_count = operators.independent_jump_count
    injection = np.zeros((full_count, active_count), dtype=float)
    injection[
        operators.active_row_indices,
        np.arange(active_count, dtype=np.int64),
    ] = 1.0
    if attachment is None:
        material_vertex_ids = (
            topology.ordered_cut_vertex_indices.copy()
        )
        permutation = np.arange(full_count, dtype=np.int64)
        sign = 1
    else:
        if not isinstance(attachment, MaterialWakeCutAttachment):
            raise DirectIndependentWakeError(
                "attachment must be MaterialWakeCutAttachment or None"
            )
        material_vertex_ids = (
            attachment.ordered_body_cut_vertex_indices.copy()
        )
        permutation = attachment.p2_trace_permutation(topology)
        sign = attachment.wake_jump_from_body_cut_sign
    return sign * injection[permutation], material_vertex_ids


def _positive_quadrature_order(name: str, value: Any) -> int:
    if (
        not isinstance(value, (int, np.integer))
        or int(value) < 2
    ):
        raise DirectIndependentWakeError(
            f"{name} must be an integer >=2"
        )
    return int(value)


def assemble_direct_independent_wake_matrix(
    mesh: ClosedTriangularMesh,
    topology: ClassifiedP2CutTopology,
    prescribed_wake_history: MaterialWakeHistory,
    *,
    prescribed_wake_attachment: (
        MaterialWakeCutAttachment | None
    ) = None,
    target_quadrature_order: int = 5,
    source_quadrature_order: int | None = None,
) -> DirectIndependentWakeAssembly:
    """Assemble ``W`` from the newest prescribed wake panels directly.

    The returned columns use the canonical active body-cut order supplied by
    :func:`cut_role_operators`.  Attachment reversal/sign is consumed only
    when those columns are mapped onto the newest material-wake row.
    """

    if not isinstance(mesh, ClosedTriangularMesh):
        raise DirectIndependentWakeError(
            "mesh must be a ClosedTriangularMesh"
        )
    if not isinstance(topology, ClassifiedP2CutTopology):
        raise DirectIndependentWakeError(
            "topology must be a ClassifiedP2CutTopology"
        )
    if topology.base_topology.local_to_global.shape[0] != len(
        mesh.faces
    ):
        raise DirectIndependentWakeError(
            "classified topology does not belong to the body mesh"
        )
    if not isinstance(prescribed_wake_history, MaterialWakeHistory):
        raise DirectIndependentWakeError(
            "prescribed_wake_history must be MaterialWakeHistory"
        )
    report = prescribed_wake_history.continuity_report()
    if not report.compatible:
        raise DirectIndependentWakeError(
            f"prescribed wake history is discontinuous: {report}"
        )
    target_order = _positive_quadrature_order(
        "target_quadrature_order",
        target_quadrature_order,
    )
    source_order = (
        target_order
        if source_quadrature_order is None
        else _positive_quadrature_order(
            "source_quadrature_order",
            source_quadrature_order,
        )
    )

    operators = cut_role_operators(topology)
    cut_count = operators.full_cut_node_count
    expected_span_nodes = len(
        topology.ordered_cut_vertex_indices
    )
    if (
        prescribed_wake_history.span_nodes
        != expected_span_nodes
        or cut_count != 2 * expected_span_nodes - 1
    ):
        raise DirectIndependentWakeError(
            "prescribed wake span topology does not match the body cut"
        )

    full_trace_map, attachment_vertex_ids = (
        _full_current_trace_from_active(
            topology,
            operators,
            prescribed_wake_attachment,
        )
    )
    newest = prescribed_wake_history.bands[-1]
    newest_current_edge = newest.surface.vertices[
        newest.span_nodes :
    ]
    if (
        np.max(
            np.linalg.norm(
                newest_current_edge
                - mesh.vertices[attachment_vertex_ids],
                axis=1,
            ),
            initial=0.0,
        )
        > 1.0e-12
    ):
        raise DirectIndependentWakeError(
            "newest prescribed wake edge is not attached to the declared "
            "body-cut material vertices"
        )

    template, face_from_temporal_cut, pattern = (
        _supported_newest_band_map(
            newest,
            cut_node_count=cut_count,
        )
    )
    face_from_active = np.einsum(
        "fik,kr->fir",
        face_from_temporal_cut[:, :, 2, :],
        full_trace_map,
    )

    points, weights, owner, barycentric = _surface_quadrature(
        mesh,
        target_order,
    )
    test = _global_shape_matrix(topology, owner, barycentric)
    weighted_test = test.T * weights[None, :]

    body_face_vertex_ids = [
        set(map(int, face)) for face in mesh.faces
    ]
    previous_vertex_ids = (
        len(mesh.vertices)
        + np.arange(newest.span_nodes, dtype=np.int64)
    )
    wake_vertex_ids = np.concatenate(
        (previous_vertex_ids, attachment_vertex_ids)
    )
    wake_face_vertex_ids = wake_vertex_ids[
        template.surface.faces
    ]
    common = np.zeros(
        (len(mesh.faces), len(template.surface.faces)),
        dtype=np.int8,
    )
    for target_face in range(len(mesh.faces)):
        for source_face, source_ids in enumerate(
            wake_face_vertex_ids
        ):
            common[target_face, source_face] = len(
                body_face_vertex_ids[target_face]
                & set(map(int, source_ids))
            )

    wake_pointwise = np.zeros(
        (len(points), operators.independent_jump_count),
        dtype=float,
    )
    for source_face, face in enumerate(template.surface.faces):
        ordinary_rows = common[owner, source_face] == 0
        local_influence = np.zeros((len(points), 6), dtype=float)
        local_influence[ordinary_rows] = (
            _element_basis_doublet_potential(
                template.surface.vertices[face],
                points[ordinary_rows],
                source_order=source_order,
            )
        )
        wake_pointwise += (
            local_influence @ face_from_active[source_face]
        )
    matrix = weighted_test @ wake_pointwise

    paired_counts = {"common_edge": 0, "common_vertex": 0}
    for target_face, target_ids in enumerate(mesh.faces):
        target_global = topology.local_to_global[target_face]
        for source_face, source_ids in enumerate(
            wake_face_vertex_ids
        ):
            count = int(common[target_face, source_face])
            if count == 0:
                continue
            if count not in (1, 2):
                raise DirectIndependentWakeError(
                    "body/newest-wake pair has invalid shared topology"
                )
            pair = paired_p2_triangle_integral(
                mesh.vertices[target_ids],
                template.surface.vertices[
                    template.surface.faces[source_face]
                ],
                target_vertex_ids=target_ids,
                source_vertex_ids=source_ids,
                quadrature_order=source_order,
            )
            matrix[target_global] += (
                pair.doublet_block
                @ face_from_active[source_face]
            )
            paired_counts[
                "common_edge" if count == 2 else "common_vertex"
            ] += 1

    if not np.all(np.isfinite(matrix)):
        raise DirectIndependentWakeError(
            "direct independent-wake matrix contains non-finite values"
        )
    return DirectIndependentWakeAssembly(
        matrix=matrix.copy(),
        active_jump=operators.active_jump.copy(),
        full_current_trace_from_active=full_trace_map.copy(),
        current_face_potential_from_active=face_from_active.copy(),
        active_row_indices=operators.active_row_indices.copy(),
        zero_row_indices=operators.zero_row_indices.copy(),
        span_diagonal_pattern=pattern,
        target_quadrature_order=target_order,
        source_quadrature_order=source_order,
        rank=int(np.linalg.matrix_rank(matrix)),
        body_wake_paired_topology_counts=paired_counts.copy(),
    )
