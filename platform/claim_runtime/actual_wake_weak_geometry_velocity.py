"""Global weak P1 normal mesh velocity for an actual vortex sheet.

The physical face-normal velocity is projected in the surface L2 inner
product onto the finite-dimensional field

    v_h = sum_i N_i s_i n_i,

where ``n_i`` is the area-weighted vertex normal.  The nodal coefficients are
mesh-gauge degrees of freedom, not pointwise boundary traces.

No point extrapolation, kernel regularization, smoothing, damping, pressure,
force, LESP, target load, or structural state is present.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .actual_wake_stage_topology import (
    ActualWakeStageTopology,
    ActualWakeStageTopologyError,
)
from .actual_wake_stage_velocity import OwnedWakeQuadrature
from .distributed_doublet import (
    MaterialWakeHistory,
    QuadraticDoubletElement,
)


@dataclass(frozen=True)
class ActualWakeWeakNormalProjectionReport:
    p1_dof_count: int
    rank: int
    rank_deficiency: int
    condition_number: float
    weak_orthogonality_relative_residual: float
    relative_surface_L2_residual: float
    maximum_tangential_nodal_velocity: float
    gauge: str = "global_consistent_P1_normal_L2_projection"


@dataclass(frozen=True)
class ActualWakeWeakNormalGeometryVelocity:
    topology: ActualWakeStageTopology
    scalar_normal_speed: np.ndarray
    dof_velocity: np.ndarray
    dof_normals: np.ndarray
    report: ActualWakeWeakNormalProjectionReport

    def __post_init__(self) -> None:
        scalar = np.asarray(self.scalar_normal_speed, dtype=float)
        velocity = np.asarray(self.dof_velocity, dtype=float)
        normals = np.asarray(self.dof_normals, dtype=float)
        count = len(self.topology.p1_vertices)
        if (
            scalar.shape != (count,)
            or velocity.shape != (count, 3)
            or normals.shape != velocity.shape
            or not np.all(np.isfinite(scalar))
            or not np.all(np.isfinite(velocity))
            or not np.all(np.isfinite(normals))
        ):
            raise ActualWakeStageTopologyError(
                "weak normal projection has incompatible arrays"
            )
        object.__setattr__(
            self,
            "scalar_normal_speed",
            scalar.copy(),
        )
        object.__setattr__(self, "dof_velocity", velocity.copy())
        object.__setattr__(self, "dof_normals", normals.copy())


def actual_wake_area_weighted_vertex_normals(
    topology: ActualWakeStageTopology,
) -> np.ndarray:
    """Return objective unit normals from the current P1 triangulation."""
    normal_sum = np.zeros_like(topology.p1_vertices)
    for face in topology.p1_faces:
        element = QuadraticDoubletElement(
            topology.p1_vertices[face],
            np.zeros(6),
        )
        for dof in face:
            normal_sum[int(dof)] += element.area_vector
    norm = np.linalg.norm(normal_sum, axis=1)
    if np.any(norm <= np.finfo(float).eps):
        raise ActualWakeStageTopologyError(
            "weak projection vertex normal is undefined"
        )
    return normal_sum / norm[:, None]


def weak_normal_collocation_velocity(
    topology: ActualWakeStageTopology,
    history: MaterialWakeHistory,
    quadrature: OwnedWakeQuadrature,
    scalar_normal_speed: Any,
) -> np.ndarray:
    """Manufacture query velocities lying exactly in the weak P1 space."""
    topology._validate_history(history, tolerance=2.0e-12)
    scalar = np.asarray(scalar_normal_speed, dtype=float)
    count = len(topology.p1_vertices)
    if scalar.shape != (count,) or not np.all(np.isfinite(scalar)):
        raise ActualWakeStageTopologyError(
            "manufactured scalar speed has incompatible shape"
        )
    if len(quadrature.face_query_rows) != len(topology.p1_faces):
        raise ActualWakeStageTopologyError(
            "weak quadrature and topology face counts differ"
        )
    normals = actual_wake_area_weighted_vertex_normals(topology)
    velocity = np.empty_like(quadrature.query.points)
    for face_index, rows in enumerate(quadrature.face_query_rows):
        face = topology.p1_faces[face_index]
        element = QuadraticDoubletElement(
            topology.p1_vertices[face],
            np.zeros(6),
        )
        barycentric = quadrature.query.barycentric[rows]
        normal_factors = normals[face] @ element.normal
        target_normal_speed = (
            barycentric
            @ (scalar[face] * normal_factors)
        )
        velocity[rows] = (
            target_normal_speed[:, None] * element.normal
        )
    return velocity


def project_actual_wake_global_weak_normal_velocity(
    topology: ActualWakeStageTopology,
    history: MaterialWakeHistory,
    quadrature: OwnedWakeQuadrature,
    collocation_velocity: Any,
) -> ActualWakeWeakNormalGeometryVelocity:
    """Project physical face-normal speed into global continuous P1."""
    topology._validate_history(history, tolerance=2.0e-12)
    if not isinstance(quadrature, OwnedWakeQuadrature):
        raise ActualWakeStageTopologyError(
            "weak projection requires OwnedWakeQuadrature"
        )
    if len(quadrature.face_query_rows) != len(topology.p1_faces):
        raise ActualWakeStageTopologyError(
            "weak quadrature and topology face counts differ"
        )
    physical = np.asarray(collocation_velocity, dtype=float)
    if (
        physical.shape != quadrature.query.points.shape
        or not np.all(np.isfinite(physical))
    ):
        raise ActualWakeStageTopologyError(
            "weak collocation_velocity must match quadrature query"
        )
    count = len(topology.p1_vertices)
    normals = actual_wake_area_weighted_vertex_normals(topology)
    gram = np.zeros((count, count), dtype=float)
    right_hand_side = np.zeros(count, dtype=float)
    physical_norm = 0.0
    residual_records = []
    for face_index, rows in enumerate(quadrature.face_query_rows):
        face = topology.p1_faces[face_index]
        element = QuadraticDoubletElement(
            topology.p1_vertices[face],
            np.zeros(6),
        )
        barycentric = quadrature.query.barycentric[rows]
        reconstructed = barycentric @ element.vertices
        if (
            np.max(
                np.linalg.norm(
                    reconstructed - quadrature.query.points[rows],
                    axis=1,
                ),
                initial=0.0,
            )
            > 2.0e-12
        ):
            raise ActualWakeStageTopologyError(
                "weak quadrature owner reconstruction failed"
            )
        weights = (
            quadrature.reference_weights
            * np.linalg.norm(element.area_vector)
        )
        factors = normals[face] @ element.normal
        design = barycentric * factors[None, :]
        target = physical[rows] @ element.normal
        gram[np.ix_(face, face)] += np.einsum(
            "qi,q,qj->ij",
            design,
            weights,
            design,
        )
        right_hand_side[face] += np.einsum(
            "qi,q,q->i",
            design,
            weights,
            target,
        )
        physical_norm += float(np.dot(weights, target**2))
        residual_records.append((face, design, weights, target))
    rank = int(np.linalg.matrix_rank(gram))
    condition = float(np.linalg.cond(gram))
    if rank != count or not np.isfinite(condition):
        raise ActualWakeStageTopologyError(
            "weak P1 normal Gram system is singular or non-finite"
        )
    scalar = np.linalg.solve(gram, right_hand_side)
    weak_residual = gram @ scalar - right_hand_side
    weak_scale = max(
        float(np.linalg.norm(right_hand_side, ord=np.inf)),
        np.finfo(float).tiny,
    )
    residual_norm = 0.0
    for face, design, weights, target in residual_records:
        error = design @ scalar[face] - target
        residual_norm += float(np.dot(weights, error**2))
    velocity = scalar[:, None] * normals
    tangent = (
        velocity
        - np.einsum("ij,ij->i", velocity, normals)[:, None]
        * normals
    )
    return ActualWakeWeakNormalGeometryVelocity(
        topology=topology,
        scalar_normal_speed=scalar,
        dof_velocity=velocity,
        dof_normals=normals,
        report=ActualWakeWeakNormalProjectionReport(
            p1_dof_count=count,
            rank=rank,
            rank_deficiency=count - rank,
            condition_number=condition,
            weak_orthogonality_relative_residual=float(
                np.linalg.norm(weak_residual, ord=np.inf)
                / weak_scale
            ),
            relative_surface_L2_residual=float(
                np.sqrt(
                    residual_norm
                    / max(physical_norm, np.finfo(float).tiny)
                )
            ),
            maximum_tangential_nodal_velocity=float(
                np.max(
                    np.linalg.norm(tangent, axis=1),
                    initial=0.0,
                )
            ),
        ),
    )
