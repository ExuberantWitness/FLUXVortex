"""Continuous-P2 actual-boundary Galerkin potential shadow.

Stage S1 of claim ``N3.1j3b6d3`` uses one globally shared quadratic
Lagrange degree of freedom at every closed-mesh vertex and edge midpoint.
The weak interior-Dirichlet equation is

    integral psi_i [0.5*mu + PV(D mu) + S sigma] dS = 0,

with prescribed ``sigma=(u_incident-u_wall).n``.  The exterior surface
potential is ``-mu``; its P2 surface gradient supplies tangential velocity.

This is an attached-flow equation/pressure oracle only.  It contains no
circulation constraint, wake, Kutta closure, material time derivative,
separated pressure, force fit, or production activation.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np

from .distributed_doublet import (
    QuadraticDoubletSurface,
    _triangle_quadrature,
    p2_shape_values,
)
from .doublet_potential import surface_doublet_potential
from .thick_body_neumann_shadow import (
    ClosedTriangularMesh,
    ThickBodyNeumannError,
    constant_source_polygon_influence,
)


@dataclass(frozen=True)
class ClosedP2Topology:
    local_to_global: np.ndarray
    edge_vertices: np.ndarray
    degree_of_freedom_coordinates: np.ndarray
    vertex_dof_count: int
    edge_dof_count: int

    @property
    def dof_count(self) -> int:
        return self.vertex_dof_count + self.edge_dof_count


@dataclass(frozen=True)
class ActualBoundaryP2GalerkinSolution:
    mesh: ClosedTriangularMesh
    topology: ClosedP2Topology
    incident_velocity: np.ndarray
    wall_velocity: np.ndarray
    source_strength: np.ndarray
    global_doublet_strength: np.ndarray
    surface: QuadraticDoubletSurface
    matrix: np.ndarray
    right_hand_side: np.ndarray
    weak_residual: np.ndarray
    relative_weak_residual: float
    condition_number: float
    source_flux: float
    relative_source_flux: float
    continuity_residual: float
    quadrature_barycentric: np.ndarray
    quadrature_weights: np.ndarray
    quadrature_face_indices: np.ndarray
    quadrature_points: np.ndarray
    quadrature_mu: np.ndarray
    quadrature_surface_gradient: np.ndarray
    quadrature_total_velocity: np.ndarray
    quadrature_interior_potential_residual: np.ndarray
    paired_quadrature_order: int | None
    paired_topology_counts: dict[str, int]

    def evaluate_potential(self, points: Any) -> np.ndarray:
        targets = _finite_points("points", points)
        source_potential = np.zeros(len(targets), dtype=float)
        for index, face in enumerate(self.mesh.faces):
            source_potential += constant_source_polygon_influence(
                self.mesh.vertices[face],
                targets,
                strength=float(self.source_strength[index]),
                on_surface_side="principal",
            ).potential
        return source_potential + surface_doublet_potential(
            self.surface,
            targets,
            quadrature_order=24,
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


def closed_p2_topology(mesh: ClosedTriangularMesh) -> ClosedP2Topology:
    if not isinstance(mesh, ClosedTriangularMesh):
        raise ThickBodyNeumannError(
            "mesh must be validated by closed_triangular_mesh"
        )
    edges: dict[tuple[int, int], int] = {}
    local = np.empty((len(mesh.faces), 6), dtype=int)
    vertex_count = len(mesh.vertices)
    for face_index, face in enumerate(mesh.faces):
        local[face_index, :3] = face
        for local_edge, (first, second) in enumerate(
            ((0, 1), (1, 2), (2, 0))
        ):
            edge = tuple(sorted((
                int(face[first]), int(face[second])
            )))
            if edge not in edges:
                edges[edge] = len(edges)
            local[face_index, 3 + local_edge] = (
                vertex_count + edges[edge]
            )
    ordered_edges = np.empty((len(edges), 2), dtype=int)
    for edge, index in edges.items():
        ordered_edges[index] = edge
    coordinates = np.concatenate(
        (
            mesh.vertices,
            0.5
            * (
                mesh.vertices[ordered_edges[:, 0]]
                + mesh.vertices[ordered_edges[:, 1]]
            ),
        ),
        axis=0,
    )
    return ClosedP2Topology(
        local_to_global=local,
        edge_vertices=ordered_edges,
        degree_of_freedom_coordinates=coordinates,
        vertex_dof_count=vertex_count,
        edge_dof_count=len(ordered_edges),
    )


def _surface_quadrature(
    mesh: ClosedTriangularMesh,
    order: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    barycentric, reference_weight = _triangle_quadrature(int(order))
    points = []
    weights = []
    owners = []
    local_barycentric = []
    for face_index, face in enumerate(mesh.faces):
        triangle = mesh.vertices[face]
        points.append(barycentric @ triangle)
        area_vector_norm = 2.0 * mesh.areas[face_index]
        weights.append(reference_weight * area_vector_norm)
        owners.append(np.full(len(barycentric), face_index, dtype=int))
        local_barycentric.append(barycentric)
    return (
        np.concatenate(points),
        np.concatenate(weights),
        np.concatenate(owners),
        np.concatenate(local_barycentric),
    )


def _global_shape_matrix(
    topology: ClosedP2Topology,
    owner: np.ndarray,
    barycentric: np.ndarray,
) -> np.ndarray:
    local_shape = p2_shape_values(barycentric)
    matrix = np.zeros((len(owner), topology.dof_count), dtype=float)
    for row, face_index in enumerate(owner):
        matrix[row, topology.local_to_global[face_index]] = (
            local_shape[row]
        )
    return matrix


def _element_basis_doublet_potential(
    vertices: np.ndarray,
    targets: np.ndarray,
    *,
    source_order: int,
) -> np.ndarray:
    barycentric, reference_weight = _triangle_quadrature(
        int(source_order)
    )
    source = barycentric @ vertices
    shape = p2_shape_values(barycentric)
    area_vector = np.cross(
        vertices[1] - vertices[0],
        vertices[2] - vertices[0],
    )
    normal = area_vector / np.linalg.norm(area_vector)
    physical_weight = reference_weight * np.linalg.norm(area_vector)
    separation = targets[:, None, :] - source[None, :, :]
    radius_square = np.einsum(
        "tqj,tqj->tq", separation, separation
    )
    if np.any(radius_square <= np.finfo(float).tiny):
        raise ThickBodyNeumannError(
            "source/target quadrature collision outside the owner PV block"
        )
    kernel = -(
        separation @ normal
    ) / (4.0 * np.pi * radius_square**1.5)
    return kernel @ (physical_weight[:, None] * shape)


@dataclass(frozen=True)
class PairedP2TriangleIntegral:
    """One target/source triangle-pair weak integral.

    ``doublet_block`` has target-test rows and source-trial columns.
    ``source_vector`` has target-test entries for a unit constant source on
    the source triangle.  ``partition_measure`` is the transformed integral
    of unity and must equal the product of the two physical triangle areas.
    """

    common_vertex_count: int
    doublet_block: np.ndarray
    source_vector: np.ndarray
    partition_measure: float


@lru_cache(maxsize=16)
def _unit_hypercube_quadrature(
    order: int,
) -> tuple[np.ndarray, np.ndarray]:
    coordinate, weight = np.polynomial.legendre.leggauss(int(order))
    coordinate = 0.5 * (coordinate + 1.0)
    weight = 0.5 * weight
    coordinate_grid = np.meshgrid(
        coordinate, coordinate, coordinate, coordinate, indexing="ij"
    )
    weight_grid = np.meshgrid(
        weight, weight, weight, weight, indexing="ij"
    )
    points = np.stack(
        [component.ravel() for component in coordinate_grid], axis=1
    )
    weights = np.prod(
        np.stack(weight_grid, axis=-1), axis=-1
    ).ravel()
    points.setflags(write=False)
    weights.setflags(write=False)
    return points, weights


def _pair_subregion_coordinates(
    common_vertex_count: int,
    region: int,
    hypercube_points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Exact Taylor-Duffy partition before radial analytic reduction.

    The triangle coordinates satisfy ``0<=xi2<=xi1<=1``.  Common-edge
    pairs use the six subregions in Reid, Johnson & White (2015),
    Appendix A; common-vertex pairs use their two subregions.  Retaining
    the shared radial coordinate as a quadrature variable is deliberate:
    the transformed kernel is regular and provides an independent oracle
    before any further analytic dimensional reduction.
    """
    w, y1, y2, y3 = hypercube_points.T
    if common_vertex_count >= 2:
        if region == 0:
            u1 = -w * y1
            u2 = -w * y1 * y2
            xi2 = w * (1.0 - y1 + y1 * y2)
            lower = xi2 + u2 - u1
            upper = np.ones_like(w)
        elif region == 1:
            u1 = w * y1
            u2 = w * y1 * y2
            xi2 = w * (1.0 - y1)
            lower = xi2
            upper = 1.0 - u1
        elif region == 2:
            u1 = -w * y1 * y2
            u2 = w * y1 * (1.0 - y2)
            xi2 = w * (1.0 - y1)
            lower = xi2 + u2 - u1
            upper = np.ones_like(w)
        elif region == 3:
            u1 = w * y1 * y2
            u2 = -w * y1 * (1.0 - y2)
            xi2 = w * (1.0 - y1 * y2)
            lower = xi2
            upper = 1.0 - u1
        elif region == 4:
            u1 = -w * y1 * y2
            u2 = -w * y1
            xi2 = w
            lower = xi2
            upper = np.ones_like(w)
        elif region == 5:
            u1 = w * y1 * y2
            u2 = w * y1
            xi2 = w * (1.0 - y1)
            lower = xi2 + u2 - u1
            upper = 1.0 - u1
        else:
            raise ThickBodyNeumannError(
                "common-edge pair region must be in [0,5]"
            )
        xi1 = lower + (upper - lower) * y3
        eta1 = u1 + xi1
        eta2 = u2 + xi2
        jacobian = w**2 * y1 * (upper - lower)
        return xi1, xi2, eta1, eta2, jacobian

    if common_vertex_count == 1:
        if region == 0:
            xi1 = w
            xi2 = w * y1
            eta1 = w * y2
            eta2 = w * y2 * y3
        elif region == 1:
            eta1 = w
            eta2 = w * y1
            xi1 = w * y2
            xi2 = w * y2 * y3
        else:
            raise ThickBodyNeumannError(
                "common-vertex pair region must be in [0,1]"
            )
        jacobian = w**3 * y2
        return xi1, xi2, eta1, eta2, jacobian
    raise ThickBodyNeumannError(
        "paired singular quadrature requires a shared vertex"
    )


def _canonical_pair_permutations(
    target_ids: np.ndarray,
    source_ids: np.ndarray,
) -> tuple[int, np.ndarray, np.ndarray]:
    shared = sorted(set(map(int, target_ids)) & set(map(int, source_ids)))
    count = len(shared)
    if count == 3:
        target_permutation = np.arange(3, dtype=int)
        source_permutation = np.asarray(
            [
                int(np.flatnonzero(source_ids == vertex)[0])
                for vertex in target_ids
            ],
            dtype=int,
        )
    elif count == 2:
        target_opposite = next(
            index
            for index, vertex in enumerate(target_ids)
            if int(vertex) not in shared
        )
        source_opposite = next(
            index
            for index, vertex in enumerate(source_ids)
            if int(vertex) not in shared
        )
        target_permutation = np.asarray(
            [
                int(np.flatnonzero(target_ids == vertex)[0])
                for vertex in shared
            ],
            dtype=int,
        )
        target_permutation = np.concatenate(
            (target_permutation, np.asarray([target_opposite], dtype=int))
        )
        source_permutation = np.asarray(
            [
                int(np.flatnonzero(source_ids == vertex)[0])
                for vertex in shared
            ],
            dtype=int,
        )
        source_permutation = np.concatenate(
            (source_permutation, np.asarray([source_opposite], dtype=int))
        )
    elif count == 1:
        shared_vertex = shared[0]
        target_shared = int(
            np.flatnonzero(target_ids == shared_vertex)[0]
        )
        source_shared = int(
            np.flatnonzero(source_ids == shared_vertex)[0]
        )
        target_permutation = np.asarray(
            [target_shared]
            + [index for index in range(3) if index != target_shared],
            dtype=int,
        )
        source_permutation = np.asarray(
            [source_shared]
            + [index for index in range(3) if index != source_shared],
            dtype=int,
        )
    else:
        raise ThickBodyNeumannError(
            "paired singular quadrature requires one or more shared vertices"
        )
    return count, target_permutation, source_permutation


def paired_p2_triangle_integral(
    target_vertices: Any,
    source_vertices: Any,
    *,
    target_vertex_ids: Any,
    source_vertex_ids: Any,
    quadrature_order: int = 8,
) -> PairedP2TriangleIntegral:
    """Integrate one intersecting P2 target/source pair as one domain."""
    target = _finite_points("target_vertices", target_vertices)
    source = _finite_points("source_vertices", source_vertices)
    if target.shape != (3, 3) or source.shape != (3, 3):
        raise ThickBodyNeumannError(
            "target_vertices and source_vertices must have shape (3,3)"
        )
    target_ids = np.asarray(target_vertex_ids)
    source_ids = np.asarray(source_vertex_ids)
    if (
        target_ids.shape != (3,)
        or source_ids.shape != (3,)
        or not np.issubdtype(target_ids.dtype, np.integer)
        or not np.issubdtype(source_ids.dtype, np.integer)
        or len(set(map(int, target_ids))) != 3
        or len(set(map(int, source_ids))) != 3
    ):
        raise ThickBodyNeumannError(
            "target/source vertex ids must each contain three unique integers"
        )
    if (
        not isinstance(quadrature_order, (int, np.integer))
        or quadrature_order < 2
    ):
        raise ThickBodyNeumannError(
            "quadrature_order must be an integer >=2"
        )
    count, target_permutation, source_permutation = (
        _canonical_pair_permutations(target_ids, source_ids)
    )
    target_canonical = target[target_permutation]
    source_canonical = source[source_permutation]
    target_area_vector = np.cross(
        target[1] - target[0], target[2] - target[0]
    )
    source_area_vector = np.cross(
        source[1] - source[0], source[2] - source[0]
    )
    target_double_area = float(np.linalg.norm(target_area_vector))
    source_double_area = float(np.linalg.norm(source_area_vector))
    scale = max(
        float(np.ptp(np.concatenate((target, source)), axis=0).max()),
        1.0,
    )
    if (
        target_double_area <= 64.0 * np.finfo(float).eps * scale**2
        or source_double_area <= 64.0 * np.finfo(float).eps * scale**2
    ):
        raise ThickBodyNeumannError("paired triangle is degenerate")
    source_normal = source_area_vector / source_double_area
    points, hypercube_weights = _unit_hypercube_quadrature(
        int(quadrature_order)
    )
    region_count = 6 if count >= 2 else 2
    doublet_block = np.zeros((6, 6), dtype=float)
    source_vector = np.zeros(6, dtype=float)
    partition_measure = 0.0
    physical_jacobian = target_double_area * source_double_area
    for region in range(region_count):
        xi1, xi2, eta1, eta2, transformed_jacobian = (
            _pair_subregion_coordinates(count, region, points)
        )
        target_barycentric_canonical = np.column_stack(
            (1.0 - xi1, xi1 - xi2, xi2)
        )
        source_barycentric_canonical = np.column_stack(
            (1.0 - eta1, eta1 - eta2, eta2)
        )
        target_barycentric = np.empty_like(
            target_barycentric_canonical
        )
        source_barycentric = np.empty_like(
            source_barycentric_canonical
        )
        target_barycentric[:, target_permutation] = (
            target_barycentric_canonical
        )
        source_barycentric[:, source_permutation] = (
            source_barycentric_canonical
        )
        target_shape = p2_shape_values(target_barycentric)
        source_shape = p2_shape_values(source_barycentric)
        target_points = (
            target_canonical[0]
            + xi1[:, None]
            * (target_canonical[1] - target_canonical[0])
            + xi2[:, None]
            * (target_canonical[2] - target_canonical[1])
        )
        source_points = (
            source_canonical[0]
            + eta1[:, None]
            * (source_canonical[1] - source_canonical[0])
            + eta2[:, None]
            * (source_canonical[2] - source_canonical[1])
        )
        separation = target_points - source_points
        radius_square = np.einsum(
            "ij,ij->i", separation, separation
        )
        if np.any(radius_square <= np.finfo(float).tiny * scale**2):
            raise ThickBodyNeumannError(
                "paired transformation produced a source/target collision"
            )
        radius = np.sqrt(radius_square)
        physical_weight = (
            hypercube_weights
            * transformed_jacobian
            * physical_jacobian
        )
        source_kernel = 1.0 / (4.0 * np.pi * radius)
        if count == 3:
            doublet_kernel = np.zeros_like(source_kernel)
        else:
            doublet_kernel = -(
                separation @ source_normal
            ) / (4.0 * np.pi * radius_square * radius)
        doublet_block += np.einsum(
            "qi,q,qj->ij",
            target_shape,
            physical_weight * doublet_kernel,
            source_shape,
        )
        source_vector += np.einsum(
            "qi,q->i",
            target_shape,
            physical_weight * source_kernel,
        )
        partition_measure += float(np.dot(
            hypercube_weights,
            transformed_jacobian * physical_jacobian,
        ))
    if (
        not np.all(np.isfinite(doublet_block))
        or not np.all(np.isfinite(source_vector))
        or not np.isfinite(partition_measure)
    ):
        raise ThickBodyNeumannError(
            "paired singular integral contains non-finite values"
        )
    return PairedP2TriangleIntegral(
        common_vertex_count=count,
        doublet_block=doublet_block,
        source_vector=source_vector,
        partition_measure=partition_measure,
    )


def element_basis_doublet_potential_line_reduced(
    vertices: Any,
    targets: Any,
    *,
    line_quadrature_order: int = 16,
) -> np.ndarray:
    """Evaluate all six P2 potential bases by exact radial integration.

    The target is projected into the source plane.  Each oriented
    projection-to-edge fan uses exact radial primitives for the constant,
    linear and quadratic density terms; only the smooth edge coordinate is
    integrated numerically.  Coplanar values are the principal value zero.
    """
    triangle = _finite_points("vertices", vertices)
    points = _finite_points("targets", targets)
    if triangle.shape != (3, 3):
        raise ThickBodyNeumannError(
            "vertices must have shape (3,3)"
        )
    if (
        not isinstance(line_quadrature_order, (int, np.integer))
        or line_quadrature_order < 2
    ):
        raise ThickBodyNeumannError(
            "line_quadrature_order must be an integer >=2"
        )
    edge01 = triangle[1] - triangle[0]
    edge02 = triangle[2] - triangle[0]
    area_vector = np.cross(edge01, edge02)
    area_norm = float(np.linalg.norm(area_vector))
    scale = max(
        float(np.linalg.norm(edge01)),
        float(np.linalg.norm(edge02)),
        float(np.linalg.norm(triangle[2] - triangle[1])),
        1.0,
    )
    if area_norm <= 64.0 * np.finfo(float).eps * scale**2:
        raise ThickBodyNeumannError("triangle is degenerate")
    normal = area_vector / area_norm
    height = (points - triangle[0]) @ normal
    projected = points - height[:, None] * normal

    gradients = np.stack(
        (
            np.cross(area_vector, triangle[2] - triangle[1])
            / area_norm**2,
            np.cross(area_vector, triangle[0] - triangle[2])
            / area_norm**2,
            np.cross(area_vector, triangle[1] - triangle[0])
            / area_norm**2,
        )
    )
    delta = projected - triangle[0]
    projected_barycentric = np.column_stack(
        (
            1.0 - delta @ gradients[1] - delta @ gradients[2],
            delta @ gradients[1],
            delta @ gradients[2],
        )
    )
    shape_at_projection = p2_shape_values(projected_barycentric)
    l0, l1, l2 = projected_barycentric.T
    g0, g1, g2 = gradients
    shape_gradient = np.empty((len(points), 6, 3), dtype=float)
    shape_gradient[:, 0] = (4.0 * l0 - 1.0)[:, None] * g0
    shape_gradient[:, 1] = (4.0 * l1 - 1.0)[:, None] * g1
    shape_gradient[:, 2] = (4.0 * l2 - 1.0)[:, None] * g2
    shape_gradient[:, 3] = 4.0 * (
        l0[:, None] * g1 + l1[:, None] * g0
    )
    shape_gradient[:, 4] = 4.0 * (
        l1[:, None] * g2 + l2[:, None] * g1
    )
    shape_gradient[:, 5] = 4.0 * (
        l2[:, None] * g0 + l0[:, None] * g2
    )

    coordinate, weight = np.polynomial.legendre.leggauss(
        int(line_quadrature_order)
    )
    coordinate = 0.5 * (coordinate + 1.0)
    weight = 0.5 * weight
    result = np.zeros((len(points), 6), dtype=float)
    absolute_height = np.abs(height)
    safe_height = np.maximum(
        absolute_height,
        np.finfo(float).tiny * scale,
    )
    sign_height = np.sign(height)
    for first, second in ((0, 1), (1, 2), (2, 0)):
        start = triangle[first]
        end = triangle[second]
        edge_vector = end - start
        source = (
            (1.0 - coordinate)[:, None] * start
            + coordinate[:, None] * end
        )
        source_barycentric = np.zeros((len(coordinate), 3))
        source_barycentric[:, first] = 1.0 - coordinate
        source_barycentric[:, second] = coordinate
        edge_shape = p2_shape_values(source_barycentric)
        radial = source[None, :, :] - projected[:, None, :]
        radius = np.linalg.norm(radial, axis=2)
        radius_floor = 128.0 * np.finfo(float).eps * scale
        if np.any(radius <= radius_floor):
            raise ThickBodyNeumannError(
                "projected target lies on a source panel edge"
            )
        direction = radial / radius[:, :, None]
        signed_angle_jacobian = (
            np.cross(radial, edge_vector) @ normal
        ) / radius**2
        linear_coefficient = np.einsum(
            "tij,tqj->tqi", shape_gradient, direction
        )
        quadratic_coefficient = (
            edge_shape[None, :, :]
            - shape_at_projection[:, None, :]
            - linear_coefficient * radius[:, :, None]
        ) / radius[:, :, None] ** 2
        distance = np.sqrt(
            radius**2 + absolute_height[:, None] ** 2
        )
        height_j0 = sign_height[:, None] * (
            1.0 - absolute_height[:, None] / distance
        )
        height_j1 = height[:, None] * (
            np.arcsinh(radius / safe_height[:, None])
            - radius / distance
        )
        height_j2 = height[:, None] * (
            distance
            + absolute_height[:, None] ** 2 / distance
            - 2.0 * absolute_height[:, None]
        )
        radial_integral = (
            shape_at_projection[:, None, :]
            * height_j0[:, :, None]
            + linear_coefficient * height_j1[:, :, None]
            + quadratic_coefficient * height_j2[:, :, None]
        )
        result -= np.einsum(
            "tqi,tq,q->ti",
            radial_integral,
            signed_angle_jacobian,
            weight,
        ) / (4.0 * np.pi)
    if not np.all(np.isfinite(result)):
        raise ThickBodyNeumannError(
            "line-reduced doublet potential contains non-finite values"
        )
    return result


def solve_actual_boundary_p2_galerkin(
    mesh: ClosedTriangularMesh,
    *,
    incident_velocity: Any,
    wall_velocity: Any | None = None,
    target_quadrature_order: int = 16,
    source_quadrature_order: int = 16,
    potential_operator: str = "tensor_duffy",
) -> ActualBoundaryP2GalerkinSolution:
    """Solve the continuous-P2 weak interior potential equation."""
    if not isinstance(mesh, ClosedTriangularMesh):
        raise ThickBodyNeumannError(
            "mesh must be validated by closed_triangular_mesh"
        )
    incident = _finite_points("incident_velocity", incident_velocity)
    if incident.shape != mesh.centroids.shape:
        raise ThickBodyNeumannError(
            "incident_velocity must match face centroids"
        )
    if wall_velocity is None:
        wall = np.zeros_like(incident)
    else:
        wall = _finite_points("wall_velocity", wall_velocity)
        if wall.shape != mesh.centroids.shape:
            raise ThickBodyNeumannError(
                "wall_velocity must match face centroids"
            )
    if (
        not isinstance(target_quadrature_order, (int, np.integer))
        or not isinstance(source_quadrature_order, (int, np.integer))
        or target_quadrature_order < 2
        or source_quadrature_order < 2
    ):
        raise ThickBodyNeumannError(
            "quadrature orders must be integers >=2"
        )
    if potential_operator not in {
        "tensor_duffy", "line_reduced", "paired_singular"
    }:
        raise ThickBodyNeumannError(
            "potential_operator must be tensor_duffy, line_reduced, "
            "or paired_singular"
        )
    paired = potential_operator == "paired_singular"
    topology = closed_p2_topology(mesh)
    points, weights, owner, barycentric = _surface_quadrature(
        mesh, int(target_quadrature_order)
    )
    test = _global_shape_matrix(topology, owner, barycentric)
    face_sets = [set(map(int, face)) for face in mesh.faces]
    common_vertex_count = np.zeros(
        (len(mesh.faces), len(mesh.faces)), dtype=np.int8
    )
    if paired:
        for target_face in range(len(mesh.faces)):
            for source_face in range(len(mesh.faces)):
                common_vertex_count[target_face, source_face] = len(
                    face_sets[target_face] & face_sets[source_face]
                )
    doublet_pv = np.zeros_like(test)
    for source_face, face in enumerate(mesh.faces):
        if paired:
            integration_rows = (
                common_vertex_count[owner, source_face] == 0
            )
        else:
            integration_rows = owner != source_face
        local_influence = np.zeros((len(points), 6), dtype=float)
        if potential_operator in {"tensor_duffy", "paired_singular"}:
            local_influence[integration_rows] = (
                _element_basis_doublet_potential(
                    mesh.vertices[face],
                    points[integration_rows],
                    source_order=int(source_quadrature_order),
                )
            )
        else:
            local_influence[integration_rows] = (
                element_basis_doublet_potential_line_reduced(
                    mesh.vertices[face],
                    points[integration_rows],
                    line_quadrature_order=int(source_quadrature_order),
                )
            )
        doublet_pv[
            :, topology.local_to_global[source_face]
        ] += local_influence

    source_strength = np.einsum(
        "ij,ij->i", incident - wall, mesh.normals
    )
    source_potential = np.zeros(len(points), dtype=float)
    weak_source_potential = np.zeros(len(points), dtype=float)
    for source_index, face in enumerate(mesh.faces):
        contribution = constant_source_polygon_influence(
            mesh.vertices[face],
            points,
            strength=float(source_strength[source_index]),
            on_surface_side="principal",
        ).potential
        source_potential += contribution
        if paired:
            disjoint_rows = (
                common_vertex_count[owner, source_index] == 0
            )
            weak_source_potential[disjoint_rows] += (
                contribution[disjoint_rows]
            )
        else:
            weak_source_potential += contribution

    weighted_test = test.T * weights[None, :]
    interior_operator = 0.5 * test + doublet_pv
    matrix = weighted_test @ interior_operator
    weak_source_integral = weighted_test @ weak_source_potential
    paired_topology_counts = {
        "common_triangle": 0,
        "common_edge": 0,
        "common_vertex": 0,
    }
    if paired:
        for target_face, target_ids in enumerate(mesh.faces):
            target_global = topology.local_to_global[target_face]
            for source_face, source_ids in enumerate(mesh.faces):
                count = int(
                    common_vertex_count[target_face, source_face]
                )
                if count == 0:
                    continue
                pair = paired_p2_triangle_integral(
                    mesh.vertices[target_ids],
                    mesh.vertices[source_ids],
                    target_vertex_ids=target_ids,
                    source_vertex_ids=source_ids,
                    quadrature_order=int(source_quadrature_order),
                )
                source_global = topology.local_to_global[source_face]
                matrix[np.ix_(target_global, source_global)] += (
                    pair.doublet_block
                )
                weak_source_integral[target_global] += (
                    float(source_strength[source_face])
                    * pair.source_vector
                )
                label = {
                    3: "common_triangle",
                    2: "common_edge",
                    1: "common_vertex",
                }[count]
                paired_topology_counts[label] += 1
    right_hand_side = -weak_source_integral
    condition_number = float(np.linalg.cond(matrix))
    if not np.isfinite(condition_number):
        raise ThickBodyNeumannError(
            "continuous-P2 Galerkin matrix is singular or non-finite"
        )
    try:
        global_mu = np.linalg.solve(matrix, right_hand_side)
    except np.linalg.LinAlgError as error:
        raise ThickBodyNeumannError(
            "continuous-P2 Galerkin solve failed"
        ) from error
    weak_residual = matrix @ global_mu - right_hand_side
    weak_scale = max(
        float(np.linalg.norm(right_hand_side)),
        np.finfo(float).tiny,
    )
    pointwise_operator = interior_operator
    if paired:
        pointwise_doublet = doublet_pv.copy()
        for source_face, face in enumerate(mesh.faces):
            adjacent_rows = (
                (common_vertex_count[owner, source_face] > 0)
                & (owner != source_face)
            )
            local_influence = np.zeros((len(points), 6), dtype=float)
            local_influence[adjacent_rows] = (
                _element_basis_doublet_potential(
                    mesh.vertices[face],
                    points[adjacent_rows],
                    source_order=int(source_quadrature_order),
                )
            )
            pointwise_doublet[
                :, topology.local_to_global[source_face]
            ] += local_influence
        pointwise_operator = 0.5 * test + pointwise_doublet
    pointwise_residual = (
        source_potential + pointwise_operator @ global_mu
    )
    face_mu = global_mu[topology.local_to_global]
    surface = QuadraticDoubletSurface(
        mesh.vertices, mesh.faces, face_mu
    )
    continuity = surface.continuity_report(tolerance=0.0)
    quadrature_mu = test @ global_mu
    gradient = np.empty_like(points)
    total_velocity = np.empty_like(points)
    for face_index in range(len(mesh.faces)):
        rows = owner == face_index
        element = surface.element(face_index)
        gradient[rows] = element.surface_gradient_barycentric(
            barycentric[rows]
        )
        normal = mesh.normals[face_index]
        relative_incident = incident[face_index]
        tangent_incident = (
            relative_incident
            - np.dot(relative_incident, normal) * normal
        )
        wall_normal = np.dot(wall[face_index], normal) * normal
        total_velocity[rows] = (
            tangent_incident[None, :]
            - gradient[rows]
            + wall_normal[None, :]
        )
    source_flux = float(np.dot(source_strength, mesh.areas))
    source_flux_scale = max(
        float(np.dot(np.abs(source_strength), mesh.areas)),
        np.finfo(float).tiny,
    )
    return ActualBoundaryP2GalerkinSolution(
        mesh=mesh,
        topology=topology,
        incident_velocity=incident.copy(),
        wall_velocity=wall.copy(),
        source_strength=source_strength,
        global_doublet_strength=global_mu,
        surface=surface,
        matrix=matrix,
        right_hand_side=right_hand_side,
        weak_residual=weak_residual,
        relative_weak_residual=float(
            np.linalg.norm(weak_residual) / weak_scale
        ),
        condition_number=condition_number,
        source_flux=source_flux,
        relative_source_flux=abs(source_flux) / source_flux_scale,
        continuity_residual=max(
            continuity.max_trace_node_jump,
            continuity.max_trace_jump,
        ),
        quadrature_barycentric=barycentric,
        quadrature_weights=weights,
        quadrature_face_indices=owner,
        quadrature_points=points,
        quadrature_mu=quadrature_mu,
        quadrature_surface_gradient=gradient,
        quadrature_total_velocity=total_velocity,
        quadrature_interior_potential_residual=pointwise_residual,
        paired_quadrature_order=(
            int(source_quadrature_order) if paired else None
        ),
        paired_topology_counts=paired_topology_counts,
    )
