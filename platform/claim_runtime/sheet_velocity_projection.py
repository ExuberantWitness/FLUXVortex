"""No-force projection from strict-interior sheet velocities to P1 geometry.

The DDE field is evaluated at Krebs' four strict-interior points per face.
Wake geometry is a gapless set of planar (P1) triangles shared at vertices.
This module exposes the missing discrete map as an overdetermined continuous-
P1 geometry-velocity projection.  It does not choose a time integrator, add a
vortex core, or book force.

The projected velocity is explicitly the sheet-average (Birkhoff--Rott)
gauge.  A different tangential reparameterization is a different claim and
must not be introduced through this operator.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .distributed_doublet import (
    DistributedDoubletError,
    QuadraticDoubletAssembly,
    QuadraticDoubletSurface,
)


@dataclass(frozen=True)
class SheetVelocityProjectionReport:
    samples: int
    degrees_of_freedom: int
    rank: int
    full_rank: bool
    condition_number: float
    max_abs_residual: float
    max_rel_residual: float
    max_input_norm: float
    max_abs_residual_fraction: float
    gauge: str


@dataclass(frozen=True)
class ProjectedSheetVelocity:
    dof_positions: np.ndarray
    dof_velocity: np.ndarray
    face_dofs: np.ndarray
    vertex_count: int
    report: SheetVelocityProjectionReport

    @property
    def vertex_velocity(self) -> np.ndarray:
        return self.dof_velocity[: self.vertex_count].copy()

    def evaluate_face(self, face_index: int, barycentric) -> np.ndarray:
        if face_index < 0 or face_index >= len(self.face_dofs):
            raise DistributedDoubletError("invalid projection face index")
        basis = np.asarray(barycentric, dtype=float)
        if (
            basis.ndim != 2
            or basis.shape[1] != 3
            or not np.all(np.isfinite(basis))
        ):
            raise DistributedDoubletError(
                "barycentric must be a finite array with shape (n,3)"
            )
        return basis @ self.dof_velocity[self.face_dofs[face_index]]


@dataclass(frozen=True)
class ProjectedAssemblySheetVelocity:
    dof_positions: np.ndarray
    dof_velocity: np.ndarray
    patch_face_dofs: tuple[np.ndarray, ...]
    patch_vertex_dofs: tuple[np.ndarray, ...]
    report: SheetVelocityProjectionReport

    def vertex_velocity(self, patch_index: int) -> np.ndarray:
        if patch_index < 0 or patch_index >= len(self.patch_vertex_dofs):
            raise DistributedDoubletError("invalid projection patch index")
        return self.dof_velocity[self.patch_vertex_dofs[patch_index]].copy()

    def evaluate_face(
        self,
        patch_index: int,
        face_index: int,
        barycentric,
    ) -> np.ndarray:
        if patch_index < 0 or patch_index >= len(self.patch_face_dofs):
            raise DistributedDoubletError("invalid projection patch index")
        faces = self.patch_face_dofs[patch_index]
        if face_index < 0 or face_index >= len(faces):
            raise DistributedDoubletError("invalid projection face index")
        basis = np.asarray(barycentric, dtype=float)
        if (
            basis.ndim != 2
            or basis.shape[1] != 3
            or not np.all(np.isfinite(basis))
        ):
            raise DistributedDoubletError(
                "barycentric must be a finite array with shape (n,3)"
            )
        return basis @ self.dof_velocity[faces[face_index]]


@dataclass(frozen=True)
class ProjectedAssemblyNormalGeometryVelocity:
    dof_positions: np.ndarray
    dof_normal_speed: np.ndarray
    dof_normals: np.ndarray
    patch_vertex_dofs: tuple[np.ndarray, ...]
    report: SheetVelocityProjectionReport

    @property
    def dof_velocity(self) -> np.ndarray:
        return self.dof_normal_speed[:, None] * self.dof_normals

    def vertex_velocity(self, patch_index: int) -> np.ndarray:
        if patch_index < 0 or patch_index >= len(self.patch_vertex_dofs):
            raise DistributedDoubletError("invalid projection patch index")
        return self.dof_velocity[self.patch_vertex_dofs[patch_index]].copy()


@dataclass(frozen=True)
class ProjectedNormalGeometryVelocity:
    vertex_normal_speed: np.ndarray
    vertex_normals: np.ndarray
    report: SheetVelocityProjectionReport

    @property
    def vertex_velocity(self) -> np.ndarray:
        return self.vertex_normal_speed[:, None] * self.vertex_normals


def _global_p1_topology(
    surface: QuadraticDoubletSurface,
) -> tuple[np.ndarray, np.ndarray]:
    return surface.faces.copy(), surface.vertices.copy()


def project_sheet_average_velocity(
    surface: QuadraticDoubletSurface,
    collocation_velocity,
    *,
    relative_rank_tolerance: float | None = None,
) -> ProjectedSheetVelocity:
    """Project 4-per-face velocities onto the continuous P1 geometry space."""
    if not isinstance(surface, QuadraticDoubletSurface):
        raise DistributedDoubletError(
            "surface must be QuadraticDoubletSurface"
        )
    velocity = np.asarray(collocation_velocity, dtype=float)
    expected_shape = (4 * len(surface), 3)
    if velocity.shape != expected_shape:
        raise DistributedDoubletError(
            "collocation_velocity must have shape "
            f"{expected_shape}, got {velocity.shape}"
        )
    if not np.all(np.isfinite(velocity)):
        raise DistributedDoubletError(
            "collocation_velocity contains non-finite values"
        )
    face_dofs, dof_positions = _global_p1_topology(surface)
    centroid = np.full(3, 1.0 / 3.0)
    barycentric = [centroid]
    for vertex_index in range(3):
        point = 0.1 * centroid
        point = point.copy()
        point[vertex_index] += 0.9
        barycentric.append(point)
    local_basis = np.asarray(barycentric)
    matrix = np.zeros(
        (4 * len(surface), len(dof_positions)),
        dtype=float,
    )
    for face_index, dofs in enumerate(face_dofs):
        matrix[4 * face_index : 4 * face_index + 4, dofs] = local_basis

    coefficients, report = _solve_projection(
        matrix,
        velocity,
        relative_rank_tolerance=relative_rank_tolerance,
    )
    return ProjectedSheetVelocity(
        dof_positions=dof_positions,
        dof_velocity=coefficients,
        face_dofs=face_dofs,
        vertex_count=len(surface.vertices),
        report=report,
    )


def project_sheet_normal_geometry_velocity(
    surface: QuadraticDoubletSurface,
    collocation_velocity,
    *,
    relative_rank_tolerance: float | None = None,
) -> ProjectedNormalGeometryVelocity:
    """Project only the physically continuous normal geometry velocity."""
    if not isinstance(surface, QuadraticDoubletSurface):
        raise DistributedDoubletError(
            "surface must be QuadraticDoubletSurface"
        )
    velocity = np.asarray(collocation_velocity, dtype=float)
    expected_shape = (4 * len(surface), 3)
    if velocity.shape != expected_shape:
        raise DistributedDoubletError(
            "collocation_velocity must have shape "
            f"{expected_shape}, got {velocity.shape}"
        )
    if not np.all(np.isfinite(velocity)):
        raise DistributedDoubletError(
            "collocation_velocity contains non-finite values"
        )
    face_dofs, dof_positions = _global_p1_topology(surface)
    centroid = np.full(3, 1.0 / 3.0)
    barycentric = [centroid]
    for vertex_index in range(3):
        point = 0.1 * centroid
        point = point.copy()
        point[vertex_index] += 0.9
        barycentric.append(point)
    local_basis = np.asarray(barycentric)
    matrix = np.zeros(
        (4 * len(surface), len(dof_positions)),
        dtype=float,
    )
    normal_speed = np.empty((4 * len(surface), 1), dtype=float)
    vertex_normal_sum = np.zeros_like(surface.vertices)
    for face_index, dofs in enumerate(face_dofs):
        rows = slice(4 * face_index, 4 * face_index + 4)
        matrix[rows, dofs] = local_basis
        element = surface.element(face_index)
        normal_speed[rows, 0] = (
            velocity[rows] @ element.normal
        )
        for vertex_index in surface.faces[face_index]:
            vertex_normal_sum[vertex_index] += element.area_vector
    normal_norm = np.linalg.norm(vertex_normal_sum, axis=1)
    if np.any(normal_norm <= np.finfo(float).eps):
        raise DistributedDoubletError(
            "vertex normal is undefined; check face orientation/topology"
        )
    vertex_normals = vertex_normal_sum / normal_norm[:, None]
    coefficients, report = _solve_projection(
        matrix,
        normal_speed,
        relative_rank_tolerance=relative_rank_tolerance,
    )
    return ProjectedNormalGeometryVelocity(
        vertex_normal_speed=coefficients[:, 0],
        vertex_normals=vertex_normals,
        report=report,
    )


def project_vertex_star_normal_geometry_velocity(
    surface: QuadraticDoubletSurface,
    collocation_velocity,
    *,
    relative_rank_tolerance: float | None = None,
) -> ProjectedNormalGeometryVelocity:
    """Extrapolate each vertex normal speed from its strict-interior star.

    Every incident face contributes all four Krebs points.  A local affine
    scalar field in the vertex tangent plane is solved without weighting,
    radius, ridge, or core; only its intercept is used as the vertex speed.
    """
    if not isinstance(surface, QuadraticDoubletSurface):
        raise DistributedDoubletError(
            "surface must be QuadraticDoubletSurface"
        )
    velocity = np.asarray(collocation_velocity, dtype=float)
    expected_shape = (4 * len(surface), 3)
    if velocity.shape != expected_shape:
        raise DistributedDoubletError(
            "collocation_velocity must have shape "
            f"{expected_shape}, got {velocity.shape}"
        )
    if not np.all(np.isfinite(velocity)):
        raise DistributedDoubletError(
            "collocation_velocity contains non-finite values"
        )
    points, _, _ = surface.interior_collocation_points()
    vertex_normal_sum = np.zeros_like(surface.vertices)
    incident_faces: list[list[int]] = [
        [] for _ in surface.vertices
    ]
    for face_index, face in enumerate(surface.faces):
        element = surface.element(face_index)
        for vertex_index in face:
            vertex_normal_sum[vertex_index] += element.area_vector
            incident_faces[vertex_index].append(face_index)
    normal_norm = np.linalg.norm(vertex_normal_sum, axis=1)
    if np.any(normal_norm <= np.finfo(float).eps):
        raise DistributedDoubletError(
            "vertex normal is undefined; check face orientation/topology"
        )
    vertex_normals = vertex_normal_sum / normal_norm[:, None]
    vertex_speed = np.empty(len(surface.vertices), dtype=float)
    max_condition = 0.0
    max_abs_residual = 0.0
    max_rel_residual = 0.0
    max_input_norm = 0.0
    total_samples = 0
    if relative_rank_tolerance is None:
        relative_rank_tolerance = (
            64.0
            * np.finfo(float).eps
            * max(4 * len(surface), 3)
        )
    if (
        relative_rank_tolerance < 0.0
        or not np.isfinite(relative_rank_tolerance)
    ):
        raise DistributedDoubletError(
            "relative_rank_tolerance must be finite and non-negative"
        )

    for vertex_index, faces in enumerate(incident_faces):
        if not faces:
            raise DistributedDoubletError(
                f"vertex {vertex_index} has no incident face"
            )
        neighbor_indices = sorted(
            {
                int(other)
                for face_index in faces
                for other in surface.faces[face_index]
                if int(other) != vertex_index
            }
        )
        normal = vertex_normals[vertex_index]
        tangent1 = None
        for neighbor in neighbor_indices:
            candidate = (
                surface.vertices[neighbor]
                - surface.vertices[vertex_index]
            )
            candidate -= np.dot(candidate, normal) * normal
            norm = np.linalg.norm(candidate)
            if norm > 64.0 * np.finfo(float).eps:
                tangent1 = candidate / norm
                break
        if tangent1 is None:
            raise DistributedDoubletError(
                f"vertex {vertex_index} has no tangent direction"
            )
        tangent2 = np.cross(normal, tangent1)
        rows = np.concatenate(
            [
                np.arange(4 * face_index, 4 * face_index + 4)
                for face_index in faces
            ]
        )
        delta = points[rows] - surface.vertices[vertex_index]
        design = np.column_stack(
            (
                np.ones(len(rows)),
                delta @ tangent1,
                delta @ tangent2,
            )
        )
        sample_speed = np.empty(len(rows), dtype=float)
        offset = 0
        for face_index in faces:
            face_rows = slice(offset, offset + 4)
            sample_speed[face_rows] = (
                velocity[
                    4 * face_index : 4 * face_index + 4
                ]
                @ surface.element(face_index).normal
            )
            offset += 4
        singular_values = np.linalg.svd(
            design,
            compute_uv=False,
            full_matrices=False,
        )
        threshold = relative_rank_tolerance * singular_values[0]
        rank = int(np.count_nonzero(singular_values > threshold))
        if rank != 3:
            raise DistributedDoubletError(
                "vertex-star normal extrapolation is rank deficient: "
                f"vertex={vertex_index}, rank={rank}"
            )
        coefficients, _, solved_rank, solved_singular = np.linalg.lstsq(
            design,
            sample_speed,
            rcond=relative_rank_tolerance,
        )
        if int(solved_rank) != 3:
            raise DistributedDoubletError(
                "vertex-star normal extrapolation lost rank"
            )
        reconstructed = design @ coefficients
        residual = np.abs(reconstructed - sample_speed)
        scale = np.maximum(
            np.abs(sample_speed),
            np.finfo(float).eps,
        )
        vertex_speed[vertex_index] = coefficients[0]
        max_condition = max(
            max_condition,
            float(solved_singular[0] / solved_singular[-1]),
        )
        max_abs_residual = max(
            max_abs_residual,
            float(np.max(residual, initial=0.0)),
        )
        max_rel_residual = max(
            max_rel_residual,
            float(np.max(residual / scale, initial=0.0)),
        )
        max_input_norm = max(
            max_input_norm,
            float(np.max(np.abs(sample_speed), initial=0.0)),
        )
        total_samples += len(rows)

    report = SheetVelocityProjectionReport(
        samples=total_samples,
        degrees_of_freedom=len(surface.vertices),
        rank=len(surface.vertices),
        full_rank=True,
        condition_number=max_condition,
        max_abs_residual=max_abs_residual,
        max_rel_residual=max_rel_residual,
        max_input_norm=max_input_norm,
        max_abs_residual_fraction=(
            max_abs_residual
            / max(max_input_norm, np.finfo(float).eps)
        ),
        gauge="sheet_average_normal_vertex_star",
    )
    return ProjectedNormalGeometryVelocity(
        vertex_normal_speed=vertex_speed,
        vertex_normals=vertex_normals,
        report=report,
    )


def _solve_projection(
    matrix: np.ndarray,
    velocity: np.ndarray,
    *,
    relative_rank_tolerance: float | None,
) -> tuple[np.ndarray, SheetVelocityProjectionReport]:
    singular_values = np.linalg.svd(
        matrix,
        compute_uv=False,
        full_matrices=False,
    )
    if relative_rank_tolerance is None:
        relative_rank_tolerance = (
            64.0
            * np.finfo(float).eps
            * max(matrix.shape)
        )
    if (
        relative_rank_tolerance < 0.0
        or not np.isfinite(relative_rank_tolerance)
    ):
        raise DistributedDoubletError(
            "relative_rank_tolerance must be finite and non-negative"
        )
    threshold = (
        relative_rank_tolerance * singular_values[0]
        if len(singular_values)
        else 0.0
    )
    rank = int(np.count_nonzero(singular_values > threshold))
    dof_count = matrix.shape[1]
    if rank != dof_count:
        raise DistributedDoubletError(
            "continuous-P1 geometry-velocity projection is rank deficient: "
            f"rank={rank}, dofs={dof_count}"
        )
    coefficients, _, solved_rank, solved_singular = np.linalg.lstsq(
        matrix,
        velocity,
        rcond=relative_rank_tolerance,
    )
    if int(solved_rank) != dof_count:
        raise DistributedDoubletError(
            "continuous-P1 geometry-velocity projection lost rank during solve"
        )
    reconstructed = matrix @ coefficients
    residual = np.linalg.norm(reconstructed - velocity, axis=1)
    scale = np.maximum(
        np.linalg.norm(velocity, axis=1),
        np.finfo(float).eps,
    )
    condition_number = float(
        solved_singular[0] / solved_singular[-1]
    )
    report = SheetVelocityProjectionReport(
        samples=len(matrix),
        degrees_of_freedom=dof_count,
        rank=rank,
        full_rank=True,
        condition_number=condition_number,
        max_abs_residual=float(np.max(residual, initial=0.0)),
        max_rel_residual=float(
            np.max(residual / scale, initial=0.0)
        ),
        max_input_norm=float(
            np.max(np.linalg.norm(velocity, axis=1), initial=0.0)
        ),
        max_abs_residual_fraction=float(
            np.max(residual, initial=0.0)
            / max(
                float(
                    np.max(
                        np.linalg.norm(velocity, axis=1),
                        initial=0.0,
                    )
                ),
                np.finfo(float).eps,
            )
        ),
        gauge="sheet_average",
    )
    return coefficients, report


def _assembly_p1_topology(
    assembly: QuadraticDoubletAssembly,
    *,
    geometry_tolerance: float,
) -> tuple[
    tuple[np.ndarray, ...],
    tuple[np.ndarray, ...],
    np.ndarray,
]:
    topology = assembly.topology_report(
        geometry_tolerance=geometry_tolerance
    )
    if not topology.compatible:
        raise DistributedDoubletError(
            f"assembly topology is incompatible: {topology}"
        )
    offsets = []
    total_vertices = 0
    for patch in assembly.patches:
        offsets.append(total_vertices)
        total_vertices += len(patch.surface.vertices)
    parent = np.arange(total_vertices, dtype=np.int64)

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = int(parent[index])
        return index

    def union(first: int, second: int) -> None:
        root_first = find(first)
        root_second = find(second)
        if root_first != root_second:
            parent[root_second] = root_first

    interfaces: dict[str, list[tuple[int, tuple[int, int]]]] = {}
    for patch_index, patch in enumerate(assembly.patches):
        for edge, role in patch.boundary_roles.items():
            if role.startswith("interface:"):
                interfaces.setdefault(
                    role.removeprefix("interface:"),
                    [],
                ).append((patch_index, edge))
    for interface_id, records in interfaces.items():
        if len(records) != 2:
            raise DistributedDoubletError(
                f"interface {interface_id!r} does not have two sides"
            )
        (first_patch, first_edge), (second_patch, second_edge) = records
        first_vertices = assembly.patches[
            first_patch
        ].surface.vertices[list(first_edge)]
        second_vertices = assembly.patches[
            second_patch
        ].surface.vertices[list(second_edge)]
        direct_gap = float(
            np.max(
                np.linalg.norm(
                    first_vertices - second_vertices,
                    axis=1,
                )
            )
        )
        reverse_gap = float(
            np.max(
                np.linalg.norm(
                    first_vertices - second_vertices[::-1],
                    axis=1,
                )
            )
        )
        if min(direct_gap, reverse_gap) > geometry_tolerance:
            raise DistributedDoubletError(
                f"interface {interface_id!r} cannot weld P1 vertices"
            )
        second_order = (
            second_edge
            if direct_gap <= reverse_gap
            else second_edge[::-1]
        )
        for first_local, second_local in zip(
            first_edge,
            second_order,
        ):
            union(
                offsets[first_patch] + first_local,
                offsets[second_patch] + second_local,
            )

    roots = [find(index) for index in range(total_vertices)]
    unique_roots = sorted(set(roots))
    root_to_dof = {
        root: dof for dof, root in enumerate(unique_roots)
    }
    patch_vertex_dofs = []
    patch_face_dofs = []
    dof_members: list[list[np.ndarray]] = [
        [] for _ in unique_roots
    ]
    for patch_index, patch in enumerate(assembly.patches):
        local_dofs = np.array(
            [
                root_to_dof[roots[offsets[patch_index] + local_index]]
                for local_index in range(len(patch.surface.vertices))
            ],
            dtype=np.int64,
        )
        patch_vertex_dofs.append(local_dofs)
        patch_face_dofs.append(local_dofs[patch.surface.faces])
        for local_index, dof in enumerate(local_dofs):
            dof_members[int(dof)].append(
                patch.surface.vertices[local_index]
            )
    dof_positions = np.array(
        [
            np.mean(np.asarray(members), axis=0)
            for members in dof_members
        ]
    )
    return (
        tuple(patch_face_dofs),
        tuple(patch_vertex_dofs),
        dof_positions,
    )


def project_assembly_sheet_average_velocity(
    assembly: QuadraticDoubletAssembly,
    collocation_velocity,
    *,
    geometry_tolerance: float = 1.0e-12,
    relative_rank_tolerance: float | None = None,
) -> ProjectedAssemblySheetVelocity:
    """Project assembly collocation velocities onto welded P1 geometry."""
    if not isinstance(assembly, QuadraticDoubletAssembly):
        raise DistributedDoubletError(
            "assembly must be QuadraticDoubletAssembly"
        )
    if geometry_tolerance < 0.0 or not np.isfinite(
        geometry_tolerance
    ):
        raise DistributedDoubletError(
            "geometry_tolerance must be finite and non-negative"
        )
    (
        patch_face_dofs,
        patch_vertex_dofs,
        dof_positions,
    ) = _assembly_p1_topology(
        assembly,
        geometry_tolerance=geometry_tolerance,
    )
    face_count = sum(
        len(patch.surface) for patch in assembly.patches
    )
    velocity = np.asarray(collocation_velocity, dtype=float)
    expected_shape = (4 * face_count, 3)
    if velocity.shape != expected_shape:
        raise DistributedDoubletError(
            "collocation_velocity must have shape "
            f"{expected_shape}, got {velocity.shape}"
        )
    if not np.all(np.isfinite(velocity)):
        raise DistributedDoubletError(
            "collocation_velocity contains non-finite values"
        )
    centroid = np.full(3, 1.0 / 3.0)
    local_basis = [centroid]
    for vertex_index in range(3):
        point = 0.1 * centroid
        point = point.copy()
        point[vertex_index] += 0.9
        local_basis.append(point)
    local_basis = np.asarray(local_basis)
    matrix = np.zeros(
        (4 * face_count, len(dof_positions)),
        dtype=float,
    )
    row = 0
    for faces in patch_face_dofs:
        for dofs in faces:
            matrix[row : row + 4, dofs] = local_basis
            row += 4
    coefficients, report = _solve_projection(
        matrix,
        velocity,
        relative_rank_tolerance=relative_rank_tolerance,
    )
    return ProjectedAssemblySheetVelocity(
        dof_positions=dof_positions,
        dof_velocity=coefficients,
        patch_face_dofs=patch_face_dofs,
        patch_vertex_dofs=patch_vertex_dofs,
        report=report,
    )


def project_assembly_vertex_star_normal_geometry_velocity(
    assembly: QuadraticDoubletAssembly,
    collocation_velocity,
    *,
    geometry_tolerance: float = 1.0e-12,
    relative_rank_tolerance: float | None = None,
) -> ProjectedAssemblyNormalGeometryVelocity:
    """Weld explicit seams, then take one normal limit per global vertex."""
    if not isinstance(assembly, QuadraticDoubletAssembly):
        raise DistributedDoubletError(
            "assembly must be QuadraticDoubletAssembly"
        )
    (
        patch_face_dofs,
        patch_vertex_dofs,
        dof_positions,
    ) = _assembly_p1_topology(
        assembly,
        geometry_tolerance=geometry_tolerance,
    )
    face_count = sum(
        len(patch.surface) for patch in assembly.patches
    )
    velocity = np.asarray(collocation_velocity, dtype=float)
    expected_shape = (4 * face_count, 3)
    if velocity.shape != expected_shape:
        raise DistributedDoubletError(
            "collocation_velocity must have shape "
            f"{expected_shape}, got {velocity.shape}"
        )
    if not np.all(np.isfinite(velocity)):
        raise DistributedDoubletError(
            "collocation_velocity contains non-finite values"
        )
    points, _, _, _ = assembly.interior_collocation_points()
    dof_normal_sum = np.zeros_like(dof_positions)
    incident_faces: list[list[tuple[int, int, int]]] = [
        [] for _ in dof_positions
    ]
    global_face_index = 0
    for patch_index, patch in enumerate(assembly.patches):
        for face_index, dofs in enumerate(patch_face_dofs[patch_index]):
            element = patch.surface.element(face_index)
            for dof in dofs:
                dof_normal_sum[dof] += element.area_vector
                incident_faces[dof].append(
                    (patch_index, face_index, global_face_index)
                )
            global_face_index += 1
    normal_norm = np.linalg.norm(dof_normal_sum, axis=1)
    if np.any(normal_norm <= np.finfo(float).eps):
        raise DistributedDoubletError(
            "assembly vertex normal is undefined"
        )
    dof_normals = dof_normal_sum / normal_norm[:, None]
    if relative_rank_tolerance is None:
        relative_rank_tolerance = (
            64.0
            * np.finfo(float).eps
            * max(4 * face_count, 3)
        )
    if (
        relative_rank_tolerance < 0.0
        or not np.isfinite(relative_rank_tolerance)
    ):
        raise DistributedDoubletError(
            "relative_rank_tolerance must be finite and non-negative"
        )
    dof_speed = np.empty(len(dof_positions))
    maximum_condition = 0.0
    maximum_abs_residual = 0.0
    maximum_rel_residual = 0.0
    maximum_input = 0.0
    total_samples = 0

    for dof, records in enumerate(incident_faces):
        if not records:
            raise DistributedDoubletError(
                f"assembly dof {dof} has no incident face"
            )
        neighbor_dofs = sorted(
            {
                int(other)
                for patch_index, face_index, _ in records
                for other in patch_face_dofs[patch_index][face_index]
                if int(other) != dof
            }
        )
        normal = dof_normals[dof]
        tangent1 = None
        for neighbor in neighbor_dofs:
            candidate = dof_positions[neighbor] - dof_positions[dof]
            candidate -= np.dot(candidate, normal) * normal
            norm = np.linalg.norm(candidate)
            if norm > 64.0 * np.finfo(float).eps:
                tangent1 = candidate / norm
                break
        if tangent1 is None:
            raise DistributedDoubletError(
                f"assembly dof {dof} has no tangent direction"
            )
        tangent2 = np.cross(normal, tangent1)
        rows = np.concatenate(
            [
                np.arange(
                    4 * global_face,
                    4 * global_face + 4,
                )
                for _, _, global_face in records
            ]
        )
        delta = points[rows] - dof_positions[dof]
        design = np.column_stack(
            (
                np.ones(len(rows)),
                delta @ tangent1,
                delta @ tangent2,
            )
        )
        sample_speed = np.empty(len(rows))
        offset = 0
        for patch_index, face_index, global_face in records:
            sample_speed[offset : offset + 4] = (
                velocity[4 * global_face : 4 * global_face + 4]
                @ assembly.patches[
                    patch_index
                ].surface.element(face_index).normal
            )
            offset += 4
        singular_values = np.linalg.svd(
            design,
            compute_uv=False,
            full_matrices=False,
        )
        rank = int(
            np.count_nonzero(
                singular_values
                > relative_rank_tolerance * singular_values[0]
            )
        )
        if rank != 3:
            raise DistributedDoubletError(
                "assembly vertex-star extrapolation is rank deficient: "
                f"dof={dof}, rank={rank}"
            )
        coefficients, _, solved_rank, solved_singular = np.linalg.lstsq(
            design,
            sample_speed,
            rcond=relative_rank_tolerance,
        )
        if int(solved_rank) != 3:
            raise DistributedDoubletError(
                "assembly vertex-star extrapolation lost rank"
            )
        residual = np.abs(design @ coefficients - sample_speed)
        scale = np.maximum(np.abs(sample_speed), np.finfo(float).eps)
        dof_speed[dof] = coefficients[0]
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
        total_samples += len(rows)
    report = SheetVelocityProjectionReport(
        samples=total_samples,
        degrees_of_freedom=len(dof_positions),
        rank=len(dof_positions),
        full_rank=True,
        condition_number=maximum_condition,
        max_abs_residual=maximum_abs_residual,
        max_rel_residual=maximum_rel_residual,
        max_input_norm=maximum_input,
        max_abs_residual_fraction=(
            maximum_abs_residual
            / max(maximum_input, np.finfo(float).eps)
        ),
        gauge="sheet_average_normal_vertex_star_assembly",
    )
    return ProjectedAssemblyNormalGeometryVelocity(
        dof_positions=dof_positions,
        dof_normal_speed=dof_speed,
        dof_normals=dof_normals,
        patch_vertex_dofs=patch_vertex_dofs,
        report=report,
    )
