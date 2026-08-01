"""Continuous-P2 semidiscrete transport for a material sheet scalar.

This no-force reference assembles the consistent mass matrix and the
relative-tangential-velocity advection matrix on a P1 triangular surface:

    M mu_dot + C mu = 0,
    C_ij = integral N_i c_t.grad_s(N_j) dS.

There is one global scalar degree of freedom per mesh vertex and edge
midpoint.  The module contains no mass lumping, upwind term, artificial
diffusion, limiter, time integrator, pressure, force, core or target.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from .distributed_doublet import (
    DistributedDoubletError,
    QuadraticDoubletAssembly,
    QuadraticDoubletElement,
    _triangle_quadrature,
    p2_shape_values,
)
from .sheet_velocity_projection import _assembly_p1_topology


VelocityProvider = Callable[[np.ndarray], np.ndarray]


def _finite(name: str, value: Any, *, ndim: int) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim != ndim or not np.all(np.isfinite(array)):
        raise DistributedDoubletError(
            f"{name} must be a finite {ndim}-D array"
        )
    return array


@dataclass(frozen=True)
class ContinuousP2ScalarTopology:
    vertices: np.ndarray
    faces: np.ndarray
    edge_vertices: np.ndarray
    local_to_global: np.ndarray
    degree_of_freedom_coordinates: np.ndarray

    @property
    def degree_of_freedom_count(self) -> int:
        return len(self.degree_of_freedom_coordinates)


@dataclass(frozen=True)
class P2SurfaceMaterialTransportOperator:
    topology: ContinuousP2ScalarTopology
    mass_matrix: np.ndarray
    advection_matrix: np.ndarray
    mass_rank: int
    mass_condition_number: float
    maximum_relative_velocity_normal_component: float
    constant_rate_residual: float

    def rate(self, potential_jump: Any) -> np.ndarray:
        value = np.asarray(potential_jump, dtype=float)
        expected = (self.topology.degree_of_freedom_count,)
        if value.shape != expected or not np.all(np.isfinite(value)):
            raise DistributedDoubletError(
                f"potential_jump must have shape {expected}"
            )
        try:
            return np.linalg.solve(
                self.mass_matrix,
                -(self.advection_matrix @ value),
            )
        except np.linalg.LinAlgError as error:
            raise DistributedDoubletError(
                "P2 transport mass solve failed"
            ) from error


@dataclass(frozen=True)
class P2PatchMaterialTransportOperator:
    operator: P2SurfaceMaterialTransportOperator
    patch_face_dofs: tuple[np.ndarray, ...]
    maximum_interface_geometry_gap: float

    def rate(self, potential_jump: Any) -> np.ndarray:
        return self.operator.rate(potential_jump)

    def patch_face_values(
        self,
        potential_jump: Any,
    ) -> tuple[np.ndarray, ...]:
        value = np.asarray(potential_jump, dtype=float)
        expected = (
            self.operator.topology.degree_of_freedom_count,
        )
        if value.shape != expected or not np.all(np.isfinite(value)):
            raise DistributedDoubletError(
                f"potential_jump must have shape {expected}"
            )
        return tuple(value[dofs] for dofs in self.patch_face_dofs)

    def extract_patch_scalar(
        self,
        assembly: QuadraticDoubletAssembly,
        *,
        tolerance: float = 1.0e-12,
    ) -> np.ndarray:
        if not isinstance(assembly, QuadraticDoubletAssembly):
            raise DistributedDoubletError(
                "assembly must be QuadraticDoubletAssembly"
            )
        if len(assembly.patches) != len(self.patch_face_dofs):
            raise DistributedDoubletError(
                "assembly patch count does not match operator"
            )
        if tolerance < 0.0 or not np.isfinite(tolerance):
            raise DistributedDoubletError(
                "tolerance must be finite and non-negative"
            )
        members: list[list[float]] = [
            []
            for _ in range(
                self.operator.topology.degree_of_freedom_count
            )
        ]
        for patch, dofs in zip(
            assembly.patches,
            self.patch_face_dofs,
        ):
            if patch.surface.face_mu.shape != dofs.shape:
                raise DistributedDoubletError(
                    "patch face P2 values do not match topology"
                )
            for local_value, global_dof in zip(
                patch.surface.face_mu.ravel(),
                dofs.ravel(),
            ):
                members[int(global_dof)].append(float(local_value))
        if any(not values for values in members):
            raise DistributedDoubletError(
                "patch scalar omitted a global P2 degree of freedom"
            )
        result = np.array(
            [float(np.mean(values)) for values in members]
        )
        maximum_spread = max(
            (
                max(values) - min(values)
                for values in members
            ),
            default=0.0,
        )
        if maximum_spread > tolerance:
            raise DistributedDoubletError(
                "patch-local P2 scalar values disagree at a shared DOF"
            )
        return result


@dataclass(frozen=True)
class P2EssentialScalarTrace:
    """Typed global P2 DOFs whose scalar values are prescribed by identity."""

    global_dof_indices: np.ndarray
    trace_id: str

    def __post_init__(self) -> None:
        indices = np.asarray(self.global_dof_indices, dtype=np.int64)
        if (
            indices.ndim != 1
            or len(indices) == 0
            or len(np.unique(indices)) != len(indices)
            or np.any(indices < 0)
        ):
            raise DistributedDoubletError(
                "essential trace DOF ids must be a nonempty unique "
                "non-negative 1-D array"
            )
        if not isinstance(self.trace_id, str) or not self.trace_id:
            raise DistributedDoubletError(
                "essential scalar trace_id must be nonempty"
            )
        object.__setattr__(self, "global_dof_indices", indices.copy())

    def free_indices(self, dof_count: int) -> np.ndarray:
        if (
            not isinstance(dof_count, (int, np.integer))
            or dof_count <= 0
            or np.any(self.global_dof_indices >= dof_count)
        ):
            raise DistributedDoubletError(
                "essential trace DOF ids exceed the operator topology"
            )
        mask = np.ones(int(dof_count), dtype=bool)
        mask[self.global_dof_indices] = False
        free = np.flatnonzero(mask)
        if len(free) == 0:
            raise DistributedDoubletError(
                "essential trace leaves no free scalar DOFs"
            )
        return free


@dataclass(frozen=True)
class P2TransportBoundaryRoles:
    """Explicit, disjoint global-P2 roles for an open material sheet."""

    body_inflow_dof_indices: np.ndarray
    old_outflow_dof_indices: np.ndarray
    lower_characteristic_dof_indices: np.ndarray
    upper_characteristic_dof_indices: np.ndarray
    role_id: str

    def __post_init__(self) -> None:
        names = (
            "body_inflow_dof_indices",
            "old_outflow_dof_indices",
            "lower_characteristic_dof_indices",
            "upper_characteristic_dof_indices",
        )
        arrays = []
        for name in names:
            value = np.asarray(getattr(self, name), dtype=np.int64)
            if (
                value.ndim != 1
                or len(value) == 0
                or len(np.unique(value)) != len(value)
                or np.any(value < 0)
            ):
                raise DistributedDoubletError(
                    f"{name} must be a nonempty unique non-negative "
                    "1-D array"
                )
            arrays.append(value.copy())
            object.__setattr__(self, name, value.copy())
        joined = np.concatenate(arrays)
        if len(np.unique(joined)) != len(joined):
            raise DistributedDoubletError(
                "P2 transport boundary roles must be pairwise disjoint"
            )
        if not isinstance(self.role_id, str) or not self.role_id:
            raise DistributedDoubletError(
                "P2 transport boundary role_id must be nonempty"
            )

    @property
    def all_boundary_dof_indices(self) -> np.ndarray:
        return np.concatenate(
            (
                self.body_inflow_dof_indices,
                self.old_outflow_dof_indices,
                self.lower_characteristic_dof_indices,
                self.upper_characteristic_dof_indices,
            )
        )

    def validate(
        self,
        dof_count: int,
        *,
        declared_boundary_dof_indices: Any,
    ) -> None:
        if (
            not isinstance(dof_count, (int, np.integer))
            or dof_count <= 0
            or np.any(self.all_boundary_dof_indices >= dof_count)
        ):
            raise DistributedDoubletError(
                "P2 transport boundary role ids exceed the topology"
            )
        declared = np.asarray(
            declared_boundary_dof_indices,
            dtype=np.int64,
        )
        if (
            declared.ndim != 1
            or len(declared) == 0
            or len(np.unique(declared)) != len(declared)
            or np.any(declared < 0)
            or np.any(declared >= dof_count)
        ):
            raise DistributedDoubletError(
                "declared P2 boundary ids must be valid and unique"
            )
        if not np.array_equal(
            np.sort(self.all_boundary_dof_indices),
            np.sort(declared),
        ):
            raise DistributedDoubletError(
                "typed P2 boundary roles do not partition the "
                "declared boundary"
            )

    def body_essential_trace(self) -> P2EssentialScalarTrace:
        return P2EssentialScalarTrace(
            self.body_inflow_dof_indices,
            f"{self.role_id}:body-inflow",
        )


@dataclass(frozen=True)
class P2ConstrainedTransportRate:
    rate: np.ndarray
    free_dof_indices: np.ndarray
    constrained_dof_indices: np.ndarray
    free_mass_rank: int
    free_mass_condition_number: float
    prescribed_value_error: float


def p2_essential_trace_transport_rate(
    operator: P2SurfaceMaterialTransportOperator,
    potential_jump: Any,
    *,
    essential_trace: P2EssentialScalarTrace,
    prescribed_value: Any,
    prescribed_time_derivative: Any,
    prescribed_value_tolerance: float = 1.0e-12,
) -> P2ConstrainedTransportRate:
    """Return the free/constrained block rate for a prescribed P2 trace."""
    if not isinstance(operator, P2SurfaceMaterialTransportOperator):
        raise DistributedDoubletError(
            "operator must be P2SurfaceMaterialTransportOperator"
        )
    if not isinstance(essential_trace, P2EssentialScalarTrace):
        raise DistributedDoubletError(
            "essential_trace must be P2EssentialScalarTrace"
        )
    if (
        prescribed_value_tolerance < 0.0
        or not np.isfinite(prescribed_value_tolerance)
    ):
        raise DistributedDoubletError(
            "prescribed_value_tolerance must be finite and non-negative"
        )
    count = operator.topology.degree_of_freedom_count
    state = np.asarray(potential_jump, dtype=float)
    if state.shape != (count,) or not np.all(np.isfinite(state)):
        raise DistributedDoubletError(
            f"potential_jump must have shape {(count,)}"
        )
    boundary = essential_trace.global_dof_indices
    free = essential_trace.free_indices(count)
    value = np.asarray(prescribed_value, dtype=float)
    derivative = np.asarray(prescribed_time_derivative, dtype=float)
    expected = (len(boundary),)
    if (
        value.shape != expected
        or derivative.shape != expected
        or not np.all(np.isfinite(value))
        or not np.all(np.isfinite(derivative))
    ):
        raise DistributedDoubletError(
            f"prescribed trace value/rate must have shape {expected}"
        )
    prescribed_error = float(
        np.max(
            np.abs(state[boundary] - value),
            initial=0.0,
        )
    )
    if prescribed_error > prescribed_value_tolerance:
        raise DistributedDoubletError(
            "state does not satisfy the essential scalar trace"
        )
    mass_ff = operator.mass_matrix[np.ix_(free, free)]
    mass_fb = operator.mass_matrix[np.ix_(free, boundary)]
    advection_ff = operator.advection_matrix[np.ix_(free, free)]
    advection_fb = operator.advection_matrix[
        np.ix_(free, boundary)
    ]
    rank = int(np.linalg.matrix_rank(mass_ff))
    condition = float(np.linalg.cond(mass_ff))
    if rank != len(free) or not np.isfinite(condition):
        raise DistributedDoubletError(
            "free P2 trace mass block is rank deficient"
        )
    right_hand_side = -(
        mass_fb @ derivative
        + advection_ff @ state[free]
        + advection_fb @ value
    )
    try:
        free_rate = np.linalg.solve(mass_ff, right_hand_side)
    except np.linalg.LinAlgError as error:
        raise DistributedDoubletError(
            "constrained P2 transport mass solve failed"
        ) from error
    rate = np.empty(count, dtype=float)
    rate[free] = free_rate
    rate[boundary] = derivative
    return P2ConstrainedTransportRate(
        rate=rate,
        free_dof_indices=free,
        constrained_dof_indices=boundary.copy(),
        free_mass_rank=rank,
        free_mass_condition_number=condition,
        prescribed_value_error=prescribed_error,
    )


def continuous_p2_scalar_topology(
    vertices: Any,
    faces: Any,
) -> ContinuousP2ScalarTopology:
    vertex_array = _finite("vertices", vertices, ndim=2)
    if vertex_array.shape[1] != 3:
        raise DistributedDoubletError(
            "vertices must have shape (n,3)"
        )
    face_array = np.asarray(faces, dtype=np.int64)
    if (
        face_array.ndim != 2
        or face_array.shape[1] != 3
        or np.any(face_array < 0)
        or np.any(face_array >= len(vertex_array))
    ):
        raise DistributedDoubletError(
            "faces must have valid shape (m,3)"
        )
    if np.any(
        (face_array[:, 0] == face_array[:, 1])
        | (face_array[:, 1] == face_array[:, 2])
        | (face_array[:, 2] == face_array[:, 0])
    ):
        raise DistributedDoubletError(
            "faces must not repeat vertices"
        )
    edges: dict[tuple[int, int], int] = {}
    local = np.empty((len(face_array), 6), dtype=np.int64)
    local[:, :3] = face_array
    for face_index, face in enumerate(face_array):
        QuadraticDoubletElement(
            vertex_array[face],
            np.zeros(6),
        )
        for local_edge, (first, second) in enumerate(
            ((0, 1), (1, 2), (2, 0))
        ):
            edge = tuple(
                sorted((int(face[first]), int(face[second])))
            )
            if edge not in edges:
                edges[edge] = len(edges)
            local[face_index, 3 + local_edge] = (
                len(vertex_array) + edges[edge]
            )
    edge_vertices = np.empty((len(edges), 2), dtype=np.int64)
    for edge, index in edges.items():
        edge_vertices[index] = edge
    coordinates = np.concatenate(
        (
            vertex_array,
            0.5
            * (
                vertex_array[edge_vertices[:, 0]]
                + vertex_array[edge_vertices[:, 1]]
            ),
        )
    )
    return ContinuousP2ScalarTopology(
        vertices=vertex_array.copy(),
        faces=face_array.copy(),
        edge_vertices=edge_vertices,
        local_to_global=local,
        degree_of_freedom_coordinates=coordinates,
    )


def assemble_p2_surface_material_transport(
    vertices: Any,
    faces: Any,
    *,
    relative_velocity_provider: VelocityProvider,
    quadrature_order: int = 10,
) -> P2SurfaceMaterialTransportOperator:
    """Assemble the unstabilized continuous-P2 ALE scalar operator."""
    if not callable(relative_velocity_provider):
        raise DistributedDoubletError(
            "relative_velocity_provider must be callable"
        )
    if (
        not isinstance(quadrature_order, (int, np.integer))
        or int(quadrature_order) < 2
    ):
        raise DistributedDoubletError(
            "quadrature_order must be an integer >=2"
        )
    topology = continuous_p2_scalar_topology(vertices, faces)
    count = topology.degree_of_freedom_count
    mass = np.zeros((count, count), dtype=float)
    advection = np.zeros_like(mass)
    maximum_normal = 0.0
    barycentric, reference_weight = _triangle_quadrature(
        int(quadrature_order)
    )
    shape = p2_shape_values(barycentric)
    for face_index, face in enumerate(topology.faces):
        element = QuadraticDoubletElement(
            topology.vertices[face],
            np.zeros(6),
        )
        points = barycentric @ element.vertices
        velocity = np.asarray(
            relative_velocity_provider(points),
            dtype=float,
        )
        if velocity.shape != points.shape or not np.all(
            np.isfinite(velocity)
        ):
            raise DistributedDoubletError(
                "relative velocity provider returned incompatible data"
            )
        normal_component = velocity @ element.normal
        maximum_normal = max(
            maximum_normal,
            float(
                np.max(
                    np.abs(normal_component),
                    initial=0.0,
                )
            ),
        )
        tangential_velocity = (
            velocity
            - normal_component[:, None] * element.normal
        )
        gradients = element.shape_gradients(barycentric)
        advective_derivative = np.einsum(
            "qi,qji->qj",
            tangential_velocity,
            gradients,
        )
        weights = (
            reference_weight * np.linalg.norm(element.area_vector)
        )
        local_mass = np.einsum(
            "qi,q,qj->ij",
            shape,
            weights,
            shape,
        )
        local_advection = np.einsum(
            "qi,q,qj->ij",
            shape,
            weights,
            advective_derivative,
        )
        dofs = topology.local_to_global[face_index]
        mass[np.ix_(dofs, dofs)] += local_mass
        advection[np.ix_(dofs, dofs)] += local_advection
    rank = int(np.linalg.matrix_rank(mass))
    condition = float(np.linalg.cond(mass))
    if rank != count or not np.isfinite(condition):
        raise DistributedDoubletError(
            "continuous-P2 mass matrix is rank deficient"
        )
    constant_rate = np.linalg.solve(
        mass,
        -(advection @ np.ones(count)),
    )
    return P2SurfaceMaterialTransportOperator(
        topology=topology,
        mass_matrix=mass,
        advection_matrix=advection,
        mass_rank=rank,
        mass_condition_number=condition,
        maximum_relative_velocity_normal_component=maximum_normal,
        constant_rate_residual=float(
            np.max(np.abs(constant_rate), initial=0.0)
        ),
    )


def assemble_p2_patch_material_transport(
    assembly: QuadraticDoubletAssembly,
    *,
    relative_velocity_provider: VelocityProvider,
    quadrature_order: int = 10,
    geometry_tolerance: float = 1.0e-12,
    strength_tolerance: float = 1.0e-12,
) -> P2PatchMaterialTransportOperator:
    """Assemble one global P2 scalar operator across explicit patch seams."""
    if not isinstance(assembly, QuadraticDoubletAssembly):
        raise DistributedDoubletError(
            "assembly must be QuadraticDoubletAssembly"
        )
    report = assembly.topology_report(
        strength_tolerance=strength_tolerance,
        geometry_tolerance=geometry_tolerance,
    )
    if not report.compatible:
        raise DistributedDoubletError(
            f"assembly topology is incompatible: {report}"
        )
    (
        patch_face_vertices,
        _patch_vertex_dofs,
        global_vertices,
    ) = _assembly_p1_topology(
        assembly,
        geometry_tolerance=geometry_tolerance,
    )
    global_faces = np.vstack(patch_face_vertices)
    operator = assemble_p2_surface_material_transport(
        global_vertices,
        global_faces,
        relative_velocity_provider=relative_velocity_provider,
        quadrature_order=quadrature_order,
    )
    patch_face_dofs = []
    offset = 0
    for patch in assembly.patches:
        count = len(patch.surface.faces)
        patch_face_dofs.append(
            operator.topology.local_to_global[
                offset : offset + count
            ].copy()
        )
        offset += count
    return P2PatchMaterialTransportOperator(
        operator=operator,
        patch_face_dofs=tuple(patch_face_dofs),
        maximum_interface_geometry_gap=(
            report.max_interface_geometry_gap
        ),
    )
