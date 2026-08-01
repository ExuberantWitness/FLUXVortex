"""No-force quadratic distributed-doublet field reference for N3.1j.

The state is a material, continuous P2 potential-jump field on planar
triangles.  The module contains CPU equation references for its off-sheet
double-layer field, parameter-free sheet-average velocity, explicit
multi-patch topology, and material wake histories:

* six material degrees of freedom live at the three vertices and three
  midsides of every triangle;
* equality of the three trace degrees of freedom on every shared edge makes
  the doublet strength continuous and removes an unpaired internal edge
  filament;
* material convection changes geometry while preserving the six strengths;
* the equivalent sheet-vorticity vector is ``grad_s(mu) x normal``;
* analytic radial reduction plus edge quadrature evaluates the off-sheet
  field, while a Cauchy finite-part identity supplies the on-sheet average.

No aerodynamic force, pressure, empirical core, source-strength smoothing, or
target-load normalization is present in this module.  The CPU operators are
reference oracles, not the compiled production kernel.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Mapping

import numpy as np


class DistributedDoubletError(ValueError):
    """Invalid geometry, topology, or material doublet state."""


def _finite(
    name: str,
    value,
    *,
    ndim: int | None = None,
    shape: tuple[int, ...] | None = None,
    dtype=float,
) -> np.ndarray:
    array = np.asarray(value, dtype=dtype)
    if ndim is not None and array.ndim != ndim:
        raise DistributedDoubletError(
            f"{name} must have ndim={ndim}, got shape {array.shape}"
        )
    if shape is not None and array.shape != shape:
        raise DistributedDoubletError(
            f"{name} must have shape {shape}, got {array.shape}"
        )
    if np.issubdtype(array.dtype, np.floating) and not np.all(np.isfinite(array)):
        raise DistributedDoubletError(f"{name} contains non-finite values")
    return array


def p2_shape_values(barycentric) -> np.ndarray:
    """Quadratic Lagrange triangle basis.

    Node order is ``(v0, v1, v2, e01, e12, e20)``.
    """
    lam = _finite("barycentric", barycentric, ndim=2)
    if lam.shape[1] != 3:
        raise DistributedDoubletError(
            f"barycentric must have shape (n,3), got {lam.shape}"
        )
    l0, l1, l2 = lam.T
    return np.column_stack(
        (
            l0 * (2.0 * l0 - 1.0),
            l1 * (2.0 * l1 - 1.0),
            l2 * (2.0 * l2 - 1.0),
            4.0 * l0 * l1,
            4.0 * l1 * l2,
            4.0 * l2 * l0,
        )
    )


@dataclass(frozen=True)
class QuadraticDoubletElement:
    """One planar P2 material doublet element."""

    vertices: np.ndarray
    material_mu: np.ndarray

    def __post_init__(self) -> None:
        vertices = _finite("vertices", self.vertices, shape=(3, 3))
        material_mu = _finite("material_mu", self.material_mu, shape=(6,))
        edge01 = vertices[1] - vertices[0]
        edge02 = vertices[2] - vertices[0]
        area_vector = np.cross(edge01, edge02)
        area2 = float(np.linalg.norm(area_vector))
        scale = max(
            float(np.linalg.norm(edge01)),
            float(np.linalg.norm(edge02)),
            float(np.linalg.norm(vertices[2] - vertices[1])),
            1.0,
        )
        if area2 <= 64.0 * np.finfo(float).eps * scale**2:
            raise DistributedDoubletError("triangle is degenerate")
        object.__setattr__(self, "vertices", vertices.copy())
        object.__setattr__(self, "material_mu", material_mu.copy())

    @property
    def area_vector(self) -> np.ndarray:
        return np.cross(
            self.vertices[1] - self.vertices[0],
            self.vertices[2] - self.vertices[0],
        )

    @property
    def area(self) -> float:
        return 0.5 * float(np.linalg.norm(self.area_vector))

    @property
    def normal(self) -> np.ndarray:
        vector = self.area_vector
        return vector / np.linalg.norm(vector)

    @property
    def material_nodes(self) -> np.ndarray:
        v0, v1, v2 = self.vertices
        return np.stack(
            (
                v0,
                v1,
                v2,
                0.5 * (v0 + v1),
                0.5 * (v1 + v2),
                0.5 * (v2 + v0),
            )
        )

    @property
    def barycentric_gradients(self) -> np.ndarray:
        """Physical surface gradients of the three barycentric coordinates."""
        n2 = float(np.dot(self.area_vector, self.area_vector))
        v0, v1, v2 = self.vertices
        return np.stack(
            (
                np.cross(self.area_vector, v2 - v1) / n2,
                np.cross(self.area_vector, v0 - v2) / n2,
                np.cross(self.area_vector, v1 - v0) / n2,
            )
        )

    def barycentric_coordinates(
        self,
        points,
        *,
        plane_tolerance: float = 1.0e-10,
    ) -> np.ndarray:
        point_array = _finite("points", points, ndim=2)
        if point_array.shape[1] != 3:
            raise DistributedDoubletError(
                f"points must have shape (n,3), got {point_array.shape}"
            )
        if plane_tolerance < 0.0 or not np.isfinite(plane_tolerance):
            raise DistributedDoubletError(
                "plane_tolerance must be finite and non-negative"
            )
        delta = point_array - self.vertices[0]
        signed_distance = delta @ self.normal
        length_scale = max(np.sqrt(2.0 * self.area), 1.0)
        if np.max(np.abs(signed_distance), initial=0.0) > (
            plane_tolerance * length_scale
        ):
            raise DistributedDoubletError("evaluation point is off the element plane")
        gradients = self.barycentric_gradients
        l1 = delta @ gradients[1]
        l2 = delta @ gradients[2]
        l0 = 1.0 - l1 - l2
        return np.column_stack((l0, l1, l2))

    def evaluate_barycentric(self, barycentric) -> np.ndarray:
        return p2_shape_values(barycentric) @ self.material_mu

    def evaluate(self, points, *, plane_tolerance: float = 1.0e-10) -> np.ndarray:
        return self.evaluate_barycentric(
            self.barycentric_coordinates(
                points,
                plane_tolerance=plane_tolerance,
            )
        )

    def shape_gradients(self, barycentric) -> np.ndarray:
        """Return ``grad_s(N_i)`` with shape ``(npoint, 6, 3)``."""
        lam = _finite("barycentric", barycentric, ndim=2)
        if lam.shape[1] != 3:
            raise DistributedDoubletError(
                f"barycentric must have shape (n,3), got {lam.shape}"
            )
        l0, l1, l2 = lam.T
        g0, g1, g2 = self.barycentric_gradients
        result = np.empty((len(lam), 6, 3), dtype=float)
        result[:, 0] = (4.0 * l0 - 1.0)[:, None] * g0
        result[:, 1] = (4.0 * l1 - 1.0)[:, None] * g1
        result[:, 2] = (4.0 * l2 - 1.0)[:, None] * g2
        result[:, 3] = 4.0 * (
            l0[:, None] * g1 + l1[:, None] * g0
        )
        result[:, 4] = 4.0 * (
            l1[:, None] * g2 + l2[:, None] * g1
        )
        result[:, 5] = 4.0 * (
            l2[:, None] * g0 + l0[:, None] * g2
        )
        return result

    def surface_gradient_barycentric(self, barycentric) -> np.ndarray:
        gradients = self.shape_gradients(barycentric)
        return np.einsum("nij,i->nj", gradients, self.material_mu)

    def surface_gradient(
        self,
        points,
        *,
        plane_tolerance: float = 1.0e-10,
    ) -> np.ndarray:
        return self.surface_gradient_barycentric(
            self.barycentric_coordinates(
                points,
                plane_tolerance=plane_tolerance,
            )
        )

    def sheet_vorticity_barycentric(self, barycentric) -> np.ndarray:
        """Equivalent sheet vector ``grad_s(mu) x n``.

        For a flat element with span coordinate ``eta=y`` and
        ``mu=A+B*eta+C*eta^2``, this gives the DVE convention
        ``gamma_x=B+2*C*eta``.
        """
        gradient = self.surface_gradient_barycentric(barycentric)
        return np.cross(gradient, self.normal)

    def integral_mu(self) -> float:
        """Exact surface integral of the P2 doublet strength."""
        # P2 vertex basis functions integrate to zero; every midside basis
        # integrates to area/3.
        return self.area * float(np.sum(self.material_mu[3:])) / 3.0

    def material_update(self, vertices) -> "QuadraticDoubletElement":
        """Convect/stretch the material triangle without changing ``mu``."""
        return QuadraticDoubletElement(vertices, self.material_mu)


@dataclass(frozen=True)
class DoubletContinuityReport:
    internal_edges: int
    boundary_edges: int
    max_trace_node_jump: float
    max_trace_jump: float
    compatible: bool


@dataclass(frozen=True)
class DoubletVorticityContinuityReport:
    internal_edges: int
    max_midpoint_vorticity_jump: float
    coplanar: bool
    compatible: bool


@dataclass(frozen=True)
class DoubletBoundaryReport:
    boundary_edges: int
    max_boundary_trace: float
    compatible: bool


@dataclass(frozen=True)
class DoubletAssemblyReport:
    patches: int
    zero_boundaries: int
    coupled_interfaces: int
    max_zero_trace: float
    max_interface_trace_jump: float
    max_interface_geometry_gap: float
    compatible: bool


@dataclass(frozen=True)
class MaterialKelvinReport:
    max_material_mu_residual: float
    topology_equal: bool
    passed: bool


@dataclass(frozen=True)
class DoubletVelocityReport:
    quadrature_order: int
    max_abs_change: float
    max_rel_change: float
    converged: bool


@dataclass(frozen=True)
class SheetAverageVelocityReport:
    quadrature_order: int
    max_abs_change: float
    max_rel_change: float
    converged: bool


@dataclass(frozen=True)
class MaterialWakeHistoryReport:
    bands: int
    internal_seams: int
    same_family: bool
    same_span_nodes: bool
    time_contiguous: bool
    max_time_gap: float
    max_geometry_gap: float
    max_trace_jump: float
    compatible: bool


@lru_cache(maxsize=None)
def _triangle_quadrature(order: int) -> tuple[np.ndarray, np.ndarray]:
    """Tensor Gauss rule mapped onto a reference triangle."""
    if not isinstance(order, (int, np.integer)) or order < 2:
        raise DistributedDoubletError(
            f"quadrature order must be an integer >=2, got {order!r}"
        )
    abscissa, weight = np.polynomial.legendre.leggauss(int(order))
    coordinate = 0.5 * (abscissa + 1.0)
    weight = 0.5 * weight
    u, t = np.meshgrid(coordinate, coordinate, indexing="ij")
    wu, wt = np.meshgrid(weight, weight, indexing="ij")
    barycentric = np.column_stack(
        (
            ((1.0 - u) * (1.0 - t)).ravel(),
            u.ravel(),
            (t * (1.0 - u)).ravel(),
        )
    )
    # Physical area weight is multiplied by |(v1-v0)x(v2-v0)| later.
    reference_weight = (wu * wt * (1.0 - u)).ravel()
    return barycentric, reference_weight


@lru_cache(maxsize=None)
def _line_quadrature(order: int) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(order, (int, np.integer)) or order < 2:
        raise DistributedDoubletError(
            f"quadrature order must be an integer >=2, got {order!r}"
        )
    abscissa, weights = np.polynomial.legendre.leggauss(int(order))
    return 0.5 * (abscissa + 1.0), 0.5 * weights


def _target_sinh_line_quadrature(
    start: np.ndarray,
    end: np.ndarray,
    target: np.ndarray,
    *,
    order: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Target-centered exact change of coordinate on one straight edge.

    If ``x(s)=start+s*(end-start)``, ``s0`` is the closest-segment
    coordinate and ``d`` is the corresponding distance normalized by the
    edge length.  The map ``s=s0+d*sinh(t)`` sends its exact endpoint images
    back to ``[0,1]`` and clusters nodes around the closest physical point.
    It changes neither the integration interval nor the kernel.
    """
    if not isinstance(order, (int, np.integer)) or order < 2:
        raise DistributedDoubletError(
            "quadrature order must be an integer >=2"
        )
    edge = np.asarray(end, dtype=float) - np.asarray(start, dtype=float)
    point = np.asarray(target, dtype=float)
    length_squared = float(edge @ edge)
    if (
        point.shape != (3,)
        or not np.all(np.isfinite(point))
        or not np.all(np.isfinite(edge))
        or length_squared <= np.finfo(float).tiny
    ):
        raise DistributedDoubletError(
            "target-sinh edge quadrature requires finite nondegenerate geometry"
        )
    length = np.sqrt(length_squared)
    closest_coordinate = float(
        np.clip(
            ((point - start) @ edge) / length_squared,
            0.0,
            1.0,
        )
    )
    closest = start + closest_coordinate * edge
    normalized_distance = float(
        np.linalg.norm(point - closest) / length
    )
    edge_tolerance = (
        128.0
        * np.finfo(float).eps
        * max(1.0, length)
        / length
    )
    if (
        not np.isfinite(normalized_distance)
        or normalized_distance <= edge_tolerance
    ):
        raise DistributedDoubletError(
            "target-sinh edge quadrature received a target on an edge"
        )

    lower = np.arcsinh(
        -closest_coordinate / normalized_distance
    )
    upper = np.arcsinh(
        (1.0 - closest_coordinate) / normalized_distance
    )
    abscissa, weights = np.polynomial.legendre.leggauss(int(order))
    half_width = 0.5 * (upper - lower)
    midpoint = 0.5 * (upper + lower)
    transformed = midpoint + half_width * abscissa
    coordinate = (
        closest_coordinate
        + normalized_distance * np.sinh(transformed)
    )
    physical_weights = (
        weights
        * half_width
        * normalized_distance
        * np.cosh(transformed)
    )
    if (
        not np.all(np.isfinite(coordinate))
        or not np.all(np.isfinite(physical_weights))
        or np.any(coordinate <= 0.0)
        or np.any(coordinate >= 1.0)
        or np.any(physical_weights <= 0.0)
    ):
        raise DistributedDoubletError(
            "target-sinh edge quadrature produced invalid nodes or weights"
        )
    return coordinate, physical_weights


def _edge_line_quadrature(
    start: np.ndarray,
    end: np.ndarray,
    target: np.ndarray,
    *,
    order: int,
    edge_quadrature: str,
) -> tuple[np.ndarray, np.ndarray]:
    if edge_quadrature == "standard":
        return _line_quadrature(order)
    if edge_quadrature in (
        "target_sinh",
        "target_sinh_analytic_boundary",
        "target_sinh_analytic_sheet",
    ):
        return _target_sinh_line_quadrature(
            start,
            end,
            target,
            order=order,
        )
    raise DistributedDoubletError(
        "edge_quadrature must be 'standard', 'target_sinh' or "
        "'target_sinh_analytic_boundary' or "
        "'target_sinh_analytic_sheet'"
    )


def _analytic_quadratic_boundary_vortex_velocity(
    element: QuadraticDoubletElement,
    start_index: int,
    end_index: int,
    points: np.ndarray,
) -> np.ndarray:
    """Exact finite-segment Biot--Savart field for one P2 edge trace."""
    start = element.vertices[start_index]
    end = element.vertices[end_index]
    edge = end - start
    length = float(np.linalg.norm(edge))
    if not np.isfinite(length) or length <= np.finfo(float).tiny:
        raise DistributedDoubletError(
            "analytic P2 boundary vortex requires a nondegenerate edge"
        )
    tangent = edge / length
    edge_points = np.vstack((start, 0.5 * (start + end), end))
    mu_start, mu_mid, mu_end = element.evaluate(edge_points)
    coefficient_a = 2.0 * (mu_end + mu_start - 2.0 * mu_mid)
    coefficient_b = 4.0 * mu_mid - 3.0 * mu_start - mu_end
    coefficient_c = mu_start

    result = np.zeros_like(points)
    tolerance = 128.0 * np.finfo(float).eps * max(1.0, length)
    for point_index, point in enumerate(points):
        line_coordinate = float((point - start) @ tangent)
        transverse = point - start - line_coordinate * tangent
        distance = float(np.linalg.norm(transverse))
        if distance <= tolerance:
            if -tolerance <= line_coordinate <= length + tolerance:
                raise DistributedDoubletError(
                    "analytic P2 boundary vortex received a target on an edge"
                )
            # A target on the infinite line but outside the finite segment has
            # exactly zero Biot--Savart cross product.
            continue

        lower = -line_coordinate
        upper = length - line_coordinate
        lower_radius = np.hypot(lower, distance)
        upper_radius = np.hypot(upper, distance)
        moment_0 = (
            upper / (distance**2 * upper_radius)
            - lower / (distance**2 * lower_radius)
        )
        moment_1 = (
            -1.0 / upper_radius + 1.0 / lower_radius
        )
        moment_2 = (
            np.arcsinh(upper / distance)
            - upper / upper_radius
            - np.arcsinh(lower / distance)
            + lower / lower_radius
        )
        normalized_line_coordinate = line_coordinate / length
        polynomial_2 = coefficient_a / length**2
        polynomial_1 = (
            2.0 * coefficient_a * normalized_line_coordinate
            + coefficient_b
        ) / length
        polynomial_0 = (
            coefficient_a * normalized_line_coordinate**2
            + coefficient_b * normalized_line_coordinate
            + coefficient_c
        )
        integrated_strength = (
            polynomial_0 * moment_0
            + polynomial_1 * moment_1
            + polynomial_2 * moment_2
        )
        result[point_index] = (
            np.cross(tangent, transverse)
            * integrated_strength
        )
    if not np.all(np.isfinite(result)):
        raise DistributedDoubletError(
            "analytic P2 boundary-vortex field is non-finite"
        )
    return result


def _x_log_x(value: float) -> float:
    if value == 0.0:
        return 0.0
    if value < 0.0 or not np.isfinite(value):
        raise DistributedDoubletError(
            "finite-part logarithmic moment received an invalid argument"
        )
    return float(value * np.log(value))


def _finite_part_edge_moments(
    lower: float,
    upper: float,
    distance: float,
    length_reference: float,
) -> tuple[float, float, float, float, float]:
    """Exact I0/I1/I2 and logarithmic J0/J1 endpoint moments."""
    if (
        not np.isfinite(lower)
        or not np.isfinite(upper)
        or not np.isfinite(distance)
        or not np.isfinite(length_reference)
        or upper <= lower
        or distance <= 0.0
        or length_reference <= 0.0
    ):
        raise DistributedDoubletError(
            "finite-part edge moments require ordered finite geometry"
        )

    def primitive(value: float):
        radius = float(np.hypot(value, distance))
        ratio = float(value / radius)
        i0 = value / (distance**2 * radius)
        i1 = -1.0 / radius
        i2 = np.arcsinh(value / distance) - value / radius
        j0 = (
            ratio * np.log(distance / length_reference)
            + 0.5 * _x_log_x(1.0 - ratio)
            - 0.5 * _x_log_x(1.0 + ratio)
            + ratio
        ) / distance**2
        j1 = -(np.log(radius / length_reference) + 1.0) / radius
        return i0, i1, i2, j0, j1

    lower_value = primitive(lower)
    upper_value = primitive(upper)
    return tuple(
        float(end - start)
        for start, end in zip(lower_value, upper_value, strict=True)
    )


def _analytic_linear_area_vorticity_velocity(
    element: QuadraticDoubletElement,
    start_index: int,
    end_index: int,
    points: np.ndarray,
    gamma_at_points: np.ndarray,
    *,
    length_reference: float,
) -> np.ndarray:
    """Exact coplanar finite-part edge term for linear sheet vorticity."""
    if np.all(element.material_mu == element.material_mu[0]):
        return np.zeros_like(points)
    start = element.vertices[start_index]
    end = element.vertices[end_index]
    edge = end - start
    length = float(np.linalg.norm(edge))
    if not np.isfinite(length) or length <= np.finfo(float).tiny:
        raise DistributedDoubletError(
            "analytic area-vorticity term requires a nondegenerate edge"
        )
    tangent = edge / length
    edge_barycentric = element.barycentric_coordinates(
        np.vstack((start, end)),
        plane_tolerance=64.0 * np.finfo(float).eps,
    )
    gamma_start, gamma_end = element.sheet_vorticity_barycentric(
        edge_barycentric
    )
    gamma_gradient = (gamma_end - gamma_start) / length
    result = np.zeros_like(points)
    tolerance = 128.0 * np.finfo(float).eps * max(1.0, length)

    for point_index, (point, gamma_point) in enumerate(
        zip(points, gamma_at_points, strict=True)
    ):
        line_coordinate = float((point - start) @ tangent)
        radial_origin = start + line_coordinate * tangent - point
        distance = float(np.linalg.norm(radial_origin))
        if distance <= tolerance:
            if -tolerance <= line_coordinate <= length + tolerance:
                raise DistributedDoubletError(
                    "analytic area-vorticity term received a target on an edge"
                )
            # Collinear exterior points have zero signed fan Jacobian.
            continue
        moments = _finite_part_edge_moments(
            -line_coordinate,
            length - line_coordinate,
            distance,
            length_reference,
        )
        i0, i1, i2, j0, j1 = moments
        gamma_line = gamma_start + line_coordinate * gamma_gradient
        gamma_delta = gamma_line - gamma_point
        constant_0 = -np.cross(gamma_point, radial_origin)
        constant_1 = -np.cross(gamma_point, tangent)
        varying_0 = -np.cross(gamma_delta, radial_origin)
        varying_1 = -(
            np.cross(gamma_delta, tangent)
            + np.cross(gamma_gradient, radial_origin)
        )
        varying_2 = -np.cross(gamma_gradient, tangent)
        signed_fan = float(
            np.cross(radial_origin, tangent) @ element.normal
        )
        result[point_index] = signed_fan * (
            constant_0 * j0
            + constant_1 * j1
            + varying_0 * i0
            + varying_1 * i1
            + varying_2 * i2
        )
    if not np.all(np.isfinite(result)):
        raise DistributedDoubletError(
            "analytic area-vorticity finite-part field is non-finite"
        )
    return result


def _element_induced_velocity(
    element: QuadraticDoubletElement,
    points: np.ndarray,
    *,
    quadrature_order: int,
    singular_tolerance: float,
) -> np.ndarray:
    """Off-sheet numerical oracle for the DDE double-layer velocity.

    The sign and ``1/(4*pi)`` convention is fixed by the constant-strength
    identity: a positively oriented triangle is equivalent to the vortex ring
    ``v0->v1->v2->v0`` carrying the same ``mu``.
    """
    if singular_tolerance < 0.0 or not np.isfinite(singular_tolerance):
        raise DistributedDoubletError(
            "singular_tolerance must be finite and non-negative"
        )
    delta = points - element.vertices[0]
    normal_distance = delta @ element.normal
    projected = points - normal_distance[:, None] * element.normal
    projected_barycentric = element.barycentric_coordinates(
        projected,
        plane_tolerance=32.0 * np.finfo(float).eps,
    )
    length_scale = max(np.sqrt(2.0 * element.area), 1.0)
    near_plane = np.abs(normal_distance) <= singular_tolerance * length_scale
    inside_projection = np.all(
        projected_barycentric >= -singular_tolerance,
        axis=1,
    ) & np.all(
        projected_barycentric <= 1.0 + singular_tolerance,
        axis=1,
    )
    if np.any(near_plane & inside_projection):
        raise DistributedDoubletError(
            "on-sheet/principal-value DDE velocity is not implemented"
        )

    barycentric, reference_weight = _triangle_quadrature(quadrature_order)
    source = barycentric @ element.vertices
    strength = element.evaluate_barycentric(barycentric)
    physical_weight = reference_weight * np.linalg.norm(element.area_vector)
    separation = points[:, None, :] - source[None, :, :]
    radius_squared = np.einsum("pqj,pqj->pq", separation, separation)
    if np.any(radius_squared <= np.finfo(float).tiny):
        raise DistributedDoubletError(
            "field point coincides with a doublet quadrature point"
        )
    radius = np.sqrt(radius_squared)
    normal_projection = separation @ element.normal
    kernel = (
        element.normal[None, None, :] / radius[:, :, None] ** 3
        - 3.0
        * normal_projection[:, :, None]
        * separation
        / radius[:, :, None] ** 5
    )
    return (
        -np.einsum(
            "pqj,q,q->pj",
            kernel,
            strength,
            physical_weight,
        )
        / (4.0 * np.pi)
    )


def _element_sheet_average_velocity(
    element: QuadraticDoubletElement,
    barycentric: np.ndarray,
    *,
    quadrature_order: int,
    allow_exterior: bool = False,
    edge_quadrature: str = "standard",
) -> np.ndarray:
    """Johnson-equivalent in-plane velocity without a core or offset.

    The double layer is first rewritten exactly as its oriented boundary
    vortex plus the continuous area-vorticity sheet
    ``gamma=grad_s(mu) x n``.  The area term is evaluated as a Cauchy
    principal value by analytic radial finite-part cancellation on the three
    point-to-edge subtriangles.  No offset, core radius, or excluded finite
    disk appears in the operator.  Owner points must be strict interior
    points.  ``allow_exterior`` additionally admits coplanar points outside
    the element and retains the signed subtriangle Jacobian needed for that
    extension; points on an element edge remain inadmissible.
    """
    lam = _finite("barycentric", barycentric, ndim=2)
    if lam.shape[1] != 3:
        raise DistributedDoubletError(
            f"barycentric must have shape (n,3), got {lam.shape}"
        )
    strict_margin = 128.0 * np.finfo(float).eps
    if allow_exterior:
        if np.any(np.abs(lam) <= strict_margin):
            raise DistributedDoubletError(
                "in-plane exterior evaluation point lies on an element edge"
            )
    elif np.any(lam <= strict_margin) or np.any(
        lam >= 1.0 - strict_margin
    ):
        raise DistributedDoubletError(
            "sheet-average velocity requires strict element-interior points"
        )
    points = lam @ element.vertices
    gamma_at_point = element.sheet_vorticity_barycentric(lam)
    velocity = np.zeros_like(points)
    length_reference = np.sqrt(2.0 * element.area)

    for start_index, end_index in ((0, 1), (1, 2), (2, 0)):
        start = element.vertices[start_index]
        end = element.vertices[end_index]
        edge_vector = end - start
        analytic_boundary = None
        if edge_quadrature in (
            "target_sinh_analytic_boundary",
            "target_sinh_analytic_sheet",
        ):
            analytic_boundary = (
                _analytic_quadratic_boundary_vortex_velocity(
                    element,
                    start_index,
                    end_index,
                    points,
                )
            )
        analytic_area = None
        if edge_quadrature == "target_sinh_analytic_sheet":
            analytic_area = _analytic_linear_area_vorticity_velocity(
                element,
                start_index,
                end_index,
                points,
                gamma_at_point,
                length_reference=length_reference,
            )
        for point_index, point in enumerate(points):
            coordinate, weights = _edge_line_quadrature(
                start,
                end,
                point,
                order=int(quadrature_order),
                edge_quadrature=edge_quadrature,
            )
            source = (
                (1.0 - coordinate)[:, None] * start
                + coordinate[:, None] * end
            )
            edge_mu = element.evaluate(source)
            edge_gamma = element.sheet_vorticity_barycentric(
                element.barycentric_coordinates(
                    source,
                    plane_tolerance=64.0 * np.finfo(float).eps,
                )
            )
            radial = source - point
            radius = np.linalg.norm(radial, axis=1)
            if np.any(radius <= np.finfo(float).tiny):
                raise DistributedDoubletError(
                    "strict interior point unexpectedly lies on element edge"
                )

            # Oriented boundary-vortex contribution.  R=point-source.
            if analytic_boundary is None:
                boundary_kernel = (
                    np.cross(edge_vector[None, :], -radial)
                    / radius[:, None] ** 3
                )
                velocity[point_index] += np.einsum(
                    "qj,q->j",
                    boundary_kernel,
                    edge_mu * weights,
                )
            else:
                velocity[point_index] += analytic_boundary[point_index]

            # Cauchy principal value of the area-vorticity contribution.
            # The log term is the finite remainder of int(ds/s); its
            # arbitrary length reference cancels exactly over the closed
            # three-edge contour.
            radial_jacobian = (
                np.cross(radial, edge_vector) @ element.normal
            )
            factor = radial_jacobian / radius**3
            constant_term = -np.cross(
                gamma_at_point[point_index][None, :],
                radial,
            )
            varying_term = -np.cross(
                edge_gamma - gamma_at_point[point_index],
                radial,
            )
            if analytic_area is None:
                velocity[point_index] += np.einsum(
                    "qj,q->j",
                    constant_term,
                    factor
                    * np.log(radius / length_reference)
                    * weights,
                )
                velocity[point_index] += np.einsum(
                    "qj,q->j",
                    varying_term,
                    factor * weights,
                )
            else:
                velocity[point_index] += analytic_area[point_index]

    velocity /= 4.0 * np.pi
    if not np.all(np.isfinite(velocity)):
        raise DistributedDoubletError(
            "sheet-average DDE velocity contains non-finite values"
        )
    return velocity


def _element_nonowner_sheet_velocity(
    element: QuadraticDoubletElement,
    points: np.ndarray,
    *,
    quadrature_order: int,
    edge_quadrature: str = "standard",
) -> np.ndarray:
    """Contribution at sheet points not owned by ``element``.

    Coplanar exterior points use the same signed finite-part identity as the
    owner self term.  Genuinely off-plane points use the double-layer oracle.
    A coplanar point inside a non-owner element indicates overlapping
    topology and is rejected instead of being silently double booked.
    """
    point_array = _finite("points", points, ndim=2)
    delta = point_array - element.vertices[0]
    normal_distance = delta @ element.normal
    length_scale = max(np.sqrt(2.0 * element.area), 1.0)
    coplanar = (
        np.abs(normal_distance)
        <= 128.0 * np.finfo(float).eps * length_scale
    )
    velocity = np.zeros_like(point_array)
    if np.any(coplanar):
        projected_barycentric = element.barycentric_coordinates(
            point_array[coplanar],
            plane_tolerance=256.0 * np.finfo(float).eps * length_scale,
        )
        margin = 128.0 * np.finfo(float).eps
        inside = np.all(projected_barycentric >= -margin, axis=1) & np.all(
            projected_barycentric <= 1.0 + margin,
            axis=1,
        )
        if np.any(inside):
            raise DistributedDoubletError(
                "coplanar point lies inside a non-owner element; "
                "sheet topology overlaps or ownership is wrong"
            )
        velocity[coplanar] = _element_sheet_average_velocity(
            element,
            projected_barycentric,
            quadrature_order=quadrature_order,
            allow_exterior=True,
            edge_quadrature=edge_quadrature,
        )
    off_plane = ~coplanar
    if np.any(off_plane):
        velocity[off_plane] = _element_induced_velocity_line_reduced(
            element,
            point_array[off_plane],
            quadrature_order=quadrature_order,
            plane_tolerance=64.0 * np.finfo(float).eps,
            edge_quadrature=edge_quadrature,
        )
    return velocity


def _element_induced_velocity_line_reduced(
    element: QuadraticDoubletElement,
    points: np.ndarray,
    *,
    quadrature_order: int,
    plane_tolerance: float,
    edge_quadrature: str = "standard",
) -> np.ndarray:
    """Off-plane doublet velocity after exact radial integration.

    The same boundary-vortex plus linear area-vorticity identity used by the
    sheet finite part is parameterized as a signed fan from the target's
    in-plane projection to each panel edge.  All radial integrals are
    analytic; only smooth one-dimensional edge integrals remain.  This is an
    equation-equivalent CPU reference, not a port of Johnson's H/F recursion.
    """
    point_array = _finite("points", points, ndim=2)
    if point_array.shape[1] != 3:
        raise DistributedDoubletError(
            f"points must have shape (n,3), got {point_array.shape}"
        )
    if (
        not isinstance(quadrature_order, (int, np.integer))
        or quadrature_order < 2
    ):
        raise DistributedDoubletError(
            "quadrature_order must be an integer >=2"
        )
    if plane_tolerance < 0.0 or not np.isfinite(plane_tolerance):
        raise DistributedDoubletError(
            "plane_tolerance must be finite and non-negative"
        )
    normal_distance = (
        point_array - element.vertices[0]
    ) @ element.normal
    length_scale = max(np.sqrt(2.0 * element.area), 1.0)
    if np.any(
        np.abs(normal_distance) <= plane_tolerance * length_scale
    ):
        raise DistributedDoubletError(
            "line-reduced off-plane operator received an on-sheet point"
        )
    projected = (
        point_array - normal_distance[:, None] * element.normal
    )
    projected_barycentric = element.barycentric_coordinates(
        projected,
        plane_tolerance=4.0 * plane_tolerance,
    )
    gamma_at_projection = element.sheet_vorticity_barycentric(
        projected_barycentric
    )
    velocity = np.zeros_like(point_array)

    for start_index, end_index in ((0, 1), (1, 2), (2, 0)):
        start = element.vertices[start_index]
        end = element.vertices[end_index]
        edge_vector = end - start
        analytic_boundary = None
        if edge_quadrature in (
            "target_sinh_analytic_boundary",
            "target_sinh_analytic_sheet",
        ):
            analytic_boundary = (
                _analytic_quadratic_boundary_vortex_velocity(
                    element,
                    start_index,
                    end_index,
                    point_array,
                )
            )
        for point_index, point in enumerate(point_array):
            coordinate, weights = _edge_line_quadrature(
                start,
                end,
                point,
                order=int(quadrature_order),
                edge_quadrature=edge_quadrature,
            )
            source = (
                (1.0 - coordinate)[:, None] * start
                + coordinate[:, None] * end
            )
            edge_mu = element.evaluate(source)
            edge_gamma = element.sheet_vorticity_barycentric(
                element.barycentric_coordinates(
                    source,
                    plane_tolerance=64.0 * np.finfo(float).eps,
                )
            )
            # Oriented boundary vortex, integrated in the edge coordinate.
            separation = point - source
            separation_norm = np.linalg.norm(separation, axis=1)
            if analytic_boundary is None:
                velocity[point_index] += np.einsum(
                    "qj,q->j",
                    np.cross(edge_vector[None, :], separation)
                    / separation_norm[:, None] ** 3,
                    edge_mu * weights,
                )
            else:
                velocity[point_index] += analytic_boundary[point_index]

            radial = source - projected[point_index]
            radius = np.linalg.norm(radial, axis=1)
            if np.any(radius <= np.finfo(float).tiny):
                raise DistributedDoubletError(
                    "projected target lies on an element edge"
                )
            direction = radial / radius[:, None]
            signed_angle_jacobian = (
                np.cross(radial, edge_vector) @ element.normal
            ) / radius**2
            height = float(normal_distance[point_index])
            absolute_height = abs(height)
            distance = np.sqrt(radius**2 + absolute_height**2)

            # Exact radial primitives for gamma(rho)=gamma_0+rho*gamma_1.
            height_j1 = np.sign(height) * (
                1.0 - absolute_height / distance
            )
            j2 = np.arcsinh(radius / absolute_height) - radius / distance
            height_j2 = height * j2
            j4 = (
                distance
                + absolute_height**2 / distance
                - 2.0 * absolute_height
            )
            gamma0 = gamma_at_projection[point_index]
            gamma1 = (
                edge_gamma - gamma0[None, :]
            ) / radius[:, None]
            radial_integral = (
                np.cross(gamma0, element.normal)[None, :]
                * height_j1[:, None]
                - np.cross(gamma0[None, :], direction) * j2[:, None]
                + np.cross(gamma1, element.normal)
                * height_j2[:, None]
                - np.cross(gamma1, direction) * j4[:, None]
            )
            velocity[point_index] += np.einsum(
                "qj,q->j",
                radial_integral,
                signed_angle_jacobian * weights,
            )

    velocity /= 4.0 * np.pi
    if not np.all(np.isfinite(velocity)):
        raise DistributedDoubletError(
            "line-reduced DDE velocity contains non-finite values"
        )
    return velocity


class QuadraticDoubletSurface:
    """Indexed triangular P2 material doublet surface.

    ``face_mu`` uses local order ``(v0,v1,v2,e01,e12,e20)``.  Keeping the
    face-local values explicit makes discontinuities observable rather than
    silently averaging them away.
    """

    _LOCAL_EDGES = (
        (0, 1, 3),
        (1, 2, 4),
        (2, 0, 5),
    )

    def __init__(self, vertices, faces, face_mu):
        vertex_array = _finite("vertices", vertices, ndim=2)
        if vertex_array.shape[1] != 3:
            raise DistributedDoubletError(
                f"vertices must have shape (n,3), got {vertex_array.shape}"
            )
        face_array = np.asarray(faces, dtype=np.int64)
        if face_array.ndim != 2 or face_array.shape[1] != 3:
            raise DistributedDoubletError(
                f"faces must have shape (m,3), got {face_array.shape}"
            )
        if np.any(face_array < 0) or np.any(face_array >= len(vertex_array)):
            raise DistributedDoubletError("faces contain invalid vertex indices")
        if np.any(
            (face_array[:, 0] == face_array[:, 1])
            | (face_array[:, 1] == face_array[:, 2])
            | (face_array[:, 2] == face_array[:, 0])
        ):
            raise DistributedDoubletError("face repeats a vertex")
        mu_array = _finite(
            "face_mu",
            face_mu,
            shape=(len(face_array), 6),
        )
        self.vertices = vertex_array.copy()
        self.faces = face_array.copy()
        self.face_mu = mu_array.copy()
        for face_index in range(len(self.faces)):
            self.element(face_index)

    def __len__(self) -> int:
        return len(self.faces)

    def element(self, face_index: int) -> QuadraticDoubletElement:
        return QuadraticDoubletElement(
            self.vertices[self.faces[face_index]],
            self.face_mu[face_index],
        )

    def _edge_records(self):
        records: dict[tuple[int, int], list[tuple[int, np.ndarray]]] = {}
        for face_index, face in enumerate(self.faces):
            mu = self.face_mu[face_index]
            for local_a, local_b, local_mid in self._LOCAL_EDGES:
                global_a = int(face[local_a])
                global_b = int(face[local_b])
                if global_a < global_b:
                    key = (global_a, global_b)
                    trace = np.array(
                        [mu[local_a], mu[local_mid], mu[local_b]]
                    )
                else:
                    key = (global_b, global_a)
                    trace = np.array(
                        [mu[local_b], mu[local_mid], mu[local_a]]
                    )
                records.setdefault(key, []).append((face_index, trace))
        return records

    def boundary_edge_traces(self) -> dict[tuple[int, int], np.ndarray]:
        """Return P2 traces on topological boundary edges.

        Traces are ordered by ascending global vertex index.  This is a
        topological query only; whether an edge is a physical zero-strength
        boundary or a seam coupled to another patch is decided by
        :class:`QuadraticDoubletAssembly`.
        """
        result = {}
        for edge, records in self._edge_records().items():
            if len(records) == 1:
                result[edge] = records[0][1].copy()
            elif len(records) > 2:
                raise DistributedDoubletError(
                    f"non-manifold edge {edge} has {len(records)} faces"
                )
        return result

    def _boundary_edge_data(
        self,
    ) -> dict[tuple[int, int], tuple[np.ndarray, np.ndarray]]:
        """Return geometry and trace in the oriented face-boundary direction."""
        boundary = set(self.boundary_edge_traces())
        result = {}
        for face_index, face in enumerate(self.faces):
            mu = self.face_mu[face_index]
            for local_a, local_b, local_mid in self._LOCAL_EDGES:
                global_a = int(face[local_a])
                global_b = int(face[local_b])
                key = tuple(sorted((global_a, global_b)))
                if key not in boundary:
                    continue
                geometry = np.stack(
                    (
                        self.vertices[global_a],
                        0.5
                        * (
                            self.vertices[global_a]
                            + self.vertices[global_b]
                        ),
                        self.vertices[global_b],
                    )
                )
                trace = np.array(
                    (mu[local_a], mu[local_mid], mu[local_b])
                )
                result[key] = (geometry, trace)
        return result

    def boundary_report(
        self,
        *,
        tolerance: float = 1.0e-12,
    ) -> DoubletBoundaryReport:
        """Test the bounded-field condition for an isolated DDE patch.

        A non-zero potential jump on an unmatched edge is an exposed vortex
        filament.  Such an edge is admissible only when an assembly explicitly
        pairs it with the oppositely oriented edge of another continuous
        patch; an isolated patch must therefore have zero trace everywhere on
        its boundary.
        """
        if tolerance < 0.0 or not np.isfinite(tolerance):
            raise DistributedDoubletError(
                "tolerance must be finite and non-negative"
            )
        traces = self.boundary_edge_traces()
        max_trace = max(
            (float(np.max(np.abs(trace))) for trace in traces.values()),
            default=0.0,
        )
        return DoubletBoundaryReport(
            boundary_edges=len(traces),
            max_boundary_trace=max_trace,
            compatible=max_trace <= tolerance,
        )

    def continuity_report(
        self,
        *,
        tolerance: float = 1.0e-12,
    ) -> DoubletContinuityReport:
        if tolerance < 0.0 or not np.isfinite(tolerance):
            raise DistributedDoubletError(
                "tolerance must be finite and non-negative"
            )
        internal_edges = 0
        boundary_edges = 0
        max_node_jump = 0.0
        max_trace_jump = 0.0
        sample = np.linspace(0.0, 1.0, 9)
        edge_basis = np.column_stack(
            (
                (1.0 - sample) * (1.0 - 2.0 * sample),
                4.0 * sample * (1.0 - sample),
                sample * (2.0 * sample - 1.0),
            )
        )
        for key, records in self._edge_records().items():
            if len(records) == 1:
                boundary_edges += 1
                continue
            if len(records) != 2:
                raise DistributedDoubletError(
                    f"non-manifold edge {key} has {len(records)} faces"
                )
            internal_edges += 1
            first = records[0][1]
            second = records[1][1]
            max_node_jump = max(
                max_node_jump,
                float(np.max(np.abs(first - second))),
            )
            max_trace_jump = max(
                max_trace_jump,
                float(np.max(np.abs(edge_basis @ (first - second)))),
            )
        return DoubletContinuityReport(
            internal_edges=internal_edges,
            boundary_edges=boundary_edges,
            max_trace_node_jump=max_node_jump,
            max_trace_jump=max_trace_jump,
            compatible=max(max_node_jump, max_trace_jump) <= tolerance,
        )

    def coplanar_midpoint_vorticity_report(
        self,
        *,
        strength_tolerance: float = 1.0e-12,
        normal_tolerance: float = 1.0e-12,
    ) -> DoubletVorticityContinuityReport:
        """Check Krebs' edge-midpoint vorticity constraint on a flat sheet.

        This deliberately has a narrow identity.  Direct equality of global
        tangent vectors is meaningful for coplanar neighbouring faces; a
        curved-sheet parallel-transport rule is a separate, still-open claim.
        """
        if (
            strength_tolerance < 0.0
            or normal_tolerance < 0.0
            or not np.isfinite(strength_tolerance)
            or not np.isfinite(normal_tolerance)
        ):
            raise DistributedDoubletError(
                "vorticity tolerances must be finite and non-negative"
            )
        internal_edges = 0
        maximum_jump = 0.0
        coplanar = True
        for edge, records in self._edge_records().items():
            if len(records) == 1:
                continue
            if len(records) != 2:
                raise DistributedDoubletError(
                    f"non-manifold edge {edge} has {len(records)} faces"
                )
            internal_edges += 1
            values = []
            normals = []
            for face_index, _ in records:
                face = self.faces[face_index]
                local = [
                    int(np.flatnonzero(face == vertex)[0])
                    for vertex in edge
                ]
                barycentric = np.zeros((1, 3))
                barycentric[0, local] = 0.5
                element = self.element(face_index)
                values.append(
                    element.sheet_vorticity_barycentric(
                        barycentric
                    )[0]
                )
                normals.append(element.normal)
            normal_alignment = abs(float(np.dot(normals[0], normals[1])))
            edge_coplanar = (
                1.0 - normal_alignment <= normal_tolerance
            )
            coplanar = coplanar and edge_coplanar
            if edge_coplanar:
                maximum_jump = max(
                    maximum_jump,
                    float(np.linalg.norm(values[0] - values[1])),
                )
            else:
                maximum_jump = np.inf
        return DoubletVorticityContinuityReport(
            internal_edges=internal_edges,
            max_midpoint_vorticity_jump=maximum_jump,
            coplanar=coplanar,
            compatible=coplanar
            and maximum_jump <= strength_tolerance,
        )

    def material_update(self, vertices) -> "QuadraticDoubletSurface":
        """Move material vertices while preserving topology and strength."""
        return QuadraticDoubletSurface(vertices, self.faces, self.face_mu)

    def kelvin_report(
        self,
        other: "QuadraticDoubletSurface",
        *,
        tolerance: float = 1.0e-12,
    ) -> MaterialKelvinReport:
        if tolerance < 0.0 or not np.isfinite(tolerance):
            raise DistributedDoubletError(
                "tolerance must be finite and non-negative"
            )
        topology_equal = np.array_equal(self.faces, other.faces)
        if self.face_mu.shape != other.face_mu.shape:
            residual = np.inf
        else:
            residual = float(
                np.max(np.abs(self.face_mu - other.face_mu), initial=0.0)
            )
        return MaterialKelvinReport(
            max_material_mu_residual=residual,
            topology_equal=topology_equal,
            passed=topology_equal and residual <= tolerance,
        )

    def interior_collocation_points(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Krebs' four strict-interior DDE collocation points per face.

        The points are the centroid and the 90% locations from the centroid
        toward each vertex.  Returned arrays are ``(points, face_index,
        barycentric)``.
        """
        centroid = np.full(3, 1.0 / 3.0)
        local = [centroid]
        for vertex_index in range(3):
            bary = 0.1 * centroid
            bary = bary.copy()
            bary[vertex_index] += 0.9
            local.append(bary)
        local_barycentric = np.asarray(local)
        points = []
        face_indices = []
        barycentric = []
        for face_index in range(len(self)):
            element = self.element(face_index)
            points.append(local_barycentric @ element.vertices)
            face_indices.extend([face_index] * len(local_barycentric))
            barycentric.append(local_barycentric)
        return (
            np.vstack(points),
            np.asarray(face_indices, dtype=np.int64),
            np.vstack(barycentric),
        )

    def induced_velocity_sheet_average(
        self,
        face_indices,
        barycentric,
        *,
        quadrature_order: int = 24,
        edge_quadrature: str = "standard",
    ) -> np.ndarray:
        """Evaluate the global sheet-average velocity at owned face points.

        The owner face uses the exact principal-value decomposition.
        Coplanar non-owner faces use its signed exterior continuation, while
        genuinely off-plane faces use the ordinary double-layer field.  This
        is the parameter-free geometric velocity of the sheet.  Tangential
        reparameterization may be chosen separately and must not be confused
        with a physical velocity jump.
        """
        owner = np.asarray(face_indices, dtype=np.int64)
        lam = _finite("barycentric", barycentric, ndim=2)
        if owner.ndim != 1 or lam.shape != (len(owner), 3):
            raise DistributedDoubletError(
                "face_indices and barycentric must have shapes (n,) and (n,3)"
            )
        if np.any(owner < 0) or np.any(owner >= len(self)):
            raise DistributedDoubletError("face_indices contain invalid owners")
        points = np.empty((len(owner), 3), dtype=float)
        for point_index, face_index in enumerate(owner):
            points[point_index] = (
                lam[point_index] @ self.vertices[self.faces[face_index]]
            )
        velocity = np.zeros_like(points)
        for source_face in range(len(self)):
            source = self.element(source_face)
            self_mask = owner == source_face
            if np.any(self_mask):
                velocity[self_mask] += _element_sheet_average_velocity(
                    source,
                    lam[self_mask],
                    quadrature_order=quadrature_order,
                    edge_quadrature=edge_quadrature,
                )
            other_mask = ~self_mask
            if np.any(other_mask):
                velocity[other_mask] += _element_nonowner_sheet_velocity(
                    source,
                    points[other_mask],
                    quadrature_order=quadrature_order,
                    edge_quadrature=edge_quadrature,
                )
        if not np.all(np.isfinite(velocity)):
            raise DistributedDoubletError(
                "global sheet-average velocity contains non-finite values"
            )
        return velocity

    def induced_velocity_sheet_average_converged(
        self,
        face_indices,
        barycentric,
        *,
        orders: tuple[int, ...] = (24, 36, 48, 64, 80),
        absolute_tolerance: float = 1.0e-10,
        relative_tolerance: float = 1.0e-8,
        edge_quadrature: str = "standard",
    ) -> tuple[np.ndarray, SheetAverageVelocityReport]:
        if len(orders) < 2 or any(
            (not isinstance(order, (int, np.integer)) or order < 2)
            for order in orders
        ):
            raise DistributedDoubletError(
                "orders must contain at least two integer values >=2"
            )
        if any(second <= first for first, second in zip(orders, orders[1:])):
            raise DistributedDoubletError("orders must be strictly increasing")
        if (
            absolute_tolerance < 0.0
            or relative_tolerance < 0.0
            or not np.isfinite(absolute_tolerance)
            or not np.isfinite(relative_tolerance)
        ):
            raise DistributedDoubletError(
                "sheet-average tolerances must be finite and non-negative"
            )
        previous = self.induced_velocity_sheet_average(
            face_indices,
            barycentric,
            quadrature_order=orders[0],
            edge_quadrature=edge_quadrature,
        )
        last_abs = np.inf
        last_rel = np.inf
        for order in orders[1:]:
            current = self.induced_velocity_sheet_average(
                face_indices,
                barycentric,
                quadrature_order=order,
                edge_quadrature=edge_quadrature,
            )
            difference = np.linalg.norm(current - previous, axis=1)
            scale = np.maximum(
                np.linalg.norm(current, axis=1),
                absolute_tolerance,
            )
            last_abs = float(np.max(difference, initial=0.0))
            last_rel = float(np.max(difference / scale, initial=0.0))
            converged = (
                last_abs <= absolute_tolerance
                or last_rel <= relative_tolerance
            )
            if converged:
                return current, SheetAverageVelocityReport(
                    quadrature_order=int(order),
                    max_abs_change=last_abs,
                    max_rel_change=last_rel,
                    converged=True,
                )
            previous = current
        return previous, SheetAverageVelocityReport(
            quadrature_order=int(orders[-1]),
            max_abs_change=last_abs,
            max_rel_change=last_rel,
            converged=False,
        )

    def induced_velocity_nonowner_sheet_points(
        self,
        points,
        *,
        quadrature_order: int = 24,
        edge_quadrature: str = "standard",
    ) -> np.ndarray:
        """Evaluate this surface at sheet points owned by another patch."""
        point_array = _finite("points", points, ndim=2)
        if point_array.shape[1] != 3:
            raise DistributedDoubletError(
                f"points must have shape (n,3), got {point_array.shape}"
            )
        velocity = np.zeros_like(point_array)
        for source_face in range(len(self)):
            velocity += _element_nonowner_sheet_velocity(
                self.element(source_face),
                point_array,
                quadrature_order=quadrature_order,
                edge_quadrature=edge_quadrature,
            )
        if not np.all(np.isfinite(velocity)):
            raise DistributedDoubletError(
                "non-owner sheet velocity contains non-finite values"
            )
        return velocity

    def induced_velocity(
        self,
        points,
        *,
        quadrature_order: int = 16,
        singular_tolerance: float = 1.0e-10,
    ) -> np.ndarray:
        """Evaluate the off-sheet distributed-doublet velocity.

        This is a numerical equation oracle, not the production wake-rollup
        kernel.  It intentionally rejects points on a source element until a
        documented principal-value/self-influence operator is implemented.
        """
        point_array = _finite("points", points, ndim=2)
        if point_array.shape[1] != 3:
            raise DistributedDoubletError(
                f"points must have shape (n,3), got {point_array.shape}"
            )
        velocity = np.zeros_like(point_array)
        for face_index in range(len(self)):
            velocity += _element_induced_velocity(
                self.element(face_index),
                point_array,
                quadrature_order=quadrature_order,
                singular_tolerance=singular_tolerance,
            )
        if not np.all(np.isfinite(velocity)):
            raise DistributedDoubletError(
                "distributed-doublet velocity contains non-finite values"
            )
        return velocity

    def induced_velocity_line_reduced(
        self,
        points,
        *,
        quadrature_order: int = 32,
        plane_tolerance: float = 128.0 * np.finfo(float).eps,
        edge_quadrature: str = "standard",
    ) -> np.ndarray:
        """Evaluate the off-plane field with analytic radial integration."""
        point_array = _finite("points", points, ndim=2)
        if point_array.shape[1] != 3:
            raise DistributedDoubletError(
                f"points must have shape (n,3), got {point_array.shape}"
            )
        velocity = np.zeros_like(point_array)
        for face_index in range(len(self)):
            velocity += _element_induced_velocity_line_reduced(
                self.element(face_index),
                point_array,
                quadrature_order=quadrature_order,
                plane_tolerance=plane_tolerance,
                edge_quadrature=edge_quadrature,
            )
        if not np.all(np.isfinite(velocity)):
            raise DistributedDoubletError(
                "surface line-reduced velocity contains non-finite values"
            )
        return velocity

    def induced_velocity_line_reduced_converged(
        self,
        points,
        *,
        orders: tuple[int, ...] = (16, 24, 36, 52),
        absolute_tolerance: float = 1.0e-10,
        relative_tolerance: float = 1.0e-8,
        plane_tolerance: float = 128.0 * np.finfo(float).eps,
        edge_quadrature: str = "standard",
    ) -> tuple[np.ndarray, DoubletVelocityReport]:
        """Convergence gate for the analytic-radial/edge-quadrature field."""
        if len(orders) < 2 or any(
            (not isinstance(order, (int, np.integer)) or order < 2)
            for order in orders
        ):
            raise DistributedDoubletError(
                "orders must contain at least two integer values >=2"
            )
        if any(second <= first for first, second in zip(orders, orders[1:])):
            raise DistributedDoubletError(
                "orders must be strictly increasing"
            )
        if (
            absolute_tolerance < 0.0
            or relative_tolerance < 0.0
            or not np.isfinite(absolute_tolerance)
            or not np.isfinite(relative_tolerance)
        ):
            raise DistributedDoubletError(
                "line-reduced tolerances must be finite and non-negative"
            )
        previous = self.induced_velocity_line_reduced(
            points,
            quadrature_order=orders[0],
            plane_tolerance=plane_tolerance,
            edge_quadrature=edge_quadrature,
        )
        last_abs = np.inf
        last_rel = np.inf
        for order in orders[1:]:
            current = self.induced_velocity_line_reduced(
                points,
                quadrature_order=order,
                plane_tolerance=plane_tolerance,
                edge_quadrature=edge_quadrature,
            )
            difference = np.linalg.norm(current - previous, axis=1)
            scale = np.maximum(
                np.linalg.norm(current, axis=1),
                absolute_tolerance,
            )
            last_abs = float(np.max(difference, initial=0.0))
            last_rel = float(
                np.max(difference / scale, initial=0.0)
            )
            if (
                last_abs <= absolute_tolerance
                or last_rel <= relative_tolerance
            ):
                return current, DoubletVelocityReport(
                    quadrature_order=int(order),
                    max_abs_change=last_abs,
                    max_rel_change=last_rel,
                    converged=True,
                )
            previous = current
        return previous, DoubletVelocityReport(
            quadrature_order=int(orders[-1]),
            max_abs_change=last_abs,
            max_rel_change=last_rel,
            converged=False,
        )

    def induced_velocity_converged(
        self,
        points,
        *,
        orders: tuple[int, ...] = (8, 12, 16, 24),
        absolute_tolerance: float = 1.0e-10,
        relative_tolerance: float = 1.0e-8,
        singular_tolerance: float = 1.0e-10,
    ) -> tuple[np.ndarray, DoubletVelocityReport]:
        """Increase quadrature order until the off-sheet field converges."""
        if len(orders) < 2 or any(
            (not isinstance(order, (int, np.integer)) or order < 2)
            for order in orders
        ):
            raise DistributedDoubletError(
                "orders must contain at least two integer values >=2"
            )
        if any(second <= first for first, second in zip(orders, orders[1:])):
            raise DistributedDoubletError("orders must be strictly increasing")
        if (
            absolute_tolerance < 0.0
            or relative_tolerance < 0.0
            or not np.isfinite(absolute_tolerance)
            or not np.isfinite(relative_tolerance)
        ):
            raise DistributedDoubletError(
                "quadrature tolerances must be finite and non-negative"
            )
        previous = self.induced_velocity(
            points,
            quadrature_order=orders[0],
            singular_tolerance=singular_tolerance,
        )
        last_abs = np.inf
        last_rel = np.inf
        for order in orders[1:]:
            current = self.induced_velocity(
                points,
                quadrature_order=order,
                singular_tolerance=singular_tolerance,
            )
            difference = np.linalg.norm(current - previous, axis=1)
            scale = np.maximum(
                np.linalg.norm(current, axis=1),
                absolute_tolerance,
            )
            last_abs = float(np.max(difference, initial=0.0))
            last_rel = float(np.max(difference / scale, initial=0.0))
            converged = (
                last_abs <= absolute_tolerance
                or last_rel <= relative_tolerance
            )
            if converged:
                return current, DoubletVelocityReport(
                    quadrature_order=int(order),
                    max_abs_change=last_abs,
                    max_rel_change=last_rel,
                    converged=True,
                )
            previous = current
        return previous, DoubletVelocityReport(
            quadrature_order=int(orders[-1]),
            max_abs_change=last_abs,
            max_rel_change=last_rel,
            converged=False,
        )


@dataclass(frozen=True)
class QuadraticDoubletPatch:
    """One named DDE patch with every exposed edge assigned a physical role.

    ``boundary_roles`` maps the patch-local, sorted vertex-index edge to
    either ``"zero"`` or ``"interface:<stable-id>"``.  Requiring complete
    classification prevents a non-zero edge filament from being silently
    accepted as a numerical detail.
    """

    name: str
    surface: QuadraticDoubletSurface
    boundary_roles: Mapping[tuple[int, int], str]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise DistributedDoubletError("patch name must be non-empty")
        if not isinstance(self.surface, QuadraticDoubletSurface):
            raise DistributedDoubletError(
                "patch surface must be QuadraticDoubletSurface"
            )
        roles = {
            tuple(sorted((int(edge[0]), int(edge[1])))): str(role)
            for edge, role in dict(self.boundary_roles).items()
        }
        actual = set(self.surface.boundary_edge_traces())
        provided = set(roles)
        missing = sorted(actual - provided)
        extra = sorted(provided - actual)
        if missing or extra:
            raise DistributedDoubletError(
                f"patch {self.name!r} boundary classification mismatch: "
                f"missing={missing}, extra={extra}"
            )
        for edge, role in roles.items():
            if role != "zero" and not role.startswith("interface:"):
                raise DistributedDoubletError(
                    f"patch {self.name!r} edge {edge} has invalid role {role!r}"
                )
            if role.startswith("interface:") and not role.removeprefix(
                "interface:"
            ):
                raise DistributedDoubletError(
                    f"patch {self.name!r} edge {edge} has empty interface id"
                )
        object.__setattr__(self, "boundary_roles", roles)


class QuadraticDoubletAssembly:
    """Global continuous-potential-jump topology for bound and free sheets."""

    def __init__(self, patches):
        patch_tuple = tuple(patches)
        if not patch_tuple:
            raise DistributedDoubletError(
                "doublet assembly requires at least one patch"
            )
        if any(not isinstance(patch, QuadraticDoubletPatch) for patch in patch_tuple):
            raise DistributedDoubletError(
                "assembly entries must be QuadraticDoubletPatch"
            )
        names = [patch.name for patch in patch_tuple]
        if len(set(names)) != len(names):
            raise DistributedDoubletError("assembly patch names must be unique")
        self.patches = patch_tuple

    def topology_report(
        self,
        *,
        strength_tolerance: float = 1.0e-12,
        geometry_tolerance: float = 1.0e-12,
    ) -> DoubletAssemblyReport:
        if (
            strength_tolerance < 0.0
            or geometry_tolerance < 0.0
            or not np.isfinite(strength_tolerance)
            or not np.isfinite(geometry_tolerance)
        ):
            raise DistributedDoubletError(
                "assembly tolerances must be finite and non-negative"
            )
        interfaces: dict[
            str,
            list[
                tuple[
                    QuadraticDoubletPatch,
                    tuple[int, int],
                    np.ndarray,
                    np.ndarray,
                ]
            ],
        ] = {}
        zero_boundaries = 0
        max_zero = 0.0
        for patch in self.patches:
            continuity = patch.surface.continuity_report(
                tolerance=strength_tolerance
            )
            if not continuity.compatible:
                return DoubletAssemblyReport(
                    patches=len(self.patches),
                    zero_boundaries=zero_boundaries,
                    coupled_interfaces=0,
                    max_zero_trace=max_zero,
                    max_interface_trace_jump=np.inf,
                    max_interface_geometry_gap=np.inf,
                    compatible=False,
                )
            traces = patch.surface.boundary_edge_traces()
            edge_data = patch.surface._boundary_edge_data()
            for edge, trace in traces.items():
                role = patch.boundary_roles[edge]
                if role == "zero":
                    zero_boundaries += 1
                    max_zero = max(max_zero, float(np.max(np.abs(trace))))
                    continue
                interface_id = role.removeprefix("interface:")
                interfaces.setdefault(interface_id, []).append(
                    (
                        patch,
                        edge,
                        edge_data[edge][1],
                        edge_data[edge][0],
                    )
                )

        max_trace_jump = 0.0
        max_geometry_gap = 0.0
        compatible = max_zero <= strength_tolerance
        for interface_id, records in interfaces.items():
            if len(records) != 2:
                raise DistributedDoubletError(
                    f"interface {interface_id!r} has {len(records)} sides; "
                    "exactly two are required"
                )
            first, second = records
            first_geometry = first[3]
            second_geometry = second[3]
            reverse_gap = float(
                np.max(
                    np.linalg.norm(
                        first_geometry - second_geometry[::-1],
                        axis=1,
                    )
                )
            )
            direct_gap = float(
                np.max(
                    np.linalg.norm(
                        first_geometry - second_geometry,
                        axis=1,
                    )
                )
            )
            if direct_gap <= reverse_gap:
                geometry_gap = direct_gap
                second_trace = second[2]
            else:
                geometry_gap = reverse_gap
                second_trace = second[2][::-1]
            trace_jump = float(np.max(np.abs(first[2] - second_trace)))
            max_trace_jump = max(max_trace_jump, trace_jump)
            max_geometry_gap = max(max_geometry_gap, geometry_gap)
            compatible = (
                compatible
                and trace_jump <= strength_tolerance
                and geometry_gap <= geometry_tolerance
            )
        return DoubletAssemblyReport(
            patches=len(self.patches),
            zero_boundaries=zero_boundaries,
            coupled_interfaces=len(interfaces),
            max_zero_trace=max_zero,
            max_interface_trace_jump=max_trace_jump,
            max_interface_geometry_gap=max_geometry_gap,
            compatible=compatible,
        )

    def interior_collocation_points(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return Krebs collocation points and their patch/face ownership."""
        points = []
        patch_indices = []
        face_indices = []
        barycentric = []
        for patch_index, patch in enumerate(self.patches):
            patch_points, patch_faces, patch_barycentric = (
                patch.surface.interior_collocation_points()
            )
            points.append(patch_points)
            patch_indices.extend([patch_index] * len(patch_points))
            face_indices.append(patch_faces)
            barycentric.append(patch_barycentric)
        return (
            np.vstack(points),
            np.asarray(patch_indices, dtype=np.int64),
            np.concatenate(face_indices),
            np.vstack(barycentric),
        )

    def induced_velocity_sheet_average(
        self,
        patch_indices,
        face_indices,
        barycentric,
        *,
        quadrature_order: int = 24,
        edge_quadrature: str = "standard",
    ) -> np.ndarray:
        """Global sheet-average velocity across all explicitly coupled patches."""
        owner_patch = np.asarray(patch_indices, dtype=np.int64)
        owner_face = np.asarray(face_indices, dtype=np.int64)
        lam = _finite("barycentric", barycentric, ndim=2)
        if (
            owner_patch.ndim != 1
            or owner_face.ndim != 1
            or len(owner_patch) != len(owner_face)
            or lam.shape != (len(owner_patch), 3)
        ):
            raise DistributedDoubletError(
                "patch_indices, face_indices, barycentric have incompatible shapes"
            )
        if np.any(owner_patch < 0) or np.any(owner_patch >= len(self.patches)):
            raise DistributedDoubletError(
                "patch_indices contain invalid owners"
            )
        points = np.empty((len(owner_patch), 3), dtype=float)
        for point_index, (patch_index, face_index) in enumerate(
            zip(owner_patch, owner_face)
        ):
            surface = self.patches[int(patch_index)].surface
            if face_index < 0 or face_index >= len(surface):
                raise DistributedDoubletError(
                    "face_indices contain invalid owners"
                )
            points[point_index] = (
                lam[point_index]
                @ surface.vertices[surface.faces[int(face_index)]]
            )

        velocity = np.zeros_like(points)
        for patch_index, patch in enumerate(self.patches):
            self_mask = owner_patch == patch_index
            if np.any(self_mask):
                velocity[self_mask] += (
                    patch.surface.induced_velocity_sheet_average(
                        owner_face[self_mask],
                        lam[self_mask],
                        quadrature_order=quadrature_order,
                        edge_quadrature=edge_quadrature,
                    )
                )
            other_mask = ~self_mask
            if np.any(other_mask):
                velocity[other_mask] += (
                    patch.surface.induced_velocity_nonowner_sheet_points(
                        points[other_mask],
                        quadrature_order=quadrature_order,
                        edge_quadrature=edge_quadrature,
                    )
                )
        if not np.all(np.isfinite(velocity)):
            raise DistributedDoubletError(
                "assembled sheet-average velocity contains non-finite values"
            )
        return velocity

    def induced_velocity_sheet_average_converged(
        self,
        patch_indices,
        face_indices,
        barycentric,
        *,
        orders: tuple[int, ...] = (24, 36, 48, 64, 80),
        absolute_tolerance: float = 1.0e-10,
        relative_tolerance: float = 1.0e-8,
        edge_quadrature: str = "standard",
    ) -> tuple[np.ndarray, SheetAverageVelocityReport]:
        if len(orders) < 2 or any(
            (not isinstance(order, (int, np.integer)) or order < 2)
            for order in orders
        ):
            raise DistributedDoubletError(
                "orders must contain at least two integer values >=2"
            )
        if any(second <= first for first, second in zip(orders, orders[1:])):
            raise DistributedDoubletError("orders must be strictly increasing")
        if (
            absolute_tolerance < 0.0
            or relative_tolerance < 0.0
            or not np.isfinite(absolute_tolerance)
            or not np.isfinite(relative_tolerance)
        ):
            raise DistributedDoubletError(
                "sheet-average tolerances must be finite and non-negative"
            )
        previous = self.induced_velocity_sheet_average(
            patch_indices,
            face_indices,
            barycentric,
            quadrature_order=orders[0],
            edge_quadrature=edge_quadrature,
        )
        last_abs = np.inf
        last_rel = np.inf
        for order in orders[1:]:
            current = self.induced_velocity_sheet_average(
                patch_indices,
                face_indices,
                barycentric,
                quadrature_order=order,
                edge_quadrature=edge_quadrature,
            )
            difference = np.linalg.norm(current - previous, axis=1)
            scale = np.maximum(
                np.linalg.norm(current, axis=1),
                absolute_tolerance,
            )
            last_abs = float(np.max(difference, initial=0.0))
            last_rel = float(np.max(difference / scale, initial=0.0))
            converged = (
                last_abs <= absolute_tolerance
                or last_rel <= relative_tolerance
            )
            if converged:
                return current, SheetAverageVelocityReport(
                    quadrature_order=int(order),
                    max_abs_change=last_abs,
                    max_rel_change=last_rel,
                    converged=True,
                )
            previous = current
        return previous, SheetAverageVelocityReport(
            quadrature_order=int(orders[-1]),
            max_abs_change=last_abs,
            max_rel_change=last_rel,
            converged=False,
        )


def _quadratic_line_basis(coordinate: np.ndarray) -> np.ndarray:
    """Lagrange basis at coordinates ``(0, 1/2, 1)``."""
    value = np.asarray(coordinate, dtype=float)
    return np.stack(
        (
            2.0 * (value - 0.5) * (value - 1.0),
            4.0 * value * (1.0 - value),
            2.0 * value * (value - 0.5),
        ),
        axis=-1,
    )


@dataclass(frozen=True)
class MaterialWakeBand:
    """One newborn material strip between two consecutive shedding edges.

    The P2 potential jump is sampled on a tensor product of three temporal
    nodes and the spanwise vertex/midside nodes.  Requiring the middle
    temporal row explicitly avoids inventing an implicit linear shedding law.
    Once born, ``material_update`` changes geometry only.
    """

    sheet_id: str
    vortex_family: str
    time_nodes: np.ndarray
    span_nodes: int
    surface: QuadraticDoubletSurface
    potential_jump_rows: np.ndarray

    _FAMILIES = frozenset(("TEV", "LEV_SUCTION", "LEV_PRESSURE"))

    def __post_init__(self) -> None:
        if not isinstance(self.sheet_id, str) or not self.sheet_id:
            raise DistributedDoubletError("sheet_id must be non-empty")
        if self.vortex_family not in self._FAMILIES:
            raise DistributedDoubletError(
                f"vortex_family must be one of {sorted(self._FAMILIES)}, "
                f"got {self.vortex_family!r}"
            )
        time_nodes = _finite("time_nodes", self.time_nodes, shape=(3,))
        if not (time_nodes[0] < time_nodes[1] < time_nodes[2]):
            raise DistributedDoubletError(
                "time_nodes must be strictly increasing"
            )
        time_scale = max(
            abs(float(time_nodes[0])),
            abs(float(time_nodes[2])),
            float(time_nodes[2] - time_nodes[0]),
            1.0,
        )
        if abs(
            float(time_nodes[1])
            - 0.5 * float(time_nodes[0] + time_nodes[2])
        ) > 64.0 * np.finfo(float).eps * time_scale:
            raise DistributedDoubletError(
                "time_nodes middle entry must be the explicit half-time"
            )
        if not isinstance(self.span_nodes, (int, np.integer)) or self.span_nodes < 2:
            raise DistributedDoubletError("span_nodes must be an integer >=2")
        if not isinstance(self.surface, QuadraticDoubletSurface):
            raise DistributedDoubletError(
                "wake-band surface must be QuadraticDoubletSurface"
            )
        expected_vertices = 2 * int(self.span_nodes)
        expected_faces = 2 * (int(self.span_nodes) - 1)
        if self.surface.vertices.shape != (expected_vertices, 3):
            raise DistributedDoubletError(
                "wake-band surface vertex count does not match span_nodes"
            )
        if self.surface.faces.shape != (expected_faces, 3):
            raise DistributedDoubletError(
                "wake-band surface face count does not match span_nodes"
            )
        rows = _finite(
            "potential_jump_rows",
            self.potential_jump_rows,
            shape=(3, 2 * int(self.span_nodes) - 1),
        )
        object.__setattr__(self, "time_nodes", time_nodes.copy())
        object.__setattr__(self, "potential_jump_rows", rows.copy())

    @property
    def upstream_edges(self) -> tuple[tuple[int, int], ...]:
        return tuple((index, index + 1) for index in range(self.span_nodes - 1))

    @property
    def downstream_edges(self) -> tuple[tuple[int, int], ...]:
        offset = self.span_nodes
        return tuple(
            (offset + index, offset + index + 1)
            for index in range(self.span_nodes - 1)
        )

    @property
    def side_edges(self) -> tuple[tuple[int, int], tuple[int, int]]:
        offset = self.span_nodes
        return (0, offset), (self.span_nodes - 1, 2 * self.span_nodes - 1)

    def material_update(self, vertices) -> "MaterialWakeBand":
        moved = self.surface.material_update(vertices)
        return MaterialWakeBand(
            sheet_id=self.sheet_id,
            vortex_family=self.vortex_family,
            time_nodes=self.time_nodes,
            span_nodes=self.span_nodes,
            surface=moved,
            potential_jump_rows=self.potential_jump_rows,
        )

    def as_patch(
        self,
        *,
        upstream_interface: str,
        downstream_interface: str,
        side_roles: tuple[str, str] = ("zero", "zero"),
    ) -> QuadraticDoubletPatch:
        if not upstream_interface or not downstream_interface:
            raise DistributedDoubletError(
                "wake-band upstream/downstream interface ids must be non-empty"
            )
        if len(side_roles) != 2:
            raise DistributedDoubletError("side_roles must contain two entries")
        roles: dict[tuple[int, int], str] = {}
        for index, edge in enumerate(self.upstream_edges):
            roles[tuple(sorted(edge))] = (
                f"interface:{upstream_interface}:{index}"
            )
        for index, edge in enumerate(self.downstream_edges):
            roles[tuple(sorted(edge))] = (
                f"interface:{downstream_interface}:{index}"
            )
        for edge, role in zip(self.side_edges, side_roles):
            roles[tuple(sorted(edge))] = role
        return QuadraticDoubletPatch(self.sheet_id, self.surface, roles)


@dataclass(frozen=True)
class MaterialWakeHistory:
    """Chronological material bands with explicit inter-step seams.

    Bands are ordered from older to newer.  The current row of band ``i``
    must be exactly the previous row of band ``i+1`` in time, geometry, and
    P2 potential jump.  The class never creates a missing row or changes a
    material strength.  Terminal and spanwise boundary roles remain explicit
    when the history is converted to a global assembly.
    """

    history_id: str
    bands: tuple[MaterialWakeBand, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.history_id, str) or not self.history_id:
            raise DistributedDoubletError("history_id must be non-empty")
        bands = tuple(self.bands)
        if not bands or any(
            not isinstance(band, MaterialWakeBand) for band in bands
        ):
            raise DistributedDoubletError(
                "material wake history requires MaterialWakeBand entries"
            )
        sheet_ids = [band.sheet_id for band in bands]
        if len(set(sheet_ids)) != len(sheet_ids):
            raise DistributedDoubletError(
                "material wake history sheet ids must be unique"
            )
        object.__setattr__(self, "bands", bands)

    @property
    def vortex_family(self) -> str:
        return self.bands[0].vortex_family

    @property
    def span_nodes(self) -> int:
        return self.bands[0].span_nodes

    def continuity_report(
        self,
        *,
        time_tolerance: float = 1.0e-12,
        geometry_tolerance: float = 1.0e-12,
        strength_tolerance: float = 1.0e-12,
    ) -> MaterialWakeHistoryReport:
        tolerances = (
            time_tolerance,
            geometry_tolerance,
            strength_tolerance,
        )
        if any(
            tolerance < 0.0 or not np.isfinite(tolerance)
            for tolerance in tolerances
        ):
            raise DistributedDoubletError(
                "history tolerances must be finite and non-negative"
            )
        same_family = all(
            band.vortex_family == self.vortex_family for band in self.bands
        )
        same_span_nodes = all(
            band.span_nodes == self.span_nodes for band in self.bands
        )
        max_time_gap = 0.0
        max_geometry_gap = 0.0
        max_trace_jump = 0.0
        time_contiguous = True
        if same_span_nodes:
            for older, newer in zip(self.bands, self.bands[1:]):
                time_gap = abs(
                    float(older.time_nodes[2] - newer.time_nodes[0])
                )
                geometry_gap = float(
                    np.max(
                        np.linalg.norm(
                            older.surface.vertices[older.span_nodes :]
                            - newer.surface.vertices[: newer.span_nodes],
                            axis=1,
                        ),
                        initial=0.0,
                    )
                )
                trace_jump = float(
                    np.max(
                        np.abs(
                            older.potential_jump_rows[2]
                            - newer.potential_jump_rows[0]
                        ),
                        initial=0.0,
                    )
                )
                max_time_gap = max(max_time_gap, time_gap)
                max_geometry_gap = max(max_geometry_gap, geometry_gap)
                max_trace_jump = max(max_trace_jump, trace_jump)
                time_contiguous = (
                    time_contiguous and time_gap <= time_tolerance
                )
        else:
            max_geometry_gap = np.inf
            max_trace_jump = np.inf
            time_contiguous = False
        compatible = (
            same_family
            and same_span_nodes
            and time_contiguous
            and max_geometry_gap <= geometry_tolerance
            and max_trace_jump <= strength_tolerance
            and all(
                band.surface.continuity_report(
                    tolerance=strength_tolerance
                ).compatible
                for band in self.bands
            )
        )
        return MaterialWakeHistoryReport(
            bands=len(self.bands),
            internal_seams=max(len(self.bands) - 1, 0),
            same_family=same_family,
            same_span_nodes=same_span_nodes,
            time_contiguous=time_contiguous,
            max_time_gap=max_time_gap,
            max_geometry_gap=max_geometry_gap,
            max_trace_jump=max_trace_jump,
            compatible=compatible,
        )

    @staticmethod
    def _terminal_role(role: str, span_index: int) -> str:
        if role == "zero":
            return role
        if isinstance(role, str) and role.startswith("interface:"):
            return f"{role}:{span_index}"
        raise DistributedDoubletError(
            "terminal/side role must be 'zero' or 'interface:<id>'"
        )

    def as_patches(
        self,
        *,
        oldest_role: str,
        newest_role: str,
        side_roles: tuple[str, str] = ("zero", "zero"),
        time_tolerance: float = 1.0e-12,
        geometry_tolerance: float = 1.0e-12,
        strength_tolerance: float = 1.0e-12,
    ) -> tuple[QuadraticDoubletPatch, ...]:
        """Expose every history boundary to ``QuadraticDoubletAssembly``."""
        report = self.continuity_report(
            time_tolerance=time_tolerance,
            geometry_tolerance=geometry_tolerance,
            strength_tolerance=strength_tolerance,
        )
        if not report.compatible:
            raise DistributedDoubletError(
                f"material wake history is discontinuous: {report}"
            )
        if len(side_roles) != 2:
            raise DistributedDoubletError(
                "side_roles must contain root and tip roles"
            )
        patches = []
        last_index = len(self.bands) - 1
        for band_index, band in enumerate(self.bands):
            roles: dict[tuple[int, int], str] = {}
            for span_index, edge in enumerate(band.upstream_edges):
                if band_index == 0:
                    role = self._terminal_role(
                        oldest_role,
                        span_index,
                    )
                else:
                    role = (
                        f"interface:{self.history_id}:"
                        f"time-{band_index}:span-{span_index}"
                    )
                roles[tuple(sorted(edge))] = role
            for span_index, edge in enumerate(band.downstream_edges):
                if band_index == last_index:
                    role = self._terminal_role(
                        newest_role,
                        span_index,
                    )
                else:
                    role = (
                        f"interface:{self.history_id}:"
                        f"time-{band_index + 1}:span-{span_index}"
                    )
                roles[tuple(sorted(edge))] = role
            for side_index, (edge, role) in enumerate(
                zip(band.side_edges, side_roles)
            ):
                roles[tuple(sorted(edge))] = self._terminal_role(
                    role,
                    band_index,
                )
            patches.append(
                QuadraticDoubletPatch(
                    band.sheet_id,
                    band.surface,
                    roles,
                )
            )
        return tuple(patches)

    def append(
        self,
        band: MaterialWakeBand,
        *,
        time_tolerance: float = 1.0e-12,
        geometry_tolerance: float = 1.0e-12,
        strength_tolerance: float = 1.0e-12,
    ) -> "MaterialWakeHistory":
        candidate = MaterialWakeHistory(
            self.history_id,
            self.bands + (band,),
        )
        report = candidate.continuity_report(
            time_tolerance=time_tolerance,
            geometry_tolerance=geometry_tolerance,
            strength_tolerance=strength_tolerance,
        )
        if not report.compatible:
            raise DistributedDoubletError(
                f"appended material band breaks history: {report}"
            )
        return candidate


def newborn_material_wake_band(
    *,
    sheet_id: str,
    vortex_family: str,
    previous_edge,
    current_edge,
    time_nodes,
    potential_jump_rows,
    span_diagonal_pattern: str = "forward",
) -> MaterialWakeBand:
    """Create a continuous P2 material TEV/LEV band without a force closure.

    ``previous_edge`` and ``current_edge`` contain the same spanwise material
    vertices.  ``potential_jump_rows`` has shape ``(3, 2*nspan-1)``: the
    previous, middle, and current temporal values at every quadratic
    spanwise node.  The function performs only basis interpolation inside the
    newborn band; it never infers the middle row or fits a target load.
    """
    previous = _finite("previous_edge", previous_edge, ndim=2)
    current = _finite("current_edge", current_edge, ndim=2)
    if previous.shape != current.shape or previous.shape[1] != 3:
        raise DistributedDoubletError(
            "previous_edge and current_edge must share shape (n,3)"
        )
    span_nodes = len(previous)
    if span_nodes < 2:
        raise DistributedDoubletError("a wake band needs at least two span nodes")
    rows = _finite(
        "potential_jump_rows",
        potential_jump_rows,
        shape=(3, 2 * span_nodes - 1),
    )
    time_array = _finite("time_nodes", time_nodes, shape=(3,))
    if not (time_array[0] < time_array[1] < time_array[2]):
        raise DistributedDoubletError(
            "time_nodes must be strictly increasing"
        )
    time_scale = max(
        abs(float(time_array[0])),
        abs(float(time_array[2])),
        float(time_array[2] - time_array[0]),
        1.0,
    )
    if abs(
        float(time_array[1])
        - 0.5 * float(time_array[0] + time_array[2])
    ) > 64.0 * np.finfo(float).eps * time_scale:
        raise DistributedDoubletError(
            "time_nodes middle entry must be the explicit half-time"
        )
    if span_diagonal_pattern not in {"forward", "mirror_symmetric"}:
        raise DistributedDoubletError(
            "span_diagonal_pattern must be 'forward' or "
            "'mirror_symmetric'"
        )
    if (
        span_diagonal_pattern == "mirror_symmetric"
        and (span_nodes - 1) % 2 != 0
    ):
        raise DistributedDoubletError(
            "mirror_symmetric wake triangulation needs an even number "
            "of span intervals"
        )

    vertices = np.vstack((previous, current))
    faces: list[tuple[int, int, int]] = []
    face_mu: list[np.ndarray] = []
    for span_index in range(span_nodes - 1):
        old_left = span_index
        old_right = span_index + 1
        new_left = span_nodes + span_index
        new_right = span_nodes + span_index + 1
        forward = (
            span_diagonal_pattern == "forward"
            or span_index < (span_nodes - 1) // 2
        )
        if forward:
            local_faces = (
                (
                    (old_left, old_right, new_right),
                    np.array(
                        ((0.0, 0.0), (0.0, 1.0), (1.0, 1.0))
                    ),
                ),
                (
                    (old_left, new_right, new_left),
                    np.array(
                        ((0.0, 0.0), (1.0, 1.0), (1.0, 0.0))
                    ),
                ),
            )
        else:
            local_faces = (
                (
                    (old_left, old_right, new_left),
                    np.array(
                        ((0.0, 0.0), (0.0, 1.0), (1.0, 0.0))
                    ),
                ),
                (
                    (old_right, new_right, new_left),
                    np.array(
                        ((0.0, 1.0), (1.0, 1.0), (1.0, 0.0))
                    ),
                ),
            )
        span_values = rows[:, 2 * span_index : 2 * span_index + 3]
        for face, parametric_vertices in local_faces:
            parametric_nodes = np.vstack(
                (
                    parametric_vertices,
                    0.5 * (parametric_vertices[0] + parametric_vertices[1]),
                    0.5 * (parametric_vertices[1] + parametric_vertices[2]),
                    0.5 * (parametric_vertices[2] + parametric_vertices[0]),
                )
            )
            temporal_basis = _quadratic_line_basis(parametric_nodes[:, 0])
            span_basis = _quadratic_line_basis(parametric_nodes[:, 1])
            values = np.einsum(
                "ni,ij,nj->n",
                temporal_basis,
                span_values,
                span_basis,
            )
            faces.append(face)
            face_mu.append(values)

    surface = QuadraticDoubletSurface(
        vertices,
        np.asarray(faces, dtype=np.int64),
        np.asarray(face_mu),
    )
    continuity = surface.continuity_report()
    if not continuity.compatible:
        raise DistributedDoubletError(
            "newborn wake-band interpolation produced a discontinuous trace"
        )
    return MaterialWakeBand(
        sheet_id=sheet_id,
        vortex_family=vortex_family,
        time_nodes=time_array,
        span_nodes=span_nodes,
        surface=surface,
        potential_jump_rows=rows,
    )
