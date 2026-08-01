"""No-force spatial P2 LEV state primitives.

This module is deliberately narrower than an aerodynamic closure.  It owns
only a continuous material potential-jump state, its off-sheet induced
velocity, causal band birth, and material geometry transport.  It introduces
no force, pressure, fitted coefficient, core radius, or target-load input.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np

from .continuous_shedding import newborn_halfwing_shedding_band
from .distributed_doublet import (
    DistributedDoubletError,
    MaterialWakeBand,
    MaterialWakeHistory,
    QuadraticDoubletSurface,
)


def _surface_tuple(
    surfaces: QuadraticDoubletSurface | Iterable[QuadraticDoubletSurface],
) -> tuple[QuadraticDoubletSurface, ...]:
    if isinstance(surfaces, QuadraticDoubletSurface):
        result = (surfaces,)
    else:
        try:
            result = tuple(surfaces)
        except TypeError as error:
            raise DistributedDoubletError(
                "surfaces must be a QuadraticDoubletSurface or an iterable"
            ) from error
    if not result or any(
        not isinstance(surface, QuadraticDoubletSurface)
        for surface in result
    ):
        raise DistributedDoubletError(
            "surfaces must contain at least one QuadraticDoubletSurface"
        )
    return result


def _mirror_halfwing_surface(
    surface: QuadraticDoubletSurface,
) -> QuadraticDoubletSurface:
    """Reflect an oriented surface through ``y=0`` with restored winding."""
    vertices = surface.vertices.copy()
    vertices[:, 1] *= -1.0
    faces = surface.faces[:, [0, 2, 1]]
    face_mu = surface.face_mu[:, [0, 2, 1, 5, 4, 3]]
    return QuadraticDoubletSurface(vertices, faces, face_mu)


def _shape_gradients(
    barycentric: np.ndarray,
    barycentric_gradients: np.ndarray,
) -> np.ndarray:
    """Vectorized P2 physical gradients, shape ``(face,...,6,3)``."""
    l0 = barycentric[..., 0]
    l1 = barycentric[..., 1]
    l2 = barycentric[..., 2]
    g0 = barycentric_gradients[:, 0]
    g1 = barycentric_gradients[:, 1]
    g2 = barycentric_gradients[:, 2]
    result = np.empty(barycentric.shape[:-1] + (6, 3), dtype=float)
    result[..., 0, :] = (4.0 * l0 - 1.0)[..., None] * g0[
        :, *(None,) * (l0.ndim - 1), :
    ]
    result[..., 1, :] = (4.0 * l1 - 1.0)[..., None] * g1[
        :, *(None,) * (l1.ndim - 1), :
    ]
    result[..., 2, :] = (4.0 * l2 - 1.0)[..., None] * g2[
        :, *(None,) * (l2.ndim - 1), :
    ]
    g0b = g0[:, *(None,) * (l0.ndim - 1), :]
    g1b = g1[:, *(None,) * (l0.ndim - 1), :]
    g2b = g2[:, *(None,) * (l0.ndim - 1), :]
    result[..., 3, :] = 4.0 * (
        l0[..., None] * g1b + l1[..., None] * g0b
    )
    result[..., 4, :] = 4.0 * (
        l1[..., None] * g2b + l2[..., None] * g1b
    )
    result[..., 5, :] = 4.0 * (
        l2[..., None] * g0b + l0[..., None] * g2b
    )
    return result


def _p2_values(barycentric: np.ndarray) -> np.ndarray:
    l0, l1, l2 = np.moveaxis(barycentric, -1, 0)
    return np.stack(
        (
            l0 * (2.0 * l0 - 1.0),
            l1 * (2.0 * l1 - 1.0),
            l2 * (2.0 * l2 - 1.0),
            4.0 * l0 * l1,
            4.0 * l1 * l2,
            4.0 * l2 * l0,
        ),
        axis=-1,
    )


def vectorized_induced_velocity(
    surfaces: QuadraticDoubletSurface | Iterable[QuadraticDoubletSurface],
    points,
    *,
    quadrature_order: int = 24,
    plane_tolerance: float = 128.0 * np.finfo(float).eps,
    mirror_halfwing: bool = False,
    face_batch_size: int | None = None,
) -> np.ndarray:
    """Evaluate one or more P2 doublet surfaces at off-sheet targets.

    Faces, targets, and standard Gauss edge nodes are evaluated in array
    batches.  The equations are the same analytic-radial/edge-quadrature
    identity as
    :meth:`QuadraticDoubletSurface.induced_velocity_line_reduced`.
    """
    surface_list = _surface_tuple(surfaces)
    if not isinstance(mirror_halfwing, (bool, np.bool_)):
        raise DistributedDoubletError("mirror_halfwing must be boolean")
    if mirror_halfwing:
        surface_list = surface_list + tuple(
            _mirror_halfwing_surface(surface) for surface in surface_list
        )
    targets = np.asarray(points, dtype=float)
    if (
        targets.ndim != 2
        or targets.shape[1] != 3
        or not np.all(np.isfinite(targets))
    ):
        raise DistributedDoubletError(
            "points must be a finite array with shape (n,3)"
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
    if (
        face_batch_size is not None
        and (
            not isinstance(face_batch_size, (int, np.integer))
            or face_batch_size < 1
        )
    ):
        raise DistributedDoubletError(
            "face_batch_size must be None or a positive integer"
        )
    if len(targets) == 0:
        return np.empty((0, 3), dtype=float)

    vertices = np.concatenate(
        [surface.vertices[surface.faces] for surface in surface_list],
        axis=0,
    )
    material_mu = np.concatenate(
        [surface.face_mu for surface in surface_list],
        axis=0,
    )
    if (
        not np.all(np.isfinite(vertices))
        or not np.all(np.isfinite(material_mu))
    ):
        raise DistributedDoubletError(
            "surface geometry and material strength must be finite"
        )

    # The line-reduced identity below contains several temporary arrays with
    # shape (face, target, quadrature, 3).  A full production wake can contain
    # O(10^4) faces, so expanding all faces at once is needlessly capable of
    # exhausting host memory.  This is a purely algebraic partition of the
    # face sum: no geometry, strength, quadrature, or physical regularisation
    # changes between batches.
    if face_batch_size is None:
        working_set_bytes = 64 * 1024**2
        vector_temporaries = 12
        bytes_per_face = max(
            1,
            len(targets)
            * int(quadrature_order)
            * 3
            * np.dtype(float).itemsize
            * vector_temporaries,
        )
        batch_size = max(1, working_set_bytes // bytes_per_face)
    else:
        batch_size = int(face_batch_size)
    if len(vertices) > batch_size:
        result = np.zeros((len(targets), 3), dtype=float)
        for start in range(0, len(vertices), batch_size):
            stop = min(start + batch_size, len(vertices))
            batch_vertices = vertices[start:stop]
            batch_count = stop - start
            isolated = QuadraticDoubletSurface(
                batch_vertices.reshape(-1, 3),
                np.arange(3 * batch_count, dtype=np.int64).reshape(-1, 3),
                material_mu[start:stop],
            )
            result += vectorized_induced_velocity(
                isolated,
                targets,
                quadrature_order=int(quadrature_order),
                plane_tolerance=float(plane_tolerance),
                mirror_halfwing=False,
                face_batch_size=batch_count,
            )
        if not np.all(np.isfinite(result)):
            raise DistributedDoubletError(
                "batched vectorized velocity contains non-finite values"
            )
        return result

    edge01 = vertices[:, 1] - vertices[:, 0]
    edge02 = vertices[:, 2] - vertices[:, 0]
    area_vector = np.cross(edge01, edge02)
    area2 = np.linalg.norm(area_vector, axis=1)
    edge12 = vertices[:, 2] - vertices[:, 1]
    geometry_scale = np.maximum.reduce(
        (
            np.linalg.norm(edge01, axis=1),
            np.linalg.norm(edge02, axis=1),
            np.linalg.norm(edge12, axis=1),
            np.ones(len(vertices)),
        )
    )
    if np.any(
        area2 <= 64.0 * np.finfo(float).eps * geometry_scale**2
    ):
        raise DistributedDoubletError("surface contains a degenerate face")
    normal = area_vector / area2[:, None]
    area = 0.5 * area2
    delta = targets[None, :, :] - vertices[:, None, 0, :]
    normal_distance = np.einsum("ftj,fj->ft", delta, normal)
    length_scale = np.maximum(np.sqrt(2.0 * area), 1.0)
    if np.any(
        np.abs(normal_distance)
        <= plane_tolerance * length_scale[:, None]
    ):
        raise DistributedDoubletError(
            "line-reduced off-plane operator received an on-sheet point"
        )
    projected = (
        targets[None, :, :]
        - normal_distance[..., None] * normal[:, None, :]
    )

    normal_squared = np.einsum("fj,fj->f", area_vector, area_vector)
    barycentric_gradients = np.stack(
        (
            np.cross(area_vector, vertices[:, 2] - vertices[:, 1])
            / normal_squared[:, None],
            np.cross(area_vector, vertices[:, 0] - vertices[:, 2])
            / normal_squared[:, None],
            np.cross(area_vector, vertices[:, 1] - vertices[:, 0])
            / normal_squared[:, None],
        ),
        axis=1,
    )
    projected_delta = projected - vertices[:, None, 0, :]
    projected_l1 = np.einsum(
        "ftj,fj->ft", projected_delta, barycentric_gradients[:, 1]
    )
    projected_l2 = np.einsum(
        "ftj,fj->ft", projected_delta, barycentric_gradients[:, 2]
    )
    projected_barycentric = np.stack(
        (1.0 - projected_l1 - projected_l2, projected_l1, projected_l2),
        axis=-1,
    )
    projected_gradients = _shape_gradients(
        projected_barycentric,
        barycentric_gradients,
    )
    gradient_at_projection = np.einsum(
        "ftij,fi->ftj", projected_gradients, material_mu
    )
    gamma_at_projection = np.cross(
        gradient_at_projection, normal[:, None, :]
    )

    abscissa, weights = np.polynomial.legendre.leggauss(
        int(quadrature_order)
    )
    coordinate = 0.5 * (abscissa + 1.0)
    weights = 0.5 * weights
    velocity = np.zeros((len(vertices), len(targets), 3), dtype=float)
    for start_index, end_index in ((0, 1), (1, 2), (2, 0)):
        start = vertices[:, start_index]
        end = vertices[:, end_index]
        edge_vector = end - start
        source = (
            (1.0 - coordinate)[None, :, None] * start[:, None, :]
            + coordinate[None, :, None] * end[:, None, :]
        )
        edge_barycentric = np.zeros(
            (len(vertices), len(coordinate), 3), dtype=float
        )
        edge_barycentric[:, :, start_index] = 1.0 - coordinate
        edge_barycentric[:, :, end_index] = coordinate
        edge_mu = np.einsum(
            "fqi,fi->fq",
            _p2_values(edge_barycentric),
            material_mu,
        )
        edge_gradients = _shape_gradients(
            edge_barycentric,
            barycentric_gradients,
        )
        edge_gradient = np.einsum(
            "fqij,fi->fqj", edge_gradients, material_mu
        )
        edge_gamma = np.cross(edge_gradient, normal[:, None, :])

        separation = targets[None, :, None, :] - source[:, None, :, :]
        separation_norm = np.linalg.norm(separation, axis=-1)
        if np.any(separation_norm <= np.finfo(float).tiny):
            raise DistributedDoubletError(
                "target lies on a source edge quadrature point"
            )
        boundary_integrand = (
            np.cross(edge_vector[:, None, None, :], separation)
            / separation_norm[..., None] ** 3
        )
        velocity += np.einsum(
            "ftqj,fq,q->ftj",
            boundary_integrand,
            edge_mu,
            weights,
        )

        radial = source[:, None, :, :] - projected[:, :, None, :]
        radius = np.linalg.norm(radial, axis=-1)
        if np.any(radius <= np.finfo(float).tiny):
            raise DistributedDoubletError(
                "projected target lies on an element edge"
            )
        direction = radial / radius[..., None]
        signed_angle_jacobian = np.einsum(
            "ftqj,fj->ftq",
            np.cross(radial, edge_vector[:, None, None, :]),
            normal,
        ) / radius**2
        height = normal_distance
        absolute_height = np.abs(height)
        distance = np.sqrt(radius**2 + absolute_height[:, :, None] ** 2)
        height_j1 = np.sign(height)[:, :, None] * (
            1.0 - absolute_height[:, :, None] / distance
        )
        j2 = np.arcsinh(
            radius / absolute_height[:, :, None]
        ) - radius / distance
        height_j2 = height[:, :, None] * j2
        j4 = (
            distance
            + absolute_height[:, :, None] ** 2 / distance
            - 2.0 * absolute_height[:, :, None]
        )
        gamma1 = (
            edge_gamma[:, None, :, :]
            - gamma_at_projection[:, :, None, :]
        ) / radius[..., None]
        radial_integral = (
            np.cross(
                gamma_at_projection, normal[:, None, :]
            )[:, :, None, :]
            * height_j1[..., None]
            - np.cross(gamma_at_projection[:, :, None, :], direction)
            * j2[..., None]
            + np.cross(gamma1, normal[:, None, None, :])
            * height_j2[..., None]
            - np.cross(gamma1, direction) * j4[..., None]
        )
        velocity += np.einsum(
            "ftqj,ftq,q->ftj",
            radial_integral,
            signed_angle_jacobian,
            weights,
        )

    result = np.sum(velocity, axis=0) / (4.0 * np.pi)
    if not np.all(np.isfinite(result)):
        raise DistributedDoubletError(
            "vectorized line-reduced velocity contains non-finite values"
        )
    return result


def causal_band(
    *,
    sheet_id: str,
    vortex_family: str = "LEV_SUCTION",
    previous_edge,
    current_edge,
    span_edges,
    time_nodes,
    q_prev,
    q_mid,
    q_now,
    residual_tolerance: float = 1.0e-11,
) -> MaterialWakeBand:
    """Create one causal band from three explicitly supplied strip states."""
    strip_strength_rows = np.stack(
        (
            np.asarray(q_prev, dtype=float),
            np.asarray(q_mid, dtype=float),
            np.asarray(q_now, dtype=float),
        )
    )
    return newborn_halfwing_shedding_band(
        sheet_id=sheet_id,
        vortex_family=vortex_family,
        previous_edge=previous_edge,
        current_edge=current_edge,
        span_edges=span_edges,
        time_nodes=time_nodes,
        strip_strength_rows=strip_strength_rows,
        residual_tolerance=residual_tolerance,
    ).band


@dataclass(frozen=True)
class OutflowIntegral:
    """Integrated state retained when an old material band exits the window."""

    sheet_id: str
    vortex_family: str
    time_start: float
    time_end: float
    area: float
    potential_jump_area_integral: float


@dataclass(frozen=True)
class HistoryIntegral:
    band_count: int
    area: float
    potential_jump_area_integral: float


def _band_integral(band: MaterialWakeBand) -> OutflowIntegral:
    elements = tuple(
        band.surface.element(index) for index in range(len(band.surface))
    )
    return OutflowIntegral(
        sheet_id=band.sheet_id,
        vortex_family=band.vortex_family,
        time_start=float(band.time_nodes[0]),
        time_end=float(band.time_nodes[2]),
        area=float(sum(element.area for element in elements)),
        potential_jump_area_integral=float(
            sum(element.integral_mu() for element in elements)
        ),
    )


class P2LEVHistory:
    """Bounded chronological history of causal material P2 LEV bands."""

    def __init__(
        self,
        history_id: str,
        max_bands: int,
        bands: Iterable[MaterialWakeBand] = (),
        outflow: Iterable[OutflowIntegral] = (),
    ) -> None:
        if not isinstance(history_id, str) or not history_id:
            raise DistributedDoubletError("history_id must be non-empty")
        if (
            not isinstance(max_bands, (int, np.integer))
            or max_bands < 1
        ):
            raise DistributedDoubletError("max_bands must be an integer >=1")
        self.history_id = history_id
        self.max_bands = int(max_bands)
        self._bands: list[MaterialWakeBand] = []
        self._outflow = list(outflow)
        if any(
            not isinstance(item, OutflowIntegral) for item in self._outflow
        ):
            raise DistributedDoubletError(
                "outflow must contain OutflowIntegral records"
            )
        for band in bands:
            self.append(band)

    @property
    def bands(self) -> tuple[MaterialWakeBand, ...]:
        return tuple(self._bands)

    @property
    def outflow(self) -> tuple[OutflowIntegral, ...]:
        return tuple(self._outflow)

    @property
    def current_edge(self) -> np.ndarray | None:
        if not self._bands:
            return None
        newest = self._bands[-1]
        return newest.surface.vertices[newest.span_nodes :].copy()

    @property
    def previous_strip_strength(self) -> np.ndarray | None:
        if not self._bands:
            return None
        return self._bands[-1].potential_jump_rows[2, 1::2].copy()

    def _validate(self) -> None:
        if not self._bands:
            return
        report = MaterialWakeHistory(
            self.history_id,
            tuple(self._bands),
        ).continuity_report()
        if not report.compatible:
            raise DistributedDoubletError(
                f"P2 LEV history is discontinuous: {report}"
            )

    def append(self, band: MaterialWakeBand) -> None:
        if not isinstance(band, MaterialWakeBand):
            raise DistributedDoubletError(
                "append requires a MaterialWakeBand"
            )
        candidate = self._bands + [band]
        if len(candidate) > 1:
            report = MaterialWakeHistory(
                self.history_id,
                tuple(candidate),
            ).continuity_report()
            if not report.compatible:
                raise DistributedDoubletError(
                    f"appended material band breaks history: {report}"
                )
        self._bands.append(band)
        while len(self._bands) > self.max_bands:
            self.evict_oldest()
        self._validate()

    def evict_oldest(self) -> OutflowIntegral:
        """Move the oldest resident band into the integral outflow ledger."""
        if not self._bands:
            raise DistributedDoubletError(
                "cannot evict from an empty P2 LEV history"
            )
        record = _band_integral(self._bands.pop(0))
        self._outflow.append(record)
        self._validate()
        return record

    @staticmethod
    def _velocity(
        callback: Callable[[np.ndarray], np.ndarray],
        points: np.ndarray,
    ) -> np.ndarray:
        velocity = np.asarray(callback(points.copy()), dtype=float)
        if velocity.shape != points.shape or not np.all(np.isfinite(velocity)):
            raise DistributedDoubletError(
                "external_velocity must return finite shape-matched velocities"
            )
        return velocity

    def convect_heun(
        self,
        external_velocity: Callable[[np.ndarray], np.ndarray],
        dt: float,
    ) -> None:
        """Advance material geometry with explicit Heun; strengths stay fixed."""
        if not callable(external_velocity):
            raise DistributedDoubletError(
                "external_velocity must be callable"
            )
        if not np.isfinite(dt) or dt <= 0.0:
            raise DistributedDoubletError("dt must be finite and positive")
        if not self._bands:
            return
        counts = [len(band.surface.vertices) for band in self._bands]
        offsets = np.cumsum((0, *counts))
        geometry = np.concatenate(
            [band.surface.vertices for band in self._bands],
            axis=0,
        )
        velocity_start = self._velocity(external_velocity, geometry)
        predictor = geometry + float(dt) * velocity_start
        velocity_end = self._velocity(external_velocity, predictor)
        corrected = geometry + 0.5 * float(dt) * (
            velocity_start + velocity_end
        )
        moved: list[MaterialWakeBand] = []
        for index, band in enumerate(self._bands):
            moved.append(
                band.material_update(
                    corrected[offsets[index] : offsets[index + 1]]
                )
            )
        for before, after in zip(self._bands, moved):
            if not before.surface.kelvin_report(after.surface).passed:
                raise DistributedDoubletError(
                    "material convection changed a P2 strength or topology"
                )
        self._bands = moved
        self._validate()

    def integral_snapshot(self) -> HistoryIntegral:
        records = [_band_integral(band) for band in self._bands]
        return HistoryIntegral(
            band_count=len(records),
            area=float(sum(record.area for record in records)),
            potential_jump_area_integral=float(
                sum(
                    record.potential_jump_area_integral
                    for record in records
                )
            ),
        )


__all__ = [
    "HistoryIntegral",
    "OutflowIntegral",
    "P2LEVHistory",
    "causal_band",
    "vectorized_induced_velocity",
]
