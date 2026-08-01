"""No-force spatial state for the N3.1i ``hirato_exact`` candidate.

The state implements the topological operation shown in Hirato Fig.4:
the nearest previously shed LEV ring is split at the Eq.7 edge, a nascent
ring is inserted between the geometric leading edge and that split edge, and
all free rings may then be convected by caller-supplied local velocities.

It intentionally contains no aerodynamic force and no first-ring placement
constant.  A NumPy reference for the complete local ring velocity is provided
only to convect the shadow state and expose a channel ledger; it cannot affect
V4.1 or be interpreted as a pressure closure.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .hirato_equations import (
    HiratoEquationError,
    _finite_array,
    rollup_displacement_eq24,
)


@dataclass(frozen=True)
class ShedReport:
    step: int
    created_ids: np.ndarray
    split_residual: np.ndarray


@dataclass(frozen=True)
class SheetSnapshot:
    rings: np.ndarray
    gamma: np.ndarray
    strip: np.ndarray
    birth_step: np.ndarray
    ring_id: np.ndarray
    sheet_id: np.ndarray
    previous_vertex_velocity: np.ndarray


@dataclass(frozen=True)
class LocalVelocityLedger:
    """Eq.23/24 velocity channels at every LEV-ring vertex."""

    freestream: np.ndarray
    bound: np.ndarray
    tev: np.ndarray
    lev: np.ndarray
    total: np.ndarray


@dataclass(frozen=True)
class Eq24ConvectionReport:
    """Audit record for one Eq.24 update.

    ``bootstrap_vertex`` marks vertices for which no material-point velocity
    existed at the previous time level.  The reference adapter uses
    ``v_previous = v_now`` only at those vertices, which is a registered
    first-order startup assumption rather than an identity stated by Hirato.
    """

    step: int
    bootstrap_vertex: np.ndarray
    displacement: np.ndarray


def ring_field_velocity(
    points: np.ndarray,
    rings: np.ndarray,
    gamma: np.ndarray,
    core_radius: float,
    *,
    denominator_floor: float = 1e-30,
) -> np.ndarray:
    """Regularized Biot--Savart velocity matching the production ring kernel.

    ``core_radius`` has length units, as in the van-Garrel denominator
    ``|r1 x r2|^2 + rc^2 |r0|^2``.  The explicit denominator floor lets the
    canonical bound-AIC adapter reproduce the frozen singular N1 kernel
    separately from the finite-core free-sheet field.
    """
    if not np.isfinite(core_radius) or core_radius < 0.0:
        raise HiratoEquationError(
            f"core_radius must be non-negative and finite, got {core_radius}"
        )
    if not np.isfinite(denominator_floor) or denominator_floor <= 0.0:
        raise HiratoEquationError(
            "denominator_floor must be positive and finite"
        )
    velocity = np.zeros_like(points)
    if rings.shape[0] == 0:
        return velocity
    for ring, strength in zip(rings, gamma, strict=True):
        for a, b in ((0, 1), (1, 2), (2, 3), (3, 0)):
            r1 = points - ring[a]
            r2 = points - ring[b]
            r0 = ring[b] - ring[a]
            cross = np.cross(r1, r2)
            denominator = (
                np.einsum("ij,ij->i", cross, cross)
                + core_radius**2 * float(np.dot(r0, r0))
                + denominator_floor
            )
            norm1 = np.sqrt(np.einsum("ij,ij->i", r1, r1) + 1e-20)
            norm2 = np.sqrt(np.einsum("ij,ij->i", r2, r2) + 1e-20)
            direction = r1 / norm1[:, None] - r2 / norm2[:, None]
            coefficient = direction @ r0
            velocity += (
                strength
                * coefficient[:, None]
                / (4.0 * np.pi * denominator[:, None])
                * cross
            )
    return velocity


_ring_field_velocity = ring_field_velocity


def ring_field_velocity_lamb_oseen(
    points: np.ndarray,
    rings: np.ndarray,
    gamma: np.ndarray,
    core_radius: float,
) -> np.ndarray:
    """Hirato Eq.25 finite-segment velocity with Lamb--Oseen cutoff.

    Hirato does not add ``rc`` to the Biot--Savart denominator.  The
    unregularized finite-segment velocity is multiplied by
    ``1-exp(-(r/rc)^2)``, where ``r`` is the perpendicular distance to the
    filament line.  The ratio is evaluated in its analytic zero-distance
    limit so a point on a filament remains finite without changing the law.
    """
    point_array = _finite_array("points", points, ndim=2)
    geometry, strength = _validated_ring_field("field", rings, gamma)
    if point_array.shape[1] != 3:
        raise HiratoEquationError(
            f"points must have shape (n,3), got {point_array.shape}"
        )
    if not np.isfinite(core_radius) or core_radius <= 0.0:
        raise HiratoEquationError(
            "Lamb-Oseen core_radius must be positive and finite"
        )
    velocity = np.zeros_like(point_array)
    for ring, circulation in zip(geometry, strength, strict=True):
        for a, b in ((0, 1), (1, 2), (2, 3), (3, 0)):
            r_start = point_array - ring[a]
            r_end = point_array - ring[b]
            filament = ring[b] - ring[a]
            filament_sq = float(np.dot(filament, filament))
            if filament_sq <= 0.0:
                raise HiratoEquationError("vortex filament has zero length")
            cross = np.cross(r_start, r_end)
            cross_sq = np.einsum("ij,ij->i", cross, cross)
            exponent = cross_sq / (filament_sq * core_radius**2)
            cutoff = -np.expm1(-exponent)
            cutoff_over_cross_sq = np.empty_like(cross_sq)
            nonzero = cross_sq > np.finfo(float).tiny
            cutoff_over_cross_sq[nonzero] = (
                cutoff[nonzero] / cross_sq[nonzero]
            )
            cutoff_over_cross_sq[~nonzero] = (
                1.0 / (filament_sq * core_radius**2)
            )
            norm_start = np.sqrt(
                np.einsum("ij,ij->i", r_start, r_start) + 1e-20
            )
            norm_end = np.sqrt(
                np.einsum("ij,ij->i", r_end, r_end) + 1e-20
            )
            direction = (
                r_start / norm_start[:, None]
                - r_end / norm_end[:, None]
            )
            coefficient = direction @ filament
            velocity += (
                circulation
                * coefficient[:, None]
                * cutoff_over_cross_sq[:, None]
                / (4.0 * np.pi)
                * cross
            )
    return velocity


def _validated_ring_field(name: str, rings, gamma) -> tuple[np.ndarray, np.ndarray]:
    geometry = _finite_array(f"{name}_rings", rings, ndim=3)
    strength = _finite_array(f"{name}_gamma", gamma, ndim=1)
    if geometry.shape[1:] != (4, 3):
        raise HiratoEquationError(
            f"{name}_rings must have shape (n, 4, 3), got {geometry.shape}"
        )
    if strength.shape != (geometry.shape[0],):
        raise HiratoEquationError(
            f"{name}_gamma must have shape {(geometry.shape[0],)}, got {strength.shape}"
        )
    return geometry, strength


def mirrored_ring_field(rings: np.ndarray) -> np.ndarray:
    mirrored = rings.copy()
    mirrored[..., 1] *= -1.0
    # Reflection changes orientation; reverse the traversal exactly as the
    # half-wing AIC symmetry operator does: (m0,m3,m2,m1).
    return mirrored[:, [0, 3, 2, 1]]


_mirrored_ring_field = mirrored_ring_field


class HiratoSheetShadow:
    """Ragged per-strip LEV-ring sheet with explicit connectivity metadata."""

    def __init__(self, ns: int):
        if ns <= 0:
            raise HiratoEquationError(f"ns must be positive, got {ns}")
        self.ns = int(ns)
        self.rings = np.empty((0, 4, 3), dtype=float)
        self.gamma = np.empty(0, dtype=float)
        self.strip = np.empty(0, dtype=np.int64)
        self.birth_step = np.empty(0, dtype=np.int64)
        self.ring_id = np.empty(0, dtype=np.int64)
        self.sheet_id = np.empty(0, dtype=np.int64)
        self.previous_vertex_velocity = np.empty((0, 4, 3), dtype=float)
        self._next_id = 0
        self._next_sheet_id = 0
        self._current_sheet = np.full(self.ns, -1, dtype=np.int64)

    def snapshot(self) -> SheetSnapshot:
        return SheetSnapshot(
            rings=self.rings.copy(),
            gamma=self.gamma.copy(),
            strip=self.strip.copy(),
            birth_step=self.birth_step.copy(),
            ring_id=self.ring_id.copy(),
            sheet_id=self.sheet_id.copy(),
            previous_vertex_velocity=self.previous_vertex_velocity.copy(),
        )

    def assign_created_strengths(
        self,
        created_ids,
        strengths,
    ) -> None:
        """Assign solved strengths to rings created with zero placeholders.

        Geometry/topology must be created before the coupled LESP solve can
        know ``Gamma_L``.  This method makes that two-stage operation explicit
        and refuses duplicate, stale, or unknown ring identities.
        """
        ids = np.asarray(created_ids, dtype=np.int64)
        values = _finite_array("strengths", strengths, ndim=1)
        if ids.ndim != 1 or values.shape != ids.shape:
            raise HiratoEquationError(
                f"created_ids and strengths must match as 1D arrays, got {ids.shape} and {values.shape}"
            )
        if len(np.unique(ids)) != len(ids):
            raise HiratoEquationError("created_ids contains duplicates")
        for ring_id, value in zip(ids, values, strict=True):
            indices = np.flatnonzero(self.ring_id == ring_id)
            if indices.size != 1:
                raise HiratoEquationError(
                    f"created ring id {int(ring_id)} is unknown or ambiguous"
                )
            self.gamma[int(indices[0])] = float(value)

    def _newest_index(self, strip: int) -> int | None:
        candidates = np.flatnonzero(self.strip == strip)
        if candidates.size == 0:
            return None
        local = np.lexsort((self.ring_id[candidates], self.birth_step[candidates]))
        return int(candidates[local[-1]])

    def shed(
        self,
        *,
        step: int,
        leading_edges,
        gamma_now,
        active,
        first_aft_edges=None,
        new_sheet=None,
    ) -> ShedReport:
        """Insert active nascent rings using the Fig.4/Eq.7 split.

        ``leading_edges`` and ``first_aft_edges`` use shape ``(ns, 2, 3)``
        with endpoint order ``[left, right]``.  ``first_aft_edges`` is required
        for a strip with no previous LEV ring or whose intermittent event
        starts a ``new_sheet``.  The caller owns the first-vortex placement
        law; this state refuses to invent one.
        """
        if step < 0:
            raise HiratoEquationError(f"step must be non-negative, got {step}")
        le = _finite_array("leading_edges", leading_edges, ndim=3)
        g = _finite_array("gamma_now", gamma_now, ndim=1)
        mask = np.asarray(active, dtype=bool)
        expected_edge = (self.ns, 2, 3)
        if le.shape != expected_edge:
            raise HiratoEquationError(
                f"leading_edges must have shape {expected_edge}, got {le.shape}"
            )
        if g.shape != (self.ns,) or mask.shape != (self.ns,):
            raise HiratoEquationError(
                f"gamma_now and active must have shape {(self.ns,)}"
            )
        starts = (
            np.zeros(self.ns, dtype=bool)
            if new_sheet is None
            else np.asarray(new_sheet, dtype=bool)
        )
        if starts.shape != (self.ns,):
            raise HiratoEquationError(
                f"new_sheet must have shape {(self.ns,)}, got {starts.shape}"
            )
        if np.any(starts & ~mask):
            raise HiratoEquationError("new_sheet may only be true on active strips")
        first = None
        if first_aft_edges is not None:
            first = _finite_array("first_aft_edges", first_aft_edges, ndim=3)
            if first.shape != expected_edge:
                raise HiratoEquationError(
                    f"first_aft_edges must have shape {expected_edge}, got {first.shape}"
                )

        new_rings: list[np.ndarray] = []
        new_gamma: list[float] = []
        new_strip: list[int] = []
        new_birth: list[int] = []
        new_ids: list[int] = []
        new_sheet_ids: list[int] = []
        residuals: list[float] = []

        for j in np.flatnonzero(mask):
            newest = self._newest_index(int(j))
            if starts[j]:
                newest = None
            if newest is None:
                if first is None:
                    raise HiratoEquationError(
                        f"strip {j} has no previous LEV ring; first_aft_edges is required"
                    )
                split = first[j].copy()
                self._current_sheet[j] = self._next_sheet_id
                self._next_sheet_id += 1
            else:
                # Ring order: front-left, front-right, aft-right, aft-left.
                # x_{L,n-2} in Fig.4 is the aft edge of the nearest old ring.
                last_edge = np.stack(
                    [self.rings[newest, 3], self.rings[newest, 2]],
                    axis=0,
                )
                split = (2.0 / 3.0) * le[j] + (1.0 / 3.0) * last_edge
                # Split the old panel instead of adding an overlapping ring.
                self.rings[newest, 0] = split[0]
                self.rings[newest, 1] = split[1]
                # The split edge is a new remeshed vertex.  It cannot inherit
                # a different material point's Eq.24 velocity history.
                self.previous_vertex_velocity[newest, 0:2] = np.nan

            ring = np.stack(
                [le[j, 0], le[j, 1], split[1], split[0]],
                axis=0,
            )
            new_rings.append(ring)
            new_gamma.append(float(g[j]))
            new_strip.append(int(j))
            new_birth.append(int(step))
            new_ids.append(self._next_id)
            new_sheet_ids.append(int(self._current_sheet[j]))
            self._next_id += 1
            if newest is None:
                residuals.append(0.0)
            else:
                residuals.append(
                    float(
                        max(
                            np.linalg.norm(ring[3] - self.rings[newest, 0]),
                            np.linalg.norm(ring[2] - self.rings[newest, 1]),
                        )
                    )
                )

        if new_rings:
            self.rings = np.concatenate([self.rings, np.stack(new_rings)], axis=0)
            self.gamma = np.concatenate([self.gamma, np.asarray(new_gamma)])
            self.strip = np.concatenate(
                [self.strip, np.asarray(new_strip, dtype=np.int64)]
            )
            self.birth_step = np.concatenate(
                [self.birth_step, np.asarray(new_birth, dtype=np.int64)]
            )
            self.ring_id = np.concatenate(
                [self.ring_id, np.asarray(new_ids, dtype=np.int64)]
            )
            self.sheet_id = np.concatenate(
                [self.sheet_id, np.asarray(new_sheet_ids, dtype=np.int64)]
            )
            self.previous_vertex_velocity = np.concatenate(
                [
                    self.previous_vertex_velocity,
                    np.full((len(new_rings), 4, 3), np.nan),
                ],
                axis=0,
            )

        return ShedReport(
            step=int(step),
            created_ids=np.asarray(new_ids, dtype=np.int64),
            split_residual=np.asarray(residuals, dtype=float),
        )

    def convect(self, vertex_velocity, dt: float) -> None:
        """Historical current-velocity Euler scaffold, not Hirato Eq.24."""
        velocity = _finite_array("vertex_velocity", vertex_velocity, ndim=3)
        if velocity.shape != self.rings.shape:
            raise HiratoEquationError(
                f"vertex_velocity must have shape {self.rings.shape}, got {velocity.shape}"
            )
        if not np.isfinite(dt) or dt <= 0.0:
            raise HiratoEquationError(f"dt must be positive and finite, got {dt}")
        self.rings = self.rings + velocity * dt

    def convect_eq24(
        self,
        vertex_velocity,
        dt: float,
        *,
        step: int,
    ) -> Eq24ConvectionReport:
        """Advance every free vertex after the current step's shedding solve.

        Hirato Fig.4 and the published procedure place the nascent LEV before
        the current step's common TEV/LEV convection, so a ring born at
        ``step`` must move in this call.  Eq.24 requires current and previous
        local velocities.  A newly born vertex or an Eq.7 remeshed split
        vertex has no previous material-point value; the no-force reference
        therefore exposes and records a zero-parameter Euler bootstrap
        (``v_previous = v_now``) at only those vertices.  This bootstrap is an
        adapter hypothesis and must pass a separate resolution sensitivity
        gate before any pressure promotion.
        """
        velocity = _finite_array("vertex_velocity", vertex_velocity, ndim=3)
        if velocity.shape != self.rings.shape:
            raise HiratoEquationError(
                f"vertex_velocity must have shape {self.rings.shape}, got {velocity.shape}"
            )
        if step < 0:
            raise HiratoEquationError(f"step must be non-negative, got {step}")
        if not np.isfinite(dt) or dt <= 0.0:
            raise HiratoEquationError(f"dt must be positive and finite, got {dt}")
        if np.any(self.birth_step > step):
            raise HiratoEquationError("sheet contains a ring born in the future")
        bootstrap_vertex = ~np.all(
            np.isfinite(self.previous_vertex_velocity),
            axis=-1,
        )
        previous = np.where(
            np.isfinite(self.previous_vertex_velocity),
            self.previous_vertex_velocity,
            velocity,
        )
        displacement = rollup_displacement_eq24(velocity, previous, dt)
        self.rings += displacement
        self.previous_vertex_velocity = velocity.copy()
        return Eq24ConvectionReport(
            step=int(step),
            bootstrap_vertex=bootstrap_vertex,
            displacement=displacement.copy(),
        )

    def full_local_velocity(
        self,
        *,
        bound_rings,
        bound_gamma,
        tev_rings,
        tev_gamma,
        u_infinity,
        core_radius: float,
        mirror_symmetry: bool,
    ) -> LocalVelocityLedger:
        """Evaluate Eq.23/24 channels at every shadow LEV vertex.

        All source fields are explicit inputs.  ``core_radius`` is a numerical
        regularization supplied by the caller and must later pass a resolution
        study; this state owns no hidden vortex-core constant.
        """
        bound, bound_g = _validated_ring_field(
            "bound", bound_rings, bound_gamma
        )
        tev, tev_g = _validated_ring_field("tev", tev_rings, tev_gamma)
        lev, lev_g = _validated_ring_field("lev", self.rings, self.gamma)
        freestream_vector = _finite_array(
            "u_infinity", u_infinity, ndim=1
        )
        if freestream_vector.shape != (3,):
            raise HiratoEquationError(
                f"u_infinity must have shape (3,), got {freestream_vector.shape}"
            )
        if not np.isfinite(core_radius) or core_radius <= 0.0:
            raise HiratoEquationError(
                f"core_radius must be positive and finite, got {core_radius}"
            )
        points = lev.reshape(-1, 3)
        shape = lev.shape
        freestream = np.broadcast_to(freestream_vector, points.shape).copy()
        bound_velocity = ring_field_velocity(
            points, bound, bound_g, core_radius
        )
        tev_velocity = ring_field_velocity(points, tev, tev_g, core_radius)
        lev_velocity = ring_field_velocity(points, lev, lev_g, core_radius)
        if mirror_symmetry:
            bound_velocity += ring_field_velocity(
                points, mirrored_ring_field(bound), bound_g, core_radius
            )
            tev_velocity += ring_field_velocity(
                points, mirrored_ring_field(tev), tev_g, core_radius
            )
            lev_velocity += ring_field_velocity(
                points, mirrored_ring_field(lev), lev_g, core_radius
            )
        total = freestream + bound_velocity + tev_velocity + lev_velocity
        return LocalVelocityLedger(
            freestream=freestream.reshape(shape),
            bound=bound_velocity.reshape(shape),
            tev=tev_velocity.reshape(shape),
            lev=lev_velocity.reshape(shape),
            total=total.reshape(shape),
        )

    def convect_full_local(self, *, dt: float, **field_inputs) -> LocalVelocityLedger:
        """Historical Euler advance with a complete channel ledger."""
        ledger = self.full_local_velocity(**field_inputs)
        self.convect(ledger.total, dt)
        return ledger

    def convect_full_local_eq24(
        self,
        *,
        dt: float,
        step: int,
        **field_inputs,
    ) -> tuple[LocalVelocityLedger, Eq24ConvectionReport]:
        """Advance the no-force shadow with the full field and Eq.24 state."""
        ledger = self.full_local_velocity(**field_inputs)
        report = self.convect_eq24(ledger.total, dt, step=step)
        return ledger, report

    def strip_observables(self) -> dict[str, np.ndarray]:
        """Return diagnostic moments; these are observers, never load generators."""
        circulation = np.zeros(self.ns)
        first_moment = np.zeros((self.ns, 3))
        second_moment = np.zeros((self.ns, 3, 3))
        centers = self.rings.mean(axis=1) if len(self.rings) else np.empty((0, 3))
        for k, j in enumerate(self.strip):
            weight = self.gamma[k]
            circulation[j] += weight
            first_moment[j] += weight * centers[k]
            second_moment[j] += weight * np.outer(centers[k], centers[k])

        centroid = np.full((self.ns, 3), np.nan)
        covariance = np.full((self.ns, 3, 3), np.nan)
        nonzero = np.abs(circulation) > np.finfo(float).eps
        centroid[nonzero] = first_moment[nonzero] / circulation[nonzero, None]
        for j in np.flatnonzero(nonzero):
            raw = second_moment[j] / circulation[j]
            covariance[j] = raw - np.outer(centroid[j], centroid[j])
        return {
            "circulation": circulation,
            "centroid": centroid,
            "covariance": covariance,
            "ring_count": np.bincount(self.strip, minlength=self.ns),
        }
