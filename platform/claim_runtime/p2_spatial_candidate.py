"""Executable N3.1j0 causal P2 near-LEV research candidate.

This module owns only the free-sheet state and its same-stage coupling to a
caller-supplied bound system.  It does not compute pressure or force.  The
caller must pass the resulting induced velocity and release-potential history
to the unique panel-pressure provider.

The current-row source is solved from the LESP equality.  LESP therefore
selects release and circulation supply; it is never converted into a force
coefficient.  Each accepted row is stored in a chronological continuous P2
material band.  The first implementation deliberately excludes P2
self-advection: the external N1 bound/TEV transport field is supplied by the
caller and this approximation is exposed in every diagnostic.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from .continuous_shedding import reconstruct_halfwing_p2_trace
from .coupled_lesp_dde import solve_coupled_lesp_dde_stage
from .distributed_doublet import DistributedDoubletError
from .hirato_equations import (
    embed_chord_normal_displacement,
    first_lev_displacement_ramesh_2d,
    lesp_eq6,
)
from .p2_spatial_lev import (
    P2LEVHistory,
    causal_band,
    vectorized_induced_velocity,
)


def _finite(name: str, value, shape: tuple[int, ...]) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise DistributedDoubletError(
            f"{name} must be finite with shape {shape}, got {array.shape}"
        )
    return array


def _strip_vectors_to_vertices(
    strip_vectors: np.ndarray,
    span_edges: np.ndarray,
) -> np.ndarray:
    """Map strip-centre geometry vectors to nondegenerate edge vertices.

    Strength reconstruction retains its evidence-constrained root/tip
    conditions.  Geometry cannot use the strength's zero-tip condition:
    doing so collapses the tip face although its strength is zero.  The
    vertex map is the local finite-volume interpolation associated with the
    supplied strip centres and introduces no tunable smoothing length.
    """

    ns = len(span_edges) - 1
    vector = _finite("strip_vectors", strip_vectors, (ns, 3))
    width = np.diff(span_edges)
    result = np.empty((ns + 1, 3), dtype=float)
    result[0] = vector[0]
    result[-1] = vector[-1]
    if ns > 1:
        left = width[:-1]
        right = width[1:]
        result[1:-1] = (
            right[:, None] * vector[:-1]
            + left[:, None] * vector[1:]
        ) / (left + right)[:, None]
    return result


@dataclass(frozen=True)
class P2SpatialStep:
    """One same-stage source/bound solution returned to the solver."""

    bound_gamma: np.ndarray
    bound_pre: np.ndarray
    induced_velocity: np.ndarray
    release_previous: np.ndarray
    release_current: np.ndarray
    a0_geometry: np.ndarray
    a0_pre: np.ndarray
    a0_post: np.ndarray
    active: np.ndarray
    bound_residual: float
    lesp_residual: float
    source_condition_number: float
    bands: int
    outflow_bands: int
    appended: bool
    self_advection_included: bool = False


class P2SpatialLEVCandidate:
    """Chronological causal P2 release histories for one half wing."""

    def __init__(
        self,
        *,
        ns: int,
        span_edges,
        u_infinity: float,
        dt: float,
        lesp_crit: float,
        quadrature_order: int,
        max_bands: int,
        mirror_halfwing: bool,
    ):
        if not isinstance(ns, (int, np.integer)) or ns < 1:
            raise DistributedDoubletError("ns must be a positive integer")
        edges = np.asarray(span_edges, dtype=float)
        if (
            edges.shape != (int(ns) + 1,)
            or not np.all(np.isfinite(edges))
            or np.any(np.diff(edges) <= 0.0)
        ):
            raise DistributedDoubletError(
                "span_edges must contain ns+1 finite increasing values"
            )
        if not np.isfinite(u_infinity) or u_infinity <= 0.0:
            raise DistributedDoubletError(
                "u_infinity must be positive and finite"
            )
        if not np.isfinite(dt) or dt <= 0.0:
            raise DistributedDoubletError("dt must be positive and finite")
        if not np.isfinite(lesp_crit) or lesp_crit <= 0.0:
            raise DistributedDoubletError(
                "lesp_crit must be positive and finite"
            )
        if (
            not isinstance(quadrature_order, (int, np.integer))
            or quadrature_order < 2
        ):
            raise DistributedDoubletError(
                "quadrature_order must be an integer >=2"
            )
        if not isinstance(max_bands, (int, np.integer)) or max_bands < 1:
            raise DistributedDoubletError(
                "max_bands must be a positive integer"
            )
        self.ns = int(ns)
        self.span_edges = edges.copy()
        self.u_infinity = float(u_infinity)
        self.dt = float(dt)
        self.lesp_crit = float(lesp_crit)
        self.quadrature_order = int(quadrature_order)
        self.max_bands = int(max_bands)
        self.mirror_halfwing = bool(mirror_halfwing)
        self.histories: list[P2LEVHistory] = []
        self.active_history: P2LEVHistory | None = None
        self.release_previous = np.zeros(self.ns)
        self.step_index = 0
        self._band_serial = 0

    @property
    def surfaces(self) -> tuple:
        return tuple(
            band.surface
            for history in self.histories
            for band in history.bands
        )

    @property
    def band_count(self) -> int:
        return sum(len(history.bands) for history in self.histories)

    @property
    def outflow_count(self) -> int:
        return sum(len(history.outflow) for history in self.histories)

    def induced_velocity(self, points) -> np.ndarray:
        point_array = np.asarray(points, dtype=float)
        if point_array.ndim != 2 or point_array.shape[1] != 3:
            raise DistributedDoubletError(
                "points must have shape (n,3)"
            )
        if not self.surfaces:
            return np.zeros_like(point_array)
        return vectorized_induced_velocity(
            self.surfaces,
            point_array,
            quadrature_order=self.quadrature_order,
            mirror_halfwing=self.mirror_halfwing,
        )

    def _new_history(self) -> P2LEVHistory:
        history = P2LEVHistory(
            history_id=f"n3.1j0-event-{len(self.histories)}",
            max_bands=self.max_bands,
        )
        self.histories.append(history)
        self.active_history = history
        self.release_previous = np.zeros(self.ns)
        return history

    def _append_band(self, history: P2LEVHistory, band) -> None:
        """Append one band and enforce one global material-wake window.

        ``max_bands`` is a solver-wide wake-row budget, not a budget that may
        be restarted for every separated-flow event.  Histories remain
        separate so a zero-strength interval is never disguised as a
        continuous material seam, while eviction always follows the global
        chronological order.
        """

        history.append(band)
        while self.band_count > self.max_bands:
            for chronological in self.histories:
                if chronological.bands:
                    chronological.evict_oldest()
                    break
            else:  # pragma: no cover - protects the count invariant itself.
                raise DistributedDoubletError(
                    "global P2 wake window could not find its oldest band"
                )

    def solve_step(
        self,
        *,
        time: float,
        aic,
        rhs_without_p2,
        collocation,
        normals,
        leading_edge,
        chord_tangent,
        suction_normal,
        alpha_rad,
        chord,
        delta_x_front,
    ) -> P2SpatialStep:
        """Solve and append the causal current release row."""

        count = np.asarray(rhs_without_p2).size
        matrix = _finite("aic", aic, (count, count))
        rhs_base = _finite("rhs_without_p2", rhs_without_p2, (count,))
        points = _finite("collocation", collocation, (count, 3))
        normal = _finite("normals", normals, (count, 3))
        le = _finite("leading_edge", leading_edge, (self.ns + 1, 3))
        tangent = _finite("chord_tangent", chord_tangent, (self.ns, 3))
        suction = _finite("suction_normal", suction_normal, (self.ns, 3))
        alpha = _finite("alpha_rad", alpha_rad, (self.ns,))
        local_chord = _finite("chord", chord, (self.ns,))
        delta_x = _finite("delta_x_front", delta_x_front, (self.ns,))

        existing_velocity = self.induced_velocity(points)
        rhs_existing = rhs_base - np.einsum(
            "ij,ij->i", existing_velocity, normal
        )
        try:
            bound_geometry = np.linalg.solve(matrix, rhs_existing)
        except np.linalg.LinAlgError as error:
            raise DistributedDoubletError(
                "P2 spatial candidate bound geometry solve is singular"
            ) from error
        a0_geometry = lesp_eq6(
            bound_geometry[: self.ns],
            self.u_infinity,
            local_chord,
            delta_x,
        )
        geometry_active = np.abs(a0_geometry) > self.lesp_crit

        previous = self.release_previous.copy()
        needs_closing_row = bool(np.any(np.abs(previous) > 0.0))
        if not np.any(geometry_active) and not needs_closing_row:
            self.active_history = None
            self.step_index += 1
            residual = matrix @ bound_geometry - rhs_existing
            return P2SpatialStep(
                bound_gamma=bound_geometry,
                bound_pre=bound_geometry.copy(),
                induced_velocity=existing_velocity,
                release_previous=previous,
                release_current=np.zeros(self.ns),
                a0_geometry=a0_geometry,
                a0_pre=a0_geometry.copy(),
                a0_post=a0_geometry.copy(),
                active=np.zeros(self.ns, dtype=bool),
                bound_residual=float(
                    np.max(np.abs(residual), initial=0.0)
                ),
                lesp_residual=0.0,
                source_condition_number=1.0,
                bands=self.band_count,
                outflow_bands=self.outflow_count,
                appended=False,
            )

        history = self.active_history
        if history is None:
            history = self._new_history()
            previous_edge = le
            previous = np.zeros(self.ns)
        else:
            current_edge = history.current_edge
            if current_edge is None:
                previous_edge = le
            else:
                previous_edge = np.asarray(current_edge, dtype=float)

        # Ramesh supplies placement only.  The continuous geometry is built
        # from the full spanwise observer; the coupled solve below, not a
        # geometry collapse, sets inactive current-row strengths to zero.
        # Retaining subcritical placement also prevents zero-area faces at
        # active/inactive span junctions.
        placement_a0 = a0_geometry
        displacement_2d = first_lev_displacement_ramesh_2d(
            self.u_infinity,
            placement_a0,
            alpha,
            self.dt,
        )
        displacement_strip = embed_chord_normal_displacement(
            displacement_2d,
            tangent,
            suction,
        )
        current_edge = le + _strip_vectors_to_vertices(
            displacement_strip,
            self.span_edges,
        )
        time_nodes = np.array(
            [time - self.dt, time - 0.5 * self.dt, time],
            dtype=float,
        )
        known_band = causal_band(
            sheet_id=f"n3.1j0-known-{self._band_serial}",
            previous_edge=previous_edge,
            current_edge=current_edge,
            span_edges=self.span_edges,
            time_nodes=time_nodes,
            q_prev=previous,
            q_mid=0.5 * previous,
            q_now=np.zeros(self.ns),
        )
        known_velocity = vectorized_induced_velocity(
            (known_band.surface,),
            points,
            quadrature_order=self.quadrature_order,
            mirror_halfwing=self.mirror_halfwing,
        )

        unit_velocity = np.empty((count, 3, self.ns), dtype=float)
        unit_bands = []
        for source_strip in range(self.ns):
            unit = np.zeros(self.ns)
            unit[source_strip] = 1.0
            band = causal_band(
                sheet_id=(
                    f"n3.1j0-unit-{self._band_serial}-{source_strip}"
                ),
                previous_edge=previous_edge,
                current_edge=current_edge,
                span_edges=self.span_edges,
                time_nodes=time_nodes,
                q_prev=np.zeros(self.ns),
                q_mid=0.5 * unit,
                q_now=unit,
            )
            unit_bands.append(band)
            unit_velocity[:, :, source_strip] = (
                vectorized_induced_velocity(
                    (band.surface,),
                    points,
                    quadrature_order=self.quadrature_order,
                    mirror_halfwing=self.mirror_halfwing,
                )
            )
        normal_influence = np.einsum(
            "iks,ik->is", unit_velocity, normal
        )
        rhs_known = rhs_base - np.einsum(
            "ij,ij->i",
            existing_velocity + known_velocity,
            normal,
        )
        stage = solve_coupled_lesp_dde_stage(
            aic=matrix,
            rhs_without_nascent_lev=rhs_known,
            unit_lev_normal_influence=normal_influence,
            u_infinity=self.u_infinity,
            chord=local_chord,
            delta_x_front=delta_x,
            lesp_crit=self.lesp_crit,
        )
        if not stage.passed:
            raise DistributedDoubletError(
                "coupled P2 source/bound equations failed: "
                f"bound={stage.bound_equation_max_abs_residual:.3e}, "
                f"LESP={stage.lesp_active_max_abs_residual:.3e}"
            )
        current = stage.gamma_lev
        actual_band = causal_band(
            sheet_id=f"n3.1j0-band-{self._band_serial}",
            previous_edge=previous_edge,
            current_edge=current_edge,
            span_edges=self.span_edges,
            time_nodes=time_nodes,
            q_prev=previous,
            q_mid=0.5 * (previous + current),
            q_now=current,
        )
        self._append_band(history, actual_band)
        self._band_serial += 1

        current_velocity = np.einsum(
            "iks,s->ik", unit_velocity, current
        )
        total_velocity = (
            existing_velocity + known_velocity + current_velocity
        )
        rhs_final = rhs_base - np.einsum(
            "ij,ij->i", total_velocity, normal
        )
        bound_residual = matrix @ stage.bound_post - rhs_final
        maximum_bound_residual = float(
            np.max(np.abs(bound_residual), initial=0.0)
        )
        if maximum_bound_residual > 1.0e-9:
            raise DistributedDoubletError(
                "linear P2 field/source ledger does not reproduce the "
                f"coupled bound solve: {maximum_bound_residual:.3e}"
            )

        self.release_previous = current.copy()
        if not np.any(np.abs(current) > 0.0):
            self.active_history = None
            self.release_previous.fill(0.0)
        self.step_index += 1
        return P2SpatialStep(
            bound_gamma=stage.bound_post,
            bound_pre=stage.bound_pre,
            induced_velocity=total_velocity,
            release_previous=previous,
            release_current=current.copy(),
            a0_geometry=a0_geometry,
            a0_pre=stage.a0_pre,
            a0_post=stage.a0_post,
            active=stage.active,
            bound_residual=maximum_bound_residual,
            lesp_residual=stage.lesp_active_max_abs_residual,
            source_condition_number=stage.lesp_condition_number,
            bands=self.band_count,
            outflow_bands=self.outflow_count,
            appended=True,
        )

    def convect_heun(
        self,
        external_velocity: Callable[[np.ndarray], np.ndarray],
    ) -> None:
        """Advance every material band with the caller's N1 transport field."""

        for history in self.histories:
            history.convect_heun(external_velocity, self.dt)

    def diagnostics(self) -> dict:
        return {
            "model": "causal-continuous-P2-near-LEV",
            "self_advection_included": False,
            "quadrature_order": self.quadrature_order,
            "histories": len(self.histories),
            "bands": self.band_count,
            "outflow_bands": self.outflow_count,
            "release_previous": self.release_previous.tolist(),
            "lesp_crit": self.lesp_crit,
            "lesp_crit_evidence_status": "inherited-uncertain",
        }
