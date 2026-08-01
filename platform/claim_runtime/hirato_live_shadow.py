"""Canonical no-force Hirato finite-wing LEV/TEV state machine.

This module is the first forward spatial-state implementation for N3.1i.  It
owns its trailing-edge and leading-edge sheets, applies Hirato Eq.6/7/9/24 in
one chronological solve, and exposes all circulation and convection ledgers.
It deliberately does not evaluate pressure or aerodynamic force.

The implementation is a research shadow, not a production closure:

* the validated N1 bound-lattice geometry and AIC identity are reused without
  modifying N1;
* first-LEV placement uses the preregistered Ramesh ``P-R`` adapter;
* the TEV track direction and missing-history Eq.24 startup are explicit
  adapter hypotheses and remain ``partial``;
* no force target, pressure cap, damping, or empirical normalization exists.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .hirato_equations import (
    HiratoEquationError,
    embed_chord_normal_displacement,
    first_lev_displacement_ramesh_2d,
    lesp_eq6,
    lesp_sensitivity_eq6,
    preconstraint_shed_mask,
    rearmost_bound_aft_edge,
    rollup_displacement_eq24,
    solve_lesp_constraint,
    tev_first_displacement_hirato,
    tev_strength_eq9,
)
from .hirato_shadow import (
    Eq24ConvectionReport,
    HiratoSheetShadow,
    LocalVelocityLedger,
    SheetSnapshot,
    mirrored_ring_field,
    ring_field_velocity,
    ring_field_velocity_lamb_oseen,
)


def _finite(name: str, value, shape: tuple[int, ...] | None = None) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if shape is not None and array.shape != shape:
        raise HiratoEquationError(
            f"{name} must have shape {shape}, got {array.shape}"
        )
    if not np.all(np.isfinite(array)):
        raise HiratoEquationError(f"{name} contains non-finite values")
    return array


def _unit(vector: np.ndarray, name: str) -> np.ndarray:
    norm = np.linalg.norm(vector, axis=-1)
    if np.any(~np.isfinite(norm)) or np.any(norm <= 0.0):
        raise HiratoEquationError(f"{name} contains zero/non-finite vectors")
    return vector / norm[..., None]


@dataclass(frozen=True)
class BoundLattice:
    """N1-compatible vortex-ring geometry at one time step."""

    rings: np.ndarray
    collocation: np.ndarray
    normals: np.ndarray
    collocation_velocity: np.ndarray
    leading_edges: np.ndarray
    trailing_edges: np.ndarray
    trailing_edge_velocity: np.ndarray
    chord_tangent: np.ndarray
    suction_normal: np.ndarray
    chord: np.ndarray
    delta_x_front: np.ndarray


def build_bound_lattice(
    corners,
    corner_velocity,
    *,
    nc: int,
    ns: int,
) -> BoundLattice:
    """Reproduce the frozen N1 quarter-chord ring/collocation geometry."""
    if nc <= 0 or ns <= 0:
        raise HiratoEquationError("nc and ns must be positive")
    expected = (nc + 1, ns + 1, 3)
    geometry = _finite("corners", corners, expected)
    velocity = _finite("corner_velocity", corner_velocity, expected)

    rings = np.empty((nc * ns, 4, 3))
    collocation = np.empty((nc * ns, 3))
    normals = np.empty((nc * ns, 3))
    collocation_velocity = np.empty((nc * ns, 3))
    for i in range(nc):
        for j in range(ns):
            p = i * ns + j
            c00 = geometry[i, j]
            c10 = geometry[i + 1, j]
            c01 = geometry[i, j + 1]
            c11 = geometry[i + 1, j + 1]
            qfl = 0.75 * c00 + 0.25 * c10
            qfr = 0.75 * c01 + 0.25 * c11
            if i < nc - 1:
                cn1 = geometry[i + 2, j]
                cn1b = geometry[i + 2, j + 1]
                qbl = 0.75 * c10 + 0.25 * cn1
                qbr = 0.75 * c11 + 0.25 * cn1b
            else:
                qbl = c10 + 0.25 * (c10 - c00)
                qbr = c11 + 0.25 * (c11 - c01)
            rings[p] = np.stack([qfl, qfr, qbr, qbl])
            collocation[p] = 0.5 * (
                0.25 * c00 + 0.75 * c10 + 0.25 * c01 + 0.75 * c11
            )
            raw_normal = np.cross(c11 - c00, c01 - c10)
            normals[p] = _unit(raw_normal[None], "panel normal")[0]

            v00 = velocity[i, j]
            v10 = velocity[i + 1, j]
            v01 = velocity[i, j + 1]
            v11 = velocity[i + 1, j + 1]
            collocation_velocity[p] = 0.5 * (
                0.25 * v00 + 0.75 * v10 + 0.25 * v01 + 0.75 * v11
            )

    ring_chord_vector = 0.5 * (
        (rings[:, 2] - rings[:, 0]) + (rings[:, 3] - rings[:, 1])
    )
    ring_chord = np.linalg.norm(ring_chord_vector, axis=1).reshape(nc, ns)
    chord = np.sum(ring_chord, axis=0)
    delta_x_front = ring_chord[0].copy()

    left_chord = geometry[-1, :-1] - geometry[0, :-1]
    right_chord = geometry[-1, 1:] - geometry[0, 1:]
    chord_tangent = _unit(left_chord + right_chord, "strip chord tangent")
    span_tangent = _unit(
        geometry[0, 1:] - geometry[0, :-1],
        "leading-edge span tangent",
    )
    suction_normal = _unit(
        np.cross(chord_tangent, span_tangent),
        "strip suction normal",
    )
    chord_tangent = _unit(
        np.cross(span_tangent, suction_normal),
        "orthogonal strip chord tangent",
    )

    return BoundLattice(
        rings=rings,
        collocation=collocation,
        normals=normals,
        collocation_velocity=collocation_velocity,
        leading_edges=np.stack(
            [geometry[0, :-1], geometry[0, 1:]],
            axis=1,
        ),
        trailing_edges=np.stack(
            [geometry[-1, :-1], geometry[-1, 1:]],
            axis=1,
        ),
        trailing_edge_velocity=np.stack(
            [velocity[-1, :-1], velocity[-1, 1:]],
            axis=1,
        ),
        chord_tangent=chord_tangent,
        suction_normal=suction_normal,
        chord=chord,
        delta_x_front=delta_x_front,
    )


def _field_velocity(
    points: np.ndarray,
    rings: np.ndarray,
    gamma: np.ndarray,
    *,
    core_radius: float,
    mirror_symmetry: bool,
    denominator_floor: float = 1e-30,
    core_model: str = "van_garrel",
) -> np.ndarray:
    if core_model == "hirato_lamb_oseen":
        velocity = ring_field_velocity_lamb_oseen(
            points,
            rings,
            gamma,
            core_radius,
        )
    elif core_model == "van_garrel":
        velocity = ring_field_velocity(
            points,
            rings,
            gamma,
            core_radius,
            denominator_floor=denominator_floor,
        )
    else:
        raise HiratoEquationError(f"unknown core model {core_model!r}")
    if mirror_symmetry and len(rings):
        mirrored = mirrored_ring_field(rings)
        if core_model == "hirato_lamb_oseen":
            velocity += ring_field_velocity_lamb_oseen(
                points,
                mirrored,
                gamma,
                core_radius,
            )
        else:
            velocity += ring_field_velocity(
                points,
                mirrored,
                gamma,
                core_radius,
                denominator_floor=denominator_floor,
            )
    return velocity


def build_bound_aic(
    lattice: BoundLattice,
    *,
    mirror_symmetry: bool,
) -> np.ndarray:
    """Build the singular N1 AIC without applying a free-sheet core."""
    npan = len(lattice.rings)
    matrix = np.empty((npan, npan))
    for source in range(npan):
        induced = _field_velocity(
            lattice.collocation,
            lattice.rings[source:source + 1],
            np.ones(1),
            core_radius=0.0,
            mirror_symmetry=mirror_symmetry,
            # Frozen N1 ``vseg`` uses this explicit cross-product guard.
            denominator_floor=1e-10,
        )
        matrix[:, source] = np.einsum(
            "ij,ij->i",
            induced,
            lattice.normals,
        )
    if not np.all(np.isfinite(matrix)):
        raise HiratoEquationError("bound AIC contains non-finite values")
    return matrix


def _empty_rings() -> np.ndarray:
    return np.empty((0, 4, 3), dtype=float)


def _empty_gamma() -> np.ndarray:
    return np.empty(0, dtype=float)


def _velocity_ledger(
    points: np.ndarray,
    shape: tuple[int, ...],
    *,
    bound_rings: np.ndarray,
    bound_gamma: np.ndarray,
    tev_rings: np.ndarray,
    tev_gamma: np.ndarray,
    lev_rings: np.ndarray,
    lev_gamma: np.ndarray,
    u_infinity: np.ndarray,
    core_radius: float,
    mirror_symmetry: bool,
    backend: str,
    warp_device: str,
) -> LocalVelocityLedger:
    flat = points.reshape(-1, 3)
    if backend == "warp":
        from .hirato_warp_backend import velocity_channels

        channels = velocity_channels(
            flat,
            bound_rings=bound_rings,
            bound_gamma=bound_gamma,
            tev_rings=tev_rings,
            tev_gamma=tev_gamma,
            lev_rings=lev_rings,
            lev_gamma=lev_gamma,
            u_infinity=u_infinity,
            core_radius=core_radius,
            mirror_symmetry=mirror_symmetry,
            device=warp_device,
        )
        return LocalVelocityLedger(
            freestream=channels.freestream.reshape(shape),
            bound=channels.bound.reshape(shape),
            tev=channels.tev.reshape(shape),
            lev=channels.lev.reshape(shape),
            total=channels.total.reshape(shape),
        )
    if backend != "numpy":
        raise HiratoEquationError(
            f"velocity backend must be 'numpy' or 'warp', got {backend!r}"
        )
    freestream = np.broadcast_to(u_infinity, flat.shape).copy()
    bound = _field_velocity(
        flat,
        bound_rings,
        bound_gamma,
        core_radius=core_radius,
        mirror_symmetry=mirror_symmetry,
        core_model="hirato_lamb_oseen",
    )
    tev = _field_velocity(
        flat,
        tev_rings,
        tev_gamma,
        core_radius=core_radius,
        mirror_symmetry=mirror_symmetry,
        core_model="hirato_lamb_oseen",
    )
    lev = _field_velocity(
        flat,
        lev_rings,
        lev_gamma,
        core_radius=core_radius,
        mirror_symmetry=mirror_symmetry,
        core_model="hirato_lamb_oseen",
    )
    total = freestream + bound + tev + lev
    return LocalVelocityLedger(
        freestream=freestream.reshape(shape),
        bound=bound.reshape(shape),
        tev=tev.reshape(shape),
        lev=lev.reshape(shape),
        total=total.reshape(shape),
    )


@dataclass(frozen=True)
class TrailingSheetSnapshot:
    rings: np.ndarray
    gamma: np.ndarray
    strip: np.ndarray
    birth_step: np.ndarray
    ring_id: np.ndarray
    previous_vertex_velocity: np.ndarray


class HiratoTrailingSheetShadow:
    """Own TEV rows with Eq.9 strength and an explicit 0.3-Udt track adapter."""

    def __init__(self, ns: int):
        if ns <= 0:
            raise HiratoEquationError("ns must be positive")
        self.ns = int(ns)
        self.rings = _empty_rings()
        self.gamma = _empty_gamma()
        self.strip = np.empty(0, dtype=np.int64)
        self.birth_step = np.empty(0, dtype=np.int64)
        self.ring_id = np.empty(0, dtype=np.int64)
        self.previous_vertex_velocity = np.empty((0, 4, 3))
        self._next_id = 0

    def snapshot(self) -> TrailingSheetSnapshot:
        return TrailingSheetSnapshot(
            rings=self.rings.copy(),
            gamma=self.gamma.copy(),
            strip=self.strip.copy(),
            birth_step=self.birth_step.copy(),
            ring_id=self.ring_id.copy(),
            previous_vertex_velocity=self.previous_vertex_velocity.copy(),
        )

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
        trailing_edges,
        track_velocity,
        u_infinity_speed: float,
        dt: float,
        gamma_now,
    ) -> np.ndarray:
        """Create the current TEV row before the bound-circulation solve.

        The published identity fixes the new forward edge's distance from the
        trailing edge at ``0.3 U_inf dt``.  Mapping that scalar distance along
        ``U_inf - V_TE`` is the registered ``T-track`` adapter.  The aft edge
        connects to the convected forward edge of the preceding row.
        """
        if step < 0:
            raise HiratoEquationError("step must be non-negative")
        edges = _finite("trailing_edges", trailing_edges, (self.ns, 2, 3))
        track = _finite("track_velocity", track_velocity, (self.ns, 2, 3))
        strengths = _finite("gamma_now", gamma_now, (self.ns,))
        if not np.isfinite(u_infinity_speed) or u_infinity_speed <= 0.0:
            raise HiratoEquationError("u_infinity_speed must be positive")
        if not np.isfinite(dt) or dt <= 0.0:
            raise HiratoEquationError("dt must be positive")
        direction = _unit(track, "TE relative-flow track")
        track_speed_vectors = direction * u_infinity_speed
        front = edges + tev_first_displacement_hirato(
            track_speed_vectors,
            dt,
        )

        rows = []
        ids = []
        for strip in range(self.ns):
            newest = self._newest_index(strip)
            if newest is None:
                aft = front[strip] + direction[strip] * (
                    u_infinity_speed * dt
                )
            else:
                aft = self.rings[newest, :2].copy()
            rows.append(
                np.stack(
                    [front[strip, 0], front[strip, 1], aft[1], aft[0]],
                    axis=0,
                )
            )
            ids.append(self._next_id)
            self._next_id += 1

        self.rings = np.concatenate([self.rings, np.stack(rows)], axis=0)
        self.gamma = np.concatenate([self.gamma, strengths])
        self.strip = np.concatenate(
            [self.strip, np.arange(self.ns, dtype=np.int64)]
        )
        self.birth_step = np.concatenate(
            [self.birth_step, np.full(self.ns, step, dtype=np.int64)]
        )
        self.ring_id = np.concatenate(
            [self.ring_id, np.asarray(ids, dtype=np.int64)]
        )
        self.previous_vertex_velocity = np.concatenate(
            [
                self.previous_vertex_velocity,
                np.full((self.ns, 4, 3), np.nan),
            ],
            axis=0,
        )
        return np.asarray(ids, dtype=np.int64)

    def convect_eq24(
        self,
        vertex_velocity,
        dt: float,
        *,
        step: int,
    ) -> Eq24ConvectionReport:
        velocity = _finite("vertex_velocity", vertex_velocity, self.rings.shape)
        if step < 0 or np.any(self.birth_step > step):
            raise HiratoEquationError("invalid TEV convection step")
        if not np.isfinite(dt) or dt <= 0.0:
            raise HiratoEquationError("dt must be positive")
        bootstrap = ~np.all(
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
            bootstrap_vertex=bootstrap,
            displacement=displacement.copy(),
        )


@dataclass(frozen=True)
class HiratoLiveStepReport:
    step: int
    a0_event: np.ndarray
    a0_zero_strength: np.ndarray
    a0_post: np.ndarray
    active: np.ndarray
    new_sheet: np.ndarray
    gamma_lev: np.ndarray
    gamma_tev: np.ndarray
    eq9_residual: np.ndarray
    lesp_active_residual: np.ndarray
    lesp_condition_number: float
    bound_gamma: np.ndarray
    bound_rings: np.ndarray
    pseudo_rings: np.ndarray
    tev_pre_convection: TrailingSheetSnapshot
    lev_pre_convection: SheetSnapshot
    tev_post_convection: TrailingSheetSnapshot
    lev_post_convection: SheetSnapshot
    tev_bootstrap_vertices: int
    lev_bootstrap_vertices: int
    convection_ledger_max_abs_residual: float


class HiratoLiveShadow:
    """Stateful Eq.6/7/9/24 solve with no pressure or force output."""

    def __init__(
        self,
        *,
        nc: int,
        ns: int,
        u_infinity,
        dt: float,
        lesp_crit: float,
        core_radius: float,
        mirror_symmetry: bool,
        velocity_backend: str = "numpy",
        warp_device: str = "cpu",
    ):
        if nc <= 0 or ns <= 0:
            raise HiratoEquationError("nc and ns must be positive")
        self.nc = int(nc)
        self.ns = int(ns)
        self.u_infinity = _finite("u_infinity", u_infinity, (3,))
        self.u_infinity_speed = float(np.linalg.norm(self.u_infinity))
        if self.u_infinity_speed <= 0.0:
            raise HiratoEquationError("u_infinity must be nonzero")
        if not np.isfinite(dt) or dt <= 0.0:
            raise HiratoEquationError("dt must be positive")
        if not np.isfinite(lesp_crit) or lesp_crit <= 0.0:
            raise HiratoEquationError("lesp_crit must be positive")
        if not np.isfinite(core_radius) or core_radius <= 0.0:
            raise HiratoEquationError("core_radius must be positive")
        self.dt = float(dt)
        self.lesp_crit = float(lesp_crit)
        self.core_radius = float(core_radius)
        self.mirror_symmetry = bool(mirror_symmetry)
        if velocity_backend not in ("numpy", "warp"):
            raise HiratoEquationError(
                "velocity_backend must be 'numpy' or 'warp'"
            )
        if not isinstance(warp_device, str) or not warp_device:
            raise HiratoEquationError("warp_device must be a nonempty string")
        self.velocity_backend = velocity_backend
        self.warp_device = warp_device
        self.tev = HiratoTrailingSheetShadow(ns)
        self.lev = HiratoSheetShadow(ns)
        self.previous_bound_rear = np.zeros(ns)
        self.previous_gamma_lev = np.zeros(ns)
        self.previous_active = np.zeros(ns, dtype=bool)
        self._next_step = 0

    def _free_induction(
        self,
        lattice: BoundLattice,
    ) -> np.ndarray:
        if self.velocity_backend == "warp":
            ledger = _velocity_ledger(
                lattice.collocation,
                lattice.collocation.shape,
                bound_rings=_empty_rings(),
                bound_gamma=_empty_gamma(),
                tev_rings=self.tev.rings,
                tev_gamma=self.tev.gamma,
                lev_rings=self.lev.rings,
                lev_gamma=self.lev.gamma,
                u_infinity=np.zeros(3),
                core_radius=self.core_radius,
                mirror_symmetry=self.mirror_symmetry,
                backend=self.velocity_backend,
                warp_device=self.warp_device,
            )
            return ledger.total
        return (
            _field_velocity(
                lattice.collocation,
                self.tev.rings,
                self.tev.gamma,
                core_radius=self.core_radius,
                mirror_symmetry=self.mirror_symmetry,
                core_model="hirato_lamb_oseen",
            )
            + _field_velocity(
                lattice.collocation,
                self.lev.rings,
                self.lev.gamma,
                core_radius=self.core_radius,
                mirror_symmetry=self.mirror_symmetry,
                core_model="hirato_lamb_oseen",
            )
        )

    def _rhs(
        self,
        lattice: BoundLattice,
    ) -> np.ndarray:
        relative = (
            self.u_infinity[None]
            - lattice.collocation_velocity
            + self._free_induction(lattice)
        )
        return -np.einsum("ij,ij->i", relative, lattice.normals)

    def _pseudo_rings(self, lattice: BoundLattice) -> np.ndarray:
        aft_right, aft_left = rearmost_bound_aft_edge(
            lattice.rings,
            self.nc,
            self.ns,
        )
        return np.stack(
            [
                lattice.leading_edges[:, 0],
                lattice.leading_edges[:, 1],
                aft_right,
                aft_left,
            ],
            axis=1,
        )

    def _first_lev_aft_edges(
        self,
        lattice: BoundLattice,
        a0_event: np.ndarray,
    ) -> np.ndarray:
        local_alpha = np.arctan2(
            -lattice.chord_tangent[:, 2],
            lattice.chord_tangent[:, 0],
        )
        displacement_2d = first_lev_displacement_ramesh_2d(
            self.u_infinity_speed,
            a0_event,
            local_alpha,
            self.dt,
        )
        displacement = embed_chord_normal_displacement(
            displacement_2d,
            lattice.chord_tangent,
            lattice.suction_normal,
        )
        return lattice.leading_edges + displacement[:, None, :]

    def _unit_lev_normal_influence(
        self,
        lattice: BoundLattice,
        created_ids: np.ndarray,
        pseudo_rings: np.ndarray,
    ) -> np.ndarray:
        influence = np.zeros((self.nc * self.ns, self.ns))
        for ring_id in created_ids:
            index = np.flatnonzero(self.lev.ring_id == ring_id)
            if index.size != 1:
                raise HiratoEquationError("new LEV identity lookup failed")
            ring_index = int(index[0])
            strip = int(self.lev.strip[ring_index])
            source_rings = np.stack(
                [self.lev.rings[ring_index], pseudo_rings[strip]],
                axis=0,
            )
            induced = _field_velocity(
                lattice.collocation,
                source_rings,
                np.ones(2),
                core_radius=self.core_radius,
                mirror_symmetry=self.mirror_symmetry,
                core_model="hirato_lamb_oseen",
            )
            influence[:, strip] = np.einsum(
                "ij,ij->i",
                induced,
                lattice.normals,
            )
        return influence

    def step(
        self,
        *,
        step: int,
        corners,
        corner_velocity,
    ) -> HiratoLiveStepReport:
        if step != self._next_step:
            raise HiratoEquationError(
                f"expected step {self._next_step}, got {step}"
            )
        lattice = build_bound_lattice(
            corners,
            corner_velocity,
            nc=self.nc,
            ns=self.ns,
        )
        aic = build_bound_aic(
            lattice,
            mirror_symmetry=self.mirror_symmetry,
        )

        gamma_tev = tev_strength_eq9(
            self.previous_bound_rear,
            self.previous_gamma_lev,
        )
        tev_ids = self.tev.shed(
            step=step,
            trailing_edges=lattice.trailing_edges,
            track_velocity=(
                self.u_infinity[None, None]
                - lattice.trailing_edge_velocity
            ),
            u_infinity_speed=self.u_infinity_speed,
            dt=self.dt,
            gamma_now=gamma_tev,
        )
        tev_indices = [
            int(np.flatnonzero(self.tev.ring_id == ring_id)[0])
            for ring_id in tev_ids
        ]
        eq9_residual = self.tev.gamma[tev_indices] - gamma_tev

        rhs_event = self._rhs(lattice)
        bound_event = np.linalg.solve(aic, rhs_event)
        a0_event = lesp_eq6(
            bound_event[:self.ns],
            self.u_infinity_speed,
            lattice.chord,
            lattice.delta_x_front,
        )
        active = preconstraint_shed_mask(a0_event, self.lesp_crit)
        new_sheet = active & ~self.previous_active
        gamma_lev = np.zeros(self.ns)
        a0_zero = a0_event.copy()
        a0_post = a0_event.copy()
        lesp_residual = np.empty(0)
        condition = 1.0
        pseudo_rings = self._pseudo_rings(lattice)

        if np.any(active):
            shed_report = self.lev.shed(
                step=step,
                leading_edges=lattice.leading_edges,
                gamma_now=np.zeros(self.ns),
                active=active,
                first_aft_edges=self._first_lev_aft_edges(
                    lattice,
                    a0_event,
                ),
                new_sheet=new_sheet,
            )
            # Eq.7 remeshing changes the old free-sheet geometry.  Recompute
            # the zero-new-strength base before forming the affine constraint.
            rhs_zero = self._rhs(lattice)
            bound_zero = np.linalg.solve(aic, rhs_zero)
            a0_zero = lesp_eq6(
                bound_zero[:self.ns],
                self.u_infinity_speed,
                lattice.chord,
                lattice.delta_x_front,
            )
            normal_influence = self._unit_lev_normal_influence(
                lattice,
                shed_report.created_ids,
                pseudo_rings,
            )
            bound_response_columns = np.linalg.solve(
                aic,
                -normal_influence,
            )
            sensitivity = lesp_sensitivity_eq6(
                bound_response_columns.T,
                self.u_infinity_speed,
                lattice.chord,
                lattice.delta_x_front,
            )
            constraint = solve_lesp_constraint(
                a0_zero,
                sensitivity,
                active,
                self.lesp_crit,
            )
            gamma_lev = constraint.gamma_lev
            created_strengths = []
            for ring_id in shed_report.created_ids:
                index = int(np.flatnonzero(self.lev.ring_id == ring_id)[0])
                created_strengths.append(gamma_lev[self.lev.strip[index]])
            self.lev.assign_created_strengths(
                shed_report.created_ids,
                created_strengths,
            )
            rhs_final = rhs_zero - normal_influence @ gamma_lev
            bound_gamma = np.linalg.solve(aic, rhs_final)
            a0_post = lesp_eq6(
                bound_gamma[:self.ns],
                self.u_infinity_speed,
                lattice.chord,
                lattice.delta_x_front,
            )
            lesp_residual = (
                a0_post[active]
                - self.lesp_crit * np.sign(a0_zero[active])
            )
            condition = constraint.condition_number
        else:
            bound_gamma = bound_event

        pseudo_gamma = gamma_lev.copy()
        tev_pre = self.tev.snapshot()
        lev_pre = self.lev.snapshot()
        bound_roll_rings = np.concatenate(
            [lattice.rings, pseudo_rings],
            axis=0,
        )
        bound_roll_gamma = np.concatenate(
            [bound_gamma, pseudo_gamma],
        )
        tev_ledger = _velocity_ledger(
            self.tev.rings,
            self.tev.rings.shape,
            bound_rings=bound_roll_rings,
            bound_gamma=bound_roll_gamma,
            tev_rings=self.tev.rings,
            tev_gamma=self.tev.gamma,
            lev_rings=self.lev.rings,
            lev_gamma=self.lev.gamma,
            u_infinity=self.u_infinity,
            core_radius=self.core_radius,
            mirror_symmetry=self.mirror_symmetry,
            backend=self.velocity_backend,
            warp_device=self.warp_device,
        )
        lev_ledger = _velocity_ledger(
            self.lev.rings,
            self.lev.rings.shape,
            bound_rings=bound_roll_rings,
            bound_gamma=bound_roll_gamma,
            tev_rings=self.tev.rings,
            tev_gamma=self.tev.gamma,
            lev_rings=self.lev.rings,
            lev_gamma=self.lev.gamma,
            u_infinity=self.u_infinity,
            core_radius=self.core_radius,
            mirror_symmetry=self.mirror_symmetry,
            backend=self.velocity_backend,
            warp_device=self.warp_device,
        )
        tev_convection = self.tev.convect_eq24(
            tev_ledger.total,
            self.dt,
            step=step,
        )
        lev_convection = self.lev.convect_eq24(
            lev_ledger.total,
            self.dt,
            step=step,
        )
        tev_closure = np.max(
            np.abs(
                tev_ledger.total
                - tev_ledger.freestream
                - tev_ledger.bound
                - tev_ledger.tev
                - tev_ledger.lev
            )
        )
        lev_closure = (
            np.max(
                np.abs(
                    lev_ledger.total
                    - lev_ledger.freestream
                    - lev_ledger.bound
                    - lev_ledger.tev
                    - lev_ledger.lev
                )
            )
            if len(self.lev.rings)
            else 0.0
        )

        self.previous_bound_rear = bound_gamma.reshape(
            self.nc,
            self.ns,
        )[-1].copy()
        self.previous_gamma_lev = gamma_lev.copy()
        self.previous_active = active.copy()
        self._next_step += 1
        return HiratoLiveStepReport(
            step=int(step),
            a0_event=a0_event,
            a0_zero_strength=a0_zero,
            a0_post=a0_post,
            active=active,
            new_sheet=new_sheet,
            gamma_lev=gamma_lev,
            gamma_tev=gamma_tev,
            eq9_residual=eq9_residual,
            lesp_active_residual=lesp_residual,
            lesp_condition_number=float(condition),
            bound_gamma=bound_gamma,
            bound_rings=lattice.rings.copy(),
            pseudo_rings=pseudo_rings,
            tev_pre_convection=tev_pre,
            lev_pre_convection=lev_pre,
            tev_post_convection=self.tev.snapshot(),
            lev_post_convection=self.lev.snapshot(),
            tev_bootstrap_vertices=int(
                np.count_nonzero(tev_convection.bootstrap_vertex)
            ),
            lev_bootstrap_vertices=int(
                np.count_nonzero(lev_convection.bootstrap_vertex)
            ),
            convection_ledger_max_abs_residual=float(
                max(tev_closure, lev_closure)
            ),
        )
