"""Unified CUDA surface-load packet for the native V5M solver family.

M1 of MODIFICATION_PLAN_ROJ_ACCURACY_PERFORMANCE_20260830: rigid wings and
Q16 membranes must share ONE surface-load scientific object.  This module
defines that object and the structure-independent resolved KJ+dGamma
evaluator that produces it.

The packet carries the five point forces per panel (four Kutta-Joukowski
filament forces + one unsteady-Bernoulli point force), their application
points, the panel ownership map, and the force/moment/circulation ledger.
Every consumer — rigid Cn, Q16 exact transpose-mapped generalized force,
future body 6-DOF loads — must PROJECT this single object and never
re-integrate a parallel pressure field.

The evaluator is extracted verbatim from ``RigidAuthorLoadAssembler``
(Pterra-faithful ``_calculate_loads``); the rigid assembler now delegates
here so there is exactly one implementation.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Mapping

import torch

from fluxvortex.warp_fsi.kernels_ring_velocity import (
    _KERNEL_V3,
    mean_chordwise_panel,
    native_ring_velocity_expanded,
)

#: What the packet's closure tolerances are measured against (2026-11-24).
#:
#:   "net"  (the historical behaviour, the default)  rtol x |net resultant|.  The moment
#:          check has no absolute floor, so at a NEAR-ZERO load the floating-point noise of
#:          the recomputed sum (about 1e-19 N m over 2250 points) can exceed rtol x |M|:
#:          observed on the rigid plate at alpha = 10 deg in the first steps of the startup
#:          ramp, "moment closure 1.212e-19 exceeds 1e-12 x |M|", i.e. |M| below 1.2e-7 N m.
#:          The same step passes with a slightly different leading-edge regularisation
#:          constant: the trip is noise, not a load inconsistency.
#:   "l1"   rtol x (|net| + sum of |each point's contribution|), the floating-point
#:          forward-error bound of the reduction.  This is how the spanwise impulse ledger
#:          was already repaired (fluxv_v5m_four_case_20260822/execution.md): a closure
#:          error is judged against the L1 size of what is being summed, never against the
#:          cancelled result alone.
#:
#: Default "net" leaves every existing verdict unchanged.
_PACKET_CLOSURE_SCALE = os.environ.get("FLUXV_PACKET_CLOSURE_SCALE", "net").strip().lower()
if _PACKET_CLOSURE_SCALE not in {"net", "l1"}:
    raise ValueError(
        f"FLUXV_PACKET_CLOSURE_SCALE must be 'net' or 'l1', got {_PACKET_CLOSURE_SCALE!r}"
    )

# Geometry objects are duck-typed (``.rings``/``.ring_velocity``/
# ``.collocation``/``.areas``/``.normals``) to avoid an aero<->warp_fsi
# import cycle; the concrete type is NativeV5MGeometry.

_RING_TILE_BYTES = 64 * 1024 * 1024

POINT_KIND_FRONT = 0
POINT_KIND_RIGHT = 1
POINT_KIND_BACK = 2
POINT_KIND_LEFT = 3
POINT_KIND_UNSTEADY = 4


@dataclass(frozen=True, slots=True)
class NativeV5MSurfaceLoadPacket:
    """Immutable resolved point-load owner (5 forces per panel, CUDA fp64).

    ``point_kind`` selects front/right/back/left KJ filaments (native ring
    traversal 0->1->2->3->0) plus the unsteady-Bernoulli point;
    ``panel_owner`` maps every point back to its panel.  ``total_force_I``
    and ``total_moment_I`` are EXACT sums of ``point_forces_I`` (moment
    about the world origin) — the closure is a hard gate, not a diagnostic.
    ``component_ledger`` carries the author decomposition (lift1 / lift2 /
    wake-history / KJ / unsteady) for audit without a second integration.
    """

    point_positions_I: torch.Tensor  # (5P, 3)
    point_forces_I: torch.Tensor  # (5P, 3)
    point_kind: torch.Tensor  # (5P,) int64
    panel_owner: torch.Tensor  # (5P,) int64
    total_force_I: torch.Tensor  # (3,)
    total_moment_I: torch.Tensor  # (3,)
    component_ledger: Mapping[str, torch.Tensor] = field(default_factory=dict)
    device_generation: int = 0

    def __post_init__(self) -> None:
        count = self.point_forces_I.shape[0]
        for name, tensor in (
            ("point_positions_I", self.point_positions_I),
            ("point_forces_I", self.point_forces_I),
        ):
            if tensor.ndim != 2 or tensor.shape[1] != 3 or tensor.shape[0] != count:
                raise ValueError(f"{name} must have shape ({count}, 3)")
        if self.point_kind.shape != (count,) or self.panel_owner.shape != (count,):
            raise ValueError("point_kind/panel_owner must have shape (5P,)")
        if self.total_force_I.shape != (3,) or self.total_moment_I.shape != (3,):
            raise ValueError("total force/moment must have shape (3,)")

    def force_closure_error(self) -> float:
        return float(
            torch.linalg.vector_norm(
                self.total_force_I - torch.sum(self.point_forces_I, dim=0)
            ).item()
        )

    def moment_closure_error(self) -> float:
        recomputed = torch.sum(
            torch.linalg.cross(self.point_positions_I, self.point_forces_I, dim=1),
            dim=0,
        )
        return float(
            torch.linalg.vector_norm(self.total_moment_I - recomputed).item()
        )


def closure_scales(packet: NativeV5MSurfaceLoadPacket) -> tuple[float, float]:
    """(force scale, moment scale) that `validate_packet` measures its tolerances against."""

    force_net = float(torch.linalg.vector_norm(packet.total_force_I).item())
    moment_net = float(torch.linalg.vector_norm(packet.total_moment_I).item())
    if _PACKET_CLOSURE_SCALE == "net":
        return max(force_net, 1e-30), max(moment_net, 1e-30)
    force_l1 = float(torch.linalg.vector_norm(packet.point_forces_I, dim=1).sum().item())
    moment_l1 = float(
        torch.linalg.vector_norm(
            torch.linalg.cross(packet.point_positions_I, packet.point_forces_I, dim=1), dim=1
        ).sum().item()
    )
    return max(force_net + force_l1, 1e-30), max(moment_net + moment_l1, 1e-30)


def validate_packet(packet: NativeV5MSurfaceLoadPacket, *, rtol: float = 1e-12) -> None:
    scale, scale_m = closure_scales(packet)
    if packet.force_closure_error() > rtol * scale:
        raise ValueError(
            f"surface-load packet force closure {packet.force_closure_error():.3e} "
            f"exceeds {rtol:.0e} x |F|"
        )
    if packet.moment_closure_error() > rtol * scale_m:
        raise ValueError(
            f"surface-load packet moment closure {packet.moment_closure_error():.3e} "
            f"exceeds {rtol:.0e} x |M| (scale {_PACKET_CLOSURE_SCALE}, {scale_m:.3e} N m)"
        )


class NativeResolvedLoadAssembler:
    """Structure-independent resolved KJ+dGamma point-load evaluator.

    Extracted from the rigid path's ``RigidAuthorLoadAssembler`` so rigid
    and Q16 share ONE load implementation (M1-2).  All methods are pure
    functions of ``(geometry, gamma, solution velocities, history)``; no
    structural state is accepted anywhere.
    """

    def __init__(
        self,
        *,
        density: float,
        device: str | torch.device,
        chordwise_panels: int | None = None,
        spanwise_panels: int | None = None,
        aerodynamic_dt: float | None = None,
        wake_history_mode: str = "bound_rate",
    ) -> None:
        self.density = float(density)
        self.device = torch.device(device)
        self.chordwise_panels = (
            None if chordwise_panels is None else int(chordwise_panels)
        )
        self.spanwise_panels = (
            None if spanwise_panels is None else int(spanwise_panels)
        )
        self.aerodynamic_dt = (
            None if aerodynamic_dt is None else float(aerodynamic_dt)
        )
        if wake_history_mode not in {"material", "bound_rate"}:
            raise ValueError("wake_history_mode must be 'material' or 'bound_rate'")
        self.wake_history_mode = wake_history_mode
        self._previous_gamma: torch.Tensor | None = None

    # -- topology ---------------------------------------------------------
    def panel_topology(self, geometry) -> tuple[int, int]:
        count = int(geometry.normals.shape[0])
        ns = self.spanwise_panels
        if ns is None:
            ns = int(geometry.leading_edge.shape[0]) - 1
        if ns < 1 or count % ns:
            raise ValueError("native V5M grid topology inconsistent with load assembler")
        nc = count // ns
        if self.chordwise_panels is not None and nc != self.chordwise_panels:
            raise ValueError("native V5M chordwise topology differs from assembler")
        return nc, ns

    # -- velocity kernels (mirrors Q16NativeV5MSolver._ring_velocity) -----
    def bound_velocity(
        self, points: torch.Tensor, rings: torch.Tensor, gamma: torch.Tensor
    ) -> torch.Tensor:
        if rings.shape[0] == 0:
            return torch.zeros_like(points)
        if _KERNEL_V3:
            # plan V3 (P0-A): the exact law, as the solver's `_ring_velocity` uses it
            core_fraction = 0.0
            reference_length = 0.0
        else:
            core_fraction = 1.0e-6
            # the wing's own panel length, as the solver's `_ring_velocity` uses it (plan V3,
            # Bug-4); it was 1/nc, the author's one-metre chord
            reference_length = mean_chordwise_panel(rings)
        full_bytes = points.shape[0] * rings.shape[0] * 3 * 8
        if full_bytes <= _RING_TILE_BYTES:
            expanded = native_ring_velocity_expanded(
                points,
                rings,
                core_fraction=core_fraction,
                reference_length=reference_length,
            )
            return torch.sum(expanded * gamma[None, :, None], dim=1)
        total = torch.zeros_like(points)
        target_cap = max(1, _RING_TILE_BYTES // (rings.shape[0] * 3 * 8))
        for start in range(0, points.shape[0], target_cap):
            stop = min(start + target_cap, points.shape[0])
            expanded = native_ring_velocity_expanded(
                points[start:stop],
                rings,
                core_fraction=core_fraction,
                reference_length=reference_length,
            )
            total[start:stop] = torch.sum(expanded * gamma[None, :, None], dim=1)
        return total

    # -- circulation/history helpers --------------------------------------
    def delta_grid(
        self, gamma: torch.Tensor, mf2_history: torch.Tensor, nc: int, ns: int
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        if self.wake_history_mode == "bound_rate":
            gamma_rate = mf2_history
            return (mf2_history * self.aerodynamic_dt).reshape(nc, ns), gamma_rate
        if self._previous_gamma is not None:
            delta = (gamma - self._previous_gamma).reshape(nc, ns)
            return delta, delta.reshape(-1) / self.aerodynamic_dt
        return None, torch.zeros_like(gamma)

    @staticmethod
    def effective_leg_strengths(
        grid: torch.Tensor, delta_grid: torch.Tensor | None
    ) -> tuple[torch.Tensor, ...]:
        right = grid.clone()
        right[:, :-1] = 0.5 * (grid[:, :-1] - grid[:, 1:])
        front = grid.clone()
        front[1:, :] = 0.5 * (grid[1:, :] - grid[:-1, :])
        left = grid.clone()
        left[:, 1:] = 0.5 * (grid[:, 1:] - grid[:, :-1])
        back = grid.clone()
        back[:-1, :] = 0.5 * (grid[:-1, :] - grid[1:, :])
        if delta_grid is not None:
            back[-1, :] = delta_grid[-1, :]
        return tuple(part.reshape(-1) for part in (front, right, back, left))

    @staticmethod
    def leg_layout(
        geometry,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        rings = geometry.rings
        ring_velocity = geometry.ring_velocity
        mids, movements, lengths = [], [], []
        for start, end in ((0, 1), (1, 2), (2, 3), (3, 0)):
            a = rings[:, start]
            b = rings[:, end]
            mids.append(0.5 * (a + b))
            movements.append(-0.5 * (ring_velocity[:, start] + ring_velocity[:, end]))
            lengths.append(b - a)
        return mids, movements, lengths

    def leg_midpoints_flat(self, geometry) -> torch.Tensor:
        mids, _, _ = self.leg_layout(geometry)
        return torch.cat(mids, dim=0)

    # -- packet assembly ---------------------------------------------------
    def assemble_packet(
        self,
        *,
        geometry,
        gamma: torch.Tensor,
        solution_velocity_flat: torch.Tensor,
        mf2_history: torch.Tensor,
        component_ledger: Mapping[str, torch.Tensor] | None = None,
        te_leg_strength_override: torch.Tensor | None = None,
        front_extra_force: torch.Tensor | None = None,
    ) -> NativeV5MSurfaceLoadPacket:
        """Assemble the resolved point-load packet.

        ``solution_velocity_flat`` is the FULL solution velocity (freestream
        + bound + wake + particles) at the 4P filament midpoints in
        front/right/back/left order — exactly what the solver's
        ``refine_kj_with_solution_velocity`` path evaluates.

        ``front_extra_force`` (P, 3), if given, is added to the force of the front
        filament of each panel, where it acts: Hirato's leading-edge suction
        (plan V3, stage 2).  It is not a Kutta-Joukowski force and depends on no
        velocity, so it enters the packet directly.
        """
        if self.aerodynamic_dt is None:
            raise ValueError("resolved load assembly requires aerodynamic_dt")
        nc, ns = self.panel_topology(geometry)
        count = nc * ns
        if solution_velocity_flat.shape != (4 * count, 3):
            raise ValueError(
                f"solution velocity must have shape ({4 * count}, 3), "
                f"got {tuple(solution_velocity_flat.shape)}"
            )
        delta_grid, gamma_rate = self.delta_grid(gamma, mf2_history, nc, ns)
        leg_strengths = self.effective_leg_strengths(
            gamma.reshape(nc, ns), delta_grid
        )
        if te_leg_strength_override is not None:
            # B2 diagnostic only: the trailing-edge leg of the last chordwise
            # row carries (Gamma_n - Gamma_{n-1}) today, while the wake row
            # actually inserted carries gamma_tev = Gamma_TE + gamma_lev.  A
            # caller may pass the alternative strength for the SAME state to
            # quantify the difference; the production paths never set this.
            override = te_leg_strength_override.reshape(-1)
            if override.shape != (ns,):
                raise ValueError(
                    f"te_leg_strength_override must have shape ({ns},), "
                    f"got {tuple(override.shape)}"
                )
            front, right, back, left = leg_strengths
            back = back.clone()
            back[(nc - 1) * ns :] = override
            leg_strengths = (front, right, back, left)
        mids, movements, lengths = self.leg_layout(geometry)
        point_forces: list[torch.Tensor] = []
        total_moment = torch.zeros(3, device=self.device, dtype=torch.float64)
        for leg, (strength, movement, length, mid) in enumerate(
            zip(leg_strengths, movements, lengths, mids, strict=True)
        ):
            velocity = solution_velocity_flat[leg * count : (leg + 1) * count]
            force = (
                self.density
                * strength.reshape(-1, 1)
                * torch.linalg.cross(velocity + movement, length, dim=1)
            )
            if leg == 0 and front_extra_force is not None:
                if tuple(front_extra_force.shape) != (count, 3):
                    raise ValueError(
                        f"front_extra_force must have shape ({count}, 3), "
                        f"got {tuple(front_extra_force.shape)}"
                    )
                force = force + front_extra_force
            point_forces.append(force)
            total_moment += torch.sum(torch.linalg.cross(mid, force, dim=1), dim=0)
        unsteady = (
            (self.density * gamma_rate)[:, None]
            * geometry.areas[:, None]
            * geometry.normals
        )
        point_forces.append(unsteady)
        total_moment = total_moment + torch.sum(
            torch.linalg.cross(geometry.collocation, unsteady, dim=1), dim=0
        )
        forces = torch.cat(point_forces, dim=0)
        positions = torch.cat(
            mids + [geometry.collocation],
            dim=0,
        )
        kinds = torch.cat(
            [
                torch.full((count,), kind, device=self.device, dtype=torch.int64)
                for kind in (
                    POINT_KIND_FRONT,
                    POINT_KIND_RIGHT,
                    POINT_KIND_BACK,
                    POINT_KIND_LEFT,
                    POINT_KIND_UNSTEADY,
                )
            ]
        )
        owner = torch.arange(count, device=self.device, dtype=torch.int64).repeat(5)
        packet = NativeV5MSurfaceLoadPacket(
            point_positions_I=positions,
            point_forces_I=forces,
            point_kind=kinds,
            panel_owner=owner,
            total_force_I=torch.sum(forces, dim=0),
            total_moment_I=total_moment,
            component_ledger=dict(component_ledger or {}),
        )
        validate_packet(packet)
        return packet


__all__ = [
    "POINT_KIND_BACK",
    "POINT_KIND_FRONT",
    "POINT_KIND_LEFT",
    "POINT_KIND_RIGHT",
    "POINT_KIND_UNSTEADY",
    "NativeResolvedLoadAssembler",
    "NativeV5MSurfaceLoadPacket",
    "validate_packet",
]
