"""Multi-surface V5M AIC assembly with cross-surface mutual induction.

Refactor plan §8.6 / U7-3: left and right wings contribute their panels to
ONE global aerodynamic influence matrix so mutual induction between the
wings is part of the same bound-circulation solve (never two independent
single-wing solves summed afterwards).  This module builds that global
matrix from per-surface SurfaceFrames with the production bound-vortex
kernel (``native_ring_velocity_expanded`` at the bound-solve core),
block by block:

    A_global = [[A(R,R), A(R,L)],    row surface supplies the collocation
                [A(L,R), A(L,L)]]    points/normals, column surface the rings

Every block uses the COLUMN surface's chordwise panel count as the kernel
reference length (the reference length floors the SOURCE ring's edge
scale, so it belongs to the ring side).  With that choice the diagonal
blocks are bit-identical to each surface's own ``native_aic``, so global
assembly cannot perturb single-surface numerics, while the off-diagonal
blocks are exactly the cross-wing influence the single-surface solve
throws away.

U7-3 scope: the global matrix is assembled, orientation-gated, and
quantified (``cross_influence_norm`` / ``MutualInductionReport``) as the
diagnostic that justifies the coupled solve.  ``MultiSurfaceV5MSolver``
wraps the per-surface solvers and exposes ``propose_independent`` which
keeps each surface's physics (LEV, wake, loads) running independently
while returning the global matrix alongside, ready for U7-4 to swap it
into the bound solve with per-surface LEV/TEV/wake ownership.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import torch

from .topology import MultiSurfaceTopology

# Bound-solve core fraction and influence-tile budget, identical to the
# production single-surface path (native_aic's core_fraction and
# Q16NativeV5MSolver._ring_velocity's _RING_TILE_BYTES).
BOUND_CORE_FRACTION = 1.0e-6
_RING_TILE_BYTES = 64 * 1024 * 1024


def _ring_velocity_expanded() -> Any:
    """Lazy import of the production kernel (keeps this module warp-free)."""
    from fluxvortex.warp_fsi.q16_flux_v5m_native import native_ring_velocity_expanded

    return native_ring_velocity_expanded


def _validate_frames(frames: Iterable[Any]) -> None:
    frames = tuple(frames)
    if not frames:
        raise ValueError("multi-surface AIC assembly requires at least one frame")
    device = frames[0].collocation_I.device
    for frame in frames:
        panels = frame.chordwise_panels * frame.spanwise_panels
        if frame.panel_rings_I.shape != (panels, 4, 3):
            raise ValueError(
                f"frame {frame.surface_id}: rings {tuple(frame.panel_rings_I.shape)} "
                f"do not match its declared {frame.chordwise_panels}x{frame.spanwise_panels} panels"
            )
        for name, tensor in (
            ("collocation_I", frame.collocation_I),
            ("normals_I", frame.normals_I),
        ):
            if tensor.shape != (panels, 3):
                raise ValueError(
                    f"frame {frame.surface_id}: {name} {tuple(tensor.shape)} "
                    f"do not match {panels} panels"
                )
        for tensor in (frame.panel_rings_I, frame.collocation_I, frame.normals_I):
            if tensor.device != device:
                raise ValueError("all frames must live on the same device")
            if tensor.dtype is not torch.float64:
                raise TypeError("multi-surface AIC assembly requires float64 frames")


def _influence_block(
    points: torch.Tensor,
    rings: torch.Tensor,
    normals: torch.Tensor,
    *,
    core_fraction: float,
    reference_length: float,
) -> torch.Tensor:
    """Normal velocity influence of ``rings`` at ``points`` (one AIC block).

    Same kernel and same tiling policy as the production
    Q16NativeV5MSolver._ring_velocity: evaluations below the tile budget
    take the historical single-shot path (bit-identical to native_aic for
    the diagonal blocks); larger ones tile over the collocation points so
    wide multi-surface assemblies cannot materialize multi-GB
    intermediates.
    """
    kernel = _ring_velocity_expanded()

    def dot_block(point_slice: torch.Tensor, normal_slice: torch.Tensor) -> torch.Tensor:
        expanded = kernel(
            point_slice,
            rings,
            core_fraction=core_fraction,
            reference_length=reference_length,
        )
        return torch.sum(expanded * normal_slice[:, None, :], dim=2)

    full_bytes = points.shape[0] * rings.shape[0] * 3 * 8
    if full_bytes <= _RING_TILE_BYTES:
        return dot_block(points, normals)
    result = torch.empty(
        (points.shape[0], rings.shape[0]), device=points.device, dtype=points.dtype
    )
    target_cap = max(1, _RING_TILE_BYTES // (rings.shape[0] * 3 * 8))
    for start in range(0, points.shape[0], target_cap):
        stop = min(start + target_cap, points.shape[0])
        result[start:stop] = dot_block(points[start:stop], normals[start:stop])
    return result


def build_global_aic(
    frames: Iterable[Any],
    *,
    topology: MultiSurfaceTopology | None = None,
    core_fraction: float = BOUND_CORE_FRACTION,
) -> torch.Tensor:
    """Assemble the (total_panels, total_panels) global bound-vortex AIC.

    Row blocks come from each surface's collocation points and stored
    normals; column blocks from each surface's rings, with the column
    surface's own 1/chordwise_panels reference length.  The diagonal
    blocks therefore reproduce the single-surface ``native_aic`` of each
    surface exactly, and the off-diagonal blocks are the cross-surface
    mutual induction.

    The same orientation gate as ``native_aic`` is enforced on the global
    diagonal: a panel's self-influence must stay negative.
    """
    frames = tuple(frames)
    _validate_frames(frames)
    topo = (
        topology
        if topology is not None
        else MultiSurfaceTopology.from_surface_frames(frames)
    )
    if len(topo.surfaces) != len(frames):
        raise ValueError(
            f"topology has {len(topo.surfaces)} surfaces for {len(frames)} frames"
        )
    for surface, frame in zip(topo.surfaces, frames):
        if surface.surface_id != frame.surface_id:
            raise ValueError(
                f"topology surface {surface.surface_id} does not match "
                f"frame {frame.surface_id}"
            )
        if surface.panel_count != frame.chordwise_panels * frame.spanwise_panels:
            raise ValueError(
                f"topology gives {surface.surface_id} {surface.panel_count} panels "
                f"but its frame declares "
                f"{frame.chordwise_panels * frame.spanwise_panels}"
            )
    first = frames[0].collocation_I
    aic = torch.zeros(
        (topo.total_panels, topo.total_panels), device=first.device, dtype=first.dtype
    )
    for i, row in enumerate(topo.surfaces):
        points = frames[i].collocation_I
        normals = frames[i].normals_I
        for j, col in enumerate(topo.surfaces):
            rings = frames[j].panel_rings_I
            reference_length = 1.0 / frames[j].chordwise_panels
            aic[
                row.panel_offset : row.panel_offset + row.panel_count,
                col.panel_offset : col.panel_offset + col.panel_count,
            ] = _influence_block(
                points,
                rings,
                normals,
                core_fraction=core_fraction,
                reference_length=reference_length,
            )
    if bool(torch.any(torch.diagonal(aic) >= 0.0).item()):
        raise RuntimeError("global AIC orientation drift")
    return aic


def build_cross_aic(
    frames: Iterable[Any],
    *,
    topology: MultiSurfaceTopology | None = None,
    core_fraction: float = BOUND_CORE_FRACTION,
) -> torch.Tensor:
    """U7-3 task-facing name for :func:`build_global_aic`.

    Builds the full global influence matrix — the (i, j) block is the
    velocity induced at surface i's collocation points by surface j's
    rings, dotted with surface i's normals — so the off-diagonal blocks
    ARE the cross-surface influence.
    """
    return build_global_aic(frames, topology=topology, core_fraction=core_fraction)


def aic_block(
    aic: torch.Tensor, topology: MultiSurfaceTopology, row_index: int, col_index: int
) -> torch.Tensor:
    """Slice the (row_index, col_index) surface block out of a global AIC."""
    row = topology.surfaces[row_index]
    col = topology.surfaces[col_index]
    return aic[
        row.panel_offset : row.panel_offset + row.panel_count,
        col.panel_offset : col.panel_offset + col.panel_count,
    ]


@dataclass(frozen=True)
class MutualInductionReport:
    """Mutual-induction quantification of one global AIC (U7-3 diagnostic)."""

    total_panels: int
    self_frobenius: float       # Frobenius norm over the diagonal blocks
    cross_frobenius: float      # Frobenius norm over the off-diagonal blocks
    ratio: float                # cross_frobenius / self_frobenius
    self_max_abs: float
    cross_max_abs: float
    cross_nonzero_fraction: float   # fraction of off-diagonal entries != 0


def mutual_induction_report(
    aic: torch.Tensor, topology: MultiSurfaceTopology
) -> MutualInductionReport:
    """Quantify |A_cross|_F / |A_self|_F block by block.

    The ratio is the U7-3 headline number: how strongly the OTHER wing's
    bound vorticity drives this wing's collocation points relative to its
    own.  Zero means the wings do not interact (independent solves would
    be exact); order one means the wings dominate each other.
    """
    self_sq = 0.0
    cross_sq = 0.0
    self_max = 0.0
    cross_max = 0.0
    cross_nonzero = 0
    cross_entries = 0
    for i, row in enumerate(topology.surfaces):
        for j, col in enumerate(topology.surfaces):
            block = aic[
                row.panel_offset : row.panel_offset + row.panel_count,
                col.panel_offset : col.panel_offset + col.panel_count,
            ]
            fro = float(torch.linalg.vector_norm(block).item())
            max_abs = float(torch.max(torch.abs(block)).item())
            if i == j:
                self_sq += fro * fro
                self_max = max(self_max, max_abs)
            else:
                cross_sq += fro * fro
                cross_max = max(cross_max, max_abs)
                cross_nonzero += int(torch.count_nonzero(block).item())
                cross_entries += block.numel()
    return MutualInductionReport(
        total_panels=topology.total_panels,
        self_frobenius=self_sq**0.5,
        cross_frobenius=cross_sq**0.5,
        ratio=(cross_sq / self_sq) ** 0.5 if self_sq > 0.0 else float("inf"),
        self_max_abs=self_max,
        cross_max_abs=cross_max,
        cross_nonzero_fraction=(
            cross_nonzero / cross_entries if cross_entries else 0.0
        ),
    )


def cross_influence_norm(
    frames: Iterable[Any],
    *,
    topology: MultiSurfaceTopology | None = None,
    core_fraction: float = BOUND_CORE_FRACTION,
) -> float:
    """|A_cross|_F / |A_self|_F for the global AIC of these frames."""
    frames = tuple(frames)
    topo = (
        topology
        if topology is not None
        else MultiSurfaceTopology.from_surface_frames(frames)
    )
    aic = build_global_aic(frames, topology=topo, core_fraction=core_fraction)
    return mutual_induction_report(aic, topo).ratio


class MultiSurfaceV5MSolver:
    """Joint V5M solver shell for multiple surfaces with mutual induction.

    Wraps N single-surface ``Q16NativeV5MSolver`` instances plus the
    global topology.  U7-3 provides the global AIC (with the
    cross-surface blocks) as a first-class artifact;
    ``propose_independent`` keeps each surface's physics (LEV, TEV,
    wake, loads) running on its own solver while returning the global
    matrix alongside the proposals so U7-4 can swap it into the bound
    solve.  The full cross-coupled solve (joint gamma over all panels,
    per-surface LEV/wake ownership, per-surface trailing edges) is U7-4.
    """

    def __init__(
        self,
        single_surface_solvers: Iterable[Any],
        topology: MultiSurfaceTopology,
    ) -> None:
        solvers = list(single_surface_solvers)
        if not solvers:
            raise ValueError("MultiSurfaceV5MSolver requires at least one solver")
        if len(solvers) != len(topology.surfaces):
            raise ValueError(
                f"{len(solvers)} solvers for a {len(topology.surfaces)}-surface topology"
            )
        for solver, surface in zip(solvers, topology.surfaces):
            settings = solver.settings
            panels = settings.chordwise_panels * settings.spanwise_panels
            if panels != surface.panel_count:
                raise ValueError(
                    f"solver for {surface.surface_id} has {panels} panels "
                    f"but the topology declares {surface.panel_count}"
                )
        devices = {torch.device(solver.settings.device) for solver in solvers}
        if len(devices) != 1:
            raise ValueError("all single-surface solvers must share one device")
        self._solvers = solvers
        self._topology = topology

    @property
    def solvers(self) -> list[Any]:
        return list(self._solvers)

    @property
    def topology(self) -> MultiSurfaceTopology:
        return self._topology

    @property
    def total_panels(self) -> int:
        return self._topology.total_panels

    def build_global_aic(
        self,
        frames: Iterable[Any],
        *,
        core_fraction: float = BOUND_CORE_FRACTION,
    ) -> torch.Tensor:
        """Global (total_panels, total_panels) AIC for these frames.

        Uses the solver's own topology (validated against the frames), so
        the block layout is exactly the offsets the coupled U7-4 solve
        will index into.
        """
        return build_global_aic(
            frames, topology=self._topology, core_fraction=core_fraction
        )

    def mutual_induction(self, frames: Iterable[Any]) -> MutualInductionReport:
        """Quantified cross-surface influence for these frames."""
        aic = self.build_global_aic(frames)
        return mutual_induction_report(aic, self._topology)

    def propose_independent(
        self,
        committed_states: Iterable[Any],
        structural_states: Iterable[Any],
        structural_velocities: Iterable[Any],
        frames: Iterable[Any],
    ) -> tuple[list[Any], torch.Tensor]:
        """One independent propose per surface plus the global AIC.

        The per-surface physics still runs on each surface's own solver
        and its own single-surface AIC (the cross-influence is NOT yet in
        the bound solve — that replacement is the U7-4 work).  The global
        matrix is built from the same frames and returned alongside so
        callers can quantify, checkpoint, or (in U7-4) swap in the
        cross-coupled system.
        """
        states = list(committed_states)
        structural = list(structural_states)
        velocities = list(structural_velocities)
        if not (len(states) == len(structural) == len(velocities) == len(self._solvers)):
            raise ValueError(
                "propose_independent needs one committed state, structural "
                "state, and structural velocity per surface"
            )
        proposals = [
            solver.propose(state, structural_state, velocity)
            for solver, state, structural_state, velocity in zip(
                self._solvers, states, structural, velocities
            )
        ]
        global_aic = self.build_global_aic(tuple(frames))
        return proposals, global_aic


__all__ = [
    "BOUND_CORE_FRACTION",
    "MultiSurfaceV5MSolver",
    "MutualInductionReport",
    "aic_block",
    "build_cross_aic",
    "build_global_aic",
    "cross_influence_norm",
    "mutual_induction_report",
]
