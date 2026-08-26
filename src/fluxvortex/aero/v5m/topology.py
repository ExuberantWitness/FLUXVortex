"""MultiSurfaceTopology: global panel/strip/edge offsets for multi-surface V5M.

Refactor plan §8.6 / §14 U4: left and right wings share one body; each
contributes panels to a single global AIC system so mutual induction between
the wings is part of the same solve (never two independent single-wing solves
summed afterwards).  The offsets maintained here are the mapping from
(surface_id, local_panel) to global_panel and back, plus the strip and
leading-edge node offsets the per-surface LEV/TEV/wake owners need to place
their state into global wake/particle arrays.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class SurfaceOffsets:
    surface_id: str
    panel_offset: int      # first global panel index
    panel_count: int
    strip_offset: int      # first global LE strip index
    strip_count: int
    le_node_offset: int    # first global leading-edge node index
    le_node_count: int     # strip_count + 1


@dataclass(frozen=True)
class MultiSurfaceTopology:
    surfaces: tuple[SurfaceOffsets, ...]

    @property
    def total_panels(self) -> int:
        return sum(s.panel_count for s in self.surfaces)

    @property
    def total_strips(self) -> int:
        return sum(s.strip_count for s in self.surfaces)

    @property
    def total_le_nodes(self) -> int:
        return sum(s.le_node_count for s in self.surfaces)

    def surface_of_panel(self, global_panel: int) -> str:
        for s in self.surfaces:
            if s.panel_offset <= global_panel < s.panel_offset + s.panel_count:
                return s.surface_id
        raise IndexError(f"panel {global_panel} outside all surfaces")

    def local_panel_index(self, global_panel: int) -> int:
        for s in self.surfaces:
            if s.panel_offset <= global_panel < s.panel_offset + s.panel_count:
                return global_panel - s.panel_offset
        raise IndexError(f"panel {global_panel} outside all surfaces")

    def global_panel_index(self, surface_id: str, local_panel: int) -> int:
        for s in self.surfaces:
            if s.surface_id == surface_id:
                if 0 <= local_panel < s.panel_count:
                    return s.panel_offset + local_panel
                raise IndexError(f"local panel {local_panel} outside {surface_id}")
        raise KeyError(f"unknown surface {surface_id}")

    def offsets_of(self, surface_id: str) -> SurfaceOffsets:
        for s in self.surfaces:
            if s.surface_id == surface_id:
                return s
        raise KeyError(f"unknown surface {surface_id}")

    @classmethod
    def from_surface_frames(cls, frames: tuple[Any, ...]) -> "MultiSurfaceTopology":
        """Build topology from a tuple of SurfaceFrames."""
        offsets = []
        panel_cursor = 0
        strip_cursor = 0
        le_cursor = 0
        for frame in frames:
            n_panels = frame.chordwise_panels * frame.spanwise_panels
            n_strips = frame.spanwise_panels
            n_le_nodes = frame.spanwise_panels + 1
            offsets.append(SurfaceOffsets(
                surface_id=frame.surface_id,
                panel_offset=panel_cursor,
                panel_count=n_panels,
                strip_offset=strip_cursor,
                strip_count=n_strips,
                le_node_offset=le_cursor,
                le_node_count=n_le_nodes,
            ))
            panel_cursor += n_panels
            strip_cursor += n_strips
            le_cursor += n_le_nodes
        return cls(surfaces=tuple(offsets))


def concatenate_surface_frames(frames: tuple[Any, ...]) -> tuple[torch.Tensor, ...]:
    """Concatenate multiple SurfaceFrames' panel data into global arrays.

    Every field below already carries a leading panel (or leading-edge node)
    dimension, so a plain ``torch.cat(..., dim=0)`` stacks the per-surface
    blocks in topology order.  A single-surface tuple returns the frame's own
    tensors bit-identically, so U4's global assembly cannot perturb U3's
    single-surface numerics.

    Returns (rings, ring_velocities, collocation, collocation_velocities,
             normals, areas, leading_edge, trailing_edge,
             leading_velocity, trailing_velocity).
    """
    if not frames:
        raise ValueError("concatenate_surface_frames requires at least one frame")
    return tuple(
        torch.cat([getattr(f, name) for f in frames], dim=0)
        for name in (
            "panel_rings_I", "panel_ring_velocity_I",
            "collocation_I", "collocation_velocity_I",
            "normals_I", "areas", "leading_edge_I", "trailing_edge_I",
            "leading_velocity_I", "trailing_velocity_I",
        )
    )
