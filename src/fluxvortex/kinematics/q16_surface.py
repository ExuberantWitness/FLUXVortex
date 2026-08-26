"""Q16SurfaceFrameAdapter: produces SurfaceFrame from Q16 state via the existing surface."""
from __future__ import annotations
import warp as wp
from .frames import SurfaceFrame

class Q16SurfaceFrameAdapter:
    """Wraps Q16NativeV5MSurface.evaluate() into the unified SurfaceFrame.

    This is the U0-P thin adapter: it calls the existing production evaluate()
    and zero-copy wraps the result.  No formulas are duplicated.
    """
    def __init__(self, native_surface, *, surface_id: str = "wing_0", body_id: str = "body_0") -> None:
        self._native = native_surface
        self.surface_id = surface_id
        self.body_id = body_id
        self.chordwise_panels = native_surface.nc
        self.spanwise_panels = native_surface.ns

    @property
    def surface_ids(self) -> tuple[str, ...]:
        return (self.surface_id,)

    def evaluate(self, structural_state: wp.array, structural_velocity: wp.array) -> SurfaceFrame:
        geometry = self._native.evaluate(structural_state, structural_velocity)
        return SurfaceFrame.from_native_geometry(
            geometry, surface_id=self.surface_id, body_id=self.body_id
        )
