"""SurfaceFrame: the unified boundary between kinematics and aerodynamics."""
from __future__ import annotations
from dataclasses import dataclass
import torch

@dataclass(frozen=True)
class SurfaceFrame:
    """All geometry + velocity the aero solver sees, in frame I (inertial).

    Fields per the refactor plan §5.5. Every tensor is CUDA float64.
    """
    surface_id: str
    body_id: str
    panel_rings_I: torch.Tensor        # (n_panels, 4, 3)
    panel_ring_velocity_I: torch.Tensor
    collocation_I: torch.Tensor        # (n_panels, 3)
    collocation_velocity_I: torch.Tensor
    normals_I: torch.Tensor            # (n_panels, 3)
    areas: torch.Tensor                # (n_panels,)
    leading_edge_I: torch.Tensor       # (n_le_nodes, 3)
    trailing_edge_I: torch.Tensor
    leading_velocity_I: torch.Tensor
    trailing_velocity_I: torch.Tensor
    chordwise_panels: int
    spanwise_panels: int
    topology_digest: str

    @classmethod
    def from_native_geometry(cls, geometry, surface_id: str, body_id: str) -> "SurfaceFrame":
        """Zero-copy wrap of the existing NativeV5MGeometry (U0-P).

        The native rings are ordered (nc * ns, 4, 3) and reshape exactly to
        (nc, ns, 4, 3); leading_edge carries the ns + 1 spanwise LE nodes
        (q16_flux_v5m_native.py, evaluate()).
        """
        spanwise_panels = int(geometry.leading_edge.shape[0] - 1)
        chordwise_panels = int(geometry.rings.shape[0] // spanwise_panels)
        return cls(
            surface_id=surface_id,
            body_id=body_id,
            panel_rings_I=geometry.rings,
            panel_ring_velocity_I=geometry.ring_velocity,
            collocation_I=geometry.collocation,
            collocation_velocity_I=geometry.collocation_velocity,
            normals_I=geometry.normals,
            areas=geometry.areas,
            leading_edge_I=geometry.leading_edge,
            trailing_edge_I=geometry.trailing_edge,
            leading_velocity_I=geometry.leading_velocity,
            trailing_velocity_I=geometry.trailing_velocity,
            chordwise_panels=chordwise_panels,
            spanwise_panels=spanwise_panels,
            topology_digest=f"native-v5m:{geometry.rings.shape[0]}",
        )
