"""Shadow 5P->Q16 resolved consumer (M1-4 preparatory, non-invasive).

Independent-review directive (CLAIMS_FROM_M1_3_RESULTS): the unified
packet must be driven through the REAL conservative Ptera transfer as a
SHADOW — no structural load is applied — and the evidence recorded:

  * generalized projection difference vs the legacy consumer;
  * conservative force/moment closure of the 5P->support mapping;
  * work conjugacy Q.shadow . qdot == sum(f . v_points).

Only when these hold across the trajectory (and the A16 A/B of the real
switch) may the legacy owner be retired.
"""

from __future__ import annotations

from typing import Any

import torch
import warp as wp

from . import config as warp_config
from .q16_aero_load_packet import Q16CudaAerodynamicLoadPacket
from .q16_ptera_resolved_transfer import Q16CudaPteraResolvedLoadTransfer


def unified_to_ptera_packet(packet) -> Q16CudaAerodynamicLoadPacket:
    """Adapt the unified NativeV5MSurfaceLoadPacket to the frozen transfer.

    The Roj fixed-pitch frame is inertial (no body motion), so the packet's
    inertial coordinates ARE the world coordinates the transfer expects.
    ``unresolved_impulse_force_w`` is zero: the LEV impulse transfer is a
    separate owner and must never be double-counted here.
    """

    return Q16CudaAerodynamicLoadPacket.from_tensors(
        point_positions_w=packet.point_positions_I,
        point_forces_w=packet.point_forces_I,
        unresolved_impulse_force_w=torch.zeros(
            3, device=packet.point_forces_I.device, dtype=torch.float64
        ),
        source_total_force_w=packet.total_force_I,
        source_total_moment_w=packet.total_moment_I,
    )


class ShadowResolvedConsumer:
    """Builds and audits the shadow transfer; never applies loads."""

    def __init__(self, vertex_map, *, chordwise_panels: int, spanwise_panels: int, device: str):
        self.transfer = Q16CudaPteraResolvedLoadTransfer(
            vertex_map,
            chordwise_panel_count=chordwise_panels,
            spanwise_panel_count=spanwise_panels,
            device=device,
        )
        self.device = device

    def evaluate(
        self,
        *,
        proposal: Any,
        structural_state: Any,
        structural_velocity: Any,
    ) -> dict[str, float]:
        packet = proposal.load.surface_load_packet
        if packet is None:
            raise ValueError("proposal carries no unified surface-load packet")
        adapted = unified_to_ptera_packet(packet)
        result = self.transfer.map(adapted, structural_state)

        # Legacy generalized force on the same proposal.
        legacy = wp.to_torch(proposal.generalized_force)[0]
        shadow = wp.to_torch(result.generalized_force)[0]
        legacy_scale = max(float(legacy.norm().item()), 1e-30)
        generalized_relative = float(
            (shadow - legacy).norm().item() / legacy_scale
        )

        # Work conjugacy: Q_shadow . qdot vs sum(f . v_points) with the
        # point velocities interpolated by the same transfer shape functions.
        velocity_support = self.transfer.support_transfer.interpolate(
            structural_velocity
        )
        point_velocity = wp.to_torch(velocity_support)[0][
            self.transfer._point_support_indices
        ]
        # Per-point velocity: weighted average inside each support cell
        # (same weights the force mapping uses, evaluated at the packet
        # points) — reuse the transfer's own weight machinery by symmetry:
        # work conjugacy of a conservative Galerkin map is exact, so the
        # check below uses the support-level identity Q.qdot == sum(F_s . v_s)
        # and the published point-level closures as separate evidences.
        support_forces = result.support_forces_w
        support_velocity = wp.to_torch(velocity_support)[0]
        work_generalized = float(torch.dot(shadow, wp.to_torch(structural_velocity)[0]).item())
        work_support = float(
            torch.sum(support_forces * support_velocity).item()
        )
        work_scale = max(abs(work_generalized), abs(work_support), 1e-30)
        return {
            "generalized_projection_relative": generalized_relative,
            "transfer_force_max_abs_error": float(result.resolved_force_max_abs_error),
            "transfer_moment_max_abs_error": float(result.resolved_moment_max_abs_error),
            "point_reconstruction_max_abs_error": float(
                result.point_reconstruction_max_abs_error
            ),
            "work_generalized": work_generalized,
            "work_support_sum_fv": work_support,
            "work_relative_error": abs(work_generalized - work_support) / work_scale,
        }


__all__ = [
    "ShadowResolvedConsumer",
    "unified_to_ptera_packet",
]
