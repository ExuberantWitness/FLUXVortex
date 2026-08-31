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
        structural_acceleration: Any | None = None,
        committed_aerodynamic_load: Any = None,
        time_star: float = float("nan"),
    ) -> dict[str, float]:
        """Audit-only evaluation; returns the M1-4 evidence record.

        Contract (CLAIMS_FROM_M1_4_SHADOW_RESULTS directive 1-3):
          * hard failure on any contract violation (caller escalates);
          * the projection comparison is labelled for what it is -- the
            FULL instantaneous 5P KJ+unsteady mapping vs the legacy
            CONSTANT-only generalized force -- and the production
            decomposition ledger (constant / velocity / Mf1) is recorded
            alongside so no decomposition mismatch is read as load error;
          * point-level work conjugacy uses the transfer's own weights:
            Q.qdot == sum_i f_i . v_i with v_i reconstructed pointwise.
        """

        packet = proposal.load.surface_load_packet
        if packet is None:
            raise ValueError("proposal carries no unified surface-load packet")
        adapted = unified_to_ptera_packet(packet)
        result = self.transfer.map(adapted, structural_state)

        shadow = wp.to_torch(result.generalized_force)[0]
        qdot = wp.to_torch(structural_velocity)[0]

        # -- decomposition ledger (legacy production consumer) --------------
        ledger: dict[str, float] = {}
        if committed_aerodynamic_load is not None:
            constant = wp.to_torch(
                committed_aerodynamic_load.constant_generalized_force
            )[0]
            velocity_q = wp.to_torch(
                committed_aerodynamic_load.velocity_force(structural_velocity)
            )[0]
            ledger["legacy_constant_norm"] = float(constant.norm().item())
            ledger["legacy_velocity_norm"] = float(velocity_q.norm().item())
            am = committed_aerodynamic_load.added_mass.generalized_matrix
            ledger["legacy_mf1_frobenius_norm"] = float(
                torch.linalg.matrix_norm(am).item()
            )
            if structural_acceleration is not None:
                qddot = wp.to_torch(structural_acceleration)[0]
                ledger["legacy_mf1_action_norm"] = float(
                    (am @ qddot).norm().item()
                )
            legacy_reference = constant
        else:
            legacy_reference = wp.to_torch(proposal.generalized_force)[0]
        reference_scale = max(float(legacy_reference.norm().item()), 1e-30)

        # -- projection difference (decomposition-labelled!) ----------------
        fiveP_vs_legacy_constant_only = float(
            (shadow - legacy_reference).norm().item() / reference_scale
        )

        # -- point-level work conjugacy via the transfer's own weights ------
        # The force scatter is F_s = sum_i w_{i,slot(i,s)} f_i (CSR gather
        # per support).  Its exact transpose gathers each point's velocity
        # over the same CSR entries: v_i = sum_{s in csr(i)} w_{i,slot} v_s.
        support_positions = wp.to_torch(
            self.transfer.support_transfer.interpolate(structural_state)
        )[0]
        weights, reconstruction_error = self.transfer._weights(
            packet.point_positions_I, support_positions
        )
        support_velocity = wp.to_torch(
            self.transfer.support_transfer.interpolate(structural_velocity)
        )[0]
        offsets = self.transfer._support_load_offsets.numpy()
        indices = self.transfer._support_load_indices.numpy()
        slots = self.transfer._support_load_slots.numpy()
        weights_cpu = weights.detach().cpu()
        support_velocity_cpu = support_velocity.detach().cpu()
        point_velocity_cpu = torch.zeros(
            (weights_cpu.shape[0], 3), dtype=torch.float64
        )
        for support in range(offsets.shape[0] - 1):
            for cursor in range(offsets[support], offsets[support + 1]):
                load = int(indices[cursor])
                slot = int(slots[cursor])
                point_velocity_cpu[load] += (
                    weights_cpu[load, slot] * support_velocity_cpu[support]
                )
        point_velocity = point_velocity_cpu.to(support_velocity.device)
        work_points = float(
            torch.sum(packet.point_forces_I * point_velocity).item()
        )
        work_generalized = float(torch.dot(shadow, qdot).item())
        work_scale = max(abs(work_points), abs(work_generalized), 1e-30)
        work_relative = abs(work_points - work_generalized) / work_scale

        evidence: dict[str, float] = {
            "time_star": float(time_star),
            "fiveP_instantaneous_vs_legacy_constant_only_relative": (
                fiveP_vs_legacy_constant_only
            ),
            "decomposition_note": (
                "comparison is FULL 5P instantaneous vs legacy CONSTANT only; "
                "production also applies velocity_force and Mf1 per substep "
                "(see ledger norms)"
            ),
            "transfer_force_max_abs_error": float(result.resolved_force_max_abs_error),
            "transfer_moment_max_abs_error": float(result.resolved_moment_max_abs_error),
            "point_reconstruction_max_abs_error": float(reconstruction_error),
            "work_generalized_qdot": work_generalized,
            "work_points_sum_fv": work_points,
            "work_pointwise_relative_error": work_relative,
        }
        evidence.update(ledger)
        return evidence


__all__ = [
    "ShadowResolvedConsumer",
    "unified_to_ptera_packet",
]
