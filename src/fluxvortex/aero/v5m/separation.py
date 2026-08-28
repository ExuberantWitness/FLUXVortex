"""Single-owner separation mask (refactor plan §8.3).

The 3D surface LESP is the only truth for whether a strip is separated.
The source bank closure receives this mask and only computes release
strength/position — it does NOT independently declare separation.
"""
from __future__ import annotations
import torch

def unified_separation_mask(
    lesp_pre_3d: torch.Tensor,
    lesp_crit: float,
) -> torch.Tensor:
    """Compute the single-owner separation mask from 3D LESP.

    Returns (n_strips,) bool: True where |LESP| > crit.
    This is the ONLY place separation is decided.
    """
    return torch.abs(lesp_pre_3d) > lesp_crit

def reconcile_release_mask(
    surface_separated: torch.Tensor,
    source_shed_lev: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reconcile the source bank's release mask with the 3D truth.

    Production passes the 3D mask INTO ``CudaLDVMSourceBank.step`` (keyword
    ``cell_release_mask``) as a gate on the bank's own 2D LESP trigger, so the
    returned ``shed_lev`` can never fire on a strip the 3D solve says is
    attached.  This function then verifies that single ownership held.

    Returns (active_mask, conflict_count):
    - active_mask: the 3D truth — every separated strip stays pinned (it
      carries existing LEV circulation), whether or not the bank sheds.
    - conflict_count: strips that RELEASED without 3D sanction — the only
      dual-ownership direction still possible, and it must be zero: the
      3D LESP is the only separation truth.  A separated strip the bank
      has not (yet) shed is NOT a conflict: it stays pinned at its free
      A0 with no new particles deposited.  NOTE (audit R3): this pinned
      state is only a REAL "continuing release" when the strip actually
      carries LEV circulation; when no LE circulation exists (and none
      was ever shed) the strip is separated-but-never-shed, which the
      caller's release-flow diagnostics must expose rather than label
      "continuing release".
    """
    # The correct mask is the 3D truth, period. The source bank's mask
    # tells us which strips the 2D closure is actively shedding for.
    # A strip that released while 3D says attached = dual ownership: the
    # 2D closure independently declared separation.  With the production
    # gate wired into the bank this cannot happen; a nonzero count means
    # the gate was bypassed.
    # A strip that 3D says separated but the bank doesn't shed =
    # continuing release (already have aLEV, no new shedding needed).
    conflict = ~surface_separated & source_shed_lev  # released without the owner's sanction
    active = surface_separated & source_shed_lev      # both agree: shed

    # The production pin_active should be surface_separated (the 3D truth),
    # NOT the union with source releases. Strips where 3D says separated
    # but source hasn't caught up yet still need to be pinned (they carry
    # existing LEV circulation).
    return surface_separated, int(conflict.sum().item())
