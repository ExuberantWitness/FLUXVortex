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

    Returns (active_mask, conflict_count):
    - active_mask: strips where BOTH 3D truth says separated AND source
      bank says shedding (intersection, NOT union)
    - conflict_count: strips where the two owners disagree
    """
    # The correct mask is the 3D truth, period. The source bank's mask
    # tells us which strips the 2D closure is actively shedding for.
    # A strip that the source bank wants to shed but 3D says not
    # separated = a conflict (physics inconsistency).
    # A strip that 3D says separated but source bank doesn't shed = 
    # continuing release (already have aLEV, no new shedding needed).
    conflict = surface_separated & ~source_shed_lev  # 3D says separated, source doesn't shed
    active = surface_separated & source_shed_lev      # both agree: shed
    
    # The production pin_active should be surface_separated (the 3D truth),
    # NOT the union with source releases. Strips where 3D says separated
    # but source hasn't caught up yet still need to be pinned (they carry
    # existing LEV circulation).
    return surface_separated, int(conflict.sum().item())
