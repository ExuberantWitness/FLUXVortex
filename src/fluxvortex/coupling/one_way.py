"""One-way prescribed coupling: surface frame → one aero proposal → gates → commit.

For rigid cases where the structure is prescribed (no FSI iteration needed).
Shares the same AeroProposal/transaction as the FSI path (plan §10.5).
"""
from __future__ import annotations


class OneWayPrescribedCoupling:
    """prescribed state → surface frame → aero propose → commit (no iteration)."""

    def __init__(self, aero_stepper) -> None:
        self._aero = aero_stepper

    def advance(self, owner, *, surface_frames, delta_time):
        proposal = self._aero.propose(owner.aero_state, surface_frames, delta_time)
        self._aero.commit(owner, proposal)
        owner.generation += 1
        return proposal
