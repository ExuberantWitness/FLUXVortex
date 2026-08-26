"""PartitionedStrongFSI: wraps the existing Q16NativeV5MFSIStepper.

U1-P: the wrapper delegates advance() to the production stepper; the
transaction semantics (parent/trial/formal/commit) are inherited from the
existing atomic-commit design. U1-F adds explicit counting and residual
exposure.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TransactionCounters:
    aero_proposal_count: int = 0
    dynamic_proposal_count: int = 0
    discarded_trial_count: int = 0
    formal_replay_count: int = 0
    commit_count: int = 0


class PartitionedStrongFSI:
    """Unified strong-FSI coupling — thin wrapper over the production stepper."""

    def __init__(self, production_stepper) -> None:
        self._stepper = production_stepper
        self.counters = TransactionCounters()

    def advance(
        self,
        owner,
        *,
        delta_time,
        prescribed_forces,
        load_betas=None,
        **kwargs,
    ):
        result = self._stepper.advance(
            owner,
            delta_time=delta_time,
            prescribed_forces=prescribed_forces,
            load_betas=load_betas,
            **kwargs,
        )
        # U1-F: true-semantics counting
        self.counters.aero_proposal_count += result.aerodynamic_evaluations
        self.counters.formal_replay_count += 1  # exactly one per accepted step
        self.counters.commit_count += 1
        # discarded = proposals that were explored but not committed
        self.counters.discarded_trial_count += result.aerodynamic_evaluations - 1
        return result
