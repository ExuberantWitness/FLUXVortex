"""Retention policies for particles and wake — conservation-enforced.

Replaces raw array slicing (forbidden by the plan §8.5). Every policy must
report, in a ``RetentionResult``, exactly what circulation/impulse it is
taking out of the world, so the ``CirculationImpulseLedger`` (plan §5.9) can
carry the accounting into the step's Kelvin residual instead of letting a
cap silently destroy circulation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class RetentionResult:
    removed_particle_count: int
    removed_circulation_sum: float
    removed_linear_impulse: Any  # (3,) tensor or None
    removed_angular_impulse: Any  # (3,) tensor or None
    conservation_error: float


class RetentionPolicy:
    """Base: no culling (short-window counterfactual/verification)."""

    name = "NoCulling"

    def apply_particles(self, particle_field, step: int) -> RetentionResult | None:
        return None

    def apply_wake(self, wake_rings, wake_gamma, max_rows: int) -> RetentionResult | None:
        return None


class NoCulling(RetentionPolicy):
    """Explicit no-op policy (plan §8.5): everything is retained.

    Used for short-window counterfactual runs and parity verification where
    retention must be provably absent.
    """


class AgeCapPolicy(RetentionPolicy):
    """Particle age cap with circulation accounting.

    Mirrors the production mask ``birth_step < step - max_age`` exactly
    (q16_flux_v5m_native.py), but audits what is removed BEFORE removing it.
    """

    name = "AgeCap"

    def __init__(self, max_age_steps: int) -> None:
        self.max_age = max_age_steps

    def apply_particles(self, particle_field, step: int) -> RetentionResult | None:
        n = particle_field.n
        if n == 0:
            return None
        horizon = step - self.max_age
        if horizon <= 0:
            return None
        birth = particle_field.birth_step[:n]
        mask = birth < horizon  # True = remove
        count = int(mask.sum().item())
        if count == 0:
            return None
        # Audit what we're removing BEFORE removing it
        removed_gamma = particle_field.gamma[:n][mask]  # (count, 3)
        removed_circ_sum = float(removed_gamma.norm(dim=1).sum().item())
        # Linear impulse of removed particles: sum(gamma x pos)
        removed_pos = particle_field.pos[:n][mask]
        removed_lin = torch.linalg.cross(removed_gamma, removed_pos).sum(dim=0)
        # Actually remove
        particle_field.remove_mask(mask)
        return RetentionResult(
            removed_particle_count=count,
            removed_circulation_sum=removed_circ_sum,
            removed_linear_impulse=removed_lin,
            removed_angular_impulse=None,  # TODO: angular about a reference point
            conservation_error=0.0,  # set by caller after full ledger check
        )


class WakeRowCapPolicy(RetentionPolicy):
    """Wake row cap with circulation accounting.

    Production order is newest-first, so the oldest rows sit at the tail of
    the arrays and the cap keeps ``[:max_rows * spanwise_panels]`` — the same
    slice the native solver performs, but the removed circulation is reported
    here instead of vanishing. The wake arrays themselves are immutable from
    the policy's seat; the caller applies the flagged truncation at commit.
    """

    name = "WakeRowCap"

    def __init__(self, max_rows: int, spanwise_panels: int) -> None:
        self.max_rows = max_rows
        self.ns = spanwise_panels

    def apply_wake(self, wake_rings, wake_gamma, max_rows: int) -> RetentionResult | None:
        n = wake_gamma.numel()
        max_items = self.max_rows * self.ns
        if n <= max_items:
            return None
        excess = n - max_items
        removed_gamma = wake_gamma[n - excess:]
        return RetentionResult(
            removed_particle_count=0,
            removed_circulation_sum=float(removed_gamma.abs().sum().item()),
            removed_linear_impulse=None,
            removed_angular_impulse=None,
            conservation_error=0.0,
        )
