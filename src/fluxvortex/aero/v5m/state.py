"""V5MWorldState: unified per-surface aero state with single separation owner.

Refactor plan §5.6/§5.9 (U2): the production solver keeps running its
numerics on ``NativeV5MState``; these structures WRAP that state and give it
the ownership/audit layer the old state lacked — one separation truth owner
per surface and a per-step circulation/impulse ledger through which every
retention decision must be accounted (see ``retention.py``, plan §8.5).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class V5MSurfaceState:
    """Per-surface aero state — one owner per truth."""

    surface_id: str
    bound_circulation: torch.Tensor      # (n_panels,)
    previous_bound_circulation: torch.Tensor
    # 3D LESP state — the SINGLE separation truth owner (plan §8.3)
    lesp_pre_3d: torch.Tensor | None     # (n_strips,) — filled by solver during propose
    surface_separated: torch.Tensor | None  # (n_strips,) bool — mask from frozen lesp_crit
    lesp_crit: float
    # LEV particles owned by this surface
    particle_field: Any                  # CudaParticleField
    # Wake rings owned by this surface (newest rows first, as in NativeV5MState)
    wake_rings: torch.Tensor             # (n_rings, 4, 3)
    wake_gamma: torch.Tensor             # (n_rings,)


@dataclass
class CirculationImpulseLedger:
    """Per-step audit of circulation and impulse conservation (§5.9).

    Every newborn LEV/TEV element, every removal/merge executed through a
    ``RetentionPolicy``, and the Kelvin residual of the whole step must land
    here; particle-age caps and wake-row caps may not be silent array
    slices.
    """

    step: int
    # Circulation
    bound_before: torch.Tensor           # (n_panels,)
    bound_after: torch.Tensor
    newborn_lev_circulation: float       # scalar sum
    newborn_tev_circulation: float
    removed_circulation: float           # from culling/merge
    kelvin_residual: float               # |sum(all) - sum_before|
    # Impulse (linear + angular) — filled by the solver when available
    linear_impulse_before: torch.Tensor | None = None   # (3,)
    linear_impulse_after: torch.Tensor | None = None
    angular_impulse_before: torch.Tensor | None = None  # (3,)
    angular_impulse_after: torch.Tensor | None = None

    @property
    def circulation_conservation_error(self) -> float:
        """Total circulation error: |bound_after + newborn + wake - bound_before|.

        This is a property the solver fills; for now return kelvin_residual.
        """
        return self.kelvin_residual


@dataclass
class V5MWorldState:
    """World-level V5M state: surfaces + wake system + ledger."""

    step_index: int
    generation: int
    surfaces: dict[str, V5MSurfaceState]
    ledger: CirculationImpulseLedger | None = None

    @classmethod
    def from_native_state(
        cls,
        native_state,
        surface_id: str = "wing_0",
        *,
        lesp_crit: float = 0.11,
    ) -> "V5MWorldState":
        """Wrap an existing NativeV5MState into the unified model.

        The wrap is by reference: bound circulation, wake, and the particle
        field stay the production solver's arrays, so the unified model can
        observe (and audit) the running numerics without duplicating truth.
        ``lesp_pre_3d``/``surface_separated`` are left empty here — the
        solver fills them during propose so the separation mask has exactly
        one owner (plan §8.3).
        """
        surface = V5MSurfaceState(
            surface_id=surface_id,
            bound_circulation=native_state.gamma_bound,
            previous_bound_circulation=native_state.gamma_previous,
            lesp_pre_3d=None,  # filled by solver during propose
            surface_separated=None,
            lesp_crit=lesp_crit,
            particle_field=native_state.particle_field,
            wake_rings=native_state.wake_rings,
            wake_gamma=native_state.wake_gamma,
        )
        return cls(
            step_index=native_state.step,
            generation=0,
            surfaces={surface_id: surface},
        )
