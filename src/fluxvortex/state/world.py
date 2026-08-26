"""Unified world state and transaction ownership."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import torch

@dataclass(frozen=True)
class WorldDynamicState:
    """Holds committed structural/rigid-body states keyed by subsystem id."""
    elastic_states: dict[str, Any]   # elastic_id -> ElasticState-like (wp arrays)
    body_states: dict[str, Any]      # body_id -> RigidBodyState-like (future)
    joint_states: dict[str, Any]     # joint_id -> JointState-like (future)

@dataclass
class WorldOwner:
    """The single committed owner of dynamic + aerodynamic state."""
    dynamic_state: WorldDynamicState
    aero_state: Any                  # V5MWorldState-like (currently NativeV5MState)
    previous_load: Any | None        # GeneralizedLoadPacket-like
    generation: int = 0

    def digest(self) -> str:
        """Roll the committed-generation digest for trial parent checks."""
        return f"gen:{self.generation}"
