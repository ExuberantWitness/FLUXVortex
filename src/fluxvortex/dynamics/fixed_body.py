"""FixedDynamics: no dynamic degrees of freedom (the structure is prescribed)."""
from __future__ import annotations
from typing import Any

class FixedDynamics:
    """For rigid prescribed-motion cases where nothing integrates.

    The state is a constant — the motion law directly determines all
    surface positions. Used with OneWayPrescribedCoupling.
    """
    def predict(self, committed: Any, dt: float) -> Any:
        return committed

    def propose(self, committed: Any, loads: Any, dt: float) -> Any:
        return committed
