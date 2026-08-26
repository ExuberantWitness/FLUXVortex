"""Q16DynamicsAdapter: wraps the existing Q16CudaNewmarkStepper into the
unified DynamicSubsystem protocol. No formulas duplicated."""
from __future__ import annotations

from typing import Any

import warp as wp


class Q16DynamicsAdapter:
    """Unified DynamicSubsystem view over the production Q16 CUDA stepper.

    The adapter owns no physics: every call delegates to the existing
    constrained Newmark predictor / Newton structural step so the unified
    framework and the frozen production path share one implementation.
    """

    def __init__(self, structural_solver: Any, elastic_id: str = "membrane_0") -> None:
        self._solver = structural_solver
        self.elastic_id = elastic_id

    def predict(self, committed_state: Any, dt: float):
        """Return the constrained Newmark predictor (delegates to solver)."""
        elastic = committed_state.elastic_states[self.elastic_id]
        q, v, a = elastic.q, elastic.qd, elastic.qdd
        return self._solver.predict_kinematics(q, v, a, delta_time=dt)

    def propose(self, committed_state: Any, loads: Any, dt: float):
        """Delegate one structural step to the existing solver."""
        elastic = committed_state.elastic_states[self.elastic_id]
        return self._solver.step(
            elastic.q,
            elastic.qd,
            elastic.qdd,
            loads,
            delta_time=dt,
        )
