"""Unified result status: execution/numerical/physics/accuracy/reproduction
are SEPARATE (refactor plan §13, fixing the 'completed' conflation)."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any

@dataclass(frozen=True)
class ResultStatus:
    execution_status: str = "pending"       # pending|running|completed|failed|aborted
    numerical_status: str = "pending"       # pending|converged|nonconverged
    physics_gate_status: str = "pending"    # pending|passed|failed
    accuracy_gate_status: str = "pending"   # pending|passed|failed|not_applicable
    reproduction_status: str = "pending"    # pending|passed|partial|failed

    @property
    def is_formal_reproduction(self) -> bool:
        return (
            self.execution_status == "completed"
            and self.numerical_status == "converged"
            and self.physics_gate_status == "passed"
            and self.accuracy_gate_status == "passed"
            and self.reproduction_status == "passed"
        )

    @property
    def exit_code(self) -> int:
        """Accuracy gate failure must produce a non-zero exit."""
        if self.execution_status == "failed":
            return 1
        if self.accuracy_gate_status == "failed":
            return 2
        if self.physics_gate_status == "failed":
            return 3
        if self.numerical_status == "nonconverged":
            return 4
        return 0
