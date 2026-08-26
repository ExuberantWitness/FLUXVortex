"""Dynamic subsystem protocol (refactor plan §6.3)."""
from __future__ import annotations
from typing import Protocol, runtime_checkable, Any

@runtime_checkable
class DynamicSubsystem(Protocol):
    def predict(self, committed: Any, dt: float) -> Any: ...
    def propose(self, committed: Any, loads: Any, dt: float) -> Any: ...
