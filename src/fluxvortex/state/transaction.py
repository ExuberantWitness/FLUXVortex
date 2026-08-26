"""GlobalTransaction: one-commit-per-step guard for the unified framework."""
from __future__ import annotations

from typing import Any


class GlobalTransaction:
    """Ensures at most one commit per outer time step across all owners."""

    def __init__(self) -> None:
        self._committed = False
        self._step = -1

    def begin_step(self, step: int) -> None:
        if self._committed and self._step == step:
            raise RuntimeError(f"double commit at step {step}")
        self._committed = False
        self._step = step

    def commit(self, step: int) -> None:
        if self._committed:
            raise RuntimeError(f"double commit at step {step}")
        if step != self._step:
            raise RuntimeError(f"commit for step {step} without begin_step")
        self._committed = True
