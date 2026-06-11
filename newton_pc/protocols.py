"""Minimal protocols for the window predictor-corrector coupler.

The coupler is solver-agnostic: it talks to the structural side and the force
(fluid) side exclusively through these two protocols plus an interpolable
force container. Keeping them minimal is what makes the scheme portable —
the same orchestration drives a hand-rolled FEM code or a Newton solver
(via ``State.particle_f`` / ``State.body_f`` force buffers).

Design notes (mapping to newton-physics PR #2848 concepts):
  - ``StructuralEntry.snapshot``/``restore`` generalize the proxy framework's
    ``iteration_restart`` state redistribution from a single top-level step to
    a multi-substep window.
  - ``StructuralEntry.substep(t, dt, forces)`` receives a *time-interpolated*
    ``ForceSet`` each substep — the capability missing from #2848, where the
    coupler-written forces are zero-order-held across substeps
    (solver_coupled.py:2152/2169). Our benchmark evidence: ZOH = 42% error vs
    interpolated = 0.0005% over 500 substeps on a 1e-6-validated reference.
  - ``ForceProvider.solve`` must be repeatable (pure w.r.t. its own auxiliary
    state) so the window can be re-solved during predictor/corrector passes or
    Picard iterations; ``commit`` advances auxiliary state (e.g. the wake)
    exactly once per accepted window.
"""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class ForceSet(Protocol):
    """Interpolable container of coupling forces (vectors and/or operators).

    Implementations carry whatever the structural side consumes per substep —
    nodal force vectors, force matrices applied to current velocities, or
    operators destined for the effective mass matrix. All entries must
    support the affine combination used for time interpolation.
    """

    def affine(self, other: "ForceSet", beta: float) -> "ForceSet":
        """Return ``self + (other - self) * beta``.

        With ``0 <= beta <= 1`` this is linear interpolation; ``beta > 1``
        extrapolates (used by the predictor pass).
        """
        ...


@runtime_checkable
class StructuralEntry(Protocol):
    """The structure-side contract: window rewind + force-driven substepping."""

    def snapshot(self) -> Any:
        """Capture all state needed to re-simulate the current window."""
        ...

    def restore(self, snap: Any) -> None:
        """Rewind to a previously captured snapshot."""
        ...

    def substep(self, t: float, dt: float, forces: ForceSet) -> None:
        """Advance one substep under the given (already interpolated) forces.

        ``t`` is the absolute end-of-substep time (the entry evaluates its own
        non-coupling external loads, e.g. an actuation pulse, at ``t``).
        """
        ...

    def state(self) -> Any:
        """Opaque state handle consumed by ``ForceProvider.solve``."""
        ...


@runtime_checkable
class ForceProvider(Protocol):
    """The force-side contract: repeatable window solves + explicit commit."""

    def solve(self, state: Any) -> ForceSet:
        """Compute coupling forces at ``state``.

        Must be repeatable: auxiliary state (wake, history) is frozen, so
        calling ``solve`` multiple times within one window (predictor pass,
        Picard iterations) yields self-consistent results.
        """
        ...

    def commit(self, forces: ForceSet) -> None:
        """Accept the window: advance auxiliary state exactly once."""
        ...
