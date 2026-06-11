"""Window predictor-corrector coupler for partitioned multiphysics.

The scheme (validated to 0.0005% over 500 substeps against an independent
MATLAB reference; see docs/newton_comparison.md for the evidence chain):

    predictor:  march the window once with forces extrapolated from the two
                previous window solves; this lands the structure at a
                *predicted* window-end state
    solve:      evaluate the force provider at the predicted state -> F_new
    rewind:     restore the window-start snapshot
    corrector:  re-march the window with forces interpolated F_old -> F_new
                per substep (linear or quadratic), optionally iterating
                (Picard) until the window residual converges

``mode="lagged"`` degrades the scheme to the zero-order-hold coupling used by
existing frameworks (one solve per window, force held constant) — kept as a
built-in baseline because the accuracy gap between the two modes is the core
quantitative argument for this coupler.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from .protocols import ForceProvider, ForceSet, StructuralEntry


@dataclass
class WindowStats:
    """Per-window bookkeeping returned by :meth:`WindowPredictorCorrector.advance`."""

    window_index: int
    t_end: float
    n_solves: int
    n_marches: int
    iterations: int
    residual: float | None


@dataclass
class WindowPredictorCorrector:
    """Drive one ``StructuralEntry`` + one ``ForceProvider`` in coupled windows.

    Args:
        entry: structure-side adapter (snapshot/restore/substep/state).
        provider: force-side adapter (repeatable solve + commit).
        substeps: structural substeps per coupling window.
        dt: structural substep size.
        mode: ``"two-pass"`` (predictor + corrector, default) or ``"lagged"``
            (zero-order hold baseline).
        interp: ``"linear"`` (default) or ``"quad"`` — corrector force
            interpolation across the window. ``"quad"`` fits a parabola
            through (F_prev, F_cur, F_new); zero extra cost, validated to
            cut heavy-loading interpolation error ~2x.
        iterations: corrector (Picard) iterations per window. 1 reproduces
            the validated baseline scheme.
        adaptive_tol: if set, iterate the corrector until the relative
            window residual (change of window-end state between successive
            iterations, measured by ``residual_norm``) drops below this
            tolerance, up to ``iterations`` as the cap.
        residual_norm: callable mapping (state_prev, state_new) -> float,
            required when ``adaptive_tol`` is set.
    """

    entry: StructuralEntry
    provider: ForceProvider
    substeps: int
    dt: float
    mode: str = "two-pass"
    interp: str = "linear"
    iterations: int = 1
    adaptive_tol: float | None = None
    residual_norm: Callable[[Any, Any], float] | None = None

    _F_prev: ForceSet | None = field(default=None, init=False, repr=False)
    _F_cur: ForceSet | None = field(default=None, init=False, repr=False)
    _t: float = field(default=0.0, init=False, repr=False)
    _window_index: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.mode not in ("two-pass", "lagged"):
            raise ValueError(f"unknown mode {self.mode!r}")
        if self.interp not in ("linear", "quad"):
            raise ValueError(f"unknown interp {self.interp!r}")
        if self.adaptive_tol is not None and self.residual_norm is None:
            raise ValueError("adaptive_tol requires residual_norm")

    # ------------------------------------------------------------------
    @property
    def window_length(self) -> float:
        return self.substeps * self.dt

    def initialize(self, zero_forces: ForceSet) -> None:
        """Boot with zero coupling forces (MATLAB-equivalent cold start).

        The first window should then be advanced with ``advance(n_substeps=1)``:
        the predictor degenerates to zero marches, the provider solves at the
        initial state, and the corrector ramps the forces in over the first
        substep — reproducing the reference scheme's boot exactly.
        """
        self._F_prev = zero_forces
        self._F_cur = zero_forces

    # ------------------------------------------------------------------
    def _march(self, t0: float, tf: float, n: int,
               force_at: Callable[[float], ForceSet]) -> None:
        """March ``n`` substeps from time ``t0``.

        ``force_at(beta)`` supplies interpolated forces; ``beta`` is
        normalized by the NOMINAL window length anchored at ``tf`` (the time
        of the last accepted force solve), matching the reference scheme even
        for irregular (e.g. boot) windows.
        """
        for k in range(n):
            t = t0 + (k + 1) * self.dt
            beta = (t - tf) / self.window_length
            self.entry.substep(t, self.dt, force_at(beta))

    def _corrector_force(self, F_new: ForceSet) -> Callable[[float], ForceSet]:
        F_prev, F_cur = self._F_prev, self._F_cur
        if self.interp == "quad" and self._window_index >= 2:
            if not hasattr(F_cur, "lincomb"):
                raise TypeError("interp='quad' requires the ForceSet to "
                                "implement lincomb(pairs)")
            # parabola through F_prev(beta=-1), F_cur(0), F_new(+1)
            def force_at(beta: float) -> ForceSet:
                w_prev = 0.5 * (beta * beta - beta)
                w_cur = 1.0 - beta * beta
                w_new = 0.5 * (beta * beta + beta)
                return F_cur.lincomb(((F_prev, w_prev), (F_cur, w_cur),
                                      (F_new, w_new)))
            return force_at
        return lambda beta: F_cur.affine(F_new, beta)

    # ------------------------------------------------------------------
    def advance(self, n_substeps: int | None = None) -> WindowStats:
        """Advance one coupling window (``n_substeps`` defaults to the nominal
        window; pass 1 for the boot window). Returns per-window statistics."""
        if self._F_cur is None:
            raise RuntimeError("call initialize() first")
        t0 = self._t
        tf = self._t  # anchor of the force interpolation = last accepted solve
        n = self.substeps if n_substeps is None else n_substeps
        n_solves = 0
        n_marches = 0
        residual = None
        iters_done = 1

        if self.mode == "lagged":
            # zero-order hold: one solve at the window-START state, force
            # held constant across all substeps (the #2848 substep semantics)
            F_new = self.provider.solve(self.entry.state())
            n_solves += 1
            self._march(t0, tf, n, lambda beta: F_new)
            n_marches += 1
        else:
            snap = self.entry.snapshot()
            # -- predictor: extrapolate F_cur + (F_cur - F_prev)*beta --
            F_prev, F_cur = self._F_prev, self._F_cur
            pred = lambda beta: F_prev.affine(F_cur, 1.0 + beta)
            if n > 1:
                self._march(t0, tf, n - 1, pred)
                n_marches += 1
            F_new = self.provider.solve(self.entry.state())
            n_solves += 1
            # -- corrector (with optional Picard iterations) --
            prev_end_state = None
            for it in range(max(1, self.iterations)):
                iters_done = it + 1
                self.entry.restore(snap)
                self._march(t0, tf, n, self._corrector_force(F_new))
                n_marches += 1
                if it + 1 >= max(1, self.iterations):
                    break
                # re-solve at the corrected pre-boundary state
                self.entry.restore(snap)
                self._march(t0, tf, n - 1, self._corrector_force(F_new))
                n_marches += 1
                end_state = self.entry.state()
                if self.adaptive_tol is not None and prev_end_state is not None:
                    residual = self.residual_norm(prev_end_state, end_state)
                    if residual < self.adaptive_tol:
                        # converged: finish with a full corrector march
                        self.entry.restore(snap)
                        self._march(t0, tf, n, self._corrector_force(F_new))
                        n_marches += 1
                        break
                prev_end_state = end_state
                F_new = self.provider.solve(end_state)
                n_solves += 1

        self.provider.commit(F_new)
        self._F_prev = self._F_cur
        self._F_cur = F_new
        self._t = t0 + n * self.dt
        self._window_index += 1
        return WindowStats(self._window_index, self._t, n_solves, n_marches,
                           iters_done, residual)

    def run(self, n_windows: int,
            on_window: Callable[[WindowStats], None] | None = None) -> None:
        for _ in range(n_windows):
            stats = self.advance()
            if on_window is not None:
                on_window(stats)
