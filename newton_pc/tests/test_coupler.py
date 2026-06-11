"""Unit tests for WindowPredictorCorrector on a tiny analytic arena.

A 1-DOF damped oscillator driven by a state-dependent "aero" force
(F = -k_c * x). The tightly-coupled limit (1-substep windows) is the
reference; tests assert the structural properties the scheme guarantees:

  - snapshot/restore replay is exact
  - two-pass beats lagged against the tightly-coupled reference
  - quad interpolation weights reproduce the parabola through the anchors
  - the boot window ramps forces (no force jump at t=0)

Run: pytest newton_pc/tests/test_coupler.py -q
"""
from __future__ import annotations

import numpy as np

from newton_pc import WindowPredictorCorrector


class ToyForces:
    def __init__(self, f: float):
        self.f = float(f)

    def affine(self, other: "ToyForces", beta: float) -> "ToyForces":
        return ToyForces(self.f + (other.f - self.f) * beta)

    def lincomb(self, pairs) -> "ToyForces":
        return ToyForces(sum(fs.f * w for fs, w in pairs))


class ToyEntry:
    """Semi-implicit 1-DOF oscillator: m x'' + c x' + k x = F."""

    def __init__(self, m=1.0, c=0.05, k=4.0):
        self.m, self.c, self.k = m, c, k
        self.x, self.v = 1.0, 0.0
        self.force_trace: list[float] = []

    def snapshot(self):
        return (self.x, self.v)

    def restore(self, snap):
        self.x, self.v = snap

    def substep(self, t, dt, forces: ToyForces):
        self.force_trace.append(forces.f)
        a = (forces.f - self.c * self.v - self.k * self.x) / self.m
        self.v += dt * a
        self.x += dt * self.v

    def state(self):
        return (self.x, self.v)


class ToyProvider:
    """State-feedback force F = -k_c * x (repeatable, no aux state)."""

    def __init__(self, k_c=2.0):
        self.k_c = k_c
        self.n_solves = 0

    def solve(self, state) -> ToyForces:
        self.n_solves += 1
        return ToyForces(-self.k_c * state[0])

    def commit(self, forces) -> None:
        pass


def _run(mode: str, substeps: int, windows: int, **kw) -> float:
    entry, provider = ToyEntry(), ToyProvider()
    pc = WindowPredictorCorrector(entry=entry, provider=provider,
                                  substeps=substeps, dt=1e-2, mode=mode, **kw)
    pc.initialize(ToyForces(0.0))
    pc.advance(n_substeps=1)
    for _ in range(windows):
        pc.advance()
    return entry.x


def test_snapshot_restore_exact():
    entry = ToyEntry()
    snap = entry.snapshot()
    for k in range(5):
        entry.substep(0.01 * (k + 1), 0.01, ToyForces(0.3))
    x1 = entry.x
    entry.restore(snap)
    for k in range(5):
        entry.substep(0.01 * (k + 1), 0.01, ToyForces(0.3))
    assert entry.x == x1


def test_two_pass_beats_lagged():
    n_sub, n_win = 20, 30
    ref = _run("two-pass", 1, n_sub * n_win)   # tightly coupled limit
    lag = _run("lagged", n_sub, n_win)
    two = _run("two-pass", n_sub, n_win)
    assert abs(two - ref) < abs(lag - ref), (
        f"two-pass ({abs(two - ref):.2e}) should beat lagged "
        f"({abs(lag - ref):.2e}) vs the tightly-coupled reference")


def test_quad_weights_reproduce_parabola():
    f_prev, f_cur, f_new = ToyForces(1.0), ToyForces(2.0), ToyForces(5.0)
    # F(b) parabola through (-1, 1.0), (0, 2.0), (1, 5.0): F(b)=2 + 2b + b^2
    for beta in (0.0, 0.25, 0.5, 1.0):
        w_prev = 0.5 * (beta * beta - beta)
        w_cur = 1.0 - beta * beta
        w_new = 0.5 * (beta * beta + beta)
        got = f_cur.lincomb(((f_prev, w_prev), (f_cur, w_cur), (f_new, w_new))).f
        want = 2.0 + 2.0 * beta + beta * beta
        assert abs(got - want) < 1e-12


def test_boot_window_ramps_forces():
    entry, provider = ToyEntry(), ToyProvider()
    pc = WindowPredictorCorrector(entry=entry, provider=provider,
                                  substeps=10, dt=1e-2)
    pc.initialize(ToyForces(0.0))
    pc.advance(n_substeps=1)
    # boot corrector ramps 0 -> F0 with beta=1/10: |force| << |F0|
    f_boot = entry.force_trace[-1]
    F0 = -provider.k_c * 1.0
    assert abs(f_boot) <= abs(F0) * 0.2


def test_adaptive_iterations_converge():
    entry, provider = ToyEntry(), ToyProvider()
    pc = WindowPredictorCorrector(
        entry=entry, provider=provider, substeps=20, dt=1e-2,
        iterations=5, adaptive_tol=1e-10,
        residual_norm=lambda a, b: abs(a[0] - b[0]) + abs(a[1] - b[1]))
    pc.initialize(ToyForces(0.0))
    pc.advance(n_substeps=1)
    stats = pc.advance()
    assert stats.iterations <= 5
    assert stats.residual is None or stats.residual < 1e-6
