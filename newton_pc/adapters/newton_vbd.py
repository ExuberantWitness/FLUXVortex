"""Newton (VBD cloth) adapters for the window predictor-corrector.

Couples a ``newton.solvers.SolverVBD`` cloth to an aerodynamic force provider
through the public ``State.particle_f`` buffer — the integration path the
Newton team recommends for external physics (no core changes, no collision
shapes). The window rewind uses plain ``particle_q``/``particle_qd`` snapshots,
which the VBD smoke test shows replay bit-identically.

The aero side reuses ``fluxvortex.standalone_uvlm.StandaloneUVLM`` (ring-vortex
lifting surface, quasi-steady) built directly on the cloth vertex grid; panel
forces are lumped to the four corner particles. This demo arena has no
external ground truth — it is used for SAME-ARENA comparisons (lagged
zero-order-hold vs two-pass window coupling); quantitative validation lives in
the FLUXV adapter where the 1e-6 MATLAB reference exists.
"""
from __future__ import annotations

from typing import Any

import numpy as np


class ParticleForceSet:
    """Per-particle force vectors; trivially affine."""

    def __init__(self, f: np.ndarray):
        self.f = f

    def affine(self, other: "ParticleForceSet", beta: float) -> "ParticleForceSet":
        return ParticleForceSet(self.f + (other.f - self.f) * beta)

    def lincomb(self, pairs) -> "ParticleForceSet":
        acc = None
        for fs, w in pairs:
            term = fs.f * w
            acc = term if acc is None else acc + term
        return ParticleForceSet(acc)


class VBDEntry:
    """StructuralEntry over a Newton VBD cloth (particle_f force injection)."""

    def __init__(self, model, solver, state_0, state_1, control,
                 extra_force: np.ndarray | None = None):
        self.model = model
        self.solver = solver
        self.s0 = state_0
        self.s1 = state_1
        self.control = control
        self.extra_force = extra_force  # e.g. gravity replacement / actuation

    def snapshot(self) -> Any:
        return (self.s0.particle_q.numpy().copy(),
                self.s0.particle_qd.numpy().copy())

    def restore(self, snap: Any) -> None:
        q, qd = snap
        self.s0.particle_q.assign(q)
        self.s0.particle_qd.assign(qd)

    def substep(self, t: float, dt: float, forces: ParticleForceSet) -> None:
        self.s0.clear_forces()
        f = forces.f
        if self.extra_force is not None:
            f = f + self.extra_force
        self.s0.particle_f.assign(f.astype(np.float32))
        self.solver.step(self.s0, self.s1, self.control, None, dt)
        self.s0, self.s1 = self.s1, self.s0

    def state(self) -> np.ndarray:
        return self.s0.particle_q.numpy().copy()


class UVLMProvider:
    """ForceProvider: quasi-steady ring-vortex UVLM on the cloth vertex grid."""

    def __init__(self, nx: int, ny: int, V_inf, rho: float = 1.225,
                 dt_wake: float = 1.0):
        import os
        import sys
        _root = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             "..", ".."))
        if os.path.join(_root, "src") not in sys.path:
            sys.path.insert(0, os.path.join(_root, "src"))
        from fluxvortex.standalone_uvlm import StandaloneUVLM
        self._mk = StandaloneUVLM
        self.nx, self.ny = nx, ny
        self.V_inf = np.asarray(V_inf, dtype=float)
        self.rho = rho
        self.dt_wake = dt_wake
        self.n_solves = 0

    def _grid(self, particle_q: np.ndarray) -> np.ndarray:
        return particle_q.reshape(self.nx + 1, self.ny + 1, 3).astype(float)

    def solve(self, state: np.ndarray) -> ParticleForceSet:
        uvlm = self._mk(self._grid(state), self.V_inf, rho=self.rho)
        uvlm.disable_wake = True   # quasi-steady demo (relative comparisons)
        uvlm.solve()
        uvlm.compute_forces(self.dt_wake)
        self.n_solves += 1
        panel_f = uvlm.forces_no_vstruct  # (nx, ny, 3) per-panel force
        f = np.zeros(((self.nx + 1) * (self.ny + 1), 3))
        fg = f.reshape(self.nx + 1, self.ny + 1, 3)
        quarter = 0.25 * panel_f
        fg[:-1, :-1] += quarter
        fg[1:, :-1] += quarter
        fg[:-1, 1:] += quarter
        fg[1:, 1:] += quarter
        return ParticleForceSet(f)

    def commit(self, forces: ParticleForceSet) -> None:
        pass  # quasi-steady provider carries no auxiliary state
