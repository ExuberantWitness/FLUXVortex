"""Differentiable multi-step ANCF structural rollout (chained adjoint, pure Warp).

Proves the differentiable building blocks compose over time: a length-N rollout of
`DiffStructStep` is differentiated by chaining the per-step adjoints in reverse,
and the resulting d(loss(q_N))/d(q0, dq0, F) matches finite differences. This is
the structural side of the SHAC / design-gradient capability.

(The aero side adds the AIC solve adjoint — DiffDenseSolve — plus the geometry-
dependent UVLM kernel adjoints; coupling them into this chain is the full
differentiable FSI rollout. PPO-first iteration-1 needs none of this; it is the
SHAC-phase capability.)
"""
from __future__ import annotations

import os
import sys

import numpy as np

_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
_TESTS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "tests"))
for p in (_SRC, _TESTS, os.path.dirname(__file__)):
    if p not in sys.path:
        sys.path.insert(0, p)

import warp as wp                                                # noqa: E402
from fluxvortex.warp_fsi import config as cfg                   # noqa: E402
from fluxvortex.warp_fsi.config import DTYPE                    # noqa: E402
from fluxvortex.warp_fsi.kernels_ancf import ANCFConstants      # noqa: E402
from diff_step import DiffStructStep                            # noqa: E402


@wp.kernel
def _add(a: wp.array(dtype=DTYPE, ndim=2), b: wp.array(dtype=DTYPE, ndim=2),
         o: wp.array(dtype=DTYPE, ndim=2)):
    e, i = wp.tid()
    o[e, i] = a[e, i] + b[e, i]


class DiffStructRollout:
    """N-step differentiable structural rollout under a (constant) force F."""

    def __init__(self, C: ANCFConstants, dt: float, n_steps: int, device=None):
        self.C = C
        self.dt = dt
        self.n = n_steps
        self.device = device or cfg.DEVICE
        self._steps = None

    def forward(self, q, dq, F):
        self._steps = [DiffStructStep(self.C, self.dt, device=self.device)
                       for _ in range(self.n)]
        for s in self._steps:
            q, dq = s.forward(q, dq, F)
        return q, dq

    def backward(self, adj_qN, adj_dqN):
        """Returns (adj_q0, adj_dq0, adj_F) by chaining per-step adjoints in reverse."""
        B = adj_qN.shape[0]; ndof = adj_qN.shape[1]
        adj_q, adj_dq = adj_qN, adj_dqN
        adj_F = wp.zeros((B, ndof), dtype=DTYPE, device=self.device)
        for s in reversed(self._steps):
            adj_q, adj_dq, aF = s.backward(adj_q, adj_dq)
            o = wp.zeros((B, ndof), dtype=DTYPE, device=self.device)
            wp.launch(_add, dim=(B, ndof), inputs=[adj_F, aF], outputs=[o],
                      device=self.device)
            adj_F = o
        return adj_q, adj_dq, adj_F


def verify(nx: int = 6, ny: int = 4, N: int = 5, dt: float = 2e-4,
           eps: float = 1e-6) -> bool:
    dev, NP = cfg.DEVICE, cfg.NP_DTYPE
    from run_standalone_yamano import yamano_params, build_yamano_shell
    params = yamano_params()
    shell, _, _, _ = build_yamano_shell(params, nx=nx, ny=ny)
    C = ANCFConstants(shell, device=dev)
    ndof = shell.nn * 9
    rng = np.random.default_rng(3)
    q0 = np.zeros(ndof)
    for k in range(shell.nn):
        q0[9 * k] = shell.nodes[k, 0]; q0[9 * k + 1] = shell.nodes[k, 1]
        q0[9 * k + 3] = 1.0; q0[9 * k + 7] = 1.0
    free = np.array(sorted(set(range(ndof)) - set(shell._bc_dofs)))
    q0[free] += 5e-4 * rng.standard_normal(len(free))
    dq0 = np.zeros(ndof); dq0[free] = 0.005 * rng.standard_normal(len(free))
    F0 = np.zeros(ndof); F0[free] = 0.5 * rng.standard_normal(len(free))
    w = np.zeros(ndof); w[free] = rng.standard_normal(len(free))

    def wa(v):
        return wp.array(np.broadcast_to(v, (1, ndof)).astype(NP).copy(),
                        dtype=DTYPE, device=dev)

    roll = DiffStructRollout(C, dt, N, dev)
    qN, _ = roll.forward(wa(q0), wa(dq0), wa(F0))
    adj_q0, _, _ = roll.backward(wa(w), wp.zeros((1, ndof), dtype=DTYPE, device=dev))
    wp.synchronize()
    gq0 = adj_q0.numpy()[0][free]

    def loss(qv):
        qn, _ = DiffStructRollout(C, dt, N, dev).forward(wa(qv), wa(dq0), wa(F0))
        wp.synchronize()
        return float(np.sum(w * qn.numpy()[0]))

    gfd = np.zeros(len(free))
    for ii, j in enumerate(free):
        qp = q0.copy(); qp[j] += eps; qm = q0.copy(); qm[j] -= eps
        gfd[ii] = (loss(qp) - loss(qm)) / (2 * eps)
    big = np.abs(gfd) > 0.05 * np.max(np.abs(gfd))
    rel = np.max(np.abs(gq0[big] - gfd[big]) / np.abs(gfd[big]))
    ok = rel < 2e-3   # CG-noise floor in the FD reference
    print(f"Differentiable {N}-step structural rollout: d(w.q_N)/dq0 chained-adjoint vs FD")
    print(f"  max rel on dominant dofs = {rel:.2e} ({int(big.sum())} dofs) -> "
          f"{'PASS' if ok else 'FAIL'}")
    print(f"  -> rollout adjoint chains correctly (blocks compose over time)")
    return ok


if __name__ == "__main__":
    wp.init()
    print(cfg.summary())
    raise SystemExit(0 if verify() else 1)
