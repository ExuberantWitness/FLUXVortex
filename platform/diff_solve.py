"""Differentiable dense linear solve (custom adjoint) — the UVLM AIC core.

The UVLM circulation solve `gamma = AIC^{-1} rhs` uses `batched_dense_solve` (a
wrapped LU); Warp cannot auto-differentiate it. The exact custom adjoint of
`x = solve(A, b)` is

    adj_b = solve(Aᵀ, adj_x)
    adj_A = − adj_b ⊗ x            (adj_A[i,j] = −adj_b[i]·x[j])

`DiffDenseSolve` implements it (pure Warp). `verify` checks d(loss(x))/db and
d(loss)/dA against finite differences. This is the linear-solve VJP that makes the
aero side differentiable; chaining A=AIC(geometry) and rhs(geometry) onto the
structural q is the next step (those kernels' own adjoints).
"""
from __future__ import annotations

import os
import sys

import numpy as np

_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import warp as wp                                                # noqa: E402
from fluxvortex.warp_fsi import config as cfg                   # noqa: E402
from fluxvortex.warp_fsi.config import DTYPE                    # noqa: E402
from fluxvortex.warp_fsi.batched_solver import batched_dense_solve  # noqa: E402


@wp.kernel
def _transpose(A: wp.array(dtype=DTYPE, ndim=3), AT: wp.array(dtype=DTYPE, ndim=3)):
    e, i, j = wp.tid()
    AT[e, i, j] = A[e, j, i]


@wp.kernel
def _outer_neg(adj_b: wp.array(dtype=DTYPE, ndim=2), x: wp.array(dtype=DTYPE, ndim=2),
               adj_A: wp.array(dtype=DTYPE, ndim=3)):
    e, i, j = wp.tid()
    adj_A[e, i, j] = -adj_b[e, i] * x[e, j]


class DiffDenseSolve:
    def __init__(self, device=None):
        self.device = device or cfg.DEVICE
        self._A = None
        self._x = None

    def forward(self, A, b):
        self._A = A
        self._x = batched_dense_solve(A, b, self.device)
        return self._x

    def backward(self, adj_x):
        B, N, _ = self._A.shape
        AT = wp.zeros((B, N, N), dtype=DTYPE, device=self.device)
        wp.launch(_transpose, dim=(B, N, N), inputs=[self._A], outputs=[AT],
                  device=self.device)
        adj_b = batched_dense_solve(AT, adj_x, self.device)      # solve(Aᵀ, adj_x)
        adj_A = wp.zeros((B, N, N), dtype=DTYPE, device=self.device)
        wp.launch(_outer_neg, dim=(B, N, N), inputs=[adj_b, self._x], outputs=[adj_A],
                  device=self.device)
        return adj_A, adj_b


def verify(N: int = 24, eps: float = 1e-6) -> bool:
    dev, NP = cfg.DEVICE, cfg.NP_DTYPE
    rng = np.random.default_rng(11)
    A0 = np.eye(N) + 0.3 * rng.standard_normal((N, N))          # well-conditioned
    b0 = rng.standard_normal(N)
    w = rng.standard_normal(N)                                  # loss = w . x

    def wA(M):
        return wp.array(M[None].astype(NP).copy(), dtype=DTYPE, device=dev)

    def wv(v):
        return wp.array(v[None].astype(NP).copy(), dtype=DTYPE, device=dev)

    op = DiffDenseSolve(dev)
    x = op.forward(wA(A0), wv(b0)); wp.synchronize()
    adj_A, adj_b = op.backward(wv(w)); wp.synchronize()
    gA = adj_A.numpy()[0]; gb = adj_b.numpy()[0]

    def loss(A, b):
        xv = batched_dense_solve(wA(A), wv(b), dev); wp.synchronize()
        return float(np.dot(w, xv.numpy()[0]))

    # FD wrt b
    gb_fd = np.zeros(N)
    for j in range(N):
        bp = b0.copy(); bp[j] += eps; bm = b0.copy(); bm[j] -= eps
        gb_fd[j] = (loss(A0, bp) - loss(A0, bm)) / (2 * eps)
    rel_b = np.max(np.abs(gb - gb_fd)) / (np.max(np.abs(gb_fd)) + 1e-30)
    # FD wrt a sample of A entries
    idx = [(i, jj) for i in range(0, N, 5) for jj in range(0, N, 5)]
    gA_fd = np.array([(loss(A0 + eps * _E(N, i, j), b0)
                       - loss(A0 - eps * _E(N, i, j), b0)) / (2 * eps) for (i, j) in idx])
    gA_an = np.array([gA[i, j] for (i, j) in idx])
    rel_A = np.max(np.abs(gA_an - gA_fd)) / (np.max(np.abs(gA_fd)) + 1e-30)
    ok = rel_b < 1e-5 and rel_A < 1e-5
    print(f"DiffDenseSolve adjoint of x=solve(A,b)  (N={N}):")
    print(f"  d(loss)/db vs FD : rel={rel_b:.2e}")
    print(f"  d(loss)/dA vs FD : rel={rel_A:.2e}  ({len(idx)} sampled entries)")
    print(f"  -> {'PASS' if ok else 'FAIL'}: UVLM AIC linear-solve adjoint works "
          f"(adj_b=solve(Aᵀ,adj_x), adj_A=-adj_b⊗x)")
    return ok


def _E(N, i, j):
    M = np.zeros((N, N)); M[i, j] = 1.0
    return M


if __name__ == "__main__":
    wp.init()
    print(cfg.summary())
    raise SystemExit(0 if verify() else 1)
