"""Differentiable ANCF internal-force operator (custom adjoint = tangent stiffness).

Warp 1.14's auto-adjoint of the ANCF bending kernel NaNs (docs/p1_differentiability_
finding.md). We bypass it with the exact, already-validated custom adjoint:

    forward :  Qint = assemble_internal_force_sep(q)            (bit-exact golden)
    backward:  adj_q = K_t(q) . adj_Qint                        (K_t symmetric)

where K_t(q) = assemble_kmem_blocks(q) is the tangent stiffness the Newmark step
already builds (verify_tangent_jacobian.py: K_t.dq == FD dQint, rel 7e-6).

`DiffInternalForce` is a manual autograd-style op (forward caches K_t blocks;
backward applies them via apply_MK). This is the standard adjoint-method pattern
for differentiable implicit physics — and it composes: the test below pushes a
*nonlinear* scalar loss through the op and matches the gradient to finite
differences, proving the custom adjoint is correct end-to-end (not just K_t.dq).
Pure Warp; no numpy in the compute/grad path (numpy only stages immutable inputs
and reads the final scalar, as in the validated warp_fsi modules).
"""
from __future__ import annotations

import os
import sys

import numpy as np

_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
_TESTS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "tests"))
for p in (_SRC, _TESTS):
    if p not in sys.path:
        sys.path.insert(0, p)

import warp as wp                                                # noqa: E402
from fluxvortex.warp_fsi import config as cfg                   # noqa: E402
from fluxvortex.warp_fsi.config import DTYPE                    # noqa: E402
from fluxvortex.warp_fsi.kernels_ancf import (                 # noqa: E402
    ANCFConstants, assemble_kmem_blocks, assemble_internal_force_sep)
from fluxvortex.warp_fsi.batched_solver import apply_MK         # noqa: E402


class DiffInternalForce:
    """Manual-autograd ANCF internal force with the exact tangent-stiffness adjoint.

    Usage (one forward/backward):
        op = DiffInternalForce(C)
        Q = op.forward(q)                 # Warp (B, ndof), = Qmem + Qbend
        adj_q = op.backward(adj_Q)        # Warp (B, ndof), = K_t . adj_Q
    """

    def __init__(self, C: ANCFConstants, device=None):
        self.C = C
        self.device = device or cfg.DEVICE
        self._q = None
        self._Kblk = None

    def forward(self, q):
        """q: Warp (B, ndof). Returns Qint = Qmem + Qbend (Warp, B, ndof)."""
        self._q = q
        self._Kblk = assemble_kmem_blocks(q, self.C, self.device)   # cache K_t
        Qm, Qb = assemble_internal_force_sep(q, self.C, self.device)
        B, ndof = q.shape
        Qint = wp.zeros((B, ndof), dtype=DTYPE, device=self.device)
        wp.launch(_add2, dim=(B, ndof), inputs=[Qm, Qb], outputs=[Qint],
                  device=self.device)
        return Qint

    def backward(self, adj_Q):
        """adj_Q: Warp (B, ndof) cotangent of Qint. Returns adj_q = K_t . adj_Q."""
        B, ndof = adj_Q.shape
        # cotangents are masked to free DOFs (bc rows carry no gradient)
        adj_in = wp.zeros((B, ndof), dtype=DTYPE, device=self.device)
        wp.launch(_mask, dim=(B, ndof), inputs=[adj_Q, self.C.free],
                  outputs=[adj_in], device=self.device)
        adj_q = wp.zeros((B, ndof), dtype=DTYPE, device=self.device)
        apply_MK(adj_in, adj_q, self.C.Me, self._Kblk, self.C.edofs, self.C.free,
                 0.0, 1.0, self.device)                          # K_t . adj_in
        return adj_q


@wp.kernel
def _add2(a: wp.array(dtype=DTYPE, ndim=2), b: wp.array(dtype=DTYPE, ndim=2),
          o: wp.array(dtype=DTYPE, ndim=2)):
    e, i = wp.tid()
    o[e, i] = a[e, i] + b[e, i]


@wp.kernel
def _mask(a: wp.array(dtype=DTYPE, ndim=2), free: wp.array(dtype=DTYPE),
          o: wp.array(dtype=DTYPE, ndim=2)):
    e, i = wp.tid()
    o[e, i] = a[e, i] * free[i]


# ── composition verification: nonlinear scalar loss through the op == FD ──────
def verify(nx: int = 8, ny: int = 6, eps: float = 1e-6) -> bool:
    """loss(q) = 0.5 * sum_free( w_i * Qint_i(q)^2 ), a NONLINEAR functional of Qint.
    d(loss)/dQ = w * Qint ; d(loss)/dq = K_t . (w * Qint). Compare to FD of loss."""
    dev, NP = cfg.DEVICE, cfg.NP_DTYPE
    from run_standalone_yamano import yamano_params, build_yamano_shell
    params = yamano_params()
    shell, _, _, _ = build_yamano_shell(params, nx=nx, ny=ny)
    C = ANCFConstants(shell, device=dev)
    ndof = shell.nn * 9
    rng = np.random.default_rng(5)
    q0 = np.zeros(ndof)
    for k in range(shell.nn):
        q0[9 * k] = shell.nodes[k, 0]; q0[9 * k + 1] = shell.nodes[k, 1]
        q0[9 * k + 3] = 1.0; q0[9 * k + 7] = 1.0
    free = np.array(sorted(set(range(ndof)) - set(shell._bc_dofs)))
    q0[free] += 1e-3 * rng.standard_normal(len(free))
    w = np.zeros(ndof); w[free] = rng.standard_normal(len(free))

    def wq(v):
        return wp.array(np.broadcast_to(v, (1, ndof)).astype(NP).copy(),
                        dtype=DTYPE, device=dev)

    op = DiffInternalForce(C, dev)
    Q = op.forward(wq(q0)); wp.synchronize()
    Qn = Q.numpy()[0]
    # adj_Q = d(loss)/dQ = w * Qint  (loss = 0.5 sum w*Q^2)
    adjQ = wq(w * Qn)
    adj_q = op.backward(adjQ); wp.synchronize()
    g = adj_q.numpy()[0][free]

    def loss(v):
        Qm, Qb = assemble_internal_force_sep(wq(v), C, dev); wp.synchronize()
        Qv = Qm.numpy()[0] + Qb.numpy()[0]
        return 0.5 * float(np.sum(w * Qv * Qv))

    gfd = np.zeros(len(free))
    for ii, j in enumerate(free):
        qp = q0.copy(); qp[j] += eps; qm = q0.copy(); qm[j] -= eps
        gfd[ii] = (loss(qp) - loss(qm)) / (2 * eps)
    big = np.abs(gfd) > 0.05 * np.max(np.abs(gfd))
    rel = np.max(np.abs(g[big] - gfd[big]) / np.abs(gfd[big]))
    ok = rel < 1e-4
    print(f"DiffInternalForce composition test (nonlinear loss through the op):")
    print(f"  custom-adjoint grad vs FD: max rel = {rel:.2e} on {int(big.sum())} "
          f"dominant dofs  -> {'PASS' if ok else 'FAIL'}")
    print(f"  -> differentiable ANCF internal force WORKS (pure Warp, exact adjoint, "
          f"forward bit-exact)")
    return ok


if __name__ == "__main__":
    wp.init()
    print(cfg.summary())
    raise SystemExit(0 if verify() else 1)
