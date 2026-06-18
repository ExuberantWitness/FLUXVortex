"""Differentiable semi-implicit ANCF structural step (adjoint method, pure Warp).

The bit-exact validation path uses `gpu_newmark_step` (implicit, 12-layer golden).
For the **differentiable rollout** (SHAC / design gradients) we use a physically
consistent, cleanly differentiable symplectic step built on the proven pieces:

    M · a = F − Qint(q)            (consistent mass solve, structural_cg coef=0)
    dq⁺  = dq + dt · a
    q⁺   = q  + dt · dq⁺           (symplectic Euler)

Differentiability is by the **adjoint method**, exact and pure-Warp:
  - Qint(q) adjoint = K_t(q)·· via `DiffInternalForce` (already verified, 4e-8);
  - the mass solve a = M⁻¹·rhs has adjoint adj_rhs = M⁻¹·adj_a (M constant &
    symmetric → the *same* solve).

`verify` checks d(loss(q⁺,dq⁺))/d(q,dq,F) from this manual adjoint against finite
differences. One step now; chaining over a window = the differentiable rollout.
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
from fluxvortex.warp_fsi.batched_solver import structural_cg    # noqa: E402
from diff_ancf import DiffInternalForce                         # noqa: E402


@wp.kernel
def _axpby(a: wp.array(dtype=DTYPE, ndim=2), ca: DTYPE,
           b: wp.array(dtype=DTYPE, ndim=2), cb: DTYPE,
           o: wp.array(dtype=DTYPE, ndim=2)):
    e, i = wp.tid()
    o[e, i] = ca * a[e, i] + cb * b[e, i]


@wp.kernel
def _maskk(a: wp.array(dtype=DTYPE, ndim=2), free: wp.array(dtype=DTYPE),
           o: wp.array(dtype=DTYPE, ndim=2)):
    e, i = wp.tid()
    o[e, i] = a[e, i] * free[i]


class DiffStructStep:
    def __init__(self, C: ANCFConstants, dt: float, cg_tol: float = 1e-12, device=None):
        self.C = C
        self.dt = float(dt)
        self.device = device or cfg.DEVICE
        self.cg_tol = cg_tol
        self.ife = DiffInternalForce(C, self.device)
        self._cache = None

    def _z(self, B, ndof):
        return wp.zeros((B, ndof), dtype=DTYPE, device=self.device)

    def forward(self, q, dq, F):
        B, ndof = q.shape
        dt = DTYPE(self.dt)
        Qint = self.ife.forward(q)                              # K_t cached
        rhs = self._z(B, ndof)
        wp.launch(_axpby, dim=(B, ndof), inputs=[F, DTYPE(1.0), Qint, DTYPE(-1.0)],
                  outputs=[rhs], device=self.device)            # rhs = F - Qint
        rhs_m = self._z(B, ndof)
        wp.launch(_maskk, dim=(B, ndof), inputs=[rhs, self.C.free], outputs=[rhs_m],
                  device=self.device)
        a, _ = structural_cg(rhs_m, self.C.Me, self.ife._Kblk, self.C.edofs,
                             self.C.free, 0.0, ndof, tol=self.cg_tol,
                             device=self.device)                # M a = rhs
        dq_new = self._z(B, ndof)
        wp.launch(_axpby, dim=(B, ndof), inputs=[dq, DTYPE(1.0), a, dt],
                  outputs=[dq_new], device=self.device)         # dq+dt a
        q_new = self._z(B, ndof)
        wp.launch(_axpby, dim=(B, ndof), inputs=[q, DTYPE(1.0), dq_new, dt],
                  outputs=[q_new], device=self.device)          # q+dt dq_new
        self._cache = (B, ndof)
        return q_new, dq_new

    def backward(self, adj_q_new, adj_dq_new):
        """Returns (adj_q, adj_dq, adj_F)."""
        B, ndof = self._cache
        dt = DTYPE(self.dt)
        # q_new = q + dt dq_new
        adj_q = wp.clone(adj_q_new)
        adj_dqn = self._z(B, ndof)
        wp.launch(_axpby, dim=(B, ndof), inputs=[adj_dq_new, DTYPE(1.0), adj_q_new, dt],
                  outputs=[adj_dqn], device=self.device)        # adj_dq_new + dt adj_q_new
        # dq_new = dq + dt a
        adj_dq = wp.clone(adj_dqn)
        adj_a = self._z(B, ndof)
        wp.launch(_axpby, dim=(B, ndof), inputs=[adj_dqn, dt, adj_dqn, DTYPE(0.0)],
                  outputs=[adj_a], device=self.device)          # dt * adj_dqn
        # a = M^{-1} rhs -> adj_rhs = M^{-1} adj_a  (M symmetric, const)
        adj_a_m = self._z(B, ndof)
        wp.launch(_maskk, dim=(B, ndof), inputs=[adj_a, self.C.free], outputs=[adj_a_m],
                  device=self.device)
        adj_rhs, _ = structural_cg(adj_a_m, self.C.Me, self.ife._Kblk, self.C.edofs,
                                   self.C.free, 0.0, ndof, tol=self.cg_tol,
                                   device=self.device)
        # rhs = F - Qint -> adj_F += adj_rhs ; adj_Qint = -adj_rhs
        adj_F = wp.clone(adj_rhs)
        adj_Qint = self._z(B, ndof)
        wp.launch(_axpby, dim=(B, ndof), inputs=[adj_rhs, DTYPE(-1.0), adj_rhs, DTYPE(0.0)],
                  outputs=[adj_Qint], device=self.device)
        adj_q_from_Q = self.ife.backward(adj_Qint)              # K_t . adj_Qint
        wp.launch(_axpby, dim=(B, ndof), inputs=[adj_q, DTYPE(1.0), adj_q_from_Q, DTYPE(1.0)],
                  outputs=[adj_q], device=self.device)
        return adj_q, adj_dq, adj_F


def verify(nx: int = 8, ny: int = 6, dt: float = 2e-4, eps: float = 1e-6) -> bool:
    dev, NP = cfg.DEVICE, cfg.NP_DTYPE
    from run_standalone_yamano import yamano_params, build_yamano_shell
    params = yamano_params()
    shell, _, _, _ = build_yamano_shell(params, nx=nx, ny=ny)
    C = ANCFConstants(shell, device=dev)
    ndof = shell.nn * 9
    rng = np.random.default_rng(7)
    q0 = np.zeros(ndof)
    for k in range(shell.nn):
        q0[9 * k] = shell.nodes[k, 0]; q0[9 * k + 1] = shell.nodes[k, 1]
        q0[9 * k + 3] = 1.0; q0[9 * k + 7] = 1.0
    free = np.array(sorted(set(range(ndof)) - set(shell._bc_dofs)))
    q0[free] += 1e-3 * rng.standard_normal(len(free))
    dq0 = np.zeros(ndof); dq0[free] = 0.01 * rng.standard_normal(len(free))
    F0 = np.zeros(ndof); F0[free] = rng.standard_normal(len(free))
    wq_, wd_ = np.zeros(ndof), np.zeros(ndof)
    wq_[free] = rng.standard_normal(len(free)); wd_[free] = rng.standard_normal(len(free))

    def wa(v):
        return wp.array(np.broadcast_to(v, (1, ndof)).astype(NP).copy(), dtype=DTYPE, device=dev)

    step = DiffStructStep(C, dt)
    step.forward(wa(q0), wa(dq0), wa(F0))
    # loss = wq.q_new + wd.dq_new  (linear -> adj seeds are wq, wd)
    adj_q, adj_dq, adj_F = step.backward(wa(wq_), wa(wd_)); wp.synchronize()
    gq = adj_q.numpy()[0][free]; gdq = adj_dq.numpy()[0][free]; gF = adj_F.numpy()[0][free]

    def loss(qv, dqv, Fv):
        qn, dqn = DiffStructStep(C, dt).forward(wa(qv), wa(dqv), wa(Fv)); wp.synchronize()
        return float(np.sum(wq_ * qn.numpy()[0]) + np.sum(wd_ * dqn.numpy()[0]))

    def fd(idx_into):  # central diff wrt one input vector
        g = np.zeros(len(free)); base = [q0.copy(), dq0.copy(), F0.copy()]
        for ii, j in enumerate(free):
            vp = [a.copy() for a in base]; vp[idx_into][j] += eps
            vm = [a.copy() for a in base]; vm[idx_into][j] -= eps
            g[ii] = (loss(*vp) - loss(*vm)) / (2 * eps)
        return g

    # d/ddq path is LINEAR (a is dq-independent): adj must equal wd + dt*wq EXACTLY.
    # (FD here is polluted by the non-deterministic GPU CG noise in the a-path, so we
    #  check the adjoint against the analytic value instead.)
    gdq_analytic = (wd_ + dt * wq_)[free]
    rel_dq = np.max(np.abs(gdq - gdq_analytic)) / (np.max(np.abs(gdq_analytic)) + 1e-30)
    # d/dq (through K_t) and d/dF (through M-solve) vs FD; FD precision is limited by
    # the non-deterministic CG (~tol) in the loss recomputation -> 1e-3 tolerance.
    rel_q = np.max(np.abs(gq - fd(0))) / (np.max(np.abs(fd(0))) + 1e-30)
    rel_F = np.max(np.abs(gF - fd(2))) / (np.max(np.abs(fd(2))) + 1e-30)
    ok = rel_q < 1e-3 and rel_dq < 1e-10 and rel_F < 1e-3
    print(f"DiffStructStep one-step adjoint (loss = wq.q+ + wd.dq+):")
    print(f"  d/ddq vs ANALYTIC (linear path): rel={rel_dq:.2e}  (exact)")
    print(f"  d/dq  vs FD (through K_t)       : rel={rel_q:.2e}  (FD/CG-noise-limited)")
    print(f"  d/dF  vs FD (through M-solve)   : rel={rel_F:.2e}")
    print(f"  -> {'PASS' if ok else 'FAIL'}: differentiable structural step works "
          f"(adjoint method, pure Warp; chains to the rollout)")
    return ok


if __name__ == "__main__":
    wp.init()
    print(cfg.summary())
    raise SystemExit(0 if verify() else 1)
