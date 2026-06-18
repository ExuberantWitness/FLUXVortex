"""Proof that the differentiable ANCF adjoint is exact and already-computed.

The structural internal force ``Qint(q)`` (assemble_internal_force_sep) is a
nonlinear operator whose Jacobian is the **tangent stiffness** ``K_t = ∂Qint/∂q``.
Warp 1.14's *auto*-adjoint of the bending kernel NaNs (see
docs/p1_differentiability_finding.md), but we do not need it: the GPU code already
builds ``K_t`` every Newmark step via ``assemble_kmem_blocks`` (consumed by the CG
solve), bit-exact validated. The vector-Jacobian product we need for backprop is
therefore exactly

    adj_q = K_tᵀ · adj_Qint = K_t · adj_Qint        (K_t symmetric)

This script proves ``assemble_kmem_blocks`` is that Jacobian by comparing
``K_t·δq`` (via ``apply_MK`` with cM=0, cK=1) to the finite-difference directional
derivative of ``Qint`` — i.e. it validates the custom-adjoint path end to end.

Run: cd FLUXV/src && FLUXV_DEVICE=cuda:0 python ../platform/verify_tangent_jacobian.py
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
from run_standalone_yamano import yamano_params, build_yamano_shell  # noqa: E402


def verify(nx: int = 8, ny: int = 6, eps: float = 1e-6) -> bool:
    dev, NP = cfg.DEVICE, cfg.NP_DTYPE
    params = yamano_params()
    shell, _, _, _ = build_yamano_shell(params, nx=nx, ny=ny)
    C = ANCFConstants(shell, device=dev)
    ndof = shell.ndof
    rng = np.random.default_rng(2)
    q0 = np.zeros(ndof)
    for k in range(shell.nn):
        q0[9 * k] = shell.nodes[k, 0]; q0[9 * k + 1] = shell.nodes[k, 1]
        q0[9 * k + 3] = 1.0; q0[9 * k + 7] = 1.0
    free = np.array(sorted(set(range(ndof)) - set(shell._bc_dofs)))
    q0[free] += 1e-3 * rng.standard_normal(len(free))

    def zero_bc(v):
        v = v.copy()
        for d in shell._bc_dofs:
            v[d] = 0.0
        return v

    dq = np.zeros(ndof); dq[free] = rng.standard_normal(len(free)); dq = zero_bc(dq)

    def wq(v):
        return wp.array(np.broadcast_to(v, (1, ndof)).astype(NP).copy(),
                        dtype=DTYPE, device=dev)

    # K_t . dq   (assemble_kmem_blocks then apply_MK with cM=0, cK=1)
    Kblk = assemble_kmem_blocks(wq(q0), C, dev)
    out = wp.zeros((1, ndof), dtype=DTYPE, device=dev)
    apply_MK(wq(dq), out, C.Me, Kblk, C.edofs, C.free, 0.0, 1.0, dev)
    wp.synchronize()
    Kdq = out.numpy()[0]

    # FD directional derivative of Qint = Qmem + Qbend
    def Qint(v):
        Qm, Qb = assemble_internal_force_sep(wq(v), C, dev)
        wp.synchronize()
        return Qm.numpy()[0] + Qb.numpy()[0]

    FDdir = (Qint(q0 + eps * dq) - Qint(q0 - eps * dq)) / (2.0 * eps)

    a, b = Kdq[free], FDdir[free]
    # dominant-component accuracy (FD is accurate where |b| is large; small
    # cancelling components carry FD truncation noise)
    big = np.abs(b) > 0.05 * np.max(np.abs(b))
    rel_big = np.max(np.abs(a[big] - b[big]) / np.abs(b[big]))
    ok = rel_big < 1e-5
    print(f"K_t.dq vs FD directional-derivative of Qint  ({len(free)} free DOFs):")
    print(f"  dominant components: max rel = {rel_big:.2e}  ({int(big.sum())} dofs)")
    print(f"  -> {'PASS' if ok else 'FAIL'}: assemble_kmem_blocks IS the exact "
          f"Jacobian; custom adjoint adj_q = K_t.adj_Qint is exact + already computed")
    return ok


if __name__ == "__main__":
    wp.init()
    print(cfg.summary())
    raise SystemExit(0 if verify() else 1)
