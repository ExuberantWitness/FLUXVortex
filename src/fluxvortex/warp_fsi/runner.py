"""GPU time-loop orchestration for the batched FSI solver.

Phase 4c: structural time loop (repeated block-reduced Newmark steps with state
carry-over). Aero coupling (UVLM solve, load transfer, wake) is layered on top in
Phase 4d. The heavy per-step work (K assembly, force assembly, Jacobi-PCG) is all
on-device; the host loop only launches kernels and advances the pulse scalar.
"""
from __future__ import annotations
import numpy as np
import warp as wp
from . import config
from .kernels_ancf import ANCFConstants, assemble_kmem_blocks, assemble_internal_force_sep
from .batched_solver import gpu_newmark_step

DTYPE = config.DTYPE


def gpu_structural_trajectory(C: ANCFConstants, q0, dq0, pulse_shape, profile, dt,
                              n_steps, alpha_v=0.5, c_damp=2.0, cg_tol=1e-12,
                              tip_dof=None, device=None):
    """Run n_steps block-reduced Newmark steps with force F_const = profile(t)·pulse_shape
    (no aero). q0/dq0 (B, ndof) device arrays; pulse_shape (ndof,) host;
    profile(t_seconds)->scalar. Returns (q, dq, tip_history (n_steps, B) if tip_dof)."""
    device = device or config.DEVICE
    NP = config.NP_DTYPE
    B, ndof = q0.shape
    q = wp.clone(q0); dq = wp.clone(dq0)
    pshape_np = np.broadcast_to(pulse_shape, (B, ndof)).astype(NP)
    Fvel0 = wp.zeros((B, ndof), dtype=DTYPE, device=device)
    tip = np.zeros((n_steps, B)) if tip_dof is not None else None

    def recompute_bend(q_p1):
        return assemble_internal_force_sep(q_p1, C, device)[1]

    for step in range(n_steps):
        t = (step + 1) * dt          # end-of-step time (MATLAB convention)
        s = float(profile(t))
        Fc = wp.array((pshape_np * s).copy(), dtype=DTYPE, device=device)  # F_const = s·pulse_shape
        Kblk = assemble_kmem_blocks(q, C, device)
        Qmem, Qbend = assemble_internal_force_sep(q, C, device)
        q, dq = gpu_newmark_step(q, dq, Kblk, C.Me, C.edofs, C.free, ndof,
                                 Fc, Qmem, Qbend, Fvel0, recompute_bend, None,
                                 alpha_v=alpha_v, c_damp=c_damp, dt=dt,
                                 cg_tol=cg_tol, device=device)
        if tip is not None:
            wp.synchronize()
            tip[step] = q.numpy()[:, tip_dof]
    return q, dq, tip
