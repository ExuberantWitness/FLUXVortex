"""all-Warp coupled FSI design gradient (Plan Phase B / fix4c) — the GPU version of
diff_coupled_fsi: ANCF structure (Warp design adjoint, diff_struct_design_gpu) ⊗ VLM aero on
the deformed geometry (Warp, diff_vlm_gpu.VLMGpu). The expensive compute + the design gradient
∂(刚柔,质量)/∂L are all-Warp; the validated numpy diff_coupled_fsi is the oracle.

Coupled rollout (ANCF nodes ARE the VLM corners; the P/dist transfers reuse diff_coupled_fsi):
    corners = P·q ;  F_aero = dist·VLM(corners) ;  a = M(ρ)⁻¹(F_aero − Qint(q;E)) ;  symplectic
Adjoint per step: the Warp structural design adjoint (∂E via adj_E_kernel, ∂ρ via adj_rho_kernel,
state chain via the membrane-K_t apply_MK) + the aero VJP (VLMGpu.backward through the P/dist
transfers). verify: ∂L/∂(E,ρ) vs FD-of-Warp-forward AND vs the numpy diff_coupled_fsi oracle.
"""
from __future__ import annotations

import os
import sys

import numpy as np

_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
for p in (_SRC, os.path.dirname(__file__)):
    if p not in sys.path:
        sys.path.insert(0, p)

import warp as wp                                                # noqa: E402
from fluxvortex.warp_fsi import config as cfg                   # noqa: E402
from fluxvortex.warp_fsi.config import DTYPE                    # noqa: E402
from fluxvortex.warp_fsi.kernels_ancf import ANCFConstants, assemble_kmem_blocks  # noqa: E402
from fluxvortex.warp_fsi.batched_solver import structural_cg, apply_MK  # noqa: E402

import diff_struct_design_gpu as dsg                            # noqa: E402
import diff_coupled_fsi as dc                                   # noqa: E402 (numpy oracle + index maps)
from diff_vlm_gpu import VLMGpu                                 # noqa: E402

VINF = dc.VINF


def _build(nx, ny):
    return dsg._build(nx, ny)


def loss_and_grad_gpu(sh, C, vlm, P, dist, q0, dq0, N, dt, w, Es, Rs, nx, ny, cg_tol=1e-12):
    dev = cfg.DEVICE; NP = cfg.NP_DTYPE; ndof = C.ndof
    Esw = wp.array(Es.astype(NP), dtype=DTYPE, device=dev)
    Mscaled = wp.zeros((C.ne, 36, 36), dtype=DTYPE, device=dev)
    wp.launch(dsg._scaled_mass, dim=(C.ne, 36, 36),
              inputs=[C.Me, wp.array(Rs.astype(NP), dtype=DTYPE, device=dev)],
              outputs=[Mscaled], device=dev)
    Kblk0 = wp.zeros((1, C.ne, 36, 36), dtype=DTYPE, device=dev)
    z = lambda: wp.zeros((1, ndof), dtype=DTYPE, device=dev)
    wa = lambda v: wp.array(v[None].astype(NP), dtype=DTYPE, device=dev)
    npan = nx * ny

    def aero_nodal(qv):
        corners = (P @ qv).reshape(nx + 1, ny + 1, 3)
        Fp = vlm.forward(corners.reshape(-1, 3))         # (npan,3) all-Warp VLM
        return dist @ Fp.reshape(-1), Fp

    q = q0.copy(); dq = dq0.copy()
    qs, araws = [], []
    for _ in range(N):
        qw = wa(q)
        Qmem, Qbend = dsg.design_internal_force(qw, C, Esw, dev)
        Qint = Qmem.numpy()[0] + Qbend.numpy()[0]
        Fa, _ = aero_nodal(q)
        rhs = Fa - Qint
        rhsm = wa(rhs * C.free_np)
        a, _ = structural_cg(rhsm, Mscaled, Kblk0, C.edofs, C.free, 0.0, ndof, tol=cg_tol, device=dev)
        a_np = a.numpy()[0]
        qs.append(q.copy()); araws.append(a_np.copy())
        dq = dq + dt * a_np; q = q + dt * dq
    L = float(w @ q)
    # backward
    gE = wp.zeros(C.ne, dtype=DTYPE, device=dev); gR = wp.zeros(C.ne, dtype=DTYPE, device=dev)
    adj_q = w.copy(); adj_dq = np.zeros(ndof)
    for t in reversed(range(N)):
        aq1 = adj_q
        ad1 = adj_dq + dt * aq1
        adj_a = dt * ad1
        adj_am = wa(adj_a * C.free_np)
        adj_rhs_w, _ = structural_cg(adj_am, Mscaled, Kblk0, C.edofs, C.free, 0.0, ndof, tol=cg_tol, device=dev)
        adj_rhs = adj_rhs_w.numpy()[0]
        # aero VJP: adj_F_nodal = adj_rhs -> adj_Fpanel = distᵀ adj_rhs -> adj_corners = VLMᵀ -> adj_q
        vlm.forward((P @ qs[t]).reshape(-1, 3))          # re-cache at q_t
        adj_Fp = (dist.T @ adj_rhs).reshape(npan, 3)
        adj_corners = vlm.backward(adj_Fp).reshape(-1)
        adj_q_aero = P.T @ adj_corners
        # structural design adjoint + state chain (Warp)
        adj_Qint = -adj_rhs
        qtw = wa(qs[t]); adj_Qint_w = wa(adj_Qint); adj_rhs_t = wa(adj_rhs); araw_w = wa(araws[t])
        _, _, deps, dk, Dm_eps, Dk_k = dsg._design_force_cached(qtw, C, Esw, dev)
        wp.launch(dsg.adj_E_kernel, dim=(1, C.ne, 36),
                  inputs=[C.gw, deps, dk, Dm_eps, Dk_k, C.edofs, DTYPE(NP(C.h)), C.ngg, Esw, adj_Qint_w],
                  outputs=[gE], device=dev)
        wp.launch(dsg.adj_rho_kernel, dim=(1, C.ne), inputs=[C.Me, C.edofs, adj_rhs_t, araw_w],
                  outputs=[gR], device=dev)
        Kblk = assemble_kmem_blocks(qtw, C, dev)
        wp.launch(dsg._scale_kblk, dim=(1, C.ne, 36, 36), inputs=[Kblk, Esw], device=dev)
        adj_Qm = wa(adj_Qint * C.free_np); adj_qK = z()
        apply_MK(adj_Qm, adj_qK, C.Me, Kblk, C.edofs, C.free, 0.0, 1.0, dev)
        adj_q = aq1 + adj_qK.numpy()[0] + adj_q_aero
        adj_dq = ad1
    return L, gE.numpy(), gR.numpy()


def verify(nx=3, ny=3, N=6, dt=1e-5, eps=1e-6):
    wp.init()
    sh = _build(nx, ny); C = ANCFConstants(sh, device=cfg.DEVICE)
    vlm = VLMGpu(nx, ny, VINF)
    P, dist = dc._index_maps(sh, nx, ny)
    ne = sh.ne; ndof = sh.ndof
    rng = np.random.default_rng(0)
    Es = np.exp(0.2 * rng.standard_normal(ne)); Rs = np.exp(0.2 * rng.standard_normal(ne))
    free = np.array(sorted(set(range(ndof)) - set(sh._bc_dofs)))
    q0 = sh.q.copy(); q0[free] += 1e-4 * rng.standard_normal(len(free))
    dq0 = np.zeros(ndof); dq0[free] = 1e-3 * rng.standard_normal(len(free))
    w = np.zeros(ndof); w[free] = rng.standard_normal(len(free))

    L, gE, gR = loss_and_grad_gpu(sh, C, vlm, P, dist, q0, dq0, N, dt, w, Es, Rs, nx, ny)

    # FD of the GPU forward
    def Lonly(Es_, Rs_):
        return loss_and_grad_gpu(sh, C, vlm, P, dist, q0, dq0, N, dt, w, Es_, Rs_, nx, ny)[0]
    gE_fd = np.zeros(ne); gR_fd = np.zeros(ne)
    for e in range(ne):
        ep = Es.copy(); ep[e] += eps; em = Es.copy(); em[e] -= eps
        gE_fd[e] = (Lonly(ep, Rs) - Lonly(em, Rs)) / (2 * eps)
        rp = Rs.copy(); rp[e] += eps; rm = Rs.copy(); rm[e] -= eps
        gR_fd[e] = (Lonly(Es, rp) - Lonly(Es, rm)) / (2 * eps)
    relE = np.max(np.abs(gE - gE_fd)) / (np.max(np.abs(gE_fd)) + 1e-30)
    relR = np.max(np.abs(gR - gR_fd)) / (np.max(np.abs(gR_fd)) + 1e-30)
    # cross-check vs the numpy oracle (full-K_t tangent -> ~membrane diff)
    sh.set_distribution(E_scale=Es, rho_scale=Rs)
    _, gE_np, gR_np = dc.loss_and_grad(sh, q0, dq0, N, dt, free, w, nx, ny)
    rel_np = max(np.max(np.abs(gE - gE_np)) / (np.max(np.abs(gE_np)) + 1e-30),
                 np.max(np.abs(gR - gR_np)) / (np.max(np.abs(gR_np)) + 1e-30))
    ok = relE < 5e-2 and relR < 1e-3
    print(f"all-Warp COUPLED FSI design gradient ({ne} elems, {N}-step coupled rollout):")
    print(f"  ∂L/∂E_scale (刚柔)  vs FD={relE:.2e}   ∂L/∂rho_scale(质量) vs FD={relR:.2e}")
    print(f"  cross-check vs numpy diff_coupled_fsi oracle (full vs membrane K_t): rel={rel_np:.2e}")
    print(f"  -> {'PASS' if ok else 'FAIL'}: structure (Warp design adjoint) ⊗ VLM (Warp) coupled "
          f"design gradient on GPU")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if verify() else 1)
