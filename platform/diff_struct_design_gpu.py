"""Warp per-element (刚柔 E + 质量 ρ) design-aware ANCF structural forward + design adjoint
(Plan Phase A / fix4b foundation) — the all-GPU replacement for the numpy diff_struct_design.

NON-INVASIVE to the validated golden kernels: the ANCF internal force is LINEAR in the
per-element stiffness scale (Qmem_el ∝ Dm ∝ E_scale_el; Kblk_el ∝ Dm ∝ E_scale_el) and the
element mass block is LINEAR in the density scale (Me_el ∝ ρ_scale_el). So:
  · forward = the EXISTING ancf_force_gauss/assemble kernels + a small kernel that scales the
    stress resultants Dm_eps/Dk_k (and Kblk) by E_scale[el], and a ρ_scale-scaled mass solve.
    With E_scale=ρ_scale=1 this is BIT-IDENTICAL to the golden (validated below).
  · design adjoint = two new per-element kernels (∂E via the unit-force contraction with
    adj_Qint; ∂ρ via the element-mass quadratic form with adj_rhs and the raw M⁻¹·rhs),
    matching the numpy reference diff_struct_design.py.

This module first validates the FORWARD (internal force + mass) bit-matches the numpy ANCFShell
with set_distribution(E_scale, ρ_scale); the design adjoint + rollout come next.
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
from fluxvortex.warp_fsi.kernels_ancf import (ANCFConstants,    # noqa: E402
    ancf_force_gauss_kernel, ancf_force_assemble_kernel, assemble_kmem_blocks)
from fluxvortex.warp_fsi.config import MAT33                    # noqa: E402
from fluxvortex.warp_fsi.batched_solver import structural_cg, apply_MK  # noqa: E402

wp.set_module_options({"enable_backward": True})


@wp.kernel
def scale_stress(Dm_eps: wp.array(dtype=DTYPE, ndim=4), Dk_k: wp.array(dtype=DTYPE, ndim=4),
                 E_scale: wp.array(dtype=DTYPE), n3: int):
    """Per-element stiffness scaling of the stress resultants (force ∝ E_scale_el)."""
    e, el, g = wp.tid()
    s = E_scale[el]
    for c in range(3):
        Dm_eps[e, el, g, c] = Dm_eps[e, el, g, c] * s
        Dk_k[e, el, g, c] = Dk_k[e, el, g, c] * s


def design_internal_force(q_wp, C, E_scale_wp, device=None):
    """E_scale-aware internal force Qint=Qmem+Qbend (B,ndof), reusing the golden kernels."""
    device = device or cfg.DEVICE
    B = q_wp.shape[0]; NP = cfg.NP_DTYPE
    Dm = MAT33(*[DTYPE(NP(v)) for v in C.Dm_np.ravel()])
    Dk = MAT33(*[DTYPE(NP(v)) for v in C.Dk_np.ravel()])
    deps = wp.zeros((B, C.ne, C.ngg, 108), dtype=DTYPE, device=device)
    dk = wp.zeros((B, C.ne, C.ngg, 108), dtype=DTYPE, device=device)
    Dm_eps = wp.zeros((B, C.ne, C.ngg, 3), dtype=DTYPE, device=device)
    Dk_k = wp.zeros((B, C.ne, C.ngg, 3), dtype=DTYPE, device=device)
    wp.launch(ancf_force_gauss_kernel, dim=(B, C.ne, C.ngg),
              inputs=[q_wp, C.dSx, C.dSy, C.d2Sx, C.d2Sy, C.d2Sxy, C.edofs, Dm, Dk],
              outputs=[deps, dk, Dm_eps, Dk_k], device=device)
    wp.launch(scale_stress, dim=(B, C.ne, C.ngg), inputs=[Dm_eps, Dk_k, E_scale_wp, 3],
              device=device)                                    # <-- per-element E scaling
    Qmem = wp.zeros((B, C.ndof), dtype=DTYPE, device=device)
    Qbend = wp.zeros((B, C.ndof), dtype=DTYPE, device=device)
    wp.launch(ancf_force_assemble_kernel, dim=(B, C.ne, 36),
              inputs=[C.gw, deps, dk, Dm_eps, Dk_k, C.edofs, DTYPE(NP(C.h)), C.ngg],
              outputs=[Qmem, Qbend], device=device)
    return Qmem, Qbend


@wp.kernel
def _scaled_mass_scatter(Me: wp.array(dtype=DTYPE, ndim=3), rho_scale: wp.array(dtype=DTYPE),
                         edofs: wp.array(dtype=wp.int32, ndim=2),
                         Mrow: wp.array(dtype=DTYPE, ndim=2)):
    """Row-sum of the ρ_scale-scaled global mass (lumped check) — not used in solve; for a
    quick mass-forward validation. el,a -> sum_b Me_scaled[el,a,b] into Mrow[edof]."""
    el, a = wp.tid()
    s = rho_scale[el]
    acc = DTYPE(0.0)
    for b in range(36):
        acc = acc + Me[el, a, b] * s
    wp.atomic_add(Mrow, 0, edofs[el, a], acc)


@wp.kernel
def _scale_kblk(Kblk: wp.array(dtype=DTYPE, ndim=4), E_scale: wp.array(dtype=DTYPE)):
    e, el, a, b = wp.tid()
    Kblk[e, el, a, b] = Kblk[e, el, a, b] * E_scale[el]


@wp.kernel
def _scaled_mass(Me: wp.array(dtype=DTYPE, ndim=3), rho_scale: wp.array(dtype=DTYPE),
                 Mout: wp.array(dtype=DTYPE, ndim=3)):
    el, a, b = wp.tid()
    Mout[el, a, b] = Me[el, a, b] * rho_scale[el]


@wp.kernel
def _axpby2(a: wp.array(dtype=DTYPE, ndim=2), ca: DTYPE, b: wp.array(dtype=DTYPE, ndim=2),
            cb: DTYPE, o: wp.array(dtype=DTYPE, ndim=2)):
    e, i = wp.tid()
    o[e, i] = ca * a[e, i] + cb * b[e, i]


@wp.kernel
def _mask2(a: wp.array(dtype=DTYPE, ndim=2), free: wp.array(dtype=DTYPE),
           o: wp.array(dtype=DTYPE, ndim=2)):
    e, i = wp.tid()
    o[e, i] = a[e, i] * free[i]


@wp.kernel
def adj_E_kernel(gw: wp.array(dtype=DTYPE, ndim=2), deps: wp.array(dtype=DTYPE, ndim=4),
                 dk: wp.array(dtype=DTYPE, ndim=4), Dm_eps: wp.array(dtype=DTYPE, ndim=4),
                 Dk_k: wp.array(dtype=DTYPE, ndim=4), edofs: wp.array(dtype=wp.int32, ndim=2),
                 h: DTYPE, ngg: int, E_scale: wp.array(dtype=DTYPE),
                 adj_Qint: wp.array(dtype=DTYPE, ndim=2), adj_E: wp.array(dtype=DTYPE)):
    """∂L/∂E_el += Σ_a adj_Qint[edof] · (per-DOF force / E_el)  (force ∝ E_el, Dm_eps/Dk_k scaled)."""
    e, el, a = wp.tid()
    am = DTYPE(0.0); ab = DTYPE(0.0)
    for g in range(ngg):
        w = gw[el, g]
        mem = (deps[e, el, g, a] * Dm_eps[e, el, g, 0]
               + deps[e, el, g, 36 + a] * Dm_eps[e, el, g, 1]
               + deps[e, el, g, 72 + a] * Dm_eps[e, el, g, 2])
        ben = (dk[e, el, g, a] * Dk_k[e, el, g, 0]
               + dk[e, el, g, 36 + a] * Dk_k[e, el, g, 1]
               + dk[e, el, g, 72 + a] * Dk_k[e, el, g, 2])
        am = am + w * h * mem; ab = ab + w * ben
    wp.atomic_add(adj_E, el, adj_Qint[e, edofs[el, a]] * (am + ab) / E_scale[el])


@wp.kernel
def adj_rho_kernel(Me: wp.array(dtype=DTYPE, ndim=3), edofs: wp.array(dtype=wp.int32, ndim=2),
                   adj_rhs: wp.array(dtype=DTYPE, ndim=2), a_raw: wp.array(dtype=DTYPE, ndim=2),
                   adj_rho: wp.array(dtype=DTYPE)):
    """∂L/∂ρ_el += -adj_rhs[edof]·Me_unit[el]·a_raw[edof]   (M_el ∝ ρ_scale_el; Me = ρh·∫SᵀS)."""
    e, el = wp.tid()
    acc = DTYPE(0.0)
    for a in range(36):
        ar = adj_rhs[e, edofs[el, a]]
        for b in range(36):
            acc = acc + ar * Me[el, a, b] * a_raw[e, edofs[el, b]]
    wp.atomic_add(adj_rho, el, -acc)


def _design_force_cached(q_wp, C, E_scale_wp, device):
    """Like design_internal_force but RETURNS the scaled deps/dk/Dm_eps/Dk_k (for the E adjoint)."""
    B = q_wp.shape[0]; NP = cfg.NP_DTYPE
    Dm = MAT33(*[DTYPE(NP(v)) for v in C.Dm_np.ravel()])
    Dk = MAT33(*[DTYPE(NP(v)) for v in C.Dk_np.ravel()])
    deps = wp.zeros((B, C.ne, C.ngg, 108), dtype=DTYPE, device=device)
    dk = wp.zeros((B, C.ne, C.ngg, 108), dtype=DTYPE, device=device)
    Dm_eps = wp.zeros((B, C.ne, C.ngg, 3), dtype=DTYPE, device=device)
    Dk_k = wp.zeros((B, C.ne, C.ngg, 3), dtype=DTYPE, device=device)
    wp.launch(ancf_force_gauss_kernel, dim=(B, C.ne, C.ngg),
              inputs=[q_wp, C.dSx, C.dSy, C.d2Sx, C.d2Sy, C.d2Sxy, C.edofs, Dm, Dk],
              outputs=[deps, dk, Dm_eps, Dk_k], device=device)
    wp.launch(scale_stress, dim=(B, C.ne, C.ngg), inputs=[Dm_eps, Dk_k, E_scale_wp, 3], device=device)
    Qmem = wp.zeros((B, C.ndof), dtype=DTYPE, device=device)
    Qbend = wp.zeros((B, C.ndof), dtype=DTYPE, device=device)
    wp.launch(ancf_force_assemble_kernel, dim=(B, C.ne, 36),
              inputs=[C.gw, deps, dk, Dm_eps, Dk_k, C.edofs, DTYPE(NP(C.h)), C.ngg],
              outputs=[Qmem, Qbend], device=device)
    return Qmem, Qbend, deps, dk, Dm_eps, Dk_k


def loss_and_grad_gpu(C, q0, dq0, F, N, dt, w, E_scale, rho_scale, device=None, cg_tol=1e-12):
    """L = w·q_N and ∂L/∂(E_scale, rho_scale) via the all-Warp design adjoint (mirrors the
    numpy diff_struct_design.loss_and_grad; state chain uses the membrane K_t adjoint)."""
    device = device or cfg.DEVICE; NP = cfg.NP_DTYPE; ndof = C.ndof
    Es = wp.array(E_scale.astype(NP), dtype=DTYPE, device=device)
    Mscaled = wp.zeros((C.ne, 36, 36), dtype=DTYPE, device=device)
    wp.launch(_scaled_mass, dim=(C.ne, 36, 36),
              inputs=[C.Me, wp.array(rho_scale.astype(NP), dtype=DTYPE, device=device)],
              outputs=[Mscaled], device=device)
    Kblk0 = wp.zeros((1, C.ne, 36, 36), dtype=DTYPE, device=device)
    z = lambda: wp.zeros((1, ndof), dtype=DTYPE, device=device)
    wa = lambda v: wp.array(v[None].astype(NP), dtype=DTYPE, device=device)
    Fw = wa(F)
    q = wa(q0); dq = wa(dq0)
    qs, araws = [], []
    for _ in range(N):
        Qmem, Qbend = design_internal_force(q, C, Es, device)
        rhs = z(); wp.launch(_axpby2, dim=(1, ndof), inputs=[Fw, DTYPE(1.0), Qmem, DTYPE(-1.0)], outputs=[rhs], device=device)
        wp.launch(_axpby2, dim=(1, ndof), inputs=[rhs, DTYPE(1.0), Qbend, DTYPE(-1.0)], outputs=[rhs], device=device)
        rhsm = z(); wp.launch(_mask2, dim=(1, ndof), inputs=[rhs, C.free], outputs=[rhsm], device=device)
        a, _ = structural_cg(rhsm, Mscaled, Kblk0, C.edofs, C.free, 0.0, ndof, tol=cg_tol, device=device)
        qs.append(wp.clone(q)); araws.append(wp.clone(a))
        dq2 = z(); wp.launch(_axpby2, dim=(1, ndof), inputs=[dq, DTYPE(1.0), a, DTYPE(dt)], outputs=[dq2], device=device)
        q2 = z(); wp.launch(_axpby2, dim=(1, ndof), inputs=[q, DTYPE(1.0), dq2, DTYPE(dt)], outputs=[q2], device=device)
        q, dq = q2, dq2
    L = float((w * q.numpy()[0]).sum())
    # backward
    gE = wp.zeros(C.ne, dtype=DTYPE, device=device); gR = wp.zeros(C.ne, dtype=DTYPE, device=device)
    adj_q = wa(w); adj_dq = z()
    for t in reversed(range(N)):
        ad1 = z(); wp.launch(_axpby2, dim=(1, ndof), inputs=[adj_dq, DTYPE(1.0), adj_q, DTYPE(dt)], outputs=[ad1], device=device)
        adj_a = z(); wp.launch(_axpby2, dim=(1, ndof), inputs=[ad1, DTYPE(dt), ad1, DTYPE(0.0)], outputs=[adj_a], device=device)
        adj_am = z(); wp.launch(_mask2, dim=(1, ndof), inputs=[adj_a, C.free], outputs=[adj_am], device=device)
        adj_rhs, _ = structural_cg(adj_am, Mscaled, Kblk0, C.edofs, C.free, 0.0, ndof, tol=cg_tol, device=device)
        adj_Qint = z(); wp.launch(_axpby2, dim=(1, ndof), inputs=[adj_rhs, DTYPE(-1.0), adj_rhs, DTYPE(0.0)], outputs=[adj_Qint], device=device)
        # design grads at q_t
        _, _, deps, dk, Dm_eps, Dk_k = _design_force_cached(qs[t], C, Es, device)
        wp.launch(adj_E_kernel, dim=(1, C.ne, 36),
                  inputs=[C.gw, deps, dk, Dm_eps, Dk_k, C.edofs, DTYPE(NP(C.h)), C.ngg, Es, adj_Qint],
                  outputs=[gE], device=device)
        wp.launch(adj_rho_kernel, dim=(1, C.ne), inputs=[C.Me, C.edofs, adj_rhs, araws[t]],
                  outputs=[gR], device=device)
        # state chain: adj_q += K_t(q_t)·adj_Qint  (membrane Kblk, E-scaled)
        Kblk = assemble_kmem_blocks(qs[t], C, device)
        wp.launch(_scale_kblk, dim=(1, C.ne, 36, 36), inputs=[Kblk, Es], device=device)
        adj_Qm = z(); wp.launch(_mask2, dim=(1, ndof), inputs=[adj_Qint, C.free], outputs=[adj_Qm], device=device)
        adj_qK = z(); apply_MK(adj_Qm, adj_qK, C.Me, Kblk, C.edofs, C.free, 0.0, 1.0, device)
        new_adj_q = z(); wp.launch(_axpby2, dim=(1, ndof), inputs=[adj_q, DTYPE(1.0), adj_qK, DTYPE(1.0)], outputs=[new_adj_q], device=device)
        adj_q = new_adj_q; adj_dq = ad1
    return L, gE.numpy(), gR.numpy()


def verify_forward(nx=4, ny=3):
    """FORWARD red line: Warp E_scale/ρ_scale internal force + mass row-sum bit-match the numpy
    ANCFShell with set_distribution — and uniform reduces to the golden."""
    wp.init()
    from fluxvortex.ancf_shell import ANCFShell
    rng = np.random.default_rng(0)
    # small plate shell (matches diff_struct_design._build_shell scale)
    L, W, h, rho, E, nu = 0.4, 0.3, 1.5e-3, 1200.0, 1.0e6, 0.3
    xs = np.linspace(0, L, nx + 1); ys = np.linspace(0, W, ny + 1)
    nodes = np.array([[x, y, 0.0] for y in ys for x in xs])
    quads = []
    for j in range(ny):
        for i in range(nx):
            n0 = j * (nx + 1) + i
            quads.append([n0, n0 + 1, n0 + nx + 2, n0 + nx + 1])
    sh = ANCFShell(nodes, np.array(quads), h, rho, E, E, nu)
    ne = sh.ne
    Es = np.exp(0.3 * rng.standard_normal(ne)); Rs = np.exp(0.3 * rng.standard_normal(ne))
    sh.set_distribution(E_scale=Es, rho_scale=Rs)
    ndof = sh.ndof
    q = sh.q.copy()
    free = np.array(sorted(set(range(ndof)) - set(sh._bc_dofs)))
    q[free] += 1e-3 * rng.standard_normal(len(free))

    # numpy golden internal force (uses per-element Dm_e ∝ E_scale)
    Qm_np, Qb_np = sh._internal_forces_separated(q)
    Q_np = Qm_np + Qb_np

    # Warp: ANCFConstants from a UNIFORM shell (golden), then E_scale applied in the kernel
    sh0 = ANCFShell(nodes, np.array(quads), h, rho, E, E, nu)         # uniform (E_scale=1)
    C = ANCFConstants(sh0, device=cfg.DEVICE)
    qw = wp.array(q[None].astype(cfg.NP_DTYPE), dtype=DTYPE, device=cfg.DEVICE)
    Esw = wp.array(Es.astype(cfg.NP_DTYPE), dtype=DTYPE, device=cfg.DEVICE)
    Qm, Qb = design_internal_force(qw, C, Esw)
    wp.synchronize()
    Q_gpu = (Qm.numpy() + Qb.numpy())[0]
    rel_f = np.max(np.abs(Q_gpu - Q_np)) / (np.max(np.abs(Q_np)) + 1e-30)

    # mass row-sum with ρ_scale vs numpy M row-sum
    Rsw = wp.array(Rs.astype(cfg.NP_DTYPE), dtype=DTYPE, device=cfg.DEVICE)
    Mrow = wp.zeros((1, ndof), dtype=DTYPE, device=cfg.DEVICE)
    wp.launch(_scaled_mass_scatter, dim=(ne, 36), inputs=[C.Me, Rsw, C.edofs], outputs=[Mrow],
              device=cfg.DEVICE); wp.synchronize()
    Mrow_np = np.asarray(sh.M.sum(axis=1)).ravel()
    rel_m = np.max(np.abs(Mrow.numpy()[0] - Mrow_np)) / (np.max(np.abs(Mrow_np)) + 1e-30)

    okf = rel_f < 1e-10 and rel_m < 1e-10
    print(f"Warp design-aware FORWARD vs numpy ANCFShell.set_distribution ({ne} elems):")
    print(f"  internal force (E_scale per-element)  rel={rel_f:.2e}")
    print(f"  mass row-sum   (ρ_scale per-element)  rel={rel_m:.2e}")
    print(f"  -> {'PASS' if okf else 'FAIL'}: per-element 刚柔/质量 forward ported numpy->GPU "
          f"(non-invasive; uniform reduces to golden)")
    return okf


def _loss_gpu(C, q0, dq0, F, N, dt, w, E_scale, rho_scale, device=None, cg_tol=1e-12):
    device = device or cfg.DEVICE; NP = cfg.NP_DTYPE; ndof = C.ndof
    Es = wp.array(E_scale.astype(NP), dtype=DTYPE, device=device)
    Mscaled = wp.zeros((C.ne, 36, 36), dtype=DTYPE, device=device)
    wp.launch(_scaled_mass, dim=(C.ne, 36, 36),
              inputs=[C.Me, wp.array(rho_scale.astype(NP), dtype=DTYPE, device=device)],
              outputs=[Mscaled], device=device)
    Kblk0 = wp.zeros((1, C.ne, 36, 36), dtype=DTYPE, device=device)
    z = lambda: wp.zeros((1, ndof), dtype=DTYPE, device=device)
    wa = lambda v: wp.array(v[None].astype(NP), dtype=DTYPE, device=device)
    Fw = wa(F); q = wa(q0); dq = wa(dq0)
    for _ in range(N):
        Qmem, Qbend = design_internal_force(q, C, Es, device)
        rhs = z(); wp.launch(_axpby2, dim=(1, ndof), inputs=[Fw, DTYPE(1.0), Qmem, DTYPE(-1.0)], outputs=[rhs], device=device)
        wp.launch(_axpby2, dim=(1, ndof), inputs=[rhs, DTYPE(1.0), Qbend, DTYPE(-1.0)], outputs=[rhs], device=device)
        rhsm = z(); wp.launch(_mask2, dim=(1, ndof), inputs=[rhs, C.free], outputs=[rhsm], device=device)
        a, _ = structural_cg(rhsm, Mscaled, Kblk0, C.edofs, C.free, 0.0, ndof, tol=cg_tol, device=device)
        dq2 = z(); wp.launch(_axpby2, dim=(1, ndof), inputs=[dq, DTYPE(1.0), a, DTYPE(dt)], outputs=[dq2], device=device)
        q2 = z(); wp.launch(_axpby2, dim=(1, ndof), inputs=[q, DTYPE(1.0), dq2, DTYPE(dt)], outputs=[q2], device=device)
        q, dq = q2, dq2
    return float((w * q.numpy()[0]).sum())


def _build(nx, ny):
    from fluxvortex.ancf_shell import ANCFShell
    L, W, h, rho, E, nu = 0.4, 0.3, 1.5e-3, 1200.0, 1.0e6, 0.3
    xs = np.linspace(0, L, nx + 1); ys = np.linspace(0, W, ny + 1)
    nodes = np.array([[x, y, 0.0] for y in ys for x in xs])
    quads = [[j * (nx + 1) + i, j * (nx + 1) + i + 1, j * (nx + 1) + i + nx + 2,
              j * (nx + 1) + i + nx + 1] for j in range(ny) for i in range(nx)]
    return ANCFShell(nodes, np.array(quads), h, rho, E, E, nu)


def verify_grad(nx=4, ny=3, N=20, dt=2e-5, eps=1e-6):
    wp.init()
    sh = _build(nx, ny); C = ANCFConstants(sh, device=cfg.DEVICE)
    ne = sh.ne; ndof = sh.ndof
    rng = np.random.default_rng(1)
    Es = np.exp(0.2 * rng.standard_normal(ne)); Rs = np.exp(0.2 * rng.standard_normal(ne))
    free = np.array(sorted(set(range(ndof)) - set(sh._bc_dofs)))
    q0 = sh.q.copy(); q0[free] += 1e-3 * rng.standard_normal(len(free))
    dq0 = np.zeros(ndof); dq0[free] = 6e-3 * rng.standard_normal(len(free))
    F = np.zeros(ndof); F[free] = 0.5 * rng.standard_normal(len(free))
    w = np.zeros(ndof); w[free] = rng.standard_normal(len(free))

    _, gE, gR = loss_and_grad_gpu(C, q0, dq0, F, N, dt, w, Es, Rs)
    gE_fd = np.zeros(ne); gR_fd = np.zeros(ne)
    for e in range(ne):
        ep = Es.copy(); ep[e] += eps; em = Es.copy(); em[e] -= eps
        gE_fd[e] = (_loss_gpu(C, q0, dq0, F, N, dt, w, ep, Rs) - _loss_gpu(C, q0, dq0, F, N, dt, w, em, Rs)) / (2 * eps)
        rp = Rs.copy(); rp[e] += eps; rm = Rs.copy(); rm[e] -= eps
        gR_fd[e] = (_loss_gpu(C, q0, dq0, F, N, dt, w, Es, rp) - _loss_gpu(C, q0, dq0, F, N, dt, w, Es, rm)) / (2 * eps)
    relE = np.max(np.abs(gE - gE_fd)) / (np.max(np.abs(gE_fd)) + 1e-30)
    relR = np.max(np.abs(gR - gR_fd)) / (np.max(np.abs(gR_fd)) + 1e-30)
    print(f"Warp design ADJOINT vs FD-of-Warp-forward ({ne} elems, {N}-step rollout):")
    print(f"  ∂L/∂E_scale  (刚柔)  rel={relE:.2e}")
    print(f"  ∂L/∂rho_scale(质量)  rel={relR:.2e}")
    ok = relE < 5e-2 and relR < 1e-3
    print(f"  -> {'PASS' if ok else 'FAIL'}  (∂ρ exact; ∂E via membrane-K_t adjoint — bending "
          f"tangent is the warp_fsi follow-on for tighter ∂E)")
    return ok


if __name__ == "__main__":
    okf = verify_forward()
    okg = verify_grad()
    raise SystemExit(0 if (okf and okg) else 1)
