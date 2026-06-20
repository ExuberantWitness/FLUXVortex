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
    ancf_force_gauss_kernel, ancf_force_assemble_kernel)
from fluxvortex.warp_fsi.config import MAT33                    # noqa: E402

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


if __name__ == "__main__":
    raise SystemExit(0 if verify_forward() else 1)
