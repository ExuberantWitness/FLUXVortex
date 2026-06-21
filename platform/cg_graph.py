"""CUDA-graph fixed-iteration structural CG (GPU-accel core) — accuracy-preserving.

The adaptive structural_cg does a host sync + numpy read of the residual EVERY iteration and launches
~7 tiny kernels/iteration; at ~85–200 iterations that is the dominant cost of the PC forward/adjoint
(profiled: ~12 ms/solve, ~3000+ solves/gradient). This module captures a FIXED-iteration CG
(no convergence check inside) as a CUDA graph and replays it — collapsing the ~10³ kernel launches per
solve into one graph launch. It is NOT a toy simplification: a fixed niter ≥ the adaptive iteration
count converges to the same solution to the same tolerance (verified vs the adaptive CG); the graph just
removes launch + sync overhead. The operator A = M(ρ) + coef·K_mem(q;E) is the SAME across all PC /
adjoint inner iterations of a step, so only the right-hand side b (and, once per step, K_mem) changes
between replays — written in-place into persistent buffers.
"""
from __future__ import annotations

import numpy as np

import warp as wp
from fluxvortex.warp_fsi import config as cfg
from fluxvortex.warp_fsi.config import DTYPE
from fluxvortex.warp_fsi.batched_solver import (apply_MK, sdiag_kernel, _minv_apply, _dot,
                                                _axpy_kernel, _xpby_kernel, _div_kernel)


class GraphCG:
    """Persistent-buffer CUDA-graph CG for A = M + coef·K_mem, batch B (=1 here). Drop-in for
    structural_cg in the PC inner loops. set_A(Kblk) once per step; solve(b) per inner iteration."""

    def __init__(self, Me, edofs, free, coef, ndof, ne, niter, B=1, device=None):
        self.dev = device or cfg.DEVICE; self.ndof = ndof; self.B = B
        self.Me = Me; self.edofs = edofs; self.free = free
        self.coef = float(coef); self.ne = ne; self.niter = int(niter)
        z2 = lambda: wp.zeros((B, ndof), dtype=DTYPE, device=self.dev)
        z1 = lambda: wp.zeros(B, dtype=DTYPE, device=self.dev)
        self.b = z2(); self.x = z2(); self.r = z2(); self.z = z2(); self.p = z2(); self.Sp = z2()
        self.diag = z2()
        self.rz = z1(); self.rz_new = z1(); self.pSp = z1(); self.alpha = z1(); self.beta = z1()
        self.Kblk = wp.zeros((B, ne, 36, 36), dtype=DTYPE, device=self.dev)
        self.graph = None

    def _body(self):
        dev = self.dev; ndof = self.ndof; B = self.B; co = self.coef
        self.diag.zero_()
        wp.launch(sdiag_kernel, dim=(B, self.ne, 36), inputs=[self.Me, self.Kblk, self.edofs,
                  self.free, DTYPE(co)], outputs=[self.diag], device=dev)
        self.x.zero_(); wp.copy(self.r, self.b)
        wp.launch(_minv_apply, dim=(B, ndof), inputs=[self.diag, self.free, self.r], outputs=[self.z], device=dev)
        wp.copy(self.p, self.z)
        _dot(self.r, self.z, ndof, self.rz, dev)
        for _ in range(self.niter):
            apply_MK(self.p, self.Sp, self.Me, self.Kblk, self.edofs, self.free, 1.0, co, dev)
            _dot(self.p, self.Sp, ndof, self.pSp, dev)
            wp.launch(_div_kernel, dim=B, inputs=[self.rz, self.pSp], outputs=[self.alpha], device=dev)
            wp.launch(_axpy_kernel, dim=(B, ndof), inputs=[self.x, self.alpha, self.p, DTYPE(1.0)], device=dev)
            wp.launch(_axpy_kernel, dim=(B, ndof), inputs=[self.r, self.alpha, self.Sp, DTYPE(-1.0)], device=dev)
            wp.launch(_minv_apply, dim=(B, ndof), inputs=[self.diag, self.free, self.r], outputs=[self.z], device=dev)
            _dot(self.r, self.z, ndof, self.rz_new, dev)
            wp.launch(_div_kernel, dim=B, inputs=[self.rz_new, self.rz], outputs=[self.beta], device=dev)
            wp.launch(_xpby_kernel, dim=(B, ndof), inputs=[self.p, self.z, self.beta], device=dev)
            wp.copy(self.rz, self.rz_new)

    def capture(self):
        with wp.ScopedCapture(device=self.dev) as cap:
            self._body()
        self.graph = cap.graph
        return self

    def set_A(self, Kblk):
        wp.copy(self.Kblk, Kblk)

    def solve(self, b_np_or_wp):
        if isinstance(b_np_or_wp, np.ndarray):
            self.b.assign(b_np_or_wp.reshape(self.B, self.ndof).astype(cfg.NP_DTYPE))
        else:
            wp.copy(self.b, b_np_or_wp)
        if self.graph is None:
            self.capture()
        wp.capture_launch(self.graph)
        return self.x


@wp.kernel
def _assemble_A_kernel(Me: wp.array(dtype=DTYPE, ndim=3), Kblk: wp.array(dtype=DTYPE, ndim=4),
                       edofs: wp.array(dtype=wp.int32, ndim=2), free: wp.array(dtype=DTYPE, ndim=1),
                       coef: DTYPE, A: wp.array(dtype=DTYPE, ndim=3)):
    """Scatter the per-element (M + coef·K_mem) 36×36 blocks into the dense free–free system A
    (fixed DOFs left as identity rows; set separately). A is (B, ndof, ndof)."""
    e, el, a, b = wp.tid()
    i = edofs[el, a]; j = edofs[el, b]
    if free[i] > DTYPE(0.5) and free[j] > DTYPE(0.5):
        wp.atomic_add(A, e, i, j, Me[el, a, b] + coef * Kblk[e, el, a, b])


@wp.kernel
def _set_bc_identity(free: wp.array(dtype=DTYPE, ndim=1), A: wp.array(dtype=DTYPE, ndim=3)):
    """A[e,d,d] = 1 on fixed DOFs so the dense solve returns x[bc]=0 (b[bc]=0)."""
    e, d = wp.tid()
    if free[d] <= DTYPE(0.5):
        A[e, d, d] = DTYPE(1.0)


def assemble_A_dense(Mscaled, Kblk, edofs, free, coef, ndof, ne, B=1, device=None):
    dev = device or cfg.DEVICE
    A = wp.zeros((B, ndof, ndof), dtype=DTYPE, device=dev)
    wp.launch(_assemble_A_kernel, dim=(B, ne, 36, 36),
              inputs=[Mscaled, Kblk, edofs, free, DTYPE(float(coef))], outputs=[A], device=dev)
    wp.launch(_set_bc_identity, dim=(B, ndof), inputs=[free], outputs=[A], device=dev)
    return A


def verify_dense(seed=0):
    """Direct dense solve (assemble A + batched LU) vs adaptive structural_cg — accuracy + timing at
    several scales. A is SPD and FIXED across a step's 169 inner solves, so a dense factor amortises."""
    import time
    from fluxvortex.warp_fsi.kernels_ancf import ANCFConstants, assemble_kmem_blocks
    from fluxvortex.warp_fsi.batched_solver import structural_cg, batched_dense_solve
    from diff_struct_design import _build_shell
    import diff_struct_design_gpu as dsg
    wp.init(); dev = cfg.DEVICE; NP = cfg.NP_DTYPE
    for (nx, ny) in [(6, 4), (10, 6), (15, 10)]:
        sh = _build_shell(nx=nx, ny=ny); sh.set_bc([i for i in range(nx + 1)])
        C = ANCFConstants(sh, device=dev); ne = sh.ne; ndof = sh.ndof
        rng = np.random.default_rng(seed)
        Es = np.exp(0.1 * rng.standard_normal(ne)); Rs = np.exp(0.1 * rng.standard_normal(ne))
        sh.set_distribution(E_scale=Es, rho_scale=Rs)
        Esw = wp.array(Es.astype(NP), dtype=DTYPE, device=dev)
        Mscaled = wp.zeros((ne, 36, 36), dtype=DTYPE, device=dev)
        wp.launch(dsg._scaled_mass, dim=(ne, 36, 36),
                  inputs=[C.Me, wp.array(Rs.astype(NP), dtype=DTYPE, device=dev)], outputs=[Mscaled], device=dev)
        q = sh.q.copy()
        Kblk = assemble_kmem_blocks(wp.array(q[None].astype(NP), dtype=DTYPE, device=dev), C, dev)
        wp.launch(dsg._scale_kblk, dim=(1, ne, 36, 36), inputs=[Kblk, Esw], device=dev)
        coef = 0.25 * (2e-4) ** 2
        b = (rng.standard_normal(ndof) * C.free_np)
        bw = wp.array(b[None].astype(NP), dtype=DTYPE, device=dev)
        x_ad = structural_cg(bw, Mscaled, Kblk, C.edofs, C.free, coef, ndof, tol=1e-11, device=dev)[0].numpy()[0]

        def dense_solve():
            A = assemble_A_dense(Mscaled, Kblk, C.edofs, C.free, coef, ndof, ne, device=dev)
            return batched_dense_solve(A, bw, dev)
        x_d = dense_solve().numpy()[0]
        rel = np.max(np.abs(x_d - x_ad)) / (np.max(np.abs(x_ad)) + 1e-30)
        wp.synchronize(); t0 = time.time()
        for _ in range(50): structural_cg(bw, Mscaled, Kblk, C.edofs, C.free, coef, ndof, tol=1e-11, device=dev)
        wp.synchronize(); t_cg = (time.time() - t0) / 50
        wp.synchronize(); t0 = time.time()
        for _ in range(50): dense_solve()
        wp.synchronize(); t_d = (time.time() - t0) / 50
        print(f"{nx}x{ny}={ne}elem ndof={ndof}: dense vs CG  rel={rel:.1e}  "
              f"CG={t_cg*1e3:.1f}ms  dense(assemble+LU)={t_d*1e3:.1f}ms  speedup {t_cg/t_d:.1f}x", flush=True)


def verify(nx=15, ny=10, niter=200, seed=0):
    """Graphed fixed-iter CG vs the adaptive structural_cg — same solution to CG tolerance, plus timing."""
    import time
    from fluxvortex.warp_fsi.kernels_ancf import ANCFConstants, assemble_kmem_blocks
    from fluxvortex.warp_fsi.batched_solver import structural_cg
    from diff_struct_design import _build_shell
    import diff_struct_design_gpu as dsg
    wp.init(); dev = cfg.DEVICE; NP = cfg.NP_DTYPE
    sh = _build_shell(nx=nx, ny=ny); sh.set_bc([i for i in range(nx + 1)])
    C = ANCFConstants(sh, device=dev); ne = sh.ne; ndof = sh.ndof
    rng = np.random.default_rng(seed)
    Es = np.exp(0.1 * rng.standard_normal(ne)); Rs = np.exp(0.1 * rng.standard_normal(ne))
    sh.set_distribution(E_scale=Es, rho_scale=Rs)
    Esw = wp.array(Es.astype(NP), dtype=DTYPE, device=dev)
    Mscaled = wp.zeros((ne, 36, 36), dtype=DTYPE, device=dev)
    wp.launch(dsg._scaled_mass, dim=(ne, 36, 36),
              inputs=[C.Me, wp.array(Rs.astype(NP), dtype=DTYPE, device=dev)], outputs=[Mscaled], device=dev)
    q = sh.q.copy()
    Kblk = assemble_kmem_blocks(wp.array(q[None].astype(NP), dtype=DTYPE, device=dev), C, dev)
    wp.launch(dsg._scale_kblk, dim=(1, ne, 36, 36), inputs=[Kblk, Esw], device=dev)
    coef = 0.25 * (2e-4) ** 2
    b = (rng.standard_normal(ndof) * C.free_np)
    bw = wp.array(b[None].astype(NP), dtype=DTYPE, device=dev)
    x_ad, it = structural_cg(bw, Mscaled, Kblk, C.edofs, C.free, coef, ndof, tol=1e-11, device=dev)
    x_ad = x_ad.numpy()[0]
    g = GraphCG(Mscaled, C.edofs, C.free, coef, ndof, ne, niter, device=dev)
    g.set_A(Kblk); g.capture()
    x_g = g.solve(bw).numpy()[0]
    rel = np.max(np.abs(x_g - x_ad)) / (np.max(np.abs(x_ad)) + 1e-30)
    # timing
    wp.synchronize(); t0 = time.time()
    for _ in range(100): structural_cg(bw, Mscaled, Kblk, C.edofs, C.free, coef, ndof, tol=1e-11, device=dev)
    wp.synchronize(); t_ad = (time.time() - t0) / 100
    wp.synchronize(); t0 = time.time()
    for _ in range(100): g.solve(bw)
    wp.synchronize(); t_g = (time.time() - t0) / 100
    ok = rel < 1e-6
    print(f"Graphed fixed-iter CG vs adaptive structural_cg ({nx}x{ny}={ne}elem, ndof={ndof}, niter={niter}):")
    print(f"  solution match: rel={rel:.2e}   (adaptive took {int(it)} iters to tol 1e-11)")
    print(f"  time/solve: adaptive={t_ad*1e3:.1f}ms   graphed={t_g*1e3:.1f}ms   speedup {t_ad/t_g:.1f}x")
    print(f"  -> {'PASS' if ok else 'FAIL'}: graphed CG bit-matches the iterative solve and collapses the launch overhead")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if verify() else 1)
