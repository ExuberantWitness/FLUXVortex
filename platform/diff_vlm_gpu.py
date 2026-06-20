"""GPU/Warp differentiable VLM (fix 4) — the all-Warp replacement for the numpy/complex-step
diff_vlm. geometry → AIC → γ=AIC⁻¹rhs → Kutta-Joukowski force, fully on GPU, differentiable.

Composition (the diff_step pattern): Warp autodiff (wp.Tape) for the geometry→AIC and the
KJ-force kernels, with the MANUAL DiffDenseSolve VJP (diff_solve.py) for the linear solve in
the middle (Warp can't auto-diff the LU). One backward gives ∂(forces)/∂(corners) on GPU.

verify(): forward matches the numpy VLM (diff_vlm); ∂(total force)/∂(corners) from this GPU
adjoint matches the numpy EXACT complex-step Jacobian (diff_vlm.jac_complex_step). This is the
numpy→GPU port of the aero kernel — the foundation for the all-GPU coupled FSI.
"""
from __future__ import annotations

import os
import sys

import numpy as np

_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import warp as wp                                                # noqa: E402
from fluxvortex.warp_fsi.config import DTYPE                    # noqa: E402
from diff_solve import DiffDenseSolve                           # noqa: E402
import diff_vlm                                                 # noqa: E402 (numpy golden)

wp.set_module_options({"enable_backward": True})
RHO = 1.225
_LEG = 50.0
V3 = wp.vec3d


@wp.func
def vseg(P: V3, A: V3, B: V3) -> V3:
    r1 = P - A; r2 = P - B; r0 = B - A
    cr = wp.cross(r1, r2)
    cr2 = wp.dot(cr, cr) + wp.float64(1.0e-12)
    n1 = wp.sqrt(wp.dot(r1, r1) + wp.float64(1.0e-24))
    n2 = wp.sqrt(wp.dot(r2, r2) + wp.float64(1.0e-24))
    k = (wp.float64(1.0) / (wp.float64(4.0) * wp.float64(3.141592653589793))) \
        * wp.dot(r0, r1 / n1 - r2 / n2) / cr2
    return k * cr


@wp.func
def horseshoe(P: V3, A: V3, B: V3, edir: V3) -> V3:
    Aw = A + wp.float64(_LEG) * edir
    Bw = B + wp.float64(_LEG) * edir
    return vseg(P, Bw, B) + vseg(P, B, A) + vseg(P, A, Aw)


@wp.kernel
def panel_geom(corners: wp.array(dtype=V3), nx: int, ny: int,
               qa: wp.array(dtype=V3), qb: wp.array(dtype=V3),
               col: wp.array(dtype=V3), nrm: wp.array(dtype=V3)):
    p = wp.tid()
    pi = p // ny; pj = p % ny
    c00 = corners[pi * (ny + 1) + pj]
    c10 = corners[(pi + 1) * (ny + 1) + pj]
    c01 = corners[pi * (ny + 1) + pj + 1]
    c11 = corners[(pi + 1) * (ny + 1) + pj + 1]
    qa[p] = wp.float64(0.75) * c00 + wp.float64(0.25) * c10
    qb[p] = wp.float64(0.75) * c01 + wp.float64(0.25) * c11
    col[p] = wp.float64(0.5) * (wp.float64(0.25) * c00 + wp.float64(0.75) * c10
                                + wp.float64(0.25) * c01 + wp.float64(0.75) * c11)
    d1 = c11 - c00; d2 = c01 - c10
    n = wp.cross(d1, d2)
    nrm[p] = n / wp.sqrt(wp.dot(n, n) + wp.float64(1.0e-24))


@wp.kernel
def aic_kernel(qa: wp.array(dtype=V3), qb: wp.array(dtype=V3), col: wp.array(dtype=V3),
               nrm: wp.array(dtype=V3), edir: V3, AIC: wp.array(dtype=DTYPE, ndim=3)):
    i, j = wp.tid()
    v = horseshoe(col[i], qa[j], qb[j], edir)
    AIC[0, i, j] = wp.dot(v, nrm[i])


@wp.kernel
def rhs_kernel(nrm: wp.array(dtype=V3), Vinf: V3, rhs: wp.array(dtype=DTYPE, ndim=2)):
    i = wp.tid()
    rhs[0, i] = -wp.dot(Vinf, nrm[i])


@wp.kernel
def kj_kernel(qa: wp.array(dtype=V3), qb: wp.array(dtype=V3),
              gamma: wp.array(dtype=DTYPE, ndim=2), Vinf: V3, rho: wp.float64,
              F: wp.array(dtype=V3)):
    p = wp.tid()
    lb = qb[p] - qa[p]
    F[p] = rho * gamma[0, p] * wp.cross(Vinf, lb)


class VLMGpu:
    """All-Warp differentiable VLM. forward(corners)->F; grad via tape + DiffDenseSolve VJP."""

    def __init__(self, nx, ny, Vinf, device="cuda"):
        self.nx, self.ny, self.dev = nx, ny, device
        self.npan = nx * ny
        self.Vinf = V3(*[float(v) for v in Vinf])
        ev = np.asarray(Vinf, float); ev = ev / (np.linalg.norm(ev) + 1e-24)
        self.edir = V3(*ev.tolist())
        self.dds = DiffDenseSolve(device)

    def _alloc(self, rg):
        z = lambda: wp.zeros(self.npan, dtype=V3, device=self.dev, requires_grad=rg)
        return z(), z(), z(), z()

    def forward(self, corners_np):
        """Cache the geometry→AIC→γ→KJ chain; return per-panel force (npan,3). Pair with
        backward(adj_Fpanel) for the VJP adj_corners=Jᵀ·adj_F used by the coupled rollout."""
        nc = (self.nx + 1) * (self.ny + 1)
        self._corners = wp.array(np.asarray(corners_np, np.float64).reshape(nc, 3), dtype=V3,
                                 device=self.dev, requires_grad=True)
        qa, qb, col, nrm = self._alloc(True)
        self._AIC = wp.zeros((1, self.npan, self.npan), dtype=DTYPE, device=self.dev, requires_grad=True)
        self._rhs = wp.zeros((1, self.npan), dtype=DTYPE, device=self.dev, requires_grad=True)
        self._t1 = wp.Tape()
        with self._t1:
            wp.launch(panel_geom, dim=self.npan, inputs=[self._corners, self.nx, self.ny],
                      outputs=[qa, qb, col, nrm], device=self.dev)
            wp.launch(aic_kernel, dim=(self.npan, self.npan),
                      inputs=[qa, qb, col, nrm, self.edir], outputs=[self._AIC], device=self.dev)
            wp.launch(rhs_kernel, dim=self.npan, inputs=[nrm, self.Vinf], outputs=[self._rhs],
                      device=self.dev)
        self._gamma = self.dds.forward(self._AIC, self._rhs)
        self._gamma.requires_grad = True
        self._F = wp.zeros(self.npan, dtype=V3, device=self.dev, requires_grad=True)
        qa2, qb2, col2, nrm2 = self._alloc(True)
        self._t2 = wp.Tape()
        with self._t2:
            wp.launch(panel_geom, dim=self.npan, inputs=[self._corners, self.nx, self.ny],
                      outputs=[qa2, qb2, col2, nrm2], device=self.dev)
            wp.launch(kj_kernel, dim=self.npan, inputs=[qa2, qb2, self._gamma, self.Vinf,
                      wp.float64(RHO)], outputs=[self._F], device=self.dev)
        return self._F.numpy()

    def backward(self, adj_Fpanel_np):
        """VJP: adj_corners (ncorner,3) = Jᵀ·adj_Fpanel for the cached forward."""
        self._corners.grad.zero_(); self._gamma.grad.zero_()
        self._AIC.grad.zero_(); self._rhs.grad.zero_()
        self._F.grad = wp.array(np.ascontiguousarray(adj_Fpanel_np, np.float64).reshape(self.npan, 3),
                                dtype=V3, device=self.dev)
        self._t2.backward()
        adj_A, adj_b = self.dds.backward(self._gamma.grad)
        self._AIC.grad = adj_A; self._rhs.grad = adj_b
        self._t1.backward()
        g = self._corners.grad.numpy().copy()
        self._t1.zero(); self._t2.zero()
        return g

    def forward_grad(self, corners_np):
        """Total force (np,3) + Jacobian ∂(total F)/∂corners (3, ncorner*3) — via forward/backward."""
        nc = (self.nx + 1) * (self.ny + 1)
        Fp = self.forward(corners_np)
        Ftot = Fp.sum(0)
        Jac = np.zeros((3, nc * 3))
        for comp in range(3):
            seed = np.zeros((self.npan, 3)); seed[:, comp] = 1.0
            Jac[comp] = self.backward(seed).reshape(-1)
        return Ftot, Jac


def verify(nx=3, ny=4):
    wp.init()
    Vinf = np.array([10.0, 0.0, 0.0])
    corners = diff_vlm._flat_wing(nx, ny)
    # forward vs numpy golden
    _, Ftot_np = diff_vlm.vlm_forces(corners, nx, ny, Vinf)
    vg = VLMGpu(nx, ny, Vinf)
    Ftot_gpu, Jac_gpu = vg.forward_grad(corners)
    rel_f = np.max(np.abs(Ftot_gpu - Ftot_np)) / (np.max(np.abs(Ftot_np)) + 1e-30)
    # gradient vs numpy EXACT complex-step
    Jac_cs = diff_vlm.jac_complex_step(corners, nx, ny, Vinf)
    rel_j = np.max(np.abs(Jac_gpu - Jac_cs)) / (np.max(np.abs(Jac_cs)) + 1e-30)
    ok = rel_f < 1e-10 and rel_j < 1e-6
    print(f"GPU/Warp differentiable VLM ({nx}x{ny} panels), all-Warp fp64:")
    print(f"  forward total force vs numpy golden: rel={rel_f:.2e}")
    print(f"  ∂(total force)/∂(corners) GPU adjoint vs numpy complex-step: rel={rel_j:.2e}")
    print(f"  -> {'PASS' if ok else 'FAIL'}: aero kernel ported numpy->GPU, differentiable "
          f"(Warp tape + DiffDenseSolve VJP)")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if verify() else 1)
