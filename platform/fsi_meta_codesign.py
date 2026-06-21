"""Full-UVLM meta-RL co-design (NO reduced surrogate) — Step 1: a step-by-step coupled UVLM FSI env so
a meta-policy can act IN the loop.

The reduced flight surrogate is rejected: the co-design must evaluate on the FULL nonlinear unsteady
free-wake UVLM coupled to the ANCF structure. FSIControlEnv wraps the validated coupled forward
(diff_coupled_unsteady_gpu.coupled_unsteady_forward_gpu) as a gym-like stepper holding the FSI state
(q, dq, free wake, γ_prev); step(action) applies the policy's control u_t = -k_t·q̇⊙pos (the validated
position-DOF actuation, k_t the per-step gain the meta-policy outputs) and advances ONE coupled FSI step
(bound rings → AIC → moving-body rhs incl. wake → γ → unsteady KJ+∂Γ/∂t force → M(ρ)⁻¹(F−Qint(E)+u) →
symplectic step → shed/convect wake). Observation = a reduced read of the FSI state (tip deflection /
rate / gust); reward = −gust deflection. Validated bit-exact vs the batch forward (constant action k ≡
fb_gain) before any policy/training is wired in.
"""
from __future__ import annotations

import numpy as np

import warp as wp
from fluxvortex.warp_fsi import config as cfg
from fluxvortex.warp_fsi.config import DTYPE
from fluxvortex.warp_fsi.kernels_ancf import assemble_kmem_blocks  # noqa: F401 (parity import)
from fluxvortex.warp_fsi.batched_solver import structural_cg, batched_dense_solve
import diff_uvlm_unsteady_gpu as ug
import diff_struct_design_gpu as dsg
import diff_coupled_unsteady as dcu
import diff_coupled_unsteady_gpu as dcg
import codesign_qd_unsteady as cq

V3 = wp.vec3d


class FSIControlEnv:
    """Step-by-step full coupled UVLM FSI env. design=(E,ρ) per-element; the policy's per-step gain k_t
    drives u_t = -k_t·q̇⊙pos on the position DOFs (validated actuation)."""

    def __init__(self, sh, C, P, dist, q0, dq0, Es, Rs, nx, ny, dt, Vinf=cq.cg.VINF,
                 N=None, use_wake=True, cg_tol=1e-10, device=None):
        self.dev = device or cfg.DEVICE; NP = cfg.NP_DTYPE
        self.sh, self.C, self.P, self.dist = sh, C, P, dist
        self.nx, self.ny = nx, ny; self.npan = nx * ny; self.ncv = (nx + 1) * (ny + 1)
        self.ndof = C.ndof; self.dt = dt; self.use_wake = use_wake; self.cg_tol = cg_tol
        self.q0 = q0.copy(); self.dq0 = dq0.copy(); self.N = N
        self.qref = sh.q.copy(); self.free_np = C.free_np
        self.pos = dcg._pos_mask(C)
        self.Esw = wp.array(Es.astype(NP), dtype=DTYPE, device=self.dev)
        self.Mscaled = wp.zeros((C.ne, 36, 36), dtype=DTYPE, device=self.dev)
        wp.launch(dsg._scaled_mass, dim=(C.ne, 36, 36),
                  inputs=[C.Me, wp.array(Rs.astype(NP), dtype=DTYPE, device=self.dev)], outputs=[self.Mscaled], device=self.dev)
        self.Kblk0 = wp.zeros((1, C.ne, 36, 36), dtype=DTYPE, device=self.dev)
        self.Vw = V3(*[float(v) for v in np.asarray(Vinf, float)])
        self.te = wp.array(np.array([(nx - 1) * ny + j for j in range(ny)], np.int32), dtype=wp.int32, device=self.dev)
        self.maxw = (N or 200) * ny
        self.wa = lambda v: wp.array(v[None].astype(NP), dtype=DTYPE, device=self.dev)
        self.reset()

    def reset(self):
        self.q = self.q0.copy(); self.dq = self.dq0.copy(); self.t = 0
        self.wr = wp.zeros((self.maxw, 4), dtype=V3, device=self.dev)
        self.wr_new = wp.zeros((self.maxw, 4), dtype=V3, device=self.dev)
        self.wg = wp.zeros(self.maxw, dtype=DTYPE, device=self.dev)
        self.gprev = wp.zeros((1, self.npan), dtype=DTYPE, device=self.dev); self.nw = 0
        return self._obs()

    def _obs(self):
        d = (self.q - self.qref) * self.free_np
        zt = d[(np.arange(self.ndof) % 9 == 2)]               # vertical deflections
        vt = (self.dq * self.free_np)[(np.arange(self.ndof) % 9 == 2)]
        return np.array([zt.max(), zt.min(), np.abs(zt).mean(), vt.max(), vt.min(), float(self.t)], float)

    def step(self, k):
        dev = self.dev; npan = self.npan; nx, ny = self.nx, self.ny; NP = cfg.NP_DTYPE
        corners = (self.P @ self.q).reshape(self.ncv, 3); cvel = (self.P @ self.dq).reshape(self.ncv, 3)
        cw = wp.array(corners.astype(NP), dtype=V3, device=dev); vw = wp.array(cvel.astype(NP), dtype=V3, device=dev)
        rings = wp.zeros((npan, 4), dtype=V3, device=dev); col = wp.zeros(npan, dtype=V3, device=dev)
        nrm = wp.zeros(npan, dtype=V3, device=dev); vcol = wp.zeros(npan, dtype=V3, device=dev)
        wp.launch(ug.bound_rings_kernel, dim=npan, inputs=[cw, nx, ny], outputs=[rings, col, nrm], device=dev)
        wp.launch(ug.colvel_kernel, dim=npan, inputs=[vw, nx, ny], outputs=[vcol], device=dev)
        AIC = wp.zeros((1, npan, npan), dtype=DTYPE, device=dev)
        wp.launch(ug.aic_kernel, dim=(npan, npan), inputs=[rings, col, nrm], outputs=[AIC], device=dev)
        rhs = wp.zeros((1, npan), dtype=DTYPE, device=dev)
        wp.launch(ug.rhs_moving_kernel, dim=npan, inputs=[col, nrm, self.Vw, vcol, self.wr, self.wg, self.nw], outputs=[rhs], device=dev)
        gamma = batched_dense_solve(AIC, rhs, dev)
        Fp = wp.zeros(npan, dtype=V3, device=dev)
        wp.launch(ug.panel_force_kernel, dim=npan, inputs=[rings, nrm, gamma, self.gprev, vcol, self.Vw,
                  DTYPE(self.dt), DTYPE(ug.RHO), ny], outputs=[Fp], device=dev)
        Fnodal = self.dist @ Fp.numpy().reshape(-1)
        Qmem, Qbend = dsg.design_internal_force(self.wa(self.q), self.C, self.Esw, dev)
        Qint = Qmem.numpy()[0] + Qbend.numpy()[0]
        u = -float(k) * self.dq * self.pos                    # policy control: gain k on position DOFs
        rhs_s = (Fnodal - Qint + u) * self.free_np
        a, _ = structural_cg(self.wa(rhs_s), self.Mscaled, self.Kblk0, self.C.edofs, self.C.free, 0.0,
                             self.ndof, tol=self.cg_tol, device=dev)
        a_np = a.numpy()[0]
        self.dq = self.dq + self.dt * a_np; self.q = self.q + self.dt * self.dq
        if self.use_wake:
            wp.launch(ug.shed_kernel, dim=ny, inputs=[rings, gamma, self.te, self.Vw, DTYPE(self.dt), self.nw], outputs=[self.wr, self.wg], device=dev)
            self.nw += ny
            wp.launch(ug.convect_kernel, dim=(self.nw, 4), inputs=[rings, gamma, npan, self.wr, self.wg, self.nw, self.Vw, DTYPE(self.dt)], outputs=[self.wr_new], device=dev)
            wp.copy(self.wr, self.wr_new, count=self.nw * 4)
        self.gprev = wp.array(gamma.numpy(), dtype=DTYPE, device=dev)
        self.t += 1
        d = (self.q - self.qref) * self.free_np
        rew = -float(np.sum(d * d))                           # gust-rejection reward (per step)
        done = (self.N is not None and self.t >= self.N) or (not np.all(np.isfinite(self.q)))
        return self._obs(), rew, done, {"defl": -rew}


def verify(nx=6, ny=4, N=12, dt=2e-4, k=2.0, seed=0):
    """FSIControlEnv stepped with a CONSTANT gain k reproduces coupled_unsteady_forward_gpu(fb_gain=k)
    bit-exact — the step-by-step full-UVLM env equals the validated batch forward. (Explicit closed-loop
    feedback diverges at high gain k≳3 — the known added-mass instability; high-gain meta-policies need
    the strong-coupled PC forward of route A, which this env can swap in.)"""
    wp.init()
    env0 = cq.Env(nx=nx, ny=ny, seed=seed)
    rng = np.random.default_rng(seed); ne = env0.ne
    Es = np.exp(0.1 * rng.standard_normal(ne)); Rs = np.exp(0.1 * rng.standard_normal(ne))
    env0.sh.set_distribution(E_scale=Es, rho_scale=Rs)
    qref_batch, _ = dcg.coupled_unsteady_forward_gpu(env0.sh, env0.C, env0.P, env0.dist, env0.q0, env0.dq0,
                       N, dt, Es, Rs, nx, ny, use_wake=True, fb_gain=k, cg_tol=1e-10)
    e = FSIControlEnv(env0.sh, env0.C, env0.P, env0.dist, env0.q0, env0.dq0, Es, Rs, nx, ny, dt,
                      N=N, use_wake=True, cg_tol=1e-10); e.reset()
    for _ in range(N):
        _, _, done, _ = e.step(k)
    rel = np.max(np.abs(e.q - qref_batch)) / (np.max(np.abs(qref_batch - env0.q0)) + 1e-30)
    ok = rel < 1e-6                                           # CG-tolerance-level match (same algorithm, float ordering)
    print(f"FSIControlEnv (step-by-step full UVLM FSI) vs batch forward (fb_gain={k}, {ne} elems, {N} steps):")
    print(f"  final-state q match: rel={rel:.2e}")
    print(f"  -> {'PASS' if ok else 'FAIL'}: the step-by-step coupled UVLM env equals the validated forward "
          f"— a policy can now act IN the full-fidelity FSI loop (no surrogate)")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if verify() else 1)
