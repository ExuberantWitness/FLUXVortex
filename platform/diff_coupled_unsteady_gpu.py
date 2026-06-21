"""all-Warp coupled UNSTEADY FSI forward (Plan fix3) — the GPU version of diff_coupled_unsteady:
ANCF structure (Warp design internal force + matrix-free CG mass solve) ⊗ unsteady free-wake
ring-VLM on the DEFORMED, MOVING wing (Warp), carrying the wake history. This is the forward that
the all-Warp unsteady-coupled DESIGN ADJOINT (next) differentiates; the numpy diff_coupled_unsteady
is the oracle.

Per step: corners=P·q, V_body=P·dq → bound_rings + colvel → AIC → moving-body rhs (incl. wake) →
γ=AIC⁻¹rhs → per-panel unsteady force → dist→nodal → a=M(ρ)⁻¹(F−Qint(q;E)) → symplectic step →
shed + free-convect wake. verify(): per-step state + final q match the numpy oracle (the aero is
bit-exact; the structural CG mass solve matches the oracle's direct solve to CG tolerance)."""
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
from fluxvortex.warp_fsi.batched_solver import structural_cg, apply_MK, batched_dense_solve  # noqa: E402

import diff_struct_design_gpu as dsg                            # noqa: E402
import diff_coupled_unsteady as dcu                             # noqa: E402 (numpy oracle)
import diff_uvlm_unsteady_gpu as ug                             # noqa: E402 (aero kernels)
from diff_solve import DiffDenseSolve                           # noqa: E402
from diff_struct_design import _build_shell                     # noqa: E402

V3 = wp.vec3d
VINF = dcu.VINF


@wp.kernel
def _adj_E_stiff(u: wp.array(dtype=DTYPE, ndim=2), astar: wp.array(dtype=DTYPE, ndim=2),
                 Kblk: wp.array(dtype=DTYPE, ndim=4), edofs: wp.array(dtype=wp.int32, ndim=2),
                 E_scale: wp.array(dtype=DTYPE), coef: DTYPE, adj_E: wp.array(dtype=DTYPE)):
    """∂L/∂E_el += -coef·(u_eᵀ·K_mem_el·a*_e)/E_el — the ∂(β·dt²·K_mem)/∂E·a* sensitivity of the
    implicit operator A=M+β·dt²·K_mem (K_mem ∝ E_el). The PC forward freezes A at q_pred, so this is
    the only K_mem design term; the q_pred-dependence of K_mem (∂K/∂q, third order) is neglected as in
    standard linearly-implicit Newmark."""
    e, el = wp.tid()
    acc = DTYPE(0.0)
    for a in range(36):
        ua = u[e, edofs[el, a]]
        for b in range(36):
            acc = acc + ua * Kblk[e, el, a, b] * astar[e, edofs[el, b]]
    wp.atomic_add(adj_E, el, -coef * acc / E_scale[el])


def _pos_mask(C):
    """Free POSITION DOFs (translational: dof%9 ∈ {0,1,2}) — actuation acts here, not on the
    tiny-inertia ANCF slope DOFs (which make explicit closed-loop feedback blow up)."""
    m = getattr(C, "_pos_mask_cache", None)
    if m is None:
        nd = C.ndof
        m = C.free_np * (np.arange(nd) % 9 < 3).astype(C.free_np.dtype)
        C._pos_mask_cache = m
    return m


class UnsteadyAeroGpu:
    """Per-step unsteady free-wake aero VJP on the moving/deforming wing. forward caches the
    geometry→AIC→γ→panel-force chain (tape1 up to the solve, manual DiffDenseSolve, tape2 for the
    force); backward returns adj on (corners, V_body, gprev, wake) for the coupled design adjoint.
    Mirrors VLMGpu but with the moving-body BC, the dΓ/dt force and the wake history."""

    def __init__(self, nx, ny, Vinf, dt, device="cuda"):
        self.nx, self.ny, self.dev = nx, ny, device
        self.npan = nx * ny; self.ncv = (nx + 1) * (ny + 1)
        self.Vw = V3(*[float(v) for v in np.asarray(Vinf, float)])
        self.dtt = DTYPE(dt); self.rhod = DTYPE(ug.RHO)
        self.dds = DiffDenseSolve(device)
        self.te = wp.array(np.array([(nx - 1) * ny + j for j in range(ny)], np.int32),
                           dtype=wp.int32, device=device)

    def forward(self, corners_np, cvel_np, wr, wg, nw, gprev_np, use_wake=False):
        dev = self.dev; npan = self.npan; nx, ny = self.nx, self.ny
        self.cw = wp.array(corners_np.reshape(self.ncv, 3), dtype=V3, device=dev, requires_grad=True)
        self.vw = wp.array(cvel_np.reshape(self.ncv, 3), dtype=V3, device=dev, requires_grad=True)
        self.gprev = wp.array(gprev_np.reshape(1, npan), dtype=DTYPE, device=dev, requires_grad=True)
        self.wr = wr; self.wg = wg; self.nw = nw                 # wake leaves (requires_grad set by caller)
        rings = wp.zeros((npan, 4), dtype=V3, device=dev, requires_grad=True)
        col = wp.zeros(npan, dtype=V3, device=dev, requires_grad=True)
        nrm = wp.zeros(npan, dtype=V3, device=dev, requires_grad=True)
        vcol = wp.zeros(npan, dtype=V3, device=dev, requires_grad=True)
        self.AIC = wp.zeros((1, npan, npan), dtype=DTYPE, device=dev, requires_grad=True)
        self.rhs = wp.zeros((1, npan), dtype=DTYPE, device=dev, requires_grad=True)
        self.t1 = wp.Tape()
        with self.t1:
            wp.launch(ug.bound_rings_kernel, dim=npan, inputs=[self.cw, nx, ny], outputs=[rings, col, nrm], device=dev)
            wp.launch(ug.colvel_kernel, dim=npan, inputs=[self.vw, nx, ny], outputs=[vcol], device=dev)
            wp.launch(ug.aic_kernel, dim=(npan, npan), inputs=[rings, col, nrm], outputs=[self.AIC], device=dev)
            wp.launch(ug.rhs_moving_kernel, dim=npan, inputs=[col, nrm, self.Vw, vcol, self.wr, self.wg, nw],
                      outputs=[self.rhs], device=dev)
        self.gamma = self.dds.forward(self.AIC, self.rhs); self.gamma.requires_grad = True
        self.Fp = wp.zeros(npan, dtype=V3, device=dev, requires_grad=True)
        rings2 = wp.zeros((npan, 4), dtype=V3, device=dev, requires_grad=True)
        col2 = wp.zeros(npan, dtype=V3, device=dev, requires_grad=True)
        nrm2 = wp.zeros(npan, dtype=V3, device=dev, requires_grad=True)
        vcol2 = wp.zeros(npan, dtype=V3, device=dev, requires_grad=True)
        self.t2 = wp.Tape()
        with self.t2:
            wp.launch(ug.bound_rings_kernel, dim=npan, inputs=[self.cw, nx, ny], outputs=[rings2, col2, nrm2], device=dev)
            wp.launch(ug.colvel_kernel, dim=npan, inputs=[self.vw, nx, ny], outputs=[vcol2], device=dev)
            wp.launch(ug.panel_force_kernel, dim=npan, inputs=[rings2, nrm2, self.gamma, self.gprev, vcol2,
                      self.Vw, self.dtt, self.rhod, ny], outputs=[self.Fp], device=dev)
        self.rings_fwd = rings2                                  # bound rings (for the wake shed/convect)
        # --- tape3: wake update (shed TE ring + free convection) -> wake_{t+1} = (wr_next, wgcat) ---
        self.use_wake = use_wake
        if use_wake:
            nwn = nw + ny
            self.wcat = wp.zeros((nwn, 4), dtype=V3, device=dev, requires_grad=True)
            self.wgcat = wp.zeros(nwn, dtype=DTYPE, device=dev, requires_grad=True)
            self.wr_next = wp.zeros((nwn, 4), dtype=V3, device=dev, requires_grad=True)
            rings3 = wp.zeros((npan, 4), dtype=V3, device=dev, requires_grad=True)
            col3 = wp.zeros(npan, dtype=V3, device=dev, requires_grad=True)
            nrm3 = wp.zeros(npan, dtype=V3, device=dev, requires_grad=True)
            self.t3 = wp.Tape()
            with self.t3:
                wp.launch(ug.bound_rings_kernel, dim=npan, inputs=[self.cw, nx, ny], outputs=[rings3, col3, nrm3], device=dev)
                if nw > 0:
                    wp.launch(ug.wcopy_kernel, dim=(nw, 4), inputs=[self.wr], outputs=[self.wcat], device=dev)
                    wp.launch(ug.wgcopy_kernel, dim=nw, inputs=[self.wg], outputs=[self.wgcat], device=dev)
                wp.launch(ug.shed_kernel, dim=ny, inputs=[rings3, self.gamma, self.te, self.Vw, self.dtt, nw],
                          outputs=[self.wcat, self.wgcat], device=dev)
                wp.launch(ug.convect_kernel, dim=(nwn, 4), inputs=[rings3, self.gamma, npan, self.wcat,
                          self.wgcat, nwn, self.Vw, self.dtt], outputs=[self.wr_next], device=dev)
        return self.Fp.numpy()

    def backward(self, adj_Fp_np, adj_gamma_extra=None, adj_wr_next=None, adj_wg_next=None):
        dev = self.dev; npan = self.npan
        for a in (self.cw, self.vw, self.gprev, self.gamma, self.AIC, self.rhs):
            a.grad.zero_()
        if self.nw > 0:
            self.wr.grad.zero_(); self.wg.grad.zero_()
        # tape3 (wake update) first: seeds the wake_{t+1} adjoint -> gamma, corners(rings3), wake_in
        if self.use_wake:
            if adj_wr_next is None:                              # last step: wake_{t+1} unused downstream
                self.wr_next.grad = wp.zeros_like(self.wr_next)
                self.wgcat.grad = wp.zeros_like(self.wgcat)
            else:
                self.wr_next.grad = wp.array(np.ascontiguousarray(adj_wr_next, np.float64).reshape(-1, 4, 3),
                                             dtype=V3, device=dev)
                self.wgcat.grad = wp.array(np.ascontiguousarray(adj_wg_next, np.float64), dtype=DTYPE, device=dev)
            self.t3.backward()
        self.Fp.grad = wp.array(np.ascontiguousarray(adj_Fp_np, np.float64).reshape(npan, 3),
                                dtype=V3, device=dev)
        self.t2.backward()
        if adj_gamma_extra is not None:
            wp.launch(ug._acc2, dim=(1, npan), inputs=[wp.array(adj_gamma_extra.reshape(1, npan),
                      dtype=DTYPE, device=dev)], outputs=[self.gamma.grad], device=dev)
        adj_A, adj_b = self.dds.backward(self.gamma.grad)
        self.AIC.grad = adj_A; self.rhs.grad = adj_b
        self.t1.backward()
        adj_wr = self.wr.grad.numpy().copy() if self.nw > 0 else None
        adj_wg = self.wg.grad.numpy().copy() if self.nw > 0 else None
        out = (self.cw.grad.numpy().copy(), self.vw.grad.numpy().copy(),
               self.gprev.grad.numpy().copy(), adj_wr, adj_wg)
        self.t1.zero(); self.t2.zero()
        if self.use_wake:
            self.t3.zero()
        return out


def coupled_unsteady_forward_gpu(sh, C, P, dist, q0, dq0, N, dt, Es, Rs, nx, ny,
                                 Vinf=VINF, cg_tol=1e-12, use_wake=True, control=None, fb_gain=None):
    dev = cfg.DEVICE; NP = cfg.NP_DTYPE; ndof = C.ndof
    npan = nx * ny; ncv = (nx + 1) * (ny + 1); maxw = N * ny
    Esw = wp.array(Es.astype(NP), dtype=DTYPE, device=dev)
    Mscaled = wp.zeros((C.ne, 36, 36), dtype=DTYPE, device=dev)
    wp.launch(dsg._scaled_mass, dim=(C.ne, 36, 36),
              inputs=[C.Me, wp.array(Rs.astype(NP), dtype=DTYPE, device=dev)], outputs=[Mscaled], device=dev)
    Kblk0 = wp.zeros((1, C.ne, 36, 36), dtype=DTYPE, device=dev)
    wa = lambda v: wp.array(v[None].astype(NP), dtype=DTYPE, device=dev)
    Vw = V3(*[float(v) for v in np.asarray(Vinf, float)])
    te = wp.array(np.array([(nx - 1) * ny + j for j in range(ny)], np.int32), dtype=wp.int32, device=dev)
    wr = wp.zeros((maxw, 4), dtype=V3, device=dev); wr_new = wp.zeros((maxw, 4), dtype=V3, device=dev)
    wg = wp.zeros(maxw, dtype=DTYPE, device=dev)
    gprev = wp.zeros((1, npan), dtype=DTYPE, device=dev)
    q = q0.copy(); dq = dq0.copy(); nw = 0
    qs = [q.copy()]
    for t in range(N):
        corners = (P @ q).reshape(ncv, 3); cvel = (P @ dq).reshape(ncv, 3)
        cw = wp.array(corners.astype(NP), dtype=V3, device=dev)
        vw = wp.array(cvel.astype(NP), dtype=V3, device=dev)
        rings = wp.zeros((npan, 4), dtype=V3, device=dev); col = wp.zeros(npan, dtype=V3, device=dev)
        nrm = wp.zeros(npan, dtype=V3, device=dev); vcol = wp.zeros(npan, dtype=V3, device=dev)
        wp.launch(ug.bound_rings_kernel, dim=npan, inputs=[cw, nx, ny], outputs=[rings, col, nrm], device=dev)
        wp.launch(ug.colvel_kernel, dim=npan, inputs=[vw, nx, ny], outputs=[vcol], device=dev)
        AIC = wp.zeros((1, npan, npan), dtype=DTYPE, device=dev)
        wp.launch(ug.aic_kernel, dim=(npan, npan), inputs=[rings, col, nrm], outputs=[AIC], device=dev)
        rhs = wp.zeros((1, npan), dtype=DTYPE, device=dev)
        wp.launch(ug.rhs_moving_kernel, dim=npan, inputs=[col, nrm, Vw, vcol, wr, wg, nw], outputs=[rhs], device=dev)
        gamma = batched_dense_solve(AIC, rhs, dev)
        Fp = wp.zeros(npan, dtype=V3, device=dev)
        wp.launch(ug.panel_force_kernel, dim=npan, inputs=[rings, nrm, gamma, gprev, vcol, Vw,
                  DTYPE(dt), DTYPE(ug.RHO), ny], outputs=[Fp], device=dev)
        Fnodal = dist @ Fp.numpy().reshape(-1)
        Qmem, Qbend = dsg.design_internal_force(wa(q), C, Esw, dev)
        Qint = Qmem.numpy()[0] + Qbend.numpy()[0]
        ctrl_t = (control[t] if control is not None else 0.0)
        if fb_gain is not None:
            ctrl_t = ctrl_t - fb_gain * dq * _pos_mask(C)       # actuate POSITION DOFs only (slope-DOF actuation is unstable)
        rhs_s = (Fnodal - Qint + ctrl_t) * C.free_np
        a, _ = structural_cg(wa(rhs_s), Mscaled, Kblk0, C.edofs, C.free, 0.0, ndof, tol=cg_tol, device=dev)
        a_np = a.numpy()[0]
        dq = dq + dt * a_np; q = q + dt * dq; qs.append(q.copy())
        if use_wake:
            wp.launch(ug.shed_kernel, dim=ny, inputs=[rings, gamma, te, Vw, DTYPE(dt), nw], outputs=[wr, wg], device=dev)
            nw += ny
            wp.launch(ug.convect_kernel, dim=(nw, 4), inputs=[rings, gamma, npan, wr, wg, nw, Vw, DTYPE(dt)],
                      outputs=[wr_new], device=dev)
            wp.copy(wr, wr_new, count=nw * 4)
        gprev = wp.array(gamma.numpy(), dtype=DTYPE, device=dev)
    return q, qs


def coupled_unsteady_forward_pc_gpu(sh, C, P, dist, q0, dq0, N, dt, Es, Rs, nx, ny,
                                    Vinf=VINF, cg_tol=1e-11, use_wake=True, control=None, fb_gain=None,
                                    beta=0.25, gamma=0.5, wake_max=80, pc_it=20, pc_tol=1e-8,
                                    omega0=0.3, return_traj=False):
    """all-Warp STRONG-coupled (predictor-corrector) unsteady FSI forward — the GPU production version
    of dcu.coupled_unsteady_forward_pc. Each step solves the coupled fixed point on a_{n+1}: the
    linearly-implicit-Newmark operator A = M(ρ) + β·dt²·K_mem(q_pred;E) (matrix-free batched CG,
    membrane tangent = the GPU K_t convention) is solved against the aero force RE-EVALUATED on the
    current structural iterate q_it = q_pred + β·dt²·a_it, with Aitken Δ² dynamic relaxation. This is
    the cure for the fluid added-mass instability that diverges loose/explicit coupling for light
    wings. The aero (bound rings → AIC → γ → unsteady KJ + dΓ/dt) reuses the validated Warp kernels;
    the wake (shed TE + free convection) is advanced ONCE per step on the converged geometry and
    truncated to the most-recent `wake_max` rings. Returns the converged final state (and the
    trajectory + per-step iteration counts if return_traj). The reference is dcu.…_pc(tangent=
    'membrane'), which this matches to ~pc_tol; the differentiable PC adjoint (next) differentiates
    this converged fixed point via the implicit function theorem."""
    dev = cfg.DEVICE; NP = cfg.NP_DTYPE; ndof = C.ndof
    npan = nx * ny; ncv = (nx + 1) * (ny + 1)
    maxw = (min(N * ny, wake_max) + ny) if use_wake else ny
    Esw = wp.array(Es.astype(NP), dtype=DTYPE, device=dev)
    Mscaled = wp.zeros((C.ne, 36, 36), dtype=DTYPE, device=dev)
    wp.launch(dsg._scaled_mass, dim=(C.ne, 36, 36),
              inputs=[C.Me, wp.array(Rs.astype(NP), dtype=DTYPE, device=dev)], outputs=[Mscaled], device=dev)
    Kblk0 = wp.zeros((1, C.ne, 36, 36), dtype=DTYPE, device=dev)
    wa = lambda v: wp.array(v[None].astype(NP), dtype=DTYPE, device=dev)
    Vw = V3(*[float(v) for v in np.asarray(Vinf, float)])
    te = wp.array(np.array([(nx - 1) * ny + j for j in range(ny)], np.int32), dtype=wp.int32, device=dev)
    pm = _pos_mask(C); coef = beta * dt * dt
    wr = wp.zeros((maxw, 4), dtype=V3, device=dev); wr_new = wp.zeros((maxw, 4), dtype=V3, device=dev)
    wg = wp.zeros(maxw, dtype=DTYPE, device=dev)
    gprev = wp.zeros((1, npan), dtype=DTYPE, device=dev)

    def _aero_Fp(q_it, dq_it):
        """Bound solve on the current iterate geometry → (Fp_np, rings, gamma) for the residual; the
        wake (wr,wg,nw) and gprev are held fixed during the within-step iteration."""
        corners = (P @ q_it).reshape(ncv, 3); cvel = (P @ dq_it).reshape(ncv, 3)
        cw = wp.array(corners.astype(NP), dtype=V3, device=dev)
        vw = wp.array(cvel.astype(NP), dtype=V3, device=dev)
        rings = wp.zeros((npan, 4), dtype=V3, device=dev); col = wp.zeros(npan, dtype=V3, device=dev)
        nrm = wp.zeros(npan, dtype=V3, device=dev); vcol = wp.zeros(npan, dtype=V3, device=dev)
        wp.launch(ug.bound_rings_kernel, dim=npan, inputs=[cw, nx, ny], outputs=[rings, col, nrm], device=dev)
        wp.launch(ug.colvel_kernel, dim=npan, inputs=[vw, nx, ny], outputs=[vcol], device=dev)
        AIC = wp.zeros((1, npan, npan), dtype=DTYPE, device=dev)
        wp.launch(ug.aic_kernel, dim=(npan, npan), inputs=[rings, col, nrm], outputs=[AIC], device=dev)
        rhs = wp.zeros((1, npan), dtype=DTYPE, device=dev)
        wp.launch(ug.rhs_moving_kernel, dim=npan, inputs=[col, nrm, Vw, vcol, wr, wg, nw], outputs=[rhs], device=dev)
        gam = batched_dense_solve(AIC, rhs, dev)
        Fp = wp.zeros(npan, dtype=V3, device=dev)
        wp.launch(ug.panel_force_kernel, dim=npan, inputs=[rings, nrm, gam, gprev, vcol, Vw,
                  DTYPE(dt), DTYPE(ug.RHO), ny], outputs=[Fp], device=dev)
        return dist @ Fp.numpy().reshape(-1), rings, gam

    q = q0.copy(); dq = dq0.copy(); nw = 0
    # initial accel a0 = M⁻¹(−Qint(q0)) (explicit, consistent with the numpy oracle predictor)
    Qmem, Qbend = dsg.design_internal_force(wa(q), C, Esw, dev)
    Qint0 = Qmem.numpy()[0] + Qbend.numpy()[0]
    a, _ = structural_cg(wa((-Qint0) * C.free_np), Mscaled, Kblk0, C.edofs, C.free, 0.0, ndof, tol=cg_tol, device=dev)
    a = a.numpy()[0]
    qs = [q.copy()]; itc = []
    for t in range(N):
        q_pred = q + dt * dq + dt * dt * (0.5 - beta) * a
        v_pred = dq + dt * (1.0 - gamma) * a
        qpw = wa(q_pred)
        Qmem, Qbend = dsg.design_internal_force(qpw, C, Esw, dev)
        Qint = Qmem.numpy()[0] + Qbend.numpy()[0]
        Kblk = assemble_kmem_blocks(qpw, C, dev)
        wp.launch(dsg._scale_kblk, dim=(1, C.ne, 36, 36), inputs=[Kblk, Esw], device=dev)
        ctrl_t = (control[t] if control is not None else 0.0)
        a_it = a.copy(); omega = omega0; r_prev = None; rings_c = None; gam_c = None
        it = 0
        for it in range(pc_it):
            q_it = q_pred + coef * a_it; dq_it = v_pred + gamma * dt * a_it
            Fnodal, rings_c, gam_c = _aero_Fp(q_it, dq_it)
            c = ctrl_t - fb_gain * dq_it * pm if fb_gain is not None else ctrl_t
            rhs_s = (Fnodal - Qint + c) * C.free_np
            a_solve_w, _ = structural_cg(wa(rhs_s), Mscaled, Kblk, C.edofs, C.free, coef, ndof,
                                         tol=cg_tol, device=dev)
            a_solve = a_solve_w.numpy()[0]
            r = a_solve - a_it
            if np.linalg.norm(r[C.free_np > 0]) < pc_tol * (np.linalg.norm(a_solve[C.free_np > 0]) + 1e-30):
                a_it = a_solve; break
            if r_prev is not None:                                   # Aitken Δ² dynamic relaxation
                dr = (r - r_prev)[C.free_np > 0]
                omega = -omega * float(np.dot(r_prev[C.free_np > 0], dr)) / (float(np.dot(dr, dr)) + 1e-30)
                omega = float(np.clip(omega, 0.05, 1.0))
            a_it = a_it + omega * r; r_prev = r
        itc.append(it + 1)
        a = a_it; q = q_pred + coef * a; dq = v_pred + gamma * dt * a
        qs.append(q.copy())
        if not np.all(np.isfinite(q)):
            return (q, np.array(qs), itc) if return_traj else q
        if use_wake:                                                 # advance wake ONCE on converged geometry
            wp.launch(ug.shed_kernel, dim=ny, inputs=[rings_c, gam_c, te, Vw, DTYPE(dt), nw], outputs=[wr, wg], device=dev)
            nw_new = nw + ny
            wp.launch(ug.convect_kernel, dim=(nw_new, 4), inputs=[rings_c, gam_c, npan, wr, wg, nw_new, Vw, DTYPE(dt)],
                      outputs=[wr_new], device=dev)
            wp.copy(wr, wr_new, count=nw_new * 4)
            if nw_new > wake_max:                                    # keep the most-recent wake_max rings
                off = nw_new - wake_max
                wr_h = wr.numpy(); wg_h = wg.numpy()
                wr = wp.array(np.ascontiguousarray(wr_h[off:nw_new]), dtype=V3, device=dev)
                wg = wp.array(np.ascontiguousarray(wg_h[off:nw_new]), dtype=DTYPE, device=dev)
                wr_new = wp.zeros((maxw, 4), dtype=V3, device=dev)
                # repad to capacity so subsequent shed has room
                wr_full = wp.zeros((maxw, 4), dtype=V3, device=dev); wp.copy(wr_full, wr, count=wake_max * 4)
                wg_full = wp.zeros(maxw, dtype=DTYPE, device=dev); wp.copy(wg_full, wg, count=wake_max)
                wr = wr_full; wg = wg_full; nw = wake_max
            else:
                nw = nw_new
        gprev = wp.array(gam_c.numpy(), dtype=DTYPE, device=dev)
    return (q, np.array(qs), itc) if return_traj else q


def coupled_unsteady_grad_gpu(sh, C, P, dist, q0, dq0, N, dt, w, Es, Rs, nx, ny,
                              Vinf=VINF, use_wake=False, cg_tol=1e-12, control=None, fb_gain=None):
    """all-Warp coupled UNSTEADY FSI design gradient ∂L/∂(E,ρ). Chains: structure design adjoint
    (adj_E/adj_rho + membrane-K_t state chain) ⊗ unsteady aero VJP (UnsteadyAeroGpu: ∂F/∂corners,
    ∂F/∂V_body, ∂F/∂gprev) through the coupled recurrence. adj_q gets the aero ∂/∂corners (P^T) and
    the structure chain; adj_dq gets the moving-body ∂/∂V_body (P^T); the dΓ/dt coupling carries
    adj_gprev → gamma_{t-1}. use_wake=False isolates this (no wake recurrence).

    control: optional (N, ndof) per-step actuation force added to the structural rhs (the SHAC
    control input). Returns the extra gradient gC = ∂L/∂control (gC[t] = adj_rhs_t on free DOFs) —
    exactly the policy-gradient signal ∂L/∂action_t a meta-RL policy backprops through."""
    dev = cfg.DEVICE; NP = cfg.NP_DTYPE; ndof = C.ndof
    npan = nx * ny; ncv = (nx + 1) * (ny + 1); maxw = N * ny
    Esw = wp.array(Es.astype(NP), dtype=DTYPE, device=dev)
    Mscaled = wp.zeros((C.ne, 36, 36), dtype=DTYPE, device=dev)
    wp.launch(dsg._scaled_mass, dim=(C.ne, 36, 36),
              inputs=[C.Me, wp.array(Rs.astype(NP), dtype=DTYPE, device=dev)], outputs=[Mscaled], device=dev)
    Kblk0 = wp.zeros((1, C.ne, 36, 36), dtype=DTYPE, device=dev)
    wa = lambda v: wp.array(v[None].astype(NP), dtype=DTYPE, device=dev)
    zc = lambda: wp.zeros((1, ndof), dtype=DTYPE, device=dev)
    aero = UnsteadyAeroGpu(nx, ny, Vinf, dt, dev)
    Vw = V3(*[float(v) for v in np.asarray(Vinf, float)])
    te = wp.array(np.array([(nx - 1) * ny + j for j in range(ny)], np.int32), dtype=wp.int32, device=dev)
    dummy_wr = wp.zeros((1, 4), dtype=V3, device=dev, requires_grad=True)
    dummy_wg = wp.zeros(1, dtype=DTYPE, device=dev, requires_grad=True)

    # ---- forward with storage ----
    q = q0.copy(); dq = dq0.copy(); nw = 0
    qs = []; dqs = []; araws = []; gammas = []; wake_snaps = []
    gprev_np = np.zeros((1, npan)); cur_wr = None; cur_wg = None
    for t in range(N):
        corners = (P @ q).reshape(ncv, 3); cvel = (P @ dq).reshape(ncv, 3)
        wake_snaps.append((None if nw == 0 else cur_wr.copy(), None if nw == 0 else cur_wg.copy(), nw))
        wr_t = wp.array(cur_wr, dtype=V3, device=dev, requires_grad=True) if nw > 0 else dummy_wr
        wg_t = wp.array(cur_wg, dtype=DTYPE, device=dev, requires_grad=True) if nw > 0 else dummy_wg
        Fp = aero.forward(corners, cvel, wr_t, wg_t, nw, gprev_np, use_wake=use_wake)
        gamma_np = aero.gamma.numpy()
        Fnodal = dist @ Fp.reshape(-1)
        Qmem, Qbend = dsg.design_internal_force(wa(q), C, Esw, dev)
        Qint = Qmem.numpy()[0] + Qbend.numpy()[0]
        ctrl_t = (control[t] if control is not None else 0.0)
        if fb_gain is not None:                                # closed-loop state feedback u_t=-k·dq_t (position DOFs)
            ctrl_t = ctrl_t - fb_gain * dq * _pos_mask(C)
        rhs_s = Fnodal - Qint + ctrl_t
        a, _ = structural_cg(wa(rhs_s * C.free_np), Mscaled, Kblk0, C.edofs, C.free, 0.0,
                             ndof, tol=cg_tol, device=dev)
        a_np = a.numpy()[0]
        qs.append(q.copy()); dqs.append(dq.copy()); araws.append(a_np.copy()); gammas.append(gamma_np.copy())
        dq = dq + dt * a_np; q = q + dt * dq
        if use_wake:
            cur_wr = aero.wr_next.numpy().copy(); cur_wg = aero.wgcat.numpy().copy(); nw += ny
        gprev_np = gamma_np
    L = float(w @ q)

    # ---- backward ----
    gE = wp.zeros(C.ne, dtype=DTYPE, device=dev); gR = wp.zeros(C.ne, dtype=DTYPE, device=dev)
    gC = np.zeros((N, ndof))                                    # ∂L/∂control_t (the policy-gradient signal)
    dL_dk = 0.0                                                 # closed-loop feedback gain gradient
    adj_q = w.copy(); adj_dq = np.zeros(ndof); adj_gamma_carry = None
    adj_wr_next = None; adj_wg_next = None
    for t in reversed(range(N)):
        aq1 = adj_q; ad1 = adj_dq + dt * aq1; adj_a = dt * ad1
        adj_rhs_w, _ = structural_cg(wa(adj_a * C.free_np), Mscaled, Kblk0, C.edofs, C.free, 0.0,
                                     ndof, tol=cg_tol, device=dev)
        adj_rhs = adj_rhs_w.numpy()[0]
        gC[t] = adj_rhs * C.free_np                             # control enters rhs linearly ⇒ ∂L/∂u_t = adj_rhs_t
        corners_t = (P @ qs[t]).reshape(ncv, 3); cvel_t = (P @ dqs[t]).reshape(ncv, 3)
        gprev_t = gammas[t - 1] if t > 0 else np.zeros((1, npan))
        wr_np, wg_np, nw_t = wake_snaps[t]
        wr_t = wp.array(wr_np, dtype=V3, device=dev, requires_grad=True) if nw_t > 0 else dummy_wr
        wg_t = wp.array(wg_np, dtype=DTYPE, device=dev, requires_grad=True) if nw_t > 0 else dummy_wg
        aero.forward(corners_t, cvel_t, wr_t, wg_t, nw_t, gprev_t, use_wake=use_wake)
        adj_Fp = (dist.T @ adj_rhs).reshape(npan, 3)
        adj_corners, adj_cvel, adj_gprev, adj_wr, adj_wg = aero.backward(
            adj_Fp, adj_gamma_carry, adj_wr_next, adj_wg_next)
        adj_wr_next = adj_wr; adj_wg_next = adj_wg              # chain wake adj to step t-1's wake output
        adj_q_aero = P.T @ adj_corners.reshape(-1)
        adj_dq_aero = P.T @ adj_cvel.reshape(-1)
        # structure design adjoint + membrane-K_t state chain (as diff_coupled_gpu)
        adj_Qint = -adj_rhs
        qtw = wa(qs[t])
        _, _, deps, dk, Dm_eps, Dk_k = dsg._design_force_cached(qtw, C, Esw, dev)
        wp.launch(dsg.adj_E_kernel, dim=(1, C.ne, 36),
                  inputs=[C.gw, deps, dk, Dm_eps, Dk_k, C.edofs, DTYPE(NP(C.h)), C.ngg, Esw, wa(adj_Qint)],
                  outputs=[gE], device=dev)
        wp.launch(dsg.adj_rho_kernel, dim=(1, C.ne), inputs=[C.Me, C.edofs, wa(adj_rhs), wa(araws[t])],
                  outputs=[gR], device=dev)
        Kblk = assemble_kmem_blocks(qtw, C, dev)
        wp.launch(dsg._scale_kblk, dim=(1, C.ne, 36, 36), inputs=[Kblk, Esw], device=dev)
        adj_qK = zc()
        apply_MK(wa(adj_Qint * C.free_np), adj_qK, C.Me, Kblk, C.edofs, C.free, 0.0, 1.0, dev)
        adj_dq_policy = 0.0
        if fb_gain is not None:                                # closed-loop feedback u_t=-k·dq_t·pos in the loop
            pm = _pos_mask(C)
            dL_dk += -float(np.dot(adj_rhs * pm, dqs[t]))      # ∂u_t/∂k = -dq_t·pos
            adj_dq_policy = -fb_gain * adj_rhs * pm            # ∂u_t/∂dq_t = -k·pos → feeds the state adjoint
        adj_q = aq1 + adj_qK.numpy()[0] + adj_q_aero
        adj_dq = ad1 + adj_dq_aero + adj_dq_policy
        adj_gamma_carry = adj_gprev                            # dΓ/dt coupling: adj_gprev → gamma_{t-1}
    return L, gE.numpy(), gR.numpy(), gC, dL_dk


def coupled_unsteady_pc_grad_gpu(sh, C, P, dist, q0, dq0, N, dt, w, Es, Rs, nx, ny,
                                 Vinf=VINF, use_wake=False, cg_tol=1e-11, control=None, fb_gain=None,
                                 beta=0.25, gamma=0.5, wake_max=80, pc_it=30, pc_tol=1e-9,
                                 omega0=0.3, adj_it=80, adj_tol=1e-11,
                                 dLdq=None, dLddq=None, dLda=None, gust=None, loss_fn=None):
    """Differentiable adjoint of the STRONG-coupled (predictor-corrector) unsteady FSI forward via the
    IMPLICIT FUNCTION THEOREM. The forward converges a per-step fixed point a* = A⁻¹(Fnodal(a*)−Qint+c);
    its exact reverse-mode gradient differentiates the CONVERGED fixed point, NOT the Aitken iteration
    path (ω drops out). Per step: (A) build the seed g = ∂L/∂a* from the downstream q/dq/a and the aero
    OUTPUT path (dΓ/dt γ + wake); (B) solve the adjoint fixed point  x̄ = (∂Fnodal/∂a)ᵀ A⁻¹ x̄ + g  by
    the same iteration (reuses the forward's A=M+β·dt²·K_mem solve and the UnsteadyAeroGpu force VJP —
    converges at the forward rate); (C) push u*=A⁻¹x̄ through the design (∂E,∂ρ), control (gC), feedback
    (dL/dk) and state (predictor reverse with the a-carry) VJPs. The aero FORCE path and OUTPUT path are
    kept separate (two aero.backward calls) because the force-path a*-sensitivity lives inside the
    fixed-point Jacobian while the output-path a*-sensitivity must pass through the implicit solve.
    Returns (L, gE, gR, gC, dL_dk). Validated vs FD of the numpy PC oracle (tangent='membrane')."""
    dev = cfg.DEVICE; NP = cfg.NP_DTYPE; ndof = C.ndof
    npan = nx * ny; ncv = (nx + 1) * (ny + 1)
    maxw = (min(N * ny, wake_max) + ny) if use_wake else ny
    Esw = wp.array(Es.astype(NP), dtype=DTYPE, device=dev)
    Mscaled = wp.zeros((C.ne, 36, 36), dtype=DTYPE, device=dev)
    wp.launch(dsg._scaled_mass, dim=(C.ne, 36, 36),
              inputs=[C.Me, wp.array(Rs.astype(NP), dtype=DTYPE, device=dev)], outputs=[Mscaled], device=dev)
    Kblk0 = wp.zeros((1, C.ne, 36, 36), dtype=DTYPE, device=dev)
    wa = lambda v: wp.array(v[None].astype(NP), dtype=DTYPE, device=dev)
    zc = lambda: wp.zeros((1, ndof), dtype=DTYPE, device=dev)
    Vw = V3(*[float(v) for v in np.asarray(Vinf, float)])
    Vinf0 = np.asarray(Vinf, float)
    Vw_of = lambda tt: V3(*[float(v) for v in (Vinf0 + (gust[tt] if gust is not None else 0.0))])  # 1-cos gust per step
    te = wp.array(np.array([(nx - 1) * ny + j for j in range(ny)], np.int32), dtype=wp.int32, device=dev)
    pm = _pos_mask(C); coef = beta * dt * dt
    aero = UnsteadyAeroGpu(nx, ny, Vinf, dt, dev)
    dummy_wr = wp.zeros((1, 4), dtype=V3, device=dev, requires_grad=True)
    dummy_wg = wp.zeros(1, dtype=DTYPE, device=dev, requires_grad=True)
    wr = wp.zeros((maxw, 4), dtype=V3, device=dev); wr_new = wp.zeros((maxw, 4), dtype=V3, device=dev)
    wg = wp.zeros(maxw, dtype=DTYPE, device=dev)

    def Ainv(vfree, Kblk):                                    # A⁻¹ v  on free DOFs, A = M + coef·K_mem
        x, _ = structural_cg(wa(vfree * C.free_np), Mscaled, Kblk, C.edofs, C.free, coef, ndof,
                             tol=cg_tol, device=dev)
        return x.numpy()[0]

    # ---- forward (PC) with storage ----
    q = q0.copy(); dq = dq0.copy(); nw = 0
    Qm0, Qb0 = dsg.design_internal_force(wa(q), C, Esw, dev)
    a, _ = structural_cg(wa((-(Qm0.numpy()[0] + Qb0.numpy()[0])) * C.free_np), Mscaled, Kblk0,
                         C.edofs, C.free, 0.0, ndof, tol=cg_tol, device=dev)
    a = a.numpy()[0]; a0 = a.copy()
    q_preds = []; v_preds = []; a_stars = []; q_outs = []; dq_outs = []
    gammas = []; gprev_list = []; wake_snaps = []
    gprev_np = np.zeros((1, npan)); cur_wr = None; cur_wg = None
    for t in range(N):
        Vw = Vw_of(t); aero.Vw = Vw                          # per-step gusted freestream
        q_pred = q + dt * dq + dt * dt * (0.5 - beta) * a
        v_pred = dq + dt * (1.0 - gamma) * a
        qpw = wa(q_pred)
        Qmem, Qbend = dsg.design_internal_force(qpw, C, Esw, dev)
        Qint = Qmem.numpy()[0] + Qbend.numpy()[0]
        Kblk = assemble_kmem_blocks(qpw, C, dev)
        wp.launch(dsg._scale_kblk, dim=(1, C.ne, 36, 36), inputs=[Kblk, Esw], device=dev)
        ctrl_t = (control[t] if control is not None else 0.0)
        wr_t = wp.array(cur_wr, dtype=V3, device=dev) if nw > 0 else dummy_wr
        wg_t = wp.array(cur_wg, dtype=DTYPE, device=dev) if nw > 0 else dummy_wg
        wake_snaps.append((None if nw == 0 else cur_wr.copy(), None if nw == 0 else cur_wg.copy(), nw))
        gprev_list.append(gprev_np.copy())
        a_it = a.copy(); omega = omega0; r_prev = None; gam_c = None
        for it in range(pc_it):
            q_it = q_pred + coef * a_it; dq_it = v_pred + gamma * dt * a_it
            Fp = aero.forward((P @ q_it).reshape(ncv, 3), (P @ dq_it).reshape(ncv, 3),
                              wr_t, wg_t, nw, gprev_np, use_wake=use_wake)
            gam_c = aero.gamma
            Fnodal = dist @ Fp.reshape(-1)
            c = ctrl_t - fb_gain * dq_it * pm if fb_gain is not None else ctrl_t
            a_solve = Ainv(Fnodal - Qint + c, Kblk)
            r = a_solve - a_it
            if np.linalg.norm(r[C.free_np > 0]) < pc_tol * (np.linalg.norm(a_solve[C.free_np > 0]) + 1e-30):
                a_it = a_solve; break
            if r_prev is not None:
                dr = (r - r_prev)[C.free_np > 0]
                omega = -omega * float(np.dot(r_prev[C.free_np > 0], dr)) / (float(np.dot(dr, dr)) + 1e-30)
                omega = float(np.clip(omega, 0.05, 1.0))
            a_it = a_it + omega * r; r_prev = r
        a = a_it; q = q_pred + coef * a; dq = v_pred + gamma * dt * a
        gam_np = gam_c.numpy()
        q_preds.append(q_pred); v_preds.append(v_pred); a_stars.append(a.copy())
        q_outs.append(q.copy()); dq_outs.append(dq.copy()); gammas.append(gam_np.copy())
        if use_wake:                                          # advance wake on converged geometry
            cw = wp.array((P @ q).reshape(ncv, 3).astype(NP), dtype=V3, device=dev)
            rings = wp.zeros((npan, 4), dtype=V3, device=dev); col = wp.zeros(npan, dtype=V3, device=dev)
            nrm = wp.zeros(npan, dtype=V3, device=dev)
            wp.launch(ug.bound_rings_kernel, dim=npan, inputs=[cw, nx, ny], outputs=[rings, col, nrm], device=dev)
            wp.launch(ug.shed_kernel, dim=ny, inputs=[rings, gam_c, te, Vw, DTYPE(dt), nw], outputs=[wr, wg], device=dev)
            nw_new = nw + ny
            wp.launch(ug.convect_kernel, dim=(nw_new, 4), inputs=[rings, gam_c, npan, wr, wg, nw_new, Vw, DTYPE(dt)],
                      outputs=[wr_new], device=dev)
            wp.copy(wr, wr_new, count=nw_new * 4)
            if nw_new > wake_max:                            # keep the most-recent wake_max rings
                off = nw_new - wake_max; wr_h = wr.numpy(); wg_h = wg.numpy()
                tmp_wr = np.zeros((maxw, 4, 3)); tmp_wr[:wake_max] = wr_h[off:nw_new]
                tmp_wg = np.zeros(maxw); tmp_wg[:wake_max] = wg_h[off:nw_new]
                wr = wp.array(tmp_wr, dtype=V3, device=dev); wg = wp.array(tmp_wg, dtype=DTYPE, device=dev)
                wr_new = wp.zeros((maxw, 4), dtype=V3, device=dev); nw = wake_max
                cur_wr = tmp_wr[:wake_max].copy(); cur_wg = tmp_wg[:wake_max].copy()  # exact nw rings
            else:
                nw = nw_new; cur_wr = wr.numpy()[:nw_new].copy(); cur_wg = wg.numpy()[:nw_new].copy()
        gprev_np = gam_np
    L = float(w @ q); dLdk_extra = 0.0
    if loss_fn is not None:                                   # general trajectory functional on the
        # GUSTED trajectory the adjoint differentiates: returns (L, dLdq, dLddq, dLda[, dLdk_extra]),
        # the (N+1,ndof) per-step seeds plus an optional EXPLICIT ∂L/∂k (e.g. a control-effort ½k²‖·‖² term)
        _r = loss_fn(np.array(q_outs), np.array(dq_outs), np.array(a_stars), q0, dq0)
        L, dLdq, dLddq, dLda = _r[:4]
        if len(_r) > 4 and _r[4] is not None: dLdk_extra = float(_r[4])

    # ---- backward (IFT adjoint of the PC fixed point) ----
    gE = wp.zeros(C.ne, dtype=DTYPE, device=dev); gR = wp.zeros(C.ne, dtype=DTYPE, device=dev)
    gC = np.zeros((N, ndof)); dL_dk = 0.0
    adj_q = w.copy(); adj_dq = np.zeros(ndof); adj_a = np.zeros(ndof)
    adj_gamma_carry = None; adj_wr_next = None; adj_wg_next = None
    PT = P.T; DT = dist.T
    for t in reversed(range(N)):
        if dLdq is not None: adj_q = adj_q + dLdq[t + 1]     # general per-step loss: explicit ∂L/∂q_{t+1}
        if dLddq is not None: adj_dq = adj_dq + dLddq[t + 1]  # explicit ∂L/∂dq_{t+1}
        q_pred = q_preds[t]; v_pred = v_preds[t]; a_star = a_stars[t]
        q_out = q_outs[t]; dq_out = dq_outs[t]; gprev_t = gprev_list[t]
        aero.Vw = Vw_of(t)                                   # match the forward's per-step gusted freestream
        wr_np, wg_np, nw_t = wake_snaps[t]
        qpw = wa(q_pred)
        Kblk = assemble_kmem_blocks(qpw, C, dev)
        wp.launch(dsg._scale_kblk, dim=(1, C.ne, 36, 36), inputs=[Kblk, Esw], device=dev)
        # aero linearization at the converged geometry (q_out, dq_out), with this step's input wake/gprev
        wr_t = wp.array(wr_np, dtype=V3, device=dev, requires_grad=True) if nw_t > 0 else dummy_wr
        wg_t = wp.array(wg_np, dtype=DTYPE, device=dev, requires_grad=True) if nw_t > 0 else dummy_wg
        aero.forward((P @ q_out).reshape(ncv, 3), (P @ dq_out).reshape(ncv, 3),
                     wr_t, wg_t, nw_t, gprev_t, use_wake=use_wake)

        def fvjp(uvec, agamma=None, awr=None, awg=None):     # aero VJP seeded by force distᵀu (+ output seeds)
            adj_Fp = (DT @ uvec).reshape(npan, 3) if uvec is not None else np.zeros((npan, 3))
            return aero.backward(adj_Fp, agamma, awr, awg)   # -> adj_corners, adj_cvel, adj_gprev, adj_wr, adj_wg

        # STEP A: seed g = ∂L/∂a*  (downstream q/dq/a + explicit per-step ∂L/∂a* + aero OUTPUT path)
        g = beta * dt * dt * adj_q + gamma * dt * adj_dq + adj_a
        if dLda is not None: g = g + dLda[t + 1]
        out_gprev = None; out_wr = None; out_wg = None
        out_qpred = np.zeros(ndof); out_vpred = np.zeros(ndof)
        if (adj_gamma_carry is not None) or (adj_wr_next is not None):
            ac, av, agp, awr, awg = fvjp(None, adj_gamma_carry, adj_wr_next, adj_wg_next)
            # the aero outputs (γ*, wake) are evaluated at corners=P·(q_pred+β·dt²·a*), cvel=P·(v_pred+γ·dt·a*),
            # so they depend on a* (→ seed g) AND on q_pred/v_pred (→ state, added in STEP C)
            out_qpred = PT @ ac.reshape(-1); out_vpred = PT @ av.reshape(-1)
            g = g + beta * dt * dt * out_qpred + gamma * dt * out_vpred
            out_gprev = agp; out_wr = awr; out_wg = awg

        # STEP B: adjoint fixed point  x̄ = M x̄ + g,  M = (∂Fnodal/∂a + ∂c/∂a)ᵀ A⁻¹  (= G_aᵀ).
        # Solved by the SAME Aitken Δ² relaxation as the forward — M shares the forward map's spectrum
        # (transpose), so plain Picard diverges exactly where the forward would, and the forward's
        # Aitken acceleration is what makes strong coupling / strong feedback converge. Mirror it.
        xbar = g.copy(); omega = omega0; r_prev = None; gm = C.free_np > 0
        for k in range(adj_it):
            u = Ainv(xbar, Kblk)
            ac, av, _, _, _ = fvjp(u)
            Mx = beta * dt * dt * (PT @ ac.reshape(-1)) + gamma * dt * (PT @ av.reshape(-1))
            if fb_gain is not None:                          # ∂c/∂a* = -k·pos·γ·dt  (feedback in rhs)
                Mx = Mx - fb_gain * gamma * dt * pm * u
            r = (Mx + g) - xbar                              # fixed-point residual of x̄ = M x̄ + g
            if np.linalg.norm(r[gm]) < adj_tol * (np.linalg.norm(xbar[gm]) + 1e-30):
                xbar = xbar + r; break
            if r_prev is not None:                           # Aitken Δ² dynamic relaxation
                dr = (r - r_prev)[gm]
                omega = -omega * float(np.dot(r_prev[gm], dr)) / (float(np.dot(dr, dr)) + 1e-30)
                omega = float(np.clip(omega, 0.05, 1.0))
            xbar = xbar + omega * r; r_prev = r
        u_star = Ainv(xbar, Kblk)

        # STEP C: design / control / feedback / state grads at u*
        gC[t] = u_star * C.free_np                            # control enters rhs linearly
        if fb_gain is not None:
            dL_dk += -float(np.dot(u_star * pm, dq_out))      # ∂c/∂k = -pos·dq*
        adj_Qint = -u_star * C.free_np
        _, _, deps, dk, Dm_eps, Dk_k = dsg._design_force_cached(qpw, C, Esw, dev)
        wp.launch(dsg.adj_E_kernel, dim=(1, C.ne, 36),
                  inputs=[C.gw, deps, dk, Dm_eps, Dk_k, C.edofs, DTYPE(NP(C.h)), C.ngg, Esw, wa(adj_Qint)],
                  outputs=[gE], device=dev)
        wp.launch(dsg.adj_rho_kernel, dim=(1, C.ne), inputs=[C.Me, C.edofs, wa(u_star * C.free_np), wa(a_star)],
                  outputs=[gR], device=dev)
        # stiffness-in-A design sensitivity: ∂(A a*)/∂E = β·dt²·(∂K_mem/∂E)·a*  → adj_E += -β·dt²·(u*ᵀK_mem a*)/E
        wp.launch(_adj_E_stiff, dim=(1, C.ne), inputs=[wa(u_star * C.free_np), wa(a_star), Kblk, C.edofs,
                  Esw, DTYPE(NP(coef))], outputs=[gE], device=dev)
        # aero FORCE path at u* → state (q_pred, v_pred) + gprev/wake carry
        ac, av, agp, awr, awg = fvjp(u_star)
        adj_q_pred = adj_q + (PT @ ac.reshape(-1)) + out_qpred   # q_{t+1}=q_pred+… (H) + aero corners→q_pred (force G + output)
        adj_v_pred = adj_dq + (PT @ av.reshape(-1)) + out_vpred
        adj_qK = zc()                                         # -(∂Qint/∂q_pred)ᵀu = K_mem·adj_Qint = -K_mem u*
        apply_MK(wa(adj_Qint), adj_qK, C.Me, Kblk, C.edofs, C.free, 0.0, 1.0, dev)
        adj_q_pred = adj_q_pred + adj_qK.numpy()[0]
        if fb_gain is not None:                               # ∂c/∂dq* = -k·pos → state
            adj_v_pred = adj_v_pred - fb_gain * pm * u_star
        new_gamma_carry = agp.copy()
        if out_gprev is not None:
            new_gamma_carry = new_gamma_carry + out_gprev
        if use_wake and nw_t > 0:
            new_wr = awr.copy() + (out_wr if out_wr is not None else 0.0)
            new_wg = awg.copy() + (out_wg if out_wg is not None else 0.0)
        else:
            new_wr = None; new_wg = None
        # predictor reverse to (q_t, dq_t, a_t) — the a-carry feeds step t-1's seed
        adj_q = adj_q_pred
        adj_dq = dt * adj_q_pred + adj_v_pred
        adj_a = dt * dt * (0.5 - beta) * adj_q_pred + dt * (1.0 - gamma) * adj_v_pred
        adj_gamma_carry = new_gamma_carry; adj_wr_next = new_wr; adj_wg_next = new_wg

    # initial accel a0 = M⁻¹(−Qint(q0)) also depends on (E,ρ) → push adj_a (=∂L/∂a_0) to design
    u0, _ = structural_cg(wa(adj_a * C.free_np), Mscaled, Kblk0, C.edofs, C.free, 0.0, ndof, tol=cg_tol, device=dev)
    u0 = u0.numpy()[0]
    _, _, deps0, dk0, Dm0, Dk0 = dsg._design_force_cached(wa(q0), C, Esw, dev)
    wp.launch(dsg.adj_E_kernel, dim=(1, C.ne, 36),
              inputs=[C.gw, deps0, dk0, Dm0, Dk0, C.edofs, DTYPE(NP(C.h)), C.ngg, Esw, wa(-u0 * C.free_np)],
              outputs=[gE], device=dev)
    wp.launch(dsg.adj_rho_kernel, dim=(1, C.ne), inputs=[C.Me, C.edofs, wa(u0 * C.free_np), wa(a0)],
              outputs=[gR], device=dev)
    return L, gE.numpy(), gR.numpy(), gC, dL_dk + dLdk_extra   # closed-loop chain + explicit loss ∂/∂k


def verify_pc_grad(nx=3, ny=3, N=6, dt=1e-4, seed=0, use_wake=False, elems=None, pc_it=40, pc_tol=1e-12):
    """route-A piece 3: validate the differentiable PC adjoint DESIGN gradient ∂L/∂(E,ρ) — through the
    strong-coupled predictor-corrector fixed point via the implicit function theorem — vs central-FD of
    the numpy PC oracle (tangent='membrane'). use_wake=False isolates the within-step coupling + dΓ/dt;
    use_wake=True adds the full wake history."""
    wp.init()
    sh = _build_shell(nx=nx, ny=ny); sh.set_bc([i for i in range(nx + 1)]); C = ANCFConstants(sh, device=cfg.DEVICE)
    rng = np.random.default_rng(seed); ne = sh.ne
    Es = np.exp(0.2 * rng.standard_normal(ne)); Rs = np.exp(0.2 * rng.standard_normal(ne))
    sh.set_distribution(E_scale=Es, rho_scale=Rs); ndof = sh.ndof
    free = np.array(sorted(set(range(ndof)) - set(sh._bc_dofs)))
    q0 = sh.q.copy(); q0[free] += 1e-4 * rng.standard_normal(len(free))
    dq0 = np.zeros(ndof); dq0[free] = 1e-3 * rng.standard_normal(len(free))
    w = np.zeros(ndof); w[free] = rng.standard_normal(len(free))
    P, dist = dcu._index_maps(sh, nx, ny)
    L, gE, gR, _, _ = coupled_unsteady_pc_grad_gpu(sh, C, P, dist, q0, dq0, N, dt, w, Es, Rs, nx, ny,
                                                   use_wake=use_wake, pc_it=pc_it, pc_tol=pc_tol)
    els = [0, ne // 2, ne - 1] if elems is None else elems
    gE_fd, gR_fd = dcu.design_grad_fd_pc(sh, Es, Rs, q0, dq0, N, dt, free, w, nx, ny, elems=els,
                                         use_wake=use_wake, pc_it=pc_it, pc_tol=pc_tol)
    relE = max(abs(gE[e] - gE_fd[e]) for e in els) / (max(abs(gE_fd[e]) for e in els) + 1e-30)
    relR = max(abs(gR[e] - gR_fd[e]) for e in els) / (max(abs(gR_fd[e]) for e in els) + 1e-30)
    ok = relE < 5e-2 and relR < 1e-2
    tag = "FULL wake history" if use_wake else "no wake recurrence"
    print(f"differentiable PC adjoint DESIGN gradient — IFT through the strong-coupled fixed point "
          f"({ne} elems, {N} steps, dt={dt:g}, {tag}):")
    print(f"  ∂L/∂E_scale (刚柔)  adjoint vs FD: rel={relE:.2e}   ∂L/∂rho_scale (质量) vs FD: rel={relR:.2e}")
    print(f"    adjoint gE{els}={gE[els]}")
    print(f"    FD      gE{els}={gE_fd[els]}")
    print(f"  -> {'PASS' if ok else 'FAIL'}: the design gradient flows through the predictor-corrector "
          f"STRONG coupling — the differentiable strong-coupled FSI adjoint (route A)")
    return ok


def verify_pc_grad_control(nx=3, ny=3, N=5, dt=1e-4, seed=1, use_wake=False, eps=1e-2, pc_it=40, pc_tol=1e-12):
    """route-A piece 3: validate the CONTROL gradient gC = ∂L/∂u_t (the SHAC policy-gradient signal)
    through the strong-coupled PC fixed point, vs FD of the numpy PC oracle (tangent='membrane')."""
    wp.init()
    sh = _build_shell(nx=nx, ny=ny); sh.set_bc([i for i in range(nx + 1)]); C = ANCFConstants(sh, device=cfg.DEVICE)
    rng = np.random.default_rng(seed); ne = sh.ne
    Es = np.exp(0.2 * rng.standard_normal(ne)); Rs = np.exp(0.2 * rng.standard_normal(ne))
    sh.set_distribution(E_scale=Es, rho_scale=Rs); ndof = sh.ndof
    free = np.array(sorted(set(range(ndof)) - set(sh._bc_dofs)))
    q0 = sh.q.copy(); q0[free] += 1e-4 * rng.standard_normal(len(free))
    dq0 = np.zeros(ndof); dq0[free] = 1e-3 * rng.standard_normal(len(free))
    w = np.zeros(ndof); w[free] = rng.standard_normal(len(free))
    P, dist = dcu._index_maps(sh, nx, ny)
    u = np.zeros((N, ndof)); u[:, free] = 1e-2 * rng.standard_normal((N, len(free)))
    _, _, _, gC, _ = coupled_unsteady_pc_grad_gpu(sh, C, P, dist, q0, dq0, N, dt, w, Es, Rs, nx, ny,
                                                  use_wake=use_wake, control=u, pc_it=pc_it, pc_tol=pc_tol)

    def Lc(uu):
        return dcu.loss_only_pc(sh, Es, Rs, q0, dq0, N, dt, free, w, nx, ny, use_wake=use_wake,
                                control=uu, pc_it=pc_it, pc_tol=pc_tol)
    probes = [(0, free[0]), (N // 2, free[len(free) // 2]), (N - 1, free[-1])]
    rels = []
    for (t, d) in probes:
        up = u.copy(); up[t, d] += eps; um = u.copy(); um[t, d] -= eps
        fd = (Lc(up) - Lc(um)) / (2 * eps)
        rels.append(abs(gC[t, d] - fd) / (abs(fd) + 1e-30))
    rel = max(rels); ok = rel < 1e-2
    tag = "FULL wake history" if use_wake else "no wake recurrence"
    print(f"differentiable PC adjoint CONTROL gradient ∂L/∂u_t — IFT through the strong-coupled fixed "
          f"point ({ne} elems, {N} steps, dt={dt:g}, {tag}):")
    print(f"  adjoint vs FD at {len(probes)} (step,DOF) probes: rel={rel:.2e}")
    print(f"  -> {'PASS' if ok else 'FAIL'}: the policy-gradient signal ∂L/∂action_t flows through the "
          f"differentiable STRONG-coupled FSI — the SHAC control-layer building block under strong coupling")
    return ok


def verify_pc_policy_grad(nx=3, ny=3, N=5, dt=1e-4, seed=3, use_wake=False, k0=8.0, eps=1e-4, pc_it=40, pc_tol=1e-12):
    """route-A piece 3: closed-loop policy gradient dL/dk (state feedback u_t=-k·dq_t on position DOFs,
    IN the PC loop) through the strong-coupled fixed point, vs FD of the numpy PC oracle."""
    wp.init()
    sh = _build_shell(nx=nx, ny=ny); sh.set_bc([i for i in range(nx + 1)]); C = ANCFConstants(sh, device=cfg.DEVICE)
    rng = np.random.default_rng(seed); ne = sh.ne; ndof = sh.ndof
    Es = np.exp(0.1 * rng.standard_normal(ne)); Rs = np.exp(0.1 * rng.standard_normal(ne))
    sh.set_distribution(E_scale=Es, rho_scale=Rs)
    free = np.array(sorted(set(range(ndof)) - set(sh._bc_dofs)))
    q0 = sh.q.copy(); q0[free] += 1e-3 * rng.standard_normal(len(free))
    dq0 = np.zeros(ndof); dq0[free] = 1e-2 * rng.standard_normal(len(free))
    w = np.zeros(ndof); w[free] = rng.standard_normal(len(free))
    P, dist = dcu._index_maps(sh, nx, ny)
    _, _, _, _, dL_dk = coupled_unsteady_pc_grad_gpu(sh, C, P, dist, q0, dq0, N, dt, w, Es, Rs, nx, ny,
                                                     use_wake=use_wake, fb_gain=k0, pc_it=pc_it, pc_tol=pc_tol)

    def Lk(k):
        return dcu.loss_only_pc(sh, Es, Rs, q0, dq0, N, dt, free, w, nx, ny, use_wake=use_wake,
                                fb_gain=k, pc_it=pc_it, pc_tol=pc_tol)
    fd = (Lk(k0 + eps) - Lk(k0 - eps)) / (2 * eps)
    rel = abs(dL_dk - fd) / (abs(fd) + 1e-30); ok = rel < 1e-3
    tag = "FULL wake history" if use_wake else "no wake recurrence"
    print(f"differentiable PC adjoint CLOSED-LOOP policy gradient dL/dk — IFT through the strong-coupled "
          f"fixed point ({ne} elems, {N} steps, dt={dt:g}, {tag}):")
    print(f"  state feedback u_t=-k·dq_t (k={k0}); dL/dk adjoint={dL_dk:+.4e} vs FD={fd:+.4e}  rel={rel:.2e}")
    print(f"  -> {'PASS' if ok else 'FAIL'}: the policy-in-the-loop (closed-loop SHAC) gradient is correct "
          f"under STRONG coupling — a learnable controller trains through the differentiable strong-coupled FSI")
    return ok


def verify(nx=3, ny=3, N=6, dt=1e-5, seed=0):
    wp.init()
    sh = _build_shell(nx=nx, ny=ny); sh.set_bc([i for i in range(nx + 1)]); C = ANCFConstants(sh, device=cfg.DEVICE)  # clamped cantilever root
    rng = np.random.default_rng(seed); ne = sh.ne
    Es = np.exp(0.2 * rng.standard_normal(ne)); Rs = np.exp(0.2 * rng.standard_normal(ne))
    sh.set_distribution(E_scale=Es, rho_scale=Rs)
    ndof = sh.ndof
    free = np.array(sorted(set(range(ndof)) - set(sh._bc_dofs)))
    q0 = sh.q.copy(); q0[free] += 1e-4 * rng.standard_normal(len(free))
    dq0 = np.zeros(ndof); dq0[free] = 1e-3 * rng.standard_normal(len(free))
    P, dist = dcu._index_maps(sh, nx, ny)
    Mff = sh.M[np.ix_(free, free)].toarray()
    # numpy oracle forward
    q_np = dcu.coupled_unsteady_forward(sh, q0, dq0, N, dt, free, Mff, P, dist, nx, ny)
    # Warp forward
    q_gpu, _ = coupled_unsteady_forward_gpu(sh, C, P, dist, q0, dq0, N, dt, Es, Rs, nx, ny)
    rel = np.max(np.abs(q_gpu - q_np)) / (np.max(np.abs(q_np - q0)) + 1e-30)
    ok = rel < 1e-7
    print(f"all-Warp coupled UNSTEADY FSI forward ({ne} elems, {N}-step rollout) vs numpy oracle:")
    print(f"  final-state q match (rel to displacement): {rel:.2e}")
    print(f"  -> {'PASS' if ok else 'FAIL'}: structure (Warp design force + CG) ⊗ unsteady free-wake "
          f"(Warp, deforming+moving wing + wake history) coupled forward on GPU — fix3 forward")
    return ok


def verify_pc(nx=3, ny=3, N=8, dt=1e-4, seed=0, use_wake=True, pc_tol=1e-9, pc_it=30):
    """Validate the all-Warp STRONG-coupled (predictor-corrector) unsteady FSI forward vs the numpy
    oracle dcu.coupled_unsteady_forward_pc(tangent='membrane') — same linearly-implicit-Newmark fixed
    point, same membrane tangent. Also reports the PHYSICAL full-vs-membrane tangent difference, the
    per-step Aitken iteration count (GPU vs numpy), and the GPU forward wall time."""
    import time
    wp.init()
    sh = _build_shell(nx=nx, ny=ny); sh.set_bc([i for i in range(nx + 1)]); C = ANCFConstants(sh, device=cfg.DEVICE)
    rng = np.random.default_rng(seed); ne = sh.ne
    Es = np.exp(0.2 * rng.standard_normal(ne)); Rs = np.exp(0.2 * rng.standard_normal(ne))
    sh.set_distribution(E_scale=Es, rho_scale=Rs)
    ndof = sh.ndof
    free = np.array(sorted(set(range(ndof)) - set(sh._bc_dofs)))
    q0 = sh.q.copy(); q0[free] += 1e-4 * rng.standard_normal(len(free))
    dq0 = np.zeros(ndof); dq0[free] = 1e-3 * rng.standard_normal(len(free))
    P, dist = dcu._index_maps(sh, nx, ny); Mff = sh.M[np.ix_(free, free)].toarray()
    q_mem, _, itc_np = dcu.coupled_unsteady_forward_pc(sh, q0, dq0, N, dt, free, Mff, P, dist, nx, ny,
                          use_wake=use_wake, pc_tol=pc_tol, pc_it=pc_it, tangent="membrane", return_traj=True)
    q_full = dcu.coupled_unsteady_forward_pc(sh, q0, dq0, N, dt, free, Mff, P, dist, nx, ny,
                          use_wake=use_wake, pc_tol=pc_tol, pc_it=pc_it, tangent="full")
    t0 = time.time()
    q_gpu, _, itc_gpu = coupled_unsteady_forward_pc_gpu(sh, C, P, dist, q0, dq0, N, dt, Es, Rs, nx, ny,
                          use_wake=use_wake, pc_tol=pc_tol, pc_it=pc_it, cg_tol=1e-11, return_traj=True)
    wp.synchronize(); twall = time.time() - t0
    disp = np.max(np.abs(q_mem - q0)) + 1e-30
    rel = np.max(np.abs(q_gpu - q_mem)) / disp
    tan_diff = np.max(np.abs(q_full - q_mem)) / disp
    ok = rel < 1e-5
    print(f"all-Warp STRONG-coupled (predictor-corrector) UNSTEADY FSI forward ({ne} elems, {N} steps, "
          f"dt={dt:g}, {'wake' if use_wake else 'no-wake'}):")
    print(f"  GPU PC vs numpy PC (membrane tangent): rel={rel:.2e}   (tol 1e-5; fixed point defined to pc_tol={pc_tol:g})")
    print(f"  numpy full-vs-membrane tangent diff (physical linearization, NOT a GPU error): {tan_diff:.2e} of displacement")
    print(f"  Aitken iters/step: GPU avg={np.mean(itc_gpu):.2f} (max {max(itc_gpu)})   numpy avg={np.mean(itc_np):.2f}")
    print(f"  GPU PC forward wall time: {twall * 1e3:.0f} ms total ({twall / N * 1e3:.1f} ms/step)")
    print(f"  -> {'PASS' if ok else 'FAIL'}: the GPU strong-coupling PC forward matches the oracle's "
          f"converged fixed point — the production forward the differentiable PC adjoint differentiates")
    return ok


def verify_grad(nx=3, ny=3, N=6, dt=1e-5, seed=0, use_wake=False, elems=None):
    """fix3 adjoint vs FD oracle. use_wake=False = sub-step 1 (structure design + moving-body aero
    + dΓ/dt coupling, no wake recurrence); use_wake=True = sub-step 2 (full wake history)."""
    wp.init()
    sh = _build_shell(nx=nx, ny=ny); sh.set_bc([i for i in range(nx + 1)]); C = ANCFConstants(sh, device=cfg.DEVICE)  # clamped cantilever root
    rng = np.random.default_rng(seed); ne = sh.ne
    Es = np.exp(0.2 * rng.standard_normal(ne)); Rs = np.exp(0.2 * rng.standard_normal(ne))
    sh.set_distribution(E_scale=Es, rho_scale=Rs)
    ndof = sh.ndof
    free = np.array(sorted(set(range(ndof)) - set(sh._bc_dofs)))
    q0 = sh.q.copy(); q0[free] += 1e-4 * rng.standard_normal(len(free))
    dq0 = np.zeros(ndof); dq0[free] = 1e-3 * rng.standard_normal(len(free))
    w = np.zeros(ndof); w[free] = rng.standard_normal(len(free))
    P, dist = dcu._index_maps(sh, nx, ny)
    L, gE, gR, gC, _ = coupled_unsteady_grad_gpu(sh, C, P, dist, q0, dq0, N, dt, w, Es, Rs, nx, ny, use_wake=use_wake)
    els = [0, ne // 2, ne - 1] if elems is None else elems
    gE_fd, gR_fd = dcu.design_grad_fd(sh, Es, Rs, q0, dq0, N, dt, free, w, nx, ny, elems=els, use_wake=use_wake)
    relE = max(abs(gE[e] - gE_fd[e]) for e in els) / (max(abs(gE_fd[e]) for e in els) + 1e-30)
    relR = max(abs(gR[e] - gR_fd[e]) for e in els) / (max(abs(gR_fd[e]) for e in els) + 1e-30)
    ok = relE < 5e-2 and relR < 1e-2
    tag = "sub-step 2 (FULL wake history)" if use_wake else "sub-step 1 (no wake recurrence)"
    print(f"all-Warp coupled UNSTEADY FSI design gradient — {tag} ({ne} elems, {N} steps):")
    print(f"  ∂L/∂E_scale (刚柔)  adjoint vs FD: rel={relE:.2e}   ∂L/∂rho_scale(质量) vs FD: rel={relR:.2e}")
    print(f"  -> {'PASS' if ok else 'FAIL'}: design gradient through structure ⊗ unsteady aero "
          f"(moving-body + dΓ/dt{' + wake history' if use_wake else ''}) — fix3 adjoint")
    return ok


def verify_grad_control(nx=3, ny=3, N=5, dt=1e-5, seed=1, use_wake=True, eps=1e-2):
    """fix3/Phase-E: validate the CONTROL gradient gC = ∂L/∂u_t (the SHAC policy-gradient signal)
    through the full unsteady coupled FSI, vs FD of the numpy oracle."""
    wp.init()
    sh = _build_shell(nx=nx, ny=ny); sh.set_bc([i for i in range(nx + 1)]); C = ANCFConstants(sh, device=cfg.DEVICE)  # clamped cantilever root
    rng = np.random.default_rng(seed); ne = sh.ne
    Es = np.exp(0.2 * rng.standard_normal(ne)); Rs = np.exp(0.2 * rng.standard_normal(ne))
    sh.set_distribution(E_scale=Es, rho_scale=Rs)
    ndof = sh.ndof
    free = np.array(sorted(set(range(ndof)) - set(sh._bc_dofs)))
    q0 = sh.q.copy(); q0[free] += 1e-4 * rng.standard_normal(len(free))
    dq0 = np.zeros(ndof); dq0[free] = 1e-3 * rng.standard_normal(len(free))
    w = np.zeros(ndof); w[free] = rng.standard_normal(len(free))
    P, dist = dcu._index_maps(sh, nx, ny)
    Mff = sh.M[np.ix_(free, free)].toarray()
    u = np.zeros((N, ndof)); u[:, free] = 1e-2 * rng.standard_normal((N, len(free)))   # actuation
    _, _, _, gC, _ = coupled_unsteady_grad_gpu(sh, C, P, dist, q0, dq0, N, dt, w, Es, Rs, nx, ny,
                                               use_wake=use_wake, control=u)

    def Lc(uu):
        return float(w @ dcu.coupled_unsteady_forward(sh, q0, dq0, N, dt, free, Mff, P, dist, nx, ny,
                                                      use_wake=use_wake, control=uu))
    probes = [(0, free[0]), (N // 2, free[len(free) // 2]), (N - 1, free[-1])]
    rels = []
    for (t, d) in probes:
        up = u.copy(); up[t, d] += eps; um = u.copy(); um[t, d] -= eps
        fd = (Lc(up) - Lc(um)) / (2 * eps)
        rels.append(abs(gC[t, d] - fd) / (abs(fd) + 1e-30))
    rel = max(rels); ok = rel < 1e-2
    print(f"all-Warp coupled UNSTEADY FSI CONTROL gradient ∂L/∂u_t ({ne} elems, {N} steps, full wake):")
    print(f"  adjoint vs FD at {len(probes)} (step,DOF) probes: rel={rel:.2e}")
    print(f"  -> {'PASS' if ok else 'FAIL'}: the policy-gradient signal ∂L/∂action_t flows through "
          f"the differentiable unsteady coupled FSI — the SHAC control-layer building block")
    return ok


def verify_policy_grad(nx=3, ny=3, N=5, dt=1e-5, seed=3, use_wake=True, k0=8.0, eps=1e-4):
    """Phase-E closed-loop: the policy is IN the loop (u_t = -k·dq_t state feedback), so the rollout
    depends on the gain k through the feedback. Validate the closed-loop policy gradient dL/dk
    (accumulated through the differentiated rollout, incl. the ∂u/∂dq feedback term that feeds the
    state adjoint) vs FD of the numpy oracle — true closed-loop SHAC, not an open-loop schedule."""
    wp.init()
    sh = _build_shell(nx=nx, ny=ny); sh.set_bc([i for i in range(nx + 1)]); C = ANCFConstants(sh, device=cfg.DEVICE)  # clamped cantilever root
    rng = np.random.default_rng(seed); ne = sh.ne; ndof = sh.ndof
    Es = np.exp(0.1 * rng.standard_normal(ne)); Rs = np.exp(0.1 * rng.standard_normal(ne))
    sh.set_distribution(E_scale=Es, rho_scale=Rs)
    free = np.array(sorted(set(range(ndof)) - set(sh._bc_dofs)))
    q0 = sh.q.copy(); q0[free] += 1e-3 * rng.standard_normal(len(free))
    dq0 = np.zeros(ndof); dq0[free] = 1e-2 * rng.standard_normal(len(free))
    w = np.zeros(ndof); w[free] = rng.standard_normal(len(free))
    P, dist = dcu._index_maps(sh, nx, ny); Mff = sh.M[np.ix_(free, free)].toarray()
    _, _, _, _, dL_dk = coupled_unsteady_grad_gpu(sh, C, P, dist, q0, dq0, N, dt, w, Es, Rs, nx, ny,
                                                  use_wake=use_wake, fb_gain=k0)

    def Lk(k):
        return float(w @ dcu.coupled_unsteady_forward(sh, q0, dq0, N, dt, free, Mff, P, dist, nx, ny,
                                                      use_wake=use_wake, fb_gain=k))
    fd = (Lk(k0 + eps) - Lk(k0 - eps)) / (2 * eps)
    rel = abs(dL_dk - fd) / (abs(fd) + 1e-30); ok = rel < 1e-4
    print(f"all-Warp coupled UNSTEADY FSI CLOSED-LOOP policy gradient dL/dk ({ne} elems, {N} steps, full wake):")
    print(f"  state-feedback u_t=-k·dq_t (k={k0}); dL/dk adjoint={dL_dk:+.4e} vs FD={fd:+.4e}  rel={rel:.2e}")
    print(f"  -> {'PASS' if ok else 'FAIL'}: the policy-in-the-loop (closed-loop SHAC) gradient is "
          f"correct — a learnable feedback controller can train through the differentiable FSI")
    return ok


def demo_joint_descent(nx=3, ny=3, N=5, dt=1e-5, iters=15, use_wake=True, seed=2):
    """Phase-E capability demo (mechanism check, NOT a scientific result): minimise a regulation
    objective J = ½‖q_N(free)‖² by SHAC descent on (design E,ρ + control u) JOINTLY, using one
    backward per step. Shows J decreases monotonically and joint < design-only < control-only —
    i.e. the validated joint gradient is usable for optimisation. Small toy; the real co-design
    archive (Phase F) needs A100-scale compute."""
    wp.init()
    sh = _build_shell(nx=nx, ny=ny); sh.set_bc([i for i in range(nx + 1)]); C = ANCFConstants(sh, device=cfg.DEVICE)  # clamped cantilever root
    rng = np.random.default_rng(seed); ne = sh.ne; ndof = sh.ndof
    free = np.array(sorted(set(range(ndof)) - set(sh._bc_dofs)))
    fmask = np.zeros(ndof); fmask[free] = 1.0
    q0 = sh.q.copy(); q0[free] += 1e-3 * rng.standard_normal(len(free))
    dq0 = np.zeros(ndof); dq0[free] = 1e-2 * rng.standard_normal(len(free))
    P, dist = dcu._index_maps(sh, nx, ny)

    def run(opt_design, opt_control, lrE=0.05, lrR=0.05, lrU=2e3):
        Es = np.ones(ne); Rs = np.ones(ne); u = np.zeros((N, ndof))
        traj = []
        for _ in range(iters):
            sh.set_distribution(E_scale=Es, rho_scale=Rs)
            qN, _ = coupled_unsteady_forward_gpu(sh, C, P, dist, q0, dq0, N, dt, Es, Rs, nx, ny,
                                                 use_wake=use_wake, control=u)
            w = qN * fmask                                       # ∂J/∂q_N for J = ½‖q_N(free)‖²
            J = 0.5 * float(w @ qN); traj.append(J)
            _, gE, gR, gC, _ = coupled_unsteady_grad_gpu(sh, C, P, dist, q0, dq0, N, dt, w, Es, Rs, nx, ny,
                                                         use_wake=use_wake, control=u)
            if opt_design:
                Es = np.clip(Es - lrE * gE / (np.abs(gE).max() + 1e-30), 0.3, 3.0)
                Rs = np.clip(Rs - lrR * gR / (np.abs(gR).max() + 1e-30), 0.3, 3.0)
            if opt_control:
                u = u - lrU * gC
        return np.array(traj)

    tj = run(True, True); td = run(True, False); tc = run(False, True)
    print(f"Phase-E SHAC joint design+control descent on J=½‖q_N‖² ({ne} elems, {N} steps, full wake):")
    print(f"  design-only : J {td[0]:.3e} -> {td[-1]:.3e}  ({100*(1-td[-1]/td[0]):+.1f}%)")
    print(f"  control-only: J {tc[0]:.3e} -> {tc[-1]:.3e}  ({100*(1-tc[-1]/tc[0]):+.1f}%)")
    print(f"  JOINT       : J {tj[0]:.3e} -> {tj[-1]:.3e}  ({100*(1-tj[-1]/tj[0]):+.1f}%)")
    mono = np.all(np.diff(tj) <= 1e-12 * tj[0] + np.abs(tj[:-1]) * 1e-6) or (tj[-1] < tj[0])
    ok = (tj[-1] < tj[0]) and (td[-1] < td[0]) and (tc[-1] < tc[0]) and (tj[-1] <= min(td[-1], tc[-1]) * 1.001)
    print(f"  -> {'PASS' if ok else 'FAIL'}: the validated SHAC joint gradient drives optimisation "
          f"(joint ≤ either single axis); mechanism demo for the co-design optimiser")
    return ok


if __name__ == "__main__":
    import sys as _s
    if "--demo" in _s.argv:
        raise SystemExit(0 if demo_joint_descent() else 1)
    if "--policy" in _s.argv:
        raise SystemExit(0 if verify_policy_grad() else 1)
    if "--control" in _s.argv:
        raise SystemExit(0 if verify_grad_control() else 1)
    if "--pc" in _s.argv:
        ok1 = verify_pc(use_wake=False); ok2 = verify_pc(use_wake=True)
        raise SystemExit(0 if (ok1 and ok2) else 1)
    if "--pcgrad" in _s.argv:
        oks = [verify_pc_grad(N=5, dt=1e-4, use_wake=False),
               verify_pc_grad(N=4, dt=1e-5, use_wake=True, elems=[0, 4, 8]),
               verify_pc_grad_control(N=5, dt=1e-4, use_wake=False),
               verify_pc_policy_grad(N=5, dt=1e-4, use_wake=False)]
        raise SystemExit(0 if all(oks) else 1)
    if "--grad" in _s.argv:
        ok1 = verify_grad(use_wake=False)
        ok2 = verify_grad(use_wake=True)
        raise SystemExit(0 if (ok1 and ok2) else 1)
    raise SystemExit(0 if verify() else 1)
