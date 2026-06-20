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

    def forward(self, corners_np, cvel_np, wr, wg, nw, gprev_np):
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
        return self.Fp.numpy()

    def backward(self, adj_Fp_np, adj_gamma_extra=None):
        dev = self.dev; npan = self.npan
        for a in (self.cw, self.vw, self.gprev, self.gamma, self.AIC, self.rhs):
            a.grad.zero_()
        if self.nw > 0:
            self.wr.grad.zero_(); self.wg.grad.zero_()
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
        self.t1.zero(); self.t2.zero()                          # zero AFTER reading (zero() clears grads)
        return out


def coupled_unsteady_forward_gpu(sh, C, P, dist, q0, dq0, N, dt, Es, Rs, nx, ny,
                                 Vinf=VINF, cg_tol=1e-12, use_wake=True):
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
    for _ in range(N):
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
        rhs_s = (Fnodal - Qint) * C.free_np
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


def coupled_unsteady_grad_gpu(sh, C, P, dist, q0, dq0, N, dt, w, Es, Rs, nx, ny,
                              Vinf=VINF, use_wake=False, cg_tol=1e-12):
    """all-Warp coupled UNSTEADY FSI design gradient ∂L/∂(E,ρ). Chains: structure design adjoint
    (adj_E/adj_rho + membrane-K_t state chain) ⊗ unsteady aero VJP (UnsteadyAeroGpu: ∂F/∂corners,
    ∂F/∂V_body, ∂F/∂gprev) through the coupled recurrence. adj_q gets the aero ∂/∂corners (P^T) and
    the structure chain; adj_dq gets the moving-body ∂/∂V_body (P^T); the dΓ/dt coupling carries
    adj_gprev → gamma_{t-1}. use_wake=False isolates this (no wake recurrence)."""
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
    wr = wp.zeros((maxw, 4), dtype=V3, device=dev, requires_grad=True)
    wr_new = wp.zeros((maxw, 4), dtype=V3, device=dev)
    wg = wp.zeros(maxw, dtype=DTYPE, device=dev, requires_grad=True)
    q = q0.copy(); dq = dq0.copy(); nw = 0
    qs = []; dqs = []; araws = []; gammas = []; wake_snaps = []
    gprev_np = np.zeros((1, npan))
    for _ in range(N):
        corners = (P @ q).reshape(ncv, 3); cvel = (P @ dq).reshape(ncv, 3)
        wake_snaps.append((wr.numpy().copy(), wg.numpy().copy(), nw))
        Fp = aero.forward(corners, cvel, wr if nw > 0 else dummy_wr, wg if nw > 0 else dummy_wg, nw, gprev_np)
        gamma_np = aero.gamma.numpy()
        Fnodal = dist @ Fp.reshape(-1)
        Qmem, Qbend = dsg.design_internal_force(wa(q), C, Esw, dev)
        Qint = Qmem.numpy()[0] + Qbend.numpy()[0]
        a, _ = structural_cg(wa((Fnodal - Qint) * C.free_np), Mscaled, Kblk0, C.edofs, C.free, 0.0,
                             ndof, tol=cg_tol, device=dev)
        a_np = a.numpy()[0]
        qs.append(q.copy()); dqs.append(dq.copy()); araws.append(a_np.copy()); gammas.append(gamma_np.copy())
        dq = dq + dt * a_np; q = q + dt * dq
        if use_wake:
            wp.launch(ug.shed_kernel, dim=ny, inputs=[aero.rings_fwd, aero.gamma, te, Vw, DTYPE(dt), nw],
                      outputs=[wr, wg], device=dev)
            nw += ny
            wp.launch(ug.convect_kernel, dim=(nw, 4), inputs=[aero.rings_fwd, aero.gamma, npan, wr, wg, nw, Vw, DTYPE(dt)],
                      outputs=[wr_new], device=dev)
            wp.copy(wr, wr_new, count=nw * 4)
        gprev_np = gamma_np
    L = float(w @ q)

    # ---- backward ----
    gE = wp.zeros(C.ne, dtype=DTYPE, device=dev); gR = wp.zeros(C.ne, dtype=DTYPE, device=dev)
    adj_q = w.copy(); adj_dq = np.zeros(ndof); adj_gamma_carry = None
    for t in reversed(range(N)):
        aq1 = adj_q; ad1 = adj_dq + dt * aq1; adj_a = dt * ad1
        adj_rhs_w, _ = structural_cg(wa(adj_a * C.free_np), Mscaled, Kblk0, C.edofs, C.free, 0.0,
                                     ndof, tol=cg_tol, device=dev)
        adj_rhs = adj_rhs_w.numpy()[0]
        corners_t = (P @ qs[t]).reshape(ncv, 3); cvel_t = (P @ dqs[t]).reshape(ncv, 3)
        gprev_t = gammas[t - 1] if t > 0 else np.zeros((1, npan))
        wr_np, wg_np, nw_t = wake_snaps[t]
        wr_t = wp.array(wr_np, dtype=V3, device=dev, requires_grad=True) if nw_t > 0 else dummy_wr
        wg_t = wp.array(wg_np, dtype=DTYPE, device=dev, requires_grad=True) if nw_t > 0 else dummy_wg
        aero.forward(corners_t, cvel_t, wr_t, wg_t, nw_t, gprev_t)
        adj_Fp = (dist.T @ adj_rhs).reshape(npan, 3)
        adj_corners, adj_cvel, adj_gprev, adj_wr, adj_wg = aero.backward(adj_Fp, adj_gamma_carry)
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
        adj_q = aq1 + adj_qK.numpy()[0] + adj_q_aero
        adj_dq = ad1 + adj_dq_aero
        adj_gamma_carry = adj_gprev                            # dΓ/dt coupling: adj_gprev → gamma_{t-1}
    return L, gE.numpy(), gR.numpy()


def verify(nx=3, ny=3, N=6, dt=1e-5, seed=0):
    wp.init()
    sh = _build_shell(nx=nx, ny=ny); C = ANCFConstants(sh, device=cfg.DEVICE)
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


def verify_grad(nx=3, ny=3, N=6, dt=1e-5, seed=0, use_wake=False, elems=None):
    """fix3 adjoint vs FD oracle. use_wake=False = sub-step 1 (structure design + moving-body aero
    + dΓ/dt coupling, no wake recurrence); use_wake=True = sub-step 2 (full wake history)."""
    wp.init()
    sh = _build_shell(nx=nx, ny=ny); C = ANCFConstants(sh, device=cfg.DEVICE)
    rng = np.random.default_rng(seed); ne = sh.ne
    Es = np.exp(0.2 * rng.standard_normal(ne)); Rs = np.exp(0.2 * rng.standard_normal(ne))
    sh.set_distribution(E_scale=Es, rho_scale=Rs)
    ndof = sh.ndof
    free = np.array(sorted(set(range(ndof)) - set(sh._bc_dofs)))
    q0 = sh.q.copy(); q0[free] += 1e-4 * rng.standard_normal(len(free))
    dq0 = np.zeros(ndof); dq0[free] = 1e-3 * rng.standard_normal(len(free))
    w = np.zeros(ndof); w[free] = rng.standard_normal(len(free))
    P, dist = dcu._index_maps(sh, nx, ny)
    L, gE, gR = coupled_unsteady_grad_gpu(sh, C, P, dist, q0, dq0, N, dt, w, Es, Rs, nx, ny, use_wake=use_wake)
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


if __name__ == "__main__":
    import sys as _s
    if "--grad" in _s.argv:
        raise SystemExit(0 if verify_grad(use_wake="--wake" in _s.argv) else 1)
    raise SystemExit(0 if verify() else 1)
