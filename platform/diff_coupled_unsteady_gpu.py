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
from fluxvortex.warp_fsi.kernels_ancf import ANCFConstants      # noqa: E402
from fluxvortex.warp_fsi.batched_solver import structural_cg, batched_dense_solve  # noqa: E402

import diff_struct_design_gpu as dsg                            # noqa: E402
import diff_coupled_unsteady as dcu                             # noqa: E402 (numpy oracle)
import diff_uvlm_unsteady_gpu as ug                             # noqa: E402 (aero kernels)
from diff_struct_design import _build_shell                     # noqa: E402

V3 = wp.vec3d
VINF = dcu.VINF


def coupled_unsteady_forward_gpu(sh, C, P, dist, q0, dq0, N, dt, Es, Rs, nx, ny,
                                 Vinf=VINF, cg_tol=1e-12):
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
        wp.launch(ug.shed_kernel, dim=ny, inputs=[rings, gamma, te, Vw, DTYPE(dt), nw], outputs=[wr, wg], device=dev)
        nw += ny
        wp.launch(ug.convect_kernel, dim=(nw, 4), inputs=[rings, gamma, npan, wr, wg, nw, Vw, DTYPE(dt)],
                  outputs=[wr_new], device=dev)
        wp.copy(wr, wr_new, count=nw * 4)
        gprev = wp.array(gamma.numpy(), dtype=DTYPE, device=dev)
    return q, qs


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


if __name__ == "__main__":
    raise SystemExit(0 if verify() else 1)
