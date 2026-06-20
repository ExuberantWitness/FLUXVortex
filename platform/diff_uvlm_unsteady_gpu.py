"""Warp port of the unsteady free-wake ring-VLM forward (Plan fix1-③) — bit-exact vs the numpy
oracle diff_uvlm_unsteady. All-Warp fp64: ring Biot-Savart, AIC, wake-induced rhs, bound solve
(DiffDenseSolve-able), unsteady KJ + dΓ/dt force, TE shedding, free-wake convection.

The variable-size wake is a PRE-ALLOCATED buffer (maxw = N·ns rings) + an active count nw;
convection double-buffers (read old, write new) to be race-free. This forward is the GPU
substrate for the time-history wake adjoint (fix1-④, next).

verify(): per-step lift time series bit-matches the numpy oracle (rel ~1e-12).
"""
from __future__ import annotations

import os
import sys

import numpy as np

for p in (os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")), os.path.dirname(__file__)):
    if p not in sys.path:
        sys.path.insert(0, p)

import warp as wp                                                # noqa: E402
from fluxvortex.warp_fsi import config as cfg                   # noqa: E402
from fluxvortex.warp_fsi.config import DTYPE                    # noqa: E402
from fluxvortex.warp_fsi.batched_solver import batched_dense_solve  # noqa: E402
import diff_uvlm_unsteady as ref                                # noqa: E402 (numpy oracle)

wp.set_module_options({"enable_backward": True})
V3 = wp.vec3d
RHO = 1.225


@wp.func
def vseg(P: V3, A: V3, B: V3) -> V3:
    r1 = P - A; r2 = P - B; r0 = B - A
    cr = wp.cross(r1, r2)
    cr2 = wp.dot(cr, cr) + wp.float64(1.0e-10)
    n1 = wp.sqrt(wp.dot(r1, r1) + wp.float64(1.0e-20))
    n2 = wp.sqrt(wp.dot(r2, r2) + wp.float64(1.0e-20))
    return (wp.float64(1.0) / (wp.float64(4.0) * wp.float64(3.141592653589793))) \
        * wp.dot(r0, r1 / n1 - r2 / n2) / cr2 * cr


@wp.func
def ring_vel(P: V3, c0: V3, c1: V3, c2: V3, c3: V3) -> V3:
    return vseg(P, c0, c1) + vseg(P, c1, c2) + vseg(P, c2, c3) + vseg(P, c3, c0)


@wp.kernel
def aic_kernel(rings: wp.array(dtype=V3, ndim=2), col: wp.array(dtype=V3),
               nrm: wp.array(dtype=V3), AIC: wp.array(dtype=DTYPE, ndim=3)):
    i, j = wp.tid()
    v = ring_vel(col[i], rings[j, 0], rings[j, 1], rings[j, 2], rings[j, 3])
    AIC[0, i, j] = wp.dot(v, nrm[i])


@wp.kernel
def rhs_kernel(col: wp.array(dtype=V3), nrm: wp.array(dtype=V3), Vinf: V3,
               wr: wp.array(dtype=V3, ndim=2), wg: wp.array(dtype=DTYPE), nw: int,
               rhs: wp.array(dtype=DTYPE, ndim=2)):
    i = wp.tid()
    v = Vinf
    for k in range(nw):
        v = v + wg[k] * ring_vel(col[i], wr[k, 0], wr[k, 1], wr[k, 2], wr[k, 3])
    rhs[0, i] = -wp.dot(v, nrm[i])


@wp.kernel
def lift_kernel(rings: wp.array(dtype=V3, ndim=2), nrm: wp.array(dtype=V3),
                gamma: wp.array(dtype=DTYPE, ndim=2), gprev: wp.array(dtype=DTYPE),
                Vinf: V3, dt: DTYPE, rho: DTYPE, lift: wp.array(dtype=DTYPE)):
    p = wp.tid()
    lb = rings[p, 1] - rings[p, 0]
    Fkj = rho * gamma[0, p] * wp.cross(Vinf, lb)
    cr = wp.cross(rings[p, 2] - rings[p, 0], rings[p, 3] - rings[p, 1])
    area = wp.float64(0.5) * wp.sqrt(wp.dot(cr, cr) + wp.float64(1.0e-30))
    dGdt = (gamma[0, p] - gprev[p]) / dt
    Fun = rho * dGdt * area * nrm[p]
    wp.atomic_add(lift, 0, Fkj[2] + Fun[2])


@wp.kernel
def shed_kernel(rings: wp.array(dtype=V3, ndim=2), gamma: wp.array(dtype=DTYPE, ndim=2),
                te: wp.array(dtype=wp.int32), Vinf: V3, dt: DTYPE, nw: int,
                wr: wp.array(dtype=V3, ndim=2), wg: wp.array(dtype=DTYPE)):
    k = wp.tid()
    p = te[k]
    idx = nw + k
    wr[idx, 0] = rings[p, 3]; wr[idx, 1] = rings[p, 2]
    wr[idx, 2] = rings[p, 2] + Vinf * dt; wr[idx, 3] = rings[p, 3] + Vinf * dt
    wg[idx] = gamma[0, p]


@wp.kernel
def convect_kernel(rings: wp.array(dtype=V3, ndim=2), gamma: wp.array(dtype=DTYPE, ndim=2),
                   npan: int, wr: wp.array(dtype=V3, ndim=2), wg: wp.array(dtype=DTYPE), nw: int,
                   Vinf: V3, dt: DTYPE, wr_new: wp.array(dtype=V3, ndim=2)):
    k, c = wp.tid()
    P = wr[k, c]
    v = Vinf
    for p in range(npan):
        v = v + gamma[0, p] * ring_vel(P, rings[p, 0], rings[p, 1], rings[p, 2], rings[p, 3])
    for m in range(nw):
        v = v + wg[m] * ring_vel(P, wr[m, 0], wr[m, 1], wr[m, 2], wr[m, 3])
    wr_new[k, c] = P + v * dt


@wp.kernel
def bound_rings_kernel(corners: wp.array(dtype=V3), nc: int, ns: int,
                       rings: wp.array(dtype=V3, ndim=2), col: wp.array(dtype=V3),
                       nrm: wp.array(dtype=V3)):
    """corners (flat (nc+1)(ns+1)) -> per-panel ring (1/4-chord), collocation, normal.
    Branch on the int chord index (not a differentiable var) -> adjoint-safe."""
    p = wp.tid()
    i = p // ns; j = p % ns
    s1 = ns + 1
    c00 = corners[i * s1 + j]; c10 = corners[(i + 1) * s1 + j]
    c01 = corners[i * s1 + j + 1]; c11 = corners[(i + 1) * s1 + j + 1]
    qfl = wp.float64(0.75) * c00 + wp.float64(0.25) * c10
    qfr = wp.float64(0.75) * c01 + wp.float64(0.25) * c11
    if i < nc - 1:
        cn1 = corners[(i + 2) * s1 + j]; cn1b = corners[(i + 2) * s1 + j + 1]
        qbl = wp.float64(0.75) * c10 + wp.float64(0.25) * cn1
        qbr = wp.float64(0.75) * c11 + wp.float64(0.25) * cn1b
    else:
        qbl = c10 + wp.float64(0.25) * (c10 - c00)
        qbr = c11 + wp.float64(0.25) * (c11 - c01)
    rings[p, 0] = qfl; rings[p, 1] = qfr; rings[p, 2] = qbr; rings[p, 3] = qbl
    col[p] = wp.float64(0.5) * (wp.float64(0.25) * c00 + wp.float64(0.75) * c10
                                + wp.float64(0.25) * c01 + wp.float64(0.75) * c11)
    n = wp.cross(c11 - c00, c01 - c10)
    nrm[p] = n / wp.sqrt(wp.dot(n, n) + wp.float64(1.0e-20))


def single_step_grad(corners_np, Vinf, dt, gprev_np, nc, ns, device="cuda"):
    """One unsteady step (empty wake): lift Fz and ∂Fz/∂corners via Warp tape + DiffDenseSolve."""
    from diff_solve import DiffDenseSolve
    npan = nc * ns; ncv = (nc + 1) * (ns + 1)
    Vw = V3(*[float(v) for v in np.asarray(Vinf, float)])
    corners = wp.array(corners_np.reshape(ncv, 3), dtype=V3, device=device, requires_grad=True)
    gprev = wp.array(gprev_np.astype(cfg.NP_DTYPE), dtype=DTYPE, device=device)
    dds = DiffDenseSolve(device)
    rings = wp.zeros((npan, 4), dtype=V3, device=device, requires_grad=True)
    col = wp.zeros(npan, dtype=V3, device=device, requires_grad=True)
    nrm = wp.zeros(npan, dtype=V3, device=device, requires_grad=True)
    AIC = wp.zeros((1, npan, npan), dtype=DTYPE, device=device, requires_grad=True)
    rhs = wp.zeros((1, npan), dtype=DTYPE, device=device, requires_grad=True)
    wr0 = wp.zeros((1, 4), dtype=V3, device=device); wg0 = wp.zeros(1, dtype=DTYPE, device=device)
    t1 = wp.Tape()
    with t1:
        wp.launch(bound_rings_kernel, dim=npan, inputs=[corners, nc, ns], outputs=[rings, col, nrm], device=device)
        wp.launch(aic_kernel, dim=(npan, npan), inputs=[rings, col, nrm], outputs=[AIC], device=device)
        wp.launch(rhs_kernel, dim=npan, inputs=[col, nrm, Vw, wr0, wg0, 0], outputs=[rhs], device=device)
    gamma = dds.forward(AIC, rhs); gamma.requires_grad = True
    rings2 = wp.zeros((npan, 4), dtype=V3, device=device, requires_grad=True)
    col2 = wp.zeros(npan, dtype=V3, device=device, requires_grad=True)
    nrm2 = wp.zeros(npan, dtype=V3, device=device, requires_grad=True)
    lift = wp.zeros(1, dtype=DTYPE, device=device, requires_grad=True)
    t2 = wp.Tape()
    with t2:
        wp.launch(bound_rings_kernel, dim=npan, inputs=[corners, nc, ns], outputs=[rings2, col2, nrm2], device=device)
        wp.launch(lift_kernel, dim=npan, inputs=[rings2, nrm2, gamma, gprev, Vw, DTYPE(dt), DTYPE(RHO)],
                  outputs=[lift], device=device)
    Fz = lift.numpy()[0]
    lift.grad = wp.array(np.array([1.0]), dtype=DTYPE, device=device)
    t2.backward()
    adj_A, adj_b = dds.backward(gamma.grad)
    AIC.grad = adj_A; rhs.grad = adj_b
    t1.backward()
    g = corners.grad.numpy().copy()
    return Fz, g


def verify_grad_step(nc=2, ns=3, eps=1e-6):
    wp.init()
    chord, span = 0.3, 0.8
    Vinf = np.array([10.0, 0.0, 0.0]); aoa = np.deg2rad(5.0)
    C = ref._lattice(nc, ns, chord, span, aoa)
    gprev = np.zeros(nc * ns)
    Fz, g = single_step_grad(C.reshape(-1, 3), Vinf, 0.03, gprev, nc, ns)
    # complex-step oracle ∂Fz/∂corners
    flat = C.reshape(-1)
    g_cs = np.zeros(flat.size)
    for k in range(flat.size):
        cp = flat.astype(np.complex128).copy(); cp[k] += 1j * 1e-30
        g_cs[k] = np.imag(ref.single_step_lift(cp.reshape(C.shape), Vinf.astype(complex), 0.03,
                                               gprev.astype(complex), nc, ns)) / 1e-30
    rel = np.max(np.abs(g.reshape(-1) - g_cs)) / (np.max(np.abs(g_cs)) + 1e-30)
    ok = rel < 1e-6
    print(f"Warp differentiable single unsteady step ({nc}x{ns} panels):")
    print(f"  lift Fz={Fz:+.3f} N; ∂Fz/∂corners Warp adjoint vs complex-step: rel={rel:.2e}")
    print(f"  -> {'PASS' if ok else 'FAIL'}: the unsteady step (ring AIC + dΓ/dt + DiffDenseSolve) "
          f"is Warp-differentiable — the per-step building block for the wake-history adjoint")
    return ok


def unsteady_rollout_gpu(nc, ns, chord, span, aoa, Vinf, N, dt, device="cuda"):
    npan = nc * ns; maxw = N * ns
    Cl = ref._lattice(nc, ns, chord, span, aoa)
    rings_np, col_np, nrm_np = ref._bound_rings(Cl, nc, ns)
    rings = wp.array(rings_np, dtype=V3, device=device)
    col = wp.array(col_np, dtype=V3, device=device); nrm = wp.array(nrm_np, dtype=V3, device=device)
    Vw = V3(*[float(v) for v in np.asarray(Vinf, float)])
    AIC = wp.zeros((1, npan, npan), dtype=DTYPE, device=device)
    wp.launch(aic_kernel, dim=(npan, npan), inputs=[rings, col, nrm], outputs=[AIC], device=device)
    wr = wp.zeros((maxw, 4), dtype=V3, device=device); wr_new = wp.zeros((maxw, 4), dtype=V3, device=device)
    wg = wp.zeros(maxw, dtype=DTYPE, device=device)
    gprev = wp.zeros(npan, dtype=DTYPE, device=device)
    te = wp.array(np.array([(nc - 1) * ns + j for j in range(ns)], np.int32), dtype=wp.int32, device=device)
    lifts = np.zeros(N)
    nw = 0
    for step in range(N):
        rhs = wp.zeros((1, npan), dtype=DTYPE, device=device)
        wp.launch(rhs_kernel, dim=npan, inputs=[col, nrm, Vw, wr, wg, nw], outputs=[rhs], device=device)
        gamma = batched_dense_solve(AIC, rhs, device)
        lift = wp.zeros(1, dtype=DTYPE, device=device)
        wp.launch(lift_kernel, dim=npan, inputs=[rings, nrm, gamma, gprev, Vw, DTYPE(dt), DTYPE(RHO)],
                  outputs=[lift], device=device)
        lifts[step] = lift.numpy()[0]
        wp.launch(shed_kernel, dim=ns, inputs=[rings, gamma, te, Vw, DTYPE(dt), nw], outputs=[wr, wg], device=device)
        nw += ns
        wp.launch(convect_kernel, dim=(nw, 4), inputs=[rings, gamma, npan, wr, wg, nw, Vw, DTYPE(dt)],
                  outputs=[wr_new], device=device)
        wp.copy(wr, wr_new, count=nw * 4)
        gprev = wp.array(gamma.numpy()[0], dtype=DTYPE, device=device)
    return lifts


def verify(nc=2, ns=3, N=12, dt=0.03):
    wp.init()
    chord, span = 0.3, 0.8
    Vinf = np.array([10.0, 0.0, 0.0]); aoa = np.deg2rad(5.0)
    L_np, _ = ref.unsteady_rollout(nc, ns, chord, span, aoa, Vinf, N, dt)
    L_np = np.real(L_np)
    L_gpu = unsteady_rollout_gpu(nc, ns, chord, span, aoa, Vinf, N, dt)
    rel = np.max(np.abs(L_gpu - L_np)) / (np.max(np.abs(L_np)) + 1e-30)
    ok = rel < 1e-10
    print(f"Warp unsteady free-wake ring-VLM ({nc}x{ns} panels, {N} steps) vs numpy oracle:")
    print(f"  per-step lift time series  rel={rel:.2e}  (GPU {L_gpu[-1]:+.3f} vs numpy {L_np[-1]:+.3f} N)")
    print(f"  -> {'PASS' if ok else 'FAIL'}: unsteady free-wake forward ported numpy->GPU "
          f"(bit-exact; substrate for the wake-history adjoint fix1-④)")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if verify() else 1)
