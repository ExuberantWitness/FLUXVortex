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
WAKE_CORE = 0.05          # regularized vortex-core fraction for free-wake self/near interactions


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


@wp.func
def vseg_core(P: V3, A: V3, B: V3, delta: DTYPE) -> V3:
    """Regularized (finite-core) Biot-Savart segment: the cross-product denominator carries a
    δ²|r0|² core (van Garrel form) so coincident/near corners give a bounded velocity AND a
    bounded adjoint — required for the free-wake self-induction to be differentiable."""
    r1 = P - A; r2 = P - B; r0 = B - A
    cr = wp.cross(r1, r2)
    cr2 = wp.dot(cr, cr) + delta * delta * wp.dot(r0, r0) + wp.float64(1.0e-30)
    n1 = wp.sqrt(wp.dot(r1, r1) + wp.float64(1.0e-20))
    n2 = wp.sqrt(wp.dot(r2, r2) + wp.float64(1.0e-20))
    return (wp.float64(1.0) / (wp.float64(4.0) * wp.float64(3.141592653589793))) \
        * wp.dot(r0, r1 / n1 - r2 / n2) / cr2 * cr


@wp.func
def ring_vel_core(P: V3, c0: V3, c1: V3, c2: V3, c3: V3, delta: DTYPE) -> V3:
    return vseg_core(P, c0, c1, delta) + vseg_core(P, c1, c2, delta) \
        + vseg_core(P, c2, c3, delta) + vseg_core(P, c3, c0, delta)


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
    ci = col[i]; ni = nrm[i]          # cache loop-invariant reads in locals → clean adjoint
    # Accumulate rhs as a SCALAR so nrm(=ni) appears explicitly in every term. A vector
    # `v = Vinf; v += …` loop then `dot(v, ni)` mis-saves v's post-loop value for ni's adjoint
    # in Warp's reverse pass (only the pre-loop Vinf survives) — scalar form sidesteps it.
    s = -wp.dot(Vinf, ni)
    for k in range(nw):
        s = s - wg[k] * wp.dot(ring_vel(ci, wr[k, 0], wr[k, 1], wr[k, 2], wr[k, 3]), ni)
    rhs[0, i] = s


@wp.kernel
def lift_kernel(rings: wp.array(dtype=V3, ndim=2), nrm: wp.array(dtype=V3),
                gamma: wp.array(dtype=DTYPE, ndim=2), gprev: wp.array(dtype=DTYPE, ndim=2),
                Vinf: V3, dt: DTYPE, rho: DTYPE, ns: int, lift: wp.array(dtype=DTYPE)):
    p = wp.tid()
    # KJ on the leading bound segment carries the NET chordwise circulation Γ_p − Γ_upstream
    # (vortex-ring lattice: the shared chordwise segment between panels telescopes); for the
    # leading row (i=0) there is no upstream panel → full Γ_p.
    gnet = gamma[0, p]
    if p // ns > 0:
        gnet = gamma[0, p] - gamma[0, p - ns]
    lb = rings[p, 1] - rings[p, 0]
    Fkj = rho * gnet * wp.cross(Vinf, lb)
    cr = wp.cross(rings[p, 2] - rings[p, 0], rings[p, 3] - rings[p, 1])
    area = wp.float64(0.5) * wp.sqrt(wp.dot(cr, cr) + wp.float64(1.0e-30))
    dGdt = (gamma[0, p] - gprev[0, p]) / dt
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
    dl = DTYPE(WAKE_CORE)
    v = Vinf
    for p in range(npan):
        v = v + gamma[0, p] * ring_vel_core(P, rings[p, 0], rings[p, 1], rings[p, 2], rings[p, 3], dl)
    for m in range(nw):
        v = v + wg[m] * ring_vel_core(P, wr[m, 0], wr[m, 1], wr[m, 2], wr[m, 3], dl)
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
    gprev = wp.array(gprev_np.astype(cfg.NP_DTYPE)[None], dtype=DTYPE, device=device)
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
        wp.launch(lift_kernel, dim=npan, inputs=[rings2, nrm2, gamma, gprev, Vw, DTYPE(dt), DTYPE(RHO), ns],
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


@wp.kernel
def lift_kj_kernel(rings: wp.array(dtype=V3, ndim=2), nrm: wp.array(dtype=V3),
                   gamma: wp.array(dtype=DTYPE, ndim=2), gprev: wp.array(dtype=DTYPE, ndim=2),
                   Vinf: V3, dt: DTYPE, rho: DTYPE, ns: int, lift: wp.array(dtype=DTYPE)):
    """KJ-only (circulatory) lift, no dΓ/dt added mass — net chordwise circulation Γ_p−Γ_upstream."""
    p = wp.tid()
    gnet = gamma[0, p]
    if p // ns > 0:
        gnet = gamma[0, p] - gamma[0, p - ns]
    lb = rings[p, 1] - rings[p, 0]
    Fkj = rho * gnet * wp.cross(Vinf, lb)
    wp.atomic_add(lift, 0, Fkj[2])


@wp.kernel
def convect_free_kernel(wr: wp.array(dtype=V3, ndim=2), nw: int, Vinf: V3, dt: DTYPE,
                        wr_new: wp.array(dtype=V3, ndim=2)):
    """Prescribed wake: convect by freestream only (no induction) — diagnostic / cheap variant."""
    k, c = wp.tid()
    wr_new[k, c] = wr[k, c] + Vinf * dt


@wp.kernel
def wcopy_kernel(src: wp.array(dtype=V3, ndim=2), dst: wp.array(dtype=V3, ndim=2)):
    k, c = wp.tid()
    dst[k, c] = src[k, c]


@wp.kernel
def wgcopy_kernel(src: wp.array(dtype=DTYPE), dst: wp.array(dtype=DTYPE)):
    k = wp.tid()
    dst[k] = src[k]


def unsteady_rollout_grad(corners_np, Vinf, dt, N, nc, ns, seed=None, free_wake=True,
                          added_mass=True, device="cuda"):
    """Differentiable N-step unsteady free-wake rollout (fix1-④): per-step lift + ∂(Σ seed·lift)/
    ∂corners through the WHOLE wake history (recurrent vortex system / BPTT). Structure:
      tape_geo: corners → rings,col,nrm → AIC          (geometry, backproped once at the end)
      per step t:  tape_a: (col,nrm,wake_t) → rhs_t
                   DiffDenseSolve: gamma_t = AIC⁻¹ rhs_t      (manual VJP, breaks the tape)
                   tape_b: (rings,nrm,gamma_t,gprev_t,wake_t) → lift_t, wake_{t+1}
    Backward walks steps in reverse; distinct per-step buffers (no in-place) so the tape is intact,
    gprev_{t+1} IS the gamma_t array so the dΓ/dt coupling chains automatically, AIC's adjoint
    accumulates across all steps' solves, and the wake-state grad chains step→step."""
    from diff_solve import DiffDenseSolve
    npan = nc * ns; ncv = (nc + 1) * (ns + 1)
    Vw = V3(*[float(v) for v in np.asarray(Vinf, float)])
    dtt = DTYPE(dt); rhod = DTYPE(RHO)
    te = wp.array(np.array([(nc - 1) * ns + j for j in range(ns)], np.int32), dtype=wp.int32, device=device)
    corners = wp.array(corners_np.reshape(ncv, 3), dtype=V3, device=device, requires_grad=True)
    rings = wp.zeros((npan, 4), dtype=V3, device=device, requires_grad=True)
    col = wp.zeros(npan, dtype=V3, device=device, requires_grad=True)
    nrm = wp.zeros(npan, dtype=V3, device=device, requires_grad=True)
    AIC = wp.zeros((1, npan, npan), dtype=DTYPE, device=device, requires_grad=True)
    tape_geo = wp.Tape()
    with tape_geo:
        wp.launch(bound_rings_kernel, dim=npan, inputs=[corners, nc, ns], outputs=[rings, col, nrm], device=device)
        wp.launch(aic_kernel, dim=(npan, npan), inputs=[rings, col, nrm], outputs=[AIC], device=device)

    steps = []
    gprev = wp.zeros((1, npan), dtype=DTYPE, device=device, requires_grad=True)   # gprev_0 = 0
    wr_t = None; wg_t = None
    lifts = np.zeros(N)
    for t in range(N):
        nw = t * ns; nwn = nw + ns
        rhs = wp.zeros((1, npan), dtype=DTYPE, device=device, requires_grad=True)
        if nw == 0:
            wr_in = wp.zeros((1, 4), dtype=V3, device=device, requires_grad=True)
            wg_in = wp.zeros(1, dtype=DTYPE, device=device, requires_grad=True)
        else:
            wr_in = wr_t; wg_in = wg_t
        tape_a = wp.Tape()
        with tape_a:
            wp.launch(rhs_kernel, dim=npan, inputs=[col, nrm, Vw, wr_in, wg_in, nw], outputs=[rhs], device=device)
        dds = DiffDenseSolve(device)
        gamma = dds.forward(AIC, rhs); gamma.requires_grad = True
        lift = wp.zeros(1, dtype=DTYPE, device=device, requires_grad=True)
        wcat = wp.zeros((nwn, 4), dtype=V3, device=device, requires_grad=True)
        wgcat = wp.zeros(nwn, dtype=DTYPE, device=device, requires_grad=True)
        wr_next = wp.zeros((nwn, 4), dtype=V3, device=device, requires_grad=True)
        lk = lift_kernel if added_mass else lift_kj_kernel
        tape_b = wp.Tape()
        with tape_b:
            wp.launch(lk, dim=npan, inputs=[rings, nrm, gamma, gprev, Vw, dtt, rhod, ns],
                      outputs=[lift], device=device)
            if nw > 0:
                wp.launch(wcopy_kernel, dim=(nw, 4), inputs=[wr_in], outputs=[wcat], device=device)
                wp.launch(wgcopy_kernel, dim=nw, inputs=[wg_in], outputs=[wgcat], device=device)
            wp.launch(shed_kernel, dim=ns, inputs=[rings, gamma, te, Vw, dtt, nw], outputs=[wcat, wgcat], device=device)
            if free_wake:
                wp.launch(convect_kernel, dim=(nwn, 4), inputs=[rings, gamma, npan, wcat, wgcat, nwn, Vw, dtt],
                          outputs=[wr_next], device=device)
            else:
                wp.launch(convect_free_kernel, dim=(nwn, 4), inputs=[wcat, nwn, Vw, dtt], outputs=[wr_next], device=device)
        lifts[t] = lift.numpy()[0]
        steps.append(dict(tape_a=tape_a, tape_b=tape_b, rhs=rhs, gamma=gamma, lift=lift, dds=dds))
        gprev = gamma; wr_t = wr_next; wg_t = wgcat   # gprev_{t+1} IS gamma_t (shares grad → dΓ/dt chains)

    sd = np.ones(N) if seed is None else np.asarray(seed, float)
    total = float(sd @ lifts)
    adj_AIC = wp.zeros((1, npan, npan), dtype=DTYPE, device=device)
    for t in reversed(range(N)):
        s = steps[t]
        s["lift"].grad = wp.array(np.array([sd[t]]), dtype=DTYPE, device=device)
        s["tape_b"].backward()                              # → gamma.grad, rings/nrm.grad, wake_t.grad
        adj_A, adj_b = s["dds"].backward(s["gamma"].grad)   # gamma.grad already holds t & t+1(gprev) parts
        wp.launch(_acc3, dim=(1, npan, npan), inputs=[adj_A], outputs=[adj_AIC], device=device)
        s["rhs"].grad = adj_b
        s["tape_a"].backward()                              # → col/nrm.grad, wake_t.grad (rhs part)
    AIC.grad = adj_AIC
    tape_geo.backward()                                     # rings/col/nrm + AIC adjoints → corners
    return lifts, total, corners.grad.numpy().copy()


@wp.kernel
def _acc3(src: wp.array(dtype=DTYPE, ndim=3), dst: wp.array(dtype=DTYPE, ndim=3)):
    e, i, j = wp.tid()
    dst[e, i, j] = dst[e, i, j] + src[e, i, j]


def _step_fwd(geo, wr_in, wg_in, gprev, nw, free_wake, added_mass):
    """One unsteady step with tapes: returns the step's tapes/arrays + (wr_next, wgcat). The
    differentiable building block shared by the all-store and gradient-checkpointed drivers."""
    from diff_solve import DiffDenseSolve
    npan, ns, dev = geo["npan"], geo["ns"], geo["device"]
    rings, col, nrm, AIC = geo["rings"], geo["col"], geo["nrm"], geo["AIC"]
    Vw, dtt, rhod, te = geo["Vw"], geo["dtt"], geo["rhod"], geo["te"]
    nwn = nw + ns
    rhs = wp.zeros((1, npan), dtype=DTYPE, device=dev, requires_grad=True)
    tape_a = wp.Tape()
    with tape_a:
        wp.launch(rhs_kernel, dim=npan, inputs=[col, nrm, Vw, wr_in, wg_in, nw], outputs=[rhs], device=dev)
    dds = DiffDenseSolve(dev)
    gamma = dds.forward(AIC, rhs); gamma.requires_grad = True
    lift = wp.zeros(1, dtype=DTYPE, device=dev, requires_grad=True)
    wcat = wp.zeros((nwn, 4), dtype=V3, device=dev, requires_grad=True)
    wgcat = wp.zeros(nwn, dtype=DTYPE, device=dev, requires_grad=True)
    wr_next = wp.zeros((nwn, 4), dtype=V3, device=dev, requires_grad=True)
    lk = lift_kernel if added_mass else lift_kj_kernel
    tape_b = wp.Tape()
    with tape_b:
        wp.launch(lk, dim=npan, inputs=[rings, nrm, gamma, gprev, Vw, dtt, rhod, ns], outputs=[lift], device=dev)
        if nw > 0:
            wp.launch(wcopy_kernel, dim=(nw, 4), inputs=[wr_in], outputs=[wcat], device=dev)
            wp.launch(wgcopy_kernel, dim=nw, inputs=[wg_in], outputs=[wgcat], device=dev)
        wp.launch(shed_kernel, dim=ns, inputs=[rings, gamma, te, Vw, dtt, nw], outputs=[wcat, wgcat], device=dev)
        if free_wake:
            wp.launch(convect_kernel, dim=(nwn, 4), inputs=[rings, gamma, npan, wcat, wgcat, nwn, Vw, dtt],
                      outputs=[wr_next], device=dev)
        else:
            wp.launch(convect_free_kernel, dim=(nwn, 4), inputs=[wcat, nwn, Vw, dtt], outputs=[wr_next], device=dev)
    return dict(tape_a=tape_a, tape_b=tape_b, rhs=rhs, gamma=gamma, lift=lift, dds=dds,
                wr_in=wr_in, wg_in=wg_in, gprev=gprev, wr_next=wr_next, wgcat=wgcat)


def unsteady_rollout_grad_ckpt(corners_np, Vinf, dt, N, nc, ns, seed=None, ckpt=None,
                               free_wake=True, added_mass=True, device="cuda"):
    """Gradient-checkpointed wake-history adjoint: identical math to unsteady_rollout_grad but
    stores per-step tapes for only ONE segment at a time (length ~√N). The forward pass keeps just
    the wake STATE (wr,wg,gprev) at √N checkpoints; backward recomputes each segment's tapes from
    its checkpoint, walks it in reverse, and carries (adj_wake, adj_gprev) across segment boundaries.
    Memory O(√N) tapes instead of O(N) → makes the long real-FSI rollout differentiable on GPU."""
    npan = nc * ns; ncv = (nc + 1) * (ns + 1)
    Vw = V3(*[float(v) for v in np.asarray(Vinf, float)])
    dtt = DTYPE(dt); rhod = DTYPE(RHO)
    te = wp.array(np.array([(nc - 1) * ns + j for j in range(ns)], np.int32), dtype=wp.int32, device=device)
    corners = wp.array(corners_np.reshape(ncv, 3), dtype=V3, device=device, requires_grad=True)
    rings = wp.zeros((npan, 4), dtype=V3, device=device, requires_grad=True)
    col = wp.zeros(npan, dtype=V3, device=device, requires_grad=True)
    nrm = wp.zeros(npan, dtype=V3, device=device, requires_grad=True)
    AIC = wp.zeros((1, npan, npan), dtype=DTYPE, device=device, requires_grad=True)
    tape_geo = wp.Tape()
    with tape_geo:
        wp.launch(bound_rings_kernel, dim=npan, inputs=[corners, nc, ns], outputs=[rings, col, nrm], device=device)
        wp.launch(aic_kernel, dim=(npan, npan), inputs=[rings, col, nrm], outputs=[AIC], device=device)
    geo = dict(npan=npan, ns=ns, device=device, rings=rings, col=col, nrm=nrm, AIC=AIC,
               Vw=Vw, dtt=dtt, rhod=rhod, te=te)
    K = ckpt or max(1, int(round(N ** 0.5)))
    bounds = list(range(0, N, K)) + [N]                     # segment starts + final
    mkleaf = lambda a: wp.array(a, dtype=V3, device=device, requires_grad=True)
    mkleafg = lambda a: wp.array(a, dtype=DTYPE, device=device, requires_grad=True)

    # ---- forward: keep only wake STATE at each segment start (numpy) + lifts ----
    ckpts = {}
    wr_np = np.zeros((1, 4, 3)); wg_np = np.zeros(1); gprev_np = np.zeros((1, npan))
    lifts = np.zeros(N)
    for t in range(N):
        if t in bounds:
            ckpts[t] = (wr_np.copy(), wg_np.copy(), gprev_np.copy())
        nw = t * ns
        wr_in = mkleaf(wr_np if nw > 0 else np.zeros((1, 4, 3)))
        wg_in = mkleafg(wg_np if nw > 0 else np.zeros(1))
        gprev = mkleafg(gprev_np)
        st = _step_fwd(geo, wr_in, wg_in, gprev, nw, free_wake, added_mass)
        lifts[t] = st["lift"].numpy()[0]
        wr_np = st["wr_next"].numpy(); wg_np = st["wgcat"].numpy(); gprev_np = st["gamma"].numpy()

    sd = np.ones(N) if seed is None else np.asarray(seed, float)
    total = float(sd @ lifts)

    # ---- backward by segments, in reverse; carry (adj_wake, adj_gprev) across boundaries ----
    adj_AIC = wp.zeros((1, npan, npan), dtype=DTYPE, device=device)
    seg_starts = list(range(0, N, K))
    adj_wr_e = None; adj_wg_e = None                        # adj on wake state at segment end
    adj_gprev_e = np.zeros((1, npan))                       # adj on gamma_{e-1} (gprev of next seg)
    for s in reversed(seg_starts):
        e = min(s + K, N)
        wr0, wg0, gp0 = ckpts[s]
        wr_in = mkleaf(wr0 if s > 0 else np.zeros((1, 4, 3)))
        wg_in = mkleafg(wg0 if s > 0 else np.zeros(1))
        gprev0 = mkleafg(gp0)
        # recompute the segment's tapes
        seg = []
        wr_c, wg_c, gp_c = wr_in, wg_in, gprev0
        for t in range(s, e):
            st = _step_fwd(geo, wr_c, wg_c, gp_c, t * ns, free_wake, added_mass)
            seg.append(st)
            wr_c, wg_c, gp_c = st["wr_next"], st["wgcat"], st["gamma"]
        # backward over the segment
        for idx in reversed(range(len(seg))):
            t = s + idx; st = seg[idx]
            st["lift"].grad = wp.array(np.array([sd[t]]), dtype=DTYPE, device=device)
            if idx == len(seg) - 1 and adj_wr_e is not None:        # carry wake adj from next seg
                st["wr_next"].grad = adj_wr_e; st["wgcat"].grad = adj_wg_e
            st["tape_b"].backward()
            if idx == len(seg) - 1:                                  # carry gprev adj (→ gamma_{e-1})
                wp.launch(_acc2, dim=(1, npan), inputs=[wp.array(adj_gprev_e, dtype=DTYPE, device=device)],
                          outputs=[st["gamma"].grad], device=device)
            adj_A, adj_b = st["dds"].backward(st["gamma"].grad)
            wp.launch(_acc3, dim=(1, npan, npan), inputs=[adj_A], outputs=[adj_AIC], device=device)
            st["rhs"].grad = adj_b
            st["tape_a"].backward()
        # carry the segment-start adjoints to the previous segment
        adj_wr_e = wp.clone(seg[0]["wr_in"].grad); adj_wg_e = wp.clone(seg[0]["wg_in"].grad)
        adj_gprev_e = seg[0]["gprev"].grad.numpy().copy()
    AIC.grad = adj_AIC
    tape_geo.backward()
    return lifts, total, corners.grad.numpy().copy()


@wp.kernel
def _acc2(src: wp.array(dtype=DTYPE, ndim=2), dst: wp.array(dtype=DTYPE, ndim=2)):
    e, i = wp.tid()
    dst[e, i] = dst[e, i] + src[e, i]


def verify_grad_rollout(nc=2, ns=3, N=12, dt=0.03):
    wp.init()
    chord, span = 0.3, 0.8
    Vinf = np.array([10.0, 0.0, 0.0]); aoa = np.deg2rad(5.0)
    C = ref._lattice(nc, ns, chord, span, aoa)
    lifts, total, g = unsteady_rollout_grad(C.reshape(-1, 3), Vinf, dt, N, nc, ns)
    # complex-step oracle ∂(Σ lift)/∂corners through the full wake history
    flat = C.reshape(-1)
    g_cs = np.zeros(flat.size)
    for k in range(flat.size):
        cp = flat.astype(np.complex128).copy(); cp[k] += 1j * 1e-30
        Lc = ref.unsteady_rollout_corners(cp.reshape(C.shape), Vinf.astype(complex), dt, N, nc, ns)
        g_cs[k] = np.imag(Lc.sum()) / 1e-30
    rel = np.max(np.abs(g.reshape(-1) - g_cs)) / (np.max(np.abs(g_cs)) + 1e-30)
    ok = rel < 1e-6
    print(f"Warp wake-history adjoint — differentiable unsteady free-wake ROLLOUT "
          f"({nc}x{ns} panels, {N} steps, {N * ns} wake rings):")
    print(f"  Σlift={total:+.3f} N; ∂(Σlift)/∂corners through the WHOLE wake recurrence (BPTT)")
    print(f"  Warp adjoint vs complex-step oracle: rel={rel:.2e}  (max|grad|={np.max(np.abs(g)):.1f})")
    print(f"  -> {'PASS' if ok else 'FAIL'}: the free-wake vortex history (shed + free convection + "
          f"dΓ/dt) is end-to-end differentiable on GPU — fix1-④ CORE\n"
          f"     (regularized vortex core δ={WAKE_CORE} desingularizes the self-induction adjoint; "
          f"bound AIC stays sharp)")
    return ok


def verify_grad_ckpt(nc=2, ns=3, N=16, dt=0.03):
    wp.init()
    chord, span = 0.3, 0.8
    Vinf = np.array([10.0, 0.0, 0.0]); aoa = np.deg2rad(5.0)
    C = ref._lattice(nc, ns, chord, span, aoa)
    _, tot0, g0 = unsteady_rollout_grad(C.reshape(-1, 3), Vinf, dt, N, nc, ns)       # all-store ref
    K = max(1, int(round(N ** 0.5)))
    _, tot1, g1 = unsteady_rollout_grad_ckpt(C.reshape(-1, 3), Vinf, dt, N, nc, ns, ckpt=K)
    rel = np.max(np.abs(g1 - g0)) / (np.max(np.abs(g0)) + 1e-30)
    nseg = (N + K - 1) // K
    ok = rel < 1e-12 and abs(tot1 - tot0) < 1e-9
    print(f"Gradient checkpointing for the wake-history adjoint ({nc}x{ns}, {N} steps):")
    print(f"  segment length K={K} → {nseg} segments; O(√N) tapes resident vs O(N) all-store")
    print(f"  ∂(Σlift)/∂corners  checkpointed vs all-store: rel={rel:.2e}")
    print(f"  -> {'PASS' if ok else 'FAIL'}: identical gradient at √N memory — the long real-FSI "
          f"rollout is differentiable on GPU within bounded memory")
    return ok


def unsteady_rollout_gpu(nc, ns, chord, span, aoa, Vinf, N, dt, added_mass=True, device="cuda"):
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
    gprev = wp.zeros((1, npan), dtype=DTYPE, device=device)
    te = wp.array(np.array([(nc - 1) * ns + j for j in range(ns)], np.int32), dtype=wp.int32, device=device)
    lifts = np.zeros(N)
    nw = 0
    for step in range(N):
        rhs = wp.zeros((1, npan), dtype=DTYPE, device=device)
        wp.launch(rhs_kernel, dim=npan, inputs=[col, nrm, Vw, wr, wg, nw], outputs=[rhs], device=device)
        gamma = batched_dense_solve(AIC, rhs, device)
        lift = wp.zeros(1, dtype=DTYPE, device=device)
        wp.launch(lift_kernel if added_mass else lift_kj_kernel, dim=npan,
                  inputs=[rings, nrm, gamma, gprev, Vw, DTYPE(dt), DTYPE(RHO), ns], outputs=[lift], device=device)
        lifts[step] = lift.numpy()[0]
        wp.launch(shed_kernel, dim=ns, inputs=[rings, gamma, te, Vw, DTYPE(dt), nw], outputs=[wr, wg], device=device)
        nw += ns
        wp.launch(convect_kernel, dim=(nw, 4), inputs=[rings, gamma, npan, wr, wg, nw, Vw, DTYPE(dt)],
                  outputs=[wr_new], device=device)
        wp.copy(wr, wr_new, count=nw * 4)
        gprev = wp.array(gamma.numpy(), dtype=DTYPE, device=device)
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
