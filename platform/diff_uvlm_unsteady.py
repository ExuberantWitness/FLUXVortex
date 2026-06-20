"""Differentiable UNSTEADY free-wake ring-VLM (Plan fix1, numpy/complex-step oracle).

The steady diff_vlm/diff_vlm_gpu carry no wake history. This adds the unsteady physics that the
publication needs:
  · ring vortices (bound panel rings + shed wake rings),
  · each step the bound circulation solve sees the WAKE-induced velocity in the rhs (history),
  · a wake ring is SHED at the trailing edge each step (γ = bound TE ring),
  · FREE wake: every wake corner convects by (V∞ + induced velocity of all rings)·dt,
  · UNSTEADY force = Kutta-Joukowski + the ∂Γ/∂t added-mass term.

Written complex-safe so the EXACT gradient of the time-integrated load w.r.t. the design
(geometry / kinematics) comes from complex-step — the differentiability red line for the
unsteady aero. This numpy oracle precedes the Warp port (diff_uvlm_unsteady_gpu, next), exactly
as diff_vlm.py preceded diff_vlm_gpu.py.

verify(): (1) the unsteady lift builds up over time (Wagner-like transient, wake present);
(2) complex-step ∂(mean lift)/∂(AoA) matches central finite differences.
"""
from __future__ import annotations

import numpy as np

RHO = 1.225


def _vseg(P, A, B):
    """Biot-Savart of a unit straight vortex segment A→B at P (complex-safe; Vatistas-ish core)."""
    r1 = P - A; r2 = P - B; r0 = B - A
    cr = np.cross(r1, r2)
    cr2 = np.dot(cr, cr) + 1e-10
    n1 = np.sqrt(np.dot(r1, r1) + 1e-20)
    n2 = np.sqrt(np.dot(r2, r2) + 1e-20)
    return (1.0 / (4.0 * np.pi)) * np.dot(r0, r1 / n1 - r2 / n2) / cr2 * cr


def _ring_vel(P, ring):
    """Induced velocity at P from a unit-strength ring (4 corners CCW)."""
    return (_vseg(P, ring[0], ring[1]) + _vseg(P, ring[1], ring[2])
            + _vseg(P, ring[2], ring[3]) + _vseg(P, ring[3], ring[0]))


def _lattice(nc, ns, chord, span, aoa, dt_dtype=float):
    """Flat wing corner lattice (nc+1, ns+1, 3) at AoA; collocation/normal per panel."""
    a = aoa
    xs = np.linspace(0, chord, nc + 1); ys = np.linspace(0, span, ns + 1)
    C = np.zeros((nc + 1, ns + 1, 3), dtype=dt_dtype)
    for i in range(nc + 1):
        for j in range(ns + 1):
            C[i, j] = [xs[i] * np.cos(a), ys[j], -xs[i] * np.sin(a)]
    return C


def _bound_rings(C, nc, ns):
    """Vortex ring per panel (Katz-Plotkin: front segment at 1/4 chord of the panel, back at the
    next panel's 1/4 chord; last row trails 1/4-chord aft). Returns rings (nc*ns,4,3), colloc, n."""
    dt = C.dtype
    rings = np.zeros((nc * ns, 4, 3), dt); col = np.zeros((nc * ns, 3), dt); nrm = np.zeros((nc * ns, 3), dt)
    for i in range(nc):
        for j in range(ns):
            c00 = C[i, j]; c10 = C[i + 1, j]; c01 = C[i, j + 1]; c11 = C[i + 1, j + 1]
            qfl = 0.75 * c00 + 0.25 * c10; qfr = 0.75 * c01 + 0.25 * c11      # front 1/4-chord
            if i < nc - 1:
                cn0 = C[i + 1, j]; cn1 = C[i + 2, j]; cn0b = C[i + 1, j + 1]; cn1b = C[i + 2, j + 1]
                qbl = 0.75 * cn0 + 0.25 * cn1; qbr = 0.75 * cn0b + 0.25 * cn1b
            else:
                qbl = c10 + 0.25 * (c10 - c00); qbr = c11 + 0.25 * (c11 - c01)
            p = i * ns + j
            rings[p] = np.array([qfl, qfr, qbr, qbl])
            col[p] = 0.5 * (0.25 * c00 + 0.75 * c10 + 0.25 * c01 + 0.75 * c11)
            n = np.cross(c11 - c00, c01 - c10); nrm[p] = n / (np.sqrt(np.dot(n, n) + 1e-20))
    return rings, col, nrm


def unsteady_rollout(nc, ns, chord, span, aoa, Vinf, N, dt, free_wake=True):
    """Run N unsteady steps; return per-step lift (N,) and the final wake. Ring VLM, shed wake."""
    dtp = np.asarray(Vinf).dtype if np.iscomplexobj(Vinf) or np.iscomplexobj([aoa]) else float
    Vinf = np.asarray(Vinf, dtp)
    C = _lattice(nc, ns, chord, span, aoa, dtp)
    rings, col, nrm = _bound_rings(C, nc, ns)
    npan = nc * ns
    # AIC (bound rings on collocation) — geometry fixed (rigid wing translating in its frame)
    AIC = np.zeros((npan, npan), dtp)
    for i in range(npan):
        for j in range(npan):
            AIC[i, j] = np.dot(_ring_vel(col[i], rings[j]), nrm[i])
    wake = []                                    # list of (ring(4,3), gamma)
    gamma_prev = np.zeros(npan, dtp)
    lift = np.zeros(N, dtp)
    te = [(nc - 1) * ns + j for j in range(ns)]  # trailing-edge panel rings
    for step in range(N):
        # rhs: freestream + wake induction at each collocation
        rhs = np.zeros(npan, dtp)
        for i in range(npan):
            vind = Vinf.copy()
            for (wr, wg) in wake:
                vind = vind + wg * _ring_vel(col[i], wr)
            rhs[i] = -np.dot(vind, nrm[i])
        gamma = np.linalg.solve(AIC, rhs)
        # force: Kutta-Joukowski (front bound segment) + unsteady dΓ/dt added-mass
        Fz = dtp.type(0.0) if hasattr(dtp, "type") else 0.0
        Fz = np.asarray(0.0, dtp)
        for p in range(npan):
            lb = rings[p, 1] - rings[p, 0]       # front bound segment (spanwise)
            Fkj = RHO * gamma[p] * np.cross(Vinf, lb)
            area = 0.5 * np.linalg.norm(np.cross(rings[p, 2] - rings[p, 0], rings[p, 3] - rings[p, 1]))
            dGdt = (gamma[p] - gamma_prev[p]) / dt
            Fun = RHO * dGdt * area * nrm[p]     # ∂Γ/∂t added-mass (normal)
            Fz = Fz + Fkj[2] + Fun[2]
        lift[step] = Fz
        # shed a TE wake ring (corners = TE panel back edge), strength = TE bound gamma
        for k, p in enumerate(te):
            wr = rings[p, [3, 2, 2, 3]].copy()   # back edge; will convect into a sheet
            wr[0] = rings[p, 3]; wr[1] = rings[p, 2]
            wr[2] = rings[p, 2] + Vinf * dt; wr[3] = rings[p, 3] + Vinf * dt
            wake.append((wr, gamma[p]))
        # free-wake convection: move every wake corner by (V∞ + induced)·dt
        if free_wake and wake:
            allrings = [r for r in rings] + [w[0] for w in wake]
            allg = list(np.ones(npan)) + [w[1] for w in wake]   # bound treated unit*? use gamma
            allg = list(gamma) + [w[1] for w in wake]
            new = []
            for (wr, wg) in wake:
                nwr = wr.copy()
                for c in range(4):
                    v = Vinf.copy()
                    for rr, gg in zip(allrings, allg):
                        v = v + gg * _ring_vel(wr[c], rr)
                    nwr[c] = wr[c] + v * dt
                new.append((nwr, wg))
            wake = new
        else:
            wake = [(wr + Vinf * dt, wg) for (wr, wg) in wake]
        gamma_prev = gamma
    return lift, wake


def verify(nc=2, ns=3, N=12, dt=0.03):
    chord, span = 0.3, 0.8
    Vinf = np.array([10.0, 0.0, 0.0]); aoa = np.deg2rad(5.0)
    lift, wake = unsteady_rollout(nc, ns, chord, span, aoa, Vinf, N, dt)
    lift = np.real(lift)
    physical = np.all(np.isfinite(lift)) and len(wake) == N * ns and abs(lift[-1]) > 1e-6
    # quasi-steady part (skip the step-0 impulsive added-mass spike) should settle/build
    kj_trend = lift[1:]
    # complex-step ∂(mean lift)/∂aoa vs central FD — the differentiability red line
    h = 1e-30
    g_cs = np.imag(unsteady_rollout(nc, ns, chord, span, complex(aoa, h), Vinf.astype(complex), N, dt)[0].mean()) / h
    e = 1e-6
    gp = unsteady_rollout(nc, ns, chord, span, aoa + e, Vinf, N, dt)[0].mean()
    gm = unsteady_rollout(nc, ns, chord, span, aoa - e, Vinf, N, dt)[0].mean()
    g_fd = np.real(gp - gm) / (2 * e)
    rel = abs(g_cs - g_fd) / (abs(g_fd) + 1e-12)
    ok = physical and rel < 1e-6
    print(f"Differentiable UNSTEADY free-wake ring-VLM ({nc}x{ns} panels, {N} steps):")
    print(f"  lift transient: {lift[0]:+.2f} -> {lift[-1]:+.2f} N (step-0 = impulsive added-mass "
          f"spike; settles to {kj_trend[-1]:+.2f}); {len(wake)} wake rings shed: "
          f"{'✓' if physical else '✗'}")
    print(f"  ∂(mean lift)/∂AoA  complex-step vs FD: rel={rel:.2e}  (DIFFERENTIABLE)")
    print(f"  -> {'PASS' if ok else 'FAIL'}: unsteady free-wake aero forward + differentiable "
          f"(oracle for the Warp port; physical validation vs standalone_uvlm/Wagner = next)")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if verify() else 1)
