"""Differentiable VLM aero (geometry → AIC → γ → Kutta-Joukowski force) — the missing aero
kernel for the differentiable COUPLED FSI (S3). The aero load on the wing depends on the
DEFORMED geometry (panel corners = reference + structural displacement), so the design
gradient must flow ∂Force/∂corners; this module provides that Jacobian.

A minimal horseshoe-vortex VLM, written complex-safe so its EXACT Jacobian ∂F/∂(corners)
comes from COMPLEX-STEP differentiation (machine precision, not finite-difference noise) —
the VJP `adj_corners = Jᵀ adj_F` then composes with the validated structural K_t adjoint
(diff_struct_design) and the AIC-solve VJP (DiffDenseSolve) into the full differentiable
coupled FSI. (Unsteady wake is a later refinement; this is the quasi-steady aero load.)

verify(): complex-step Jacobian vs central finite differences of the same VLM forward.
"""
from __future__ import annotations

import numpy as np

RHO = 1.225
_LEG = 50.0           # trailing-leg length (semi-infinite horseshoe approx, chords)


def _vseg(P, A, B):
    """Biot-Savart induced velocity at P from a unit-strength straight vortex segment A→B.
    Complex-safe (norms via sqrt(x·x), no abs) so complex-step differentiation is exact."""
    r1 = P - A; r2 = P - B; r0 = B - A
    cr = np.cross(r1, r2)
    cr2 = np.dot(cr, cr) + 1e-12
    n1 = np.sqrt(np.dot(r1, r1) + 1e-24)
    n2 = np.sqrt(np.dot(r2, r2) + 1e-24)
    k = (1.0 / (4.0 * np.pi)) * np.dot(r0, r1 / n1 - r2 / n2) / cr2
    return k * cr


def _horseshoe(P, A, B, edir):
    """Induced velocity at P from a unit horseshoe: bound A→B + trailing legs to +edir·_LEG."""
    Aw = A + _LEG * edir; Bw = B + _LEG * edir
    return _vseg(P, Bw, B) + _vseg(P, B, A) + _vseg(P, A, Aw)


def vlm_forces(corners, nx, ny, Vinf):
    """corners: (nx+1, ny+1, 3) lattice. Returns per-panel KJ force (nx, ny, 3) and total.
    Horseshoe at the panel quarter-chord; collocation at 3/4-chord midspan."""
    dt = corners.dtype
    edir = np.asarray(Vinf, dt) / (np.sqrt(np.dot(Vinf, Vinf)) + 1e-24)
    npan = nx * ny
    qa = np.zeros((nx, ny, 3), dt); qb = np.zeros((nx, ny, 3), dt)
    col = np.zeros((nx, ny, 3), dt); nrm = np.zeros((nx, ny, 3), dt)
    for i in range(nx):
        for j in range(ny):
            c00 = corners[i, j]; c10 = corners[i + 1, j]
            c01 = corners[i, j + 1]; c11 = corners[i + 1, j + 1]
            # quarter-chord bound vortex (chord = i-direction), spanwise endpoints
            qa[i, j] = 0.75 * c00 + 0.25 * c10
            qb[i, j] = 0.75 * c01 + 0.25 * c11
            col[i, j] = 0.5 * (0.25 * c00 + 0.75 * c10 + 0.25 * c01 + 0.75 * c11)
            d1 = c11 - c00; d2 = c01 - c10
            n = np.cross(d1, d2); nrm[i, j] = n / (np.sqrt(np.dot(n, n) + 1e-24))
    qa = qa.reshape(npan, 3); qb = qb.reshape(npan, 3)
    col = col.reshape(npan, 3); nrm = nrm.reshape(npan, 3)
    AIC = np.zeros((npan, npan), dt); rhs = np.zeros(npan, dt)
    for i in range(npan):
        for j in range(npan):
            v = _horseshoe(col[i], qa[j], qb[j], edir)
            AIC[i, j] = np.dot(v, nrm[i])
        rhs[i] = -np.dot(np.asarray(Vinf, dt), nrm[i])
    gamma = np.linalg.solve(AIC, rhs)
    F = np.zeros((npan, 3), dt)
    for j in range(npan):
        lb = qb[j] - qa[j]                         # bound vortex segment
        F[j] = RHO * gamma[j] * np.cross(np.asarray(Vinf, dt), lb)   # Kutta-Joukowski
    return F.reshape(nx, ny, 3), F.sum(0)


def _flat_wing(nx, ny, chord=0.3, span=0.8, aoa_deg=5.0):
    a = np.deg2rad(aoa_deg)
    xs = np.linspace(0, chord, nx + 1); ys = np.linspace(0, span, ny + 1)
    corners = np.zeros((nx + 1, ny + 1, 3))
    for i in range(nx + 1):
        for j in range(ny + 1):
            corners[i, j] = [xs[i] * np.cos(a), ys[j], -xs[i] * np.sin(a)]
    return corners


def total_force(corners, nx, ny, Vinf):
    _, Ftot = vlm_forces(corners, nx, ny, Vinf)
    return Ftot


def panel_forces_flat(corners, nx, ny, Vinf):
    Fp, _ = vlm_forces(corners, nx, ny, Vinf)
    return Fp.reshape(-1)                                   # (nx*ny*3,)


def panel_jacobian(corners, nx, ny, Vinf, h=1e-30):
    """EXACT ∂(all panel forces)/∂(all corners) via complex-step. Shape (nx·ny·3, ncorner·3).
    This is the VLM block for the coupled-FSI adjoint: adj_corners = Jᵀ·adj_panelForces."""
    shape = corners.shape; flat = corners.reshape(-1)
    m = nx * ny * 3
    J = np.zeros((m, flat.size))
    Vc = np.asarray(Vinf, np.complex128)
    for k in range(flat.size):
        cp = flat.astype(np.complex128).copy(); cp[k] += 1j * h
        J[:, k] = np.imag(panel_forces_flat(cp.reshape(shape), nx, ny, Vc)) / h
    return J


def jac_complex_step(corners, nx, ny, Vinf, h=1e-30):
    """EXACT ∂(total force)/∂(corners) via complex-step (machine precision)."""
    shape = corners.shape
    flat = corners.reshape(-1)
    J = np.zeros((3, flat.size))
    for k in range(flat.size):
        cp = flat.astype(np.complex128).copy(); cp[k] += 1j * h
        Ft = total_force(cp.reshape(shape), nx, ny, np.asarray(Vinf, np.complex128))
        J[:, k] = np.imag(Ft) / h
    return J


def verify(nx=3, ny=4, eps=1e-6):
    Vinf = np.array([10.0, 0.0, 0.0])
    corners = _flat_wing(nx, ny)
    F0, Ftot = vlm_forces(corners, nx, ny, Vinf)
    print(f"Differentiable VLM ({nx}x{ny} panels): total force = "
          f"[{Ftot[0]:+.3f}, {Ftot[1]:+.3f}, {Ftot[2]:+.3f}] N (lift=Fz)")
    Jcs = jac_complex_step(corners, nx, ny, Vinf)          # exact (complex-step)
    flat = corners.reshape(-1)
    Jfd = np.zeros_like(Jcs)
    for k in range(flat.size):
        cp = flat.copy(); cp[k] += eps; cm = flat.copy(); cm[k] -= eps
        Jfd[:, k] = (total_force(cp.reshape(corners.shape), nx, ny, Vinf)
                     - total_force(cm.reshape(corners.shape), nx, ny, Vinf)) / (2 * eps)
    rel = np.max(np.abs(Jcs - Jfd)) / (np.max(np.abs(Jfd)) + 1e-30)
    ok = rel < 1e-6
    print(f"  ∂(total force)/∂(corners): complex-step vs FD  rel={rel:.2e}  "
          f"-> {'PASS' if ok else 'FAIL'}")
    print(f"  -> differentiable VLM aero load ready to couple into the FSI design gradient "
          f"(Jᵀ·adj_F composes with the structural K_t adjoint)")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if verify() else 1)
