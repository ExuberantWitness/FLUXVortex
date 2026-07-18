"""rVPM 3D particle transport for the production lattice (S3, PROJECT_rvpm).

Revives the `part_lev` particle path of _v2_robo with the rVPM upgrades
(research_rvpm_arch.md V4/V6): transport = f=0, g=1/5 (transposed stretching +
sigma evolution) + Williamson LSRK3 + corrected-Pedrizzetti — the S1 math
(src/fluxvortex/warp_vpm_rvpm.py) generalized to advect in the FULL field:
freestream + bound rings + TEV ring wake + mutual particle induction.

Increment note (recorded, revisit at the S3 gate): STRETCHING uses the
particle-field Jacobian only — the ring-induced strain is deferred. The LEV
core's two-zone dynamics are dominated by particle-particle interaction; if the
qualitative gate (inboard-stable / outboard-shedding + tip modulation) fails,
add a ring-Jacobian kernel before touching any constant.

Kernel reuse: particle-particle velocity/Jacobian = validated warp_vpm
(Gaussian-erf, same family as _v2_robo.part_vel); ring induction = _v2_robo
ring_vel_core (WAKE_CORE regularization, byte-identical to the legacy path).
Zero-fit: all constants literature values (S1 ledger)."""
import numpy as np
import warp as wp

import _v2_robo as vr
from fluxvortex.warp_vpm import (velocity_from_particles_gpu,
                                 jacobian_from_particles_gpu)
from fluxvortex.warp_vpm_rvpm import pedrizzetti_corrected, EPS_G2

V3, DTYPE = vr.V3, vr.DTYPE
_LSRK3_A = (0.0, -5.0 / 9.0, -153.0 / 128.0)     # Williamson 1980 (S1 ledger)
_LSRK3_B = (1.0 / 3.0, 15.0 / 16.0, 8.0 / 15.0)


@wp.kernel
def _ring_field_vel_kernel(pts: wp.array(dtype=V3), rings: wp.array(dtype=V3, ndim=2),
                           gamma: wp.array(dtype=DTYPE, ndim=2), npan: int,
                           wr: wp.array(dtype=V3, ndim=2), wg: wp.array(dtype=DTYPE),
                           nw: int, Vinf: V3, Vout: wp.array(dtype=V3)):
    """Freestream + bound-ring + TEV-ring-wake velocity at arbitrary points (the non-particle
    part of the advection field; regularization = WAKE_CORE, as the legacy advect kernel)."""
    k = wp.tid(); P = pts[k]
    dl = DTYPE(vr.WAKE_CORE)
    v = Vinf
    for q in range(npan):
        v = v + gamma[0, q] * vr.ring_vel_core(P, rings[q, 0], rings[q, 1], rings[q, 2],
                                               rings[q, 3], dl)
    for m in range(nw):
        v = v + wg[m] * vr.ring_vel_core(P, wr[m, 0], wr[m, 1], wr[m, 2], wr[m, 3], dl)
    Vout[k] = v


def ring_field_vel(X, rings, gamma, npan, wr, wg, nw, Vinf, dev):
    """numpy (n,3) -> numpy (n,3): ring+freestream field at points X."""
    pts = wp.array(X.astype(np.float64), dtype=V3, device=dev)
    Vout = wp.zeros(len(X), dtype=V3, device=dev)
    wp.launch(_ring_field_vel_kernel, dim=len(X),
              inputs=[pts, rings, gamma, npan, wr, wg, nw, Vinf], outputs=[Vout], device=dev)
    return Vout.numpy().astype(np.float64)


def rvpm_step(X, G, sig, ringvel_fn, dt, f=0.0, g=0.2, relax=0.3):
    """One LSRK3 step of the rVPM transport in the full field.

    X,G,sig: numpy (n,3),(n,3),(n,) particle positions / vortex moments / cores.
    ringvel_fn(X) -> (n,3) freestream+ring-induced velocity at positions X.
    Returns updated (X, G, sig). f=0, g=1/5 conservation-derived (rvpm-02);
    relax = corrected-Pedrizzetti rlxf (0.3/step, particle-field omega)."""
    qX = np.zeros_like(X); qG = np.zeros_like(G); qs = np.zeros_like(sig)
    for a, b in zip(_LSRK3_A, _LSRK3_B):
        u = ringvel_fn(X) + velocity_from_particles_gpu(X, X, G, sig)
        J = jacobian_from_particles_gpu(X, G, X, G, sig)          # particle-part strain only
        S = np.einsum("nji,nj->ni", J, G)                          # transposed: (grad u)^T . Gamma
        g2 = np.einsum("ni,ni->n", G, G) + EPS_G2
        Z = ((f + g) / (1.0 + 3.0 * f)) * np.einsum("ni,ni->n", S, G) / g2
        dG = S - 3.0 * Z[:, None] * G
        ds = -sig * Z
        qX = a * qX + dt * u
        qG = a * qG + dt * dG
        qs = a * qs + dt * ds
        X = X + b * qX
        G = G + b * qG
        sig = np.maximum(sig + b * qs, 1e-9)
    if relax > 0.0 and len(X) > 1:
        G = pedrizzetti_corrected(X, G, sig, rlxf=relax)
    return X, G, sig


if __name__ == "__main__":
    # smoke: (1) no rings -> reduces to S1 transport (leapfrog ring survives);
    #        (2) uniform ring field advection: particles translate rigidly, sigma frozen
    import time
    from fluxvortex.warp_vpm_rvpm import make_ring, step_lsrk3
    wp.init()
    X, G, sig = make_ring(R=1.0, Gamma=1.0, a=0.1, n=100)
    X2, G2, s2 = X.copy(), G.copy(), sig.copy()
    zero = lambda P: np.zeros_like(P)
    t0 = time.time()
    for _ in range(50):
        X, G, sig = rvpm_step(X, G, sig, zero, 0.02)
        X2, G2, s2 = step_lsrk3(X2, G2, s2, 0.02)
    d = np.max(np.abs(X - X2))
    print(f"(1) vs S1 step_lsrk3 after 50 steps: max|dX|={d:.2e} (expect ~0)  "
          f"({time.time()-t0:.0f}s)", flush=True)
    Xc, Gc, sc = make_ring(R=1.0, Gamma=1e-8, a=0.1, n=64)   # near-passive tracers
    uni = lambda P: np.tile(np.array([1.0, 0.0, 0.0]), (len(P), 1))
    x0 = Xc.copy()
    for _ in range(100):
        Xc, Gc, sc = rvpm_step(Xc, Gc, sc, uni, 0.01, relax=0.0)
    adv = Xc - x0
    print(f"(2) uniform field: mean dx={adv[:,0].mean():.4f} (expect 1.000)  "
          f"max|dy,dz|={np.abs(adv[:,1:]).max():.2e}  dsig={np.abs(sc-0.1).max():.2e}", flush=True)
    print("SMOKE DONE", flush=True)
