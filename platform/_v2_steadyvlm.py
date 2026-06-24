"""Localize the UVLM lift bug with a clean STEADY, NO-WAKE bound VLM. Compare CL at 5deg AoA to the
analytic finite-wing slope, for (a) half-wing x2 (as flap_flight does) vs (b) a full mirrored wing.
If full-wing CL ~ analytic but half-wing x2 is ~0.4x, the bug is the MISSING ROOT SYMMETRY plane.
Also sweep mesh (steady, no wake) to see if the bound VLM itself converges."""
import numpy as np, sys
sys.path.insert(0, '.'); sys.path.insert(0, '..')
import warp as wp
from fluxvortex.warp_fsi import config as cfg
from fluxvortex.warp_fsi.config import DTYPE
from fluxvortex.warp_fsi.batched_solver import batched_dense_solve
import diff_uvlm_unsteady_gpu as ug

V3 = wp.vec3d
wp.init(); dev = cfg.DEVICE; NP = cfg.NP_DTYPE
U = 8.0; RHO = 1.225


def full_wing(nc, ns, chord, half_span):
    """Full wing: y from -half_span..+half_span, ns panels over the FULL span."""
    xs = np.linspace(0.0, chord, nc + 1); ys = np.linspace(-half_span, half_span, ns + 1)
    C = np.zeros((nc + 1, ns + 1, 3))
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            C[i, j] = [x, y, 0.0]
    return C


def steady_CL(C0, nc, ns, chord, span_area, aoa_deg, double=False):
    npan = nc * ns; ncv = (nc + 1) * (ns + 1)
    cw = wp.array(C0.reshape(ncv, 3).astype(NP), dtype=V3, device=dev)
    rings = wp.zeros((npan, 4), dtype=V3, device=dev); col = wp.zeros(npan, dtype=V3, device=dev)
    nrm = wp.zeros(npan, dtype=V3, device=dev)
    wp.launch(ug.bound_rings_kernel, dim=npan, inputs=[cw, nc, ns], outputs=[rings, col, nrm], device=dev)
    AIC = wp.zeros((1, npan, npan), dtype=DTYPE, device=dev)
    wp.launch(ug.aic_kernel, dim=(npan, npan), inputs=[rings, col, nrm], outputs=[AIC], device=dev)
    Vinf = np.array([U, 0.0, U * np.tan(np.radians(aoa_deg))]); Vw = V3(*[float(v) for v in Vinf])
    wr = wp.zeros((1, 4), dtype=V3, device=dev); wg = wp.zeros(1, dtype=DTYPE, device=dev)
    rhs = wp.zeros((1, npan), dtype=DTYPE, device=dev)
    wp.launch(ug.rhs_kernel, dim=npan, inputs=[col, nrm, Vw, wr, wg, 0], outputs=[rhs], device=dev)
    gamma = batched_dense_solve(AIC, rhs, dev)
    vcol = wp.zeros(npan, dtype=V3, device=dev)            # steady: no body motion
    Fp = wp.zeros(npan, dtype=V3, device=dev)
    wp.launch(ug.panel_force_kernel, dim=npan, inputs=[rings, nrm, gamma, gamma, vcol, Vw,
              DTYPE(1.0), DTYPE(RHO), ns], outputs=[Fp], device=dev)   # gprev=gamma -> no dG/dt
    L = float(np.sum(Fp.numpy()[:, 2])) * (2.0 if double else 1.0)
    q = 0.5 * RHO * U ** 2
    return L, L / (q * span_area)


chord = 0.287; hs = 0.80; aoa = 5.0
S_full = 2.0 * chord * hs                                  # both-wing area
AR = (2 * hs) ** 2 / S_full
CLa = 2 * np.pi * AR / (AR + 2)
print(f"full wing: span {2*hs}m chord {chord}m area {S_full:.3f} AR {AR:.2f}", flush=True)
print(f"analytic finite-wing: CLa={CLa:.2f}/rad -> CL@{aoa}deg = {CLa*np.radians(aoa):.3f}  "
      f"-> L = {0.5*RHO*U**2*S_full*CLa*np.radians(aoa):.2f}N", flush=True)
print("\nSTEADY no-wake VLM (lift should = analytic if correct):", flush=True)
for nc, ns in ((4, 10), (8, 20), (12, 30)):
    Lh, CLh = steady_CL(__import__('flap_flight_validate').flat_wing(nc, ns, chord, hs), nc, ns, chord, S_full, aoa, double=True)
    Lf, CLf = steady_CL(full_wing(nc, 2 * ns, chord, hs), nc, 2 * ns, chord, S_full, aoa, double=False)
    print(f"  nc={nc:2d} ns={ns:2d}:  half-wing x2  CL={CLh:.3f} (L={Lh:.2f}N)   |   "
          f"FULL wing  CL={CLf:.3f} (L={Lf:.2f}N)   [analytic CL={CLa*np.radians(aoa):.3f}]", flush=True)
print("DONE", flush=True)
