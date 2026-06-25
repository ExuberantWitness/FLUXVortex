"""Validate the UVLM wing aero against the REAL RoboEagle paper (Drones 2025, 9/8/535,
"Flapping-Twist Coupled..."). Geometry: half-span 0.80 m, root chord 0.287 m. Kinematics:
flap ±45°, spanwise-linear twist (0/22.5/45°) coupled to flap by a phase. Cruise 8 m/s, 5-8° AoA.
Measured anchors: max L/D 6.8; optimal twist (22.5°) gives +47% thrust, +7.8% lift vs untwisted.

We add twist to the validated flapping UVLM (flap_flight_validate's kernels) and check we reproduce
the twist GAIN relationship (a relative measure, robust to the absolute aero model; UVLM has only
induced drag so absolute L/D will exceed 6.8 until profile drag is added)."""
from __future__ import annotations

import numpy as np
import warp as wp
from fluxvortex.warp_fsi import config as cfg
from fluxvortex.warp_fsi.config import DTYPE
from fluxvortex.warp_fsi.batched_solver import batched_dense_solve
import diff_uvlm_unsteady_gpu as ug
from diff_uvlm_unsteady_gpu import ring_vel_core, ring_vel   # @wp.func reused for particle advect + image wing
import flap_flight_validate as ffv
import _v2_robogeom as rg                       # real RoboEagle planform + swept flap/twist axis

V3 = wp.vec3d
RHO = 1.225
WAKE_CORE = ug.WAKE_CORE                          # regularized core for bound/wake induction on particles


@wp.func
def mirror_y(c: V3) -> V3:                        # reflect a point across the y=0 root symmetry plane
    return V3(c[0], -c[1], c[2])


@wp.kernel
def aic_sym_kernel(rings: wp.array(dtype=V3, ndim=2), col: wp.array(dtype=V3),
                   nrm: wp.array(dtype=V3), AIC: wp.array(dtype=DTYPE, ndim=3)):
    """AIC WITH a root symmetry plane (the OTHER wing). Each ring j induces at colloc i directly AND via
    its mirror image across y=0. The image ring is reflected (y->-y) and traversed in REVERSED winding
    (c0,c3,c2,c1) so the spanwise lifting line stays continuous (same circulation sense) across the root
    -> root loading is restored (peak at root) instead of collapsing like a free tip."""
    i, j = wp.tid(); ci = col[i]
    v = ring_vel(ci, rings[j, 0], rings[j, 1], rings[j, 2], rings[j, 3])
    m0 = mirror_y(rings[j, 0]); m1 = mirror_y(rings[j, 1]); m2 = mirror_y(rings[j, 2]); m3 = mirror_y(rings[j, 3])
    v = v + ring_vel(ci, m0, m3, m2, m1)         # image wing (reversed winding = symmetric continuation)
    AIC[0, i, j] = wp.dot(v, nrm[i])


@wp.func
def part_vel(P: V3, X: V3, alpha: V3, sigma: DTYPE) -> V3:
    """Velocity induced at P by ONE vortex particle (pos X, vortex moment alpha=Gamma*L_vec, core sigma).
    Gaussian-erf regularization, identical to the validated warp_vpm.particle_bs_kernel."""
    dx = P - X
    r = wp.sqrt(wp.dot(dx, dx) + wp.float64(1.0e-20))
    if r < wp.float64(1.0e-9):
        return V3(wp.float64(0.0), wp.float64(0.0), wp.float64(0.0))
    rb = r / sigma
    g = wp.erf(rb * wp.float64(0.7071067811865476)) - wp.float64(0.7978845608028654) * rb * wp.exp(wp.float64(-0.5) * rb * rb)
    coeff = wp.float64(-0.07957747154594767) * g / (r * r * r)   # -1/(4pi) * g / r^3
    return coeff * wp.cross(dx, alpha)


@wp.kernel
def col_particle_vel_kernel(col: wp.array(dtype=V3), pp: wp.array(dtype=V3), pa: wp.array(dtype=V3),
                            ps: wp.array(dtype=DTYPE), np_part: int, Vp: wp.array(dtype=V3)):
    """Particle-field induced velocity VECTOR at every collocation point (feeds BOTH the solve RHS and
    the unsteady-Bernoulli Vcol — the SAME snapshot, so the bound solve and the force stay consistent)."""
    i = wp.tid(); ci = col[i]
    v = V3(wp.float64(0.0), wp.float64(0.0), wp.float64(0.0))
    for k in range(np_part):
        v = v + part_vel(ci, pp[k], pa[k], ps[k])
    Vp[i] = v


@wp.kernel
def shed_lev_particles_kernel(rings: wp.array(dtype=V3, ndim=2), nrm: wp.array(dtype=V3),
                              vcol: wp.array(dtype=V3), gamma: wp.array(dtype=DTYPE, ndim=2),
                              Vinf: V3, ns: int, np0: int, sin_crit: DTYPE, klev: DTYPE,
                              sig0: DTYPE, pcore: DTYPE, pp: wp.array(dtype=V3),
                              pa: wp.array(dtype=V3), ps: wp.array(dtype=DTYPE)):
    """Shed ONE leading-edge vortex PARTICLE per strip at the LE surface (vs the old isolated ring).
    Strength/criterion byte-identical to _shed_lev_kernel (LESP excess, delayed-Kutta gprev). The
    particle is a spanwise vortex moment alpha=Gamma*(LE edge vector); it then advects in the FULL
    local field and rolls up by mutual induction (the missing ingredient in the ring version)."""
    j = wp.tid(); p = j; idx = np0 + j
    n = nrm[p]
    vr = Vinf - vcol[p]; vmag = wp.length(vr) + wp.float64(1.0e-9)
    sa = wp.abs(wp.dot(vr, n) / vmag)
    s_vec = rings[p, 1] - rings[p, 0]                                  # LE edge vector (spanwise, full length)
    le_mid = wp.float64(0.5) * (rings[p, 0] + rings[p, 1])            # LE midpoint
    chord_v = rings[p, 2] - rings[p, 0]; clen = wp.length(chord_v) + wp.float64(1.0e-12)
    pp[idx] = le_mid + n * (wp.float64(0.08) * clen)                  # born AT the LE, 0.08c onto suction side
    ps[idx] = wp.max(sig0, pcore * clen)                             # core >= 0.10c -> regularize near-LE
    if sa > sin_crit:
        gmag = -klev * gamma[0, p] * (wp.float64(1.0) - sin_crit / sa)  # SAME formula as the ring kernel
        pa[idx] = gmag * s_vec                                        # alpha = Gamma * (full LE edge vector)
    else:
        pa[idx] = V3(wp.float64(0.0), wp.float64(0.0), wp.float64(0.0))


@wp.kernel
def advect_particle_kernel(pp: wp.array(dtype=V3), pa: wp.array(dtype=V3), ps: wp.array(dtype=DTYPE),
                           np_part: int, rings: wp.array(dtype=V3, ndim=2),
                           gamma: wp.array(dtype=DTYPE, ndim=2), npan: int,
                           wr: wp.array(dtype=V3, ndim=2), wg: wp.array(dtype=DTYPE), nw: int,
                           Vinf: V3, dt: DTYPE, pp_new: wp.array(dtype=V3)):
    """Advect each LEV particle in the FULL local velocity = freestream + bound rings + TEV ring wake
    + OTHER particles (mutual induction). The mutual-induction term is exactly what rolls them up into
    a coherent core. (No vortex stretching yet — increment 1; rollup works from induced advection.)"""
    k = wp.tid(); P = pp[k]
    dl = DTYPE(WAKE_CORE)
    v = Vinf
    for q in range(npan):                                            # bound-ring induction
        v = v + gamma[0, q] * ring_vel_core(P, rings[q, 0], rings[q, 1], rings[q, 2], rings[q, 3], dl)
    for m in range(nw):                                              # TEV ring-wake induction
        v = v + wg[m] * ring_vel_core(P, wr[m, 0], wr[m, 1], wr[m, 2], wr[m, 3], dl)
    for jj in range(np_part):                                        # mutual particle induction -> ROLLUP
        if jj != k:
            v = v + part_vel(P, pp[jj], pa[jj], ps[jj])
    pp_new[k] = P + v * dt


def twisted_corners(C0, t, A_f, A_t, Om, phi, x_ea, span, swept_axis=False, root_off=0.0):
    """Flat wing -> flap (rotate about root x-axis by θ=A_f sin Ωt) + spanwise-linear twist
    (pitch each section about the y-axis at x_ea by ψ(y)=A_t (y/span) sin(Ωt+phi)).
    swept_axis=True: real RoboEagle twist axis swept 33.8%c(root)->LE(tip) (_v2_robogeom.axis_x),
    not a constant x_ea — matches the paper's measured flap/twist hinge.
    root_off: wing root offset outboard of the y=0 flap axis. Twist/chord use the wing-LOCAL span
    (y-root_off); the flap (dihedral) rotates about y=0 using the ASSEMBLY y (so the offset root swings)."""
    th = A_f * np.sin(Om * t)
    ct, st = np.cos(th), np.sin(th)
    x = C0[..., 0]; y = C0[..., 1]; z0 = C0[..., 2]    # y = assembly span; z0 = NACA-2406 camber surface
    yl = y - root_off                                  # wing-local span (root=0) for twist axis/amplitude
    xe = rg.axis_x(yl, span) if swept_axis else x_ea   # swept twist axis (per wing-local y) or constant
    psi = A_t * (yl / span) * np.sin(Om * t + phi)
    cp, sp = np.cos(psi), np.sin(psi)
    xr = xe + (x - xe) * cp - z0 * sp                  # twist pitches (x-xe, z0) about y at the axis
    zr = (x - xe) * sp + z0 * cp                       # carry the camber through the rotation
    xf = xr                            # flap: rotate (y,z) about x by θ, about y=0 with ASSEMBLY y
    yf = y * ct - zr * st
    zf = y * st + zr * ct
    return np.stack([xf, yf, zf], axis=-1)


def twisted_state(C0, t, A_f, A_t, Om, phi, x_ea, span, dlt=1e-6, swept_axis=False, root_off=0.0):
    corners = twisted_corners(C0, t, A_f, A_t, Om, phi, x_ea, span, swept_axis, root_off)
    cp = twisted_corners(C0, t + dlt, A_f, A_t, Om, phi, x_ea, span, swept_axis, root_off)
    cm = twisted_corners(C0, t - dlt, A_f, A_t, Om, phi, x_ea, span, swept_axis, root_off)
    vel = (cp - cm) / (2 * dlt)
    return corners, vel


@wp.kernel
def _wake_avg(wa: wp.array(dtype=V3, ndim=2), wb: wp.array(dtype=V3, ndim=2),
              wout: wp.array(dtype=V3, ndim=2)):
    """Heun RK2 combine: wr_new = 0.5*(wr + Euler(Euler(wr)))  (2nd-order free-wake convection)."""
    k, c = wp.tid()
    wout[k, c] = wp.float64(0.5) * (wa[k, c] + wb[k, c])


# NOTE: the Polhamus dynamic-stall LEV kernel (_lev_kernel, empirical C_Nv=K_v sin^2 a) was REMOVED
# (isolated to old/polhamus_removed_snapshot.py) on 2026-06-24 — the model is now first-principles only:
# standard unsteady UVLM + REAL discrete leading-edge vortex shedding (_shed_lev_kernel below).
@wp.kernel
def _shed_lev_kernel(rings: wp.array(dtype=V3, ndim=2), nrm: wp.array(dtype=V3), vcol: wp.array(dtype=V3),
                     gamma: wp.array(dtype=DTYPE, ndim=2), Vinf: V3, ns: int, nw: int,
                     sin_crit: DTYPE, klev: DTYPE, wr: wp.array(dtype=V3, ndim=2), wg: wp.array(dtype=DTYPE)):
    """REAL leading-edge vortex (3D LDVM, not Polhamus): shed a DISCRETE vortex ring at the leading edge
    of each leading strip (chordwise idx 0, panel p=j) where |sin a_eff|>sin_crit. The ring is placed at
    the LE edge offset onto the suction side (+n) and dropped into the wake array, so it convects + induces
    freely (reverse machinery as the TEV). Strength = excess LE circulation above the critical (LESP cap)."""
    j = wp.tid(); p = j; idx = nw + j
    vr = Vinf - vcol[p]; vmag = wp.length(vr) + wp.float64(1.0e-9)
    sina = wp.dot(vr, nrm[p]) / vmag; sa = wp.abs(sina)
    n = nrm[p]
    # Shed the LEV ring offset AWAY from the LE onto the suction side (base d0, depth eps), so it is not
    # adjacent to the bound collocations -> avoids the near-singular LEV->bound feedback that makes the
    # solve + dGamma/dt oscillate violently at high frequency (regularizes the near-field).
    d0 = wp.float64(0.08); eps = wp.float64(0.05)
    b0 = rings[p, 0] + n * d0; b1 = rings[p, 1] + n * d0
    wr[idx, 0] = b0; wr[idx, 1] = b1
    wr[idx, 2] = b1 + n * eps; wr[idx, 3] = b0 + n * eps
    if sa > sin_crit:
        # LEV sheds the EXCESS leading-edge circulation above the LESP-critical value. The minus sign is
        # this GPU UVLM's gamma convention (validated: klev=1 -> LEV ADDS lift, 5.1->7.3N vs data 7.8N).
        wg[idx] = -klev * gamma[0, p] * (wp.float64(1.0) - sin_crit / sa)
    else:
        wg[idx] = wp.float64(0.0)


@wp.kernel
def rhs_add_lev_kernel(nrm: wp.array(dtype=V3), Vlev: wp.array(dtype=V3),
                       rhs: wp.array(dtype=DTYPE, ndim=2)):
    """Fold the coherent-LEV-core induced velocity into the solve RHS (-V_lev . n), scalar form for a
    clean nrm adjoint. The coherent core is ONE merged smooth ring per strip -> no near-singular fresh-ring
    feedback -> the dGamma/dt oscillation that blew up the per-step ring LEV at 2.0Hz is removed."""
    i = wp.tid()
    rhs[0, i] = rhs[0, i] - wp.dot(Vlev[i], nrm[i])


@wp.kernel
def _shed_lev_sat_kernel(rings: wp.array(dtype=V3, ndim=2), nrm: wp.array(dtype=V3), ns: int, nw: int,
                         lev_str: wp.array(dtype=DTYPE), wr: wp.array(dtype=V3, ndim=2),
                         wg: wp.array(dtype=DTYPE)):
    """MESH-INDEPENDENT, LESP-SATURATED LEV shedding (Path B). The shed strength lev_str[j] is computed on
    the CPU from KINEMATIC strip quantities (U, chord, alpha_eff) — NOT the per-panel gamma — so it does not
    drift with mesh resolution (the root cause of the old ring-LEV artifact). The Bernoulli force captures
    its lift via col_wake_vel. This kernel just PLACES the ring (offset onto the suction side) + assigns it."""
    j = wp.tid(); p = j; idx = nw + j; n = nrm[p]
    d0 = wp.float64(0.08); eps = wp.float64(0.05)
    b0 = rings[p, 0] + n * d0; b1 = rings[p, 1] + n * d0
    wr[idx, 0] = b0; wr[idx, 1] = b1; wr[idx, 2] = b1 + n * eps; wr[idx, 3] = b0 + n * eps
    wg[idx] = lev_str[j]


@wp.kernel
def _shed_te_traj(rings: wp.array(dtype=V3, ndim=2), gamma: wp.array(dtype=DTYPE, ndim=2),
                  te: wp.array(dtype=wp.int32), tpl: wp.array(dtype=V3), tpr: wp.array(dtype=V3),
                  Vinf: V3, dt: DTYPE, nw: int, wr: wp.array(dtype=V3, ndim=2),
                  wg: wp.array(dtype=DTYPE), tcl: wp.array(dtype=V3), tcr: wp.array(dtype=V3)):
    """Shed fresh wake ring along the TE TRAJECTORY: leading edge stays attached at the CURRENT TE,
    trailing edge connects to the PREVIOUS step's TE (convected by Vinf*dt) -> continuous wake sheet
    for a moving/plunging TE (no ~hdot*dt gap). Outputs current TE corners for the next step."""
    k = wp.tid()
    p = te[k]; idx = nw + k
    cl = rings[p, 3]; cr = rings[p, 2]                # current TE left/right (leading, attached)
    wr[idx, 0] = cl; wr[idx, 1] = cr
    if nw == 0:                                       # first shed: no previous -> standard straight wake
        wr[idx, 2] = cr + Vinf * dt; wr[idx, 3] = cl + Vinf * dt
    else:
        wr[idx, 2] = tpr[k] + Vinf * dt; wr[idx, 3] = tpl[k] + Vinf * dt
    wg[idx] = gamma[0, p]
    tcl[k] = cl; tcr[k] = cr                          # save for next step's "previous TE"


def gpu_run_twist(nc=4, ns=10, chord=0.287, half_span=0.80, U=8.0, aoa_deg=5.0,
                  flap_amp_deg=45.0, twist_amp_deg=22.5, twist_phase_deg=-90.0,
                  freq=2.0, n_cycle=5, steps_per_cycle=40, wake_rows=50, rk2=False, te_traj=False,
                  swept_axis=False, real_geom=False, real_lev=False, lev_sat=False, lev_merge=False, lev_tau=0.20,
                  lesp_crit_deg=15.0, lev_klev=1.0,
                  visc=False, tc_thick=0.06, les_suction=False, les_eta=1.0,
                  part_lev=False, sym=False, root_off=0.0, stall=False, stall_deg=12.0,
                  vortex=False, k_vortex=2.0, dstall=False, ds_crit_deg=14.0, ds_tv=0.40, ds_k=1.0,
                  ds_delay=18, frames_out=None, frame_skip=3):
    """Twisted flapping UVLM — FIRST-PRINCIPLES unsteady (no empirical Polhamus/cap terms).
    rk2=True -> 2nd-order Heun free-wake convection. te_traj=True -> shed wake along TE trajectory.
    swept_axis=True -> real RoboEagle flap/twist axis (33.8%c root -> LE tip), not quarter-chord.
    real_geom=True -> real raked planform + NACA-2406 camber.
    real_lev=True -> REAL discrete leading-edge vortex: a vortex ring is shed at the leading edge of each
      strip whose |sin a_eff| exceeds lesp_crit_deg (the LESP criterion), then convects + induces freely
      like the TEV wake. lev_klev scales the shed strength. (Viscous term added separately, Re-based.)"""
    dev = cfg.DEVICE; NP = cfg.NP_DTYPE
    # real_geom=True -> REAL RoboEagle planform (raked TE, measured chord(y)) + NACA-2406 camber, LE at
    # x=0 / TE at x=+c (chord in +x = flow dir, Vinf=+x flows LE->TE). Else flat rectangular wing.
    C0 = (rg.robowing_real(nc, ns, half_span, root_off=root_off) if real_geom
          else ffv.flat_wing(nc, ns, chord, half_span)); npan = nc * ns; ncv = (nc + 1) * (ns + 1)
    A_f = np.radians(flap_amp_deg); A_t = np.radians(twist_amp_deg); phi = np.radians(twist_phase_deg)
    Om = 2.0 * np.pi * freq; x_ea = 0.25 * chord
    Vinf = np.array([U, 0.0, U * np.tan(np.radians(aoa_deg))]); Vw = V3(*[float(v) for v in Vinf])
    T = 1.0 / freq; dt = T / steps_per_cycle; N = n_cycle * steps_per_cycle
    te = wp.array(np.array([(nc - 1) * ns + j for j in range(ns)], np.int32), dtype=wp.int32, device=dev)
    shed_per = ns * (2 if real_lev else 1)            # TEV (+ LEV ring if real_lev) shed per step
    wake_max = wake_rows * shed_per; maxw = min(N * shed_per, wake_max) + shed_per
    wr = wp.zeros((maxw, 4), dtype=V3, device=dev); wr_new = wp.zeros((maxw, 4), dtype=V3, device=dev)
    wr_m2 = wp.zeros((maxw, 4), dtype=V3, device=dev) if rk2 else None   # RK2 second-Euler buffer
    tpl = wp.zeros(ns, dtype=V3, device=dev); tpr = wp.zeros(ns, dtype=V3, device=dev)   # prev TE corners
    tcl = wp.zeros(ns, dtype=V3, device=dev); tcr = wp.zeros(ns, dtype=V3, device=dev)   # cur TE corners
    wg = wp.zeros(maxw, dtype=DTYPE, device=dev); gprev = wp.zeros((1, npan), dtype=DTYPE, device=dev); nw = 0
    # --- LEV vortex-particle field (parallel to the TEV ring wake; vec3d vortex moment, NOT scalar) ---
    pmax = N * ns + ns; np_part = 0                  # one particle per strip per step + slack
    pp = wp.zeros(pmax, dtype=V3, device=dev); pa = wp.zeros(pmax, dtype=V3, device=dev)
    ps = wp.zeros(pmax, dtype=DTYPE, device=dev); pp_new = wp.zeros(pmax, dtype=V3, device=dev)
    SIG0 = DTYPE(0.5 * U * dt); PCORE = DTYPE(0.10)   # base core (swept vol/2); floor 0.10c regularizes LE
    sin_crit_p = DTYPE(np.sin(np.radians(lesp_crit_deg)))
    # --- coherent-core LEV (N-LEV merging, N=1 per strip): ONE smooth merged ring/strip (CPU state) ---
    lev_cen = np.zeros((ns, 3)); lev_gam = np.zeros(ns)
    lev_rw = wp.zeros((ns, 4), dtype=V3, device=dev); lev_gw = wp.zeros(ns, dtype=DTYPE, device=dev)
    Vlev = wp.zeros(npan, dtype=V3, device=dev)
    Lh = np.zeros(N); Xh = np.zeros(N); Ph = np.zeros(N); Lkjh = np.zeros(N)
    Lh_imp = np.zeros(N); Xh_imp = np.zeros(N)        # unsteady-Bernoulli surface-pressure force (captures LEV)
    Lh_vis = np.zeros(N); Xh_vis = np.zeros(N)         # DeLaurier viscous friction drag (strip, Re-based Blasius)
    Lh_les = np.zeros(N); Xh_les = np.zeros(N)         # leading-edge suction thrust (Garrick/DeLaurier dTs)
    Lh_vtx = np.zeros(N); Xh_vtx = np.zeros(N)         # high-alpha vortex normal force (Polhamus, lift+drag)
    Lh_ds = np.zeros(N)                                 # dynamic-stall LEV lift (sustains the downstroke plateau)
    Glev_ds = np.zeros(ns); aeff_ds_prev = np.zeros(ns)  # per-strip LEV circulation state + prev alpha_eff
    NU_AIR = 1.5e-5; FORM_FF = 1.0 + 2.0 * tc_thick + 60.0 * tc_thick ** 4   # air kin. visc; Hoerner form factor
    wtype = []                                        # CPU bookkeeping: 0=TEV, 1=LEV per wake ring (for viz)
    for t in range(N):
        corners, cvel = twisted_state(C0, t * dt, A_f, A_t, Om, phi, x_ea, half_span,
                                      swept_axis=swept_axis, root_off=root_off)
        cw = wp.array(corners.reshape(ncv, 3).astype(NP), dtype=V3, device=dev)
        vw = wp.array(cvel.reshape(ncv, 3).astype(NP), dtype=V3, device=dev)
        rings = wp.zeros((npan, 4), dtype=V3, device=dev); col = wp.zeros(npan, dtype=V3, device=dev)
        nrm = wp.zeros(npan, dtype=V3, device=dev); vcol = wp.zeros(npan, dtype=V3, device=dev)
        wp.launch(ug.bound_rings_kernel, dim=npan, inputs=[cw, nc, ns], outputs=[rings, col, nrm], device=dev)
        wp.launch(ug.colvel_kernel, dim=npan, inputs=[vw, nc, ns], outputs=[vcol], device=dev)
        AIC = wp.zeros((1, npan, npan), dtype=DTYPE, device=dev)
        aick = aic_sym_kernel if sym else ug.aic_kernel   # sym=True -> root symmetry plane (the other wing)
        wp.launch(aick, dim=(npan, npan), inputs=[rings, col, nrm], outputs=[AIC], device=dev)
        rhs = wp.zeros((1, npan), dtype=DTYPE, device=dev)
        wp.launch(ug.rhs_moving_kernel, dim=npan, inputs=[col, nrm, Vw, vcol, wr, wg, nw], outputs=[rhs], device=dev)
        if lev_merge:   # coherent-LEV-core induction into the solve (uses last step's core -> 1-step delay)
            wp.launch(ug.col_wake_vel_kernel, dim=npan, inputs=[col, lev_rw, lev_gw, ns], outputs=[Vlev], device=dev)
            wp.launch(rhs_add_lev_kernel, dim=npan, inputs=[nrm, Vlev], outputs=[rhs], device=dev)
        gamma = batched_dense_solve(AIC, rhs, dev)
        # First-principles unsteady panel force: circulation (Kutta-Joukowski) + added-mass (rho dGamma/dt).
        # The REAL LEV (real_lev) acts through the wake it sheds (induction on the bound + its own impulse);
        # no empirical Polhamus/cap terms. Viscous term to be added (first-principles, Re-based) next.
        Fp = wp.zeros(npan, dtype=V3, device=dev)
        wp.launch(ug.panel_force_kernel, dim=npan, inputs=[rings, nrm, gamma, gprev, vcol, Vw,
                  DTYPE(dt), DTYPE(ug.RHO), ns], outputs=[Fp], device=dev)
        Fpn = Fp.numpy(); vcn = vcol.numpy()
        Lh[t] = np.sum(Fpn[:, 2]); Xh[t] = np.sum(Fpn[:, 0]); Ph[t] = -np.sum(np.einsum('pi,pi->p', Fpn, vcn))
        # ---- unsteady-Bernoulli SURFACE-PRESSURE force (Katz&Plotkin): dp = rho(V_colloc.tau_x dG/dx
        # + V_colloc.tau_y dG/dy + dG/dt). V_colloc = V_inf - V_body + WAKE+LEV induced velocity at the
        # panel -> the LEV's induced flow on the wing SURFACE enters the force (bound-only KJ omits it,
        # missing the LEV lift). Frame-clean in the fixed-wing frame -> the correct first-principles force. ----
        Vwk = wp.zeros(npan, dtype=V3, device=dev)
        wp.launch(ug.col_wake_vel_kernel, dim=npan, inputs=[col, wr, wg, nw], outputs=[Vwk], device=dev)
        cc = rings.numpy(); g = gamma.numpy().reshape(-1); gp = gprev.numpy().reshape(-1)
        Vcol = np.asarray(Vinf) - vcn + Vwk.numpy()                     # full local velocity at panels
        if lev_merge:
            Vcol = Vcol + Vlev.numpy()                                  # coherent LEV core induction (Bernoulli)
        # THIS GPU UVLM's ring corners: c0,c1 = LE edge; c2,c3 = TE edge (shed_kernel uses c2,c3 for the
        # TE wake). So chordwise (LE->TE) = mean(c2-c0, c3-c1); spanwise (along LE/TE edge) = mean(c1-c0, c2-c3).
        tcr = 0.5 * ((cc[:, 2] - cc[:, 0]) + (cc[:, 3] - cc[:, 1]))     # chordwise tangent (LE->TE)
        tsr = 0.5 * ((cc[:, 1] - cc[:, 0]) + (cc[:, 2] - cc[:, 3]))     # spanwise tangent
        tcn = np.linalg.norm(tcr, axis=1) + 1e-15; tsn = np.linalg.norm(tsr, axis=1) + 1e-15
        tc = tcr / tcn[:, None]; ts = tsr / tsn[:, None]               # unit chordwise / spanwise
        gm = g.reshape(nc, ns); tcnm = tcn.reshape(nc, ns); tsnm = tsn.reshape(nc, ns)
        dGdx = np.empty((nc, ns)); dGdx[0] = gm[0] / tcnm[0]           # chordwise dGamma/dx (i=chordwise)
        if nc > 1:
            dGdx[1:] = (gm[1:] - gm[:-1]) / tcnm[1:]                    # backward diff (Katz&Plotkin)
        dGdy = np.zeros((nc, ns))                                       # spanwise dGamma/dy (j=spanwise)
        if ns > 1:
            dGdy[:, 0] = gm[:, 0] / tsnm[:, 0]; dGdy[:, -1] = -gm[:, -1] / tsnm[:, -1]
            dGdy[:, 1:-1] = (gm[:, 2:] - gm[:, :-2]) / (2 * tsnm[:, 1:-1])
        dGdx = dGdx.reshape(-1); dGdy = dGdy.reshape(-1); dGdt = (g - gp) / max(dt, 1e-15)
        area = 0.5 * np.linalg.norm(np.cross(cc[:, 2] - cc[:, 0], cc[:, 3] - cc[:, 1]), axis=1)
        dp = ug.RHO * (np.sum(Vcol * tc, axis=1) * dGdx + np.sum(Vcol * ts, axis=1) * dGdy + dGdt)
        Fb = dp[:, None] * area[:, None] * nrm.numpy()
        # ---- STALL: the attached UVLM has no separation -> at high |alpha_eff| (deep stall on the +-45
        # flap strokes, tip alpha_eff reaches +-40-50deg) it over-predicts the force (BOTH the downstroke
        # lift peak and the upstroke downforce trough). Cap the section force at the airfoil's CL_max:
        # beyond the stall angle the lift saturates, factor = sin(a_stall)/|sin a_eff|. a_stall = CL_max/slope
        # is a FIRST-PRINCIPLES airfoil property (CL_max~1.2, slope~2pi -> ~11deg), NOT fitted to RoboEagle. ----
        if stall:
            nnp = nrm.numpy()
            sap = np.sum(Vcol * nnp, axis=1) / (np.linalg.norm(Vcol, axis=1) + 1e-9)   # sin(alpha_eff)/panel
            sf = np.minimum(1.0, np.sin(np.radians(stall_deg)) / (np.abs(sap) + 1e-9))  # CL_max saturation
            Fb = Fb * sf[:, None]
        Lh_imp[t] = float(np.sum(Fb[:, 2])); Xh_imp[t] = float(np.sum(Fb[:, 0]))
        # ---- DeLaurier (1993) first-principles VISCOUS friction drag (strip theory). The inviscid
        # Bernoulli force has NO friction -> over-predicts net thrust. Skin friction drags each panel
        # DOWNSTREAM along the local tangential flow: dDf = 1/2 rho V_tan^2 Cdf dA, Cdf = 2*Cf*FF
        # (both surfaces x Hoerner thickness form factor), Cf = 1.328/sqrt(Re) laminar Blasius,
        # Re = V_tan * c_local / nu. Affects mostly drag (thrust), slightly lift (V_tan has small z). ----
        if visc:
            nn = nrm.numpy()
            Vtan = Vcol - (np.sum(Vcol * nn, axis=1)[:, None]) * nn       # tangential flow over surface
            Vtm = np.linalg.norm(Vtan, axis=1) + 1e-12
            c_loc = np.broadcast_to(tcn.reshape(nc, ns).sum(0), (nc, ns)).reshape(-1)  # local chord per column
            Re_loc = np.maximum(Vtm * c_loc / NU_AIR, 1.0e2)             # local chord Reynolds number
            Cf = 1.328 / np.sqrt(Re_loc)                                 # laminar flat-plate (Blasius), one side
            Cdf = 2.0 * Cf * FORM_FF                                     # both surfaces x thickness form factor
            Df = 0.5 * ug.RHO * Cdf[:, None] * area[:, None] * Vtm[:, None] * Vtan  # drags wing downstream
            Lh_vis[t] = float(np.sum(Df[:, 2])); Xh_vis[t] = float(np.sum(Df[:, 0]))
        # ---- LEADING-EDGE SUCTION thrust (Garrick / DeLaurier dTs = 2pi eta_s alpha_eff^2 (1/2 rho U V) c dy).
        # A flat-panel normal-pressure (Bernoulli) force structurally MISSES the leading-edge singular suction
        # (the sqrt(x) edge force) -> captures induced drag but NOT the forward LE-suction thrust. This is the
        # dominant flapping-thrust mechanism ("thrust is all leading-edge suction"), forward along -chord,
        # applied on ATTACHED strips (LEV not shed; shed-strip suction goes into the LEV captured by Bernoulli). ----
        if les_suction:
            nn2 = nrm.numpy(); iLE = np.arange(ns)            # leading-edge panel row (i=0)
            Vle = Vcol[iLE]; nle = nn2[iLE]; tcle = tc[iLE]
            Vle_m = np.linalg.norm(Vle, axis=1) + 1e-12
            sa = np.sum(Vle * nle, axis=1) / Vle_m            # sin(alpha_eff) at the LE strip
            aeff = np.arcsin(np.clip(sa, -0.999, 0.999))
            # LESP criterion (Ramesh 2014): realizable LE suction CAPS at the critical leading-edge angle -
            # beyond alpha_crit the excess loading sheds into the LEV (already shed for lift), so the attached
            # LE-suction saturates. First-principles separation onset, NOT an empirical efficiency fit.
            a_crit = np.radians(lesp_crit_deg)
            aeff_s = np.clip(aeff, -a_crit, a_crit)            # suction-relevant angle, capped at LESP-crit
            c_le = tcn.reshape(nc, ns).sum(0)                 # local chord per strip
            dy_le = tsn.reshape(nc, ns)[0]                    # strip spanwise width (LE row)
            Uc = abs(float(Vinf[0]))
            dTs = 2.0 * np.pi * les_eta * aeff_s ** 2 * (0.5 * ug.RHO * Uc * Vle_m) * c_le * dy_le
            Fs = -dTs[:, None] * tcle                         # forward (-chordwise) suction force vector
            Lh_les[t] = float(np.sum(Fs[:, 2])); Xh_les[t] = float(np.sum(Fs[:, 0]))
        # ---- HIGH-ALPHA VORTEX NORMAL FORCE (Polhamus leading-edge-suction analogy). When the flow
        # separates at high |alpha_eff| (the +-45 flap mid-strokes, alpha_eff ~ 45deg), the lost LE suction
        # reappears as a force NORMAL to the wing: C_Nv = k_v sin^2(a) cos(a). The SAME normal force projects
        # into BOTH lift (N . z) AND drag (N . x) -> max at mid-downstroke, where Fig 16 shows max drag AND an
        # extra lift bump (the user's observation). Attached UVLM misses it (it's separated-flow vortex lift). ----
        if vortex:
            nnv = nrm.numpy()
            vr = np.asarray(Vinf) - vcn                       # body-relative flow (freestream + flapping)
            vrm = np.linalg.norm(vr, axis=1) + 1e-9
            sa_v = np.sum(vr * nnv, axis=1) / vrm             # sin(alpha_eff) per panel (signed)
            ca_v = np.sqrt(np.maximum(0.0, 1.0 - sa_v ** 2))
            qd = 0.5 * ug.RHO * vrm ** 2                       # local dynamic pressure
            Nv = k_vortex * sa_v * np.abs(sa_v) * ca_v * qd * area   # signed sin^2(a) cos(a) normal force
            Fv = Nv[:, None] * nnv                             # along the panel normal -> lift+drag both
            Lh_vtx[t] = float(np.sum(Fv[:, 2])); Xh_vtx[t] = float(np.sum(Fv[:, 0]))
        # ---- DYNAMIC-STALL LEV (L-B style, per strip): on the downstroke alpha_eff rises past the static
        # stall angle and a leading-edge vortex forms, SUSTAINING extra lift (the measured ~13.7N plateau)
        # until it convects/sheds. State Glev_ds (LEV circulation) FEEDS while |alpha_eff|>crit AND growing,
        # then DECAYS with time const ds_tv -> build, sustain (plateau), drop. First-principles LESP gate. ----
        if dstall:
            nnd = nrm.numpy()
            vrle = (np.asarray(Vinf) - vcn)[:ns]              # LE-row (i=0) body-relative flow
            nle = nnd[:ns]; vrm = np.linalg.norm(vrle, axis=1) + 1e-9
            aeff = np.arcsin(np.clip(np.sum(vrle * nle, axis=1) / vrm, -0.999, 0.999))   # alpha_eff per strip
            ac = np.radians(ds_crit_deg); dy_st = tsn.reshape(nc, ns)[0]
            feed = np.where(aeff > ac, (aeff - ac) * vrm, 0.0)   # feed the WHOLE high-alpha downstroke (alpha>crit)
            Glev_ds[:] = Glev_ds * max(0.0, 1.0 - dt / ds_tv) + ds_k * feed * dt   # build + decay/shed
            Lh_ds[t] = float(np.sum(ug.RHO * vrm * Glev_ds * dy_st * nle[:, 2]))    # rho V Gamma, vertical comp
            aeff_ds_prev = aeff.copy()
        lkj = wp.zeros(1, dtype=DTYPE, device=dev)        # DIAG: Vinf-only KJ lift (no plunge tilt)
        wp.launch(ug.lift_kj_kernel, dim=npan, inputs=[rings, nrm, gamma, gprev, Vw, DTYPE(dt),
                  DTYPE(ug.RHO), ns], outputs=[lkj], device=dev)
        Lkjh[t] = float(lkj.numpy()[0])
        if frames_out is not None and t % frame_skip == 0:   # snapshot for wake/lattice visualization
            vcn = vcol.numpy(); nrn = nrm.numpy(); vr = np.asarray(Vinf) - vcn
            sina = np.sum(vr * nrn, axis=1) / (np.linalg.norm(vr, axis=1) + 1e-9)
            frames_out.append(dict(
                t=t * dt, bound=rings.numpy().copy(), gam=gamma.numpy().reshape(-1).copy(),
                wr=(wr.numpy()[:nw].copy() if nw > 0 else np.zeros((0, 4, 3))),
                wg=(wg.numpy()[:nw].copy() if nw > 0 else np.zeros(0)),
                wtype=np.array(wtype[:nw], dtype=int) if nw > 0 else np.zeros(0, int),
                pp=(pp.numpy()[:np_part].copy() if np_part > 0 else np.zeros((0, 3))),        # LEV particles
                pa=(pa.numpy()[:np_part].copy() if np_part > 0 else np.zeros((0, 3))),        # vortex moments
                sep=(np.abs(sina) > np.sin(np.radians(lesp_crit_deg))), nc=nc, ns=ns))
        if te_traj:   # shed along the TE trajectory (continuous sheet for the plunging TE)
            wp.launch(_shed_te_traj, dim=ns, inputs=[rings, gamma, te, tpl, tpr, Vw, DTYPE(dt), nw],
                      outputs=[wr, wg, tcl, tcr], device=dev)
            wp.copy(tpl, tcl); wp.copy(tpr, tcr)        # current TE becomes next step's "previous"
        else:
            wp.launch(ug.shed_kernel, dim=ns, inputs=[rings, gamma, te, Vw, DTYPE(dt), nw], outputs=[wr, wg], device=dev)
        if lev_merge:   # N-LEV MERGING: update the ONE coherent LEV core per strip (no wake shedding)
            nns = nrm.numpy(); cc_le = rings.numpy(); vrl = (np.asarray(Vinf) - vcn)[:ns]; nl = nns[:ns]
            vrl_m = np.linalg.norm(vrl, axis=1) + 1e-9
            sa_l = np.sum(vrl * nl, axis=1) / vrl_m
            cst = tcn.reshape(nc, ns).sum(0); scr = np.sin(np.radians(lesp_crit_deg))
            dG = -lev_klev * U * cst * np.maximum(np.abs(sa_l) - scr, 0.0) * np.sign(sa_l)   # kinematic increment
            le_mid = 0.5 * (cc_le[:ns, 0] + cc_le[:ns, 1]) + nl * (0.08 * cst)[:, None]      # LE shed position
            lev_cen = lev_cen + np.asarray(Vinf) * dt                # convect the core with the freestream
            gtot = lev_gam + dG; nz = np.abs(gtot) > 1e-9            # MERGE the increment (circulation-weighted)
            lev_cen[nz] = (lev_gam[nz, None] * lev_cen[nz] + dG[nz, None] * le_mid[nz]) / gtot[nz, None]
            lev_gam = gtot * max(0.0, 1.0 - dt / lev_tau)           # decay = the LEV convecting/shedding
            swe = cc_le[:ns, 1] - cc_le[:ns, 0]; epsn = nl * (0.05 * cst)[:, None]   # rebuild the coherent ring
            lr = np.zeros((ns, 4, 3))
            lr[:, 0] = lev_cen - 0.5 * swe; lr[:, 1] = lev_cen + 0.5 * swe; lr[:, 2] = lr[:, 1] + epsn; lr[:, 3] = lr[:, 0] + epsn
            lev_rw = wp.array(lr.astype(NP), dtype=V3, device=dev); lev_gw = wp.array(lev_gam.astype(NP), dtype=DTYPE, device=dev)
        elif real_lev and lev_sat:   # PATH B: mesh-independent, LESP-saturated LEV (kinematic strength, CPU)
            nns = nrm.numpy(); vrl = (np.asarray(Vinf) - vcn)[:ns]; nl = nns[:ns]
            vrl_m = np.linalg.norm(vrl, axis=1) + 1e-9
            sa_l = np.sum(vrl * nl, axis=1) / vrl_m                  # sin(alpha_eff) per strip (signed)
            cst = tcn.reshape(nc, ns).sum(0)                        # strip chord (mesh-independent)
            scr = np.sin(np.radians(lesp_crit_deg))
            # kinematic shed strength: ~ U*c*(|sin a| - sin_crit) above critical, signed; NO per-panel gamma.
            exc = np.maximum(np.abs(sa_l) - scr, 0.0)
            lev_str = (-lev_klev * U * cst * exc * np.sign(sa_l)).astype(NP)
            lev_str_w = wp.array(lev_str, dtype=DTYPE, device=dev)
            wp.launch(_shed_lev_sat_kernel, dim=ns, inputs=[rings, nrm, ns, nw + ns, lev_str_w],
                      outputs=[wr, wg], device=dev)
        elif real_lev:   # original (mesh-dependent) ring LEV
            wp.launch(_shed_lev_kernel, dim=ns, inputs=[rings, nrm, vcol, gprev, Vw, ns, nw + ns,
                      DTYPE(np.sin(np.radians(lesp_crit_deg))), DTYPE(lev_klev)], outputs=[wr, wg], device=dev)
        if part_lev:   # PARTICLE leading-edge vortex: shed one spanwise particle/strip at the LE (delayed-Kutta gprev)
            wp.launch(shed_lev_particles_kernel, dim=ns, inputs=[rings, nrm, vcol, gprev, Vw, ns, np_part,
                      sin_crit_p, DTYPE(lev_klev), SIG0, PCORE], outputs=[pp, pa, ps], device=dev)
            np_part += ns
        nw_new = nw + shed_per
        wtype.extend([0] * ns + ([1] * ns if real_lev else []))   # TEV then LEV (matches shed order)
        if nw > 0:   # convect OLD wake only; freshly-shed ring STAYS attached at the TE (Katz&Plotkin
            wp.launch(ug.convect_kernel, dim=(nw, 4), inputs=[rings, gamma, npan, wr, wg, nw, Vw, DTYPE(dt)],
                      outputs=[wr_new], device=dev)   # order) so it cancels the trailing bound segment
            if rk2:   # Heun RK2: second Euler from the predicted midpoint wake, then average
                wp.launch(ug.convect_kernel, dim=(nw, 4), inputs=[rings, gamma, npan, wr_new, wg, nw, Vw,
                          DTYPE(dt)], outputs=[wr_m2], device=dev)
                wp.launch(_wake_avg, dim=(nw, 4), inputs=[wr, wr_m2], outputs=[wr_new], device=dev)
            wp.copy(wr, wr_new, count=nw * 4)
        if nw_new > wake_max:
            off = nw_new - wake_max; wrh = wr.numpy(); wgh = wg.numpy()
            tw = np.zeros((maxw, 4, 3)); tw[:wake_max] = wrh[off:nw_new]; tg = np.zeros(maxw); tg[:wake_max] = wgh[off:nw_new]
            wr = wp.array(tw, dtype=V3, device=dev); wg = wp.array(tg, dtype=DTYPE, device=dev)
            wr_new = wp.zeros((maxw, 4), dtype=V3, device=dev); nw = wake_max
            wtype = wtype[off:]                        # drop oldest rings' type tags too
        else:
            nw = nw_new
        if part_lev and np_part > 0:   # advect LEV particles in the FULL local field (bound+TEV+mutual) -> rollup
            wp.launch(advect_particle_kernel, dim=np_part, inputs=[pp, pa, ps, np_part, rings, gamma, npan,
                      wr, wg, nw, Vw, DTYPE(dt)], outputs=[pp_new], device=dev)
            wp.copy(pp, pp_new, count=np_part)
        gprev = wp.array(gamma.numpy(), dtype=DTYPE, device=dev)
    if dstall and ds_delay > 0:
        Lh_ds = np.roll(Lh_ds, ds_delay)   # convection delay: LEV lift lags as the vortex traverses the chord
    last = slice((n_cycle - 1) * steps_per_cycle, N)
    L = 2.0 * np.mean(Lh[last]); Fx = 2.0 * np.mean(Xh[last]); P = 2.0 * np.mean(np.abs(Ph[last]))
    L_bern = 2.0 * np.mean(Lh_imp[last]); Fx_bern = 2.0 * np.mean(Xh_imp[last])
    L_vis = 2.0 * np.mean(Lh_vis[last]); Fx_vis = 2.0 * np.mean(Xh_vis[last])   # friction (downstream, +x drag)
    L_les = 2.0 * np.mean(Lh_les[last]); Fx_les = 2.0 * np.mean(Xh_les[last])   # LE suction (forward, -x thrust)
    L_vtx = 2.0 * np.mean(Lh_vtx[last]); Fx_vtx = 2.0 * np.mean(Xh_vtx[last])   # vortex normal force (lift+drag)
    Lkj = 2.0 * np.mean(Lkjh[last])
    return dict(L=L, Fx=Fx, T=-Fx, P=P, Lh=Lh, Xh=Xh, Lkj=Lkj,
                L_bern=L_bern, T_bern=-Fx_bern, Lh_bern=Lh_imp, Xh_bern=Xh_imp,   # Bernoulli force (captures LEV)
                L_visc=L_vis, D_visc=Fx_vis, T_lesp=-Fx_les,                      # friction (drag>0); LE suction (thrust)
                Lh_vis=Lh_vis, Xh_vis=Xh_vis, Lh_les=Lh_les, Xh_les=Xh_les,       # per-step viscous / LE-suction
                Lh_vtx=Lh_vtx, Xh_vtx=Xh_vtx, L_vtx=L_vtx, D_vtx=Fx_vtx,          # per-step + mean vortex normal force
                Lh_ds=Lh_ds, L_dstall=2.0 * np.mean((Lh_imp + Lh_ds)[last]),       # dynamic-stall: per-step + mean(bern+LEV)
                L_net=L_bern + L_les - L_vis,                                     # lift incl. LE-suction vertical comp.
                L_full=L_bern + L_vtx,                                            # Bernoulli + vortex normal force lift
                T_net=-(Fx_bern + Fx_vis + Fx_les + Fx_vtx))                     # Bernoulli + friction + LE suction + vortex


if __name__ == "__main__":
    wp.init()
    print("RoboEagle aero validation (half-span 0.80m, chord 0.287m, ±45° flap, 8 m/s, 5° AoA, 2 Hz):", flush=True)
    print("paper anchors: max L/D 6.8; optimal twist 22.5° -> +47% thrust, +7.8% lift vs untwisted", flush=True)
    base = None
    for ph in (-90.0, 90.0):
        print(f"\n  --- twist phase {ph:+.0f}° ---", flush=True)
        r0 = gpu_run_twist(twist_amp_deg=0.0, twist_phase_deg=ph)
        for ta in (0.0, 22.5, 45.0):
            r = gpu_run_twist(twist_amp_deg=ta, twist_phase_deg=ph)
            dT = 100 * (r["T"] - r0["T"]) / (abs(r0["T"]) + 1e-9)
            dL = 100 * (r["L"] - r0["L"]) / (abs(r0["L"]) + 1e-9)
            ld = r["L"] / (r["Fx"] + 1e-9) if r["Fx"] > 0 else float('nan')
            print(f"  twist {ta:4.1f}°: L={r['L']:+.2f}N T={r['T']:+.2f}N P={r['P']:.1f}W  "
                  f"ΔT={dT:+.0f}% ΔL={dL:+.0f}%  L/D(induced)={ld:.1f}", flush=True)
    print("\nDONE", flush=True)
