"""Validate the UVLM wing aero against the REAL RoboEagle paper (Drones 2025, 9/8/535,
"Flapping-Twist Coupled..."). Geometry: half-span 0.80 m, root chord 0.287 m. Kinematics:
flap ±45°, spanwise-linear twist (0/22.5/45°) coupled to flap by a phase. Cruise 8 m/s, 5-8° AoA.
Measured anchors: max L/D 6.8; optimal twist (22.5°) gives +47% thrust, +7.8% lift vs untwisted.

We add twist to the validated flapping UVLM (flap_flight_validate's kernels) and check we reproduce
the twist GAIN relationship (a relative measure, robust to the absolute aero model; UVLM has only
induced drag so absolute L/D will exceed 6.8 until profile drag is added)."""
from __future__ import annotations

import os
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
                              sig0: DTYPE, pcore: DTYPE, sa_prev: wp.array(dtype=DTYPE),
                              pp: wp.array(dtype=V3),
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
    sgn = wp.sign(wp.dot(vr, n) / vmag)                              # SIGNED stroke: +1 downstroke, -1 upstroke
    # born ON the suction side, which FLIPS with the stroke (+n downstroke, -n upstroke) -> the up/down LEV
    # particles are mirror images that convect away and cancel at AoA=0 by construction (conserved, physical).
    pp[idx] = le_mid + n * (sgn * wp.float64(0.08) * clen)            # born AT the LE, on the real suction side
    ps[idx] = wp.max(sig0, pcore * clen)                             # core >= 0.10c -> regularize near-LE
    rising = sa - sa_prev[p]                                          # d|LESP|/dt: LEV grows on the BUILD-UP phase
    sa_prev[p] = sa                                                  # store for next step's rate
    # SHED only while the LE suction is SUPERCRITICAL and INCREASING (Ramesh/flap_ldvm up-stroke gate). This is
    # self-adjusting: at AoA=0 both strokes build symmetrically -> symmetric shedding -> cancels; at AoA>0 the
    # lift-producing stroke builds MORE -> downstroke-dominated net lift (the cruise overshoot) WITHOUT a fixed
    # asymmetry that would break AoA=0. The detach (stop shedding on the decreasing phase) gives rise-peak-drop.
    if sa > sin_crit and rising > wp.float64(0.0):
        gmag = -klev * sgn * vmag * clen * (sa - sin_crit)           # mesh-independent, signed, LESP excess
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


def twisted_corners(C0, t, A_f, A_t, Om, phi, x_ea, span, swept_axis=False, root_off=0.0, ramp=None):
    """Flat wing -> flap (rotate about root x-axis by θ=A_f sin Ωt) + spanwise-linear twist
    (pitch each section about the y-axis at x_ea by ψ(y)=A_t (y/span) sin(Ωt+phi)).
    swept_axis=True: real RoboEagle twist axis swept 33.8%c(root)->LE(tip) (_v2_robogeom.axis_x),
    not a constant x_ea — matches the paper's measured flap/twist hinge.
    root_off: wing root offset outboard of the y=0 flap axis. Twist/chord use the wing-LOCAL span
    (y-root_off); the flap (dihedral) rotates about y=0 using the ASSEMBLY y (so the offset root swings).
    ramp=(pmax_rad, t_start, t_dur): HIRATO pitch-ramp mode — UNIFORM pitch about x_ea, smootherstep
    0->pmax over [t_start, t_start+t_dur], NO flap/twist (validates LEV rollup vs Hirato Fig.11)."""
    if ramp is not None:
        # CANONICAL ELDREDGE/GRANLUND smoothed pitch ramp-hold (matches Hirato Fig.9): near-constant pitch rate
        # 2K (convective) with smoothed corners, evaluated from a numerically-stable lncosh. SIGN: nose-UP
        # (positive aerodynamic AoA -> positive lift) needs psi<0 in this frame (Vinf=[U,0,0], wing normal
        # n=(-sin psi,0,cos psi) -> Vinf.n=-U sin psi>0 only for psi<0), matching the freestream-tilt +aoa
        # convention -> pitch_max>0 = nose-up.
        _, pmax, t1s, t2s, a_sm, tconv = ramp
        that = t / max(tconv, 1e-12)                                   # convective time t* = U t / c
        lnc = lambda z: np.logaddexp(z, -z) - np.log(2.0)             # stable ln(cosh(z))
        fval = lnc(a_sm * (that - t1s)) - lnc(a_sm * (that - t2s))
        D = a_sm * (t2s - t1s)
        frac = np.clip((fval + D) / (2.0 * D + 1e-12), 0.0, 1.0)       # 0 (before t1) .. 1 (after t2, held)
        x = C0[..., 0]; y = C0[..., 1]; z0 = C0[..., 2]
        # uniform pitch ramp + optional GEOMETRIC spanwise TWIST (Hirato case 2: A_t deg higher incidence at the
        # tip than the root, constant). Twist adds tip-ward nose-up incidence (same sign as the ramp pitch) so the
        # LESP peaks OUTBOARD -> tip-first LEV onset (vs root-first for the untwisted case 1). yl = wing-local span.
        yl = y - root_off
        psi = -pmax * frac - A_t * (yl / max(span, 1e-9))            # A_t=radians(twist_amp_deg); 0 -> pure uniform pitch
        cp, sp = np.cos(psi), np.sin(psi)
        xr = x_ea + (x - x_ea) * cp - z0 * sp
        zr = (x - x_ea) * sp + z0 * cp
        return np.stack([xr, y, zr], axis=-1)                       # no flap
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


def twisted_state(C0, t, A_f, A_t, Om, phi, x_ea, span, dlt=1e-6, swept_axis=False, root_off=0.0, ramp=None):
    corners = twisted_corners(C0, t, A_f, A_t, Om, phi, x_ea, span, swept_axis, root_off, ramp)
    cp = twisted_corners(C0, t + dlt, A_f, A_t, Om, phi, x_ea, span, swept_axis, root_off, ramp)
    cm = twisted_corners(C0, t - dlt, A_f, A_t, Om, phi, x_ea, span, swept_axis, root_off, ramp)
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


# ==== (E) PER-RING vortex-core induction: give the LEV rings a SMALL core (-> tight roll-up + strong held-lift
# induction) while keeping TEV at the standard WAKE_CORE. ring_vel_core(...,delta) is the regularized Biot-Savart
# (van Garrel). The held-LEV must ROLL UP into a coherent vortex to induce the held lift (Hirato); a large/uniform
# core smears it (flat sheet, under-lift). Per-ring core wcore[m] lets LEV roll up without TEV near-singular noise.
@wp.kernel
def _convect_wcore(rings: wp.array(dtype=V3, ndim=2), gamma: wp.array(dtype=DTYPE, ndim=2), npan: int,
                   wr: wp.array(dtype=V3, ndim=2), wg: wp.array(dtype=DTYPE), wcore: wp.array(dtype=DTYPE),
                   nw: int, bcore: DTYPE, Vinf: V3, dt: DTYPE, wr_new: wp.array(dtype=V3, ndim=2)):
    k, c = wp.tid(); P = wr[k, c]; v = Vinf
    for p in range(npan):
        v = v + gamma[0, p] * ring_vel_core(P, rings[p, 0], rings[p, 1], rings[p, 2], rings[p, 3], bcore)
    for m in range(nw):
        v = v + wg[m] * ring_vel_core(P, wr[m, 0], wr[m, 1], wr[m, 2], wr[m, 3], wcore[m])   # LEV small core -> roll-up
    wr_new[k, c] = P + v * dt


@wp.kernel
def _rhs_moving_wcore(col: wp.array(dtype=V3), nrm: wp.array(dtype=V3), Vinf: V3, vcol: wp.array(dtype=V3),
                     wr: wp.array(dtype=V3, ndim=2), wg: wp.array(dtype=DTYPE), wcore: wp.array(dtype=DTYPE),
                     nw: int, rhs: wp.array(dtype=DTYPE, ndim=2)):
    """Moving-body BC with PER-RING core: the wake (incl. LEV) induction on the bound collocations uses the
    regularized ring_vel_core(...,wcore[k]) instead of the singular ring_vel. This keeps the near-singular LEV
    feedback OUT OF THE SOLVE -> stops the fine-grid blow-up at its source (the dp clamp only protected the force)."""
    i = wp.tid(); ci = col[i]; ni = nrm[i]
    s = -wp.dot(Vinf - vcol[i], ni)
    for k in range(nw):
        s = s - wg[k] * wp.dot(ring_vel_core(ci, wr[k, 0], wr[k, 1], wr[k, 2], wr[k, 3], wcore[k]), ni)
    rhs[0, i] = s


@wp.kernel
def _col_wake_wcore(col: wp.array(dtype=V3), wr: wp.array(dtype=V3, ndim=2), wg: wp.array(dtype=DTYPE),
                    wcore: wp.array(dtype=DTYPE), nw: int, Vw: wp.array(dtype=V3)):
    """Wake-induced velocity at collocations using PER-RING core (LEV small -> stronger, regularized held-lift
    induction; replaces the singular col_wake_vel_kernel that spiked). This is how the rolled-up LEV's lift
    enters the unsteady-Bernoulli surface force."""
    i = wp.tid(); ci = col[i]
    vx = wp.float64(0.0); vy = wp.float64(0.0); vz = wp.float64(0.0)
    for k in range(nw):
        vv = wg[k] * ring_vel_core(ci, wr[k, 0], wr[k, 1], wr[k, 2], wr[k, 3], wcore[k])
        vx = vx + vv[0]; vy = vy + vv[1]; vz = vz + vv[2]
    Vw[i] = V3(vx, vy, vz)


# ================= fp32 FAST WAKE-INDUCTION PATH (2026-07-09 perf) =================
# GeForce fp64 throughput is 1/64 of fp32 (4090: 1.3 vs 83 TFLOPS) and the three N²
# wake kernels dominate the step cost (see docs/diag/perf_uvlm.md). These twins keep
# ALL state fp64 (positions/strengths/outputs accumulate in fp64) and cast per element
# inside the kernel: only the induced-velocity ARITHMETIC runs fp32. Force error vs
# the fp64 path must sit inside the ±0.15N run-to-run band (validated before default-on).
V3F = wp.vec3f


@wp.func
def _f3(v: V3) -> V3F:
    return V3F(wp.float32(v[0]), wp.float32(v[1]), wp.float32(v[2]))


@wp.func
def _vseg_f(P: V3F, A: V3F, B: V3F) -> V3F:
    r1 = P - A; r2 = P - B; r0 = B - A
    cr = wp.cross(r1, r2)
    cr2 = wp.dot(cr, cr) + wp.float32(1.0e-10)
    n1 = wp.sqrt(wp.dot(r1, r1) + wp.float32(1.0e-12))
    n2 = wp.sqrt(wp.dot(r2, r2) + wp.float32(1.0e-12))
    return wp.float32(0.07957747154594767) * wp.dot(r0, r1 / n1 - r2 / n2) / cr2 * cr


@wp.func
def _ring_vel_f(P: V3F, c0: V3F, c1: V3F, c2: V3F, c3: V3F) -> V3F:
    return _vseg_f(P, c0, c1) + _vseg_f(P, c1, c2) + _vseg_f(P, c2, c3) + _vseg_f(P, c3, c0)


@wp.func
def _vseg_core_f(P: V3F, A: V3F, B: V3F, delta: wp.float32) -> V3F:
    r1 = P - A; r2 = P - B; r0 = B - A
    cr = wp.cross(r1, r2)
    cr2 = wp.dot(cr, cr) + delta * delta * wp.dot(r0, r0) + wp.float32(1.0e-18)
    n1 = wp.sqrt(wp.dot(r1, r1) + wp.float32(1.0e-12))
    n2 = wp.sqrt(wp.dot(r2, r2) + wp.float32(1.0e-12))
    return wp.float32(0.07957747154594767) * wp.dot(r0, r1 / n1 - r2 / n2) / cr2 * cr


@wp.func
def _ring_vel_core_f(P: V3F, c0: V3F, c1: V3F, c2: V3F, c3: V3F, delta: wp.float32) -> V3F:
    return _vseg_core_f(P, c0, c1, delta) + _vseg_core_f(P, c1, c2, delta) \
        + _vseg_core_f(P, c2, c3, delta) + _vseg_core_f(P, c3, c0, delta)


@wp.kernel
def _convect_wcore_f32(rings: wp.array(dtype=V3, ndim=2), gamma: wp.array(dtype=DTYPE, ndim=2), npan: int,
                       wr: wp.array(dtype=V3, ndim=2), wg: wp.array(dtype=DTYPE), wcore: wp.array(dtype=DTYPE),
                       nw: int, bcore: DTYPE, Vinf: V3, dt: DTYPE, wr_new: wp.array(dtype=V3, ndim=2)):
    k, c = wp.tid(); Pd = wr[k, c]; P = _f3(Pd)
    bc = wp.float32(bcore)
    v = _f3(Vinf)
    for p in range(npan):
        v = v + wp.float32(gamma[0, p]) * _ring_vel_core_f(
            P, _f3(rings[p, 0]), _f3(rings[p, 1]), _f3(rings[p, 2]), _f3(rings[p, 3]), bc)
    for m in range(nw):
        v = v + wp.float32(wg[m]) * _ring_vel_core_f(
            P, _f3(wr[m, 0]), _f3(wr[m, 1]), _f3(wr[m, 2]), _f3(wr[m, 3]), wp.float32(wcore[m]))
    wr_new[k, c] = Pd + V3(wp.float64(v[0]), wp.float64(v[1]), wp.float64(v[2])) * dt


@wp.kernel
def _convect_f32(rings: wp.array(dtype=V3, ndim=2), gamma: wp.array(dtype=DTYPE, ndim=2), npan: int,
                 wr: wp.array(dtype=V3, ndim=2), wg: wp.array(dtype=DTYPE), nw: int,
                 Vinf: V3, dt: DTYPE, bcore: DTYPE, wr_new: wp.array(dtype=V3, ndim=2)):
    k, c = wp.tid(); Pd = wr[k, c]; P = _f3(Pd)
    dl = wp.float32(bcore)
    v = _f3(Vinf)
    for p in range(npan):
        v = v + wp.float32(gamma[0, p]) * _ring_vel_core_f(
            P, _f3(rings[p, 0]), _f3(rings[p, 1]), _f3(rings[p, 2]), _f3(rings[p, 3]), dl)
    for m in range(nw):
        v = v + wp.float32(wg[m]) * _ring_vel_core_f(
            P, _f3(wr[m, 0]), _f3(wr[m, 1]), _f3(wr[m, 2]), _f3(wr[m, 3]), dl)
    wr_new[k, c] = Pd + V3(wp.float64(v[0]), wp.float64(v[1]), wp.float64(v[2])) * dt


@wp.kernel
def _rhs_moving_f32(col: wp.array(dtype=V3), nrm: wp.array(dtype=V3), Vinf: V3, vcol: wp.array(dtype=V3),
                    wr: wp.array(dtype=V3, ndim=2), wg: wp.array(dtype=DTYPE), nw: int,
                    rhs: wp.array(dtype=DTYPE, ndim=2)):
    i = wp.tid(); ci = _f3(col[i]); ni = _f3(nrm[i])
    s = wp.float32(0.0)
    for k in range(nw):
        s = s - wp.float32(wg[k]) * wp.dot(_ring_vel_f(
            ci, _f3(wr[k, 0]), _f3(wr[k, 1]), _f3(wr[k, 2]), _f3(wr[k, 3])), ni)
    rhs[0, i] = -wp.dot(Vinf - vcol[i], nrm[i]) + wp.float64(s)


@wp.kernel
def _col_wake_f32(col: wp.array(dtype=V3), wr: wp.array(dtype=V3, ndim=2), wg: wp.array(dtype=DTYPE),
                  nw: int, Vw: wp.array(dtype=V3)):
    i = wp.tid(); ci = _f3(col[i])
    v = V3F(wp.float32(0.0), wp.float32(0.0), wp.float32(0.0))
    for k in range(nw):
        v = v + wp.float32(wg[k]) * _ring_vel_f(
            ci, _f3(wr[k, 0]), _f3(wr[k, 1]), _f3(wr[k, 2]), _f3(wr[k, 3]))
    Vw[i] = V3(wp.float64(v[0]), wp.float64(v[1]), wp.float64(v[2]))


# CHUNKED variants: dim=npan(=192) kernels are LATENCY-bound (6 warps can't hide the
# 11.5k-source serial loop; measured 142 ms each). Split the source loop over _NCH
# chunks -> npan*_NCH threads, fp32 partials atomically added into the fp64 output.
_NCH = 64


@wp.kernel
def _rhs_base_k(col: wp.array(dtype=V3), nrm: wp.array(dtype=V3), Vinf: V3, vcol: wp.array(dtype=V3),
                rhs: wp.array(dtype=DTYPE, ndim=2)):
    i = wp.tid()
    rhs[0, i] = -wp.dot(Vinf - vcol[i], nrm[i])


@wp.kernel
def _rhs_wake_chunk_wcore_f32(col: wp.array(dtype=V3), nrm: wp.array(dtype=V3),
                              wr: wp.array(dtype=V3, ndim=2), wg: wp.array(dtype=DTYPE),
                              wcore: wp.array(dtype=DTYPE), nw: int, nch: int,
                              rhs: wp.array(dtype=DTYPE, ndim=2)):
    i, c = wp.tid()
    ci = _f3(col[i]); ni = _f3(nrm[i])
    s = wp.float32(0.0)
    k = c
    while k < nw:
        s = s - wp.float32(wg[k]) * wp.dot(_ring_vel_core_f(
            ci, _f3(wr[k, 0]), _f3(wr[k, 1]), _f3(wr[k, 2]), _f3(wr[k, 3]), wp.float32(wcore[k])), ni)
        k = k + nch
    wp.atomic_add(rhs, 0, i, wp.float64(s))


@wp.kernel
def _col_wake_chunk_wcore_f32(col: wp.array(dtype=V3), wr: wp.array(dtype=V3, ndim=2),
                              wg: wp.array(dtype=DTYPE), wcore: wp.array(dtype=DTYPE),
                              nw: int, nch: int, Vw: wp.array(dtype=V3)):
    i, c = wp.tid()
    ci = _f3(col[i])
    v = V3F(wp.float32(0.0), wp.float32(0.0), wp.float32(0.0))
    k = c
    while k < nw:
        v = v + wp.float32(wg[k]) * _ring_vel_core_f(
            ci, _f3(wr[k, 0]), _f3(wr[k, 1]), _f3(wr[k, 2]), _f3(wr[k, 3]), wp.float32(wcore[k]))
        k = k + nch
    wp.atomic_add(Vw, i, V3(wp.float64(v[0]), wp.float64(v[1]), wp.float64(v[2])))
# ================= end fp32 fast path =================


@wp.kernel
def _shed_lev_traj(lel: wp.array(dtype=V3), ler: wp.array(dtype=V3),
                   lpl: wp.array(dtype=V3), lpr: wp.array(dtype=V3), lev_str: wp.array(dtype=DTYPE),
                   Vinf: V3, dt: DTYPE, nw: int, first: int, wr: wp.array(dtype=V3, ndim=2),
                   wg: wp.array(dtype=DTYPE), lcl: wp.array(dtype=V3), lcr: wp.array(dtype=V3)):
    """CONNECTED leading-edge vortex SHEET (mirror of _shed_te_traj, from the LE): the new LEV ring's leading
    edge attaches at the CURRENT geometric LE (lel/ler, offset onto the suction side), its trailing edge
    connects to the PREVIOUS step's LE-shed corners (convected) -> a CONTINUOUS sheet trailing from the LE over
    the suction surface, free to ROLL UP (self-induction) into a coherent LEV. Strength = LESP-excess circulation
    (same sign as the bound -> the rolled-up LEV ADDS lift, as in all flapping-wing DVM)."""
    j = wp.tid(); idx = nw + j
    cl = lel[j]; cr = ler[j]                          # current LE corners (leading edge of the new sheet ring)
    wr[idx, 0] = cl; wr[idx, 1] = cr
    if first == 1:
        wr[idx, 2] = cr + Vinf * dt; wr[idx, 3] = cl + Vinf * dt
    else:
        wr[idx, 2] = lpr[j] + Vinf * dt; wr[idx, 3] = lpl[j] + Vinf * dt
    wg[idx] = lev_str[j]
    lcl[j] = cl; lcr[j] = cr                          # save current LE corners for next step's "previous"


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


@wp.kernel
def _lev_inf_kernel(col: wp.array(dtype=V3), nrm: wp.array(dtype=V3),
                    lwr: wp.array(dtype=V3, ndim=2), core: DTYPE, nlev: int,
                    INF: wp.array(dtype=DTYPE, ndim=2)):
    """(HIRATO Fig.6 implicit LESP constraint) influence matrix: INF[i,j] = (UNIT-strength nascent LEV ring j
    induced velocity at bound collocation i) . n_i. Because the wing-bound circulation is linear in the RHS and
    the LESP (=A0) is a linear functional of the bound circulation, LESP is AFFINE in the shed LEV strengths.
    This matrix lets us solve — in ONE step — for the LEV ring strengths that drive each supercritical strip's
    LESP exactly back to LESP_crit (Hirato Eq.8/9 Kelvin + the LESP=LESP_crit constraint), which the paper does
    by fixed-point iteration (Fig.6). ring_vel_core(...,core) is the regularized (Lamb-Oseen) Biot-Savart ring."""
    i, j = wp.tid()
    v = ring_vel_core(col[i], lwr[j, 0], lwr[j, 1], lwr[j, 2], lwr[j, 3], core)
    INF[i, j] = wp.dot(v, nrm[i])


@wp.kernel
def _place_rings_kernel(src: wp.array(dtype=V3, ndim=2), sg: wp.array(dtype=DTYPE), off: int,
                        wr: wp.array(dtype=V3, ndim=2), wg: wp.array(dtype=DTYPE)):
    """Write ns freshly-built vortex rings (corners src[j,0..3], strength sg[j]) into the wake arrays at
    offset off. Used by the 'hirato' path to shed the Eq.7-placed LEV rings directly (full geometry control)."""
    j = wp.tid()
    wr[off + j, 0] = src[j, 0]; wr[off + j, 1] = src[j, 1]
    wr[off + j, 2] = src[j, 2]; wr[off + j, 3] = src[j, 3]
    wg[off + j] = sg[j]


def gpu_run_twist(nc=4, ns=10, chord=0.287, half_span=0.80, U=8.0, aoa_deg=5.0,
                  flap_amp_deg=45.0, twist_amp_deg=22.5, twist_phase_deg=90.0,   # +90: twist LEADS flap 90deg
                  # (paper double-crank: psi~cos(wt), nose-down on downstroke = washout of the deep-stall AoA)
                  freq=2.0, n_cycle=5, steps_per_cycle=40, wake_rows=50, rk2=False, te_traj=False,
                  swept_axis=False, real_geom=False, real_lev=False, lev_sat=False, lev_merge=False, lev_tau=0.20,
                  lev_detach_deg=90.0,
                  lesp_crit_deg=15.0, lev_klev=1.0,
                  visc=False, tc_thick=0.06, prof_drag=False, cd_form=1.98, cd_sat_deg=30.0, cd_dp=1.2, d_para=0.0, les_suction=False, les_eta=1.0,
                  les_kin=False,           # (C4 2026-07-04) LE-suction/vnf velocity scale = the KINEMATIC LE velocity (Vinf - vbody),
                  #   i.e. the SAME velocity that normalizes A0 (Urel_le / constraint kA0). WHY (verified vs Hirato/Ramesh sources):
                  #   F_S = pi*rho*c*U^2*A0^2 with A0 = 1.13*Gamma_1/(U*c*(th+sin th)) -> U CANCELS (F_S ~ Gamma_1^2/c): the velocity
                  #   in the suction formula is the DENORMALIZATION of A0, not a local dynamic pressure -> it must equal the A0-
                  #   normalization velocity or the identity breaks. All induction (LEV sheet + TEV wake) enters ONLY through A0
                  #   (Hirato Eq.10 solve); on shed strips A0 is PINNED at a0_crit, so letting Vcol's LEV-sheet induction also cut
                  #   Vle^2 counts the same suppression twice (P1: les +10.4 vs kelvin-path +19.4 at 10/5/2.6/tw45, same cap).
                  #   Default (False) keeps the legacy Vcol path. No new constants.
                  les_pre=False,           # (C7 2026-07-04) 'hirato' realized LE suction keyed on the PRE-constraint A0pre
                  #   (crit-capped), consistent with the vnf excess + faure gate; fixes the subcritical post-solve A0
                  #   collapse under sustained shedding (discrete-ring over-suppression artifact). See S4 block.
                  les_sep='plateau',       # (2026-07-09 GAP-f2) chordwise LE suction on SEPARATED (LESP-supercritical)
                  les_free=False,          # (T1 DIAG) uncapped-A0 suction counterfactual (diagnostic only)
                  fn_Tstar=4.0,            # (R2) LEV optimal formation number T* gating suction collapse (Gharib
                  #   lineage; flapping-wing LEV pinch-off: Onoue & Breuer 2016 3.7±0.3 -> 4.0; literature, not fit)
                  #   strips: 'plateau' = held at the crit value (LDVM legacy, Katz-1981 postulate); 'zero' = collapses
                  #   to 0 (Narsipur et al. 2020 JFM 900 A25: viscous LESP -> near-zero at LE separation, Re 1e4-1e5);
                  #   'polhamus' = magnitude conserved but ROTATED to the panel normal (vortex force, NASA TN D-3767)
                  #   -> chordwise thrust still 0 while separated, lift keeps the separated-flow contribution.
                  #   The held plateau rides the local dynamic pressure ~f^2 = fictitious flapping thrust
                  #   (docs/diag/gap_thrust_f2.md: les carries 95% of the model's f^2 thrust slope).
                  fp_lev=False, lev_kv=4.62, lev_trans_deg=15.0,
                  # --- 2026-06-27 first-principles LESP-LEV: orthogonal MODE switches (candidate-model matrix) ---
                  lev_shed_mode='none',    # 'none'|'kelvin'(explicit excess)|'varA0'(Modulation Eq.11-12)|'kinematic'(legacy)|'hirato'(FAITHFUL Fig.6 implicit LESP=LESP_crit constraint solve)
                  lev_iter=1,              # ('hirato') extra correction sweeps for spanwise LESP coupling (1 = single affine solve; the LESP=crit constraint is affine so 1 is near-exact, 2-3 tightens coupling)
                  lev_pseudo=True,         # ('hirato') include the Hirato pseudovortex ring (Fig.5): cancels the geometric-LE filament + makes the LEV shed EFFECTIVE at reducing LESP (physical strength)
                  lev_cap_exc=False,       # (H11) cap the shed ring strength at EXACTLY the Kelvin excess (physical circulation the
                  #   bound gives up; no factor). Fixes the over-strong-ring wake pollution (P0 runaway / A0 collapse / tw45 fiction);
                  #   the constraint then under-delivers like the kelvin path -> the crit-clip closures carry the semantics.
                  lev_hold_mode='inviscid',# 'inviscid'(convect freely)|'hold'(viscous τ_hold)|'hold_detach'(Li 4-phase cutoff)
                  a0_crit=0.25,            # critical LESP (airfoil/Re property; anchor via 2D flap_ldvm). 0.12 thin@Re10k .. 0.27 SD7003@Re20k
                  a0_mode='xref',          # LESP A0 extraction: 'xref'(Hirato@x_ref=0.10c, nc-robust) | 'sqrtx'(√x limit from the RESOLVED 1st LE panel -> 3D, needs fine cosine LE)
                  tau_hold_scale=1.0,      # ×c/(0.4U) viscous-hold timescale
                  lev_roll_core=0.01,      # FLOOR LEV vortex-core (chord frac); the actual core is resolution-adaptive (below)
                  lev_force_core=0.0,      # (FORCE) LEV force-core chord frac (0 -> = lev_roll_core). DECOUPLES the force core
                  #   from the convection core: a SMALL convect core lets the LEV roll up (Fig.11), a LARGER force core
                  #   keeps the near-surface rolled-up LEV from spiking the Bernoulli force (Hirato Lamb-Oseen Eq.25 ~0.5*ring)
                  lev_vnf=True,            # (HIRATO force closure) recover the CAPPED LE suction (A0>a0_crit excess) as a
                  #   VORTEX NORMAL force. Garrick/Polhamus: the LE suction realizable only to A0_crit; the unrealizable
                  #   excess pi*rho*c*U^2*(A0^2-a0_crit^2) rotates 90deg onto the panel normal = the LEV vortex lift the
                  #   shed sheet delivers (the discrete-ring induction is numerically too weak to deliver it directly).
                  lev_core_ring=0.0,       # (HIRATO Eq.25) per-ring Lamb-Oseen core = lev_core_ring × LOCAL ring size (0 -> off,
                  #   use fixed chord-frac cores). Scales the core WITH each ring so the compressed near-LE rings keep a
                  #   STRONG bounded induction (recovers Polhamus vortex lift) while large/far rings stay smooth. r_c<0.5*ring.
                  lev_overlap=1.0,         # (STAB) LEV core = overlap × shed-spacing (∝U·dt & strip width) -> shrinks as grid refines, never near-singular
                  lev_consistent=True,     # apply the adaptive core in solve+force too (not just convect) -> grid-CONVERGENT LEV (vs singular drift/blow-up)
                  tev_core=0.0,            # (案升力分辨率子案 2026-07-12) TEV near-wake Lamb-Oseen core in SOLVE+FORCE,
                  #   = tev_core x local ring mean-edge (resolution-adaptive, SAME rule as lev_core_ring). Under
                  #   flapping the near-wake rings carry uncancelled dG/dt circulation; sampling their singular
                  #   field at collocations diverges the bound solve with nc (bern nc8->16: -5.6N; static immune,
                  #   ddG=0 cancels). The FRESHEST tev_fresh rows keep the near-singular treatment: their TE-shared
                  #   filament must cancel against the SINGULAR bound lattice or the Kutta condition softens
                  #   (archived -30% C_N failure). 0 = off (bit-identical legacy path).
                  tev_fresh=1,             #   number of freshest shed rows kept near-singular (Kutta zone)
                  lev_sub=1,               # (FINE) spanwise sub-rings of LEV per strip (lev_sub=5 -> 5× finer LEV sheet, independent of wing grid)
                  wake_f32=True,           # (2026-07-09 perf) N² wake-induction kernels in fp32 ARITHMETIC (state
                  #   stays fp64; positions update in fp64). GeForce fp64 = 1/64 fp32 throughput -> the dominant
                  #   step cost drops ~10-30x. False = legacy fp64-exact path (adjoint/repro). ±0.15N band gated.
                  lev_sheet=True,          # (E2) shed LEV as a CONNECTED trailing sheet from the LE (rolls up) instead of fixed-offset rings
                  lev_place='ansari',      # 'ansari' = Hirato Eq.7 placement, LEV sheet OVER the suction surface anchored at the LE; 'wake' = old (trails off the back, wrong)
                  lev_rollh=0.5,           # LEV roll-up height as it convects aft (chord frac) — the sheet lifts off the suction surface (Hirato Fig.11 spiral)
                  lev_fmax=1.4,            # drop LEV rings once they convect past this chord fraction (detach off the TE)
                  lev_sign=1.0,            # LEV circulation sign vs bound (+1 = same sign -> adds lift; test both)
                  lev_le_off=0.0,          # LEV sheet ORIGIN = the geometric leading-edge POINT (physically correct: the shear layer separates at the sharp LE, then rolls up above the surface). Stability comes from the convect core, not an offset.
                  attached_drag='none',    # 'none'|'faure'(static C_D(α_rel))|'legacy'(old visc/prof_drag)
                  vnf_sat=True,            # (A/B) Polhamus/flat-plate saturation of the vnf excess (min branch); False = pure pi branch
                  faure_gate_pre=True,     # (A/B) faure attached gate uses PRE-constraint A0pre strict < ; False = legacy post-A0 <=
                  # --- 2026-06-30 additive empirical-residual corrections (default OFF; physics-anchored) ---
                  geo_stall=False,         # Fix1: quasi-steady GEOMETRIC-pitch static stall lift loss (twist-driven, freq-independent)
                  geo_stall_deg=12.0,      # static stall angle alpha_ss (NACA-2406 @ Re~1e5; airfoil property)
                  geo_stall_width=16.0,    # separation-spread angle: alpha past stall over which TE separation goes full (airfoil property)
                  geo_stall_peak=False,    # False=instantaneous psi(t) each step; True=cycle-peak |psi| amplitude
                  geo_stall_vec=False,     # (C8) Kirchhoff factor scales the strip Bernoulli force VECTOR (pressure-differential
                  #   collapse, both lift AND its backward tilt at deep twist) instead of the legacy +z-only removal
                  kirch_tw=False,          # (2026-07-10 案A 变体B, research_bern_twist.md M1) LB/Kirchhoff attenuation
                  #   applied ONLY to the TWIST-induced share of the circulatory pressure (first-order incidence
                  #   split: dp_c' = dp_c*(1-(1-fac)*psi_t/alpha_eff), fac=((1+sqrt(fsep))/2)^2 from the LAGGED
                  #   flow-incidence separation state at 3c/4). tw0 forces untouched BY CONSTRUCTION (psi_t=0);
                  #   the full-force variant (kirch_cn) failed the tw0/lift regression gates (measured: lift 5.9->2.9).
                  #   Constants unchanged: alpha_ss/width (NACA-2406) + fsep_tau=4.5 (GK literature).
                  kirch_cn=False,          # (H10) alpha_eff-Kirchhoff CN factor on the CIRCULATORY Bernoulli pressure, vectorial,
                  kirch_blend=False,       # (P1 2026-07-05) DOUBLE-COUNT FIX: blend attached Bernoulli + flat-plate CN by the
                  #   (lagged) separated fraction (1-fsep) into ONE consistent force vector (replace, not add). The flat-plate
                  #   CN's backward tilt yields pressure drag from the SAME vector that loses lift -> no double-count with the
                  #   bound's backward-tilt component. prof_drag is gated OFF when this is on (the flat-plate CN replaces it).
                  #   per-strip from the kinematic LE incidence; added-mass untouched. Constants = geo_stall_deg/width (NACA-2406).
                  les_att=False,           # (H10) realized LE suction x fsep(alpha_eff): dies at full separation (rotates into vnf)
                  fsep_lag=False,          # (H13) Goman-Khrabrov 1st-order lag on the fsep separation state: tau = fsep_tau*c/(2 U_rel).
                  lev_impulse=False,       # (H16 Li/Feng) LEV force from the VORTEX IMPULSE derivative F = -d/dt[rho*Sum(Gamma*A)],
                  #   grid-INDEPENDENT (no surface-pressure resolution). Replaces the Vlev_a->Vcol Bernoulli contribution (which
                  #   under-resolves the LEV at coarse nc) -> the LEV lift emerges without the unstable LESP-constraint fold.
                  fsep_tau=4.5,            #   tau_star, LITERATURE airfoil dynamic-stall constant (GK-family ~3-9; NOT a RoboEagle fit)
                  gk_stall=False,          # (病灶#1 2026-07-19) FULL Goman-Khrabrov dynamic-stall delay on geo_stall: the missing
                  #   tau2*psidot term (static-stall-angle SHIFT) that makes lift GROW with frequency. v4's geo_stall_vec gates the
                  #   Kirchhoff separation by the GEOMETRIC pitch (frequency-INDEPENDENT by design) -> zero dL/df. gk_stall shifts the
                  #   stall input by tau2*psi_dot (psi_dot = A_t*yfrac*Om*cos, ANALYTIC, ~f) + tau1 relaxation. Ayancik-Mulleners 2022
                  #   JFM 942:A38 constants (tau1=4.24 c/U; tau2=[4.24+0.0815*r^-7/9] c/U, r=reduced pitch rate). Zero new fitted consts.
                  gk_tau1=4.24, gk_c2=0.0815, gk_rfloor=0.02,   # Ayancik-Mulleners generalized-GK power-law (r_floor guards r->0 blowup)
                  lb_closure=False,        # (L-B 2026-07-20) Leishman-Beddoes dynamic-stall closure replacing
                  #   geo_stall's static Kirchhoff with TIME-LAGGED f2 (Tp/Tf) + LEV vortex lift CNv (Tv/Tvl) +
                  #   tangential-suction sqrt(f2) decay. Per-strip LBDynStrip states (lb_dyn.py). Zero-fit
                  #   (research_lb_formula.md): Tp/Tf/Tv/Tvl NACA0012 lit defaults; f_qs from S0 polar inversion;
                  #   eta=0.95; LESP_crit existing. Replaces gk_stall/geo_stall_vec when on. Default OFF = v4 bit-exact.
                  lb_lesp_crit=0.18, lb_eta=0.95, lb_Tp=1.7, lb_Tf=3.0, lb_Tv=6.0, lb_Tvl=6.0,
                  fric_drag=False,         # Fix2: flap-velocity^2 friction drag (turbulent flat-plate Cf; reuses visc structure)
                  cf_mode='turbulent',     # 'turbulent'(0.074/Re^0.2) | 'laminar'(1.328/sqrt Re)
                  drag_polar=False, cd0_polar=0.018, oswald=0.85,
                  d0_drag=0.0,
                  part_lev=False, lev_cons=False, lev_core=0.10, lev_sig0=0.5, lev_owin=2.0,
                  part_mode='kinematic',   # 'kinematic' legacy shed | 'rvpm' (S3b): A0-pin closed-form strength
                  #   (needs a0_mode='downwash'), 1/3 placement, sigma=1.3x local shed spacing, frozen circulation
                  part_transport='euler',  # 'euler' legacy advect kernel | 'rvpm' (S3a): LSRK3 + transposed
                  #   stretching + sigma evolution + corrected-Pedrizzetti in the full field (rvpm3d.py)
                  sym=False, root_off=0.0, stall=False, stall_deg=12.0,
                  vortex=False, k_vortex=2.0, dstall=False, ds_crit_deg=14.0, ds_tv=0.40, ds_k=1.0,
                  ds_delay=18, frames_out=None, frame_skip=3,
                  cosine_chord=True,       # (网格无关性研究 2026-07-12) chordwise spacing law passthrough. Cosine
                  #   clusters LE/TE quadratically: TE panel ~c·(pi/2nc)^2 -> the TE-row collocation collapses
                  #   onto the near-singular fresh-wake rows quadratically with nc (lift-vs-nc divergence
                  #   amplifier under flapping). uniform = c/nc (linear).
                  sep_drag=False,          # (u-trend ②) separated-strip flat-plate drag CD90_AR*sin^2 in the faure/uiuc slot
                  vnf_kelvin=False,        # (L2) vnf excess normal force on the KELVIN path (DSV cycle-mean lift;
                  #   Polhamus-saturated, zero new constants); disables the impulse force accumulation (single
                  #   accounting). See gap_l2_liftgrowth.md.
                  cp_cap=8.0,              # per-panel |Cp| clamp vs q_ref (near-field artifact guard). DIAG: the
                  #   bound-sheet LE pressure peak is PHYSICAL and grows ~1/sqrt(dx_LE) under chordwise refinement;
                  #   a fixed cap clips it progressively harder as nc grows (lift-vs-nc divergence suspect, 案升力
                  #   分辨率子案 2026-07-12). Parametrized for the convergence audit; default unchanged.
                  pitch_ramp=False, pitch_max=45.0, pitch_K=0.3, pitch_t0star=1.0,   # HIRATO pitch-ramp validation (Fig.9/11)
                  deform_hook=None):   # (P2-S6) flexible-wing REPLAY: callable t -> (du, dv) added to the
                  #   rigid-kinematic corners/velocities, shapes (nc+1, ns+1, 3). The S5 coupled solve
                  #   produces the deformation field; THIS validated closure stack evaluates the forces
                  #   on the deformed motion (one-way replay, S6 v1). None -> bit-identical rigid path.
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
    C0 = (rg.robowing_real(nc, ns, half_span, root_off=root_off, cosine_chord=cosine_chord) if real_geom
          else ffv.flat_wing(nc, ns, chord, half_span)); npan = nc * ns; ncv = (nc + 1) * (ns + 1)
    A_f = np.radians(flap_amp_deg); A_t = np.radians(twist_amp_deg); phi = np.radians(twist_phase_deg)
    Om = 2.0 * np.pi * freq; x_ea = 0.25 * chord
    # HIRATO pitch-ramp (Fig.9): CANONICAL ELDREDGE/GRANLUND smoothed ramp-hold (Granlund et al. 2013, ref [30]),
    # the same profile the paper uses — NOT a smootherstep. alpha(t*) ramps 0->pitch_max at the reduced pitch rate
    # K = adot*c/(2U): a nearly-CONSTANT pitch rate (2K in convective time) with smoothed corners, then held. This
    # gives the paper's C_L history (apparent-mass spike at ramp start, lift rising with alpha to a plateau) instead
    # of the mid-ramp hump a smootherstep produces. In convective time t*=U t/c: alpha = pmax*(f+D)/(2D) with
    # f = lncosh(a(t*-t1)) - lncosh(a(t*-t2)), D = a(t2-t1), t2 = t1 + pmax/(2K) (so the linear-region slope = 2K),
    # a = corner-smoothing rate. ramp=None -> normal flap+twist.
    ramp = None
    if pitch_ramp:
        pmax_r = np.radians(pitch_max); tconv = chord / max(U, 1e-6)
        t2star = pitch_t0star + pmax_r / (2.0 * max(pitch_K, 1e-6))   # ramp-end convective time (slope=2K over t1..t2)
        ramp = ('eldredge', pmax_r, pitch_t0star, t2star, 6.0, tconv)  # (tag, pmax, t1*, t2*, smoothing a, conv-time)
    Vinf = np.array([U, 0.0, U * np.tan(np.radians(aoa_deg))]); Vw = V3(*[float(v) for v in Vinf])
    T = 1.0 / freq; dt = T / steps_per_cycle; N = n_cycle * steps_per_cycle
    te = wp.array(np.array([(nc - 1) * ns + j for j in range(ns)], np.int32), dtype=wp.int32, device=dev)
    # NEW first-principles LESP-LEV sheds a ring/strip into the SAME wake (enters rhs + Bernoulli surface force);
    # reuses the real_lev plumbing. fp_shed True for any A0-based / kinematic LEV shed mode.
    fp_shed = lev_shed_mode in ('kelvin', 'varA0', 'kinematic', 'hirato')
    use_ansari = fp_shed and lev_sheet and lev_place == 'ansari'   # Hirato Eq.7: LEV is a SEPARATE sheet over the suction surface (NOT in the TEV wake)
    lev_in_wake = (real_lev or fp_shed) and not use_ansari         # a LEV ring goes into the TEV wake this run
    nsub = max(int(lev_sub), 1)                        # spanwise sub-rings of LEV per strip (FINE LEV sheet)
    lev_count = 0 if use_ansari else ((ns * nsub) if (fp_shed and lev_sheet) else (ns if lev_in_wake else 0))
    shed_per = ns + lev_count                          # TEV (ns) + LEV-in-wake (lev_count) per step
    wake_max = wake_rows * shed_per; maxw = min(N * shed_per, wake_max) + shed_per
    wr = wp.zeros((maxw, 4), dtype=V3, device=dev); wr_new = wp.zeros((maxw, 4), dtype=V3, device=dev)
    wr_m2 = wp.zeros((maxw, 4), dtype=V3, device=dev) if rk2 else None   # RK2 second-Euler buffer
    tpl = wp.zeros(ns, dtype=V3, device=dev); tpr = wp.zeros(ns, dtype=V3, device=dev)   # prev TE corners
    tcl = wp.zeros(ns, dtype=V3, device=dev); tcr = wp.zeros(ns, dtype=V3, device=dev)   # cur TE corners
    wg = wp.zeros(maxw, dtype=DTYPE, device=dev); gprev = wp.zeros((1, npan), dtype=DTYPE, device=dev); nw = 0
    # --- LEV vortex-particle field (parallel to the TEV ring wake; vec3d vortex moment, NOT scalar) ---
    pmax = N * ns * 8 + ns; np_part = 0              # up to 8 spanwise sub-particles/strip/step (rvpm)
    pp = wp.zeros(pmax, dtype=V3, device=dev); pa = wp.zeros(pmax, dtype=V3, device=dev)
    ps = wp.zeros(pmax, dtype=DTYPE, device=dev); pp_new = wp.zeros(pmax, dtype=V3, device=dev)
    sa_prev_p = wp.zeros(ns, dtype=DTYPE, device=dev)   # previous |sin a_eff| per strip (LESP-rate shed gate)
    I_lev_prev = np.zeros(3); I_lev_have = False         # previous LE-referenced LEV impulse (for -dI/dt force)
    SIG0 = DTYPE(lev_sig0 * U * dt); PCORE = DTYPE(lev_core)   # LEV particle core (smaller -> stronger induction)
    sin_crit_p = DTYPE(np.sin(np.radians(lesp_crit_deg)))
    # (S3b rvpm feeding) numpy source-of-truth mirrors + per-strip shed-event trackers
    pp_np = np.zeros((pmax, 3)); pa_np = np.zeros((pmax, 3)); ps_np = np.zeros(pmax)
    lev_prev_idx = np.full(ns, -1, dtype=int); lev_prev_it = np.full(ns, -99, dtype=int)
    # --- coherent-core LEV (N-LEV merging, N=1 per strip): ONE smooth merged ring/strip (CPU state) ---
    lev_cen = np.zeros((ns, 3)); lev_gam = np.zeros(ns); lev_gam_raw = np.zeros(ns)
    lev_rw = wp.zeros((ns, 4), dtype=V3, device=dev); lev_gw = wp.zeros(ns, dtype=DTYPE, device=dev)
    Vlev = wp.zeros(npan, dtype=V3, device=dev)
    Vpart = wp.zeros(npan, dtype=V3, device=dev)     # LEV-particle induced velocity at collocations (rVPM force)
    Lh = np.zeros(N); Xh = np.zeros(N); Ph = np.zeros(N); Lkjh = np.zeros(N)
    Lh_imp = np.zeros(N); Xh_imp = np.zeros(N)        # unsteady-Bernoulli surface-pressure force (captures LEV)
    Lh_vimp = np.zeros(N); Xh_vimp = np.zeros(N)       # (H16) LEV VORTEX-IMPULSE force (Li/Feng): F = -d/dt[rho*Sum(Gamma*A)]
    Fxb_tot = np.zeros(N); Fzb_tot = np.zeros(N)      # TOTAL body-frame force per step (sum of ALL force vectors:
    #   Bernoulli + LE-suction + friction + form-drag + vortex) -> the clean body force to rotate into wind axes
    Lh_vis = np.zeros(N); Xh_vis = np.zeros(N)         # DeLaurier viscous friction drag (strip, Re-based Blasius)
    Lh_pd = np.zeros(N); Xh_pd = np.zeros(N)           # separated-flow form/pressure drag (high-alpha, viscous-origin)
    _UT_LOG = []                                        # (T1b) per-step phase diagnostic (UTREND_DBG)
    Lh_les = np.zeros(N); Xh_les = np.zeros(N)         # leading-edge suction thrust (Garrick/DeLaurier dTs)
    Lh_vtx = np.zeros(N); Xh_vtx = np.zeros(N)         # high-alpha vortex normal force (Polhamus, lift+drag)
    Lh_ds = np.zeros(N)                                 # dynamic-stall LEV lift (sustains the downstroke plateau)
    Lh_stall = np.zeros(N); Xh_stall = np.zeros(N)      # Fix1: geometric quasi-steady stall loss (z; x only in vec mode)
    Lh_fric = np.zeros(N); Xh_fric = np.zeros(N)        # Fix2: flap-velocity^2 friction drag
    # per-strip wing-local span fraction yfrac=(y-root_off)/half_span (for the geometric twist pitch psi(y,t)=A_t*yfrac*sin(Om t+phi))
    _C0r = C0.reshape(nc + 1, ns + 1, 3)
    _ystrip = 0.5 * (_C0r[0, :-1, 1] + _C0r[0, 1:, 1])                       # spanwise center y of each strip (ns,)
    yfrac = np.clip((np.abs(_ystrip) - root_off) / max(half_span, 1e-9), 0.0, 1.0)
    aoa_rad = np.radians(aoa_deg)
    Glev_ds = np.zeros(ns); aeff_ds_prev = np.zeros(ns)  # per-strip LEV circulation state + prev alpha_eff
    NU_AIR = 1.5e-5; FORM_FF = 1.0 + 2.0 * tc_thick + 60.0 * tc_thick ** 4   # air kin. visc; Hoerner form factor
    wtype = []                                        # CPU bookkeeping: 0=TEV, 1=LEV per wake ring (for viz)
    lev_born = []; lev_s0 = []                         # per-wake-ring: birth step (-1=TEV) and original LEV strength
    tau_hold = tau_hold_scale * chord / (0.4 * max(U, 1e-6))   # Li-JFM viscous-hold timescale (s); single airfoil/Re scale
    use_wcore = lev_in_wake and (lev_roll_core > 0.0 or lev_overlap > 0.0)   # per-ring core in solve+force+convect
    # RESOLUTION-ADAPTIVE LEV core = overlap × inter-vortex spacing (max of temporal U·dt and spanwise strip width
    # /lev_sub). Big enough to kill near-singular blow-ups (stability), shrinks as the grid refines -> CONVERGES.
    span_sp = half_span / max(ns * lev_sub, 1)
    lev_core_abs = max(lev_roll_core * chord, lev_overlap * max(U * dt, 0.5 * span_sp))   # SOLVE/convect: stabilizing
    lev_core_force = (lev_force_core if lev_force_core > 0.0 else lev_roll_core) * chord   # FORCE core (decoupled option)
    use_lev_sheet = fp_shed and lev_sheet                      # (E2) connected LEV sheet from the LE (rolls up)
    nls = ns * nsub                                            # number of LEV sub-rings shed per step (sheet)
    lpl = wp.zeros(nls, dtype=V3, device=dev); lpr = wp.zeros(nls, dtype=V3, device=dev)   # prev LE-shed corners
    lcl = wp.zeros(nls, dtype=V3, device=dev); lcr = wp.zeros(nls, dtype=V3, device=dev)   # cur LE-shed corners
    lev_first = 1                                               # 1 until the first LEV row is shed
    fsep_state = None                                           # (H13) Goman-Khrabrov lagged separation state (per strip)
    lb_strips = None                                            # (L-B) per-strip LBDynStrip dynamic-stall states
    gk_X = None                                                 # (病灶#1) full-GK dynamic-stall separation state (per strip)
    fn_state = None                                             # (R2) per-strip LEV vortex-formation-number accumulator
    fsep_state_cn = None                                        # (案A 变体B) lagged separation state at the 3c/4 row
    # (ANSARI / Hirato Eq.7) parametric LEV sheet OVER the suction surface: each ring stored by its strip index,
    # chordwise fraction f (0=LE, grows aft as it convects), and strength. Lifted off the surface by lev_rollh*f
    # (roll-up). Anchored at the LE, NOT convected into the TEV wake -> sits over the wing, induces on bound+force.
    lev_aj = np.zeros(0, np.int64); lev_af = np.zeros(0); lev_ag = np.zeros(0)
    lev_ids = np.zeros(0, np.int64); lev_id_next = 0        # per-ring identity (material impulse accounting)
    imp_prev = None                                         # (ids, Gamma*Avec) of the previous step's ledger
    Vlev_a = wp.zeros(npan, dtype=V3, device=dev)              # LEV-sheet induced velocity at collocations
    lev_frame_rings = np.zeros((0, 4, 3)); lev_frame_g = np.zeros(0)   # LEV-sheet geometry for viz frames
    use_tevcore = tev_core > 0.0
    for t in range(N):
        wcore_dev = None; wcore_force_dev = None; wcore_conv_dev = None
        if (use_wcore or use_tevcore) and nw > 0:               # per-ring core: LEV gets a core, TEV = standard
            islev_np = np.asarray(lev_born[:nw]) >= 0
            tev_base = max(0.5 * lev_roll_core * chord, 1e-4)   # legacy near-singular TEV solve/force treatment
            if use_tevcore:
                # (分辨率子案) aged TEV rows get a van-Garrel core. NOTE the kernels' delta is DIMENSIONLESS
                # (cr2 += delta^2*|r0|^2, i.e. effective core radius = delta x SEGMENT length): delta_ring =
                # tev_core x (streamwise spacing)/(span edge) puts the SPAN filaments' effective core at
                # tev_core x streamwise spacing — smoothing the between-filament sampling oscillation that
                # biases fine-grid collocations (the nc divergence) without touching the sheet's macroscopic
                # field. The freshest tev_fresh rows use delta=0 (EXACTLY singular): even a 'tiny' absolute
                # floor (1.4mm as dimensionless 0.0014 -> 0.07mm on a 50mm span edge) cuts the attached TE
                # filament's induction ~80% at the cosine-nc16 TE collocation (0.7mm) and collapses the
                # STATIC solve (validated failure: 16.92->4.84N A/A).
                wrn_t = wr.numpy()[:nw]
                stream_t = 0.5 * (np.linalg.norm(wrn_t[:, 2] - wrn_t[:, 1], axis=1)
                                  + np.linalg.norm(wrn_t[:, 0] - wrn_t[:, 3], axis=1))
                span_t = 0.5 * (np.linalg.norm(wrn_t[:, 1] - wrn_t[:, 0], axis=1)
                                + np.linalg.norm(wrn_t[:, 3] - wrn_t[:, 2], axis=1)) + 1e-12
                fresh_t = np.zeros(nw, bool); fresh_t[max(0, nw - tev_fresh * shed_per):] = True
                # fresh delta=1e-4 (5um effective on a 50mm edge): physically singular, numerically restores
                # the ~1e-10 cr2 guard the plain singular kernel carries (the cored kernel's guard is 1e-18)
                tev_arr = np.where(fresh_t, 1e-4, tev_core * stream_t / span_t)
            else:
                tev_arr = np.full(nw, tev_base)
            if lev_core_ring > 0.0:
                # (HIRATO Eq.25) per-ring Lamb-Oseen core = lev_core_ring × LOCAL ring size (mean edge length). The
                # compressed near-LE rings are SMALL -> small core -> STRONG bounded induction (recovers vortex lift);
                # far/large rings get a proportionally larger core (smooth). Floored at lev_roll_core*c for stability.
                wrn = wr.numpy()[:nw]
                rsz = 0.25 * (np.linalg.norm(wrn[:, 1] - wrn[:, 0], axis=1) + np.linalg.norm(wrn[:, 2] - wrn[:, 1], axis=1)
                              + np.linalg.norm(wrn[:, 3] - wrn[:, 2], axis=1) + np.linalg.norm(wrn[:, 0] - wrn[:, 3], axis=1))
                levc = np.maximum(lev_core_ring * rsz, lev_roll_core * chord)
                # CRITICAL: the TEV keeps the STANDARD near-singular treatment in the SOLVE/FORCE (matches the
                # attached-UVLM baseline that uses the singular ring_vel — WAKE_CORE on the TEV softens the Kutta
                # condition and drops the baseline C_N ~30%). The Lamb-Oseen per-ring core applies ONLY to the LEV.
                # CONVECTION keeps WAKE_CORE on the TEV for free-wake stability.
                cc_np = np.where(islev_np, levc, tev_arr).astype(NP)           # rhs + force: TEV fresh~singular, aged cored (tev_core)
                cf_np = cc_np
                cconv_np = np.where(islev_np, levc, ug.WAKE_CORE).astype(NP)   # convect: TEV = WAKE_CORE (stability)
            else:
                # SOLVE/convect core (stabilizing, resolution-adaptive) and FORCE core (small, for held-lift sharpness)
                tev_legacy = tev_arr if use_tevcore else np.full(nw, ug.WAKE_CORE)
                cc_np = np.where(islev_np, lev_core_abs, tev_legacy).astype(NP)
                cf_np = np.where(islev_np, lev_core_force, tev_legacy).astype(NP)
                cconv_np = cc_np
            wcore_dev = wp.array(cc_np, dtype=DTYPE, device=dev)
            wcore_force_dev = wp.array(cf_np, dtype=DTYPE, device=dev)
            wcore_conv_dev = wp.array(cconv_np, dtype=DTYPE, device=dev)
        # ==== S5 holding / detachment envelope (Li JFM 2023 four-phase): modulate each LEV wake ring's strength
        # by its age since shedding. 'hold' = sustain for tau_hold then gentle viscous decay; 'hold_detach' =
        # sustain then SHARP cut (secondary vortex severs the feeding shear layer -> rapid lift collapse = the
        # rise-peak-FALL). Applied BEFORE the solve/force so rhs + Bernoulli see the enveloped LEV. ====
        if lev_hold_mode != 'inviscid' and lev_in_wake and nw > 0:
            born = np.asarray(lev_born[:nw]); s0 = np.asarray(lev_s0[:nw]); islev = born >= 0
            if np.any(islev):
                age = (t - born[islev]) * dt
                ov = np.maximum(age - tau_hold, 0.0)
                if lev_hold_mode == 'hold':
                    env = np.where(age < tau_hold, 1.0, np.exp(-ov / max(tau_hold, 1e-9)))
                else:                                    # 'hold_detach': sharp cut over 0.3*tau_hold
                    env = np.clip(1.0 - ov / max(0.3 * tau_hold, 1e-9), 0.0, 1.0)
                wgh = wg.numpy(); idxs = np.nonzero(islev)[0]
                wgh[idxs] = (s0[islev] * env).astype(wgh.dtype)
                wg = wp.array(wgh, dtype=DTYPE, device=dev)
        corners, cvel = twisted_state(C0, t * dt, A_f, A_t, Om, phi, x_ea, half_span,
                                      swept_axis=swept_axis, root_off=root_off, ramp=ramp)
        if deform_hook is not None:                # (P2-S6) flexible replay offset
            du_, dv_ = deform_hook(t * dt)
            corners = corners + du_
            cvel = cvel + dv_
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
        # CONSISTENT resolution-adaptive core in the SOLVE too (toggle lev_consistent): regularizes the singular
        # LEV→bound feedback that makes nc/ns NON-convergent. The core shrinks as the grid refines -> the result
        # CONVERGES to a grid-independent value (vs the singular kernel which drifts 7→3 with nc + blows up).
        if ((use_wcore and lev_consistent) or use_tevcore) and nw > 0:
            if wake_f32:
                wp.launch(_rhs_base_k, dim=npan, inputs=[col, nrm, Vw, vcol], outputs=[rhs], device=dev)
                wp.launch(_rhs_wake_chunk_wcore_f32, dim=(npan, _NCH),
                          inputs=[col, nrm, wr, wg, wcore_dev, nw, _NCH], outputs=[rhs], device=dev)
            else:
                wp.launch(_rhs_moving_wcore, dim=npan, inputs=[col, nrm, Vw, vcol, wr, wg, wcore_dev, nw],
                          outputs=[rhs], device=dev)
        else:
            wp.launch(_rhs_moving_f32 if wake_f32 else ug.rhs_moving_kernel,
                      dim=npan, inputs=[col, nrm, Vw, vcol, wr, wg, nw], outputs=[rhs], device=dev)
        if use_ansari:   # (HIRATO) LEV sheet OVER the suction surface: build ring geometry from the persistent
            # state (strip j, chordwise fraction f, strength g) using the CURRENT wing geometry; the ring sits at
            # LE + f*chord*chordhat + (lev_rollh*f*c)*normal -> over the suction surface, lifting off as it rolls
            # aft (Hirato Fig.11 spiral). Induce on the bound (fold into rhs) + keep Vlev_a for the Bernoulli force.
            Vlev_a.zero_(); lev_frame_rings = np.zeros((0, 4, 3)); lev_frame_g = np.zeros(0)
            if len(lev_aj) > 0:
                c3a = corners.reshape(nc + 1, ns + 1, 3); nle_a = nrm.numpy()[:ns]
                LEl = c3a[0, lev_aj]; LEr = c3a[0, lev_aj + 1]
                chl = c3a[nc, lev_aj] - LEl; chrr = c3a[nc, lev_aj + 1] - LEr     # chord vectors (LE->TE) per ring's strip
                n_k = nle_a[lev_aj]; c_k = 0.5 * (np.linalg.norm(chl, axis=1) + np.linalg.norm(chrr, axis=1)) + 1e-9
                # ROLL-UP geometry (Hirato Fig.11 spiral): chordwise position SATURATES near the forward chord while
                # the height grows -> the sheet curls UP above the forward suction surface instead of spreading flat.
                fpos = 0.45 * (1.0 - np.exp(-2.2 * lev_af))             # chordwise fraction, saturates ~0.45c (curl)
                hk = (lev_rollh * c_k * (lev_af + 0.06))[:, None]       # roll-up height above the suction surface (lifts off)
                f0 = fpos[:, None]; dch = (U * dt / c_k)[:, None]
                a0c = LEl + f0 * chl + hk * n_k; a1c = LEr + f0 * chrr + hk * n_k
                a2c = a1c + dch * chrr; a3c = a0c + dch * chl
                levring = np.stack([a0c, a1c, a2c, a3c], axis=1).astype(NP)
                lev_frame_rings = levring; lev_frame_g = lev_ag.copy()   # for viz
                # Lamb-Oseen regularization (Hirato Eq.25): small core so the near-surface LEV sheet does not induce
                # a singular velocity on the collocations (the L=26N blow-up). Core ~ a few % chord.
                core_a = np.full(len(lev_aj), max(lev_roll_core, 0.05) * chord, dtype=NP)
                lev_wr_a = wp.array(levring, dtype=V3, device=dev); lev_wg_a = wp.array(lev_ag.astype(NP), dtype=DTYPE, device=dev)
                lev_core_a = wp.array(core_a, dtype=DTYPE, device=dev)
                wp.launch(_col_wake_wcore, dim=npan, inputs=[col, lev_wr_a, lev_wg_a, lev_core_a, len(lev_aj)], outputs=[Vlev_a], device=dev)
                # NOTE: the LEV-sheet induction is used for the Bernoulli FORCE only (added to Vcol), NOT folded into
                # the solve rhs. The sheet stays anchored over the wing (does not convect away), so folding it into
                # the solve creates an unstable near-field feedback that accumulates (L blows up). Force-only is the
                # validated approach (cf. lev_merge): the LEV's suction on the wing surface ADDS lift via Bernoulli.
        if part_lev and lev_cons and np_part > 0:   # CONSERVATIVE rVPM: fold the CONVECTING LEV-particle induction
            # into the solve RHS -> the bound circulation is REDUCED by the shed LEV (Kelvin). Unlike the FIXED
            # core (which exploded - persistent near-field feedback), the particles CONVECT downstream so their
            # induction on the bound DECAYS -> stable. The LEV lift then emerges through the (reduced) bound KJ +
            # the particle induction in the Bernoulli, consistently (convention C) -> counts the convecting LEV.
            wp.launch(col_particle_vel_kernel, dim=npan, inputs=[col, pp, pa, ps, np_part], outputs=[Vpart], device=dev)
            wp.launch(rhs_add_lev_kernel, dim=npan, inputs=[nrm, Vpart], outputs=[rhs], device=dev)
        if lev_merge:   # coherent-LEV-core induced velocity at collocations (for the Bernoulli force ONLY -
            # NOT folded into the solve: coupling it reduces the bound circulation and the bound-reduction
            # dominates the LEV's own lift -> net DROP. Force-only -> the LEV's suction ADDS lift (overshoot).
            wp.launch(ug.col_wake_vel_kernel, dim=npan, inputs=[col, lev_rw, lev_gw, ns], outputs=[Vlev], device=dev)
        gamma = batched_dense_solve(AIC, rhs, dev)
        hirato_gL = None; hirato_wr = None; hirato_A0pre = None
        if lev_shed_mode == 'hirato' and (lev_in_wake or use_ansari):   # (H14) run the LESP=crit constraint for the
            # ================= FAITHFUL HIRATO LEV: implicit LESP = LESP_crit constraint (paper Fig.6) =========
            # The paper's core: when a strip's LESP (=A0) exceeds LESP_crit, an LEV ring is shed whose strength is
            # set — BY ITERATION (Fig.6) — so that the RE-SOLVED bound circulation brings that strip's LESP back
            # EXACTLY to LESP_crit. Because the bound circulation is linear in the RHS and A0 is a linear functional
            # of the bound circulation, A0 is AFFINE in the shed LEV strengths -> we solve for the strengths in ONE
            # linear step (the exact fixed point) instead of iterating. This is the mechanism the current explicit
            # 'kelvin' shed only APPROXIMATES (it shed the excess without re-solving, so post-solve A0 != crit). ==
            cc0 = rings.numpy(); g0 = gamma.numpy().reshape(-1); vcn0 = vcol.numpy()
            tcr0 = 0.5 * ((cc0[:, 2] - cc0[:, 0]) + (cc0[:, 3] - cc0[:, 1])); tcn0 = np.linalg.norm(tcr0, axis=1) + 1e-15
            tcnm0 = tcn0.reshape(nc, ns); c_str0 = tcnm0.sum(0); cum0 = np.cumsum(tcnm0, axis=0)
            vr_le0 = (np.asarray(Vinf) - vcn0)[:ns]; Ur0 = np.linalg.norm(vr_le0, axis=1) + 1e-9
            xref0 = 0.10; i_ref0 = np.argmax(cum0 >= (xref0 * c_str0)[None, :], axis=0)
            th10 = np.arccos(np.clip(1.0 - 2.0 * xref0, -1.0, 1.0))
            kA0 = 1.13 / (Ur0 * c_str0 * (th10 + np.sin(th10)) + 1e-12)   # A0[j] = kA0[j] * Gamma_1[j]  (Hirato Eq.6)
            refidx = i_ref0 * ns + np.arange(ns)                           # flat index of the forwardmost-leg (Γ_1) ring
            A0_0 = kA0 * g0[refidx]                                        # LESP per strip from the LEV-free solve
            shed_m = np.abs(A0_0) > a0_crit                               # supercritical strips shed an LEV (Fig.6 gate)
            hirato_A0pre = A0_0.copy()                                    # PRE-constraint LESP (before the LEV pulls it to crit)
            hirato_gL = np.zeros(ns, dtype=NP)
            # nascent LEV ring geometry — built ALWAYS (the shed section writes it directly for 'hirato'). Hirato
            # Eq.7 / Ansari 1/3 rule: the ring FRONT sits at the geometric LE; its AFT edge (the shared edge with the
            # previous LEV ring) is placed at x_{L,n-1} = 2/3 x_LE + 1/3 x_{L,n-2} — only 1/3 of the way toward the
            # previous (convected) shed edge -> the LEV sheet stays COMPRESSED near the LE and rolls up over the
            # suction surface (Fig.11), rather than trailing straight into the freestream wake and losing its
            # (Polhamus) vortex lift. lpl/lpr hold the previous step's shared (aft) edge.
            nle0 = nrm.numpy()[:ns]; c3 = corners.reshape(nc + 1, ns + 1, 3); offv = lev_le_off * chord
            fL = c3[0, :ns] + nle0 * offv; fR = c3[0, 1:ns + 1] + nle0 * offv    # FRONT = geometric LE
            lpl0 = lpl.numpy(); lpr0 = lpr.numpy(); Vd = np.asarray(Vinf) * dt
            if lev_first == 0:
                aL = (2.0 / 3.0) * fL + (1.0 / 3.0) * (lpl0[:ns] + Vd)           # aft = 2/3 LE + 1/3 (prev shared edge, convected)
                aR = (2.0 / 3.0) * fR + (1.0 / 3.0) * (lpr0[:ns] + Vd)
            else:
                aL = fL + Vd; aR = fR + Vd
            hirato_wr = np.stack([fL, fR, aR, aL], axis=1).astype(NP)            # (ns,4,3): 0,1=front(LE); 2,3=aft(shared)
            if shed_m.any():
                lev_wr_w = wp.array(hirato_wr, dtype=V3, device=dev)
                INFw = wp.zeros((npan, ns), dtype=DTYPE, device=dev)
                wp.launch(_lev_inf_kernel, dim=(npan, ns), inputs=[col, nrm, lev_wr_w, DTYPE(lev_core_abs), ns],
                          outputs=[INFw], device=dev)
                INF = INFw.numpy()                                        # (npan,ns): unit-LEV-j normal-flow at colloc i
                # PSEUDOVORTEX RING (Hirato Fig.5, Eq.8->9): a ring from the GEOMETRIC leading edge (front) to the
                # aft edge of the FRONTMOST bound-vortex ring (aft), of the SAME strength Gamma_tau = Gamma_L as the
                # nascent LEV ring. It (a) cancels the spurious spanwise vortex filament that the LEV ring would
                # otherwise leave at the geometric LE, and (b) places strong circulation right where the LESP is
                # measured -> the shed LEV becomes EFFECTIVE at reducing LESP (physical strength, not the huge value
                # a bare cored ring needs), which lets the constraint reach LESP_crit. It is a per-step near-LE
                # device (NOT convected into the wake). Combined shed influence = LEV ring + pseudovortex ring.
                if lev_pseudo:
                    pf_L = c3[0, :ns]; pf_R = c3[0, 1:ns + 1]              # front = geometric LE (x=0 line)
                    pa_R = cc0[:ns, 2]; pa_L = cc0[:ns, 3]                 # aft = frontmost bound ring's aft edge
                    pseudo_wr0 = np.stack([pf_L, pf_R, pa_R, pa_L], axis=1).astype(NP)
                    ps_w = wp.array(pseudo_wr0, dtype=V3, device=dev)
                    INFp = wp.zeros((npan, ns), dtype=DTYPE, device=dev)
                    wp.launch(_lev_inf_kernel, dim=(npan, ns), inputs=[col, nrm, ps_w, DTYPE(lev_core_force), ns],
                              outputs=[INFp], device=dev)
                    INF = INF + INFp.numpy()                              # Gamma_tau = Gamma_L -> add unit influences
                # sensitivity of the bound to each unit LEV: G[j] = AIC^{-1} (-INF[:,j])  (rhs gets -wg·V·n)
                AIC0 = AIC.numpy()[0]
                A_tiled = wp.array(np.broadcast_to(AIC0, (ns, npan, npan)).copy(), dtype=DTYPE, device=dev)
                rhs_cols = wp.array(np.ascontiguousarray(-INF.T).astype(NP), dtype=DTYPE, device=dev)  # (ns,npan)
                G = batched_dense_solve(A_tiled, rhs_cols, dev).numpy()   # (ns,npan) bound response to each unit LEV
                S = kA0[:, None] * G[:, refidx].T                        # S[k,j] = dA0[k]/dgL[j] (affine LESP sensitivity)
                idx = np.where(shed_m)[0]
                target = a0_crit * np.sign(A0_0)                         # drive LESP to +/- LESP_crit (Hirato: hold at crit)
                for _ in range(max(int(lev_iter), 1)):                    # affine -> 1 sweep exact; >1 tightens spanwise coupling
                    resid = (target - A0_0)[idx]
                    Ssub = S[np.ix_(idx, idx)]
                    reg = 1e-6 * (np.abs(np.diag(Ssub)).mean() + 1e-12)
                    try:
                        gsub = np.linalg.solve(Ssub + reg * np.eye(len(idx)), resid)
                    except np.linalg.LinAlgError:
                        gsub = resid / (np.diag(Ssub) + np.sign(np.diag(Ssub) + 1e-30) * 1e-9)
                    # ACCUMULATE the correction and advance the predicted LESP by the INCREMENT only. The old
                    # `hirato_gL[idx] = gsub; A0_0 += S @ hirato_gL` pair zeroed the strengths on sweep 2 (resid~0
                    # -> gsub~0 REPLACES sweep 1) and double-added sweep-1's effect -> lev_iter>1 silently disabled
                    # the shed. lev_iter=1 (production) is bit-identical under both forms.
                    hirato_gL[idx] += gsub
                    A0_0 = A0_0 + (S[:, idx] @ gsub)                     # predicted post-solve LESP (affine); loop refines
                # SAFETY cap: guard ONLY a near-singular S-row runaway. The LEV's regularized (cored) induction is
                # WEAK per unit strength, so the strength needed to reach LESP_crit is several x the naive Kelvin
                # excess Gamma_1 — a tight cap would prevent the constraint from actually reaching crit. Cap at the
                # larger of 30x the excess and half the strip's bound circulation (physical ceiling: the LEV cannot
                # exceed the loading that fed it), which never binds in normal operation.
                excG = np.maximum(np.abs(kA0 * g0[refidx]) - a0_crit, 0.0)
                if lev_cap_exc:
                    # (H11 2026-07-04) cap the shed ring at EXACTLY the Kelvin excess circulation — the physical
                    # amount the bound must give up (Gamma units, no factor). The implicit constraint's cored-ring
                    # near-field is WEAK per unit Gamma (needs several x the excess to pin LESP=crit at the ref
                    # point), but the ring's WAKE-borne circulation acts at FULL strength -> over-strong rings
                    # pollute everything downstream (P0 runaway feedback, P1 A0 sub-critical collapse, tw45
                    # fictional bern/vnf loads at kinematically-feathered phases). With the 1x cap the constraint
                    # under-delivers (post-solve A0 may stay >crit, same as the kelvin path) — the crit-clip in
                    # the les/faure closures carries that semantic, while the wake stays physically sized.
                    capG = excG / (kA0 + 1e-12) + 1e-6
                else:
                    capG = np.maximum(excG / (kA0 + 1e-12) * 30.0, 0.5 * np.abs(g0[refidx])) + 1e-6
                hirato_gL = np.clip(hirato_gL, -capG, capG)
                # RE-SOLVE the bound with the LEV rings in the RHS (rhs_final = rhs_base - INF @ gL) -> post-solve
                # LESP is at LESP_crit on the shedding strips (the paper's converged constraint).
                rhs_f = rhs.numpy().reshape(-1) - INF @ hirato_gL
                gamma = batched_dense_solve(AIC, wp.array(rhs_f[None, :].astype(NP), dtype=DTYPE, device=dev), dev)
                if os.environ.get('HIRATO_DBG'):
                    A0_post = kA0 * gamma.numpy().reshape(-1)[refidx]        # ACTUAL post-solve LESP
                    sm = shed_m
                    print(f"  [hirato t={t:3d}] shed={int(sm.sum()):2d}/{ns}  |A0_pre|max={np.abs(kA0*g0[refidx]).max():.3f}"
                          f"  |A0_post[shed]|: {np.abs(A0_post[sm]).min():.3f}..{np.abs(A0_post[sm]).max():.3f}"
                          f"  (crit={a0_crit})  |gL|max={np.abs(hirato_gL).max():.4f}", flush=True)
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
        # FORCE induction: cored (consistent with solve+convect) so the held-lift contribution CONVERGES with grid
        # instead of drifting with the singular near field. Same adaptive core shrinks toward the singular limit as refined.
        if ((use_wcore and lev_consistent) or use_tevcore) and nw > 0:
            if wake_f32:
                wp.launch(_col_wake_chunk_wcore_f32, dim=(npan, _NCH),
                          inputs=[col, wr, wg, wcore_dev, nw, _NCH], outputs=[Vwk], device=dev)
            else:
                wp.launch(_col_wake_wcore, dim=npan, inputs=[col, wr, wg, wcore_dev, nw],
                          outputs=[Vwk], device=dev)
        else:
            wp.launch(_col_wake_f32 if wake_f32 else ug.col_wake_vel_kernel,
                      dim=npan, inputs=[col, wr, wg, nw], outputs=[Vwk], device=dev)
        cc = rings.numpy(); g = gamma.numpy().reshape(-1); gp = gprev.numpy().reshape(-1)
        Vcol = np.asarray(Vinf) - vcn + Vwk.numpy()                     # full local velocity at panels
        if part_lev and lev_cons and np_part > 0:
            # convention C completion (S3b force audit 2026-07-19): the particles already reduce
            # the bound circulation via the rhs fold-in; their OWN induced velocity at the surface
            # (the LEV suction) must enter the Bernoulli dp too — same Vpart snapshot as the solve.
            Vcol = Vcol + Vpart.numpy()
        if lev_merge:
            Vcol = Vcol + Vlev.numpy()                                  # coherent LEV core induction (Bernoulli)
        if use_ansari and not lev_impulse:
            Vcol = Vcol + Vlev_a.numpy()                                # (HIRATO) LEV-sheet induction enters the Bernoulli force.
            # Under lev_impulse (H16) the LEV force comes from the vortex-IMPULSE derivative instead (grid-independent),
            # so the Vlev_a surface-pressure contribution is SKIPPED here to avoid double-counting.
        # NOTE: the LEV-particle force is the LE-referenced VORTEX-IMPULSE (added to Lh_imp below), NOT the
        # surface-Bernoulli induction (that only gets the LEV x bound cross-term, missing the LEV's own KJ lift).
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
        # PER-PANEL PRESSURE CLAMP (near-field regularization): a per-element LEV ring that convects through the
        # near-field of a bound collocation can drive a single panel's dp near-singular (|Cp|>>1, unphysical). Cap
        # |dp| at |Cp|<=8 of a STABLE reference dynamic pressure (freestream + max flap-tip speed) so one singular
        # panel can't poison the cycle force. Physical surface Cp stays well within +-8; only artifacts are clipped.
        Vtip = 2.0 * np.pi * freq * half_span * np.sin(A_f)            # max flap-tip speed (stable, kinematic)
        q_ref = 0.5 * ug.RHO * (np.linalg.norm(Vinf) + Vtip) ** 2
        # ==== (H10 2026-07-04) UNIFIED alpha_eff-KIRCHHOFF SEPARATION STATE. Fig16 instantaneous data shows the
        # attached UVLM over-predicts force amplitude 2.5-4x at ALL twist (mid-stroke alpha_eff~44deg = deep dynamic
        # stall; measured lift amplitude collapses, measured thrust has NO mid-stroke suction pulses). ONE per-strip
        # separation state fsep(alpha_eff) — SAME NACA-2406 constants as Fix1 (alpha_ss=geo_stall_deg, width=
        # geo_stall_width), alpha_eff from the KINEMATIC LE flow (freestream+body, the A0-normalization family) —
        # drives all three force closures: (1) kirch_cn: Bernoulli CIRCULATORY pressure x Kirchhoff CN factor
        # ((1+sqrt f)/2)^2, VECTORIAL (pressure collapse has no preferred axis; fixes the deep-twist backward-tilt
        # drag fiction); added-mass (rho dG/dt) is NON-circulatory inviscid inertia -> NOT scaled. (2) les_att (in
        # the les block): realized LE suction x fsep -> suction DIES at full separation (Polhamus: it rotates into
        # the vnf normal force, which is already k_v-capped). (3) vnf unchanged (Polhamus branch is the separated-
        # flow limit). Quasi-steady limitation (no dynamic-stall lag; would need new time constants) -> documented.
        fsep_le = None
        if kirch_cn or kirch_tw or les_att or prof_drag or kirch_blend:   # separation state consumers (incl. P1 kirch_blend)
            vr_k = np.asarray(Vinf) - vcn                                   # kinematic relative flow (no induction)
            nn_k = nrm.numpy()

            def _fsep_row(i0):                                              # Kirchhoff fsep from row i0's strip incidence
                sl = slice(i0 * ns, (i0 + 1) * ns)
                v_r = vr_k[sl]; n_r = nn_k[sl]
                sa_r = np.sum(v_r * n_r, axis=1) / (np.linalg.norm(v_r, axis=1) + 1e-9)
                a_r = np.abs(np.arcsin(np.clip(sa_r, -0.999, 0.999)))
                ass_k = np.radians(geo_stall_deg)
                return np.where(a_r <= ass_k, 1.0,
                                np.clip(1.0 - (a_r - ass_k) / np.radians(geo_stall_width), 0.0, 1.0))
            fsep_le = _fsep_row(0)                                          # LE row: gates the LE suction (les_att)
            if fsep_lag:
                # (H13 2026-07-04) GOMAN-KHRABROV 1st-order separation-state lag: d fsep/dt = (fsep_qs - fsep)/tau,
                # tau = tau_star * c / (2 U_rel) (convective; tau_star ~ 4.5, LITERATURE airfoil constant, not a
                # RoboEagle fit). One standard mechanism, two measured effects the quasi-steady gate misses:
                # (a) separation DELAY at the stroke reversals (the wing arrives at deep alpha faster than the
                # boundary layer separates -> measured reversal drag << static-stall value), (b) reattachment lag
                # -> downstroke/upstroke HYSTERESIS (the measured mean-lift asymmetry). State kept across steps.
                Ur_st = np.linalg.norm((np.asarray(Vinf) - vcn)[:ns], axis=1) + 1e-9
                c_st = tcn.reshape(nc, ns).sum(0)                           # local strip chord (this step)
                tau_st = fsep_tau * c_st / (2.0 * Ur_st)
                if fsep_state is None:
                    fsep_state = fsep_le.copy()
                else:
                    fsep_state = fsep_state + (fsep_le - fsep_state) * np.clip(dt / tau_st, 0.0, 1.0)
                fsep_le = fsep_state
        if kirch_blend:
            # (P1 2026-07-05) DOUBLE-COUNT FIX. H19's prof_drag (Hoerner) ADDED to the uncapped bound Bernoulli's
            # backward-tilt component -> deep-twist over-drag (-4.7N). Fix: ONE consistent force vector blended by
            # the separated fraction. Attached: full Bernoulli (circulatory + added mass). Separated: flat-plate CN
            # normal pressure dp_fp = q*Cd_max*sin^2(aeff)*sign(sa) (Cd_max=cd_form, flat-plate constant). The CN
            # force projects to BOTH lift (reduced, sin^2*cos) AND pressure drag (sin^3) from the SAME vector -> no
            # double-count. prof_drag is gated OFF below when kirch_blend is on.
            dp_c = ug.RHO * (np.sum(Vcol * tc, axis=1) * dGdx + np.sum(Vcol * ts, axis=1) * dGdy)
            dp_a = ug.RHO * dGdt
            dp_c = np.clip(dp_c, -cp_cap * q_ref, cp_cap * q_ref); dp_a = np.clip(dp_a, -cp_cap * q_ref, cp_cap * q_ref)
            dp_att = dp_c + dp_a                                              # attached Bernoulli pressure
            vr_k = np.asarray(Vinf) - vcn; vrm_k = np.linalg.norm(vr_k, axis=1) + 1e-9
            sa_k = np.sum(vr_k * nrm.numpy(), axis=1) / vrm_k                 # sin(aeff) per panel (kinematic)
            dp_fp = q_ref * (cd_form * sa_k ** 2) * np.sign(sa_k)             # flat-plate CN pressure (along +n)
            dp_fp = np.clip(dp_fp, -cp_cap * q_ref, cp_cap * q_ref)
            fsep_p = np.tile(fsep_le, nc)                                     # per-panel separated fraction (panel p -> strip j=p%ns)
            dp_blend = fsep_p * dp_att + (1.0 - fsep_p) * dp_fp
            Fb = dp_blend[:, None] * area[:, None] * nrm.numpy()
        elif kirch_cn:
            dp_c = ug.RHO * (np.sum(Vcol * tc, axis=1) * dGdx + np.sum(Vcol * ts, axis=1) * dGdy)
            dp_a = ug.RHO * dGdt                                            # added mass: survives separation
            dp_c = np.clip(dp_c, -cp_cap * q_ref, cp_cap * q_ref); dp_a = np.clip(dp_a, -cp_cap * q_ref, cp_cap * q_ref)
            # CIRCULATORY loading gate at the 3/4-CHORD row: the classic unsteady thin-airfoil result — the
            # circulatory load follows the effective incidence at 3c/4, which carries the PITCH-RATE term
            # theta_dot*(3c/4 - x_ea)/V that the LE row misses. At tw45 mid-stroke the section is kinematically
            # FEATHERED at the LE (fsep_LE~1) while the twist RATE peaks (twist zero-crossing) -> only the 3c/4
            # incidence sees the rotation-induced stall that the measured Fig16 mid-downstroke collapse shows.
            # vcn already contains the full rotation velocity distribution -> no new constants, standard theory.
            fsep_cn = _fsep_row(min(int(round(0.75 * nc)), nc - 1))
            fac_cn = np.tile(((1.0 + np.sqrt(fsep_cn)) / 2.0) ** 2, nc)     # Kirchhoff CN factor, per strip
            Fb = (dp_c * fac_cn + dp_a)[:, None] * area[:, None] * nrm.numpy()
        elif kirch_tw:
            # (2026-07-10 案A 变体B, research_bern_twist.md M1) the separated-state circulatory SENSITIVITY is
            # over-predicted ~1/fac (Kirchhoff bound 2.8x at the floor) -> the model over-relieves the load that
            # feathering twist removes. Correct ONLY the twist-induced share via the first-order incidence split
            # dp_c' = dp_c * (1 - (1-fac) * psi_t/alpha_34): tw0 identical by construction (psi_t = 0); the
            # full-force gate (kirch_cn) is measured to kill the flapping main lift (5.9 -> 2.9) and is rejected.
            dp_c = ug.RHO * (np.sum(Vcol * tc, axis=1) * dGdx + np.sum(Vcol * ts, axis=1) * dGdy)
            dp_a = ug.RHO * dGdt                                            # added mass: survives separation
            dp_c = np.clip(dp_c, -cp_cap * q_ref, cp_cap * q_ref); dp_a = np.clip(dp_a, -cp_cap * q_ref, cp_cap * q_ref)
            i34 = min(int(round(0.75 * nc)), nc - 1)
            sl34 = slice(i34 * ns, (i34 + 1) * ns)
            v34 = vr_k[sl34]; n34 = nn_k[sl34]
            sa34 = np.sum(v34 * n34, axis=1) / (np.linalg.norm(v34, axis=1) + 1e-9)
            a34 = np.arcsin(np.clip(sa34, -0.999, 0.999))                   # SIGNED 3c/4 strip incidence
            fsep_cn = _fsep_row(i34)
            if fsep_lag:                                                    # same GK lag, own state (3c/4 row)
                Ur34 = np.linalg.norm(v34, axis=1) + 1e-9
                c34 = tcn.reshape(nc, ns).sum(0)
                tau34 = fsep_tau * c34 / (2.0 * Ur34)
                if fsep_state_cn is None:
                    fsep_state_cn = fsep_cn.copy()
                else:
                    fsep_state_cn = fsep_state_cn + (fsep_cn - fsep_state_cn) * np.clip(dt / tau34, 0.0, 1.0)
                fsep_cn = fsep_state_cn
            fac_tw = ((1.0 + np.sqrt(fsep_cn)) / 2.0) ** 2
            psi_t = A_t * yfrac * np.sin(Om * (t * dt) + phi)               # geometric twist pitch (= geo_stall's)
            a_gd = np.where(np.abs(a34) < 0.05, np.sign(a34) * 0.05 + (a34 == 0) * 0.05, a34)
            ratio = np.clip(psi_t / a_gd, -1.0, 1.0)                        # twist share of the 3c/4 incidence
            w_tw = np.tile(1.0 - (1.0 - fac_tw) * ratio, nc)                # tw0 -> exactly 1
            Fb = (dp_c * w_tw + dp_a)[:, None] * area[:, None] * nrm.numpy()
        else:
            dp = np.clip(dp, -cp_cap * q_ref, cp_cap * q_ref)
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
        Fzb_tot[t] = float(np.sum(Fb[:, 2])); Fxb_tot[t] = float(np.sum(Fb[:, 0]))   # body-force accumulator (base)
        # ==== Fix1: QUASI-STEADY GEOMETRIC STALL lift loss (twist-driven, frequency-independent). The inviscid
        # UVLM has no separation -> lift rises monotonically with twist; the real wing's outer span exceeds the
        # static stall angle (geometric pitch psi_geo = aoa + twist*y/span > alpha_ss ~12deg) and LOSES lift
        # (measured rises-peaks~15deg-falls). The loss uses the GEOMETRIC pitch (NOT the flap-dominated alpha_eff,
        # which would fire everywhere on the +-45 stroke and destroy cruise) -> identically zero at twist=0 (cruise
        # untouched by construction), and frequency-independent (no Om-velocity term). Per-strip lift-loss FRACTION
        # from the NACA-2406 static polar shape (linear post-stall decay, strip_aero.py:108): loss_frac ~ linear in
        # over-angle -> span-integral ~ twist^2 (stalled-fraction prop twist x mean-over-angle prop twist). ====
        if geo_stall:
            psi_t = A_t * yfrac * (1.0 if geo_stall_peak else np.sin(Om * (t * dt) + phi))  # geom twist pitch per strip (rad)
            psi_abs = np.abs(aoa_rad + psi_t)                                  # geometric section incidence magnitude
            ass = np.radians(geo_stall_deg)
            # KIRCHHOFF trailing-edge separation: f = separation point (1=attached, 0=fully separated), drops over
            # [alpha_ss, alpha_ss+width]; CL factor = ((1+sqrt f)/2)^2 (1 attached, 0.25 fully separated -> bounded,
            # plateaus at the flat-plate lift; does NOT vanish). loss_frac = 1 - factor (0..0.75), ~linear past stall.
            if gk_stall:
                # (病灶#1) FULL Goman-Khrabrov with TRUE TIME DELAY. The static geometric-stall gate above has no
                # frequency term -> dL/df=0. GK evaluates the Kirchhoff separation at the DELAYED incidence
                # psi(t - tau2) (static-stall-angle delay), then tau1-relaxes. A first-order alpha-tau2*alphadot
                # is WRONG here: Om*tau2 ~ 1.9 (not <<1), so it amplitudes the swing 2.15x (deeper stall, dL/df
                # MORE negative — smoke 2026-07-19) instead of phase-delaying. The true delay leaves the amplitude
                # unchanged and shifts phase: at the incidence peak the delayed value is the smaller pre-peak angle
                # -> peak stall RELIEVED, more so at higher f (larger phase lag) -> mean lift GROWS with frequency
                # (Chiereghin-Cleaver-Gursul 2019, alpha-gated). tau1/tau2 = Ayancik-Mulleners 2022 power-law.
                c_st_geo = tcn.reshape(nc, ns).sum(0)                          # per-strip chord
                psi_sgn = aoa_rad + psi_t                                      # SIGNED geometric section incidence
                if gk_X is None:                                              # motion-fixed per-strip delay (once)
                    psidot_pk = np.abs(A_t * yfrac * Om)                       # peak geometric pitch rate (~f)
                    r_pk = np.maximum(psidot_pk * c_st_geo / (2.0 * max(U, 1e-6)), gk_rfloor)
                    tau2 = (gk_tau1 + gk_c2 * r_pk ** (-7.0 / 9.0)) * c_st_geo / max(U, 1e-6)
                    gk_ndelay = np.clip(np.round(tau2 / dt).astype(int), 0, N - 1)
                    gk_tau1_st = gk_tau1 * c_st_geo / max(U, 1e-6)
                    gk_hist = np.zeros((N, ns))
                    gk_X = np.ones(ns)
                gk_hist[t] = psi_sgn
                psi_del = np.abs(gk_hist[np.maximum(t - gk_ndelay, 0), np.arange(ns)])
                fsep_qs = np.clip(1.0 - (psi_del - ass) / np.radians(geo_stall_width), 0.0, 1.0)
                fsep_qs = np.where(psi_del <= ass, 1.0, fsep_qs)
                gk_X = gk_X + (fsep_qs - gk_X) * (1.0 - np.exp(-dt / np.maximum(gk_tau1_st, 1e-9)))
                loss_frac = 1.0 - ((1.0 + np.sqrt(gk_X)) / 2.0) ** 2
            else:
                fsep = np.clip(1.0 - (psi_abs - ass) / np.radians(geo_stall_width), 0.0, 1.0)
                fsep = np.where(psi_abs <= ass, 1.0, fsep)
                loss_frac = 1.0 - ((1.0 + np.sqrt(fsep)) / 2.0) ** 2           # 0 attached (twist=0), ->0.75 separated
            if geo_stall_vec:
                # (C8 2026-07-04) VECTORIAL Kirchhoff: separation collapses the surface-pressure DIFFERENTIAL, i.e.
                # the strip's whole Bernoulli force VECTOR (normal-directed) — not just its +z projection. At deep
                # twist the attached-UVLM normal force tilts backward (bern T=-14N at tw45 vs measured ~-2N flat);
                # scaling the vector recovers the measured collapse+redirection (paper Fig16 text: twist "reduces
                # the wing surface pressure differential ... alters the pressure force direction, creating a larger
                # forward thrust component"). Same NACA-2406 alpha_ss/width constants as the legacy z-only branch.
                dFv = -loss_frac[None, :].repeat(nc, 0).reshape(-1)[:, None] * Fb    # per-panel (strip-wise factor)
                Lh_stall[t] = float(np.sum(dFv[:, 2])); Xh_stall[t] = float(np.sum(dFv[:, 0]))
                Fzb_tot[t] += Lh_stall[t]; Fxb_tot[t] += Xh_stall[t]
            else:
                Fb_strip_z = Fb[:, 2].reshape(nc, ns).sum(0)                   # per-strip Bernoulli lift (ns,)
                dLz = -loss_frac * np.maximum(Fb_strip_z, 0.0)                 # remove only positive lift on stalled strips
                Lh_stall[t] = float(np.sum(dLz)); Fzb_tot[t] += Lh_stall[t]    # lift-axis only (thrust handled by Fix2)
        # ==== FIRST-PRINCIPLES per-strip LESP A0 (Hirato/Gopalarathnam 2019, Eq.6) from the bound LE-row
        # circulation: A0[j] = 1.13*Gamma_{b,1}[j] / (U_rel*c*(arccos(1-2dx1/c)+sin(arccos(...)))). One value
        # per spanwise strip -> per-element. Drives both the LE-suction cap (S4) and the LEV shed rate (S3). ====
        # nc-CONVERGENT LESP: use the cumulative bound circulation up to a FIXED chord fraction x_ref (NOT the
        # single first panel with dx1=c/nc, which drifts with nc). In a telescoping ring lattice the cumulative
        # circulation at chord x = gamma at the panel reaching x, so Γ_ref[j] -> bound circ at x_ref (converges as
        # nc->inf). Δx1 = x_ref (fixed) -> th1 fixed -> A0 is grid-independent. (Hirato Eq.6 evaluated at fixed x_ref.)
        gm2 = g.reshape(nc, ns)                             # bound ring circulation (nc chordwise × ns spanwise)
        tcnm2 = tcn.reshape(nc, ns)                         # per-panel chordwise length
        c_strip = tcnm2.sum(0)                             # local chord per strip
        cumpos = np.cumsum(tcnm2, axis=0)                  # cumulative chordwise position (panel trailing edges)
        vr_le = (np.asarray(Vinf) - vcn)[:ns]              # LE-row body-relative flow
        Urel_le = np.linalg.norm(vr_le, axis=1) + 1e-9
        if a0_mode == 'sqrtx':
            # √x LIMIT (3D, FIRST-PRINCIPLES): the thin-airfoil bound sheet γ(x)→A0·√(2/x) near the LE, so the
            # FIRST panel's circulation Γ_1 → A0·√(8·Δx1). With a fine COSINE LE (Δx1→0) this extracts A0 from the
            # RESOLVED √x leading-edge behavior per spanwise strip (vs the crude x_ref=0.10c average). Hirato Eq.6
            # evaluated at the actual first-panel width Δx1 (per strip), so as nc↑ A0 → the true √x coefficient.
            dx1 = tcnm2[0]                                  # first (LE-most) panel chord per strip
            th1 = np.arccos(np.clip(1.0 - 2.0 * dx1 / (c_strip + 1e-12), -1.0, 1.0))   # per-strip (→0 as Δx1→0)
            A0 = 1.13 * gm2[0] / (Urel_le * c_strip * (th1 + np.sin(th1)) + 1e-12)
        elif a0_mode == 'downwash':
            # S2-EXACT LESP (2026-07-18 backport, research_rvpm_s2cases.md): A0 = -(1/pi)∫(W/U)dθ
            # per strip from the solver's OWN rhs (= -(Vinf+V_wake-vcol)·n̂ at collocation = the
            # thin-airfoil downwash functional W, wake/LEV-sheet fold-in included). Literature-exact
            # (Ramesh/UNSflow update_a0anda1) — no 1.13 grid factor, no x_ref truncation. S2 audit:
            # Eq.6-xref reads +12% (flat steady — the 1.13 itself), +18-26% (camber), and
            # 0.37→-0.14→0.67 (heavy LEV wake) vs exact — dynamically distorted exactly where the
            # gate matters. θ from the strip's own 3/4-panel collocation stations, bin-edge weights
            # extended to [0,π] (Σw=π exact, S2 quadrature family).
            rhs2 = rhs.numpy().reshape(nc, ns)
            xcol = cumpos - 0.25 * tcnm2                   # 3/4-panel collocation chord position
            thc = np.arccos(np.clip(1.0 - 2.0 * xcol / (c_strip[None, :] + 1e-12), -1.0, 1.0))
            ed = np.concatenate([np.zeros((1, ns)), 0.5 * (thc[1:] + thc[:-1]),
                                 np.full((1, ns), np.pi)], axis=0)
            wq = np.diff(ed, axis=0)                       # (nc, ns), Σ_i wq = π per strip
            A0 = -np.sum(wq * rhs2, axis=0) / (np.pi * Urel_le + 1e-12)
            th1 = np.arccos(np.clip(1.0 - 2.0 * 0.10, -1.0, 1.0))   # for the legacy dG1 conversion below
        else:                                              # 'xref': cumulative bound circ up to FIXED x_ref=0.10c (nc-robust legacy)
            xref_frac = 0.10
            i_ref = np.argmax(cumpos >= (xref_frac * c_strip)[None, :], axis=0)   # first panel reaching x_ref per strip
            Gamma_ref = gm2[i_ref, np.arange(ns)]          # cumulative bound circulation up to x_ref
            th1 = np.arccos(np.clip(1.0 - 2.0 * xref_frac, -1.0, 1.0))
            A0 = 1.13 * Gamma_ref / (Urel_le * c_strip * (th1 + np.sin(th1)) + 1e-12)   # finite-wing LESP per strip
        A0 = np.clip(np.nan_to_num(A0, nan=0.0, posinf=0.0, neginf=0.0), -3.0, 3.0)   # guard near-field blow-up
        # ==== L-B dynamic-stall closure (lb_closure, 2026-07-20, research_lb_formula.md) ====
        # Per-strip signed effective AoA from the LE-row kinematic flow, advance LBDynStrip states,
        # get the lagged separation f2 + LEV vortex lift CNv + tangential CT. These feed the force
        # section below (dynamic f2 replaces static geo_stall loss_frac; CNv/CT added as increments).
        lb_f2 = None; lb_CNv = None; lb_CT = None
        if lb_closure:
            from lb_dyn import LBDynStrip
            from lb_static import StaticPolar
            if lb_strips is None:
                _lb_pol = StaticPolar()
                lb_strips = [LBDynStrip(_lb_pol, lesp_crit=lb_lesp_crit, eta=lb_eta,
                                        Tp=lb_Tp, Tf=lb_Tf, Tv=lb_Tv, Tvl=lb_Tvl) for _ in range(ns)]
            vr_le_s = (np.asarray(Vinf) - vcn)[:ns]            # LE-row body-relative flow per strip
            sa_le = np.sum(vr_le_s, axis=1) / (np.linalg.norm(vr_le_s, axis=1) + 1e-9)  # signed sin a_eff
            aeff_le = np.arcsin(np.clip(sa_le, -0.999, 0.999))
            lb_f2 = np.ones(ns); lb_CNv = np.zeros(ns); lb_CT = np.zeros(ns)
            for j in range(ns):
                rj = lb_strips[j].step(float(aeff_le[j]), float(A0[j]),
                                       float(Urel_le[j]), float(c_strip[j]), dt)
                lb_f2[j] = rj["f2"]; lb_CNv[j] = rj["CNv"]; lb_CT[j] = rj["CT"]
            if os.environ.get("LB_DBG") and t % max(steps_per_cycle // 4, 1) == 0:
                lev_n = int(np.sum(np.array([s.lev_active_prev for s in lb_strips])))
                print(f"[lb t={t:4d}] |A0|max={np.abs(A0).max():.3f} (crit {lb_lesp_crit}) "
                      f"lev_active {lev_n}/{ns} | aeff_max={np.degrees(np.abs(aeff_le)).max():.0f}deg "
                      f"f2_mean={lb_f2.mean():.2f} CNv_max={np.abs(lb_CNv).max():.3f}", flush=True)
            # ---- L-B force increments on Fb (mutually exclusive with geo_stall; cruise-invariant as
            # f2->1/CNv->0/CT small in attached flow). (1) vectorial Kirchhoff scaling by dynamic f2;
            # (2) LEV vortex lift CNv along panel normal; (3) tangential CT suction decay as drag. ----
            nnp = nrm.numpy()
            loss_frac = 1.0 - ((1.0 + np.sqrt(lb_f2)) / 2.0) ** 2
            dFv = -np.tile(loss_frac, nc)[:, None] * Fb
            Lh_stall[t] = float(np.sum(dFv[:, 2])); Xh_stall[t] = float(np.sum(dFv[:, 0]))
            Fzb_tot[t] += Lh_stall[t]; Fxb_tot[t] += Xh_stall[t]
            vr_lb = (np.asarray(Vinf) - vcn); vrm_lb = np.linalg.norm(vr_lb, axis=1) + 1e-9
            qdyn_lb = 0.5 * ug.RHO * vrm_lb ** 2
            cpan_lb = np.tile(tcn.reshape(nc, ns).sum(0), nc)
            dy_lb = 2.0 * half_span / ns                                        # per-strip width (scalar, broadcasts)
            dF_lev = (qdyn_lb * np.tile(lb_CNv, nc) * cpan_lb * dy_lb)[:, None] * nnp
            Lh_vtx[t] = float(np.sum(dF_lev[:, 2])); Xh_vtx[t] = float(np.sum(dF_lev[:, 0]))
            Fzb_tot[t] += Lh_vtx[t]; Fxb_tot[t] += Xh_vtx[t]
            dD_ct = qdyn_lb * np.tile(np.abs(lb_CT), nc) * cpan_lb * dy_lb
            Xh_pd[t] = float(-np.sum(dD_ct))
            Fxb_tot[t] += Xh_pd[t]
        # ==== S3b rVPM LEV-particle feeding (research_rvpm_arch V3): one particle per supercritical
        # strip per step; strength from the A0-PIN closed form (A0 is a LINEAR functional of the rhs in
        # 'downwash' mode, and the particle enters the rhs directly -> per-strip 1x1 solve, no AIC
        # inverse; cross-strip coupling neglected = increment note); 1/3 placement toward the previous
        # (convected) particle; sigma = 1.3 x local shed spacing; circulation FROZEN after birth. ====
        if part_lev and part_mode == 'rvpm':
            assert a0_mode == 'downwash', "part_mode='rvpm' requires a0_mode='downwash'"
            from scipy.special import erf as _erf
            colm = col.numpy(); nrmm = nrm.numpy(); rrN = rings.numpy()
            sup = np.where(np.abs(A0) > a0_crit)[0]
            for j in sup:
                if np_part >= pmax:
                    break
                r0 = rrN[j, 0]; r1 = rrN[j, 1]
                le_mid = 0.5 * (r0 + r1); s_vec = r1 - r0            # LE edge (spanwise, full length)
                if lev_prev_it[j] == t - 1 and lev_prev_idx[j] >= 0:
                    prev = pp_np[lev_prev_idx[j]]
                    pos = le_mid + (prev - le_mid) / 3.0             # continuing event: 1/3 rule
                else:
                    pos = le_mid + 0.5 * vr_le[j] * dt               # new event: LE + 0.5*v_le*dt
                # sigma = 1.3 x LOCAL shedding spacing = |v_le,rel|*dt (shear-layer travel per step).
                # NOT the distance to the convected previous particle: on a PLUNGING wing the LE sweeps
                # at several m/s, that distance balloons -> fat core -> dA0 -> 0 -> pin strength diverges
                # (t=40 blow-up fingerprint, RVPM_DBG trace 2026-07-18).
                spc_loc = float(np.linalg.norm(vr_le[j])) * dt
                sgp = 1.3 * max(spc_loc, 0.25 * U * dt)
                # SPANWISE SUBDIVISION (A5-C6 Fu-Laurendeau scale consistency): one particle per 5cm
                # strip with a 2cm chordwise-derived core violates spanwise overlap (sigma/d=0.4<<1.3)
                # -> spurious axial strain -> spurious Z -> core collapse (RVPM_DBG sigma_min trace).
                # Subdivide the LE edge so sub-spacing <= shed spacing (isotropic overlap); circulation
                # split equally and FROZEN (total unchanged). Cap 8 = pmax sizing (logged, not silent).
                wlen = float(np.linalg.norm(s_vec))
                n_sub = int(np.clip(np.ceil(wlen / max(spc_loc, 1e-6)), 1, 8))
                off = pos - le_mid                                   # placement offset off the LE line
                fr = (np.arange(n_sub) + 0.5) / n_sub
                base = r0[None, :] + fr[:, None] * s_vec[None, :] + off[None, :]   # (n_sub,3)
                if np_part + n_sub > pmax:
                    break
                ii = j + ns * np.arange(nc)                          # strip-j collocation indices
                dx = colm[ii][:, None, :] - base[None, :, :]         # (nc, n_sub, 3)
                r = np.sqrt(np.sum(dx * dx, 2) + 1e-20); rb = r / sgp
                gg = _erf(rb * 0.7071067811865476) - 0.7978845608028654 * rb * np.exp(-0.5 * rb * rb)
                vv = np.sum((-0.07957747154594767 * gg / (r ** 3))[:, :, None]
                            * np.cross(dx, (s_vec / n_sub)[None, None, :]), axis=1)   # unit-gL velocity
                dW = -np.sum(vv * nrmm[ii], 1)                       # unit-strength downwash column
                dA0 = -np.sum(wq[:, j] * dW) / (np.pi * Urel_le[j] + 1e-12)
                if abs(dA0) < 1e-10:
                    continue
                gL = (np.sign(A0[j]) * a0_crit - A0[j]) / dA0        # pin post-shed A0 at +-crit
                # KELVIN-CONSERVATIVE ceiling (same physics as the production kelvin mode): the shed
                # circulation cannot exceed the supercritical-excess LE circulation. Exact thin-airfoil
                # conversion dGamma = dA0 * U_rel * c * (th1 + sin th1) — no 1.13 grid factor.
                gexc = (np.abs(A0[j]) - a0_crit) * Urel_le[j] * c_strip[j] * (th1 + np.sin(th1))
                gL = float(np.clip(gL, -gexc, gexc))
                sl = slice(np_part, np_part + n_sub)
                pp_np[sl] = base; pa_np[sl] = (gL / n_sub) * s_vec[None, :]; ps_np[sl] = sgp
                lev_prev_idx[j] = np_part + n_sub // 2               # track the MID sub-particle
                lev_prev_it[j] = t
                np_part += n_sub
            if len(sup):
                pp = wp.array(pp_np, dtype=V3, device=dev)
                pa = wp.array(pa_np, dtype=V3, device=dev)
                ps = wp.array(ps_np, dtype=DTYPE, device=dev)
            if os.environ.get("RVPM_SNAP") and t % 20 == 0 and np_part > 0:
                np.savez(f"{os.environ['RVPM_SNAP']}_t{t:04d}.npz",
                         pp=pp_np[:np_part], pa=pa_np[:np_part], ps=ps_np[:np_part],
                         rings=rings.numpy(), A0=A0)
            if os.environ.get("RVPM_DBG") and t % 10 == 0:
                gLs = float(np.sum(np.linalg.norm(pa_np[max(0, np_part - len(sup)):np_part], axis=1)))
                amx = float(np.max(np.linalg.norm(pa_np[:np_part], axis=1))) if np_part else 0.0
                smn = float(np.min(ps_np[:np_part])) if np_part else 0.0
                print(f"[rvpm t={t:3d}] np={np_part:4d} sup={len(sup):2d} |A0|max={np.abs(A0).max():.3f} "
                      f"step_sum|aL|={gLs:.3f} |a|max={amx:.3f} sig_min={smn:.4f} "
                      f"Fz={Fzb_tot[t]:+.1f}", flush=True)
        # ---- LEV shed strength per strip (placed by _shed_lev_sat_kernel at the LE, enters wake -> rhs +
        # Bernoulli surface force, so the LEV LIFT/DRAG is per-panel and NOT double-counted). Three modes. ----
        # KELVIN-CONSERVATIVE bound on the shed strength: the LEV ring we add to the wake is NOT removed from the
        # bound (the bound is re-solved each step), so an unbounded shed pumps the near-field unstable. The physical
        # ceiling is the SUPERCRITICAL EXCESS LE circulation available (A0 above a0_crit -> excess Gamma_1, Eq.6):
        exc = np.maximum(np.abs(A0) - a0_crit, 0.0)
        dG1_exc = exc * Urel_le * c_strip * (th1 + np.sin(th1)) / 1.13       # excess A0 -> excess Gamma_1 (Hirato Eq.6 inverse)
        lev_str_fp = np.zeros(ns, dtype=NP)
        if lev_shed_mode == 'varA0':       # Modulation paper Eq.11-12: rate Gamma_i = U_rel^2 * A0^2 * dt / r_LE,
            # r_LE = 1.1019*t^2 (NONDIM leading-edge radius in chords; pure geometry). The raw shear-layer rate is
            # huge vs the bound -> CAP at the Kelvin-conservative excess (dG1_exc) for stability (documented: varA0
            # and kelvin converge in this ring framework whenever supercritical; the holding mode carries the diff).
            r_LE = 1.1019 * (tc_thick ** 2)
            rate = (Urel_le ** 2) * (A0 ** 2) * dt / max(r_LE, 1e-12)
            shed = np.where(np.abs(A0) > a0_crit, np.minimum(rate, dG1_exc), 0.0)
            lev_str_fp = (-lev_klev * shed * np.sign(A0)).astype(NP)   # minus = this UVLM's gamma convention
        elif lev_shed_mode == 'kelvin':    # Hirato: shed exactly the excess LE circulation to bring A0 back to A0_crit
            lev_str_fp = (-lev_klev * dG1_exc * np.sign(A0)).astype(NP)
        elif lev_shed_mode == 'kinematic': # legacy Path-B kinematic strength (~U*c*(|A0|-crit)), for the ML anchor
            lev_str_fp = (-lev_klev * Urel_le * c_strip * exc * np.sign(A0)).astype(NP)
        elif lev_shed_mode == 'hirato':    # FAITHFUL (paper Fig.6): the LEV ring strength was SOLVED above (implicit
            # LESP=LESP_crit constraint) so the RE-SOLVED bound already holds each supercritical strip's LESP at crit.
            # Use that solved strength directly (physical sign already correct -> callers keep lev_sign=1).
            lev_str_fp = (hirato_gL if hirato_gL is not None else np.zeros(ns, dtype=NP)).astype(NP)
        if use_ansari:   # (HIRATO) update the parametric LEV sheet: convect rings aft (chordwise fraction f += U*dt/c),
            # drop those past lev_fmax (detach off the TE), and SHED a new ring at the LE (f=0) for every strip whose
            # |A0|>a0_crit (LESP supercritical), with the S3 strength. The sheet thus stays anchored at the LE.
            if len(lev_aj) > 0:
                lev_af = lev_af + (U * dt) / np.maximum(c_strip[lev_aj], 1e-9)   # chordwise convection (fraction of chord)
                keep = lev_af < lev_fmax
                lev_aj = lev_aj[keep]; lev_af = lev_af[keep]; lev_ag = lev_ag[keep]
                lev_ids = lev_ids[keep]
            js = np.where(np.abs(A0) > a0_crit)[0]                       # supercritical strips -> shed a new LEV ring at the LE
            if len(js) > 0:
                lev_aj = np.concatenate([lev_aj, js.astype(np.int64)])
                lev_af = np.concatenate([lev_af, np.zeros(len(js))])
                lev_ag = np.concatenate([lev_ag, (lev_sign * lev_str_fp[js]).astype(float)])
                lev_ids = np.concatenate([lev_ids, lev_id_next + np.arange(len(js), dtype=np.int64)])
                lev_id_next += len(js)
        if lev_impulse and use_ansari and len(lev_aj) > 0:
            # (H16 Li/Feng vortex-impulse) LEV force = -d/dt[rho * Sum_j Gamma_j * A_j], A_j = vector area of the
            # quad ring. Grid-INDEPENDENT (no surface-pressure resolution) -> the DSV lift emerges without the
            # unstable LESP-constraint fold. Geometry mirrors the L591 ansari build, now on the UPDATED sheet
            # (post-convection + post-shed). Half-wing impulse; the x2 full-wing factor is applied at cycle-mean
            # (consistent with every other channel). F_wing = -dI_fluid/dt; sign verified empirically below.
            c3v = corners.reshape(nc + 1, ns + 1, 3); nle_v = nrm.numpy()[:ns]
            LElv = c3v[0, lev_aj]; LErv = c3v[0, lev_aj + 1]
            chlv = c3v[nc, lev_aj] - LElv; chrrv = c3v[nc, lev_aj + 1] - LErv
            c_kv = 0.5 * (np.linalg.norm(chlv, axis=1) + np.linalg.norm(chrrv, axis=1)) + 1e-9
            n_kv = nle_v[lev_aj]
            fposv = 0.45 * (1.0 - np.exp(-2.2 * lev_af)); hkv = (lev_rollh * c_kv * (lev_af + 0.06))[:, None]
            f0v = fposv[:, None]; dchv = (U * dt / c_kv)[:, None]
            a0v = LElv + f0v * chlv + hkv * n_kv; a1v = LErv + f0v * chrrv + hkv * n_kv
            a2v = a1v + dchv * chrrv; a3v = a0v + dchv * chlv
            Avec = 0.5 * (np.cross(a0v, a1v) + np.cross(a1v, a2v) + np.cross(a2v, a3v) + np.cross(a3v, a0v))
            # (2026-07-11 案LEV, MATERIAL impulse accounting) the legacy ledger-difference
            # F = -(I_ledger(t)-I_ledger(t-dt))/dt mixes ring BIRTHS and DROPS into the derivative;
            # for a (statistically) periodic ledger its cycle mean is IDENTICALLY ZERO — measured:
            # T_vimp = 0.000 at every twist, std 0.36N, 96% steps nonzero (periodic-derivative null).
            # Physical bookkeeping (Li/Feng impulse theory; Otomo 2021 vortex-impulse practice):
            #   survivors: -rho*Gamma*(A_t - A_{t-dt})/dt   (deformation/rotation of the SAME ring)
            #   births:    -rho*Gamma*A_born/dt             (LE vorticity creation reacts on the wing)
            #   drops:     EXCLUDED                         (impulse advects away with the fluid)
            GA = lev_ag[:, None] * Avec                                    # per-ring Gamma*Avec (n,3)
            dI = np.zeros(3)
            if imp_prev is not None:
                ids_p, GA_p = imp_prev
                common, ia, ib = np.intersect1d(lev_ids, ids_p, return_indices=True)
                dI += (GA[ia] - GA_p[ib]).sum(axis=0)                      # survivors: material change
                born = ~np.isin(lev_ids, common)
                dI += GA[born].sum(axis=0)                                 # births: created from zero
                Fvi = -ug.RHO * dI / max(dt, 1e-15)
                Lh_vimp[t] = float(Fvi[2]); Xh_vimp[t] = float(Fvi[0])
                if not vnf_kelvin:
                    # (L2 2026-07-17) single LEV-force accounting: with vnf_kelvin the DSV force is closed
                    # by the vnf excess normal force (Polhamus/hirato closure) — the impulse ledger provably
                    # under-delivers the cycle-mean DSV lift (vimp ~ -0.2..-0.7N, anti-phased; gap_l2 case).
                    # Channel stays as DIAGNOSTIC only. Closure selection, not stacking (no double count).
                    Fzb_tot[t] += Lh_vimp[t]; Fxb_tot[t] += Xh_vimp[t]
            imp_prev = (lev_ids.copy(), GA.copy())
        if part_lev and np_part > 0:   # rVPM LEV force via QUASI-STEADY KUTTA-JOUKOWSKI on the OVER-WING LEV.
            # The full vortex-impulse sum(x x alpha) is WILD because it accumulates ALL shed particles -> the
            # far-wake convection term rho*U*sum(alpha) grows unbounded. The physical LEV lift is the KJ of the
            # circulation CURRENTLY OVER THE WING (the coherent LEV near the forward suction surface): as the LEV
            # builds it grows, as it convects off the TE it drops (the rise-peak-drop). Count only particles within
            # a chord-window of the LE; their spanwise circulation alpha_y gives L = rho * U * sum(alpha_y).
            le_ref = np.mean(0.5 * (cc[:ns, 0] + cc[:ns, 1]), axis=0)    # current wing-LE centroid (moves with flap)
            pph = pp.numpy()[:np_part]; pah = pa.numpy()[:np_part]
            chdir = np.mean(tc[:ns], axis=0); chdir = chdir / (np.linalg.norm(chdir) + 1e-9)   # WING-CHORD dir (LE->TE)
            dchord = (pph - le_ref) @ chdir                             # chordwise distance along the (tilted) chord
            cbar = float(np.mean(0.5 * (np.linalg.norm(cc[:ns, 2] - cc[:ns, 0], axis=1) +
                                        np.linalg.norm(cc[:ns, 3] - cc[:ns, 1], axis=1))))   # mean chord
            ow = (dchord > -0.2 * cbar) & (dchord < lev_owin * cbar)    # OVER-WING window (LE .. owin*chord aft)
            Urel = abs(float(Vinf[0]))
            Lh_imp[t] += ug.RHO * Urel * float(np.sum(pah[ow, 1]))      # KJ lift of the over-wing LEV (spanwise circ)
            Xh_imp[t] += ug.RHO * Urel * float(np.sum(pah[ow, 2]))      # chordwise circ -> streamwise force
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
            Fzb_tot[t] += float(np.sum(Df[:, 2])); Fxb_tot[t] += float(np.sum(Df[:, 0]))   # friction force vector
        # ==== Fix2: FLAP-VELOCITY^2 friction drag (the inviscid potential flow has NO viscous tractions -> over-
        # predicts net thrust, growing ~f^2 and twist-independent). Reuses the visc structure but with a TURBULENT
        # flat-plate Cf (laminar Blasius ~0.15N is too small for the observed ~1N@2.6Hz). V_tan is dominated by the
        # flap plunge (fixed +-45deg) -> drag ∝ V_tan^2 ∝ V_flap^2 ∝ f^2; Cf is alpha-independent -> twist-independent. ====
        if fric_drag:
            nnf = nrm.numpy()
            Vtanf = Vcol - (np.sum(Vcol * nnf, axis=1)[:, None]) * nnf
            Vtmf = np.linalg.norm(Vtanf, axis=1) + 1e-12
            c_locf = np.broadcast_to(tcn.reshape(nc, ns).sum(0), (nc, ns)).reshape(-1)
            Re_f = np.maximum(Vtmf * c_locf / NU_AIR, 1.0e2)
            Cf_f = (1.328 / np.sqrt(Re_f)) if cf_mode == 'laminar' else (0.074 / Re_f ** 0.2)  # turbulent flat plate
            Cdf_f = 2.0 * Cf_f * FORM_FF
            Dff = 0.5 * ug.RHO * Cdf_f[:, None] * area[:, None] * Vtmf[:, None] * Vtanf
            Lh_fric[t] = float(np.sum(Dff[:, 2])); Xh_fric[t] = float(np.sum(Dff[:, 0]))
            Fzb_tot[t] += float(np.sum(Dff[:, 2])); Fxb_tot[t] += float(np.sum(Dff[:, 0]))   # friction drag vector
        # ---- SEPARATED-FLOW FORM/PRESSURE DRAG (viscous-origin). Blasius friction (above) is only ~0.15N;
        # the BIG viscous drag is the pressure drag from boundary-layer SEPARATION at high alpha_eff (the
        # +-45 flap strokes reach alpha_eff~45deg). Cd_form = cd_form*sin^2(alpha_eff) (flat-plate-separated,
        # ~0 attached -> grows when stalled), drag along the relative wind -> the missing thrust-axis drag. ----
        if prof_drag and not kirch_blend:   # (P1) skip when kirch_blend: the flat-plate CN already in the blend
            nnq = nrm.numpy(); vr = np.asarray(Vinf) - vcn       # relative wind (freestream + flapping)
            vrm = np.linalg.norm(vr, axis=1) + 1e-9
            sap = np.sum(vr * nnq, axis=1) / vrm                 # sin(alpha_eff) per panel
            # CROSS-FLOW (Hoerner) separated drag: Cd = Cd_max*sin^2(a_eff), Cd_max=cd_form~1.98 (flat-plate
            # slope), PLATEAUED at the deep-stall value cd_dp (~1.2, airfoil deep-stall Cd / Hoerner). The bluff-
            # body Cd saturates past full separation (does NOT climb to the 2.0 broadside value) -> form drag
            # stays ~f^2 at high frequency (matches measured net-thrust trend). First-principles (no RoboEagle fit).
            Cdp = np.minimum(cd_form * sap ** 2, cd_dp)         # cross-flow form drag, deep-stall plateau cd_dp
            if fsep_le is not None:
                # (H19) gate the SEPARATED drag by the (lagged) separated fraction (1-fsep): in attached flow the
                # Hoerner crossflow drag is zero (only profile friction, handled by `visc`); it switches ON as the
                # boundary layer separates. With fsep_lag the gate is delayed through the brief mid-stroke alpha peak
                # -> the static-Hoerner overshoot (H18 tw45 -11.8N) is removed, matching the measured dynamic-stall
                # delay. Per-strip fsep tiled to per-panel (panel p uses strip j=p%ns).
                Cdp = Cdp * np.tile(1.0 - fsep_le, nc)
            Dp = 0.5 * ug.RHO * vrm[:, None] * Cdp[:, None] * area[:, None] * vr   # along relative wind
            Lh_pd[t] = float(np.sum(Dp[:, 2])); Xh_pd[t] = float(np.sum(Dp[:, 0]))
            Fzb_tot[t] += float(np.sum(Dp[:, 2])); Fxb_tot[t] += float(np.sum(Dp[:, 0]))   # form-drag force vector
        # ---- LEADING-EDGE SUCTION thrust (Garrick / DeLaurier dTs = 2pi eta_s alpha_eff^2 (1/2 rho U V) c dy).
        # A flat-panel normal-pressure (Bernoulli) force structurally MISSES the leading-edge singular suction
        # (the sqrt(x) edge force) -> captures induced drag but NOT the forward LE-suction thrust. This is the
        # dominant flapping-thrust mechanism ("thrust is all leading-edge suction"), forward along -chord,
        # applied on ATTACHED strips (LEV not shed; shed-strip suction goes into the LEV captured by Bernoulli). ----
        if les_suction:
            nn2 = nrm.numpy(); iLE = np.arange(ns)            # leading-edge panel row (i=0)
            # C4: kinematic LE velocity (matches the LESP-constraint Ur and Hirato Eq.20); default keeps Vcol (legacy)
            Vle = ((np.asarray(Vinf) - vcn) if les_kin else Vcol)[iLE]; nle = nn2[iLE]; tcle = tc[iLE]
            Vle_m = np.linalg.norm(Vle, axis=1) + 1e-12
            sa = np.sum(Vle * nle, axis=1) / Vle_m            # sin(alpha_eff) at the LE strip
            aeff = np.arcsin(np.clip(sa, -0.999, 0.999))
            # LESP criterion (Ramesh 2014): realizable LE suction CAPS at the critical leading-edge angle -
            # beyond alpha_crit the excess loading sheds into the LEV (already shed for lift), so the attached
            # LE-suction saturates. First-principles separation onset, NOT an empirical efficiency fit.
            a_crit = np.radians(lesp_crit_deg)
            aeff_s = np.clip(aeff, -a_crit, a_crit)            # LESP saturation: A0 caps at A0_crit when LEV sheds
            c_le = tcn.reshape(nc, ns).sum(0)                 # local chord per strip
            dy_le = tsn.reshape(nc, ns)[0]                    # strip spanwise width (LE row)
            # GARRICK LE-suction thrust F_A = rho*pi*c * U_rel^2 * A0^2 (Ramesh/Gordillo). A0 = sin(alpha_eff)
            # CAPS at sin(alpha_crit) (LESP), but the LOCAL dynamic pressure ~Vle_m^2 still grows with flap speed
            # -> F_A ~ Vle_m^2 ~ f^2 even when saturated. Using Vle_m^2 (both factors LOCAL) is the correct
            # quadratic scaling; the earlier rho*U_inf*Vle_m mix gave only ~f^1. This is the f^2 propulsion.
            sa_s = np.sin(aeff_s)                              # saturated LE-suction parameter A0 = sin(a_crit) max
            sup_le = np.abs(aeff) > a_crit                     # separated-strip detector (geometric fallback)
            sgn_le = np.sign(sa)
            if lev_shed_mode in ('kelvin', 'varA0', 'kinematic', 'hirato'):
                # S4 (Hirato Eq.20): realized LE suction caps at the FIRST-PRINCIPLES A0 (from bound circulation),
                # bounded by a0_crit; the EXCESS above a0_crit is what S3 sheds into the LEV (no double-count).
                # (C7 les_pre, 2026-07-04) under 'hirato' the POST-solve A0 collapses SUBCRITICAL after heavy shedding
                # (prior-step rings over-suppress: measured effective |A0|~0.17 vs crit 0.27 at 10/5/2.6/tw45 ->
                # les_T +10.7 vs kelvin-path +19.8) — a DISCRETE-ring artifact; in sustained shedding the physical
                # LESP hovers AT crit (Hirato Fig.6 re-converges every step). les_pre keys the realized suction on
                # the PRE-constraint A0pre — the SAME quantity that sizes the vnf excess and the faure gate (one
                # consistent closure, no new constants): suction = crit-capped attached-flow value while shedding.
                A0f = hirato_A0pre if (les_pre and lev_shed_mode == 'hirato' and hirato_A0pre is not None) else A0
                # (T1 DIAG 2026-07-14) les_free: counterfactual UNCAPPED suction (A0 not clipped at a0_crit in the
                # SUCTION channel only; shedding gate untouched) — quantifies the static-ceiling share of the
                # missing thrust f-growth (gap_t1_thrust_growth.md §2). Diagnostic, not production.
                sa_s = A0f if les_free else np.clip(A0f, -a0_crit, a0_crit)
                sup_le = np.abs(A0f) > a0_crit                 # separated = LESP supercritical (the ansari shed gate)
                sgn_le = np.sign(A0f)
            dTs = np.pi * les_eta * ug.RHO * c_le * dy_le * (Vle_m ** 2) * (sa_s ** 2)
            if les_sep == 'plateau_fn':
                # (R2 2026-07-15, research_lev_closure.md) VORTEX-FORMATION-NUMBER gate — the published
                # transition between the two suction schools: while a strip's LESP is supercritical the LEV
                # feeds and the chordwise suction HOLDS at the critical plateau (Katz-1981 / standard Ramesh
                # LDVM) — but only within the feeding window T_hat = ∫u_LE dt / c < T* (optimal vortex
                # formation number, Gharib lineage; flapping-wing LEV: Onoue & Breuer 3.7±0.3 -> T*=4.0).
                # Past T* the vortex pinches off and the chordwise suction COLLAPSES (Narsipur 2020).
                # Reset when the strip falls subcritical (feeding stops -> reattachment; A0* hysteresis is
                # the optional R3 refinement). Inert at moderate feathering (window never reached within a
                # half-stroke), auto-collapse at deep feathering. Zero fitted constants.
                if fn_state is None:
                    fn_state = np.zeros(ns)
                fn_state = np.where(sup_le, fn_state + Vle_m * dt / np.maximum(c_le, 1e-9), 0.0)
                dTs = np.where(fn_state >= fn_Tstar, 0.0, dTs)
            elif les_sep != 'plateau':
                # (2026-07-09 GAP-f2 fix) the LDVM "suction held at crit while shedding" is the unproven Katz-1981
                # postulate; viscously the LE suction COLLAPSES when the LE separates (Narsipur 2020 JFM 900 A25).
                # 'polhamus': the retained (crit-capped) magnitude rotates onto the panel normal -> vortex force
                # (continuous in |F|, chordwise thrust -> 0); 'zero': hard collapse (Narsipur ablation).
                # 'kt' (2026-07-10 案B step-3): Carlson NASA TP-1500 ATTAINABLE-thrust split — the separated-strip
                #   chordwise suction keeps the attainable fraction K_T (closed form, eqs 8-10: local Mach from
                #   Vle, strip Reynolds, geometry tau/c + r/c; NO free constants) and the unattainable (1-K_T)
                #   rotates to the panel normal (same vortex-force bookkeeping as 'polhamus'). K_T -> 1 attached,
                #   ~0.2-0.5 at deep flapping incidence. EXTRAPOLATION NOTE: our section (tau/c=0.028, r/tau=0.5)
                #   sits outside the TP-1500 correlation range (tau 6-18%c, r/tau 2-16%) — documented, not tuned.
                if les_sep == 'kt':
                    Mn_s = np.clip(Vle_m / 340.0, 1e-3, 0.5)
                    Rn_s = np.maximum(Vle_m * c_le / 15.06e-6, 1.0)
                    expn = 0.05 + 0.35 * (1.0 - Mn_s) ** 2
                    cplim = (-2.0 / (1.4 * Mn_s ** 2)
                             * (Rn_s * 1e-6 / (Rn_s * 1e-6 + 10.0 ** (4.0 - 3.0 * Mn_s))) ** expn)
                    g_ = 1.4 * np.abs(cplim) * np.sqrt(np.maximum(1.0 - Mn_s ** 2, 1e-6))
                    Me_s = (np.sqrt(2.0) / g_) * np.sqrt(np.sqrt(1.0 + g_ ** 2) - 1.0)
                    ct_th = 2.0 * np.pi * np.maximum(np.sin(aeff) ** 2, 1e-6)      # theoretical 2D ct at local incidence
                    KT = np.minimum(1.0, (2.0 * (1.0 - Me_s ** 2) / np.maximum(Me_s, 1e-6))
                                    * ((0.028 * 0.0139 ** 0.4) / ct_th) ** 0.6)     # tau/c, r/c: section geometry
                    keep = np.where(sup_le, KT, 1.0)                               # split only on separated strips
                    dNp = dTs * (1.0 - keep)
                    Fp = (dNp * sgn_le)[:, None] * nle
                    Lh_vtx[t] += float(np.sum(Fp[:, 2])); Xh_vtx[t] += float(np.sum(Fp[:, 0]))
                    Fzb_tot[t] += float(np.sum(Fp[:, 2])); Fxb_tot[t] += float(np.sum(Fp[:, 0]))
                    dTs = dTs * keep
                else:
                    if les_sep == 'polhamus':
                        dNp = dTs * sup_le
                        Fp = (dNp * sgn_le)[:, None] * nle
                        Lh_vtx[t] += float(np.sum(Fp[:, 2])); Xh_vtx[t] += float(np.sum(Fp[:, 0]))
                        Fzb_tot[t] += float(np.sum(Fp[:, 2])); Fxb_tot[t] += float(np.sum(Fp[:, 0]))
                    dTs = dTs * (~sup_le)
            if les_att and fsep_le is not None:
                # (H10) realized suction only on the ATTACHED fraction: fsep->0 at deep stall kills the fictional
                # mid-stroke suction pulses (Fig16: measured thrust has none); the lost suction is the vnf's job.
                dTs = dTs * fsep_le
            Fs = -dTs[:, None] * tcle                         # forward (-chordwise) suction force vector
            Lh_les[t] = float(np.sum(Fs[:, 2])); Xh_les[t] = float(np.sum(Fs[:, 0]))
            if lev_vnf and (lev_shed_mode == 'hirato' or vnf_kelvin):
                # (HIRATO force closure) the LE suction capped at a0_crit -> the UNREALIZABLE excess (A0^2 - a0_crit^2)
                # reappears as a force NORMAL to the wing (the LEV vortex lift). Same Garrick coefficient pi*rho*c*U^2
                # as the suction, so the split is continuous at A0=a0_crit (no discontinuity). This is the vortex lift
                # the shed LEV sheet delivers through the surface pressure (Eq.13 vL term); the discrete-ring induction
                # captures it only weakly, so it is closed here first-principles from the constraint's A0 (no free coeff).
                # excess from the PRE-constraint LESP (A0pre); the post-solve A0 has already been pulled to a0_crit by
                # the constraint, so A0_post^2 - a0_crit^2 == 0. A0pre^2 - a0_crit^2 is the physical unrealized suction.
                A0x = hirato_A0pre if hirato_A0pre is not None else A0
                exc2 = np.maximum(A0x ** 2 - a0_crit ** 2, 0.0)                       # capped (excess) suction parameter^2
                if vnf_kelvin and les_sep == 'plateau_fn' and fn_state is not None:
                    # (L2 iter-2) the vnf normal force follows the VORTEX LIFECYCLE: active only inside the
                    # feeding window T_hat < T* (same formation-number gate as the chordwise suction, R2);
                    # past pinch-off the DSV convects away and its wing-normal force collapses with it.
                    # Fixes the uniform mid-aoa over-supply of the ungated variant (battery 2026-07-17).
                    exc2 = exc2 * (fn_state < fn_Tstar)
                # POLHAMUS / FLAT-PLATE SATURATION (no free coeff): A0pre comes from the LEV-free VIRTUAL solve — a
                # potential-flow fiction at deep stall (can reach 2-3 at large twist+flap) whose square EXPLODES the
                # rotated force. The realizable separated-flow normal force saturates at the Polhamus finite-wing
                # vortex-force limit: coef_pol = 1/2*k_v*sin^2(a_eff)*|cos(a_eff)|, k_v = 2*pi/(1+2/AR) (NASA TN
                # D-4739, analytic from geometry). min() keeps the pi-branch (continuous at A0=crit, ->0 there) for
                # small excess and caps at the Polhamus branch in deep stall (sin^2*cos <= 0.385 @54.7deg).
                S_half = float(np.sum(area)) + 1e-12                                  # actual half-wing area (this step)
                AR_w_v = (2.0 * half_span) ** 2 / (2.0 * S_half)                      # geometric AR (both wings)
                k_v_pol = 2.0 * np.pi / (1.0 + 2.0 / max(AR_w_v, 1e-6))
                coef_pi = np.pi * exc2
                coef_pol = 0.5 * k_v_pol * (np.sin(aeff) ** 2) * np.abs(np.cos(aeff))
                coef_v = np.minimum(coef_pi, coef_pol) if vnf_sat else coef_pi
                dNr = les_eta * ug.RHO * c_le * dy_le * (Vle_m ** 2) * coef_v
                Frec = (dNr * np.sign(A0x))[:, None] * nle                            # along the panel normal (vortex lift)
                Lh_vtx[t] += float(np.sum(Frec[:, 2])); Xh_vtx[t] += float(np.sum(Frec[:, 0]))
                Fzb_tot[t] += float(np.sum(Frec[:, 2])); Fxb_tot[t] += float(np.sum(Frec[:, 0]))
            Fzb_tot[t] += float(np.sum(Fs[:, 2])); Fxb_tot[t] += float(np.sum(Fs[:, 0]))   # LE-suction force vector
            if fp_lev:
                # ---- FIRST-PRINCIPLES held-LEV / dynamic-stall lift (NO fitted klev). The LE suction realizable
                # only up to A0_crit (LESP, Ramesh 2014); the EXCESS sheds into the leading-edge vortex. By the
                # Polhamus leading-edge-suction analogy the lost suction is conserved as a NORMAL force (vortex
                # lift): dN = K_v*(A0^2 - A0_crit^2)*1/2 rho V_le^2 c dy, applied along the panel normal (-> lift
                # AND drag projections automatically). K_v = 2pi/(1+2/AR) = finite-wing potential LE-suction
                # factor (NASA TN D-4739), ANALYTIC -- no fit. A0 = sin(alpha_eff) from the LE kinematics, so the
                # LEV scales with design (AR via K_v) and kinematics (A0) -> generalizes for co-design. LEV
                # detaches (sheds) past lev_detach_deg: cap A0 there so the recovered suction stops growing. ----
                A0c = np.sin(a_crit)
                exc = np.maximum(np.abs(sa) ** 2 - A0c ** 2, 0.0)  # recovered (excess) LE suction (RISE w/ a_eff)
                # DETACHMENT (rise-peak-DROP): past the dynamic-stall angle the LEV sheds off the TE and its lift
                # collapses. Smooth taper f_det: 1 below a_det (LEV over the wing), ->0 by a_det+trans (shed).
                # a_det = dynamic-stall angle (airfoil/Re property), NOT a RoboEagle fit. This single mechanism
                # gives BOTH the high-AoA roll-off AND the lift-vs-twist peak-then-fall.
                a_det = np.radians(lev_detach_deg); a_tr = np.radians(lev_trans_deg)
                f_det = 0.5 * (1.0 + np.cos(np.pi * np.clip((np.abs(aeff) - a_det) / max(a_tr, 1e-6), 0.0, 1.0)))
                dNv = lev_kv * exc * f_det * 0.5 * ug.RHO * (Vle_m ** 2) * c_le * dy_le * np.sign(sa)
                Flev = dNv[:, None] * nle                          # vortex normal force -> lift (N.z) + drag (N.x)
                Lh_vtx[t] = float(np.sum(Flev[:, 2])); Xh_vtx[t] = float(np.sum(Flev[:, 0]))
                Fzb_tot[t] += float(np.sum(Flev[:, 2])); Fxb_tot[t] += float(np.sum(Flev[:, 0]))
        # ---- S6 ATTACHED VISCOUS DRAG (Faure 2023): static sectional profile drag C_D(alpha_rel) along the
        # relative wind, applied where the flow is ATTACHED. C_D = cd0_polar*(1 + (alpha_rel/a_ref)^2) (airfoil
        # profile-drag bucket, Re~1e5), alpha_rel = arctan(w_n/U) per panel. When the LEV sheds (|A0|>a0_crit)
        # the drag is borne by the vortex (Bernoulli) -> gate OFF on separated strips. Replaces visc/prof_drag. ----
        if attached_drag in ('faure', 'uiuc'):
            nnf = nrm.numpy(); vrf = np.asarray(Vinf) - vcn          # body-relative flow per panel
            vrm = np.linalg.norm(vrf, axis=1) + 1e-9
            arel = np.abs(np.arcsin(np.clip(np.sum(vrf * nnf, axis=1) / vrm, -0.999, 0.999)))   # |alpha_rel|
            if attached_drag == 'uiuc':
                # (u-trend ② 2026-07-17, research_utrend.md §4.2) MEASURED sectional polar CD(alpha,Re):
                # UIUC LSAT Vol.1 SD7003 wind-tunnel drag table (researchpaper/uiuc_polars/), bilinear in
                # (alpha, logRe), CLAMPED at the table edge (no post-stall extrapolation — honest wall).
                # Replaces the generic cd0_polar parabola in the SAME bookkeeping slot (attached strips,
                # along relative wind, LESP-gated) -> adds the Re-slope the constant polar lacks. Zero fit.
                global _CDT
                try:
                    _CDT
                except NameError:
                    from cd_table import CdTableP as CdTable
                    _CDT = CdTable()
                c_pan_d = np.tile(tcn.reshape(nc, ns).sum(0), nc)
                re_pan = vrm * c_pan_d / 15.06e-6
                Cd_att = _CDT(np.degrees(arel), re_pan)
            else:
                a_ref = np.radians(12.0)                             # profile-drag bucket half-width (airfoil property)
                Cd_att = cd0_polar * (1.0 + (arel / a_ref) ** 2)     # static profile-drag polar
            att = np.ones(npan)                                       # gate: attached strips only (LEV not shed)
            if lev_shed_mode in ('kelvin', 'varA0', 'kinematic', 'hirato'):
                # ATTACHED gate must use the PRE-constraint LESP under 'hirato': the implicit constraint pins the
                # post-solve A0 EXACTLY at a0_crit on shedding strips, so |A0|<=crit would pass SEPARATED strips
                # (whose alpha_rel is huge on the flap stroke -> the quadratic bucket explodes into fake drag).
                # |A0pre|<crit (strict) = the strip is genuinely attached this step.
                A0_gate = hirato_A0pre if (faure_gate_pre and lev_shed_mode == 'hirato' and hirato_A0pre is not None) else A0
                att = (np.abs(A0_gate) < a0_crit) if faure_gate_pre else (np.abs(A0_gate) <= a0_crit)
                att = np.tile(att.astype(NP), nc)                # panel p uses strip j=p%ns gate (attached only)
            if sep_drag:
                # (u-trend ② 补漏 2026-07-17) SEPARATED-strip drag bookkeeping: the att gate switches the
                # attached polar OFF on LESP-supercritical strips under "drag borne by the vortex" — but no
                # vortex channel books cycle-mean drag (audited hole). Book the separated-state flat-plate
                # drag CD90_AR*sin^2(alpha_rel) there (Hoerner 2D 1.98 x finite-AR -> 1.20 @AR6, verified
                # research_caseA_redirect.md R1; DRAG-only along relative wind — the lift-redirection
                # variant kirch_blend failed and is NOT used). Dynamic share carries the U-trend
                # (longer/deeper feeding windows at low U). Zero fitted constants.
                cd_sep = 1.20 * np.sin(arel) ** 2
                Cd_att = Cd_att * att + cd_sep * (1.0 - att)
                att = np.ones(npan)
            Dfa = 0.5 * ug.RHO * vrm[:, None] * Cd_att[:, None] * area[:, None] * att[:, None] * vrf  # along rel. wind
            Lh_pd[t] += float(np.sum(Dfa[:, 2])); Xh_pd[t] += float(np.sum(Dfa[:, 0]))
            Fzb_tot[t] += float(np.sum(Dfa[:, 2])); Fxb_tot[t] += float(np.sum(Dfa[:, 0]))
            if os.environ.get("UTREND_DBG"):
                # (T1b 病灶#2) per-step per-strip phase diagnostic. Two separation-drag potentials:
                # (a) LESP-att-gated (what the current att硬门 zeros); (b) KIRCHHOFF-gated via the
                # continuous separation fraction (1-fsep) on alpha_rel — what T2's sep_drag_gk uses.
                # The two阈值 are NOT aligned (Explore audit): LESP att_frac~0.9-1.0 vs Kirchhoff sees
                # the high-alpha_rel strips as partly separated. (b) is the true relaxation-share test.
                arel_s = arel.reshape(nc, ns).mean(0)
                cd_att_p = 1.20 * np.sin(arel) ** 2                        # Hoerner sep-drag coeff/panel
                x_lesp = 0.5 * ug.RHO * vrm * cd_att_p * (1.0 - att) * area * vrf[:, 0]  # LESP-gated
                ass_k = np.radians(geo_stall_deg); wid_k = np.radians(max(geo_stall_width, 1e-6))
                sepf = np.clip((arel - ass_k) / wid_k, 0.0, 1.0)          # Kirchhoff separated fraction
                x_kir = 0.5 * ug.RHO * vrm * cd_att_p * sepf * area * vrf[:, 0]          # Kirchhoff-gated
                _UT_LOG.append((t, float(np.degrees(np.mean(arel_s))),
                                float(np.sum(x_lesp)), float(np.sum(x_kir))))
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
            # Polhamus rotated-normal force, gated to past separation onset (|sin a_eff| > sin a_crit). NOTE: its
            # STREAMWISE projection was measured ~0 (fore/aft panel normals cancel) -> NOT the thrust source; the
            # f^2 thrust is the Garrick LE-suction (les_suction) above. Kept only for the high-alpha lift bump.
            sep = (np.abs(sa_v) > np.sin(np.radians(lesp_crit_deg))).astype(NP)
            Nv = k_vortex * sa_v * np.abs(sa_v) * ca_v * qd * area * sep   # sin^2(a) cos(a) normal force, separated
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
                lev_rings=lev_frame_rings.copy(), lev_g=lev_frame_g.copy(),   # (HIRATO) LEV sheet over the suction surface
                dp=dp.copy(), cp=(dp / max(q_ref, 1e-9)).copy(),             # per-panel unsteady-Bernoulli pressure jump + Cp
                sep=(np.abs(sina) > np.sin(np.radians(lesp_crit_deg))), nc=nc, ns=ns))
        if te_traj:   # shed along the TE trajectory (continuous sheet for the plunging TE)
            wp.launch(_shed_te_traj, dim=ns, inputs=[rings, gamma, te, tpl, tpr, Vw, DTYPE(dt), nw],
                      outputs=[wr, wg, tcl, tcr], device=dev)
            wp.copy(tpl, tcl); wp.copy(tpr, tcr)        # current TE becomes next step's "previous"
        else:
            wp.launch(ug.shed_kernel, dim=ns, inputs=[rings, gamma, te, Vw, DTYPE(dt), nw], outputs=[wr, wg], device=dev)
        if fp_shed:   # FIRST-PRINCIPLES LESP-LEV with S3 strength lev_str_fp (varA0/kelvin/kinematic).
            lev_str_w = wp.array((lev_sign * lev_str_fp).astype(NP), dtype=DTYPE, device=dev)
            if lev_shed_mode == 'hirato' and lev_in_wake and hirato_wr is not None:   # wake path: shed Eq.7 rings into
                # (geometry already built in the constraint block, strength = the LESP-constrained gL). Write the ns
                # LEV rings into the wake at [nw+ns .. nw+2ns); track lpl/lpr = the aft (shared) edge for next step's
                # Eq.7 placement so consecutive rings connect and the sheet stays compressed near the LE.
                hw = wp.array(hirato_wr, dtype=V3, device=dev)
                hg = wp.array((hirato_gL if hirato_gL is not None else np.zeros(ns, dtype=NP)).astype(NP),
                              dtype=DTYPE, device=dev)
                wp.launch(_place_rings_kernel, dim=ns, inputs=[hw, hg, nw + ns], outputs=[wr, wg], device=dev)
                lpl = wp.array(np.ascontiguousarray(hirato_wr[:, 3]), dtype=V3, device=dev)   # aft-left  (shared edge)
                lpr = wp.array(np.ascontiguousarray(hirato_wr[:, 2]), dtype=V3, device=dev)   # aft-right (shared edge)
                lev_first = 0
            elif use_lev_sheet:   # (E2) CONNECTED LEV sheet from the geometric LE (offset onto suction side) -> rolls up
                nle_np = nrm.numpy()[:ns]                              # LE-row panel normals (suction-side direction)
                corners3 = corners.reshape(nc + 1, ns + 1, 3)
                off = lev_le_off * chord
                le0 = corners3[0, :ns] + nle_np * off                  # left geometric LE corner per strip (offset)
                le1 = corners3[0, 1:ns + 1] + nle_np * off             # right
                # (FINE) subdivide each strip's LE edge into nsub sub-rings -> nls = ns*nsub LEV elements/step,
                # each carrying 1/nsub of the strip's LEV circulation. Refines the LEV sheet independent of the wing grid.
                frac = np.linspace(0.0, 1.0, nsub + 1)
                subL = (le0[:, None, :] + frac[None, :nsub, None] * (le1 - le0)[:, None, :]).reshape(nls, 3).astype(NP)
                subR = (le0[:, None, :] + frac[None, 1:, None] * (le1 - le0)[:, None, :]).reshape(nls, 3).astype(NP)
                # (HIRATO Eq.7 / Ansari) place the new LEV ring's shed edge at x_L = 2/3*x_LE + 1/3*x_{L,prev}
                # (between the geometric LE and the last-shed LEV ring), over the suction surface. lpl/lpr hold
                # the previous step's LE-shed corners. First row (lev_first) -> geometric LE (no previous).
                if lev_first == 0:
                    subL = (2.0 / 3.0) * subL + (1.0 / 3.0) * lpl.numpy()
                    subR = (2.0 / 3.0) * subR + (1.0 / 3.0) * lpr.numpy()
                substr = (np.repeat(lev_sign * lev_str_fp, nsub) / nsub).astype(NP)
                lel_w = wp.array(subL, dtype=V3, device=dev); ler_w = wp.array(subR, dtype=V3, device=dev)
                lev_str_w = wp.array(substr, dtype=DTYPE, device=dev)
                wp.launch(_shed_lev_traj, dim=nls, inputs=[lel_w, ler_w, lpl, lpr, lev_str_w, Vw, DTYPE(dt),
                          nw + ns, lev_first], outputs=[wr, wg, lcl, lcr], device=dev)
                wp.copy(lpl, lcl); wp.copy(lpr, lcr); lev_first = 0    # current LE-shed -> next step's previous
            else:               # legacy fixed-offset ring (does NOT roll up)
                wp.launch(_shed_lev_sat_kernel, dim=ns, inputs=[rings, nrm, ns, nw + ns, lev_str_w],
                          outputs=[wr, wg], device=dev)
        if lev_merge:   # N-LEV MERGING: ONE coherent LEV core per strip (no wake shedding), LESP-saturated
            nns = nrm.numpy(); cc_le = rings.numpy(); vrl = (np.asarray(Vinf) - vcn)[:ns]; nl = nns[:ns]
            vrl_m = np.linalg.norm(vrl, axis=1) + 1e-9
            sa_l = np.sum(vrl * nl, axis=1) / vrl_m                  # sin(alpha_eff) per strip
            cst = tcn.reshape(nc, ns).sum(0); scr = np.sin(np.radians(lesp_crit_deg))
            # Position the core OVER the suction surface (strip mid-chord, 0.10c above), FIXED relative to the
            # wing -> the induction depends only on the (capped) circulation, NOT on a convecting position
            # (which made the increment grow with frequency). The dynamic-stall LEV sits on the suction side.
            te_idx = (nc - 1) * ns + np.arange(ns)
            le_mid = 0.5 * (cc_le[:ns, 0] + cc_le[:ns, 1]); te_mid = 0.5 * (cc_le[te_idx, 2] + cc_le[te_idx, 3])
            lev_cen = 0.5 * (le_mid + te_mid) + nl * (0.10 * cst)[:, None]
            # LESP SATURATION: relax the core circulation toward +/-cap when STALLED (|sin a|>sin_crit), toward
            # 0 when attached. cap = klev*U*c*sin_crit is the Garrick LE-suction at A0_crit -> FREQUENCY-
            # INDEPENDENT increment (~pi rho U^2 c A0_crit^2), matching the measured ~constant ~3N. lev_tau =
            # build/shed time. The core is at the cap WHENEVER stalled (not proportional to the excess).
            cap = lev_klev * U * cst * scr
            # LEV active only for sin_crit < |alpha_eff| < sin(detach): below crit = attached (no LEV); ABOVE
            # detach = FULL stall, the LEV detaches/sheds and its lift is LOST (the measured needed-LEV peaks
            # ~10deg then DROPS at 15deg). Cruise (5deg, alpha_eff<=~45deg) is below detach -> unaffected.
            sdet = np.sin(np.radians(lev_detach_deg))
            active = (np.abs(sa_l) > scr) & (np.abs(sa_l) < sdet)
            target = np.where(active, -cap * np.sign(sa_l), 0.0)              # sign: LEV core ADDS lift (overshoot)
            # 2nd-order lag = CONVECTION DELAY: the stall feeds lev_gam_raw, whose response lev_gam LAGS ->
            # the LEV lift peaks AFTER the attached peak (the vortex convects over the chord) = the plateau,
            # not a boost of the instantaneous peak.
            lev_gam_raw = lev_gam_raw + (target - lev_gam_raw) * (dt / lev_tau)
            lev_gam = lev_gam + (lev_gam_raw - lev_gam) * (dt / lev_tau)
            swe = cc_le[:ns, 1] - cc_le[:ns, 0]                     # spanwise edge (the LEV vortex carrier)
            dn = (np.asarray(Vinf) / (np.linalg.norm(Vinf) + 1e-9)) * (10.0 * float(cst.mean()))   # horseshoe return
            lr = np.zeros((ns, 4, 3))
            lr[:, 0] = lev_cen - 0.5 * swe; lr[:, 1] = lev_cen + 0.5 * swe; lr[:, 2] = lr[:, 1] + dn; lr[:, 3] = lr[:, 0] + dn
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
        if part_lev and part_mode != 'rvpm':   # legacy PARTICLE LEV shed (kinematic strength, sin-crit gate)
            wp.launch(shed_lev_particles_kernel, dim=ns, inputs=[rings, nrm, vcol, gprev, Vw, ns, np_part,
                      sin_crit_p, DTYPE(lev_klev), SIG0, PCORE, sa_prev_p], outputs=[pp, pa, ps], device=dev)
            np_part += ns
        nw_new = nw + shed_per
        # bookkeeping: ns TEV then lev_count LEV (matches shed order). lev_count = ns*nsub for the subdivided sheet.
        if lev_shed_mode == 'hirato':
            lev_strengths = list(hirato_gL if hirato_gL is not None else np.zeros(ns, dtype=NP))
        elif use_lev_sheet:
            lev_strengths = list(substr)
        elif fp_shed:
            lev_strengths = list(lev_sign * lev_str_fp)
        else:
            lev_strengths = [0.0] * lev_count if lev_in_wake else []
        wtype.extend([0] * ns + [1] * lev_count)
        lev_born.extend([-1] * ns + [t] * lev_count)
        lev_s0.extend([0.0] * ns + lev_strengths)
        if nw > 0:   # convect OLD wake only; freshly-shed ring STAYS attached at the TE (Katz&Plotkin
            if use_wcore:   # (E) per-ring core: LEV rolls up tight (small core); TEV keeps WAKE_CORE for wake stability
                wp.launch(_convect_wcore_f32 if wake_f32 else _convect_wcore,
                          dim=(nw, 4), inputs=[rings, gamma, npan, wr, wg, wcore_conv_dev, nw,
                          DTYPE(ug.WAKE_CORE), Vw, DTYPE(dt)], outputs=[wr_new], device=dev)
            elif wake_f32:
                wp.launch(_convect_f32, dim=(nw, 4), inputs=[rings, gamma, npan, wr, wg, nw, Vw,
                          DTYPE(dt), DTYPE(ug.WAKE_CORE)], outputs=[wr_new], device=dev)
            else:
                wp.launch(ug.convect_kernel, dim=(nw, 4), inputs=[rings, gamma, npan, wr, wg, nw, Vw, DTYPE(dt)],
                          outputs=[wr_new], device=dev)   # order) so it cancels the trailing bound segment
            if rk2:   # Heun RK2: second Euler from the predicted midpoint wake, then average
                if wake_f32:
                    wp.launch(_convect_f32, dim=(nw, 4), inputs=[rings, gamma, npan, wr_new, wg, nw, Vw,
                              DTYPE(dt), DTYPE(ug.WAKE_CORE)], outputs=[wr_m2], device=dev)
                else:
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
            lev_born = lev_born[off:]; lev_s0 = lev_s0[off:]   # keep S5 hold state aligned with the shifted wake
        else:
            nw = nw_new
        if part_lev and np_part > 0 and part_transport == 'rvpm':
            # S3a rVPM transport (rvpm3d): LSRK3 + transposed stretching + sigma evolution +
            # corrected-Pedrizzetti; advection in the full field (rings frozen within the step).
            import rvpm3d                                            # lazy (rvpm3d imports this module)
            _rfn = lambda P: rvpm3d.ring_field_vel(P, rings, gamma, npan, wr, wg, nw, Vw, dev)
            Xn, Gn, Sn = rvpm3d.rvpm_step(pp_np[:np_part], pa_np[:np_part], ps_np[:np_part],
                                          _rfn, dt)
            pp_np[:np_part] = Xn; pa_np[:np_part] = Gn; ps_np[:np_part] = Sn
            # SURFACE GUARD (hybrid particle-panel standard practice, H4/DUST family): the discrete
            # lattice enforces no-penetration only AT collocations — during the upstroke the wing
            # sweeps through its own LEV cloud and particles leak between collocations into the
            # ring near-field -> near-singular strain, dt*|J|>1 stiff blow-up (t=160 fingerprint).
            # Push any particle closer than d_min=0.5*sigma+WAKE_CORE to its nearest collocation
            # OUT along that panel's normal to d_min. Geometric no-penetration floor, no constants
            # beyond the particle's own core scale.
            colm2 = col.numpy()
            dmin = 0.3 * ps_np[:np_part]                 # below the 0.5*v_le*dt birth offset scale
            # nearest collocation distance per particle (chunked):
            ndist = np.empty(np_part)
            CHG = 4096
            Pcur = pp_np[:np_part]
            for s0 in range(0, np_part, CHG):
                dxs = Pcur[s0:s0 + CHG, None, :] - colm2[None, :, :]
                dd = np.einsum("pqi,pqi->pq", dxs, dxs)
                ndist[s0:s0 + CHG] = np.sqrt(np.min(dd, 1))
            # ABSORB surface-penetrating particles (DUST-family practice): the lattice enforces
            # no-penetration only at collocations; a particle leaking inside sits in the ring
            # near-field -> dt*|J|>1 stiff blow-up. A snap-out guard CLUSTERS particles on the
            # collocation normals (coincident pairs -> singular mutual strain, worse). Deleting
            # models absorption into the boundary layer; removed circulation is bounded per step.
            # far-field cull (wake-truncation legality, cf. max_wake rings): particles convected
            # past 4 chords downstream contribute nothing to the LEV gate physics; bounds the
            # N^2 transport cost. Tracked prev-particle indices remapped (-1 -> new shed event).
            keep = (pp_np[:np_part, 0] < 4.0 * chord) & (ndist >= dmin)
            if not keep.all():
                nk = int(keep.sum())
                remap = -np.ones(np_part, dtype=int)
                remap[np.where(keep)[0]] = np.arange(nk)
                pp_np[:nk] = pp_np[:np_part][keep]; pa_np[:nk] = pa_np[:np_part][keep]
                ps_np[:nk] = ps_np[:np_part][keep]
                lev_prev_idx = np.where(lev_prev_idx >= 0, remap[np.maximum(lev_prev_idx, 0)], -1)
                np_part = nk
            pp = wp.array(pp_np, dtype=V3, device=dev)
            pa = wp.array(pa_np, dtype=V3, device=dev)
            ps = wp.array(ps_np, dtype=DTYPE, device=dev)
        elif part_lev and np_part > 0:   # legacy Euler advect (bound+TEV+mutual) -> rollup
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
    # FIRST-PRINCIPLES STEADY drag polar CD = CD0 + CL^2/(pi*AR*e), on the CYCLE-MEAN (base-AoA) lift only.
    # The flapping lift makes thrust (Knoller-Betz), not induced drag, so the polar must NOT use the instantaneous
    # CL (over-counts); the steady part (base AoA) gives the induced drag that grows with AoA. CD0/AR/e physical.
    S_full = 2.0 * half_span * chord; AR_w = 2.0 * half_span / max(chord, 1e-9); qd0 = 0.5 * ug.RHO * U ** 2
    CL_s = (L_bern + L_les - L_vis) / (qd0 * S_full + 1e-9)
    D_polar = (cd0_polar + CL_s ** 2 / (np.pi * AR_w * oswald)) * qd0 * S_full if drag_polar else 0.0
    # ---- BODY vs WIND axes. The model builds forces in the BODY frame (AoA via tilted freestream -> Fz is the
    # wing-normal force, Fx the chord-axial force). The wind-tunnel "lift / net thrust" convention is WIND axes
    # (lift _|_ freestream, thrust // freestream). The rotation by the body AoA: Fz*sin(a) is the lift's streamwise
    # projection = the induced-drag-like term (first-principles geometry, NOT a fitted drag polar). ----
    # ROBUST cycle-mean (winsorize to median +/- 8*MAD): the per-element LEV rings can convect through the
    # near-field of a bound collocation and produce a single-step near-SINGULAR Bernoulli spike (e.g. 1e4 N vs
    # ~4 N median) that self-heals one step later. These are numerical DVM artifacts, NOT physical force; clip
    # them before averaging. For well-behaved runs (no spike) all values lie inside the band -> identical result.
    def _robmean(a):
        a = np.asarray(a[last], float)
        m = np.median(a); mad = np.median(np.abs(a - m)) + 1e-12
        lo, hi = m - 8.0 * 1.4826 * mad, m + 8.0 * 1.4826 * mad
        return 2.0 * np.mean(np.clip(a, lo, hi))
    Fx_body = _robmean(Fxb_tot); Fz_body = _robmean(Fzb_tot)                          # total body force (both wings)
    _ca = np.cos(np.radians(aoa_deg)); _sa = np.sin(np.radians(aoa_deg))
    # RIG PARASITIC DRAG (~U^2): the wind-tunnel support plates (paper rig = plates + 2 wings, NO fuselage) add
    # a drag ~ Cd*A*1/2 rho U^2, FREQUENCY-INDEPENDENT. Applied along the flight/freestream direction -> reduces
    # T_wind by D_para, leaves L_wind unchanged. d_para = parasitic at U=8 m/s (calibrated; rig geometry not in
    # the paper). NOTE: the cross-flow FORM drag (prof_drag) was found to REVERSE the net-thrust vs freq trend
    # (Cd*sin^2(a_eff) grows faster than f^2) -> use this ~U^2 parasitic instead, which preserves the freq trend.
    D_para = d_para * (U / 8.0) ** 2
    Fx_body = Fx_body + D_para * _ca; Fz_body = Fz_body + D_para * _sa
    L_bodyf = Fz_body;              T_bodyf = -Fx_body                               # BODY frame lift / thrust
    L_windf = Fz_body * _ca - Fx_body * _sa                                          # WIND frame lift (_|_ freestream)
    T_windf = -(Fx_body * _ca + Fz_body * _sa)                                       # WIND frame thrust (// freestream)
    if os.environ.get("UTREND_DBG") and _UT_LOG:
        np.save(os.environ["UTREND_DBG"], np.array(_UT_LOG))   # (t, mean|arel|deg, mean_att, sum_xgated)
    return dict(L=L, Fx=Fx, T=-Fx, P=P, Lh=Lh, Xh=Xh, Lkj=Lkj, D_polar=D_polar,
                Fx_body=Fx_body, Fz_body=Fz_body, L_body=L_bodyf, T_body_f=T_bodyf,
                L_wind=L_windf, T_wind=T_windf,                                       # rotated wind-axes lift/thrust
                L_bern=L_bern, T_bern=-Fx_bern, Lh_bern=Lh_imp, Xh_bern=Xh_imp,   # Bernoulli force (captures LEV)
                L_visc=L_vis, D_visc=Fx_vis, T_lesp=-Fx_les,                      # friction (drag>0); LE suction (thrust)
                D_prof=2.0 * np.mean(Xh_pd[last]),                                # separated-flow form drag (>0 = drag)
                D_para=d0_drag,   # constant baseline drag (support plates + rig + flap-cycle separation), both wings
                                  # NOTE: empirically ~U-, f-independent over the tested 6-10 m/s x 1.4-2.6 Hz range
                                  # (the U,f dependence lives in the Garrick suction, which grows with V_rel^2)
                Lh_vis=Lh_vis, Xh_vis=Xh_vis, Lh_les=Lh_les, Xh_les=Xh_les,       # per-step viscous / LE-suction
                Lh_vtx=Lh_vtx, Xh_vtx=Xh_vtx, L_vtx=L_vtx, D_vtx=Fx_vtx,          # per-step + mean vortex normal force
                Lh_pd=Lh_pd, Xh_pd=Xh_pd, Lh_stall=Lh_stall, Xh_stall=Xh_stall,   # per-step form/faure drag + Fix1 stall (diag)
                Lh_vimp=Lh_vimp, Xh_vimp=Xh_vimp,                                  # (H16) per-step LEV vortex-impulse force
                Lh_ds=Lh_ds, L_dstall=2.0 * np.mean((Lh_imp + Lh_ds)[last]),       # dynamic-stall: per-step + mean(bern+LEV)
                L_stall=2.0 * np.mean(Lh_stall[last]),                            # Fix1 geometric-stall lift loss (<=0)
                L_fric=2.0 * np.mean(Lh_fric[last]), D_fric=2.0 * np.mean(Xh_fric[last]),  # Fix2 friction (lift/thrust comp)
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
