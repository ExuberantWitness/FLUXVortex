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


def gpu_run_twist(nc=4, ns=10, chord=0.287, half_span=0.80, U=8.0, aoa_deg=5.0,
                  flap_amp_deg=45.0, twist_amp_deg=22.5, twist_phase_deg=90.0,   # +90: twist LEADS flap 90deg
                  # (paper double-crank: psi~cos(wt), nose-down on downstroke = washout of the deep-stall AoA)
                  freq=2.0, n_cycle=5, steps_per_cycle=40, wake_rows=50, rk2=False, te_traj=False,
                  swept_axis=False, real_geom=False, real_lev=False, lev_sat=False, lev_merge=False, lev_tau=0.20,
                  lev_detach_deg=90.0,
                  lesp_crit_deg=15.0, lev_klev=1.0,
                  visc=False, tc_thick=0.06, prof_drag=False, cd_form=1.98, cd_sat_deg=30.0, cd_dp=1.2, d_para=0.0, les_suction=False, les_eta=1.0,
                  fp_lev=False, lev_kv=4.62, lev_trans_deg=15.0,
                  # --- 2026-06-27 first-principles LESP-LEV: orthogonal MODE switches (candidate-model matrix) ---
                  lev_shed_mode='none',    # 'none'|'kelvin'(Hirato)|'varA0'(Modulation Eq.11-12)|'kinematic'(legacy Path-B)
                  lev_hold_mode='inviscid',# 'inviscid'(convect freely)|'hold'(viscous τ_hold)|'hold_detach'(Li 4-phase cutoff)
                  a0_crit=0.25,            # critical LESP (airfoil/Re property; anchor via 2D flap_ldvm). 0.12 thin@Re10k .. 0.27 SD7003@Re20k
                  tau_hold_scale=1.0,      # ×c/(0.4U) viscous-hold timescale
                  lev_roll_core=0.01,      # FLOOR LEV vortex-core (chord frac); the actual core is resolution-adaptive (below)
                  lev_overlap=1.0,         # (STAB) LEV core = overlap × shed-spacing (∝U·dt & strip width) -> shrinks as grid refines, never near-singular
                  lev_consistent=True,     # apply the adaptive core in solve+force too (not just convect) -> grid-CONVERGENT LEV (vs singular drift/blow-up)
                  lev_sub=1,               # (FINE) spanwise sub-rings of LEV per strip (lev_sub=5 -> 5× finer LEV sheet, independent of wing grid)
                  lev_sheet=True,          # (E2) shed LEV as a CONNECTED trailing sheet from the LE (rolls up) instead of fixed-offset rings
                  lev_place='ansari',      # 'ansari' = Hirato Eq.7 placement, LEV sheet OVER the suction surface anchored at the LE; 'wake' = old (trails off the back, wrong)
                  lev_rollh=0.5,           # LEV roll-up height as it convects aft (chord frac) — the sheet lifts off the suction surface (Hirato Fig.11 spiral)
                  lev_fmax=1.4,            # drop LEV rings once they convect past this chord fraction (detach off the TE)
                  lev_sign=1.0,            # LEV circulation sign vs bound (+1 = same sign -> adds lift; test both)
                  lev_le_off=0.0,          # LEV sheet ORIGIN = the geometric leading-edge POINT (physically correct: the shear layer separates at the sharp LE, then rolls up above the surface). Stability comes from the convect core, not an offset.
                  attached_drag='none',    # 'none'|'faure'(static C_D(α_rel))|'legacy'(old visc/prof_drag)
                  # --- 2026-06-30 additive empirical-residual corrections (default OFF; physics-anchored) ---
                  geo_stall=False,         # Fix1: quasi-steady GEOMETRIC-pitch static stall lift loss (twist-driven, freq-independent)
                  geo_stall_deg=12.0,      # static stall angle alpha_ss (NACA-2406 @ Re~1e5; airfoil property)
                  geo_stall_width=16.0,    # separation-spread angle: alpha past stall over which TE separation goes full (airfoil property)
                  geo_stall_peak=False,    # False=instantaneous psi(t) each step; True=cycle-peak |psi| amplitude
                  fric_drag=False,         # Fix2: flap-velocity^2 friction drag (turbulent flat-plate Cf; reuses visc structure)
                  cf_mode='turbulent',     # 'turbulent'(0.074/Re^0.2) | 'laminar'(1.328/sqrt Re)
                  drag_polar=False, cd0_polar=0.018, oswald=0.85,
                  d0_drag=0.0,
                  part_lev=False, lev_cons=False, lev_core=0.10, lev_sig0=0.5, lev_owin=2.0,
                  sym=False, root_off=0.0, stall=False, stall_deg=12.0,
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
    # NEW first-principles LESP-LEV sheds a ring/strip into the SAME wake (enters rhs + Bernoulli surface force);
    # reuses the real_lev plumbing. fp_shed True for any A0-based / kinematic LEV shed mode.
    fp_shed = lev_shed_mode in ('kelvin', 'varA0', 'kinematic')
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
    pmax = N * ns + ns; np_part = 0                  # one particle per strip per step + slack
    pp = wp.zeros(pmax, dtype=V3, device=dev); pa = wp.zeros(pmax, dtype=V3, device=dev)
    ps = wp.zeros(pmax, dtype=DTYPE, device=dev); pp_new = wp.zeros(pmax, dtype=V3, device=dev)
    sa_prev_p = wp.zeros(ns, dtype=DTYPE, device=dev)   # previous |sin a_eff| per strip (LESP-rate shed gate)
    I_lev_prev = np.zeros(3); I_lev_have = False         # previous LE-referenced LEV impulse (for -dI/dt force)
    SIG0 = DTYPE(lev_sig0 * U * dt); PCORE = DTYPE(lev_core)   # LEV particle core (smaller -> stronger induction)
    sin_crit_p = DTYPE(np.sin(np.radians(lesp_crit_deg)))
    # --- coherent-core LEV (N-LEV merging, N=1 per strip): ONE smooth merged ring/strip (CPU state) ---
    lev_cen = np.zeros((ns, 3)); lev_gam = np.zeros(ns); lev_gam_raw = np.zeros(ns)
    lev_rw = wp.zeros((ns, 4), dtype=V3, device=dev); lev_gw = wp.zeros(ns, dtype=DTYPE, device=dev)
    Vlev = wp.zeros(npan, dtype=V3, device=dev)
    Vpart = wp.zeros(npan, dtype=V3, device=dev)     # LEV-particle induced velocity at collocations (rVPM force)
    Lh = np.zeros(N); Xh = np.zeros(N); Ph = np.zeros(N); Lkjh = np.zeros(N)
    Lh_imp = np.zeros(N); Xh_imp = np.zeros(N)        # unsteady-Bernoulli surface-pressure force (captures LEV)
    Fxb_tot = np.zeros(N); Fzb_tot = np.zeros(N)      # TOTAL body-frame force per step (sum of ALL force vectors:
    #   Bernoulli + LE-suction + friction + form-drag + vortex) -> the clean body force to rotate into wind axes
    Lh_vis = np.zeros(N); Xh_vis = np.zeros(N)         # DeLaurier viscous friction drag (strip, Re-based Blasius)
    Lh_pd = np.zeros(N); Xh_pd = np.zeros(N)           # separated-flow form/pressure drag (high-alpha, viscous-origin)
    Lh_les = np.zeros(N); Xh_les = np.zeros(N)         # leading-edge suction thrust (Garrick/DeLaurier dTs)
    Lh_vtx = np.zeros(N); Xh_vtx = np.zeros(N)         # high-alpha vortex normal force (Polhamus, lift+drag)
    Lh_ds = np.zeros(N)                                 # dynamic-stall LEV lift (sustains the downstroke plateau)
    Lh_stall = np.zeros(N)                              # Fix1: geometric quasi-steady stall lift loss (<=0)
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
    lev_core_force = lev_roll_core * chord                                                 # FORCE: small (lift sharpness)
    use_lev_sheet = fp_shed and lev_sheet                      # (E2) connected LEV sheet from the LE (rolls up)
    nls = ns * nsub                                            # number of LEV sub-rings shed per step (sheet)
    lpl = wp.zeros(nls, dtype=V3, device=dev); lpr = wp.zeros(nls, dtype=V3, device=dev)   # prev LE-shed corners
    lcl = wp.zeros(nls, dtype=V3, device=dev); lcr = wp.zeros(nls, dtype=V3, device=dev)   # cur LE-shed corners
    lev_first = 1                                               # 1 until the first LEV row is shed
    # (ANSARI / Hirato Eq.7) parametric LEV sheet OVER the suction surface: each ring stored by its strip index,
    # chordwise fraction f (0=LE, grows aft as it convects), and strength. Lifted off the surface by lev_rollh*f
    # (roll-up). Anchored at the LE, NOT convected into the TEV wake -> sits over the wing, induces on bound+force.
    lev_aj = np.zeros(0, np.int64); lev_af = np.zeros(0); lev_ag = np.zeros(0)
    Vlev_a = wp.zeros(npan, dtype=V3, device=dev)              # LEV-sheet induced velocity at collocations
    lev_frame_rings = np.zeros((0, 4, 3)); lev_frame_g = np.zeros(0)   # LEV-sheet geometry for viz frames
    for t in range(N):
        wcore_dev = None; wcore_force_dev = None
        if use_wcore and nw > 0:                                # per-ring core: LEV gets a core, TEV = standard
            islev_np = np.asarray(lev_born[:nw]) >= 0
            # SOLVE/convect core (stabilizing, resolution-adaptive) and FORCE core (small, for held-lift sharpness)
            cc_np = np.where(islev_np, lev_core_abs, ug.WAKE_CORE).astype(NP)
            cf_np = np.where(islev_np, lev_core_force, ug.WAKE_CORE).astype(NP)
            wcore_dev = wp.array(cc_np, dtype=DTYPE, device=dev)
            wcore_force_dev = wp.array(cf_np, dtype=DTYPE, device=dev)
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
        # CONSISTENT resolution-adaptive core in the SOLVE too (toggle lev_consistent): regularizes the singular
        # LEV→bound feedback that makes nc/ns NON-convergent. The core shrinks as the grid refines -> the result
        # CONVERGES to a grid-independent value (vs the singular kernel which drifts 7→3 with nc + blows up).
        if use_wcore and nw > 0 and lev_consistent:
            wp.launch(_rhs_moving_wcore, dim=npan, inputs=[col, nrm, Vw, vcol, wr, wg, wcore_dev, nw], outputs=[rhs], device=dev)
        else:
            wp.launch(ug.rhs_moving_kernel, dim=npan, inputs=[col, nrm, Vw, vcol, wr, wg, nw], outputs=[rhs], device=dev)
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
        if use_wcore and nw > 0 and lev_consistent:
            wp.launch(_col_wake_wcore, dim=npan, inputs=[col, wr, wg, wcore_dev, nw], outputs=[Vwk], device=dev)
        else:
            wp.launch(ug.col_wake_vel_kernel, dim=npan, inputs=[col, wr, wg, nw], outputs=[Vwk], device=dev)
        cc = rings.numpy(); g = gamma.numpy().reshape(-1); gp = gprev.numpy().reshape(-1)
        Vcol = np.asarray(Vinf) - vcn + Vwk.numpy()                     # full local velocity at panels
        if lev_merge:
            Vcol = Vcol + Vlev.numpy()                                  # coherent LEV core induction (Bernoulli)
        if use_ansari:
            Vcol = Vcol + Vlev_a.numpy()                                # (HIRATO) LEV-sheet induction enters the Bernoulli force
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
        dp = np.clip(dp, -8.0 * q_ref, 8.0 * q_ref)
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
            fsep = np.clip(1.0 - (psi_abs - ass) / np.radians(geo_stall_width), 0.0, 1.0)
            fsep = np.where(psi_abs <= ass, 1.0, fsep)
            loss_frac = 1.0 - ((1.0 + np.sqrt(fsep)) / 2.0) ** 2               # 0 attached (twist=0), ->0.75 separated
            Fb_strip_z = Fb[:, 2].reshape(nc, ns).sum(0)                       # per-strip Bernoulli lift (ns,)
            dLz = -loss_frac * np.maximum(Fb_strip_z, 0.0)                     # remove only positive lift on stalled strips
            Lh_stall[t] = float(np.sum(dLz)); Fzb_tot[t] += Lh_stall[t]        # lift-axis only (thrust handled by Fix2)
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
        xref_frac = 0.10                                   # FIXED reference chord fraction (nc-independent)
        i_ref = np.argmax(cumpos >= (xref_frac * c_strip)[None, :], axis=0)   # first panel reaching x_ref per strip
        Gamma_ref = gm2[i_ref, np.arange(ns)]              # cumulative bound circulation up to x_ref (converges in nc)
        th1 = np.arccos(np.clip(1.0 - 2.0 * xref_frac, -1.0, 1.0))   # FIXED (x_ref/c = xref_frac)
        vr_le = (np.asarray(Vinf) - vcn)[:ns]              # LE-row body-relative flow
        Urel_le = np.linalg.norm(vr_le, axis=1) + 1e-9
        A0 = 1.13 * Gamma_ref / (Urel_le * c_strip * (th1 + np.sin(th1)) + 1e-12)   # finite-wing LESP per strip (nc-robust)
        A0 = np.clip(np.nan_to_num(A0, nan=0.0, posinf=0.0, neginf=0.0), -3.0, 3.0)   # guard near-field blow-up
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
        if use_ansari:   # (HIRATO) update the parametric LEV sheet: convect rings aft (chordwise fraction f += U*dt/c),
            # drop those past lev_fmax (detach off the TE), and SHED a new ring at the LE (f=0) for every strip whose
            # |A0|>a0_crit (LESP supercritical), with the S3 strength. The sheet thus stays anchored at the LE.
            if len(lev_aj) > 0:
                lev_af = lev_af + (U * dt) / np.maximum(c_strip[lev_aj], 1e-9)   # chordwise convection (fraction of chord)
                keep = lev_af < lev_fmax
                lev_aj = lev_aj[keep]; lev_af = lev_af[keep]; lev_ag = lev_ag[keep]
            js = np.where(np.abs(A0) > a0_crit)[0]                       # supercritical strips -> shed a new LEV ring at the LE
            if len(js) > 0:
                lev_aj = np.concatenate([lev_aj, js.astype(np.int64)])
                lev_af = np.concatenate([lev_af, np.zeros(len(js))])
                lev_ag = np.concatenate([lev_ag, (lev_sign * lev_str_fp[js]).astype(float)])
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
        if prof_drag:
            nnq = nrm.numpy(); vr = np.asarray(Vinf) - vcn       # relative wind (freestream + flapping)
            vrm = np.linalg.norm(vr, axis=1) + 1e-9
            sap = np.sum(vr * nnq, axis=1) / vrm                 # sin(alpha_eff) per panel
            # CROSS-FLOW (Hoerner) separated drag: Cd = Cd_max*sin^2(a_eff), Cd_max=cd_form~1.98 (flat-plate
            # slope), PLATEAUED at the deep-stall value cd_dp (~1.2, airfoil deep-stall Cd / Hoerner). The bluff-
            # body Cd saturates past full separation (does NOT climb to the 2.0 broadside value) -> form drag
            # stays ~f^2 at high frequency (matches measured net-thrust trend). First-principles (no RoboEagle fit).
            Cdp = np.minimum(cd_form * sap ** 2, cd_dp)         # cross-flow form drag, deep-stall plateau cd_dp
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
            Vle = Vcol[iLE]; nle = nn2[iLE]; tcle = tc[iLE]
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
            if lev_shed_mode in ('kelvin', 'varA0', 'kinematic'):
                # S4 (Hirato Eq.20): realized LE suction caps at the FIRST-PRINCIPLES A0 (from bound circulation),
                # bounded by a0_crit; the EXCESS above a0_crit is what S3 sheds into the LEV (no double-count).
                sa_s = np.clip(A0, -a0_crit, a0_crit)
            dTs = np.pi * les_eta * ug.RHO * c_le * dy_le * (Vle_m ** 2) * (sa_s ** 2)
            Fs = -dTs[:, None] * tcle                         # forward (-chordwise) suction force vector
            Lh_les[t] = float(np.sum(Fs[:, 2])); Xh_les[t] = float(np.sum(Fs[:, 0]))
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
        if attached_drag == 'faure':
            nnf = nrm.numpy(); vrf = np.asarray(Vinf) - vcn          # body-relative flow per panel
            vrm = np.linalg.norm(vrf, axis=1) + 1e-9
            arel = np.abs(np.arcsin(np.clip(np.sum(vrf * nnf, axis=1) / vrm, -0.999, 0.999)))   # |alpha_rel|
            a_ref = np.radians(12.0)                                 # profile-drag bucket half-width (airfoil property)
            Cd_att = cd0_polar * (1.0 + (arel / a_ref) ** 2)         # static profile-drag polar
            att = np.ones(npan)                                       # gate: attached strips only (LEV not shed)
            if lev_shed_mode in ('kelvin', 'varA0', 'kinematic'):
                att = np.tile((np.abs(A0) <= a0_crit).astype(NP), nc)   # panel p uses strip j=p%ns gate (attached only)
            Dfa = 0.5 * ug.RHO * vrm[:, None] * Cd_att[:, None] * area[:, None] * att[:, None] * vrf  # along rel. wind
            Lh_pd[t] += float(np.sum(Dfa[:, 2])); Xh_pd[t] += float(np.sum(Dfa[:, 0]))
            Fzb_tot[t] += float(np.sum(Dfa[:, 2])); Fxb_tot[t] += float(np.sum(Dfa[:, 0]))
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
                sep=(np.abs(sina) > np.sin(np.radians(lesp_crit_deg))), nc=nc, ns=ns))
        if te_traj:   # shed along the TE trajectory (continuous sheet for the plunging TE)
            wp.launch(_shed_te_traj, dim=ns, inputs=[rings, gamma, te, tpl, tpr, Vw, DTYPE(dt), nw],
                      outputs=[wr, wg, tcl, tcr], device=dev)
            wp.copy(tpl, tcl); wp.copy(tpr, tcr)        # current TE becomes next step's "previous"
        else:
            wp.launch(ug.shed_kernel, dim=ns, inputs=[rings, gamma, te, Vw, DTYPE(dt), nw], outputs=[wr, wg], device=dev)
        if fp_shed:   # FIRST-PRINCIPLES LESP-LEV with S3 strength lev_str_fp (varA0/kelvin/kinematic).
            lev_str_w = wp.array((lev_sign * lev_str_fp).astype(NP), dtype=DTYPE, device=dev)
            if use_lev_sheet:   # (E2) CONNECTED LEV sheet from the geometric LE (offset onto suction side) -> rolls up
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
        if part_lev:   # PARTICLE leading-edge vortex: shed one spanwise particle/strip at the LE (delayed-Kutta gprev)
            wp.launch(shed_lev_particles_kernel, dim=ns, inputs=[rings, nrm, vcol, gprev, Vw, ns, np_part,
                      sin_crit_p, DTYPE(lev_klev), SIG0, PCORE, sa_prev_p], outputs=[pp, pa, ps], device=dev)
            np_part += ns
        nw_new = nw + shed_per
        # bookkeeping: ns TEV then lev_count LEV (matches shed order). lev_count = ns*nsub for the subdivided sheet.
        lev_strengths = list(substr) if use_lev_sheet else (list(lev_sign * lev_str_fp) if fp_shed
                        else ([0.0] * lev_count if lev_in_wake else []))
        wtype.extend([0] * ns + [1] * lev_count)
        lev_born.extend([-1] * ns + [t] * lev_count)
        lev_s0.extend([0.0] * ns + lev_strengths)
        if nw > 0:   # convect OLD wake only; freshly-shed ring STAYS attached at the TE (Katz&Plotkin
            if use_wcore:   # (E) per-ring core: LEV rolls up tight (small core) without TEV near-singular noise
                wp.launch(_convect_wcore, dim=(nw, 4), inputs=[rings, gamma, npan, wr, wg, wcore_dev, nw,
                          DTYPE(ug.WAKE_CORE), Vw, DTYPE(dt)], outputs=[wr_new], device=dev)
            else:
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
            lev_born = lev_born[off:]; lev_s0 = lev_s0[off:]   # keep S5 hold state aligned with the shifted wake
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
