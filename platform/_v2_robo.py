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
import flap_flight_validate as ffv
import _v2_robogeom as rg                       # real RoboEagle planform + swept flap/twist axis

V3 = wp.vec3d
RHO = 1.225


def twisted_corners(C0, t, A_f, A_t, Om, phi, x_ea, span, swept_axis=False):
    """Flat wing -> flap (rotate about root x-axis by θ=A_f sin Ωt) + spanwise-linear twist
    (pitch each section about the y-axis at x_ea by ψ(y)=A_t (y/span) sin(Ωt+phi)).
    swept_axis=True: real RoboEagle twist axis swept 33.8%c(root)->LE(tip) (_v2_robogeom.axis_x),
    not a constant x_ea — matches the paper's measured flap/twist hinge."""
    th = A_f * np.sin(Om * t)
    ct, st = np.cos(th), np.sin(th)
    x = C0[..., 0]; y = C0[..., 1]; z0 = C0[..., 2]    # z0 = NACA-2406 camber surface (was discarded!)
    xe = rg.axis_x(y, span) if swept_axis else x_ea    # swept twist axis (per spanwise y) or constant
    psi = A_t * (y / span) * np.sin(Om * t + phi)
    cp, sp = np.cos(psi), np.sin(psi)
    xr = xe + (x - xe) * cp - z0 * sp                  # twist pitches (x-xe, z0) about y at the axis
    zr = (x - xe) * sp + z0 * cp                       # carry the camber through the rotation
    xf = xr                            # flap: rotate (y,z) about x by θ
    yf = y * ct - zr * st
    zf = y * st + zr * ct
    return np.stack([xf, yf, zf], axis=-1)


def twisted_state(C0, t, A_f, A_t, Om, phi, x_ea, span, dlt=1e-6, swept_axis=False):
    corners = twisted_corners(C0, t, A_f, A_t, Om, phi, x_ea, span, swept_axis)
    cp = twisted_corners(C0, t + dlt, A_f, A_t, Om, phi, x_ea, span, swept_axis)
    cm = twisted_corners(C0, t - dlt, A_f, A_t, Om, phi, x_ea, span, swept_axis)
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
                  swept_axis=False, real_geom=False, real_lev=False, lesp_crit_deg=15.0, lev_klev=1.0,
                  frames_out=None, frame_skip=3):
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
    C0 = (rg.robowing_real(nc, ns, half_span) if real_geom
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
    Lh = np.zeros(N); Xh = np.zeros(N); Ph = np.zeros(N); Lkjh = np.zeros(N)
    Lh_imp = np.zeros(N); Xh_imp = np.zeros(N)        # unsteady-Bernoulli surface-pressure force (captures LEV)
    wtype = []                                        # CPU bookkeeping: 0=TEV, 1=LEV per wake ring (for viz)
    for t in range(N):
        corners, cvel = twisted_state(C0, t * dt, A_f, A_t, Om, phi, x_ea, half_span, swept_axis=swept_axis)
        cw = wp.array(corners.reshape(ncv, 3).astype(NP), dtype=V3, device=dev)
        vw = wp.array(cvel.reshape(ncv, 3).astype(NP), dtype=V3, device=dev)
        rings = wp.zeros((npan, 4), dtype=V3, device=dev); col = wp.zeros(npan, dtype=V3, device=dev)
        nrm = wp.zeros(npan, dtype=V3, device=dev); vcol = wp.zeros(npan, dtype=V3, device=dev)
        wp.launch(ug.bound_rings_kernel, dim=npan, inputs=[cw, nc, ns], outputs=[rings, col, nrm], device=dev)
        wp.launch(ug.colvel_kernel, dim=npan, inputs=[vw, nc, ns], outputs=[vcol], device=dev)
        AIC = wp.zeros((1, npan, npan), dtype=DTYPE, device=dev)
        wp.launch(ug.aic_kernel, dim=(npan, npan), inputs=[rings, col, nrm], outputs=[AIC], device=dev)
        rhs = wp.zeros((1, npan), dtype=DTYPE, device=dev)
        wp.launch(ug.rhs_moving_kernel, dim=npan, inputs=[col, nrm, Vw, vcol, wr, wg, nw], outputs=[rhs], device=dev)
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
        Lh_imp[t] = float(np.sum(Fb[:, 2])); Xh_imp[t] = float(np.sum(Fb[:, 0]))
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
                sep=(np.abs(sina) > np.sin(np.radians(lesp_crit_deg))), nc=nc, ns=ns))
        if te_traj:   # shed along the TE trajectory (continuous sheet for the plunging TE)
            wp.launch(_shed_te_traj, dim=ns, inputs=[rings, gamma, te, tpl, tpr, Vw, DTYPE(dt), nw],
                      outputs=[wr, wg, tcl, tcr], device=dev)
            wp.copy(tpl, tcl); wp.copy(tpr, tcr)        # current TE becomes next step's "previous"
        else:
            wp.launch(ug.shed_kernel, dim=ns, inputs=[rings, gamma, te, Vw, DTYPE(dt), nw], outputs=[wr, wg], device=dev)
        if real_lev:   # REAL leading-edge vortex: shed discrete LEV rings at the LE (after the TEV block)
            wp.launch(_shed_lev_kernel, dim=ns, inputs=[rings, nrm, vcol, gamma, Vw, ns, nw + ns,
                      DTYPE(np.sin(np.radians(lesp_crit_deg))), DTYPE(lev_klev)], outputs=[wr, wg], device=dev)
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
        gprev = wp.array(gamma.numpy(), dtype=DTYPE, device=dev)
    last = slice((n_cycle - 1) * steps_per_cycle, N)
    L = 2.0 * np.mean(Lh[last]); Fx = 2.0 * np.mean(Xh[last]); P = 2.0 * np.mean(np.abs(Ph[last]))
    L_bern = 2.0 * np.mean(Lh_imp[last]); Fx_bern = 2.0 * np.mean(Xh_imp[last])
    Lkj = 2.0 * np.mean(Lkjh[last])
    return dict(L=L, Fx=Fx, T=-Fx, P=P, Lh=Lh, Xh=Xh, Lkj=Lkj,
                L_bern=L_bern, T_bern=-Fx_bern, Lh_bern=Lh_imp, Xh_bern=Xh_imp)   # Bernoulli force (captures LEV)


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
