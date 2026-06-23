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


@wp.kernel
def _lev_kernel(rings: wp.array(dtype=V3, ndim=2), nrm: wp.array(dtype=V3), vcol: wp.array(dtype=V3),
                Vinf: V3, rho: DTYPE, K_v: DTYPE, sin_onset: DTYPE, sin_ds: DTYPE,
                out: wp.array(dtype=DTYPE)):
    """Dynamic-stall LEV (Polhamus): per-panel vortex normal force C_Nv=K_v*sin^2(a_eff) (signed) at the
    LOCAL dynamic pressure when |sin a_eff|>sin_onset, SATURATED beyond the deep-stall angle (sin_ds): the
    LEV detaches so its contribution caps (matches the measured lift plateau at high AoA, not unbounded)."""
    p = wp.tid()
    vr = Vinf - vcol[p]
    vmag = wp.length(vr) + wp.float64(1.0e-9)
    n = nrm[p]
    sina = wp.dot(vr, n) / vmag
    sa = wp.abs(sina)
    if sa > sin_onset:
        sc = wp.min(sa, sin_ds)              # cap magnitude at deep stall (LEV detaches beyond)
        sgn = wp.float64(1.0)
        if sina < wp.float64(0.0):
            sgn = wp.float64(-1.0)
        cr = wp.cross(rings[p, 2] - rings[p, 0], rings[p, 3] - rings[p, 1])
        area = wp.float64(0.5) * wp.length(cr)
        dN = K_v * sgn * sc * sc * wp.float64(0.5) * rho * vmag * vmag * area
        wp.atomic_add(out, 0, dN * n[2])     # vertical (lift), dihedral-projected
        wp.atomic_add(out, 1, dN * n[0])     # streamwise (LE suction -> thrust)


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
                  swept_axis=False, lev=False, K_v=2.5, lev_onset_deg=15.0, lev_ds_deg=33.0,
                  induced_drag=False):
    """Twisted flapping UVLM (same validated kernels as flap_flight_validate.gpu_run).
    rk2=True -> 2nd-order Heun free-wake convection. te_traj=True -> shed wake along TE trajectory.
    swept_axis=True -> real RoboEagle flap/twist axis (33.8%c root -> LE tip), not quarter-chord.
    lev=True -> add the DYNAMIC-STALL LEV via the Polhamus leading-edge-suction analogy: when a panel's
    effective AoA exceeds lev_onset, the lost LE suction reappears as a vortex normal force C_Nv=K_v*sin^2a
    (signed), evaluated at the LOCAL dynamic pressure (incl. flap velocity -> rises with frequency)."""
    dev = cfg.DEVICE; NP = cfg.NP_DTYPE
    C0 = ffv.flat_wing(nc, ns, chord, half_span); npan = nc * ns; ncv = (nc + 1) * (ns + 1)
    A_f = np.radians(flap_amp_deg); A_t = np.radians(twist_amp_deg); phi = np.radians(twist_phase_deg)
    Om = 2.0 * np.pi * freq; x_ea = 0.25 * chord
    Vinf = np.array([U, 0.0, U * np.tan(np.radians(aoa_deg))]); Vw = V3(*[float(v) for v in Vinf])
    T = 1.0 / freq; dt = T / steps_per_cycle; N = n_cycle * steps_per_cycle
    te = wp.array(np.array([(nc - 1) * ns + j for j in range(ns)], np.int32), dtype=wp.int32, device=dev)
    wake_max = wake_rows * ns; maxw = min(N * ns, wake_max) + ns
    wr = wp.zeros((maxw, 4), dtype=V3, device=dev); wr_new = wp.zeros((maxw, 4), dtype=V3, device=dev)
    wr_m2 = wp.zeros((maxw, 4), dtype=V3, device=dev) if rk2 else None   # RK2 second-Euler buffer
    tpl = wp.zeros(ns, dtype=V3, device=dev); tpr = wp.zeros(ns, dtype=V3, device=dev)   # prev TE corners
    tcl = wp.zeros(ns, dtype=V3, device=dev); tcr = wp.zeros(ns, dtype=V3, device=dev)   # cur TE corners
    wg = wp.zeros(maxw, dtype=DTYPE, device=dev); gprev = wp.zeros((1, npan), dtype=DTYPE, device=dev); nw = 0
    Lh = np.zeros(N); Xh = np.zeros(N); Ph = np.zeros(N); Lkjh = np.zeros(N)
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
        Fp = wp.zeros(npan, dtype=V3, device=dev)
        if induced_drag:   # add wake-induced velocity in the force -> induced drag (cancels inviscid thrust)
            wp.launch(ug.panel_force_ind_kernel, dim=npan, inputs=[rings, nrm, gamma, gprev, vcol, Vw,
                      DTYPE(dt), DTYPE(ug.RHO), ns, wr, wg, nw], outputs=[Fp], device=dev)
        else:
            wp.launch(ug.panel_force_kernel, dim=npan, inputs=[rings, nrm, gamma, gprev, vcol, Vw,
                      DTYPE(dt), DTYPE(ug.RHO), ns], outputs=[Fp], device=dev)
        Fpn = Fp.numpy(); vcn = vcol.numpy()
        Lh[t] = np.sum(Fpn[:, 2]); Xh[t] = np.sum(Fpn[:, 0]); Ph[t] = -np.sum(np.einsum('pi,pi->p', Fpn, vcn))
        if lev:   # dynamic-stall LEV (Polhamus LE-suction analogy), on-GPU
            levout = wp.zeros(2, dtype=DTYPE, device=dev)
            wp.launch(_lev_kernel, dim=npan, inputs=[rings, nrm, vcol, Vw, DTYPE(ug.RHO), DTYPE(K_v),
                      DTYPE(np.sin(np.radians(lev_onset_deg))), DTYPE(np.sin(np.radians(lev_ds_deg)))],
                      outputs=[levout], device=dev)
            lo = levout.numpy(); Lh[t] += float(lo[0]); Xh[t] += float(lo[1])
        lkj = wp.zeros(1, dtype=DTYPE, device=dev)        # DIAG: Vinf-only KJ lift (no plunge tilt)
        wp.launch(ug.lift_kj_kernel, dim=npan, inputs=[rings, nrm, gamma, gprev, Vw, DTYPE(dt),
                  DTYPE(ug.RHO), ns], outputs=[lkj], device=dev)
        Lkjh[t] = float(lkj.numpy()[0])
        if te_traj:   # shed along the TE trajectory (continuous sheet for the plunging TE)
            wp.launch(_shed_te_traj, dim=ns, inputs=[rings, gamma, te, tpl, tpr, Vw, DTYPE(dt), nw],
                      outputs=[wr, wg, tcl, tcr], device=dev)
            wp.copy(tpl, tcl); wp.copy(tpr, tcr)        # current TE becomes next step's "previous"
        else:
            wp.launch(ug.shed_kernel, dim=ns, inputs=[rings, gamma, te, Vw, DTYPE(dt), nw], outputs=[wr, wg], device=dev)
        nw_new = nw + ns
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
        else:
            nw = nw_new
        gprev = wp.array(gamma.numpy(), dtype=DTYPE, device=dev)
    last = slice((n_cycle - 1) * steps_per_cycle, N)
    L = 2.0 * np.mean(Lh[last]); Fx = 2.0 * np.mean(Xh[last]); P = 2.0 * np.mean(np.abs(Ph[last]))
    Lkj = 2.0 * np.mean(Lkjh[last])
    return dict(L=L, Fx=Fx, T=-Fx, P=P, Lh=Lh, Xh=Xh, Lkj=Lkj)   # Lkj = Vinf-only KJ lift (no plunge tilt)


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
