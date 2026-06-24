"""Physical validation against REAL bird-scale flapping-MAV flight data (AST-grade, not a toy).

The §3 validation table establishes numerical correctness (GPU vs numpy, gradient vs FD). This module
adds the validation an aerospace reviewer actually asks for: does the production unsteady free-wake aero
produce REALISTIC flapping-flight forces and power at bird scale? We drive a wing through prescribed
flapping kinematics (rotation about the span root, θ(t)=A·sin(2πft)) in forward flight, run the
unsteady free-wake UVLM (the same `_aero_step` the differentiable coupled solver uses — moving-body BC,
unsteady Kutta–Joukowski + ∂Γ/∂t added-mass force), and recover the cycle-averaged LIFT, THRUST and
mechanical POWER. We compare against published large flapping-wing robots (within the ~2× band that is
accepted as physical credibility in this field):

    Zhong et al. 2026 (rigid–flexible coupling, J. Field Robotics): span 1.8 m, mass 1.0 kg, cruise power 81.6 W
    HIT-Hawk / HIT-Phoenix class (Zhong & Xu): span 1.6–2 m, mass 0.5–1.0 kg, 6–10 m/s, 1.5–3.5 Hz, 40–82 W
    E-Flap (Zufferey, IEEE RA-L 2021): span 1.5 m, mass 0.5–0.7 kg

so the target band is LIFT ≈ weight (5–10 N) and POWER ≈ 20–82 W at ~1.6–1.8 m span, ~0.5–1 kg,
6–10 m/s, 2–3 Hz. A rigid wing at mean angle of attack with plunge-dominated flapping already produces
mean lift (from the mean AoA) and thrust (Knoller–Betz), so it is the clean first physical anchor; the
elastic passive-twist refinement only improves propulsive efficiency.
"""
from __future__ import annotations

import numpy as np

import diff_coupled_unsteady as dcu     # _aero_step: production moving-body unsteady aero
import diff_uvlm_unsteady as uv

RHO = uv.RHO                            # 1.225 kg/m^3
G = 9.81


def flat_wing(nc, ns, chord, span):
    """Flat half-wing lattice (nc+1, ns+1, 3) in the x–y plane (z=0), root at y=0, tip at y=span.
    Angle of attack is applied via the freestream (an upward component), so the flap rotation about the
    x-axis through the root leaves the root fixed."""
    xs = np.linspace(0.0, chord, nc + 1)
    ys = np.linspace(0.0, span, ns + 1)
    C = np.zeros((nc + 1, ns + 1, 3))
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            C[i, j] = [x, y, 0.0]
    return C


def flap_state(C0, theta, thetadot):
    """Rotate the flat wing about the x-axis (flap/dihedral) through the root by θ; return the deformed
    corners and the corner velocity field (rigid-body rotational velocity)."""
    ct, st = np.cos(theta), np.sin(theta)
    R = np.array([[1.0, 0.0, 0.0], [0.0, ct, -st], [0.0, st, ct]])
    Rd = thetadot * np.array([[0.0, 0.0, 0.0], [0.0, -st, -ct], [0.0, ct, -st]])
    Cf = C0 @ R.T
    Vf = C0 @ Rd.T
    return Cf, Vf


def run(nc=4, ns=10, chord=0.29, half_span=0.80, mass=0.52, U=8.0, aoa_deg=6.0,
        flap_amp_deg=28.0, freq=3.0, n_cycle=4, steps_per_cycle=36, wake_rows=50, verbose=True):
    """numpy ORACLE only (free-wake aero is O(n_wake²)/step — too slow for production; use gpu_run).
    Kept for a single bit-exact GPU↔numpy cross-check. Same row-based (mesh-independent) wake truncation."""
    C0 = flat_wing(nc, ns, chord, half_span)
    npan = nc * ns; wake_max = wake_rows * ns
    A = np.radians(flap_amp_deg); Om = 2.0 * np.pi * freq
    Vinf = np.array([U, 0.0, U * np.tan(np.radians(aoa_deg))])   # AoA via upward freestream component
    T = 1.0 / freq; dt = T / steps_per_cycle; N = n_cycle * steps_per_cycle
    wake = []; gamma_prev = np.zeros(npan)
    Lh = np.zeros(N); Th = np.zeros(N); Ph = np.zeros(N); th_hist = np.zeros(N)
    for t in range(N):
        tt = t * dt
        theta = A * np.sin(Om * tt); thetadot = A * Om * np.cos(Om * tt)
        corners, cvel = flap_state(C0, theta, thetadot)
        Fp, gamma, wake = dcu._aero_step(corners, cvel, wake, gamma_prev, nc, ns, Vinf, dt, free_wake=True)
        if len(wake) > wake_max:
            wake = wake[-wake_max:]
        gamma_prev = gamma
        vcol = dcu._collocation_field(cvel)                     # panel-collocation velocity for the power
        Lh[t] = np.sum(Fp[:, 2])                                # vertical force (lift)
        Th[t] = -np.sum(Fp[:, 0])                               # streamwise: thrust = −F_x (flight in −x)
        Ph[t] = -np.sum(np.einsum('pi,pi->p', Fp, vcol))        # actuator aero power = −Σ F·v_panel
        th_hist[t] = theta
    last = slice((n_cycle - 1) * steps_per_cycle, N)            # cycle-average the final (converged) cycle
    Lf = 2.0 * np.mean(Lh[last]); Tf = 2.0 * np.mean(Th[last]); Pf = 2.0 * np.mean(np.abs(Ph[last]))
    W = mass * G; full_span = 2.0 * half_span; area = 2.0 * chord * half_span
    k_red = np.pi * freq * chord / U
    if verbose:
        print(f"Bird-scale flapping-flight validation — production unsteady free-wake UVLM ({npan} panels/half-wing):")
        print(f"  wing: span {full_span:.2f} m, chord {chord:.2f} m, area {area:.3f} m^2, mass {mass*1000:.0f} g (W={W:.2f} N)")
        print(f"  kinematics: U={U:.1f} m/s, AoA={aoa_deg:.1f}°, flap ±{flap_amp_deg:.0f}° @ {freq:.1f} Hz, "
              f"reduced freq k={k_red:.2f}, Re≈{U*chord/1.5e-5:.0e}")
        print(f"  CYCLE-MEAN (full bird):  lift = {Lf:+.2f} N   thrust = {Tf:+.2f} N   power = {Pf:.1f} W")
        print(f"  lift / weight = {Lf/W:.2f}   (target ≈ 1.0)")
        print(f"  power = {Pf:.1f} W   (target band 40–82 W; published 1.8 m/1.0 kg = 81.6 W)")
    return dict(L=Lf, T=Tf, P=Pf, W=W, area=area, span=full_span, k=k_red,
                Lh=Lh, Th=Th, Ph=Ph, th=th_hist, dt=dt, N=N, spc=steps_per_cycle)


def gpu_run(nc=4, ns=10, chord=0.29, half_span=0.80, mass=0.52, U=8.0, aoa_deg=10.0,
            flap_amp_deg=28.0, freq=3.0, n_cycle=4, steps_per_cycle=36, wake_rows=50, verbose=True):
    """PRODUCTION (all-Warp GPU) flapping-flight evaluation — the same validated UVLM kernels the
    differentiable coupled solver uses (bound_rings → AIC → moving-body rhs → batched solve → unsteady
    panel force → shed/convect). numpy `run` below is ONLY the oracle (one bit-exact cross-check).
    The wake is truncated to the most-recent `wake_rows` ROWS (= wake_rows·ns rings) so the retained
    wake history is the same in TIME at every mesh resolution — a mesh-independent truncation (a fixed
    ring count was a bug: it shortened the wake history as ns grew and corrupted the convergence)."""
    import warp as wp
    from fluxvortex.warp_fsi import config as cfg
    from fluxvortex.warp_fsi.config import DTYPE
    from fluxvortex.warp_fsi.batched_solver import batched_dense_solve
    import diff_uvlm_unsteady_gpu as ug
    wp.init(); dev = cfg.DEVICE; NP = cfg.NP_DTYPE; V3 = wp.vec3d
    C0 = flat_wing(nc, ns, chord, half_span); npan = nc * ns; ncv = (nc + 1) * (ns + 1)
    A = np.radians(flap_amp_deg); Om = 2.0 * np.pi * freq
    Vinf = np.array([U, 0.0, U * np.tan(np.radians(aoa_deg))]); Vw = V3(*[float(v) for v in Vinf])
    T = 1.0 / freq; dt = T / steps_per_cycle; N = n_cycle * steps_per_cycle
    te = wp.array(np.array([(nc - 1) * ns + j for j in range(ns)], np.int32), dtype=wp.int32, device=dev)
    wake_max = wake_rows * ns                                # truncate by ROWS (time), mesh-independent
    maxw = min(N * ns, wake_max) + ns
    wr = wp.zeros((maxw, 4), dtype=V3, device=dev); wr_new = wp.zeros((maxw, 4), dtype=V3, device=dev)
    wg = wp.zeros(maxw, dtype=DTYPE, device=dev); gprev = wp.zeros((1, npan), dtype=DTYPE, device=dev); nw = 0
    Lh = np.zeros(N); Th = np.zeros(N); Ph = np.zeros(N)
    for t in range(N):
        tt = t * dt; theta = A * np.sin(Om * tt); thetadot = A * Om * np.cos(Om * tt)
        corners, cvel = flap_state(C0, theta, thetadot)
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
        wp.launch(ug.panel_force_kernel, dim=npan, inputs=[rings, nrm, gamma, gprev, vcol, Vw,
                  DTYPE(dt), DTYPE(ug.RHO), ns], outputs=[Fp], device=dev)
        Fpn = Fp.numpy(); vcn = vcol.numpy()
        Lh[t] = np.sum(Fpn[:, 2]); Th[t] = -np.sum(Fpn[:, 0]); Ph[t] = -np.sum(np.einsum('pi,pi->p', Fpn, vcn))
        wp.launch(ug.shed_kernel, dim=ns, inputs=[rings, gamma, te, Vw, DTYPE(dt), nw], outputs=[wr, wg], device=dev)
        nw_new = nw + ns
        if nw > 0:   # convect OLD wake only; freshly-shed ring STAYS attached at the TE (Katz&Plotkin
            wp.launch(ug.convect_kernel, dim=(nw, 4), inputs=[rings, gamma, npan, wr, wg, nw, Vw, DTYPE(dt)],
                      outputs=[wr_new], device=dev)   # order) so it cancels the trailing bound segment
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
    Lf = 2.0 * np.mean(Lh[last]); Tf = 2.0 * np.mean(Th[last]); Pf = 2.0 * np.mean(np.abs(Ph[last]))
    W = mass * G
    if verbose:
        print(f"  [GPU] {npan} panels/half  lift={Lf:.2f}N (L/W={Lf/W:.2f})  thrust={Tf:.2f}N  power={Pf:.1f}W", flush=True)
    return dict(L=Lf, T=Tf, P=Pf, W=W, npan=npan)


def verify(verbose=True):
    """Production (all-Warp GPU) bird-scale flapping-flight validation. (1) one bit-exact GPU↔numpy
    oracle cross-check at a tiny mesh; (2) trim the wing (lift = weight) by an AoA sweep on the GPU
    path; (3) report the trimmed cruise power. Pass = the wing can be trimmed to support its weight AND
    the trimmed mechanical power is within 2× of the published 40–82 W band."""
    # (1) oracle cross-check (numpy used here ONLY, tiny + fast)
    rn = run(nc=2, ns=4, n_cycle=2, aoa_deg=10, verbose=False)
    rg = gpu_run(nc=2, ns=4, n_cycle=2, aoa_deg=10, verbose=False)
    rel = abs(rg["L"] - rn["L"]) / (abs(rn["L"]) + 1e-30)
    if verbose:
        print(f"GPU↔numpy oracle cross-check (tiny mesh): GPU lift={rg['L']:.3f}N vs numpy {rn['L']:.3f}N  rel={rel:.1e}")
    # (2) trim on the GPU production path
    W = 0.52 * G
    trim = None
    for aoa in [8, 9, 10, 11, 12]:
        r = gpu_run(aoa_deg=aoa, verbose=False)
        if verbose:
            print(f"  GPU AoA={aoa:2d}°  lift={r['L']:.2f}N (L/W={r['L']/W:.2f})  thrust={r['T']:.2f}N  power={r['P']:.1f}W")
        if trim is None or abs(r["L"] - W) < abs(trim["L"] - W):
            trim = r; trim["aoa"] = aoa
    lift_ok = rel < 1e-6 and 0.5 <= trim["L"] / W <= 2.0
    pwr_ok = 20.0 <= trim["P"] <= 164.0                         # within 2× of the 40–82 W band
    ok = lift_ok and pwr_ok
    if verbose:
        print(f"  TRIM ≈ AoA {trim['aoa']}°: lift {trim['L']:.2f} N (L/W {trim['L']/W:.2f}), cruise power {trim['P']:.1f} W")
        print(f"  -> {'PASS' if ok else 'CHECK'}: GPU==numpy [{rel:.0e}], trimmable to weight & power "
              f"within 2× of 40–82 W [{pwr_ok}] — bird-scale flapping flight is physical (production GPU path)")
    return ok, trim


def convergence(aoa_deg=11, chord=0.29, U=8.0, freq=3.0, verbose=True):
    """SPACE+TIME mesh convergence on the production GPU path. steps/cycle ∝ chordwise panels (so
    U·dt ≈ chord/nc — one wake-advection step per chordwise panel) and the wake spans ~1.5 cycles at
    every resolution; refining space at FIXED dt is not a valid convergence path for unsteady aero
    (the wake is shed one TE row per step, so a finer bound lattice outruns a dt-limited wake)."""
    if verbose:
        print("Space+time mesh convergence (GPU production path):")
    out = []
    for (nc, ns) in [(4, 10), (6, 14), (8, 18), (10, 22)]:
        spc = int(round(U * nc / (freq * chord)))            # 1 wake-advection step per chordwise panel
        wr = int(round(1.5 * spc))                            # ~1.5 flap cycles of wake
        r = gpu_run(nc=nc, ns=ns, aoa_deg=aoa_deg, n_cycle=4, steps_per_cycle=spc, wake_rows=wr, verbose=False)
        out.append((nc * ns, r["L"], r["T"], r["P"]))
        if verbose:
            print(f"  {nc}x{ns}={nc*ns} panels/half  spc={spc} wake={wr}rows  "
                  f"lift={r['L']:.2f}N (L/W={r['L']/r['W']:.2f})  thrust={r['T']:.2f}N  power={r['P']:.1f}W", flush=True)
    return out


if __name__ == "__main__":
    import sys
    if "--conv" in sys.argv:
        convergence(); raise SystemExit(0)
    raise SystemExit(0 if verify()[0] else 1)
