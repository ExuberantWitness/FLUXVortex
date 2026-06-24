"""Flapping RoboEagle wing via strip-theory LDVM with leading-edge vortex shedding.
Each spanwise strip runs a 2D plunging+pitching FlapLDVM (flap_ldvm.py, validated: steady CL=2pi*a,
pitch/plunge thrust from the LEV). Strip kinematics: geometric pitch alpha = body_AoA + twist(y,t);
plunge hdot = y*thetadot (the +-45deg flap is a plunge of the strip); the section LIFT is along the
strip-local normal, tilted by the dihedral theta(t) -> world-vertical lift = lift*cos(theta).
Integrate over the span (rounded-tip chord distribution) and the cycle; x2 for both wings.

The LEV sustains the section lift at the high effective AoA the flap produces (where attached UVLM
collapsed to ~0.25x) and provides the leading-edge-suction thrust. Validate vs RoboEagle Fig 17/18/19.
NOTE: strip theory lacks the 3D downwash, so the ATTACHED part is over-predicted ~1+2/AR (~1.36 at
AR 5.6); the LEV part is more 2D. We report raw strip-LDVM here, then add the 3D correction next."""
import numpy as np
from flap_ldvm import FlapLDVM


def chord_at(y, chord=0.287, half_span=0.80):
    """RoboEagle planform: constant root chord, quarter-circle rounded tip over the last chord/2."""
    r = chord / 2.0
    y_round = half_span - r
    if y <= y_round:
        return chord
    d = y - y_round
    return 2.0 * np.sqrt(max(r * r - d * d, 0.0))


def flapping_wing(U=8.0, aoa_deg=5.0, flap_amp_deg=45.0, twist_amp_deg=0.0, twist_phase_deg=270.0,
                  freq=2.0, half_span=0.80, chord=0.287, ns=6, nc=40, n_cycle=4,
                  steps_per_cycle=60, lesp_crit=0.20, max_wake=120, dihedral_project=True,
                  dihedral_aoa=False, lev_shed=False, camber_m=0.02, camber_p=0.40, cd0=0.0,
                  ):
    Om = 2 * np.pi * freq
    A_f = np.radians(flap_amp_deg); A_t = np.radians(twist_amp_deg); phi = np.radians(twist_phase_deg)
    a_b = np.radians(aoa_deg)
    dt = (1.0 / freq) / steps_per_cycle
    # spanwise strip centers + chords + widths (uniform spanwise spacing)
    edges = np.linspace(0.0, half_span, ns + 1)
    ys = 0.5 * (edges[:-1] + edges[1:]); widths = np.diff(edges)
    chords = np.array([chord_at(y, chord, half_span) for y in ys])
    ldvms = [FlapLDVM(U=U, c=max(chords[k], 1e-3), n=nc, dt=dt, rho=1.225,
                      lesp_crit=lesp_crit, max_wake=max_wake, lev_shed=lev_shed,
                      camber_m=camber_m, camber_p=camber_p)
             for k in range(ns)]
    lift_key = "lift_p"      # first-principles attached pressure lift (real LEV via lev_shed if enabled)
    Lh = []; Th = []
    for it in range(n_cycle * steps_per_cycle):
        t = it * dt
        th = A_f * np.sin(Om * t); thd = A_f * Om * np.cos(Om * t)        # dihedral angle + rate
        Lt = 0.0; Tt = 0.0
        for k, y in enumerate(ys):
            if chords[k] < 1e-3:
                continue
            psi = A_t * (y / half_span) * np.sin(Om * t + phi)             # twist (pitch)
            psid = A_t * (y / half_span) * Om * np.cos(Om * t + phi)
            ab = a_b * np.cos(th) if dihedral_aoa else a_b                 # dihedral reduces body AoA (cos th)?
            alpha = ab + psi
            hdot = y * thd                                                # flap plunge of the strip
            r = ldvms[k].step(alpha, psid, hdot)
            cth = np.cos(th) if dihedral_project else 1.0
            # lift_p (instantaneous pressure normal force) has a RELIABLE cycle-mean under large plunge;
            # the x-impulse lift drifts when the wake is truncated (max_wake). Attached wing -> lift_p is
            # the complete lift (LEV off). thrust = LE suction (saturated at A0_crit) + N*sin(a).
            Lt += r[lift_key] * cth * widths[k]                           # dihedral-projected lift
            # sectional profile (viscous) drag: cd0 * q_local * c * width, opposing the local relative
            # wind (U, -hdot); its streamwise component = cd0*0.5*rho*U*Vloc*c*width reduces thrust.
            Vloc = np.hypot(U, hdot)
            Dpro = cd0 * 0.5 * 1.225 * U * Vloc * chords[k] * widths[k]
            Tt += r["thrust"] * widths[k] - Dpro
        if it >= (n_cycle - 1) * steps_per_cycle:
            Lh.append(Lt); Th.append(Tt)
    return dict(L=2.0 * np.mean(Lh), T=2.0 * np.mean(Th))                 # both wings


if __name__ == "__main__":
    GF = 9.81 / 1000.0
    print("RoboEagle flapping (strip-LDVM, lift_p + saturated LE suction) vs measured (8 m/s, both wings):", flush=True)
    print("  lift = instantaneous pressure (reliable cycle-mean); thrust = LE suction capped at A0_crit.", flush=True)
    print("  remaining gaps: flat-plate (no camber -> 0deg lift ~0 vs 2.9N); no stall (15deg over);"
          " no profile/induced drag (net-thrust vs data needs it).\n", flush=True)
    # 5 deg AoA, ~2.3 Hz sweep vs twist (Fig 18b/19b region); paper lift ~7.8N at 5deg untwisted
    cases = [
        ("5deg twist0  ", dict(aoa_deg=5.0, twist_amp_deg=0.0, freq=2.3), 7.80),
        ("5deg twist22 ", dict(aoa_deg=5.0, twist_amp_deg=22.5, freq=2.3), 7.80 * 1.078),
        ("0deg twist0  ", dict(aoa_deg=0.0, twist_amp_deg=0.0, freq=2.3), 2.90),
        ("10deg twist0 ", dict(aoa_deg=10.0, twist_amp_deg=0.0, freq=2.3), 12.11),
        ("15deg twist0 ", dict(aoa_deg=15.0, twist_amp_deg=0.0, freq=2.3), 14.27),
    ]
    AR = (2 * 0.80) / 0.287                      # aspect ratio ~5.6
    c3d = AR / (AR + 2.0)                         # finite-wing 3D-downwash correction (~0.74)
    print(f"  [attached unsteady LDVM, LEV off; 3D-downwash correction x{c3d:.2f} for AR={AR:.1f}]\n", flush=True)
    for name, kw, paper in cases:
        # suction saturated at A0_crit=0.20 (physical separation cap), no divergent discrete LEV shedding
        r = flapping_wing(lesp_crit=0.20, lev_shed=False, steps_per_cycle=120, max_wake=300, **kw)
        L3d = r["L"] * c3d
        print(f"  {name}: L_strip={r['L']:+6.2f}N  L_3dcorr={L3d:+6.2f}N (paper {paper:+6.2f}N, "
              f"ratio {L3d/paper:+.2f})  T={r['T']:+6.2f}N", flush=True)
    print("DONE", flush=True)
