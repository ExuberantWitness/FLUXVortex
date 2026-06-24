"""Clean bird-scale flapping aircraft, built on VALIDATED pieces (user-chosen approach).

Build order (each step validated before the next):
  STEP 1 (this file, trim_check): calibrated force balance at the spec point — rigid flapping wings
    (flap_flight_validate.gpu_run, the validated UVLM) + a high-AoA tail, NO FSI yet. Validates the
    CALIBRATION (mass 1.5-2 kg, V~10 m/s, wing 5%, tail 20-40% lift) that earlier piecemeal guesses
    got wrong: does total lift = weight with the tail carrying the researched 20-40%?
  STEP 2: replace rigid wings with the validated strong-coupled ANCF FSI (Aitken) + prescribed flapping.
  STEP 3: 6-DOF body + V-tail UVLM; trim in flight.
  STEP 4: per-element stiffness/mass design + gust + co-design.

Spec (research_notes_wing_params.md): span ~1.8 m, chord 0.29 m, wing area ~0.49 m², total 1.5-2.0 kg,
cruise ~10 m/s, body AoA ~45°, wing feathered to ~6° (attached), flap 2.5 Hz ±25°, tail 25-40% wing area.
"""
from __future__ import annotations

import numpy as np

import flap_flight_validate as ffv

G = 9.81
RHO = 1.225


def flat_plate_CL_CD(alpha_rad):
    """Flat-plate high-AoA aero (Newtonian/post-stall): CL=sin(2α), CD=2 sin²α. Valid for the tail at
    the ~45° body AoA where thin-airfoil theory fails (the tail is a stalled lifting plate)."""
    return np.sin(2 * alpha_rad), 2 * np.sin(alpha_rad) ** 2


def tail_lift(S_tail, alpha_tail_rad, U, dihedral_deg=40.0):
    """Vertical lift from a V-tail at its AoA. Stalled flat plate; the V-dihedral reduces the vertical
    lift component by cos²Γ (the horizontal components of the two panels cancel). Returns (L_vert, D)."""
    q = 0.5 * RHO * U ** 2
    CL, CD = flat_plate_CL_CD(alpha_tail_rad)
    c2 = np.cos(np.radians(dihedral_deg)) ** 2
    L = q * S_tail * CL * c2
    D = q * S_tail * CD
    return L, D


def trim_check(total_mass=1.7, U=10.0, body_aoa_deg=45.0, wing_aoa_deg=13.0,
               chord=0.29, half_span=0.85, flap_amp_deg=25.0, freq=2.5,
               tail_area_frac=0.30, tail_dihedral_deg=40.0, tail_incidence_deg=0.0, verbose=True):
    """Force balance at the spec point. Wing: validated flapping UVLM at the attached local AoA
    (~13° near the attached limit, where the high-lift 45° regime puts it — NOT feathered to ~0).
    Tail: stalled V-tail plate at (body AoA − incidence), vertical lift reduced by cos²(dihedral).
    Check total vertical lift vs weight and the tail's lift fraction (researched 20-40%)."""
    W = total_mass * G
    S_wing = 2.0 * chord * half_span                 # both wings
    S_tail = tail_area_frac * S_wing
    # wing: validated rigid flapping UVLM at the attached local AoA -> both-wing lift
    rg = ffv.gpu_run(nc=4, ns=10, chord=chord, half_span=half_span, mass=total_mass, U=U,
                     aoa_deg=wing_aoa_deg, flap_amp_deg=flap_amp_deg, freq=freq,
                     n_cycle=4, steps_per_cycle=36, verbose=False)
    L_wing = rg["L"]; P_wing = rg["P"]
    # tail: stalled V-tail plate at (body AoA − tail incidence)
    L_tail, D_tail = tail_lift(S_tail, np.radians(body_aoa_deg - tail_incidence_deg), U,
                               dihedral_deg=tail_dihedral_deg)
    L_tot = L_wing + L_tail
    tail_frac = L_tail / (L_tot + 1e-30)
    if verbose:
        print(f"  spec: m={total_mass}kg W={W:.1f}N  V={U}m/s  body_aoa={body_aoa_deg}deg  "
              f"wing_feather={wing_feather_deg}deg  flap {freq}Hz ±{flap_amp_deg}deg", flush=True)
        print(f"  wing area={S_wing:.3f}m^2  tail area={S_tail:.3f}m^2 ({tail_area_frac*100:.0f}% wing)", flush=True)
        print(f"  L_wing={L_wing:.2f}N  L_tail={L_tail:.2f}N  L_total={L_tot:.2f}N  (W={W:.1f}N, L/W={L_tot/W:.2f})", flush=True)
        print(f"  tail lift fraction = {tail_frac*100:.0f}%  (researched 20-40%)", flush=True)
        print(f"  wing mech power={P_wing:.1f}W (published band 40-82W)", flush=True)
    trimmed = 0.85 < L_tot / W < 1.20
    tail_ok = 0.15 < tail_frac < 0.45
    print(f"  -> {'TRIM OK' if trimmed else 'NOT TRIMMED'}; tail fraction {'OK' if tail_ok else 'OFF'}", flush=True)
    return dict(L_wing=L_wing, L_tail=L_tail, L_tot=L_tot, W=W, tail_frac=tail_frac,
                P_wing=P_wing, trimmed=trimmed, tail_ok=tail_ok)


if __name__ == "__main__":
    print("STEP 1: calibrated trim/force-balance at the spec point (validated rigid flapping UVLM + tail)", flush=True)
    print("--- sweep cruise speed to find the trim point (L=W) at 1.7 kg ---", flush=True)
    for U in (8.0, 10.0, 12.0):
        trim_check(total_mass=1.7, U=U, tail_area_frac=0.30)
        print("", flush=True)
    print("--- 2.0 kg at V=11 ---", flush=True)
    trim_check(total_mass=2.0, U=11.0, tail_area_frac=0.30)
    print("DONE", flush=True)
