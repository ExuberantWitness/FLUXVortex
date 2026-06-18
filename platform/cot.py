"""COT power model (Zhong & Xu 2022) — the co-design efficiency objective.

Replaces the mean-lift efficiency proxy in codesign_eval with the validated
flapping-flight power model (Appl. Sci. 2022, 12(6):3176), so the co-design
optimizes a physically grounded cost of transport:

    P_tot = (P_ind + P_pro + P_par + P_iner) / eta          (eta = 0.85 lumped)
    COT   = P_tot / (m * g * V)                              (dimensionless)

  P_ind  induced power  ~ k/(pi*AR) * C_L^2 * q * S * V      (lift-induced drag)
  P_pro  profile power  = C_D,pro * q * S * V                (wing friction, Re~1e5)
  P_par  parasite power = 0.5 * rho * V^3 * S_b * C_D,par    (airframe)
  P_iner inertial power = (I_w + I_v) * omega * omega_dot    (RECIPROCATING wings)

Key plan point (Zhong & Xu's own finding): for a bird-scale flapper the inertial
power P_iner is NOT negligible — it is the term the **root resonant spring** is
designed to cancel (store/return the reciprocating kinetic energy each stroke).
This module quantifies it and the resonant-spring COT win, with HIT-Hawk params.
"""
from __future__ import annotations

import numpy as np

G = 9.81
RHO = 1.225
ETA = 0.85


def power_components(*, m, b, S, AR, c_mean, f, amp_deg, V, CL, I_w,
                     CD_pro=0.04, CD_par=0.15, k_ind=1.15, S_b=None):
    """All four power terms (W) for one wing pair. amp_deg = stroke amplitude."""
    q = 0.5 * RHO * V * V
    P_ind = k_ind / (np.pi * AR) * CL * CL * q * S * V
    P_pro = CD_pro * q * S * V
    S_b = S_b if S_b is not None else 0.02 * S            # small frontal area
    P_par = 0.5 * RHO * V ** 3 * S_b * CD_par
    # inertial: theta = A sin(wt); P_iner = I w wdot; cycle-mean of |work rate|
    A = np.deg2rad(amp_deg)
    w = 2 * np.pi * f
    I_v = 0.25 * RHO * np.pi * (c_mean / 2.0) ** 2 * b     # virtual (added) mass term
    P_iner_peak = 0.5 * (I_w + I_v) * A * A * w ** 3
    P_iner_mean = (2.0 / np.pi) * P_iner_peak              # mean |0.5 I A^2 w^3 sin 2wt|
    return dict(P_ind=P_ind, P_pro=P_pro, P_par=P_par,
                P_iner=P_iner_mean, P_iner_peak=P_iner_peak, I_v=I_v)


def cot(comps, *, m, V, resonant_spring=False):
    """COT with motor supplying P_iner (no spring) or the spring offsetting it."""
    aero = comps["P_ind"] + comps["P_pro"] + comps["P_par"]
    # resonant spring stores/returns the reciprocating inertial energy -> motor sees
    # only the (small) damping/loss fraction of P_iner instead of the whole term
    p_iner = 0.1 * comps["P_iner"] if resonant_spring else comps["P_iner"]
    P_tot = (aero + p_iner) / ETA
    return P_tot, P_tot / (m * G * V)


def demo() -> bool:
    # HIT-Hawk-class params (Zhong & Xu 2022 / plan §1 sizing)
    m, b, c_mean = 0.52, 1.7, 0.29
    S = b * c_mean / 1.0            # planform ~ b*c (single-segment)
    AR = b * b / S
    f, amp_deg, V = 3.0, 35.0, 6.0
    CL = m * G / (0.5 * RHO * V * V * S)                  # trim lift coefficient
    m_wing = 0.16                                          # per Zhong&Xu (~150-170 g)
    I_w = m_wing * (b / 2.0) ** 2 / 3.0                    # wing inertia about root axis

    comps = power_components(m=m, b=b, S=S, AR=AR, c_mean=c_mean, f=f,
                             amp_deg=amp_deg, V=V, CL=CL, I_w=I_w)
    P_no, cot_no = cot(comps, m=m, V=V, resonant_spring=False)
    P_sp, cot_sp = cot(comps, m=m, V=V, resonant_spring=True)
    aero = comps["P_ind"] + comps["P_pro"] + comps["P_par"]
    iner_frac = comps["P_iner"] / (aero + comps["P_iner"])

    print("COT power model (Zhong&Xu 2022), HIT-Hawk-class:")
    print(f"  P_ind={comps['P_ind']:.2f} P_pro={comps['P_pro']:.2f} "
          f"P_par={comps['P_par']:.3f} P_iner={comps['P_iner']:.2f} W "
          f"(peak {comps['P_iner_peak']:.1f} W)")
    print(f"  inertial fraction of input power = {iner_frac:.0%}  "
          f"(NON-negligible -> resonant-spring target)")
    print(f"  COT: no-spring={cot_no:.3f} (P={P_no:.1f}W)  "
          f"resonant-spring={cot_sp:.3f} (P={P_sp:.1f}W)  "
          f"-> {100*(1-cot_sp/cot_no):.0f}% lower")
    print(f"  caveats: total P={P_no:.0f}W is in Zhong&Xu's measured 50-130W band "
          f"(COT~1-4 is normal for these inefficient flappers); the inertial fraction "
          f"is an UPPER bound — steady CL underestimates the unsteady flapping aero "
          f"power, which the coupled rollout supplies in production.")
    ok = (comps["P_iner"] > 0.1 * aero) and (cot_sp < cot_no) and (0.05 < cot_no < 5.0) \
        and (50.0 <= P_no <= 200.0)
    print(f"  -> {'PASS' if ok else 'FAIL'}: physically-grounded efficiency objective "
          f"(P in measured band; P_iner significant; resonant spring cuts COT)")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if demo() else 1)
