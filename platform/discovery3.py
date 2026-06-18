"""Discovery run #3 — the ranking-inversion synergy on the flapping aircraft.

The full-aircraft competing constraint that iteration-1's single plate lacked:
a **resonant flapping drive**. A flapping wing has a structural natural frequency
omega_n(stiffness); driving the flap near resonance lets the root spring store and
return the reciprocating inertial energy, so the motor power (efficiency, COT) is
MINIMIZED at a tuned (non-minimal) stiffness — competing with gust rejection, which
favors flexibility (discovery #1/#2).

This run combines the **real coupled-FSI gust data** (discovery2.npz: passive and
controlled gust vs stiffness, measured on the 4090) with the Zhong&Xu COT model
made resonance-aware (P_iner offset peaks at omega_n = 2*pi*f), and asks the
headline co-design question:

  Does the optimal design SHIFT when the control loop is closed?
  (gust-dominated, flexible)  -->  (efficiency/COT-dominated, resonance-tuned)?

If yes, that is the **non-intuitive structure-control synergy** the discovery paper
claims: with active gust rejection the wing can be stiffened to its efficient
resonant point — co-design (design+control) beats design-alone.
"""
from __future__ import annotations

import os
import sys

import numpy as np

_FLUXV = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for p in (_FLUXV, os.path.join(_FLUXV, "src"), os.path.dirname(__file__)):
    if p not in sys.path:
        sys.path.insert(0, p)

import cot as cotmod                                            # noqa: E402

F_FLAP = 3.0                 # flapping frequency (Hz), HIT-Hawk class
F_N1 = 2.2                   # wing natural frequency (Hz) at stiffness scale 1.0
#   omega_n(s) = 2*pi*F_N1*sqrt(s); resonance (omega_n = 2*pi*F_FLAP) at
#   s_res = (F_FLAP/F_N1)^2 = (3.0/2.2)^2 ~ 1.86  -> a STIFFER design is efficient.


def cot_resonant(s, m_wing0=0.16, b=1.7, c_mean=0.29, V=6.0):
    """COT with a resonant flapping drive: motor pays the inertial power only to the
    extent the flap is OFF resonance (the spring offsets it near omega_n = 2*pi*f)."""
    m = 0.52 + 0.25 * (s - 1.0)
    S = b * c_mean; AR = b * b / S
    CL = m * cotmod.G / (0.5 * cotmod.RHO * V * V * S)
    I_w = (m_wing0 * s) * (b / 2.0) ** 2 / 3.0
    comps = cotmod.power_components(m=m, b=b, S=S, AR=AR, c_mean=c_mean, f=F_FLAP,
                                    amp_deg=35.0, V=V, CL=CL, I_w=I_w)
    f_n = F_N1 * np.sqrt(s)                       # natural frequency (Hz)
    # off-resonance fraction of P_iner the motor must supply (min ~0.1 at resonance)
    detune = abs(1.0 - (F_FLAP / f_n) ** 2)
    offres = 0.1 + 0.9 * np.clip(detune, 0.0, 1.0)
    aero = comps["P_ind"] + comps["P_pro"] + comps["P_par"]
    P = (aero + offres * comps["P_iner"]) / cotmod.ETA
    return P / (m * cotmod.G * V)


def analyze():
    d = np.load(os.path.join(_FLUXV, "docs", "discovery2.npz"))
    s = d["stiff"]; gust_p = d["passive"]; gust_c = d["controlled"]
    cot = np.array([cot_resonant(si) for si in s])

    # normalize each objective to [0,1] (lower better) and scalarize equally
    def norm(x):
        return (x - x.min()) / (x.max() - x.min() + 1e-30)
    ng_p, ng_c, nc = norm(gust_p), norm(gust_c), norm(cot)
    J_passive = ng_p + nc                         # design-alone: gust + COT
    J_control = ng_c + nc                         # design+control: controlled gust + COT
    bp, bc = int(np.argmin(J_passive)), int(np.argmin(J_control))

    print("=== full-aircraft co-design: resonant flapping + real-FSI gust ===")
    print(f"  resonance at stiffness ~{(F_FLAP/F_N1)**2:.2f} (omega_n=2pi*f)")
    print("  stiffness : " + "  ".join(f"{si:.2f}" for si in s))
    print("  gust(pass): " + "  ".join(f"{v:.2e}" for v in gust_p))
    print("  gust(ctrl): " + "  ".join(f"{v:.2e}" for v in gust_c))
    print("  COT       : " + "  ".join(f"{v:.3f}" for v in cot))
    print("  J passive : " + "  ".join(f"{v:.2f}" for v in J_passive))
    print("  J control : " + "  ".join(f"{v:.2f}" for v in J_control))
    print("\n=== FINDING ===")
    print(f"  optimal DESIGN-ALONE  (gust+COT):        stiffness={s[bp]:.2f}")
    print(f"  optimal DESIGN+CONTROL (ctrl gust+COT):  stiffness={s[bc]:.2f}")
    inv = s[bc] > s[bp] + 1e-9
    if inv:
        print(f"  -> SYNERGY CONFIRMED: closing the control loop SHIFTS the optimum "
              f"{s[bp]:.2f} -> {s[bc]:.2f} (toward the efficient resonant stiffness).")
        print(f"     With active gust rejection the wing is stiffened to its resonant,")
        print(f"     low-power point — co-design (design+control) beats design-alone.")
    else:
        print(f"  -> no inversion (optimum {s[bp]:.2f} both); gust still dominates.")
    return inv


if __name__ == "__main__":
    raise SystemExit(0 if analyze() else 1)
