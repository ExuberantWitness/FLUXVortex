"""L-B S2/S3 dynamic-stall state module (2026-07-20, research_lb_formula.md A.2/A.3).

Per-strip dynamic-stall states advanced in lockstep with the UVLM time loop. Driven by the
strip effective angle of attack alpha_eff(t) from the UVLM (which already carries the
attached-flow circulatory unsteadiness via its wake induction — S1 decision: do NOT add a
Wagner indicial layer, it would double-count). This module supplies ONLY what the quasi-
steady v4 lacks: the TIME-LAGGED trailing-edge separation f2 (A.2, Tp/Tf) and the LEV
vortex lift CNv (A.3, Tv/Tvl).

A.2 trailing-edge separation (Bangga 2020 Eqs 10-19, zero-fit constants):
  - pressure lag (Tp=1.7, airfoil-independent per Leishman-Beddoes 1989):
        CNp1 = CNp - Dp;  Dp <- Dp*exp(-ds/Tp) + (CNp-CNp_prev)*exp(-ds/(2Tp))
  - lagged incidence alpha_f = alpha0 + CNp1/CLa
  - quasi-steady separation f_qs(alpha_f) from the static polar inversion (lb_static.S0)
  - separation-point lag (Tf=3.0): f2 = f_qs - Df; Df <- Df*exp(-ds/Tf)+(f_qs-f_qs_prev)*exp(-ds/(2Tf))
  - viscous normal force CNf = CLa*(alpha_eff-alpha0)*((1+sqrt(f2))/2)^2   [alpha_eff = UVLM input]
  - tangential (LE-suction) independent decay: CT = -eta*CLa*alpha_eff^2*sqrt(f2), eta=0.95
    (Bangga Eq.19; or Eq.70 static-polar CT lookup — selectable). This is disease#3's key.

A.3 LEV vortex lift (Bangga Eqs 21-27, onset via existing LESP_crit — zero-fit):
  - onset when |LESP| > LESP_crit (Ramesh route, airfoil/Re only — already in v4)
  - CV = CNc*(1-K), K=((1+sqrt(f2))/2)^2  (vortex lift = linear-minus-Kirchhoff)
  - CNv accumulates while tau_v < Tvl, pure decay after (rise-peak-DROP):
        CNv <- CNv*exp(-ds/Tv) + (CV-CV_prev)*exp(-ds/(2Tv));  Tv=6.0, Tvl=6-7
  - tau_v vortex clock (Eq.24/35): advances while supercritical, resets on reattach
  - disease#1 (dL/df) key: CNv accumulates more at high k (vortex sheds later) -> mean lift up.

Integration (research_lb_integration.md C): UVLM keeps attached circulatory + added mass;
this module REPLACES the static Kirchhoff scaling (geo_stall) with the lagged f2 dynamics on
the normal force, adds the CT suction decay, and adds CNv. No double-count (Wagner stripped).
Zero-fit: Tp/Tf/Tv/Tvl = NACA0012 literature defaults (declared inherited empiricism, Mert/
Pereira/LB sensitivity to be run); f_qs from S0 polar inversion; eta=0.95 Bangga; LESP_crit
existing. ds = 2*V*dt/c (semi-chord convective time)."""
from __future__ import annotations

import numpy as np

from lb_static import StaticPolar


class LBDynStrip:
    """One strip's L-B dynamic-stall state. step(alpha_eff, lesp, V, c, dt) advances one step."""

    def __init__(self, polar: StaticPolar, lesp_crit=0.18, eta=0.95,
                 Tp=1.7, Tf=3.0, Tv=6.0, Tvl=6.0):
        self.sp = polar
        self.cla = polar.cla
        self.a0 = polar.a0
        self.lesp_crit = lesp_crit
        self.eta = eta
        self.Tp, self.Tf, self.Tv, self.Tvl = Tp, Tf, Tv, Tvl
        # lag states
        self.Dp = 0.0          # pressure-lag deficiency
        self.CNp_prev = 0.0    # previous attached normal-force coeff (for Dp)
        self.Df = 0.0          # separation-point lag deficiency
        self.f_qs_prev = 1.0   # previous quasi-steady separation point
        self.f2 = 1.0          # current lagged separation point (output)
        # LEV states
        self.CNv = 0.0
        self.CV_prev = 0.0
        self.tau_v = 0.0
        self.lev_active_prev = False

    def _deficiency_update(self, D_prev, val, val_prev, T, ds):
        """First-order exponential deficiency (Duhamel discrete convolution, Bangga Eq.11/17)."""
        return D_prev * np.exp(-ds / T) + (val - val_prev) * np.exp(-ds / (2.0 * T))

    def step(self, alpha_eff, lesp, V, c, dt):
        """Advance one step. alpha_eff/lesp scalars (strip values), V=rel speed, c=strip chord.
        Returns dict(f2, CNf, CT, CNv, CN_total, f_qs, lev_active). All coeffs 2D (per q*c)."""
        ds = 2.0 * V * dt / c                                  # semi-chord convective time
        cla = self.cla

        # --- A.2 pressure lag -> lagged incidence -> quasi-steady separation ---
        CNp = cla * (alpha_eff - self.a0)                      # attached normal force (no indicial; UVLM has wake)
        self.Dp = self._deficiency_update(self.Dp, CNp, self.CNp_prev, self.Tp, ds)
        CNp1 = CNp - self.Dp
        alpha_f = self.a0 + CNp1 / cla
        f_qs = float(self.sp.f_inversion(np.array([alpha_f]))[0])
        self.Df = self._deficiency_update(self.Df, f_qs, self.f_qs_prev, self.Tf, ds)
        f2 = np.clip(f_qs - self.Df, 0.0, 1.0)
        self.CNp_prev = CNp
        self.f_qs_prev = f_qs
        self.f2 = f2

        # --- viscous normal force (Kirchhoff with lagged f2) + tangential suction decay ---
        CNf = cla * (alpha_eff - self.a0) * ((1.0 + np.sqrt(f2)) / 2.0) ** 2
        CT = -self.eta * cla * alpha_eff ** 2 * np.sqrt(f2)    # Bangga Eq.19 (drag-direction positive = -CT)

        # --- A.3 LEV vortex lift (onset via LESP_crit) ---
        lev_active = abs(lesp) > self.lesp_crit
        K = ((1.0 + np.sqrt(f2)) / 2.0) ** 2                   # Kirchhoff factor
        CNc = cla * (alpha_eff - self.a0)                      # circulatory (linear)
        CV = CNc * (1.0 - K)                                   # vortex lift source (linear minus Kirchhoff)
        # vortex clock (Eq.24/35): advances while supercritical, resets on downstroke reattach
        if lev_active:
            self.tau_v = self.tau_v + 0.45 * ds
        elif (not lev_active) and (alpha_eff < 0):             # reattach on negative-alpha downstroke
            self.tau_v = 0.0
        # CNv: accumulate while tau_v < Tvl, pure decay after (rise-peak-DROP)
        if 0.0 < self.tau_v < self.Tvl:
            self.CNv = self.CNv * np.exp(-ds / self.Tv) + (CV - self.CV_prev) * np.exp(-ds / (2.0 * self.Tv))
        else:
            self.CNv = self.CNv * np.exp(-ds / self.Tv)
        if not lev_active and not self.lev_active_prev:
            self.CNv = self.CNv * np.exp(-ds / self.Tv)        # ensure decay when fully subcritical
        self.CV_prev = CV
        self.lev_active_prev = lev_active

        CN_total = CNf + self.CNv
        return dict(f2=f2, f_qs=f_qs, CNf=CNf, CT=CT, CNv=self.CNv, CN_total=CN_total,
                    lev_active=lev_active, tau_v=self.tau_v)


if __name__ == "__main__":
    # canonical: harmonic alpha oscillation across stall -> f2 lag hysteresis + CNv rise-peak-drop
    sp = StaticPolar()
    strip = LBDynStrip(sp, lesp_crit=0.18)
    V, c, f = 8.0, 0.287, 2.3
    dt = 0.015 * c / V
    Om = 2 * np.pi * f
    a_mean, a_amp = np.radians(8.0), np.radians(8.0)   # 8+-8 deg straddles stall (~10deg)
    N = int(6 / f / dt)
    print(f"canonical: harmonic alpha 8+-8deg @f{f}, k={Om*c/(2*V):.3f}, straddling stall", flush=True)
    print(f"{'t*':>6}{'a_eff':>7}{'f_qs':>6}{'f2':>6}{'CNf':>7}{'CNv':>7}{'CNtot':>7}", flush=True)
    CNv_peak = 0.0
    for it in range(N):
        t = it * dt
        a = a_mean + a_amp * np.sin(Om * t)
        lesp = np.sin(a)                                       # kinematic LESP proxy (onset test)
        r = strip.step(a, lesp, V, c, dt)
        CNv_peak = max(CNv_peak, r["CNv"])
        if it > int(4/f/dt) and it % 30 == 0:                  # last cycles
            print(f"{t*V/c:>6.1f}{np.degrees(a):>7.1f}{r['f_qs']:>6.2f}{r['f2']:>6.2f}"
                  f"{r['CNf']:>7.2f}{r['CNv']:>7.2f}{r['CN_total']:>7.2f}", flush=True)
    print(f"\nS2/S3 canonical: CNv peak={CNv_peak:.2f} (LEV rises on up-stroke, expect >0); "
          f"f2<f_qs on up-stroke (lag=stall delay), f2>f_qs on down-stroke (reattachment lag) "
          f"= hysteresis loop direction check.", flush=True)
