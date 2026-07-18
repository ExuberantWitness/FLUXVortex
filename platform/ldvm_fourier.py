"""LDVM with a FOURIER (thin-airfoil) bound layer — line-by-line faithful port of UNSflow's
`ldvm` solver (KiranRamesh-Aero/UnsteadyFlowSolvers.jl) = Ramesh et al. 2014 (JFM 751 ==
NCSU thesis Ch.4). Case definitions and source audit: research_rvpm_s2cases.md. This is the
S2 reference testbed; platform/flap_ldvm.py (lumped-vortex) stays as the legacy strip closure.

Why not the lumped solver: n collocation DOFs respond POINTWISE to wake close encounters, so
grid refinement makes the force MORE singular (S2 scan: n=140 -> CL peak 164, non-convergent).
The published LDVM truncates the bound vorticity to naterm Fourier modes (35 << ndiv=70):
a near particle projects only weakly onto the low modes — built-in low-pass, refinement-stable.

Port map (UNSflow src file : here):
  theta grid incl. endpoints, x=c/2(1-cos th), trapz integrals  (typedefs TwoDSurf)
  update_downwash: W = -(U)sa - uind*sa + hdot*ca - wind*ca - adot*(x-pvt*c)
                   + dzc*(uind*ca + U*ca + hdot*sa - wind*sa)      [hdot UP-positive]
  KelvinCondition/KelvinKutta -> linear closed forms (system is linear in strengths)
  update_a2a3adot: RATES from the TEV-ONLY (pre-LEV) coefficients minus previous POST-LEV
                   values — during supercritical shedding a0dot = (a0_unpinned-crit)/dt > 0
                   (the sustained apparent-mass lift; pinning-then-differencing kills it)
  update_bv: segment circulations Uc[a0(1+cos)+sum an sin(n th) sin th]*dth (trapz, midpoints)
  place_tev: first = TE + (0.5*U*dt, 0); else 1/3 rule.  place_lev: first = LE + 0.5*v_le*dt
             (v_le = relative fluid velocity at LE incl. pitch-rate arm and induction),
             else 1/3 rule; 'first' whenever the previous step did NOT shed (levflag)
  wakeroll: mutual induction + bound-segment induction; convect AFTER solve, BEFORE next W
  calc_forces: cnc = 2pi*(U ca + hdot sa)/U*(A0+A1/2); cnnc = 2pi*c*(3/4 A0dot + 1/4 A1dot
               + 1/8 A2dot)/U; nonl = 2*sum((uind ca - wind sa)*bv)/(U^2 c); cs = 2pi A0^2;
               Cl = CN ca + CS sa; Cd = CN sa - CS ca
  Vatistas n=2 core, rc = 0.02c for TEV/LEV/bound alike (hardcoded in UNSflow)

Conventions here: world frame x downstream, y up; plate pivot recedes at -U, plunges +hdot
(UP-positive, same as UNSflow z-up hdot). Our wake kernel is ccw-positive; UNSflow's s is
cw-positive — wake strengths here = MINUS UNSflow's, bound segments passed as -bv to the
kernel. Kelvin kept in total form with a trimmed-circulation sink (== UNSflow incremental).
Zero-fit: all constants literature-anchored."""
from __future__ import annotations

import numpy as np

from flap_ldvm import _induced_many


class LDVM2D:
    def __init__(self, U=1.0, c=1.0, ndiv=70, naterm=35, dt=0.015, rho=1.0,
                 lesp_crit=0.11, camber_m=0.0, camber_p=0.40, pivot_xc=0.0,
                 core_rc=0.02, max_wake=100000):
        self.U, self.c, self.rho, self.dt = float(U), float(c), float(rho), float(dt)
        self.ndiv, self.naterm = int(ndiv), int(naterm)
        self.lesp_crit = float(lesp_crit)
        self.xp = float(pivot_xc) * self.c
        self.rc = float(core_rc) * self.c
        self.max_wake = int(max_wake)
        self.th = np.linspace(0.0, np.pi, self.ndiv)                 # incl. endpoints (UNSflow)
        self.xs = 0.5 * self.c * (1.0 - np.cos(self.th))
        xc = self.xs / self.c
        m, p = camber_m, camber_p
        self.dzc = np.where(xc < p, 2.0 * m / (p * p) * (p - xc),
                            2.0 * m / ((1.0 - p) ** 2) * (p - xc)) if m > 0 else np.zeros(self.ndiv)
        self.tx = []; self.ty = []; self.tg = []                     # TEV (ccw = -UNSflow s)
        self.lx = []; self.ly = []; self.lg = []                     # LEV
        self.it = 0
        self._lev_prev_it = -99                                      # levflag
        self.gam_lost = 0.0                                          # trimmed circulation sink
        self._AF_prev = np.zeros(4)                                  # previous POST-LEV A0..A3
        self.sx = 0.0; self.sy = 0.0                                 # pivot world position

    # ---------------------------------------------------------------- helpers
    def _world(self, x):
        return (self.sx + (x - self.xp) * self._ca,
                self.sy - (x - self.xp) * self._sa)

    def _a0(self, W):
        return -np.trapezoid(W, self.th) / (np.pi * self.U)

    def _an(self, W, n):
        return 2.0 * np.trapezoid(W * np.cos(n * self.th), self.th) / (np.pi * self.U)

    def _gamb(self, W):
        """Bound circulation (thin-airfoil positive = positive lift)."""
        return np.pi * self.c * self.U * (self._a0(W) + 0.5 * self._an(W, 1))

    def _wcol(self, px, py):
        """Downwash column of a unit-ccw vortex at the plate stations (role as in W)."""
        u, w = _induced_many(self._wx, self._wy, np.array([px]), np.array([py]),
                             np.array([1.0]), np.array([self.rc]))
        return -(u * self._sa + w * self._ca) + self.dzc * (u * self._ca - w * self._sa)

    # ---------------------------------------------------------------- step
    def step(self, alpha, dalpha, hdot=0.0):
        self.it += 1
        U, c, dt = self.U, self.c, self.dt
        ca, sa = np.cos(alpha), np.sin(alpha)
        self._ca, self._sa = ca, sa
        self.sx -= U * dt
        self.sy += hdot * dt
        self._wx, self._wy = self._world(self.xs)

        tvx = np.array(self.tx); tvy = np.array(self.ty); tvg = np.array(self.tg)
        lvx = np.array(self.lx); lvy = np.array(self.ly); lvg = np.array(self.lg)
        rct = np.full(len(tvx), self.rc); rcl = np.full(len(lvx), self.rc)

        # wake-induced velocities at the stations, then downwash (UNSflow update_downwash;
        # hdot UP-positive: +hdot*ca; camber term uses the full local chordwise velocity)
        uu, ww = _induced_many(self._wx, self._wy, tvx, tvy, tvg, rct)
        ul, wl = _induced_many(self._wx, self._wy, lvx, lvy, lvg, rcl)
        ui = uu + ul; wi = ww + wl
        W = (-U * sa - ui * sa + hdot * ca - wi * ca
             - dalpha * (self.xs - self.xp)
             + self.dzc * (ui * ca + U * ca + hdot * sa - wi * sa))

        # fresh TEV: first = TE + (0.5*U*dt, 0); else 1/3 rule (place_tev)
        tex, tey = self._world(c)
        if len(self.tx) > 0:
            ntx = tex + (self.tx[-1] - tex) / 3.0
            nty = tey + (self.ty[-1] - tey) / 3.0
        else:
            ntx, nty = tex + 0.5 * U * dt, tey
        tcol = self._wcol(ntx, nty)

        # Kelvin (total form; == UNSflow incremental): -Gamma_b + S_old + gT = 0
        S_old = float(np.sum(tvg) + np.sum(lvg)) + self.gam_lost
        G0 = self._gamb(W); GT = self._gamb(tcol)
        gT = (G0 - S_old) / (1.0 - GT)

        # PRE-LEV (TEV-only) coefficients -> rates (UNSflow update_a2a3adot semantics)
        W_pre = W + gT * tcol
        AF_pre = np.array([self._a0(W_pre), self._an(W_pre, 1),
                           self._an(W_pre, 2), self._an(W_pre, 3)])
        dAF = (AF_pre - self._AF_prev) / dt if self.it > 1 else np.zeros(4)
        a0 = AF_pre[0]                                               # LESP

        shed_lev = abs(a0) > self.lesp_crit
        gL = 0.0
        if shed_lev:
            lex, ley = self._world(0.0)
            if self._lev_prev_it == self.it - 1 and len(self.lx) > 0:
                nlx = lex + (self.lx[-1] - lex) / 3.0
                nly = ley + (self.ly[-1] - ley) / 3.0
            else:
                # place_lev: v_le = relative fluid velocity at the LE (kinematic + induced)
                s0 = 0.0 - self.xp
                le_u = U + s0 * sa * dalpha + float(ui[0])
                le_w = -hdot + s0 * ca * dalpha + float(wi[0])
                nlx, nly = lex + 0.5 * le_u * dt, ley + 0.5 * le_w * dt
            lcol = self._wcol(nlx, nly)
            # 2x2 (KelvinKutta, linear): Kelvin + A0 pinned at sign(a0)*crit
            GL = self._gamb(lcol)
            a0T = self._a0(tcol); a0L = self._a0(lcol)
            Amat = np.array([[1.0 - GT, 1.0 - GL], [a0T, a0L]])
            rhs = np.array([G0 - S_old, np.sign(a0) * self.lesp_crit - self._a0(W)])
            gT, gL = np.linalg.solve(Amat, rhs)
            self.lx.append(nlx); self.ly.append(nly); self.lg.append(float(gL))
            self._lev_prev_it = self.it
        self.tx.append(ntx); self.ty.append(nty); self.tg.append(float(gT))

        # final (POST-LEV) downwash, coefficients, and previous-value bookkeeping
        Wt = W + gT * tcol + (gL * lcol if shed_lev else 0.0)
        AF = np.array([self._a0(Wt), self._an(Wt, 1), self._an(Wt, 2), self._an(Wt, 3)])
        self._AF_prev = AF.copy()

        # bound segment circulations (update_bv): trapz of Uc[a0(1+cos)+sum an sin(n th) sin th]
        gam_th = AF[0] * (1.0 + np.cos(self.th))
        Wn = Wt / U
        for k in range(1, self.naterm + 1):
            ak = AF[k] if k <= 3 else 2.0 * np.trapezoid(Wn * np.cos(k * self.th), self.th) / np.pi
            gam_th = gam_th + ak * np.sin(k * self.th) * np.sin(self.th)
        gam_th = gam_th * U * c
        bv = 0.5 * (gam_th[1:] + gam_th[:-1]) * np.diff(self.th)     # thin-positive segments
        bx = 0.5 * (self._wx[1:] + self._wx[:-1]); by = 0.5 * (self._wy[1:] + self._wy[:-1])

        # forces (calc_forces): uind/wind at stations from the FULL wake (incl. fresh)
        tvx = np.array(self.tx); tvy = np.array(self.ty); tvg = np.array(self.tg)
        lvx = np.array(self.lx); lvy = np.array(self.ly); lvg = np.array(self.lg)
        rct = np.full(len(tvx), self.rc); rcl = np.full(len(lvx), self.rc)
        u2, w2 = _induced_many(self._wx, self._wy, tvx, tvy, tvg, rct)
        u3, w3 = _induced_many(self._wx, self._wy, lvx, lvy, lvg, rcl)
        u_wk = (u2 + u3) * ca - (w2 + w3) * sa
        cnc = 2.0 * np.pi * (U * ca + hdot * sa) / U * (AF[0] + 0.5 * AF[1])
        cnnc = 2.0 * np.pi * c * (0.75 * dAF[0] + 0.25 * dAF[1] + 0.125 * dAF[2]) / U
        nonl = 2.0 * float(np.sum(u_wk[:-1] * bv)) / (U * U * c)
        cn = cnc + cnnc + nonl
        cs = 2.0 * np.pi * AF[0] * AF[0]

        # convect wake (wakeroll): mutual + bound segments (ccw = -bv), then trim
        nt, nl = len(self.tx), len(self.lx)
        wxa = np.concatenate([tvx, lvx]); wya = np.concatenate([tvy, lvy])
        ub, wb = _induced_many(wxa, wya, bx, by, -bv, np.full(len(bv), self.rc))
        ut, wt = _induced_many(wxa, wya, tvx, tvy, tvg, rct)
        ulv, wlv = _induced_many(wxa, wya, lvx, lvy, lvg, rcl)
        uc = ub + ut + ulv; wc = wb + wt + wlv
        for i in range(nt):
            self.tx[i] += uc[i] * dt; self.ty[i] += wc[i] * dt
        for i in range(nl):
            self.lx[i] += uc[nt + i] * dt; self.ly[i] += wc[nt + i] * dt
        for arr_x, arr_y, arr_g in ((self.tx, self.ty, self.tg), (self.lx, self.ly, self.lg)):
            if len(arr_x) > self.max_wake:
                k = len(arr_x) - self.max_wake
                self.gam_lost += float(np.sum(arr_g[:k]))
                del arr_x[:k], arr_y[:k], arr_g[:k]

        # cumulative bound circulation up to x_ref (production Hirato-Eq.6 A0-extraction audit)
        xmid = 0.5 * (self.xs[1:] + self.xs[:-1])
        gcum01 = float(np.sum(bv[xmid <= 0.10 * c]))
        return dict(CLf=cn * ca + cs * sa, CDf=cn * sa - cs * ca, CNf=cn, CSf=cs,
                    A0=AF[0], AF=AF, dAF=dAF, gamb=float(np.sum(bv)), gcum01=gcum01,
                    lesp=a0, n_lev=len(self.lx), n_tev=len(self.tx))


if __name__ == "__main__":
    # sanity ladder: steady CL -> Wagner-limited 2*pi*sin(a); A0 -> sin(a); plunge-down lift +
    for ad in (5.0, 8.0):
        m = LDVM2D(U=1.0, c=1.0, dt=0.015, lesp_crit=99.0)
        a = np.radians(ad)
        for it in range(400):
            r = m.step(a, 0.0, 0.0)
        print(f"a={ad}: CLf={r['CLf']:+.3f} (2pi sa={2*np.pi*np.sin(a):+.3f}, Wagner@t*=6 ~0.91x)"
              f"  A0={r['A0']:+.4f} (sa={np.sin(a):+.4f})  Gb={r['gamb']:+.4f}", flush=True)
    m = LDVM2D(U=1.0, c=1.0, dt=0.015, lesp_crit=99.0)
    for it in range(200):
        r = m.step(0.0, 0.0, -0.05)                                  # steady plunge DOWN
    print(f"plunge-down hdot=-0.05: CLf={r['CLf']:+.3f} (expect ~ +2pi*0.05*Wagner=+0.28)",
          flush=True)
