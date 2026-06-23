"""2D plunging+pitching LDVM with leading-edge vortex-particle shedding — a faithful Python port of
the user's FW/wingDVM/oscillation.m, GENERALIZED to add PLUNGE (heave), which a flapping-wing strip
needs (the +-45deg flap is dominantly a plunge, not a pitch).

Mechanism (exactly oscillation.m): a flat-plate lumped-vortex thin airfoil (n bound vortices at the
1/4-panel, collocation at 3/4-panel). Every step a trailing-edge vortex PARTICLE (TEV) is shed just
aft of the TE (Kelvin), and — when the leading-edge suction exceeds a threshold — a leading-edge
vortex PARTICLE (LEV) is shed just ahead of the LE. Both wakes convect freely with the local induced
velocity (Biot-Savart), i.e. the LE "sprays vortex particles just like the TE wake" (user's words).

Force = unsteady vortex-impulse force (oscillation.m lines 115-198): rate of change of the bound
impulse + each wake particle's force rho*(V_induced x Gamma). The LEV particles' contribution to the
CHORDWISE force IS the leading-edge suction = the thrust (user: "thrust is basically the LE-suction
component"). No empirical correction model (no Leishman-Beddoes) — this is pure unsteady potential flow
with discrete LE/TE vortex shedding.

Validation ladder (self-test): (1) steady small-alpha -> CL ~ 2*pi*alpha; (2) pure pitch alpha=30deg
-> the FW reference's net thrust from the LEV; (3) pure plunge small-amp -> Garrick/Theodorsen thrust.
Then driven per spanwise strip for the RoboEagle flapping wing (separate driver).
"""
from __future__ import annotations

import numpy as np

SIGMA = 0.005       # Chorin vortex-core (vor2d.m)
RMIN = 0.001


def _vor2d(x, y, x1, y1, gamma):
    rx, ry = x - x1, y - y1
    r = np.hypot(rx, ry)
    if r <= RMIN:
        return 0.0, 0.0
    v = 0.5 * gamma / np.pi * (r / (r * r + SIGMA * SIGMA))
    return v * (-ry / r), v * (rx / r)


def _induced(px, py, vx, vy, vg):
    """Vectorized Biot-Savart: velocity at (px,py) from vortex particles (vx,vy,vg arrays)."""
    if len(vx) == 0:
        return 0.0, 0.0
    rx = px - vx; ry = py - vy
    r2 = rx * rx + ry * ry
    r = np.sqrt(r2)
    mask = r > RMIN
    v = np.where(mask, 0.5 * vg / np.pi * (r / (r2 + SIGMA * SIGMA)) / np.where(mask, r, 1.0), 0.0)
    return float(np.sum(v * (-ry))), float(np.sum(v * rx))


class FlapLDVM:
    """One 2D strip. step(alpha, dalpha, hdot) advances one dt with plunge velocity hdot.
    Returns sectional lift (normal) and chordwise force; thrust = -Fx (LE suction is in here)."""

    def __init__(self, U=1.0, c=1.0, n=80, dt=None, rho=1.225,
                 lesp_crit=0.20, alpha_lev_deg=None, max_wake=300, lev_shed=True,
                 camber_m=0.0, camber_p=0.40, dynamic_stall=False, kv=1.0):
        self.U, self.c, self.n = float(U), float(c), int(n)
        self.dt = dt if dt else c / U / 50.0
        self.rho = rho
        self.lesp_crit = lesp_crit                       # LESP threshold: suction saturates here; if
        self.lev_shed = lev_shed                         # lev_shed -> also shed discrete LEV particles
        self.alpha_lev = np.radians(alpha_lev_deg) if alpha_lev_deg else None
        self.dynamic_stall = dynamic_stall               # Polhamus LE-suction-recovery dynamic-stall lift
        self.kv = kv                                     # vortex-lift gain (1.0 = exact suction recovery)
        self.max_wake = int(max_wake)
        self.dl = self.c / self.n
        self.pvor = (np.arange(self.n) + 0.25) * self.dl     # 1/4-panel bound vortices
        self.pcol = (np.arange(self.n) + 0.75) * self.dl     # 3/4-panel collocation
        # NACA 4-digit camber slope dzc/dx at the collocation points (real RoboEagle section = NACA-2406);
        # enters the downwash BC like a chord-wise twist -> gives the zero-lift angle (0deg lift offset).
        xc = self.pcol / self.c; m, p = camber_m, camber_p
        self.dzc = np.where(xc < p, 2.0 * m / (p * p) * (p - xc),
                            2.0 * m / ((1.0 - p) ** 2) * (p - xc)) if m > 0 else np.zeros(self.n)
        self.Abb = np.zeros((self.n, self.n))                # bound-bound (plate frame)
        for i in range(self.n):
            for j in range(self.n):
                _, w = _vor2d(self.pcol[i], 0.0, self.pvor[j], 0.0, 1.0)
                self.Abb[i, j] = w
        self.phi = self.U * self.dt / self.dl
        self.tx = []; self.ty = []; self.tg = []             # TEV particles
        self.lx = []; self.ly = []; self.lg = []             # LEV particles
        self.it = 0
        self._lesp_old = 0.0                                 # for the up-stroke shed gate (dLESP/dt>0)
        self.gammaold = 0.0                                  # total bound circ (Kelvin)
        self.gprev = np.zeros(self.n)                        # gammas at previous step (for dGamma/dt)
        self.gcum_old = np.zeros(self.n)                     # cumulative bound (for pressure lift)
        self.sx = 0.0; self.sy = 0.0                         # LE world position (recedes -U, plunges +h)

    def _theta(self):
        return np.arccos(np.clip(1.0 - 2.0 * self.pvor / self.c, -1.0, 1.0))

    def _lesp(self, wx):
        th = self._theta(); dth = np.gradient(th)
        return -1.0 / np.pi * np.sum(dth * wx / self.U)

    def step(self, alpha, dalpha, hdot=0.0):
        self.it += 1
        U, c, n, dt = self.U, self.c, self.n, self.dt
        ca, sa = np.cos(alpha), np.sin(alpha)
        self.sx -= U * dt                                    # plate recedes in -x (freestream +U)
        self.sy += hdot * dt                                 # PLUNGE: LE plunges with the strip
        sx, sy = self.sx, self.sy
        dxw = 0.3 * U * dt

        # shed a TEV particle just aft of the TE (world frame, at the plunged/pitched plate)
        self.tx.append((c + dxw) * ca + sx); self.ty.append(-(c + dxw) * sa + sy); self.tg.append(0.0)
        tvx = np.array(self.tx); tvy = np.array(self.ty); tvg = np.array(self.tg)
        lvx = np.array(self.lx); lvy = np.array(self.ly); lvg = np.array(self.lg)

        # collocation world positions
        xn = sx + self.pcol * ca; yn = sy - self.pcol * sa
        # RHS: freestream + pitch-rate + PLUNGE + CAMBER slope + existing wake downwash (on plate normal)
        rhs_b = -U * sa - dalpha * self.pcol - hdot * ca + U * ca * self.dzc   # camber dzc/dx (Ramesh 2.17)
        for i in range(n):
            uu, ww = _induced(xn[i], yn[i], tvx[:-1], tvy[:-1], tvg[:-1])   # old TEV (exclude fresh)
            ul, wl = _induced(xn[i], yn[i], lvx, lvy, lvg)                  # LEV
            rhs_b[i] += -(uu + ul) * sa - (ww + wl) * ca

        # LESP criterion -> shed an LEV ONLY while the LE suction is high AND GROWING (up-stroke):
        # DVM.m gates by (alpha>thresh AND dalpha>0) so the LEV grows on the up-stroke then detaches on
        # the down-stroke (no new shedding) -> rise-peak-drop. Shedding every step = continuous deep stall.
        lesp = self._lesp(rhs_b)
        dlesp = lesp - self._lesp_old; self._lesp_old = lesp
        if self.alpha_lev is not None:
            shed_lev = abs(alpha) > self.alpha_lev and dalpha > 0
        else:
            shed_lev = abs(lesp) > self.lesp_crit and lesp * dlesp > 0   # |LESP| increasing
        shed_lev = shed_lev and self.lev_shed     # discrete LEV particles optional (suction still caps)
        if shed_lev:
            self.lx.append((-dxw) * ca + sx); self.ly.append(-(-dxw) * sa + sy); self.lg.append(0.0)

        # assemble [bound | TEV | (LEV)] with Kelvin + LE-suction-cap rows
        m = n + 1 + (1 if shed_lev else 0)
        A = np.zeros((m, m)); rhs = np.zeros(m)
        A[:n, :n] = self.Abb
        for i in range(n):
            _, wt = _vor2d(self.pcol[i], 0.0, c + dxw, 0.0, 1.0)
            A[i, n] = wt
            if shed_lev:
                _, wl = _vor2d(self.pcol[i], 0.0, -dxw, 0.0, 1.0)
                A[i, n + 1] = wl
            rhs[i] = rhs_b[i]
        A[n, :n] = 1.0; A[n, n] = 1.0                        # Kelvin: sum(bound) + TEV = old total
        if shed_lev:
            A[n, n + 1] = 1.0
        rhs[n] = self.gammaold
        if shed_lev:
            # Ramesh LESP modulation: HOLD A0 at the critical value (NOT zero the LE vortex). A0 is a
            # linear functional of the downwash; the shed TEV/LEV reduce it. Constraint: A0_post = A0_crit.
            dth = np.gradient(self._theta())
            lcol = lambda col: -1.0 / np.pi * np.sum(dth * col / U)   # LESP of a downwash column
            A[n + 1, n] = -lcol(A[:n, n]); A[n + 1, n + 1] = -lcol(A[:n, n + 1])
            rhs[n + 1] = self.lesp_crit * np.sign(lesp) - lesp        # bring A0 down to +/- A0_crit
        gam = np.linalg.solve(A, rhs)
        gb = gam[:n]
        self.tg[-1] = float(gam[n])
        if shed_lev:
            self.lg[-1] = float(gam[n + 1])
        tvx = np.array(self.tx); tvy = np.array(self.ty); tvg = np.array(self.tg)
        lvx = np.array(self.lx); lvy = np.array(self.ly); lvg = np.array(self.lg)

        xv = sx + self.pvor * ca; yv = sy - self.pvor * sa   # bound-vortex world positions
        dG = (gb - self.gprev) / dt

        # ---- normal force N (unsteady pressure, Bernoulli) + leading-edge SUCTION (analytic) ----
        # Canonical Ramesh-2014/2020 LDVM force split: the flat-plate pressure gives the plate-NORMAL
        # force N; the streamwise THRUST is the leading-edge SUCTION, which the discrete bound vortices
        # do NOT resolve (the LE sqrt(x) velocity singularity) and so MUST be added analytically.
        #   Ramesh 2020 JFM 886 A13: S = lim_{x->0} (1/2) gamma sqrt(x) = U*sqrt(c)*A0  (eqn 3.3),
        #   Garrick 1937 / von Karman-Burgers: suction force  F_s = pi*rho*S^2 = pi*rho*U^2*c*A0^2.
        N = 0.0                                              # plate-normal force (code sign: Fz=-N*ca)
        gcum = np.cumsum(gb)
        for i in range(n):
            uu, ww = self._vel_at(xv[i], yv[i], gb, xv, yv, tvx, tvy, tvg, lvx, lvy, lvg, U,
                                  freestream=False)          # wake-induced velocity at bound vortex i
            u_ch = uu * ca - ww * sa                         # chordwise component of wake-induced velocity
            dp = self.rho * ((U * ca + u_ch) * gb[i] / self.dl + (gcum[i] - self.gcum_old[i]) / dt)
            N += dp * self.dl
        # A0 = LESP; while an LEV is shedding it is HELD at A0_crit -> the LE suction is CAPPED (the
        # excess goes into the shed LEV, Polhamus). The detached-LEV thrust then enters through N: the
        # LEVs' induced downwash enhances the bound circulation/pressure -> larger N -> N*sin(a) thrust.
        # The LE suction SATURATES at A0_crit: beyond the critical LESP the flow separates (LEV), so the
        # suction cannot grow without bound (the excess goes into the LEV, Polhamus). This caps the thrust
        # physically WITHOUT needing the (divergent at large amplitude) discrete LEV shedding.
        A0_eff = max(-self.lesp_crit, min(self.lesp_crit, lesp))
        Fs = np.pi * self.rho * U * U * c * A0_eff * A0_eff  # LE suction (>=0), along chord toward LE
        Fz = -N * ca                                         # pressure lift (secondary; primary = Fz_imp)
        # Garrick thrust = F_s*cos(a) - N_phys*sin(a); code N=-N_phys -> thrust = Fs*ca + N*sa.
        # This is the PRIMARY thrust: exact for attached flow (steady flat plate -> 0, d'Alembert), and
        # with A0 wake-corrected (rhs_b includes the wake downwash) it gives the right plunge/Knoller-Betz
        # thrust scaling with reduced frequency. While an LEV sheds the suction is capped at A0_crit, so
        # this UNDER-predicts the deep-stall LEV thrust (a flat-plate-normal limit) -> see thrust_lev.
        thrust = Fs * ca + N * sa
        Fx = -thrust

        # ---- dynamic-stall LIFT (Polhamus leading-edge-suction recovery), stable, no discrete LEV ----
        # Beyond A0_crit the flow separates: the attached circulatory lift caps at the stall-onset value,
        # and the LE suction lost above A0_crit reappears as a vortex NORMAL force (Polhamus analogy):
        #   N_vortex = kv * pi*rho*U^2*c * (A0^2 - A0_crit^2),  one-sided (only |A0|>A0_crit).
        # Consistent with the validated F_s, and it is THE mechanism making cycle-mean lift RISE with flap
        # frequency: A0 swings harder at higher reduced freq -> stronger one-sided LEV. (Attached lift_p
        # alone only DECLINES with freq -- Theodorsen.)
        if self.dynamic_stall and abs(lesp) > self.lesp_crit:
            att_cap = self.lesp_crit / abs(lesp)             # attached circulation caps at stall onset
            vloc2 = U * U + hdot * hdot                       # LOCAL dynamic pressure (rises with flap freq)
            Nv = self.kv * np.pi * self.rho * vloc2 * c * (lesp * lesp - self.lesp_crit ** 2)
            lift_ds = Fz * att_cap + Nv * ca * np.sign(lesp)
        else:
            lift_ds = Fz

        # convect the wakes (induced-only; airfoil already recedes), then form the TOTAL x-impulse LIFT
        # rho*d/dt(sum_all Gamma*x) over bound+LEV+TEV (sum Gamma=0 -> frame-clean; captures the LEV's own
        # lift, which the bound pressure N alone misses). The y-impulse is a DIAGNOSTIC only (it picks up
        # spurious differential-plunge terms, so thrust uses the Garrick split above).
        self._convect(gb, xv, yv, tvx, tvy, tvg, lvx, lvy, lvg, U, dt)
        self._cap()
        sxr = self.sx                                        # reference x to the airfoil (Sum Gamma=0 -> clean)
        xg = float(np.sum(gb * (xv - sxr)) + np.sum(np.array(self.tg) * (np.array(self.tx) - sxr))
                   + np.sum(np.array(self.lg) * (np.array(self.lx) - sxr)))
        if self.it == 1:
            self._xg_old = xg
        Fz_imp = self.rho * (xg - self._xg_old) / dt         # lift = rho d/dt(sum Gamma*x) (x-impulse)
        self._xg_old = xg

        self.gammaold = float(np.sum(gb)); self.gprev = gb.copy(); self.gcum_old = gcum.copy()
        q = 0.5 * self.rho * U * U
        return dict(lift=Fz_imp, lift_p=Fz, lift_ds=lift_ds, Fx=Fx, thrust=thrust, N=N, Fs=Fs, A0=A0_eff,
                    CL=Fz_imp / (q * c + 1e-12), CT=thrust / (q * c + 1e-12),
                    lesp=lesp, n_lev=len(self.lx), n_tev=len(self.tx))

    def _vel_at(self, px, py, gb, xv, yv, tvx, tvy, tvg, lvx, lvy, lvg, U, skip_t=-1, skip_l=-1,
                freestream=True):
        uu, ww = (U, 0.0) if freestream else (0.0, 0.0)      # freestream for convect, induced-only for force
        u1, w1 = _induced(px, py, xv, yv, gb); uu += u1; ww += w1
        if skip_t >= 0:
            mt = np.ones(len(tvx), bool); mt[skip_t] = False
            u2, w2 = _induced(px, py, tvx[mt], tvy[mt], tvg[mt])
        else:
            u2, w2 = _induced(px, py, tvx, tvy, tvg)
        uu += u2; ww += w2
        if skip_l >= 0:
            ml = np.ones(len(lvx), bool); ml[skip_l] = False
            u3, w3 = _induced(px, py, lvx[ml], lvy[ml], lvg[ml])
        else:
            u3, w3 = _induced(px, py, lvx, lvy, lvg)
        uu += u3; ww += w3
        return uu, ww

    def _convect(self, gb, xv, yv, tvx, tvy, tvg, lvx, lvy, lvg, U, dt):
        nt = len(self.tx); nl = len(self.lx)
        mt = np.zeros((nt, 2)); ml = np.zeros((nl, 2))
        # convect with INDUCED velocity only (global frame; the airfoil/sx already recedes at -U, so adding
        # the freestream here would double-count it -> wake at 2U relative; DVM.m convects induced-only).
        for i in range(nt):
            mt[i] = self._vel_at(self.tx[i], self.ty[i], gb, xv, yv, tvx, tvy, tvg, lvx, lvy, lvg, U,
                                 skip_t=i, freestream=False)
        for i in range(nl):
            ml[i] = self._vel_at(self.lx[i], self.ly[i], gb, xv, yv, tvx, tvy, tvg, lvx, lvy, lvg, U,
                                 skip_l=i, freestream=False)
        for i in range(nt):
            self.tx[i] += mt[i, 0] * dt; self.ty[i] += mt[i, 1] * dt
        for i in range(nl):
            self.lx[i] += ml[i, 0] * dt; self.ly[i] += ml[i, 1] * dt

    def _cap(self):
        if len(self.tx) > self.max_wake:
            k = len(self.tx) - self.max_wake
            self.tx = self.tx[k:]; self.ty = self.ty[k:]; self.tg = self.tg[k:]
        if len(self.lx) > self.max_wake:
            k = len(self.lx) - self.max_wake
            self.lx = self.lx[k:]; self.ly = self.ly[k:]; self.lg = self.lg[k:]


if __name__ == "__main__":
    # Validation ladder
    print("=== (1) steady small alpha: CL -> 2*pi*alpha ===", flush=True)
    for ad in (2.0, 5.0):
        m = FlapLDVM(U=1.0, c=1.0, n=60, dt=0.02, rho=1.0, lesp_crit=99.0)  # LEV off (high crit)
        a = np.radians(ad); cl = []
        for it in range(300):
            r = m.step(a, 0.0, 0.0)
            if it > 200: cl.append(r["CL"])
        print(f"  alpha={ad}deg: CL={np.mean(cl):+.3f}  (2*pi*alpha={2*np.pi*a:+.3f})", flush=True)

    print("=== (2) pure pitch alpha=30deg*sin (FW reference): expect net thrust from LEV ===", flush=True)
    U = 1.0; c = 1.0; k = 0.1; Om = 2 * U / c * k; dt = 0.02
    m = FlapLDVM(U=U, c=c, n=80, dt=dt, rho=1.0, alpha_lev_deg=17.0)
    amax = np.radians(30.0); Fx = []; Fy = []
    for it in range(1200):
        t = it * dt
        a = amax * np.sin(Om * t); da = amax * Om * np.cos(Om * t)
        r = m.step(a, da, 0.0)
        if it > 800: Fx.append(r["Fx"]); Fy.append(r["thrust"])
    print(f"  mean Fx={np.mean(Fx):+.4f} (thrust=-Fx={-np.mean(Fx):+.4f})  LEVs={len(m.lx)}", flush=True)

    print("=== (3) pure plunge h=h0*sin (Knoller-Betz): expect net THRUST, ~0 mean lift ===", flush=True)
    for k in (0.2, 0.4):
        U = 1.0; c = 1.0; Om = 2 * U / c * k; dt = 0.02; h0 = 0.2 * c
        m = FlapLDVM(U=U, c=c, n=80, dt=dt, rho=1.0, lesp_crit=99.0)   # LEV off -> clean Garrick check
        Th = []; Li = []
        for it in range(1000):
            t = it * dt
            hdot = h0 * Om * np.cos(Om * t)
            r = m.step(0.0, 0.0, hdot)
            if it > 700: Th.append(r["thrust"]); Li.append(r["lift"])
        print(f"  k={k}: mean thrust={np.mean(Th):+.4f} (>0)  mean lift={np.mean(Li):+.4f} (~0)", flush=True)
    print("DONE", flush=True)
