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

S2 (PROJECT_rvpm) retrofit — opt-in keywords, legacy defaults untouched:
  pivot_xc   : pitch pivot x/c (legacy 0.0 = LE; Eldredge canonical cases pitch about 0.25/0.5)
  shed_rule  : 'growing' legacy (|LESP|>crit AND rising) | 'ramesh' (shed EVERY supercritical step,
               A0 pinned at +-crit — Ramesh 2014 JFM 751 semantics, needed for the hold-plateau)
  placement  : 'fixed' legacy (0.3*U*dt off the edge) | 'third' (Ansari/Ramesh rule: new vortex at
               1/3 of edge->last-shed; first vortex of an event at first_fac*U*dt off the edge)
  core       : 'legacy' fixed-SIGMA Scully | 'vatistas2' per-particle Vatistas n=2,
               r_c = core_fac * U * dt (Ramesh 2014: 1.3 x mean spacing = 0.02c at dt*=0.015)
Circulation of shed particles is FROZEN after the solve step (impulse-matching-free path, A3-C3).

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


def _induced_many(px, py, vx, vy, vg, rc=None):
    """Vectorized many-to-many Biot-Savart: velocities at targets (px,py) from sources (vx,vy,vg).
    rc=None  -> legacy fixed-SIGMA Scully core with RMIN cutoff (matches the old _induced).
    rc=array -> per-particle Vatistas n=2 core u_th = G/(2pi) * r/sqrt(r^4+rc^4) (Ramesh 2014 LDVM);
                regular at r=0 (self-term vanishes), so no cutoff is applied."""
    px = np.atleast_1d(np.asarray(px, float)); py = np.atleast_1d(np.asarray(py, float))
    if len(vx) == 0:
        z = np.zeros(len(px))
        return z, z.copy()
    vx = np.asarray(vx, float); vy = np.asarray(vy, float); vg = np.asarray(vg, float)
    rc4 = None if rc is None else np.asarray(rc, float)[None, :] ** 4
    u = np.empty(len(px)); w = np.empty(len(px))
    CH = max(1, int(4.0e6 / max(len(vx), 1)))            # chunk targets: <=4M pair temporaries
    for s in range(0, len(px), CH):
        rx = px[s:s + CH, None] - vx[None, :]; ry = py[s:s + CH, None] - vy[None, :]
        r2 = rx * rx + ry * ry
        if rc4 is None:
            F = np.where(r2 > RMIN * RMIN, 1.0 / (r2 + SIGMA * SIGMA), 0.0)
        else:
            F = 1.0 / np.sqrt(r2 * r2 + rc4)
        g = vg[None, :] / (2.0 * np.pi) * F
        u[s:s + CH] = np.sum(g * (-ry), axis=1); w[s:s + CH] = np.sum(g * rx, axis=1)
    return u, w


def _induced(px, py, vx, vy, vg, rc=None):
    """Single-target wrapper (legacy signature)."""
    u, w = _induced_many(np.array([float(px)]), np.array([float(py)]), vx, vy, vg, rc)
    return float(u[0]), float(w[0])


class FlapLDVM:
    """One 2D strip. step(alpha, dalpha, hdot) advances one dt with plunge velocity hdot.
    Returns sectional lift (normal) and chordwise force; thrust = -Fx (LE suction is in here)."""

    def __init__(self, U=1.0, c=1.0, n=80, dt=None, rho=1.225,
                 lesp_crit=0.20, alpha_lev_deg=None, max_wake=300, lev_shed=True,
                 camber_m=0.0, camber_p=0.40,
                 pivot_xc=0.0, shed_rule="growing", placement="fixed", core="legacy",
                 core_fac=1.3, first_fac=0.5, core_rc=None, a0_quad="legacy"):
        self.U, self.c, self.n = float(U), float(c), int(n)
        self.dt = dt if dt else c / U / 50.0
        self.rho = rho
        self.lesp_crit = lesp_crit                       # LESP threshold: suction saturates here; if
        self.lev_shed = lev_shed                         # lev_shed -> also shed discrete LEV particles
        self.alpha_lev = np.radians(alpha_lev_deg) if alpha_lev_deg else None
        self.max_wake = int(max_wake)
        self.xp = float(pivot_xc) * self.c               # pitch pivot (chord station); sx,sy = pivot
        self.a0_quad = a0_quad
        self.shed_rule = shed_rule
        self.placement = placement
        self.core = core
        self.first_fac = float(first_fac)
        # Vatistas r_c = 1.3 x mean shed spacing; UNSflow HARDCODES 0.02c even when dt is refined
        # below 0.015 (find_tstep), so gate runs pass core_rc=0.02*c explicitly.
        self.rc_shed = float(core_rc) if core_rc is not None else float(core_fac) * self.U * self.dt
        self.dl = self.c / self.n
        self.pvor = (np.arange(self.n) + 0.25) * self.dl     # 1/4-panel bound vortices
        self.pcol = (np.arange(self.n) + 0.75) * self.dl     # 3/4-panel collocation
        # exact-pi quadrature for A_n (a0_quad='exact'): theta at pcol, bin edges -> [0,pi]
        self._thc = np.arccos(np.clip(1.0 - 2.0 * self.pcol / self.c, -1.0, 1.0))
        ed = np.concatenate([[0.0], 0.5 * (self._thc[1:] + self._thc[:-1]), [np.pi]])
        self._dthc = np.diff(ed)
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
        self.tx = []; self.ty = []; self.tg = []; self.tr = []   # TEV particles (+core radius)
        self.lx = []; self.ly = []; self.lg = []; self.lr = []   # LEV particles (+core radius)
        self.it = 0
        self._lesp_old = 0.0                                 # for the up-stroke shed gate (dLESP/dt>0)
        self._AF_old = np.zeros(4)                           # A0..A3 at previous step (for rates)
        self._lev_prev_it = -99                              # last step that shed an LEV (1/3-rule event)
        self.gammaold = 0.0                                  # total bound circ (Kelvin)
        self.gprev = np.zeros(self.n)                        # gammas at previous step (for dGamma/dt)
        self.gcum_old = np.zeros(self.n)                     # cumulative bound (for pressure lift)
        self.sx = 0.0; self.sy = 0.0                         # pivot world position (recedes -U, plunges +h)

    def _theta(self):
        return np.arccos(np.clip(1.0 - 2.0 * self.pvor / self.c, -1.0, 1.0))

    def _quad(self):
        """(theta, weights) for the A_n downwash integrals. 'legacy' = gradient(theta(pvor))
        (sum(dth)~2.92<pi, ~7% low — kept bit-exact for the strip closure). 'exact' = theta at
        the COLLOCATION points (where the downwash lives) with bin-edge weights extended to
        [0,pi] (sum==pi exactly -> steady A0 = sin(alpha) exact; S2 literature-crit mode)."""
        if self.a0_quad == "exact":
            return self._thc, self._dthc
        th = self._theta()
        return th, np.gradient(th)

    def _lesp(self, wx):
        th, dth = self._quad()
        return -1.0 / np.pi * np.sum(dth * wx / self.U)

    def _wake_rc(self, arr):
        """Per-source core array for the chosen kernel (None -> legacy Scully branch)."""
        return np.asarray(arr, float) if self.core == "vatistas2" else None

    def _bound_rc(self):
        """Bound vortices act on the wake with the small fixed core in either kernel family."""
        return np.full(self.n, SIGMA) if self.core == "vatistas2" else None

    def _world(self, x):
        """Chord station(s) -> world coordinates (pivot at sx,sy, plate at incidence self._ca/_sa)."""
        return (self.sx + (x - self.xp) * self._ca,
                self.sy - (x - self.xp) * self._sa)

    def _edge_rel_vel(self, station, dalpha, hdot, ui=0.0, wi=0.0):
        """Relative (fluid - plate) velocity at a chord-station edge point, world frame — the
        first-shed-particle placement velocity (Ramesh: 'determined using the velocity at the
        shedding edge'; UNSflow place_tev/place_lev). Kinematic part from d/dt of _world():
        v_plate = (-U - (s-xp)*sa*adot,  hdot - (s-xp)*ca*adot)."""
        s = station - self.xp
        return (self.U + s * self._sa * dalpha + ui,
                -hdot + s * self._ca * dalpha + wi)

    def step(self, alpha, dalpha, hdot=0.0):
        self.it += 1
        U, c, n, dt = self.U, self.c, self.n, self.dt
        ca, sa = np.cos(alpha), np.sin(alpha)
        self._ca, self._sa = ca, sa
        self.sx -= U * dt                                    # pivot recedes in -x (freestream +U)
        self.sy += hdot * dt                                 # PLUNGE: pivot plunges with the strip
        rc_new = self.rc_shed if self.core == "vatistas2" else SIGMA

        # shed a TEV particle (every step): legacy = fixed offset on the chord extension;
        # 'third' = Ansari/Ramesh rule (1/3 of the way from the TE to the last shed TEV).
        tex, tey = self._world(c)
        if self.placement == "third" and len(self.tx) > 0:
            ntx = tex + (self.tx[-1] - tex) / 3.0
            nty = tey + (self.ty[-1] - tey) / 3.0
        elif self.placement == "third":
            ur, _ = self._edge_rel_vel(c, dalpha, hdot)      # first TEV: TE + 0.5*u*dt (x only)
            ntx, nty = tex + 0.5 * ur * dt, tey
        else:
            ntx, nty = self._world(c + 0.3 * U * dt)
        self.tx.append(ntx); self.ty.append(nty); self.tg.append(0.0); self.tr.append(rc_new)
        tvx = np.array(self.tx); tvy = np.array(self.ty); tvg = np.array(self.tg)
        lvx = np.array(self.lx); lvy = np.array(self.ly); lvg = np.array(self.lg)
        trc = np.array(self.tr); lrc = np.array(self.lr)

        # collocation world positions
        xn, yn = self._world(self.pcol)
        # RHS: freestream + pitch-rate (about the pivot) + PLUNGE + CAMBER slope + existing wake
        # downwash (on plate normal)
        rhs_b = (-U * sa - dalpha * (self.pcol - self.xp) - hdot * ca
                 + U * ca * self.dzc)                        # camber dzc/dx (Ramesh 2.17)
        uu, ww = _induced_many(xn, yn, tvx[:-1], tvy[:-1], tvg[:-1],
                               self._wake_rc(trc[:-1]))      # old TEV (exclude fresh)
        ul, wl = _induced_many(xn, yn, lvx, lvy, lvg, self._wake_rc(lrc))   # LEV
        rhs_b += -(uu + ul) * sa - (ww + wl) * ca

        # LESP criterion. 'growing' (legacy DVM.m): shed ONLY while the LE suction is high AND
        # GROWING (up-stroke) -> rise-peak-drop. 'ramesh' (Ramesh 2014): shed EVERY step while
        # |LESP| > crit, A0 pinned at +-crit -> sustained shedding during a supercritical hold.
        lesp = self._lesp(rhs_b)
        dlesp = lesp - self._lesp_old; self._lesp_old = lesp
        if self.alpha_lev is not None:
            shed_lev = abs(alpha) > self.alpha_lev and dalpha > 0
        elif self.shed_rule == "ramesh":
            shed_lev = abs(lesp) > self.lesp_crit
        else:
            shed_lev = abs(lesp) > self.lesp_crit and lesp * dlesp > 0   # |LESP| increasing
        shed_lev = shed_lev and self.lev_shed     # discrete LEV particles optional (suction still caps)
        if shed_lev:
            lex, ley = self._world(0.0)
            if self.placement == "third" and self._lev_prev_it == self.it - 1 and len(self.lx) > 0:
                nlx = lex + (self.lx[-1] - lex) / 3.0        # continuing shed event: 1/3 rule
                nly = ley + (self.ly[-1] - ley) / 3.0
            elif self.placement == "third":
                # new event: LE + 0.5*v_LE*dt, v_LE = kinematic + wake-induced (UNSflow place_lev)
                uw, ww_ = _induced(lex, ley, tvx, tvy, tvg, self._wake_rc(trc))
                uw2, ww2 = _induced(lex, ley, lvx, lvy, lvg, self._wake_rc(lrc))
                ur, wr = self._edge_rel_vel(0.0, dalpha, hdot, uw + uw2, ww_ + ww2)
                nlx, nly = lex + 0.5 * ur * dt, ley + 0.5 * wr * dt
            else:
                nlx, nly = self._world(-0.3 * U * dt)        # legacy: just ahead of the LE
            self.lx.append(nlx); self.ly.append(nly); self.lg.append(0.0); self.lr.append(rc_new)
            self._lev_prev_it = self.it

        # assemble [bound | TEV | (LEV)] with Kelvin + LE-suction-cap rows
        m = n + 1 + (1 if shed_lev else 0)
        A = np.zeros((m, m)); rhs = np.zeros(m)
        A[:n, :n] = self.Abb
        # fresh-vortex columns: unit-strength normal velocity at the collocations (world frame;
        # normal projection u*sa + w*ca — reduces to the old plate-frame w for on-chord placement)
        uT, wT = _induced_many(xn, yn, np.array([ntx]), np.array([nty]), np.array([1.0]),
                               np.array([rc_new]) if self.core == "vatistas2" else None)
        A[:n, n] = uT * sa + wT * ca
        if shed_lev:
            uL, wL = _induced_many(xn, yn, np.array([self.lx[-1]]), np.array([self.ly[-1]]),
                                   np.array([1.0]),
                                   np.array([rc_new]) if self.core == "vatistas2" else None)
            A[:n, n + 1] = uL * sa + wL * ca
        rhs[:n] = rhs_b
        A[n, :n] = 1.0; A[n, n] = 1.0                        # Kelvin: sum(bound) + TEV = old total
        if shed_lev:
            A[n, n + 1] = 1.0
        rhs[n] = self.gammaold
        if shed_lev:
            # Ramesh LESP modulation: HOLD A0 at the critical value (NOT zero the LE vortex). A0 is a
            # linear functional of the downwash; the shed TEV/LEV reduce it. Constraint: A0_post = A0_crit.
            _, dth = self._quad()
            lcol = lambda col: -1.0 / np.pi * np.sum(dth * col / U)   # LESP of a downwash column
            A[n + 1, n] = -lcol(A[:n, n]); A[n + 1, n + 1] = -lcol(A[:n, n + 1])
            rhs[n + 1] = self.lesp_crit * np.sign(lesp) - lesp        # bring A0 down to +/- A0_crit
        gam = np.linalg.solve(A, rhs)
        gb = gam[:n]
        self.tg[-1] = float(gam[n])
        if shed_lev:
            self.lg[-1] = float(gam[n + 1])
        # post-solve Fourier coefficients of the EFFECTIVE downwash (fresh vortices moved to the
        # RHS): thin-airfoil A0..A3 — the low-pass force channel of Ramesh 2014 (A0 equals the
        # pinned value while an LEV sheds; consistency check of the modulation row).
        wx_eff = rhs_b - A[:n, n] * gam[n]
        if shed_lev:
            wx_eff = wx_eff - A[:n, n + 1] * gam[n + 1]
        thf, dthf = self._quad()
        AF = np.empty(4)
        AF[0] = -1.0 / np.pi * np.sum(dthf * wx_eff / U)
        for kk in (1, 2, 3):
            AF[kk] = 2.0 / np.pi * np.sum(dthf * (wx_eff / U) * np.cos(kk * thf))
        dAF = (AF - self._AF_old) / dt if self.it > 1 else np.zeros(4)
        self._AF_old = AF.copy()
        tvg = np.array(self.tg)
        lvx = np.array(self.lx); lvy = np.array(self.ly); lvg = np.array(self.lg)
        lrc = np.array(self.lr)

        xv, yv = self._world(self.pvor)                      # bound-vortex world positions
        dG = (gb - self.gprev) / dt

        # ---- normal force N (unsteady pressure, Bernoulli) + leading-edge SUCTION (analytic) ----
        # Canonical Ramesh-2014/2020 LDVM force split: the flat-plate pressure gives the plate-NORMAL
        # force N; the streamwise THRUST is the leading-edge SUCTION, which the discrete bound vortices
        # do NOT resolve (the LE sqrt(x) velocity singularity) and so MUST be added analytically.
        #   Ramesh 2020 JFM 886 A13: S = lim_{x->0} (1/2) gamma sqrt(x) = U*sqrt(c)*A0  (eqn 3.3),
        #   Garrick 1937 / von Karman-Burgers: suction force  F_s = pi*rho*S^2 = pi*rho*U^2*c*A0^2.
        gcum = np.cumsum(gb)
        u1, w1 = _induced_many(xv, yv, xv, yv, gb, self._bound_rc())     # bound-on-bound (self -> 0)
        u2, w2 = _induced_many(xv, yv, tvx, tvy, tvg, self._wake_rc(trc))
        u3, w3 = _induced_many(xv, yv, lvx, lvy, lvg, self._wake_rc(lrc))
        u_ch = (u1 + u2 + u3) * ca - (w1 + w2 + w3) * sa     # chordwise wake-induced velocity
        dp = self.rho * ((U * ca + u_ch) * gb / self.dl + (gcum - self.gcum_old) / dt)
        N = float(np.sum(dp) * self.dl)                      # plate-normal force (code sign: Fz=-N*ca)
        # A0 = LESP; while an LEV is shedding it is HELD at A0_crit -> the LE suction is CAPPED (the
        # excess goes into the shed LEV, Polhamus). The detached-LEV thrust then enters through N: the
        # LEVs' induced downwash enhances the bound circulation/pressure -> larger N -> N*sin(a) thrust.
        # The LE suction SATURATES at A0_crit: beyond the critical LESP the flow separates (LEV), so the
        # suction cannot grow without bound (the excess goes into the LEV, Polhamus). This caps the thrust
        # physically WITHOUT needing the (divergent at large amplitude) discrete LEV shedding.
        A0_eff = max(-self.lesp_crit, min(self.lesp_crit, lesp))
        Fs = np.pi * self.rho * U * U * c * A0_eff * A0_eff  # LE suction (>=0), along chord toward LE
        # ---- Ramesh 2014 Fourier force channel (thesis eqs 4.26-4.32 = JFS 2015 eqs 12-14;
        # UNSflow postprocess.jl calc_forces) — the published LDVM force, low-pass in the Fourier
        # coefficients (a close wake particle projects only weakly onto A0..A2, unlike the pointwise
        # pressure loop above). Nonlinear wake term = sum over bound points of (wake-induced
        # chordwise velocity)*(bound circulation), WAKE-induced only:
        u_wk = (u2 + u3) * ca - (w2 + w3) * sa
        Ucirc = U * ca + hdot * sa                           # (hdot sign: negligible for gates)
        # NOTE sign: thin-airfoil Gamma (positive = positive lift) = MINUS our lumped gb (our
        # solve returns clockwise-negative gb for positive lift; ladder case 1 fixes this).
        FNf = (self.rho * np.pi * c * U * (Ucirc * (AF[0] + 0.5 * AF[1])
               + c * (0.75 * dAF[0] + 0.25 * dAF[1] + 0.125 * dAF[2]))
               - self.rho * float(np.sum(u_wk * gb)))
        FSf = Fs                                             # F_S = rho*pi*c*U^2*A0^2 (A0 pinned=capped)
        Fz = -N * ca                                         # pressure lift (secondary; primary = Fz_imp)
        # Garrick thrust = F_s*cos(a) - N_phys*sin(a); code N=-N_phys -> thrust = Fs*ca + N*sa.
        # This is the PRIMARY thrust: exact for attached flow (steady flat plate -> 0, d'Alembert), and
        # with A0 wake-corrected (rhs_b includes the wake downwash) it gives the right plunge/Knoller-Betz
        # thrust scaling with reduced frequency. While an LEV sheds the suction is capped at A0_crit, so
        # this UNDER-predicts the deep-stall LEV thrust (a flat-plate-normal limit) -> see thrust_lev.
        thrust = Fs * ca + N * sa
        Fx = -thrust

        # (The empirical Polhamus dynamic-stall LIFT recovery was REMOVED on 2026-06-24 — first-principles
        # only. The real dynamic-stall lift comes from the discrete LEV particles, lev_shed=True.)
        lift_ds = Fz

        # convect the wakes (induced-only; airfoil already recedes), then form the TOTAL x-impulse LIFT
        # rho*d/dt(sum_all Gamma*x) over bound+LEV+TEV (sum Gamma=0 -> frame-clean; captures the LEV's own
        # lift, which the bound pressure N alone misses). The y-impulse is a DIAGNOSTIC only (it picks up
        # spurious differential-plunge terms, so thrust uses the Garrick split above).
        self._convect(gb, xv, yv, tvx, tvy, tvg, trc, lvx, lvy, lvg, lrc, dt)
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
        # CLp/CDp = the Ramesh pressure+suction channel in wind axes (N_phys = -N):
        #   CL = (N_phys*ca + Fs*sa)/qc,  CD = (N_phys*sa - Fs*ca)/qc  (gate channel for S2)
        return dict(lift=Fz_imp, lift_p=Fz, lift_ds=lift_ds, Fx=Fx, thrust=thrust, N=N, Fs=Fs, A0=A0_eff,
                    CL=Fz_imp / (q * c + 1e-12), CT=thrust / (q * c + 1e-12),
                    CLp=(-N * ca + Fs * sa) / (q * c + 1e-12),
                    CDp=(-N * sa - Fs * ca) / (q * c + 1e-12),
                    # published-LDVM channel: Cl = CN ca + CS sa, Cd = CN sa - CS ca (gate channel)
                    CLf=(FNf * ca + FSf * sa) / (q * c + 1e-12),
                    CDf=(FNf * sa - FSf * ca) / (q * c + 1e-12),
                    CNf=FNf / (q * c + 1e-12),
                    AF=AF, dAF=dAF,
                    lesp=lesp, n_lev=len(self.lx), n_tev=len(self.tx))

    def _convect(self, gb, xv, yv, tvx, tvy, tvg, trc, lvx, lvy, lvg, lrc, dt):
        # convect with INDUCED velocity only (global frame; the airfoil/sx already recedes at -U, so adding
        # the freestream here would double-count it -> wake at 2U relative; DVM.m convects induced-only).
        nt = len(tvx); nl = len(lvx)
        if nt + nl == 0:
            return
        wx = np.concatenate([tvx, lvx]); wy = np.concatenate([tvy, lvy])
        u1, w1 = _induced_many(wx, wy, xv, yv, gb, self._bound_rc())
        u2, w2 = _induced_many(wx, wy, tvx, tvy, tvg, self._wake_rc(trc))   # self-pair -> 0
        u3, w3 = _induced_many(wx, wy, lvx, lvy, lvg, self._wake_rc(lrc))
        u = u1 + u2 + u3; w = w1 + w2 + w3
        for i in range(nt):
            self.tx[i] += u[i] * dt; self.ty[i] += w[i] * dt
        for i in range(nl):
            self.lx[i] += u[nt + i] * dt; self.ly[i] += w[nt + i] * dt

    def _cap(self):
        if len(self.tx) > self.max_wake:
            k = len(self.tx) - self.max_wake
            self.tx = self.tx[k:]; self.ty = self.ty[k:]; self.tg = self.tg[k:]; self.tr = self.tr[k:]
        if len(self.lx) > self.max_wake:
            k = len(self.lx) - self.max_wake
            self.lx = self.lx[k:]; self.ly = self.ly[k:]; self.lg = self.lg[k:]; self.lr = self.lr[k:]


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
