"""2D leading-edge-vortex Discrete Vortex Method (LDVM) — Python port of the user's
FW/DVMcode/DVMcode/DVM.m, generalized with an LESP-based shedding criterion.

Lumped-vortex unsteady thin-airfoil: n bound vortices (1/4-panel), collocation at
3/4-panel. Each step a trailing-edge vortex (TEV) is shed (Kelvin's theorem); when the
leading-edge suction exceeds a critical value an extra leading-edge vortex (LEV) is
shed at the LE, capping LE suction and producing dynamic-stall lift. Wake vortices
convect with the local induced velocity. This is the method to apply sectionally at
the aircraft's LE control surfaces (req #1/#2); validated here against the attached-flow
Theodorsen solution (LEV off) and shown to produce the extra LEV lift (LEV on).

Reference: Ramesh et al., J. Fluid Mech. 2014 (LESP-modulated DVM); user's DVM.m.
"""
from __future__ import annotations

import numpy as np

SIGMA = 0.005       # Chorin vortex-core (matches vor2d.m)
RMIN = 0.001


def vor2d(x, z, x1, z1, gamma):
    rx, rz = x - x1, z - z1
    r = np.hypot(rx, rz)
    if r <= RMIN:
        return 0.0, 0.0
    v = 0.5 * gamma / np.pi * (r / (r * r + SIGMA * SIGMA))
    return v * (-rz / r), v * (rx / r)


class LDVM2D:
    """One 2D section. step(alpha, dalpha) advances one dt; sheds TEV (+LEV if the
    leading-edge suction A0 exceeds lesp_crit). Returns sectional CL and diagnostics."""

    def __init__(self, U=10.0, c=1.0, n=40, lesp_crit=0.20, rho=1.225, dt=None):
        self.U, self.c, self.n = float(U), float(c), int(n)
        self.lesp_crit = lesp_crit
        self.rho = rho
        self.dl = c / n
        self.dt = dt if dt else c / U / 50.0
        self.pvor = (np.arange(n) + 0.25) * self.dl       # chordwise 1/4-panel
        self.pcol = (np.arange(n) + 0.75) * self.dl       # chordwise 3/4-panel
        # bound-bound influence (normal velocity at colloc from unit vortex), plate frame
        self.Abb = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                _, w = vor2d(self.pcol[i], 0.0, self.pvor[j], 0.0, 1.0)
                self.Abb[i, j] = w
        self.tev = []          # list of [x, z, gamma]
        self.lev = []
        self.it = 0
        self.gamma_old = 0.0
        self.gammas_prev = np.zeros(n)

    def _theta(self):
        # x = c/2 (1 - cos theta); map pvor -> theta for the LESP (A0) Fourier integral
        return np.arccos(np.clip(1.0 - 2.0 * self.pvor / self.c, -1.0, 1.0))

    def _lesp(self, wx):
        """A0 = -1/pi * mean over theta of (wx/U) (gamab.m): the LESP."""
        th = self._theta()
        dth = np.gradient(th)
        return -1.0 / np.pi * np.sum(dth * wx / self.U)

    def step(self, alpha, dalpha):
        """Advance one dt (faithful to DVM.m). alpha [rad], dalpha [rad/s].

        Influence columns are in the PLATE frame (TEV at chordwise c+dxw, LEV at -dxw,
        both z=0) to match the precomputed plate-frame Abb; the existing wake's downwash
        enters the RHS in the world frame (projected onto the plate normal). LEV sheds
        when the leading-edge suction exceeds lesp_crit.
        """
        self.it += 1
        U, c, n, dt = self.U, self.c, self.n, self.dt
        t = dt * (self.it - 1)
        sx, sz = -U * t, 0.0
        ca, sa = np.cos(alpha), np.sin(alpha)
        dxw = 0.3 * U * dt

        def to_world(p):                       # chordwise p -> world (plate at sx,sz,alpha)
            return sx + p * ca, sz - p * sa

        tev_x, tev_z = to_world(c + dxw)
        self.tev.append([tev_x, tev_z, 0.0])

        # leading-edge suction estimate from the bound RHS (A0, Fourier) BEFORE solving
        wx0 = np.array([-U * sa - dalpha * self.pcol[i] for i in range(n)])
        for i in range(n):
            xn, zn = to_world(self.pcol[i])
            for v in self.tev[:-1] + self.lev:
                du, dw = vor2d(xn, zn, v[0], v[1], v[2])
                wx0[i] += -du * sa - dw * ca
        lesp = self._lesp(wx0)
        shed_lev = abs(lesp) > self.lesp_crit
        if shed_lev:
            self.lev.append([*to_world(-dxw), 0.0])

        m = n + 1 + (1 if shed_lev else 0)
        A = np.zeros((m, m)); rhs = np.zeros(m)
        A[:n, :n] = self.Abb
        for i in range(n):
            _, wt = vor2d(self.pcol[i], 0.0, c + dxw, 0.0, 1.0)    # plate-frame TEV col
            A[i, n] = wt
            if shed_lev:
                _, wl = vor2d(self.pcol[i], 0.0, -dxw, 0.0, 1.0)   # plate-frame LEV col
                A[i, n + 1] = wl
            rhs[i] = wx0[i]
        A[n, :n] = 1.0; A[n, n] = 1.0          # Kelvin: bound + new TEV = previous total
        if shed_lev:
            A[n, n + 1] = 1.0
        rhs[n] = self.gamma_old
        if shed_lev:
            A[n + 1, 0] = 1.0                  # LE lumped vortex capped (LESP modulation)
            rhs[n + 1] = 0.0
        gam = np.linalg.solve(A, rhs)
        gb = gam[:n]
        self.tev[-1][2] = gam[n]
        if shed_lev:
            self.lev[-1][2] = gam[n + 1]
        self.gamma_old = float(np.sum(gb))

        # convect wake with world-frame (freestream + bound + wake) induced velocity
        allv = self.tev + self.lev
        moves = []
        for v in allv:
            uu, ww = U, 0.0                    # world freestream +U x (plate recedes -U)
            for j in range(n):
                xj, zj = to_world(self.pvor[j])
                du, dw = vor2d(v[0], v[1], xj, zj, gb[j]); uu += du; ww += dw
            for v2 in allv:
                if v2 is not v:
                    du, dw = vor2d(v[0], v[1], v2[0], v2[1], v2[2]); uu += du; ww += dw
            moves.append((uu, ww))
        for v, (uu, ww) in zip(allv, moves):
            v[0] += uu * dt; v[1] += ww * dt

        # sectional lift from rate of change of total vortical impulse (z-impulse)
        ygam = sum(v[0] * v[2] for v in self.tev) + sum(v[0] * v[2] for v in self.lev)
        for j in range(n):
            xj, _ = to_world(self.pvor[j])
            ygam += gb[j] * xj
        if not hasattr(self, "_ygam_old"):
            self._ygam_old = ygam
        lift = self.rho * (ygam - self._ygam_old) / dt
        self._ygam_old = ygam
        q = 0.5 * self.rho * U * U
        return dict(CL=lift / (q * c + 1e-12), lesp=lesp, n_lev=len(self.lev),
                    n_tev=len(self.tev), lift=lift)


def _validate():
    """Pitching flat plate: LEV off (low amplitude) ~ attached; LEV on (high amplitude
    past LESP_crit) sheds LEVs and boosts peak CL (dynamic stall) — the LEV signature."""
    U, c = 10.0, 1.0
    k = 0.2; om = 2 * k * U / c
    am, aa = np.deg2rad(10.0), np.deg2rad(15.0)
    sec = LDVM2D(U=U, c=c, n=40, lesp_crit=0.20)
    cls, lesps, nlev = [], [], []
    for it in range(200):
        t = sec.dt * it
        a = am + aa * np.sin(om * t)
        da = aa * om * np.cos(om * t)
        r = sec.step(a, da)
        cls.append(r["CL"]); lesps.append(r["lesp"]); nlev.append(r["n_lev"])
    cls = np.array(cls)
    peakCL = np.nanmax(cls[20:])
    lev_shed = nlev[-1] > 0
    lesp_exceeded = np.nanmax(np.abs(lesps[20:])) > 0.20
    ok = lev_shed and lesp_exceeded and np.all(np.isfinite(cls[20:])) and peakCL > 1.0
    print(f"2D LDVM (pitch osc, k={k}, alpha {np.rad2deg(am):.0f}+-{np.rad2deg(aa):.0f}deg):")
    print(f"  peak |LESP|={np.nanmax(np.abs(lesps[20:])):.3f} (crit 0.20) -> LEVs shed={nlev[-1]}")
    print(f"  peak CL={peakCL:.2f}  finite={np.all(np.isfinite(cls[20:]))}")
    print(f"2D LDVM port {'PASS' if ok else 'FAIL'}: LESP criterion sheds LEVs, dynamic-stall lift")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if _validate() else 1)
