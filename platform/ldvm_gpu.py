"""GPU-batched 2D LDVM engine (user directive 2026-07-19: 100% GPU, throughput-first).

Math = ldvm_fourier.LDVM2D verbatim (S2-verified UNSflow port: Fourier bound layer,
pre-LEV rate semantics, wake term force), vectorized over LANES = strips x {LEV-on,
LEV-off} x (optionally conditions). The O(N^2) pairwise work (wake->stations induction,
wake convection with bound segments) runs in warp fp32 kernels on the GPU (RTX 4090
fp32:fp64 = 64:1 — the decisive lever; repo precedent: validated fp32 fast paths).
The per-step small algebra (trapz Fourier integrals as matmuls, 2x2 pin solves,
placement bookkeeping, force assembly) is lane-vectorized numpy — O(L*ndiv), negligible.

Per-step transfers: up ~90KB (stations, bound segments, fresh vortices), down ~40KB
(station velocities, last-shed positions). Wake state lives on the GPU; strengths are
FROZEN after birth so the CPU strength mirror is exact (Kelvin bookkeeping without
downloads). Validation harness in __main__: steady ladder + unsteady case vs the fp64
CPU LDVM2D within a tolerance band."""
from __future__ import annotations

import numpy as np
import warp as wp

F32 = wp.float32


@wp.kernel
def k_wake_at_pts(px: wp.array(dtype=F32, ndim=2), py: wp.array(dtype=F32, ndim=2),
                  wx: wp.array(dtype=F32, ndim=2), wy: wp.array(dtype=F32, ndim=2),
                  wg: wp.array(dtype=F32, ndim=2), wn: wp.array(dtype=wp.int32),
                  rc4: wp.array(dtype=F32),
                  ou: wp.array(dtype=F32, ndim=2), ow: wp.array(dtype=F32, ndim=2)):
    """Vatistas-n2 wake induction at per-lane points: (l, i) thread, loop wake slots."""
    l, i = wp.tid()
    x = px[l, i]; y = py[l, i]
    u = F32(0.0); w = F32(0.0)
    c4 = rc4[l]
    for j in range(wn[l]):
        rx = x - wx[l, j]; ry = y - wy[l, j]
        r2 = rx * rx + ry * ry
        f = wg[l, j] * F32(0.15915494) / wp.sqrt(r2 * r2 + c4)
        u = u - f * ry
        w = w + f * rx
    ou[l, i] = u; ow[l, i] = w


@wp.kernel
def k_convect(wx: wp.array(dtype=F32, ndim=2), wy: wp.array(dtype=F32, ndim=2),
              wg: wp.array(dtype=F32, ndim=2), wn: wp.array(dtype=wp.int32),
              rc4: wp.array(dtype=F32),
              bx: wp.array(dtype=F32, ndim=2), by: wp.array(dtype=F32, ndim=2),
              bg: wp.array(dtype=F32, ndim=2), nb: int, dt: F32,
              nx: wp.array(dtype=F32, ndim=2), ny: wp.array(dtype=F32, ndim=2)):
    """Convect wake slot (l,j) in wake (skip self) + bound-segment field. Induced-only
    (world frame: the plate recedes). New positions to nx/ny (double-buffer)."""
    l, j = wp.tid()
    if j >= wn[l]:
        return
    x = wx[l, j]; y = wy[l, j]
    u = F32(0.0); w = F32(0.0)
    c4 = rc4[l]
    for k in range(wn[l]):
        rx = x - wx[l, k]; ry = y - wy[l, k]
        r2 = rx * rx + ry * ry
        f = wg[l, k] * F32(0.15915494) / wp.sqrt(r2 * r2 + c4)
        u = u - f * ry
        w = w + f * rx
    for m in range(nb):
        rx = x - bx[l, m]; ry = y - by[l, m]
        r2 = rx * rx + ry * ry
        f = bg[l, m] * F32(0.15915494) / wp.sqrt(r2 * r2 + c4)
        u = u - f * ry
        w = w + f * rx
    nx[l, j] = x + u * dt
    ny[l, j] = y + w * dt


@wp.kernel
def k_append(wx: wp.array(dtype=F32, ndim=2), wy: wp.array(dtype=F32, ndim=2),
             wg: wp.array(dtype=F32, ndim=2), wn: wp.array(dtype=wp.int32),
             fx: wp.array(dtype=F32), fy: wp.array(dtype=F32), fg: wp.array(dtype=F32),
             mask: wp.array(dtype=wp.int32)):
    """Append one fresh vortex per masked lane at slot wn[l] (count updated CPU-side)."""
    l = wp.tid()
    if mask[l] == 0:
        return
    s = wn[l]
    wx[l, s] = fx[l]; wy[l, s] = fy[l]; wg[l, s] = fg[l]


@wp.kernel
def k_gather(wx: wp.array(dtype=F32, ndim=2), wy: wp.array(dtype=F32, ndim=2),
             idx: wp.array(dtype=wp.int32), ox: wp.array(dtype=F32),
             oy: wp.array(dtype=F32)):
    """Gather one slot position per lane (last TEV / last LEV placement reference)."""
    l = wp.tid()
    j = idx[l]
    if j >= 0:
        ox[l] = wx[l, j]; oy[l] = wy[l, j]


class LDVMBatch:
    """L independent 2D LDVM lanes advanced in lock-step (same U, dt). Math == LDVM2D."""

    def __init__(self, chords, lesp_crit, U=8.0, dt=5.4e-4, ndiv=70, naterm=35,
                 rho=1.225, camber_m=0.02, camber_p=0.40, wmax=24000, dev="cuda:0"):
        self.L = L = len(chords)
        self.c = np.asarray(chords, float)
        self.crit = np.asarray(lesp_crit, float)
        self.U, self.dt, self.rho = float(U), float(dt), float(rho)
        self.ndiv, self.naterm = int(ndiv), int(naterm)
        self.dev = dev
        self.th = np.linspace(0.0, np.pi, ndiv)
        self.xs = 0.5 * self.c[:, None] * (1.0 - np.cos(self.th))[None, :]   # (L, ndiv)
        xc = self.xs / self.c[:, None]
        m, p = camber_m, camber_p
        self.dzc = (np.where(xc < p, 2.0 * m / (p * p) * (p - xc),
                             2.0 * m / ((1.0 - p) ** 2) * (p - xc)) if m > 0
                    else np.zeros_like(xc))
        # trapz weights on th + Fourier cos matrices (An integrals as one matmul)
        wt = np.gradient(self.th); wt[0] *= 0.5; wt[-1] *= 0.5
        wt = np.diff(self.th, prepend=self.th[0]) * 0.0  # placeholder replaced below
        wt = np.empty(ndiv); wt[1:-1] = (self.th[2:] - self.th[:-2]) / 2.0
        wt[0] = (self.th[1] - self.th[0]) / 2.0; wt[-1] = (self.th[-1] - self.th[-2]) / 2.0
        self.wt = wt                                                          # trapz weights
        nvec = np.arange(1, naterm + 1)
        self.cosn = np.cos(nvec[None, :] * self.th[:, None])                  # (ndiv, naterm)
        self.sinn = np.sin(nvec[None, :] * self.th[:, None])
        self.rc = 0.02 * self.c
        # GPU state
        self.wmax = wmax
        z = lambda: wp.zeros((L, wmax), dtype=F32, device=dev)
        self.wx, self.wy, self.wgd = z(), z(), z()
        self.nx_, self.ny_ = z(), z()
        self.wn_np = np.zeros(L, dtype=np.int32)
        self.wn = wp.zeros(L, dtype=wp.int32, device=dev)
        self.rc4 = wp.array((self.rc ** 4).astype(np.float32), dtype=F32, device=dev)
        self.wg_np = np.zeros((L, wmax))                                      # exact strength mirror
        self.last_tev = np.full(L, -1, dtype=np.int32)
        self.last_lev = np.full(L, -1, dtype=np.int32)
        self.lev_prev_it = np.full(L, -99)
        self._AF_prev = np.zeros((L, 4))
        self.it = 0
        self.sx = np.zeros(L); self.sy = np.zeros(L)
        # device scratch
        self.px = wp.zeros((L, ndiv), dtype=F32, device=dev)
        self.py = wp.zeros((L, ndiv), dtype=F32, device=dev)
        self.ou = wp.zeros((L, ndiv), dtype=F32, device=dev)
        self.owd = wp.zeros((L, ndiv), dtype=F32, device=dev)
        self.bxd = wp.zeros((L, ndiv - 1), dtype=F32, device=dev)
        self.byd = wp.zeros((L, ndiv - 1), dtype=F32, device=dev)
        self.bgd = wp.zeros((L, ndiv - 1), dtype=F32, device=dev)
        self.fx = wp.zeros(L, dtype=F32, device=dev); self.fy = wp.zeros(L, dtype=F32, device=dev)
        self.fg = wp.zeros(L, dtype=F32, device=dev)
        self.msk = wp.zeros(L, dtype=wp.int32, device=dev)
        self.gx = wp.zeros(L, dtype=F32, device=dev); self.gy = wp.zeros(L, dtype=F32, device=dev)
        self.gi = wp.zeros(L, dtype=wp.int32, device=dev)

    # -------------------------------------------------------------- helpers
    def _a0(self, W):
        return -(W @ self.wt) / (np.pi * self.U)                              # (L,)

    def _an1(self, W):
        return 2.0 * (W * self.wt[None, :]) @ self.cosn[:, 0] / (np.pi * self.U)

    def _gamb(self, W):
        return np.pi * self.c * self.U * (self._a0(W) + 0.5 * self._an1(W))

    def _ucol(self, px, py, sx, sy):
        """Closed-form unit-ccw Vatistas velocity of one source per lane at that lane's
        stations: returns downwash-column contribution (L, ndiv)."""
        rx = px - sx[:, None]; ry = py - sy[:, None]
        r2 = rx * rx + ry * ry
        f = (1.0 / (2.0 * np.pi)) / np.sqrt(r2 * r2 + (self.rc ** 4)[:, None])
        return -f * ry, f * rx

    def _wake_at_stations(self, wxs, wys):
        wp.copy(self.px, wp.array(wxs.astype(np.float32), dtype=F32, device=self.dev))
        wp.copy(self.py, wp.array(wys.astype(np.float32), dtype=F32, device=self.dev))
        wp.launch(k_wake_at_pts, dim=(self.L, self.ndiv),
                  inputs=[self.px, self.py, self.wx, self.wy, self.wgd, self.wn, self.rc4],
                  outputs=[self.ou, self.owd], device=self.dev)
        return self.ou.numpy().astype(float), self.owd.numpy().astype(float)

    def _gather(self, idx):
        wp.copy(self.gi, wp.array(idx.astype(np.int32), dtype=wp.int32, device=self.dev))
        wp.launch(k_gather, dim=self.L, inputs=[self.wx, self.wy, self.gi],
                  outputs=[self.gx, self.gy], device=self.dev)
        return self.gx.numpy().astype(float), self.gy.numpy().astype(float)

    def _append(self, fx, fy, fg, mask):
        wp.copy(self.fx, wp.array(fx.astype(np.float32), dtype=F32, device=self.dev))
        wp.copy(self.fy, wp.array(fy.astype(np.float32), dtype=F32, device=self.dev))
        wp.copy(self.fg, wp.array(fg.astype(np.float32), dtype=F32, device=self.dev))
        wp.copy(self.msk, wp.array(mask.astype(np.int32), dtype=wp.int32, device=self.dev))
        wp.copy(self.wn, wp.array(self.wn_np, dtype=wp.int32, device=self.dev))
        wp.launch(k_append, dim=self.L, inputs=[self.wx, self.wy, self.wgd, self.wn,
                  self.fx, self.fy, self.fg, self.msk], device=self.dev)
        slot = self.wn_np.copy()
        self.wg_np[np.arange(self.L)[mask > 0], slot[mask > 0]] = fg[mask > 0]
        self.wn_np = self.wn_np + (mask > 0)
        return slot                                                            # birth slots

    # -------------------------------------------------------------- step
    def step(self, alpha, dalpha, hdot):
        """alpha/dalpha/hdot: (L,) arrays. Returns dict of (L,) CLf/CDf/A0."""
        self.it += 1
        L, U, dt, ndiv = self.L, self.U, self.dt, self.ndiv
        ca, sa = np.cos(alpha), np.sin(alpha)
        self.sx -= U * dt; self.sy += hdot * dt
        wxs = self.sx[:, None] + self.xs * ca[:, None]                        # stations world
        wys = self.sy[:, None] - self.xs * sa[:, None]
        assert self.wn_np.max() + 2 < self.wmax, "wake capacity exceeded"

        ui, wi = self._wake_at_stations(wxs, wys)
        Wd = (-U * sa[:, None] - ui * sa[:, None] + hdot[:, None] * ca[:, None]
              - wi * ca[:, None] - dalpha[:, None] * self.xs
              + self.dzc * (ui * ca[:, None] + U * ca[:, None]
                            + hdot[:, None] * sa[:, None] - wi * sa[:, None]))

        # fresh TEV placement: 1/3 rule toward last TEV (first: TE + 0.5*U*dt in x)
        tex = self.sx + self.c * ca; tey = self.sy - self.c * sa
        ltx, lty = self._gather(self.last_tev)
        first = self.last_tev < 0
        ntx = np.where(first, tex + 0.5 * U * dt, tex + (ltx - tex) / 3.0)
        nty = np.where(first, tey, tey + (lty - tey) / 3.0)
        tcu, tcw = self._ucol(wxs, wys, ntx, nty)
        tcol = (-(tcu * sa[:, None] + tcw * ca[:, None])
                + self.dzc * (tcu * ca[:, None] - tcw * sa[:, None]))

        S_old = self.wg_np[np.arange(L), :].sum(1)
        G0 = self._gamb(Wd); GT = self._gamb(tcol)
        gT = (G0 - S_old) / (1.0 - GT)

        W_pre = Wd + gT[:, None] * tcol
        AFp = np.stack([self._a0(W_pre),
                        2.0 * (W_pre * self.wt[None, :]) @ self.cosn[:, 0] / (np.pi * U),
                        2.0 * (W_pre * self.wt[None, :]) @ self.cosn[:, 1] / (np.pi * U),
                        2.0 * (W_pre * self.wt[None, :]) @ self.cosn[:, 2] / (np.pi * U)], 1)
        dAF = (AFp - self._AF_prev) / dt if self.it > 1 else np.zeros((L, 4))
        a0 = AFp[:, 0]

        shed = np.abs(a0) > self.crit
        gL = np.zeros(L)
        nlx = np.zeros(L); nly = np.zeros(L)
        if shed.any():
            lex = self.sx.copy(); ley = self.sy.copy()                        # LE (pivot) world
            llx, lly = self._gather(self.last_lev)
            cont = (self.lev_prev_it == self.it - 1) & (self.last_lev >= 0)
            le_u = U + ui[:, 0]; le_w = -hdot + wi[:, 0]
            nlx = np.where(cont, lex + (llx - lex) / 3.0, lex + 0.5 * le_u * dt)
            nly = np.where(cont, ley + (lly - ley) / 3.0, ley + 0.5 * le_w * dt)
            lcu, lcw = self._ucol(wxs, wys, nlx, nly)
            lcol = (-(lcu * sa[:, None] + lcw * ca[:, None])
                    + self.dzc * (lcu * ca[:, None] - lcw * sa[:, None]))
            GLc = self._gamb(lcol)
            a0T = self._a0(tcol); a0L = self._a0(lcol)
            # 2x2: [[1-GT, 1-GL],[a0T, a0L]] @ [gT,gL] = [G0-S_old, sign*crit - a0(Wd)]
            b1 = G0 - S_old; b2 = np.sign(a0) * self.crit - self._a0(Wd)
            det = (1.0 - GT) * a0L - (1.0 - GLc) * a0T
            det = np.where(np.abs(det) < 1e-12, 1e-12, det)
            gT2 = (b1 * a0L - (1.0 - GLc) * b2) / det
            gL2 = ((1.0 - GT) * b2 - a0T * b1) / det
            gT = np.where(shed, gT2, gT)
            gL = np.where(shed, gL2, 0.0)

        # final downwash + full Fourier series
        Wt = Wd + gT[:, None] * tcol
        if shed.any():
            Wt = Wt + gL[:, None] * lcol
        AF = np.stack([self._a0(Wt),
                       2.0 * (Wt * self.wt[None, :]) @ self.cosn[:, 0] / (np.pi * U),
                       2.0 * (Wt * self.wt[None, :]) @ self.cosn[:, 1] / (np.pi * U),
                       2.0 * (Wt * self.wt[None, :]) @ self.cosn[:, 2] / (np.pi * U)], 1)
        self._AF_prev = AF.copy()

        An = 2.0 * (Wt * self.wt[None, :]) @ self.cosn / (np.pi * U)          # (L, naterm)
        gam_th = AF[:, 0][:, None] * (1.0 + np.cos(self.th))[None, :] \
            + (An @ self.sinn.T) * np.sin(self.th)[None, :]
        gam_th = gam_th * U * self.c[:, None]
        bv = 0.5 * (gam_th[:, 1:] + gam_th[:, :-1]) * np.diff(self.th)[None, :]
        bmx = 0.5 * (wxs[:, 1:] + wxs[:, :-1]); bmy = 0.5 * (wys[:, 1:] + wys[:, :-1])

        # append fresh vortices (TEV all lanes; LEV shed lanes), track placement refs
        slots = self._append(ntx, nty, gT, np.ones(L, dtype=np.int32))
        self.last_tev = slots.astype(np.int32)
        if shed.any():
            slots2 = self._append(nlx, nly, gL, shed.astype(np.int32))
            self.last_lev = np.where(shed, slots2, self.last_lev).astype(np.int32)
            self.lev_prev_it = np.where(shed, self.it, self.lev_prev_it)

        # forces (calc_forces): wake induction incl. fresh at stations
        u2, w2 = self._wake_at_stations(wxs, wys)
        u_wk = u2 * ca[:, None] - w2 * sa[:, None]
        cnc = 2.0 * np.pi * (U * ca + hdot * sa) / U * (AF[:, 0] + 0.5 * AF[:, 1])
        cnnc = 2.0 * np.pi * self.c * (0.75 * dAF[:, 0] + 0.25 * dAF[:, 1]
                                       + 0.125 * dAF[:, 2]) / U
        nonl = 2.0 * np.einsum("li,li->l", u_wk[:, :-1], bv) / (U * U * self.c)
        cn = cnc + cnnc + nonl
        cs = 2.0 * np.pi * AF[:, 0] ** 2

        # convect (double-buffer) with bound segments
        wp.copy(self.bxd, wp.array(bmx.astype(np.float32), dtype=F32, device=self.dev))
        wp.copy(self.byd, wp.array(bmy.astype(np.float32), dtype=F32, device=self.dev))
        wp.copy(self.bgd, wp.array((-bv).astype(np.float32), dtype=F32, device=self.dev))
        wp.copy(self.wn, wp.array(self.wn_np, dtype=wp.int32, device=self.dev))
        wp.launch(k_convect, dim=(self.L, int(self.wn_np.max())),
                  inputs=[self.wx, self.wy, self.wgd, self.wn, self.rc4,
                          self.bxd, self.byd, self.bgd, ndiv - 1, F32(dt)],
                  outputs=[self.nx_, self.ny_], device=self.dev)
        self.wx, self.nx_ = self.nx_, self.wx
        self.wy, self.ny_ = self.ny_, self.wy

        return dict(CLf=cn * ca + cs * sa, CDf=cn * sa - cs * ca, A0=AF[:, 0], lesp=a0)


if __name__ == "__main__":
    import time
    sys_path_note = None
    wp.init()
    # (1) steady ladder: CL -> Wagner-limited 2*pi*sin(a)
    L = 4
    eng = LDVMBatch(chords=np.full(L, 1.0), lesp_crit=np.full(L, 99.0), U=1.0, dt=0.015,
                    rho=1.0, camber_m=0.0)
    als = np.radians([2.0, 5.0, 8.0, 12.0])
    t0 = time.time()
    for _ in range(400):
        r = eng.step(als, np.zeros(L), np.zeros(L))
    print("(1) steady:", " ".join(f"a={np.degrees(a):.0f}: CLf={c:+.3f}/{2*np.pi*np.sin(a):+.3f}"
          for a, c in zip(als, r["CLf"])), f"({time.time()-t0:.0f}s)", flush=True)
    # (2) unsteady vs CPU fp64 LDVM2D: pitching+plunging strip with LEV
    import sys as _s
    _s.path.insert(0, "platform")
    from ldvm_fourier import LDVM2D
    U, c, f = 8.0, 0.287, 2.3
    dt = 0.015 * c / U
    Om = 2 * np.pi * f
    eng2 = LDVMBatch(chords=np.array([c, c]), lesp_crit=np.array([0.14, 99.0]), U=U, dt=dt)
    cpu = LDVM2D(U=U, c=c, dt=dt, lesp_crit=0.14, camber_m=0.02, camber_p=0.40, rho=1.225)
    N = int(2.0 / f / dt)
    clg = []; clc = []
    t0 = time.time()
    for it in range(N):
        t = it * dt
        a = np.radians(5.0) + 0.3 * np.sin(Om * t)
        da = 0.3 * Om * np.cos(Om * t)
        hd = 0.5 * 2.0 * np.pi * f * 0.4 * np.cos(Om * t)
        r = eng2.step(np.array([a, a]), np.array([da, da]), np.array([hd, hd]))
        clg.append(r["CLf"][0])
        rc = cpu.step(a, da, hd)
        clc.append(rc["CLf"])
    clg = np.array(clg); clc = np.array(clc)
    m = slice(N // 2, N)
    print(f"(2) unsteady vs CPU fp64: mean CLf gpu={clg[m].mean():+.4f} cpu={clc[m].mean():+.4f} "
          f"maxdiff={np.max(np.abs(clg-clc)):.4f}  ({time.time()-t0:.0f}s)", flush=True)
