"""CUDA-only numerical port of :class:`ldvm_fourier.LDVM2D`.

Python controls the step loop and performs final serialization.  All
aerodynamic arithmetic, wake induction, Fourier projection, the coupled
TEV/LEV solve, force assembly, and wake convection use torch CUDA float64.
There is deliberately no CPU numerical fallback.
"""
from __future__ import annotations

import math

import torch


class LDVM2DCuda:
    """One source-faithful LDVM lane with all numerical state on CUDA."""

    def __init__(
        self,
        *,
        U: float = 1.0,
        c: float = 1.0,
        ndiv: int = 70,
        naterm: int = 35,
        dt: float = 0.015,
        rho: float = 1.0,
        lesp_crit: float = 0.11,
        pivot_xc: float = 0.0,
        core_rc: float = 0.02,
        max_wake: int = 100000,
        source_parity: bool = False,
        device: str = "cuda:0",
    ) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("LDVM2DCuda requires an available CUDA device")
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise RuntimeError(f"LDVM2DCuda forbids device {device!r}")
        self.dtype = torch.float64
        self.U = float(U)
        self.c = float(c)
        self.rho = float(rho)
        self.dt = float(dt)
        self.ndiv = int(ndiv)
        self.naterm = int(naterm)
        self.lesp_crit = float(lesp_crit)
        self.xp = float(pivot_xc) * self.c
        self.rc = float(core_rc) * self.c
        self.max_wake = int(max_wake)
        if type(source_parity) is not bool:
            raise TypeError("source_parity must be an exact bool")
        self.source_parity = source_parity
        if self.ndiv < 4 or self.naterm < 3 or self.max_wake < 2:
            raise ValueError("invalid LDVM discretization")

        kw = {"device": self.device, "dtype": self.dtype}
        self.th = torch.linspace(0.0, math.pi, self.ndiv, **kw)
        self.xs = 0.5 * self.c * (1.0 - torch.cos(self.th))
        self.dzc = torch.zeros(self.ndiv, **kw)
        self._cosn = torch.cos(
            self.th[:, None]
            * torch.arange(1, self.naterm + 1, device=self.device, dtype=self.dtype)[
                None, :
            ]
        )
        self._sinn = torch.sin(
            self.th[:, None]
            * torch.arange(1, self.naterm + 1, device=self.device, dtype=self.dtype)[
                None, :
            ]
        )

        # One spare slot permits append -> convect -> trim, matching LDVM2D order.
        capacity = self.max_wake + 1
        self.tx = torch.zeros(capacity, **kw)
        self.ty = torch.zeros(capacity, **kw)
        self.tg = torch.zeros(capacity, **kw)
        self.lx = torch.zeros(capacity, **kw)
        self.ly = torch.zeros(capacity, **kw)
        self.lg = torch.zeros(capacity, **kw)
        self.nt = 0
        # LEV shedding is data dependent.  Keep its count, circular-buffer slot,
        # and consecutive-shed state on CUDA so no host ``.item()`` branch can
        # alter the aerodynamic trajectory.
        self._nl_total = torch.zeros((), device=self.device, dtype=torch.int64)
        self._lev_prev_it = torch.full((), -99, device=self.device, dtype=torch.int64)
        self._last_lev_x = torch.zeros((), **kw)
        self._last_lev_y = torch.zeros((), **kw)
        self._lev_shed_current = torch.zeros((), device=self.device, dtype=torch.bool)
        self._lev_buffer_was_full = torch.zeros(
            (), device=self.device, dtype=torch.bool
        )
        self.it = 0
        self.gam_lost = torch.zeros((), **kw)
        self._AF_prev = torch.zeros(4, **kw)
        self.sx = torch.zeros((), **kw)
        self.sy = torch.zeros((), **kw)
        self._ca = torch.ones((), **kw)
        self._sa = torch.zeros((), **kw)
        self.wake_convection_count = 0
        self.cuda_stream = torch.cuda.current_stream(self.device)

    def __getstate__(self) -> dict[str, object]:
        """Serialize scientific CUDA state without the process-local stream.

        Q16 predictor/corrector trials fork the complete aerodynamic owner via
        pickle.  ``torch.cuda.Stream`` is a process-local execution handle and
        cannot be pickled; it is not scientific state.  Every tensor containing
        the LDVM trajectory remains in the serialized dictionary.
        """
        state = self.__dict__.copy()
        state.pop("cuda_stream", None)
        return state

    def __setstate__(self, state: dict[str, object]) -> None:
        """Restore a branch and bind it to the current CUDA stream."""
        self.__dict__.update(state)
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("LDVM2DCuda branch restore requires CUDA")
        self.cuda_stream = torch.cuda.current_stream(self.device)

    @property
    def numerical_device(self) -> str:
        return str(self.device)

    def _world(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            self.sx + (x - self.xp) * self._ca,
            self.sy - (x - self.xp) * self._sa,
        )

    def _a0(self, W: torch.Tensor) -> torch.Tensor:
        return -torch.trapezoid(W, self.th) / (math.pi * self.U)

    def _an(self, W: torch.Tensor, n: int) -> torch.Tensor:
        return (
            2.0
            * torch.trapezoid(W * torch.cos(float(n) * self.th), self.th)
            / (math.pi * self.U)
        )

    def _gamb(self, W: torch.Tensor) -> torch.Tensor:
        return math.pi * self.c * self.U * (self._a0(W) + 0.5 * self._an(W, 1))

    def _induced_many(
        self,
        px: torch.Tensor,
        py: torch.Tensor,
        vx: torch.Tensor,
        vy: torch.Tensor,
        vg: torch.Tensor,
        rc: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if vx.numel() == 0:
            return torch.zeros_like(px), torch.zeros_like(py)
        rx = px[:, None] - vx[None, :]
        ry = py[:, None] - vy[None, :]
        r2 = rx * rx + ry * ry
        factor = torch.rsqrt(r2 * r2 + rc[None, :] ** 4)
        weighted = vg[None, :] * (0.5 / math.pi) * factor
        return torch.sum(weighted * (-ry), dim=1), torch.sum(weighted * rx, dim=1)

    def _wcol(self, px: torch.Tensor, py: torch.Tensor) -> torch.Tensor:
        src_x = px.reshape(1)
        src_y = py.reshape(1)
        src_g = torch.ones(1, device=self.device, dtype=self.dtype)
        src_r = torch.full((1,), self.rc, device=self.device, dtype=self.dtype)
        u, w = self._induced_many(self._wx, self._wy, src_x, src_y, src_g, src_r)
        return -(u * self._sa + w * self._ca) + self.dzc * (u * self._ca - w * self._sa)

    def _append_te(self, x: torch.Tensor, y: torch.Tensor, g: torch.Tensor) -> None:
        self.tx[self.nt] = x
        self.ty[self.nt] = y
        self.tg[self.nt] = g
        self.nt += 1

    def _append_le_masked(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        g: torch.Tensor,
        shed: torch.Tensor,
    ) -> None:
        """Append to the packed CUDA LEV buffer without a host decision."""
        slot = torch.minimum(
            self._nl_total,
            torch.as_tensor(self.max_wake, device=self.device),
        ).reshape(1)
        buffer_full = self._nl_total >= self.max_wake
        append_x = self.lx.clone().scatter(0, slot, x.reshape(1))
        append_y = self.ly.clone().scatter(0, slot, y.reshape(1))
        append_g = self.lg.clone().scatter(0, slot, g.reshape(1))
        self.lx.copy_(torch.where(shed, append_x, self.lx))
        self.ly.copy_(torch.where(shed, append_y, self.ly))
        self.lg.copy_(torch.where(shed, append_g, self.lg))
        self._last_lev_x.copy_(torch.where(shed, x, self._last_lev_x))
        self._last_lev_y.copy_(torch.where(shed, y, self._last_lev_y))
        self._lev_shed_current.copy_(shed)
        self._lev_buffer_was_full.copy_(buffer_full)
        self._lev_prev_it.copy_(
            torch.where(
                shed,
                torch.as_tensor(self.it, device=self.device, dtype=torch.int64),
                self._lev_prev_it,
            )
        )
        self._nl_total.add_(shed.to(dtype=torch.int64))

    def _trim_le_masked(self) -> None:
        """Apply append-convect-trim ordering entirely on CUDA."""
        trim = self._lev_shed_current & self._lev_buffer_was_full
        self.gam_lost.add_(torch.where(trim, self.lg[0], 0.0))
        zero = torch.zeros(1, device=self.device, dtype=self.dtype)
        shifted_x = torch.cat((self.lx[1:], zero))
        shifted_y = torch.cat((self.ly[1:], zero))
        shifted_g = torch.cat((self.lg[1:], zero))
        self.lx.copy_(torch.where(trim, shifted_x, self.lx))
        self.ly.copy_(torch.where(trim, shifted_y, self.ly))
        self.lg.copy_(torch.where(trim, shifted_g, self.lg))

    def _trim(self, x: torch.Tensor, y: torch.Tensor, g: torch.Tensor, n: int) -> int:
        if n <= self.max_wake:
            return n
        k = n - self.max_wake
        self.gam_lost.add_(torch.sum(g[:k]))
        x[: self.max_wake].copy_(x[k:n].clone())
        y[: self.max_wake].copy_(y[k:n].clone())
        g[: self.max_wake].copy_(g[k:n].clone())
        return self.max_wake

    @torch.no_grad()
    def step(
        self,
        alpha: float | torch.Tensor,
        dalpha: float | torch.Tensor,
        hdot: float | torch.Tensor = 0.0,
    ) -> dict[str, torch.Tensor]:
        self.it += 1
        alpha_t = torch.as_tensor(alpha, device=self.device, dtype=self.dtype)
        dalpha_t = torch.as_tensor(dalpha, device=self.device, dtype=self.dtype)
        hdot_t = torch.as_tensor(hdot, device=self.device, dtype=self.dtype)
        ca, sa = torch.cos(alpha_t), torch.sin(alpha_t)
        self._ca, self._sa = ca, sa
        self.sx.sub_(self.U * self.dt)
        self.sy.add_(hdot_t * self.dt)
        self._wx, self._wy = self._world(self.xs)

        tvx, tvy, tvg = self.tx[: self.nt], self.ty[: self.nt], self.tg[: self.nt]
        lvx, lvy, lvg = self.lx, self.ly, self.lg
        rct = torch.full((self.nt,), self.rc, device=self.device, dtype=self.dtype)
        rcl = torch.full(
            (self.max_wake + 1,), self.rc, device=self.device, dtype=self.dtype
        )
        uu, ww = self._induced_many(self._wx, self._wy, tvx, tvy, tvg, rct)
        ul, wl = self._induced_many(self._wx, self._wy, lvx, lvy, lvg, rcl)
        ui, wi = uu + ul, ww + wl
        W = (
            -self.U * sa
            - ui * sa
            + hdot_t * ca
            - wi * ca
            - dalpha_t * (self.xs - self.xp)
            + self.dzc * (ui * ca + self.U * ca + hdot_t * sa - wi * sa)
        )

        c_t = torch.as_tensor(self.c, device=self.device, dtype=self.dtype)
        tex, tey = self._world(c_t)
        if self.nt:
            ntx = tex + (self.tx[self.nt - 1] - tex) / 3.0
            nty = tey + (self.ty[self.nt - 1] - tey) / 3.0
        else:
            ntx, nty = tex + 0.5 * self.U * self.dt, tey
        tcol = self._wcol(ntx, nty)

        S_old = torch.sum(tvg) + torch.sum(lvg) + self.gam_lost
        G0, GT = self._gamb(W), self._gamb(tcol)
        gT = (G0 - S_old) / (1.0 - GT)
        gT_te_only_provisional = gT.clone()
        W_pre = W + gT * tcol
        AF_pre = torch.stack(
            (
                self._a0(W_pre),
                self._an(W_pre, 1),
                self._an(W_pre, 2),
                self._an(W_pre, 3),
            )
        )
        dAF = (
            (AF_pre - self._AF_prev) / self.dt
            if self.it > 1
            else torch.zeros_like(AF_pre)
        )
        a0 = AF_pre[0]
        shed_lev = torch.abs(a0) > self.lesp_crit
        zero = torch.zeros((), device=self.device, dtype=self.dtype)
        lex, ley = self._world(zero)
        consecutive = self._lev_prev_it == self.it - 1
        s0 = -self.xp
        new_tev_u = zero
        new_tev_w = zero
        if self.source_parity:
            provisional_u, provisional_w = self._induced_many(
                lex.reshape(1),
                ley.reshape(1),
                ntx.reshape(1),
                nty.reshape(1),
                gT_te_only_provisional.reshape(1),
                torch.full((1,), self.rc, device=self.device, dtype=self.dtype),
            )
            new_tev_u = provisional_u[0]
            new_tev_w = provisional_w[0]
        le_u = self.U + s0 * sa * dalpha_t + ui[0] + new_tev_u
        le_w = -hdot_t + s0 * ca * dalpha_t + wi[0] + new_tev_w
        first_nlx = lex + 0.5 * le_u * self.dt
        first_nly = ley + 0.5 * le_w * self.dt
        previous_slot = torch.clamp(
            torch.minimum(
                self._nl_total,
                torch.as_tensor(self.max_wake, device=self.device),
            )
            - 1,
            min=0,
        ).reshape(1)
        previous_nlx = torch.gather(self.lx, 0, previous_slot).reshape(())
        previous_nly = torch.gather(self.ly, 0, previous_slot).reshape(())
        nlx = torch.where(
            consecutive,
            lex + (previous_nlx - lex) / 3.0,
            first_nlx,
        )
        nly = torch.where(
            consecutive,
            ley + (previous_nly - ley) / 3.0,
            first_nly,
        )
        lcol = self._wcol(nlx, nly)
        GL = self._gamb(lcol)
        a0T, a0L = self._a0(tcol), self._a0(lcol)
        b1 = G0 - S_old
        b2 = torch.sign(a0) * self.lesp_crit - self._a0(W)
        det = (1.0 - GT) * a0L - (1.0 - GL) * a0T
        coupled_gT = (b1 * a0L - (1.0 - GL) * b2) / det
        coupled_gL = ((1.0 - GT) * b2 - a0T * b1) / det
        gT = torch.where(shed_lev, coupled_gT, gT)
        gL = torch.where(shed_lev, coupled_gL, zero)
        self._append_le_masked(nlx, nly, gL, shed_lev)
        first_tev_zeroed = self.source_parity and self.it == 1
        gT_stored = zero if first_tev_zeroed else gT
        self._append_te(ntx, nty, gT_stored)

        Wt = W + gT * tcol + gL * lcol
        AF = torch.stack(
            (self._a0(Wt), self._an(Wt, 1), self._an(Wt, 2), self._an(Wt, 3))
        )
        self._AF_prev.copy_(AF)
        An = (
            2.0
            * torch.trapezoid(
                Wt[:, None] * self._cosn,
                self.th,
                dim=0,
            )
            / (math.pi * self.U)
        )
        gam_th = AF[0] * (1.0 + torch.cos(self.th)) + torch.sum(
            An[None, :] * self._sinn, dim=1
        ) * torch.sin(self.th)
        gam_th = gam_th * self.U * self.c
        bv = 0.5 * (gam_th[1:] + gam_th[:-1]) * torch.diff(self.th)
        bx = 0.5 * (self._wx[1:] + self._wx[:-1])
        by = 0.5 * (self._wy[1:] + self._wy[:-1])

        tvx, tvy, tvg = self.tx[: self.nt], self.ty[: self.nt], self.tg[: self.nt]
        lvx, lvy, lvg = self.lx, self.ly, self.lg
        rct = torch.full((self.nt,), self.rc, device=self.device, dtype=self.dtype)
        rcl = torch.full(
            (self.max_wake + 1,), self.rc, device=self.device, dtype=self.dtype
        )
        u2, w2 = self._induced_many(self._wx, self._wy, tvx, tvy, tvg, rct)
        u3, w3 = self._induced_many(self._wx, self._wy, lvx, lvy, lvg, rcl)
        u_wk = (u2 + u3) * ca - (w2 + w3) * sa
        cnc = (
            2.0 * math.pi * (self.U * ca + hdot_t * sa) / self.U * (AF[0] + 0.5 * AF[1])
        )
        cnnc = (
            2.0
            * math.pi
            * self.c
            * (0.75 * dAF[0] + 0.25 * dAF[1] + 0.125 * dAF[2])
            / self.U
        )
        nonl = 2.0 * torch.sum(u_wk[:-1] * bv) / (self.U * self.U * self.c)
        cn = cnc + cnnc + nonl
        cs = 2.0 * math.pi * AF[0] * AF[0]

        wxa = torch.cat((tvx, lvx))
        wya = torch.cat((tvy, lvy))
        ub, wb = self._induced_many(
            wxa,
            wya,
            bx,
            by,
            -bv,
            torch.full((bv.numel(),), self.rc, device=self.device, dtype=self.dtype),
        )
        ut, wt = self._induced_many(wxa, wya, tvx, tvy, tvg, rct)
        ulv, wlv = self._induced_many(wxa, wya, lvx, lvy, lvg, rcl)
        uc, wc = ub + ut + ulv, wb + wt + wlv
        self.tx[: self.nt].add_(uc[: self.nt] * self.dt)
        self.ty[: self.nt].add_(wc[: self.nt] * self.dt)
        self.lx.add_(uc[self.nt :] * self.dt)
        self.ly.add_(wc[self.nt :] * self.dt)
        self.wake_convection_count += 1
        self.nt = self._trim(self.tx, self.ty, self.tg, self.nt)
        self._trim_le_masked()

        gcum01 = torch.sum(bv[(0.5 * (self.xs[1:] + self.xs[:-1])) <= 0.10 * self.c])
        return {
            "CLf": cn * ca + cs * sa,
            "CDf": cn * sa - cs * ca,
            "CNf": cn,
            "CNc": cnc,
            "CNnc": cnnc,
            "CNnonl": nonl,
            "CSf": cs,
            "A0": AF[0],
            "lesp": a0,
            "lesp_pre": a0,
            "lesp_constraint_residual": torch.where(
                shed_lev,
                AF[0] - torch.sign(a0) * self.lesp_crit,
                zero,
            ),
            "gcum01": gcum01,
            "n_lev": torch.minimum(
                self._nl_total,
                torch.as_tensor(self.max_wake, device=self.device),
            ),
            "n_tev": torch.as_tensor(
                self.nt, device=self.device, dtype=torch.int64
            ),
            "shed_lev": shed_lev,
            "gamma_lev_new": gL,
            "gamma_tev_new_solved": gT,
            "gamma_tev_new_persisted": gT_stored,
            "lev_birth_position": torch.stack((nlx, nly)),
            "tev_birth_position": torch.stack((ntx, nty)),
            "lev_edge_position": torch.stack((lex, ley)),
            "tev_edge_position": torch.stack((tex, tey)),
            "tev_strength_te_only_provisional": gT_te_only_provisional,
            "first_tev_zeroed": torch.as_tensor(
                first_tev_zeroed, device=self.device, dtype=torch.bool
            ),
            "source_parity": self.source_parity,
        }


__all__ = ["LDVM2DCuda"]
