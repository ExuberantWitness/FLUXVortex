"""Span-parallel CUDA source bank for node-owned V5M LEV ribbons.

This is the source-only, batched counterpart of :mod:`ldvm_torch_gpu`.  One
CUDA tensor dimension owns all independent sectional lanes, so an entire
spanwise row is advanced by the same kernel launches.  It exports newborn
circulation and placement facts only; it never exports a two-dimensional load.

The Python object is deliberately pickleable because Q16 predictor/corrector
transactions clone the complete aerodynamic owner.  All trajectory-defining
arrays remain CUDA float64 tensors and there is no CPU numerical fallback.
"""
from __future__ import annotations

import math
from typing import Any

import torch


def _cuda_vector(
    name: str,
    value: float | torch.Tensor,
    *,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    if type(value) is torch.Tensor:
        if value.device != device or value.dtype is not torch.float64:
            raise ValueError(f"{name} must be CUDA float64 on {device}")
        if value.ndim == 0:
            result = value.expand(batch_size).clone()
        elif value.shape == (batch_size,):
            result = value.clone()
        else:
            raise ValueError(f"{name} must be scalar or shape (batch_size,)")
    else:
        scalar = float(value)
        result = torch.full(
            (batch_size,), scalar, device=device, dtype=torch.float64
        )
    if not bool(torch.isfinite(result).all().item()):
        raise ValueError(f"{name} must be finite")
    return result


class CudaLDVMSourceBank:
    """A synchronous row of independent source-faithful LDVM sections."""

    def __init__(
        self,
        *,
        batch_size: int,
        ndiv: int = 70,
        naterm: int = 35,
        delta_time_convective: float | torch.Tensor = 0.015,
        lesp_crit: float | torch.Tensor = 0.11,
        pivot_fraction_chord: float | torch.Tensor = 0.0,
        core_radius_chord: float | torch.Tensor = 0.02,
        max_wake: int = 100000,
        source_parity: bool = True,
        device: str = "cuda:0",
    ) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("CudaLDVMSourceBank requires CUDA")
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise RuntimeError("CudaLDVMSourceBank forbids CPU devices")
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch_size must be a positive exact int")
        if type(ndiv) is not int or ndiv < 4:
            raise ValueError("ndiv must be an exact int >= 4")
        if type(naterm) is not int or naterm < 3:
            raise ValueError("naterm must be an exact int >= 3")
        if type(max_wake) is not int or max_wake < 2:
            raise ValueError("max_wake must be an exact int >= 2")
        if type(source_parity) is not bool or not source_parity:
            raise ValueError("the production source bank requires source_parity=True")

        self.batch_size = batch_size
        self.ndiv = ndiv
        self.naterm = naterm
        self.max_wake = max_wake
        self.source_parity = source_parity
        self.dtype = torch.float64
        self.dt = _cuda_vector(
            "delta_time_convective",
            delta_time_convective,
            batch_size=batch_size,
            device=self.device,
        )
        self.lesp_crit = _cuda_vector(
            "lesp_crit", lesp_crit, batch_size=batch_size, device=self.device
        )
        self.xp = _cuda_vector(
            "pivot_fraction_chord",
            pivot_fraction_chord,
            batch_size=batch_size,
            device=self.device,
        )
        self.rc = _cuda_vector(
            "core_radius_chord",
            core_radius_chord,
            batch_size=batch_size,
            device=self.device,
        )
        if bool(torch.any(self.dt <= 0.0).item()):
            raise ValueError("delta_time_convective must be positive")
        if bool(torch.any(self.lesp_crit <= 0.0).item()):
            raise ValueError("lesp_crit must be positive")
        if bool(torch.any((self.xp < 0.0) | (self.xp > 1.0)).item()):
            raise ValueError("pivot_fraction_chord must lie in [0,1]")
        if bool(torch.any(self.rc <= 0.0).item()):
            raise ValueError("core_radius_chord must be positive")

        kw = {"device": self.device, "dtype": self.dtype}
        self.th = torch.linspace(0.0, math.pi, ndiv, **kw)
        self.xs = 0.5 * (1.0 - torch.cos(self.th))
        self.dzc = torch.zeros((batch_size, ndiv), **kw)
        orders = torch.arange(1, naterm + 1, **kw)
        self._cosn = torch.cos(self.th[:, None] * orders[None, :])
        self._sinn = torch.sin(self.th[:, None] * orders[None, :])

        capacity = max_wake + 1
        shape = (batch_size, capacity)
        self.tx = torch.zeros(shape, **kw)
        self.ty = torch.zeros(shape, **kw)
        self.tg = torch.zeros(shape, **kw)
        self.lx = torch.zeros(shape, **kw)
        self.ly = torch.zeros(shape, **kw)
        self.lg = torch.zeros(shape, **kw)
        self.nt = 0
        self._nl_total = torch.zeros(
            batch_size, device=self.device, dtype=torch.int64
        )
        self._lev_prev_it = torch.full(
            (batch_size,), -99, device=self.device, dtype=torch.int64
        )
        self._last_lev_x = torch.zeros(batch_size, **kw)
        self._last_lev_y = torch.zeros(batch_size, **kw)
        self._lev_shed_current = torch.zeros(
            batch_size, device=self.device, dtype=torch.bool
        )
        self._lev_buffer_was_full = torch.zeros_like(self._lev_shed_current)
        self.it = 0
        self.gam_lost = torch.zeros(batch_size, **kw)
        self._AF_prev = torch.zeros((batch_size, 4), **kw)
        self.sx = torch.zeros(batch_size, **kw)
        self.sy = torch.zeros(batch_size, **kw)
        self._ca = torch.ones(batch_size, **kw)
        self._sa = torch.zeros(batch_size, **kw)
        self.wake_convection_count = 0
        self.cuda_stream = torch.cuda.current_stream(self.device)

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state.pop("cuda_stream", None)
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("CudaLDVMSourceBank restore requires CUDA")
        self.cuda_stream = torch.cuda.current_stream(self.device)

    @property
    def numerical_device(self) -> str:
        return str(self.device)

    def _world(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        relative = x - self.xp[:, None]
        return (
            self.sx[:, None] + relative * self._ca[:, None],
            self.sy[:, None] - relative * self._sa[:, None],
        )

    def _a0(self, w: torch.Tensor) -> torch.Tensor:
        return -torch.trapezoid(w, self.th, dim=1) / math.pi

    def _an(self, w: torch.Tensor, order: int) -> torch.Tensor:
        return (
            2.0
            * torch.trapezoid(
                w * torch.cos(float(order) * self.th)[None, :], self.th, dim=1
            )
            / math.pi
        )

    def _gamb(self, w: torch.Tensor) -> torch.Tensor:
        return math.pi * (self._a0(w) + 0.5 * self._an(w, 1))

    def _induced_many(
        self,
        px: torch.Tensor,
        py: torch.Tensor,
        vx: torch.Tensor,
        vy: torch.Tensor,
        vg: torch.Tensor,
        rc: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if vx.shape[1] == 0:
            return torch.zeros_like(px), torch.zeros_like(py)
        rx = px[:, :, None] - vx[:, None, :]
        ry = py[:, :, None] - vy[:, None, :]
        r2 = rx * rx + ry * ry
        factor = torch.rsqrt(r2 * r2 + rc[:, None, :] ** 4)
        weighted = vg[:, None, :] * (0.5 / math.pi) * factor
        return (
            torch.sum(weighted * (-ry), dim=2),
            torch.sum(weighted * rx, dim=2),
        )

    def _wcol(self, px: torch.Tensor, py: torch.Tensor) -> torch.Tensor:
        unit = torch.ones((self.batch_size, 1), device=self.device, dtype=self.dtype)
        u, w = self._induced_many(
            self._wx,
            self._wy,
            px[:, None],
            py[:, None],
            unit,
            self.rc[:, None],
        )
        return -(u * self._sa[:, None] + w * self._ca[:, None]) + self.dzc * (
            u * self._ca[:, None] - w * self._sa[:, None]
        )

    def _append_le_masked(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        gamma: torch.Tensor,
        shed: torch.Tensor,
    ) -> None:
        slot = torch.minimum(
            self._nl_total,
            torch.as_tensor(self.max_wake, device=self.device, dtype=torch.int64),
        )[:, None]
        full = self._nl_total >= self.max_wake
        append_x = self.lx.clone().scatter(1, slot, x[:, None])
        append_y = self.ly.clone().scatter(1, slot, y[:, None])
        append_g = self.lg.clone().scatter(1, slot, gamma[:, None])
        mask = shed[:, None]
        self.lx.copy_(torch.where(mask, append_x, self.lx))
        self.ly.copy_(torch.where(mask, append_y, self.ly))
        self.lg.copy_(torch.where(mask, append_g, self.lg))
        self._last_lev_x.copy_(torch.where(shed, x, self._last_lev_x))
        self._last_lev_y.copy_(torch.where(shed, y, self._last_lev_y))
        self._lev_shed_current.copy_(shed)
        self._lev_buffer_was_full.copy_(full)
        current_it = torch.full_like(self._lev_prev_it, self.it)
        self._lev_prev_it.copy_(
            torch.where(shed, current_it, self._lev_prev_it)
        )
        self._nl_total.add_(shed.to(torch.int64))

    def _trim_le_masked(self) -> None:
        trim = self._lev_shed_current & self._lev_buffer_was_full
        self.gam_lost.add_(torch.where(trim, self.lg[:, 0], 0.0))
        zero = torch.zeros(
            (self.batch_size, 1), device=self.device, dtype=self.dtype
        )
        mask = trim[:, None]
        self.lx.copy_(torch.where(mask, torch.cat((self.lx[:, 1:], zero), 1), self.lx))
        self.ly.copy_(torch.where(mask, torch.cat((self.ly[:, 1:], zero), 1), self.ly))
        self.lg.copy_(torch.where(mask, torch.cat((self.lg[:, 1:], zero), 1), self.lg))

    def _trim_te(self) -> None:
        if self.nt <= self.max_wake:
            return
        count = self.nt - self.max_wake
        self.gam_lost.add_(torch.sum(self.tg[:, :count], dim=1))
        self.tx[:, : self.max_wake].copy_(self.tx[:, count : self.nt].clone())
        self.ty[:, : self.max_wake].copy_(self.ty[:, count : self.nt].clone())
        self.tg[:, : self.max_wake].copy_(self.tg[:, count : self.nt].clone())
        self.nt = self.max_wake

    @torch.no_grad()
    def step(
        self,
        alpha: torch.Tensor,
        alpha_rate_per_convective_time: torch.Tensor,
        heave_rate_over_u: torch.Tensor | None = None,
        *,
        node_topology_from_cell_count: int | None = None,
    ) -> dict[str, torch.Tensor | bool]:
        """Advance every section at one common source clock tick.

        When ``node_topology_from_cell_count`` is supplied, the first ``S``
        lanes own physical ribbon-cell release and the following ``S+1`` lanes
        own only their shared endpoint kinematics.  Endpoint lanes then inherit
        the union of their adjacent cell events; they do not cast an independent
        material-release vote.
        """
        alpha = _cuda_vector(
            "alpha", alpha, batch_size=self.batch_size, device=self.device
        )
        alpha_rate = _cuda_vector(
            "alpha_rate_per_convective_time",
            alpha_rate_per_convective_time,
            batch_size=self.batch_size,
            device=self.device,
        )
        heave_rate = _cuda_vector(
            "heave_rate_over_u",
            0.0 if heave_rate_over_u is None else heave_rate_over_u,
            batch_size=self.batch_size,
            device=self.device,
        )
        self.it += 1
        ca, sa = torch.cos(alpha), torch.sin(alpha)
        self._ca.copy_(ca)
        self._sa.copy_(sa)
        self.sx.sub_(self.dt)
        self.sy.add_(heave_rate * self.dt)
        self._wx, self._wy = self._world(
            self.xs[None, :].expand(self.batch_size, -1)
        )

        tvx = self.tx[:, : self.nt]
        tvy = self.ty[:, : self.nt]
        tvg = self.tg[:, : self.nt]
        uu, ww = self._induced_many(
            self._wx,
            self._wy,
            tvx,
            tvy,
            tvg,
            self.rc[:, None].expand(-1, self.nt),
        )
        ul, wl = self._induced_many(
            self._wx,
            self._wy,
            self.lx,
            self.ly,
            self.lg,
            self.rc[:, None].expand(-1, self.max_wake + 1),
        )
        ui, wi = uu + ul, ww + wl
        relative_x = self.xs[None, :] - self.xp[:, None]
        w_base = (
            -sa[:, None]
            - ui * sa[:, None]
            + heave_rate[:, None] * ca[:, None]
            - wi * ca[:, None]
            - alpha_rate[:, None] * relative_x
            + self.dzc
            * (
                ui * ca[:, None]
                + ca[:, None]
                + heave_rate[:, None] * sa[:, None]
                - wi * sa[:, None]
            )
        )

        tex, tey = self._world(
            torch.ones((self.batch_size, 1), device=self.device, dtype=self.dtype)
        )
        tex, tey = tex[:, 0], tey[:, 0]
        if self.nt:
            ntx = tex + (self.tx[:, self.nt - 1] - tex) / 3.0
            nty = tey + (self.ty[:, self.nt - 1] - tey) / 3.0
        else:
            ntx = tex + 0.5 * self.dt
            nty = tey
        tcol = self._wcol(ntx, nty)

        old_circulation = (
            torch.sum(tvg, dim=1)
            + torch.sum(self.lg, dim=1)
            + self.gam_lost
        )
        gamma0 = self._gamb(w_base)
        gamma_t_column = self._gamb(tcol)
        gamma_tev = (gamma0 - old_circulation) / (1.0 - gamma_t_column)
        gamma_tev_provisional = gamma_tev.clone()
        w_pre = w_base + gamma_tev[:, None] * tcol
        af_pre = torch.stack(
            (self._a0(w_pre), self._an(w_pre, 1), self._an(w_pre, 2), self._an(w_pre, 3)),
            dim=1,
        )
        lesp_pre = af_pre[:, 0]
        raw_shed_lev = torch.abs(lesp_pre) > self.lesp_crit
        shed_lev = raw_shed_lev
        if node_topology_from_cell_count is not None:
            if (
                type(node_topology_from_cell_count) is not int
                or node_topology_from_cell_count < 1
                or self.batch_size != 2 * node_topology_from_cell_count + 1
            ):
                raise ValueError(
                    "node topology requires batch_size == 2*cell_count+1"
                )
            cell_count = node_topology_from_cell_count
            cell_active = raw_shed_lev[:cell_count]
            node_active = torch.empty(
                cell_count + 1, device=self.device, dtype=torch.bool
            )
            node_active[0] = cell_active[0]
            node_active[-1] = cell_active[-1]
            if cell_count > 1:
                node_active[1:-1] = cell_active[:-1] | cell_active[1:]
            shed_lev = torch.cat((cell_active, node_active))

        lex, ley = self._world(
            torch.zeros((self.batch_size, 1), device=self.device, dtype=self.dtype)
        )
        lex, ley = lex[:, 0], ley[:, 0]
        provisional_u, provisional_w = self._induced_many(
            lex[:, None],
            ley[:, None],
            ntx[:, None],
            nty[:, None],
            gamma_tev_provisional[:, None],
            self.rc[:, None],
        )
        lever = -self.xp
        le_u = 1.0 + lever * sa * alpha_rate + ui[:, 0] + provisional_u[:, 0]
        le_w = -heave_rate + lever * ca * alpha_rate + wi[:, 0] + provisional_w[:, 0]
        first_x = lex + 0.5 * le_u * self.dt
        first_y = ley + 0.5 * le_w * self.dt
        previous_slot = torch.clamp(
            torch.minimum(
                self._nl_total,
                torch.as_tensor(self.max_wake, device=self.device, dtype=torch.int64),
            )
            - 1,
            min=0,
        )[:, None]
        previous_x = torch.gather(self.lx, 1, previous_slot)[:, 0]
        previous_y = torch.gather(self.ly, 1, previous_slot)[:, 0]
        consecutive = self._lev_prev_it == self.it - 1
        nlx = torch.where(consecutive, lex + (previous_x - lex) / 3.0, first_x)
        nly = torch.where(consecutive, ley + (previous_y - ley) / 3.0, first_y)
        lcol = self._wcol(nlx, nly)
        gamma_l_column = self._gamb(lcol)
        a0_t = self._a0(tcol)
        a0_l = self._a0(lcol)
        b1 = gamma0 - old_circulation
        b2 = torch.sign(lesp_pre) * self.lesp_crit - self._a0(w_base)
        determinant = (1.0 - gamma_t_column) * a0_l - (
            1.0 - gamma_l_column
        ) * a0_t
        coupled_tev = (b1 * a0_l - (1.0 - gamma_l_column) * b2) / determinant
        coupled_lev = ((1.0 - gamma_t_column) * b2 - a0_t * b1) / determinant
        gamma_tev = torch.where(shed_lev, coupled_tev, gamma_tev)
        gamma_lev = torch.where(shed_lev, coupled_lev, 0.0)
        self._append_le_masked(nlx, nly, gamma_lev, shed_lev)
        first_tev_zeroed = self.it == 1
        gamma_tev_stored = torch.zeros_like(gamma_tev) if first_tev_zeroed else gamma_tev
        self.tx[:, self.nt].copy_(ntx)
        self.ty[:, self.nt].copy_(nty)
        self.tg[:, self.nt].copy_(gamma_tev_stored)
        self.nt += 1

        w_total = w_base + gamma_tev[:, None] * tcol + gamma_lev[:, None] * lcol
        af = torch.stack(
            (self._a0(w_total), self._an(w_total, 1), self._an(w_total, 2), self._an(w_total, 3)),
            dim=1,
        )
        self._AF_prev.copy_(af)
        an = (
            2.0
            * torch.trapezoid(
                w_total[:, :, None] * self._cosn[None, :, :], self.th, dim=1
            )
            / math.pi
        )
        gamma_theta = af[:, :1] * (1.0 + torch.cos(self.th))[None, :] + torch.sum(
            an[:, None, :] * self._sinn[None, :, :], dim=2
        ) * torch.sin(self.th)[None, :]
        bound_gamma = 0.5 * (gamma_theta[:, 1:] + gamma_theta[:, :-1]) * torch.diff(
            self.th
        )[None, :]
        bound_x = 0.5 * (self._wx[:, 1:] + self._wx[:, :-1])
        bound_y = 0.5 * (self._wy[:, 1:] + self._wy[:, :-1])

        tvx = self.tx[:, : self.nt]
        tvy = self.ty[:, : self.nt]
        tvg = self.tg[:, : self.nt]
        wake_x = torch.cat((tvx, self.lx), dim=1)
        wake_y = torch.cat((tvy, self.ly), dim=1)
        bound_u, bound_w = self._induced_many(
            wake_x,
            wake_y,
            bound_x,
            bound_y,
            -bound_gamma,
            self.rc[:, None].expand(-1, bound_gamma.shape[1]),
        )
        tev_u, tev_w = self._induced_many(
            wake_x,
            wake_y,
            tvx,
            tvy,
            tvg,
            self.rc[:, None].expand(-1, self.nt),
        )
        lev_u, lev_w = self._induced_many(
            wake_x,
            wake_y,
            self.lx,
            self.ly,
            self.lg,
            self.rc[:, None].expand(-1, self.max_wake + 1),
        )
        convect_u = bound_u + tev_u + lev_u
        convect_w = bound_w + tev_w + lev_w
        self.tx[:, : self.nt].add_(convect_u[:, : self.nt] * self.dt[:, None])
        self.ty[:, : self.nt].add_(convect_w[:, : self.nt] * self.dt[:, None])
        self.lx.add_(convect_u[:, self.nt :] * self.dt[:, None])
        self.ly.add_(convect_w[:, self.nt :] * self.dt[:, None])
        self.wake_convection_count += 1
        self._trim_te()
        self._trim_le_masked()

        constraint = torch.where(
            shed_lev,
            af[:, 0] - torch.sign(lesp_pre) * self.lesp_crit,
            0.0,
        )
        if not bool(
            torch.isfinite(
                torch.cat(
                    (
                        af.reshape(-1),
                        gamma_lev,
                        gamma_tev,
                        torch.stack((nlx, nly, ntx, nty), dim=1).reshape(-1),
                    )
                )
            ).all().item()
        ):
            raise FloatingPointError("batched LDVM source produced non-finite state")
        return {
            "A0": af[:, 0],
            "lesp_pre": lesp_pre,
            "lesp_constraint_residual": constraint,
            "shed_lev": shed_lev,
            "raw_shed_lev": raw_shed_lev,
            "gamma_lev_new": gamma_lev,
            "gamma_tev_new_solved": gamma_tev,
            "gamma_tev_new_persisted": gamma_tev_stored,
            "lev_birth_position": torch.stack((nlx, nly), dim=1),
            "tev_birth_position": torch.stack((ntx, nty), dim=1),
            "lev_edge_position": torch.stack((lex, ley), dim=1),
            "tev_edge_position": torch.stack((tex, tey), dim=1),
            "tev_strength_te_only_provisional": gamma_tev_provisional,
            "first_tev_zeroed": torch.full(
                (self.batch_size,),
                first_tev_zeroed,
                device=self.device,
                dtype=torch.bool,
            ),
            "n_lev": torch.minimum(
                self._nl_total,
                torch.as_tensor(self.max_wake, device=self.device, dtype=torch.int64),
            ),
            "n_tev": torch.full(
                (self.batch_size,), self.nt, device=self.device, dtype=torch.int64
            ),
            "source_parity": True,
        }


__all__ = ["CudaLDVMSourceBank"]
