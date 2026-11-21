"""Persistent CUDA particle storage for the FLUX-V5M production solver."""
from __future__ import annotations

import math
from collections.abc import Callable

import torch
import warp as wp

BS_BLOCK_SIZE = 512
#: ``(f+g)/(1+3f)`` of the frozen rVPM note (research_rvpm_arch.md V4) at f=0, g=1/5.
RVPM_Z_WEIGHT = 0.2


@wp.kernel
def _pfield_bs_partials(
    targets: wp.array2d(dtype=wp.float64),
    source_pos: wp.array2d(dtype=wp.float64),
    source_gamma: wp.array2d(dtype=wp.float64),
    source_sigma: wp.array1d(dtype=wp.float64),
    block_size: int,
    partials: wp.array3d(dtype=wp.float64),
):
    t, b = wp.tid()
    n_s = int(source_sigma.shape[0])
    start = b * block_size
    stop = wp.min(start + block_size, n_s)
    c_half = wp.float64(0.5)
    c_inv_sqrt2 = wp.float64(0.7071067811865476)
    c_sqrt2_over_pi = wp.float64(0.7978845608028654)
    c_inv_4pi = wp.float64(-0.07957747154594767)
    zero = wp.float64(0.0)
    accx = zero
    accy = zero
    accz = zero
    tx = targets[t, 0]
    ty = targets[t, 1]
    tz = targets[t, 2]
    for s in range(start, stop):
        dx = tx - source_pos[s, 0]
        dy = ty - source_pos[s, 1]
        dz = tz - source_pos[s, 2]
        r2 = dx * dx + dy * dy + dz * dz
        if r2 > zero:
            r = wp.sqrt(r2)
            rho = r / source_sigma[s]
            # Far-field early-out: for rho > 12 the Gaussian regularization
            # erf(rho/sqrt2) - sqrt(2/pi)*rho*exp(-rho^2/2) equals 1 to the
            # last fp64 bit (the corrections are < 1e-31), so the erf/exp
            # pair is skipped for the overwhelmingly far-field pairs.
            if rho > 12.0:
                reg = wp.float64(1.0)
            else:
                reg = wp.erf(rho * c_inv_sqrt2) - c_sqrt2_over_pi * rho * wp.exp(
                    -c_half * rho * rho
                )
            w = c_inv_4pi * reg / (r2 * r)
            gx = source_gamma[s, 0]
            gy = source_gamma[s, 1]
            gz = source_gamma[s, 2]
            accx += w * (dy * gz - dz * gy)
            accy += w * (dz * gx - dx * gz)
            accz += w * (dx * gy - dy * gx)
    partials[t, b, 0] = accx
    partials[t, b, 1] = accy
    partials[t, b, 2] = accz


@wp.kernel
def _pfield_jacobian_partials(
    targets: wp.array2d(dtype=wp.float64),
    source_pos: wp.array2d(dtype=wp.float64),
    source_gamma: wp.array2d(dtype=wp.float64),
    source_sigma: wp.array1d(dtype=wp.float64),
    block_size: int,
    partials: wp.array3d(dtype=wp.float64),
):
    """Analytic velocity-gradient Jacobian of the Gaussian blob field.

    STORAGE CONVENTION (documented once, because mixing it up is the classic bug):
    component index ``3*i + m`` holds ``J[i, m] = d u_i / d x_m``.  Vortex
    stretching is therefore ``d(gamma_i)/dt = J[i, m] gamma[m]``, i.e. contract
    over the SECOND index (``einsum("nim,nm->ni", J, gamma)``).  This is
    equivalent to ``rvpm_transport.py``'s ``einsum("nji,nj->ni", ...)`` because
    that module stores the transpose; ``src/fluxvortex/particles.py:149`` stores
    this convention but contracts the wrong index, which a prior audit flagged.

    Same source loop and far-field early-out as ``_pfield_bs_partials``, but it
    accumulates the 9 components as well.  With ``u_i = w A_i``,
    ``A = d x gamma``, ``w = -G/(4 pi r^3)`` and
    ``G(rho) = erf(rho/sqrt2) - sqrt(2/pi) rho exp(-rho^2/2)``:

        J_im = wbr (d_m/r) A_i + w (e_m x gamma)_i,
        wbr  = (3 G/r - G'/sigma) / (4 pi r^3),   G' = sqrt(2/pi) rho^2 exp(-rho^2/2)

    Written with ``wbr`` rather than ``G'/G`` because ``G ~ rho^3`` at small rho,
    so a ``G'/G`` form would be a 0/0 cancellation.  The trace vanishes
    identically (``d . (d x gamma) = 0`` and the antisymmetric term is
    trace-free), which the contract test uses as a kernel-level check.
    """

    t, b = wp.tid()
    n_s = int(source_sigma.shape[0])
    start = b * block_size
    stop = wp.min(start + block_size, n_s)
    c_half = wp.float64(0.5)
    c_inv_sqrt2 = wp.float64(0.7071067811865476)
    c_sqrt2_over_pi = wp.float64(0.7978845608028654)
    c_inv_4pi = wp.float64(-0.07957747154594767)
    zero = wp.float64(0.0)
    j00 = zero
    j01 = zero
    j02 = zero
    j10 = zero
    j11 = zero
    j12 = zero
    j20 = zero
    j21 = zero
    j22 = zero
    tx = targets[t, 0]
    ty = targets[t, 1]
    tz = targets[t, 2]
    for s in range(start, stop):
        dx = tx - source_pos[s, 0]
        dy = ty - source_pos[s, 1]
        dz = tz - source_pos[s, 2]
        r2 = dx * dx + dy * dy + dz * dz
        if r2 > zero:
            r = wp.sqrt(r2)
            sigma = source_sigma[s]
            rho = r / sigma
            # warp scoping: a variable first assigned inside an if/else is
            # branch-local, so both must be declared before the branch (this
            # silently produced an all-zero Jacobian until it was caught).
            g_reg = zero
            g_prime = zero
            if rho > 12.0:
                g_reg = wp.float64(1.0)
                g_prime = zero
            else:
                e = wp.exp(-c_half * rho * rho)
                g_reg = wp.erf(rho * c_inv_sqrt2) - c_sqrt2_over_pi * rho * e
                g_prime = c_sqrt2_over_pi * rho * rho * e
            w = c_inv_4pi * g_reg / (r2 * r)
            # d/dx_m w = (d_m/r) * (1/(4 pi r^3)) (3G/r - G'/sigma); note the
            # PLUS sign (c_inv_4pi is -1/(4 pi), so negate it here).
            wbr = -c_inv_4pi * (
                wp.float64(3.0) * g_reg / r - g_prime / sigma
            ) / (r2 * r)
            gx = source_gamma[s, 0]
            gy = source_gamma[s, 1]
            gz = source_gamma[s, 2]
            ax = dy * gz - dz * gy
            ay = dz * gx - dx * gz
            az = dx * gy - dy * gx
            fac = wbr / r
            # wbr (d_m/r) A_i
            j00 += fac * dx * ax
            j01 += fac * dy * ax
            j02 += fac * dz * ax
            j10 += fac * dx * ay
            j11 += fac * dy * ay
            j12 += fac * dz * ay
            j20 += fac * dx * az
            j21 += fac * dy * az
            j22 += fac * dz * az
            # w (e_m x gamma)_i = w * eps_imk gamma_k  (index 3*i+m)
            j01 += w * gz
            j02 += w * (-gy)
            j10 += w * (-gz)
            j12 += w * gx
            j20 += w * gy
            j21 += w * (-gx)
    partials[t, b, 0] = j00
    partials[t, b, 1] = j01
    partials[t, b, 2] = j02
    partials[t, b, 3] = j10
    partials[t, b, 4] = j11
    partials[t, b, 5] = j12
    partials[t, b, 6] = j20
    partials[t, b, 7] = j21
    partials[t, b, 8] = j22


TYPE_FREE = 1.0
TYPE_FRESH_SHED = 1.1
TYPE_BOUND = 5.0
TYPE_PROBE = 9.0


def _require_tensor(
    name: str,
    value: torch.Tensor,
    *,
    device: torch.device,
    shape_tail: tuple[int, ...] = (),
    check_finite: bool = True,
) -> torch.Tensor:
    if type(value) is not torch.Tensor:
        raise TypeError(f"{name} must be an exact torch.Tensor")
    if value.device != device or value.device.type != "cuda":
        raise ValueError(f"{name} must be on {device}; implicit upload is forbidden")
    if value.dtype is not torch.float64:
        raise TypeError(f"{name} must use torch.float64")
    if shape_tail and tuple(value.shape[-len(shape_tail) :]) != shape_tail:
        raise ValueError(f"{name} must end in shape {shape_tail}")
    if check_finite and not bool(torch.isfinite(value).all().item()):
        raise FloatingPointError(f"{name} contains non-finite values")
    return value


def _require_source_strip(
    value: torch.Tensor,
    *,
    device: torch.device,
    count: int,
) -> torch.Tensor:
    if type(value) is not torch.Tensor:
        raise TypeError("source_strip must be an exact torch.Tensor")
    if value.device != device or value.device.type != "cuda":
        raise ValueError(
            f"source_strip must be on {device}; implicit upload is forbidden"
        )
    if value.dtype is not torch.int64:
        raise TypeError("source_strip must use torch.int64")
    if value.shape != (count,):
        raise ValueError("source_strip must have shape (particle_count,)")
    if bool(torch.any(value < -1).item()):
        raise ValueError("source_strip values must be -1 or non-negative")
    return value


class CudaParticleField:
    """Capacity-bounded, persistent float64 particle state on one CUDA GPU."""

    __slots__ = (
        "capacity",
        "device",
        "n",
        "pos",
        "gamma",
        "sigma",
        "circul",
        "vol",
        "ptype",
        "birth_step",
        "source_strip",
        "source_chunk_size",
        "target_chunk_size",
        "kernel_calls",
        "last_connected_ribbon_diagnostics",
        "last_open_filament_diagnostics",
    )

    def __init__(
        self,
        capacity: int = 200_000,
        *,
        device: str | torch.device = "cuda:0",
        source_chunk_size: int = 4096,
        target_chunk_size: int = 1024,
    ) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA particle field requires CUDA")
        resolved = torch.device(device)
        if resolved.type != "cuda":
            raise ValueError("CUDA particle field device must be CUDA")
        if type(capacity) is not int or capacity < 1:
            raise ValueError("particle capacity must be a positive exact int")
        if min(source_chunk_size, target_chunk_size) < 1:
            raise ValueError("particle chunk sizes must be positive")
        self.capacity = capacity
        self.device = resolved
        self.n = 0
        self.pos = torch.zeros((capacity, 3), device=resolved, dtype=torch.float64)
        self.gamma = torch.zeros((capacity, 3), device=resolved, dtype=torch.float64)
        self.sigma = torch.zeros(capacity, device=resolved, dtype=torch.float64)
        self.circul = torch.zeros(capacity, device=resolved, dtype=torch.float64)
        self.vol = torch.zeros(capacity, device=resolved, dtype=torch.float64)
        self.ptype = torch.full(
            (capacity,), TYPE_FREE, device=resolved, dtype=torch.float64
        )
        self.birth_step = torch.zeros(capacity, device=resolved, dtype=torch.int64)
        self.source_strip = torch.full(
            (capacity,), -1, device=resolved, dtype=torch.int64
        )
        self.source_chunk_size = int(source_chunk_size)
        self.target_chunk_size = int(target_chunk_size)
        self.kernel_calls = 0
        self.last_connected_ribbon_diagnostics: dict[str, float | int] | None = None
        self.last_open_filament_diagnostics: dict[str, float | int] | None = None

    @property
    def positions_cuda(self) -> torch.Tensor:
        return self.pos[: self.n]

    @property
    def gammas_cuda(self) -> torch.Tensor:
        return self.gamma[: self.n]

    @property
    def sigmas_cuda(self) -> torch.Tensor:
        return self.sigma[: self.n]

    @property
    def types_cuda(self) -> torch.Tensor:
        return self.ptype[: self.n]

    @property
    def source_strips_cuda(self) -> torch.Tensor:
        return self.source_strip[: self.n]

    def add_particles(
        self,
        pos: torch.Tensor,
        gamma: torch.Tensor,
        sigma: torch.Tensor,
        *,
        circul: torch.Tensor | None = None,
        vol: torch.Tensor | None = None,
        ptype: float = TYPE_FRESH_SHED,
        birth_step: int = 0,
        source_strip: torch.Tensor | None = None,
    ) -> None:
        pos = _require_tensor("pos", pos, device=self.device, shape_tail=(3,))
        gamma = _require_tensor("gamma", gamma, device=self.device, shape_tail=(3,))
        sigma = _require_tensor("sigma", sigma, device=self.device)
        if pos.ndim != 2 or gamma.shape != pos.shape or sigma.shape != pos.shape[:1]:
            raise ValueError("particle pos/gamma/sigma shapes must align")
        if not bool(torch.all(sigma > 0.0).item()):
            raise ValueError("particle sigma must be positive")
        count = pos.shape[0]
        if self.n + count > self.capacity:
            raise OverflowError(
                f"particle capacity exceeded: {self.n}+{count}>{self.capacity}"
            )
        if circul is None:
            circul = torch.linalg.vector_norm(gamma, dim=1) + 1.0e-30
        else:
            circul = _require_tensor("circul", circul, device=self.device)
        if vol is None:
            vol = torch.zeros(count, device=self.device, dtype=torch.float64)
        else:
            vol = _require_tensor("vol", vol, device=self.device)
        if circul.shape != (count,) or vol.shape != (count,):
            raise ValueError("particle circul/vol shapes must align")
        if source_strip is None:
            source_strip = torch.full(
                (count,), -1, device=self.device, dtype=torch.int64
            )
        else:
            source_strip = _require_source_strip(
                source_strip,
                device=self.device,
                count=count,
            )
        start, stop = self.n, self.n + count
        self.pos[start:stop].copy_(pos)
        self.gamma[start:stop].copy_(gamma)
        self.sigma[start:stop].copy_(sigma)
        self.circul[start:stop].copy_(circul)
        self.vol[start:stop].copy_(vol)
        self.ptype[start:stop].fill_(float(ptype))
        self.birth_step[start:stop].fill_(int(birth_step))
        self.source_strip[start:stop].copy_(source_strip)
        self.n = stop

    def promote_fresh(self) -> None:
        if self.n:
            current = self.ptype[: self.n]
            current.copy_(torch.where(current == TYPE_FRESH_SHED, TYPE_FREE, current))

    def remove_mask(self, remove: torch.Tensor) -> None:
        if type(remove) is not torch.Tensor or remove.device != self.device:
            raise ValueError("particle remove mask must be a CUDA tensor")
        if remove.dtype is not torch.bool or remove.shape != (self.n,):
            raise ValueError("particle remove mask must be bool with shape (n,)")
        keep = ~remove
        count = int(torch.count_nonzero(keep).item())
        for field in (
            self.pos,
            self.gamma,
            self.sigma,
            self.circul,
            self.vol,
            self.ptype,
            self.birth_step,
            self.source_strip,
        ):
            field[:count].copy_(field[: self.n][keep])
        self.n = count

    def remove_type(self, type_code: float) -> None:
        if self.n:
            self.remove_mask(self.ptype[: self.n] == float(type_code))

    def velocity_at_cuda_chunked(
        self,
        targets: torch.Tensor,
        *,
        source_start: int = 0,
        source_stop: int | None = None,
    ) -> torch.Tensor:
        targets = _require_tensor(
            "targets",
            targets,
            device=self.device,
            shape_tail=(3,),
            check_finite=False,
        )
        if targets.ndim != 2:
            raise ValueError("particle targets must have shape (m,3)")
        result = torch.zeros_like(targets)
        stop = self.n if source_stop is None else source_stop
        if (
            type(source_start) is not int
            or type(stop) is not int
            or not 0 <= source_start <= stop <= self.n
        ):
            raise ValueError("particle source range is outside the live field")
        wp.init()
        if source_start == stop or targets.shape[0] == 0:
            return result
        source_pos = self.pos[source_start:stop]
        source_gamma = self.gamma[source_start:stop]
        source_sigma = self.sigma[source_start:stop]
        source_count = stop - source_start
        constant = -1.0 / (4.0 * math.pi)
        sqrt_two = math.sqrt(2.0)
        sqrt_two_over_pi = math.sqrt(2.0 / math.pi)
        for target_start in range(0, targets.shape[0], self.target_chunk_size):
            target_stop = min(target_start + self.target_chunk_size, targets.shape[0])
            target = targets[target_start:target_stop]
            partial = torch.zeros_like(target)
            for chunk_start in range(0, source_count, self.source_chunk_size):
                chunk_stop = min(chunk_start + self.source_chunk_size, source_count)
                delta = target[:, None, :] - source_pos[None, chunk_start:chunk_stop]
                radius_sq = torch.sum(delta * delta, dim=2)
                radius = torch.sqrt(radius_sq)
                sigma = source_sigma[chunk_start:chunk_stop].unsqueeze(0)
                rho = radius / sigma
                regularization = torch.erf(rho / sqrt_two) - (
                    sqrt_two_over_pi * rho * torch.exp(-0.5 * rho * rho)
                )
                safe_radius_sq = torch.where(
                    radius_sq > 0.0, radius_sq, torch.ones_like(radius_sq)
                )
                safe_radius = torch.where(radius > 0.0, radius, torch.ones_like(radius))
                weight = constant * regularization / (safe_radius_sq * safe_radius)
                cross = torch.linalg.cross(
                    delta,
                    source_gamma[None, chunk_start:chunk_stop],
                    dim=2,
                )
                partial = partial + torch.sum(
                    torch.where(
                        (radius_sq > 0.0).unsqueeze(2),
                        weight.unsqueeze(2) * cross,
                        0.0,
                    ),
                    dim=1,
                )
                self.kernel_calls += 1
            result[target_start:target_stop] = partial
        return result

    def velocity_at_cuda(
        self,
        targets: torch.Tensor,
        *,
        source_start: int = 0,
        source_stop: int | None = None,
    ) -> torch.Tensor:
        """Gaussian-regularized Biot-Savart via one fused warp kernel.

        Each (target, source-block) thread accumulates its block's sources
        into partials; one torch reduction finishes the sum.  Mathematically
        identical to the chunked-torch path (fp64 summation-order differences
        only); ~10-30x faster by eliminating per-tile op materialization.
        """

        targets = _require_tensor(
            "targets",
            targets,
            device=self.device,
            shape_tail=(3,),
            check_finite=False,
        )
        if targets.ndim != 2:
            raise ValueError("particle targets must have shape (m,3)")
        stop = self.n if source_stop is None else source_stop
        if (
            type(source_start) is not int
            or type(stop) is not int
            or not 0 <= source_start <= stop <= self.n
        ):
            raise ValueError("particle source range is outside the live field")
        result = torch.zeros_like(targets)
        wp.init()
        if source_start == stop or targets.shape[0] == 0:
            return result
        n_source = stop - source_start
        n_blocks = (n_source + BS_BLOCK_SIZE - 1) // BS_BLOCK_SIZE
        source_pos = self.pos[source_start:stop]
        source_gamma = self.gamma[source_start:stop]
        source_sigma = self.sigma[source_start:stop]
        partials = torch.zeros(
            (targets.shape[0], n_blocks, 3),
            device=self.device,
            dtype=torch.float64,
        )
        wp.launch(
            _pfield_bs_partials,
            dim=(targets.shape[0], n_blocks),
            inputs=[
                wp.from_torch(targets.contiguous(), dtype=wp.float64),
                wp.from_torch(source_pos.contiguous(), dtype=wp.float64),
                wp.from_torch(source_gamma.contiguous(), dtype=wp.float64),
                wp.from_torch(source_sigma.contiguous(), dtype=wp.float64),
                BS_BLOCK_SIZE,
            ],
            outputs=[wp.from_torch(partials, dtype=wp.float64)],
            device="cuda:0",
        )
        self.kernel_calls += 1
        torch.sum(partials, dim=1, out=result)
        return result

    def velocity_self_cuda(self) -> torch.Tensor:
        return self.velocity_at_cuda(self.pos[: self.n])

    def velocity_and_jacobian_self_cuda(
        self, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Velocity and analytic Jacobian ``J[i,m] = du_i/dx_m`` at ``targets``.

        ``targets`` defaults to the live source positions (self-evaluation, which
        is what vortex stretching needs).  Returns ``(m,3)`` and ``(m,3,3)`` as a
        single fused warp launch, mirroring ``velocity_at_cuda``'s block/partials
        structure so the two kernels cannot drift apart.
        """

        if targets is None:
            targets = self.pos[: self.n]
        targets = _require_tensor(
            "targets",
            targets,
            device=self.device,
            shape_tail=(3,),
            check_finite=False,
        )
        if targets.ndim != 2:
            raise ValueError("particle jacobian targets must have shape (m,3)")
        count = targets.shape[0]
        velocity = torch.zeros_like(targets)
        jacobian = torch.zeros(
            (count, 3, 3), device=self.device, dtype=torch.float64
        )
        wp.init()
        if self.n == 0 or count == 0:
            return velocity, jacobian
        n_blocks = (self.n + BS_BLOCK_SIZE - 1) // BS_BLOCK_SIZE
        source_pos = self.pos[: self.n]
        source_gamma = self.gamma[: self.n]
        source_sigma = self.sigma[: self.n]
        partials = torch.zeros(
            (count, n_blocks, 9), device=self.device, dtype=torch.float64
        )
        wp.launch(
            _pfield_jacobian_partials,
            dim=(count, n_blocks),
            inputs=[
                wp.from_torch(targets.contiguous(), dtype=wp.float64),
                wp.from_torch(source_pos.contiguous(), dtype=wp.float64),
                wp.from_torch(source_gamma.contiguous(), dtype=wp.float64),
                wp.from_torch(source_sigma.contiguous(), dtype=wp.float64),
                BS_BLOCK_SIZE,
            ],
            outputs=[wp.from_torch(partials, dtype=wp.float64)],
            device="cuda:0",
        )
        self.kernel_calls += 1
        reduced = torch.sum(partials, dim=1)
        jacobian.copy_(reduced.reshape(count, 3, 3))
        # The velocity is the same kernel's first-order part; reuse the fused
        # velocity launch so both quantities come from the identical source loop.
        velocity.copy_(self.velocity_at_cuda(targets))
        return velocity, jacobian

    def advance_wrk3(
        self,
        delta_time: float,
        external_velocity: Callable[[torch.Tensor], torch.Tensor],
    ) -> None:
        """Advance positions with the legacy three-stage recurrence on CUDA."""

        if self.n == 0:
            return
        if not math.isfinite(delta_time) or delta_time <= 0.0:
            raise ValueError("particle delta_time must be finite and positive")
        original = self.pos[: self.n].clone()

        def velocity(at: torch.Tensor) -> torch.Tensor:
            self.pos[: self.n].copy_(at)
            external = _require_tensor(
                "external particle velocity",
                external_velocity(at),
                device=self.device,
                shape_tail=(3,),
                check_finite=False,
            )
            if external.shape != at.shape:
                raise ValueError("external particle velocity shape mismatch")
            return external + self.velocity_self_cuda()

        u1 = velocity(original)
        u2 = velocity(original + 0.5 * delta_time * u1)
        u3 = velocity(original + delta_time * (-u1 + 2.0 * u2))
        final = original + (delta_time / 6.0) * (u1 + 4.0 * u2 + u3)
        if not bool(torch.isfinite(final).all().item()):
            self.pos[: self.n].copy_(original)
            raise FloatingPointError("particle advance produced non-finite positions")
        self.pos[: self.n].copy_(final)
        self.promote_fresh()

    def add_ring_particles(
        self,
        rings: torch.Tensor,
        strengths: torch.Tensor,
        *,
        sigma_factor: float,
        birth_step: int,
        reverse: bool = False,
    ) -> int:
        rings = _require_tensor("rings", rings, device=self.device, shape_tail=(4, 3))
        strengths = _require_tensor("strengths", strengths, device=self.device)
        if rings.ndim != 3 or strengths.shape != rings.shape[:1]:
            raise ValueError("ring and strength shapes must align")
        if not math.isfinite(sigma_factor) or sigma_factor <= 0.0:
            raise ValueError("sigma_factor must be finite and positive")
        direction = -1 if reverse else 1
        origin = rings
        destination = torch.roll(rings, shifts=-direction, dims=1)
        vector = destination - origin
        length = torch.linalg.vector_norm(vector, dim=2)
        strength = strengths.unsqueeze(1).expand_as(length)
        active = (torch.abs(strength) >= 1.0e-14) & (length >= 1.0e-12)
        pos = (0.5 * (origin + destination))[active]
        gamma = (vector * strength.unsqueeze(2))[active]
        sigma = (length / sigma_factor)[active]
        circul = strength[active]
        source_strip = (
            torch.arange(rings.shape[0], device=self.device, dtype=torch.int64)
            .unsqueeze(1)
            .expand_as(length)
        )[active]
        count = pos.shape[0]
        if count:
            self.add_particles(
                pos,
                gamma,
                sigma,
                circul=circul,
                ptype=TYPE_FRESH_SHED,
                birth_step=birth_step,
                source_strip=source_strip,
            )
        return count

    def add_connected_ribbon_particles(
        self,
        anchor_nodes: torch.Tensor,
        frontier_nodes: torch.Tensor,
        strengths: torch.Tensor,
        *,
        smoothing_radius: float,
        target_spacing: float,
        birth_step: int,
        connector_source_strips: torch.Tensor | None = None,
    ) -> int:
        """Deposit one node-owned spanwise ribbon without duplicate seam edges.

        Cell ``i`` has traversal ``anchor_i -> anchor_i+1 -> frontier_i+1 ->
        frontier_i``.  The two incidences at every interior chordwise edge are
        reduced to one signed circulation before deposition.  All geometry,
        incidence reduction, subdivision, and particle vectors are evaluated
        as CUDA float64 tensors; the host observes only allocation counts and
        fail-closed diagnostics.
        """

        anchor = _require_tensor(
            "anchor_nodes", anchor_nodes, device=self.device, shape_tail=(3,)
        )
        frontier = _require_tensor(
            "frontier_nodes", frontier_nodes, device=self.device, shape_tail=(3,)
        )
        circulation = _require_tensor(
            "strengths", strengths, device=self.device
        )
        if (
            anchor.ndim != 2
            or frontier.shape != anchor.shape
            or circulation.ndim != 1
            or anchor.shape[0] != circulation.shape[0] + 1
            or circulation.numel() < 1
        ):
            raise ValueError(
                "connected ribbon requires (s+1,3) nodes and (s,) strengths"
            )
        if not math.isfinite(smoothing_radius) or smoothing_radius <= 0.0:
            raise ValueError("smoothing_radius must be finite and positive")
        if not math.isfinite(target_spacing) or target_spacing <= 0.0:
            raise ValueError("target_spacing must be finite and positive")
        minimum_overlap = 2.125
        if smoothing_radius / target_spacing < minimum_overlap:
            raise ValueError(
                "target_spacing violates the frozen minimum overlap 2.125"
            )
        if type(birth_step) is not int or birth_step < 0:
            raise ValueError("birth_step must be a nonnegative exact int")

        span_cells = circulation.shape[0]
        if connector_source_strips is None:
            connector_owners = torch.full(
                (span_cells + 1,), -1, device=self.device, dtype=torch.int64
            )
        else:
            connector_owners = _require_source_strip(
                connector_source_strips,
                device=self.device,
                count=span_cells + 1,
            )
            if bool(
                torch.any(
                    (connector_owners < 0) | (connector_owners >= span_cells)
                ).item()
            ):
                raise ValueError(
                    "connector_source_strips must name a live span cell"
                )
        connector_circulation = torch.cat(
            (
                -circulation[:1],
                circulation[:-1] - circulation[1:],
                circulation[-1:],
            )
        )
        starts = torch.cat((anchor[:-1], frontier[:-1], anchor), dim=0)
        ends = torch.cat((anchor[1:], frontier[1:], frontier), dim=0)
        edge_circulation = torch.cat(
            (circulation, -circulation, connector_circulation)
        )
        source_edge_strip = torch.cat(
            (
                torch.arange(span_cells, device=self.device, dtype=torch.int64),
                torch.arange(span_cells, device=self.device, dtype=torch.int64),
                connector_owners,
            )
        )
        edge_vector = ends - starts
        edge_length = torch.linalg.vector_norm(edge_vector, dim=1)
        active = (torch.abs(edge_circulation) >= 1.0e-14) & (
            edge_length >= 1.0e-12
        )
        starts = starts[active]
        edge_vector = edge_vector[active]
        edge_length = edge_length[active]
        edge_circulation = edge_circulation[active]
        source_edge_strip = source_edge_strip[active]
        retained_edges = int(torch.count_nonzero(active).item())
        if retained_edges == 0:
            self.last_connected_ribbon_diagnostics = {
                "span_cell_count": int(span_cells),
                "retained_edge_count": 0,
                "particle_count": 0,
                "seam_count": 0,
                "global_vector_closure": 0.0,
            }
            return 0

        spacing = torch.as_tensor(
            target_spacing, device=self.device, dtype=torch.float64
        )
        counts = torch.clamp(torch.ceil(edge_length / spacing), min=1.0).to(
            torch.int64
        )
        particle_count = int(torch.sum(counts).item())
        if self.n + particle_count > self.capacity:
            raise OverflowError(
                "particle capacity exceeded by connected ribbon: "
                f"{self.n}+{particle_count}>{self.capacity}"
            )
        edge_index = torch.repeat_interleave(
            torch.arange(retained_edges, device=self.device, dtype=torch.int64),
            counts,
        )
        offsets = torch.cumsum(counts, dim=0) - counts
        local_index = torch.arange(
            particle_count, device=self.device, dtype=torch.int64
        ) - torch.repeat_interleave(offsets, counts)
        particle_counts = counts[edge_index]
        fraction = (
            local_index.to(torch.float64) + 0.5
        ) / particle_counts.to(torch.float64)
        segment = edge_vector[edge_index] / particle_counts[:, None]
        positions = starts[edge_index] + fraction[:, None] * edge_vector[edge_index]
        particle_circulation = edge_circulation[edge_index]
        gamma = particle_circulation[:, None] * segment
        sigma = torch.full(
            (particle_count,),
            smoothing_radius,
            device=self.device,
            dtype=torch.float64,
        )
        source_strip = source_edge_strip[edge_index]

        closure_vector = torch.sum(gamma, dim=0)
        closure = torch.linalg.vector_norm(closure_vector)
        closure_scale = torch.sum(torch.linalg.vector_norm(gamma, dim=1))
        closure_limit = 2048.0 * torch.finfo(torch.float64).eps * torch.clamp(
            closure_scale, min=1.0
        )
        if bool((closure > closure_limit).item()):
            raise FloatingPointError(
                "connected ribbon violates global vector-moment closure"
            )
        self.add_particles(
            positions,
            gamma,
            sigma,
            circul=particle_circulation,
            ptype=TYPE_FRESH_SHED,
            birth_step=birth_step,
            source_strip=source_strip,
        )
        self.last_connected_ribbon_diagnostics = {
            "span_cell_count": int(span_cells),
            "retained_edge_count": retained_edges,
            "particle_count": particle_count,
            "seam_count": 0,
            "global_vector_closure": float(closure.item()),
            "connectors_have_strip_owner": connector_source_strips is not None,
        }
        return particle_count

    def add_open_filament_particles(
        self,
        nodes: torch.Tensor,
        strengths: torch.Tensor,
        *,
        smoothing_radius: float,
        target_spacing: float,
        birth_step: int,
        source_strips: torch.Tensor | None = None,
    ) -> int:
        """Deposit one spanwise filament WITHOUT closing it (nonzero net vorticity).

        ``add_connected_ribbon_particles`` closes every deposit into a loop so the
        net vorticity ``sum(gamma_p)`` is machine zero; measured on the Roj A15
        case that deposit induces only 3.9% of a monopole's normal velocity at the
        first collocation row (S2.0), because the closure sheet sits ``ds`` away
        and cancels the leading-edge sheet to 96%.  A real leading-edge vortex is a
        *single-signed* sheet, so this entry point deliberately deposits only the
        spanwise edges and leaves ``sum(gamma_p)`` nonzero.

        The edge subdivision mirrors ``add_connected_ribbon_particles`` exactly
        (same ``ceil(length/spacing)`` counts, mid-edge particle placement, and
        ``gamma = circulation * segment``) so a caller can swap the two and change
        nothing but the closure.

        Returns the number of particles deposited.
        """

        nodes = _require_tensor(
            "nodes", nodes, device=self.device, shape_tail=(3,)
        )
        circulation = _require_tensor("strengths", strengths, device=self.device)
        if (
            nodes.ndim != 2
            or circulation.ndim != 1
            or nodes.shape[0] != circulation.shape[0] + 1
            or circulation.numel() < 1
        ):
            raise ValueError(
                "open filament requires (s+1,3) nodes and (s,) strengths"
            )
        if not math.isfinite(smoothing_radius) or smoothing_radius <= 0.0:
            raise ValueError("smoothing_radius must be finite and positive")
        if not math.isfinite(target_spacing) or target_spacing <= 0.0:
            raise ValueError("target_spacing must be finite and positive")
        if smoothing_radius / target_spacing < 2.125:
            raise ValueError(
                "target_spacing violates the frozen minimum overlap 2.125"
            )
        if type(birth_step) is not int or birth_step < 0:
            raise ValueError("birth_step must be a nonnegative exact int")

        span_cells = circulation.shape[0]
        if source_strips is None:
            source_strips = torch.arange(
                span_cells, device=self.device, dtype=torch.int64
            )
        else:
            source_strips = _require_source_strip(
                source_strips, device=self.device, count=span_cells
            )

        starts = nodes[:-1]
        edge_vector = nodes[1:] - starts
        edge_length = torch.linalg.vector_norm(edge_vector, dim=1)
        active = (torch.abs(circulation) >= 1.0e-14) & (
            edge_length >= 1.0e-12
        )
        starts = starts[active]
        edge_vector = edge_vector[active]
        edge_length = edge_length[active]
        circulation = circulation[active]
        source_strips = source_strips[active]
        if starts.shape[0] == 0:
            self.last_open_filament_diagnostics = {
                "span_cell_count": int(span_cells),
                "retained_edge_count": 0,
                "particle_count": 0,
                "net_vector_moment": 0.0,
            }
            return 0

        spacing = torch.as_tensor(
            target_spacing, device=self.device, dtype=torch.float64
        )
        counts = torch.clamp(torch.ceil(edge_length / spacing), min=1.0).to(
            torch.int64
        )
        particle_count = int(torch.sum(counts).item())
        if self.n + particle_count > self.capacity:
            raise OverflowError(
                "particle capacity exceeded by open filament: "
                f"{self.n}+{particle_count}>{self.capacity}"
            )
        edge_index = torch.repeat_interleave(
            torch.arange(starts.shape[0], device=self.device, dtype=torch.int64),
            counts,
        )
        offsets = torch.cumsum(counts, dim=0) - counts
        local_index = torch.arange(
            particle_count, device=self.device, dtype=torch.int64
        ) - torch.repeat_interleave(offsets, counts)
        particle_counts = counts[edge_index]
        fraction = (
            local_index.to(torch.float64) + 0.5
        ) / particle_counts.to(torch.float64)
        segment = edge_vector[edge_index] / particle_counts[:, None]
        positions = starts[edge_index] + fraction[:, None] * edge_vector[edge_index]
        particle_circulation = circulation[edge_index]
        gamma = particle_circulation[:, None] * segment
        sigma = torch.full(
            (particle_count,),
            smoothing_radius,
            device=self.device,
            dtype=torch.float64,
        )
        self.add_particles(
            positions,
            gamma,
            sigma,
            circul=particle_circulation,
            ptype=TYPE_FRESH_SHED,
            birth_step=birth_step,
            source_strip=source_strips[edge_index],
        )
        self.last_open_filament_diagnostics = {
            "span_cell_count": int(span_cells),
            "retained_edge_count": int(starts.shape[0]),
            "particle_count": particle_count,
            "net_vector_moment": float(
                torch.linalg.vector_norm(torch.sum(gamma, dim=0)).item()
            ),
        }
        return particle_count

    def apply_vortex_stretching(
        self,
        delta_time: float,
        *,
        core_spreading: bool = False,
        rvpm: bool = False,
    ) -> dict[str, float]:
        """One Euler step of 3-D vortex stretching on the live particles.

        ``S = J . gamma`` with ``J[i,m] = du_i/dx_m`` (see
        ``_pfield_jacobian_partials`` for the storage convention).

        Two forms, because the repo's frozen design note specifies one and an
        earlier version of this code implemented the other:

        ``rvpm=True`` -- the reformulated VPM form of
        ``platform/docs/diag/research_rvpm_arch.md`` (V4, quoting rvpm-02):
        ``Z = [(f+g)/(1+3f) (S . gamma)] / |gamma|^2``, ``dgamma/dt = S - 3 Z gamma``,
        ``dsigma/dt = -sigma Z``.  With ``f=0, g=1/5`` the weight is ``1/5``.  The
        ``-3 Z gamma`` term is what makes a stretched element BOUNDED: it removes
        the along-gamma component so the growth is redistributed into the core
        instead of accumulating without limit.

        ``rvpm=False`` -- the bare Helmholtz term ``dgamma/dt = S``.  Kept so the
        raw form stays reproducible for A/B attribution; it is the configuration
        that grew ``|gamma|`` without bound.

        ``C epsilon`` from the note is set to 0: the note does not fix its value
        and the viscous scheme is recorded there as inviscid, so introducing a
        non-zero dissipation coefficient would be a new parameter, which the
        project forbids.

        First order in time for the strength (positions keep their RK3 update).
        Leaves the field unchanged when empty.
        """

        summary = {
            "particle_count": int(self.n),
            "stretching_form": "rvpm" if rvpm else "raw",
            "stretching_growth_max": 1.0,
            "stretching_shrink_max": 1.0,
            "stretching_redistribution_max": 0.0,
            "core_radius_ratio_min": 1.0,
            "core_radius_ratio_max": 1.0,
        }
        if self.n == 0:
            return summary
        if not math.isfinite(delta_time) or delta_time <= 0.0:
            raise ValueError("stretching delta_time must be finite and positive")

        _, jacobian = self.velocity_and_jacobian_self_cuda(self.pos[: self.n])
        gamma = self.gamma[: self.n]
        stretch_rate = torch.einsum("nim,nm->ni", jacobian, gamma)
        dot = torch.sum(stretch_rate * gamma, dim=1)
        norm_sq = torch.clamp(torch.sum(gamma * gamma, dim=1), min=1.0e-300)
        # (f+g)/(1+3f) with f=0, g=1/5 == 1/5 (frozen V4 note).
        weight = RVPM_Z_WEIGHT if rvpm else 1.0
        z = weight * dot / norm_sq

        if rvpm:
            redistribution = 3.0 * z[:, None] * gamma
            updated = gamma + delta_time * (stretch_rate - redistribution)
            summary["stretching_redistribution_max"] = float(
                torch.linalg.vector_norm(redistribution, dim=1).max().item()
            )
        else:
            updated = gamma + delta_time * stretch_rate
        if not bool(torch.isfinite(updated).all().item()):
            raise FloatingPointError("vortex stretching produced non-finite strength")
        magnitude = torch.linalg.vector_norm(gamma, dim=1)
        ratio = torch.linalg.vector_norm(updated, dim=1) / torch.clamp(
            magnitude, min=1.0e-300
        )

        if core_spreading:
            sigma = self.sigma[: self.n]
            new_sigma = torch.clamp(
                sigma * (1.0 - delta_time * z),
                min=sigma * 0.5,
                max=sigma * 2.0,
            )
            summary["core_radius_ratio_min"] = float((new_sigma / sigma).min().item())
            summary["core_radius_ratio_max"] = float((new_sigma / sigma).max().item())
            self.sigma[: self.n].copy_(new_sigma)

        self.gamma[: self.n].copy_(updated)
        self.circul[: self.n].copy_(torch.linalg.vector_norm(updated, dim=1))
        summary["stretching_growth_max"] = float(ratio.max().item())
        summary["stretching_shrink_max"] = float(1.0 / ratio.min().item())
        return summary

    def snapshot_numpy(self) -> dict[str, object]:
        """Explicit serialization boundary; never used by time integration."""

        return {
            "positions": self.pos[: self.n].detach().cpu().numpy().copy(),
            "gamma": self.gamma[: self.n].detach().cpu().numpy().copy(),
            "sigma": self.sigma[: self.n].detach().cpu().numpy().copy(),
            "circul": self.circul[: self.n].detach().cpu().numpy().copy(),
            "vol": self.vol[: self.n].detach().cpu().numpy().copy(),
            "ptype": self.ptype[: self.n].detach().cpu().numpy().copy(),
            "birth_step": self.birth_step[: self.n].detach().cpu().numpy().copy(),
            "source_strip": self.source_strip[: self.n].detach().cpu().numpy().copy(),
        }


__all__ = [
    "TYPE_BOUND",
    "TYPE_FREE",
    "TYPE_FRESH_SHED",
    "TYPE_PROBE",
    "CudaParticleField",
]
