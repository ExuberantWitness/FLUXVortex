"""Persistent CUDA particle storage for the FLUX-V5M production solver."""
from __future__ import annotations

import math
from collections.abc import Callable

import torch

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
) -> torch.Tensor:
    if type(value) is not torch.Tensor:
        raise TypeError(f"{name} must be an exact torch.Tensor")
    if value.device != device or value.device.type != "cuda":
        raise ValueError(f"{name} must be on {device}; implicit upload is forbidden")
    if value.dtype is not torch.float64:
        raise TypeError(f"{name} must use torch.float64")
    if shape_tail and tuple(value.shape[-len(shape_tail) :]) != shape_tail:
        raise ValueError(f"{name} must end in shape {shape_tail}")
    if not bool(torch.isfinite(value).all().item()):
        raise FloatingPointError(f"{name} contains non-finite values")
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
        "source_chunk_size",
        "target_chunk_size",
        "kernel_calls",
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
        self.source_chunk_size = int(source_chunk_size)
        self.target_chunk_size = int(target_chunk_size)
        self.kernel_calls = 0

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
        start, stop = self.n, self.n + count
        self.pos[start:stop].copy_(pos)
        self.gamma[start:stop].copy_(gamma)
        self.sigma[start:stop].copy_(sigma)
        self.circul[start:stop].copy_(circul)
        self.vol[start:stop].copy_(vol)
        self.ptype[start:stop].fill_(float(ptype))
        self.birth_step[start:stop].fill_(int(birth_step))
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
        ):
            field[:count].copy_(field[: self.n][keep])
        self.n = count

    def remove_type(self, type_code: float) -> None:
        if self.n:
            self.remove_mask(self.ptype[: self.n] == float(type_code))

    def velocity_at_cuda(self, targets: torch.Tensor) -> torch.Tensor:
        targets = _require_tensor(
            "targets", targets, device=self.device, shape_tail=(3,)
        )
        if targets.ndim != 2:
            raise ValueError("particle targets must have shape (m,3)")
        result = torch.zeros_like(targets)
        if self.n == 0 or targets.shape[0] == 0:
            return result
        source_pos = self.pos[: self.n]
        source_gamma = self.gamma[: self.n]
        source_sigma = self.sigma[: self.n]
        constant = -1.0 / (4.0 * math.pi)
        sqrt_two = math.sqrt(2.0)
        sqrt_two_over_pi = math.sqrt(2.0 / math.pi)
        for target_start in range(0, targets.shape[0], self.target_chunk_size):
            target_stop = min(target_start + self.target_chunk_size, targets.shape[0])
            target = targets[target_start:target_stop]
            partial = torch.zeros_like(target)
            for source_start in range(0, self.n, self.source_chunk_size):
                source_stop = min(source_start + self.source_chunk_size, self.n)
                delta = target[:, None, :] - source_pos[None, source_start:source_stop]
                radius_sq = torch.sum(delta * delta, dim=2)
                radius = torch.sqrt(radius_sq)
                sigma = source_sigma[source_start:source_stop].unsqueeze(0)
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
                    source_gamma[None, source_start:source_stop],
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

    def velocity_self_cuda(self) -> torch.Tensor:
        return self.velocity_at_cuda(self.pos[: self.n])

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
        count = pos.shape[0]
        if count:
            self.add_particles(
                pos,
                gamma,
                sigma,
                circul=circul,
                ptype=TYPE_FRESH_SHED,
                birth_step=birth_step,
            )
        return count

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
        }


__all__ = [
    "TYPE_BOUND",
    "TYPE_FREE",
    "TYPE_FRESH_SHED",
    "TYPE_PROBE",
    "CudaParticleField",
]
