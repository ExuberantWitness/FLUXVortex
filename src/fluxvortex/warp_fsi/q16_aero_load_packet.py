"""Audited CUDA load packet and work-conjugate Q16 transfer boundary.

The real aerodynamic solver can produce two scientifically different objects:

* resolved point forces with physical application points (four bound-vortex
  leg forces and one unsteady-pressure force per panel), and
* the time derivative of the global LEV/bound-sheet impulse, for which the
  current aerodynamic model does not define a surface application point.

The legacy Hirato contract exposes both objects.  In the DVM node-ribbon
contract the particles already enter the same-step Ptera solve, so ``KJ+dGamma``
is the unique production load owner and vortex impulse is diagnostic only.
Only a source-owned load with a declared application point can be passed through
the exact transpose Q16 surface map; silently smearing an unresolved force over
the wing would invent structural work.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any

import numpy as np
import torch
import warp as wp

from fluxvortex.q16_work_conjugate_transfer import Q16SurfaceTransferMap

from . import config
from .kernels_q16_transfer import Q16CudaSurfaceTransfer
from .q16_mandatory_aero_mode import require_q16_mandatory_aero_mode

_PACKET_SCHEMA = "fluxv-q16-cuda-aerodynamic-load-packet-v1"
_EPS = float(np.finfo(np.float64).eps)
_CLOSURE_FACTOR = 4096.0
_GEOMETRY_FACTOR = 512.0


def _require_cuda_float64_tensor(
    name: str,
    value: Any,
    *,
    shape: tuple[int, ...] | None = None,
    shape_tail: tuple[int, ...] | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    if type(value) is not torch.Tensor:
        raise TypeError(f"{name} must be an exact torch.Tensor")
    if value.device.type != "cuda":
        raise ValueError(f"{name} must reside on CUDA")
    if device is not None and value.device != device:
        raise ValueError(f"{name} crossed CUDA device boundary")
    if value.dtype is not torch.float64:
        raise TypeError(f"{name} must use torch.float64")
    if value.requires_grad:
        raise ValueError(f"{name} must not require gradients")
    if shape is not None and tuple(value.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if shape_tail is not None and tuple(value.shape[-len(shape_tail) :]) != shape_tail:
        raise ValueError(f"{name} must end with shape {shape_tail}")
    if not bool(torch.isfinite(value).all().item()):
        raise FloatingPointError(f"{name} contains non-finite values")
    return value


def _tensor_bytes(value: torch.Tensor) -> bytes:
    """Canonical evidence bytes; the scientific tensor remains on CUDA."""

    return value.detach().contiguous().cpu().numpy().tobytes(order="C")


def _packet_sha256(
    point_positions_w: torch.Tensor,
    point_forces_w: torch.Tensor,
    unresolved_impulse_force_w: torch.Tensor,
    source_total_force_w: torch.Tensor,
    source_total_moment_w: torch.Tensor,
) -> str:
    digest = hashlib.sha256()
    metadata = {
        "schema": _PACKET_SCHEMA,
        "device": str(point_positions_w.device),
        "point_count": int(point_positions_w.shape[0]),
        "dtype": "float64",
    }
    digest.update(
        json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("ascii")
    )
    for value in (
        point_positions_w,
        point_forces_w,
        unresolved_impulse_force_w,
        source_total_force_w,
        source_total_moment_w,
    ):
        digest.update(_tensor_bytes(value))
    return digest.hexdigest()


def _scaled_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    factor: float = _CLOSURE_FACTOR,
) -> bool:
    scale = max(
        1.0,
        float(torch.max(torch.abs(actual)).item()),
        float(torch.max(torch.abs(expected)).item()),
    )
    return (
        float(torch.max(torch.abs(actual - expected)).item()) <= factor * _EPS * scale
    )


@dataclass(frozen=True, slots=True)
class Q16CudaAerodynamicLoadPacket:
    """Sealed real-solver load packet kept as CUDA float64 tensors."""

    point_positions_w: torch.Tensor
    point_forces_w: torch.Tensor
    unresolved_impulse_force_w: torch.Tensor
    source_total_force_w: torch.Tensor
    source_total_moment_w: torch.Tensor
    packet_sha256: str

    @classmethod
    def from_tensors(
        cls,
        *,
        point_positions_w: torch.Tensor,
        point_forces_w: torch.Tensor,
        unresolved_impulse_force_w: torch.Tensor,
        source_total_force_w: torch.Tensor,
        source_total_moment_w: torch.Tensor,
    ) -> Q16CudaAerodynamicLoadPacket:
        if cls is not Q16CudaAerodynamicLoadPacket:
            raise TypeError("load packet subclasses are forbidden")
        positions = _require_cuda_float64_tensor(
            "point_positions_w", point_positions_w, shape_tail=(3,)
        )
        if positions.ndim != 2 or positions.shape[0] < 1:
            raise ValueError("point_positions_w must have shape (positive_points, 3)")
        device = positions.device
        forces = _require_cuda_float64_tensor(
            "point_forces_w",
            point_forces_w,
            shape=tuple(positions.shape),
            device=device,
        )
        impulse = _require_cuda_float64_tensor(
            "unresolved_impulse_force_w",
            unresolved_impulse_force_w,
            shape=(3,),
            device=device,
        )
        total_force = _require_cuda_float64_tensor(
            "source_total_force_w",
            source_total_force_w,
            shape=(3,),
            device=device,
        )
        total_moment = _require_cuda_float64_tensor(
            "source_total_moment_w",
            source_total_moment_w,
            shape=(3,),
            device=device,
        )
        frozen = tuple(
            value.detach().contiguous().clone()
            for value in (positions, forces, impulse, total_force, total_moment)
        )
        packet = Q16CudaAerodynamicLoadPacket(
            *frozen,
            _packet_sha256(*frozen),
        )
        return packet.validate()

    @classmethod
    def from_solver(cls, solver: Any) -> Q16CudaAerodynamicLoadPacket:
        mode_guard = require_q16_mandatory_aero_mode(solver)
        required = (
            "_q16_resolved_load_points_w",
            "_q16_resolved_load_forces_w",
            "_q16_unresolved_impulse_force_w",
            "_q16_total_force_w",
            "_q16_total_moment_w",
        )
        missing = [name for name in required if not hasattr(solver, name)]
        if missing:
            raise RuntimeError(
                "solver has not produced a Q16 load packet: " + ", ".join(missing)
            )
        packet = cls.from_tensors(
            point_positions_w=getattr(solver, required[0]),
            point_forces_w=getattr(solver, required[1]),
            unresolved_impulse_force_w=getattr(solver, required[2]),
            source_total_force_w=getattr(solver, required[3]),
            source_total_moment_w=getattr(solver, required[4]),
        )
        diagnostics = getattr(solver, "diag", None)
        if not diagnostics or diagnostics[-1].get("step") != getattr(
            solver, "_current_step", None
        ):
            raise RuntimeError("solver has no current aerodynamic load-owner record")
        expected_owner = (
            "ptera_kj_plus_dgamma"
            if mode_guard.separated_source == "dvm_node_ribbon"
            else "ptera_plus_legacy_vortex_impulse"
        )
        if diagnostics[-1].get("load_owner") != expected_owner:
            raise RuntimeError("aerodynamic load owner drifted")
        if mode_guard.separated_source == "dvm_node_ribbon":
            if int(torch.count_nonzero(packet.unresolved_impulse_force_w).item()) != 0:
                raise RuntimeError("DVM node-ribbon revived unresolved impulse load")
            counters = getattr(solver, "cuda_counters", None)
            if not isinstance(counters, dict) or counters.get("impulse") != 0:
                raise RuntimeError("DVM node-ribbon revived production impulse counter")
        mode_guard.verify(solver)
        return packet

    @property
    def point_count(self) -> int:
        return int(self.point_positions_w.shape[0])

    @property
    def resolved_force_w(self) -> torch.Tensor:
        return torch.sum(self.point_forces_w, dim=0)

    @property
    def resolved_moment_w(self) -> torch.Tensor:
        return torch.sum(
            torch.linalg.cross(self.point_positions_w, self.point_forces_w, dim=1),
            dim=0,
        )

    def validate(self) -> Q16CudaAerodynamicLoadPacket:
        if type(self) is not Q16CudaAerodynamicLoadPacket:
            raise TypeError("load packet must have exact frozen type")
        positions = _require_cuda_float64_tensor(
            "point_positions_w", self.point_positions_w, shape_tail=(3,)
        )
        if positions.ndim != 2 or positions.shape[0] < 1:
            raise ValueError("point_positions_w must have shape (positive_points, 3)")
        device = positions.device
        _require_cuda_float64_tensor(
            "point_forces_w",
            self.point_forces_w,
            shape=tuple(positions.shape),
            device=device,
        )
        for name, value in (
            ("unresolved_impulse_force_w", self.unresolved_impulse_force_w),
            ("source_total_force_w", self.source_total_force_w),
            ("source_total_moment_w", self.source_total_moment_w),
        ):
            _require_cuda_float64_tensor(name, value, shape=(3,), device=device)
        expected_sha = _packet_sha256(
            self.point_positions_w,
            self.point_forces_w,
            self.unresolved_impulse_force_w,
            self.source_total_force_w,
            self.source_total_moment_w,
        )
        if type(self.packet_sha256) is not str or self.packet_sha256 != expected_sha:
            raise RuntimeError("Q16 aerodynamic load packet content drift")
        if not _scaled_close(
            self.resolved_force_w + self.unresolved_impulse_force_w,
            self.source_total_force_w,
        ):
            raise ValueError("resolved plus impulse force does not close source total")
        if not _scaled_close(self.resolved_moment_w, self.source_total_moment_w):
            raise ValueError("resolved point moment does not close source total")
        return self


class Q16CudaResolvedLoadTransfer:
    """Map a sealed resolved packet through the exact CUDA Q16 transpose."""

    __slots__ = ("surface_transfer",)

    def __init__(self, transfer_map: Q16SurfaceTransferMap, *, device: str) -> None:
        self.surface_transfer = Q16CudaSurfaceTransfer(transfer_map, device=device)

    def map(
        self,
        packet: Q16CudaAerodynamicLoadPacket,
        structural_state: wp.array,
    ) -> wp.array:
        if type(packet) is not Q16CudaAerodynamicLoadPacket:
            raise TypeError("packet must be an exact Q16CudaAerodynamicLoadPacket")
        packet.validate()
        if packet.point_count != self.surface_transfer.point_count:
            raise ValueError("packet point count does not match Q16 surface map")
        expected_device = torch.device(self.surface_transfer.device)
        if packet.point_positions_w.device != expected_device:
            raise ValueError("packet and Q16 transfer use different CUDA devices")
        if int(torch.count_nonzero(packet.unresolved_impulse_force_w).item()) != 0:
            raise RuntimeError(
                "unresolved LEV impulse has no work-conjugate Q16 application point"
            )

        q16_points_wp = self.surface_transfer.interpolate(structural_state)
        if q16_points_wp.shape[0] != 1:
            raise ValueError("one aerodynamic packet requires one structural state")
        q16_points = wp.to_torch(q16_points_wp)[0]
        scale = max(
            1.0,
            float(torch.max(torch.abs(q16_points)).item()),
            float(torch.max(torch.abs(packet.point_positions_w)).item()),
        )
        mismatch = float(
            torch.max(torch.abs(q16_points - packet.point_positions_w)).item()
        )
        if mismatch > _GEOMETRY_FACTOR * _EPS * scale:
            raise RuntimeError("aero/Q16 geometry mismatch")

        force_wp = wp.from_torch(packet.point_forces_w.unsqueeze(0), dtype=config.VEC3)
        generalized = self.surface_transfer.transpose(force_wp)
        packet.validate()
        return generalized


__all__ = [
    "Q16CudaAerodynamicLoadPacket",
    "Q16CudaResolvedLoadTransfer",
]
