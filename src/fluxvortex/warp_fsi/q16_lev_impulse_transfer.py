"""Source-line work contract for active-LEV impulse on a CUDA Q16 surface.

The aerodynamic solver owns one impulse force per causal span strip and the
two real leading-edge endpoints that generated that strip's LEV ring. This
module declares the structural work model explicitly:

``delta W = F_strip dot 0.5 * (delta x_left + delta x_right)``.

The implementation therefore puts half the force on each exact Q16 endpoint
row and applies the same surface interpolation transpose. It does not infer a
pressure distribution or an unreported aerodynamic impulse couple.
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

_SCHEMA = "fluxv-q16-cuda-lev-impulse-strip-load-v1"
_EPS = float(np.finfo(np.float64).eps)
_CLOSURE_FACTOR = 8192.0
_GEOMETRY_FACTOR = 512.0


def _require_cuda_float64(
    name: str,
    value: Any,
    *,
    shape: tuple[int, ...] | None = None,
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
    if not bool(torch.isfinite(value).all().item()):
        raise FloatingPointError(f"{name} contains non-finite values")
    return value


def _require_cuda_int64(
    name: str,
    value: Any,
    *,
    device: torch.device,
) -> torch.Tensor:
    if type(value) is not torch.Tensor:
        raise TypeError(f"{name} must be an exact torch.Tensor")
    if value.device.type != "cuda" or value.device != device:
        raise ValueError(f"{name} must remain on {device}")
    if value.dtype is not torch.int64:
        raise TypeError(f"{name} must use torch.int64")
    if value.ndim != 1:
        raise ValueError(f"{name} must be a vector")
    return value


def _tensor_bytes(value: torch.Tensor) -> bytes:
    return value.detach().contiguous().cpu().numpy().tobytes(order="C")


def _load_sha256(
    strip_forces_w: torch.Tensor,
    leading_edge_endpoints_w: torch.Tensor,
    particle_source_strips: torch.Tensor,
    source_total_force_w: torch.Tensor,
    source_midpoint_moment_w: torch.Tensor,
) -> str:
    digest = hashlib.sha256()
    metadata = json.dumps(
        {
            "schema_id": _SCHEMA,
            "device": str(strip_forces_w.device),
            "dtype": "float64",
            "strip_count": int(strip_forces_w.shape[0]),
            "particle_count": int(particle_source_strips.shape[0]),
        },
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    digest.update(metadata)
    for value in (
        strip_forces_w,
        leading_edge_endpoints_w,
        particle_source_strips,
        source_total_force_w,
        source_midpoint_moment_w,
    ):
        digest.update(_tensor_bytes(value))
    return digest.hexdigest()


def _scaled_close(actual: torch.Tensor, expected: torch.Tensor) -> bool:
    scale = max(
        1.0,
        float(torch.max(torch.abs(actual)).item()),
        float(torch.max(torch.abs(expected)).item()),
    )
    return (
        float(torch.max(torch.abs(actual - expected)).item())
        <= _CLOSURE_FACTOR * _EPS * scale
    )


@dataclass(frozen=True, slots=True)
class Q16CudaLEVImpulseStripLoad:
    """Sealed CUDA strip force and its causal leading-edge source endpoints."""

    strip_forces_w: torch.Tensor
    leading_edge_endpoints_w: torch.Tensor
    particle_source_strips: torch.Tensor
    source_total_force_w: torch.Tensor
    source_midpoint_moment_w: torch.Tensor
    load_sha256: str

    @classmethod
    def from_tensors(
        cls,
        *,
        strip_forces_w: torch.Tensor,
        leading_edge_endpoints_w: torch.Tensor,
        particle_source_strips: torch.Tensor,
    ) -> Q16CudaLEVImpulseStripLoad:
        if cls is not Q16CudaLEVImpulseStripLoad:
            raise TypeError("LEV impulse load subclasses are forbidden")
        forces = _require_cuda_float64("strip_forces_w", strip_forces_w)
        if forces.ndim != 2 or forces.shape[0] <= 0 or forces.shape[1] != 3:
            raise ValueError("strip_forces_w must have shape (positive_strips,3)")
        device = forces.device
        strip_count = int(forces.shape[0])
        endpoints = _require_cuda_float64(
            "leading_edge_endpoints_w",
            leading_edge_endpoints_w,
            shape=(strip_count, 2, 3),
            device=device,
        )
        sources = _require_cuda_int64(
            "particle_source_strips",
            particle_source_strips,
            device=device,
        )
        if bool(torch.any((sources < 0) | (sources >= strip_count)).item()):
            raise ValueError("particle source strip is outside the load topology")
        frozen_forces = forces.detach().contiguous().clone()
        frozen_endpoints = endpoints.detach().contiguous().clone()
        frozen_sources = sources.detach().contiguous().clone()
        total_force = torch.sum(frozen_forces, dim=0)
        midpoints = 0.5 * (frozen_endpoints[:, 0, :] + frozen_endpoints[:, 1, :])
        midpoint_moment = torch.sum(
            torch.linalg.cross(midpoints, frozen_forces, dim=1), dim=0
        )
        load = Q16CudaLEVImpulseStripLoad(
            strip_forces_w=frozen_forces,
            leading_edge_endpoints_w=frozen_endpoints,
            particle_source_strips=frozen_sources,
            source_total_force_w=total_force.detach().clone(),
            source_midpoint_moment_w=midpoint_moment.detach().clone(),
            load_sha256=_load_sha256(
                frozen_forces,
                frozen_endpoints,
                frozen_sources,
                total_force,
                midpoint_moment,
            ),
        )
        return load.validate()

    @classmethod
    def from_solver(cls, solver: Any) -> Q16CudaLEVImpulseStripLoad:
        mode_guard = require_q16_mandatory_aero_mode(solver)
        required = (
            "_q16_impulse_strip_force_w",
            "_q16_impulse_strip_le_endpoints_w",
            "lev_pf",
        )
        missing = [name for name in required if not hasattr(solver, name)]
        if missing:
            raise RuntimeError(
                "solver has not produced a Q16 LEV impulse strip load: "
                + ", ".join(missing)
            )
        particle_field = solver.lev_pf
        if not hasattr(particle_field, "source_strips_cuda"):
            raise RuntimeError("solver particle field has no source-strip owner")
        load = cls.from_tensors(
            strip_forces_w=solver._q16_impulse_strip_force_w,
            leading_edge_endpoints_w=solver._q16_impulse_strip_le_endpoints_w,
            particle_source_strips=particle_field.source_strips_cuda,
        )
        if not hasattr(solver, "_q16_unresolved_impulse_force_w") or not _scaled_close(
            load.source_total_force_w,
            solver._q16_unresolved_impulse_force_w,
        ):
            raise ValueError("strip load does not close solver global impulse force")
        mode_guard.verify(solver)
        return load

    @property
    def strip_count(self) -> int:
        return int(self.strip_forces_w.shape[0])

    @property
    def particle_count(self) -> int:
        return int(self.particle_source_strips.shape[0])

    def validate(self) -> Q16CudaLEVImpulseStripLoad:
        if type(self) is not Q16CudaLEVImpulseStripLoad:
            raise TypeError("LEV impulse strip load must have exact frozen type")
        forces = _require_cuda_float64("strip_forces_w", self.strip_forces_w)
        if forces.ndim != 2 or forces.shape[0] <= 0 or forces.shape[1] != 3:
            raise ValueError("strip_forces_w must have shape (positive_strips,3)")
        device = forces.device
        strip_count = int(forces.shape[0])
        endpoints = _require_cuda_float64(
            "leading_edge_endpoints_w",
            self.leading_edge_endpoints_w,
            shape=(strip_count, 2, 3),
            device=device,
        )
        sources = _require_cuda_int64(
            "particle_source_strips",
            self.particle_source_strips,
            device=device,
        )
        if bool(torch.any((sources < 0) | (sources >= strip_count)).item()):
            raise ValueError("particle source strip is outside the load topology")
        _require_cuda_float64(
            "source_total_force_w",
            self.source_total_force_w,
            shape=(3,),
            device=device,
        )
        _require_cuda_float64(
            "source_midpoint_moment_w",
            self.source_midpoint_moment_w,
            shape=(3,),
            device=device,
        )
        expected_force = torch.sum(forces, dim=0)
        midpoints = 0.5 * (endpoints[:, 0, :] + endpoints[:, 1, :])
        expected_moment = torch.sum(torch.linalg.cross(midpoints, forces, dim=1), dim=0)
        expected_sha = _load_sha256(
            forces,
            endpoints,
            sources,
            self.source_total_force_w,
            self.source_midpoint_moment_w,
        )
        if type(self.load_sha256) is not str or self.load_sha256 != expected_sha:
            raise RuntimeError("Q16 LEV impulse strip load content drift")
        if not _scaled_close(self.source_total_force_w, expected_force):
            raise ValueError("strip resultant does not close source total force")
        if not _scaled_close(self.source_midpoint_moment_w, expected_moment):
            raise ValueError("source midpoint moment does not close strip forces")
        return self


class Q16CudaLEVImpulseTransfer:
    """Apply the declared source-line midpoint work through the Q16 transpose."""

    __slots__ = ("_leading_indices", "surface_transfer")

    def __init__(
        self,
        transfer_map: Q16SurfaceTransferMap,
        *,
        leading_edge_point_indices: np.ndarray,
        device: str,
    ) -> None:
        if type(transfer_map) is not Q16SurfaceTransferMap:
            raise TypeError("transfer_map must be an exact Q16SurfaceTransferMap")
        if type(leading_edge_point_indices) is not np.ndarray:
            raise TypeError("leading_edge_point_indices must be an exact ndarray")
        if (
            leading_edge_point_indices.dtype != np.int64
            or leading_edge_point_indices.ndim != 1
            or leading_edge_point_indices.size < 2
            or not leading_edge_point_indices.flags.c_contiguous
        ):
            raise ValueError(
                "leading_edge_point_indices must be a contiguous int64 vector"
            )
        if bool(
            (leading_edge_point_indices < 0).any()
            or (leading_edge_point_indices >= transfer_map.point_count).any()
            or (np.diff(leading_edge_point_indices) <= 0).any()
        ):
            raise ValueError("leading-edge point indices are invalid or unordered")
        self.surface_transfer = Q16CudaSurfaceTransfer(transfer_map, device=device)
        self._leading_indices = torch.as_tensor(
            np.ascontiguousarray(leading_edge_point_indices),
            dtype=torch.int64,
            device=device,
        )

    def map(
        self,
        load: Q16CudaLEVImpulseStripLoad,
        structural_state: wp.array,
    ) -> wp.array:
        if type(load) is not Q16CudaLEVImpulseStripLoad:
            raise TypeError("load must be an exact Q16CudaLEVImpulseStripLoad")
        load.validate()
        if load.strip_count + 1 != int(self._leading_indices.shape[0]):
            raise ValueError("strip load and leading-edge topology differ")
        expected_device = torch.device(self.surface_transfer.device)
        if load.strip_forces_w.device != expected_device:
            raise ValueError("strip load and Q16 transfer use different CUDA devices")

        all_points = wp.to_torch(self.surface_transfer.interpolate(structural_state))
        if all_points.shape[0] != 1:
            raise ValueError("one strip load requires one structural state")
        leading = torch.index_select(all_points[0], 0, self._leading_indices)
        actual_endpoints = torch.stack((leading[:-1], leading[1:]), dim=1)
        scale = max(
            1.0,
            float(torch.max(torch.abs(actual_endpoints)).item()),
            float(torch.max(torch.abs(load.leading_edge_endpoints_w)).item()),
        )
        mismatch = float(
            torch.max(
                torch.abs(actual_endpoints - load.leading_edge_endpoints_w)
            ).item()
        )
        if mismatch > _GEOMETRY_FACTOR * _EPS * scale:
            raise RuntimeError("LEV source/Q16 leading-edge geometry mismatch")

        strip_forces = load.strip_forces_w
        leading_forces = torch.cat(
            (
                0.5 * strip_forces[:1],
                0.5 * (strip_forces[:-1] + strip_forces[1:]),
                0.5 * strip_forces[-1:],
            ),
            dim=0,
        )
        point_forces = torch.zeros(
            (self.surface_transfer.point_count, 3),
            dtype=torch.float64,
            device=expected_device,
        )
        point_forces[self._leading_indices] = leading_forces
        point_force_wp = wp.from_torch(point_forces.unsqueeze(0), dtype=config.VEC3)
        generalized = self.surface_transfer.transpose(point_force_wp)
        load.validate()
        return generalized


__all__ = [
    "Q16CudaLEVImpulseStripLoad",
    "Q16CudaLEVImpulseTransfer",
]
