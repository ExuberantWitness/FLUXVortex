"""Local conservative CUDA transfer for real Ptera point loads.

Ptera evaluates four effective bound-vortex leg forces and one unsteady
pressure force per panel.  Some vortex centers (notably the trailing back
vortex) are not material midsurface points, so inverse-fitting one Q16
``(xi, eta, zeta)`` coordinate is not a valid general transfer.

This module instead freezes one current-configuration algebraic stencil per
load point.  The stencil uses the owning panel's four Q16 vertices on both
shell faces.  CUDA solves the minimum-norm affine reconstruction and the exact
transpose maps the point forces back through those support points.  The map is
therefore locally conservative in force, moment and virtual work for the
frozen predictor/corrector trial.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any

import numpy as np
import torch
import warp as wp

from fluxvortex.q16_work_conjugate_transfer import Q16SurfaceTransferMap

from . import config
from .kernels_q16_transfer import Q16CudaSurfaceTransfer
from .q16_aero_load_packet import Q16CudaAerodynamicLoadPacket
from .q16_lev_impulse_transfer import (
    Q16CudaLEVImpulseStripLoad,
    Q16CudaLEVImpulseTransfer,
)

DTYPE = config.DTYPE
VEC3 = config.VEC3
_EPS = float(np.finfo(np.float64).eps)
_SQRT_TINY = math.sqrt(float(np.finfo(np.float64).tiny))
_RANK_FACTOR = 4096.0
_CLOSURE_FACTOR = 65536.0
_GEOMETRY_FACTOR = 32768.0
_MAX_ABS_WEIGHT = 64.0
_RESULT_SCHEMA = "fluxv-q16-ptera-resolved-transfer-v1"
_COMPLETE_SCHEMA = "fluxv-q16-complete-aero-load-transfer-v1"


@wp.kernel
def _q16_local_support_force_transpose_kernel(
    load_forces: wp.array(dtype=VEC3, ndim=2),
    load_weights: wp.array(dtype=DTYPE, ndim=2),
    support_load_offsets: wp.array(dtype=wp.int32, ndim=1),
    support_load_indices: wp.array(dtype=wp.int32, ndim=1),
    support_load_slots: wp.array(dtype=wp.int32, ndim=1),
    support_forces: wp.array(dtype=VEC3, ndim=2),
):
    support = wp.tid()
    value = VEC3(0.0, 0.0, 0.0)
    for cursor in range(
        support_load_offsets[support], support_load_offsets[support + 1]
    ):
        load = support_load_indices[cursor]
        slot = support_load_slots[cursor]
        value += load_weights[load, slot] * load_forces[0, load]
    support_forces[0, support] = value


def _positive_exact_int(name: str, value: int) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive exact int")
    return value


def _tensor_bytes(value: torch.Tensor) -> bytes:
    return value.detach().contiguous().cpu().numpy().tobytes(order="C")


def _warp_bytes(value: wp.array) -> bytes:
    return np.ascontiguousarray(value.numpy(), dtype=np.float64).tobytes(order="C")


def _scaled_error(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float]:
    error = float(torch.max(torch.abs(actual - expected)).item())
    scale = max(
        1.0,
        float(torch.max(torch.abs(actual)).item()),
        float(torch.max(torch.abs(expected)).item()),
    )
    return error, scale


def _require_structural_state(
    value: Any,
    *,
    device: str,
    structural_dof_count: int,
) -> wp.array:
    if not isinstance(value, wp.array):
        raise TypeError("structural_state must be a Warp array")
    if not value.device.is_cuda or value.device.alias != device:
        raise ValueError(f"structural_state must reside on CUDA device {device}")
    if value.dtype != DTYPE:
        raise TypeError("structural_state must use Warp float64")
    if value.ndim != 2 or value.shape != (1, structural_dof_count):
        raise ValueError("structural_state must have shape (1, structural_dof_count)")
    return value


def _support_map(vertex_map: Q16SurfaceTransferMap) -> Q16SurfaceTransferMap:
    indices = np.repeat(vertex_map.element_indices, 2).astype(np.int64, copy=False)
    coordinates = np.repeat(vertex_map.parametric_coordinates, 2, axis=0).copy()
    coordinates[0::2, 2] = -1.0
    coordinates[1::2, 2] = 1.0
    return Q16SurfaceTransferMap(
        mesh=vertex_map.mesh,
        element_indices=np.ascontiguousarray(indices),
        parametric_coordinates=np.ascontiguousarray(coordinates),
    )


def _panel_support_indices(chord_count: int, span_count: int) -> np.ndarray:
    result = np.empty((chord_count * span_count, 8), dtype=np.int64)
    panel = 0
    for chord in range(chord_count):
        for span in range(span_count):
            vertices = (
                chord * (span_count + 1) + span,
                chord * (span_count + 1) + span + 1,
                (chord + 1) * (span_count + 1) + span,
                (chord + 1) * (span_count + 1) + span + 1,
            )
            result[panel] = np.asarray(
                [2 * vertex + face for vertex in vertices for face in (0, 1)],
                dtype=np.int64,
            )
            panel += 1
    if panel != chord_count * span_count:
        raise AssertionError("Ptera panel topology count drift")
    return np.ascontiguousarray(result)


def _support_csr(
    point_support_indices: np.ndarray, support_count: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    occurrences: list[tuple[int, int, int]] = []
    for load in range(point_support_indices.shape[0]):
        for slot in range(point_support_indices.shape[1]):
            occurrences.append((int(point_support_indices[load, slot]), load, slot))
    occurrences.sort(key=lambda item: (item[0], item[1], item[2]))
    support = np.asarray([item[0] for item in occurrences], dtype=np.int64)
    if support.size == 0 or support.min() < 0 or support.max() >= support_count:
        raise ValueError("point support index is outside the Q16 support map")
    counts = np.bincount(support, minlength=support_count)
    offsets = np.empty(support_count + 1, dtype=np.int32)
    offsets[0] = 0
    np.cumsum(counts, dtype=np.int64, out=offsets[1:])
    loads = np.asarray([item[1] for item in occurrences], dtype=np.int32)
    slots = np.asarray([item[2] for item in occurrences], dtype=np.int32)
    return (
        np.ascontiguousarray(offsets),
        np.ascontiguousarray(loads),
        np.ascontiguousarray(slots),
    )


def _resolved_result_sha256(
    *,
    point_count: int,
    load_weights: torch.Tensor,
    support_forces_w: torch.Tensor,
    generalized_force: wp.array,
    point_reconstruction_max_abs_error: float,
    resolved_force_max_abs_error: float,
    resolved_moment_max_abs_error: float,
) -> str:
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            {
                "schema": _RESULT_SCHEMA,
                "point_count": point_count,
                "point_reconstruction_max_abs_error_hex": (
                    point_reconstruction_max_abs_error.hex()
                ),
                "resolved_force_max_abs_error_hex": (
                    resolved_force_max_abs_error.hex()
                ),
                "resolved_moment_max_abs_error_hex": (
                    resolved_moment_max_abs_error.hex()
                ),
            },
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    )
    digest.update(_tensor_bytes(load_weights))
    digest.update(_tensor_bytes(support_forces_w))
    digest.update(_warp_bytes(generalized_force))
    return digest.hexdigest()


@dataclass(frozen=True, slots=True, eq=False)
class Q16PteraResolvedTransferResult:
    """Sealed resolved-load transfer for one current predictor trial."""

    point_count: int
    load_weights: torch.Tensor
    support_forces_w: torch.Tensor
    generalized_force: wp.array
    point_reconstruction_max_abs_error: float
    resolved_force_max_abs_error: float
    resolved_moment_max_abs_error: float
    result_sha256: str

    def validate(self) -> Q16PteraResolvedTransferResult:
        if type(self) is not Q16PteraResolvedTransferResult:
            raise TypeError("resolved transfer result must have exact frozen type")
        if type(self.point_count) is not int or self.point_count <= 0:
            raise ValueError("resolved transfer point_count drift")
        for name, value in (
            ("load_weights", self.load_weights),
            ("support_forces_w", self.support_forces_w),
        ):
            if type(value) is not torch.Tensor or value.device.type != "cuda":
                raise ValueError(f"{name} must be an exact CUDA tensor")
            if value.dtype is not torch.float64 or not bool(
                torch.isfinite(value).all()
            ):
                raise TypeError(f"{name} must contain finite CUDA float64 values")
        if self.load_weights.shape != (self.point_count, 8):
            raise ValueError("resolved transfer weight shape drift")
        if self.support_forces_w.ndim != 2 or self.support_forces_w.shape[1] != 3:
            raise ValueError("resolved transfer support-force shape drift")
        if not isinstance(self.generalized_force, wp.array):
            raise TypeError("generalized_force must be a Warp array")
        for name, value in (
            (
                "point_reconstruction_max_abs_error",
                self.point_reconstruction_max_abs_error,
            ),
            ("resolved_force_max_abs_error", self.resolved_force_max_abs_error),
            ("resolved_moment_max_abs_error", self.resolved_moment_max_abs_error),
        ):
            if type(value) is not float or not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        expected = _resolved_result_sha256(
            point_count=self.point_count,
            load_weights=self.load_weights,
            support_forces_w=self.support_forces_w,
            generalized_force=self.generalized_force,
            point_reconstruction_max_abs_error=(
                self.point_reconstruction_max_abs_error
            ),
            resolved_force_max_abs_error=self.resolved_force_max_abs_error,
            resolved_moment_max_abs_error=self.resolved_moment_max_abs_error,
        )
        if type(self.result_sha256) is not str or self.result_sha256 != expected:
            raise RuntimeError("resolved transfer result content drift")
        return self


class Q16CudaPteraResolvedLoadTransfer:
    """Conservative five-block Ptera load transfer on CUDA float64."""

    __slots__ = (
        "_point_support_indices",
        "_support_load_indices",
        "_support_load_offsets",
        "_support_load_slots",
        "chordwise_panel_count",
        "device",
        "panel_count",
        "point_count",
        "spanwise_panel_count",
        "support_transfer",
        "vertex_map",
    )

    def __init__(
        self,
        vertex_map: Q16SurfaceTransferMap,
        *,
        chordwise_panel_count: int,
        spanwise_panel_count: int,
        device: str,
    ) -> None:
        if type(vertex_map) is not Q16SurfaceTransferMap:
            raise TypeError("vertex_map must be an exact Q16SurfaceTransferMap")
        chord_count = _positive_exact_int(
            "chordwise_panel_count", chordwise_panel_count
        )
        span_count = _positive_exact_int("spanwise_panel_count", spanwise_panel_count)
        expected_vertices = (chord_count + 1) * (span_count + 1)
        if vertex_map.point_count != expected_vertices:
            raise ValueError("Ptera vertex count differs from Q16 vertex map")
        selected = wp.get_device(device)
        if not selected.is_cuda:
            raise ValueError("Ptera resolved transfer requires a CUDA device")
        if DTYPE != wp.float64 or config.dtype_name() != "float64":
            raise RuntimeError("Ptera resolved transfer requires float64")
        self.device = selected.alias
        self.chordwise_panel_count = chord_count
        self.spanwise_panel_count = span_count
        self.panel_count = chord_count * span_count
        self.point_count = 5 * self.panel_count
        self.vertex_map = vertex_map
        self.support_transfer = Q16CudaSurfaceTransfer(
            _support_map(vertex_map), device=self.device
        )

        panel_support = _panel_support_indices(chord_count, span_count)
        owners = np.tile(np.arange(self.panel_count, dtype=np.int64), 5)
        point_support = np.ascontiguousarray(panel_support[owners], dtype=np.int64)
        offsets, loads, slots = _support_csr(
            point_support, self.support_transfer.point_count
        )
        self._point_support_indices = torch.as_tensor(
            point_support, dtype=torch.int64, device=self.device
        )
        self._support_load_offsets = wp.array(
            offsets, dtype=wp.int32, device=self.device
        )
        self._support_load_indices = wp.array(loads, dtype=wp.int32, device=self.device)
        self._support_load_slots = wp.array(slots, dtype=wp.int32, device=self.device)

    @property
    def structural_dof_count(self) -> int:
        return self.support_transfer.structural_dof_count

    def _weights(
        self,
        point_positions_w: torch.Tensor,
        support_positions_w: torch.Tensor,
    ) -> tuple[torch.Tensor, float]:
        local = support_positions_w[self._point_support_indices]
        origin = torch.mean(local, dim=1)
        offsets = local - origin.unsqueeze(1)
        scale = torch.amax(torch.abs(offsets), dim=(1, 2))
        if bool(torch.any(scale <= _SQRT_TINY).item()):
            raise RuntimeError("same-panel Q16 support lost affine rank")
        normalized = offsets / scale[:, None, None]
        matrix = torch.cat(
            (
                torch.ones(
                    (self.point_count, 1, 8),
                    dtype=torch.float64,
                    device=self.device,
                ),
                normalized.transpose(1, 2),
            ),
            dim=1,
        )
        rhs = torch.cat(
            (
                torch.ones(
                    (self.point_count, 1),
                    dtype=torch.float64,
                    device=self.device,
                ),
                (point_positions_w - origin) / scale[:, None],
            ),
            dim=1,
        )
        gram = matrix @ matrix.transpose(1, 2)
        eigenvalues = torch.linalg.eigvalsh(gram)
        maximum = eigenvalues[:, -1]
        minimum = eigenvalues[:, 0]
        if bool(torch.any(~torch.isfinite(eigenvalues)).item()) or bool(
            torch.any(minimum <= _RANK_FACTOR * _EPS * maximum).item()
        ):
            raise RuntimeError("same-panel Q16 support lost affine rank")
        dual = torch.linalg.solve(gram, rhs.unsqueeze(2))
        weights = (matrix.transpose(1, 2) @ dual).squeeze(2)
        if not bool(torch.isfinite(weights).all().item()):
            raise FloatingPointError("same-panel affine weights became non-finite")
        if float(torch.max(torch.abs(weights)).item()) > _MAX_ABS_WEIGHT:
            raise RuntimeError(
                "Ptera point requires excessive same-panel extrapolation"
            )
        reconstructed = torch.sum(weights.unsqueeze(2) * local, dim=1)
        error, geometry_scale = _scaled_error(reconstructed, point_positions_w)
        if error > _GEOMETRY_FACTOR * _EPS * geometry_scale:
            raise RuntimeError("Ptera point is outside the conservative Q16 stencil")
        return weights, error

    def _support_forces(
        self, point_forces_w: torch.Tensor, load_weights: torch.Tensor
    ) -> wp.array:
        load_force_wp = wp.from_torch(
            point_forces_w.unsqueeze(0), dtype=VEC3, requires_grad=False
        )
        weight_wp = wp.from_torch(load_weights, dtype=DTYPE, requires_grad=False)
        support_force_wp = wp.zeros(
            (1, self.support_transfer.point_count),
            dtype=VEC3,
            device=self.device,
        )
        wp.launch(
            _q16_local_support_force_transpose_kernel,
            dim=self.support_transfer.point_count,
            inputs=[
                load_force_wp,
                weight_wp,
                self._support_load_offsets,
                self._support_load_indices,
                self._support_load_slots,
            ],
            outputs=[support_force_wp],
            device=self.device,
        )
        return support_force_wp

    def map(
        self,
        packet: Q16CudaAerodynamicLoadPacket,
        structural_state: Any,
    ) -> Q16PteraResolvedTransferResult:
        if type(packet) is not Q16CudaAerodynamicLoadPacket:
            raise TypeError("packet must be an exact Q16CudaAerodynamicLoadPacket")
        packet.validate()
        if packet.point_count != self.point_count:
            raise ValueError("packet does not contain five complete Ptera load blocks")
        expected_device = torch.device(self.device)
        if packet.point_positions_w.device != expected_device:
            raise ValueError("packet and Ptera transfer use different CUDA devices")
        state = _require_structural_state(
            structural_state,
            device=self.device,
            structural_dof_count=self.structural_dof_count,
        )
        support_position_wp = self.support_transfer.interpolate(state)
        support_positions = wp.to_torch(support_position_wp)[0]
        weights, reconstruction_error = self._weights(
            packet.point_positions_w, support_positions
        )
        support_force_wp = self._support_forces(packet.point_forces_w, weights)
        support_forces = wp.to_torch(support_force_wp)[0]

        support_total = torch.sum(support_forces, dim=0)
        support_moment = torch.sum(
            torch.linalg.cross(support_positions, support_forces, dim=1), dim=0
        )
        force_error, force_scale = _scaled_error(support_total, packet.resolved_force_w)
        moment_error, moment_scale = _scaled_error(
            support_moment, packet.resolved_moment_w
        )
        if force_error > _CLOSURE_FACTOR * _EPS * force_scale:
            raise RuntimeError("same-panel transfer does not close resolved force")
        if moment_error > _CLOSURE_FACTOR * _EPS * moment_scale:
            raise RuntimeError("same-panel transfer does not close resolved moment")
        generalized = self.support_transfer.transpose(support_force_wp)
        frozen_weights = weights.detach().contiguous().clone()
        frozen_support_forces = support_forces.detach().contiguous().clone()
        result_sha = _resolved_result_sha256(
            point_count=self.point_count,
            load_weights=frozen_weights,
            support_forces_w=frozen_support_forces,
            generalized_force=generalized,
            point_reconstruction_max_abs_error=reconstruction_error,
            resolved_force_max_abs_error=force_error,
            resolved_moment_max_abs_error=moment_error,
        )
        result = Q16PteraResolvedTransferResult(
            point_count=self.point_count,
            load_weights=frozen_weights,
            support_forces_w=frozen_support_forces,
            generalized_force=generalized,
            point_reconstruction_max_abs_error=reconstruction_error,
            resolved_force_max_abs_error=force_error,
            resolved_moment_max_abs_error=moment_error,
            result_sha256=result_sha,
        )
        packet.validate()
        return result.validate()

    def interpolate_frozen_point_direction(
        self,
        result: Q16PteraResolvedTransferResult,
        structural_direction: Any,
    ) -> torch.Tensor:
        if type(result) is not Q16PteraResolvedTransferResult:
            raise TypeError("result must be an exact resolved transfer result")
        result.validate()
        if result.point_count != self.point_count:
            raise ValueError("resolved result topology differs from transfer")
        direction = _require_structural_state(
            structural_direction,
            device=self.device,
            structural_dof_count=self.structural_dof_count,
        )
        point_direction = self._interpolate_frozen_point_direction_prechecked(
            result, direction
        )
        result.validate()
        return point_direction

    def _interpolate_frozen_point_direction_prechecked(
        self,
        result: Q16PteraResolvedTransferResult,
        direction: wp.array,
    ) -> torch.Tensor:
        support_direction = wp.to_torch(self.support_transfer.interpolate(direction))[0]
        local = support_direction[self._point_support_indices]
        return torch.sum(result.load_weights.unsqueeze(2) * local, dim=1)

    def transpose_frozen_point_forces(
        self,
        result: Q16PteraResolvedTransferResult,
        point_forces_w: torch.Tensor,
    ) -> wp.array:
        """Apply an issued trial's exact transpose stencil to new point forces.

        The support geometry and affine weights remain frozen to ``result``;
        only the force operands change.  This is the work-conjugate boundary
        needed by structural-clock velocity terms.  It does not re-fit load
        points and cannot advance aerodynamic state.
        """

        if type(result) is not Q16PteraResolvedTransferResult:
            raise TypeError("result must be an exact frozen resolved result")
        result.validate()
        if (
            type(point_forces_w) is not torch.Tensor
            or point_forces_w.device != torch.device(self.device)
            or point_forces_w.dtype is not torch.float64
            or point_forces_w.requires_grad
            or tuple(point_forces_w.shape) != (self.point_count, 3)
        ):
            raise ValueError(
                "point_forces_w must be detached CUDA float64 with frozen shape"
            )
        if not bool(torch.isfinite(point_forces_w).all().item()):
            raise FloatingPointError("point force update contains non-finite values")
        generalized = self._transpose_frozen_point_forces_prechecked(
            result, point_forces_w
        )
        result.validate()
        return generalized

    def _transpose_frozen_point_forces_prechecked(
        self,
        result: Q16PteraResolvedTransferResult,
        point_forces_w: torch.Tensor,
    ) -> wp.array:
        support_force_wp = self._support_forces(
            point_forces_w, result.load_weights
        )
        return self.support_transfer.transpose(support_force_wp)


def _complete_result_sha256(
    resolved_sha256: str,
    lev_impulse_generalized_force: wp.array,
    generalized_force: wp.array,
) -> str:
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            {"schema": _COMPLETE_SCHEMA, "resolved_sha256": resolved_sha256},
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    )
    digest.update(_warp_bytes(lev_impulse_generalized_force))
    digest.update(_warp_bytes(generalized_force))
    return digest.hexdigest()


@dataclass(frozen=True, slots=True, eq=False)
class Q16CompleteAeroLoadResult:
    resolved: Q16PteraResolvedTransferResult
    lev_impulse_generalized_force: wp.array
    generalized_force: wp.array
    result_sha256: str

    def validate(self) -> Q16CompleteAeroLoadResult:
        if type(self) is not Q16CompleteAeroLoadResult:
            raise TypeError("complete load result must have exact frozen type")
        self.resolved.validate()
        for name, value in (
            ("lev_impulse_generalized_force", self.lev_impulse_generalized_force),
            ("generalized_force", self.generalized_force),
        ):
            if not isinstance(value, wp.array) or not value.device.is_cuda:
                raise ValueError(f"{name} must be a CUDA Warp array")
            if (
                value.dtype != DTYPE
                or value.shape != self.resolved.generalized_force.shape
            ):
                raise ValueError(f"{name} shape or dtype drift")
        expected = _complete_result_sha256(
            self.resolved.result_sha256,
            self.lev_impulse_generalized_force,
            self.generalized_force,
        )
        if type(self.result_sha256) is not str or self.result_sha256 != expected:
            raise RuntimeError("complete Q16 aerodynamic load content drift")
        return self


class Q16CudaCompleteAeroLoadTransfer:
    """Compose real resolved and source-owned separated-LEV loads."""

    __slots__ = ("impulse_transfer", "resolved_transfer")

    def __init__(
        self,
        resolved_transfer: Q16CudaPteraResolvedLoadTransfer,
        impulse_transfer: Q16CudaLEVImpulseTransfer,
    ) -> None:
        if type(resolved_transfer) is not Q16CudaPteraResolvedLoadTransfer:
            raise TypeError("resolved_transfer must have exact production type")
        if type(impulse_transfer) is not Q16CudaLEVImpulseTransfer:
            raise TypeError("impulse_transfer must have exact production type")
        if (
            resolved_transfer.device != impulse_transfer.surface_transfer.device
            or resolved_transfer.structural_dof_count
            != impulse_transfer.surface_transfer.structural_dof_count
        ):
            raise ValueError("resolved and LEV transfers use different Q16 owners")
        self.resolved_transfer = resolved_transfer
        self.impulse_transfer = impulse_transfer

    def map(
        self,
        packet: Q16CudaAerodynamicLoadPacket,
        lev_impulse_load: Q16CudaLEVImpulseStripLoad,
        structural_state: Any,
    ) -> Q16CompleteAeroLoadResult:
        if type(packet) is not Q16CudaAerodynamicLoadPacket:
            raise TypeError("packet must be an exact Q16CudaAerodynamicLoadPacket")
        if type(lev_impulse_load) is not Q16CudaLEVImpulseStripLoad:
            raise TypeError("lev_impulse_load must have exact production type")
        packet.validate()
        lev_impulse_load.validate()
        error, scale = _scaled_error(
            packet.unresolved_impulse_force_w,
            lev_impulse_load.source_total_force_w,
        )
        if error > _CLOSURE_FACTOR * _EPS * scale:
            raise RuntimeError("LEV impulse owner does not match aerodynamic packet")
        resolved = self.resolved_transfer.map(packet, structural_state)
        impulse = self.impulse_transfer.map(lev_impulse_load, structural_state)
        total_torch = wp.to_torch(resolved.generalized_force) + wp.to_torch(impulse)
        total = wp.clone(wp.from_torch(total_torch, dtype=DTYPE, requires_grad=False))
        result_sha = _complete_result_sha256(
            resolved.result_sha256,
            impulse,
            total,
        )
        result = Q16CompleteAeroLoadResult(
            resolved=resolved,
            lev_impulse_generalized_force=impulse,
            generalized_force=total,
            result_sha256=result_sha,
        )
        packet.validate()
        lev_impulse_load.validate()
        return result.validate()


__all__ = [
    "Q16CompleteAeroLoadResult",
    "Q16CudaCompleteAeroLoadTransfer",
    "Q16CudaPteraResolvedLoadTransfer",
    "Q16PteraResolvedTransferResult",
]
