"""CUDA float64 Q16 surface interpolation and transpose load transfer.

The point map is immutable setup data.  Structural states, velocities, surface
forces and results remain Warp arrays on the selected CUDA device.  The two
kernels are exact transposes of one another, including the director contribution
for nonzero through-thickness surface coordinates.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import warp as wp

from fluxvortex.q16_ancf_shell import Q16_NODE_COUNT
from fluxvortex.q16_work_conjugate_transfer import Q16SurfaceTransferMap

from . import config

DTYPE = config.DTYPE
VEC3 = config.VEC3


@wp.kernel
def _q16_scalar_nonfinite_kernel(
    value: wp.array(dtype=DTYPE, ndim=2),
    flag: wp.array(dtype=wp.int32, ndim=1),
):
    batch, index = wp.tid()
    if not wp.isfinite(value[batch, index]):
        wp.atomic_max(flag, 0, 1)


@wp.kernel
def _q16_vector_nonfinite_kernel(
    value: wp.array(dtype=VEC3, ndim=2),
    flag: wp.array(dtype=wp.int32, ndim=1),
):
    batch, index = wp.tid()
    vector = value[batch, index]
    if (
        not wp.isfinite(vector[0])
        or not wp.isfinite(vector[1])
        or not wp.isfinite(vector[2])
    ):
        wp.atomic_max(flag, 0, 1)


@wp.kernel
def q16_surface_interpolate_kernel(
    state: wp.array(dtype=DTYPE, ndim=2),
    element_indices: wp.array(dtype=wp.int32, ndim=1),
    connectivity: wp.array(dtype=wp.int32, ndim=2),
    zeta: wp.array(dtype=DTYPE, ndim=1),
    shape_values: wp.array(dtype=DTYPE, ndim=2),
    output: wp.array(dtype=VEC3, ndim=2),
):
    batch, point = wp.tid()
    element = element_indices[point]
    value = VEC3(0.0, 0.0, 0.0)
    for node in range(Q16_NODE_COUNT):
        global_node = connectivity[element, node]
        base = global_node * 6
        weight = shape_values[point, node]
        director_weight = zeta[point] * weight
        value += VEC3(
            weight * state[batch, base] + director_weight * state[batch, base + 3],
            weight * state[batch, base + 1] + director_weight * state[batch, base + 4],
            weight * state[batch, base + 2] + director_weight * state[batch, base + 5],
        )
    output[batch, point] = value


@wp.kernel
def q16_surface_transpose_kernel(
    aerodynamic_force: wp.array(dtype=VEC3, ndim=2),
    global_node_point_offsets: wp.array(dtype=wp.int32, ndim=1),
    global_node_point_indices: wp.array(dtype=wp.int32, ndim=1),
    global_node_local_nodes: wp.array(dtype=wp.int32, ndim=1),
    zeta: wp.array(dtype=DTYPE, ndim=1),
    shape_values: wp.array(dtype=DTYPE, ndim=2),
    generalized_force: wp.array(dtype=DTYPE, ndim=2),
):
    batch, global_node, component = wp.tid()
    base = global_node * 6
    value = DTYPE(0.0)
    for cursor in range(
        global_node_point_offsets[global_node],
        global_node_point_offsets[global_node + 1],
    ):
        point = global_node_point_indices[cursor]
        local_node = global_node_local_nodes[cursor]
        weight = shape_values[point, local_node]
        if component >= 3:
            weight *= zeta[point]
        value += weight * aerodynamic_force[batch, point][component % 3]
    generalized_force[batch, base + component] = value


def _require_cuda_array(
    name: str,
    value: Any,
    *,
    device: str,
    dtype: Any,
    ndim: int,
) -> None:
    if not isinstance(value, wp.array):
        raise TypeError(f"{name} must be a Warp array")
    if not value.device.is_cuda or value.device.alias != device:
        raise ValueError(f"{name} must reside on CUDA device {device}")
    if value.dtype != dtype:
        raise TypeError(f"{name} must use the frozen float64 Warp dtype")
    if value.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions")


@wp.kernel
def _q16_copy_eye_kernel(eye: wp.array2d(dtype=DTYPE)):
    d = wp.tid()
    eye[d, d] = DTYPE(1.0)


# Set True only around an active wp.ScopedCapture region; host-side
# validation is illegal inside CUDA-graph capture, so capture-mode callers
# must validate the replayed static outputs once the capture ends.
_CAPTURING = False


def _require_finite_cuda_array(
    name: str, value: wp.array, *, device: str, vector: bool
) -> None:
    if _CAPTURING:
        return
    flag = wp.zeros(1, dtype=wp.int32, device=device)
    kernel = _q16_vector_nonfinite_kernel if vector else _q16_scalar_nonfinite_kernel
    wp.launch(kernel, dim=value.shape, inputs=[value], outputs=[flag], device=device)
    wp.synchronize_device(device)
    if int(flag.numpy()[0]) != 0:
        raise FloatingPointError(f"{name} contains non-finite values")


class Q16CudaSurfaceTransfer:
    """Batched CUDA application of one immutable Q16 surface point map."""

    __slots__ = (
        "_connectivity",
        "_dense_map_torch",
        "_element_indices",
        "_global_node_local_nodes",
        "_global_node_point_indices",
        "_global_node_point_offsets",
        "_shape_values",
        "_zeta",
        "device",
        "element_count",
        "point_count",
        "structural_dof_count",
    )

    def __init__(self, transfer_map: Q16SurfaceTransferMap, *, device: str) -> None:
        if type(transfer_map) is not Q16SurfaceTransferMap:
            raise TypeError("transfer_map must be an exact Q16SurfaceTransferMap")
        if config.dtype_name() != "float64" or DTYPE != wp.float64:
            raise RuntimeError("Q16 production transfer requires float64")
        selected = wp.get_device(device)
        if not selected.is_cuda:
            raise ValueError("Q16 production transfer requires a CUDA device")
        self.device = selected.alias
        self.element_count = transfer_map.element_count
        self.point_count = transfer_map.point_count
        self.structural_dof_count = transfer_map.structural_dof_count
        indices = np.ascontiguousarray(transfer_map.element_indices, dtype=np.int32)
        connectivity = np.ascontiguousarray(
            transfer_map.mesh.connectivity, dtype=np.int32
        )
        zeta = np.ascontiguousarray(
            transfer_map.parametric_coordinates[:, 2], dtype=np.float64
        )
        shapes = np.ascontiguousarray(transfer_map.shape_values, dtype=np.float64)
        occurrences: list[tuple[int, int, int]] = []
        for point, element in enumerate(indices):
            for local_node, global_node in enumerate(connectivity[element]):
                occurrences.append((int(global_node), point, local_node))
        occurrences.sort(key=lambda item: item[0])
        global_nodes = np.fromiter((item[0] for item in occurrences), dtype=np.int32)
        point_indices = np.fromiter((item[1] for item in occurrences), dtype=np.int32)
        local_nodes = np.fromiter((item[2] for item in occurrences), dtype=np.int32)
        counts = np.bincount(global_nodes, minlength=transfer_map.mesh.node_count)
        point_offsets = np.empty(transfer_map.mesh.node_count + 1, dtype=np.int32)
        point_offsets[0] = 0
        np.cumsum(counts, dtype=np.int64, out=point_offsets[1:])
        self._connectivity = wp.array(connectivity, dtype=wp.int32, device=self.device)
        self._element_indices = wp.array(indices, dtype=wp.int32, device=self.device)
        self._global_node_point_indices = wp.array(
            np.ascontiguousarray(point_indices),
            dtype=wp.int32,
            device=self.device,
        )
        self._global_node_point_offsets = wp.array(
            np.ascontiguousarray(point_offsets),
            dtype=wp.int32,
            device=self.device,
        )
        self._global_node_local_nodes = wp.array(
            np.ascontiguousarray(local_nodes),
            dtype=wp.int32,
            device=self.device,
        )
        self._zeta = wp.array(zeta, dtype=DTYPE, device=self.device)
        self._shape_values = wp.array(shapes, dtype=DTYPE, device=self.device)
        self._dense_map_torch = None

    def dense_map(self):
        """One-time extraction of the fixed linear map as a dense torch matrix.

        interpolate(state) is exactly ``dense_map() @ state[0]`` (same
        weights; fp64 summation-order differences only).  Replaces the
        kernel scatter with one cuBLAS GEMV on hot paths.
        """
        cached = self._dense_map_torch
        if cached is not None:
            return cached
        import torch
        dof = self.structural_dof_count
        eye = wp.zeros((dof, dof), dtype=DTYPE, device=self.device)
        wp.launch(
            _q16_copy_eye_kernel,
            dim=dof,
            inputs=[],
            outputs=[eye],
            device=self.device,
        )
        columns = self.interpolate(eye)
        wp.synchronize_device(self.device)
        # columns[d, p, c] = map(e_d)[p, c] -> matrix[p*3+c, d]
        matrix = (
            wp.to_torch(columns)
            .permute(1, 2, 0)
            .reshape(self.point_count * 3, dof)
            .contiguous()
        )
        self._dense_map_torch = matrix
        return matrix

    def interpolate_dense(self, flat_state) -> "torch.Tensor":
        """Dense-GEMV equivalent of interpolate for one (dof,) sample."""
        import torch
        q = flat_state if torch.is_tensor(flat_state) else wp.to_torch(flat_state)
        if q.ndim == 2:
            q = q[0]
        return (self.dense_map() @ q).reshape(self.point_count, 3)

    def interpolate(self, state: Any) -> wp.array:
        """Interpolate batched Q16 states or velocities at all mapped points."""

        _require_cuda_array("state", state, device=self.device, dtype=DTYPE, ndim=2)
        if state.shape[0] <= 0 or state.shape[1] != self.structural_dof_count:
            raise ValueError(
                "state must have shape (positive_batch, structural_dof_count)"
            )
        _require_finite_cuda_array("state", state, device=self.device, vector=False)
        output = wp.zeros(
            (state.shape[0], self.point_count),
            dtype=VEC3,
            device=self.device,
        )
        wp.launch(
            q16_surface_interpolate_kernel,
            dim=(state.shape[0], self.point_count),
            inputs=[
                state,
                self._element_indices,
                self._connectivity,
                self._zeta,
                self._shape_values,
            ],
            outputs=[output],
            device=self.device,
        )
        return output

    def transpose(self, aerodynamic_force: Any) -> wp.array:
        """Map batched point forces into Q16 generalized coordinates on CUDA."""

        _require_cuda_array(
            "aerodynamic_force",
            aerodynamic_force,
            device=self.device,
            dtype=VEC3,
            ndim=2,
        )
        if (
            aerodynamic_force.shape[0] <= 0
            or aerodynamic_force.shape[1] != self.point_count
        ):
            raise ValueError(
                "aerodynamic_force must have shape (positive_batch, point_count)"
            )
        _require_finite_cuda_array(
            "aerodynamic_force",
            aerodynamic_force,
            device=self.device,
            vector=True,
        )
        output = wp.zeros(
            (aerodynamic_force.shape[0], self.structural_dof_count),
            dtype=DTYPE,
            device=self.device,
        )
        wp.launch(
            q16_surface_transpose_kernel,
            dim=(
                aerodynamic_force.shape[0],
                self.structural_dof_count // 6,
                6,
            ),
            inputs=[
                aerodynamic_force,
                self._global_node_point_offsets,
                self._global_node_point_indices,
                self._global_node_local_nodes,
                self._zeta,
                self._shape_values,
            ],
            outputs=[output],
            device=self.device,
        )
        return output


__all__ = [
    "Q16CudaSurfaceTransfer",
    "q16_surface_interpolate_kernel",
    "q16_surface_transpose_kernel",
]
