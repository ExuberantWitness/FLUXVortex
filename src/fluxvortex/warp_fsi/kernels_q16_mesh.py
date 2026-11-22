"""Deterministic CUDA gather/assembly for shared-node Q16 meshes."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import warp as wp

from fluxvortex.q16_ancf_mesh import Q16ContinuumMesh, Q16MITC16EASMesh
from fluxvortex.q16_ancf_shell import Q16_DOF_PER_ELEMENT, Q16_NODE_COUNT

from . import config
from .kernels_q16_ancf import Q16CudaContinuumOperator
from .kernels_q16_ans_eas import Q16CudaMITC16EASOperator

DTYPE = config.DTYPE


@wp.kernel
def q16_global_to_local_kernel(
    global_state: wp.array(dtype=DTYPE, ndim=2),
    connectivity: wp.array(dtype=wp.int32, ndim=2),
    local_state: wp.array(dtype=DTYPE, ndim=3),
):
    batch, element, local_node, coordinate = wp.tid()
    global_node = connectivity[element, local_node]
    local_state[batch, element, local_node * 6 + coordinate] = global_state[
        batch, global_node * 6 + coordinate
    ]


@wp.kernel
def q16_local_to_global_kernel(
    local_vector: wp.array(dtype=DTYPE, ndim=3),
    node_occurrence_offsets: wp.array(dtype=wp.int32, ndim=1),
    occurrence_elements: wp.array(dtype=wp.int32, ndim=1),
    occurrence_local_nodes: wp.array(dtype=wp.int32, ndim=1),
    global_vector: wp.array(dtype=DTYPE, ndim=2),
):
    batch, global_node, coordinate = wp.tid()
    value = DTYPE(0.0)
    for occurrence in range(
        node_occurrence_offsets[global_node],
        node_occurrence_offsets[global_node + 1],
    ):
        element = occurrence_elements[occurrence]
        local_node = occurrence_local_nodes[occurrence]
        value += local_vector[batch, element, local_node * 6 + coordinate]
    global_vector[batch, global_node * 6 + coordinate] = value


@wp.kernel
def _q16_global_nonfinite_kernel(
    value: wp.array(dtype=DTYPE, ndim=2),
    flag: wp.array(dtype=wp.int32, ndim=1),
):
    batch, coordinate = wp.tid()
    if not wp.isfinite(value[batch, coordinate]):
        wp.atomic_max(flag, 0, 1)


def _require_global_state(
    name: str, value: Any, *, device: str, dof_count: int
) -> None:
    if not isinstance(value, wp.array):
        raise TypeError(f"{name} must be a Warp array")
    if not value.device.is_cuda or value.device.alias != device:
        raise ValueError(f"{name} must reside on CUDA device {device}")
    if value.dtype != DTYPE:
        raise TypeError(f"{name} must use the frozen float64 Warp dtype")
    if value.ndim != 2 or value.shape[0] <= 0 or value.shape[1] != dof_count:
        raise ValueError(f"{name} must have shape (positive_batch, {dof_count})")
    flag = wp.zeros(1, dtype=wp.int32, device=device)
    wp.launch(
        _q16_global_nonfinite_kernel,
        dim=value.shape,
        inputs=[value],
        outputs=[flag],
        device=device,
    )
    wp.synchronize_device(device)
    if int(flag.numpy()[0]) != 0:
        raise FloatingPointError(f"{name} contains non-finite values")


class Q16CudaMeshOperator:
    """Shared-node wrapper over the verified element-local CUDA operators."""

    __slots__ = (
        "_connectivity",
        "_element_operator",
        "_node_occurrence_offsets",
        "_occurrence_elements",
        "_occurrence_local_nodes",
        "device",
        "dof_count",
        "element_count",
        "node_count",
    )

    def __init__(self, model: Q16ContinuumMesh, *, device: str) -> None:
        if type(model) is not Q16ContinuumMesh:
            raise TypeError("model must be an exact Q16ContinuumMesh")
        selected = wp.get_device(device)
        if not selected.is_cuda:
            raise ValueError("Q16 shared-mesh production operator requires CUDA")
        self.device = selected.alias
        self.node_count = model.mesh.node_count
        self.element_count = model.mesh.element_count
        self.dof_count = model.mesh.dof_count
        connectivity = np.ascontiguousarray(model.mesh.connectivity, dtype=np.int32)
        occurrence_elements: list[int] = []
        occurrence_local_nodes: list[int] = []
        offsets = np.zeros(self.node_count + 1, dtype=np.int32)
        for global_node in range(self.node_count):
            for element in range(self.element_count):
                for local_node in range(Q16_NODE_COUNT):
                    if connectivity[element, local_node] == global_node:
                        occurrence_elements.append(element)
                        occurrence_local_nodes.append(local_node)
            offsets[global_node + 1] = len(occurrence_elements)
        if len(occurrence_elements) != self.element_count * Q16_NODE_COUNT:
            raise AssertionError("Q16 mesh occurrence ledger is incomplete")
        self._connectivity = wp.array(connectivity, dtype=wp.int32, device=self.device)
        self._node_occurrence_offsets = wp.array(
            offsets, dtype=wp.int32, device=self.device
        )
        self._occurrence_elements = wp.array(
            np.asarray(occurrence_elements, dtype=np.int32),
            dtype=wp.int32,
            device=self.device,
        )
        self._occurrence_local_nodes = wp.array(
            np.asarray(occurrence_local_nodes, dtype=np.int32),
            dtype=wp.int32,
            device=self.device,
        )
        self._element_operator = Q16CudaContinuumOperator(
            model.elements, device=self.device
        )

    def _gather(self, value: Any, name: str) -> wp.array:
        _require_global_state(name, value, device=self.device, dof_count=self.dof_count)
        return self._gather_prechecked(value)

    def _gather_prechecked(self, value: wp.array) -> wp.array:
        local = wp.zeros(
            (value.shape[0], self.element_count, Q16_DOF_PER_ELEMENT),
            dtype=DTYPE,
            device=self.device,
        )
        wp.launch(
            q16_global_to_local_kernel,
            dim=(value.shape[0], self.element_count, Q16_NODE_COUNT, 6),
            inputs=[value, self._connectivity],
            outputs=[local],
            device=self.device,
        )
        return local

    def _assemble(self, local: wp.array, batch: int) -> wp.array:
        global_vector = wp.zeros(
            (batch, self.dof_count), dtype=DTYPE, device=self.device
        )
        wp.launch(
            q16_local_to_global_kernel,
            dim=(batch, self.node_count, 6),
            inputs=[
                local,
                self._node_occurrence_offsets,
                self._occurrence_elements,
                self._occurrence_local_nodes,
            ],
            outputs=[global_vector],
            device=self.device,
        )
        return global_vector

    def internal_force(self, global_state: Any) -> wp.array:
        local_state = self._gather(global_state, "global state")
        return self._internal_force_prechecked_local(local_state, global_state.shape[0])

    def _internal_force_prechecked_local(
        self, local_state: wp.array, batch: int
    ) -> wp.array:
        local_force = self._element_operator.internal_force(local_state)
        return self._assemble(local_force, batch)

    def _internal_force_prechecked(self, global_state: wp.array) -> wp.array:
        return self._internal_force_prechecked_local(
            self._gather_prechecked(global_state), global_state.shape[0]
        )

    def tangent_action(self, global_state: Any, global_direction: Any) -> wp.array:
        local_state = self._gather(global_state, "global state")
        local_direction = self._gather(global_direction, "global direction")
        if global_state.shape != global_direction.shape:
            raise ValueError("global direction shape differs from global state")
        return self._tangent_action_prechecked_local(
            local_state, local_direction, global_state.shape[0]
        )

    def _tangent_action_prechecked_local(
        self, local_state: wp.array, local_direction: wp.array, batch: int
    ) -> wp.array:
        local_action = self._element_operator.tangent_action(
            local_state, local_direction
        )
        return self._assemble(local_action, batch)

    def _tangent_action_prechecked(
        self, global_state: wp.array, global_direction: wp.array
    ) -> wp.array:
        return self._tangent_action_prechecked_local(
            self._gather_prechecked(global_state),
            self._gather_prechecked(global_direction),
            global_state.shape[0],
        )

    def mass_action(self, global_acceleration: Any) -> wp.array:
        local_acceleration = self._gather(global_acceleration, "global acceleration")
        local_action = self._element_operator.mass_action(local_acceleration)
        return self._assemble(local_action, global_acceleration.shape[0])


class Q16CudaMITC16EASMeshOperator:
    """Shared-node CUDA wrapper for projected/condensed Q16 elements."""

    __slots__ = (
        "_connectivity",
        "_element_operator",
        "_mass_operator",
        "_node_occurrence_offsets",
        "_occurrence_elements",
        "_occurrence_local_nodes",
        "device",
        "dof_count",
        "element_count",
        "node_count",
    )

    def __init__(self, model: Q16MITC16EASMesh, *, device: str) -> None:
        if type(model) is not Q16MITC16EASMesh:
            raise TypeError("model must be an exact Q16MITC16EASMesh")
        selected = wp.get_device(device)
        if not selected.is_cuda:
            raise ValueError("Q16 shared-mesh production operator requires CUDA")
        self.device = selected.alias
        self.node_count = model.mesh.node_count
        self.element_count = model.mesh.element_count
        self.dof_count = model.mesh.dof_count
        connectivity = np.ascontiguousarray(model.mesh.connectivity, dtype=np.int32)
        occurrence_elements: list[int] = []
        occurrence_local_nodes: list[int] = []
        offsets = np.zeros(self.node_count + 1, dtype=np.int32)
        for global_node in range(self.node_count):
            for element in range(self.element_count):
                for local_node in range(Q16_NODE_COUNT):
                    if connectivity[element, local_node] == global_node:
                        occurrence_elements.append(element)
                        occurrence_local_nodes.append(local_node)
            offsets[global_node + 1] = len(occurrence_elements)
        if len(occurrence_elements) != self.element_count * Q16_NODE_COUNT:
            raise AssertionError("Q16 mesh occurrence ledger is incomplete")
        self._connectivity = wp.array(connectivity, dtype=wp.int32, device=self.device)
        self._node_occurrence_offsets = wp.array(
            offsets, dtype=wp.int32, device=self.device
        )
        self._occurrence_elements = wp.array(
            np.asarray(occurrence_elements, dtype=np.int32),
            dtype=wp.int32,
            device=self.device,
        )
        self._occurrence_local_nodes = wp.array(
            np.asarray(occurrence_local_nodes, dtype=np.int32),
            dtype=wp.int32,
            device=self.device,
        )
        self._element_operator = Q16CudaMITC16EASOperator(
            model.elements, device=self.device
        )
        self._mass_operator = Q16CudaContinuumOperator(
            model.mass_elements, device=self.device
        )

    def _gather(self, value: Any, name: str) -> wp.array:
        _require_global_state(name, value, device=self.device, dof_count=self.dof_count)
        return self._gather_prechecked(value)

    def _gather_prechecked(self, value: wp.array) -> wp.array:
        local = wp.zeros(
            (value.shape[0], self.element_count, Q16_DOF_PER_ELEMENT),
            dtype=DTYPE,
            device=self.device,
        )
        wp.launch(
            q16_global_to_local_kernel,
            dim=(value.shape[0], self.element_count, Q16_NODE_COUNT, 6),
            inputs=[value, self._connectivity],
            outputs=[local],
            device=self.device,
        )
        return local

    def _assemble(self, local: wp.array, batch: int) -> wp.array:
        global_vector = wp.zeros(
            (batch, self.dof_count), dtype=DTYPE, device=self.device
        )
        wp.launch(
            q16_local_to_global_kernel,
            dim=(batch, self.node_count, 6),
            inputs=[
                local,
                self._node_occurrence_offsets,
                self._occurrence_elements,
                self._occurrence_local_nodes,
            ],
            outputs=[global_vector],
            device=self.device,
        )
        return global_vector

    def internal_force(self, global_state: Any) -> wp.array:
        local_state = self._gather(global_state, "global state")
        return self._internal_force_prechecked_local(local_state, global_state.shape[0])

    def _internal_force_prechecked_local(
        self, local_state: wp.array, batch: int
    ) -> wp.array:
        local_force = self._element_operator._internal_force_prechecked(local_state)
        return self._assemble(local_force, batch)

    def _linearization_prechecked(self, global_state: wp.array):
        local_state = self._gather_prechecked(global_state)
        return self._element_operator._linearization_prechecked(local_state)

    def validity_only_prechecked(self, global_state: wp.array) -> None:
        """Delegate the S3c validity-only path (same raises, no ``projected_b``=75%)."""

        local_state = self._gather_prechecked(global_state)
        self._element_operator.validity_only_prechecked(local_state)

    def _internal_force_linearized_prechecked(self, linearization) -> wp.array:
        local_force = self._element_operator._internal_force_linearized_prechecked(
            linearization
        )
        return self._assemble(local_force, local_force.shape[0])

    def _internal_force_prechecked(self, global_state: wp.array) -> wp.array:
        return self._internal_force_prechecked_local(
            self._gather_prechecked(global_state), global_state.shape[0]
        )

    def tangent_action(self, global_state: Any, global_direction: Any) -> wp.array:
        local_state = self._gather(global_state, "global state")
        local_direction = self._gather(global_direction, "global direction")
        if global_state.shape != global_direction.shape:
            raise ValueError("global direction shape differs from global state")
        return self._tangent_action_prechecked_local(
            local_state, local_direction, global_state.shape[0]
        )

    def _tangent_action_prechecked_local(
        self, local_state: wp.array, local_direction: wp.array, batch: int
    ) -> wp.array:
        local_action = self._element_operator._tangent_action_prechecked(
            local_state, local_direction
        )
        return self._assemble(local_action, batch)

    def _tangent_action_prechecked(
        self, global_state: wp.array, global_direction: wp.array
    ) -> wp.array:
        return self._tangent_action_linearized_prechecked(
            global_direction,
            self._linearization_prechecked(global_state),
        )

    def _tangent_action_linearized_prechecked(
        self, global_direction: wp.array, linearization
    ) -> wp.array:
        local_direction = self._gather_prechecked(global_direction)
        local_action = self._element_operator._tangent_action_linearized_prechecked(
            local_direction, linearization
        )
        return self._assemble(local_action, global_direction.shape[0])

    def element_block_tangent(self, linearization) -> torch.Tensor:
        """S3-v2A: build the global tangent from per-element 96x96 blocks.

        The element operator acts on the LOCAL layout ``(batch, element_count, 96)`` and is
        independent per element (verified: a direction nonzero in one element produces an
        action nonzero only in that element, and the actions are additive to the last bit
        -- see ``diagnose_tangent_element_locality.py``).  The production probe therefore
        wastes ~96% of its work: it feeds 2,976 GLOBAL unit columns and every kernel still
        runs over all 50 elements, i.e. 47 batches where a local probe needs 2.

        Feeding 96 local unit directions (``e_j`` in every element) yields each element's
        full ``K_e`` in two batches, and the global matrix is their connectivity scatter:

            K[g(e,i), g(e,j)] += K_e[i, j],   g(e,n) = connectivity[e,n]*6 + c

        ALGEBRAIC EXACTNESS, NOT BITWISE: the scatter accumulates in a different order than
        the kernel's node-occurrence summation, and floating-point addition is not
        associative, so the result differs from the column probe at ~1e-15 relative.  The
        caller compares against the column probe with a tolerance and keeps the linear and
        nonlinear residual gates as the acceptance criteria.
        """

        local = Q16_DOF_PER_ELEMENT
        device = self.device
        directions = torch.zeros(
            (local, self.element_count, local), device=device, dtype=torch.float64
        )
        rows = torch.arange(local, device=device)
        directions[rows, :, rows] = 1.0
        # Two batches (64 + 32) keep the per-direction workspace inside the same budget the
        # production probe uses.
        blocks = torch.empty(
            (local, self.element_count, local), device=device, dtype=torch.float64
        )
        for start in range(0, local, 64):
            stop = min(start + 64, local)
            count = stop - start
            # The element operator indexes the linearization PER BATCH, so it has to be
            # broadcast to ``count`` exactly like the production probe does.  Passing the
            # batch-1 linearization with a 96-direction batch reads out of bounds and
            # returns garbage at ~1e10 (found by the equivalence test, not by inspection).
            repeated_linearization = tuple(
                wp.from_torch(
                    wp.to_torch(value)
                    .expand((count,) + tuple(value.shape[1:]))
                    .contiguous(),
                    dtype=DTYPE,
                    requires_grad=False,
                )
                for value in linearization
            )
            action = self._element_operator._tangent_action_linearized_prechecked(
                wp.from_torch(
                    directions[start:stop].contiguous(),
                    dtype=DTYPE,
                    requires_grad=False,
                ),
                repeated_linearization,
            )
            blocks[start:stop] = wp.to_torch(action)
        # blocks[j, e, i] = K_e[i, j]  ->  K_e[i, j] = blocks.permute(1, 2, 0)
        connectivity = wp.to_torch(self._connectivity).to(torch.long)  # (elements, nodes)
        component = torch.arange(6, device=device, dtype=torch.long)
        global_dof = (
            connectivity[:, :, None] * 6 + component[None, None, :]
        ).reshape(self.element_count, local)
        row_index = global_dof[:, :, None].expand(-1, local, local)
        column_index = global_dof[:, None, :].expand(-1, local, local)
        tangent = torch.zeros(
            (self.dof_count, self.dof_count), device=device, dtype=torch.float64
        )
        tangent.index_put_(
            (row_index.reshape(-1), column_index.reshape(-1)),
            blocks.permute(1, 2, 0).reshape(-1),
            accumulate=True,
        )
        return tangent

    def _material_diagonal_linearized_prechecked(self, linearization) -> wp.array:
        local_diagonal = (
            self._element_operator._material_diagonal_linearized_prechecked(
                linearization
            )
        )
        return self._assemble(local_diagonal, local_diagonal.shape[0])

    def _mass_diagonal_prechecked(self, batch: int) -> wp.array:
        local_diagonal = self._mass_operator._mass_diagonal_prechecked(batch)
        return self._assemble(local_diagonal, batch)

    def mass_action(self, global_acceleration: Any) -> wp.array:
        local_acceleration = self._gather(global_acceleration, "global acceleration")
        return self._mass_action_prechecked_local(
            local_acceleration, global_acceleration.shape[0]
        )

    def _mass_action_prechecked_local(
        self, local_acceleration: wp.array, batch: int
    ) -> wp.array:
        local_action = self._mass_operator.mass_action(local_acceleration)
        return self._assemble(local_action, batch)

    def _mass_action_prechecked(self, global_acceleration: wp.array) -> wp.array:
        return self._mass_action_prechecked_local(
            self._gather_prechecked(global_acceleration),
            global_acceleration.shape[0],
        )


__all__ = [
    "Q16CudaMITC16EASMeshOperator",
    "Q16CudaMeshOperator",
    "q16_global_to_local_kernel",
    "q16_local_to_global_kernel",
]
