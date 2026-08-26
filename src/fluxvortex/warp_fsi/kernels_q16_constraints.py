"""CUDA projections for immutable shared-node Q16 boundary constraints."""

from __future__ import annotations

from typing import Any

import numpy as np
import warp as wp

from fluxvortex.q16_boundary_constraints import Q16BoundaryConstraints

from . import config

DTYPE = config.DTYPE


@wp.kernel
def _q16_project_free_kernel(
    value: wp.array(dtype=DTYPE, ndim=2),
    constrained_mask: wp.array(dtype=wp.int32, ndim=1),
    result: wp.array(dtype=DTYPE, ndim=2),
):
    batch, dof = wp.tid()
    if constrained_mask[dof] == 0:
        result[batch, dof] = value[batch, dof]
    else:
        result[batch, dof] = DTYPE(0.0)


@wp.kernel
def _q16_extract_reaction_kernel(
    value: wp.array(dtype=DTYPE, ndim=2),
    constrained_mask: wp.array(dtype=wp.int32, ndim=1),
    result: wp.array(dtype=DTYPE, ndim=2),
):
    batch, dof = wp.tid()
    if constrained_mask[dof] == 1:
        result[batch, dof] = value[batch, dof]
    else:
        result[batch, dof] = DTYPE(0.0)


@wp.kernel
def _q16_enforce_state_kernel(
    state: wp.array(dtype=DTYPE, ndim=2),
    constrained_mask: wp.array(dtype=wp.int32, ndim=1),
    prescribed_state: wp.array(dtype=DTYPE, ndim=1),
    result: wp.array(dtype=DTYPE, ndim=2),
):
    batch, dof = wp.tid()
    if constrained_mask[dof] == 1:
        result[batch, dof] = prescribed_state[dof]
    else:
        result[batch, dof] = state[batch, dof]


@wp.kernel
def _q16_constraint_nonfinite_kernel(
    value: wp.array(dtype=DTYPE, ndim=2),
    flag: wp.array(dtype=wp.int32, ndim=1),
):
    batch, dof = wp.tid()
    if not wp.isfinite(value[batch, dof]):
        wp.atomic_max(flag, 0, 1)


@wp.kernel
def _q16_constraint_kinematic_violation_kernel(
    state: wp.array(dtype=DTYPE, ndim=2),
    velocity: wp.array(dtype=DTYPE, ndim=2),
    acceleration: wp.array(dtype=DTYPE, ndim=2),
    constrained_mask: wp.array(dtype=wp.int32, ndim=1),
    prescribed_state: wp.array(dtype=DTYPE, ndim=1),
    flag: wp.array(dtype=wp.int32, ndim=1),
):
    batch, dof = wp.tid()
    if constrained_mask[dof] == 1:
        if (
            state[batch, dof] != prescribed_state[dof]
            or velocity[batch, dof] != DTYPE(0.0)
            or acceleration[batch, dof] != DTYPE(0.0)
        ):
            wp.atomic_max(flag, 0, 1)


class Q16CudaBoundaryConstraints:
    """Device-resident counterpart of one exact Q16 boundary owner."""

    __slots__ = (
        "_constrained_mask",
        "_prescribed_state",
        "device",
        "dof_count",
    )

    def __init__(self, boundary: Q16BoundaryConstraints, *, device: str) -> None:
        if type(boundary) is not Q16BoundaryConstraints:
            raise TypeError("boundary must be an exact Q16BoundaryConstraints")
        selected = wp.get_device(device)
        if not selected.is_cuda:
            raise ValueError("Q16 boundary production operator requires CUDA")
        self.device = selected.alias
        self.dof_count = boundary.mesh.dof_count
        mask = np.zeros(self.dof_count, dtype=np.int32)
        mask[boundary.constrained_dofs] = 1
        prescribed = np.zeros(self.dof_count, dtype=np.float64)
        prescribed[boundary.constrained_dofs] = boundary.prescribed_values
        self._constrained_mask = wp.array(mask, dtype=wp.int32, device=self.device)
        self._prescribed_state = wp.array(prescribed, dtype=DTYPE, device=self.device)

    def _checked(self, name: str, value: Any) -> wp.array:
        if not isinstance(value, wp.array):
            raise TypeError(f"{name} must be a Warp array")
        if not value.device.is_cuda or value.device.alias != self.device:
            raise ValueError(f"{name} must reside on CUDA device {self.device}")
        if value.dtype != DTYPE:
            raise TypeError(f"{name} must use the frozen float64 Warp dtype")
        if value.ndim != 2 or value.shape[0] <= 0 or value.shape[1] != self.dof_count:
            raise ValueError(
                f"{name} must have shape (positive_batch, {self.dof_count})"
            )
        flag = wp.zeros(1, dtype=wp.int32, device=self.device)
        wp.launch(
            _q16_constraint_nonfinite_kernel,
            dim=value.shape,
            inputs=[value],
            outputs=[flag],
            device=self.device,
        )
        wp.synchronize_device(self.device)
        if int(flag.numpy()[0]) != 0:
            raise FloatingPointError(f"{name} contains non-finite values")
        return value

    def _output(self, batch: int) -> wp.array:
        return wp.zeros((batch, self.dof_count), dtype=DTYPE, device=self.device)

    def project_free(self, vector: Any) -> wp.array:
        checked = self._checked("vector", vector)
        return self._project_free_prechecked(checked)

    def _project_free_prechecked(self, vector: wp.array) -> wp.array:
        result = self._output(vector.shape[0])
        wp.launch(
            _q16_project_free_kernel,
            dim=vector.shape,
            inputs=[vector, self._constrained_mask],
            outputs=[result],
            device=self.device,
        )
        return result

    def extract_reaction(self, vector: Any) -> wp.array:
        checked = self._checked("vector", vector)
        return self._extract_reaction_prechecked(checked)

    def _extract_reaction_prechecked(self, vector: wp.array) -> wp.array:
        result = self._output(vector.shape[0])
        wp.launch(
            _q16_extract_reaction_kernel,
            dim=vector.shape,
            inputs=[vector, self._constrained_mask],
            outputs=[result],
            device=self.device,
        )
        return result

    def enforce_state(self, state: Any) -> wp.array:
        checked = self._checked("state", state)
        return self._enforce_state_prechecked(checked)

    def _enforce_state_prechecked(self, state: wp.array) -> wp.array:
        result = self._output(state.shape[0])
        wp.launch(
            _q16_enforce_state_kernel,
            dim=state.shape,
            inputs=[state, self._constrained_mask, self._prescribed_state],
            outputs=[result],
            device=self.device,
        )
        return result

    def require_kinematics(self, state: Any, velocity: Any, acceleration: Any) -> None:
        checked_state = self._checked("state", state)
        checked_velocity = self._checked("velocity", velocity)
        checked_acceleration = self._checked("acceleration", acceleration)
        if not (
            checked_state.shape == checked_velocity.shape == checked_acceleration.shape
        ):
            raise ValueError("state, velocity and acceleration shapes differ")
        flag = wp.zeros(1, dtype=wp.int32, device=self.device)
        wp.launch(
            _q16_constraint_kinematic_violation_kernel,
            dim=checked_state.shape,
            inputs=[
                checked_state,
                checked_velocity,
                checked_acceleration,
                self._constrained_mask,
                self._prescribed_state,
            ],
            outputs=[flag],
            device=self.device,
        )
        wp.synchronize_device(self.device)
        if int(flag.numpy()[0]) != 0:
            raise ValueError("structural kinematics violate the frozen boundary")


__all__ = ["Q16CudaBoundaryConstraints"]
