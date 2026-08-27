"""Moving-root boundary strategy for Q16 wings on a flapping body.

The frozen production owner (:class:`~fluxvortex.q16_boundary_constraints.
Q16BoundaryConstraints`) stays immutable for all time.  This strategy
re-uploads only the device-side prescribed state of the CUDA boundary
operator, each time from the body pose and joint angles, following plan
§7.5's three-layer consistency:

- position: root nodes placed at the joint transform ``R p_ref + t``;
- velocity: root node velocity ``v_I + omega x r_I`` (body + joint rate);
- acceleration: consistent with the above (wired with the Newmark
  predictor reaction extraction in a later U7 checkpoint).

Per plan §7.5 the update belongs to trial composition: it must be applied
between steps, before the structural solver consumes the boundary, and
never retroactively mutates a committed step.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import warp as wp

from fluxvortex.q16_boundary_constraints import Q16BoundaryConstraints
from fluxvortex.warp_fsi.config import DTYPE as _CUDA_DTYPE

_FLOAT64 = np.dtype(np.float64)
_INT64 = np.dtype(np.int64)
_ORTHO_TOLERANCE = 1.0e-9


def _frozen_vector3(name: str, value: Any) -> np.ndarray:
    if type(value) is not np.ndarray:
        raise TypeError(f"{name} must be an exact numpy.ndarray")
    if value.dtype != _FLOAT64:
        raise TypeError(f"{name} must use float64")
    if value.shape != (3,):
        raise ValueError(f"{name} must have shape (3,)")
    if not bool(np.isfinite(value).all()):
        raise ValueError(f"{name} must contain only finite values")
    return np.array(value, dtype=np.float64, copy=True, order="C")


@dataclass(frozen=True, eq=False)
class MovingRootTransform:
    """The rigid transform applied to the root boundary at one time step."""

    position_I: np.ndarray      # (3,) root frame origin in inertial frame
    rotation_IB: np.ndarray     # (3,3) rotation matrix
    velocity_I: np.ndarray      # (3,) root frame origin velocity
    angular_velocity_I: np.ndarray  # (3,) body + joint rate in inertial frame

    def __post_init__(self) -> None:
        rotation = self.rotation_IB
        if type(rotation) is not np.ndarray:
            raise TypeError("rotation_IB must be an exact numpy.ndarray")
        if rotation.dtype != _FLOAT64:
            raise TypeError("rotation_IB must use float64")
        if rotation.shape != (3, 3):
            raise ValueError("rotation_IB must have shape (3, 3)")
        if not bool(np.isfinite(rotation).all()):
            raise ValueError("rotation_IB must contain only finite values")
        rotation = np.array(rotation, dtype=np.float64, copy=True, order="C")
        gram = rotation @ rotation.T
        if not bool(
            np.allclose(
                gram, np.eye(3, dtype=np.float64), rtol=0.0, atol=_ORTHO_TOLERANCE
            )
        ):
            raise ValueError("rotation_IB must be orthonormal (R R^T = I)")
        if abs(float(np.linalg.det(rotation)) - 1.0) > _ORTHO_TOLERANCE:
            raise ValueError("rotation_IB must be a proper rotation (det = +1)")
        object.__setattr__(self, "rotation_IB", rotation)
        object.__setattr__(
            self, "position_I", _frozen_vector3("position_I", self.position_I)
        )
        object.__setattr__(
            self, "velocity_I", _frozen_vector3("velocity_I", self.velocity_I)
        )
        object.__setattr__(
            self,
            "angular_velocity_I",
            _frozen_vector3("angular_velocity_I", self.angular_velocity_I),
        )


class MovingRootBoundary:
    """Updates Q16 boundary prescribed values from body/joint state.

    The original boundary is created by ``make_clamped_q16_nodes`` (fixed
    for all time).  This class replaces the prescribed values of the root
    node subset each step with the rigidly transformed reference positions,
    leaving every other constrained node at its frozen reference value.
    Passing the boundary mesh's own ``reference_rows`` makes the identity
    transform reproduce the frozen boundary exactly.
    """

    def __init__(
        self,
        reference_boundary: Q16BoundaryConstraints,
        root_node_ids: np.ndarray,
        mesh_reference_rows: np.ndarray,
    ) -> None:
        """Store the reference state for transformation.

        reference_boundary: the Q16BoundaryConstraints from
            make_clamped_q16_nodes (the immutable frozen owner).
        root_node_ids: (n_root_nodes,) sorted global node IDs of the root
            edge; they must be a subset of the constrained nodes.
        mesh_reference_rows: (n_nodes, 6) the full mesh reference state
            [x, y, z, gx, gy, gz] the transform acts on.
        """

        if type(reference_boundary) is not Q16BoundaryConstraints:
            raise TypeError(
                "reference_boundary must be an exact Q16BoundaryConstraints"
            )
        mesh = reference_boundary.mesh
        if type(root_node_ids) is not np.ndarray:
            raise TypeError("root_node_ids must be an exact numpy.ndarray")
        if root_node_ids.dtype != _INT64:
            raise TypeError("root_node_ids must use int64")
        if root_node_ids.ndim != 1 or root_node_ids.size == 0:
            raise ValueError("root_node_ids must be a non-empty vector")
        if not root_node_ids.flags.c_contiguous:
            raise ValueError("root_node_ids must be C-contiguous")
        if bool((root_node_ids < 0).any()) or bool(
            (root_node_ids >= mesh.node_count).any()
        ):
            raise ValueError("root_node_ids contains a node outside the global range")
        if root_node_ids.size > 1 and not bool(
            np.all(root_node_ids[1:] > root_node_ids[:-1])
        ):
            raise ValueError("root_node_ids must be sorted and unique")
        if not bool(np.isin(root_node_ids, reference_boundary.constrained_nodes).all()):
            raise ValueError(
                "every root node must be owned by the reference boundary"
            )
        if type(mesh_reference_rows) is not np.ndarray:
            raise TypeError("mesh_reference_rows must be an exact numpy.ndarray")
        if mesh_reference_rows.dtype != _FLOAT64:
            raise TypeError("mesh_reference_rows must use float64")
        if mesh_reference_rows.shape != (mesh.node_count, 6):
            raise ValueError(
                f"mesh_reference_rows must have shape ({mesh.node_count}, 6)"
            )
        if not bool(np.isfinite(mesh_reference_rows).all()):
            raise ValueError("mesh_reference_rows must contain only finite values")
        self._boundary = reference_boundary
        self._root_ids = np.array(root_node_ids, dtype=np.int64, copy=True)
        self._ref_rows = np.array(mesh_reference_rows, dtype=np.float64, copy=True)
        self._root_dofs = np.ascontiguousarray(
            (self._root_ids[:, None] * 6 + np.arange(6, dtype=np.int64)[None, :])
            .reshape(-1),
            dtype=np.int64,
        )

    @property
    def root_node_ids(self) -> np.ndarray:
        """Read-only copy of the root edge node IDs."""

        return self._root_ids.copy()

    def prescribed_rows(self, transform: MovingRootTransform) -> np.ndarray:
        """Return the (n_root, 6) root rows under one transform.

        Positions follow ``p_new = R @ p_ref + t`` and directors rotate
        rigidly, ``d_new = R @ d_ref``.
        """

        if type(transform) is not MovingRootTransform:
            raise TypeError("transform must be an exact MovingRootTransform")
        ref = self._ref_rows[self._root_ids]
        positions = (transform.rotation_IB @ ref[:, :3].T).T + transform.position_I
        directors = transform.rotation_IB @ ref[:, 3:].T
        return np.ascontiguousarray(
            np.concatenate([positions, directors.T], axis=1), dtype=np.float64
        )

    def root_velocities(self, transform: MovingRootTransform) -> np.ndarray:
        """Return the (n_root, 3) root node velocities (plan §7.5 layer 2).

        ``v = v_I + omega x (R @ p_ref)``: the cross product acts on the
        rotated reference position relative to the moving frame origin (the
        analytic time derivative of ``R(theta(t)) p_ref + position_I(t)``),
        so a pure translation contributes no spurious rotation term.
        """

        if type(transform) is not MovingRootTransform:
            raise TypeError("transform must be an exact MovingRootTransform")
        ref = self._ref_rows[self._root_ids]
        relative = (transform.rotation_IB @ ref[:, :3].T).T
        return np.ascontiguousarray(
            transform.velocity_I[None, :]
            + np.cross(transform.angular_velocity_I[None, :], relative),
            dtype=np.float64,
        )

    def update(self, transform: MovingRootTransform, boundary_operator: Any) -> None:
        """Update the boundary prescribed values under the given transform.

        boundary_operator: the Q16CudaBoundaryConstraints that the solver
        uses; its device-side prescribed state is replaced with the frozen
        reference values, the root node blocks transformed.
        """

        if type(transform) is not MovingRootTransform:
            raise TypeError("transform must be an exact MovingRootTransform")
        dof_count = self._boundary.mesh.dof_count
        if int(getattr(boundary_operator, "dof_count", -1)) != dof_count:
            raise ValueError(
                "boundary_operator does not own the reference boundary DOF range"
            )
        full_state = np.zeros(dof_count, dtype=np.float64)
        full_state[self._boundary.constrained_dofs] = self._boundary.prescribed_values
        full_state[self._root_dofs] = self.prescribed_rows(transform).reshape(-1)
        new_state = wp.array(
            np.ascontiguousarray(full_state, dtype=_CUDA_DTYPE),
            dtype=_CUDA_DTYPE,
            device=boundary_operator.device,
        )
        boundary_operator.update_prescribed_values(new_state)

    def constraint_reaction(self) -> np.ndarray:
        """Return the constraint reaction forces (to be wired in U7+).

        The reaction = internal force at the boundary nodes, which the
        structural solver already computes.  This will be extracted and
        routed back to the body/joint in the full free-flight coupling.
        """
        raise NotImplementedError("reaction extraction is U7+ work")


__all__ = ["MovingRootBoundary", "MovingRootTransform"]
