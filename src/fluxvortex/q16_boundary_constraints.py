"""Immutable essential-boundary ownership for shared-node Q16 meshes."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .q16_ancf_mesh import Q16Mesh

_FLOAT64 = np.dtype(np.float64)
_INT64 = np.dtype(np.int64)


def _readonly(value: np.ndarray, dtype: np.dtype) -> np.ndarray:
    contiguous = np.ascontiguousarray(value, dtype=dtype)
    frozen = np.frombuffer(contiguous.tobytes(order="C"), dtype=dtype)
    return frozen.reshape(contiguous.shape)


@dataclass(frozen=True, slots=True, init=False, eq=False)
class Q16BoundaryConstraints:
    """One immutable owner for prescribed global Q16 degrees of freedom."""

    mesh: Q16Mesh
    constrained_dofs: np.ndarray
    prescribed_values: np.ndarray
    constrained_nodes: np.ndarray
    free_dofs: np.ndarray
    constrained_dof_count: int
    free_dof_count: int

    def __init__(
        self,
        mesh: Q16Mesh,
        constrained_dofs: np.ndarray,
        prescribed_values: np.ndarray,
    ) -> None:
        if type(mesh) is not Q16Mesh:
            raise TypeError("mesh must be an exact Q16Mesh")
        if type(constrained_dofs) is not np.ndarray:
            raise TypeError("constrained_dofs must be an exact numpy.ndarray")
        if constrained_dofs.dtype != _INT64:
            raise TypeError("constrained_dofs must use int64")
        if constrained_dofs.ndim != 1 or constrained_dofs.size == 0:
            raise ValueError("constrained_dofs must be a non-empty vector")
        if not constrained_dofs.flags.c_contiguous:
            raise ValueError("constrained_dofs must be C-contiguous")
        if bool((constrained_dofs < 0).any()) or bool(
            (constrained_dofs >= mesh.dof_count).any()
        ):
            raise ValueError("constrained_dofs contains a DOF outside the global range")
        if constrained_dofs.size > 1 and not bool(
            np.all(constrained_dofs[1:] > constrained_dofs[:-1])
        ):
            raise ValueError("constrained_dofs must be sorted and unique")
        if type(prescribed_values) is not np.ndarray:
            raise TypeError("prescribed_values must be an exact numpy.ndarray")
        if prescribed_values.dtype != _FLOAT64:
            raise TypeError("prescribed_values must use float64")
        if prescribed_values.shape != constrained_dofs.shape:
            raise ValueError("prescribed_values shape differs from constrained_dofs")
        if not prescribed_values.flags.c_contiguous:
            raise ValueError("prescribed_values must be C-contiguous")
        if not bool(np.isfinite(prescribed_values).all()):
            raise ValueError("prescribed_values must contain only finite values")

        frozen_dofs = _readonly(constrained_dofs, _INT64)
        frozen_values = _readonly(prescribed_values, _FLOAT64)
        all_dofs = np.arange(mesh.dof_count, dtype=np.int64)
        free_dofs = _readonly(np.setdiff1d(all_dofs, frozen_dofs), _INT64)
        constrained_nodes = _readonly(np.unique(frozen_dofs // 6), _INT64)
        object.__setattr__(self, "mesh", mesh)
        object.__setattr__(self, "constrained_dofs", frozen_dofs)
        object.__setattr__(self, "prescribed_values", frozen_values)
        object.__setattr__(self, "constrained_nodes", constrained_nodes)
        object.__setattr__(self, "free_dofs", free_dofs)
        object.__setattr__(self, "constrained_dof_count", int(frozen_dofs.size))
        object.__setattr__(self, "free_dof_count", int(free_dofs.size))

    def _vector(self, name: str, value: np.ndarray) -> np.ndarray:
        if type(value) is not np.ndarray:
            raise TypeError(f"{name} must be an exact numpy.ndarray")
        if value.dtype != _FLOAT64:
            raise TypeError(f"{name} must use float64")
        if value.shape != (self.mesh.dof_count,):
            raise ValueError(f"{name} must have shape ({self.mesh.dof_count},)")
        if not value.flags.c_contiguous:
            raise ValueError(f"{name} must be C-contiguous")
        if not bool(np.isfinite(value).all()):
            raise ValueError(f"{name} must contain only finite values")
        return value

    def enforce_state(self, state: np.ndarray) -> np.ndarray:
        """Return a frozen state with every prescribed value restored exactly."""

        checked = self._vector("state", state)
        result = checked.copy()
        result[self.constrained_dofs] = self.prescribed_values
        return _readonly(result, _FLOAT64)

    def project_free(self, vector: np.ndarray) -> np.ndarray:
        """Project a vector into the admissible free-DOF subspace."""

        checked = self._vector("vector", vector)
        result = checked.copy()
        result[self.constrained_dofs] = 0.0
        return _readonly(result, _FLOAT64)

    def extract_reaction(self, vector: np.ndarray) -> np.ndarray:
        """Return the complementary constrained-DOF component of a vector."""

        checked = self._vector("vector", vector)
        result = np.zeros(self.mesh.dof_count, dtype=np.float64)
        result[self.constrained_dofs] = checked[self.constrained_dofs]
        return _readonly(result, _FLOAT64)


def make_clamped_q16_nodes(
    mesh: Q16Mesh, constrained_nodes: np.ndarray
) -> Q16BoundaryConstraints:
    """Clamp an explicit, stable set of global Q16 node identifiers."""

    if type(mesh) is not Q16Mesh:
        raise TypeError("mesh must be an exact Q16Mesh")
    if type(constrained_nodes) is not np.ndarray:
        raise TypeError("constrained_nodes must be an exact numpy.ndarray")
    if constrained_nodes.dtype != _INT64:
        raise TypeError("constrained_nodes must use int64")
    if constrained_nodes.ndim != 1 or constrained_nodes.size == 0:
        raise ValueError("constrained_nodes must be a non-empty vector")
    if not constrained_nodes.flags.c_contiguous:
        raise ValueError("constrained_nodes must be C-contiguous")
    if bool((constrained_nodes < 0).any()) or bool(
        (constrained_nodes >= mesh.node_count).any()
    ):
        raise ValueError("constrained_nodes contains a node outside the global range")
    if constrained_nodes.size > 1 and not bool(
        np.all(constrained_nodes[1:] > constrained_nodes[:-1])
    ):
        raise ValueError("constrained_nodes must be sorted and unique")
    local_dofs = np.arange(6, dtype=np.int64)
    constrained_dofs = np.ascontiguousarray(
        (constrained_nodes[:, None] * 6 + local_dofs[None, :]).reshape(-1),
        dtype=np.int64,
    )
    prescribed_values = np.ascontiguousarray(
        mesh.reference_state[constrained_dofs], dtype=np.float64
    )
    return Q16BoundaryConstraints(mesh, constrained_dofs, prescribed_values)


def make_clamped_span_root(
    mesh: Q16Mesh, *, span_axis: int = 1
) -> Q16BoundaryConstraints:
    """Infer a rectangular-mesh root, then delegate to explicit node ownership.

    Production geometries that may be swept or rigidly rotated should call
    :func:`make_clamped_q16_nodes` with their topology-owned root node IDs.
    """

    if type(mesh) is not Q16Mesh:
        raise TypeError("mesh must be an exact Q16Mesh")
    if isinstance(span_axis, bool) or type(span_axis) is not int:
        raise TypeError("span_axis must be an exact int")
    if span_axis not in (0, 1, 2):
        raise ValueError("span_axis must be 0, 1, or 2")
    coordinates = mesh.reference_rows[:, span_axis]
    root_coordinate = float(np.min(coordinates))
    if not math.isfinite(root_coordinate):
        raise ValueError("root coordinate must be finite")
    scale = max(float(np.max(np.abs(coordinates))), 1.0)
    tolerance = 256.0 * np.finfo(np.float64).eps * scale
    nodes = np.flatnonzero(np.abs(coordinates - root_coordinate) <= tolerance)
    if nodes.size == 0:
        raise ValueError("Q16 mesh has no span-root nodes")
    return make_clamped_q16_nodes(mesh, np.ascontiguousarray(nodes, dtype=np.int64))


__all__ = [
    "Q16BoundaryConstraints",
    "make_clamped_q16_nodes",
    "make_clamped_span_root",
]
