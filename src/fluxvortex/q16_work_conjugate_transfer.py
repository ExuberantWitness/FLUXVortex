"""Independent Q16 surface interpolation and transpose load-transfer oracle.

The fixed map uses the same Q16 interpolation for structural surface motion and
for the transpose aerodynamic load map.  This makes virtual work exact up to
floating-point accumulation order and retains the director moment arm for
loads acting away from the shell midsurface.

This NumPy module is an independent small-problem oracle.  Production FSI uses
the CUDA implementation in :mod:`fluxvortex.warp_fsi.kernels_q16_transfer`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .q16_ancf_mesh import Q16Mesh
from .q16_ancf_shell import Q16_NODE_COUNT, q16_shape

_FLOAT64 = np.dtype(np.float64)
_INT64 = np.dtype(np.int64)
MAX_Q16_SURFACE_POINT_COUNT = 1_000_000


def _readonly(value: np.ndarray, dtype: np.dtype) -> np.ndarray:
    contiguous = np.ascontiguousarray(value, dtype=dtype)
    frozen = np.frombuffer(contiguous.tobytes(order="C"), dtype=dtype)
    return frozen.reshape(contiguous.shape)


def _exact_float64(name: str, value: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    if type(value) is not np.ndarray:
        raise TypeError(f"{name} must be an exact numpy.ndarray")
    if value.dtype != _FLOAT64:
        raise TypeError(f"{name} must use float64")
    if value.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if not value.flags.c_contiguous:
        raise ValueError(f"{name} must be C-contiguous")
    if not bool(np.isfinite(value).all()):
        raise ValueError(f"{name} must contain only finite values")
    return _readonly(value, _FLOAT64)


def _algebraic_point_key(
    mesh: Q16Mesh, element: int, shape: np.ndarray, zeta: float
) -> tuple[tuple[int, str, str], ...]:
    """Canonicalize one sparse global interpolation row without dense storage."""

    entries: list[tuple[int, str, str]] = []
    for local_node, global_node in enumerate(mesh.connectivity[element]):
        position_weight = float(shape[local_node])
        director_weight = zeta * position_weight
        if position_weight == 0.0 and director_weight == 0.0:
            continue
        entries.append(
            (
                int(global_node),
                position_weight.hex(),
                director_weight.hex(),
            )
        )
    entries.sort(key=lambda item: item[0])
    return tuple(entries)


@dataclass(frozen=True, slots=True, eq=False)
class Q16ForceMomentBalance:
    aerodynamic_force: np.ndarray
    generalized_force: np.ndarray
    aerodynamic_moment: np.ndarray
    generalized_moment: np.ndarray


@dataclass(frozen=True, slots=True, init=False, eq=False)
class Q16SurfaceTransferMap:
    """Immutable mapping from surface points to one shared-node Q16 mesh."""

    mesh: Q16Mesh
    element_count: int
    element_indices: np.ndarray
    parametric_coordinates: np.ndarray
    shape_values: np.ndarray
    point_count: int

    def __init__(
        self,
        *,
        mesh: Q16Mesh,
        element_indices: np.ndarray,
        parametric_coordinates: np.ndarray,
    ) -> None:
        if type(mesh) is not Q16Mesh:
            raise TypeError("mesh must be an exact Q16Mesh")
        if type(element_indices) is not np.ndarray:
            raise TypeError("element_indices must be an exact numpy.ndarray")
        if element_indices.dtype != _INT64:
            raise TypeError("element_indices must use int64")
        if element_indices.ndim != 1 or element_indices.size == 0:
            raise ValueError("element_indices must be a non-empty vector")
        if element_indices.size > MAX_Q16_SURFACE_POINT_COUNT:
            raise ValueError("surface point count exceeds the fixed cap")
        if not element_indices.flags.c_contiguous:
            raise ValueError("element_indices must be C-contiguous")
        point_count = int(element_indices.size)
        coordinates = _exact_float64(
            "parametric_coordinates",
            parametric_coordinates,
            (point_count, 3),
        )
        if bool((coordinates < -1.0).any()) or bool((coordinates > 1.0).any()):
            raise ValueError("parametric coordinates must lie in [-1, 1]")
        if bool((element_indices < 0).any()) or bool(
            (element_indices >= mesh.element_count).any()
        ):
            raise ValueError("element index is outside the declared element range")

        shapes = np.empty((point_count, Q16_NODE_COUNT), dtype=np.float64)
        for point in range(point_count):
            shapes[point] = q16_shape(
                float(coordinates[point, 0]), float(coordinates[point, 1])
            )[0]
        # Independent aerodynamic mechanisms may legitimately apply separate
        # forces at one physical point (for example two adjacent vortex legs).
        # Repeating the row through the same element owner preserves exact
        # linear superposition.  Registering the same algebraic row through
        # different Q16 elements is ambiguous and remains forbidden.
        algebraic_point_owners: dict[tuple[tuple[int, str, str], ...], int] = {}
        for point in range(point_count):
            element = int(element_indices[point])
            key = _algebraic_point_key(
                mesh,
                element,
                shapes[point],
                float(coordinates[point, 2]),
            )
            previous_owner = algebraic_point_owners.get(key)
            if previous_owner is not None and previous_owner != element:
                raise ValueError(
                    "surface map registers one algebraic point through "
                    "multiple element owners"
                )
            algebraic_point_owners[key] = element

        object.__setattr__(self, "mesh", mesh)
        object.__setattr__(self, "element_count", mesh.element_count)
        object.__setattr__(self, "element_indices", _readonly(element_indices, _INT64))
        object.__setattr__(self, "parametric_coordinates", coordinates)
        object.__setattr__(self, "shape_values", _readonly(shapes, _FLOAT64))
        object.__setattr__(self, "point_count", point_count)

    @property
    def structural_dof_count(self) -> int:
        return self.mesh.dof_count

    def _state(self, name: str, value: np.ndarray) -> np.ndarray:
        return self.mesh._global_state(name, value)

    def _point_vectors(self, name: str, value: np.ndarray) -> np.ndarray:
        return _exact_float64(name, value, (self.point_count, 3))

    def interpolate(self, state: np.ndarray) -> np.ndarray:
        """Map one structural state/velocity vector to all surface points."""

        rows = self._state("state", state)
        output = np.zeros((self.point_count, 3), dtype=np.float64)
        for point in range(self.point_count):
            element = int(self.element_indices[point])
            zeta = self.parametric_coordinates[point, 2]
            shape = self.shape_values[point]
            element_rows = rows[self.mesh.connectivity[element]]
            output[point] = shape @ (element_rows[:, :3] + zeta * element_rows[:, 3:])
        return _readonly(output, _FLOAT64)

    def transpose(self, aerodynamic_force: np.ndarray) -> np.ndarray:
        """Apply the exact transpose surface map to aerodynamic point forces."""

        forces = self._point_vectors("aerodynamic_force", aerodynamic_force)
        generalized = np.zeros((self.mesh.node_count, 6), dtype=np.float64)
        for point in range(self.point_count):
            element = int(self.element_indices[point])
            zeta = self.parametric_coordinates[point, 2]
            for node in range(Q16_NODE_COUNT):
                global_node = int(self.mesh.connectivity[element, node])
                weight = self.shape_values[point, node]
                generalized[global_node, :3] += weight * forces[point]
                generalized[global_node, 3:] += zeta * weight * forces[point]
        return _readonly(generalized.reshape(self.structural_dof_count), _FLOAT64)

    def force_moment_balance(
        self, state: np.ndarray, aerodynamic_force: np.ndarray
    ) -> Q16ForceMomentBalance:
        """Return independently accumulated physical and generalized balances."""

        rows = self._state("state", state)
        forces = self._point_vectors("aerodynamic_force", aerodynamic_force)
        points = self.interpolate(state)
        generalized = self.transpose(forces).reshape(self.mesh.node_count, 6)
        aerodynamic_total = np.sum(forces, axis=0)
        generalized_total = np.sum(generalized[:, :3], axis=0)
        aerodynamic_moment = np.sum(np.cross(points, forces), axis=0)
        generalized_moment = np.sum(
            np.cross(rows[:, :3], generalized[:, :3])
            + np.cross(rows[:, 3:], generalized[:, 3:]),
            axis=0,
        )
        return Q16ForceMomentBalance(
            aerodynamic_force=_readonly(aerodynamic_total, _FLOAT64),
            generalized_force=_readonly(generalized_total, _FLOAT64),
            aerodynamic_moment=_readonly(aerodynamic_moment, _FLOAT64),
            generalized_moment=_readonly(generalized_moment, _FLOAT64),
        )


__all__ = [
    "MAX_Q16_SURFACE_POINT_COUNT",
    "Q16ForceMomentBalance",
    "Q16SurfaceTransferMap",
]
