"""Shared-node mesh ownership for fixed Q16 ANCF continuum elements."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .q16_ancf_continuum import Q16ContinuumElement
from .q16_ans_eas_continuum import Q16MITC16EASContinuumElement
from .q16_ancf_shell import (
    Q16_DOF_PER_ELEMENT,
    Q16_DOF_PER_NODE,
    Q16_NODE_COUNT,
    Q16_PARAMETRIC_NODES_1D,
    Q16ReferenceElement,
)

_FLOAT64 = np.dtype(np.float64)
_INT64 = np.dtype(np.int64)
MAX_Q16_MESH_NODE_COUNT = 1_000_000
MAX_Q16_MESH_ELEMENT_COUNT = 100_000


def _readonly(value: np.ndarray, dtype: np.dtype) -> np.ndarray:
    contiguous = np.ascontiguousarray(value, dtype=dtype)
    frozen = np.frombuffer(contiguous.tobytes(order="C"), dtype=dtype)
    return frozen.reshape(contiguous.shape)


def _positive_scalar(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise TypeError(f"{name} must be a real scalar")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _poisson_scalar(value: float) -> float:
    if isinstance(value, bool) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise TypeError("poisson_ratio must be a real scalar")
    result = float(value)
    if not math.isfinite(result) or not -1.0 < result < 0.5:
        raise ValueError("poisson_ratio must be finite and lie in (-1, 0.5)")
    return result


def _macro_property_vector(name: str, value: np.ndarray) -> np.ndarray:
    if type(value) is not np.ndarray:
        raise TypeError(f"{name} must be an exact numpy.ndarray")
    if value.dtype != _FLOAT64:
        raise TypeError(f"{name} must use float64")
    if value.ndim != 1 or value.size == 0:
        raise ValueError(f"{name} must be a non-empty vector")
    if not value.flags.c_contiguous:
        raise ValueError(f"{name} must be C-contiguous")
    return _readonly(value, _FLOAT64)


@dataclass(frozen=True, slots=True, init=False, eq=False)
class Q16MacroPropertyField:
    """Homogenized stiffness and mass controls owned per Q16 macro element.

    The field represents the early-design rigid/flexible and mass distribution;
    it does not prescribe explicit ribs, spars, laminates or other detailed
    substructure inside a macro element.
    """

    young_modulus: np.ndarray
    poisson_ratio: np.ndarray
    density: np.ndarray
    element_count: int

    def __init__(
        self,
        *,
        young_modulus: np.ndarray,
        poisson_ratio: np.ndarray,
        density: np.ndarray,
    ) -> None:
        young = _macro_property_vector("young_modulus", young_modulus)
        poisson = _macro_property_vector("poisson_ratio", poisson_ratio)
        rho = _macro_property_vector("density", density)
        if not (young.shape == poisson.shape == rho.shape):
            raise ValueError(
                "macro property vectors must have the same non-empty shape"
            )
        if not bool(np.isfinite(young).all()) or not bool((young > 0.0).all()):
            raise ValueError("young_modulus must be finite and positive")
        if not bool(np.isfinite(rho).all()) or not bool((rho > 0.0).all()):
            raise ValueError("density must be finite and positive")
        if not bool(np.isfinite(poisson).all()) or not bool(
            ((poisson > -1.0) & (poisson < 0.5)).all()
        ):
            raise ValueError("poisson_ratio must be finite and lie in (-1, 0.5)")
        object.__setattr__(self, "young_modulus", young)
        object.__setattr__(self, "poisson_ratio", poisson)
        object.__setattr__(self, "density", rho)
        object.__setattr__(self, "element_count", int(young.size))


@dataclass(frozen=True, slots=True, init=False, eq=False)
class Q16Mesh:
    """Immutable shared global-node ownership and local Q16 connectivity."""

    reference_rows: np.ndarray
    reference_state: np.ndarray
    connectivity: np.ndarray
    node_count: int
    element_count: int
    dof_count: int
    reference_elements: tuple[Q16ReferenceElement, ...]

    def __init__(self, *, reference_rows: np.ndarray, connectivity: np.ndarray) -> None:
        if type(reference_rows) is not np.ndarray:
            raise TypeError("reference_rows must be an exact numpy.ndarray")
        if reference_rows.dtype != _FLOAT64:
            raise TypeError("reference_rows must use float64")
        if (
            reference_rows.ndim != 2
            or reference_rows.shape[1] != Q16_DOF_PER_NODE
            or reference_rows.shape[0] == 0
        ):
            raise ValueError("reference_rows must have non-empty shape (node_count, 6)")
        if reference_rows.shape[0] > MAX_Q16_MESH_NODE_COUNT:
            raise ValueError("Q16 mesh node count exceeds the fixed cap")
        if not reference_rows.flags.c_contiguous:
            raise ValueError("reference_rows must be C-contiguous")
        if not bool(np.isfinite(reference_rows).all()):
            raise ValueError("reference_rows must contain only finite values")
        if type(connectivity) is not np.ndarray:
            raise TypeError("connectivity must be an exact numpy.ndarray")
        if connectivity.dtype != _INT64:
            raise TypeError("connectivity must use int64")
        if (
            connectivity.ndim != 2
            or connectivity.shape[1] != Q16_NODE_COUNT
            or connectivity.shape[0] == 0
        ):
            raise ValueError(
                "connectivity must have non-empty shape (element_count, 16)"
            )
        if connectivity.shape[0] > MAX_Q16_MESH_ELEMENT_COUNT:
            raise ValueError("Q16 mesh element count exceeds the fixed cap")
        if not connectivity.flags.c_contiguous:
            raise ValueError("connectivity must be C-contiguous")
        node_count = int(reference_rows.shape[0])
        if bool((connectivity < 0).any()) or bool((connectivity >= node_count).any()):
            raise ValueError("connectivity contains a node outside the global range")
        for element, row in enumerate(connectivity):
            if np.unique(row).size != Q16_NODE_COUNT:
                raise ValueError(f"element {element} contains duplicate local nodes")
        if np.unique(connectivity).size != node_count:
            raise ValueError(
                "every global Q16 node must be owned by at least one element"
            )

        rows = _readonly(reference_rows, _FLOAT64)
        conn = _readonly(connectivity, _INT64)
        elements = tuple(Q16ReferenceElement(rows[row]) for row in conn)
        object.__setattr__(self, "reference_rows", rows)
        object.__setattr__(
            self,
            "reference_state",
            rows.reshape(node_count * Q16_DOF_PER_NODE),
        )
        object.__setattr__(self, "connectivity", conn)
        object.__setattr__(self, "node_count", node_count)
        object.__setattr__(self, "element_count", int(conn.shape[0]))
        object.__setattr__(self, "dof_count", node_count * Q16_DOF_PER_NODE)
        object.__setattr__(self, "reference_elements", elements)

    def _global_state(self, name: str, value: np.ndarray) -> np.ndarray:
        if type(value) is not np.ndarray:
            raise TypeError(f"{name} must be an exact numpy.ndarray")
        if value.dtype != _FLOAT64:
            raise TypeError(f"{name} must use float64")
        if value.shape != (self.dof_count,):
            raise ValueError(f"{name} must have shape ({self.dof_count},)")
        if not value.flags.c_contiguous:
            raise ValueError(f"{name} must be C-contiguous")
        if not bool(np.isfinite(value).all()):
            raise ValueError(f"{name} must contain only finite values")
        return value.reshape(self.node_count, Q16_DOF_PER_NODE)

    def gather(self, global_state: np.ndarray) -> np.ndarray:
        """Gather one global state to deterministic element-local 96-DOF rows."""

        rows = self._global_state("global_state", global_state)
        local = rows[self.connectivity].reshape(self.element_count, Q16_DOF_PER_ELEMENT)
        return _readonly(local, _FLOAT64)

    def assemble(self, element_vectors: np.ndarray) -> np.ndarray:
        """Sum element-local vectors into the unique global-node state."""

        if type(element_vectors) is not np.ndarray:
            raise TypeError("element_vectors must be an exact numpy.ndarray")
        if element_vectors.dtype != _FLOAT64:
            raise TypeError("element_vectors must use float64")
        if element_vectors.shape != (self.element_count, Q16_DOF_PER_ELEMENT):
            raise ValueError("element_vectors must have shape (element_count, 96)")
        if not element_vectors.flags.c_contiguous:
            raise ValueError("element_vectors must be C-contiguous")
        if not bool(np.isfinite(element_vectors).all()):
            raise ValueError("element_vectors must contain only finite values")
        local = element_vectors.reshape(self.element_count, Q16_NODE_COUNT, 6)
        global_rows = np.zeros((self.node_count, 6), dtype=np.float64)
        for global_node in range(self.node_count):
            for element in range(self.element_count):
                for local_node in range(Q16_NODE_COUNT):
                    if self.connectivity[element, local_node] == global_node:
                        global_rows[global_node] += local[element, local_node]
        return _readonly(global_rows.reshape(self.dof_count), _FLOAT64)


@dataclass(frozen=True, slots=True, init=False, eq=False)
class Q16ContinuumMesh:
    """Shared-node CPU oracle assembled from verified Q16 continuum elements."""

    mesh: Q16Mesh
    elements: tuple[Q16ContinuumElement, ...]
    young_modulus: float
    poisson_ratio: float
    density: float
    total_reference_volume: float

    def __init__(
        self,
        mesh: Q16Mesh,
        *,
        young_modulus: float,
        poisson_ratio: float,
        density: float,
    ) -> None:
        if type(mesh) is not Q16Mesh:
            raise TypeError("mesh must be an exact Q16Mesh")
        elements = tuple(
            Q16ContinuumElement(
                reference,
                young_modulus=young_modulus,
                poisson_ratio=poisson_ratio,
                density=density,
            )
            for reference in mesh.reference_elements
        )
        object.__setattr__(self, "mesh", mesh)
        object.__setattr__(self, "elements", elements)
        object.__setattr__(self, "young_modulus", elements[0].young_modulus)
        object.__setattr__(self, "poisson_ratio", elements[0].poisson_ratio)
        object.__setattr__(self, "density", elements[0].density)
        object.__setattr__(
            self,
            "total_reference_volume",
            float(sum(element.reference_volume for element in elements)),
        )

    def strain_energy(self, state: np.ndarray) -> float:
        local = self.mesh.gather(state)
        return float(
            sum(
                element.strain_energy(element_state)
                for element, element_state in zip(self.elements, local, strict=True)
            )
        )

    def internal_force(self, state: np.ndarray) -> np.ndarray:
        local = self.mesh.gather(state)
        element_force = np.ascontiguousarray(
            np.stack(
                [
                    element.internal_force(element_state)
                    for element, element_state in zip(self.elements, local, strict=True)
                ]
            )
        )
        return self.mesh.assemble(element_force)

    def tangent_action(self, state: np.ndarray, direction: np.ndarray) -> np.ndarray:
        local_state = self.mesh.gather(state)
        local_direction = self.mesh.gather(direction)
        element_action = np.ascontiguousarray(
            np.stack(
                [
                    element.tangent_action(element_state, element_direction)
                    for element, element_state, element_direction in zip(
                        self.elements,
                        local_state,
                        local_direction,
                        strict=True,
                    )
                ]
            )
        )
        return self.mesh.assemble(element_action)

    def mass_action(self, acceleration: np.ndarray) -> np.ndarray:
        local = self.mesh.gather(acceleration)
        element_action = np.ascontiguousarray(
            np.stack(
                [
                    element.mass_action(element_acceleration)
                    for element, element_acceleration in zip(
                        self.elements, local, strict=True
                    )
                ]
            )
        )
        return self.mesh.assemble(element_action)


@dataclass(frozen=True, slots=True, init=False, eq=False)
class Q16MITC16EASMesh:
    """Shared-node mesh using the projected/condensed Q16 structural operator."""

    mesh: Q16Mesh
    elements: tuple[Q16MITC16EASContinuumElement, ...]
    mass_elements: tuple[Q16ContinuumElement, ...]
    macro_properties: Q16MacroPropertyField
    young_modulus: float
    poisson_ratio: float
    density: float
    total_reference_volume: float
    total_reference_mass: float

    def __init__(
        self,
        mesh: Q16Mesh,
        *,
        young_modulus: float | None = None,
        poisson_ratio: float | None = None,
        density: float | None = None,
        macro_properties: Q16MacroPropertyField | None = None,
    ) -> None:
        if type(mesh) is not Q16Mesh:
            raise TypeError("mesh must be an exact Q16Mesh")
        supplied_scalars = (young_modulus, poisson_ratio, density)
        if macro_properties is None:
            if any(value is None for value in supplied_scalars):
                raise ValueError("provide either uniform scalars or macro_properties")
            young = _positive_scalar("young_modulus", young_modulus)
            poisson = _poisson_scalar(poisson_ratio)
            rho = _positive_scalar("density", density)
            properties = Q16MacroPropertyField(
                young_modulus=np.full(mesh.element_count, young, dtype=np.float64),
                poisson_ratio=np.full(mesh.element_count, poisson, dtype=np.float64),
                density=np.full(mesh.element_count, rho, dtype=np.float64),
            )
        else:
            if type(macro_properties) is not Q16MacroPropertyField:
                raise TypeError(
                    "macro_properties must be an exact Q16MacroPropertyField"
                )
            if any(value is not None for value in supplied_scalars):
                raise ValueError(
                    "provide either uniform scalars or macro_properties, not both"
                )
            if macro_properties.element_count != mesh.element_count:
                raise ValueError("macro property count differs from mesh element count")
            properties = macro_properties
        elements = tuple(
            Q16MITC16EASContinuumElement(
                reference,
                young_modulus=float(properties.young_modulus[element]),
                poisson_ratio=float(properties.poisson_ratio[element]),
                density=float(properties.density[element]),
            )
            for element, reference in enumerate(mesh.reference_elements)
        )
        mass_elements = tuple(
            Q16ContinuumElement(
                reference,
                young_modulus=float(properties.young_modulus[element]),
                poisson_ratio=float(properties.poisson_ratio[element]),
                density=float(properties.density[element]),
            )
            for element, reference in enumerate(mesh.reference_elements)
        )
        object.__setattr__(self, "mesh", mesh)
        object.__setattr__(self, "elements", elements)
        object.__setattr__(self, "mass_elements", mass_elements)
        object.__setattr__(self, "macro_properties", properties)
        object.__setattr__(self, "young_modulus", elements[0].young_modulus)
        object.__setattr__(self, "poisson_ratio", elements[0].poisson_ratio)
        object.__setattr__(self, "density", elements[0].density)
        object.__setattr__(
            self,
            "total_reference_volume",
            float(sum(element.reference_volume for element in elements)),
        )
        object.__setattr__(
            self,
            "total_reference_mass",
            float(
                sum(
                    element.reference_volume * element.density
                    for element in mass_elements
                )
            ),
        )

    def strain_energy(self, state: np.ndarray) -> float:
        local = self.mesh.gather(state)
        return float(
            sum(
                element.strain_energy(element_state)
                for element, element_state in zip(self.elements, local, strict=True)
            )
        )

    def internal_force(self, state: np.ndarray) -> np.ndarray:
        local = self.mesh.gather(state)
        element_force = np.ascontiguousarray(
            np.stack(
                [
                    element.internal_force(element_state)
                    for element, element_state in zip(self.elements, local, strict=True)
                ]
            )
        )
        return self.mesh.assemble(element_force)

    def tangent_action(self, state: np.ndarray, direction: np.ndarray) -> np.ndarray:
        local_state = self.mesh.gather(state)
        local_direction = self.mesh.gather(direction)
        element_action = np.ascontiguousarray(
            np.stack(
                [
                    element.tangent_action(element_state, element_direction)
                    for element, element_state, element_direction in zip(
                        self.elements,
                        local_state,
                        local_direction,
                        strict=True,
                    )
                ]
            )
        )
        return self.mesh.assemble(element_action)

    def mass_action(self, acceleration: np.ndarray) -> np.ndarray:
        local = self.mesh.gather(acceleration)
        element_action = np.ascontiguousarray(
            np.stack(
                [
                    element.mass_action(element_acceleration)
                    for element, element_acceleration in zip(
                        self.mass_elements, local, strict=True
                    )
                ]
            )
        )
        return self.mesh.assemble(element_action)


def _structured_axis(element_count: int, extent: float) -> np.ndarray:
    node_count = 3 * element_count + 1
    coordinates = np.empty(node_count, dtype=np.float64)
    element_extent = extent / element_count
    for global_node in range(node_count):
        element = min(global_node // 3, element_count - 1)
        local_node = global_node - 3 * element
        coordinates[global_node] = element_extent * (
            element + 0.5 * (Q16_PARAMETRIC_NODES_1D[local_node] + 1.0)
        )
    return coordinates


def make_rectangular_q16_mesh(
    *,
    chordwise_element_count: int,
    spanwise_element_count: int,
    chord: float,
    span: float,
    thickness: float,
) -> Q16Mesh:
    """Create a planar rectangular shared-node Q16 macro mesh.

    Chordwise nodes are the first structured index and vary fastest inside each
    Q16 element.  Spanwise nodes form the second structured index.
    """

    for name, count in (
        ("chordwise_element_count", chordwise_element_count),
        ("spanwise_element_count", spanwise_element_count),
    ):
        if type(count) is not int or count <= 0:
            raise ValueError(f"{name} must be a positive exact int")
    chord_value = _positive_scalar("chord", chord)
    span_value = _positive_scalar("span", span)
    thickness_value = _positive_scalar("thickness", thickness)
    chord_nodes = _structured_axis(chordwise_element_count, chord_value)
    span_nodes = _structured_axis(spanwise_element_count, span_value)
    chord_node_count = chord_nodes.size
    span_node_count = span_nodes.size
    rows = np.zeros((chord_node_count * span_node_count, 6), dtype=np.float64)
    for span_node, span_coordinate in enumerate(span_nodes):
        for chord_node, chord_coordinate in enumerate(chord_nodes):
            node = span_node * chord_node_count + chord_node
            rows[node, :3] = [chord_coordinate, span_coordinate, 0.0]
            rows[node, 3:] = [0.0, 0.0, 0.5 * thickness_value]
    connectivity = np.empty(
        (chordwise_element_count * spanwise_element_count, 16), dtype=np.int64
    )
    element = 0
    for span_element in range(spanwise_element_count):
        for chord_element in range(chordwise_element_count):
            for local_span in range(4):
                for local_chord in range(4):
                    local_node = 4 * local_span + local_chord
                    global_span = 3 * span_element + local_span
                    global_chord = 3 * chord_element + local_chord
                    connectivity[element, local_node] = (
                        global_span * chord_node_count + global_chord
                    )
            element += 1
    return Q16Mesh(reference_rows=rows, connectivity=connectivity)


__all__ = [
    "MAX_Q16_MESH_ELEMENT_COUNT",
    "MAX_Q16_MESH_NODE_COUNT",
    "Q16ContinuumMesh",
    "Q16MacroPropertyField",
    "Q16MITC16EASMesh",
    "Q16Mesh",
    "make_rectangular_q16_mesh",
]
