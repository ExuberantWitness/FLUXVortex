"""Conservative ring-edge to rVPM diagnostic-shadow bridge.

This module is the pure-topology R2/B2 bridge.  It deliberately has no Ptera
adapter and no feedback path: rings remain the physical owner, while the
particles emitted here are diagnostic shadows only.  Edge identity is based
solely on explicit stable node IDs; coordinates are never rounded or used as
topological keys.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, fsum
from numbers import Integral, Real
from typing import Any, Sequence, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]
StableId: TypeAlias = int | str
EdgeKey: TypeAlias = tuple[StableId, StableId]
ParticleId: TypeAlias = tuple[Any, ...]

FROZEN_OVERLAP_LAMBDA = 2.125
RING_PHYSICAL_OWNER = "ring"
DIAGNOSTIC_SHADOW_OWNER = "diagnostic_shadow"


@dataclass(frozen=True)
class BridgeNode:
    """A source node with an explicit, coordinate-independent identity."""

    node_id: StableId
    position: ArrayLike


@dataclass(frozen=True)
class DirectedRing:
    """A four-node ring in its source traversal order."""

    ring_id: StableId
    node_ids: Sequence[StableId]
    circulation: float


@dataclass(frozen=True)
class EdgeIncidence:
    """One directed ring contribution to a canonical physical edge."""

    ring_id: StableId
    traversal_index: int
    source_start_id: StableId
    source_end_id: StableId
    canonical_sign: int
    ring_circulation: float
    signed_circulation: float


@dataclass(frozen=True)
class EdgeLedger:
    """All signed incidences and their reconstructed physical edge."""

    key: EdgeKey
    start_position: tuple[float, float, float]
    end_position: tuple[float, float, float]
    incidences: tuple[EdgeIncidence, ...]
    circulation: float
    vector_moment: tuple[float, float, float]

    @property
    def retained(self) -> bool:
        """Whether this edge has nonzero net circulation."""

        return self.circulation != 0.0


@dataclass(frozen=True)
class EdgeGraph:
    """Validated global edge ledger, including canceled shared edges."""

    nodes: tuple[BridgeNode, ...]
    edges: tuple[EdgeLedger, ...]
    incidence_residual: float
    edge_reconstruction_residual: float
    global_vector_moment: tuple[float, float, float]

    @property
    def retained_edges(self) -> tuple[EdgeLedger, ...]:
        return tuple(edge for edge in self.edges if edge.retained)


@dataclass(frozen=True)
class ParticleLineage:
    """Auditable ownership and source record for one shadow particle."""

    particle_id: ParticleId
    source_edge: EdgeKey
    subdivision_index: int
    subdivision_count: int
    ring_incidences: tuple[EdgeIncidence, ...]
    step: int
    physical_owner: str
    owner_state: str


@dataclass(frozen=True)
class BridgeDiagnostics:
    """Mechanical-gate diagnostics; successful builds have zero event counts."""

    enabled: bool
    source_edge_count: int
    retained_edge_count: int
    particle_count: int
    incidence_residual: float
    edge_reconstruction_residual: float
    max_edge_conservation_abs: float
    max_edge_conservation_rel: float
    global_conservation_abs: float
    global_conservation_rel: float
    clip_count: int
    nonfinite_count: int
    owner_conflict_count: int
    feedback_call_count: int


@dataclass(frozen=True)
class ShadowBridgeResult:
    """Particles and ledger produced by the one-way diagnostic bridge."""

    edge_graph: EdgeGraph | None
    positions: FloatArray
    gamma: FloatArray
    sigma: FloatArray
    particle_ids: tuple[ParticleId, ...]
    lineage: tuple[ParticleLineage, ...]
    feedback_velocity: None
    diagnostics: BridgeDiagnostics


def _stable_id(name: str, value: object) -> StableId:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an explicit integer or string ID")
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, str) and value:
        return value
    raise ValueError(f"{name} must be a nonempty string or integer ID")


def _id_sort_key(value: StableId) -> tuple[str, int | str]:
    if isinstance(value, int):
        return ("integer", value)
    return ("string", value)


def canonical_edge_key(start_id: StableId, end_id: StableId) -> EdgeKey:
    """Return a deterministic unoriented key from explicit node identities."""

    start = _stable_id("start_id", start_id)
    end = _stable_id("end_id", end_id)
    if start == end and type(start) is type(end):
        raise ValueError("an edge must connect two distinct node IDs")
    if _id_sort_key(start) < _id_sort_key(end):
        return (start, end)
    return (end, start)


def _finite_position(name: str, value: ArrayLike) -> tuple[float, float, float]:
    original = np.asarray(value)
    if original.dtype.kind not in "iuf" or original.dtype.kind == "b":
        raise ValueError(f"{name} must use a real numeric dtype")
    array = np.asarray(original, dtype=np.float64)
    if array.shape != (3,):
        raise ValueError(f"{name} must be a length-3 position")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return (float(array[0]), float(array[1]), float(array[2]))


def _finite_real(name: str, value: object) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be a finite real number")
    return result


def _positive_integer(name: str, value: object) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be a positive integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _nonnegative_integer(name: str, value: object) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be a nonnegative integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return result


def _relative_error(absolute_error: float, reference_norm: float) -> float:
    if reference_norm > 0.0:
        return absolute_error / reference_norm
    return 0.0 if absolute_error == 0.0 else float("inf")


def _stable_vector_sum(values: ArrayLike) -> FloatArray:
    """Sum 3-vectors with componentwise ``fsum`` for audit-grade closure."""

    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return np.zeros(3, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError("stable vector sum requires an (n, 3) array")
    return np.asarray(
        [fsum(float(value) for value in array[:, axis]) for axis in range(3)],
        dtype=np.float64,
    )


def assemble_ring_edge_graph(
    nodes: Sequence[BridgeNode],
    rings: Sequence[DirectedRing],
) -> EdgeGraph:
    """Assemble a signed global edge graph from arbitrary ring input order.

    Each ring is traversed exactly in the supplied node order.  Canonical edge
    orientation is selected only from stable node IDs, so coincident or nearly
    coincident coordinates can never merge topology.
    """

    try:
        node_inputs = tuple(nodes)
        ring_inputs = tuple(rings)
    except TypeError as error:
        raise ValueError("nodes and rings must be finite sequences") from error
    if not node_inputs:
        raise ValueError("nodes must not be empty")
    if not ring_inputs:
        raise ValueError("rings must not be empty")

    node_positions: dict[StableId, tuple[float, float, float]] = {}
    for index, node in enumerate(node_inputs):
        if not isinstance(node, BridgeNode):
            raise ValueError(f"nodes[{index}] must be a BridgeNode")
        node_id = _stable_id(f"nodes[{index}].node_id", node.node_id)
        position = _finite_position(f"nodes[{index}].position", node.position)
        previous = node_positions.get(node_id)
        if previous is not None and previous != position:
            raise ValueError(
                f"repeated node ID {node_id!r} has inconsistent coordinates"
            )
        node_positions[node_id] = position

    incidence_buckets: dict[EdgeKey, list[EdgeIncidence]] = {}
    seen_ring_ids: set[StableId] = set()
    for ring_index, ring in enumerate(ring_inputs):
        if not isinstance(ring, DirectedRing):
            raise ValueError(f"rings[{ring_index}] must be a DirectedRing")
        ring_id = _stable_id(f"rings[{ring_index}].ring_id", ring.ring_id)
        if ring_id in seen_ring_ids:
            raise ValueError(f"ring ID {ring_id!r} is repeated")
        seen_ring_ids.add(ring_id)
        try:
            ring_node_ids = tuple(
                _stable_id(f"rings[{ring_index}].node_ids", item)
                for item in ring.node_ids
            )
        except TypeError as error:
            raise ValueError("ring node_ids must be a finite sequence") from error
        if len(ring_node_ids) != 4:
            raise ValueError("each source ring must contain exactly four node IDs")
        if len(set(ring_node_ids)) != 4:
            raise ValueError("a source ring must contain four distinct node IDs")
        missing = [item for item in ring_node_ids if item not in node_positions]
        if missing:
            raise ValueError(
                f"ring {ring_id!r} references unknown node IDs {missing!r}"
            )
        circulation = _finite_real(f"rings[{ring_index}].circulation", ring.circulation)

        for traversal_index in range(4):
            source_start = ring_node_ids[traversal_index]
            source_end = ring_node_ids[(traversal_index + 1) % 4]
            key = canonical_edge_key(source_start, source_end)
            canonical_sign = 1 if (source_start, source_end) == key else -1
            start_position = np.asarray(node_positions[source_start])
            end_position = np.asarray(node_positions[source_end])
            length = float(np.linalg.norm(end_position - start_position))
            if not np.isfinite(length) or length <= 0.0:
                raise ValueError(
                    f"edge {source_start!r}->{source_end!r} has nonpositive length"
                )
            incidence_buckets.setdefault(key, []).append(
                EdgeIncidence(
                    ring_id=ring_id,
                    traversal_index=traversal_index,
                    source_start_id=source_start,
                    source_end_id=source_end,
                    canonical_sign=canonical_sign,
                    ring_circulation=circulation,
                    signed_circulation=canonical_sign * circulation,
                )
            )

    ledgers: list[EdgeLedger] = []
    incidence_residual = 0.0
    reconstruction_residual = 0.0
    global_components: list[list[float]] = [[], [], []]
    for key in sorted(
        incidence_buckets,
        key=lambda item: (_id_sort_key(item[0]), _id_sort_key(item[1])),
    ):
        incidences = tuple(
            sorted(
                incidence_buckets[key],
                key=lambda item: (_id_sort_key(item.ring_id), item.traversal_index),
            )
        )
        if len(incidences) > 2:
            raise ValueError(
                f"edge {key!r} is non-manifold with {len(incidences)} incidences"
            )
        if len(incidences) == 2 and {item.canonical_sign for item in incidences} != {
            -1,
            1,
        }:
            raise ValueError(f"edge {key!r} has two co-oriented surface incidences")
        signed_values = [item.signed_circulation for item in incidences]
        circulation = fsum(signed_values)
        if circulation == 0.0:
            circulation = 0.0
        incidence_residual = max(
            incidence_residual,
            max(
                abs(
                    item.signed_circulation
                    - item.canonical_sign * item.ring_circulation
                )
                for item in incidences
            ),
        )

        start_position = node_positions[key[0]]
        end_position = node_positions[key[1]]
        delta = np.subtract(end_position, start_position, dtype=np.float64)
        expected_vector = circulation * delta
        if not np.all(np.isfinite(expected_vector)):
            raise FloatingPointError(
                f"edge {key!r} produced a non-finite vector moment"
            )
        incidence_vector = np.array(
            [
                fsum(item.signed_circulation * delta[j] for item in incidences)
                for j in range(3)
            ],
            dtype=np.float64,
        )
        residual = float(np.linalg.norm(incidence_vector - expected_vector))
        reconstruction_residual = max(reconstruction_residual, residual)
        vector_moment = tuple(float(value) for value in expected_vector)
        for component in range(3):
            global_components[component].append(vector_moment[component])
        ledgers.append(
            EdgeLedger(
                key=key,
                start_position=start_position,
                end_position=end_position,
                incidences=incidences,
                circulation=circulation,
                vector_moment=vector_moment,
            )
        )

    if incidence_residual > 1.0e-12 or reconstruction_residual > 1.0e-12:
        raise FloatingPointError("edge incidence reconstruction exceeds tolerance")
    global_vector = tuple(fsum(component) for component in global_components)
    validated_nodes = tuple(
        BridgeNode(node_id=node_id, position=node_positions[node_id])
        for node_id in sorted(node_positions, key=_id_sort_key)
    )
    return EdgeGraph(
        nodes=validated_nodes,
        edges=tuple(ledgers),
        incidence_residual=incidence_residual,
        edge_reconstruction_residual=reconstruction_residual,
        global_vector_moment=global_vector,
    )


def _validate_owner_transition(physical_owner: object, owner_state: object) -> None:
    if not isinstance(physical_owner, str) or physical_owner != RING_PHYSICAL_OWNER:
        raise ValueError(
            f"unsupported physical owner {physical_owner!r}; rings must remain owner"
        )
    if not isinstance(owner_state, str) or owner_state != DIAGNOSTIC_SHADOW_OWNER:
        raise ValueError(
            f"unsupported particle owner state {owner_state!r}; only diagnostic_shadow is allowed"
        )


def _validate_graph_for_deposition(graph: object) -> EdgeGraph:
    if not isinstance(graph, EdgeGraph):
        raise ValueError("edge_graph must be produced by assemble_ring_edge_graph")
    if not np.isfinite(graph.incidence_residual) or not np.isfinite(
        graph.edge_reconstruction_residual
    ):
        raise ValueError("edge graph residuals must be finite")
    if (
        graph.incidence_residual > 1.0e-12
        or graph.edge_reconstruction_residual > 1.0e-12
    ):
        raise ValueError("edge graph residuals exceed the bridge tolerance")
    node_positions: dict[StableId, tuple[float, float, float]] = {}
    if not graph.nodes:
        raise ValueError("edge graph nodes must not be empty")
    for index, node in enumerate(graph.nodes):
        if not isinstance(node, BridgeNode):
            raise ValueError(f"edge graph nodes[{index}] must be a BridgeNode")
        node_id = _stable_id(f"edge graph nodes[{index}].node_id", node.node_id)
        if node_id in node_positions:
            raise ValueError(f"edge graph repeats node ID {node_id!r}")
        node_positions[node_id] = _finite_position(
            f"edge graph nodes[{index}].position", node.position
        )

    seen_keys: set[EdgeKey] = set()
    reconstructed_global = np.zeros(3, dtype=np.float64)
    for edge in graph.edges:
        if not isinstance(edge, EdgeLedger):
            raise ValueError("edge graph contains an invalid ledger")
        if edge.key in seen_keys:
            raise ValueError(f"edge graph repeats canonical edge {edge.key!r}")
        seen_keys.add(edge.key)
        if canonical_edge_key(*edge.key) != edge.key:
            raise ValueError(f"edge key {edge.key!r} is not canonical")
        if edge.key[0] not in node_positions or edge.key[1] not in node_positions:
            raise ValueError(f"edge {edge.key!r} references an unknown graph node")
        start = np.asarray(_finite_position("edge start_position", edge.start_position))
        end = np.asarray(_finite_position("edge end_position", edge.end_position))
        if tuple(start) != node_positions[edge.key[0]]:
            raise ValueError(
                f"edge {edge.key!r} start position disagrees with its graph node"
            )
        if tuple(end) != node_positions[edge.key[1]]:
            raise ValueError(
                f"edge {edge.key!r} end position disagrees with its graph node"
            )
        length = float(np.linalg.norm(end - start))
        if not np.isfinite(length) or length <= 0.0:
            raise ValueError(f"edge {edge.key!r} has nonpositive length")
        circulation = _finite_real("edge circulation", edge.circulation)
        if not edge.incidences:
            raise ValueError(f"edge {edge.key!r} has no incidence ledger")
        if len(edge.incidences) > 2:
            raise ValueError(f"edge {edge.key!r} is non-manifold")
        if len(edge.incidences) == 2 and {
            item.canonical_sign for item in edge.incidences
        } != {-1, 1}:
            raise ValueError(
                f"edge {edge.key!r} has two co-oriented surface incidences"
            )
        for incidence in edge.incidences:
            if not isinstance(incidence, EdgeIncidence):
                raise ValueError(f"edge {edge.key!r} contains an invalid incidence")
            canonical_sign = incidence.canonical_sign
            if canonical_sign not in (-1, 1):
                raise ValueError(f"edge {edge.key!r} has an invalid incidence sign")
            expected_sign = (
                1
                if (
                    incidence.source_start_id,
                    incidence.source_end_id,
                )
                == edge.key
                else -1
            )
            if canonical_sign != expected_sign:
                raise ValueError(
                    f"edge {edge.key!r} incidence sign disagrees with traversal"
                )
            ring_circulation = _finite_real(
                "incidence ring_circulation", incidence.ring_circulation
            )
            signed_circulation = _finite_real(
                "incidence signed_circulation", incidence.signed_circulation
            )
            if signed_circulation != canonical_sign * ring_circulation:
                raise ValueError(
                    f"edge {edge.key!r} has an inconsistent signed incidence"
                )
            if (
                canonical_edge_key(incidence.source_start_id, incidence.source_end_id)
                != edge.key
            ):
                raise ValueError(f"edge {edge.key!r} has an incidence on another edge")
        reconstructed = fsum(item.signed_circulation for item in edge.incidences)
        if abs(reconstructed - circulation) > 1.0e-12:
            raise ValueError(f"edge {edge.key!r} has an inconsistent incidence ledger")
        expected_vector = circulation * (end - start)
        stored_vector = np.asarray(
            _finite_position("edge vector_moment", edge.vector_moment)
        )
        if float(np.linalg.norm(stored_vector - expected_vector)) > 1.0e-12:
            raise ValueError(f"edge {edge.key!r} has an inconsistent vector moment")
        reconstructed_global += stored_vector
    stored_global = np.asarray(
        _finite_position("edge graph global_vector_moment", graph.global_vector_moment)
    )
    if float(np.linalg.norm(reconstructed_global - stored_global)) > 1.0e-12:
        raise ValueError("edge graph has an inconsistent global vector moment")
    return graph


def _deposit_validated_graph(
    graph: EdgeGraph,
    *,
    count_by_edge: dict[EdgeKey, int],
    sigma_by_edge: dict[EdgeKey, float],
    step_number: int,
    physical_owner: str,
    owner_state: str,
    particle_schema: str,
) -> ShadowBridgeResult:
    """Deposit a validated graph using explicit per-edge counts and radii."""

    positions: list[FloatArray] = []
    gamma_vectors: list[FloatArray] = []
    sigma_values: list[float] = []
    particle_ids: list[ParticleId] = []
    lineage: list[ParticleLineage] = []
    edge_errors_abs: list[float] = []
    edge_errors_rel: list[float] = []

    for edge in graph.edges:
        if not edge.retained:
            continue
        try:
            count = _positive_integer("edge particle count", count_by_edge[edge.key])
            sigma = _finite_real("edge smoothing radius", sigma_by_edge[edge.key])
        except KeyError as error:
            raise ValueError(
                f"missing deposition rule for retained edge {edge.key!r}"
            ) from error
        if sigma <= 0.0:
            raise ValueError(f"edge {edge.key!r} produced nonpositive sigma")
        start = np.asarray(edge.start_position, dtype=np.float64)
        end = np.asarray(edge.end_position, dtype=np.float64)
        delta_l = (end - start) / count
        subsegment_length = float(np.linalg.norm(delta_l))
        if not np.isfinite(subsegment_length) or subsegment_length <= 0.0:
            raise ValueError(f"edge {edge.key!r} has nonpositive subsegment length")
        edge_start_index = len(positions)
        incidence_signature = tuple(
            (
                item.ring_id,
                item.traversal_index,
                item.source_start_id,
                item.source_end_id,
                item.canonical_sign,
                item.ring_circulation,
                item.signed_circulation,
            )
            for item in edge.incidences
        )
        for subdivision_index in range(count):
            position = start + (subdivision_index + 0.5) * delta_l
            gamma_vector = edge.circulation * delta_l
            if particle_schema == "rvpm-edge-shadow-v1":
                particle_id: ParticleId = (
                    particle_schema,
                    edge.key,
                    subdivision_index,
                    count,
                    incidence_signature,
                    step_number,
                    owner_state,
                )
            else:
                particle_id = (
                    particle_schema,
                    edge.key,
                    subdivision_index,
                    count,
                    sigma,
                    incidence_signature,
                    step_number,
                    owner_state,
                )
            positions.append(position)
            gamma_vectors.append(gamma_vector.copy())
            sigma_values.append(sigma)
            particle_ids.append(particle_id)
            lineage.append(
                ParticleLineage(
                    particle_id=particle_id,
                    source_edge=edge.key,
                    subdivision_index=subdivision_index,
                    subdivision_count=count,
                    ring_incidences=edge.incidences,
                    step=step_number,
                    physical_owner=physical_owner,
                    owner_state=owner_state,
                )
            )
        actual = _stable_vector_sum(gamma_vectors[edge_start_index:])
        expected = np.asarray(edge.vector_moment, dtype=np.float64)
        absolute = float(np.linalg.norm(actual - expected))
        edge_errors_abs.append(absolute)
        edge_errors_rel.append(
            _relative_error(absolute, float(np.linalg.norm(expected)))
        )

    if positions:
        position_array = np.ascontiguousarray(np.vstack(positions), dtype=np.float64)
        gamma_array = np.ascontiguousarray(np.vstack(gamma_vectors), dtype=np.float64)
        sigma_array = np.ascontiguousarray(np.asarray(sigma_values), dtype=np.float64)
    else:
        position_array = np.empty((0, 3), dtype=np.float64)
        gamma_array = np.empty((0, 3), dtype=np.float64)
        sigma_array = np.empty((0,), dtype=np.float64)
    if not (
        np.all(np.isfinite(position_array))
        and np.all(np.isfinite(gamma_array))
        and np.all(np.isfinite(sigma_array))
    ):
        raise FloatingPointError("edge deposition produced non-finite particle state")
    if sigma_array.size and np.any(sigma_array <= 0.0):
        raise FloatingPointError("edge deposition produced nonpositive sigma")
    if len(set(particle_ids)) != len(particle_ids):
        raise FloatingPointError("edge deposition produced duplicate particle IDs")
    if len({item.particle_id for item in lineage}) != len(lineage):
        raise FloatingPointError("edge deposition produced duplicate lineage records")

    expected_global = np.asarray(graph.global_vector_moment, dtype=np.float64)
    actual_global = _stable_vector_sum(gamma_array)
    global_absolute = float(np.linalg.norm(actual_global - expected_global))
    # A closed ring has an exactly zero net vector moment.  Use the sum of
    # physical edge moments as the relative scale so roundoff is not divided
    # by a zero resultant.
    global_reference_scale = fsum(
        float(np.linalg.norm(np.asarray(edge.vector_moment, dtype=np.float64)))
        for edge in graph.retained_edges
    )
    global_relative = _relative_error(global_absolute, global_reference_scale)
    max_edge_absolute = max(edge_errors_abs, default=0.0)
    max_edge_relative = max(edge_errors_rel, default=0.0)
    if max_edge_absolute > 1.0e-14 or max_edge_relative > 1.0e-12:
        raise FloatingPointError("per-edge vector conservation exceeds tolerance")
    if global_absolute > 1.0e-14 or global_relative > 1.0e-12:
        raise FloatingPointError("global vector conservation exceeds tolerance")

    diagnostics = BridgeDiagnostics(
        enabled=True,
        source_edge_count=len(graph.edges),
        retained_edge_count=len(graph.retained_edges),
        particle_count=len(particle_ids),
        incidence_residual=graph.incidence_residual,
        edge_reconstruction_residual=graph.edge_reconstruction_residual,
        max_edge_conservation_abs=max_edge_absolute,
        max_edge_conservation_rel=max_edge_relative,
        global_conservation_abs=global_absolute,
        global_conservation_rel=global_relative,
        clip_count=0,
        nonfinite_count=0,
        owner_conflict_count=0,
        feedback_call_count=0,
    )
    return ShadowBridgeResult(
        edge_graph=graph,
        positions=position_array,
        gamma=gamma_array,
        sigma=sigma_array,
        particle_ids=tuple(particle_ids),
        lineage=tuple(lineage),
        feedback_velocity=None,
        diagnostics=diagnostics,
    )


def deposit_edge_graph(
    edge_graph: EdgeGraph,
    *,
    subdivisions: int,
    step: int,
    physical_owner: str = RING_PHYSICAL_OWNER,
    owner_state: str = DIAGNOSTIC_SHADOW_OWNER,
    overlap_lambda: float = FROZEN_OVERLAP_LAMBDA,
) -> ShadowBridgeResult:
    """Deposit each edge with the same count and edge-local smoothing radius.

    This legacy R2/B2 mode is retained for analytic finite-segment convergence:
    every edge has exactly ``subdivisions`` particles and therefore its own
    ``sigma = overlap_lambda * edge_length / subdivisions``.  It must not be
    used to infer a transport core for a graph containing both macroscopic
    span edges and O(dt) newborn edges; use :func:`deposit_edge_graph_fixed_sigma`
    for that FLOWUnsteady-style spatial contract.
    """

    graph = _validate_graph_for_deposition(edge_graph)
    count = _positive_integer("subdivisions", subdivisions)
    step_number = _nonnegative_integer("step", step)
    _validate_owner_transition(physical_owner, owner_state)
    overlap = _finite_real("overlap_lambda", overlap_lambda)
    if overlap <= 0.0:
        raise ValueError("overlap_lambda must produce strictly positive sigma")
    if overlap != FROZEN_OVERLAP_LAMBDA:
        raise ValueError(
            f"overlap_lambda is frozen at {FROZEN_OVERLAP_LAMBDA} for R2/B2"
        )
    count_by_edge: dict[EdgeKey, int] = {}
    sigma_by_edge: dict[EdgeKey, float] = {}
    for edge in graph.retained_edges:
        start = np.asarray(edge.start_position, dtype=np.float64)
        end = np.asarray(edge.end_position, dtype=np.float64)
        subsegment_length = float(np.linalg.norm(end - start)) / count
        sigma = overlap * subsegment_length
        if not np.isfinite(sigma) or sigma <= 0.0:
            raise ValueError(f"edge {edge.key!r} produced nonpositive sigma")
        count_by_edge[edge.key] = count
        sigma_by_edge[edge.key] = sigma
    return _deposit_validated_graph(
        graph,
        count_by_edge=count_by_edge,
        sigma_by_edge=sigma_by_edge,
        step_number=step_number,
        physical_owner=physical_owner,
        owner_state=owner_state,
        particle_schema="rvpm-edge-shadow-v1",
    )


def deposit_edge_graph_fixed_sigma(
    edge_graph: EdgeGraph,
    *,
    smoothing_radius: float,
    step: int,
    physical_owner: str = RING_PHYSICAL_OWNER,
    owner_state: str = DIAGNOSTIC_SHADOW_OWNER,
    overlap_lambda: float = FROZEN_OVERLAP_LAMBDA,
) -> ShadowBridgeResult:
    """Deposit edges with one spatially prescribed smoothing radius.

    This adapts FLOWUnsteady's fixed-radius/count rule for static vortex-sheet
    discretization: ``sigma`` is a spatial input, the maximum center spacing is
    ``sigma / overlap``, and each edge receives
    ``ceil(edge_length / max_spacing)`` midpoint particles.
    Consequently the realized local overlap is at least the requested value;
    short newborn edges retain the prescribed core instead of shrinking it in
    proportion to the release time step.
    """

    graph = _validate_graph_for_deposition(edge_graph)
    sigma = _finite_real("smoothing_radius", smoothing_radius)
    if sigma <= 0.0:
        raise ValueError("smoothing_radius must be strictly positive")
    step_number = _nonnegative_integer("step", step)
    _validate_owner_transition(physical_owner, owner_state)
    overlap = _finite_real("overlap_lambda", overlap_lambda)
    if overlap <= 0.0:
        raise ValueError("overlap_lambda must be strictly positive")
    if overlap != FROZEN_OVERLAP_LAMBDA:
        raise ValueError(
            f"overlap_lambda is frozen at {FROZEN_OVERLAP_LAMBDA} for R2/B2"
        )
    maximum_spacing = sigma / overlap
    if not np.isfinite(maximum_spacing) or maximum_spacing <= 0.0:
        raise ValueError(
            "smoothing_radius / overlap_lambda must be finite and positive"
        )

    count_by_edge: dict[EdgeKey, int] = {}
    sigma_by_edge: dict[EdgeKey, float] = {}
    for edge in graph.retained_edges:
        start = np.asarray(edge.start_position, dtype=np.float64)
        end = np.asarray(edge.end_position, dtype=np.float64)
        edge_length = float(np.linalg.norm(end - start))
        count_float = edge_length / maximum_spacing
        if not np.isfinite(count_float):
            raise ValueError(f"edge {edge.key!r} produced a non-finite particle count")
        count = max(1, ceil(count_float))
        count_by_edge[edge.key] = count
        sigma_by_edge[edge.key] = sigma
    return _deposit_validated_graph(
        graph,
        count_by_edge=count_by_edge,
        sigma_by_edge=sigma_by_edge,
        step_number=step_number,
        physical_owner=physical_owner,
        owner_state=owner_state,
        particle_schema="rvpm-edge-shadow-fixed-sigma-v1",
    )


def deposit_edge_graph_prescribed_sigma_and_spacing(
    edge_graph: EdgeGraph,
    *,
    smoothing_radius: float,
    target_spacing: float,
    step: int,
    physical_owner: str = RING_PHYSICAL_OWNER,
    owner_state: str = DIAGNOSTIC_SHADOW_OWNER,
    minimum_overlap: float = FROZEN_OVERLAP_LAMBDA,
) -> ShadowBridgeResult:
    """Deposit a graph while keeping core radius and quadrature independent.

    ``smoothing_radius`` is one fixed SI-space regularization scale for every
    particle. ``target_spacing`` independently bounds the midpoint quadrature
    spacing through ``ceil(edge_length / target_spacing)``.  The function
    rejects a requested spacing that cannot provide the frozen minimum overlap
    ``sigma / h >= 2.125``; it never silently changes either input to make the
    inequality pass.

    This API is intended for fixed-core quadrature-refinement studies.  A scan
    over ``smoothing_radius`` is a separate core-sensitivity study and must not
    be described as mesh convergence.
    """

    graph = _validate_graph_for_deposition(edge_graph)
    sigma = _finite_real("smoothing_radius", smoothing_radius)
    spacing = _finite_real("target_spacing", target_spacing)
    overlap = _finite_real("minimum_overlap", minimum_overlap)
    if sigma <= 0.0:
        raise ValueError("smoothing_radius must be strictly positive")
    if spacing <= 0.0:
        raise ValueError("target_spacing must be strictly positive")
    if overlap != FROZEN_OVERLAP_LAMBDA:
        raise ValueError(
            f"minimum_overlap is frozen at {FROZEN_OVERLAP_LAMBDA} for R2/B2"
        )
    requested_overlap = sigma / spacing
    if not np.isfinite(requested_overlap) or requested_overlap < overlap:
        raise ValueError("target_spacing is too large for the frozen minimum overlap")

    step_number = _nonnegative_integer("step", step)
    _validate_owner_transition(physical_owner, owner_state)
    count_by_edge: dict[EdgeKey, int] = {}
    sigma_by_edge: dict[EdgeKey, float] = {}
    for edge in graph.retained_edges:
        start = np.asarray(edge.start_position, dtype=np.float64)
        end = np.asarray(edge.end_position, dtype=np.float64)
        edge_length = float(np.linalg.norm(end - start))
        count_float = edge_length / spacing
        if not np.isfinite(count_float):
            raise ValueError(f"edge {edge.key!r} produced a non-finite particle count")
        count = max(1, ceil(count_float))
        realized_spacing = edge_length / count
        realized_overlap = sigma / realized_spacing
        if not np.isfinite(realized_overlap) or realized_overlap < overlap:
            raise FloatingPointError(
                f"edge {edge.key!r} violates the frozen minimum overlap"
            )
        count_by_edge[edge.key] = count
        sigma_by_edge[edge.key] = sigma
    return _deposit_validated_graph(
        graph,
        count_by_edge=count_by_edge,
        sigma_by_edge=sigma_by_edge,
        step_number=step_number,
        physical_owner=physical_owner,
        owner_state=owner_state,
        particle_schema="rvpm-edge-shadow-prescribed-sigma-spacing-v1",
    )


def build_diagnostic_shadow_bridge(
    nodes: Sequence[BridgeNode] | object,
    rings: Sequence[DirectedRing] | object,
    *,
    subdivisions: int | object,
    step: int | object,
    enabled: bool = True,
    physical_owner: str | object = RING_PHYSICAL_OWNER,
    owner_state: str | object = DIAGNOSTIC_SHADOW_OWNER,
    overlap_lambda: float | object = FROZEN_OVERLAP_LAMBDA,
) -> ShadowBridgeResult:
    """Dispatch the pure bridge, with a genuinely input-blind exact-off path."""

    if not isinstance(enabled, (bool, np.bool_)):
        raise ValueError("enabled must be boolean")
    if not bool(enabled):
        return ShadowBridgeResult(
            edge_graph=None,
            positions=np.empty((0, 3), dtype=np.float64),
            gamma=np.empty((0, 3), dtype=np.float64),
            sigma=np.empty((0,), dtype=np.float64),
            particle_ids=(),
            lineage=(),
            feedback_velocity=None,
            diagnostics=BridgeDiagnostics(
                enabled=False,
                source_edge_count=0,
                retained_edge_count=0,
                particle_count=0,
                incidence_residual=0.0,
                edge_reconstruction_residual=0.0,
                max_edge_conservation_abs=0.0,
                max_edge_conservation_rel=0.0,
                global_conservation_abs=0.0,
                global_conservation_rel=0.0,
                clip_count=0,
                nonfinite_count=0,
                owner_conflict_count=0,
                feedback_call_count=0,
            ),
        )

    _validate_owner_transition(physical_owner, owner_state)
    graph = assemble_ring_edge_graph(nodes, rings)  # type: ignore[arg-type]
    return deposit_edge_graph(
        graph,
        subdivisions=subdivisions,  # type: ignore[arg-type]
        step=step,  # type: ignore[arg-type]
        physical_owner=physical_owner,  # type: ignore[arg-type]
        owner_state=owner_state,  # type: ignore[arg-type]
        overlap_lambda=overlap_lambda,  # type: ignore[arg-type]
    )


__all__ = [
    "BridgeDiagnostics",
    "BridgeNode",
    "DIAGNOSTIC_SHADOW_OWNER",
    "DirectedRing",
    "EdgeGraph",
    "EdgeIncidence",
    "EdgeLedger",
    "FROZEN_OVERLAP_LAMBDA",
    "ParticleLineage",
    "RING_PHYSICAL_OWNER",
    "ShadowBridgeResult",
    "assemble_ring_edge_graph",
    "build_diagnostic_shadow_bridge",
    "canonical_edge_key",
    "deposit_edge_graph",
    "deposit_edge_graph_fixed_sigma",
    "deposit_edge_graph_prescribed_sigma_and_spacing",
]
