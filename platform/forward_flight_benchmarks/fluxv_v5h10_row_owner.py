"""Transactional global release-row owner for the FluxV v5h10 C1 gate.

This module is a manufactured, observation-free mechanical owner.  One call
accepts a complete spanwise row, builds one global canonical edge graph, and
then either issues an exact-compatible proposal or returns ``remesh_required``.
It has no Ptera, load, feedback, target-data, or force-scoring path.

For a compatible continuation the preceding live downstream particles keep
their support, smoothing radii, and stable IDs.  Their vector strengths receive
the new row's upstream incidence in place; only retained side edges and the new
downstream boundary are appended.  Clone, counter, and fresh-upstream particles
are never admitted to the committed physical state.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass, replace
import dis
from hashlib import sha256
from math import ceil, fsum
from numbers import Real
from pathlib import Path
from threading import RLock
from types import FunctionType, ModuleType
from typing import Any, Literal
import weakref
from weakref import WeakKeyDictionary, WeakSet

import numpy as np
from numpy.typing import ArrayLike, NDArray

import fluxvortex.rvpm_edge_bridge as _edge_bridge
from fluxvortex.rvpm_edge_bridge import (
    BridgeNode,
    DIAGNOSTIC_SHADOW_OWNER,
    DirectedRing,
    EdgeGraph,
    EdgeIncidence,
    EdgeKey,
    EdgeLedger,
    FROZEN_OVERLAP_LAMBDA,
    RING_PHYSICAL_OWNER,
)


FloatArray = NDArray[np.float64]
ProposalStatus = Literal["compatible", "remesh_required"]
OwnerPhase = Literal["post_commit_pre_transport", "post_transport"]
ParticleRole = Literal[
    "initial_boundary",
    "side",
    "interior_boundary",
    "live_downstream",
]

INTERFACE_ID = "fluxv-v5h10-global-release-row-owner-v1"
PARTICLE_SCHEMA = "fluxv-v5h10-global-row-particle-v1"
OPERATOR_ORDER = "global_row_commit_before_ptera_before_lsrk3"
_GENESIS_SHA256 = "0" * 64

_EDGE_SOURCE_PATH = Path(_edge_bridge.__file__).resolve()
_EDGE_SOURCE_SHA256 = sha256(_EDGE_SOURCE_PATH.read_bytes()).hexdigest()
_FROZEN_ASSEMBLE = _edge_bridge.assemble_ring_edge_graph
_FROZEN_CANONICAL_EDGE_KEY = _edge_bridge.canonical_edge_key
DEPENDENCY_SHA256 = (("fluxvortex.rvpm_edge_bridge", _EDGE_SOURCE_SHA256),)
_EDGE_NAMESPACE_NAMES = frozenset(vars(_edge_bridge))
_EDGE_NAMESPACE_ITEMS = tuple(vars(_edge_bridge).items())
_EDGE_FUNCTION_SEALS = tuple(
    (
        name,
        value,
        value.__code__,
        value.__defaults__,
        value.__kwdefaults__,
        value.__globals__,
    )
    for name, value in _EDGE_NAMESPACE_ITEMS
    if type(value) is FunctionType and value.__module__ == _edge_bridge.__name__
)
_EDGE_CLASS_SEALS = tuple(
    (class_type, frozenset(vars(class_type)), tuple(vars(class_type).items()))
    for class_type in (BridgeNode, DirectedRing, EdgeIncidence, EdgeLedger, EdgeGraph)
)


def _runtime_resolution_seals(
    functions: tuple[tuple[str, FunctionType, object, object, object, object], ...],
) -> tuple[
    tuple[tuple[str, str, tuple[str, ...], ModuleType, object], ...],
    tuple[tuple[str, str, dict[str, object], object], ...],
]:
    module_attributes: list[tuple[str, str, tuple[str, ...], ModuleType, object]] = []
    builtins: list[tuple[str, str, dict[str, object], object]] = []
    seen_attributes: set[tuple[int, tuple[str, ...]]] = set()
    seen_builtins: set[tuple[int, str]] = set()
    for function_name, function, _, _, _, _ in functions:
        instructions = tuple(dis.get_instructions(function))
        globals_mapping = function.__globals__
        builtins_mapping = function.__builtins__
        if type(builtins_mapping) is not dict:
            raise RuntimeError("edge function builtins mapping is invalid")
        for index, instruction in enumerate(instructions):
            if (
                instruction.opname != "LOAD_GLOBAL"
                or type(instruction.argval) is not str
            ):
                continue
            global_name = instruction.argval
            if global_name not in globals_mapping and global_name in builtins_mapping:
                key = (id(builtins_mapping), global_name)
                if key not in seen_builtins:
                    seen_builtins.add(key)
                    builtins.append(
                        (
                            function_name,
                            global_name,
                            builtins_mapping,
                            builtins_mapping[global_name],
                        )
                    )
            root = globals_mapping.get(global_name)
            if type(root) is not ModuleType:
                continue
            current: object = root
            path: list[str] = []
            for follower in instructions[index + 1 :]:
                if follower.opname not in ("LOAD_ATTR", "LOAD_METHOD"):
                    break
                if type(follower.argval) is not str:
                    break
                path.append(follower.argval)
                current = getattr(current, follower.argval)
                key = (id(root), tuple(path))
                if key not in seen_attributes:
                    seen_attributes.add(key)
                    module_attributes.append(
                        (
                            function_name,
                            global_name,
                            tuple(path),
                            root,
                            current,
                        )
                    )
    return tuple(module_attributes), tuple(builtins)


_EDGE_MODULE_ATTRIBUTE_SEALS, _EDGE_BUILTIN_SEALS = _runtime_resolution_seals(
    _EDGE_FUNCTION_SEALS
)


def _assert_dependencies() -> None:
    if (
        _edge_bridge.assemble_ring_edge_graph is not _FROZEN_ASSEMBLE
        or _edge_bridge.canonical_edge_key is not _FROZEN_CANONICAL_EDGE_KEY
        or sha256(_EDGE_SOURCE_PATH.read_bytes()).hexdigest() != _EDGE_SOURCE_SHA256
    ):
        raise RuntimeError("v5h10 row-owner edge-bridge dependency changed")
    namespace = vars(_edge_bridge)
    if frozenset(namespace) != _EDGE_NAMESPACE_NAMES or any(
        namespace.get(name) is not value for name, value in _EDGE_NAMESPACE_ITEMS
    ):
        raise RuntimeError("v5h10 row-owner edge-bridge runtime global changed")
    for (
        name,
        function,
        code,
        defaults,
        kwdefaults,
        globals_mapping,
    ) in _EDGE_FUNCTION_SEALS:
        if (
            namespace.get(name) is not function
            or function.__code__ is not code
            or function.__defaults__ is not defaults
            or function.__kwdefaults__ is not kwdefaults
            or function.__globals__ is not globals_mapping
        ):
            raise RuntimeError(f"v5h10 edge-bridge function changed: {name}")
    for class_type, names, items in _EDGE_CLASS_SEALS:
        namespace = vars(class_type)
        if frozenset(namespace) != names or any(
            namespace.get(name) is not value for name, value in items
        ):
            raise RuntimeError(
                f"v5h10 edge-bridge class changed: {class_type.__qualname__}"
            )
    for (
        function_name,
        global_name,
        path,
        root,
        expected,
    ) in _EDGE_MODULE_ATTRIBUTE_SEALS:
        current: object = root
        for attribute in path:
            current = getattr(current, attribute)
        if current is not expected:
            raise RuntimeError(
                "v5h10 edge-bridge module attribute changed: "
                f"{function_name}:{global_name}.{'.'.join(path)}"
            )
    for function_name, name, mapping, expected in _EDGE_BUILTIN_SEALS:
        if mapping.get(name) is not expected:
            raise RuntimeError(
                f"v5h10 edge-bridge builtin changed: {function_name}:{name}"
            )


def _exact_positive_integer(name: str, value: object) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an exact integer")
    if value <= 0:
        raise ValueError(f"{name} must be strictly positive")
    return value


def _exact_nonnegative_integer(name: str, value: object) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an exact integer")
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return value


def _nonempty_string(name: str, value: object) -> str:
    if type(value) is not str:
        raise TypeError(f"{name} must be an exact string")
    if not value:
        raise ValueError(f"{name} must not be empty")
    return value


def _finite_float(name: str, value: object) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite real")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be a finite real")
    return result


def _positive_float(name: str, value: object) -> float:
    result = _finite_float(name, value)
    if result <= 0.0:
        raise ValueError(f"{name} must be strictly positive")
    return result


def _readonly_float64(name: str, value: ArrayLike, *, ndim: int) -> FloatArray:
    original = np.asarray(value)
    if original.dtype.kind not in "iuf" or original.dtype.kind == "b":
        raise TypeError(f"{name} must use a real numeric dtype")
    array = np.ascontiguousarray(np.asarray(original, dtype=np.float64))
    if array.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    immutable = np.frombuffer(array.tobytes(order="C"), dtype=np.float64).reshape(
        array.shape
    )
    immutable.setflags(write=False)
    return immutable


def _exact_array_equal(left: object, right: object) -> bool:
    return (
        type(left) is np.ndarray
        and type(right) is np.ndarray
        and left.dtype == right.dtype
        and left.shape == right.shape
        and left.strides == right.strides
        and memoryview(left).cast("B") == memoryview(right).cast("B")
    )


def _hash_value(digest: Any, value: object) -> None:
    if value is None:
        digest.update(b"N")
    elif type(value) is bool:
        digest.update(b"B1" if value else b"B0")
    elif type(value) is int:
        digest.update(b"I" + str(value).encode("ascii") + b";")
    elif type(value) is float:
        digest.update(b"F" + value.hex().encode("ascii") + b";")
    elif type(value) is str:
        encoded = value.encode("utf-8")
        digest.update(b"S" + len(encoded).to_bytes(8, "big") + encoded)
    elif type(value) is tuple:
        digest.update(b"T" + len(value).to_bytes(8, "big"))
        for item in value:
            _hash_value(digest, item)
    elif type(value) is np.ndarray:
        digest.update(b"A")
        _hash_value(digest, value.dtype.str)
        _hash_value(digest, tuple(value.shape))
        _hash_value(digest, tuple(value.strides))
        digest.update(memoryview(value).cast("B"))
    elif is_dataclass(value) and not isinstance(value, type):
        digest.update(b"D")
        _hash_value(digest, type(value).__module__)
        _hash_value(digest, type(value).__qualname__)
        for field in fields(value):
            _hash_value(digest, field.name)
            _hash_value(digest, getattr(value, field.name))
    else:
        raise TypeError(f"unsupported digest value type: {type(value).__name__}")


def _digest(domain: str, *values: object) -> str:
    result = sha256(domain.encode("ascii"))
    for value in values:
        _hash_value(result, value)
    return result.hexdigest()


def _record_digest(domain: str, value: object, excluded: frozenset[str]) -> str:
    return _digest(
        domain,
        tuple(
            (field.name, getattr(value, field.name))
            for field in fields(value)
            if field.name not in excluded
        ),
    )


def _node_id(sheet_id: str, plane: int, span_index: int) -> str:
    return f"v5h10:{sheet_id}:plane:{plane}:span:{span_index}"


def _ring_id(sheet_id: str, release: int, cell: int) -> str:
    return f"v5h10:{sheet_id}:release:{release}:cell:{cell}"


def _canonical_edge_key(start_id: str | int, end_id: str | int) -> EdgeKey:
    if type(start_id) not in (str, int) or type(end_id) not in (str, int):
        raise TypeError("edge node IDs must be exact strings or integers")
    if start_id == end_id and type(start_id) is type(end_id):
        raise ValueError("an edge must connect distinct node IDs")
    start_key = ("integer", start_id) if type(start_id) is int else ("string", start_id)
    end_key = ("integer", end_id) if type(end_id) is int else ("string", end_id)
    return (start_id, end_id) if start_key < end_key else (end_id, start_id)


@dataclass(frozen=True, slots=True)
class ReleaseRow:
    """One complete conformal spanwise row of quadrilateral release cells."""

    sheet_id: str
    release_index: int
    source_time_s: float
    upstream_nodes: FloatArray
    downstream_nodes: FloatArray
    circulation_m2_s: FloatArray
    row_sha256: str


@dataclass(frozen=True, slots=True)
class RowParticleLineage:
    """Exact physical provenance for one globally canonical edge particle."""

    particle_id: tuple[Any, ...]
    birth_release_index: int
    role: ParticleRole
    source_edge: EdgeKey
    subdivision_index: int
    subdivision_count: int
    birth_incidences: tuple[EdgeIncidence, ...]
    update_incidences: tuple[EdgeIncidence, ...]
    physical_owner: str = RING_PHYSICAL_OWNER
    owner_state: str = DIAGNOSTIC_SHADOW_OWNER


@dataclass(frozen=True, slots=True, weakref_slot=True, eq=False)
class ReleaseRowState:
    """Immutable committed physical net cloud."""

    interface_id: str
    positions: FloatArray
    gamma: FloatArray
    sigma: FloatArray
    particle_ids: tuple[tuple[Any, ...], ...]
    lineage: tuple[RowParticleLineage, ...]
    rows: tuple[ReleaseRow, ...]
    row_parent_transport_digests: tuple[str | None, ...]
    live_boundary_indices_by_cell: tuple[tuple[int, ...], ...]
    live_boundary_nodes: FloatArray
    release_index: int
    state_epoch: int
    phase: OwnerPhase
    transport_generation: int
    transport_source_step_index: int | None
    transport_end_time_s: float | None
    parent_transport_digest: str | None
    parent_transport_attestation_sha256: str | None
    state_sha256: str
    clone_count: int = 0
    counter_particle_count: int = 0
    fresh_upstream_particle_count: int = 0


@dataclass(frozen=True, slots=True, weakref_slot=True, eq=False)
class ReleaseRowEvent:
    """One append-only atomic row-commit record."""

    proposal_id: str
    release_index: int
    changed_indices: tuple[int, ...]
    appended_indices: tuple[int, ...]
    parent_state_sha256: str
    parent_transport_digest: str
    parent_transport_event_sha256: str
    upstream_nodes_sha256: str
    committed_state_sha256: str
    row_sha256: str
    before_gamma_sha256: str
    added_gamma_sha256: str
    after_gamma_sha256: str
    operator_order: str
    global_graph_build_count: int
    clone_count: int
    counter_particle_count: int
    fresh_upstream_particle_count: int
    remesh_count: int
    ptera_call_count: int
    load_call_count: int
    feedback_call_count: int
    transport_call_count: int
    parent_owner_sha256: str
    previous_event_sha256: str
    event_sha256: str


@dataclass(frozen=True, slots=True, weakref_slot=True, eq=False)
class ReleaseRowTransportEvent:
    """Exact handoff from a committed zero-time row to transported parent state."""

    release_index: int
    source_step_index: int
    transport_end_time_s: float
    parent_state_sha256: str
    transported_state_sha256: str
    parent_transport_digest: str
    common_transport_attestation_sha256: str
    transported_arrays_sha256: str
    live_boundary_nodes_sha256: str
    previous_transport_event_sha256: str
    transport_event_sha256: str


@dataclass(frozen=True, slots=True, weakref_slot=True, eq=False)
class ReleaseRowCommonTransport:
    """Live capability defining one common material-tracer transport pack.

    The pack starts with every live particle support in exact
    cell/subdivision order and ends with the complete downstream frontier.
    Callers must advance this single pack in the same LSRK stages as the
    physical particle cloud before requesting an attestation.
    """

    parent_epoch: int
    parent_owner_sha256: str
    parent_state_sha256: str
    live_particle_indices_by_cell: tuple[tuple[int, ...], ...]
    live_particle_ids: tuple[tuple[Any, ...], ...]
    material_tracer_positions: FloatArray
    live_particle_sigma: FloatArray
    frontier_node_offset: int
    common_transport_sha256: str


@dataclass(frozen=True, slots=True, weakref_slot=True, eq=False)
class ReleaseRowTransportAttestation:
    """Same-process proof that particles and material tracers moved together."""

    common_transport_sha256: str
    parent_epoch: int
    parent_owner_sha256: str
    parent_state_sha256: str
    source_step_index: int
    transport_end_time_s: float
    transport_epoch: int
    transported_arrays_sha256: str
    transported_material_tracers_sha256: str
    transported_live_material_sigma_sha256: str
    transported_live_boundary_nodes: FloatArray
    attestation_sha256: str


@dataclass(frozen=True, slots=True, weakref_slot=True, eq=False)
class ReleaseRowOwner:
    """Versioned same-process capability for one global row sheet."""

    owner_id: str
    sheet_id: str
    particle_cap: int
    smoothing_radius_m: float
    target_spacing_m: float
    root_source_time_s: float
    release_dt_s: float
    epoch: int
    state: ReleaseRowState
    events: tuple[ReleaseRowEvent, ...]
    transport_events: tuple[ReleaseRowTransportEvent, ...]
    dependency_sha256: tuple[tuple[str, str], ...]
    owner_sha256: str


@dataclass(frozen=True, slots=True, weakref_slot=True, eq=False)
class ReleaseRowProposal:
    """Issued compatible plan or non-committable remesh classification."""

    status: ProposalStatus
    proposal_id: str
    parent_epoch: int
    parent_owner_sha256: str
    parent_state_sha256: str
    row_sha256: str
    planned_particle_count: int
    appended_particle_count: int
    changed_indices: tuple[int, ...]
    first_mismatch: str | None
    proposal_sha256: str
    clone_count: int = 0
    counter_particle_count: int = 0
    fresh_upstream_particle_count: int = 0


@dataclass(frozen=True, slots=True)
class RowCommitResult:
    """Convenience result; authoritative access remains through ``owner``."""

    committed: bool
    status: ProposalStatus
    owner: ReleaseRowOwner
    state: ReleaseRowState
    event: ReleaseRowEvent | None
    first_mismatch: str | None


@dataclass(frozen=True, slots=True)
class _EdgePlan:
    edge: EdgeLedger
    role: ParticleRole
    count: int


@dataclass(frozen=True, slots=True)
class _ProposalPlan:
    parent_owner: weakref.ReferenceType[ReleaseRowOwner]
    parent_state_sha256: str
    parent_transport_digest: str
    parent_transport_attestation_sha256: str
    parent_live_boundary_nodes: FloatArray
    row: ReleaseRow
    upstream_additions: tuple[
        tuple[int, FloatArray, float, float, float, tuple[EdgeIncidence, ...]], ...
    ]
    suffix_positions: FloatArray
    suffix_gamma: FloatArray
    suffix_sigma: FloatArray
    suffix_ids: tuple[tuple[Any, ...], ...]
    suffix_lineage: tuple[RowParticleLineage, ...]
    suffix_live_by_cell: tuple[tuple[int, ...], ...]
    issued_proposal: ReleaseRowProposal
    issued_sha256: str


@dataclass(frozen=True, slots=True)
class _CommonTransportPlan:
    parent_owner: weakref.ReferenceType[ReleaseRowOwner]
    issued_session: ReleaseRowCommonTransport
    issued_sha256: str


@dataclass(frozen=True, slots=True)
class _TransportAttestationPlan:
    parent_owner: weakref.ReferenceType[ReleaseRowOwner]
    common_transport: weakref.ReferenceType[ReleaseRowCommonTransport]
    transported_positions: FloatArray
    transported_gamma: FloatArray
    transported_sigma: FloatArray
    transported_material_tracers: FloatArray
    transported_live_material_sigma: FloatArray
    issued_attestation: ReleaseRowTransportAttestation
    issued_sha256: str


def _row_digest(value: ReleaseRow) -> str:
    return _record_digest(
        "fluxv-v5h10-release-row-v1", value, frozenset(("row_sha256",))
    )


def _state_digest(value: ReleaseRowState) -> str:
    return _record_digest(
        "fluxv-v5h10-row-state-v1", value, frozenset(("state_sha256",))
    )


def _event_digest(value: ReleaseRowEvent) -> str:
    return _record_digest(
        "fluxv-v5h10-row-event-v1", value, frozenset(("event_sha256",))
    )


def _transport_event_digest(value: ReleaseRowTransportEvent) -> str:
    return _record_digest(
        "fluxv-v5h10-row-transport-event-v1",
        value,
        frozenset(("transport_event_sha256",)),
    )


def _common_transport_digest(value: ReleaseRowCommonTransport) -> str:
    return _record_digest(
        "fluxv-v5h10-common-transport-v1",
        value,
        frozenset(("common_transport_sha256",)),
    )


def _transport_attestation_digest(value: ReleaseRowTransportAttestation) -> str:
    return _record_digest(
        "fluxv-v5h10-common-transport-attestation-v1",
        value,
        frozenset(("attestation_sha256",)),
    )


def _owner_digest(value: ReleaseRowOwner) -> str:
    return _record_digest(
        "fluxv-v5h10-row-owner-v1", value, frozenset(("owner_sha256",))
    )


def _proposal_digest(value: ReleaseRowProposal) -> str:
    return _record_digest(
        "fluxv-v5h10-row-proposal-v1", value, frozenset(("proposal_sha256",))
    )


def _proposal_matches_issued(
    proposal: ReleaseRowProposal, issued: ReleaseRowProposal
) -> bool:
    for field in fields(issued):
        actual = getattr(proposal, field.name)
        expected = getattr(issued, field.name)
        if type(actual) is not type(expected) or actual != expected:
            return False
    return True


def make_release_row(
    upstream_nodes: ArrayLike,
    downstream_nodes: ArrayLike,
    circulation_m2_s: ArrayLike,
    *,
    release_index: int,
    source_time_s: float,
    sheet_id: str,
) -> ReleaseRow:
    """Build one canonical row; at least two cells are required."""

    upstream = _readonly_float64("upstream_nodes", upstream_nodes, ndim=2)
    downstream = _readonly_float64("downstream_nodes", downstream_nodes, ndim=2)
    circulation = _readonly_float64("circulation_m2_s", circulation_m2_s, ndim=1)
    if upstream.shape[1:] != (3,) or downstream.shape != upstream.shape:
        raise ValueError("row nodes must have matching shape (span_nodes, 3)")
    if upstream.shape[0] < 3:
        raise ValueError("a release row must contain at least two cells")
    if circulation.shape != (upstream.shape[0] - 1,):
        raise ValueError("circulation must contain one value per spanwise cell")
    if np.any(circulation == 0.0):
        raise ValueError("every row circulation must be nonzero")
    release = _exact_positive_integer("release_index", release_index)
    source_time = _positive_float("source_time_s", source_time_s)
    sheet = _nonempty_string("sheet_id", sheet_id)
    for cell in range(circulation.shape[0]):
        points = (
            upstream[cell],
            upstream[cell + 1],
            downstream[cell + 1],
            downstream[cell],
        )
        if any(
            float(np.linalg.norm(points[(index + 1) % 4] - points[index])) <= 0.0
            for index in range(4)
        ):
            raise ValueError(f"row cell {cell} has a zero-length edge")
        if len({tuple(point) for point in points}) != 4:
            raise ValueError(f"row cell {cell} has repeated vertices")
        normal_a = np.cross(points[1] - points[0], points[3] - points[0])
        normal_b = np.cross(points[2] - points[3], points[2] - points[1])
        if (
            float(np.linalg.norm(normal_a)) <= 0.0
            or float(np.linalg.norm(normal_b)) <= 0.0
            or float(np.dot(normal_a, normal_b)) <= 0.0
        ):
            raise ValueError(f"row cell {cell} must have coherent nonzero area")
    row = ReleaseRow(
        sheet_id=sheet,
        release_index=release,
        source_time_s=source_time,
        upstream_nodes=upstream,
        downstream_nodes=downstream,
        circulation_m2_s=circulation,
        row_sha256="",
    )
    return replace(row, row_sha256=_row_digest(row))


def _validate_row(value: object, *, expected_release: int | None = None) -> ReleaseRow:
    if type(value) is not ReleaseRow:
        raise TypeError("row must be an exact ReleaseRow")
    rebuilt = make_release_row(
        value.upstream_nodes,
        value.downstream_nodes,
        value.circulation_m2_s,
        release_index=value.release_index,
        source_time_s=value.source_time_s,
        sheet_id=value.sheet_id,
    )
    if value.row_sha256 != rebuilt.row_sha256 or value.row_sha256 != _row_digest(value):
        raise ValueError("row digest is invalid")
    if not (
        _exact_array_equal(value.upstream_nodes, rebuilt.upstream_nodes)
        and _exact_array_equal(value.downstream_nodes, rebuilt.downstream_nodes)
        and _exact_array_equal(value.circulation_m2_s, rebuilt.circulation_m2_s)
    ):
        raise ValueError("row arrays are not canonical immutable float64 arrays")
    if expected_release is not None and value.release_index != expected_release:
        raise ValueError("row release index is not sequential")
    return value


def _row_graph(row: ReleaseRow) -> EdgeGraph:
    """Build the row's canonical edge incidence exactly once."""

    release = row.release_index
    nodes: list[BridgeNode] = []
    node_positions: dict[str, tuple[float, float, float]] = {}
    for span_index, position in enumerate(row.upstream_nodes):
        node_id = _node_id(row.sheet_id, release - 1, span_index)
        point = tuple(float(item) for item in position)
        nodes.append(BridgeNode(node_id, point))
        node_positions[node_id] = point
    for span_index, position in enumerate(row.downstream_nodes):
        node_id = _node_id(row.sheet_id, release, span_index)
        point = tuple(float(item) for item in position)
        nodes.append(BridgeNode(node_id, point))
        node_positions[node_id] = point

    buckets: dict[EdgeKey, list[EdgeIncidence]] = {}
    for cell in range(row.circulation_m2_s.shape[0]):
        ring = DirectedRing(
            ring_id=_ring_id(row.sheet_id, release, cell),
            node_ids=(
                _node_id(row.sheet_id, release - 1, cell),
                _node_id(row.sheet_id, release - 1, cell + 1),
                _node_id(row.sheet_id, release, cell + 1),
                _node_id(row.sheet_id, release, cell),
            ),
            circulation=float(row.circulation_m2_s[cell]),
        )
        ring_nodes = tuple(ring.node_ids)
        for traversal in range(4):
            source_start = ring_nodes[traversal]
            source_end = ring_nodes[(traversal + 1) % 4]
            key = _canonical_edge_key(source_start, source_end)
            sign = 1 if (source_start, source_end) == key else -1
            buckets.setdefault(key, []).append(
                EdgeIncidence(
                    ring_id=ring.ring_id,
                    traversal_index=traversal,
                    source_start_id=source_start,
                    source_end_id=source_end,
                    canonical_sign=sign,
                    ring_circulation=ring.circulation,
                    signed_circulation=sign * ring.circulation,
                )
            )
    ledgers: list[EdgeLedger] = []
    global_components: list[list[float]] = [[], [], []]
    for key in sorted(buckets):
        incidences = tuple(
            sorted(
                buckets[key], key=lambda item: (str(item.ring_id), item.traversal_index)
            )
        )
        if len(incidences) > 2 or (
            len(incidences) == 2
            and {item.canonical_sign for item in incidences} != {-1, 1}
        ):
            raise ValueError(
                "global row contains a non-manifold/co-oriented shared edge"
            )
        circulation = fsum(item.signed_circulation for item in incidences)
        if circulation == 0.0:
            circulation = 0.0
        start = np.asarray(node_positions[key[0]], dtype=np.float64)
        end = np.asarray(node_positions[key[1]], dtype=np.float64)
        vector = circulation * (end - start)
        vector_moment = tuple(float(item) for item in vector)
        for axis in range(3):
            global_components[axis].append(vector_moment[axis])
        ledgers.append(
            EdgeLedger(
                key=key,
                start_position=node_positions[key[0]],
                end_position=node_positions[key[1]],
                incidences=incidences,
                circulation=circulation,
                vector_moment=vector_moment,
            )
        )
    return EdgeGraph(
        nodes=tuple(sorted(nodes, key=lambda item: item.node_id)),
        edges=tuple(ledgers),
        incidence_residual=0.0,
        edge_reconstruction_residual=0.0,
        global_vector_moment=tuple(fsum(values) for values in global_components),
    )


def _row_edge_keys(row: ReleaseRow) -> tuple[tuple[EdgeKey, ...], tuple[EdgeKey, ...]]:
    release = row.release_index
    upstream = tuple(
        _canonical_edge_key(
            _node_id(row.sheet_id, release - 1, cell),
            _node_id(row.sheet_id, release - 1, cell + 1),
        )
        for cell in range(row.circulation_m2_s.shape[0])
    )
    downstream = tuple(
        _canonical_edge_key(
            _node_id(row.sheet_id, release, cell),
            _node_id(row.sheet_id, release, cell + 1),
        )
        for cell in range(row.circulation_m2_s.shape[0])
    )
    return upstream, downstream


def _edge_plans(
    row: ReleaseRow,
    graph: EdgeGraph,
    *,
    target_spacing_m: float,
    bootstrap: bool,
) -> tuple[_EdgePlan, ...]:
    upstream, downstream = _row_edge_keys(row)
    upstream_set = set(upstream)
    downstream_set = set(downstream)
    plans: list[_EdgePlan] = []
    for edge in graph.edges:
        if type(edge) is not EdgeLedger:
            raise TypeError("global row graph contains a foreign edge ledger")
        if not edge.retained:
            continue
        start = np.asarray(edge.start_position, dtype=np.float64)
        end = np.asarray(edge.end_position, dtype=np.float64)
        count = max(1, ceil(float(np.linalg.norm(end - start)) / target_spacing_m))
        if edge.key in upstream_set:
            if not bootstrap:
                continue
            role: ParticleRole = "initial_boundary"
        elif edge.key in downstream_set:
            role = "live_downstream"
        else:
            role = "side"
        plans.append(_EdgePlan(edge=edge, role=role, count=count))
    return tuple(plans)


def _particle_id(
    row: ReleaseRow, plan: _EdgePlan, subdivision_index: int
) -> tuple[Any, ...]:
    return (
        PARTICLE_SCHEMA,
        row.sheet_id,
        row.release_index,
        plan.role,
        plan.edge.key,
        subdivision_index,
        plan.count,
    )


def _deposit_plans(
    row: ReleaseRow,
    plans: tuple[_EdgePlan, ...],
    smoothing_radius_m: float,
) -> tuple[
    FloatArray,
    FloatArray,
    FloatArray,
    tuple[tuple[Any, ...], ...],
    tuple[RowParticleLineage, ...],
    dict[EdgeKey, tuple[int, ...]],
]:
    positions: list[np.ndarray] = []
    gamma: list[np.ndarray] = []
    sigma: list[float] = []
    particle_ids: list[tuple[Any, ...]] = []
    lineage: list[RowParticleLineage] = []
    by_edge: dict[EdgeKey, tuple[int, ...]] = {}
    for plan in plans:
        edge = plan.edge
        if any(type(item) is not EdgeIncidence for item in edge.incidences):
            raise TypeError("row edge incidence must use the exact bridge type")
        start = np.asarray(edge.start_position, dtype=np.float64)
        end = np.asarray(edge.end_position, dtype=np.float64)
        delta = (end - start) / plan.count
        first = len(positions)
        for subdivision_index in range(plan.count):
            particle_id = _particle_id(row, plan, subdivision_index)
            positions.append(start + (subdivision_index + 0.5) * delta)
            gamma.append(float(edge.circulation) * delta)
            sigma.append(smoothing_radius_m)
            particle_ids.append(particle_id)
            lineage.append(
                RowParticleLineage(
                    particle_id=particle_id,
                    birth_release_index=row.release_index,
                    role=plan.role,
                    source_edge=edge.key,
                    subdivision_index=subdivision_index,
                    subdivision_count=plan.count,
                    birth_incidences=edge.incidences,
                    update_incidences=(),
                )
            )
        by_edge[edge.key] = tuple(range(first, len(positions)))
    if positions:
        position_array = _readonly_float64("positions", np.vstack(positions), ndim=2)
        gamma_array = _readonly_float64("gamma", np.vstack(gamma), ndim=2)
        sigma_array = _readonly_float64("sigma", np.asarray(sigma), ndim=1)
    else:
        position_array = _readonly_float64("positions", np.empty((0, 3)), ndim=2)
        gamma_array = _readonly_float64("gamma", np.empty((0, 3)), ndim=2)
        sigma_array = _readonly_float64("sigma", np.empty((0,)), ndim=1)
    return (
        position_array,
        gamma_array,
        sigma_array,
        tuple(particle_ids),
        tuple(lineage),
        by_edge,
    )


def _state_from_parts(
    *,
    positions: ArrayLike,
    gamma: ArrayLike,
    sigma: ArrayLike,
    particle_ids: tuple[tuple[Any, ...], ...],
    lineage: tuple[RowParticleLineage, ...],
    rows: tuple[ReleaseRow, ...],
    row_parent_transport_digests: tuple[str | None, ...],
    live_boundary_indices_by_cell: tuple[tuple[int, ...], ...],
    live_boundary_nodes: ArrayLike,
    epoch: int,
    phase: OwnerPhase,
    transport_generation: int,
    transport_source_step_index: int | None = None,
    transport_end_time_s: float | None = None,
    parent_transport_digest: str | None = None,
    parent_transport_attestation_sha256: str | None = None,
) -> ReleaseRowState:
    state = ReleaseRowState(
        interface_id=INTERFACE_ID,
        positions=_readonly_float64("positions", positions, ndim=2),
        gamma=_readonly_float64("gamma", gamma, ndim=2),
        sigma=_readonly_float64("sigma", sigma, ndim=1),
        particle_ids=tuple(particle_ids),
        lineage=tuple(lineage),
        rows=tuple(rows),
        row_parent_transport_digests=tuple(row_parent_transport_digests),
        live_boundary_indices_by_cell=tuple(live_boundary_indices_by_cell),
        live_boundary_nodes=_readonly_float64(
            "live_boundary_nodes", live_boundary_nodes, ndim=2
        ),
        release_index=len(rows),
        state_epoch=epoch,
        phase=phase,
        transport_generation=transport_generation,
        transport_source_step_index=transport_source_step_index,
        transport_end_time_s=transport_end_time_s,
        parent_transport_digest=parent_transport_digest,
        parent_transport_attestation_sha256=parent_transport_attestation_sha256,
        state_sha256="",
        clone_count=0,
        counter_particle_count=0,
        fresh_upstream_particle_count=0,
    )
    return replace(state, state_sha256=_state_digest(state))


def _incidence_circulation(incidences: tuple[EdgeIncidence, ...]) -> float:
    return fsum(item.signed_circulation for item in incidences)


def _all_node_positions(rows: tuple[ReleaseRow, ...]) -> dict[str, np.ndarray]:
    positions: dict[str, np.ndarray] = {}
    first = rows[0]
    for span, point in enumerate(first.upstream_nodes):
        positions[_node_id(first.sheet_id, 0, span)] = point
    for row in rows:
        for span, point in enumerate(row.downstream_nodes):
            positions[_node_id(row.sheet_id, row.release_index, span)] = point
    return positions


def _validate_incidence(incidence: object, source_edge: EdgeKey) -> EdgeIncidence:
    if type(incidence) is not EdgeIncidence:
        raise TypeError("lineage incidence must use the exact EdgeIncidence type")
    if type(incidence.ring_id) not in (str, int) or type(incidence.ring_id) is bool:
        raise TypeError("lineage ring ID is invalid")
    if (
        type(incidence.traversal_index) is not int
        or not 0 <= incidence.traversal_index < 4
    ):
        raise ValueError("lineage traversal index is invalid")
    if type(incidence.source_start_id) not in (str, int) or type(
        incidence.source_end_id
    ) not in (str, int):
        raise TypeError("lineage incidence node IDs are invalid")
    if (
        incidence.canonical_sign not in (-1, 1)
        or type(incidence.canonical_sign) is not int
    ):
        raise ValueError("lineage incidence sign is invalid")
    if (
        _canonical_edge_key(incidence.source_start_id, incidence.source_end_id)
        != source_edge
    ):
        raise ValueError("lineage incidence references another edge")
    expected_sign = (
        1 if (incidence.source_start_id, incidence.source_end_id) == source_edge else -1
    )
    if incidence.canonical_sign != expected_sign:
        raise ValueError("lineage incidence canonical sign is inconsistent")
    ring_gamma = _finite_float("ring_circulation", incidence.ring_circulation)
    signed = _finite_float("signed_circulation", incidence.signed_circulation)
    if signed != expected_sign * ring_gamma:
        raise ValueError("lineage signed circulation is inconsistent")
    return incidence


def _validate_state_contents(value: object) -> ReleaseRowState:
    if type(value) is not ReleaseRowState:
        raise TypeError("state must be an exact ReleaseRowState")
    if value.interface_id != INTERFACE_ID or type(value.interface_id) is not str:
        raise ValueError("state interface ID is invalid")
    for name, array, ndim in (
        ("positions", value.positions, 2),
        ("gamma", value.gamma, 2),
        ("sigma", value.sigma, 1),
        ("live_boundary_nodes", value.live_boundary_nodes, 2),
    ):
        if type(array) is not np.ndarray or array.dtype != np.dtype(np.float64):
            raise TypeError(f"state {name} must be an exact float64 ndarray")
        if array.ndim != ndim or not array.flags.c_contiguous or array.flags.writeable:
            raise ValueError(f"state {name} must be immutable and C-contiguous")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"state {name} must be finite")
    count = value.positions.shape[0]
    if value.positions.shape != (count, 3) or value.gamma.shape != (count, 3):
        raise ValueError("state positions/gamma must have shape (n, 3)")
    if value.sigma.shape != (count,) or np.any(value.sigma <= 0.0):
        raise ValueError("state sigma must be positive with shape (n,)")
    if type(value.rows) is not tuple or not value.rows:
        raise ValueError("state rows must be a nonempty exact tuple")
    for release, row in enumerate(value.rows, start=1):
        _validate_row(row, expected_release=release)
    if (
        type(value.row_parent_transport_digests) is not tuple
        or len(value.row_parent_transport_digests) != len(value.rows)
        or value.row_parent_transport_digests[0] is not None
    ):
        raise ValueError("state row-parent transport ledger is invalid")
    for digest in value.row_parent_transport_digests[1:]:
        if (
            type(digest) is not str
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError("state row-parent transport digest is invalid")
    if value.release_index != len(value.rows) or value.state_epoch != len(value.rows):
        raise ValueError("state release, epoch, and row count disagree")
    if value.live_boundary_nodes.shape != value.rows[-1].downstream_nodes.shape:
        raise ValueError("state live boundary nodes have an invalid shape")
    generation = _exact_nonnegative_integer(
        "transport_generation", value.transport_generation
    )
    if value.phase == "post_commit_pre_transport":
        if generation != value.release_index - 1:
            raise ValueError("pre-transport state has an invalid transport generation")
        if any(
            item is not None
            for item in (
                value.transport_source_step_index,
                value.transport_end_time_s,
                value.parent_transport_digest,
                value.parent_transport_attestation_sha256,
            )
        ):
            raise ValueError("pre-transport state carries transported-parent facts")
        if not _exact_array_equal(
            value.live_boundary_nodes, value.rows[-1].downstream_nodes
        ):
            raise ValueError("pre-transport live nodes disagree with the committed row")
    elif value.phase == "post_transport":
        if generation != value.release_index:
            raise ValueError("post-transport state has an invalid transport generation")
        _exact_nonnegative_integer(
            "transport_source_step_index", value.transport_source_step_index
        )
        _positive_float("transport_end_time_s", value.transport_end_time_s)
        if (
            type(value.parent_transport_digest) is not str
            or len(value.parent_transport_digest) != 64
            or any(
                character not in "0123456789abcdef"
                for character in value.parent_transport_digest
            )
        ):
            raise ValueError("post-transport parent digest is invalid")
        if (
            type(value.parent_transport_attestation_sha256) is not str
            or len(value.parent_transport_attestation_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in value.parent_transport_attestation_sha256
            )
        ):
            raise ValueError("post-transport common attestation digest is invalid")
    else:
        raise ValueError("state owner phase is invalid")
    if (
        type(value.particle_ids) is not tuple
        or type(value.lineage) is not tuple
        or len(value.particle_ids) != count
        or len(value.lineage) != count
    ):
        raise ValueError("state arrays, IDs, and lineage disagree")
    if len(set(value.particle_ids)) != count:
        raise ValueError("state particle IDs must be unique")
    node_positions = _all_node_positions(value.rows)
    for index, record in enumerate(value.lineage):
        if type(record) is not RowParticleLineage:
            raise TypeError("state lineage must use exact RowParticleLineage")
        if record.particle_id != value.particle_ids[index]:
            raise ValueError("lineage particle ID disagrees with state")
        if record.role not in (
            "initial_boundary",
            "side",
            "interior_boundary",
            "live_downstream",
        ):
            raise ValueError("lineage role is invalid")
        if (
            record.physical_owner != RING_PHYSICAL_OWNER
            or record.owner_state != DIAGNOSTIC_SHADOW_OWNER
        ):
            raise ValueError("lineage ownership is invalid")
        if type(record.source_edge) is not tuple or len(record.source_edge) != 2:
            raise TypeError("lineage source edge is invalid")
        if type(record.birth_incidences) is not tuple or not record.birth_incidences:
            raise ValueError("lineage needs birth incidences")
        if type(record.update_incidences) is not tuple:
            raise TypeError("lineage update incidences must be an exact tuple")
        for incidence in record.birth_incidences + record.update_incidences:
            _validate_incidence(incidence, record.source_edge)
        if not 0 <= record.subdivision_index < record.subdivision_count:
            raise ValueError("lineage subdivision is invalid")
        # Before the first transport the geometry-derived mechanical closure is
        # independently reconstructible.  Thereafter x/gamma/sigma are material
        # transport facts bound by a transport event, not by birth geometry.
        if value.transport_generation == 0:
            try:
                start = node_positions[record.source_edge[0]]
                end = node_positions[record.source_edge[1]]
            except KeyError as error:
                raise ValueError(
                    "lineage edge references an unknown state node"
                ) from error
            delta = (end - start) / record.subdivision_count
            expected_position = start + (record.subdivision_index + 0.5) * delta
            total_circulation = _incidence_circulation(
                record.birth_incidences + record.update_incidences
            )
            expected_gamma = total_circulation * delta
            if not np.array_equal(value.positions[index], expected_position):
                raise ValueError("particle support disagrees with exact lineage")
            tolerance = (
                16.0
                * np.finfo(np.float64).eps
                * max(1.0, float(np.max(np.abs(expected_gamma))))
            )
            if float(np.max(np.abs(value.gamma[index] - expected_gamma))) > tolerance:
                raise ValueError("particle gamma disagrees with exact lineage")
    live = value.live_boundary_indices_by_cell
    if type(live) is not tuple or len(live) != value.rows[-1].circulation_m2_s.size:
        raise ValueError("state live boundary must contain one slice per cell")
    flattened: list[int] = []
    for cell, indices in enumerate(live):
        if type(indices) is not tuple or not indices:
            raise ValueError("every live cell must contain a nonempty exact tuple")
        expected_edge = _row_edge_keys(value.rows[-1])[1][cell]
        for index in indices:
            if type(index) is not int or not 0 <= index < count:
                raise ValueError("live boundary index is invalid")
            record = value.lineage[index]
            if record.role != "live_downstream" or record.source_edge != expected_edge:
                raise ValueError(
                    "live boundary slice is not the latest downstream edge"
                )
            flattened.append(index)
    if len(set(flattened)) != len(flattened):
        raise ValueError("live boundary cells overlap")
    for name in (
        "clone_count",
        "counter_particle_count",
        "fresh_upstream_particle_count",
    ):
        if _exact_nonnegative_integer(name, getattr(value, name)) != 0:
            raise ValueError("physical row state must contain zero gross particles")
    if value.state_sha256 != _state_digest(value):
        raise ValueError("state digest is invalid")
    return value


def _validate_event_contents(event: object, owner: ReleaseRowOwner) -> ReleaseRowEvent:
    if type(event) is not ReleaseRowEvent:
        raise TypeError("owner events must be exact ReleaseRowEvent objects")
    if event.operator_order != OPERATOR_ORDER or event.global_graph_build_count != 1:
        raise ValueError("event operator/global-row contract is invalid")
    for name in (
        "clone_count",
        "counter_particle_count",
        "fresh_upstream_particle_count",
        "remesh_count",
        "ptera_call_count",
        "load_call_count",
        "feedback_call_count",
        "transport_call_count",
    ):
        if _exact_nonnegative_integer(name, getattr(event, name)) != 0:
            raise ValueError("mechanical owner event counters must be zero")
    if event.event_sha256 != _event_digest(event):
        raise ValueError("event digest is invalid")
    if event.release_index < 2 or event.release_index > owner.epoch:
        raise ValueError("event release index is invalid")
    return event


def _validate_transport_event_contents(
    event: object, owner: ReleaseRowOwner
) -> ReleaseRowTransportEvent:
    if type(event) is not ReleaseRowTransportEvent:
        raise TypeError("transport events must be exact ReleaseRowTransportEvent")
    if not 1 <= event.release_index <= owner.epoch:
        raise ValueError("transport event release index is invalid")
    _exact_nonnegative_integer("source_step_index", event.source_step_index)
    _positive_float("transport_end_time_s", event.transport_end_time_s)
    for name in (
        "parent_state_sha256",
        "transported_state_sha256",
        "parent_transport_digest",
        "common_transport_attestation_sha256",
        "transported_arrays_sha256",
        "live_boundary_nodes_sha256",
        "previous_transport_event_sha256",
        "transport_event_sha256",
    ):
        value = getattr(event, name)
        if (
            type(value) is not str
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError(f"transport event {name} is not a SHA-256")
    if event.transport_event_sha256 != _transport_event_digest(event):
        raise ValueError("transport event digest is invalid")
    return event


def _validate_owner_contents(value: object) -> ReleaseRowOwner:
    if type(value) is not ReleaseRowOwner:
        raise TypeError("owner must be an exact ReleaseRowOwner")
    _nonempty_string("owner_id", value.owner_id)
    _nonempty_string("sheet_id", value.sheet_id)
    cap = _exact_positive_integer("particle_cap", value.particle_cap)
    _positive_float("smoothing_radius_m", value.smoothing_radius_m)
    spacing = _positive_float("target_spacing_m", value.target_spacing_m)
    if value.smoothing_radius_m / spacing < FROZEN_OVERLAP_LAMBDA:
        raise ValueError("owner violates the frozen minimum overlap")
    _positive_float("root_source_time_s", value.root_source_time_s)
    _positive_float("release_dt_s", value.release_dt_s)
    if type(value.epoch) is not int or value.epoch < 1:
        raise ValueError("owner epoch is invalid")
    state = _validate_state_contents(value.state)
    if state.state_epoch != value.epoch or state.rows[-1].sheet_id != value.sheet_id:
        raise ValueError("owner state identity disagrees with owner")
    if state.positions.shape[0] > cap:
        raise ValueError("owner particle cap is smaller than committed state")
    if value.dependency_sha256 != DEPENDENCY_SHA256:
        raise ValueError("owner dependency digest is invalid")
    if type(value.events) is not tuple or len(value.events) != value.epoch - 1:
        raise ValueError("owner event history disagrees with epoch")
    previous = _GENESIS_SHA256
    for release, event in enumerate(value.events, start=2):
        _validate_event_contents(event, value)
        if event.release_index != release or event.previous_event_sha256 != previous:
            raise ValueError("owner event chain is invalid")
        if event.row_sha256 != value.state.rows[release - 1].row_sha256:
            raise ValueError("owner event does not bind its row")
        previous = event.event_sha256
    if type(value.transport_events) is not tuple or len(value.transport_events) != (
        state.transport_generation
    ):
        raise ValueError("owner transport history disagrees with state")
    previous_transport = _GENESIS_SHA256
    for release, event in enumerate(value.transport_events, start=1):
        _validate_transport_event_contents(event, value)
        if (
            event.release_index != release
            or event.previous_transport_event_sha256 != previous_transport
        ):
            raise ValueError("owner transport event chain is invalid")
        previous_transport = event.transport_event_sha256
    for release, event in enumerate(value.events, start=2):
        parent_transport = value.transport_events[release - 2]
        row = state.rows[release - 1]
        if (
            event.parent_state_sha256 != parent_transport.transported_state_sha256
            or event.parent_transport_digest != parent_transport.parent_transport_digest
            or event.parent_transport_event_sha256
            != parent_transport.transport_event_sha256
            or event.upstream_nodes_sha256
            != parent_transport.live_boundary_nodes_sha256
            or event.upstream_nodes_sha256
            != _digest("fluxv-v5h10-transported-live-nodes-v1", row.upstream_nodes)
            or state.row_parent_transport_digests[release - 1]
            != parent_transport.parent_transport_digest
        ):
            raise ValueError("release row is not bound to its transported parent")
    if state.phase == "post_commit_pre_transport":
        if (
            value.events
            and value.events[-1].committed_state_sha256 != state.state_sha256
        ):
            raise ValueError("owner latest release event does not bind its state")
    else:
        latest_transport = value.transport_events[-1]
        if (
            latest_transport.transported_state_sha256 != state.state_sha256
            or latest_transport.common_transport_attestation_sha256
            != state.parent_transport_attestation_sha256
        ):
            raise ValueError("owner latest transport event does not bind its state")
        if value.events:
            committed_sha = value.events[-1].committed_state_sha256
        else:
            committed_sha = latest_transport.parent_state_sha256
        if latest_transport.parent_state_sha256 != committed_sha:
            raise ValueError("latest transport does not bind the committed row state")
    if value.owner_sha256 != _owner_digest(value):
        raise ValueError("owner digest is invalid")
    return value


_LOCK = RLock()
_LIVE_STATES: WeakKeyDictionary[ReleaseRowState, str] = WeakKeyDictionary()
_LIVE_EVENTS: WeakKeyDictionary[ReleaseRowEvent, str] = WeakKeyDictionary()
_LIVE_TRANSPORT_EVENTS: WeakKeyDictionary[
    ReleaseRowTransportEvent, str
] = WeakKeyDictionary()
_LIVE_OWNERS: WeakKeyDictionary[ReleaseRowOwner, str] = WeakKeyDictionary()
_CURRENT_OWNERS: dict[str, ReleaseRowOwner] = {}
_LIVE_PROPOSALS: WeakKeyDictionary[
    ReleaseRowProposal, _ProposalPlan
] = WeakKeyDictionary()
_LIVE_COMMON_TRANSPORTS: WeakKeyDictionary[
    ReleaseRowCommonTransport, _CommonTransportPlan
] = WeakKeyDictionary()
_LIVE_TRANSPORT_ATTESTATIONS: WeakKeyDictionary[
    ReleaseRowTransportAttestation, _TransportAttestationPlan
] = WeakKeyDictionary()
_CURRENT_COMMON_TRANSPORTS: dict[tuple[str, int, str], ReleaseRowCommonTransport] = {}
_CURRENT_TRANSPORT_ATTESTATIONS: dict[
    ReleaseRowCommonTransport, ReleaseRowTransportAttestation
] = {}
_CONSUMED_PROPOSALS: WeakSet[ReleaseRowProposal] = WeakSet()
_CONSUMED_COMMON_TRANSPORTS: WeakSet[ReleaseRowCommonTransport] = WeakSet()
_CONSUMED_TRANSPORT_ATTESTATIONS: WeakSet[ReleaseRowTransportAttestation] = WeakSet()
_CONSUMED_GENERATIONS: set[tuple[str, int, str]] = set()


def _generation_key(owner: ReleaseRowOwner) -> tuple[str, int, str]:
    return owner.owner_id, owner.epoch, owner.owner_sha256


def _validate_live_owner(value: object, *, require_current: bool) -> ReleaseRowOwner:
    owner = _validate_owner_contents(value)
    with _LOCK:
        if _LIVE_OWNERS.get(owner) != owner.owner_sha256:
            raise RuntimeError("owner is not a live issued capability")
        if _LIVE_STATES.get(owner.state) != owner.state.state_sha256:
            raise RuntimeError("owner state is not live-attested")
        if any(_LIVE_EVENTS.get(event) != event.event_sha256 for event in owner.events):
            raise RuntimeError("owner event is not live-attested")
        if any(
            _LIVE_TRANSPORT_EVENTS.get(event) != event.transport_event_sha256
            for event in owner.transport_events
        ):
            raise RuntimeError("owner transport event is not live-attested")
        if require_current and _CURRENT_OWNERS.get(owner.owner_id) is not owner:
            raise RuntimeError("owner generation is stale or already consumed")
    return owner


def validate_release_row_owner(value: object) -> ReleaseRowOwner:
    """Validate exact contents and issued identity, including historical owners."""

    _assert_dependencies()
    return _validate_live_owner(value, require_current=False)


def validate_current_release_row_owner(value: object) -> ReleaseRowOwner:
    """Validate the unique current owner generation for live coupling use."""

    _assert_dependencies()
    return _validate_live_owner(value, require_current=True)


def bootstrap_release_row_owner(
    row: object,
    *,
    smoothing_radius_m: float,
    target_spacing_m: float,
    release_dt_s: float,
    particle_cap: int = 100_000,
    owner_id: str,
) -> ReleaseRowOwner:
    """Atomically create epoch one from one complete global row."""

    _assert_dependencies()
    first = _validate_row(row, expected_release=1)
    sigma = _positive_float("smoothing_radius_m", smoothing_radius_m)
    spacing = _positive_float("target_spacing_m", target_spacing_m)
    if sigma / spacing < FROZEN_OVERLAP_LAMBDA:
        raise ValueError("target spacing violates the frozen minimum overlap")
    release_dt = _positive_float("release_dt_s", release_dt_s)
    cap = _exact_positive_integer("particle_cap", particle_cap)
    identifier = _nonempty_string("owner_id", owner_id)
    graph = _row_graph(first)
    plans = _edge_plans(first, graph, target_spacing_m=spacing, bootstrap=True)
    planned_count = sum(plan.count for plan in plans)
    if planned_count > cap:
        raise RuntimeError("v5h10 particle cap exceeded before materialization")
    positions, gamma, radii, ids, lineage, by_edge = _deposit_plans(first, plans, sigma)
    downstream_keys = _row_edge_keys(first)[1]
    live = tuple(by_edge[key] for key in downstream_keys)
    state = _state_from_parts(
        positions=positions,
        gamma=gamma,
        sigma=radii,
        particle_ids=ids,
        lineage=lineage,
        rows=(first,),
        row_parent_transport_digests=(None,),
        live_boundary_indices_by_cell=live,
        live_boundary_nodes=first.downstream_nodes,
        epoch=1,
        phase="post_commit_pre_transport",
        transport_generation=0,
    )
    owner = ReleaseRowOwner(
        owner_id=identifier,
        sheet_id=first.sheet_id,
        particle_cap=cap,
        smoothing_radius_m=sigma,
        target_spacing_m=spacing,
        root_source_time_s=first.source_time_s,
        release_dt_s=release_dt,
        epoch=1,
        state=state,
        events=(),
        transport_events=(),
        dependency_sha256=DEPENDENCY_SHA256,
        owner_sha256="",
    )
    owner = replace(owner, owner_sha256=_owner_digest(owner))
    _validate_owner_contents(owner)
    with _LOCK:
        if identifier in _CURRENT_OWNERS:
            raise RuntimeError("owner_id already has a live current generation")
        try:
            _LIVE_STATES[state] = state.state_sha256
            _LIVE_OWNERS[owner] = owner.owner_sha256
            _CURRENT_OWNERS[identifier] = owner
        except BaseException:
            _LIVE_STATES.pop(state, None)
            _LIVE_OWNERS.pop(owner, None)
            if _CURRENT_OWNERS.get(identifier) is owner:
                _CURRENT_OWNERS.pop(identifier, None)
            raise
    return validate_release_row_owner(owner)


def _remesh_proposal(
    owner: ReleaseRowOwner,
    row: ReleaseRow,
    *,
    proposal_id: str,
    mismatch: str,
    planned_count: int,
) -> ReleaseRowProposal:
    proposal = ReleaseRowProposal(
        status="remesh_required",
        proposal_id=proposal_id,
        parent_epoch=owner.epoch,
        parent_owner_sha256=owner.owner_sha256,
        parent_state_sha256=owner.state.state_sha256,
        row_sha256=row.row_sha256,
        planned_particle_count=planned_count,
        appended_particle_count=max(0, planned_count - owner.state.positions.shape[0]),
        changed_indices=(),
        first_mismatch=mismatch,
        proposal_sha256="",
    )
    return replace(proposal, proposal_sha256=_proposal_digest(proposal))


def _edge_arrays(
    edge: EdgeLedger, count: int, sigma: float
) -> tuple[FloatArray, FloatArray, FloatArray]:
    start = np.asarray(edge.start_position, dtype=np.float64)
    end = np.asarray(edge.end_position, dtype=np.float64)
    delta = (end - start) / count
    positions = np.vstack([start + (index + 0.5) * delta for index in range(count)])
    gamma = np.vstack([float(edge.circulation) * delta for _ in range(count)])
    radii = np.full(count, sigma, dtype=np.float64)
    return (
        _readonly_float64("edge positions", positions, ndim=2),
        _readonly_float64("edge gamma", gamma, ndim=2),
        _readonly_float64("edge sigma", radii, ndim=1),
    )


def propose_release_row_update(
    owner: object,
    row: object,
    *,
    proposal_id: str,
) -> ReleaseRowProposal:
    """Issue one exact-compatible global-row update or ``remesh_required``."""

    _assert_dependencies()
    parent = _validate_live_owner(owner, require_current=True)
    identifier = _nonempty_string("proposal_id", proposal_id)
    candidate = _validate_row(row)
    expected_release = parent.epoch + 1
    mismatch: str | None = None
    if parent.state.phase != "post_transport":
        mismatch = "transport_required"
    elif candidate.sheet_id != parent.sheet_id:
        mismatch = "sheet_id"
    elif candidate.release_index != expected_release:
        mismatch = "release_index"
    elif candidate.source_time_s != (
        parent.root_source_time_s + (expected_release - 1) * parent.release_dt_s
    ):
        mismatch = "source_time_s"
    elif candidate.upstream_nodes.shape != parent.state.live_boundary_nodes.shape:
        mismatch = "span_cell_count"
    elif not _exact_array_equal(
        candidate.upstream_nodes, parent.state.live_boundary_nodes
    ):
        mismatch = "support"
    elif np.any(
        np.signbit(candidate.circulation_m2_s)
        != np.signbit(parent.state.rows[-1].circulation_m2_s)
    ):
        mismatch = "circulation_sign"
    elif np.any(
        np.abs(candidate.circulation_m2_s)
        <= np.abs(parent.state.rows[-1].circulation_m2_s)
    ):
        mismatch = "circulation_lifecycle"

    graph = _row_graph(candidate)
    suffix_plans = _edge_plans(
        candidate,
        graph,
        target_spacing_m=parent.target_spacing_m,
        bootstrap=False,
    )
    planned_count = parent.state.positions.shape[0] + sum(
        plan.count for plan in suffix_plans
    )
    if mismatch is None and planned_count > parent.particle_cap:
        mismatch = "particle_cap"
    if mismatch is not None:
        return _remesh_proposal(
            parent,
            candidate,
            proposal_id=identifier,
            mismatch=mismatch,
            planned_count=planned_count,
        )

    # The committed-net cap is now closed.  Only after that gate may exact
    # material-basis additions and the physical suffix be materialized.
    upstream_keys, downstream_keys = _row_edge_keys(candidate)
    edge_by_key = {edge.key: edge for edge in graph.retained_edges}
    additions: list[
        tuple[int, FloatArray, float, float, float, tuple[EdgeIncidence, ...]]
    ] = []
    changed: list[int] = []
    for cell, key in enumerate(upstream_keys):
        edge = edge_by_key.get(key)
        old_indices = parent.state.live_boundary_indices_by_cell[cell]
        if edge is None or len(old_indices) == 0:
            mismatch = "upstream_edge"
            break
        if len(edge.incidences) != 1 or type(edge.incidences[0]) is not EdgeIncidence:
            mismatch = "upstream_incidence"
            break
        authoritative_new = _incidence_circulation(edge.incidences)
        if authoritative_new == 0.0 or authoritative_new != edge.circulation:
            mismatch = "upstream_incidence"
            break
        old_net_circulation: float | None = None
        for old_index in old_indices:
            record = parent.state.lineage[old_index]
            if record.source_edge != key:
                mismatch = "lineage"
                break
            authoritative_old = _incidence_circulation(
                record.birth_incidences + record.update_incidences
            )
            if authoritative_old == 0.0:
                mismatch = "upstream_incidence"
                break
            if old_net_circulation is None:
                old_net_circulation = authoritative_old
            elif authoritative_old != old_net_circulation:
                mismatch = "upstream_incidence"
                break
            # Transport may curve and stretch the material subdivision.  Its
            # current vector is the authoritative transported basis scaled by
            # the old net incidence; no endpoint-chord reconstruction is
            # scientifically valid here.
            delta_gamma = _readonly_float64(
                "upstream additive gamma",
                authoritative_new * (parent.state.gamma[old_index] / authoritative_old),
                ndim=1,
            )
            additions.append(
                (
                    old_index,
                    delta_gamma,
                    authoritative_old,
                    authoritative_new,
                    fsum((authoritative_old, authoritative_new)),
                    edge.incidences,
                )
            )
            changed.append(old_index)
        if mismatch is not None:
            break
    if mismatch is not None:
        return _remesh_proposal(
            parent,
            candidate,
            proposal_id=identifier,
            mismatch=mismatch,
            planned_count=planned_count,
        )

    (
        suffix_positions,
        suffix_gamma,
        suffix_sigma,
        suffix_ids,
        suffix_lineage,
        by_edge,
    ) = _deposit_plans(candidate, suffix_plans, parent.smoothing_radius_m)
    prefix_count = parent.state.positions.shape[0]
    suffix_live = tuple(
        tuple(prefix_count + index for index in by_edge[key]) for key in downstream_keys
    )
    proposal = ReleaseRowProposal(
        status="compatible",
        proposal_id=identifier,
        parent_epoch=parent.epoch,
        parent_owner_sha256=parent.owner_sha256,
        parent_state_sha256=parent.state.state_sha256,
        row_sha256=candidate.row_sha256,
        planned_particle_count=planned_count,
        appended_particle_count=suffix_positions.shape[0],
        changed_indices=tuple(changed),
        first_mismatch=None,
        proposal_sha256="",
    )
    proposal = replace(proposal, proposal_sha256=_proposal_digest(proposal))
    plan = _ProposalPlan(
        parent_owner=weakref.ref(parent),
        parent_state_sha256=parent.state.state_sha256,
        parent_transport_digest=parent.state.parent_transport_digest,
        parent_transport_attestation_sha256=(
            parent.state.parent_transport_attestation_sha256
        ),
        parent_live_boundary_nodes=_readonly_float64(
            "proposal parent live boundary nodes",
            parent.state.live_boundary_nodes,
            ndim=2,
        ),
        row=candidate,
        upstream_additions=tuple(additions),
        suffix_positions=suffix_positions,
        suffix_gamma=suffix_gamma,
        suffix_sigma=suffix_sigma,
        suffix_ids=suffix_ids,
        suffix_lineage=suffix_lineage,
        suffix_live_by_cell=suffix_live,
        issued_proposal=replace(proposal),
        issued_sha256=proposal.proposal_sha256,
    )
    with _LOCK:
        if _CURRENT_OWNERS.get(parent.owner_id) is not parent:
            raise RuntimeError("owner generation changed during proposal")
        _LIVE_PROPOSALS[proposal] = plan
    return proposal


def _validate_common_transport(
    value: object,
    *,
    require_current: bool,
) -> tuple[ReleaseRowCommonTransport, _CommonTransportPlan, ReleaseRowOwner]:
    if type(value) is not ReleaseRowCommonTransport:
        raise TypeError("common_transport must be an exact ReleaseRowCommonTransport")
    if value.common_transport_sha256 != _common_transport_digest(value):
        raise ValueError("common transport digest is invalid")
    with _LOCK:
        plan = _LIVE_COMMON_TRANSPORTS.get(value)
        if (
            plan is None
            or plan.issued_session is not value
            or plan.issued_sha256 != value.common_transport_sha256
            or value in _CONSUMED_COMMON_TRANSPORTS
        ):
            raise RuntimeError("common transport is stale, forged, or not live")
        parent = plan.parent_owner()
        if parent is None:
            raise RuntimeError("common transport parent is no longer live")
        parent = _validate_live_owner(parent, require_current=require_current)
        generation = _generation_key(parent)
        if (
            value.parent_epoch != parent.epoch
            or value.parent_owner_sha256 != parent.owner_sha256
            or value.parent_state_sha256 != parent.state.state_sha256
            or value.live_particle_indices_by_cell
            != parent.state.live_boundary_indices_by_cell
            or value.frontier_node_offset != len(value.live_particle_ids)
            or (
                require_current
                and _CURRENT_COMMON_TRANSPORTS.get(generation) is not value
            )
        ):
            raise RuntimeError("common transport parent binding is invalid")
    flat = tuple(
        index
        for indices in parent.state.live_boundary_indices_by_cell
        for index in indices
    )
    expected_ids = tuple(parent.state.particle_ids[index] for index in flat)
    expected_positions = _readonly_float64(
        "expected material tracer positions",
        np.vstack(
            (
                parent.state.positions[np.asarray(flat, dtype=np.int64)],
                parent.state.live_boundary_nodes,
            )
        ),
        ndim=2,
    )
    expected_sigma = _readonly_float64(
        "expected live particle sigma",
        parent.state.sigma[np.asarray(flat, dtype=np.int64)],
        ndim=1,
    )
    if (
        value.live_particle_ids != expected_ids
        or not _exact_array_equal(value.material_tracer_positions, expected_positions)
        or not _exact_array_equal(value.live_particle_sigma, expected_sigma)
    ):
        raise RuntimeError("common transport material pack binding is invalid")
    return value, plan, parent


def validate_release_row_common_transport(
    value: object,
) -> ReleaseRowCommonTransport:
    """Validate a live current common-transport session capability."""

    _assert_dependencies()
    session, _, _ = _validate_common_transport(value, require_current=True)
    return session


def begin_release_row_common_transport(
    owner: object,
    committed_state: object,
) -> ReleaseRowCommonTransport:
    """Issue the exact material-support/frontier tracer pack for one transport."""

    _assert_dependencies()
    parent = _validate_live_owner(owner, require_current=True)
    if committed_state is not parent.state:
        raise RuntimeError("common transport parent state is stale or not owned")
    if parent.state.phase != "post_commit_pre_transport":
        raise RuntimeError("common transport requires an untransported committed row")
    flat = tuple(
        index
        for indices in parent.state.live_boundary_indices_by_cell
        for index in indices
    )
    flat_array = np.asarray(flat, dtype=np.int64)
    supports = parent.state.positions[flat_array]
    tracers = _readonly_float64(
        "material_tracer_positions",
        np.vstack((supports, parent.state.live_boundary_nodes)),
        ndim=2,
    )
    live_sigma = _readonly_float64(
        "live_particle_sigma", parent.state.sigma[flat_array], ndim=1
    )
    session = ReleaseRowCommonTransport(
        parent_epoch=parent.epoch,
        parent_owner_sha256=parent.owner_sha256,
        parent_state_sha256=parent.state.state_sha256,
        live_particle_indices_by_cell=parent.state.live_boundary_indices_by_cell,
        live_particle_ids=tuple(parent.state.particle_ids[index] for index in flat),
        material_tracer_positions=tracers,
        live_particle_sigma=live_sigma,
        frontier_node_offset=len(flat),
        common_transport_sha256="",
    )
    session = replace(
        session, common_transport_sha256=_common_transport_digest(session)
    )
    plan = _CommonTransportPlan(
        parent_owner=weakref.ref(parent),
        issued_session=session,
        issued_sha256=session.common_transport_sha256,
    )
    generation = _generation_key(parent)
    with _LOCK:
        if _CURRENT_OWNERS.get(parent.owner_id) is not parent:
            raise RuntimeError("owner generation changed during transport begin")
        if generation in _CURRENT_COMMON_TRANSPORTS:
            raise RuntimeError(
                "common transport was already issued for this generation"
            )
        try:
            _LIVE_COMMON_TRANSPORTS[session] = plan
            _CURRENT_COMMON_TRANSPORTS[generation] = session
        except BaseException:
            _LIVE_COMMON_TRANSPORTS.pop(session, None)
            if _CURRENT_COMMON_TRANSPORTS.get(generation) is session:
                _CURRENT_COMMON_TRANSPORTS.pop(generation, None)
            raise
    return validate_release_row_common_transport(session)


def _validate_transport_attestation(
    value: object,
    *,
    require_current: bool,
) -> tuple[
    ReleaseRowTransportAttestation,
    _TransportAttestationPlan,
    ReleaseRowCommonTransport,
    ReleaseRowOwner,
]:
    if type(value) is not ReleaseRowTransportAttestation:
        raise TypeError(
            "common_transport_attestation must be an exact "
            "ReleaseRowTransportAttestation"
        )
    if value.attestation_sha256 != _transport_attestation_digest(value):
        raise ValueError("common transport attestation digest is invalid")
    with _LOCK:
        plan = _LIVE_TRANSPORT_ATTESTATIONS.get(value)
        if (
            plan is None
            or plan.issued_attestation is not value
            or plan.issued_sha256 != value.attestation_sha256
            or value in _CONSUMED_TRANSPORT_ATTESTATIONS
        ):
            raise RuntimeError(
                "common transport attestation is stale, forged, or not live"
            )
        session = plan.common_transport()
        if session is None:
            raise RuntimeError("common transport session is no longer live")
        session, _, parent = _validate_common_transport(
            session, require_current=require_current
        )
        if _CURRENT_TRANSPORT_ATTESTATIONS.get(session) is not value:
            raise RuntimeError("common transport attestation is not current")
    if (
        value.common_transport_sha256 != session.common_transport_sha256
        or value.parent_epoch != parent.epoch
        or value.parent_owner_sha256 != parent.owner_sha256
        or value.parent_state_sha256 != parent.state.state_sha256
        or value.transport_epoch != parent.epoch
        or value.transported_arrays_sha256
        != _digest(
            "fluxv-v5h10-transported-arrays-v1",
            plan.transported_positions,
            plan.transported_gamma,
            plan.transported_sigma,
        )
        or value.transported_material_tracers_sha256
        != _digest(
            "fluxv-v5h10-transported-material-tracers-v1",
            plan.transported_material_tracers,
        )
        or value.transported_live_material_sigma_sha256
        != _digest(
            "fluxv-v5h10-transported-live-material-sigma-v1",
            plan.transported_live_material_sigma,
        )
        or not _exact_array_equal(
            value.transported_live_boundary_nodes,
            plan.transported_material_tracers[session.frontier_node_offset :],
        )
    ):
        raise RuntimeError("common transport attestation binding is invalid")
    return value, plan, session, parent


def validate_release_row_transport_attestation(
    value: object,
) -> ReleaseRowTransportAttestation:
    """Validate a live current common-transport attestation capability."""

    _assert_dependencies()
    attestation, _, _, _ = _validate_transport_attestation(value, require_current=True)
    return attestation


def attest_release_row_common_transport(
    common_transport: object,
    transported_positions: ArrayLike,
    transported_gamma: ArrayLike,
    transported_sigma: ArrayLike,
    transported_material_tracer_positions: ArrayLike,
    transported_live_material_sigma: ArrayLike,
    *,
    source_step_index: int,
    transport_end_time_s: float,
    transport_epoch: int,
) -> ReleaseRowTransportAttestation:
    """Bind one exact same-stage material-tracer result to its particle cloud."""

    _assert_dependencies()
    session, _, parent = _validate_common_transport(
        common_transport, require_current=True
    )
    step = _exact_nonnegative_integer("source_step_index", source_step_index)
    if (
        parent.transport_events
        and step != parent.transport_events[-1].source_step_index + 1
    ):
        raise ValueError("transport source step is not sequential")
    end_time = _positive_float("transport_end_time_s", transport_end_time_s)
    expected_end = parent.state.rows[-1].source_time_s + parent.release_dt_s
    if end_time != expected_end:
        raise ValueError("transport end time violates the release cadence")
    epoch = _exact_positive_integer("transport_epoch", transport_epoch)
    if epoch != parent.epoch:
        raise ValueError("transport epoch disagrees with committed owner")
    positions = _readonly_float64(
        "transported_positions", transported_positions, ndim=2
    )
    gamma = _readonly_float64("transported_gamma", transported_gamma, ndim=2)
    sigma = _readonly_float64("transported_sigma", transported_sigma, ndim=1)
    tracers = _readonly_float64(
        "transported_material_tracer_positions",
        transported_material_tracer_positions,
        ndim=2,
    )
    live_sigma = _readonly_float64(
        "transported_live_material_sigma",
        transported_live_material_sigma,
        ndim=1,
    )
    count = parent.state.positions.shape[0]
    if positions.shape != (count, 3) or gamma.shape != (count, 3):
        raise ValueError("transport must preserve particle count and vector shapes")
    if sigma.shape != (count,) or np.any(sigma <= 0.0):
        raise ValueError("transport sigma must stay positive and preserve count")
    if tracers.shape != session.material_tracer_positions.shape:
        raise ValueError("transported material tracer pack shape changed")
    flat = tuple(
        index for indices in session.live_particle_indices_by_cell for index in indices
    )
    flat_array = np.asarray(flat, dtype=np.int64)
    support_count = session.frontier_node_offset
    if live_sigma.shape != (support_count,):
        raise ValueError("transported live material sigma shape changed")
    if not _exact_array_equal(tracers[:support_count], positions[flat_array]):
        raise ValueError(
            "transported live supports lack exact common-transport attestation"
        )
    if not _exact_array_equal(live_sigma, sigma[flat_array]):
        raise ValueError(
            "transported live sigma lacks exact common-transport attestation"
        )
    nodes = _readonly_float64(
        "transported_live_boundary_nodes", tracers[support_count:], ndim=2
    )
    if nodes.shape != parent.state.live_boundary_nodes.shape:
        raise ValueError("transported live boundary node count changed")
    attestation = ReleaseRowTransportAttestation(
        common_transport_sha256=session.common_transport_sha256,
        parent_epoch=parent.epoch,
        parent_owner_sha256=parent.owner_sha256,
        parent_state_sha256=parent.state.state_sha256,
        source_step_index=step,
        transport_end_time_s=end_time,
        transport_epoch=epoch,
        transported_arrays_sha256=_digest(
            "fluxv-v5h10-transported-arrays-v1", positions, gamma, sigma
        ),
        transported_material_tracers_sha256=_digest(
            "fluxv-v5h10-transported-material-tracers-v1", tracers
        ),
        transported_live_material_sigma_sha256=_digest(
            "fluxv-v5h10-transported-live-material-sigma-v1", live_sigma
        ),
        transported_live_boundary_nodes=nodes,
        attestation_sha256="",
    )
    attestation = replace(
        attestation,
        attestation_sha256=_transport_attestation_digest(attestation),
    )
    plan = _TransportAttestationPlan(
        parent_owner=weakref.ref(parent),
        common_transport=weakref.ref(session),
        transported_positions=positions,
        transported_gamma=gamma,
        transported_sigma=sigma,
        transported_material_tracers=tracers,
        transported_live_material_sigma=live_sigma,
        issued_attestation=attestation,
        issued_sha256=attestation.attestation_sha256,
    )
    with _LOCK:
        if _CURRENT_OWNERS.get(parent.owner_id) is not parent:
            raise RuntimeError("owner generation changed during transport attestation")
        if session in _CURRENT_TRANSPORT_ATTESTATIONS:
            raise RuntimeError("common transport was already attested")
        try:
            _LIVE_TRANSPORT_ATTESTATIONS[attestation] = plan
            _CURRENT_TRANSPORT_ATTESTATIONS[session] = attestation
        except BaseException:
            _LIVE_TRANSPORT_ATTESTATIONS.pop(attestation, None)
            if _CURRENT_TRANSPORT_ATTESTATIONS.get(session) is attestation:
                _CURRENT_TRANSPORT_ATTESTATIONS.pop(session, None)
            raise
    return validate_release_row_transport_attestation(attestation)


def release_row_transport_digest(
    committed_state: object,
    transported_positions: ArrayLike,
    transported_gamma: ArrayLike,
    transported_sigma: ArrayLike,
    transported_live_boundary_nodes: ArrayLike,
    *,
    common_transport_attestation: object,
    source_step_index: int,
    transport_end_time_s: float,
    transport_epoch: int,
) -> str:
    """Return the digest for one live common-transport-attested handoff."""

    _assert_dependencies()
    if type(committed_state) is not ReleaseRowState:
        raise TypeError("committed_state must be an exact ReleaseRowState")
    attestation, plan, _, parent = _validate_transport_attestation(
        common_transport_attestation, require_current=True
    )
    if committed_state is not parent.state:
        raise RuntimeError("transport digest parent state is stale or not owned")
    positions = _readonly_float64(
        "transported_positions", transported_positions, ndim=2
    )
    gamma = _readonly_float64("transported_gamma", transported_gamma, ndim=2)
    sigma = _readonly_float64("transported_sigma", transported_sigma, ndim=1)
    nodes = _readonly_float64(
        "transported_live_boundary_nodes",
        transported_live_boundary_nodes,
        ndim=2,
    )
    step = _exact_nonnegative_integer("source_step_index", source_step_index)
    end_time = _positive_float("transport_end_time_s", transport_end_time_s)
    epoch = _exact_positive_integer("transport_epoch", transport_epoch)
    if (
        step != attestation.source_step_index
        or end_time != attestation.transport_end_time_s
        or epoch != attestation.transport_epoch
        or not _exact_array_equal(positions, plan.transported_positions)
        or not _exact_array_equal(gamma, plan.transported_gamma)
        or not _exact_array_equal(sigma, plan.transported_sigma)
        or not _exact_array_equal(nodes, attestation.transported_live_boundary_nodes)
    ):
        raise ValueError(
            "transported arrays or metadata lack their live common attestation"
        )
    return _digest(
        "fluxv-v5h10-transport-parent-v1",
        committed_state.state_sha256,
        positions,
        gamma,
        sigma,
        nodes,
        step,
        end_time,
        epoch,
        attestation.common_transport_sha256,
        attestation.attestation_sha256,
    )


def advance_release_row_transport_parent(
    owner: object,
    committed_state: object,
    transported_positions: ArrayLike,
    transported_gamma: ArrayLike,
    transported_sigma: ArrayLike,
    transported_live_boundary_nodes: ArrayLike,
    *,
    common_transport_attestation: object,
    parent_transport_digest: str,
    source_step_index: int,
    transport_end_time_s: float,
    transport_epoch: int,
) -> ReleaseRowOwner:
    """Atomically bind exactly one transport to the current committed row.

    The handoff preserves particle count, stable IDs, lineage, and live-slice
    indexing.  It changes only material x/gamma/sigma and the transported
    downstream-node frontier.  A next row cannot be proposed before this
    transition has published successfully.
    """

    _assert_dependencies()
    parent = _validate_live_owner(owner, require_current=True)
    attestation, _, session, attested_parent = _validate_transport_attestation(
        common_transport_attestation, require_current=True
    )
    if attested_parent is not parent:
        raise RuntimeError("common transport attestation belongs to another owner")
    if committed_state is not parent.state:
        raise RuntimeError("transport parent state is stale or not owned")
    if parent.state.phase != "post_commit_pre_transport":
        raise RuntimeError("current row was already transported")
    epoch = _exact_positive_integer("transport_epoch", transport_epoch)
    if epoch != parent.epoch:
        raise ValueError("transport epoch disagrees with committed owner")
    step = _exact_nonnegative_integer("source_step_index", source_step_index)
    if (
        parent.transport_events
        and step != parent.transport_events[-1].source_step_index + 1
    ):
        raise ValueError("transport source step is not sequential")
    end_time = _positive_float("transport_end_time_s", transport_end_time_s)
    expected_end = parent.state.rows[-1].source_time_s + parent.release_dt_s
    if end_time != expected_end:
        raise ValueError("transport end time violates the release cadence")
    if (
        type(parent_transport_digest) is not str
        or len(parent_transport_digest) != 64
        or any(
            character not in "0123456789abcdef" for character in parent_transport_digest
        )
    ):
        raise ValueError("parent_transport_digest must be a lowercase SHA-256")
    positions = _readonly_float64(
        "transported_positions", transported_positions, ndim=2
    )
    gamma = _readonly_float64("transported_gamma", transported_gamma, ndim=2)
    sigma = _readonly_float64("transported_sigma", transported_sigma, ndim=1)
    nodes = _readonly_float64(
        "transported_live_boundary_nodes",
        transported_live_boundary_nodes,
        ndim=2,
    )
    count = parent.state.positions.shape[0]
    if positions.shape != (count, 3) or gamma.shape != (count, 3):
        raise ValueError("transport must preserve particle count and vector shapes")
    if sigma.shape != (count,) or np.any(sigma <= 0.0):
        raise ValueError("transport sigma must stay positive and preserve count")
    if nodes.shape != parent.state.live_boundary_nodes.shape:
        raise ValueError("transported live boundary node count changed")
    expected_digest = release_row_transport_digest(
        parent.state,
        positions,
        gamma,
        sigma,
        nodes,
        common_transport_attestation=attestation,
        source_step_index=step,
        transport_end_time_s=end_time,
        transport_epoch=epoch,
    )
    if parent_transport_digest != expected_digest:
        raise ValueError("transported arrays or metadata disagree with their digest")

    transported_state = _state_from_parts(
        positions=positions,
        gamma=gamma,
        sigma=sigma,
        particle_ids=parent.state.particle_ids,
        lineage=parent.state.lineage,
        rows=parent.state.rows,
        row_parent_transport_digests=parent.state.row_parent_transport_digests,
        live_boundary_indices_by_cell=parent.state.live_boundary_indices_by_cell,
        live_boundary_nodes=nodes,
        epoch=parent.epoch,
        phase="post_transport",
        transport_generation=parent.epoch,
        transport_source_step_index=step,
        transport_end_time_s=end_time,
        parent_transport_digest=parent_transport_digest,
        parent_transport_attestation_sha256=attestation.attestation_sha256,
    )
    _validate_state_contents(transported_state)
    previous_transport = (
        parent.transport_events[-1].transport_event_sha256
        if parent.transport_events
        else _GENESIS_SHA256
    )
    transport_event = ReleaseRowTransportEvent(
        release_index=parent.epoch,
        source_step_index=step,
        transport_end_time_s=end_time,
        parent_state_sha256=parent.state.state_sha256,
        transported_state_sha256=transported_state.state_sha256,
        parent_transport_digest=parent_transport_digest,
        common_transport_attestation_sha256=attestation.attestation_sha256,
        transported_arrays_sha256=_digest(
            "fluxv-v5h10-transported-arrays-v1", positions, gamma, sigma
        ),
        live_boundary_nodes_sha256=_digest(
            "fluxv-v5h10-transported-live-nodes-v1", nodes
        ),
        previous_transport_event_sha256=previous_transport,
        transport_event_sha256="",
    )
    transport_event = replace(
        transport_event,
        transport_event_sha256=_transport_event_digest(transport_event),
    )
    next_owner = ReleaseRowOwner(
        owner_id=parent.owner_id,
        sheet_id=parent.sheet_id,
        particle_cap=parent.particle_cap,
        smoothing_radius_m=parent.smoothing_radius_m,
        target_spacing_m=parent.target_spacing_m,
        root_source_time_s=parent.root_source_time_s,
        release_dt_s=parent.release_dt_s,
        epoch=parent.epoch,
        state=transported_state,
        events=parent.events,
        transport_events=parent.transport_events + (transport_event,),
        dependency_sha256=parent.dependency_sha256,
        owner_sha256="",
    )
    next_owner = replace(next_owner, owner_sha256=_owner_digest(next_owner))
    _validate_owner_contents(next_owner)
    generation = _generation_key(parent)
    with _LOCK:
        if (
            _CURRENT_OWNERS.get(parent.owner_id) is not parent
            or generation in _CONSUMED_GENERATIONS
            or session in _CONSUMED_COMMON_TRANSPORTS
            or attestation in _CONSUMED_TRANSPORT_ATTESTATIONS
        ):
            raise RuntimeError("transport owner generation was already consumed")
        siblings = tuple(
            (candidate, candidate_plan)
            for candidate, candidate_plan in tuple(_LIVE_PROPOSALS.items())
            if candidate_plan.parent_owner() is parent
        )
        old_current = _CURRENT_OWNERS.get(parent.owner_id)
        old_common = _CURRENT_COMMON_TRANSPORTS.get(generation)
        old_attestation = _CURRENT_TRANSPORT_ATTESTATIONS.get(session)
        try:
            _LIVE_STATES[transported_state] = transported_state.state_sha256
            _LIVE_TRANSPORT_EVENTS[
                transport_event
            ] = transport_event.transport_event_sha256
            _LIVE_OWNERS[next_owner] = next_owner.owner_sha256
            _CONSUMED_GENERATIONS.add(generation)
            _CONSUMED_COMMON_TRANSPORTS.add(session)
            _CONSUMED_TRANSPORT_ATTESTATIONS.add(attestation)
            _CURRENT_COMMON_TRANSPORTS.pop(generation, None)
            _CURRENT_TRANSPORT_ATTESTATIONS.pop(session, None)
            for sibling, _ in siblings:
                _CONSUMED_PROPOSALS.add(sibling)
                _LIVE_PROPOSALS.pop(sibling, None)
            _CURRENT_OWNERS[parent.owner_id] = next_owner
        except BaseException:
            _LIVE_STATES.pop(transported_state, None)
            _LIVE_TRANSPORT_EVENTS.pop(transport_event, None)
            _LIVE_OWNERS.pop(next_owner, None)
            _CONSUMED_GENERATIONS.discard(generation)
            _CONSUMED_COMMON_TRANSPORTS.discard(session)
            _CONSUMED_TRANSPORT_ATTESTATIONS.discard(attestation)
            if old_common is not None:
                _CURRENT_COMMON_TRANSPORTS[generation] = old_common
            if old_attestation is not None:
                _CURRENT_TRANSPORT_ATTESTATIONS[session] = old_attestation
            for sibling, sibling_plan in siblings:
                _CONSUMED_PROPOSALS.discard(sibling)
                _LIVE_PROPOSALS[sibling] = sibling_plan
            if old_current is None:
                _CURRENT_OWNERS.pop(parent.owner_id, None)
            else:
                _CURRENT_OWNERS[parent.owner_id] = old_current
            raise
    return validate_release_row_owner(next_owner)


def _build_committed_state(
    owner: ReleaseRowOwner,
    proposal: ReleaseRowProposal,
    plan: _ProposalPlan,
) -> ReleaseRowState:
    gamma = np.array(owner.state.gamma, dtype=np.float64, order="C", copy=True)
    lineage = list(owner.state.lineage)
    for (
        index,
        addition,
        old_net,
        new_net,
        combined_net,
        incidences,
    ) in plan.upstream_additions:
        old = lineage[index]
        current_old_net = _incidence_circulation(
            old.birth_incidences + old.update_incidences
        )
        current_new_net = _incidence_circulation(incidences)
        if current_old_net == 0.0:
            raise RuntimeError("proposal additive incidence plan changed before commit")
        expected_addition = _readonly_float64(
            "recomputed upstream additive gamma",
            current_new_net * (owner.state.gamma[index] / current_old_net),
            ndim=1,
        )
        if (
            current_old_net != old_net
            or current_new_net != new_net
            or not _exact_array_equal(addition, expected_addition)
            or _incidence_circulation(
                old.birth_incidences + old.update_incidences + incidences
            )
            != combined_net
            or combined_net != fsum((old_net, new_net))
        ):
            raise RuntimeError("proposal additive incidence plan changed before commit")
        gamma[index] += addition
        lineage[index] = replace(
            old,
            role="interior_boundary",
            update_incidences=old.update_incidences + incidences,
        )
    positions = np.vstack((owner.state.positions, plan.suffix_positions))
    all_gamma = np.vstack((gamma, plan.suffix_gamma))
    sigma = np.concatenate((owner.state.sigma, plan.suffix_sigma))
    return _state_from_parts(
        positions=positions,
        gamma=all_gamma,
        sigma=sigma,
        particle_ids=owner.state.particle_ids + plan.suffix_ids,
        lineage=tuple(lineage) + plan.suffix_lineage,
        rows=owner.state.rows + (plan.row,),
        row_parent_transport_digests=owner.state.row_parent_transport_digests
        + (owner.state.parent_transport_digest,),
        live_boundary_indices_by_cell=plan.suffix_live_by_cell,
        live_boundary_nodes=plan.row.downstream_nodes,
        epoch=owner.epoch + 1,
        phase="post_commit_pre_transport",
        transport_generation=owner.state.transport_generation,
    )


def commit_release_row_update(owner: object, proposal: object) -> RowCommitResult:
    """Atomically consume one compatible proposal; remesh proposals are no-ops."""

    _assert_dependencies()
    if type(owner) is not ReleaseRowOwner:
        raise TypeError("owner must be an exact ReleaseRowOwner")
    if type(proposal) is not ReleaseRowProposal:
        raise TypeError("proposal must be an exact ReleaseRowProposal")
    with _LOCK:
        parent = _validate_live_owner(owner, require_current=True)
        if proposal.status == "remesh_required":
            if (
                proposal.parent_epoch != parent.epoch
                or proposal.parent_owner_sha256 != parent.owner_sha256
                or proposal.parent_state_sha256 != parent.state.state_sha256
                or proposal.proposal_sha256 != _proposal_digest(proposal)
            ):
                raise RuntimeError("remesh proposal is stale or invalid")
            return RowCommitResult(
                committed=False,
                status="remesh_required",
                owner=parent,
                state=parent.state,
                event=None,
                first_mismatch=proposal.first_mismatch,
            )
        generation = _generation_key(parent)
        if generation in _CONSUMED_GENERATIONS or proposal in _CONSUMED_PROPOSALS:
            raise RuntimeError("owner generation or proposal was already consumed")
        plan = _LIVE_PROPOSALS.get(proposal)
        if plan is None or plan.parent_owner() is not parent:
            raise RuntimeError("proposal is stale, forged, or not live")
        if (
            not _proposal_matches_issued(proposal, plan.issued_proposal)
            or plan.issued_proposal.proposal_sha256 != plan.issued_sha256
            or _proposal_digest(plan.issued_proposal) != plan.issued_sha256
            or proposal.proposal_sha256 != plan.issued_sha256
            or proposal.proposal_sha256 != _proposal_digest(proposal)
            or proposal.parent_epoch != parent.epoch
            or proposal.parent_owner_sha256 != parent.owner_sha256
            or proposal.parent_state_sha256 != parent.state.state_sha256
            or plan.parent_state_sha256 != parent.state.state_sha256
            or plan.parent_transport_digest != parent.state.parent_transport_digest
            or plan.parent_transport_attestation_sha256
            != parent.state.parent_transport_attestation_sha256
            or not _exact_array_equal(
                plan.parent_live_boundary_nodes, parent.state.live_boundary_nodes
            )
            or not _exact_array_equal(
                plan.parent_live_boundary_nodes, plan.row.upstream_nodes
            )
            or proposal.row_sha256 != plan.row.row_sha256
            or plan.row.row_sha256 != _row_digest(plan.row)
        ):
            raise RuntimeError("proposal issuance binding is invalid")

        new_state = _build_committed_state(parent, proposal, plan)
        if new_state.positions.shape[0] != proposal.planned_particle_count:
            raise RuntimeError("committed particle count disagrees with proposal")
        _validate_state_contents(new_state)
        changed = np.asarray(proposal.changed_indices, dtype=np.int64)
        before = parent.state.gamma[changed]
        after = new_state.gamma[changed]
        added = after - before
        previous_event = (
            parent.events[-1].event_sha256 if parent.events else _GENESIS_SHA256
        )
        prefix_count = parent.state.positions.shape[0]
        event = ReleaseRowEvent(
            proposal_id=proposal.proposal_id,
            release_index=new_state.release_index,
            changed_indices=proposal.changed_indices,
            appended_indices=tuple(range(prefix_count, new_state.positions.shape[0])),
            parent_state_sha256=parent.state.state_sha256,
            parent_transport_digest=parent.state.parent_transport_digest,
            parent_transport_event_sha256=(
                parent.transport_events[-1].transport_event_sha256
            ),
            upstream_nodes_sha256=_digest(
                "fluxv-v5h10-transported-live-nodes-v1", plan.row.upstream_nodes
            ),
            committed_state_sha256=new_state.state_sha256,
            row_sha256=plan.row.row_sha256,
            before_gamma_sha256=_digest("v5h10-before-gamma", before),
            added_gamma_sha256=_digest("v5h10-added-gamma", added),
            after_gamma_sha256=_digest("v5h10-after-gamma", after),
            operator_order=OPERATOR_ORDER,
            global_graph_build_count=1,
            clone_count=0,
            counter_particle_count=0,
            fresh_upstream_particle_count=0,
            remesh_count=0,
            ptera_call_count=0,
            load_call_count=0,
            feedback_call_count=0,
            transport_call_count=0,
            parent_owner_sha256=parent.owner_sha256,
            previous_event_sha256=previous_event,
            event_sha256="",
        )
        event = replace(event, event_sha256=_event_digest(event))
        next_owner = ReleaseRowOwner(
            owner_id=parent.owner_id,
            sheet_id=parent.sheet_id,
            particle_cap=parent.particle_cap,
            smoothing_radius_m=parent.smoothing_radius_m,
            target_spacing_m=parent.target_spacing_m,
            root_source_time_s=parent.root_source_time_s,
            release_dt_s=parent.release_dt_s,
            epoch=parent.epoch + 1,
            state=new_state,
            events=parent.events + (event,),
            transport_events=parent.transport_events,
            dependency_sha256=parent.dependency_sha256,
            owner_sha256="",
        )
        next_owner = replace(next_owner, owner_sha256=_owner_digest(next_owner))
        _validate_owner_contents(next_owner)
        siblings = tuple(
            (candidate, candidate_plan)
            for candidate, candidate_plan in tuple(_LIVE_PROPOSALS.items())
            if candidate_plan.parent_owner() is parent
        )
        old_current = _CURRENT_OWNERS.get(parent.owner_id)
        try:
            _LIVE_STATES[new_state] = new_state.state_sha256
            _LIVE_EVENTS[event] = event.event_sha256
            _LIVE_OWNERS[next_owner] = next_owner.owner_sha256
            _CONSUMED_GENERATIONS.add(generation)
            for sibling, _ in siblings:
                _CONSUMED_PROPOSALS.add(sibling)
                _LIVE_PROPOSALS.pop(sibling, None)
            _CURRENT_OWNERS[parent.owner_id] = next_owner
        except BaseException:
            _LIVE_STATES.pop(new_state, None)
            _LIVE_EVENTS.pop(event, None)
            _LIVE_OWNERS.pop(next_owner, None)
            _CONSUMED_GENERATIONS.discard(generation)
            for sibling, sibling_plan in siblings:
                _CONSUMED_PROPOSALS.discard(sibling)
                _LIVE_PROPOSALS[sibling] = sibling_plan
            if old_current is None:
                _CURRENT_OWNERS.pop(parent.owner_id, None)
            else:
                _CURRENT_OWNERS[parent.owner_id] = old_current
            raise
    return RowCommitResult(
        committed=True,
        status="compatible",
        owner=next_owner,
        state=new_state,
        event=event,
        first_mismatch=None,
    )


__all__ = [
    "DEPENDENCY_SHA256",
    "INTERFACE_ID",
    "OPERATOR_ORDER",
    "ReleaseRow",
    "ReleaseRowCommonTransport",
    "ReleaseRowEvent",
    "ReleaseRowOwner",
    "ReleaseRowProposal",
    "ReleaseRowState",
    "ReleaseRowTransportEvent",
    "ReleaseRowTransportAttestation",
    "RowCommitResult",
    "RowParticleLineage",
    "advance_release_row_transport_parent",
    "attest_release_row_common_transport",
    "begin_release_row_common_transport",
    "bootstrap_release_row_owner",
    "commit_release_row_update",
    "make_release_row",
    "propose_release_row_update",
    "release_row_transport_digest",
    "validate_current_release_row_owner",
    "validate_release_row_common_transport",
    "validate_release_row_owner",
    "validate_release_row_transport_attestation",
]
