"""Pre-registered dyadic midpoint deposition for an assembled edge graph.

The physical smoothing radius is fixed in SI units.  Spatial refinement acts
only on the number of equal panels on each retained edge::

    n0_e = max(1, ceil(length_e / base_target_spacing))
    n_e  = n0_e * 2**refinement_level

Level zero directly calls the existing public prescribed-spacing API.  Higher
levels call its frozen internal deposition primitive with the same particle
schema.  Thus level zero is an exact physical-state and ledger reduction,
while an immutable sidecar records the dyadic construction at every level.

This module is source-only mechanics: rings remain the physical owner and the
particles remain diagnostic shadows.  It has no feedback, load, target-data,
or surface-solver path.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass, replace
from hashlib import sha256
import hmac
import inspect
import json
from math import ceil
from numbers import Integral, Real
from pathlib import Path
import re
import threading
from typing import Any
import weakref

import numpy as np

import fluxvortex.rvpm_edge_bridge as edge_bridge_module
from fluxvortex.rvpm_edge_bridge import (
    DIAGNOSTIC_SHADOW_OWNER,
    FROZEN_OVERLAP_LAMBDA,
    RING_PHYSICAL_OWNER,
    BridgeDiagnostics,
    BridgeNode,
    DirectedRing,
    EdgeGraph,
    EdgeIncidence,
    EdgeKey,
    EdgeLedger,
    ParticleLineage,
    ShadowBridgeResult,
    _deposit_validated_graph,
    _validate_graph_for_deposition,
    assemble_ring_edge_graph,
    deposit_edge_graph_prescribed_sigma_and_spacing,
)


DYADIC_PARTICLE_SCHEMA = "rvpm-edge-shadow-prescribed-sigma-spacing-v1"
DYADIC_BRIDGE_INTERFACE_ID = "fluxv-rvpm-dyadic-edge-bridge-v1"
DYADIC_PLAN_INTERFACE_ID = "fluxv-rvpm-dyadic-edge-plan-v1"
DYADIC_BRIDGE_PRODUCER_ID = "prescribed-sigma-dyadic-midpoint-panels-v1"
DYADIC_PARENT_ATTESTATION_KIND = "sha256-complete-edge-graph-v1"
MAX_REFINEMENT_LEVEL = 30
MAX_PARTICLES_PER_EDGE = 250_000
MAX_TOTAL_PARTICLES = 1_000_000


@dataclass(frozen=True, slots=True)
class DyadicEdgePanelLedger:
    """Exact preregistered panel plan for one retained parent edge."""

    edge_key: EdgeKey
    edge_length_m: float
    base_panel_count: int
    refinement_level: int
    panel_count: int
    realized_spacing_m: float
    smoothing_radius_m: float
    parent_edge_graph_sha256: str


@dataclass(frozen=True, slots=True)
class DyadicDepositionPlan:
    """Immutable, independently recomputable plan made before allocation."""

    interface_id: str
    producer_id: str
    producer_artifact_sha256: str
    edge_bridge_artifact_sha256: str
    parent_attestation_kind: str
    parent_edge_graph_sha256: str
    smoothing_radius_m: float
    base_target_spacing_m: float
    refinement_level: int
    refinement_multiplier: int
    step: int
    predicted_total_particle_count: int
    particle_schema: str
    physical_owner: str
    owner_state: str
    edge_panels: tuple[DyadicEdgePanelLedger, ...]
    plan_sha256: str


@dataclass(frozen=True, slots=True)
class DyadicEdgeParticleLedger:
    """Post-deposition identity binding for one planned edge slice."""

    edge_key: EdgeKey
    particle_slice_start: int
    particle_slice_stop: int
    particle_ids_sha256: str
    lineage_sha256: str
    parent_edge_graph_sha256: str


@dataclass(frozen=True, slots=True, weakref_slot=True)
class DyadicBridgeResult:
    """Frozen dyadic sidecar plus the unchanged physical shadow result."""

    interface_id: str
    producer_id: str
    producer_artifact_sha256: str
    plan: DyadicDepositionPlan
    particle_ledger: tuple[DyadicEdgeParticleLedger, ...]
    shadow_state_sha256: str
    manifest_sha256: str
    bridge: ShadowBridgeResult

    @property
    def shadow(self) -> ShadowBridgeResult:
        return self.bridge

    @property
    def edge_panels(self) -> tuple[DyadicEdgePanelLedger, ...]:
        return self.plan.edge_panels

    @property
    def parent_edge_graph_sha256(self) -> str:
        return self.plan.parent_edge_graph_sha256

    @property
    def smoothing_radius_m(self) -> float:
        return self.plan.smoothing_radius_m

    @property
    def base_target_spacing_m(self) -> float:
        return self.plan.base_target_spacing_m

    @property
    def refinement_level(self) -> int:
        return self.plan.refinement_level

    @property
    def refinement_multiplier(self) -> int:
        return self.plan.refinement_multiplier

    @property
    def step(self) -> int:
        return self.plan.step

    @property
    def edge_graph(self) -> EdgeGraph | None:
        return self.bridge.edge_graph

    @property
    def positions(self) -> np.ndarray:
        return self.bridge.positions

    @property
    def gamma(self) -> np.ndarray:
        return self.bridge.gamma

    @property
    def sigma(self) -> np.ndarray:
        return self.bridge.sigma

    @property
    def particle_ids(self) -> tuple[Any, ...]:
        return self.bridge.particle_ids

    @property
    def lineage(self) -> tuple[Any, ...]:
        return self.bridge.lineage

    @property
    def feedback_velocity(self) -> None:
        return self.bridge.feedback_velocity

    @property
    def diagnostics(self) -> Any:
        return self.bridge.diagnostics


_RESULT_LOCK = threading.RLock()
_RESULT_REGISTRY: dict[int, tuple[weakref.ReferenceType[DyadicBridgeResult], str]] = {}


def _callable_source_sha256(name: str, value: object) -> str:
    if not inspect.isfunction(value):
        raise ValueError(f"{name} binding is not a Python function")
    try:
        source = inspect.getsource(value).encode("utf-8")
    except (OSError, TypeError) as error:
        raise ValueError(f"{name} binding has no auditable source") from error
    return sha256(source).hexdigest()


_FROZEN_ROOT_BINDINGS = (
    (
        "public_deposition",
        deposit_edge_graph_prescribed_sigma_and_spacing,
        edge_bridge_module,
        "deposit_edge_graph_prescribed_sigma_and_spacing",
        _callable_source_sha256(
            "public_deposition", deposit_edge_graph_prescribed_sigma_and_spacing
        ),
    ),
    (
        "private_deposition",
        _deposit_validated_graph,
        edge_bridge_module,
        "_deposit_validated_graph",
        _callable_source_sha256("private_deposition", _deposit_validated_graph),
    ),
    (
        "graph_validator",
        _validate_graph_for_deposition,
        edge_bridge_module,
        "_validate_graph_for_deposition",
        _callable_source_sha256("graph_validator", _validate_graph_for_deposition),
    ),
    (
        "graph_assembler",
        assemble_ring_edge_graph,
        edge_bridge_module,
        "assemble_ring_edge_graph",
        _callable_source_sha256("graph_assembler", assemble_ring_edge_graph),
    ),
)


def _freeze_transitive_callable_bindings(
    roots: tuple[tuple[Any, ...], ...],
) -> tuple[tuple[Any, ...], ...]:
    pending = [(str(binding[0]), binding[1]) for binding in roots]
    visited_functions: set[int] = set()
    visited_slots: set[tuple[int, str]] = set()
    dependencies: list[tuple[Any, ...]] = []
    while pending:
        root_label, function = pending.pop(0)
        if id(function) in visited_functions:
            continue
        visited_functions.add(id(function))
        code = getattr(function, "__code__", None)
        function_globals = getattr(function, "__globals__", None)
        if code is None or not isinstance(function_globals, dict):
            continue
        owner_module = function_globals.get("__name__")
        if not isinstance(owner_module, str) or not owner_module:
            raise ValueError(f"{root_label} dependency has no owner module")
        for attribute in sorted(set(code.co_names)):
            dependency = function_globals.get(attribute)
            if not inspect.isfunction(dependency):
                continue
            slot = (id(function_globals), attribute)
            if slot not in visited_slots:
                visited_slots.add(slot)
                dependencies.append(
                    (
                        root_label,
                        function_globals,
                        owner_module,
                        attribute,
                        dependency,
                        _callable_source_sha256(
                            f"{owner_module}.{attribute}", dependency
                        ),
                    )
                )
            pending.append((root_label, dependency))
    return tuple(
        sorted(
            dependencies,
            key=lambda item: (str(item[2]), str(item[3]), str(item[0])),
        )
    )


_FROZEN_TRANSITIVE_BINDINGS = _freeze_transitive_callable_bindings(
    _FROZEN_ROOT_BINDINGS
)


def _assert_runtime_bindings(
    _root_bindings: tuple[tuple[Any, ...], ...] = _FROZEN_ROOT_BINDINGS,
    _dependency_bindings: tuple[tuple[Any, ...], ...] = _FROZEN_TRANSITIVE_BINDINGS,
) -> dict[str, Any]:
    if (
        globals().get("_FROZEN_ROOT_BINDINGS") is not _root_bindings
        or globals().get("_FROZEN_TRANSITIVE_BINDINGS") is not _dependency_bindings
    ):
        raise ValueError("dyadic bridge frozen binding registry changed at runtime")
    local_targets = {
        "public_deposition": globals().get(
            "deposit_edge_graph_prescribed_sigma_and_spacing"
        ),
        "private_deposition": globals().get("_deposit_validated_graph"),
        "graph_validator": globals().get("_validate_graph_for_deposition"),
        "graph_assembler": globals().get("assemble_ring_edge_graph"),
    }
    trusted: dict[str, Any] = {}
    for label, frozen, module, attribute, expected_source in _root_bindings:
        if (
            local_targets.get(label) is not frozen
            or getattr(module, attribute, None) is not frozen
            or _callable_source_sha256(label, frozen) != expected_source
        ):
            raise ValueError(f"{label} binding changed at runtime")
        trusted[str(label)] = frozen
    for (
        root_label,
        function_globals,
        owner_module,
        attribute,
        frozen,
        expected_source,
    ) in _dependency_bindings:
        if (
            function_globals.get(attribute) is not frozen
            or _callable_source_sha256(
                f"{root_label}:{owner_module}.{attribute}", frozen
            )
            != expected_source
        ):
            raise ValueError(
                f"{root_label} dependency {owner_module}.{attribute} changed at runtime"
            )
    return trusted


def _strict_stable_id(name: str, value: object) -> int | str:
    if type(value) is int:
        return value
    if type(value) is str and value:
        return value
    raise ValueError(f"{name} must be an exact nonempty string or integer ID")


def _id_sort_key(value: int | str) -> tuple[str, int | str]:
    return ("integer", value) if type(value) is int else ("string", value)


def _strict_int(name: str, value: object, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an exact integer >= {minimum}")
    return value


def _strict_float(name: str, value: object, *, positive: bool = False) -> float:
    if type(value) is not float or not np.isfinite(value):
        raise ValueError(f"{name} must be an exact finite binary64 value")
    if positive and value <= 0.0:
        raise ValueError(f"{name} must be strictly positive")
    return value


def _strict_sha256(name: str, value: object) -> str:
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _require_exact_legacy_dataclass(
    name: str,
    value: object,
    expected_type: type[Any],
) -> None:
    if type(value) is not expected_type:
        raise ValueError(f"{name} has a foreign runtime type")
    namespace = getattr(value, "__dict__", None)
    expected_fields = {item.name for item in fields(expected_type)}
    if type(namespace) is not dict or set(namespace) != expected_fields:
        raise ValueError(f"{name} contains undeclared or missing fields")


def _exact_nested_equal(left: object, right: object) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, float):
        return left.hex() == right.hex()  # type: ignore[union-attr]
    if isinstance(left, tuple):
        return len(left) == len(right) and all(  # type: ignore[arg-type]
            _exact_nested_equal(a, b) for a, b in zip(left, right, strict=True)  # type: ignore[arg-type]
        )
    if is_dataclass(left) and not isinstance(left, type):
        return all(
            _exact_nested_equal(getattr(left, item.name), getattr(right, item.name))
            for item in fields(left)
        )
    return bool(left == right)


def _strict_float_tuple(name: str, value: object, length: int) -> tuple[float, ...]:
    if type(value) is not tuple or len(value) != length:
        raise ValueError(f"{name} must be an exact length-{length} tuple")
    for index, item in enumerate(value):
        _strict_float(f"{name}[{index}]", item)
    return value


def _strict_edge_graph(
    edge_graph: object,
    *,
    trusted: dict[str, Any] | None = None,
) -> EdgeGraph:
    bindings = _assert_runtime_bindings() if trusted is None else trusted
    graph = bindings["graph_validator"](edge_graph)
    _require_exact_legacy_dataclass("edge_graph", graph, EdgeGraph)
    if type(graph.nodes) is not tuple or not graph.nodes:
        raise ValueError("edge_graph.nodes must be a nonempty exact tuple")
    if type(graph.edges) is not tuple or not graph.edges:
        raise ValueError("edge_graph.edges must be a nonempty exact tuple")
    _strict_float("edge_graph.incidence_residual", graph.incidence_residual)
    _strict_float(
        "edge_graph.edge_reconstruction_residual",
        graph.edge_reconstruction_residual,
    )
    _strict_float_tuple(
        "edge_graph.global_vector_moment", graph.global_vector_moment, 3
    )

    node_ids: list[int | str] = []
    for node_index, node in enumerate(graph.nodes):
        _require_exact_legacy_dataclass(
            f"edge_graph.nodes[{node_index}]", node, BridgeNode
        )
        node_ids.append(
            _strict_stable_id(f"edge_graph.nodes[{node_index}].node_id", node.node_id)
        )
        _strict_float_tuple(
            f"edge_graph.nodes[{node_index}].position", node.position, 3
        )
    if node_ids != sorted(node_ids, key=_id_sort_key) or len(set(node_ids)) != len(
        node_ids
    ):
        raise ValueError("edge_graph nodes are not in canonical unique order")

    ring_buckets: dict[int | str, list[EdgeIncidence]] = {}
    edge_keys: list[EdgeKey] = []
    for edge_index, edge in enumerate(graph.edges):
        _require_exact_legacy_dataclass(
            f"edge_graph.edges[{edge_index}]", edge, EdgeLedger
        )
        if type(edge.key) is not tuple or len(edge.key) != 2:
            raise ValueError("edge key must be an exact two-ID tuple")
        start_id = _strict_stable_id("edge.key[0]", edge.key[0])
        end_id = _strict_stable_id("edge.key[1]", edge.key[1])
        if _id_sort_key(start_id) >= _id_sort_key(end_id):
            raise ValueError("edge key is not in canonical orientation")
        edge_keys.append(edge.key)
        _strict_float_tuple("edge.start_position", edge.start_position, 3)
        _strict_float_tuple("edge.end_position", edge.end_position, 3)
        _strict_float("edge.circulation", edge.circulation)
        _strict_float_tuple("edge.vector_moment", edge.vector_moment, 3)
        if type(edge.incidences) is not tuple or not edge.incidences:
            raise ValueError("edge incidences must be a nonempty exact tuple")
        incidence_order: list[tuple[tuple[str, int | str], int]] = []
        for incidence_index, incidence in enumerate(edge.incidences):
            _require_exact_legacy_dataclass(
                f"edge.incidences[{incidence_index}]", incidence, EdgeIncidence
            )
            ring_id = _strict_stable_id("incidence.ring_id", incidence.ring_id)
            traversal_index = _strict_int(
                "incidence.traversal_index", incidence.traversal_index
            )
            if traversal_index > 3:
                raise ValueError("incidence traversal index must be in [0, 3]")
            source_start = _strict_stable_id(
                "incidence.source_start_id", incidence.source_start_id
            )
            source_end = _strict_stable_id(
                "incidence.source_end_id", incidence.source_end_id
            )
            sign = incidence.canonical_sign
            if type(sign) is not int or sign not in (-1, 1):
                raise ValueError("incidence canonical_sign must be exact -1 or 1")
            _strict_float("incidence.ring_circulation", incidence.ring_circulation)
            _strict_float("incidence.signed_circulation", incidence.signed_circulation)
            canonical_key = (
                (source_start, source_end)
                if _id_sort_key(source_start) < _id_sort_key(source_end)
                else (source_end, source_start)
            )
            if canonical_key != edge.key:
                raise ValueError("incidence is bound to another canonical edge")
            incidence_order.append((_id_sort_key(ring_id), traversal_index))
            ring_buckets.setdefault(ring_id, []).append(incidence)
        if incidence_order != sorted(incidence_order):
            raise ValueError("edge incidences are not in canonical order")
    expected_edge_order = sorted(
        edge_keys, key=lambda item: (_id_sort_key(item[0]), _id_sort_key(item[1]))
    )
    if edge_keys != expected_edge_order or len(set(edge_keys)) != len(edge_keys):
        raise ValueError("edge_graph edges are not in canonical unique order")

    rings: list[DirectedRing] = []
    for ring_id in sorted(ring_buckets, key=_id_sort_key):
        incidences = sorted(
            ring_buckets[ring_id], key=lambda item: item.traversal_index
        )
        if len(incidences) != 4 or [item.traversal_index for item in incidences] != [
            0,
            1,
            2,
            3,
        ]:
            raise ValueError("each reconstructed ring must contain traversals 0..3")
        circulation = incidences[0].ring_circulation
        if any(item.ring_circulation.hex() != circulation.hex() for item in incidences):
            raise ValueError("ring incidence circulation changed across traversals")
        for index, incidence in enumerate(incidences):
            next_incidence = incidences[(index + 1) % 4]
            if not _exact_nested_equal(
                incidence.source_end_id, next_incidence.source_start_id
            ):
                raise ValueError("ring incidence traversal is not contiguous")
        node_sequence = tuple(item.source_start_id for item in incidences)
        if len(set(node_sequence)) != 4:
            raise ValueError("reconstructed ring does not have four distinct nodes")
        rings.append(
            DirectedRing(
                ring_id=ring_id,
                node_ids=node_sequence,
                circulation=circulation,
            )
        )
    rebuilt = bindings["graph_assembler"](graph.nodes, tuple(rings))
    if not _exact_nested_equal(rebuilt, graph):
        raise ValueError("edge_graph is not the canonical assembler output")
    return graph


def _strict_array(name: str, value: object, shape: tuple[int, ...]) -> np.ndarray:
    if type(value) is not np.ndarray:
        raise ValueError(f"{name} must be an exact numpy.ndarray")
    if value.dtype != np.dtype(np.float64) or not value.dtype.isnative:
        raise ValueError(f"{name} must use native float64 storage")
    if value.shape != shape:
        raise ValueError(f"{name} has an unexpected shape")
    if not value.flags.c_contiguous or not value.flags.aligned:
        raise ValueError(f"{name} must be aligned C-contiguous storage")
    if not value.flags.writeable or not value.flags.owndata:
        raise ValueError(f"{name} must be an owned, writeable physical-state array")
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{name} contains non-finite state")
    return value


def _strict_lineage(value: object, index: int) -> ParticleLineage:
    _require_exact_legacy_dataclass(f"lineage[{index}]", value, ParticleLineage)
    if type(value.particle_id) is not tuple:
        raise ValueError("particle lineage ID must be an exact tuple")
    if type(value.source_edge) is not tuple or len(value.source_edge) != 2:
        raise ValueError("particle lineage source_edge must be an exact edge key")
    _strict_stable_id("lineage.source_edge[0]", value.source_edge[0])
    _strict_stable_id("lineage.source_edge[1]", value.source_edge[1])
    _strict_int("lineage.subdivision_index", value.subdivision_index)
    _strict_int("lineage.subdivision_count", value.subdivision_count, minimum=1)
    if type(value.ring_incidences) is not tuple:
        raise ValueError("particle lineage incidences must be an exact tuple")
    for incidence_index, incidence in enumerate(value.ring_incidences):
        _require_exact_legacy_dataclass(
            f"lineage[{index}].ring_incidences[{incidence_index}]",
            incidence,
            EdgeIncidence,
        )
    _strict_int("lineage.step", value.step)
    if type(value.physical_owner) is not str or type(value.owner_state) is not str:
        raise ValueError("particle lineage owner fields must be exact strings")
    return value


def _strict_diagnostics(value: object) -> BridgeDiagnostics:
    _require_exact_legacy_dataclass("bridge.diagnostics", value, BridgeDiagnostics)
    if type(value.enabled) is not bool:
        raise ValueError("bridge diagnostics enabled must be exact bool")
    for name in (
        "source_edge_count",
        "retained_edge_count",
        "particle_count",
        "clip_count",
        "nonfinite_count",
        "owner_conflict_count",
        "feedback_call_count",
    ):
        _strict_int(f"bridge.diagnostics.{name}", getattr(value, name))
    for name in (
        "incidence_residual",
        "edge_reconstruction_residual",
        "max_edge_conservation_abs",
        "max_edge_conservation_rel",
        "global_conservation_abs",
        "global_conservation_rel",
    ):
        _strict_float(f"bridge.diagnostics.{name}", getattr(value, name))
    return value


def _strict_shadow_bridge(
    value: object,
    *,
    trusted: dict[str, Any] | None = None,
) -> ShadowBridgeResult:
    _require_exact_legacy_dataclass("bridge", value, ShadowBridgeResult)
    graph = _strict_edge_graph(value.edge_graph, trusted=trusted)
    particle_count = _strict_int(
        "bridge.diagnostics.particle_count",
        _strict_diagnostics(value.diagnostics).particle_count,
    )
    _strict_array("bridge.positions", value.positions, (particle_count, 3))
    _strict_array("bridge.gamma", value.gamma, (particle_count, 3))
    sigma = _strict_array("bridge.sigma", value.sigma, (particle_count,))
    if particle_count and np.any(sigma <= 0.0):
        raise ValueError("bridge sigma must be strictly positive")
    if (
        type(value.particle_ids) is not tuple
        or len(value.particle_ids) != particle_count
    ):
        raise ValueError("bridge particle_ids must be an exact complete tuple")
    if type(value.lineage) is not tuple or len(value.lineage) != particle_count:
        raise ValueError("bridge lineage must be an exact complete tuple")
    for index, particle_id in enumerate(value.particle_ids):
        if type(particle_id) is not tuple:
            raise ValueError(f"particle_ids[{index}] must be an exact tuple")
    for index, lineage in enumerate(value.lineage):
        _strict_lineage(lineage, index)
    if value.feedback_velocity is not None:
        raise ValueError("diagnostic bridge cannot expose feedback velocity")
    if value.edge_graph is not graph:
        raise ValueError("bridge edge graph identity changed during validation")
    return value


def _strict_dyadic_plan(value: object) -> DyadicDepositionPlan:
    if type(value) is not DyadicDepositionPlan:
        raise ValueError("dyadic plan has a foreign schema")
    for name in (
        "interface_id",
        "producer_id",
        "parent_attestation_kind",
        "particle_schema",
        "physical_owner",
        "owner_state",
    ):
        if type(getattr(value, name)) is not str:
            raise ValueError(f"plan.{name} must be an exact string")
    for name in (
        "producer_artifact_sha256",
        "edge_bridge_artifact_sha256",
        "parent_edge_graph_sha256",
        "plan_sha256",
    ):
        _strict_sha256(f"plan.{name}", getattr(value, name))
    _strict_float("plan.smoothing_radius_m", value.smoothing_radius_m, positive=True)
    _strict_float(
        "plan.base_target_spacing_m", value.base_target_spacing_m, positive=True
    )
    for name, minimum in (
        ("refinement_level", 0),
        ("refinement_multiplier", 1),
        ("step", 0),
        ("predicted_total_particle_count", 0),
    ):
        _strict_int(f"plan.{name}", getattr(value, name), minimum=minimum)
    if type(value.edge_panels) is not tuple:
        raise ValueError("plan.edge_panels must be an exact tuple")
    for index, panel in enumerate(value.edge_panels):
        if type(panel) is not DyadicEdgePanelLedger:
            raise ValueError(f"plan.edge_panels[{index}] has a foreign schema")
        if type(panel.edge_key) is not tuple or len(panel.edge_key) != 2:
            raise ValueError("panel edge_key must be an exact edge key")
        _strict_stable_id("panel.edge_key[0]", panel.edge_key[0])
        _strict_stable_id("panel.edge_key[1]", panel.edge_key[1])
        _strict_float("panel.edge_length_m", panel.edge_length_m, positive=True)
        _strict_float(
            "panel.realized_spacing_m", panel.realized_spacing_m, positive=True
        )
        _strict_float(
            "panel.smoothing_radius_m", panel.smoothing_radius_m, positive=True
        )
        for name, minimum in (
            ("base_panel_count", 1),
            ("refinement_level", 0),
            ("panel_count", 1),
        ):
            _strict_int(f"panel.{name}", getattr(panel, name), minimum=minimum)
        _strict_sha256("panel.parent_edge_graph_sha256", panel.parent_edge_graph_sha256)
    return value


def _strict_particle_ledgers(
    value: object,
) -> tuple[DyadicEdgeParticleLedger, ...]:
    if type(value) is not tuple:
        raise ValueError("particle_ledger must be an exact tuple")
    for index, item in enumerate(value):
        if type(item) is not DyadicEdgeParticleLedger:
            raise ValueError(f"particle_ledger[{index}] has a foreign schema")
        if type(item.edge_key) is not tuple or len(item.edge_key) != 2:
            raise ValueError("particle ledger edge_key must be exact")
        _strict_stable_id("particle_ledger.edge_key[0]", item.edge_key[0])
        _strict_stable_id("particle_ledger.edge_key[1]", item.edge_key[1])
        _strict_int("particle_ledger.start", item.particle_slice_start)
        _strict_int("particle_ledger.stop", item.particle_slice_stop)
        if item.particle_slice_stop < item.particle_slice_start:
            raise ValueError("particle ledger slice is reversed")
        _strict_sha256("particle_ledger.particle_ids", item.particle_ids_sha256)
        _strict_sha256("particle_ledger.lineage", item.lineage_sha256)
        _strict_sha256("particle_ledger.parent", item.parent_edge_graph_sha256)
    return value


def _finite_positive_real(name: str, value: object) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite positive real number")
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be a finite positive real number")
    return result


def _refinement_level(value: object) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError("refinement_level must be a nonnegative integer")
    result = int(value)
    if result < 0:
        raise ValueError("refinement_level must be a nonnegative integer")
    if result > MAX_REFINEMENT_LEVEL:
        raise ValueError("refinement_level exceeds the preregistered level cap")
    return result


def _step_number(value: object) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError("step must be a nonnegative integer")
    result = int(value)
    if result < 0:
        raise ValueError("step must be a nonnegative integer")
    return result


def _canonical(value: object) -> object:
    if value is None or isinstance(value, (bool, str, int)):
        return value
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (float, np.floating)):
        converted = float(value)
        if not np.isfinite(converted):
            raise ValueError("digest payload contains a non-finite float")
        return {"binary64_hex": converted.hex()}
    if isinstance(value, tuple):
        return {"tuple": [_canonical(item) for item in value]}
    if isinstance(value, list):
        return {"list": [_canonical(item) for item in value]}
    if isinstance(value, dict):
        return {
            str(key): _canonical(item)
            for key, item in sorted(value.items(), key=lambda row: str(row[0]))
        }
    raise TypeError(f"unsupported canonical payload type: {type(value).__name__}")


def _payload_sha256(domain: str, payload: object) -> str:
    encoded = json.dumps(
        _canonical(payload),
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return sha256(domain.encode("ascii") + b"\0" + encoded).hexdigest()


def _producer_artifact_sha256() -> str:
    return sha256(Path(__file__).resolve().read_bytes()).hexdigest()


def _edge_bridge_artifact_sha256() -> str:
    return sha256(Path(edge_bridge_module.__file__).resolve().read_bytes()).hexdigest()


def _incidence_payload(value: object) -> dict[str, object]:
    return {
        "ring_id": value.ring_id,
        "traversal_index": value.traversal_index,
        "source_start_id": value.source_start_id,
        "source_end_id": value.source_end_id,
        "canonical_sign": value.canonical_sign,
        "ring_circulation": value.ring_circulation,
        "signed_circulation": value.signed_circulation,
    }


def _edge_graph_payload(graph: EdgeGraph) -> dict[str, object]:
    return {
        "nodes": [
            {"node_id": node.node_id, "position": tuple(node.position)}
            for node in graph.nodes
        ],
        "edges": [
            {
                "key": edge.key,
                "start_position": edge.start_position,
                "end_position": edge.end_position,
                "incidences": [
                    _incidence_payload(incidence) for incidence in edge.incidences
                ],
                "circulation": edge.circulation,
                "vector_moment": edge.vector_moment,
            }
            for edge in graph.edges
        ],
        "incidence_residual": graph.incidence_residual,
        "edge_reconstruction_residual": graph.edge_reconstruction_residual,
        "global_vector_moment": graph.global_vector_moment,
    }


def edge_graph_parent_digest_sha256(edge_graph: EdgeGraph) -> str:
    """Hash every node, edge, incidence, geometry, and circulation field."""

    graph = _strict_edge_graph(edge_graph)
    return _payload_sha256(
        "fluxv-rvpm-dyadic-parent-edge-graph-v1",
        _edge_graph_payload(graph),
    )


def _dyadic_count_plan(
    graph: EdgeGraph,
    *,
    base_target_spacing: float,
    refinement_level: int,
) -> dict[EdgeKey, int]:
    """Validate all counts and resource limits before particle allocation."""

    multiplier = 1 << refinement_level
    count_by_edge: dict[EdgeKey, int] = {}
    total_count = 0
    for edge in graph.retained_edges:
        start = np.asarray(edge.start_position, dtype=np.float64)
        end = np.asarray(edge.end_position, dtype=np.float64)
        edge_length = float(np.linalg.norm(end - start))
        quotient = edge_length / base_target_spacing
        if not np.isfinite(quotient):
            raise ValueError(
                f"edge {edge.key!r} produced a non-finite base panel count"
            )
        if quotient > MAX_PARTICLES_PER_EDGE:
            raise ValueError(
                f"edge {edge.key!r} exceeds the particle-per-edge resource cap"
            )
        base_count = max(1, ceil(quotient))
        if base_count > MAX_PARTICLES_PER_EDGE // multiplier:
            raise ValueError(
                f"edge {edge.key!r} exceeds the particle-per-edge resource cap"
            )
        count = base_count * multiplier
        if total_count > MAX_TOTAL_PARTICLES - count:
            raise ValueError(
                "dyadic deposition exceeds the total-particle resource cap"
            )
        count_by_edge[edge.key] = count
        total_count += count
    return count_by_edge


def _panel_payload(panel: DyadicEdgePanelLedger) -> dict[str, object]:
    return {
        "edge_key": panel.edge_key,
        "edge_length_m": panel.edge_length_m,
        "base_panel_count": panel.base_panel_count,
        "refinement_level": panel.refinement_level,
        "panel_count": panel.panel_count,
        "realized_spacing_m": panel.realized_spacing_m,
        "smoothing_radius_m": panel.smoothing_radius_m,
        "parent_edge_graph_sha256": panel.parent_edge_graph_sha256,
    }


def _plan_digest(plan: DyadicDepositionPlan) -> str:
    return _payload_sha256(
        "fluxv-rvpm-dyadic-deposition-plan-v1",
        {
            "interface_id": plan.interface_id,
            "producer_id": plan.producer_id,
            "producer_artifact_sha256": plan.producer_artifact_sha256,
            "edge_bridge_artifact_sha256": plan.edge_bridge_artifact_sha256,
            "parent_attestation_kind": plan.parent_attestation_kind,
            "parent_edge_graph_sha256": plan.parent_edge_graph_sha256,
            "smoothing_radius_m": plan.smoothing_radius_m,
            "base_target_spacing_m": plan.base_target_spacing_m,
            "refinement_level": plan.refinement_level,
            "refinement_multiplier": plan.refinement_multiplier,
            "step": plan.step,
            "predicted_total_particle_count": (plan.predicted_total_particle_count),
            "particle_schema": plan.particle_schema,
            "physical_owner": plan.physical_owner,
            "owner_state": plan.owner_state,
            "edge_panels": [_panel_payload(panel) for panel in plan.edge_panels],
        },
    )


def plan_edge_graph_prescribed_sigma_dyadic_panels(
    edge_graph: EdgeGraph,
    *,
    smoothing_radius: float,
    base_target_spacing: float,
    refinement_level: int,
    step: int,
) -> DyadicDepositionPlan:
    """Build the complete immutable dyadic sidecar before allocation."""

    trusted = _assert_runtime_bindings()
    graph = _strict_edge_graph(edge_graph, trusted=trusted)
    sigma = _finite_positive_real("smoothing_radius", smoothing_radius)
    base_spacing = _finite_positive_real("base_target_spacing", base_target_spacing)
    level = _refinement_level(refinement_level)
    step_number = _step_number(step)
    requested_base_overlap = sigma / base_spacing
    if (
        not np.isfinite(requested_base_overlap)
        or requested_base_overlap < FROZEN_OVERLAP_LAMBDA
    ):
        raise ValueError(
            "base_target_spacing is too large for the frozen minimum overlap"
        )

    count_by_edge = _dyadic_count_plan(
        graph,
        base_target_spacing=base_spacing,
        refinement_level=level,
    )
    multiplier = 1 << level
    parent_digest = edge_graph_parent_digest_sha256(graph)
    panels: list[DyadicEdgePanelLedger] = []
    for edge in graph.retained_edges:
        start = np.asarray(edge.start_position, dtype=np.float64)
        end = np.asarray(edge.end_position, dtype=np.float64)
        length = float(np.linalg.norm(end - start))
        panel_count = count_by_edge[edge.key]
        base_count = panel_count // multiplier
        realized_spacing = length / panel_count
        if (
            base_count < 1
            or panel_count != base_count * multiplier
            or not np.isfinite(realized_spacing)
            or realized_spacing <= 0.0
        ):
            raise RuntimeError("dyadic panel plan is internally inconsistent")
        panels.append(
            DyadicEdgePanelLedger(
                edge_key=edge.key,
                edge_length_m=length,
                base_panel_count=base_count,
                refinement_level=level,
                panel_count=panel_count,
                realized_spacing_m=realized_spacing,
                smoothing_radius_m=sigma,
                parent_edge_graph_sha256=parent_digest,
            )
        )
    placeholder = DyadicDepositionPlan(
        interface_id=DYADIC_PLAN_INTERFACE_ID,
        producer_id=DYADIC_BRIDGE_PRODUCER_ID,
        producer_artifact_sha256=_producer_artifact_sha256(),
        edge_bridge_artifact_sha256=_edge_bridge_artifact_sha256(),
        parent_attestation_kind=DYADIC_PARENT_ATTESTATION_KIND,
        parent_edge_graph_sha256=parent_digest,
        smoothing_radius_m=sigma,
        base_target_spacing_m=base_spacing,
        refinement_level=level,
        refinement_multiplier=multiplier,
        step=step_number,
        predicted_total_particle_count=sum(count_by_edge.values()),
        particle_schema=DYADIC_PARTICLE_SCHEMA,
        physical_owner=RING_PHYSICAL_OWNER,
        owner_state=DIAGNOSTIC_SHADOW_OWNER,
        edge_panels=tuple(panels),
        plan_sha256="0" * 64,
    )
    return replace(placeholder, plan_sha256=_plan_digest(placeholder))


def validate_dyadic_deposition_plan(
    plan: object,
    edge_graph: EdgeGraph,
) -> DyadicDepositionPlan:
    """Independently rebuild a portable plan and verify its parent binding."""

    _assert_runtime_bindings()
    validated_plan = _strict_dyadic_plan(plan)
    rebuilt = plan_edge_graph_prescribed_sigma_dyadic_panels(
        edge_graph,
        smoothing_radius=validated_plan.smoothing_radius_m,
        base_target_spacing=validated_plan.base_target_spacing_m,
        refinement_level=validated_plan.refinement_level,
        step=validated_plan.step,
    )
    if not _exact_nested_equal(rebuilt, validated_plan):
        raise ValueError("dyadic plan is stale, tampered, or bound to another graph")
    return validated_plan


def _lineage_payload(value: object) -> dict[str, object]:
    return {
        "particle_id": value.particle_id,
        "source_edge": value.source_edge,
        "subdivision_index": value.subdivision_index,
        "subdivision_count": value.subdivision_count,
        "ring_incidences": [
            _incidence_payload(incidence) for incidence in value.ring_incidences
        ],
        "step": value.step,
        "physical_owner": value.physical_owner,
        "owner_state": value.owner_state,
    }


def _identity_digest(values: tuple[Any, ...], *, domain: str) -> str:
    return _payload_sha256(domain, list(values))


def _lineage_digest(values: tuple[Any, ...], *, domain: str) -> str:
    return _payload_sha256(
        domain,
        [_lineage_payload(value) for value in values],
    )


def _array_digest(digest: Any, array_like: object) -> None:
    if type(array_like) is not np.ndarray:
        raise ValueError("digest array must be an exact numpy.ndarray")
    array = array_like
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(array.shape, separators=(",", ":")).encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(array.strides, separators=(",", ":")).encode("ascii"))
    digest.update(b"\0")
    digest.update(b"C" if array.flags.c_contiguous else b"N")
    digest.update(b"\0")
    digest.update(array.tobytes(order="A"))
    digest.update(b"\0")


def _shadow_state_digest(shadow: ShadowBridgeResult, parent_digest: str) -> str:
    digest = sha256(b"fluxv-rvpm-dyadic-shadow-state-v1\0")
    digest.update(parent_digest.encode("ascii"))
    digest.update(b"\0")
    _array_digest(digest, shadow.positions)
    _array_digest(digest, shadow.gamma)
    _array_digest(digest, shadow.sigma)
    digest.update(
        _identity_digest(
            shadow.particle_ids,
            domain="fluxv-rvpm-dyadic-particle-identities-v1",
        ).encode("ascii")
    )
    digest.update(b"\0")
    digest.update(
        _lineage_digest(
            shadow.lineage,
            domain="fluxv-rvpm-dyadic-lineage-v1",
        ).encode("ascii")
    )
    digest.update(b"\0")
    digest.update(
        _payload_sha256(
            "fluxv-rvpm-dyadic-diagnostics-v1",
            {
                name: getattr(shadow.diagnostics, name)
                for name in shadow.diagnostics.__dataclass_fields__
            },
        ).encode("ascii")
    )
    return digest.hexdigest()


def _particle_ledger_payload(value: DyadicEdgeParticleLedger) -> dict[str, object]:
    return {
        "edge_key": value.edge_key,
        "particle_slice_start": value.particle_slice_start,
        "particle_slice_stop": value.particle_slice_stop,
        "particle_ids_sha256": value.particle_ids_sha256,
        "lineage_sha256": value.lineage_sha256,
        "parent_edge_graph_sha256": value.parent_edge_graph_sha256,
    }


def _particle_sidecar(
    shadow: ShadowBridgeResult,
    plan: DyadicDepositionPlan,
) -> tuple[tuple[DyadicEdgeParticleLedger, ...], str]:
    trusted = _assert_runtime_bindings()
    validated_shadow = _strict_shadow_bridge(shadow, trusted=trusted)
    _strict_dyadic_plan(plan)
    graph = validated_shadow.edge_graph
    parent_digest = edge_graph_parent_digest_sha256(graph)
    if parent_digest != plan.parent_edge_graph_sha256:
        raise ValueError("physical shadow is bound to another parent edge graph")
    count = plan.predicted_total_particle_count
    if (
        shadow.positions.shape != (count, 3)
        or shadow.gamma.shape != (count, 3)
        or shadow.sigma.shape != (count,)
        or len(shadow.particle_ids) != count
        or len(shadow.lineage) != count
        or shadow.diagnostics.particle_count != count
    ):
        raise ValueError(
            "physical shadow particle counts disagree with the dyadic plan"
        )
    if not (
        np.all(np.isfinite(shadow.positions))
        and np.all(np.isfinite(shadow.gamma))
        and np.all(np.isfinite(shadow.sigma))
        and np.all(shadow.sigma == plan.smoothing_radius_m)
    ):
        raise ValueError("physical shadow contains invalid fixed-core state")
    if shadow.feedback_velocity is not None or shadow.diagnostics.feedback_call_count:
        raise ValueError("dyadic physical shadow cannot contain feedback")
    diagnostics = shadow.diagnostics
    if (
        not diagnostics.enabled
        or diagnostics.source_edge_count != len(graph.edges)
        or diagnostics.retained_edge_count != len(graph.retained_edges)
        or diagnostics.incidence_residual != graph.incidence_residual
        or diagnostics.edge_reconstruction_residual
        != graph.edge_reconstruction_residual
        or diagnostics.max_edge_conservation_abs > 1.0e-14
        or diagnostics.max_edge_conservation_rel > 1.0e-12
        or diagnostics.global_conservation_abs > 1.0e-14
        or diagnostics.global_conservation_rel > 1.0e-12
        or diagnostics.clip_count
        or diagnostics.nonfinite_count
        or diagnostics.owner_conflict_count
    ):
        raise ValueError("dyadic physical shadow diagnostics are inconsistent")
    if len(set(shadow.particle_ids)) != count:
        raise ValueError("dyadic physical shadow contains duplicate particle IDs")

    edges = {edge.key: edge for edge in graph.retained_edges}
    cursor = 0
    ledgers: list[DyadicEdgeParticleLedger] = []
    for panel in plan.edge_panels:
        edge = edges.get(panel.edge_key)
        if edge is None:
            raise ValueError("dyadic plan references an absent retained edge")
        start_index = cursor
        stop_index = cursor + panel.panel_count
        block_lineage = shadow.lineage[start_index:stop_index]
        block_ids = shadow.particle_ids[start_index:stop_index]
        if len(block_lineage) != panel.panel_count:
            raise ValueError("dyadic particle slice is truncated")
        start = np.asarray(edge.start_position, dtype=np.float64)
        end = np.asarray(edge.end_position, dtype=np.float64)
        delta_l = (end - start) / panel.panel_count
        expected_gamma = edge.circulation * delta_l
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
        for local_index, (particle_id, lineage) in enumerate(
            zip(block_ids, block_lineage, strict=True)
        ):
            global_index = start_index + local_index
            expected_position = start + (local_index + 0.5) * delta_l
            expected_id = (
                plan.particle_schema,
                edge.key,
                local_index,
                panel.panel_count,
                plan.smoothing_radius_m,
                incidence_signature,
                plan.step,
                plan.owner_state,
            )
            if (
                not np.array_equal(shadow.positions[global_index], expected_position)
                or not np.array_equal(shadow.gamma[global_index], expected_gamma)
                or shadow.sigma[global_index] != plan.smoothing_radius_m
                or not _exact_nested_equal(particle_id, expected_id)
                or not _exact_nested_equal(lineage.particle_id, particle_id)
                or not _exact_nested_equal(lineage.source_edge, edge.key)
                or lineage.subdivision_index != local_index
                or lineage.subdivision_count != panel.panel_count
                or not _exact_nested_equal(lineage.ring_incidences, edge.incidences)
                or lineage.step != plan.step
                or lineage.physical_owner != plan.physical_owner
                or lineage.owner_state != plan.owner_state
            ):
                raise ValueError(
                    "dyadic particle physical state, identity, or lineage is inconsistent"
                )
        ledgers.append(
            DyadicEdgeParticleLedger(
                edge_key=edge.key,
                particle_slice_start=start_index,
                particle_slice_stop=stop_index,
                particle_ids_sha256=_identity_digest(
                    block_ids,
                    domain="fluxv-rvpm-dyadic-edge-particle-identities-v1",
                ),
                lineage_sha256=_lineage_digest(
                    block_lineage,
                    domain="fluxv-rvpm-dyadic-edge-lineage-v1",
                ),
                parent_edge_graph_sha256=parent_digest,
            )
        )
        cursor = stop_index
    if cursor != count or len(ledgers) != len(graph.retained_edges):
        raise ValueError("dyadic particle slices do not exactly cover the shadow")
    return tuple(ledgers), _shadow_state_digest(shadow, parent_digest)


def _manifest_digest(
    *,
    producer_artifact_sha256: str,
    plan: DyadicDepositionPlan,
    particle_ledger: tuple[DyadicEdgeParticleLedger, ...],
    shadow_state_sha256: str,
) -> str:
    return _payload_sha256(
        "fluxv-rvpm-dyadic-bridge-manifest-v1",
        {
            "interface_id": DYADIC_BRIDGE_INTERFACE_ID,
            "producer_id": DYADIC_BRIDGE_PRODUCER_ID,
            "producer_artifact_sha256": producer_artifact_sha256,
            "plan_sha256": plan.plan_sha256,
            "particle_ledger": [
                _particle_ledger_payload(item) for item in particle_ledger
            ],
            "shadow_state_sha256": shadow_state_sha256,
        },
    )


def recompute_dyadic_bridge_sidecar(result: object) -> dict[str, object]:
    """Rebuild the plan, parent binding, slices, and digests independently."""

    _assert_runtime_bindings()
    if type(result) is not DyadicBridgeResult:
        raise ValueError("dyadic bridge result has a foreign schema")
    _strict_dyadic_plan(result.plan)
    _strict_particle_ledgers(result.particle_ledger)
    _strict_sha256("result.shadow_state_sha256", result.shadow_state_sha256)
    _strict_sha256("result.manifest_sha256", result.manifest_sha256)
    graph = _strict_shadow_bridge(result.bridge).edge_graph
    rebuilt_plan = plan_edge_graph_prescribed_sigma_dyadic_panels(
        graph,
        smoothing_radius=result.plan.smoothing_radius_m,
        base_target_spacing=result.plan.base_target_spacing_m,
        refinement_level=result.plan.refinement_level,
        step=result.plan.step,
    )
    particle_ledger, shadow_digest = _particle_sidecar(result.bridge, rebuilt_plan)
    producer_hash = _producer_artifact_sha256()
    return {
        "plan": rebuilt_plan,
        "particle_ledger": particle_ledger,
        "shadow_state_sha256": shadow_digest,
        "manifest_sha256": _manifest_digest(
            producer_artifact_sha256=producer_hash,
            plan=rebuilt_plan,
            particle_ledger=particle_ledger,
            shadow_state_sha256=shadow_digest,
        ),
    }


def _validate_result_semantics(result: object) -> DyadicBridgeResult:
    _assert_runtime_bindings()
    if type(result) is not DyadicBridgeResult:
        raise ValueError("dyadic bridge result has a foreign schema")
    _strict_dyadic_plan(result.plan)
    _strict_particle_ledgers(result.particle_ledger)
    _strict_shadow_bridge(result.bridge)
    _strict_sha256("result.shadow_state_sha256", result.shadow_state_sha256)
    _strict_sha256("result.manifest_sha256", result.manifest_sha256)
    producer_hash = _producer_artifact_sha256()
    if (
        result.interface_id != DYADIC_BRIDGE_INTERFACE_ID
        or result.producer_id != DYADIC_BRIDGE_PRODUCER_ID
        or result.producer_artifact_sha256 != producer_hash
    ):
        raise ValueError("dyadic bridge producer identity is stale or tampered")
    recomputed = recompute_dyadic_bridge_sidecar(result)
    if not _exact_nested_equal(recomputed["plan"], result.plan):
        raise ValueError("dyadic bridge plan does not match independent recomputation")
    if not _exact_nested_equal(recomputed["particle_ledger"], result.particle_ledger):
        raise ValueError("dyadic bridge particle ledger is stale or tampered")
    if recomputed["shadow_state_sha256"] != result.shadow_state_sha256:
        raise ValueError("dyadic bridge shadow-state digest is stale or tampered")
    if not hmac.compare_digest(
        str(recomputed["manifest_sha256"]), result.manifest_sha256
    ):
        raise ValueError("dyadic bridge manifest digest is stale or tampered")
    return result


def _register_result(result: DyadicBridgeResult) -> None:
    result_id = id(result)

    def cleanup(reference: weakref.ReferenceType[DyadicBridgeResult]) -> None:
        with _RESULT_LOCK:
            registered = _RESULT_REGISTRY.get(result_id)
            if registered is not None and registered[0] is reference:
                _RESULT_REGISTRY.pop(result_id, None)

    reference = weakref.ref(result, cleanup)
    with _RESULT_LOCK:
        _RESULT_REGISTRY[result_id] = (reference, result.manifest_sha256)


def validate_dyadic_bridge_result(result: object) -> DyadicBridgeResult:
    """Validate semantics and require a directly produced live wrapper."""

    validated = _validate_result_semantics(result)
    with _RESULT_LOCK:
        registered = _RESULT_REGISTRY.get(id(validated))
        if (
            registered is None
            or registered[0]() is not validated
            or not hmac.compare_digest(registered[1], validated.manifest_sha256)
        ):
            raise ValueError(
                "dyadic bridge result is not a directly produced live object"
            )
    return validated


def deposit_edge_graph_prescribed_sigma_dyadic_panels(
    edge_graph: EdgeGraph,
    *,
    smoothing_radius: float,
    base_target_spacing: float,
    refinement_level: int,
    step: int,
) -> DyadicBridgeResult:
    """Deposit fixed-core particles on an exact dyadic per-edge panel family.

    ``smoothing_radius`` and ``base_target_spacing`` are independent physical
    inputs.  The latter defines level-zero counts, while ``refinement_level``
    multiplies every retained-edge count by exactly ``2**level``.  The frozen
    minimum-overlap contract is checked against the *base* spacing, so every
    refined level preserves or increases overlap without changing ``sigma``.

    All graph, scalar, overflow, per-edge, and total-resource checks complete
    before the shared deposition primitive allocates particle arrays.
    """

    trusted = _assert_runtime_bindings()
    plan = plan_edge_graph_prescribed_sigma_dyadic_panels(
        edge_graph,
        smoothing_radius=smoothing_radius,
        base_target_spacing=base_target_spacing,
        refinement_level=refinement_level,
        step=step,
    )
    graph = _strict_edge_graph(edge_graph, trusted=trusted)
    if plan.refinement_level == 0:
        bridge = trusted["public_deposition"](
            graph,
            smoothing_radius=plan.smoothing_radius_m,
            target_spacing=plan.base_target_spacing_m,
            step=plan.step,
        )
    else:
        count_by_edge = {
            panel.edge_key: panel.panel_count for panel in plan.edge_panels
        }
        bridge = trusted["private_deposition"](
            graph,
            count_by_edge=count_by_edge,
            sigma_by_edge={
                panel.edge_key: plan.smoothing_radius_m for panel in plan.edge_panels
            },
            step_number=plan.step,
            physical_owner=RING_PHYSICAL_OWNER,
            owner_state=DIAGNOSTIC_SHADOW_OWNER,
            particle_schema=DYADIC_PARTICLE_SCHEMA,
        )

    particle_ledger, shadow_digest = _particle_sidecar(bridge, plan)
    producer_hash = _producer_artifact_sha256()
    placeholder = DyadicBridgeResult(
        interface_id=DYADIC_BRIDGE_INTERFACE_ID,
        producer_id=DYADIC_BRIDGE_PRODUCER_ID,
        producer_artifact_sha256=producer_hash,
        plan=plan,
        particle_ledger=particle_ledger,
        shadow_state_sha256=shadow_digest,
        manifest_sha256="0" * 64,
        bridge=bridge,
    )
    result = replace(
        placeholder,
        manifest_sha256=_manifest_digest(
            producer_artifact_sha256=producer_hash,
            plan=plan,
            particle_ledger=particle_ledger,
            shadow_state_sha256=shadow_digest,
        ),
    )
    _validate_result_semantics(result)
    _register_result(result)
    return validate_dyadic_bridge_result(result)


__all__ = [
    "DYADIC_BRIDGE_INTERFACE_ID",
    "DYADIC_BRIDGE_PRODUCER_ID",
    "DYADIC_PARENT_ATTESTATION_KIND",
    "DYADIC_PARTICLE_SCHEMA",
    "DYADIC_PLAN_INTERFACE_ID",
    "DyadicBridgeResult",
    "DyadicDepositionPlan",
    "DyadicEdgePanelLedger",
    "DyadicEdgeParticleLedger",
    "MAX_PARTICLES_PER_EDGE",
    "MAX_REFINEMENT_LEVEL",
    "MAX_TOTAL_PARTICLES",
    "deposit_edge_graph_prescribed_sigma_dyadic_panels",
    "edge_graph_parent_digest_sha256",
    "plan_edge_graph_prescribed_sigma_dyadic_panels",
    "recompute_dyadic_bridge_sidecar",
    "validate_dyadic_bridge_result",
    "validate_dyadic_deposition_plan",
]
