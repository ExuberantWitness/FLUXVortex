"""Exact-append cumulative rVPM transport for the FluxV v5h research gate.

This module is deliberately a one-way, source-only mechanical adapter.  A
live node-owned DVM ribbon contributes a new release, an already transported
cloud (when present) remains the exact prefix, and the concatenated cloud is
advanced once in one shared direct-rVPM LSRK3 field.  Passive node frontiers
are replayed against the same stage-pre fields.

There is no Ptera solver, force, load, feedback, target observation, particle
sorting, welding, cancellation, deletion, or remeshing path here.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from hashlib import sha256
import hmac
import inspect
import json
from math import ceil
from numbers import Integral, Real
from pathlib import Path
import re
import threading
from typing import Any, Literal, TypeAlias
import weakref

import numpy as np
from numpy.typing import ArrayLike, NDArray

import fluxvortex.rvpm_edge_bridge as edge_bridge_module
import fluxvortex.rvpm_reference as reference_module
import fluxvortex.rvpm_transport as transport_module
from fluxvortex.rvpm_edge_bridge import (
    FROZEN_OVERLAP_LAMBDA,
    ParticleLineage,
    deposit_edge_graph_prescribed_sigma_and_spacing,
)
from fluxvortex.rvpm_reference import direct_gaussian_erf_velocity_jacobian
from fluxvortex.rvpm_transport import (
    ParticleState,
    lsrk3_step_direct,
    make_particle_state,
)

from forward_flight_benchmarks.v5h_passive_frontier_transport import (
    NodeFrontierFact,
    parent_frontier_digest_sha256,
    ribbon_parent_digest_sha256,
)


FloatArray = NDArray[np.float64]
StableId: TypeAlias = int | str
SourceFamily = Literal["lev", "tev_persisted"]

CUMULATIVE_HANDOFF_INTERFACE_ID = "fluxv-v5h-cumulative-ribbon-handoff-v2"
CUMULATIVE_CLOUD_INTERFACE_ID = "fluxv-v5h-cumulative-particle-cloud-v2"
CUMULATIVE_REPORT_INTERFACE_ID = "fluxv-v5h-cumulative-cloud-report-v2"
CUMULATIVE_PRODUCER_ID = "v5h-exact-append-one-field-lsrk3-v2"
CUMULATIVE_ATTESTATION_KIND = "same-process-weak-object-identity-v2"
CUMULATIVE_TIME_LAYER = "pre_release_t_n"
CUMULATIVE_FACT_OWNER = "rvpm_transport"
CUMULATIVE_CONTINUATION_SCOPE = "single-wing-single-source-family-cumulative-v2"
TRANSPORT_BACKEND_ID = (
    "fluxvortex.rvpm_transport.lsrk3_step_direct+"
    "gaussian_erf_passive_stage_replay-v2"
)

# These caps are part of the preregistered auxiliary-mechanics contract.  They
# are checked before edge deposition or time integration can allocate work.
MAX_TRANSPORT_SUBSTEPS = 4_096
MAX_PARTICLES_PER_RELEASE = 250_000
MAX_CUMULATIVE_PARTICLES = 1_000_000
MAX_PARTICLE_COUNT = MAX_CUMULATIVE_PARTICLES

# Bind the exact imported callables once.  Runtime checks below require both
# the local call target and its defining-module attribute to remain these
# objects, and bind their source text into the transport artifact digest.
_FROZEN_DEPOSITION_CALLABLE = deposit_edge_graph_prescribed_sigma_and_spacing
_FROZEN_FIELD_CALLABLE = direct_gaussian_erf_velocity_jacobian
_FROZEN_LSRK3_CALLABLE = lsrk3_step_direct
_FROZEN_STATE_CALLABLE = make_particle_state
_FROZEN_MODULE_ARTIFACTS = tuple(
    (
        label,
        Path(module.__file__).resolve(),
        sha256(Path(module.__file__).resolve().read_bytes()).hexdigest(),
    )
    for label, module in (
        ("rvpm_edge_bridge", edge_bridge_module),
        ("rvpm_reference", reference_module),
        ("rvpm_transport", transport_module),
    )
)


@dataclass(frozen=True, slots=True, weakref_slot=True)
class CumulativeRibbonHandoff:
    """Live attestation of one mapper's current release-layer ribbon."""

    interface_id: str
    producer_id: str
    producer_artifact_sha256: str
    attestation_kind: str
    wing_id: StableId
    source_family: SourceFamily
    source_step_index: int
    source_time_s: float
    current_ribbon_digest_sha256: str
    previous_report_sha256: str | None
    previous_cloud_sha256: str | None
    mapper_commit_version: int
    active_birth_count: int
    continuous_birth_count: int
    restart_birth_count: int
    handoff_sha256: str
    current_ribbon: object = field(repr=False, compare=False)
    previous_report: CumulativeCloudTransportReport | None = field(
        default=None,
        repr=False,
        compare=False,
    )


@dataclass(frozen=True, slots=True)
class ReleaseSliceLedger:
    """Exact half-open particle slice owned by one physical source release."""

    release_index: int
    source_step_index: int
    source_time_s: float
    start_index: int
    stop_index: int
    particle_count: int
    parent_ribbon_digest_sha256: str
    deposited_cloud_digest_sha256: str
    smoothing_radius_m: float
    deposition_target_spacing_m: float
    particle_ids_sha256: str
    lineage_sha256: str


@dataclass(frozen=True, slots=True)
class CumulativeParticleCloud:
    """Immutable release-ordered particle state and ownership ledger."""

    interface_id: str
    positions_gp1_m: tuple[tuple[float, float, float], ...]
    gamma_vector_m3_per_s: tuple[tuple[float, float, float], ...]
    sigma_m: tuple[float, ...]
    particle_ids: tuple[tuple[Any, ...], ...]
    lineage: tuple[ParticleLineage, ...]
    release_slices: tuple[ReleaseSliceLedger, ...]
    cloud_sha256: str


@dataclass(frozen=True, slots=True, weakref_slot=True)
class CumulativeCloudTransportReport:
    """Live-attested result of one exact append and one-field advance."""

    enabled: bool
    interface_id: str
    producer_id: str
    producer_artifact_sha256: str
    transport_backend_id: str
    transport_artifact_sha256: str
    wing_id: StableId | None
    source_family: SourceFamily | None
    parent_source_step_index: int | None
    for_source_step_index: int | None
    time_layer: str
    transport_start_time_s: float | None
    transport_end_time_s: float | None
    transport_substeps: int
    freestream_velocity_gp1_m_per_s: tuple[float, float, float]
    parent_ribbon_digest_sha256: str | None
    current_ribbon_digest_sha256: str | None
    handoff_sha256: str | None
    parent_report_sha256: str | None
    parent_cloud_digest_before_append_sha256: str | None
    deposited_new_release_digest_sha256: str | None
    appended_cloud_digest_before_transport_sha256: str | None
    transported_cloud_digest_after_sha256: str
    smoothing_radius_m: float | None
    deposition_target_spacing_m: float | None
    previous_particle_count: int
    new_particle_count: int
    total_particle_count: int
    transported_particle_cloud: CumulativeParticleCloud
    facts: tuple[NodeFrontierFact, ...]
    report_sha256: str
    attestation_kind: str
    continuation_scope: str
    exact_append_passed: bool
    one_combined_field_passed: bool
    stage_pre_replay_passed: bool
    deposition_call_count: int
    predicted_new_particle_count: int
    lsrk3_call_count: int
    lsrk3_stage_count: int
    stage_pre_field_call_count: int
    combined_stage_particle_counts: tuple[int, ...]
    exact_append_prefix_max_abs: float
    stage_pre_replay_max_abs: float
    transport_trace: tuple[str, ...]
    transport_trace_sha256: str
    edge_bridge_artifact_sha256: str
    sort_count: int
    weld_count: int
    delete_count: int
    cancel_count: int
    remesh_count: int
    feedback_call_count: int
    parent_write_count: int
    load_write_count: int
    observation_access: str
    target_case_branch: str


_LOCK = threading.RLock()
_HANDOFF_REGISTRY: dict[
    int, tuple[weakref.ReferenceType[CumulativeRibbonHandoff], str]
] = {}
_REPORT_REGISTRY: dict[
    int, tuple[weakref.ReferenceType[CumulativeCloudTransportReport], str]
] = {}
_RIBBON_CONSUMPTIONS: dict[int, tuple[weakref.ReferenceType[object], str]] = {}
_REPORT_PARENT_CONSUMPTIONS: dict[
    int, tuple[weakref.ReferenceType[CumulativeCloudTransportReport], str]
] = {}
_HANDOFF_CONSUMPTIONS: dict[
    int, tuple[weakref.ReferenceType[CumulativeRibbonHandoff], str]
] = {}


def _stable_id(name: str, value: object) -> StableId:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an explicit integer or string ID")
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, str) and value:
        return value
    raise ValueError(f"{name} must be an explicit integer or string ID")


def _id_payload(value: StableId) -> list[object]:
    return ["integer" if isinstance(value, int) else "string", value]


def _id_key(value: StableId) -> tuple[str, int | str]:
    return ("integer", value) if isinstance(value, int) else ("string", value)


def _finite_real(name: str, value: object) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be a finite real number")
    return result


def _positive_real(name: str, value: object) -> float:
    result = _finite_real(name, value)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
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


def _finite_vector(name: str, value: ArrayLike) -> FloatArray:
    original = np.asarray(value)
    if original.dtype.kind not in "iuf" or original.dtype.kind == "b":
        raise ValueError(f"{name} must use a real numeric dtype")
    result = np.asarray(original, dtype=np.float64)
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite length-3 vector")
    return np.ascontiguousarray(result)


def _lower_sha256(name: str, value: object) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _producer_artifact_sha256() -> str:
    return sha256(Path(__file__).resolve().read_bytes()).hexdigest()


def _callable_source_sha256(name: str, value: object) -> str:
    if not callable(value):
        raise ValueError(f"{name} transport binding is not callable")
    try:
        source = inspect.getsource(value).encode("utf-8")
    except (OSError, TypeError) as error:
        raise ValueError(f"{name} transport binding has no auditable source") from error
    return sha256(source).hexdigest()


_FROZEN_CALLABLE_BINDINGS = (
    (
        "edge_deposition",
        _FROZEN_DEPOSITION_CALLABLE,
        edge_bridge_module,
        "deposit_edge_graph_prescribed_sigma_and_spacing",
        "fluxvortex.rvpm_edge_bridge",
        "deposit_edge_graph_prescribed_sigma_and_spacing",
        _callable_source_sha256("edge_deposition", _FROZEN_DEPOSITION_CALLABLE),
    ),
    (
        "direct_field",
        _FROZEN_FIELD_CALLABLE,
        reference_module,
        "direct_gaussian_erf_velocity_jacobian",
        "fluxvortex.rvpm_reference",
        "direct_gaussian_erf_velocity_jacobian",
        _callable_source_sha256("direct_field", _FROZEN_FIELD_CALLABLE),
    ),
    (
        "lsrk3",
        _FROZEN_LSRK3_CALLABLE,
        transport_module,
        "lsrk3_step_direct",
        "fluxvortex.rvpm_transport",
        "lsrk3_step_direct",
        _callable_source_sha256("lsrk3", _FROZEN_LSRK3_CALLABLE),
    ),
    (
        "particle_state",
        _FROZEN_STATE_CALLABLE,
        transport_module,
        "make_particle_state",
        "fluxvortex.rvpm_transport",
        "make_particle_state",
        _callable_source_sha256("particle_state", _FROZEN_STATE_CALLABLE),
    ),
)


def _freeze_transitive_callable_bindings(
    roots: tuple[tuple[Any, ...], ...],
) -> tuple[tuple[Any, ...], ...]:
    """Freeze every Python-function global reached by a transport root.

    A function object is not a closed implementation: bytecode resolves helper
    functions through its mutable ``__globals__`` mapping at call time.  Record
    those slots recursively so replacing a private helper (or an imported field
    kernel used inside LSRK3) cannot leave the public root identity unchanged
    while altering the computation.
    """

    pending = [(str(binding[0]), binding[1]) for binding in roots]
    visited_functions: set[int] = set()
    visited_slots: set[tuple[int, str]] = set()
    dependencies: list[tuple[Any, ...]] = []
    while pending:
        root_label, function = pending.pop(0)
        function_id = id(function)
        if function_id in visited_functions:
            continue
        visited_functions.add(function_id)
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
                dependency_module = getattr(dependency, "__module__", None)
                dependency_qualname = getattr(dependency, "__qualname__", None)
                if (
                    not isinstance(dependency_module, str)
                    or not dependency_module
                    or not isinstance(dependency_qualname, str)
                    or not dependency_qualname
                ):
                    raise ValueError(
                        f"{root_label} dependency {owner_module}.{attribute} "
                        "has no stable identity"
                    )
                dependencies.append(
                    (
                        root_label,
                        function_globals,
                        owner_module,
                        attribute,
                        dependency,
                        dependency_module,
                        dependency_qualname,
                        _callable_source_sha256(
                            f"{owner_module}.{attribute}", dependency
                        ),
                    )
                )
            pending.append((root_label, dependency))
    return tuple(
        sorted(
            dependencies,
            key=lambda binding: (
                str(binding[2]),
                str(binding[3]),
                str(binding[5]),
                str(binding[6]),
            ),
        )
    )


_FROZEN_TRANSITIVE_CALLABLE_BINDINGS = _freeze_transitive_callable_bindings(
    _FROZEN_CALLABLE_BINDINGS
)


def _assert_transport_bindings(
    _root_bindings: tuple[tuple[Any, ...], ...] = _FROZEN_CALLABLE_BINDINGS,
    _dependency_bindings: tuple[
        tuple[Any, ...], ...
    ] = _FROZEN_TRANSITIVE_CALLABLE_BINDINGS,
) -> tuple[tuple[dict[str, str], ...], tuple[dict[str, str], ...], dict[str, Any],]:
    root_metadata: list[dict[str, str]] = []
    dependency_metadata: list[dict[str, str]] = []
    local_targets = {
        "edge_deposition": deposit_edge_graph_prescribed_sigma_and_spacing,
        "direct_field": direct_gaussian_erf_velocity_jacobian,
        "lsrk3": lsrk3_step_direct,
        "particle_state": make_particle_state,
    }
    runtime_frozen_targets = {
        "edge_deposition": globals().get("_FROZEN_DEPOSITION_CALLABLE"),
        "direct_field": globals().get("_FROZEN_FIELD_CALLABLE"),
        "lsrk3": globals().get("_FROZEN_LSRK3_CALLABLE"),
        "particle_state": globals().get("_FROZEN_STATE_CALLABLE"),
    }
    trusted_callables: dict[str, Any] = {}
    for (
        label,
        frozen,
        module,
        attribute,
        expected_module,
        expected_qualname,
        expected_source_sha256,
    ) in _root_bindings:
        current_module_target = getattr(module, attribute, None)
        current_local_target = local_targets[label]
        current_frozen_target = runtime_frozen_targets[label]
        if (
            current_module_target is not frozen
            or current_local_target is not frozen
            or current_frozen_target is not frozen
        ):
            raise ValueError(f"{label} transport callable was replaced")
        if (
            getattr(frozen, "__module__", None) != expected_module
            or getattr(frozen, "__qualname__", None) != expected_qualname
        ):
            raise ValueError(f"{label} transport callable identity is foreign")
        source_sha256 = _callable_source_sha256(label, frozen)
        if not hmac.compare_digest(source_sha256, expected_source_sha256):
            raise ValueError(f"{label} transport callable source changed")
        root_metadata.append(
            {
                "label": label,
                "module": expected_module,
                "qualname": expected_qualname,
                "source_sha256": source_sha256,
            }
        )
        trusted_callables[label] = frozen

    for (
        root_label,
        owner_globals,
        owner_module,
        attribute,
        frozen,
        expected_module,
        expected_qualname,
        expected_source_sha256,
    ) in _dependency_bindings:
        current = owner_globals.get(attribute)
        if current is not frozen:
            raise ValueError(
                f"{root_label} transitive dependency "
                f"{owner_module}.{attribute} was replaced"
            )
        if (
            getattr(frozen, "__module__", None) != expected_module
            or getattr(frozen, "__qualname__", None) != expected_qualname
        ):
            raise ValueError(
                f"{root_label} transitive dependency "
                f"{owner_module}.{attribute} has foreign identity"
            )
        source_sha256 = _callable_source_sha256(f"{owner_module}.{attribute}", frozen)
        if not hmac.compare_digest(source_sha256, expected_source_sha256):
            raise ValueError(
                f"{root_label} transitive dependency "
                f"{owner_module}.{attribute} source changed"
            )
        dependency_metadata.append(
            {
                "root_label": root_label,
                "owner_module": owner_module,
                "attribute": attribute,
                "module": expected_module,
                "qualname": expected_qualname,
                "source_sha256": source_sha256,
            }
        )
    return (
        tuple(root_metadata),
        tuple(dependency_metadata),
        trusted_callables,
    )


def _edge_bridge_artifact_sha256() -> str:
    label, frozen_path, frozen_digest = _FROZEN_MODULE_ARTIFACTS[0]
    current_path = Path(edge_bridge_module.__file__).resolve()
    current_digest = sha256(current_path.read_bytes()).hexdigest()
    if label != "rvpm_edge_bridge" or current_path != frozen_path:
        raise ValueError("edge bridge artifact path changed after import")
    if not hmac.compare_digest(current_digest, frozen_digest):
        raise ValueError("edge bridge source changed after import")
    return current_digest


def _transport_artifact_sha256() -> str:
    root_bindings, dependency_bindings, _ = _assert_transport_bindings()
    digest = sha256()
    modules = {
        "rvpm_edge_bridge": edge_bridge_module,
        "rvpm_reference": reference_module,
        "rvpm_transport": transport_module,
    }
    for label, frozen_path, frozen_digest in _FROZEN_MODULE_ARTIFACTS:
        module = modules[label]
        digest.update(label.encode("ascii"))
        digest.update(b"\0")
        current_path = Path(module.__file__).resolve()
        current_bytes = current_path.read_bytes()
        current_digest = sha256(current_bytes).hexdigest()
        if current_path != frozen_path:
            raise ValueError(f"{label} source artifact path changed after import")
        if not hmac.compare_digest(current_digest, frozen_digest):
            raise ValueError(f"{label} module source changed after import")
        digest.update(current_bytes)
        digest.update(b"\0")
    digest.update(
        json.dumps(
            {
                "root_bindings": root_bindings,
                "transitive_callable_bindings": dependency_bindings,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    return digest.hexdigest()


def _trace_sha256(trace: tuple[str, ...]) -> str:
    if not isinstance(trace, tuple) or any(
        not isinstance(item, str) or not item for item in trace
    ):
        raise ValueError("transport trace must contain explicit string events")
    digest = sha256(b"fluxv-v5h-cumulative-transport-trace-v2\0")
    for event in trace:
        digest.update(event.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _max_abs(*values: ArrayLike) -> float:
    maximum = 0.0
    for value in values:
        array = np.asarray(value, dtype=np.float64)
        if array.size:
            if not np.all(np.isfinite(array)):
                return float("inf")
            maximum = max(maximum, float(np.max(np.abs(array))))
    return maximum


def _difference_max_abs(left: ArrayLike, right: ArrayLike) -> float:
    left_array = np.asarray(left, dtype=np.float64)
    right_array = np.asarray(right, dtype=np.float64)
    if left_array.shape != right_array.shape:
        return float("inf")
    return _max_abs(left_array - right_array)


def _predict_release_particle_count(
    graph: object | None,
    *,
    smoothing_radius_m: float,
    target_spacing_m: float,
) -> int:
    """Bound a prescribed-spacing release before the bridge allocates arrays."""

    if smoothing_radius_m / target_spacing_m < FROZEN_OVERLAP_LAMBDA:
        raise ValueError("target spacing violates the frozen minimum overlap")
    if graph is None:
        return 0
    retained_edges = tuple(getattr(graph, "retained_edges"))
    predicted = 0
    for edge in retained_edges:
        start = np.asarray(edge.start_position, dtype=np.float64)
        end = np.asarray(edge.end_position, dtype=np.float64)
        if start.shape != (3,) or end.shape != (3,):
            raise ValueError("edge positions must be finite length-3 vectors")
        length = float(np.linalg.norm(end - start))
        if not np.isfinite(length) or length <= 0.0:
            raise ValueError("retained edge has nonpositive or non-finite length")
        quotient = length / target_spacing_m
        if not np.isfinite(quotient) or quotient > MAX_PARTICLES_PER_RELEASE:
            raise ValueError("predicted release particle count exceeds resource cap")
        predicted += max(1, ceil(quotient))
        if predicted > MAX_PARTICLES_PER_RELEASE:
            raise ValueError("predicted release particle count exceeds resource cap")
    return predicted


def _canonical(value: object) -> object:
    if value is None:
        return ["none"]
    if isinstance(value, (bool, np.bool_)):
        return ["boolean", bool(value)]
    if isinstance(value, Integral):
        return ["integer", int(value)]
    if isinstance(value, str):
        return ["string", value]
    if isinstance(value, Real):
        return ["real-hex", _finite_real("canonical real", value).hex()]
    if isinstance(value, (tuple, list)):
        return ["sequence", [_canonical(item) for item in value]]
    raise ValueError(f"unsupported canonical value {type(value).__name__}")


def _lineage_payload(item: ParticleLineage) -> dict[str, object]:
    if not isinstance(item, ParticleLineage):
        raise ValueError("particle lineage has a foreign schema")
    subdivision_index = _nonnegative_integer(
        "lineage.subdivision_index", item.subdivision_index
    )
    subdivision_count = _positive_integer(
        "lineage.subdivision_count", item.subdivision_count
    )
    if subdivision_index >= subdivision_count:
        raise ValueError("particle lineage subdivision index is out of range")
    step = _nonnegative_integer("lineage.step", item.step)
    if not isinstance(item.physical_owner, str) or not item.physical_owner:
        raise ValueError("particle lineage physical owner must be explicit")
    if not isinstance(item.owner_state, str) or not item.owner_state:
        raise ValueError("particle lineage owner state must be explicit")
    return {
        "particle_id": _canonical(item.particle_id),
        "source_edge": _canonical(item.source_edge),
        "subdivision_index": subdivision_index,
        "subdivision_count": subdivision_count,
        "ring_incidences": _canonical(
            tuple(
                (
                    incidence.ring_id,
                    incidence.traversal_index,
                    incidence.source_start_id,
                    incidence.source_end_id,
                    incidence.canonical_sign,
                    incidence.ring_circulation,
                    incidence.signed_circulation,
                )
                for incidence in item.ring_incidences
            )
        ),
        "step": step,
        "physical_owner": item.physical_owner,
        "owner_state": item.owner_state,
    }


def _identity_digest(
    particle_ids: tuple[tuple[Any, ...], ...],
    lineage: tuple[ParticleLineage, ...],
) -> str:
    payload = {
        "particle_ids": [_canonical(item) for item in particle_ids],
        "lineage": [_lineage_payload(item) for item in lineage],
    }
    return sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _array_digest(
    positions: ArrayLike,
    gamma: ArrayLike,
    sigma: ArrayLike,
    particle_ids: tuple[tuple[Any, ...], ...],
    lineage: tuple[ParticleLineage, ...],
    *,
    domain: bytes,
) -> str:
    digest = sha256(domain + b"\0")
    for name, value in (
        ("positions", positions),
        ("gamma", gamma),
        ("sigma", sigma),
    ):
        array = np.ascontiguousarray(np.asarray(value, dtype="<f8"))
        digest.update(name.encode("ascii") + b"\0")
        digest.update(json.dumps(array.shape, separators=(",", ":")).encode("ascii"))
        digest.update(b"\0" + array.tobytes() + b"\0")
    digest.update(_identity_digest(particle_ids, lineage).encode("ascii"))
    return digest.hexdigest()


def _slice_payload(item: ReleaseSliceLedger) -> dict[str, object]:
    release_index = _positive_integer("release.release_index", item.release_index)
    source_step = _positive_integer("release.source_step_index", item.source_step_index)
    source_time = _finite_real("release.source_time_s", item.source_time_s)
    start_index = _nonnegative_integer("release.start_index", item.start_index)
    stop_index = _nonnegative_integer("release.stop_index", item.stop_index)
    particle_count = _positive_integer("release.particle_count", item.particle_count)
    smoothing_radius = _positive_real(
        "release.smoothing_radius_m", item.smoothing_radius_m
    )
    target_spacing = _positive_real(
        "release.deposition_target_spacing_m",
        item.deposition_target_spacing_m,
    )
    return {
        "release_index": release_index,
        "source_step_index": source_step,
        "source_time_hex": source_time.hex(),
        "start_index": start_index,
        "stop_index": stop_index,
        "particle_count": particle_count,
        "parent_ribbon_digest_sha256": _lower_sha256(
            "release.parent_ribbon_digest_sha256",
            item.parent_ribbon_digest_sha256,
        ),
        "deposited_cloud_digest_sha256": _lower_sha256(
            "release.deposited_cloud_digest_sha256",
            item.deposited_cloud_digest_sha256,
        ),
        "smoothing_radius_hex": smoothing_radius.hex(),
        "deposition_target_spacing_hex": target_spacing.hex(),
        "particle_ids_sha256": _lower_sha256(
            "release.particle_ids_sha256", item.particle_ids_sha256
        ),
        "lineage_sha256": _lower_sha256("release.lineage_sha256", item.lineage_sha256),
    }


def _cloud_digest(
    positions: ArrayLike,
    gamma: ArrayLike,
    sigma: ArrayLike,
    particle_ids: tuple[tuple[Any, ...], ...],
    lineage: tuple[ParticleLineage, ...],
    slices: tuple[ReleaseSliceLedger, ...],
) -> str:
    digest = sha256()
    digest.update(
        _array_digest(
            positions,
            gamma,
            sigma,
            particle_ids,
            lineage,
            domain=b"fluxv-v5h-cumulative-cloud-v2",
        ).encode("ascii")
    )
    digest.update(
        json.dumps(
            [_slice_payload(item) for item in slices],
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    return digest.hexdigest()


def _cloud_arrays(
    cloud: CumulativeParticleCloud,
    *,
    allow_empty: bool,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    count = _cloud_particle_count_gate(cloud, allow_empty=allow_empty)
    for name, value in (
        ("positions_gp1_m", cloud.positions_gp1_m),
        ("gamma_vector_m3_per_s", cloud.gamma_vector_m3_per_s),
        ("sigma_m", cloud.sigma_m),
        ("particle_ids", cloud.particle_ids),
        ("lineage", cloud.lineage),
        ("release_slices", cloud.release_slices),
    ):
        if not isinstance(value, tuple):
            raise ValueError(
                f"transported particle cloud {name} has a foreign mutable schema"
            )
    positions = np.ascontiguousarray(cloud.positions_gp1_m, dtype=np.float64)
    gamma = np.ascontiguousarray(cloud.gamma_vector_m3_per_s, dtype=np.float64)
    sigma = np.ascontiguousarray(cloud.sigma_m, dtype=np.float64)
    if count == 0:
        positions = positions.reshape((0, 3))
        gamma = gamma.reshape((0, 3))
    if positions.shape != (count, 3) or gamma.shape != (count, 3):
        raise ValueError("transported particle cloud vector shape is invalid")
    if sigma.shape != (count,) or len(cloud.lineage) != count:
        raise ValueError("transported particle cloud scalar/lineage shape is invalid")
    if not (
        np.all(np.isfinite(positions))
        and np.all(np.isfinite(gamma))
        and np.all(np.isfinite(sigma))
    ):
        raise ValueError("transported particle cloud contains non-finite values")
    if count and np.any(sigma <= 0.0):
        raise ValueError("transported particle cloud contains nonpositive sigma")
    if len(set(cloud.particle_ids)) != count:
        raise ValueError("transported particle cloud contains duplicate particle IDs")
    for index, item in enumerate(cloud.lineage):
        _lineage_payload(item)
        if item.particle_id != cloud.particle_ids[index]:
            raise ValueError("particle lineage/ID order disagrees")
    cursor = 0
    for release_index, item in enumerate(cloud.release_slices, start=1):
        if not isinstance(item, ReleaseSliceLedger):
            raise ValueError("release slice has a foreign schema")
        validated_release_index = _positive_integer(
            "release.release_index", item.release_index
        )
        source_step = _positive_integer(
            "release.source_step_index", item.source_step_index
        )
        start_index = _nonnegative_integer("release.start_index", item.start_index)
        stop_index = _nonnegative_integer("release.stop_index", item.stop_index)
        particle_count = _positive_integer(
            "release.particle_count", item.particle_count
        )
        _finite_real("release.source_time_s", item.source_time_s)
        if validated_release_index != release_index or start_index != cursor:
            raise ValueError("release slices are reordered or noncontiguous")
        if stop_index - start_index != particle_count:
            raise ValueError("release slice particle count is inconsistent")
        if stop_index > count or particle_count > MAX_PARTICLES_PER_RELEASE:
            raise ValueError("release slice bounds are invalid")
        ids = cloud.particle_ids[start_index:stop_index]
        lineages = cloud.lineage[start_index:stop_index]
        if any(lineage.step != source_step for lineage in lineages):
            raise ValueError("release slice source-step lineage is inconsistent")
        identity_digest = _identity_digest(ids, lineages)
        if item.particle_ids_sha256 != identity_digest:
            raise ValueError("release slice particle-ID digest is inconsistent")
        if item.lineage_sha256 != identity_digest:
            raise ValueError("release slice lineage digest is inconsistent")
        _lower_sha256("release parent ribbon digest", item.parent_ribbon_digest_sha256)
        _lower_sha256("release deposited digest", item.deposited_cloud_digest_sha256)
        _positive_real("release smoothing radius", item.smoothing_radius_m)
        _positive_real("release target spacing", item.deposition_target_spacing_m)
        cursor = stop_index
    if cursor != count and count:
        raise ValueError("release slices do not cover the particle cloud")
    expected_digest = _cloud_digest(
        positions,
        gamma,
        sigma,
        cloud.particle_ids,
        cloud.lineage,
        cloud.release_slices,
    )
    if not hmac.compare_digest(expected_digest, cloud.cloud_sha256):
        raise ValueError("transported particle cloud digest is inconsistent")
    return positions, gamma, sigma


def _cloud_particle_count_gate(
    cloud: object,
    *,
    allow_empty: bool,
) -> int:
    """Reject oversized clouds before touching any array-like field."""

    if not isinstance(cloud, CumulativeParticleCloud):
        raise ValueError("transported particle cloud has a foreign schema")
    if cloud.interface_id != CUMULATIVE_CLOUD_INTERFACE_ID:
        raise ValueError("transported particle cloud has a foreign interface")
    try:
        count = len(cloud.particle_ids)
    except (OverflowError, TypeError) as error:
        raise ValueError(
            "transported particle cloud has no finite particle count"
        ) from error
    if count > MAX_CUMULATIVE_PARTICLES:
        raise ValueError("transported particle cloud exceeds cumulative resource cap")
    if not allow_empty and count == 0:
        raise ValueError("enabled cumulative cloud must not be empty")
    return count


def _handoff_payload(handoff: CumulativeRibbonHandoff) -> dict[str, object]:
    return {
        "interface_id": handoff.interface_id,
        "producer_id": handoff.producer_id,
        "producer_artifact_sha256": handoff.producer_artifact_sha256,
        "attestation_kind": handoff.attestation_kind,
        "wing_id": _id_payload(handoff.wing_id),
        "source_family": handoff.source_family,
        "source_step_index": _positive_integer(
            "handoff.source_step_index", handoff.source_step_index
        ),
        "source_time_hex": _finite_real(
            "handoff.source_time_s", handoff.source_time_s
        ).hex(),
        "current_ribbon_digest_sha256": _lower_sha256(
            "handoff.current_ribbon_digest_sha256",
            handoff.current_ribbon_digest_sha256,
        ),
        "previous_report_sha256": handoff.previous_report_sha256,
        "previous_cloud_sha256": handoff.previous_cloud_sha256,
        "mapper_commit_version": _positive_integer(
            "handoff.mapper_commit_version", handoff.mapper_commit_version
        ),
        "active_birth_count": _nonnegative_integer(
            "handoff.active_birth_count", handoff.active_birth_count
        ),
        "continuous_birth_count": _nonnegative_integer(
            "handoff.continuous_birth_count", handoff.continuous_birth_count
        ),
        "restart_birth_count": _nonnegative_integer(
            "handoff.restart_birth_count", handoff.restart_birth_count
        ),
    }


def _handoff_digest(handoff: CumulativeRibbonHandoff) -> str:
    return sha256(
        json.dumps(
            _handoff_payload(handoff), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def _report_payload(report: CumulativeCloudTransportReport) -> dict[str, object]:
    facts = []
    for fact in report.facts:
        facts.append(
            {
                "wing_id": _id_payload(_stable_id("fact.wing_id", fact.wing_id)),
                "node_id": _id_payload(_stable_id("fact.node_id", fact.node_id)),
                "source_family": fact.source_family,
                "lineage_epoch": _nonnegative_integer(
                    "fact.lineage_epoch", fact.lineage_epoch
                ),
                "parent_frontier_id": fact.parent_frontier_id,
                "parent_frontier_digest_sha256": fact.parent_frontier_digest_sha256,
                "parent_ribbon_digest_sha256": fact.parent_ribbon_digest_sha256,
                "parent_birth_step_index": _positive_integer(
                    "fact.parent_birth_step_index", fact.parent_birth_step_index
                ),
                "for_source_step_index": _positive_integer(
                    "fact.for_source_step_index", fact.for_source_step_index
                ),
                "time_layer": fact.time_layer,
                "transport_backend_id": fact.transport_backend_id,
                "transport_artifact_sha256": fact.transport_artifact_sha256,
                "producer_id": fact.producer_id,
                "producer_artifact_sha256": fact.producer_artifact_sha256,
                "transport_start_time_hex": _finite_real(
                    "fact.transport_start_time_s", fact.transport_start_time_s
                ).hex(),
                "transport_end_time_hex": _finite_real(
                    "fact.transport_end_time_s", fact.transport_end_time_s
                ).hex(),
                "transport_substeps": _positive_integer(
                    "fact.transport_substeps", fact.transport_substeps
                ),
                "advected_position_hex": [
                    float(value).hex()
                    for value in _finite_vector(
                        "fact.advected_position_gp1_m",
                        fact.advected_position_gp1_m,
                    )
                ],
                "fact_owner": fact.fact_owner,
            }
        )
    return {
        name: value
        for name, value in {
            "enabled": report.enabled,
            "interface_id": report.interface_id,
            "producer_id": report.producer_id,
            "producer_artifact_sha256": report.producer_artifact_sha256,
            "transport_backend_id": report.transport_backend_id,
            "transport_artifact_sha256": report.transport_artifact_sha256,
            "wing_id": None if report.wing_id is None else _id_payload(report.wing_id),
            "source_family": report.source_family,
            "parent_source_step_index": report.parent_source_step_index,
            "for_source_step_index": report.for_source_step_index,
            "time_layer": report.time_layer,
            "transport_start_time_hex": (
                None
                if report.transport_start_time_s is None
                else _finite_real(
                    "report.transport_start_time_s", report.transport_start_time_s
                ).hex()
            ),
            "transport_end_time_hex": (
                None
                if report.transport_end_time_s is None
                else _finite_real(
                    "report.transport_end_time_s", report.transport_end_time_s
                ).hex()
            ),
            "transport_substeps": report.transport_substeps,
            "freestream_hex": [
                float(value).hex()
                for value in _finite_vector(
                    "report.freestream_velocity_gp1_m_per_s",
                    report.freestream_velocity_gp1_m_per_s,
                )
            ],
            "parent_ribbon_digest_sha256": report.parent_ribbon_digest_sha256,
            "current_ribbon_digest_sha256": report.current_ribbon_digest_sha256,
            "handoff_sha256": report.handoff_sha256,
            "parent_report_sha256": report.parent_report_sha256,
            "parent_cloud_digest_before_append_sha256": (
                report.parent_cloud_digest_before_append_sha256
            ),
            "deposited_new_release_digest_sha256": (
                report.deposited_new_release_digest_sha256
            ),
            "appended_cloud_digest_before_transport_sha256": (
                report.appended_cloud_digest_before_transport_sha256
            ),
            "transported_cloud_digest_after_sha256": (
                report.transported_cloud_digest_after_sha256
            ),
            "smoothing_radius_hex": (
                None
                if report.smoothing_radius_m is None
                else _positive_real(
                    "report.smoothing_radius_m", report.smoothing_radius_m
                ).hex()
            ),
            "spacing_hex": (
                None
                if report.deposition_target_spacing_m is None
                else _positive_real(
                    "report.deposition_target_spacing_m",
                    report.deposition_target_spacing_m,
                ).hex()
            ),
            "previous_particle_count": report.previous_particle_count,
            "new_particle_count": report.new_particle_count,
            "total_particle_count": report.total_particle_count,
            "cloud_sha256": report.transported_particle_cloud.cloud_sha256,
            "facts": facts,
            "attestation_kind": report.attestation_kind,
            "continuation_scope": report.continuation_scope,
            "exact_append_passed": report.exact_append_passed,
            "one_combined_field_passed": report.one_combined_field_passed,
            "stage_pre_replay_passed": report.stage_pre_replay_passed,
            "deposition_call_count": report.deposition_call_count,
            "predicted_new_particle_count": report.predicted_new_particle_count,
            "lsrk3_call_count": report.lsrk3_call_count,
            "lsrk3_stage_count": report.lsrk3_stage_count,
            "stage_pre_field_call_count": report.stage_pre_field_call_count,
            "combined_stage_particle_counts": list(
                report.combined_stage_particle_counts
            ),
            "exact_append_prefix_max_abs_hex": _finite_real(
                "report.exact_append_prefix_max_abs",
                report.exact_append_prefix_max_abs,
            ).hex(),
            "stage_pre_replay_max_abs_hex": _finite_real(
                "report.stage_pre_replay_max_abs",
                report.stage_pre_replay_max_abs,
            ).hex(),
            "transport_trace": list(report.transport_trace),
            "transport_trace_sha256": report.transport_trace_sha256,
            "edge_bridge_artifact_sha256": report.edge_bridge_artifact_sha256,
            "sort_count": report.sort_count,
            "weld_count": report.weld_count,
            "delete_count": report.delete_count,
            "cancel_count": report.cancel_count,
            "remesh_count": report.remesh_count,
            "feedback_call_count": report.feedback_call_count,
            "parent_write_count": report.parent_write_count,
            "load_write_count": report.load_write_count,
            "observation_access": report.observation_access,
            "target_case_branch": report.target_case_branch,
        }.items()
    }


def _report_digest(report: CumulativeCloudTransportReport) -> str:
    return sha256(
        json.dumps(
            _report_payload(report), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def _register_live(registry: dict[int, Any], value: object, digest: str) -> None:
    value_id = id(value)

    def discard(reference: weakref.ReferenceType[object]) -> None:
        with _LOCK:
            existing = registry.get(value_id)
            if existing is not None and existing[0] is reference:
                registry.pop(value_id, None)

    registry[value_id] = (weakref.ref(value, discard), digest)


def _validate_handoff(handoff: object) -> CumulativeRibbonHandoff:
    if not isinstance(handoff, CumulativeRibbonHandoff):
        raise ValueError("handoff must be a directly produced live cumulative handoff")
    if (
        handoff.interface_id != CUMULATIVE_HANDOFF_INTERFACE_ID
        or handoff.producer_id != CUMULATIVE_PRODUCER_ID
        or handoff.attestation_kind != CUMULATIVE_ATTESTATION_KIND
        or handoff.producer_artifact_sha256 != _producer_artifact_sha256()
    ):
        raise ValueError("cumulative handoff producer identity is stale or foreign")
    _stable_id("handoff.wing_id", handoff.wing_id)
    if handoff.source_family not in ("lev", "tev_persisted"):
        raise ValueError("cumulative handoff source family is invalid")
    _positive_integer("handoff.source_step_index", handoff.source_step_index)
    _finite_real("handoff.source_time_s", handoff.source_time_s)
    _positive_integer("handoff.mapper_commit_version", handoff.mapper_commit_version)
    active_count = _nonnegative_integer(
        "handoff.active_birth_count", handoff.active_birth_count
    )
    continuous_count = _nonnegative_integer(
        "handoff.continuous_birth_count", handoff.continuous_birth_count
    )
    restart_count = _nonnegative_integer(
        "handoff.restart_birth_count", handoff.restart_birth_count
    )
    if continuous_count + restart_count > active_count:
        raise ValueError("cumulative handoff birth-mode counts are inconsistent")
    _lower_sha256(
        "handoff.current_ribbon_digest_sha256",
        handoff.current_ribbon_digest_sha256,
    )
    if handoff.previous_report_sha256 is None:
        if handoff.previous_cloud_sha256 is not None:
            raise ValueError("cumulative handoff has a phantom previous cloud")
    else:
        _lower_sha256("handoff.previous_report_sha256", handoff.previous_report_sha256)
        _lower_sha256("handoff.previous_cloud_sha256", handoff.previous_cloud_sha256)
    digest = _handoff_digest(handoff)
    if not hmac.compare_digest(digest, handoff.handoff_sha256):
        raise ValueError("cumulative handoff digest is inconsistent")
    registered = _HANDOFF_REGISTRY.get(id(handoff))
    if (
        registered is None
        or registered[0]() is not handoff
        or not hmac.compare_digest(registered[1], digest)
    ):
        raise ValueError("handoff is not a directly produced live cumulative object")
    return handoff


def attest_cumulative_ribbon_handoff(
    mapper: object,
    current_ribbon: object,
    *,
    wing_id: StableId,
    source_time_s: object,
    previous_report: CumulativeCloudTransportReport | None = None,
) -> CumulativeRibbonHandoff:
    """Bind the exact current mapper/ribbon state to one cumulative advance."""

    # Local import keeps the future ribbon-side lazy v2 dispatcher acyclic.
    from forward_flight_benchmarks.v5h_dvm_node_ribbon import (
        NodeOwnedDVMRibbonShadow,
        validate_live_dvm_ribbon_shadow_result,
    )

    if not isinstance(mapper, NodeOwnedDVMRibbonShadow):
        raise ValueError("mapper must be a NodeOwnedDVMRibbonShadow")
    ribbon = validate_live_dvm_ribbon_shadow_result(current_ribbon)
    wing = _stable_id("wing_id", wing_id)
    if _id_key(wing) != _id_key(getattr(mapper, "_wing_id")):
        raise ValueError("cumulative handoff is bound to another wing")
    diagnostics = ribbon.diagnostics
    if not diagnostics.enabled or ribbon.feedback_velocity is not None:
        raise ValueError("cumulative handoff requires a clean enabled ribbon")
    if diagnostics.feedback_call_count:
        raise ValueError("cumulative handoff cannot consume feedback")
    source_family = diagnostics.source_family
    if source_family != getattr(mapper, "_source_family"):
        raise ValueError("cumulative handoff source family disagrees with mapper")
    step = _positive_integer("source_step_index", diagnostics.source_step_index)
    source_time = _finite_real("source_time_s", source_time_s)
    current_digest = ribbon_parent_digest_sha256(ribbon)
    last_digest, _ = mapper.transport_handoff_snapshot
    if current_digest != last_digest:
        raise ValueError("current ribbon is stale for this mapper")
    commit_version = int(getattr(mapper, "_commit_version"))
    active_births = tuple(birth for birth in ribbon.node_births if birth.active)
    continuous_count = sum(birth.mode == "continuous" for birth in active_births)
    restart_count = sum(birth.mode == "restart" for birth in active_births)

    previous_digest: str | None = None
    previous_cloud_digest: str | None = None
    if previous_report is None:
        if step != 1:
            raise ValueError(
                "source steps after one require a cumulative parent report"
            )
        if continuous_count:
            raise ValueError(
                "a first cumulative handoff cannot contain continuous nodes"
            )
    else:
        parent = validate_cumulative_cloud_transport_report(previous_report)
        if not parent.enabled:
            raise ValueError("a disabled report cannot parent a cumulative handoff")
        if (
            parent.wing_id != wing
            or parent.source_family != source_family
            or parent.for_source_step_index != step
        ):
            raise ValueError("parent report is stale or bound to another wing/family")
        if parent.transport_end_time_s != source_time:
            raise ValueError("parent report is stale for the source time layer")
        previous_digest = parent.report_sha256
        previous_cloud_digest = parent.transported_particle_cloud.cloud_sha256

    producer_hash = _producer_artifact_sha256()
    placeholder = CumulativeRibbonHandoff(
        interface_id=CUMULATIVE_HANDOFF_INTERFACE_ID,
        producer_id=CUMULATIVE_PRODUCER_ID,
        producer_artifact_sha256=producer_hash,
        attestation_kind=CUMULATIVE_ATTESTATION_KIND,
        wing_id=wing,
        source_family=source_family,
        source_step_index=step,
        source_time_s=source_time,
        current_ribbon_digest_sha256=current_digest,
        previous_report_sha256=previous_digest,
        previous_cloud_sha256=previous_cloud_digest,
        mapper_commit_version=commit_version,
        active_birth_count=len(active_births),
        continuous_birth_count=continuous_count,
        restart_birth_count=restart_count,
        handoff_sha256="0" * 64,
        current_ribbon=ribbon,
        previous_report=previous_report,
    )
    handoff = replace(placeholder, handoff_sha256=_handoff_digest(placeholder))
    with _LOCK:
        if getattr(mapper, "_commit_version") != commit_version:
            raise RuntimeError("ribbon mapper changed during cumulative attestation")
        consumed = _RIBBON_CONSUMPTIONS.get(id(ribbon))
        if consumed is not None and consumed[0]() is ribbon:
            raise ValueError("live ribbon was already consumed by a cumulative handoff")
        if previous_report is not None:
            parent_consumed = _REPORT_PARENT_CONSUMPTIONS.get(id(previous_report))
            if parent_consumed is not None and parent_consumed[0]() is previous_report:
                raise ValueError(
                    "cumulative parent report was already consumed (replay)"
                )
        _register_live(_HANDOFF_REGISTRY, handoff, handoff.handoff_sha256)
        _register_live(_RIBBON_CONSUMPTIONS, ribbon, handoff.handoff_sha256)
        if previous_report is not None:
            _register_live(
                _REPORT_PARENT_CONSUMPTIONS,
                previous_report,
                handoff.handoff_sha256,
            )
    return _validate_handoff(handoff)


def _empty_cloud() -> CumulativeParticleCloud:
    positions = np.empty((0, 3), dtype=np.float64)
    gamma = np.empty((0, 3), dtype=np.float64)
    sigma = np.empty((0,), dtype=np.float64)
    digest = _cloud_digest(positions, gamma, sigma, (), (), ())
    return CumulativeParticleCloud(
        interface_id=CUMULATIVE_CLOUD_INTERFACE_ID,
        positions_gp1_m=(),
        gamma_vector_m3_per_s=(),
        sigma_m=(),
        particle_ids=(),
        lineage=(),
        release_slices=(),
        cloud_sha256=digest,
    )


def _disabled_report() -> CumulativeCloudTransportReport:
    cloud = _empty_cloud()
    producer_hash = _producer_artifact_sha256()
    transport_hash = _transport_artifact_sha256()
    transport_trace = ("disabled:input-blind",)
    placeholder = CumulativeCloudTransportReport(
        enabled=False,
        interface_id=CUMULATIVE_REPORT_INTERFACE_ID,
        producer_id=CUMULATIVE_PRODUCER_ID,
        producer_artifact_sha256=producer_hash,
        transport_backend_id=TRANSPORT_BACKEND_ID,
        transport_artifact_sha256=transport_hash,
        wing_id=None,
        source_family=None,
        parent_source_step_index=None,
        for_source_step_index=None,
        time_layer=CUMULATIVE_TIME_LAYER,
        transport_start_time_s=None,
        transport_end_time_s=None,
        transport_substeps=0,
        freestream_velocity_gp1_m_per_s=(0.0, 0.0, 0.0),
        parent_ribbon_digest_sha256=None,
        current_ribbon_digest_sha256=None,
        handoff_sha256=None,
        parent_report_sha256=None,
        parent_cloud_digest_before_append_sha256=None,
        deposited_new_release_digest_sha256=None,
        appended_cloud_digest_before_transport_sha256=None,
        transported_cloud_digest_after_sha256=cloud.cloud_sha256,
        smoothing_radius_m=None,
        deposition_target_spacing_m=None,
        previous_particle_count=0,
        new_particle_count=0,
        total_particle_count=0,
        transported_particle_cloud=cloud,
        facts=(),
        report_sha256="0" * 64,
        attestation_kind=CUMULATIVE_ATTESTATION_KIND,
        continuation_scope=CUMULATIVE_CONTINUATION_SCOPE,
        exact_append_passed=False,
        one_combined_field_passed=False,
        stage_pre_replay_passed=False,
        deposition_call_count=0,
        predicted_new_particle_count=0,
        lsrk3_call_count=0,
        lsrk3_stage_count=0,
        stage_pre_field_call_count=0,
        combined_stage_particle_counts=(),
        exact_append_prefix_max_abs=0.0,
        stage_pre_replay_max_abs=0.0,
        transport_trace=transport_trace,
        transport_trace_sha256=_trace_sha256(transport_trace),
        edge_bridge_artifact_sha256=_edge_bridge_artifact_sha256(),
        sort_count=0,
        weld_count=0,
        delete_count=0,
        cancel_count=0,
        remesh_count=0,
        feedback_call_count=0,
        parent_write_count=0,
        load_write_count=0,
        observation_access="none",
        target_case_branch="none",
    )
    report = replace(placeholder, report_sha256=_report_digest(placeholder))
    _validate_report_semantics(report)
    with _LOCK:
        _register_live(_REPORT_REGISTRY, report, report.report_sha256)
    return report


def transport_accumulated_particle_cloud(
    handoff: object,
    *,
    smoothing_radius_m: object,
    deposition_target_spacing_m: object,
    transport_end_time_s: object,
    transport_substeps: object,
    freestream_velocity_gp1_m_per_s: ArrayLike,
    enabled: bool = True,
) -> CumulativeCloudTransportReport:
    """Append one release exactly, then advance the entire cloud once."""

    if not isinstance(enabled, (bool, np.bool_)):
        raise ValueError("enabled must be Boolean")
    if not bool(enabled):
        return _disabled_report()

    validated = _validate_handoff(handoff)
    consumed = _HANDOFF_CONSUMPTIONS.get(id(validated))
    if consumed is not None and consumed[0]() is validated:
        raise ValueError("cumulative handoff was already consumed (replay)")
    sigma_birth = _positive_real("smoothing_radius_m", smoothing_radius_m)
    target_spacing = _positive_real(
        "deposition_target_spacing_m", deposition_target_spacing_m
    )
    end_time = _finite_real("transport_end_time_s", transport_end_time_s)
    if end_time <= validated.source_time_s:
        raise ValueError("transport interval must be positive")
    substeps = _positive_integer("transport_substeps", transport_substeps)
    if substeps > MAX_TRANSPORT_SUBSTEPS:
        raise ValueError("transport_substeps exceeds the preregistered resource cap")
    freestream = _finite_vector(
        "freestream_velocity_gp1_m_per_s", freestream_velocity_gp1_m_per_s
    )

    ribbon = validated.current_ribbon
    graph = ribbon.edge_graph
    predicted_new_particle_count = _predict_release_particle_count(
        graph,
        smoothing_radius_m=sigma_birth,
        target_spacing_m=target_spacing,
    )
    parent_particle_count = 0
    if validated.previous_report is not None:
        parent_particle_count = _cloud_particle_count_gate(
            validated.previous_report.transported_particle_cloud,
            allow_empty=False,
        )
    if parent_particle_count + predicted_new_particle_count > MAX_CUMULATIVE_PARTICLES:
        raise ValueError("predicted cumulative particle count exceeds resource cap")

    old_positions = np.empty((0, 3), dtype=np.float64)
    old_gamma = np.empty((0, 3), dtype=np.float64)
    old_sigma = np.empty((0,), dtype=np.float64)
    old_ids: tuple[tuple[Any, ...], ...] = ()
    old_lineage: tuple[ParticleLineage, ...] = ()
    old_slices: tuple[ReleaseSliceLedger, ...] = ()
    parent_cloud_digest: str | None = None
    if validated.previous_report is not None:
        parent = validate_cumulative_cloud_transport_report(validated.previous_report)
        parent_cloud = parent.transported_particle_cloud
        old_positions, old_gamma, old_sigma = _cloud_arrays(
            parent_cloud, allow_empty=False
        )
        old_ids = parent_cloud.particle_ids
        old_lineage = parent_cloud.lineage
        old_slices = parent_cloud.release_slices
        parent_cloud_digest = parent_cloud.cloud_sha256
        for release in old_slices:
            if release.smoothing_radius_m != sigma_birth:
                raise ValueError("birth smoothing radius changed across releases")
            if release.deposition_target_spacing_m != target_spacing:
                raise ValueError("deposition spacing changed across releases")

    if len(old_ids) > MAX_CUMULATIVE_PARTICLES:
        raise ValueError("parent cloud exceeds cumulative particle resource cap")
    if len(old_ids) + predicted_new_particle_count > MAX_CUMULATIVE_PARTICLES:
        raise ValueError("predicted cumulative particle count exceeds resource cap")
    _, _, trusted_callables = _assert_transport_bindings()
    deposition_callable = trusted_callables["edge_deposition"]
    field_callable = trusted_callables["direct_field"]
    lsrk3_callable = trusted_callables["lsrk3"]
    state_callable = trusted_callables["particle_state"]
    trace_events: list[str] = []
    new_positions = np.empty((0, 3), dtype=np.float64)
    new_gamma = np.empty((0, 3), dtype=np.float64)
    new_sigma = np.empty((0,), dtype=np.float64)
    new_ids: tuple[tuple[Any, ...], ...] = ()
    new_lineage: tuple[ParticleLineage, ...] = ()
    deposited_digest: str | None = None
    release_slices = old_slices
    if graph is not None:
        trace_events.append(
            "deposit_prescribed_sigma_spacing:"
            f"step={validated.source_step_index}:"
            f"predicted={predicted_new_particle_count}"
        )
        deposited = deposition_callable(
            graph,
            smoothing_radius=sigma_birth,
            target_spacing=target_spacing,
            step=validated.source_step_index,
        )
        new_positions = deposited.positions
        new_gamma = deposited.gamma
        new_sigma = deposited.sigma
        new_ids = tuple(deposited.particle_ids)
        new_lineage = tuple(deposited.lineage)
        if not new_ids:
            raise ValueError("active release deposited no retained particles")
        if len(new_ids) != predicted_new_particle_count:
            raise RuntimeError("deposition particle count disagrees with resource gate")
        deposited_digest = _array_digest(
            new_positions,
            new_gamma,
            new_sigma,
            new_ids,
            new_lineage,
            domain=b"fluxv-v5h-cumulative-new-release-v2",
        )
        identity_digest = _identity_digest(new_ids, new_lineage)
        start_index = len(old_ids)
        release_slices = old_slices + (
            ReleaseSliceLedger(
                release_index=len(old_slices) + 1,
                source_step_index=validated.source_step_index,
                source_time_s=validated.source_time_s,
                start_index=start_index,
                stop_index=start_index + len(new_ids),
                particle_count=len(new_ids),
                parent_ribbon_digest_sha256=(validated.current_ribbon_digest_sha256),
                deposited_cloud_digest_sha256=deposited_digest,
                smoothing_radius_m=sigma_birth,
                deposition_target_spacing_m=target_spacing,
                particle_ids_sha256=identity_digest,
                lineage_sha256=identity_digest,
            ),
        )
    if not old_ids and not new_ids:
        raise ValueError("enabled cumulative transport requires a particle cloud")

    positions = np.ascontiguousarray(
        np.concatenate((old_positions, new_positions), axis=0), dtype=np.float64
    )
    gamma = np.ascontiguousarray(
        np.concatenate((old_gamma, new_gamma), axis=0), dtype=np.float64
    )
    sigma = np.ascontiguousarray(
        np.concatenate((old_sigma, new_sigma), axis=0), dtype=np.float64
    )
    particle_ids = old_ids + new_ids
    lineage = old_lineage + new_lineage
    old_count = len(old_ids)
    append_arrays_equal = (
        np.array_equal(positions[:old_count], old_positions)
        and np.array_equal(gamma[:old_count], old_gamma)
        and np.array_equal(sigma[:old_count], old_sigma)
        and np.array_equal(positions[old_count:], new_positions)
        and np.array_equal(gamma[old_count:], new_gamma)
        and np.array_equal(sigma[old_count:], new_sigma)
    )
    append_identity_equal = (
        particle_ids[:old_count] == old_ids
        and lineage[:old_count] == old_lineage
        and particle_ids[old_count:] == new_ids
        and lineage[old_count:] == new_lineage
    )
    exact_append_prefix_max_abs = _max_abs(
        positions[:old_count] - old_positions,
        gamma[:old_count] - old_gamma,
        sigma[:old_count] - old_sigma,
        positions[old_count:] - new_positions,
        gamma[old_count:] - new_gamma,
        sigma[old_count:] - new_sigma,
    )
    if not append_arrays_equal or not append_identity_equal:
        raise RuntimeError("old particle prefix changed during exact append")
    if len(set(particle_ids)) != len(particle_ids):
        raise ValueError("exact append produced duplicate particle IDs")
    appended_digest = _array_digest(
        positions,
        gamma,
        sigma,
        particle_ids,
        lineage,
        domain=b"fluxv-v5h-cumulative-appended-pre-transport-v2",
    )
    trace_events.append(
        "exact_append:"
        f"old={old_count}:new={len(new_ids)}:total={len(particle_ids)}:"
        f"max_abs={exact_append_prefix_max_abs.hex()}"
    )

    births = tuple(
        sorted(
            (birth for birth in ribbon.node_births if birth.active),
            key=lambda birth: _id_key(_stable_id("birth.node_id", birth.node_id)),
        )
    )
    seeds: list[FloatArray] = []
    for birth in births:
        if (
            birth.birth_position_gp1_m is None
            or birth.birth_node_id is None
            or birth.lineage_epoch is None
            or birth.birth_step_index != validated.source_step_index
        ):
            raise ValueError("active ribbon birth has incomplete frontier identity")
        seeds.append(_finite_vector("birth position", birth.birth_position_gp1_m))
    tracers = (
        np.ascontiguousarray(np.vstack(seeds), dtype=np.float64)
        if seeds
        else np.empty((0, 3), dtype=np.float64)
    )
    initial_state = state_callable(positions, gamma, sigma)
    initial_tracers = tracers.copy()
    state = state_callable(
        initial_state.positions,
        initial_state.gamma,
        initial_state.sigma,
    )
    delta_time = (end_time - validated.source_time_s) / substeps
    combined_stage_particle_counts_list: list[int] = []
    for substep_index in range(substeps):
        trace_events.append(
            f"combined_lsrk3_call:substep={substep_index + 1}:"
            f"particles={len(particle_ids)}"
        )
        next_state, stages = lsrk3_callable(
            state,
            delta_time,
            freestream_velocity=freestream,
        )
        if not isinstance(stages, tuple) or len(stages) != 3:
            raise RuntimeError("LSRK3 backend did not return its three frozen stages")
        tracer_storage = np.zeros_like(tracers)
        for expected_stage, stage in enumerate(stages, start=1):
            if stage.stage != expected_stage:
                raise RuntimeError("LSRK3 backend returned a reordered stage trace")
            stage_particle_count = int(stage.pre.positions.shape[0])
            if stage_particle_count != len(particle_ids):
                raise RuntimeError("LSRK3 stage did not use the full cumulative cloud")
            combined_stage_particle_counts_list.append(stage_particle_count)
            trace_events.append(
                f"combined_lsrk3_stage:substep={substep_index + 1}:"
                f"stage={expected_stage}:particles={stage_particle_count}"
            )
            field_value = field_callable(
                stage.pre.positions,
                stage.pre.gamma,
                stage.pre.sigma,
                target_positions=tracers,
            )
            trace_events.append(
                f"frontier_stage_pre_field:substep={substep_index + 1}:"
                f"stage={expected_stage}:particles={stage_particle_count}:"
                f"frontiers={len(tracers)}"
            )
            tracer_storage = stage.a * tracer_storage + delta_time * (
                field_value.velocity + freestream[None, :]
            )
            tracers = tracers + stage.b * tracer_storage
            if not np.all(np.isfinite(tracers)):
                raise FloatingPointError("frontier transport became non-finite")
        final_stage = stages[-1].post
        if not (
            np.array_equal(next_state.positions, final_stage.positions)
            and np.array_equal(next_state.gamma, final_stage.gamma)
            and np.array_equal(next_state.sigma, final_stage.sigma)
        ):
            raise RuntimeError("LSRK3 returned state disagrees with its final stage")
        state = next_state

    # Independently replay the same initial combined state and frontier seeds.
    # This is intentionally a second loop rather than reuse of the recorded
    # state/stages, so the report gate is tied to recomputed numerical output.
    replay_state = state_callable(
        initial_state.positions,
        initial_state.gamma,
        initial_state.sigma,
    )
    replay_tracers = initial_tracers.copy()
    for _ in range(substeps):
        replay_next, replay_stages = lsrk3_callable(
            replay_state,
            delta_time,
            freestream_velocity=freestream,
        )
        if not isinstance(replay_stages, tuple) or len(replay_stages) != 3:
            raise RuntimeError("independent replay did not return three LSRK3 stages")
        replay_storage = np.zeros_like(replay_tracers)
        for expected_stage, replay_stage in enumerate(replay_stages, start=1):
            if replay_stage.stage != expected_stage or replay_stage.pre.positions.shape[
                0
            ] != len(particle_ids):
                raise RuntimeError("independent replay stage ownership is invalid")
            replay_field = field_callable(
                replay_stage.pre.positions,
                replay_stage.pre.gamma,
                replay_stage.pre.sigma,
                target_positions=replay_tracers,
            )
            replay_storage = replay_stage.a * replay_storage + delta_time * (
                replay_field.velocity + freestream[None, :]
            )
            replay_tracers = replay_tracers + replay_stage.b * replay_storage
        replay_final_stage = replay_stages[-1].post
        if not (
            np.array_equal(replay_next.positions, replay_final_stage.positions)
            and np.array_equal(replay_next.gamma, replay_final_stage.gamma)
            and np.array_equal(replay_next.sigma, replay_final_stage.sigma)
        ):
            raise RuntimeError("independent replay state/final-stage mismatch")
        replay_state = replay_next
    stage_pre_replay_max_abs = _max_abs(
        state.positions - replay_state.positions,
        state.gamma - replay_state.gamma,
        state.sigma - replay_state.sigma,
        tracers - replay_tracers,
    )
    if not np.isfinite(stage_pre_replay_max_abs):
        raise FloatingPointError("independent stage-pre replay residual is non-finite")
    trace_events.append(
        "independent_stage_pre_replay:" f"max_abs={stage_pre_replay_max_abs.hex()}"
    )
    transport_trace = tuple(trace_events)
    deposition_call_count = sum(
        event.startswith("deposit_prescribed_sigma_spacing:")
        for event in transport_trace
    )
    lsrk3_call_count = sum(
        event.startswith("combined_lsrk3_call:") for event in transport_trace
    )
    lsrk3_stage_count = sum(
        event.startswith("combined_lsrk3_stage:") for event in transport_trace
    )
    stage_pre_field_call_count = sum(
        event.startswith("frontier_stage_pre_field:") for event in transport_trace
    )
    sort_count = sum(event.startswith("particle_sort:") for event in transport_trace)
    weld_count = sum(event.startswith("particle_weld:") for event in transport_trace)
    delete_count = sum(
        event.startswith("particle_delete:") for event in transport_trace
    )
    cancel_count = sum(
        event.startswith("particle_cancel:") for event in transport_trace
    )
    remesh_count = sum(
        event.startswith("particle_remesh:") for event in transport_trace
    )
    feedback_call_count = sum(
        event.startswith("feedback_write:") for event in transport_trace
    )
    parent_write_count = sum(
        event.startswith("parent_write:") for event in transport_trace
    )
    load_write_count = sum(event.startswith("load_write:") for event in transport_trace)
    combined_stage_particle_counts = tuple(combined_stage_particle_counts_list)
    exact_append_passed = (
        append_arrays_equal
        and append_identity_equal
        and exact_append_prefix_max_abs == 0.0
        and len(particle_ids) == old_count + len(new_ids)
        and len(new_ids) == predicted_new_particle_count
    )
    one_combined_field_passed = (
        lsrk3_call_count == substeps
        and lsrk3_stage_count == 3 * substeps
        and len(combined_stage_particle_counts) == 3 * substeps
        and all(count == len(particle_ids) for count in combined_stage_particle_counts)
        and not any("split" in event for event in transport_trace)
    )
    stage_pre_replay_passed = (
        stage_pre_field_call_count == 3 * substeps and stage_pre_replay_max_abs == 0.0
    )
    if not (
        exact_append_passed and one_combined_field_passed and stage_pre_replay_passed
    ):
        raise RuntimeError("cumulative transport mechanical replay gate failed")

    cloud_digest = _cloud_digest(
        state.positions,
        state.gamma,
        state.sigma,
        particle_ids,
        lineage,
        release_slices,
    )
    cloud = CumulativeParticleCloud(
        interface_id=CUMULATIVE_CLOUD_INTERFACE_ID,
        positions_gp1_m=tuple(
            tuple(float(value) for value in row) for row in state.positions
        ),
        gamma_vector_m3_per_s=tuple(
            tuple(float(value) for value in row) for row in state.gamma
        ),
        sigma_m=tuple(float(value) for value in state.sigma),
        particle_ids=particle_ids,
        lineage=lineage,
        release_slices=release_slices,
        cloud_sha256=cloud_digest,
    )
    producer_hash = _producer_artifact_sha256()
    transport_hash = _transport_artifact_sha256()
    fact_cores: list[NodeFrontierFact] = []
    for birth, position in zip(births, tracers, strict=True):
        node_id = _stable_id("birth.node_id", birth.node_id)
        parent_digest = parent_frontier_digest_sha256(
            wing_id=validated.wing_id,
            node_id=node_id,
            source_family=validated.source_family,
            lineage_epoch=int(birth.lineage_epoch),
            parent_frontier_id=str(birth.birth_node_id),
            parent_birth_step_index=validated.source_step_index,
            parent_position_gp1_m=birth.birth_position_gp1_m,
        )
        fact_cores.append(
            NodeFrontierFact(
                wing_id=validated.wing_id,
                node_id=node_id,
                source_family=validated.source_family,
                lineage_epoch=int(birth.lineage_epoch),
                parent_frontier_id=str(birth.birth_node_id),
                parent_frontier_digest_sha256=parent_digest,
                parent_ribbon_digest_sha256=(validated.current_ribbon_digest_sha256),
                parent_birth_step_index=validated.source_step_index,
                for_source_step_index=validated.source_step_index + 1,
                time_layer=CUMULATIVE_TIME_LAYER,
                transport_backend_id=TRANSPORT_BACKEND_ID,
                transport_artifact_sha256=transport_hash,
                producer_id=CUMULATIVE_PRODUCER_ID,
                producer_artifact_sha256=producer_hash,
                producer_report_sha256="0" * 64,
                transport_start_time_s=validated.source_time_s,
                transport_end_time_s=end_time,
                transport_substeps=substeps,
                advected_position_gp1_m=tuple(float(value) for value in position),
                fact_owner=CUMULATIVE_FACT_OWNER,
            )
        )

    placeholder = CumulativeCloudTransportReport(
        enabled=True,
        interface_id=CUMULATIVE_REPORT_INTERFACE_ID,
        producer_id=CUMULATIVE_PRODUCER_ID,
        producer_artifact_sha256=producer_hash,
        transport_backend_id=TRANSPORT_BACKEND_ID,
        transport_artifact_sha256=transport_hash,
        wing_id=validated.wing_id,
        source_family=validated.source_family,
        parent_source_step_index=validated.source_step_index,
        for_source_step_index=validated.source_step_index + 1,
        time_layer=CUMULATIVE_TIME_LAYER,
        transport_start_time_s=validated.source_time_s,
        transport_end_time_s=end_time,
        transport_substeps=substeps,
        freestream_velocity_gp1_m_per_s=tuple(float(value) for value in freestream),
        parent_ribbon_digest_sha256=validated.current_ribbon_digest_sha256,
        current_ribbon_digest_sha256=validated.current_ribbon_digest_sha256,
        handoff_sha256=validated.handoff_sha256,
        parent_report_sha256=validated.previous_report_sha256,
        parent_cloud_digest_before_append_sha256=parent_cloud_digest,
        deposited_new_release_digest_sha256=deposited_digest,
        appended_cloud_digest_before_transport_sha256=appended_digest,
        transported_cloud_digest_after_sha256=cloud_digest,
        smoothing_radius_m=sigma_birth,
        deposition_target_spacing_m=target_spacing,
        previous_particle_count=old_count,
        new_particle_count=len(new_ids),
        total_particle_count=len(particle_ids),
        transported_particle_cloud=cloud,
        facts=tuple(fact_cores),
        report_sha256="0" * 64,
        attestation_kind=CUMULATIVE_ATTESTATION_KIND,
        continuation_scope=CUMULATIVE_CONTINUATION_SCOPE,
        exact_append_passed=exact_append_passed,
        one_combined_field_passed=one_combined_field_passed,
        stage_pre_replay_passed=stage_pre_replay_passed,
        deposition_call_count=deposition_call_count,
        predicted_new_particle_count=predicted_new_particle_count,
        lsrk3_call_count=lsrk3_call_count,
        lsrk3_stage_count=lsrk3_stage_count,
        stage_pre_field_call_count=stage_pre_field_call_count,
        combined_stage_particle_counts=combined_stage_particle_counts,
        exact_append_prefix_max_abs=exact_append_prefix_max_abs,
        stage_pre_replay_max_abs=stage_pre_replay_max_abs,
        transport_trace=transport_trace,
        transport_trace_sha256=_trace_sha256(transport_trace),
        edge_bridge_artifact_sha256=_edge_bridge_artifact_sha256(),
        sort_count=sort_count,
        weld_count=weld_count,
        delete_count=delete_count,
        cancel_count=cancel_count,
        remesh_count=remesh_count,
        feedback_call_count=feedback_call_count,
        parent_write_count=parent_write_count,
        load_write_count=load_write_count,
        observation_access="none",
        target_case_branch="none",
    )
    report_digest = _report_digest(placeholder)
    facts = tuple(
        replace(fact, producer_report_sha256=report_digest) for fact in fact_cores
    )
    report = replace(placeholder, facts=facts, report_sha256=report_digest)
    _validate_report_semantics(report)

    # Commit exact-once only after all deposition, append, transport, digest and
    # report construction gates succeed.  Any exception above leaves the live
    # handoff retryable with identical inputs.
    with _LOCK:
        consumed = _HANDOFF_CONSUMPTIONS.get(id(validated))
        if consumed is not None and consumed[0]() is validated:
            raise ValueError("cumulative handoff was already consumed (replay)")
        _register_live(_REPORT_REGISTRY, report, report.report_sha256)
        _register_live(_HANDOFF_CONSUMPTIONS, validated, report.report_sha256)
    return validate_cumulative_cloud_transport_report(report)


def _validate_report_semantics(
    report: object,
) -> tuple[CumulativeCloudTransportReport, str]:
    """Recompute every report gate without consulting the live registry."""

    if not isinstance(report, CumulativeCloudTransportReport):
        raise ValueError("cumulative report has a foreign schema")
    if not isinstance(report.enabled, (bool, np.bool_)):
        raise ValueError("report.enabled must be Boolean")
    report_enabled = bool(report.enabled)
    previous_count = _nonnegative_integer(
        "report.previous_particle_count", report.previous_particle_count
    )
    new_count = _nonnegative_integer(
        "report.new_particle_count", report.new_particle_count
    )
    predicted_count = _nonnegative_integer(
        "report.predicted_new_particle_count",
        report.predicted_new_particle_count,
    )
    total_count = _nonnegative_integer(
        "report.total_particle_count", report.total_particle_count
    )
    if new_count > MAX_PARTICLES_PER_RELEASE or total_count > MAX_CUMULATIVE_PARTICLES:
        raise ValueError("cumulative report exceeds a particle resource cap")
    cloud_count = _cloud_particle_count_gate(
        report.transported_particle_cloud,
        allow_empty=not report_enabled,
    )
    if (
        previous_count + new_count != total_count
        or predicted_count != new_count
        or cloud_count != total_count
    ):
        raise ValueError("cumulative append particle counts are inconsistent")
    if (
        report.interface_id != CUMULATIVE_REPORT_INTERFACE_ID
        or report.producer_id != CUMULATIVE_PRODUCER_ID
        or report.attestation_kind != CUMULATIVE_ATTESTATION_KIND
        or report.continuation_scope != CUMULATIVE_CONTINUATION_SCOPE
        or report.time_layer != CUMULATIVE_TIME_LAYER
        or report.producer_artifact_sha256 != _producer_artifact_sha256()
        or report.transport_backend_id != TRANSPORT_BACKEND_ID
        or report.transport_artifact_sha256 != _transport_artifact_sha256()
        or report.edge_bridge_artifact_sha256 != _edge_bridge_artifact_sha256()
    ):
        raise ValueError("cumulative report producer/backend identity is stale")

    trace = report.transport_trace
    trace_digest = _trace_sha256(trace)
    if not hmac.compare_digest(
        trace_digest,
        _lower_sha256("report.transport_trace_sha256", report.transport_trace_sha256),
    ):
        raise ValueError("cumulative transport trace digest is inconsistent")
    trace_counts = {
        "deposition": sum(
            event.startswith("deposit_prescribed_sigma_spacing:") for event in trace
        ),
        "lsrk3_call": sum(event.startswith("combined_lsrk3_call:") for event in trace),
        "lsrk3_stage": sum(
            event.startswith("combined_lsrk3_stage:") for event in trace
        ),
        "stage_pre": sum(
            event.startswith("frontier_stage_pre_field:") for event in trace
        ),
        "sort": sum(event.startswith("particle_sort:") for event in trace),
        "weld": sum(event.startswith("particle_weld:") for event in trace),
        "delete": sum(event.startswith("particle_delete:") for event in trace),
        "cancel": sum(event.startswith("particle_cancel:") for event in trace),
        "remesh": sum(event.startswith("particle_remesh:") for event in trace),
        "feedback": sum(event.startswith("feedback_write:") for event in trace),
        "parent_write": sum(event.startswith("parent_write:") for event in trace),
        "load_write": sum(event.startswith("load_write:") for event in trace),
    }
    reported_trace_counts = {
        "deposition": _nonnegative_integer(
            "report.deposition_call_count", report.deposition_call_count
        ),
        "lsrk3_call": _nonnegative_integer(
            "report.lsrk3_call_count", report.lsrk3_call_count
        ),
        "lsrk3_stage": _nonnegative_integer(
            "report.lsrk3_stage_count", report.lsrk3_stage_count
        ),
        "stage_pre": _nonnegative_integer(
            "report.stage_pre_field_call_count", report.stage_pre_field_call_count
        ),
        "sort": _nonnegative_integer("report.sort_count", report.sort_count),
        "weld": _nonnegative_integer("report.weld_count", report.weld_count),
        "delete": _nonnegative_integer("report.delete_count", report.delete_count),
        "cancel": _nonnegative_integer("report.cancel_count", report.cancel_count),
        "remesh": _nonnegative_integer("report.remesh_count", report.remesh_count),
        "feedback": _nonnegative_integer(
            "report.feedback_call_count", report.feedback_call_count
        ),
        "parent_write": _nonnegative_integer(
            "report.parent_write_count", report.parent_write_count
        ),
        "load_write": _nonnegative_integer(
            "report.load_write_count", report.load_write_count
        ),
    }
    if trace_counts != reported_trace_counts:
        raise ValueError("reported transport call counts disagree with the trace")
    if any(
        reported_trace_counts[name]
        for name in (
            "sort",
            "weld",
            "delete",
            "cancel",
            "remesh",
            "feedback",
            "parent_write",
            "load_write",
        )
    ):
        raise ValueError("cumulative report violates one-way exact-append ownership")
    if report.observation_access != "none" or report.target_case_branch != "none":
        raise ValueError("cumulative report accessed target observations")

    positions, _, _ = _cloud_arrays(
        report.transported_particle_cloud,
        allow_empty=not report_enabled,
    )
    transported_digest = _lower_sha256(
        "report.transported_cloud_digest_after_sha256",
        report.transported_cloud_digest_after_sha256,
    )
    if transported_digest != report.transported_particle_cloud.cloud_sha256:
        raise ValueError("report/cloud transported digest disagrees")
    exact_residual = _finite_real(
        "report.exact_append_prefix_max_abs",
        report.exact_append_prefix_max_abs,
    )
    replay_residual = _finite_real(
        "report.stage_pre_replay_max_abs",
        report.stage_pre_replay_max_abs,
    )
    if exact_residual < 0.0 or replay_residual < 0.0:
        raise ValueError("transport replay residual must be nonnegative")

    if total_count != positions.shape[0]:
        raise ValueError("cumulative append particle counts are inconsistent")
    stage_particle_counts = tuple(
        _nonnegative_integer("combined stage particle count", value)
        for value in report.combined_stage_particle_counts
    )

    if report.enabled:
        if report.wing_id is None or report.source_family not in (
            "lev",
            "tev_persisted",
        ):
            raise ValueError("enabled cumulative report has invalid ownership")
        wing = _stable_id("report.wing_id", report.wing_id)
        parent_step = _positive_integer(
            "report.parent_source_step_index", report.parent_source_step_index
        )
        if report.for_source_step_index != parent_step + 1:
            raise ValueError("cumulative report source steps are nonconsecutive")
        start = _finite_real(
            "report.transport_start_time_s", report.transport_start_time_s
        )
        end = _finite_real("report.transport_end_time_s", report.transport_end_time_s)
        if end <= start:
            raise ValueError("enabled cumulative report has invalid time interval")
        substeps = _positive_integer(
            "report.transport_substeps", report.transport_substeps
        )
        if substeps > MAX_TRANSPORT_SUBSTEPS:
            raise ValueError("report substeps exceed the preregistered resource cap")
        smoothing_radius = _positive_real(
            "report.smoothing_radius_m", report.smoothing_radius_m
        )
        target_spacing = _positive_real(
            "report.deposition_target_spacing_m",
            report.deposition_target_spacing_m,
        )
        if smoothing_radius / target_spacing < FROZEN_OVERLAP_LAMBDA:
            raise ValueError("report target spacing violates minimum overlap")
        _finite_vector(
            "report.freestream_velocity_gp1_m_per_s",
            report.freestream_velocity_gp1_m_per_s,
        )
        _lower_sha256("report.handoff_sha256", report.handoff_sha256)
        parent_ribbon_digest = _lower_sha256(
            "report.parent_ribbon_digest_sha256",
            report.parent_ribbon_digest_sha256,
        )
        if report.current_ribbon_digest_sha256 != parent_ribbon_digest:
            raise ValueError("cumulative current and parent ribbon digests disagree")
        _lower_sha256(
            "report.appended_cloud_digest_before_transport_sha256",
            report.appended_cloud_digest_before_transport_sha256,
        )
        if previous_count:
            _lower_sha256("report.parent_report_sha256", report.parent_report_sha256)
            _lower_sha256(
                "report.parent_cloud_digest_before_append_sha256",
                report.parent_cloud_digest_before_append_sha256,
            )
        elif (
            report.parent_report_sha256 is not None
            or report.parent_cloud_digest_before_append_sha256 is not None
        ):
            raise ValueError("first cumulative report has a phantom parent cloud")
        if new_count:
            _lower_sha256(
                "report.deposited_new_release_digest_sha256",
                report.deposited_new_release_digest_sha256,
            )
        elif report.deposited_new_release_digest_sha256 is not None:
            raise ValueError("inactive cumulative step has a phantom deposition")

        expected_deposition_calls = 1 if new_count else 0
        exact_event = (
            "exact_append:"
            f"old={previous_count}:new={new_count}:total={total_count}:"
            f"max_abs={exact_residual.hex()}"
        )
        replay_event = (
            "independent_stage_pre_replay:" f"max_abs={replay_residual.hex()}"
        )
        exact_gate = (
            exact_residual == 0.0
            and predicted_count == new_count
            and trace_counts["deposition"] == expected_deposition_calls
            and trace.count(exact_event) == 1
        )
        combined_gate = (
            trace_counts["lsrk3_call"] == substeps
            and trace_counts["lsrk3_stage"] == 3 * substeps
            and len(stage_particle_counts) == 3 * substeps
            and all(value == total_count for value in stage_particle_counts)
            and not any("split" in event for event in trace)
        )
        replay_gate = (
            trace_counts["stage_pre"] == 3 * substeps
            and replay_residual == 0.0
            and trace.count(replay_event) == 1
        )
        for name, actual, expected in (
            ("exact_append_passed", report.exact_append_passed, exact_gate),
            (
                "one_combined_field_passed",
                report.one_combined_field_passed,
                combined_gate,
            ),
            (
                "stage_pre_replay_passed",
                report.stage_pre_replay_passed,
                replay_gate,
            ),
        ):
            if not isinstance(actual, (bool, np.bool_)) or bool(actual) is not expected:
                raise ValueError(f"{name} disagrees with recomputed transport evidence")
        if not (exact_gate and combined_gate and replay_gate):
            raise ValueError("cumulative report mechanical gate is false")

        slices = report.transported_particle_cloud.release_slices
        prior_source_step = 0
        prior_source_time = -float("inf")
        for release in slices:
            release_step = _positive_integer(
                "release.source_step_index", release.source_step_index
            )
            release_time = _finite_real("release.source_time_s", release.source_time_s)
            if release_step <= prior_source_step or release_time <= prior_source_time:
                raise ValueError("release slice time/step order is inconsistent")
            if release_step > parent_step or release_time > start:
                raise ValueError("release slice is from a future time layer")
            if (
                release.smoothing_radius_m != smoothing_radius
                or release.deposition_target_spacing_m != target_spacing
            ):
                raise ValueError("release birth core/spacing changed across the cloud")
            prior_source_step = release_step
            prior_source_time = release_time
        if new_count:
            latest = slices[-1]
            if (
                latest.source_step_index != parent_step
                or latest.source_time_s != start
                or latest.particle_count != new_count
            ):
                raise ValueError("new release slice time/count is inconsistent")

        fact_keys: set[tuple[str, int | str]] = set()
        for fact in report.facts:
            if not isinstance(fact, NodeFrontierFact):
                raise ValueError("cumulative report contains a foreign frontier fact")
            key = _id_key(_stable_id("fact.node_id", fact.node_id))
            if key in fact_keys:
                raise ValueError("cumulative frontier facts contain duplicate nodes")
            fact_keys.add(key)
            lineage_epoch = _nonnegative_integer(
                "fact.lineage_epoch", fact.lineage_epoch
            )
            _ = lineage_epoch
            if (
                not isinstance(fact.parent_frontier_id, str)
                or not fact.parent_frontier_id
            ):
                raise ValueError("frontier fact parent identity must be explicit")
            _lower_sha256(
                "fact.parent_frontier_digest_sha256",
                fact.parent_frontier_digest_sha256,
            )
            _finite_vector("fact.advected_position_gp1_m", fact.advected_position_gp1_m)
            fact_start = _finite_real(
                "fact.transport_start_time_s", fact.transport_start_time_s
            )
            fact_end = _finite_real(
                "fact.transport_end_time_s", fact.transport_end_time_s
            )
            fact_substeps = _positive_integer(
                "fact.transport_substeps", fact.transport_substeps
            )
            if (
                fact.wing_id != wing
                or fact.source_family != report.source_family
                or fact.parent_ribbon_digest_sha256 != parent_ribbon_digest
                or fact.parent_birth_step_index != parent_step
                or fact.for_source_step_index != report.for_source_step_index
                or fact.time_layer != report.time_layer
                or fact.transport_backend_id != report.transport_backend_id
                or fact.transport_artifact_sha256 != report.transport_artifact_sha256
                or fact.producer_id != report.producer_id
                or fact.producer_artifact_sha256 != report.producer_artifact_sha256
                or fact.producer_report_sha256 != report.report_sha256
                or fact_start != start
                or fact_end != end
                or fact_substeps != substeps
                or fact.fact_owner != CUMULATIVE_FACT_OWNER
            ):
                raise ValueError("cumulative frontier fact time/identity disagrees")
    else:
        if (
            positions.shape[0]
            or report.facts
            or total_count
            or previous_count
            or new_count
            or predicted_count
            or trace != ("disabled:input-blind",)
            or any(trace_counts.values())
            or report.transport_substeps != 0
            or report.transport_start_time_s is not None
            or report.transport_end_time_s is not None
            or report.wing_id is not None
            or report.source_family is not None
            or report.parent_source_step_index is not None
            or report.for_source_step_index is not None
            or report.smoothing_radius_m is not None
            or report.deposition_target_spacing_m is not None
            or stage_particle_counts
            or report.exact_append_passed
            or report.one_combined_field_passed
            or report.stage_pre_replay_passed
        ):
            raise ValueError(
                "disabled cumulative report is not an input-blind empty state"
            )

    digest = _report_digest(report)
    if not hmac.compare_digest(
        digest, _lower_sha256("report_sha256", report.report_sha256)
    ):
        raise ValueError("cumulative report digest is inconsistent")
    return report, digest


def validate_cumulative_cloud_transport_report(
    report: object,
) -> CumulativeCloudTransportReport:
    """Validate a direct live cumulative report and all immutable ledgers."""

    validated, digest = _validate_report_semantics(report)
    registered = _REPORT_REGISTRY.get(id(validated))
    if (
        registered is None
        or registered[0]() is not validated
        or not hmac.compare_digest(registered[1], digest)
    ):
        raise ValueError("cumulative report is not a directly produced live object")
    return validated


def materialize_cumulative_particle_state(report: object) -> ParticleState:
    """Return a fresh mutable-array copy of an attested cumulative cloud."""

    validated = validate_cumulative_cloud_transport_report(report)
    positions, gamma, sigma = _cloud_arrays(
        validated.transported_particle_cloud,
        allow_empty=not validated.enabled,
    )
    if not validated.enabled:
        return ParticleState(
            positions=positions.copy(),
            gamma=gamma.copy(),
            sigma=sigma.copy(),
        )
    return make_particle_state(positions, gamma, sigma)


__all__ = [
    "CUMULATIVE_ATTESTATION_KIND",
    "CUMULATIVE_CLOUD_INTERFACE_ID",
    "CUMULATIVE_CONTINUATION_SCOPE",
    "CUMULATIVE_FACT_OWNER",
    "CUMULATIVE_HANDOFF_INTERFACE_ID",
    "CUMULATIVE_PRODUCER_ID",
    "CUMULATIVE_REPORT_INTERFACE_ID",
    "CUMULATIVE_TIME_LAYER",
    "CumulativeCloudTransportReport",
    "CumulativeParticleCloud",
    "CumulativeRibbonHandoff",
    "ReleaseSliceLedger",
    "MAX_CUMULATIVE_PARTICLES",
    "MAX_PARTICLE_COUNT",
    "MAX_PARTICLES_PER_RELEASE",
    "MAX_TRANSPORT_SUBSTEPS",
    "TRANSPORT_BACKEND_ID",
    "attest_cumulative_ribbon_handoff",
    "materialize_cumulative_particle_state",
    "transport_accumulated_particle_cloud",
    "validate_cumulative_cloud_transport_report",
]
