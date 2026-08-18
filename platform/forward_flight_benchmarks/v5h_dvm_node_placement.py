"""Fail-closed node-local DVM placement adapter for FluxV v5h.

The sectional DVM used for circulation remains cell-centred.  This module
consumes a *different*, directly attested :class:`V5hDVMSource` event at every
shared span node and exports geometry only.  First and restarted LEV births
use the source's relative half-step displacement; continuous births explicitly
delegate topology to the already-attested rVPM frontier consumed by
``v5h_dvm_node_ribbon``.

No circulation, aerodynamic force, or surface-load value is exported.  Ptera
remains the unique surface-load owner and the cell-centred DVM remains the
unique sectional-strength source.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import hmac
import json
from numbers import Integral, Real
import re
import threading
from typing import Any, Literal, Sequence, TypeAlias
import weakref

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .v5h_dvm_node_ribbon import SpanNodeKinematics
from .v5h_dvm_source import DVMSourceEvent, validate_dvm_source_event


FloatArray = NDArray[np.float64]
StableId: TypeAlias = int | str
NodePlacementMode = Literal["inactive", "first", "continuous", "restart"]
TopologyOwner = Literal[
    "none",
    "node_local_dvm_relative_birth",
    "attested_rvpm_frontier",
]

INTERFACE_ID = "fluxv-v5h-node-local-dvm-placement-gp1-v1"
STATUS = "evaluated_geometry_only_noncanonical"
DISABLED_STATUS = "not_evaluated_disabled"
NODE_GEOMETRY_ROLE = "node-local-dvm-geometry-only"
EXCLUSIVE_STRENGTH_OWNER = "cell-center-dvm"
EXCLUSIVE_SURFACE_OWNER = "ptera"
POSITION_UNITS = "m"
VELOCITY_UNITS = "m/s"

_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True, slots=True)
class GP1NodeSectionFact:
    """One caller-owned GP1 section fact backed by one node-local DVM event.

    ``node_lineage_id`` and ``geometry_token`` are not free-form labels: they
    must exactly equal the corresponding identities in ``event``.  The
    explicit units and owner roles prevent an otherwise valid geometry record
    from being silently repurposed as a strength or surface-load source.
    """

    wing_id: str
    node_id: StableId
    source_step_index: int
    source_time_s: float
    event: DVMSourceEvent
    lev_edge_anchor_gp1_m: ArrayLike
    tev_edge_anchor_gp1_m: ArrayLike
    reference_chord_m: float
    reference_speed_m_per_s: float
    dvm_x_axis_gp1: ArrayLike
    dvm_z_axis_gp1: ArrayLike
    positive_span_axis_gp1: ArrayLike
    topology_patch_id: str
    coordinate_frame_id: str
    node_lineage_id: str
    geometry_token: str
    position_units: str = POSITION_UNITS
    velocity_units: str = VELOCITY_UNITS
    node_geometry_role: str = NODE_GEOMETRY_ROLE
    exclusive_strength_owner: str = EXCLUSIVE_STRENGTH_OWNER
    exclusive_surface_owner: str = EXCLUSIVE_SURFACE_OWNER


@dataclass(frozen=True, slots=True)
class DVMNodePlacementCell:
    """One oriented cell-centre source with its two shared node facts.

    The cell-centre event is inspected only for live identity, causality, and
    LEV activity/mode coverage.  Its sectional strength is never copied into
    this adapter's result.
    """

    cell_id: StableId
    left_node_fact: GP1NodeSectionFact
    right_node_fact: GP1NodeSectionFact
    cell_source_event: DVMSourceEvent


@dataclass(frozen=True, slots=True)
class DVMNodePlacementLedger:
    """Immutable geometry-only audit record for one unique span node."""

    wing_id: str
    node_id: StableId
    source_step_index: int
    source_time_s: float
    active: bool
    placement_mode: NodePlacementMode
    topology_owner: TopologyOwner
    edge_velocity_used_by_ribbon: bool
    dvm_absolute_birth_used_for_topology: bool
    node_fact_manifest_sha256: str
    source_event_manifest_sha256: str
    source_parent_event_manifest_sha256: str
    source_newborn_id: str | None
    continuous_parent_source_id: str | None
    topology_patch_id: str
    coordinate_frame_id: str
    node_lineage_id: str
    geometry_token: str
    lev_edge_anchor_gp1_m: tuple[float, float, float]
    tev_edge_anchor_gp1_m: tuple[float, float, float]
    edge_velocity_gp1_m_per_s: tuple[float, float, float]
    mapped_relative_birth_gp1_m: tuple[float, float, float] | None
    half_step_reconstruction_residual_m: float


@dataclass(frozen=True, slots=True)
class DVMCellEndpointCoverage:
    """Proof that a cell-centre activity decision has two node facts."""

    wing_id: str
    cell_id: StableId
    source_step_index: int
    active: bool
    placement_mode: NodePlacementMode
    cell_source_event_manifest_sha256: str
    left_node_id: StableId
    right_node_id: StableId
    left_node_fact_manifest_sha256: str
    right_node_fact_manifest_sha256: str
    topology_patch_id: str
    coordinate_frame_id: str
    endpoint_coverage_complete: bool
    shared_fact_identity_verified: bool


@dataclass(frozen=True, slots=True, weakref_slot=True)
class DVMNodePlacementResult:
    """One live-attested geometry handoff compatible with the ribbon mapper."""

    wing_id: str
    enabled: bool
    status: str
    interface_id: str
    canonical: bool
    feedback_call_count: int
    source_step_index: int | None
    source_time_s: float | None
    delta_time_s: float | None
    node_geometry_role: str
    exclusive_strength_owner: str
    exclusive_surface_owner: str
    kinematics: tuple[SpanNodeKinematics, ...]
    node_ledgers: tuple[DVMNodePlacementLedger, ...]
    cell_coverage: tuple[DVMCellEndpointCoverage, ...]
    feedback_velocity: None
    producer_manifest_sha256: str

    def manifest(self) -> dict[str, Any]:
        """Return a deterministic JSON-compatible evidence manifest."""

        return _result_payload(self, include_attestation=True)


@dataclass(frozen=True, slots=True)
class _ValidatedNode:
    fact: GP1NodeSectionFact
    node_id: StableId
    source_step_index: int
    source_time_s: float
    event: DVMSourceEvent
    event_digest: str
    fact_digest: str
    lev_anchor: FloatArray
    tev_anchor: FloatArray
    chord: float
    speed: float
    x_axis: FloatArray
    z_axis: FloatArray
    span_axis: FloatArray
    patch_id: str
    frame_id: str
    lineage_id: str
    geometry_token: str
    active: bool
    mode: NodePlacementMode
    velocity: FloatArray
    mapped_birth: FloatArray | None
    reconstruction_residual: float
    topology_owner: TopologyOwner
    velocity_used: bool


@dataclass(frozen=True, slots=True)
class _NodeBinding:
    physical_section_id: str
    physical_strip_id: str
    section_lineage_id: str
    topology_patch_id: str
    coordinate_frame_id: str
    geometry_token: str
    reference_chord_m: float
    reference_speed_m_per_s: float


@dataclass(frozen=True, slots=True)
class _CellBinding:
    left_node_id: StableId
    right_node_id: StableId
    physical_section_id: str
    physical_strip_id: str
    section_lineage_id: str
    topology_patch_id: str
    coordinate_frame_id: str


@dataclass(frozen=True, slots=True)
class _ActivityState:
    ever_active: bool
    active_last_step: bool


_DIRECT_RESULT_REGISTRY: dict[
    int, tuple[weakref.ReferenceType[DVMNodePlacementResult], str]
] = {}
_LIVE_EVENT_CONSUMPTION_REGISTRY: dict[
    int,
    tuple[
        weakref.ReferenceType[DVMSourceEvent],
        str,
        str,
        StableId,
        int,
    ],
] = {}
_LIVE_EVENT_CONSUMPTION_LOCK = threading.RLock()


def _assert_events_unconsumed_locked(
    consumptions: Sequence[tuple[DVMSourceEvent, str, StableId, int]],
    *,
    wing_id: str,
) -> None:
    """Fail if any live source event already belongs to a placement result."""

    for event, _role, _owner_id, _source_step in consumptions:
        event_id = id(event)
        existing = _LIVE_EVENT_CONSUMPTION_REGISTRY.get(event_id)
        if existing is None:
            continue
        live_event = existing[0]()
        if live_event is None:
            _LIVE_EVENT_CONSUMPTION_REGISTRY.pop(event_id, None)
            continue
        if live_event is not event:
            raise RuntimeError("live-event identity registry collision")
        if existing[1] != wing_id:
            raise ValueError("a live DVM event was already consumed by another wing")
        raise ValueError("a live DVM event was already consumed by a placement result")


def _register_event_consumptions_locked(
    consumptions: Sequence[tuple[DVMSourceEvent, str, StableId, int]],
    *,
    wing_id: str,
) -> None:
    """Commit exact-once live event ownership after all other gates pass."""

    for event, role, owner_id, source_step in consumptions:
        event_id = id(event)

        def discard(
            reference: weakref.ReferenceType[DVMSourceEvent],
            *,
            consumed_event_id: int = event_id,
        ) -> None:
            with _LIVE_EVENT_CONSUMPTION_LOCK:
                existing = _LIVE_EVENT_CONSUMPTION_REGISTRY.get(consumed_event_id)
                if existing is not None and existing[0] is reference:
                    _LIVE_EVENT_CONSUMPTION_REGISTRY.pop(consumed_event_id, None)

        reference = weakref.ref(event, discard)
        _LIVE_EVENT_CONSUMPTION_REGISTRY[event_id] = (
            reference,
            wing_id,
            role,
            owner_id,
            source_step,
        )


def _stable_id(name: str, value: object) -> StableId:
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise ValueError(f"{name} must be a non-Boolean integer or explicit string")
    if isinstance(value, str):
        if not value.strip() or value != value.strip():
            raise ValueError(f"{name} must be an explicit canonical string")
    return value


def _id_key(value: StableId) -> tuple[int, str]:
    return (0 if isinstance(value, int) else 1, str(value))


def _explicit_text(name: str, value: object) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{name} must be an explicit canonical string")
    return value


def _strict_boolean(name: str, value: object) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be Boolean")
    return bool(value)


def _strict_integer(name: str, value: object, *, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _finite_real(name: str, value: object) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real scalar")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be a finite real scalar")
    return result


def _positive_real(name: str, value: object) -> float:
    result = _finite_real(name, value)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _finite_vector(name: str, value: object) -> FloatArray:
    raw = np.asarray(value, dtype=object)
    if raw.shape != (3,):
        raise ValueError(f"{name} must have shape (3,)")
    if any(
        isinstance(item, (bool, np.bool_)) or not isinstance(item, Real)
        for item in raw.flat
    ):
        raise ValueError(f"{name} must contain finite real values")
    result = np.asarray(raw, dtype=np.float64)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain finite real values")
    return result


def _unit_vector(name: str, value: object) -> FloatArray:
    result = _finite_vector(name, value)
    norm = float(np.linalg.norm(result))
    if not np.isfinite(norm) or abs(norm - 1.0) > 1.0e-12:
        raise ValueError(f"{name} must be unit length to 1e-12")
    return result


def _close(left: float, right: float, *, factor: float = 256.0) -> bool:
    scale = max(1.0, abs(left), abs(right))
    return abs(left - right) <= factor * np.finfo(float).eps * scale


def _vector_residual(left: FloatArray, right: FloatArray) -> float:
    return float(np.linalg.norm(left - right))


def _vector_tolerance(*vectors: FloatArray) -> float:
    scale = max(1.0, *(float(np.linalg.norm(value)) for value in vectors))
    return 512.0 * np.finfo(float).eps * scale


def _physical_node_key(node: _ValidatedNode) -> tuple[object, ...]:
    """Return an exact, patch-scoped GP1 section identity.

    Python float tuple equality is exact for finite values and intentionally
    treats signed zero as the same coordinate.  Therefore a one-ULP neighbour
    remains a different physical node; no distance tolerance, averaging, or
    coordinate welding is performed.
    """

    return (
        node.patch_id,
        node.frame_id,
        *(float(item) for item in node.lev_anchor),
        *(float(item) for item in node.tev_anchor),
    )


def _physical_dt_from_event(event: DVMSourceEvent) -> float:
    if event.delta_time_convective is None:
        raise ValueError("enabled DVM event has no convective time step")
    dt_star = _positive_real("event.delta_time_convective", event.delta_time_convective)
    chord = _positive_real(
        "event position scale", event.provenance.position_scale_chord_m
    )
    strength_scale = _positive_real(
        "event U*c scale", event.provenance.circulation_scale_u_times_c_m2_per_s
    )
    speed = strength_scale / chord
    return _positive_real("event physical time step", dt_star * chord / speed)


def _event_mode(event: DVMSourceEvent) -> NodePlacementMode:
    if not event.lesp_active:
        if event.lev_birth_mode != "none":
            raise ValueError("inactive node event carries an active LEV mode")
        return "inactive"
    mode = event.lev_placement.placement_mode
    if mode not in {"first", "continuous", "restart"}:
        raise ValueError("active node event has no supported placement mode")
    if event.lev_birth_mode != mode:
        raise ValueError("event and placement LEV modes disagree")
    return mode


def _expected_mode(state: _ActivityState, active: bool) -> NodePlacementMode:
    if not active:
        return "inactive"
    if not state.ever_active:
        return "first"
    if state.active_last_step:
        return "continuous"
    return "restart"


def _node_fact_payload(
    fact: GP1NodeSectionFact,
    *,
    wing_id: str,
    node_id: StableId,
    source_step_index: int,
    source_time_s: float,
    event_digest: str,
    lev_anchor: FloatArray,
    tev_anchor: FloatArray,
    chord: float,
    speed: float,
    x_axis: FloatArray,
    z_axis: FloatArray,
    span_axis: FloatArray,
    patch_id: str,
    frame_id: str,
    lineage_id: str,
    geometry_token: str,
) -> dict[str, Any]:
    del fact
    return {
        "schema_id": "fluxv-v5h-gp1-node-section-fact-v1",
        "wing_id": wing_id,
        "node_id": node_id,
        "source_step_index": source_step_index,
        "source_time_s": source_time_s,
        "source_event_manifest_sha256": event_digest,
        "lev_edge_anchor_gp1_m": tuple(float(item) for item in lev_anchor),
        "tev_edge_anchor_gp1_m": tuple(float(item) for item in tev_anchor),
        "reference_chord_m": chord,
        "reference_speed_m_per_s": speed,
        "dvm_x_axis_gp1": tuple(float(item) for item in x_axis),
        "dvm_z_axis_gp1": tuple(float(item) for item in z_axis),
        "positive_span_axis_gp1": tuple(float(item) for item in span_axis),
        "topology_patch_id": patch_id,
        "coordinate_frame_id": frame_id,
        "node_lineage_id": lineage_id,
        "geometry_token": geometry_token,
        "position_units": POSITION_UNITS,
        "velocity_units": VELOCITY_UNITS,
        "node_geometry_role": NODE_GEOMETRY_ROLE,
        "exclusive_strength_owner": EXCLUSIVE_STRENGTH_OWNER,
        "exclusive_surface_owner": EXCLUSIVE_SURFACE_OWNER,
    }


def _digest_payload(domain: bytes, payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return sha256(domain + b"\0" + encoded).hexdigest()


def _validate_node_fact(
    fact: object,
    *,
    delta_time_s: float,
    expected_wing_id: str,
) -> _ValidatedNode:
    if type(fact) is not GP1NodeSectionFact:
        raise ValueError("cell endpoints must be exactly GP1NodeSectionFact")
    wing_id = _explicit_text("node fact wing_id", fact.wing_id)
    if wing_id != expected_wing_id:
        raise ValueError("node fact is bound to another wing")
    node_id = _stable_id("node_id", fact.node_id)
    source_step = _strict_integer(
        "node source_step_index", fact.source_step_index, minimum=1
    )
    source_time = _finite_real("node source_time_s", fact.source_time_s)
    if source_time < 0.0:
        raise ValueError("node source_time_s cannot be negative")
    event = validate_dvm_source_event(fact.event)
    if not event.enabled:
        raise ValueError("an enabled placement handoff requires enabled node events")
    if event.provenance.canonical:
        raise ValueError("node DVM source unexpectedly claims canonical status")
    if (
        event.provenance.observation_access != "none"
        or event.provenance.target_case_branch != "none"
    ):
        raise ValueError("node DVM source is not isolated from target observations")
    if event.lineage.source_step_index != source_step:
        raise ValueError("node fact source step disagrees with its DVM event")
    event_digest = event.producer_manifest_sha256
    if _SHA256_PATTERN.fullmatch(event_digest) is None:
        raise ValueError("node event attestation is not a lowercase SHA-256")

    chord = _positive_real("reference_chord_m", fact.reference_chord_m)
    speed = _positive_real("reference_speed_m_per_s", fact.reference_speed_m_per_s)
    if not _close(chord, event.provenance.position_scale_chord_m):
        raise ValueError("node fact chord disagrees with source dimensionalization")
    if not _close(
        speed * chord,
        event.provenance.circulation_scale_u_times_c_m2_per_s,
    ):
        raise ValueError(
            "node fact speed/chord disagree with source dimensionalization"
        )
    expected_dt = _positive_real(
        "node event physical time step",
        float(event.delta_time_convective) * chord / speed,
    )
    if not _close(expected_dt, delta_time_s, factor=1024.0):
        raise ValueError("delta_time_s != event_dt*c/U for a node-local source")

    if fact.position_units != POSITION_UNITS or fact.velocity_units != VELOCITY_UNITS:
        raise ValueError("node fact GP1 units are not pinned to m and m/s")
    if fact.node_geometry_role != NODE_GEOMETRY_ROLE:
        raise ValueError("node fact is not explicitly geometry-only")
    if fact.exclusive_strength_owner != EXCLUSIVE_STRENGTH_OWNER:
        raise ValueError("node fact does not preserve the cell-centre strength owner")
    if fact.exclusive_surface_owner != EXCLUSIVE_SURFACE_OWNER:
        raise ValueError("node fact does not preserve Ptera surface ownership")

    lev_anchor = _finite_vector("lev_edge_anchor_gp1_m", fact.lev_edge_anchor_gp1_m)
    tev_anchor = _finite_vector("tev_edge_anchor_gp1_m", fact.tev_edge_anchor_gp1_m)
    x_axis = _unit_vector("dvm_x_axis_gp1", fact.dvm_x_axis_gp1)
    z_axis = _unit_vector("dvm_z_axis_gp1", fact.dvm_z_axis_gp1)
    span_axis = _unit_vector("positive_span_axis_gp1", fact.positive_span_axis_gp1)
    if abs(float(np.dot(x_axis, z_axis))) > 1.0e-12:
        raise ValueError("DVM x/z basis vectors must be orthogonal")
    if _vector_residual(np.cross(x_axis, z_axis), span_axis) > 1.0e-12:
        raise ValueError("DVM basis must be right-handed: x cross z = positive span")

    patch_id = _explicit_text("topology_patch_id", fact.topology_patch_id)
    frame_id = _explicit_text("coordinate_frame_id", fact.coordinate_frame_id)
    lineage_id = _explicit_text("node_lineage_id", fact.node_lineage_id)
    geometry_token = _explicit_text("geometry_token", fact.geometry_token)
    if lineage_id != event.lineage.section_lineage_id:
        raise ValueError("node lineage token disagrees with its direct DVM source")
    if geometry_token != event.provenance.geometry_hash_sha256:
        raise ValueError("node geometry token disagrees with source provenance")

    lev_edge_2d = event.lev_placement.edge_anchor_position_over_chord_backend_world
    tev_edge_2d = event.tev_placement.edge_anchor_position_over_chord_backend_world
    if lev_edge_2d is None or tev_edge_2d is None:
        raise ValueError("enabled node event has no LE/TE edge anchors")
    edge_delta_2d = np.asarray(tev_edge_2d, dtype=float) - np.asarray(
        lev_edge_2d, dtype=float
    )
    reconstructed_tev = lev_anchor + chord * (
        edge_delta_2d[0] * x_axis + edge_delta_2d[1] * z_axis
    )
    if _vector_residual(reconstructed_tev, tev_anchor) > _vector_tolerance(
        reconstructed_tev, tev_anchor
    ):
        raise ValueError("LE/TE GP1 anchors disagree with the DVM frame and chord")

    active = _strict_boolean("node event.lesp_active", event.lesp_active)
    mode = _event_mode(event)
    placement = event.lev_placement
    mapped_birth: FloatArray | None = None
    residual = 0.0
    if mode in {"first", "restart"}:
        if not placement.used_for_topology_eligible:
            raise ValueError(
                "first/restart relative placement is not topology eligible"
            )
        displacement = placement.birth_displacement_from_edge_over_chord_backend_world
        q_birth = placement.q_birth_over_u_backend_world
        if displacement is None or q_birth is None:
            raise ValueError("first/restart placement lacks displacement or q_birth")
        displacement_2d = np.asarray(displacement, dtype=float)
        q_2d = np.asarray(q_birth, dtype=float)
        if displacement_2d.shape != (2,) or q_2d.shape != (2,):
            raise ValueError("node placement vectors must have two components")
        if not np.all(np.isfinite(displacement_2d)) or not np.all(np.isfinite(q_2d)):
            raise ValueError("node placement vectors must be finite")
        mapped_birth = lev_anchor + chord * (
            displacement_2d[0] * x_axis + displacement_2d[1] * z_axis
        )
        velocity = speed * (q_2d[0] * x_axis + q_2d[1] * z_axis)
        half_step_birth = lev_anchor + 0.5 * velocity * delta_time_s
        residual = _vector_residual(mapped_birth, half_step_birth)
        if residual > _vector_tolerance(mapped_birth, half_step_birth):
            raise ValueError(
                "mapped DVM birth fails the independent GP1 half-step check"
            )
        topology_owner: TopologyOwner = "node_local_dvm_relative_birth"
        velocity_used = True
    elif mode == "continuous":
        if placement.used_for_topology_eligible:
            raise ValueError("continuous DVM absolute placement cannot own topology")
        parent_id = placement.continuous_parent_source_id
        if not isinstance(parent_id, str) or not parent_id:
            raise ValueError("continuous node placement has no parent source identity")
        # Deliberately do not read or transform the continuous absolute birth,
        # displacement, or parent point.  The ribbon obtains its position from
        # the live-attested rVPM frontier and applies the one-third rule.
        velocity = np.zeros(3, dtype=float)
        topology_owner = "attested_rvpm_frontier"
        velocity_used = False
    else:
        if mode != "inactive" or active:
            raise ValueError("inactive node placement semantics are inconsistent")
        velocity = np.zeros(3, dtype=float)
        topology_owner = "none"
        velocity_used = False

    payload = _node_fact_payload(
        fact,
        wing_id=wing_id,
        node_id=node_id,
        source_step_index=source_step,
        source_time_s=source_time,
        event_digest=event_digest,
        lev_anchor=lev_anchor,
        tev_anchor=tev_anchor,
        chord=chord,
        speed=speed,
        x_axis=x_axis,
        z_axis=z_axis,
        span_axis=span_axis,
        patch_id=patch_id,
        frame_id=frame_id,
        lineage_id=lineage_id,
        geometry_token=geometry_token,
    )
    fact_digest = _digest_payload(b"fluxv-v5h-node-section-fact-v1", payload)
    return _ValidatedNode(
        fact=fact,
        node_id=node_id,
        source_step_index=source_step,
        source_time_s=source_time,
        event=event,
        event_digest=event_digest,
        fact_digest=fact_digest,
        lev_anchor=lev_anchor,
        tev_anchor=tev_anchor,
        chord=chord,
        speed=speed,
        x_axis=x_axis,
        z_axis=z_axis,
        span_axis=span_axis,
        patch_id=patch_id,
        frame_id=frame_id,
        lineage_id=lineage_id,
        geometry_token=geometry_token,
        active=active,
        mode=mode,
        velocity=velocity,
        mapped_birth=mapped_birth,
        reconstruction_residual=residual,
        topology_owner=topology_owner,
        velocity_used=velocity_used,
    )


def _kinematics_payload(value: SpanNodeKinematics) -> dict[str, Any]:
    if type(value) is not SpanNodeKinematics:
        raise ValueError("kinematics entries must be exactly SpanNodeKinematics")
    if set(vars(value)) != {
        "node_id",
        "anchor_position_gp1_m",
        "edge_velocity_gp1_m_per_s",
    }:
        raise ValueError("SpanNodeKinematics carries an injected or missing field")
    node_id = _stable_id("kinematics.node_id", value.node_id)
    anchor = _finite_vector(
        "kinematics.anchor_position_gp1_m", value.anchor_position_gp1_m
    )
    velocity = _finite_vector(
        "kinematics.edge_velocity_gp1_m_per_s", value.edge_velocity_gp1_m_per_s
    )
    return {
        "node_id": node_id,
        "anchor_position_gp1_m": tuple(float(item) for item in anchor),
        "edge_velocity_gp1_m_per_s": tuple(float(item) for item in velocity),
    }


def _result_payload(
    result: DVMNodePlacementResult, *, include_attestation: bool
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "wing_id": result.wing_id,
        "enabled": result.enabled,
        "status": result.status,
        "interface_id": result.interface_id,
        "canonical": result.canonical,
        "feedback_call_count": result.feedback_call_count,
        "source_step_index": result.source_step_index,
        "source_time_s": result.source_time_s,
        "delta_time_s": result.delta_time_s,
        "node_geometry_role": result.node_geometry_role,
        "exclusive_strength_owner": result.exclusive_strength_owner,
        "exclusive_surface_owner": result.exclusive_surface_owner,
        "kinematics": tuple(_kinematics_payload(value) for value in result.kinematics),
        "node_ledgers": tuple(asdict(value) for value in result.node_ledgers),
        "cell_coverage": tuple(asdict(value) for value in result.cell_coverage),
        "feedback_velocity": result.feedback_velocity,
    }
    if include_attestation:
        payload["producer_manifest_sha256"] = result.producer_manifest_sha256
    return payload


def _result_digest(result: DVMNodePlacementResult) -> str:
    return _digest_payload(
        b"fluxv-v5h-direct-node-placement-result-v1",
        _result_payload(result, include_attestation=False),
    )


def _validate_result_semantics(result: object) -> DVMNodePlacementResult:
    if type(result) is not DVMNodePlacementResult:
        raise ValueError("result must be exactly DVMNodePlacementResult")
    wing_id = _explicit_text("result.wing_id", result.wing_id)
    enabled = _strict_boolean("result.enabled", result.enabled)
    if result.interface_id != INTERFACE_ID or result.canonical is not False:
        raise ValueError(
            "node placement result identity/canonical status is not pinned"
        )
    if result.feedback_call_count != 0 or result.feedback_velocity is not None:
        raise ValueError("node placement result cannot carry feedback")
    if (
        result.node_geometry_role != NODE_GEOMETRY_ROLE
        or result.exclusive_strength_owner != EXCLUSIVE_STRENGTH_OWNER
        or result.exclusive_surface_owner != EXCLUSIVE_SURFACE_OWNER
    ):
        raise ValueError("node placement result owner boundary is not pinned")
    if not enabled:
        if result.status != DISABLED_STATUS:
            raise ValueError("disabled placement result status is not pinned")
        if (
            any(
                value is not None
                for value in (
                    result.source_step_index,
                    result.source_time_s,
                    result.delta_time_s,
                )
            )
            or result.kinematics
            or result.node_ledgers
            or result.cell_coverage
        ):
            raise ValueError("disabled placement result carries evaluated data")
        return result

    if result.status != STATUS:
        raise ValueError("enabled placement result status is not pinned")
    source_step = _strict_integer(
        "result.source_step_index", result.source_step_index, minimum=1
    )
    source_time = _finite_real("result.source_time_s", result.source_time_s)
    delta_time = _positive_real("result.delta_time_s", result.delta_time_s)
    if source_time < 0.0:
        raise ValueError("result source time cannot be negative")
    if not result.kinematics or len(result.kinematics) != len(result.node_ledgers):
        raise ValueError("result must carry one kinematics entry per node ledger")
    if not result.cell_coverage:
        raise ValueError("result must carry cell endpoint coverage")

    kinematics = {
        payload["node_id"]: payload
        for payload in (_kinematics_payload(item) for item in result.kinematics)
    }
    if len(kinematics) != len(result.kinematics):
        raise ValueError("result repeats a kinematics node ID")
    ledgers: dict[StableId, DVMNodePlacementLedger] = {}
    for ledger in result.node_ledgers:
        if type(ledger) is not DVMNodePlacementLedger:
            raise ValueError("node ledger type is not pinned")
        if ledger.wing_id != wing_id:
            raise ValueError("node ledger is bound to another wing")
        node_id = _stable_id("node ledger ID", ledger.node_id)
        if node_id in ledgers:
            raise ValueError("result repeats a node ledger ID")
        ledgers[node_id] = ledger
        if ledger.source_step_index != source_step or not _close(
            ledger.source_time_s, source_time
        ):
            raise ValueError("node ledger source layer disagrees with result")
        if ledger.placement_mode not in {
            "inactive",
            "first",
            "continuous",
            "restart",
        }:
            raise ValueError("node ledger mode is unsupported")
        expected_active = ledger.placement_mode != "inactive"
        if ledger.active is not expected_active:
            raise ValueError("node ledger activity and mode disagree")
        expected_owner: TopologyOwner = (
            "none"
            if ledger.placement_mode == "inactive"
            else (
                "attested_rvpm_frontier"
                if ledger.placement_mode == "continuous"
                else "node_local_dvm_relative_birth"
            )
        )
        if ledger.topology_owner != expected_owner:
            raise ValueError("node ledger topology owner disagrees with mode")
        expected_velocity_use = ledger.placement_mode in {"first", "restart"}
        if ledger.edge_velocity_used_by_ribbon is not expected_velocity_use:
            raise ValueError("node ledger velocity-use flag disagrees with mode")
        if ledger.dvm_absolute_birth_used_for_topology is not False:
            raise ValueError("DVM absolute birth cannot own GP1 topology")
        for digest in (
            ledger.node_fact_manifest_sha256,
            ledger.source_event_manifest_sha256,
            ledger.source_parent_event_manifest_sha256,
        ):
            if not isinstance(digest, str) or _SHA256_PATTERN.fullmatch(digest) is None:
                raise ValueError("node ledger digest is not lowercase SHA-256")
        payload = kinematics.get(node_id)
        if payload is None:
            raise ValueError("node ledger has no matching kinematics entry")
        if tuple(payload["anchor_position_gp1_m"]) != ledger.lev_edge_anchor_gp1_m:
            raise ValueError("node ledger and kinematics anchors disagree")
        if tuple(payload["edge_velocity_gp1_m_per_s"]) != (
            ledger.edge_velocity_gp1_m_per_s
        ):
            raise ValueError("node ledger and kinematics velocities disagree")
        if ledger.placement_mode in {"inactive", "continuous"}:
            if ledger.edge_velocity_gp1_m_per_s != (0.0, 0.0, 0.0):
                raise ValueError("inactive/continuous ribbon velocity must be zero")
            if ledger.mapped_relative_birth_gp1_m is not None:
                raise ValueError(
                    "inactive/continuous mode cannot map DVM birth topology"
                )
        elif ledger.mapped_relative_birth_gp1_m is None:
            raise ValueError("first/restart mode has no mapped relative birth")
        if not np.isfinite(ledger.half_step_reconstruction_residual_m):
            raise ValueError("node ledger half-step residual must be finite")

    if set(kinematics) != set(ledgers):
        raise ValueError("kinematics and node ledger identity sets differ")
    cell_ids: set[StableId] = set()
    for coverage in result.cell_coverage:
        if type(coverage) is not DVMCellEndpointCoverage:
            raise ValueError("cell coverage type is not pinned")
        if coverage.wing_id != wing_id:
            raise ValueError("cell coverage is bound to another wing")
        cell_id = _stable_id("coverage cell ID", coverage.cell_id)
        if cell_id in cell_ids:
            raise ValueError("result repeats a cell coverage ID")
        cell_ids.add(cell_id)
        if coverage.source_step_index != source_step:
            raise ValueError("cell coverage source step disagrees with result")
        if (
            coverage.endpoint_coverage_complete is not True
            or coverage.shared_fact_identity_verified is not True
        ):
            raise ValueError("cell endpoint coverage is incomplete")
        left = ledgers.get(coverage.left_node_id)
        right = ledgers.get(coverage.right_node_id)
        if left is None or right is None:
            raise ValueError("cell coverage references an unknown node")
        if (
            left.node_fact_manifest_sha256 != coverage.left_node_fact_manifest_sha256
            or right.node_fact_manifest_sha256
            != coverage.right_node_fact_manifest_sha256
        ):
            raise ValueError("cell coverage node fact digest is inconsistent")
        if not (
            left.active == right.active == coverage.active
            and left.placement_mode == right.placement_mode == coverage.placement_mode
        ):
            raise ValueError("cell coverage has mixed endpoint activity/mode")
        if not (
            left.topology_patch_id
            == right.topology_patch_id
            == coverage.topology_patch_id
            and left.coordinate_frame_id
            == right.coordinate_frame_id
            == coverage.coordinate_frame_id
        ):
            raise ValueError("cell coverage crosses a patch or frame boundary")
        if (
            _SHA256_PATTERN.fullmatch(coverage.cell_source_event_manifest_sha256)
            is None
        ):
            raise ValueError("cell source event digest is not lowercase SHA-256")
    del delta_time
    return result


def _attest_result(result: DVMNodePlacementResult) -> DVMNodePlacementResult:
    _validate_result_semantics(result)
    digest = _result_digest(result)
    object.__setattr__(result, "producer_manifest_sha256", digest)
    result_id = id(result)

    def discard(reference: weakref.ReferenceType[DVMNodePlacementResult]) -> None:
        registered = _DIRECT_RESULT_REGISTRY.get(result_id)
        if registered is not None and registered[0] is reference:
            _DIRECT_RESULT_REGISTRY.pop(result_id, None)

    reference = weakref.ref(result, discard)
    _DIRECT_RESULT_REGISTRY[result_id] = (reference, digest)
    return result


def validate_live_dvm_node_placement_result(
    result: object,
    *,
    expected_wing_id: str | None = None,
) -> DVMNodePlacementResult:
    """Validate a live producer result; copied/recomputed manifests fail."""

    validated = _validate_result_semantics(result)
    if expected_wing_id is not None:
        expected_wing = _explicit_text("expected_wing_id", expected_wing_id)
        if validated.wing_id != expected_wing:
            raise ValueError("node placement result is bound to another wing")
    digest = validated.producer_manifest_sha256
    if not isinstance(digest, str) or _SHA256_PATTERN.fullmatch(digest) is None:
        raise ValueError("result producer attestation is not lowercase SHA-256")
    recomputed = _result_digest(validated)
    if not hmac.compare_digest(digest, recomputed):
        raise ValueError("result producer manifest digest does not match its fields")
    registered = _DIRECT_RESULT_REGISTRY.get(id(validated))
    if (
        registered is None
        or registered[0]() is not validated
        or not hmac.compare_digest(registered[1], recomputed)
    ):
        raise ValueError("result is not a directly produced live placement object")
    return validated


class NodeLocalDVMPlacementAdapter:
    """Stateful, transactional node-local placement-to-GP1 adapter."""

    def __init__(self, *, wing_id: str) -> None:
        self._wing_id = _explicit_text("wing_id", wing_id)
        self._last_source_step: int | None = None
        self._last_source_time_s: float | None = None
        self._last_delta_time_s: float | None = None
        self._node_bindings: dict[StableId, _NodeBinding] = {}
        self._cell_bindings: dict[StableId, _CellBinding] = {}
        self._node_activity: dict[StableId, _ActivityState] = {}
        self._cell_activity: dict[StableId, _ActivityState] = {}
        self._node_event_digests: dict[StableId, str] = {}
        self._cell_event_digests: dict[StableId, str] = {}

    def state_manifest(self) -> dict[str, Any]:
        """Return a deterministic read-only snapshot for rollback auditing."""

        return {
            "wing_id": self._wing_id,
            "last_source_step": self._last_source_step,
            "last_source_time_s": self._last_source_time_s,
            "last_delta_time_s": self._last_delta_time_s,
            "node_bindings": tuple(
                (node_id, asdict(binding))
                for node_id, binding in sorted(
                    self._node_bindings.items(), key=lambda item: _id_key(item[0])
                )
            ),
            "cell_bindings": tuple(
                (cell_id, asdict(binding))
                for cell_id, binding in sorted(
                    self._cell_bindings.items(), key=lambda item: _id_key(item[0])
                )
            ),
            "node_activity": tuple(
                (node_id, asdict(state))
                for node_id, state in sorted(
                    self._node_activity.items(), key=lambda item: _id_key(item[0])
                )
            ),
            "cell_activity": tuple(
                (cell_id, asdict(state))
                for cell_id, state in sorted(
                    self._cell_activity.items(), key=lambda item: _id_key(item[0])
                )
            ),
            "node_event_digests": tuple(
                sorted(
                    self._node_event_digests.items(), key=lambda item: _id_key(item[0])
                )
            ),
            "cell_event_digests": tuple(
                sorted(
                    self._cell_event_digests.items(), key=lambda item: _id_key(item[0])
                )
            ),
        }

    def _disabled_result(self) -> DVMNodePlacementResult:
        return _attest_result(
            DVMNodePlacementResult(
                wing_id=self._wing_id,
                enabled=False,
                status=DISABLED_STATUS,
                interface_id=INTERFACE_ID,
                canonical=False,
                feedback_call_count=0,
                source_step_index=None,
                source_time_s=None,
                delta_time_s=None,
                node_geometry_role=NODE_GEOMETRY_ROLE,
                exclusive_strength_owner=EXCLUSIVE_STRENGTH_OWNER,
                exclusive_surface_owner=EXCLUSIVE_SURFACE_OWNER,
                kinematics=(),
                node_ledgers=(),
                cell_coverage=(),
                feedback_velocity=None,
                producer_manifest_sha256="",
            )
        )

    def map_step(
        self,
        cells: Sequence[DVMNodePlacementCell],
        *,
        delta_time_s: object,
        enabled: bool = True,
    ) -> DVMNodePlacementResult:
        """Map one source layer transactionally into ribbon-compatible facts."""

        enabled_value = _strict_boolean("enabled", enabled)
        if not enabled_value:
            return self._disabled_result()

        dt_s = _positive_real("delta_time_s", delta_time_s)
        try:
            cell_values = tuple(cells)
        except TypeError as error:
            raise ValueError("cells must be a finite sequence") from error
        if not cell_values:
            raise ValueError("cells must not be empty")

        validated_nodes: dict[StableId, _ValidatedNode] = {}
        live_node_facts: dict[StableId, GP1NodeSectionFact] = {}
        cell_events: dict[StableId, DVMSourceEvent] = {}
        cell_modes: dict[StableId, NodePlacementMode] = {}
        cell_records: list[
            tuple[StableId, _ValidatedNode, _ValidatedNode, DVMSourceEvent]
        ] = []
        endpoint_pairs: set[frozenset[StableId]] = set()
        geometric_edge_pairs: set[frozenset[tuple[object, ...]]] = set()
        node_degree: dict[StableId, int] = {}
        node_indegree: dict[StableId, int] = {}
        node_outdegree: dict[StableId, int] = {}
        physical_node_ids: dict[tuple[object, ...], StableId] = {}
        node_physical_keys: dict[StableId, tuple[object, ...]] = {}
        patch_nodes: dict[tuple[str, str], set[StableId]] = {}
        patch_edges: dict[tuple[str, str], list[tuple[StableId, StableId]]] = {}
        all_live_event_ids: set[int] = set()
        source_lineages: set[str] = set()
        physical_strips: set[str] = set()

        def consume_node(fact: object) -> _ValidatedNode:
            if type(fact) is not GP1NodeSectionFact:
                raise ValueError("cell endpoints must be GP1NodeSectionFact")
            node_id = _stable_id("endpoint node_id", fact.node_id)
            previous_fact = live_node_facts.get(node_id)
            if previous_fact is not None:
                if previous_fact is not fact:
                    raise ValueError(
                        "a shared node must reuse the exact same live node fact object"
                    )
                return validated_nodes[node_id]
            validated = _validate_node_fact(
                fact,
                delta_time_s=dt_s,
                expected_wing_id=self._wing_id,
            )
            physical_key = _physical_node_key(validated)
            physical_owner = physical_node_ids.get(physical_key)
            if physical_owner is not None and physical_owner != node_id:
                raise ValueError(
                    "one physical GP1 section in a patch/frame has two node IDs"
                )
            event_id = id(validated.event)
            lineage = validated.event.lineage.section_lineage_id
            strip = validated.event.lineage.physical_strip_id
            if (
                event_id in all_live_event_ids
                or lineage in source_lineages
                or strip in physical_strips
            ):
                raise ValueError(
                    "each span node requires an independent node-local DVM source"
                )
            all_live_event_ids.add(event_id)
            source_lineages.add(lineage)
            physical_strips.add(strip)
            live_node_facts[node_id] = fact
            validated_nodes[node_id] = validated
            physical_node_ids[physical_key] = node_id
            node_physical_keys[node_id] = physical_key
            return validated

        seen_cells: set[StableId] = set()
        for index, cell in enumerate(cell_values):
            if type(cell) is not DVMNodePlacementCell:
                raise ValueError(f"cells[{index}] must be exactly DVMNodePlacementCell")
            cell_id = _stable_id(f"cells[{index}].cell_id", cell.cell_id)
            if cell_id in seen_cells:
                raise ValueError("cell ID is repeated")
            seen_cells.add(cell_id)
            left = consume_node(cell.left_node_fact)
            right = consume_node(cell.right_node_fact)
            if left.node_id == right.node_id and type(left.node_id) is type(
                right.node_id
            ):
                raise ValueError("a cell must connect two distinct node IDs")
            pair = frozenset((left.node_id, right.node_id))
            if pair in endpoint_pairs:
                raise ValueError("two cells use the same unordered endpoint pair")
            endpoint_pairs.add(pair)
            geometric_pair = frozenset(
                (node_physical_keys[left.node_id], node_physical_keys[right.node_id])
            )
            if geometric_pair in geometric_edge_pairs:
                raise ValueError("a geometric span edge is duplicated or reversed")
            geometric_edge_pairs.add(geometric_pair)
            for endpoint in (left.node_id, right.node_id):
                node_degree[endpoint] = node_degree.get(endpoint, 0) + 1
                if node_degree[endpoint] > 2:
                    raise ValueError("span-cell topology is non-manifold at a node")
            node_outdegree[left.node_id] = node_outdegree.get(left.node_id, 0) + 1
            node_indegree[right.node_id] = node_indegree.get(right.node_id, 0) + 1
            if node_outdegree[left.node_id] > 1:
                raise ValueError("an ordered span node cannot have two outgoing cells")
            if node_indegree[right.node_id] > 1:
                raise ValueError("an ordered span node cannot have two incoming cells")
            if left.patch_id != right.patch_id:
                raise ValueError("a cell cannot cross a topology patch/hinge boundary")
            if left.frame_id != right.frame_id:
                raise ValueError("a cell cannot mix coordinate frames")
            patch_key = (left.patch_id, left.frame_id)
            patch_nodes.setdefault(patch_key, set()).update(
                (left.node_id, right.node_id)
            )
            patch_edges.setdefault(patch_key, []).append((left.node_id, right.node_id))
            span_delta_le = right.lev_anchor - left.lev_anchor
            span_delta_te = right.tev_anchor - left.tev_anchor
            if (
                min(
                    float(np.dot(span_delta_le, left.span_axis)),
                    float(np.dot(span_delta_le, right.span_axis)),
                    float(np.dot(span_delta_te, left.span_axis)),
                    float(np.dot(span_delta_te, right.span_axis)),
                )
                <= 0.0
            ):
                raise ValueError(
                    "cell endpoint order disagrees with positive span bases"
                )

            centre = validate_dvm_source_event(cell.cell_source_event)
            if not centre.enabled:
                raise ValueError(
                    "enabled placement coverage requires enabled cell events"
                )
            centre_id = id(centre)
            centre_lineage = centre.lineage.section_lineage_id
            centre_strip = centre.lineage.physical_strip_id
            if (
                centre_id in all_live_event_ids
                or centre_lineage in source_lineages
                or centre_strip in physical_strips
            ):
                raise ValueError(
                    "cell-centre and node-local DVM sources must be distinct"
                )
            all_live_event_ids.add(centre_id)
            source_lineages.add(centre_lineage)
            physical_strips.add(centre_strip)
            if not _close(_physical_dt_from_event(centre), dt_s, factor=1024.0):
                raise ValueError(
                    "cell source physical time step disagrees with handoff"
                )
            centre_step = centre.lineage.source_step_index
            if centre_step is None:
                raise ValueError("cell source event has no source step")
            centre_mode = _event_mode(centre)
            if not (left.source_step_index == right.source_step_index == centre_step):
                raise ValueError("cell and endpoint source steps disagree")
            if not (
                left.active == right.active == bool(centre.lesp_active)
                and left.mode == right.mode == centre_mode
            ):
                raise ValueError(
                    "cell endpoint coverage has mixed activity/mode (underresolved)"
                )
            cell_events[cell_id] = centre
            cell_modes[cell_id] = centre_mode
            cell_records.append((cell_id, left, right, centre))

        for patch_key, nodes_in_patch in patch_nodes.items():
            edges_in_patch = patch_edges[patch_key]
            if len(edges_in_patch) != len(nodes_in_patch) - 1:
                raise ValueError(
                    "each topology patch/frame must form one connected span chain"
                )
            starts = [
                node_id
                for node_id in nodes_in_patch
                if node_indegree.get(node_id, 0) == 0
            ]
            ends = [
                node_id
                for node_id in nodes_in_patch
                if node_outdegree.get(node_id, 0) == 0
            ]
            if len(starts) != 1 or len(ends) != 1:
                raise ValueError(
                    "each topology patch/frame must have one span-chain start and end"
                )
            successor = {left_id: right_id for left_id, right_id in edges_in_patch}
            visited: set[StableId] = set()
            cursor: StableId | None = starts[0]
            while cursor is not None and cursor not in visited:
                visited.add(cursor)
                cursor = successor.get(cursor)
            if cursor is not None or visited != nodes_in_patch:
                raise ValueError(
                    "each topology patch/frame must be one acyclic ordered span chain"
                )

        source_steps = {node.source_step_index for node in validated_nodes.values()}
        source_times = {node.source_time_s for node in validated_nodes.values()}
        if len(source_steps) != 1:
            raise ValueError("all node facts must share one source step")
        if len(source_times) != 1:
            raise ValueError("all node facts must share one exact source time")
        source_step = next(iter(source_steps))
        source_time = next(iter(source_times))
        if self._last_source_step is None:
            if source_step != 1:
                raise ValueError(
                    "an adapter's first enabled handoff must be source step 1"
                )
        else:
            if source_step != self._last_source_step + 1:
                raise ValueError("node placement source steps must be consecutive")
            assert self._last_source_time_s is not None
            assert self._last_delta_time_s is not None
            expected_time = self._last_source_time_s + self._last_delta_time_s
            if not _close(source_time, expected_time, factor=1024.0):
                raise ValueError("node placement source time is stale or noncausal")

        candidate_node_bindings: dict[StableId, _NodeBinding] = {}
        candidate_cell_bindings: dict[StableId, _CellBinding] = {}
        candidate_node_activity: dict[StableId, _ActivityState] = {}
        candidate_cell_activity: dict[StableId, _ActivityState] = {}
        candidate_node_digests: dict[StableId, str] = {}
        candidate_cell_digests: dict[StableId, str] = {}

        for node_id, node in validated_nodes.items():
            binding = _NodeBinding(
                physical_section_id=node.event.lineage.physical_section_id,
                physical_strip_id=node.event.lineage.physical_strip_id,
                section_lineage_id=node.event.lineage.section_lineage_id,
                topology_patch_id=node.patch_id,
                coordinate_frame_id=node.frame_id,
                geometry_token=node.geometry_token,
                reference_chord_m=node.chord,
                reference_speed_m_per_s=node.speed,
            )
            previous_binding = self._node_bindings.get(node_id)
            if previous_binding is not None and binding != previous_binding:
                raise ValueError(
                    "node/source/frame binding changed after first handoff"
                )
            previous_state = self._node_activity.get(
                node_id, _ActivityState(ever_active=False, active_last_step=False)
            )
            if node.mode != _expected_mode(previous_state, node.active):
                raise ValueError("node DVM mode is inconsistent with adapter history")
            previous_digest = self._node_event_digests.get(node_id)
            if previous_digest is not None and not hmac.compare_digest(
                node.event.parent_event_manifest_sha256, previous_digest
            ):
                raise ValueError("node DVM event chain does not match prior handoff")
            candidate_node_bindings[node_id] = binding
            candidate_node_activity[node_id] = _ActivityState(
                ever_active=previous_state.ever_active or node.active,
                active_last_step=node.active,
            )
            candidate_node_digests[node_id] = node.event_digest

        for cell_id, left, right, centre in cell_records:
            binding = _CellBinding(
                left_node_id=left.node_id,
                right_node_id=right.node_id,
                physical_section_id=centre.lineage.physical_section_id,
                physical_strip_id=centre.lineage.physical_strip_id,
                section_lineage_id=centre.lineage.section_lineage_id,
                topology_patch_id=left.patch_id,
                coordinate_frame_id=left.frame_id,
            )
            previous_binding = self._cell_bindings.get(cell_id)
            if previous_binding is not None and binding != previous_binding:
                raise ValueError(
                    "cell/source/endpoint binding changed after first handoff"
                )
            previous_state = self._cell_activity.get(
                cell_id, _ActivityState(ever_active=False, active_last_step=False)
            )
            centre_mode = cell_modes[cell_id]
            if centre_mode != _expected_mode(previous_state, bool(centre.lesp_active)):
                raise ValueError("cell DVM mode is inconsistent with adapter history")
            previous_digest = self._cell_event_digests.get(cell_id)
            if previous_digest is not None and not hmac.compare_digest(
                centre.parent_event_manifest_sha256, previous_digest
            ):
                raise ValueError("cell DVM event chain does not match prior handoff")
            candidate_cell_bindings[cell_id] = binding
            candidate_cell_activity[cell_id] = _ActivityState(
                ever_active=previous_state.ever_active or bool(centre.lesp_active),
                active_last_step=bool(centre.lesp_active),
            )
            candidate_cell_digests[cell_id] = centre.producer_manifest_sha256

        if self._node_bindings and set(validated_nodes) != set(self._node_bindings):
            raise ValueError("node identity set changed after first handoff")
        if self._cell_bindings and seen_cells != set(self._cell_bindings):
            raise ValueError("cell identity set changed after first handoff")

        ledgers = tuple(
            DVMNodePlacementLedger(
                wing_id=self._wing_id,
                node_id=node.node_id,
                source_step_index=source_step,
                source_time_s=source_time,
                active=node.active,
                placement_mode=node.mode,
                topology_owner=node.topology_owner,
                edge_velocity_used_by_ribbon=node.velocity_used,
                dvm_absolute_birth_used_for_topology=False,
                node_fact_manifest_sha256=node.fact_digest,
                source_event_manifest_sha256=node.event_digest,
                source_parent_event_manifest_sha256=(
                    node.event.parent_event_manifest_sha256
                ),
                source_newborn_id=(
                    node.event.lineage.newborn_lev_source_id if node.active else None
                ),
                continuous_parent_source_id=(
                    node.event.lev_placement.continuous_parent_source_id
                    if node.mode == "continuous"
                    else None
                ),
                topology_patch_id=node.patch_id,
                coordinate_frame_id=node.frame_id,
                node_lineage_id=node.lineage_id,
                geometry_token=node.geometry_token,
                lev_edge_anchor_gp1_m=tuple(float(item) for item in node.lev_anchor),
                tev_edge_anchor_gp1_m=tuple(float(item) for item in node.tev_anchor),
                edge_velocity_gp1_m_per_s=tuple(float(item) for item in node.velocity),
                mapped_relative_birth_gp1_m=(
                    None
                    if node.mapped_birth is None
                    else tuple(float(item) for item in node.mapped_birth)
                ),
                half_step_reconstruction_residual_m=node.reconstruction_residual,
            )
            for node in sorted(
                validated_nodes.values(), key=lambda item: _id_key(item.node_id)
            )
        )
        kinematics = tuple(
            SpanNodeKinematics(
                node_id=ledger.node_id,
                anchor_position_gp1_m=ledger.lev_edge_anchor_gp1_m,
                edge_velocity_gp1_m_per_s=ledger.edge_velocity_gp1_m_per_s,
            )
            for ledger in ledgers
        )
        coverage = tuple(
            DVMCellEndpointCoverage(
                wing_id=self._wing_id,
                cell_id=cell_id,
                source_step_index=source_step,
                active=bool(centre.lesp_active),
                placement_mode=cell_modes[cell_id],
                cell_source_event_manifest_sha256=(centre.producer_manifest_sha256),
                left_node_id=left.node_id,
                right_node_id=right.node_id,
                left_node_fact_manifest_sha256=left.fact_digest,
                right_node_fact_manifest_sha256=right.fact_digest,
                topology_patch_id=left.patch_id,
                coordinate_frame_id=left.frame_id,
                endpoint_coverage_complete=True,
                shared_fact_identity_verified=True,
            )
            for cell_id, left, right, centre in sorted(
                cell_records, key=lambda item: _id_key(item[0])
            )
        )
        raw_result = DVMNodePlacementResult(
            wing_id=self._wing_id,
            enabled=True,
            status=STATUS,
            interface_id=INTERFACE_ID,
            canonical=False,
            feedback_call_count=0,
            source_step_index=source_step,
            source_time_s=source_time,
            delta_time_s=dt_s,
            node_geometry_role=NODE_GEOMETRY_ROLE,
            exclusive_strength_owner=EXCLUSIVE_STRENGTH_OWNER,
            exclusive_surface_owner=EXCLUSIVE_SURFACE_OWNER,
            kinematics=kinematics,
            node_ledgers=ledgers,
            cell_coverage=coverage,
            feedback_velocity=None,
            producer_manifest_sha256="",
        )
        event_consumptions = tuple(
            (
                node.event,
                "lev-node-geometry",
                node.node_id,
                source_step,
            )
            for node in validated_nodes.values()
        ) + tuple(
            (
                centre,
                "lev-cell-coverage",
                cell_id,
                source_step,
            )
            for cell_id, _left, _right, centre in cell_records
        )

        # The global live-event ownership and adapter state are committed in
        # one critical section.  Every rejected call leaves both untouched,
        # so the exact same direct events can be used for a clean retry.
        with _LIVE_EVENT_CONSUMPTION_LOCK:
            _assert_events_unconsumed_locked(
                event_consumptions,
                wing_id=self._wing_id,
            )
            result = _attest_result(raw_result)
            _register_event_consumptions_locked(
                event_consumptions,
                wing_id=self._wing_id,
            )
            self._node_bindings = candidate_node_bindings
            self._cell_bindings = candidate_cell_bindings
            self._node_activity = candidate_node_activity
            self._cell_activity = candidate_cell_activity
            self._node_event_digests = candidate_node_digests
            self._cell_event_digests = candidate_cell_digests
            self._last_source_step = source_step
            self._last_source_time_s = source_time
            self._last_delta_time_s = dt_s
        return result


__all__ = [
    "DISABLED_STATUS",
    "DVMCellEndpointCoverage",
    "DVMNodePlacementCell",
    "DVMNodePlacementLedger",
    "DVMNodePlacementResult",
    "EXCLUSIVE_STRENGTH_OWNER",
    "EXCLUSIVE_SURFACE_OWNER",
    "GP1NodeSectionFact",
    "INTERFACE_ID",
    "NODE_GEOMETRY_ROLE",
    "NodeLocalDVMPlacementAdapter",
    "POSITION_UNITS",
    "STATUS",
    "VELOCITY_UNITS",
    "validate_live_dvm_node_placement_result",
]
