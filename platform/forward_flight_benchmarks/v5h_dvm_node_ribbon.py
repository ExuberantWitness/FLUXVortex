"""Node-owned three-dimensional ribbon shadow for v5h DVM sources.

This module is deliberately a mechanical adapter, not an aerodynamic model.
It consumes already-evaluated :class:`DVMSourceEvent` records, requires an
explicit two-dimensional-to-GP1 frame map, constructs newborn endpoints at
shared span-node identities, and delegates signed edge assembly to the frozen
``rvpm_edge_bridge``.  It never feeds velocity back to FluxV/Ptera and never
writes a surface-load quantity.

The sectional DVM birth coordinate is retained only as a transformed audit
point.  It is never interpolated into the three-dimensional ribbon: endpoint
geometry comes from node-local kinematics and persistent node frontier state.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import hmac
import json
from numbers import Integral, Real
import threading
from typing import Literal, Sequence, TypeAlias
import weakref

import numpy as np
from numpy.typing import ArrayLike, NDArray

from fluxvortex.rvpm_edge_bridge import (
    BridgeNode,
    DirectedRing,
    EdgeGraph,
    assemble_ring_edge_graph,
)

from .v5h_passive_frontier_transport import (
    NodeFrontierFact,
    PassiveFrontierTransportReport,
    parent_frontier_digest_sha256,
    ribbon_parent_digest_sha256,
    validate_passive_frontier_transport_report,
)
from .v5h_dvm_source import DVMSourceEvent, validate_dvm_source_event


FloatArray = NDArray[np.float64]
StableId: TypeAlias = int | str
SourceFamily = Literal["lev", "tev_persisted"]
NodeBirthMode = Literal["inactive", "first", "continuous", "restart"]

INTERFACE_ID = "fluxv-v5h-dvm-node-ribbon-shadow-v1"
OWNERSHIP_MODE = "diagnostic_shadow"


@dataclass(frozen=True)
class DVMPlaneToGP1Map:
    """Explicit map from the DVM x-z plane into the Ptera GP1 frame."""

    origin_gp1_m: ArrayLike
    x_axis_gp1: ArrayLike
    z_axis_gp1: ArrayLike
    positive_circulation_axis_gp1: ArrayLike
    circulation_to_ring_traversal_sign: int
    provenance: str


@dataclass(frozen=True)
class SpanNodeKinematics:
    """Current leading/trailing-edge node fact used by a source family."""

    node_id: StableId
    anchor_position_gp1_m: ArrayLike
    edge_velocity_gp1_m_per_s: ArrayLike


@dataclass(frozen=True)
class DVMSpanCellSource:
    """One evaluated sectional source attached to an oriented span cell."""

    cell_id: StableId
    left_node_id: StableId
    right_node_id: StableId
    event: DVMSourceEvent
    plane_to_gp1: DVMPlaneToGP1Map


@dataclass(frozen=True)
class NodeBirthRecord:
    """Auditable shared-node birth result for the current source step."""

    node_id: StableId
    mode: NodeBirthMode
    active: bool
    lineage_epoch: int | None
    birth_step_index: int | None
    anchor_node_id: str
    birth_node_id: str | None
    previous_frontier_id: str | None
    anchor_position_gp1_m: tuple[float, float, float]
    birth_position_gp1_m: tuple[float, float, float] | None


@dataclass(frozen=True)
class DVMCellRibbonLedger:
    """Per-cell source dimensionalization and topology ledger."""

    cell_id: StableId
    physical_strip_id: str
    source_lineage_id: str
    source_step_index: int
    source_family: SourceFamily
    active: bool
    gamma_star: float
    circulation_scale_u_times_c_m2_per_s: float
    gamma_cell_m2_per_s: float
    circulation_to_ring_traversal_sign: int
    ring_circulation_m2_per_s: float
    kelvin_residual_m2_per_s: float
    section_birth_audit_point_gp1_m: tuple[float, float, float] | None
    section_birth_point_used_for_topology: bool
    ring_id: str | None


@dataclass(frozen=True)
class DVMRibbonDiagnostics:
    """Mechanical-gate diagnostics for one source handoff."""

    enabled: bool
    interface_id: str
    ownership_mode: str
    canonical_eligible: bool
    feedback_call_count: int
    transport_advance_count: int
    source_family: SourceFamily
    source_step_index: int | None
    source_cell_count: int
    active_cell_count: int
    shared_node_count: int
    first_node_count: int
    continuous_node_count: int
    restart_node_count: int
    inactive_node_count: int
    max_kelvin_residual_m2_per_s: float
    incidence_residual: float
    edge_reconstruction_residual: float
    seam_count: int
    nonfinite_count: int
    source_reuse_count: int
    node_placement_bound: bool = False
    node_placement_manifest_sha256: str | None = None
    topology_patch_ids: tuple[str, ...] = ()
    coordinate_frame_ids: tuple[str, ...] = ()
    patch_binding_passed: bool = False


@dataclass(frozen=True)
class DVMRibbonShadowResult:
    """One-way source ribbon and audit ledger; no physical feedback exists."""

    edge_graph: EdgeGraph | None
    node_births: tuple[NodeBirthRecord, ...]
    cell_ledgers: tuple[DVMCellRibbonLedger, ...]
    diagnostics: DVMRibbonDiagnostics
    feedback_velocity: None


@dataclass(frozen=True)
class _NodeState:
    ever_active: bool
    active_last_step: bool
    lineage_epoch: int
    frontier_id: str | None
    frontier_birth_step_index: int | None
    frontier_position_gp1_m: tuple[float, float, float] | None


@dataclass(frozen=True)
class _CellBinding:
    left_node_id: StableId
    right_node_id: StableId
    physical_strip_id: str
    physical_section_id: str
    section_lineage_id: str
    source_family: SourceFamily


@dataclass(frozen=True)
class _CellHistory:
    source_step_index: int
    event_manifest_sha256: str
    gamma_tev_persisted_after: float
    gamma_lev_persisted_after: float
    gamma_deleted_after: float


@dataclass(frozen=True, slots=True)
class _PlacementCellBinding:
    """Attested node-placement identity bound to one ribbon cell."""

    topology_patch_id: str
    coordinate_frame_id: str
    left_node_fact_manifest_sha256: str
    right_node_fact_manifest_sha256: str


@dataclass(frozen=True, slots=True)
class _PlacementBinding:
    """Validated live node-placement result metadata for one handoff."""

    result: object
    producer_manifest_sha256: str
    cells: tuple[tuple[StableId, _PlacementCellBinding], ...]
    node_modes: tuple[tuple[StableId, NodeBirthMode, bool], ...]
    topology_patch_ids: tuple[str, ...]
    coordinate_frame_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _LiveEventConsumptionToken:
    """Context bound to one exact live DVM event consumption."""

    token_sha256: str
    wing_id: StableId
    source_family: SourceFamily
    cell_id: StableId
    left_node_id: StableId
    right_node_id: StableId
    physical_section_id: str
    physical_strip_id: str
    section_lineage_id: str
    source_step_index: int
    event_manifest_sha256: str
    parent_event_manifest_sha256: str
    source_placement_manifest_sha256: str
    plane_map_manifest_sha256: str
    node_placement_manifest_sha256: str | None
    node_placement_cell_binding_sha256: str | None
    topology_patch_id: str | None
    coordinate_frame_id: str | None


# A ribbon result is a trusted live producer input only while the exact object
# returned by ``map_step`` remains alive.  Its digest is public evidence, not a
# signing key, so copies made with ``dataclasses.replace`` are intentionally
# rejected by the passive-frontier producer.
_DIRECT_RIBBON_REGISTRY: dict[
    int, tuple[weakref.ReferenceType[DVMRibbonShadowResult], str]
] = {}

# Exact-once is deliberately a property of the *live source object*, not its
# public digest.  Two fresh source sessions may produce byte-identical events;
# they remain independent live facts and are therefore allowed.  Conversely,
# the exact same still-live event cannot be consumed by a second mapper, wing,
# or source-family role.  Weak references prevent this development gate from
# extending source-event lifetimes or turning object IDs into persistent IDs.
_LIVE_EVENT_CONSUMPTION_REGISTRY: dict[
    int,
    tuple[
        weakref.ReferenceType[DVMSourceEvent],
        _LiveEventConsumptionToken,
    ],
] = {}
_LIVE_PLACEMENT_CONSUMPTION_REGISTRY: dict[
    int,
    tuple[
        weakref.ReferenceType[object],
        str,
        StableId,
        SourceFamily,
    ],
] = {}
_LIVE_CUMULATIVE_REPORT_CONSUMPTION_REGISTRY: dict[
    int,
    tuple[
        weakref.ReferenceType[object],
        str,
    ],
] = {}
_LIVE_CONSUMPTION_LOCK = threading.RLock()


def _stable_id(name: str, value: object) -> StableId:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an explicit integer or string ID")
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, str) and value:
        return value
    raise ValueError(f"{name} must be a nonempty string or integer ID")


def _id_key(value: StableId) -> tuple[str, int | str]:
    return ("integer", value) if isinstance(value, int) else ("string", value)


def _id_token(value: StableId) -> str:
    payload = json.dumps(
        ["integer" if isinstance(value, int) else "string", value],
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256(payload).hexdigest()[:20]


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


def _finite_vector(name: str, value: ArrayLike) -> FloatArray:
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


def _unit_vector(name: str, value: ArrayLike) -> FloatArray:
    result = _finite_vector(name, value)
    norm = float(np.linalg.norm(result))
    if not np.isfinite(norm) or abs(norm - 1.0) > 1.0e-12:
        raise ValueError(f"{name} must be unit length to 1e-12")
    return result


def _strict_sign(name: str, value: object) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be exactly +1 or -1")
    result = int(value)
    if result not in (-1, 1):
        raise ValueError(f"{name} must be exactly +1 or -1")
    return result


def _explicit_text(name: str, value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be explicit")
    return value.strip()


def _validated_plane_map(
    value: object,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, int]:
    if not isinstance(value, DVMPlaneToGP1Map):
        raise ValueError("plane_to_gp1 must be an explicit DVMPlaneToGP1Map")
    origin = _finite_vector("plane origin", value.origin_gp1_m)
    x_axis = _unit_vector("DVM x axis", value.x_axis_gp1)
    z_axis = _unit_vector("DVM z axis", value.z_axis_gp1)
    positive_axis = _unit_vector(
        "DVM positive-circulation axis", value.positive_circulation_axis_gp1
    )
    if abs(float(np.dot(x_axis, z_axis))) > 1.0e-12:
        raise ValueError("DVM x and z axes must be orthogonal")
    oriented_axis = np.cross(x_axis, z_axis)
    if float(np.linalg.norm(oriented_axis - positive_axis)) > 1.0e-12:
        raise ValueError(
            "positive-circulation axis must explicitly match x_axis cross z_axis"
        )
    sign = _strict_sign(
        "circulation_to_ring_traversal_sign",
        value.circulation_to_ring_traversal_sign,
    )
    _explicit_text("plane-map provenance", value.provenance)
    return origin, x_axis, z_axis, positive_axis, sign


def _frontier_report_kind(
    value: object,
) -> Literal["passive_v1", "cumulative_v2", "dyadic_v5h2"]:
    """Classify only the three pre-registered transport report schemas."""

    if isinstance(value, PassiveFrontierTransportReport):
        return "passive_v1"
    # The cumulative producer imports this ribbon module lazily for handoff
    # attestation.  Keep the reverse dependency equally lazy and type-based so
    # a v2 report can never fall through the v1 validator by duck typing.
    from .v5h_cumulative_cloud_transport import CumulativeCloudTransportReport

    if isinstance(value, CumulativeCloudTransportReport):
        return "cumulative_v2"
    from .v5h2_dyadic_cumulative_cloud_transport import (
        DyadicCumulativeCloudTransportReport,
    )

    if isinstance(value, DyadicCumulativeCloudTransportReport):
        return "dyadic_v5h2"
    raise ValueError("frontier transport report has an unsupported live schema")


def _validate_frontier_report(
    value: object,
) -> tuple[Literal["passive_v1", "cumulative_v2", "dyadic_v5h2"], object]:
    kind = _frontier_report_kind(value)
    if kind == "passive_v1":
        return kind, validate_passive_frontier_transport_report(value)

    if kind == "cumulative_v2":
        from .v5h_cumulative_cloud_transport import (
            validate_cumulative_cloud_transport_report,
        )

        return kind, validate_cumulative_cloud_transport_report(value)

    from .v5h2_dyadic_cumulative_cloud_transport import (
        validate_dyadic_cumulative_cloud_transport_report,
    )

    return kind, validate_dyadic_cumulative_cloud_transport_report(value)


def _assert_live_cumulative_report_unconsumed_locked(
    report: object | None,
    *,
    report_sha256: str | None,
) -> None:
    """Fail if this exact live v2 report already drove another ribbon step."""

    if report is None:
        return
    if report_sha256 is None:
        raise RuntimeError("validated cumulative report lost its digest")
    report_id = id(report)
    existing = _LIVE_CUMULATIVE_REPORT_CONSUMPTION_REGISTRY.get(report_id)
    if existing is None:
        return
    live_report = existing[0]()
    if live_report is None:
        _LIVE_CUMULATIVE_REPORT_CONSUMPTION_REGISTRY.pop(report_id, None)
        return
    if live_report is not report:
        raise RuntimeError("live cumulative report identity registry collision")
    if not hmac.compare_digest(existing[1], report_sha256):
        raise RuntimeError("live cumulative report registry digest changed")
    raise ValueError("cumulative frontier report was already consumed (replay)")


def _register_live_cumulative_report_locked(
    report: object | None,
    *,
    report_sha256: str | None,
) -> None:
    """Commit weak global v2 report ownership after every ribbon gate passes."""

    if report is None:
        return
    if report_sha256 is None:
        raise RuntimeError("validated cumulative report lost its digest")
    report_id = id(report)

    def discard(
        reference: weakref.ReferenceType[object],
        *,
        consumed_report_id: int = report_id,
    ) -> None:
        with _LIVE_CONSUMPTION_LOCK:
            existing = _LIVE_CUMULATIVE_REPORT_CONSUMPTION_REGISTRY.get(
                consumed_report_id
            )
            if existing is not None and existing[0] is reference:
                _LIVE_CUMULATIVE_REPORT_CONSUMPTION_REGISTRY.pop(
                    consumed_report_id,
                    None,
                )

    reference = weakref.ref(report, discard)
    _LIVE_CUMULATIVE_REPORT_CONSUMPTION_REGISTRY[report_id] = (
        reference,
        report_sha256,
    )


def _sha256_json(domain: bytes, payload: object) -> str:
    encoded = json.dumps(
        payload,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    digest = sha256()
    digest.update(domain)
    digest.update(b"\0")
    digest.update(encoded)
    return digest.hexdigest()


def _source_placement_manifest_sha256(
    event: DVMSourceEvent,
    source_family: SourceFamily,
) -> str:
    placement = event.lev_placement if source_family == "lev" else event.tev_placement
    return _sha256_json(
        b"fluxv-v5h-ribbon-source-placement-binding-v1",
        {
            "source_family": source_family,
            "placement": asdict(placement),
        },
    )


def _plane_map_manifest_sha256(
    *,
    origin: FloatArray,
    x_axis: FloatArray,
    z_axis: FloatArray,
    positive_axis: FloatArray,
    sign: int,
    provenance: str,
) -> str:
    return _sha256_json(
        b"fluxv-v5h-ribbon-plane-map-binding-v1",
        {
            "origin_hex": tuple(float(item).hex() for item in origin),
            "x_axis_hex": tuple(float(item).hex() for item in x_axis),
            "z_axis_hex": tuple(float(item).hex() for item in z_axis),
            "positive_axis_hex": tuple(float(item).hex() for item in positive_axis),
            "circulation_to_ring_traversal_sign": sign,
            "provenance": provenance,
        },
    )


def _validate_live_node_placement_binding(
    value: object | None,
    *,
    wing_id: StableId,
    source_family: SourceFamily,
    delta_time_s: float,
    source_step_index: int,
    source_time_s: object | None,
    cells: tuple[DVMSpanCellSource, ...],
    nodes: tuple[SpanNodeKinematics, ...],
) -> _PlacementBinding | None:
    """Bind an optional live node-local placement result to this handoff.

    The import is intentionally local: ``v5h_dvm_node_placement`` consumes the
    public ``SpanNodeKinematics`` type from this module, so a module-level
    import would create a cycle.  The producer's own live validator remains
    the authority for its attestation and geometry-only ownership boundary.
    """

    if value is None:
        return None

    from .v5h_dvm_node_placement import (  # local import avoids a module cycle
        validate_live_dvm_node_placement_result,
    )

    result = validate_live_dvm_node_placement_result(value)
    if not result.enabled:
        raise ValueError("enabled ribbon handoff requires enabled node placement")
    if _id_key(_stable_id("node placement wing_id", result.wing_id)) != _id_key(
        wing_id
    ):
        raise ValueError("node placement result is bound to another wing")
    if source_family != "lev":
        raise ValueError(
            "LEV node placement result cannot be consumed by another source family"
        )
    if result.source_step_index != source_step_index:
        raise ValueError("node placement result is stale for this source step")
    if result.delta_time_s != delta_time_s:
        raise ValueError("node placement and ribbon physical time steps disagree")
    if source_time_s is not None:
        source_time = _finite_real("source_time_s", source_time_s)
        if result.source_time_s != source_time:
            raise ValueError("node placement result is stale for the source time")

    placement_nodes = {
        _stable_id("node placement kinematics ID", item.node_id): item
        for item in result.kinematics
    }
    if len(placement_nodes) != len(result.kinematics):
        raise ValueError("node placement result repeats a kinematics node ID")
    if {_id_key(item) for item in placement_nodes} != {
        _id_key(_stable_id("ribbon node ID", item.node_id)) for item in nodes
    }:
        raise ValueError("node placement and ribbon node identity sets differ")
    for node in nodes:
        node_id = _stable_id("ribbon node ID", node.node_id)
        expected = next(
            (
                item
                for key, item in placement_nodes.items()
                if _id_key(key) == _id_key(node_id)
            ),
            None,
        )
        if expected is not node:
            raise ValueError(
                "ribbon nodes must reuse the exact live placement kinematics"
            )

    coverage_by_key = {
        _id_key(_stable_id("node placement cell ID", item.cell_id)): item
        for item in result.cell_coverage
    }
    if len(coverage_by_key) != len(result.cell_coverage):
        raise ValueError("node placement result repeats a cell coverage ID")
    ribbon_cell_keys = {
        _id_key(_stable_id("ribbon cell ID", item.cell_id)) for item in cells
    }
    if set(coverage_by_key) != ribbon_cell_keys:
        raise ValueError("node placement and ribbon cell identity sets differ")

    cell_bindings: list[tuple[StableId, _PlacementCellBinding]] = []
    for cell in cells:
        cell_id = _stable_id("ribbon cell ID", cell.cell_id)
        coverage = coverage_by_key[_id_key(cell_id)]
        left = _stable_id("ribbon left node ID", cell.left_node_id)
        right = _stable_id("ribbon right node ID", cell.right_node_id)
        if _id_key(
            _stable_id("coverage left node ID", coverage.left_node_id)
        ) != _id_key(left) or _id_key(
            _stable_id("coverage right node ID", coverage.right_node_id)
        ) != _id_key(
            right
        ):
            raise ValueError("node placement cell endpoints disagree with ribbon")
        if not hmac.compare_digest(
            coverage.cell_source_event_manifest_sha256,
            cell.event.producer_manifest_sha256,
        ):
            raise ValueError("node placement is bound to another cell source event")
        expected_mode = cell.event.lev_placement.placement_mode
        if coverage.active is not bool(cell.event.lesp_active) or (
            coverage.placement_mode != expected_mode
        ):
            raise ValueError("node placement cell activity/mode disagrees with source")
        cell_bindings.append(
            (
                cell_id,
                _PlacementCellBinding(
                    topology_patch_id=_explicit_text(
                        "topology_patch_id", coverage.topology_patch_id
                    ),
                    coordinate_frame_id=_explicit_text(
                        "coordinate_frame_id", coverage.coordinate_frame_id
                    ),
                    left_node_fact_manifest_sha256=(
                        coverage.left_node_fact_manifest_sha256
                    ),
                    right_node_fact_manifest_sha256=(
                        coverage.right_node_fact_manifest_sha256
                    ),
                ),
            )
        )

    node_modes = tuple(
        (
            _stable_id("node placement ledger ID", ledger.node_id),
            ledger.placement_mode,
            ledger.active,
        )
        for ledger in result.node_ledgers
    )
    if {_id_key(item[0]) for item in node_modes} != {
        _id_key(_stable_id("ribbon node ID", item.node_id)) for item in nodes
    }:
        raise ValueError("node placement ledger and ribbon node sets differ")
    return _PlacementBinding(
        result=result,
        producer_manifest_sha256=result.producer_manifest_sha256,
        cells=tuple(sorted(cell_bindings, key=lambda item: _id_key(item[0]))),
        node_modes=tuple(sorted(node_modes, key=lambda item: _id_key(item[0]))),
        topology_patch_ids=tuple(
            sorted({item[1].topology_patch_id for item in cell_bindings})
        ),
        coordinate_frame_ids=tuple(
            sorted({item[1].coordinate_frame_id for item in cell_bindings})
        ),
    )


def _source_kind_values(
    event: DVMSourceEvent,
    family: SourceFamily,
) -> tuple[bool, float, tuple[float, float] | None, str]:
    if family == "lev":
        if event.lesp_active:
            if event.lineage.newborn_lev_source_id is None:
                raise ValueError("active LEV event has no newborn source identity")
            if event.lev_birth_position_over_chord_backend_world is None:
                raise ValueError("active LEV event has no captured birth position")
            return (
                True,
                _finite_real("Gamma_LEV/(Uc)", event.gamma_lev_new_over_u_c),
                event.lev_birth_position_over_chord_backend_world,
                event.lineage.newborn_lev_source_id,
            )
        if event.gamma_lev_new_over_u_c != 0.0:
            raise ValueError("inactive LEV event carries nonzero newborn circulation")
        return (
            False,
            0.0,
            None,
            (
                f"{event.lineage.section_lineage_id}:step:"
                f"{event.lineage.source_step_index}:lev-inactive"
            ),
        )
    if family == "tev_persisted":
        if event.tev_birth_position_over_chord_backend_world is None:
            raise ValueError("TEV event has no captured birth position")
        if event.lineage.newborn_tev_source_id is None:
            raise ValueError("TEV event has no newborn source identity")
        return (
            True,
            _finite_real(
                "Gamma_TEV_persisted/(Uc)",
                event.gamma_tev_new_persisted_over_u_c,
            ),
            event.tev_birth_position_over_chord_backend_world,
            event.lineage.newborn_tev_source_id,
        )
    raise ValueError("source_family must be 'lev' or 'tev_persisted'")


def _live_event_consumption_token(
    *,
    event: DVMSourceEvent,
    wing_id: StableId,
    source_family: SourceFamily,
    cell_id: StableId,
    left_node_id: StableId,
    right_node_id: StableId,
    source_step_index: int,
    plane_map_manifest_sha256: str,
    node_placement_manifest_sha256: str | None,
    placement_cell_binding: _PlacementCellBinding | None,
) -> _LiveEventConsumptionToken:
    placement_binding_sha256 = (
        None
        if placement_cell_binding is None
        else _sha256_json(
            b"fluxv-v5h-ribbon-node-placement-cell-binding-v1",
            asdict(placement_cell_binding),
        )
    )
    source_placement_digest = _source_placement_manifest_sha256(
        event,
        source_family,
    )
    payload = {
        "wing_id": _id_key(wing_id),
        "source_family": source_family,
        "cell_id": _id_key(cell_id),
        "left_node_id": _id_key(left_node_id),
        "right_node_id": _id_key(right_node_id),
        "physical_section_id": event.lineage.physical_section_id,
        "physical_strip_id": event.lineage.physical_strip_id,
        "section_lineage_id": event.lineage.section_lineage_id,
        "source_step_index": source_step_index,
        "event_manifest_sha256": event.producer_manifest_sha256,
        "parent_event_manifest_sha256": event.parent_event_manifest_sha256,
        "source_placement_manifest_sha256": source_placement_digest,
        "plane_map_manifest_sha256": plane_map_manifest_sha256,
        "node_placement_manifest_sha256": node_placement_manifest_sha256,
        "node_placement_cell_binding_sha256": placement_binding_sha256,
        "topology_patch_id": (
            None
            if placement_cell_binding is None
            else placement_cell_binding.topology_patch_id
        ),
        "coordinate_frame_id": (
            None
            if placement_cell_binding is None
            else placement_cell_binding.coordinate_frame_id
        ),
    }
    return _LiveEventConsumptionToken(
        token_sha256=_sha256_json(
            b"fluxv-v5h-ribbon-live-event-consumption-token-v1",
            payload,
        ),
        wing_id=wing_id,
        source_family=source_family,
        cell_id=cell_id,
        left_node_id=left_node_id,
        right_node_id=right_node_id,
        physical_section_id=event.lineage.physical_section_id,
        physical_strip_id=event.lineage.physical_strip_id,
        section_lineage_id=event.lineage.section_lineage_id,
        source_step_index=source_step_index,
        event_manifest_sha256=event.producer_manifest_sha256,
        parent_event_manifest_sha256=event.parent_event_manifest_sha256,
        source_placement_manifest_sha256=source_placement_digest,
        plane_map_manifest_sha256=plane_map_manifest_sha256,
        node_placement_manifest_sha256=node_placement_manifest_sha256,
        node_placement_cell_binding_sha256=placement_binding_sha256,
        topology_patch_id=(
            None
            if placement_cell_binding is None
            else placement_cell_binding.topology_patch_id
        ),
        coordinate_frame_id=(
            None
            if placement_cell_binding is None
            else placement_cell_binding.coordinate_frame_id
        ),
    )


def _assert_live_consumptions_unconsumed_locked(
    consumptions: Sequence[tuple[DVMSourceEvent, _LiveEventConsumptionToken]],
    *,
    placement_binding: _PlacementBinding | None,
) -> None:
    """Check module-wide exact-once ownership without mutating a registry."""

    candidate_event_ids: set[int] = set()
    for event, token in consumptions:
        event_id = id(event)
        if event_id in candidate_event_ids:
            raise ValueError("one live DVM event appears twice in a ribbon handoff")
        candidate_event_ids.add(event_id)
        existing = _LIVE_EVENT_CONSUMPTION_REGISTRY.get(event_id)
        if existing is None:
            continue
        live_event = existing[0]()
        if live_event is None:
            _LIVE_EVENT_CONSUMPTION_REGISTRY.pop(event_id, None)
            continue
        if live_event is not event:
            raise RuntimeError("live DVM event identity registry collision")
        prior = existing[1]
        if _id_key(prior.wing_id) != _id_key(token.wing_id):
            raise ValueError(
                "a live DVM event was already consumed by another wing's ribbon"
            )
        if prior.source_family != token.source_family:
            raise ValueError(
                "a live DVM event was already consumed by another source-family role"
            )
        if not hmac.compare_digest(prior.token_sha256, token.token_sha256):
            raise ValueError(
                "a live DVM event was already consumed under another patch/cell binding"
            )
        raise ValueError(
            "a live DVM event was already consumed by another ribbon mapper"
        )

    if placement_binding is None:
        return
    result = placement_binding.result
    result_id = id(result)
    existing_placement = _LIVE_PLACEMENT_CONSUMPTION_REGISTRY.get(result_id)
    if existing_placement is None:
        return
    live_result = existing_placement[0]()
    if live_result is None:
        _LIVE_PLACEMENT_CONSUMPTION_REGISTRY.pop(result_id, None)
        return
    if live_result is not result:
        raise RuntimeError("live node-placement identity registry collision")
    if _id_key(existing_placement[2]) != _id_key(consumptions[0][1].wing_id):
        raise ValueError("live node placement was already consumed by another wing")
    if existing_placement[3] != consumptions[0][1].source_family:
        raise ValueError(
            "live node placement was already consumed by another source-family role"
        )
    if not hmac.compare_digest(
        existing_placement[1],
        placement_binding.producer_manifest_sha256,
    ):
        raise RuntimeError("live node-placement registry manifest changed")
    raise ValueError(
        "live node placement was already consumed by another ribbon mapper"
    )


def _register_live_consumptions_locked(
    consumptions: Sequence[tuple[DVMSourceEvent, _LiveEventConsumptionToken]],
    *,
    placement_binding: _PlacementBinding | None,
) -> None:
    """Commit weak exact-once ownership after every mechanical gate passes."""

    prepared_events: list[
        tuple[
            int,
            weakref.ReferenceType[DVMSourceEvent],
            _LiveEventConsumptionToken,
        ]
    ] = []
    for event, token in consumptions:
        event_id = id(event)

        def discard_event(
            reference: weakref.ReferenceType[DVMSourceEvent],
            *,
            consumed_event_id: int = event_id,
        ) -> None:
            with _LIVE_CONSUMPTION_LOCK:
                existing = _LIVE_EVENT_CONSUMPTION_REGISTRY.get(consumed_event_id)
                if existing is not None and existing[0] is reference:
                    _LIVE_EVENT_CONSUMPTION_REGISTRY.pop(consumed_event_id, None)

        prepared_events.append((event_id, weakref.ref(event, discard_event), token))

    prepared_placement: tuple[int, weakref.ReferenceType[object]] | None = None
    if placement_binding is not None:
        result = placement_binding.result
        result_id = id(result)

        def discard_placement(
            reference: weakref.ReferenceType[object],
            *,
            consumed_result_id: int = result_id,
        ) -> None:
            with _LIVE_CONSUMPTION_LOCK:
                existing = _LIVE_PLACEMENT_CONSUMPTION_REGISTRY.get(consumed_result_id)
                if existing is not None and existing[0] is reference:
                    _LIVE_PLACEMENT_CONSUMPTION_REGISTRY.pop(
                        consumed_result_id,
                        None,
                    )

        prepared_placement = (
            result_id,
            weakref.ref(result, discard_placement),
        )

    for event_id, reference, token in prepared_events:
        _LIVE_EVENT_CONSUMPTION_REGISTRY[event_id] = (reference, token)
    if placement_binding is not None and prepared_placement is not None:
        first_token = consumptions[0][1]
        _LIVE_PLACEMENT_CONSUMPTION_REGISTRY[prepared_placement[0]] = (
            prepared_placement[1],
            placement_binding.producer_manifest_sha256,
            first_token.wing_id,
            first_token.source_family,
        )


def _validate_source_event_ledger(event: DVMSourceEvent, scale: float) -> float:
    """Independently reconstruct the source's LESP and Kelvin ledgers."""

    if event.a0_pre is None or event.a0_post is None:
        raise ValueError("evaluated DVM event has no A0 source state")
    a0_pre = _finite_real("A0_pre", event.a0_pre)
    a0_post = _finite_real("A0_post", event.a0_post)
    critical = _positive_real("LESP critical", event.lesp_critical)
    target = float(np.clip(a0_pre, -critical, critical))
    if event.lesp_signed_target is None or event.lesp_constraint_residual is None:
        raise ValueError("evaluated DVM event has no signed LESP ledger")
    saved_target = _finite_real("signed LESP target", event.lesp_signed_target)
    saved_residual = _finite_real(
        "LESP constraint residual", event.lesp_constraint_residual
    )
    tolerance = (
        1024.0 * np.finfo(float).eps * max(1.0, abs(a0_pre), abs(a0_post), critical)
    )
    if abs(saved_target - target) > tolerance:
        raise ValueError("signed LESP target is not independently reproducible")
    if abs(saved_residual - (a0_post - target)) > tolerance:
        raise ValueError("LESP residual is not independently reproducible")
    if event.lesp_active != (abs(a0_pre) > critical):
        raise ValueError("LESP activity disagrees with the strict source threshold")

    ledger = event.kelvin_ledger
    if ledger is None:
        raise ValueError("evaluated DVM event has no Kelvin ledger")
    if ledger.circulation_units != event.provenance.circulation_units:
        raise ValueError("Kelvin ledger and source circulation units disagree")
    values = {
        name: _finite_real(name, getattr(ledger, name))
        for name in (
            "gamma_bound_post",
            "gamma_old_tev_persisted",
            "gamma_old_lev_persisted",
            "gamma_deleted_before",
            "gamma_tev_new_solved",
            "gamma_tev_new_persisted",
            "gamma_lev_new_solved",
            "gamma_lev_new_persisted",
            "gamma_deleted_after",
            "gamma_deleted_delta",
            "gamma_tev_persisted_after",
            "gamma_lev_persisted_after",
            "tev_solved_to_persisted_delta",
            "kelvin_solve_residual",
            "persistence_residual",
        )
    }
    solve = (
        -values["gamma_bound_post"]
        + values["gamma_old_tev_persisted"]
        + values["gamma_old_lev_persisted"]
        + values["gamma_deleted_before"]
        + values["gamma_tev_new_solved"]
        + values["gamma_lev_new_solved"]
    )
    persistence = (
        values["gamma_tev_persisted_after"]
        + values["gamma_lev_persisted_after"]
        + values["gamma_deleted_after"]
        - (
            values["gamma_old_tev_persisted"]
            + values["gamma_old_lev_persisted"]
            + values["gamma_deleted_before"]
            + values["gamma_tev_new_persisted"]
            + values["gamma_lev_new_persisted"]
        )
    )
    identities = (
        (solve, values["kelvin_solve_residual"], "Kelvin solve"),
        (persistence, values["persistence_residual"], "persistence"),
        (
            values["gamma_deleted_after"] - values["gamma_deleted_before"],
            values["gamma_deleted_delta"],
            "deleted-circulation delta",
        ),
        (
            values["gamma_tev_new_persisted"] - values["gamma_tev_new_solved"],
            values["tev_solved_to_persisted_delta"],
            "TEV solved-to-persisted delta",
        ),
    )
    for recomputed, saved, name in identities:
        if abs(recomputed - saved) > tolerance:
            raise ValueError(f"{name} ledger is not independently reproducible")
    if event.kelvin_residual_over_u_c != values["kelvin_solve_residual"]:
        raise ValueError("event and detailed Kelvin residuals disagree")
    if event.gamma_lev_new_over_u_c != values["gamma_lev_new_persisted"]:
        raise ValueError("event and Kelvin-ledger LEV source strengths disagree")
    if (
        event.gamma_tev_new_solved_over_u_c != values["gamma_tev_new_solved"]
        or event.gamma_tev_new_persisted_over_u_c != values["gamma_tev_new_persisted"]
    ):
        raise ValueError("event and Kelvin-ledger TEV source strengths disagree")
    if abs(values["kelvin_solve_residual"] * scale) > 1.0e-10:
        raise ValueError("dimensional Kelvin residual exceeds 1e-10 m^2/s")
    return abs(values["kelvin_solve_residual"] * scale)


def _attest_direct_ribbon_result(
    result: DVMRibbonShadowResult,
    digest: str,
) -> DVMRibbonShadowResult:
    result_id = id(result)

    def discard(reference: weakref.ReferenceType[DVMRibbonShadowResult]) -> None:
        existing = _DIRECT_RIBBON_REGISTRY.get(result_id)
        if existing is not None and existing[0] is reference:
            _DIRECT_RIBBON_REGISTRY.pop(result_id, None)

    reference = weakref.ref(result, discard)
    _DIRECT_RIBBON_REGISTRY[result_id] = (reference, digest)
    return result


def validate_live_dvm_ribbon_shadow_result(
    result: object,
) -> DVMRibbonShadowResult:
    """Require the exact live result object returned by this mapper."""

    if not isinstance(result, DVMRibbonShadowResult):
        raise ValueError("parent ribbon must be a DVMRibbonShadowResult")
    digest = ribbon_parent_digest_sha256(result)
    registered = _DIRECT_RIBBON_REGISTRY.get(id(result))
    if (
        registered is None
        or registered[0]() is not result
        or not hmac.compare_digest(registered[1], digest)
    ):
        raise ValueError("parent ribbon is not a directly produced live mapper result")
    return result


class NodeOwnedDVMRibbonShadow:
    """Stateful shared-node mapper for one wing and one DVM source family."""

    def __init__(self, *, wing_id: StableId, source_family: SourceFamily) -> None:
        self._wing_id = _stable_id("wing_id", wing_id)
        if source_family not in ("lev", "tev_persisted"):
            raise ValueError("source_family must be 'lev' or 'tev_persisted'")
        self._source_family = source_family
        self._node_state: dict[StableId, _NodeState] = {}
        self._cell_bindings: dict[StableId, _CellBinding] = {}
        self._cell_history: dict[StableId, _CellHistory] = {}
        self._consumed_source_keys: set[tuple[str, int, SourceFamily]] = set()
        self._consumed_frontier_report_sha256: set[str] = set()
        self._last_source_step: int | None = None
        self._last_ribbon_digest_sha256: str | None = None
        self._commit_version = 0

    @property
    def state_snapshot(self) -> tuple[tuple[StableId, _NodeState], ...]:
        """Return a deterministic immutable snapshot for mechanical audits."""

        return tuple(
            sorted(self._node_state.items(), key=lambda item: _id_key(item[0]))
        )

    @property
    def cell_binding_snapshot(self) -> tuple[tuple[StableId, _CellBinding], ...]:
        """Return the first-success cell-to-source/topology identity freeze."""

        return tuple(
            sorted(self._cell_bindings.items(), key=lambda item: _id_key(item[0]))
        )

    @property
    def cell_history_snapshot(self) -> tuple[tuple[StableId, _CellHistory], ...]:
        """Return the committed per-cell event-chain and wake-history ledger."""

        return tuple(
            sorted(self._cell_history.items(), key=lambda item: _id_key(item[0]))
        )

    @property
    def transport_handoff_snapshot(self) -> tuple[str | None, tuple[str, ...]]:
        """Return parent-ribbon identity and exact consumed-report ledger."""

        return (
            self._last_ribbon_digest_sha256,
            tuple(sorted(self._consumed_frontier_report_sha256)),
        )

    def _disabled_result(self) -> DVMRibbonShadowResult:
        diagnostics = DVMRibbonDiagnostics(
            enabled=False,
            interface_id=INTERFACE_ID,
            ownership_mode=OWNERSHIP_MODE,
            canonical_eligible=False,
            feedback_call_count=0,
            transport_advance_count=0,
            source_family=self._source_family,
            source_step_index=None,
            source_cell_count=0,
            active_cell_count=0,
            shared_node_count=0,
            first_node_count=0,
            continuous_node_count=0,
            restart_node_count=0,
            inactive_node_count=0,
            max_kelvin_residual_m2_per_s=0.0,
            incidence_residual=0.0,
            edge_reconstruction_residual=0.0,
            seam_count=0,
            nonfinite_count=0,
            source_reuse_count=0,
        )
        return DVMRibbonShadowResult(
            edge_graph=None,
            node_births=(),
            cell_ledgers=(),
            diagnostics=diagnostics,
            feedback_velocity=None,
        )

    def map_step(
        self,
        cells: Sequence[DVMSpanCellSource],
        nodes: Sequence[SpanNodeKinematics],
        *,
        delta_time_s: object,
        enabled: bool = True,
        transport_enabled: bool = False,
        source_time_s: object | None = None,
        frontier_transport_report: object | None = None,
        node_placement_result: object | None = None,
    ) -> DVMRibbonShadowResult:
        """Map one evaluated source layer into a node-owned ribbon shadow.

        With ``transport_enabled=True``, every continuous node must consume an
        attested fact from a registered frontier-report schema.  First/restart
        nodes consume no fact.  The report and mapper state are committed only
        after all source, topology, time-layer, digest, and edge gates pass.
        """

        if not isinstance(enabled, (bool, np.bool_)):
            raise ValueError("enabled must be Boolean")
        if not bool(enabled):
            return self._disabled_result()

        starting_commit_version = self._commit_version

        if not isinstance(transport_enabled, (bool, np.bool_)):
            raise ValueError("transport_enabled must be Boolean")
        transport_enabled_value = bool(transport_enabled)
        if not transport_enabled_value and frontier_transport_report is not None:
            raise ValueError(
                "frontier_transport_report requires transport_enabled=True"
            )
        if transport_enabled_value and frontier_transport_report is not None:
            _frontier_report_kind(frontier_transport_report)
            if (
                frontier_transport_report.report_sha256
                in self._consumed_frontier_report_sha256
            ):
                raise ValueError(
                    "frontier transport report was already consumed (replay)"
                )

        dt_s = _positive_real("delta_time_s", delta_time_s)
        try:
            cell_values = tuple(cells)
            node_values = tuple(nodes)
        except TypeError as error:
            raise ValueError("cells and nodes must be finite sequences") from error
        if not cell_values or not node_values:
            raise ValueError("cells and nodes must not be empty")

        node_facts: dict[StableId, tuple[FloatArray, FloatArray]] = {}
        for index, node in enumerate(node_values):
            if not isinstance(node, SpanNodeKinematics):
                raise ValueError(f"nodes[{index}] must be SpanNodeKinematics")
            node_id = _stable_id(f"nodes[{index}].node_id", node.node_id)
            if node_id in node_facts:
                raise ValueError(f"node ID {node_id!r} is repeated")
            node_facts[node_id] = (
                _finite_vector(
                    f"nodes[{index}].anchor_position_gp1_m",
                    node.anchor_position_gp1_m,
                ),
                _finite_vector(
                    f"nodes[{index}].edge_velocity_gp1_m_per_s",
                    node.edge_velocity_gp1_m_per_s,
                ),
            )

        cell_records: list[
            tuple[
                StableId,
                StableId,
                StableId,
                DVMSourceEvent,
                FloatArray,
                FloatArray,
                FloatArray,
                FloatArray,
                int,
                bool,
                float,
                tuple[float, float] | None,
                str,
            ]
        ] = []
        cell_ids: set[StableId] = set()
        physical_strips: set[str] = set()
        endpoint_pairs: set[frozenset[StableId]] = set()
        referenced_nodes: set[StableId] = set()
        source_steps: set[int] = set()
        candidate_consumed: set[tuple[str, int, SourceFamily]] = set()
        candidate_bindings: dict[StableId, _CellBinding] = {}
        candidate_history: dict[StableId, _CellHistory] = {}
        plane_map_manifest_sha256: dict[StableId, str] = {}
        node_degree: dict[StableId, int] = {node_id: 0 for node_id in node_facts}

        for index, cell in enumerate(cell_values):
            if not isinstance(cell, DVMSpanCellSource):
                raise ValueError(f"cells[{index}] must be DVMSpanCellSource")
            cell_id = _stable_id(f"cells[{index}].cell_id", cell.cell_id)
            left = _stable_id(f"cells[{index}].left_node_id", cell.left_node_id)
            right = _stable_id(f"cells[{index}].right_node_id", cell.right_node_id)
            if cell_id in cell_ids:
                raise ValueError(f"cell ID {cell_id!r} is repeated")
            cell_ids.add(cell_id)
            if left == right and type(left) is type(right):
                raise ValueError("a span cell must connect two distinct node IDs")
            if left not in node_facts or right not in node_facts:
                raise ValueError("a span cell references an unknown node ID")
            pair = frozenset((left, right))
            if pair in endpoint_pairs:
                raise ValueError("two span cells use the same endpoint pair")
            endpoint_pairs.add(pair)
            referenced_nodes.update((left, right))
            node_degree[left] += 1
            node_degree[right] += 1
            if node_degree[left] > 2 or node_degree[right] > 2:
                raise ValueError("span-cell topology is non-manifold at a node")

            event = cell.event
            if not isinstance(event, DVMSourceEvent) or not event.enabled:
                raise ValueError("each mapped cell requires an enabled DVMSourceEvent")
            event = validate_dvm_source_event(event)
            if not event.status.startswith("evaluated_source_only_"):
                raise ValueError(
                    "DVM source status is not an evaluated source-only state"
                )
            if event.provenance.canonical:
                raise ValueError("DVM source unexpectedly claims canonical eligibility")
            if (
                event.provenance.observation_access != "none"
                or event.provenance.target_case_branch != "none"
            ):
                raise ValueError("DVM source is not isolated from target observations")
            source_step = event.lineage.source_step_index
            if source_step is None or isinstance(source_step, bool) or source_step < 1:
                raise ValueError("DVM source step identity is invalid")
            source_steps.add(int(source_step))
            physical_strip = _explicit_text(
                "physical_strip_id", event.lineage.physical_strip_id
            )
            if physical_strip in physical_strips:
                raise ValueError("physical_strip_id is reused by two span cells")
            physical_strips.add(physical_strip)
            binding = _CellBinding(
                left_node_id=left,
                right_node_id=right,
                physical_strip_id=physical_strip,
                physical_section_id=_explicit_text(
                    "physical_section_id", event.lineage.physical_section_id
                ),
                section_lineage_id=_explicit_text(
                    "section_lineage_id", event.lineage.section_lineage_id
                ),
                source_family=self._source_family,
            )
            frozen_binding = self._cell_bindings.get(cell_id)
            if frozen_binding is not None and binding != frozen_binding:
                raise ValueError(
                    "span-cell binding changed after its first successful handoff"
                )
            candidate_bindings[cell_id] = binding

            origin, x_axis, z_axis, positive_axis, sign = _validated_plane_map(
                cell.plane_to_gp1
            )
            plane_map_manifest_sha256[cell_id] = _plane_map_manifest_sha256(
                origin=origin,
                x_axis=x_axis,
                z_axis=z_axis,
                positive_axis=positive_axis,
                sign=sign,
                provenance=_explicit_text(
                    "plane-map provenance",
                    cell.plane_to_gp1.provenance,
                ),
            )
            anchor_span = node_facts[right][0] - node_facts[left][0]
            span_length = float(np.linalg.norm(anchor_span))
            if not np.isfinite(span_length) or span_length <= 0.0:
                raise ValueError("span-cell anchor edge must have positive length")
            traversal_axis = sign * anchor_span / span_length
            if float(np.linalg.norm(traversal_axis - positive_axis)) > 1.0e-10:
                raise ValueError(
                    "explicit circulation sign disagrees with the mapped GP1 span axis"
                )

            active, gamma_star, birth_2d, source_id = _source_kind_values(
                event, self._source_family
            )
            birth_2d_tuple: tuple[float, float] | None = None
            if birth_2d is not None:
                birth_2d_array = np.asarray(birth_2d, dtype=float)
                if birth_2d_array.shape != (2,) or not np.all(
                    np.isfinite(birth_2d_array)
                ):
                    raise ValueError("DVM section birth audit point is invalid")
                birth_2d_tuple = (
                    float(birth_2d_array[0]),
                    float(birth_2d_array[1]),
                )
            source_key = (
                event.lineage.section_lineage_id,
                int(source_step),
                self._source_family,
            )
            if (
                source_key in self._consumed_source_keys
                or source_key in candidate_consumed
            ):
                raise ValueError("a DVM source layer cannot be consumed more than once")
            candidate_consumed.add(source_key)

            ledger = event.kelvin_ledger
            if ledger is None:
                raise ValueError("evaluated DVM event has no Kelvin history ledger")
            prior_history = self._cell_history.get(cell_id)
            if prior_history is None:
                if source_step != 1:
                    raise ValueError(
                        "a cell's first DVM handoff must begin at source step 1"
                    )
            else:
                if source_step != prior_history.source_step_index + 1:
                    raise ValueError("per-cell DVM source steps are not consecutive")
                if (
                    event.parent_event_manifest_sha256
                    != prior_history.event_manifest_sha256
                ):
                    raise ValueError(
                        "DVM parent-event digest does not continue the cell history"
                    )
                history_pairs = (
                    (
                        ledger.gamma_old_tev_persisted,
                        prior_history.gamma_tev_persisted_after,
                        "TEV",
                    ),
                    (
                        ledger.gamma_old_lev_persisted,
                        prior_history.gamma_lev_persisted_after,
                        "LEV",
                    ),
                    (
                        ledger.gamma_deleted_before,
                        prior_history.gamma_deleted_after,
                        "deleted circulation",
                    ),
                )
                for current_value, previous_value, label in history_pairs:
                    if current_value != previous_value:
                        raise ValueError(
                            f"DVM {label} history is discontinuous across source steps"
                        )
            candidate_history[cell_id] = _CellHistory(
                source_step_index=int(source_step),
                event_manifest_sha256=event.producer_manifest_sha256,
                gamma_tev_persisted_after=float(ledger.gamma_tev_persisted_after),
                gamma_lev_persisted_after=float(ledger.gamma_lev_persisted_after),
                gamma_deleted_after=float(ledger.gamma_deleted_after),
            )

            scale = _positive_real(
                "circulation scale",
                event.provenance.circulation_scale_u_times_c_m2_per_s,
            )
            chord = _positive_real(
                "position scale", event.provenance.position_scale_chord_m
            )
            dt_convective = _positive_real(
                "DVM convective time step", event.delta_time_convective
            )
            expected_dt = dt_convective * chord * chord / scale
            if not np.isfinite(expected_dt) or abs(expected_dt - dt_s) > (
                1.0e-12 * max(1.0, dt_s, abs(expected_dt))
            ):
                raise ValueError("DVM convective and physical time steps disagree")
            _validate_source_event_ledger(event, scale)
            cell_records.append(
                (
                    cell_id,
                    left,
                    right,
                    event,
                    origin,
                    x_axis,
                    z_axis,
                    positive_axis,
                    sign,
                    active,
                    gamma_star,
                    birth_2d_tuple,
                    source_id,
                )
            )

        if self._cell_bindings and set(candidate_bindings) != set(self._cell_bindings):
            raise ValueError(
                "span-cell identity set changed after first successful handoff"
            )
        if self._cell_history and set(candidate_history) != set(self._cell_history):
            raise ValueError(
                "DVM cell-history identity set changed after first successful handoff"
            )

        if referenced_nodes != set(node_facts):
            raise ValueError("nodes must exactly match the span-cell topology")
        if len(source_steps) != 1:
            raise ValueError("all DVM cells must refer to one source time layer")
        source_step_index = next(iter(source_steps))
        if self._last_source_step is not None and source_step_index != (
            self._last_source_step + 1
        ):
            raise ValueError("DVM source layers must be consumed consecutively")

        # The full span topology must be one chain.  Active masks may split,
        # merge, grow, or shrink on this fixed chain without changing identity.
        if len(node_facts) != len(cell_records) + 1:
            raise ValueError("span-cell topology must be a connected open chain")
        visited = {next(iter(node_facts))}
        changed = True
        while changed:
            changed = False
            for _, left, right, *_ in cell_records:
                if left in visited and right not in visited:
                    visited.add(right)
                    changed = True
                elif right in visited and left not in visited:
                    visited.add(left)
                    changed = True
        if visited != set(node_facts):
            raise ValueError("span-cell topology must be connected")

        placement_binding = _validate_live_node_placement_binding(
            node_placement_result,
            wing_id=self._wing_id,
            source_family=self._source_family,
            delta_time_s=dt_s,
            source_step_index=source_step_index,
            source_time_s=source_time_s,
            cells=cell_values,
            nodes=node_values,
        )
        placement_cells = (
            {} if placement_binding is None else dict(placement_binding.cells)
        )

        active_nodes: set[StableId] = set()
        for _, left, right, _, _, _, _, _, _, active, _, _, _ in cell_records:
            if active:
                active_nodes.update((left, right))

        previous_state = self._node_state
        continuous_nodes = {
            node_id
            for node_id in active_nodes
            if node_id in previous_state
            and previous_state[node_id].ever_active
            and previous_state[node_id].active_last_step
        }
        accepted_frontier_facts: dict[StableId, NodeFrontierFact] = {}
        candidate_frontier_report_sha256: str | None = None
        candidate_cumulative_frontier_report: object | None = None
        if transport_enabled_value:
            if continuous_nodes and frontier_transport_report is None:
                raise ValueError(
                    "every continuous node requires an attested frontier fact"
                )
            if not continuous_nodes and frontier_transport_report is not None:
                raise ValueError("first/restart nodes consume no frontier report")
            if frontier_transport_report is not None:
                report_kind, report = _validate_frontier_report(
                    frontier_transport_report
                )
                if report_kind in ("cumulative_v2", "dyadic_v5h2"):
                    if not report.enabled:
                        raise ValueError(
                            "continuous nodes require an enabled cumulative report"
                        )
                    candidate_cumulative_frontier_report = report
                if _id_key(report.wing_id) != _id_key(self._wing_id):
                    raise ValueError("frontier report is bound to another wing")
                if report.source_family != self._source_family:
                    raise ValueError(
                        "frontier report is bound to another source family"
                    )
                if (
                    self._last_ribbon_digest_sha256 is None
                    or report.parent_ribbon_digest_sha256
                    != self._last_ribbon_digest_sha256
                ):
                    raise ValueError(
                        "frontier report is bound to another parent source/ribbon"
                    )
                if report_kind in ("cumulative_v2", "dyadic_v5h2") and (
                    report.current_ribbon_digest_sha256
                    != self._last_ribbon_digest_sha256
                    or report.current_ribbon_digest_sha256
                    != report.parent_ribbon_digest_sha256
                ):
                    raise ValueError(
                        "cumulative frontier report is bound to another current "
                        "parent ribbon"
                    )
                if (
                    report.parent_source_step_index != source_step_index - 1
                    or report.for_source_step_index != source_step_index
                ):
                    raise ValueError("frontier report is stale for this source step")
                if source_time_s is None:
                    raise ValueError(
                        "source_time_s is required for transported continuous nodes"
                    )
                source_time = _finite_real("source_time_s", source_time_s)
                time_tolerance = 1.0e-12 * max(1.0, abs(source_time), abs(dt_s))
                if abs(report.transport_end_time_s - source_time) > time_tolerance:
                    raise ValueError(
                        "frontier report is stale for the source time layer"
                    )
                if (
                    report_kind in ("cumulative_v2", "dyadic_v5h2")
                    and abs(report.transport_start_time_s - (source_time - dt_s))
                    > time_tolerance
                ):
                    raise ValueError(
                        "cumulative frontier report starts at another source "
                        "time layer"
                    )
                if (
                    abs(
                        (report.transport_end_time_s - report.transport_start_time_s)
                        - dt_s
                    )
                    > time_tolerance
                ):
                    raise ValueError(
                        "frontier report interval disagrees with the release step"
                    )
                report_facts: dict[StableId, NodeFrontierFact] = {}
                for fact in report.facts:
                    node_id = _stable_id("frontier fact node_id", fact.node_id)
                    if report_kind in ("cumulative_v2", "dyadic_v5h2") and (
                        _id_key(_stable_id("frontier fact wing_id", fact.wing_id))
                        != _id_key(self._wing_id)
                        or fact.source_family != self._source_family
                        or fact.for_source_step_index != source_step_index
                        or fact.parent_ribbon_digest_sha256
                        != self._last_ribbon_digest_sha256
                        or fact.producer_report_sha256 != report.report_sha256
                        or fact.transport_start_time_s != report.transport_start_time_s
                        or fact.transport_end_time_s != report.transport_end_time_s
                    ):
                        raise ValueError(
                            "cumulative frontier fact is not bound to this live "
                            "report/source layer"
                        )
                    if node_id not in previous_state:
                        raise ValueError("frontier report contains a cross-node fact")
                    previous = previous_state[node_id]
                    if (
                        not previous.ever_active
                        or not previous.active_last_step
                        or previous.frontier_id is None
                        or previous.frontier_birth_step_index is None
                        or previous.frontier_position_gp1_m is None
                    ):
                        raise ValueError(
                            "frontier fact has no active parent node frontier"
                        )
                    expected_digest = parent_frontier_digest_sha256(
                        wing_id=self._wing_id,
                        node_id=node_id,
                        source_family=self._source_family,
                        lineage_epoch=previous.lineage_epoch,
                        parent_frontier_id=previous.frontier_id,
                        parent_birth_step_index=(previous.frontier_birth_step_index),
                        parent_position_gp1_m=(previous.frontier_position_gp1_m),
                    )
                    if (
                        fact.lineage_epoch != previous.lineage_epoch
                        or fact.parent_frontier_id != previous.frontier_id
                        or fact.parent_birth_step_index
                        != previous.frontier_birth_step_index
                        or fact.parent_frontier_digest_sha256 != expected_digest
                    ):
                        raise ValueError(
                            "frontier fact does not match its parent birth/frontier"
                        )
                    report_facts[node_id] = fact
                expected_report_nodes = {
                    node_id
                    for node_id, state in previous_state.items()
                    if state.ever_active and state.active_last_step
                }
                if set(report_facts) != expected_report_nodes:
                    missing_parent = sorted(
                        expected_report_nodes - set(report_facts), key=_id_key
                    )
                    extra_parent = sorted(
                        set(report_facts) - expected_report_nodes, key=_id_key
                    )
                    raise ValueError(
                        "frontier report facts must exactly match all active "
                        "parent nodes; "
                        f"missing={missing_parent!r}, extra={extra_parent!r}"
                    )
                missing = sorted(continuous_nodes - set(report_facts), key=_id_key)
                if missing:
                    raise ValueError(
                        f"continuous nodes are missing frontier facts: {missing!r}"
                    )
                accepted_frontier_facts = {
                    node_id: report_facts[node_id] for node_id in continuous_nodes
                }
                candidate_frontier_report_sha256 = report.report_sha256

        next_state = dict(self._node_state)
        birth_records: dict[StableId, NodeBirthRecord] = {}
        for node_id in sorted(node_facts, key=_id_key):
            anchor, velocity = node_facts[node_id]
            previous = next_state.get(
                node_id,
                _NodeState(
                    ever_active=False,
                    active_last_step=False,
                    lineage_epoch=-1,
                    frontier_id=None,
                    frontier_birth_step_index=None,
                    frontier_position_gp1_m=None,
                ),
            )
            anchor_id = (
                f"v5h:{_id_token(self._wing_id)}:{self._source_family}:"
                f"anchor:{_id_token(node_id)}"
            )
            if node_id not in active_nodes:
                next_state[node_id] = _NodeState(
                    ever_active=previous.ever_active,
                    active_last_step=False,
                    lineage_epoch=previous.lineage_epoch,
                    frontier_id=previous.frontier_id,
                    frontier_birth_step_index=(previous.frontier_birth_step_index),
                    frontier_position_gp1_m=previous.frontier_position_gp1_m,
                )
                birth_records[node_id] = NodeBirthRecord(
                    node_id=node_id,
                    mode="inactive",
                    active=False,
                    lineage_epoch=None,
                    birth_step_index=None,
                    anchor_node_id=anchor_id,
                    birth_node_id=None,
                    previous_frontier_id=previous.frontier_id,
                    anchor_position_gp1_m=tuple(float(item) for item in anchor),
                    birth_position_gp1_m=None,
                )
                continue

            if not previous.ever_active:
                mode: NodeBirthMode = "first"
                epoch = 0
                birth = anchor + 0.5 * velocity * dt_s
                previous_frontier_id = None
            elif previous.active_last_step:
                mode = "continuous"
                epoch = previous.lineage_epoch
                if (
                    previous.frontier_id is None
                    or previous.frontier_position_gp1_m is None
                ):
                    raise ValueError("continuous node has no unique previous frontier")
                if transport_enabled_value:
                    fact = accepted_frontier_facts[node_id]
                    previous_frontier = np.asarray(
                        fact.advected_position_gp1_m, dtype=float
                    )
                else:
                    previous_frontier = np.asarray(
                        previous.frontier_position_gp1_m, dtype=float
                    )
                birth = anchor + (previous_frontier - anchor) / 3.0
                previous_frontier_id = previous.frontier_id
            else:
                mode = "restart"
                epoch = previous.lineage_epoch + 1
                birth = anchor + 0.5 * velocity * dt_s
                previous_frontier_id = previous.frontier_id
            if not np.all(np.isfinite(birth)):
                raise FloatingPointError("node birth position is non-finite")
            birth_id = (
                f"v5h:{_id_token(self._wing_id)}:{self._source_family}:"
                f"node:{_id_token(node_id)}:epoch:{epoch}:step:{source_step_index}"
            )
            birth_tuple = tuple(float(item) for item in birth)
            next_state[node_id] = _NodeState(
                ever_active=True,
                active_last_step=True,
                lineage_epoch=epoch,
                frontier_id=birth_id,
                frontier_birth_step_index=source_step_index,
                frontier_position_gp1_m=birth_tuple,
            )
            birth_records[node_id] = NodeBirthRecord(
                node_id=node_id,
                mode=mode,
                active=True,
                lineage_epoch=epoch,
                birth_step_index=source_step_index,
                anchor_node_id=anchor_id,
                birth_node_id=birth_id,
                previous_frontier_id=previous_frontier_id,
                anchor_position_gp1_m=tuple(float(item) for item in anchor),
                birth_position_gp1_m=birth_tuple,
            )

        bridge_nodes: dict[str, BridgeNode] = {}
        rings: list[DirectedRing] = []
        ledgers: list[DVMCellRibbonLedger] = []
        max_kelvin = 0.0
        for (
            cell_id,
            left,
            right,
            event,
            origin,
            x_axis,
            z_axis,
            _positive_axis,
            sign,
            active,
            gamma_star,
            birth_2d,
            source_id,
        ) in sorted(cell_records, key=lambda item: _id_key(item[0])):
            scale = float(event.provenance.circulation_scale_u_times_c_m2_per_s)
            chord = float(event.provenance.position_scale_chord_m)
            gamma_cell = gamma_star * scale
            if not np.isfinite(gamma_cell):
                raise FloatingPointError("dimensional DVM circulation is non-finite")
            ring_gamma = sign * gamma_cell
            kelvin_dimensional = abs(event.kelvin_residual_over_u_c * scale)
            max_kelvin = max(max_kelvin, kelvin_dimensional)
            section_point: FloatArray | None = None
            if birth_2d is not None:
                section_point = origin + chord * (
                    birth_2d[0] * x_axis + birth_2d[1] * z_axis
                )
                if not np.all(np.isfinite(section_point)):
                    raise FloatingPointError(
                        "mapped DVM section audit point is non-finite"
                    )
            ring_id: str | None = None
            if active:
                left_birth = birth_records[left]
                right_birth = birth_records[right]
                if (
                    left_birth.birth_node_id is None
                    or right_birth.birth_node_id is None
                ):
                    raise RuntimeError(
                        "active cell did not receive both shared endpoints"
                    )
                for record in (left_birth, right_birth):
                    if record.birth_position_gp1_m is None:
                        raise RuntimeError("active node birth position is missing")
                    bridge_nodes.setdefault(
                        record.birth_node_id,
                        BridgeNode(record.birth_node_id, record.birth_position_gp1_m),
                    )
                    bridge_nodes.setdefault(
                        record.anchor_node_id,
                        BridgeNode(record.anchor_node_id, record.anchor_position_gp1_m),
                    )
                ring_id = (
                    f"v5h:{_id_token(self._wing_id)}:{self._source_family}:"
                    f"cell:{_id_token(cell_id)}:step:{source_step_index}"
                )
                rings.append(
                    DirectedRing(
                        ring_id=ring_id,
                        node_ids=(
                            left_birth.birth_node_id,
                            right_birth.birth_node_id,
                            right_birth.anchor_node_id,
                            left_birth.anchor_node_id,
                        ),
                        circulation=ring_gamma,
                    )
                )
            ledgers.append(
                DVMCellRibbonLedger(
                    cell_id=cell_id,
                    physical_strip_id=event.lineage.physical_strip_id,
                    source_lineage_id=source_id,
                    source_step_index=source_step_index,
                    source_family=self._source_family,
                    active=active,
                    gamma_star=gamma_star,
                    circulation_scale_u_times_c_m2_per_s=scale,
                    gamma_cell_m2_per_s=gamma_cell,
                    circulation_to_ring_traversal_sign=sign,
                    ring_circulation_m2_per_s=ring_gamma,
                    kelvin_residual_m2_per_s=kelvin_dimensional,
                    section_birth_audit_point_gp1_m=(
                        None
                        if section_point is None
                        else tuple(float(item) for item in section_point)
                    ),
                    section_birth_point_used_for_topology=False,
                    ring_id=ring_id,
                )
            )

        graph = (
            assemble_ring_edge_graph(tuple(bridge_nodes.values()), tuple(rings))
            if rings
            else None
        )
        incidence_residual = 0.0 if graph is None else graph.incidence_residual
        reconstruction_residual = (
            0.0 if graph is None else graph.edge_reconstruction_residual
        )
        if incidence_residual > 1.0e-12 or reconstruction_residual > 1.0e-12:
            raise FloatingPointError("node-owned edge incidence failed to close")

        births = tuple(
            birth_records[node_id] for node_id in sorted(birth_records, key=_id_key)
        )
        if placement_binding is not None:
            expected_node_modes = {
                node_id: (mode, active)
                for node_id, mode, active in placement_binding.node_modes
            }
            for birth in births:
                expected = expected_node_modes[birth.node_id]
                if expected != (birth.mode, birth.active):
                    raise ValueError(
                        "node placement mode/activity disagrees with ribbon history"
                    )
        mode_counts = {
            mode: sum(record.mode == mode for record in births)
            for mode in ("first", "continuous", "restart", "inactive")
        }
        diagnostics = DVMRibbonDiagnostics(
            enabled=True,
            interface_id=INTERFACE_ID,
            ownership_mode=OWNERSHIP_MODE,
            canonical_eligible=False,
            feedback_call_count=0,
            transport_advance_count=len(accepted_frontier_facts),
            source_family=self._source_family,
            source_step_index=source_step_index,
            source_cell_count=len(cell_records),
            active_cell_count=len(rings),
            shared_node_count=len(node_facts),
            first_node_count=mode_counts["first"],
            continuous_node_count=mode_counts["continuous"],
            restart_node_count=mode_counts["restart"],
            inactive_node_count=mode_counts["inactive"],
            max_kelvin_residual_m2_per_s=max_kelvin,
            incidence_residual=incidence_residual,
            edge_reconstruction_residual=reconstruction_residual,
            seam_count=0,
            nonfinite_count=0,
            source_reuse_count=0,
            node_placement_bound=placement_binding is not None,
            node_placement_manifest_sha256=(
                None
                if placement_binding is None
                else placement_binding.producer_manifest_sha256
            ),
            topology_patch_ids=(
                ()
                if placement_binding is None
                else placement_binding.topology_patch_ids
            ),
            coordinate_frame_ids=(
                ()
                if placement_binding is None
                else placement_binding.coordinate_frame_ids
            ),
            patch_binding_passed=placement_binding is not None,
        )
        result = DVMRibbonShadowResult(
            edge_graph=graph,
            node_births=births,
            cell_ledgers=tuple(ledgers),
            diagnostics=diagnostics,
            feedback_velocity=None,
        )
        result_digest = ribbon_parent_digest_sha256(result)

        live_consumptions = tuple(
            (
                event,
                _live_event_consumption_token(
                    event=event,
                    wing_id=self._wing_id,
                    source_family=self._source_family,
                    cell_id=cell_id,
                    left_node_id=left,
                    right_node_id=right,
                    source_step_index=source_step_index,
                    plane_map_manifest_sha256=plane_map_manifest_sha256[cell_id],
                    node_placement_manifest_sha256=(
                        None
                        if placement_binding is None
                        else placement_binding.producer_manifest_sha256
                    ),
                    placement_cell_binding=placement_cells.get(cell_id),
                ),
            )
            for (
                cell_id,
                left,
                right,
                event,
                _origin,
                _x_axis,
                _z_axis,
                _positive_axis,
                _sign,
                _active,
                _gamma_star,
                _birth_2d,
                _source_id,
            ) in cell_records
        )

        # Commit only after every topology, dimensionalization, ledger, result,
        # parent-digest, and module-wide live-consumption gate has succeeded.
        # Global ownership and mapper state share one critical section, making
        # rejected races and all ordinary validation failures cleanly retryable.
        with _LIVE_CONSUMPTION_LOCK:
            _assert_live_cumulative_report_unconsumed_locked(
                candidate_cumulative_frontier_report,
                report_sha256=candidate_frontier_report_sha256,
            )
            _assert_live_consumptions_unconsumed_locked(
                live_consumptions,
                placement_binding=placement_binding,
            )
            if self._commit_version != starting_commit_version:
                raise RuntimeError("ribbon mapper state changed during handoff")
            _attest_direct_ribbon_result(result, result_digest)
            _register_live_consumptions_locked(
                live_consumptions,
                placement_binding=placement_binding,
            )
            _register_live_cumulative_report_locked(
                candidate_cumulative_frontier_report,
                report_sha256=candidate_frontier_report_sha256,
            )
            self._node_state = next_state
            self._cell_bindings = candidate_bindings
            self._cell_history = candidate_history
            self._consumed_source_keys.update(candidate_consumed)
            if candidate_frontier_report_sha256 is not None:
                self._consumed_frontier_report_sha256.add(
                    candidate_frontier_report_sha256
                )
            self._last_source_step = source_step_index
            self._last_ribbon_digest_sha256 = result_digest
            self._commit_version += 1
        return result


__all__ = [
    "DVMCellRibbonLedger",
    "DVMPlaneToGP1Map",
    "DVMRibbonDiagnostics",
    "DVMRibbonShadowResult",
    "DVMSpanCellSource",
    "INTERFACE_ID",
    "NodeFrontierFact",
    "NodeBirthRecord",
    "NodeOwnedDVMRibbonShadow",
    "OWNERSHIP_MODE",
    "SpanNodeKinematics",
    "validate_live_dvm_ribbon_shadow_result",
]
