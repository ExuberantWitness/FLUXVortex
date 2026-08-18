"""Exact-append cumulative rVPM transport on per-edge dyadic panels.

This is an isolated v5h2 producer.  It reuses the frozen v2 particle-cloud and
release-slice schemas so level zero can be bitwise identical, but it owns a new
handoff/report attestation domain and calls only the frozen dyadic edge bridge
for newborn deposition.  The full old+new cloud is advanced in one shared
LSRK3 field.  There is no force, load, feedback, target-data, sorting, welding,
deletion, cancellation, or remeshing path.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from hashlib import sha256
import hmac
import inspect
import json
from numbers import Integral
from pathlib import Path
import re
from typing import Any, Literal, TypeAlias
import weakref

import numpy as np
from numpy.typing import ArrayLike, NDArray

import fluxvortex.rvpm_dyadic_edge_bridge as dyadic_bridge_module
from fluxvortex.rvpm_dyadic_edge_bridge import (
    DyadicBridgeResult,
    DyadicDepositionPlan,
    DyadicEdgePanelLedger,
    DyadicEdgeParticleLedger,
    deposit_edge_graph_prescribed_sigma_dyadic_panels,
    plan_edge_graph_prescribed_sigma_dyadic_panels,
    recompute_dyadic_bridge_sidecar,
    _strict_dyadic_plan,
    validate_dyadic_bridge_result,
    validate_dyadic_deposition_plan,
)
import forward_flight_benchmarks.v5h_cumulative_cloud_transport as v2
from forward_flight_benchmarks.v5h_cumulative_cloud_transport import (
    CumulativeParticleCloud,
    ReleaseSliceLedger,
)
from forward_flight_benchmarks.v5h_passive_frontier_transport import (
    NodeFrontierFact,
    parent_frontier_digest_sha256,
    ribbon_parent_digest_sha256,
)


FloatArray = NDArray[np.float64]
StableId: TypeAlias = int | str
SourceFamily = Literal["lev", "tev_persisted"]

DYADIC_CUMULATIVE_HANDOFF_INTERFACE_ID = (
    "fluxv-v5h2-dyadic-cumulative-ribbon-handoff-v1"
)
DYADIC_CUMULATIVE_REPORT_INTERFACE_ID = "fluxv-v5h2-dyadic-cumulative-cloud-report-v1"
DYADIC_CUMULATIVE_PRODUCER_ID = "v5h2-dyadic-exact-append-one-field-lsrk3-v1"
DYADIC_CUMULATIVE_ATTESTATION_KIND = "same-process-weak-object-identity-v1"
DYADIC_CUMULATIVE_CONTINUATION_SCOPE = (
    "single-wing-single-source-family-dyadic-cumulative-v1"
)
DYADIC_CUMULATIVE_TIME_LAYER = v2.CUMULATIVE_TIME_LAYER
DYADIC_CUMULATIVE_FACT_OWNER = v2.CUMULATIVE_FACT_OWNER
TRANSPORT_BACKEND_ID = v2.TRANSPORT_BACKEND_ID

MAX_TRANSPORT_SUBSTEPS = v2.MAX_TRANSPORT_SUBSTEPS
MAX_PARTICLES_PER_RELEASE = v2.MAX_PARTICLES_PER_RELEASE
MAX_CUMULATIVE_PARTICLES = v2.MAX_CUMULATIVE_PARTICLES
MAX_PARTICLE_COUNT = MAX_CUMULATIVE_PARTICLES

FROZEN_V2_CORE_SHA256 = (
    "8b4c3efb19293952a854308508d5d76d55f95268a0ee03acd916eb13287f3d49"
)

# Development import binding only.  This is replaced by the independently
# audited bridge SHA once that API/source freeze is announced; it prevents
# mid-process mutation without recording the superseded candidate digest.
_DYADIC_BRIDGE_IMPORT_SHA256 = sha256(
    Path(dyadic_bridge_module.__file__).resolve().read_bytes()
).hexdigest()

_FROZEN_DYADIC_DEPOSIT = deposit_edge_graph_prescribed_sigma_dyadic_panels
_FROZEN_DYADIC_PLAN = plan_edge_graph_prescribed_sigma_dyadic_panels
_FROZEN_DYADIC_VALIDATE_RESULT = validate_dyadic_bridge_result
_FROZEN_DYADIC_VALIDATE_PLAN = validate_dyadic_deposition_plan
_FROZEN_DYADIC_RECOMPUTE = recompute_dyadic_bridge_sidecar
_FROZEN_DYADIC_STRICT_PLAN = _strict_dyadic_plan
_FROZEN_V2_LOCK = v2._LOCK
_FROZEN_SHARED_RIBBON_CONSUMPTIONS = v2._RIBBON_CONSUMPTIONS
_FROZEN_SHARED_REGISTER_LIVE = v2._register_live
_FROZEN_V2_HELPERS = {
    name: getattr(v2, name)
    for name in (
        "_array_digest",
        "_canonical",
        "_cloud_arrays",
        "_cloud_digest",
        "_cloud_particle_count_gate",
        "_empty_cloud",
        "_finite_real",
        "_finite_vector",
        "_id_key",
        "_id_payload",
        "_identity_digest",
        "_lineage_payload",
        "_lower_sha256",
        "_max_abs",
        "_nonnegative_integer",
        "_positive_integer",
        "_positive_real",
        "_slice_payload",
        "_stable_id",
    )
}


@dataclass(frozen=True, slots=True, weakref_slot=True)
class DyadicCumulativeRibbonHandoff:
    """Live attestation of one ribbon release for the v5h2 producer."""

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
    previous_report: DyadicCumulativeCloudTransportReport | None = field(
        default=None,
        repr=False,
        compare=False,
    )


@dataclass(frozen=True, slots=True)
class DyadicReleasePlanSidecar:
    """Portable bridge plan and particle-ledger proof for one release."""

    release_index: int
    source_step_index: int
    source_time_s: float
    deposited_cloud_digest_sha256: str
    plan: DyadicDepositionPlan
    particle_ledger: tuple[DyadicEdgeParticleLedger, ...]
    shadow_state_sha256: str
    bridge_manifest_sha256: str
    sidecar_sha256: str


@dataclass(frozen=True, slots=True, weakref_slot=True)
class DyadicCumulativeCloudTransportReport:
    """Live-attested exact append, dyadic plan, and one-field advance."""

    enabled: bool
    interface_id: str
    producer_id: str
    producer_artifact_sha256: str
    transport_backend_id: str
    transport_artifact_sha256: str
    dyadic_bridge_artifact_sha256: str
    v2_core_artifact_sha256: str
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
    base_target_spacing_m: float | None
    refinement_level: int | None
    nominal_target_spacing_m: float | None
    previous_particle_count: int
    new_particle_count: int
    total_particle_count: int
    transported_particle_cloud: CumulativeParticleCloud
    release_plan_sidecars: tuple[DyadicReleasePlanSidecar, ...]
    facts: tuple[NodeFrontierFact, ...]
    report_sha256: str
    attestation_kind: str
    continuation_scope: str
    exact_append_passed: bool
    one_combined_field_passed: bool
    stage_pre_replay_passed: bool
    dyadic_plan_passed: bool
    plan_call_count: int
    deposition_call_count: int
    sidecar_recompute_count: int
    predicted_new_particle_count: int
    lsrk3_call_count: int
    lsrk3_stage_count: int
    stage_pre_field_call_count: int
    combined_stage_particle_counts: tuple[int, ...]
    exact_append_prefix_max_abs: float
    stage_pre_replay_max_abs: float
    transport_trace: tuple[str, ...]
    transport_trace_sha256: str
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


_HANDOFF_REGISTRY: dict[int, tuple[weakref.ReferenceType[object], str]] = {}
_REPORT_REGISTRY: dict[int, tuple[weakref.ReferenceType[object], str]] = {}
_REPORT_PARENT_CONSUMPTIONS: dict[int, tuple[weakref.ReferenceType[object], str]] = {}
_HANDOFF_CONSUMPTIONS: dict[int, tuple[weakref.ReferenceType[object], str]] = {}


def _producer_artifact_sha256() -> str:
    return sha256(Path(__file__).resolve().read_bytes()).hexdigest()


def _source_sha256(value: object) -> str:
    try:
        return sha256(inspect.getsource(value).encode("utf-8")).hexdigest()
    except (OSError, TypeError) as error:
        raise ValueError("frozen callable has no auditable source") from error


_DYADIC_ROOTS = (
    (
        "dyadic_deposit",
        _FROZEN_DYADIC_DEPOSIT,
        dyadic_bridge_module,
        "deposit_edge_graph_prescribed_sigma_dyadic_panels",
        _source_sha256(_FROZEN_DYADIC_DEPOSIT),
    ),
    (
        "dyadic_plan",
        _FROZEN_DYADIC_PLAN,
        dyadic_bridge_module,
        "plan_edge_graph_prescribed_sigma_dyadic_panels",
        _source_sha256(_FROZEN_DYADIC_PLAN),
    ),
    (
        "dyadic_validate_result",
        _FROZEN_DYADIC_VALIDATE_RESULT,
        dyadic_bridge_module,
        "validate_dyadic_bridge_result",
        _source_sha256(_FROZEN_DYADIC_VALIDATE_RESULT),
    ),
    (
        "dyadic_validate_plan",
        _FROZEN_DYADIC_VALIDATE_PLAN,
        dyadic_bridge_module,
        "validate_dyadic_deposition_plan",
        _source_sha256(_FROZEN_DYADIC_VALIDATE_PLAN),
    ),
    (
        "dyadic_recompute",
        _FROZEN_DYADIC_RECOMPUTE,
        dyadic_bridge_module,
        "recompute_dyadic_bridge_sidecar",
        _source_sha256(_FROZEN_DYADIC_RECOMPUTE),
    ),
    (
        "dyadic_strict_plan",
        _FROZEN_DYADIC_STRICT_PLAN,
        dyadic_bridge_module,
        "_strict_dyadic_plan",
        _source_sha256(_FROZEN_DYADIC_STRICT_PLAN),
    ),
)
_DYADIC_TRANSITIVE_ROOTS = tuple(
    (
        label,
        frozen,
        module,
        attribute,
        frozen.__module__,
        frozen.__qualname__,
        source_sha,
    )
    for label, frozen, module, attribute, source_sha in _DYADIC_ROOTS
)
_FROZEN_DYADIC_TRANSITIVE = v2._freeze_transitive_callable_bindings(
    _DYADIC_TRANSITIVE_ROOTS
)


def _assert_frozen_dependencies() -> dict[str, Any]:
    bridge_path = Path(dyadic_bridge_module.__file__).resolve()
    v2_path = Path(v2.__file__).resolve()
    if sha256(bridge_path.read_bytes()).hexdigest() != _DYADIC_BRIDGE_IMPORT_SHA256:
        raise ValueError("dyadic bridge source changed after import")
    if sha256(v2_path.read_bytes()).hexdigest() != FROZEN_V2_CORE_SHA256:
        raise ValueError("frozen v2 cumulative core source changed")
    if (
        v2._LOCK is not _FROZEN_V2_LOCK
        or v2._RIBBON_CONSUMPTIONS is not _FROZEN_SHARED_RIBBON_CONSUMPTIONS
        or v2._register_live is not _FROZEN_SHARED_REGISTER_LIVE
        or _source_sha256(v2._register_live)
        != _source_sha256(_FROZEN_SHARED_REGISTER_LIVE)
    ):
        raise ValueError("shared v2 live-ribbon ownership binding was replaced")
    for name, frozen in _FROZEN_V2_HELPERS.items():
        if getattr(v2, name, None) is not frozen:
            raise ValueError(f"frozen v2 helper {name} was replaced")
    trusted: dict[str, Any] = {}
    for label, frozen, module, attribute, expected_sha in _DYADIC_ROOTS:
        local = globals().get(attribute)
        if getattr(module, attribute, None) is not frozen or local is not frozen:
            raise ValueError(f"{label} callable was replaced")
        if _source_sha256(frozen) != expected_sha:
            raise ValueError(f"{label} callable source changed")
        trusted[label] = frozen
    for (
        root_label,
        owner_globals,
        owner_module,
        attribute,
        frozen,
        expected_module,
        expected_qualname,
        expected_source_sha256,
    ) in _FROZEN_DYADIC_TRANSITIVE:
        current = owner_globals.get(attribute)
        if (
            current is not frozen
            or getattr(frozen, "__module__", None) != expected_module
            or getattr(frozen, "__qualname__", None) != expected_qualname
            or _source_sha256(frozen) != expected_source_sha256
        ):
            raise ValueError(
                f"{root_label} transitive dependency "
                f"{owner_module}.{attribute} was replaced"
            )
    _, _, transport = v2._assert_transport_bindings()
    trusted.update(transport)
    return trusted


def _transport_artifact_sha256() -> str:
    _assert_frozen_dependencies()
    digest = sha256(b"fluxv-v5h2-dyadic-transport-artifact-v1\0")
    for module in (dyadic_bridge_module, v2):
        digest.update(Path(module.__file__).resolve().read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _lower_sha256(name: str, value: object) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _level(value: object) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError("refinement_level must be an integer in [0, 3]")
    result = int(value)
    if result not in (0, 1, 2, 3):
        raise ValueError("refinement_level must be preregistered in [0, 3]")
    return result


def _trace_sha256(trace: tuple[str, ...]) -> str:
    if not isinstance(trace, tuple) or any(
        not isinstance(item, str) or not item for item in trace
    ):
        raise ValueError("transport trace must contain explicit string events")
    digest = sha256(b"fluxv-v5h2-dyadic-cumulative-transport-trace-v1\0")
    for item in trace:
        digest.update(item.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _register_live(registry: dict[int, Any], value: object, digest: str) -> None:
    _FROZEN_SHARED_REGISTER_LIVE(registry, value, digest)


def _live_registry_entry(registry: dict[int, Any], value: object) -> Any | None:
    entry = registry.get(id(value))
    if entry is None:
        return None
    live = entry[0]()
    if live is None:
        registry.pop(id(value), None)
        return None
    if live is not value:
        raise RuntimeError("live identity registry collision")
    return entry


def _panel_payload(panel: DyadicEdgePanelLedger) -> dict[str, object]:
    return {
        "edge_key": v2._canonical(panel.edge_key),
        "edge_length_hex": v2._finite_real("edge length", panel.edge_length_m).hex(),
        "base_panel_count": v2._positive_integer(
            "base panel count", panel.base_panel_count
        ),
        "refinement_level": _level(panel.refinement_level),
        "panel_count": v2._positive_integer("panel count", panel.panel_count),
        "realized_spacing_hex": v2._positive_real(
            "realized spacing", panel.realized_spacing_m
        ).hex(),
        "smoothing_radius_hex": v2._positive_real(
            "panel smoothing radius", panel.smoothing_radius_m
        ).hex(),
        "parent_edge_graph_sha256": _lower_sha256(
            "panel parent graph", panel.parent_edge_graph_sha256
        ),
    }


def _particle_ledger_payload(
    item: DyadicEdgeParticleLedger,
) -> dict[str, object]:
    return {
        "edge_key": v2._canonical(item.edge_key),
        "particle_slice_start": v2._nonnegative_integer(
            "particle slice start", item.particle_slice_start
        ),
        "particle_slice_stop": v2._nonnegative_integer(
            "particle slice stop", item.particle_slice_stop
        ),
        "particle_ids_sha256": _lower_sha256(
            "edge particle IDs", item.particle_ids_sha256
        ),
        "lineage_sha256": _lower_sha256("edge lineage", item.lineage_sha256),
        "parent_edge_graph_sha256": _lower_sha256(
            "edge parent graph", item.parent_edge_graph_sha256
        ),
    }


def _sidecar_payload(item: DyadicReleasePlanSidecar) -> dict[str, object]:
    plan = item.plan
    return {
        "release_index": v2._positive_integer(
            "sidecar release index", item.release_index
        ),
        "source_step_index": v2._positive_integer(
            "sidecar source step", item.source_step_index
        ),
        "source_time_hex": v2._finite_real(
            "sidecar source time", item.source_time_s
        ).hex(),
        "deposited_cloud_digest_sha256": _lower_sha256(
            "sidecar deposited cloud", item.deposited_cloud_digest_sha256
        ),
        "plan": {
            "interface_id": plan.interface_id,
            "producer_id": plan.producer_id,
            "producer_artifact_sha256": plan.producer_artifact_sha256,
            "edge_bridge_artifact_sha256": plan.edge_bridge_artifact_sha256,
            "parent_attestation_kind": plan.parent_attestation_kind,
            "parent_edge_graph_sha256": plan.parent_edge_graph_sha256,
            "smoothing_radius_hex": float(plan.smoothing_radius_m).hex(),
            "base_target_spacing_hex": float(plan.base_target_spacing_m).hex(),
            "refinement_level": plan.refinement_level,
            "refinement_multiplier": plan.refinement_multiplier,
            "step": plan.step,
            "predicted_total_particle_count": plan.predicted_total_particle_count,
            "particle_schema": plan.particle_schema,
            "physical_owner": plan.physical_owner,
            "owner_state": plan.owner_state,
            "edge_panels": [_panel_payload(panel) for panel in plan.edge_panels],
            "plan_sha256": plan.plan_sha256,
        },
        "particle_ledger": [
            _particle_ledger_payload(row) for row in item.particle_ledger
        ],
        "shadow_state_sha256": _lower_sha256(
            "sidecar shadow state", item.shadow_state_sha256
        ),
        "bridge_manifest_sha256": _lower_sha256(
            "sidecar bridge manifest", item.bridge_manifest_sha256
        ),
    }


def _sidecar_digest(item: DyadicReleasePlanSidecar) -> str:
    return sha256(
        b"fluxv-v5h2-dyadic-release-plan-sidecar-v1\0"
        + json.dumps(
            _sidecar_payload(item),
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _handoff_payload(
    handoff: DyadicCumulativeRibbonHandoff,
) -> dict[str, object]:
    return {
        "interface_id": handoff.interface_id,
        "producer_id": handoff.producer_id,
        "producer_artifact_sha256": handoff.producer_artifact_sha256,
        "attestation_kind": handoff.attestation_kind,
        "wing_id": v2._id_payload(v2._stable_id("handoff.wing_id", handoff.wing_id)),
        "source_family": handoff.source_family,
        "source_step_index": v2._positive_integer(
            "handoff.source_step_index", handoff.source_step_index
        ),
        "source_time_hex": v2._finite_real(
            "handoff.source_time_s", handoff.source_time_s
        ).hex(),
        "current_ribbon_digest_sha256": _lower_sha256(
            "handoff.current_ribbon_digest_sha256",
            handoff.current_ribbon_digest_sha256,
        ),
        "previous_report_sha256": handoff.previous_report_sha256,
        "previous_cloud_sha256": handoff.previous_cloud_sha256,
        "mapper_commit_version": v2._positive_integer(
            "handoff.mapper_commit_version", handoff.mapper_commit_version
        ),
        "active_birth_count": v2._nonnegative_integer(
            "handoff.active_birth_count", handoff.active_birth_count
        ),
        "continuous_birth_count": v2._nonnegative_integer(
            "handoff.continuous_birth_count", handoff.continuous_birth_count
        ),
        "restart_birth_count": v2._nonnegative_integer(
            "handoff.restart_birth_count", handoff.restart_birth_count
        ),
    }


def _handoff_digest(handoff: DyadicCumulativeRibbonHandoff) -> str:
    return sha256(
        b"fluxv-v5h2-dyadic-cumulative-handoff-v1\0"
        + json.dumps(
            _handoff_payload(handoff),
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _fact_payload(fact: NodeFrontierFact) -> dict[str, object]:
    return {
        "wing_id": v2._id_payload(v2._stable_id("fact.wing_id", fact.wing_id)),
        "node_id": v2._id_payload(v2._stable_id("fact.node_id", fact.node_id)),
        "source_family": fact.source_family,
        "lineage_epoch": v2._nonnegative_integer(
            "fact.lineage_epoch", fact.lineage_epoch
        ),
        "parent_frontier_id": fact.parent_frontier_id,
        "parent_frontier_digest_sha256": fact.parent_frontier_digest_sha256,
        "parent_ribbon_digest_sha256": fact.parent_ribbon_digest_sha256,
        "parent_birth_step_index": v2._positive_integer(
            "fact.parent_birth_step_index", fact.parent_birth_step_index
        ),
        "for_source_step_index": v2._positive_integer(
            "fact.for_source_step_index", fact.for_source_step_index
        ),
        "time_layer": fact.time_layer,
        "transport_backend_id": fact.transport_backend_id,
        "transport_artifact_sha256": fact.transport_artifact_sha256,
        "producer_id": fact.producer_id,
        "producer_artifact_sha256": fact.producer_artifact_sha256,
        # producer_report_sha256 is bound after this acyclic report digest.
        "transport_start_time_hex": v2._finite_real(
            "fact.transport_start_time_s", fact.transport_start_time_s
        ).hex(),
        "transport_end_time_hex": v2._finite_real(
            "fact.transport_end_time_s", fact.transport_end_time_s
        ).hex(),
        "transport_substeps": v2._positive_integer(
            "fact.transport_substeps", fact.transport_substeps
        ),
        "advected_position_hex": [
            float(value).hex()
            for value in v2._finite_vector(
                "fact.advected_position_gp1_m", fact.advected_position_gp1_m
            )
        ],
        "fact_owner": fact.fact_owner,
    }


def _report_payload(
    report: DyadicCumulativeCloudTransportReport,
) -> dict[str, object]:
    return {
        "enabled": report.enabled,
        "interface_id": report.interface_id,
        "producer_id": report.producer_id,
        "producer_artifact_sha256": report.producer_artifact_sha256,
        "transport_backend_id": report.transport_backend_id,
        "transport_artifact_sha256": report.transport_artifact_sha256,
        "dyadic_bridge_artifact_sha256": report.dyadic_bridge_artifact_sha256,
        "v2_core_artifact_sha256": report.v2_core_artifact_sha256,
        "wing_id": (None if report.wing_id is None else v2._id_payload(report.wing_id)),
        "source_family": report.source_family,
        "parent_source_step_index": report.parent_source_step_index,
        "for_source_step_index": report.for_source_step_index,
        "time_layer": report.time_layer,
        "transport_start_time_hex": (
            None
            if report.transport_start_time_s is None
            else v2._finite_real(
                "report.transport_start_time_s", report.transport_start_time_s
            ).hex()
        ),
        "transport_end_time_hex": (
            None
            if report.transport_end_time_s is None
            else v2._finite_real(
                "report.transport_end_time_s", report.transport_end_time_s
            ).hex()
        ),
        "transport_substeps": report.transport_substeps,
        "freestream_hex": [
            float(value).hex()
            for value in v2._finite_vector(
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
            else v2._positive_real(
                "report.smoothing_radius_m", report.smoothing_radius_m
            ).hex()
        ),
        "base_target_spacing_hex": (
            None
            if report.base_target_spacing_m is None
            else v2._positive_real(
                "report.base_target_spacing_m", report.base_target_spacing_m
            ).hex()
        ),
        "refinement_level": report.refinement_level,
        "nominal_target_spacing_hex": (
            None
            if report.nominal_target_spacing_m is None
            else v2._positive_real(
                "report.nominal_target_spacing_m",
                report.nominal_target_spacing_m,
            ).hex()
        ),
        "previous_particle_count": report.previous_particle_count,
        "new_particle_count": report.new_particle_count,
        "total_particle_count": report.total_particle_count,
        "cloud_sha256": report.transported_particle_cloud.cloud_sha256,
        "release_plan_sidecars": [
            {**_sidecar_payload(item), "sidecar_sha256": item.sidecar_sha256}
            for item in report.release_plan_sidecars
        ],
        "facts": [_fact_payload(fact) for fact in report.facts],
        "attestation_kind": report.attestation_kind,
        "continuation_scope": report.continuation_scope,
        "exact_append_passed": report.exact_append_passed,
        "one_combined_field_passed": report.one_combined_field_passed,
        "stage_pre_replay_passed": report.stage_pre_replay_passed,
        "dyadic_plan_passed": report.dyadic_plan_passed,
        "plan_call_count": report.plan_call_count,
        "deposition_call_count": report.deposition_call_count,
        "sidecar_recompute_count": report.sidecar_recompute_count,
        "predicted_new_particle_count": report.predicted_new_particle_count,
        "lsrk3_call_count": report.lsrk3_call_count,
        "lsrk3_stage_count": report.lsrk3_stage_count,
        "stage_pre_field_call_count": report.stage_pre_field_call_count,
        "combined_stage_particle_counts": list(report.combined_stage_particle_counts),
        "exact_append_prefix_max_abs_hex": v2._finite_real(
            "report.exact_append_prefix_max_abs",
            report.exact_append_prefix_max_abs,
        ).hex(),
        "stage_pre_replay_max_abs_hex": v2._finite_real(
            "report.stage_pre_replay_max_abs", report.stage_pre_replay_max_abs
        ).hex(),
        "transport_trace": list(report.transport_trace),
        "transport_trace_sha256": report.transport_trace_sha256,
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
    }


def _report_digest(report: DyadicCumulativeCloudTransportReport) -> str:
    return sha256(
        b"fluxv-v5h2-dyadic-cumulative-report-v1\0"
        + json.dumps(
            _report_payload(report),
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _validate_handoff(handoff: object) -> DyadicCumulativeRibbonHandoff:
    _assert_frozen_dependencies()
    if type(handoff) is not DyadicCumulativeRibbonHandoff:
        raise ValueError("handoff must be a directly produced live dyadic handoff")
    if (
        handoff.interface_id != DYADIC_CUMULATIVE_HANDOFF_INTERFACE_ID
        or handoff.producer_id != DYADIC_CUMULATIVE_PRODUCER_ID
        or handoff.producer_artifact_sha256 != _producer_artifact_sha256()
        or handoff.attestation_kind != DYADIC_CUMULATIVE_ATTESTATION_KIND
    ):
        raise ValueError("dyadic cumulative handoff producer is stale or foreign")
    v2._stable_id("handoff.wing_id", handoff.wing_id)
    if handoff.source_family not in ("lev", "tev_persisted"):
        raise ValueError("dyadic cumulative handoff source family is invalid")
    v2._positive_integer("handoff.source_step_index", handoff.source_step_index)
    v2._finite_real("handoff.source_time_s", handoff.source_time_s)
    v2._positive_integer("handoff.mapper_commit_version", handoff.mapper_commit_version)
    active = v2._nonnegative_integer(
        "handoff.active_birth_count", handoff.active_birth_count
    )
    continuous = v2._nonnegative_integer(
        "handoff.continuous_birth_count", handoff.continuous_birth_count
    )
    restart = v2._nonnegative_integer(
        "handoff.restart_birth_count", handoff.restart_birth_count
    )
    if continuous + restart > active:
        raise ValueError("dyadic cumulative handoff birth counts are inconsistent")
    _lower_sha256(
        "handoff.current_ribbon_digest_sha256",
        handoff.current_ribbon_digest_sha256,
    )
    if handoff.previous_report_sha256 is None:
        if (
            handoff.previous_cloud_sha256 is not None
            or handoff.previous_report is not None
        ):
            raise ValueError("dyadic handoff has a phantom previous report/cloud")
    else:
        _lower_sha256("handoff.previous_report_sha256", handoff.previous_report_sha256)
        _lower_sha256("handoff.previous_cloud_sha256", handoff.previous_cloud_sha256)
        if type(handoff.previous_report) is not DyadicCumulativeCloudTransportReport:
            raise ValueError("dyadic handoff parent report has a foreign schema")
    digest = _handoff_digest(handoff)
    if not hmac.compare_digest(digest, handoff.handoff_sha256):
        raise ValueError("dyadic cumulative handoff digest is inconsistent")
    registered = _live_registry_entry(_HANDOFF_REGISTRY, handoff)
    if registered is None or not hmac.compare_digest(registered[1], digest):
        raise ValueError("handoff is not a directly produced live dyadic object")
    return handoff


def attest_dyadic_cumulative_ribbon_handoff(
    mapper: object,
    current_ribbon: object,
    *,
    wing_id: StableId,
    source_time_s: object,
    previous_report: DyadicCumulativeCloudTransportReport | None = None,
) -> DyadicCumulativeRibbonHandoff:
    """Bind one live ribbon and optional dyadic parent cloud exactly once."""

    _assert_frozen_dependencies()
    from forward_flight_benchmarks.v5h_dvm_node_ribbon import (
        NodeOwnedDVMRibbonShadow,
        validate_live_dvm_ribbon_shadow_result,
    )

    if not isinstance(mapper, NodeOwnedDVMRibbonShadow):
        raise ValueError("mapper must be a NodeOwnedDVMRibbonShadow")
    ribbon = validate_live_dvm_ribbon_shadow_result(current_ribbon)
    wing = v2._stable_id("wing_id", wing_id)
    if v2._id_key(wing) != v2._id_key(getattr(mapper, "_wing_id")):
        raise ValueError("dyadic cumulative handoff is bound to another wing")
    diagnostics = ribbon.diagnostics
    if not diagnostics.enabled or ribbon.feedback_velocity is not None:
        raise ValueError("dyadic cumulative handoff requires a clean enabled ribbon")
    if diagnostics.feedback_call_count:
        raise ValueError("dyadic cumulative handoff cannot consume feedback")
    family = diagnostics.source_family
    if family != getattr(mapper, "_source_family"):
        raise ValueError("dyadic cumulative handoff family disagrees with mapper")
    step = v2._positive_integer("source_step_index", diagnostics.source_step_index)
    source_time = v2._finite_real("source_time_s", source_time_s)
    current_digest = ribbon_parent_digest_sha256(ribbon)
    last_digest, _ = mapper.transport_handoff_snapshot
    if current_digest != last_digest:
        raise ValueError("current ribbon is stale for this mapper")
    commit_version = v2._positive_integer(
        "mapper commit version", getattr(mapper, "_commit_version")
    )
    active_births = tuple(birth for birth in ribbon.node_births if birth.active)
    continuous_count = sum(birth.mode == "continuous" for birth in active_births)
    restart_count = sum(birth.mode == "restart" for birth in active_births)
    previous_digest: str | None = None
    previous_cloud_digest: str | None = None
    if previous_report is None:
        if step != 1 or continuous_count:
            raise ValueError("first dyadic handoff must be a non-continuous step one")
    else:
        parent = validate_dyadic_cumulative_cloud_transport_report(previous_report)
        if not parent.enabled:
            raise ValueError("a disabled report cannot parent a dyadic handoff")
        if (
            parent.wing_id != wing
            or parent.source_family != family
            or parent.for_source_step_index != step
            or parent.transport_end_time_s != source_time
        ):
            raise ValueError("dyadic parent report is stale or foreign")
        previous_digest = parent.report_sha256
        previous_cloud_digest = parent.transported_particle_cloud.cloud_sha256

    placeholder = DyadicCumulativeRibbonHandoff(
        interface_id=DYADIC_CUMULATIVE_HANDOFF_INTERFACE_ID,
        producer_id=DYADIC_CUMULATIVE_PRODUCER_ID,
        producer_artifact_sha256=_producer_artifact_sha256(),
        attestation_kind=DYADIC_CUMULATIVE_ATTESTATION_KIND,
        wing_id=wing,
        source_family=family,
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
    with _FROZEN_V2_LOCK:
        if getattr(mapper, "_commit_version") != commit_version:
            raise RuntimeError("ribbon mapper changed during dyadic attestation")
        consumed = _live_registry_entry(_FROZEN_SHARED_RIBBON_CONSUMPTIONS, ribbon)
        if consumed is not None:
            raise ValueError("live ribbon was already consumed by a cumulative handoff")
        if previous_report is not None:
            parent_consumed = _live_registry_entry(
                _REPORT_PARENT_CONSUMPTIONS, previous_report
            )
            if parent_consumed is not None:
                raise ValueError("dyadic parent report was already consumed")
        _register_live(_HANDOFF_REGISTRY, handoff, handoff.handoff_sha256)
        _register_live(
            _FROZEN_SHARED_RIBBON_CONSUMPTIONS, ribbon, handoff.handoff_sha256
        )
        if previous_report is not None:
            _register_live(
                _REPORT_PARENT_CONSUMPTIONS,
                previous_report,
                handoff.handoff_sha256,
            )
    return _validate_handoff(handoff)


def _empty_cloud() -> CumulativeParticleCloud:
    return v2._empty_cloud()


def _disabled_report() -> DyadicCumulativeCloudTransportReport:
    cloud = _empty_cloud()
    trace = ("disabled:input-blind",)
    placeholder = DyadicCumulativeCloudTransportReport(
        enabled=False,
        interface_id=DYADIC_CUMULATIVE_REPORT_INTERFACE_ID,
        producer_id=DYADIC_CUMULATIVE_PRODUCER_ID,
        producer_artifact_sha256=_producer_artifact_sha256(),
        transport_backend_id=TRANSPORT_BACKEND_ID,
        transport_artifact_sha256=_transport_artifact_sha256(),
        dyadic_bridge_artifact_sha256=_DYADIC_BRIDGE_IMPORT_SHA256,
        v2_core_artifact_sha256=FROZEN_V2_CORE_SHA256,
        wing_id=None,
        source_family=None,
        parent_source_step_index=None,
        for_source_step_index=None,
        time_layer=DYADIC_CUMULATIVE_TIME_LAYER,
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
        base_target_spacing_m=None,
        refinement_level=None,
        nominal_target_spacing_m=None,
        previous_particle_count=0,
        new_particle_count=0,
        total_particle_count=0,
        transported_particle_cloud=cloud,
        release_plan_sidecars=(),
        facts=(),
        report_sha256="0" * 64,
        attestation_kind=DYADIC_CUMULATIVE_ATTESTATION_KIND,
        continuation_scope=DYADIC_CUMULATIVE_CONTINUATION_SCOPE,
        exact_append_passed=False,
        one_combined_field_passed=False,
        stage_pre_replay_passed=False,
        dyadic_plan_passed=False,
        plan_call_count=0,
        deposition_call_count=0,
        sidecar_recompute_count=0,
        predicted_new_particle_count=0,
        lsrk3_call_count=0,
        lsrk3_stage_count=0,
        stage_pre_field_call_count=0,
        combined_stage_particle_counts=(),
        exact_append_prefix_max_abs=0.0,
        stage_pre_replay_max_abs=0.0,
        transport_trace=trace,
        transport_trace_sha256=_trace_sha256(trace),
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
    with _FROZEN_V2_LOCK:
        _register_live(_REPORT_REGISTRY, report, report.report_sha256)
    return report


def _release_sidecar(
    result: DyadicBridgeResult,
    *,
    release_index: int,
    source_step_index: int,
    source_time_s: float,
    deposited_cloud_digest_sha256: str,
) -> DyadicReleasePlanSidecar:
    placeholder = DyadicReleasePlanSidecar(
        release_index=release_index,
        source_step_index=source_step_index,
        source_time_s=source_time_s,
        deposited_cloud_digest_sha256=deposited_cloud_digest_sha256,
        plan=result.plan,
        particle_ledger=result.particle_ledger,
        shadow_state_sha256=result.shadow_state_sha256,
        bridge_manifest_sha256=result.manifest_sha256,
        sidecar_sha256="0" * 64,
    )
    return replace(placeholder, sidecar_sha256=_sidecar_digest(placeholder))


def transport_dyadic_accumulated_particle_cloud(
    handoff: object,
    *,
    smoothing_radius_m: object,
    base_target_spacing_m: object,
    refinement_level: object,
    transport_end_time_s: object,
    transport_substeps: object,
    freestream_velocity_gp1_m_per_s: ArrayLike,
    enabled: bool = True,
) -> DyadicCumulativeCloudTransportReport:
    """Append one dyadic release and advance the complete cloud once."""

    if not isinstance(enabled, (bool, np.bool_)):
        raise ValueError("enabled must be Boolean")
    if not bool(enabled):
        return _disabled_report()
    trusted = _assert_frozen_dependencies()
    validated = _validate_handoff(handoff)
    if _live_registry_entry(_HANDOFF_CONSUMPTIONS, validated) is not None:
        raise ValueError("dyadic cumulative handoff was already consumed")
    sigma_birth = v2._positive_real("smoothing_radius_m", smoothing_radius_m)
    base_spacing = v2._positive_real("base_target_spacing_m", base_target_spacing_m)
    level = _level(refinement_level)
    nominal_spacing = base_spacing / (2**level)
    if not np.isfinite(nominal_spacing) or nominal_spacing <= 0.0:
        raise ValueError("nominal dyadic target spacing is invalid")
    end_time = v2._finite_real("transport_end_time_s", transport_end_time_s)
    if end_time <= validated.source_time_s:
        raise ValueError("transport interval must be positive")
    substeps = v2._positive_integer("transport_substeps", transport_substeps)
    if substeps > MAX_TRANSPORT_SUBSTEPS:
        raise ValueError("transport_substeps exceeds the preregistered cap")
    freestream = v2._finite_vector(
        "freestream_velocity_gp1_m_per_s", freestream_velocity_gp1_m_per_s
    )

    ribbon = validated.current_ribbon
    graph = ribbon.edge_graph
    predicted_plan: DyadicDepositionPlan | None = None
    predicted_new_count = 0
    if graph is not None:
        predicted_plan = trusted["dyadic_plan"](
            graph,
            smoothing_radius=sigma_birth,
            base_target_spacing=base_spacing,
            refinement_level=level,
            step=validated.source_step_index,
        )
        trusted["dyadic_validate_plan"](predicted_plan, graph)
        predicted_new_count = predicted_plan.predicted_total_particle_count
        if predicted_new_count > MAX_PARTICLES_PER_RELEASE:
            raise ValueError("dyadic release exceeds the cumulative release cap")

    old_positions = np.empty((0, 3), dtype=np.float64)
    old_gamma = np.empty((0, 3), dtype=np.float64)
    old_sigma = np.empty((0,), dtype=np.float64)
    old_ids: tuple[tuple[Any, ...], ...] = ()
    old_lineage: tuple[Any, ...] = ()
    old_slices: tuple[ReleaseSliceLedger, ...] = ()
    old_sidecars: tuple[DyadicReleasePlanSidecar, ...] = ()
    parent_cloud_digest: str | None = None
    if validated.previous_report is not None:
        parent = validate_dyadic_cumulative_cloud_transport_report(
            validated.previous_report
        )
        parent_cloud = parent.transported_particle_cloud
        old_positions, old_gamma, old_sigma = v2._cloud_arrays(
            parent_cloud, allow_empty=False
        )
        old_ids = parent_cloud.particle_ids
        old_lineage = parent_cloud.lineage
        old_slices = parent_cloud.release_slices
        old_sidecars = parent.release_plan_sidecars
        parent_cloud_digest = parent_cloud.cloud_sha256
        if (
            parent.smoothing_radius_m != sigma_birth
            or parent.base_target_spacing_m != base_spacing
            or parent.refinement_level != level
        ):
            raise ValueError(
                "dyadic core, base spacing, or level changed across releases"
            )
    if len(old_ids) + predicted_new_count > MAX_CUMULATIVE_PARTICLES:
        raise ValueError("predicted dyadic cumulative count exceeds resource cap")

    trace_events: list[str] = []
    new_positions = np.empty((0, 3), dtype=np.float64)
    new_gamma = np.empty((0, 3), dtype=np.float64)
    new_sigma = np.empty((0,), dtype=np.float64)
    new_ids: tuple[tuple[Any, ...], ...] = ()
    new_lineage: tuple[Any, ...] = ()
    deposited_digest: str | None = None
    release_slices = old_slices
    release_sidecars = old_sidecars
    if graph is not None:
        trace_events.append(
            "dyadic_plan:"
            f"step={validated.source_step_index}:level={level}:"
            f"predicted={predicted_new_count}"
        )
        deposited = trusted["dyadic_deposit"](
            graph,
            smoothing_radius=sigma_birth,
            base_target_spacing=base_spacing,
            refinement_level=level,
            step=validated.source_step_index,
        )
        trusted["dyadic_validate_result"](deposited)
        recomputed = trusted["dyadic_recompute"](deposited)
        if (
            recomputed["plan"] != deposited.plan
            or recomputed["particle_ledger"] != deposited.particle_ledger
            or recomputed["shadow_state_sha256"] != deposited.shadow_state_sha256
            or recomputed["manifest_sha256"] != deposited.manifest_sha256
            or deposited.plan != predicted_plan
        ):
            raise RuntimeError("dyadic deposition disagrees with preflight or sidecar")
        trace_events.append(
            "dyadic_deposit:"
            f"step={validated.source_step_index}:level={level}:"
            f"particles={predicted_new_count}"
        )
        trace_events.append(
            "dyadic_sidecar_recompute:"
            f"step={validated.source_step_index}:manifest={deposited.manifest_sha256}"
        )
        new_positions = deposited.positions
        new_gamma = deposited.gamma
        new_sigma = deposited.sigma
        new_ids = tuple(deposited.particle_ids)
        new_lineage = tuple(deposited.lineage)
        if not new_ids or len(new_ids) != predicted_new_count:
            raise RuntimeError("dyadic deposition count disagrees with preflight")
        deposited_digest = v2._array_digest(
            new_positions,
            new_gamma,
            new_sigma,
            new_ids,
            new_lineage,
            domain=b"fluxv-v5h-cumulative-new-release-v2",
        )
        identity_digest = v2._identity_digest(new_ids, new_lineage)
        start_index = len(old_ids)
        release_slices = old_slices + (
            ReleaseSliceLedger(
                release_index=len(old_slices) + 1,
                source_step_index=validated.source_step_index,
                source_time_s=validated.source_time_s,
                start_index=start_index,
                stop_index=start_index + len(new_ids),
                particle_count=len(new_ids),
                parent_ribbon_digest_sha256=validated.current_ribbon_digest_sha256,
                deposited_cloud_digest_sha256=deposited_digest,
                smoothing_radius_m=sigma_birth,
                deposition_target_spacing_m=nominal_spacing,
                particle_ids_sha256=identity_digest,
                lineage_sha256=identity_digest,
            ),
        )
        release_sidecars = old_sidecars + (
            _release_sidecar(
                deposited,
                release_index=len(old_sidecars) + 1,
                source_step_index=validated.source_step_index,
                source_time_s=validated.source_time_s,
                deposited_cloud_digest_sha256=deposited_digest,
            ),
        )
    if not old_ids and not new_ids:
        raise ValueError("enabled dyadic cumulative transport needs a particle cloud")

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
    append_equal = (
        np.array_equal(positions[:old_count], old_positions)
        and np.array_equal(gamma[:old_count], old_gamma)
        and np.array_equal(sigma[:old_count], old_sigma)
        and np.array_equal(positions[old_count:], new_positions)
        and np.array_equal(gamma[old_count:], new_gamma)
        and np.array_equal(sigma[old_count:], new_sigma)
        and particle_ids[:old_count] == old_ids
        and lineage[:old_count] == old_lineage
        and particle_ids[old_count:] == new_ids
        and lineage[old_count:] == new_lineage
    )
    append_residual = v2._max_abs(
        positions[:old_count] - old_positions,
        gamma[:old_count] - old_gamma,
        sigma[:old_count] - old_sigma,
        positions[old_count:] - new_positions,
        gamma[old_count:] - new_gamma,
        sigma[old_count:] - new_sigma,
    )
    if not append_equal or append_residual != 0.0:
        raise RuntimeError("dyadic cumulative exact append changed a prefix/suffix")
    if len(set(particle_ids)) != len(particle_ids):
        raise ValueError("dyadic exact append produced duplicate particle IDs")
    appended_digest = v2._array_digest(
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
        f"max_abs={append_residual.hex()}"
    )

    births = tuple(
        sorted(
            (birth for birth in ribbon.node_births if birth.active),
            key=lambda birth: v2._id_key(v2._stable_id("birth.node_id", birth.node_id)),
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
        seeds.append(v2._finite_vector("birth position", birth.birth_position_gp1_m))
    tracers = (
        np.ascontiguousarray(np.vstack(seeds), dtype=np.float64)
        if seeds
        else np.empty((0, 3), dtype=np.float64)
    )
    state_callable = trusted["particle_state"]
    lsrk3_callable = trusted["lsrk3"]
    field_callable = trusted["direct_field"]
    initial_state = state_callable(positions, gamma, sigma)
    initial_tracers = tracers.copy()
    state = state_callable(
        initial_state.positions, initial_state.gamma, initial_state.sigma
    )
    delta_time = (end_time - validated.source_time_s) / substeps
    combined_counts: list[int] = []
    for substep_index in range(substeps):
        trace_events.append(
            f"combined_lsrk3_call:substep={substep_index + 1}:"
            f"particles={len(particle_ids)}"
        )
        next_state, stages = lsrk3_callable(
            state, delta_time, freestream_velocity=freestream
        )
        if type(stages) is not tuple or len(stages) != 3:
            raise RuntimeError("LSRK3 backend did not return three frozen stages")
        storage = np.zeros_like(tracers)
        for expected_stage, stage in enumerate(stages, start=1):
            count = int(stage.pre.positions.shape[0])
            if stage.stage != expected_stage or count != len(particle_ids):
                raise RuntimeError("LSRK3 stage does not own the complete cloud")
            combined_counts.append(count)
            trace_events.append(
                f"combined_lsrk3_stage:substep={substep_index + 1}:"
                f"stage={expected_stage}:particles={count}"
            )
            field_value = field_callable(
                stage.pre.positions,
                stage.pre.gamma,
                stage.pre.sigma,
                target_positions=tracers,
            )
            trace_events.append(
                f"frontier_stage_pre_field:substep={substep_index + 1}:"
                f"stage={expected_stage}:particles={count}:frontiers={len(tracers)}"
            )
            storage = stage.a * storage + delta_time * (
                field_value.velocity + freestream[None, :]
            )
            tracers = tracers + stage.b * storage
        final = stages[-1].post
        if not (
            np.array_equal(next_state.positions, final.positions)
            and np.array_equal(next_state.gamma, final.gamma)
            and np.array_equal(next_state.sigma, final.sigma)
            and np.all(np.isfinite(tracers))
        ):
            raise RuntimeError("LSRK3 state/final stage or frontier is invalid")
        state = next_state

    replay_state = state_callable(
        initial_state.positions, initial_state.gamma, initial_state.sigma
    )
    replay_tracers = initial_tracers.copy()
    for _ in range(substeps):
        replay_next, replay_stages = lsrk3_callable(
            replay_state, delta_time, freestream_velocity=freestream
        )
        if type(replay_stages) is not tuple or len(replay_stages) != 3:
            raise RuntimeError("independent replay returned a foreign stage trace")
        storage = np.zeros_like(replay_tracers)
        for expected_stage, stage in enumerate(replay_stages, start=1):
            if stage.stage != expected_stage or stage.pre.positions.shape[0] != len(
                particle_ids
            ):
                raise RuntimeError("independent replay stage ownership is invalid")
            field_value = field_callable(
                stage.pre.positions,
                stage.pre.gamma,
                stage.pre.sigma,
                target_positions=replay_tracers,
            )
            storage = stage.a * storage + delta_time * (
                field_value.velocity + freestream[None, :]
            )
            replay_tracers = replay_tracers + stage.b * storage
        replay_state = replay_next
    replay_residual = v2._max_abs(
        state.positions - replay_state.positions,
        state.gamma - replay_state.gamma,
        state.sigma - replay_state.sigma,
        tracers - replay_tracers,
    )
    if not np.isfinite(replay_residual):
        raise FloatingPointError("dyadic independent replay is non-finite")
    trace_events.append(f"independent_stage_pre_replay:max_abs={replay_residual.hex()}")
    trace = tuple(trace_events)
    counts = {
        "plan": sum(event.startswith("dyadic_plan:") for event in trace),
        "deposit": sum(event.startswith("dyadic_deposit:") for event in trace),
        "sidecar": sum(
            event.startswith("dyadic_sidecar_recompute:") for event in trace
        ),
        "call": sum(event.startswith("combined_lsrk3_call:") for event in trace),
        "stage": sum(event.startswith("combined_lsrk3_stage:") for event in trace),
        "field": sum(event.startswith("frontier_stage_pre_field:") for event in trace),
    }
    exact_gate = append_equal and append_residual == 0.0
    combined_gate = (
        counts["call"] == substeps
        and counts["stage"] == 3 * substeps
        and len(combined_counts) == 3 * substeps
        and all(value == len(particle_ids) for value in combined_counts)
        and not any("split" in event for event in trace)
    )
    replay_gate = counts["field"] == 3 * substeps and replay_residual == 0.0
    plan_gate = (
        counts["plan"] == (1 if new_ids else 0)
        and counts["deposit"] == (1 if new_ids else 0)
        and counts["sidecar"] == (1 if new_ids else 0)
        and len(release_slices) == len(release_sidecars)
    )
    if not (exact_gate and combined_gate and replay_gate and plan_gate):
        raise RuntimeError("dyadic cumulative mechanical replay gate failed")

    cloud_digest = v2._cloud_digest(
        state.positions,
        state.gamma,
        state.sigma,
        particle_ids,
        lineage,
        release_slices,
    )
    cloud = CumulativeParticleCloud(
        interface_id=v2.CUMULATIVE_CLOUD_INTERFACE_ID,
        positions_gp1_m=tuple(tuple(float(x) for x in row) for row in state.positions),
        gamma_vector_m3_per_s=tuple(
            tuple(float(x) for x in row) for row in state.gamma
        ),
        sigma_m=tuple(float(x) for x in state.sigma),
        particle_ids=particle_ids,
        lineage=lineage,
        release_slices=release_slices,
        cloud_sha256=cloud_digest,
    )
    producer_hash = _producer_artifact_sha256()
    transport_hash = _transport_artifact_sha256()
    fact_cores: list[NodeFrontierFact] = []
    for birth, position in zip(births, tracers, strict=True):
        node_id = v2._stable_id("birth.node_id", birth.node_id)
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
                parent_ribbon_digest_sha256=validated.current_ribbon_digest_sha256,
                parent_birth_step_index=validated.source_step_index,
                for_source_step_index=validated.source_step_index + 1,
                time_layer=DYADIC_CUMULATIVE_TIME_LAYER,
                transport_backend_id=TRANSPORT_BACKEND_ID,
                transport_artifact_sha256=transport_hash,
                producer_id=DYADIC_CUMULATIVE_PRODUCER_ID,
                producer_artifact_sha256=producer_hash,
                producer_report_sha256="0" * 64,
                transport_start_time_s=validated.source_time_s,
                transport_end_time_s=end_time,
                transport_substeps=substeps,
                advected_position_gp1_m=tuple(float(x) for x in position),
                fact_owner=DYADIC_CUMULATIVE_FACT_OWNER,
            )
        )
    zero_events = {
        name: sum(event.startswith(prefix) for event in trace)
        for name, prefix in (
            ("sort", "particle_sort:"),
            ("weld", "particle_weld:"),
            ("delete", "particle_delete:"),
            ("cancel", "particle_cancel:"),
            ("remesh", "particle_remesh:"),
            ("feedback", "feedback_write:"),
            ("parent", "parent_write:"),
            ("load", "load_write:"),
        )
    }
    placeholder = DyadicCumulativeCloudTransportReport(
        enabled=True,
        interface_id=DYADIC_CUMULATIVE_REPORT_INTERFACE_ID,
        producer_id=DYADIC_CUMULATIVE_PRODUCER_ID,
        producer_artifact_sha256=producer_hash,
        transport_backend_id=TRANSPORT_BACKEND_ID,
        transport_artifact_sha256=transport_hash,
        dyadic_bridge_artifact_sha256=_DYADIC_BRIDGE_IMPORT_SHA256,
        v2_core_artifact_sha256=FROZEN_V2_CORE_SHA256,
        wing_id=validated.wing_id,
        source_family=validated.source_family,
        parent_source_step_index=validated.source_step_index,
        for_source_step_index=validated.source_step_index + 1,
        time_layer=DYADIC_CUMULATIVE_TIME_LAYER,
        transport_start_time_s=validated.source_time_s,
        transport_end_time_s=end_time,
        transport_substeps=substeps,
        freestream_velocity_gp1_m_per_s=tuple(float(x) for x in freestream),
        parent_ribbon_digest_sha256=validated.current_ribbon_digest_sha256,
        current_ribbon_digest_sha256=validated.current_ribbon_digest_sha256,
        handoff_sha256=validated.handoff_sha256,
        parent_report_sha256=validated.previous_report_sha256,
        parent_cloud_digest_before_append_sha256=parent_cloud_digest,
        deposited_new_release_digest_sha256=deposited_digest,
        appended_cloud_digest_before_transport_sha256=appended_digest,
        transported_cloud_digest_after_sha256=cloud_digest,
        smoothing_radius_m=sigma_birth,
        base_target_spacing_m=base_spacing,
        refinement_level=level,
        nominal_target_spacing_m=nominal_spacing,
        previous_particle_count=old_count,
        new_particle_count=len(new_ids),
        total_particle_count=len(particle_ids),
        transported_particle_cloud=cloud,
        release_plan_sidecars=release_sidecars,
        facts=tuple(fact_cores),
        report_sha256="0" * 64,
        attestation_kind=DYADIC_CUMULATIVE_ATTESTATION_KIND,
        continuation_scope=DYADIC_CUMULATIVE_CONTINUATION_SCOPE,
        exact_append_passed=exact_gate,
        one_combined_field_passed=combined_gate,
        stage_pre_replay_passed=replay_gate,
        dyadic_plan_passed=plan_gate,
        plan_call_count=counts["plan"],
        deposition_call_count=counts["deposit"],
        sidecar_recompute_count=counts["sidecar"],
        predicted_new_particle_count=predicted_new_count,
        lsrk3_call_count=counts["call"],
        lsrk3_stage_count=counts["stage"],
        stage_pre_field_call_count=counts["field"],
        combined_stage_particle_counts=tuple(combined_counts),
        exact_append_prefix_max_abs=append_residual,
        stage_pre_replay_max_abs=replay_residual,
        transport_trace=trace,
        transport_trace_sha256=_trace_sha256(trace),
        sort_count=zero_events["sort"],
        weld_count=zero_events["weld"],
        delete_count=zero_events["delete"],
        cancel_count=zero_events["cancel"],
        remesh_count=zero_events["remesh"],
        feedback_call_count=zero_events["feedback"],
        parent_write_count=zero_events["parent"],
        load_write_count=zero_events["load"],
        observation_access="none",
        target_case_branch="none",
    )
    digest = _report_digest(placeholder)
    facts = tuple(replace(fact, producer_report_sha256=digest) for fact in fact_cores)
    report = replace(placeholder, facts=facts, report_sha256=digest)
    _validate_report_semantics(report)
    with _FROZEN_V2_LOCK:
        if _live_registry_entry(_HANDOFF_CONSUMPTIONS, validated) is not None:
            raise ValueError("dyadic cumulative handoff was already consumed")
        _register_live(_REPORT_REGISTRY, report, report.report_sha256)
        _register_live(_HANDOFF_CONSUMPTIONS, validated, report.report_sha256)
    return validate_dyadic_cumulative_cloud_transport_report(report)


def _validate_report_semantics(
    report: object,
) -> tuple[DyadicCumulativeCloudTransportReport, str]:
    """Recompute the portable dyadic report gates without the live registry."""

    trusted = _assert_frozen_dependencies()
    if type(report) is not DyadicCumulativeCloudTransportReport:
        raise ValueError("dyadic cumulative report has a foreign schema")
    if type(report.enabled) is not bool:
        raise ValueError("report.enabled must be an exact Boolean")
    enabled = report.enabled
    previous_count = v2._nonnegative_integer(
        "report.previous_particle_count", report.previous_particle_count
    )
    new_count = v2._nonnegative_integer(
        "report.new_particle_count", report.new_particle_count
    )
    total_count = v2._nonnegative_integer(
        "report.total_particle_count", report.total_particle_count
    )
    predicted_count = v2._nonnegative_integer(
        "report.predicted_new_particle_count", report.predicted_new_particle_count
    )
    if new_count > MAX_PARTICLES_PER_RELEASE or total_count > MAX_CUMULATIVE_PARTICLES:
        raise ValueError("dyadic cumulative report exceeds a particle cap")
    cloud_count = v2._cloud_particle_count_gate(
        report.transported_particle_cloud, allow_empty=not enabled
    )
    if (
        previous_count + new_count != total_count
        or predicted_count != new_count
        or cloud_count != total_count
    ):
        raise ValueError("dyadic cumulative append counts are inconsistent")
    if (
        report.interface_id != DYADIC_CUMULATIVE_REPORT_INTERFACE_ID
        or report.producer_id != DYADIC_CUMULATIVE_PRODUCER_ID
        or report.producer_artifact_sha256 != _producer_artifact_sha256()
        or report.transport_backend_id != TRANSPORT_BACKEND_ID
        or report.transport_artifact_sha256 != _transport_artifact_sha256()
        or report.dyadic_bridge_artifact_sha256 != _DYADIC_BRIDGE_IMPORT_SHA256
        or report.v2_core_artifact_sha256 != FROZEN_V2_CORE_SHA256
        or report.attestation_kind != DYADIC_CUMULATIVE_ATTESTATION_KIND
        or report.continuation_scope != DYADIC_CUMULATIVE_CONTINUATION_SCOPE
        or report.time_layer != DYADIC_CUMULATIVE_TIME_LAYER
    ):
        raise ValueError("dyadic cumulative producer/backend identity is stale")
    trace = report.transport_trace
    if _trace_sha256(trace) != _lower_sha256(
        "report.transport_trace_sha256", report.transport_trace_sha256
    ):
        raise ValueError("dyadic cumulative trace digest is inconsistent")
    trace_counts = {
        "plan": sum(event.startswith("dyadic_plan:") for event in trace),
        "deposit": sum(event.startswith("dyadic_deposit:") for event in trace),
        "sidecar": sum(
            event.startswith("dyadic_sidecar_recompute:") for event in trace
        ),
        "call": sum(event.startswith("combined_lsrk3_call:") for event in trace),
        "stage": sum(event.startswith("combined_lsrk3_stage:") for event in trace),
        "field": sum(event.startswith("frontier_stage_pre_field:") for event in trace),
        "sort": sum(event.startswith("particle_sort:") for event in trace),
        "weld": sum(event.startswith("particle_weld:") for event in trace),
        "delete": sum(event.startswith("particle_delete:") for event in trace),
        "cancel": sum(event.startswith("particle_cancel:") for event in trace),
        "remesh": sum(event.startswith("particle_remesh:") for event in trace),
        "feedback": sum(event.startswith("feedback_write:") for event in trace),
        "parent": sum(event.startswith("parent_write:") for event in trace),
        "load": sum(event.startswith("load_write:") for event in trace),
    }
    reported_counts = {
        "plan": v2._nonnegative_integer("plan_call_count", report.plan_call_count),
        "deposit": v2._nonnegative_integer(
            "deposition_call_count", report.deposition_call_count
        ),
        "sidecar": v2._nonnegative_integer(
            "sidecar_recompute_count", report.sidecar_recompute_count
        ),
        "call": v2._nonnegative_integer("lsrk3_call_count", report.lsrk3_call_count),
        "stage": v2._nonnegative_integer("lsrk3_stage_count", report.lsrk3_stage_count),
        "field": v2._nonnegative_integer(
            "stage_pre_field_call_count", report.stage_pre_field_call_count
        ),
        "sort": v2._nonnegative_integer("sort_count", report.sort_count),
        "weld": v2._nonnegative_integer("weld_count", report.weld_count),
        "delete": v2._nonnegative_integer("delete_count", report.delete_count),
        "cancel": v2._nonnegative_integer("cancel_count", report.cancel_count),
        "remesh": v2._nonnegative_integer("remesh_count", report.remesh_count),
        "feedback": v2._nonnegative_integer(
            "feedback_call_count", report.feedback_call_count
        ),
        "parent": v2._nonnegative_integer(
            "parent_write_count", report.parent_write_count
        ),
        "load": v2._nonnegative_integer("load_write_count", report.load_write_count),
    }
    if trace_counts != reported_counts:
        raise ValueError("dyadic cumulative counters disagree with the trace")
    if any(
        trace_counts[key]
        for key in (
            "sort",
            "weld",
            "delete",
            "cancel",
            "remesh",
            "feedback",
            "parent",
            "load",
        )
    ):
        raise ValueError("dyadic cumulative ownership contains a forbidden write")
    if report.observation_access != "none" or report.target_case_branch != "none":
        raise ValueError("dyadic cumulative report accessed target observations")

    positions, _, _ = v2._cloud_arrays(
        report.transported_particle_cloud, allow_empty=not enabled
    )
    if positions.shape[0] != total_count:
        raise ValueError("dyadic report/cloud count disagrees")
    if report.transported_particle_cloud.cloud_sha256 != _lower_sha256(
        "report.transported_cloud_digest_after_sha256",
        report.transported_cloud_digest_after_sha256,
    ):
        raise ValueError("dyadic report/cloud digest disagrees")
    append_residual = v2._finite_real(
        "report.exact_append_prefix_max_abs", report.exact_append_prefix_max_abs
    )
    replay_residual = v2._finite_real(
        "report.stage_pre_replay_max_abs", report.stage_pre_replay_max_abs
    )
    if append_residual < 0.0 or replay_residual < 0.0:
        raise ValueError("dyadic replay residuals must be nonnegative")
    stage_counts = tuple(
        v2._nonnegative_integer("combined stage particle count", value)
        for value in report.combined_stage_particle_counts
    )

    if enabled:
        wing = v2._stable_id("report.wing_id", report.wing_id)
        if report.source_family not in ("lev", "tev_persisted"):
            raise ValueError("enabled dyadic report has an invalid family")
        parent_step = v2._positive_integer(
            "report.parent_source_step_index", report.parent_source_step_index
        )
        if report.for_source_step_index != parent_step + 1:
            raise ValueError("dyadic source steps are nonconsecutive")
        start = v2._finite_real(
            "report.transport_start_time_s", report.transport_start_time_s
        )
        end = v2._finite_real(
            "report.transport_end_time_s", report.transport_end_time_s
        )
        if end <= start:
            raise ValueError("dyadic transport interval is invalid")
        substeps = v2._positive_integer(
            "report.transport_substeps", report.transport_substeps
        )
        if substeps > MAX_TRANSPORT_SUBSTEPS:
            raise ValueError("dyadic report exceeds the substep cap")
        sigma_birth = v2._positive_real(
            "report.smoothing_radius_m", report.smoothing_radius_m
        )
        base_spacing = v2._positive_real(
            "report.base_target_spacing_m", report.base_target_spacing_m
        )
        level = _level(report.refinement_level)
        nominal_spacing = v2._positive_real(
            "report.nominal_target_spacing_m", report.nominal_target_spacing_m
        )
        if nominal_spacing != base_spacing / (2**level):
            raise ValueError("dyadic nominal spacing disagrees with base/level")
        v2._finite_vector(
            "report.freestream_velocity_gp1_m_per_s",
            report.freestream_velocity_gp1_m_per_s,
        )
        _lower_sha256("report.handoff_sha256", report.handoff_sha256)
        parent_ribbon = _lower_sha256(
            "report.parent_ribbon_digest_sha256",
            report.parent_ribbon_digest_sha256,
        )
        if report.current_ribbon_digest_sha256 != parent_ribbon:
            raise ValueError("dyadic current and parent ribbon digests disagree")
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
            raise ValueError("first dyadic report has a phantom parent")
        if new_count:
            _lower_sha256(
                "report.deposited_new_release_digest_sha256",
                report.deposited_new_release_digest_sha256,
            )
        elif report.deposited_new_release_digest_sha256 is not None:
            raise ValueError("inactive dyadic report has a phantom deposition")

        expected_deposition = 1 if new_count else 0
        exact_event = (
            "exact_append:"
            f"old={previous_count}:new={new_count}:total={total_count}:"
            f"max_abs={append_residual.hex()}"
        )
        replay_event = f"independent_stage_pre_replay:max_abs={replay_residual.hex()}"
        exact_gate = (
            append_residual == 0.0
            and predicted_count == new_count
            and trace.count(exact_event) == 1
        )
        combined_gate = (
            trace_counts["call"] == substeps
            and trace_counts["stage"] == 3 * substeps
            and len(stage_counts) == 3 * substeps
            and all(value == total_count for value in stage_counts)
            and not any("split" in event for event in trace)
        )
        replay_gate = (
            trace_counts["field"] == 3 * substeps
            and replay_residual == 0.0
            and trace.count(replay_event) == 1
        )
        plan_gate = (
            trace_counts["plan"] == expected_deposition
            and trace_counts["deposit"] == expected_deposition
            and trace_counts["sidecar"] == expected_deposition
            and len(report.release_plan_sidecars)
            == len(report.transported_particle_cloud.release_slices)
        )
        for name, actual, expected in (
            ("exact_append_passed", report.exact_append_passed, exact_gate),
            (
                "one_combined_field_passed",
                report.one_combined_field_passed,
                combined_gate,
            ),
            ("stage_pre_replay_passed", report.stage_pre_replay_passed, replay_gate),
            ("dyadic_plan_passed", report.dyadic_plan_passed, plan_gate),
        ):
            if type(actual) is not bool or actual is not expected:
                raise ValueError(f"{name} disagrees with recomputed evidence")
        if not (exact_gate and combined_gate and replay_gate and plan_gate):
            raise ValueError("dyadic cumulative mechanical gate is false")

        slices = report.transported_particle_cloud.release_slices
        for release_index, (release, sidecar) in enumerate(
            zip(slices, report.release_plan_sidecars, strict=True), start=1
        ):
            if type(sidecar) is not DyadicReleasePlanSidecar:
                raise ValueError("dyadic release sidecar has a foreign schema")
            if (
                sidecar.release_index != release_index
                or sidecar.release_index != release.release_index
                or sidecar.source_step_index != release.source_step_index
                or sidecar.source_time_s != release.source_time_s
                or sidecar.deposited_cloud_digest_sha256
                != release.deposited_cloud_digest_sha256
                or sidecar.sidecar_sha256 != _sidecar_digest(sidecar)
            ):
                raise ValueError("dyadic release sidecar/slice binding disagrees")
            trusted["dyadic_strict_plan"](sidecar.plan)
            plan = sidecar.plan
            if (
                plan.smoothing_radius_m != sigma_birth
                or plan.base_target_spacing_m != base_spacing
                or plan.refinement_level != level
                or plan.step != release.source_step_index
                or plan.predicted_total_particle_count != release.particle_count
                or release.smoothing_radius_m != sigma_birth
                or release.deposition_target_spacing_m != nominal_spacing
                or len(sidecar.particle_ledger) != len(plan.edge_panels)
            ):
                raise ValueError("dyadic release plan changed across the cloud")
            cursor = 0
            for panel, ledger in zip(
                plan.edge_panels, sidecar.particle_ledger, strict=True
            ):
                if (
                    ledger.edge_key != panel.edge_key
                    or ledger.particle_slice_start != cursor
                    or ledger.particle_slice_stop - cursor != panel.panel_count
                    or ledger.parent_edge_graph_sha256 != plan.parent_edge_graph_sha256
                ):
                    raise ValueError("dyadic per-edge particle ledger is inconsistent")
                _lower_sha256("edge particle IDs", ledger.particle_ids_sha256)
                _lower_sha256("edge lineage", ledger.lineage_sha256)
                cursor = ledger.particle_slice_stop
            if cursor != release.particle_count:
                raise ValueError("dyadic edge ledgers do not cover the release")

        if new_count:
            latest = slices[-1]
            latest_sidecar = report.release_plan_sidecars[-1]
            if (
                latest.source_step_index != parent_step
                or latest.source_time_s != start
                or latest.particle_count != new_count
                or latest_sidecar.plan.predicted_total_particle_count != new_count
            ):
                raise ValueError("latest dyadic release time/count is inconsistent")

        fact_keys: set[tuple[str, int | str]] = set()
        for fact in report.facts:
            if type(fact) is not NodeFrontierFact:
                raise ValueError("dyadic report contains a foreign frontier fact")
            key = v2._id_key(v2._stable_id("fact.node_id", fact.node_id))
            if key in fact_keys:
                raise ValueError("dyadic frontier facts contain duplicate nodes")
            fact_keys.add(key)
            v2._nonnegative_integer("fact.lineage_epoch", fact.lineage_epoch)
            if (
                not isinstance(fact.parent_frontier_id, str)
                or not fact.parent_frontier_id
            ):
                raise ValueError("dyadic fact parent frontier identity is invalid")
            _lower_sha256(
                "fact.parent_frontier_digest_sha256",
                fact.parent_frontier_digest_sha256,
            )
            v2._finite_vector(
                "fact.advected_position_gp1_m", fact.advected_position_gp1_m
            )
            if (
                fact.wing_id != wing
                or fact.source_family != report.source_family
                or fact.parent_ribbon_digest_sha256 != parent_ribbon
                or fact.parent_birth_step_index != parent_step
                or fact.for_source_step_index != report.for_source_step_index
                or fact.time_layer != report.time_layer
                or fact.transport_backend_id != report.transport_backend_id
                or fact.transport_artifact_sha256 != report.transport_artifact_sha256
                or fact.producer_id != report.producer_id
                or fact.producer_artifact_sha256 != report.producer_artifact_sha256
                or fact.producer_report_sha256 != report.report_sha256
                or fact.transport_start_time_s != start
                or fact.transport_end_time_s != end
                or fact.transport_substeps != substeps
                or fact.fact_owner != DYADIC_CUMULATIVE_FACT_OWNER
            ):
                raise ValueError("dyadic frontier fact identity/time disagrees")
    else:
        if (
            total_count
            or previous_count
            or new_count
            or predicted_count
            or positions.shape[0]
            or report.release_plan_sidecars
            or report.facts
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
            or report.base_target_spacing_m is not None
            or report.refinement_level is not None
            or report.nominal_target_spacing_m is not None
            or stage_counts
            or report.exact_append_passed
            or report.one_combined_field_passed
            or report.stage_pre_replay_passed
            or report.dyadic_plan_passed
        ):
            raise ValueError("disabled dyadic report is not input-blind and empty")

    digest = _report_digest(report)
    if not hmac.compare_digest(
        digest, _lower_sha256("report.report_sha256", report.report_sha256)
    ):
        raise ValueError("dyadic cumulative report digest is inconsistent")
    return report, digest


def validate_dyadic_cumulative_cloud_transport_report(
    report: object,
) -> DyadicCumulativeCloudTransportReport:
    """Validate a directly produced live dyadic cumulative report."""

    validated, digest = _validate_report_semantics(report)
    registered = _live_registry_entry(_REPORT_REGISTRY, validated)
    if registered is None or not hmac.compare_digest(registered[1], digest):
        raise ValueError("dyadic report is not a directly produced live object")
    return validated


def materialize_dyadic_cumulative_particle_state(report: object) -> object:
    """Return fresh mutable arrays for one live dyadic cumulative cloud."""

    trusted = _assert_frozen_dependencies()
    validated = validate_dyadic_cumulative_cloud_transport_report(report)
    positions, gamma, sigma = v2._cloud_arrays(
        validated.transported_particle_cloud, allow_empty=not validated.enabled
    )
    state_callable = trusted["particle_state"]
    return state_callable(positions.copy(), gamma.copy(), sigma.copy())


__all__ = [
    "DYADIC_CUMULATIVE_ATTESTATION_KIND",
    "DYADIC_CUMULATIVE_CONTINUATION_SCOPE",
    "DYADIC_CUMULATIVE_FACT_OWNER",
    "DYADIC_CUMULATIVE_HANDOFF_INTERFACE_ID",
    "DYADIC_CUMULATIVE_PRODUCER_ID",
    "DYADIC_CUMULATIVE_REPORT_INTERFACE_ID",
    "DYADIC_CUMULATIVE_TIME_LAYER",
    "DyadicCumulativeCloudTransportReport",
    "DyadicCumulativeRibbonHandoff",
    "DyadicReleasePlanSidecar",
    "MAX_CUMULATIVE_PARTICLES",
    "MAX_PARTICLES_PER_RELEASE",
    "MAX_PARTICLE_COUNT",
    "MAX_TRANSPORT_SUBSTEPS",
    "attest_dyadic_cumulative_ribbon_handoff",
    "materialize_dyadic_cumulative_particle_state",
    "transport_dyadic_accumulated_particle_cloud",
    "validate_dyadic_cumulative_cloud_transport_report",
]
