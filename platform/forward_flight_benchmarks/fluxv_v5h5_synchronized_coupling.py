"""Three-layer synchronized DVM--Ptera--rVPM mechanical vertical slice.

The v5h5 owner orders every time layer as

``release -> native Ptera solve/load -> one combined rVPM transport``.

It deliberately does not call the already-transporting v5h2 cumulative
producer.  A new post-release/pre-transport cloud is assembled from live DVM
events and a dyadic edge deposition, used read-only by the v5h3 native Ptera
feedback path, and advanced once after Ptera loads close by the v5h4 frozen
parent-field step.  The transported node frontiers seed the next continuous
release.  This is a generic straight-wing mechanical slice, not a target-load
or aerodynamic-accuracy model.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256
import hmac
from numbers import Integral, Real
from pathlib import Path
from threading import RLock
from typing import Any, Final, Literal, Sequence
import weakref

import numpy as np

import fluxvortex.rvpm_dyadic_edge_bridge as _dyadic_bridge_module
import fluxvortex.rvpm_edge_bridge as _edge_bridge_module
from fluxvortex.rvpm_dyadic_edge_bridge import (
    DyadicBridgeResult,
    deposit_edge_graph_prescribed_sigma_dyadic_panels,
    validate_dyadic_bridge_result,
)
from fluxvortex.rvpm_edge_bridge import (
    BridgeNode,
    DirectedRing,
    ParticleLineage,
    assemble_ring_edge_graph,
)
from fluxvortex.rvpm_reference import direct_gaussian_erf_velocity_jacobian
from fluxvortex.rvpm_transport import ParticleState, make_particle_state
from fluxvortex.solver import UVPMHybridSolver

from . import fluxv_v5h3_native_feedback as _v5h3
from . import fluxv_v5h4_ptera_rvpm_transport as _v5h4
from . import v5h_dvm_node_placement as _node_placement_module
from . import v5h_dvm_source as _source_module
from .fluxv_v5h3_native_feedback import (
    NativePteraRVPMFeedbackConfig,
    NativePteraRVPMFeedbackSolver,
    NativePteraRVPMFeedbackStepReport,
    NativePteraRVPMVelocityEvaluation,
    _PendingFeedbackStep,
)
from .fluxv_v5h4_ptera_rvpm_transport import (
    FORCE_SCORING_STATUS,
    NOMINAL_RELATIVE_EPSILON,
    FrozenExternalField,
    FrozenPteraRVPMTransportResult,
    PteraFieldEvaluation,
    evaluate_frozen_parent_ptera_field,
    lsrk3_step_with_external_field,
    ptera_parent_state_sha256,
)
from .v5h_dvm_node_ribbon import (
    DVMPlaneToGP1Map,
    DVMSpanCellSource,
    NodeBirthRecord,
    SpanNodeKinematics,
)
from .v5h_dvm_node_placement import (
    DVMNodePlacementResult,
    validate_live_dvm_node_placement_result,
)
from .v5h_dvm_source import DVMSourceEvent, validate_dvm_source_event
from .v5h_passive_frontier_transport import (
    NodeFrontierFact,
    parent_frontier_digest_sha256,
)


SYNCHRONIZED_INTERFACE_ID: Final = "fluxv-v5h5-synchronized-coupling-v1"
SYNCHRONIZED_OWNER: Final = "dvm-release-ptera-native-rvpm-single-transport"
SYNCHRONIZED_TIME_LAYER: Final = (
    "post-release-pre-transport-t_n-to-pre-release-t_n-plus-1"
)
SYNCHRONIZED_FACT_OWNER: Final = "v5h5-combined-stage-frontier"
OBSERVATION_ACCESS: Final = "none"
TARGET_CASE_BRANCH: Final = "none"
MAX_PARTICLES: Final = 1_000_000

StableId = int | str
SourceFamily = Literal["lev"]


@dataclass(frozen=True, slots=True)
class SynchronizedReleaseLayer:
    """One pre-evaluated live DVM layer supplied to the synchronized solver."""

    source_step_index: int
    source_time_s: float
    cells: tuple[DVMSpanCellSource, ...]
    node_placement_result: DVMNodePlacementResult


@dataclass(frozen=True, slots=True)
class SynchronizedReleaseSlice:
    """Exact particle interval owned by one DVM release."""

    release_index: int
    source_step_index: int
    source_time_s: float
    start_index: int
    stop_index: int
    particle_count: int
    edge_graph_sha256: str
    bridge_manifest_sha256: str
    source_event_manifest_sha256: tuple[str, ...]


@dataclass(frozen=True, slots=True, weakref_slot=True)
class PreparedSynchronizedCloud:
    """Post-release/pre-transport cloud staged for one native Ptera step."""

    enabled: bool
    interface_id: str
    wing_id: StableId
    source_family: SourceFamily
    ptera_step_index: int
    for_source_step_index: int
    transport_end_time_s: float
    positions_gp1_m: np.ndarray
    gamma_vector_m3_per_s: np.ndarray
    sigma_m: np.ndarray
    particle_ids: tuple[tuple[Any, ...], ...]
    lineage: tuple[ParticleLineage, ...]
    release_slices: tuple[SynchronizedReleaseSlice, ...]
    node_placement_manifest_sha256: str
    node_kinematics: tuple[SpanNodeKinematics, ...]
    node_births: tuple[NodeBirthRecord, ...]
    previous_particle_count: int
    new_particle_count: int
    total_particle_count: int
    exact_append_prefix_max_abs: float
    report_sha256: str


@dataclass(frozen=True, slots=True, weakref_slot=True)
class SynchronizedCloudTransportReport:
    """One committed Ptera-feedback plus combined-transport layer."""

    enabled: bool
    interface_id: str
    wing_id: StableId
    source_family: SourceFamily
    parent_source_step_index: int
    for_source_step_index: int
    time_layer: str
    transport_start_time_s: float
    transport_end_time_s: float
    parent_ribbon_digest_sha256: str
    current_ribbon_digest_sha256: str
    parent_report_sha256: str | None
    prepared_report_sha256: str
    report_sha256: str
    prepared_cloud: PreparedSynchronizedCloud
    transported_state: ParticleState
    particle_ids: tuple[tuple[Any, ...], ...]
    lineage: tuple[ParticleLineage, ...]
    release_slices: tuple[SynchronizedReleaseSlice, ...]
    facts: tuple[NodeFrontierFact, ...]
    ptera_feedback_report: NativePteraRVPMFeedbackStepReport | None
    transport_result: FrozenPteraRVPMTransportResult
    previous_particle_count: int
    new_particle_count: int
    total_particle_count: int
    release_call_count: int
    ptera_feedback_enabled: bool
    combined_transport_enabled: bool
    ptera_feedback_call_count: int
    combined_transport_call_count: int
    ptera_center_call_count: int
    ptera_finite_difference_call_count: int
    frontier_transport_parent_call_count: int
    frontier_replay_parent_call_count: int
    frontier_stage_replay_max_abs: float
    exact_append_prefix_max_abs: float
    parent_state_sha256_before_transport: str
    parent_state_sha256_after_transport: str
    parent_state_unchanged: bool
    feedback_write_count: int
    parent_write_count: int
    load_write_count: int
    observation_access: str
    target_case_branch: str
    force_scoring_status: str


_LOCK = RLock()
_EVENT_CONSUMPTIONS: dict[int, tuple[weakref.ReferenceType[object], str]] = {}
_REPORT_REGISTRY: dict[int, tuple[weakref.ReferenceType[object], str]] = {}


def _source_sha256(module: object) -> str:
    path = getattr(module, "__file__", None)
    if type(path) is not str or not path.endswith(".py"):
        raise RuntimeError("v5h5 dependency has no auditable Python source")
    return sha256(Path(path).read_bytes()).hexdigest()


def _producer_artifact_sha256() -> str:
    return sha256(Path(__file__).read_bytes()).hexdigest()


_FROZEN_SOURCES = {
    "v5h3": _source_sha256(_v5h3),
    "v5h4": _source_sha256(_v5h4),
    "node_placement": _source_sha256(_node_placement_module),
    "source": _source_sha256(_source_module),
    "dyadic_bridge": _source_sha256(_dyadic_bridge_module),
    "edge_bridge": _source_sha256(_edge_bridge_module),
}
_FROZEN_DEPOSIT = deposit_edge_graph_prescribed_sigma_dyadic_panels
_FROZEN_VALIDATE_BRIDGE = validate_dyadic_bridge_result
_FROZEN_ASSEMBLE = assemble_ring_edge_graph
_FROZEN_VALIDATE_EVENT = validate_dvm_source_event
_FROZEN_VALIDATE_NODE_PLACEMENT = validate_live_dvm_node_placement_result
_FROZEN_DIRECT_FIELD = direct_gaussian_erf_velocity_jacobian


def _assert_bindings() -> None:
    if (
        _dyadic_bridge_module.deposit_edge_graph_prescribed_sigma_dyadic_panels
        is not _FROZEN_DEPOSIT
        or _dyadic_bridge_module.validate_dyadic_bridge_result
        is not _FROZEN_VALIDATE_BRIDGE
        or _edge_bridge_module.assemble_ring_edge_graph is not _FROZEN_ASSEMBLE
        or _source_module.validate_dvm_source_event is not _FROZEN_VALIDATE_EVENT
        or _node_placement_module.validate_live_dvm_node_placement_result
        is not _FROZEN_VALIDATE_NODE_PLACEMENT
        or any(
            _source_sha256(module) != _FROZEN_SOURCES[name]
            for name, module in (
                ("v5h3", _v5h3),
                ("v5h4", _v5h4),
                ("node_placement", _node_placement_module),
                ("source", _source_module),
                ("dyadic_bridge", _dyadic_bridge_module),
                ("edge_bridge", _edge_bridge_module),
            )
        )
    ):
        raise ValueError("v5h5 dependency callable or source was replaced")
    _v5h3._assert_frozen_bindings()
    _v5h4._assert_bindings()


def _id_key(value: StableId) -> tuple[str, int | str]:
    if isinstance(value, bool) or not isinstance(value, (int, str)) or value == "":
        raise ValueError("stable IDs must be explicit integers or nonempty strings")
    return ("integer", value) if isinstance(value, int) else ("string", value)


def _finite_real(name: str, value: object) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be a finite real")
    return result


def _positive_integer(name: str, value: object) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be a positive integer")
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _vector(name: str, value: object) -> np.ndarray:
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf" or raw.dtype.kind == "b":
        raise ValueError(f"{name} must be numeric")
    result = np.ascontiguousarray(raw, dtype=np.float64)
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite length-three vector")
    return result


def _readonly(array: object, *, ndim: int) -> np.ndarray:
    raw = np.asarray(array)
    if raw.dtype.kind not in "iuf" or raw.dtype.kind == "b":
        raise ValueError("cloud arrays must be numeric")
    result = np.ascontiguousarray(raw, dtype=np.float64).copy()
    if result.ndim != ndim or not np.all(np.isfinite(result)):
        raise ValueError("cloud array has invalid shape or non-finite values")
    result.setflags(write=False)
    return result


def _array_hash(digest: Any, value: np.ndarray) -> None:
    array = np.asarray(value)
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(repr(array.shape).encode("ascii"))
    digest.update(array.tobytes(order="C"))


def _prepared_digest(value: PreparedSynchronizedCloud) -> str:
    digest = sha256()
    digest.update(SYNCHRONIZED_INTERFACE_ID.encode("ascii"))
    digest.update(repr(value.wing_id).encode("utf-8"))
    digest.update(value.source_family.encode("ascii"))
    digest.update(str(value.ptera_step_index).encode("ascii"))
    digest.update(str(value.for_source_step_index).encode("ascii"))
    digest.update(value.transport_end_time_s.hex().encode("ascii"))
    for array in (
        value.positions_gp1_m,
        value.gamma_vector_m3_per_s,
        value.sigma_m,
    ):
        _array_hash(digest, array)
    digest.update(repr(value.particle_ids).encode("utf-8"))
    digest.update(repr(value.lineage).encode("utf-8"))
    digest.update(repr(value.release_slices).encode("utf-8"))
    digest.update(value.node_placement_manifest_sha256.encode("ascii"))
    digest.update(repr(value.node_kinematics).encode("utf-8"))
    digest.update(repr(value.node_births).encode("utf-8"))
    return digest.hexdigest()


def _report_digest(value: SynchronizedCloudTransportReport) -> str:
    digest = sha256()
    digest.update(SYNCHRONIZED_INTERFACE_ID.encode("ascii"))
    digest.update(repr(value.wing_id).encode("utf-8"))
    digest.update(str(value.parent_source_step_index).encode("ascii"))
    digest.update(str(value.for_source_step_index).encode("ascii"))
    digest.update(value.transport_start_time_s.hex().encode("ascii"))
    digest.update(value.transport_end_time_s.hex().encode("ascii"))
    digest.update(value.prepared_report_sha256.encode("ascii"))
    digest.update(_prepared_digest(value.prepared_cloud).encode("ascii"))
    digest.update((value.parent_report_sha256 or "none").encode("ascii"))
    for array in (
        value.transported_state.positions,
        value.transported_state.gamma,
        value.transported_state.sigma,
    ):
        _array_hash(digest, array)
    digest.update(repr(value.particle_ids).encode("utf-8"))
    digest.update(repr(value.lineage).encode("utf-8"))
    digest.update(repr(value.release_slices).encode("utf-8"))
    digest.update(
        repr(
            tuple(
                (
                    fact.node_id,
                    fact.lineage_epoch,
                    fact.advected_position_gp1_m,
                )
                for fact in value.facts
            )
        ).encode("utf-8")
    )
    digest.update(value.parent_state_sha256_before_transport.encode("ascii"))
    digest.update(value.parent_state_sha256_after_transport.encode("ascii"))
    digest.update(
        repr(
            (
                value.previous_particle_count,
                value.new_particle_count,
                value.total_particle_count,
                value.release_call_count,
                value.ptera_feedback_enabled,
                value.combined_transport_enabled,
                value.ptera_feedback_call_count,
                value.combined_transport_call_count,
                value.ptera_center_call_count,
                value.ptera_finite_difference_call_count,
                value.frontier_transport_parent_call_count,
                value.frontier_replay_parent_call_count,
                value.frontier_stage_replay_max_abs,
                value.exact_append_prefix_max_abs,
                value.parent_state_unchanged,
                value.feedback_write_count,
                value.parent_write_count,
                value.load_write_count,
                value.observation_access,
                value.target_case_branch,
                value.force_scoring_status,
            )
        ).encode("utf-8")
    )
    return digest.hexdigest()


def _register_live(
    registry: dict[int, tuple[weakref.ReferenceType[object], str]],
    value: object,
    digest: str,
) -> None:
    identity = id(value)
    old = registry.get(identity)
    if old is not None and old[0]() is not None:
        raise ValueError("v5h5 live identity was already consumed or registered")
    registry[identity] = (weakref.ref(value), digest)


def _live_digest(
    registry: dict[int, tuple[weakref.ReferenceType[object], str]], value: object
) -> str | None:
    old = registry.get(id(value))
    if old is None:
        return None
    if old[0]() is not value:
        if old[0]() is None:
            registry.pop(id(value), None)
            return None
        raise RuntimeError("v5h5 live registry identity collision")
    return old[1]


def _plane_sign(value: object) -> int:
    if type(value) is not DVMPlaneToGP1Map:
        raise ValueError("every synchronized cell needs an exact plane map")
    x_axis = _vector("plane x axis", value.x_axis_gp1)
    z_axis = _vector("plane z axis", value.z_axis_gp1)
    positive = _vector("plane circulation axis", value.positive_circulation_axis_gp1)
    if (
        abs(float(np.linalg.norm(x_axis)) - 1.0) > 1.0e-12
        or abs(float(np.linalg.norm(z_axis)) - 1.0) > 1.0e-12
        or abs(float(np.dot(x_axis, z_axis))) > 1.0e-12
        or float(np.linalg.norm(np.cross(x_axis, z_axis) - positive)) > 1.0e-12
    ):
        raise ValueError("DVM plane map is not an oriented orthonormal frame")
    sign = value.circulation_to_ring_traversal_sign
    if type(sign) is not int or sign not in (-1, 1):
        raise ValueError("plane circulation traversal sign must be exactly +/-1")
    if type(value.provenance) is not str or not value.provenance:
        raise ValueError("plane map requires explicit provenance")
    return sign


def _edge_graph_sha256(graph: object) -> str:
    digest = sha256()
    digest.update(repr(graph).encode("utf-8"))
    return digest.hexdigest()


def _consume_event(event: DVMSourceEvent) -> None:
    digest = event.producer_manifest_sha256
    with _LOCK:
        old = _live_digest(_EVENT_CONSUMPTIONS, event)
        if old is not None:
            raise ValueError("live DVM event was already consumed by v5h5")
        _register_live(_EVENT_CONSUMPTIONS, event, digest)


def validate_synchronized_cloud_transport_report(
    value: object,
) -> SynchronizedCloudTransportReport:
    """Validate one directly produced live synchronized report."""

    _assert_bindings()
    if type(value) is not SynchronizedCloudTransportReport:
        raise ValueError("synchronized report has a foreign schema")
    if not value.enabled or value.interface_id != SYNCHRONIZED_INTERFACE_ID:
        raise ValueError("synchronized report identity is invalid")
    if value.for_source_step_index != value.parent_source_step_index + 1:
        raise ValueError("synchronized report source steps are nonconsecutive")
    if value.transport_end_time_s <= value.transport_start_time_s:
        raise ValueError("synchronized report interval is invalid")
    prepared = value.prepared_cloud
    if type(prepared) is not PreparedSynchronizedCloud:
        raise ValueError("synchronized prepared cloud has a foreign schema")
    arrays = (
        prepared.positions_gp1_m,
        prepared.gamma_vector_m3_per_s,
        prepared.sigma_m,
        value.transported_state.positions,
        value.transported_state.gamma,
        value.transported_state.sigma,
    )
    if any(not np.all(np.isfinite(array)) for array in arrays):
        raise ValueError("synchronized report contains non-finite particle state")
    if (
        _prepared_digest(prepared) != value.prepared_report_sha256
        or prepared.report_sha256 != value.prepared_report_sha256
        or prepared.total_particle_count != value.total_particle_count
        or value.total_particle_count != len(value.particle_ids)
        or value.total_particle_count != len(value.lineage)
        or len(set(value.particle_ids)) != value.total_particle_count
        or prepared.particle_ids != value.particle_ids
        or prepared.lineage != value.lineage
        or prepared.release_slices != value.release_slices
        or value.transported_state.positions.shape != (value.total_particle_count, 3)
        or value.transported_state.gamma.shape != (value.total_particle_count, 3)
        or value.transported_state.sigma.shape != (value.total_particle_count,)
        or np.any(value.transported_state.sigma <= 0.0)
    ):
        raise ValueError("synchronized report cloud count/shape is invalid")
    expected_start = 0
    for release_index, release_slice in enumerate(value.release_slices, start=1):
        if (
            type(release_slice) is not SynchronizedReleaseSlice
            or release_slice.release_index != release_index
            or release_slice.start_index != expected_start
            or release_slice.stop_index
            != release_slice.start_index + release_slice.particle_count
            or release_slice.particle_count <= 0
        ):
            raise ValueError("synchronized release slices are not contiguous")
        expected_start = release_slice.stop_index
    if expected_start != value.total_particle_count:
        raise ValueError("synchronized release slices do not cover the cloud")
    births = {birth.node_id: birth for birth in prepared.node_births}
    facts = {fact.node_id: fact for fact in value.facts}
    if len(births) != len(prepared.node_births) or set(births) != set(facts):
        raise ValueError("synchronized frontier facts do not cover the release nodes")
    for node_id, fact in facts.items():
        birth = births[node_id]
        if (
            fact.wing_id != value.wing_id
            or fact.source_family != value.source_family
            or fact.parent_frontier_id != birth.birth_node_id
            or fact.parent_ribbon_digest_sha256 != value.prepared_report_sha256
            or fact.parent_birth_step_index != value.parent_source_step_index
            or fact.for_source_step_index != value.for_source_step_index
            or fact.time_layer != SYNCHRONIZED_TIME_LAYER
            or fact.producer_id != SYNCHRONIZED_OWNER
            or fact.producer_artifact_sha256 != _producer_artifact_sha256()
            or fact.producer_report_sha256 != value.report_sha256
            or fact.transport_start_time_s != value.transport_start_time_s
            or fact.transport_end_time_s != value.transport_end_time_s
            or fact.transport_substeps != 1
            or fact.fact_owner != SYNCHRONIZED_FACT_OWNER
            or not np.all(np.isfinite(fact.advected_position_gp1_m))
        ):
            raise ValueError("synchronized frontier fact ownership/time gate failed")
    feedback = value.ptera_feedback_report
    transport = value.transport_result
    if (
        not value.combined_transport_enabled
        or value.ptera_feedback_call_count != int(value.ptera_feedback_enabled)
        or (
            value.ptera_feedback_enabled
            and (
                type(feedback) is not NativePteraRVPMFeedbackStepReport
                or feedback.ptera_step_index != prepared.ptera_step_index
                or feedback.dvm_for_source_step_index != prepared.for_source_step_index
                or feedback.particle_count != value.total_particle_count
                or feedback.collocation_evaluation_count != 1
                or feedback.load_leg_evaluation_count != 4
                or feedback.parent_load_call_count != 1
                or feedback.extension_force_write_count != 0
                or feedback.extension_moment_write_count != 0
                or feedback.extension_load_processor_call_count != 0
                or not feedback.prescribed_wake
                or not np.isfinite(feedback.no_penetration_max_abs)
                or feedback.no_penetration_max_abs > 1.0e-12
            )
        )
        or (not value.ptera_feedback_enabled and feedback is not None)
        or transport.source_report_sha256 != value.prepared_report_sha256
        or transport.ptera_step_index != prepared.ptera_step_index
        or transport.dvm_for_source_step_index != prepared.for_source_step_index
        or transport.initial_particle_count != value.total_particle_count
        or len(transport.stages) != 3
        or transport.ptera_center_call_count != 3
        or transport.ptera_finite_difference_call_count != 18
        or not transport.ptera_parent_state_unchanged
        or transport.feedback_write_count != 0
        or transport.parent_write_count != 0
        or transport.load_write_count != 0
        or not np.array_equal(
            transport.final_state.positions, value.transported_state.positions
        )
        or not np.array_equal(
            transport.final_state.gamma, value.transported_state.gamma
        )
        or not np.array_equal(
            transport.final_state.sigma, value.transported_state.sigma
        )
    ):
        raise ValueError("synchronized Ptera/transport nested ledger failed")
    if (
        value.release_call_count != 1
        or value.combined_transport_call_count != 1
        or value.ptera_center_call_count != 3
        or value.ptera_finite_difference_call_count != 18
        or value.frontier_transport_parent_call_count != 3
        or value.frontier_replay_parent_call_count != 3
        or value.frontier_stage_replay_max_abs != 0.0
        or value.exact_append_prefix_max_abs != 0.0
        or not value.parent_state_unchanged
        or value.parent_state_sha256_before_transport
        != value.parent_state_sha256_after_transport
        or value.feedback_write_count
        or value.parent_write_count
        or value.load_write_count
        or value.observation_access != OBSERVATION_ACCESS
        or value.target_case_branch != TARGET_CASE_BRANCH
    ):
        raise ValueError("synchronized report mechanical ownership gate failed")
    digest = _report_digest(value)
    if not hmac.compare_digest(digest, value.report_sha256):
        raise ValueError("synchronized report digest is inconsistent")
    with _LOCK:
        registered = _live_digest(_REPORT_REGISTRY, value)
    if registered is None or not hmac.compare_digest(registered, digest):
        raise ValueError("synchronized report is not a live producer object")
    return value


def materialize_synchronized_particle_state(value: object) -> ParticleState:
    """Return a mutable copy of a validated synchronized cloud."""

    report = validate_synchronized_cloud_transport_report(value)
    return make_particle_state(
        report.transported_state.positions.copy(),
        report.transported_state.gamma.copy(),
        report.transported_state.sigma.copy(),
    )


class SynchronizedPteraRVPMCouplingSolver(NativePteraRVPMFeedbackSolver):
    """v5h3 native Ptera feedback with one ordered v5h5 cloud transport."""

    def __init__(
        self,
        unsteady_problem: Any,
        *,
        release_layers: Sequence[SynchronizedReleaseLayer],
        wing_id: StableId = "wing",
        source_family: SourceFamily = "lev",
        smoothing_radius_m: float = 0.085,
        base_target_spacing_m: float = 0.04,
        refinement_level: int = 0,
        relative_epsilon: float = NOMINAL_RELATIVE_EPSILON,
        feedback_enabled: bool = True,
        transport_enabled: bool = True,
        **uvpm_kwargs: Any,
    ) -> None:
        _assert_bindings()
        if source_family != "lev":
            raise ValueError("v5h5 vertical slice supports only the LEV family")
        self._v5h5_wing_id = wing_id
        _id_key(wing_id)
        self._v5h5_source_family = source_family
        self._v5h5_sigma_birth = _finite_real("smoothing_radius_m", smoothing_radius_m)
        self._v5h5_base_spacing = _finite_real(
            "base_target_spacing_m", base_target_spacing_m
        )
        if self._v5h5_sigma_birth <= 0.0 or self._v5h5_base_spacing <= 0.0:
            raise ValueError("v5h5 core and spacing must be positive")
        if type(refinement_level) is not int or refinement_level < 0:
            raise ValueError("refinement_level must be a nonnegative exact integer")
        self._v5h5_level = refinement_level
        self._v5h5_relative_epsilon = _finite_real("relative_epsilon", relative_epsilon)
        if self._v5h5_relative_epsilon not in _v5h4.PREREGISTERED_RELATIVE_EPSILONS:
            raise ValueError("relative_epsilon is outside the v5h4 family")
        if type(feedback_enabled) is not bool or type(transport_enabled) is not bool:
            raise TypeError("v5h5 reduction switches must be exact bools")
        if not feedback_enabled and not transport_enabled:
            raise ValueError("use the disabled factory for the exact parent reduction")
        self._v5h5_feedback_enabled = feedback_enabled
        self._v5h5_transport_enabled = transport_enabled
        layers = tuple(release_layers)
        self._v5h5_layers = layers
        self._v5h5_previous_report: SynchronizedCloudTransportReport | None = None
        self._v5h5_prepared: PreparedSynchronizedCloud | None = None
        self._v5h5_node_anchors: dict[StableId, tuple[float, float, float]] = {}
        self._v5h5_previous_event_by_cell: dict[StableId, DVMSourceEvent] = {}
        self.v5h5_transport_reports: list[SynchronizedCloudTransportReport] = []
        self.v5h5_prepared_history: list[PreparedSynchronizedCloud] = []
        super().__init__(
            unsteady_problem,
            feedback_config=NativePteraRVPMFeedbackConfig(
                enabled=True,
                expected_wing_id=str(wing_id),
                expected_source_family=source_family,
            ),
            feedback_reports=(),
            **uvpm_kwargs,
        )
        if len(layers) != self.num_steps:
            raise ValueError("v5h5 requires exactly one release layer per Ptera step")
        if not transport_enabled and len(layers) != 1:
            raise ValueError("feedback-only reduction is a single-layer gate")
        for index, layer in enumerate(layers, start=1):
            if type(layer) is not SynchronizedReleaseLayer:
                raise ValueError("release_layers contain a foreign schema")
            if layer.source_step_index != index or layer.source_time_s != (
                index - 1
            ) * float(self.delta_time):
                raise ValueError("release layer step/time is inconsistent")

    def _build_prepared_cloud(self) -> PreparedSynchronizedCloud:
        _assert_bindings()
        ptera_step = int(self._current_step)
        layer = self._v5h5_layers[ptera_step]
        source_step = layer.source_step_index
        cells = tuple(layer.cells)
        placement = _FROZEN_VALIDATE_NODE_PLACEMENT(
            layer.node_placement_result,
            expected_wing_id=str(self._v5h5_wing_id),
        )
        if (
            not placement.enabled
            or placement.source_step_index != source_step
            or placement.source_time_s != layer.source_time_s
            or placement.delta_time_s != float(self.delta_time)
        ):
            raise ValueError("node-local DVM placement has a wrong step/time layer")
        nodes = tuple(placement.kinematics)
        if not cells or not nodes:
            raise ValueError("synchronized release layer cannot be empty")
        node_positions: dict[StableId, np.ndarray] = {}
        node_velocities: dict[StableId, np.ndarray] = {}
        for node in nodes:
            if type(node) is not SpanNodeKinematics:
                raise ValueError("release nodes have a foreign schema")
            node_id = node.node_id
            _id_key(node_id)
            if node_id in node_positions:
                raise ValueError("release layer repeats a node ID")
            anchor = _vector("node anchor", node.anchor_position_gp1_m)
            velocity = _vector("node velocity", node.edge_velocity_gp1_m_per_s)
            frozen_anchor = self._v5h5_node_anchors.get(node_id)
            anchor_tuple = tuple(float(item) for item in anchor)
            if frozen_anchor is not None and anchor_tuple != frozen_anchor:
                raise ValueError("node anchor changed across synchronized releases")
            self._v5h5_node_anchors.setdefault(node_id, anchor_tuple)
            node_positions[node_id] = anchor
            node_velocities[node_id] = velocity

        previous_facts = (
            {}
            if self._v5h5_previous_report is None
            else {fact.node_id: fact for fact in self._v5h5_previous_report.facts}
        )
        active_nodes: set[StableId] = set()
        validated_cells: list[tuple[DVMSpanCellSource, DVMSourceEvent, int]] = []
        event_manifests: list[str] = []
        coverage_by_cell = {
            coverage.cell_id: coverage for coverage in placement.cell_coverage
        }
        seen_cells: set[StableId] = set()
        for cell in cells:
            if type(cell) is not DVMSpanCellSource:
                raise ValueError("release cells have a foreign schema")
            _id_key(cell.cell_id)
            if cell.cell_id in seen_cells:
                raise ValueError("release layer repeats a cell ID")
            seen_cells.add(cell.cell_id)
            if (
                cell.left_node_id not in node_positions
                or cell.right_node_id not in node_positions
            ):
                raise ValueError("release cell references an unknown node")
            event = _FROZEN_VALIDATE_EVENT(cell.event)
            coverage = coverage_by_cell.get(cell.cell_id)
            if (
                coverage is None
                or not coverage.endpoint_coverage_complete
                or not coverage.shared_fact_identity_verified
                or coverage.left_node_id != cell.left_node_id
                or coverage.right_node_id != cell.right_node_id
                or coverage.cell_source_event_manifest_sha256
                != event.producer_manifest_sha256
            ):
                raise ValueError("cell source is not bound to its node placement")
            if (
                not event.enabled
                or not event.lesp_active
                or event.lineage.source_step_index != source_step
                or event.provenance.observation_access != "none"
                or event.provenance.target_case_branch != "none"
            ):
                raise ValueError("v5h5 requires an active isolated DVM source event")
            expected_mode = "first" if source_step == 1 else "continuous"
            if event.lev_birth_mode != expected_mode:
                raise ValueError(
                    "DVM event birth mode disagrees with synchronized step"
                )
            previous_event = self._v5h5_previous_event_by_cell.get(cell.cell_id)
            if previous_event is None:
                if source_step != 1:
                    raise ValueError("synchronized DVM cell history starts late")
            elif (
                event.parent_event_manifest_sha256
                != previous_event.producer_manifest_sha256
                or event.lineage.physical_strip_id
                != previous_event.lineage.physical_strip_id
                or event.lineage.physical_section_id
                != previous_event.lineage.physical_section_id
            ):
                raise ValueError("synchronized DVM event history is discontinuous")
            sign = _plane_sign(cell.plane_to_gp1)
            validated_cells.append((cell, event, sign))
            event_manifests.append(event.producer_manifest_sha256)
            active_nodes.update((cell.left_node_id, cell.right_node_id))

        if active_nodes != set(node_positions):
            raise ValueError("v5h5 first slice requires complete active node coverage")
        births: dict[StableId, NodeBirthRecord] = {}
        for node_id in sorted(active_nodes, key=_id_key):
            anchor = node_positions[node_id]
            if source_step == 1:
                mode = "first"
                birth = anchor + 0.5 * node_velocities[node_id] * float(self.delta_time)
                previous_frontier_id = None
            else:
                mode = "continuous"
                fact = previous_facts.get(node_id)
                if fact is None or fact.for_source_step_index != source_step:
                    raise ValueError(
                        "continuous node lacks a current v5h5 frontier fact"
                    )
                birth = (
                    anchor
                    + (
                        np.asarray(fact.advected_position_gp1_m, dtype=np.float64)
                        - anchor
                    )
                    / 3.0
                )
                previous_frontier_id = fact.parent_frontier_id
            birth_id = (
                f"v5h5:{self._v5h5_wing_id}:node:{node_id}:"
                f"epoch:0:step:{source_step}"
            )
            births[node_id] = NodeBirthRecord(
                node_id=node_id,
                mode=mode,
                active=True,
                lineage_epoch=0,
                birth_step_index=source_step,
                anchor_node_id=f"v5h5:{self._v5h5_wing_id}:anchor:{node_id}",
                birth_node_id=birth_id,
                previous_frontier_id=previous_frontier_id,
                anchor_position_gp1_m=tuple(float(item) for item in anchor),
                birth_position_gp1_m=tuple(float(item) for item in birth),
            )

        bridge_nodes: dict[str, BridgeNode] = {}
        rings: list[DirectedRing] = []
        for cell, event, sign in sorted(
            validated_cells, key=lambda value: _id_key(value[0].cell_id)
        ):
            left = births[cell.left_node_id]
            right = births[cell.right_node_id]
            assert left.birth_node_id is not None and right.birth_node_id is not None
            assert left.birth_position_gp1_m is not None
            assert right.birth_position_gp1_m is not None
            for record in (left, right):
                bridge_nodes.setdefault(
                    record.birth_node_id,
                    BridgeNode(record.birth_node_id, record.birth_position_gp1_m),
                )
                bridge_nodes.setdefault(
                    record.anchor_node_id,
                    BridgeNode(record.anchor_node_id, record.anchor_position_gp1_m),
                )
            gamma = (
                sign
                * float(event.gamma_lev_new_over_u_c)
                * float(event.provenance.circulation_scale_u_times_c_m2_per_s)
            )
            if not np.isfinite(gamma):
                raise FloatingPointError("synchronized DVM circulation is non-finite")
            rings.append(
                DirectedRing(
                    ring_id=(
                        f"v5h5:{self._v5h5_wing_id}:cell:{cell.cell_id}:"
                        f"step:{source_step}"
                    ),
                    node_ids=(
                        left.birth_node_id,
                        right.birth_node_id,
                        right.anchor_node_id,
                        left.anchor_node_id,
                    ),
                    circulation=gamma,
                )
            )
        graph = _FROZEN_ASSEMBLE(tuple(bridge_nodes.values()), tuple(rings))
        deposited: DyadicBridgeResult = _FROZEN_DEPOSIT(
            graph,
            smoothing_radius=self._v5h5_sigma_birth,
            base_target_spacing=self._v5h5_base_spacing,
            refinement_level=self._v5h5_level,
            step=source_step,
        )
        _FROZEN_VALIDATE_BRIDGE(deposited)
        new_positions = np.asarray(deposited.positions, dtype=np.float64)
        new_gamma = np.asarray(deposited.gamma, dtype=np.float64)
        new_sigma = np.asarray(deposited.sigma, dtype=np.float64)
        old_positions = np.empty((0, 3), dtype=np.float64)
        old_gamma = np.empty((0, 3), dtype=np.float64)
        old_sigma = np.empty((0,), dtype=np.float64)
        old_ids: tuple[tuple[Any, ...], ...] = ()
        old_lineage: tuple[ParticleLineage, ...] = ()
        old_slices: tuple[SynchronizedReleaseSlice, ...] = ()
        parent = self._v5h5_previous_report
        if parent is not None:
            validated_parent = validate_synchronized_cloud_transport_report(parent)
            old_positions = np.asarray(validated_parent.transported_state.positions)
            old_gamma = np.asarray(validated_parent.transported_state.gamma)
            old_sigma = np.asarray(validated_parent.transported_state.sigma)
            old_ids = validated_parent.particle_ids
            old_lineage = validated_parent.lineage
            old_slices = validated_parent.release_slices
        positions = np.concatenate((old_positions, new_positions), axis=0)
        gamma = np.concatenate((old_gamma, new_gamma), axis=0)
        sigma = np.concatenate((old_sigma, new_sigma), axis=0)
        particle_ids = old_ids + tuple(deposited.particle_ids)
        lineage = old_lineage + tuple(deposited.lineage)
        old_count = len(old_ids)
        new_count = len(deposited.particle_ids)
        if old_count + new_count > MAX_PARTICLES:
            raise ValueError("synchronized cumulative particle cap exceeded")
        if len(set(particle_ids)) != len(particle_ids):
            raise ValueError("synchronized append produced duplicate particle IDs")
        append_residual = float(
            max(
                np.max(np.abs(positions[:old_count] - old_positions), initial=0.0),
                np.max(np.abs(gamma[:old_count] - old_gamma), initial=0.0),
                np.max(np.abs(sigma[:old_count] - old_sigma), initial=0.0),
            )
        )
        if append_residual != 0.0:
            raise RuntimeError("synchronized exact append changed the old prefix")
        graph_sha = _edge_graph_sha256(graph)
        release_slice = SynchronizedReleaseSlice(
            release_index=len(old_slices) + 1,
            source_step_index=source_step,
            source_time_s=layer.source_time_s,
            start_index=old_count,
            stop_index=old_count + new_count,
            particle_count=new_count,
            edge_graph_sha256=graph_sha,
            bridge_manifest_sha256=deposited.manifest_sha256,
            source_event_manifest_sha256=tuple(event_manifests),
        )
        prepared_positions = _readonly(positions, ndim=2)
        prepared_gamma = _readonly(gamma, ndim=2)
        prepared_sigma = _readonly(sigma, ndim=1)
        placeholder = PreparedSynchronizedCloud(
            enabled=True,
            interface_id=SYNCHRONIZED_INTERFACE_ID,
            wing_id=self._v5h5_wing_id,
            source_family=self._v5h5_source_family,
            ptera_step_index=ptera_step,
            for_source_step_index=source_step,
            transport_end_time_s=layer.source_time_s,
            positions_gp1_m=prepared_positions,
            gamma_vector_m3_per_s=prepared_gamma,
            sigma_m=prepared_sigma,
            particle_ids=particle_ids,
            lineage=lineage,
            release_slices=old_slices + (release_slice,),
            node_placement_manifest_sha256=placement.producer_manifest_sha256,
            node_kinematics=nodes,
            node_births=tuple(
                births[node_id] for node_id in sorted(births, key=_id_key)
            ),
            previous_particle_count=old_count,
            new_particle_count=new_count,
            total_particle_count=old_count + new_count,
            exact_append_prefix_max_abs=append_residual,
            report_sha256="0" * 64,
        )
        prepared = replace(placeholder, report_sha256=_prepared_digest(placeholder))
        for _, event, _ in validated_cells:
            _consume_event(event)
        self._v5h5_previous_event_by_cell.update(
            {cell.cell_id: event for cell, event, _ in validated_cells}
        )
        return prepared

    def _stage_feedback(self) -> None:
        if self._v5h3_pending is not None or self._v5h5_prepared is not None:
            raise RuntimeError("a previous synchronized layer was not committed")
        prepared = self._build_prepared_cloud()
        self.v5h5_prepared_history.append(prepared)
        self._v5h5_prepared = prepared
        if not self._v5h5_feedback_enabled:
            return
        parent = np.asarray(
            self._currentStackWakeWingInfluences__E, dtype=np.float64
        ).copy()
        empty = np.empty((0, 3), dtype=np.float64)
        empty.setflags(write=False)
        dummy = NativePteraRVPMVelocityEvaluation(
            channel="collocation_rhs",
            ptera_step_index=int(self._current_step),
            target_points_gp1_m=empty,
            induced_velocity_gp1_m_per_s=empty,
            target_sha256=_v5h3._array_sha256(empty),
            velocity_sha256=_v5h3._array_sha256(empty),
        )
        self._v5h3_pending = _PendingFeedbackStep(
            report=prepared,  # type: ignore[arg-type]
            positions=prepared.positions_gp1_m,
            gamma=prepared.gamma_vector_m3_per_s,
            sigma=prepared.sigma_m,
            parent_wake_normal=parent,
            feedback_normal=np.empty(0, dtype=np.float64),
            combined_wake_normal=np.empty(0, dtype=np.float64),
            collocation_evaluation=dummy,
            load_evaluations=[],
        )
        evaluation = self._field_velocity(
            self.stackCpp_GP1_CgP1,
            channel="collocation_rhs",
        )
        feedback_normal = np.einsum(
            "ij,ij->i",
            evaluation.induced_velocity_gp1_m_per_s,
            np.asarray(self.stackUnitNormals_GP1, dtype=np.float64),
        )
        combined = parent + feedback_normal
        if not np.all(np.isfinite(combined)):
            raise FloatingPointError("synchronized Ptera RHS is non-finite")
        self._v5h3_pending.collocation_evaluation = evaluation
        self._v5h3_pending.feedback_normal = feedback_normal.copy()
        self._v5h3_pending.combined_wake_normal = combined.copy()
        self._currentStackWakeWingInfluences__E = combined

    def _transport_prepared_cloud(
        self,
        prepared: PreparedSynchronizedCloud,
        feedback_report: NativePteraRVPMFeedbackStepReport | None,
    ) -> SynchronizedCloudTransportReport:
        initial = make_particle_state(
            prepared.positions_gp1_m,
            prepared.gamma_vector_m3_per_s,
            prepared.sigma_m,
        )
        reference_length = min(
            float(np.min(initial.sigma)),
            min(float(airplane.c_ref) for airplane in self.current_airplanes),
        )
        epsilon = self._v5h5_relative_epsilon * reference_length
        evaluations: list[PteraFieldEvaluation] = []

        def external(points: np.ndarray) -> FrozenExternalField:
            evaluation = evaluate_frozen_parent_ptera_field(
                self, points, epsilon_m=epsilon
            )
            evaluations.append(evaluation)
            return FrozenExternalField(
                evaluation.velocity_gp1_m_per_s,
                evaluation.jacobian_per_s,
            )

        before = ptera_parent_state_sha256(self)
        generic = lsrk3_step_with_external_field(
            initial,
            float(self.delta_time),
            external_field=external,
            baseline_freestream_velocity_gp1_m_per_s=np.asarray(
                self._currentVInf_GP1__E, dtype=np.float64
            ),
            enabled=True,
        )
        if len(generic.stages) != 3 or len(evaluations) != 3:
            raise RuntimeError("synchronized transport did not use three stages")
        transport_result = replace(
            generic,
            ptera_step_index=int(self._current_step),
            dvm_for_source_step_index=prepared.for_source_step_index,
            source_report_sha256=prepared.report_sha256,
            relative_epsilon=self._v5h5_relative_epsilon,
            epsilon_m=epsilon,
            stages=tuple(
                replace(stage, ptera_field=evaluation)
                for stage, evaluation in zip(generic.stages, evaluations, strict=True)
            ),
            ptera_center_call_count=3,
            ptera_finite_difference_call_count=18,
            ptera_parent_state_sha256_before=before,
        )

        tracers = np.asarray(
            [birth.birth_position_gp1_m for birth in prepared.node_births],
            dtype=np.float64,
        )
        initial_tracers = tracers.copy()
        storage = np.zeros_like(tracers)
        parent_velocity = _v5h4._assert_bindings()["parent_velocity"]
        for stage in transport_result.stages:
            self_field = _FROZEN_DIRECT_FIELD(
                stage.pre.positions,
                stage.pre.gamma,
                stage.pre.sigma,
                target_positions=tracers,
            ).velocity
            parent_field = parent_velocity(self, tracers)
            storage = stage.a * storage + float(self.delta_time) * (
                self_field + parent_field
            )
            tracers = tracers + stage.b * storage
        replay = initial_tracers.copy()
        replay_storage = np.zeros_like(replay)
        for stage in transport_result.stages:
            self_field = _FROZEN_DIRECT_FIELD(
                stage.pre.positions,
                stage.pre.gamma,
                stage.pre.sigma,
                target_positions=replay,
            ).velocity
            parent_field = parent_velocity(self, replay)
            replay_storage = stage.a * replay_storage + float(self.delta_time) * (
                self_field + parent_field
            )
            replay = replay + stage.b * replay_storage
        replay_residual = float(np.max(np.abs(tracers - replay), initial=0.0))
        if replay_residual != 0.0 or not np.all(np.isfinite(tracers)):
            raise RuntimeError("synchronized frontier stage replay failed")
        after = ptera_parent_state_sha256(self)
        if after != before:
            raise RuntimeError("synchronized transport mutated Ptera parent state")
        transport_result = replace(
            transport_result,
            ptera_parent_state_sha256_after=after,
            ptera_parent_state_unchanged=True,
        )
        final = make_particle_state(
            generic.final_state.positions,
            generic.final_state.gamma,
            generic.final_state.sigma,
        )
        final.positions.setflags(write=False)
        final.gamma.setflags(write=False)
        final.sigma.setflags(write=False)
        parent_report = self._v5h5_previous_report
        fact_cores: list[NodeFrontierFact] = []
        for birth, position in zip(prepared.node_births, tracers, strict=True):
            assert birth.birth_node_id is not None
            assert birth.birth_position_gp1_m is not None
            fact_cores.append(
                NodeFrontierFact(
                    wing_id=self._v5h5_wing_id,
                    node_id=birth.node_id,
                    source_family=self._v5h5_source_family,
                    lineage_epoch=int(birth.lineage_epoch),
                    parent_frontier_id=birth.birth_node_id,
                    parent_frontier_digest_sha256=parent_frontier_digest_sha256(
                        wing_id=self._v5h5_wing_id,
                        node_id=birth.node_id,
                        source_family=self._v5h5_source_family,
                        lineage_epoch=int(birth.lineage_epoch),
                        parent_frontier_id=birth.birth_node_id,
                        parent_birth_step_index=prepared.for_source_step_index,
                        parent_position_gp1_m=birth.birth_position_gp1_m,
                    ),
                    parent_ribbon_digest_sha256=prepared.report_sha256,
                    parent_birth_step_index=prepared.for_source_step_index,
                    for_source_step_index=prepared.for_source_step_index + 1,
                    time_layer=SYNCHRONIZED_TIME_LAYER,
                    transport_backend_id=_v5h4.RVPM_TRANSPORT_OWNER,
                    transport_artifact_sha256=_source_sha256(_v5h4),
                    producer_id=SYNCHRONIZED_OWNER,
                    producer_artifact_sha256=_producer_artifact_sha256(),
                    producer_report_sha256="0" * 64,
                    transport_start_time_s=prepared.transport_end_time_s,
                    transport_end_time_s=prepared.transport_end_time_s
                    + float(self.delta_time),
                    transport_substeps=1,
                    advected_position_gp1_m=tuple(float(item) for item in position),
                    fact_owner=SYNCHRONIZED_FACT_OWNER,
                )
            )
        placeholder = SynchronizedCloudTransportReport(
            enabled=True,
            interface_id=SYNCHRONIZED_INTERFACE_ID,
            wing_id=self._v5h5_wing_id,
            source_family=self._v5h5_source_family,
            parent_source_step_index=prepared.for_source_step_index,
            for_source_step_index=prepared.for_source_step_index + 1,
            time_layer=SYNCHRONIZED_TIME_LAYER,
            transport_start_time_s=prepared.transport_end_time_s,
            transport_end_time_s=prepared.transport_end_time_s + float(self.delta_time),
            parent_ribbon_digest_sha256=prepared.report_sha256,
            current_ribbon_digest_sha256=prepared.report_sha256,
            parent_report_sha256=(
                None if parent_report is None else parent_report.report_sha256
            ),
            prepared_report_sha256=prepared.report_sha256,
            report_sha256="0" * 64,
            prepared_cloud=prepared,
            transported_state=final,
            particle_ids=prepared.particle_ids,
            lineage=prepared.lineage,
            release_slices=prepared.release_slices,
            facts=tuple(fact_cores),
            ptera_feedback_report=feedback_report,
            transport_result=transport_result,
            previous_particle_count=prepared.previous_particle_count,
            new_particle_count=prepared.new_particle_count,
            total_particle_count=prepared.total_particle_count,
            release_call_count=1,
            ptera_feedback_enabled=self._v5h5_feedback_enabled,
            combined_transport_enabled=True,
            ptera_feedback_call_count=int(self._v5h5_feedback_enabled),
            combined_transport_call_count=1,
            ptera_center_call_count=3,
            ptera_finite_difference_call_count=18,
            frontier_transport_parent_call_count=3,
            frontier_replay_parent_call_count=3,
            frontier_stage_replay_max_abs=replay_residual,
            exact_append_prefix_max_abs=prepared.exact_append_prefix_max_abs,
            parent_state_sha256_before_transport=before,
            parent_state_sha256_after_transport=after,
            parent_state_unchanged=True,
            feedback_write_count=0,
            parent_write_count=0,
            load_write_count=0,
            observation_access=OBSERVATION_ACCESS,
            target_case_branch=TARGET_CASE_BRANCH,
            force_scoring_status=FORCE_SCORING_STATUS,
        )
        digest = _report_digest(placeholder)
        facts = tuple(
            replace(fact, producer_report_sha256=digest) for fact in fact_cores
        )
        report = replace(placeholder, facts=facts, report_sha256=digest)
        with _LOCK:
            _register_live(_REPORT_REGISTRY, report, digest)
        return validate_synchronized_cloud_transport_report(report)

    def _populate_next_airplanes_wake(self) -> None:
        prepared = self._v5h5_prepared
        if prepared is None:
            raise RuntimeError("v5h5 Ptera step has no prepared release cloud")
        super()._populate_next_airplanes_wake()
        feedback_report: NativePteraRVPMFeedbackStepReport | None = None
        if self._v5h5_feedback_enabled:
            if not self.v5h3_feedback_step_reports:
                raise RuntimeError("v5h5 Ptera feedback ledger was not committed")
            feedback_report = self.v5h3_feedback_step_reports[-1]
            if feedback_report.ptera_step_index != int(self._current_step):
                raise RuntimeError("v5h5 feedback ledger belongs to another Ptera step")
        if self._v5h5_transport_enabled:
            report = self._transport_prepared_cloud(prepared, feedback_report)
            self.v5h5_transport_reports.append(report)
            self._v5h5_previous_report = report
        self._v5h5_prepared = None


def make_fluxv_v5h5_synchronized_solver(
    unsteady_problem: Any,
    *,
    enabled: bool,
    release_layers: Sequence[SynchronizedReleaseLayer] | object = (),
    **kwargs: Any,
) -> UVPMHybridSolver:
    """Return the exact FluxV parent when disabled, otherwise v5h5."""

    if type(enabled) is not bool:
        raise TypeError("enabled must be a bool")
    if not enabled:
        return UVPMHybridSolver(unsteady_problem, **kwargs)
    if not isinstance(release_layers, Sequence):
        raise TypeError("enabled release_layers must be a finite sequence")
    return SynchronizedPteraRVPMCouplingSolver(
        unsteady_problem,
        release_layers=release_layers,
        **kwargs,
    )


__all__ = [
    "MAX_PARTICLES",
    "OBSERVATION_ACCESS",
    "SYNCHRONIZED_FACT_OWNER",
    "SYNCHRONIZED_INTERFACE_ID",
    "SYNCHRONIZED_OWNER",
    "SYNCHRONIZED_TIME_LAYER",
    "TARGET_CASE_BRANCH",
    "PreparedSynchronizedCloud",
    "SynchronizedCloudTransportReport",
    "SynchronizedPteraRVPMCouplingSolver",
    "SynchronizedReleaseLayer",
    "SynchronizedReleaseSlice",
    "make_fluxv_v5h5_synchronized_solver",
    "materialize_synchronized_particle_state",
    "validate_synchronized_cloud_transport_report",
]
