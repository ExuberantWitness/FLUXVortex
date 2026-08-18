"""Attested passive-node transport for v5h DVM ribbon frontiers.

The public producer never accepts a claimed final frontier position.  It
starts from active ``NodeBirthRecord`` positions in an actual ribbon result,
requires a conservative diagnostic particle deposition of that same edge
graph, advances the particles with the frozen direct rVPM LSRK3 backend, and
advances passive frontier tracers against every corresponding stage-pre
particle field.  The resulting immutable report is hash-bound to its inputs,
producer, transport backend, time interval, and output facts.

This is a one-way mechanical producer.  It has no Ptera parent, feedback, load,
target observation, source release, deletion, merge, or force channel.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from hashlib import sha256
import hmac
import json
from math import ceil, fsum
from numbers import Integral, Real
from pathlib import Path
import re
from typing import Any, Literal, TypeAlias
import weakref

import numpy as np
from numpy.typing import ArrayLike, NDArray

from fluxvortex.rvpm_edge_bridge import (
    BridgeDiagnostics,
    DIAGNOSTIC_SHADOW_OWNER,
    EdgeGraph,
    FROZEN_OVERLAP_LAMBDA,
    ParticleLineage,
    RING_PHYSICAL_OWNER,
    ShadowBridgeResult,
)
from fluxvortex.rvpm_reference import direct_gaussian_erf_velocity_jacobian
from fluxvortex.rvpm_transport import (
    ParticleState,
    lsrk3_step_direct,
    make_particle_state,
)


FloatArray = NDArray[np.float64]
StableId: TypeAlias = int | str
SourceFamily = Literal["lev", "tev_persisted"]

PASSIVE_FRONTIER_INTERFACE_ID = "fluxv-v5h-passive-frontier-report-v1"
PASSIVE_FRONTIER_PRODUCER_ID = "v5h-passive-frontier-direct-stage-aligned-lsrk3-v1"
PASSIVE_FRONTIER_TIME_LAYER = "pre_release_t_n"
PASSIVE_FRONTIER_FACT_OWNER = "rvpm_transport"
PASSIVE_FRONTIER_ATTESTATION_KIND = "same-process-weak-object-identity-v1"
PASSIVE_FRONTIER_CONTINUATION_SCOPE = (
    "single-release-only; step2-to-step3 requires a future "
    "transport_accumulated_particle_cloud exact-once merge API"
)
TRANSPORTED_PARTICLE_CLOUD_INTERFACE_ID = "fluxv-v5h-transported-particle-cloud-v1"
TRANSPORT_BACKEND_ID = (
    "fluxvortex.rvpm_transport.lsrk3_step_direct+"
    "gaussian_erf_passive_stage_replay-v1"
)
_FIXED_SIGMA_PARTICLE_SCHEMA = "rvpm-edge-shadow-fixed-sigma-v1"
_PRESCRIBED_SPACING_PARTICLE_SCHEMA = "rvpm-edge-shadow-prescribed-sigma-spacing-v1"


@dataclass(frozen=True)
class NodeFrontierFact:
    """One transported frontier fact bound to a producer report."""

    wing_id: StableId
    node_id: StableId
    source_family: SourceFamily
    lineage_epoch: int
    parent_frontier_id: str
    parent_frontier_digest_sha256: str
    parent_ribbon_digest_sha256: str
    parent_birth_step_index: int
    for_source_step_index: int
    time_layer: str
    transport_backend_id: str
    transport_artifact_sha256: str
    producer_id: str
    producer_artifact_sha256: str
    producer_report_sha256: str
    transport_start_time_s: float
    transport_end_time_s: float
    transport_substeps: int
    advected_position_gp1_m: tuple[float, float, float]
    fact_owner: str


@dataclass(frozen=True)
class TransportedParticleCloud:
    """Immutable particle/lineage handoff for a later release-layer merger."""

    interface_id: str
    positions_gp1_m: tuple[tuple[float, float, float], ...]
    gamma_vector_m3_per_s: tuple[tuple[float, float, float], ...]
    sigma_m: tuple[float, ...]
    particle_ids: tuple[tuple[Any, ...], ...]
    lineage: tuple[ParticleLineage, ...]


@dataclass(frozen=True)
class PassiveFrontierTransportReport:
    """Auditable result of an actual isolated particle/tracer transport."""

    interface_id: str
    producer_id: str
    producer_artifact_sha256: str
    transport_backend_id: str
    transport_artifact_sha256: str
    wing_id: StableId
    source_family: SourceFamily
    parent_source_step_index: int
    for_source_step_index: int
    time_layer: str
    transport_start_time_s: float
    transport_end_time_s: float
    transport_substeps: int
    freestream_velocity_gp1_m_per_s: tuple[float, float, float]
    parent_ribbon_digest_sha256: str
    deposited_cloud_digest_before_sha256: str
    transported_cloud_digest_after_sha256: str
    transported_particle_cloud: TransportedParticleCloud
    facts: tuple[NodeFrontierFact, ...]
    report_sha256: str
    attestation_kind: str
    continuation_scope: str
    feedback_call_count: int
    parent_write_count: int
    load_write_count: int
    observation_access: str
    target_case_branch: str


# Reports are trusted live handoffs only in the process that actually ran the
# passive transport.  Their public digest is an audit aid, not a signing key.
# The weak object-identity registry prevents ``dataclasses.replace`` (including
# a replacement with a recomputed private digest) from manufacturing a report.
_DIRECT_REPORT_REGISTRY: dict[
    int, tuple[weakref.ReferenceType[PassiveFrontierTransportReport], str]
] = {}


def _stable_id(name: str, value: object) -> StableId:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an explicit integer or string ID")
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, str) and value:
        return value
    raise ValueError(f"{name} must be a nonempty string or integer ID")


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


def _positive_integer(name: str, value: object) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be a positive integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _finite_vector(name: str, value: ArrayLike) -> FloatArray:
    original = np.asarray(value)
    if original.dtype.kind not in "iuf" or original.dtype.kind == "b":
        raise ValueError(f"{name} must use a real numeric dtype")
    result = np.asarray(original, dtype=np.float64)
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite length-3 vector")
    return result


def _sha256_file(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _producer_artifact_sha256() -> str:
    return _sha256_file(Path(__file__).resolve())


def _transport_artifact_sha256() -> str:
    import fluxvortex.rvpm_reference as reference_module
    import fluxvortex.rvpm_transport as transport_module

    digest = sha256()
    for label, module in (
        ("rvpm_reference", reference_module),
        ("rvpm_transport", transport_module),
    ):
        path = Path(module.__file__).resolve()
        digest.update(label.encode("ascii"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _lower_sha256(name: str, value: object) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _canonical_value(value: object) -> object:
    """Encode the stable scalar/tuple domain used by particle identities."""

    if value is None:
        return ["none"]
    if isinstance(value, (bool, np.bool_)):
        return ["boolean", bool(value)]
    if isinstance(value, Integral):
        return ["integer", int(value)]
    if isinstance(value, str):
        return ["string", value]
    if isinstance(value, Real):
        scalar = _finite_real("canonical real", value)
        return ["real-hex", scalar.hex()]
    if isinstance(value, (tuple, list)):
        return ["sequence", [_canonical_value(item) for item in value]]
    raise ValueError(
        f"particle identity contains unsupported value type {type(value).__name__}"
    )


def _hash_array(digest: Any, name: str, value: ArrayLike) -> None:
    array = np.ascontiguousarray(np.asarray(value, dtype="<f8"))
    digest.update(name.encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(array.shape, separators=(",", ":")).encode("ascii"))
    digest.update(b"\0")
    digest.update(array.tobytes())
    digest.update(b"\0")


def parent_frontier_digest_sha256(
    *,
    wing_id: StableId,
    node_id: StableId,
    source_family: SourceFamily,
    lineage_epoch: int,
    parent_frontier_id: str,
    parent_birth_step_index: int,
    parent_position_gp1_m: ArrayLike,
) -> str:
    """Return the canonical digest shared by the producer and consumer."""

    wing = _stable_id("wing_id", wing_id)
    node = _stable_id("node_id", node_id)
    if source_family not in ("lev", "tev_persisted"):
        raise ValueError("source_family is invalid")
    epoch = _positive_integer("lineage_epoch+1", lineage_epoch + 1) - 1
    step = _positive_integer("parent_birth_step_index", parent_birth_step_index)
    if not isinstance(parent_frontier_id, str) or not parent_frontier_id:
        raise ValueError("parent_frontier_id must be explicit")
    position = _finite_vector("parent_position_gp1_m", parent_position_gp1_m)
    payload = {
        "schema": "fluxv-v5h-parent-frontier-digest-v1",
        "wing_id": _id_payload(wing),
        "node_id": _id_payload(node),
        "source_family": source_family,
        "lineage_epoch": epoch,
        "parent_frontier_id": parent_frontier_id,
        "parent_birth_step_index": step,
        "parent_position_hex": [float(item).hex() for item in position],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256(encoded).hexdigest()


def ribbon_parent_digest_sha256(result: object) -> str:
    """Digest the source/ribbon layer that owns transported frontiers."""

    diagnostics = getattr(result, "diagnostics")
    births = tuple(getattr(result, "node_births"))
    ledgers = tuple(getattr(result, "cell_ledgers"))
    graph = getattr(result, "edge_graph")
    payload = {
        "schema": "fluxv-v5h-parent-ribbon-digest-v1",
        "diagnostics": {
            field.name: _canonical_value(getattr(diagnostics, field.name))
            for field in fields(diagnostics)
        },
        "births": [
            {
                "node_id": _id_payload(_stable_id("node_id", item.node_id)),
                "mode": item.mode,
                "active": item.active,
                "lineage_epoch": item.lineage_epoch,
                "birth_step_index": item.birth_step_index,
                "anchor_node_id": item.anchor_node_id,
                "birth_node_id": item.birth_node_id,
                "previous_frontier_id": item.previous_frontier_id,
                "anchor_position_hex": [
                    float(value).hex() for value in item.anchor_position_gp1_m
                ],
                "birth_position_hex": (
                    None
                    if item.birth_position_gp1_m is None
                    else [float(value).hex() for value in item.birth_position_gp1_m]
                ),
            }
            for item in births
        ],
        "cell_ledgers": [
            {
                "cell_id": _id_payload(_stable_id("cell_id", item.cell_id)),
                "physical_strip_id": item.physical_strip_id,
                "source_lineage_id": item.source_lineage_id,
                "source_step_index": item.source_step_index,
                "source_family": item.source_family,
                "active": item.active,
                "gamma_star_hex": float(item.gamma_star).hex(),
                "circulation_scale_hex": float(
                    item.circulation_scale_u_times_c_m2_per_s
                ).hex(),
                "gamma_cell_hex": float(item.gamma_cell_m2_per_s).hex(),
                "traversal_sign": item.circulation_to_ring_traversal_sign,
                "ring_gamma_hex": float(item.ring_circulation_m2_per_s).hex(),
                "kelvin_residual_hex": float(item.kelvin_residual_m2_per_s).hex(),
                "section_birth_audit_point_hex": (
                    None
                    if item.section_birth_audit_point_gp1_m is None
                    else [
                        float(value).hex()
                        for value in item.section_birth_audit_point_gp1_m
                    ]
                ),
                "section_birth_used": item.section_birth_point_used_for_topology,
                "ring_id": item.ring_id,
            }
            for item in ledgers
        ],
        "edge_graph": (
            None
            if graph is None
            else {
                "nodes": [
                    {
                        "node_id": _id_payload(
                            _stable_id("graph node_id", node.node_id)
                        ),
                        "position_hex": [float(value).hex() for value in node.position],
                    }
                    for node in graph.nodes
                ],
                "edges": [
                    {
                        "key": _canonical_value(edge.key),
                        "start_hex": [
                            float(value).hex() for value in edge.start_position
                        ],
                        "end_hex": [float(value).hex() for value in edge.end_position],
                        "incidences": _canonical_value(_incidence_signature(edge)),
                        "circulation_hex": float(edge.circulation).hex(),
                        "vector_moment_hex": [
                            float(value).hex() for value in edge.vector_moment
                        ],
                    }
                    for edge in graph.edges
                ],
                "incidence_residual_hex": float(graph.incidence_residual).hex(),
                "edge_reconstruction_residual_hex": float(
                    graph.edge_reconstruction_residual
                ).hex(),
                "global_vector_moment_hex": [
                    float(value).hex() for value in graph.global_vector_moment
                ],
            }
        ),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256(encoded).hexdigest()


def _cloud_digest(bridge: ShadowBridgeResult) -> str:
    digest = sha256()
    digest.update(b"fluxv-v5h-deposited-cloud-v1\0")
    _hash_array(digest, "positions", bridge.positions)
    _hash_array(digest, "gamma", bridge.gamma)
    _hash_array(digest, "sigma", bridge.sigma)
    identity_payload = {
        "particle_ids": [
            _canonical_value(particle_id) for particle_id in bridge.particle_ids
        ],
        "lineage": [_lineage_payload(item) for item in bridge.lineage],
    }
    digest.update(
        json.dumps(identity_payload, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    )
    return digest.hexdigest()


def _particle_state_digest(state: object) -> str:
    digest = sha256()
    digest.update(b"fluxv-v5h-transported-cloud-v1\0")
    _hash_array(digest, "positions", getattr(state, "positions"))
    _hash_array(digest, "gamma", getattr(state, "gamma"))
    _hash_array(digest, "sigma", getattr(state, "sigma"))
    return digest.hexdigest()


def _lineage_payload(lineage: ParticleLineage) -> dict[str, object]:
    return {
        "particle_id": _canonical_value(lineage.particle_id),
        "source_edge": _canonical_value(lineage.source_edge),
        "subdivision_index": lineage.subdivision_index,
        "subdivision_count": lineage.subdivision_count,
        "ring_incidences": _canonical_value(
            tuple(
                (
                    item.ring_id,
                    item.traversal_index,
                    item.source_start_id,
                    item.source_end_id,
                    item.canonical_sign,
                    item.ring_circulation,
                    item.signed_circulation,
                )
                for item in lineage.ring_incidences
            )
        ),
        "step": lineage.step,
        "physical_owner": lineage.physical_owner,
        "owner_state": lineage.owner_state,
    }


def _transported_cloud_payload(
    cloud: TransportedParticleCloud,
) -> dict[str, object]:
    return {
        "interface_id": cloud.interface_id,
        "positions_hex": [
            [float(value).hex() for value in row] for row in cloud.positions_gp1_m
        ],
        "gamma_hex": [
            [float(value).hex() for value in row] for row in cloud.gamma_vector_m3_per_s
        ],
        "sigma_hex": [float(value).hex() for value in cloud.sigma_m],
        "particle_ids": [
            _canonical_value(particle_id) for particle_id in cloud.particle_ids
        ],
        "lineage": [_lineage_payload(item) for item in cloud.lineage],
    }


def _cloud_arrays(
    cloud: TransportedParticleCloud,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    if not isinstance(cloud, TransportedParticleCloud):
        raise ValueError("transported particle cloud has a foreign schema")
    if cloud.interface_id != TRANSPORTED_PARTICLE_CLOUD_INTERFACE_ID:
        raise ValueError("transported particle cloud has a foreign interface")
    positions = np.asarray(cloud.positions_gp1_m, dtype=np.float64)
    gamma = np.asarray(cloud.gamma_vector_m3_per_s, dtype=np.float64)
    sigma = np.asarray(cloud.sigma_m, dtype=np.float64)
    count = len(cloud.particle_ids)
    if positions.shape != (count, 3) or gamma.shape != (count, 3):
        raise ValueError("transported particle cloud vector shapes are invalid")
    if sigma.shape != (count,):
        raise ValueError("transported particle cloud sigma shape is invalid")
    if len(cloud.lineage) != count:
        raise ValueError("transported particle cloud lineage count is invalid")
    if count == 0:
        raise ValueError("transported particle cloud must not be empty")
    if not (
        np.all(np.isfinite(positions))
        and np.all(np.isfinite(gamma))
        and np.all(np.isfinite(sigma))
    ):
        raise ValueError("transported particle cloud contains non-finite values")
    if np.any(sigma <= 0.0):
        raise ValueError("transported particle cloud contains nonpositive sigma")
    if len(set(cloud.particle_ids)) != count:
        raise ValueError("transported particle cloud contains duplicate particle IDs")
    for index, lineage in enumerate(cloud.lineage):
        if not isinstance(lineage, ParticleLineage):
            raise ValueError("transported particle cloud contains foreign lineage")
        if lineage.particle_id != cloud.particle_ids[index]:
            raise ValueError("transported particle cloud lineage/ID order disagrees")
        _lineage_payload(lineage)
    return (
        np.ascontiguousarray(positions),
        np.ascontiguousarray(gamma),
        np.ascontiguousarray(sigma),
    )


def _fact_core_payload(fact: NodeFrontierFact) -> dict[str, object]:
    return {
        "wing_id": _id_payload(_stable_id("fact.wing_id", fact.wing_id)),
        "node_id": _id_payload(_stable_id("fact.node_id", fact.node_id)),
        "source_family": fact.source_family,
        "lineage_epoch": fact.lineage_epoch,
        "parent_frontier_id": fact.parent_frontier_id,
        "parent_frontier_digest_sha256": fact.parent_frontier_digest_sha256,
        "parent_ribbon_digest_sha256": fact.parent_ribbon_digest_sha256,
        "parent_birth_step_index": fact.parent_birth_step_index,
        "for_source_step_index": fact.for_source_step_index,
        "time_layer": fact.time_layer,
        "transport_backend_id": fact.transport_backend_id,
        "transport_artifact_sha256": fact.transport_artifact_sha256,
        "producer_id": fact.producer_id,
        "producer_artifact_sha256": fact.producer_artifact_sha256,
        "transport_start_time_hex": fact.transport_start_time_s.hex(),
        "transport_end_time_hex": fact.transport_end_time_s.hex(),
        "transport_substeps": fact.transport_substeps,
        "advected_position_hex": [
            float(value).hex() for value in fact.advected_position_gp1_m
        ],
        "fact_owner": fact.fact_owner,
    }


def _report_payload(
    report: PassiveFrontierTransportReport,
) -> dict[str, object]:
    return {
        "interface_id": report.interface_id,
        "producer_id": report.producer_id,
        "producer_artifact_sha256": report.producer_artifact_sha256,
        "transport_backend_id": report.transport_backend_id,
        "transport_artifact_sha256": report.transport_artifact_sha256,
        "wing_id": _id_payload(_stable_id("report.wing_id", report.wing_id)),
        "source_family": report.source_family,
        "parent_source_step_index": report.parent_source_step_index,
        "for_source_step_index": report.for_source_step_index,
        "time_layer": report.time_layer,
        "transport_start_time_hex": report.transport_start_time_s.hex(),
        "transport_end_time_hex": report.transport_end_time_s.hex(),
        "transport_substeps": report.transport_substeps,
        "freestream_velocity_hex": [
            float(value).hex() for value in report.freestream_velocity_gp1_m_per_s
        ],
        "parent_ribbon_digest_sha256": report.parent_ribbon_digest_sha256,
        "deposited_cloud_digest_before_sha256": (
            report.deposited_cloud_digest_before_sha256
        ),
        "transported_cloud_digest_after_sha256": (
            report.transported_cloud_digest_after_sha256
        ),
        "transported_particle_cloud": _transported_cloud_payload(
            report.transported_particle_cloud
        ),
        "facts": [_fact_core_payload(fact) for fact in report.facts],
        "attestation_kind": report.attestation_kind,
        "continuation_scope": report.continuation_scope,
        "feedback_call_count": report.feedback_call_count,
        "parent_write_count": report.parent_write_count,
        "load_write_count": report.load_write_count,
        "observation_access": report.observation_access,
        "target_case_branch": report.target_case_branch,
    }


def _report_digest(report: PassiveFrontierTransportReport) -> str:
    encoded = json.dumps(
        _report_payload(report), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def _attest_direct_report(
    report: PassiveFrontierTransportReport,
) -> PassiveFrontierTransportReport:
    digest = _report_digest(report)
    if not hmac.compare_digest(report.report_sha256, digest):
        raise ValueError("frontier report cannot be attested with a bad digest")
    report_id = id(report)

    def discard(
        reference: weakref.ReferenceType[PassiveFrontierTransportReport],
    ) -> None:
        existing = _DIRECT_REPORT_REGISTRY.get(report_id)
        if existing is not None and existing[0] is reference:
            _DIRECT_REPORT_REGISTRY.pop(report_id, None)

    reference = weakref.ref(report, discard)
    _DIRECT_REPORT_REGISTRY[report_id] = (reference, digest)
    return report


def _strict_float64_array(
    name: str,
    value: object,
    shape: tuple[int, ...],
) -> FloatArray:
    if not isinstance(value, np.ndarray):
        raise ValueError(f"{name} must be a NumPy array")
    if value.dtype != np.dtype(np.float64):
        raise ValueError(f"{name} must use Float64 storage")
    if value.shape != shape:
        raise ValueError(f"{name} has an invalid shape")
    if not value.flags.c_contiguous:
        raise ValueError(f"{name} must use contiguous storage")
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{name} contains non-finite values")
    return value


def _stable_vector_sum(values: ArrayLike) -> FloatArray:
    """Mirror the bridge's componentwise audit-grade circulation sum."""

    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return np.zeros(3, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError("stable vector sum requires an (n, 3) array")
    return np.asarray(
        [fsum(float(value) for value in array[:, axis]) for axis in range(3)],
        dtype=np.float64,
    )


def _incidence_signature(edge: object) -> tuple[tuple[object, ...], ...]:
    return tuple(
        (
            item.ring_id,
            item.traversal_index,
            item.source_start_id,
            item.source_end_id,
            item.canonical_sign,
            item.ring_circulation,
            item.signed_circulation,
        )
        for item in getattr(edge, "incidences")
    )


def _validate_fixed_sigma_deposition(
    graph: EdgeGraph,
    bridge: ShadowBridgeResult,
    *,
    expected_step: int,
    deposition_target_spacing_m: object | None = None,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Reconstruct a uniform-sigma midpoint cloud from the edge ledger.

    The legacy fixed-sigma schema derives its maximum spacing from
    ``sigma / FROZEN_OVERLAP_LAMBDA`` and therefore requires
    ``deposition_target_spacing_m is None``.  The prescribed-spacing schema
    requires an explicit spacing and independently recomputes every edge
    count from ``ceil(edge_length / spacing)``.

    The frozen bridge does not encode the requested spacing itself.  Two
    spacings that produce the same count on every edge are consequently the
    same observable discrete state and are accepted as equivalent; any
    spacing that changes a count is rejected by the exact ledger replay.
    """

    if bridge.edge_graph != graph:
        raise ValueError("deposited cloud is not bound to the parent edge graph")
    count = len(bridge.particle_ids)
    positions = _strict_float64_array(
        "deposited positions", bridge.positions, (count, 3)
    )
    gamma = _strict_float64_array("deposited gamma", bridge.gamma, (count, 3))
    sigma = _strict_float64_array("deposited sigma", bridge.sigma, (count,))
    if count == 0:
        raise ValueError("fixed-sigma transport requires a nonempty particle cloud")
    if len(bridge.lineage) != count:
        raise ValueError("deposited particle and lineage counts disagree")
    if len(set(bridge.particle_ids)) != count:
        raise ValueError("deposited cloud contains duplicate particle IDs")
    if np.any(sigma <= 0.0) or not np.all(sigma == sigma[0]):
        raise ValueError("transport requires one positive uniform fixed sigma")
    smoothing_radius = float(sigma[0])

    particle_schemas: set[str] = set()
    for particle_id in bridge.particle_ids:
        if (
            not isinstance(particle_id, tuple)
            or not particle_id
            or not isinstance(particle_id[0], str)
        ):
            raise ValueError("deposited cloud has an invalid particle schema")
        particle_schemas.add(particle_id[0])
    if len(particle_schemas) != 1:
        raise ValueError("deposited cloud mixes particle schemas")
    particle_schema = next(iter(particle_schemas))

    if particle_schema == _FIXED_SIGMA_PARTICLE_SCHEMA:
        if deposition_target_spacing_m is not None:
            raise ValueError(
                "fixed-sigma deposition requires deposition_target_spacing_m=None"
            )
        target_spacing = smoothing_radius / FROZEN_OVERLAP_LAMBDA
        if not np.isfinite(target_spacing) or target_spacing <= 0.0:
            raise ValueError("fixed-sigma spacing is invalid")
    elif particle_schema == _PRESCRIBED_SPACING_PARTICLE_SCHEMA:
        if deposition_target_spacing_m is None:
            raise ValueError(
                "prescribed-spacing deposition requires deposition_target_spacing_m"
            )
        target_spacing = _finite_real(
            "deposition_target_spacing_m", deposition_target_spacing_m
        )
        if target_spacing <= 0.0:
            raise ValueError("deposition_target_spacing_m must be strictly positive")
        requested_overlap = smoothing_radius / target_spacing
        if (
            not np.isfinite(requested_overlap)
            or requested_overlap < FROZEN_OVERLAP_LAMBDA
        ):
            raise ValueError(
                "deposition_target_spacing_m is too large for the frozen "
                "minimum overlap"
            )
    else:
        raise ValueError("deposited cloud is not a supported fixed-sigma schema")

    expected_positions: list[FloatArray] = []
    expected_gamma: list[FloatArray] = []
    expected_ids: list[tuple[Any, ...]] = []
    expected_lineage: list[ParticleLineage] = []
    edge_errors_abs: list[float] = []
    edge_errors_rel: list[float] = []
    for edge in graph.edges:
        if not edge.retained:
            continue
        edge_start_index = len(expected_ids)
        start = np.asarray(edge.start_position, dtype=np.float64)
        end = np.asarray(edge.end_position, dtype=np.float64)
        delta = end - start
        edge_length = float(np.linalg.norm(delta))
        count_float = edge_length / target_spacing
        if not np.isfinite(count_float):
            raise ValueError("deposition spacing produced a non-finite particle count")
        edge_count = max(1, ceil(count_float))
        if particle_schema == _PRESCRIBED_SPACING_PARTICLE_SCHEMA:
            realized_spacing = edge_length / edge_count
            realized_overlap = smoothing_radius / realized_spacing
            if (
                not np.isfinite(realized_overlap)
                or realized_overlap < FROZEN_OVERLAP_LAMBDA
            ):
                raise ValueError(
                    "prescribed-spacing edge violates the frozen minimum overlap"
                )
        segment = delta / edge_count
        signature = _incidence_signature(edge)
        for subdivision_index in range(edge_count):
            particle_id = (
                particle_schema,
                edge.key,
                subdivision_index,
                edge_count,
                smoothing_radius,
                signature,
                expected_step,
                DIAGNOSTIC_SHADOW_OWNER,
            )
            expected_positions.append(start + (subdivision_index + 0.5) * segment)
            expected_gamma.append(edge.circulation * segment)
            expected_ids.append(particle_id)
            expected_lineage.append(
                ParticleLineage(
                    particle_id=particle_id,
                    source_edge=edge.key,
                    subdivision_index=subdivision_index,
                    subdivision_count=edge_count,
                    ring_incidences=edge.incidences,
                    step=expected_step,
                    physical_owner=RING_PHYSICAL_OWNER,
                    owner_state=DIAGNOSTIC_SHADOW_OWNER,
                )
            )
        edge_stop_index = len(expected_ids)
        actual_edge_vector = _stable_vector_sum(gamma[edge_start_index:edge_stop_index])
        expected_edge_vector = np.asarray(edge.vector_moment, dtype=np.float64)
        edge_absolute = float(np.linalg.norm(actual_edge_vector - expected_edge_vector))
        edge_reference = float(np.linalg.norm(expected_edge_vector))
        edge_relative = (
            edge_absolute / edge_reference
            if edge_reference > 0.0
            else (0.0 if edge_absolute == 0.0 else float("inf"))
        )
        edge_errors_abs.append(edge_absolute)
        edge_errors_rel.append(edge_relative)

    if len(expected_ids) != count:
        raise ValueError("fixed-sigma particle count is not reproducible")
    expected_position_array = np.ascontiguousarray(
        np.vstack(expected_positions), dtype=np.float64
    )
    expected_gamma_array = np.ascontiguousarray(
        np.vstack(expected_gamma), dtype=np.float64
    )
    if not np.array_equal(positions, expected_position_array):
        raise ValueError("fixed-sigma particle midpoint ledger is inconsistent")
    if not np.array_equal(gamma, expected_gamma_array):
        raise ValueError("fixed-sigma particle gamma-vector ledger is inconsistent")
    if tuple(bridge.particle_ids) != tuple(expected_ids):
        raise ValueError("fixed-sigma particle ID ledger is inconsistent")
    if tuple(bridge.lineage) != tuple(expected_lineage):
        raise ValueError("fixed-sigma particle lineage ledger is inconsistent")

    expected_global = np.asarray(graph.global_vector_moment, dtype=np.float64)
    actual_global = _stable_vector_sum(gamma)
    global_absolute = float(np.linalg.norm(actual_global - expected_global))
    reference_scale = fsum(
        float(np.linalg.norm(np.asarray(edge.vector_moment, dtype=np.float64)))
        for edge in graph.retained_edges
    )
    global_relative = (
        global_absolute / reference_scale
        if reference_scale > 0.0
        else (0.0 if global_absolute == 0.0 else float("inf"))
    )
    max_edge_absolute = max(edge_errors_abs, default=0.0)
    max_edge_relative = max(edge_errors_rel, default=0.0)
    diagnostics = bridge.diagnostics
    if not isinstance(diagnostics, BridgeDiagnostics):
        raise ValueError("deposited cloud has foreign diagnostics")
    if (
        not diagnostics.enabled
        or diagnostics.source_edge_count != len(graph.edges)
        or diagnostics.retained_edge_count != len(graph.retained_edges)
        or diagnostics.particle_count != count
        or diagnostics.incidence_residual != graph.incidence_residual
        or diagnostics.edge_reconstruction_residual
        != graph.edge_reconstruction_residual
        or not np.isfinite(diagnostics.max_edge_conservation_abs)
        or not np.isfinite(diagnostics.max_edge_conservation_rel)
        or not np.isfinite(diagnostics.global_conservation_abs)
        or not np.isfinite(diagnostics.global_conservation_rel)
        or diagnostics.max_edge_conservation_abs != max_edge_absolute
        or diagnostics.max_edge_conservation_rel != max_edge_relative
        or diagnostics.global_conservation_abs != global_absolute
        or diagnostics.global_conservation_rel != global_relative
        or max_edge_absolute > 1.0e-14
        or max_edge_relative > 1.0e-12
        or diagnostics.clip_count
        or diagnostics.nonfinite_count
        or diagnostics.owner_conflict_count
        or diagnostics.feedback_call_count
        or bridge.feedback_velocity is not None
    ):
        raise ValueError("deposited cloud diagnostics are not reproducible")
    return positions, gamma, sigma


def validate_passive_frontier_transport_report(
    report: object,
) -> PassiveFrontierTransportReport:
    """Validate a live report produced by this process's real transport path."""

    if not isinstance(report, PassiveFrontierTransportReport):
        raise ValueError("frontier transport requires a PassiveFrontierTransportReport")
    if report.interface_id != PASSIVE_FRONTIER_INTERFACE_ID:
        raise ValueError("frontier report has a foreign interface")
    if report.attestation_kind != PASSIVE_FRONTIER_ATTESTATION_KIND:
        raise ValueError("frontier report has a foreign attestation kind")
    if report.continuation_scope != PASSIVE_FRONTIER_CONTINUATION_SCOPE:
        raise ValueError("frontier report overclaims its continuation scope")
    if report.producer_id != PASSIVE_FRONTIER_PRODUCER_ID:
        raise ValueError("frontier report has a cross-producer identity")
    producer_hash = _lower_sha256(
        "report.producer_artifact_sha256", report.producer_artifact_sha256
    )
    if producer_hash != _producer_artifact_sha256():
        raise ValueError("frontier producer artifact hash is stale or foreign")
    if report.transport_backend_id != TRANSPORT_BACKEND_ID:
        raise ValueError("frontier report has a cross-backend identity")
    transport_hash = _lower_sha256(
        "report.transport_artifact_sha256", report.transport_artifact_sha256
    )
    if transport_hash != _transport_artifact_sha256():
        raise ValueError("frontier transport artifact hash is stale or foreign")
    if report.source_family not in ("lev", "tev_persisted"):
        raise ValueError("frontier report source family is invalid")
    _stable_id("report.wing_id", report.wing_id)
    parent_step = _positive_integer(
        "report.parent_source_step_index", report.parent_source_step_index
    )
    for_step = _positive_integer(
        "report.for_source_step_index", report.for_source_step_index
    )
    if for_step != parent_step + 1:
        raise ValueError("frontier report source steps are stale or nonconsecutive")
    if report.time_layer != PASSIVE_FRONTIER_TIME_LAYER:
        raise ValueError("frontier report has the wrong time layer")
    start = _finite_real("report.transport_start_time_s", report.transport_start_time_s)
    end = _finite_real("report.transport_end_time_s", report.transport_end_time_s)
    if end <= start:
        raise ValueError("frontier report transport interval is not positive")
    substeps = _positive_integer("report.transport_substeps", report.transport_substeps)
    _finite_vector(
        "report.freestream_velocity_gp1_m_per_s",
        report.freestream_velocity_gp1_m_per_s,
    )
    if report.observation_access != "none" or report.target_case_branch != "none":
        raise ValueError("frontier report is not isolated from target observations")
    if (
        report.feedback_call_count
        or report.parent_write_count
        or report.load_write_count
    ):
        raise ValueError("frontier report contains feedback or parent/load writes")
    if not report.facts:
        raise ValueError("frontier report contains no transported facts")
    parent_ribbon_digest = _lower_sha256(
        "report.parent_ribbon_digest_sha256",
        report.parent_ribbon_digest_sha256,
    )
    _lower_sha256(
        "report.deposited_cloud_digest_before_sha256",
        report.deposited_cloud_digest_before_sha256,
    )
    transported_digest = _lower_sha256(
        "report.transported_cloud_digest_after_sha256",
        report.transported_cloud_digest_after_sha256,
    )
    transported_positions, transported_gamma, transported_sigma = _cloud_arrays(
        report.transported_particle_cloud
    )
    materialized = make_particle_state(
        transported_positions,
        transported_gamma,
        transported_sigma,
    )
    if transported_digest != _particle_state_digest(materialized):
        raise ValueError("transported particle cloud digest is inconsistent")
    node_ids: set[tuple[str, int | str]] = set()
    for fact in report.facts:
        if not isinstance(fact, NodeFrontierFact):
            raise ValueError("frontier report contains a foreign fact")
        key = _id_key(_stable_id("fact.node_id", fact.node_id))
        if key in node_ids:
            raise ValueError("frontier report contains duplicate node facts")
        node_ids.add(key)
        if (
            fact.wing_id != report.wing_id
            or fact.source_family != report.source_family
            or fact.parent_ribbon_digest_sha256 != parent_ribbon_digest
            or fact.parent_birth_step_index != parent_step
            or fact.for_source_step_index != for_step
            or fact.time_layer != report.time_layer
            or fact.transport_backend_id != report.transport_backend_id
            or fact.transport_artifact_sha256 != report.transport_artifact_sha256
            or fact.producer_id != report.producer_id
            or fact.producer_artifact_sha256 != report.producer_artifact_sha256
            or fact.transport_start_time_s != start
            or fact.transport_end_time_s != end
            or fact.transport_substeps != substeps
            or fact.fact_owner != PASSIVE_FRONTIER_FACT_OWNER
        ):
            raise ValueError("frontier fact disagrees with its producer report")
        _positive_integer("fact.lineage_epoch+1", fact.lineage_epoch + 1)
        _lower_sha256(
            "fact.parent_frontier_digest_sha256",
            fact.parent_frontier_digest_sha256,
        )
        _lower_sha256(
            "fact.parent_ribbon_digest_sha256",
            fact.parent_ribbon_digest_sha256,
        )
        _finite_vector("fact.advected_position_gp1_m", fact.advected_position_gp1_m)
        if fact.producer_report_sha256 != report.report_sha256:
            raise ValueError("frontier fact has a foreign report digest")
    report_digest = _lower_sha256("report.report_sha256", report.report_sha256)
    if not hmac.compare_digest(_report_digest(report), report_digest):
        raise ValueError("frontier report digest cannot be independently reproduced")
    registered = _DIRECT_REPORT_REGISTRY.get(id(report))
    if (
        registered is None
        or registered[0]() is not report
        or not hmac.compare_digest(registered[1], report_digest)
    ):
        raise ValueError(
            "frontier report is not a directly produced live transport object"
        )
    return report


def materialize_transported_particle_state(
    report: object,
) -> ParticleState:
    """Copy the attested transported cloud for a future release-layer merge."""

    validated = validate_passive_frontier_transport_report(report)
    positions, gamma, sigma = _cloud_arrays(validated.transported_particle_cloud)
    return make_particle_state(positions, gamma, sigma)


def transport_passive_node_frontiers(
    parent_ribbon: object,
    deposited_shadow: ShadowBridgeResult,
    *,
    wing_id: StableId,
    transport_start_time_s: object,
    transport_end_time_s: object,
    transport_substeps: object,
    deposition_target_spacing_m: object | None = None,
    freestream_velocity_gp1_m_per_s: ArrayLike = (0.0, 0.0, 0.0),
) -> PassiveFrontierTransportReport:
    """Actually transport every active birth node and attest the result.

    The producer accepts neither output positions nor a node subset: every
    active parent frontier is transported exactly once.  Missing/duplicate
    facts can therefore arise only from an untrusted copied/tampered report,
    which the live-object attestation rejects.

    ``deposition_target_spacing_m`` is absent for the legacy fixed-sigma
    bridge schema and mandatory for the prescribed-sigma/spacing schema.  In
    the latter case it is used to independently replay every edge count and
    the complete midpoint, circulation, identity, lineage, and diagnostics
    ledgers before transport begins.
    """

    # The local import avoids a schema/producer import cycle at module load.
    from forward_flight_benchmarks.v5h_dvm_node_ribbon import (
        DVMRibbonShadowResult,
        validate_live_dvm_ribbon_shadow_result,
    )

    if not isinstance(parent_ribbon, DVMRibbonShadowResult):
        raise ValueError("parent_ribbon must be an actual DVMRibbonShadowResult")
    parent_ribbon = validate_live_dvm_ribbon_shadow_result(parent_ribbon)
    if not isinstance(deposited_shadow, ShadowBridgeResult):
        raise ValueError("deposited_shadow must be an actual ShadowBridgeResult")
    diagnostics = parent_ribbon.diagnostics
    if (
        not diagnostics.enabled
        or parent_ribbon.edge_graph is None
        or parent_ribbon.feedback_velocity is not None
        or diagnostics.feedback_call_count
    ):
        raise ValueError(
            "parent ribbon is outside the clean single-release v1 scope; "
            "step2-to-step3 requires the cumulative-cloud v2 API"
        )
    if diagnostics.source_family not in ("lev", "tev_persisted"):
        raise ValueError("parent ribbon source family is invalid")
    parent_step = _positive_integer("parent source step", diagnostics.source_step_index)
    if parent_step != 1 or diagnostics.transport_advance_count != 0:
        raise ValueError(
            "single-release v1 requires an untransported source-step-1 parent; "
            "step2-to-step3 requires the cumulative-cloud v2 API"
        )
    wing = _stable_id("wing_id", wing_id)
    (
        deposited_positions,
        deposited_gamma,
        deposited_sigma,
    ) = _validate_fixed_sigma_deposition(
        parent_ribbon.edge_graph,
        deposited_shadow,
        expected_step=parent_step,
        deposition_target_spacing_m=deposition_target_spacing_m,
    )

    start = _finite_real("transport_start_time_s", transport_start_time_s)
    end = _finite_real("transport_end_time_s", transport_end_time_s)
    if end <= start:
        raise ValueError("transport interval must be positive")
    count = _positive_integer("transport_substeps", transport_substeps)
    freestream = _finite_vector(
        "freestream_velocity_gp1_m_per_s",
        freestream_velocity_gp1_m_per_s,
    )
    active_births = {
        _stable_id("birth.node_id", birth.node_id): birth
        for birth in parent_ribbon.node_births
        if birth.active
    }
    if not active_births:
        raise ValueError("parent ribbon has no active frontier births")
    selected_ids = tuple(sorted(active_births, key=_id_key))

    seeds: list[FloatArray] = []
    seed_metadata: list[dict[str, Any]] = []
    for node_id in selected_ids:
        birth = active_births[node_id]
        if (
            birth.lineage_epoch is None
            or birth.birth_step_index != parent_step
            or birth.birth_node_id is None
            or birth.birth_position_gp1_m is None
        ):
            raise ValueError("active parent birth has incomplete frontier identity")
        position = _finite_vector(
            "birth.birth_position_gp1_m", birth.birth_position_gp1_m
        )
        seeds.append(position)
        seed_metadata.append(
            {
                "node_id": node_id,
                "lineage_epoch": int(birth.lineage_epoch),
                "parent_frontier_id": birth.birth_node_id,
                "parent_birth_step_index": int(birth.birth_step_index),
                "parent_position": position,
                "parent_digest": parent_frontier_digest_sha256(
                    wing_id=wing,
                    node_id=node_id,
                    source_family=diagnostics.source_family,
                    lineage_epoch=int(birth.lineage_epoch),
                    parent_frontier_id=birth.birth_node_id,
                    parent_birth_step_index=int(birth.birth_step_index),
                    parent_position_gp1_m=position,
                ),
            }
        )

    tracer_positions = np.ascontiguousarray(np.vstack(seeds), dtype=np.float64)
    particle_state = make_particle_state(
        deposited_positions,
        deposited_gamma,
        deposited_sigma,
    )
    delta_time = (end - start) / count
    for _ in range(count):
        next_particle_state, stages = lsrk3_step_direct(
            particle_state,
            delta_time,
            freestream_velocity=freestream,
        )
        tracer_storage = np.zeros_like(tracer_positions)
        for stage in stages:
            field = direct_gaussian_erf_velocity_jacobian(
                stage.pre.positions,
                stage.pre.gamma,
                stage.pre.sigma,
                target_positions=tracer_positions,
            )
            tracer_storage = stage.a * tracer_storage + delta_time * (
                field.velocity + freestream[None, :]
            )
            tracer_positions = tracer_positions + stage.b * tracer_storage
            if not np.all(np.isfinite(tracer_positions)):
                raise FloatingPointError(
                    "passive frontier transport produced a non-finite position"
                )
        particle_state = next_particle_state

    producer_hash = _producer_artifact_sha256()
    transport_hash = _transport_artifact_sha256()
    parent_ribbon_digest = ribbon_parent_digest_sha256(parent_ribbon)
    fact_cores: list[NodeFrontierFact] = []
    placeholder_digest = "0" * 64
    for metadata, position in zip(seed_metadata, tracer_positions, strict=True):
        fact_cores.append(
            NodeFrontierFact(
                wing_id=wing,
                node_id=metadata["node_id"],
                source_family=diagnostics.source_family,
                lineage_epoch=metadata["lineage_epoch"],
                parent_frontier_id=metadata["parent_frontier_id"],
                parent_frontier_digest_sha256=metadata["parent_digest"],
                parent_ribbon_digest_sha256=parent_ribbon_digest,
                parent_birth_step_index=parent_step,
                for_source_step_index=parent_step + 1,
                time_layer=PASSIVE_FRONTIER_TIME_LAYER,
                transport_backend_id=TRANSPORT_BACKEND_ID,
                transport_artifact_sha256=transport_hash,
                producer_id=PASSIVE_FRONTIER_PRODUCER_ID,
                producer_artifact_sha256=producer_hash,
                producer_report_sha256=placeholder_digest,
                transport_start_time_s=start,
                transport_end_time_s=end,
                transport_substeps=count,
                advected_position_gp1_m=tuple(float(value) for value in position),
                fact_owner=PASSIVE_FRONTIER_FACT_OWNER,
            )
        )

    transported_cloud = TransportedParticleCloud(
        interface_id=TRANSPORTED_PARTICLE_CLOUD_INTERFACE_ID,
        positions_gp1_m=tuple(
            tuple(float(value) for value in row) for row in particle_state.positions
        ),
        gamma_vector_m3_per_s=tuple(
            tuple(float(value) for value in row) for row in particle_state.gamma
        ),
        sigma_m=tuple(float(value) for value in particle_state.sigma),
        particle_ids=tuple(deposited_shadow.particle_ids),
        lineage=tuple(deposited_shadow.lineage),
    )
    placeholder_report = PassiveFrontierTransportReport(
        interface_id=PASSIVE_FRONTIER_INTERFACE_ID,
        producer_id=PASSIVE_FRONTIER_PRODUCER_ID,
        producer_artifact_sha256=producer_hash,
        transport_backend_id=TRANSPORT_BACKEND_ID,
        transport_artifact_sha256=transport_hash,
        wing_id=wing,
        source_family=diagnostics.source_family,
        parent_source_step_index=parent_step,
        for_source_step_index=parent_step + 1,
        time_layer=PASSIVE_FRONTIER_TIME_LAYER,
        transport_start_time_s=start,
        transport_end_time_s=end,
        transport_substeps=count,
        freestream_velocity_gp1_m_per_s=tuple(float(value) for value in freestream),
        parent_ribbon_digest_sha256=parent_ribbon_digest,
        deposited_cloud_digest_before_sha256=_cloud_digest(deposited_shadow),
        transported_cloud_digest_after_sha256=_particle_state_digest(particle_state),
        transported_particle_cloud=transported_cloud,
        facts=tuple(fact_cores),
        report_sha256=placeholder_digest,
        attestation_kind=PASSIVE_FRONTIER_ATTESTATION_KIND,
        continuation_scope=PASSIVE_FRONTIER_CONTINUATION_SCOPE,
        feedback_call_count=0,
        parent_write_count=0,
        load_write_count=0,
        observation_access="none",
        target_case_branch="none",
    )
    report_digest = _report_digest(placeholder_report)
    facts = tuple(
        replace(fact, producer_report_sha256=report_digest) for fact in fact_cores
    )
    report = replace(
        placeholder_report,
        facts=facts,
        report_sha256=report_digest,
    )
    # The report digest excludes only its own digest field, while fact payloads
    # also exclude their copy of that report digest.  Revalidation is therefore
    # non-circular and deterministic.
    _attest_direct_report(report)
    return validate_passive_frontier_transport_report(report)


__all__ = [
    "NodeFrontierFact",
    "PASSIVE_FRONTIER_ATTESTATION_KIND",
    "PASSIVE_FRONTIER_CONTINUATION_SCOPE",
    "PASSIVE_FRONTIER_FACT_OWNER",
    "PASSIVE_FRONTIER_INTERFACE_ID",
    "PASSIVE_FRONTIER_PRODUCER_ID",
    "PASSIVE_FRONTIER_TIME_LAYER",
    "PassiveFrontierTransportReport",
    "TRANSPORT_BACKEND_ID",
    "TRANSPORTED_PARTICLE_CLOUD_INTERFACE_ID",
    "TransportedParticleCloud",
    "materialize_transported_particle_state",
    "parent_frontier_digest_sha256",
    "ribbon_parent_digest_sha256",
    "transport_passive_node_frontiers",
    "validate_passive_frontier_transport_report",
]
