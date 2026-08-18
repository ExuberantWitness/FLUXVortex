"""Audited no-GT executor for the V5H12 execution-only repair.

The runner is intentionally not imported here.  It injects the exact sink and
STOP class identities through ``build_fluxv_v5h12_w2_executor`` only after its
external dependency preflight.  Runtime scientific imports are deferred until
their manifest leaves have been rehashed, then every imported source is bound
back to its canonical leaf path and digest before a source or solver is built.

This module contains no observation-data or scoring path.  Its public helpers
are deliberately small so the evidence conversion and rollback semantics can
be tested without importing or running Ptera.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
from hashlib import sha256
from importlib import import_module, metadata as importlib_metadata
import importlib.abc
import importlib.machinery
import importlib.util
import json
import math
from pathlib import Path
import sys
from types import MappingProxyType, ModuleType
from typing import Callable, Final, Mapping, Sequence

import numpy as np


EXECUTOR_API_SCHEMA_ID: Final = "fluxv-v5h12-baik-w2-formal-executor-api-v1"
DISPOSABLE_SMOKE_SUMMARY_SCHEMA_ID: Final = (
    "fluxv-v5h12-baik-w2-disposable-n32-layer1-summary-v1"
)
FORMAL_LEVELS: Final = (32, 64, 128)
DISPOSABLE_SMOKE_LEVEL: Final = FORMAL_LEVELS[0]
DISPOSABLE_SMOKE_LAYER_LIMIT: Final = 1
LAYERS: Final = (1, 2, 3)
SOURCE_STEPS: Final = (4, 5, 6)
PTERA_STEPS: Final = (3, 4, 5)
SOURCE_CELL_COUNT: Final = 8
ALL_SOURCE_STEPS: Final = (1, 2, 3, 4, 5, 6)
ACTIVE_BIRTH_MODES: Final = ("first", "continuous", "continuous")
DELTA_TIME_S: Final = 0.11125
STEPS_PER_CYCLE: Final = 32
ROW_PARTICLE_CAP: Final = 1_000_000
ROW_SMOOTHING_RADIUS_M: Final = 0.00152
ROW_TARGET_SPACING_M: Final = 0.0007152941176470589
ROW_SHEET_ID: Final = "baik-w2-global-lev-row"
MAX_SOURCE_KELVIN_M2_PER_S: Final = 1.0e-10
INVARIANT_NORMALIZED_GATE: Final = 512.0 * np.finfo(np.float64).eps
STAGE_EVIDENCE_SCHEMA: Final = "v5h11-stage-fd-stability-v1"
STAGE_CHAIN_GENESIS: Final = sha256(b"fluxv-ir-wrk3-stream-stage-chain-v1").hexdigest()
SOURCE_INTERFACE_ID: Final = "fluxv-v5h-dvm-source-only-v3"
SOURCE_PLACEMENT_SCHEMA_ID: Final = "fluxv-v5h-dvm-source-placement-v3"
SOURCE_BACKEND_ID: Final = (
    "platform.ldvm_fourier.LDVM2D-source-parity-clean-linear-provisional-tev-v3"
)
SOURCE_EVENT_CHAIN_DOMAIN: Final = "fluxv-v5h-dvm-event-chain-v3"
SOURCE_EVENT_DIGEST_PREFIX: Final = b"fluxv-v5h-direct-dvm-event-v3\0"
SOURCE_PREHISTORY_MANIFEST_DOMAIN: Final = "fluxv-v5h11-source-prehistory-manifests-v1"

RK_A: Final = (0.0, -5.0 / 9.0, -153.0 / 128.0)
RK_B: Final = (1.0 / 3.0, 15.0 / 16.0, 8.0 / 15.0)

SOURCE_EVENT_FIELDS: Final = (
    "transport_substeps",
    "source_step_index",
    "ptera_step_index",
    "source_time_s",
    "cell_count",
    "status",
    "birth_modes",
    "a0_pre",
    "a0_post",
    "gamma_lev_new_m2_s",
    "gamma_lev_persisted_m2_s",
    "gamma_tev_new_m2_s",
    "kelvin_residual_m2_s",
    "event_sha256",
    "parent_event_sha256",
)

SOURCE_CELL_MANIFEST_FIELDS: Final = (
    "enabled",
    "status",
    "delta_time_convective",
    "a0_pre",
    "a0_post",
    "lesp_critical",
    "lesp_active",
    "lesp_signed_target",
    "lesp_constraint_residual",
    "lev_birth_mode",
    "restart",
    "gamma_lev_new_over_u_c",
    "gamma_tev_new_solved_over_u_c",
    "gamma_tev_new_persisted_over_u_c",
    "lev_placement",
    "tev_placement",
    "lev_birth_position_over_chord_backend_world",
    "tev_birth_position_over_chord_backend_world",
    "kelvin_residual_over_u_c",
    "kelvin_ledger",
    "lineage",
    "provenance",
    "parent_event_manifest_sha256",
    "producer_manifest_sha256",
)

STAGE_FIELDS: Final = (
    "transport_substeps",
    "layer",
    "source_step_index",
    "ptera_step_index",
    "substep",
    "stage",
    "status",
    "substep_delta_time",
    "rk_a",
    "rk_b",
    "pre_state_sha256",
    "post_state_sha256",
    "tracer_pre_sha256",
    "tracer_post_sha256",
    "velocity_sha256",
    "jacobian_sha256",
    "gamma_rate_sha256",
    "tracer_velocity_sha256",
    "invariant_residual_sha256",
    "invariant_residual_over_slog_max",
    "h_jacobian_frobenius",
    "h_convective_over_sigma",
    "position_storage_pre_sha256",
    "gamma_storage_pre_sha256",
    "tracer_storage_pre_sha256",
    "position_storage_post_sha256",
    "gamma_storage_post_sha256",
    "tracer_storage_post_sha256",
    "fd_physical_evaluation_sha256",
    "fd_tracer_evaluation_sha256",
    "ptera_parent_sha256_before",
    "ptera_parent_sha256_after",
    "direct_field_call_count",
    "ptera_center_call_count",
    "ptera_offset_call_count",
    "stream_record_sha256",
    "previous_chain_sha256",
    "chain_sha256",
    "failure_type",
    "failure_message",
)

RAW_STEP_FIELDS: Final = (
    "transport_substeps",
    "layer",
    "source_step_index",
    "ptera_step_index",
    "status",
    "particle_count",
    "material_tracer_count",
    "material_support_tracer_count",
    "frontier_node_tracer_count",
    "stream_result_sha256",
    "stream_stage_chain_sha256",
    "fd_ledger_sha256",
    "load_ledger_sha256",
    "layer_result_sha256",
    "row_owner_before_sha256",
    "advanced_owner_sha256",
    "ptera_parent_sha256_before",
    "ptera_parent_sha256_after",
    "no_penetration_max_abs",
    "kelvin_residual_max_abs",
    "raw_cl",
    "raw_cd",
    "direct_field_call_count",
    "ptera_center_call_count",
    "ptera_offset_call_count",
    "fd_physical_evaluation_count",
    "fd_tracer_evaluation_count",
    "fd_evaluator_call_count",
    "transport_substep_count",
    "transport_stage_count",
    "physical_field_call_count",
    "tracer_field_call_count",
    "observer_call_count",
    "stage_pre_reconstruction_count",
    "stage_post_reconstruction_count",
    "physical_rhs_call_count",
    "storage_reset_count",
    "tracer_storage_reset_count",
    "invariant_reference_freeze_count",
    "sigma_storage_update_count",
    "relaxation_call_count",
    "compact_stage_record_count",
    "retained_stage_array_count",
    "evidence_byte_count",
    "native_collocation_evaluation_count",
    "native_load_batch_evaluation_count",
    "native_load_call_count",
    "transport_macro_call_count",
    "row_commit_count",
)

OWNER_EVENT_FIELDS: Final = (
    "schema_id",
    "transport_substeps",
    "layer",
    "source_step_index",
    "ptera_step_index",
    "status",
    "row_owner_before_sha256",
    "row_state_before_sha256",
    "common_transport_sha256",
    "transport_attestation_sha256",
    "transport_parent_digest",
    "advanced_owner_sha256",
    "advanced_state_sha256",
    "changed_particle_ids",
    "appended_particle_ids",
    "commit_event",
    "transport_event",
)

PARTICLE_COUNT_FIELDS: Final = (
    "transport_substeps",
    "layer",
    "status",
    "particle_count",
    "material_tracer_count",
    "material_support_tracer_count",
    "frontier_node_tracer_count",
    "changed_particle_count",
    "appended_particle_count",
)

RAW_LOAD_FIELDS: Final = (
    "transport_substeps",
    "layer",
    "scope",
    "panel_id",
    "force_x_n",
    "force_y_n",
    "force_z_n",
    "moment_x_nm",
    "moment_y_nm",
    "moment_z_nm",
    "force_coefficient_x",
    "force_coefficient_y",
    "force_coefficient_z",
    "raw_cl",
    "raw_cd",
)

REQUIRED_TRAJECTORY_ARRAYS: Final = (
    "start_positions",
    "start_gamma",
    "start_sigma",
    "end_positions",
    "end_gamma",
    "end_sigma",
    "material_tracer_positions",
    "frontier_tracer_positions",
    "probe_velocity",
    "probe_jacobian",
    "force",
    "moment",
    "invariant_start",
    "invariant_end",
    "no_penetration_residual",
)

MODULE_LEAVES: Final = MappingProxyType(
    {
        "rvpm_ir_wrk3": "fluxvortex.rvpm_ir_wrk3",
        "rvpm_ir_wrk3_fd_adapter": "fluxvortex.rvpm_ir_wrk3_fd_adapter",
        "rvpm_ir_wrk3_stream": "fluxvortex.rvpm_ir_wrk3_stream",
        "rvpm_reference": "fluxvortex.rvpm_reference",
        "rvpm_edge_bridge": "fluxvortex.rvpm_edge_bridge",
        "rvpm_dyadic_edge_bridge": "fluxvortex.rvpm_dyadic_edge_bridge",
        "rvpm_transport": "fluxvortex.rvpm_transport",
        "fluxvortex_kernel": "fluxvortex.kernel",
        "fluxvortex_particles": "fluxvortex.particles",
        "fluxvortex_solver_source": "fluxvortex.solver",
        "v5h10_row_owner": ("forward_flight_benchmarks.fluxv_v5h10_row_owner"),
        "v5h11_baik_coupling": ("forward_flight_benchmarks.fluxv_v5h11_baik_coupling"),
        "v5h3_native_feedback": (
            "forward_flight_benchmarks.fluxv_v5h3_native_feedback"
        ),
        "v5h4_ptera_transport": (
            "forward_flight_benchmarks.fluxv_v5h4_ptera_rvpm_transport"
        ),
        "v5h2_dyadic_cumulative_cloud_transport": (
            "forward_flight_benchmarks.v5h2_dyadic_cumulative_cloud_transport"
        ),
        "v5h_cumulative_cloud_transport": (
            "forward_flight_benchmarks.v5h_cumulative_cloud_transport"
        ),
        "v5h_dvm_source": "forward_flight_benchmarks.v5h_dvm_source",
        "uvlm_correction": ("forward_flight_benchmarks.ldvm_uvlm_correction"),
        "ldvm_fourier": "ldvm_fourier",
        "flap_ldvm": "flap_ldvm",
        "pterasoftware_solver_source": (
            "pterasoftware.unsteady_ring_vortex_lattice_method"
        ),
    }
)

DISTRIBUTION_LEAVES: Final = MappingProxyType(
    {
        "pterasoftware_distribution_metadata": "pterasoftware",
        "fluxvortex_distribution_metadata": "fluxvortex",
        "numpy_distribution_metadata": "numpy",
        "scipy_distribution_metadata": "scipy",
    }
)


class ExecutorContractError(RuntimeError):
    """The injected ABI or a durable scientific binding is inconsistent."""


class DependencyBindingError(ExecutorContractError):
    """An imported runtime dependency differs from its verified manifest leaf."""


@dataclass(frozen=True, slots=True)
class DisposableLayer1SmokeSummary:
    """Scalar-only, non-persistent evidence from the bounded N32 diagnostic."""

    schema_id: str
    case_id: str
    scope: str
    transport_substeps: int
    active_layer_limit: int
    layer: int
    source_step_index: int
    ptera_step_index: int
    source_event_sha256: str
    source_cell_manifest_sha256: str
    source_prehistory_manifest_sha256: str
    layer_result_sha256: str
    stream_result_sha256: str
    stream_stage_chain_sha256: str
    fd_ledger_sha256: str
    load_ledger_sha256: str
    source_kelvin_evidence_sha256: str
    row_owner_before_sha256: str
    advanced_owner_sha256: str
    advanced_state_sha256: str
    ptera_parent_sha256_before: str
    ptera_parent_sha256_after: str
    particle_count: int
    material_tracer_count: int
    material_support_tracer_count: int
    frontier_node_tracer_count: int
    transport_stage_count: int
    direct_field_call_count: int
    ptera_center_call_count: int
    ptera_offset_call_count: int
    fd_physical_evaluation_count: int
    fd_tracer_evaluation_count: int
    fd_evaluator_call_count: int
    max_invariant_residual_over_slog: float
    max_h_jacobian_frobenius: float
    max_h_convective_over_sigma: float
    sigma_min_m: float
    sigma_max_m: float
    no_penetration_max_abs: float
    kelvin_residual_max_abs_m2_s: float
    observation_access: str
    force_scoring_status: str
    artifact_persistence: str
    summary_sha256: str


def _is_sha256(value: object) -> bool:
    return bool(
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _disposable_smoke_summary_payload(
    value: DisposableLayer1SmokeSummary,
) -> dict[str, object]:
    payload = asdict(value)
    payload.pop("summary_sha256")
    for name in (
        "max_invariant_residual_over_slog",
        "max_h_jacobian_frobenius",
        "max_h_convective_over_sigma",
        "sigma_min_m",
        "sigma_max_m",
        "no_penetration_max_abs",
        "kelvin_residual_max_abs_m2_s",
    ):
        payload[name] = float(payload[name]).hex()
    return payload


def _disposable_smoke_summary_sha256(
    value: DisposableLayer1SmokeSummary,
    _payload: Callable[[DisposableLayer1SmokeSummary], dict[str, object]] = (
        _disposable_smoke_summary_payload
    ),
    _json_bytes: Callable[[object], bytes] = _canonical_json_bytes,
    _hash_constructor: Callable[[bytes], object] = sha256,
) -> str:
    digest = _hash_constructor(
        DISPOSABLE_SMOKE_SUMMARY_SCHEMA_ID.encode("ascii")
        + b"\0"
        + _json_bytes(_payload(value))
    )
    return digest.hexdigest()  # type: ignore[no-any-return, union-attr]


def validate_disposable_layer1_smoke_summary(
    value: DisposableLayer1SmokeSummary,
    _summary_sha256: Callable[[DisposableLayer1SmokeSummary], str] = (
        _disposable_smoke_summary_sha256
    ),
) -> DisposableLayer1SmokeSummary:
    """Revalidate the exact no-artifact diagnostic completion contract."""

    if type(value) is not DisposableLayer1SmokeSummary:
        raise ExecutorContractError(
            "disposable smoke summary must have the exact frozen type"
        )
    if (
        value.schema_id != DISPOSABLE_SMOKE_SUMMARY_SCHEMA_ID
        or value.case_id != "W2"
        or value.scope != "diagnostic_smoke"
        or value.transport_substeps != DISPOSABLE_SMOKE_LEVEL
        or value.active_layer_limit != DISPOSABLE_SMOKE_LAYER_LIMIT
        or value.layer != 1
        or value.source_step_index != SOURCE_STEPS[0]
        or value.ptera_step_index != PTERA_STEPS[0]
        or value.observation_access != "none"
        or value.force_scoring_status != "blocked_no_gt_inner_mechanics_only"
        or value.artifact_persistence != "none"
    ):
        raise ExecutorContractError("disposable smoke summary scope drift")
    digest_fields = (
        "source_event_sha256",
        "source_cell_manifest_sha256",
        "source_prehistory_manifest_sha256",
        "layer_result_sha256",
        "stream_result_sha256",
        "stream_stage_chain_sha256",
        "fd_ledger_sha256",
        "load_ledger_sha256",
        "source_kelvin_evidence_sha256",
        "row_owner_before_sha256",
        "advanced_owner_sha256",
        "advanced_state_sha256",
        "ptera_parent_sha256_before",
        "ptera_parent_sha256_after",
    )
    if any(not _is_sha256(getattr(value, name)) for name in digest_fields):
        raise ExecutorContractError("disposable smoke summary has an invalid digest")
    if value.ptera_parent_sha256_before != value.ptera_parent_sha256_after:
        raise ExecutorContractError("disposable smoke mutated the Ptera parent")
    expected_stages = 3 * DISPOSABLE_SMOKE_LEVEL
    count_fields = (
        value.transport_stage_count,
        value.direct_field_call_count,
        value.ptera_center_call_count,
        value.ptera_offset_call_count,
        value.fd_physical_evaluation_count,
        value.fd_tracer_evaluation_count,
        value.fd_evaluator_call_count,
    )
    if any(type(count) is not int for count in count_fields) or count_fields != (
        expected_stages,
        2 * expected_stages,
        2 * expected_stages,
        6 * expected_stages,
        expected_stages,
        expected_stages,
        8 * expected_stages,
    ):
        raise ExecutorContractError("disposable smoke stage/FD counts drift")
    particle_counts = (
        value.particle_count,
        value.material_tracer_count,
        value.material_support_tracer_count,
        value.frontier_node_tracer_count,
    )
    if (
        any(type(count) is not int or count < 0 for count in particle_counts)
        or value.particle_count < value.material_support_tracer_count
        or value.material_tracer_count
        != value.material_support_tracer_count + value.frontier_node_tracer_count
        or value.frontier_node_tracer_count != 9
    ):
        raise ExecutorContractError("disposable smoke particle/tracer counts drift")
    measurements = (
        value.max_invariant_residual_over_slog,
        value.max_h_jacobian_frobenius,
        value.max_h_convective_over_sigma,
        value.sigma_min_m,
        value.sigma_max_m,
        value.no_penetration_max_abs,
        value.kelvin_residual_max_abs_m2_s,
    )
    if any(type(item) is not float or not math.isfinite(item) for item in measurements):
        raise ExecutorContractError("disposable smoke measurements are not finite")
    if (
        value.max_invariant_residual_over_slog < 0.0
        or value.max_invariant_residual_over_slog > INVARIANT_NORMALIZED_GATE
        or value.max_h_jacobian_frobenius < 0.0
        or value.max_h_jacobian_frobenius > 1.5
        or value.max_h_convective_over_sigma < 0.0
        or value.max_h_convective_over_sigma > 0.5
        or value.sigma_min_m <= 0.0
        or value.sigma_max_m < value.sigma_min_m
        or value.no_penetration_max_abs < 0.0
        or value.no_penetration_max_abs > 1.0e-12
        or value.kelvin_residual_max_abs_m2_s < 0.0
        or abs(value.kelvin_residual_max_abs_m2_s) > MAX_SOURCE_KELVIN_M2_PER_S
    ):
        raise ExecutorContractError("disposable smoke mechanical gate failed")
    if value.summary_sha256 != _summary_sha256(value):
        raise ExecutorContractError("disposable smoke summary digest mismatch")
    return value


def _strict_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> object:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _strict_ascii_json_object(payload: object) -> dict[str, object]:
    if type(payload) is not bytes:
        raise TypeError("stage evidence payload must be exact bytes")
    try:
        text = payload.decode("ascii")
    except UnicodeDecodeError as error:
        raise ValueError("stage evidence payload must be ASCII JSON") from error
    value = json.loads(
        text,
        object_pairs_hook=_strict_pairs,
        parse_constant=_reject_constant,
    )
    if type(value) is not dict:
        raise ValueError("stage evidence payload must be a JSON object")
    if _canonical_json_bytes(value) != payload:
        raise ValueError("stage evidence payload is not canonical JSON")
    return value


def _hex_float(name: str, value: object) -> float:
    if type(value) is not str:
        raise ValueError(f"{name} must be an exact hexadecimal float string")
    try:
        result = float.fromhex(value)
    except ValueError as error:
        raise ValueError(f"{name} is not a hexadecimal float") from error
    if not math.isfinite(result) or result < 0.0 or result.hex() != value:
        raise ValueError(f"{name} is not a canonical finite non-negative float")
    return result


def _freeze_array(name: str, value: object, shape_tail: tuple[int, ...]) -> np.ndarray:
    original = np.asarray(value)
    if original.dtype.kind not in "iuf" or original.dtype.kind == "b":
        raise TypeError(f"{name} must have a real numeric dtype")
    result = np.array(original, dtype=np.float64, order="C", copy=True)
    if (
        result.ndim != len(shape_tail) + 1
        or result.shape[1:] != shape_tail
        or not np.all(np.isfinite(result))
    ):
        raise ValueError(f"{name} has invalid shape or non-finite values")
    frozen = np.frombuffer(result.tobytes(order="C"), dtype=np.float64).reshape(
        result.shape
    )
    return frozen


def _freeze_vector(name: str, value: object, length: int) -> np.ndarray:
    original = np.asarray(value)
    if original.dtype.kind not in "iuf" or original.dtype.kind == "b":
        raise TypeError(f"{name} must have a real numeric dtype")
    result = np.array(original, dtype=np.float64, order="C", copy=True)
    if result.shape != (length,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must have exact shape ({length},) and be finite")
    return np.frombuffer(result.tobytes(order="C"), dtype=np.float64)


@dataclass(frozen=True, slots=True)
class ExecutorAPIContract:
    sink_type: type
    stop_constructor: type
    leaf_paths: Mapping[str, str]
    leaf_hashes: Mapping[str, str]
    runtime_module_paths: Mapping[str, str]
    runtime_module_hashes: Mapping[str, str]
    frontier_node_ids: tuple[str, ...]
    fixed_probes: tuple[tuple[float, float, float], ...]


def _validated_api(api: object) -> ExecutorAPIContract:
    required = (
        "schema_id",
        "formal_levels",
        "artifact_sink_type",
        "stop_constructor",
        "source_event_fields",
        "stage_fields",
        "frontier_node_ids",
        "fixed_probes_gp1_m",
        "dependency_leaf_file_path",
        "dependency_leaf_file_sha256",
        "dependency_runtime_module_file_path",
        "dependency_runtime_module_file_sha256",
    )
    missing = tuple(name for name in required if not hasattr(api, name))
    if missing:
        raise ExecutorContractError(
            f"injected executor API is missing fields: {missing}"
        )
    if getattr(api, "schema_id") != EXECUTOR_API_SCHEMA_ID:
        raise ExecutorContractError("injected executor API schema drift")
    if tuple(getattr(api, "formal_levels")) != FORMAL_LEVELS:
        raise ExecutorContractError("formal level contract drift")
    if tuple(getattr(api, "source_event_fields")) != SOURCE_EVENT_FIELDS:
        raise ExecutorContractError("source-event schema drift")
    if tuple(getattr(api, "stage_fields")) != STAGE_FIELDS:
        raise ExecutorContractError("stage schema drift")
    sink_type = getattr(api, "artifact_sink_type")
    stop_constructor = getattr(api, "stop_constructor")
    if type(sink_type) is not type or type(stop_constructor) is not type:
        raise ExecutorContractError(
            "injected sink/STOP constructors must be exact types"
        )
    frontier = tuple(getattr(api, "frontier_node_ids"))
    if (
        len(frontier) != 9
        or any(type(item) is not str or not item for item in frontier)
        or len(set(frontier)) != len(frontier)
    ):
        raise ExecutorContractError("frontier node IDs are not the frozen nine-ID pack")
    probes = tuple(tuple(point) for point in getattr(api, "fixed_probes_gp1_m"))
    if len(probes) != 3 or any(
        len(point) != 3
        or any(type(value) is not float or not math.isfinite(value) for value in point)
        for point in probes
    ):
        raise ExecutorContractError("fixed probe ABI is invalid")
    path_map = getattr(api, "dependency_leaf_file_path")
    hash_map = getattr(api, "dependency_leaf_file_sha256")
    if not isinstance(path_map, Mapping) or not isinstance(hash_map, Mapping):
        raise ExecutorContractError("dependency leaf maps must be mappings")
    paths: dict[str, str] = {}
    hashes: dict[str, str] = {}
    if set(path_map) != set(hash_map):
        raise ExecutorContractError("dependency leaf path/hash keys disagree")
    for name in path_map:
        path = path_map[name]
        digest = hash_map[name]
        if type(name) is not str or not name or type(path) is not str or not path:
            raise ExecutorContractError("dependency leaf map contains invalid text")
        if not _is_sha256(digest):
            raise ExecutorContractError(f"dependency leaf digest is invalid: {name}")
        paths[name] = path
        hashes[name] = digest
    required_leaves = set(MODULE_LEAVES) | set(DISTRIBUTION_LEAVES)
    if not required_leaves.issubset(paths):
        missing_leaves = sorted(required_leaves - set(paths))
        raise ExecutorContractError(
            f"injected dependency map lacks runtime leaves: {missing_leaves}"
        )
    runtime_path_map = getattr(api, "dependency_runtime_module_file_path")
    runtime_hash_map = getattr(api, "dependency_runtime_module_file_sha256")
    if not isinstance(runtime_path_map, Mapping) or not isinstance(
        runtime_hash_map, Mapping
    ):
        raise ExecutorContractError("runtime-module origin maps must be mappings")
    if set(runtime_path_map) != set(runtime_hash_map):
        raise ExecutorContractError("runtime-module origin path/hash keys disagree")
    runtime_paths: dict[str, str] = {}
    runtime_hashes: dict[str, str] = {}
    for module_name in runtime_path_map:
        path = runtime_path_map[module_name]
        digest = runtime_hash_map[module_name]
        if (
            type(module_name) is not str
            or not module_name
            or type(path) is not str
            or not path
            or not _is_sha256(digest)
        ):
            raise ExecutorContractError("runtime-module origin map is invalid")
        if not (
            module_name in ("ldvm_fourier", "flap_ldvm")
            or module_name == "fluxvortex"
            or module_name.startswith("fluxvortex.")
            or module_name == "forward_flight_benchmarks"
            or module_name.startswith("forward_flight_benchmarks.")
            or module_name == "pterasoftware"
            or module_name.startswith("pterasoftware.")
        ):
            raise ExecutorContractError(
                f"runtime-module inventory has a foreign namespace: {module_name}"
            )
        runtime_paths[module_name] = path
        runtime_hashes[module_name] = digest
    return ExecutorAPIContract(
        sink_type=sink_type,
        stop_constructor=stop_constructor,
        leaf_paths=MappingProxyType(paths),
        leaf_hashes=MappingProxyType(hashes),
        runtime_module_paths=MappingProxyType(runtime_paths),
        runtime_module_hashes=MappingProxyType(runtime_hashes),
        frontier_node_ids=frontier,
        fixed_probes=probes,
    )


def _rehash_leaf(contract: ExecutorAPIContract, leaf_name: str) -> Path:
    try:
        path = Path(contract.leaf_paths[leaf_name]).resolve(strict=True)
        expected = contract.leaf_hashes[leaf_name]
    except (KeyError, FileNotFoundError, OSError) as error:
        raise DependencyBindingError(
            f"dependency leaf unavailable: {leaf_name}"
        ) from error
    try:
        actual = sha256(path.read_bytes()).hexdigest()
    except OSError as error:
        raise DependencyBindingError(
            f"dependency leaf unreadable: {leaf_name}"
        ) from error
    if actual != expected:
        raise DependencyBindingError(f"dependency leaf changed: {leaf_name}")
    return path


def _verify_imported_module(
    contract: ExecutorAPIContract,
    leaf_name: str,
    module: ModuleType,
) -> None:
    expected = _rehash_leaf(contract, leaf_name)
    module_file = getattr(module, "__file__", None)
    if type(module_file) is not str:
        raise DependencyBindingError(f"imported module has no source path: {leaf_name}")
    try:
        actual = Path(module_file).resolve(strict=True)
    except (FileNotFoundError, OSError) as error:
        raise DependencyBindingError(
            f"imported module source unavailable: {leaf_name}"
        ) from error
    if actual != expected:
        raise DependencyBindingError(
            f"imported module path differs from manifest leaf: {leaf_name}"
        )
    if sha256(actual.read_bytes()).hexdigest() != contract.leaf_hashes[leaf_name]:
        raise DependencyBindingError(
            f"imported module source differs from manifest leaf: {leaf_name}"
        )


def _verify_distribution_metadata(
    contract: ExecutorAPIContract,
    leaf_name: str,
    distribution_name: str,
) -> None:
    expected_path = _rehash_leaf(contract, leaf_name)
    try:
        distribution = importlib_metadata.distribution(distribution_name)
        metadata_text = distribution.read_text("METADATA")
    except importlib_metadata.PackageNotFoundError as error:
        raise DependencyBindingError(
            f"runtime distribution is unavailable: {distribution_name}"
        ) from error
    if metadata_text is None:
        raise DependencyBindingError(
            f"runtime distribution has no METADATA: {distribution_name}"
        )
    metadata_bytes = metadata_text.encode("utf-8")
    if metadata_bytes != expected_path.read_bytes():
        raise DependencyBindingError(
            f"runtime distribution METADATA differs from manifest leaf: {leaf_name}"
        )
    distribution_path = getattr(distribution, "_path", None)
    if distribution_path is not None:
        actual_path = (Path(distribution_path) / "METADATA").resolve(strict=True)
        if actual_path != expected_path:
            raise DependencyBindingError(
                f"runtime distribution METADATA path differs: {leaf_name}"
            )


def _preflight_all_leaf_bytes(contract: ExecutorAPIContract) -> None:
    for leaf_name in sorted(contract.leaf_paths):
        _rehash_leaf(contract, leaf_name)
    for module_name in sorted(contract.runtime_module_paths):
        try:
            path = Path(contract.runtime_module_paths[module_name]).resolve(strict=True)
        except (FileNotFoundError, OSError) as error:
            raise DependencyBindingError(
                f"runtime-module source unavailable: {module_name}"
            ) from error
        if (
            sha256(path.read_bytes()).hexdigest()
            != contract.runtime_module_hashes[module_name]
        ):
            raise DependencyBindingError(
                f"runtime-module source changed: {module_name}"
            )


def _attest_loaded_runtime_modules(
    contract: ExecutorAPIContract,
    *,
    require_complete: bool,
) -> None:
    observed: set[str] = set()
    for module_name, module in tuple(sys.modules.items()):
        audited_name = (
            module_name in ("ldvm_fourier", "flap_ldvm")
            or module_name == "fluxvortex"
            or module_name.startswith("fluxvortex.")
            or module_name == "forward_flight_benchmarks"
            or module_name.startswith("forward_flight_benchmarks.")
            or module_name == "pterasoftware"
            or module_name.startswith("pterasoftware.")
        )
        if not audited_name or module is None:
            continue
        module_file = getattr(module, "__file__", None)
        if type(module_file) is not str:
            raise DependencyBindingError(
                f"loaded runtime module has no file origin: {module_name}"
            )
        if module_name not in contract.runtime_module_paths:
            raise DependencyBindingError(
                f"loaded runtime module is absent from manifest inventory: {module_name}"
            )
        try:
            actual_path = Path(module_file).resolve(strict=True)
            expected_path = Path(contract.runtime_module_paths[module_name]).resolve(
                strict=True
            )
        except (FileNotFoundError, OSError) as error:
            raise DependencyBindingError(
                f"loaded runtime module origin is unavailable: {module_name}"
            ) from error
        if actual_path != expected_path:
            raise DependencyBindingError(
                f"loaded runtime module origin differs: {module_name}"
            )
        if (
            sha256(actual_path.read_bytes()).hexdigest()
            != contract.runtime_module_hashes[module_name]
        ):
            raise DependencyBindingError(
                f"loaded runtime module source differs: {module_name}"
            )
        observed.add(module_name)
    if require_complete and observed != set(contract.runtime_module_paths):
        missing = sorted(set(contract.runtime_module_paths) - observed)
        foreign = sorted(observed - set(contract.runtime_module_paths))
        raise DependencyBindingError(
            f"runtime-module inventory did not close: missing={missing}, foreign={foreign}"
        )


@dataclass(frozen=True, slots=True)
class RuntimeModules:
    coupling: ModuleType
    row_owner: ModuleType
    source: ModuleType
    correction: ModuleType
    reference: ModuleType
    stream: ModuleType
    ptera_transport: ModuleType
    pterasoftware: ModuleType


class _AuditedModuleFinder(importlib.abc.MetaPathFinder):
    """Resolve project modules only from the injected runtime inventory."""

    def __init__(self, contract: ExecutorAPIContract) -> None:
        self.contract = contract

    @staticmethod
    def _is_project_name(fullname: str) -> bool:
        return (
            fullname in ("ldvm_fourier", "flap_ldvm")
            or fullname == "fluxvortex"
            or fullname.startswith("fluxvortex.")
            or fullname == "forward_flight_benchmarks"
            or fullname.startswith("forward_flight_benchmarks.")
        )

    def find_spec(
        self,
        fullname: str,
        path: Sequence[str] | None,
        target: ModuleType | None = None,
    ) -> importlib.machinery.ModuleSpec | None:
        del path, target
        if not self._is_project_name(fullname):
            return None
        module_path = self.contract.runtime_module_paths.get(fullname)
        if module_path is None:
            raise ModuleNotFoundError(
                f"project import absent from audited runtime inventory: {fullname}"
            )
        source = Path(module_path).resolve(strict=True)
        if (
            sha256(source.read_bytes()).hexdigest()
            != self.contract.runtime_module_hashes[fullname]
        ):
            raise DependencyBindingError(
                f"project import changed before execution: {fullname}"
            )
        is_package = source.name == "__init__.py"
        return importlib.util.spec_from_file_location(
            fullname,
            source,
            submodule_search_locations=[str(source.parent)] if is_package else None,
        )


def _install_unexecuted_package(
    contract: ExecutorAPIContract,
    package_name: str,
) -> None:
    if package_name in sys.modules:
        raise DependencyBindingError(
            f"project package was loaded before audited namespace setup: {package_name}"
        )
    path_text = contract.runtime_module_paths.get(package_name)
    if path_text is None:
        raise DependencyBindingError(
            f"runtime inventory lacks package origin: {package_name}"
        )
    path = Path(path_text).resolve(strict=True)
    if (
        path.name != "__init__.py"
        or sha256(path.read_bytes()).hexdigest()
        != contract.runtime_module_hashes[package_name]
    ):
        raise DependencyBindingError(
            f"runtime package origin/hash is invalid: {package_name}"
        )
    module = ModuleType(package_name)
    module.__file__ = str(path)
    module.__package__ = package_name
    module.__path__ = [str(path.parent)]  # type: ignore[attr-defined]
    spec = importlib.machinery.ModuleSpec(package_name, loader=None, is_package=True)
    spec.origin = str(path)
    spec.submodule_search_locations = [str(path.parent)]
    module.__spec__ = spec
    sys.modules[package_name] = module


def _load_verified_runtime(contract: ExecutorAPIContract) -> RuntimeModules:
    _preflight_all_leaf_bytes(contract)
    _attest_loaded_runtime_modules(contract, require_complete=False)
    installed_packages: list[str] = []
    for package_name in ("fluxvortex", "forward_flight_benchmarks"):
        _install_unexecuted_package(contract, package_name)
        installed_packages.append(package_name)
    finder = _AuditedModuleFinder(contract)
    sys.meta_path.insert(0, finder)
    imported: dict[str, ModuleType] = {}
    try:
        for leaf_name, module_name in MODULE_LEAVES.items():
            module = import_module(module_name)
            if not isinstance(module, ModuleType):
                raise DependencyBindingError(
                    f"runtime import is not a module: {module_name}"
                )
            _verify_imported_module(contract, leaf_name, module)
            imported[leaf_name] = module
        pterasoftware = import_module("pterasoftware")
        if not isinstance(pterasoftware, ModuleType):
            raise DependencyBindingError("pterasoftware import is not a module")
        for leaf_name, distribution_name in DISTRIBUTION_LEAVES.items():
            _verify_distribution_metadata(contract, leaf_name, distribution_name)
        _attest_loaded_runtime_modules(contract, require_complete=False)
    except BaseException:
        for module_name in tuple(sys.modules):
            if (
                module_name in installed_packages
                or module_name.startswith("fluxvortex.")
                or module_name.startswith("forward_flight_benchmarks.")
            ):
                sys.modules.pop(module_name, None)
        raise
    finally:
        if finder in sys.meta_path:
            sys.meta_path.remove(finder)
    return RuntimeModules(
        coupling=imported["v5h11_baik_coupling"],
        row_owner=imported["v5h10_row_owner"],
        source=imported["v5h_dvm_source"],
        correction=imported["uvlm_correction"],
        reference=imported["rvpm_reference"],
        stream=imported["rvpm_ir_wrk3_stream"],
        ptera_transport=imported["v5h4_ptera_transport"],
        pterasoftware=pterasoftware,
    )


OBSERVER_EVIDENCE_FIELDS: Final = frozenset(
    {
        "fd_physical_evaluation_sha256",
        "fd_tracer_evaluation_sha256",
        "h_convective_over_sigma",
        "h_jacobian_frobenius",
        "source_state_sha256",
        "stage",
        "substep",
    }
)


def parse_stage_evidence(record: object) -> dict[str, object]:
    """Strictly decode and bind one coupling observer evidence payload."""

    evidence = getattr(record, "evidence", None)
    if evidence is None or getattr(evidence, "schema", None) != STAGE_EVIDENCE_SCHEMA:
        raise ExecutorContractError("compact stage evidence schema drift")
    payload = getattr(evidence, "payload", None)
    if type(payload) is not bytes:
        raise ExecutorContractError("compact stage evidence payload is not bytes")
    payload_sha = sha256(payload).hexdigest()
    if getattr(evidence, "payload_sha256", None) != payload_sha:
        raise ExecutorContractError("compact stage evidence payload digest mismatch")
    expected_evidence_sha = sha256(
        STAGE_EVIDENCE_SCHEMA.encode("utf-8") + b"\0" + payload_sha.encode("ascii")
    ).hexdigest()
    if getattr(evidence, "evidence_sha256", None) != expected_evidence_sha:
        raise ExecutorContractError("compact stage evidence digest mismatch")
    decoded = _strict_ascii_json_object(payload)
    if set(decoded) != OBSERVER_EVIDENCE_FIELDS:
        raise ExecutorContractError("compact stage evidence field set drift")
    substep = getattr(record, "substep", None)
    stage = getattr(record, "stage", None)
    if (
        type(substep) is not int
        or type(stage) is not int
        or decoded["substep"] != substep
        or decoded["stage"] != stage
        or decoded["source_state_sha256"]
        != getattr(record, "source_state_sha256", None)
    ):
        raise ExecutorContractError("compact stage evidence coordinate/source drift")
    for name in (
        "fd_physical_evaluation_sha256",
        "fd_tracer_evaluation_sha256",
        "source_state_sha256",
    ):
        if not _is_sha256(decoded[name]):
            raise ExecutorContractError(f"compact stage evidence SHA invalid: {name}")
    jacobian = _hex_float("h_jacobian_frobenius", decoded["h_jacobian_frobenius"])
    convective = _hex_float(
        "h_convective_over_sigma", decoded["h_convective_over_sigma"]
    )
    normalized = getattr(record, "invariant_residual_over_slog_max", None)
    if type(normalized) is not float:
        raise ExecutorContractError(
            "stream compact normalized-invariant field is not an exact float"
        )
    if not math.isfinite(normalized) or normalized < 0.0:
        raise ExecutorContractError(
            "stream compact normalized-invariant field is outside its domain"
        )
    if normalized > INVARIANT_NORMALIZED_GATE:
        raise ExecutorContractError("normalized invariant residual exceeded its gate")
    return {
        **decoded,
        "invariant_residual_over_slog_max": normalized,
        "h_jacobian_frobenius": jacobian,
        "h_convective_over_sigma": convective,
    }


def completed_stage_row(
    record: object,
    *,
    result: object,
    layer: int,
    verify_fd_ledger: bool = True,
) -> tuple[dict[str, object], dict[str, object]]:
    """Convert one validated compact record and its exact FD pair to a row."""

    if type(layer) is not int or layer not in LAYERS:
        raise ExecutorContractError("layer is outside the formal three-layer slice")
    evidence = parse_stage_evidence(record)
    substep = getattr(record, "substep", None)
    stage = getattr(record, "stage", None)
    level = getattr(result, "transport_substeps", None)
    if type(level) is not int or level not in FORMAL_LEVELS:
        raise ExecutorContractError("layer result has a non-formal N")
    if (
        type(substep) is not int
        or not 1 <= substep <= level
        or type(stage) is not int
        or stage not in (1, 2, 3)
    ):
        raise ExecutorContractError("compact stage coordinate is invalid")
    expected_index = 2 * ((substep - 1) * 3 + (stage - 1))
    ledger = getattr(result, "fd_call_ledger", None)
    evaluations = getattr(ledger, "evaluations", None)
    if type(verify_fd_ledger) is not bool:
        raise TypeError("verify_fd_ledger must be an exact bool")
    if verify_fd_ledger:
        if type(evaluations) is not tuple or expected_index + 1 >= len(evaluations):
            raise ExecutorContractError(
                "FD ledger lacks the exact stage evaluation pair"
            )
        physical = evaluations[expected_index]
        tracer = evaluations[expected_index + 1]
        if (
            getattr(physical, "evaluation_index", None) != expected_index + 1
            or getattr(tracer, "evaluation_index", None) != expected_index + 2
            or getattr(physical, "target_kind", None) != "physical"
            or getattr(tracer, "target_kind", None) != "tracer"
            or getattr(physical, "source_state_sha256", None)
            != getattr(record, "source_state_sha256", None)
            or getattr(tracer, "source_state_sha256", None)
            != getattr(record, "source_state_sha256", None)
            or evidence["fd_physical_evaluation_sha256"]
            != getattr(physical, "evaluation_sha256", None)
            or evidence["fd_tracer_evaluation_sha256"]
            != getattr(tracer, "evaluation_sha256", None)
        ):
            raise ExecutorContractError("FD evaluation pair/order/source binding drift")
    row = {
        "transport_substeps": level,
        "layer": layer,
        "source_step_index": SOURCE_STEPS[layer - 1],
        "ptera_step_index": PTERA_STEPS[layer - 1],
        "substep": substep,
        "stage": stage,
        "status": "completed",
        "substep_delta_time": DELTA_TIME_S / level,
        "rk_a": getattr(record, "a"),
        "rk_b": getattr(record, "b"),
        "pre_state_sha256": getattr(record, "pre_state_sha256"),
        "post_state_sha256": getattr(record, "post_state_sha256"),
        "tracer_pre_sha256": getattr(record, "tracer_pre_sha256"),
        "tracer_post_sha256": getattr(record, "tracer_post_sha256"),
        "velocity_sha256": getattr(record, "velocity_sha256"),
        "jacobian_sha256": getattr(record, "jacobian_sha256"),
        "gamma_rate_sha256": getattr(record, "gamma_rate_sha256"),
        "tracer_velocity_sha256": getattr(record, "tracer_velocity_sha256"),
        "invariant_residual_sha256": getattr(record, "invariant_residual_sha256"),
        "invariant_residual_over_slog_max": evidence[
            "invariant_residual_over_slog_max"
        ],
        "h_jacobian_frobenius": evidence["h_jacobian_frobenius"],
        "h_convective_over_sigma": evidence["h_convective_over_sigma"],
        "position_storage_pre_sha256": getattr(record, "position_storage_pre_sha256"),
        "gamma_storage_pre_sha256": getattr(record, "gamma_storage_pre_sha256"),
        "tracer_storage_pre_sha256": getattr(record, "tracer_storage_pre_sha256"),
        "position_storage_post_sha256": getattr(record, "position_storage_post_sha256"),
        "gamma_storage_post_sha256": getattr(record, "gamma_storage_post_sha256"),
        "tracer_storage_post_sha256": getattr(record, "tracer_storage_post_sha256"),
        "fd_physical_evaluation_sha256": evidence["fd_physical_evaluation_sha256"],
        "fd_tracer_evaluation_sha256": evidence["fd_tracer_evaluation_sha256"],
        "ptera_parent_sha256_before": getattr(
            result, "ptera_parent_sha256_before_transport"
        ),
        "ptera_parent_sha256_after": getattr(
            result, "ptera_parent_sha256_after_transport"
        ),
        "direct_field_call_count": 2,
        "ptera_center_call_count": 2,
        "ptera_offset_call_count": 6,
        "stream_record_sha256": getattr(record, "record_sha256"),
        "previous_chain_sha256": getattr(record, "previous_chain_sha256"),
        "chain_sha256": getattr(record, "chain_sha256"),
        "failure_type": "",
        "failure_message": "",
    }
    if set(row) != set(STAGE_FIELDS):
        raise AssertionError("executor completed-stage row schema is inconsistent")
    if row["rk_a"] != RK_A[stage - 1] or row["rk_b"] != RK_B[stage - 1]:
        raise ExecutorContractError("compact stage RK coefficient drift")
    compact = {
        "invariant_residual_over_slog_max": evidence[
            "invariant_residual_over_slog_max"
        ],
        "h_jacobian_frobenius": evidence["h_jacobian_frobenius"],
        "h_convective_over_sigma": evidence["h_convective_over_sigma"],
    }
    return row, compact


def failed_stage_row(
    error: object,
    *,
    level: int,
    layer: int,
) -> tuple[dict[str, object], dict[str, object]]:
    """Create an honest nullable failed row from StreamStopped metadata only."""

    substep = getattr(error, "substep", None)
    stage = getattr(error, "stage", None)
    chain = getattr(error, "completed_stage_chain_sha256", None)
    cause = getattr(error, "original_cause", None)
    if (
        type(level) is not int
        or level not in FORMAL_LEVELS
        or layer not in LAYERS
        or type(substep) is not int
        or not 1 <= substep <= level
        or type(stage) is not int
        or stage not in (1, 2, 3)
        or not _is_sha256(chain)
        or not isinstance(cause, BaseException)
    ):
        raise ExecutorContractError("StreamStopped metadata is invalid")
    row = {name: None for name in STAGE_FIELDS}
    row.update(
        {
            "transport_substeps": level,
            "layer": layer,
            "source_step_index": SOURCE_STEPS[layer - 1],
            "ptera_step_index": PTERA_STEPS[layer - 1],
            "substep": substep,
            "stage": stage,
            "status": "failed",
            "substep_delta_time": DELTA_TIME_S / level,
            "rk_a": RK_A[stage - 1],
            "rk_b": RK_B[stage - 1],
            "previous_chain_sha256": chain,
            "chain_sha256": chain,
            "failure_type": type(cause).__name__,
            "failure_message": str(cause),
        }
    )
    compact = {
        "invariant_residual_over_slog_max": None,
        "h_jacobian_frobenius": None,
        "h_convective_over_sigma": None,
    }
    return row, compact


def _json_tree(value: object) -> object:
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("metadata contains a non-finite float")
        return value
    if isinstance(value, np.generic):
        return _json_tree(value.item())
    if isinstance(value, Mapping):
        result: dict[str, object] = {}
        for key, item in value.items():
            if type(key) is not str or key in result:
                raise TypeError("metadata mappings require unique exact string keys")
            result[key] = _json_tree(item)
        return result
    if type(value) in (tuple, list):
        return [_json_tree(item) for item in value]
    if hasattr(value, "__dataclass_fields__"):
        return _json_tree(asdict(value))
    raise TypeError(f"metadata contains a foreign value: {type(value).__qualname__}")


def stable_particle_id(value: object) -> str:
    """Encode one row-owner semantic ID as an exact stable JSON string."""

    tree = _json_tree(value)
    if type(tree) is not list:
        raise ExecutorContractError("row-owner particle ID must be a tuple/list tree")
    encoded = _canonical_json_bytes(tree).decode("utf-8")
    return "fluxv-v5h10-particle-id-v1:" + encoded


def _payload_digest(domain: str, value: object) -> str:
    return sha256(
        domain.encode("ascii") + b"\0" + _canonical_json_bytes(value)
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class FrozenFloat64Array:
    shape: tuple[int, ...]
    payload: bytes
    sha256: str

    def array(self) -> np.ndarray:
        value = np.frombuffer(self.payload, dtype=np.float64).reshape(self.shape)
        if value.flags.writeable or not value.flags.c_contiguous:
            raise ExecutorContractError("frozen capture array lost readonly C layout")
        return value


def _capture_array(
    name: str, value: object, expected_tail: tuple[int, ...]
) -> FrozenFloat64Array:
    array = _freeze_array(name, value, expected_tail)
    payload = array.tobytes(order="C")
    digest = sha256(
        name.encode("utf-8")
        + b"\0"
        + array.dtype.str.encode("ascii")
        + b"\0"
        + _canonical_json_bytes(list(array.shape))
        + b"\0"
        + payload
    ).hexdigest()
    return FrozenFloat64Array(array.shape, payload, digest)


@dataclass(frozen=True, slots=True)
class LayerCommitCapture:
    level: int
    layer: int
    source_step_index: int
    ptera_step_index: int
    source_row_payload: bytes
    source_cell_manifest_payload: bytes
    source_cell_manifest_sha256: str
    source_prehistory_manifest_payload: bytes
    source_prehistory_manifest_sha256: str
    source_kelvin_ledger_sha256: str
    kelvin_residual_max_abs: float
    committed_owner_sha256: str
    committed_state_sha256: str
    start_positions: FrozenFloat64Array
    start_gamma: FrozenFloat64Array
    start_sigma: FrozenFloat64Array
    particle_ids: tuple[str, ...]
    material_tracer_ids: tuple[str, ...]
    frontier_start_positions: FrozenFloat64Array
    changed_particle_ids: tuple[str, ...]
    appended_particle_ids: tuple[str, ...]
    commit_event: object
    source_kelvin_evidence_sha256: str

    def source_row(self) -> dict[str, object]:
        value = json.loads(
            self.source_row_payload.decode("utf-8"),
            object_pairs_hook=_strict_pairs,
            parse_constant=_reject_constant,
        )
        if (
            type(value) is not dict
            or _canonical_json_bytes(value) != self.source_row_payload
        ):
            raise ExecutorContractError("captured source row changed after commit")
        return value

    def source_cell_manifests(self) -> list[object]:
        value = json.loads(
            self.source_cell_manifest_payload.decode("utf-8"),
            object_pairs_hook=_strict_pairs,
            parse_constant=_reject_constant,
        )
        if type(value) is not list:
            raise ExecutorContractError("captured source-cell ledger is not a list")
        if (
            _canonical_json_bytes(value) != self.source_cell_manifest_payload
            or _payload_digest("fluxv-v5h11-source-cell-manifests-v1", value)
            != self.source_cell_manifest_sha256
        ):
            raise ExecutorContractError("captured source-cell ledger digest mismatch")
        return value

    def source_prehistory_manifests(self) -> list[object]:
        value = json.loads(
            self.source_prehistory_manifest_payload.decode("utf-8"),
            object_pairs_hook=_strict_pairs,
            parse_constant=_reject_constant,
        )
        if type(value) is not list:
            raise ExecutorContractError(
                "captured source-prehistory ledger is not a list"
            )
        if (
            _canonical_json_bytes(value) != self.source_prehistory_manifest_payload
            or _payload_digest(SOURCE_PREHISTORY_MANIFEST_DOMAIN, value)
            != self.source_prehistory_manifest_sha256
        ):
            raise ExecutorContractError(
                "captured source-prehistory ledger digest mismatch"
            )
        return value


def _event_manifest(event: object) -> dict[str, object]:
    manifest_method = getattr(event, "manifest", None)
    if callable(manifest_method):
        value = manifest_method()
    elif hasattr(event, "__dataclass_fields__"):
        value = asdict(event)
    else:
        raise TypeError("source event has no auditable manifest")
    tree = _json_tree(value)
    if type(tree) is not dict or set(tree) != set(SOURCE_CELL_MANIFEST_FIELDS):
        raise TypeError("source event manifest does not use the frozen v3 schema")
    _canonical_json_bytes(tree)
    producer = tree["producer_manifest_sha256"]
    parent = tree["parent_event_manifest_sha256"]
    unsigned = dict(tree)
    del unsigned["producer_manifest_sha256"]
    if (
        not _is_sha256(parent)
        or not _is_sha256(producer)
        or producer
        != sha256(
            SOURCE_EVENT_DIGEST_PREFIX + _canonical_json_bytes(unsigned)
        ).hexdigest()
    ):
        raise ExecutorContractError("source event v3 digest/parent binding is invalid")
    provenance = tree["provenance"]
    if (
        not isinstance(provenance, Mapping)
        or provenance.get("interface_id") != SOURCE_INTERFACE_ID
        or provenance.get("backend_id") != SOURCE_BACKEND_ID
    ):
        raise ExecutorContractError("source event v3 provenance domain drifted")
    for name in ("lev_placement", "tev_placement"):
        placement = tree[name]
        if (
            not isinstance(placement, Mapping)
            or placement.get("schema_id") != SOURCE_PLACEMENT_SCHEMA_ID
        ):
            raise ExecutorContractError("source event v3 placement domain drifted")
    return tree


def _freeze_source_prehistory(
    source_prehistory_events: tuple[tuple[object, ...], ...], *, layer: int
) -> tuple[bytes, str]:
    if (
        type(layer) is not int
        or layer not in LAYERS
        or type(source_prehistory_events) is not tuple
        or (
            layer == 1
            and (
                len(source_prehistory_events) != 3
                or any(
                    type(step_events) is not tuple
                    or len(step_events) != SOURCE_CELL_COUNT
                    for step_events in source_prehistory_events
                )
            )
        )
        or (layer != 1 and source_prehistory_events != ())
    ):
        raise ExecutorContractError("source-prehistory scope/shape is invalid")
    manifests = [
        [_event_manifest(event) for event in step_events]
        for step_events in source_prehistory_events
    ]
    payload = _canonical_json_bytes(manifests)
    return payload, _payload_digest(SOURCE_PREHISTORY_MANIFEST_DOMAIN, manifests)


@dataclass(frozen=True, slots=True)
class SourceAggregate:
    row: Mapping[str, object]
    cell_manifest_payload: bytes
    cell_manifest_sha256: str
    kelvin_ledger_sha256: str
    residual_max_abs: float


def aggregate_source_events(
    events: Sequence[object],
    *,
    level: int,
    source_step_index: int,
    ptera_step_index: int,
    source_time_s: float,
) -> SourceAggregate:
    """Freeze the full eight-cell source ledger and its compact runner row."""

    if (
        type(events) is not tuple
        or len(events) != SOURCE_CELL_COUNT
        or type(level) is not int
        or level not in FORMAL_LEVELS
        or source_step_index not in SOURCE_STEPS
        or ptera_step_index != source_step_index - 1
        or type(source_time_s) is not float
        or not math.isfinite(source_time_s)
        or source_time_s <= 0.0
    ):
        raise ExecutorContractError("source aggregate coordinate/shape is invalid")
    manifests = [_event_manifest(event) for event in events]
    manifest_payload = _canonical_json_bytes(manifests)
    manifest_sha = _payload_digest("fluxv-v5h11-source-cell-manifests-v1", manifests)
    producer_hashes: list[str] = []
    parent_hashes: list[str] = []
    ledgers: list[object] = []
    a0_pre: list[float] = []
    a0_post: list[float] = []
    birth_modes: list[str] = []
    gamma_lev_new: list[float] = []
    gamma_lev_persisted: list[float] = []
    gamma_tev_new: list[float] = []
    kelvin_residuals: list[float] = []
    for event in events:
        producer = getattr(event, "producer_manifest_sha256", None)
        parent = getattr(event, "parent_event_manifest_sha256", None)
        ledger = getattr(event, "kelvin_ledger", None)
        provenance = getattr(event, "provenance", None)
        scale = getattr(provenance, "circulation_scale_u_times_c_m2_per_s", None)
        if (
            not _is_sha256(producer)
            or not _is_sha256(parent)
            or ledger is None
            or type(scale) is not float
            or not math.isfinite(scale)
            or scale <= 0.0
        ):
            raise ExecutorContractError(
                "source event lacks exact provenance/Kelvin evidence"
            )
        producer_hashes.append(producer)
        parent_hashes.append(parent)
        ledgers.append(_json_tree(ledger))
        pre = getattr(event, "a0_pre", None)
        post = getattr(event, "a0_post", None)
        mode = getattr(event, "lev_birth_mode", None)
        persisted = getattr(ledger, "gamma_lev_persisted_after", None)
        values = (
            pre,
            post,
            getattr(event, "gamma_lev_new_over_u_c", None),
            persisted,
            getattr(event, "gamma_tev_new_solved_over_u_c", None),
            getattr(event, "kelvin_residual_over_u_c", None),
        )
        if type(mode) is not str or any(
            type(value) is not float or not math.isfinite(value) for value in values
        ):
            raise ExecutorContractError(
                "source event carries invalid active-step scalars"
            )
        a0_pre.append(pre)
        a0_post.append(post)
        birth_modes.append(mode)
        gamma_lev_new.append(values[2] * scale)
        gamma_lev_persisted.append(values[3] * scale)
        gamma_tev_new.append(values[4] * scale)
        kelvin_residuals.append(values[5] * scale)
    residual_max = max(abs(value) for value in kelvin_residuals)
    if residual_max > MAX_SOURCE_KELVIN_M2_PER_S:
        raise ExecutorContractError("eight-cell source Kelvin gate failed")
    ledger_sha = _payload_digest("fluxv-v5h11-source-kelvin-ledgers-v1", ledgers)
    row: dict[str, object] = {
        "transport_substeps": level,
        "source_step_index": source_step_index,
        "ptera_step_index": ptera_step_index,
        "source_time_s": source_time_s,
        "cell_count": SOURCE_CELL_COUNT,
        "status": "completed",
        "birth_modes": birth_modes,
        "a0_pre": a0_pre,
        "a0_post": a0_post,
        "gamma_lev_new_m2_s": gamma_lev_new,
        "gamma_lev_persisted_m2_s": gamma_lev_persisted,
        "gamma_tev_new_m2_s": gamma_tev_new,
        "kelvin_residual_m2_s": kelvin_residuals,
        "event_sha256": _payload_digest(
            "fluxv-v5h11-source-event-aggregate-v1", producer_hashes
        ),
        "parent_event_sha256": _payload_digest(
            "fluxv-v5h11-source-parent-aggregate-v1", parent_hashes
        ),
    }
    if set(row) != set(SOURCE_EVENT_FIELDS):
        raise AssertionError("executor source aggregate schema is inconsistent")
    return SourceAggregate(
        row=MappingProxyType(row),
        cell_manifest_payload=manifest_payload,
        cell_manifest_sha256=manifest_sha,
        kelvin_ledger_sha256=ledger_sha,
        residual_max_abs=residual_max,
    )


def capture_committed_row(
    *,
    level: int,
    layer: int,
    request: object,
    events: tuple[object, ...],
    source_prehistory_events: tuple[tuple[object, ...], ...],
    commit_result: object,
    source_kelvin_evidence: object,
) -> LayerCommitCapture:
    """Immediately bytes-freeze macro-start arrays, IDs, and source evidence."""

    owner = getattr(commit_result, "owner", None)
    state = getattr(commit_result, "state", None)
    if owner is None or state is None or getattr(owner, "state", None) is not state:
        raise ExecutorContractError("row commit result lacks its exact owner/state")
    source_step = getattr(request, "source_step_index", None)
    ptera_step = getattr(request, "ptera_step_index", None)
    aggregate = aggregate_source_events(
        events,
        level=level,
        source_step_index=source_step,
        ptera_step_index=ptera_step,
        source_time_s=getattr(request, "source_time_s", None),
    )
    prehistory_payload, prehistory_sha = _freeze_source_prehistory(
        source_prehistory_events, layer=layer
    )
    raw_ids = getattr(state, "particle_ids", None)
    live_by_cell = getattr(state, "live_boundary_indices_by_cell", None)
    if type(raw_ids) is not tuple or type(live_by_cell) is not tuple:
        raise ExecutorContractError("committed row lacks exact semantic IDs/frontier")
    particle_ids = tuple(stable_particle_id(value) for value in raw_ids)
    if len(set(particle_ids)) != len(particle_ids):
        raise ExecutorContractError("committed row particle IDs are not unique")
    flat_live = tuple(index for group in live_by_cell for index in group)
    if any(
        type(index) is not int or not 0 <= index < len(raw_ids) for index in flat_live
    ):
        raise ExecutorContractError("committed row live-particle index is invalid")
    material_ids = tuple(particle_ids[index] for index in flat_live)
    event = getattr(commit_result, "event", None)
    if event is None:
        changed_indices: tuple[int, ...] = ()
        appended_indices = tuple(range(len(particle_ids)))
        commit_event: object = None
    else:
        changed_indices = getattr(event, "changed_indices", None)
        appended_indices = getattr(event, "appended_indices", None)
        if type(changed_indices) is not tuple or type(appended_indices) is not tuple:
            raise ExecutorContractError("row commit event index ledger is invalid")
        commit_event = _json_tree(event)
    if any(
        type(index) is not int or not 0 <= index < len(particle_ids)
        for index in (*changed_indices, *appended_indices)
    ):
        raise ExecutorContractError(
            "row commit event contains an invalid particle index"
        )
    kelvin_sha = getattr(source_kelvin_evidence, "evidence_sha256", None)
    if (
        not _is_sha256(getattr(owner, "owner_sha256", None))
        or not _is_sha256(getattr(state, "state_sha256", None))
        or not _is_sha256(kelvin_sha)
        or getattr(source_kelvin_evidence, "source_event_sha256", None)
        != aggregate.row["event_sha256"]
        or getattr(source_kelvin_evidence, "kelvin_ledger_sha256", None)
        != aggregate.kelvin_ledger_sha256
    ):
        raise ExecutorContractError(
            "source Kelvin evidence is not bound to the capture"
        )
    source_row_payload = _canonical_json_bytes(dict(aggregate.row))
    return LayerCommitCapture(
        level=level,
        layer=layer,
        source_step_index=source_step,
        ptera_step_index=ptera_step,
        source_row_payload=source_row_payload,
        source_cell_manifest_payload=aggregate.cell_manifest_payload,
        source_cell_manifest_sha256=aggregate.cell_manifest_sha256,
        source_prehistory_manifest_payload=prehistory_payload,
        source_prehistory_manifest_sha256=prehistory_sha,
        source_kelvin_ledger_sha256=aggregate.kelvin_ledger_sha256,
        kelvin_residual_max_abs=aggregate.residual_max_abs,
        committed_owner_sha256=owner.owner_sha256,
        committed_state_sha256=state.state_sha256,
        start_positions=_capture_array("start_positions", state.positions, (3,)),
        start_gamma=_capture_array("start_gamma", state.gamma, (3,)),
        start_sigma=_capture_array("start_sigma", state.sigma[:, None], (1,)),
        particle_ids=particle_ids,
        material_tracer_ids=material_ids,
        frontier_start_positions=_capture_array(
            "frontier_start_positions", state.live_boundary_nodes, (3,)
        ),
        changed_particle_ids=tuple(particle_ids[index] for index in changed_indices),
        appended_particle_ids=tuple(particle_ids[index] for index in appended_indices),
        commit_event=commit_event,
        source_kelvin_evidence_sha256=kelvin_sha,
    )


def _stable_row_norms(gamma: np.ndarray) -> np.ndarray:
    maximum = np.max(np.abs(gamma), axis=1)
    result = np.zeros(gamma.shape[0], dtype=np.float64)
    active = maximum != 0.0
    if np.any(active):
        scaled = gamma[active] / maximum[active, None]
        result[active] = maximum[active] * np.sqrt(
            np.einsum("ni,ni->n", scaled, scaled)
        )
    if not np.all(np.isfinite(result)):
        raise ExecutorContractError("trajectory gamma norm is non-finite")
    return result


def trajectory_array_record(
    result: object,
    capture: LayerCommitCapture,
    *,
    contract: ExecutorAPIContract,
    direct_evaluator: Callable[..., object],
) -> dict[str, object]:
    """Build the exact ID-aligned replay trajectory and fixed direct probes."""

    if getattr(result, "transport_substeps", None) != capture.level:
        raise ExecutorContractError("trajectory result/capture level drift")
    stream = getattr(result, "stream_result", None)
    final_state = getattr(stream, "final_state", None)
    if final_state is None:
        raise ExecutorContractError("layer result lacks final stream state")
    end_positions = _freeze_array("end_positions", final_state.positions, (3,))
    end_gamma = _freeze_array("end_gamma", final_state.gamma, (3,))
    end_sigma_matrix = _freeze_array("end_sigma", final_state.sigma[:, None], (1,))
    end_sigma = end_sigma_matrix[:, 0]
    if (
        end_positions.shape[0] != len(capture.particle_ids)
        or end_gamma.shape != end_positions.shape
        or end_sigma.shape != (end_positions.shape[0],)
        or np.any(end_sigma <= 0.0)
    ):
        raise ExecutorContractError("final physical state is not ID-aligned")
    tracers = _freeze_array(
        "final_tracer_positions", stream.final_tracer_positions, (3,)
    )
    material_count = len(capture.material_tracer_ids)
    frontier_count = len(contract.frontier_node_ids)
    if tracers.shape != (material_count + frontier_count, 3):
        raise ExecutorContractError("final tracer pack does not match semantic IDs")
    probes = np.asarray(contract.fixed_probes, dtype=np.float64)
    direct = direct_evaluator(
        end_positions,
        end_gamma,
        end_sigma,
        target_positions=probes,
    )
    probe_velocity = _freeze_array("probe_velocity", direct.velocity, (3,))
    probe_jacobian_source = np.asarray(direct.jacobian)
    if (
        probe_jacobian_source.dtype.kind not in "iuf"
        or probe_jacobian_source.dtype.kind == "b"
    ):
        raise TypeError("probe_jacobian must have a real numeric dtype")
    probe_jacobian = np.array(
        probe_jacobian_source, dtype=np.float64, order="C", copy=True
    )
    if probe_jacobian.shape != (3, 3, 3) or not np.all(np.isfinite(probe_jacobian)):
        raise ExecutorContractError("fixed probe Jacobian has invalid shape/values")
    ledger = getattr(result, "load_ledger", None)
    force = _freeze_vector("force", getattr(ledger, "forces_w", None), 3)
    moment = _freeze_vector("moment", getattr(ledger, "moments_w_cgp1", None), 3)
    residual = _freeze_vector(
        "no_penetration_residual",
        getattr(ledger, "no_penetration_residual", None),
        16,
    )
    start_positions = capture.start_positions.array()
    start_gamma = capture.start_gamma.array()
    start_sigma = capture.start_sigma.array()[:, 0]
    arrays = {
        "start_positions": start_positions,
        "start_gamma": start_gamma,
        "start_sigma": start_sigma,
        "end_positions": end_positions,
        "end_gamma": end_gamma,
        "end_sigma": end_sigma,
        "material_tracer_positions": tracers[:material_count],
        "frontier_tracer_positions": tracers[material_count:],
        "probe_velocity": probe_velocity,
        "probe_jacobian": probe_jacobian,
        "force": force,
        "moment": moment,
        "invariant_start": _stable_row_norms(start_gamma) * start_sigma**2,
        "invariant_end": _stable_row_norms(end_gamma) * end_sigma**2,
        "no_penetration_residual": residual,
    }
    if set(arrays) != set(REQUIRED_TRAJECTORY_ARRAYS) or any(
        type(value) is not np.ndarray or value.dtype != np.dtype(np.float64)
        for value in arrays.values()
    ):
        raise AssertionError("executor trajectory array schema/dtype is inconsistent")
    metadata = {
        "source_cell_manifest_sha256": capture.source_cell_manifest_sha256,
        "source_cell_manifests": capture.source_cell_manifests(),
        "source_prehistory_manifest_sha256": (
            capture.source_prehistory_manifest_sha256
        ),
        "source_prehistory_manifests": capture.source_prehistory_manifests(),
        "source_kelvin_ledger_sha256": capture.source_kelvin_ledger_sha256,
        "source_kelvin_evidence_sha256": capture.source_kelvin_evidence_sha256,
        "particle_id_sequence_sha256": _payload_digest(
            "fluxv-v5h11-particle-id-sequence-v1", list(capture.particle_ids)
        ),
        "material_tracer_id_sequence_sha256": _payload_digest(
            "fluxv-v5h11-material-id-sequence-v1",
            list(capture.material_tracer_ids),
        ),
        "frontier_start_positions_sha256": capture.frontier_start_positions.sha256,
        "fixed_probe_contract": [list(point) for point in contract.fixed_probes],
    }
    return {
        "transport_substeps": capture.level,
        "layer": capture.layer,
        "status": "completed",
        "particle_ids": capture.particle_ids,
        "material_tracer_ids": capture.material_tracer_ids,
        "frontier_node_ids": contract.frontier_node_ids,
        "arrays": arrays,
        "metadata": metadata,
    }


def raw_step_row(result: object, capture: LayerCommitCapture) -> dict[str, object]:
    stream = getattr(result, "stream_result", None)
    counters = getattr(result, "counters", None)
    stream_counters = getattr(stream, "counters", None)
    fd = getattr(result, "fd_call_ledger", None)
    load = getattr(result, "load_ledger", None)
    coefficients = np.asarray(getattr(load, "force_coefficients_w", None))
    if coefficients.shape != (3,) or not np.all(np.isfinite(coefficients)):
        raise ExecutorContractError("raw-step force coefficient vector is invalid")
    final_tracers = np.asarray(getattr(stream, "final_tracer_positions", None))
    if final_tracers.ndim != 2 or final_tracers.shape[1:] != (3,):
        raise ExecutorContractError("raw-step final tracer pack is invalid")
    row = {
        "transport_substeps": capture.level,
        "layer": capture.layer,
        "source_step_index": capture.source_step_index,
        "ptera_step_index": capture.ptera_step_index,
        "status": "completed",
        "particle_count": len(capture.particle_ids),
        "material_tracer_count": int(final_tracers.shape[0]),
        "material_support_tracer_count": len(capture.material_tracer_ids),
        "frontier_node_tracer_count": int(
            final_tracers.shape[0] - len(capture.material_tracer_ids)
        ),
        "stream_result_sha256": getattr(stream, "result_sha256", None),
        "stream_stage_chain_sha256": getattr(stream, "stage_chain_sha256", None),
        "fd_ledger_sha256": getattr(fd, "ledger_sha256", None),
        "load_ledger_sha256": getattr(load, "ledger_sha256", None),
        "layer_result_sha256": getattr(result, "result_sha256", None),
        "row_owner_before_sha256": getattr(result, "row_owner_before_sha256", None),
        "advanced_owner_sha256": getattr(result, "advanced_owner_sha256", None),
        "ptera_parent_sha256_before": getattr(
            result, "ptera_parent_sha256_before_transport", None
        ),
        "ptera_parent_sha256_after": getattr(
            result, "ptera_parent_sha256_after_transport", None
        ),
        "no_penetration_max_abs": float(getattr(load, "no_penetration_max_abs")),
        "kelvin_residual_max_abs": capture.kelvin_residual_max_abs,
        "raw_cl": -float(coefficients[2]),
        "raw_cd": -float(coefficients[0]),
        "direct_field_call_count": getattr(counters, "direct_field_call_count", None),
        "ptera_center_call_count": getattr(counters, "ptera_center_call_count", None),
        "ptera_offset_call_count": getattr(counters, "ptera_offset_call_count", None),
        "fd_physical_evaluation_count": getattr(fd, "physical_evaluation_count", None),
        "fd_tracer_evaluation_count": getattr(fd, "tracer_evaluation_count", None),
        "fd_evaluator_call_count": getattr(fd, "evaluator_call_count", None),
        "transport_substep_count": getattr(stream_counters, "substep_count", None),
        "transport_stage_count": getattr(stream_counters, "stage_count", None),
        "physical_field_call_count": getattr(
            stream_counters, "physical_field_call_count", None
        ),
        "tracer_field_call_count": getattr(
            stream_counters, "tracer_field_call_count", None
        ),
        "observer_call_count": getattr(stream_counters, "observer_call_count", None),
        "stage_pre_reconstruction_count": getattr(
            stream_counters, "stage_pre_reconstruction_count", None
        ),
        "stage_post_reconstruction_count": getattr(
            stream_counters, "stage_post_reconstruction_count", None
        ),
        "physical_rhs_call_count": getattr(
            stream_counters, "physical_rhs_call_count", None
        ),
        "storage_reset_count": getattr(stream_counters, "storage_reset_count", None),
        "tracer_storage_reset_count": getattr(
            stream_counters, "tracer_storage_reset_count", None
        ),
        "invariant_reference_freeze_count": getattr(
            stream_counters, "invariant_reference_freeze_count", None
        ),
        "sigma_storage_update_count": getattr(
            stream_counters, "sigma_storage_update_count", None
        ),
        "relaxation_call_count": getattr(
            stream_counters, "relaxation_call_count", None
        ),
        "compact_stage_record_count": getattr(
            stream_counters, "compact_stage_record_count", None
        ),
        "retained_stage_array_count": getattr(
            stream_counters, "retained_stage_array_count", None
        ),
        "evidence_byte_count": getattr(stream_counters, "evidence_byte_count", None),
        "native_collocation_evaluation_count": getattr(
            counters, "native_collocation_evaluation_count", None
        ),
        "native_load_batch_evaluation_count": getattr(
            counters, "native_load_batch_evaluation_count", None
        ),
        "native_load_call_count": getattr(counters, "native_load_call_count", None),
        "transport_macro_call_count": 1,
        "row_commit_count": 1,
    }
    if set(row) != set(RAW_STEP_FIELDS):
        raise AssertionError("executor raw-step row schema is inconsistent")
    for name in (
        "stream_result_sha256",
        "stream_stage_chain_sha256",
        "fd_ledger_sha256",
        "load_ledger_sha256",
        "layer_result_sha256",
        "row_owner_before_sha256",
        "advanced_owner_sha256",
        "ptera_parent_sha256_before",
        "ptera_parent_sha256_after",
    ):
        if not _is_sha256(row[name]):
            raise ExecutorContractError(f"raw-step SHA is invalid: {name}")
    return row


def owner_event_row(result: object, capture: LayerCommitCapture) -> dict[str, object]:
    advanced = getattr(result, "advanced_owner", None)
    transport_events = getattr(advanced, "transport_events", None)
    if type(transport_events) is not tuple or not transport_events:
        raise ExecutorContractError("advanced owner lacks its transport event")
    row = {
        "schema_id": "fluxv-v5h11-baik-w2-owner-event-v1",
        "transport_substeps": capture.level,
        "layer": capture.layer,
        "source_step_index": capture.source_step_index,
        "ptera_step_index": capture.ptera_step_index,
        "status": "completed",
        "row_owner_before_sha256": getattr(result, "row_owner_before_sha256", None),
        "row_state_before_sha256": getattr(result, "row_state_before_sha256", None),
        "common_transport_sha256": getattr(result, "common_transport_sha256", None),
        "transport_attestation_sha256": getattr(
            result, "transport_attestation_sha256", None
        ),
        "transport_parent_digest": getattr(result, "transport_parent_digest", None),
        "advanced_owner_sha256": getattr(result, "advanced_owner_sha256", None),
        "advanced_state_sha256": getattr(result, "advanced_state_sha256", None),
        "changed_particle_ids": list(capture.changed_particle_ids),
        "appended_particle_ids": list(capture.appended_particle_ids),
        "commit_event": capture.commit_event,
        "transport_event": _json_tree(transport_events[-1]),
    }
    if set(row) != set(OWNER_EVENT_FIELDS):
        raise AssertionError("executor owner-event row schema is inconsistent")
    return row


def particle_count_row(
    result: object,
    capture: LayerCommitCapture,
    *,
    frontier_count: int,
) -> dict[str, object]:
    final_tracers = np.asarray(result.stream_result.final_tracer_positions)
    row = {
        "transport_substeps": capture.level,
        "layer": capture.layer,
        "status": "completed",
        "particle_count": len(capture.particle_ids),
        "material_tracer_count": int(final_tracers.shape[0]),
        "material_support_tracer_count": len(capture.material_tracer_ids),
        "frontier_node_tracer_count": frontier_count,
        "changed_particle_count": len(capture.changed_particle_ids),
        "appended_particle_count": len(capture.appended_particle_ids),
    }
    if set(row) != set(PARTICLE_COUNT_FIELDS):
        raise AssertionError("executor particle-count row schema is inconsistent")
    return row


def raw_load_rows(
    result: object, capture: LayerCommitCapture
) -> tuple[dict[str, object], ...]:
    ledger = getattr(result, "load_ledger", None)
    panel_ids = getattr(ledger, "panel_ids", None)
    forces = np.asarray(getattr(ledger, "panel_forces_w", None))
    moments = np.asarray(getattr(ledger, "panel_moments_w_cgp1", None))
    total_force = np.asarray(getattr(ledger, "forces_w", None))
    total_moment = np.asarray(getattr(ledger, "moments_w_cgp1", None))
    coefficients = np.asarray(getattr(ledger, "force_coefficients_w", None))
    if (
        type(panel_ids) is not tuple
        or len(panel_ids) != 16
        or forces.shape != (16, 3)
        or moments.shape != (16, 3)
        or total_force.shape != (3,)
        or total_moment.shape != (3,)
        or coefficients.shape != (3,)
        or any(
            not np.all(np.isfinite(value))
            for value in (forces, moments, total_force, total_moment, coefficients)
        )
    ):
        raise ExecutorContractError("native load ledger cannot form 16+1 raw rows")
    rows: list[dict[str, object]] = []
    for panel_id, force, moment in zip(panel_ids, forces, moments, strict=True):
        row = {
            "transport_substeps": capture.level,
            "layer": capture.layer,
            "scope": "panel",
            "panel_id": panel_id,
            "force_x_n": float(force[0]),
            "force_y_n": float(force[1]),
            "force_z_n": float(force[2]),
            "moment_x_nm": float(moment[0]),
            "moment_y_nm": float(moment[1]),
            "moment_z_nm": float(moment[2]),
            "force_coefficient_x": None,
            "force_coefficient_y": None,
            "force_coefficient_z": None,
            "raw_cl": None,
            "raw_cd": None,
        }
        rows.append(row)
    rows.append(
        {
            "transport_substeps": capture.level,
            "layer": capture.layer,
            "scope": "total",
            "panel_id": "TOTAL",
            "force_x_n": float(total_force[0]),
            "force_y_n": float(total_force[1]),
            "force_z_n": float(total_force[2]),
            "moment_x_nm": float(total_moment[0]),
            "moment_y_nm": float(total_moment[1]),
            "moment_z_nm": float(total_moment[2]),
            "force_coefficient_x": float(coefficients[0]),
            "force_coefficient_y": float(coefficients[1]),
            "force_coefficient_z": float(coefficients[2]),
            "raw_cl": -float(coefficients[2]),
            "raw_cd": -float(coefficients[0]),
        }
    )
    if len(rows) != 17 or any(set(row) != set(RAW_LOAD_FIELDS) for row in rows):
        raise AssertionError("executor raw-load block schema is inconsistent")
    return tuple(rows)


CONVERSION_STOP_CODE: Final = "stage_evidence_conversion_error"
CONVERSION_PHASE: Final = "artifact_stage_conversion"


def emit_completed_layer(
    *,
    sink: object,
    result: object,
    capture: LayerCommitCapture,
    contract: ExecutorAPIContract,
    direct_evaluator: Callable[..., object],
) -> None:
    """Durably append the source parent, then 3N stages, then the layer bundle."""

    if (
        getattr(result, "transport_substeps", None) != capture.level
        or getattr(result, "source_step_index", None) != capture.source_step_index
        or getattr(result, "ptera_step_index", None) != capture.ptera_step_index
        or getattr(result, "row_owner_before_sha256", None)
        != capture.committed_owner_sha256
        or getattr(result, "row_state_before_sha256", None)
        != capture.committed_state_sha256
    ):
        raise ExecutorContractError(
            "layer result differs from its commit-boundary capture"
        )
    stream = getattr(result, "stream_result", None)
    stages = getattr(stream, "stages", None)
    if type(stages) is not tuple or len(stages) != 3 * capture.level:
        raise ExecutorContractError(
            "completed layer lacks its exact 3N compact records"
        )
    add_stage = getattr(sink, "add_transport_stage_from_compact_evidence", None)
    add_source = getattr(sink, "add_source_event", None)
    commit_layer = getattr(sink, "commit_completed_layer", None)
    if (
        not callable(add_stage)
        or not callable(commit_layer)
        or not callable(add_source)
    ):
        raise ExecutorContractError("injected sink lacks the formal stage/layer ABI")
    source_row = capture.source_row()
    add_source(source_row)
    for index, record in enumerate(stages):
        if (
            getattr(record, "substep", None) != index // 3 + 1
            or getattr(record, "stage", None) != index % 3 + 1
        ):
            raise ExecutorContractError("completed compact stage order drift")
        try:
            row, compact = completed_stage_row(
                record, result=result, layer=capture.layer
            )
            add_stage(row, compact)
        except Exception as error:
            raise contract.stop_constructor(
                CONVERSION_STOP_CODE,
                {
                    "transport_substeps": capture.level,
                    "layer": capture.layer,
                    "source_step_index": capture.source_step_index,
                    "ptera_step_index": capture.ptera_step_index,
                    "substep": getattr(record, "substep", None),
                    "stage": getattr(record, "stage", None),
                    "phase": CONVERSION_PHASE,
                    "stage_began": False,
                },
                f"{type(error).__name__}: {error}",
            ) from error
    trajectory = trajectory_array_record(
        result,
        capture,
        contract=contract,
        direct_evaluator=direct_evaluator,
    )
    commit_layer(
        raw_step=raw_step_row(result, capture),
        source_event=source_row,
        owner_event=owner_event_row(result, capture),
        particle_count=particle_count_row(
            result, capture, frontier_count=len(contract.frontier_node_ids)
        ),
        raw_loads=raw_load_rows(result, capture),
        trajectory_array_record=trajectory,
    )


@dataclass(frozen=True, slots=True)
class W2Case:
    strouhal: float = 0.32
    reduced_frequency: float = 1.0
    heave_to_chord: float = 0.50
    period_s: float = 3.56
    chord_m: float = 0.076
    span_m: float = 0.600
    mean_alpha_deg: float = 8.0
    effective_alpha_amplitude_deg: float = 14.0
    pivot_fraction_chord: float = 0.25
    reynolds: float = 5_000.0
    rho_kg_m3: float = 998.2

    @property
    def frequency_hz(self) -> float:
        return 1.0 / self.period_s

    @property
    def freestream_m_s(self) -> float:
        return np.pi * self.frequency_hz * self.chord_m / self.reduced_frequency

    @property
    def nu_m2_s(self) -> float:
        return self.freestream_m_s * self.chord_m / self.reynolds

    @property
    def area_m2(self) -> float:
        return self.chord_m * self.span_m

    @property
    def peak_plunge_induced_alpha_deg(self) -> float:
        return float(
            np.rad2deg(
                _solve_peak_plunge_alpha_rad(
                    self.reduced_frequency * self.heave_to_chord
                )
            )
        )

    @property
    def implemented_pitch_amplitude_deg(self) -> float:
        return self.peak_plunge_induced_alpha_deg - self.effective_alpha_amplitude_deg


W2_CASE: Final = W2Case()


@lru_cache(maxsize=None)
def _solve_peak_plunge_alpha_rad(k_times_h0: float) -> float:
    nodes, weights = np.polynomial.legendre.leggauss(128)
    xi = 0.25 * np.pi * (nodes + 1.0)
    quadrature_weights = 0.25 * np.pi * weights
    target = 2.0 * k_times_h0

    def residual(angle: float) -> float:
        return float(np.sum(quadrature_weights * np.tan(angle * np.sin(xi))) - target)

    lower = 0.0
    upper = float(np.deg2rad(89.0))
    for _ in range(80):
        midpoint = 0.5 * (lower + upper)
        if residual(midpoint) < 0.0:
            lower = midpoint
        else:
            upper = midpoint
    return 0.5 * (lower + upper)


def w2_kinematics(phase: object) -> dict[str, np.ndarray]:
    tau = np.mod(np.asarray(phase, dtype=np.float64), 1.0)
    omega_tau = 2.0 * np.pi * tau
    pitch_deg = -W2_CASE.implemented_pitch_amplitude_deg * np.sin(omega_tau)
    alpha_plunge_rad = np.deg2rad(W2_CASE.peak_plunge_induced_alpha_deg) * np.sin(
        omega_tau
    )
    heave_rate_over_u = -np.tan(alpha_plunge_rad)
    return {
        "pitch_deg": pitch_deg,
        "geometric_alpha_deg": W2_CASE.mean_alpha_deg + pitch_deg,
        "alpha_plunge_deg": np.rad2deg(alpha_plunge_rad),
        "effective_alpha_deg": (
            W2_CASE.mean_alpha_deg
            + pitch_deg
            + np.rad2deg(np.arctan2(-heave_rate_over_u, 1.0))
        ),
        "heave_rate_over_u": heave_rate_over_u,
    }


def _periodic_cumulative_integral(values: np.ndarray, phase: np.ndarray) -> np.ndarray:
    extended_phase = np.concatenate((phase, [1.0]))
    extended_value = np.concatenate((values, values[:1]))
    integral = np.zeros(extended_phase.size, dtype=np.float64)
    integral[1:] = np.cumsum(
        0.5 * (extended_value[1:] + extended_value[:-1]) * np.diff(extended_phase)
    )
    integral -= extended_phase * integral[-1]
    return integral[:-1]


@lru_cache(maxsize=None)
def _heave_spacing_samples(samples: int = 16_384) -> tuple[np.ndarray, ...]:
    phase = np.arange(samples, dtype=np.float64) / samples
    velocity = w2_kinematics(phase)["heave_rate_over_u"]
    displacement = _periodic_cumulative_integral(
        W2_CASE.period_s * W2_CASE.freestream_m_s * velocity,
        phase,
    )
    displacement -= 0.5 * (np.max(displacement) + np.min(displacement))
    amplitude = 0.5 * np.ptp(displacement)
    if amplitude <= 0.0:
        raise ExecutorContractError("W2 heave integration produced zero amplitude")
    return phase, displacement / amplitude, displacement


def _w2_heave_spacing(phase_rad: np.ndarray) -> np.ndarray:
    phase_grid, normalized, _ = _heave_spacing_samples()
    phase = (
        np.mod(np.asarray(phase_rad, dtype=np.float64), 2.0 * np.pi) / (2.0 * np.pi)
        - 0.25
    ) % 1.0
    return np.interp(phase, phase_grid, normalized, period=1.0)


_w2_heave_spacing.__name__ = "baik_w2_sinusoidal_incidence_heave"


def build_w2_movement(pterasoftware: ModuleType) -> object:
    """Build only the audited W2 smoke movement, without importing baik2012."""

    ps = pterasoftware
    outline = np.asarray(
        [[1.0, 1.0e-4], [0.5, 1.0e-4], [0.0, 0.0], [0.5, -1.0e-4], [1.0, -1.0e-4]],
        dtype=np.float64,
    )
    airfoil = ps.geometry.airfoil.Airfoil(
        name="Baik-2012-zero-camber-mean-surface-adapter",
        outline_A_lp=outline,
        resample=False,
    )
    root = ps.geometry.wing_cross_section.WingCrossSection(
        airfoil=airfoil,
        chord=W2_CASE.chord_m,
        num_spanwise_panels=8,
        spanwise_spacing="cosine",
    )
    tip = ps.geometry.wing_cross_section.WingCrossSection(
        airfoil=airfoil,
        chord=W2_CASE.chord_m,
        Lp_Wcsp_Lpp=(0.0, W2_CASE.span_m, 0.0),
        num_spanwise_panels=None,
        spanwise_spacing=None,
    )
    pivot = W2_CASE.pivot_fraction_chord * W2_CASE.chord_m
    wing = ps.geometry.wing.Wing(
        name="Baik 2012 W2 end-plated rectangular adapter",
        wing_cross_sections=[root, tip],
        Ler_Gs_Cgs=(-pivot, 0.0, 0.0),
        angles_Gs_to_Wn_ixyz=(0.0, W2_CASE.mean_alpha_deg, 0.0),
        symmetric=False,
        num_chordwise_panels=2,
        chordwise_spacing="uniform",
    )
    airplane = ps.geometry.airplane.Airplane(
        wings=[wing],
        name="Baik 2012 W2",
        s_ref=W2_CASE.area_m2,
        c_ref=W2_CASE.chord_m,
        b_ref=W2_CASE.span_m,
    )
    section_movements = [
        ps.movements.wing_cross_section_movement.WingCrossSectionMovement(
            base_wing_cross_section=section
        )
        for section in (root, tip)
    ]
    wing_movement = ps.movements.wing_movement.WingMovement(
        base_wing=wing,
        wing_cross_section_movements=section_movements,
        ampLer_Gs_Cgs=(0.0, 0.0, W2_CASE.heave_to_chord * W2_CASE.chord_m),
        periodLer_Gs_Cgs=(0.0, 0.0, W2_CASE.period_s),
        spacingLer_Gs_Cgs=("sine", "sine", _w2_heave_spacing),
        phaseLer_Gs_Cgs=(0.0, 0.0, 90.0),
        ampAngles_Gs_to_Wn_ixyz=(
            0.0,
            W2_CASE.implemented_pitch_amplitude_deg,
            0.0,
        ),
        periodAngles_Gs_to_Wn_ixyz=(0.0, W2_CASE.period_s, 0.0),
        phaseAngles_Gs_to_Wn_ixyz=(0.0, 180.0, 0.0),
        rotationPointOffset_Gs_Ler=(pivot, 0.0, 0.0),
    )
    airplane_movement = ps.movements.airplane_movement.AirplaneMovement(
        base_airplane=airplane,
        wing_movements=[wing_movement],
    )
    operating_point = ps.operating_point.OperatingPoint(
        rho=W2_CASE.rho_kg_m3,
        vCg__E=W2_CASE.freestream_m_s,
        alpha=0.0,
        beta=0.0,
        nu=W2_CASE.nu_m2_s,
    )
    return ps.movements.movement.Movement(
        airplane_movements=[airplane_movement],
        operating_point_movement=(
            ps.movements.operating_point_movement.OperatingPointMovement(
                base_operating_point=operating_point
            )
        ),
        delta_time=W2_CASE.period_s / STEPS_PER_CYCLE,
        num_cycles=2,
        max_wake_cycles=2,
    )


def build_w2_source_events(runtime: RuntimeModules) -> tuple[tuple[object, ...], ...]:
    threshold = runtime.correction.LESPThreshold(
        value=0.11,
        section_family="rounded flat plate",
        reynolds=W2_CASE.reynolds,
        source=(
            "Ramesh 2013 thesis flat-plate Re=1000 sections 4.3.5 and "
            "Figures 4.19/4.21 use Lcrit=0.11; frozen cross-Re/thickness "
            "transfer hypothesis with no Baik force fit"
        ),
        source_role="published_source_input",
    )
    settings = runtime.correction.LDVMSectionSettings(
        ndiv=32,
        naterm=14,
        max_wake_steps=64,
        core_radius_chord=0.02,
    )
    convective_dt = (
        W2_CASE.freestream_m_s * W2_CASE.period_s / W2_CASE.chord_m / STEPS_PER_CYCLE
    )
    sources = tuple(
        runtime.source.V5hDVMSource(
            physical_section_id=f"baik-w2:cell:{cell}:section",
            physical_strip_id=f"baik-w2:cell:{cell}:strip",
            geometry_identity="explicit zero-camber rounded-flat-plate mean-line surrogate",
            reference_speed_m_per_s=W2_CASE.freestream_m_s,
            reference_chord_m=W2_CASE.chord_m,
            zero_camber_surrogate=True,
            delta_time_convective=convective_dt,
            pivot_fraction_chord=W2_CASE.pivot_fraction_chord,
            threshold=threshold,
            settings=settings,
        )
        for cell in range(SOURCE_CELL_COUNT)
    )
    rows: list[tuple[object, ...]] = []
    pitch_amplitude_rad = float(np.deg2rad(W2_CASE.implemented_pitch_amplitude_deg))
    for source_step in ALL_SOURCE_STEPS:
        phase = (source_step - 1) / STEPS_PER_CYCLE
        kinematics = w2_kinematics(phase)
        alpha_rad = float(np.deg2rad(kinematics["geometric_alpha_deg"]))
        alpha_rate = float(
            -2.0
            * W2_CASE.reduced_frequency
            * pitch_amplitude_rad
            * np.cos(2.0 * np.pi * phase)
        )
        heave_rate = float(kinematics["heave_rate_over_u"])
        events = tuple(
            source.step(alpha_rad, alpha_rate, heave_rate) for source in sources
        )
        for event in events:
            if runtime.source.validate_dvm_source_event(event) is not event:
                raise ExecutorContractError(
                    "DVM validator changed source event identity"
                )
            if event.lineage.source_step_index != source_step:
                raise ExecutorContractError("DVM source clock drift")
        rows.append(events)
    result = tuple(rows)
    observed_modes = tuple(
        result[step - 1][0].lev_birth_mode for step in ALL_SOURCE_STEPS
    )
    if observed_modes != ("none", "none", "none", *ACTIVE_BIRTH_MODES):
        raise ExecutorContractError("W2 active source birth schedule drift")
    return result


def _ptera_edge_nodes(solver: object) -> tuple[np.ndarray, np.ndarray]:
    airplanes = tuple(getattr(solver, "current_airplanes", ()))
    if len(airplanes) != 1 or len(airplanes[0].wings) != 1:
        raise ExecutorContractError("W2 row mapping requires one current wing")
    panels = np.asarray(airplanes[0].wings[0].panels, dtype=object)
    if panels.shape != (2, SOURCE_CELL_COUNT):
        raise ExecutorContractError("W2 row mapping requires exact 2x8 panels")
    result: list[np.ndarray] = []
    for chordwise, leading in ((0, True), (-1, False)):
        nodes: list[np.ndarray] = []
        for cell, panel in enumerate(panels[chordwise]):
            if leading:
                left = np.asarray(panel.Flpp_GP1_CgP1, dtype=np.float64)
                right = np.asarray(panel.Frpp_GP1_CgP1, dtype=np.float64)
                edge_flag = bool(panel.is_leading_edge)
            else:
                left = np.asarray(panel.Blpp_GP1_CgP1, dtype=np.float64)
                right = np.asarray(panel.Brpp_GP1_CgP1, dtype=np.float64)
                edge_flag = bool(panel.is_trailing_edge)
            if not edge_flag or left.shape != (3,) or right.shape != (3,):
                raise ExecutorContractError("live Ptera edge topology/shape drift")
            if cell == 0:
                nodes.append(left.copy())
            elif not np.array_equal(nodes[-1], left):
                raise ExecutorContractError("adjacent Ptera edge anchors do not close")
            nodes.append(right.copy())
        edge = np.ascontiguousarray(nodes, dtype=np.float64)
        if edge.shape != (9, 3) or not np.all(np.isfinite(edge)):
            raise ExecutorContractError("mapped Ptera edge is invalid")
        result.append(edge)
    return result[0], result[1]


def _validate_dvm_axes(
    leading: np.ndarray, trailing: np.ndarray, events: tuple[object, ...]
) -> None:
    tolerance = 512.0 * np.finfo(np.float64).eps * W2_CASE.chord_m
    for cell, event in enumerate(events):
        lev = event.lev_placement.edge_anchor_position_over_chord_backend_world
        tev = event.tev_placement.edge_anchor_position_over_chord_backend_world
        if lev is None or tev is None:
            raise ExecutorContractError("source event lacks live LE/TE anchors")
        dvm = np.asarray(
            (
                (tev[0] - lev[0]) * W2_CASE.chord_m,
                0.0,
                (tev[1] - lev[1]) * W2_CASE.chord_m,
            ),
            dtype=np.float64,
        )
        for node in (cell, cell + 1):
            if (
                float(np.max(np.abs((trailing[node] - leading[node]) - dvm)))
                > tolerance
            ):
                raise ExecutorContractError("DVM/Ptera inertial axes disagree")


def _first_birth_nodes(leading: np.ndarray, events: tuple[object, ...]) -> np.ndarray:
    displacements = tuple(
        event.lev_placement.birth_displacement_from_edge_over_chord_backend_world
        for event in events
    )
    if any(event.lev_birth_mode != "first" for event in events):
        raise ExecutorContractError("first source row lacks first-birth events")
    if displacements[0] is None or any(
        value != displacements[0] for value in displacements
    ):
        raise ExecutorContractError("first-birth cell mappings disagree")
    offset = np.asarray(
        (
            displacements[0][0] * W2_CASE.chord_m,
            0.0,
            displacements[0][1] * W2_CASE.chord_m,
        ),
        dtype=np.float64,
    )
    return np.ascontiguousarray(leading + offset, dtype=np.float64)


def _cumulative_lev(events: tuple[object, ...]) -> np.ndarray:
    values = tuple(
        event.kelvin_ledger.gamma_lev_persisted_after
        * event.provenance.circulation_scale_u_times_c_m2_per_s
        for event in events
    )
    result = np.asarray(values, dtype=np.float64)
    if result.shape != (8,) or not np.all(np.isfinite(result)) or np.any(result == 0.0):
        raise ExecutorContractError("active cumulative LEV circulation is invalid")
    return result


class BaikW2RowCommitter:
    """Fresh per-N row owner plus immediate macro-start/source capture."""

    __slots__ = ("runtime", "events", "level", "owner_id", "solver", "captures")

    def __init__(
        self,
        runtime: RuntimeModules,
        events: tuple[tuple[object, ...], ...],
        *,
        level: int,
    ) -> None:
        if (
            level not in FORMAL_LEVELS
            or len(events) != 6
            or any(len(row) != 8 for row in events)
        ):
            raise ExecutorContractError(
                "row committer requires fresh formal N and 6x8 events"
            )
        self.runtime = runtime
        self.events = events
        self.level = level
        self.owner_id = f"fluxv-v5h11-baik-w2-formal-owner-N{level}"
        self.solver: object | None = None
        self.captures: list[LayerCommitCapture] = []

    def bind_solver(self, solver: object) -> None:
        if self.solver is not None:
            raise ExecutorContractError(
                "row committer solver binding is single-assignment"
            )
        self.solver = solver

    def __call__(self, request: object) -> object:
        coupling = self.runtime.coupling
        row_owner = self.runtime.row_owner
        if type(request) is not coupling.V5H11GlobalRowCommitRequest:
            raise TypeError("row request must use the exact v5h11 schema")
        if self.solver is None or request.source_step_index not in SOURCE_STEPS:
            raise ExecutorContractError(
                "row request lacks solver/registered source step"
            )
        layer = SOURCE_STEPS.index(request.source_step_index) + 1
        if layer != len(self.captures) + 1:
            raise ExecutorContractError("row request is not the next layer")
        events = self.events[request.source_step_index - 1]
        if any(event.lev_birth_mode != request.expected_birth_mode for event in events):
            raise ExecutorContractError(
                "row request birth mode differs from source ledger"
            )
        leading, trailing = _ptera_edge_nodes(self.solver)
        _validate_dvm_axes(leading, trailing, events)
        circulation = _cumulative_lev(events)
        if request.previous_owner is None:
            if layer != 1 or request.transported_parent is not None:
                raise ExecutorContractError("first row unexpectedly has ancestry")
            upstream = leading
            downstream = _first_birth_nodes(leading, events)
        else:
            if layer == 1 or request.transported_parent is None:
                raise ExecutorContractError("continuous row lacks transported ancestry")
            upstream = np.ascontiguousarray(
                request.previous_owner.state.live_boundary_nodes, dtype=np.float64
            )
            downstream = np.ascontiguousarray(
                leading + (upstream - leading) / 3.0, dtype=np.float64
            )
        row = row_owner.make_release_row(
            upstream,
            downstream,
            circulation,
            release_index=layer,
            source_time_s=request.source_time_s,
            sheet_id=ROW_SHEET_ID,
        )
        if request.previous_owner is None:
            owner = row_owner.bootstrap_release_row_owner(
                row,
                smoothing_radius_m=ROW_SMOOTHING_RADIUS_M,
                target_spacing_m=ROW_TARGET_SPACING_M,
                release_dt_s=DELTA_TIME_S,
                particle_cap=ROW_PARTICLE_CAP,
                owner_id=self.owner_id,
            )
            result = row_owner.RowCommitResult(
                committed=True,
                status="compatible",
                owner=owner,
                state=owner.state,
                event=None,
                first_mismatch=None,
            )
        else:
            proposal = row_owner.propose_release_row_update(
                request.previous_owner,
                row,
                proposal_id=f"baik-w2-N{self.level}-source-{request.source_step_index}",
            )
            if proposal.status != "compatible":
                raise ExecutorContractError(
                    f"row remesh required: {proposal.first_mismatch}"
                )
            result = row_owner.commit_release_row_update(
                request.previous_owner, proposal
            )
            if not result.committed or result.status != "compatible":
                raise ExecutorContractError("compatible row proposal did not commit")
        aggregate = aggregate_source_events(
            events,
            level=self.level,
            source_step_index=request.source_step_index,
            ptera_step_index=request.ptera_step_index,
            source_time_s=request.source_time_s,
        )
        kelvin = coupling.make_v5h11_source_kelvin_evidence(
            source_step_index=request.source_step_index,
            row_owner_sha256=result.owner.owner_sha256,
            source_event_sha256=aggregate.row["event_sha256"],
            kelvin_ledger_sha256=aggregate.kelvin_ledger_sha256,
            residual_m2_s=aggregate.residual_max_abs,
        )
        capture = capture_committed_row(
            level=self.level,
            layer=layer,
            request=request,
            events=events,
            source_prehistory_events=(self.events[:3] if layer == 1 else ()),
            commit_result=result,
            source_kelvin_evidence=kelvin,
        )
        self.captures.append(capture)
        return coupling.V5H11RowCommitEnvelope(
            commit_result=result,
            source_kelvin_evidence=kelvin,
        )


def _make_disposable_layer1_smoke_summary(
    *,
    runtime: RuntimeModules,
    contract: ExecutorAPIContract,
    result: object,
    capture: LayerCommitCapture,
) -> DisposableLayer1SmokeSummary:
    """Reduce one registry-validated live result to scalar diagnostic evidence."""

    if runtime.coupling.validate_v5h11_layer_result(result) is not result:
        raise ExecutorContractError(
            "coupling validator changed disposable layer identity"
        )
    if (
        capture.level != DISPOSABLE_SMOKE_LEVEL
        or capture.layer != 1
        or capture.source_step_index != SOURCE_STEPS[0]
        or capture.ptera_step_index != PTERA_STEPS[0]
        or getattr(result, "scope", None) != "diagnostic_smoke"
        or getattr(result, "substep_role", None) != "diagnostic_smoke"
        or getattr(result, "transport_substeps", None) != capture.level
        or getattr(result, "source_step_index", None) != capture.source_step_index
        or getattr(result, "ptera_step_index", None) != capture.ptera_step_index
        or getattr(result, "row_owner_before_sha256", None)
        != capture.committed_owner_sha256
    ):
        raise ExecutorContractError(
            "disposable layer result differs from its commit-boundary capture"
        )
    stream = getattr(result, "stream_result", None)
    stages = getattr(stream, "stages", None)
    expected_stages = 3 * DISPOSABLE_SMOKE_LEVEL
    if type(stages) is not tuple or len(stages) != expected_stages:
        raise ExecutorContractError(
            "disposable layer lacks its exact 96 compact stage records"
        )
    invariant_values = tuple(
        getattr(stage, "invariant_residual_over_slog_max", None) for stage in stages
    )
    if any(
        type(value) is not float or not math.isfinite(value) or value < 0.0
        for value in invariant_values
    ):
        raise ExecutorContractError(
            "disposable compact stages lack finite invariant evidence"
        )
    final_state = getattr(stream, "final_state", None)
    positions = np.asarray(getattr(final_state, "positions", None), dtype=np.float64)
    gamma = np.asarray(getattr(final_state, "gamma", None), dtype=np.float64)
    sigma = np.asarray(getattr(final_state, "sigma", None), dtype=np.float64)
    final_tracers = np.asarray(getattr(stream, "final_tracer_positions", None))
    if (
        positions.ndim != 2
        or positions.shape[1:] != (3,)
        or gamma.shape != positions.shape
        or sigma.shape != (positions.shape[0],)
        or sigma.size == 0
        or not np.all(np.isfinite(positions))
        or not np.all(np.isfinite(gamma))
        or not np.all(np.isfinite(sigma))
        or np.any(sigma <= 0.0)
        or final_tracers.ndim < 1
    ):
        raise ExecutorContractError("disposable endpoint state is invalid")
    counters = getattr(result, "counters", None)
    fd_ledger = getattr(result, "fd_call_ledger", None)
    stability = getattr(result, "stability_envelope", None)
    support = getattr(result, "support_envelope", None)
    load = getattr(result, "load_ledger", None)
    kelvin = getattr(result, "source_kelvin_evidence", None)
    if (
        getattr(stability, "stage_count", None) != expected_stages
        or getattr(counters, "transport_stage_count", None) != expected_stages
        or getattr(counters, "direct_field_call_count", None) != 2 * expected_stages
        or getattr(counters, "ptera_center_call_count", None) != 2 * expected_stages
        or getattr(counters, "ptera_offset_call_count", None) != 6 * expected_stages
        or getattr(fd_ledger, "physical_evaluation_count", None) != expected_stages
        or getattr(fd_ledger, "tracer_evaluation_count", None) != expected_stages
        or getattr(fd_ledger, "center_call_count", None) != 2 * expected_stages
        or getattr(fd_ledger, "offset_call_count", None) != 6 * expected_stages
        or getattr(fd_ledger, "evaluator_call_count", None) != 8 * expected_stages
    ):
        raise ExecutorContractError("disposable stage/FD count contract drift")
    particle_count = len(capture.particle_ids)
    material_support_count = len(capture.material_tracer_ids)
    frontier_count = len(contract.frontier_node_ids)
    material_count = int(final_tracers.shape[0])
    if (
        particle_count != positions.shape[0]
        or getattr(support, "support_count", None) != material_support_count
        or getattr(support, "frontier_count", None) != frontier_count
        or material_count != material_support_count + frontier_count
    ):
        raise ExecutorContractError("disposable material/frontier counts drift")
    source_row = capture.source_row()
    source_event_sha = source_row.get("event_sha256")
    draft = DisposableLayer1SmokeSummary(
        schema_id=DISPOSABLE_SMOKE_SUMMARY_SCHEMA_ID,
        case_id="W2",
        scope="diagnostic_smoke",
        transport_substeps=DISPOSABLE_SMOKE_LEVEL,
        active_layer_limit=DISPOSABLE_SMOKE_LAYER_LIMIT,
        layer=1,
        source_step_index=SOURCE_STEPS[0],
        ptera_step_index=PTERA_STEPS[0],
        source_event_sha256=source_event_sha,
        source_cell_manifest_sha256=capture.source_cell_manifest_sha256,
        source_prehistory_manifest_sha256=(capture.source_prehistory_manifest_sha256),
        layer_result_sha256=getattr(result, "result_sha256", None),
        stream_result_sha256=getattr(result, "stream_result_sha256", None),
        stream_stage_chain_sha256=getattr(result, "stream_stage_chain_sha256", None),
        fd_ledger_sha256=getattr(result, "fd_ledger_sha256", None),
        load_ledger_sha256=getattr(result, "load_ledger_sha256", None),
        source_kelvin_evidence_sha256=getattr(
            result, "source_kelvin_evidence_sha256", None
        ),
        row_owner_before_sha256=getattr(result, "row_owner_before_sha256", None),
        advanced_owner_sha256=getattr(result, "advanced_owner_sha256", None),
        advanced_state_sha256=getattr(result, "advanced_state_sha256", None),
        ptera_parent_sha256_before=getattr(
            result, "ptera_parent_sha256_before_transport", None
        ),
        ptera_parent_sha256_after=getattr(
            result, "ptera_parent_sha256_after_transport", None
        ),
        particle_count=particle_count,
        material_tracer_count=material_count,
        material_support_tracer_count=material_support_count,
        frontier_node_tracer_count=frontier_count,
        transport_stage_count=getattr(counters, "transport_stage_count", None),
        direct_field_call_count=getattr(counters, "direct_field_call_count", None),
        ptera_center_call_count=getattr(counters, "ptera_center_call_count", None),
        ptera_offset_call_count=getattr(counters, "ptera_offset_call_count", None),
        fd_physical_evaluation_count=getattr(
            fd_ledger, "physical_evaluation_count", None
        ),
        fd_tracer_evaluation_count=getattr(fd_ledger, "tracer_evaluation_count", None),
        fd_evaluator_call_count=getattr(fd_ledger, "evaluator_call_count", None),
        max_invariant_residual_over_slog=max(invariant_values),
        max_h_jacobian_frobenius=getattr(stability, "max_h_jacobian_frobenius", None),
        max_h_convective_over_sigma=getattr(
            stability, "max_h_convective_over_sigma", None
        ),
        sigma_min_m=float(np.min(sigma)),
        sigma_max_m=float(np.max(sigma)),
        no_penetration_max_abs=getattr(load, "no_penetration_max_abs", None),
        kelvin_residual_max_abs_m2_s=abs(getattr(kelvin, "residual_m2_s", None)),
        observation_access=getattr(result, "observation_access", None),
        force_scoring_status=getattr(result, "force_scoring_status", None),
        artifact_persistence="none",
        summary_sha256="",
    )
    values = asdict(draft)
    values["summary_sha256"] = _disposable_smoke_summary_sha256(draft)
    summary = DisposableLayer1SmokeSummary(**values)
    return validate_disposable_layer1_smoke_summary(summary)


def _callable_binding(value: object) -> tuple[object, object, object, object]:
    """Capture the executable parts of one trusted Python callable."""

    function = getattr(value, "__func__", value)
    return (
        function,
        getattr(function, "__code__", None),
        getattr(function, "__defaults__", None),
        getattr(function, "__kwdefaults__", None),
    )


def _assert_callable_binding(
    name: str,
    value: object,
    expected: tuple[object, object, object, object],
) -> None:
    # Keep this verifier independent of its module-global helper binding.  The
    # disposable smoke entry point stores this function and verifies its own
    # executable tuple before using it, so rebinding ``_callable_binding`` (or
    # this public module name) cannot turn the verifier into a no-op.
    function = getattr(value, "__func__", value)
    observed = (
        function,
        getattr(function, "__code__", None),
        getattr(function, "__defaults__", None),
        getattr(function, "__kwdefaults__", None),
    )
    if any(current is not frozen for current, frozen in zip(observed, expected)):
        raise DependencyBindingError(f"trusted disposable-smoke callable drift: {name}")


class FormalCouplingExecutor:
    """Class-identity-safe executor constructed only from the injected API."""

    __slots__ = (
        "_contract",
        "_level_runner",
        "_runtime_loader",
        "_smoke_runner",
        "_smoke_validator",
        "_dependency_attester",
        "_smoke_binding_verifier",
        "_smoke_assert_callable",
        "_smoke_callable_bindings",
        "_production_smoke_runner",
    )

    def __init__(
        self,
        api: object,
        *,
        runtime_loader: Callable[[ExecutorAPIContract], RuntimeModules] = (
            _load_verified_runtime
        ),
        level_runner: Callable[[RuntimeModules, ExecutorAPIContract, int, object], None]
        | None = None,
        smoke_runner: Callable[
            [RuntimeModules, ExecutorAPIContract], DisposableLayer1SmokeSummary
        ]
        | None = None,
        smoke_validator: Callable[
            [DisposableLayer1SmokeSummary], DisposableLayer1SmokeSummary
        ] = validate_disposable_layer1_smoke_summary,
    ) -> None:
        self._contract = _validated_api(api)
        _preflight_all_leaf_bytes(self._contract)
        _attest_loaded_runtime_modules(self._contract, require_complete=False)
        self._runtime_loader = runtime_loader
        self._level_runner = level_runner
        self._smoke_runner = smoke_runner
        self._smoke_validator = smoke_validator
        self._dependency_attester = self.attest_dependency_origins
        self._smoke_binding_verifier = self._assert_smoke_callable_bindings
        self._smoke_assert_callable = _assert_callable_binding
        self._production_smoke_runner = bool(
            smoke_runner is not None
            and globals().get("_run_disposable_n32_layer1_smoke") is smoke_runner
        )
        self._smoke_callable_bindings = (
            _callable_binding(self._dependency_attester),
            _callable_binding(smoke_validator),
            _callable_binding(_disposable_smoke_summary_sha256),
            _callable_binding(_disposable_smoke_summary_payload),
            None if smoke_runner is None else _callable_binding(smoke_runner),
            _callable_binding(self._smoke_binding_verifier),
            _callable_binding(self._smoke_assert_callable),
            _callable_binding(_make_disposable_layer1_smoke_summary),
            _callable_binding(runtime_loader),
        )

    def attest_dependency_origins(
        self,
        _preflight: Callable[[ExecutorAPIContract], None] = _preflight_all_leaf_bytes,
        _attest_runtime: Callable[..., None] = _attest_loaded_runtime_modules,
    ) -> None:
        """Idempotently rehash leaves and attest every already-loaded origin."""

        _preflight(self._contract)
        _attest_runtime(self._contract, require_complete=False)

    def _assert_smoke_callable_bindings(
        self,
        _assert_binding: Callable[
            [str, object, tuple[object, object, object, object]], None
        ] = _assert_callable_binding,
    ) -> None:
        bindings = self._smoke_callable_bindings
        _assert_binding("dependency_attester", self._dependency_attester, bindings[0])
        if type(self).attest_dependency_origins is not getattr(
            self._dependency_attester, "__func__", None
        ):
            raise DependencyBindingError(
                "trusted disposable-smoke dependency attester was rebound"
            )
        _assert_binding("summary_validator", self._smoke_validator, bindings[1])
        _assert_binding(
            "summary_sha256",
            _disposable_smoke_summary_sha256,
            bindings[2],
        )
        _assert_binding(
            "summary_payload",
            _disposable_smoke_summary_payload,
            bindings[3],
        )
        _assert_binding(
            "summary_builder",
            _make_disposable_layer1_smoke_summary,
            bindings[7],
        )
        _assert_binding("runtime_loader", self._runtime_loader, bindings[8])
        if globals().get("validate_disposable_layer1_smoke_summary") is not (
            self._smoke_validator
        ):
            raise DependencyBindingError(
                "trusted disposable-smoke summary validator was rebound"
            )
        runner = self._smoke_runner
        if runner is None or bindings[4] is None:
            raise DependencyBindingError("trusted disposable-smoke runner is unbound")
        _assert_binding("smoke_runner", runner, bindings[4])
        if (
            self._production_smoke_runner
            and globals().get("_run_disposable_n32_layer1_smoke") is not runner
        ):
            raise DependencyBindingError(
                "trusted disposable-smoke production runner was rebound"
            )

    def run_formal_matrix(self, *, levels: tuple[int, int, int], sink: object) -> None:
        if type(levels) is not tuple or levels != FORMAL_LEVELS:
            raise ExecutorContractError(
                "formal executor requires exact levels (32,64,128)"
            )
        if type(sink) is not self._contract.sink_type:
            raise ExecutorContractError("formal executor received a foreign sink type")
        self.attest_dependency_origins()
        runtime = self._runtime_loader(self._contract)
        _attest_loaded_runtime_modules(self._contract, require_complete=False)
        runner = self._level_runner or _run_formal_level
        for level in levels:
            runner(runtime, self._contract, level, sink)

    def run_disposable_n32_layer1_smoke(self) -> DisposableLayer1SmokeSummary:
        """Run one real N32 layer without a sink, files, GT, or scoring."""

        bindings = self._smoke_callable_bindings

        def assert_trusted_bindings() -> None:
            # Establish the verifier root with raw executable-identity checks
            # before invoking either stored verifier.  This closes the
            # second-order attack in which a callback rebinds the class method
            # or the module-global assertion helper and then forges the runner
            # and validator together.
            assert_callable = self._smoke_assert_callable
            function = getattr(assert_callable, "__func__", assert_callable)
            observed = (
                function,
                getattr(function, "__code__", None),
                getattr(function, "__defaults__", None),
                getattr(function, "__kwdefaults__", None),
            )
            if any(
                current is not frozen for current, frozen in zip(observed, bindings[6])
            ):
                raise DependencyBindingError(
                    "trusted disposable-smoke binding verifier drift"
                )
            verifier = self._smoke_binding_verifier
            assert_callable("binding_verifier", verifier, bindings[5])
            verifier()

        assert_trusted_bindings()
        self._dependency_attester()
        runtime = self._runtime_loader(self._contract)
        assert_trusted_bindings()
        self._dependency_attester()
        assert_trusted_bindings()
        runner = self._smoke_runner
        assert runner is not None
        raw_summary = runner(runtime, self._contract)
        assert_trusted_bindings()
        summary = self._smoke_validator(raw_summary)
        assert_trusted_bindings()
        self._dependency_attester()
        assert_trusted_bindings()
        final_summary = self._smoke_validator(summary)
        assert_trusted_bindings()
        return final_summary


@dataclass(frozen=True, slots=True)
class _StoppedLayerContext:
    transport_substeps: int
    ptera_parent_sha256_before_transport: str
    ptera_parent_sha256_after_transport: str
    fd_call_ledger: None = None


def _emit_stream_stop(
    *,
    runtime: RuntimeModules,
    contract: ExecutorAPIContract,
    sink: object,
    solver: object,
    error: object,
    capture: LayerCommitCapture,
) -> None:
    add_source = getattr(sink, "add_source_event", None)
    add_completed_stage = getattr(
        sink, "add_transport_stage_from_compact_evidence", None
    )
    add_failed_stage = getattr(sink, "add_failed_transport_stage", None)
    if (
        not callable(add_source)
        or not callable(add_completed_stage)
        or not callable(add_failed_stage)
    ):
        raise ExecutorContractError("injected sink lacks STOP prefix append methods")
    add_source(capture.source_row())
    failure_time_parent_sha = runtime.ptera_transport.ptera_parent_state_sha256(solver)
    if not _is_sha256(failure_time_parent_sha):
        raise ExecutorContractError("failed layer Ptera parent digest is invalid")
    completed = getattr(error, "completed_stages", None)
    count = getattr(error, "completed_stage_count", None)
    if (
        type(completed) is not tuple
        or type(count) is not int
        or count != len(completed)
    ):
        raise ExecutorContractError("StreamStopped completed prefix schema is invalid")
    substep = getattr(error, "substep", None)
    stage = getattr(error, "stage", None)
    if (
        type(substep) is not int
        or not 1 <= substep <= capture.level
        or type(stage) is not int
        or stage not in (1, 2, 3)
        or count != 3 * (substep - 1) + stage - 1
    ):
        raise ExecutorContractError(
            "StreamStopped coordinate/completed prefix disagree"
        )
    parent_sha = failure_time_parent_sha
    if completed:
        expected_prefix = f"v5h11:ptera-step:{capture.ptera_step_index}:parent:"
        tokens = tuple(getattr(record, "parent_token", None) for record in completed)
        if (
            any(
                type(token) is not str or not token.startswith(expected_prefix)
                for token in tokens
            )
            or len(set(tokens)) != 1
        ):
            raise ExecutorContractError("StreamStopped parent token prefix drift")
        parent_sha = tokens[0][len(expected_prefix) :]
        if not _is_sha256(parent_sha) or parent_sha != failure_time_parent_sha:
            raise ExecutorContractError(
                "StreamStopped frozen parent differs from failure-time parent"
            )
    context = _StoppedLayerContext(
        transport_substeps=capture.level,
        ptera_parent_sha256_before_transport=parent_sha,
        ptera_parent_sha256_after_transport=failure_time_parent_sha,
    )
    for index, record in enumerate(completed):
        if (
            getattr(record, "substep", None) != index // 3 + 1
            or getattr(record, "stage", None) != index % 3 + 1
        ):
            raise ExecutorContractError("StreamStopped completed prefix order drift")
        row, compact = completed_stage_row(
            record,
            result=context,
            layer=capture.layer,
            verify_fd_ledger=False,
        )
        add_completed_stage(row, compact)
    stage_began = getattr(error, "stage_began", None)
    if type(stage_began) is not bool:
        raise ExecutorContractError("StreamStopped stage_began is not an exact bool")
    if stage_began:
        row, _ = failed_stage_row(
            error,
            level=capture.level,
            layer=capture.layer,
        )
        add_failed_stage(row)
    coordinate = {
        "transport_substeps": capture.level,
        "layer": capture.layer,
        "source_step_index": capture.source_step_index,
        "ptera_step_index": capture.ptera_step_index,
        "substep": substep,
        "stage": stage,
        "phase": "stream:" + str(getattr(error, "failure_phase", "unknown")),
        "stage_began": stage_began,
    }
    raise contract.stop_constructor(
        "ir_wrk3_stream_stopped",
        coordinate,
        str(error),
    ) from error


def _emit_completed_result_prefix(
    *,
    runtime: RuntimeModules,
    contract: ExecutorAPIContract,
    sink: object,
    results: tuple[object, ...],
    captures: tuple[LayerCommitCapture, ...],
) -> None:
    if len(results) != len(captures) or len(results) > 3:
        raise ExecutorContractError("completed result/capture prefix lengths disagree")
    for result, capture in zip(results, captures, strict=True):
        if runtime.coupling.validate_v5h11_layer_result(result) is not result:
            raise ExecutorContractError(
                "coupling validator changed layer prefix identity"
            )
        emit_completed_layer(
            sink=sink,
            result=result,
            capture=capture,
            contract=contract,
            direct_evaluator=runtime.reference.direct_gaussian_erf_velocity_jacobian,
        )


def _emit_generic_solver_stop(
    *,
    runtime: RuntimeModules,
    contract: ExecutorAPIContract,
    sink: object,
    level: int,
    error: BaseException,
    results: tuple[object, ...],
    captures: tuple[LayerCommitCapture, ...],
) -> None:
    """Preserve all completed layers and an optional committed failing source."""

    if len(captures) not in (len(results), len(results) + 1) or len(results) >= 3:
        raise ExecutorContractError(
            "generic solver failure result/capture prefix is inconsistent"
        ) from error
    _emit_completed_result_prefix(
        runtime=runtime,
        contract=contract,
        sink=sink,
        results=results,
        captures=captures[: len(results)],
    )
    failing_capture = captures[-1] if len(captures) == len(results) + 1 else None
    if failing_capture is not None:
        add_source = getattr(sink, "add_source_event", None)
        if not callable(add_source):
            raise ExecutorContractError("injected sink lacks source-prefix append")
        add_source(failing_capture.source_row())
    next_layer = len(results) + 1
    source_step = (
        failing_capture.source_step_index
        if failing_capture is not None
        else SOURCE_STEPS[next_layer - 1]
    )
    ptera_step = (
        failing_capture.ptera_step_index
        if failing_capture is not None
        else PTERA_STEPS[next_layer - 1]
    )
    coordinate = {
        "transport_substeps": level,
        "layer": None,
        "source_step_index": source_step,
        "ptera_step_index": ptera_step,
        "substep": None,
        "stage": None,
        "phase": "solver:non_stream_failure",
        "stage_began": False,
    }
    raise contract.stop_constructor(
        "formal_solver_stopped",
        coordinate,
        f"{type(error).__name__}: {error}",
    ) from error


def _run_disposable_n32_layer1_smoke(
    runtime: RuntimeModules,
    contract: ExecutorAPIContract,
    _summary_builder: Callable[..., DisposableLayer1SmokeSummary] = (
        _make_disposable_layer1_smoke_summary
    ),
) -> DisposableLayer1SmokeSummary:
    """Use the formal runtime to close only N32/layer-1, without a sink."""

    events = build_w2_source_events(runtime)
    movement = build_w2_movement(runtime.pterasoftware)
    problem = runtime.pterasoftware.problems.UnsteadyProblem(
        movement=movement,
        only_final_results=False,
    )
    committer = BaikW2RowCommitter(
        runtime,
        events,
        level=DISPOSABLE_SMOKE_LEVEL,
    )
    config = runtime.coupling.V5H11BaikCouplingConfig(
        transport_substeps=DISPOSABLE_SMOKE_LEVEL,
        formal_matrix=False,
        test_mode=False,
        diagnostic_smoke=True,
        active_layer_limit=DISPOSABLE_SMOKE_LAYER_LIMIT,
        particle_cap=ROW_PARTICLE_CAP,
    )
    solver = runtime.coupling.make_fluxv_v5h11_baik_w2_solver(
        problem,
        row_committer=committer,
        config=config,
        max_particles=ROW_PARTICLE_CAP,
        stretch=False,
        free_wake=False,
    )
    committer.bind_solver(solver)
    solver.run(
        prescribed_wake=True,
        calculate_streamlines=False,
        show_progress=False,
    )
    results = tuple(getattr(solver, "v5h11_layer_results", ()))
    captures = tuple(committer.captures)
    if len(results) != 1 or len(captures) != 1:
        raise ExecutorContractError(
            "disposable N32 diagnostic did not close exactly one layer"
        )
    return _summary_builder(
        runtime=runtime,
        contract=contract,
        result=results[0],
        capture=captures[0],
    )


def _run_formal_level(
    runtime: RuntimeModules,
    contract: ExecutorAPIContract,
    level: int,
    sink: object,
) -> None:
    """Create one fresh source/owner/problem/solver and commit its three layers."""

    if level not in FORMAL_LEVELS:
        raise ExecutorContractError("formal level is outside (32,64,128)")
    events = build_w2_source_events(runtime)
    movement = build_w2_movement(runtime.pterasoftware)
    problem = runtime.pterasoftware.problems.UnsteadyProblem(
        movement=movement,
        only_final_results=False,
    )
    committer = BaikW2RowCommitter(runtime, events, level=level)
    config = runtime.coupling.V5H11BaikCouplingConfig(
        transport_substeps=level,
        formal_matrix=True,
        test_mode=False,
        particle_cap=ROW_PARTICLE_CAP,
    )
    solver = runtime.coupling.make_fluxv_v5h11_baik_w2_solver(
        problem,
        row_committer=committer,
        config=config,
        max_particles=ROW_PARTICLE_CAP,
        stretch=False,
        free_wake=False,
    )
    committer.bind_solver(solver)
    try:
        solver.run(
            prescribed_wake=True,
            calculate_streamlines=False,
            show_progress=False,
        )
    except runtime.stream.IRWRK3StreamStopped as error:
        completed_results = tuple(getattr(solver, "v5h11_layer_results", ()))
        if (
            len(completed_results) >= 3
            or len(committer.captures) != len(completed_results) + 1
        ):
            raise ExecutorContractError(
                "failed formal level result/capture prefix is inconsistent"
            ) from error
        _emit_completed_result_prefix(
            runtime=runtime,
            contract=contract,
            sink=sink,
            results=completed_results,
            captures=tuple(committer.captures[: len(completed_results)]),
        )
        _emit_stream_stop(
            runtime=runtime,
            contract=contract,
            sink=sink,
            solver=solver,
            error=error,
            capture=committer.captures[-1],
        )
    except Exception as error:
        _emit_generic_solver_stop(
            runtime=runtime,
            contract=contract,
            sink=sink,
            level=level,
            error=error,
            results=tuple(getattr(solver, "v5h11_layer_results", ())),
            captures=tuple(committer.captures),
        )
    results = tuple(getattr(solver, "v5h11_layer_results", ()))
    captures = tuple(committer.captures)
    if len(results) != 3 or len(captures) != 3:
        raise ExecutorContractError("formal level did not close exactly three layers")
    for layer, (result, capture) in enumerate(
        zip(results, captures, strict=True), start=1
    ):
        if capture.layer != layer:
            raise ExecutorContractError("formal layer capture order drift")
        if runtime.coupling.validate_v5h11_layer_result(result) is not result:
            raise ExecutorContractError("coupling validator changed layer identity")
        emit_completed_layer(
            sink=sink,
            result=result,
            capture=capture,
            contract=contract,
            direct_evaluator=runtime.reference.direct_gaussian_erf_velocity_jacobian,
        )


def build_fluxv_v5h12_w2_executor(api: object) -> FormalCouplingExecutor:
    """Factory literal consumed by the runner after all leaf preflight gates."""

    return FormalCouplingExecutor(
        api,
        smoke_runner=_run_disposable_n32_layer1_smoke,
        smoke_validator=validate_disposable_layer1_smoke_summary,
    )


__all__ = (
    "DependencyBindingError",
    "DisposableLayer1SmokeSummary",
    "ExecutorContractError",
    "FormalCouplingExecutor",
    "build_fluxv_v5h12_w2_executor",
    "completed_stage_row",
    "failed_stage_row",
    "parse_stage_evidence",
    "validate_disposable_layer1_smoke_summary",
)
