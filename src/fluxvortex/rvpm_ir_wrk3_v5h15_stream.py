"""Streaming invariant-reconstructed Williamson RK3 macro integrator.

Unlike the audit-heavy reference integrator, this module retains no stage
arrays.  Each stage exposes one immutable, ephemeral view to an optional
observer and stores only bounded evidence plus hashes in the final result.
The continuous mechanics remain the pinned ``f=0, g=1/5`` IR-WRK3 equations.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import partial
import hashlib
import inspect
import json
import math
import threading
from collections.abc import Sequence
from typing import Callable, Final

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .rvpm_ir_wrk3 import (
    INVARIANT_LOG_ATOL,
    IRWRK3Field,
    IRWRK3TracerField,
    InvariantReference,
    freeze_invariant_reference,
    make_ir_wrk3_field,
    make_ir_wrk3_tracer_field,
    reconstruct_sigma,
)
from .rvpm_transport import ParticleState

FloatArray = NDArray[np.float64]

MAX_PARTICLE_COUNT: Final[int] = 1_000_000
MAX_TRACER_COUNT: Final[int] = 1_000_000
MAX_SUBSTEPS: Final[int] = 4096
MAX_EVIDENCE_BYTES_PER_STAGE: Final[int] = 4096
MAX_TOTAL_EVIDENCE_BYTES: Final[int] = 16_777_216
MAX_LIVE_STREAM_RESULT_COUNT: Final[int] = 4096

_RK_A: Final[tuple[float, float, float]] = (
    0.0,
    -5.0 / 9.0,
    -153.0 / 128.0,
)
_RK_B: Final[tuple[float, float, float]] = (
    1.0 / 3.0,
    15.0 / 16.0,
    8.0 / 15.0,
)
_STAGE_CHAIN_GENESIS: Final[str] = hashlib.sha256(
    b"fluxv-ir-wrk3-stream-stage-chain-v1"
).hexdigest()

_CONTRACT_LITERAL: Final[tuple[object, ...]] = (
    1_000_000,
    1_000_000,
    4096,
    4096,
    16_777_216,
    4096,
    (0.0, -5.0 / 9.0, -153.0 / 128.0),
    (1.0 / 3.0, 15.0 / 16.0, 8.0 / 15.0),
    512.0 * np.finfo(np.float64).eps,
)
_NUMPY_BINDING_NAMES: Final[tuple[str, ...]] = (
    "abs",
    "all",
    "any",
    "array",
    "asarray",
    "ascontiguousarray",
    "bool_",
    "count_nonzero",
    "dtype",
    "einsum",
    "empty",
    "finfo",
    "float64",
    "floating",
    "frombuffer",
    "integer",
    "isfinite",
    "log",
    "max",
    "maximum",
    "ndarray",
    "ones",
    "ones_like",
    "sqrt",
    "zeros",
    "zeros_like",
)
_FROZEN_NUMPY_BINDINGS: Final[tuple[tuple[str, object], ...]] = tuple(
    (name, getattr(np, name)) for name in _NUMPY_BINDING_NAMES
)
_FROZEN_PUBLIC_BINDINGS: Final[tuple[tuple[str, object], ...]] = (
    ("ParticleState", ParticleState),
    ("IRWRK3Field", IRWRK3Field),
    ("IRWRK3TracerField", IRWRK3TracerField),
    ("InvariantReference", InvariantReference),
    ("freeze_invariant_reference", freeze_invariant_reference),
    ("make_ir_wrk3_field", make_ir_wrk3_field),
    ("make_ir_wrk3_tracer_field", make_ir_wrk3_tracer_field),
    ("reconstruct_sigma", reconstruct_sigma),
)
_FROZEN_MATH_BINDINGS: Final[tuple[tuple[str, object], ...]] = (
    ("isfinite", math.isfinite),
)
_FROZEN_HASHLIB_BINDINGS: Final[tuple[tuple[str, object], ...]] = (
    ("sha256", hashlib.sha256),
)


def _assert_runtime_bindings(
    numpy_bindings: tuple[tuple[str, object], ...] = _FROZEN_NUMPY_BINDINGS,
    public_bindings: tuple[tuple[str, object], ...] = _FROZEN_PUBLIC_BINDINGS,
    math_bindings: tuple[tuple[str, object], ...] = _FROZEN_MATH_BINDINGS,
    hashlib_bindings: tuple[tuple[str, object], ...] = _FROZEN_HASHLIB_BINDINGS,
    contract: tuple[object, ...] = _CONTRACT_LITERAL,
) -> None:
    registries = (
        ("_FROZEN_NUMPY_BINDINGS", numpy_bindings),
        ("_FROZEN_PUBLIC_BINDINGS", public_bindings),
        ("_FROZEN_MATH_BINDINGS", math_bindings),
        ("_FROZEN_HASHLIB_BINDINGS", hashlib_bindings),
        ("_CONTRACT_LITERAL", contract),
    )
    for name, frozen in registries:
        if globals().get(name) is not frozen:
            raise RuntimeError(f"IR-WRK3 stream runtime registry drift: {name}")
    for name, frozen in numpy_bindings:
        if getattr(np, name) is not frozen:
            raise RuntimeError(f"IR-WRK3 stream NumPy binding drift: {name}")
    for name, frozen in public_bindings:
        if globals().get(name) is not frozen:
            raise RuntimeError(f"IR-WRK3 stream public binding drift: {name}")
    for name, frozen in math_bindings:
        if getattr(math, name) is not frozen:
            raise RuntimeError(f"IR-WRK3 stream math binding drift: {name}")
    for name, frozen in hashlib_bindings:
        if getattr(hashlib, name) is not frozen:
            raise RuntimeError(f"IR-WRK3 stream hashlib binding drift: {name}")
    observed = (
        MAX_PARTICLE_COUNT,
        MAX_TRACER_COUNT,
        MAX_SUBSTEPS,
        MAX_EVIDENCE_BYTES_PER_STAGE,
        MAX_TOTAL_EVIDENCE_BYTES,
        MAX_LIVE_STREAM_RESULT_COUNT,
        _RK_A,
        _RK_B,
        INVARIANT_LOG_ATOL,
    )
    if any(
        type(current) is not type(expected) or current != expected
        for current, expected in zip(observed, contract, strict=True)
    ):
        raise RuntimeError("IR-WRK3 stream frozen contract drift")


def _preflight_count(
    name: str,
    value: object,
    shape_tail: tuple[int, ...],
    *,
    cap: int,
) -> int:
    """Reject oversize inputs from metadata before any array materialization."""

    shape = getattr(value, "shape", None)
    if shape is not None:
        if type(shape) is not tuple or any(
            type(dimension) is not int for dimension in shape
        ):
            raise ValueError(f"{name} must expose an exact integer shape tuple")
    else:
        try:
            count = len(value)  # type: ignore[arg-type]
        except (TypeError, AttributeError) as error:
            raise ValueError(f"{name} must expose a leading dimension") from error
        if type(count) is not int:
            raise ValueError(f"{name} leading dimension must be an exact integer")
        if count < 0:
            raise ValueError(f"{name} leading dimension must be nonnegative")
        if count > cap:
            raise ValueError(f"{name} cap exceeded before array materialization")
        return count
    expected_rank = 1 + len(shape_tail)
    if len(shape) != expected_rank or tuple(shape[1:]) != shape_tail:
        expected = (
            "(n,)" if not shape_tail else f"(n, {', '.join(map(str, shape_tail))})"
        )
        raise ValueError(f"{name} must have shape {expected}")
    count = int(shape[0])
    if count < 0:
        raise ValueError(f"{name} leading dimension must be nonnegative")
    if count > cap:
        raise ValueError(f"{name} cap exceeded before array materialization")
    return count


def _frozen_float64(
    name: str,
    value: ArrayLike,
    shape_tail: tuple[int, ...],
) -> FloatArray:
    original = np.asarray(value)
    if original.dtype.kind not in "iuf":
        raise ValueError(f"{name} must use a real numeric dtype")
    array = np.asarray(original, dtype=np.float64)
    if array.ndim != 1 + len(shape_tail) or array.shape[1:] != shape_tail:
        expected = (
            "(n,)" if not shape_tail else f"(n, {', '.join(map(str, shape_tail))})"
        )
        raise ValueError(f"{name} must have shape {expected}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    contiguous = np.ascontiguousarray(array, dtype=np.float64)
    frozen = np.frombuffer(contiguous.tobytes(order="C"), dtype=np.float64)
    return frozen.reshape(contiguous.shape)


def _frozen_state(state: object) -> ParticleState:
    if type(state) is not ParticleState:
        raise ValueError("state must be an exact ParticleState")
    position_count = _preflight_count(
        "state.positions", state.positions, (3,), cap=MAX_PARTICLE_COUNT
    )
    gamma_count = _preflight_count(
        "state.gamma", state.gamma, (3,), cap=MAX_PARTICLE_COUNT
    )
    sigma_count = _preflight_count(
        "state.sigma", state.sigma, (), cap=MAX_PARTICLE_COUNT
    )
    if position_count != gamma_count or position_count != sigma_count:
        raise ValueError("state arrays must have the same leading dimension")
    positions = _frozen_float64("state.positions", state.positions, (3,))
    gamma = _frozen_float64("state.gamma", state.gamma, (3,))
    sigma = _frozen_float64("state.sigma", state.sigma, ())
    if np.any(sigma <= 0.0):
        raise ValueError("state.sigma must be strictly positive")
    return ParticleState(positions, gamma, sigma)


def _array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(contiguous.dtype.str.encode("ascii"))
    digest.update(json.dumps(contiguous.shape, separators=(",", ":")).encode("ascii"))
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def _stream_state_sha256(state: ParticleState) -> str:
    return hashlib.sha256(
        (
            "fluxv-ir-wrk3-stream-state-v1\0"
            + _array_sha256(state.positions)
            + _array_sha256(state.gamma)
            + _array_sha256(state.sigma)
        ).encode("ascii")
    ).hexdigest()


def _validate_sha256(name: str, value: object) -> str:
    if type(value) is not str or len(value) != 64:
        raise ValueError(f"{name} must be a SHA-256 digest")
    if any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} must be lowercase hexadecimal")
    return value


def _require_bytes_backed_float64(
    name: str,
    value: object,
    shape: tuple[int, ...],
) -> FloatArray:
    if type(value) is not np.ndarray:
        raise ValueError(f"{name} must be an exact ndarray")
    if value.dtype != np.dtype(np.float64) or value.shape != shape:
        raise ValueError(f"{name} has an invalid dtype or shape")
    if value.flags.writeable or not value.flags.c_contiguous:
        raise ValueError(f"{name} must be readonly and C-contiguous")
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must contain only finite values")
    ancestor: object = value
    while type(ancestor) is np.ndarray:
        if ancestor.flags.writeable or not ancestor.flags.c_contiguous:
            raise ValueError(f"{name} has a mutable or non-contiguous base")
        ancestor = ancestor.base
    if type(ancestor) is not bytes:
        raise ValueError(f"{name} must have an immutable bytes backing")
    return value


@dataclass(frozen=True, slots=True)
class _FunctionGuard:
    function: object
    code: object
    defaults: object
    kwdefaults: object
    globals: tuple[tuple[dict[str, object], str, object], ...]
    closures: tuple[tuple[object, object], ...]


@dataclass(frozen=True, slots=True)
class _CallableGuard:
    role: str
    root: object
    kind: str
    identity: tuple[object, ...]
    functions: tuple[_FunctionGuard, ...]


def _callable_parts(
    value: object,
) -> tuple[str, tuple[object, ...], tuple[object, ...]]:
    if inspect.isfunction(value):
        return "function", (value,), (value,)
    if inspect.ismethod(value) and inspect.isfunction(value.__func__):
        return (
            "bound_method",
            (value, value.__self__, value.__func__),
            (value.__func__,),
        )
    if isinstance(value, partial):
        kind, identity, functions = _callable_parts(value.func)
        return (
            f"partial[{kind}]",
            (value, value.func, value.args, value.keywords, *identity),
            functions,
        )
    if callable(value):
        call = getattr(type(value), "__call__", None)
        if inspect.isfunction(call):
            return "callable_object", (value, type(value), call), (call,)
    raise ValueError("callback must have auditable Python function code")


def _freeze_callable_guard(role: str, callback: object) -> _CallableGuard:
    kind, identity, functions = _callable_parts(callback)
    guards: list[_FunctionGuard] = []
    for function in functions:
        global_bindings: list[tuple[dict[str, object], str, object]] = []
        for name in sorted(set(function.__code__.co_names)):
            dependency = function.__globals__.get(name)
            if callable(dependency):
                global_bindings.append((function.__globals__, name, dependency))
        closure_bindings: list[tuple[object, object]] = []
        if function.__closure__ is not None:
            for cell in function.__closure__:
                try:
                    dependency = cell.cell_contents
                except ValueError:
                    continue
                if callable(dependency):
                    closure_bindings.append((cell, dependency))
        guards.append(
            _FunctionGuard(
                function=function,
                code=function.__code__,
                defaults=function.__defaults__,
                kwdefaults=function.__kwdefaults__,
                globals=tuple(global_bindings),
                closures=tuple(closure_bindings),
            )
        )
    return _CallableGuard(
        role=role,
        root=callback,
        kind=kind,
        identity=identity,
        functions=tuple(guards),
    )


def _assert_callable_guard(guard: _CallableGuard) -> None:
    kind, identity, _ = _callable_parts(guard.root)
    if (
        kind != guard.kind
        or len(identity) != len(guard.identity)
        or any(
            current is not frozen
            for current, frozen in zip(identity, guard.identity, strict=True)
        )
    ):
        raise RuntimeError(f"{guard.role} callable identity drift")
    for function_guard in guard.functions:
        function = function_guard.function
        if (
            function.__code__ is not function_guard.code
            or function.__defaults__ is not function_guard.defaults
            or function.__kwdefaults__ is not function_guard.kwdefaults
        ):
            raise RuntimeError(f"{guard.role} callable code/default drift")
        for namespace, name, frozen in function_guard.globals:
            if namespace.get(name) is not frozen:
                raise RuntimeError(f"{guard.role} callable global drift: {name}")
        for cell, frozen in function_guard.closures:
            try:
                current = cell.cell_contents
            except ValueError as error:
                raise RuntimeError(f"{guard.role} callable closure drift") from error
            if current is not frozen:
                raise RuntimeError(f"{guard.role} callable closure drift")


@dataclass(frozen=True, slots=True)
class IRWRK3StreamRHSView:
    stretching: FloatArray
    chi: FloatArray
    gamma_rate: FloatArray
    sigma_rate_diagnostic: FloatArray
    chain_rule_relative_residual: FloatArray


@dataclass(frozen=True, slots=True)
class IRWRK3StreamStageView:
    substep: int
    stage: int
    a: float
    b: float
    substep_delta_time: float
    pre: ParticleState
    field: IRWRK3Field
    rhs: IRWRK3StreamRHSView
    tracer_pre: FloatArray
    tracer_field: IRWRK3TracerField
    position_storage_pre: FloatArray
    gamma_storage_pre: FloatArray
    tracer_storage_pre: FloatArray
    invariant_reference_sha256: str
    parent_token: str


@dataclass(frozen=True, slots=True)
class IRWRK3StreamEvidence:
    schema: str
    payload: bytes
    payload_sha256: str
    evidence_sha256: str


@dataclass(frozen=True, slots=True)
class IRWRK3StreamStageRecord:
    substep: int
    stage: int
    a: float
    b: float
    substep_delta_time: float
    source_state_sha256: str
    pre_state_sha256: str
    post_state_sha256: str
    tracer_pre_sha256: str
    tracer_post_sha256: str
    velocity_sha256: str
    jacobian_sha256: str
    gamma_rate_sha256: str
    tracer_velocity_sha256: str
    invariant_residual_sha256: str
    invariant_residual_max: float
    invariant_residual_over_slog_max: float
    position_storage_pre_sha256: str
    gamma_storage_pre_sha256: str
    tracer_storage_pre_sha256: str
    position_storage_post_sha256: str
    gamma_storage_post_sha256: str
    tracer_storage_post_sha256: str
    parent_token: str
    evidence: IRWRK3StreamEvidence
    record_sha256: str
    previous_chain_sha256: str
    chain_sha256: str


# The STOP-bundle vocabulary calls the retained row a compact stage record.
# Keep the established public name as the canonical class so success-path
# record types and hashes remain unchanged.
IRWRK3CompactStageRecord = IRWRK3StreamStageRecord


@dataclass(frozen=True, slots=True)
class IRWRK3StreamCounters:
    invariant_reference_freeze_count: int
    substep_count: int
    stage_count: int
    physical_field_call_count: int
    tracer_field_call_count: int
    observer_call_count: int
    stage_pre_reconstruction_count: int
    stage_post_reconstruction_count: int
    physical_rhs_call_count: int
    storage_reset_count: int
    tracer_storage_reset_count: int
    sigma_storage_update_count: int
    relaxation_call_count: int
    compact_stage_record_count: int
    retained_stage_array_count: int
    evidence_byte_count: int


@dataclass(frozen=True, slots=True)
class IRWRK3StreamResult:
    final_state: ParticleState
    final_tracer_positions: FloatArray
    initial_state_sha256: str
    final_state_sha256: str
    initial_tracer_sha256: str
    final_tracer_sha256: str
    invariant_reference_sha256: str
    stages: tuple[IRWRK3StreamStageRecord, ...]
    counters: IRWRK3StreamCounters
    delta_time: float
    transport_substeps: int
    parent_token: str
    stage_chain_sha256: str
    result_sha256: str


class IRWRK3StreamStopped(RuntimeError, ValueError):
    """Fail-closed observer stop with an auditable completed-stage prefix."""

    def __init__(
        self,
        failure_phase: str,
        stage_began: bool,
        failed_coordinate: tuple[int, int],
        completed_stages: tuple[IRWRK3StreamStageRecord, ...],
        completed_stage_count: int,
        completed_stage_chain_sha256: str,
        cause: BaseException,
    ) -> None:
        if type(failure_phase) is not str or not failure_phase:
            raise ValueError("stream failure phase must be a non-empty exact string")
        if type(stage_began) is not bool:
            raise ValueError("stream stage_began must be an exact boolean")
        if (
            type(failed_coordinate) is not tuple
            or len(failed_coordinate) != 2
            or any(type(value) is not int for value in failed_coordinate)
        ):
            raise ValueError("stream failed coordinate must be an exact integer pair")
        substep, stage = failed_coordinate
        if substep < 1 or stage not in (1, 2, 3):
            raise ValueError("stream failed coordinate is outside the stage domain")
        if type(completed_stages) is not tuple or any(
            type(record) is not IRWRK3StreamStageRecord for record in completed_stages
        ):
            raise ValueError(
                "stream completed stages must be an exact compact-record tuple"
            )
        if type(completed_stage_count) is not int or completed_stage_count != len(
            completed_stages
        ):
            raise ValueError("stream completed stage count/prefix mismatch")
        expected_chain = _STAGE_CHAIN_GENESIS
        for index, record in enumerate(completed_stages):
            if record.substep != index // 3 + 1 or record.stage != index % 3 + 1:
                raise ValueError("stream completed stage prefix ordering mismatch")
            if record.previous_chain_sha256 != expected_chain:
                raise ValueError("stream completed stage prefix chain mismatch")
            expected_chain = record.chain_sha256
        if completed_stage_chain_sha256 != expected_chain:
            raise ValueError("stream completed stage terminal chain mismatch")
        if (
            type(completed_stage_chain_sha256) is not str
            or len(completed_stage_chain_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in completed_stage_chain_sha256
            )
        ):
            raise ValueError("completed stage chain must be a SHA-256 digest")
        if not isinstance(cause, BaseException):
            raise ValueError("stream failure cause must be an exception")
        super().__init__(
            "stream stage failed during "
            f"{failure_phase} at substep {substep}, stage {stage}: "
            f"{type(cause).__name__}: {cause}"
        )
        self.failure_phase = failure_phase
        self.stage_began = stage_began
        self.failed_coordinate = failed_coordinate
        self.substep = substep
        self.stage = stage
        self.completed_stages = completed_stages
        self.completed_stage_count = completed_stage_count
        self.completed_stage_chain_sha256 = completed_stage_chain_sha256
        self.completed_prefix_sha256 = completed_stage_chain_sha256
        self.original_cause = cause


IRWRK3StreamFieldEvaluator = Callable[[ParticleState], IRWRK3Field]
IRWRK3StreamTracerEvaluator = Callable[
    [ParticleState, FloatArray, str], IRWRK3TracerField
]
IRWRK3StreamObserver = Callable[[IRWRK3StreamStageView], IRWRK3StreamEvidence]


def make_ir_wrk3_stream_evidence(
    schema: str,
    payload: bytes = b"",
) -> IRWRK3StreamEvidence:
    """Create one bounded, immutable observer-evidence payload."""

    if type(schema) is not str or not schema or len(schema.encode("utf-8")) > 128:
        raise ValueError("evidence schema must be a non-empty <=128-byte string")
    if type(payload) is not bytes:
        raise ValueError("evidence payload must be exact bytes")
    if len(payload) > MAX_EVIDENCE_BYTES_PER_STAGE:
        raise ValueError("per-stage evidence byte cap exceeded")
    payload_sha = hashlib.sha256(payload).hexdigest()
    evidence_sha = hashlib.sha256(
        schema.encode("utf-8") + b"\0" + payload_sha.encode("ascii")
    ).hexdigest()
    return IRWRK3StreamEvidence(
        schema=schema,
        payload=payload,
        payload_sha256=payload_sha,
        evidence_sha256=evidence_sha,
    )


_EMPTY_EVIDENCE: Final[IRWRK3StreamEvidence] = make_ir_wrk3_stream_evidence("none")


class _DuplicateStreamSemanticIssuance(ValueError):
    """Signal that a valid semantic result already has one canonical issuance."""

    def __init__(self, canonical_result: IRWRK3StreamResult) -> None:
        super().__init__("duplicate stream result semantic issuance")
        self.canonical_result = canonical_result


def _make_live_stream_result_registry(
    max_count: int = MAX_LIVE_STREAM_RESULT_COUNT,
    hash_constructor: Callable[[bytes], object] = hashlib.sha256,
    frame_getter: Callable[[], object] = inspect.currentframe,
    function_predicate: Callable[[object], bool] = inspect.isfunction,
    duplicate_error_type: type[
        _DuplicateStreamSemanticIssuance
    ] = _DuplicateStreamSemanticIssuance,
) -> tuple[Callable[..., object], ...]:
    """Build a sealed issuance registry without exposing its mutable mapping."""

    lock = threading.RLock()
    registry: dict[int, tuple[IRWRK3StreamResult, str]] = {}
    semantics: dict[tuple[str, str, str, str], IRWRK3StreamResult] = {}
    issuance_counter = 0

    def semantic_key(result: IRWRK3StreamResult) -> tuple[str, str, str, str]:
        return (
            result.result_sha256,
            result.stage_chain_sha256,
            result.final_state_sha256,
            result.final_tracer_sha256,
        )

    def compute_seal() -> str:
        payload = (
            issuance_counter,
            tuple(
                (key, id(entry[0]), entry[1], entry[0].result_sha256)
                for key, entry in sorted(registry.items())
            ),
            tuple(
                (key, id(result), result.result_sha256)
                for key, result in sorted(semantics.items())
            ),
        )
        digest = hash_constructor(repr(payload).encode("ascii"))
        return digest.hexdigest()  # type: ignore[no-any-return, union-attr]

    seal = compute_seal()

    def verify_seal() -> None:
        if compute_seal() != seal:
            raise RuntimeError("IR-WRK3 stream live-result registry integrity drift")

    def register(result: IRWRK3StreamResult) -> None:
        nonlocal issuance_counter, seal
        frame = frame_getter()
        caller = None if frame is None else frame.f_back  # type: ignore[union-attr]
        try:
            issuer = globals().get("ir_wrk3_stream_macro")
            if (
                caller is None
                or not function_predicate(issuer)
                or caller.f_code is not issuer.__code__
                or caller.f_locals.get("register_result") is not register
            ):
                raise ValueError(
                    "stream result registration requires the trusted macro call site"
                )
            validator = caller.f_locals.get("validate_tree")
            if (
                not function_predicate(validator)
                or globals().get("_validate_result_tree") is not validator
                or validator(result) is not result
            ):
                raise ValueError(
                    "stream result registration requires a fully valid tree"
                )
        finally:
            del caller
            del frame
        with lock:
            verify_seal()
            result_semantic_key = semantic_key(result)
            canonical = semantics.get(result_semantic_key)
            if canonical is not None:
                raise duplicate_error_type(canonical)
            if len(registry) >= max_count:
                raise RuntimeError("IR-WRK3 stream live-result registry cap exceeded")
            key = id(result)
            if key in registry:
                raise RuntimeError("IR-WRK3 stream result identity collision")
            registry[key] = (result, result.result_sha256)
            semantics[result_semantic_key] = result
            issuance_counter += 1
            seal = compute_seal()

    def assert_capacity() -> None:
        with lock:
            verify_seal()
            if len(registry) >= max_count:
                raise RuntimeError("IR-WRK3 stream live-result registry cap exceeded")

    def attest(result: IRWRK3StreamResult) -> None:
        with lock:
            verify_seal()
            entry = registry.get(id(result))
            if (
                entry is None
                or entry[0] is not result
                or entry[1] != result.result_sha256
                or semantics.get(semantic_key(result)) is not result
            ):
                raise ValueError("stream result is not the exact live issued report")

    def snapshot() -> tuple[object, ...]:
        with lock:
            verify_seal()
            return (
                issuance_counter,
                seal,
                tuple(
                    (key, entry[0], entry[1]) for key, entry in sorted(registry.items())
                ),
            )

    def assert_snapshot_unchanged(expected: tuple[object, ...]) -> None:
        with lock:
            verify_seal()
            observed = snapshot()
            if (
                type(expected) is not tuple
                or len(expected) != 3
                or observed[0] != expected[0]
                or observed[1] != expected[1]
            ):
                raise RuntimeError(
                    "IR-WRK3 stream live-result registry changed during callback"
                )
            observed_entries = observed[2]
            expected_entries = expected[2]
            if (
                type(observed_entries) is not tuple
                or type(expected_entries) is not tuple
                or len(observed_entries) != len(expected_entries)
            ):
                raise RuntimeError(
                    "IR-WRK3 stream live-result registry changed during callback"
                )
            for observed_entry, expected_entry in zip(
                observed_entries, expected_entries, strict=True
            ):
                if (
                    observed_entry[0] != expected_entry[0]
                    or observed_entry[1] is not expected_entry[1]
                    or observed_entry[2] != expected_entry[2]
                ):
                    raise RuntimeError(
                        "IR-WRK3 stream live-result registry changed during callback"
                    )

    functions = (
        register,
        assert_capacity,
        attest,
        snapshot,
        assert_snapshot_unchanged,
    )
    return functions


(
    _register_result,
    _assert_registry_capacity,
    _attest_live_result,
    _snapshot_live_stream_result_registry,
    _assert_live_stream_result_registry_unchanged,
) = _make_live_stream_result_registry()


@dataclass(frozen=True, slots=True)
class _RegistryClosureBinding:
    freevar: str
    cell: object
    role: str
    value: object
    code: object | None
    defaults: object | None
    kwdefaults: object | None


@dataclass(frozen=True, slots=True)
class _RegistryFunctionBinding:
    name: str
    function: object
    code: object
    defaults: object
    kwdefaults: object
    closure: tuple[_RegistryClosureBinding, ...]


_REGISTRY_FUNCTION_NAMES: Final[tuple[str, ...]] = (
    "_register_result",
    "_assert_registry_capacity",
    "_attest_live_result",
    "_snapshot_live_stream_result_registry",
    "_assert_live_stream_result_registry_unchanged",
)


def _freeze_registry_function(name: str) -> _RegistryFunctionBinding:
    function = globals()[name]
    if not inspect.isfunction(function):
        raise RuntimeError(f"stream registry binding is not a function: {name}")
    closure_bindings: list[_RegistryClosureBinding] = []
    cells = function.__closure__ or ()
    for freevar, cell in zip(function.__code__.co_freevars, cells, strict=True):
        value = cell.cell_contents
        if freevar in ("registry", "semantics"):
            role = "registry"
        elif freevar == "lock":
            role = "lock"
        elif freevar == "issuance_counter":
            role = "counter"
        elif freevar == "seal":
            role = "seal"
        elif inspect.isfunction(value):
            role = "function"
        else:
            role = "fixed"
        closure_bindings.append(
            _RegistryClosureBinding(
                freevar=freevar,
                cell=cell,
                role=role,
                value=value,
                code=value.__code__ if inspect.isfunction(value) else None,
                defaults=value.__defaults__ if inspect.isfunction(value) else None,
                kwdefaults=value.__kwdefaults__ if inspect.isfunction(value) else None,
            )
        )
    return _RegistryFunctionBinding(
        name=name,
        function=function,
        code=function.__code__,
        defaults=function.__defaults__,
        kwdefaults=function.__kwdefaults__,
        closure=tuple(closure_bindings),
    )


_FROZEN_REGISTRY_FUNCTION_BINDINGS: Final[tuple[_RegistryFunctionBinding, ...]] = tuple(
    _freeze_registry_function(name) for name in _REGISTRY_FUNCTION_NAMES
)


def _assert_registry_function_bindings(
    frozen: tuple[_RegistryFunctionBinding, ...] = _FROZEN_REGISTRY_FUNCTION_BINDINGS,
) -> None:
    if globals().get("_FROZEN_REGISTRY_FUNCTION_BINDINGS") is not frozen:
        raise RuntimeError("stream registry verifier binding drift")
    for binding in frozen:
        current = globals().get(binding.name)
        if (
            current is not binding.function
            or current.__code__ is not binding.code
            or current.__defaults__ is not binding.defaults
            or current.__kwdefaults__ is not binding.kwdefaults
        ):
            raise RuntimeError(f"stream registry function drift: {binding.name}")
        cells = current.__closure__ or ()
        if len(cells) != len(binding.closure):
            raise RuntimeError(f"stream registry closure drift: {binding.name}")
        for cell, expected in zip(cells, binding.closure, strict=True):
            if cell is not expected.cell:
                raise RuntimeError(
                    f"stream registry closure cell drift: {binding.name}"
                )
            value = cell.cell_contents
            if expected.role in ("registry", "lock", "fixed"):
                if value is not expected.value:
                    raise RuntimeError(
                        f"stream registry {expected.role} closure drift: "
                        f"{binding.name}.{expected.freevar}"
                    )
            elif expected.role == "counter":
                if type(value) is not int or value < 0:
                    raise RuntimeError("stream registry counter closure drift")
            elif expected.role == "seal":
                _validate_sha256("stream registry seal", value)
            elif expected.role == "function":
                if (
                    value is not expected.value
                    or value.__code__ is not expected.code
                    or value.__defaults__ is not expected.defaults
                    or value.__kwdefaults__ is not expected.kwdefaults
                ):
                    raise RuntimeError(
                        f"stream registry function closure drift: "
                        f"{binding.name}.{expected.freevar}"
                    )
            else:
                raise RuntimeError("unknown stream registry closure role")


def _validate_evidence(value: object) -> IRWRK3StreamEvidence:
    if type(value) is not IRWRK3StreamEvidence:
        raise ValueError("observer must return an exact IRWRK3StreamEvidence")
    expected = make_ir_wrk3_stream_evidence(value.schema, value.payload)
    if (
        value.payload_sha256 != expected.payload_sha256
        or value.evidence_sha256 != expected.evidence_sha256
    ):
        raise ValueError("observer evidence digest mismatch")
    return value


def _stable_row_norms(gamma: FloatArray) -> tuple[FloatArray, FloatArray]:
    max_abs = np.max(np.abs(gamma), axis=1)
    norms = np.zeros(gamma.shape[0], dtype=np.float64)
    active = max_abs != 0.0
    if np.any(active):
        scaled = gamma[active] / max_abs[active, None]
        norms[active] = max_abs[active] * np.sqrt(np.einsum("ni,ni->n", scaled, scaled))
    if not np.all(np.isfinite(norms)):
        raise FloatingPointError("stream scaled gamma norm is non-finite")
    return norms, max_abs


def _stream_rhs(
    gamma: FloatArray,
    sigma: FloatArray,
    field: IRWRK3Field,
) -> IRWRK3StreamRHSView:
    """Repeat the pinned public IR-WRK3 formula, not any core private helper.

    The operation order intentionally matches the frozen reference definition:
    ``S=J.T@Gamma``, ``chi=(S.Gamma)/||Gamma||^2``,
    ``Gamma_dot=S-3/5 chi Gamma`` and diagnostic
    ``sigma_dot=-1/5 chi sigma``.  This local repetition is the explicit
    stream/core differential contract.
    """

    stretching = np.einsum("nji,nj->ni", field.jacobian, gamma)
    norms, _ = _stable_row_norms(gamma)
    active = norms != 0.0
    chi = np.zeros(gamma.shape[0], dtype=np.float64)
    if np.any(active):
        direction = gamma[active] / norms[active, None]
        chi[active] = (
            np.einsum("ni,ni->n", stretching[active], direction) / norms[active]
        )
    gamma_rate = stretching - 0.6 * chi[:, None] * gamma
    sigma_rate = -0.2 * chi * sigma
    chain = np.zeros(gamma.shape[0], dtype=np.float64)
    if np.any(active):
        gamma_projection = np.einsum("ni,ni->n", gamma[active], gamma_rate[active]) / (
            norms[active] * norms[active]
        )
        lhs = sigma_rate[active] / sigma[active]
        numerator = np.abs(lhs + 0.5 * gamma_projection)
        denominator = np.maximum.reduce(
            (np.ones_like(lhs), np.abs(lhs), 0.5 * np.abs(gamma_projection))
        )
        chain[active] = numerator / denominator
    arrays = (stretching, chi, gamma_rate, sigma_rate, chain)
    if any(not np.all(np.isfinite(item)) for item in arrays):
        raise FloatingPointError("stream IR-WRK3 RHS produced non-finite values")
    return IRWRK3StreamRHSView(
        stretching=_frozen_float64("rhs.stretching", stretching, (3,)),
        chi=_frozen_float64("rhs.chi", chi, ()),
        gamma_rate=_frozen_float64("rhs.gamma_rate", gamma_rate, (3,)),
        sigma_rate_diagnostic=_frozen_float64(
            "rhs.sigma_rate_diagnostic", sigma_rate, ()
        ),
        chain_rule_relative_residual=_frozen_float64(
            "rhs.chain_rule_relative_residual", chain, ()
        ),
    )


def _invariant_log_residual(
    gamma: FloatArray,
    sigma: FloatArray,
    reference: InvariantReference,
) -> FloatArray:
    norms, _ = _stable_row_norms(gamma)
    residual = np.zeros(gamma.shape[0], dtype=np.float64)
    active = ~reference.exact_zero_mask
    if np.any(active):
        residual[active] = np.abs(
            np.log(norms[active])
            - reference.log_gamma_norm_star[active]
            + 2.0 * (np.log(sigma[active]) - np.log(reference.sigma_star[active]))
        )
    return _frozen_float64("invariant_log_residual", residual, ())


def _enforce_invariant_gate(
    post: ParticleState,
    reference: InvariantReference,
    residual: FloatArray,
) -> float:
    active = ~reference.exact_zero_mask
    if not np.any(active):
        return 0.0
    norms, _ = _stable_row_norms(post.gamma)
    scale = np.maximum.reduce(
        (
            np.ones(np.count_nonzero(active)),
            np.abs(np.log(norms[active]) - reference.log_gamma_norm_star[active]),
            2.0
            * np.abs(np.log(post.sigma[active]) - np.log(reference.sigma_star[active])),
        )
    )
    if np.any(residual[active] > INVARIANT_LOG_ATOL * scale):
        raise FloatingPointError("stream IR-WRK3 invariant residual exceeds gate")
    normalized = residual[active] / scale
    if not np.all(np.isfinite(normalized)) or np.any(normalized < 0.0):
        raise FloatingPointError(
            "stream IR-WRK3 normalized invariant residual is invalid"
        )
    return float(np.max(normalized))


def _stage_record_payload(record: IRWRK3StreamStageRecord) -> dict[str, object]:
    return {
        "a": float(record.a).hex(),
        "b": float(record.b).hex(),
        "domain": "fluxv-ir-wrk3-stream-stage-v1",
        "evidence_sha256": record.evidence.evidence_sha256,
        "gamma_rate_sha256": record.gamma_rate_sha256,
        "invariant_residual_max": float(record.invariant_residual_max).hex(),
        "invariant_residual_over_slog_max": float(
            record.invariant_residual_over_slog_max
        ).hex(),
        "invariant_residual_sha256": record.invariant_residual_sha256,
        "jacobian_sha256": record.jacobian_sha256,
        "parent_token": record.parent_token,
        "post_state_sha256": record.post_state_sha256,
        "pre_state_sha256": record.pre_state_sha256,
        "previous_chain_sha256": record.previous_chain_sha256,
        "source_state_sha256": record.source_state_sha256,
        "stage": record.stage,
        "substep_delta_time": float(record.substep_delta_time).hex(),
        "storage": {
            "gamma_post": record.gamma_storage_post_sha256,
            "gamma_pre": record.gamma_storage_pre_sha256,
            "position_post": record.position_storage_post_sha256,
            "position_pre": record.position_storage_pre_sha256,
            "tracer_post": record.tracer_storage_post_sha256,
            "tracer_pre": record.tracer_storage_pre_sha256,
        },
        "substep": record.substep,
        "tracer_post_sha256": record.tracer_post_sha256,
        "tracer_pre_sha256": record.tracer_pre_sha256,
        "tracer_velocity_sha256": record.tracer_velocity_sha256,
        "velocity_sha256": record.velocity_sha256,
    }


def _stage_record_sha256(record: IRWRK3StreamStageRecord) -> str:
    return hashlib.sha256(
        json.dumps(
            _stage_record_payload(record),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _counter_values(counters: IRWRK3StreamCounters) -> tuple[int, ...]:
    return tuple(getattr(counters, name) for name in counters.__dataclass_fields__)


def _result_payload(result: IRWRK3StreamResult) -> dict[str, object]:
    return {
        "counters": _counter_values(result.counters),
        "delta_time": float(result.delta_time).hex(),
        "domain": "fluxv-ir-wrk3-stream-result-v1",
        "final_state_sha256": result.final_state_sha256,
        "final_tracer_sha256": result.final_tracer_sha256,
        "initial_state_sha256": result.initial_state_sha256,
        "initial_tracer_sha256": result.initial_tracer_sha256,
        "invariant_reference_sha256": result.invariant_reference_sha256,
        "parent_token": result.parent_token,
        "stage_chain_sha256": result.stage_chain_sha256,
        "stage_record_sha256": [record.record_sha256 for record in result.stages],
        "transport_substeps": result.transport_substeps,
    }


def _result_sha256(result: IRWRK3StreamResult) -> str:
    return hashlib.sha256(
        json.dumps(
            _result_payload(result),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _validate_state_tree(name: str, state: object) -> int:
    if type(state) is not ParticleState:
        raise ValueError(f"{name} must be an exact ParticleState")
    if type(state.sigma) is not np.ndarray or state.sigma.ndim != 1:
        raise ValueError(f"{name}.sigma must be an exact one-dimensional ndarray")
    count = state.sigma.shape[0]
    if type(count) is not int or count < 0 or count > MAX_PARTICLE_COUNT:
        raise ValueError(f"{name} particle count is outside the frozen cap")
    _require_bytes_backed_float64(f"{name}.positions", state.positions, (count, 3))
    _require_bytes_backed_float64(f"{name}.gamma", state.gamma, (count, 3))
    sigma = _require_bytes_backed_float64(f"{name}.sigma", state.sigma, (count,))
    if np.any(sigma <= 0.0):
        raise ValueError(f"{name}.sigma must be strictly positive")
    return count


def _validate_stage_tree(
    record: object,
    *,
    expected_substep: int,
    expected_stage: int,
    expected_parent_token: str,
    expected_previous_chain: str,
) -> tuple[str, int]:
    if type(record) is not IRWRK3StreamStageRecord:
        raise ValueError("stream stages must contain exact compact stage records")
    if type(record.substep) is not int or record.substep != expected_substep:
        raise ValueError("compact stage substep ordering mismatch")
    if type(record.stage) is not int or record.stage != expected_stage:
        raise ValueError("compact stage ordering mismatch")
    expected_a = _RK_A[expected_stage - 1]
    expected_b = _RK_B[expected_stage - 1]
    if type(record.a) is not float or record.a != expected_a:
        raise ValueError("compact stage Williamson a coefficient mismatch")
    if type(record.b) is not float or record.b != expected_b:
        raise ValueError("compact stage Williamson b coefficient mismatch")
    if record.parent_token != expected_parent_token:
        raise ValueError("compact stage parent token mismatch")
    if type(record.parent_token) is not str or not record.parent_token:
        raise ValueError("compact stage parent token is invalid")
    if (
        type(record.invariant_residual_max) is not float
        or not math.isfinite(record.invariant_residual_max)
        or record.invariant_residual_max < 0.0
    ):
        raise ValueError("compact stage invariant residual maximum is invalid")
    if (
        type(record.invariant_residual_over_slog_max) is not float
        or not math.isfinite(record.invariant_residual_over_slog_max)
        or record.invariant_residual_over_slog_max < 0.0
    ):
        raise ValueError(
            "compact stage normalized invariant residual maximum is invalid"
        )
    if record.invariant_residual_over_slog_max > INVARIANT_LOG_ATOL:
        raise ValueError("compact stage normalized invariant residual exceeds gate")
    digest_names = (
        "source_state_sha256",
        "pre_state_sha256",
        "post_state_sha256",
        "tracer_pre_sha256",
        "tracer_post_sha256",
        "velocity_sha256",
        "jacobian_sha256",
        "gamma_rate_sha256",
        "tracer_velocity_sha256",
        "invariant_residual_sha256",
        "position_storage_pre_sha256",
        "gamma_storage_pre_sha256",
        "tracer_storage_pre_sha256",
        "position_storage_post_sha256",
        "gamma_storage_post_sha256",
        "tracer_storage_post_sha256",
        "record_sha256",
        "previous_chain_sha256",
        "chain_sha256",
    )
    for name in digest_names:
        _validate_sha256(f"stage.{name}", getattr(record, name))
    if record.previous_chain_sha256 != expected_previous_chain:
        raise ValueError("compact stage chain predecessor mismatch")
    evidence = _validate_evidence(record.evidence)
    if len(evidence.payload) > MAX_EVIDENCE_BYTES_PER_STAGE:
        raise ValueError("per-stage evidence byte cap exceeded")
    expected_record_sha = _stage_record_sha256(record)
    if record.record_sha256 != expected_record_sha:
        raise ValueError("compact stage record digest mismatch")
    expected_chain = hashlib.sha256(
        (
            "fluxv-ir-wrk3-stream-stage-link-v1\0"
            + expected_previous_chain
            + expected_record_sha
        ).encode("ascii")
    ).hexdigest()
    if record.chain_sha256 != expected_chain:
        raise ValueError("compact stage chain digest mismatch")
    return expected_chain, len(evidence.payload)


def _validate_result_tree(result: object) -> IRWRK3StreamResult:
    """Validate the complete immutable compact tree and every bound digest."""

    if type(result) is not IRWRK3StreamResult:
        raise ValueError("result must be an exact IRWRK3StreamResult")
    if type(result.delta_time) is not float or not math.isfinite(result.delta_time):
        raise ValueError("result delta_time must be an exact finite float")
    if result.delta_time <= 0.0:
        raise ValueError("result delta_time must be positive")
    if (
        type(result.transport_substeps) is not int
        or not 1 <= result.transport_substeps <= MAX_SUBSTEPS
    ):
        raise ValueError("result transport_substeps is outside the frozen cap")
    stage_count = 3 * result.transport_substeps
    if len(result.stages) != stage_count:
        raise ValueError("result stage count does not match its substep count")
    substep_dts = tuple(float(record.substep_delta_time) for record in result.stages)
    for value in substep_dts:
        if type(value) is not float or not math.isfinite(value) or value <= 0.0:
            raise ValueError("result per-substep delta_time is invalid")
    if math.fsum(substep_dts[::3]) != result.delta_time:
        raise ValueError(
            "result per-substep delta_times must sum exactly to delta_time"
        )
    if type(result.parent_token) is not str or not result.parent_token:
        raise ValueError("result parent token must be a non-empty exact string")

    particle_count = _validate_state_tree("result.final_state", result.final_state)
    if (
        type(result.final_tracer_positions) is not np.ndarray
        or result.final_tracer_positions.ndim != 2
    ):
        raise ValueError(
            "result.final_tracer_positions must be an exact two-dimensional ndarray"
        )
    tracer_count = result.final_tracer_positions.shape[0]
    if tracer_count > MAX_TRACER_COUNT:
        raise ValueError("result tracer count is outside the frozen cap")
    _require_bytes_backed_float64(
        "result.final_tracer_positions",
        result.final_tracer_positions,
        (tracer_count, 3),
    )
    del particle_count

    if type(result.stages) is not tuple:
        raise ValueError("result stages must be an exact immutable tuple")
    if type(result.counters) is not IRWRK3StreamCounters:
        raise ValueError("result counters must have the exact frozen type")
    for name in IRWRK3StreamCounters.__dataclass_fields__:
        value = getattr(result.counters, name)
        if type(value) is not int or value < 0:
            raise ValueError(f"counter {name} must be a nonnegative exact integer")
    for name in (
        "initial_state_sha256",
        "final_state_sha256",
        "initial_tracer_sha256",
        "final_tracer_sha256",
        "invariant_reference_sha256",
        "stage_chain_sha256",
        "result_sha256",
    ):
        _validate_sha256(f"result.{name}", getattr(result, name))

    stage_count = 3 * result.transport_substeps
    if len(result.stages) != stage_count:
        raise ValueError("result compact stage count mismatch")
    expected_counters = {
        "invariant_reference_freeze_count": 1,
        "substep_count": result.transport_substeps,
        "stage_count": stage_count,
        "physical_field_call_count": stage_count,
        "tracer_field_call_count": stage_count if tracer_count else 0,
        "stage_pre_reconstruction_count": stage_count,
        "stage_post_reconstruction_count": stage_count,
        "physical_rhs_call_count": stage_count,
        "storage_reset_count": result.transport_substeps,
        "tracer_storage_reset_count": result.transport_substeps if tracer_count else 0,
        "sigma_storage_update_count": 0,
        "relaxation_call_count": 0,
        "compact_stage_record_count": stage_count,
        "retained_stage_array_count": 0,
    }
    for name, expected in expected_counters.items():
        if getattr(result.counters, name) != expected:
            raise ValueError(f"counter {name} does not match the stream ledger")
    if result.counters.observer_call_count not in (0, stage_count):
        raise ValueError("observer counter does not match an all-stage callback ledger")

    previous_chain = _STAGE_CHAIN_GENESIS
    previous_post_state: str | None = None
    previous_tracer_post: str | None = None
    evidence_byte_count = 0
    for index, record in enumerate(result.stages):
        expected_substep = index // 3 + 1
        expected_stage = index % 3 + 1
        if record.substep_delta_time != substep_dts[3 * (expected_substep - 1)]:
            raise ValueError(
                "compact stage delta_time disagrees with its sub-step grid slot"
            )
        previous_chain, payload_bytes = _validate_stage_tree(
            record,
            expected_substep=expected_substep,
            expected_stage=expected_stage,
            expected_parent_token=result.parent_token,
            expected_previous_chain=previous_chain,
        )
        evidence_byte_count += payload_bytes
        if evidence_byte_count > MAX_TOTAL_EVIDENCE_BYTES:
            raise ValueError("total stream evidence byte cap exceeded")
        if index == 0:
            if record.pre_state_sha256 != result.initial_state_sha256:
                raise ValueError(
                    "initial state is not bound to the first compact stage"
                )
            if record.tracer_pre_sha256 != result.initial_tracer_sha256:
                raise ValueError(
                    "initial tracers are not bound to the first compact stage"
                )
        else:
            if record.pre_state_sha256 != previous_post_state:
                raise ValueError("compact particle state chain is discontinuous")
            if record.tracer_pre_sha256 != previous_tracer_post:
                raise ValueError("compact tracer state chain is discontinuous")
        previous_post_state = record.post_state_sha256
        previous_tracer_post = record.tracer_post_sha256

    if previous_chain != result.stage_chain_sha256:
        raise ValueError("result stage chain digest mismatch")
    if previous_post_state != result.final_state_sha256:
        raise ValueError("final state is not bound to the last compact stage")
    if previous_tracer_post != result.final_tracer_sha256:
        raise ValueError("final tracers are not bound to the last compact stage")
    if result.final_state_sha256 != _stream_state_sha256(result.final_state):
        raise ValueError("result final-state digest mismatch")
    if result.final_tracer_sha256 != _array_sha256(result.final_tracer_positions):
        raise ValueError("result final-tracer digest mismatch")
    if result.counters.evidence_byte_count != evidence_byte_count:
        raise ValueError("evidence byte counter does not match compact records")
    if result.counters.observer_call_count == 0 and any(
        record.evidence != _EMPTY_EVIDENCE for record in result.stages
    ):
        raise ValueError("zero observer calls cannot retain observer evidence")
    if result.result_sha256 != _result_sha256(result):
        raise ValueError("stream result digest mismatch")
    return result


def ir_wrk3_stream_macro(
    state: ParticleState,
    delta_time: float,
    field_evaluator: IRWRK3StreamFieldEvaluator,
    *,
    transport_substeps: int = 1,
    tracer_positions: ArrayLike | None = None,
    tracer_field_evaluator: IRWRK3StreamTracerEvaluator | None = None,
    parent_token: str = "analytic-frozen-parent",
    observer: IRWRK3StreamObserver | None = None,
    substep_delta_times: Sequence[float] | None = None,
) -> IRWRK3StreamResult:
    """Advance one macro step while retaining only compact stage evidence.

    When ``substep_delta_times`` is provided it is the exact per-sub-step
    graded time grid (V5H15 birth-window policy): its values are integrated
    verbatim, its length replaces ``transport_substeps`` as the sub-step
    count, and its ``math.fsum`` must equal ``delta_time`` bit-for-bit.
    Stage gates downstream must use each record's own
    ``substep_delta_time``; nothing may average the grid.
    """

    verify_runtime = _assert_runtime_bindings
    verify_registry_bindings = _assert_registry_function_bindings
    freeze_guard = _freeze_callable_guard
    verify_callable = _assert_callable_guard
    assert_capacity = _assert_registry_capacity
    snapshot_registry = _snapshot_live_stream_result_registry
    verify_registry_unchanged = _assert_live_stream_result_registry_unchanged
    validate_tree = _validate_result_tree
    register_result = _register_result
    attest_result = _attest_live_result
    duplicate_error_type = _DuplicateStreamSemanticIssuance
    trusted_functions = (
        ("_assert_runtime_bindings", verify_runtime),
        ("_assert_registry_function_bindings", verify_registry_bindings),
        ("_freeze_callable_guard", freeze_guard),
        ("_assert_callable_guard", verify_callable),
        ("_callable_parts", _callable_parts),
        ("_validate_sha256", _validate_sha256),
        ("_validate_result_tree", validate_tree),
        ("_assert_registry_capacity", assert_capacity),
        ("_snapshot_live_stream_result_registry", snapshot_registry),
        (
            "_assert_live_stream_result_registry_unchanged",
            verify_registry_unchanged,
        ),
        ("_register_result", register_result),
        ("_attest_live_result", attest_result),
    )
    trusted_bindings = tuple(
        (
            name,
            function,
            function.__code__,
            function.__defaults__,
            function.__kwdefaults__,
        )
        for name, function in trusted_functions
    )

    def verify_trusted_functions() -> None:
        if (
            globals().get("_DuplicateStreamSemanticIssuance")
            is not duplicate_error_type
        ):
            raise RuntimeError("stream duplicate semantic error type drift")
        for name, function, code, defaults, kwdefaults in trusted_bindings:
            if (
                globals().get(name) is not function
                or function.__code__ is not code
                or function.__defaults__ is not defaults
                or function.__kwdefaults__ is not kwdefaults
            ):
                raise RuntimeError(f"stream trusted function drift: {name}")

    verify_trusted_functions()
    verify_runtime()
    verify_registry_bindings()
    assert_capacity()
    registry_snapshot = snapshot_registry()
    if isinstance(delta_time, (bool, np.bool_)) or not isinstance(
        delta_time, (int, float, np.integer, np.floating)
    ):
        raise ValueError("delta_time must be a real scalar")
    if not math.isfinite(float(delta_time)) or float(delta_time) <= 0.0:
        raise ValueError("delta_time must be finite and positive")
    if (
        isinstance(transport_substeps, (bool, np.bool_))
        or type(transport_substeps) is not int
    ):
        raise ValueError("transport_substeps must be an exact integer")
    if not 1 <= transport_substeps <= MAX_SUBSTEPS:
        raise ValueError("transport_substeps is outside the frozen cap")
    if substep_delta_times is None:
        substep_dts: tuple[float, ...] = (
            float(delta_time) / transport_substeps,
        ) * transport_substeps
        substep_delta_time = substep_dts[0]
        if not math.isfinite(substep_delta_time) or substep_delta_time <= 0.0:
            raise ValueError("per-substep delta_time underflowed or is not finite")
    else:
        if isinstance(substep_delta_times, (str, bytes)) or not isinstance(
            substep_delta_times, Sequence
        ):
            raise ValueError("substep_delta_times must be a sequence of floats")
        substep_dts = tuple(substep_delta_times)
        if not 1 <= len(substep_dts) <= MAX_SUBSTEPS:
            raise ValueError("graded substep count is outside the frozen cap")
        for value in substep_dts:
            if type(value) is not float or not math.isfinite(value) or value <= 0.0:
                raise ValueError(
                    "every graded substep delta_time must be a positive finite float"
                )
        if not math.isclose(
            math.fsum(substep_dts), float(delta_time), rel_tol=1e-12, abs_tol=0.0
        ):
            raise ValueError("graded substep delta_times must sum to delta_time")
        if len(substep_dts) != transport_substeps:
            raise ValueError("graded substep count must equal transport_substeps")
    effective_substeps = len(substep_dts)
    integrated_total = math.fsum(substep_dts)
    if type(parent_token) is not str or not parent_token:
        raise ValueError("parent_token must be a non-empty exact string")

    field_guard = freeze_guard("field_evaluator", field_evaluator)
    tracer_guard: _CallableGuard | None = None
    observer_guard: _CallableGuard | None = None
    if tracer_positions is None:
        if tracer_field_evaluator is not None:
            raise ValueError("tracer evaluator requires tracer_positions")
    else:
        if tracer_field_evaluator is None:
            raise ValueError("tracer_field_evaluator is required for tracers")
        tracer_guard = freeze_guard("tracer_field_evaluator", tracer_field_evaluator)
    if observer is not None:
        observer_guard = freeze_guard("observer", observer)
    guards = tuple(
        guard
        for guard in (field_guard, tracer_guard, observer_guard)
        if guard is not None
    )

    def verify_all() -> None:
        verify_trusted_functions()
        verify_runtime()
        verify_registry_bindings()
        verify_registry_unchanged(registry_snapshot)
        for guard in guards:
            verify_callable(guard)

    records: list[IRWRK3StreamStageRecord] = []
    chain_sha = _STAGE_CHAIN_GENESIS
    evidence_byte_count = 0

    def stopped(
        failure_phase: str,
        stage_began: bool,
        failed_coordinate: tuple[int, int],
        cause: BaseException,
    ) -> IRWRK3StreamStopped:
        return IRWRK3StreamStopped(
            failure_phase=failure_phase,
            stage_began=stage_began,
            failed_coordinate=failed_coordinate,
            completed_stages=tuple(records),
            completed_stage_count=len(records),
            completed_stage_chain_sha256=chain_sha,
            cause=cause,
        )

    def execute_before_stage(
        failure_phase: str, operation: Callable[[], object]
    ) -> object:
        try:
            return operation()
        except Exception as error:
            cause: BaseException = error
            try:
                verify_all()
            except Exception as integrity_error:
                cause = integrity_error
            failure = stopped(failure_phase, False, (1, 1), cause)
            raise failure from cause

    def freeze_inputs() -> tuple[ParticleState, FloatArray, str, str]:
        verify_all()
        initial_state = _frozen_state(state)
        if tracer_positions is None:
            frozen_tracers = _frozen_float64("tracer_positions", np.empty((0, 3)), (3,))
        else:
            _preflight_count(
                "tracer_positions",
                tracer_positions,
                (3,),
                cap=MAX_TRACER_COUNT,
            )
            frozen_tracers = _frozen_float64("tracer_positions", tracer_positions, (3,))
        return (
            initial_state,
            frozen_tracers,
            _stream_state_sha256(initial_state),
            _array_sha256(frozen_tracers),
        )

    (
        initial,
        tracer_frozen,
        initial_state_sha,
        initial_tracer_sha,
    ) = execute_before_stage("input_freeze", freeze_inputs)
    tracer_count = tracer_frozen.shape[0]

    def freeze_reference() -> InvariantReference:
        # This is the sole invariant-freeze call for the entire macro step.
        frozen_reference = freeze_invariant_reference(initial)
        if type(frozen_reference) is not InvariantReference:
            raise RuntimeError("public invariant-reference type drift")
        _validate_sha256(
            "invariant_reference.reference_sha256",
            frozen_reference.reference_sha256,
        )
        verify_all()
        return frozen_reference

    reference = execute_before_stage("invariant_reference", freeze_reference)
    reference_sha = reference.reference_sha256

    def initialize_working_state() -> tuple[FloatArray, ...]:
        return (
            np.array(initial.positions, dtype=np.float64, order="C", copy=True),
            np.array(initial.gamma, dtype=np.float64, order="C", copy=True),
            np.array(initial.sigma, dtype=np.float64, order="C", copy=True),
            np.array(tracer_frozen, dtype=np.float64, order="C", copy=True),
        )

    positions, gamma, sigma, tracer = execute_before_stage(
        "state_initialization", initialize_working_state
    )

    for substep in range(1, effective_substeps + 1):
        substep_delta_time = substep_dts[substep - 1]
        try:
            position_storage = np.zeros_like(positions)
            gamma_storage = np.zeros_like(gamma)
            tracer_storage = np.zeros_like(tracer)
        except Exception as error:
            cause: BaseException = error
            try:
                verify_all()
            except Exception as integrity_error:
                cause = integrity_error
            failure = stopped("storage_reset", False, (substep, 1), cause)
            raise failure from cause

        for stage, (a_coefficient, b_coefficient) in enumerate(
            zip(_RK_A, _RK_B, strict=True),
            start=1,
        ):

            def execute_phase(
                failure_phase: str, operation: Callable[[], object]
            ) -> object:
                try:
                    return operation()
                except Exception as error:
                    cause: BaseException = error
                    try:
                        verify_all()
                    except Exception as integrity_error:
                        cause = integrity_error
                    failure = stopped(failure_phase, True, (substep, stage), cause)
                    raise failure from cause

            sigma = execute_phase(
                "pre_reconstruction",
                lambda: np.array(reconstruct_sigma(gamma, reference), copy=True),
            )
            execute_phase("pre_reconstruction_attestation", verify_all)

            def make_stage_pre() -> tuple[object, ...]:
                stage_pre = ParticleState(
                    _frozen_float64("stage_pre.positions", positions, (3,)),
                    _frozen_float64("stage_pre.gamma", gamma, (3,)),
                    _frozen_float64("stage_pre.sigma", sigma, ()),
                )
                return (
                    stage_pre,
                    _stream_state_sha256(stage_pre),
                    _frozen_float64("position_storage_pre", position_storage, (3,)),
                    _frozen_float64("gamma_storage_pre", gamma_storage, (3,)),
                    _frozen_float64("tracer_pre", tracer, (3,)),
                    _frozen_float64("tracer_storage_pre", tracer_storage, (3,)),
                )

            (
                pre,
                pre_state_sha,
                position_storage_pre,
                gamma_storage_pre,
                tracer_pre,
                tracer_storage_pre,
            ) = execute_phase("stage_pre", make_stage_pre)

            field_response = execute_phase(
                "physical_field", lambda: field_evaluator(pre)
            )

            def attest_physical_field() -> IRWRK3Field:
                verify_all()
                if type(field_response) is not IRWRK3Field:
                    raise ValueError("field_evaluator must return an exact IRWRK3Field")
                if field_response.parent_token != parent_token:
                    raise ValueError("field parent token mismatch")
                attested = make_ir_wrk3_field(
                    pre,
                    field_response.velocity,
                    field_response.jacobian,
                    parent_token=parent_token,
                )
                if field_response.source_state_sha256 != attested.source_state_sha256:
                    raise ValueError("field is not bound to the stage-pre state")
                verify_all()
                return attested

            field = execute_phase("physical_attestation", attest_physical_field)

            if tracer_count:
                tracer_response = execute_phase(
                    "tracer_field",
                    lambda: tracer_field_evaluator(pre, tracer_pre, parent_token),
                )

                def attest_tracer_field() -> IRWRK3TracerField:
                    verify_all()
                    if type(tracer_response) is not IRWRK3TracerField:
                        raise ValueError(
                            "tracer_field_evaluator must return an exact "
                            "IRWRK3TracerField"
                        )
                    if tracer_response.parent_token != parent_token:
                        raise ValueError("tracer field parent token mismatch")
                    attested = make_ir_wrk3_tracer_field(
                        pre,
                        tracer_pre,
                        tracer_response.velocity,
                        parent_token=parent_token,
                    )
                    if (
                        tracer_response.source_state_sha256
                        != attested.source_state_sha256
                    ):
                        raise ValueError("tracer field source-state digest mismatch")
                    if (
                        tracer_response.tracer_state_sha256
                        != attested.tracer_state_sha256
                    ):
                        raise ValueError("tracer field tracer-state digest mismatch")
                    verify_all()
                    return attested

                tracer_field = execute_phase("tracer_attestation", attest_tracer_field)
            else:

                def make_empty_tracer_field() -> IRWRK3TracerField:
                    attested = make_ir_wrk3_tracer_field(
                        pre,
                        tracer_pre,
                        np.empty((0, 3)),
                        parent_token=parent_token,
                    )
                    verify_all()
                    return attested

                tracer_field = execute_phase(
                    "tracer_attestation", make_empty_tracer_field
                )

            def compute_rhs() -> IRWRK3StreamRHSView:
                result_rhs = _stream_rhs(gamma, sigma, field)
                if np.any(result_rhs.chain_rule_relative_residual > INVARIANT_LOG_ATOL):
                    raise FloatingPointError(
                        "stream IR-WRK3 chain-rule residual exceeds gate"
                    )
                return result_rhs

            rhs = execute_phase("rhs", compute_rhs)

            def make_stage_view() -> IRWRK3StreamStageView:
                return IRWRK3StreamStageView(
                    substep=substep,
                    stage=stage,
                    a=float(a_coefficient),
                    b=float(b_coefficient),
                    substep_delta_time=substep_delta_time,
                    pre=pre,
                    field=field,
                    rhs=rhs,
                    tracer_pre=tracer_pre,
                    tracer_field=tracer_field,
                    position_storage_pre=position_storage_pre,
                    gamma_storage_pre=gamma_storage_pre,
                    tracer_storage_pre=tracer_storage_pre,
                    invariant_reference_sha256=reference_sha,
                    parent_token=parent_token,
                )

            view = execute_phase("observer_setup", make_stage_view)
            if observer is None:
                evidence = _EMPTY_EVIDENCE
            else:
                evidence = execute_phase("observer", lambda: observer(view))
                execute_phase("observer_attestation", verify_all)
                evidence = execute_phase(
                    "evidence_validation", lambda: _validate_evidence(evidence)
                )

            def check_evidence_cap() -> int:
                next_count = evidence_byte_count + len(evidence.payload)
                if next_count > MAX_TOTAL_EVIDENCE_BYTES:
                    raise ValueError("total stream evidence byte cap exceeded")
                return next_count

            next_evidence_bytes = execute_phase("evidence_cap", check_evidence_cap)

            def update_stage_state() -> tuple[FloatArray, ...]:
                position_storage_next = (
                    a_coefficient * position_storage
                    + substep_delta_time * field.velocity
                )
                gamma_storage_next = (
                    a_coefficient * gamma_storage + substep_delta_time * rhs.gamma_rate
                )
                tracer_storage_next = (
                    a_coefficient * tracer_storage
                    + substep_delta_time * tracer_field.velocity
                )
                positions_next = positions + b_coefficient * position_storage_next
                gamma_next = gamma + b_coefficient * gamma_storage_next
                tracer_next = tracer + b_coefficient * tracer_storage_next
                gamma_next[reference.exact_zero_mask] = initial.gamma[
                    reference.exact_zero_mask
                ]
                if not (
                    np.all(np.isfinite(positions_next))
                    and np.all(np.isfinite(gamma_next))
                    and np.all(np.isfinite(tracer_next))
                ):
                    raise FloatingPointError(
                        f"non-finite state after stream IR-WRK3 stage {stage}"
                    )
                return (
                    position_storage_next,
                    gamma_storage_next,
                    tracer_storage_next,
                    positions_next,
                    gamma_next,
                    tracer_next,
                )

            (
                position_storage,
                gamma_storage,
                tracer_storage,
                positions,
                gamma,
                tracer,
            ) = execute_phase("state_update", update_stage_state)

            def reconstruct_post() -> tuple[FloatArray, ParticleState]:
                sigma_next = np.array(reconstruct_sigma(gamma, reference), copy=True)
                verify_all()
                stage_post = ParticleState(
                    _frozen_float64("stage_post.positions", positions, (3,)),
                    _frozen_float64("stage_post.gamma", gamma, (3,)),
                    _frozen_float64("stage_post.sigma", sigma_next, ()),
                )
                return sigma_next, stage_post

            sigma, post = execute_phase("post_reconstruction", reconstruct_post)

            def check_invariant() -> tuple[FloatArray, float]:
                residual = _invariant_log_residual(post.gamma, post.sigma, reference)
                normalized_max = _enforce_invariant_gate(post, reference, residual)
                return residual, normalized_max

            (
                invariant_residual,
                invariant_residual_over_slog_max,
            ) = execute_phase("invariant_gate", check_invariant)

            def freeze_stage_post_evidence() -> tuple[FloatArray, ...]:
                return (
                    _frozen_float64("tracer_post", tracer, (3,)),
                    _frozen_float64("position_storage_post", position_storage, (3,)),
                    _frozen_float64("gamma_storage_post", gamma_storage, (3,)),
                    _frozen_float64("tracer_storage_post", tracer_storage, (3,)),
                )

            (
                tracer_post,
                position_storage_post,
                gamma_storage_post,
                tracer_storage_post,
            ) = execute_phase("post_evidence", freeze_stage_post_evidence)

            def make_compact_record() -> tuple[IRWRK3StreamStageRecord, str]:
                draft = IRWRK3StreamStageRecord(
                    substep=substep,
                    stage=stage,
                    a=float(a_coefficient),
                    b=float(b_coefficient),
                    substep_delta_time=float(substep_delta_time),
                    source_state_sha256=field.source_state_sha256,
                    pre_state_sha256=pre_state_sha,
                    post_state_sha256=_stream_state_sha256(post),
                    tracer_pre_sha256=tracer_field.tracer_state_sha256,
                    tracer_post_sha256=_array_sha256(tracer_post),
                    velocity_sha256=_array_sha256(field.velocity),
                    jacobian_sha256=_array_sha256(field.jacobian),
                    gamma_rate_sha256=_array_sha256(rhs.gamma_rate),
                    tracer_velocity_sha256=_array_sha256(tracer_field.velocity),
                    invariant_residual_sha256=_array_sha256(invariant_residual),
                    invariant_residual_max=(
                        float(np.max(invariant_residual))
                        if invariant_residual.shape[0]
                        else 0.0
                    ),
                    invariant_residual_over_slog_max=(invariant_residual_over_slog_max),
                    position_storage_pre_sha256=_array_sha256(position_storage_pre),
                    gamma_storage_pre_sha256=_array_sha256(gamma_storage_pre),
                    tracer_storage_pre_sha256=_array_sha256(tracer_storage_pre),
                    position_storage_post_sha256=_array_sha256(position_storage_post),
                    gamma_storage_post_sha256=_array_sha256(gamma_storage_post),
                    tracer_storage_post_sha256=_array_sha256(tracer_storage_post),
                    parent_token=parent_token,
                    evidence=evidence,
                    record_sha256="",
                    previous_chain_sha256=chain_sha,
                    chain_sha256="",
                )
                record_sha = _stage_record_sha256(draft)
                terminal_chain = hashlib.sha256(
                    (
                        "fluxv-ir-wrk3-stream-stage-link-v1\0" + chain_sha + record_sha
                    ).encode("ascii")
                ).hexdigest()
                return (
                    replace(
                        draft,
                        record_sha256=record_sha,
                        chain_sha256=terminal_chain,
                    ),
                    terminal_chain,
                )

            record, next_chain_sha = execute_phase(
                "compact_record", make_compact_record
            )

            def commit_compact_record() -> None:
                records.append(record)

            execute_phase("compact_commit", commit_compact_record)
            chain_sha = next_chain_sha
            evidence_byte_count = next_evidence_bytes

    final_state = ParticleState(
        _frozen_float64("final_state.positions", positions, (3,)),
        _frozen_float64("final_state.gamma", gamma, (3,)),
        _frozen_float64("final_state.sigma", sigma, ()),
    )
    final_tracer = _frozen_float64("final_tracer_positions", tracer, (3,))
    final_state_sha = _stream_state_sha256(final_state)
    final_tracer_sha = _array_sha256(final_tracer)
    stage_count = 3 * effective_substeps
    counters = IRWRK3StreamCounters(
        invariant_reference_freeze_count=1,
        substep_count=transport_substeps,
        stage_count=stage_count,
        physical_field_call_count=stage_count,
        tracer_field_call_count=stage_count if tracer_count else 0,
        observer_call_count=stage_count if observer is not None else 0,
        stage_pre_reconstruction_count=stage_count,
        stage_post_reconstruction_count=stage_count,
        physical_rhs_call_count=stage_count,
        storage_reset_count=transport_substeps,
        tracer_storage_reset_count=(transport_substeps if tracer_count else 0),
        sigma_storage_update_count=0,
        relaxation_call_count=0,
        compact_stage_record_count=stage_count,
        retained_stage_array_count=0,
        evidence_byte_count=evidence_byte_count,
    )
    result = IRWRK3StreamResult(
        final_state=final_state,
        final_tracer_positions=final_tracer,
        initial_state_sha256=initial_state_sha,
        final_state_sha256=final_state_sha,
        initial_tracer_sha256=initial_tracer_sha,
        final_tracer_sha256=final_tracer_sha,
        invariant_reference_sha256=reference_sha,
        stages=tuple(records),
        counters=counters,
        delta_time=integrated_total,
        transport_substeps=effective_substeps,
        parent_token=parent_token,
        stage_chain_sha256=chain_sha,
        result_sha256="",
    )
    result = replace(result, result_sha256=_result_sha256(result))
    validate_tree(result)
    verify_all()
    try:
        register_result(result)
    except duplicate_error_type as duplicate:
        canonical = duplicate.canonical_result
        validate_tree(canonical)
        attest_result(canonical)
        return canonical
    return result


def validate_ir_wrk3_stream_result(
    result: IRWRK3StreamResult,
) -> IRWRK3StreamResult:
    """Validate a live-issued immutable stream result without rerunning fields."""

    verify_runtime = _assert_runtime_bindings
    verify_registry = _assert_registry_function_bindings
    validate_tree = _validate_result_tree
    attest = _attest_live_result
    bindings = (
        ("_assert_runtime_bindings", verify_runtime),
        ("_assert_registry_function_bindings", verify_registry),
        ("_validate_result_tree", validate_tree),
        ("_attest_live_result", attest),
    )
    frozen = tuple(
        (
            name,
            function,
            function.__code__,
            function.__defaults__,
            function.__kwdefaults__,
        )
        for name, function in bindings
    )
    for name, function, code, defaults, kwdefaults in frozen:
        if (
            globals().get(name) is not function
            or function.__code__ is not code
            or function.__defaults__ is not defaults
            or function.__kwdefaults__ is not kwdefaults
        ):
            raise RuntimeError(f"stream validator trusted function drift: {name}")
    verify_runtime()
    verify_registry()
    validated = validate_tree(result)
    verify_registry()
    attest(validated)
    return validated


__all__ = (
    "IRWRK3CompactStageRecord",
    "IRWRK3StreamCounters",
    "IRWRK3StreamEvidence",
    "IRWRK3StreamResult",
    "IRWRK3StreamStageRecord",
    "IRWRK3StreamStageView",
    "IRWRK3StreamStopped",
    "ir_wrk3_stream_macro",
    "make_ir_wrk3_stream_evidence",
    "validate_ir_wrk3_stream_result",
)
