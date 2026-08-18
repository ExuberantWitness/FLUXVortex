"""Invariant-reconstructed Williamson RK3 for the pinned rVPM model.

This module is an isolated, observation-free mechanics oracle.  It advances
the reduced ``(X, Gamma)`` system for exactly ``f=0, g=1/5`` and reconstructs
``sigma`` from the macro-frozen per-particle invariant
``||Gamma|| * sigma**2``.  It does not call Ptera, apply relaxation, clip a
core radius, or change the continuous rVPM right-hand side.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import math
import threading
import types
from typing import Callable, Final

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .rvpm_transport import ParticleState

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
_FUNCTION_TYPE: Final[type] = types.FunctionType

FORMULATION_F: Final[float] = 0.0
FORMULATION_G: Final[float] = 0.2
ACTIVE_GAMMA_MAXABS_MIN: Final[float] = math.sqrt(np.finfo(np.float64).tiny)
MAX_PARTICLE_COUNT: Final[int] = 1_000_000
MAX_TRACER_COUNT: Final[int] = 1_000_000
MAX_SUBSTEPS: Final[int] = 4096
MAX_RETAINED_STAGE_ROWS: Final[int] = 1_000_000
MAX_LIVE_RESULT_COUNT: Final[int] = 4096
INVARIANT_LOG_ATOL: Final[float] = 512.0 * np.finfo(np.float64).eps

_CONTRACT_LITERAL: Final[tuple[float | int, ...]] = (
    0.0,
    0.2,
    math.sqrt(np.finfo(np.float64).tiny),
    512.0 * np.finfo(np.float64).eps,
    1_000_000,
    1_000_000,
    4096,
    1_000_000,
    4096,
)

_NUMPY_BINDING_NAMES: Final[tuple[str, ...]] = (
    "abs",
    "all",
    "any",
    "array",
    "array_equal",
    "asarray",
    "ascontiguousarray",
    "bool_",
    "count_nonzero",
    "dtype",
    "einsum",
    "empty",
    "exp",
    "finfo",
    "float64",
    "frombuffer",
    "integer",
    "isfinite",
    "log",
    "max",
    "maximum",
    "ndarray",
    "nextafter",
    "ones",
    "ones_like",
    "floating",
    "sqrt",
    "zeros",
    "zeros_like",
)
_FROZEN_NUMPY_BINDINGS: Final[tuple[tuple[str, object], ...]] = tuple(
    (name, getattr(np, name)) for name in _NUMPY_BINDING_NAMES
)
_FROZEN_MATH_BINDINGS: Final[tuple[tuple[str, object], ...]] = tuple(
    (name, getattr(math, name)) for name in ("isfinite", "log", "sqrt")
)
_FROZEN_STANDARD_LIBRARY_BINDINGS: Final[
    tuple[tuple[str, object, str, object], ...]
] = (
    ("hashlib.sha256", hashlib, "sha256", hashlib.sha256),
    ("json.dumps", json, "dumps", json.dumps),
)


def _assert_runtime_bindings(
    numpy_bindings: tuple[tuple[str, object], ...] = _FROZEN_NUMPY_BINDINGS,
    math_bindings: tuple[tuple[str, object], ...] = _FROZEN_MATH_BINDINGS,
    standard_bindings: tuple[
        tuple[str, object, str, object], ...
    ] = _FROZEN_STANDARD_LIBRARY_BINDINGS,
) -> None:
    for name, frozen in numpy_bindings:
        if getattr(np, name) is not frozen:
            raise RuntimeError(f"NumPy runtime binding drift: {name}")
    for name, frozen in math_bindings:
        if getattr(math, name) is not frozen:
            raise RuntimeError(f"math runtime binding drift: {name}")
    for label, module, attribute, frozen in standard_bindings:
        if getattr(module, attribute) is not frozen:
            raise RuntimeError(f"standard-library runtime binding drift: {label}")


def _assert_runtime_contract(
    literal: tuple[float | int, ...] = _CONTRACT_LITERAL,
) -> tuple[float | int, ...]:
    observed = (
        FORMULATION_F,
        FORMULATION_G,
        ACTIVE_GAMMA_MAXABS_MIN,
        INVARIANT_LOG_ATOL,
        MAX_PARTICLE_COUNT,
        MAX_TRACER_COUNT,
        MAX_SUBSTEPS,
        MAX_RETAINED_STAGE_ROWS,
        MAX_LIVE_RESULT_COUNT,
    )
    for current, expected in zip(observed, literal, strict=True):
        if type(current) is not type(expected) or current != expected:
            raise RuntimeError("IR-WRK3 frozen contract global drift")
    return literal


def _preflight_leading_count(
    name: str,
    value: object,
    shape_tail: tuple[int, ...],
    *,
    cap: int,
) -> int:
    """Read only shape/length metadata and reject oversize inputs before copying."""

    shape = getattr(value, "shape", None)
    if shape is not None:
        if type(shape) is not tuple or any(
            type(dimension) is not int for dimension in shape
        ):
            raise ValueError(f"{name} must expose an exact integer shape tuple")
        if len(shape) != 1 + len(shape_tail) or shape[1:] != shape_tail:
            expected = (
                "(n,)" if not shape_tail else f"(n, {', '.join(map(str, shape_tail))})"
            )
            raise ValueError(f"{name} must have shape {expected}")
        count = shape[0]
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


def _frozen_float64(
    name: str, value: ArrayLike, shape_tail: tuple[int, ...]
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


def _frozen_bool(value: ArrayLike, shape: tuple[int, ...]) -> BoolArray:
    array = np.asarray(value)
    if array.dtype != np.dtype(np.bool_) or array.shape != shape:
        raise ValueError("boolean ledger has an invalid dtype or shape")
    frozen = np.frombuffer(
        np.ascontiguousarray(array).tobytes(order="C"), dtype=np.bool_
    )
    return frozen.reshape(shape)


def _require_frozen_array(
    name: str,
    value: object,
    *,
    dtype: np.dtype[np.generic],
    shape: tuple[int, ...],
) -> np.ndarray:
    if type(value) is not np.ndarray:
        raise ValueError(f"{name} must be an exact ndarray")
    if value.dtype != dtype or value.shape != shape:
        raise ValueError(f"{name} has an invalid dtype or shape")
    if value.flags.writeable or not value.flags.c_contiguous:
        raise ValueError(f"{name} must be readonly and C-contiguous")
    # Every array emitted by this module owns an immutable ``bytes`` payload.
    # Merely clearing WRITEABLE on a view is insufficient: a caller could retain
    # a mutable base array and mutate the supposedly frozen evidence through it.
    ancestor: object = value
    seen: set[int] = set()
    while type(ancestor) is np.ndarray:
        if id(ancestor) in seen:
            raise ValueError(f"{name} has a cyclic array ownership chain")
        seen.add(id(ancestor))
        if ancestor.flags.writeable or not ancestor.flags.c_contiguous:
            raise ValueError(f"{name} must have an immutable C-contiguous base chain")
        ancestor = ancestor.base
    if type(ancestor) is not bytes:
        raise ValueError(f"{name} must be backed by exact immutable bytes")
    if dtype == np.dtype(np.float64) and not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must contain only finite values")
    return value


def _frozen_state(
    positions: ArrayLike,
    gamma: ArrayLike,
    sigma: ArrayLike,
    particle_cap: int = 1_000_000,
) -> ParticleState:
    counts = (
        _preflight_leading_count("positions", positions, (3,), cap=particle_cap),
        _preflight_leading_count("gamma", gamma, (3,), cap=particle_cap),
        _preflight_leading_count("sigma", sigma, (), cap=particle_cap),
    )
    if counts[0] != counts[1] or counts[0] != counts[2]:
        raise ValueError("particle arrays must have the same leading dimension")
    positions_array = _frozen_float64("positions", positions, (3,))
    gamma_array = _frozen_float64("gamma", gamma, (3,))
    sigma_array = _frozen_float64("sigma", sigma, ())
    if not (positions_array.shape[0] == gamma_array.shape[0] == sigma_array.shape[0]):
        raise ValueError("particle arrays must have the same leading dimension")
    if positions_array.shape[0] > particle_cap:
        raise ValueError("particle cap exceeded")
    if np.any(sigma_array <= 0.0):
        raise ValueError("sigma must be strictly positive")
    return ParticleState(positions_array, gamma_array, sigma_array)


def _array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(contiguous.dtype.str.encode("ascii"))
    digest.update(json.dumps(contiguous.shape, separators=(",", ":")).encode("ascii"))
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def _state_sha256(state: ParticleState) -> str:
    return hashlib.sha256(
        (
            "fluxv-ir-wrk3-state-v1\0"
            + _array_sha256(state.positions)
            + _array_sha256(state.gamma)
            + _array_sha256(state.sigma)
        ).encode("ascii")
    ).hexdigest()


def _stable_row_norms(gamma: FloatArray) -> tuple[FloatArray, FloatArray]:
    max_abs = np.max(np.abs(gamma), axis=1)
    norms = np.zeros(gamma.shape[0], dtype=np.float64)
    active = max_abs != 0.0
    if np.any(active):
        scaled = gamma[active] / max_abs[active, None]
        norms[active] = max_abs[active] * np.sqrt(np.einsum("ni,ni->n", scaled, scaled))
    if not np.all(np.isfinite(norms)):
        raise FloatingPointError("scaled gamma norm is non-finite")
    return norms, max_abs


def _validate_formulation(formulation_f: float, formulation_g: float) -> None:
    if isinstance(formulation_f, (bool, np.bool_)) or isinstance(
        formulation_g, (bool, np.bool_)
    ):
        raise ValueError("formulation parameters must be real numbers, not booleans")
    if type(formulation_f) not in (float, int) or type(formulation_g) not in (
        float,
        int,
    ):
        raise ValueError("formulation parameters must be built-in real scalars")
    if not math.isfinite(float(formulation_f)) or not math.isfinite(
        float(formulation_g)
    ):
        raise ValueError("formulation parameters must be finite")
    if float(formulation_f) != 0.0 or float(formulation_g) != 0.2:
        raise ValueError("IR-WRK3 is defined only for f=0 and g=1/5")


@dataclass(frozen=True, slots=True)
class IRWRK3Field:
    velocity: FloatArray
    jacobian: FloatArray
    source_state_sha256: str
    parent_token: str


@dataclass(frozen=True, slots=True)
class IRWRK3TracerField:
    velocity: FloatArray
    source_state_sha256: str
    tracer_state_sha256: str
    parent_token: str


@dataclass(frozen=True, slots=True)
class InvariantReference:
    gamma_norm_star: FloatArray
    log_gamma_norm_star: FloatArray
    sigma_star: FloatArray
    exact_zero_mask: BoolArray
    parent_state_sha256: str
    reference_sha256: str


@dataclass(frozen=True, slots=True)
class IRWRK3RHS:
    velocity: FloatArray
    jacobian: FloatArray
    stretching: FloatArray
    chi: FloatArray
    gamma_rate: FloatArray
    sigma_rate_diagnostic: FloatArray
    chain_rule_relative_residual: FloatArray


@dataclass(frozen=True, slots=True)
class IRWRK3StageRecord:
    substep: int
    stage: int
    a: float
    b: float
    pre: ParticleState
    field: IRWRK3Field
    rhs: IRWRK3RHS
    post: ParticleState
    position_storage_pre: FloatArray
    gamma_storage_pre: FloatArray
    position_storage_post: FloatArray
    gamma_storage_post: FloatArray
    tracer_pre: FloatArray
    tracer_field: IRWRK3TracerField
    tracer_post: FloatArray
    tracer_storage_pre: FloatArray
    tracer_storage_post: FloatArray
    invariant_log_residual: FloatArray
    source_state_sha256: str
    post_state_sha256: str
    trace_sha256: str

    @property
    def tracer_velocity(self) -> FloatArray:
        return self.tracer_field.velocity


@dataclass(frozen=True, slots=True)
class IRWRK3Counters:
    invariant_reference_freeze_count: int
    substep_count: int
    stage_count: int
    physical_field_call_count: int
    tracer_field_call_count: int
    stage_pre_reconstruction_count: int
    stage_post_reconstruction_count: int
    physical_rhs_call_count: int
    sigma_storage_update_count: int
    relaxation_call_count: int
    tracer_storage_reset_count: int


@dataclass(frozen=True, slots=True)
class IRWRK3StepResult:
    final_state: ParticleState
    final_tracer_positions: FloatArray
    invariant_reference: InvariantReference
    stages: tuple[IRWRK3StageRecord, ...]
    counters: IRWRK3Counters
    delta_time: float
    parent_token: str
    operator_trace: tuple[str, ...]
    stage_chain_sha256: str
    result_sha256: str


IRWRK3FieldEvaluator = Callable[[ParticleState], IRWRK3Field]
IRWRK3TracerEvaluator = Callable[[ParticleState, FloatArray, str], IRWRK3TracerField]


def _make_live_result_registry() -> tuple[Callable[..., object], ...]:
    """Create a sealed same-process issuance registry without a module-level dict."""

    lock = threading.RLock()
    registry: dict[int, tuple[IRWRK3StepResult, str]] = {}
    issuance_counter = 0

    def compute_seal() -> str:
        payload = (
            issuance_counter,
            tuple(
                (key, id(entry[0]), entry[1], entry[0].result_sha256)
                for key, entry in sorted(registry.items())
            ),
        )
        return hashlib.sha256(repr(payload).encode("ascii")).hexdigest()

    seal = compute_seal()

    def verify_seal() -> None:
        if compute_seal() != seal:
            raise RuntimeError("IR-WRK3 live result registry integrity drift")

    def register(result: IRWRK3StepResult) -> None:
        nonlocal issuance_counter, seal
        with lock:
            verify_seal()
            if len(registry) >= MAX_LIVE_RESULT_COUNT:
                raise RuntimeError("IR-WRK3 live result registry cap exceeded")
            key = id(result)
            if key in registry:
                raise RuntimeError("IR-WRK3 live result identity collision")
            registry[key] = (result, result.result_sha256)
            issuance_counter += 1
            seal = compute_seal()

    def assert_capacity() -> None:
        with lock:
            verify_seal()
            if len(registry) >= MAX_LIVE_RESULT_COUNT:
                raise RuntimeError("IR-WRK3 live result registry cap exceeded")

    def attest(result: IRWRK3StepResult) -> None:
        with lock:
            verify_seal()
            entry = registry.get(id(result))
            if (
                entry is None
                or entry[0] is not result
                or entry[1] != result.result_sha256
            ):
                raise ValueError("result is not the exact live issued IR-WRK3 report")

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
            if len(observed) != len(expected):
                raise RuntimeError(
                    "IR-WRK3 live result registry changed during callback"
                )
            if observed[0] != expected[0] or observed[1] != expected[1]:
                raise RuntimeError(
                    "IR-WRK3 live result registry changed during callback"
                )
            observed_entries = observed[2]
            expected_entries = expected[2]
            if len(observed_entries) != len(expected_entries):
                raise RuntimeError(
                    "IR-WRK3 live result registry changed during callback"
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
                        "IR-WRK3 live result registry changed during callback"
                    )

    return (register, assert_capacity, attest, snapshot, assert_snapshot_unchanged)


(
    _register_live_result,
    _assert_live_result_capacity,
    _attest_live_result,
    _snapshot_live_result_registry,
    _assert_live_result_registry_unchanged,
) = _make_live_result_registry()


def _validate_state_tree(name: str, state: ParticleState) -> int:
    if type(state) is not ParticleState:
        raise ValueError(f"{name} must be an exact ParticleState")
    if type(state.sigma) is not np.ndarray or state.sigma.ndim != 1:
        raise ValueError(f"{name}.sigma must be a one-dimensional exact ndarray")
    count = state.sigma.shape[0]
    _require_frozen_array(
        f"{name}.positions",
        state.positions,
        dtype=np.dtype(np.float64),
        shape=(count, 3),
    )
    _require_frozen_array(
        f"{name}.gamma",
        state.gamma,
        dtype=np.dtype(np.float64),
        shape=(count, 3),
    )
    sigma = _require_frozen_array(
        f"{name}.sigma",
        state.sigma,
        dtype=np.dtype(np.float64),
        shape=(count,),
    )
    if np.any(sigma <= 0.0):
        raise ValueError(f"{name}.sigma must be strictly positive")
    return count


def _validate_field_tree(name: str, field: IRWRK3Field, count: int) -> None:
    if type(field) is not IRWRK3Field:
        raise ValueError(f"{name} must be an exact IRWRK3Field")
    _require_frozen_array(
        f"{name}.velocity",
        field.velocity,
        dtype=np.dtype(np.float64),
        shape=(count, 3),
    )
    _require_frozen_array(
        f"{name}.jacobian",
        field.jacobian,
        dtype=np.dtype(np.float64),
        shape=(count, 3, 3),
    )
    if (
        type(field.source_state_sha256) is not str
        or len(field.source_state_sha256) != 64
    ):
        raise ValueError(f"{name}.source_state_sha256 is invalid")
    if type(field.parent_token) is not str or not field.parent_token:
        raise ValueError(f"{name}.parent_token is invalid")


def _validate_tracer_field_tree(
    name: str,
    field: IRWRK3TracerField,
    tracer_count: int,
) -> None:
    if type(field) is not IRWRK3TracerField:
        raise ValueError(f"{name} must be an exact IRWRK3TracerField")
    _require_frozen_array(
        f"{name}.velocity",
        field.velocity,
        dtype=np.dtype(np.float64),
        shape=(tracer_count, 3),
    )
    for label, value in (
        ("source_state_sha256", field.source_state_sha256),
        ("tracer_state_sha256", field.tracer_state_sha256),
    ):
        if type(value) is not str or len(value) != 64:
            raise ValueError(f"{name}.{label} is invalid")
    if type(field.parent_token) is not str or not field.parent_token:
        raise ValueError(f"{name}.parent_token is invalid")


def _validate_rhs_tree(name: str, rhs: IRWRK3RHS, count: int) -> None:
    if type(rhs) is not IRWRK3RHS:
        raise ValueError(f"{name} must be an exact IRWRK3RHS")
    for field_name, shape in (
        ("velocity", (count, 3)),
        ("jacobian", (count, 3, 3)),
        ("stretching", (count, 3)),
        ("chi", (count,)),
        ("gamma_rate", (count, 3)),
        ("sigma_rate_diagnostic", (count,)),
        ("chain_rule_relative_residual", (count,)),
    ):
        _require_frozen_array(
            f"{name}.{field_name}",
            getattr(rhs, field_name),
            dtype=np.dtype(np.float64),
            shape=shape,
        )


def _validate_stage_tree(record: IRWRK3StageRecord) -> None:
    if type(record) is not IRWRK3StageRecord:
        raise ValueError("stage records must have the exact frozen type")
    count = _validate_state_tree("stage.pre", record.pre)
    if _validate_state_tree("stage.post", record.post) != count:
        raise ValueError("stage pre/post particle count mismatch")
    _validate_field_tree("stage.field", record.field, count)
    _validate_rhs_tree("stage.rhs", record.rhs, count)
    if type(record.tracer_pre) is not np.ndarray or record.tracer_pre.ndim != 2:
        raise ValueError("stage.tracer_pre must be a two-dimensional exact ndarray")
    tracer_count = record.tracer_pre.shape[0]
    _validate_tracer_field_tree("stage.tracer_field", record.tracer_field, tracer_count)
    for field_name, shape in (
        ("position_storage_pre", (count, 3)),
        ("gamma_storage_pre", (count, 3)),
        ("position_storage_post", (count, 3)),
        ("gamma_storage_post", (count, 3)),
        ("tracer_pre", (tracer_count, 3)),
        ("tracer_post", (tracer_count, 3)),
        ("tracer_storage_pre", (tracer_count, 3)),
        ("tracer_storage_post", (tracer_count, 3)),
        ("invariant_log_residual", (count,)),
    ):
        _require_frozen_array(
            f"stage.{field_name}",
            getattr(record, field_name),
            dtype=np.dtype(np.float64),
            shape=shape,
        )
    for field_name in ("source_state_sha256", "post_state_sha256", "trace_sha256"):
        value = getattr(record, field_name)
        if type(value) is not str or len(value) != 64:
            raise ValueError(f"stage.{field_name} is invalid")


def _reference_digest(
    parent_state_sha256: str,
    gamma_norm_star: FloatArray,
    log_gamma_norm_star: FloatArray,
    sigma_star: FloatArray,
    exact_zero_mask: BoolArray,
) -> str:
    return hashlib.sha256(
        (
            "fluxv-ir-wrk3-invariant-reference-v1\0"
            + parent_state_sha256
            + _array_sha256(gamma_norm_star)
            + _array_sha256(log_gamma_norm_star)
            + _array_sha256(sigma_star)
            + _array_sha256(exact_zero_mask)
        ).encode("ascii")
    ).hexdigest()


def _validate_invariant_reference(
    reference: InvariantReference,
) -> InvariantReference:
    if type(reference) is not InvariantReference:
        raise ValueError("reference must be an exact InvariantReference")
    if type(reference.sigma_star) is not np.ndarray or reference.sigma_star.ndim != 1:
        raise ValueError("reference.sigma_star must be a one-dimensional exact ndarray")
    count = reference.sigma_star.shape[0]
    norm = _require_frozen_array(
        "reference.gamma_norm_star",
        reference.gamma_norm_star,
        dtype=np.dtype(np.float64),
        shape=(count,),
    )
    log_norm = _require_frozen_array(
        "reference.log_gamma_norm_star",
        reference.log_gamma_norm_star,
        dtype=np.dtype(np.float64),
        shape=(count,),
    )
    sigma = _require_frozen_array(
        "reference.sigma_star",
        reference.sigma_star,
        dtype=np.dtype(np.float64),
        shape=(count,),
    )
    zero = _require_frozen_array(
        "reference.exact_zero_mask",
        reference.exact_zero_mask,
        dtype=np.dtype(np.bool_),
        shape=(count,),
    )
    if np.any(sigma <= 0.0):
        raise ValueError("reference sigma must be strictly positive")
    if np.any(norm[zero] != 0.0) or np.any(log_norm[zero] != 0.0):
        raise ValueError("exact-zero reference rows are inconsistent")
    active = ~zero
    if np.any(norm[active] <= math.sqrt(np.finfo(np.float64).tiny)):
        raise ValueError("active invariant reference is below the frozen threshold")
    if np.any(np.log(norm[active]) != log_norm[active]):
        raise ValueError("reference norm/log ledger is inconsistent")
    if (
        type(reference.parent_state_sha256) is not str
        or len(reference.parent_state_sha256) != 64
    ):
        raise ValueError("reference parent digest is invalid")
    expected = _reference_digest(
        reference.parent_state_sha256,
        norm,
        log_norm,
        sigma,
        zero,
    )
    if expected != reference.reference_sha256:
        raise ValueError("invariant reference digest mismatch")
    return reference


def make_ir_wrk3_field(
    state: ParticleState,
    velocity: ArrayLike,
    jacobian: ArrayLike,
    *,
    parent_token: str = "analytic-frozen-parent",
) -> IRWRK3Field:
    """Create a frozen field response explicitly bound to ``state``."""

    _assert_runtime_bindings()
    contract = _assert_runtime_contract()
    particle_count = _preflight_leading_count(
        "positions", state.positions, (3,), cap=int(contract[4])
    )
    gamma_count = _preflight_leading_count(
        "gamma", state.gamma, (3,), cap=int(contract[4])
    )
    sigma_count = _preflight_leading_count(
        "sigma", state.sigma, (), cap=int(contract[4])
    )
    velocity_count = _preflight_leading_count(
        "velocity", velocity, (3,), cap=int(contract[4])
    )
    jacobian_count = _preflight_leading_count(
        "jacobian", jacobian, (3, 3), cap=int(contract[4])
    )
    if (
        len(
            {
                particle_count,
                gamma_count,
                sigma_count,
                velocity_count,
                jacobian_count,
            }
        )
        != 1
    ):
        raise ValueError("field arrays must have the same leading dimension")
    validated = _frozen_state(state.positions, state.gamma, state.sigma)
    velocity_array = _frozen_float64("velocity", velocity, (3,))
    jacobian_array = _frozen_float64("jacobian", jacobian, (3, 3))
    if velocity_array.shape[0] != validated.positions.shape[0]:
        raise ValueError("velocity particle count mismatch")
    if jacobian_array.shape[0] != validated.positions.shape[0]:
        raise ValueError("jacobian particle count mismatch")
    if type(parent_token) is not str or not parent_token:
        raise ValueError("parent_token must be a non-empty exact string")
    return IRWRK3Field(
        velocity=velocity_array,
        jacobian=jacobian_array,
        source_state_sha256=_state_sha256(validated),
        parent_token=parent_token,
    )


def make_ir_wrk3_tracer_field(
    state: ParticleState,
    tracer_positions: ArrayLike,
    velocity: ArrayLike,
    *,
    parent_token: str = "analytic-frozen-parent",
) -> IRWRK3TracerField:
    """Create a frozen tracer field bound to one stage-pre source and tracer state."""

    _assert_runtime_bindings()
    contract = _assert_runtime_contract()
    particle_count = _preflight_leading_count(
        "positions", state.positions, (3,), cap=int(contract[4])
    )
    gamma_count = _preflight_leading_count(
        "gamma", state.gamma, (3,), cap=int(contract[4])
    )
    sigma_count = _preflight_leading_count(
        "sigma", state.sigma, (), cap=int(contract[4])
    )
    tracer_count = _preflight_leading_count(
        "tracer_positions", tracer_positions, (3,), cap=int(contract[5])
    )
    velocity_count = _preflight_leading_count(
        "tracer_velocity", velocity, (3,), cap=int(contract[5])
    )
    if particle_count != gamma_count or particle_count != sigma_count:
        raise ValueError("particle arrays must have the same leading dimension")
    if tracer_count != velocity_count:
        raise ValueError("tracer velocity shape mismatch")
    validated = _frozen_state(state.positions, state.gamma, state.sigma)
    tracers = _frozen_float64("tracer_positions", tracer_positions, (3,))
    velocity_array = _frozen_float64("tracer_velocity", velocity, (3,))
    if tracers.shape != velocity_array.shape:
        raise ValueError("tracer velocity shape mismatch")
    if tracers.shape[0] > int(contract[5]):
        raise ValueError("tracer cap exceeded")
    if type(parent_token) is not str or not parent_token:
        raise ValueError("parent_token must be a non-empty exact string")
    return IRWRK3TracerField(
        velocity=velocity_array,
        source_state_sha256=_state_sha256(validated),
        tracer_state_sha256=_array_sha256(tracers),
        parent_token=parent_token,
    )


def freeze_invariant_reference(state: ParticleState) -> InvariantReference:
    """Freeze one macro-step invariant reference and reject active underflow."""

    _assert_runtime_bindings()
    _assert_runtime_contract()
    validated = _frozen_state(state.positions, state.gamma, state.sigma)
    norms, max_abs = _stable_row_norms(validated.gamma)
    zero_mask = max_abs == 0.0
    invalid = (~zero_mask) & (max_abs <= math.sqrt(np.finfo(np.float64).tiny))
    if np.any(invalid):
        raise ValueError("active gamma is at or below the frozen near-zero threshold")
    if np.any((~zero_mask) & (norms <= 0.0)):
        raise FloatingPointError("active scaled gamma norm is not positive")
    log_norms = np.zeros_like(norms)
    log_norms[~zero_mask] = np.log(norms[~zero_mask])
    parent_sha = _state_sha256(validated)
    zero_frozen = _frozen_bool(zero_mask, zero_mask.shape)
    norm_frozen = _frozen_float64("gamma_norm_star", norms, ())
    log_frozen = _frozen_float64("log_gamma_norm_star", log_norms, ())
    sigma_frozen = _frozen_float64("sigma_star", validated.sigma, ())
    result = InvariantReference(
        gamma_norm_star=norm_frozen,
        log_gamma_norm_star=log_frozen,
        sigma_star=sigma_frozen,
        exact_zero_mask=zero_frozen,
        parent_state_sha256=parent_sha,
        reference_sha256=_reference_digest(
            parent_sha,
            norm_frozen,
            log_frozen,
            sigma_frozen,
            zero_frozen,
        ),
    )
    return _validate_invariant_reference(result)


def reconstruct_sigma(gamma: ArrayLike, reference: InvariantReference) -> FloatArray:
    """Reconstruct a finite positive core radius from a macro-frozen reference."""

    _assert_runtime_bindings()
    contract = _assert_runtime_contract()
    gamma_count = _preflight_leading_count("gamma", gamma, (3,), cap=int(contract[4]))
    reference = _validate_invariant_reference(reference)
    if gamma_count != reference.sigma_star.shape[0]:
        raise ValueError("gamma/reference particle count mismatch")
    gamma_array = _frozen_float64("gamma", gamma, (3,))
    if gamma_array.shape[0] != reference.sigma_star.shape[0]:
        raise ValueError("gamma/reference particle count mismatch")
    norms, max_abs = _stable_row_norms(gamma_array)
    zero_now = max_abs == 0.0
    if not np.array_equal(zero_now, reference.exact_zero_mask):
        raise ValueError("active/exact-zero particle classification changed")
    invalid = (~zero_now) & (max_abs <= math.sqrt(np.finfo(np.float64).tiny))
    if np.any(invalid):
        raise ValueError("active gamma reached the frozen near-zero threshold")
    result = np.asarray(reference.sigma_star, dtype=np.float64).copy()
    active = ~zero_now
    changed = active & (norms != reference.gamma_norm_star)
    if np.any(changed):
        log_norm = np.log(norms[changed])
        log_sigma = np.log(reference.sigma_star[changed]) + 0.5 * (
            reference.log_gamma_norm_star[changed] - log_norm
        )
        lower = math.log(float(np.nextafter(np.float64(0.0), np.float64(1.0))))
        upper = math.log(float(np.finfo(np.float64).max))
        if (
            np.any(~np.isfinite(log_sigma))
            or np.any(log_sigma < lower)
            or np.any(log_sigma > upper)
        ):
            raise FloatingPointError(
                "reconstructed sigma is outside finite positive Float64 range"
            )
        result[changed] = np.exp(log_sigma)
    if np.any(~np.isfinite(result)) or np.any(result <= 0.0):
        raise FloatingPointError("reconstructed sigma is not finite and positive")
    return _frozen_float64("reconstructed_sigma", result, ())


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


def _rhs(gamma: FloatArray, sigma: FloatArray, field: IRWRK3Field) -> IRWRK3RHS:
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
        raise FloatingPointError("IR-WRK3 RHS produced non-finite values")
    return IRWRK3RHS(
        velocity=_frozen_float64("rhs.velocity", field.velocity, (3,)),
        jacobian=_frozen_float64("rhs.jacobian", field.jacobian, (3, 3)),
        stretching=_frozen_float64("rhs.stretching", stretching, (3,)),
        chi=_frozen_float64("rhs.chi", chi, ()),
        gamma_rate=_frozen_float64("rhs.gamma_rate", gamma_rate, (3,)),
        sigma_rate_diagnostic=_frozen_float64("rhs.sigma_rate", sigma_rate, ()),
        chain_rule_relative_residual=_frozen_float64("rhs.chain_rule", chain, ()),
    )


def _stage_record_sha256(record: IRWRK3StageRecord) -> str:
    payload = {
        "domain": "fluxv-ir-wrk3-stage-v1",
        "substep": record.substep,
        "stage": record.stage,
        "a": float(record.a).hex(),
        "b": float(record.b).hex(),
        "pre": _state_sha256(record.pre),
        "post": _state_sha256(record.post),
        "field_source": record.field.source_state_sha256,
        "field_parent": record.field.parent_token,
        "field_velocity": _array_sha256(record.field.velocity),
        "field_jacobian": _array_sha256(record.field.jacobian),
        "rhs": {
            "velocity": _array_sha256(record.rhs.velocity),
            "jacobian": _array_sha256(record.rhs.jacobian),
            "stretching": _array_sha256(record.rhs.stretching),
            "chi": _array_sha256(record.rhs.chi),
            "gamma_rate": _array_sha256(record.rhs.gamma_rate),
            "sigma_rate": _array_sha256(record.rhs.sigma_rate_diagnostic),
            "chain": _array_sha256(record.rhs.chain_rule_relative_residual),
        },
        "storage": {
            "position_pre": _array_sha256(record.position_storage_pre),
            "gamma_pre": _array_sha256(record.gamma_storage_pre),
            "position_post": _array_sha256(record.position_storage_post),
            "gamma_post": _array_sha256(record.gamma_storage_post),
            "tracer_pre": _array_sha256(record.tracer_storage_pre),
            "tracer_post": _array_sha256(record.tracer_storage_post),
        },
        "tracer": {
            "pre": _array_sha256(record.tracer_pre),
            "velocity": _array_sha256(record.tracer_field.velocity),
            "source_state": record.tracer_field.source_state_sha256,
            "tracer_state": record.tracer_field.tracer_state_sha256,
            "parent": record.tracer_field.parent_token,
            "post": _array_sha256(record.tracer_post),
        },
        "invariant": _array_sha256(record.invariant_log_residual),
        "source_state": record.source_state_sha256,
        "post_state": record.post_state_sha256,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("ascii")
    ).hexdigest()


def _require_same_array_bytes(
    name: str, actual: np.ndarray, expected: np.ndarray
) -> None:
    if _array_sha256(actual) != _array_sha256(expected):
        raise ValueError(f"{name} does not match the independently replayed mechanics")


def _replay_rhs_arrays(
    gamma: FloatArray,
    sigma: FloatArray,
    field: IRWRK3Field,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray]:
    """Re-evaluate the pinned RHS without consuming the stored RHS record."""

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
        raise ValueError("independent RHS replay produced non-finite values")
    return arrays


_CORE_CLASS_NAMES: Final[tuple[str, ...]] = (
    "ParticleState",
    "IRWRK3Field",
    "IRWRK3TracerField",
    "InvariantReference",
    "IRWRK3RHS",
    "IRWRK3StageRecord",
    "IRWRK3Counters",
    "IRWRK3StepResult",
)
_DATACLASS_FIELD_ATTRIBUTE_NAMES: Final[tuple[str, ...]] = (
    "name",
    "type",
    "default",
    "default_factory",
    "repr",
    "hash",
    "init",
    "compare",
    "metadata",
    "kw_only",
    "_field_type",
)


def _freeze_class_function(value: object) -> tuple[object, ...] | None:
    if type(value) is _FUNCTION_TYPE:
        keyword_defaults = (
            None
            if value.__kwdefaults__ is None
            else (
                value.__kwdefaults__,
                tuple(
                    (key, item) for key, item in sorted(value.__kwdefaults__.items())
                ),
            )
        )
        closure = (
            None
            if value.__closure__ is None
            else tuple((cell, cell.cell_contents) for cell in value.__closure__)
        )
        defaults = (
            None
            if value.__defaults__ is None
            else (value.__defaults__, tuple(value.__defaults__))
        )
        return (value, value.__code__, defaults, keyword_defaults, closure)
    if type(value) is property:
        return (
            value,
            _freeze_class_function(value.fget),
            _freeze_class_function(value.fset),
            _freeze_class_function(value.fdel),
        )
    return None


def _freeze_core_class(name: str) -> tuple[object, ...]:
    class_object = globals()[name]
    class_namespace = vars(class_object)
    members = tuple(
        (
            member_name,
            member,
            _freeze_class_function(member),
        )
        for member_name, member in sorted(class_namespace.items())
    )
    annotations = class_namespace.get("__annotations__")
    annotation_items = (
        None if annotations is None else (annotations, tuple(annotations.items()))
    )
    dataclass_fields = class_namespace.get("__dataclass_fields__")
    field_items = (
        None
        if dataclass_fields is None
        else (
            dataclass_fields,
            tuple(
                (
                    field_name,
                    field,
                    tuple(
                        (attribute, getattr(field, attribute))
                        for attribute in _DATACLASS_FIELD_ATTRIBUTE_NAMES
                    ),
                )
                for field_name, field in dataclass_fields.items()
            ),
        )
    )
    dataclass_params = class_namespace.get("__dataclass_params__")
    params = (
        None
        if dataclass_params is None
        else (
            dataclass_params,
            tuple(
                (attribute, getattr(dataclass_params, attribute))
                for attribute in (
                    "init",
                    "repr",
                    "eq",
                    "order",
                    "unsafe_hash",
                    "frozen",
                )
            ),
        )
    )
    return (name, class_object, members, annotation_items, field_items, params)


_FROZEN_CORE_CLASS_BINDINGS: Final[tuple[tuple[object, ...], ...]] = tuple(
    _freeze_core_class(name) for name in _CORE_CLASS_NAMES
)


def _assert_frozen_class_function(
    name: str, current: object, frozen: tuple[object, ...] | None
) -> None:
    if frozen is None:
        return
    if type(current) is _FUNCTION_TYPE:
        expected, code, defaults, keyword_defaults, closure = frozen
        if current is not expected or current.__code__ is not code:
            raise RuntimeError(f"IR-WRK3 class function drift: {name}")
        if defaults is None:
            if current.__defaults__ is not None:
                raise RuntimeError(f"IR-WRK3 class function defaults drift: {name}")
        else:
            defaults_object, default_items = defaults
            if current.__defaults__ is not defaults_object or any(
                observed is not expected_item
                for observed, expected_item in zip(
                    current.__defaults__, default_items, strict=True
                )
            ):
                raise RuntimeError(f"IR-WRK3 class function defaults drift: {name}")
        if keyword_defaults is None:
            if current.__kwdefaults__ is not None:
                raise RuntimeError(f"IR-WRK3 class function kwdefaults drift: {name}")
        else:
            keyword_object, keyword_items = keyword_defaults
            if current.__kwdefaults__ is not keyword_object or tuple(
                current.__kwdefaults__.keys()
            ) != tuple(key for key, _ in keyword_items):
                raise RuntimeError(f"IR-WRK3 class function kwdefaults drift: {name}")
            for key, expected_item in keyword_items:
                if current.__kwdefaults__[key] is not expected_item:
                    raise RuntimeError(
                        f"IR-WRK3 class function kwdefaults drift: {name}"
                    )
        if closure is None:
            if current.__closure__ is not None:
                raise RuntimeError(f"IR-WRK3 class function closure drift: {name}")
        elif current.__closure__ is None or len(current.__closure__) != len(closure):
            raise RuntimeError(f"IR-WRK3 class function closure drift: {name}")
        else:
            for current_cell, (expected_cell, expected_value) in zip(
                current.__closure__, closure, strict=True
            ):
                if (
                    current_cell is not expected_cell
                    or current_cell.cell_contents is not expected_value
                ):
                    raise RuntimeError(f"IR-WRK3 class function closure drift: {name}")
        return
    if type(current) is property:
        expected, fget, fset, fdel = frozen
        if current is not expected:
            raise RuntimeError(f"IR-WRK3 class property drift: {name}")
        _assert_frozen_class_function(f"{name}.fget", current.fget, fget)
        _assert_frozen_class_function(f"{name}.fset", current.fset, fset)
        _assert_frozen_class_function(f"{name}.fdel", current.fdel, fdel)
        return
    raise RuntimeError(f"IR-WRK3 class member kind drift: {name}")


def _assert_core_class_bindings(
    frozen: tuple[tuple[object, ...], ...] = _FROZEN_CORE_CLASS_BINDINGS,
) -> None:
    for (
        name,
        expected_class,
        members,
        annotation_items,
        field_items,
        params,
    ) in frozen:
        current_class = globals().get(name)
        if current_class is not expected_class:
            raise RuntimeError(f"IR-WRK3 core class drift: {name}")
        namespace = vars(current_class)
        if tuple(sorted(namespace)) != tuple(
            member_name for member_name, _, _ in members
        ):
            raise RuntimeError(f"IR-WRK3 core class namespace drift: {name}")
        for member_name, expected_member, function_binding in members:
            current_member = namespace.get(member_name)
            if current_member is not expected_member:
                raise RuntimeError(
                    f"IR-WRK3 core class member drift: {name}.{member_name}"
                )
            _assert_frozen_class_function(
                f"{name}.{member_name}", current_member, function_binding
            )
        if annotation_items is not None:
            expected_mapping, expected_items = annotation_items
            current_mapping = namespace.get("__annotations__")
            if current_mapping is not expected_mapping or tuple(
                current_mapping.keys()
            ) != tuple(key for key, _ in expected_items):
                raise RuntimeError(f"IR-WRK3 class annotations drift: {name}")
            for key, expected_value in expected_items:
                if current_mapping[key] is not expected_value:
                    raise RuntimeError(f"IR-WRK3 class annotations drift: {name}")
        if field_items is not None:
            expected_mapping, expected_items = field_items
            current_mapping = namespace.get("__dataclass_fields__")
            if current_mapping is not expected_mapping or tuple(
                current_mapping.keys()
            ) != tuple(field_name for field_name, _, _ in expected_items):
                raise RuntimeError(f"IR-WRK3 dataclass field mapping drift: {name}")
            for field_name, expected_field, attributes in expected_items:
                current_field = current_mapping[field_name]
                if current_field is not expected_field:
                    raise RuntimeError(f"IR-WRK3 dataclass field drift: {name}")
                for attribute, expected_value in attributes:
                    if getattr(current_field, attribute) is not expected_value:
                        raise RuntimeError(
                            f"IR-WRK3 dataclass field attribute drift: "
                            f"{name}.{field_name}.{attribute}"
                        )
        if params is not None:
            expected_params, attributes = params
            current_params = namespace.get("__dataclass_params__")
            if current_params is not expected_params:
                raise RuntimeError(f"IR-WRK3 dataclass params drift: {name}")
            for attribute, expected_value in attributes:
                if getattr(current_params, attribute) is not expected_value:
                    raise RuntimeError(
                        f"IR-WRK3 dataclass params drift: {name}.{attribute}"
                    )


_CORE_BINDING_NAMES: Final[tuple[str, ...]] = (
    "np",
    "math",
    "threading",
    "types",
    "hashlib",
    "json",
    "replace",
    "ParticleState",
    "IRWRK3Field",
    "IRWRK3TracerField",
    "IRWRK3RHS",
    "IRWRK3StageRecord",
    "IRWRK3Counters",
    "IRWRK3StepResult",
    "_FUNCTION_TYPE",
    "_CORE_CLASS_NAMES",
    "_DATACLASS_FIELD_ATTRIBUTE_NAMES",
    "_FROZEN_CORE_CLASS_BINDINGS",
    "_make_live_result_registry",
    "_register_live_result",
    "_assert_live_result_capacity",
    "_attest_live_result",
    "_snapshot_live_result_registry",
    "_assert_live_result_registry_unchanged",
    "_assert_runtime_bindings",
    "_assert_runtime_contract",
    "_freeze_class_function",
    "_freeze_core_class",
    "_assert_frozen_class_function",
    "_assert_core_class_bindings",
    "_validate_formulation",
    "_preflight_leading_count",
    "_frozen_float64",
    "_frozen_bool",
    "_require_frozen_array",
    "_frozen_state",
    "_state_sha256",
    "_array_sha256",
    "_stable_row_norms",
    "_validate_state_tree",
    "_validate_field_tree",
    "_validate_tracer_field_tree",
    "_validate_rhs_tree",
    "_validate_stage_tree",
    "_reference_digest",
    "_validate_invariant_reference",
    "_invariant_log_residual",
    "_rhs",
    "_stage_record_sha256",
    "_require_same_array_bytes",
    "_replay_rhs_arrays",
    "freeze_invariant_reference",
    "reconstruct_sigma",
    "make_ir_wrk3_field",
    "make_ir_wrk3_tracer_field",
)
_CORE_BINDING_NAMESPACE: Final[dict[str, object]] = globals()
_MUTABLE_CLOSURE_BINDING_NAMES: Final[tuple[str, ...]] = (
    "_register_live_result",
    "_assert_live_result_capacity",
    "_attest_live_result",
    "_snapshot_live_result_registry",
    "_assert_live_result_registry_unchanged",
)


def _freeze_core_binding(
    name: str,
) -> tuple[
    str,
    object,
    object | None,
    object | None,
    tuple[tuple[str, object], ...] | None,
    tuple[str, tuple[object, ...]] | None,
]:
    value = _CORE_BINDING_NAMESPACE[name]
    if type(value) is not _FUNCTION_TYPE:
        return (name, value, None, None, None, None)
    keyword_defaults = (
        None
        if value.__kwdefaults__ is None
        else tuple(sorted(value.__kwdefaults__.items()))
    )
    if value.__closure__ is None:
        closure = None
    elif name in _MUTABLE_CLOSURE_BINDING_NAMES:
        closure_bindings: list[object] = []
        for freevar, cell in zip(
            value.__code__.co_freevars, value.__closure__, strict=True
        ):
            content = cell.cell_contents
            if freevar in ("issuance_counter", "seal"):
                closure_bindings.append(("dynamic", freevar, cell, type(content)))
            elif type(content) is _FUNCTION_TYPE:
                closure_bindings.append(
                    (
                        "function",
                        freevar,
                        cell,
                        content,
                        content.__code__,
                        content.__defaults__,
                        content.__kwdefaults__,
                    )
                )
            else:
                closure_bindings.append(("identity", freevar, cell, content))
        closure = ("mutable-state", tuple(closure_bindings))
    else:
        closure = (
            "contents",
            tuple(cell.cell_contents for cell in value.__closure__),
        )
    return (
        name,
        value,
        value.__code__,
        value.__defaults__,
        keyword_defaults,
        closure,
    )


_FROZEN_CORE_BINDINGS: Final[
    tuple[
        tuple[
            str,
            object,
            object | None,
            object | None,
            tuple[tuple[str, object], ...] | None,
            tuple[str, tuple[object, ...]] | None,
        ],
        ...,
    ]
] = tuple(_freeze_core_binding(name) for name in _CORE_BINDING_NAMES)


def _assert_core_bindings(
    frozen: tuple[
        tuple[
            str,
            object,
            object | None,
            object | None,
            tuple[tuple[str, object], ...] | None,
            tuple[str, tuple[object, ...]] | None,
        ],
        ...,
    ] = _FROZEN_CORE_BINDINGS,
    namespace: dict[str, object] = _CORE_BINDING_NAMESPACE,
    function_type: type = _FUNCTION_TYPE,
) -> None:
    for name, expected, code, defaults, keyword_defaults, closure in frozen:
        current = namespace.get(name)
        if current is not expected:
            raise RuntimeError(f"IR-WRK3 core runtime binding drift: {name}")
        if code is None:
            continue
        if type(current) is not function_type or current.__code__ is not code:
            raise RuntimeError(f"IR-WRK3 core function code drift: {name}")
        if current.__defaults__ is not defaults:
            raise RuntimeError(f"IR-WRK3 core function defaults drift: {name}")
        observed_keyword_defaults = (
            None
            if current.__kwdefaults__ is None
            else tuple(sorted(current.__kwdefaults__.items()))
        )
        if observed_keyword_defaults != keyword_defaults:
            raise RuntimeError(f"IR-WRK3 core function kwdefaults drift: {name}")
        if closure is None:
            if current.__closure__ is not None:
                raise RuntimeError(f"IR-WRK3 core function closure drift: {name}")
        elif current.__closure__ is None:
            raise RuntimeError(f"IR-WRK3 core function closure drift: {name}")
        elif closure[0] == "mutable-state":
            bindings = closure[1]
            if len(current.__closure__) != len(bindings):
                raise RuntimeError(f"IR-WRK3 core function closure drift: {name}")
            for current_cell, binding in zip(
                current.__closure__, bindings, strict=True
            ):
                mode, freevar, expected_cell, *details = binding
                if current_cell is not expected_cell:
                    raise RuntimeError(
                        f"IR-WRK3 core function closure cell drift: {name}.{freevar}"
                    )
                content = current_cell.cell_contents
                if mode == "dynamic":
                    if type(content) is not details[0]:
                        raise RuntimeError(
                            f"IR-WRK3 core dynamic closure type drift: "
                            f"{name}.{freevar}"
                        )
                    if freevar == "issuance_counter" and content < 0:
                        raise RuntimeError(
                            "IR-WRK3 issuance counter closure is invalid"
                        )
                    if freevar == "seal" and len(content) != 64:
                        raise RuntimeError("IR-WRK3 registry seal closure is invalid")
                elif mode == "function":
                    expected_function, code, defaults, keyword_defaults = details
                    if (
                        content is not expected_function
                        or type(content) is not function_type
                        or content.__code__ is not code
                        or content.__defaults__ is not defaults
                        or content.__kwdefaults__ is not keyword_defaults
                    ):
                        raise RuntimeError(
                            f"IR-WRK3 core closure function drift: " f"{name}.{freevar}"
                        )
                elif content is not details[0]:
                    raise RuntimeError(
                        f"IR-WRK3 core closure identity drift: {name}.{freevar}"
                    )
        else:
            observed_contents = tuple(
                cell.cell_contents for cell in current.__closure__
            )
            if observed_contents != closure[1]:
                raise RuntimeError(f"IR-WRK3 core function closure drift: {name}")
    _assert_core_class_bindings()


_FROZEN_CORE_VERIFIER: Final[object] = _assert_core_bindings
_FROZEN_CORE_VERIFIER_CODE: Final[object] = _assert_core_bindings.__code__
_FROZEN_CORE_VERIFIER_DEFAULTS: Final[object] = _assert_core_bindings.__defaults__


def ir_wrk3_step_with_external_field(
    state: ParticleState,
    delta_time: float,
    field_evaluator: IRWRK3FieldEvaluator,
    *,
    transport_substeps: int = 1,
    formulation_f: float = FORMULATION_F,
    formulation_g: float = FORMULATION_G,
    tracer_positions: ArrayLike | None = None,
    tracer_field_evaluator: IRWRK3TracerEvaluator | None = None,
    parent_token: str = "analytic-frozen-parent",
) -> IRWRK3StepResult:
    """Advance one macro step on the invariant leaf.

    When tracers are present, both physical and tracer fields are evaluated
    from the same immutable stage-pre particle state before either state is
    updated.  Low-storage residuals reset at every inner substep, while the
    invariant reference is frozen exactly once for the whole macro step.
    """

    # Keep the trusted verifiers in locals so a callback cannot redirect the
    # post-callback checks by rebinding their module globals.
    verify_runtime_bindings = _assert_runtime_bindings
    verify_core_bindings = _FROZEN_CORE_VERIFIER
    verifier_code = _FROZEN_CORE_VERIFIER_CODE
    verifier_defaults = _FROZEN_CORE_VERIFIER_DEFAULTS
    snapshot_live_registry = _snapshot_live_result_registry
    verify_live_registry_unchanged = _assert_live_result_registry_unchanged
    if (
        _assert_core_bindings is not verify_core_bindings
        or verify_core_bindings.__code__ is not verifier_code
        or verify_core_bindings.__defaults__ is not verifier_defaults
    ):
        raise RuntimeError("IR-WRK3 core verifier binding drift")
    verify_runtime_bindings()
    verify_core_bindings()
    _assert_live_result_capacity()
    live_registry_snapshot = snapshot_live_registry()
    contract = _assert_runtime_contract()
    _validate_formulation(formulation_f, formulation_g)
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
    if not 1 <= transport_substeps <= int(contract[6]):
        raise ValueError("transport_substeps is outside the frozen cap")
    h = float(delta_time) / transport_substeps
    if not math.isfinite(h) or h <= 0.0:
        raise ValueError("per-substep delta_time underflowed or is not finite")
    if not callable(field_evaluator):
        raise ValueError("field_evaluator must be callable")
    if type(parent_token) is not str or not parent_token:
        raise ValueError("parent_token must be a non-empty exact string")

    particle_count = _preflight_leading_count(
        "positions", state.positions, (3,), cap=int(contract[4])
    )
    gamma_count = _preflight_leading_count(
        "gamma", state.gamma, (3,), cap=int(contract[4])
    )
    sigma_count = _preflight_leading_count(
        "sigma", state.sigma, (), cap=int(contract[4])
    )
    if particle_count != gamma_count or particle_count != sigma_count:
        raise ValueError("particle arrays must have the same leading dimension")
    tracer_count = 0
    if tracer_positions is not None:
        tracer_count = _preflight_leading_count(
            "tracer_positions", tracer_positions, (3,), cap=int(contract[5])
        )
    retained_stage_rows = 3 * transport_substeps * (particle_count + tracer_count)
    if retained_stage_rows > int(contract[7]):
        raise ValueError("retained stage-evidence row cap exceeded")

    initial = _frozen_state(state.positions, state.gamma, state.sigma)
    reference = freeze_invariant_reference(initial)
    positions = np.array(initial.positions, dtype=np.float64, order="C", copy=True)
    gamma = np.array(initial.gamma, dtype=np.float64, order="C", copy=True)
    sigma = np.array(initial.sigma, dtype=np.float64, order="C", copy=True)

    if tracer_positions is None:
        tracer = np.empty((0, 3), dtype=np.float64)
        if tracer_field_evaluator is not None:
            raise ValueError("tracer evaluator requires tracer_positions")
    else:
        tracer_frozen = _frozen_float64("tracer_positions", tracer_positions, (3,))
        if not callable(tracer_field_evaluator):
            raise ValueError("tracer_field_evaluator is required for tracers")
        tracer = np.array(tracer_frozen, dtype=np.float64, order="C", copy=True)

    stage_records: list[IRWRK3StageRecord] = []
    trace: list[str] = ["invariant_reference_frozen"]

    for substep in range(1, transport_substeps + 1):
        position_storage = np.zeros_like(positions)
        gamma_storage = np.zeros_like(gamma)
        tracer_storage = np.zeros_like(tracer)
        trace.append(f"substep:{substep}:storage_reset")

        for stage, (a_coefficient, b_coefficient) in enumerate(
            zip(
                (0.0, -5.0 / 9.0, -153.0 / 128.0),
                (1.0 / 3.0, 15.0 / 16.0, 8.0 / 15.0),
                strict=True,
            ),
            start=1,
        ):
            sigma = np.array(reconstruct_sigma(gamma, reference), copy=True)
            pre = _frozen_state(positions, gamma, sigma)
            pre_sha = _state_sha256(pre)
            pos_storage_pre = _frozen_float64(
                "position_storage_pre", position_storage, (3,)
            )
            gamma_storage_pre = _frozen_float64(
                "gamma_storage_pre", gamma_storage, (3,)
            )
            tracer_pre = _frozen_float64("tracer_pre", tracer, (3,))
            tracer_storage_pre = _frozen_float64(
                "tracer_storage_pre", tracer_storage, (3,)
            )

            field_response = field_evaluator(pre)
            if (
                verify_core_bindings.__code__ is not verifier_code
                or verify_core_bindings.__defaults__ is not verifier_defaults
            ):
                raise RuntimeError("IR-WRK3 core verifier changed during callback")
            verify_runtime_bindings()
            verify_core_bindings()
            verify_live_registry_unchanged(live_registry_snapshot)
            if type(field_response) is not IRWRK3Field:
                raise ValueError("field_evaluator must return an exact IRWRK3Field")
            if field_response.source_state_sha256 != pre_sha:
                raise ValueError("field is not bound to the stage-pre state")
            if field_response.parent_token != parent_token:
                raise ValueError("field parent token mismatch")
            if (
                field_response.velocity.shape != positions.shape
                or field_response.jacobian.shape
                != (
                    positions.shape[0],
                    3,
                    3,
                )
            ):
                raise ValueError("field response shape mismatch")
            if not (
                np.all(np.isfinite(field_response.velocity))
                and np.all(np.isfinite(field_response.jacobian))
            ):
                raise ValueError("field response must be finite")
            field = make_ir_wrk3_field(
                pre,
                field_response.velocity,
                field_response.jacobian,
                parent_token=parent_token,
            )

            if tracer.shape[0]:
                assert tracer_field_evaluator is not None
                tracer_response = tracer_field_evaluator(pre, tracer_pre, parent_token)
                if (
                    verify_core_bindings.__code__ is not verifier_code
                    or verify_core_bindings.__defaults__ is not verifier_defaults
                ):
                    raise RuntimeError("IR-WRK3 core verifier changed during callback")
                verify_runtime_bindings()
                verify_core_bindings()
                verify_live_registry_unchanged(live_registry_snapshot)
                if type(tracer_response) is not IRWRK3TracerField:
                    raise ValueError(
                        "tracer_field_evaluator must return an exact IRWRK3TracerField"
                    )
                if tracer_response.source_state_sha256 != pre_sha:
                    raise ValueError("tracer field source-state digest mismatch")
                if tracer_response.tracer_state_sha256 != _array_sha256(tracer_pre):
                    raise ValueError("tracer field tracer-state digest mismatch")
                if tracer_response.parent_token != parent_token:
                    raise ValueError("tracer field parent token mismatch")
                tracer_field = make_ir_wrk3_tracer_field(
                    pre,
                    tracer_pre,
                    tracer_response.velocity,
                    parent_token=parent_token,
                )
            else:
                tracer_field = make_ir_wrk3_tracer_field(
                    pre,
                    tracer_pre,
                    np.empty((0, 3)),
                    parent_token=parent_token,
                )
            tracer_velocity = tracer_field.velocity

            rhs = _rhs(gamma, sigma, field)
            if np.any(rhs.chain_rule_relative_residual > float(contract[3])):
                raise FloatingPointError(
                    "IR-WRK3 chain-rule residual exceeds the frozen gate"
                )

            position_storage = a_coefficient * position_storage + h * field.velocity
            gamma_storage = a_coefficient * gamma_storage + h * rhs.gamma_rate
            tracer_storage = a_coefficient * tracer_storage + h * tracer_velocity
            positions = positions + b_coefficient * position_storage
            gamma = gamma + b_coefficient * gamma_storage
            tracer = tracer + b_coefficient * tracer_storage
            gamma[reference.exact_zero_mask] = initial.gamma[reference.exact_zero_mask]

            if not (
                np.all(np.isfinite(positions))
                and np.all(np.isfinite(gamma))
                and np.all(np.isfinite(tracer))
            ):
                raise FloatingPointError(
                    f"non-finite state after IR-WRK3 stage {stage}"
                )
            sigma = np.array(reconstruct_sigma(gamma, reference), copy=True)
            post = _frozen_state(positions, gamma, sigma)
            post_sha = _state_sha256(post)
            invariant_residual = _invariant_log_residual(
                post.gamma, post.sigma, reference
            )
            active = ~reference.exact_zero_mask
            if np.any(active):
                norms, _ = _stable_row_norms(post.gamma)
                slog = np.maximum.reduce(
                    (
                        np.ones(np.count_nonzero(active)),
                        np.abs(
                            np.log(norms[active])
                            - reference.log_gamma_norm_star[active]
                        ),
                        2.0
                        * np.abs(
                            np.log(post.sigma[active])
                            - np.log(reference.sigma_star[active])
                        ),
                    )
                )
                if np.any(invariant_residual[active] > float(contract[3]) * slog):
                    raise FloatingPointError(
                        "IR-WRK3 invariant residual exceeds the frozen gate"
                    )

            tracer_post = _frozen_float64("tracer_post", tracer, (3,))
            pos_storage_post = _frozen_float64(
                "position_storage_post", position_storage, (3,)
            )
            gamma_storage_post = _frozen_float64(
                "gamma_storage_post", gamma_storage, (3,)
            )
            tracer_storage_post = _frozen_float64(
                "tracer_storage_post", tracer_storage, (3,)
            )
            stage_record = IRWRK3StageRecord(
                substep=substep,
                stage=stage,
                a=float(a_coefficient),
                b=float(b_coefficient),
                pre=pre,
                field=field,
                rhs=rhs,
                post=post,
                position_storage_pre=pos_storage_pre,
                gamma_storage_pre=gamma_storage_pre,
                position_storage_post=pos_storage_post,
                gamma_storage_post=gamma_storage_post,
                tracer_pre=tracer_pre,
                tracer_field=tracer_field,
                tracer_post=tracer_post,
                tracer_storage_pre=tracer_storage_pre,
                tracer_storage_post=tracer_storage_post,
                invariant_log_residual=invariant_residual,
                source_state_sha256=pre_sha,
                post_state_sha256=post_sha,
                trace_sha256="",
            )
            stage_records.append(
                replace(stage_record, trace_sha256=_stage_record_sha256(stage_record))
            )
            trace.extend(
                (
                    f"substep:{substep}:stage:{stage}:pre_reconstruct",
                    f"substep:{substep}:stage:{stage}:physical_field",
                    f"substep:{substep}:stage:{stage}:tracer_field"
                    if tracer.shape[0]
                    else f"substep:{substep}:stage:{stage}:no_tracer",
                    f"substep:{substep}:stage:{stage}:rhs",
                    f"substep:{substep}:stage:{stage}:update",
                    f"substep:{substep}:stage:{stage}:post_reconstruct",
                )
            )

    final_state = _frozen_state(positions, gamma, sigma)
    final_tracer = _frozen_float64("final_tracer_positions", tracer, (3,))
    stages = tuple(stage_records)
    stage_chain = hashlib.sha256(
        (
            "fluxv-ir-wrk3-stage-chain-v1\0"
            + "".join(record.trace_sha256 for record in stages)
        ).encode("ascii")
    ).hexdigest()
    counters = IRWRK3Counters(
        invariant_reference_freeze_count=1,
        substep_count=transport_substeps,
        stage_count=3 * transport_substeps,
        physical_field_call_count=3 * transport_substeps,
        tracer_field_call_count=3 * transport_substeps if final_tracer.shape[0] else 0,
        stage_pre_reconstruction_count=3 * transport_substeps,
        stage_post_reconstruction_count=3 * transport_substeps,
        physical_rhs_call_count=3 * transport_substeps,
        sigma_storage_update_count=0,
        relaxation_call_count=0,
        tracer_storage_reset_count=transport_substeps if final_tracer.shape[0] else 0,
    )
    result_payload = {
        "domain": "fluxv-ir-wrk3-step-result-v1",
        "final_state": _state_sha256(final_state),
        "final_tracer": _array_sha256(final_tracer),
        "reference": reference.reference_sha256,
        "stage_chain": stage_chain,
        "delta_time": float(delta_time).hex(),
        "parent": parent_token,
        "counters": [getattr(counters, name) for name in counters.__dataclass_fields__],
        "trace": trace,
    }
    result_sha = hashlib.sha256(
        json.dumps(result_payload, sort_keys=True, separators=(",", ":")).encode(
            "ascii"
        )
    ).hexdigest()
    result = IRWRK3StepResult(
        final_state=final_state,
        final_tracer_positions=final_tracer,
        invariant_reference=reference,
        stages=stages,
        counters=counters,
        delta_time=float(delta_time),
        parent_token=parent_token,
        operator_trace=tuple(trace),
        stage_chain_sha256=stage_chain,
        result_sha256=result_sha,
    )
    verify_live_registry_unchanged(live_registry_snapshot)
    _register_live_result(result)
    return result


def validate_ir_wrk3_result(result: IRWRK3StepResult) -> IRWRK3StepResult:
    """Validate immutable structural and digest evidence without re-running fields."""

    if (
        _assert_core_bindings is not _FROZEN_CORE_VERIFIER
        or _assert_core_bindings.__code__ is not _FROZEN_CORE_VERIFIER_CODE
        or _assert_core_bindings.__defaults__ is not _FROZEN_CORE_VERIFIER_DEFAULTS
    ):
        raise RuntimeError("IR-WRK3 core verifier binding drift")
    _assert_runtime_bindings()
    _assert_core_bindings()
    _assert_runtime_contract()
    if type(result) is not IRWRK3StepResult:
        raise ValueError("result must be an exact IRWRK3StepResult")
    if type(result.stages) is not tuple:
        raise ValueError("result stages must be an exact immutable tuple")
    if type(result.delta_time) is not float or not math.isfinite(result.delta_time):
        raise ValueError("result delta_time must be an exact finite float")
    if result.delta_time <= 0.0:
        raise ValueError("result delta_time must be positive")
    _validate_invariant_reference(result.invariant_reference)
    if type(result.counters) is not IRWRK3Counters:
        raise ValueError("result counters must have the exact frozen type")
    for name in result.counters.__dataclass_fields__:
        value = getattr(result.counters, name)
        if type(value) is not int or value < 0:
            raise ValueError(f"counter {name} must be a nonnegative exact integer")
    final_count = _validate_state_tree("result.final_state", result.final_state)
    if (
        type(result.final_tracer_positions) is not np.ndarray
        or result.final_tracer_positions.ndim != 2
    ):
        raise ValueError("result.final_tracer_positions must be an exact 2-D ndarray")
    _require_frozen_array(
        "result.final_tracer_positions",
        result.final_tracer_positions,
        dtype=np.dtype(np.float64),
        shape=(result.final_tracer_positions.shape[0], 3),
    )
    if final_count > 1_000_000 or result.final_tracer_positions.shape[0] > 1_000_000:
        raise ValueError("result particle or tracer cap exceeded")
    if type(result.parent_token) is not str or not result.parent_token:
        raise ValueError("result parent token is invalid")
    if type(result.operator_trace) is not tuple or any(
        type(item) is not str for item in result.operator_trace
    ):
        raise ValueError("result operator trace must be an exact string tuple")
    for name in ("stage_chain_sha256", "result_sha256"):
        value = getattr(result, name)
        if type(value) is not str or len(value) != 64:
            raise ValueError(f"result {name} is invalid")
    if len(result.stages) != result.counters.stage_count:
        raise ValueError("stage count mismatch")
    if result.counters.stage_count != 3 * result.counters.substep_count:
        raise ValueError("each substep must contain exactly three stages")
    expected_stage_count = result.counters.stage_count
    expected_tracer_calls = (
        expected_stage_count if result.final_tracer_positions.shape[0] else 0
    )
    expected_counter_values = {
        "physical_field_call_count": expected_stage_count,
        "tracer_field_call_count": expected_tracer_calls,
        "stage_pre_reconstruction_count": expected_stage_count,
        "stage_post_reconstruction_count": expected_stage_count,
        "physical_rhs_call_count": expected_stage_count,
        "tracer_storage_reset_count": (
            result.counters.substep_count
            if result.final_tracer_positions.shape[0]
            else 0
        ),
    }
    for name, expected_value in expected_counter_values.items():
        if getattr(result.counters, name) != expected_value:
            raise ValueError(f"counter {name} does not match the executed stage ledger")
    if result.counters.invariant_reference_freeze_count != 1:
        raise ValueError("invariant reference must be frozen exactly once")
    if (
        result.counters.sigma_storage_update_count != 0
        or result.counters.relaxation_call_count != 0
    ):
        raise ValueError("forbidden sigma storage or relaxation was reported")
    if not result.stages:
        raise ValueError("result must contain at least one stage")
    substep_delta_time = result.delta_time / result.counters.substep_count
    if not math.isfinite(substep_delta_time) or substep_delta_time <= 0.0:
        raise ValueError("result per-substep delta_time is invalid")
    if (
        _state_sha256(result.stages[0].pre)
        != result.invariant_reference.parent_state_sha256
    ):
        raise ValueError("invariant reference is not bound to the first stage input")
    previous_post_sha: str | None = None
    previous_record: IRWRK3StageRecord | None = None
    initial_gamma = result.stages[0].pre.gamma
    for index, record in enumerate(result.stages):
        _validate_stage_tree(record)
        expected_substep = index // 3 + 1
        expected_stage = index % 3 + 1
        expected_a = (0.0, -5.0 / 9.0, -153.0 / 128.0)[expected_stage - 1]
        expected_b = (1.0 / 3.0, 15.0 / 16.0, 8.0 / 15.0)[expected_stage - 1]
        if record.substep != expected_substep or record.stage != expected_stage:
            raise ValueError("stage ordering ledger mismatch")
        if type(record.a) is not float or record.a != expected_a:
            raise ValueError("stage Williamson a coefficient mismatch")
        if type(record.b) is not float or record.b != expected_b:
            raise ValueError("stage Williamson b coefficient mismatch")
        if record.source_state_sha256 != _state_sha256(record.pre):
            raise ValueError("stage source-state digest mismatch")
        if record.post_state_sha256 != _state_sha256(record.post):
            raise ValueError("stage post-state digest mismatch")
        if (
            previous_post_sha is not None
            and record.source_state_sha256 != previous_post_sha
        ):
            raise ValueError("stage state chain is discontinuous")
        if previous_record is not None:
            _require_same_array_bytes(
                "stage tracer state continuity",
                record.tracer_pre,
                previous_record.tracer_post,
            )
        if record.field.source_state_sha256 != record.source_state_sha256:
            raise ValueError("physical field source-state binding mismatch")
        if record.field.parent_token != result.parent_token:
            raise ValueError("physical field parent binding mismatch")
        if record.tracer_field.source_state_sha256 != record.source_state_sha256:
            raise ValueError("tracer field source-state binding mismatch")
        if record.tracer_field.tracer_state_sha256 != _array_sha256(record.tracer_pre):
            raise ValueError("tracer field tracer-state binding mismatch")
        if record.tracer_field.parent_token != result.parent_token:
            raise ValueError("tracer field parent binding mismatch")

        _require_same_array_bytes(
            "RHS velocity", record.rhs.velocity, record.field.velocity
        )
        _require_same_array_bytes(
            "RHS jacobian", record.rhs.jacobian, record.field.jacobian
        )
        stretching, chi, gamma_rate, sigma_rate, chain = _replay_rhs_arrays(
            record.pre.gamma, record.pre.sigma, record.field
        )
        for label, actual, expected_array in (
            ("RHS stretching", record.rhs.stretching, stretching),
            ("RHS chi", record.rhs.chi, chi),
            ("RHS gamma_rate", record.rhs.gamma_rate, gamma_rate),
            ("RHS sigma_rate", record.rhs.sigma_rate_diagnostic, sigma_rate),
            ("RHS chain residual", record.rhs.chain_rule_relative_residual, chain),
        ):
            _require_same_array_bytes(label, actual, expected_array)

        if expected_stage == 1:
            expected_position_storage_pre = np.zeros_like(record.pre.positions)
            expected_gamma_storage_pre = np.zeros_like(record.pre.gamma)
            expected_tracer_storage_pre = np.zeros_like(record.tracer_pre)
        else:
            assert previous_record is not None
            expected_position_storage_pre = previous_record.position_storage_post
            expected_gamma_storage_pre = previous_record.gamma_storage_post
            expected_tracer_storage_pre = previous_record.tracer_storage_post
        _require_same_array_bytes(
            "position storage pre",
            record.position_storage_pre,
            expected_position_storage_pre,
        )
        _require_same_array_bytes(
            "gamma storage pre",
            record.gamma_storage_pre,
            expected_gamma_storage_pre,
        )
        _require_same_array_bytes(
            "tracer storage pre",
            record.tracer_storage_pre,
            expected_tracer_storage_pre,
        )

        expected_position_storage_post = (
            record.a * record.position_storage_pre
            + substep_delta_time * record.field.velocity
        )
        expected_gamma_storage_post = (
            record.a * record.gamma_storage_pre + substep_delta_time * gamma_rate
        )
        expected_tracer_storage_post = (
            record.a * record.tracer_storage_pre
            + substep_delta_time * record.tracer_field.velocity
        )
        _require_same_array_bytes(
            "position storage post",
            record.position_storage_post,
            expected_position_storage_post,
        )
        _require_same_array_bytes(
            "gamma storage post",
            record.gamma_storage_post,
            expected_gamma_storage_post,
        )
        _require_same_array_bytes(
            "tracer storage post",
            record.tracer_storage_post,
            expected_tracer_storage_post,
        )

        expected_positions = (
            record.pre.positions + record.b * expected_position_storage_post
        )
        expected_gamma = record.pre.gamma + record.b * expected_gamma_storage_post
        expected_gamma = np.array(expected_gamma, dtype=np.float64, copy=True)
        expected_gamma[result.invariant_reference.exact_zero_mask] = initial_gamma[
            result.invariant_reference.exact_zero_mask
        ]
        expected_sigma = reconstruct_sigma(expected_gamma, result.invariant_reference)
        expected_tracer = record.tracer_pre + record.b * expected_tracer_storage_post
        _require_same_array_bytes(
            "stage post positions", record.post.positions, expected_positions
        )
        _require_same_array_bytes("stage post gamma", record.post.gamma, expected_gamma)
        _require_same_array_bytes("stage post sigma", record.post.sigma, expected_sigma)
        _require_same_array_bytes(
            "stage tracer post", record.tracer_post, expected_tracer
        )
        expected_invariant = _invariant_log_residual(
            expected_gamma, expected_sigma, result.invariant_reference
        )
        _require_same_array_bytes(
            "stage invariant residual",
            record.invariant_log_residual,
            expected_invariant,
        )
        previous_post_sha = record.post_state_sha256
        previous_record = record
    if previous_post_sha != _state_sha256(result.final_state):
        raise ValueError("final state is not the last stage post-state")
    assert previous_record is not None
    _require_same_array_bytes(
        "final tracer positions",
        result.final_tracer_positions,
        previous_record.tracer_post,
    )
    expected_trace: list[str] = ["invariant_reference_frozen"]
    for substep in range(1, result.counters.substep_count + 1):
        expected_trace.append(f"substep:{substep}:storage_reset")
        for stage in range(1, 4):
            expected_trace.extend(
                (
                    f"substep:{substep}:stage:{stage}:pre_reconstruct",
                    f"substep:{substep}:stage:{stage}:physical_field",
                    (
                        f"substep:{substep}:stage:{stage}:tracer_field"
                        if result.final_tracer_positions.shape[0]
                        else f"substep:{substep}:stage:{stage}:no_tracer"
                    ),
                    f"substep:{substep}:stage:{stage}:rhs",
                    f"substep:{substep}:stage:{stage}:update",
                    f"substep:{substep}:stage:{stage}:post_reconstruct",
                )
            )
    if result.operator_trace != tuple(expected_trace):
        raise ValueError("operator trace does not match the independent stage replay")
    expected_chain = hashlib.sha256(
        (
            "fluxv-ir-wrk3-stage-chain-v1\0"
            + "".join(record.trace_sha256 for record in result.stages)
        ).encode("ascii")
    ).hexdigest()
    if expected_chain != result.stage_chain_sha256:
        raise ValueError("stage chain digest mismatch")
    for record in result.stages:
        if _stage_record_sha256(record) != record.trace_sha256:
            raise ValueError("stage record digest mismatch")
    result_payload = {
        "domain": "fluxv-ir-wrk3-step-result-v1",
        "final_state": _state_sha256(result.final_state),
        "final_tracer": _array_sha256(result.final_tracer_positions),
        "reference": result.invariant_reference.reference_sha256,
        "stage_chain": result.stage_chain_sha256,
        "delta_time": result.delta_time.hex(),
        "parent": result.parent_token,
        "counters": [
            getattr(result.counters, name)
            for name in result.counters.__dataclass_fields__
        ],
        "trace": list(result.operator_trace),
    }
    expected = hashlib.sha256(
        json.dumps(result_payload, sort_keys=True, separators=(",", ":")).encode(
            "ascii"
        )
    ).hexdigest()
    if expected != result.result_sha256:
        raise ValueError("result digest mismatch")
    _attest_live_result(result)
    return result


__all__ = [
    "ACTIVE_GAMMA_MAXABS_MIN",
    "FORMULATION_F",
    "FORMULATION_G",
    "INVARIANT_LOG_ATOL",
    "IRWRK3Counters",
    "IRWRK3Field",
    "IRWRK3RHS",
    "IRWRK3StageRecord",
    "IRWRK3StepResult",
    "IRWRK3TracerField",
    "InvariantReference",
    "freeze_invariant_reference",
    "ir_wrk3_step_with_external_field",
    "make_ir_wrk3_field",
    "make_ir_wrk3_tracer_field",
    "reconstruct_sigma",
    "validate_ir_wrk3_result",
]
