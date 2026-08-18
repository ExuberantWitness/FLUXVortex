"""Observation-free Baik-W2 coupling for the v5h15 IR-WRK3 gate.

The module owns only the mechanics boundary needed by preregistered block B3:
one live global row is committed, native Ptera consumes that row at its
collocation and four load batches, Ptera loads are copied, and the physical
cloud plus the common material/frontier tracer pack are advanced by one
streaming IR-WRK3 macro step.  Ptera remains the sole surface-load owner.

No paper observation, score, fitting path, or artifact writer is imported.
Failure after native Ptera work does not claim to roll Ptera back.  It does,
however, leave the release-row owner unadvanced and publishes no successful
v5h15 layer result unless the transport handoff closes completely.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from hashlib import sha256
import json
import math
from numbers import Real
import inspect
from threading import RLock
from typing import Any, Callable, Final, Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray

from fluxvortex.rvpm_ir_wrk3 import (
    IRWRK3Field,
    IRWRK3TracerField,
    make_ir_wrk3_field,
    make_ir_wrk3_tracer_field,
)
from fluxvortex.rvpm_ir_wrk3_fd_adapter import (
    DEFAULT_MAX_TARGET_COUNT,
    FDCallLedger,
    FDEvaluationRecord,
    FrozenParentCenteredFDAdapter,
    FrozenParentVelocity,
    FrozenParentVelocityEvaluator,
    ParentStateSHA256Getter,
    make_frozen_parent_velocity,
    validate_fd_call_ledger,
)
from fluxvortex.rvpm_ir_wrk3_v5h15_stream import (
    IRWRK3StreamResult,
    IRWRK3StreamStageView,
    ir_wrk3_stream_macro,
    make_ir_wrk3_stream_evidence,
    validate_ir_wrk3_stream_result,
)
from fluxvortex.rvpm_reference import direct_gaussian_erf_velocity_jacobian
from fluxvortex.rvpm_transport import ParticleState
from fluxvortex.solver import UVPMHybridSolver

from .fluxv_v5h3_native_feedback import (
    SURFACE_LOAD_OWNER,
    NativePteraRVPMFeedbackConfig,
    NativePteraRVPMFeedbackSolver,
    NativePteraRVPMFeedbackStepReport,
    NativePteraRVPMVelocityEvaluation,
    _PendingFeedbackStep,
    _array_sha256 as _v5h3_array_sha256,
)
from .fluxv_v5h4_ptera_rvpm_transport import (
    NOMINAL_RELATIVE_EPSILON,
    _assert_bindings as _assert_v5h4_bindings,
    ptera_parent_state_sha256,
)
from .fluxv_v5h10_row_owner import (
    ReleaseRowOwner,
    RowCommitResult,
    advance_release_row_transport_parent,
    attest_release_row_common_transport,
    begin_release_row_common_transport,
    release_row_transport_digest,
    validate_current_release_row_owner,
    validate_release_row_common_transport,
    validate_release_row_owner,
    validate_release_row_transport_attestation,
)


FloatArray = NDArray[np.float64]

COUPLING_INTERFACE_ID: Final = "fluxv-v5h15-baik-w2-ir-wrk3-coupling-v1"
CASE_ID: Final = "W2"
OBSERVATION_ACCESS: Final = "none"
SOURCE_LOAD_OWNER: Final = "forbidden"
SURFACE_FORCE_OWNER: Final = SURFACE_LOAD_OWNER
FORCE_SCORING_STATUS: Final = "blocked_no_gt_inner_mechanics_only"

PTERA_CHORDWISE_PANELS: Final = 2
PTERA_SPANWISE_PANELS: Final = 8
PTERA_STEPS_PER_CYCLE: Final = 32
PTERA_CYCLES: Final = 2
PTERA_DELTA_TIME_S: Final = 0.11125
ACTIVE_PTERA_STEPS: Final = (3, 4, 5)
ACTIVE_SOURCE_STEPS: Final = (4, 5, 6)
ACTIVE_BIRTH_MODES: Final = ("first", "continuous", "continuous")
FORMAL_TRANSPORT_SUBSTEPS: Final = (32, 64, 128)
BIRTH_WINDOW_K: Final = 5
BIRTH_WINDOW_R: Final = 4
FORMAL_EFFECTIVE_TRANSPORT_SUBSTEPS: Final = tuple(
    n - BIRTH_WINDOW_K + BIRTH_WINDOW_K * BIRTH_WINDOW_R
    for n in FORMAL_TRANSPORT_SUBSTEPS
)
DIAGNOSTIC_EFFECTIVE_TRANSPORT_SUBSTEPS: Final = (
    FORMAL_EFFECTIVE_TRANSPORT_SUBSTEPS[0],
)
EFFECTIVE_ROLE_BY_N: Final = {
    FORMAL_EFFECTIVE_TRANSPORT_SUBSTEPS[0]: "coarse",
    FORMAL_EFFECTIVE_TRANSPORT_SUBSTEPS[1]: "candidate",
    FORMAL_EFFECTIVE_TRANSPORT_SUBSTEPS[2]: "reference",
}


def graded_substep_delta_times(
    delta_time_s: float, transport_substeps: int
) -> tuple[float, ...]:
    """V5H15 birth-window grid: first K sub-steps refined by R, rest coarse.

    The nominal ``transport_substeps`` defines both widths; the returned grid
    has ``N_eff = N - K + K*R`` entries whose fsum equals ``delta_time_s`` up
    to float rounding (the stream records the exact integrated fsum).
    """

    fine = float(delta_time_s) / (transport_substeps * BIRTH_WINDOW_R)
    coarse = float(delta_time_s) / transport_substeps
    return (fine,) * (BIRTH_WINDOW_K * BIRTH_WINDOW_R) + (coarse,) * (
        transport_substeps - BIRTH_WINDOW_K
    )


MAX_ACTIVE_LAYERS: Final = 3
NO_PENETRATION_ATOL: Final = 1.0e-12
MAX_SOURCE_KELVIN_M2_PER_S: Final = 1.0e-10
STABILITY_J_H_MAX: Final = 1.5
STABILITY_CONVECTIVE_H_MAX: Final = 0.5
MAX_TEST_SUBSTEPS: Final = 16
SYNTHETIC_EFFECTIVE_MAX_SUBSTEPS: Final = MAX_TEST_SUBSTEPS + BIRTH_WINDOW_K * (
    BIRTH_WINDOW_R - 1
)
MAX_LIVE_LAYER_RESULTS: Final = 4096

_HEX: Final[frozenset[str]] = frozenset("0123456789abcdef")


def _exact_int(name: str, value: object, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an exact integer >= {minimum}")
    return value


def _finite_float(name: str, value: object, *, positive: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite real")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0.0):
        qualifier = " positive" if positive else ""
        raise ValueError(f"{name} must be a finite{qualifier} real")
    return result


def _validate_sha256(name: str, value: object) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in _HEX for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _preflight_count(
    name: str, value: object, trailing: tuple[int, ...], cap: int
) -> int:
    shape = getattr(value, "shape", None)
    if shape is not None:
        try:
            normalized = tuple(shape)
        except TypeError as error:
            raise ValueError(f"{name}.shape is invalid") from error
        if len(normalized) != len(trailing) + 1 or normalized[1:] != trailing:
            raise ValueError(f"{name} has invalid shape")
        count = normalized[0]
        if type(count) is not int or count < 0 or count > cap:
            raise ValueError(f"{name} count is outside the cap")
        return count
    try:
        count = len(value)  # type: ignore[arg-type]
    except (TypeError, AttributeError) as error:
        raise ValueError(f"{name} has no auditable leading count") from error
    if type(count) is not int or count < 0 or count > cap:
        raise ValueError(f"{name} count is outside the cap")
    return count


def _frozen_float64(
    name: str,
    value: ArrayLike,
    trailing: tuple[int, ...],
    *,
    cap: int = DEFAULT_MAX_TARGET_COUNT,
) -> FloatArray:
    count = _preflight_count(name, value, trailing, cap)
    original = np.asarray(value)
    if original.dtype.kind not in "iuf" or original.dtype.kind == "b":
        raise ValueError(f"{name} must have a real numeric dtype")
    array = np.array(original, dtype=np.float64, order="C", copy=True)
    if array.shape != (count, *trailing) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} has invalid shape or non-finite values")
    payload = array.tobytes(order="C")
    frozen = np.frombuffer(payload, dtype=np.float64).reshape(array.shape)
    if frozen.flags.writeable or not frozen.flags.c_contiguous:
        raise RuntimeError(f"{name} did not freeze as a bytes-backed array")
    return frozen


def _require_frozen_float64(
    name: str,
    value: object,
    shape: tuple[int, ...],
) -> FloatArray:
    if (
        type(value) is not np.ndarray
        or value.dtype != np.dtype(np.float64)
        or value.shape != shape
        or value.flags.writeable
        or not value.flags.c_contiguous
        or not np.all(np.isfinite(value))
    ):
        raise ValueError(f"{name} is not an exact readonly Float64 array")
    base: object = value
    seen: set[int] = set()
    while isinstance(base, np.ndarray):
        if id(base) in seen:
            raise ValueError(f"{name} has a cyclic ndarray base chain")
        seen.add(id(base))
        base = base.base
    if type(base) is not bytes:
        raise ValueError(f"{name} is not backed by exact immutable bytes")
    return value


def _array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = sha256()
    digest.update(contiguous.dtype.str.encode("ascii"))
    digest.update(json.dumps(contiguous.shape, separators=(",", ":")).encode("ascii"))
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def _stream_state_sha256(state: ParticleState) -> str:
    return sha256(
        (
            "fluxv-ir-wrk3-stream-state-v1\0"
            + _array_sha256(state.positions)
            + _array_sha256(state.gamma)
            + _array_sha256(state.sigma)
        ).encode("ascii")
    ).hexdigest()


def _hash_payload(domain: str, payload: object) -> str:
    return sha256(
        (
            domain + "\0" + json.dumps(payload, sort_keys=True, separators=(",", ":"))
        ).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class V5H15BaikCouplingConfig:
    """Frozen B3 settings with formal, diagnostic, and synthetic scopes."""

    transport_substeps: int
    formal_matrix: bool = False
    test_mode: bool = False
    diagnostic_smoke: bool = False
    active_layer_limit: int = MAX_ACTIVE_LAYERS
    delta_time_s: float = PTERA_DELTA_TIME_S
    birth_window_refinement: bool = False
    relative_epsilon: float = NOMINAL_RELATIVE_EPSILON
    particle_cap: int = DEFAULT_MAX_TARGET_COUNT
    no_penetration_atol: float = NO_PENETRATION_ATOL
    stability_j_h_max: float = STABILITY_J_H_MAX
    stability_convective_h_max: float = STABILITY_CONVECTIVE_H_MAX

    def __post_init__(self) -> None:
        _assert_execution_bindings()
        modes = (self.formal_matrix, self.test_mode, self.diagnostic_smoke)
        if any(type(value) is not bool for value in modes):
            raise TypeError(
                "formal_matrix, test_mode, and diagnostic_smoke must be exact bools"
            )
        if sum(modes) != 1:
            raise ValueError(
                "select exactly one of formal_matrix, test_mode, or diagnostic_smoke"
            )
        substeps = _exact_int("transport_substeps", self.transport_substeps, minimum=1)
        layer_limit = _exact_int(
            "active_layer_limit", self.active_layer_limit, minimum=1
        )
        if layer_limit > MAX_ACTIVE_LAYERS:
            raise ValueError("active_layer_limit exceeds the three-layer B3 slice")
        if self.formal_matrix:
            if substeps not in FORMAL_TRANSPORT_SUBSTEPS:
                raise ValueError("formal B3 N must be exactly one of 32, 64, 128")
            if (
                type(self.delta_time_s) is not float
                or self.delta_time_s != PTERA_DELTA_TIME_S
            ):
                raise ValueError("formal W2 delta_time_s is frozen")
            if layer_limit != MAX_ACTIVE_LAYERS:
                raise ValueError("formal B3 requires all three active layers")
        elif self.diagnostic_smoke:
            if substeps != FORMAL_TRANSPORT_SUBSTEPS[0]:
                raise ValueError("diagnostic smoke requires N=32")
            if layer_limit != 1:
                raise ValueError("diagnostic smoke requires one active layer")
            if (
                type(self.delta_time_s) is not float
                or self.delta_time_s != PTERA_DELTA_TIME_S
            ):
                raise ValueError("diagnostic W2 delta_time_s is frozen")
        elif substeps > MAX_TEST_SUBSTEPS:
            raise ValueError("synthetic test N exceeds the isolated small-test cap")
        relative = _finite_float(
            "relative_epsilon", self.relative_epsilon, positive=True
        )
        if relative != NOMINAL_RELATIVE_EPSILON:
            raise ValueError("B3 centered-FD relative epsilon is frozen at 2^-10")
        cap = _exact_int("particle_cap", self.particle_cap, minimum=1)
        if cap > DEFAULT_MAX_TARGET_COUNT:
            raise ValueError("particle_cap exceeds the FD/stream target cap")
        _finite_float("delta_time_s", self.delta_time_s, positive=True)
        if type(self.birth_window_refinement) is not bool:
            raise ValueError("birth_window_refinement must be an exact bool")
        if self.birth_window_refinement and (self.transport_substeps < BIRTH_WINDOW_K):
            raise ValueError("birth-window refinement requires transport_substeps >= K")
        no_penetration = _finite_float(
            "no_penetration_atol", self.no_penetration_atol, positive=True
        )
        if no_penetration > NO_PENETRATION_ATOL:
            raise ValueError("no_penetration_atol cannot relax the preregistered gate")
        if (
            _finite_float("stability_j_h_max", self.stability_j_h_max, positive=True)
            != STABILITY_J_H_MAX
        ):
            raise ValueError("Jacobian stability gate is frozen")
        if (
            _finite_float(
                "stability_convective_h_max",
                self.stability_convective_h_max,
                positive=True,
            )
            != STABILITY_CONVECTIVE_H_MAX
        ):
            raise ValueError("convective stability gate is frozen")

    @property
    def scope(self) -> str:
        if self.formal_matrix:
            return "formal"
        return "diagnostic_smoke" if self.diagnostic_smoke else "synthetic"

    @property
    def role(self) -> str:
        if self.diagnostic_smoke:
            return "diagnostic_smoke"
        if self.test_mode:
            return "synthetic"
        return {32: "coarse", 64: "candidate", 128: "reference"}[
            self.transport_substeps
        ]


@dataclass(frozen=True, slots=True)
class V5H15GlobalRowCommitRequest:
    ptera_step_index: int
    source_step_index: int
    source_time_s: float
    expected_birth_mode: str
    previous_owner: ReleaseRowOwner | None
    transported_parent: ParticleState | None
    transported_parent_sha256: str | None


class V5H15GlobalRowCommitter(Protocol):
    def __call__(self, request: V5H15GlobalRowCommitRequest, /) -> object:
        ...


@dataclass(frozen=True, slots=True)
class V5H15RowCommitEnvelope:
    """Exact row-owner commit plus source-event Kelvin evidence."""

    commit_result: RowCommitResult
    source_kelvin_evidence: V5H15SourceKelvinEvidence


@dataclass(frozen=True, slots=True)
class V5H15SourceKelvinEvidence:
    """Source-event Kelvin residual bound to the exact committed row owner."""

    source_step_index: int
    row_owner_sha256: str
    source_event_sha256: str
    kelvin_ledger_sha256: str
    residual_m2_s: float
    atol_m2_s: float
    evidence_sha256: str


def _kelvin_payload(value: V5H15SourceKelvinEvidence) -> dict[str, object]:
    return {
        "atol_m2_s": float(value.atol_m2_s).hex(),
        "kelvin_ledger_sha256": value.kelvin_ledger_sha256,
        "residual_m2_s": float(value.residual_m2_s).hex(),
        "row_owner_sha256": value.row_owner_sha256,
        "source_event_sha256": value.source_event_sha256,
        "source_step_index": value.source_step_index,
    }


def make_v5h15_source_kelvin_evidence(
    *,
    source_step_index: int,
    row_owner_sha256: str,
    source_event_sha256: str,
    kelvin_ledger_sha256: str,
    residual_m2_s: float,
    atol_m2_s: float = MAX_SOURCE_KELVIN_M2_PER_S,
) -> V5H15SourceKelvinEvidence:
    _assert_execution_bindings()
    step = _exact_int("source_step_index", source_step_index)
    row_sha = _validate_sha256("row_owner_sha256", row_owner_sha256)
    event_sha = _validate_sha256("source_event_sha256", source_event_sha256)
    ledger_sha = _validate_sha256("kelvin_ledger_sha256", kelvin_ledger_sha256)
    residual = _finite_float("residual_m2_s", residual_m2_s)
    atol = _finite_float("atol_m2_s", atol_m2_s, positive=True)
    if atol != MAX_SOURCE_KELVIN_M2_PER_S:
        raise ValueError("source Kelvin gate is frozen at 1e-10 m2/s")
    if abs(residual) > atol:
        raise RuntimeError("source Kelvin residual exceeded the B3 gate")
    draft = V5H15SourceKelvinEvidence(
        source_step_index=step,
        row_owner_sha256=row_sha,
        source_event_sha256=event_sha,
        kelvin_ledger_sha256=ledger_sha,
        residual_m2_s=residual,
        atol_m2_s=atol,
        evidence_sha256="",
    )
    return replace(
        draft,
        evidence_sha256=_hash_payload(
            "fluxv-v5h15-source-kelvin-v1", _kelvin_payload(draft)
        ),
    )


def validate_v5h15_source_kelvin_evidence(
    value: V5H15SourceKelvinEvidence,
) -> V5H15SourceKelvinEvidence:
    _assert_execution_bindings()
    if type(value) is not V5H15SourceKelvinEvidence:
        raise ValueError("source Kelvin evidence has a foreign type")
    _validate_sha256("row_owner_sha256", value.row_owner_sha256)
    _validate_sha256("source_event_sha256", value.source_event_sha256)
    _validate_sha256("kelvin_ledger_sha256", value.kelvin_ledger_sha256)
    if (
        type(value.source_step_index) is not int
        or value.source_step_index < 0
        or type(value.residual_m2_s) is not float
        or not math.isfinite(value.residual_m2_s)
        or value.atol_m2_s != MAX_SOURCE_KELVIN_M2_PER_S
        or abs(value.residual_m2_s) > value.atol_m2_s
        or value.evidence_sha256
        != _hash_payload("fluxv-v5h15-source-kelvin-v1", _kelvin_payload(value))
    ):
        raise ValueError("source Kelvin evidence is invalid")
    return value


@dataclass(frozen=True, slots=True)
class V5H15LayerLoadLedger:
    """Detached native-Ptera load snapshot for one solved active layer."""

    ptera_step_index: int
    parent_state_sha256: str
    reference_chord_m: float
    panel_ids: tuple[str, ...]
    forces_w: FloatArray
    force_coefficients_w: FloatArray
    moments_w_cgp1: FloatArray
    moment_coefficients_w: FloatArray
    panel_forces_w: FloatArray
    panel_moments_w_cgp1: FloatArray
    panel_force_sum_w: FloatArray
    panel_moment_sum_w_cgp1: FloatArray
    panel_force_sum_max_abs_residual: float
    panel_moment_sum_max_abs_residual: float
    panel_force_sum_atol: float
    panel_moment_sum_atol: float
    no_penetration_residual: FloatArray
    no_penetration_max_abs: float
    no_penetration_atol: float
    collocation_evaluation_count: int
    load_batch_evaluation_count: int
    native_load_call_count: int
    surface_load_owner: str
    ledger_sha256: str


def _load_ledger_payload(ledger: V5H15LayerLoadLedger) -> dict[str, object]:
    return {
        "collocation_evaluation_count": ledger.collocation_evaluation_count,
        "force_coefficients_w_sha256": _array_sha256(ledger.force_coefficients_w),
        "forces_w_sha256": _array_sha256(ledger.forces_w),
        "load_batch_evaluation_count": ledger.load_batch_evaluation_count,
        "moment_coefficients_w_sha256": _array_sha256(ledger.moment_coefficients_w),
        "moments_w_cgp1_sha256": _array_sha256(ledger.moments_w_cgp1),
        "native_load_call_count": ledger.native_load_call_count,
        "no_penetration_atol": float(ledger.no_penetration_atol).hex(),
        "no_penetration_max_abs": float(ledger.no_penetration_max_abs).hex(),
        "no_penetration_residual_sha256": _array_sha256(ledger.no_penetration_residual),
        "panel_force_sum_atol": float(ledger.panel_force_sum_atol).hex(),
        "panel_force_sum_max_abs_residual": float(
            ledger.panel_force_sum_max_abs_residual
        ).hex(),
        "panel_force_sum_w_sha256": _array_sha256(ledger.panel_force_sum_w),
        "panel_forces_w_sha256": _array_sha256(ledger.panel_forces_w),
        "panel_ids": list(ledger.panel_ids),
        "panel_moment_sum_atol": float(ledger.panel_moment_sum_atol).hex(),
        "panel_moment_sum_max_abs_residual": float(
            ledger.panel_moment_sum_max_abs_residual
        ).hex(),
        "panel_moment_sum_w_cgp1_sha256": _array_sha256(ledger.panel_moment_sum_w_cgp1),
        "panel_moments_w_cgp1_sha256": _array_sha256(ledger.panel_moments_w_cgp1),
        "parent_state_sha256": ledger.parent_state_sha256,
        "ptera_step_index": ledger.ptera_step_index,
        "reference_chord_m": float(ledger.reference_chord_m).hex(),
        "surface_load_owner": ledger.surface_load_owner,
    }


def make_v5h15_layer_load_ledger(
    *,
    ptera_step_index: int,
    parent_state_sha256: str,
    reference_chord_m: float,
    panel_ids: tuple[str, ...],
    forces_w: ArrayLike,
    force_coefficients_w: ArrayLike,
    moments_w_cgp1: ArrayLike,
    moment_coefficients_w: ArrayLike,
    panel_forces_w: ArrayLike,
    panel_moments_w_cgp1: ArrayLike,
    no_penetration_residual: ArrayLike,
    no_penetration_atol: float = NO_PENETRATION_ATOL,
    collocation_evaluation_count: int = 1,
    load_batch_evaluation_count: int = 4,
    native_load_call_count: int = 1,
) -> V5H15LayerLoadLedger:
    """Build and independently summation-check a detached Ptera load ledger."""

    _assert_execution_bindings()
    step = _exact_int("ptera_step_index", ptera_step_index)
    parent_sha = _validate_sha256("parent_state_sha256", parent_state_sha256)
    chord = _finite_float("reference_chord_m", reference_chord_m, positive=True)
    if type(panel_ids) is not tuple or not panel_ids:
        raise ValueError("panel_ids must be a non-empty exact tuple")
    if any(type(item) is not str or not item for item in panel_ids):
        raise ValueError("panel_ids must contain non-empty exact strings")
    if len(set(panel_ids)) != len(panel_ids):
        raise ValueError("panel_ids must be unique")
    forces = _frozen_float64("forces_w", forces_w, (), cap=3)
    force_coefficients = _frozen_float64(
        "force_coefficients_w", force_coefficients_w, (), cap=3
    )
    moments = _frozen_float64("moments_w_cgp1", moments_w_cgp1, (), cap=3)
    moment_coefficients = _frozen_float64(
        "moment_coefficients_w", moment_coefficients_w, (), cap=3
    )
    if any(
        array.shape != (3,)
        for array in (forces, force_coefficients, moments, moment_coefficients)
    ):
        raise ValueError("airplane force/moment vectors must have shape (3,)")
    panel_forces = _frozen_float64(
        "panel_forces_w", panel_forces_w, (3,), cap=DEFAULT_MAX_TARGET_COUNT
    )
    panel_moments = _frozen_float64(
        "panel_moments_w_cgp1",
        panel_moments_w_cgp1,
        (3,),
        cap=DEFAULT_MAX_TARGET_COUNT,
    )
    if (
        panel_forces.shape != (len(panel_ids), 3)
        or panel_moments.shape != panel_forces.shape
    ):
        raise ValueError("panel load arrays must match panel_ids with shape (P,3)")
    force_sum = _frozen_float64(
        "panel_force_sum_w", np.sum(panel_forces, axis=0), (), cap=3
    )
    moment_sum = _frozen_float64(
        "panel_moment_sum_w_cgp1", np.sum(panel_moments, axis=0), (), cap=3
    )
    force_residual = float(np.max(np.abs(force_sum - forces), initial=0.0))
    moment_residual = float(np.max(np.abs(moment_sum - moments), initial=0.0))
    eps = np.finfo(np.float64).eps
    force_atol = float(64.0 * eps * max(1.0, float(np.sum(np.abs(panel_forces)))))
    moment_atol = float(64.0 * eps * max(1.0, float(np.sum(np.abs(panel_moments)))))
    if force_residual > force_atol:
        raise RuntimeError("native Ptera panel forces do not sum to airplane total")
    if moment_residual > moment_atol:
        raise RuntimeError("native Ptera panel moments do not sum to airplane total")
    no_penetration = _frozen_float64(
        "no_penetration_residual",
        no_penetration_residual,
        (),
        cap=DEFAULT_MAX_TARGET_COUNT,
    )
    if no_penetration.shape != (len(panel_ids),):
        raise ValueError("no-penetration residual must have one value per panel")
    no_penetration_max = float(np.max(np.abs(no_penetration), initial=0.0))
    no_penetration_gate = _finite_float(
        "no_penetration_atol", no_penetration_atol, positive=True
    )
    if no_penetration_gate > NO_PENETRATION_ATOL:
        raise ValueError("no-penetration gate cannot be relaxed")
    if no_penetration_max > no_penetration_gate:
        raise RuntimeError("native Ptera no-penetration residual exceeded the gate")
    counts = (
        _exact_int("collocation_evaluation_count", collocation_evaluation_count),
        _exact_int("load_batch_evaluation_count", load_batch_evaluation_count),
        _exact_int("native_load_call_count", native_load_call_count),
    )
    if counts != (1, 4, 1):
        raise RuntimeError("native Ptera collocation/load ledger is not exactly 1/4/1")
    draft = V5H15LayerLoadLedger(
        ptera_step_index=step,
        parent_state_sha256=parent_sha,
        reference_chord_m=chord,
        panel_ids=panel_ids,
        forces_w=forces,
        force_coefficients_w=force_coefficients,
        moments_w_cgp1=moments,
        moment_coefficients_w=moment_coefficients,
        panel_forces_w=panel_forces,
        panel_moments_w_cgp1=panel_moments,
        panel_force_sum_w=force_sum,
        panel_moment_sum_w_cgp1=moment_sum,
        panel_force_sum_max_abs_residual=force_residual,
        panel_moment_sum_max_abs_residual=moment_residual,
        panel_force_sum_atol=force_atol,
        panel_moment_sum_atol=moment_atol,
        no_penetration_residual=no_penetration,
        no_penetration_max_abs=no_penetration_max,
        no_penetration_atol=no_penetration_gate,
        collocation_evaluation_count=counts[0],
        load_batch_evaluation_count=counts[1],
        native_load_call_count=counts[2],
        surface_load_owner=SURFACE_FORCE_OWNER,
        ledger_sha256="",
    )
    return replace(
        draft,
        ledger_sha256=_hash_payload(
            "fluxv-v5h15-load-ledger-v1", _load_ledger_payload(draft)
        ),
    )


def validate_v5h15_layer_load_ledger(
    ledger: V5H15LayerLoadLedger,
) -> V5H15LayerLoadLedger:
    _assert_execution_bindings()
    if type(ledger) is not V5H15LayerLoadLedger:
        raise ValueError("load ledger must have the exact frozen type")
    _validate_sha256("load ledger parent", ledger.parent_state_sha256)
    arrays = (
        ("forces_w", ledger.forces_w, (3,)),
        ("force_coefficients_w", ledger.force_coefficients_w, (3,)),
        ("moments_w_cgp1", ledger.moments_w_cgp1, (3,)),
        ("moment_coefficients_w", ledger.moment_coefficients_w, (3,)),
        ("panel_forces_w", ledger.panel_forces_w, (len(ledger.panel_ids), 3)),
        (
            "panel_moments_w_cgp1",
            ledger.panel_moments_w_cgp1,
            (len(ledger.panel_ids), 3),
        ),
        ("panel_force_sum_w", ledger.panel_force_sum_w, (3,)),
        ("panel_moment_sum_w_cgp1", ledger.panel_moment_sum_w_cgp1, (3,)),
        (
            "no_penetration_residual",
            ledger.no_penetration_residual,
            (len(ledger.panel_ids),),
        ),
    )
    for name, array, shape in arrays:
        _require_frozen_float64(name, array, shape)
    if (
        type(ledger.panel_ids) is not tuple
        or not ledger.panel_ids
        or len(set(ledger.panel_ids)) != len(ledger.panel_ids)
        or any(type(item) is not str or not item for item in ledger.panel_ids)
    ):
        raise ValueError("load ledger panel ID tree is invalid")
    expected_force_sum = np.sum(ledger.panel_forces_w, axis=0)
    expected_moment_sum = np.sum(ledger.panel_moments_w_cgp1, axis=0)
    if not np.array_equal(expected_force_sum, ledger.panel_force_sum_w):
        raise ValueError("load ledger force sum was tampered")
    if not np.array_equal(expected_moment_sum, ledger.panel_moment_sum_w_cgp1):
        raise ValueError("load ledger moment sum was tampered")
    force_residual = float(
        np.max(np.abs(expected_force_sum - ledger.forces_w), initial=0.0)
    )
    moment_residual = float(
        np.max(np.abs(expected_moment_sum - ledger.moments_w_cgp1), initial=0.0)
    )
    eps = np.finfo(np.float64).eps
    force_atol = float(
        64.0 * eps * max(1.0, float(np.sum(np.abs(ledger.panel_forces_w))))
    )
    moment_atol = float(
        64.0 * eps * max(1.0, float(np.sum(np.abs(ledger.panel_moments_w_cgp1))))
    )
    if (
        ledger.panel_force_sum_max_abs_residual != force_residual
        or ledger.panel_moment_sum_max_abs_residual != moment_residual
        or ledger.panel_force_sum_atol != force_atol
        or ledger.panel_moment_sum_atol != moment_atol
        or force_residual > force_atol
        or moment_residual > moment_atol
    ):
        raise ValueError("load ledger sum gate was tampered or failed")
    no_penetration_max = float(
        np.max(np.abs(ledger.no_penetration_residual), initial=0.0)
    )
    if (
        ledger.no_penetration_max_abs != no_penetration_max
        or ledger.no_penetration_atol > NO_PENETRATION_ATOL
        or no_penetration_max > ledger.no_penetration_atol
    ):
        raise ValueError("load ledger no-penetration gate was tampered or failed")
    if (
        ledger.collocation_evaluation_count,
        ledger.load_batch_evaluation_count,
        ledger.native_load_call_count,
    ) != (1, 4, 1):
        raise ValueError("load ledger native counts are not 1/4/1")
    if ledger.surface_load_owner != SURFACE_FORCE_OWNER:
        raise ValueError("load ledger surface owner drift")
    expected = _hash_payload("fluxv-v5h15-load-ledger-v1", _load_ledger_payload(ledger))
    if ledger.ledger_sha256 != expected:
        raise ValueError("load ledger digest mismatch")
    return ledger


@dataclass(frozen=True, slots=True)
class V5H15StabilityEnvelope:
    stage_count: int
    max_h_jacobian_frobenius: float
    max_h_convective_over_sigma: float
    jacobian_gate: float
    convective_gate: float
    passed: bool
    envelope_sha256: str


@dataclass(frozen=True, slots=True)
class V5H15SupportEnvelope:
    common_transport_sha256: str
    transport_attestation_sha256: str
    transported_live_sigma_attestation_sha256: str
    support_count: int
    frontier_count: int
    transported_support_sha256: str
    selected_particle_position_sha256: str
    exact_support_match: bool
    final_tracer_sha256: str
    envelope_sha256: str


@dataclass(frozen=True, slots=True)
class V5H15LayerCounters:
    direct_field_call_count: int
    ptera_center_call_count: int
    ptera_offset_call_count: int
    transport_stage_count: int
    invariant_reference_freeze_count: int
    physical_field_call_count: int
    tracer_field_call_count: int
    stage_pre_reconstruction_count: int
    stage_post_reconstruction_count: int
    physical_rhs_call_count: int
    sigma_storage_update_count: int
    relaxation_call_count: int
    native_collocation_evaluation_count: int
    native_load_batch_evaluation_count: int
    native_load_call_count: int


@dataclass(frozen=True, slots=True)
class V5H15LayerResult:
    interface_id: str
    case_id: str
    scope: str
    substep_role: str
    transport_substeps: int
    ptera_step_index: int
    source_step_index: int
    source_time_s: float
    row_owner_before_sha256: str
    row_state_before_sha256: str
    common_transport_sha256: str
    transport_attestation_sha256: str
    transport_parent_digest: str
    advanced_owner: ReleaseRowOwner
    advanced_owner_sha256: str
    advanced_state_sha256: str
    stream_result: IRWRK3StreamResult
    stream_stage_chain_sha256: str
    stream_result_sha256: str
    fd_call_ledger: FDCallLedger
    fd_ledger_sha256: str
    load_ledger: V5H15LayerLoadLedger
    load_ledger_sha256: str
    source_kelvin_evidence: V5H15SourceKelvinEvidence
    source_kelvin_evidence_sha256: str
    stability_envelope: V5H15StabilityEnvelope
    support_envelope: V5H15SupportEnvelope
    counters: V5H15LayerCounters
    parent_token: str
    ptera_parent_sha256_before_transport: str
    ptera_parent_sha256_after_transport: str
    ptera_parent_state_unchanged: bool
    observation_access: str
    force_scoring_status: str
    result_sha256: str


def _stability_payload(value: V5H15StabilityEnvelope) -> dict[str, object]:
    return {
        "convective_gate": float(value.convective_gate).hex(),
        "jacobian_gate": float(value.jacobian_gate).hex(),
        "max_h_convective_over_sigma": float(value.max_h_convective_over_sigma).hex(),
        "max_h_jacobian_frobenius": float(value.max_h_jacobian_frobenius).hex(),
        "passed": value.passed,
        "stage_count": value.stage_count,
    }


def _support_payload(value: V5H15SupportEnvelope) -> dict[str, object]:
    return {
        field.name: getattr(value, field.name)
        for field in fields(value)
        if field.name != "envelope_sha256"
    }


def _counter_tuple(value: V5H15LayerCounters) -> tuple[int, ...]:
    return tuple(getattr(value, field.name) for field in fields(value))


def _layer_payload(result: V5H15LayerResult) -> dict[str, object]:
    return {
        "advanced_owner_sha256": result.advanced_owner_sha256,
        "advanced_state_sha256": result.advanced_state_sha256,
        "case_id": result.case_id,
        "common_transport_sha256": result.common_transport_sha256,
        "counters": _counter_tuple(result.counters),
        "fd_ledger_sha256": result.fd_ledger_sha256,
        "force_scoring_status": result.force_scoring_status,
        "interface_id": result.interface_id,
        "load_ledger_sha256": result.load_ledger_sha256,
        "source_kelvin_evidence_sha256": result.source_kelvin_evidence_sha256,
        "observation_access": result.observation_access,
        "parent_token": result.parent_token,
        "ptera_parent_sha256_after_transport": result.ptera_parent_sha256_after_transport,
        "ptera_parent_sha256_before_transport": result.ptera_parent_sha256_before_transport,
        "ptera_parent_state_unchanged": result.ptera_parent_state_unchanged,
        "ptera_step_index": result.ptera_step_index,
        "result_scope": result.scope,
        "row_owner_before_sha256": result.row_owner_before_sha256,
        "row_state_before_sha256": result.row_state_before_sha256,
        "source_step_index": result.source_step_index,
        "source_time_s": float(result.source_time_s).hex(),
        "stability_envelope_sha256": result.stability_envelope.envelope_sha256,
        "stream_result_sha256": result.stream_result_sha256,
        "stream_stage_chain_sha256": result.stream_stage_chain_sha256,
        "substep_role": result.substep_role,
        "support_envelope_sha256": result.support_envelope.envelope_sha256,
        "transport_attestation_sha256": result.transport_attestation_sha256,
        "transport_parent_digest": result.transport_parent_digest,
        "transport_substeps": result.transport_substeps,
    }


class _StageFieldComposer:
    """Compose direct self and audited parent fields without retaining arrays."""

    __slots__ = (
        "adapter",
        "convective_gate",
        "direct_call_count",
        "galilean_velocity",
        "jacobian_gate",
        "latest_physical",
        "latest_tracer",
        "max_convective",
        "max_jacobian",
        "observer_count",
        "parent_token",
        "substep_delta_time",
    )

    def __init__(
        self,
        adapter: FrozenParentCenteredFDAdapter,
        *,
        parent_token: str,
        galilean_velocity: FloatArray,
        substep_delta_time: float,
        jacobian_gate: float,
        convective_gate: float,
    ) -> None:
        self.adapter = adapter
        self.parent_token = parent_token
        self.galilean_velocity = galilean_velocity
        self.substep_delta_time = substep_delta_time
        self.jacobian_gate = jacobian_gate
        self.convective_gate = convective_gate
        self.direct_call_count = 0
        self.observer_count = 0
        self.max_jacobian = 0.0
        self.max_convective = 0.0
        self.latest_physical: FDEvaluationRecord | None = None
        self.latest_tracer: FDEvaluationRecord | None = None

    def physical_field(self, state: ParticleState) -> IRWRK3Field:
        direct = direct_gaussian_erf_velocity_jacobian(
            state.positions, state.gamma, state.sigma
        )
        self.direct_call_count += 1
        parent = self.adapter.evaluate_physical(state)
        self.latest_physical = parent.evaluation
        return make_ir_wrk3_field(
            state,
            np.asarray(direct.velocity) + np.asarray(parent.field.velocity),
            np.asarray(direct.jacobian) + np.asarray(parent.field.jacobian),
            parent_token=self.parent_token,
        )

    def tracer_field(
        self,
        state: ParticleState,
        tracer_positions: FloatArray,
        parent_token: str,
    ) -> IRWRK3TracerField:
        if parent_token != self.parent_token:
            raise RuntimeError("stage tracer parent token drift")
        direct = direct_gaussian_erf_velocity_jacobian(
            state.positions,
            state.gamma,
            state.sigma,
            target_positions=tracer_positions,
        )
        self.direct_call_count += 1
        parent = self.adapter.evaluate_tracer(state, tracer_positions, parent_token)
        self.latest_tracer = parent.evaluation
        return make_ir_wrk3_tracer_field(
            state,
            tracer_positions,
            np.asarray(direct.velocity) + np.asarray(parent.field.velocity),
            parent_token=self.parent_token,
        )

    def observe(self, view: IRWRK3StreamStageView):
        physical = self.latest_physical
        tracer = self.latest_tracer
        if physical is None or tracer is None:
            raise RuntimeError("stage observer lacks physical/tracer FD evidence")
        expected_physical = 2 * self.observer_count + 1
        expected_tracer = expected_physical + 1
        if (
            physical.evaluation_index != expected_physical
            or tracer.evaluation_index != expected_tracer
            or physical.source_state_sha256 != view.field.source_state_sha256
            or tracer.source_state_sha256 != view.field.source_state_sha256
        ):
            raise RuntimeError("stage FD evidence order/source binding mismatch")
        # V5H15 gate-accounting amendment: evaluate the stability gates with
        # the sub-step delta time actually integrated for this stage.
        actual_delta_time = view.substep_delta_time
        if type(actual_delta_time) is not float or not (
            math.isfinite(actual_delta_time) and actual_delta_time > 0.0
        ):
            raise RuntimeError("stage view sub-step delta time is invalid")
        h_j = actual_delta_time * float(
            np.max(np.linalg.norm(view.field.jacobian, axis=(1, 2)), initial=0.0)
        )
        relative_velocity = view.field.velocity - self.galilean_velocity[None, :]
        h_u_sigma = actual_delta_time * float(
            np.max(
                np.linalg.norm(relative_velocity, axis=1) / view.pre.sigma,
                initial=0.0,
            )
        )
        if not math.isfinite(h_j) or not math.isfinite(h_u_sigma):
            raise FloatingPointError("stage stability envelope is non-finite")
        if h_j > self.jacobian_gate:
            raise FloatingPointError("h*max||J_total||F exceeded the B3 gate")
        if h_u_sigma > self.convective_gate:
            raise FloatingPointError("h*max|U-U_gal|/sigma exceeded the B3 gate")
        self.max_jacobian = max(self.max_jacobian, h_j)
        self.max_convective = max(self.max_convective, h_u_sigma)
        self.observer_count += 1
        self.latest_physical = None
        self.latest_tracer = None
        payload = json.dumps(
            {
                "fd_physical_evaluation_sha256": physical.evaluation_sha256,
                "fd_tracer_evaluation_sha256": tracer.evaluation_sha256,
                "h_convective_over_sigma": h_u_sigma.hex(),
                "h_jacobian_frobenius": h_j.hex(),
                "source_state_sha256": view.field.source_state_sha256,
                "stage": view.stage,
                "substep": view.substep,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
        return make_ir_wrk3_stream_evidence("v5h15-stage-fd-stability-v1", payload)


def _make_stability_envelope(
    composer: _StageFieldComposer,
    expected_stage_count: int,
) -> V5H15StabilityEnvelope:
    if composer.observer_count != expected_stage_count:
        raise RuntimeError("stream observer count did not close")
    draft = V5H15StabilityEnvelope(
        stage_count=expected_stage_count,
        max_h_jacobian_frobenius=composer.max_jacobian,
        max_h_convective_over_sigma=composer.max_convective,
        jacobian_gate=composer.jacobian_gate,
        convective_gate=composer.convective_gate,
        passed=(
            composer.max_jacobian <= composer.jacobian_gate
            and composer.max_convective <= composer.convective_gate
        ),
        envelope_sha256="",
    )
    return replace(
        draft,
        envelope_sha256=_hash_payload(
            "fluxv-v5h15-stability-envelope-v1", _stability_payload(draft)
        ),
    )


def _make_support_envelope(
    common_transport_sha256: str,
    transport_attestation_sha256: str,
    transported_live_sigma_attestation_sha256: str,
    stream: IRWRK3StreamResult,
    flat_live_indices: tuple[int, ...],
    frontier_node_offset: int,
) -> V5H15SupportEnvelope:
    flat = np.asarray(flat_live_indices, dtype=np.int64)
    supports = stream.final_tracer_positions[:frontier_node_offset]
    selected_positions = stream.final_state.positions[flat]
    exact_positions = np.array_equal(supports, selected_positions)
    draft = V5H15SupportEnvelope(
        common_transport_sha256=common_transport_sha256,
        transport_attestation_sha256=transport_attestation_sha256,
        transported_live_sigma_attestation_sha256=(
            transported_live_sigma_attestation_sha256
        ),
        support_count=frontier_node_offset,
        frontier_count=stream.final_tracer_positions.shape[0] - frontier_node_offset,
        transported_support_sha256=_array_sha256(supports),
        selected_particle_position_sha256=_array_sha256(selected_positions),
        exact_support_match=bool(exact_positions),
        final_tracer_sha256=stream.final_tracer_sha256,
        envelope_sha256="",
    )
    if not draft.exact_support_match:
        raise RuntimeError("exact common material-support envelope failed")
    return replace(
        draft,
        envelope_sha256=_hash_payload(
            "fluxv-v5h15-support-envelope-v1", _support_payload(draft)
        ),
    )


def _expected_layer_counters(
    substeps: int,
    stream: IRWRK3StreamResult,
    ledger: FDCallLedger,
    load: V5H15LayerLoadLedger,
    direct_call_count: int,
) -> V5H15LayerCounters:
    stage_count = 3 * substeps
    expected = V5H15LayerCounters(
        direct_field_call_count=direct_call_count,
        ptera_center_call_count=ledger.center_call_count,
        ptera_offset_call_count=ledger.offset_call_count,
        transport_stage_count=stream.counters.stage_count,
        invariant_reference_freeze_count=stream.counters.invariant_reference_freeze_count,
        physical_field_call_count=stream.counters.physical_field_call_count,
        tracer_field_call_count=stream.counters.tracer_field_call_count,
        stage_pre_reconstruction_count=stream.counters.stage_pre_reconstruction_count,
        stage_post_reconstruction_count=stream.counters.stage_post_reconstruction_count,
        physical_rhs_call_count=stream.counters.physical_rhs_call_count,
        sigma_storage_update_count=stream.counters.sigma_storage_update_count,
        relaxation_call_count=stream.counters.relaxation_call_count,
        native_collocation_evaluation_count=load.collocation_evaluation_count,
        native_load_batch_evaluation_count=load.load_batch_evaluation_count,
        native_load_call_count=load.native_load_call_count,
    )
    if _counter_tuple(expected) != (
        6 * substeps,
        6 * substeps,
        18 * substeps,
        stage_count,
        1,
        stage_count,
        stage_count,
        stage_count,
        stage_count,
        stage_count,
        0,
        0,
        1,
        4,
        1,
    ):
        raise RuntimeError("v5h15 layer call ledger did not close exactly")
    return expected


def _make_live_layer_result_registry(
    transport_engine: Callable[..., V5H15LayerResult],
    *,
    max_count: int = MAX_LIVE_LAYER_RESULTS,
    hash_constructor: Callable[[bytes], object] = sha256,
) -> tuple[Callable[..., object], ...]:
    """Create one closed transport-and-issue path over a sealed registry."""

    lock = RLock()
    registry: dict[int, tuple[V5H15LayerResult, str]] = {}
    semantic_registry: dict[tuple[str, str, str], tuple[str, V5H15LayerResult]] = {}
    reservations: dict[int, object] = {}
    issuance_counter = 0

    def compute_seal() -> str:
        payload = (
            issuance_counter,
            tuple(sorted((key, id(value)) for key, value in reservations.items())),
            tuple(
                (key, id(entry[0]), entry[1], entry[0].result_sha256)
                for key, entry in sorted(registry.items())
            ),
            tuple(
                (key, entry[0], id(entry[1]))
                for key, entry in sorted(semantic_registry.items())
            ),
        )
        digest = hash_constructor(repr(payload).encode("ascii"))
        return digest.hexdigest()  # type: ignore[no-any-return, union-attr]

    seal = compute_seal()

    def result_semantic_key(result: V5H15LayerResult) -> tuple[str, str, str]:
        event = result.advanced_owner.transport_events[-1]
        return (
            result.advanced_owner_sha256,
            event.transport_event_sha256,
            result.transport_parent_digest,
        )

    def verify_semantic_index() -> None:
        if len(registry) != len(semantic_registry):
            raise RuntimeError("v5h15 semantic registry cardinality drift")
        seen_result_digests: set[str] = set()
        for result, result_digest in registry.values():
            semantic_key = result_semantic_key(result)
            semantic_entry = semantic_registry.get(semantic_key)
            if (
                result_digest in seen_result_digests
                or semantic_entry is None
                or semantic_entry[0] != result_digest
                or semantic_entry[1] is not result
            ):
                raise RuntimeError("v5h15 semantic registry bijection drift")
            seen_result_digests.add(result_digest)

    def verify_seal() -> None:
        if compute_seal() != seal:
            raise RuntimeError("v5h15 live layer-result registry integrity drift")
        verify_semantic_index()

    def execute(
        owner: ReleaseRowOwner,
        *,
        config: V5H15BaikCouplingConfig,
        ptera_step_index: int,
        source_step_index: int,
        load_ledger: V5H15LayerLoadLedger,
        source_kelvin_evidence: V5H15SourceKelvinEvidence,
        parent_velocity_evaluator: FrozenParentVelocityEvaluator,
        parent_state_sha256_getter: ParentStateSHA256Getter,
        parent_token: str,
        galilean_velocity_gp1_m_per_s: ArrayLike,
    ) -> V5H15LayerResult:
        nonlocal issuance_counter, seal
        verify_runtime = _assert_execution_bindings
        verify_runtime_code = verify_runtime.__code__
        verify_runtime_defaults = verify_runtime.__defaults__
        verify_registry = _assert_registry_function_bindings
        validate_tree = _validate_v5h15_layer_result_tree
        validate_tree_code = validate_tree.__code__
        validate_tree_defaults = validate_tree.__defaults__
        verify_runtime()
        verify_registry()
        with lock:
            verify_seal()
            if len(registry) + len(reservations) >= max_count:
                raise RuntimeError("v5h15 live layer-result registry cap exceeded")
            reservation = object()
            reservations[id(reservation)] = reservation
            seal = compute_seal()
            registry_snapshot = snapshot()

        def verify_all() -> None:
            if (
                globals().get("_assert_execution_bindings") is not verify_runtime
                or verify_runtime.__code__ is not verify_runtime_code
                or verify_runtime.__defaults__ is not verify_runtime_defaults
            ):
                raise RuntimeError("v5h15 trusted runtime verifier drift")
            verify_runtime()
            verify_registry()
            if (
                globals().get("_validate_v5h15_layer_result_tree") is not validate_tree
                or validate_tree.__code__ is not validate_tree_code
                or validate_tree.__defaults__ is not validate_tree_defaults
            ):
                raise RuntimeError("layer result-tree validator drift")
            assert_snapshot_unchanged(registry_snapshot)

        try:
            result = transport_engine(
                owner,
                config=config,
                ptera_step_index=ptera_step_index,
                source_step_index=source_step_index,
                load_ledger=load_ledger,
                source_kelvin_evidence=source_kelvin_evidence,
                parent_velocity_evaluator=parent_velocity_evaluator,
                parent_state_sha256_getter=parent_state_sha256_getter,
                parent_token=parent_token,
                galilean_velocity_gp1_m_per_s=galilean_velocity_gp1_m_per_s,
                verify_all=verify_all,
            )
            verify_all()
            if validate_tree(result) is not result:
                raise RuntimeError(
                    "layer result-tree validator changed issued identity"
                )
            semantic_key = result_semantic_key(result)
        except BaseException:
            with lock:
                verify_seal()
                if reservations.get(id(reservation)) is not reservation:
                    raise RuntimeError(
                        "failed layer transaction lost its reserved slot"
                    )
                reservations.pop(id(reservation))
                seal = compute_seal()
            raise
        with lock:
            verify_seal()
            if reservations.get(id(reservation)) is not reservation:
                raise RuntimeError("completed layer transaction lost its reserved slot")
            key = id(result)
            if key in registry:
                raise RuntimeError("v5h15 layer-result identity collision")
            if semantic_key in semantic_registry or any(
                entry[0] == result.result_sha256 for entry in semantic_registry.values()
            ):
                raise RuntimeError("duplicate v5h15 layer-result semantic issuance")
            reservations.pop(id(reservation))
            registry[key] = (result, result.result_sha256)
            semantic_registry[semantic_key] = (result.result_sha256, result)
            issuance_counter += 1
            seal = compute_seal()
        return result

    def attest(result: V5H15LayerResult) -> None:
        with lock:
            verify_seal()
            entry = registry.get(id(result))
            if (
                entry is None
                or entry[0] is not result
                or entry[1] != result.result_sha256
            ):
                raise ValueError("layer result is not the exact live issued report")

    def snapshot() -> tuple[object, ...]:
        with lock:
            verify_seal()
            return (
                issuance_counter,
                seal,
                tuple((key, value) for key, value in sorted(reservations.items())),
                tuple(
                    (key, entry[0], entry[1]) for key, entry in sorted(registry.items())
                ),
                tuple(
                    (key, entry[0], id(entry[1]))
                    for key, entry in sorted(semantic_registry.items())
                ),
            )

    def assert_snapshot_unchanged(expected: tuple[object, ...]) -> None:
        with lock:
            verify_seal()
            observed = snapshot()
            if type(expected) is not tuple or len(expected) != 5:
                raise RuntimeError("invalid v5h15 registry snapshot")
            if observed[0] != expected[0] or observed[1] != expected[1]:
                raise RuntimeError("v5h15 registry changed during callback")
            for observed_items, expected_items in zip(
                observed[2:4], expected[2:4], strict=True
            ):
                if (
                    type(observed_items) is not tuple
                    or type(expected_items) is not tuple
                    or len(observed_items) != len(expected_items)
                ):
                    raise RuntimeError("v5h15 registry changed during callback")
                for observed_item, expected_item in zip(
                    observed_items, expected_items, strict=True
                ):
                    if (
                        observed_item[0] != expected_item[0]
                        or observed_item[1] is not expected_item[1]
                        or (
                            len(observed_item) == 3
                            and observed_item[2] != expected_item[2]
                        )
                    ):
                        raise RuntimeError("v5h15 registry changed during callback")
            if observed[4] != expected[4]:
                raise RuntimeError("v5h15 semantic registry changed during callback")

    return execute, attest, snapshot, assert_snapshot_unchanged


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
    "_execute_layer_result_transaction",
    "_attest_live_layer_result",
    "_snapshot_live_layer_result_registry",
    "_assert_live_layer_result_registry_unchanged",
)


def _freeze_registry_function(name: str) -> _RegistryFunctionBinding:
    function = globals()[name]
    if not inspect.isfunction(function):
        raise RuntimeError(f"layer registry binding is not a function: {name}")
    closure: list[_RegistryClosureBinding] = []
    cells = function.__closure__ or ()
    for freevar, cell in zip(function.__code__.co_freevars, cells, strict=True):
        value = cell.cell_contents
        if freevar in ("registry", "reservations", "semantic_registry", "lock"):
            role = "fixed"
        elif freevar == "issuance_counter":
            role = "counter"
        elif freevar == "seal":
            role = "seal"
        elif inspect.isfunction(value):
            role = "function"
        else:
            role = "fixed"
        closure.append(
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
        closure=tuple(closure),
    )


def _assert_registry_function_bindings(
    frozen: tuple[_RegistryFunctionBinding, ...] | None = None,
) -> None:
    if frozen is None:
        candidate = globals().get("_FROZEN_REGISTRY_FUNCTION_BINDINGS")
        if type(candidate) is not tuple:
            raise RuntimeError("layer registry verifier is not initialized")
        frozen = candidate
    if globals().get("_FROZEN_REGISTRY_FUNCTION_BINDINGS") is not frozen:
        raise RuntimeError("layer registry verifier binding drift")
    for binding in frozen:
        current = globals().get(binding.name)
        if (
            current is not binding.function
            or current.__code__ is not binding.code
            or current.__defaults__ is not binding.defaults
            or current.__kwdefaults__ is not binding.kwdefaults
        ):
            raise RuntimeError(f"layer registry function drift: {binding.name}")
        cells = current.__closure__ or ()
        if len(cells) != len(binding.closure):
            raise RuntimeError(f"layer registry closure drift: {binding.name}")
        for cell, expected in zip(cells, binding.closure, strict=True):
            if cell is not expected.cell:
                raise RuntimeError(f"layer registry closure cell drift: {binding.name}")
            value = cell.cell_contents
            if expected.role == "fixed":
                if value is not expected.value:
                    raise RuntimeError(
                        f"layer registry fixed closure drift: {binding.name}"
                    )
            elif expected.role == "counter":
                if type(value) is not int or value < 0:
                    raise RuntimeError("layer registry counter closure drift")
            elif expected.role == "seal":
                _validate_sha256("layer registry seal", value)
            elif expected.role == "function":
                if (
                    value is not expected.value
                    or value.__code__ is not expected.code
                    or value.__defaults__ is not expected.defaults
                    or value.__kwdefaults__ is not expected.kwdefaults
                ):
                    raise RuntimeError(
                        f"layer registry helper closure drift: {binding.name}"
                    )
            else:
                raise RuntimeError("unknown layer registry closure role")


_EXECUTION_BINDING_NAMES: Final[tuple[str, ...]] = (
    "np",
    "json",
    "math",
    "sha256",
    "fields",
    "replace",
    "ParticleState",
    "IRWRK3Field",
    "IRWRK3TracerField",
    "IRWRK3StreamResult",
    "FrozenParentCenteredFDAdapter",
    "V5H15BaikCouplingConfig",
    "V5H15LayerLoadLedger",
    "V5H15StabilityEnvelope",
    "V5H15SupportEnvelope",
    "V5H15LayerCounters",
    "V5H15LayerResult",
    "V5H15SourceKelvinEvidence",
    "V5H15RowCommitEnvelope",
    "V5H15NativeBaikCouplingSolver",
    "direct_gaussian_erf_velocity_jacobian",
    "make_ir_wrk3_field",
    "make_ir_wrk3_tracer_field",
    "ir_wrk3_stream_macro",
    "make_ir_wrk3_stream_evidence",
    "validate_ir_wrk3_stream_result",
    "validate_fd_call_ledger",
    "validate_current_release_row_owner",
    "validate_release_row_owner",
    "validate_release_row_common_transport",
    "validate_release_row_transport_attestation",
    "begin_release_row_common_transport",
    "attest_release_row_common_transport",
    "release_row_transport_digest",
    "advance_release_row_transport_parent",
    "validate_v5h15_layer_load_ledger",
    "make_v5h15_source_kelvin_evidence",
    "validate_v5h15_source_kelvin_evidence",
    "_frozen_float64",
    "_stream_state_sha256",
    "_hash_payload",
    "_kelvin_payload",
    "_counter_tuple",
    "_layer_payload",
    "_make_stability_envelope",
    "_make_support_envelope",
    "_expected_layer_counters",
    "_validate_v5h15_layer_result_tree",
    "_execution_literal",
    "_assert_registry_function_bindings",
    "_NativePteraParentHash",
    "_NativePteraParentVelocity",
    "_execute_layer_result_transaction",
    "transport_v5h15_committed_layer",
    "make_fluxv_v5h15_baik_w2_solver",
)


def _execution_literal() -> tuple[object, ...]:
    return (
        COUPLING_INTERFACE_ID,
        CASE_ID,
        OBSERVATION_ACCESS,
        SOURCE_LOAD_OWNER,
        SURFACE_FORCE_OWNER,
        FORCE_SCORING_STATUS,
        PTERA_CHORDWISE_PANELS,
        PTERA_SPANWISE_PANELS,
        PTERA_STEPS_PER_CYCLE,
        PTERA_CYCLES,
        PTERA_DELTA_TIME_S,
        ACTIVE_PTERA_STEPS,
        ACTIVE_SOURCE_STEPS,
        ACTIVE_BIRTH_MODES,
        FORMAL_TRANSPORT_SUBSTEPS,
        MAX_ACTIVE_LAYERS,
        NO_PENETRATION_ATOL,
        MAX_SOURCE_KELVIN_M2_PER_S,
        STABILITY_J_H_MAX,
        STABILITY_CONVECTIVE_H_MAX,
        MAX_TEST_SUBSTEPS,
        MAX_LIVE_LAYER_RESULTS,
        json.dumps,
        json.dumps.__code__,
        json.dumps.__defaults__,
        tuple(sorted((json.dumps.__kwdefaults__ or {}).items())),
        V5H15LayerResult.__new__,
        V5H15LayerResult.__init__,
        V5H15LayerResult.__init__.__code__,
        V5H15LayerResult.__init__.__defaults__,
        V5H15LayerResult.__init__.__kwdefaults__,
        tuple(V5H15LayerResult.__slots__),
        tuple(
            (name, getattr(V5H15LayerResult, name))
            for name in V5H15LayerResult.__slots__
        ),
        _counter_tuple.__code__,
        _counter_tuple.__defaults__,
        _counter_tuple.__kwdefaults__,
        _layer_payload.__code__,
        _layer_payload.__defaults__,
        _layer_payload.__kwdefaults__,
        _hash_payload.__code__,
        _hash_payload.__defaults__,
        _hash_payload.__kwdefaults__,
        validate_release_row_owner.__code__,
        validate_release_row_owner.__defaults__,
        validate_release_row_owner.__kwdefaults__,
    )


class _GuardedParentVelocity:
    __slots__ = ("callback", "verifier", "verifier_code", "verifier_defaults")

    def __init__(
        self, callback: FrozenParentVelocityEvaluator, verifier: Callable[[], None]
    ) -> None:
        self.callback = callback
        self.verifier = verifier
        self.verifier_code = verifier.__code__
        self.verifier_defaults = verifier.__defaults__

    def _verify(self) -> None:
        if (
            self.verifier.__code__ is not self.verifier_code
            or self.verifier.__defaults__ is not self.verifier_defaults
        ):
            raise RuntimeError("v5h15 trusted callback verifier drift")
        self.verifier()

    def __call__(self, targets: FloatArray) -> FrozenParentVelocity:
        self._verify()
        response = self.callback(targets)
        self._verify()
        return response


class _GuardedParentHash:
    __slots__ = ("callback", "verifier", "verifier_code", "verifier_defaults")

    def __init__(
        self, callback: ParentStateSHA256Getter, verifier: Callable[[], None]
    ) -> None:
        self.callback = callback
        self.verifier = verifier
        self.verifier_code = verifier.__code__
        self.verifier_defaults = verifier.__defaults__

    def _verify(self) -> None:
        if (
            self.verifier.__code__ is not self.verifier_code
            or self.verifier.__defaults__ is not self.verifier_defaults
        ):
            raise RuntimeError("v5h15 trusted callback verifier drift")
        self.verifier()

    def __call__(self) -> str:
        self._verify()
        response = self.callback()
        self._verify()
        return response


def _transport_v5h15_committed_layer_impl(
    owner: ReleaseRowOwner,
    *,
    config: V5H15BaikCouplingConfig,
    ptera_step_index: int,
    source_step_index: int,
    load_ledger: V5H15LayerLoadLedger,
    source_kelvin_evidence: V5H15SourceKelvinEvidence,
    parent_velocity_evaluator: FrozenParentVelocityEvaluator,
    parent_state_sha256_getter: ParentStateSHA256Getter,
    parent_token: str,
    galilean_velocity_gp1_m_per_s: ArrayLike,
    verify_all: Callable[[], None],
) -> V5H15LayerResult:
    """Transport one already committed row and atomically advance its owner."""

    verify_all()
    # Capture every operation that remains after the owner-consuming advance.
    # User callbacks run later in the stream, so this snapshot is re-attested
    # immediately before advance and only these locals are used afterwards.
    result_type = V5H15LayerResult
    result_new = result_type.__new__
    result_init = result_type.__init__
    result_init_code = result_init.__code__
    result_init_defaults = result_init.__defaults__
    result_init_kwdefaults = result_init.__kwdefaults__
    result_init_closure = tuple(
        (cell, cell.cell_contents) for cell in (result_init.__closure__ or ())
    )
    result_slots = tuple(result_type.__slots__)
    result_slot_descriptors = tuple(
        (name, getattr(result_type, name)) for name in result_slots
    )
    layer_payload = _layer_payload
    counter_tuple = _counter_tuple
    hash_payload = _hash_payload
    advanced_validator = validate_release_row_owner
    json_dumps = json.dumps
    fields_function = fields
    sha256_constructor = sha256
    post_advance_functions = (
        (
            "_layer_payload",
            layer_payload,
            layer_payload.__code__,
            layer_payload.__defaults__,
            layer_payload.__kwdefaults__,
        ),
        (
            "_counter_tuple",
            counter_tuple,
            counter_tuple.__code__,
            counter_tuple.__defaults__,
            counter_tuple.__kwdefaults__,
        ),
        (
            "_hash_payload",
            hash_payload,
            hash_payload.__code__,
            hash_payload.__defaults__,
            hash_payload.__kwdefaults__,
        ),
        (
            "validate_release_row_owner",
            advanced_validator,
            advanced_validator.__code__,
            advanced_validator.__defaults__,
            advanced_validator.__kwdefaults__,
        ),
    )
    auxiliary_functions = (
        (
            "json.dumps",
            json_dumps,
            json_dumps.__code__,
            json_dumps.__defaults__,
            json_dumps.__kwdefaults__,
        ),
        (
            "dataclasses.fields",
            fields_function,
            fields_function.__code__,
            fields_function.__defaults__,
            fields_function.__kwdefaults__,
        ),
    )

    def preattest_post_advance_dependencies() -> None:
        verify_all()
        for name, function, code, defaults, kwdefaults in post_advance_functions:
            if (
                globals().get(name) is not function
                or function.__code__ is not code
                or function.__defaults__ is not defaults
                or function.__kwdefaults__ is not kwdefaults
            ):
                raise RuntimeError(f"post-advance dependency drift: {name}")
        observed_auxiliary = (
            json.dumps,
            fields,
        )
        for observed, snapshot in zip(
            observed_auxiliary, auxiliary_functions, strict=True
        ):
            name, function, code, defaults, kwdefaults = snapshot
            if (
                observed is not function
                or function.__code__ is not code
                or function.__defaults__ is not defaults
                or function.__kwdefaults__ is not kwdefaults
            ):
                raise RuntimeError(f"post-advance auxiliary drift: {name}")
        if sha256 is not sha256_constructor:
            raise RuntimeError("post-advance SHA-256 constructor drift")
        if (
            V5H15LayerResult is not result_type
            or result_type.__new__ is not result_new
            or result_type.__init__ is not result_init
            or result_init.__code__ is not result_init_code
            or result_init.__defaults__ is not result_init_defaults
            or result_init.__kwdefaults__ is not result_init_kwdefaults
            or tuple(result_type.__slots__) != result_slots
        ):
            raise RuntimeError("post-advance LayerResult constructor drift")
        observed_closure = result_init.__closure__ or ()
        if len(observed_closure) != len(result_init_closure):
            raise RuntimeError("post-advance LayerResult constructor closure drift")
        for observed_cell, (expected_cell, expected_value) in zip(
            observed_closure, result_init_closure, strict=True
        ):
            if (
                observed_cell is not expected_cell
                or observed_cell.cell_contents is not expected_value
            ):
                raise RuntimeError("post-advance LayerResult constructor closure drift")
        if any(
            getattr(result_type, name) is not descriptor
            for name, descriptor in result_slot_descriptors
        ):
            raise RuntimeError("post-advance LayerResult slot descriptor drift")

    if type(config) is not V5H15BaikCouplingConfig:
        raise TypeError("config must be an exact V5H15BaikCouplingConfig")
    committed = validate_current_release_row_owner(owner)
    if committed is not owner or committed.state.phase != "post_commit_pre_transport":
        raise RuntimeError("transport requires the exact current committed row owner")
    if owner.release_dt_s != config.delta_time_s:
        raise ValueError("row-owner release cadence disagrees with coupling delta_time")
    load = validate_v5h15_layer_load_ledger(load_ledger)
    ptera_step = _exact_int("ptera_step_index", ptera_step_index)
    source_step = _exact_int("source_step_index", source_step_index)
    kelvin = validate_v5h15_source_kelvin_evidence(source_kelvin_evidence)
    if load.ptera_step_index != ptera_step:
        raise ValueError("load ledger belongs to another Ptera step")
    if (
        kelvin.source_step_index != source_step
        or kelvin.row_owner_sha256 != owner.owner_sha256
    ):
        raise ValueError("source Kelvin evidence is not bound to this row/source step")
    if type(parent_token) is not str or not parent_token:
        raise ValueError("parent_token must be a non-empty exact string")
    if not callable(parent_velocity_evaluator) or not callable(
        parent_state_sha256_getter
    ):
        raise TypeError("parent velocity/hash callbacks must be callable")
    guarded_velocity = _GuardedParentVelocity(parent_velocity_evaluator, verify_all)
    guarded_hash = _GuardedParentHash(parent_state_sha256_getter, verify_all)
    before = _validate_sha256("parent_state_sha256_getter result", guarded_hash())
    if before != load.parent_state_sha256:
        raise RuntimeError("load ledger and frozen parent state are not identical")
    galilean = _frozen_float64(
        "galilean_velocity_gp1_m_per_s",
        galilean_velocity_gp1_m_per_s,
        (),
        cap=3,
    )
    if galilean.shape != (3,):
        raise ValueError("galilean velocity must have shape (3,)")
    particle_count = owner.state.positions.shape[0]
    if particle_count == 0 or particle_count > config.particle_cap:
        raise ValueError("committed particle count is empty or outside the cap")
    state = ParticleState(owner.state.positions, owner.state.gamma, owner.state.sigma)
    if _stream_state_sha256(state) == "":
        raise RuntimeError("unreachable state digest failure")

    flat_live_indices = tuple(
        index
        for indices in owner.state.live_boundary_indices_by_cell
        for index in indices
    )
    flat_live_array = np.asarray(flat_live_indices, dtype=np.int64)
    frontier_node_offset = len(flat_live_indices)
    material_tracer_positions = _frozen_float64(
        "preissued common material tracer pack",
        np.vstack(
            (
                owner.state.positions[flat_live_array],
                owner.state.live_boundary_nodes,
            )
        ),
        (3,),
    )
    if (
        frontier_node_offset > particle_count
        or material_tracer_positions.shape[0] > DEFAULT_MAX_TARGET_COUNT
    ):
        raise RuntimeError("common material tracer pack is inconsistent or too large")

    epsilon = config.relative_epsilon * min(
        float(np.min(state.sigma)), load.reference_chord_m
    )
    if not math.isfinite(epsilon) or epsilon <= 0.0:
        raise FloatingPointError("centered-FD epsilon is not finite and positive")
    adapter = FrozenParentCenteredFDAdapter(
        guarded_velocity,
        guarded_hash,
        epsilon=epsilon,
        parent_token=parent_token,
        max_target_count=DEFAULT_MAX_TARGET_COUNT,
    )
    substep_delta_time = config.delta_time_s / config.transport_substeps
    if config.birth_window_refinement:
        substep_dts = graded_substep_delta_times(
            config.delta_time_s, config.transport_substeps
        )
        effective_substeps = len(substep_dts)
    else:
        substep_dts = None
        effective_substeps = config.transport_substeps
    composer = _StageFieldComposer(
        adapter,
        parent_token=parent_token,
        galilean_velocity=galilean,
        substep_delta_time=substep_delta_time,
        jacobian_gate=config.stability_j_h_max,
        convective_gate=config.stability_convective_h_max,
    )
    physical_callback = composer.physical_field
    tracer_callback = composer.tracer_field
    observer_callback = composer.observe
    stream = ir_wrk3_stream_macro(
        state,
        config.delta_time_s,
        physical_callback,
        transport_substeps=effective_substeps,
        tracer_positions=material_tracer_positions,
        tracer_field_evaluator=tracer_callback,
        parent_token=parent_token,
        observer=observer_callback,
        substep_delta_times=substep_dts,
    )
    if validate_ir_wrk3_stream_result(stream) is not stream:
        raise RuntimeError("stream result validator changed live identity")
    verify_all()
    fd_ledger = adapter.snapshot()
    if validate_fd_call_ledger(fd_ledger) is not fd_ledger:
        raise RuntimeError("FD ledger validator changed live identity")
    after = _validate_sha256("parent_state_sha256_getter result", guarded_hash())
    if after != before:
        raise RuntimeError("stream transport mutated the frozen Ptera parent")
    stage_count = 3 * effective_substeps
    stability = _make_stability_envelope(composer, stage_count)
    counters = _expected_layer_counters(
        effective_substeps,
        stream,
        fd_ledger,
        load,
        composer.direct_call_count,
    )

    # Issue the row-owner capability only after the fallible field/stream path
    # has completed.  The locally derived pack is then required to match the
    # owner's canonical pack byte-for-byte before attestation.
    common_transport = begin_release_row_common_transport(owner, owner.state)
    if validate_release_row_common_transport(common_transport) is not common_transport:
        raise RuntimeError("common transport validator changed live identity")
    if (
        common_transport.live_particle_indices_by_cell
        != owner.state.live_boundary_indices_by_cell
        or common_transport.frontier_node_offset != frontier_node_offset
        or not np.array_equal(
            common_transport.material_tracer_positions, material_tracer_positions
        )
    ):
        raise RuntimeError("issued common material pack differs from preissued pack")
    flat = np.asarray(flat_live_indices, dtype=np.int64)
    live_sigma = stream.final_state.sigma[flat]
    end_time = owner.state.rows[-1].source_time_s + owner.release_dt_s
    attestation = attest_release_row_common_transport(
        common_transport,
        stream.final_state.positions,
        stream.final_state.gamma,
        stream.final_state.sigma,
        stream.final_tracer_positions,
        live_sigma,
        source_step_index=source_step,
        transport_end_time_s=end_time,
        transport_epoch=owner.epoch,
    )
    if validate_release_row_transport_attestation(attestation) is not attestation:
        raise RuntimeError("row transport attestation validator changed live identity")
    support = _make_support_envelope(
        common_transport.common_transport_sha256,
        attestation.attestation_sha256,
        attestation.transported_live_material_sigma_sha256,
        stream,
        flat_live_indices,
        common_transport.frontier_node_offset,
    )
    transport_digest = release_row_transport_digest(
        owner.state,
        stream.final_state.positions,
        stream.final_state.gamma,
        stream.final_state.sigma,
        attestation.transported_live_boundary_nodes,
        common_transport_attestation=attestation,
        source_step_index=source_step,
        transport_end_time_s=end_time,
        transport_epoch=owner.epoch,
    )
    result_common = {
        "interface_id": COUPLING_INTERFACE_ID,
        "case_id": CASE_ID,
        "scope": config.scope,
        "substep_role": config.role,
        "transport_substeps": effective_substeps,
        "ptera_step_index": ptera_step,
        "source_step_index": source_step,
        "source_time_s": owner.state.rows[-1].source_time_s,
        "row_owner_before_sha256": owner.owner_sha256,
        "row_state_before_sha256": owner.state.state_sha256,
        "common_transport_sha256": common_transport.common_transport_sha256,
        "transport_attestation_sha256": attestation.attestation_sha256,
        "transport_parent_digest": transport_digest,
        "stream_result": stream,
        "stream_stage_chain_sha256": stream.stage_chain_sha256,
        "stream_result_sha256": stream.result_sha256,
        "fd_call_ledger": fd_ledger,
        "fd_ledger_sha256": fd_ledger.ledger_sha256,
        "load_ledger": load,
        "load_ledger_sha256": load.ledger_sha256,
        "source_kelvin_evidence": kelvin,
        "source_kelvin_evidence_sha256": kelvin.evidence_sha256,
        "stability_envelope": stability,
        "support_envelope": support,
        "counters": counters,
        "parent_token": parent_token,
        "ptera_parent_sha256_before_transport": before,
        "ptera_parent_sha256_after_transport": after,
        "ptera_parent_state_unchanged": True,
        "observation_access": OBSERVATION_ACCESS,
        "force_scoring_status": FORCE_SCORING_STATUS,
    }
    # Exercise the exact constructor, payload builder, and hash path while the
    # owner is still current.  Only the two advance-derived digests are replaced
    # in this exact-dict plan after the single owner-consuming operation.
    preattest_post_advance_dependencies()
    provisional = result_type(
        **result_common,
        advanced_owner=owner,
        advanced_owner_sha256=owner.owner_sha256,
        advanced_state_sha256=owner.state.state_sha256,
        result_sha256="",
    )
    if type(provisional) is not result_type:
        raise RuntimeError("LayerResult constructor preflight changed exact type")
    payload_plan = layer_payload(provisional)
    if (
        type(payload_plan) is not dict
        or payload_plan.get("advanced_owner_sha256") != owner.owner_sha256
        or payload_plan.get("advanced_state_sha256") != owner.state.state_sha256
    ):
        raise RuntimeError("LayerResult payload preflight is inconsistent")
    preflight_result_sha256 = hash_payload("fluxv-v5h15-layer-result-v1", payload_plan)
    _validate_sha256("preflight LayerResult digest", preflight_result_sha256)
    # Close every callback-reachable and independently recomputable gate before
    # consuming the row owner.  The registry slot is already reserved; no user
    # callback or dynamically resolved global operation occurs after this point.
    preattest_post_advance_dependencies()
    # This is the single owner-consuming operation and occurs only after every
    # field, stability, ledger, load, support, parent, and attestation gate.
    advanced = advance_release_row_transport_parent(
        owner,
        owner.state,
        stream.final_state.positions,
        stream.final_state.gamma,
        stream.final_state.sigma,
        attestation.transported_live_boundary_nodes,
        common_transport_attestation=attestation,
        parent_transport_digest=transport_digest,
        source_step_index=source_step,
        transport_end_time_s=end_time,
        transport_epoch=owner.epoch,
    )
    if advanced_validator(advanced) is not advanced:
        raise RuntimeError("advanced row-owner validator changed live identity")
    payload_plan["advanced_owner_sha256"] = advanced.owner_sha256
    payload_plan["advanced_state_sha256"] = advanced.state.state_sha256
    result_sha256 = hash_payload("fluxv-v5h15-layer-result-v1", payload_plan)
    result = result_type(
        **result_common,
        advanced_owner=advanced,
        advanced_owner_sha256=advanced.owner_sha256,
        advanced_state_sha256=advanced.state.state_sha256,
        result_sha256=result_sha256,
    )
    return result


(
    _execute_layer_result_transaction,
    _attest_live_layer_result,
    _snapshot_live_layer_result_registry,
    _assert_live_layer_result_registry_unchanged,
) = _make_live_layer_result_registry(
    _transport_v5h15_committed_layer_impl,
)
_FROZEN_REGISTRY_FUNCTION_BINDINGS: Final[tuple[_RegistryFunctionBinding, ...]] = tuple(
    _freeze_registry_function(name) for name in _REGISTRY_FUNCTION_NAMES
)


def transport_v5h15_committed_layer(
    owner: ReleaseRowOwner,
    *,
    config: V5H15BaikCouplingConfig,
    ptera_step_index: int,
    source_step_index: int,
    load_ledger: V5H15LayerLoadLedger,
    source_kelvin_evidence: V5H15SourceKelvinEvidence,
    parent_velocity_evaluator: FrozenParentVelocityEvaluator,
    parent_state_sha256_getter: ParentStateSHA256Getter,
    parent_token: str,
    galilean_velocity_gp1_m_per_s: ArrayLike,
) -> V5H15LayerResult:
    """Reserve, execute, and publish one fail-closed committed-layer result."""

    _assert_execution_bindings()
    return _execute_layer_result_transaction(
        owner,
        config=config,
        ptera_step_index=ptera_step_index,
        source_step_index=source_step_index,
        load_ledger=load_ledger,
        source_kelvin_evidence=source_kelvin_evidence,
        parent_velocity_evaluator=parent_velocity_evaluator,
        parent_state_sha256_getter=parent_state_sha256_getter,
        parent_token=parent_token,
        galilean_velocity_gp1_m_per_s=galilean_velocity_gp1_m_per_s,
    )


def _validate_v5h15_layer_result_tree(result: object) -> V5H15LayerResult:
    if type(result) is not V5H15LayerResult:
        raise ValueError("result must be an exact V5H15LayerResult")
    if result.interface_id != COUPLING_INTERFACE_ID or result.case_id != CASE_ID:
        raise ValueError("layer result interface/case drift")
    if result.scope == "formal":
        if result.transport_substeps not in FORMAL_EFFECTIVE_TRANSPORT_SUBSTEPS:
            raise ValueError("formal layer result has an invalid graded substep count")
        expected_role = EFFECTIVE_ROLE_BY_N[result.transport_substeps]
    elif result.scope == "synthetic":
        if not 1 <= result.transport_substeps <= SYNTHETIC_EFFECTIVE_MAX_SUBSTEPS:
            raise ValueError("synthetic layer result has an invalid N")
        expected_role = "synthetic"
    elif result.scope == "diagnostic_smoke":
        if result.transport_substeps != DIAGNOSTIC_EFFECTIVE_TRANSPORT_SUBSTEPS[0]:
            raise ValueError("diagnostic layer result must use the graded N=32 policy")
        expected_role = "diagnostic_smoke"
    else:
        raise ValueError("layer result scope is invalid")
    if result.substep_role != expected_role:
        raise ValueError("layer result substep role is invalid")
    stream = validate_ir_wrk3_stream_result(result.stream_result)
    ledger = validate_fd_call_ledger(result.fd_call_ledger)
    load = validate_v5h15_layer_load_ledger(result.load_ledger)
    kelvin = validate_v5h15_source_kelvin_evidence(result.source_kelvin_evidence)
    owner = validate_release_row_owner(result.advanced_owner)
    if (
        result.stream_stage_chain_sha256 != stream.stage_chain_sha256
        or result.stream_result_sha256 != stream.result_sha256
        or result.fd_ledger_sha256 != ledger.ledger_sha256
        or result.load_ledger_sha256 != load.ledger_sha256
        or result.source_kelvin_evidence_sha256 != kelvin.evidence_sha256
        or result.advanced_owner_sha256 != owner.owner_sha256
        or result.advanced_state_sha256 != owner.state.state_sha256
    ):
        raise ValueError("layer result child digest binding mismatch")
    if (
        kelvin.source_step_index != result.source_step_index
        or kelvin.row_owner_sha256 != result.row_owner_before_sha256
    ):
        raise ValueError("layer result source Kelvin binding mismatch")
    if not owner.transport_events:
        raise ValueError("advanced owner lacks its transport event")
    event = owner.transport_events[-1]
    if (
        event.parent_state_sha256 != result.row_state_before_sha256
        or event.parent_transport_digest != result.transport_parent_digest
        or event.common_transport_attestation_sha256
        != result.transport_attestation_sha256
        or owner.state.parent_transport_digest != result.transport_parent_digest
        or owner.state.parent_transport_attestation_sha256
        != result.transport_attestation_sha256
        or owner.state.transport_source_step_index != result.source_step_index
        or owner.state.phase != "post_transport"
    ):
        raise ValueError("layer result row attestation/advance binding mismatch")
    if (
        not np.array_equal(owner.state.positions, stream.final_state.positions)
        or not np.array_equal(owner.state.gamma, stream.final_state.gamma)
        or not np.array_equal(owner.state.sigma, stream.final_state.sigma)
    ):
        raise ValueError("advanced owner endpoint differs from stream endpoint")
    stability = result.stability_envelope
    if type(stability) is not V5H15StabilityEnvelope:
        raise ValueError("stability envelope has a foreign type")
    expected_stability_sha = _hash_payload(
        "fluxv-v5h15-stability-envelope-v1", _stability_payload(stability)
    )
    if (
        stability.envelope_sha256 != expected_stability_sha
        or stability.stage_count != 3 * result.transport_substeps
        or stability.jacobian_gate != STABILITY_J_H_MAX
        or stability.convective_gate != STABILITY_CONVECTIVE_H_MAX
        or not stability.passed
        or stability.max_h_jacobian_frobenius > stability.jacobian_gate
        or stability.max_h_convective_over_sigma > stability.convective_gate
    ):
        raise ValueError("stability envelope is invalid")
    support = result.support_envelope
    if type(support) is not V5H15SupportEnvelope:
        raise ValueError("support envelope has a foreign type")
    if (
        support.envelope_sha256
        != _hash_payload("fluxv-v5h15-support-envelope-v1", _support_payload(support))
        or support.common_transport_sha256 != result.common_transport_sha256
        or support.transport_attestation_sha256 != result.transport_attestation_sha256
        or not support.exact_support_match
        or support.final_tracer_sha256 != stream.final_tracer_sha256
        or support.support_count + support.frontier_count
        != stream.final_tracer_positions.shape[0]
    ):
        raise ValueError("support envelope is invalid")
    expected_counters = _expected_layer_counters(
        result.transport_substeps,
        stream,
        ledger,
        load,
        result.counters.direct_field_call_count,
    )
    if (
        type(result.counters) is not V5H15LayerCounters
        or result.counters != expected_counters
    ):
        raise ValueError("layer counters are invalid")
    if (
        result.parent_token != stream.parent_token
        or result.parent_token != ledger.parent_token
        or result.ptera_parent_sha256_before_transport != ledger.parent_state_sha256
        or result.ptera_parent_sha256_after_transport
        != result.ptera_parent_sha256_before_transport
        or not result.ptera_parent_state_unchanged
        or result.observation_access != OBSERVATION_ACCESS
        or result.force_scoring_status != FORCE_SCORING_STATUS
    ):
        raise ValueError("layer parent/scope contract is invalid")
    if result.result_sha256 != _hash_payload(
        "fluxv-v5h15-layer-result-v1", _layer_payload(result)
    ):
        raise ValueError("layer result digest mismatch")
    return result


def validate_v5h15_layer_result(result: V5H15LayerResult) -> V5H15LayerResult:
    """Recompute the full compact tree and require exact issued identity."""

    _assert_execution_bindings()
    validated = _validate_v5h15_layer_result_tree(result)
    _attest_live_layer_result(validated)
    return validated


@dataclass(frozen=True, slots=True)
class _FeedbackRowView:
    for_source_step_index: int
    transport_end_time_s: float
    report_sha256: str


class _NativePteraParentHash:
    __slots__ = ("expected_sha256", "solver")

    def __init__(self, solver: UVPMHybridSolver, expected_sha256: str) -> None:
        self.solver = solver
        self.expected_sha256 = expected_sha256

    def __call__(self) -> str:
        observed = ptera_parent_state_sha256(self.solver)
        if observed != self.expected_sha256:
            raise RuntimeError("native Ptera parent hash drift")
        return observed


class _NativePteraParentVelocity:
    __slots__ = (
        "expected_sha256",
        "parent_token",
        "parent_velocity",
        "solver",
    )

    def __init__(
        self,
        solver: UVPMHybridSolver,
        *,
        expected_sha256: str,
        parent_token: str,
        parent_velocity: Callable[..., object],
    ) -> None:
        self.solver = solver
        self.expected_sha256 = expected_sha256
        self.parent_token = parent_token
        self.parent_velocity = parent_velocity

    def __call__(self, targets: FloatArray) -> FrozenParentVelocity:
        before = ptera_parent_state_sha256(self.solver)
        if before != self.expected_sha256:
            raise RuntimeError("native Ptera parent changed before velocity call")
        velocity = self.parent_velocity(self.solver, targets)
        after = ptera_parent_state_sha256(self.solver)
        if after != before:
            raise RuntimeError("native Ptera velocity evaluation mutated its parent")
        return make_frozen_parent_velocity(
            targets,
            velocity,
            parent_token=self.parent_token,
            parent_state_sha256=self.expected_sha256,
        )


class _BoundedV5H15SliceComplete(RuntimeError):
    pass


class V5H15NativeBaikCouplingSolver(NativePteraRVPMFeedbackSolver):
    """Native-Ptera W2 prefix with v5h15 row/IR-WRK3 transport ownership."""

    def __init__(
        self,
        unsteady_problem: Any,
        *,
        row_committer: V5H15GlobalRowCommitter,
        config: V5H15BaikCouplingConfig,
        **uvpm_kwargs: Any,
    ) -> None:
        _assert_execution_bindings()
        if type(config) is not V5H15BaikCouplingConfig:
            raise TypeError("config must be an exact V5H15BaikCouplingConfig")
        if not callable(row_committer):
            raise TypeError("row_committer must be callable")
        self.v5h15_config = config
        self._v5h15_row_committer = row_committer
        self._v5h15_previous_owner: ReleaseRowOwner | None = None
        self._v5h15_transported_parent: ParticleState | None = None
        self._v5h15_active_owner: ReleaseRowOwner | None = None
        self._v5h15_active_state: ParticleState | None = None
        self._v5h15_active_kelvin: V5H15SourceKelvinEvidence | None = None
        self._v5h15_slice_complete = False
        self.v5h15_layer_results: list[V5H15LayerResult] = []
        super().__init__(
            unsteady_problem,
            feedback_config=NativePteraRVPMFeedbackConfig(
                enabled=True,
                expected_wing_id="baik-w2-wing",
                expected_source_family="lev",
                rhs_residual_atol=config.no_penetration_atol,
            ),
            feedback_reports=(),
            **uvpm_kwargs,
        )
        if self.num_steps != PTERA_STEPS_PER_CYCLE * PTERA_CYCLES:
            raise ValueError("v5h15 requires the frozen two-cycle W2 movement")
        if float(self.delta_time) != config.delta_time_s:
            raise ValueError("v5h15 movement delta_time disagrees with config")
        if self.num_panels != PTERA_CHORDWISE_PANELS * PTERA_SPANWISE_PANELS:
            raise ValueError("v5h15 W2 movement panel grid is not 2x8")
        if self.first_results_step != 0:
            raise ValueError("v5h15 requires native loads at every prefix step")

    @property
    def v5h15_slice_complete(self) -> bool:
        return self._v5h15_slice_complete

    def _prepare_active_row(self) -> None:
        step = int(self._current_step)
        active_steps = ACTIVE_PTERA_STEPS[: self.v5h15_config.active_layer_limit]
        if step not in active_steps:
            if self._v5h15_active_owner is not None or self._v5h3_pending is not None:
                raise RuntimeError("v5h15 active row leaked into an inactive step")
            return
        active_index = active_steps.index(step)
        source_step = ACTIVE_SOURCE_STEPS[active_index]
        previous = self._v5h15_previous_owner
        parent = self._v5h15_transported_parent
        if active_index == 0:
            if previous is not None or parent is not None:
                raise RuntimeError("first v5h15 layer unexpectedly has a parent")
        elif previous is None or parent is None:
            raise RuntimeError("continuous v5h15 layer lacks a transported parent")
        parent_sha = None if parent is None else _stream_state_sha256(parent)
        request = V5H15GlobalRowCommitRequest(
            ptera_step_index=step,
            source_step_index=source_step,
            source_time_s=step * float(self.delta_time),
            expected_birth_mode=ACTIVE_BIRTH_MODES[active_index],
            previous_owner=previous,
            transported_parent=parent,
            transported_parent_sha256=parent_sha,
        )
        verify = _assert_execution_bindings
        verify_code = verify.__code__
        verify_defaults = verify.__defaults__
        parent_before = ptera_parent_state_sha256(self)
        verify()
        candidate = self._v5h15_row_committer(request)
        if (
            globals().get("_assert_execution_bindings") is not verify
            or verify.__code__ is not verify_code
            or verify.__defaults__ is not verify_defaults
        ):
            raise RuntimeError("v5h15 runtime verifier changed in row callback")
        verify()
        if ptera_parent_state_sha256(self) != parent_before:
            raise RuntimeError("row callback mutated native Ptera state")
        if type(candidate) is not V5H15RowCommitEnvelope:
            raise TypeError("row_committer must return V5H15RowCommitEnvelope")
        commit = candidate.commit_result
        if type(commit) is not RowCommitResult:
            raise TypeError("row commit envelope has a foreign commit schema")
        if (
            not commit.committed
            or commit.status != "compatible"
            or commit.first_mismatch is not None
        ):
            raise RuntimeError("row callback did not produce a compatible commit")
        owner = validate_current_release_row_owner(commit.owner)
        if owner is not commit.owner or commit.state is not owner.state:
            raise RuntimeError("row commit is not bound to its live owner")
        if owner.state.phase != "post_commit_pre_transport":
            raise RuntimeError("row callback did not return a fresh committed row")
        if (
            owner.epoch != active_index + 1
            or owner.state.release_index != active_index + 1
            or owner.state.rows[-1].source_time_s != request.source_time_s
            or owner.state.clone_count
            or owner.state.counter_particle_count
            or owner.state.fresh_upstream_particle_count
        ):
            raise RuntimeError("committed row epoch/time/physical ownership is invalid")
        if active_index == 0:
            if commit.event is not None or owner.events or owner.transport_events:
                raise RuntimeError("bootstrap row has an unexpected event history")
        elif (
            previous is None
            or not owner.events
            or commit.event is not owner.events[-1]
            or owner.owner_id != previous.owner_id
            or owner.epoch != previous.epoch + 1
            or owner.events[-1].parent_owner_sha256 != previous.owner_sha256
        ):
            raise RuntimeError("continuous row is not descended from transported owner")
        if (
            owner.state.positions.shape[0] == 0
            or owner.state.positions.shape[0] > self.v5h15_config.particle_cap
        ):
            raise ValueError("committed row particle count is empty or over cap")
        state = ParticleState(
            _frozen_float64("row positions", owner.state.positions, (3,)),
            _frozen_float64("row gamma", owner.state.gamma, (3,)),
            _frozen_float64("row sigma", owner.state.sigma, ()),
        )
        kelvin = validate_v5h15_source_kelvin_evidence(candidate.source_kelvin_evidence)
        if (
            kelvin.source_step_index != source_step
            or kelvin.row_owner_sha256 != owner.owner_sha256
        ):
            raise RuntimeError("row source Kelvin evidence is not bound to the commit")
        self._v5h15_active_owner = owner
        self._v5h15_active_state = state
        self._v5h15_active_kelvin = kelvin

    def _stage_feedback(self) -> None:
        step = int(self._current_step)
        active_steps = ACTIVE_PTERA_STEPS[: self.v5h15_config.active_layer_limit]
        if step not in active_steps:
            return
        owner = self._v5h15_active_owner
        state = self._v5h15_active_state
        if owner is None or state is None or self._v5h15_active_kelvin is None:
            raise RuntimeError("native feedback lacks a prevalidated v5h15 row")
        if self._v5h3_pending is not None:
            raise RuntimeError("a previous native feedback layer is pending")
        if self.current_operating_point.surfaceReflect_T_act_GP1_CgP1 is not None:
            raise ValueError("v5h15 does not support Ptera image surfaces")
        active_index = active_steps.index(step)
        source_step = ACTIVE_SOURCE_STEPS[active_index]
        empty = _frozen_float64("empty feedback targets", np.empty((0, 3)), (3,))
        dummy = NativePteraRVPMVelocityEvaluation(
            channel="collocation_rhs",
            ptera_step_index=step,
            target_points_gp1_m=empty,
            induced_velocity_gp1_m_per_s=empty,
            target_sha256=_v5h3_array_sha256(empty),
            velocity_sha256=_v5h3_array_sha256(empty),
        )
        row_view = _FeedbackRowView(
            for_source_step_index=source_step,
            transport_end_time_s=owner.state.rows[-1].source_time_s,
            report_sha256=owner.owner_sha256,
        )
        parent_normal = np.asarray(
            self._currentStackWakeWingInfluences__E, dtype=np.float64
        ).copy()
        self._v5h3_pending = _PendingFeedbackStep(
            report=row_view,  # type: ignore[arg-type]
            positions=state.positions,
            gamma=state.gamma,
            sigma=state.sigma,
            parent_wake_normal=parent_normal,
            feedback_normal=np.empty(0),
            combined_wake_normal=np.empty(0),
            collocation_evaluation=dummy,
            load_evaluations=[],
        )
        evaluation = self._field_velocity(
            self.stackCpp_GP1_CgP1, channel="collocation_rhs"
        )
        feedback_normal = np.einsum(
            "ij,ij->i",
            evaluation.induced_velocity_gp1_m_per_s,
            np.asarray(self.stackUnitNormals_GP1, dtype=np.float64),
        )
        combined = parent_normal + feedback_normal
        if not np.all(np.isfinite(combined)):
            raise FloatingPointError("v5h15 combined native Ptera RHS is non-finite")
        self._v5h3_pending.collocation_evaluation = evaluation
        self._v5h3_pending.feedback_normal = feedback_normal.copy()
        self._v5h3_pending.combined_wake_normal = combined.copy()
        self._currentStackWakeWingInfluences__E = combined

    def _calculate_wake_wing_influences(self) -> None:
        self._raise_if_poisoned()
        try:
            self._prepare_active_row()
        except Exception as error:
            self._poison(error)
            raise
        super()._calculate_wake_wing_influences()

    def _native_load_ledger(
        self,
        feedback: NativePteraRVPMFeedbackStepReport,
    ) -> V5H15LayerLoadLedger:
        airplanes = tuple(self.current_airplanes)
        if len(airplanes) != 1:
            raise RuntimeError("v5h15 W2 slice requires exactly one airplane")
        airplane = airplanes[0]
        panels: list[object] = []
        panel_ids: list[str] = []
        for airplane_index, current_airplane in enumerate(airplanes):
            for wing_index, wing in enumerate(current_airplane.wings):
                topology = np.asarray(wing.panels, dtype=object)
                if topology.ndim != 2:
                    raise RuntimeError("native Ptera panel topology is not 2D")
                for chord in range(topology.shape[0]):
                    for span in range(topology.shape[1]):
                        panels.append(topology[chord, span])
                        panel_ids.append(
                            f"airplane:{airplane_index}/wing:{wing_index}/chord:{chord}/span:{span}"
                        )
        if len(panels) != self.num_panels or any(
            left is not right
            for left, right in zip(panels, tuple(self.panels), strict=True)
        ):
            raise RuntimeError("native Ptera panel topology/order changed")
        parent_arrays = (
            airplane.forces_W,
            airplane.forceCoefficients_W,
            airplane.moments_W_CgP1,
            airplane.momentCoefficients_W_CgP1,
            *(
                array
                for panel in panels
                for array in (panel.forces_W, panel.moments_W_CgP1)
            ),
        )
        snapshot = tuple(
            (
                np.asarray(array).dtype.str,
                np.asarray(array).shape,
                np.asarray(array).tobytes(order="C"),
                np.asarray(array).flags.writeable,
            )
            for array in parent_arrays
        )
        parent_sha = ptera_parent_state_sha256(self)
        ledger = make_v5h15_layer_load_ledger(
            ptera_step_index=int(self._current_step),
            parent_state_sha256=parent_sha,
            reference_chord_m=float(airplane.c_ref),
            panel_ids=tuple(panel_ids),
            forces_w=airplane.forces_W,
            force_coefficients_w=airplane.forceCoefficients_W,
            moments_w_cgp1=airplane.moments_W_CgP1,
            moment_coefficients_w=airplane.momentCoefficients_W_CgP1,
            panel_forces_w=np.vstack([panel.forces_W for panel in panels]),
            panel_moments_w_cgp1=np.vstack([panel.moments_W_CgP1 for panel in panels]),
            no_penetration_residual=feedback.no_penetration_residual,
            no_penetration_atol=self.v5h15_config.no_penetration_atol,
            collocation_evaluation_count=feedback.collocation_evaluation_count,
            load_batch_evaluation_count=feedback.load_leg_evaluation_count,
            native_load_call_count=feedback.parent_load_call_count,
        )
        observed = tuple(
            (
                np.asarray(array).dtype.str,
                np.asarray(array).shape,
                np.asarray(array).tobytes(order="C"),
                np.asarray(array).flags.writeable,
            )
            for array in parent_arrays
        )
        if observed != snapshot or ptera_parent_state_sha256(self) != parent_sha:
            raise RuntimeError("native load snapshot mutated Ptera storage")
        return ledger

    def _populate_next_airplanes_wake(self) -> None:
        active = self._v5h15_active_owner
        if active is None:
            return super()._populate_next_airplanes_wake()
        feedback_count = len(self.v5h3_feedback_step_reports)
        super()._populate_next_airplanes_wake()
        if len(self.v5h3_feedback_step_reports) != feedback_count + 1:
            raise RuntimeError("native Ptera feedback ledger did not commit")
        feedback = self.v5h3_feedback_step_reports[-1]
        load = self._native_load_ledger(feedback)
        parent_sha = load.parent_state_sha256
        parent_token = f"v5h15:ptera-step:{int(self._current_step)}:parent:{parent_sha}"
        parent_velocity = _assert_v5h4_bindings()["parent_velocity"]
        evaluator = _NativePteraParentVelocity(
            self,
            expected_sha256=parent_sha,
            parent_token=parent_token,
            parent_velocity=parent_velocity,
        )
        getter = _NativePteraParentHash(self, parent_sha)
        active_index = len(self.v5h15_layer_results)
        kelvin = self._v5h15_active_kelvin
        if kelvin is None:
            raise RuntimeError("active native layer lost source Kelvin evidence")
        result = transport_v5h15_committed_layer(
            active,
            config=self.v5h15_config,
            ptera_step_index=int(self._current_step),
            source_step_index=ACTIVE_SOURCE_STEPS[active_index],
            load_ledger=load,
            source_kelvin_evidence=kelvin,
            parent_velocity_evaluator=evaluator,
            parent_state_sha256_getter=getter,
            parent_token=parent_token,
            galilean_velocity_gp1_m_per_s=np.asarray(
                self._currentVInf_GP1__E, dtype=np.float64
            ),
        )
        if validate_v5h15_layer_result(result) is not result:
            raise RuntimeError("native layer validator changed issued identity")
        self.v5h15_layer_results.append(result)
        self._v5h15_previous_owner = result.advanced_owner
        self._v5h15_transported_parent = result.stream_result.final_state
        self._v5h15_active_owner = None
        self._v5h15_active_state = None
        self._v5h15_active_kelvin = None
        if len(self.v5h15_layer_results) == self.v5h15_config.active_layer_limit:
            self._v5h15_slice_complete = True
            raise _BoundedV5H15SliceComplete

    def run(
        self,
        prescribed_wake: bool | np.bool_ = True,
        calculate_streamlines: bool | np.bool_ = False,
        show_progress: bool | np.bool_ = False,
    ) -> None:
        if calculate_streamlines:
            raise ValueError("bounded v5h15 slice forbids streamline postprocessing")
        if self._v5h15_slice_complete:
            raise RuntimeError("bounded v5h15 slice cannot run twice")
        try:
            super().run(
                prescribed_wake=prescribed_wake,
                calculate_streamlines=False,
                show_progress=show_progress,
            )
        except _BoundedV5H15SliceComplete:
            if len(self.v5h15_layer_results) != self.v5h15_config.active_layer_limit:
                raise RuntimeError("v5h15 bounded stop occurred at the wrong layer")
            return
        except Exception as error:
            self._poison(error)
            raise
        raise RuntimeError("full Ptera run escaped the bounded v5h15 stop")


def make_fluxv_v5h15_baik_w2_solver(
    unsteady_problem: Any,
    *,
    row_committer: V5H15GlobalRowCommitter,
    config: V5H15BaikCouplingConfig,
    **uvpm_kwargs: Any,
) -> V5H15NativeBaikCouplingSolver:
    """Construct the isolated bounded native-Ptera v5h15 W2 solver."""

    _assert_execution_bindings()
    return V5H15NativeBaikCouplingSolver(
        unsteady_problem,
        row_committer=row_committer,
        config=config,
        **uvpm_kwargs,
    )


_FROZEN_EXECUTION_BINDINGS: Final[tuple[tuple[str, object], ...]] = tuple(
    (name, globals()[name]) for name in _EXECUTION_BINDING_NAMES
)
_FROZEN_EXECUTION_LITERAL: Final[tuple[object, ...]] = _execution_literal()


def _assert_execution_bindings(
    frozen_bindings: tuple[tuple[str, object], ...] = _FROZEN_EXECUTION_BINDINGS,
    frozen_literal: tuple[object, ...] = _FROZEN_EXECUTION_LITERAL,
) -> None:
    if globals().get("_FROZEN_EXECUTION_BINDINGS") is not frozen_bindings:
        raise RuntimeError("v5h15 execution binding registry drift")
    if globals().get("_FROZEN_EXECUTION_LITERAL") is not frozen_literal:
        raise RuntimeError("v5h15 execution literal registry drift")
    if globals().get("_EXECUTION_BINDING_NAMES") is not _EXECUTION_BINDING_NAMES:
        raise RuntimeError("v5h15 execution binding-name registry drift")
    for name, frozen in frozen_bindings:
        if globals().get(name) is not frozen:
            raise RuntimeError(f"v5h15 execution binding drift: {name}")
    if _execution_literal() != frozen_literal:
        raise RuntimeError("v5h15 execution constant drift")
    _assert_registry_function_bindings()


del _make_live_layer_result_registry
del _freeze_registry_function
del _transport_v5h15_committed_layer_impl


__all__ = (
    "ACTIVE_BIRTH_MODES",
    "ACTIVE_PTERA_STEPS",
    "ACTIVE_SOURCE_STEPS",
    "CASE_ID",
    "COUPLING_INTERFACE_ID",
    "FORCE_SCORING_STATUS",
    "FORMAL_TRANSPORT_SUBSTEPS",
    "OBSERVATION_ACCESS",
    "V5H15BaikCouplingConfig",
    "V5H15GlobalRowCommitRequest",
    "V5H15GlobalRowCommitter",
    "V5H15LayerCounters",
    "V5H15LayerLoadLedger",
    "V5H15LayerResult",
    "V5H15SourceKelvinEvidence",
    "V5H15NativeBaikCouplingSolver",
    "V5H15RowCommitEnvelope",
    "V5H15StabilityEnvelope",
    "V5H15SupportEnvelope",
    "make_v5h15_layer_load_ledger",
    "make_v5h15_source_kelvin_evidence",
    "make_fluxv_v5h15_baik_w2_solver",
    "transport_v5h15_committed_layer",
    "validate_v5h15_layer_load_ledger",
    "validate_v5h15_source_kelvin_evidence",
    "validate_v5h15_layer_result",
)
