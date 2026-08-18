"""Frozen native-Ptera field for one rVPM transport step.

The v5h4 auxiliary gate closes only the reverse velocity direction of the
partitioned FluxV coupling.  It reads one already solved Ptera time layer and
advances one live v5h2 particle cloud with a shared LSRK3 field:

``U = U_rVPM,self + U_Ptera,bound+wake+freestream``.

The parent-only Ptera method is called explicitly on
:class:`fluxvortex.solver.UVPMHybridSolver`, bypassing the v5h3 override so the
particle cloud cannot induce on itself twice.  Ptera's spatial velocity
Jacobian is evaluated with the preregistered centred-difference family.  The
Ptera state is hashed before and after and must remain bitwise unchanged.

This is a one-step, frozen-field mechanical transport gate.  It does not add a
new DVM release, update Ptera, score loads, or claim a fully coupled solution.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import inspect
from numbers import Integral, Real
from pathlib import Path
from threading import RLock
from typing import Any, Callable, Final
import weakref

import numpy as np

import fluxvortex.rvpm_reference as _reference_module
import fluxvortex.rvpm_transport as _transport_module
import fluxvortex.solver as _solver_module
from fluxvortex.rvpm_reference import (
    DirectField,
    direct_gaussian_erf_velocity_jacobian,
)
from fluxvortex.rvpm_transport import (
    LSRKStageRecord,
    ParticleState,
    RK_A,
    RK_B,
    RVPMStepRHS,
    lsrk3_step_direct,
    make_particle_state,
    reformulated_vpm_rhs,
)
from fluxvortex.solver import UVPMHybridSolver

from . import v5h2_dyadic_cumulative_cloud_transport as _dyadic_module
from .v5h2_dyadic_cumulative_cloud_transport import (
    DyadicCumulativeCloudTransportReport,
    materialize_dyadic_cumulative_particle_state,
    validate_dyadic_cumulative_cloud_transport_report,
)


PTERA_RVPM_TRANSPORT_INTERFACE_ID: Final = "fluxv-v5h4-frozen-ptera-rvpm-transport-v1"
PTERA_FIELD_OWNER: Final = "native-ptera-bound-ring-wake-freestream-read-only"
RVPM_TRANSPORT_OWNER: Final = "flowvpm-parity-rvpm-lsrk3"
FORCE_SCORING_STATUS: Final = "blocked_frozen_ptera_transport_mechanics_only"
PREREGISTERED_RELATIVE_EPSILONS: Final = (2.0**-8, 2.0**-10, 2.0**-12)
NOMINAL_RELATIVE_EPSILON: Final = 2.0**-10
MAX_PARTICLES: Final = 1_000_000


@dataclass(frozen=True, slots=True)
class FrozenExternalField:
    """Velocity/Jacobian at one particle stage."""

    velocity: np.ndarray
    jacobian: np.ndarray


@dataclass(frozen=True, slots=True)
class PteraFieldEvaluation:
    """One centred-difference evaluation of the frozen Ptera field."""

    target_positions_gp1_m: np.ndarray
    velocity_gp1_m_per_s: np.ndarray
    jacobian_per_s: np.ndarray
    epsilon_m: float
    center_call_count: int
    finite_difference_call_count: int
    target_sha256: str
    velocity_sha256: str
    jacobian_sha256: str


@dataclass(frozen=True, slots=True)
class CoupledLSRKStageRecord:
    """Auditable combined-field LSRK3 stage."""

    stage: int
    a: float
    b: float
    pre: ParticleState
    self_field: FrozenExternalField
    ptera_field: PteraFieldEvaluation
    total_rhs: RVPMStepRHS
    post: ParticleState
    position_storage_pre: np.ndarray
    gamma_storage_pre: np.ndarray
    sigma_storage_pre: np.ndarray
    position_storage_post: np.ndarray
    gamma_storage_post: np.ndarray
    sigma_storage_post: np.ndarray


@dataclass(frozen=True, slots=True)
class FrozenPteraRVPMTransportResult:
    """Result and ownership ledger for one v5h4 transport call."""

    enabled: bool
    interface_id: str
    ptera_step_index: int | None
    dvm_for_source_step_index: int | None
    source_report_sha256: str | None
    relative_epsilon: float | None
    epsilon_m: float | None
    initial_particle_count: int
    final_state: ParticleState
    stages: tuple[CoupledLSRKStageRecord, ...]
    baseline_stages: tuple[LSRKStageRecord, ...]
    self_field_call_count: int
    ptera_center_call_count: int
    ptera_finite_difference_call_count: int
    ptera_parent_state_sha256_before: str | None
    ptera_parent_state_sha256_after: str | None
    ptera_parent_state_unchanged: bool
    parent_only_bypass: bool
    feedback_write_count: int
    parent_write_count: int
    load_write_count: int
    ptera_field_owner: str
    rvpm_transport_owner: str
    force_scoring_status: str


_LOCK = RLock()
_REPORT_CONSUMPTIONS: dict[int, tuple[weakref.ReferenceType[object], str, int]] = {}


def _source_sha256(module: object) -> str:
    path = getattr(module, "__file__", None)
    if type(path) is not str or not path.endswith(".py"):
        raise RuntimeError("v5h4 dependency has no auditable Python source")
    return sha256(Path(path).read_bytes()).hexdigest()


_FROZEN_SOLVER_CLASS = UVPMHybridSolver
_FROZEN_PARENT_VELOCITY = UVPMHybridSolver.calculate_solution_velocity
_FROZEN_DIRECT_FIELD = direct_gaussian_erf_velocity_jacobian
_FROZEN_BASELINE_STEP = lsrk3_step_direct
_FROZEN_MAKE_STATE = make_particle_state
_FROZEN_RHS = reformulated_vpm_rhs
_FROZEN_VALIDATE_REPORT = validate_dyadic_cumulative_cloud_transport_report
_FROZEN_MATERIALIZE_REPORT = materialize_dyadic_cumulative_particle_state
_FROZEN_RK_A = RK_A
_FROZEN_RK_B = RK_B
_PTERA_PARENT_GLOBALS = _FROZEN_PARENT_VELOCITY.__globals__
_FROZEN_PTERA_MODULES = {
    name: _PTERA_PARENT_GLOBALS[name]
    for name in (
        "_aerodynamics_functions",
        "_parameter_validation",
        "_transformations",
    )
}
_FROZEN_PTERA_CALLABLES = {
    "collapsed_velocities_from_ring_vortices": (
        _FROZEN_PTERA_MODULES["_aerodynamics_functions"],
        "collapsed_velocities_from_ring_vortices",
        _FROZEN_PTERA_MODULES[
            "_aerodynamics_functions"
        ].collapsed_velocities_from_ring_vortices,
    ),
    "arrayLike_of_threeD_number_vectorLikes_return_float": (
        _FROZEN_PTERA_MODULES["_parameter_validation"],
        "arrayLike_of_threeD_number_vectorLikes_return_float",
        _FROZEN_PTERA_MODULES[
            "_parameter_validation"
        ].arrayLike_of_threeD_number_vectorLikes_return_float,
    ),
    "apply_T_to_vectors": (
        _FROZEN_PTERA_MODULES["_transformations"],
        "apply_T_to_vectors",
        _FROZEN_PTERA_MODULES["_transformations"].apply_T_to_vectors,
    ),
}
_FROZEN_CLASS_BINDINGS = (
    (_reference_module, "DirectField", DirectField),
    (_transport_module, "LSRKStageRecord", LSRKStageRecord),
    (_transport_module, "ParticleState", ParticleState),
    (_transport_module, "RVPMStepRHS", RVPMStepRHS),
)
_FROZEN_SOURCE_HASHES = {
    "solver": _source_sha256(_solver_module),
    "reference": _source_sha256(_reference_module),
    "transport": _source_sha256(_transport_module),
    "dyadic": _source_sha256(_dyadic_module),
    "ptera_parent": _source_sha256(inspect.getmodule(_FROZEN_PARENT_VELOCITY)),
    **{
        f"ptera_{name}": _source_sha256(module)
        for name, module in _FROZEN_PTERA_MODULES.items()
    },
}


def _freeze_transitive_python_functions(
    roots: tuple[Callable[..., Any], ...],
) -> tuple[tuple[dict[str, Any], str, Callable[..., Any]], ...]:
    """Freeze Python helper slots recursively for the rVPM roots."""

    pending = list(roots)
    visited_functions: set[int] = set()
    visited_slots: set[tuple[int, str]] = set()
    bindings: list[tuple[dict[str, Any], str, Callable[..., Any]]] = []
    while pending:
        function = pending.pop(0)
        if id(function) in visited_functions:
            continue
        visited_functions.add(id(function))
        code = getattr(function, "__code__", None)
        function_globals = getattr(function, "__globals__", None)
        if code is None or not isinstance(function_globals, dict):
            continue
        for name in sorted(set(code.co_names)):
            dependency = function_globals.get(name)
            if not inspect.isfunction(dependency):
                continue
            slot = (id(function_globals), name)
            if slot not in visited_slots:
                visited_slots.add(slot)
                bindings.append((function_globals, name, dependency))
            pending.append(dependency)
    return tuple(bindings)


_FROZEN_TRANSITIVE_FUNCTIONS = _freeze_transitive_python_functions(
    (
        _FROZEN_DIRECT_FIELD,
        _FROZEN_BASELINE_STEP,
        _FROZEN_MAKE_STATE,
        _FROZEN_RHS,
    )
)


def _assert_bindings() -> dict[str, Any]:
    ptera_parent_module = inspect.getmodule(_FROZEN_PARENT_VELOCITY)
    if (
        _solver_module.UVPMHybridSolver is not _FROZEN_SOLVER_CLASS
        or UVPMHybridSolver.calculate_solution_velocity is not _FROZEN_PARENT_VELOCITY
        or _reference_module.direct_gaussian_erf_velocity_jacobian
        is not _FROZEN_DIRECT_FIELD
        or _transport_module.lsrk3_step_direct is not _FROZEN_BASELINE_STEP
        or _transport_module.make_particle_state is not _FROZEN_MAKE_STATE
        or _transport_module.reformulated_vpm_rhs is not _FROZEN_RHS
        or _dyadic_module.validate_dyadic_cumulative_cloud_transport_report
        is not _FROZEN_VALIDATE_REPORT
        or _dyadic_module.materialize_dyadic_cumulative_particle_state
        is not _FROZEN_MATERIALIZE_REPORT
        or _transport_module.RK_A is not _FROZEN_RK_A
        or _transport_module.RK_B is not _FROZEN_RK_B
        or RK_A is not _FROZEN_RK_A
        or RK_B is not _FROZEN_RK_B
        or any(
            _PTERA_PARENT_GLOBALS.get(name) is not module
            for name, module in _FROZEN_PTERA_MODULES.items()
        )
        or any(
            getattr(module, attribute, None) is not frozen
            for module, attribute, frozen in _FROZEN_PTERA_CALLABLES.values()
        )
        or any(
            getattr(module, attribute, None) is not frozen
            for module, attribute, frozen in _FROZEN_CLASS_BINDINGS
        )
        or any(
            function_globals.get(name) is not frozen
            for function_globals, name, frozen in _FROZEN_TRANSITIVE_FUNCTIONS
        )
        or any(
            _source_sha256(module) != _FROZEN_SOURCE_HASHES[name]
            for name, module in (
                ("solver", _solver_module),
                ("reference", _reference_module),
                ("transport", _transport_module),
                ("dyadic", _dyadic_module),
                ("ptera_parent", ptera_parent_module),
                *(
                    (f"ptera_{ptera_name}", module)
                    for ptera_name, module in _FROZEN_PTERA_MODULES.items()
                ),
            )
        )
    ):
        raise ValueError("v5h4 dependency callable or source was replaced")
    return {
        "parent_velocity": _FROZEN_PARENT_VELOCITY,
        "direct_field": _FROZEN_DIRECT_FIELD,
        "baseline_step": _FROZEN_BASELINE_STEP,
        "make_state": _FROZEN_MAKE_STATE,
        "rhs": _FROZEN_RHS,
        "validate_report": _FROZEN_VALIDATE_REPORT,
        "materialize_report": _FROZEN_MATERIALIZE_REPORT,
    }


def _finite_positive(name: str, value: object) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite positive real")
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be a finite positive real")
    return result


def _strict_integer(name: str, value: object, *, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer >= {minimum}")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return result


def _float64_array(name: str, value: object, *, ndim: int) -> np.ndarray:
    original = np.asarray(value)
    if original.dtype.kind not in "iuf" or original.dtype.kind == "b":
        raise ValueError(f"{name} must use a real numeric dtype")
    result = np.ascontiguousarray(original, dtype=np.float64)
    if result.ndim != ndim or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} has invalid shape or non-finite values")
    return result


def _array_sha256(array: np.ndarray) -> str:
    value = np.asarray(array)
    digest = sha256()
    digest.update(value.dtype.str.encode("ascii"))
    digest.update(repr(value.shape).encode("ascii"))
    digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _state_bytes(state: ParticleState) -> tuple[bytes, bytes, bytes]:
    return (
        state.positions.tobytes(order="C"),
        state.gamma.tobytes(order="C"),
        state.sigma.tobytes(order="C"),
    )


def _update_hash_with_array(digest: Any, name: str, value: object) -> None:
    array = np.asarray(value)
    digest.update(name.encode("utf-8"))
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(repr(array.shape).encode("ascii"))
    digest.update(array.tobytes(order="C"))


def ptera_parent_state_sha256(solver: object) -> str:
    """Hash every parent array/object touched by the v5h4 read-only gate."""

    if not isinstance(solver, UVPMHybridSolver):
        raise TypeError("solver must inherit UVPMHybridSolver")
    digest = sha256()
    digest.update(f"current_step:{int(solver._current_step)}".encode("ascii"))
    digest.update(float(solver.delta_time).hex().encode("ascii"))
    operating_point = solver.current_operating_point
    digest.update(float(operating_point.nu).hex().encode("ascii"))
    reflection = operating_point.surfaceReflect_T_act_GP1_CgP1
    if reflection is None:
        digest.update(b"surface_reflection:none")
    else:
        _update_hash_with_array(digest, "surface_reflection", reflection)
    for name in (
        "_current_bound_vortex_strengths",
        "_current_wake_vortex_strengths",
        "_current_wake_vortex_ages",
        "_currentStackBrwrvp_GP1_CgP1",
        "_currentStackFrwrvp_GP1_CgP1",
        "_currentStackFlwrvp_GP1_CgP1",
        "_currentStackBlwrvp_GP1_CgP1",
        "_currentStackBoundRc0s",
        "_currentStackWakeRc0s",
        "stackBrbrvp_GP1_CgP1",
        "stackFrbrvp_GP1_CgP1",
        "stackFlbrvp_GP1_CgP1",
        "stackBlbrvp_GP1_CgP1",
        "_currentVInf_GP1__E",
    ):
        _update_hash_with_array(digest, name, getattr(solver, name))
    for panel_index, panel in enumerate(np.ravel(solver.panels)):
        digest.update(f"panel:{panel_index}".encode("ascii"))
        digest.update(float(panel.ring_vortex.strength).hex().encode("ascii"))
        for name in ("forces_GP1", "moments_GP1_CgP1"):
            _update_hash_with_array(digest, name, getattr(panel, name))
    for airplane_index, airplane in enumerate(solver.current_airplanes):
        digest.update(f"airplane:{airplane_index}".encode("ascii"))
        digest.update(float(airplane.c_ref).hex().encode("ascii"))
        for name in (
            "forces_W",
            "forceCoefficients_W",
            "moments_W_CgP1",
            "momentCoefficients_W_CgP1",
        ):
            _update_hash_with_array(digest, name, getattr(airplane, name))
    vpm = solver._vpm_field
    digest.update(f"vpm_count:{int(vpm.np)}".encode("ascii"))
    for name in ("_pos", "_gamma", "_sigma", "_age"):
        _update_hash_with_array(digest, f"vpm.{name}", getattr(vpm, name))
    return digest.hexdigest()


def _ptera_reference_length(solver: UVPMHybridSolver, sigma: np.ndarray) -> float:
    sigma_min = float(np.min(sigma))
    chord_values: list[float] = []
    for airplane in solver.current_airplanes:
        chord = getattr(airplane, "c_ref", None)
        if chord is not None:
            chord_values.append(_finite_positive("airplane.c_ref", chord))
    if not chord_values:
        raise RuntimeError("Ptera solver has no positive reference chord")
    return min(sigma_min, min(chord_values))


def evaluate_frozen_parent_ptera_field(
    solver: object,
    target_positions_gp1_m: object,
    *,
    epsilon_m: object,
) -> PteraFieldEvaluation:
    """Evaluate parent-only Ptera velocity and centred spatial Jacobian."""

    trusted = _assert_bindings()
    if not isinstance(solver, UVPMHybridSolver):
        raise TypeError("solver must inherit UVPMHybridSolver")
    targets = _float64_array("target_positions_gp1_m", target_positions_gp1_m, ndim=2)
    if targets.shape[1:] != (3,):
        raise ValueError("target_positions_gp1_m must have shape (N,3)")
    epsilon = _finite_positive("epsilon_m", epsilon_m)
    parent_velocity = trusted["parent_velocity"]
    center = _float64_array(
        "Ptera center velocity",
        parent_velocity(solver, targets),
        ndim=2,
    )
    if center.shape != targets.shape:
        raise RuntimeError("Ptera center velocity shape mismatch")
    jacobian = np.empty((targets.shape[0], 3, 3), dtype=np.float64)
    for axis in range(3):
        offset = np.zeros_like(targets)
        offset[:, axis] = epsilon
        plus = _float64_array(
            "Ptera plus velocity",
            parent_velocity(solver, targets + offset),
            ndim=2,
        )
        minus = _float64_array(
            "Ptera minus velocity",
            parent_velocity(solver, targets - offset),
            ndim=2,
        )
        if plus.shape != targets.shape or minus.shape != targets.shape:
            raise RuntimeError("Ptera finite-difference velocity shape mismatch")
        jacobian[:, :, axis] = (plus - minus) / (2.0 * epsilon)
    if not np.all(np.isfinite(jacobian)):
        raise FloatingPointError("Ptera finite-difference Jacobian is non-finite")
    return PteraFieldEvaluation(
        target_positions_gp1_m=targets.copy(),
        velocity_gp1_m_per_s=center.copy(),
        jacobian_per_s=jacobian,
        epsilon_m=epsilon,
        center_call_count=1,
        finite_difference_call_count=6,
        target_sha256=_array_sha256(targets),
        velocity_sha256=_array_sha256(center),
        jacobian_sha256=_array_sha256(jacobian),
    )


def lsrk3_step_with_external_field(
    state: ParticleState,
    delta_time_s: object,
    *,
    external_field: Callable[[np.ndarray], FrozenExternalField] | object,
    baseline_freestream_velocity_gp1_m_per_s: object,
    enabled: bool = True,
) -> FrozenPteraRVPMTransportResult:
    """Advance one state with self field plus an external velocity/Jacobian.

    Disabled mode is input-blind to ``external_field`` and delegates directly
    to the frozen v5h2 reference step, providing the exact-reduction baseline.
    """

    if type(enabled) is not bool:
        raise TypeError("enabled must be a bool")
    trusted = _assert_bindings()
    dt = _finite_positive("delta_time_s", delta_time_s)
    initial = trusted["make_state"](state.positions, state.gamma, state.sigma)
    if initial.positions.shape[0] > MAX_PARTICLES:
        raise ValueError("particle state exceeds the v5h4 resource cap")
    freestream = _float64_array(
        "baseline_freestream_velocity_gp1_m_per_s",
        baseline_freestream_velocity_gp1_m_per_s,
        ndim=1,
    )
    if freestream.shape != (3,):
        raise ValueError("baseline freestream must have shape (3,)")
    if not enabled:
        baseline_state, baseline_stages = trusted["baseline_step"](
            initial,
            dt,
            freestream_velocity=freestream,
        )
        return FrozenPteraRVPMTransportResult(
            enabled=False,
            interface_id=PTERA_RVPM_TRANSPORT_INTERFACE_ID,
            ptera_step_index=None,
            dvm_for_source_step_index=None,
            source_report_sha256=None,
            relative_epsilon=None,
            epsilon_m=None,
            initial_particle_count=initial.positions.shape[0],
            final_state=baseline_state,
            stages=(),
            baseline_stages=baseline_stages,
            self_field_call_count=3,
            ptera_center_call_count=0,
            ptera_finite_difference_call_count=0,
            ptera_parent_state_sha256_before=None,
            ptera_parent_state_sha256_after=None,
            ptera_parent_state_unchanged=True,
            parent_only_bypass=True,
            feedback_write_count=0,
            parent_write_count=0,
            load_write_count=0,
            ptera_field_owner=PTERA_FIELD_OWNER,
            rvpm_transport_owner=RVPM_TRANSPORT_OWNER,
            force_scoring_status=FORCE_SCORING_STATUS,
        )
    if not callable(external_field):
        raise TypeError("enabled external_field must be callable")

    positions = initial.positions.copy()
    gamma = initial.gamma.copy()
    sigma = initial.sigma.copy()
    position_storage = np.zeros_like(positions)
    gamma_storage = np.zeros_like(gamma)
    sigma_storage = np.zeros_like(sigma)
    stages: list[CoupledLSRKStageRecord] = []
    for stage, (a_coefficient, b_coefficient) in enumerate(
        zip(RK_A, RK_B, strict=True), start=1
    ):
        pre = trusted["make_state"](positions, gamma, sigma)
        position_storage_pre = position_storage.copy()
        gamma_storage_pre = gamma_storage.copy()
        sigma_storage_pre = sigma_storage.copy()
        self_direct: DirectField = trusted["direct_field"](positions, gamma, sigma)
        self_field = FrozenExternalField(
            velocity=self_direct.velocity.copy(),
            jacobian=self_direct.jacobian.copy(),
        )
        supplied = external_field(positions.copy())
        if type(supplied) is not FrozenExternalField:
            raise TypeError("external_field must return FrozenExternalField")
        external_velocity = _float64_array(
            "external velocity", supplied.velocity, ndim=2
        )
        external_jacobian = _float64_array(
            "external Jacobian", supplied.jacobian, ndim=3
        )
        if external_velocity.shape != positions.shape:
            raise ValueError("external velocity shape mismatch")
        if external_jacobian.shape != (positions.shape[0], 3, 3):
            raise ValueError("external Jacobian shape mismatch")
        total_velocity = self_direct.velocity + external_velocity
        total_jacobian = self_direct.jacobian + external_jacobian
        stretching, z_rate, gamma_rate, sigma_rate = trusted["rhs"](
            gamma, sigma, total_jacobian
        )
        rhs = RVPMStepRHS(
            velocity=total_velocity.copy(),
            jacobian=total_jacobian.copy(),
            stretching=stretching.copy(),
            z_rate=z_rate.copy(),
            gamma_rate=gamma_rate.copy(),
            sigma_rate=sigma_rate.copy(),
        )
        position_storage = a_coefficient * position_storage + dt * total_velocity
        gamma_storage = a_coefficient * gamma_storage + dt * gamma_rate
        sigma_storage = a_coefficient * sigma_storage + dt * sigma_rate
        positions = positions + b_coefficient * position_storage
        gamma = gamma + b_coefficient * gamma_storage
        sigma = sigma + b_coefficient * sigma_storage
        if not (
            np.all(np.isfinite(positions))
            and np.all(np.isfinite(gamma))
            and np.all(np.isfinite(sigma))
        ):
            raise FloatingPointError(f"non-finite v5h4 state after stage {stage}")
        if np.any(sigma <= 0.0):
            raise FloatingPointError(f"non-positive v5h4 sigma after stage {stage}")
        post = trusted["make_state"](positions, gamma, sigma)
        # Generic external records do not claim Ptera calls.  The live wrapper
        # replaces this placeholder with its measured field ledger.
        placeholder = PteraFieldEvaluation(
            target_positions_gp1_m=pre.positions.copy(),
            velocity_gp1_m_per_s=external_velocity.copy(),
            jacobian_per_s=external_jacobian.copy(),
            epsilon_m=0.0,
            center_call_count=0,
            finite_difference_call_count=0,
            target_sha256=_array_sha256(pre.positions),
            velocity_sha256=_array_sha256(external_velocity),
            jacobian_sha256=_array_sha256(external_jacobian),
        )
        stages.append(
            CoupledLSRKStageRecord(
                stage=stage,
                a=a_coefficient,
                b=b_coefficient,
                pre=pre,
                self_field=self_field,
                ptera_field=placeholder,
                total_rhs=rhs,
                post=post,
                position_storage_pre=position_storage_pre,
                gamma_storage_pre=gamma_storage_pre,
                sigma_storage_pre=sigma_storage_pre,
                position_storage_post=position_storage.copy(),
                gamma_storage_post=gamma_storage.copy(),
                sigma_storage_post=sigma_storage.copy(),
            )
        )
    final = trusted["make_state"](positions, gamma, sigma)
    return FrozenPteraRVPMTransportResult(
        enabled=True,
        interface_id=PTERA_RVPM_TRANSPORT_INTERFACE_ID,
        ptera_step_index=None,
        dvm_for_source_step_index=None,
        source_report_sha256=None,
        relative_epsilon=None,
        epsilon_m=None,
        initial_particle_count=initial.positions.shape[0],
        final_state=final,
        stages=tuple(stages),
        baseline_stages=(),
        self_field_call_count=3,
        ptera_center_call_count=0,
        ptera_finite_difference_call_count=0,
        ptera_parent_state_sha256_before=None,
        ptera_parent_state_sha256_after=None,
        ptera_parent_state_unchanged=True,
        parent_only_bypass=False,
        feedback_write_count=0,
        parent_write_count=0,
        load_write_count=0,
        ptera_field_owner=PTERA_FIELD_OWNER,
        rvpm_transport_owner=RVPM_TRANSPORT_OWNER,
        force_scoring_status=FORCE_SCORING_STATUS,
    )


def _register_report_consumption(
    report: DyadicCumulativeCloudTransportReport, *, ptera_step: int
) -> None:
    identity = id(report)
    with _LOCK:
        existing = _REPORT_CONSUMPTIONS.get(identity)
        if existing is not None:
            live = existing[0]()
            if live is report:
                raise ValueError("v5h2 cloud report was already transported by v5h4")
            if live is None:
                _REPORT_CONSUMPTIONS.pop(identity, None)
            else:
                raise ValueError("v5h4 report identity collision")
        _REPORT_CONSUMPTIONS[identity] = (
            weakref.ref(report),
            report.report_sha256,
            ptera_step,
        )


def transport_live_v5h2_cloud_with_frozen_ptera(
    report: object,
    solver: object,
    *,
    relative_epsilon: object = NOMINAL_RELATIVE_EPSILON,
) -> FrozenPteraRVPMTransportResult:
    """Advance one live v5h2 cloud with one frozen native-Ptera time layer."""

    trusted = _assert_bindings()
    if not isinstance(solver, UVPMHybridSolver):
        raise TypeError("solver must inherit UVPMHybridSolver")
    validated = trusted["validate_report"](report)
    if not validated.enabled:
        raise ValueError("v5h4 transport requires an enabled live v5h2 report")
    ptera_step = _strict_integer("solver._current_step", solver._current_step)
    if validated.for_source_step_index != ptera_step + 1:
        raise ValueError("v5h2 DVM/Ptera step time layer is inconsistent")
    expected_time = ptera_step * float(solver.delta_time)
    if validated.transport_end_time_s != expected_time:
        raise ValueError("v5h2 cloud end time does not match frozen Ptera time")
    relative = _finite_positive("relative_epsilon", relative_epsilon)
    if relative not in PREREGISTERED_RELATIVE_EPSILONS:
        raise ValueError("relative_epsilon is outside the preregistered family")
    state = trusted["materialize_report"](validated)
    initial = trusted["make_state"](state.positions, state.gamma, state.sigma)
    if initial.positions.shape[0] != validated.total_particle_count:
        raise ValueError("live v5h2 cloud count disagrees with report")
    if initial.positions.shape[0] > MAX_PARTICLES:
        raise ValueError("live v5h2 cloud exceeds the v5h4 resource cap")
    epsilon = relative * _ptera_reference_length(solver, initial.sigma)
    _register_report_consumption(validated, ptera_step=ptera_step)
    before = ptera_parent_state_sha256(solver)
    evaluations: list[PteraFieldEvaluation] = []

    def external(points: np.ndarray) -> FrozenExternalField:
        evaluation = evaluate_frozen_parent_ptera_field(
            solver, points, epsilon_m=epsilon
        )
        evaluations.append(evaluation)
        return FrozenExternalField(
            velocity=evaluation.velocity_gp1_m_per_s,
            jacobian=evaluation.jacobian_per_s,
        )

    try:
        generic = lsrk3_step_with_external_field(
            initial,
            float(solver.delta_time),
            external_field=external,
            baseline_freestream_velocity_gp1_m_per_s=np.asarray(
                solver._currentVInf_GP1__E, dtype=np.float64
            ),
            enabled=True,
        )
        after = ptera_parent_state_sha256(solver)
        if after != before:
            raise RuntimeError("v5h4 transport mutated the frozen Ptera parent")
        if len(evaluations) != 3 or len(generic.stages) != 3:
            raise RuntimeError("v5h4 did not execute exactly three LSRK3 stages")
        stages = tuple(
            CoupledLSRKStageRecord(
                stage=stage.stage,
                a=stage.a,
                b=stage.b,
                pre=stage.pre,
                self_field=stage.self_field,
                ptera_field=evaluation,
                total_rhs=stage.total_rhs,
                post=stage.post,
                position_storage_pre=stage.position_storage_pre,
                gamma_storage_pre=stage.gamma_storage_pre,
                sigma_storage_pre=stage.sigma_storage_pre,
                position_storage_post=stage.position_storage_post,
                gamma_storage_post=stage.gamma_storage_post,
                sigma_storage_post=stage.sigma_storage_post,
            )
            for stage, evaluation in zip(generic.stages, evaluations, strict=True)
        )
    except Exception:
        # The report remains consumed.  A failure after reading a live Ptera
        # time layer is fail-stop; no resumability claim is made.
        raise
    return FrozenPteraRVPMTransportResult(
        enabled=True,
        interface_id=PTERA_RVPM_TRANSPORT_INTERFACE_ID,
        ptera_step_index=ptera_step,
        dvm_for_source_step_index=int(validated.for_source_step_index),
        source_report_sha256=validated.report_sha256,
        relative_epsilon=relative,
        epsilon_m=epsilon,
        initial_particle_count=initial.positions.shape[0],
        final_state=generic.final_state,
        stages=stages,
        baseline_stages=(),
        self_field_call_count=3,
        ptera_center_call_count=sum(
            evaluation.center_call_count for evaluation in evaluations
        ),
        ptera_finite_difference_call_count=sum(
            evaluation.finite_difference_call_count for evaluation in evaluations
        ),
        ptera_parent_state_sha256_before=before,
        ptera_parent_state_sha256_after=after,
        ptera_parent_state_unchanged=True,
        parent_only_bypass=True,
        feedback_write_count=0,
        parent_write_count=0,
        load_write_count=0,
        ptera_field_owner=PTERA_FIELD_OWNER,
        rvpm_transport_owner=RVPM_TRANSPORT_OWNER,
        force_scoring_status=FORCE_SCORING_STATUS,
    )


__all__ = [
    "FORCE_SCORING_STATUS",
    "MAX_PARTICLES",
    "NOMINAL_RELATIVE_EPSILON",
    "PREREGISTERED_RELATIVE_EPSILONS",
    "PTERA_FIELD_OWNER",
    "PTERA_RVPM_TRANSPORT_INTERFACE_ID",
    "RVPM_TRANSPORT_OWNER",
    "CoupledLSRKStageRecord",
    "FrozenExternalField",
    "FrozenPteraRVPMTransportResult",
    "PteraFieldEvaluation",
    "evaluate_frozen_parent_ptera_field",
    "lsrk3_step_with_external_field",
    "ptera_parent_state_sha256",
    "transport_live_v5h2_cloud_with_frozen_ptera",
]
