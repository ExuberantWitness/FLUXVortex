"""Native FluxV/Ptera velocity feedback from a transported v5h2 cloud.

This auxiliary adapter keeps :class:`fluxvortex.solver.UVPMHybridSolver` and
therefore Ptera's AIC, bound-circulation, prescribed ring-wake and
Kutta--Joukowski plus unsteady load ledgers intact.  A live-attested v5h2
particle cloud at ``t_n`` contributes only an induced velocity:

* once at Ptera's collocation points before the native bound solve; and
* once at each of Ptera's four native LineVortex-centre load batches.

There is no DVM/rVPM force, pressure, circulation, TE-wake or parent-load
write.  The transported cloud is read-only and is not advanced here.  This is
the preregistered one-way feedback vertical slice, not two-way wake coupling
and not a paper-scoring model.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from numbers import Real
from pathlib import Path
from threading import RLock
from typing import Any, Final, Literal, Sequence
import weakref

import numpy as np

import fluxvortex.rvpm_reference as _rvpm_reference_module
import fluxvortex.solver as _solver_module
from fluxvortex.rvpm_reference import direct_gaussian_erf_velocity_jacobian
from fluxvortex.solver import UVPMHybridSolver

from . import v5h2_dyadic_cumulative_cloud_transport as _dyadic_module
from .v5h2_dyadic_cumulative_cloud_transport import (
    DyadicCumulativeCloudTransportReport,
    materialize_dyadic_cumulative_particle_state,
    validate_dyadic_cumulative_cloud_transport_report,
)


FEEDBACK_INTERFACE_ID: Final = "fluxv-v5h3-native-ptera-rvpm-feedback-v1"
FEEDBACK_OWNER: Final = "ptera-native-aic-and-load-velocity"
SURFACE_LOAD_OWNER: Final = "ptera-native-kj-plus-unsteady"
FORCE_SCORING_STATUS: Final = "blocked_non_target_feedback_mechanics_only"
RHS_RESIDUAL_ATOL: Final = 1.0e-12
DIRECT_REPLAY_ATOL_M_S: Final = 1.0e-13

FeedbackChannel = Literal[
    "collocation_rhs", "load_right", "load_front", "load_left", "load_back"
]


@dataclass(frozen=True, slots=True)
class NativePteraRVPMFeedbackConfig:
    """Frozen controls for the one-way native feedback gate."""

    enabled: bool = False
    expected_wing_id: str = "wing"
    expected_source_family: str = "lev"
    rhs_residual_atol: float = RHS_RESIDUAL_ATOL
    direct_replay_atol_m_s: float = DIRECT_REPLAY_ATOL_M_S

    def __post_init__(self) -> None:
        if type(self.enabled) is not bool:
            raise TypeError("enabled must be a bool")
        if type(self.expected_wing_id) is not str or not self.expected_wing_id:
            raise ValueError("expected_wing_id must be a non-empty string")
        if (
            type(self.expected_source_family) is not str
            or not self.expected_source_family
        ):
            raise ValueError("expected_source_family must be a non-empty string")
        for name, value in (
            ("rhs_residual_atol", self.rhs_residual_atol),
            ("direct_replay_atol_m_s", self.direct_replay_atol_m_s),
        ):
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
                raise TypeError(f"{name} must be a finite positive real")
            if not np.isfinite(float(value)) or float(value) <= 0.0:
                raise ValueError(f"{name} must be a finite positive real")


DEFAULT_FEEDBACK_CONFIG: Final = NativePteraRVPMFeedbackConfig()


@dataclass(frozen=True, slots=True)
class NativePteraRVPMVelocityEvaluation:
    """One exact particle-field evaluation at a native Ptera target batch."""

    channel: FeedbackChannel
    ptera_step_index: int
    target_points_gp1_m: np.ndarray
    induced_velocity_gp1_m_per_s: np.ndarray
    target_sha256: str
    velocity_sha256: str


@dataclass(frozen=True, slots=True)
class NativePteraRVPMFeedbackStepReport:
    """Committed evidence for one active native Ptera feedback step."""

    interface_id: str
    ptera_step_index: int
    dvm_for_source_step_index: int
    source_time_s: float
    feedback_report_sha256: str
    particle_count: int
    collocation_evaluation: NativePteraRVPMVelocityEvaluation
    load_evaluations: tuple[NativePteraRVPMVelocityEvaluation, ...]
    parent_wake_normal: np.ndarray
    feedback_normal: np.ndarray
    combined_wake_normal: np.ndarray
    bound_strengths_m2_s: np.ndarray
    no_penetration_residual: np.ndarray
    no_penetration_max_abs: float
    collocation_evaluation_count: int
    load_leg_evaluation_count: int
    parent_load_call_count: int
    extension_force_write_count: int
    extension_moment_write_count: int
    extension_load_processor_call_count: int
    feedback_owner: str
    surface_load_owner: str
    prescribed_wake: bool
    force_scoring_status: str


@dataclass(slots=True)
class _PendingFeedbackStep:
    report: DyadicCumulativeCloudTransportReport
    positions: np.ndarray
    gamma: np.ndarray
    sigma: np.ndarray
    parent_wake_normal: np.ndarray
    feedback_normal: np.ndarray
    combined_wake_normal: np.ndarray
    collocation_evaluation: NativePteraRVPMVelocityEvaluation
    load_evaluations: list[NativePteraRVPMVelocityEvaluation]
    bound_strengths: np.ndarray | None = None
    no_penetration_residual: np.ndarray | None = None
    no_penetration_max_abs: float | None = None
    parent_load_call_count: int = 0


_CONSUMPTION_LOCK = RLock()
_REPORT_CONSUMPTIONS: dict[int, tuple[weakref.ReferenceType[object], str, int]] = {}

_FROZEN_VALIDATE = validate_dyadic_cumulative_cloud_transport_report
_FROZEN_MATERIALIZE = materialize_dyadic_cumulative_particle_state
_FROZEN_DIRECT_FIELD = direct_gaussian_erf_velocity_jacobian
_FROZEN_PARENT_CLASS = UVPMHybridSolver
_FROZEN_PARENT_METHODS = {
    "wake": UVPMHybridSolver._calculate_wake_wing_influences,
    "solve": UVPMHybridSolver._calculate_vortex_strengths,
    "velocity": UVPMHybridSolver.calculate_solution_velocity,
    "loads": UVPMHybridSolver._calculate_loads,
    "populate": UVPMHybridSolver._populate_next_airplanes_wake,
}


def _module_file_sha256(module: object) -> str:
    path = getattr(module, "__file__", None)
    if type(path) is not str or not path.endswith(".py"):
        raise RuntimeError("v5h3 dependency has no auditable Python source file")
    return sha256(Path(path).read_bytes()).hexdigest()


_FROZEN_DEPENDENCY_SOURCE_HASHES = {
    "dyadic": _module_file_sha256(_dyadic_module),
    "rvpm_reference": _module_file_sha256(_rvpm_reference_module),
    "fluxv_solver": _module_file_sha256(_solver_module),
}


def _assert_frozen_bindings() -> dict[str, Any]:
    if (
        validate_dyadic_cumulative_cloud_transport_report is not _FROZEN_VALIDATE
        or materialize_dyadic_cumulative_particle_state is not _FROZEN_MATERIALIZE
        or direct_gaussian_erf_velocity_jacobian is not _FROZEN_DIRECT_FIELD
        or _dyadic_module.validate_dyadic_cumulative_cloud_transport_report
        is not _FROZEN_VALIDATE
        or _dyadic_module.materialize_dyadic_cumulative_particle_state
        is not _FROZEN_MATERIALIZE
        or _rvpm_reference_module.direct_gaussian_erf_velocity_jacobian
        is not _FROZEN_DIRECT_FIELD
        or _solver_module.UVPMHybridSolver is not _FROZEN_PARENT_CLASS
        or any(
            getattr(UVPMHybridSolver, attribute) is not frozen
            for attribute, frozen in (
                ("_calculate_wake_wing_influences", _FROZEN_PARENT_METHODS["wake"]),
                ("_calculate_vortex_strengths", _FROZEN_PARENT_METHODS["solve"]),
                ("calculate_solution_velocity", _FROZEN_PARENT_METHODS["velocity"]),
                ("_calculate_loads", _FROZEN_PARENT_METHODS["loads"]),
                ("_populate_next_airplanes_wake", _FROZEN_PARENT_METHODS["populate"]),
            )
        )
        or any(
            _module_file_sha256(module) != _FROZEN_DEPENDENCY_SOURCE_HASHES[name]
            for name, module in (
                ("dyadic", _dyadic_module),
                ("rvpm_reference", _rvpm_reference_module),
                ("fluxv_solver", _solver_module),
            )
        )
    ):
        raise ValueError("v5h3 feedback dependency callable was replaced")
    return {
        "validate": _FROZEN_VALIDATE,
        "materialize": _FROZEN_MATERIALIZE,
        "direct_field": _FROZEN_DIRECT_FIELD,
    }


def _array_sha256(array: np.ndarray) -> str:
    value = np.asarray(array)
    digest = sha256()
    digest.update(value.dtype.str.encode("ascii"))
    digest.update(repr(value.shape).encode("ascii"))
    digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _readonly_float64(value: object, *, name: str, ndim: int) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.kind not in "iuf" or array.dtype.kind == "b":
        raise ValueError(f"{name} must have a real numeric dtype")
    result = np.ascontiguousarray(array, dtype=np.float64)
    if result.ndim != ndim or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} has invalid shape or non-finite values")
    result.setflags(write=False)
    return result


def _register_feedback_consumption(
    report: DyadicCumulativeCloudTransportReport,
    *,
    ptera_step_index: int,
) -> None:
    identity = id(report)
    with _CONSUMPTION_LOCK:
        old = _REPORT_CONSUMPTIONS.get(identity)
        if old is not None:
            live = old[0]()
            if live is report:
                raise ValueError("v5h2 feedback report was already consumed")
            if live is None:
                _REPORT_CONSUMPTIONS.pop(identity, None)
            else:
                raise ValueError("feedback report identity collision")
        _REPORT_CONSUMPTIONS[identity] = (
            weakref.ref(report),
            report.report_sha256,
            ptera_step_index,
        )


class NativePteraRVPMFeedbackSolver(UVPMHybridSolver):
    """Original FluxV/Ptera solver with a staged read-only rVPM velocity field."""

    def __init__(
        self,
        unsteady_problem: Any,
        *,
        feedback_config: NativePteraRVPMFeedbackConfig,
        feedback_reports: Sequence[object] = (),
        **uvpm_kwargs: Any,
    ) -> None:
        if not isinstance(feedback_config, NativePteraRVPMFeedbackConfig):
            raise TypeError("feedback_config must be NativePteraRVPMFeedbackConfig")
        if not feedback_config.enabled:
            raise ValueError("disabled feedback must use the exact-parent factory")
        self.feedback_config = feedback_config
        self.v5h3_feedback_step_reports: list[NativePteraRVPMFeedbackStepReport] = []
        self._v5h3_feedback_by_step: dict[
            int, DyadicCumulativeCloudTransportReport
        ] = {}
        self._v5h3_pending: _PendingFeedbackStep | None = None
        self._v5h3_phase: Literal["idle", "loads"] = "idle"
        self._v5h3_poisoned = False
        self._v5h3_poison_reason: str | None = None
        super().__init__(unsteady_problem, **uvpm_kwargs)
        self._index_feedback_reports(feedback_reports)

    @property
    def v5h3_poisoned(self) -> bool:
        return self._v5h3_poisoned

    def _poison(self, error: BaseException) -> None:
        self._v5h3_poisoned = True
        self._v5h3_poison_reason = f"{type(error).__name__}: {error}"

    def _raise_if_poisoned(self) -> None:
        if self._v5h3_poisoned:
            raise RuntimeError(
                "v5h3 feedback solver is poisoned after extension failure: "
                f"{self._v5h3_poison_reason}"
            )

    def _index_feedback_reports(self, reports: Sequence[object]) -> None:
        trusted = _assert_frozen_bindings()
        for candidate in tuple(reports):
            report = trusted["validate"](candidate)
            if not report.enabled:
                raise ValueError("feedback sequence cannot contain a disabled report")
            if report.wing_id != self.feedback_config.expected_wing_id:
                raise ValueError("feedback report wing_id does not match configuration")
            if report.source_family != self.feedback_config.expected_source_family:
                raise ValueError(
                    "feedback report source_family does not match configuration"
                )
            if (
                report.feedback_call_count
                or report.parent_write_count
                or report.load_write_count
            ):
                raise ValueError(
                    "feedback report already contains a forbidden owner write"
                )
            dvm_step = report.for_source_step_index
            if type(dvm_step) is not int or dvm_step < 2:
                raise ValueError("feedback report must target DVM source step >= 2")
            ptera_step = dvm_step - 1
            if ptera_step <= 0 or ptera_step >= self.num_steps:
                raise ValueError("feedback report is outside the Ptera step range")
            expected_time = ptera_step * float(self.delta_time)
            if report.transport_end_time_s != expected_time:
                raise ValueError("feedback report has the wrong Ptera time layer")
            if ptera_step in self._v5h3_feedback_by_step:
                raise ValueError("duplicate feedback report for one Ptera step")
            self._v5h3_feedback_by_step[ptera_step] = report
        steps = tuple(sorted(self._v5h3_feedback_by_step))
        if steps and steps != tuple(range(1, steps[-1] + 1)):
            raise ValueError("feedback reports must form a contiguous step-one prefix")
        previous: DyadicCumulativeCloudTransportReport | None = None
        for ptera_step in steps:
            report = self._v5h3_feedback_by_step[ptera_step]
            if report.parent_source_step_index != ptera_step:
                raise ValueError("feedback report parent source step is inconsistent")
            if report.transport_start_time_s != (ptera_step - 1) * float(
                self.delta_time
            ):
                raise ValueError("feedback report start time is inconsistent")
            if previous is None:
                if report.parent_report_sha256 is not None:
                    raise ValueError("first feedback report unexpectedly has a parent")
            else:
                if report.parent_report_sha256 != previous.report_sha256:
                    raise ValueError("feedback report parent chain is discontinuous")
                if (
                    report.smoothing_radius_m != previous.smoothing_radius_m
                    or report.base_target_spacing_m != previous.base_target_spacing_m
                    or report.refinement_level != previous.refinement_level
                ):
                    raise ValueError(
                        "feedback cloud discretization changed across steps"
                    )
            previous = report

    def _field_velocity(
        self,
        points: object,
        *,
        channel: FeedbackChannel,
    ) -> NativePteraRVPMVelocityEvaluation:
        pending = self._v5h3_pending
        if pending is None:
            raise RuntimeError("feedback field was requested without staged cloud")
        targets = _readonly_float64(points, name="Ptera feedback targets", ndim=2)
        if targets.shape[1:] != (3,):
            raise ValueError("Ptera feedback targets must have shape (N,3)")
        field = _assert_frozen_bindings()["direct_field"](
            pending.positions,
            pending.gamma,
            pending.sigma,
            target_positions=targets,
        )
        velocity = _readonly_float64(
            field.velocity,
            name="rVPM feedback velocity",
            ndim=2,
        )
        if velocity.shape != targets.shape:
            raise RuntimeError("rVPM feedback velocity shape mismatch")
        return NativePteraRVPMVelocityEvaluation(
            channel=channel,
            ptera_step_index=int(self._current_step),
            target_points_gp1_m=targets,
            induced_velocity_gp1_m_per_s=velocity,
            target_sha256=_array_sha256(targets),
            velocity_sha256=_array_sha256(velocity),
        )

    def _stage_feedback(self) -> None:
        report = self._v5h3_feedback_by_step.get(int(self._current_step))
        if report is None:
            return
        if self._v5h3_pending is not None:
            raise RuntimeError("a previous v5h3 feedback step was not committed")
        if self.current_operating_point.surfaceReflect_T_act_GP1_CgP1 is not None:
            raise ValueError(
                "v5h3 first-stage feedback does not support image surfaces"
            )
        trusted = _assert_frozen_bindings()
        validated = trusted["validate"](report)
        state = trusted["materialize"](validated)
        positions = _readonly_float64(state.positions, name="cloud positions", ndim=2)
        gamma = _readonly_float64(state.gamma, name="cloud gamma", ndim=2)
        sigma = _readonly_float64(state.sigma, name="cloud sigma", ndim=1)
        if positions.shape[1:] != (3,) or gamma.shape != positions.shape:
            raise ValueError("v5h2 cloud has an invalid particle shape")
        if sigma.shape != (positions.shape[0],) or np.any(sigma <= 0.0):
            raise ValueError("v5h2 cloud has invalid smoothing radii")
        if positions.shape[0] != validated.total_particle_count:
            raise ValueError("v5h2 cloud count disagrees with its report")
        _register_feedback_consumption(
            validated,
            ptera_step_index=int(self._current_step),
        )
        parent = np.asarray(
            self._currentStackWakeWingInfluences__E, dtype=np.float64
        ).copy()
        # Temporarily create the pending state so _field_velocity uses exactly the
        # same frozen arrays as later native load evaluations.
        placeholder = np.empty((0, 3), dtype=np.float64)
        placeholder.setflags(write=False)
        dummy = NativePteraRVPMVelocityEvaluation(
            channel="collocation_rhs",
            ptera_step_index=int(self._current_step),
            target_points_gp1_m=placeholder,
            induced_velocity_gp1_m_per_s=placeholder,
            target_sha256=_array_sha256(placeholder),
            velocity_sha256=_array_sha256(placeholder),
        )
        self._v5h3_pending = _PendingFeedbackStep(
            report=validated,
            positions=positions,
            gamma=gamma,
            sigma=sigma,
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
            raise FloatingPointError("combined v5h3 wake-side RHS is non-finite")
        self._v5h3_pending.collocation_evaluation = evaluation
        self._v5h3_pending.feedback_normal = feedback_normal.copy()
        self._v5h3_pending.combined_wake_normal = combined.copy()
        self._currentStackWakeWingInfluences__E = combined

    def _calculate_wake_wing_influences(self) -> None:
        self._raise_if_poisoned()
        if int(self._current_step) in self._v5h3_feedback_by_step:
            try:
                _assert_frozen_bindings()
            except Exception as error:
                self._poison(error)
                raise
        super()._calculate_wake_wing_influences()
        try:
            self._stage_feedback()
        except Exception as error:
            self._poison(error)
            raise

    def _calculate_vortex_strengths(self) -> None:
        self._raise_if_poisoned()
        try:
            if self._v5h3_pending is not None:
                _assert_frozen_bindings()
            super()._calculate_vortex_strengths()
            pending = self._v5h3_pending
            if pending is None:
                return
            gamma = np.asarray(
                self._current_bound_vortex_strengths, dtype=np.float64
            ).copy()
            residual = (
                np.asarray(self._currentGridWingWingInfluences__E, dtype=np.float64)
                @ gamma
                + pending.combined_wake_normal
                + np.asarray(
                    self._currentStackFreestreamWingInfluences__E, dtype=np.float64
                )
            )
            maximum = float(np.max(np.abs(residual), initial=0.0))
            if not np.isfinite(maximum) or maximum > float(
                self.feedback_config.rhs_residual_atol
            ):
                raise FloatingPointError("v5h3 native no-penetration residual failed")
            pending.bound_strengths = gamma
            pending.no_penetration_residual = residual.copy()
            pending.no_penetration_max_abs = maximum
        except Exception as error:
            self._poison(error)
            raise

    def calculate_solution_velocity(
        self,
        stackP_GP1_CgP1: np.ndarray | Sequence[Sequence[float | int]],
        bound_singularity_counts: np.ndarray | None = None,
        wake_singularity_counts: np.ndarray | None = None,
    ) -> np.ndarray:
        if self._v5h3_pending is not None:
            _assert_frozen_bindings()
        parent = super().calculate_solution_velocity(
            stackP_GP1_CgP1,
            bound_singularity_counts=bound_singularity_counts,
            wake_singularity_counts=wake_singularity_counts,
        )
        pending = self._v5h3_pending
        if pending is None:
            return parent
        if self._v5h3_phase != "loads":
            raise RuntimeError("active v5h3 cloud was queried outside native loads")
        channels: tuple[FeedbackChannel, ...] = (
            "load_right",
            "load_front",
            "load_left",
            "load_back",
        )
        index = len(pending.load_evaluations)
        if index >= len(channels):
            raise RuntimeError("Ptera requested more than four feedback load batches")
        evaluation = self._field_velocity(stackP_GP1_CgP1, channel=channels[index])
        pending.load_evaluations.append(evaluation)
        result = (
            np.asarray(parent, dtype=np.float64)
            + evaluation.induced_velocity_gp1_m_per_s
        )
        if not np.all(np.isfinite(result)):
            raise FloatingPointError("v5h3 augmented solution velocity is non-finite")
        return result

    def _calculate_loads(self) -> None:
        self._raise_if_poisoned()
        pending = self._v5h3_pending
        if pending is None:
            return super()._calculate_loads()
        try:
            _assert_frozen_bindings()
            self._v5h3_phase = "loads"
            super()._calculate_loads()
            pending.parent_load_call_count += 1
        except Exception as error:
            self._poison(error)
            raise
        finally:
            self._v5h3_phase = "idle"
        expected = (
            self.stackCblvpr_GP1_CgP1,
            self.stackCblvpf_GP1_CgP1,
            self.stackCblvpl_GP1_CgP1,
            self.stackCblvpb_GP1_CgP1,
        )
        if len(pending.load_evaluations) != 4:
            error = RuntimeError(
                "Ptera did not consume four native feedback load batches"
            )
            self._poison(error)
            raise error
        for evaluation, points in zip(pending.load_evaluations, expected, strict=True):
            if not np.array_equal(evaluation.target_points_gp1_m, np.asarray(points)):
                error = RuntimeError("native Ptera load target order changed")
                self._poison(error)
                raise error

    def _populate_next_airplanes_wake(self) -> None:
        self._raise_if_poisoned()
        pending = self._v5h3_pending
        if pending is None:
            return super()._populate_next_airplanes_wake()
        try:
            _assert_frozen_bindings()
            super()._populate_next_airplanes_wake()
            if (
                pending.bound_strengths is None
                or pending.no_penetration_residual is None
                or pending.no_penetration_max_abs is None
                or pending.parent_load_call_count != 1
                or len(pending.load_evaluations) != 4
            ):
                raise RuntimeError("v5h3 feedback step did not close its native ledger")
            if not bool(self._prescribed_wake):
                raise RuntimeError("v5h3 first-stage gate requires prescribed wake")
            self.v5h3_feedback_step_reports.append(
                NativePteraRVPMFeedbackStepReport(
                    interface_id=FEEDBACK_INTERFACE_ID,
                    ptera_step_index=int(self._current_step),
                    dvm_for_source_step_index=int(pending.report.for_source_step_index),
                    source_time_s=float(pending.report.transport_end_time_s),
                    feedback_report_sha256=pending.report.report_sha256,
                    particle_count=pending.positions.shape[0],
                    collocation_evaluation=pending.collocation_evaluation,
                    load_evaluations=tuple(pending.load_evaluations),
                    parent_wake_normal=pending.parent_wake_normal.copy(),
                    feedback_normal=pending.feedback_normal.copy(),
                    combined_wake_normal=pending.combined_wake_normal.copy(),
                    bound_strengths_m2_s=pending.bound_strengths.copy(),
                    no_penetration_residual=pending.no_penetration_residual.copy(),
                    no_penetration_max_abs=pending.no_penetration_max_abs,
                    collocation_evaluation_count=1,
                    load_leg_evaluation_count=4,
                    parent_load_call_count=1,
                    extension_force_write_count=0,
                    extension_moment_write_count=0,
                    extension_load_processor_call_count=0,
                    feedback_owner=FEEDBACK_OWNER,
                    surface_load_owner=SURFACE_LOAD_OWNER,
                    prescribed_wake=True,
                    force_scoring_status=FORCE_SCORING_STATUS,
                )
            )
            self._v5h3_pending = None
        except Exception as error:
            self._poison(error)
            raise


def make_fluxv_v5h3_native_feedback_solver(
    unsteady_problem: Any,
    *,
    config: NativePteraRVPMFeedbackConfig = DEFAULT_FEEDBACK_CONFIG,
    feedback_reports: Sequence[object] = (),
    **uvpm_kwargs: Any,
) -> UVPMHybridSolver:
    """Return the exact FluxV parent or the one-way native feedback subclass."""

    if not isinstance(config, NativePteraRVPMFeedbackConfig):
        raise TypeError("config must be NativePteraRVPMFeedbackConfig")
    if not config.enabled:
        # Deliberately input-blind: disabled construction never inspects reports.
        return UVPMHybridSolver(unsteady_problem, **uvpm_kwargs)
    return NativePteraRVPMFeedbackSolver(
        unsteady_problem,
        feedback_config=config,
        feedback_reports=feedback_reports,
        **uvpm_kwargs,
    )


__all__ = [
    "DEFAULT_FEEDBACK_CONFIG",
    "DIRECT_REPLAY_ATOL_M_S",
    "FEEDBACK_INTERFACE_ID",
    "FEEDBACK_OWNER",
    "FORCE_SCORING_STATUS",
    "NativePteraRVPMFeedbackConfig",
    "NativePteraRVPMFeedbackSolver",
    "NativePteraRVPMFeedbackStepReport",
    "NativePteraRVPMVelocityEvaluation",
    "RHS_RESIDUAL_ATOL",
    "SURFACE_LOAD_OWNER",
    "make_fluxv_v5h3_native_feedback_solver",
]
