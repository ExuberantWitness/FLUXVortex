"""Bounded Baik-W2 global-row/native-Ptera/rVPM coupling slice.

This module is deliberately a raw, observation-free development diagnostic.
It does not construct DVM source events, own a surface load, score a candidate,
or write an artifact.  Ptera remains the only surface-load owner.  A caller
supplies a source-step callback which must commit one exact live global row at
the active Ptera time layer.  After that commit the row is exposed once to
native Ptera collocation and its four native load batches, then advanced by
exactly one combined three-stage rVPM transport.

The callback is invoked lazily: the second and third row cannot be proposed
until the preceding transported state exists.  This prevents three unrelated
zero-time row snapshots from masquerading as a continuous physical history.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from numbers import Real
from typing import Any, Final, Protocol

import numpy as np

from fluxvortex.rvpm_transport import ParticleState, make_particle_state
from .fluxv_v5h3_native_feedback import (
    SURFACE_LOAD_OWNER,
    NativePteraRVPMFeedbackConfig,
    NativePteraRVPMFeedbackSolver,
    NativePteraRVPMFeedbackStepReport,
    NativePteraRVPMVelocityEvaluation,
    _PendingFeedbackStep,
    _assert_frozen_bindings,
    _array_sha256,
)
from .fluxv_v5h4_ptera_rvpm_transport import (
    MAX_PARTICLES,
    NOMINAL_RELATIVE_EPSILON,
    PREREGISTERED_RELATIVE_EPSILONS,
    FrozenExternalField,
    FrozenPteraRVPMTransportResult,
    PteraFieldEvaluation,
    evaluate_frozen_parent_ptera_field,
    lsrk3_step_with_external_field,
    ptera_parent_state_sha256,
    _assert_bindings as _assert_v5h4_bindings,
)
from . import fluxv_v5h10_row_owner as _row_owner_module
from .fluxv_v5h10_row_owner import (
    ReleaseRowCommonTransport,
    ReleaseRowOwner,
    ReleaseRowTransportAttestation,
    RowCommitResult,
    advance_release_row_transport_parent,
    attest_release_row_common_transport,
    begin_release_row_common_transport,
    release_row_transport_digest,
    validate_current_release_row_owner,
    validate_release_row_common_transport,
    validate_release_row_transport_attestation,
)


COUPLING_INTERFACE_ID: Final = "fluxv-v5h10-baik-w2-global-row-coupling-v1"
CASE_ID: Final = "W2"
OBSERVATION_ACCESS: Final = "none"
SURFACE_FORCE_OWNER: Final = SURFACE_LOAD_OWNER
SOURCE_LOAD_OWNER: Final = "forbidden"
FORCE_SCORING_STATUS: Final = "blocked_raw_three_layer_diagnostic_only"
PTERA_CHORDWISE_PANELS: Final = 2
PTERA_SPANWISE_PANELS: Final = 8
PTERA_STEPS_PER_CYCLE: Final = 32
PTERA_CYCLES: Final = 2
PTERA_DELTA_TIME_S: Final = 0.11125
CONVECTIVE_DELTA_TIME: Final = 0.09817477042468103
ACTIVE_PTERA_STEPS: Final = (3, 4, 5)
ACTIVE_SOURCE_STEPS: Final = (4, 5, 6)
ACTIVE_BIRTH_MODES: Final = ("first", "continuous", "continuous")
MAX_ACTIVE_LAYERS: Final = 3
NO_PENETRATION_ATOL: Final = 1.0e-12


def _exact_int(name: str, value: object, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an exact integer >= {minimum}")
    return value


def _finite_float(name: str, value: object, *, positive: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite real")
    result = float(value)
    if not np.isfinite(result) or (positive and result <= 0.0):
        qualifier = " positive" if positive else ""
        raise ValueError(f"{name} must be a finite{qualifier} real")
    return result


def _readonly_float64(name: str, value: object, *, ndim: int) -> np.ndarray:
    original = np.asarray(value)
    if original.dtype.kind not in "iuf" or original.dtype.kind == "b":
        raise ValueError(f"{name} must have a real numeric dtype")
    # Always detach before changing writeability.  ``ascontiguousarray`` may
    # alias an already-C-contiguous parent Ptera array and would then make the
    # parent's force/load storage read-only.
    result = np.array(original, dtype=np.float64, order="C", copy=True)
    if result.ndim != ndim or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} has invalid shape or non-finite values")
    result.setflags(write=False)
    return result


def _particle_state_sha256(state: ParticleState) -> str:
    digest = sha256()
    for name, array in (
        ("positions", state.positions),
        ("gamma", state.gamma),
        ("sigma", state.sigma),
    ):
        value = np.asarray(array)
        digest.update(name.encode("ascii"))
        digest.update(value.dtype.str.encode("ascii"))
        digest.update(repr(value.shape).encode("ascii"))
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class BaikW2CouplingConfig:
    """Frozen one-shot settings from the W2 preregistration."""

    case_id: str = CASE_ID
    chordwise_panels: int = PTERA_CHORDWISE_PANELS
    spanwise_panels: int = PTERA_SPANWISE_PANELS
    steps_per_cycle: int = PTERA_STEPS_PER_CYCLE
    cycles: int = PTERA_CYCLES
    delta_time_s: float = PTERA_DELTA_TIME_S
    convective_delta_time: float = CONVECTIVE_DELTA_TIME
    active_ptera_steps: tuple[int, ...] = ACTIVE_PTERA_STEPS
    active_source_steps: tuple[int, ...] = ACTIVE_SOURCE_STEPS
    active_birth_modes: tuple[str, ...] = ACTIVE_BIRTH_MODES
    relative_epsilon: float = NOMINAL_RELATIVE_EPSILON
    particle_cap: int = MAX_PARTICLES
    no_penetration_atol: float = NO_PENETRATION_ATOL

    def __post_init__(self) -> None:
        frozen = (
            self.case_id == CASE_ID
            and type(self.chordwise_panels) is int
            and self.chordwise_panels == PTERA_CHORDWISE_PANELS
            and type(self.spanwise_panels) is int
            and self.spanwise_panels == PTERA_SPANWISE_PANELS
            and type(self.steps_per_cycle) is int
            and self.steps_per_cycle == PTERA_STEPS_PER_CYCLE
            and type(self.cycles) is int
            and self.cycles == PTERA_CYCLES
            and type(self.delta_time_s) is float
            and self.delta_time_s == PTERA_DELTA_TIME_S
            and type(self.convective_delta_time) is float
            and self.convective_delta_time == CONVECTIVE_DELTA_TIME
            and type(self.active_ptera_steps) is tuple
            and self.active_ptera_steps == ACTIVE_PTERA_STEPS
            and type(self.active_source_steps) is tuple
            and self.active_source_steps == ACTIVE_SOURCE_STEPS
            and type(self.active_birth_modes) is tuple
            and self.active_birth_modes == ACTIVE_BIRTH_MODES
        )
        if not frozen:
            raise ValueError("Baik W2 paper-case settings are preregistered and frozen")
        relative = _finite_float(
            "relative_epsilon", self.relative_epsilon, positive=True
        )
        if relative not in PREREGISTERED_RELATIVE_EPSILONS:
            raise ValueError("relative_epsilon is outside the preregistered family")
        _exact_int("particle_cap", self.particle_cap, minimum=1)
        if self.particle_cap > MAX_PARTICLES:
            raise ValueError("particle_cap exceeds the audited v5h4 resource cap")
        _finite_float("no_penetration_atol", self.no_penetration_atol, positive=True)
        if self.no_penetration_atol > NO_PENETRATION_ATOL:
            raise ValueError("no_penetration_atol cannot relax the frozen gate")


DEFAULT_CONFIG: Final = BaikW2CouplingConfig()


@dataclass(frozen=True, slots=True)
class GlobalRowCommitRequest:
    """The only input delivered to the lazy global-row commit callback."""

    ptera_step_index: int
    source_step_index: int
    source_time_s: float
    expected_birth_mode: str
    previous_owner: ReleaseRowOwner | None
    transported_parent: ParticleState | None
    transported_parent_sha256: str | None


class GlobalRowCommitter(Protocol):
    """Lazy source/row boundary; implementations may not own Ptera loads."""

    def __call__(self, request: GlobalRowCommitRequest, /) -> object:
        ...


@dataclass(frozen=True, slots=True)
class BaikW2RawStepRecord:
    """In-memory raw evidence for one completed active W2 layer."""

    interface_id: str
    case_id: str
    ptera_step_index: int
    source_step_index: int
    source_time_s: float
    birth_mode: str
    row_commit_sha256: str
    row_state_sha256: str
    transported_parent_sha256: str | None
    transported_state_sha256: str
    transported_arrays_sha256: str
    transport_parent_digest: str
    common_transport_sha256: str
    common_transport_attestation_sha256: str
    transported_material_tracers_sha256: str
    advanced_owner_sha256: str
    transported_live_boundary_nodes: np.ndarray
    material_tracer_count: int
    material_support_tracer_count: int
    frontier_node_tracer_count: int
    particle_count: int
    feedback_report: NativePteraRVPMFeedbackStepReport
    transport_result: FrozenPteraRVPMTransportResult
    ptera_forces_w: np.ndarray
    ptera_forces_w_sha256: str
    ptera_force_coefficients_w: np.ndarray
    ptera_force_coefficients_w_sha256: str
    ptera_moments_w_cgp1: np.ndarray
    ptera_moments_w_cgp1_sha256: str
    ptera_moment_coefficients_w: np.ndarray
    ptera_moment_coefficients_w_sha256: str
    ptera_panel_ids: tuple[str, ...]
    ptera_panel_forces_w: np.ndarray
    ptera_panel_forces_w_sha256: str
    ptera_panel_moments_w_cgp1: np.ndarray
    ptera_panel_moments_w_cgp1_sha256: str
    ptera_panel_force_sum_w: np.ndarray
    ptera_panel_moment_sum_w_cgp1: np.ndarray
    ptera_panel_force_sum_max_abs_residual: float
    ptera_panel_moment_sum_max_abs_residual: float
    ptera_panel_force_sum_atol: float
    ptera_panel_moment_sum_atol: float
    collocation_evaluation_count: int
    load_batch_evaluation_count: int
    native_load_call_count: int
    transport_call_count: int
    transport_stage_count: int
    common_transport_self_field_call_count: int
    common_transport_ptera_center_call_count: int
    common_transport_stage_count: int
    frontier_self_field_call_count: int
    frontier_ptera_center_call_count: int
    frontier_transport_stage_count: int
    ptera_parent_sha256_before_transport: str
    ptera_parent_sha256_after_transport: str
    ptera_parent_state_unchanged: bool
    ptera_parent_sha256_after_raw_record: str
    ptera_parent_state_unchanged_after_raw_record: bool
    observation_access: str
    surface_load_owner: str
    source_load_owner: str
    force_scoring_status: str


class _BoundedSliceComplete(RuntimeError):
    pass


class GlobalRowPteraRVPMCouplingSolver(NativePteraRVPMFeedbackSolver):
    """Run native Ptera from step zero through the third active W2 layer."""

    def __init__(
        self,
        unsteady_problem: Any,
        *,
        row_committer: GlobalRowCommitter,
        config: BaikW2CouplingConfig = DEFAULT_CONFIG,
        **uvpm_kwargs: Any,
    ) -> None:
        if type(config) is not BaikW2CouplingConfig:
            raise TypeError("config must be an exact BaikW2CouplingConfig")
        if not callable(row_committer):
            raise TypeError("row_committer must be callable")
        self.v5h10_config = config
        self._v5h10_row_committer = row_committer
        self._v5h10_previous_owner: ReleaseRowOwner | None = None
        self._v5h10_transported_parent: ParticleState | None = None
        self._v5h10_active_row: ReleaseRowOwner | None = None
        self._v5h10_active_row_state: ParticleState | None = None
        self._v5h10_active_row_digest: str | None = None
        self._v5h10_active_state_digest: str | None = None
        self._v5h10_slice_complete = False
        self.v5h10_raw_step_records: list[BaikW2RawStepRecord] = []
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
        if self.num_steps != config.steps_per_cycle * config.cycles:
            raise ValueError("v5h10 requires the frozen two-cycle W2 movement")
        if float(self.delta_time) != config.delta_time_s:
            raise ValueError("v5h10 W2 movement delta_time is not frozen")
        if self.num_panels != config.chordwise_panels * config.spanwise_panels:
            raise ValueError("v5h10 W2 movement panel grid is not frozen 2x8")
        if self.first_results_step != 0:
            raise ValueError("v5h10 raw slice requires native loads at every step")

    @property
    def v5h10_slice_complete(self) -> bool:
        return self._v5h10_slice_complete

    def _row_api(self) -> tuple[Any, ...]:
        """Return the frozen exact live-owner and transported-parent API."""

        if (
            _row_owner_module.RowCommitResult is not RowCommitResult
            or _row_owner_module.ReleaseRowOwner is not ReleaseRowOwner
            or _row_owner_module.validate_current_release_row_owner
            is not validate_current_release_row_owner
            or _row_owner_module.ReleaseRowCommonTransport
            is not ReleaseRowCommonTransport
            or _row_owner_module.ReleaseRowTransportAttestation
            is not ReleaseRowTransportAttestation
            or _row_owner_module.validate_release_row_common_transport
            is not validate_release_row_common_transport
            or _row_owner_module.validate_release_row_transport_attestation
            is not validate_release_row_transport_attestation
            or _row_owner_module.begin_release_row_common_transport
            is not begin_release_row_common_transport
            or _row_owner_module.attest_release_row_common_transport
            is not attest_release_row_common_transport
            or _row_owner_module.release_row_transport_digest
            is not release_row_transport_digest
            or _row_owner_module.advance_release_row_transport_parent
            is not advance_release_row_transport_parent
        ):
            raise RuntimeError("v5h10 global-row owner API binding was replaced")
        return (
            RowCommitResult,
            ReleaseRowOwner,
            ReleaseRowCommonTransport,
            ReleaseRowTransportAttestation,
            validate_current_release_row_owner,
            validate_release_row_common_transport,
            validate_release_row_transport_attestation,
            begin_release_row_common_transport,
            attest_release_row_common_transport,
            release_row_transport_digest,
            advance_release_row_transport_parent,
        )

    def _prepare_active_row(self) -> None:
        """Commit and validate the active row before any Ptera field call."""

        step = int(self._current_step)
        if step not in self.v5h10_config.active_ptera_steps:
            if self._v5h3_pending is not None or self._v5h10_active_row is not None:
                raise RuntimeError("v5h10 active layer leaked into an inactive step")
            return
        if self._v5h10_slice_complete:
            raise RuntimeError("v5h10 active slice was already completed")
        active_index = self.v5h10_config.active_ptera_steps.index(step)
        source_step = self.v5h10_config.active_source_steps[active_index]
        birth_mode = self.v5h10_config.active_birth_modes[active_index]
        parent = self._v5h10_transported_parent
        parent_digest = None if parent is None else _particle_state_sha256(parent)
        if active_index == 0:
            if self._v5h10_previous_owner is not None or parent is not None:
                raise RuntimeError("first v5h10 layer unexpectedly has a parent")
        elif self._v5h10_previous_owner is None or parent is None:
            raise RuntimeError("continuous v5h10 layer lacks its transported parent")
        request = GlobalRowCommitRequest(
            ptera_step_index=step,
            source_step_index=source_step,
            source_time_s=step * float(self.delta_time),
            expected_birth_mode=birth_mode,
            previous_owner=self._v5h10_previous_owner,
            transported_parent=parent,
            transported_parent_sha256=parent_digest,
        )
        candidate = self._v5h10_row_committer(request)
        result_type, owner_type, _, _, validate, *unused = self._row_api()
        del unused
        if type(candidate) is not result_type:
            raise TypeError("row_committer returned a foreign commit-result schema")
        if (
            not candidate.committed
            or candidate.status != "compatible"
            or candidate.first_mismatch is not None
        ):
            raise RuntimeError("row callback did not produce a compatible commit")
        committed = validate(candidate.owner)
        if committed is not candidate.owner:
            raise RuntimeError("row validator did not preserve live identity")
        if type(committed) is not owner_type or candidate.state is not committed.state:
            raise RuntimeError("row commit result is not bound to its live owner")
        if active_index == 0:
            if candidate.event is not None or committed.events:
                raise RuntimeError("bootstrap row has an unexpected update event")
        elif not committed.events or candidate.event is not committed.events[-1]:
            raise RuntimeError("row update result is not bound to its owner event")
        state = committed.state
        if state.phase != "post_commit_pre_transport":
            raise RuntimeError("row callback did not return a newly committed row")
        if (
            state.release_index != active_index + 1
            or committed.epoch != active_index + 1
            or state.state_epoch != committed.epoch
            or state.rows[-1].source_time_s != request.source_time_s
        ):
            raise RuntimeError("committed row release/time/epoch is inconsistent")
        if (
            state.clone_count
            or state.counter_particle_count
            or state.fresh_upstream_particle_count
        ):
            raise RuntimeError("committed row contains forbidden physical particles")
        previous_owner = request.previous_owner
        if previous_owner is None:
            if committed.transport_events:
                raise RuntimeError("first committed row has an unexpected transport")
        elif (
            committed.owner_id != previous_owner.owner_id
            or committed.sheet_id != previous_owner.sheet_id
            or committed.epoch != previous_owner.epoch + 1
            or committed.events[-1].parent_owner_sha256 != previous_owner.owner_sha256
            or len(committed.transport_events) != len(previous_owner.transport_events)
            or any(
                current is not prior
                for current, prior in zip(
                    committed.transport_events,
                    previous_owner.transport_events,
                    strict=True,
                )
            )
        ):
            raise RuntimeError("committed row is not descended from transported owner")
        frozen = make_particle_state(state.positions, state.gamma, state.sigma)
        if frozen.positions.shape[0] > self.v5h10_config.particle_cap:
            raise ValueError("committed W2 row exceeds the frozen particle cap")
        if frozen.positions.shape[0] == 0:
            raise ValueError("committed W2 row cannot be empty")
        if not (
            np.all(np.isfinite(frozen.positions))
            and np.all(np.isfinite(frozen.gamma))
            and np.all(np.isfinite(frozen.sigma))
            and np.all(frozen.sigma > 0.0)
        ):
            raise ValueError("committed W2 row has invalid physical arrays")
        positions = _readonly_float64("row positions", frozen.positions, ndim=2)
        gamma = _readonly_float64("row gamma", frozen.gamma, ndim=2)
        sigma = _readonly_float64("row sigma", frozen.sigma, ndim=1)
        if positions.shape[1:] != (3,) or gamma.shape != positions.shape:
            raise ValueError("committed W2 row has invalid vector shape")
        if sigma.shape != (positions.shape[0],):
            raise ValueError("committed W2 row has invalid sigma shape")
        row_digest = committed.owner_sha256
        state_digest = state.state_sha256
        if (
            type(row_digest) is not str
            or len(row_digest) != 64
            or type(state_digest) is not str
            or len(state_digest) != 64
        ):
            raise ValueError("committed W2 row lacks exact SHA-256 provenance")
        prepared = make_particle_state(positions.copy(), gamma.copy(), sigma.copy())
        prepared.positions.setflags(write=False)
        prepared.gamma.setflags(write=False)
        prepared.sigma.setflags(write=False)
        self._v5h10_active_row = committed
        self._v5h10_active_row_state = prepared
        self._v5h10_active_row_digest = row_digest
        self._v5h10_active_state_digest = state_digest

    def _stage_feedback(self) -> None:
        """Evaluate collocation feedback only from a prevalidated live row."""

        step = int(self._current_step)
        if step not in self.v5h10_config.active_ptera_steps:
            if self._v5h3_pending is not None or self._v5h10_active_row is not None:
                raise RuntimeError("v5h10 active layer leaked into an inactive step")
            return
        committed = self._v5h10_active_row
        prepared = self._v5h10_active_row_state
        row_digest = self._v5h10_active_row_digest
        if (
            committed is None
            or prepared is None
            or row_digest is None
            or self._v5h10_active_state_digest is None
        ):
            raise RuntimeError("active Ptera feedback lacks a prevalidated row")
        if self._v5h3_pending is not None:
            raise RuntimeError("a previous v5h10 feedback layer is still pending")
        active_index = self.v5h10_config.active_ptera_steps.index(step)
        source_step = self.v5h10_config.active_source_steps[active_index]
        positions = prepared.positions
        gamma = prepared.gamma
        sigma = prepared.sigma
        empty = np.empty((0, 3), dtype=np.float64)
        empty.setflags(write=False)
        dummy = NativePteraRVPMVelocityEvaluation(
            channel="collocation_rhs",
            ptera_step_index=step,
            target_points_gp1_m=empty,
            induced_velocity_gp1_m_per_s=empty,
            target_sha256=_array_sha256(empty),
            velocity_sha256=_array_sha256(empty),
        )
        row_view = _FeedbackRowView(
            for_source_step_index=source_step,
            transport_end_time_s=committed.state.rows[-1].source_time_s,
            report_sha256=row_digest,
        )
        parent_normal = np.asarray(
            self._currentStackWakeWingInfluences__E, dtype=np.float64
        ).copy()
        self._v5h3_pending = _PendingFeedbackStep(
            report=row_view,  # type: ignore[arg-type]
            positions=positions,
            gamma=gamma,
            sigma=sigma,
            parent_wake_normal=parent_normal,
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
        combined = parent_normal + feedback_normal
        if not np.all(np.isfinite(combined)):
            raise FloatingPointError("v5h10 combined Ptera RHS is non-finite")
        self._v5h3_pending.collocation_evaluation = evaluation
        self._v5h3_pending.feedback_normal = feedback_normal.copy()
        self._v5h3_pending.combined_wake_normal = combined.copy()
        self._currentStackWakeWingInfluences__E = combined

    def _calculate_wake_wing_influences(self) -> None:
        """Prepare the row before inherited parent-wake/collocation work."""

        self._raise_if_poisoned()
        try:
            self._prepare_active_row()
        except Exception as error:
            self._poison(error)
            raise
        super()._calculate_wake_wing_influences()

    def _transport_active_row(
        self,
    ) -> tuple[
        FrozenPteraRVPMTransportResult,
        ReleaseRowCommonTransport,
        ReleaseRowTransportAttestation,
    ]:
        state = self._v5h10_active_row_state
        owner = self._v5h10_active_row
        if state is None or owner is None:
            raise RuntimeError("v5h10 transport has no committed active row")
        (
            _,
            _,
            common_transport_type,
            attestation_type,
            _,
            validate_common_transport,
            validate_attestation,
            begin_common_transport,
            attest_common_transport,
            _,
            _,
        ) = self._row_api()
        common_transport = begin_common_transport(owner, owner.state)
        if (
            type(common_transport) is not common_transport_type
            or validate_common_transport(common_transport) is not common_transport
            or common_transport.parent_epoch != owner.epoch
            or common_transport.parent_owner_sha256 != owner.owner_sha256
            or common_transport.parent_state_sha256 != owner.state.state_sha256
        ):
            raise RuntimeError("common material transport session is inconsistent")
        flat_live_indices = tuple(
            index
            for indices in common_transport.live_particle_indices_by_cell
            for index in indices
        )
        if (
            common_transport.live_particle_indices_by_cell
            != owner.state.live_boundary_indices_by_cell
            or common_transport.frontier_node_offset != len(flat_live_indices)
            or common_transport.material_tracer_positions.shape
            != (
                common_transport.frontier_node_offset
                + owner.state.live_boundary_nodes.shape[0],
                3,
            )
        ):
            raise RuntimeError("common material transport pack is inconsistent")
        reference_length = min(
            float(np.min(state.sigma)),
            min(float(airplane.c_ref) for airplane in self.current_airplanes),
        )
        epsilon = self.v5h10_config.relative_epsilon * reference_length
        evaluations: list[PteraFieldEvaluation] = []

        def external(points: np.ndarray) -> FrozenExternalField:
            evaluation = evaluate_frozen_parent_ptera_field(
                self, points, epsilon_m=epsilon
            )
            evaluations.append(evaluation)
            return FrozenExternalField(
                velocity=evaluation.velocity_gp1_m_per_s,
                jacobian=evaluation.jacobian_per_s,
            )

        before = ptera_parent_state_sha256(self)
        generic = lsrk3_step_with_external_field(
            state,
            float(self.delta_time),
            external_field=external,
            baseline_freestream_velocity_gp1_m_per_s=np.asarray(
                self._currentVInf_GP1__E, dtype=np.float64
            ),
            enabled=True,
        )
        if len(generic.stages) != 3 or len(evaluations) != 3:
            raise RuntimeError("v5h10 transport did not execute exactly three stages")
        material_tracers = np.array(
            common_transport.material_tracer_positions,
            dtype=np.float64,
            order="C",
            copy=True,
        )
        tracer_storage = np.zeros_like(material_tracers)
        direct = _assert_frozen_bindings()["direct_field"]
        parent_velocity = _assert_v5h4_bindings()["parent_velocity"]
        for stage in generic.stages:
            self_velocity = direct(
                stage.pre.positions,
                stage.pre.gamma,
                stage.pre.sigma,
                target_positions=material_tracers,
            ).velocity
            ptera_velocity = parent_velocity(self, material_tracers)
            total_velocity = np.asarray(self_velocity, dtype=np.float64) + np.asarray(
                ptera_velocity, dtype=np.float64
            )
            if total_velocity.shape != material_tracers.shape or not np.all(
                np.isfinite(total_velocity)
            ):
                raise FloatingPointError(
                    "v5h10 common material-tracer velocity is invalid"
                )
            tracer_storage = (
                stage.a * tracer_storage + float(self.delta_time) * total_velocity
            )
            material_tracers = material_tracers + stage.b * tracer_storage
            if not np.all(np.isfinite(material_tracers)):
                raise FloatingPointError(
                    "v5h10 common material-tracer transport is non-finite"
                )
        material_tracers = _readonly_float64(
            "transported common material tracers", material_tracers, ndim=2
        )
        after = ptera_parent_state_sha256(self)
        if after != before:
            raise RuntimeError("v5h10 transport mutated native Ptera parent state")
        final = generic.final_state
        flat_live_array = np.asarray(flat_live_indices, dtype=np.int64)
        transported_live_sigma = _readonly_float64(
            "transported live material sigma",
            final.sigma[flat_live_array],
            ndim=1,
        )
        source_step = self.v5h10_config.active_source_steps[
            len(self.v5h10_raw_step_records)
        ]
        transport_end_time = owner.state.rows[-1].source_time_s + owner.release_dt_s
        attestation = attest_common_transport(
            common_transport,
            final.positions,
            final.gamma,
            final.sigma,
            material_tracers,
            transported_live_sigma,
            source_step_index=source_step,
            transport_end_time_s=transport_end_time,
            transport_epoch=owner.epoch,
        )
        if (
            type(attestation) is not attestation_type
            or validate_attestation(attestation) is not attestation
            or attestation.common_transport_sha256
            != common_transport.common_transport_sha256
            or attestation.parent_epoch != owner.epoch
            or attestation.parent_owner_sha256 != owner.owner_sha256
            or attestation.parent_state_sha256 != owner.state.state_sha256
            or attestation.source_step_index != source_step
            or attestation.transport_end_time_s != transport_end_time
            or attestation.transport_epoch != owner.epoch
        ):
            raise RuntimeError("common material transport attestation is inconsistent")
        from dataclasses import replace

        stages = tuple(
            replace(stage, ptera_field=evaluation)
            for stage, evaluation in zip(generic.stages, evaluations, strict=True)
        )
        result = replace(
            generic,
            ptera_step_index=int(self._current_step),
            dvm_for_source_step_index=self.v5h10_config.active_source_steps[
                len(self.v5h10_raw_step_records)
            ],
            source_report_sha256=self._v5h10_active_row_digest,
            relative_epsilon=self.v5h10_config.relative_epsilon,
            epsilon_m=epsilon,
            stages=stages,
            ptera_center_call_count=3,
            ptera_finite_difference_call_count=18,
            ptera_parent_state_sha256_before=before,
            ptera_parent_state_sha256_after=after,
            ptera_parent_state_unchanged=True,
        )
        return result, common_transport, attestation

    def _populate_next_airplanes_wake(self) -> None:
        active = self._v5h10_active_row
        if active is None:
            return super()._populate_next_airplanes_wake()
        feedback_count = len(self.v5h3_feedback_step_reports)
        super()._populate_next_airplanes_wake()
        if len(self.v5h3_feedback_step_reports) != feedback_count + 1:
            raise RuntimeError("v5h10 native Ptera feedback ledger did not commit")
        feedback = self.v5h3_feedback_step_reports[-1]
        if feedback.no_penetration_max_abs > self.v5h10_config.no_penetration_atol:
            raise RuntimeError("v5h10 no-penetration residual exceeded the hard gate")
        (
            transport,
            common_transport,
            transport_attestation,
        ) = self._transport_active_row()
        transported_nodes = _readonly_float64(
            "attested transported live-boundary nodes",
            transport_attestation.transported_live_boundary_nodes,
            ndim=2,
        )
        final = make_particle_state(
            transport.final_state.positions,
            transport.final_state.gamma,
            transport.final_state.sigma,
        )
        final.positions.setflags(write=False)
        final.gamma.setflags(write=False)
        final.sigma.setflags(write=False)
        (
            _,
            owner_type,
            _,
            _,
            validate_owner,
            _,
            _,
            _,
            _,
            make_transport_digest,
            advance_transport_parent,
        ) = self._row_api()
        source_step = self.v5h10_config.active_source_steps[
            len(self.v5h10_raw_step_records)
        ]
        transport_end_time = active.state.rows[-1].source_time_s + active.release_dt_s
        transport_parent_digest = make_transport_digest(
            active.state,
            final.positions,
            final.gamma,
            final.sigma,
            transported_nodes,
            common_transport_attestation=transport_attestation,
            source_step_index=source_step,
            transport_end_time_s=transport_end_time,
            transport_epoch=active.epoch,
        )
        airplanes = tuple(self.current_airplanes)
        if len(airplanes) != 1:
            raise RuntimeError("v5h10 raw slice requires one Baik airplane")
        airplane = airplanes[0]
        record_index = len(self.v5h10_raw_step_records)
        ordered_panels: list[Any] = []
        panel_ids: list[str] = []
        for airplane_index, current_airplane in enumerate(airplanes):
            for wing_index, wing in enumerate(current_airplane.wings):
                wing_panels = np.asarray(wing.panels, dtype=object)
                if wing_panels.ndim != 2:
                    raise RuntimeError(
                        "native Ptera panel topology is not two-dimensional"
                    )
                for chordwise_index in range(wing_panels.shape[0]):
                    for spanwise_index in range(wing_panels.shape[1]):
                        ordered_panels.append(
                            wing_panels[chordwise_index, spanwise_index]
                        )
                        panel_ids.append(
                            f"airplane:{airplane_index}/wing:{wing_index}/"
                            f"chord:{chordwise_index}/span:{spanwise_index}"
                        )
        solver_panels = tuple(self.panels)
        if (
            len(ordered_panels) != self.num_panels
            or len(solver_panels) != self.num_panels
            or any(
                topology_panel is not solver_panel
                for topology_panel, solver_panel in zip(
                    ordered_panels, solver_panels, strict=True
                )
            )
        ):
            raise RuntimeError("native Ptera panel topology/order is inconsistent")
        parent_load_arrays = (
            airplane.forces_W,
            airplane.forceCoefficients_W,
            airplane.moments_W_CgP1,
            airplane.momentCoefficients_W_CgP1,
        )
        panel_parent_load_arrays = tuple(
            array
            for panel in ordered_panels
            for array in (
                panel.forces_GP1,
                panel.moments_GP1_CgP1,
                panel.forces_W,
                panel.moments_W_CgP1,
            )
        )
        all_parent_load_arrays = parent_load_arrays + panel_parent_load_arrays
        parent_load_snapshots = tuple(
            (
                np.asarray(array).dtype.str,
                np.asarray(array).shape,
                np.asarray(array).tobytes(order="C"),
                np.asarray(array).flags.writeable,
            )
            for array in all_parent_load_arrays
        )
        raw_forces = _readonly_float64("Ptera forces_W", parent_load_arrays[0], ndim=1)
        raw_force_coefficients = _readonly_float64(
            "Ptera forceCoefficients_W", parent_load_arrays[1], ndim=1
        )
        raw_moments = _readonly_float64(
            "Ptera moments_W_CgP1", parent_load_arrays[2], ndim=1
        )
        raw_moment_coefficients = _readonly_float64(
            "Ptera momentCoefficients_W", parent_load_arrays[3], ndim=1
        )
        raw_panel_forces = _readonly_float64(
            "Ptera panel forces_W",
            np.vstack([panel.forces_W for panel in ordered_panels]),
            ndim=2,
        )
        raw_panel_moments = _readonly_float64(
            "Ptera panel moments_W_CgP1",
            np.vstack([panel.moments_W_CgP1 for panel in ordered_panels]),
            ndim=2,
        )
        if raw_panel_forces.shape != (
            self.num_panels,
            3,
        ) or raw_panel_moments.shape != (
            self.num_panels,
            3,
        ):
            raise RuntimeError("native Ptera panel load shape is inconsistent")
        raw_panel_force_sum = _readonly_float64(
            "Ptera panel force sum_W", np.sum(raw_panel_forces, axis=0), ndim=1
        )
        raw_panel_moment_sum = _readonly_float64(
            "Ptera panel moment sum_W_CgP1",
            np.sum(raw_panel_moments, axis=0),
            ndim=1,
        )
        force_sum_residual = float(np.max(np.abs(raw_panel_force_sum - raw_forces)))
        moment_sum_residual = float(np.max(np.abs(raw_panel_moment_sum - raw_moments)))
        force_sum_atol = float(
            64.0
            * np.finfo(np.float64).eps
            * max(1.0, float(np.sum(np.abs(raw_panel_forces))))
        )
        moment_sum_atol = float(
            64.0
            * np.finfo(np.float64).eps
            * max(1.0, float(np.sum(np.abs(raw_panel_moments))))
        )
        if force_sum_residual > force_sum_atol:
            raise RuntimeError("native Ptera panel forces do not sum to airplane total")
        if moment_sum_residual > moment_sum_atol:
            raise RuntimeError(
                "native Ptera panel moments do not sum to airplane total"
            )
        raw_load_arrays = (
            raw_forces,
            raw_force_coefficients,
            raw_moments,
            raw_moment_coefficients,
        )
        if any(
            np.shares_memory(raw, parent)
            for raw, parent in zip(raw_load_arrays, parent_load_arrays, strict=True)
        ):
            raise RuntimeError("raw Ptera load extraction retained a parent alias")
        if any(
            np.shares_memory(raw_panel_forces, panel.forces_W)
            or np.shares_memory(raw_panel_moments, panel.moments_W_CgP1)
            for panel in ordered_panels
        ):
            raise RuntimeError(
                "raw Ptera panel load extraction retained a parent alias"
            )
        parent_load_snapshots_after = tuple(
            (
                np.asarray(array).dtype.str,
                np.asarray(array).shape,
                np.asarray(array).tobytes(order="C"),
                np.asarray(array).flags.writeable,
            )
            for array in all_parent_load_arrays
        )
        if parent_load_snapshots_after != parent_load_snapshots:
            raise RuntimeError("raw record extraction mutated Ptera load storage")
        parent_after_raw_record = ptera_parent_state_sha256(self)
        expected_parent_hash = transport.ptera_parent_state_sha256_after
        if (
            type(expected_parent_hash) is not str
            or parent_after_raw_record != expected_parent_hash
        ):
            raise RuntimeError(
                "raw record extraction mutated native Ptera parent state"
            )
        # Consume the live transport capability only after every raw Ptera
        # output has passed its finite/copy/sum/parent-integrity gates.
        advanced_owner = advance_transport_parent(
            active,
            active.state,
            final.positions,
            final.gamma,
            final.sigma,
            transported_nodes,
            common_transport_attestation=transport_attestation,
            parent_transport_digest=transport_parent_digest,
            source_step_index=source_step,
            transport_end_time_s=transport_end_time,
            transport_epoch=active.epoch,
        )
        if validate_owner(advanced_owner) is not advanced_owner:
            raise RuntimeError("transported row-owner validator changed live identity")
        if (
            type(advanced_owner) is not owner_type
            or advanced_owner.epoch != active.epoch
            or advanced_owner.state.phase != "post_transport"
            or advanced_owner.state.transport_source_step_index != source_step
            or advanced_owner.state.transport_end_time_s != transport_end_time
            or advanced_owner.state.parent_transport_digest != transport_parent_digest
            or advanced_owner.state.parent_transport_attestation_sha256
            != transport_attestation.attestation_sha256
            or advanced_owner.state.positions.shape != final.positions.shape
            or not np.array_equal(advanced_owner.state.positions, final.positions)
            or not np.array_equal(advanced_owner.state.gamma, final.gamma)
            or not np.array_equal(advanced_owner.state.sigma, final.sigma)
            or not np.array_equal(
                advanced_owner.state.live_boundary_nodes, transported_nodes
            )
        ):
            raise RuntimeError("transported row-owner reattestation is inconsistent")
        record = BaikW2RawStepRecord(
            interface_id=COUPLING_INTERFACE_ID,
            case_id=CASE_ID,
            ptera_step_index=int(self._current_step),
            source_step_index=self.v5h10_config.active_source_steps[record_index],
            source_time_s=int(self._current_step) * float(self.delta_time),
            birth_mode=self.v5h10_config.active_birth_modes[record_index],
            row_commit_sha256=self._v5h10_active_row_digest or "",
            row_state_sha256=self._v5h10_active_state_digest or "",
            transported_parent_sha256=(
                None
                if self._v5h10_transported_parent is None
                else _particle_state_sha256(self._v5h10_transported_parent)
            ),
            transported_state_sha256=advanced_owner.state.state_sha256,
            transported_arrays_sha256=_particle_state_sha256(final),
            transport_parent_digest=transport_parent_digest,
            common_transport_sha256=common_transport.common_transport_sha256,
            common_transport_attestation_sha256=(
                transport_attestation.attestation_sha256
            ),
            transported_material_tracers_sha256=(
                transport_attestation.transported_material_tracers_sha256
            ),
            advanced_owner_sha256=advanced_owner.owner_sha256,
            transported_live_boundary_nodes=transported_nodes,
            material_tracer_count=(common_transport.material_tracer_positions.shape[0]),
            material_support_tracer_count=common_transport.frontier_node_offset,
            frontier_node_tracer_count=(
                common_transport.material_tracer_positions.shape[0]
                - common_transport.frontier_node_offset
            ),
            particle_count=final.positions.shape[0],
            feedback_report=feedback,
            transport_result=transport,
            ptera_forces_w=raw_forces,
            ptera_forces_w_sha256=_array_sha256(raw_forces),
            ptera_force_coefficients_w=raw_force_coefficients,
            ptera_force_coefficients_w_sha256=_array_sha256(raw_force_coefficients),
            ptera_moments_w_cgp1=raw_moments,
            ptera_moments_w_cgp1_sha256=_array_sha256(raw_moments),
            ptera_moment_coefficients_w=raw_moment_coefficients,
            ptera_moment_coefficients_w_sha256=_array_sha256(raw_moment_coefficients),
            ptera_panel_ids=tuple(panel_ids),
            ptera_panel_forces_w=raw_panel_forces,
            ptera_panel_forces_w_sha256=_array_sha256(raw_panel_forces),
            ptera_panel_moments_w_cgp1=raw_panel_moments,
            ptera_panel_moments_w_cgp1_sha256=_array_sha256(raw_panel_moments),
            ptera_panel_force_sum_w=raw_panel_force_sum,
            ptera_panel_moment_sum_w_cgp1=raw_panel_moment_sum,
            ptera_panel_force_sum_max_abs_residual=force_sum_residual,
            ptera_panel_moment_sum_max_abs_residual=moment_sum_residual,
            ptera_panel_force_sum_atol=force_sum_atol,
            ptera_panel_moment_sum_atol=moment_sum_atol,
            collocation_evaluation_count=feedback.collocation_evaluation_count,
            load_batch_evaluation_count=feedback.load_leg_evaluation_count,
            native_load_call_count=feedback.parent_load_call_count,
            transport_call_count=1,
            transport_stage_count=len(transport.stages),
            common_transport_self_field_call_count=3,
            common_transport_ptera_center_call_count=3,
            common_transport_stage_count=3,
            frontier_self_field_call_count=3,
            frontier_ptera_center_call_count=3,
            frontier_transport_stage_count=3,
            ptera_parent_sha256_before_transport=(
                transport.ptera_parent_state_sha256_before or ""
            ),
            ptera_parent_sha256_after_transport=(
                transport.ptera_parent_state_sha256_after or ""
            ),
            ptera_parent_state_unchanged=transport.ptera_parent_state_unchanged,
            ptera_parent_sha256_after_raw_record=parent_after_raw_record,
            ptera_parent_state_unchanged_after_raw_record=True,
            observation_access=OBSERVATION_ACCESS,
            surface_load_owner=SURFACE_FORCE_OWNER,
            source_load_owner=SOURCE_LOAD_OWNER,
            force_scoring_status=FORCE_SCORING_STATUS,
        )
        self.v5h10_raw_step_records.append(record)
        self._v5h10_previous_owner = advanced_owner
        self._v5h10_transported_parent = final
        self._v5h10_active_row = None
        self._v5h10_active_row_state = None
        self._v5h10_active_row_digest = None
        self._v5h10_active_state_digest = None
        if len(self.v5h10_raw_step_records) == MAX_ACTIVE_LAYERS:
            self._v5h10_slice_complete = True
            raise _BoundedSliceComplete

    def run(
        self,
        prescribed_wake: bool | np.bool_ = True,
        calculate_streamlines: bool | np.bool_ = False,
        show_progress: bool | np.bool_ = False,
    ) -> None:
        """Run only through the third active layer; never finalize a full case."""

        if calculate_streamlines:
            raise ValueError(
                "bounded v5h10 raw slice forbids streamline postprocessing"
            )
        if self._v5h10_slice_complete:
            raise RuntimeError("bounded v5h10 raw slice cannot be run twice")
        try:
            super().run(
                prescribed_wake=prescribed_wake,
                calculate_streamlines=False,
                show_progress=show_progress,
            )
        except _BoundedSliceComplete:
            if len(self.v5h10_raw_step_records) != MAX_ACTIVE_LAYERS:
                raise RuntimeError("v5h10 bounded stop occurred at the wrong layer")
            return
        except Exception as error:
            self._poison(error)
            raise
        raise RuntimeError("v5h10 full Ptera run escaped the bounded three-layer stop")


@dataclass(frozen=True, slots=True)
class _FeedbackRowView:
    for_source_step_index: int
    transport_end_time_s: float
    report_sha256: str


def make_fluxv_v5h10_baik_w2_solver(
    unsteady_problem: Any,
    *,
    row_committer: GlobalRowCommitter,
    config: BaikW2CouplingConfig = DEFAULT_CONFIG,
    **uvpm_kwargs: Any,
) -> GlobalRowPteraRVPMCouplingSolver:
    """Construct the frozen bounded W2 coupling solver."""

    return GlobalRowPteraRVPMCouplingSolver(
        unsteady_problem,
        row_committer=row_committer,
        config=config,
        **uvpm_kwargs,
    )


__all__ = [
    "ACTIVE_BIRTH_MODES",
    "ACTIVE_PTERA_STEPS",
    "ACTIVE_SOURCE_STEPS",
    "BaikW2CouplingConfig",
    "BaikW2RawStepRecord",
    "CASE_ID",
    "CONVECTIVE_DELTA_TIME",
    "COUPLING_INTERFACE_ID",
    "DEFAULT_CONFIG",
    "FORCE_SCORING_STATUS",
    "GlobalRowCommitRequest",
    "GlobalRowCommitter",
    "GlobalRowPteraRVPMCouplingSolver",
    "OBSERVATION_ACCESS",
    "PTERA_DELTA_TIME_S",
    "SURFACE_FORCE_OWNER",
    "make_fluxv_v5h10_baik_w2_solver",
]
