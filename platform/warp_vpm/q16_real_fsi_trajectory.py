"""Bounded multi-step ownership for real Q16 separated-flow FSI.

The trajectory owner intentionally does not make an entire time history one
atomic transaction.  Each successful physical step is committed exactly once;
if the next step fails, the completed prefix remains live and the failed step
must leave that last committed parent unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import math
from typing import Any

import numpy as np
import torch
import warp as wp

from fluxvortex.warp_fsi.q16_mandatory_aero_mode import (
    require_q16_mandatory_aero_mode,
)
from q16_real_aero_branch_transaction import _solver_sha256
from q16_real_fsi_coupling import (
    Q16CudaRealFSIOwner,
    Q16CudaRealFSIStepper,
    Q16RealFSIStepResult,
    Q16RealFSIStepStopped,
)

_RECORD_DOMAIN = "flux-v5m-q16-real-fsi-trajectory-record-v4"
_CHAIN_DOMAIN = "flux-v5m-q16-real-fsi-trajectory-chain-v1"
_RESULT_DOMAIN = "flux-v5m-q16-real-fsi-trajectory-result-v5"
MAX_Q16_REAL_FSI_TRAJECTORY_STEP_COUNT = 4096


def _canonical_bytes(value: dict[str, Any]) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _payload_sha256(domain: str, value: dict[str, Any]) -> str:
    return hashlib.sha256(
        domain.encode("ascii") + b"\0" + _canonical_bytes(value)
    ).hexdigest()


def _exact_positive_float(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise TypeError(f"{name} must be a real scalar")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _exact_step_count(value: int) -> int:
    if (
        type(value) is not int
        or not 1 <= value <= MAX_Q16_REAL_FSI_TRAJECTORY_STEP_COUNT
    ):
        raise ValueError(
            "step_count must be an exact positive int within the trajectory cap"
        )
    return value


def _sha256_text(name: str, value: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _finite_nonnegative(name: str, value: float) -> float:
    if type(value) is not float or not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be a finite non-negative exact float")
    return value


def _finite_float(name: str, value: float) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite exact float")
    return value


def _exact_nonnegative_int(name: str, value: int) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative exact int")
    return value


@dataclass(frozen=True, slots=True, eq=False)
class Q16RealFSITrajectoryRecord:
    step_index: int
    owner_generation: int
    aero_generation: int
    parent_structural_state_sha256: str
    result_structural_state_sha256: str
    parent_aero_state_sha256: str
    result_aero_state_sha256: str
    parent_solver_steps_done: int
    solver_steps_done: int
    parent_solver_current_step: int
    solver_current_step: int
    parent_lev_particle_count: int
    lev_particle_count: int
    parent_wake_convection_count: int
    wake_convection_count: int
    step_result_sha256: str
    complete_load_sha256: str
    complete_load_norm: float
    operating_point_velocity: float
    aerodynamic_force_x_w: float
    aerodynamic_force_y_w: float
    aerodynamic_force_z_w: float
    span_tip_centroid_displacement_x_w: float
    span_tip_centroid_displacement_y_w: float
    span_tip_centroid_displacement_z_w: float
    coupling_iteration_count: int
    aerodynamic_evaluation_count: int
    coupling_relative_residual: float
    structural_newton_iteration_count: int
    structural_cg_iteration_count: int
    structural_gmres_iteration_count: int
    structural_indefinite_fallback_count: int
    structural_relative_residual: float
    relaxation_factor_count: int
    relaxation_factor_min: float
    relaxation_factor_max: float
    relaxation_factor_final: float
    kinetic_energy_start: float
    kinetic_energy_end: float
    kinetic_energy_change: float
    internal_trapezoidal_work: float
    damping_trapezoidal_work: float
    external_trapezoidal_work: float
    work_balance_residual: float
    work_balance_relative_residual: float
    state_increment_norm: float
    deformation_norm_end: float
    velocity_norm_end: float
    acceleration_norm_end: float
    previous_trajectory_chain_sha256: str
    record_sha256: str
    trajectory_chain_sha256: str


def _record_payload(record: Q16RealFSITrajectoryRecord) -> dict[str, Any]:
    return {
        "aero_generation": record.aero_generation,
        "aerodynamic_evaluation_count": record.aerodynamic_evaluation_count,
        "complete_load_norm_hex": record.complete_load_norm.hex(),
        "complete_load_sha256": record.complete_load_sha256,
        "operating_point_velocity_hex": record.operating_point_velocity.hex(),
        "aerodynamic_force_x_w_hex": record.aerodynamic_force_x_w.hex(),
        "aerodynamic_force_y_w_hex": record.aerodynamic_force_y_w.hex(),
        "aerodynamic_force_z_w_hex": record.aerodynamic_force_z_w.hex(),
        "span_tip_centroid_displacement_x_w_hex": (
            record.span_tip_centroid_displacement_x_w.hex()
        ),
        "span_tip_centroid_displacement_y_w_hex": (
            record.span_tip_centroid_displacement_y_w.hex()
        ),
        "span_tip_centroid_displacement_z_w_hex": (
            record.span_tip_centroid_displacement_z_w.hex()
        ),
        "coupling_iteration_count": record.coupling_iteration_count,
        "coupling_relative_residual_hex": record.coupling_relative_residual.hex(),
        "lev_particle_count": record.lev_particle_count,
        "kinetic_energy_change_hex": record.kinetic_energy_change.hex(),
        "kinetic_energy_end_hex": record.kinetic_energy_end.hex(),
        "kinetic_energy_start_hex": record.kinetic_energy_start.hex(),
        "internal_trapezoidal_work_hex": (record.internal_trapezoidal_work.hex()),
        "damping_trapezoidal_work_hex": (record.damping_trapezoidal_work.hex()),
        "owner_generation": record.owner_generation,
        "parent_aero_state_sha256": record.parent_aero_state_sha256,
        "parent_lev_particle_count": record.parent_lev_particle_count,
        "parent_solver_current_step": record.parent_solver_current_step,
        "parent_solver_steps_done": record.parent_solver_steps_done,
        "parent_structural_state_sha256": record.parent_structural_state_sha256,
        "parent_wake_convection_count": record.parent_wake_convection_count,
        "previous_trajectory_chain_sha256": record.previous_trajectory_chain_sha256,
        "result_aero_state_sha256": record.result_aero_state_sha256,
        "result_structural_state_sha256": record.result_structural_state_sha256,
        "relaxation_factor_count": record.relaxation_factor_count,
        "relaxation_factor_final_hex": record.relaxation_factor_final.hex(),
        "relaxation_factor_max_hex": record.relaxation_factor_max.hex(),
        "relaxation_factor_min_hex": record.relaxation_factor_min.hex(),
        "solver_current_step": record.solver_current_step,
        "solver_steps_done": record.solver_steps_done,
        "step_index": record.step_index,
        "step_result_sha256": record.step_result_sha256,
        "structural_cg_iteration_count": record.structural_cg_iteration_count,
        "structural_gmres_iteration_count": record.structural_gmres_iteration_count,
        "structural_indefinite_fallback_count": (
            record.structural_indefinite_fallback_count
        ),
        "structural_newton_iteration_count": record.structural_newton_iteration_count,
        "structural_relative_residual_hex": record.structural_relative_residual.hex(),
        "external_trapezoidal_work_hex": (record.external_trapezoidal_work.hex()),
        "work_balance_residual_hex": record.work_balance_residual.hex(),
        "work_balance_relative_residual_hex": (
            record.work_balance_relative_residual.hex()
        ),
        "state_increment_norm_hex": record.state_increment_norm.hex(),
        "deformation_norm_end_hex": record.deformation_norm_end.hex(),
        "velocity_norm_end_hex": record.velocity_norm_end.hex(),
        "acceleration_norm_end_hex": record.acceleration_norm_end.hex(),
        "wake_convection_count": record.wake_convection_count,
    }


def _record_sha256(record: Q16RealFSITrajectoryRecord) -> str:
    return _payload_sha256(_RECORD_DOMAIN, _record_payload(record))


def _chain_sha256(previous: str, record_sha256: str) -> str:
    return _payload_sha256(
        _CHAIN_DOMAIN,
        {
            "previous_trajectory_chain_sha256": previous,
            "record_sha256": record_sha256,
        },
    )


@dataclass(frozen=True, slots=True, eq=False)
class Q16RealFSITrajectoryResult:
    delta_time: float
    coupling_tolerance: float
    mass_damping_coefficient: float
    relaxation_method: str
    initial_relaxation: float
    max_coupling_iterations: int
    requested_step_count: int
    completed_step_count: int
    initial_owner_generation: int
    final_owner_generation: int
    initial_aero_generation: int
    final_aero_generation: int
    initial_structural_state_sha256: str
    final_structural_state_sha256: str
    initial_aero_state_sha256: str
    final_aero_state_sha256: str
    initial_trajectory_chain_sha256: str
    trajectory_chain_sha256: str
    records: tuple[Q16RealFSITrajectoryRecord, ...]
    result_sha256: str


def _initial_chain_sha256(
    *,
    delta_time: float,
    coupling_tolerance: float,
    mass_damping_coefficient: float,
    relaxation_method: str,
    initial_relaxation: float,
    max_coupling_iterations: int,
    initial_owner_generation: int,
    initial_aero_generation: int,
    initial_structural_state_sha256: str,
    initial_aero_state_sha256: str,
) -> str:
    return _payload_sha256(
        _CHAIN_DOMAIN,
        {
            "coupling_tolerance_hex": coupling_tolerance.hex(),
            "delta_time_hex": delta_time.hex(),
            "mass_damping_coefficient_hex": mass_damping_coefficient.hex(),
            "initial_relaxation_hex": initial_relaxation.hex(),
            "initial_aero_generation": initial_aero_generation,
            "initial_aero_state_sha256": initial_aero_state_sha256,
            "initial_owner_generation": initial_owner_generation,
            "initial_structural_state_sha256": initial_structural_state_sha256,
            "max_coupling_iterations": max_coupling_iterations,
            "relaxation_method": relaxation_method,
        },
    )


def _result_payload(result: Q16RealFSITrajectoryResult) -> dict[str, Any]:
    return {
        "completed_step_count": result.completed_step_count,
        "coupling_tolerance_hex": result.coupling_tolerance.hex(),
        "delta_time_hex": result.delta_time.hex(),
        "initial_relaxation_hex": result.initial_relaxation.hex(),
        "mass_damping_coefficient_hex": result.mass_damping_coefficient.hex(),
        "final_aero_generation": result.final_aero_generation,
        "final_aero_state_sha256": result.final_aero_state_sha256,
        "final_owner_generation": result.final_owner_generation,
        "final_structural_state_sha256": result.final_structural_state_sha256,
        "initial_aero_generation": result.initial_aero_generation,
        "initial_aero_state_sha256": result.initial_aero_state_sha256,
        "initial_owner_generation": result.initial_owner_generation,
        "initial_structural_state_sha256": result.initial_structural_state_sha256,
        "initial_trajectory_chain_sha256": result.initial_trajectory_chain_sha256,
        "max_coupling_iterations": result.max_coupling_iterations,
        "record_sha256": [record.record_sha256 for record in result.records],
        "requested_step_count": result.requested_step_count,
        "relaxation_method": result.relaxation_method,
        "trajectory_chain_sha256": result.trajectory_chain_sha256,
    }


def _result_sha256(result: Q16RealFSITrajectoryResult) -> str:
    return _payload_sha256(_RESULT_DOMAIN, _result_payload(result))


def validate_q16_real_fsi_trajectory(
    result: Q16RealFSITrajectoryResult,
) -> Q16RealFSITrajectoryResult:
    """Independently replay scalar lineage, record digests and hash chain."""

    if type(result) is not Q16RealFSITrajectoryResult:
        raise TypeError("result must have exact Q16RealFSITrajectoryResult type")
    if type(result.records) is not tuple or any(
        type(record) is not Q16RealFSITrajectoryRecord for record in result.records
    ):
        raise TypeError("trajectory records must be an exact record tuple")
    dt = _exact_positive_float("delta_time", result.delta_time)
    tolerance = _exact_positive_float("coupling_tolerance", result.coupling_tolerance)
    damping = _finite_nonnegative(
        "mass_damping_coefficient", result.mass_damping_coefficient
    )
    initial_relaxation = _exact_positive_float(
        "initial_relaxation", result.initial_relaxation
    )
    if initial_relaxation > 1.0:
        raise ValueError("initial_relaxation must be at most one")
    if type(result.relaxation_method) is not str or result.relaxation_method not in {
        "aitken",
        "fixed",
    }:
        raise ValueError("trajectory relaxation_method is invalid")
    if (
        type(result.max_coupling_iterations) is not int
        or result.max_coupling_iterations <= 0
    ):
        raise ValueError("max_coupling_iterations must be a positive exact int")
    requested = _exact_step_count(result.requested_step_count)
    if tolerance >= 1.0:
        raise ValueError("coupling_tolerance must be smaller than one")
    _exact_nonnegative_int("completed_step_count", result.completed_step_count)
    _exact_nonnegative_int("initial_owner_generation", result.initial_owner_generation)
    _exact_nonnegative_int("final_owner_generation", result.final_owner_generation)
    _exact_nonnegative_int("initial_aero_generation", result.initial_aero_generation)
    _exact_nonnegative_int("final_aero_generation", result.final_aero_generation)
    if result.completed_step_count != requested or len(result.records) != requested:
        raise ValueError("successful trajectory step counts differ")
    if result.final_owner_generation != result.initial_owner_generation + requested:
        raise ValueError("final structural owner generation drift")
    if result.final_aero_generation != result.initial_aero_generation + requested:
        raise ValueError("final aerodynamic owner generation drift")
    for name in (
        "initial_structural_state_sha256",
        "final_structural_state_sha256",
        "initial_aero_state_sha256",
        "final_aero_state_sha256",
        "initial_trajectory_chain_sha256",
        "trajectory_chain_sha256",
        "result_sha256",
    ):
        _sha256_text(name, getattr(result, name))
    initial_chain = _initial_chain_sha256(
        delta_time=dt,
        coupling_tolerance=tolerance,
        mass_damping_coefficient=damping,
        relaxation_method=result.relaxation_method,
        initial_relaxation=initial_relaxation,
        max_coupling_iterations=result.max_coupling_iterations,
        initial_owner_generation=result.initial_owner_generation,
        initial_aero_generation=result.initial_aero_generation,
        initial_structural_state_sha256=result.initial_structural_state_sha256,
        initial_aero_state_sha256=result.initial_aero_state_sha256,
    )
    if initial_chain != result.initial_trajectory_chain_sha256:
        raise ValueError("initial trajectory chain digest mismatch")

    previous_chain = initial_chain
    previous_structural = result.initial_structural_state_sha256
    previous_aero = result.initial_aero_state_sha256
    previous_steps_done: int | None = None
    previous_current_step: int | None = None
    previous_lev: int | None = None
    previous_wake: int | None = None
    previous_kinetic_end: float | None = None
    for offset, record in enumerate(result.records, start=1):
        if record.step_index != offset:
            raise ValueError("trajectory record step order drift")
        if record.owner_generation != result.initial_owner_generation + offset:
            raise ValueError("structural owner generation continuity drift")
        if record.aero_generation != result.initial_aero_generation + offset:
            raise ValueError("aerodynamic owner generation continuity drift")
        for name in (
            "parent_structural_state_sha256",
            "result_structural_state_sha256",
            "parent_aero_state_sha256",
            "result_aero_state_sha256",
            "step_result_sha256",
            "complete_load_sha256",
            "previous_trajectory_chain_sha256",
            "record_sha256",
            "trajectory_chain_sha256",
        ):
            _sha256_text(name, getattr(record, name))
        if record.parent_structural_state_sha256 != previous_structural:
            raise ValueError("structural trajectory lineage drift")
        if record.parent_aero_state_sha256 != previous_aero:
            raise ValueError("aerodynamic trajectory lineage drift")
        if record.previous_trajectory_chain_sha256 != previous_chain:
            raise ValueError("trajectory chain parent drift")
        for name in (
            "parent_solver_steps_done",
            "solver_steps_done",
            "parent_solver_current_step",
            "solver_current_step",
            "parent_lev_particle_count",
            "lev_particle_count",
            "parent_wake_convection_count",
            "wake_convection_count",
            "coupling_iteration_count",
            "aerodynamic_evaluation_count",
            "structural_newton_iteration_count",
            "structural_cg_iteration_count",
            "structural_gmres_iteration_count",
            "structural_indefinite_fallback_count",
            "relaxation_factor_count",
        ):
            value = getattr(record, name)
            _exact_nonnegative_int(name, value)
        if record.solver_steps_done != record.parent_solver_steps_done + 1:
            raise ValueError("Ptera steps_done did not advance exactly once")
        if record.solver_current_step != record.parent_solver_current_step + 1:
            raise ValueError("Ptera current step did not advance exactly once")
        if record.wake_convection_count != record.parent_wake_convection_count + 1:
            raise ValueError("free-wake convection did not advance exactly once")
        if record.lev_particle_count < record.parent_lev_particle_count:
            raise ValueError("LEV particle history regressed")
        if record.coupling_iteration_count == 0:
            raise ValueError("trajectory record contains no coupling iteration")
        if record.aerodynamic_evaluation_count <= record.coupling_iteration_count:
            raise ValueError(
                "trajectory record lacks the accepted aerodynamic evaluation"
            )
        if record.structural_newton_iteration_count == 0:
            raise ValueError(
                "trajectory record contains no structural Newton iteration"
            )
        if record.structural_cg_iteration_count == 0:
            raise ValueError(
                "trajectory record contains no structural Krylov iteration"
            )
        for name in (
            "relaxation_factor_min",
            "relaxation_factor_max",
            "relaxation_factor_final",
        ):
            _finite_nonnegative(name, getattr(record, name))
        if record.relaxation_factor_count == 0:
            if not (
                record.relaxation_factor_min
                == record.relaxation_factor_max
                == record.relaxation_factor_final
                == 0.0
            ):
                raise ValueError("unused trajectory relaxation summary is nonzero")
        elif not (
            0.0
            < record.relaxation_factor_min
            <= record.relaxation_factor_final
            <= record.relaxation_factor_max
            <= 1.0
        ):
            raise ValueError("trajectory relaxation factor summary is invalid")
        if previous_steps_done is not None and (
            record.parent_solver_steps_done != previous_steps_done
            or record.parent_solver_current_step != previous_current_step
            or record.parent_lev_particle_count != previous_lev
            or record.parent_wake_convection_count != previous_wake
        ):
            raise ValueError("cross-step aerodynamic history continuity drift")
        if (
            previous_kinetic_end is not None
            and record.kinetic_energy_start != previous_kinetic_end
        ):
            raise ValueError("cross-step structural kinetic-energy continuity drift")
        _finite_nonnegative("complete_load_norm", record.complete_load_norm)
        if record.complete_load_norm == 0.0:
            raise ValueError("completed trajectory load must be nonzero")
        _exact_positive_float(
            "operating_point_velocity", record.operating_point_velocity
        )
        for name in (
            "aerodynamic_force_x_w",
            "aerodynamic_force_y_w",
            "aerodynamic_force_z_w",
            "span_tip_centroid_displacement_x_w",
            "span_tip_centroid_displacement_y_w",
            "span_tip_centroid_displacement_z_w",
        ):
            _finite_float(name, getattr(record, name))
        _finite_nonnegative(
            "coupling_relative_residual", record.coupling_relative_residual
        )
        _finite_nonnegative(
            "structural_relative_residual", record.structural_relative_residual
        )
        for name in (
            "kinetic_energy_start",
            "kinetic_energy_end",
            "work_balance_relative_residual",
            "state_increment_norm",
            "deformation_norm_end",
            "velocity_norm_end",
            "acceleration_norm_end",
        ):
            _finite_nonnegative(name, getattr(record, name))
        for name in (
            "kinetic_energy_change",
            "internal_trapezoidal_work",
            "damping_trapezoidal_work",
            "external_trapezoidal_work",
            "work_balance_residual",
        ):
            _finite_float(name, getattr(record, name))
        arithmetic_scale = max(
            1.0,
            abs(record.kinetic_energy_start),
            abs(record.kinetic_energy_end),
            abs(record.kinetic_energy_change),
            abs(record.internal_trapezoidal_work),
            abs(record.damping_trapezoidal_work),
            abs(record.external_trapezoidal_work),
        )
        arithmetic_tolerance = 128.0 * np.finfo(np.float64).eps * arithmetic_scale
        if not math.isclose(
            record.kinetic_energy_change,
            record.kinetic_energy_end - record.kinetic_energy_start,
            rel_tol=0.0,
            abs_tol=arithmetic_tolerance,
        ):
            raise ValueError("trajectory kinetic-energy arithmetic drift")
        expected_balance = (
            record.kinetic_energy_change
            + record.internal_trapezoidal_work
            + record.damping_trapezoidal_work
            - record.external_trapezoidal_work
        )
        if not math.isclose(
            record.work_balance_residual,
            expected_balance,
            rel_tol=0.0,
            abs_tol=arithmetic_tolerance,
        ):
            raise ValueError("trajectory endpoint-work arithmetic drift")
        expected_relative = abs(record.work_balance_residual) / max(
            1.0,
            abs(record.kinetic_energy_change),
            abs(record.internal_trapezoidal_work),
            abs(record.damping_trapezoidal_work),
            abs(record.external_trapezoidal_work),
        )
        if not math.isclose(
            record.work_balance_relative_residual,
            expected_relative,
            rel_tol=0.0,
            abs_tol=128.0 * np.finfo(np.float64).eps,
        ):
            raise ValueError("trajectory normalized work balance drift")
        if record.coupling_relative_residual > tolerance:
            raise ValueError("trajectory record violates coupling tolerance")
        if _record_sha256(record) != record.record_sha256:
            raise ValueError("trajectory record digest mismatch")
        chain = _chain_sha256(previous_chain, record.record_sha256)
        if chain != record.trajectory_chain_sha256:
            raise ValueError("trajectory record chain digest mismatch")
        previous_chain = chain
        previous_structural = record.result_structural_state_sha256
        previous_aero = record.result_aero_state_sha256
        previous_steps_done = record.solver_steps_done
        previous_current_step = record.solver_current_step
        previous_lev = record.lev_particle_count
        previous_wake = record.wake_convection_count
        previous_kinetic_end = record.kinetic_energy_end

    last = result.records[-1]
    if (
        result.final_owner_generation != last.owner_generation
        or result.final_aero_generation != last.aero_generation
        or result.final_structural_state_sha256 != last.result_structural_state_sha256
        or result.final_aero_state_sha256 != last.result_aero_state_sha256
        or result.trajectory_chain_sha256 != last.trajectory_chain_sha256
    ):
        raise ValueError("trajectory result final binding drift")
    if _result_sha256(result) != result.result_sha256:
        raise ValueError("trajectory result digest mismatch")
    return result


class Q16RealFSITrajectoryStopped(RuntimeError):
    """A failed time coordinate left its last completed prefix committed."""

    def __init__(
        self,
        message: str,
        *,
        phase: str,
        failed_step_index: int,
        completed_records: tuple[Q16RealFSITrajectoryRecord, ...],
        completed_trajectory_chain_sha256: str,
        failed_parent_structural_state_sha256: str,
        failed_parent_aero_state_sha256: str,
    ) -> None:
        super().__init__(message)
        self.phase = phase
        self.failed_step_index = failed_step_index
        self.step_began = True
        self.completed_records = completed_records
        self.completed_step_count = len(completed_records)
        self.completed_trajectory_chain_sha256 = completed_trajectory_chain_sha256
        self.failed_parent_structural_state_sha256 = (
            failed_parent_structural_state_sha256
        )
        self.failed_parent_aero_state_sha256 = failed_parent_aero_state_sha256


class Q16CudaRealFSITrajectory:
    """Sequentially commit a bounded real Q16 FSI trajectory."""

    __slots__ = ("stepper",)

    def __init__(self, stepper: Q16CudaRealFSIStepper) -> None:
        if type(stepper) is not Q16CudaRealFSIStepper:
            raise TypeError("stepper must have exact Q16CudaRealFSIStepper type")
        self.stepper = stepper

    def advance(
        self,
        owner: Q16CudaRealFSIOwner,
        *,
        step_count: int,
        delta_time: float,
    ) -> Q16RealFSITrajectoryResult:
        if type(owner) is not Q16CudaRealFSIOwner:
            raise TypeError("owner must have exact Q16CudaRealFSIOwner type")
        count = _exact_step_count(step_count)
        dt = _exact_positive_float("delta_time", delta_time)
        owner._assert_live()
        require_q16_mandatory_aero_mode(owner.aero_owner.current_solver)
        initial_owner_generation = owner.generation
        initial_aero_generation = owner.aero_owner.generation
        initial_structural = owner.state_sha256
        initial_aero = _solver_sha256(owner.aero_owner.current_solver)
        chain = _initial_chain_sha256(
            delta_time=dt,
            coupling_tolerance=self.stepper.coupling_tolerance,
            mass_damping_coefficient=(
                self.stepper.structural_stepper.mass_damping_coefficient
            ),
            relaxation_method=self.stepper.relaxation_method,
            initial_relaxation=self.stepper.relaxation,
            max_coupling_iterations=self.stepper.max_coupling_iterations,
            initial_owner_generation=initial_owner_generation,
            initial_aero_generation=initial_aero_generation,
            initial_structural_state_sha256=initial_structural,
            initial_aero_state_sha256=initial_aero,
        )
        return self._advance_from_prefix(
            owner,
            delta_time=dt,
            total_step_count=count,
            initial_owner_generation=initial_owner_generation,
            initial_aero_generation=initial_aero_generation,
            initial_structural_state_sha256=initial_structural,
            initial_aero_state_sha256=initial_aero,
            initial_trajectory_chain_sha256=chain,
            records=[],
        )

    def resume(
        self,
        owner: Q16CudaRealFSIOwner,
        previous: Q16RealFSITrajectoryResult,
        *,
        additional_step_count: int,
        delta_time: float,
    ) -> Q16RealFSITrajectoryResult:
        """Continue one exact live-owner prefix without resetting its chain."""

        if type(owner) is not Q16CudaRealFSIOwner:
            raise TypeError("owner must have exact Q16CudaRealFSIOwner type")
        validated = validate_q16_real_fsi_trajectory(previous)
        additional = _exact_step_count(additional_step_count)
        total = _exact_step_count(validated.completed_step_count + additional)
        dt = _exact_positive_float("delta_time", delta_time)
        if dt != validated.delta_time:
            raise ValueError("resumed trajectory delta_time drift")
        if self.stepper.coupling_tolerance != validated.coupling_tolerance:
            raise ValueError("resumed trajectory coupling tolerance drift")
        if (
            self.stepper.structural_stepper.mass_damping_coefficient
            != validated.mass_damping_coefficient
        ):
            raise ValueError("resumed trajectory damping coefficient drift")
        if (
            self.stepper.relaxation_method != validated.relaxation_method
            or self.stepper.relaxation != validated.initial_relaxation
            or self.stepper.max_coupling_iterations != validated.max_coupling_iterations
        ):
            raise ValueError("resumed trajectory coupling algorithm drift")
        owner._assert_live()
        require_q16_mandatory_aero_mode(owner.aero_owner.current_solver)
        if (
            owner.generation != validated.final_owner_generation
            or owner.aero_owner.generation != validated.final_aero_generation
            or owner.state_sha256 != validated.final_structural_state_sha256
            or _solver_sha256(owner.aero_owner.current_solver)
            != validated.final_aero_state_sha256
        ):
            raise ValueError("resumed trajectory owner does not match its prefix")
        return self._advance_from_prefix(
            owner,
            delta_time=dt,
            total_step_count=total,
            initial_owner_generation=validated.initial_owner_generation,
            initial_aero_generation=validated.initial_aero_generation,
            initial_structural_state_sha256=(validated.initial_structural_state_sha256),
            initial_aero_state_sha256=validated.initial_aero_state_sha256,
            initial_trajectory_chain_sha256=(validated.initial_trajectory_chain_sha256),
            records=list(validated.records),
        )

    def _advance_from_prefix(
        self,
        owner: Q16CudaRealFSIOwner,
        *,
        delta_time: float,
        total_step_count: int,
        initial_owner_generation: int,
        initial_aero_generation: int,
        initial_structural_state_sha256: str,
        initial_aero_state_sha256: str,
        initial_trajectory_chain_sha256: str,
        records: list[Q16RealFSITrajectoryRecord],
    ) -> Q16RealFSITrajectoryResult:
        dt = delta_time
        chain = (
            records[-1].trajectory_chain_sha256
            if records
            else initial_trajectory_chain_sha256
        )
        start_step = len(records) + 1
        for relative_step in range(start_step, total_step_count + 1):
            parent_solver = owner.aero_owner.current_solver
            parent_owner_generation = owner.generation
            parent_aero_generation = owner.aero_owner.generation
            parent_structural = owner.state_sha256
            parent_aero = _solver_sha256(parent_solver)
            parent_steps_done = int(parent_solver._steps_done)
            parent_current_step = int(parent_solver._current_step)
            parent_lev = int(parent_solver.lev_pf.n)
            parent_wake = int(parent_solver.cuda_counters["wake_convection"])
            try:
                step_result = self.stepper.advance(owner, delta_time=dt)
            except Exception as error:
                if (
                    owner.generation != parent_owner_generation
                    or owner.aero_owner.generation != parent_aero_generation
                    or owner.state_sha256 != parent_structural
                    or owner.aero_owner.current_solver is not parent_solver
                    or _solver_sha256(owner.aero_owner.current_solver) != parent_aero
                ):
                    raise RuntimeError(
                        "failed Q16 trajectory step corrupted its committed parent"
                    ) from error
                phase = (
                    error.phase
                    if isinstance(error, Q16RealFSIStepStopped)
                    else type(error).__name__
                )
                raise Q16RealFSITrajectoryStopped(
                    str(error),
                    phase=phase,
                    failed_step_index=relative_step,
                    completed_records=tuple(records),
                    completed_trajectory_chain_sha256=chain,
                    failed_parent_structural_state_sha256=parent_structural,
                    failed_parent_aero_state_sha256=parent_aero,
                ) from error

            record = self._record_step(
                owner=owner,
                result=step_result,
                relative_step=relative_step,
                initial_owner_generation=initial_owner_generation,
                initial_aero_generation=initial_aero_generation,
                parent_structural=parent_structural,
                parent_aero=parent_aero,
                parent_steps_done=parent_steps_done,
                parent_current_step=parent_current_step,
                parent_lev=parent_lev,
                parent_wake=parent_wake,
                previous_chain=chain,
            )
            records.append(record)
            chain = record.trajectory_chain_sha256

        result = Q16RealFSITrajectoryResult(
            delta_time=dt,
            coupling_tolerance=self.stepper.coupling_tolerance,
            mass_damping_coefficient=(
                self.stepper.structural_stepper.mass_damping_coefficient
            ),
            relaxation_method=self.stepper.relaxation_method,
            initial_relaxation=self.stepper.relaxation,
            max_coupling_iterations=self.stepper.max_coupling_iterations,
            requested_step_count=total_step_count,
            completed_step_count=len(records),
            initial_owner_generation=initial_owner_generation,
            final_owner_generation=owner.generation,
            initial_aero_generation=initial_aero_generation,
            final_aero_generation=owner.aero_owner.generation,
            initial_structural_state_sha256=initial_structural_state_sha256,
            final_structural_state_sha256=owner.state_sha256,
            initial_aero_state_sha256=initial_aero_state_sha256,
            final_aero_state_sha256=_solver_sha256(owner.aero_owner.current_solver),
            initial_trajectory_chain_sha256=initial_trajectory_chain_sha256,
            trajectory_chain_sha256=chain,
            records=tuple(records),
            result_sha256="0" * 64,
        )
        result = replace(result, result_sha256=_result_sha256(result))
        return validate_q16_real_fsi_trajectory(result)

    def _record_step(
        self,
        *,
        owner: Q16CudaRealFSIOwner,
        result: Q16RealFSIStepResult,
        relative_step: int,
        initial_owner_generation: int,
        initial_aero_generation: int,
        parent_structural: str,
        parent_aero: str,
        parent_steps_done: int,
        parent_current_step: int,
        parent_lev: int,
        parent_wake: int,
        previous_chain: str,
    ) -> Q16RealFSITrajectoryRecord:
        solver = owner.aero_owner.current_solver
        if (
            owner.generation != initial_owner_generation + relative_step
            or owner.aero_owner.generation != initial_aero_generation + relative_step
            or result.owner_generation != owner.generation
            or solver is not result.committed_solver
        ):
            raise RuntimeError("Q16 trajectory owner publication continuity drift")
        require_q16_mandatory_aero_mode(solver)
        if solver._tev_solved is None or solver._prescribed_wake is not False:
            raise RuntimeError("Q16 trajectory lost joint TEV or free-wake state")
        load = wp.to_torch(result.complete_load.generalized_force)
        if load.device.type != "cuda" or load.dtype is not torch.float64:
            raise RuntimeError("Q16 trajectory complete load left CUDA float64")
        load_norm = float(torch.linalg.vector_norm(load).item())
        if not math.isfinite(load_norm) or load_norm <= 0.0:
            raise FloatingPointError("Q16 trajectory complete load norm is invalid")
        source_force = getattr(solver, "_q16_total_force_w", None)
        if (
            type(source_force) is not torch.Tensor
            or source_force.device.type != "cuda"
            or source_force.dtype is not torch.float64
            or tuple(source_force.shape) != (3,)
            or not bool(torch.isfinite(source_force).all().item())
        ):
            raise RuntimeError(
                "Q16 trajectory source total force is not finite CUDA float64"
            )
        reference = wp.to_torch(self.stepper.structural_stepper._reference_state)
        current = wp.to_torch(owner.state)
        if (
            reference.device.type != "cuda"
            or current.device.type != "cuda"
            or reference.dtype is not torch.float64
            or current.dtype is not torch.float64
            or reference.shape != current.shape
            or reference.ndim != 2
            or reference.shape[0] != 1
            or reference.shape[1] % 6 != 0
        ):
            raise RuntimeError("Q16 trajectory structural observation left CUDA")
        reference_rows = reference.reshape(-1, 6)
        current_rows = current.reshape(-1, 6)
        tip_mask = reference_rows[:, 1] == torch.max(reference_rows[:, 1])
        if int(torch.count_nonzero(tip_mask).item()) < 2:
            raise RuntimeError("Q16 trajectory has no span-tip node section")
        tip_displacement = torch.mean(
            current_rows[tip_mask, :3] - reference_rows[tip_mask, :3], dim=0
        )
        if not bool(torch.isfinite(tip_displacement).all().item()):
            raise FloatingPointError("Q16 span-tip displacement became non-finite")
        operating_point_velocity = float(
            solver.steady_problems[solver._current_step].operating_point.vCg__E
        )
        force_values = tuple(float(value.item()) for value in source_force)
        tip_values = tuple(float(value.item()) for value in tip_displacement)
        record = Q16RealFSITrajectoryRecord(
            step_index=relative_step,
            owner_generation=owner.generation,
            aero_generation=owner.aero_owner.generation,
            parent_structural_state_sha256=parent_structural,
            result_structural_state_sha256=owner.state_sha256,
            parent_aero_state_sha256=parent_aero,
            result_aero_state_sha256=_solver_sha256(solver),
            parent_solver_steps_done=parent_steps_done,
            solver_steps_done=int(solver._steps_done),
            parent_solver_current_step=parent_current_step,
            solver_current_step=int(solver._current_step),
            parent_lev_particle_count=parent_lev,
            lev_particle_count=int(solver.lev_pf.n),
            parent_wake_convection_count=parent_wake,
            wake_convection_count=int(solver.cuda_counters["wake_convection"]),
            step_result_sha256=result.result_sha256,
            complete_load_sha256=result.complete_load.result_sha256,
            complete_load_norm=load_norm,
            operating_point_velocity=operating_point_velocity,
            aerodynamic_force_x_w=force_values[0],
            aerodynamic_force_y_w=force_values[1],
            aerodynamic_force_z_w=force_values[2],
            span_tip_centroid_displacement_x_w=tip_values[0],
            span_tip_centroid_displacement_y_w=tip_values[1],
            span_tip_centroid_displacement_z_w=tip_values[2],
            coupling_iteration_count=result.coupling_iteration_count,
            aerodynamic_evaluation_count=result.aerodynamic_evaluation_count,
            coupling_relative_residual=float(result.relative_residual),
            structural_newton_iteration_count=(
                result.structural.newton_iteration_count
            ),
            structural_cg_iteration_count=result.structural.cg_iteration_count,
            structural_gmres_iteration_count=(result.structural.gmres_iteration_count),
            structural_indefinite_fallback_count=(
                result.structural.indefinite_fallback_count
            ),
            structural_relative_residual=float(result.structural.relative_residual_max),
            relaxation_factor_count=len(result.relaxation_factor_history),
            relaxation_factor_min=(
                min(result.relaxation_factor_history)
                if result.relaxation_factor_history
                else 0.0
            ),
            relaxation_factor_max=(
                max(result.relaxation_factor_history)
                if result.relaxation_factor_history
                else 0.0
            ),
            relaxation_factor_final=(
                result.relaxation_factor_history[-1]
                if result.relaxation_factor_history
                else 0.0
            ),
            kinetic_energy_start=result.work_balance.kinetic_energy_start,
            kinetic_energy_end=result.work_balance.kinetic_energy_end,
            kinetic_energy_change=result.work_balance.kinetic_energy_change,
            internal_trapezoidal_work=(result.work_balance.internal_trapezoidal_work),
            damping_trapezoidal_work=(result.work_balance.damping_trapezoidal_work),
            external_trapezoidal_work=(result.work_balance.external_trapezoidal_work),
            work_balance_residual=result.work_balance.balance_residual,
            work_balance_relative_residual=(
                result.work_balance.relative_balance_residual
            ),
            state_increment_norm=result.work_balance.state_increment_norm,
            deformation_norm_end=result.work_balance.deformation_norm_end,
            velocity_norm_end=result.work_balance.velocity_norm_end,
            acceleration_norm_end=result.work_balance.acceleration_norm_end,
            previous_trajectory_chain_sha256=previous_chain,
            record_sha256="0" * 64,
            trajectory_chain_sha256="0" * 64,
        )
        record_sha = _record_sha256(record)
        return replace(
            record,
            record_sha256=record_sha,
            trajectory_chain_sha256=_chain_sha256(previous_chain, record_sha),
        )


__all__ = [
    "MAX_Q16_REAL_FSI_TRAJECTORY_STEP_COUNT",
    "Q16CudaRealFSITrajectory",
    "Q16RealFSITrajectoryRecord",
    "Q16RealFSITrajectoryResult",
    "Q16RealFSITrajectoryStopped",
    "validate_q16_real_fsi_trajectory",
]
