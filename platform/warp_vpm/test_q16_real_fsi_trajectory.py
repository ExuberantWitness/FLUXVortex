"""Bounded multi-step Q16 / separated-LEV / free-wake trajectory gates."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch
import warp as wp

from q16_incremental_ptera_owner import Q16CudaIncrementalAeroSession
from q16_real_aero_branch_transaction import Q16CudaAeroSolverOwner
from q16_real_aero_branch_transaction import _solver_sha256
from q16_real_fsi_coupling import Q16CudaRealFSIOwner
from q16_real_fsi_trajectory import (
    MAX_Q16_REAL_FSI_TRAJECTORY_STEP_COUNT,
    Q16CudaRealFSITrajectory,
    Q16RealFSITrajectoryResult,
    Q16RealFSITrajectoryStopped,
    _chain_sha256,
    _record_sha256,
    validate_q16_real_fsi_trajectory,
)
from test_q16_real_fsi_coupling import _build


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not wp.is_cuda_available(), reason="CUDA required"
)


@pytest.fixture(scope="module")
def completed_two_step_trajectory():
    owner, stepper = _build(max_coupling_iterations=64)
    initial_state_sha256 = owner.state_sha256
    prefix_owner = Q16CudaRealFSIOwner(
        aero_owner=Q16CudaAeroSolverOwner(
            Q16CudaIncrementalAeroSession.resume(owner.aero_owner.current_solver)
            .fork()
            .solver
        ),
        state=owner.state,
        velocity=owner.velocity,
        acceleration=owner.acceleration,
    )
    runner = Q16CudaRealFSITrajectory(stepper)
    result = runner.advance(owner, step_count=2, delta_time=0.04)
    return owner, result, initial_state_sha256, prefix_owner, stepper


def test_two_steps_preserve_structural_lev_tev_and_free_wake_continuity(
    completed_two_step_trajectory,
) -> None:
    owner, result, initial_state_sha256, _, _ = completed_two_step_trajectory

    assert type(result) is Q16RealFSITrajectoryResult
    assert validate_q16_real_fsi_trajectory(result) is result
    assert result.completed_step_count == 2
    assert len(result.records) == 2
    assert result.initial_structural_state_sha256 == initial_state_sha256
    assert result.final_structural_state_sha256 == owner.state_sha256
    assert result.final_owner_generation == owner.generation == 2
    assert result.final_aero_generation == owner.aero_owner.generation == 2
    assert result.records[0].step_index == 1
    assert result.records[1].step_index == 2
    assert result.records[0].owner_generation == 1
    assert result.records[1].owner_generation == 2
    assert result.records[0].solver_steps_done == 2
    assert result.records[1].solver_steps_done == 3
    assert result.records[0].solver_current_step == 1
    assert result.records[1].solver_current_step == 2
    assert result.records[0].wake_convection_count == 1
    assert result.records[1].wake_convection_count == 2
    assert result.records[0].lev_particle_count == 24
    assert result.records[1].lev_particle_count == 36
    assert result.records[0].result_structural_state_sha256 == (
        result.records[1].parent_structural_state_sha256
    )
    assert result.records[0].result_aero_state_sha256 == (
        result.records[1].parent_aero_state_sha256
    )
    assert all(record.coupling_relative_residual <= 2.0e-7 for record in result.records)
    assert all(record.complete_load_norm > 0.0 for record in result.records)
    assert all(record.operating_point_velocity == 4.0 for record in result.records)
    final_force = owner.aero_owner.current_solver._q16_total_force_w
    assert tuple(
        getattr(result.records[-1], f"aerodynamic_force_{axis}_w")
        for axis in "xyz"
    ) == tuple(float(value.item()) for value in final_force)
    reference_rows = (
        completed_two_step_trajectory[4]
        .structural_stepper._reference_state.numpy()
        .reshape(-1, 6)
    )
    current_rows = owner.state.numpy().reshape(-1, 6)
    tip = reference_rows[:, 1] == np.max(reference_rows[:, 1])
    expected_tip = np.mean(current_rows[tip, :3] - reference_rows[tip, :3], axis=0)
    np.testing.assert_array_equal(
        np.asarray(
            [
                result.records[-1].span_tip_centroid_displacement_x_w,
                result.records[-1].span_tip_centroid_displacement_y_w,
                result.records[-1].span_tip_centroid_displacement_z_w,
            ]
        ),
        expected_tip,
    )
    assert all(
        record.work_balance_relative_residual <= 1.0e-6 for record in result.records
    )
    assert result.records[0].kinetic_energy_end == (
        result.records[1].kinetic_energy_start
    )
    assert result.records[1].trajectory_chain_sha256 == result.trajectory_chain_sha256


def test_record_tamper_is_rejected_by_independent_trajectory_validator(
    completed_two_step_trajectory,
) -> None:
    _, result, _, _, _ = completed_two_step_trajectory
    forged_record = replace(
        result.records[1],
        lev_particle_count=result.records[1].lev_particle_count + 1,
    )
    forged = replace(result, records=(result.records[0], forged_record))
    with pytest.raises(ValueError, match="record digest"):
        validate_q16_real_fsi_trajectory(forged)


def test_signed_periodic_observation_tamper_is_sealed_by_record_digest(
    completed_two_step_trajectory,
) -> None:
    _, result, _, _, _ = completed_two_step_trajectory
    forged_record = replace(
        result.records[1],
        aerodynamic_force_z_w=result.records[1].aerodynamic_force_z_w + 1.0,
    )
    forged = replace(result, records=(result.records[0], forged_record))
    with pytest.raises(ValueError, match="record digest"):
        validate_q16_real_fsi_trajectory(forged)


@pytest.mark.parametrize(
    "bad_count", [0, -1, MAX_Q16_REAL_FSI_TRAJECTORY_STEP_COUNT + 1, True]
)
def test_trajectory_count_cap_rejects_before_owner_mutation(
    completed_two_step_trajectory, bad_count: int
) -> None:
    owner, _, _, _, stepper = completed_two_step_trajectory
    before = (
        owner.generation,
        owner.aero_owner.generation,
        owner.state_sha256,
        owner.aero_owner.current_solver,
    )
    with pytest.raises(ValueError, match="step_count"):
        Q16CudaRealFSITrajectory(stepper).advance(
            owner, step_count=bad_count, delta_time=0.04
        )
    assert (
        owner.generation,
        owner.aero_owner.generation,
        owner.state_sha256,
        owner.aero_owner.current_solver,
    ) == before


def test_exhausted_horizon_keeps_exact_two_step_prefix_and_failed_owner_unchanged(
    completed_two_step_trajectory,
) -> None:
    _, expected, _, owner, stepper = completed_two_step_trajectory
    runner = Q16CudaRealFSITrajectory(stepper)

    with pytest.raises(Q16RealFSITrajectoryStopped) as caught:
        runner.advance(owner, step_count=3, delta_time=0.04)

    stopped = caught.value
    assert stopped.failed_step_index == 3
    assert stopped.step_began is True
    assert stopped.completed_step_count == 2
    previous_chain = stopped.completed_records[0].previous_trajectory_chain_sha256
    for record in stopped.completed_records:
        assert _record_sha256(record) == record.record_sha256
        assert _chain_sha256(previous_chain, record.record_sha256) == (
            record.trajectory_chain_sha256
        )
        previous_chain = record.trajectory_chain_sha256
    assert stopped.completed_trajectory_chain_sha256 == previous_chain
    semantic_fields = (
        "solver_steps_done",
        "solver_current_step",
        "lev_particle_count",
        "wake_convection_count",
        "complete_load_norm",
        "operating_point_velocity",
        "aerodynamic_force_x_w",
        "aerodynamic_force_y_w",
        "aerodynamic_force_z_w",
        "span_tip_centroid_displacement_x_w",
        "span_tip_centroid_displacement_y_w",
        "span_tip_centroid_displacement_z_w",
        "coupling_iteration_count",
        "aerodynamic_evaluation_count",
        "coupling_relative_residual",
        "structural_newton_iteration_count",
        "structural_cg_iteration_count",
        "structural_gmres_iteration_count",
        "structural_indefinite_fallback_count",
        "structural_relative_residual",
        "kinetic_energy_start",
        "kinetic_energy_end",
        "kinetic_energy_change",
        "internal_trapezoidal_work",
        "damping_trapezoidal_work",
        "external_trapezoidal_work",
        "work_balance_residual",
        "work_balance_relative_residual",
        "state_increment_norm",
        "deformation_norm_end",
        "velocity_norm_end",
        "acceleration_norm_end",
    )
    for observed, reference in zip(
        stopped.completed_records, expected.records, strict=True
    ):
        assert tuple(getattr(observed, field) for field in semantic_fields) == tuple(
            getattr(reference, field) for field in semantic_fields
        )
    assert owner.generation == owner.aero_owner.generation == 2
    assert owner.state_sha256 == expected.final_structural_state_sha256
    assert stopped.failed_parent_structural_state_sha256 == owner.state_sha256
    assert stopped.failed_parent_aero_state_sha256 == (
        stopped.completed_records[-1].result_aero_state_sha256
    )
    assert stopped.failed_parent_aero_state_sha256 == _solver_sha256(
        owner.aero_owner.current_solver
    )
    assert stopped.__cause__ is not None
