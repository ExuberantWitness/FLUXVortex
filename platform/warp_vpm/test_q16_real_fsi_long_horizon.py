"""Long-horizon gate for real Q16 mandatory-separated-flow FSI."""

from __future__ import annotations

import pytest
import torch
import warp as wp

from q16_real_aero_branch_transaction import _solver_sha256
from q16_real_fsi_trajectory import (
    Q16CudaRealFSITrajectory,
    Q16RealFSITrajectoryStopped,
    validate_q16_real_fsi_trajectory,
)
from test_q16_real_fsi_coupling import _build


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not wp.is_cuda_available(), reason="CUDA required"
)


def test_eight_steps_resume_one_exact_q16_lev_tev_free_wake_lineage() -> None:
    owner, stepper = _build(
        max_coupling_iterations=64,
        newton_tolerance=5.0e-8,
        max_newton_iterations=30,
        aerodynamic_step_count=9,
        young_modulus=1.0e9,
        mass_damping_coefficient=20.0,
    )
    runner = Q16CudaRealFSITrajectory(stepper)

    prefix = runner.advance(owner, step_count=4, delta_time=0.04)
    prefix_digests = tuple(record.record_sha256 for record in prefix.records)
    prefix_chain = prefix.trajectory_chain_sha256
    assert validate_q16_real_fsi_trajectory(prefix) is prefix

    completed = runner.resume(
        owner,
        prefix,
        additional_step_count=4,
        delta_time=0.04,
    )
    assert validate_q16_real_fsi_trajectory(completed) is completed
    assert completed.completed_step_count == 8
    assert len(completed.records) == 8
    assert completed.requested_step_count == 8
    assert completed.final_owner_generation == owner.generation == 8
    assert completed.final_aero_generation == owner.aero_owner.generation == 8
    assert completed.records[3].trajectory_chain_sha256 == prefix_chain
    assert tuple(record.record_sha256 for record in completed.records[:4]) == (
        prefix_digests
    )
    assert completed.records[-1].solver_steps_done == 9
    assert completed.records[-1].solver_current_step == 8
    assert completed.records[-1].wake_convection_count == 8
    assert all(
        record.step_index == index
        and record.owner_generation == index
        and record.aero_generation == index
        and record.wake_convection_count == index
        for index, record in enumerate(completed.records, start=1)
    )
    assert all(
        current.lev_particle_count >= previous.lev_particle_count
        for previous, current in zip(
            completed.records, completed.records[1:], strict=False
        )
    )
    assert tuple(record.lev_particle_count for record in completed.records) == (
        24,
        36,
        36,
        36,
        48,
        60,
        72,
        84,
    )
    assert all(
        record.coupling_relative_residual <= stepper.coupling_tolerance
        and record.structural_relative_residual
        <= stepper.structural_stepper.newton_tolerance
        and record.work_balance_relative_residual <= 1.0e-6
        for record in completed.records
    )
    assert all(
        previous.kinetic_energy_end == current.kinetic_energy_start
        for previous, current in zip(
            completed.records, completed.records[1:], strict=False
        )
    )
    assert all(record.damping_trapezoidal_work > 0.0 for record in completed.records)
    assert (
        sum(record.structural_indefinite_fallback_count for record in completed.records)
        >= 1
    )
    before = (
        owner.generation,
        owner.aero_owner.generation,
        owner.state_sha256,
        owner.aero_owner.current_solver,
        _solver_sha256(owner.aero_owner.current_solver),
    )
    with pytest.raises(Q16RealFSITrajectoryStopped) as caught:
        runner.resume(
            owner,
            completed,
            additional_step_count=1,
            delta_time=0.04,
        )
    stopped = caught.value
    assert stopped.failed_step_index == 9
    assert stopped.completed_step_count == 8
    assert stopped.completed_trajectory_chain_sha256 == (
        completed.trajectory_chain_sha256
    )
    assert tuple(record.record_sha256 for record in stopped.completed_records) == tuple(
        record.record_sha256 for record in completed.records
    )
    assert (
        owner.generation,
        owner.aero_owner.generation,
        owner.state_sha256,
        owner.aero_owner.current_solver,
        _solver_sha256(owner.aero_owner.current_solver),
    ) == before
