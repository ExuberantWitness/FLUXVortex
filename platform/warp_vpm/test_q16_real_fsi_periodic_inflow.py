"""Three-cycle real Q16 FSI gate under native periodic Ptera inflow."""

from __future__ import annotations

import json
import math
import os

import numpy as np
import pytest
import torch
import warp as wp

from q16_periodic_fsi import audit_q16_periodic_fsi
from q16_real_fsi_trajectory import (
    Q16CudaRealFSITrajectory,
    Q16RealFSITrajectoryStopped,
    validate_q16_real_fsi_trajectory,
)
from test_ptera_gpu_active_lev import _problem
from test_q16_incremental_trial_geometry import _solver
from test_q16_real_fsi_coupling import _build


MEAN_VELOCITY = 4.0
VELOCITY_AMPLITUDE = 0.4
PERIOD = 0.64
DELTA_TIME = 0.04
STEPS_PER_CYCLE = 16
CYCLE_COUNT = 3
FSI_STEP_COUNT = STEPS_PER_CYCLE * CYCLE_COUNT
AERODYNAMIC_COORDINATE_COUNT = FSI_STEP_COUNT + 1


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not wp.is_cuda_available(), reason="CUDA required"
)


def _periodic_solver():
    problem = _problem(
        AERODYNAMIC_COORDINATE_COUNT,
        delta_time=DELTA_TIME,
        velocity_amplitude=VELOCITY_AMPLITUDE,
        velocity_period=PERIOD,
    )
    return _solver(problem=problem, particle_capacity=2048)


def test_periodic_ptera_coordinates_repeat_exact_declared_phases() -> None:
    solver = _periodic_solver()
    observed = np.asarray(
        [problem.operating_point.vCg__E for problem in solver.steady_problems],
        dtype=np.float64,
    )
    times = DELTA_TIME * np.arange(AERODYNAMIC_COORDINATE_COUNT, dtype=np.float64)
    expected = MEAN_VELOCITY + VELOCITY_AMPLITUDE * np.sin(
        2.0 * math.pi * times / PERIOD
    )
    np.testing.assert_allclose(observed, expected, rtol=0.0, atol=2.0e-15)
    phase_rows = observed[1:].reshape(CYCLE_COUNT, STEPS_PER_CYCLE)
    np.testing.assert_allclose(
        phase_rows[1:],
        np.repeat(phase_rows[0:1], CYCLE_COUNT - 1, axis=0),
        rtol=0.0,
        atol=2.0e-15,
    )
    assert solver.jcfg.enable_lev is True
    assert solver.jcfg.joint_tev is True
    assert solver._prescribed_wake is False
    assert solver.lev_pf.capacity == 2048


@pytest.mark.skipif(
    os.environ.get("FLUXV_RUN_PERIODIC_FSI") != "1",
    reason="opt-in exploration; CASE reproduction has priority",
)
def test_three_periods_real_q16_lev_tev_free_wake_observed_repeatability() -> None:
    owner, stepper = _build(
        max_coupling_iterations=64,
        newton_tolerance=3.0e-7,
        max_newton_iterations=30,
        aerodynamic_solver=_periodic_solver(),
        young_modulus=1.0e10,
        mass_damping_coefficient=100.0,
    )
    try:
        result = Q16CudaRealFSITrajectory(stepper).advance(
            owner,
            step_count=FSI_STEP_COUNT,
            delta_time=DELTA_TIME,
        )
    except Q16RealFSITrajectoryStopped as error:
        cause = error.__cause__
        last = error.completed_records[-1] if error.completed_records else None
        print(
            json.dumps(
                {
                    "status": "STOPPED",
                    "failed_step_index": error.failed_step_index,
                    "completed_step_count": error.completed_step_count,
                    "phase": error.phase,
                    "completed_trajectory_chain_sha256": (
                        error.completed_trajectory_chain_sha256
                    ),
                    "owner_generation_after_failure": owner.generation,
                    "cause_type": type(cause).__name__,
                    "cause_phase": getattr(cause, "phase", None),
                    "cause_newton_iterations": getattr(
                        cause, "newton_iteration_count", None
                    ),
                    "cause_relative_residual": getattr(
                        cause, "relative_residual_max", None
                    ),
                    "last_lev_particle_count": (
                        last.lev_particle_count if last is not None else None
                    ),
                    "last_wake_convection_count": (
                        last.wake_convection_count if last is not None else None
                    ),
                    "last_deformation_norm": (
                        last.deformation_norm_end if last is not None else None
                    ),
                    "last_tip_z": (
                        last.span_tip_centroid_displacement_z_w
                        if last is not None
                        else None
                    ),
                    "last_force_z": (
                        last.aerodynamic_force_z_w if last is not None else None
                    ),
                },
                allow_nan=False,
                sort_keys=True,
            ),
            flush=True,
        )
        raise
    assert validate_q16_real_fsi_trajectory(result) is result
    audit = audit_q16_periodic_fsi(
        result,
        period=PERIOD,
        steps_per_cycle=STEPS_PER_CYCLE,
        input_repeat_tolerance=1.0e-13,
        force_tolerance=0.10,
        span_tip_tolerance=0.05,
        device="cuda:0",
    )

    assert result.completed_step_count == FSI_STEP_COUNT
    assert result.final_owner_generation == owner.generation == FSI_STEP_COUNT
    assert result.final_aero_generation == owner.aero_owner.generation == FSI_STEP_COUNT
    assert result.records[-1].solver_steps_done == AERODYNAMIC_COORDINATE_COUNT
    assert result.records[-1].wake_convection_count == FSI_STEP_COUNT
    assert all(
        record.coupling_relative_residual <= 2.0e-7
        and record.structural_relative_residual <= 3.0e-7
        and record.work_balance_relative_residual <= 1.0e-6
        for record in result.records
    )
    assert owner.aero_owner.current_solver._prescribed_wake is False
    assert owner.aero_owner.current_solver._tev_solved is not None
    assert owner.aero_owner.current_solver.lev_pf.n > 0
    assert audit.multi_cycle_integration_pass is True

    print(
        json.dumps(
            {
                "trajectory_result_sha256": result.result_sha256,
                "trajectory_chain_sha256": result.trajectory_chain_sha256,
                "audit_sha256": audit.audit_sha256,
                "completed_steps": result.completed_step_count,
                "final_lev_particle_count": result.records[-1].lev_particle_count,
                "max_coupling_relative_residual": max(
                    record.coupling_relative_residual for record in result.records
                ),
                "max_structural_relative_residual": max(
                    record.structural_relative_residual for record in result.records
                ),
                "max_work_balance_relative_residual": max(
                    record.work_balance_relative_residual for record in result.records
                ),
                "input_repeat_max_abs_error": audit.input_repeat_max_abs_error,
                "cycle_comparisons": [
                    {
                        "previous_cycle": comparison.previous_cycle,
                        "current_cycle": comparison.current_cycle,
                        "force_waveform_relative_l2": (
                            comparison.force_waveform_relative_l2
                        ),
                        "span_tip_waveform_relative_l2": (
                            comparison.span_tip_waveform_relative_l2
                        ),
                    }
                    for comparison in audit.comparisons
                ],
                "multi_cycle_integration_pass": (
                    audit.multi_cycle_integration_pass
                ),
                "observed_periodic_steady_pass": (
                    audit.observed_periodic_steady_pass
                ),
            },
            allow_nan=False,
            sort_keys=True,
        ),
        flush=True,
    )
    assert audit.observed_periodic_steady_pass is True
