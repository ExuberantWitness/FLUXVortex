"""Real Q16 FSI gates for conditional DVM node-ribbon separation."""

from __future__ import annotations

import hashlib
import pickle

import pytest
import torch
import warp as wp

from bing_joint_ptera import JointConfig
from bing_joint_ptera_gpu import CudaJointLEVTEVSolver
from fluxvortex.warp_fsi import config
from fluxvortex.warp_fsi.q16_prescribed_endpoint_load import (
    Q16CudaPrescribedEndpointLoad,
)
from q16_real_aero_branch_transaction import _solver_sha256
from q16_real_fsi_coupling import Q16RealFSIStepStopped
from q16_real_fsi_trajectory import (
    Q16CudaRealFSITrajectory,
    Q16RealFSITrajectoryStopped,
    validate_q16_real_fsi_trajectory,
)
from test_ptera_gpu_active_lev import _problem
from test_q16_real_fsi_coupling import _build


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not wp.is_cuda_available(), reason="CUDA required"
)


def _dvm_solver(*, steps: int = 4) -> CudaJointLEVTEVSolver:
    solver = CudaJointLEVTEVSolver(
        _problem(steps),
        JointConfig(
            enable_lev=True,
            joint_tev=True,
            lesp_crit=0.11,
            lev_start_step=0,
            separated_source="dvm_node_ribbon",
            particle_capacity=4096,
            dvm_ndiv=20,
            dvm_naterm=8,
            dvm_max_wake=32,
        ),
        device="cuda:0",
    )
    solver._prescribed_wake = False
    return solver


def _active_build(**kwargs: object):
    options: dict[str, object] = {
        "aerodynamic_solver": _dvm_solver(),
        "max_coupling_iterations": 30,
        "pitch_angle_degrees": 20.0,
        "required_separated_source": "dvm_node_ribbon",
    }
    options.update(kwargs)
    return _build(**options)


def _long_active_build():
    return _build(
        aerodynamic_solver=_dvm_solver(steps=9),
        max_coupling_iterations=64,
        newton_tolerance=5.0e-8,
        max_newton_iterations=30,
        young_modulus=1.0e9,
        mass_damping_coefficient=20.0,
        pitch_angle_degrees=20.0,
        required_separated_source="dvm_node_ribbon",
    )


def _pickle_sha(value: object) -> str:
    torch.cuda.synchronize()
    return hashlib.sha256(pickle.dumps(value, protocol=5)).hexdigest()


def test_subcritical_dvm_fsi_keeps_mechanism_live_without_forcing_release() -> None:
    owner, stepper = _build(
        aerodynamic_solver=_dvm_solver(),
        max_coupling_iterations=30,
        pitch_angle_degrees=5.0,
        required_separated_source="dvm_node_ribbon",
    )
    assert owner.aero_owner.current_solver.lev_pf.n == 0

    result = stepper.advance(owner, delta_time=0.04)
    solver = result.committed_solver

    assert solver.lev_pf.n == 0
    assert solver.dvm_source_bank.it == solver._steps_done == 2
    assert all(row["lev_strips"] == 0 for row in solver.diag)
    assert all(row["dvm_ptera_pin_strips"] == 0 for row in solver.diag)
    assert solver.diag[-1]["load_owner"] == "ptera_kj_plus_dgamma"
    assert solver.cuda_counters["impulse"] == 0


def test_active_dvm_ribbon_enters_real_q16_fsi_through_unique_surface_load() -> None:
    owner, stepper = _active_build()
    initial = owner.aero_owner.current_solver
    initial_particles = initial.lev_pf.n
    assert initial_particles > 0

    result = stepper.advance(owner, delta_time=0.04)
    solver = result.committed_solver
    resolved = wp.to_torch(result.complete_load.resolved.generalized_force)
    impulse = wp.to_torch(result.complete_load.lev_impulse_generalized_force)
    total = wp.to_torch(result.complete_load.generalized_force)

    assert owner.generation == owner.aero_owner.generation == 1
    assert solver.lev_pf.n > initial_particles
    assert solver.dvm_source_bank.it == solver._steps_done == 2
    assert solver.cuda_counters["wake_convection"] == 1
    assert solver.cuda_counters["dvm_ribbon_shed"] == solver.lev_pf.n
    assert all(row["lev_strips"] > 0 for row in solver.diag)
    assert solver.diag[-1]["load_owner"] == "ptera_kj_plus_dgamma"
    assert solver.cuda_counters["impulse"] == 0
    assert torch.count_nonzero(solver._q16_unresolved_impulse_force_w).item() == 0
    assert (
        float(torch.linalg.vector_norm(solver._diagnostic_vortex_impulse_force_w))
        > 0.0
    )
    assert torch.count_nonzero(impulse).item() == 0
    torch.testing.assert_close(total, resolved, rtol=0.0, atol=0.0)
    assert result.relative_residual <= stepper.coupling_tolerance
    assert result.work_balance.relative_balance_residual <= 1.0e-6


def test_prescribed_load_does_not_replace_conditional_dvm_aero_owner() -> None:
    owner, stepper = _active_build()
    prescribed_t = torch.zeros_like(wp.to_torch(owner.state))
    prescribed_t[:, 2::6] = 1.0e-4
    prescribed = Q16CudaPrescribedEndpointLoad.from_force(
        wp.from_torch(prescribed_t, dtype=config.DTYPE),
        source_id="dvm-additive-regression",
        endpoint_time_s=0.04,
    )

    result = stepper.advance(
        owner,
        delta_time=0.04,
        prescribed_load=prescribed,
    )
    solver = result.committed_solver
    aerodynamic = wp.to_torch(result.complete_load.generalized_force)
    total = wp.to_torch(result.total_external_force)

    assert solver.jcfg.separated_source == "dvm_node_ribbon"
    assert solver.jcfg.lesp_crit == 0.11
    assert solver.diag[-1]["load_owner"] == "ptera_kj_plus_dgamma"
    assert solver.cuda_counters["impulse"] == 0
    assert solver.lev_pf.n == solver.cuda_counters["dvm_ribbon_shed"]
    torch.testing.assert_close(total, aerodynamic + prescribed_t, rtol=0.0, atol=0.0)


def test_active_dvm_fsi_nonconvergence_leaves_both_parents_unchanged() -> None:
    owner, stepper = _active_build(
        coupling_tolerance=1.0e-30,
        max_coupling_iterations=1,
    )
    parent = owner.aero_owner.current_solver
    parent_solver_sha = _solver_sha256(parent)
    parent_pickle_sha = _pickle_sha(parent)
    parent_structural_sha = owner.state_sha256
    parent_particles = parent.lev_pf.n
    parent_dvm_step = parent.dvm_source_bank.it

    with pytest.raises(Q16RealFSIStepStopped, match="did not converge"):
        stepper.advance(owner, delta_time=0.04)

    assert owner.generation == owner.aero_owner.generation == 0
    assert owner.aero_owner.current_solver is parent
    assert owner.state_sha256 == parent_structural_sha
    assert parent.lev_pf.n == parent_particles
    assert parent.dvm_source_bank.it == parent_dvm_step
    assert _solver_sha256(parent) == parent_solver_sha
    assert _pickle_sha(parent) == parent_pickle_sha


def test_two_step_active_dvm_fsi_preserves_source_and_wake_lineage() -> None:
    owner, stepper = _active_build()
    initial_particles = owner.aero_owner.current_solver.lev_pf.n
    result = Q16CudaRealFSITrajectory(stepper).advance(
        owner,
        step_count=2,
        delta_time=0.04,
    )
    assert validate_q16_real_fsi_trajectory(result) is result
    solver = owner.aero_owner.current_solver

    assert result.completed_step_count == 2
    assert owner.generation == owner.aero_owner.generation == 2
    assert solver.jcfg.separated_source == "dvm_node_ribbon"
    assert solver.dvm_source_bank.it == solver._steps_done == 3
    assert solver.cuda_counters["wake_convection"] == 2
    assert solver.lev_pf.n > initial_particles
    assert solver.cuda_counters["impulse"] == 0
    assert all(row["load_owner"] == "ptera_kj_plus_dgamma" for row in solver.diag)
    assert all(record.lev_particle_count > 0 for record in result.records)
    assert all(
        record.work_balance_relative_residual <= 1.0e-6
        for record in result.records
    )


def test_eight_step_active_dvm_fsi_resumes_one_conditional_release_lineage() -> None:
    owner, stepper = _long_active_build()
    runner = Q16CudaRealFSITrajectory(stepper)

    prefix = runner.advance(owner, step_count=4, delta_time=0.04)
    assert validate_q16_real_fsi_trajectory(prefix) is prefix
    prefix_record_sha256 = tuple(record.record_sha256 for record in prefix.records)
    prefix_chain_sha256 = prefix.trajectory_chain_sha256

    completed = runner.resume(
        owner,
        prefix,
        additional_step_count=4,
        delta_time=0.04,
    )
    assert validate_q16_real_fsi_trajectory(completed) is completed
    solver = owner.aero_owner.current_solver

    assert completed.completed_step_count == 8
    assert owner.generation == owner.aero_owner.generation == 8
    assert tuple(record.record_sha256 for record in completed.records[:4]) == (
        prefix_record_sha256
    )
    assert completed.records[3].trajectory_chain_sha256 == prefix_chain_sha256
    assert solver.jcfg.separated_source == "dvm_node_ribbon"
    assert solver.jcfg.lesp_crit == 0.11
    assert solver.dvm_source_bank.it == solver._steps_done == 9
    assert solver.cuda_counters["dvm_source_steps"] == 9
    assert solver.cuda_counters["wake_convection"] == 8
    assert solver.cuda_counters["impulse"] == 0
    assert solver.lev_pf.n == solver.cuda_counters["dvm_ribbon_shed"]
    assert solver.lev_pf.n < solver.jcfg.particle_capacity
    assert torch.count_nonzero(solver._q16_unresolved_impulse_force_w).item() == 0
    assert all(
        row["separated_source"] == "dvm_node_ribbon"
        and row["load_owner"] == "ptera_kj_plus_dgamma"
        and 0 <= row["lev_strips"] <= 3
        and 0 <= row["dvm_ptera_pin_strips"] <= 3
        and 0 <= row["dvm_node_topology_active_count"] <= 4
        and ((row["lev_strips"] > 0) == (row["lesp_pre_max_abs"] > 0.11))
        for row in solver.diag
    )
    assert any(row["lev_strips"] > 0 for row in solver.diag)
    assert all(
        record.solver_steps_done == index + 1
        and record.wake_convection_count == index
        and record.owner_generation == index
        and record.aero_generation == index
        and record.coupling_relative_residual <= stepper.coupling_tolerance
        and record.structural_relative_residual
        <= stepper.structural_stepper.newton_tolerance
        and record.work_balance_relative_residual <= 1.0e-6
        for index, record in enumerate(completed.records, start=1)
    )

    before_failed_coordinate = (
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
    assert caught.value.failed_step_index == 9
    assert caught.value.completed_step_count == 8
    assert (
        owner.generation,
        owner.aero_owner.generation,
        owner.state_sha256,
        owner.aero_owner.current_solver,
        _solver_sha256(owner.aero_owner.current_solver),
    ) == before_failed_coordinate


def test_dvm_fsi_contract_rejects_legacy_source_before_trial() -> None:
    owner, stepper = _build(required_separated_source="dvm_node_ribbon")
    parent = owner.aero_owner.current_solver
    parent_sha = _solver_sha256(parent)
    structural_sha = owner.state_sha256

    with pytest.raises(RuntimeError, match="source differs"):
        stepper.advance(owner, delta_time=0.04)

    assert owner.generation == owner.aero_owner.generation == 0
    assert owner.aero_owner.current_solver is parent
    assert owner.state_sha256 == structural_sha
    assert _solver_sha256(parent) == parent_sha
