"""Tests-first gates for the real incremental CUDA aerodynamic owner."""

from __future__ import annotations

import hashlib
import pickle

import numpy as np
import pytest
import torch

from bing_joint_ptera import JointConfig
from bing_joint_ptera_gpu import CudaJointLEVTEVSolver
from fluxvortex.warp_fsi.q16_aero_load_packet import Q16CudaAerodynamicLoadPacket
from q16_incremental_ptera_owner import (
    Q16CudaIncrementalAeroSession,
    Q16IncrementalAeroLifecycleError,
)
from test_ptera_gpu_active_lev import _problem


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _solver(*, steps: int = 6) -> CudaJointLEVTEVSolver:
    solver = CudaJointLEVTEVSolver(
        _problem(steps),
        JointConfig(
            enable_lev=True,
            joint_tev=True,
            lev_start_step=1,
            particle_capacity=512,
        ),
        device="cuda:0",
    )
    solver._prescribed_wake = False
    return solver


def _pickle_sha(solver: CudaJointLEVTEVSolver) -> str:
    torch.cuda.synchronize(solver.cuda_device)
    return hashlib.sha256(pickle.dumps(solver, protocol=5)).hexdigest()


def _assert_tensor_equal(actual: torch.Tensor, expected: torch.Tensor) -> None:
    assert actual.device == expected.device
    assert actual.dtype is expected.dtype
    assert torch.equal(actual, expected)


def _assert_ledger_equal(actual: list[dict], expected: list[dict]) -> None:
    assert len(actual) == len(expected)
    for actual_row, expected_row in zip(actual, expected, strict=True):
        assert actual_row.keys() == expected_row.keys()
        for key in actual_row:
            left = actual_row[key]
            right = expected_row[key]
            if type(left) is torch.Tensor:
                _assert_tensor_equal(left, right)
            elif type(left) is np.ndarray:
                np.testing.assert_array_equal(left, right)
            else:
                assert left == right


def test_incremental_all_steps_are_bitwise_equal_to_monolithic_run() -> None:
    monolithic = _solver()
    incremental = _solver()
    monolithic.run(
        prescribed_wake=False,
        calculate_streamlines=False,
        show_progress=False,
    )

    session = Q16CudaIncrementalAeroSession.begin(incremental)
    receipts = [session.advance_one_step() for _ in range(incremental.num_steps)]
    session.finalize()

    assert [receipt.step_index for receipt in receipts] == list(
        range(incremental.num_steps)
    )
    assert receipts[-1].lev_particle_count == incremental.lev_pf.n > 0
    assert receipts[-1].wake_convection_count == incremental.num_steps - 1
    assert (
        receipts[-1].load_packet_sha256
        == Q16CudaAerodynamicLoadPacket.from_solver(incremental).packet_sha256
    )
    assert session.status == "finalized"
    assert incremental.ran is monolithic.ran is True
    assert incremental._steps_done == monolithic._steps_done == incremental.num_steps
    assert incremental.cuda_counters == monolithic.cuda_counters
    assert incremental.lev_pf.n == monolithic.lev_pf.n
    _assert_tensor_equal(
        incremental.lev_pf.positions_cuda, monolithic.lev_pf.positions_cuda
    )
    _assert_tensor_equal(incremental.lev_pf.gammas_cuda, monolithic.lev_pf.gammas_cuda)
    _assert_tensor_equal(incremental.lev_pf.sigmas_cuda, monolithic.lev_pf.sigmas_cuda)
    _assert_tensor_equal(incremental._tev_solved, monolithic._tev_solved)
    _assert_ledger_equal(incremental.ledger, monolithic.ledger)
    for actual, expected in zip(
        incremental.steady_problems, monolithic.steady_problems, strict=True
    ):
        np.testing.assert_array_equal(
            actual.airplanes[0].forces_W, expected.airplanes[0].forces_W
        )
        np.testing.assert_array_equal(
            actual.airplanes[0].moments_W_CgP1,
            expected.airplanes[0].moments_W_CgP1,
        )
        actual_wing = actual.airplanes[0].wings[0]
        expected_wing = expected.airplanes[0].wings[0]
        np.testing.assert_array_equal(
            actual_wing.gridWrvp_GP1_CgP1,
            expected_wing.gridWrvp_GP1_CgP1,
        )


def test_completed_step_can_branch_and_advance_without_mutating_parent() -> None:
    parent = _solver(steps=4)
    parent_session = Q16CudaIncrementalAeroSession.begin(parent)
    first = parent_session.advance_one_step()
    parent_sha = _pickle_sha(parent)
    parent_particle_count = parent.lev_pf.n
    parent_counters = parent.cuda_counters.copy()

    branch_session_a = parent_session.fork()
    branch_session_b = parent_session.fork()
    branch_a = branch_session_a.solver
    branch_b = branch_session_b.solver
    receipt_a = branch_session_a.advance_one_step()
    receipt_b = branch_session_b.advance_one_step()

    assert first.step_index == 0
    assert receipt_a == receipt_b
    assert receipt_a.step_index == 1
    assert branch_a._steps_done == branch_b._steps_done == 2
    assert _pickle_sha(parent) == parent_sha
    assert parent._steps_done == 1
    assert parent.lev_pf.n == parent_particle_count
    assert parent.cuda_counters == parent_counters


def test_incremental_lifecycle_rejects_out_of_order_and_mixed_execution() -> None:
    solver = _solver(steps=2)
    session = Q16CudaIncrementalAeroSession.begin(solver)
    with pytest.raises(Q16IncrementalAeroLifecycleError, match="all steps"):
        session.finalize()
    session.advance_one_step()
    session.advance_one_step()
    with pytest.raises(Q16IncrementalAeroLifecycleError, match="no remaining"):
        session.advance_one_step()
    session.finalize()
    with pytest.raises(Q16IncrementalAeroLifecycleError, match="finalized"):
        session.finalize()
    with pytest.raises(Q16IncrementalAeroLifecycleError, match="already run"):
        Q16CudaIncrementalAeroSession.begin(solver)

    mixed = _solver(steps=2)
    mixed_session = Q16CudaIncrementalAeroSession.begin(mixed)
    mixed.run(
        prescribed_wake=False,
        calculate_streamlines=False,
        show_progress=False,
    )
    with pytest.raises(Q16IncrementalAeroLifecycleError, match="outside session"):
        mixed_session.advance_one_step()


@pytest.mark.parametrize(
    ("enable_lev", "joint_tev", "prescribed_wake"),
    [(False, True, False), (True, False, False), (True, True, True)],
)
def test_incremental_owner_rejects_every_reduced_aerodynamic_mode(
    enable_lev: bool,
    joint_tev: bool,
    prescribed_wake: bool,
) -> None:
    solver = CudaJointLEVTEVSolver(
        _problem(2),
        JointConfig(
            enable_lev=enable_lev,
            joint_tev=joint_tev,
            lev_start_step=0,
            particle_capacity=128,
        ),
        device="cuda:0",
    )
    solver._prescribed_wake = prescribed_wake
    with pytest.raises(RuntimeError):
        Q16CudaIncrementalAeroSession.begin(solver)


def test_resume_rejects_unissued_or_drifted_partial_solver() -> None:
    pristine = _solver(steps=3)
    with pytest.raises(Q16IncrementalAeroLifecycleError, match="not initialized"):
        Q16CudaIncrementalAeroSession.resume(pristine)

    solver = _solver(steps=3)
    session = Q16CudaIncrementalAeroSession.begin(solver)
    session.advance_one_step()
    assert Q16CudaIncrementalAeroSession.resume(solver).next_step == 1
    solver._steps_done = 2
    with pytest.raises(Q16IncrementalAeroLifecycleError, match="state drift"):
        Q16CudaIncrementalAeroSession.resume(solver)
