"""Real-owner gates for batched DVM -> node ribbon -> TE/free-wake Q16 trials."""
from __future__ import annotations

import hashlib
import pickle

import pytest
import torch

from bing_joint_ptera import JointConfig
from bing_joint_ptera_gpu import (
    CudaJointLEVTEVSolver,
    _dvm_ptera_pin_active_mask,
)
from q16_incremental_ptera_owner import Q16CudaIncrementalAeroSession
from test_ptera_gpu_active_lev import _problem


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _solver(steps: int = 4) -> CudaJointLEVTEVSolver:
    solver = CudaJointLEVTEVSolver(
        _problem(steps),
        JointConfig(
            enable_lev=True,
            joint_tev=True,
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


def _sha(solver: CudaJointLEVTEVSolver) -> str:
    torch.cuda.synchronize(solver.cuda_device)
    return hashlib.sha256(pickle.dumps(solver, protocol=5)).hexdigest()


def test_dvm_release_event_and_ptera_separated_state_are_distinct() -> None:
    cell_active = torch.tensor(
        [False, True, False], dtype=torch.bool, device="cuda:0"
    )
    ptera_separated = torch.tensor(
        [True, False, False], dtype=torch.bool, device="cuda:0"
    )
    pin_active = _dvm_ptera_pin_active_mask(cell_active, ptera_separated)

    assert torch.equal(
        pin_active,
        torch.tensor([True, True, False], dtype=torch.bool, device="cuda:0"),
    )
    assert cell_active.tolist() == [False, True, False]
    assert ptera_separated.tolist() == [True, False, False]


def test_incremental_dvm_ribbon_matches_monolithic_real_owner() -> None:
    monolithic = _solver()
    incremental = _solver()
    monolithic.run(
        prescribed_wake=False, calculate_streamlines=False, show_progress=False
    )
    session = Q16CudaIncrementalAeroSession.begin(incremental)
    receipts = [session.advance_one_step() for _ in range(incremental.num_steps)]
    session.finalize()

    assert receipts[-1].lev_particle_count == incremental.lev_pf.n > 0
    assert receipts[-1].wake_convection_count == incremental.num_steps - 1
    assert incremental.cuda_counters == monolithic.cuda_counters
    assert incremental.dvm_source_bank.it == incremental.num_steps
    assert incremental.dvm_source_bank.wake_convection_count == incremental.num_steps
    assert incremental.dvm_source_bank.batch_size == 7
    assert incremental.lev_pf.last_connected_ribbon_diagnostics is not None
    assert incremental.lev_pf.last_connected_ribbon_diagnostics["seam_count"] == 0
    assert bool(torch.all(incremental.lev_pf.source_strips_cuda >= 0).item())
    assert torch.equal(
        incremental._cuda_particle_bound_strengths,
        incremental._cuda_bound_strengths,
    )
    assert torch.equal(
        torch.as_tensor(
            incremental._last_bound,
            dtype=torch.float64,
            device=incremental.cuda_device,
        ),
        incremental._cuda_bound_strengths,
    )
    assert torch.equal(incremental.lev_pf.positions_cuda, monolithic.lev_pf.positions_cuda)
    assert torch.equal(incremental.lev_pf.gammas_cuda, monolithic.lev_pf.gammas_cuda)
    assert torch.equal(incremental.dvm_source_bank.tg, monolithic.dvm_source_bank.tg)
    assert torch.equal(
        incremental._cuda_dvm_frontier_nodes,
        monolithic._cuda_dvm_frontier_nodes,
    )
    assert any(
        row["dvm_newborn_normal_influence_max_abs"] > 0.0
        for row in incremental.diag
    )
    assert max(row["kelvin_eq9_max_abs"] for row in incremental.diag) <= 1.0e-12
    assert max(
        row["dvm_ptera_lesp_pin_max_abs"] for row in incremental.diag
    ) <= 1.0e-12
    assert max(
        row["dvm_ptera_retained_neumann_max_abs"] for row in incremental.diag
    ) <= 1.0e-12
    final_active = incremental._cuda_dvm_last_ptera_pin_active
    torch.testing.assert_close(
        incremental._cuda_dvm_last_ptera_lesp[final_active],
        incremental._cuda_dvm_last_source_a0[final_active],
        rtol=0.0,
        atol=1.0e-12,
    )
    assert len(incremental.impulse_force) == 0
    assert incremental.cuda_counters["impulse"] == 0
    assert torch.count_nonzero(
        incremental._q16_unresolved_impulse_force_w
    ).item() == 0
    assert torch.count_nonzero(
        incremental._q16_impulse_strip_force_w
    ).item() == 0
    assert float(
        torch.max(torch.abs(incremental._diagnostic_vortex_impulse_force_w)).item()
    ) > 0.0
    assert incremental.diag[-1]["load_owner"] == "ptera_kj_plus_dgamma"


def test_predictor_forks_advance_dvm_particles_and_free_wake_without_parent_drift() -> None:
    parent = _solver()
    parent_session = Q16CudaIncrementalAeroSession.begin(parent)
    parent_session.advance_one_step()
    parent_sha = _sha(parent)
    parent_particles = parent.lev_pf.n
    parent_source_step = parent.dvm_source_bank.it
    parent_wake_count = parent.cuda_counters["wake_convection"]

    trial_a = parent_session.fork()
    trial_b = parent_session.fork()
    receipt_a = trial_a.advance_one_step()
    receipt_b = trial_b.advance_one_step()
    assert receipt_a == receipt_b
    assert trial_a.solver.dvm_source_bank.it == parent_source_step + 1
    assert trial_a.solver.lev_pf.n > parent_particles
    assert (
        trial_a.solver.cuda_counters["wake_convection"]
        == parent_wake_count + 1
    )
    assert trial_a.solver.cuda_counters["dvm_frontier_advance"] == 1

    assert _sha(parent) == parent_sha
    assert parent._steps_done == 1
    assert parent.lev_pf.n == parent_particles
    assert parent.dvm_source_bank.it == parent_source_step
    assert parent.cuda_counters["wake_convection"] == parent_wake_count


@pytest.mark.parametrize(
    ("joint_tev", "lev_start_step"),
    ((False, 0), (True, 1)),
)
def test_dvm_ribbon_mode_rejects_reduced_or_delayed_configuration(
    joint_tev: bool, lev_start_step: int
) -> None:
    with pytest.raises(ValueError):
        CudaJointLEVTEVSolver(
            _problem(2),
            JointConfig(
                enable_lev=True,
                joint_tev=joint_tev,
                lev_start_step=lev_start_step,
                separated_source="dvm_node_ribbon",
            ),
            device="cuda:0",
        )
