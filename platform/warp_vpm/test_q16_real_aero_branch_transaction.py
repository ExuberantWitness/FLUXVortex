"""Real CUDA solver ownership for Q16 predictor/corrector branches."""

from __future__ import annotations

import hashlib
import pickle

import numpy as np
import pytest
import torch
import warp as wp

from bing_joint_ptera import JointConfig
from bing_joint_ptera_gpu import CudaJointLEVTEVSolver, _transform_wrench_cuda
from fluxvortex.q16_ancf_mesh import make_rectangular_q16_mesh
from fluxvortex.q16_work_conjugate_transfer import Q16SurfaceTransferMap
from fluxvortex.warp_fsi import config
from fluxvortex.warp_fsi.q16_aero_load_packet import (
    Q16CudaAerodynamicLoadPacket,
    Q16CudaResolvedLoadTransfer,
)
from q16_real_aero_branch_transaction import (
    Q16AeroBranchTransactionError,
    Q16AeroBranchTransactionViolation,
    Q16CudaAeroSolverOwner,
)
from test_ptera_gpu_active_lev import _problem


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _solver() -> CudaJointLEVTEVSolver:
    solver = CudaJointLEVTEVSolver(
        _problem(2),
        JointConfig(
            enable_lev=True,
            joint_tev=True,
            lev_start_step=0,
            particle_capacity=128,
        ),
        device="cuda:0",
    )
    solver._prescribed_wake = False
    return solver


def _pickle_sha(solver: CudaJointLEVTEVSolver) -> str:
    return hashlib.sha256(pickle.dumps(solver, protocol=5)).hexdigest()


def _trial(scale: float = 1.0) -> tuple[wp.array, wp.array]:
    q = np.array([[0.1, -0.2, 0.3, 0.4, -0.5, 0.6]], dtype=np.float64) * scale
    dq = -0.3 * q
    return (
        wp.array(q, dtype=config.DTYPE, device=config.DEVICE),
        wp.array(dq, dtype=config.DTYPE, device=config.DEVICE),
    )


def _advance_branch(
    branch: CudaJointLEVTEVSolver, q: wp.array, _dq: wp.array
) -> wp.array:
    device = branch.cuda_device
    branch.lev_pf.add_particles(
        torch.tensor([[0.1, 0.2, 0.3]], dtype=torch.float64, device=device),
        torch.tensor([[0.0, 0.4, 0.0]], dtype=torch.float64, device=device),
        torch.tensor([0.02], dtype=torch.float64, device=device),
        birth_step=branch._steps_done,
    )
    branch._steps_done += 1
    branch._lev_hist.append(np.array([0.4], dtype=np.float64))
    branch._tev_hist.append(np.array([-0.1], dtype=np.float64))
    branch.ledger.append({"branch": branch._steps_done})
    branch.cuda_counters["particle_shed"] += 1
    result = wp.clone(q)
    wp.launch(
        _scale_kernel,
        dim=result.shape,
        inputs=[result, config.DTYPE(2.0)],
        device=config.DEVICE,
    )
    return result


@wp.kernel
def _scale_kernel(value: wp.array(dtype=config.DTYPE, ndim=2), scale: config.DTYPE):
    batch, dof = wp.tid()
    value[batch, dof] = scale * value[batch, dof]


def test_real_solver_trials_branch_full_cuda_state_and_commit_one_owner() -> None:
    parent = _solver()
    parent_sha = _pickle_sha(parent)
    owner = Q16CudaAeroSolverOwner(parent)
    transaction = owner.begin(_advance_branch)

    first = transaction.evaluate(*_trial(1.0))
    second = transaction.evaluate(*_trial(2.0))
    assert owner.current_solver is parent
    assert _pickle_sha(parent) == parent_sha
    assert first.proposed_solver is not second.proposed_solver
    assert first.proposed_solver.lev_pf.n == second.proposed_solver.lev_pf.n == 1
    with pytest.raises(Q16AeroBranchTransactionError, match="latest issued"):
        transaction.commit(first)

    committed = transaction.commit(second)
    assert committed is second.proposed_solver
    assert owner.current_solver is committed
    assert committed.lev_pf.n == 1
    assert committed._steps_done == 1
    assert len(committed.ledger) == 1
    assert committed.cuda_counters["particle_shed"] == 1
    assert _pickle_sha(parent) == parent_sha
    with pytest.raises(Q16AeroBranchTransactionError, match="closed"):
        transaction.commit(second)


def test_failed_branch_is_discarded_and_clean_retry_matches_fresh_counts() -> None:
    owner = Q16CudaAeroSolverOwner(_solver())
    parent = owner.current_solver
    parent_sha = _pickle_sha(parent)

    def fail(branch: CudaJointLEVTEVSolver, q: wp.array, dq: wp.array) -> wp.array:
        _advance_branch(branch, q, dq)
        raise RuntimeError("injected branch failure")

    transaction = owner.begin(fail)
    with pytest.raises(RuntimeError, match="injected branch failure"):
        transaction.evaluate(*_trial())
    assert transaction.status == "failed"
    assert owner.current_solver is parent
    assert _pickle_sha(parent) == parent_sha
    assert parent.lev_pf.n == 0
    assert parent._steps_done == 0

    retry = owner.begin(_advance_branch)
    proposal = retry.evaluate(*_trial())
    retry.commit(proposal)
    assert owner.current_solver.lev_pf.n == 1
    assert owner.current_solver._steps_done == 1


def test_real_free_wake_lev_tev_trajectory_advances_only_the_selected_branch() -> None:
    parent = _solver()
    parent_sha = _pickle_sha(parent)
    owner = Q16CudaAeroSolverOwner(parent)

    def run_real_branch(
        branch: CudaJointLEVTEVSolver, q: wp.array, _dq: wp.array
    ) -> wp.array:
        branch.run(
            prescribed_wake=False,
            calculate_streamlines=False,
            show_progress=False,
        )
        return wp.zeros_like(q)

    transaction = owner.begin(run_real_branch)
    proposal = transaction.evaluate(*_trial())
    assert owner.current_solver is parent
    assert _pickle_sha(parent) == parent_sha
    assert parent.ran is False
    branch = proposal.proposed_solver
    assert branch.ran is True
    assert branch._prescribed_wake is False
    assert branch.lev_pf.n > 0
    assert branch._tev_solved is not None and branch._tev_solved.is_cuda
    assert branch.cuda_counters["particle_shed"] > 0
    assert branch.cuda_counters["particle_advance"] > 0
    assert branch.cuda_counters["wake_convection"] == branch.num_steps - 1

    packet = Q16CudaAerodynamicLoadPacket.from_solver(branch)
    assert packet.point_count == 5 * len(branch.panels)
    torch.testing.assert_close(
        packet.resolved_force_w + packet.unresolved_impulse_force_w,
        packet.source_total_force_w,
        rtol=0.0,
        atol=2.0e-12,
    )
    torch.testing.assert_close(
        packet.resolved_moment_w,
        packet.source_total_moment_w,
        rtol=0.0,
        atol=2.0e-12,
    )
    assert float(torch.linalg.vector_norm(packet.unresolved_impulse_force_w)) > 0.0

    transaction.commit(proposal)
    assert owner.current_solver is branch
    assert owner.current_solver.ran is True


def test_force_moment_transform_is_a_rigid_wrench_transform() -> None:
    angle = 0.37
    rotation = torch.tensor(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float64,
        device="cuda:0",
    )
    translation = torch.tensor([0.7, -0.3, 1.2], dtype=torch.float64, device="cuda:0")
    forces = torch.tensor(
        [[0.4, -0.8, 1.1], [-0.2, 0.6, 0.3]],
        dtype=torch.float64,
        device="cuda:0",
    )
    moments = torch.tensor(
        [[0.2, 0.1, -0.4], [0.7, -0.3, 0.5]],
        dtype=torch.float64,
        device="cuda:0",
    )
    force_w, moment_w = _transform_wrench_cuda(rotation, translation, forces, moments)
    expected_force = torch.einsum("ij,nj->ni", rotation, forces)
    expected_moment = torch.einsum("ij,nj->ni", rotation, moments) + torch.linalg.cross(
        translation.expand_as(expected_force), expected_force, dim=1
    )
    torch.testing.assert_close(force_w, expected_force, rtol=0.0, atol=0.0)
    torch.testing.assert_close(moment_w, expected_moment, rtol=0.0, atol=0.0)


def test_one_step_real_lev_tev_packet_reaches_q16_generalized_load() -> None:
    branch = CudaJointLEVTEVSolver(
        _problem(1),
        JointConfig(
            enable_lev=True,
            joint_tev=True,
            lev_start_step=0,
            particle_capacity=128,
        ),
        device="cuda:0",
    )
    branch._prescribed_wake = False
    branch.run(
        prescribed_wake=False,
        calculate_streamlines=False,
        show_progress=False,
    )
    packet = Q16CudaAerodynamicLoadPacket.from_solver(branch)
    assert branch.lev_pf.n > 0
    assert branch._tev_solved is not None
    assert branch._prescribed_wake is False
    torch.testing.assert_close(
        packet.unresolved_impulse_force_w,
        torch.zeros(3, dtype=torch.float64, device="cuda:0"),
        rtol=0.0,
        atol=0.0,
    )

    # The fixed rectangular pilot wing is an affine Q16 midsurface.  Register
    # every actual vortex-leg / pressure point using its exact local chord/span
    # coordinate; coincident forces remain separate rows but share one element
    # owner and therefore one algebraic structural point.
    mesh = make_rectangular_q16_mesh(
        chordwise_element_count=2,
        spanwise_element_count=1,
        chord=1.0,
        span=4.0,
        thickness=0.02,
    )
    points = packet.point_positions_w.detach().cpu().numpy()
    chord_coordinate = points[:, 0]
    span_coordinate = points[:, 1]
    element_indices = np.ascontiguousarray(
        np.minimum((chord_coordinate / 0.5).astype(np.int64), 1)
    )
    local_chord_origin = 0.5 * element_indices
    coordinates = np.ascontiguousarray(
        np.column_stack(
            [
                4.0 * (chord_coordinate - local_chord_origin) - 1.0,
                0.5 * span_coordinate - 1.0,
                np.zeros(packet.point_count),
            ]
        ),
        dtype=np.float64,
    )
    transfer_map = Q16SurfaceTransferMap(
        mesh=mesh,
        element_indices=element_indices,
        parametric_coordinates=coordinates,
    )
    rows = mesh.reference_rows.copy()
    interpolation = np.zeros((packet.point_count, mesh.node_count), dtype=np.float64)
    for point, element in enumerate(element_indices):
        for local_node, global_node in enumerate(mesh.connectivity[element]):
            interpolation[point, global_node] += transfer_map.shape_values[
                point, local_node
            ]
    nodal_z, *_ = np.linalg.lstsq(interpolation, points[:, 2], rcond=None)
    rows[:, :3] = np.column_stack(
        [mesh.reference_rows[:, 0], mesh.reference_rows[:, 1], nodal_z]
    )
    rows[:, 3:] = np.array([0.0, 0.0, 0.01], dtype=np.float64)
    q = wp.array(
        np.ascontiguousarray(rows.ravel()[None, :]),
        dtype=config.DTYPE,
        device=config.DEVICE,
    )
    transfer = Q16CudaResolvedLoadTransfer(transfer_map, device=config.DEVICE)
    generalized = transfer.map(packet, q)
    wp.synchronize_device(config.DEVICE)
    balance = transfer_map.force_moment_balance(
        np.ascontiguousarray(rows.ravel()),
        np.ascontiguousarray(packet.point_forces_w.detach().cpu().numpy()),
    )
    generalized_rows = generalized.numpy()[0].reshape(mesh.node_count, 6)
    np.testing.assert_allclose(
        np.sum(generalized_rows[:, :3], axis=0),
        balance.aerodynamic_force,
        rtol=0.0,
        atol=5.0e-12,
    )
    np.testing.assert_allclose(
        balance.generalized_moment,
        balance.aerodynamic_moment,
        rtol=0.0,
        atol=5.0e-12,
    )


def test_hostile_parent_mutation_restores_pristine_owner_and_rejects_trial() -> None:
    owner = Q16CudaAeroSolverOwner(_solver())

    def hostile(branch: CudaJointLEVTEVSolver, q: wp.array, _dq: wp.array) -> wp.array:
        owner.current_solver._steps_done = 99
        owner.current_solver.cuda_counters["solve"] = 77
        return wp.clone(q)

    transaction = owner.begin(hostile)
    with pytest.raises(Q16AeroBranchTransactionViolation, match="live parent drift"):
        transaction.evaluate(*_trial())
    assert transaction.status == "failed"
    assert owner.current_solver._steps_done == 0
    assert owner.current_solver.cuda_counters["solve"] == 0

    retry = owner.begin(_advance_branch)
    proposal = retry.evaluate(*_trial())
    retry.commit(proposal)
    assert owner.current_solver._steps_done == 1


def test_branch_or_force_mutation_after_issue_is_rejected() -> None:
    owner = Q16CudaAeroSolverOwner(_solver())
    transaction = owner.begin(_advance_branch)
    proposal = transaction.evaluate(*_trial())
    proposal.proposed_solver._steps_done += 1
    with pytest.raises(Q16AeroBranchTransactionViolation, match="proposal drift"):
        transaction.commit(proposal)

    owner = Q16CudaAeroSolverOwner(_solver())
    transaction = owner.begin(_advance_branch)
    proposal = transaction.evaluate(*_trial())
    wp.launch(
        _scale_kernel,
        dim=proposal.generalized_force.shape,
        inputs=[proposal.generalized_force, config.DTYPE(3.0)],
        device=config.DEVICE,
    )
    with pytest.raises(Q16AeroBranchTransactionViolation, match="force drift"):
        transaction.commit(proposal)


@pytest.mark.parametrize(
    "enable_lev,joint_tev,prescribed_wake",
    [(False, True, False), (True, False, False), (True, True, True)],
)
def test_real_owner_rejects_every_reduced_aero_mode(
    enable_lev: bool, joint_tev: bool, prescribed_wake: bool
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
        Q16CudaAeroSolverOwner(solver)
