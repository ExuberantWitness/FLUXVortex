"""Real Q16 trial geometry on trusted incremental aerodynamic branches."""

from __future__ import annotations

import hashlib
import pickle

import numpy as np
import pytest
import torch
import warp as wp

from bing_joint_ptera import JointConfig
from bing_joint_ptera_gpu import CudaJointLEVTEVSolver
from fluxvortex.q16_ancf_mesh import Q16Mesh
from fluxvortex.warp_fsi import config
from fluxvortex.warp_fsi.q16_aero_load_packet import Q16CudaAerodynamicLoadPacket
from q16_incremental_ptera_owner import (
    Q16CudaIncrementalAeroSession,
    Q16IncrementalAeroLifecycleError,
)
from q16_ptera_trial_kinematics import (
    Q16CudaPteraIncrementalGeometry,
    Q16PteraPanelVertexTopology,
)
from test_ptera_gpu_active_lev import _problem
from test_q16_ptera_trial_kinematics import _mesh_and_map


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _solver(
    *,
    steps: int = 3,
    problem: object | None = None,
    particle_capacity: int = 256,
) -> CudaJointLEVTEVSolver:
    solver = CudaJointLEVTEVSolver(
        _problem(steps) if problem is None else problem,
        JointConfig(
            enable_lev=True,
            joint_tev=True,
            lesp_crit=0.001,
            lev_start_step=0,
            particle_capacity=particle_capacity,
        ),
        device="cuda:0",
    )
    solver._prescribed_wake = False
    return solver


def _state(mesh: Q16Mesh, deformation_scale: float) -> wp.array:
    rows = mesh.reference_rows.copy()
    x = rows[:, 0].copy()
    y = rows[:, 1].copy()
    eta = y / 4.0
    rows[:, :3] = np.column_stack(
        [
            x + 0.003 * deformation_scale * eta,
            y,
            deformation_scale * (0.03 * x * eta + 0.012 * eta * eta),
        ]
    )
    rows[:, 3:] = np.array([0.0, 0.001 * deformation_scale, 0.01], dtype=np.float64)
    return wp.array(
        np.ascontiguousarray(rows.ravel()[None, :]),
        dtype=config.DTYPE,
        device=config.DEVICE,
    )


def _pickle_sha(solver: CudaJointLEVTEVSolver) -> str:
    torch.cuda.synchronize(solver.cuda_device)
    return hashlib.sha256(pickle.dumps(solver, protocol=5)).hexdigest()


def _binder() -> tuple[Q16Mesh, Q16CudaPteraIncrementalGeometry]:
    mesh, transfer_map = _mesh_and_map()
    return mesh, Q16CudaPteraIncrementalGeometry(
        transfer_map,
        Q16PteraPanelVertexTopology(2, 3),
        device=config.DEVICE,
    )


def test_same_parent_and_q_issue_identical_wake_load_and_receipt() -> None:
    mesh, binder = _binder()
    parent = _solver()
    parent_session = Q16CudaIncrementalAeroSession.begin(parent)
    binder.bind_next(parent_session, _state(mesh, 0.0))
    parent_session.advance_one_step()
    parent_sha = _pickle_sha(parent)

    branch_a = parent_session.fork()
    branch_b = parent_session.fork()
    evidence_a = binder.bind_next(branch_a, _state(mesh, 1.0))
    evidence_b = binder.bind_next(branch_b, _state(mesh, 1.0))
    receipt_a = branch_a.advance_one_step()
    receipt_b = branch_b.advance_one_step()
    packet_a = Q16CudaAerodynamicLoadPacket.from_solver(branch_a.solver)
    packet_b = Q16CudaAerodynamicLoadPacket.from_solver(branch_b.solver)

    assert evidence_a == evidence_b
    assert receipt_a == receipt_b
    assert packet_a.packet_sha256 == packet_b.packet_sha256
    assert np.array_equal(
        branch_a.solver.steady_problems[1].airplanes[0].wings[0].gridWrvp_GP1_CgP1,
        branch_b.solver.steady_problems[1].airplanes[0].wings[0].gridWrvp_GP1_CgP1,
    )
    assert _pickle_sha(parent) == parent_sha
    assert parent._steps_done == 1
    assert parent_session.next_step == 1


def test_different_q_trials_change_real_wake_and_load_but_not_parent() -> None:
    mesh, binder = _binder()
    parent = _solver()
    parent_session = Q16CudaIncrementalAeroSession.begin(parent)
    binder.bind_next(parent_session, _state(mesh, 0.0))
    parent_session.advance_one_step()
    parent_sha = _pickle_sha(parent)
    parent_counters = parent.cuda_counters.copy()
    parent_particles = parent.lev_pf.snapshot_numpy()

    branch_a = parent_session.fork()
    branch_b = parent_session.fork()
    evidence_a = binder.bind_next(branch_a, _state(mesh, 0.7))
    evidence_b = binder.bind_next(branch_b, _state(mesh, 1.4))
    receipt_a = branch_a.advance_one_step()
    receipt_b = branch_b.advance_one_step()
    packet_a = Q16CudaAerodynamicLoadPacket.from_solver(branch_a.solver)
    packet_b = Q16CudaAerodynamicLoadPacket.from_solver(branch_b.solver)

    assert evidence_a.current_vertices_sha256 != evidence_b.current_vertices_sha256
    assert receipt_a.solver_state_sha256 != receipt_b.solver_state_sha256
    assert packet_a.packet_sha256 != packet_b.packet_sha256
    assert not torch.equal(packet_a.source_total_force_w, packet_b.source_total_force_w)
    wake_a = branch_a.solver.steady_problems[1].airplanes[0].wings[0].gridWrvp_GP1_CgP1
    wake_b = branch_b.solver.steady_problems[1].airplanes[0].wings[0].gridWrvp_GP1_CgP1
    assert not np.array_equal(wake_a, wake_b)
    assert branch_a.solver.lev_pf.n > 0 and branch_b.solver.lev_pf.n > 0
    assert branch_a.solver._tev_solved is not None
    assert branch_b.solver._tev_solved is not None
    assert _pickle_sha(parent) == parent_sha
    assert parent.cuda_counters == parent_counters
    for name, expected in parent_particles.items():
        np.testing.assert_array_equal(parent.lev_pf.snapshot_numpy()[name], expected)


def test_next_geometry_is_detached_and_cannot_be_rebound_in_one_branch() -> None:
    mesh, binder = _binder()
    session = Q16CudaIncrementalAeroSession.begin(_solver())
    q = _state(mesh, 0.8)
    evidence = binder.bind_next(session, q)
    q.zero_()
    assert binder.current_vertices_sha256(session.solver, 0) == (
        evidence.current_vertices_sha256
    )
    with pytest.raises(Q16IncrementalAeroLifecycleError, match="already bound"):
        binder.bind_next(session, _state(mesh, 1.0))


def test_incremental_geometry_rejects_host_state_wrong_topology_and_lifecycle() -> None:
    mesh, binder = _binder()
    q = _state(mesh, 0.0)
    host_q = wp.array(q.numpy(), dtype=config.DTYPE, device="cpu")
    with pytest.raises(ValueError, match="CUDA"):
        binder.bind_next(Q16CudaIncrementalAeroSession.begin(_solver()), host_q)

    _, transfer_map = _mesh_and_map()
    wrong = Q16CudaPteraIncrementalGeometry(
        transfer_map,
        Q16PteraPanelVertexTopology(1, 5),
        device=config.DEVICE,
    )
    with pytest.raises(ValueError, match="panel topology"):
        wrong.bind_next(Q16CudaIncrementalAeroSession.begin(_solver()), q)

    completed = Q16CudaIncrementalAeroSession.begin(_solver(steps=2))
    binder.bind_next(completed, q)
    completed.advance_one_step()
    binder.bind_next(completed, _state(mesh, 0.5))
    completed.advance_one_step()
    with pytest.raises(Q16IncrementalAeroLifecycleError, match="no remaining"):
        binder.bind_next(completed, _state(mesh, 1.0))
