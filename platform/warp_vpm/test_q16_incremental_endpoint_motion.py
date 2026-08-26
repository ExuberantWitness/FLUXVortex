"""Tests-first contract for Q16 endpoint velocity on real aero branches."""

from __future__ import annotations

import hashlib
import pickle

import numpy as np
import pytest
import torch
import warp as wp

from fluxvortex.q16_ancf_mesh import Q16Mesh
from fluxvortex.warp_fsi import config
from fluxvortex.warp_fsi.q16_aero_load_packet import Q16CudaAerodynamicLoadPacket
from q16_incremental_ptera_owner import (
    Q16CudaIncrementalAeroSession,
    Q16IncrementalAeroLifecycleError,
)
from q16_ptera_trial_kinematics import Q16PteraIncrementalMotionEvidence
from test_q16_incremental_trial_geometry import _binder, _solver, _state


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _velocity(mesh: Q16Mesh, scale: float) -> wp.array:
    rows = np.zeros_like(mesh.reference_rows)
    x = mesh.reference_rows[:, 0]
    y = mesh.reference_rows[:, 1]
    eta = y / 4.0
    rows[:, :3] = np.column_stack(
        [
            0.004 * scale * eta,
            np.zeros_like(eta),
            scale * (0.08 * x * eta + 0.025 * eta * eta),
        ]
    )
    rows[:, 3:] = np.array([0.0, 0.002 * scale, 0.0], dtype=np.float64)
    return wp.array(
        np.ascontiguousarray(rows.ravel()[None, :]),
        dtype=config.DTYPE,
        device=config.DEVICE,
    )


def _pickle_sha(value: object) -> str:
    torch.cuda.synchronize()
    return hashlib.sha256(pickle.dumps(value, protocol=5)).hexdigest()


def _committed_parent() -> tuple[Q16Mesh, object, Q16CudaIncrementalAeroSession]:
    mesh, binder = _binder()
    session = Q16CudaIncrementalAeroSession.begin(_solver())
    first = binder.bind_next_state(session, _state(mesh, 0.0), _velocity(mesh, 0.0))
    assert type(first) is Q16PteraIncrementalMotionEvidence
    session.advance_one_step()
    return mesh, binder, session


def test_same_q_and_dq_are_exact_but_different_dq_changes_real_flow() -> None:
    mesh, binder, parent = _committed_parent()
    parent_sha = _pickle_sha(parent.solver)
    q = _state(mesh, 1.1)

    same_a = parent.fork()
    same_b = parent.fork()
    evidence_same_a = binder.bind_next_state(same_a, q, _velocity(mesh, 0.7))
    evidence_same_b = binder.bind_next_state(same_b, q, _velocity(mesh, 0.7))
    receipt_same_a = same_a.advance_one_step()
    receipt_same_b = same_b.advance_one_step()
    packet_same_a = Q16CudaAerodynamicLoadPacket.from_solver(same_a.solver)
    packet_same_b = Q16CudaAerodynamicLoadPacket.from_solver(same_b.solver)

    assert evidence_same_a == evidence_same_b
    assert receipt_same_a == receipt_same_b
    assert packet_same_a.packet_sha256 == packet_same_b.packet_sha256

    slow = parent.fork()
    fast = parent.fork()
    slow_evidence = binder.bind_next_state(slow, q, _velocity(mesh, 0.25))
    fast_evidence = binder.bind_next_state(fast, q, _velocity(mesh, 1.75))
    slow_receipt = slow.advance_one_step()
    fast_receipt = fast.advance_one_step()
    slow_packet = Q16CudaAerodynamicLoadPacket.from_solver(slow.solver)
    fast_packet = Q16CudaAerodynamicLoadPacket.from_solver(fast.solver)

    assert (
        slow_evidence.current_vertices_sha256 == fast_evidence.current_vertices_sha256
    )
    assert slow_evidence.vertex_velocity_sha256 != fast_evidence.vertex_velocity_sha256
    motion_limit = 128.0 * float(np.finfo(np.float64).eps)
    assert slow_evidence.motion_shadow_max_abs_error <= motion_limit
    assert fast_evidence.motion_shadow_max_abs_error <= motion_limit
    assert slow_receipt.solver_state_sha256 != fast_receipt.solver_state_sha256
    assert slow_packet.packet_sha256 != fast_packet.packet_sha256
    assert not torch.equal(
        slow_packet.source_total_force_w, fast_packet.source_total_force_w
    )
    slow_wake = slow.solver.steady_problems[1].airplanes[0].wings[0].gridWrvp_GP1_CgP1
    fast_wake = fast.solver.steady_problems[1].airplanes[0].wings[0].gridWrvp_GP1_CgP1
    assert not np.array_equal(slow_wake, fast_wake)
    slow_wake_scientific = slow.solver._v5m_gp_points_to_scientific_cuda(
        torch.as_tensor(
            np.ascontiguousarray(slow_wake.reshape(-1, 3)),
            device=slow.solver.cuda_device,
            dtype=torch.float64,
        )
    ).reshape(slow_wake.shape)
    v_inf_scientific = slow.solver._v5m_gp_vectors_to_scientific_cuda(
        torch.as_tensor(
            np.array(
                slow.solver.current_operating_point.vInf_GP1__E,
                dtype=np.float64,
                copy=True,
            ),
            device=slow.solver.cuda_device,
            dtype=torch.float64,
        ).unsqueeze(0)
    )[0]
    downstream = torch.sum(
        (slow_wake_scientific[1:] - slow_wake_scientific[:-1])
        * v_inf_scientific,
        dim=2,
    )
    assert torch.all(downstream > 0.0).item()
    assert slow.solver._v5m_downstream_wake_contract_step == 1
    slow_wake_velocity = slow.solver._q16_cuda_wake_vertex_velocity_grids_gp[1]
    assert slow_wake_velocity is not None
    assert slow_wake_velocity.device.type == "cuda"
    assert slow_wake_velocity.dtype is torch.float64
    assert tuple(slow_wake_velocity.shape) == tuple(slow_wake.shape)
    slow_mf2_pressure = slow.solver._author_wake_motion_pressure_cuda()
    fast_mf2_pressure = fast.solver._author_wake_motion_pressure_cuda()
    assert slow_mf2_pressure.device.type == "cuda"
    assert slow_mf2_pressure.dtype is torch.float64
    assert torch.isfinite(slow_mf2_pressure).all().item()
    assert not torch.equal(slow_mf2_pressure, fast_mf2_pressure)
    assert slow.solver.lev_pf.n > 0 and fast.solver.lev_pf.n > 0
    assert slow.solver._tev_solved is not None and fast.solver._tev_solved is not None
    expected_velocity_w = wp.to_torch(
        binder.surface_transfer.interpolate(_velocity(mesh, 0.25))
    )[0]
    expected_velocity_gp = slow.solver._v5m_scientific_vectors_to_gp_cuda(
        expected_velocity_w
    ).reshape(3, 4, 3)
    expected_relative_le = (
        torch.as_tensor(
            np.array(
                slow.solver.steady_problems[1].operating_point.vInf_GP1__E,
                dtype=np.float64,
                copy=True,
            ),
            dtype=torch.float64,
            device=expected_velocity_w.device,
        )
        - expected_velocity_gp[0]
    )
    torch.testing.assert_close(
        slow.solver.ledger[-1]["v_rel_st"],
        expected_relative_le,
        rtol=128.0 * torch.finfo(torch.float64).eps,
        atol=128.0 * torch.finfo(torch.float64).eps,
    )
    assert _pickle_sha(parent.solver) == parent_sha
    assert parent.next_step == 1


def test_bound_motion_is_detached_and_rebinding_fails_closed() -> None:
    mesh, binder, parent = _committed_parent()
    branch = parent.fork()
    q = _state(mesh, 0.8)
    dq = _velocity(mesh, 0.9)
    evidence = binder.bind_next_state(branch, q, dq)
    dq.zero_()
    assert branch.bound_vertex_velocity_sha256 == evidence.vertex_velocity_sha256
    with pytest.raises(Q16IncrementalAeroLifecycleError, match="already bound"):
        binder.bind_next_state(branch, q, _velocity(mesh, 1.0))

    hostile = parent.fork()
    binder.bind_next_state(hostile, q, _velocity(mesh, 0.9))
    setattr(
        hostile.solver,
        "_q16_incremental_bound_vertex_velocity_gp",
        np.zeros((3, 4, 3), dtype=np.float64),
    )
    with pytest.raises(Q16IncrementalAeroLifecycleError, match="state drift"):
        hostile.advance_one_step()

    clean = parent.fork()
    binder.bind_next_state(clean, q, _velocity(mesh, 0.9))
    assert clean.advance_one_step().step_index == 1


def test_endpoint_motion_rejects_missing_history_and_host_velocity() -> None:
    mesh, binder = _binder()
    initial = Q16CudaIncrementalAeroSession.begin(_solver())
    with pytest.raises(ValueError, match="initial.*zero"):
        binder.bind_next_state(initial, _state(mesh, 0.0), _velocity(mesh, 1.0))

    parent = Q16CudaIncrementalAeroSession.begin(_solver())
    binder.bind_next_state(parent, _state(mesh, 0.0), _velocity(mesh, 0.0))
    parent.advance_one_step()
    host_dq = wp.array(_velocity(mesh, 1.0).numpy(), dtype=config.DTYPE, device="cpu")
    with pytest.raises(ValueError, match="CUDA"):
        binder.bind_next_state(parent.fork(), _state(mesh, 0.5), host_dq)


def test_q_and_dq_shapes_devices_and_inputs_are_immutable() -> None:
    mesh, binder, parent = _committed_parent()
    branch = parent.fork()
    q = _state(mesh, 0.4)
    dq = _velocity(mesh, 0.6)
    q_before = q.numpy().copy()
    dq_before = dq.numpy().copy()
    binder.bind_next_state(branch, q, dq)
    np.testing.assert_array_equal(q.numpy(), q_before)
    np.testing.assert_array_equal(dq.numpy(), dq_before)

    wrong = wp.array(
        np.zeros((1, q.shape[1] - 1), dtype=np.float64),
        dtype=config.DTYPE,
        device=config.DEVICE,
    )
    with pytest.raises(ValueError, match="shapes differ"):
        binder.bind_next_state(parent.fork(), q, wrong)
