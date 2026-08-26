"""Real CUDA Q16 q/dq to Ptera panel-geometry integration gates."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import warp as wp

from bing_joint_ptera import JointConfig
from bing_joint_ptera_gpu import CudaJointLEVTEVSolver
from fluxvortex.q16_ancf_mesh import Q16Mesh, make_rectangular_q16_mesh
from fluxvortex.q16_work_conjugate_transfer import Q16SurfaceTransferMap
from fluxvortex.warp_fsi import config
from fluxvortex.warp_fsi.q16_aero_load_packet import Q16CudaAerodynamicLoadPacket
from fluxvortex.warp_fsi.q16_aero_load_packet import Q16CudaResolvedLoadTransfer
from q16_ptera_trial_kinematics import (
    Q16CudaPteraTwoStateKinematics,
    Q16PteraPanelVertexTopology,
)
from q16_real_aero_branch_transaction import Q16CudaAeroSolverOwner
from test_ptera_gpu_active_lev import _problem


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _mesh_and_map() -> tuple[Q16Mesh, Q16SurfaceTransferMap]:
    mesh = make_rectangular_q16_mesh(
        chordwise_element_count=1,
        spanwise_element_count=1,
        chord=1.0,
        span=4.0,
        thickness=0.02,
    )
    chord = (0.0, 0.5, 1.0)
    span = (0.0, 1.0, 3.0, 4.0)
    coordinates = np.ascontiguousarray(
        np.array(
            [
                [2.0 * chord_value - 1.0, 0.5 * span_value - 1.0, 0.0]
                for chord_value in chord
                for span_value in span
            ],
            dtype=np.float64,
        )
    )
    return mesh, Q16SurfaceTransferMap(
        mesh=mesh,
        element_indices=np.zeros(12, dtype=np.int64),
        parametric_coordinates=coordinates,
    )


def _trial(mesh: Q16Mesh, *, velocity_scale: float) -> tuple[wp.array, wp.array]:
    rows = mesh.reference_rows.copy()
    x = rows[:, 0].copy()
    y = rows[:, 1].copy()
    rows[:, :3] = np.column_stack(
        [
            x,
            y,
            0.04 * x * (y / 4.0) + 0.015 * (y / 4.0) ** 2,
        ]
    )
    normal = np.array([0.0, 0.0, 0.01], dtype=np.float64)
    rows[:, 3:] = normal
    velocity = np.zeros_like(rows)
    velocity[:, 0] = 0.01 * velocity_scale * y / 4.0
    velocity[:, 2] = 0.025 * velocity_scale * (0.3 + y / 4.0)
    velocity[:, 3:] = np.array([0.0, 0.002 * velocity_scale, 0.0], dtype=np.float64)
    return (
        wp.array(
            np.ascontiguousarray(rows.ravel()[None, :]),
            dtype=config.DTYPE,
            device=config.DEVICE,
        ),
        wp.array(
            np.ascontiguousarray(velocity.ravel()[None, :]),
            dtype=config.DTYPE,
            device=config.DEVICE,
        ),
    )


def _solver(*, steps: int = 2) -> CudaJointLEVTEVSolver:
    solver = CudaJointLEVTEVSolver(
        _problem(steps),
        JointConfig(
            enable_lev=True,
            joint_tev=True,
            lesp_crit=0.001,
            lev_start_step=0,
            particle_capacity=128,
        ),
        device="cuda:0",
    )
    solver._prescribed_wake = False
    return solver


def _world_vertices(solver: CudaJointLEVTEVSolver, step: int) -> np.ndarray:
    problem = solver.steady_problems[step]
    panels = problem.airplanes[0].wings[0].panels
    assert panels is not None
    chordwise, spanwise = panels.shape
    gp = np.empty((chordwise + 1, spanwise + 1, 3), dtype=np.float64)
    for chord in range(chordwise):
        for span in range(spanwise):
            panel = panels[chord, span]
            gp[chord, span] = panel.Flpp_GP1_CgP1
            gp[chord, span + 1] = panel.Frpp_GP1_CgP1
            gp[chord + 1, span] = panel.Blpp_GP1_CgP1
            gp[chord + 1, span + 1] = panel.Brpp_GP1_CgP1
    scientific = solver._v5m_gp_points_to_scientific_cuda(
        torch.as_tensor(
            np.ascontiguousarray(gp.reshape(-1, 3)),
            device=solver.cuda_device,
            dtype=torch.float64,
        )
    )
    return scientific.detach().cpu().numpy()


def test_two_state_q16_kinematics_reconstructs_positions_and_velocity() -> None:
    mesh, transfer_map = _mesh_and_map()
    adapter = Q16CudaPteraTwoStateKinematics(
        transfer_map,
        Q16PteraPanelVertexTopology(
            chordwise_panel_count=2,
            spanwise_panel_count=3,
        ),
        device=config.DEVICE,
    )
    solver = _solver()
    q, dq = _trial(mesh, velocity_scale=1.0)
    expected_current = wp.to_torch(adapter.surface_transfer.interpolate(q))[0]
    expected_velocity = wp.to_torch(adapter.surface_transfer.interpolate(dq))[0]
    evidence = adapter.apply(solver, q, dq)

    previous = torch.as_tensor(
        _world_vertices(solver, 0), dtype=torch.float64, device="cuda:0"
    )
    current = torch.as_tensor(
        _world_vertices(solver, 1), dtype=torch.float64, device="cuda:0"
    )
    inferred_velocity = (current - previous) / float(solver.delta_time)
    torch.testing.assert_close(current, expected_current, rtol=0.0, atol=2.0e-14)
    torch.testing.assert_close(
        inferred_velocity, expected_velocity, rtol=0.0, atol=3.0e-14
    )
    assert evidence.vertex_count == 12
    assert evidence.velocity_reconstruction_max_abs_error <= 3.0e-14


def test_real_lev_tev_free_wake_load_changes_when_q16_trial_velocity_changes() -> None:
    mesh, transfer_map = _mesh_and_map()
    adapter = Q16CudaPteraTwoStateKinematics(
        transfer_map,
        Q16PteraPanelVertexTopology(2, 3),
        device=config.DEVICE,
    )
    q_zero, dq_zero = _trial(mesh, velocity_scale=0.0)
    q_moving, dq_moving = _trial(mesh, velocity_scale=1.0)
    np.testing.assert_array_equal(q_zero.numpy(), q_moving.numpy())

    stationary = _solver()
    moving = _solver()
    adapter.apply(stationary, q_zero, dq_zero)
    adapter.apply(moving, q_moving, dq_moving)
    stationary.run(
        prescribed_wake=False, calculate_streamlines=False, show_progress=False
    )
    moving.run(prescribed_wake=False, calculate_streamlines=False, show_progress=False)
    stationary_packet = Q16CudaAerodynamicLoadPacket.from_solver(stationary)
    moving_packet = Q16CudaAerodynamicLoadPacket.from_solver(moving)

    assert stationary_packet.packet_sha256 != moving_packet.packet_sha256
    assert not torch.equal(
        stationary_packet.source_total_force_w,
        moving_packet.source_total_force_w,
    )
    assert stationary.lev_pf.n > 0 and moving.lev_pf.n > 0
    assert stationary._tev_solved is not None and moving._tev_solved is not None
    assert stationary._prescribed_wake is moving._prescribed_wake is False


def test_kinematics_rejects_wrong_topology_lifecycle_and_host_state() -> None:
    mesh, transfer_map = _mesh_and_map()
    adapter = Q16CudaPteraTwoStateKinematics(
        transfer_map,
        Q16PteraPanelVertexTopology(2, 3),
        device=config.DEVICE,
    )
    q, dq = _trial(mesh, velocity_scale=0.0)
    with pytest.raises(ValueError, match="exactly two aerodynamic states"):
        adapter.apply(_solver(steps=1), q, dq)

    ran = _solver()
    adapter.apply(ran, q, dq)
    ran.run(prescribed_wake=False, calculate_streamlines=False, show_progress=False)
    with pytest.raises(RuntimeError, match="pristine solver branch"):
        adapter.apply(ran, q, dq)

    host_q = wp.array(q.numpy(), dtype=config.DTYPE, device="cpu")
    with pytest.raises(ValueError, match="CUDA"):
        adapter.apply(_solver(), host_q, dq)

    wrong = Q16CudaPteraTwoStateKinematics(
        transfer_map,
        Q16PteraPanelVertexTopology(1, 5),
        device=config.DEVICE,
    )
    with pytest.raises(ValueError, match="panel topology"):
        wrong.apply(_solver(), q, dq)


def test_real_trial_transaction_stops_at_unresolved_lev_work_without_commit() -> None:
    mesh, vertex_map = _mesh_and_map()
    adapter = Q16CudaPteraTwoStateKinematics(
        vertex_map,
        Q16PteraPanelVertexTopology(2, 3),
        device=config.DEVICE,
    )
    parent = _solver()
    owner = Q16CudaAeroSolverOwner(parent)
    observed: dict[str, float | int] = {}

    def evaluate_real_trial(
        branch: CudaJointLEVTEVSolver,
        q_trial: wp.array,
        dq_trial: wp.array,
    ) -> wp.array:
        adapter.apply(branch, q_trial, dq_trial)
        branch.run(
            prescribed_wake=False,
            calculate_streamlines=False,
            show_progress=False,
        )
        packet = Q16CudaAerodynamicLoadPacket.from_solver(branch)
        observed["lev_particles"] = branch.lev_pf.n
        observed["impulse_norm"] = float(
            torch.linalg.vector_norm(packet.unresolved_impulse_force_w).item()
        )
        # Match the real packet row count so the production transfer reaches
        # its first scientific gate: active-LEV impulse has no declared Q16
        # flexible-work application point.  No arbitrary smearing is allowed.
        load_map = Q16SurfaceTransferMap(
            mesh=mesh,
            element_indices=np.zeros(packet.point_count, dtype=np.int64),
            parametric_coordinates=np.zeros((packet.point_count, 3), dtype=np.float64),
        )
        return Q16CudaResolvedLoadTransfer(load_map, device=config.DEVICE).map(
            packet, q_trial
        )

    q, dq = _trial(mesh, velocity_scale=1.0)
    transaction = owner.begin(evaluate_real_trial)
    with pytest.raises(
        RuntimeError,
        match="unresolved LEV impulse has no work-conjugate Q16 application point",
    ):
        transaction.evaluate(q, dq)

    assert observed["lev_particles"] > 0
    assert observed["impulse_norm"] > 0.0
    assert transaction.status == "failed"
    assert owner.current_solver is parent
    assert owner.generation == 0
    assert parent.ran is False
    assert parent.lev_pf.n == 0
    assert parent._steps_done == 0


@pytest.mark.parametrize(
    ("enable_lev", "joint_tev", "prescribed_wake"),
    [(False, True, False), (True, False, False), (True, True, True)],
)
def test_kinematics_rejects_every_reduced_aerodynamic_mode(
    enable_lev: bool,
    joint_tev: bool,
    prescribed_wake: bool,
) -> None:
    mesh, transfer_map = _mesh_and_map()
    adapter = Q16CudaPteraTwoStateKinematics(
        transfer_map,
        Q16PteraPanelVertexTopology(2, 3),
        device=config.DEVICE,
    )
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
    q, dq = _trial(mesh, velocity_scale=0.0)
    with pytest.raises(RuntimeError):
        adapter.apply(solver, q, dq)
