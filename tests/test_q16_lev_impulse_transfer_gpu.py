"""Work-conjugate CUDA transfer of source-owned LEV strip impulse forces."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import warp as wp

from fluxvortex.warp_fsi import config
from fluxvortex.warp_fsi.q16_aero_load_packet import Q16CudaAerodynamicLoadPacket
from fluxvortex.warp_fsi.q16_lev_impulse_transfer import (
    Q16CudaLEVImpulseStripLoad,
    Q16CudaLEVImpulseTransfer,
)
from fluxvortex.warp_fsi.kernels_q16_transfer import Q16CudaSurfaceTransfer
from test_q16_incremental_endpoint_motion import _committed_parent, _velocity
from test_q16_incremental_trial_geometry import _state
from test_q16_ptera_trial_kinematics import _mesh_and_map


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not wp.is_cuda_available(), reason="CUDA required"
)


def _synthetic() -> (
    tuple[
        object,
        object,
        wp.array,
        Q16CudaLEVImpulseStripLoad,
        Q16CudaLEVImpulseTransfer,
    ]
):
    mesh, transfer_map = _mesh_and_map()
    surface_transfer = Q16CudaSurfaceTransfer(transfer_map, device=config.DEVICE)
    q = _state(mesh, 0.8)
    points = wp.to_torch(surface_transfer.interpolate(q))[0]
    leading_indices = np.arange(4, dtype=np.int64)
    endpoints = torch.stack(
        (points[leading_indices[:-1]], points[leading_indices[1:]]), dim=1
    )
    forces = torch.tensor(
        [
            [4.0, -0.5, 1.2],
            [-1.5, 2.1, 0.8],
            [0.7, -0.4, 3.2],
        ],
        dtype=torch.float64,
        device="cuda:0",
    )
    source = torch.tensor([0, 0, 2, 1, 2, 2, 0, 1], dtype=torch.int64, device="cuda:0")
    load = Q16CudaLEVImpulseStripLoad.from_tensors(
        strip_forces_w=forces,
        leading_edge_endpoints_w=endpoints,
        particle_source_strips=source,
    )
    transfer = Q16CudaLEVImpulseTransfer(
        transfer_map,
        leading_edge_point_indices=leading_indices,
        device=config.DEVICE,
    )
    return mesh, surface_transfer, q, load, transfer


def test_synthetic_source_line_transfer_preserves_force_moment_and_work() -> None:
    mesh, surface_transfer, q, load, transfer = _synthetic()
    generalized = transfer.map(load, q)
    generalized_rows = generalized.numpy()[0].reshape(mesh.node_count, 6)
    q_rows = q.numpy()[0].reshape(mesh.node_count, 6)
    generalized_force = np.sum(generalized_rows[:, :3], axis=0)
    generalized_moment = np.sum(
        np.cross(q_rows[:, :3], generalized_rows[:, :3])
        + np.cross(q_rows[:, 3:], generalized_rows[:, 3:]),
        axis=0,
    )
    np.testing.assert_allclose(
        generalized_force,
        load.source_total_force_w.cpu().numpy(),
        rtol=0.0,
        atol=3.0e-15,
    )
    np.testing.assert_allclose(
        generalized_moment,
        load.source_midpoint_moment_w.cpu().numpy(),
        rtol=0.0,
        atol=8.0e-14,
    )

    direction = _velocity(mesh, 0.63)
    leading_velocity = wp.to_torch(surface_transfer.interpolate(direction))[0, :4]
    midpoint_velocity = 0.5 * (leading_velocity[:-1] + leading_velocity[1:])
    aerodynamic_work = float(torch.sum(midpoint_velocity * load.strip_forces_w).item())
    structural_work = float(np.sum(direction.numpy() * generalized.numpy()))
    assert structural_work == pytest.approx(aerodynamic_work, rel=0.0, abs=2.0e-14)


def test_real_incremental_lev_impulse_reaches_q16_generalized_force() -> None:
    mesh, binder, parent = _committed_parent()
    _, transfer_map = _mesh_and_map()
    branch = parent.fork()
    q_mid = _state(mesh, 1.1)
    binder.bind_next_state(branch, q_mid, _velocity(mesh, 0.7))
    branch.advance_one_step()
    q = _state(mesh, 1.6)
    binder.bind_next_state(branch, q, _velocity(mesh, 1.2))
    branch.advance_one_step()
    load = Q16CudaLEVImpulseStripLoad.from_solver(branch.solver)
    packet = Q16CudaAerodynamicLoadPacket.from_solver(branch.solver)
    transfer = Q16CudaLEVImpulseTransfer(
        transfer_map,
        leading_edge_point_indices=np.arange(4, dtype=np.int64),
        device=config.DEVICE,
    )
    generalized = transfer.map(load, q)

    assert load.strip_count == 3
    assert load.particle_count == branch.solver.lev_pf.n > 0
    torch.testing.assert_close(
        load.source_total_force_w,
        packet.unresolved_impulse_force_w,
        rtol=0.0,
        atol=8.0e-13,
    )
    assert float(torch.linalg.vector_norm(wp.to_torch(generalized)).item()) > 0.0
    assert branch.solver._prescribed_wake is False
    assert branch.solver._tev_solved is not None


def test_load_content_source_and_geometry_drift_fail_closed() -> None:
    mesh, _, q, load, transfer = _synthetic()
    clean_generalized = transfer.map(load, q)
    load.strip_forces_w[0, 0] += 1.0
    with pytest.raises(RuntimeError, match="content drift"):
        transfer.map(load, q)

    _, _, clean_q, clean_load, clean_transfer = _synthetic()
    clean_load.particle_source_strips[0] = 2
    with pytest.raises(RuntimeError, match="content drift"):
        clean_transfer.map(clean_load, clean_q)

    _, _, original_q, geometric_load, geometric_transfer = _synthetic()
    shifted = original_q.numpy()
    shifted[0, 2] += 1.0e-5
    shifted_q = wp.array(
        np.ascontiguousarray(shifted), dtype=config.DTYPE, device=config.DEVICE
    )
    with pytest.raises(RuntimeError, match="geometry mismatch"):
        geometric_transfer.map(geometric_load, shifted_q)

    assert clean_generalized.device.is_cuda
    assert mesh.node_count > 0


def test_load_rejects_host_float32_and_invalid_source_identity() -> None:
    cuda = torch.device("cuda:0")
    forces = torch.zeros((2, 3), dtype=torch.float64, device=cuda)
    endpoints = torch.zeros((2, 2, 3), dtype=torch.float64, device=cuda)
    source = torch.tensor([0, 1], dtype=torch.int64, device=cuda)
    with pytest.raises(ValueError, match="CUDA"):
        Q16CudaLEVImpulseStripLoad.from_tensors(
            strip_forces_w=forces.cpu(),
            leading_edge_endpoints_w=endpoints,
            particle_source_strips=source,
        )
    with pytest.raises(TypeError, match="float64"):
        Q16CudaLEVImpulseStripLoad.from_tensors(
            strip_forces_w=forces.float(),
            leading_edge_endpoints_w=endpoints,
            particle_source_strips=source,
        )
    with pytest.raises(TypeError, match="int64"):
        Q16CudaLEVImpulseStripLoad.from_tensors(
            strip_forces_w=forces,
            leading_edge_endpoints_w=endpoints,
            particle_source_strips=source.int(),
        )
    with pytest.raises(ValueError, match="source strip"):
        Q16CudaLEVImpulseStripLoad.from_tensors(
            strip_forces_w=forces,
            leading_edge_endpoints_w=endpoints,
            particle_source_strips=torch.tensor([0, 2], dtype=torch.int64, device=cuda),
        )


def test_zero_particle_base_state_retains_explicit_strip_topology() -> None:
    cuda = torch.device("cuda:0")
    forces = torch.zeros((3, 3), dtype=torch.float64, device=cuda)
    endpoints = torch.tensor(
        [
            [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [[0.0, 1.0, 0.0], [0.0, 2.0, 0.0]],
            [[0.0, 2.0, 0.0], [0.0, 3.0, 0.0]],
        ],
        dtype=torch.float64,
        device=cuda,
    )
    load = Q16CudaLEVImpulseStripLoad.from_tensors(
        strip_forces_w=forces,
        leading_edge_endpoints_w=endpoints,
        particle_source_strips=torch.empty(0, dtype=torch.int64, device=cuda),
    )

    assert load.strip_count == 3
    assert load.particle_count == 0
    torch.testing.assert_close(
        load.source_total_force_w,
        torch.zeros(3, dtype=torch.float64, device=cuda),
    )
    assert load.validate() is load
