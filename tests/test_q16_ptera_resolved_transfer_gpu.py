"""Conservative CUDA transfer for the five real Ptera load-point blocks."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import warp as wp

from fluxvortex.q16_work_conjugate_transfer import Q16SurfaceTransferMap
from fluxvortex.warp_fsi import config
from fluxvortex.warp_fsi.kernels_q16_transfer import Q16CudaSurfaceTransfer
from fluxvortex.warp_fsi.q16_aero_load_packet import Q16CudaAerodynamicLoadPacket
from fluxvortex.warp_fsi.q16_lev_impulse_transfer import (
    Q16CudaLEVImpulseStripLoad,
    Q16CudaLEVImpulseTransfer,
)
from fluxvortex.warp_fsi.q16_ptera_resolved_transfer import (
    Q16CudaCompleteAeroLoadTransfer,
    Q16CudaPteraResolvedLoadTransfer,
    Q16PteraResolvedTransferResult,
)
from test_q16_incremental_endpoint_motion import _committed_parent, _velocity
from test_q16_incremental_trial_geometry import _state
from test_q16_ptera_trial_kinematics import _mesh_and_map, _trial


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not wp.is_cuda_available(), reason="CUDA required"
)


def _support_map(vertex_map: Q16SurfaceTransferMap) -> Q16SurfaceTransferMap:
    indices = np.repeat(vertex_map.element_indices, 2).astype(np.int64, copy=False)
    coordinates = np.repeat(vertex_map.parametric_coordinates, 2, axis=0).copy()
    coordinates[0::2, 2] = -1.0
    coordinates[1::2, 2] = 1.0
    return Q16SurfaceTransferMap(
        mesh=vertex_map.mesh,
        element_indices=np.ascontiguousarray(indices),
        parametric_coordinates=np.ascontiguousarray(coordinates),
    )


def _panel_support_indices(chord_count: int, span_count: int) -> np.ndarray:
    rows: list[list[int]] = []
    for chord in range(chord_count):
        for span in range(span_count):
            vertices = (
                chord * (span_count + 1) + span,
                chord * (span_count + 1) + span + 1,
                (chord + 1) * (span_count + 1) + span,
                (chord + 1) * (span_count + 1) + span + 1,
            )
            rows.append([2 * vertex + face for vertex in vertices for face in (0, 1)])
    return np.ascontiguousarray(rows, dtype=np.int64)


def _synthetic_packet() -> (
    tuple[
        object,
        Q16SurfaceTransferMap,
        wp.array,
        Q16CudaAerodynamicLoadPacket,
    ]
):
    mesh, vertex_map = _mesh_and_map()
    q, _ = _trial(mesh, velocity_scale=0.7)
    support_transfer = Q16CudaSurfaceTransfer(
        _support_map(vertex_map), device=config.DEVICE
    )
    supports = wp.to_torch(support_transfer.interpolate(q))[0]
    panel_support = torch.as_tensor(
        _panel_support_indices(2, 3), dtype=torch.int64, device="cuda:0"
    )
    owners = torch.arange(6, dtype=torch.int64, device="cuda:0").repeat(5)
    local = supports[panel_support[owners]]
    generator = torch.Generator(device="cuda:0")
    generator.manual_seed(20260822)
    weights = torch.rand(
        (30, 8), dtype=torch.float64, device="cuda:0", generator=generator
    )
    weights = weights / torch.sum(weights, dim=1, keepdim=True)
    points = torch.sum(weights.unsqueeze(2) * local, dim=1)
    forces = torch.randn(
        (30, 3), dtype=torch.float64, device="cuda:0", generator=generator
    )
    resolved_force = torch.sum(forces, dim=0)
    resolved_moment = torch.sum(torch.linalg.cross(points, forces, dim=1), dim=0)
    packet = Q16CudaAerodynamicLoadPacket.from_tensors(
        point_positions_w=points,
        point_forces_w=forces,
        unresolved_impulse_force_w=torch.zeros(3, dtype=torch.float64, device="cuda:0"),
        source_total_force_w=resolved_force,
        source_total_moment_w=resolved_moment,
    )
    return mesh, vertex_map, q, packet


def test_local_same_panel_transfer_closes_force_moment_and_virtual_work() -> None:
    mesh, vertex_map, q, packet = _synthetic_packet()
    transfer = Q16CudaPteraResolvedLoadTransfer(
        vertex_map,
        chordwise_panel_count=2,
        spanwise_panel_count=3,
        device=config.DEVICE,
    )
    result = transfer.map(packet, q)
    assert type(result) is Q16PteraResolvedTransferResult
    assert result.point_count == packet.point_count == 30
    assert result.point_reconstruction_max_abs_error <= 2.0e-12
    assert result.resolved_force_max_abs_error <= 2.0e-12
    assert result.resolved_moment_max_abs_error <= 2.0e-12

    direction_np = np.ascontiguousarray(
        np.random.default_rng(20260822).normal(
            scale=0.02, size=(1, vertex_map.structural_dof_count)
        ),
        dtype=np.float64,
    )
    direction = wp.array(direction_np, dtype=config.DTYPE, device=config.DEVICE)
    point_direction = transfer.interpolate_frozen_point_direction(result, direction)
    structural_work = float(np.sum(result.generalized_force.numpy() * direction_np))
    aerodynamic_work = float(torch.sum(point_direction * packet.point_forces_w).item())
    assert structural_work == pytest.approx(aerodynamic_work, rel=0.0, abs=2.0e-11)

    rows = q.numpy()[0].reshape(mesh.node_count, 6)
    generalized = result.generalized_force.numpy()[0].reshape(mesh.node_count, 6)
    generalized_force = np.sum(generalized[:, :3], axis=0)
    generalized_moment = np.sum(
        np.cross(rows[:, :3], generalized[:, :3])
        + np.cross(rows[:, 3:], generalized[:, 3:]),
        axis=0,
    )
    np.testing.assert_allclose(
        generalized_force,
        packet.resolved_force_w.cpu().numpy(),
        rtol=0.0,
        atol=2.0e-11,
    )
    np.testing.assert_allclose(
        generalized_moment,
        packet.resolved_moment_w.cpu().numpy(),
        rtol=0.0,
        atol=2.0e-11,
    )


def test_real_ptera_resolved_and_source_owned_lev_loads_compose_on_q16() -> None:
    mesh, binder, parent = _committed_parent()
    _, vertex_map = _mesh_and_map()
    branch = parent.fork()
    q = _state(mesh, 1.15)
    binder.bind_next_state(branch, q, _velocity(mesh, 0.85))
    branch.advance_one_step()
    packet = Q16CudaAerodynamicLoadPacket.from_solver(branch.solver)
    lev_load = Q16CudaLEVImpulseStripLoad.from_solver(branch.solver)
    resolved = Q16CudaPteraResolvedLoadTransfer(
        vertex_map,
        chordwise_panel_count=2,
        spanwise_panel_count=3,
        device=config.DEVICE,
    )
    impulse = Q16CudaLEVImpulseTransfer(
        vertex_map,
        leading_edge_point_indices=np.arange(4, dtype=np.int64),
        device=config.DEVICE,
    )
    complete = Q16CudaCompleteAeroLoadTransfer(resolved, impulse)
    result = complete.map(packet, lev_load, q)

    assert packet.point_count == 30
    assert lev_load.particle_count == branch.solver.lev_pf.n > 0
    assert result.resolved.point_reconstruction_max_abs_error <= 5.0e-11
    np.testing.assert_allclose(
        result.generalized_force.numpy(),
        result.resolved.generalized_force.numpy()
        + result.lev_impulse_generalized_force.numpy(),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        packet.unresolved_impulse_force_w,
        lev_load.source_total_force_w,
        rtol=0.0,
        atol=8.0e-13,
    )
    assert float(np.linalg.norm(result.generalized_force.numpy())) > 0.0
    assert branch.solver._prescribed_wake is False
    assert branch.solver._tev_solved is not None


def test_transfer_rejects_packet_geometry_topology_and_rank_drift() -> None:
    mesh, vertex_map, q, packet = _synthetic_packet()
    transfer = Q16CudaPteraResolvedLoadTransfer(
        vertex_map,
        chordwise_panel_count=2,
        spanwise_panel_count=3,
        device=config.DEVICE,
    )
    packet.point_positions_w[0, 0] += 1.0e-3
    with pytest.raises(RuntimeError, match="packet content drift"):
        transfer.map(packet, q)

    with pytest.raises(ValueError, match="vertex count"):
        Q16CudaPteraResolvedLoadTransfer(
            vertex_map,
            chordwise_panel_count=1,
            spanwise_panel_count=4,
            device=config.DEVICE,
        )

    _, _, _, clean_packet = _synthetic_packet()
    collapsed = np.ascontiguousarray(mesh.reference_state[None, :].copy())
    rows = collapsed.reshape(1, mesh.node_count, 6)
    rows[:, :, 3:] = 0.0
    collapsed_q = wp.array(
        np.ascontiguousarray(collapsed), dtype=config.DTYPE, device=config.DEVICE
    )
    with pytest.raises(RuntimeError, match="rank"):
        transfer.map(clean_packet, collapsed_q)


def test_complete_transfer_rejects_impulse_owner_mismatch() -> None:
    mesh, binder, parent = _committed_parent()
    _, vertex_map = _mesh_and_map()
    branch = parent.fork()
    q = _state(mesh, 0.95)
    binder.bind_next_state(branch, q, _velocity(mesh, 0.65))
    branch.advance_one_step()
    packet = Q16CudaAerodynamicLoadPacket.from_solver(branch.solver)
    load = Q16CudaLEVImpulseStripLoad.from_solver(branch.solver)
    hostile = Q16CudaLEVImpulseStripLoad.from_tensors(
        strip_forces_w=load.strip_forces_w
        + torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64, device="cuda:0").expand_as(
            load.strip_forces_w
        ),
        leading_edge_endpoints_w=load.leading_edge_endpoints_w,
        particle_source_strips=load.particle_source_strips,
    )
    complete = Q16CudaCompleteAeroLoadTransfer(
        Q16CudaPteraResolvedLoadTransfer(
            vertex_map,
            chordwise_panel_count=2,
            spanwise_panel_count=3,
            device=config.DEVICE,
        ),
        Q16CudaLEVImpulseTransfer(
            vertex_map,
            leading_edge_point_indices=np.arange(4, dtype=np.int64),
            device=config.DEVICE,
        ),
    )
    with pytest.raises(RuntimeError, match="impulse owner"):
        complete.map(packet, hostile, q)


def test_transfer_rejects_host_and_float32_scientific_inputs() -> None:
    mesh, vertex_map, q, packet = _synthetic_packet()
    transfer = Q16CudaPteraResolvedLoadTransfer(
        vertex_map,
        chordwise_panel_count=2,
        spanwise_panel_count=3,
        device=config.DEVICE,
    )
    host_q = wp.array(q.numpy(), dtype=config.DTYPE, device="cpu")
    with pytest.raises(ValueError, match="CUDA"):
        transfer.map(packet, host_q)
    float32_q = wp.array(
        q.numpy().astype(np.float32), dtype=wp.float32, device=config.DEVICE
    )
    with pytest.raises(TypeError, match="float64"):
        transfer.map(packet, float32_q)
